"""In-process decision model: one forward pass per question, letter readout, per-type temperature.

Optional "think when unsure" (config key "think" in chon_config.json):
questions whose one-pass answer is unsure (calibrated confidence below tau for their type; for yes/no,
max(P(yes), P(no)) < tau) get a short reasoning written inside the reply - by the base model with the LoRA
switched off when generator == "base" (needs an unmerged adapter) - and the letter is read again after it.
The reasoning tokens are reported as output_tokens in usage.

    "think": {"tau": {"choice": 0.75, "noul": 0.92, "score": 0.7}, "budget": 512, "generator": "base",
              "mix": true, "temperatures": {"choice": 1.0, "noul": 0.5, "score": 1.0}}

Prompt layout (config key "prompt_layout"): "state_first" (default), "header" (every question of the request listed
before the state; chon.prompt.HEADER_LINE) or "hybrid_header" (the header for a request with two or more distinct
questions, state-first for one). Any other value is refused.

Prompt style (config key "prompt_style"): "chon" (default) or "decider" (chon.prompt: "Context:" / "Question:" /
"Options:" with "(A) key: description" lines, yes/no shown as (A) no, (B) yes, every question of the request listed in
a header read, the letter read after "Answer: ("); "uncap_logits": true reads the letters with the model's final logit
soft cap off (chon.modeling.uncap_logits). A model trained with either is read the same way; any other style is refused.

Optional second read of unsure answers (config key "reread", or $CHON_REREAD="below,layout,max_tokens,combine"):
a choice or yes/no question whose top calibrated probability is under "below", in a prompt of at most "max_tokens"
tokens, is read again in a second-read layout ("twice": the request written out twice, "twice_header" for a header
read; or "sandwich") and the two distributions are averaged ("combine": "avg") or the second kept ("replace").
Skipped when the request's deadline leaves no time for it.

    "reread": {"below": 0.9, "layout": "twice", "max_tokens": 2000, "combine": "avg"}

Shared-prefix read (config key "prefix_cache", default true; chon.prefix): the tokens every read of a request shares
(the chat template's start, a header model's question list, the state) run through the model once, and each question
reads only its own tokens against that cached prefix, all of them in one packed pass (at most "prefix_pack_tokens"
suffix tokens per pass, default 8192). The answers equal separate passes (up to floating-point rounding). Used for a
request's first reads, the second reads that start with the same prefix ("twice" / "twice_header") and the group and
final rounds of a choice over 16 options; a request with one question (and no grouped choice), a shared part under
"prefix_min_tokens" (default 128) or a model it does not support (printed once) reads plainly.

    "prefix_cache": true, "prefix_min_tokens": 128, "prefix_pack_tokens": 8192
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Any, Sequence

import torch

from .answers import format_answer
from .modeling import (letter_logits, left_pad, load_causal_lm, load_tokenizer, read_model_config,
                       uncap_logits)
import contextlib

from .prefix import SharedPrefix, Unsupported, check_supported, common_prefix, pack_budget, packed_batches
from .prompt import INLINE_PREFIX, INLINE_SUFFIX, PROMPT_STYLES, REREAD_LAYOUTS, PromptBuilder, suffix_letter_ids
from .schema import MAX_OPTIONS, CapacityError, ItemError, options_for, validate_item

PROMPT_LAYOUTS = ("state_first", "header", "hybrid_header")


class DeadlineExceeded(RuntimeError):
    """The optional reasoning did not fit the request's time budget: the one-pass answer stands. tokens: what was
    generated per question before the cut (reported as output tokens: they were paid for)."""

    def __init__(self, msg: str, tokens: list[int] | None = None):
        super().__init__(msg)
        self.tokens = tokens


def log_softmax(xs: Sequence[float]) -> list[float]:
    m = max(xs)
    z = math.log(sum(math.exp(x - m) for x in xs))
    return [x - m - z for x in xs]


def mean_logprob(lps: Sequence[Sequence[float]]) -> list[float]:
    """Log-linear pool of distributions over the same labels."""
    return log_softmax([sum(col) / len(col) for col in zip(*lps)])


def softmax(xs: Sequence[float], temperature: float = 1.0) -> list[float]:
    t = max(float(temperature), 1e-6)
    m = max(xs)
    e = [math.exp((x - m) / t) for x in xs]
    s = sum(e)
    return [v / s for v in e]


REREAD_DEFAULTS = {"below": 0.9, "layout": "twice", "max_tokens": 2000, "types": ["choice", "noul"], "combine": "avg"}
REREAD_COMBINE = ("avg", "replace")   # average the two reads, or keep the second


def parse_reread(cfg_value: dict | None, env: str | None = None) -> dict | None:
    """The second-read setting: the model config's "reread" (a dict, or a number = its threshold), overridden by
    $CHON_REREAD ("0.9", "0.9,sandwich", "0.9,twice,2000", "0.9,twice,,replace"; "0" or "off" switches it off).
    None = one read."""
    rr = None
    if isinstance(cfg_value, (int, float)) and not isinstance(cfg_value, bool):
        cfg_value = {"below": cfg_value}
    if isinstance(cfg_value, dict) and float(cfg_value.get("below", 0) or 0) > 0:
        rr = {**REREAD_DEFAULTS, **cfg_value}
    if env is not None and env.strip() != "":
        parts = [x.strip() for x in env.split(",")]
        if parts[0].lower() in ("0", "off", "none", "false"):
            return None
        rr = {**(rr or REREAD_DEFAULTS), "below": float(parts[0])}
        if len(parts) > 1 and parts[1]:
            rr["layout"] = parts[1]
        if len(parts) > 2 and parts[2]:
            rr["max_tokens"] = int(parts[2])
        if len(parts) > 3 and parts[3]:
            rr["combine"] = parts[3]
    if rr is None:
        return None
    if rr["layout"] not in REREAD_LAYOUTS:
        raise ValueError(f"unknown reread layout {rr['layout']!r} (one of {REREAD_LAYOUTS})")
    if rr["combine"] not in REREAD_COMBINE:
        raise ValueError(f"unknown reread combine {rr['combine']!r} (one of {REREAD_COMBINE})")
    rr["below"], rr["max_tokens"], rr["types"] = float(rr["below"]), int(rr["max_tokens"]), list(rr["types"])
    return rr


# what a shared-prefix read that a transformers version or model cannot run raises: the request reads plainly instead
_PREFIX_FAILURES = (Unsupported, TypeError, AttributeError, NotImplementedError)


def flag(v) -> bool:
    """A setting that may arrive as a string ("true", "off", "0", ...) from a command line or engine option."""
    return v if isinstance(v, bool) else str(v).strip().lower() in ("1", "true", "yes", "on")


def _state_key(state: Any) -> str:
    """Questions about the same state (by its text) belong to one request."""
    return state if isinstance(state, str) else json.dumps(state, sort_keys=True, default=str)


def generation_context(model, generator: str):
    """Which weights write the reasoning: the base model (LoRA off), the reasoning adapter, or the model as is."""
    if generator == "base" and hasattr(model, "disable_adapter"):
        return model.disable_adapter
    if generator == "reason" and "reason" in getattr(model, "peft_config", {}):
        @contextlib.contextmanager
        def use_reason():
            model.set_adapter("reason")
            try:
                yield
            finally:
                model.set_adapter("default")
        return use_reason
    return contextlib.nullcontext


def resolve_package(path: str) -> str:
    """A Hugging Face id of a release package ("user/name" or "user/name@revision") -> its local snapshot.
    Anything else (a local folder, a plain base-model id) is returned unchanged."""
    if os.path.exists(path) or path.count("/") != 1:
        return path
    from huggingface_hub import snapshot_download

    repo, _, rev = path.partition("@")
    try:
        probe = snapshot_download(repo, revision=rev or None, allow_patterns=["chon_config.json"])
    except Exception:  # noqa: BLE001  (not on the Hub, or offline without a cached copy)
        return path
    if not read_model_config(probe).get("base_model"):
        return path  # a plain model id: transformers loads it as usual
    return snapshot_download(repo, revision=rev or None)


def _subquestion(question: dict, opts: list[tuple[str, str | None]]) -> dict:
    """The same choice restricted to some of its options (criteria kept as given: dict or list)."""
    crit = question["criteria"]
    keep = [lab for lab, _ in opts]
    sub = {k: crit[k] for k in keep} if isinstance(crit, dict) else keep
    return {**question, "criteria": sub}


def resolve_device(device: str | None) -> str:
    """"auto" (or None): an NVIDIA GPU if there is one, else Apple's GPU (mps), else the CPU."""
    if device not in (None, "", "auto"):
        return str(device)
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


class DecisionModel:
    def __init__(self, model, tokenizer, cfg: dict | None = None, device: str | torch.device = "cpu",
                 max_batch_tokens: int = 32768, name: str = "chon"):
        cfg = dict(cfg or {})
        self.model = model
        self.tok = tokenizer
        self.cfg = cfg
        self.name = cfg.get("name", name)
        self.device = torch.device(device)
        # prompt style the model was trained (or is read) in: "chon" (default) or "decider" (chon.prompt); refused
        # otherwise, like an unknown layout
        self.style = cfg.get("prompt_style") or "chon"
        if self.style not in PROMPT_STYLES:
            raise ValueError(f"unknown prompt_style {self.style!r} in the model config (one of {PROMPT_STYLES}); "
                             f"this model needs a newer version of the chon serving code")
        self.builder = PromptBuilder(tokenizer, max_state_tokens=cfg.get("max_state_tokens", 6000),
                                     chat_template_kwargs=cfg.get("chat_template_kwargs"),
                                     answer_prefix=cfg.get("answer_prefix", ""), style=self.style)
        # "uncap_logits": the letters are read with the final logit soft cap off, as the model was trained
        self.uncapped = bool(cfg.get("uncap_logits"))
        if self.uncapped and model is not None:
            uncap_logits(model)
        self.letter_ids = torch.tensor(self.builder.letter_ids, dtype=torch.long, device=self.device)
        # prompt layout the model was trained (or is read) with: "state_first" (default), "header" (the request's
        # questions listed before the state) or "hybrid_header" (the header for a request with two or more distinct
        # questions, state-first for one). Refused otherwise: a model read in the wrong layout answers worse silently.
        self.layout = cfg.get("prompt_layout") or "state_first"
        if self.layout not in PROMPT_LAYOUTS:
            raise ValueError(f"unknown prompt_layout {self.layout!r} in the model config (one of {PROMPT_LAYOUTS}); "
                             f"this model needs a newer version of the chon serving code")
        self._req_layout: str | None = None
        self._req_asked: list[str] | None = None   # the header layout's question list for the current request
        self.temperatures = {"choice": 1.0, "noul": 1.0, "score": 1.0, **cfg.get("temperatures", {})}
        self.max_batch_tokens = int(max_batch_tokens)
        # yes/no commit rule: a P(yes) strictly between 0.2 and 0.8 is an abstention under common scoring, so a served
        # P(yes) inside that band moves just outside its nearer edge, e.g. [0.199, 0.801]
        # a last per-type calibration step on the served distribution (after thinking, before the commit
        # rule), p ** (1 / t) renormalised
        self.post_temperatures = dict(cfg.get("post_temperatures") or {})
        nc = cfg.get("noul_commit")
        self.noul_commit = (float(nc[0]), float(nc[1])) if nc else None
        # second read of an unsure answer: a choice or yes/no question whose top calibrated probability is under
        # reread["below"], in a prompt of at most reread["max_tokens"] tokens, is read again in a second-read layout
        # (prompt.REREAD_LAYOUTS) and the two distributions are averaged (or the second kept: combine "replace").
        # Off unless the config has "reread" or $CHON_REREAD sets it; never for choices over 16 options.
        self.reread = parse_reread(cfg.get("reread"), os.environ.get("CHON_REREAD"))
        if self.reread and self.style == "decider" and self.reread["layout"] != "twice":
            # the decider style has no sandwich layout: refused here rather than when an unsure answer is read again
            raise ValueError(f"prompt_style 'decider' has no {self.reread['layout']!r} second read (use 'twice')")
        self.reread_stats = {"questions": 0, "reread": 0, "extra_tokens": 0, "deadline_skipped": 0}
        self.think = dict(cfg["think"]) if cfg.get("think") else None
        self.think_tokens: list[int | None] = []
        # deadline (predict(..., deadline=)): the one-pass answer is always kept; the reasoning runs only with the
        # tokens the time left allows (seconds per token measured per call, first at the server's warm-up), and a
        # call cut by the clock keeps the one-pass answer
        self._deadline: float | None = None
        self._read_s = 0.15
        self._gen_spt: float | None = None
        self.deadline_stats = {"requests": 0, "skipped": 0, "shortened": 0, "cut": 0}
        # shared-prefix read (cfg "prefix_cache", default on): the request's common prefix runs once per request; its
        # keys/values live in self._shared for that request only. _prefix_off: the model cannot take it (plain from
        # then on, said once)
        self._shared: SharedPrefix | None = None
        self._prefix_off = False
        self.prefix_stats = {"requests": 0, "prefix_tokens": 0, "reads": 0}
        if model is not None and self.prefix_cache:
            try:
                check_supported(model)
            except Unsupported as e:
                self._prefix_off_notice(e)
        if self.think:
            self.think.setdefault("budget", 512)
            self.think.setdefault("generator", "base")
            self.think.setdefault("mix", True)
            self.think.setdefault("tau", {"choice": 0.6, "noul": 0.8, "score": 0.55})
            self.think["temperatures"] = {"choice": 1.0, "noul": 1.0, "score": 1.0,
                                          **(self.think.get("temperatures") or {})}
            self._prefix = tokenizer(self.think.get("prefix", INLINE_PREFIX), add_special_tokens=False)["input_ids"]
            suffix = self.think.get("suffix", INLINE_SUFFIX)
            self._suffix = tokenizer(suffix, add_special_tokens=False)["input_ids"]
            self._suffix_letters = torch.tensor(suffix_letter_ids(tokenizer, suffix), dtype=torch.long,
                                                device=self.device)
            # median-latency guard: at most max_share of the last `window` questions may think, so a set where
            # most questions look unsure cannot push the median response time onto the slow path
            from collections import deque

            self._recent = deque(maxlen=int(self.think.get("window", 1000)))
            if self.think["generator"] == "base" and not hasattr(model, "disable_adapter"):
                print("[think] generator=base needs an unmerged adapter; the loaded model writes the reasoning")

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, path: str, adapter: str | None = None, device: str = "cuda", dtype: str = "bfloat16",
             quantize: str | None = None, attn_implementation: str | None = "sdpa",
             max_batch_tokens: int = 32768, compile: bool = False, think: dict | None | bool = None,
             merge: bool | None = None) -> "DecisionModel":
        """think: a think config (dict), False to switch it off, None to use the model's config.
        merge: merge the adapter into the weights (default: yes, unless thinking needs the base model)."""
        device = resolve_device(device)
        path = resolve_package(path)
        own = read_model_config(path)
        if adapter is None and own.get("base_model") and os.path.exists(os.path.join(path, "adapter_config.json")):
            adapter, path = path, own["base_model"]  # a release package: adapter + config, base model by name
        cfg = {**read_model_config(path), **(read_model_config(adapter) if adapter else {})}
        if think is False:
            cfg.pop("think", None)
        elif think:
            cfg["think"] = think
        if merge is None:
            merge = not (cfg.get("think") and (cfg["think"].get("generator", "base") in ("base", "reason")))
        tok_dir = adapter if adapter and os.path.exists(os.path.join(adapter, "tokenizer_config.json")) else path
        tok = load_tokenizer(tok_dir)
        model = load_causal_lm(path, dtype=dtype, attn_implementation=attn_implementation,
                               device_map=None if device == "cpu" else device)
        if adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter)
            reason = (cfg.get("think") or {}).get("reason_adapter")
            if reason and cfg["think"].get("generator") == "reason":
                if not os.path.isabs(reason):
                    reason = os.path.join(adapter, reason)  # inside a release package
                model.load_adapter(reason, adapter_name="reason")  # writes the reasoning; "default" decides
                model.set_adapter("default")
            if merge:
                model = model.merge_and_unload()
        model.eval()
        model.requires_grad_(False)   # reading only: weights that want gradients would make autograd keep activations
        if quantize == "fp8":
            from .quantize import fp8_quantize

            fp8_quantize(model)
        elif quantize:
            raise ValueError(f"unknown quantization {quantize!r}")
        if compile:
            model.forward = torch.compile(model.forward, dynamic=True)
        dm = cls(model, tok, cfg, device=device, max_batch_tokens=max_batch_tokens)
        if dm.think and dm.think.get("engine") in ("vllm", "hf-static") and not str(device).startswith("cuda"):
            dm.think["engine"] = "hf"  # CPU: plain generate (vLLM and the compiled static cache are GPU paths)
        elif dm.think and dm.think.get("engine") == "vllm" and not dm.think.get("vllm_url"):
            from .vllm_sidecar import ensure

            url = ensure(path, dm.think.get("vllm"))  # after our model: vLLM gets what memory is left
            if url:
                dm.think["vllm_url"] = url
            else:
                dm.think["engine"] = dm.think.get("fallback_engine", "hf-static")
                print(f"[think] vLLM unavailable: reasoning with engine {dm.think['engine']!r}")
        return dm

    # ------------------------------------------------------------------ core
    def layout_for(self, n_questions: int) -> str:
        """The layout for one of n_questions (distinct) questions asked about the same state."""
        if self.layout == "hybrid_header":
            return "header" if n_questions > 1 else "state_first"
        return self.layout

    def served_reads(self, pairs: Sequence[tuple[Any, dict]]) -> list[tuple[str | None, list[str] | None]]:
        """(layout, asked) per (state, question), as distributions() reads them when they arrive together: every
        question about the same state listed in the header (the style's list: chon, the distinct instructions in order;
        decider, every question's instructions, duplicates kept). For offline reads that
        call encode() directly, so they match serving."""
        if self.layout not in ("header", "hybrid_header"):
            return [(None, None)] * len(pairs)
        qs_of: dict = {}
        for st, q in pairs:
            qs_of.setdefault(_state_key(st), []).append(q)
        asked_of = {k: self.builder.asked_list(v) for k, v in qs_of.items()}   # (the style's list)
        return [(self.layout_for(len(asked_of[_state_key(st)])), asked_of[_state_key(st)]) for st, _ in pairs]

    def encode(self, state: Any, question: dict, order: Sequence[str] | None = None,
               fitted_state: str | None = None, layout: str | None = None,
               asked: Sequence[str] | None = None) -> list[int]:
        """layout: None = the current request's layout (inside predict), else the model's layout for one question.
        asked: the header's question list (default: the current request's, else this question alone)."""
        layout = layout or self._req_layout or self.layout_for(1)
        if layout == "twice" and (self.layout == "header" or (   # a header read's second read: its layout, twice
                self.layout == "hybrid_header" and len(asked or self._req_asked or []) > 1)):
            layout = "twice_header"
        if layout in ("header", "twice_header"):
            return self.builder.encode(state, question, order=order, fitted_state=fitted_state, layout=layout,
                                       asked=asked or self._req_asked or self.builder.asked_list([question]))
        return self.builder.encode(state, question, order=order, fitted_state=fitted_state, layout=layout)

    @torch.inference_mode()
    def raw_logits(self, seqs: Sequence[Sequence[int]], n_options: Sequence[int],
                   letter_ids: torch.Tensor | None = None) -> list[list[float]]:
        """Letter logits for each sequence (first n_options letters), batched by padded token budget."""
        letter_ids = self.letter_ids if letter_ids is None else letter_ids
        order = sorted(range(len(seqs)), key=lambda i: -len(seqs[i]))
        out: list[list[float] | None] = [None] * len(seqs)
        pad = self.tok.pad_token_id
        i = 0
        while i < len(order):
            longest = len(seqs[order[i]])
            n = max(1, self.max_batch_tokens // max(longest, 1))
            idx = order[i:i + n]
            ids, mask = left_pad([seqs[j] for j in idx], pad)
            logits = letter_logits(self.model, ids.to(self.device), mask.to(self.device), letter_ids)
            logits = logits.cpu().tolist()
            for row, j in zip(logits, idx):
                out[j] = row[: n_options[j]]
            i += n
        return out  # type: ignore[return-value]

    # ------------------------------------------------------------------ shared-prefix read
    @property
    def prefix_cache(self) -> bool:
        """Whether the shared-prefix read is on (cfg "prefix_cache", default true) and the model can take it."""
        return flag(self.cfg.get("prefix_cache", True)) and not self._prefix_off

    def _prefix_off_notice(self, err) -> None:
        self._prefix_off = True
        self._shared = None
        print(f"[prefix] shared-prefix read off for this model ({err}): every question reads its whole prompt",
              flush=True)

    def _prepare_prefix(self, seqs: Sequence[Sequence[int]], force: bool = False) -> int:
        """Run the common prefix of seqs once (cfg "prefix_cache", or force). Returns its length (0 = plain path)."""
        self._shared = None
        if not (force or flag(self.cfg.get("prefix_cache", True))) or self._prefix_off or len(seqs) < 2 \
                or self.model is None:
            return 0
        p = min(common_prefix(seqs), min(len(s) for s in seqs) - 1)  # every question keeps its reply position
        if p < int(self.cfg.get("prefix_min_tokens", 128)):
            return 0
        try:
            self._shared = SharedPrefix(self.model, seqs[0][:p], self.device)
        except _PREFIX_FAILURES as e:
            self._prefix_off_notice(e)
            return 0
        self.prefix_stats["requests"] += 1
        self.prefix_stats["prefix_tokens"] += p
        return p

    def _logits(self, seqs: Sequence[Sequence[int]], n_options: Sequence[int]) -> list[list[float]]:
        """Letter logits: from the shared prefix when one is prepared and starts every sequence, else raw_logits.
        The suffixes go in packed passes of at most cfg "prefix_pack_tokens" tokens (chon.prefix.pack_budget)."""
        sp = self._shared
        if sp is None or not all(len(s) > sp.length and s[:sp.length] == sp.tokens for s in seqs):
            return self.raw_logits(seqs, n_options)
        sufs = [s[sp.length:] for s in seqs]
        budget = pack_budget(int(self.cfg.get("prefix_pack_tokens", 8192)), sp.kept.get("full_attention", 0))
        out: list[list[float] | None] = [None] * len(seqs)
        try:
            for idx in packed_batches([len(x) for x in sufs], budget):
                rows = sp.letter_logits([sufs[i] for i in idx], self.letter_ids).cpu().tolist()
                for i, row in zip(idx, rows):
                    out[i] = row[: n_options[i]]
        except _PREFIX_FAILURES as e:
            self._prefix_off_notice(e)
            return self.raw_logits(seqs, n_options)
        self.prefix_stats["reads"] += len(seqs)
        return out  # type: ignore[return-value]

    def _grouped_logits(self, seqs, n_options, keys) -> list[list[float]]:
        """Letter logits with the reads grouped by key (their state): each group of two or more through its shared
        prefix, the rest (and any group whose prefix is too short or unsupported) in one plain batch."""
        groups: dict[Any, list[int]] = {}
        for i, k in enumerate(keys):
            groups.setdefault(k, []).append(i)
        out: list[list[float] | None] = [None] * len(seqs)
        plain: list[int] = []
        for idx in groups.values():
            if len(idx) < 2 or not self._prepare_prefix([seqs[i] for i in idx], force=True):
                plain.extend(idx)
                continue
            try:
                for i, row in zip(idx, self._logits([seqs[i] for i in idx], [n_options[i] for i in idx])):
                    out[i] = row
            finally:
                self._shared = None
        if plain:
            for i, row in zip(plain, self.raw_logits([seqs[i] for i in plain], [n_options[i] for i in plain])):
                out[i] = row
        return out  # type: ignore[return-value]

    # ------------------------------------------------------------------ second read when unsure
    def _apply_reread(self, qtypes, orders, seqs, probs, encode_at, logits_fn) -> tuple[int, int]:
        """Average a second read into the probabilities of unsure questions, or put it in place of the first
        (reread["combine"] = "replace"), in place (calibrated probabilities). Skipped when the request's deadline
        leaves no time for another read. encode_at(i, layout) -> the second-read ids of question i.
        Returns (questions re-read, their prompt tokens)."""
        rr = self.reread
        self.reread_stats["questions"] += len(seqs)
        if not rr:
            return 0, 0
        if self._deadline is not None and time.monotonic() + 2 * self._read_s + 0.5 > self._deadline:
            self.reread_stats["deadline_skipped"] += 1
            return 0, 0   # no time for a second read: the first stands
        idx = [i for i in range(len(seqs)) if qtypes[i] in rr["types"] and max(probs[i]) < rr["below"]
               and len(seqs[i]) <= rr["max_tokens"]]
        if not idx:
            return 0, 0
        seqs2 = [encode_at(i, rr["layout"]) for i in idx]
        for i, row in zip(idx, logits_fn(seqs2, [len(orders[i]) for i in idx])):
            p2 = softmax(row, self.temperatures.get(qtypes[i], 1.0))
            probs[i] = list(p2) if rr.get("combine") == "replace" else [(a + b) / 2 for a, b in zip(probs[i], p2)]
        extra = sum(len(x) for x in seqs2)
        self.reread_stats["reread"] += len(idx)
        self.reread_stats["extra_tokens"] += extra
        return len(idx), extra

    # ------------------------------------------------------------------ think when unsure
    def unsure(self, qtype: str, order: Sequence[str], probs: Sequence[float]) -> bool:
        tau = self.think["tau"]
        if qtype == "noul":
            py = probs[list(order).index("yes")]
            return max(py, 1 - py) < tau["noul"]
        return max(probs) < tau[qtype]

    def _stop_ids(self) -> set[int]:
        eos = self.tok.eos_token_id
        stop = set(eos if isinstance(eos, list) else [eos])
        if "<end_of_turn>" in self.tok.get_vocab():
            stop.add(self.tok.convert_tokens_to_ids("<end_of_turn>"))
        return stop

    def _generate_vllm(self, prompts: list[list[int]], max_new: int, max_time: float | None = None) -> list[list[int]]:
        """The reasoning written by a vLLM server running the plain base model (think "engine": "vllm"; only
        for generator "base", whose reasoning comes from the base weights anyway). Same greedy decoding from the
        same token ids; the letters are still read by this model. Server: think["vllm_url"] or $CHON_VLLM_URL
        (default http://127.0.0.1:8100). Requests go out in parallel, so the server batches them."""
        import json as _json
        import urllib.request
        from concurrent.futures import ThreadPoolExecutor

        if self.think.get("generator", "base") != "base":
            raise ValueError('think engine "vllm" writes with the base model: needs generator "base"')
        url = (self.think.get("vllm_url") or os.environ.get("CHON_VLLM_URL") or "http://127.0.0.1:8100").rstrip("/")
        if not getattr(self, "_vllm_model", None):
            with urllib.request.urlopen(url + "/v1/models", timeout=30) as r:
                self._vllm_model = _json.load(r)["data"][0]["id"]
        stop = self._stop_ids()

        def one(ids: list[int]) -> list[int]:
            body = {"model": self._vllm_model, "prompt": ids, "max_tokens": max_new, "temperature": 0.0,
                    "stop_token_ids": sorted(stop), "skip_special_tokens": False, "logprobs": 1,
                    "return_tokens_as_token_ids": True}
            req = urllib.request.Request(url + "/v1/completions", data=_json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=max_time if max_time else 900) as r:
                ch = _json.load(r)["choices"][0]
            toks = (ch.get("logprobs") or {}).get("tokens") or []
            out = [int(t.split(":", 1)[1]) for t in toks if isinstance(t, str) and t.startswith("token_id:")]
            if not out and ch.get("text"):  # an older server: re-tokenise the text
                out = self.tok(ch["text"], add_special_tokens=False)["input_ids"]
            return [t for t in out if t not in stop]

        if len(prompts) == 1:
            return [one(prompts[0])]
        with ThreadPoolExecutor(max_workers=min(int(self.think.get("max_batch", 32)), len(prompts))) as ex:
            return list(ex.map(one, prompts))

    @torch.inference_mode()
    def _generate(self, prompts: list[list[int]], max_new: int, max_time: float | None = None) -> list[list[int]]:
        """Greedy continuations (stop tokens dropped), longest first, batches capped by prompt+new tokens.
        think "engine": "hf" (default), "hf-static" (static KV cache, which transformers compiles), "vllm"."""
        engine = self.think.get("engine", "hf")
        if engine == "vllm":
            return self._generate_vllm(prompts, max_new, max_time)
        extra = {"cache_implementation": "static"} if engine == "hf-static" else {}
        if max_time:
            extra["max_time"] = float(max_time)   # transformers stops generating at this many seconds
        ctx = generation_context(self.model, self.think["generator"])
        pad = self.tok.pad_token_id
        stop = self._stop_ids()
        budget = int(self.think.get("batch_tokens", 48000))
        order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
        out: list[list[int]] = [[] for _ in prompts]
        i = 0
        cap = int(self.think.get("static_max_batch", 1)) if engine == "hf-static" else int(self.think.get("max_batch", 32))
        while i < len(order):  # a static cache is reserved at full length per sequence: one at a time by default
            n = max(1, min(cap, budget // (len(prompts[order[i]]) + max_new)))
            idx = order[i:i + n]
            ids, mask = left_pad([prompts[j] for j in idx], pad)
            with ctx():
                gen = self.model.generate(input_ids=ids.to(self.device), attention_mask=mask.to(self.device),
                                          max_new_tokens=max_new, do_sample=False, use_cache=True,
                                          pad_token_id=pad, eos_token_id=sorted(stop), **extra)
            for j, row in zip(idx, gen[:, ids.shape[1]:].tolist()):
                toks = []
                for t in row:
                    if t in stop:
                        break
                    if t != pad:
                        toks.append(t)
                out[j] = toks
            i += n
        return out

    def think_logprobs(self, seqs: list[list[int]], n_options: list[int], budget: int | None = None,
                       max_time: float | None = None) -> tuple[list[list[float]], list[int]]:
        """Reason (up to budget tokens) after each served prompt, then read the letters after the suffix.
        max_time: seconds the reasoning may take; past it the call raises DeadlineExceeded."""
        budget = int(self.think["budget"]) if budget is None else int(budget)
        t0 = time.monotonic()
        try:
            gens = self._generate([s + self._prefix for s in seqs], budget, max_time)
        except OSError as e:   # the vLLM request timed out
            raise DeadlineExceeded(f"reasoning did not finish in {max_time} s") from e
        took = time.monotonic() - t0
        longest = max((len(g) for g in gens), default=0)
        if longest >= 8:   # seconds per generated token for this engine, a moving average
            spt = took / longest
            self._gen_spt = spt if self._gen_spt is None else 0.7 * self._gen_spt + 0.3 * spt
        if max_time and took >= 0.97 * max_time and longest < budget:
            raise DeadlineExceeded(f"reasoning cut at {took:.1f} s", [len(g) for g in gens])
        reads = [s + self._prefix + g + self._suffix for s, g in zip(seqs, gens)]
        logits = self.raw_logits(reads, n_options, letter_ids=self._suffix_letters)
        return [log_softmax(lg) for lg in logits], [len(g) for g in gens]

    def _apply_think(self, qtypes: list[str], orders: list[list[str]], seqs: list[list[int]],
                     logits: list[list[float]], probs: list[list[float]]) -> list[int | None]:
        """Replace the probabilities of unsure questions in place. Returns, per question, the number of
        reasoning tokens generated (None = answered in one pass)."""
        gen_tokens: list[int | None] = [None] * len(seqs)
        if not self.think:
            return gen_tokens
        idx = [i for i in range(len(seqs)) if self.unsure(qtypes[i], orders[i], probs[i])]
        max_share = self.think.get("max_share")
        if max_share is not None:  # questions in arrival order, one request at a time
            keep = []
            for i in range(len(seqs)):
                want = i in idx
                share = sum(self._recent) / len(self._recent) if self._recent else 0.0
                ok = want and (len(self._recent) < int(self.think.get("guard_warmup", 100)) or share < max_share)
                self._recent.append(1 if ok else 0)
                if ok:
                    keep.append(i)
            idx = keep
        if not idx:
            return gen_tokens
        budget, max_time = int(self.think["budget"]), None
        if self._deadline is not None:   # only the tokens the time left allows; none if too few to help
            left = self._deadline - time.monotonic() - 2 * self._read_s - float(self.think.get("deadline_margin_s", 0.5))
            spt = self._gen_spt or float(self.think.get("seconds_per_token", 0.05))
            allowed = int(left / spt) if left > 0 else 0
            if allowed < int(self.think.get("min_think_tokens", 48)):
                self.deadline_stats["skipped"] += len(idx)
                return gen_tokens
            if allowed < budget:
                self.deadline_stats["shortened"] += len(idx)
                budget = allowed
            max_time = left
        try:
            lps, ntoks = self.think_logprobs([seqs[i] for i in idx], [len(orders[i]) for i in idx], budget, max_time)
        except DeadlineExceeded as e:
            self.deadline_stats["cut"] += len(idx)
            for i, n in zip(idx, e.tokens or []):
                gen_tokens[i] = n
            return gen_tokens
        for i, lp, n in zip(idx, lps, ntoks):
            if self.think["mix"]:
                lp = mean_logprob([log_softmax(logits[i]), lp])
            probs[i] = softmax(lp, self.think["temperatures"].get(qtypes[i], 1.0))
            gen_tokens[i] = n
        return gen_tokens

    def _commit(self, qtypes, orders, probs) -> None:
        for i, qt in enumerate(qtypes):  # post-calibration first, so the commit rule sees the final P(yes)
            t = float(self.post_temperatures.get(qt, 1.0))
            if t != 1.0:
                v = [max(float(x), 1e-12) ** (1.0 / t) for x in probs[i]]
                z = sum(v)
                probs[i] = [x / z for x in v]
        if not self.noul_commit:
            return
        lo, hi = self.noul_commit
        for i, (qt, order) in enumerate(zip(qtypes, orders)):
            if qt != "noul" or "yes" not in order:
                continue
            y = order.index("yes")
            py = float(probs[i][y])
            if 0.2 < py < 0.8:
                py = hi if py >= 0.5 else lo
                probs[i] = [py if j == y else 1.0 - py for j in range(len(order))]

    def distributions(self, requests: Sequence[tuple[Any, dict, Sequence[str] | None]],
                      calibrated: bool = True, think: bool = True, share_prefix: bool = False,
                      group_states: bool = True) -> list[dict[str, float]]:
        """requests: (state, question, option order or None). Returns label -> probability.
        group_states=False: every request is its own request (one question); by default the questions about one
        state are read as one request (a header model lists them all). The second read applies to calibrated reads
        only (temperatures are fitted on single reads).
        share_prefix: the first reads about one state go through the shared-prefix read (the state once, each
        question masked to it), which equals separate passes; models it does not support read plainly."""
        seqs, orders, fitted, asked = [], [], [], []
        fitted_cache: dict[int, str] = {}
        asked_of: dict = {}
        if self.layout in ("header", "hybrid_header"):  # every question asked about each state, in order, once each
            qs_of: dict = {}
            for r, (state, question, _) in enumerate(requests):
                qs_of.setdefault(_state_key(state) if group_states else r, []).append(question)
            asked_of = {k: self.builder.asked_list(v) for k, v in qs_of.items()}   # (the style's list)
        for r, (state, question, order) in enumerate(requests):
            if order is None:
                order = self.builder.natural_order(question)
            key = id(state)
            if key not in fitted_cache:
                fitted_cache[key] = self.builder.fit_state(state)[0]
            asked.append(asked_of.get(_state_key(state) if group_states else r))
            seqs.append(self.encode(state, question, order, fitted_state=fitted_cache[key],
                                    layout=self.layout_for(len(asked[-1] or [None])), asked=asked[-1]))
            orders.append(list(order))
            fitted.append(fitted_cache[key])
        n_opts = [len(o) for o in orders]
        logits = self._grouped_logits(seqs, n_opts, fitted) if share_prefix else self.raw_logits(seqs, n_opts)
        qtypes = [q["type"] for _, q, _ in requests]
        probs = [softmax(lg, self.temperatures.get(qt, 1.0) if calibrated else 1.0) for lg, qt in zip(logits, qtypes)]
        if calibrated:
            self._apply_reread(qtypes, orders, seqs, probs,
                               lambda i, lay: self.encode(requests[i][0], requests[i][1], orders[i],
                                                          fitted_state=fitted[i], layout=lay, asked=asked[i]),
                               self.raw_logits)
        self.think_tokens = self._apply_think(qtypes, orders, seqs, logits, probs) if (think and calibrated) \
            else [None] * len(seqs)
        if calibrated:
            self._commit(qtypes, orders, probs)
        return [dict(zip(order, p)) for order, p in zip(orders, probs)]

    # ------------------------------------------------------------------ API shape
    def predict(self, state: Any, questions: dict[str, dict], deadline: float | None = None) -> tuple[dict, dict]:
        """TypeSafe /v1/systemone semantics. Returns (answers, usage).
        deadline: a time.monotonic() by which to answer (chon.serve sets it from the request's arrival): the
        one-pass answers always stand; a second read or the reasoning of unsure answers runs only if it fits before it.
        A choice with more than MAX_OPTIONS options (up to MAX_OPTIONS ** 2) is read in groups, then a final round
        between the group winners (_predict_many). cfg "truncate": false refuses a state over max_state_tokens with
        a CapacityError instead of cutting it. A header model lists the request's questions in every read of this
        request, the groups and final rounds included: in the chon style the distinct instructions in request order,
        in the decider style every question's instructions in request order (PromptBuilder.asked_list)."""
        if not isinstance(questions, dict) or not questions:
            raise ItemError("questions must be a non-empty object")
        names = list(questions)
        many: dict[str, dict] = {}
        for n in names:
            q = questions[n]
            big = (isinstance(q, dict) and q.get("type") == "choice" and isinstance(q.get("criteria"), (dict, list))
                   and len(q["criteria"]) > MAX_OPTIONS and self.cfg.get("many_options", True))
            validate_item({"id": n, "state": state, "question": q}, require_expected=False, allow_empty_state=True,
                          max_options=MAX_OPTIONS ** 2 if big else None)
            if big:
                many[n] = q
        if state is None:
            state = ""
        fitted, truncated = self.builder.fit_state(state)
        if truncated and not self.cfg.get("truncate", True):
            raise CapacityError(f"the state is longer than {self.builder.max_state_tokens} tokens: maximum context "
                                f"length exceeded (this model does not truncate)")
        normal = [n for n in names if n not in many]
        asked_all = self.builder.asked_list([questions[n] for n in names])   # (the style's list)
        answers: dict[str, dict] = {}
        usage = {"input_tokens": 0, "output_tokens": 0, "forward_s": 0.0}
        t0 = time.perf_counter()
        # hybrid_header counts the questions listed (chon style: two names with the same instructions are one question
        # listed; decider style: every name is listed): every encode of this request (first reads, second reads, groups, finals) uses this layout and list
        self._req_layout = self.layout_for(len(asked_all) if self.layout == "hybrid_header" else len(names))
        self._req_asked = asked_all if self._req_layout == "header" else None
        self._deadline = deadline
        if deadline is not None:
            self.deadline_stats["requests"] += 1
        try:
            seqs, orders = [], []
            for n in normal:
                q = questions[n]
                order = self.builder.natural_order(q)
                seqs.append(self.encode(state, q, order, fitted_state=fitted))
                orders.append(order)
            plan = self._plan_many(state, many, fitted) if many else None
            # the shared-prefix read: the part every first read (and first-round group read) of this request shares
            shared = self._prepare_prefix(seqs + (plan[1] if plan else []))
            if shared:
                usage["prefix_tokens"] = shared
            if normal:
                t_read = time.monotonic()
                logits = self._logits(seqs, [len(o) for o in orders])
                self._read_s = 0.8 * self._read_s + 0.2 * (time.monotonic() - t_read)
                qtypes = [questions[n]["type"] for n in normal]
                probs = [softmax(lg, self.temperatures.get(qt, 1.0)) for lg, qt in zip(logits, qtypes)]
                n_rr, rr_tokens = self._apply_reread(
                    qtypes, orders, seqs, probs,
                    lambda i, lay: self.encode(state, questions[normal[i]], orders[i], fitted_state=fitted, layout=lay),
                    self._logits)
                if n_rr:
                    usage["input_tokens"] += rr_tokens
                    usage["reread"] = n_rr
                thought = self._apply_think(qtypes, orders, seqs, logits, probs)
                self._commit(qtypes, orders, probs)
                for n, order, p in zip(normal, orders, probs):
                    answers[n] = format_answer(questions[n], dict(zip(order, p)))
                n_thought = sum(1 for t in thought if t is not None)
                extra_in = n_thought * (len(self._prefix) + len(self._suffix)) if self.think else 0
                usage["input_tokens"] += sum(len(s) for s in seqs) + extra_in
                usage["output_tokens"] += sum(t for t in thought if t is not None)
                if self.think:
                    usage["thought"] = n_thought
            else:
                self.think_tokens = []
            if plan:
                more, n_in = self._finish_many(state, plan, fitted)
                answers.update(more)
                usage["input_tokens"] += n_in
        finally:   # the prefix keys/values, layout, question list and deadline belong to this request only
            self._shared = None
            self._req_layout = None
            self._req_asked = None
            self._deadline = None
        usage["forward_s"] = round(time.perf_counter() - t0, 4)
        return {n: answers[n] for n in names}, usage

    def _predict_many(self, state: Any, qs: dict[str, dict], fitted: str) -> tuple[dict, int]:
        return self._finish_many(state, self._plan_many(state, qs, fitted), fitted)

    def _plan_many(self, state: Any, qs: dict[str, dict], fitted: str):
        """Groups of at most MAX_OPTIONS options per big choice and the first-round sequences."""
        plan, seqs, owner = [], [], []
        for n, q in qs.items():
            opts = options_for(q)
            k = math.ceil(len(opts) / MAX_OPTIONS)
            size = math.ceil(len(opts) / k)
            groups = [opts[i:i + size] for i in range(0, len(opts), size)]
            plan.append((n, q, groups))
            for gi, grp in enumerate(groups):
                seqs.append(self.encode(state, _subquestion(q, grp), [lab for lab, _ in grp], fitted_state=fitted))
                owner.append((n, gi, len(grp)))
        return plan, seqs, owner

    def _finish_many(self, state: Any, planned, fitted: str) -> tuple[dict, int]:
        """Choices with more than MAX_OPTIONS options: each group of at most MAX_OPTIONS options is read as its own
        question, then one final question between the group winners. P(option) = P(its group wins the final) x
        P(option | its group), so every option gets a probability and they sum to 1. No thinking on these. Both
        rounds read through the request's shared prefix when one is prepared."""
        t = self.temperatures.get("choice", 1.0)
        plan, seqs, owner = planned
        lg1 = self._logits(seqs, [m for _, _, m in owner])
        p1 = {(n, gi): softmax(lg, t) for (n, gi, _), lg in zip(owner, lg1)}
        seqs2, finals = [], []
        for n, q, groups in plan:
            winners = [grp[max(range(len(grp)), key=p1[(n, gi)].__getitem__)] for gi, grp in enumerate(groups)]
            finals.append(len(winners))
            seqs2.append(self.encode(state, _subquestion(q, winners), [lab for lab, _ in winners], fitted_state=fitted))
        lg2 = self._logits(seqs2, finals)
        out = {}
        for (n, q, groups), lg in zip(plan, lg2):
            pg = softmax(lg, t)
            probs = {}
            for gi, grp in enumerate(groups):
                for (lab, _), pi in zip(grp, p1[(n, gi)]):
                    probs[lab] = pg[gi] * pi
            labels = [lab for lab, _ in options_for(q)]
            plist = [probs[lab] for lab in labels]
            self._commit(["choice"], [labels], [plist])
            out[n] = format_answer(q, dict(zip(labels, plist)))
        return out, sum(len(s) for s in seqs) + sum(len(s) for s in seqs2)
