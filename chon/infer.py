"""In-process decision model: one forward pass per question, letter readout, per-type temperature.

Optional "think when unsure" (config key "think" in chon_config.json):
questions whose one-pass answer is unsure (calibrated confidence below tau for their type; for yes/no,
max(P(yes), P(no)) < tau) get a short reasoning written inside the reply - by the base model with the LoRA
switched off when generator == "base" (needs an unmerged adapter) - and the letter is read again after it.
The reasoning tokens are reported as output_tokens in usage.

    "think": {"tau": {"choice": 0.75, "noul": 0.92, "score": 0.7}, "budget": 512, "generator": "base",
              "mix": true, "temperatures": {"choice": 1.0, "noul": 0.5, "score": 1.0}}
"""

from __future__ import annotations

import math
import os
import time
from typing import Any, Sequence

import torch

from .answers import format_answer
from .modeling import (letter_logits, left_pad, load_causal_lm, load_tokenizer, read_model_config)
import contextlib

from .prompt import INLINE_PREFIX, INLINE_SUFFIX, PromptBuilder, suffix_letter_ids
from .schema import ItemError, options_for, validate_item


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


class DecisionModel:
    def __init__(self, model, tokenizer, cfg: dict | None = None, device: str | torch.device = "cpu",
                 max_batch_tokens: int = 32768, name: str = "chon"):
        cfg = dict(cfg or {})
        self.model = model
        self.tok = tokenizer
        self.cfg = cfg
        self.name = cfg.get("name", name)
        self.device = torch.device(device)
        self.builder = PromptBuilder(tokenizer, max_state_tokens=cfg.get("max_state_tokens", 6000),
                                     chat_template_kwargs=cfg.get("chat_template_kwargs"),
                                     answer_prefix=cfg.get("answer_prefix", ""))
        self.letter_ids = torch.tensor(self.builder.letter_ids, dtype=torch.long, device=self.device)
        self.temperatures = {"choice": 1.0, "noul": 1.0, "score": 1.0, **cfg.get("temperatures", {})}
        self.max_batch_tokens = int(max_batch_tokens)
        # yes/no commit rule: a P(yes) strictly between 0.2 and 0.8 is an abstention under common scoring, so a served
        # P(yes) inside that band moves just outside its nearer edge, e.g. [0.199, 0.801]
        # a last per-type calibration step on the served distribution (after thinking, before the commit
        # rule), p ** (1 / t) renormalised
        self.post_temperatures = dict(cfg.get("post_temperatures") or {})
        nc = cfg.get("noul_commit")
        self.noul_commit = (float(nc[0]), float(nc[1])) if nc else None
        self.think = dict(cfg["think"]) if cfg.get("think") else None
        self.think_tokens: list[int | None] = []
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
        if quantize == "fp8":
            from .quantize import fp8_quantize

            fp8_quantize(model)
        elif quantize:
            raise ValueError(f"unknown quantization {quantize!r}")
        if compile:
            model.forward = torch.compile(model.forward, dynamic=True)
        return cls(model, tok, cfg, device=device, max_batch_tokens=max_batch_tokens)

    # ------------------------------------------------------------------ core
    def encode(self, state: Any, question: dict, order: Sequence[str] | None = None,
               fitted_state: str | None = None) -> list[int]:
        return self.builder.encode(state, question, order=order, fitted_state=fitted_state)

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

    # ------------------------------------------------------------------ think when unsure
    def unsure(self, qtype: str, order: Sequence[str], probs: Sequence[float]) -> bool:
        tau = self.think["tau"]
        if qtype == "noul":
            py = probs[list(order).index("yes")]
            return max(py, 1 - py) < tau["noul"]
        return max(probs) < tau[qtype]

    @torch.inference_mode()
    def _generate(self, prompts: list[list[int]], max_new: int) -> list[list[int]]:
        """Greedy continuations (stop tokens dropped), longest first, batches capped by prompt+new tokens."""
        ctx = generation_context(self.model, self.think["generator"])
        pad = self.tok.pad_token_id
        eos = self.tok.eos_token_id
        stop = set(eos if isinstance(eos, list) else [eos])
        if "<end_of_turn>" in self.tok.get_vocab():
            stop.add(self.tok.convert_tokens_to_ids("<end_of_turn>"))
        budget = int(self.think.get("batch_tokens", 48000))
        order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
        out: list[list[int]] = [[] for _ in prompts]
        i = 0
        while i < len(order):
            n = max(1, min(int(self.think.get("max_batch", 32)), budget // (len(prompts[order[i]]) + max_new)))
            idx = order[i:i + n]
            ids, mask = left_pad([prompts[j] for j in idx], pad)
            with ctx():
                gen = self.model.generate(input_ids=ids.to(self.device), attention_mask=mask.to(self.device),
                                          max_new_tokens=max_new, do_sample=False, use_cache=True,
                                          pad_token_id=pad, eos_token_id=sorted(stop))
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

    def think_logprobs(self, seqs: list[list[int]], n_options: list[int]) -> tuple[list[list[float]], list[int]]:
        """Reason (up to budget tokens) after each served prompt, then read the letters after the suffix."""
        gens = self._generate([s + self._prefix for s in seqs], int(self.think["budget"]))
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
        lps, ntoks = self.think_logprobs([seqs[i] for i in idx], [len(orders[i]) for i in idx])
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
                      calibrated: bool = True, think: bool = True) -> list[dict[str, float]]:
        """requests: (state, question, option order or None). Returns label -> probability."""
        seqs, orders = [], []
        fitted_cache: dict[int, str] = {}
        for state, question, order in requests:
            if order is None:
                order = [lab for lab, _ in options_for(question)]
            key = id(state)
            if key not in fitted_cache:
                fitted_cache[key] = self.builder.fit_state(state)[0]
            seqs.append(self.encode(state, question, order, fitted_state=fitted_cache[key]))
            orders.append(list(order))
        logits = self.raw_logits(seqs, [len(o) for o in orders])
        qtypes = [q["type"] for _, q, _ in requests]
        probs = [softmax(lg, self.temperatures.get(qt, 1.0) if calibrated else 1.0) for lg, qt in zip(logits, qtypes)]
        self.think_tokens = self._apply_think(qtypes, orders, seqs, logits, probs) if (think and calibrated) \
            else [None] * len(seqs)
        if calibrated:
            self._commit(qtypes, orders, probs)
        return [dict(zip(order, p)) for order, p in zip(orders, probs)]

    # ------------------------------------------------------------------ API shape
    def predict(self, state: Any, questions: dict[str, dict]) -> tuple[dict, dict]:
        """TypeSafe /v1/systemone semantics. Returns (answers, usage)."""
        if not isinstance(questions, dict) or not questions:
            raise ItemError("questions must be a non-empty object")
        names = list(questions)
        for n in names:
            validate_item({"id": n, "state": state, "question": questions[n]}, require_expected=False)
        fitted = self.builder.fit_state(state)[0]
        seqs, orders = [], []
        for n in names:
            q = questions[n]
            order = [lab for lab, _ in options_for(q)]
            seqs.append(self.encode(state, q, order, fitted_state=fitted))
            orders.append(order)
        t0 = time.perf_counter()
        logits = self.raw_logits(seqs, [len(o) for o in orders])
        qtypes = [questions[n]["type"] for n in names]
        probs = [softmax(lg, self.temperatures.get(qt, 1.0)) for lg, qt in zip(logits, qtypes)]
        thought = self._apply_think(qtypes, orders, seqs, logits, probs)
        self._commit(qtypes, orders, probs)
        elapsed = time.perf_counter() - t0
        answers = {n: format_answer(questions[n], dict(zip(order, p))) for n, order, p in zip(names, orders, probs)}
        n_thought = sum(1 for t in thought if t is not None)
        out_tokens = sum(t for t in thought if t is not None)
        extra_in = n_thought * (len(self._prefix) + len(self._suffix)) if self.think else 0
        usage = {"input_tokens": sum(len(s) for s in seqs) + extra_in, "output_tokens": out_tokens,
                 "forward_s": round(elapsed, 4)}
        if self.think:
            usage["thought"] = n_thought
        return answers, usage
