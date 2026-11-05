"""Read the option letters through vLLM instead of transformers (CHON_READER=vllm, or "reader": "vllm" in the config).

The prompt is the same token ids our PromptBuilder makes; vLLM generates one token with only the question's option
letters allowed, and returns their log-probabilities. The server runs with --logprobs-mode processed_logprobs, so
those are taken after the allowed-token mask: a distribution over the letters, which is all the decision needs (our
own per-type temperatures, second read, commit rule and label biases apply on top, unchanged). vLLM's prefix cache
shares a state between the questions of one request, and between requests, by itself.

    python -m chon.serve --model ninhnv/chon-1.1-31b --reader vllm --vllm-quant fp8      # the server
    python -m decision_index run --engine chon.dindex:ChonEngine --option model=ninhnv/chon-1.1-31b \
        --option reader=vllm --option vllm_quant=fp8 ...                                  # the Decision Index engine
    CHON_READER=vllm CHON_VLLM_QUANT=fp8 ...                      # the same, set by environment
    CHON_VLLM_READER_URL=http://127.0.0.1:8200 ...                # a reader server you started (several processes share it)

An adapter is merged into its base once (chon.merge, cached under $CHON_MERGED_DIR, default
/workspace/merged or ~/.cache/chon/merged) and the merged checkpoint is served; a model read with the soft cap off
("uncap_logits") is served from a copy of its config without the cap (uncapped). vLLM lives in its own environment, as
for the reasoning (README, "vLLM as the reader"); nothing is imported from it here. Not with "think when unsure" (the
reasoning still needs the HF model) or a pair read.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

SERVED_NAME = "chon-reader"
NEG = -1.0e4   # a letter vLLM did not return (it should return all of them): effectively impossible, never NaN
MISSING = -9999.0   # vLLM reports a masked-out token as -9999.0 (27 Oct review): anything at or below is missing


def _log(msg: str) -> None:
    print(f"[reader/vllm] {msg}", file=sys.stderr, flush=True)


class VLLMReader:
    """Letter log-probabilities from a vLLM OpenAI-compatible server, one request per prompt, in parallel."""

    def __init__(self, url: str, model: str = SERVED_NAME, workers: int = 64, timeout: float = 900.0, post=None):
        self.url, self.model, self.workers, self.timeout = url.rstrip("/"), model, int(workers), float(timeout)
        self._post = post or self._http_post

    def _http_post(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(self.url + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.load(r)

    def one(self, ids, letters: list[int]) -> list[float]:
        # greedy (temperature 0) with every sampling filter off: vLLM fills unset fields from the model folder's
        # generation_config.json (Gemma: top_k 64, top_p 0.95), and on the sampling path its processed log-probs come
        # after top-k/top-p, which zeroes unlikely letters (27 Oct review). On the greedy path they are taken right
        # after the allowed-token mask: the letters' distribution, which our temperatures then scale.
        body = {"model": self.model, "prompt": [int(t) for t in ids], "max_tokens": 1, "temperature": 0.0,
                "top_p": 1.0, "top_k": -1, "min_p": 0.0, "repetition_penalty": 1.0, "presence_penalty": 0.0,
                "frequency_penalty": 0.0, "logprobs": len(letters), "allowed_token_ids": [int(t) for t in letters],
                "return_tokens_as_token_ids": True, "skip_special_tokens": False}
        ch = self._post("/v1/completions", body)["choices"][0]
        top = ((ch.get("logprobs") or {}).get("top_logprobs") or [{}])[0] or {}
        got = {}
        for k, v in top.items():
            if isinstance(k, str) and k.startswith("token_id:"):
                got[int(k.split(":", 1)[1])] = float(v)
        return [got[int(t)] if got.get(int(t), NEG) > MISSING else NEG for t in letters]

    def letter_logprobs(self, seqs, n_options, letter_ids) -> list[list[float]]:
        letters = [int(t) for t in (letter_ids.tolist() if hasattr(letter_ids, "tolist") else letter_ids)]
        jobs = [(s, letters[:n]) for s, n in zip(seqs, n_options)]
        if len(jobs) == 1:
            return [self.one(*jobs[0])]
        with ThreadPoolExecutor(max_workers=min(self.workers, len(jobs))) as ex:
            return list(ex.map(lambda j: self.one(*j), jobs))

    def check(self, ids, letters: list[int]) -> None:
        """The server must give every allowed letter, normalised over the letters (processed log-probs)."""
        lp = self.one(ids, letters)
        mass = sum(math.exp(x) for x in lp)
        if any(x <= MISSING for x in lp) or abs(mass - 1.0) > 0.02:
            raise RuntimeError(f"the vLLM server does not return the letters' processed log-probabilities (mass "
                               f"{mass:.3f}, {sum(x <= NEG for x in lp)} letters missing): start it with "
                               "--logprobs-mode processed_logprobs (vLLM >= 0.10)")


def merged_dir(base: str, adapter: str) -> str:
    """The merged checkpoint of base + adapter, made once (keyed by the adapter's weights) and reused."""
    w = os.path.join(adapter, "adapter_model.safetensors")
    h = hashlib.sha256(f"{base}\n".encode())   # the same adapter merged into another base is another checkpoint
    with open(w, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    root = os.environ.get("CHON_MERGED_DIR") or ("/workspace/merged" if os.path.isdir("/workspace")
                                                 else os.path.expanduser("~/.cache/chon/merged"))
    out = os.path.join(root, h.hexdigest()[:16])
    if os.path.exists(os.path.join(out, "config.json")):
        return out
    os.makedirs(root, exist_ok=True)
    tmp = out + ".tmp"
    _log(f"merging {adapter} into {base} -> {out} (once)")
    subprocess.run([sys.executable, "-m", "chon.merge", "--adapter", adapter, "--base", base, "--out", tmp,
                    "--device-map", os.environ.get("CHON_MERGE_DEVICE", "auto")], check=True)
    os.replace(tmp, out)
    return out


def uncapped(model_dir: str) -> str:
    """model_dir as vLLM must serve a model read without Gemma's final logit soft cap (cfg "uncap_logits": the 1.1-31B):
    a sibling folder of links to its files with config.json rewritten, every final_logit_softcapping set to null, so
    vLLM's letters match chon.modeling.uncap_logits (31 Oct; otherwise vLLM would read them under the cap)."""
    out = model_dir.rstrip("/") + "-uncap"
    if os.path.exists(os.path.join(out, "config.json")):
        return out

    def strip(x):
        if isinstance(x, dict):
            return {k: (None if k == "final_logit_softcapping" else strip(v)) for k, v in x.items()}
        return [strip(v) for v in x] if isinstance(x, list) else x

    tmp = out + ".tmp"
    os.makedirs(tmp, exist_ok=True)
    for name in os.listdir(model_dir):
        if name != "config.json" and not os.path.exists(os.path.join(tmp, name)):
            os.symlink(os.path.realpath(os.path.join(model_dir, name)), os.path.join(tmp, name))
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)
    with open(os.path.join(tmp, "config.json"), "w") as f:
        json.dump(strip(cfg), f, indent=2)
    os.replace(tmp, out)
    return out


def start(model_dir: str, cfg: dict | None = None) -> VLLMReader:
    """A reader on $CHON_VLLM_READER_URL, else on a vLLM server of model_dir started here (chon.vllm_sidecar)."""
    from chon import vllm_sidecar

    cfg = dict(cfg or {})
    url = os.environ.get("CHON_VLLM_READER_URL") or cfg.get("url")
    if url:
        name = vllm_sidecar._alive(url) or SERVED_NAME
        return VLLMReader(url, name)
    quant = os.environ.get("CHON_VLLM_QUANT", cfg.get("quantization") or "")
    side = {"served_name": SERVED_NAME, "gpu_memory_utilization": float(os.environ.get("CHON_VLLM_MEM", cfg.get(
                "gpu_memory_utilization", 0.85))), "reserve_gb": 1, "min_free_gb": 12,
            "max_model_len": int(os.environ.get("CHON_VLLM_MAX_LEN", cfg.get("max_model_len", 34816))),
            # every option code's log-prob is returned: up to 255 for a model read with many options (max_options)
            "extra_args": ["--logprobs-mode", "processed_logprobs",
                           "--max-logprobs", str(max(20, int(cfg.get("max_logprobs", 20)))),
                           "--generation-config", "vllm", *cfg.get("extra_args", [])],   # vLLM's neutral defaults
            "start_timeout_s": cfg.get("start_timeout_s", 2400)}
    if quant and quant != "none":
        side["quantization"] = quant
    url = vllm_sidecar.ensure(model_dir, side)
    if not url:
        raise RuntimeError("no vLLM reader server could start (see the [think/vllm] lines above)")
    return VLLMReader(url, SERVED_NAME)
