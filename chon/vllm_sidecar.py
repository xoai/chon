"""Start a vLLM server of the plain base model to write the "think when unsure" reasoning (think engine "vllm").

The reasoning of a generator-"base" release is written by the base weights with the adapter switched off, so a
stock vLLM server of the base model can write it from the same token ids, several times faster than Hugging Face
generate(); the answer is still read by our own model. DecisionModel.load calls ensure() after loading our model,
so vLLM gets only the memory that is left:

    "think": {..., "engine": "vllm", "fallback_engine": "hf-static",
              "vllm": {"quantization": "fp8", "gpu_memory_utilization": 0.4, "max_model_len": 9216}}

Order of preference: $CHON_VLLM_URL (a server you started yourself), else a vLLM binary ($CHON_VLLM_BIN, `vllm` on
PATH, or /opt/vllm-env/bin/vllm) started here on a free local port, else the fallback engine ("hf-static": static
KV cache, which transformers compiles; about 2x faster than plain generate, no extra memory for a second copy).
vLLM lives in its own environment (its own torch); nothing is imported from it here.
"""
from __future__ import annotations

import atexit
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request

GIB = 1024 ** 3
DEFAULT_BINS = ("/opt/vllm-env/bin/vllm", "/workspace/vllm-env/bin/vllm")
_procs: list[subprocess.Popen] = []


def _log(msg: str) -> None:
    print(f"[think/vllm] {msg}", file=sys.stderr, flush=True)


def _alive(url: str, timeout: float = 5.0) -> str | None:
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/v1/models", timeout=timeout) as r:
            return json.load(r)["data"][0]["id"]
    except Exception:  # noqa: BLE001
        return None


def find_bin(cfg: dict) -> str | None:
    for b in (cfg.get("bin"), os.environ.get("CHON_VLLM_BIN"), shutil.which("vllm"), *DEFAULT_BINS):
        if b and os.path.exists(b) and os.access(b, os.X_OK):
            return b
    return None


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _stop_all() -> None:
    for p in _procs:
        if p.poll() is None:
            try:
                os.killpg(p.pid, 15)
            except Exception:  # noqa: BLE001
                p.terminate()
    for p in _procs:
        try:
            p.wait(timeout=20)
        except Exception:  # noqa: BLE001
            try:
                os.killpg(p.pid, 9)
            except Exception:  # noqa: BLE001
                pass


def gpu_budget(cfg: dict) -> float | None:
    """The fraction of the GPU vLLM may take, from what is free now (our model is loaded already), or None when
    too little is left (the fallback engine then writes the reasoning)."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free, total = torch.cuda.mem_get_info()
    except Exception:  # noqa: BLE001
        return None
    need = float(cfg.get("min_free_gb", 18 if cfg.get("quantization") == "fp8" else 30)) * GIB
    reserve = float(cfg.get("reserve_gb", 6)) * GIB  # for our own forward passes on long inputs
    if free - reserve < need:
        _log(f"only {free / GIB:.1f} GiB free on the GPU (need {need / GIB:.0f} + {reserve / GIB:.0f} reserved)")
        return None
    return round(min(float(cfg.get("gpu_memory_utilization", 0.4)), (free - reserve) / total), 3)


def local_snapshot(model: str) -> str:
    """The local copy of a Hub model when it is cached (vLLM then never contacts the Hub: no token needed for a
    gated model, works on offline evaluator pods); the id unchanged otherwise."""
    if os.path.isdir(model):
        return model
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(model, local_files_only=True)
    except Exception:  # noqa: BLE001
        return model


def _tail(path: str, n: int = 40) -> None:
    try:
        lines = open(path, encoding="utf-8", errors="replace").read().splitlines()[-n:]
    except OSError:
        return
    for line in lines:
        print(f"[think/vllm log] {line}", file=sys.stderr, flush=True)


def ensure(base_model: str, cfg: dict | None = None) -> str | None:
    """The URL of a vLLM server of base_model for the reasoning, starting one if needed; None if none can run."""
    cfg = dict(cfg or {})
    url = cfg.get("url") or os.environ.get("CHON_VLLM_URL")
    if url:
        if _alive(url):
            return url
        _log(f"{url} does not answer")
        return None
    binary = find_bin(cfg)
    if not binary:
        _log("no vLLM binary found (set CHON_VLLM_BIN or install vLLM in /opt/vllm-env)")
        return None
    util = gpu_budget(cfg)
    if util is None:
        return None
    model_path = local_snapshot(base_model)
    port = int(cfg.get("port") or _free_port())
    url = f"http://127.0.0.1:{port}"
    base_cmd = [binary, "serve", model_path, "--host", "127.0.0.1", "--port", str(port), "--dtype", "bfloat16",
                "--max-model-len", str(int(cfg.get("max_model_len", 9216))), "--gpu-memory-utilization", str(util),
                "--served-model-name", "chon-reasoner", "--enable-prefix-caching"]
    if cfg.get("quantization"):
        base_cmd += ["--quantization", str(cfg["quantization"])]
    base_cmd += [str(x) for x in cfg.get("extra_args", [])]
    # the reasoning is text only: no image/video budget to profile at start-up (minutes on a slow machine)
    text_only = ["--limit-mm-per-prompt", json.dumps(cfg.get("mm_limits", {"image": 0, "video": 0}))]
    env = dict(os.environ)
    env["PATH"] = os.path.dirname(binary) + os.pathsep + env.get("PATH", "")  # its own ninja etc.
    if model_path != base_model:
        env["HF_HUB_OFFLINE"] = "1"  # everything it needs is local
    elif not env.get("HF_TOKEN"):
        try:
            from huggingface_hub import get_token

            tok = get_token()
            if tok:
                env["HF_TOKEN"] = tok
        except Exception:  # noqa: BLE001
            pass
    log_path = cfg.get("log") or os.environ.get("CHON_VLLM_LOG") or os.path.join(
        os.environ.get("TMPDIR", "/tmp"), f"chon-vllm-{port}.log")
    _log(f"starting {binary} (base {base_model}{' from its local copy' if model_path != base_model else ''}, "
         f"{cfg.get('quantization') or 'bf16'}, {util:.0%} of the GPU, log {log_path})")
    tries = [base_cmd + text_only, base_cmd] if cfg.get("text_only", True) else [base_cmd]
    deadline = time.time() + float(cfg.get("start_timeout_s", 1800))
    for k, cmd in enumerate(tries):
        t0 = time.time()
        logf = open(log_path, "ab")
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        if not _procs:
            atexit.register(_stop_all)
        _procs.append(proc)
        while time.time() < deadline:
            if proc.poll() is not None:
                break
            if _alive(url, timeout=3):
                _log(f"ready at {url} after {time.time() - t0:.0f} s")
                return url
            time.sleep(5)
        else:
            _log("vLLM did not come up in time; stopping it")
            _tail(log_path)
            proc.terminate()
            return None
        _log(f"vLLM exited with code {proc.returncode} during start-up after {time.time() - t0:.0f} s "
             f"(see {log_path})")
        _tail(log_path)
        if k + 1 < len(tries) and time.time() - t0 < 300:  # rejected an option at once: retry without it
            _log("retrying without the text-only option")
            continue
        return None
    return None
