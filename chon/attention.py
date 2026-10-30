"""Attention in query chunks for long prompts (1.5.1, 30 Oct).

Gemma 4's full-attention layers have 512-dimensional heads. PyTorch's fused SDPA kernels (flash, memory-efficient)
do not take heads that wide, so SDPA falls back to its reference ("math") path, which holds every head's whole
query x key score matrix at once: heads x n^2 values for an n-token prompt. A 31B reading a 20k-token request
then needs tens of GB on top of the weights. On a 141 GB H200 that fits; on the board's 96 GB RTX PRO 6000 it does
not, and the run stops with "CUDA out of memory" (the Decision Index kit stops at a request's first error).

install() wraps transformers' "sdpa" attention: when one call's score matrices would exceed the budget, the queries
go through in chunks, each against every key with its own slice of the mask. Every query row's softmax is computed
exactly as before (rows are independent), so the answers do not change; small calls (everything under the budget,
every parity case) take the original path untouched. $CHON_ATTN_CHUNK_MB sets the budget (default 1024 MB; 0 turns
chunking off)."""
from __future__ import annotations

import os

_STATE = {"installed": False, "orig": None, "budget": None, "chunked_calls": 0}
DEFAULT_BUDGET_MB = 1024
# heads up to this width go to PyTorch's fused kernels, which never hold the score matrix: left alone (1.5.2; 1.5.1
# chunked them too, which kept them off the fused kernels and slowed every long request)
FUSED_MAX_HEAD_DIM = 256


def budget_bytes() -> int:
    if _STATE["budget"] is not None:
        return _STATE["budget"]
    try:
        mb = float(os.environ.get("CHON_ATTN_CHUNK_MB", DEFAULT_BUDGET_MB))
    except ValueError:
        mb = DEFAULT_BUDGET_MB
    return int(mb * 2 ** 20)


def set_budget(n_bytes: int | None) -> None:
    """Override the budget (None: back to $CHON_ATTN_CHUNK_MB / the default). The engine lowers it to retry a request
    that ran out of memory."""
    _STATE["budget"] = None if n_bytes is None else int(n_bytes)


def chunk_rows(batch: int, heads: int, kv_len: int, budget: int) -> int:
    """Query rows per chunk so that one chunk's float32 scores fit the budget (at least 64)."""
    per_row = max(1, batch * heads * kv_len * 4)
    return max(64, budget // per_row)


def _wrap(orig):
    import torch

    def chunked_sdpa(module, query, key, value, attention_mask, dropout=0.0, scaling=None, is_causal=None, **kwargs):
        budget = budget_bytes()
        b, h, q_len, d = query.shape
        kv_len = key.shape[2]
        if budget <= 0 or d <= FUSED_MAX_HEAD_DIM or kwargs.get("cache") is not None or kwargs.get("position_bias") is not None \
                or b * h * q_len * kv_len * 4 <= budget:
            return orig(module, query, key, value, attention_mask, dropout, scaling, is_causal, **kwargs)
        rows = chunk_rows(b, h, kv_len, budget)
        if rows >= q_len:
            return orig(module, query, key, value, attention_mask, dropout, scaling, is_causal, **kwargs)
        causal = is_causal if is_causal is not None else getattr(module, "is_causal", True)
        causal = bool(causal) and q_len > 1 and attention_mask is None
        if causal and kv_len > q_len:   # as the original: keys past the queries are a static cache's padding
            key, value, kv_len = key[:, :, :q_len], value[:, :, :q_len], q_len
        outs = []
        for s in range(0, q_len, rows):
            e = min(q_len, s + rows)
            if attention_mask is not None:
                m = attention_mask[:, :, s:e, :] if attention_mask.shape[-2] == q_len else attention_mask
            elif causal:
                m = (torch.arange(s, e, device=query.device)[:, None] + (kv_len - q_len)
                     >= torch.arange(kv_len, device=query.device)[None, :]).view(1, 1, e - s, kv_len)
            else:
                m = None
            out, _ = orig(module, query[:, :, s:e], key, value, m, dropout, scaling, False, **kwargs)
            outs.append(out)
        _STATE["chunked_calls"] += 1
        return torch.cat(outs, dim=1), None

    chunked_sdpa.__wrapped__ = orig
    return chunked_sdpa


def install() -> bool:
    """Wrap transformers' "sdpa" attention once per process. Returns whether the wrapper is in place."""
    if _STATE["installed"]:
        return True
    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    except ImportError:
        return False
    orig = ALL_ATTENTION_FUNCTIONS.get("sdpa") if hasattr(ALL_ATTENTION_FUNCTIONS, "get") else None
    if orig is None:
        return False
    ALL_ATTENTION_FUNCTIONS["sdpa"] = _wrap(orig)
    _STATE.update(installed=True, orig=orig)
    return True


def stats() -> dict:
    return {"installed": _STATE["installed"], "budget_mb": budget_bytes() / 2 ** 20,
            "chunked_calls": _STATE["chunked_calls"]}
