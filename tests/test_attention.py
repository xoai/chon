"""Attention in query chunks and the out-of-memory retry (1.5.1): bounded GPU memory, the same answers."""
from __future__ import annotations

import json
import shutil

import pytest
import torch

from chon import attention

STATE = "Order 1182: two parcels, one damaged on arrival, refund requested by the customer after a week."
Q = {"route": {"type": "choice", "instructions": "Which team handles it?",
               "criteria": {"billing": "refunds and invoices", "shipping": "transport damage", "other": "anything else"}},
     "ok": {"type": "noul", "instructions": "Is a refund due?"}}


class _Mod:
    def __init__(self, groups=1, causal=True):
        self.num_key_value_groups, self.is_causal, self.layer_idx = groups, causal, 0


@pytest.fixture
def small_budget():
    attention.install()
    attention.set_budget(4 * 97 * 160 * 4 // 3)   # a few chunks for these sizes
    yield
    attention.set_budget(None)


def _qkv(b=1, h=4, kvh=4, q=97, kv=97, d=272, seed=0):   # wider than the fused kernels take, as Gemma 4's 512
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(b, h, q, d, generator=g), torch.randn(b, kvh, kv, d, generator=g),
            torch.randn(b, kvh, kv, d, generator=g))


@pytest.mark.parametrize("case", ["causal", "bool_mask", "float_mask", "prefix_mask", "not_causal", "gqa_mask",
                                  "causal_static_padding"])
def test_chunked_equals_the_original(case, small_budget):
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    fn = ALL_ATTENTION_FUNCTIONS["sdpa"]
    orig = fn.__wrapped__
    mod, mask, kv = _Mod(), None, 97
    if case == "prefix_mask":
        kv = 160                             # a suffix pass against a cached prefix: an explicit mask
    if case == "causal_static_padding":
        kv = 130
    if case == "gqa_mask":
        mod = _Mod(groups=2)
    q, k, v = _qkv(kvh=2 if case == "gqa_mask" else 4, kv=kv)
    if case in ("bool_mask", "gqa_mask", "prefix_mask"):
        m = torch.ones(97, kv, dtype=torch.bool).tril(kv - 97)
        m[:, :5] = False                     # some left padding
        mask = m.view(1, 1, 97, kv)
    elif case == "float_mask":
        mask = torch.zeros(1, 1, 97, kv).masked_fill(~torch.ones(97, kv, dtype=torch.bool).tril(), float("-inf"))
    if case == "not_causal":
        mod = _Mod(causal=False)
    before = attention.stats()["chunked_calls"]
    got, _ = fn(mod, q, k, v, mask, 0.0, 0.25, None)
    want, _ = orig(mod, q, k, v, mask, 0.0, 0.25, None)
    assert attention.stats()["chunked_calls"] == before + 1          # this call did go through in chunks
    assert got.shape == want.shape and torch.allclose(got, want, atol=1e-5, rtol=1e-5)


def test_small_calls_take_the_original_path():
    attention.install()
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    q, k, v = _qkv(q=8, kv=8)
    before = attention.stats()["chunked_calls"]
    ALL_ATTENTION_FUNCTIONS["sdpa"](_Mod(), q, k, v, None, 0.0, None, None)
    assert attention.stats()["chunked_calls"] == before
    attention.set_budget(64)                  # heads the fused kernels take are never chunked, whatever the budget
    try:
        q, k, v = _qkv(q=97, kv=97, d=64)
        ALL_ATTENTION_FUNCTIONS["sdpa"](_Mod(), q, k, v, None, 0.0, None, None)
        assert attention.stats()["chunked_calls"] == before
    finally:
        attention.set_budget(None)
    assert attention.chunk_rows(1, 32, 20000, 2 ** 30) == 2 ** 30 // (32 * 20000 * 4)
    assert attention.chunk_rows(1, 32, 10 ** 9, 2 ** 20) == 64


def test_a_whole_request_reads_the_same_in_chunks(tiny_dm, monkeypatch):
    attention.install()
    monkeypatch.setattr(attention, "FUSED_MAX_HEAD_DIM", 0)   # the tiny model's heads are narrow
    attention.set_budget(None)
    want, _ = tiny_dm.predict(STATE, Q)
    attention.set_budget(1024)
    try:
        before = attention.stats()["chunked_calls"]
        got, _ = tiny_dm.predict(STATE, Q)
        assert attention.stats()["chunked_calls"] > before
    finally:
        attention.set_budget(None)
    for n in Q:
        for o, p in want[n]["probabilities"].items():
            assert abs(got[n]["probabilities"][o] - p) < 1e-5


def test_engine_retries_a_request_that_ran_out_of_memory(tiny_model_dir, tmp_path, monkeypatch):
    from chon.dindex import ChonEngine

    monkeypatch.delenv("CHON_REREAD", raising=False)
    d = tmp_path / "pkg"
    shutil.copytree(tiny_model_dir, d)
    (d / "chon_config.json").write_text(json.dumps({"prompt_layout": "header"}))
    e = ChonEngine(model=str(d), device="cpu")
    want, _ = e(STATE, Q)
    real, seen = e.dm.predict, []

    def flaky(state, questions, **kw):
        seen.append((e.dm.max_batch_tokens, attention.budget_bytes(), e.dm.cfg.get("prefix_pack_tokens")))
        if len(seen) == 1:
            raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 7.28 GiB.")
        return real(state, questions, **kw)

    monkeypatch.setattr(e.dm, "predict", flaky)
    got, _ = e(STATE, Q)
    assert got["usage"]["oom_retry"] == 1 and e.oom_retries == 1
    assert seen[1][0] == 32768 // 4 and seen[1][1] == attention.DEFAULT_BUDGET_MB * 2 ** 20 // 4 and seen[1][2] == 2048
    assert e.dm.max_batch_tokens == 32768 and attention.budget_bytes() == attention.DEFAULT_BUDGET_MB * 2 ** 20
    assert "prefix_pack_tokens" not in e.dm.cfg                       # every budget put back
    for n in Q:
        for o, p in want["answers"][n]["probabilities"].items():
            assert abs(got["answers"][n]["probabilities"][o] - p) < 1e-6

    def always(state, questions, **kw):
        raise torch.OutOfMemoryError("CUDA out of memory.")

    monkeypatch.setattr(e.dm, "predict", always)
    from decision_index.engines.base import Unsupported
    with pytest.raises(Unsupported, match="out of GPU memory"):     # a second failure: unsupported, the run goes on
        e(STATE, Q)
    assert e.dm.max_batch_tokens == 32768 and e.oom_unsupported == 1
    from chon import prefix
    assert prefix.prefill_chunk() == prefix.DEFAULT_PREFILL_CHUNK

    def other(state, questions, **kw):
        raise ValueError("not memory")

    monkeypatch.setattr(e.dm, "predict", other)
    with pytest.raises(ValueError):                                   # other errors are not retried
        e(STATE, Q)
    assert "expandable_segments" in __import__("os").environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    assert "query chunks" in e.provenance["policy"]


# ---- a long prompt fills the cache in chunks (1.5.2) ---------------------------------------------------------------

LONG_STATE = " ".join(f"line {i}: parcel {i % 7} weighed {i * 3 % 11} kg, status {'late' if i % 5 == 0 else 'ok'}."
                      for i in range(60))


@pytest.mark.parametrize("questions", [Q, {"route": Q["route"]}], ids=["two_questions_shared_prefix", "one_question"])
def test_chunked_prefill_reads_the_same(tiny_dm, questions):
    from chon import prefix
    prefix.set_prefill_chunk(0)
    try:
        want, _ = tiny_dm.predict(LONG_STATE, questions)
        prefix.set_prefill_chunk(7)            # far smaller than the prompt and the tiny model's window
        calls = []
        real = prefix.SharedPrefix.__init__

        def spy(self, model, tokens, device):
            calls.append(len(tokens))
            return real(self, model, tokens, device)

        prefix.SharedPrefix.__init__ = spy
        try:
            got, _ = tiny_dm.predict(LONG_STATE, questions)
        finally:
            prefix.SharedPrefix.__init__ = real
    finally:
        prefix.set_prefill_chunk(None)
    assert calls and max(calls) > 7                      # a prefix longer than the chunk went through in chunks
    for n in questions:
        for o, p in want[n]["probabilities"].items():
            assert abs(got[n]["probabilities"][o] - p) < 1e-5
