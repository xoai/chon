"""The shared-prefix read: the same answers as the plain path, the same reads as the training code's shared-prefix read.

tests/golden_prefix.json was written by `tests/make_golden.py --prefix` from the training repository's DecisionModel
with its shared-prefix read on (chon.core.prefix), over tests/parity_cases.py's PREFIX_CASES (every parity case, plus
states longer than the tiny model's 64-token sliding window).
"""

from __future__ import annotations

import json
import os
import shutil
import time

import pytest
import torch

from parity_cases import CASES, LONG_STATE, MANY, PREFIX_CASES, ROUTE, RR_ALL, SEVERE, WITHIN, run_case

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN = json.load(open(os.path.join(HERE, "golden_prefix.json"), encoding="utf-8"))
PLAIN = json.load(open(os.path.join(HERE, "golden_parity.json"), encoding="utf-8"))
BY_NAME = {c["name"]: c for c in PREFIX_CASES}
TOL_TRAINING = 1e-9   # the training code's shared-prefix read: the same kernels, the same numbers
TOL_PLAIN = 1e-4      # the plain path: different kernel shapes, rounding only


@pytest.fixture(scope="module")
def fixture_ok(tiny_model_dir):
    from tiny_model import fingerprint

    fp, want = fingerprint(tiny_model_dir), GOLDEN["made_with"]["fixture"]
    assert fp["tokenizer"] == want["tokenizer"], "tiny tokenizer differs: regenerate tests/golden_prefix.json"
    return fp["weights"] == want["weights"]


@pytest.fixture(scope="module")
def adapter_ok(tiny_dm):
    """Whether the tiny LoRA adapter's weights are the ones the golden pair cases were made with."""
    from parity_cases import adapter_model
    from tiny_model import adapter_fingerprint

    return adapter_fingerprint(adapter_model(tiny_dm.model)) == GOLDEN["made_with"]["fixture"].get("adapter")


def _floats(x, path=""):
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            out.update(_floats(v, f"{path}/{k}"))
        return out
    if isinstance(x, (list, tuple)):
        out = {}
        for i, v in enumerate(x):
            out.update(_floats(v, f"{path}/{i}"))
        return out
    return {path: x}


def _close(got, want, tol, label):
    a, b = _floats(got), _floats(want)
    assert set(a) == set(b), f"{label}: different keys {set(a) ^ set(b)}"
    for k in b:
        if isinstance(b[k], float):
            assert abs(a[k] - b[k]) <= tol, f"{label}{k}: {a[k]!r} != {b[k]!r}"
        else:
            assert a[k] == b[k], f"{label}{k}: {a[k]!r} != {b[k]!r}"


def _out(r):
    return r.get("answers", r.get("distributions"))


def test_golden_covers_every_case():
    assert set(GOLDEN["cases"]) == set(BY_NAME) and GOLDEN["made_with"]["prefix_cache"]
    assert not GOLDEN["made_with"]["training_tree_dirty"]


@pytest.mark.parametrize("name", [c["name"] for c in PREFIX_CASES])
def test_prefix_read_matches_training_code_and_plain_path(name, tiny_dm, fixture_ok, adapter_ok):
    from chon.infer import DecisionModel

    case = BY_NAME[name]
    got = run_case(DecisionModel, tiny_dm.model, tiny_dm.tok, case, prefix_cache=True)
    want = GOLDEN["cases"][name]
    assert got["reread"] == want["reread"]
    assert got["prefix"] == want["prefix"], "a different shared prefix than the training code"
    assert got["letter_reads"] == want["letter_reads"], "different reads (ids, options or prefix) than the training code"
    assert got["reads"] == want["reads"], "different plain-path reads than the training code"
    for k in ("usage", "served_reads"):
        if k in want:
            assert got[k] == want[k]
    if not fixture_ok or (case.get("adapter") and not adapter_ok):
        pytest.skip("reads match; the tiny weights differ from the golden ones (torch version?)")
    _close(_out(got), _out(want), TOL_TRAINING, name)
    # and the plain path's answers: the golden ones for the parity cases, a plain run for the long states
    plain = PLAIN["cases"][name] if name in PLAIN["cases"] else \
        run_case(DecisionModel, tiny_dm.model, tiny_dm.tok, case, prefix_cache=False)
    _close(_out(got), _out(plain), TOL_PLAIN, name)
    if "usage" in want:
        assert {k: v for k, v in got["usage"].items() if k != "prefix_tokens"} == plain["usage"]


def test_cases_cover_what_they_claim():
    """The golden shared-prefix reads cross the sliding window and cover first reads, group and final rounds, second
    reads in both layouts (sandwich falls back), offline reads, and one-question requests (plain)."""
    g = GOLDEN["cases"]
    shared = lambda n: [r["prefix"] for r in g[n]["letter_reads"]]
    assert g["long_state_first"]["prefix"][0] > 64 and g["long_header"]["prefix"][0] > 64
    assert shared("long_header_many_reread") == [g["long_header_many_reread"]["prefix"][0]] * 4   # first, reread,
    assert [r["n_options"] for r in g["long_header_many_reread"]["letter_reads"]][2] == [14, 14, 12, 10, 10]  # groups,
    assert g["long_header_many_reread"]["letter_reads"][3]["n_options"] == [3, 2]                            # finals
    assert g["long_header_many_reread"]["usage"]["reread"] == 2 and not g["long_header_many_reread"]["reads"]
    assert g["long_sandwich"]["reads"] and g["long_sandwich"]["usage"]["reread"] == 2   # sandwich: plain second read
    assert g["header_one"]["prefix"] == [0] and len(g["header_one"]["reads"]) == 1     # one question: plain
    assert len(g["long_dist_header"]["prefix"]) == 2 and len(g["long_dist_header"]["reads"]) == 1


def test_long_reads_cross_the_window(tiny_dm):
    from chon.prompt import HEADER_LINE

    g = GOLDEN["cases"]["long_header_many_reread"]
    p = g["prefix"][0]
    head = tiny_dm.tok.decode(g["letter_reads"][0]["ids"][0][:p])
    assert HEADER_LINE in head and LONG_STATE[-40:] in head   # question list and state, up to where questions differ
    assert head.split(LONG_STATE[-40:])[1] in ("\n\n---\n", "\n\n---\nQuestion:", "\n\n---\nQuestion: Which")


# ---- the module ----------------------------------------------------------------------------------------------------

def test_helpers():
    from chon.prefix import common_prefix, pack_budget, packed_batches

    assert common_prefix([[1, 2, 3], [1, 2, 4], [1, 2]]) == 2
    assert common_prefix([[5], [6]]) == 0 and common_prefix([]) == 0 and common_prefix([[1, 2]]) == 2
    assert common_prefix([[1, 2, 3, 9], [1, 2, 3, 9, 4], [1, 2, 3, 8]]) == 3
    assert packed_batches([5, 5, 5, 20, 1], 10) == [[0, 1], [2], [3], [4]]
    assert pack_budget(8192, 50_000) == 8192                 # a 50,000-token state: the configured pack
    assert 256 <= pack_budget(8192, 500_000) < 8192           # a very long one: the mask stays under 2**30 cells
    assert pack_budget(100, 10) == 100


def _tiny_text_model(attn="sdpa", kv_shared=0, k_eq_v=False):
    from transformers import Gemma4ForCausalLM, Gemma4TextConfig

    torch.manual_seed(0)
    cfg = Gemma4TextConfig(vocab_size=300, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                           num_attention_heads=2, num_key_value_heads=1, head_dim=32, global_head_dim=32,
                           layer_types=["sliding_attention", "full_attention"] * 2, sliding_window=16,
                           num_kv_shared_layers=kv_shared, attention_k_eq_v=k_eq_v,
                           vocab_size_per_layer_input=300, hidden_size_per_layer_input=8,
                           max_position_embeddings=4096, pad_token_id=0, eos_token_id=1, bos_token_id=2)
    cfg._attn_implementation = attn
    return Gemma4ForCausalLM(cfg).eval()


@pytest.mark.parametrize("attn,kv_shared,k_eq_v", [("sdpa", 0, False), ("eager", 0, False), ("sdpa", 2, True)])
def test_shared_prefix_equals_separate_passes(attn, kv_shared, k_eq_v):
    """Sliding window crossed by the prefix and by a suffix, KV-shared layers (E2B/E4B) and full layers with k = v
    (the 31B), packed in one pass and in several."""
    from chon.prefix import SharedPrefix

    model = _tiny_text_model(attn, kv_shared, k_eq_v)
    g = torch.Generator().manual_seed(1)
    prefix = torch.randint(3, 300, (50,), generator=g).tolist()
    sufs = [torch.randint(3, 300, (n,), generator=g).tolist() for n in (7, 20, 3, 1)]
    letters = torch.tensor([10, 11, 12, 13])
    with torch.inference_mode():
        want = [model(input_ids=torch.tensor([prefix + s]), use_cache=False, logits_to_keep=1).logits[0, -1, letters]
                for s in sufs]
    sp = SharedPrefix(model, prefix, torch.device("cpu"))
    for got in (sp.letter_logits(sufs, letters), torch.cat([sp.letter_logits([s], letters) for s in sufs])):
        for row, w in zip(got, want):
            assert torch.allclose(row, w.float(), atol=1e-5)


def test_sliding_layers_keep_only_their_window():
    """A sliding layer's cached keys/values are a copy of its last window - 1 positions, not a view that pins the whole
    prefix (Gemma 4 31B at 50,000 tokens: 0.8 GB instead of 41 GB); full layers keep the prefix."""
    from chon.prefix import SharedPrefix

    model = _tiny_text_model()
    sp = SharedPrefix(model, list(range(3, 203)), torch.device("cpu"))
    assert sp.kept == {"sliding_attention": 15, "full_attention": 200}
    for kind, k, v in zip(sp.types, sp.keys, sp.values):
        for t in (k, v):
            assert t.untyped_storage().nbytes() == t.numel() * t.element_size()
            assert t.shape[-2] == sp.kept[kind]
    per_pos = 2 * 1 * 32 * 4   # keys + values, 1 head of 32 floats
    assert sp.nbytes() == 2 * per_pos * (15 + 200)
    assert not any(t.requires_grad for t in sp.keys + sp.values)


# ---- DecisionModel: settings, fallback, packing, deadline, reset ----------------------------------------------------

QS = {"route": ROUTE, "within": WITHIN, "severe": SEVERE, "line": MANY}


def _dm(tiny_dm, **cfg):
    from chon.infer import DecisionModel

    return DecisionModel(tiny_dm.model, tiny_dm.tok, {"prompt_layout": "header", **cfg}, device="cpu")


def test_on_by_default_and_off_by_setting(tiny_dm):
    on, off = _dm(tiny_dm), _dm(tiny_dm, prefix_cache=False)
    assert on.prefix_cache and not off.prefix_cache and not _dm(tiny_dm, prefix_cache="off").prefix_cache
    a1, u1 = on.predict(LONG_STATE, QS)
    a0, u0 = off.predict(LONG_STATE, QS)
    assert u1["prefix_tokens"] > 128 and "prefix_tokens" not in u0
    assert on.prefix_stats["requests"] == 1 and on.prefix_stats["reads"] == 3 + 3 + 1   # first reads, groups, final
    assert on._shared is None   # released after the request
    _close(a1, a0, TOL_PLAIN, "default")
    _, u = on.predict("short state", {"route": ROUTE, "within": WITHIN})   # under prefix_min_tokens: plain
    assert "prefix_tokens" not in u


def test_small_packs_give_the_same_answers(tiny_dm, monkeypatch):
    from chon.prefix import SharedPrefix

    a1, _ = _dm(tiny_dm).predict(LONG_STATE, QS)
    passes = []
    real = SharedPrefix.letter_logits
    monkeypatch.setattr(SharedPrefix, "letter_logits", lambda self, sufs, ids: passes.append(len(sufs)) or
                        real(self, sufs, ids))
    a2, _ = _dm(tiny_dm, prefix_pack_tokens=16).predict(LONG_STATE, QS)   # every suffix in its own pass
    assert passes == [1] * 7
    _close(a2, a1, 1e-6, "packs")


def test_unsupported_model_reads_plainly(tiny_dm, capsys):
    """A model the shared-prefix read cannot serve: said once when it loads, then the plain path's exact reads."""
    from types import SimpleNamespace

    from chon.infer import DecisionModel

    class Other:   # another architecture: a layer type the masks do not know
        config = SimpleNamespace(layer_types=["linear_attention", "full_attention"], _attn_implementation="sdpa")

        def __call__(self, *a, **k):
            return tiny_dm.model(*a, **k)

        def parameters(self):
            return tiny_dm.model.parameters()

    dm = DecisionModel(Other(), tiny_dm.tok, {"prompt_layout": "header"}, device="cpu")
    assert not dm.prefix_cache and capsys.readouterr().out.count("[prefix] shared-prefix read off") == 1
    a, u = dm.predict(LONG_STATE, QS)
    assert "prefix_tokens" not in u
    _close(a, _dm(tiny_dm, prefix_cache=False).predict(LONG_STATE, QS)[0], 0.0, "unsupported")


def test_a_failing_prefix_read_falls_back_once(tiny_dm, monkeypatch, capsys):
    import chon.infer as infer

    dm = _dm(tiny_dm)

    def broken(*a, **k):
        raise infer.Unsupported("no cache from this model")
    monkeypatch.setattr(infer, "SharedPrefix", broken)
    a, u = dm.predict(LONG_STATE, QS)
    assert dm._prefix_off and not dm.prefix_cache and "prefix_tokens" not in u
    dm.predict(LONG_STATE, QS)
    assert capsys.readouterr().out.count("[prefix] shared-prefix read off") == 1
    _close(a, _dm(tiny_dm, prefix_cache=False).predict(LONG_STATE, QS)[0], 0.0, "fallback")


def test_deadline_with_the_prefix_read(tiny_dm):
    dm = _dm(tiny_dm, reread=RR_ALL)
    a_free, u_free = dm.predict(LONG_STATE, {"route": ROUTE, "within": WITHIN})
    a_late, u_late = dm.predict(LONG_STATE, {"route": ROUTE, "within": WITHIN}, deadline=time.monotonic() + 60)
    assert u_free["reread"] == 2 and u_late == {**u_free, "forward_s": u_late["forward_s"]} and a_late == a_free
    a_out, u_out = dm.predict(LONG_STATE, {"route": ROUTE, "within": WITHIN}, deadline=time.monotonic() - 1)
    assert "reread" not in u_out and u_out["prefix_tokens"] == u_free["prefix_tokens"]   # first reads still shared
    first = _dm(tiny_dm).predict(LONG_STATE, {"route": ROUTE, "within": WITHIN})[0]
    assert a_out == first and dm._deadline is None and dm._shared is None


def test_request_state_is_reset_after_an_error(tiny_dm):
    dm = _dm(tiny_dm)

    def boom(*a, **k):
        raise RuntimeError("forward failed")
    dm._logits = boom
    with pytest.raises(RuntimeError):
        dm.predict(LONG_STATE, QS, deadline=time.monotonic() + 10)
    assert dm._shared is None and dm._req_layout is None and dm._req_asked is None and dm._deadline is None


def test_distributions_share_prefix(tiny_dm):
    dm = _dm(tiny_dm, prompt_layout="state_first")
    reqs = [(LONG_STATE, ROUTE, None), (LONG_STATE, WITHIN, None), ("A short other note.", WITHIN, None)]
    calls = []
    real = dm.raw_logits
    dm.raw_logits = lambda seqs, n, letter_ids=None: calls.append(len(seqs)) or real(seqs, n)
    shared = dm.distributions(reqs, calibrated=False, share_prefix=True)
    assert calls == [1]   # only the lone request read plainly
    plain = dm.distributions(reqs, calibrated=False)
    _close(shared, plain, TOL_PLAIN, "distributions")
    assert dm._shared is None and "prefix_cache" not in dm.cfg


def test_settings_from_the_model_folder_and_the_engine(tiny_model_dir, tmp_path, monkeypatch):
    from chon.dindex import ChonEngine
    from chon.infer import DecisionModel

    monkeypatch.delenv("CHON_REREAD", raising=False)
    d = tmp_path / "pkg"
    shutil.copytree(tiny_model_dir, d)
    (d / "chon_config.json").write_text(json.dumps({"prefix_cache": False}))
    assert not DecisionModel.load(str(d), device="cpu", dtype="float32").prefix_cache
    e = ChonEngine(model=str(d), device="cpu")
    assert e.provenance["prefix_cache"] is False and "no shared-prefix read" in e.provenance["policy"]
    e = ChonEngine(model=str(d), device="cpu", prefix_cache="true")   # the kit's --option prefix_cache=true
    assert e.provenance["prefix_cache"] is True and "key/value cache" in e.provenance["policy"]
    (d / "chon_config.json").write_text("{}")
    e = ChonEngine(model=str(d), device="cpu")
    assert e.dm.prefix_cache and e.provenance["prefix_cache"] is True
    assert not ChonEngine(model=str(d), device="cpu", prefix_cache="false").dm.prefix_cache
    out, _ = e(LONG_STATE, {"route": ROUTE, "within": WITHIN})
    assert out["usage"]["prefix_tokens"] > 128


def test_health_reports_the_setting(tiny_dm):
    from fastapi.testclient import TestClient

    from chon.serve import create_app

    h = TestClient(create_app(_dm(tiny_dm), None)).get("/health").json()
    assert h["prefix_cache"] is True and set(h["prefix_stats"]) == {"requests", "prefix_tokens", "reads"}


def test_plain_parity_cases_stay_on_the_plain_path(tiny_dm):
    """The parity harness pins the plain path (the golden values of tests/test_parity.py), whatever the default."""
    from chon.infer import DecisionModel

    got = run_case(DecisionModel, tiny_dm.model, tiny_dm.tok, next(c for c in CASES if c["name"] == "header_multi"))
    assert "prefix_tokens" not in got["usage"] and got["reads"] == PLAIN["cases"]["header_multi"]["reads"]


def test_pair_and_reverse_reads_go_through_the_shared_prefix():
    """1.5.0: each member of a pair read prepares the request's shared prefix (the base read's keys and values with the
    adapter off), and a reverse second read, which starts with the first read's prefix, reads against it."""
    g = GOLDEN["cases"]
    for name in ("long_pair_header_reread", "long_pair_dec_reverse", "long_pair_many_commit", "pair_reverse"):
        r = g[name]
        assert len(r["prefix"]) == 2 and r["prefix"][0] == r["prefix"][1] > 0 and not r["reads"]
        assert all(x["prefix"] == r["prefix"][0] for x in r["letter_reads"]) and r["usage"]["pair"] == 2
    for name in ("long_rev_header", "rev_header_avg"):
        r = g[name]
        assert len(r["letter_reads"]) == 2 and r["letter_reads"][1]["prefix"] == r["prefix"][0] > 0
