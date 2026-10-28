"""Parity with the training code: the same token ids for every read and the same output probabilities.

tests/golden_parity.json was written by tests/make_golden.py from the training repository's DecisionModel on the
tiny model of tests/tiny_model.py; every case of tests/parity_cases.py is run here through chon.infer and compared.
"""

from __future__ import annotations

import json
import os
import shutil
import time

import pytest

from parity_cases import CAP, CASES, DEC, RR_ALL, ROUTE, SEVERE, TEXT_STATE, WITHIN, run_case

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN = json.load(open(os.path.join(HERE, "golden_parity.json"), encoding="utf-8"))
BY_NAME = {c["name"]: c for c in CASES}
TOL = 1e-9


@pytest.fixture(scope="module")
def fixture_ok(tiny_model_dir):
    from tiny_model import fingerprint

    fp, want = fingerprint(tiny_model_dir), GOLDEN["made_with"]["fixture"]
    assert fp["tokenizer"] == want["tokenizer"], (
        "the tiny tokenizer differs from the one the golden values were made with (tokenizers version?): "
        f"regenerate tests/golden_parity.json with tests/make_golden.py ({fp} vs {want})")
    return fp["weights"] == want["weights"]


def _floats(x, path=""):
    """Flatten the numbers of an answers / distributions structure: {path: value}."""
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


def _same_outputs(got, want, label):
    a, b = _floats(got), _floats(want)
    assert set(a) == set(b), f"{label}: different keys {set(a) ^ set(b)}"
    for k in b:
        if isinstance(b[k], float):
            assert abs(a[k] - b[k]) <= TOL, f"{label}{k}: {a[k]!r} != {b[k]!r}"
        else:
            assert a[k] == b[k], f"{label}{k}: {a[k]!r} != {b[k]!r}"


def test_golden_covers_every_case():
    assert set(GOLDEN["cases"]) == set(BY_NAME)
    assert not GOLDEN["made_with"]["training_tree_dirty"]


@pytest.mark.parametrize("name", [c["name"] for c in CASES])
def test_parity_with_training_code(name, tiny_dm, fixture_ok):
    from chon.infer import DecisionModel

    got = run_case(DecisionModel, tiny_dm.model, tiny_dm.tok, BY_NAME[name])
    want = GOLDEN["cases"][name]
    assert got["reread"] == want["reread"]
    assert len(got["reads"]) == len(want["reads"]), "a different number of forward calls"
    for k, (g, w) in enumerate(zip(got["reads"], want["reads"])):
        assert g["n_options"] == w["n_options"], f"call {k}: options"
        assert g["ids"] == w["ids"], f"call {k}: token ids differ from the training code"
    if "usage" in want:
        assert got["usage"] == want["usage"]
    if "served_reads" in want:
        assert got["served_reads"] == want["served_reads"]
    if not fixture_ok:
        pytest.skip("token ids match; the tiny weights differ from the golden ones (torch version?): "
                    "regenerate tests/golden_parity.json to compare probabilities")
    _same_outputs(got.get("answers", got.get("distributions")), want.get("answers", want.get("distributions")), name)


def test_cases_cover_what_they_claim():
    """The golden reads exercise each feature: header lists, group + final rounds, second reads in both layouts."""
    g = GOLDEN["cases"]
    assert len(g["state_first_many"]["reads"]) == 3 and g["state_first_many"]["reads"][1]["n_options"] == [14, 14, 12]
    assert g["state_first_many"]["reads"][2]["n_options"] == [3]
    assert g["reread_header_avg"]["usage"]["reread"] == 2 and g["reread_header_replace"]["usage"]["reread"] == 2
    assert g["reread_max_tokens"]["usage"]["reread"] == 1 and g["reread_types_choice"]["usage"]["reread"] == 1
    assert "reread" not in g["env_off"]["usage"] and g["env_header_replace"]["reread"]["combine"] == "replace"
    assert g["dist_hybrid"]["served_reads"][0][0] == "header" and g["dist_hybrid"]["served_reads"][3][0] == "state_first"


def _decoded(dm, case_name, call=0, row=0):
    return dm.tok.decode(GOLDEN["cases"][case_name]["reads"][call]["ids"][row])


def test_golden_prompts_read_as_documented(tiny_dm):
    from chon.prompt import HEADER_LINE, REREAD_LINE

    hm = _decoded(tiny_dm, "header_multi")   # distinct instructions in request order, whitespace collapsed
    assert hm.index(HEADER_LINE) < hm.index("- Which action") < hm.index("- Is the parcel") \
        < hm.index("- How severe is it? (rate it)") < hm.index('- {"task":"Pick the team"') < hm.index("Parcel 7")
    same = _decoded(tiny_dm, "header_same_instructions")
    assert same.count("- Which action does the ticket request?") == 1 and same.count("\n- ") == 2
    assert HEADER_LINE not in _decoded(tiny_dm, "hybrid_same_instructions")    # one distinct question: state-first
    assert HEADER_LINE in _decoded(tiny_dm, "hybrid_same_plus_one")
    assert HEADER_LINE not in _decoded(tiny_dm, "hybrid_one") and HEADER_LINE in _decoded(tiny_dm, "hybrid_multi")
    final = _decoded(tiny_dm, "header_many", call=2, row=0)    # the final round lists the request's questions
    assert final.count("\n- ") == 3 and "- Which warehouse ships it?" in final
    tw = _decoded(tiny_dm, "reread_header_avg", call=1)        # a header model's second read: twice_header
    assert tw.count(HEADER_LINE) == 2 and tw.count(REREAD_LINE) == 1 and tw.count("Answer with one letter.") == 1
    tw1 = _decoded(tiny_dm, "reread_hybrid_one_avg", call=1)   # one question of a hybrid model: state-first twice
    assert HEADER_LINE not in tw1 and tw1.count(TEXT_STATE) == 2 and REREAD_LINE in tw1
    sw = _decoded(tiny_dm, "reread_sandwich", call=1)
    assert HEADER_LINE not in sw and sw.count("Question:") == 2 and sw.count(TEXT_STATE) == 1


# ---- the deadline and the second read --------------------------------------------------------------------------

def test_reread_runs_within_a_deadline(tiny_dm, fixture_ok):
    from chon.infer import DecisionModel

    if not fixture_ok:
        pytest.skip("tiny weights differ from the golden ones")
    for name in ("reread_header_avg", "reread_hybrid_multi_replace", "reread_state_first_avg"):
        got = run_case(DecisionModel, tiny_dm.model, tiny_dm.tok, BY_NAME[name], deadline=time.monotonic() + 60)
        want = GOLDEN["cases"][name]
        assert [r["ids"] for r in got["reads"]] == [r["ids"] for r in want["reads"]]
        _same_outputs(got["answers"], want["answers"], name)


def test_reread_is_skipped_when_out_of_time(tiny_dm):
    from chon.infer import DecisionModel

    case = BY_NAME["reread_header_avg"]
    once = run_case(DecisionModel, tiny_dm.model, tiny_dm.tok, {**case, "cfg": {k: v for k, v in case["cfg"].items()
                                                                                  if k != "reread"}})
    late = run_case(DecisionModel, tiny_dm.model, tiny_dm.tok, case, deadline=time.monotonic() + 0.2)
    assert len(late["reads"]) == 1 and "reread" not in late["usage"]       # the first read stands
    assert late["answers"] == once["answers"] and late["usage"] == once["usage"]
    dm = DecisionModel(tiny_dm.model, tiny_dm.tok, dict(case["cfg"]), device="cpu")
    dm.predict(case["state"], case["questions"], deadline=time.monotonic() - 1)
    assert dm.reread_stats["deadline_skipped"] == 1 and dm.reread_stats["reread"] == 0
    assert dm._deadline is None and dm._req_layout is None and dm._req_asked is None


# ---- settings ---------------------------------------------------------------------------------------------------

def test_parse_reread_config_and_env():
    from chon.infer import REREAD_DEFAULTS, parse_reread

    assert parse_reread(None) is None and parse_reread({"below": 0}) is None
    assert parse_reread(0.9) == {**REREAD_DEFAULTS, "below": 0.9}
    assert parse_reread(None, "0.85,sandwich,1500") == {**REREAD_DEFAULTS, "below": 0.85, "layout": "sandwich",
                                                       "max_tokens": 1500}
    assert parse_reread({"below": 0.9}, "off") is None
    assert parse_reread({"below": 0.9, "types": ["choice"]}, "0.95")["types"] == ["choice"]
    assert parse_reread(None, "0.9,twice,,replace") == {**REREAD_DEFAULTS, "combine": "replace"}
    assert parse_reread(0.9)["combine"] == "avg"
    for bad in ("0.9,thrice", "0.9,twice,,max", "0.9,twice_header"):
        with pytest.raises(ValueError):
            parse_reread(None, bad)


@pytest.mark.parametrize("layout", ["question_first", "hybrid", "headers", "twice"])
def test_unknown_prompt_layout_is_refused(tiny_dm, layout):
    from chon.infer import DecisionModel

    with pytest.raises(ValueError, match="unknown prompt_layout"):
        DecisionModel(tiny_dm.model, tiny_dm.tok, {"prompt_layout": layout}, device="cpu")
    for ok in ("state_first", "header", "hybrid_header"):
        assert DecisionModel(tiny_dm.model, tiny_dm.tok, {"prompt_layout": ok}, device="cpu").layout == ok


def test_settings_load_from_the_model_folder(tiny_model_dir, tmp_path, monkeypatch):
    """chon_config.json's prompt_layout and reread, read by DecisionModel.load, give the golden reads."""
    from chon.infer import DecisionModel

    monkeypatch.delenv("CHON_REREAD", raising=False)
    d = tmp_path / "pkg"
    shutil.copytree(tiny_model_dir, d)
    case = BY_NAME["reread_hybrid_multi_avg"]
    (d / "chon_config.json").write_text(json.dumps(case["cfg"]))
    dm = DecisionModel.load(str(d), device="cpu", dtype="float32")
    assert dm.layout == "hybrid_header" and dm.reread["below"] == RR_ALL["below"]
    seen = []
    real = dm.raw_logits
    dm.raw_logits = lambda seqs, n, letter_ids=None: (seen.append([list(s) for s in seqs]), real(seqs, n))[1]
    _, usage = dm.predict(case["state"], case["questions"])
    want = GOLDEN["cases"][case["name"]]
    assert seen == [r["ids"] for r in want["reads"]] and usage["reread"] == want["usage"]["reread"]
    (d / "chon_config.json").write_text(json.dumps({"prompt_layout": "question_first"}))
    with pytest.raises(ValueError, match="unknown prompt_layout 'question_first'"):
        DecisionModel.load(str(d), device="cpu", dtype="float32")


def test_request_state_is_reset_after_an_error(tiny_dm):
    from chon.infer import DecisionModel

    dm = DecisionModel(tiny_dm.model, tiny_dm.tok, {"prompt_layout": "header", "reread": RR_ALL}, device="cpu")

    def boom(*a, **k):
        raise RuntimeError("forward failed")
    dm.raw_logits = boom
    with pytest.raises(RuntimeError):
        dm.predict(TEXT_STATE, {"a": ROUTE, "b": WITHIN}, deadline=time.monotonic() + 10)
    assert dm._req_layout is None and dm._req_asked is None and dm._deadline is None


def test_dindex_provenance_states_layout_and_second_read(tiny_model_dir, tmp_path, monkeypatch):
    from chon.dindex import ChonEngine

    monkeypatch.delenv("CHON_REREAD", raising=False)
    d = tmp_path / "pkg"
    shutil.copytree(tiny_model_dir, d)
    (d / "chon_config.json").write_text(json.dumps({"prompt_layout": "hybrid_header",
                                                    "reread": {"below": 0.8, "combine": "replace"}}))
    e = ChonEngine(model=str(d), device="cpu")
    p = e.provenance
    assert p["prompt_layout"] == "hybrid_header" and p["reread"]["below"] == 0.8 and p["reread"]["combine"] == "replace"
    assert "'hybrid_header'" in p["policy"] and "under 0.8" in p["policy"] and "replaces the first" in p["policy"]
    monkeypatch.setenv("CHON_REREAD", "off")
    (d / "chon_config.json").write_text("{}")
    p = ChonEngine(model=str(d), device="cpu").provenance
    assert p["prompt_layout"] == "state_first" and p["reread"] is None and "no second read" in p["policy"]


# ---- prompt style "decider" and uncapped letters (1.4.0) ----------------------------------------------------------

def test_decider_cases_cover_what_they_claim():
    g = GOLDEN["cases"]
    assert g["dec_reread_header_avg"]["usage"]["reread"] == 2 and g["dec_reread_header_threshold"]["usage"]["reread"] == 2
    assert g["dec_reread_max_tokens"]["usage"]["reread"] == 1 and g["dec_env_header_replace"]["reread"]["combine"] == "replace"
    assert [r["n_options"] for r in g["dec_header_many"]["reads"]][1] == [14, 14, 12, 10, 10]
    # the decider list keeps duplicates: two questions with the same instructions make a hybrid model read the header
    assert g["dec_dist_hybrid_reread"]["served_reads"][0] == ["header", [ROUTE["instructions"]] * 2]
    assert g["dist_hybrid"]["served_reads"][3][0] == "state_first"   # (the chon style lists them once)
    # uncap_logits: the capped read differs, the uncapped read is the read of a model without a cap
    for capped, uncapped, plain in (("dec_header_capped", "dec_header_uncapped", "dec_header_multi"),
                                    ("chon_state_first_capped", "chon_state_first_uncapped", "state_first_default")):
        assert g[uncapped]["answers"] == g[plain]["answers"] and g[uncapped]["reads"] == g[plain]["reads"]
        assert g[capped]["answers"] != g[plain]["answers"] and g[capped]["reads"] == g[plain]["reads"]
    assert g["dec_reread_header_uncapped"]["answers"] == g["dec_reread_header_avg"]["answers"]


def test_decider_prompts_read_as_documented(tiny_dm):
    from chon.prompt import DEC_AGAIN, DEC_ANSWER, DEC_HEADER, DEC_NOUL_TEXT, HEADER_LINE

    sf = [_decoded(tiny_dm, "dec_state_first", row=r) for r in range(4)]
    assert all(x.endswith("<start_of_turn>model\n" + DEC_ANSWER) for x in sf)
    assert "Context:\n" + TEXT_STATE + "\n\nQuestion: Which action" in sf[0] and DEC_HEADER not in sf[0]
    assert "Options:\n(A) refund: the customer wants to refund\n(B) replace:" in sf[0]
    assert "(A) no: weight > limit\n(B) yes: weight <= limit<end_of_turn>" in sf[1]        # yes/no as no, yes
    assert "Question: " + SEVERE["instructions"] + "\nOptions:\n(A) 0: low" in sf[2]       # instructions as written
    assert 'Question: {"task": "Pick  the  team", "note": "billing\\nor shipping"}\nOptions:\n(A) billing\n' in sf[3]
    js = _decoded(tiny_dm, "dec_state_first_json_array")      # a config's answer_prefix gives way to "Answer: ("
    assert js.endswith("model\n" + DEC_ANSWER) and '{"_index": 8, "sku": "B8", "qty": 2}' in js
    assert '"scores": [{"_index": 0, "value": 0}' in js and '"seven": ["x", "x", "x", "x", "x", "x", "x"]' in js
    assert '"rows": [{"_index": 0, "value": [0, 1]}' in js and "Mức độ 3" in js
    assert '"content": "Mức độ 7, the box arrived damaged."' in _decoded(tiny_dm, "dec_state_first_conv")
    hm = _decoded(tiny_dm, "dec_header_multi", row=1)
    assert hm.startswith("<bos><start_of_turn>user\n" + DEC_HEADER + "\n- Which action does the ticket request?\n"
                         "- Is the parcel within the limit?\n- " + SEVERE["instructions"] + "\n- {")
    assert hm.index("\n\nContext:\n" + TEXT_STATE) < hm.index("Question: Is the parcel") and HEADER_LINE not in hm
    same = _decoded(tiny_dm, "dec_header_same_instructions")
    assert same.count("- Which action does the ticket request?") == 2
    assert DEC_HEADER in _decoded(tiny_dm, "dec_hybrid_same_instructions")
    assert DEC_HEADER not in _decoded(tiny_dm, "dec_hybrid_one")
    tw = _decoded(tiny_dm, "dec_reread_header_avg", call=1)   # twice_header: the whole user turn, the line, again
    first = tw.split(DEC_AGAIN)[0].split("user\n", 1)[1]
    assert tw.count(DEC_HEADER) == 2 and tw.count(DEC_AGAIN) == 1 and tw.endswith(first + "<end_of_turn>\n"
                                                                                  "<start_of_turn>model\n" + DEC_ANSWER)
    tw1 = _decoded(tiny_dm, "dec_reread_hybrid_one", call=1)
    assert DEC_HEADER not in tw1 and tw1.count("Context:") == 2 and tw1.count(DEC_AGAIN) == 1
    bare = tiny_dm.tok.decode(GOLDEN["cases"]["dec_dist_header"]["reads"][0]["ids"][4])
    assert "- " + DEC_NOUL_TEXT in bare and "Question: " + DEC_NOUL_TEXT + "\nOptions:\n(A) no\n(B) yes<end" in bare
    yn = tiny_dm.tok.decode(GOLDEN["cases"]["dec_dist_header"]["reads"][0]["ids"][5])   # an explicit order stands
    assert "(A) yes: weight <= limit\n(B) no:" in yn


def test_no_style_keys_read_as_before(tiny_dm):
    """A config without prompt_style / uncap_logits: the chon style, no answer prefix, the model's cap untouched."""
    from chon.infer import DecisionModel
    from parity_cases import _softcap_configs

    caps = _softcap_configs(tiny_dm.model)
    for c in caps:
        c.final_logit_softcapping = CAP
    try:
        dm = DecisionModel(tiny_dm.model, tiny_dm.tok, {"prompt_layout": "header"}, device="cpu")
        assert dm.style == "chon" and not dm.uncapped and dm.builder.answer_prefix == ""
        assert all(c.final_logit_softcapping == CAP for c in caps)
        assert dm.builder.natural_order(WITHIN) == ["yes", "no"]
        assert dm.builder.asked_list([ROUTE, ROUTE, WITHIN]) == [ROUTE["instructions"], WITHIN["instructions"]]
        dm = DecisionModel(tiny_dm.model, tiny_dm.tok, {**DEC, "uncap_logits": True}, device="cpu")
        assert all(c.final_logit_softcapping is None for c in caps)
        assert dm.builder.natural_order(WITHIN) == ["no", "yes"]
        assert dm.builder.asked_list([ROUTE, ROUTE]) == [ROUTE["instructions"]] * 2
    finally:
        for c in caps:
            c.final_logit_softcapping = None


def test_uncap_logits_helper(tiny_dm):
    from chon.modeling import uncap_logits
    from parity_cases import _softcap_configs

    caps = _softcap_configs(tiny_dm.model)
    assert caps and not uncap_logits(tiny_dm.model)    # the tiny model has no cap: nothing to turn off
    for c in caps:
        c.final_logit_softcapping = 30.0
    try:
        assert uncap_logits(tiny_dm.model) and all(c.final_logit_softcapping is None for c in caps)
    finally:
        for c in caps:
            c.final_logit_softcapping = None


def test_unknown_style_and_sandwich_are_refused(tiny_dm, monkeypatch):
    from chon.infer import DecisionModel
    from chon.prompt import PromptBuilder

    monkeypatch.delenv("CHON_REREAD", raising=False)
    for bad in ("flexdm", "Decider"):
        with pytest.raises(ValueError, match="unknown prompt_style"):
            DecisionModel(tiny_dm.model, tiny_dm.tok, {"prompt_style": bad}, device="cpu")
        with pytest.raises(ValueError, match="unknown prompt_style"):
            PromptBuilder(tiny_dm.tok, style=bad)
    with pytest.raises(ValueError, match="no 'sandwich' second read"):
        DecisionModel(tiny_dm.model, tiny_dm.tok, {**DEC, "reread": {**RR_ALL, "layout": "sandwich"}}, device="cpu")
    monkeypatch.setenv("CHON_REREAD", "0.9,sandwich")
    with pytest.raises(ValueError, match="no 'sandwich' second read"):
        DecisionModel(tiny_dm.model, tiny_dm.tok, dict(DEC), device="cpu")
    assert DecisionModel(tiny_dm.model, tiny_dm.tok, {}, device="cpu").reread["layout"] == "sandwich"
    with pytest.raises(ValueError, match="no 'sandwich' layout"):
        PromptBuilder(tiny_dm.tok, style="decider").encode(TEXT_STATE, ROUTE, layout="sandwich")


def test_decider_settings_load_from_the_model_folder(tiny_model_dir, tmp_path, monkeypatch):
    """A model folder whose config.json has a final soft cap and whose chon_config.json sets prompt_style "decider" and
    uncap_logits reads as the golden uncapped reads; without uncap_logits, as the golden capped reads."""
    from chon.infer import DecisionModel

    monkeypatch.delenv("CHON_REREAD", raising=False)
    d = tmp_path / "pkg"
    shutil.copytree(tiny_model_dir, d)
    mc = json.loads((d / "config.json").read_text())
    mc["text_config"]["final_logit_softcapping"] = CAP
    (d / "config.json").write_text(json.dumps(mc))
    for name, expect_cap in (("dec_header_uncapped", None), ("dec_header_capped", CAP)):
        case = BY_NAME[name]
        (d / "chon_config.json").write_text(json.dumps(case["cfg"]))
        dm = DecisionModel.load(str(d), device="cpu", dtype="float32")
        assert dm.style == "decider" and dm.layout == "header" and dm.uncapped == (expect_cap is None)
        assert dm.model.config.get_text_config().final_logit_softcapping == expect_cap
        seen = []
        real = dm.raw_logits
        dm.raw_logits = lambda seqs, n, letter_ids=None: (seen.append([list(s) for s in seqs]), real(seqs, n))[1]
        dm.cfg["prefix_cache"] = False
        answers, _ = dm.predict(case["state"], case["questions"])
        want = GOLDEN["cases"][name]
        assert seen == [r["ids"] for r in want["reads"]]
        _same_outputs(answers, want["answers"], name)


def test_dindex_and_health_state_the_style(tiny_model_dir, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from chon.dindex import ChonEngine
    from chon.serve import create_app

    monkeypatch.delenv("CHON_REREAD", raising=False)
    d = tmp_path / "pkg"
    shutil.copytree(tiny_model_dir, d)
    (d / "chon_config.json").write_text(json.dumps({"prompt_style": "decider", "uncap_logits": True,
                                                    "prompt_layout": "hybrid_header"}))
    e = ChonEngine(model=str(d), device="cpu")
    p = e.provenance
    assert p["prompt_style"] == "decider" and p["uncap_logits"] is True
    assert "'decider'" in p["policy"] and "soft cap off" in p["policy"] and "distinct" not in p["policy"]
    h = TestClient(create_app(e.dm, None)).get("/health").json()
    assert h["prompt_style"] == "decider" and h["uncap_logits"] is True
    (d / "chon_config.json").write_text("{}")
    p = ChonEngine(model=str(d), device="cpu").provenance
    assert p["prompt_style"] == "chon" and p["uncap_logits"] is False
    assert "decider" not in p["policy"] and "soft cap" not in p["policy"]
