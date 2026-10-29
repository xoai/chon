"""Wire-format checks with a stand-in model (no GPU, no download)."""
from fastapi.testclient import TestClient

from chon.answers import format_answer
from chon.serve import create_app


class FakeModel:
    name = "Chon-test"

    def predict(self, state, questions):
        answers = {}
        for name, q in questions.items():
            if q["type"] == "noul":
                probs = {"yes": 0.9, "no": 0.1}
            elif q["type"] == "score":
                probs = {str(i): 1 / len(q["criteria"]) for i in range(len(q["criteria"]))}
            else:
                labels = list(q["criteria"])
                probs = {lab: 1 / len(labels) for lab in labels}
            answers[name] = format_answer(q, probs)
        return answers, {"input_tokens": 10, "output_tokens": 0}


def test_systemone_shapes():
    c = TestClient(create_app(FakeModel(), None))
    assert c.get("/health").json()["ok"]
    body = {"state": "s", "questions": {
        "a": {"type": "choice", "instructions": "pick", "criteria": {"x": "X", "y": "Y"}},
        "b": {"type": "noul", "instructions": "is it?"},
        "c": {"type": "score", "instructions": "rate", "criteria": ["low", "mid", "high"]}}}
    r = c.post("/v1/systemone", json=body).json()
    assert r["answers"]["a"]["type"] == "choice" and set(r["answers"]["a"]["probabilities"]) == {"x", "y"}
    assert abs(r["answers"]["b"]["noul"] - 0.9) < 1e-9
    assert r["answers"]["c"]["type"] == "score" and "latency_s" in r["usage"]
    assert c.post("/v1/systemone", json={"state": "s"}).status_code == 400


def test_api_key():
    c = TestClient(create_app(FakeModel(), "k"))
    body = {"state": "s", "questions": {"b": {"type": "noul", "instructions": "?"}}}
    assert c.post("/v1/systemone", json=body).status_code == 401
    assert c.post("/v1/systemone", json=body, headers={"Authorization": "Bearer k"}).status_code == 200


def test_too_long_is_413_with_the_capacity_marker():
    from chon.schema import CapacityError

    class TooLong(FakeModel):
        def predict(self, state, questions):
            raise CapacityError("the state is longer than 32768 tokens: maximum context length exceeded")

    r = TestClient(create_app(TooLong(), None)).post(
        "/v1/systemone", json={"state": "s", "questions": {"a": {"type": "noul", "instructions": "?"}}})
    assert r.status_code == 413 and "maximum context length" in r.json()["error"]


def test_request_forms():
    import pytest

    from chon.schema import ItemError, instructions_text, validate_item

    q = {"type": "choice", "instructions": {"task": "pick"}, "criteria": {f"o{i}": None for i in range(40)}}
    assert instructions_text(q) == '{"task":"pick"}'
    with pytest.raises(ItemError):  # over 16 options without the grouped readout
        validate_item({"id": "x", "state": "s", "question": q}, require_expected=False)
    it = validate_item({"id": "x", "state": "", "question": q}, require_expected=False, allow_empty_state=True,
                       max_options=256)
    assert it["state"] == "" and len(it["labels"]) == 40


def test_noul_probabilities_and_hidden_keys():
    from chon.answers import format_answer
    from chon.prompt import question_text

    a = format_answer({"type": "noul", "instructions": "x"}, {"yes": 0.25, "no": 0.75})
    assert a["noul"] == 0.25 and a["probabilities"] == {"true": 0.25, "false": 0.75}
    q = {"type": "choice", "instructions": "Which?", "criteria": {"0": "refund", "1": "replace"}, "option_keys": "hide"}
    assert "A) refund" in question_text(q, ["0", "1"]) and "0 —" not in question_text(q, ["0", "1"])


def test_deadline_is_passed_from_arrival():
    import time

    class Timed(FakeModel):
        got = "unset"

        def predict(self, state, questions, deadline=None):
            Timed.got = deadline
            return super().predict(state, questions)
    t = time.monotonic()
    c = TestClient(create_app(Timed(), None, deadline_s=8.0))
    body = {"state": "s", "questions": {"b": {"type": "noul", "instructions": "is it?"}}}
    assert c.post("/v1/systemone", json=body).status_code == 200
    assert t + 7.5 < Timed.got <= time.monotonic() + 8.0 and c.get("/health").json()["deadline_s"] == 8.0


def _thinker(spt=0.01):
    """A DecisionModel with only what _apply_think reads (no weights): every answer counts as unsure."""
    from collections import deque

    from chon.infer import DecisionModel
    m = object.__new__(DecisionModel)
    m.think = {"tau": {"choice": 1.01, "noul": 1.01, "score": 1.01}, "budget": 512, "mix": True,
               "temperatures": {"choice": 1.0, "noul": 1.0, "score": 1.0}}
    m._recent, m._read_s, m._gen_spt, m._deadline = deque(maxlen=10), 0.01, spt, None
    m.deadline_stats = {"requests": 0, "skipped": 0, "shortened": 0, "cut": 0}
    return m


def test_deadline_shortens_skips_and_cuts_the_reasoning():
    import time

    from chon.infer import DeadlineExceeded
    seen = {}

    def think(seqs, n_options, budget=None, max_time=None):
        seen.update(budget=budget, max_time=max_time)
        return [[0.0, -5.0] for _ in seqs], [budget for _ in seqs]
    m = _thinker(spt=0.01)
    m.think_logprobs = think
    probs = [[0.5, 0.5]]
    m._deadline = time.monotonic() + 2.0      # ~1.5 s of reasoning at 100 tokens/s: 512 shortened to ~150
    assert m._apply_think(["noul"], [["yes", "no"]], [[1, 2]], [[0.0, 0.0]], probs) == [seen["budget"]]
    assert 48 <= seen["budget"] < 512 and m.deadline_stats["shortened"] == 1 and probs[0][0] > 0.5
    probs = [[0.5, 0.5]]
    m._deadline = time.monotonic() + 0.1      # no time left: the one-pass answer stands
    assert m._apply_think(["noul"], [["yes", "no"]], [[1, 2]], [[0.0, 0.0]], probs) == [None]
    assert probs == [[0.5, 0.5]] and m.deadline_stats["skipped"] == 1

    def cut(seqs, n_options, budget=None, max_time=None):
        raise DeadlineExceeded("cut", [30])
    m.think_logprobs = cut
    m._deadline = time.monotonic() + 3.0
    assert m._apply_think(["noul"], [["yes", "no"]], [[1, 2]], [[0.0, 0.0]], probs) == [30]   # tokens paid for
    assert probs == [[0.5, 0.5]] and m.deadline_stats["cut"] == 1


# ---- --noul-commit (1.6.0): the yes/no commit band set at start-up -------------------------------------------------

def _serve(tiny_model_dir, tmp_path, monkeypatch, capsys, cfg, *flags):
    """chon.serve.main on a copy of the tiny model with this chon_config.json: (a test client, the start line)."""
    import json
    import shutil

    import uvicorn

    from chon.serve import main

    d = tmp_path / f"m{len(list(tmp_path.iterdir()))}"
    shutil.copytree(tiny_model_dir, d)
    (d / "chon_config.json").write_text(json.dumps(cfg))
    apps = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: apps.append(app))
    capsys.readouterr()
    main(["--model", str(d), "--device", "cpu", "--dtype", "float32", "--deadline", "0", *flags])
    start = next(x for x in capsys.readouterr().out.splitlines() if x.startswith("[serve] ") and " on cpu" in x)
    return TestClient(apps[0]), start


def test_noul_commit_flag_serves_as_the_config_key(tiny_model_dir, tmp_path, monkeypatch, capsys):
    """--noul-commit LO,HI changes a served yes/no answer exactly as chon_config.json "noul_commit" does, and
    --noul-commit off removes a config's band; /health and the start line say which band is served."""
    import pytest

    from chon.dindex import ChonEngine

    monkeypatch.delenv("CHON_REREAD", raising=False)
    temps = {"temperatures": {"choice": 0.9, "noul": 1.2, "score": 0.7}}
    body = {"state": "Parcel 7: weight 5 kg, limit 4 kg. The customer asks for a refund and is upset.",
            "questions": {"within": {"type": "noul", "instructions": "Is the parcel within the limit?",
                                     "criteria": {"true": "weight <= limit", "false": "weight > limit"}}}}

    def served(cfg, *flags):
        c, start = _serve(tiny_model_dir, tmp_path, monkeypatch, capsys, cfg, *flags)
        return c.post("/v1/systemone", json=body).json()["answers"]["within"], c.get("/health").json(), start

    plain, h_plain, s_plain = served(temps)
    assert 0.2 < plain["noul"] < 0.8 and h_plain["noul_commit"] is None and s_plain.endswith(", no yes/no commit band")
    by_cfg, h_cfg, s_cfg = served({**temps, "noul_commit": [0.199, 0.801]})
    by_flag, h_flag, s_flag = served(temps, "--noul-commit", "0.199,0.801")
    assert by_flag == by_cfg and by_flag["noul"] == (0.801 if plain["noul"] >= 0.5 else 0.199)
    assert h_flag["noul_commit"] == h_cfg["noul_commit"] == [0.199, 0.801]
    assert s_flag.endswith(", yes/no commit band 0.199/0.801") and s_cfg == s_flag
    wide, _, _ = served({**temps, "noul_commit": [0.199, 0.801]}, "--noul-commit", "0.1,0.9")   # the flag wins
    assert wide["noul"] == (0.9 if plain["noul"] >= 0.5 else 0.1)
    off, h_off, s_off = served({**temps, "noul_commit": [0.199, 0.801]}, "--noul-commit", "off")
    assert off == plain and h_off["noul_commit"] is None and s_off.endswith(", no yes/no commit band")
    for bad in ("0.6,0.9", "0.1,0.4", "0,0.8", "0.2,1", "0.2", "x,y"):
        with pytest.raises(SystemExit):
            served(temps, "--noul-commit", bad)
    # the Decision Index engine: --option noul_commit=LO,HI the same; true keeps the config's; off by default
    d = tmp_path / "engine"
    import json
    import shutil
    shutil.copytree(tiny_model_dir, d)
    (d / "chon_config.json").write_text(json.dumps(temps))
    state, qs = body["state"], body["questions"]
    e = ChonEngine(model=str(d), device="cpu")      # (the engine reads in bfloat16: its own numbers)
    e_plain = e(state, qs)[0]["answers"]["within"]
    assert 0.2 < e_plain["noul"] < 0.8 and e.provenance["noul_commit_band"] is None
    assert "commit band" not in e.provenance["policy"]
    e = ChonEngine(model=str(d), device="cpu", noul_commit="0.199,0.801")
    e_band = e(state, qs)[0]["answers"]["within"]
    assert e_band["noul"] == (0.801 if e_plain["noul"] >= 0.5 else 0.199) and e_band["noul"] == by_flag["noul"]
    assert e.provenance["noul_commit_band"] == [0.199, 0.801] and e.provenance["noul_commit"] is True
    assert "served as 0.199 or 0.801" in e.provenance["policy"]
    (d / "chon_config.json").write_text(json.dumps({**temps, "noul_commit": [0.199, 0.801]}))
    assert ChonEngine(model=str(d), device="cpu", noul_commit="true")(state, qs)[0]["answers"]["within"] == e_band
    assert ChonEngine(model=str(d), device="cpu", noul_commit="false")(state, qs)[0]["answers"]["within"] == e_plain
    assert ChonEngine(model=str(d), device="cpu")(state, qs)[0]["answers"]["within"] == e_plain   # board default
    with pytest.raises(ValueError, match="needs 0 < LO < 0.5 < HI < 1"):
        ChonEngine(model=str(d), device="cpu", noul_commit="0.3,0.4")


def test_parse_noul_commit():
    import pytest

    from chon.infer import parse_noul_commit

    assert parse_noul_commit("0.199,0.801") == (0.199, 0.801) == parse_noul_commit([0.199, 0.801])
    assert parse_noul_commit(" 0.1 , 0.9 ") == (0.1, 0.9)
    for off in ("off", "OFF", "none", "false", "0", ""):
        assert parse_noul_commit(off) is None
    for bad in ("0.5,0.8", "0.2,0.5", "-0.1,0.8", "0.2,1.0", "0.2", "0.1,0.5,0.9", "a,b", 0.2, None):
        with pytest.raises(ValueError):
            parse_noul_commit(bad)
