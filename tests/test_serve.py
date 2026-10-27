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
