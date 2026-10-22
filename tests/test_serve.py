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
