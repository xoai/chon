"""vLLM as the reader (1.7.0): letter log-probabilities from a vLLM server behind DecisionModel, the merged checkpoint
served without the soft cap for an uncapped model, and the engine's and server's options."""
import json
import math
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _q(i=0):
    return {"type": "choice", "instructions": f"Which tier fits case {i}?",
            "criteria": {"low": "small", "mid": "medium", "high": "large"}}


def test_reader_parses_letter_logprobs_and_sends_the_mask():
    from chon.vllm_reader import NEG, VLLMReader
    seen = []

    def post(path, body):
        seen.append((path, body))
        return {"choices": [{"logprobs": {"top_logprobs": [{"token_id:7": -0.2, "token_id:5": -1.7}]}}]}

    r = VLLMReader("http://x", post=post)
    assert r.one([1, 2, 3], [5, 7, 9]) == [-1.7, -0.2, NEG]
    path, body = seen[0]
    assert path == "/v1/completions" and body["max_tokens"] == 1 and body["allowed_token_ids"] == [5, 7, 9]
    assert body["temperature"] == 0.0 and body["top_p"] == 1.0 and body["top_k"] == -1 and body["min_p"] == 0.0
    out = r.letter_logprobs([[1], [2]], [2, 3], [5, 7, 9, 11])
    assert [len(x) for x in out] == [2, 3]
    with pytest.raises(RuntimeError, match="processed_logprobs"):
        r.check([1], [5, 7, 9])
    ok = VLLMReader("http://x", post=lambda p, b: {"choices": [{"logprobs": {"top_logprobs": [
        {"token_id:5": math.log(0.25), "token_id:7": math.log(0.75)}]}}]})
    ok.check([1], [5, 7])


class HFAsServer:
    """A stand-in vLLM backend that reads the letters with the tiny HF model itself (so the answers must match)."""

    def __init__(self, dm):
        self.dm, self.calls = dm, 0

    def letter_logprobs(self, seqs, n_options, letter_ids):
        self.calls += len(seqs)
        backend, self.dm.backend = self.dm.backend, None
        try:
            lg = self.dm.raw_logits(seqs, n_options, letter_ids)
        finally:
            self.dm.backend = backend
        return [[x - max(r) - math.log(sum(math.exp(y - max(r)) for y in r)) for x in r] for r in lg]

    def check(self, ids, letters):
        pass


def test_a_backend_gives_the_hf_readers_answers(tiny_model_dir):
    from chon.infer import DecisionModel
    hf = DecisionModel.load(tiny_model_dir, device="cpu", dtype="float32")
    ref, _ = hf.predict("A shared state about three cases.", {"a": _q(1), "b": _q(2)})
    dm = DecisionModel(hf.model, hf.tok, dict(hf.cfg), device="cpu", backend="placeholder")
    assert dm._prefix_off and not dm.prefix_cache            # vLLM shares prefixes itself
    dm.backend = HFAsServer(dm)
    got, usage = dm.predict("A shared state about three cases.", {"a": _q(1), "b": _q(2)})
    for k in ref:
        for lab, p in ref[k]["probabilities"].items():
            assert abs(got[k]["probabilities"][lab] - p) < 1e-5
    assert dm.backend.calls >= 2


def test_load_with_the_vllm_reader(tiny_model_dir, monkeypatch, tmp_path):
    from chon import infer, vllm_reader
    started = {}

    class Fake:
        def letter_logprobs(self, seqs, n_options, letter_ids):
            return [[math.log(1.0 / n)] * n for n in n_options]

        def check(self, ids, letters):
            started["checked"] = len(letters)

    monkeypatch.setattr(vllm_reader, "start", lambda served, cfg=None: started.setdefault("served", served) and Fake())
    monkeypatch.delenv("CHON_VLLM_QUANT", raising=False)
    dm = infer.DecisionModel.load(tiny_model_dir, device="cpu", dtype="float32", reader="vllm", vllm_quant="fp8")
    assert dm.model is None and dm.backend is not None and started["served"] == tiny_model_dir
    assert started["checked"] == 2 and os.environ["CHON_VLLM_QUANT"] == "fp8"
    ans, _ = dm.predict("state", {"q": _q()})
    assert abs(sum(ans["q"]["probabilities"].values()) - 1) < 1e-6
    # an uncapped model is served from a copy of its config without the soft cap
    up = tmp_path / "uncapped"
    up.mkdir()
    for f in os.listdir(tiny_model_dir):
        src = os.path.join(tiny_model_dir, f)
        if os.path.isfile(src):
            (up / f).write_bytes(open(src, "rb").read())
    cc = json.load(open(up / "chon_config.json")) if (up / "chon_config.json").exists() else {}
    json.dump({**cc, "uncap_logits": True}, open(up / "chon_config.json", "w"))
    started.clear()
    infer.DecisionModel.load(str(up), device="cpu", dtype="float32", reader="vllm")
    assert started["served"] == str(up) + "-uncap"
    monkeypatch.setenv("CHON_READER", "onnx")
    with pytest.raises(ValueError, match="unknown reader"):
        infer.DecisionModel.load(tiny_model_dir, device="cpu", dtype="float32")


def test_uncapped_copy(tmp_path):
    from chon.vllm_reader import uncapped
    m = tmp_path / "merged"
    m.mkdir()
    json.dump({"text_config": {"final_logit_softcapping": 30.0, "hidden_size": 8}, "final_logit_softcapping": 30.0},
              open(m / "config.json", "w"))
    (m / "model.safetensors").write_text("weights")
    out = uncapped(str(m))
    cfg = json.load(open(os.path.join(out, "config.json")))
    assert cfg["final_logit_softcapping"] is None and cfg["text_config"]["final_logit_softcapping"] is None
    assert os.path.islink(os.path.join(out, "model.safetensors")) and uncapped(str(m)) == out
    assert json.load(open(m / "config.json"))["final_logit_softcapping"] == 30.0


def test_engine_and_server_take_the_reader_options():
    src = open(os.path.join(ROOT, "chon", "dindex.py")).read()
    assert "reader=None, vllm_quant=None" in src and "reader=reader or None" in src and '"reader": "vllm"' in src
    serve = open(os.path.join(ROOT, "chon", "serve.py")).read()
    assert '"--reader"' in serve and '"--vllm-quant"' in serve and "vllm_quant=args.vllm_quant" in serve
    reader = open(os.path.join(ROOT, "chon", "vllm_reader.py")).read()
    assert '"-m", "chon.merge"' in reader and "from chon import vllm_sidecar" in reader
