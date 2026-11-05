"""1.8.0: the two settings of models trained from 5 Nov (chon_config.json, prompt_version 3).

- max_options (16 to 255): up to that many options read in one pass, coded A-Z and then the two-letter codes AA, AB, ...
  that are one token at the reply boundary; beyond the codes, in groups and a final round as before;
- twice_max_chars: in the header layout a request (its question list and state) of at most that many characters is
  written twice, then the question asked; the questions about one state still share the doubled request as a prefix.

A model without them reads exactly as before (tests/test_parity.py; the read-twice reads are golden cases there too,
tests/parity_cases.py TWICE_CASES, made with the training code)."""
from __future__ import annotations

import json
import math
import os
import shutil

import pytest

from chon.prompt import LETTERS, PromptBuilder


@pytest.fixture(scope="module")
def wide_dir(tmp_path_factory):
    """A tiny Gemma whose vocabulary holds A-Z and AA, AB, ... as single tokens, as Gemma 4's does."""
    from tiny_model import build_tiny_model

    return build_tiny_model(str(tmp_path_factory.mktemp("wide")), wide_labels=True)


def _tok(d):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(d)


def _choice(n, instr="Which intent is this?"):
    return {"type": "choice", "instructions": instr, "criteria": {f"intent_{i}": f"intent number {i}" for i in range(n)}}


def _package(src, dst, **cfg):
    shutil.copytree(src, dst)
    json.dump({"prompt_layout": "header", "prompt_style": "decider", "prompt_version": 3, **cfg},
              open(os.path.join(dst, "chon_config.json"), "w"))
    return str(dst)


def test_codes_extend_the_letters_and_old_models_keep_them(wide_dir):
    tok = _tok(wide_dir)
    old = PromptBuilder(tok, 512, style="decider")
    assert old.codes == list(LETTERS) and len(old.letter_ids) == 16
    new = PromptBuilder(tok, 512, style="decider", max_options=255)
    assert len(new.codes) == len(new.letter_ids) == 255 and len(set(new.letter_ids)) == 255
    assert new.codes[:16] == list(LETTERS) and new.letter_ids[:16] == old.letter_ids   # the same letters first
    assert new.codes[25] == "Z" and new.codes[26] == "AA"
    text = new.render("A bank transfer did not arrive.", _choice(77), list(_choice(77)["criteria"]))
    assert "(Z) intent_25: intent number 25" in text and "(AA) intent_26" in text and "intent_76" in text
    n = len(tok(text, add_special_tokens=False)["input_ids"])
    for k, code in enumerate(new.codes):          # every code is one token right after the prompt
        ids = tok(text + code, add_special_tokens=False)["input_ids"]
        assert ids[-1] == new.letter_ids[k] and len(ids) == n + 1
    chon = PromptBuilder(tok, 512, max_options=40)                  # the chon style: "AA) ..." lines
    assert "\nAA) intent_26 — intent number 26\n" in chon.render("x", _choice(30), list(_choice(30)["criteria"]))
    with pytest.raises(ValueError, match="max_options"):
        PromptBuilder(tok, 512, max_options=300)


def test_too_few_codes_in_the_vocabulary_is_an_error(tiny_model_dir):
    with pytest.raises(ValueError, match="option codes are one token"):
        PromptBuilder(_tok(tiny_model_dir), 512, style="decider", max_options=255)


def test_a_model_with_codes_reads_many_options_in_one_pass(wide_dir, tmp_path):
    from chon.infer import DecisionModel

    dm = DecisionModel.load(_package(wide_dir, tmp_path / "a", max_options=255), device="cpu", dtype="float32")
    calls = []
    dm._finish_many = lambda *a, **k: calls.append(1) or ({}, 0)
    seen = []
    real = dm.raw_logits
    dm.raw_logits = lambda seqs, n, letter_ids=None: seen.extend(n) or real(seqs, n, letter_ids)
    ans, _ = dm.predict("Card payment was declined twice.",
                        {"intent": _choice(77), "yn": {"type": "noul", "instructions": "Urgent?"}})
    assert not calls and len(ans["intent"]["probabilities"]) == 77 and 77 in seen
    assert abs(sum(ans["intent"]["probabilities"].values()) - 1) < 1e-5
    # 300 options is more than the codes: groups of up to 255 and a final round
    dm2 = DecisionModel.load(_package(wide_dir, tmp_path / "b", max_options=255), device="cpu", dtype="float32")
    ans2, _ = dm2.predict("x", {"q": _choice(300)})
    assert len(ans2["q"]["probabilities"]) == 300
    # a model of 16 letters reads 77 options in groups, as before
    old = DecisionModel.load(_package(wide_dir, tmp_path / "c", prompt_version=2), device="cpu", dtype="float32")
    groups = []
    real_many = old._finish_many
    old._finish_many = lambda *a, **k: groups.append(1) or real_many(*a, **k)
    ans3, _ = old.predict("Card payment was declined twice.", {"intent": _choice(77)})
    assert groups and len(ans3["intent"]["probabilities"]) == 77


def test_requests_written_twice_share_their_prefix(tiny_model_dir):
    tok = _tok(tiny_model_dir)
    for style in ("decider", "chon"):
        b = PromptBuilder(tok, 512, style=style, twice_max_chars=400)
        plain = PromptBuilder(tok, 512, style=style)
        asked = ["Is the order late?", "Was it refunded?"]
        q1 = {"type": "noul", "instructions": asked[0]}
        q2 = {"type": "noul", "instructions": asked[1]}
        short = "Order 1182 arrived damaged."
        t1 = b.render(short, q1, ["no", "yes"], layout="header", asked=asked)
        t2 = b.render(short, q2, ["no", "yes"], layout="header", asked=asked)
        assert t1.count(short) == 2 and t1.count(asked[1]) == 2        # the request (list and state) twice
        assert os.path.commonprefix([t1, t2]).count(short) == 2         # siblings share the doubled request
        assert t1 == b.render(short, q1, ["no", "yes"], layout="header_twice", asked=asked)
        assert plain.render(short, q1, ["no", "yes"], layout="header", asked=asked).count(short) == 1
        long = "Order 1182 arrived damaged. " * 30                        # over 400 characters: once
        assert b.render(long, q1, ["no", "yes"], layout="header", asked=asked).count("Order 1182") == 30
        assert b.render(short, q1, ["no", "yes"], layout="state_first").count(short) == 1   # header layout only
        # the second read's own layout is unchanged
        assert b.render(short, q1, ["no", "yes"], layout="twice_header", asked=asked) == \
            plain.render(short, q1, ["no", "yes"], layout="twice_header", asked=asked)


def test_prompts_written_twice_read_as_documented(tiny_model_dir):
    from chon.prompt import DEC_AGAIN, DEC_HEADER, HEADER_LINE, REREAD_LINE

    tok = _tok(tiny_model_dir)
    q = {"type": "noul", "instructions": "Is it late?"}
    dec = PromptBuilder(tok, 512, style="decider", twice_max_chars=400).render("S1", q, ["no", "yes"], layout="header",
                                                                                 asked=["Is it late?"])
    req = f"{DEC_HEADER}\n- Is it late?\n\nContext:\nS1"
    assert f"{req}{DEC_AGAIN}{req}\n\nQuestion: Is it late?\nOptions:\n(A) no\n(B) yes" in dec
    chon = PromptBuilder(tok, 512, twice_max_chars=400).render("S1", q, ["no", "yes"], layout="header",
                                                               asked=["Is it late?"])
    top = f"{HEADER_LINE}\n- Is it late?\n---\nS1\n\n---\n"
    assert f"{top}{REREAD_LINE}\n\n{top}Question: Is it late?\nA) no\nB) yes\nAnswer with one letter." in chon


def test_newer_prompt_versions_are_refused(tiny_model_dir, tmp_path):
    from chon.infer import DecisionModel

    d = tmp_path / "p"
    shutil.copytree(tiny_model_dir, d)
    (d / "chon_config.json").write_text(json.dumps({"prompt_version": 4}))
    with pytest.raises(ValueError, match="newer version of the chon serving code"):
        DecisionModel.load(str(d), device="cpu", dtype="float32")


def test_the_vllm_reader_asks_for_every_code(wide_dir, tmp_path, monkeypatch):
    from chon import infer, vllm_reader, vllm_sidecar

    seen = {}
    monkeypatch.setattr(vllm_sidecar, "ensure", lambda model, side: seen.update(side) or "http://x")
    monkeypatch.delenv("CHON_VLLM_READER_URL", raising=False)
    vllm_reader.start("/m", {"max_logprobs": 255})
    a = seen["extra_args"]
    assert a[a.index("--max-logprobs") + 1] == "255"
    vllm_reader.start("/m", {})
    a = seen["extra_args"]
    assert a[a.index("--max-logprobs") + 1] == "20"
    # the reader asks the server for max_options log-probs and reads 77 options at once
    started = {}

    class Fake:
        def letter_logprobs(self, seqs, n_options, letter_ids):
            started.setdefault("n", []).extend(n_options)
            return [[math.log(1.0 / n)] * n for n in n_options]

        def check(self, ids, letters):
            pass

    monkeypatch.setattr(vllm_reader, "start", lambda served, cfg=None: started.update(cfg=cfg) or Fake())
    dm = infer.DecisionModel.load(_package(wide_dir, tmp_path / "v", max_options=255), device="cpu", dtype="float32",
                                  reader="vllm")
    assert started["cfg"]["max_logprobs"] == 255
    ans, _ = dm.predict("x", {"q": _choice(77)})
    assert 77 in started["n"] and len(ans["q"]["probabilities"]) == 77
