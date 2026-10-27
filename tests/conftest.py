"""Fixtures: a tiny random Gemma 4 with a tiny BPE tokenizer, built on the CPU (tests/tiny_model.py)."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # tiny_model, parity_cases


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory) -> str:
    from tiny_model import build_tiny_model

    return build_tiny_model(str(tmp_path_factory.mktemp("tiny-gemma4")))


@pytest.fixture(scope="session")
def tiny_dm(tiny_model_dir):
    """The tiny model loaded once (no chon_config.json: state-first, one read)."""
    from chon.infer import DecisionModel

    return DecisionModel.load(tiny_model_dir, device="cpu", dtype="float32")
