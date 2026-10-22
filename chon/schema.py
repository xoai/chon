"""Decision requests: a state plus one typed question (TypeSafe /v1/systemone conventions).

    {"state": str | dict | list,
     "question": {"type": "choice" | "noul" | "score", "instructions": str, "criteria": ...}}

Criteria shapes accepted (TypeSafe /v1/systemone conventions):
  choice: {label: description | None}  or  [label, ...]
  noul:   {"true": description, "false": description}  or  None
  score:  [level_0_description, level_1_description, ...]  or  {"0": desc, "1": desc, ...}
"""

from __future__ import annotations

import json
import math
from typing import Any, Iterable

QTYPES = ("choice", "noul", "score")
NOUL_LABELS = ("yes", "no")  # option order used at inference: A = yes/true, B = no/false
MAX_OPTIONS = 16
TIERS = ("easy", "standard", "judge", "hard")
LANGS = ("en", "vi")


class ItemError(ValueError):
    pass


class CapacityError(ItemError):
    """A request the model cannot take whole (no truncation allowed): the message carries the standard capacity
    marker ("maximum context length") that benchmark runners map to "unsupported"."""


def options_for(question: dict) -> list[tuple[str, str | None]]:
    """Return the options of a question as (label, description) in natural order.

    natural order = criteria order for choice, yes/no for noul, level 0..n-1 for score.
    """
    qtype = question.get("type")
    crit = question.get("criteria")
    if qtype == "noul":
        d = crit if isinstance(crit, dict) else {}
        return [("yes", _desc(d.get("true"))), ("no", _desc(d.get("false")))]
    if qtype == "choice":
        if isinstance(crit, dict):
            opts = [(str(k), _desc(v)) for k, v in crit.items()]
        elif isinstance(crit, list):
            opts = [(str(x), None) for x in crit]
        else:
            raise ItemError("choice criteria must be a dict {label: description} or a list of labels")
        _check_unique([o[0] for o in opts])
        return opts
    if qtype == "score":
        if isinstance(crit, list):
            return [(str(i), _desc(d)) for i, d in enumerate(crit)]
        if isinstance(crit, dict):
            try:
                keys = sorted(crit, key=lambda k: int(k))
            except (TypeError, ValueError):
                raise ItemError("score criteria dict keys must be level integers") from None
            if [int(k) for k in keys] != list(range(len(keys))):
                raise ItemError("score levels must be 0..n-1")
            return [(str(int(k)), _desc(crit[k])) for k in keys]
        raise ItemError("score criteria must be a list of level descriptions")
    raise ItemError(f"unknown question type {qtype!r}")


def _desc(v: Any) -> str | None:
    if v is None:
        return None
    s = (v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, separators=(",", ":"))).strip()
    return s or None


def instructions_text(q: dict) -> str:
    """Instructions may be a string or a JSON object (rendered compactly)."""
    v = q.get("instructions")
    return (v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, separators=(",", ":"))).strip() \
        if v is not None else ""


def _check_unique(labels: list[str]) -> None:
    if len(set(labels)) != len(labels):
        raise ItemError("duplicate option labels")


def labels_for(question: dict) -> list[str]:
    return [lab for lab, _ in options_for(question)]


def normalize_expected(item: dict) -> str:
    exp = item.get("expected")
    if isinstance(exp, bool):
        return "yes" if exp else "no"
    if item["question"]["type"] == "noul" and isinstance(exp, str):
        low = exp.strip().lower()
        return {"true": "yes", "false": "no"}.get(low, low)
    return str(exp)


def validate_item(item: dict, require_expected: bool = True, allow_empty_state: bool = False,
                  max_options: int | None = None) -> dict:
    """Validate and normalise an item in place-safe fashion; returns a cleaned copy.

    Raises ItemError with a human-readable reason.
    """
    if not isinstance(item, dict):
        raise ItemError("item is not an object")
    it = dict(item)
    if not it.get("id"):
        raise ItemError("missing id")
    state = it.get("state")
    if allow_empty_state and (state is None or (isinstance(state, (str, list, dict)) and len(state) == 0)):
        state = it["state"] = ""
    elif state is None or (isinstance(state, (str, list, dict)) and len(state) == 0):
        raise ItemError("empty state")
    if not isinstance(state, (str, list, dict)):
        raise ItemError("state must be a string, object or list")
    q = it.get("question")
    if not isinstance(q, dict):
        raise ItemError("missing question")
    if q.get("type") not in QTYPES:
        raise ItemError(f"bad question type {q.get('type')!r}")
    if not instructions_text(q):
        raise ItemError("missing instructions")
    opts = options_for(q)
    cap = MAX_OPTIONS if max_options is None else max_options
    if not (2 <= len(opts) <= cap):
        raise ItemError(f"need 2..{cap} options, got {len(opts)}")
    labels = [lab for lab, _ in opts]
    if it.get("labels") is not None and set(map(str, it["labels"])) != set(labels):
        raise ItemError("labels do not match the criteria")
    it["labels"] = labels
    if require_expected or it.get("expected") is not None:
        exp = normalize_expected(it)
        if exp not in labels:
            raise ItemError(f"expected {exp!r} not in labels")
        it["expected"] = exp
    for key in ("target",):
        if it.get(key) is not None:
            it[key] = check_distribution(it[key], labels, name=key)
    if it.get("target") is not None and it.get("expected") is not None:
        tgt = it["target"]
        top = max(tgt.values())
        if tgt[it["expected"]] < top - 1e-9:
            raise ItemError("target argmax disagrees with expected")
    if it.get("tier") is not None and it["tier"] not in TIERS:
        raise ItemError(f"bad tier {it['tier']!r}")
    if it.get("lang") is not None and it["lang"] not in LANGS:
        raise ItemError(f"bad lang {it['lang']!r}")
    return it


def check_distribution(dist: Any, labels: Iterable[str], name: str = "distribution") -> dict[str, float]:
    labels = list(labels)
    if not isinstance(dist, dict):
        raise ItemError(f"{name} must be an object")
    d = {str(k): float(v) for k, v in dist.items()}
    if set(d) != set(labels):
        raise ItemError(f"{name} keys must equal the labels")
    if any((not math.isfinite(v)) or v < 0 for v in d.values()):
        raise ItemError(f"{name} has negative or non-finite values")
    s = sum(d.values())
    if not (0.98 <= s <= 1.02):
        raise ItemError(f"{name} sums to {s:.3f}")
    return {k: d[k] / s for k in labels}


def one_hot(labels: list[str], gold: str, smoothing: float = 0.0) -> dict[str, float]:
    k = len(labels)
    eps = smoothing / k
    return {lab: (1.0 - smoothing + eps if lab == gold else eps) for lab in labels}


def read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: Iterable[dict]) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def append_jsonl(path: str, rows: Iterable[dict]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
