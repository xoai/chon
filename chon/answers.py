"""Turn a label distribution into a TypeSafe /v1/systemone answer object."""

from __future__ import annotations

from .schema import options_for


def format_answer(question: dict, probs: dict[str, float]) -> dict:
    qtype = question["type"]
    opts = options_for(question)
    labels = [lab for lab, _ in opts]
    total = sum(probs[lab] for lab in labels)
    p = {lab: probs[lab] / total for lab in labels}
    if qtype == "noul":
        return {"type": "noul", "noul": p["yes"]}
    if qtype == "choice":
        best = max(labels, key=lambda lab: p[lab])
        return {"type": "choice", "choice": best, "confidence": p[best], "probabilities": p}
    if qtype == "score":
        expected = sum(int(lab) * p[lab] for lab in labels)
        legend = {lab: desc for lab, desc in opts if desc}
        out = {"type": "score", "score": expected, "level": int(max(labels, key=lambda lab: p[lab])),
               "probabilities": p}
        if legend:
            out["legend"] = legend
        return out
    raise ValueError(f"unknown question type {qtype!r}")
