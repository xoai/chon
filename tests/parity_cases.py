"""Requests and model settings whose reads must match the training code token for token (tests/test_parity.py).

run_case() takes the DecisionModel class as an argument and imports nothing from chon, so tests/make_golden.py runs
the same cases against the training repository's chon.core.infer.DecisionModel.
"""

from __future__ import annotations

import os

TEMPS = {"choice": 1.3, "noul": 0.8, "score": 1.1}

ROUTE = {"type": "choice", "instructions": "Which action does the ticket request?",
         "criteria": {lab: f"the customer wants to {lab}" for lab in ["refund", "replace", "escalate", "close"]}}
ROUTE_LIST = {"type": "choice", "instructions": "Which action does the ticket request?",   # same text, other options
              "criteria": ["refund", "close", "wait"]}
WITHIN = {"type": "noul", "instructions": "Is the parcel within the limit?",
          "criteria": {"true": "weight <= limit", "false": "weight > limit"}}
SEVERE = {"type": "score", "instructions": "How   severe\nis it?\t(rate it)", "criteria": ["low", "medium", "high"]}
DICT_INSTR = {"type": "choice", "instructions": {"task": "Pick  the  team", "note": "billing\nor shipping"},
              "criteria": {"0": "billing", "1": "shipping", "2": "other"}, "option_keys": "hide"}
MANY = {"type": "choice", "instructions": "Which product line is it about?",
        "criteria": {f"line{i}": f"product line number {i}" for i in range(40)}}
MANY_LIST = {"type": "choice", "instructions": "Which warehouse ships it?", "criteria": [f"wh{i}" for i in range(20)]}

TEXT_STATE = "Parcel 7: weight 5 kg, limit 4 kg. The customer asks for a refund and is upset."
JSON_STATE = {"ticket": 12, "request": "replace", "note": "code 9", "history": ["opened", "escalated"]}
CONV_STATE = [{"role": "user", "content": "Mức độ 7, the box arrived damaged."},
              {"role": "assistant", "content": "Đã ghi nhận."}]


def _p(name, layout, questions, state=TEXT_STATE, reread=None, env=None, **cfg):
    c = {"temperatures": TEMPS, **cfg}
    if layout is not None:
        c["prompt_layout"] = layout
    if reread is not None:
        c["reread"] = reread
    return {"name": name, "call": "predict", "cfg": c, "env": env, "state": state, "questions": questions}


def _d(name, layout, requests, reread=None, env=None, calibrated=True, group_states=True):
    c = {"temperatures": TEMPS}
    if layout is not None:
        c["prompt_layout"] = layout
    if reread is not None:
        c["reread"] = reread
    return {"name": name, "call": "distributions", "cfg": c, "env": env, "requests": requests,
            "calibrated": calibrated, "group_states": group_states}


RR = {"below": 0.6, "layout": "twice", "max_tokens": 4000, "combine": "avg"}
RR_ALL = {"below": 1.01, "layout": "twice", "max_tokens": 4000, "combine": "avg"}

CASES = [
    # first reads: layouts
    _p("state_first_default", None, {"route": ROUTE, "within": WITHIN, "severe": SEVERE}),
    _p("state_first_json", "state_first", {"route": ROUTE, "team": DICT_INSTR}, state=JSON_STATE, noul_commit=[0.199, 0.801]),
    _p("header_one", "header", {"route": ROUTE}),
    _p("header_multi", "header", {"route": ROUTE, "within": WITHIN, "severe": SEVERE, "team": DICT_INSTR}),
    _p("header_multi_conv", "header", {"severe": SEVERE, "within": WITHIN}, state=CONV_STATE),
    _p("header_same_instructions", "header", {"a": ROUTE, "b": ROUTE_LIST, "within": WITHIN}),
    _p("hybrid_one", "hybrid_header", {"route": ROUTE}),
    _p("hybrid_multi", "hybrid_header", {"route": ROUTE, "within": WITHIN, "severe": SEVERE}),
    _p("hybrid_same_instructions", "hybrid_header", {"a": ROUTE, "b": ROUTE_LIST}),
    _p("hybrid_same_plus_one", "hybrid_header", {"a": ROUTE, "b": ROUTE_LIST, "within": WITHIN}),
    # choices over 16 options: group rounds, then a final round, in the request's layout
    _p("state_first_many", "state_first", {"line": MANY, "within": WITHIN}),
    _p("header_many", "header", {"line": MANY, "wh": MANY_LIST, "within": WITHIN}),
    _p("header_many_alone", "header", {"line": MANY}, state=JSON_STATE),
    _p("hybrid_many_alone", "hybrid_header", {"line": MANY}),
    _p("hybrid_many_multi", "hybrid_header", {"line": MANY, "within": WITHIN}),
    # the second read
    _p("reread_state_first_avg", None, {"route": ROUTE, "within": WITHIN, "severe": SEVERE}, reread=RR_ALL),
    _p("reread_state_first_replace", "state_first", {"route": ROUTE, "within": WITHIN},
       reread={**RR_ALL, "combine": "replace"}),
    _p("reread_state_first_threshold", "state_first", {"route": ROUTE, "within": WITHIN, "team": DICT_INSTR},
       state=JSON_STATE, reread=RR),
    _p("reread_header_avg", "header", {"route": ROUTE, "within": WITHIN, "severe": SEVERE}, reread=RR_ALL),
    _p("reread_header_replace", "header", {"route": ROUTE, "within": WITHIN}, reread={**RR_ALL, "combine": "replace"}),
    _p("reread_header_one_avg", "header", {"within": WITHIN}, state=CONV_STATE, reread=RR_ALL),
    _p("reread_hybrid_one_avg", "hybrid_header", {"route": ROUTE}, reread=RR_ALL),
    _p("reread_hybrid_one_replace", "hybrid_header", {"within": WITHIN}, reread={**RR_ALL, "combine": "replace"}),
    _p("reread_hybrid_multi_avg", "hybrid_header", {"route": ROUTE, "within": WITHIN}, reread=RR_ALL),
    _p("reread_hybrid_multi_replace", "hybrid_header", {"route": ROUTE, "within": WITHIN, "severe": SEVERE},
       reread={**RR_ALL, "combine": "replace"}),
    _p("reread_hybrid_same_instructions", "hybrid_header", {"a": ROUTE, "b": ROUTE_LIST}, reread=RR_ALL),
    _p("reread_max_tokens", "state_first", {"route": ROUTE, "within": WITHIN}, reread={**RR_ALL, "max_tokens": 90}),
    _p("reread_types_choice", "header", {"route": ROUTE, "within": WITHIN}, reread={**RR_ALL, "types": ["choice"]}),
    _p("reread_many", "header", {"line": MANY, "route": ROUTE, "within": WITHIN}, reread=RR_ALL),
    _p("reread_sandwich", "header", {"route": ROUTE, "within": WITHIN}, reread={**RR_ALL, "layout": "sandwich"}),
    _p("reread_number", "state_first", {"route": ROUTE, "within": WITHIN}, reread=0.7),
    # $CHON_REREAD
    _p("env_header_replace", "header", {"route": ROUTE, "within": WITHIN}, env="0.95,twice,,replace"),
    _p("env_hybrid_sandwich", "hybrid_header", {"route": ROUTE, "within": WITHIN}, env="1.01,sandwich,3000"),
    _p("env_off", "header", {"route": ROUTE, "within": WITHIN}, reread=RR_ALL, env="off"),
    _p("env_over_config", "state_first", {"route": ROUTE, "within": WITHIN}, reread=RR, env="0.99"),
    # offline reads (distributions): every question about one state listed in its header
    _d("dist_header", "header", [(TEXT_STATE, ROUTE, None), (JSON_STATE, WITHIN, None), (TEXT_STATE, SEVERE, None),
                                 (JSON_STATE, ROUTE, ["close", "escalate", "replace", "refund"])]),
    _d("dist_hybrid", "hybrid_header", [(TEXT_STATE, ROUTE, None), (TEXT_STATE, WITHIN, None),
                                        (JSON_STATE, ROUTE, None), (CONV_STATE, ROUTE, None),
                                        (CONV_STATE, ROUTE_LIST, None)]),
    _d("dist_hybrid_reread", "hybrid_header", [(TEXT_STATE, ROUTE, None), (TEXT_STATE, WITHIN, None),
                                               (JSON_STATE, ROUTE, None)], reread=RR_ALL),
    _d("dist_header_reread_replace", "header", [(TEXT_STATE, ROUTE, None), (TEXT_STATE, WITHIN, None),
                                                (CONV_STATE, SEVERE, None)], reread={**RR_ALL, "combine": "replace"}),
    _d("dist_header_ungrouped", "header", [(TEXT_STATE, ROUTE, None), (TEXT_STATE, WITHIN, None)],
       reread=RR_ALL, group_states=False),
    _d("dist_uncalibrated_no_reread", "header", [(TEXT_STATE, ROUTE, None), (TEXT_STATE, WITHIN, None)],
       reread=RR_ALL, calibrated=False),
    _d("dist_state_first_reread", "state_first", [(TEXT_STATE, ROUTE, None), (JSON_STATE, WITHIN, None)],
       reread=RR_ALL),
]


# ---- the shared-prefix read: every case above with it on, plus states longer than the tiny model's 64-token sliding
# window (so the cached prefix crosses it) -----------------------------------------------------------------------------
LONG_STATE = " ".join(f"Parcel {i} weighs {i % 7} kg and goes to depot {i % 3}." for i in range(30))
LONG_JSON = {"ticket": 31, "lines": [{"sku": f"A{i}", "qty": i % 4, "note": f"box {i} of the order"} for i in range(16)]}

PREFIX_EXTRA = [
    _p("long_state_first", "state_first", {"route": ROUTE, "within": WITHIN, "severe": SEVERE, "team": DICT_INSTR},
       state=LONG_STATE),
    _p("long_header", "header", {"route": ROUTE, "within": WITHIN, "severe": SEVERE}, state=LONG_JSON),
    _p("long_header_many_reread", "header", {"line": MANY, "wh": MANY_LIST, "route": ROUTE, "within": WITHIN},
       state=LONG_STATE, reread=RR_ALL),
    _p("long_state_first_many_reread", "state_first", {"line": MANY, "route": ROUTE, "within": WITHIN},
       state=LONG_JSON, reread={**RR_ALL, "combine": "replace"}),
    _p("long_hybrid_reread", "hybrid_header", {"a": ROUTE, "b": ROUTE_LIST, "within": WITHIN}, state=LONG_STATE,
       reread=RR_ALL),
    _p("long_many_alone", "header", {"line": MANY}, state=LONG_STATE),
    _p("long_sandwich", "header", {"route": ROUTE, "within": WITHIN}, state=LONG_STATE,
       reread={**RR_ALL, "layout": "sandwich"}),
    _d("long_dist_header", "header", [(LONG_STATE, ROUTE, None), (LONG_STATE, WITHIN, None), (LONG_JSON, SEVERE, None),
                                      (LONG_JSON, ROUTE, None), (TEXT_STATE, WITHIN, None)]),
    _d("long_dist_state_first", "state_first", [(LONG_STATE, ROUTE, None), (LONG_STATE, WITHIN, None),
                                                (LONG_STATE, SEVERE, None)], reread=RR_ALL),
]
PREFIX_CASES = CASES + PREFIX_EXTRA


def run_case(DecisionModel, model, tok, case: dict, deadline: float | None = None, prefix_cache: bool = False) -> dict:
    """Run one case; returns every read (token ids, number of options, in call order) and the outputs.
    prefix_cache=False: the plain path (one forward pass per read), whatever the code's default. True: the shared-prefix
    read for any shared part (prefix_min_tokens 1; distributions with share_prefix), and the record also holds each
    prepared prefix length ("prefix") and every letter read ("letter_reads", with the prefix it was read against);
    "reads" then holds the reads that went through the plain path."""
    old = os.environ.pop("CHON_REREAD", None)
    if case.get("env") is not None:
        os.environ["CHON_REREAD"] = case["env"]
    cfg = {**case["cfg"], "prefix_cache": bool(prefix_cache)}
    if prefix_cache:
        cfg["prefix_min_tokens"] = 1
    try:
        dm = DecisionModel(model, tok, cfg, device="cpu")
    finally:
        os.environ.pop("CHON_REREAD", None)
        if old is not None:
            os.environ["CHON_REREAD"] = old
    reads = []
    real = dm.raw_logits

    def recording(seqs, n_options, letter_ids=None):
        reads.append({"ids": [list(map(int, s)) for s in seqs], "n_options": list(map(int, n_options))})
        return real(seqs, n_options, letter_ids)
    dm.raw_logits = recording
    out = {"reread": dm.reread}
    if prefix_cache:
        out["prefix"], out["letter_reads"] = [], []
        real_prepare, real_logits = dm._prepare_prefix, dm._logits

        def prepare(seqs, *a, **k):
            n = real_prepare(seqs, *a, **k)
            out["prefix"].append(int(n))
            return n

        def letters(seqs, n_options):
            sp = dm._shared
            out["letter_reads"].append({"ids": [list(map(int, s)) for s in seqs], "n_options": list(map(int, n_options)),
                                        "prefix": sp.length if sp is not None else 0})
            return real_logits(seqs, n_options)
        dm._prepare_prefix, dm._logits = prepare, letters
    if case["call"] == "predict":
        if deadline is None:
            answers, usage = dm.predict(case["state"], case["questions"])
        else:
            answers, usage = dm.predict(case["state"], case["questions"], deadline=deadline)
        usage.pop("forward_s", None)
        out.update(answers=answers, usage=usage)
    else:
        reqs = [(s, q, o) for s, q, o in case["requests"]]
        kw = {"share_prefix": True} if prefix_cache else {}
        out["distributions"] = dm.distributions(reqs, calibrated=case["calibrated"],
                                                group_states=case["group_states"], **kw)
        out["served_reads"] = [list(x) for x in dm.served_reads([(s, q) for s, q, _ in reqs])]
    out["reads"] = reads
    return out
