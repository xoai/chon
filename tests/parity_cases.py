"""Requests and model settings whose reads must match the training code token for token (tests/test_parity.py).

run_case() takes the DecisionModel class as an argument and imports nothing from chon, so tests/make_golden.py runs
the same cases against the training repository's chon.core.infer.DecisionModel. A case with "adapter" reads the tiny
model with the tiny LoRA adapter on it, unmerged (tests/tiny_model.py tiny_adapter; the pair read needs one).
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


def _p(name, layout, questions, state=TEXT_STATE, reread=None, env=None, softcap=None, adapter=False, **cfg):
    c = {"temperatures": TEMPS, **cfg}
    if layout is not None:
        c["prompt_layout"] = layout
    if reread is not None:
        c["reread"] = reread
    case = {"name": name, "call": "predict", "cfg": c, "env": env, "state": state, "questions": questions}
    if softcap is not None:   # the model's final logit soft cap for this case (the tiny model has none)
        case["softcap"] = softcap
    if adapter:   # read with the tiny LoRA adapter, unmerged
        case["adapter"] = True
    return case


def _d(name, layout, requests, reread=None, env=None, calibrated=True, group_states=True, softcap=None, adapter=False,
       **cfg):
    c = {"temperatures": TEMPS, **cfg}
    if layout is not None:
        c["prompt_layout"] = layout
    if reread is not None:
        c["reread"] = reread
    case = {"name": name, "call": "distributions", "cfg": c, "env": env, "requests": requests,
            "calibrated": calibrated, "group_states": group_states}
    if softcap is not None:
        case["softcap"] = softcap
    if adapter:
        case["adapter"] = True
    return case


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


# ---- prompt style "decider" (1.4.0): "Context:" / "Question:" / "Options:" lines, yes/no read as (A) no, (B) yes, every
# question listed in a header read (duplicates kept), JSON states with spaces and long arrays numbered, the letter read
# after "Answer: (" (a config's own answer_prefix is overridden); and "uncap_logits" (the model's final soft cap off)
DEC = {"prompt_style": "decider"}
UNCAP = {"uncap_logits": True}
CAP = 2.0   # a final logit soft cap for the tiny model (which has none): its letter logits are about 1-3
BARE_NOUL = {"type": "noul", "instructions": "", "criteria": None}   # offline reads only (a request needs instructions)
ARRAY_STATE = {"order": 77, "items": [{"sku": f"B{i}", "qty": i % 3} for i in range(9)], "scores": list(range(8)),
               "seven": ["x"] * 7, "nested": {"rows": [[i, i + 1] for i in range(8)]}, "note": "Mức độ 3"}

DECIDER_CASES = [
    # first reads
    _p("dec_state_first", None, {"route": ROUTE, "within": WITHIN, "severe": SEVERE, "team": DICT_INSTR}, **DEC),
    _p("dec_state_first_json_array", "state_first", {"route": ROUTE, "within": WITHIN}, state=ARRAY_STATE,
       noul_commit=[0.199, 0.801], answer_prefix="Answer:", **DEC),
    _p("dec_state_first_conv", "state_first", {"severe": SEVERE, "within": WITHIN}, state=CONV_STATE, **DEC),
    _p("dec_header_multi", "header", {"route": ROUTE, "within": WITHIN, "severe": SEVERE, "team": DICT_INSTR}, **DEC),
    _p("dec_header_same_instructions", "header", {"a": ROUTE, "b": ROUTE_LIST, "within": WITHIN}, **DEC),
    _p("dec_header_json_array", "header", {"route": ROUTE, "within": WITHIN, "severe": SEVERE}, state=ARRAY_STATE,
       **DEC),
    _p("dec_header_one", "header", {"within": WITHIN}, state=CONV_STATE, **DEC),
    _p("dec_hybrid_one", "hybrid_header", {"within": WITHIN}, **DEC),
    _p("dec_hybrid_same_instructions", "hybrid_header", {"a": ROUTE, "b": ROUTE_LIST}, **DEC),
    _p("dec_header_many", "header", {"line": MANY, "wh": MANY_LIST, "within": WITHIN}, **DEC),
    _p("dec_state_first_many", "state_first", {"line": MANY, "within": WITHIN}, state=ARRAY_STATE, **DEC),
    # the second read: "twice", and "twice_header" for a header read
    _p("dec_reread_header_avg", "header", {"route": ROUTE, "within": WITHIN, "severe": SEVERE}, reread=RR_ALL, **DEC),
    _p("dec_reread_header_replace", "header", {"route": ROUTE, "within": WITHIN}, state=ARRAY_STATE,
       reread={**RR_ALL, "combine": "replace"}, **DEC),
    _p("dec_reread_header_threshold", "header", {"route": ROUTE, "within": WITHIN, "severe": SEVERE, "team": DICT_INSTR},
       reread=RR, **DEC),
    _p("dec_reread_state_first_avg", "state_first", {"route": ROUTE, "within": WITHIN, "team": DICT_INSTR},
       reread=RR_ALL, **DEC),
    _p("dec_reread_hybrid_one", "hybrid_header", {"within": WITHIN}, reread=RR_ALL, **DEC),
    _p("dec_reread_hybrid_multi_replace", "hybrid_header", {"route": ROUTE, "within": WITHIN},
       reread={**RR_ALL, "combine": "replace"}, **DEC),
    _p("dec_reread_hybrid_same_instructions", "hybrid_header", {"a": ROUTE, "b": ROUTE_LIST}, reread=RR_ALL, **DEC),
    _p("dec_reread_many", "header", {"line": MANY, "route": ROUTE, "within": WITHIN}, reread=RR_ALL, **DEC),
    _p("dec_reread_max_tokens", "state_first", {"route": ROUTE, "within": WITHIN},
       reread={**RR_ALL, "max_tokens": 110}, **DEC),
    _p("dec_env_header_replace", "header", {"route": ROUTE, "within": WITHIN}, env="0.95,twice,,replace", **DEC),
    _p("dec_env_over_config", "state_first", {"route": ROUTE, "within": WITHIN}, reread=RR, env="0.99", **DEC),
    # uncapped letters: the same requests under a soft cap, read with it (capped) and without it (uncap_logits)
    _p("dec_header_capped", "header", {"route": ROUTE, "within": WITHIN, "severe": SEVERE, "team": DICT_INSTR},
       softcap=CAP, **DEC),
    _p("dec_header_uncapped", "header", {"route": ROUTE, "within": WITHIN, "severe": SEVERE, "team": DICT_INSTR},
       softcap=CAP, **DEC, **UNCAP),
    _p("dec_reread_header_uncapped", "header", {"route": ROUTE, "within": WITHIN, "severe": SEVERE}, reread=RR_ALL,
       softcap=CAP, **DEC, **UNCAP),
    _p("chon_state_first_capped", None, {"route": ROUTE, "within": WITHIN, "severe": SEVERE}, softcap=CAP),
    _p("chon_state_first_uncapped", None, {"route": ROUTE, "within": WITHIN, "severe": SEVERE}, softcap=CAP, **UNCAP),
    # offline reads
    _d("dec_dist_header", "header", [(TEXT_STATE, ROUTE, None), (ARRAY_STATE, WITHIN, None), (TEXT_STATE, SEVERE, None),
                                     (ARRAY_STATE, ROUTE, ["close", "escalate", "replace", "refund"]),
                                     (TEXT_STATE, BARE_NOUL, None), (ARRAY_STATE, WITHIN, ["yes", "no"])], **DEC),
    _d("dec_dist_hybrid_reread", "hybrid_header", [(TEXT_STATE, ROUTE, None), (TEXT_STATE, ROUTE_LIST, None),
                                                   (ARRAY_STATE, WITHIN, None), (CONV_STATE, BARE_NOUL, None)],
       reread=RR_ALL, **DEC),
    _d("dec_dist_state_first_uncapped", "state_first", [(TEXT_STATE, ROUTE, None), (ARRAY_STATE, WITHIN, None),
                                                        (CONV_STATE, SEVERE, None)],
       reread=RR_ALL, softcap=CAP, **DEC, **UNCAP),
    _d("dec_dist_header_ungrouped", "header", [(TEXT_STATE, ROUTE, None), (TEXT_STATE, WITHIN, None)],
       reread=RR_ALL, group_states=False, **DEC),
    _d("dec_dist_uncalibrated", "header", [(TEXT_STATE, ROUTE, None), (TEXT_STATE, WITHIN, None)],
       reread=RR_ALL, calibrated=False, **DEC),
]
CASES = CASES + DECIDER_CASES


# ---- the "reverse" second read (1.5.0): the first read's prompt (its layout and question list) with the options in
# reverse order, mapped back to the options, then averaged in or put in place of the first read
RRV = {**RR_ALL, "layout": "reverse"}
RRV_REPLACE = {**RRV, "combine": "replace"}

REVERSE_CASES = [
    _p("rev_state_first_avg", None, {"route": ROUTE, "within": WITHIN, "severe": SEVERE}, reread=RRV),
    _p("rev_state_first_replace", "state_first", {"route": ROUTE, "within": WITHIN, "team": DICT_INSTR},
       state=JSON_STATE, reread=RRV_REPLACE),
    _p("rev_state_first_threshold", "state_first", {"route": ROUTE, "within": WITHIN, "team": DICT_INSTR},
       state=JSON_STATE, reread={**RR, "layout": "reverse"}),
    _p("rev_header_avg", "header", {"route": ROUTE, "within": WITHIN, "severe": SEVERE, "team": DICT_INSTR}, reread=RRV),
    _p("rev_header_replace", "header", {"route": ROUTE, "within": WITHIN}, state=CONV_STATE, reread=RRV_REPLACE),
    _p("rev_hybrid_one", "hybrid_header", {"within": WITHIN}, reread=RRV),
    _p("rev_hybrid_multi_replace", "hybrid_header", {"route": ROUTE, "within": WITHIN}, reread=RRV_REPLACE),
    _p("rev_many", "header", {"line": MANY, "route": ROUTE, "within": WITHIN}, reread=RRV),
    _p("rev_max_tokens", "state_first", {"route": ROUTE, "within": WITHIN}, reread={**RRV, "max_tokens": 90}),
    _p("rev_env_header", "header", {"route": ROUTE, "within": WITHIN}, env="1.01,reverse,3000"),
    _p("rev_env_over_config", "state_first", {"route": ROUTE, "within": WITHIN}, reread=RR, env="1.01,reverse,,replace"),
    _p("rev_dec_state_first_avg", "state_first", {"route": ROUTE, "within": WITHIN, "severe": SEVERE,
                                                  "team": DICT_INSTR}, reread=RRV, **DEC),
    _p("rev_dec_header_replace", "header", {"route": ROUTE, "within": WITHIN, "severe": SEVERE}, state=ARRAY_STATE,
       reread=RRV_REPLACE, **DEC),
    _p("rev_dec_header_avg", "header", {"route": ROUTE, "within": WITHIN}, reread=RRV, **DEC),
    _p("rev_dec_state_first_replace", "state_first", {"within": WITHIN, "team": DICT_INSTR}, state=CONV_STATE,
       reread=RRV_REPLACE, **DEC),
    _p("rev_dec_hybrid_same_instructions", "hybrid_header", {"a": ROUTE, "b": ROUTE_LIST}, reread=RRV, **DEC),
    _d("rev_dist_header", "header", [(TEXT_STATE, ROUTE, None), (JSON_STATE, WITHIN, None), (TEXT_STATE, SEVERE, None),
                                     (JSON_STATE, ROUTE, ["close", "escalate", "replace", "refund"])], reread=RRV),
    _d("rev_dist_state_first_ungrouped", "state_first", [(TEXT_STATE, ROUTE, None), (JSON_STATE, WITHIN, ["no", "yes"])],
       reread=RRV_REPLACE, group_states=False),
    _d("rev_dist_dec_hybrid_replace", "hybrid_header", [(TEXT_STATE, ROUTE, None), (TEXT_STATE, ROUTE_LIST, None),
                                                        (ARRAY_STATE, WITHIN, None), (CONV_STATE, BARE_NOUL, None),
                                                        (ARRAY_STATE, WITHIN, ["yes", "no"])],
       reread=RRV_REPLACE, **DEC),
]
CASES = CASES + REVERSE_CASES


# ---- the pair read (1.5.0): every request read with the adapter (TEMPS) and with it switched off (PAIR_T), each read
# second-read as set, averaged (weight x adapter + (1 - weight) x base), then post-calibration and the yes/no commit band
# once on the average. "adapter_alone" and "base_alone" are the two members read on their own.
PAIR_T = {"choice": 0.9, "noul": 1.2, "score": 0.7}
PAIR = {"pair": {"weight": 0.5, "temperatures": PAIR_T}}
PAIR25 = {"pair": {"weight": 0.25, "temperatures": PAIR_T}}
COMMIT = {"noul_commit": [0.199, 0.801]}
TRIO = {"route": ROUTE, "within": WITHIN, "severe": SEVERE}

PAIR_CASES = [
    _p("adapter_alone", None, TRIO, adapter=True),
    _p("base_alone", None, TRIO, temperatures=PAIR_T),
    _p("pair_w50", None, TRIO, adapter=True, **PAIR),
    _p("pair_w25", None, TRIO, adapter=True, **PAIR25),
    _p("pair_weight_one", None, TRIO, adapter=True, pair={"weight": 1.0, "temperatures": PAIR_T}),
    _p("pair_default_temperatures", None, TRIO, adapter=True, pair={"weight": 0.5}),
    _p("pair_noul_commit", None, TRIO, adapter=True, **PAIR, **COMMIT),
    _p("pair_post_temperatures", None, TRIO, adapter=True, post_temperatures={"choice": 0.7, "noul": 1.5}, **PAIR),
    _p("pair_header", "header", {**TRIO, "team": DICT_INSTR}, adapter=True, **PAIR),
    _p("pair_dec_header", "header", {**TRIO, "team": DICT_INSTR}, adapter=True, **PAIR, **DEC),
    _p("pair_dec_hybrid_one", "hybrid_header", {"within": WITHIN}, state=CONV_STATE, adapter=True, **PAIR25, **DEC),
    _p("pair_dec_state_first_commit", "state_first", {"route": ROUTE, "within": WITHIN}, state=ARRAY_STATE,
       adapter=True, **PAIR25, **COMMIT, **DEC),
    _p("pair_reread_twice", "header", TRIO, reread=RR_ALL, adapter=True, **PAIR),
    _p("pair_reread_threshold", "state_first", {"route": ROUTE, "within": WITHIN, "team": DICT_INSTR}, state=JSON_STATE,
       reread=RR, adapter=True, **PAIR),
    _p("pair_dec_reread_twice_replace", "header", TRIO, reread={**RR_ALL, "combine": "replace"}, adapter=True,
       **PAIR25, **DEC),
    _p("pair_reverse", "header", TRIO, reread=RRV, adapter=True, **PAIR),
    _p("pair_dec_reverse_replace", "state_first", {"route": ROUTE, "within": WITHIN, "team": DICT_INSTR},
       reread=RRV_REPLACE, adapter=True, **PAIR25, **DEC),
    _p("pair_env_reverse_commit", "header", {"route": ROUTE, "within": WITHIN}, env="1.01,reverse,3000", adapter=True,
       **PAIR, **COMMIT),
    _p("pair_many", "header", {"line": MANY, "within": WITHIN}, adapter=True, **PAIR),
    _d("pair_dist_header_reverse", "header", [(TEXT_STATE, ROUTE, None), (JSON_STATE, WITHIN, None),
                                              (TEXT_STATE, SEVERE, None), (JSON_STATE, ROUTE, ["close", "refund"])],
       reread=RRV, adapter=True, **PAIR),
    _d("pair_dist_dec_twice_commit", "hybrid_header", [(TEXT_STATE, ROUTE, None), (TEXT_STATE, WITHIN, None),
                                                       (CONV_STATE, WITHIN, None)],
       reread=RR_ALL, adapter=True, **PAIR25, **COMMIT, **DEC),
    _d("pair_dist_uncalibrated", "header", [(TEXT_STATE, ROUTE, None), (TEXT_STATE, WITHIN, None)], reread=RR_ALL,
       calibrated=False, adapter=True, **PAIR),
]
CASES = CASES + PAIR_CASES


# ---- temperatures per category (1.6.0): a question takes the temperature of "<type>:<options>:<state>", then
# "<type>:<options>", else its type's (TEMPS); options binned 2 / 3-4 / 5-9 / 10-25 / 26+, the state empty / text / json.
# Every value differs from the others and from TEMPS, so a read at the wrong level gives other probabilities.
TWO = {"type": "choice", "instructions": "Billing or shipping?", "criteria": ["billing", "shipping"]}
SEVEN = {"type": "choice", "instructions": "Which colour is the parcel?",
         "criteria": {c: f"the parcel is {c}" for c in ["red", "orange", "yellow", "green", "blue", "indigo", "violet"]}}
TWELVE = {"type": "choice", "instructions": "Which month was it sent?",
          "criteria": ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]}
SCORE5 = {"type": "score", "instructions": "How urgent is it?", "criteria": ["none", "low", "some", "high", "top"]}
CATS = {
    "choice:3-4:text": 1.9, "choice:3-4": 0.6,          # ROUTE: text state; any other state
    "choice:2": 1.55,                                    # TWO, and ROUTE read with two options
    "choice:5-9:text": 0.5,                              # SEVEN with a text state (a JSON state: the type's)
    "choice:10-25": 0.7, "choice:26+": 1.75,             # TWELVE and MANY_LIST; MANY
    "noul:2:json": 0.45, "noul:2:empty": 2.2,            # WITHIN: JSON / conversation; empty (text: the type's)
    "score:3-4": 1.6, "score:5-9:json": 0.65,            # SEVERE: any state; SCORE5 with a JSON state
    "choice:3-4:empty": 2.4,
}
CAT = {"temperature_categories": CATS}
PAIR_CATS = {"choice:3-4": 0.75, "noul:2:text": 1.45, "score:3-4:text": 0.55}   # the base read's own
PAIRC = {"pair": {"weight": 0.5, "temperatures": PAIR_T, "temperature_categories": PAIR_CATS}}
PAIRC25 = {"pair": {"weight": 0.25, "temperatures": PAIR_T, "temperature_categories": PAIR_CATS}}
ALL6 = {"route": ROUTE, "within": WITHIN, "severe": SEVERE, "two": TWO, "seven": SEVEN, "twelve": TWELVE}
RRO = {**RR_ALL, "layout": "rotate"}
RRO_REPLACE = {**RRO, "combine": "replace"}

CATEGORY_CASES = [
    # every level and bin, every state kind (first reads)
    _p("cat_text", None, {**ALL6, "urgent": SCORE5}, **CAT),
    _p("cat_json", "state_first", {**ALL6, "urgent": SCORE5}, state=JSON_STATE, **CAT),
    _p("cat_conv", "state_first", {"route": ROUTE, "within": WITHIN, "severe": SEVERE, "urgent": SCORE5},
       state=CONV_STATE, **CAT),
    _p("cat_empty", "state_first", {"route": ROUTE, "within": WITHIN, "two": TWO}, state="", **CAT),
    _p("cat_none_state", "state_first", {"route": ROUTE, "within": WITHIN}, state=None, **CAT),
    _p("cat_blank_state", "header", {"route": ROUTE, "within": WITHIN, "severe": SEVERE}, state="   \n", **CAT),
    _p("cat_header", "header", ALL6, **CAT),
    _p("cat_hybrid_json", "hybrid_header", {"route": ROUTE, "within": WITHIN, "seven": SEVEN}, state=JSON_STATE, **CAT),
    _p("cat_dec_header", "header", {**ALL6, "team": DICT_INSTR}, state=ARRAY_STATE, **CAT, **DEC),
    _p("cat_dec_state_first", "state_first", ALL6, **CAT, **DEC),
    _p("cat_commit_post", None, {"route": ROUTE, "within": WITHIN}, state=JSON_STATE,
       post_temperatures={"choice": 0.8, "noul": 1.3}, noul_commit=[0.199, 0.801], **CAT),
    # choices over 16 options: both rounds at the category of the request's largest one (26+ or 10-25)
    _p("cat_many_26", "state_first", {"line": MANY, "within": WITHIN}, **CAT),
    _p("cat_many_10_25", "state_first", {"wh": MANY_LIST, "route": ROUTE}, state=JSON_STATE, **CAT),
    _p("cat_many_both", "header", {"line": MANY, "wh": MANY_LIST, "route": ROUTE}, **CAT),
    _p("cat_many_empty", "header", {"wh": MANY_LIST}, state="", **CAT),
    _p("cat_dec_many", "header", {"wh": MANY_LIST, "within": WITHIN}, state=CONV_STATE, **CAT, **DEC),
    # with a second read: each question's own temperature in its second read
    _p("cat_twice", "header", ALL6, reread=RR_ALL, **CAT),
    _p("cat_twice_threshold", "state_first", {"route": ROUTE, "within": WITHIN, "two": TWO, "seven": SEVEN},
       state=JSON_STATE, reread=RR, **CAT),
    _p("cat_twice_replace_conv", "hybrid_header", {"route": ROUTE, "within": WITHIN}, state=CONV_STATE,
       reread={**RR_ALL, "combine": "replace"}, **CAT),
    _p("cat_sandwich", "header", {"route": ROUTE, "within": WITHIN}, reread={**RR_ALL, "layout": "sandwich"}, **CAT),
    _p("cat_rotate", "state_first", ALL6, reread=RRO, **CAT),
    _p("cat_rotate_replace_json", "header", {"route": ROUTE, "within": WITHIN, "seven": SEVEN, "two": TWO},
       state=JSON_STATE, reread=RRO_REPLACE, **CAT),
    _p("cat_rotate_empty", "state_first", {"route": ROUTE, "within": WITHIN}, state="", reread=RRO, **CAT),
    _p("cat_rotate_threshold", "header", ALL6, reread={**RR, "layout": "rotate"}, **CAT),
    _p("cat_reverse", "header", {"route": ROUTE, "within": WITHIN, "seven": SEVEN}, state=CONV_STATE, reread=RRV, **CAT),
    _p("cat_env_rotate", "hybrid_header", {"route": ROUTE, "within": WITHIN, "seven": SEVEN}, env="1.01,rotate,3000",
       **CAT),
    _p("cat_dec_rotate", "header", {"route": ROUTE, "within": WITHIN, "seven": SEVEN}, state=JSON_STATE,
       reread=RRO_REPLACE, **CAT, **DEC),
    _p("cat_dec_twice", "state_first", {"route": ROUTE, "within": WITHIN, "two": TWO}, state=CONV_STATE,
       reread=RR_ALL, **CAT, **DEC),
    _p("cat_many_rotate", "header", {"line": MANY, "route": ROUTE, "within": WITHIN}, reread=RRO, **CAT),
    # with the pair read: the adapter read takes the model's categories, the base read the pair's (or none)
    _p("cat_pair", None, {"route": ROUTE, "within": WITHIN, "severe": SEVERE, "two": TWO}, adapter=True, **CAT,
       **PAIRC),
    _p("cat_pair_json", "header", {"route": ROUTE, "within": WITHIN, "severe": SEVERE}, state=JSON_STATE, adapter=True,
       **CAT, **PAIRC25),
    _p("cat_pair_without_pair_cats", None, {"route": ROUTE, "within": WITHIN, "severe": SEVERE}, adapter=True, **CAT,
       **PAIR),
    _p("cat_pair_only_pair_cats", None, {"route": ROUTE, "within": WITHIN, "severe": SEVERE}, adapter=True, **PAIRC),
    _p("cat_pair_rotate", "header", {"route": ROUTE, "within": WITHIN, "seven": SEVEN}, reread=RRO, adapter=True,
       **CAT, **PAIRC),
    _p("cat_pair_twice_commit", "state_first", {"route": ROUTE, "within": WITHIN}, state="", reread=RR_ALL,
       adapter=True, **CAT, **PAIRC25, **COMMIT),
    _p("cat_pair_many", "header", {"line": MANY, "within": WITHIN}, adapter=True, **CAT, **PAIRC),
    _p("cat_pair_dec_rotate", "state_first", {"route": ROUTE, "within": WITHIN, "two": TWO}, state=CONV_STATE,
       reread=RRO_REPLACE, adapter=True, **CAT, **PAIRC, **DEC),
    # offline reads: a request's own option order sets its bin (ROUTE read with two options is "choice:2")
    _d("cat_dist", "header", [(TEXT_STATE, ROUTE, None), (JSON_STATE, WITHIN, None), (CONV_STATE, SEVERE, None),
                              (JSON_STATE, ROUTE, ["close", "refund"]), (None, WITHIN, None), ("", ROUTE, None),
                              (TEXT_STATE, SEVEN, None), (JSON_STATE, SEVEN, None), (TEXT_STATE, TWELVE, None)],
       **CAT),
    _d("cat_dist_rotate", "hybrid_header", [(TEXT_STATE, ROUTE, None), (TEXT_STATE, SEVEN, None),
                                            (JSON_STATE, WITHIN, None), (CONV_STATE, ROUTE, ["escalate", "close", "refund"]),
                                            ("", WITHIN, None)], reread=RRO, **CAT),
    _d("cat_dist_twice_ungrouped", "state_first", [(TEXT_STATE, ROUTE, None), (JSON_STATE, WITHIN, None),
                                                   (None, TWO, None)], reread=RR_ALL, group_states=False, **CAT),
    _d("cat_dist_uncalibrated", "header", [(TEXT_STATE, ROUTE, None), (JSON_STATE, WITHIN, None)], reread=RRO,
       calibrated=False, **CAT),
    _d("cat_dist_pair_rotate", "header", [(TEXT_STATE, ROUTE, None), (JSON_STATE, WITHIN, None),
                                          (TEXT_STATE, SEVERE, None), ("", ROUTE, ["close", "refund"])],
       reread=RRO, adapter=True, **CAT, **PAIRC),
    _d("cat_dist_dec", "header", [(ARRAY_STATE, ROUTE, None), (ARRAY_STATE, WITHIN, None), (TEXT_STATE, SCORE5, None),
                                  (ARRAY_STATE, SCORE5, None), (CONV_STATE, BARE_NOUL, None)], reread=RR_ALL,
       **CAT, **DEC),
]
CASES = CASES + CATEGORY_CASES


# ---- the "rotate" second read (1.6.0): the first read's prompt with the options rotated by half (the first len // 2
# moved to the end), mapped back by label, then averaged in or put in place of the first read
RRO_ALL_TYPES = {**RRO, "types": ["choice", "noul", "score"]}

ROTATE_CASES = [
    _p("rot_state_first_avg", None, {"route": ROUTE, "seven": SEVEN, "within": WITHIN, "severe": SEVERE}, reread=RRO),
    _p("rot_state_first_replace", "state_first", {"route": ROUTE, "seven": SEVEN, "within": WITHIN, "team": DICT_INSTR},
       state=JSON_STATE, reread=RRO_REPLACE),
    _p("rot_state_first_threshold", "state_first", {"route": ROUTE, "within": WITHIN, "team": DICT_INSTR},
       state=JSON_STATE, reread={**RR, "layout": "rotate"}),
    _p("rot_score", "state_first", {"severe": SEVERE, "urgent": SCORE5, "route": ROUTE}, reread=RRO_ALL_TYPES),
    _p("rot_score_replace", "header", {"severe": SEVERE, "urgent": SCORE5, "within": WITHIN}, state=CONV_STATE,
       reread={**RRO_ALL_TYPES, "combine": "replace"}),
    _p("rot_two_options", "state_first", {"two": TWO, "within": WITHIN}, reread=RRO),
    _p("rot_twelve", "header", {"twelve": TWELVE, "seven": SEVEN}, reread=RRO),
    _p("rot_header_avg", "header", {"route": ROUTE, "within": WITHIN, "severe": SEVERE, "team": DICT_INSTR}, reread=RRO),
    _p("rot_header_replace", "header", {"route": ROUTE, "seven": SEVEN, "within": WITHIN}, state=CONV_STATE,
       reread=RRO_REPLACE),
    _p("rot_hybrid_one", "hybrid_header", {"seven": SEVEN}, reread=RRO),
    _p("rot_hybrid_multi_replace", "hybrid_header", {"route": ROUTE, "within": WITHIN}, reread=RRO_REPLACE),
    _p("rot_hybrid_same_instructions", "hybrid_header", {"a": ROUTE, "b": ROUTE_LIST}, reread=RRO),
    _p("rot_many", "header", {"line": MANY, "route": ROUTE, "within": WITHIN}, reread=RRO),
    _p("rot_max_tokens", "state_first", {"route": ROUTE, "within": WITHIN}, reread={**RRO, "max_tokens": 90}),
    _p("rot_env_header", "header", {"route": ROUTE, "seven": SEVEN, "within": WITHIN}, env="1.01,rotate,3000"),
    _p("rot_env_over_config", "state_first", {"route": ROUTE, "within": WITHIN}, reread=RRV, env="1.01,rotate,,replace"),
    _p("rot_dec_state_first_avg", "state_first", {"route": ROUTE, "seven": SEVEN, "within": WITHIN, "severe": SEVERE,
                                                  "team": DICT_INSTR}, reread=RRO, **DEC),
    _p("rot_dec_header_replace", "header", {"route": ROUTE, "within": WITHIN, "severe": SEVERE}, state=ARRAY_STATE,
       reread=RRO_REPLACE, **DEC),
    _p("rot_dec_header_avg", "header", {"route": ROUTE, "seven": SEVEN, "within": WITHIN}, reread=RRO, **DEC),
    _p("rot_dec_score", "state_first", {"severe": SEVERE, "urgent": SCORE5}, state=CONV_STATE, reread=RRO_ALL_TYPES,
       **DEC),
    _p("rot_dec_hybrid_same_instructions", "hybrid_header", {"a": ROUTE, "b": ROUTE_LIST}, reread=RRO, **DEC),
    _p("rot_pair", "header", {"route": ROUTE, "seven": SEVEN, "within": WITHIN}, reread=RRO, adapter=True, **PAIR),
    _d("rot_dist_header", "header", [(TEXT_STATE, ROUTE, None), (JSON_STATE, WITHIN, None), (TEXT_STATE, SEVEN, None),
                                     (JSON_STATE, ROUTE, ["close", "escalate", "replace"])], reread=RRO),
    _d("rot_dist_state_first_ungrouped", "state_first", [(TEXT_STATE, ROUTE, None), (JSON_STATE, WITHIN, ["no", "yes"]),
                                                         (TEXT_STATE, SEVERE, None)],
       reread={**RRO_ALL_TYPES, "combine": "replace"}, group_states=False),
    _d("rot_dist_dec_hybrid_replace", "hybrid_header", [(TEXT_STATE, ROUTE, None), (TEXT_STATE, SEVEN, None),
                                                        (ARRAY_STATE, WITHIN, None), (CONV_STATE, BARE_NOUL, None),
                                                        (ARRAY_STATE, WITHIN, ["yes", "no"])],
       reread=RRO_REPLACE, **DEC),
]
CASES = CASES + ROTATE_CASES


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
PREFIX_EXTRA += [   # the decider style, long states
    _p("long_dec_state_first", "state_first", {"route": ROUTE, "within": WITHIN, "severe": SEVERE, "team": DICT_INSTR},
       state=LONG_STATE, **DEC),
    _p("long_dec_header_array", "header", {"route": ROUTE, "within": WITHIN, "severe": SEVERE}, state=LONG_JSON, **DEC),
    _p("long_dec_header_many_reread", "header", {"line": MANY, "wh": MANY_LIST, "route": ROUTE, "within": WITHIN},
       state=LONG_STATE, reread=RR_ALL, **DEC),
    _p("long_dec_capped", "header", {"a": ROUTE, "b": ROUTE_LIST, "within": WITHIN}, state=LONG_JSON, softcap=CAP,
       **DEC),
    _p("long_dec_uncapped_reread", "header", {"a": ROUTE, "b": ROUTE_LIST, "within": WITHIN}, state=LONG_JSON,
       reread={**RR_ALL, "combine": "replace"}, softcap=CAP, **DEC, **UNCAP),
    _p("long_dec_hybrid_reread", "hybrid_header", {"route": ROUTE, "within": WITHIN}, state=LONG_STATE, reread=RR_ALL,
       **DEC),
    _d("long_dec_dist_header", "header", [(LONG_STATE, ROUTE, None), (LONG_STATE, WITHIN, None),
                                          (LONG_JSON, SEVERE, None), (LONG_JSON, BARE_NOUL, None),
                                          (TEXT_STATE, WITHIN, None)], **DEC),
]
PREFIX_EXTRA += [   # the reverse second read and the pair read, long states
    _p("long_rev_header", "header", {"route": ROUTE, "within": WITHIN, "severe": SEVERE}, state=LONG_STATE, reread=RRV),
    _p("long_rev_dec_state_first_replace", "state_first", {"route": ROUTE, "within": WITHIN, "team": DICT_INSTR},
       state=LONG_JSON, reread=RRV_REPLACE, **DEC),
    _p("long_pair_header_reread", "header", TRIO, state=LONG_STATE, reread=RR_ALL, adapter=True, **PAIR),
    _p("long_pair_dec_reverse", "header", {"a": ROUTE, "b": ROUTE_LIST, "within": WITHIN}, state=LONG_JSON,
       reread=RRV, adapter=True, **PAIR25, **DEC),
    _p("long_pair_many_commit", "state_first", {"line": MANY, "route": ROUTE, "within": WITHIN}, state=LONG_STATE,
       adapter=True, **PAIR, **COMMIT),
    _d("long_pair_dist_header", "header", [(LONG_STATE, ROUTE, None), (LONG_STATE, WITHIN, None),
                                           (LONG_JSON, SEVERE, None), (TEXT_STATE, WITHIN, None)],
       reread=RRV, adapter=True, **PAIR),
]
PREFIX_EXTRA += [   # temperatures per category and the rotate second read, long states
    _p("long_cat_header", "header", ALL6, state=LONG_STATE, **CAT),
    _p("long_cat_json_many", "state_first", {"line": MANY, "wh": MANY_LIST, "route": ROUTE, "within": WITHIN},
       state=LONG_JSON, **CAT),
    _p("long_cat_twice", "hybrid_header", {"route": ROUTE, "within": WITHIN, "seven": SEVEN}, state=LONG_JSON,
       reread=RR_ALL, **CAT),
    _p("long_rot_header", "header", {"route": ROUTE, "seven": SEVEN, "within": WITHIN, "severe": SEVERE},
       state=LONG_STATE, reread=RRO),
    _p("long_rot_dec_state_first_replace", "state_first", {"route": ROUTE, "seven": SEVEN, "within": WITHIN,
                                                           "team": DICT_INSTR}, state=LONG_JSON, reread=RRO_REPLACE,
       **DEC),
    _p("long_rot_score_many", "header", {"line": MANY, "severe": SEVERE, "urgent": SCORE5, "within": WITHIN},
       state=LONG_STATE, reread=RRO_ALL_TYPES),
    _p("long_cat_rot_pair", "header", {"route": ROUTE, "within": WITHIN, "seven": SEVEN}, state=LONG_STATE,
       reread=RRO, adapter=True, **CAT, **PAIRC),
    _p("long_cat_dec_rotate", "header", {"a": ROUTE, "b": ROUTE_LIST, "within": WITHIN}, state=LONG_JSON,
       reread=RRO, **CAT, **DEC),
    _d("long_cat_dist_rotate", "header", [(LONG_STATE, ROUTE, None), (LONG_STATE, SEVEN, None),
                                          (LONG_JSON, WITHIN, None), (LONG_JSON, SEVERE, None), (TEXT_STATE, WITHIN, None)],
       reread=RRO, **CAT),
]
PREFIX_CASES = CASES + PREFIX_EXTRA

_ADAPTED: dict = {}


def adapter_model(model):
    """The tiny model with the tiny LoRA adapter on a copy of it (tests/tiny_model.py tiny_adapter), made once per
    model (the model itself is left as it is)."""
    if id(model) not in _ADAPTED:
        from tiny_model import tiny_adapter

        _ADAPTED[id(model)] = (model, tiny_adapter(model))
    return _ADAPTED[id(model)][1]


def _softcap_configs(model) -> list:
    """Every config object in the model's tree that has a final logit soft cap setting (set or None)."""
    seen = {}
    for m in model.modules():
        c = getattr(m, "config", None)
        for cc in (c, getattr(c, "text_config", None)):
            if cc is not None and hasattr(cc, "final_logit_softcapping"):
                seen[id(cc)] = cc
    return list(seen.values())


def run_case(DecisionModel, model, tok, case: dict, deadline: float | None = None, prefix_cache: bool = False) -> dict:
    """Run one case; returns every read (token ids, number of options, in call order) and the outputs.
    prefix_cache=False: the plain path (one forward pass per read), whatever the code's default. True: the shared-prefix
    read for any shared part (prefix_min_tokens 1; distributions with share_prefix), and the record also holds each
    prepared prefix length ("prefix") and every letter read ("letter_reads", with the prefix it was read against);
    "reads" then holds the reads that went through the plain path.
    case["softcap"]: the model reads under this final logit soft cap for the case. The model's soft cap settings are
    put back afterwards, so a case with "uncap_logits" (which turns the cap off on the shared model) leaves the next
    case's model as it found it.
    case["adapter"]: the case reads adapter_model(model) (unmerged LoRA) instead of the model."""
    if case.get("adapter"):
        model = adapter_model(model)
    saved = [(c, c.final_logit_softcapping) for c in _softcap_configs(model)]
    if case.get("softcap") is not None:
        for c, _ in saved:
            c.final_logit_softcapping = float(case["softcap"])
    try:
        return _run_case(DecisionModel, model, tok, case, deadline, prefix_cache)
    finally:
        for c, v in saved:
            c.final_logit_softcapping = v


def _run_case(DecisionModel, model, tok, case: dict, deadline: float | None, prefix_cache: bool) -> dict:
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
