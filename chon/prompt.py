"""The decision prompt and the letter readout.

One forward pass per question: the prompt ends exactly where the model's reply starts, and the
answer is the next-token distribution over the option letters A, B, C, ... restricted to the
options that exist. Nothing is generated, so a decision costs input tokens only.

Prompt layout (user turn of the base model's chat template, thinking disabled):

    <state, compact; JSON states as compact JSON, conversations as "role: text" lines>

    ---
    Question: <instructions>
    A) <label> — <description>
    B) ...
    Answer with one letter.

The state comes first so that several questions about one state share a prefix.
Keep this scaffold short: every token here is paid for on JevBench's Cost axis.

Other layouts (PromptBuilder.render `layout`; chosen by the model's chon_config.json "prompt_layout"):

    header        Questions that will be asked about this record:
                  - <instructions of each question of the request, distinct, in request order>
                  ---
                  <state>

                  ---
                  Question: ...            (the question being read, as above)

    twice         the state-first prompt up to the options, "Read the request again:", then the whole
                  state-first prompt again (a second read of an unsure answer; "twice_header" for a header model)
    sandwich      the question, the state, the question again (another second-read layout)

Prompt style "decider" (PromptBuilder `style`; chosen by chon_config.json "prompt_style", default "chon" as above):

    (header only) Questions that will be asked about the context:
                  - <instructions of every question of the request, in request order, duplicates kept>

    Context:
    <state; JSON with spaces, each element of an array of 8 or more numbered with "_index">

    Question: <instructions>          ("Which answer fits the context?" for a yes/no question without any)
    Options:
    (A) <label>: <description>        (yes/no questions show (A) no, (B) yes)
    (B) ...

and the letter is read after "Answer: (" at the start of the reply. Layouts: state_first, header, and their second
reads twice / twice_header (the whole user turn, "Read the same input again before answering:", then it again).
"""

from __future__ import annotations

import json
from typing import Any, Sequence

from .schema import MAX_OPTIONS, instructions_text, options_for

PROMPT_VERSION = 1
LETTERS = "ABCDEFGHIJKLMNOP"[:MAX_OPTIONS]
TRUNC_HEAD = "\n[... truncated]"
TRUNC_CONV = "[earlier messages omitted]\n"
ANSWER_LINE = "Answer with one letter."
REREAD_LINE = "Read the request again:"
# layouts of the second read of an unsure answer (DecisionModel "reread"): read only when the first read is unsure.
# "twice": the whole request written out twice (each token of the second copy sees the whole first copy, which a
# one-way model otherwise never does). "sandwich": the question, the state, the question again.
REREAD_LAYOUTS = ("twice", "sandwich")
# "header": every question of the request listed first, then the state, then the question being read, so the state
# is read knowing everything that will be asked about it. A header model's second read ("twice") is "twice_header".
HEADER_LINE = "Questions that will be asked about this record:"
# prompt style "decider": the chat layout of decider-ai 1.6.0 (Apache-2.0) as FlexDM v1.0 reads stock Gemma-4-31B (MIT,
# github.com/1000luca/FlexDM): "Context:" + the state (JSON with spaces, the positions of long arrays written in),
# "Question:", "Options:" with "(A) key: description" lines, and the letter read after "Answer: (" in the reply. Their
# header list ("Questions that will be asked about the context:") and read-twice line are kept verbatim. A model trained
# in a style is served in it (chon_config.json "prompt_style").
PROMPT_STYLES = ("chon", "decider")
DEC_HEADER = "Questions that will be asked about the context:"
DEC_AGAIN = "\n\nRead the same input again before answering:\n"
DEC_ANSWER = "Answer: ("
DEC_NOUL_TEXT = "Which answer fits the context?"
ANNOTATE_MIN = 8


def annotate_indices(x: Any, min_len: int = ANNOTATE_MIN) -> Any:
    """Each element's position written into arrays of min_len or more (decider style): a question naming records[47]
    becomes a lookup, not a count."""
    if isinstance(x, list):
        if len(x) >= min_len:
            return [{"_index": i, **annotate_indices(v, min_len)} if isinstance(v, dict)
                    else {"_index": i, "value": annotate_indices(v, min_len)} for i, v in enumerate(x)]
        return [annotate_indices(v, min_len) for v in x]
    if isinstance(x, dict):
        return {k: annotate_indices(v, min_len) for k, v in x.items()}
    return x


def decider_state_text(state: Any) -> str:
    return state if isinstance(state, str) else json.dumps(annotate_indices(state), ensure_ascii=False)


def _dec_txt(v: Any) -> str:
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)


def decider_question_text(question: dict, order: Sequence[str]) -> str:
    """'Question: <instructions>\\nOptions:\\n(A) key: description ...' (a noul question's options are yes / no)."""
    raw = question.get("instructions")
    text = DEC_NOUL_TEXT if question.get("type") == "noul" and raw in (None, "") else _dec_txt(raw if raw is not None else "")
    opts = dict(options_for(question))
    hide = question.get("option_keys") == "hide"
    lines = [f"Question: {text}", "Options:"]
    for letter, label in zip(LETTERS, order):
        desc = opts[label]
        lines.append(f"({letter}) {desc}" if desc and hide else f"({letter}) {label}: {desc}" if desc else f"({letter}) {label}")
    return "\n".join(lines)


def decider_question_label(question: dict) -> str:
    """The question as the decider header lists it: the instructions as written."""
    raw = question.get("instructions")
    return DEC_NOUL_TEXT if question.get("type") == "noul" and raw in (None, "") else _dec_txt(raw if raw is not None else "")


def _is_message(x: Any) -> bool:
    return isinstance(x, dict) and "role" in x and ("content" in x or "text" in x)


def conversation_of(state: Any) -> list[dict] | None:
    if isinstance(state, list) and state and all(_is_message(m) for m in state):
        return state
    if isinstance(state, dict):
        for key in ("messages", "conversation"):
            conv = state.get(key)
            if isinstance(conv, list) and conv and all(_is_message(m) for m in conv) and len(state) == 1:
                return conv
    return None


def _message_line(m: dict) -> str:
    content = m.get("content", m.get("text"))
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False, separators=(",", ":"))
    return f"{m['role']}: {content}"


def state_text(state: Any) -> str:
    if isinstance(state, str):
        return state.strip()
    conv = conversation_of(state)
    if conv is not None:
        return "\n".join(_message_line(m) for m in conv)
    return json.dumps(state, ensure_ascii=False, separators=(",", ":"))


def question_text(question: dict, order: Sequence[str]) -> str:
    opts = dict(options_for(question))
    lines = [f"Question: {instructions_text(question)}"]
    hide = question.get("option_keys") == "hide"  # keys are ids (e.g. "0", "1"): show only the descriptions
    for letter, label in zip(LETTERS, order):
        desc = opts[label]
        if desc and hide:
            lines.append(f"{letter}) {desc}")
        else:
            lines.append(f"{letter}) {label} — {desc}" if desc else f"{letter}) {label}")
    lines.append(ANSWER_LINE)
    return "\n".join(lines)


class PromptBuilder:
    """Builds token ids for (state, question, option order) and knows the letter token ids."""

    def __init__(self, tokenizer, max_state_tokens: int = 6000, chat_template_kwargs: dict | None = None,
                 answer_prefix: str = "", style: str = "chon"):
        if style not in PROMPT_STYLES:
            raise ValueError(f"unknown prompt_style {style!r} (one of {PROMPT_STYLES})")
        self.tok = tokenizer
        self.style = style
        self.max_state_tokens = int(max_state_tokens)
        self.chat_template_kwargs = {"enable_thinking": False} if chat_template_kwargs is None else dict(chat_template_kwargs)
        self.answer_prefix = DEC_ANSWER if style == "decider" else answer_prefix
        if not getattr(tokenizer, "chat_template", None):
            raise ValueError("tokenizer has no chat template; use the instruct (-it) checkpoint's tokenizer")
        self.letter_ids = self._find_letter_ids()

    # ---- text ---------------------------------------------------------------------------------
    def _n_tokens(self, text: str) -> int:
        return len(self.tok(text, add_special_tokens=False)["input_ids"])

    def fit_state(self, state: Any) -> tuple[str, bool]:
        """Render the state and cut it to max_state_tokens. Returns (text, truncated)."""
        text = decider_state_text(state) if self.style == "decider" else state_text(state)
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        if len(ids) <= self.max_state_tokens:
            return text, False
        conv = conversation_of(state) if self.style == "chon" else None
        if conv is not None:  # keep the most recent turns
            kept: list[str] = []
            budget = self.max_state_tokens - self._n_tokens(TRUNC_CONV)
            for m in reversed(conv):
                line = _message_line(m)
                n = self._n_tokens(line + "\n")
                if n > budget:
                    if not kept:  # a single huge last message: keep its end
                        tail = self.tok(line, add_special_tokens=False)["input_ids"][-budget:]
                        kept.append(self.tok.decode(tail))
                    break
                kept.append(line)
                budget -= n
            return TRUNC_CONV + "\n".join(reversed(kept)), True
        head = ids[: self.max_state_tokens - self._n_tokens(TRUNC_HEAD)]
        return self.tok.decode(head) + TRUNC_HEAD, True

    def render(self, state_txt: str, question: dict, order: Sequence[str], layout: str = "state_first",
               asked: Sequence[str] | None = None) -> str:
        """layout "state_first" (the state, then the question block: siblings share the state as a prefix), "header"
        (the request's questions `asked`, each with its whitespace collapsed, then the state, then the question), or
        a second-read layout: "twice", "sandwich" (REREAD_LAYOUTS) or "twice_header" (a header read, twice).
        The decider style has state_first, header, twice and twice_header (_render_decider)."""
        if self.style == "decider":
            return self._render_decider(state_txt, question, order, layout, asked)
        q = question_text(question, order)
        head = q[: -len(ANSWER_LINE)].rstrip("\n")
        if layout == "state_first":
            content = f"{state_txt}\n\n---\n{q}"
        elif layout == "twice":
            content = f"{state_txt}\n\n---\n{head}\n\n{REREAD_LINE}\n\n{state_txt}\n\n---\n{q}"
        elif layout == "sandwich":
            content = f"{head}\n---\n{state_txt}\n\n---\n{q}"
        elif layout in ("header", "twice_header"):
            lst = [" ".join(str(a).split()) for a in (asked or [instructions_text(question)])]
            top = HEADER_LINE + "\n" + "\n".join(f"- {a}" for a in lst) + f"\n---\n{state_txt}\n\n---\n"
            content = top + q if layout == "header" else f"{top}{head}\n\n{REREAD_LINE}\n\n{top}{q}"
        else:
            raise ValueError(f"unknown prompt layout {layout!r}")
        msgs = [{"role": "user", "content": content}]
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                            **self.chat_template_kwargs)
        return text + self.answer_prefix

    def natural_order(self, question: dict) -> list[str]:
        """The option order a request is read in: the criteria's, with yes/no questions shown as (A) no, (B) yes in the
        decider style."""
        labels = [lab for lab, _ in options_for(question)]
        return labels[::-1] if self.style == "decider" and question.get("type") == "noul" else labels

    def asked_list(self, questions: Sequence[dict]) -> list[str]:
        """The header's list for a request's questions: chon style, the distinct instructions (whitespace collapsed
        when rendered); decider style, every question's instructions as written (duplicates kept)."""
        if self.style == "decider":
            return [decider_question_label(q) for q in questions]
        return list(dict.fromkeys(instructions_text(q) for q in questions))

    def _render_decider(self, state_txt: str, question: dict, order: Sequence[str], layout: str,
                        asked: Sequence[str] | None) -> str:
        """The decider style: state_first, header, and their second reads (twice, twice_header)."""
        if layout not in ("state_first", "header", "twice", "twice_header"):
            raise ValueError(f"prompt_style 'decider' has no {layout!r} layout")
        top = ""
        if layout in ("header", "twice_header"):
            lst = list(asked) if asked else [decider_question_label(question)]
            top = DEC_HEADER + "\n" + "\n".join(f"- {a}" for a in lst) + "\n\n"
        user = f"{top}Context:\n{state_txt}\n\n{decider_question_text(question, order)}"
        if layout in ("twice", "twice_header"):
            user = user + DEC_AGAIN + user
        text = self.tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                            add_generation_prompt=True, **self.chat_template_kwargs)
        return text + self.answer_prefix

    def encode(self, state: Any, question: dict, order: Sequence[str] | None = None,
               fitted_state: str | None = None, layout: str = "state_first",
               asked: Sequence[str] | None = None) -> list[int]:
        if order is None:
            order = self.natural_order(question)
        if len(order) > len(self.letter_ids):
            raise ValueError(f"at most {len(self.letter_ids)} options")
        st = fitted_state if fitted_state is not None else self.fit_state(state)[0]
        return self.tok(self.render(st, question, order, layout, asked), add_special_tokens=False)["input_ids"]

    # ---- letters ------------------------------------------------------------------------------
    def _find_letter_ids(self) -> list[int]:
        """Token id of each letter as the first token of the reply.

        Checked at the real prompt boundary, so tokenizers that merge the letter with the
        preceding characters are caught here rather than silently mis-reading the answer.
        """
        probe = self.render("probe", {"type": "noul", "instructions": "probe"}, ["yes", "no"])
        base = self.tok(probe, add_special_tokens=False)["input_ids"]
        for letters, variant in [(L, v) for L in (LETTERS, LETTERS[:10]) for v in ("{}", " {}")]:
            ids = []
            for letter in letters:  # all 16 if they are single tokens here, else the 10 the model was trained on
                full = self.tok(probe + variant.format(letter), add_special_tokens=False)["input_ids"]
                if len(full) != len(base) + 1 or full[: len(base)] != base:
                    ids = None
                    break
                ids.append(full[-1])
            if ids and len(set(ids)) == len(ids):
                return ids
        raise ValueError("option letters are not single tokens at the reply boundary; "
                         "set answer_prefix (e.g. 'Answer:') so that ' A'... are single tokens")


# ---- thinking on unsure questions (inline: the model reasons inside its reply, then the letter is read
# after a fixed suffix; one sequence that shares the served prompt)
INLINE_PREFIX = "Let me check the deciding facts first.\n"
INLINE_SUFFIX = "\n\nFinal answer:"


def suffix_letter_ids(tok, suffix: str = INLINE_SUFFIX) -> list[int]:
    """Token id of each option letter right after `suffix` (as ids are concatenated, not re-tokenised)."""
    base = tok(suffix, add_special_tokens=False)["input_ids"]
    for letters, variant in [(L, v) for L in (LETTERS, LETTERS[:10]) for v in (" {}", "{}")]:
        ids = []
        for letter in letters:
            full = tok(suffix + variant.format(letter), add_special_tokens=False)["input_ids"]
            if len(full) != len(base) + 1 or full[: len(base)] != base:
                ids = None
                break
            ids.append(full[-1])
        if ids and len(set(ids)) == len(ids):
            return ids
    raise ValueError(f"option letters are not single tokens after {suffix!r}")
