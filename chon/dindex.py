"""Chon on the Decision Index (github.com/apolinario/decision-index): an in-process engine for its runner, plus
shard / merge helpers so one GPU can run several runner processes side by side (the runner sends one request at a
time; results are the same rows whichever process answered them).

    # engine, as the kit's own commands use it
    python -m decision_index run --engine chon.dindex:ChonEngine --option model=ninhnv/chon-1.0-12b \\
        --rows sample-100.jsonl.gz --out runs/di/sample
    # the full public suite in N parallel shards, then one run folder that the kit scores and a PR points at
    python -m chon.dindex shard --suite-dir suite-0.3 --n 6 --out runs/di/shards
    python -m chon.dindex merge --suite-dir suite-0.3 --shards runs/di/shards --out runs/di/chon-1.0-12b

Engine settings for the board (its rules: no truncation, every option gets a probability, one fixed rendering):
  think=false (the board's latency rule: median, mean and p80 under 1 s per request), a long state limit with a
  capacity refusal ("maximum context length") instead of cutting, choices over 16 options read in groups of at most
  16 then a final round between the group winners, and no yes/no commit band (a JevBench-specific rule). The prompt
  layout (state_first, header, hybrid_header), the prompt style (chon, decider), whether the letters are read under
  the final logit soft cap, any second read of unsure answers ("twice", "sandwich" or "reverse") and a pair read (every
  request read with the adapter and without it, averaged) are the model's own settings (chon_config.json
  "prompt_layout" / "prompt_style" / "uncap_logits" / "reread" / "pair"; $CHON_REREAD overrides the second read) and are
  stated in the run's provenance.

Latency (the board lists a model only if its median, mean and 80th-percentile request time are each under 1 s on one
RTX PRO 6000, 96 GB): the shared-prefix read is on by default (chon_config.json "prefix_cache", or
--option prefix_cache=false to measure without it). The part every read of a request shares (template start, a header
model's question list, the state) runs once; the questions then read their own tokens against it in packed passes,
the group and final rounds of a choice over 16 options and second reads that start with the same prefix included. A
request with k questions about a P-token state costs about P + k x (question tokens) instead of k x P, with the same
answers (bf16 rounding). Memory bound: see README, "Shared-prefix read".
"""
from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
import os
from pathlib import Path

# fewer "reserved but unused" blocks between requests of very different lengths (1.5.1); set before torch starts CUDA
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

try:
    from decision_index.engines.base import Engine, Unsupported
except ImportError:  # the kit is not installed: the helpers below still import
    Engine, Unsupported = object, ValueError


class ChonEngine(Engine):
    name = "chon"
    latency = ("In-process request wall time, one request at a time: prompt construction and the forward passes "
               "(with the shared-prefix read on, the default: the part all reads of the request share once, then "
               "every question's own tokens in one packed pass; otherwise one pass per question; two rounds for a "
               "choice with more than 16 options; one more for an unsure answer when a second read is set); "
               "excludes model loading.")

    def __init__(self, model="ninhnv/chon-1.0-12b", device="cuda", think=False, max_state_tokens=32768,
                 max_batch_tokens=32768, noul_commit=False, prefix_cache=None, **options):
        """prefix_cache: None = the model's setting (chon_config.json "prefix_cache", default on); true / false
        overrides it. Options from the kit's --option key=value arrive as strings ("false" is off)."""
        super().__init__(**options)
        import torch

        from chon.infer import DecisionModel, flag

        think, noul_commit = flag(think), flag(noul_commit)
        self.dm = DecisionModel.load(model, device=device, think=None if think else False,
                                     max_batch_tokens=int(max_batch_tokens))
        self.dm.builder.max_state_tokens = int(max_state_tokens)
        self.dm.cfg["truncate"] = False
        if prefix_cache is not None and str(prefix_cache).strip() != "":
            self.dm.cfg["prefix_cache"] = flag(prefix_cache)
        if not noul_commit:
            self.dm.noul_commit = None
        self._cuda = str(device).startswith("cuda") and torch.cuda.is_available()
        self.oom_retries = 0
        rr = self.dm.reread
        layout_text = {
            "state_first": "the state, then the question and its options",
            "header": "every distinct question of the request listed first (by its instructions, in request order), "
                      "then the state, then the question and its options",
            "hybrid_header": "for a request with two or more distinct questions, every distinct question listed "
                             "first (by its instructions, in request order), then the state, then the question and "
                             "its options; for a request with one, the state, then the question and its options",
        }[self.dm.layout]
        if self.dm.style == "decider":
            layout_text = layout_text.replace("distinct question", "question").replace(
                "(by its instructions, in request order)", "(every question's instructions, in request order)")
        style_text = {
            "chon": "",
            "decider": ("Prompt style 'decider': 'Context:' and the state (JSON with spaces, elements of arrays of 8 "
                        "or more numbered with _index), 'Question:', 'Options:' with '(A) key: description' lines "
                        "(yes/no as (A) no, (B) yes), the letter read after 'Answer: ('. "),
        }[self.dm.style] + ("The letters are read with the model's final logit soft cap off. " if self.dm.uncapped
                            else "")
        reread_text = (
            f"A choice or yes/no answer whose top calibrated probability is under {rr['below']:g}, in a prompt of at "
            f"most {rr['max_tokens']} tokens, is read a second time in the {rr['layout']!r} layout"
            + ((" (the request written out twice" + (", the question list included when the first read had one)"
                                                       if self.dm.layout != "state_first" else ")"))
               if rr["layout"] == "twice" else "")
            + (" (the first read's prompt with the options in reverse order, mapped back to the options)"
               if rr["layout"] == "reverse" else "")
            + (" and the two distributions are averaged" if rr["combine"] == "avg"
               else " and the second read replaces the first")
            + "; choices over 16 options are read once. " if rr else "Every question is read once (no second read). ")
        pair = self.dm.pair
        pair_text = (
            "Every request is read twice, with the adapter and with it switched off (the base model in the same "
            "prompt), each read calibrated with its own temperatures (the base read's: "
            + ", ".join(f"{k} {v:g}" for k, v in pair["temperatures"].items())
            + ") and second-read as set, and the two are averaged "
            f"({pair['weight']:g} x adapter + {1 - pair['weight']:g} x base) before the final rules (post-calibration"
            + (", the yes/no commit band" if noul_commit else "") + ") run once on the average. " if pair else "")
        if pair:   # two reads per request
            self.latency = type(self).latency + (" With the pair read, each request is read twice (with the adapter "
                                                 "and with it switched off).")
        prefix_text = ("The tokens all reads of a request share (template start, question list, state) are run "
                       "once per request and shared by its questions through the key/value cache, each question masked "
                       "to the prefix and its own tokens (the group and final rounds and second reads that start with "
                       "the same tokens included), so the answers equal separate passes. ")
        self.provenance = {
            "kind": "chon", "model": model, "think": bool(think), "max_state_tokens": int(max_state_tokens),
            "noul_commit": bool(noul_commit), "temperatures": self.dm.temperatures,
            "post_temperatures": self.dm.post_temperatures,
            "prompt_layout": self.dm.layout, "prompt_style": self.dm.style, "uncap_logits": self.dm.uncapped,
            "pair": pair, "reread": rr, "prefix_cache": self.dm.prefix_cache,
            "policy": "One read per question of the option letters (A-P) at the end of one fixed prompt "
                      f"(layout {self.dm.layout!r}: {layout_text}); " + style_text +
                      "choices with more than 16 options are read in groups of at most 16, then a final round "
                      "between the group winners, P = P(group wins) x P(option | group). " + reread_text + pair_text +
                      (prefix_text if self.dm.prefix_cache else
                       "Every read runs its whole prompt (no shared-prefix read). ") +
                      "A state over max_state_tokens is refused as unsupported (maximum context length), never "
                      "truncated. Long prompts' attention runs in query chunks (bounded GPU memory, the same "
                      "answers), and a request that runs out of GPU memory is read once more in smaller pieces."}

    def __call__(self, state, questions):
        from chon.schema import ItemError

        try:
            try:
                answers, usage = self.dm.predict(state, questions)
            except Exception as e:   # noqa: BLE001 -- out of GPU memory: once more with smaller pieces (1.5.1)
                if not _is_oom(e):
                    raise
                answers, usage = self._retry_smaller(state, questions)
        except ItemError as e:  # capacity (too long) or a request this engine cannot take: unsupported
            raise Unsupported(str(e)) from None
        return {"model": self.dm.name, "answers": answers, "usage": usage}, None

    def _retry_smaller(self, state, questions):
        """A request that ran out of GPU memory, read again with a quarter of the attention budget, the batch budget
        and the shared-prefix pack: the same answers in smaller pieces. A second failure is the request's error."""
        from chon import attention

        _free_gpu()
        old = (attention.budget_bytes(), self.dm.max_batch_tokens, self.dm.cfg.get("prefix_pack_tokens"))
        attention.set_budget(max(64 * 2 ** 20, old[0] // 4))
        self.dm.max_batch_tokens = max(1024, old[1] // 4)
        self.dm.cfg["prefix_pack_tokens"] = max(512, int(old[2] or 8192) // 4)
        try:
            answers, usage = self.dm.predict(state, questions)
        finally:
            attention.set_budget(None if old[0] == attention.DEFAULT_BUDGET_MB * 2 ** 20 else old[0])
            self.dm.max_batch_tokens = old[1]
            if old[2] is None:
                self.dm.cfg.pop("prefix_pack_tokens", None)
            else:
                self.dm.cfg["prefix_pack_tokens"] = old[2]
            _free_gpu()
        self.oom_retries += 1
        return answers, {**usage, "oom_retry": 1}

    def synchronize(self):
        if self._cuda:
            import torch

            torch.cuda.synchronize()

    def runtime(self):
        import torch
        import transformers

        return {"torch": torch.__version__, "transformers": transformers.__version__,
                "device": torch.cuda.get_device_name(0) if self._cuda else "cpu"}


def _is_oom(e: BaseException) -> bool:
    import torch

    oom = getattr(torch, "OutOfMemoryError", None) or getattr(torch.cuda, "OutOfMemoryError", None)
    return (oom is not None and isinstance(e, oom)) or "out of memory" in str(e).lower()


def _free_gpu() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _suite(directory, edition=None):
    from decision_index.suite.io import Suite

    return Suite(directory, edition)


def cmd_shard(a):
    """In-edition rows (exclusions are applied at scoring time, as the kit's pipeline does), split by case so a
    linked case stays in one shard; skips rows already answered under --done (a results.jsonl)."""
    suite = _suite(a.suite_dir, a.edition)
    done = set()
    for p in a.done or []:
        for line in open(p, encoding="utf-8"):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("status") != "error":
                done.add(r["run_id"])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = [gzip.open(out / f"shard-{k}.jsonl.gz", "wt", encoding="utf-8") for k in range(a.n)]
    counts = [0] * a.n
    for r in suite.rows(apply_exclusions=False):
        e = r["_evaluation"]
        if e["run_id"] in done:
            continue
        k = int(hashlib.sha256(f"{e['catalog_id']}:{e['group_id']}".encode()).hexdigest(), 16) % a.n
        files[k].write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
        counts[k] += 1
    for f in files:
        f.close()
    print(json.dumps({"shards": a.n, "rows": counts, "total": sum(counts), "skipped_done": len(done)}))


def cmd_merge(a):
    """shard-*/results.jsonl -> <out>/results.jsonl (rows untouched; for a run_id seen twice the last non-error
    row wins), environment.json from shard 0 with the suite's corpus hash, then the kit's scorer."""
    from decision_index.pipeline import score_run

    suite = _suite(a.suite_dir, a.edition)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    shards = sorted(Path(a.shards).glob("shard-*/results.jsonl"))
    rows = {}
    for p in ([out / "results.jsonl"] if (out / "results.jsonl").exists() else []) + shards:
        for line in open(p, encoding="utf-8"):
            if not line.endswith("\n"):
                continue
            r = json.loads(line)
            if r["run_id"] not in rows or rows[r["run_id"]]["status"] == "error" or r["status"] != "error":
                rows[r["run_id"]] = r
    tmp = out / "results.jsonl.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows.values():
            f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(tmp, out / "results.jsonl")
    env0 = Path(a.shards) / "shard-0" / "environment.json"
    if env0.exists():
        env = json.load(open(env0))
        env["frozen_corpus_sha256"] = suite.verify(strict=False).get("sha256")
        env["rows_path"] = "full public suite, run in %d parallel shards (chon.dindex shard/merge)" % len(shards)
        json.dump(env, open(out / "environment.json", "w"), indent=2)
    scores = score_run(suite, out / "results.jsonl", a.engine, out)
    print(json.dumps({"rows": len(rows), "decision_index": scores.get("decision_index"),
                      "complete": scores.get("complete"), "completed": scores.get("completed")}, indent=2))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("shard")
    s.add_argument("--suite-dir", default="suite-0.3")
    s.add_argument("--edition", default=None)
    s.add_argument("--n", type=int, default=4)
    s.add_argument("--out", required=True)
    s.add_argument("--done", nargs="*", help="results.jsonl files whose answered rows are skipped (resume)")
    m = sub.add_parser("merge")
    m.add_argument("--suite-dir", default="suite-0.3")
    m.add_argument("--edition", default=None)
    m.add_argument("--shards", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--engine", default="chon")
    a = ap.parse_args(argv)
    {"shard": cmd_shard, "merge": cmd_merge}[a.cmd](a)


if __name__ == "__main__":
    main()
