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
  16 then a final round between the group winners, and no yes/no commit band (a JevBench-specific rule).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path

try:
    from decision_index.engines.base import Engine, Unsupported
except ImportError:  # the kit is not installed: the helpers below still import
    Engine, Unsupported = object, ValueError


class ChonEngine(Engine):
    name = "chon"
    latency = ("In-process request wall time, one request at a time: prompt construction and the forward passes "
               "(one per question; two rounds for a choice with more than 16 options); excludes model loading.")

    def __init__(self, model="ninhnv/chon-1.0-12b", device="cuda", think=False, max_state_tokens=32768,
                 max_batch_tokens=32768, noul_commit=False, **options):
        super().__init__(**options)
        import torch

        from chon.infer import DecisionModel

        self.dm = DecisionModel.load(model, device=device, think=None if think else False,
                                     max_batch_tokens=int(max_batch_tokens))
        self.dm.builder.max_state_tokens = int(max_state_tokens)
        self.dm.cfg["truncate"] = False
        if not noul_commit:
            self.dm.noul_commit = None
        self._cuda = str(device).startswith("cuda") and torch.cuda.is_available()
        self.provenance = {
            "kind": "chon", "model": model, "think": bool(think), "max_state_tokens": int(max_state_tokens),
            "noul_commit": bool(noul_commit), "temperatures": self.dm.temperatures,
            "post_temperatures": self.dm.post_temperatures,
            "policy": "One forward pass per question reading the option letters (A-P) after one fixed prompt; "
                      "choices with more than 16 options are read in groups of at most 16, then a final round "
                      "between the group winners, P = P(group wins) x P(option | group). A state over "
                      "max_state_tokens is refused as unsupported (maximum context length), never truncated."}

    def __call__(self, state, questions):
        from chon.schema import ItemError

        try:
            answers, usage = self.dm.predict(state, questions)
        except ItemError as e:  # capacity (too long) or a request this engine cannot take: unsupported
            raise Unsupported(str(e)) from None
        return {"model": self.dm.name, "answers": answers, "usage": usage}, None

    def synchronize(self):
        if self._cuda:
            import torch

            torch.cuda.synchronize()

    def runtime(self):
        import torch
        import transformers

        return {"torch": torch.__version__, "transformers": transformers.__version__,
                "device": torch.cuda.get_device_name(0) if self._cuda else "cpu"}


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
