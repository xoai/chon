"""Write tests/golden_parity.json from the training code (the reference implementation of the reads).

    python tests/make_golden.py --training /path/to/chon-training [--out tests/golden_parity.json]
    python tests/make_golden.py --training /path/to/chon-training --prefix     # -> tests/golden_prefix.json

Runs with the training repository first on sys.path, so `chon` is the training package: it builds the tiny model
(tests/tiny_model.py), runs every case of tests/parity_cases.py through chon.core.infer.DecisionModel and records each
read's token ids and the outputs. The public tests compare against this file without needing the training code.
--prefix runs tests/parity_cases.py's PREFIX_CASES with the training code's shared-prefix read on instead (its
chon.core.prefix), recording each prepared prefix and letter read too (tests/golden_prefix.json).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--training", required=True, help="the training repository (its chon.core.infer)")
    ap.add_argument("--out", default=None, help="default tests/golden_parity.json (tests/golden_prefix.json with --prefix)")
    ap.add_argument("--prefix", action="store_true", help="the shared-prefix read (PREFIX_CASES)")
    a = ap.parse_args(argv)
    a.out = a.out or os.path.join(HERE, "golden_prefix.json" if a.prefix else "golden_parity.json")
    training = os.path.abspath(a.training)
    sys.path[:0] = [training, HERE]
    import torch
    import transformers

    from chon.core.infer import DecisionModel   # the training package: fails here if the path is wrong
    from parity_cases import CASES, PREFIX_CASES, run_case
    from tiny_model import build_tiny_model, fingerprint

    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as tmp:
        model_dir = build_tiny_model(os.path.join(tmp, "tiny"))
        base = DecisionModel.load(model_dir, device="cpu", dtype="float32")
        results = {c["name"]: run_case(DecisionModel, base.model, base.tok, c, prefix_cache=a.prefix)
                   for c in (PREFIX_CASES if a.prefix else CASES)}
        fp = fingerprint(model_dir)
    commit = subprocess.run(["git", "-C", training, "rev-parse", "HEAD"], capture_output=True, text=True,
                            check=False).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", training, "status", "--porcelain", "--", "chon"], capture_output=True,
                                text=True, check=False).stdout.strip())
    golden = {"made_with": {"training_commit": commit, "training_tree_dirty": dirty, "torch": torch.__version__,
                            "transformers": transformers.__version__, "fixture": fp,
                            **({"prefix_cache": True} if a.prefix else {})},
              "cases": results}
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("{\n" + f'"made_with": {json.dumps(golden["made_with"])},\n"cases": {{\n')
        f.write(",\n".join(f"{json.dumps(k)}: {json.dumps(v, ensure_ascii=False, separators=(',', ':'))}"
                           for k, v in results.items()))
        f.write("\n}\n}\n")
    n_reads = sum(len(r["reads"]) for r in results.values())
    print(json.dumps({"out": a.out, "cases": len(results), "reads": n_reads, **golden["made_with"]}))


if __name__ == "__main__":
    main()
