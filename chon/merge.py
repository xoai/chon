"""Merge a Chon adapter into its base model and write a standalone checkpoint (what vLLM serves: chon.vllm_reader).

    python -m chon.merge --adapter ./chon-1.1-31b --out ./chon-1.1-31b-merged

The output loads like any Gemma checkpoint and carries the adapter's chon_config.json (prompt settings, temperatures).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

COPY_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "tokenizer.model",
              "chat_template.jinja", "chat_template.json", "processor_config.json", "preprocessor_config.json",
              "generation_config.json")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--base", default=None, help="defaults to base_model in the adapter's chon_config.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device-map", default="cpu", help="'cpu' (needs ~2x the model's size in RAM) or 'auto'")
    a = ap.parse_args(argv)

    import torch
    from peft import PeftModel
    from transformers.utils import cached_file

    from .modeling import CONFIG_NAME, load_causal_lm, read_model_config, write_model_config

    cfg = read_model_config(a.adapter)
    base = a.base or cfg.get("base_model")
    if not base:
        raise SystemExit("pass --base (the adapter's chon_config.json names none)")
    model = load_causal_lm(base, dtype=a.dtype, device_map=a.device_map)
    model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload()
    model.save_pretrained(a.out, safe_serialization=True, max_shard_size="5GB")
    for name in COPY_FILES:
        src = os.path.join(a.adapter, name)
        if not os.path.exists(src):
            try:
                src = cached_file(base, name)
            except Exception:  # noqa: BLE001
                src = None
        if src and os.path.exists(src):
            shutil.copy(src, os.path.join(a.out, name))
    cfg["merged_from"] = {"base": base, "adapter": os.path.abspath(a.adapter)}
    write_model_config(a.out, cfg)
    print(json.dumps({"merged": a.out, "config": os.path.join(a.out, CONFIG_NAME)}))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
