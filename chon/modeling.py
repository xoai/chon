"""Model loading, LoRA targeting, batching and the letter-logit forward shared by training and inference."""

from __future__ import annotations

import json
import os
from typing import Sequence

import torch
from torch import nn

CONFIG_NAME = "chon_config.json"
LEGACY_CONFIG_NAMES = ()
LORA_PROJ = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
NON_TEXT = ("vision", "audio", "multi_modal", "embed_vision", "embed_audio", "visual")

DTYPES = {"bfloat16": torch.bfloat16, "bf16": torch.bfloat16, "float16": torch.float16,
          "fp16": torch.float16, "float32": torch.float32, "fp32": torch.float32}


def resolve_dtype(d) -> torch.dtype:
    return d if isinstance(d, torch.dtype) else DTYPES[str(d)]


def load_tokenizer(path: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


def load_causal_lm(path: str, dtype="bfloat16", device_map=None, attn_implementation: str | None = "sdpa",
                   low_cpu_mem_usage: bool = True):
    """Load a decoder (text-only or multimodal *ForConditionalGeneration*). Only the text path is used."""
    import transformers
    from transformers import AutoModelForCausalLM

    kwargs = dict(dtype=resolve_dtype(dtype), low_cpu_mem_usage=low_cpu_mem_usage)
    if device_map is not None:
        kwargs["device_map"] = device_map
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    if isinstance(attn_implementation, str) and attn_implementation != "eager":
        try:  # towers without SDPA support (Gemma 4 E2B/E4B audio encoder) get eager; the text decoder keeps it
            from transformers import AutoConfig

            subs = list(getattr(AutoConfig.from_pretrained(path), "sub_configs", {}) or {})
            if any("audio" in k for k in subs):
                kwargs["attn_implementation"] = {k: ("eager" if "audio" in k else attn_implementation) for k in subs}
        except Exception:  # noqa: BLE001  (config unreadable here: let from_pretrained report it)
            pass
    try:
        model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
    except (ValueError, KeyError) as first:  # vision-language checkpoints (e.g. Qwen3.6+) register elsewhere
        model = None
        for cls_name in ("AutoModelForImageTextToText", "AutoModelForMultimodalLM"):
            cls = getattr(transformers, cls_name, None)
            if cls is None:
                continue
            try:
                model = cls.from_pretrained(path, **kwargs)
                break
            except (ValueError, KeyError):
                continue
        if model is None:
            raise first
    model.config.use_cache = False
    return model


def lora_target_modules(model: nn.Module, projections: Sequence[str] = LORA_PROJ,
                        exclude: Sequence[str] = NON_TEXT) -> list[str]:
    """Full names of the text-decoder linear layers to adapt (vision/audio towers excluded)."""
    names = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if name.rsplit(".", 1)[-1] not in projections:
            continue
        low = name.lower()
        if any(x in low for x in exclude):
            continue
        names.append(name)
    if not names:
        raise ValueError(f"no linear layers named {projections} found outside {exclude}")
    return names


def left_pad(seqs: Sequence[Sequence[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    n = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), n), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), n), dtype=torch.long)
    for i, s in enumerate(seqs):
        if len(s):
            ids[i, n - len(s):] = torch.tensor(s, dtype=torch.long)
            mask[i, n - len(s):] = 1
    return ids, mask


def letter_logits(model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                  letter_ids: torch.Tensor) -> torch.Tensor:
    """Logits of the option letters at the reply position. Inputs must be LEFT padded.

    Returns float32 [batch, n_letters]. Only the last position's logits are materialised
    (logits_to_keep=1), which matters with a 262k vocabulary.
    """
    position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)
    out = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                logits_to_keep=1, use_cache=False)
    last = out.logits[:, -1, :]
    return last.index_select(-1, letter_ids.to(last.device)).float()


def read_model_config(path: str) -> dict:
    for name in (CONFIG_NAME, *LEGACY_CONFIG_NAMES):
        p = os.path.join(path, name)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                return json.load(f)
    return {}


def write_model_config(path: str, cfg: dict) -> None:
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, CONFIG_NAME), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def _inner(model: nn.Module) -> nn.Module:
    while hasattr(model, "module") and not hasattr(model, "config"):  # DDP and similar wrappers
        model = model.module
    return model


def uncap_logits(model: nn.Module) -> bool:
    """Turn off Gemma's final logit soft cap (cap * tanh(x / cap), cap 30 on the real checkpoints) so letters are read
    from the logits under it: the cap squeezes every confident answer into the same narrow band below it. The top answer
    does not change (tanh is monotone); the probabilities do. Returns whether the model had a cap. A model trained this
    way says so in its chon_config.json ("uncap_logits": true) and is read the same way."""
    found = False
    cfgs = {id(c): c for c in [getattr(m, "config", None) for m in _inner(model).modules()] if hasattr(c, "to_dict")}
    for cfg in cfgs.values():   # every config in the tree: the wrapper's, the text model's (PEFT included)
        for c in (cfg, getattr(cfg, "text_config", None), cfg.get_text_config() if hasattr(cfg, "get_text_config") else None):
            if c is not None and getattr(c, "final_logit_softcapping", None) is not None:
                c.final_logit_softcapping = None
                found = True
    return found
