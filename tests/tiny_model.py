"""A tiny random Gemma 4 (the multimodal class, like the real checkpoints) with a tiny BPE tokenizer, on the CPU.

The same configuration as the training repository's test fixture, with two changes so that golden values made with
the training code stay valid across library versions: the weights are drawn from torch generators seeded by each
parameter's name (not from transformers' initialisers, which change between versions), and are scaled so that the
letter distributions are far from uniform (a parity check on near-uniform distributions would prove little).

Imports only `chon.prompt` / `chon.schema` (identical in the training and serving code), so tests/make_golden.py can
build the same folder with the training repository first on the path.

tiny_adapter() puts a small LoRA adapter (the release adapter's shape: rank 4 on the text decoder's projections, as the
training code's chon.eval.smoke.tiny_adapter) on a copy of the tiny model, its weights drawn the same way (seeded by
parameter name, not peft's initialiser), for the pair-read cases.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import zlib

CHAT_TEMPLATE = ("{{ bos_token }}{% for m in messages %}<start_of_turn>{{ m['role'] }}\n{{ m['content'] }}"
                 "<end_of_turn>\n{% endfor %}{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}")
WEIGHT_STD = 0.5
LORA_STD = 0.3    # the adapter moves the letter distributions clearly away from the base model's
LORA_PROJ = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
NON_TEXT = ("vision", "audio", "multi_modal", "embed_vision", "embed_audio", "visual")


def make_items(n: int, seed: int = 0, prefix: str = "t") -> list[dict]:
    """The training fixture's toy task over all three question types (its tokenizer corpus)."""
    rng = random.Random(seed)
    items = []
    for i in range(n):
        value, limit = rng.randint(1, 12), rng.randint(2, 9)
        kind = i % 3
        if kind == 0:
            items.append({"id": f"{prefix}{i}", "state": f"Parcel {i}: weight {value} kg, limit {limit} kg.",
                          "question": {"type": "noul", "instructions": "Is the parcel within the limit?",
                                       "criteria": {"true": "weight <= limit", "false": "weight > limit"}}})
        elif kind == 1:
            labels = ["refund", "replace", "escalate", "close"]
            exp = labels[value % 4]
            items.append({"id": f"{prefix}{i}", "state": {"ticket": i, "request": exp, "note": f"code {value}"},
                          "question": {"type": "choice", "instructions": "Which action does the ticket request?",
                                       "criteria": {lab: f"the customer wants to {lab}" for lab in labels}}})
        else:
            items.append({"id": f"{prefix}{i}",
                          "state": [{"role": "user", "content": f"Mức độ {value}"},
                                    {"role": "assistant", "content": "Đã ghi nhận."}],
                          "question": {"type": "score", "instructions": "How severe is it?",
                                       "criteria": ["low", "medium", "high"]}})
    return items


def build_tiny_model(out: str) -> str:
    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import (Gemma4Config, Gemma4ForConditionalGeneration, Gemma4TextConfig,
                              Gemma4VisionConfig, PreTrainedTokenizerFast)

    from chon.prompt import question_text, state_text
    from chon.schema import labels_for

    os.makedirs(out, exist_ok=True)
    corpus = [state_text(it["state"]) + "\n\n---\n" + question_text(it["question"], labels_for(it["question"]))
              for it in make_items(60)]
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(corpus * 5, trainers.BpeTrainer(
        vocab_size=700, special_tokens=["<pad>", "<eos>", "<bos>", "<start_of_turn>", "<end_of_turn>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    ft = PreTrainedTokenizerFast(tokenizer_object=tok, bos_token="<bos>", eos_token="<eos>", pad_token="<pad>")
    ft.chat_template = CHAT_TEMPLATE
    ft.save_pretrained(out)

    text = Gemma4TextConfig(vocab_size=len(ft), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                            num_attention_heads=2, num_key_value_heads=1, head_dim=32, global_head_dim=32,
                            layer_types=["sliding_attention", "full_attention"], sliding_window=64,
                            vocab_size_per_layer_input=len(ft), hidden_size_per_layer_input=8,
                            max_position_embeddings=8192, pad_token_id=0, eos_token_id=1, bos_token_id=2)
    vision = Gemma4VisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
                                num_key_value_heads=2, head_dim=16)
    cfg = Gemma4Config(text_config=text.to_dict(), vision_config=vision.to_dict(), audio_config=None)
    torch.manual_seed(0)
    model = Gemma4ForConditionalGeneration(cfg)
    with torch.no_grad():
        for name, p in sorted(model.named_parameters()):
            g = torch.Generator().manual_seed(zlib.crc32(name.encode()))
            p.copy_(torch.randn(p.shape, generator=g, dtype=torch.float32).to(p.dtype) * WEIGHT_STD)
    model.save_pretrained(out)
    return out


def fingerprint(model_dir: str) -> dict:
    """What the golden values depend on: the tokenizer's vocabulary and merges, and the weights."""
    import torch
    from safetensors.torch import load_file

    tj = json.load(open(os.path.join(model_dir, "tokenizer.json"), encoding="utf-8"))
    vocab = hashlib.sha256(json.dumps([tj["model"]["vocab"], tj["model"]["merges"]], sort_keys=True,
                                      ensure_ascii=False).encode()).hexdigest()[:16]
    sd = {}
    for f in sorted(os.listdir(model_dir)):
        if f.endswith(".safetensors"):
            sd.update(load_file(os.path.join(model_dir, f)))
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(sd[k].to(torch.float32).contiguous().numpy().tobytes())
    return {"tokenizer": vocab, "weights": h.hexdigest()[:16]}


def lora_targets(model) -> list[str]:
    """Full names of the text decoder's projection layers (vision and audio towers left out), as the training code's
    chon.core.modeling.lora_target_modules picks them."""
    import torch

    return [name for name, mod in model.named_modules()
            if isinstance(mod, torch.nn.Linear) and name.rsplit(".", 1)[-1] in LORA_PROJ
            and not any(x in name.lower() for x in NON_TEXT)]


def tiny_adapter(model):
    """A copy of `model` with a rank-4 LoRA adapter ("default") on lora_targets, unmerged (a peft PeftModel), its
    weights drawn from generators seeded by each parameter's name. The model passed in is not changed."""
    import copy

    import torch
    from peft import LoraConfig, get_peft_model

    m = copy.deepcopy(model)
    peft = get_peft_model(m, LoraConfig(r=4, lora_alpha=8, target_modules=lora_targets(m), init_lora_weights=False))
    with torch.no_grad():
        for name, p in sorted(peft.named_parameters()):
            if "lora_" in name:
                g = torch.Generator().manual_seed(zlib.crc32(name.encode()))
                p.copy_(torch.randn(p.shape, generator=g, dtype=torch.float32).to(p.dtype) * LORA_STD)
    peft.eval()
    peft.requires_grad_(False)
    return peft


def adapter_fingerprint(peft) -> str:
    """What the pair cases' golden values depend on besides the tiny model: the adapter's weights."""
    import torch

    h = hashlib.sha256()
    for name, p in sorted(peft.named_parameters()):
        if "lora_" in name:
            h.update(name.encode())
            h.update(p.detach().to(torch.float32).contiguous().numpy().tobytes())
    return h.hexdigest()[:16]
