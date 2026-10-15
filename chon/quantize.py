"""Optional FP8 (dynamic activation + weight) quantisation of the text decoder with torchao.

Hopper/Blackwell GPUs only. The vision/audio towers and lm_head stay in bf16. Always re-run
`python -m chon.evaluate` after quantising: FP8 usually costs well under a point of accuracy
but must be checked against the configuration you actually serve.
"""

from __future__ import annotations

from torch import nn

from .modeling import NON_TEXT


def fp8_quantize(model: nn.Module) -> nn.Module:
    from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, PerRow, quantize_

    def keep(mod: nn.Module, fqn: str) -> bool:
        low = fqn.lower()
        return isinstance(mod, nn.Linear) and "lm_head" not in low and not any(x in low for x in NON_TEXT)

    quantize_(model, Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()), filter_fn=keep)
    return model
