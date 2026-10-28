"""Shared-prefix read: the questions of one request share everything up to where their prompts differ (the chat
template's start, a header model's question list and the state), so that part is run through the model once and every
question reads it from the key/value cache.

    prefix pass   [ <template> (question list) state \\n\\n---\\n ]       -> keys/values per layer (kept)
    suffix pass   [ question 1 ][ question 2 ] ... [ question k ]          -> one packed row, one forward pass

In the suffix pass each question attends to the cached prefix and to its own tokens only (a block mask), with
positions continuing from the prefix, so every question sees exactly the tokens it would see on its own: the letter
logits are the ones the plain path computes, up to floating-point rounding of different kernel shapes (1e-6 in float32
on the tiny test model, bf16 rounding on a GPU). Sliding-window layers get the model's own window rule (key position >
query position - window).

Works for decoders whose layers are sliding_attention / full_attention (Gemma 4 and 3, including KV-shared layers and
k = v layers) with the sdpa or eager attention; anything else raises Unsupported and the caller reads plainly.

Memory (see README, "Shared-prefix read"): the prefix pass runs under torch.inference_mode (no activations kept even
when the weights still want gradients); a sliding layer keeps a copy of its last window - 1 prefix positions, not a
view that would pin the whole prefix's keys and values (for Gemma 4 31B at 50,000 tokens that is 0.8 GB kept instead
of 41 GB); and the suffix passes are packed up to a token budget (packed_batches).
"""

from __future__ import annotations

from typing import Sequence

import torch


class Unsupported(RuntimeError):
    pass


def common_prefix(seqs: Sequence[Sequence[int]]) -> int:
    """Length of the longest common token prefix. That of all the sequences is that of the lexicographically first
    and last (compared in C), so a 50,000-token state with 20 questions costs one Python pass, not twenty."""
    if not seqs:
        return 0
    lo, hi = min(map(list, seqs)), max(map(list, seqs))
    n = min(len(lo), len(hi))
    for i in range(n):
        if lo[i] != hi[i]:
            return i
    return n


def _text_config(model):
    cfg = model.config
    return cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg


def layer_types(model) -> list[str]:
    tc = _text_config(model)
    types = list(getattr(tc, "layer_types", None) or [])
    if not types or any(t not in ("sliding_attention", "full_attention") for t in types):
        raise Unsupported(f"layer types {sorted(set(types))} (needs sliding_attention / full_attention)")
    return types


def attn_impl(model) -> str:
    tc = _text_config(model)
    impl = getattr(tc, "_attn_implementation", None) or getattr(model.config, "_attn_implementation", None) or "eager"
    if impl not in ("sdpa", "eager"):
        raise Unsupported(f"attention implementation {impl!r}")
    return impl


def check_supported(model) -> None:
    """Raise Unsupported when the shared-prefix read cannot serve this model (checked from its config only)."""
    layer_types(model)
    attn_impl(model)


def _prefix_cache(model):
    """The cache for the prefix pass: transformers' own (so the model builds its usual masks), with each sliding layer
    keeping a copy of its window instead of a view of the full keys/values. None = let the model make one."""
    try:
        from transformers import DynamicCache

        cache = DynamicCache(config=model.config)
    except Exception:  # noqa: BLE001  (another transformers version: the model makes its own cache)
        return None
    for layer in getattr(cache, "layers", None) or []:
        if getattr(layer, "is_sliding", False):
            def update(key_states, value_states, *args, _layer=layer, _orig=layer.update, **kwargs):
                out = _orig(key_states, value_states, *args, **kwargs)
                if torch.is_tensor(_layer.keys) and _layer.keys.numel():
                    _layer.keys, _layer.values = _layer.keys.clone(), _layer.values.clone()
                return out
            layer.update = update
    return cache


class _ReadOnlyPrefix:
    """A cache stand-in for the suffix pass: each layer's keys/values are the cached prefix followed by the new
    tokens; nothing is stored, so the same prefix serves any number of passes."""

    is_compileable = False

    def __init__(self, keys: list, values: list, length: int):
        self.keys, self.values, self.length = keys, values, length

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        pk = self.keys[layer_idx] if layer_idx < len(self.keys) else None
        if pk is None:
            raise Unsupported(f"no cached prefix for layer {layer_idx}")
        pv = self.values[layer_idx]
        return torch.cat([pk, key_states], dim=-2), torch.cat([pv, value_states], dim=-2)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.length

    def __len__(self):
        return len(self.keys)


class SharedPrefix:
    """Keys/values of one prefix (a list of token ids), ready for any number of suffix passes."""

    def __init__(self, model, tokens: Sequence[int], device):
        self.model = model
        self.tokens = list(tokens)
        self.length = len(self.tokens)
        self.device = device
        self.types = layer_types(model)
        self.impl = attn_impl(model)
        self.window = int(getattr(_text_config(model), "sliding_window", 0) or 0)
        ids = torch.tensor([self.tokens], dtype=torch.long, device=device)
        # inference mode: a model whose weights still want gradients would otherwise keep every activation of the
        # prefix pass alive through the cached keys (the training repository measured 105-131 GB against 60-65 GB)
        with torch.inference_mode():
            cache = _prefix_cache(model)
            out = model(input_ids=ids, attention_mask=torch.ones_like(ids), past_key_values=cache, use_cache=True,
                        logits_to_keep=1)
        cache = out.past_key_values
        if cache is None:
            raise Unsupported("the model returned no cache")
        self.keys, self.values = [], []
        for i in range(len(self.types)):
            k = v = None
            layers = getattr(cache, "layers", None)
            if layers is not None and i < len(layers):
                k, v = getattr(layers[i], "keys", None), getattr(layers[i], "values", None)
            elif hasattr(cache, "key_cache") and i < len(cache.key_cache):
                k, v = cache.key_cache[i], cache.value_cache[i]
            if k is not None and (not torch.is_tensor(k) or k.numel() == 0):
                k = v = None
            self.keys.append(k)
            self.values.append(v)
        # how many prefix positions each layer type keeps (sliding layers: the last window - 1)
        self.kept = {}
        for t, k in zip(self.types, self.keys):
            if k is not None:
                self.kept.setdefault(t, k.shape[-2])
        for t in set(self.types):
            self.kept.setdefault(t, self.length if t == "full_attention" else min(self.length, max(self.window - 1, 0)))
        del out, cache

    def nbytes(self) -> int:
        """GPU memory held by the cached keys/values (storage, so a view that pins more is counted in full)."""
        seen, total = set(), 0
        for t in self.keys + self.values:
            if t is not None and t.untyped_storage().data_ptr() not in seen:
                seen.add(t.untyped_storage().data_ptr())
                total += t.untyped_storage().nbytes()
        return total

    # ------------------------------------------------------------------ masks
    def _mask(self, lengths: Sequence[int], kind: str, dtype):
        """[1, 1, S, kept + S]: query token (question j, offset t) may see prefix key p when p <= P + t (always) and,
        for sliding layers, P + t - p < window; and key (question j, offset u <= t) under the same window rule."""
        dev = self.device
        P, S = self.length, int(sum(lengths))
        kept = self.kept[kind]
        seg = torch.repeat_interleave(torch.arange(len(lengths), device=dev),
                                      torch.tensor(list(lengths), device=dev))
        starts = torch.cumsum(torch.tensor([0] + list(lengths[:-1]), device=dev), 0)
        off = torch.arange(S, device=dev) - starts[seg]
        qpos = P + off  # logical position of each suffix token
        kpos_prefix = torch.arange(P - kept, P, device=dev)
        # prefix keys: all earlier by construction
        m_prefix = torch.ones(S, kept, dtype=torch.bool, device=dev)
        if kind == "sliding_attention" and self.window:
            m_prefix &= (qpos[:, None] - kpos_prefix[None, :]) < self.window
        m_suffix = (seg[:, None] == seg[None, :]) & (off[None, :] <= off[:, None])
        if kind == "sliding_attention" and self.window:
            m_suffix &= (qpos[:, None] - qpos[None, :]).abs() < self.window
        m = torch.cat([m_prefix, m_suffix], dim=1)[None, None]
        if self.impl == "eager":
            z = torch.zeros((), dtype=dtype, device=dev)
            return torch.where(m, z, torch.tensor(torch.finfo(dtype).min, dtype=dtype, device=dev))
        return m

    # ------------------------------------------------------------------ readout
    @torch.inference_mode()
    def letter_logits(self, suffixes: Sequence[Sequence[int]], letter_ids: torch.Tensor) -> torch.Tensor:
        """Letter logits at the last token of each suffix, all suffixes in one packed pass. float32 [k, n_letters]."""
        lengths = [len(s) for s in suffixes]
        if min(lengths) < 1:
            raise ValueError("every suffix needs at least one token (the reply position)")
        dev = self.device
        flat = [t for s in suffixes for t in s]
        ids = torch.tensor([flat], dtype=torch.long, device=dev)
        pos = torch.cat([torch.arange(self.length, self.length + n, device=dev) for n in lengths])[None]
        dtype = next(self.model.parameters()).dtype
        masks = {kind: self._mask(lengths, kind, dtype) for kind in set(self.types)}
        cache = _ReadOnlyPrefix(self.keys, self.values, self.length)
        last = torch.cumsum(torch.tensor(lengths, device=dev), 0) - 1
        out = self.model(input_ids=ids, attention_mask=masks, position_ids=pos, past_key_values=cache,
                         use_cache=True, logits_to_keep=last)
        logits = out.logits[0]
        return logits.index_select(-1, letter_ids.to(logits.device)).float()


# the largest suffix-pass mask, in query x key cells (a full-attention layer sees the whole prefix): 2**30 cells is
# 1 GiB as booleans (2 GiB once attention turns it into a bf16 bias), reached only past ~123,000 prefix tokens with
# the default 8,192-token pack
MAX_MASK_CELLS = 2 ** 30


def pack_budget(tokens: int, prefix_kept: int, max_cells: int = MAX_MASK_CELLS) -> int:
    """Suffix tokens per packed pass: `tokens`, lowered for a very long prefix so the mask stays under max_cells."""
    return min(int(tokens), max(256, max_cells // max(prefix_kept + int(tokens), 1)))


def packed_batches(lengths: Sequence[int], budget: int) -> list[list[int]]:
    """Indices grouped so each group's total suffix length stays within budget (one oversize suffix alone)."""
    groups, cur, size = [], [], 0
    for i, n in enumerate(lengths):
        if cur and size + n > budget:
            groups.append(cur)
            cur, size = [], 0
        cur.append(i)
        size += n
    if cur:
        groups.append(cur)
    return groups
