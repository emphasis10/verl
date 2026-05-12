# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Gemma 4 mixed attention backend for Hugging Face models.

This module is intended to be loaded through ``model.external_lib`` before
``AutoConfig.from_pretrained`` is called. It registers the attention
implementation name ``gemma4_mixed_fa2_sdpa`` with Transformers.

Gemma 4 E2B uses two attention shapes: sliding-window layers with
``head_dim=256`` and global/full layers with ``global_head_dim=512``.
FlashAttention 2 can handle the sliding layers but not the global 512-dim
heads, so this backend dispatches sliding layers to FA2 and global layers to a
configurable fallback backend.
"""

from __future__ import annotations

import importlib
import os
import warnings
from collections.abc import Callable
from typing import Optional

import torch

ATTN_IMPLEMENTATION_NAME = "gemma4_mixed_fa2_sdpa"
GLOBAL_BACKEND_ENV = "VERL_GEMMA4_GLOBAL_BACKEND"
FA512_MODULE_ENV = "VERL_GEMMA4_FA512_MODULE"
TRACE_ENV = "VERL_GEMMA4_MIXED_ATTN_TRACE"

_VALID_GLOBAL_BACKENDS = {"custom_fa512", "xformers", "sdpa_mem_efficient", "sdpa"}
_WARNED_MESSAGES: set[str] = set()


def _warn_once(message: str) -> None:
    if message not in _WARNED_MESSAGES:
        _WARNED_MESSAGES.add(message)
        warnings.warn(message, RuntimeWarning, stacklevel=3)


def _trace(message: str) -> None:
    if os.getenv(TRACE_ENV, "").lower() in {"1", "true", "yes"}:
        print(f"[{ATTN_IMPLEMENTATION_NAME}] {message}")


def _attention_registry():
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    return ALL_ATTENTION_FUNCTIONS


def _get_attention_forward(name: str) -> Callable:
    registry = _attention_registry()
    if hasattr(registry, "get_interface"):
        return registry.get_interface(name)
    return registry[name]


def _register_attention_forward(name: str, fn: Callable) -> None:
    registry = _attention_registry()
    if hasattr(registry, "register"):
        registry.register(name, fn)
    else:
        registry[name] = fn


def _is_gemma4_text_attention(module: torch.nn.Module) -> bool:
    if module.__class__.__name__ == "Gemma4TextAttention":
        return True

    config = getattr(module, "config", None)
    return getattr(config, "model_type", None) in {"gemma4_text", "gemma4"}


def _is_sliding_attention(module: torch.nn.Module, sliding_window: Optional[int]) -> bool:
    if getattr(module, "is_sliding", False):
        return True
    if getattr(module, "layer_type", None) == "sliding_attention":
        return True
    return sliding_window is not None


def _position_ids_from_kwargs(kwargs: dict) -> Optional[torch.Tensor]:
    position_ids = kwargs.get("position_ids")
    if position_ids is None:
        return None
    if not isinstance(position_ids, torch.Tensor):
        return None
    if position_ids.ndim == 3:
        # Some multimodal models use [ndim, batch, seq] position ids. The text
        # packed path only needs one monotonic sequence coordinate.
        position_ids = position_ids[-1]
    if position_ids.ndim != 2:
        return None
    return position_ids


def _segment_ids_from_position_ids(position_ids: torch.Tensor) -> torch.Tensor:
    resets = torch.ones_like(position_ids, dtype=torch.bool)
    resets[:, 1:] = position_ids[:, 1:] <= position_ids[:, :-1]
    return resets.cumsum(dim=-1)


def _packed_sequence_lengths(position_ids: torch.Tensor) -> list[int]:
    """Infer per-sequence lengths from packed position ids.

    ``verl`` remove-padding mode commonly packs a batch into one row with
    position ids resetting at sequence boundaries.
    """

    segment_ids = _segment_ids_from_position_ids(position_ids)
    lengths: list[int] = []
    for row_segment_ids in segment_ids:
        unique_ids, counts = torch.unique_consecutive(row_segment_ids, return_counts=True)
        del unique_ids
        lengths.extend(int(count) for count in counts.detach().cpu().tolist())
    return lengths


def _build_causal_or_packed_mask(
    query: torch.Tensor,
    key: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    *,
    sliding_window: Optional[int],
    position_ids: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    if attention_mask is not None:
        return attention_mask
    if position_ids is None:
        return None

    query_len = query.shape[2]
    key_len = key.shape[2]
    if query_len != key_len or position_ids.shape[-1] != query_len:
        return None

    batch = query.shape[0]
    if position_ids.shape[0] != batch:
        return None

    token_ids = torch.arange(query_len, device=query.device)
    causal = token_ids[None, :, None] >= token_ids[None, None, :]

    segment_ids = _segment_ids_from_position_ids(position_ids.to(device=query.device))
    same_segment = segment_ids[:, :, None] == segment_ids[:, None, :]

    mask = causal & same_segment
    if sliding_window is not None:
        local = token_ids[None, None, :] > token_ids[None, :, None] - sliding_window
        mask = mask & local

    return mask[:, None, :, :]


def _sdpa_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    *,
    sliding_window: Optional[int],
    position_ids: Optional[torch.Tensor],
    force_mem_efficient: bool,
    **kwargs,
):
    sdpa_attention_forward = _get_attention_forward("sdpa")
    sdpa_mask = _build_causal_or_packed_mask(
        query,
        key,
        attention_mask,
        sliding_window=sliding_window,
        position_ids=position_ids,
    )

    # Once we build an explicit mask, SDPA must not also apply its implicit
    # causal mask. This preserves packed sequence boundaries.
    if sdpa_mask is not None:
        kwargs["is_causal"] = False

    if not force_mem_efficient:
        return sdpa_attention_forward(module, query, key, value, sdpa_mask, **kwargs)

    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
            return sdpa_attention_forward(module, query, key, value, sdpa_mask, **kwargs)
    except Exception as exc:
        _warn_once(f"sdpa_mem_efficient backend failed; falling back to sdpa. Error: {exc}")
        return sdpa_attention_forward(module, query, key, value, sdpa_mask, **kwargs)


def _repeat_kv_for_xformers(key: torch.Tensor, value: torch.Tensor, num_query_heads: int):
    if key.shape[1] == num_query_heads:
        return key, value
    if num_query_heads % key.shape[1] != 0:
        raise ValueError(f"Cannot repeat {key.shape[1]} KV heads to {num_query_heads} query heads.")

    repeats = num_query_heads // key.shape[1]
    return key.repeat_interleave(repeats, dim=1), value.repeat_interleave(repeats, dim=1)


def _xformers_global_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    *,
    position_ids: Optional[torch.Tensor],
    **kwargs,
):
    if attention_mask is not None:
        raise ValueError("xformers global backend only supports mask-free or position_ids-packed inputs.")

    import xformers.ops as xops

    key, value = _repeat_kv_for_xformers(key, value, query.shape[1])

    # xFormers expects [batch, seq, heads, dim].
    q = query.transpose(1, 2).contiguous()
    k = key.transpose(1, 2).contiguous()
    v = value.transpose(1, 2).contiguous()

    batch, seqlen, num_heads, head_dim = q.shape
    del module, num_heads, head_dim

    if position_ids is None:
        seqlens = [seqlen] * batch
    else:
        seqlens = _packed_sequence_lengths(position_ids)

    # Flatten batch and packed segments into one block-diagonal attention call.
    q = q.reshape(1, batch * seqlen, q.shape[2], q.shape[3])
    k = k.reshape(1, batch * seqlen, k.shape[2], k.shape[3])
    v = v.reshape(1, batch * seqlen, v.shape[2], v.shape[3])

    attn_bias = xops.fmha.attn_bias.BlockDiagonalCausalMask.from_seqlens(seqlens)
    output = xops.memory_efficient_attention(
        q,
        k,
        v,
        attn_bias=attn_bias,
        p=kwargs.get("dropout", 0.0),
        scale=kwargs.get("scaling"),
    )
    output = output.reshape(batch, seqlen, output.shape[2], output.shape[3]).contiguous()
    return output, None


def _custom_fa512_global_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    **kwargs,
):
    module_name = os.getenv(FA512_MODULE_ENV, "gemma4_fa512")
    custom_module = importlib.import_module(module_name)
    forward = getattr(custom_module, "gemma4_fa512_attention_forward", None)
    if forward is None:
        raise AttributeError(
            f"{module_name} must define gemma4_fa512_attention_forward("
            "module, query, key, value, attention_mask, **kwargs)."
        )
    return forward(module, query, key, value, attention_mask, **kwargs)


def _global_backend_name() -> str:
    backend = os.getenv(GLOBAL_BACKEND_ENV, "sdpa").lower()
    if backend not in _VALID_GLOBAL_BACKENDS:
        _warn_once(
            f"Unsupported {GLOBAL_BACKEND_ENV}={backend!r}; expected one of {sorted(_VALID_GLOBAL_BACKENDS)}. "
            "Falling back to sdpa."
        )
        return "sdpa"
    return backend


def _run_global_backend(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    *,
    position_ids: Optional[torch.Tensor],
    sliding_window: Optional[int],
    **kwargs,
):
    backend = _global_backend_name()
    fallback_errors: list[str] = []

    if backend == "custom_fa512":
        try:
            _trace(f"layer={getattr(module, 'layer_idx', '?')} global backend=custom_fa512")
            return _custom_fa512_global_forward(module, query, key, value, attention_mask, **kwargs)
        except Exception as exc:
            fallback_errors.append(f"custom_fa512 failed: {exc}")
            backend = "xformers"

    if backend == "xformers":
        try:
            _trace(f"layer={getattr(module, 'layer_idx', '?')} global backend=xformers")
            return _xformers_global_forward(
                module,
                query,
                key,
                value,
                attention_mask,
                position_ids=position_ids,
                **kwargs,
            )
        except Exception as exc:
            fallback_errors.append(f"xformers failed: {exc}")
            backend = "sdpa_mem_efficient"

    if backend == "sdpa_mem_efficient":
        _trace(f"layer={getattr(module, 'layer_idx', '?')} global backend=sdpa_mem_efficient")
        if fallback_errors:
            _warn_once("; ".join(fallback_errors) + "; falling back to sdpa_mem_efficient.")
        return _sdpa_forward(
            module,
            query,
            key,
            value,
            attention_mask,
            sliding_window=sliding_window,
            position_ids=position_ids,
            force_mem_efficient=True,
            **kwargs,
        )

    _trace(f"layer={getattr(module, 'layer_idx', '?')} global backend=sdpa")
    if fallback_errors:
        _warn_once("; ".join(fallback_errors) + "; falling back to sdpa.")
    return _sdpa_forward(
        module,
        query,
        key,
        value,
        attention_mask,
        sliding_window=sliding_window,
        position_ids=position_ids,
        force_mem_efficient=False,
        **kwargs,
    )


def gemma4_mixed_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    sliding_window: Optional[int] = None,
    **kwargs,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Dispatch Gemma4 sliding layers to FA2 and global layers to fallback backends."""

    kwargs["dropout"] = dropout
    kwargs["scaling"] = scaling
    position_ids = _position_ids_from_kwargs(kwargs)

    if not _is_gemma4_text_attention(module):
        return _sdpa_forward(
            module,
            query,
            key,
            value,
            attention_mask,
            sliding_window=sliding_window,
            position_ids=position_ids,
            force_mem_efficient=False,
            **kwargs,
        )

    if _is_sliding_attention(module, sliding_window):
        flash_attention_forward = _get_attention_forward("flash_attention_2")
        try:
            _trace(f"layer={getattr(module, 'layer_idx', '?')} sliding backend=flash_attention_2")
            return flash_attention_forward(
                module,
                query,
                key,
                value,
                attention_mask,
                sliding_window=sliding_window,
                **kwargs,
            )
        except Exception as exc:
            _warn_once(f"Gemma4 sliding FA2 backend failed; falling back to sdpa. Error: {exc}")
            return _sdpa_forward(
                module,
                query,
                key,
                value,
                attention_mask,
                sliding_window=sliding_window,
                position_ids=position_ids,
                force_mem_efficient=False,
                **kwargs,
            )

    return _run_global_backend(
        module,
        query,
        key,
        value,
        attention_mask,
        position_ids=position_ids,
        sliding_window=None,
        **kwargs,
    )


def register_gemma4_mixed_attention() -> None:
    _register_attention_forward(ATTN_IMPLEMENTATION_NAME, gemma4_mixed_attention_forward)


register_gemma4_mixed_attention()
