# Gemma 4 Mixed Attention for SFT

Gemma 4 E2B uses sliding-window attention layers with `head_dim=256` and
global/full attention layers with `global_head_dim=512`. FlashAttention 2 can
serve the sliding layers, but the global layers require a fallback backend.

verl provides an external-lib attention implementation named
`gemma4_mixed_fa2_sdpa`:

```text
sliding attention -> flash_attention_2
global attention  -> configurable fallback backend
```

## Usage

Load the registration module before the Hugging Face config is created:

```bash
torchrun --standalone --nnodes=1 --nproc-per-node="${NUM_GPUS}" \
  -m verl.trainer.sft_trainer \
  model.path=google/gemma-4-E2B \
  model.external_lib=verl.models.transformers.gemma4_mixed_attention \
  +model.override_config.attn_implementation=gemma4_mixed_fa2_sdpa \
  model.use_remove_padding=True \
  model.use_liger=True \
  model.use_fused_kernels=True \
  model.fused_kernel_options.impl_backend=triton \
  engine=fsdp \
  engine.strategy=fsdp2 \
  data.use_dynamic_bsz=True
```

An example script is available at
`examples/sft/gsm8k/run_gemma4_e2b_mixed_attention_fsdp.sh`.

## Global Backend Selection

The global layer backend is selected with `VERL_GEMMA4_GLOBAL_BACKEND`.

| Value | Behavior |
| --- | --- |
| `sdpa` | Default PyTorch SDPA fallback. |
| `sdpa_mem_efficient` | Forces PyTorch efficient SDPA when available, otherwise falls back to `sdpa`. |
| `xformers` | Tries xFormers block-diagonal causal attention, then falls back to `sdpa_mem_efficient` and `sdpa`. |
| `custom_fa512` | Calls a user-provided FA512 module, then falls back to `xformers`, `sdpa_mem_efficient`, and `sdpa`. |

For `custom_fa512`, set `VERL_GEMMA4_FA512_MODULE` to a module exposing:

```python
def gemma4_fa512_attention_forward(module, query, key, value, attention_mask, **kwargs):
    ...
```

## Benchmark Matrix

Run each sequence length with the same data, token budget, GPU count, and
checkpoint settings.

| Variant | Settings |
| --- | --- |
| all-SDPA | `+model.override_config.attn_implementation=sdpa` |
| mixed SDPA | `gemma4_mixed_fa2_sdpa`, `VERL_GEMMA4_GLOBAL_BACKEND=sdpa` |
| mixed efficient SDPA | `gemma4_mixed_fa2_sdpa`, `VERL_GEMMA4_GLOBAL_BACKEND=sdpa_mem_efficient` |
| mixed xFormers | `gemma4_mixed_fa2_sdpa`, `VERL_GEMMA4_GLOBAL_BACKEND=xformers` |

Suggested sequence lengths: 2k, 4k, 8k, and 16k. Record:

- step time
- tokens/sec
- peak memory
- GPU utilization
- attention kernel time
- lm-head/logprob kernel time
- FSDP communication time

Enable routing traces only for short smoke tests:

```bash
export VERL_GEMMA4_MIXED_ATTN_TRACE=1
```

## Notes

- The mixed backend preserves `use_remove_padding=True` by building an explicit
  block-diagonal causal mask for SDPA fallbacks when packed `position_ids` are
  available.
- Sliding layers use FA2 first and fall back to SDPA if FA2 is unavailable.
- This module does not implement the FA512 CUDA/Triton kernel itself. It
  provides the production integration slot for such a kernel through
  `VERL_GEMMA4_GLOBAL_BACKEND=custom_fa512`.

