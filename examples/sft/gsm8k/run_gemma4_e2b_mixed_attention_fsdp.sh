# Tested with NVIDIA CUDA GPUs.

set -xeuo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: run_gemma4_e2b_mixed_attention_fsdp.sh <nproc_per_node> <save_path> [other_configs...]"
    exit 1
fi

nproc_per_node=$1
save_path=$2

shift 2

MODEL_PATH=${MODEL_PATH:-google/gemma-4-E2B}
GLOBAL_BACKEND=${VERL_GEMMA4_GLOBAL_BACKEND:-sdpa}

export VERL_GEMMA4_GLOBAL_BACKEND=${GLOBAL_BACKEND}

torchrun --standalone --nnodes=1 --nproc_per_node="${nproc_per_node}" \
    -m verl.trainer.sft_trainer \
    data.train_files="${HOME}/data/gsm8k/train.parquet" \
    data.val_files="${HOME}/data/gsm8k/test.parquet" \
    data.messages_key=messages \
    data.train_batch_size=128 \
    data.use_dynamic_bsz=True \
    data.max_token_len_per_gpu=4096 \
    model.path="${MODEL_PATH}" \
    model.external_lib=verl.models.transformers.gemma4_mixed_attention \
    +model.override_config.attn_implementation=gemma4_mixed_fa2_sdpa \
    model.use_remove_padding=True \
    model.use_liger=True \
    model.use_fused_kernels=True \
    model.fused_kernel_options.impl_backend=triton \
    optim.lr=1e-5 \
    engine=fsdp \
    engine.strategy=fsdp2 \
    trainer.default_local_dir="${save_path}" \
    trainer.project_name=gsm8k-sft \
    trainer.experiment_name="gsm8k-sft-gemma4-e2b-mixed-${GLOBAL_BACKEND}" \
    trainer.total_epochs=2 \
    trainer.logger='["console","wandb"]' "$@"

