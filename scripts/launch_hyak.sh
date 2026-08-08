#!/bin/bash
# Launch Epsilon training on UW Hyak (klone) via SLURM.
#
#   sbatch scripts/launch_hyak.sh configs/train_rtx6k.yaml
#   sbatch scripts/launch_hyak.sh configs/train_rtx6k.yaml training.lr=2e-4
#
# Verified against `hyakalloc` on 2026-08-07: the rao account holds exactly one
# GPU partition, `gpu-rtx6k` — 4x NVIDIA Quadro RTX 6000, 24 GB each, 20 CPUs
# and 188 G of RAM in total. (There is no gpu-a40 under this account. The
# gpu-a100 / gpu-l40s rows hyakalloc prints belong to the `cse` account, which
# is a different allocation — do not submit there without checking you are
# entitled to it.)
#
# Quadro RTX 6000 is Turing (sm_75): **no bf16 tensor cores**. Configs for this
# partition must use mixed_precision=fp16; the trainer warns and falls back
# automatically, but set it explicitly so the intent is on the record.
#
# For more/better hardware, submit to the preemptible checkpoint partition,
# which is idle capacity from across the whole cluster (hyakalloc showed 66
# idle GPUs there on 2026-08-07). Command-line flags override the #SBATCH
# directives below, so no edit is needed:
#
#   sbatch --partition=ckpt-all --gpus=a40:4 scripts/launch_hyak.sh configs/train_unet.yaml
#
# Ampere/Ada cards there (a40, a100, l40s) DO support bf16, so pair that with
# training.mixed_precision=bf16. Checkpoint jobs are requeued after ~4 hours of
# runtime, so keep logging.ckpt_every small enough that a requeue costs minutes
# rather than hours — losing 10k steps of progress to preemption is the classic
# way to waste a week here.
#
#SBATCH --job-name=epsilon
#SBATCH --account=rao
#SBATCH --partition=gpu-rtx6k
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=180G
#SBATCH --gpus=4
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --chdir=/gscratch/rao/aaravs07/epsilon
#SBATCH --output=/gscratch/rao/aaravs07/epsilon/logs/slurm-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=aaravs07@uw.edu

set -euo pipefail

CONFIG=${1:?usage: sbatch launch_hyak.sh <config.yaml> [overrides...]}
shift || true

mkdir -p logs

# --- environment -----------------------------------------------------------
# No `module load cuda` on purpose: the venv's torch wheels bundle their own
# CUDA 13 runtime and find it via RPATH. Verified working on gpu-rtx6k
# (Quadro RTX 6000) 2026-08-08. Loading a cuda module here only risks skew.

if [[ ! -d .venv ]]; then
    echo "No .venv here. On the login node, run:"
    echo "  cd /gscratch/rao/aaravs07/epsilon"
    echo "  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt -e ."
    exit 1
fi
source .venv/bin/activate

export OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-16} / ${SLURM_GPUS:-4} ))
export TOKENIZERS_PARALLELISM=false
# Compute nodes have no outbound internet: log offline and `wandb sync` later.
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_DIR=${WANDB_DIR:-/gscratch/rao/aaravs07/epsilon/logs}
# Keep HF/torch caches off the small home quota (10 GB).
export HF_HOME=/gscratch/rao/aaravs07/.cache/huggingface
export TORCH_HOME=/gscratch/rao/aaravs07/.cache/torch

# --- auto-resume -----------------------------------------------------------
# Preemption on ckpt-* partitions requeues the job; pick up where we left off.
OUTPUT_DIR=$(python -c "
from eps.config import load_config
print(load_config('$CONFIG').logging.output_dir)")
RESUME_ARGS=()
if [[ -f "$OUTPUT_DIR/ckpt_latest.pt" ]]; then
    RESUME_ARGS=(--resume "$OUTPUT_DIR/ckpt_latest.pt")
    echo "Resuming from $OUTPUT_DIR/ckpt_latest.pt"
fi

# --- train -----------------------------------------------------------------
GPUS=${SLURM_GPUS:-${SLURM_GPUS_ON_NODE:-4}}
echo "Launching on $GPUS GPU(s): $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

srun torchrun \
    --standalone \
    --nproc_per_node="$GPUS" \
    scripts/train.py --config "$CONFIG" "${RESUME_ARGS[@]}" "$@"
