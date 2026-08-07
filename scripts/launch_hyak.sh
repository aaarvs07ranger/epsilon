#!/bin/bash
# Launch Epsilon training on UW Hyak (klone) via SLURM.
#
#   sbatch scripts/launch_hyak.sh configs/train_unet.yaml
#   sbatch scripts/launch_hyak.sh configs/train_dit.yaml training.lr=2e-4
#
# Account/partition are set for the Rao lab (group u_hyak_rao). VERIFY your
# actual allocation before the first real run:
#     hyakalloc                 # shows accounts + partitions + idle GPUs
# If `hyakalloc` lists a different account name or GPU type, edit the two
# #SBATCH lines below to match. `ckpt-all` is the preemptible checkpoint
# partition — free idle capacity across the cluster, jobs can be requeued at
# any time, which this script handles by auto-resuming.
#
#SBATCH --job-name=epsilon
#SBATCH --account=rao
#SBATCH --partition=gpu-a40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
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
# Hyak provides CUDA via modules; the venv carries torch and the rest.
module load cuda/12.4.1 2>/dev/null || module load cuda 2>/dev/null || true

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
