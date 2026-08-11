#!/usr/bin/env bash
# One-shot setup for a RunPod (or any Linux/CUDA) box: verify the GPU, install
# Epsilon, fetch ImageNet-64, and launch the U-Net and DiT arms on two GPUs.
#
# BOOTSTRAP — on a fresh pod you do not have the repo yet, and this script is
# inside it. It clones the repo itself, so run it straight off GitHub:
#
#   curl -fsSL https://raw.githubusercontent.com/aaarvs07ranger/epsilon/main/scripts/setup_runpod.sh | bash
#
# Once the repo exists (or from a laptop checkout):
#
#   bash scripts/setup_runpod.sh              # full: verify -> install -> data -> launch
#   bash scripts/setup_runpod.sh --no-launch  # stop after the data is ready
#   bash scripts/setup_runpod.sh --verify     # just the preflight checks, then exit
#
# To pass a flag through the curl form:
#   curl -fsSL <url> | bash -s -- --verify
#
# Expected timeline on a 2xH100 box:
#   install ~3 min, dataset ~25 min, then ~6 h of training for BOTH arms in
#   parallel (one GPU each). Budget roughly $40 at typical H100 pricing.
#
# WHY TWO SINGLE-GPU PROCESSES AND NOT torchrun/DDP: the trainer's DDP path has
# never been executed — Hyak was abandoned before any training run happened.
# The single-process path has verified hours behind it. Two independent
# processes give us both models in the same wall-clock with zero DDP risk.

set -euo pipefail

REPO="${REPO:-https://github.com/aaarvs07ranger/epsilon}"
WORK="${WORK:-/workspace}"
DIR="${DIR:-$WORK/epsilon}"
MODE="${1:-}"

say() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
die() { printf '\n\033[1;31mFATAL: %s\033[0m\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- preflight --
say "GPU preflight"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found — is this a GPU box?"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')
echo "GPUs visible: $NGPU"
[ "$NGPU" -ge 2 ] || echo "WARNING: fewer than 2 GPUs — the two arms will have to run sequentially."

# CLAUDE.md 10.11: torch 2.13 ships a CUDA 13 build that needs driver r580+.
# On a container that already has a working torch we reuse it rather than
# risk installing one the driver cannot run. This check is the whole reason
# that trap is not a problem here.
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
echo "Driver major version: $DRV"

say "Disk check"
df -h "$WORK" | tail -1
AVAIL=$(df -Pk "$WORK" | tail -1 | awk '{print int($4/1048576)}')
echo "Free under $WORK: ${AVAIL} GB"
[ "$AVAIL" -ge 60 ] || die "Need >=60 GB free ($WORK has ${AVAIL} GB). The fetch peaks at ~32 GB (15.7 GB scratch + 15.7 GB shards) and checkpoints add ~30 GB. Resize the volume."

[ "$MODE" = "--verify" ] && { say "Preflight only — stopping here."; exit 0; }

# ------------------------------------------------------------------ install --
say "Fetching the repo"
mkdir -p "$WORK"

# Keep every cache on the PERSISTENT volume, not the container disk. $HOME is
# /root, which lives on the (smaller, ephemeral) container disk — the same
# class of mistake as Hyak's 10 GB home quota, which launch_hyak.sh solves the
# same way. The HF cache self-cleans as it decodes (the fetcher unlinks each
# parquet), so this is insurance rather than a hard requirement, but it also
# means the caches survive a pod stop/start.
export HF_HOME="$WORK/.cache/huggingface"
export TORCH_HOME="$WORK/.cache/torch"
export PIP_CACHE_DIR="$WORK/.cache/pip"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$PIP_CACHE_DIR"
if [ -d "$DIR/.git" ]; then git -C "$DIR" pull --ff-only; else git clone "$REPO" "$DIR"; fi
cd "$DIR"

# The repo MUST contain eps/data/. It was missing from every commit until
# 2026-08-11 (CLAUDE.md 10.17); if this clone predates that push, stop now
# rather than fail confusingly 25 minutes later after the data fetch.
[ -f eps/data/imagenet.py ] && [ -f eps/data/imagenet_classes.txt ] \
  || die "This clone has no eps/data/ — the fix for CLAUDE.md 10.17 has not been pushed. Push it from the laptop first, or nothing below will run."

say "Python environment"
# --system-site-packages so we inherit the container's torch, which is already
# matched to its driver. Installing our own could reintroduce the CUDA 13 trap.
python3 -m venv --system-site-packages .venv
./.venv/bin/python -m pip install -qU pip

if ./.venv/bin/python -c "import torch" 2>/dev/null; then
  echo "Reusing the container's torch: $(./.venv/bin/python -c 'import torch;print(torch.__version__)')"
  ./.venv/bin/pip install -q --no-deps -e .
  ./.venv/bin/pip install -q numpy pyyaml pillow pyarrow "huggingface_hub>=0.25" pytest torch-fidelity clean-fid
else
  echo "No torch in the container — installing the pinned stack."
  ./.venv/bin/pip install -q -e ".[data,dev,eval,web]"
fi

say "Verifying torch sees the GPU"
./.venv/bin/python - <<'PY'
import sys, torch
print("torch     ", torch.__version__)
print("cuda build", torch.version.cuda)
ok = torch.cuda.is_available()
print("cuda avail", ok)
if not ok:
    sys.exit("torch cannot see the GPU. If torch was just installed, this is almost certainly "
             "the CUDA-13-build vs driver mismatch in CLAUDE.md 10.11 — install a build matching "
             "the driver, e.g. pip install torch --index-url https://download.pytorch.org/whl/cu124")
for i in range(torch.cuda.device_count()):
    cap = torch.cuda.get_device_capability(i)
    print(f"  gpu{i}: {torch.cuda.get_device_name(i)}  sm_{cap[0]}{cap[1]}  "
          f"bf16={'native' if cap[0] >= 8 else 'NO — trainer will fall back to fp16 (CLAUDE.md 10.13)'}")
x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
torch.cuda.synchronize()
print("bf16 matmul OK:", float((x @ x).float().abs().sum()) > 0)
PY

say "Test suite (expect 66 passed)"
./.venv/bin/python -m pytest -q

# --------------------------------------------------------------------- data --
if [ -f data/imagenet64/train_data_batch_1.npz ]; then
  say "Dataset already present — skipping fetch"
  du -sh data/imagenet64
else
  say "Fetching ImageNet-64 (~25 min, ~1.8 GB down, 15.7 GB written, ~32 GB peak)"
  ./.venv/bin/python scripts/fetch_imagenet_hf.py --split train --out data/imagenet64
  # Small; needed for the FID reference later without loading the 15.7 GB train split.
  ./.venv/bin/python scripts/fetch_imagenet_hf.py --split validation --out data/imagenet64
fi

say "Sanity-checking the data (the CLAUDE.md 6.3 ordering trap)"
./.venv/bin/python - <<'PY'
import numpy as np
from eps.data import ImageNet64
ds = ImageNet64("data/imagenet64", horizontal_flip=False, max_samples=50000)
n = len(np.unique(ds.labels))
print(f"50k prefix covers {n}/1000 classes")
assert n >= 900, "prefix is class-ordered — the global shuffle did not happen; FID would be meaningless"
print("OK")
PY

[ "$MODE" = "--no-launch" ] && { say "Data ready. Launch skipped."; exit 0; }

# ------------------------------------------------------------------- launch --
say "Launching both arms"
mkdir -p runs
CUDA_VISIBLE_DEVICES=0 nohup ./.venv/bin/python scripts/train.py \
    --config configs/train_cloud_unet.yaml > runs/unet.log 2>&1 &
echo "  U-Net -> GPU 0, pid $!, log runs/unet.log"
if [ "$NGPU" -ge 2 ]; then
  CUDA_VISIBLE_DEVICES=1 nohup ./.venv/bin/python scripts/train.py \
      --config configs/train_cloud_dit.yaml > runs/dit.log 2>&1 &
  echo "  DiT   -> GPU 1, pid $!, log runs/dit.log"
else
  echo "  DiT   -> NOT started (only $NGPU GPU). Run it after the U-Net finishes:"
  echo "     ./.venv/bin/python scripts/train.py --config configs/train_cloud_dit.yaml"
fi

cat <<EOF

$(say "Running")
  tail -f runs/unet.log runs/dit.log     # loss + it/s every 100 steps
  nvidia-smi                             # confirm both GPUs are busy
  ls runs/cloud_*/previews/              # sample grids every 2500 steps

FIRST THING TO CHECK: the it/s in each log. Multiply by 256 for images/s.
The M5 Max baseline is 0.21 it/s = 54 img/s. If a GPU is showing under ~2 it/s
something is wrong (thermal throttle, a shared/oversubscribed host, or the
dataloader starving) — do not let it burn hours unnoticed.

IF MEMORY IS UNDERUSED (nvidia-smi shows < ~45 GB), restart with
  training.batch_size=256 training.grad_accum_steps=1
appended to the command for ~10-15% more throughput at the same global batch.

When training finishes: bash scripts/eval_compare.sh
EOF
