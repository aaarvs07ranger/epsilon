#!/usr/bin/env bash
# Baselines + visualisations for the U-Net vs DiT comparison. Run after both
# arms of scripts/setup_runpod.sh have finished.
#
#   bash scripts/eval_compare.sh              # full: reference -> FID sweep -> grids
#   bash scripts/eval_compare.sh --grids-only # just the pictures (fast, ~2 min)
#
# Produces, under results/:
#   fid.md                    the table to paste into the writeup
#   grid_<model>_w<g>.png     class-conditional samples, both models, 3 guidances
#   grid_<model>_sde.png      SDE vs ODE at matched settings
#   traj_<model>.png          the same seed at increasing guidance
#
# Cost note: the FID sweep is the expensive part — 6 evaluations x 10k samples
# x 100 ODE steps, and CFG doubles the function evaluations. MEASURED-BASIS
# ESTIMATE on 2x RTX PRO 6000: **~2.5 h**, not the ~45 min originally guessed
# here. Roughly 80 min for the U-Net arm and 40 for the DiT, plus reference
# export and Inception feature extraction.
#
# To halve it, drop to 5k samples: `NUM=5000 bash scripts/eval_compare.sh`.
# FID is biased at low sample counts, but both models get identical treatment
# so the *comparison* stays valid — just disclose the sample count.

set -euo pipefail

PY="${PY:-./.venv/bin/python}"
UNET="${UNET:-runs/cloud_unet/ckpt_latest.pt}"
DIT="${DIT:-runs/cloud_dit/ckpt_latest.pt}"
REF="${REF:-data/fid_ref}"
NUM="${NUM:-10000}"          # bump to 50000 for the headline number
STEPS="${STEPS:-100}"
CLASSES="${CLASSES:-207 88 979 417 279 972 387 360}"
OUT=results

say() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
mkdir -p "$OUT"

for c in "$UNET" "$DIT"; do
  [ -f "$c" ] || { echo "missing checkpoint: $c" >&2; exit 1; }
done

# ------------------------------------------------------------------- grids ---
# Do the pictures FIRST: they are cheap, and if something is wrong with the
# sampler you find out in 2 minutes rather than 45.
say "Sample grids (guidance sweep)"
for g in 1.5 3.0 6.0; do
  for pair in "unet:$UNET" "dit:$DIT"; do
    name=${pair%%:*}; ckpt=${pair#*:}
    $PY scripts/sample.py --ckpt "$ckpt" --classes $CLASSES --per-class 2 \
        --guidance "$g" --steps "$STEPS" --method ode --seed 0 \
        --out "$OUT/grid_${name}_w${g}.png"
  done
done

say "SDE vs ODE at matched settings"
for pair in "unet:$UNET" "dit:$DIT"; do
  name=${pair%%:*}; ckpt=${pair#*:}
  $PY scripts/sample.py --ckpt "$ckpt" --classes $CLASSES --per-class 2 \
      --guidance 3.0 --steps "$STEPS" --method sde --seed 0 \
      --out "$OUT/grid_${name}_sde.png"
done

# The score parameterisation is a reparameterisation of velocity for Gaussian
# paths (Prop. 1), so this should look like the velocity grid, not like noise.
# It is a cheap end-to-end check that the conversion coefficients are right.
say "Score-parameterised sampling (should match velocity — Prop. 1)"
$PY scripts/sample.py --ckpt "$UNET" --classes $CLASSES --per-class 2 \
    --guidance 3.0 --steps "$STEPS" --method ode --parameterization score \
    --seed 0 --out "$OUT/grid_unet_score.png"

[ "${1:-}" = "--grids-only" ] && { say "Grids in $OUT/"; exit 0; }

# --------------------------------------------------------------------- FID ---
if [ -d "$REF" ] && [ -n "$(ls -A "$REF" 2>/dev/null)" ]; then
  say "Reusing FID reference in $REF"
else
  # Use the VALIDATION split: it is 50k images (~0.6 GB) and already shuffled
  # upstream, so exporting it does not load the 15.7 GB training split into RAM.
  say "Exporting FID reference from the validation split"
  $PY scripts/evaluate_fid.py export-ref \
      --data-root data/imagenet64 --out "$REF" --num 50000 --split val
fi

say "FID / IS / precision-recall sweep"
echo "# U-Net vs DiT — $NUM samples, $STEPS-step Euler ODE, EMA weights" > "$OUT/fid.md"
echo "" >> "$OUT/fid.md"
echo "| model | params | guidance w | metrics |" >> "$OUT/fid.md"
echo "|---|---|---|---|" >> "$OUT/fid.md"

# Guidance sweep over [1.0, 2.0]: FID's diversity term punishes the high w that
# makes the preview grids look good, so the FID optimum is usually near 1.5.
for pair in "unet:92.5M:$UNET" "dit:130.4M:$DIT"; do
  name=$(echo "$pair" | cut -d: -f1)
  prm=$(echo "$pair" | cut -d: -f2)
  ckpt=$(echo "$pair" | cut -d: -f3)
  for g in 1.0 1.5 2.0; do
    say "$name @ w=$g"
    m=$($PY scripts/evaluate_fid.py run --ckpt "$ckpt" --ref "$REF" \
          --num "$NUM" --guidance "$g" --steps "$STEPS" --seed 0 2>&1 | tail -3 | tr '\n' ' ')
    echo "| $name | $prm | $g | $m |" >> "$OUT/fid.md"
    echo "$m"
  done
done

say "Done"
cat "$OUT/fid.md"
cat <<'EOF'

READ THE NUMBERS HONESTLY:
  * These are computed against a reference exported from the Lanczos/JPEG HF
    repack, not the official image-net.org release (CLAUDE.md 6.4). They are
    internally consistent but NOT comparable to published ImageNet-64 FIDs
    (ADM 2.07). Say so explicitly.
  * 100k steps x batch 256 = 25.6M images. The flagship recipe is 410M images
    with a 3x larger model. This is a method demonstration at ~1/16 the
    compute, not a scale comparison.
  * DiT is expected to lose at a matched 100k steps — it wants longer training
    than a U-Net. That is a statement about this compute budget, not about the
    architecture.

>>> BEFORE YOU TERMINATE THE POD, GET YOUR RESULTS OFF IT. <<<
Everything below is deleted with the pod and is not recoverable:

    results/                      the FID table and every grid
    runs/cloud_unet/ckpt_latest.pt
    runs/cloud_dit/ckpt_latest.pt        (~1.4 GB each — the trained weights)
    runs/cloud_*/previews/               training-progress grids

Smallest useful bundle (a few MB, excludes the checkpoints):

    tar czf epsilon_results.tgz results runs/cloud_*/previews runs/*.log

Then pull it down from your laptop, or use `runpodctl send epsilon_results.tgz`.
The checkpoints are worth keeping too if you want to sample or evaluate later —
without them you would have to retrain to change anything.

And TERMINATE the pod, do not just stop it: a stopped pod keeps billing for its
storage.
EOF
