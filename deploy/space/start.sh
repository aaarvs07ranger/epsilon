#!/usr/bin/env bash
# Fetch the weights, then serve. Runs on every container start.
#
# Weights live in a HF *model* repo rather than in the Space itself: Spaces are
# for code, model repos are for large files, and this way a weight update does
# not force a Space rebuild.
set -euo pipefail

REPO="${EPSILON_MODEL_REPO:?set EPSILON_MODEL_REPO}"
FILES="${EPSILON_MODEL_FILES:-unet_60k.pt,dit_100k.pt}"
DEST="${HOME}/app/weights"
mkdir -p "$DEST"

echo "[start] fetching weights from ${REPO}"
python - "$REPO" "$FILES" "$DEST" <<'PY'
import sys, pathlib
from huggingface_hub import hf_hub_download
repo, files, dest = sys.argv[1], sys.argv[2].split(","), pathlib.Path(sys.argv[3])
for f in [f.strip() for f in files if f.strip()]:
    p = hf_hub_download(repo, f, local_dir=str(dest))
    print(f"[start]   {f}  {pathlib.Path(p).stat().st_size / 1e6:.0f} MB")
PY

# Comma-separated absolute paths; the app derives its own labels from each
# checkpoint's embedded config, so nothing needs to be named consistently here.
export EPSILON_MODELS="$(python - "$DEST" "$FILES" <<'PY'
import sys, pathlib
dest, files = pathlib.Path(sys.argv[1]), sys.argv[2].split(",")
print(",".join(str(dest / f.strip()) for f in files if f.strip()))
PY
)"
echo "[start] EPSILON_MODELS=$EPSILON_MODELS"

exec uvicorn eps.web.app:app --host 0.0.0.0 --port 7860
