# Deploying the Epsilon demo (free)

Two Hugging Face repos: a **model** repo for the weights, a **Space** for the
app. Roughly ten minutes, no cost.

Everything in `deploy/space/` is the Space; `deploy/*.pt` are the weights.

---

## 0. Prepare the weights (already done)

```bash
python scripts/export_inference_ckpt.py --ckpt runs/cloud_unet/ckpt_final.pt --out deploy/unet_60k.pt
python scripts/export_inference_ckpt.py --ckpt runs/cloud_dit/ckpt_final.pt  --out deploy/dit_100k.pt
```

This folds the EMA weights into `model`, drops the optimizer state, and casts
to fp16 — **1.48 GB → 185 MB** for the U-Net, 2.09 GB → 261 MB for the DiT, an
8× reduction with no change to sampling beyond fp16 rounding (verified: max
weight deviation 1.8e-3 against values spanning ±4.5).

## 1. Log in

```bash
pip install -U huggingface_hub
hf auth login          # paste a WRITE token from huggingface.co/settings/tokens
```

## 2. Push the weights to a model repo

```bash
hf repo create epsilon-imagenet64 --repo-type model
hf upload aaarvs07ranger/epsilon-imagenet64 deploy/unet_60k.pt  unet_60k.pt
hf upload aaarvs07ranger/epsilon-imagenet64 deploy/dit_100k.pt  dit_100k.pt
```

Weights go in a *model* repo, not the Space: Spaces are for code, and this way
swapping a checkpoint never forces a Space rebuild.

## 3. Create the Space

```bash
hf repo create epsilon --repo-type space --space_sdk docker
git clone https://huggingface.co/spaces/aaarvs07ranger/epsilon /tmp/epsilon-space
cp deploy/space/{Dockerfile,start.sh,README.md} /tmp/epsilon-space/
cd /tmp/epsilon-space && git add -A && git commit -m "Epsilon demo" && git push
```

It builds in ~5 minutes, then downloads ~450 MB of weights on first start.

If your HF username is not `aaarvs07ranger`, change `EPSILON_MODEL_REPO` in the
Dockerfile (or set it in the Space's **Settings → Variables**, which does not
require a rebuild).

## 4. Optional: GPU

The free CPU tier works — the UI detects it and defaults to 30 Heun steps
(2nd-order, so ≈60 function evaluations; much better than 30 Euler steps at
similar cost). Expect tens of seconds per image.

For **ZeroGPU** (free, quota'd): set the Space hardware to ZeroGPU and delete
the `--index-url https://download.pytorch.org/whl/cpu` line from the Dockerfile
so pip resolves the CUDA build. The app picks up CUDA automatically and the UI
reverts to 100-step defaults.

---

## Updating later

| Change | What to do |
|---|---|
| App code | Push to GitHub, then **Restart** the Space (the Dockerfile pins `main`) |
| Weights | `hf upload` the new file, restart the Space |
| A different checkpoint | Edit `EPSILON_MODEL_FILES` in Space Variables |

## Sanity check before you push

```bash
EPSILON_MODELS="deploy/unet_60k.pt,deploy/dit_100k.pt" \
  uvicorn eps.web.app:app --port 7860
curl -s localhost:7860/api/health | python -m json.tool   # expect both models
```

This is exactly what the Space does, minus the download — if it works locally
it will work there.
