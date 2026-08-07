# ε Epsilon

**A from-scratch text-to-image generative model: flow matching and score-based
diffusion, class-conditional ImageNet-1K at 64×64, with a public web demo.**

Built as the flagship capstone for the UW PRIME Summer Research Program 2026.
Every equation is implemented directly from the MIT 6.S184 lecture notes
(Holderrieth & Erives, *An Introduction to Flow Matching and Diffusion Models*,
2026 — `PRIME_LECTURE_NOTES.pdf` in this repo), and every closed-form formula
is verified by unit tests against an independent construction (autograd,
analytic Gaussian marginals, exact flow maps).

---

## The mathematics (and where it lives)

Conventions follow the lecture notes exactly — **time runs from t = 0 (noise)
to t = 1 (data)**, opposite to much of the DDPM literature.

| Object | Formula | Notes ref | Code |
|---|---|---|---|
| Gaussian path | p_t(x\|z) = N(α_t z, β_t² I) | Eq. 15 | `eps/paths.py` |
| CondOT scheduler | α_t = t, β_t = 1−t | Eq. 15 | `CondOTScheduler` |
| Conditional velocity | u_t(x\|z) = (α̇_t − β̇_t/β_t α_t) z + β̇_t/β_t x | Eq. 20 | `conditional_velocity` |
| CFM loss | E‖u_θ(α_t z + β_t ε) − (α̇_t z + β̇_t ε)‖² | Eq. 31 | `eps/losses.py` |
| Conditional score | ∇log p_t(x\|z) = −(x − α_t z)/β_t² | Eq. 40 | `conditional_score` |
| DSM/CSM loss | E‖s_θ(x_t) + ε/β_t‖² | Eq. 54 | `DenoisingScoreMatchingLoss` |
| Score ↔ velocity | u_t(x) = a_t ∇log p_t(x) + b_t x | Prop. 1 (Eq. 41–42) | `velocity_from_score` |
| Denoiser | D_t(x) = E[z\|x_t] | Eq. 43 | `denoiser_from_velocity` |
| Flow sampling | Euler / Heun on dX = u dt | Alg. 1 | `eps/sampling/ode.py` |
| SDE extension | dX = [u + σ²/2 ∇log p] dt + σ dW | Thm. 17 (Eq. 44) | `eps/sampling/sde.py` |
| Euler–Maruyama | X ← X + h·drift + σ√h ε | Alg. 2 (Eq. 9) | `integrate_sde` |
| CFG | ũ = (1−w) u(x\|∅) + w u(x\|y) | Eq. 65 | `eps/sampling/guidance.py` |
| CFG training | drop y → ∅ with prob. η | Alg. 5 | `cfg_label_dropout` |
| Fourier time emb. | √(2/d)[cos 2πw_i t, sin 2πw_i t] | Eq. 68–69 | `FourierTimeEmbedding` |
| DiT block | AdaLN-Zero + self/cross-attention | Remark 29 | `eps/models/dit.py` |
| β-VAE | recon/(2σ̄²) + β·KL | Alg. 6 | `eps/models/vae.py` |

Two properties worth internalising, both load-bearing in this codebase:

1. **One network, four samplers.** For Gaussian paths, the velocity field and
   the score are linear reparameterizations of each other (Proposition 1).
   Whatever the checkpoint was trained to predict, `GuidedModel` exposes
   *both* fields, so any model can be sampled with the flow ODE **or** the
   reverse SDE, driven by the velocity **or** score parameterization — the
   web demo's toggles are exactly these choices.
2. **σ is a free knob.** The SDE extension trick (Theorem 17) holds for *any*
   diffusion coefficient σ_t ≥ 0; σ = 0 recovers the ODE. The conversion is
   singular at the endpoints (a_t, b_t diverge at t = 0; the score stiffens at
   t = 1), which is why score-mediated sampling integrates on
   [t_start, t_end] ⊂ (0, 1) — see `SamplingConfig`.

## Repository layout

The project directory is `epsilon/`; the importable Python package inside it is
`eps/` (so nothing is ambiguously nested, and `import eps` is unmistakable —
ε is also the noise symbol throughout the math).

```
epsilon/                      # project root
├── configs/                  # YAML experiment configs (train_unet / train_dit / inference)
├── eps/                      # the Python package  ->  `import eps`
│   ├── config.py             # typed dataclass config system + dotted CLI overrides
│   ├── paths.py              # schedulers, Gaussian path, conversions   [math core]
│   ├── losses.py             # CFM, DSM, CFG label dropout              [math core]
│   ├── utils.py              # image conversion, grid saving
│   ├── sampling/             # ode.py (Euler/Heun), sde.py (E–M), guidance.py (CFG)
│   ├── models/               # embeddings.py, unet.py (AdaGN), dit.py (AdaLN-Zero), vae.py
│   ├── training/             # trainer.py, distributed.py, ema.py
│   ├── data/                 # ImageNet-64 npz / ImageFolder / flat-zip datasets
│   ├── evaluation/           # sample generation + FID/IS/precision-recall
│   └── web/                  # FastAPI app + single-page frontend (KaTeX About page)
├── scripts/                  # train.py, sample.py, evaluate_fid.py, pack_data.py, launch_hyak.sh
└── tests/                    # 66 unit tests for the mathematical core
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -e .
pytest                        # 66 tests, all math verified, ~2 s
```

### Data

**Class-conditional training needs labels.** The FID<12 target, classifier-free
guidance, and the demo's class picker all require paired (image, label) data.

The one command that gets you there — no account, no approval queue (~1.8 GB
download, ~16 GB of npz out):

```bash
pip install -e .[data]
python scripts/fetch_imagenet_hf.py --split train --out data/imagenet64
python scripts/fetch_imagenet_hf.py --split validation --out data/imagenet64
```

This pulls [`benjamin-paine/imagenet-1k-64x64`](https://huggingface.co/datasets/benjamin-paine/imagenet-1k-64x64)
(ungated repack of ImageNet-1K at 64×64, integer labels, standard sorted-synset
order so it is already index-aligned with `eps/data/imagenet_classes.txt`) and
writes it in the **official Downsampled-ImageNet npz layout** —
`train_data_batch_*.npz` with `data` (N, 12288) uint8 and `labels` 1..1000 —
which `data.name=imagenet64` reads unchanged. That repo's train split is stored
in ascending class order, so the fetcher globally permutes before sharding
(`--shuffle`, on by default); without it any prefix of the data is a handful of
classes. Downsampling was Lanczos + JPEG rather than the official box filter, so
FID is self-consistent against your own reference but not directly comparable to
published ImageNet-64 numbers.

The official npz archives from [image-net.org](https://image-net.org/download-images.php)
→ *Download* → *Downsampled ImageNet* (free account, `.edu` approval usually
same-day) are the strictly-comparable option, and drop into the same directory.

The Kaggle mirror (`train_64x64.zip`) is **images only — 1,281,149 PNGs, flat,
no labels** — so it supports *unconditional* training only (`data.name=flat`,
or pack it first):

```bash
# turn 1.3M small files into a few large npz shards (never extracts the zip)
python scripts/pack_data.py --input eps/data/imagenet64/train_64x64.zip \
    --out data/imagenet64_packed --shard-size 128000
```

Packing matters on Hyak: 1.3M inodes on `/gscratch` is slow to read every
epoch and counts against quota; a handful of shards is a few sequential reads.

## Training

### MacBook (M-series, development scale)

Runs natively on MPS — used for smoke tests and small experiments:

```bash
python scripts/train.py --config configs/train_unet.yaml \
    data.max_samples=50000 training.batch_size=32 \
    training.mixed_precision=none logging.wandb=false
```

### Hyak (production scale)

Configured for the Rao lab (`u_hyak_rao`, NetID `aaravs07`). Work out of
`/gscratch/rao/aaravs07/` — home is capped at 10 GB, and per the access email
do **not** use `/gscratch/cse`.

```bash
# once, on the klone login node:
ssh aaravs07@klone.hyak.uw.edu
mkdir -p /gscratch/rao/aaravs07 && cd /gscratch/rao/aaravs07
git clone <your-repo> epsilon && cd epsilon
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt -e .
hyakalloc                     # confirm account name + GPU partitions you hold

# submit (verify the #SBATCH account/partition match hyakalloc first):
sbatch scripts/launch_hyak.sh configs/train_unet.yaml
squeue -u aaravs07            # watch the queue
tail -f logs/slurm-*.out      # watch training
```

Compute nodes have no outbound internet, so W&B logs offline; run
`wandb sync logs/wandb/offline-*` from the login node to upload.

The launcher auto-resumes from `ckpt_latest.pt` if the job is requeued
(checkpoint partitions preempt), uses `torchrun` with one process per GPU
(DDP), bf16 autocast, gradient clipping, and EMA (0.9999).

**The FID < 12 recipe** (`configs/train_unet.yaml`): ADM-class U-Net (273M
params: 192 base channels, mult 1/2/3/4, 3 res blocks, attention at 16 & 8),
CFM loss on the CondOT path, global batch 1024, constant LR 1e-4 after 5k
warmup, η = 0.1 label dropout, 400k steps. Evaluate the EMA weights with
100-step Euler and sweep guidance w ∈ [1.0, 2.0] — FID is optimised by mild
guidance (large w improves per-image fidelity but destroys the diversity term;
the w ≈ 1.5 sweet spot is standard for ImageNet-64). The DiT config
(`train_dit.yaml`, DiT-B/4) is the comparison backbone; scale it to DiT-L/4
after the baseline is verified.

## Evaluation

```bash
# one-time: export real reference images
python scripts/evaluate_fid.py export-ref --data-root data/imagenet64 \
    --out data/fid_ref --num 50000

# FID + Inception Score + precision/recall (torch-fidelity backend)
python scripts/evaluate_fid.py run --ckpt runs/unet64/ckpt_latest.pt \
    --ref data/fid_ref --num 50000 --guidance 1.5 --steps 100
```

In-training FID is also supported (`eval.fid_every`, `eval.fid_reference_dir`).

## Sampling & web demo

```bash
# grids from the CLI
python scripts/sample.py --ckpt runs/unet64/ckpt_latest.pt \
    --classes 207 88 979 417 --guidance 4 --method ode

# the demo
EPSILON_CKPT=runs/unet64/ckpt_latest.pt \
    uvicorn eps.web.app:app --host 0.0.0.0 --port 7860
```

The demo (single-page app at `/`) lets a visitor type a prompt (fuzzy-matched
to an ImageNet class), choose **Flow Matching (ODE)** vs **Diffusion (SDE)**,
**velocity** vs **score** parameterization, guidance scale, and step count;
generated images can be downloaded and saved to a local gallery. The About tab
renders the full mathematical story with KaTeX. Deploy anywhere a Python
process runs (HF Spaces / Railway / VPS): `pip install -e .[web]`, set
`EPSILON_CKPT`, run uvicorn.

## Latent-space scaling (256/512)

The codebase is latent-ready: set `training.latent_space=true` and
`vae.pretrained=stabilityai/sd-vae-ft-ema` (any diffusers `AutoencoderKL`) and
the identical trainer/samplers operate on 4-channel latents at stride 8; a
from-scratch β-VAE (Algorithm 6, KL warm-up, fixed decoder variance) is
included in `eps/models/vae.py` for training your own autoencoder.

## Design decisions

* **Plain YAML + dataclasses instead of Hydra** — full typing, unknown-key
  rejection, dotted overrides, zero magic, and the resolved config is embedded
  in every checkpoint for exact reproducibility.
* **DSM defaults to the DDPM weighting** (‖β_t s_θ + ε‖²): identical minimiser
  to the notes' uniform CSM loss but numerically stable at β_t → 0; the exact
  uniform weighting is one flag away.
* **Zero-initialised residual/output branches everywhere** (ADM & AdaLN-Zero
  convention): every network is the zero function at init, which the tests
  assert — training starts from an unbiased velocity/score estimate.
* **The MIT labs** (`lab_one/two/three.ipynb`) were used as convention
  cross-checks (time direction, α/β abstraction, CFG combination); no lab code
  is reused.

## References

* P. Holderrieth & E. Erives, *An Introduction to Flow Matching and Diffusion
  Models*, MIT 6.S184 lecture notes, 2026.
* Lipman et al., *Flow Matching for Generative Modeling*, 2023.
* Song et al., *Score-Based Generative Modeling through SDEs*, 2021.
* Ho & Salimans, *Classifier-Free Diffusion Guidance*, 2022.
* Peebles & Xie, *Scalable Diffusion Models with Transformers* (DiT), 2023.
* Dhariwal & Nichol, *Diffusion Models Beat GANs* (ADM U-Net), 2021.
