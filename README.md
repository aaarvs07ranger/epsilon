# ε Epsilon

A flow-matching and score-based diffusion image generator, written from
scratch. Class-conditional ImageNet-1K at 64×64, two backbones, and a web demo
you can poke at.

Built as the capstone for the UW PRIME Summer Research Program 2026. Every
equation comes straight out of the MIT 6.S184 lecture notes (Holderrieth &
Erives, *An Introduction to Flow Matching and Diffusion Models*, 2026 —
included here as `PRIME_LECTURE_NOTES.pdf`), and every closed-form expression
has a unit test that checks it against an independent construction: autograd
where a derivative is claimed, analytic Gaussian marginals where a
distribution is claimed, exact flow maps where a trajectory is claimed. That
verification discipline is the point of the project. The pictures are a
consequence of it.

**On scope, up front:** this is *class-conditional*, not text-to-image. The
demo has a prompt box, but it matches what you type to the nearest of 1000
ImageNet class names. Type "a cat riding a skateboard" and it finds `tabby
cat` and draws a cat. There is no text encoder in this repository.

---

## What it makes

![U-Net samples at guidance 3.0](results/grid_unet_w3.0.png)

Golden retriever, macaw, valley, balloon, arctic fox, cliff, lesser panda,
otter — two samples each from the 92.5M U-Net at guidance w = 3.0, 100 Euler
steps. Natively 64×64, shown here nearest-neighbour upscaled so you can see the
pixels.

## What actually got trained

| | U-Net (ADM-style, AdaGN) | DiT-B/4 (AdaLN-Zero) |
|---|---|---|
| Parameters | 92.5M | 130.4M |
| Steps | 60,000 | 100,000 |
| Images seen | 15.4M (~12 epochs) | 25.6M (~20 epochs) |
| Throughput | 1.64 it/s | 3.20 it/s |
| Hardware | 1× RTX PRO 6000 | 1× RTX PRO 6000 |

Both arms ran an identical recipe — same data, same global batch of 256, same
constant 1e-4 learning rate after 5k warmup, same EMA decay of 0.9999, same 10%
classifier-free-guidance dropout, same seed. The two configs differ in the
`model:` block and nowhere else, which was verified programmatically rather
than by reading them side by side. Total cost, about $49 of rented GPU time.

### Two findings worth reporting

**The bigger model trains twice as fast.** DiT-B/4 carries 41% more parameters
than the U-Net and runs at 3.20 it/s against 1.64. That inverts the obvious
intuition, and it isn't a bug: DiT is dense matmuls that saturate tensor cores,
while a U-Net at 64×64 with 128 base channels is a great many small
convolutions, GroupNorm calls, and attention at 16×16 and 8×8. Low arithmetic
intensity per kernel means it cannot keep a fast GPU fed. Doubling the
micro-batch from 128 to 256 moved the rate by under 1%, and both settings sat
at 100% utilisation drawing 504W — so it is compute-bound, not launch-bound.

**But at matched steps the U-Net still looks better.** Compared at 60k steps
each, the U-Net gives sharper macaws, cleaner red pandas, more coherent
balloons. That is the expected result rather than a verdict on the
architecture: DiT is known to need longer training for equivalent quality,
which is exactly why the original recipes budget it 600k steps against the
U-Net's 400k. It is a statement about this compute budget and nothing more.

## The mathematics, and where it lives

Conventions follow the notes exactly — **time runs from t = 0 (noise) to t = 1
(data)**, the opposite of most of the DDPM literature. If you import an idea
from a DDPM paper, flip the time axis first.

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

Two properties are load-bearing throughout:

**One network, four samplers.** For Gaussian paths the velocity field and the
score are linear reparameterisations of one another (Proposition 1). Whatever a
checkpoint was trained to predict, `GuidedModel` exposes both, so any model can
be driven by the flow ODE or the reverse SDE, in either parameterisation. The
demo's four toggles are exactly these choices — one set of weights, not four
models, which makes the proposition something you can click on.

**σ is a free knob.** The SDE extension (Theorem 17) holds for any diffusion
coefficient σ_t ≥ 0, and σ = 0 recovers the ODE exactly. The score↔velocity
conversion is singular at both ends — a_t and b_t diverge at t = 0 through the
division by α_t, and the score stiffens at t = 1 as β_t → 0 — so anything
score-mediated integrates on [1e-4, 0.9999] ⊂ (0, 1). That interval is
deliberate, not a rounding artefact, and shortening it will bite you.

## Layout

The project directory is `epsilon/`; the package inside it is `eps/`, so
`import eps` is unmistakable — ε is the noise symbol throughout the maths.

```
eps/
├── paths.py            schedulers, the Gaussian path, score↔velocity conversion
├── losses.py           CFM and DSM objectives, CFG label dropout
├── config.py           typed dataclass config; rejects unknown keys on purpose
├── models/             unet.py (AdaGN), dit.py (AdaLN-Zero), embeddings.py, vae.py
├── sampling/           ode.py (Euler/Heun), sde.py (Euler–Maruyama), guidance.py
├── training/           trainer.py, ema.py, distributed.py
├── data/               imagenet.py — npz shards, ImageFolder, flat zip
├── evaluation/         fid.py — sample generation, FID / IS / precision-recall
└── web/                app.py (FastAPI) + static/index.html (the demo)

configs/   one YAML per run; the resolved config lands inside every checkpoint
scripts/   train, sample, evaluate, fetch data, export deployment weights
tests/     66 tests, all of them about the maths
results/   sample grids from the trained models
deploy/    Dockerfile and instructions for the hosted demo
```

## Running it

```bash
python3.13 -m venv .venv
./.venv/bin/pip install -e ".[data,web,dev,eval]"
./.venv/bin/python -m pytest -q          # 66 passed
```

Fetch the data — 1.28M labelled 64×64 images, roughly 25 minutes and 16 GB:

```bash
./.venv/bin/python scripts/fetch_imagenet_hf.py --split train --out data/imagenet64
```

Train:

```bash
./.venv/bin/python scripts/train.py --config configs/train_cloud_unet.yaml
```

`configs/` carries variants for a single CUDA GPU (`train_cloud_*.yaml`), Apple
silicon (`train_m5.yaml`), and the full-scale recipes the hardware here could
not fund (`train_unet.yaml`, `train_dit.yaml`).

Sample from a checkpoint:

```bash
./.venv/bin/python scripts/sample.py --ckpt runs/cloud_unet/ckpt_final.pt \
    --classes 207 88 979 417 --guidance 3 --method ode
```

Run the demo locally with both backbones loaded:

```bash
EPSILON_MODELS="deploy/unet_60k.pt,deploy/dit_100k.pt" \
  ./.venv/bin/uvicorn eps.web.app:app --port 7860
```

## The demo

Type a prompt, pick a backbone, choose ODE or SDE and velocity or score, set
the guidance scale and step count, watch it integrate. **Compare** runs both
backbones from *identical* noise, so what you see is the architecture and not
the luck of the draw.

Deployment weights get stripped first — EMA folded in, optimizer state dropped,
cast to fp16:

```bash
python scripts/export_inference_ckpt.py \
    --ckpt runs/cloud_unet/ckpt_final.pt --out deploy/unet_60k.pt
```

1.48 GB becomes 185 MB, an 8× reduction, the only difference being fp16
rounding — worst weight deviation 1.8e-3 against values spanning ±4.5.
`deploy/DEPLOY.md` covers the rest of the hosting setup.

## Honest limitations

- **No FID number yet.** The sweep is written and tested
  (`scripts/eval_compare.sh`) but was skipped to stay inside the GPU budget. It
  runs locally for free, just slowly.
- **FID here wouldn't be comparable anyway.** The 64×64 data is a Lanczos-
  resized, JPEG-stored repack of ImageNet, not the official box-filtered
  release. Numbers from it are internally consistent but should not be set
  beside published ImageNet-64 results such as ADM's 2.07.
- **This is a small model trained briefly** — 15.4M images against the flagship
  recipe's 410M, at a third of the parameters. A method demonstration at about
  1/16 the compute, and it reads as one.
- **Multi-GPU is untested.** The DDP path exists and has never once been
  executed; both runs here were deliberately single-GPU.

## Design decisions

* **Plain YAML and dataclasses instead of Hydra** — full typing, unknown keys
  rejected loudly, dotted overrides, no magic, and the resolved config embedded
  in every checkpoint so a run can be reproduced exactly.
* **DSM defaults to the DDPM weighting** (‖β_t s_θ + ε‖²): the same minimiser
  as the notes' uniform CSM loss but numerically stable as β_t → 0. The exact
  uniform weighting is one argument away.
* **Zero-initialised residual and output branches everywhere** (ADM and
  AdaLN-Zero convention), so every network is the zero function at
  initialisation and training begins from an unbiased estimate. The tests
  assert it, so it cannot quietly regress.
* **EMA weights for all sampling and evaluation**, with a warmup ramp so early
  EMA tracks the fast-moving young model instead of the random init.
* **The MIT labs** (`lab_one/two/three.ipynb`) served only as convention
  cross-checks — time direction, the α/β abstraction, how CFG combines. No lab
  code is reused, which matters for the originality of the capstone.

## References

* P. Holderrieth & E. Erives, *An Introduction to Flow Matching and Diffusion Models*, MIT 6.S184, 2026.
* Lipman et al., *Flow Matching for Generative Modeling*, 2023.
* Song et al., *Score-Based Generative Modeling through SDEs*, 2021.
* Ho & Salimans, *Classifier-Free Diffusion Guidance*, 2022.
* Peebles & Xie, *Scalable Diffusion Models with Transformers* (DiT), 2023.
* Dhariwal & Nichol, *Diffusion Models Beat GANs* (ADM U-Net), 2021.

## License

MIT, per `LICENSE`. The lecture notes PDF is redistributed under its own terms.
