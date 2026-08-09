# CLAUDE.md — Epsilon project context

> **Purpose.** This file is the durable memory for the Epsilon project. Chat
> sessions end and their context is lost; this file is what survives. Anything
> learned that is *not* recoverable by reading the code — decisions and their
> reasons, dead ends, external account state, verified measurements, what is
> blocked on whom — belongs here.
>
> **Maintenance contract (for Claude).** At the end of any session that changes
> the code, the plan, or an external dependency: update §1 (Status), append to
> §12 (Decision log) with today's date, and add anything newly discovered to
> §10 (Traps). Do not let §1 go stale — a wrong status board is worse than none.
> Prefer editing an existing line over appending a duplicate. Keep entries
> factual and dated; write `verified — <how>` or "unverified" rather than
> implying certainty you do not have.

---

## 1. Status board

**Last updated: 2026-08-08 (evening)**

| Area | State |
|---|---|
| Math core (`paths`, `losses`, samplers) | ✅ Complete, 66 tests pass (~1.8 s) |
| Models (U-Net 273.0M, DiT-B/4 130.4M, 92.5M small) | ✅ Complete, verified on `meta` device |
| Trainer (DDP, EMA, AMP, ckpt, resume) | ✅ Complete; AMP now correct on CUDA **and** MPS |
| Web demo (FastAPI + SPA) | ✅ Complete, loads without a checkpoint |
| **Labeled data** | ✅ Source unblocked 2026-08-06 (§6); **not yet fetched anywhere** |
| Git / GitHub | ✅ Public repo `aaarvs07ranger/epsilon` |
| **Execution venue** | 🔀 **CHANGED 2026-08-08: moving off Hyak to a single M5 Max.** See §7.0 |
| Hyak (rao) | ⛔ **Abandoned for Epsilon** — `/gscratch` 100% full (§10.15), GPUs needed by NSL (§10.14) |
| Measured throughput | ✅ **M4: 0.26 it/s @ batch 32 = 8.3 img/s** (§7.0). M5 Max: ⏳ unmeasured |
| **FID < 12 target** | 🔴 **Unfunded by any hardware in play.** Needs honest restatement — §7.2 |
| Dataset fetched | ⏳ Not done (do it on the M5 Max — ~700 GB free, no quota) |
| Real training run | ⏳ Not started |
| FID reference set | ⏳ Not exported |
| Public demo deployment | ⏳ Not started — **can ship before training finishes** (§9) |

Local-only convenience: `data/imagenet64_small` — 20k images but **only 16
classes** (fish/sharks; `--limit` takes a class prefix, §6.3). Smoke tests only.

### Immediate next actions, in order

**Blocked until ~2026-08-09/10, when Aarav is home with the M5 Max.** Nothing
useful remains to be done on Hyak; do not restart that path.

0. **On the laptop, now:** `git push` so the M5 Max can clone. (Done 2026-08-08.)
1. **Clean Epsilon off Hyak** — §7.0 has the exact commands. Frees ~6 GB for NSL.
   *Only* `/gscratch/rao/aaravs07/epsilon`. **Never touch `.../nsl` (§10.14).**
2. **M5 Max setup:** clone → venv → `pip install -e ".[data,web]"` →
   `pytest -q` (expect 66 passed).
3. **MEASURE IT/S FIRST — 200 steps, ~10 min.** This is the one number the
   whole plan hangs on and it has never been measured on the real machine:
   ```
   python scripts/train.py --config configs/train_m5.yaml \
       data.root=data/imagenet64_small data.max_samples=4096 \
       training.total_steps=200 logging.output_dir=/tmp/bench
   ```
   Multiply by 256 (the global batch) for images/s. Set `total_steps` from
   this, not from the config's placeholder 150000.
4. **Fetch the full dataset** (§6.2) — ~32 GB peak, trivial locally.
5. **Deploy the web demo immediately**, without a checkpoint (§9). Decouples
   "live" from "trained" and gets a public URL on day one.
6. **Launch training** detached under `caffeinate -is nohup` (§7.0). Use
   `eval.sample_every=1000` on the first run — the default 5000 is ~5 h
   between preview grids at these rates.
7. **Export the 50k FID reference**, evaluate EMA weights, report honestly.

---

## 2. What this project is

A from-scratch flow-matching / score-based diffusion image generator:
class-conditional ImageNet-1K at 64×64, plus a public web demo. Flagship
capstone for the **UW PRIME Summer Research Program 2026**.

Every equation is implemented directly from the MIT 6.S184 lecture notes
(Holderrieth & Erives, *An Introduction to Flow Matching and Diffusion Models*,
2026 — `PRIME_LECTURE_NOTES.pdf`, 12 MB, in this repo). Every closed-form
formula is checked by a unit test against an independent construction
(autograd, analytic Gaussian marginals, exact flow maps). That verification
discipline is the point of the project and should not be relaxed.

**Target:** FID(50k) < 12 on ImageNet-64, class-conditional, with CFG.

**Explainer artifact (live):**
https://claude.ai/code/artifact/36b232f5-b663-4628-9ba7-e26a8743221b
Built as a zoom ladder (1× / 10× / 100× / 1000×) then eleven sections: repo map,
math core, both networks, training, sampling, data, evaluation, demo, runbook,
glossary. Colour convention in all its diagrams: **teal = noise / t=0**,
**amber = data / t=1**. Section 10 holds the exact runbook commands.

---

## 3. Environment

| | |
|---|---|
| Project root | `/Users/aarav/Desktop/epsilon` |
| Python package | `eps/` — so `from eps.paths import ...` |
| Machine | macOS (darwin 25.5.0), Apple Silicon, ~730 GB free |
| Python | 3.13.6 (in `.venv`) |
| torch | 2.13.0, **MPS available**, no CUDA |
| numpy | 2.5.1 |
| Tests | `./.venv/bin/python -m pytest -q` → 66 passed |

### Installed vs. not (local venv, as of 2026-08-06)

Present: `torch`, `numpy`, `pyyaml`, `pillow`, `fastapi`, `uvicorn`,
`pyarrow` 25.0.0, `huggingface_hub` 1.26.0.

**Missing:** `wandb`, `torch-fidelity`, `clean-fid`. Consequences:
- Any config with `logging.wandb: true` (both training configs have it) will
  crash locally on `import wandb`. Always pass `logging.wandb=false` for local
  runs, or install wandb.
- `scripts/evaluate_fid.py run` cannot compute metrics locally yet.

### venv gotcha (bites every session)

`source .venv/bin/activate && pip install ...` resolves to the **system** pip
and fails with a PEP 668 "externally-managed-environment" error. `python` does
resolve correctly. Always install with the explicit interpreter path:

```bash
./.venv/bin/pip install <pkg>          # correct
./.venv/bin/python -m pytest -q        # correct
```

---

## 4. Repository layout

```
epsilon/                      # project root (NOT a git repo yet — see §10.1)
├── CLAUDE.md                 # this file
├── README.md                 # public-facing docs
├── PRIME_LECTURE_NOTES.pdf   # mathematical source of truth (MIT 6.S184)
├── lab_one/two/three.ipynb   # MIT labs — convention cross-checks only, no code reused
├── configs/                  # train_unet.yaml, train_dit.yaml, inference.yaml
├── eps/                      # the Python package
│   ├── config.py             # typed dataclass config + dotted CLI overrides
│   ├── paths.py              # schedulers, Gaussian path, conversions   [math core]
│   ├── losses.py             # CFM, DSM, CFG label dropout              [math core]
│   ├── utils.py              # image conversion, grid saving
│   ├── sampling/             # ode.py (Euler/Heun), sde.py (E–M), guidance.py (CFG)
│   ├── models/               # embeddings.py, unet.py (AdaGN), dit.py (AdaLN-Zero), vae.py
│   ├── training/             # trainer.py, distributed.py, ema.py
│   ├── data/                 # imagenet.py (npz / ImageFolder / flat-zip), imagenet_classes.txt
│   ├── evaluation/           # fid.py — sample generation + FID/IS/precision-recall
│   └── web/                  # app.py (FastAPI) + static/index.html (SPA, KaTeX About)
├── scripts/
│   ├── train.py              # entry point, single-process or torchrun
│   ├── sample.py             # CLI sample grids
│   ├── evaluate_fid.py       # export-ref | run
│   ├── pack_data.py          # zip/dir -> npz shards (unlabeled sources)
│   ├── fetch_imagenet_hf.py  # labeled ImageNet-64 from HF -> npz shards   [added 2026-08-06]
│   └── launch_hyak.sh        # SLURM launcher
└── tests/                    # 66 unit tests for the mathematical core
```

Non-source files also living in the root, unrelated to the project:
`ProtonVPN_mac_v6.5.1.dmg` (125 MB), empty `main.py`. Safe to delete; add a
`.gitignore` before `git init` so neither they nor the 11 GB zip get committed.

---

## 5. Mathematical conventions — read before touching the math

These are the things that cause subtle, silent bugs if misremembered.

1. **Time runs t = 0 (noise) → t = 1 (data).** This is the lecture notes'
   convention and is **opposite** to most of the DDPM literature. Every
   scheduler, loss, sampler, and config in this repo follows it. If you import
   an idea from a DDPM paper, flip the time axis first.
2. **Gaussian path** p_t(x|z) = N(α_t z, β_t² I) (Eq. 15). Boundary conditions
   α_0 = β_1 = 0, α_1 = β_0 = 1. Default is CondOT: α_t = t, β_t = 1 − t.
3. **One network, four samplers.** For Gaussian paths velocity and score are
   linear reparameterizations of each other (Prop. 1), so `GuidedModel` exposes
   *both* regardless of what the checkpoint predicts. The demo's toggles
   (ODE/SDE × velocity/score) are exactly these choices.
4. **σ is a free knob.** The SDE extension (Thm. 17) holds for any σ_t ≥ 0;
   σ = 0 recovers the ODE.
5. **Singularities are real.** The score↔velocity conversion coefficients
   (a_t, b_t) diverge at t = 0 (division by α_t) and the score stiffens at
   t = 1 (β_t → 0). Anything score-mediated integrates on
   [`t_start`, `t_end`] = [1e-4, 0.9999] ⊂ (0, 1). Do not "clean this up".
6. **`velocity_target` / `score_target` are division-free** forms of
   `conditional_velocity` / `conditional_score`, valid where the latter are
   undefined. Training uses the targets; only tests use the conditionals.
7. **DSM defaults to DDPM weighting** ‖β_t s_θ + ε‖². Same minimiser as the
   notes' uniform CSM loss, numerically stable as β_t → 0. Uniform weighting is
   available via `DenoisingScoreMatchingLoss(weighting="uniform")` — pair it
   with `training.t_max < 1`.
8. **Null token = `num_classes`.** `LabelEmbedder` has `num_classes + 1`
   entries; index 1000 is the null/unconditional token. `UNLABELED = -1` in the
   dataset is mapped to it in `trainer.py` *before* CFG dropout.
9. **Zero-initialised output/residual branches everywhere** (ADM & AdaLN-Zero).
   Every network is the zero function at init; tests assert this. It means
   training starts from an unbiased velocity/score estimate — don't "fix" the
   zero inits.

Equation-to-code map lives in the README table and in Section 3 of the artifact.

---

## 6. Data — the whole story

### 6.1 The problem (discovered 2026-08-05)

`eps/data/imagenet64/train_64x64.zip` (11.3 GB) is the **Kaggle mirror**:
1,281,149 PNGs in one flat folder, filenames `train_64x64/0000001.png` …,
**no labels, no class subdirectories, no label file inside the archive**.
Verified by listing the archive. Filenames cannot be reliably mapped back to
classes because the official downsampled release is shuffled.

Class-conditional generation, CFG, the FID<12 target, and the demo's class
picker all need `(image, label)` pairs. So this download **cannot** be the
capstone dataset. It is still useful for unconditional throughput smoke tests
(`data.name=flat` reads straight out of the zip without extracting).

### 6.2 The fix (2026-08-06) — use this

[`benjamin-paine/imagenet-1k-64x64`](https://huggingface.co/datasets/benjamin-paine/imagenet-1k-64x64)
is an **ungated** (verified: `gated: False` via the HF API) repack of
ImageNet-1K at 64×64 with integer labels. 1,281,167 train / 50,000 val /
100,000 test. 5 parquet train shards, ~1.74 GB; 1.94 GB total.

```bash
./.venv/bin/pip install -e .[data]      # pyarrow + huggingface_hub

# full training split: ~1.8 GB download -> ~16 GB of npz shards
./.venv/bin/python scripts/fetch_imagenet_hf.py --split train --out data/imagenet64

# validation split -> val_data.npz
./.venv/bin/python scripts/fetch_imagenet_hf.py --split validation --out data/imagenet64

# smoke test first if you like (fast, but see the --limit caveat below)
./.venv/bin/python scripts/fetch_imagenet_hf.py --split train --limit 30000 \
    --shard-size 12000 --out /tmp/in64_test
```

`scripts/fetch_imagenet_hf.py` writes the **official Downsampled-ImageNet npz
layout** — `train_data_batch_N.npz` with `data` (N, 12288) uint8 CHW-flattened
and `labels` 1..1000 — so `data.name=imagenet64` reads it with zero changes.

**Verified end to end on 2026-08-06:** fetch → `ImageNet64` loads → labels
0..999 after the loader's 1-based shift → 20-step training run on MPS →
checkpoint → ODE sample → SDE sample. All green.

**Label alignment verified:** index 0 = `tench`, 207 = `golden retriever`,
979 = `valley` — matches `eps/data/imagenet_classes.txt` exactly (1000 entries;
`wc -l` reports 999 because there is no trailing newline — this is fine, the
loader `.strip().split("\n")` yields 1000).

### 6.3 Ordering trap (important, cost an hour to find)

The HF repo's **train split is stored in ascending class order.** The
**validation split is already shuffled** upstream (verified: 1000 distinct
classes in the first 5000 rows).

Left alone, this silently breaks anything that takes a *prefix* of the data:
`data.max_samples=50000` would be ~39 classes, and a 50k FID reference exported
as a prefix would have almost no class diversity — making FID meaningless while
looking perfectly normal.

Two defences are now in place:
- `fetch_imagenet_hf.py --shuffle` (**default on for train**) does a true global
  permutation via a temporary disk-backed memmap (~16 GB scratch, deleted
  afterwards), *and* permutes within each shard. Peak RAM is one shard.
- `evaluate_fid.py export-ref` now takes a **seeded random subset**, never a
  prefix, and prints a warning if the reference covers < 900 classes.

`--limit` still takes a class *prefix* (it stops downloading early on purpose).
It proves the pipeline runs; it is **not** a miniature ImageNet.

### 6.4 Caveat to disclose in the writeup

This repack was resized with **Lanczos and stored as JPEG**; the official
image-net.org npz release is raw uint8 from a box filter. FID computed against
your own exported reference is internally self-consistent, but absolute numbers
are **not** directly comparable to published ImageNet-64 FIDs (ADM 2.07, etc.).
Say so explicitly rather than quietly comparing.

If strict comparability matters, request the official archives:
image-net.org → Download → *Downsampled ImageNet 64×64* → `train_data_batch_1..10.npz`.
Free account, `.edu` approval usually same-day; **the download page only shows
these links once you are logged in** — logged out it shows just the ILSVRC/Kaggle
blurb and the terms of access. They drop into the same directory with no code
change.

### 6.5 Dataset classes

| `data.name` | Class | Labels? | Notes |
|---|---|---|---|
| `imagenet64` | `ImageNet64` | ✅ | Official npz layout; holds all data in RAM as uint8 (~16 GB full split) |
| `imagefolder` | `ImageFolder64` | ✅ | `root/<class>/*.png`, resize-short-side + centre crop |
| `flat` | `FlatImageDataset` | ❌ | Flat dir **or .zip read in place**; every label is `UNLABELED = -1` |

`scripts/pack_data.py` converts a zip/flat dir into npz shards. Packing matters
on Hyak: 1.3M inodes on `/gscratch` is slow every epoch and counts against
quota; a handful of shards is a few sequential reads.

---

## 7. Training

### 7.0 Execution venue: a single M5 Max, not Hyak (decided 2026-08-08)

**Epsilon no longer runs on Hyak.** Two independent reasons, either sufficient:

1. `/gscratch/rao` is **100% full** (4077/4096 GB group-wide) — §10.15. The
   data fetch died at 1.12M/1.28M images with `Errno 122 Disk quota exceeded`.
2. The rao allocation's 4 GPUs and its remaining disk are needed by the **NSL
   project** (§10.14), which has a conference deadline. Epsilon is the side
   project. Contending for that filesystem actively costs Aarav his paper.

Hardware: **M5 Max MacBook Pro, 18-core CPU / 32-core GPU / 36 GB unified**,
inherited ~2026-07-31. The M4 (10-core GPU, 32 GB, ~700 GB free) stays as the
dev/smoke machine — it is what these notes were written on.

**This is not a speed win, and the notes should not pretend otherwise.** Best
estimate is that 4× RTX 6000 is still **2–4× faster** than one M5 Max. The win
is availability (no queue, no 2FA, no preemption), disk (~700 GB vs 19 GB), and
not stealing resources from the higher-priority project.

#### Measured throughput — the project's first real number

| Machine | Config | Rate |
|---|---|---|
| **M4, 10-core GPU** | 92.5M U-Net, 64×64, batch 32, bf16, grad-ckpt ON | **0.26 it/s = 8.3 img/s** |
| M5 Max, 32-core GPU | same model, grad-ckpt OFF | **UNMEASURED — do this first** |

Rough extrapolation for the M5 Max is 5–12× the M4 (3.2× the GPU cores, plus
per-core neural accelerators, plus ~1.35× from dropping gradient checkpointing)
→ ~50–90 img/s. At global batch 256 that is roughly:

| Steps | Est. wall-clock |
|---|---|
| 50k | 1.5–3 days |
| 100k | 3–6 days |
| 150k | 5–9 days |

**Treat every one of those numbers as unfunded until step 3 of §1 is done.**
The whole point of the 2026-08-07 lesson is to stop planning from arithmetic.

#### Running it

`configs/train_m5.yaml` — 92.5M params, bf16, **no** gradient checkpointing
(that exists only to fit 24 GB and costs ~30% compute), global batch 256 via
64 × 4 accumulation, `num_workers: 2` because `ImageNet64` holds the whole
split in RAM as uint8 (~16 GB) against 36 GB shared with the GPU.

```bash
caffeinate -is nohup ./.venv/bin/python scripts/train.py \
    --config configs/train_m5.yaml > runs/train.log 2>&1 &
```

`caffeinate -is` is not optional — the run dies with the display otherwise.

If the measured rate makes 64×64 infeasible, `data.image_size=32` (and
`attention_resolutions=[8,4]`) is ~4× less compute per step; the codebase is
resolution-agnostic.

#### Cleaning Epsilon off Hyak

Epsilon's entire Hyak footprint is `/gscratch/rao/aaravs07/epsilon` — the repo
(~13 MB) plus a ~6 GB venv. The HF parquet cache cleaned itself up as it went
(the fetcher unlinks each shard after decoding), and home shows 0/10 GB.

```bash
du -sh /gscratch/rao/aaravs07/epsilon          # look before you delete
rm -rf /gscratch/rao/aaravs07/epsilon          # exact path, no wildcard
```

🚫 **`/gscratch/rao/aaravs07/nsl` is a different project and must never be
touched.** Never use a wildcard under `/gscratch/rao/aaravs07/`.

---

### Local (MacBook, MPS) — development scale only

```bash
./.venv/bin/python scripts/train.py --config configs/train_unet.yaml \
    data.max_samples=50000 training.batch_size=32 \
    training.mixed_precision=none logging.wandb=false
```

`mixed_precision=none` and `logging.wandb=false` are both required locally
(autocast bf16/fp16 is CUDA-gated in the trainer; wandb is not installed).

### Hyak (klone) — production scale

Account state, from the access email — **do not re-derive this**:

| | |
|---|---|
| Login | `ssh aaravs07@klone.hyak.uw.edu` |
| Group / account | `u_hyak_rao` → `--account=rao` |
| Work directory | `/gscratch/rao/aaravs07/epsilon` |
| Home quota | **10 GB** — keep caches off it (the launcher redirects `HF_HOME`, `TORCH_HOME`) |
| Do **not** use | `/gscratch/cse` (explicitly per the access email) |
| Job mail | `aaravs07@uw.edu` |
| Partition | `gpu-rtx6k` — ✅ verified via `hyakalloc`, 2026-08-07 |

### 7.1 What the rao allocation actually is

`hyakalloc`, 2026-08-07:

| Account | Partition | CPUs | Memory | GPUs |
|---|---|---|---|---|
| **rao** | **gpu-rtx6k** | 20 | 188 G | **4** (all idle) |
| cse | gpu-a100 | 26 | 377 G | 4 (1 free) |
| cse | gpu-l40s | 224 | 2645 G | 14 (1 free) |
| — | ckpt-all (idle cluster-wide) | 663 | — | **66** |

**There is no `gpu-a40` under this account** — the earlier config was a guess
and was wrong. The `cse` rows are a *different allocation*; the access email
said not to use `/gscratch/cse` storage, and entitlement to `cse` **compute**
has not been confirmed. Do not submit there without checking.

`gpu-rtx6k` = NVIDIA **Quadro RTX 6000**: Turing (sm_75), 24 GB GDDR6.
Turing means **no bf16 tensor cores** (§10.13) and only 20 CPUs total across
4 GPUs, so `num_workers` must stay modest.

### 7.2 The scale problem — read before promising FID < 12

The flagship recipe (273M U-Net, global batch 1024, 400k steps) was written for
**8×A100**. The rao allocation is **4× Quadro RTX 6000**. The gap is not small:

- Per card: A100 does ~312 TFLOPS bf16; Quadro RTX 6000 does ~32.6 TFLOPS fp16
  with fp32 accumulate (the mode mixed-precision training actually uses).
  That is roughly an order of magnitude per GPU, before counting half as many.
- 24 GB vs 80 GB forces per-GPU batch 128 → 16 plus gradient checkpointing,
  and grad accumulation to hold the global batch, which costs more wall-clock.

Combined, the same recipe is **1–2 orders of magnitude slower here**. 400k
steps is not reachable in a summer program on this partition. Treat any
"400k steps / FID < 12" claim as unfunded until it is backed by measured it/s.

Options, roughly in order of preference:

1. **Use `ckpt-all`** — 66 idle GPUs cluster-wide, including Ampere/Ada parts
   that *do* support bf16. Preemptible with ~4 h requeues, which the launcher
   already auto-resumes through; needs a small `ckpt_every`.
   `sbatch --partition=ckpt-all --gpus=a40:4 scripts/launch_hyak.sh ...`
2. **Shrink the model** — `configs/train_rtx6k.yaml` is 92.5M params (128
   channels, 2 res blocks), which fits 24 GB comfortably.
3. **Cut the step budget** and report the FID you actually reach, honestly.
4. **Drop to 32×32**, which is ~4× less compute per step, if 64×64 proves out
   of reach. The codebase is resolution-agnostic.

The decision is downstream of one measurement: **run the smoke test, read the
it/s, multiply.** Do that before committing to any target.

A partition mismatch is rejected at submit time, so it is cheap to verify but
will waste a cycle if you skip it.

```bash
# once, on the login node
mkdir -p /gscratch/rao/aaravs07 && cd /gscratch/rao/aaravs07
git clone <your-repo> epsilon && cd epsilon
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt -e .
hyakalloc                              # confirm account + partition, then edit launch_hyak.sh

sbatch scripts/launch_hyak.sh configs/train_unet.yaml
squeue -u aaravs07
tail -f logs/slurm-*.out
```

`launch_hyak.sh` already handles: `--requeue` with auto-resume from
`ckpt_latest.pt` (checkpoint partitions preempt), `torchrun` one process per
GPU (DDP), bf16 autocast, gradient clipping, EMA 0.9999, and **W&B in offline
mode** — compute nodes have no outbound internet. Run
`wandb sync logs/wandb/offline-*` from the login node to upload.

### The FID < 12 recipe (`configs/train_unet.yaml`)

ADM-class U-Net, **273.0M params** (verified): 192 base channels, mult 1/2/3/4,
3 res blocks, attention at 16 and 8. CFM loss on the CondOT path. Global batch
1024 (128/GPU × 8). Constant LR 1e-4 after 5k warmup. η = 0.1 label dropout.
400k steps. Evaluate the **EMA** weights with 100-step Euler.

DiT comparison backbone: `train_dit.yaml`, DiT-B/4, **130.4M params** (verified),
600k steps. Scale to DiT-L/4 only after the U-Net baseline is verified.

⚠️ **400k steps is aspirational until measured.** Step 5 of §1 exists precisely
to replace it with a number derived from observed it/s on the actual allocation.

### Guidance scale: two different numbers, on purpose

- `eval.sample_guidance_scale = 4.0` — for **preview grids**. High w looks
  better per-image.
- FID evaluation sweeps **w ∈ [1.0, 2.0]**, optimum usually ≈ 1.5. High w
  destroys FID's diversity term.

These disagreeing is correct, not a bug. Don't "unify" them.

---

## 8. Evaluation

```bash
# one-time: export the real reference (seeded random subset, never a prefix)
./.venv/bin/python scripts/evaluate_fid.py export-ref \
    --data-root data/imagenet64 --out data/fid_ref --num 50000

# FID + IS + precision/recall
./.venv/bin/python scripts/evaluate_fid.py run --ckpt runs/unet64/ckpt_latest.pt \
    --ref data/fid_ref --num 50000 --guidance 1.5 --steps 100
```

Report the **50k** number. In-training FID is available via `eval.fid_every`
and `eval.fid_reference_dir` (heavy; off by default). `_fid()` in the trainer
swallows exceptions by design — FID must never kill a long run.

Needs `torch-fidelity` (not installed locally yet).

---

## 9. Sampling & web demo

```bash
./.venv/bin/python scripts/sample.py --ckpt runs/unet64/ckpt_latest.pt \
    --classes 207 88 979 417 --guidance 4 --method ode

EPSILON_CKPT=runs/unet64/ckpt_latest.pt \
    ./.venv/bin/uvicorn eps.web.app:app --host 0.0.0.0 --port 7860
```

The SPA lets a visitor type a prompt (fuzzy-matched to an ImageNet class),
choose ODE vs SDE, velocity vs score, guidance scale, and step count. Without a
checkpoint the UI still loads and `/api/generate` returns 503 — deployable
before training finishes. Deploy anywhere a Python process runs (HF Spaces /
Railway / VPS): `pip install -e .[web]`, set `EPSILON_CKPT`, run uvicorn.

---

## 10. Traps and known issues

**10.1 — The repo is not under git.** `git status` → "not a git repository".
The entire Hyak deploy path is `git clone`, so this blocks everything. Before
`git init`, write a `.gitignore` covering at minimum:
`.venv/`, `data/`, `runs/`, `logs/`, `*.zip`, `*.dmg`, `*.pt`, `*.npz`,
`__pycache__/`, `.pytest_cache/`, `eps/data/imagenet64/`.
The 11.3 GB zip and the 125 MB dmg must never enter git history.

**10.2 — venv pip.** See §3. Use `./.venv/bin/pip`, never `pip` after activate.

**10.3 — `logging.wandb: true` in both training configs** crashes locally
because wandb isn't installed. Override to `false` for every local run.

**10.4 — ✅ FIXED 2026-08-07.** The web demo used to assign
`cfg.sampling.solver` / `.sigma` from the request *before* taking `_lock`, so
concurrent visitors could generate with each other's settings. `sample_batch`
now takes `solver=` and `sigma=` overrides and the app passes them per request;
`cfg` is never mutated. Verified: Euler vs Heun and σ=0 vs σ=2 produce
different output while `cfg.sampling` stays at its defaults.

**10.5 — `@app.on_event("startup")`** is deprecated in the installed FastAPI
(0.141). Works today, emits a DeprecationWarning; migrate to a `lifespan`
handler eventually.

**10.6 — `ImageNet64` holds the whole split in RAM** as uint8: ~16 GB for the
full 1.28M training set. Fine on a Hyak node with `--mem=180G`; will thrash a
laptop. Use `data.max_samples` locally.

**10.7 — Config rejects unknown keys** (`_build` raises `KeyError`). A typo in
a YAML key is a hard failure, deliberately. Dotted overrides validate too.

**10.8 — `--limit` on the fetcher is a class prefix, not a random sample.**
See §6.3.

**10.9 — `configs/train_unet.yaml`'s header comment says "~280M params"**; the
measured value is 273.0M. Cosmetic.

**10.10 — `pip install` on Hyak looks hung at "68/73 [torch]". It isn't.**
torch (526 MB wheel) plus the CUDA libraries unpack ~6 GB of small files onto
`/mmfs1/gscratch` (GPFS). 5–15 minutes is normal. Interrupting leaves a
half-written torch. Afterwards run `.venv/bin/pip cache purge` — pip caches
~2.7 GB of wheels in `~/.cache/pip`, against the 10 GB home quota.

**10.16 — `hyakstorage` output is a cached report, not live.** After deleting
Epsilon's ~6 GB tree on 2026-08-10 it still showed an unchanged `70GB /
164250 files`, byte- and file-identical to before — the tell that it is stale,
since removing a venv deletes tens of thousands of files. Use
`du -sh /gscratch/rao/aaravs07/` for live usage; `hyakstorage` catches up on its
own schedule. Do not conclude a delete failed from an unchanged quota report.

**2026-08-10: Epsilon's Hyak tree is deleted.** `/gscratch/rao/aaravs07/` now
holds only `conda-envs`, `miniforge3`, and `nsl` — all NSL infrastructure, all
off limits (§10.14).

**10.14 — 🚫 DO NOT DELETE ANYTHING UNDER `/gscratch/rao/aaravs07` TO FREE
SPACE.** As of 2026-08-08 that path holds ~70 GB / 164k files, and most of it
belongs to a **separate, active research project of Aarav's** — not Epsilon:
`/gscratch/rao/aaravs07/nsl`, the Neural Systems Lab ProcTHOR zero-shot-transfer
benchmark (PI Rajesh Rao, mentor Vishwas Sathish), which is targeting a LEAP @
CoRL 2026 workshop deadline. That tree holds its replay buffers, per-seed result
trees, and checkpoints. It is the **higher-priority** project of the two;
Epsilon is the side project.
Aarav said explicitly on 2026-08-08 not to touch it. Epsilon's disk problems
are to be solved by putting Epsilon's data *elsewhere* (see §10.15), never by
reclaiming that 70 GB. Do not propose it again.

**10.15 — `/gscratch/rao` is 100% FULL** (4077 / 4096 GB group-wide,
`hyakstorage`, 2026-08-08). Roughly 19 GB of headroom exists for the entire rao
group, which is why the data fetch died with `OSError: [Errno 122] Disk quota
exceeded` at 1.12M / 1.28M images. Note this also means Aarav's *other* project
cannot write to /gscratch/rao right now either.

The fetcher's peak requirement is **~2x the dataset size** — it writes a
~16 GB flat scratch file, then reads it back permuted into ~16 GB of npz
shards, so ~32 GB at peak. Neither the 32 GB peak nor the 16 GB result fits in
19 GB. Home (10 GB) is far too small. `/gscratch/cse` has ~3.5 TB free but the
access email says do not use it. Resolution in progress — see the 2026-08-08
log entry.

**10.13 — Quadro RTX 6000 is Turing (sm_75) and has NO bf16 tensor cores.**
Before 2026-08-07 the trainer silently ran **fp32** when `mixed_precision=bf16`
was set on such a card (the `elif` only caught an explicit `fp16`), and newer
PyTorch's `torch.cuda.is_bf16_supported()` can return True for *emulated* bf16,
which is slower than fp32. `trainer.py` now gates on compute capability
directly, falls back to fp16, and prints a warning. Configs for `gpu-rtx6k`
should still say `fp16` explicitly.

**10.11 — ✅ RESOLVED 2026-08-08.** torch 2.13.0 installs a CUDA **13** build (`nvidia-cudnn-cu13`,
`nvidia-nccl-cu13`, `cuda-toolkit==13.0.3`, `nvidia-cublas==13.1.1.3`).
CUDA 13 needs NVIDIA driver r580+. If Hyak's GPU nodes run an older driver,
torch imports fine on the login node and fails on compute nodes with "CUDA
driver version is insufficient for CUDA runtime version". **Check before
queueing anything:**

```bash
srun -A rao -p gpu-rtx6k --gpus=1 --time=00:10:00 --pty bash
nvidia-smi     # driver version
.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Ran on compute node `g3026`, 2026-08-08: `True Quadro RTX 6000`. The driver is
new enough for the CUDA 13 runtime; **no cu124 reinstall is needed.** Kept here
because it is the first thing to re-check if torch is ever reinstalled or the
partition changes — the failure mode is invisible on the login node.

**10.12 — ✅ FIXED 2026-08-08.** `launch_hyak.sh` used to `module load
cuda/12.4.1` while the pip torch bundles its own CUDA 13 runtime. Now that
10.11 is settled the line is gone: the venv's wheels carry their CUDA via
RPATH, so no `module load cuda` is wanted at all. Do not re-add one.

---

## 11. Design decisions already settled — do not relitigate

- **Package named `eps/`, project dir `epsilon/`.** ε is the noise symbol
  throughout the math, so `import eps` reads as intentional rather than as an
  abbreviation, and nothing is ambiguously nested. Renamed 2026-08-04; all
  imports, configs, scripts and the README were rewritten and re-verified.
- **Plain YAML + dataclasses instead of Hydra.** Full typing, unknown-key
  rejection, dotted overrides, zero magic, and the resolved config is embedded
  in every checkpoint for exact reproducibility.
- **DSM defaults to DDPM weighting.** §5.7.
- **Zero-initialised residual/output branches everywhere.** §5.9.
- **EMA always used for sampling and evaluation.** Large, consistent FID
  improvement at zero training cost. `EMA` has a warmup ramp
  `min(decay, (1+step)/(10+step))` so early weights track the young model
  rather than the random init.
- **The MIT labs** (`lab_one/two/three.ipynb`) were convention cross-checks
  only (time direction, α/β abstraction, CFG combination). **No lab code is
  reused** — this matters for the capstone's originality claim.

---

## 12. Decision & event log

Append here; newest last. Date every entry.

**2026-08-04** — Renamed the package to `eps/`. All imports, configs, scripts,
README rewritten. 66 tests pass; training, sampling, and the web API
re-verified after the move.

**2026-08-04** — Wired `launch_hyak.sh` to the real account: `--account=rao`,
chdir `/gscratch/rao/aaravs07/epsilon`, mail to `aaravs07@uw.edu`, `--requeue`
+ auto-resume, W&B offline, caches redirected off the 10 GB home quota.
`--partition=gpu-a40` guessed from the group name — **later proved wrong**, see
the 2026-08-07 entry.

**2026-08-05** — Opened `train_64x64.zip`: 1,281,149 flat PNGs, no labels, no
label file. Concluded it cannot support the class-conditional capstone. Added
`FlatImageDataset` (reads in place from the zip) and `scripts/pack_data.py` so
the download is still useful for unconditional throughput smoke tests.

**2026-08-05** — Published the explainer artifact (link in §2).

**2026-08-06** — **Data unblocked.** Found `benjamin-paine/imagenet-1k-64x64`:
ungated (verified via HF API), labeled, 1.28M train images, 1.74 GB of parquet.
Removes the image-net.org approval queue from the critical path entirely.
Wrote `scripts/fetch_imagenet_hf.py` to convert it into the official npz layout
the loader already reads. Verified fetch → load → train → sample end to end.

**2026-08-06** — Found and fixed the class-ordering trap (§6.3): the HF train
split is class-sorted, so any prefix of it is a handful of classes. Added a
global shuffle to the fetcher (disk-backed memmap, permuted across *and* within
shards) and changed `evaluate_fid.py export-ref` to take a seeded random subset
with a low-diversity warning. First attempt at the fix was wrong — sorting
indices for sequential I/O left rows class-ordered inside each shard, which
defeated the purpose; the shard block is now permuted in memory after the
sequential read, keeping both the I/O win and the shuffle.

**2026-08-06** — Added `pyarrow` + `huggingface_hub` to `requirements.txt` and
as the `[data]` extra; installed both into the local venv. Updated the README's
data section, which had been telling readers to wait for image-net.org approval.

**2026-08-06** — Measured parameter counts: U-Net 273.0M, DiT-B/4 130.4M.

**2026-08-07** — `git init`, first commit, pushed to the new public repo
`https://github.com/aaarvs07ranger/epsilon`. GitHub had pre-created an MIT
LICENSE; rebased onto it rather than force-pushing. Repo is ~13 MB — the 11 GB
zip, the dmg, and `.venv` are all gitignored. Aarav decided to keep
`PRIME_LECTURE_NOTES.pdf` and the three MIT lab notebooks public after being
shown the copyright/course-policy consideration; that was an explicit call, do
not re-raise it unprompted.

**2026-08-07** — Fetched `data/imagenet64_small` locally (20k images, 235 MB).
Only 16 classes — it is a class prefix, not a miniature ImageNet. Local smoke
tests only; the real dataset gets fetched on Hyak directly from HF, never
transferred from the laptop.

**2026-08-07** — Cloned to `/gscratch/rao/aaravs07/epsilon` on klone and ran
the venv install. Discovered §10.10 (pip appears hung on torch; it isn't) and
§10.11 (torch 2.13.0 pulls a CUDA **13** build — driver compatibility on the
GPU nodes is still unverified and is now the top gating item).

**2026-08-07** — `hyakalloc` run. The rao account holds **`gpu-rtx6k`** (4×
Quadro RTX 6000, 24 GB, Turing) — **not** `gpu-a40`, which does not exist here.
Consequences worked through in §7.1/§7.2 and acted on:
- Fixed a silent trainer bug (§10.13): `mixed_precision=bf16` on a pre-Ampere
  card fell through to **fp32** for the entire run without saying anything.
  Now gated on compute capability, falls back to fp16, prints a warning.
- `launch_hyak.sh` corrected to `gpu-rtx6k`, `--cpus-per-task=20`, and
  documented the `ckpt-all` override path for real scale.
- Added `configs/train_rtx6k.yaml`: 92.5M params, per-GPU batch 16 with 16×
  grad accumulation (global 1024 preserved), gradient checkpointing on, fp16.
  Static GPU memory 1.85 GB of 24 GB, leaving room for activations.
- **Recorded that the FID<12 / 400k-step target is not funded by this
  allocation** (§7.2). Needs `ckpt-all`, a smaller scope, or an honest
  restatement — decided after measuring it/s, not before.

**2026-08-07** — Fixed §10.4, the web demo's shared-config race, by threading
`solver` and `sigma` through `sample_batch` as per-request overrides.

**2026-08-08** — **GPU + driver verified**, closing the top gating item.
`srun` onto `gpu-rtx6k` (node `g3026`) →
`torch.cuda.is_available() == True`, device `Quadro RTX 6000`. The CUDA 13
build works on klone's driver; the cu124 fallback in §10.11 is **not** needed.
Removed the now-pointless `module load cuda/12.4.1` from `launch_hyak.sh`
(§10.12). Critical path is now: purge pip cache → fetch data on the **login
node** (compute nodes have no outbound internet) → smoke test → measure it/s.

**2026-08-08** — **Measured Apple-silicon throughput and proposed moving Epsilon
off Hyak entirely.** Two facts forced this:
- `/gscratch/rao` is 100% full (§10.15) and the 70 GB there belongs to the NSL
  project (§10.14), which has a real conference deadline. Epsilon competing for
  that filesystem and those 4 GPUs actively hurts the higher-priority project.
- The trainer was silently running **fp32 on MPS** — the AMP block armed
  autocast only on CUDA. Same class of bug as the Turing bf16 fallthrough found
  2026-08-07. Fixed: bf16 autocast on MPS (bf16 specifically because MPS has no
  GradScaler and unscaled fp16 underflows). Verified on an M4.

**Measured, M4 (10-core GPU), 92.5M U-Net, 64x64, batch 32, bf16, gradient
checkpointing ON: 0.26 it/s = 8.3 images/s.** This is the first real throughput
number this project has ever had — every earlier estimate was arithmetic.

Extrapolating to the M5 Max (32-core GPU + per-core neural accelerators, and
no gradient checkpointing needed at 36 GB) gives a **rough 5-12x**, i.e. ~50-90
images/s — **unverified, and the single most important thing to measure next.**
At that rate, global batch 256: 100k steps ~ 3-6 days, 150k steps ~ 5-9 days.

Honest comparison: 4x RTX 6000 is probably still **2-4x faster than one M5
Max**, so this is not a speed win — it is an availability, disk, and
project-priority win. Added `configs/train_m5.yaml` accordingly.

**FID < 12 remains unfunded by any hardware now in play.** Recommended
restatement of the target: 92.5M model, 64x64, ~100-150k steps, report the FID
actually achieved with the compute disclosed. Decision on the final target is
Aarav's and is not yet made.

**2026-08-08 (evening)** — **Execution venue changed: Epsilon leaves Hyak.**
Decided with Aarav after the disk failure and the throughput measurement.
Actions taken: added `configs/train_m5.yaml`; rewrote §1 and added §7.0; agreed
Epsilon's Hyak tree gets deleted to give the ~6 GB back to NSL. Aarav is home
with the M5 Max ~2026-08-09/10; everything is blocked until then, and the first
thing to do on that machine is **measure it/s**, not launch a run.

Standing correction to earlier sessions: the repeated pattern in this project
has been *planning from arithmetic and being wrong* — `gpu-a40` guessed from a
group name, "400k steps" carried for weeks, bf16 assumed on Turing, autocast
assumed on MPS, ~19 GB of free disk assumed on a 4 TB filesystem. Every one was
caught only by running something. Measure first.
