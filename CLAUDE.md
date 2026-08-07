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
> factual and dated; write "verified <how>" or "unverified" rather than implying
> certainty you do not have.

---

## 1. Status board

**Last updated: 2026-08-06**

| Area | State |
|---|---|
| Math core (`paths`, `losses`, samplers) | ✅ Complete, 66 tests pass (~1.6 s) |
| Models (U-Net 273.0M, DiT-B/4 130.4M) | ✅ Complete, verified on `meta` device |
| Trainer (DDP, EMA, AMP, ckpt, resume) | ✅ Complete, smoke-tested locally on MPS |
| Web demo (FastAPI + SPA) | ✅ Complete, loads without a checkpoint |
| **Labeled data** | ✅ **Unblocked 2026-08-06** — see §6 |
| Hyak deployment | ⏳ Not started. **Blocked on: repo is not under git** (§10.1) |
| Partition name in `launch_hyak.sh` | ⚠️ `gpu-a40` is a *guess* — run `hyakalloc` and correct |
| Real training run | ⏳ Not started |
| FID reference set | ⏳ Not exported |
| Public demo deployment | ⏳ Not started |

### Immediate next actions, in order

1. **`git init` and push.** Nothing reaches Hyak until this exists — the deploy
   path is `git clone`. This is the only hard blocker right now.
2. **Fetch the labeled data** (§6). ~1.8 GB down, ~16 GB out, one command.
3. **Get onto Hyak**, build the venv, run `hyakalloc`, fix the partition line.
4. **200-step smoke test on Hyak** — proves venv builds, GPUs visible, DDP
   initialises across cards.
5. **Measure it/s** and convert 400k steps into real wall-clock hours. Decide
   step budget from that number, not from the config's aspirational 400k.
6. **Export the 50k FID reference**, then launch the real run.

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
| Partition | `gpu-a40` — ⚠️ **a guess.** Run `hyakalloc` and fix the `#SBATCH` line |

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

**10.4 — Web demo mutates shared config outside its lock.** In
`eps/web/app.py`, `cfg.sampling.solver` and `cfg.sampling.sigma` are assigned
from the request *before* `with _lock:`. Two concurrent requests can interleave
so one generates with the other's solver/σ. Not a crash, and single-user demos
never hit it, but fix before any public deployment — move both assignments
inside the lock, or (better) thread them through `sample_batch`'s existing
override arguments instead of mutating `cfg`.

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
`--partition=gpu-a40` guessed from the group name — still unverified.

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
