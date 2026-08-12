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

**Last updated: 2026-08-12 — 🎉 BOTH MODELS TRAINED. Pod terminated, results local.**

| Area | State |
|---|---|
| Math core (`paths`, `losses`, samplers) | ✅ Complete, 66 tests pass |
| **Trained models** | ✅ **U-Net 92.5M @60k steps · DiT-B/4 130.4M @100k steps.** In `runs/cloud_{unet,dit}/` |
| Sample grids | ✅ 9 grids in `results/` + 64 training previews in `runs/cloud_*/previews/` |
| Trainer | ✅ Single-GPU path proven on CUDA + MPS. **DDP still never executed** |
| `eps/data/` package | ✅ Reconstructed + pushed 2026-08-11 (§10.17) |
| **Web demo** | ✅ **Upgraded 2026-08-12**: loads *both* backbones, has a Compare mode (same seed, both models), CPU-adaptive defaults (§9) |
| **Deployment weights** | ✅ `deploy/*.pt` — EMA-folded fp16, **8× smaller** (1.48 GB → 185 MB), verified against reference (§9.1) |
| **Hosting** | ⏳ Files written and tested (`deploy/space/`, `deploy/DEPLOY.md`); **needs Aarav's HF account to push** |
| README | ✅ Rewritten 2026-08-12 with real results and honest limitations |
| Git / GitHub | ⚠️ several local commits unpushed — SSH key now exists, see actions |
| RunPod | ⛔ **Pod terminated 2026-08-12.** Total spend ~$49 of $60. Nothing left on it |
| M5 Max | Local dev box. Aborted run kept in `runs/m5` (4.2 GB) as the correctness witness |
| Hyak (rao) | ⛔ Abandoned (§10.14/§10.15) |
| Measured throughput | ✅ U-Net 1.64 it/s, DiT 3.20 it/s. **The bigger model is 2× faster** (§7.4) |
| **FID** | ⏳ **Not computed** — deliberately skipped to protect the GPU budget. Runs free locally (§8) |
| **Slides deck** | 📌 **Requested 2026-08-12, not started** — see §13 |

Local-only convenience: `data/imagenet64_small` — 20k images but **only 16
classes** (fish/sharks; `--limit` takes a class prefix, §6.3). Smoke tests only.

### The result, in one line

Two class-conditional 64×64 generators trained from scratch for ~$49, with
recognisable golden retrievers, macaws, red pandas and balloons. At **matched**
60k steps the U-Net is visibly sharper than the DiT — expected, since DiT wants
longer training — while training at **half** the DiT's speed. Both facts are
measured and both belong in the writeup.

### Immediate next actions

Training is done and the compute is switched off. Everything remaining is
writing, shipping, and presenting.

1. **Push.** Several commits sit local. An SSH key now exists at
   `~/.ssh/id_ed25519`; add it at github.com → Settings → SSH keys, then:
   ```bash
   git remote set-url origin git@github.com:aaarvs07ranger/epsilon.git
   git push origin main
   ```
2. **Deploy the demo** — everything is written and locally verified; it needs
   an HF account to push. Follow `deploy/DEPLOY.md`: create a model repo for
   the two fp16 weights, create a Docker Space from `deploy/space/`. Free tier
   works (the UI detects CPU and drops to 30 Heun steps); ZeroGPU is one
   Dockerfile line away.
3. **Build the 25-minute slide deck** — §13 has the outline, the demo beats,
   and which assets to use. Requested 2026-08-12.
4. **Optionally compute FID locally** (§8) — free, just slow, and it would turn
   the qualitative U-Net-vs-DiT claim into a number. `NUM=5000` keeps it
   overnight-sized on an M5 Max.
5. **Revise the explainer artifact** (§2) — stale since 2026-08-08, and
   everything since adds to it.

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
**amber = data / t=1**. Section 10 holds the runbook commands.

⚠️ **The artifact is now STALE as of 2026-08-10 and needs a revision pass.**
Read 2026-08-08's decisions before quoting it. Specifically wrong: its whole
Hyak section (abandoned, §7.0); `--partition=gpu-a40` (never existed, §7.1);
"request the labeled ImageNet download from image-net.org, do this first" (the
HF repack replaced it, §6.2); `mixed_precision=none` for local runs (bf16 works
on MPS now); and the FID<12 / 273M framing (§7.2). The math sections 01-09 are
still accurate — the rot is confined to the runbook and the plan.

---

## 3. Environment

**Three machines now.** Training moved to **rented RunPod GPUs** on 2026-08-11
(§7.4) — 2× RTX PRO 6000 Blackwell, torch 2.8.0+cu128, driver 570, Ubuntu
24.04, reached at `ssh root@82.221.170.234 -p 27225 -i ~/.ssh/id_ed25519`.
The M5 Max below is now the local dev box (it *can* train, at ~5.5 days/run);
the M4 is where everything before 2026-08-11 was measured.

An SSH keypair was created at `~/.ssh/id_ed25519` on 2026-08-11 — there was
none before, which is why HTTPS `git push` failed with "could not read
Username". The same key works for RunPod and GitHub.

| | M5 Max — training | M4 — dev |
|---|---|---|
| Project root | `/Users/mohit/Desktop/epsilon` | `/Users/aarav/Desktop/epsilon` |
| Chip | **M5 Max, 18-core CPU / 32-core GPU** | M4, 10-core GPU |
| Memory | 36 GB unified (MPS may use ~30.2 GB) | 32 GB unified |
| Free disk | **1.7 TB** | ~700 GB |
| OS | macOS 26.5 (darwin 25.5.0) | macOS (darwin 25.5.0) |
| Python | 3.13.13 (in `.venv`) | 3.13.6 |
| torch / numpy | 2.13.0 / 2.5.2, **MPS + bf16 verified** | 2.13.0 / 2.5.1 |
| Tests | `./.venv/bin/python -m pytest -q` → 66 passed | 66 passed |

Python package is `eps/` — so `from eps.paths import ...`.

### Installed vs. not (M5 Max venv, as of 2026-08-11)

Installed via `pip install -e ".[data,web,dev,eval]"`. Present: `torch`,
`torchvision`, `numpy`, `scipy`, `pyyaml`, `pillow`, `fastapi`, `uvicorn`,
`pyarrow` 25.0.1, `huggingface_hub` 1.27.0, `pytest`, and — **new on this
machine** — `torch-fidelity` 0.4.0 and `clean-fid` 0.1.35, so
`scripts/evaluate_fid.py run` works here. That closes the gap noted on the M4.

**Still missing, deliberately:** `wandb`. `train_m5.yaml` already sets
`logging.wandb: false`, so nothing needs overriding for the M5 run — but
`train_unet.yaml` / `train_dit.yaml` still say `true` and will crash on
`import wandb`. Pass `logging.wandb=false` if you use those (§10.3).

### venv gotchas (bite every session)

1. `source .venv/bin/activate && pip install ...` resolves to the **system**
   pip and fails with a PEP 668 "externally-managed-environment" error.
   Always use the explicit interpreter path:

   ```bash
   ./.venv/bin/pip install <pkg>          # correct
   ./.venv/bin/python -m pytest -q        # correct
   ```

2. **The stock `python3` on this Mac is 3.9.6** (Xcode CLT), below the
   `requires-python = ">=3.10"` floor. A `.venv` built with it installs
   nothing useful — that is exactly what was sitting in the fresh clone on
   2026-08-11. Real 3.13 lives at `~/.local/bin/python3.13`, with `uv` beside
   it at `~/.local/bin/uv`. Rebuild with:

   ```bash
   rm -rf .venv && uv venv --python ~/.local/bin/python3.13 .venv
   VIRTUAL_ENV=$PWD/.venv uv pip install -e ".[data,web,dev,eval]"
   ```

3. **`uv venv` does not put `pip` in the venv.** `./.venv/bin/pip` then does
   not exist and gotcha 1's advice silently has nothing to run. `pip` has been
   installed into this venv explicitly so the documented commands work; if you
   ever rebuild the venv with `uv`, run `uv pip install pip` again.

---

## 4. Repository layout

```
epsilon/                      # project root (git repo since 2026-08-07)
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

> **Read this first — the subsections are NOT in numerical order in this file,
> they are in chronological order of the venue changes.** Where to look:
>
> | If you want | Go to | Status |
> |---|---|---|
> | **What is running right now, and the numbers it is running at** | **§7.4** | ✅ current |
> | How the cloud run is set up and why no DDP | §7.3 | ✅ current |
> | Apple-silicon throughput and the local config | §7.0 | dev box only |
> | Why FID < 12 is dead, with the arithmetic | §7.2 | ✅ current |
> | The Hyak allocation | §7.1 | ⛔ historical |
>
> Venue history: Hyak → M5 Max (2026-08-08) → RunPod (2026-08-11). Sections
> from dead venues are kept because they explain the decisions, not because
> anyone should act on them.

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

#### Measured throughput — ✅ MEASURED 2026-08-11, no longer an estimate

| Machine | Config | Rate |
|---|---|---|
| **M4, 10-core GPU** | 92.5M U-Net, 64×64, batch 32, bf16, grad-ckpt ON | **0.26 it/s = 8.3 img/s** |
| **M5 Max, 32-core GPU** | same model, batch 64 × 4 accum = 256, grad-ckpt OFF, `num_workers=2` | **0.22 it/s = 56 img/s** |
| **M5 Max — as actually configured** | same, but `num_workers=0` (required, §10.18) | **0.20 it/s = 51 img/s** |

200 steps, stable across all eight 25-step windows (no thermal decay over
~15 min); loss 1.236 → 1.154. **6.2× the M4** at the configuration actually
used — inside the 5–12× band that was predicted, so for once the extrapolation
held. Use **0.20 it/s** for planning: `num_workers=0` is forced by §10.18, and
its ~9% cost is real.

Wall-clock at 0.20 it/s, global batch 256:

| Steps | Wall-clock (measured basis) |
|---|---|
| 50k | **2.9 days** |
| 100k | **5.8 days** |
| 150k | **8.7 days** |

That is the pessimistic end of the earlier 3–6 / 5–9 day estimate.

**Confirmed at full scale** on the real run (all 1,281,167 images resident,
not the 4,096-image benchmark subset): steps 50 and 100 gave **0.22 and 0.20
it/s** — the benchmark transferred honestly, so 0.20–0.22 is the real range and
~5.3–5.8 days for 100k steps. The remaining caveat is thermal: a run of days
throttles harder than 15 minutes does.

⚠️ **Watch item — memory is fully committed.** With the split resident,
`vm_stat` during the run shows ~21 GB wired, ~6 GB in the compressor, and
<1 GB free of 36 GB; swap is heavily used (partly by other apps — Cursor, a
Streamlit server). Throughput is holding so far, but each step touches 256
random rows across 15.7 GB, so if pages get pushed to swap the rate will decay.
If `it/s` drops materially over the coming hours, in increasing order of effort:
1. Close other memory-hungry apps.
2. `data.max_samples=640000` — halves resident memory, still 500 images/class.
3. **Back `ImageNet64` with a memmap over the npz members instead of loading
   into RAM.** The shards are uncompressed, so this is very doable and is the
   right design for this machine: resident memory drops to ~0 and the OS page
   cache serves reads from a local SSD, with no swap involved. It would also
   make `num_workers > 0` safe again (§10.18), recovering that 9%.

Larger micro-batches at the same global batch (128 × 2, 256 × 1) were tried and
**abandoned as slower** — 128 had not finished 45 steps in the time batch 64
needed for ~100. Not fully characterised, but there is no win there; 64 × 4
stands.

#### Running it

`configs/train_m5.yaml` — 92.5M params, bf16, **no** gradient checkpointing
(that exists only to fit 24 GB and costs ~30% compute), global batch 256 via
64 × 4 accumulation, and `num_workers: 0` — which is a **correctness**
requirement on macOS, not a tuning knob. See §10.18 before changing it.

The command actually used to launch the 2026-08-11 run:

```bash
mkdir -p runs
PYTHONUNBUFFERED=1 nohup caffeinate -is ./.venv/bin/python scripts/train.py \
    --config configs/train_m5.yaml \
    training.total_steps=100000 \
    eval.sample_every=1000 \
    logging.ckpt_every=2500 \
    > runs/train.log 2>&1 &
```

- `caffeinate -is` is not optional — the run dies with the display otherwise.
- `PYTHONUNBUFFERED=1` matters: piped to a file, Python block-buffers stdout
  and the log looks frozen for many minutes. Cost an interim reading during
  the benchmark before it was added.
- `eval.sample_every=1000` ≈ 83 min between preview grids at 0.20 it/s; the
  config default of 5000 is ~7 h, too coarse to notice a problem early.
- `logging.ckpt_every=2500` ≈ 3.5 h, bounding what a crash costs. 40
  checkpoints × 1.48 GB ≈ 59 GB — fine at 1.7 TB free, and worth it.

**`total_steps` only sets where the run stops.** `lr_schedule: constant` means
no schedule depends on it, so stopping early at any checkpoint is legitimate,
and extending later is just:

```bash
... scripts/train.py --config configs/train_m5.yaml \
    --resume runs/m5/ckpt_latest.pt training.total_steps=150000
```

Monitoring:

```bash
tail -f runs/train.log                   # loss + it/s every 50 steps
open runs/m5/previews/                   # a grid every 1000 steps
```

If the measured rate makes 64×64 infeasible, `data.image_size=32` (and
`attention_resolutions=[8,4]`) is ~4× less compute per step; the codebase is
resolution-agnostic.

### 7.3 Cloud execution — RunPod, two single-GPU arms (decided 2026-08-11)

**Why.** The M5 Max works but costs ~5.5 days per run, and the project wants
*two* models (U-Net and DiT) plus FID and visualisations. Aarav opted to rent
GPUs. Measured local rate 0.21 it/s = 54 img/s; expected ~1100–1350 img/s on an
H100, i.e. **~6 h for 100k steps instead of 5.5 days, for roughly $40 total.**

**Why two single-GPU processes and NOT DDP / torchrun.** The trainer's DDP path
has **never been executed** — Hyak was abandoned before any training happened,
so `launch_hyak.sh`, `wrap_ddp`, and the `DistributedSampler` branch are all
unproven. The single-process path has verified hours behind it. Renting an
8-GPU box to debug DDP by the hour is the expensive way to find that out. So:
one arm per GPU, two independent processes, zero DDP risk, both models in the
same wall-clock.

If DDP is ever wanted, validate it on the cheapest possible 2-GPU box first and
treat that as its own task — do not fold it into a paid training run.

**The configs are a controlled experiment.** `train_cloud_unet.yaml` and
`train_cloud_dit.yaml` are byte-identical apart from `model:` and
`logging.output_dir`. Verified programmatically 2026-08-11: comparing every
field of `path`, `data`, `training`, `eval` between the two produced **no
differences**. Keep it that way — if you change a training knob in one, change
it in the other or the comparison means nothing.

| Arm | Model | Params | Global batch | Steps | Images |
|---|---|---|---|---|---|
| `train_cloud_unet.yaml` | ADM-style U-Net | 92.5M | 256 | 100k | 25.6M |
| `train_cloud_dit.yaml` | DiT-B/4, pixel space | 130.4M | 256 | 100k | 25.6M |

Both are the *same* 92.5M U-Net measured locally, so the cloud speedup reads
directly off the it/s. Note the arms are **not** parameter-matched (92.5M vs
130.4M); there is no clean equal-size pairing at standard DiT sizes, so
describe it as "DiT-B/4 vs a 92.5M U-Net", not as matched capacity.

⚠️ **DiT is expected to lose at a matched 100k steps.** The original configs
budgeted DiT 600k steps against the U-Net's 400k precisely because DiT wants
longer training for the same FID. A U-Net win here is a statement about *this
compute budget*, not about the architecture. Say that explicitly.

**Scripts.**

```bash
bash scripts/setup_runpod.sh --verify   # GPU/driver/disk preflight only
bash scripts/setup_runpod.sh            # install -> data -> launch both arms
bash scripts/eval_compare.sh            # FID sweep + grids -> results/
bash scripts/eval_compare.sh --grids-only   # just the pictures, ~2 min
```

`setup_runpod.sh` deliberately **reuses the container's existing torch** via
`venv --system-site-packages` rather than installing our own. That sidesteps
§10.11 entirely: the container's torch already matches its driver, whereas
`pip install torch` pulls a CUDA 13 build that needs r580+. It also hard-fails
early if the clone lacks `eps/data/` (§10.17), rather than crashing 25 minutes
later after the data fetch.

`num_workers: 8` in both cloud configs is safe **because Linux forks** its
dataloader workers and they share the 15.7 GB split copy-on-write. That same
value on macOS is fatal — see §10.18 before copying it anywhere local.

**What to check in the first two minutes of a paid run:** the `it/s` in each
log. Multiply by 256 for images/s. Under ~2 it/s on an H100 means something is
wrong (oversubscribed host, throttling, starved dataloader) — catch it then,
not six hours and $40 later.

### 7.4 The live RunPod run — measured 2026-08-11

**Pod.** `prime_genai_epsilon`, id `x850wa2dgcc0aj`, Secure Cloud, On-Demand.
2× **NVIDIA RTX PRO 6000 Blackwell Server Edition** (sm_120, 97,887 MiB each,
native bf16), 16 vCPU, **$4.20/hr**. 100 GB volume at `/workspace`, 30 GB
container disk (ample — all caches are pinned to the volume). Ubuntu 24.04,
driver **570.195.03**, container torch **2.8.0+cu128** (reused, not reinstalled
— §7.3). SSH: `ssh root@82.221.170.234 -p 27225 -i ~/.ssh/id_ed25519`
(the "SSH over exposed TCP" endpoint — the other one does not support SCP).

Setup went clean end to end: 66 tests passed, full fetch (1,281,167 train +
50,000 val, **1000/1000 classes**), 50k-prefix diversity check passed. Data
fetch took ~25 min as predicted.

#### Measured throughput — and a genuinely surprising result

| Arm | Params | Rate | images/s | vs M5 Max |
|---|---|---|---|---|
| U-Net | 92.5M | **1.64 it/s** | 420 | 7.8× |
| DiT-B/4 | 130.4M | **3.20 it/s** | 819 | ~15× |

**The bigger model is 2× faster.** DiT-B/4 has 41% more parameters and trains
at double the U-Net's rate. This is not a bug: DiT is dense matmuls that
saturate tensor cores, whereas the U-Net at 64×64 with 128 base channels is
many small convolutions, GroupNorm, and attention at 16×16/8×8 — low arithmetic
intensity per kernel. On a fast GPU the U-Net simply cannot feed the machine.
Worth reporting in the writeup; it is a real, measured architectural finding
and it inverts the naive parameter-count intuition.

⚠️ **A hypothesis that was tested and REJECTED — do not retry it.** The obvious
read of the above is "the U-Net is kernel-launch-bound, so give it a bigger
micro-batch." That was tried: batch 128×2-accum → **1.63 it/s**, batch 256×1 →
**1.64 it/s**. *No change.* `nvidia-smi` showed 100% utilisation and 504W/600W
at both settings — it is genuinely compute-bound, not launch-bound. Memory sat
at 27.9 GB (batch 128) / 53.9 GB (batch 256) of 97.9 GB, so there was plenty of
headroom; the headroom simply was not the constraint. Note this is the *reverse*
of the M5 Max finding (§7.0), where bigger batches were slower for memory
reasons. Different machine, different regime, and neither generalises.

#### The scope cut this forced

At 1.64 it/s the U-Net needs **17 h** for 100k steps — over both the 12-hour
window Aarav wanted and the $60 budget (~$71 for the pod alone). Decision:

| Arm | Steps | Wall-clock | Why |
|---|---|---|---|
| U-Net | **60,000** | ~10.1 h | Largest count that fits 12 h at 1.64 it/s |
| DiT | 100,000 | ~8.2 h | Already fast; no reason to cut it |

15.4M images ≈ 12 epochs for the U-Net. **The comparison stays matched for
free**: checkpoints save every 10k, so the head-to-head is U-Net@60k vs DiT's
`ckpt_0060000.pt`, with DiT@100k as a bonus data point. Nothing is wasted.

The U-Net was relaunched with `training.total_steps=60000` so it stops on its
own — no babysitting, and no risk of burning credits overnight.

Cost tracking: $60.00 start → $57.75 after setup → projected **~$53 total**
(10.1 h training + ~2.5 h eval at $4.20/hr). Aarav topped up ~$15 for margin.

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

### Local smoke tests — development scale only

```bash
./.venv/bin/python scripts/train.py --config configs/train_m5.yaml \
    data.root=data/imagenet64_small data.max_samples=4096 \
    training.total_steps=200 logging.output_dir=/tmp/bench
```

⚠️ **Corrected 2026-08-11.** This block used to pass
`training.mixed_precision=none`, on the grounds that "autocast bf16/fp16 is
CUDA-gated in the trainer". **That has been false since 2026-08-08**, when the
MPS autocast bug was fixed — bf16 autocast now works on Apple silicon and
forcing `none` just runs fp32 and throws away ~30% of the speed. Leave
`mixed_precision: bf16`.

`logging.wandb=false` is still needed for `train_unet.yaml` / `train_dit.yaml`
(wandb is not installed, §10.3). `train_m5.yaml` already sets it, so the
command above needs no override.

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

**2026-08-11: this is no longer an argument, it is arithmetic on a measured
number.** Epsilon runs on one M5 Max at **0.20 it/s at global batch 256**
(§7.0). Against the flagship recipe:

| | Flagship (8×A100) | What this machine does |
|---|---|---|
| Params | 273.0M | 92.5M |
| Global batch | 1024 | 256 |
| Steps | 400k | ~100k (5.8 days) |
| **Images seen** | **410M ≈ 320 epochs** | **25.6M ≈ 20 epochs** |

So the run now training sees **1/16 the images** with **1/3 the parameters**.
Running the *actual* flagship recipe here would cost roughly 4× (batch) × ~3×
(params) ≈ 12× per step, i.e. ~0.017 it/s → **400k steps ≈ 270 days.** It is
not a question of patience.

**Therefore: FID < 12 will not be reached, and the writeup should not claim
it.** The honest framing, and the one the artifact and README should adopt:

> A 92.5M-parameter class-conditional flow-matching model trained from scratch
> on ImageNet-64 for N steps (25.6M images) on a single consumer laptop GPU,
> reaching FID X. Published ImageNet-64 numbers (ADM 2.07) use 1–2 orders of
> magnitude more compute; the comparison is one of method, not of scale.

That framing is *stronger* for a capstone than a missed target, because the
verification discipline (§2) is the actual contribution. Pair it with the §6.4
caveat about the Lanczos/JPEG repack, which independently makes absolute FID
non-comparable.

**The final target is Aarav's call and is still not made.** What is now settled
is that "FID < 12" is not among the options.

---

Historical, for the Hyak allocation (kept because it explains the decisions):

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

⛔ **Do not run `export-ref` while training is running.** It builds
`ImageNet64(...)` with **no `max_samples`**, so it loads a second full 15.7 GB
copy of the training split. The trainer already holds one and the machine sits
at <1 GB free (§10.6) — running both will thrash and can take the run down.
Either wait until training stops, or point it at the (much smaller) validation
split:

```bash
./.venv/bin/python scripts/fetch_imagenet_hf.py --split validation \
    --out data/imagenet64                       # 50k images -> val_data.npz, ~0.6 GB
./.venv/bin/python scripts/evaluate_fid.py export-ref \
    --data-root data/imagenet64 --out data/fid_ref --num 50000 --split val
```

⚠️ **The two scripts spell the split differently.** The fetcher takes
`--split validation` (choices: `train|validation|test`); `evaluate_fid.py`
takes `--split val` (choices: `train|val`). Passing `validation` to the
evaluator is an argparse error. `ImageNet64` itself accepts `val`, `valid`, or
`validation` and globs `val_data*.npz` for all three.

The validation split is already shuffled upstream (§6.3), so it needs no global
permutation, and 50k images is exactly the reference size. Note that FID
against a *validation*-derived reference is a slightly different quantity than
against a train-derived one — pick one and say which in the writeup.

**Budget ~2.5 h for the full sweep**, not the ~45 min originally guessed here.
`scripts/eval_compare.sh` runs 6 evaluations (2 models × 3 guidance values) ×
10k samples × 100 ODE steps, and CFG doubles the function evaluations per step.
On 2× RTX PRO 6000 that is roughly 80 min for the U-Net arm and 40 for the DiT,
plus reference export and Inception feature extraction. `NUM=5000` halves it;
FID is biased at low sample counts but both arms get identical treatment, so
the *comparison* stays valid as long as the count is disclosed.

Report the **50k** number if you can afford it, **10k** is the usual ablation
size. In-training FID is available via `eval.fid_every` and
`eval.fid_reference_dir` (heavy; off by default). `_fid()` in the trainer
swallows exceptions by design — FID must never kill a long run, and as of
2026-08-11 `_preview()` is guarded the same way.

`torch-fidelity` 0.4.0 and `clean-fid` 0.1.35 **are** installed on the M5 Max
(§3), so this works here — the "not installed locally" note applied to the M4.

---

## 9. Sampling & web demo

```bash
./.venv/bin/python scripts/sample.py --ckpt runs/cloud_unet/ckpt_final.pt \
    --classes 207 88 979 417 --guidance 3 --method ode

# the demo, with BOTH trained backbones loaded
EPSILON_MODELS="deploy/unet_60k.pt,deploy/dit_100k.pt" \
    ./.venv/bin/uvicorn eps.web.app:app --host 0.0.0.0 --port 7860
```

### 9.0 What the demo does (upgraded 2026-08-12)

The SPA takes a prompt (**fuzzy-matched to one of 1000 ImageNet classes** —
this is not text-to-image, §11), and exposes ODE vs SDE, velocity vs score,
guidance scale, step count, σ, and seed. Without a checkpoint the UI still
loads and `/api/generate` returns 503, so it is deployable before any model
exists.

Three things were added on 2026-08-12:

- **Multi-model loading.** `EPSILON_MODELS` takes a comma-separated list of
  checkpoints; labels are derived from each checkpoint's own embedded config,
  so nothing has to be kept in sync by hand. `EPSILON_CKPT` still works and is
  appended to the list. `_models` is a dict keyed by architecture slug; two
  checkpoints of the same architecture are disambiguated by step.
- **Compare mode.** Generates from every loaded backbone at the **same seed**
  and shows them side by side. The seed is pinned client-side before the
  requests go out — otherwise you are comparing two random draws, not two
  models. This is the project's headline result made clickable, and it is the
  single best thing to demo live.
- **CPU-adaptive defaults.** `/api/health` reports the device; if it is CPU the
  UI drops to 30 **Heun** steps and says so in a hint. Heun is 2nd-order, so 30
  Heun steps ≈ 60 function evaluations and looks far better than 30 Euler ones
  at similar cost. This is what makes the free Spaces tier usable rather than
  "minutes per image".

Verified end to end on 2026-08-12: both models load on MPS, `/api/health`
reports them, generation works per-model, `model: "nope"` is rejected with a
clear 404, and the sde+score+heun path runs.

### 9.1 Deployment weights — strip before shipping

A training checkpoint holds four copies of the network (live, EMA, and two Adam
moments). Inference needs one. `scripts/export_inference_ckpt.py` folds the EMA
weights into `model`, drops `optimizer`/`scaler`/`ema`, and casts to fp16:

| | full | slim |
|---|---|---|
| U-Net 92.5M | 1.48 GB | **185 MB** |
| DiT-B/4 130.4M | 2.09 GB | **261 MB** |

8× in both cases. **Verified, not assumed:** rebuilding both models from the
slim files and diffing against the EMA-applied reference gives a worst-case
weight deviation of 1.8e-3 (U-Net) / 1.9e-3 (DiT) against values spanning ±4.8
— pure fp16 rounding. `_load_checkpoint` applies EMA only when an `ema` key is
present, which is why folding it in and omitting the key is safe.

### 9.2 Hosting

Everything is written and locally tested in `deploy/`; it needs an HF account
to actually push. `deploy/DEPLOY.md` is the runbook. Shape of it:

- Weights → a HF **model** repo (not the Space: Spaces are for code, and this
  way a weight swap does not force a rebuild).
- App → a **Docker** Space from `deploy/space/` (Gradio/Streamlit templates do
  not fit a FastAPI + static-HTML app). The Dockerfile pip-installs the package
  from GitHub at build time and `start.sh` pulls the weights at container
  start, so shipping a code change is a push plus a Space restart.
- Free CPU tier works given the adaptive defaults above. For **ZeroGPU**, set
  the hardware and delete the `--index-url .../whl/cpu` line so pip resolves
  the CUDA build; the app picks up CUDA on its own.

**Label the UI honestly.** It says 1000 ImageNet classes at 64×64. A visitor
who types a scene description and gets one object should read the demo as
scoped, not broken.

---

## 10. Traps and known issues

**10.1 — ✅ RESOLVED 2026-08-07.** The repo is under git and pushed to
`https://github.com/aaarvs07ranger/epsilon`. The `.gitignore` covers `.venv/`,
`/data/`, `runs/`, `logs/`, `*.zip`, `*.dmg`, `*.pt`, `*.npz`, `*.egg-info/`,
`__pycache__/`, `.pytest_cache/`, `eps/data/imagenet64/`. The 11.3 GB zip and
the 125 MB dmg never entered history.

⚠️ This entry used to recommend a bare **`data/`** in that list. That exact
instruction is what caused **§10.17** — it also matched the source package
`eps/data/` and kept it out of every commit for four days. It now reads
`/data/`, anchored. If you are copying this list somewhere, copy the slash.

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

**10.6 — `ImageNet64` holds the whole split in RAM** as uint8: **15.7 GB
measured** for the full 1.28M training set (11 shards). Fine on a Hyak node
with `--mem=180G`.

Updated 2026-08-11: the full split *does* load and train on the 36 GB M5 Max —
"will thrash a laptop" was too pessimistic — but it leaves the machine fully
committed (~21 GB wired, ~6 GB compressed, <1 GB free) and leaning on swap.
It works; it has no headroom. See the watch item in §7.0 for what to do if
throughput decays, and **§10.18 for why this forces `num_workers: 0`** — the
RAM-resident design and macOS spawn semantics interact badly.

To be safe on a smaller machine, cap with `data.max_samples`. The loader fills
one preallocated array shard-by-shard rather than concatenating, so peak is
`split + one shard` (~17.3 GB), not 2× the split.

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

**10.17 — ✅ FIXED 2026-08-11. `.gitignore` silently excluded the `eps/data/`
*source package* from every commit for four days.** The entry was a bare
`data/`. A gitignore pattern with no leading slash matches a directory of that
name **at any depth**, so alongside the intended dataset directory it also
swallowed `eps/data/` — `imagenet.py` and `imagenet_classes.txt`. They were
never committed: `git log --all --diff-filter=A -- 'eps/data/*'` was empty from
`git init` (2026-08-07) until 2026-08-11.

Consequence: the public repo `aaarvs07ranger/epsilon` was **not runnable from a
fresh clone**. `import eps.training` → `ModuleNotFoundError: No module named
'eps.data'`, which breaks training, `evaluate_fid.py`, and the web demo. It was
invisible on the original machine, where the files sit untracked on disk and
everything imports fine. It surfaced the first time the repo was cloned
somewhere else — which was also the machine that was supposed to do the
training.

The fix is `/data/`, anchored to the repo root. **Do not remove the leading
slash.** The comment in `.gitignore` says so; leave it there.

Two general lessons, both cheap to apply:
- After `git init` on an existing tree, run `git status --ignored --short` and
  read what got excluded. A source file and a 16 GB dataset look identical in
  a gitignore pattern.
- "The tests pass" did not catch this, because the tests ran on the machine
  that had the untracked files. **A fresh `git clone` into a temp dir plus
  `pytest` is the only check that would have.** Worth doing before any push
  that others (or a second machine) will clone.

**10.18 — 🚫 On macOS, `data.num_workers > 0` duplicates the entire in-RAM
dataset into every worker. Keep it at 0.** Measured 2026-08-11.

macOS defaults `multiprocessing` to **spawn**, not fork (verified:
`mp.get_start_method()` → `spawn` on Python 3.13). DataLoader workers therefore
do **not** inherit the parent's pages copy-on-write — the `Dataset` object is
pickled by value into each one. Verified directly: pickling an `ImageNet64`
slice produces a blob the same size as its uint8 array (61.5 MB vs 61.4 MB).

`ImageNet64` holds the whole split in RAM (§10.6), so the full 1.28M-image
training set is **15.7 GB per worker**. `num_workers: 2` therefore wants
15.7 x 3 = **47 GB against 36 GB of unified memory**, ~30 GB of which MPS wants
for the GPU. The run dies at DataLoader startup — *after* the 30-minute data
fetch, which is a slow way to find out.

Reducing the worker count does not fix it; even one worker duplicates the whole
split. `configs/train_m5.yaml` now sets `num_workers: 0`, and its comment
explains why so nobody "optimises" it back up. The cost is ~1%: the data is
already resident, and a step takes ~4.5 s.

Note the *old* comment in that config justified `num_workers: 2` by reasoning
about "forked worker copy-on-write growth" — correct reasoning for Linux,
wrong platform. This never bit on Hyak (Linux, fork, and `--mem=180G` anyway).
If Epsilon ever moves back to Linux, workers become cheap again.

**Confirmed on RunPod 2026-08-11:** `num_workers: 8` is fine on Linux. `ps`
shows 8 worker PIDs per arm, each reporting ~15.9 GB RSS — that is the *same*
copy-on-write pages counted repeatedly, not 8 real copies. Actual host memory
is ~32 GB for both arms combined, on a 141 GB box.

**10.19 — 🚫 The RunPod dashboard's utilisation/memory/disk graphs lie. Do not
diagnose from them.** 2026-08-11: with both GPUs pinned at 100% and 500W, the
Pods list showed **Utilization 1% / 0%, Memory 0% / 0%, Disk 0% / 0%**. It
looked exactly like both jobs had died, and cost a round of panic.

The tell is that **Disk 0% is impossible** — 32 GB of dataset was sitting on a
100 GB volume. When one number in a row is provably false, distrust the whole
row; the telemetry agent drops out while the pod runs perfectly well.

Diagnose **on the box**, never from the dashboard:

```bash
nvidia-smi                              # real utilisation, memory, power, temp
ps aux | grep "[t]rain.py"              # are the processes alive
tail -n 3 runs/unet.log runs/dit.log    # are the step counters advancing
```

The last one is the real answer: if `[step N]` is increasing, everything is
fine no matter what any UI says.

(Also normal and not a problem: 84–85 °C and ~500W/600W on these cards under
sustained load. The rates were not decaying, so nothing was throttling.)

**10.20 — `tail -2 file1 file2` fails over non-interactive SSH.** It returns
`tail: option used in invalid context -- 2`. The bare `-2` is the obsolete
option form, and GNU coreutils rejects it with multiple files in that context.
It works interactively, which makes it look random. **Always write `tail -n 2`**
in anything that might run over `ssh host "..."`.

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

- **Epsilon is CLASS-conditional, not text-to-image. Settled 2026-08-11; do
  not reopen without a deliberate scope change.** Asked directly whether to
  scale up "to get the best possible text-to-image generator", the answer is
  that no amount of training produces one. Verified by grep: there is **no
  text encoder anywhere in the codebase** — no `open_clip`, no `CLIPText`, no
  tokeniser. Conditioning is `LabelEmbedder` = `nn.Embedding(1000 + 1, hidden)`,
  a lookup over 1000 ImageNet class indices plus the null token. The web demo's
  prompt box **fuzzy-matches free text onto one of those 1000 class names**
  (`_match_class` in `eps/web/app.py`), so "a cat riding a skateboard" resolves
  to `tabby cat`. That is a text box over a class picker.

  Real text-to-image would need all of: a *captioned* dataset (ImageNet has no
  captions — COCO / CC3M / CC12M), a frozen text encoder, cross-attention
  actually wired and tested (`DiTConfig.cross_attention` / `context_dim` are
  scaffolding that nothing feeds), and realistically latent diffusion at 256px,
  because 64x64 text-to-image looks bad at any training budget. That is a
  different project — weeks of work and four-to-five figures of compute
  (SD-1.4 was ~150k A100-hours). The `text` and `latent` extras in
  `pyproject.toml` are declared but **unused**; their presence is not evidence
  that the feature is close.

  The capstone's contribution is §2 — every equation implemented from the
  6.S184 notes and checked against an independent construction. A well-trained,
  honestly-reported class-conditional model demonstrates that fully; a
  half-trained text-to-image model demonstrates it worse and costs ~50x more.
  **Aarav chose to keep it class-conditional (2026-08-11).** Label the demo UI
  as "1000 ImageNet classes" so it reads as an honest demo rather than a broken
  text-to-image model.

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

**2026-08-11** — **First session on the M5 Max. Two blockers found and fixed,
and the gating measurement finally taken.**

*Blocker 1 — the repo did not run from a fresh clone (§10.17).* `.gitignore`
had a bare `data/`, which matches at any depth and so excluded the **source
package** `eps/data/` from every commit since `git init`. `imagenet.py` and
`imagenet_classes.txt` had never been pushed; `import eps.training` failed with
`ModuleNotFoundError: No module named 'eps.data'`. Invisible on the M4, where
the files sit untracked on disk. Fixed the pattern to `/data/` and rebuilt the
package from the interfaces its callers require:
- Class names taken from the **same HF repo that produces the labels**
  (`benjamin-paine/imagenet-1k-64x64`, `dataset_info.features`), so index
  alignment is guaranteed rather than assumed. Its map has **1001** keys —
  index 1000 is `'none'`, the null token — and was excluded to keep the file at
  exactly 1000 lines. Verified 0=`tench`, 207=`golden retriever`, 979=`valley`,
  `wc -l` = 999 (no trailing newline), matching §6.2 exactly.
- `ImageNet64` preallocates and fills shard-by-shard rather than concatenating:
  concatenating the full 15.7 GB split holds two copies at once (~31 GB against
  36 GB shared with the GPU).
- Labels that do not land in [0, 999] after the 1-based shift now raise instead
  of silently becoming -1, which `UNLABELED` would have routed to the
  unconditional branch — a silent corruption of class-conditional training.
- Verified beyond "it imports": 66/66 tests, exact uint8 round-trip through
  model space, flip, `image_size=32` rescale, missing-`labels` → UNLABELED,
  label order preserved across a shard boundary, and real decoded images
  rendered and eyeballed against their class names.

*Blocker 2 — the venv.* The clone's `.venv` was built from the stock Xcode
`python3` (3.9.6), below the project's 3.10 floor, and contained nothing.
Rebuilt on 3.13.13 with `uv`. Two new gotchas recorded in §3: `uv venv` omits
`pip` entirely (breaking the `./.venv/bin/pip` workflow §3 documents, so `pip`
was installed into the venv explicitly), and `torch-fidelity` + `clean-fid` now
install cleanly on Apple silicon — so FID can be computed on this machine,
closing a gap the M4 notes list as open.

*The measurement.* 92.5M U-Net, 64x64, bf16, **no** gradient checkpointing,
global batch 256 (64 x 4 accum), 200 steps on `data/imagenet64_small`:

**0.22 it/s = 56 images/s**, stable across all eight 25-step windows, loss
1.236 -> 1.154. That is **6.8x the M4's 8.3 img/s** — inside the 5-12x band
§7.0 predicted, so the extrapolation held for once.

*Blocker 3, found while sanity-checking that measurement — and it would have
killed the real run at startup (§10.18).* macOS defaults `multiprocessing` to
**spawn**, not fork, so DataLoader workers get the Dataset pickled **by value**
rather than sharing pages copy-on-write. `ImageNet64` is RAM-resident, so the
full split is **15.7 GB per worker**: `num_workers: 2` wants 47 GB against
36 GB. The benchmark missed it only because `max_samples=4096` made the dataset
0.1 GB — it would have surfaced *after* the 25-minute data fetch. Note the
config's old comment justified `num_workers: 2` by reasoning about "forked
worker copy-on-write" — right reasoning, wrong platform, and it had never run
on this platform at full scale. Set to 0, which costs a measured **9%**
(0.22 -> 0.20 it/s) and is the only setting that fits.

**Planning number is therefore 0.20 it/s = 51 img/s**: 50k ~ 2.9 days,
100k ~ 5.8 days, 150k ~ 8.7 days — the pessimistic end of §7.0's old estimate,
and now measured rather than guessed. Larger micro-batches at the same global
batch (128 x 2, 256 x 1) were tried and are **slower**; 64 x 4 stands.

*Hardening for an unattended multi-day run.* `_fid()` was already wrapped in
try/except with the comment "FID must never kill a long training run", but
`_preview()` — which fires every `sample_every` steps and touches the ODE
integrator, CFG, the EMA swap and a PNG write, none of which the training step
exercises — was **not**. One transient failure on day 4 would have ended the
run. Guarded it the same way, for the same stated reason.

*Data.* Fetched the full training split on this machine: 1,281,167 labeled
images -> 11 npz shards, 15.7 GB, ~25 min, peak 23 GB on disk (1.7 TB free, so
the §10.15 disk problem is simply gone). `classes present: 1000/1000`, scratch
memmap cleaned up. **Re-verified the §6.3 ordering trap on the real data**: a
50,000-row prefix covers **1000/1000 classes** with near-uniform counts (min
18, median 50, max 72), so the fetcher's global shuffle works and
`data.max_samples` / a FID prefix export are safe.

*Launched.* `configs/train_m5.yaml`, 100k steps (~5.8 days), global batch 256,
bf16, `eval.sample_every=1000` (~83 min between preview grids) and
`logging.ckpt_every=2500` (~3.5 h, bounding crash loss; 40 checkpoints x
1.48 GB ~ 59 GB, fine at 1.7 TB free). Detached under `caffeinate -is nohup`,
logging to `runs/train.log`, output in `runs/m5`.

Because `lr_schedule: constant`, `total_steps` sets **only** the stopping point
— no schedule is baked in — so the run can be stopped at any checkpoint and
extended later with
`--resume runs/m5/ckpt_latest.pt training.total_steps=<bigger>`.

**2026-08-11 (later)** — **Venue changed again: renting RunPod GPUs, and the
goal became a U-Net vs DiT comparison.** Aarav's call, after seeing the local
run at 4% after 5.3 hours: paying ~$40 to get both models in ~6 hours beats
5.5 days of laptop time per model.

The local run was stopped at **step 5,100 / 100,000** — 5.3 h, loss
1.2947 → 0.1668, zero errors, `_preview` fired five times cleanly. Artifacts
kept in `runs/m5` (4.2 GB). Not wasted: it is the only end-to-end evidence the
training loop is correct, and its 0.21 it/s is the baseline the cloud speedup
is measured against. Sample grids at 5.1k steps already show correct
class-conditional colour and texture (green parrot for macaw 88, hazy landscape
for valley 979, white fluff for arctic fox 279) — shapes not yet formed, which
is right for 5% of training.

Added `configs/train_cloud_{unet,dit}.yaml` (verified to differ **only** in the
model block), `scripts/setup_runpod.sh`, and `scripts/eval_compare.sh`. Full
reasoning in §7.3; the load-bearing decision is **two single-GPU processes
instead of DDP**, because the DDP path has never been run and a rented 8-GPU
box is a bad place to discover that.

Both bash scripts syntax-check, and the exact `sample.py` invocation
`eval_compare.sh` uses was executed against the real `runs/m5` checkpoint
before shipping — so the CLI forms are proven, not assumed.

**2026-08-11 (evening)** — **Cloud run launched and in flight. Full operational
detail in §7.4; the durable lessons are here.**

*The repo is finally runnable from a clone.* Aarav pushed `eps/data/` and the
`.gitignore` fix. `git ls-tree origin/main eps/data/` now returns all three
files. Two commits remain local (`3187354`, `77f4f68`).

*Infrastructure.* Generated an SSH keypair at `~/.ssh/id_ed25519` — the machine
had **no** `~/.ssh` at all, which is also why the earlier HTTPS `git push`
failed with "could not read Username". The same key serves RunPod and GitHub.
Deployed 2× RTX PRO 6000 Blackwell on Secure Cloud at $4.20/hr. Setup ran clean
first time: 66 tests, full 1.28M fetch, 1000/1000 classes, shuffle verified.

*The headline measurement, and a real finding.* **DiT-B/4 (130.4M) trains at
3.20 it/s while the smaller U-Net (92.5M) manages 1.64.** The bigger model is
2× faster. Cause is arithmetic intensity, not a bug — DiT is dense matmuls that
saturate tensor cores; the U-Net's many small convolutions at 64×64 cannot feed
a fast GPU. **This inverts the naive parameter-count intuition and belongs in
the writeup**, alongside the FID numbers, as a measured architectural result.

*A hypothesis I got wrong, tested, and rejected.* I predicted the U-Net was
kernel-launch-bound and would speed up with a larger micro-batch. It did not:
128×2-accum gave 1.63 it/s, 256×1 gave 1.64. `nvidia-smi` showed 100% util and
504W at both. It is compute-bound. Recorded in §7.4 so nobody retries it. Note
this is the **opposite** of the M5 Max result, where bigger batches were slower
for memory reasons — neither finding generalises across machines, which is
itself the point.

*Scope cut, forced by the measurement.* 100k steps for the U-Net = 17 h = ~$71,
over both the 12-hour window and the $60 budget. Cut the U-Net to **60,000
steps** (~10.1 h, 15.4M images, ~12 epochs); left DiT at 100k since it is fast
anyway. **The matched comparison survives at zero cost** because checkpoints
save every 10k: evaluate U-Net@60k against DiT's `ckpt_0060000.pt`, and keep
DiT@100k as a bonus. Relaunched the U-Net with `total_steps=60000` so it halts
on its own rather than burning credits overnight.

*Two new traps, both of which cost real time tonight.* §10.19: the RunPod
dashboard reported 0% GPU/memory/disk while both cards were pinned at 100% and
500W — it looked exactly like a crash. The giveaway was Disk 0% on a volume
holding 32 GB, i.e. provably false. **Diagnose on the box with `nvidia-smi` and
the step counter, never from the UI.** §10.20: `tail -2 a b` fails over
non-interactive SSH with "option used in invalid context"; use `tail -n 2`.

*Estimate corrected.* The eval sweep is **~2.5 h**, not the ~45 min this file
and `eval_compare.sh` originally claimed — 6 runs × 10k samples × 100 ODE steps,
with CFG doubling the NFE. Fixed in §8 and in the script header.

*Still true and still the biggest risk:* the pod's volume is deleted on
terminate. Everything must be `scp`'d off first. That is the only step in this
whole plan where work can be irrecoverably lost.

**2026-08-12** — **Both models trained. Pod terminated. Project has results.**

Overnight both arms finished cleanly: U-Net `ckpt_final.pt` at 60k, DiT at
100k, no errors, `_preview` fired throughout. Generated the 9 sample grids on
the pod, pulled 5.3 GB down, verified, terminated. **Total spend ~$49 of $60.**

*The samples are better than predicted.* I told Aarav to expect "rough shapes
and colours"; the actual output has recognisable golden retrievers, vivid
macaws, red pandas, hot-air balloons and arctic foxes. Underpromising was the
right error to make, but worth recording that 60k steps at 92.5M on 64×64 gets
further than I expected.

*The matched comparison, qualitatively:* at 60k steps each, **the U-Net is
visibly better** — sharper macaws, cleaner red pandas, more coherent balloons;
the DiT's dogs are comparable but its other classes are blobbier. Exactly what
§7.3 predicted. Combined with §7.4's throughput inversion, the project now has
two honest measured findings that point in opposite directions, which is a far
more interesting result than either alone.

*Skipped FID deliberately.* With $11 left and the sweep costing ~2.5 h ≈ $10.5,
running it risked the pod terminating mid-eval and taking the models. Chose
models over numbers. FID runs free locally whenever wanted.

*Delivery work, all verified rather than assumed:*
- Moved everything into the repo: `runs/cloud_*/` (gitignored, 9.4 GB) and
  `results/` (tracked, 1.3 MB of grids, used by the README).
- Wrote `scripts/export_inference_ckpt.py` → §9.1. 8× smaller, diffed against
  the EMA reference to prove it.
- Upgraded the web app to load both backbones with a **Compare** mode at
  matched seed, plus CPU-adaptive Heun defaults → §9.0. Tested live.
- Wrote `deploy/space/` (Dockerfile, start.sh, Space README) and
  `deploy/DEPLOY.md` → §9.2. Cannot be pushed without Aarav's HF account.
- Rewrote `README.md` around the actual results, with an honest-limitations
  section. **Fixed its opening line, which claimed "text-to-image"** — the same
  error §11 exists to prevent, sitting in the most-read file in the repo.

---

## 13. The capstone presentation (requested 2026-08-12, not started)

Aarav wants a **Google Slides deck, ~25 minutes, with live demos woven in.**
Not started; this section is the brief so a later session can pick it up cold.

**Assets that already exist and should be used:**

| Asset | Where | Good for |
|---|---|---|
| 9 sample grids (guidance sweep, SDE, score) | `results/` | the payoff slides |
| 64 training previews, every 2500 steps | `runs/cloud_*/previews/` | an animated "learning over time" build |
| Loss + throughput logs | `runs/{unet,dit}.log` | training-curve plot |
| Equation → code table | `README.md` | the verification-discipline slide |
| Explainer artifact (⚠️ stale) | §2 | zoom-ladder framing, if revised first |

**The three things that make this stand out**, and which the deck should be
built around rather than burying:
1. **Verification discipline** — every closed-form expression checked against
   an independent construction *before* anything was trained. This is the
   actual contribution; the pictures follow from it.
2. **The throughput inversion** — the 130.4M DiT trains 2× faster than the
   92.5M U-Net, with the arithmetic-intensity explanation and the rejected
   launch-bound hypothesis (§7.4). Real measured engineering.
3. **Honest scoping** — FID<12 shown to be unfunded by arithmetic, target
   restated, and the "text-to-image" framing rejected on the evidence (§11).

**Live demo beats**, in rising order of impact — rehearse offline as a fallback,
since the free tier will be slow and a queue mid-talk is a bad look:
- generate one class, low steps, to show it works;
- slide guidance 1 → 8 and watch typicality trade against diversity;
- flip **velocity → score** on the *same* seed: near-identical output from the
  same weights, which is Proposition 1 happening live;
- flip **ODE → SDE** and vary σ, with σ = 0 recovering the ODE;
- **Compare** mode, same seed, U-Net vs DiT side by side — close on this.

**Timing sketch (25 min):** 3 problem/framing · 5 maths and the verification
table · 3 architectures · 4 the training story including the scope cuts · 6
live demo · 2 results and honest limitations · 2 what I'd do next · Q&A.

Suggested slide count ~22. Do not fill slides with equations the audience
cannot read — put one per slide and say it out loud.
