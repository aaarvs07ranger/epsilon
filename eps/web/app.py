"""Epsilon web demo: a FastAPI app serving the generation API and the static
single-page frontend (eps/web/static/index.html).

Run locally:
    EPSILON_CKPT=runs/unet64/ckpt_latest.pt \
        uvicorn eps.web.app:app --host 0.0.0.0 --port 7860

Without a checkpoint the UI still loads (with a banner); /api/generate
returns 503. Deployment (HF Spaces / Railway / VPS) needs nothing beyond
`pip install -e .[web]` and the environment variable above.
"""

from __future__ import annotations

import base64
import difflib
import io
import os
import threading
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..config import from_dict
from ..data import imagenet_class_names
from ..evaluation.fid import sample_batch
from ..models import build_model
from ..paths import GaussianProbabilityPath, build_scheduler
from ..training.ema import EMA
from ..utils import tensor_to_pil

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Epsilon", docs_url="/api/docs")

# One entry per loaded backbone, keyed by a short slug ("unet", "dit"). Holding
# both at once is what lets the demo A/B the two architectures on an identical
# prompt and seed — the project's headline comparison, made interactive.
_models: dict[str, dict] = {}
_lock = threading.Lock()  # one generation at a time; MPS/CUDA context safety
_class_names = imagenet_class_names()


def _pick_device() -> torch.device:
    return torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )


def _load_one(ckpt_path: str, device: torch.device) -> Optional[dict]:
    """Load a single checkpoint into a model entry, or None if unusable."""
    if not ckpt_path or not Path(ckpt_path).exists():
        print(f"[epsilon-web] skipping '{ckpt_path or '(unset)'}' — not found")
        return None
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = from_dict(ckpt["config"])
    size = cfg.data.image_size // 8 if cfg.training.latent_space else cfg.data.image_size
    net = build_model(cfg.model, size).to(device)
    net.load_state_dict(ckpt["model"])
    # Slim inference exports (scripts/export_inference_ckpt.py) fold the EMA
    # weights into "model" and omit this key; training checkpoints carry it.
    if "ema" in ckpt:
        ema = EMA(net)
        ema.load_state_dict(ckpt["ema"])
        ema.copy_to(net)
    net.eval()
    step = ckpt.get("step")
    return {
        "net": net,
        "cfg": cfg,
        "path": GaussianProbabilityPath(build_scheduler(cfg.path.scheduler)),
        "device": device,
        "arch": cfg.model.name,
        "step": step,
        "params": sum(p.numel() for p in net.parameters()),
        "source": ckpt_path,
    }


def _load_models() -> None:
    """Load every checkpoint named by EPSILON_MODELS (or EPSILON_CKPT).

    EPSILON_MODELS is a comma-separated list of checkpoint paths; labels are
    derived from each checkpoint's own config, so nothing has to be kept in
    sync by hand:

        EPSILON_MODELS="deploy/unet_60k.pt,deploy/dit_100k.pt"

    EPSILON_CKPT (single path) still works and is appended to the list.
    """
    paths = [p.strip() for p in os.environ.get("EPSILON_MODELS", "").split(",") if p.strip()]
    single = os.environ.get("EPSILON_CKPT", "").strip()
    if single and single not in paths:
        paths.append(single)
    if not paths:
        print("[epsilon-web] no EPSILON_MODELS / EPSILON_CKPT set — demo mode (UI loads, /api/generate 503s)")
        return

    device = _pick_device()
    for p in paths:
        entry = _load_one(p, device)
        if entry is None:
            continue
        key = entry["arch"]
        if key in _models:  # two of the same architecture: disambiguate by step
            key = f"{key}-{entry['step']}"
        _models[key] = entry
        print(
            f"[epsilon-web] loaded '{key}': {entry['arch']} "
            f"{entry['params'] / 1e6:.1f}M params, step {entry['step']}, on {device}"
        )
    if not _models:
        print("[epsilon-web] every checkpoint failed to load — demo mode")


@app.on_event("startup")
def _startup() -> None:
    _load_models()


def resolve_class(prompt: str) -> tuple[int, str]:
    """Map a free-text prompt to the best-matching ImageNet class."""
    q = prompt.strip().lower()
    if not q:
        raise HTTPException(400, "Empty prompt")
    # 1) exact / substring matches win
    contains = [i for i, n in enumerate(_class_names) if q == n.lower()]
    if not contains:
        contains = [i for i, n in enumerate(_class_names) if q in n.lower()]
    if not contains:
        contains = [
            i for i, n in enumerate(_class_names)
            if any(w in n.lower() for w in q.split() if len(w) > 3)
        ]
    if contains:
        idx = min(contains, key=lambda i: len(_class_names[i]))
        return idx, _class_names[idx]
    # 2) fuzzy fallback
    close = difflib.get_close_matches(q, [n.lower() for n in _class_names], n=1, cutoff=0.5)
    if close:
        idx = [n.lower() for n in _class_names].index(close[0])
        return idx, _class_names[idx]
    raise HTTPException(
        404, f"No ImageNet class matches '{prompt}'. Try e.g. 'golden retriever' or 'volcano'."
    )


class GenerateRequest(BaseModel):
    prompt: Optional[str] = None
    class_id: Optional[int] = Field(default=None, ge=0, le=999)
    model: Optional[str] = None  # "unet" | "dit"; default = first loaded
    method: str = Field(default="ode", pattern="^(ode|sde)$")
    parameterization: str = Field(default="velocity", pattern="^(velocity|score)$")
    solver: str = Field(default="euler", pattern="^(euler|heun)$")
    guidance_scale: float = Field(default=4.0, ge=0.0, le=20.0)
    num_steps: int = Field(default=100, ge=1, le=1000)
    sigma: float = Field(default=1.0, ge=0.0, le=5.0)
    seed: Optional[int] = None


_ARCH_LABEL = {"unet": "U-Net", "dit": "DiT-B/4"}


def _describe(key: str, m: dict) -> dict:
    step = m["step"]
    return {
        "key": key,
        "label": _ARCH_LABEL.get(m["arch"], m["arch"]),
        "arch": m["arch"],
        "params_m": round(m["params"] / 1e6, 1),
        "step": step,
        "steps_label": f"{step // 1000}k steps" if isinstance(step, int) else "",
        "prediction": m["cfg"].model.prediction,
    }


@app.get("/api/health")
def health() -> dict:
    ready = bool(_models)
    first = next(iter(_models.values()), None)
    return {
        "ready": ready,
        "device": str(first["device"]) if first else None,
        "models": [_describe(k, m) for k, m in _models.items()],
        # kept for older clients
        "model": first["arch"] if first else None,
        "prediction": first["cfg"].model.prediction if first else None,
    }


@app.get("/api/classes")
def classes() -> list[dict]:
    return [{"id": i, "name": n} for i, n in enumerate(_class_names)]


@app.post("/api/generate")
def generate(req: GenerateRequest) -> dict:
    if not _models:
        raise HTTPException(503, "No checkpoint loaded (set EPSILON_MODELS and restart)")
    if req.class_id is not None:
        class_id, class_name = req.class_id, _class_names[req.class_id]
    elif req.prompt:
        class_id, class_name = resolve_class(req.prompt)
    else:
        raise HTTPException(400, "Provide 'prompt' or 'class_id'")

    key = req.model or next(iter(_models))
    if key not in _models:
        raise HTTPException(404, f"Unknown model '{key}'. Loaded: {', '.join(_models)}")
    entry = _models[key]
    cfg, net, path, device = entry["cfg"], entry["net"], entry["path"], entry["device"]
    seed = req.seed if req.seed is not None else int.from_bytes(os.urandom(4), "little")
    in_ch = cfg.model.unet.in_channels if cfg.model.name == "unet" else cfg.model.dit.in_channels
    size = cfg.data.image_size // 8 if cfg.training.latent_space else cfg.data.image_size

    with _lock, torch.no_grad():
        generator = torch.Generator(device=device).manual_seed(seed)
        y = torch.tensor([class_id], device=device)
        x = sample_batch(
            net, path, cfg, y, 1, (in_ch, size, size), device,
            generator=generator,
            guidance_scale=req.guidance_scale,
            num_steps=req.num_steps,
            method=req.method,
            parameterization=req.parameterization,
            # Passed per-request, never written onto the shared cfg: two
            # concurrent visitors must not see each other's solver/sigma.
            solver=req.solver,
            sigma=req.sigma,
        )

    buf = io.BytesIO()
    tensor_to_pil(x[0]).resize((256, 256), resample=0).save(buf, format="PNG")
    return {
        "image": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(),
        "class_id": class_id,
        "class_name": class_name,
        "model": key,
        "model_label": _ARCH_LABEL.get(entry["arch"], entry["arch"]),
        "seed": seed,
        "method": req.method,
        "parameterization": req.parameterization,
        "guidance_scale": req.guidance_scale,
        "num_steps": req.num_steps,
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
