"""The training loop: distributed, mixed-precision, EMA, checkpointing,
logging, previews, and optional in-training FID.

One Trainer serves every configuration: U-Net or DiT, velocity (CFM) or score
(DSM) prediction, pixel space or VAE latent space, single MacBook process or
multi-node torchrun on Hyak.
"""

from __future__ import annotations

import math
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Iterator, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from ..config import EpsilonConfig, save_config, to_dict
from ..data import build_dataset
from ..evaluation.fid import compute_metrics, generate_samples, sample_batch
from ..losses import build_loss, cfg_label_dropout
from ..models import build_model
from ..models.vae import PretrainedVAE
from ..paths import GaussianProbabilityPath, build_scheduler
from ..utils import save_image_grid
from .distributed import (
    barrier,
    get_world_size,
    is_distributed,
    is_main_process,
    seed_everything,
    setup_distributed,
    unwrap_ddp,
    wrap_ddp,
)
from .ema import EMA


class Trainer:
    def __init__(self, cfg: EpsilonConfig, resume: Optional[str] = None) -> None:
        self.cfg = cfg
        self.device = setup_distributed()
        seed_everything(cfg.training.seed)

        self.out_dir = Path(cfg.logging.output_dir)
        if is_main_process():
            self.out_dir.mkdir(parents=True, exist_ok=True)
            save_config(cfg, self.out_dir / "config.yaml")

        # Mathematical core -------------------------------------------------
        self.path = GaussianProbabilityPath(build_scheduler(cfg.path.scheduler))
        self.loss_fn = build_loss(
            cfg.model.prediction, self.path, t_min=cfg.training.t_min, t_max=cfg.training.t_max
        )

        # Latent space (Sec. 6.2) -------------------------------------------
        self.vae: Optional[PretrainedVAE] = None
        self.sample_size = cfg.data.image_size
        if cfg.training.latent_space:
            if cfg.vae.pretrained is None:
                raise ValueError(
                    "training.latent_space=true requires vae.pretrained "
                    "(a diffusers AutoencoderKL id, e.g. stabilityai/sd-vae-ft-ema)"
                )
            self.vae = PretrainedVAE(cfg.vae.pretrained).to(self.device)
            self.sample_size = cfg.data.image_size // 8  # AutoencoderKL stride

        # Model -------------------------------------------------------------
        model = build_model(cfg.model, self.sample_size).to(self.device)
        if cfg.training.compile:
            model = torch.compile(model)
        self.model = wrap_ddp(model, self.device)
        self.ema = EMA(unwrap_ddp(self.model), decay=cfg.training.ema_decay)

        n_params = sum(p.numel() for p in model.parameters())
        if is_main_process():
            print(f"[epsilon] {cfg.model.name} | {n_params / 1e6:.1f}M params | "
                  f"prediction={cfg.model.prediction} | device={self.device} | "
                  f"world_size={get_world_size()}")

        # Optimiser ---------------------------------------------------------
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.training.lr,
            betas=cfg.training.betas,
            weight_decay=cfg.training.weight_decay,
        )

        # Mixed precision ---------------------------------------------------
        # bf16 tensor cores need Ampere (sm_80+). On older cards
        # torch.cuda.is_bf16_supported() can still return True because recent
        # PyTorch counts *emulated* bf16, which is slower than plain fp32 — so
        # gate on the compute capability directly and fall back to fp16 rather
        # than silently dropping to fp32 for the whole run.
        mp = cfg.training.mixed_precision
        self.autocast_dtype: Optional[torch.dtype] = None
        if self.device.type == "cuda":
            major = torch.cuda.get_device_capability(self.device)[0]
            if mp == "bf16":
                if major >= 8:
                    self.autocast_dtype = torch.bfloat16
                else:
                    self.autocast_dtype = torch.float16
                    if is_main_process():
                        print(
                            f"[epsilon] WARNING: mixed_precision=bf16 but "
                            f"{torch.cuda.get_device_name(self.device)} is sm_{major}x "
                            "(bf16 needs sm_80+). Using fp16 instead."
                        )
            elif mp == "fp16":
                self.autocast_dtype = torch.float16
        elif self.device.type == "mps" and mp in ("bf16", "fp16"):
            # Apple silicon: autocast works, but GradScaler does not support MPS,
            # so bf16 is the only safe choice — its fp32 exponent range needs no
            # loss scaling. fp16 without a scaler underflows the small gradients
            # this loss produces at large t, so promote it rather than obey it.
            self.autocast_dtype = torch.bfloat16
            if mp == "fp16" and is_main_process():
                print(
                    "[epsilon] mixed_precision=fp16 on MPS needs a GradScaler, "
                    "which MPS lacks. Using bf16 instead (no scaler required)."
                )
        elif mp in ("bf16", "fp16") and is_main_process():
            print(f"[epsilon] mixed_precision={mp} ignored on {self.device.type}; using fp32.")
        # GradScaler is CUDA-only in practice; bf16 and fp32 never need it.
        self.scaler = torch.amp.GradScaler(
            enabled=self.device.type == "cuda" and self.autocast_dtype == torch.float16
        )

        # Data --------------------------------------------------------------
        self.dataset: Dataset = build_dataset(cfg.data)
        sampler = (
            DistributedSampler(self.dataset, shuffle=True, seed=cfg.training.seed)
            if is_distributed()
            else None
        )
        self.loader = DataLoader(
            self.dataset,
            batch_size=cfg.training.batch_size,
            sampler=sampler,
            shuffle=sampler is None,
            num_workers=cfg.data.num_workers,
            pin_memory=self.device.type == "cuda",
            drop_last=True,
            persistent_workers=cfg.data.num_workers > 0,
        )
        self.sampler = sampler

        # Logging -----------------------------------------------------------
        self.wandb = None
        if cfg.logging.wandb and is_main_process():
            import wandb

            self.wandb = wandb
            wandb.init(
                project=cfg.logging.wandb_project,
                name=cfg.logging.wandb_run_name,
                config=to_dict(cfg),
                dir=str(self.out_dir),
            )

        self.step = 0
        if resume is not None:
            self.load_checkpoint(resume)

    # -- data ---------------------------------------------------------------

    def _infinite_batches(self) -> Iterator[tuple[Tensor, Tensor]]:
        epoch = 0
        while True:
            if self.sampler is not None:
                self.sampler.set_epoch(epoch)
            for batch in self.loader:
                yield batch
            epoch += 1

    # -- optimisation ---------------------------------------------------------

    def _lr_at(self, step: int) -> float:
        t = self.cfg.training
        lr = t.lr
        if t.warmup_steps > 0:
            lr *= min(1.0, (step + 1) / t.warmup_steps)
        if t.lr_schedule == "cosine":
            progress = min(1.0, step / max(1, t.total_steps))
            lr *= 0.5 * (1.0 + math.cos(math.pi * progress))
        return lr

    def _autocast(self):
        if self.autocast_dtype is None:
            return nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype)

    def train_step(self, batches: Iterator[tuple[Tensor, Tensor]]) -> float:
        t = self.cfg.training
        lr = self._lr_at(self.step)
        for group in self.optimizer.param_groups:
            group["lr"] = lr

        self.optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for micro in range(t.grad_accum_steps):
            x, y = next(batches)
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            if self.vae is not None:
                with torch.no_grad():
                    x = self.vae.sample_latent(x)
            # Unlabeled samples (data.UNLABELED = -1) train the unconditional
            # branch: route them to the null token before dropout.
            null_index = self.cfg.model.num_classes
            y = torch.where(y < 0, torch.full_like(y, null_index), y)
            # CFG training (Algorithm 5): drop the label with probability eta.
            y = cfg_label_dropout(y, null_index=null_index, p=t.cfg_dropout)

            sync_ctx = (
                self.model.no_sync()
                if is_distributed() and micro < t.grad_accum_steps - 1
                else nullcontext()
            )
            with sync_ctx, self._autocast():
                loss = self.loss_fn(self.model, x, y) / t.grad_accum_steps
            self.scaler.scale(loss).backward()
            total_loss += loss.item()

        if t.grad_clip > 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), t.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.ema.update(unwrap_ddp(self.model))
        self.step += 1
        return total_loss

    def train(self) -> None:
        cfg = self.cfg
        batches = self._infinite_batches()
        self.model.train()
        t_last, steps_last = time.time(), self.step

        while self.step < cfg.training.total_steps:
            loss = self.train_step(batches)

            if self.step % cfg.logging.log_every == 0 and is_main_process():
                now = time.time()
                rate = (self.step - steps_last) / max(now - t_last, 1e-9)
                t_last, steps_last = now, self.step
                lr = self.optimizer.param_groups[0]["lr"]
                print(f"[step {self.step:>7d}] loss={loss:.4f} lr={lr:.2e} {rate:.2f} it/s")
                if self.wandb:
                    self.wandb.log(
                        {"train/loss": loss, "train/lr": lr, "train/steps_per_s": rate},
                        step=self.step,
                    )

            if (
                cfg.eval.sample_every > 0
                and self.step % cfg.eval.sample_every == 0
                and is_main_process()
            ):
                self._preview()

            if cfg.eval.fid_every > 0 and self.step % cfg.eval.fid_every == 0:
                self._fid()

            if self.step % cfg.logging.ckpt_every == 0:
                self.save_checkpoint(self.out_dir / f"ckpt_{self.step:07d}.pt")

        self.save_checkpoint(self.out_dir / "ckpt_final.pt")
        if self.wandb:
            self.wandb.finish()

    # -- evaluation -----------------------------------------------------------

    def _with_ema(self):
        """Context helper: swap EMA weights in, restore on exit."""
        trainer = self

        class _Ctx:
            def __enter__(self) -> nn.Module:
                net = unwrap_ddp(trainer.model)
                trainer.ema.store(net)
                trainer.ema.copy_to(net)
                net.eval()
                return net

            def __exit__(self, *exc) -> None:
                net = unwrap_ddp(trainer.model)
                trainer.ema.restore(net)
                net.train()

        return _Ctx()

    @torch.no_grad()
    def _preview(self) -> None:
        cfg = self.cfg
        classes = torch.tensor(list(cfg.eval.sample_classes), device=self.device)
        shape = (
            self.cfg.model.unet.in_channels
            if cfg.model.name == "unet"
            else self.cfg.model.dit.in_channels,
            self.sample_size,
            self.sample_size,
        )
        try:
            with self._with_ema() as net:
                x = sample_batch(
                    net,
                    self.path,
                    cfg,
                    classes,
                    classes.shape[0],
                    shape,
                    self.device,
                    guidance_scale=cfg.eval.sample_guidance_scale,
                    num_steps=cfg.eval.sample_steps,
                    method="ode",
                )
                if self.vae is not None:
                    x = self.vae.decode(x)
            out = self.out_dir / "previews" / f"step_{self.step:07d}.png"
            save_image_grid(x, out, num_columns=4)
            if self.wandb:
                import wandb

                self.wandb.log({"eval/preview": wandb.Image(str(out))}, step=self.step)
        except Exception as e:
            # Same rule as _fid: a preview grid is a convenience, and must never
            # take down a run that has been going for days. Sampling touches
            # code the training step does not (the ODE integrator, CFG, EMA
            # swap, PNG write), so it has its own ways to fail.
            print(f"[epsilon] preview sampling failed at step {self.step}: {e}")

    def _fid(self) -> None:
        cfg = self.cfg
        if cfg.eval.fid_reference_dir is None:
            if is_main_process():
                print("[epsilon] eval.fid_reference_dir not set; skipping FID")
            return
        barrier()
        if not is_main_process():
            barrier()
            return
        try:
            gen_dir = self.out_dir / "fid_samples" / f"step_{self.step:07d}"
            with self._with_ema() as net:
                generate_samples(
                    net, self.path, cfg, gen_dir,
                    num_samples=cfg.eval.fid_num_samples,
                    batch_size=cfg.training.batch_size,
                    device=self.device,
                )
            metrics = compute_metrics(gen_dir, cfg.eval.fid_reference_dir)
            print(f"[step {self.step}] " + " ".join(f"{k}={v:.3f}" for k, v in metrics.items()))
            if self.wandb:
                self.wandb.log({f"eval/{k}": v for k, v in metrics.items()}, step=self.step)
        except Exception as e:  # FID must never kill a long training run
            print(f"[epsilon] FID evaluation failed: {e}")
        barrier()

    # -- checkpointing --------------------------------------------------------

    def save_checkpoint(self, path: Path) -> None:
        if not is_main_process():
            barrier()
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "step": self.step,
                "model": unwrap_ddp(self.model).state_dict(),
                "ema": self.ema.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scaler": self.scaler.state_dict(),
                "config": to_dict(self.cfg),
            },
            path,
        )
        import shutil

        shutil.copyfile(path, path.parent / "ckpt_latest.pt")
        print(f"[epsilon] saved checkpoint {path}")
        barrier()

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        unwrap_ddp(self.model).load_state_dict(ckpt["model"])
        self.ema.load_state_dict(ckpt["ema"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scaler.load_state_dict(ckpt["scaler"])
        self.step = ckpt["step"]
        if is_main_process():
            print(f"[epsilon] resumed from {path} at step {self.step}")
