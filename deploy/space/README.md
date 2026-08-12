---
title: Epsilon — Flow Matching & Diffusion from Scratch
emoji: 🌀
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Epsilon

A class-conditional image generator for **ImageNet-1K at 64×64**, written from
scratch on the mathematics of MIT 6.S184 — every closed-form expression checked
against an independent construction before it was trained on anything.

Two backbones are loaded side by side:

| | params | trained |
|---|---|---|
| U-Net (ADM-style, AdaGN) | 92.5M | 60k steps |
| DiT-B/4 (AdaLN-Zero) | 130.4M | 100k steps |

**Compare** generates from both with the *same noise*, so the difference you
see is the architecture and not the draw.

### What the controls do

- **Flow Matching (ODE)** vs **Diffusion (SDE)** — deterministic transport, or
  the same marginals with Langevin noise added. σ = 0 recovers the ODE exactly.
- **Velocity** vs **Score** — for Gaussian paths these are linear
  reparameterisations of each other, so *one* trained network drives both.
  Flipping this changes the sampler's algebra, not the weights.
- **Guidance w** — classifier-free guidance. Higher is more class-typical and
  less diverse; the FID optimum is usually near 1.5 even though 4+ looks nicer.

### Scope, stated plainly

This is **class-conditional, not text-to-image.** The prompt box matches your
text to the nearest of 1000 ImageNet class names — "a cat riding a skateboard"
resolves to `tabby cat` and you get a cat. There is no text encoder.

Source: <https://github.com/aaarvs07ranger/epsilon>
