"""DDIM inversion — a prompt-assisted REFERENCE, not a baseline (PLAN.md §5.3).

Rule D says the extractor sees only Y. DDIM inversion needs the conditioning
prompt c, so under the MVP protocol (`H(Y)`, prompt NOT given to the receiver)
this is not a fair comparison. It is reported as a dashed line labelled

    prompt-assisted reference — violates the receiver protocol

Its only job is diagnostic: if prompt-assisted inversion recovers z but the CNN
extractor cannot, the extractor is underpowered rather than the channel empty.
"""
from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def encode_images(gen, images: np.ndarray) -> torch.Tensor:
    """uint8 [B,H,W,3] -> VAE latent. The 0.18215 here belongs to the VAE and has
    nothing to do with z_T (PLAN.md §1.3)."""
    x = torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2).float() / 255.0
    x = (x * 2 - 1).to(gen.device, gen.dtype)
    posterior = gen.pipe.vae.encode(x).latent_dist
    return posterior.mean * gen.pipe.vae.config.scaling_factor


@torch.no_grad()
def _eps(gen, latent: torch.Tensor, t, embeds: torch.Tensor, guidance: float) -> torch.Tensor:
    if guidance == 1.0:
        return gen.pipe.unet(latent, t, encoder_hidden_states=embeds).sample
    model_in = torch.cat([latent] * 2)
    out = gen.pipe.unet(model_in, t, encoder_hidden_states=embeds).sample
    uncond, cond = out.chunk(2)
    return uncond + guidance * (cond - uncond)


@torch.no_grad()
def ddim_invert(gen, images: np.ndarray, prompts, num_steps: int | None = None,
                guidance_scale: float | None = None) -> torch.Tensor:
    """Y -> z_hat by running the deterministic DDIM ODE backwards.

    Exact only in the continuous limit: discretisation error grows with the
    guidance scale, which is itself part of what this diagnostic measures.
    """
    pipe = gen.pipe
    steps = int(num_steps or gen.spec.num_inference_steps)
    guidance = float(gen.spec.guidance_scale if guidance_scale is None else guidance_scale)
    B = len(prompts)

    pipe.scheduler.set_timesteps(steps, device=gen.device)
    timesteps = pipe.scheduler.timesteps                       # descending
    alphas = pipe.scheduler.alphas_cumprod.to(gen.device)
    final_alpha = pipe.scheduler.final_alpha_cumprod.to(gen.device)

    cond, uncond = pipe.encode_prompt(list(prompts), gen.device, 1, guidance > 1.0,
                                      [gen.spec.negative_prompt] * B)
    embeds = torch.cat([uncond, cond]) if guidance > 1.0 else cond
    latent = encode_images(gen, images)
    stride = 1000 // steps
    for i in range(steps):
        t = timesteps[len(timesteps) - i - 1]
        eps = _eps(gen, latent, t, embeds, guidance)
        t_prev = int(t) - stride
        a_prev = alphas[t_prev] if t_prev >= 0 else final_alpha
        a_next = alphas[int(t)]
        x0 = (latent - (1 - a_prev).sqrt() * eps) / a_prev.sqrt()
        latent = a_next.sqrt() * x0 + (1 - a_next).sqrt() * eps
    return latent.float()
