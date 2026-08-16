"""The frozen generative channel  X = G(Z).

Rule A: nothing in here is ever fine-tuned. The pipeline is loaded once, put in
eval mode with requires_grad_(False) on every module, and driven only through
`latents=`.

Three things this wrapper exists to guarantee (PLAN.md §1.2, Gate 0):

  1. `scheduler.init_noise_sigma == 1.0`, so `pipe(latents=z)` passes z through
     UNSCALED. Euler/LMS return sqrt(1+sigma_max^2) here and would silently
     rescale z into a different distribution.
  2. The scheduler is DDIM with eta = 0. The InvCISD tree forces
     `sde-dpmsolver++` for its reference model; a stochastic scheduler would put
     noise between Z and X that FIBER never sees and cannot invert.
  3. z is stored bit-exactly as the generator consumed it. The UNet runs fp16, so
     the *consumed* z is the fp16 rounding of the drawn z; the extractor's target
     must be that same value, not the float32 pre-image.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..utils.logging import get_logger
from ..utils.seeding import derive_seed

log = get_logger("fiber.generator")


@dataclass
class GeneratorSpec:
    checkpoint: str
    scheduler: str = "DDIMScheduler"
    eta: float = 0.0
    num_inference_steps: int = 25
    guidance_scale: float = 3.0
    negative_prompt: str = ""
    resolution: int = 512
    dtype: str = "float16"
    device: str = "cuda:0"

    @classmethod
    def from_config(cls, cfg) -> "GeneratorSpec":
        m = cfg["model"]
        return cls(**{k: m[k] for k in cls.__dataclass_fields__ if k in m})


class FrozenGenerator:
    def __init__(self, spec: GeneratorSpec, latent_cfg: dict | None = None):
        from diffusers import DDIMScheduler, StableDiffusionPipeline

        if spec.scheduler != "DDIMScheduler":
            raise ValueError(f"FIBER pins DDIM; got {spec.scheduler!r}")
        self.spec = spec
        self.dtype = getattr(torch, spec.dtype)
        self.device = torch.device(spec.device)
        lc = latent_cfg or {}
        self.latent_shape = (int(lc.get("channels", 4)),
                             int(lc.get("height", spec.resolution // 8)),
                             int(lc.get("width", spec.resolution // 8)))
        self.d = int(np.prod(self.latent_shape))
        if lc.get("dim") and int(lc["dim"]) != self.d:
            raise ValueError(f"config latent.dim={lc['dim']} != {self.d}")

        pipe = StableDiffusionPipeline.from_pretrained(
            spec.checkpoint, torch_dtype=self.dtype, safety_checker=None,
            requires_safety_checker=False,
        )
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(self.device)
        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe
        self._freeze()
        self.assert_channel_is_native()

    # -- Rule A ----------------------------------------------------------
    def _freeze(self) -> None:
        for name in ("unet", "vae", "text_encoder"):
            module = getattr(self.pipe, name, None)
            if module is not None:
                module.requires_grad_(False)
                module.eval()

    def assert_channel_is_native(self) -> None:
        sched = self.pipe.scheduler
        cls = type(sched).__name__
        if cls != "DDIMScheduler":
            raise AssertionError(f"scheduler is {cls}, expected DDIMScheduler")
        algo = str(getattr(sched.config, "algorithm_type", "")).lower()
        if algo.startswith("sde") or "sde" in cls.lower():
            raise AssertionError(f"stochastic scheduler detected: {cls}/{algo}")
        sigma = float(sched.init_noise_sigma)
        if abs(sigma - 1.0) > 1e-6:
            raise AssertionError(
                f"init_noise_sigma = {sigma} != 1.0: z would be silently rescaled")
        if abs(float(self.spec.eta)) > 0:
            raise AssertionError(f"eta = {self.spec.eta} != 0: generation is stochastic")
        for name in ("unet", "vae", "text_encoder"):
            module = getattr(self.pipe, name, None)
            if module is not None and any(p.requires_grad for p in module.parameters()):
                raise AssertionError(f"{name} has trainable parameters (Rule A)")
        log.info("channel is native: DDIM, eta=0, init_noise_sigma=%.6f, all modules frozen", sigma)

    # -- the latent ------------------------------------------------------
    def sample_latent(self, *seed_parts, batch: int = 1) -> torch.Tensor:
        """z ~ N(0, I) on the CPU (machine-independent), rounded to the dtype the
        generator will actually consume. Returns float32 [B, 4, 64, 64]."""
        g = torch.Generator(device="cpu").manual_seed(derive_seed(*seed_parts) % (2**63 - 1))
        z = torch.randn(batch, *self.latent_shape, generator=g, dtype=torch.float32)
        return z.to(self.dtype).float()      # what the UNet sees IS the target

    @torch.no_grad()
    def generate(self, z: torch.Tensor, prompts, guidance_scale: float | None = None,
                 num_inference_steps: int | None = None, return_float: bool = False):
        """X = G(Z). z: [B,4,64,64] float32 (already dtype-rounded)."""
        if z.dim() == 3:
            z = z.unsqueeze(0)
        if isinstance(prompts, str):
            prompts = [prompts] * z.shape[0]
        if len(prompts) != z.shape[0]:
            raise ValueError(f"{len(prompts)} prompts for {z.shape[0]} latents")
        latents = z.to(self.device, self.dtype)
        out = self.pipe(
            prompt=list(prompts),
            latents=latents,
            negative_prompt=[self.spec.negative_prompt] * z.shape[0],
            num_inference_steps=int(num_inference_steps or self.spec.num_inference_steps),
            guidance_scale=float(self.spec.guidance_scale if guidance_scale is None else guidance_scale),
            eta=float(self.spec.eta),
            height=self.spec.resolution, width=self.spec.resolution,
            output_type="np",
        )
        images = out.images                                   # [B,H,W,3] float32 in [0,1]
        if not np.isfinite(images).all():
            raise RuntimeError("generator produced non-finite pixels")
        if return_float:
            return images
        return np.clip(images * 255.0 + 0.5, 0, 255).astype(np.uint8)

    # -- Gate 0.2 --------------------------------------------------------
    @torch.no_grad()
    def latent_roundtrip_error(self, z: torch.Tensor) -> float:
        """max |prepare_latents(z) - z|. Must be 0: this is the assertion that
        `pipe(latents=z)` really is X = G(Z) and not X = G(c*z)."""
        latents = z.to(self.device, self.dtype)
        self.pipe.scheduler.set_timesteps(self.spec.num_inference_steps, device=self.device)
        prepared = self.pipe.prepare_latents(
            batch_size=z.shape[0], num_channels_latents=self.latent_shape[0],
            height=self.spec.resolution, width=self.spec.resolution,
            dtype=self.dtype, device=self.device, generator=None, latents=latents,
        )
        return float((prepared.float() - z.to(self.device).float()).abs().max())
