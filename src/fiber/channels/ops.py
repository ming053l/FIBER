"""The channel T: X -> Y.

Non-differentiable on purpose. X is a cached constant (PLAN.md §2a), so nothing
here needs a gradient and we can use the real codecs a real channel would use.

Convention: images are uint8 HWC RGB numpy arrays, and every op returns an image
of the SAME shape as its input (resize goes down and back up), so Y always has
the carrier's geometry and the extractor never faces an alignment problem.
"""
from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image

_INTERP = {
    "bilinear": Image.BILINEAR,
    "bicubic": Image.BICUBIC,
    "nearest": Image.NEAREST,
    "lanczos": Image.LANCZOS,
}


def _to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(img, mode="RGB")


def identity(img: np.ndarray, *, rng=None) -> np.ndarray:
    return img.copy()


def jpeg(img: np.ndarray, *, quality: int, rng=None) -> np.ndarray:
    buf = io.BytesIO()
    # subsampling pinned so a Pillow upgrade cannot silently change the channel
    _to_pil(img).save(buf, format="JPEG", quality=int(quality), subsampling=2, optimize=False)
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"), dtype=np.uint8)


def webp(img: np.ndarray, *, quality: int, rng=None) -> np.ndarray:
    buf = io.BytesIO()
    _to_pil(img).save(buf, format="WEBP", quality=int(quality), method=4)
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"), dtype=np.uint8)


def resize(img: np.ndarray, *, scale: float, interp: str = "bilinear", rng=None) -> np.ndarray:
    h, w = img.shape[:2]
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    f = _INTERP[interp]
    small = _to_pil(img).resize((nw, nh), f)
    return np.array(small.resize((w, h), f).convert("RGB"), dtype=np.uint8)


def gaussian_noise(img: np.ndarray, *, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """sigma is in [0,1] units of image range, applied before quantisation."""
    x = img.astype(np.float32) / 255.0
    x = x + rng.standard_normal(x.shape, dtype=np.float32) * float(sigma)
    return np.clip(x * 255.0 + 0.5, 0, 255).astype(np.uint8)


def gaussian_blur(img: np.ndarray, *, sigma: float, kernel: int, rng=None) -> np.ndarray:
    k = int(kernel)
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(img, (k, k), sigmaX=float(sigma), sigmaY=float(sigma),
                            borderType=cv2.BORDER_REFLECT_101)


OPS = {
    "identity": identity,
    "jpeg": jpeg,
    "webp": webp,
    "resize": resize,
    "gaussian_noise": gaussian_noise,
    "gaussian_blur": gaussian_blur,
}

# Stochastic ops must receive a derived rng; deterministic ones must not depend on one.
STOCHASTIC = {"gaussian_noise"}
