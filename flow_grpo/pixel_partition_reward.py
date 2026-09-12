"""Rule-based seven-mode color partition rewards (paper Sec. 5.2, Appendix on color rewards).

Every pixel ``(r, g, b) in [0, 1]^3`` receives seven logits::

    red_dom    = r - (g + b) / 2
    green_dom  = g - (r + b) / 2
    blue_dom   = b - (r + g) / 2
    warm       = (r + g) / 2 - b
    cool       = (g + b) / 2 - r
    bright_ach = alpha - beta * saturation + gamma * (lightness - 0.5)
    dark_ach   = alpha - beta * saturation - gamma * (lightness - 0.5)

with ``saturation = max(r, g, b) - min(r, g, b)`` and ``lightness = (r + g + b) / 3``.
A temperature softmax over the seven logits gives per-pixel memberships, and the
image-level reward of a mode is the mean membership over pixels. The seven
rewards therefore sum to one for every image: the axes form a soft competing
partition rather than independent color heuristics.

Reward keys are ``pixel_partition_<mode>``. The alias ``pixel_partition_basis``
expands to all seven keys in the fixed mode order used by the paper. Tunables
are read from the environment once at import time and default to the reported
values: ``PARTITION_TAU=0.1``, ``PARTITION_ACH_ALPHA=0.1``,
``PARTITION_ACH_BETA=1.0``, ``PARTITION_ACH_GAMMA=1.0``.

The module depends on NumPy only. Torch tensors and PIL images are accepted as
inputs and converted on the fly.

Limitations: the axes are pixel statistics, not perceptual color categories;
they respond to global tint and exposure and carry no information about prompt
fidelity or quality, which is why training multiplies them by frozen PickScore.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

PARTITION_MODES: tuple[str, ...] = (
    "red_dom",
    "green_dom",
    "blue_dom",
    "warm",
    "cool",
    "bright_ach",
    "dark_ach",
)
PIXEL_PARTITION_PREFIX = "pixel_partition_"
PIXEL_PARTITION_BASIS_ALIAS = "pixel_partition_basis"
PIXEL_PARTITION_KEYS: tuple[str, ...] = tuple(
    f"{PIXEL_PARTITION_PREFIX}{mode}" for mode in PARTITION_MODES
)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


@dataclass(frozen=True)
class PartitionParams:
    """Softmax temperature and achromatic-logit coefficients."""

    tau: float = 0.1
    alpha: float = 0.1
    beta: float = 1.0
    gamma: float = 1.0

    @classmethod
    def from_env(cls) -> PartitionParams:
        params = cls(
            tau=_env_float("PARTITION_TAU", cls.tau),
            alpha=_env_float("PARTITION_ACH_ALPHA", cls.alpha),
            beta=_env_float("PARTITION_ACH_BETA", cls.beta),
            gamma=_env_float("PARTITION_ACH_GAMMA", cls.gamma),
        )
        if params.tau <= 0:
            raise ValueError("PARTITION_TAU must be > 0")
        return params


DEFAULT_PARAMS = PartitionParams.from_env()


def partition_logits(pixels: np.ndarray, params: PartitionParams = DEFAULT_PARAMS) -> np.ndarray:
    """Return the seven per-pixel logits for an array shaped ``(..., 3)`` in ``[0, 1]``."""

    pixels = np.asarray(pixels, dtype=np.float64)
    if pixels.shape[-1] != 3:
        raise ValueError(f"expected trailing RGB dimension of size 3, got {pixels.shape}")
    r = pixels[..., 0]
    g = pixels[..., 1]
    b = pixels[..., 2]
    saturation = pixels.max(axis=-1) - pixels.min(axis=-1)
    lightness = pixels.mean(axis=-1)
    return np.stack(
        [
            r - 0.5 * (g + b),
            g - 0.5 * (r + b),
            b - 0.5 * (r + g),
            0.5 * (r + g) - b,
            0.5 * (g + b) - r,
            params.alpha - params.beta * saturation + params.gamma * (lightness - 0.5),
            params.alpha - params.beta * saturation - params.gamma * (lightness - 0.5),
        ],
        axis=-1,
    )


def partition_fractions(image: np.ndarray, params: PartitionParams = DEFAULT_PARAMS) -> np.ndarray:
    """Return the length-7 mode fractions of one ``HxWx3`` image in ``[0, 1]``.

    The output is ``float64``, ordered as :data:`PARTITION_MODES`, and sums to one.
    """

    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"expected an HxWx3 image, got shape {image.shape}")
    logits = partition_logits(image, params) / params.tau
    logits -= logits.max(axis=-1, keepdims=True)
    weights = np.exp(logits)
    memberships = weights / weights.sum(axis=-1, keepdims=True)
    return memberships.reshape(-1, len(PARTITION_MODES)).mean(axis=0)


def to_hwc_float_batch(images) -> list[np.ndarray]:
    """Coerce a trainer or evaluation image batch to a list of ``HxWx3`` floats in ``[0, 1]``.

    Accepts torch tensors shaped ``(B, 3, H, W)`` or ``(B, H, W, 3)``, sequences of
    PIL images, and NumPy arrays. Integer inputs and inputs whose maximum exceeds
    one are interpreted as ``0..255`` values.
    """

    if hasattr(images, "detach") and hasattr(images, "cpu"):
        array = images.detach().cpu().float().numpy()
    else:
        try:
            first = images[0]
        except (TypeError, IndexError, KeyError):
            first = None
        if first is not None and hasattr(first, "convert"):
            return [np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0 for image in images]
        array = np.asarray(images)
    if array.ndim == 3:
        array = array[None]
    if array.ndim != 4:
        raise ValueError(f"expected a batch of images, got shape {array.shape}")
    if array.shape[1] == 3 and array.shape[-1] != 3:
        array = array.transpose(0, 2, 3, 1)
    array = array.astype(np.float32, copy=False)
    if array.size and (array.dtype.kind in "ui" or float(array.max()) > 1.0 + 1e-3):
        array = array / 255.0
    array = np.clip(array, 0.0, 1.0)
    return [array[index] for index in range(array.shape[0])]


def partition_fractions_batch(images, params: PartitionParams = DEFAULT_PARAMS) -> np.ndarray:
    """Return an array shaped ``(B, 7)`` of mode fractions for a batch of images."""

    arrays = to_hwc_float_batch(images)
    if not arrays:
        return np.zeros((0, len(PARTITION_MODES)), dtype=np.float32)
    fractions = np.stack([partition_fractions(image, params) for image in arrays], axis=0)
    return np.where(np.isfinite(fractions), fractions, 0.0).astype(np.float32)


def is_pixel_partition_reward_key(key: str) -> bool:
    return key in PIXEL_PARTITION_KEYS


def parse_pixel_partition_reward_key(key: str) -> int:
    """Return the mode index of a ``pixel_partition_<mode>`` key."""

    if not is_pixel_partition_reward_key(key):
        raise ValueError(
            f"{key!r} is not a pixel partition reward key; expected one of {PIXEL_PARTITION_KEYS}"
        )
    return PIXEL_PARTITION_KEYS.index(key)


def expand_pixel_partition_reward_keys(keys: Sequence[str]) -> list[str]:
    """Expand ``pixel_partition_basis`` to the seven mode keys, preserving order."""

    expanded: list[str] = []
    for key in keys:
        if key == PIXEL_PARTITION_BASIS_ALIAS:
            expanded.extend(PIXEL_PARTITION_KEYS)
        else:
            expanded.append(key)
    seen: set[str] = set()
    ordered: list[str] = []
    for key in expanded:
        if key not in seen:
            seen.add(key)
            ordered.append(key)
    return ordered


class PixelPartitionRewardScorer:
    """Batch scorer with the same ``score_keys`` contract as the CLIP prompt scorer."""

    def __init__(self, params: PartitionParams | None = None):
        self.params = params or DEFAULT_PARAMS

    def score_all(self, images) -> np.ndarray:
        return partition_fractions_batch(images, self.params)

    def score_keys(self, images, prompts, keys: Sequence[str]) -> dict[str, np.ndarray]:
        del prompts  # the rule-based reward is prompt-agnostic
        indices = [parse_pixel_partition_reward_key(key) for key in keys]
        fractions = self.score_all(images)
        return {key: fractions[:, index] for key, index in zip(keys, indices, strict=True)}
