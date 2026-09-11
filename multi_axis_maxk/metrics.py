"""Paper-level pooled-composition and finite-batch metrics."""

from __future__ import annotations

import numpy as np


def fairness_score(frequencies, *, target=None) -> float:
    """Normalized total-variation fairness score in ``[0, 1]``.

    With the default uniform target this is Eq. (5) in the paper. Inputs may be
    counts or probabilities and are normalized internally.
    """

    values = np.asarray(frequencies, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or np.any(values < 0):
        raise ValueError("frequencies must be a non-negative 1-D vector of length >= 2")
    if values.sum() <= 0:
        raise ValueError("frequencies must have positive mass")
    probabilities = values / values.sum()
    if target is None:
        desired = np.full(values.size, 1.0 / values.size)
    else:
        desired = np.asarray(target, dtype=np.float64)
        if desired.shape != values.shape or np.any(desired < 0) or desired.sum() <= 0:
            raise ValueError("target must be a same-shaped non-negative vector")
        desired = desired / desired.sum()
    max_tv = 1.0 - desired.min()
    return float(1.0 - 0.5 * np.abs(probabilities - desired).sum() / max_tv)


def finite_batch_visibility(scores, threshold: float) -> np.ndarray:
    """Return per-group mode visibility for arrays shaped ``(..., M, D)``."""

    values = np.asarray(scores)
    if values.ndim < 2:
        raise ValueError("scores must have sample and mode dimensions")
    return values.max(axis=-2) >= threshold
