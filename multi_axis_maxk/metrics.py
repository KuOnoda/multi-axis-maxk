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


def axiswise_batch_max_coverage(scores, m: int | None = None) -> float:
    """Prompt-level axis-wise batch-max coverage, Eq. (pixel-batchmax) in the paper.

    ``scores`` has shape ``(P, M, D)``: per-prompt sample scores on ``D`` axes.
    The metric takes the maximum over the first ``m`` samples of every prompt and
    axis, then averages over prompts and axes. With the seven rule-based color
    axes and ``m=12`` this is the paper's ``C_pix@12``.
    """

    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("scores must have shape (prompts, samples, axes)")
    if m is None:
        m = values.shape[1]
    if not 1 <= m <= values.shape[1]:
        raise ValueError(f"m must satisfy 1 <= m <= {values.shape[1]}, got {m}")
    return float(values[:, :m, :].max(axis=1).mean())
