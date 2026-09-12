"""Exact finite-batch multi-axis max@K advantages.

The EI and leave-two-out kernels are adapted from the internal ``remax_utils``
implementation used by the research code. This module adds the public API,
validation, multi-axis composition, and count-control implementation.

The last two reward dimensions are candidate and axis, respectively:
``rewards[..., m, d]``. Each axis retains its identity through EI+L2O credit
assignment; axes are summed only after per-axis normalization.

The rank formulas implement the without-replacement estimator used in the
experiments. This file is deliberately independent of the SD3 trainer so its
numerical contract can be tested on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def _binomial(n: torch.Tensor | int, k: torch.Tensor | int) -> torch.Tensor:
    """Numerically stable generalized binomial coefficient, zero when n < k."""

    n64 = torch.as_tensor(n, dtype=torch.float64)
    k64 = torch.as_tensor(k, dtype=torch.float64, device=n64.device)
    valid = (n64 >= k64) & (k64 >= 0)
    safe_n = torch.where(valid, n64, torch.zeros_like(n64))
    safe_k = torch.where(valid, k64, torch.zeros_like(k64))
    value = torch.exp(
        torch.lgamma(safe_n + 1) - torch.lgamma(safe_k + 1) - torch.lgamma(safe_n - safe_k + 1)
    )
    return torch.where(valid, value, torch.zeros_like(value))


def expected_improvement(rewards: torch.Tensor, k: int) -> torch.Tensor:
    """Return exact finite-batch EI for a one-dimensional reward vector.

    Comparator subsets contain ``k - 1`` elements sampled without replacement
    from the other ``m - 1`` candidates. For ``k == 1``, EI is the image reward
    itself, matching the paper's diagnostic endpoint before group centering.
    For ``k == m`` the only comparator subset is the whole rest of the group, so
    EI reduces to the leave-one-out marginal ``[r_i - max_{j != i} r_j]_+``.
    """

    if int(k) != k:
        raise ValueError(f"k must be integer-valued, got {k}")
    k = int(k)
    if rewards.ndim != 1:
        raise ValueError(f"expected a 1-D reward vector, got {tuple(rewards.shape)}")
    m = rewards.numel()
    if not 1 <= k <= m:
        raise ValueError(f"k must satisfy 1 <= k <= m; got k={k}, m={m}")
    if k == 1:
        return rewards.clone()

    order = torch.argsort(rewards, stable=True)
    sorted_rewards = rewards[order]
    inverse = torch.argsort(order)

    improvement = (sorted_rewards[:, None] - sorted_rewards[None, :]).clamp_min(0)
    lower_ranks = torch.arange(m, device=rewards.device)
    denominator = _binomial(torch.tensor(m - 1, device=rewards.device), k - 1)
    weights = _binomial(lower_ranks, k - 2) / denominator
    result = improvement @ weights.to(dtype=rewards.dtype)
    return result[inverse]


def leave_two_out_baseline(rewards: torch.Tensor, k: int) -> torch.Tensor:
    """Return the exact L2O baseline for each focal candidate.

    The baseline for candidate ``i`` averages other candidates' EI while
    excluding ``i`` from both the focal and comparator roles. It is defined for
    ``2 <= k <= m - 1``.
    """

    if int(k) != k:
        raise ValueError(f"k must be integer-valued, got {k}")
    k = int(k)
    if rewards.ndim != 1:
        raise ValueError(f"expected a 1-D reward vector, got {tuple(rewards.shape)}")
    m = rewards.numel()
    if not 2 <= k <= m - 1:
        raise ValueError(f"L2O requires 2 <= k <= m - 1; got k={k}, m={m}")

    order = torch.argsort(rewards, stable=True)
    sorted_rewards = rewards[order]
    inverse = torch.argsort(order)
    improvement = (sorted_rewards[:, None] - sorted_rewards[None, :]).clamp_min(0)

    # Row i is the excluded sample. For comparator rank ell, remove i from the
    # count of lower-ranked candidates whenever ell lies above i.
    excluded_rank = torch.arange(m, device=rewards.device)[:, None]
    comparator_rank = torch.arange(m, device=rewards.device)[None, :]
    rank_shift = (comparator_rank > excluded_rank).to(dtype=comparator_rank.dtype)
    effective_lower = comparator_rank - rank_shift
    denominator = _binomial(torch.tensor(m - 2, device=rewards.device), k - 1)
    comparator_weight = _binomial(effective_lower, k - 2) / denominator
    comparator_weight = comparator_weight.to(dtype=rewards.dtype)
    comparator_weight.fill_diagonal_(0)

    # Mean improvement of all permissible focal candidates over comparator ell.
    column_sum = improvement.sum(dim=0)
    mean_without_excluded = (column_sum[None, :] - improvement) / (m - 1)
    baseline_sorted = (mean_without_excluded * comparator_weight).sum(dim=1)
    return baseline_sorted[inverse]


@dataclass(frozen=True)
class AxiswiseAdvantageOutput:
    """Outputs of :func:`axiswise_maxk_advantages`."""

    advantages: torch.Tensor
    per_axis_advantages: torch.Tensor
    expected_improvement: torch.Tensor
    baseline: torch.Tensor


def group_count_advantages(
    rewards: torch.Tensor,
    *,
    clip: float = 5.0,
    eps: float = 1e-6,
    valid_atol: float = 1e-6,
    normalize: bool = True,
) -> torch.Tensor:
    """Return the matched hard-count GRPO control used in the paper.

    ``rewards`` has shape ``(..., m, d)``. Each candidate receives the clipped
    inverse-frequency score of its top-scoring mode within the prompt group.
    The result is centered (and optionally standardized) over candidates.

    Near-uniform scores (range <= ``valid_atol``) are excluded from counts.
    This fallback guard extends the count definition in paper Eqs. (18)-(21).
    """

    if rewards.ndim < 2:
        raise ValueError("rewards must have candidate and axis dimensions")
    m, d = rewards.shape[-2:]
    if m < 2 or d < 2:
        raise ValueError("count credit requires at least two candidates and two axes")
    modes = rewards.argmax(dim=-1)
    valid = rewards.amax(dim=-1) - rewards.amin(dim=-1) > valid_atol
    membership = torch.nn.functional.one_hot(modes, num_classes=d) * valid.unsqueeze(-1)
    counts = membership.sum(dim=-2)
    valid_count = valid.sum(dim=-1, keepdim=True)
    class_scores = torch.log((valid_count - counts + eps) / (counts + eps))
    class_scores = (class_scores - class_scores.mean(dim=-1, keepdim=True)).clamp(-clip, clip)
    raw = class_scores.gather(dim=-1, index=modes) * valid
    raw = torch.where(valid_count >= 2, raw, torch.zeros_like(raw))
    centered = raw - raw.mean(dim=-1, keepdim=True)
    if not normalize:
        return centered
    scale = centered.std(dim=-1, keepdim=True)
    scale = torch.where(scale < 1e-8, torch.ones_like(scale), scale)
    return centered / (scale + 1e-4)


def advantages_from_group_ids(
    group_ids: torch.Tensor,
    rewards: torch.Tensor,
    *,
    candidates_per_group: int,
    k: int,
    method: str = "maxk",
    m_eq_k_baseline: str = "group_mean",
) -> torch.Tensor:
    """Compute credit for shuffled distributed samples and restore their order."""

    if group_ids.ndim != 1 or rewards.ndim != 2 or rewards.shape[0] != group_ids.numel():
        raise ValueError("expected group_ids (N,) and rewards (N, D)")
    order = torch.argsort(group_ids, stable=True)
    sorted_ids = group_ids[order]
    unique_ids, counts = torch.unique_consecutive(sorted_ids, return_counts=True)
    if not torch.all(counts == candidates_per_group):
        invalid = counts != candidates_per_group
        bad = list(
            zip(  # noqa: B905 - Python 3.9 compatibility for local CPU checks
                unique_ids[invalid].tolist(), counts[invalid].tolist()
            )
        )
        raise ValueError(
            f"malformed groups (expected {candidates_per_group} candidates): {bad[:5]}"
        )
    grouped = rewards[order].reshape(-1, candidates_per_group, rewards.shape[-1])
    if method == "maxk":
        grouped_advantages = axiswise_maxk_advantages(
            grouped, k=k, m_eq_k_baseline=m_eq_k_baseline
        ).advantages
    elif method == "count":
        grouped_advantages = group_count_advantages(grouped)
    else:
        raise ValueError("method must be 'maxk' or 'count'")
    restored = torch.empty_like(grouped_advantages.reshape(-1))
    restored[order] = grouped_advantages.reshape(-1)
    return restored


def axiswise_maxk_advantages(
    rewards: torch.Tensor,
    k: int,
    *,
    axis_weights: torch.Tensor | None = None,
    normalize: bool = True,
    eps: float = 1e-4,
    m_eq_k_baseline: str = "group_mean",
) -> AxiswiseAdvantageOutput:
    """Compute axis-preserving max@K credit for grouped rewards.

    Args:
        rewards: Tensor of shape ``(..., m, d)``.
        k: Representative window. ``2 <= k <= m-1`` uses EI with the L2O
            baseline; ``k=1`` uses centered image rewards; ``k == m`` uses the
            leave-one-out marginal EI ``[r_i - max_{j != i} r_j]_+`` with the
            baseline selected by ``m_eq_k_baseline``.
        axis_weights: Optional length-``d`` weights. Defaults to all ones.
        normalize: Divide each realized group/axis advantage by its sample std.
        eps: Numerical stabilizer used after the zero-scale guard.
        m_eq_k_baseline: Baseline used only when ``k == m``, where L2O is
            undefined. ``"group_mean"`` subtracts the per-axis group mean of EI,
            as in the reported ``m=k=7`` color runs; ``"none"`` uses EI
            directly.
    """

    if int(k) != k:
        raise ValueError(f"k must be integer-valued, got {k}")
    k = int(k)
    if rewards.ndim < 2:
        raise ValueError("rewards must have candidate and axis dimensions")
    m, d = rewards.shape[-2:]
    if k == 1:
        ei = rewards.clone()
        baseline = rewards.mean(dim=-2, keepdim=True).expand_as(rewards)
        raw = ei - baseline
    else:
        if not 2 <= k <= m:
            raise ValueError(f"max@K credit requires 2 <= k <= m; got k={k}, m={m}")
        if m_eq_k_baseline not in {"group_mean", "none"}:
            raise ValueError("m_eq_k_baseline must be 'group_mean' or 'none'")
        flat = rewards.reshape(-1, m, d)
        ei_flat = torch.empty_like(flat)
        baseline_flat = torch.empty_like(flat)
        for group in range(flat.shape[0]):
            for axis in range(d):
                vector = flat[group, :, axis]
                ei_flat[group, :, axis] = expected_improvement(vector, k)
                if k <= m - 1:
                    baseline_flat[group, :, axis] = leave_two_out_baseline(vector, k)
                elif m_eq_k_baseline == "group_mean":
                    baseline_flat[group, :, axis] = ei_flat[group, :, axis].mean().expand(m)
                else:
                    baseline_flat[group, :, axis] = torch.zeros_like(vector)
        ei = ei_flat.reshape_as(rewards)
        baseline = baseline_flat.reshape_as(rewards)
        raw = ei - baseline

    per_axis = raw
    if normalize:
        scale = raw.std(dim=-2, keepdim=True)
        scale = torch.where(scale < 1e-8, torch.ones_like(scale), scale)
        per_axis = raw / (scale + eps)

    if axis_weights is None:
        weights = torch.ones(d, dtype=rewards.dtype, device=rewards.device)
    else:
        weights = torch.as_tensor(axis_weights, dtype=rewards.dtype, device=rewards.device)
        if weights.shape != (d,):
            raise ValueError(f"axis_weights must have shape ({d},), got {tuple(weights.shape)}")

    return AxiswiseAdvantageOutput(
        advantages=(per_axis * weights).sum(dim=-1),
        per_axis_advantages=per_axis,
        expected_improvement=ei,
        baseline=baseline,
    )
