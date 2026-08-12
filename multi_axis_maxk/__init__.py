"""Core, model-independent utilities for multi-axis max@K."""

from .advantages import (
    advantages_from_group_ids,
    axiswise_maxk_advantages,
    group_count_advantages,
)
from .metrics import fairness_score

__all__ = [
    "advantages_from_group_ids",
    "axiswise_maxk_advantages",
    "fairness_score",
    "group_count_advantages",
]
