"""Distributed prompt grouping used by the paper trainer."""

from __future__ import annotations

import torch
from torch.utils.data import Sampler


class DistributedKRepeatSampler(Sampler):
    """Create global prompt groups of exactly ``k`` samples across all ranks."""

    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed=0):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.k = int(k)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        if self.batch_size < 1 or self.k < 1 or self.num_replicas < 1:
            raise ValueError("batch_size, k, and num_replicas must be positive")
        if not 0 <= self.rank < self.num_replicas:
            raise ValueError(
                f"rank must satisfy 0 <= rank < num_replicas; got {self.rank} and "
                f"{self.num_replicas}"
            )
        total = self.num_replicas * self.batch_size
        if total % self.k:
            raise ValueError(
                "world_size * sample_batch_size must be divisible by candidates per prompt; "
                f"got {self.num_replicas} * {self.batch_size} and m={self.k}"
            )
        self.unique_per_batch = total // self.k
        if len(self.dataset) < self.unique_per_batch:
            raise ValueError(
                f"dataset has {len(self.dataset)} prompts but each global batch needs "
                f"{self.unique_per_batch} unique prompts"
            )
        self.epoch = 0

    def __iter__(self):
        while True:
            generator = torch.Generator().manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=generator)[
                : self.unique_per_batch
            ].tolist()
            repeated = [index for index in indices for _ in range(self.k)]
            permutation = torch.randperm(len(repeated), generator=generator).tolist()
            shuffled = [repeated[index] for index in permutation]
            start = self.rank * self.batch_size
            yield shuffled[start : start + self.batch_size]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
