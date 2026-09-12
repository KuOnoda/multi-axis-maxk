from collections import Counter

import pytest

torch = pytest.importorskip("torch")
sampling_module = pytest.importorskip("multi_axis_maxk.sampling")
DistributedKRepeatSampler = sampling_module.DistributedKRepeatSampler


def test_distributed_sampler_forms_exact_global_groups():
    dataset = list(range(1080))
    for epoch in (0, 1, 42):
        global_batch = []
        for rank in range(32):
            sampler = DistributedKRepeatSampler(
                dataset,
                batch_size=8,
                k=16,
                num_replicas=32,
                rank=rank,
                seed=7,
            )
            sampler.set_epoch(epoch)
            global_batch.extend(next(iter(sampler)))
        counts = Counter(global_batch)
        assert len(counts) == 16
        assert set(counts.values()) == {16}


def test_distributed_sampler_rejects_incompatible_batch_geometry():
    with pytest.raises(ValueError, match="must be divisible"):
        DistributedKRepeatSampler(
            list(range(100)),
            batch_size=3,
            k=16,
            num_replicas=2,
            rank=0,
        )


@pytest.mark.parametrize("rank", [-1, 2])
def test_distributed_sampler_rejects_invalid_rank(rank):
    with pytest.raises(ValueError, match="rank must satisfy"):
        DistributedKRepeatSampler(
            list(range(100)),
            batch_size=8,
            k=8,
            num_replicas=2,
            rank=rank,
        )


def test_distributed_sampler_requires_enough_unique_prompts():
    with pytest.raises(ValueError, match="unique prompts"):
        DistributedKRepeatSampler(
            list(range(3)),
            batch_size=8,
            k=4,
            num_replicas=2,
            rank=0,
        )
