import itertools

import pytest

torch = pytest.importorskip("torch")
advantages_module = pytest.importorskip("multi_axis_maxk.advantages")
axiswise_maxk_advantages = advantages_module.axiswise_maxk_advantages
expected_improvement = advantages_module.expected_improvement
leave_two_out_baseline = advantages_module.leave_two_out_baseline
group_count_advantages = advantages_module.group_count_advantages
advantages_from_group_ids = advantages_module.advantages_from_group_ids


def _brute_ei(rewards, focal, k):
    others = [i for i in range(len(rewards)) if i != focal]
    improvements = []
    for subset in itertools.combinations(others, k - 1):
        improvements.append(max(rewards[focal] - max(rewards[j] for j in subset), 0.0))
    return sum(improvements) / len(improvements)


def _brute_l2o(rewards, excluded, k):
    values = []
    for focal in range(len(rewards)):
        if focal == excluded:
            continue
        pool = [i for i in range(len(rewards)) if i not in {excluded, focal}]
        improvements = []
        for subset in itertools.combinations(pool, k - 1):
            improvements.append(max(rewards[focal] - max(rewards[j] for j in subset), 0.0))
        values.append(sum(improvements) / len(improvements))
    return sum(values) / len(values)


@pytest.mark.parametrize("k", [2, 3, 4])
def test_rank_formula_matches_combinatorial_definition(k):
    rewards = torch.tensor([0.7, -0.2, 1.1, 0.4, 0.4, 2.0], dtype=torch.float64)
    ei = expected_improvement(rewards, k)
    baseline = leave_two_out_baseline(rewards, k)
    expected_ei = torch.tensor(
        [_brute_ei(rewards.tolist(), i, k) for i in range(len(rewards))],
        dtype=torch.float64,
    )
    expected_baseline = torch.tensor(
        [_brute_l2o(rewards.tolist(), i, k) for i in range(len(rewards))],
        dtype=torch.float64,
    )
    torch.testing.assert_close(ei, expected_ei)
    torch.testing.assert_close(baseline, expected_baseline)


def test_l2o_advantage_is_exactly_centered_per_axis():
    generator = torch.Generator().manual_seed(7)
    rewards = torch.rand((3, 8, 5), generator=generator)
    output = axiswise_maxk_advantages(rewards, k=5, normalize=False)
    torch.testing.assert_close(
        output.per_axis_advantages.mean(dim=1),
        torch.zeros((3, 5)),
        atol=2e-6,
        rtol=0,
    )


def test_axes_are_transformed_before_aggregation():
    rewards = torch.tensor(
        [[[0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.2, 0.8]]],
        dtype=torch.float64,
    )
    output = axiswise_maxk_advantages(rewards, k=2, normalize=False)
    assert output.per_axis_advantages.shape == rewards.shape
    assert output.advantages.shape == (1, 4)
    assert output.advantages[0, 0] > 0
    assert output.advantages[0, 2] > 0


def test_k1_is_centered_grpo_endpoint():
    rewards = torch.tensor([[[1.0, 4.0], [2.0, 1.0], [3.0, 1.0]]])
    output = axiswise_maxk_advantages(rewards, k=1, normalize=False)
    torch.testing.assert_close(output.per_axis_advantages.mean(dim=1), torch.zeros((1, 2)))


def test_count_credit_rewards_underrepresented_modes():
    rewards = torch.tensor(
        [[[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.1, 0.9]]],
        dtype=torch.float64,
    )
    advantage = group_count_advantages(rewards, normalize=False)
    assert advantage[0, 3] > 0
    assert torch.all(advantage[0, :3] < 0)
    torch.testing.assert_close(advantage.mean(dim=1), torch.zeros(1))


def test_count_credit_ignores_uniform_no_face_fallback():
    rewards = torch.tensor(
        [[[0.9, 0.1], [0.8, 0.2], [0.5, 0.5], [0.1, 0.9]]],
        dtype=torch.float64,
    )
    advantage = group_count_advantages(rewards, normalize=False)
    assert advantage[0, 0] == advantage[0, 1]
    assert advantage[0, 0] < advantage[0, 2] < advantage[0, 3]


def test_shuffled_distributed_groups_are_restored_to_input_order():
    grouped = torch.tensor(
        [
            [[0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.2, 0.8]],
            [[0.7, 0.3], [0.6, 0.4], [0.3, 0.7], [0.4, 0.6]],
        ],
        dtype=torch.float64,
    )
    expected = axiswise_maxk_advantages(grouped, k=2).advantages.reshape(-1)
    permutation = torch.tensor([5, 0, 7, 2, 4, 1, 6, 3])
    group_ids = torch.tensor([10] * 4 + [20] * 4)[permutation]
    rewards = grouped.reshape(-1, 2)[permutation]
    actual = advantages_from_group_ids(
        group_ids,
        rewards,
        candidates_per_group=4,
        k=2,
    )
    torch.testing.assert_close(actual, expected[permutation])


def test_malformed_group_is_rejected():
    with pytest.raises(ValueError, match="malformed groups"):
        advantages_from_group_ids(
            torch.tensor([0, 0, 0, 1]),
            torch.rand(4, 2),
            candidates_per_group=2,
            k=1,
        )


@pytest.mark.parametrize("m", [3, 5, 7])
def test_m_equal_k_expected_improvement_is_leave_one_out_marginal(m):
    torch.manual_seed(m)
    rewards = torch.rand(m, dtype=torch.float64)
    ei = expected_improvement(rewards, m)
    for focal in range(m):
        others = torch.cat([rewards[:focal], rewards[focal + 1 :]])
        assert ei[focal].item() == pytest.approx(
            max(float(rewards[focal] - others.max()), 0.0), abs=1e-10
        )
        assert ei[focal].item() == pytest.approx(_brute_ei(rewards.tolist(), focal, m), abs=1e-10)


def test_m_equal_k_uses_group_mean_baseline_by_default():
    rewards = torch.tensor([[[0.9, 0.1], [0.2, 0.7], [0.4, 0.3], [0.1, 0.2]]], dtype=torch.float64)
    result = axiswise_maxk_advantages(rewards, k=4, normalize=False)
    expected_ei = torch.tensor(
        [[[0.5, 0.0], [0.0, 0.4], [0.0, 0.0], [0.0, 0.0]]], dtype=torch.float64
    )
    group_mean = expected_ei.mean(dim=-2, keepdim=True)
    torch.testing.assert_close(result.expected_improvement, expected_ei)
    torch.testing.assert_close(result.baseline, group_mean.expand_as(expected_ei))
    torch.testing.assert_close(result.per_axis_advantages, expected_ei - group_mean)
    assert result.advantages.shape == (1, 4)
    assert result.advantages.sum().item() == pytest.approx(0.0, abs=1e-12)


def test_m_equal_k_none_baseline_returns_marginals():
    rewards = torch.tensor([[[0.9, 0.1], [0.2, 0.7], [0.4, 0.3]]], dtype=torch.float64)
    result = axiswise_maxk_advantages(rewards, k=3, normalize=False, m_eq_k_baseline="none")
    torch.testing.assert_close(result.baseline, torch.zeros_like(rewards))
    torch.testing.assert_close(result.advantages, result.expected_improvement.sum(dim=-1))
    with pytest.raises(ValueError):
        axiswise_maxk_advantages(rewards, k=3, m_eq_k_baseline="median")


def test_k_larger_than_m_is_rejected():
    rewards = torch.rand(1, 4, 2, dtype=torch.float64)
    with pytest.raises(ValueError):
        axiswise_maxk_advantages(rewards, k=5)
