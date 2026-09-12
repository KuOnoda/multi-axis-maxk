import numpy as np
import pytest

from multi_axis_maxk.metrics import fairness_score, finite_batch_visibility


def test_uniform_distribution_has_unit_fairness():
    assert fairness_score([10, 10, 10, 10, 10]) == pytest.approx(1.0)


def test_single_mode_distribution_has_zero_fairness():
    assert fairness_score([10, 0, 0, 0, 0]) == pytest.approx(0.0)


def test_finite_batch_visibility_reduces_sample_dimension():
    scores = np.array([[[0.7, 0.1], [0.2, 0.8]], [[0.1, 0.4], [0.2, 0.3]]])
    np.testing.assert_array_equal(
        finite_batch_visibility(scores, 0.5),
        np.array([[True, True], [False, False]]),
    )


def test_axiswise_batch_max_coverage_uses_first_m_samples():
    from multi_axis_maxk.metrics import axiswise_batch_max_coverage

    scores = np.array(
        [
            [[0.1, 0.9], [0.8, 0.2], [1.0, 1.0]],
            [[0.5, 0.5], [0.6, 0.4], [0.0, 0.0]],
        ]
    )
    # m=2: prompt 0 -> max over first two samples = (0.8, 0.9); prompt 1 -> (0.6, 0.5).
    assert axiswise_batch_max_coverage(scores, m=2) == pytest.approx((0.8 + 0.9 + 0.6 + 0.5) / 4)
    assert axiswise_batch_max_coverage(scores) == pytest.approx((1.0 + 1.0 + 0.6 + 0.5) / 4)
    with pytest.raises(ValueError):
        axiswise_batch_max_coverage(scores, m=4)
