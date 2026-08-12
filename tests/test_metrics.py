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
