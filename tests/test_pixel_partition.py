import numpy as np
import pytest

from flow_grpo.pixel_partition_reward import (
    PARTITION_MODES,
    PIXEL_PARTITION_BASIS_ALIAS,
    PIXEL_PARTITION_KEYS,
    PartitionParams,
    PixelPartitionRewardScorer,
    expand_pixel_partition_reward_keys,
    is_pixel_partition_reward_key,
    parse_pixel_partition_reward_key,
    partition_fractions,
    partition_fractions_batch,
    partition_logits,
)

PARAMS = PartitionParams(tau=0.1, alpha=0.1, beta=1.0, gamma=1.0)


def _solid(rgb, size=8):
    return np.broadcast_to(np.asarray(rgb, dtype=np.float32), (size, size, 3)).copy()


def test_mode_order_matches_paper():
    assert PARTITION_MODES == (
        "red_dom",
        "green_dom",
        "blue_dom",
        "warm",
        "cool",
        "bright_ach",
        "dark_ach",
    )
    assert PIXEL_PARTITION_KEYS[0] == "pixel_partition_red_dom"


def test_logits_follow_the_appendix_formulas():
    pixel = np.array([0.9, 0.4, 0.2])
    r, g, b = pixel
    logits = partition_logits(pixel, PARAMS)
    saturation = pixel.max() - pixel.min()
    lightness = pixel.mean()
    expected = [
        r - (g + b) / 2,
        g - (r + b) / 2,
        b - (r + g) / 2,
        (r + g) / 2 - b,
        (g + b) / 2 - r,
        0.1 - saturation + (lightness - 0.5),
        0.1 - saturation - (lightness - 0.5),
    ]
    np.testing.assert_allclose(logits, expected, atol=1e-12)


@pytest.mark.parametrize(
    ("rgb", "mode"),
    [
        ((1.0, 0.0, 0.0), "red_dom"),
        ((0.0, 1.0, 0.0), "green_dom"),
        ((0.0, 0.0, 1.0), "blue_dom"),
        ((1.0, 1.0, 0.0), "warm"),
        ((0.0, 1.0, 1.0), "cool"),
        ((1.0, 1.0, 1.0), "bright_ach"),
        ((0.0, 0.0, 0.0), "dark_ach"),
    ],
)
def test_solid_colors_are_assigned_to_their_mode(rgb, mode):
    fractions = partition_fractions(_solid(rgb), PARAMS)
    assert fractions.shape == (7,)
    assert fractions.sum() == pytest.approx(1.0)
    assert PARTITION_MODES[int(fractions.argmax())] == mode
    assert fractions.max() > 0.9


def test_fractions_form_a_partition_for_random_images():
    rng = np.random.default_rng(0)
    images = rng.random((5, 12, 10, 3), dtype=np.float32)
    fractions = partition_fractions_batch(images, PARAMS)
    assert fractions.shape == (5, 7)
    np.testing.assert_allclose(fractions.sum(axis=1), np.ones(5), atol=1e-5)
    assert np.all(fractions >= 0)


def test_batch_coercion_accepts_uint8_and_channel_first_inputs():
    hwc_uint8 = (np.stack([_solid((1.0, 0.0, 0.0)), _solid((0.0, 0.0, 1.0))]) * 255).astype(
        np.uint8
    )
    chw_float = hwc_uint8.astype(np.float32).transpose(0, 3, 1, 2) / 255.0
    from_uint8 = partition_fractions_batch(hwc_uint8, PARAMS)
    from_chw = partition_fractions_batch(chw_float, PARAMS)
    np.testing.assert_allclose(from_uint8, from_chw, atol=1e-5)
    assert PARTITION_MODES[int(from_uint8[0].argmax())] == "red_dom"
    assert PARTITION_MODES[int(from_uint8[1].argmax())] == "blue_dom"


def test_torch_tensor_inputs_match_numpy():
    torch = pytest.importorskip("torch")
    images = torch.rand(3, 3, 6, 6)
    expected = partition_fractions_batch(images.numpy(), PARAMS)
    np.testing.assert_allclose(partition_fractions_batch(images, PARAMS), expected, atol=1e-6)


def test_key_helpers_and_alias_expansion():
    assert expand_pixel_partition_reward_keys([PIXEL_PARTITION_BASIS_ALIAS]) == list(
        PIXEL_PARTITION_KEYS
    )
    assert (
        expand_pixel_partition_reward_keys(["pixel_partition_warm", PIXEL_PARTITION_BASIS_ALIAS])[0]
        == "pixel_partition_warm"
    )
    assert is_pixel_partition_reward_key("pixel_partition_cool")
    assert not is_pixel_partition_reward_key("clip_prompt_race5_v4_0")
    assert parse_pixel_partition_reward_key("pixel_partition_dark_ach") == 6
    with pytest.raises(ValueError):
        parse_pixel_partition_reward_key("pixel_partition_magenta")


def test_scorer_returns_requested_columns():
    scorer = PixelPartitionRewardScorer(PARAMS)
    images = np.stack([_solid((1.0, 0.0, 0.0)), _solid((1.0, 1.0, 1.0))])
    scored = scorer.score_keys(images, ["ignored", "prompts"], list(PIXEL_PARTITION_KEYS))
    assert set(scored) == set(PIXEL_PARTITION_KEYS)
    assert scored["pixel_partition_red_dom"][0] > 0.9
    assert scored["pixel_partition_bright_ach"][1] > 0.9
    total = np.sum([scored[key] for key in PIXEL_PARTITION_KEYS], axis=0)
    np.testing.assert_allclose(total, np.ones(2), atol=1e-5)


def test_temperature_and_achromatic_parameters_are_used():
    image = _solid((0.5, 0.5, 0.5))
    flat = partition_fractions(image, PartitionParams(tau=10.0))
    sharp = partition_fractions(image, PartitionParams(tau=0.01))
    assert flat.max() < sharp.max()
    biased = partition_fractions(image, PartitionParams(alpha=5.0))
    assert biased[5] + biased[6] > 0.99
