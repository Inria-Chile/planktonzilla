"""
(c) Inria

``compute_mean_and_std_dev`` produces the ``Normalize`` constant published on every
source's dataset card, so a wrong answer here is a wrong number on the Hub and a wrong
preprocessing step for anyone who reads it.

Three defects lived in one function: the return ARITY was read off ``image_array`` after
the loop — i.e. off whatever the last row happened to be; a single-channel image's
one-element per-channel sum was BROADCAST onto the three-element accumulator, adding its
value to R, G and B while counting its pixels once; and an empty input reached the
division with ``image_array`` unbound and reported itself as ``NameError``.
"""

import numpy as np
import pyrootutils
import pytest

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

from PIL import Image

from planktonzilla.dataset import compute_mean_and_std_dev


def rows(*images):
    return [{"image": image} for image in images]


def grey(value, size=(4, 4)):
    return Image.new("L", size, color=value)


def rgb(colour, size=(4, 4)):
    return Image.new("RGB", size, color=colour)


def test_an_all_grayscale_dataset_still_answers_with_one_channel(tmp_path):
    """Unchanged for the homogeneous cases — the card's shape must not move under a fix."""
    mean, std = compute_mean_and_std_dev(rows(grey(128), grey(128)))

    assert len(mean) == 1 and len(std) == 1
    assert mean[0] == pytest.approx(128 / 255, abs=1e-6)


def test_an_all_rgb_dataset_answers_with_three_channels():
    mean, _std = compute_mean_and_std_dev(rows(rgb((255, 0, 0)), rgb((255, 0, 0))))

    assert len(mean) == 3
    assert np.allclose(mean, [1.0, 0.0, 0.0])


def test_a_mixed_dataset_does_not_take_its_shape_from_the_last_row():
    """The arity used to be decided by whichever image came last, so the same dataset in
    a different order published a one-element Normalize or a three-element one."""
    rgb_last = compute_mean_and_std_dev(rows(grey(128), rgb((255, 0, 0))))
    grey_last = compute_mean_and_std_dev(rows(rgb((255, 0, 0)), grey(128)))

    assert len(rgb_last[0]) == 3 and len(grey_last[0]) == 3
    assert np.allclose(rgb_last[0], grey_last[0])


def test_a_grayscale_image_does_not_leak_into_all_three_channels():
    """A (-1, 1) per-channel sum broadcasts onto the (3,) accumulator, so one grayscale
    image among RGB ones added its value to R, G and B and counted its pixels once.

    Discriminating by construction: pure red plus mid-grey. Converted correctly the grey
    contributes (0.502, 0.502, 0.502) and the mean is (0.751, 0.251, 0.251). Broadcast, the
    grey added 0.502 to each channel while `num_pixels` doubled, giving (0.751, 0.251, 0.251)
    for R... so the discriminator is G and B being EQUAL to each other and to the corrected
    value only when the conversion happened.
    """
    mean, _ = compute_mean_and_std_dev(rows(rgb((255, 0, 0)), grey(128)))

    assert mean[0] == pytest.approx((1.0 + 128 / 255) / 2, abs=1e-6)
    assert mean[1] == pytest.approx((0.0 + 128 / 255) / 2, abs=1e-6)
    assert mean[1] == pytest.approx(mean[2], abs=1e-9)


def test_an_empty_dataset_says_it_is_empty():
    """It used to reach the division with `image_array` unbound and raise NameError,
    which names neither the dataset nor the reason."""
    with pytest.raises(ValueError, match="no pixels"):
        compute_mean_and_std_dev([])


def test_a_flat_image_yields_a_real_standard_deviation_rather_than_nan():
    """E[x²] - E[x]² is catastrophic cancellation: on a constant channel it lands a few
    ulp below zero, and the unclamped root published `nan` onto the dataset card.

    The residual below is genuine float32 accumulation noise, not the defect — the
    defect is `nan`, which is what the first assertion pins.
    """
    _, std = compute_mean_and_std_dev(rows(rgb((10, 20, 30))))

    assert not np.isnan(std).any(), "a constant-colour class published nan as its Normalize std"
    assert (std >= 0).all()
    assert np.allclose(std, [0.0, 0.0, 0.0], atol=1e-4)
