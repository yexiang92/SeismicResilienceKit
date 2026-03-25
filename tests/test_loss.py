"""Tests for seismicresiliencekit.loss module."""

import math

import numpy as np
import pytest

from seismicresiliencekit.fragility import FragilityCurve, FragilitySet
from seismicresiliencekit.hazard import HazardCurve
from seismicresiliencekit.loss import (
    LossFunction,
    compute_expected_loss,
    compute_loss_curve,
    expected_annual_loss,
)


@pytest.fixture
def simple_loss_fn():
    return LossFunction(
        [0.05, 0.10, 0.20, 0.40, 0.80],
        [0.01, 0.05, 0.15, 0.40, 0.80],
    )


@pytest.fixture
def simple_hazard():
    im = np.array([0.05, 0.10, 0.20, 0.40, 0.80])
    rates = np.array([0.10, 0.04, 0.01, 0.002, 0.0002])
    return HazardCurve(im, rates)


class TestLossFunction:
    def test_loss_ratio_at_node(self, simple_loss_fn):
        assert math.isclose(simple_loss_fn.loss_ratio(0.10), 0.05, rel_tol=1e-6)

    def test_loss_ratio_interpolated(self, simple_loss_fn):
        # Between 0.10 (0.05) and 0.20 (0.15) → at 0.15 should be ≈ 0.10
        lr = simple_loss_fn.loss_ratio(0.15)
        assert 0.05 < lr < 0.15

    def test_loss_ratio_clipped_at_zero(self, simple_loss_fn):
        # Below minimum IM, extrapolation is clipped to first value
        lr = simple_loss_fn.loss_ratio(0.001)
        assert lr >= 0.0

    def test_loss_ratio_in_range(self, simple_loss_fn):
        for im in [0.05, 0.2, 0.5, 1.0]:
            lr = simple_loss_fn.loss_ratio(im)
            assert 0.0 <= lr <= 1.0

    def test_expected_loss_with_replacement_value(self, simple_loss_fn):
        el = simple_loss_fn.expected_loss(0.10, replacement_value=100_000)
        assert math.isclose(el, 5_000, rel_tol=1e-6)

    def test_invalid_non_increasing_im(self):
        with pytest.raises(ValueError):
            LossFunction([0.2, 0.1], [0.05, 0.10])

    def test_invalid_loss_ratio_above_one(self):
        with pytest.raises(ValueError):
            LossFunction([0.1, 0.2], [0.5, 1.5])

    def test_repr(self, simple_loss_fn):
        assert "LossFunction" in repr(simple_loss_fn)

    def test_from_fragility_set(self):
        fs = FragilitySet([
            FragilityCurve("slight",    median=0.10, beta=0.4),
            FragilityCurve("moderate",  median=0.20, beta=0.4),
            FragilityCurve("extensive", median=0.40, beta=0.4),
            FragilityCurve("complete",  median=0.80, beta=0.4),
        ])
        state_lr = {"none": 0.0, "slight": 0.05, "moderate": 0.2, "extensive": 0.5, "complete": 1.0}
        lf = LossFunction.from_fragility_set(fs, state_lr)
        # Loss ratio should be 0 at very low IM
        assert lf.loss_ratio(0.001) < 0.01
        # Loss ratio should increase with IM
        assert lf.loss_ratio(0.5) > lf.loss_ratio(0.1)


class TestComputeExpectedLoss:
    def test_single_asset(self, simple_loss_fn):
        loss = compute_expected_loss(0.10, [simple_loss_fn], [100_000])
        assert math.isclose(loss, 5_000, rel_tol=1e-6)

    def test_multiple_assets(self, simple_loss_fn):
        loss = compute_expected_loss(0.10, [simple_loss_fn, simple_loss_fn], [100_000, 200_000])
        assert math.isclose(loss, 5_000 + 10_000, rel_tol=1e-6)

    def test_default_replacement_value(self, simple_loss_fn):
        loss = compute_expected_loss(0.10, [simple_loss_fn])
        assert math.isclose(loss, 0.05, rel_tol=1e-6)

    def test_mismatched_lengths_raises(self, simple_loss_fn):
        with pytest.raises(ValueError):
            compute_expected_loss(0.10, [simple_loss_fn], [1.0, 2.0])


class TestComputeLossCurve:
    def test_returns_two_arrays(self, simple_loss_fn, simple_hazard):
        lr, rates = compute_loss_curve(simple_loss_fn, simple_hazard)
        assert isinstance(lr, np.ndarray)
        assert isinstance(rates, np.ndarray)
        assert len(lr) == len(rates)

    def test_loss_ratios_are_positive(self, simple_loss_fn, simple_hazard):
        lr, _ = compute_loss_curve(simple_loss_fn, simple_hazard)
        assert np.all(lr >= 0)

    def test_annual_rates_are_positive(self, simple_loss_fn, simple_hazard):
        _, rates = compute_loss_curve(simple_loss_fn, simple_hazard)
        assert np.all(rates >= 0)


class TestExpectedAnnualLoss:
    def test_positive(self, simple_loss_fn, simple_hazard):
        eal = expected_annual_loss(simple_loss_fn, simple_hazard)
        assert eal > 0

    def test_in_range(self, simple_loss_fn, simple_hazard):
        eal = expected_annual_loss(simple_loss_fn, simple_hazard)
        assert 0.0 <= eal <= 1.0
