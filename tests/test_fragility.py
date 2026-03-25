"""Tests for seismicresiliencekit.fragility module."""

import math

import numpy as np
import pytest

from seismicresiliencekit.fragility import FragilityCurve, FragilitySet


class TestFragilityCurve:
    def test_median_returns_half(self):
        fc = FragilityCurve("slight", median=0.1, beta=0.4)
        assert math.isclose(fc.probability(0.1), 0.5, abs_tol=1e-9)

    def test_probability_zero_at_zero_im(self):
        fc = FragilityCurve("slight", median=0.1, beta=0.4)
        assert fc.probability(0.0) == 0.0

    def test_probability_increases_with_im(self):
        fc = FragilityCurve("slight", median=0.2, beta=0.5)
        probs = [fc.probability(im) for im in [0.1, 0.2, 0.4, 0.8]]
        assert all(probs[i] < probs[i + 1] for i in range(len(probs) - 1))

    def test_probability_in_range(self):
        fc = FragilityCurve("slight", median=0.2, beta=0.5)
        for im in [0.01, 0.1, 0.5, 2.0]:
            p = fc.probability(im)
            assert 0.0 <= p <= 1.0

    def test_probabilities_vectorised(self):
        fc = FragilityCurve("slight", median=0.2, beta=0.5)
        im_arr = np.array([0.05, 0.1, 0.2, 0.5])
        result = fc.probabilities(im_arr)
        assert result.shape == (4,)
        assert math.isclose(result[2], 0.5, abs_tol=1e-9)

    def test_invalid_median_raises(self):
        with pytest.raises(ValueError):
            FragilityCurve("slight", median=-0.1, beta=0.4)

    def test_invalid_beta_raises(self):
        with pytest.raises(ValueError):
            FragilityCurve("slight", median=0.1, beta=0.0)

    def test_repr(self):
        fc = FragilityCurve("slight", median=0.1, beta=0.4)
        assert "FragilityCurve" in repr(fc)
        assert "slight" in repr(fc)


class TestFragilitySet:
    @pytest.fixture
    def four_state_set(self):
        return FragilitySet([
            FragilityCurve("slight",    median=0.10, beta=0.4),
            FragilityCurve("moderate",  median=0.20, beta=0.4),
            FragilityCurve("extensive", median=0.40, beta=0.4),
            FragilityCurve("complete",  median=0.80, beta=0.4),
        ])

    def test_damage_states_property(self, four_state_set):
        assert four_state_set.damage_states == ["slight", "moderate", "extensive", "complete"]

    def test_exceedance_probabilities_at_median(self, four_state_set):
        ep = four_state_set.exceedance_probabilities(0.10)
        assert math.isclose(ep["slight"], 0.5, abs_tol=1e-9)

    def test_damage_state_probabilities_sum_to_one(self, four_state_set):
        for im in [0.05, 0.10, 0.25, 0.5, 1.0]:
            probs = four_state_set.damage_state_probabilities(im)
            assert math.isclose(sum(probs.values()), 1.0, abs_tol=1e-9)

    def test_damage_state_probabilities_none_dominant_at_low_im(self, four_state_set):
        probs = four_state_set.damage_state_probabilities(0.001)
        assert probs["none"] > 0.99

    def test_damage_state_probabilities_complete_dominant_at_high_im(self, four_state_set):
        probs = four_state_set.damage_state_probabilities(10.0)
        assert probs["complete"] > 0.99

    def test_mean_damage_ratio_zero_at_no_im(self, four_state_set):
        mdr = four_state_set.mean_damage_ratio(0.001)
        assert mdr < 0.01

    def test_mean_damage_ratio_in_range(self, four_state_set):
        for im in [0.05, 0.2, 1.0]:
            mdr = four_state_set.mean_damage_ratio(im)
            assert 0.0 <= mdr <= 1.0

    def test_mean_damage_ratio_increases_with_im(self, four_state_set):
        mdr_low = four_state_set.mean_damage_ratio(0.1)
        mdr_high = four_state_set.mean_damage_ratio(0.5)
        assert mdr_high > mdr_low

    def test_median_ordering_enforced(self):
        with pytest.raises(ValueError):
            FragilitySet([
                FragilityCurve("slight",   median=0.20, beta=0.4),
                FragilityCurve("moderate", median=0.10, beta=0.4),  # out of order
            ])

    def test_empty_set_raises(self):
        with pytest.raises(ValueError):
            FragilitySet([])

    def test_repr(self, four_state_set):
        assert "FragilitySet" in repr(four_state_set)
