"""Tests for seismicresiliencekit.hazard module."""

import math

import numpy as np
import pytest

from seismicresiliencekit.hazard import HazardCurve, compute_mean_annual_rate, disaggregate


@pytest.fixture
def simple_hazard_curve():
    im = np.array([0.05, 0.10, 0.20, 0.40, 0.80])
    rates = np.array([0.10, 0.04, 0.01, 0.002, 0.0002])
    return HazardCurve(im, rates)


class TestHazardCurve:
    def test_annual_rate_at_node(self, simple_hazard_curve):
        assert math.isclose(simple_hazard_curve.annual_rate(0.10), 0.04, rel_tol=1e-6)

    def test_annual_rate_at_node_second(self, simple_hazard_curve):
        assert math.isclose(simple_hazard_curve.annual_rate(0.20), 0.01, rel_tol=1e-6)

    def test_return_period(self, simple_hazard_curve):
        assert math.isclose(simple_hazard_curve.return_period(0.10), 25.0, rel_tol=1e-6)

    def test_im_for_return_period(self, simple_hazard_curve):
        im = simple_hazard_curve.im_for_return_period(25.0)
        assert math.isclose(im, 0.10, rel_tol=1e-3)

    def test_probability_of_exceedance(self, simple_hazard_curve):
        p = simple_hazard_curve.probability_of_exceedance(0.10, exposure_years=50)
        # Poisson: 1 - exp(-0.04 * 50) = 1 - exp(-2) ≈ 0.8647
        expected = 1.0 - math.exp(-0.04 * 50)
        assert math.isclose(p, expected, rel_tol=1e-6)

    def test_annual_rates_array(self, simple_hazard_curve):
        im_arr = np.array([0.10, 0.20, 0.40])
        rates = simple_hazard_curve.annual_rates_array(im_arr)
        assert rates.shape == (3,)
        assert math.isclose(rates[0], 0.04, rel_tol=1e-6)

    def test_invalid_non_increasing_im(self):
        with pytest.raises(ValueError):
            HazardCurve([0.2, 0.1], [0.05, 0.01])

    def test_invalid_negative_im(self):
        with pytest.raises(ValueError):
            HazardCurve([-0.1, 0.1], [0.05, 0.01])

    def test_invalid_negative_rate(self):
        with pytest.raises(ValueError):
            HazardCurve([0.1, 0.2], [-0.01, 0.01])

    def test_repr(self, simple_hazard_curve):
        assert "HazardCurve" in repr(simple_hazard_curve)

    def test_annual_loss_rate(self, simple_hazard_curve):
        im_pts = np.array([0.05, 0.10, 0.20, 0.40, 0.80])
        loss_pts = np.array([0.01, 0.05, 0.15, 0.40, 0.80])
        eal = simple_hazard_curve.annual_loss_rate(loss_pts, im_pts)
        assert eal >= 0.0


class TestComputeMeanAnnualRate:
    def _simple_gmpe(self, mag, dist):
        """Toy GMPE: median = 0.01 * exp(1.5*M - 0.005*R), sigma = 0.6."""
        median = 0.01 * math.exp(1.5 * mag - 0.005 * dist)
        return median, 0.6

    def test_returns_array(self):
        sources = [
            {
                "activity_rate": 0.1,
                "magnitude_pmf": [(6.0, 0.5), (7.0, 0.5)],
                "distance": 20.0,
            }
        ]
        im = np.array([0.05, 0.10, 0.20, 0.40])
        rates = compute_mean_annual_rate(sources, im, self._simple_gmpe)
        assert rates.shape == (4,)

    def test_rates_decrease_with_im(self):
        sources = [
            {
                "activity_rate": 0.2,
                "magnitude_pmf": [(6.5, 1.0)],
                "distance": 15.0,
            }
        ]
        im = np.array([0.05, 0.10, 0.20, 0.40, 0.80])
        rates = compute_mean_annual_rate(sources, im, self._simple_gmpe)
        assert np.all(np.diff(rates) <= 0)


class TestDisaggregate:
    def _simple_gmpe(self, mag, dist):
        median = 0.01 * math.exp(1.5 * mag - 0.005 * dist)
        return median, 0.6

    def test_contributions_sum_to_one(self):
        im = np.array([0.05, 0.10, 0.20, 0.40, 0.80])
        rates = np.array([0.10, 0.04, 0.01, 0.002, 0.0002])
        hc = HazardCurve(im, rates)
        sources = [
            {
                "activity_rate": 0.1,
                "magnitude_pmf": [(5.5, 0.3), (6.5, 0.5), (7.0, 0.2)],
                "distance": 15.0,
            },
            {
                "activity_rate": 0.05,
                "magnitude_pmf": [(6.0, 1.0)],
                "distance": 50.0,
            },
        ]
        result = disaggregate(hc, target_im=0.20, sources=sources, attenuation_fn=self._simple_gmpe)
        total = sum(r["contribution"] for r in result)
        assert math.isclose(total, 1.0, abs_tol=1e-9)

    def test_returns_list_of_dicts(self):
        im = np.array([0.05, 0.10, 0.20, 0.40])
        rates = np.array([0.05, 0.02, 0.005, 0.001])
        hc = HazardCurve(im, rates)
        sources = [{"activity_rate": 0.1, "magnitude_pmf": [(6.0, 1.0)], "distance": 20.0}]
        result = disaggregate(hc, 0.10, sources, self._simple_gmpe)
        assert isinstance(result, list)
        assert "magnitude" in result[0]
        assert "distance" in result[0]
        assert "contribution" in result[0]
