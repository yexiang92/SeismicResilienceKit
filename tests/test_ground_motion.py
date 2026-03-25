"""Tests for seismicresiliencekit.ground_motion module."""

import math

import numpy as np
import pytest

from seismicresiliencekit.ground_motion import (
    arias_intensity,
    cumulative_absolute_velocity,
    peak_ground_acceleration,
    significant_duration,
    spectral_acceleration,
)


class TestPeakGroundAcceleration:
    def test_simple_sine(self):
        t = np.linspace(0, 10, 1000)
        acc = 0.3 * np.sin(2 * np.pi * t)
        assert math.isclose(peak_ground_acceleration(acc), 0.3, rel_tol=1e-3)

    def test_all_zeros(self):
        acc = np.zeros(100)
        assert peak_ground_acceleration(acc) == 0.0

    def test_negative_values(self):
        acc = np.array([-0.5, 0.3, -0.2])
        assert peak_ground_acceleration(acc) == 0.5

    def test_single_value(self):
        assert peak_ground_acceleration([0.25]) == 0.25


class TestSpectralAcceleration:
    @pytest.fixture
    def harmonic_record(self):
        """1 Hz harmonic motion, 20 s at 0.01 s step."""
        dt = 0.01
        t = np.arange(0, 20, dt)
        acc = 0.5 * np.sin(2 * np.pi * 1.0 * t)
        return acc, dt

    def test_returns_positive(self, harmonic_record):
        acc, dt = harmonic_record
        sa = spectral_acceleration(acc, dt, period=1.0)
        assert sa > 0

    def test_resonant_amplification(self, harmonic_record):
        """Sa at resonant period should exceed PGA (for low damping)."""
        acc, dt = harmonic_record
        pga = peak_ground_acceleration(acc)
        sa = spectral_acceleration(acc, dt, period=1.0, damping_ratio=0.05)
        assert sa >= pga

    def test_invalid_period_raises(self, harmonic_record):
        acc, dt = harmonic_record
        with pytest.raises(ValueError):
            spectral_acceleration(acc, dt, period=0.0)

    def test_invalid_dt_raises(self, harmonic_record):
        acc, dt = harmonic_record
        with pytest.raises(ValueError):
            spectral_acceleration(acc, dt=-0.01, period=1.0)

    def test_invalid_damping_raises(self, harmonic_record):
        acc, dt = harmonic_record
        with pytest.raises(ValueError):
            spectral_acceleration(acc, dt, period=1.0, damping_ratio=1.5)

    def test_short_period_sa_closer_to_pga(self, harmonic_record):
        """Sa at very short period (rigid oscillator) approaches PGA."""
        acc, dt = harmonic_record
        pga = peak_ground_acceleration(acc)
        sa_short = spectral_acceleration(acc, dt, period=0.05)
        # For very short periods, Sa ≈ PGA (within reasonable tolerance)
        assert abs(sa_short - pga) / pga < 0.5


class TestAriasIntensity:
    def test_constant_acceleration(self):
        dt = 0.001
        t = np.arange(0, 1.0, dt)
        acc = np.ones_like(t) * 9.81  # 1 g
        ia = arias_intensity(acc, dt, g=9.81)
        # π/2 * 1 * (1 g)^2 * T = π/2 * 9.81^2 / 9.81 * 1 = π/2 * 9.81 ≈ 15.41
        expected = math.pi / 2.0 * 9.81**2 / 9.81 * 1.0
        assert math.isclose(ia, expected, rel_tol=0.01)

    def test_zero_motion_gives_zero(self):
        acc = np.zeros(1000)
        assert arias_intensity(acc, 0.01) == 0.0

    def test_invalid_dt_raises(self):
        with pytest.raises(ValueError):
            arias_intensity(np.ones(10), dt=0.0)

    def test_positive_result(self):
        t = np.linspace(0, 10, 1001)
        dt = t[1] - t[0]
        acc = np.sin(2 * np.pi * t)
        ia = arias_intensity(acc, dt)
        assert ia > 0


class TestSignificantDuration:
    def test_sine_wave_duration(self):
        dt = 0.01
        t = np.arange(0, 20, dt)
        acc = np.sin(2 * np.pi * t / 2) * 0.3
        d = significant_duration(acc, dt, pct_start=5, pct_end=95)
        assert d > 0
        assert d <= 20.0

    def test_zero_motion_gives_zero(self):
        acc = np.zeros(1000)
        assert significant_duration(acc, 0.01) == 0.0

    def test_invalid_pct_raises(self):
        with pytest.raises(ValueError):
            significant_duration(np.ones(100), 0.01, pct_start=80, pct_end=20)

    def test_d5_95_longer_than_d5_75(self):
        dt = 0.01
        t = np.arange(0, 30, dt)
        acc = np.sin(2 * np.pi * t / 3) * 0.5
        d_5_75 = significant_duration(acc, dt, pct_start=5, pct_end=75)
        d_5_95 = significant_duration(acc, dt, pct_start=5, pct_end=95)
        assert d_5_95 >= d_5_75


class TestCumulativeAbsoluteVelocity:
    def test_constant_motion(self):
        dt = 0.01
        t = np.arange(0, 10, dt)
        acc = np.ones_like(t) * 2.0
        cav = cumulative_absolute_velocity(acc, dt)
        assert math.isclose(cav, 2.0 * 10.0, rel_tol=0.01)

    def test_zero_motion(self):
        acc = np.zeros(1000)
        assert cumulative_absolute_velocity(acc, 0.01) == 0.0

    def test_windowed_cav_less_or_equal_full(self):
        dt = 0.01
        t = np.arange(0, 20, dt)
        acc = np.sin(2 * np.pi * t) * 0.5
        cav_full = cumulative_absolute_velocity(acc, dt, threshold=0.0)
        cav_std = cumulative_absolute_velocity(acc, dt, threshold=0.3)
        assert cav_std <= cav_full

    def test_invalid_dt_raises(self):
        with pytest.raises(ValueError):
            cumulative_absolute_velocity(np.ones(10), dt=0.0)
