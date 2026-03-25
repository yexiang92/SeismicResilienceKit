"""Tests for seismicresiliencekit.resilience module."""

import math

import pytest

from seismicresiliencekit.resilience import (
    ResilienceFunction,
    compute_functionality,
    recovery_time_from_damage,
    trapezoid_resilience,
)


class TestResilienceFunction:
    @pytest.fixture
    def linear_rf(self):
        return ResilienceFunction(t_event=0, q_post=0.5, t_recovery=100, recovery_shape="linear")

    def test_pre_event_functionality(self, linear_rf):
        assert linear_rf.functionality(-10) == 1.0

    def test_post_event_drop(self, linear_rf):
        assert linear_rf.functionality(0) == 0.5

    def test_midway_recovery(self, linear_rf):
        assert math.isclose(linear_rf.functionality(50), 0.75, abs_tol=1e-9)

    def test_full_recovery(self, linear_rf):
        assert math.isclose(linear_rf.functionality(100), 1.0, abs_tol=1e-9)

    def test_beyond_recovery_unchanged(self, linear_rf):
        assert math.isclose(linear_rf.functionality(200), 1.0, abs_tol=1e-9)

    def test_functionalities_vectorised(self, linear_rf):
        import numpy as np
        times = np.array([-10, 0, 50, 100, 200])
        result = linear_rf.functionalities(times)
        assert result.shape == (5,)
        assert math.isclose(result[0], 1.0, abs_tol=1e-9)
        assert math.isclose(result[1], 0.5, abs_tol=1e-9)

    def test_resilience_index_perfect(self):
        rf = ResilienceFunction(t_event=0, q_post=1.0, t_recovery=100)
        ri = rf.resilience_index(0, 100)
        assert math.isclose(ri, 1.0, abs_tol=1e-6)

    def test_resilience_index_in_range(self, linear_rf):
        ri = linear_rf.resilience_index(0, 100)
        assert 0.0 <= ri <= 1.0

    def test_loss_of_resilience_positive(self, linear_rf):
        lor = linear_rf.loss_of_resilience(0, 100)
        assert lor > 0

    def test_loss_of_resilience_linear(self):
        rf = ResilienceFunction(t_event=0, q_post=0.5, t_recovery=100, recovery_shape="linear")
        # Triangle: 0.5 * (1 - 0.5) * 100 = 25
        lor = rf.loss_of_resilience(0, 100)
        assert math.isclose(lor, 25.0, rel_tol=1e-4)

    def test_exponential_shape(self):
        rf = ResilienceFunction(t_event=0, q_post=0.5, t_recovery=100, recovery_shape="exponential")
        # Just check it runs and is in range
        q_mid = rf.functionality(50)
        assert 0.5 <= q_mid <= 1.0

    def test_trigonometric_shape(self):
        rf = ResilienceFunction(t_event=0, q_post=0.5, t_recovery=100, recovery_shape="trigonometric")
        q_mid = rf.functionality(50)
        assert math.isclose(q_mid, 0.75, abs_tol=1e-9)

    def test_invalid_q_post_raises(self):
        with pytest.raises(ValueError):
            ResilienceFunction(q_post=1.5)

    def test_invalid_t_recovery_raises(self):
        with pytest.raises(ValueError):
            ResilienceFunction(t_recovery=-10)

    def test_invalid_shape_raises(self):
        with pytest.raises(ValueError):
            ResilienceFunction(recovery_shape="unknown")

    def test_repr(self, linear_rf):
        assert "ResilienceFunction" in repr(linear_rf)


class TestTrapezoidResilience:
    def test_half_functionality(self):
        result = trapezoid_resilience(q_post=0.5, t_recovery=100)
        assert math.isclose(result, 25.0, abs_tol=1e-9)

    def test_zero_damage(self):
        result = trapezoid_resilience(q_post=1.0, t_recovery=100)
        assert math.isclose(result, 0.0, abs_tol=1e-9)

    def test_complete_loss(self):
        result = trapezoid_resilience(q_post=0.0, t_recovery=100)
        assert math.isclose(result, 50.0, abs_tol=1e-9)


class TestComputeFunctionality:
    def test_linear_no_damage(self):
        assert compute_functionality(0.0, "linear") == 1.0

    def test_linear_complete_damage(self):
        assert compute_functionality(1.0, "linear") == 0.0

    def test_linear_partial(self):
        assert math.isclose(compute_functionality(0.4, "linear"), 0.6, abs_tol=1e-9)

    def test_step_below_threshold(self):
        assert compute_functionality(0.4, "step") == 1

    def test_step_above_threshold(self):
        assert compute_functionality(0.6, "step") == 0

    def test_hayashi_no_damage(self):
        assert math.isclose(compute_functionality(0.0, "hayashi"), 1.0, abs_tol=1e-9)

    def test_hayashi_decreases_with_damage(self):
        q1 = compute_functionality(0.2, "hayashi")
        q2 = compute_functionality(0.5, "hayashi")
        assert q1 > q2

    def test_invalid_damage_ratio_raises(self):
        with pytest.raises(ValueError):
            compute_functionality(-0.1)

    def test_invalid_model_raises(self):
        with pytest.raises(ValueError):
            compute_functionality(0.5, "unknown_model")


class TestRecoveryTimeFromDamage:
    def test_linear_half_damage(self):
        rt = recovery_time_from_damage(0.5)
        assert math.isclose(rt, 182.5, abs_tol=1e-9)

    def test_linear_zero_damage(self):
        assert recovery_time_from_damage(0.0) == 0.0

    def test_linear_complete_damage(self):
        assert recovery_time_from_damage(1.0) == 365.0

    def test_power_less_than_linear(self):
        rt_linear = recovery_time_from_damage(0.5, model="linear")
        rt_power = recovery_time_from_damage(0.5, model="power")
        assert rt_power < rt_linear

    def test_invalid_model_raises(self):
        with pytest.raises(ValueError):
            recovery_time_from_damage(0.5, model="cubic")
