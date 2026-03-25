"""
resilience.py
-------------
Seismic resilience metrics and recovery modeling.

Resilience is quantified as the ability of a system to withstand, absorb,
and recover from the impacts of a seismic event.  A common metric is the
*resilience triangle* or the area under the functionality curve over a
recovery period.

Reference: Bruneau et al. (2003). "A Framework to Quantitatively Assess
and Enhance the Seismic Resilience of Communities."
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.integrate import quad


class ResilienceFunction:
    """Time-varying functionality (performance) curve for a system.

    The functionality Q(t) ∈ [0, 1] represents the fraction of the
    pre-earthquake performance level.  After an earthquake at time *t_event*,
    Q drops to a residual level and then recovers to the target level by
    time *t_event + t_recovery*.

    Parameters
    ----------
    t_event : float
        Time of the seismic event (same units as recovery time).
    q_pre : float
        Pre-event functionality level (default 1.0 = 100 %).
    q_post : float
        Immediate post-event functionality (drop due to earthquake damage).
    t_recovery : float
        Total recovery time required to return to *q_target*.
    q_target : float
        Target functionality level after recovery (default = *q_pre*).
    recovery_shape : str
        Shape of the recovery curve: ``"linear"``, ``"exponential"``, or
        ``"trigonometric"`` (default ``"linear"``).

    Examples
    --------
    >>> rf = ResilienceFunction(t_event=0, q_post=0.5, t_recovery=100)
    >>> rf.functionality(0)   # immediately after event
    0.5
    >>> round(rf.functionality(50), 2)  # halfway through recovery
    0.75
    >>> rf.functionality(100)  # fully recovered
    1.0
    """

    SHAPES = ("linear", "exponential", "trigonometric")

    def __init__(
        self,
        t_event: float = 0.0,
        q_pre: float = 1.0,
        q_post: float = 0.5,
        t_recovery: float = 100.0,
        q_target: Optional[float] = None,
        recovery_shape: str = "linear",
    ) -> None:
        if not 0.0 <= q_post <= q_pre <= 1.0:
            raise ValueError("Must satisfy 0 ≤ q_post ≤ q_pre ≤ 1.")
        if t_recovery <= 0:
            raise ValueError("t_recovery must be positive.")
        if recovery_shape not in self.SHAPES:
            raise ValueError(f"recovery_shape must be one of {self.SHAPES}.")
        self.t_event = float(t_event)
        self.q_pre = float(q_pre)
        self.q_post = float(q_post)
        self.t_recovery = float(t_recovery)
        self.q_target = float(q_target if q_target is not None else q_pre)
        self.recovery_shape = recovery_shape

    def functionality(self, t: float) -> float:
        """System functionality Q(t) at time *t*.

        Parameters
        ----------
        t : float
            Time (same units as *t_recovery*).

        Returns
        -------
        float
            Functionality in [0, 1].
        """
        if t < self.t_event:
            return self.q_pre
        tau = t - self.t_event
        if tau >= self.t_recovery:
            return self.q_target
        frac = tau / self.t_recovery  # in [0, 1]
        delta_q = self.q_target - self.q_post
        if self.recovery_shape == "linear":
            return self.q_post + delta_q * frac
        elif self.recovery_shape == "exponential":
            # Convex recovery: slow at first, fast later
            return self.q_post + delta_q * (1.0 - np.exp(-5.0 * frac)) / (1.0 - np.exp(-5.0))
        else:  # trigonometric (S-curve)
            return self.q_post + delta_q * 0.5 * (1.0 - np.cos(np.pi * frac))

    def functionalities(self, times: "array-like") -> np.ndarray:
        """Vectorised :meth:`functionality`."""
        return np.array([self.functionality(t) for t in np.asarray(times, dtype=float)])

    def resilience_index(self, t_start: float, t_end: float) -> float:
        """Normalised resilience index R over the interval [t_start, t_end].

        R = (1 / (T * Q_ref)) ∫_{t_start}^{t_end} Q(t) dt

        where T = t_end - t_start and Q_ref = q_pre.

        Parameters
        ----------
        t_start : float
            Start of assessment window.
        t_end : float
            End of assessment window.

        Returns
        -------
        float
            Resilience index in [0, 1].
        """
        if t_end <= t_start:
            raise ValueError("t_end must be greater than t_start.")
        integral, _ = quad(self.functionality, t_start, t_end)
        t_span = t_end - t_start
        return float(integral / (t_span * self.q_pre)) if self.q_pre > 0 else 0.0

    def loss_of_resilience(self, t_start: float, t_end: float) -> float:
        """Resilience loss (area of the 'resilience triangle').

        ΔR = ∫_{t_event}^{t_event + t_recovery} (Q_target - Q(t)) dt

        Parameters
        ----------
        t_start : float
            Start of integration (typically t_event).
        t_end : float
            End of integration (typically t_event + t_recovery).

        Returns
        -------
        float
            Area of lost functionality (resilience loss).
        """
        if t_end <= t_start:
            raise ValueError("t_end must be greater than t_start.")
        integral, _ = quad(lambda t: self.q_target - self.functionality(t), t_start, t_end)
        return float(max(0.0, integral))

    def __repr__(self) -> str:
        return (
            f"ResilienceFunction(t_event={self.t_event}, q_post={self.q_post}, "
            f"t_recovery={self.t_recovery}, shape={self.recovery_shape!r})"
        )


def trapezoid_resilience(
    q_post: float,
    t_recovery: float,
    q_pre: float = 1.0,
    q_target: Optional[float] = None,
) -> float:
    """Resilience loss using the classic triangular/trapezoidal approximation.

    For a linear recovery from *q_post* to *q_target* over *t_recovery*:

        ΔR = 0.5 * (q_target - q_post) * t_recovery

    Parameters
    ----------
    q_post : float
        Immediate post-event functionality.
    t_recovery : float
        Recovery time.
    q_pre : float
        Pre-event functionality (default 1.0).
    q_target : float, optional
        Target functionality after recovery (default = q_pre).

    Returns
    -------
    float
        Resilience loss (area of the resilience triangle).

    Examples
    --------
    >>> round(trapezoid_resilience(q_post=0.5, t_recovery=100), 1)
    25.0
    """
    if q_target is None:
        q_target = q_pre
    return float(0.5 * (q_target - q_post) * t_recovery)


def compute_functionality(
    damage_ratio: float,
    functionality_model: str = "linear",
) -> float:
    """Map a structural damage ratio to a post-event functionality level.

    Parameters
    ----------
    damage_ratio : float
        Mean damage ratio in [0, 1] (0 = no damage, 1 = complete damage).
    functionality_model : str
        Model to convert damage to functionality:

        - ``"linear"`` : Q = 1 - damage_ratio
        - ``"step"``   : Q = 0 if damage_ratio > 0.5, else 1
        - ``"hayashi"`` : Q = exp(-4.6 * damage_ratio)

    Returns
    -------
    float
        Functionality Q in [0, 1].

    Examples
    --------
    >>> round(compute_functionality(0.4, "linear"), 2)
    0.6
    >>> compute_functionality(0.6, "step")
    0
    >>> round(compute_functionality(0.0, "hayashi"), 4)
    1.0
    """
    if not 0.0 <= damage_ratio <= 1.0:
        raise ValueError("damage_ratio must be in [0, 1].")
    if functionality_model == "linear":
        return float(max(0.0, 1.0 - damage_ratio))
    elif functionality_model == "step":
        return 0 if damage_ratio > 0.5 else 1
    elif functionality_model == "hayashi":
        return float(np.exp(-4.6 * damage_ratio))
    else:
        raise ValueError(
            f"Unknown functionality_model {functionality_model!r}. "
            "Choose from 'linear', 'step', or 'hayashi'."
        )


def recovery_time_from_damage(
    damage_ratio: float,
    max_recovery_days: float = 365.0,
    model: str = "linear",
) -> float:
    """Estimate recovery time from a damage ratio.

    Parameters
    ----------
    damage_ratio : float
        Mean damage ratio in [0, 1].
    max_recovery_days : float
        Maximum recovery time (days) for complete damage (default 365).
    model : str
        ``"linear"`` or ``"power"`` (quadratic scaling).

    Returns
    -------
    float
        Estimated recovery time in days.

    Examples
    --------
    >>> recovery_time_from_damage(0.5)
    182.5
    >>> round(recovery_time_from_damage(0.5, model="power"), 2)
    91.25
    """
    if not 0.0 <= damage_ratio <= 1.0:
        raise ValueError("damage_ratio must be in [0, 1].")
    if model == "linear":
        return float(damage_ratio * max_recovery_days)
    elif model == "power":
        return float(damage_ratio**2 * max_recovery_days)
    else:
        raise ValueError(f"Unknown model {model!r}. Choose 'linear' or 'power'.")
