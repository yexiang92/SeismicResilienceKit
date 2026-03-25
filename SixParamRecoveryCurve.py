# -*- coding: utf-8 -*-
"""
Six-parameter functionality recovery model (Cimellaro et al.)

This script implements the six-parameter post-event functionality recovery model
and reproduces all cases listed in Table 1 (Fig.2a–2h and Fig.3a–3f).

Notes
-----
- This is a faithful re-organization of the original implementation: the
  computational logic is unchanged.
- The parameter `s` is internally mapped to an effective value `s_eff = 1 - s`
  to flip the flex-location trend, consistent with the original code.
"""

import math
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np


class SixParamRecoveryCurve:
    """
    Six-parameter functionality recovery curve after Cimellaro et al.

    Parameters
    ----------
    Qr : float
        Residual functionality immediately after the event (0 <= Qr <= 1).
    delta_i : float
        Idle time δ_i (no recovery yet).
    delta_r : float
        Recovery duration δ_r (active restoration period).
    Qt : float
        Target functionality level at the end of recovery (Qt <= 1).
    A : float, optional
        Amplitude parameter of the sinusoid (controls shape), default 0.0.
    s : float, optional
        Flex-location parameter in (0, 1); s=0.5 -> inflection at mid-time.
    t0 : float, optional
        Time of the extreme event (default 0.0).
    """

    def __init__(
        self,
        Qr: float,
        delta_i: float,
        delta_r: float,
        Qt: float,
        A: float = 0.0,
        s: float = 0.5,
        t0: float = 0.0,
    ) -> None:
        self.Qr = float(Qr)
        self.delta_i = float(delta_i)
        self.delta_r = float(delta_r)
        self.Qt = float(Qt)
        self.A = float(A)

        # External flex-location parameter
        self.s = float(s)

        # Effective flex-location parameter for mapping:
        # key design: use (1 - s) to flip the trend (as in the original code)
        self.s_eff = 1.0 - self.s

        self.t0 = float(t0)

        # Section 3.3: rotation angle α and sinusoid length x_r
        # α = arctan((1 - Qr) / δ_r)
        self.alpha = math.atan((1.0 - self.Qr) / self.delta_r)
        # x_r = δ_r / cos α
        self.x_r = self.delta_r / math.cos(self.alpha)

        # Section 3.4: pre-compute k1, k2 for flex-location mapping using s_eff
        self.k1, self.k2 = self._compute_k1_k2(self.s_eff)

    # ------------------------------------------------------------------
    # Basic numerical utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _bisect(
        func, a: float, b: float, tol: float = 1e-10, maxiter: int = 100
    ) -> float:
        """
        Simple bisection root finder.

        Requirements
        ------------
        func(a) * func(b) < 0.
        """
        fa = func(a)
        fb = func(b)
        if fa * fb > 0:
            raise RuntimeError("Bisection interval does not bracket a root.")
        for _ in range(maxiter):
            m = 0.5 * (a + b)
            fm = func(m)
            if abs(fm) < tol or 0.5 * (b - a) < tol:
                return m
            if fa * fm < 0:
                b, fb = m, fm
            else:
                a, fa = m, fm
        return 0.5 * (a + b)

    # ------------------------------------------------------------------
    # Section 3.4: k1, k2 as a function of s
    # ------------------------------------------------------------------

    def _compute_k1_k2(self, s: float) -> Tuple[float, float]:
        """
        Compute k1 and k2 for the flex-location mapping z(ŷ).

        Mapping type
        -----------
        - If s < 0.5:  z(ŷ) = k1 * tan(k2 ŷ)
        - If s > 0.5:  z(ŷ) = k1 * atan(k2 ŷ)

        Constraints
        -----------
        Enforce:
            z(1)   = 1
            z(0.5) = s

        Special case
        ------------
        If s = 0.5, use identity mapping z(ŷ)=ŷ (no k1,k2 needed).
        """
        if abs(s - 0.5) < 1e-8:
            return None, None  # identity mapping

        if s < 0.5:
            # Earlier inflection
            def f(k2: float) -> float:
                return math.tan(0.5 * k2) / math.tan(k2) - s

            a = 1e-6
            b = math.pi / 2.0 - 1e-6
            k2 = self._bisect(f, a, b)
            k1 = 1.0 / math.tan(k2)
            return k1, k2
        else:
            # Later inflection
            def f(k2: float) -> float:
                return math.atan(0.5 * k2) / math.atan(k2) - s

            a = 1e-6
            b = 1e4
            k2 = self._bisect(f, a, b)
            k1 = 1.0 / math.atan(k2)
            return k1, k2

    # ------------------------------------------------------------------
    # Sections 3.2–3.3: base sinusoid & rotation
    # ------------------------------------------------------------------

    def _y_from_x(self, x: float) -> float:
        """Given parameter x, compute rotated horizontal coordinate y(x)."""
        # f1(x) = -A sin(2π x / x_r)
        f1 = -self.A * math.sin(2.0 * math.pi * x / self.x_r)
        # y = x cos α - f1 sin α
        y = x * math.cos(self.alpha) - f1 * math.sin(self.alpha)
        return y

    def _f2_from_y(self, y_target: float) -> float:
        """
        Given rotated horizontal coordinate y, invert for x and compute f2(y).

        Uses bisection to solve y(x) = y_target, then:
            f2 = x sin α + f1 cos α
        """
        a, b = 0.0, self.x_r
        for _ in range(80):
            m = 0.5 * (a + b)
            y_m = self._y_from_x(m)
            if y_m < y_target:
                a = m
            else:
                b = m
        x = 0.5 * (a + b)
        f1 = -self.A * math.sin(2.0 * math.pi * x / self.x_r)
        f2 = x * math.sin(self.alpha) + f1 * math.cos(self.alpha)
        return f2

    # ------------------------------------------------------------------
    # Section 3.4: flex-location mapping z(ŷ)
    # ------------------------------------------------------------------

    def _z_of_yhat(self, y_hat: float) -> float:
        """
        Map normalized time y_hat ∈ [0, 1] (i.e., y/δ_r) to z_hat ∈ [0, 1].

        Notes
        -----
        Uses `s_eff` (not `s`) exactly as in the original implementation.
        """
        if y_hat <= 0.0:
            return 0.0
        if y_hat >= 1.0:
            return 1.0

        s = self.s_eff

        if abs(s - 0.5) < 1e-8:
            return y_hat

        if s < 0.5:
            return self.k1 * math.tan(self.k2 * y_hat)
        else:
            return self.k1 * math.atan(self.k2 * y_hat)

    # ------------------------------------------------------------------
    # Assemble the base recovery curve increment (f3/f4 in the paper)
    # ------------------------------------------------------------------

    def _increment_base(self, t: float) -> float:
        """
        Base recovery increment over t ∈ [0, δ_r].

        Returns an increment starting from 0 and ending at approximately (1 - Qr).
        """
        if t <= 0.0:
            return 0.0
        if t >= self.delta_r:
            return 1.0 - self.Qr

        # y = t on the recovery axis
        y = t
        y_hat = y / self.delta_r

        # Apply flex-location mapping
        z_hat = self._z_of_yhat(y_hat)
        z = z_hat * self.delta_r

        # Evaluate the rotated sinusoid at z
        f2z = self._f2_from_y(z)
        return f2z

    # ------------------------------------------------------------------
    # Public interface: Q(t)
    # ------------------------------------------------------------------

    def Q(self, t: Iterable[float]):
        """
        Evaluate functionality Q(t) for scalar or array-like time input.

        Conventions
        -----------
        - For t < t0: Q(t) = 1 (pre-event).
        - For t0 < t <= t0 + δ_i: Q(t) = Qr (idle phase).
        - For t0 + δ_i < t < t0 + δ_i + δ_r: recovery curve.
        - For t >= t0 + δ_i + δ_r: Q(t) = Qt.

        Returns
        -------
        Q_vals : np.ndarray
            Functionality values (including an enforced instantaneous drop at t0).
        t_arr : np.ndarray
            Time array aligned with Q_vals.
        """
        t_arr = np.asarray(t, dtype=float)

        # Insert t0 if not already present
        idx = np.searchsorted(t_arr, self.t0)
        if not (idx < len(t_arr) and np.isclose(t_arr[idx], self.t0)):
            t_arr = np.insert(t_arr, idx, self.t0)

        Q_vals = np.empty_like(t_arr)

        # Pre-event
        before = t_arr <= self.t0
        Q_vals[before] = 1.0

        # Idle phase
        idle = (t_arr > self.t0) & (t_arr <= self.t0 + self.delta_i)
        Q_vals[idle] = self.Qr

        # Recovery phase: Qr + scaled increment
        recov = (t_arr > self.t0 + self.delta_i) & (
            t_arr < self.t0 + self.delta_i + self.delta_r
        )
        tau = t_arr[recov] - self.t0 - self.delta_i  # 0 ~ δ_r

        # f3(z) -> f4(z) = f3 * (Qt-Qr)/(1-Qr)
        inc = np.array([self._increment_base(tt) for tt in tau])
        f4 = inc * (self.Qt - self.Qr) / (1.0 - self.Qr)
        Q_vals[recov] = self.Qr + f4

        # Post-recovery
        after = t_arr >= self.t0 + self.delta_i + self.delta_r
        Q_vals[after] = self.Qt

        # Enforce instantaneous drop at t0: insert an additional (t0, Qr) point
        idx_t0 = np.argmin(np.abs(t_arr - self.t0))
        Q_vals = np.insert(Q_vals, idx_t0 + 1, self.Qr)
        t_arr = np.insert(t_arr, idx_t0 + 1, self.t0)

        return Q_vals, t_arr

    __call__ = Q  # convenient callable interface: curve(t)


# ----------------------------------------------------------------------
# Plot Fig.2a–2h cases
# ----------------------------------------------------------------------
def plot_fig2_cases(t: np.ndarray) -> None:
    """
    Plot the eight Fig.2a–2h cases (Table 1) as individual subplots.
    """
    cases = [
        (
            "Fig.2a\nLinear-type",
            dict(Qr=0.2, delta_i=2.0, delta_r=4.0, Qt=0.8, A=0.0, s=0.5),
        ),
        (
            "Fig.2b\nLinear-type without idle",
            dict(Qr=0.2, delta_i=0.0, delta_r=6.0, Qt=0.8, A=0.0, s=0.5),
        ),
        (
            "Fig.2c\nSinusoidal-type",
            dict(Qr=0.2, delta_i=2.0, delta_r=4.0, Qt=0.8, A=0.1, s=0.5),
        ),
        (
            "Fig.2d\nStepwise-type",
            dict(Qr=0.2, delta_i=2.0, delta_r=4.0, Qt=0.8, A=-0.1, s=0.5),
        ),
        (
            "Fig.2e\nPositive-exp. type",
            dict(Qr=0.2, delta_i=2.0, delta_r=4.0, Qt=0.8, A=0.1, s=0.95),
        ),
        (
            "Fig.2f\nNegative-exp. type",
            dict(Qr=0.2, delta_i=2.0, delta_r=4.0, Qt=0.8, A=0.1, s=0.05),
        ),
        (
            "Fig.2g\nStepwise-to-lin.",
            dict(Qr=0.2, delta_i=2.0, delta_r=4.0, Qt=0.8, A=-0.1, s=0.25),
        ),
        (
            "Fig.2h\nLinear-to-stepwise",
            dict(Qr=0.2, delta_i=2.0, delta_r=4.0, Qt=0.8, A=-0.1, s=0.75),
        ),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(12, 6), sharex=True, sharey=True)
    axes = axes.ravel()

    for ax, (title, params) in zip(axes, cases):
        curve = SixParamRecoveryCurve(**params)
        Q_vals, t_vals = curve.Q(t)
        ax.plot(t_vals, Q_vals)
        ax.set_title(title, fontsize=9)
        ax.grid(True, linestyle="--", linewidth=0.5)
        ax.set_ylim(0.0, 1.05)

    for ax in axes[4:]:
        ax.set_xlabel("Time (months)")
    for ax in [axes[0], axes[4]]:
        ax.set_ylabel("Functionality Q(t)")

    fig.suptitle("Six-parameter recovery model – Fig.2a–2h cases", fontsize=12)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    plt.show()


# ----------------------------------------------------------------------
# Plot Fig.3a–3f cases (parameter variations)
# ----------------------------------------------------------------------
def plot_fig3_cases(t: np.ndarray) -> None:
    """
    Plot Fig.3a–3f parameter variation cases (Table 1).

    Notes
    -----
    The original table defines parameter ranges. Here we select representative
    values within each range, consistent with the original script.
    """
    base = dict(Qr=0.2, delta_i=2.0, delta_r=4.0, Qt=0.8, A=0.1, s=0.5)

    fig, axes = plt.subplots(3, 2, figsize=(10, 9), sharex=True, sharey=True)
    axes = axes.ravel()

    # Fig.3a: variation of Qr
    ax = axes[0]
    Qr_vals = [0.0, 0.1, 0.2, 0.3, 0.4]
    for Qr in Qr_vals:
        params = base.copy()
        params["Qr"] = Qr
        curve = SixParamRecoveryCurve(**params)
        Q_vals, t_vals = curve.Q(t)
        ax.plot(t_vals, Q_vals, label=f"Qr={Qr:.1f}")
    ax.set_title("Fig.3a – Variation of Qr")
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.legend(fontsize=8)

    # Fig.3b: variation of δ_i
    ax = axes[1]
    di_vals = [0.0, 1.0, 2.0, 3.0, 4.0]
    for di in di_vals:
        params = base.copy()
        params["delta_i"] = di
        curve = SixParamRecoveryCurve(**params)
        Q_vals, t_vals = curve.Q(t)
        ax.plot(t_vals, Q_vals, label=f"δi={di:.0f}")
    ax.set_title("Fig.3b – Variation of δi")
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.legend(fontsize=8)

    # Fig.3c: variation of δ_r
    ax = axes[2]
    dr_vals = [2.0, 3.0, 4.0, 5.0, 6.0]
    for dr in dr_vals:
        params = base.copy()
        params["delta_r"] = dr
        curve = SixParamRecoveryCurve(**params)
        Q_vals, t_vals = curve.Q(t)
        ax.plot(t_vals, Q_vals, label=f"δr={dr:.0f}")
    ax.set_title("Fig.3c – Variation of δr")
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.legend(fontsize=8)

    # Fig.3d: variation of Qt
    ax = axes[3]
    Qt_vals = [0.6, 0.7, 0.8, 0.9, 1.0]
    for Qt in Qt_vals:
        params = base.copy()
        params["Qt"] = Qt
        curve = SixParamRecoveryCurve(**params)
        Q_vals, t_vals = curve.Q(t)
        ax.plot(t_vals, Q_vals, label=f"Qt={Qt:.1f}")
    ax.set_title("Fig.3d – Variation of Qt")
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.legend(fontsize=8)

    # Fig.3e: variation of A
    ax = axes[4]
    A_vals = [-0.1, -0.05, 0.0, 0.05, 0.1]
    for A in A_vals:
        params = base.copy()
        params["A"] = A
        curve = SixParamRecoveryCurve(**params)
        Q_vals, t_vals = curve.Q(t)
        ax.plot(t_vals, Q_vals, label=f"A={A:+.2f}")
    ax.set_title("Fig.3e – Variation of A")
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.legend(fontsize=8)

    # Fig.3f: variation of s
    ax = axes[5]
    s_vals = [0.1, 0.3, 0.5, 0.7, 0.9]
    for s in s_vals:
        params = base.copy()
        params["s"] = s
        curve = SixParamRecoveryCurve(**params)
        Q_vals, t_vals = curve.Q(t)
        ax.plot(t_vals, Q_vals, label=f"s={s:.1f}")
    ax.set_title("Fig.3f – Variation of s")
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.legend(fontsize=8)

    # Shared axis labels
    for ax in axes[4:]:
        ax.set_xlabel("Time (months)")
    for ax in [axes[0], axes[2], axes[4]]:
        ax.set_ylabel("Functionality Q(t)")

    fig.suptitle(
        "Six-parameter recovery model – Fig.3a–3f parameter variations", fontsize=12
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# ----------------------------------------------------------------------
# Main: reproduce all cases in Table 1
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Time axis (-2 ~ 12 months) aligned with the paper figures
    t = np.linspace(-2, 12.0, 101)

    # Fig.2a–2h
    plot_fig2_cases(t)

    # Fig.3a–3f
    plot_fig3_cases(t)

    print("Done.")
