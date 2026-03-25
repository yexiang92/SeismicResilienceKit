"""
ground_motion.py
----------------
Ground motion intensity measure (IM) calculation utilities.

Provides functions to compute common IMs from acceleration time-series:
- Peak ground acceleration (PGA)
- Spectral acceleration (Sa) at a given period
- Arias intensity (Ia)
- Significant duration (D5-75, D5-95)
- Cumulative absolute velocity (CAV)
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.integrate import cumulative_trapezoid


def peak_ground_acceleration(acceleration: "array-like") -> float:
    """Peak ground acceleration (PGA).

    Parameters
    ----------
    acceleration : array-like
        Acceleration time series in units of *g* or m/s².

    Returns
    -------
    float
        Maximum absolute acceleration value (same units as input).

    Examples
    --------
    >>> import numpy as np
    >>> t = np.linspace(0, 10, 1000)
    >>> acc = 0.3 * np.sin(2 * np.pi * t)
    >>> round(peak_ground_acceleration(acc), 4)
    0.3
    """
    acc = np.asarray(acceleration, dtype=float)
    return float(np.max(np.abs(acc)))


def spectral_acceleration(
    acceleration: "array-like",
    dt: float,
    period: float,
    damping_ratio: float = 0.05,
) -> float:
    """Spectral acceleration Sa(T, ξ) via linear elastic response spectrum.

    Uses Newmark's constant average acceleration method (β = 0.25, γ = 0.5)
    to integrate the equation of motion of an SDOF oscillator.

    Parameters
    ----------
    acceleration : array-like
        Ground acceleration time series (same units as desired Sa output).
    dt : float
        Time step of the record (seconds).
    period : float
        Oscillator natural period (seconds, > 0).
    damping_ratio : float
        Viscous damping ratio (default 0.05 = 5 %).

    Returns
    -------
    float
        Spectral acceleration Sa(T, ξ) – maximum absolute total acceleration
        (same units as *acceleration*).

    Examples
    --------
    >>> import numpy as np
    >>> t = np.linspace(0, 20, 2001)
    >>> dt = t[1] - t[0]
    >>> acc = np.zeros_like(t)
    >>> acc[100] = 1.0   # impulse
    >>> sa = spectral_acceleration(acc, dt, period=1.0)
    >>> sa > 0
    True
    """
    acc = np.asarray(acceleration, dtype=float)
    if period <= 0:
        raise ValueError("period must be positive.")
    if dt <= 0:
        raise ValueError("dt must be positive.")
    if not 0.0 <= damping_ratio < 1.0:
        raise ValueError("damping_ratio must be in [0, 1).")

    omega = 2.0 * np.pi / period
    xi = damping_ratio
    omega_d = omega * np.sqrt(1.0 - xi**2)

    # Newmark constants (constant average acceleration: β=1/4, γ=1/2)
    beta_nm = 0.25
    gamma_nm = 0.5

    n = len(acc)
    u = np.zeros(n)   # relative displacement
    v = np.zeros(n)   # relative velocity
    a = np.zeros(n)   # relative acceleration

    # Initial acceleration from equation of motion
    a[0] = -acc[0] - 2.0 * xi * omega * v[0] - omega**2 * u[0]

    # Newmark effective stiffness (constant throughout for linear system)
    # k_hat = k + gamma/(beta*dt)*c + 1/(beta*dt^2)*m  (unit mass: m=1, k=omega^2, c=2*xi*omega)
    k_hat = (
        omega**2
        + gamma_nm / (beta_nm * dt) * 2.0 * xi * omega
        + 1.0 / (beta_nm * dt**2)
    )

    for i in range(n - 1):
        # Effective force at step i+1 (p = -m*ag for unit mass)
        # p_hat = p_{i+1} + m*[u_i/(beta*dt^2) + v_i/(beta*dt) + (1/(2*beta)-1)*a_i]
        #                  + c*[gamma*u_i/(beta*dt) + (gamma/beta - 1)*v_i + dt*(gamma/(2*beta)-1)*a_i]
        p_hat = (
            -acc[i + 1]
            + (1.0 / (beta_nm * dt**2)) * u[i]
            + (1.0 / (beta_nm * dt)) * v[i]
            + (1.0 / (2.0 * beta_nm) - 1.0) * a[i]
            + 2.0 * xi * omega * (
                gamma_nm / (beta_nm * dt) * u[i]
                + (gamma_nm / beta_nm - 1.0) * v[i]
                + dt * (gamma_nm / (2.0 * beta_nm) - 1.0) * a[i]
            )
        )
        u[i + 1] = p_hat / k_hat
        v[i + 1] = (
            gamma_nm / (beta_nm * dt) * (u[i + 1] - u[i])
            + (1.0 - gamma_nm / beta_nm) * v[i]
            + dt * (1.0 - gamma_nm / (2.0 * beta_nm)) * a[i]
        )
        a[i + 1] = (
            1.0 / (beta_nm * dt**2) * (u[i + 1] - u[i])
            - 1.0 / (beta_nm * dt) * v[i]
            - (1.0 / (2.0 * beta_nm) - 1.0) * a[i]
        )

    # Pseudo-spectral acceleration = omega^2 * max(|u|)
    # Total acceleration = |a_ground + a_relative|
    total_acc = np.abs(acc + a)
    return float(np.max(total_acc))


def arias_intensity(
    acceleration: "array-like",
    dt: float,
    g: float = 9.81,
) -> float:
    """Arias intensity (Ia) of a ground motion record.

    .. math::

        I_a = \\frac{\\pi}{2g} \\int_0^{T_d} a(t)^2 \\, dt

    Parameters
    ----------
    acceleration : array-like
        Acceleration time series in m/s² (or g if *g* = 1).
    dt : float
        Time step (seconds).
    g : float
        Acceleration due to gravity in same units as *acceleration*
        (default 9.81 m/s²; use 1.0 for records already in *g*).

    Returns
    -------
    float
        Arias intensity in m/s.

    Examples
    --------
    >>> import numpy as np
    >>> t = np.linspace(0, 1, 1001)
    >>> dt = t[1] - t[0]
    >>> acc = np.ones_like(t) * 9.81   # 1g constant (unrealistic but simple)
    >>> ia = arias_intensity(acc, dt)
    >>> round(ia, 2)
    15.41
    """
    acc = np.asarray(acceleration, dtype=float)
    if dt <= 0:
        raise ValueError("dt must be positive.")
    return float(np.pi / (2.0 * g) * np.trapezoid(acc**2, dx=dt))


def significant_duration(
    acceleration: "array-like",
    dt: float,
    pct_start: float = 5.0,
    pct_end: float = 75.0,
    g: float = 9.81,
) -> float:
    """Significant duration of a ground motion record.

    Computes the time interval during which the Arias intensity increases
    from *pct_start* % to *pct_end* % of its total value (default: D5-75).

    Parameters
    ----------
    acceleration : array-like
        Acceleration time series.
    dt : float
        Time step (seconds).
    pct_start : float
        Lower Arias intensity threshold in percent (default 5).
    pct_end : float
        Upper Arias intensity threshold in percent (default 75).
    g : float
        Gravitational acceleration (default 9.81 m/s²).

    Returns
    -------
    float
        Significant duration in seconds.

    Examples
    --------
    >>> import numpy as np
    >>> t = np.linspace(0, 20, 2001)
    >>> dt = t[1] - t[0]
    >>> acc = np.sin(2 * np.pi * t / 2) * 0.3 * 9.81
    >>> d = significant_duration(acc, dt, pct_start=5, pct_end=95)
    >>> d > 0
    True
    """
    acc = np.asarray(acceleration, dtype=float)
    if dt <= 0:
        raise ValueError("dt must be positive.")
    if not 0.0 <= pct_start < pct_end <= 100.0:
        raise ValueError("pct_start must be < pct_end, both in [0, 100].")

    husid = cumulative_trapezoid(acc**2, dx=dt, initial=0.0)
    total = husid[-1]
    if total == 0.0:
        return 0.0

    t = np.arange(len(acc)) * dt
    t_start = t[husid >= total * pct_start / 100.0][0]
    t_end = t[husid >= total * pct_end / 100.0][0]
    return float(t_end - t_start)


def cumulative_absolute_velocity(
    acceleration: "array-like",
    dt: float,
    threshold: float = 0.0,
) -> float:
    """Cumulative absolute velocity (CAV).

    .. math::

        \\text{CAV} = \\int_0^{T_d} |a(t)| \\, dt

    An optional acceleration *threshold* (in the same units as *acceleration*)
    can be used to compute the standardised CAV (CAV_std) which only
    integrates windows where the PGA exceeds the threshold.

    Parameters
    ----------
    acceleration : array-like
        Acceleration time series in m/s² (or g).
    dt : float
        Time step (seconds).
    threshold : float
        PGA threshold for windowed CAV (default 0 = full CAV, no windowing).

    Returns
    -------
    float
        CAV in m/s (or g·s if input is in *g*).

    Examples
    --------
    >>> import numpy as np
    >>> t = np.linspace(0, 10, 1001)
    >>> dt = t[1] - t[0]
    >>> acc = np.ones_like(t) * 2.0
    >>> round(cumulative_absolute_velocity(acc, dt), 1)
    20.0
    """
    acc = np.asarray(acceleration, dtype=float)
    if dt <= 0:
        raise ValueError("dt must be positive.")

    if threshold <= 0:
        return float(np.trapezoid(np.abs(acc), dx=dt))

    # Windowed CAV: integrate 1-second windows where abs(PGA) > threshold
    window_samples = max(1, int(round(1.0 / dt)))
    n = len(acc)
    cav = 0.0
    for start in range(0, n, window_samples):
        window = acc[start : start + window_samples]
        if np.max(np.abs(window)) >= threshold:
            cav += np.trapezoid(np.abs(window), dx=dt)
    return float(cav)
