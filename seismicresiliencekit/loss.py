"""
loss.py
-------
Loss estimation from damage states and fragility/vulnerability functions.

Provides:
- LossFunction: maps intensity measure to expected loss ratio
- compute_expected_loss: expected loss given IM for a building portfolio
- compute_loss_curve: annual loss curve by convolving with a hazard curve
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.interpolate import interp1d

from .fragility import FragilitySet
from .hazard import HazardCurve


class LossFunction:
    """Maps an intensity measure to an expected loss ratio.

    Two construction modes are supported:

    1. **Direct** – supply paired ``(im_values, loss_ratios)`` arrays.
    2. **Fragility-based** – supply a :class:`~seismicresiliencekit.fragility.FragilitySet`
       and per-state loss ratios; the vulnerability function is computed
       analytically.

    Parameters
    ----------
    im_values : array-like
        IM levels (strictly increasing, > 0).
    loss_ratios : array-like
        Expected loss ratio at each IM level (values in [0, 1]).

    Examples
    --------
    >>> lf = LossFunction([0.05, 0.10, 0.20, 0.40, 0.80],
    ...                   [0.01, 0.05, 0.15, 0.40, 0.80])
    >>> round(lf.loss_ratio(0.10), 2)
    0.05
    """

    def __init__(
        self,
        im_values: "array-like",
        loss_ratios: "array-like",
    ) -> None:
        im = np.asarray(im_values, dtype=float)
        lr = np.asarray(loss_ratios, dtype=float)
        if im.ndim != 1 or lr.ndim != 1:
            raise ValueError("im_values and loss_ratios must be 1-D.")
        if len(im) != len(lr):
            raise ValueError("im_values and loss_ratios must have the same length.")
        if not np.all(np.diff(im) > 0):
            raise ValueError("im_values must be strictly increasing.")
        if np.any(im <= 0):
            raise ValueError("im_values must be positive.")
        if np.any(lr < 0) or np.any(lr > 1):
            raise ValueError("loss_ratios must be in [0, 1].")
        self.im_values = im
        self.loss_ratios = lr
        self._interp = interp1d(
            im,
            lr,
            kind="linear",
            bounds_error=False,
            fill_value=(lr[0], lr[-1]),
        )

    @classmethod
    def from_fragility_set(
        cls,
        fragility_set: FragilitySet,
        state_loss_ratios: Dict[str, float],
        im_values: Optional["array-like"] = None,
    ) -> "LossFunction":
        """Construct a vulnerability function from a fragility set.

        Parameters
        ----------
        fragility_set : FragilitySet
            Ordered set of fragility curves.
        state_loss_ratios : dict
            Mapping from damage-state label (including ``"none"``) to
            mean loss ratio in [0, 1].
        im_values : array-like, optional
            IM levels at which to evaluate the vulnerability function.
            Defaults to 50 log-spaced points between the smallest and
            largest fragility medians (×0.1 and ×5).

        Returns
        -------
        LossFunction
        """
        if im_values is None:
            medians = [c.median for c in fragility_set.curves]
            im_values = np.logspace(
                np.log10(medians[0] * 0.1),
                np.log10(medians[-1] * 5),
                50,
            )
        im_arr = np.asarray(im_values, dtype=float)
        lr_arr = np.array(
            [fragility_set.mean_damage_ratio(im, state_loss_ratios) for im in im_arr]
        )
        return cls(im_arr, lr_arr)

    def loss_ratio(self, im: float) -> float:
        """Expected loss ratio for a given IM.

        Parameters
        ----------
        im : float
            Intensity measure value.

        Returns
        -------
        float
            Expected loss ratio in [0, 1].
        """
        return float(np.clip(self._interp(im), 0.0, 1.0))

    def loss_ratios_array(self, im_values: "array-like") -> np.ndarray:
        """Vectorised :meth:`loss_ratio`."""
        return np.clip(self._interp(np.asarray(im_values, dtype=float)), 0.0, 1.0)

    def expected_loss(self, im: float, replacement_value: float = 1.0) -> float:
        """Absolute expected loss for a given IM.

        Parameters
        ----------
        im : float
            Intensity measure value.
        replacement_value : float
            Full replacement value of the asset (default 1.0 = use loss ratio).

        Returns
        -------
        float
            Expected loss in the same units as *replacement_value*.
        """
        return self.loss_ratio(im) * replacement_value

    def __repr__(self) -> str:
        return (
            f"LossFunction(im_range=[{self.im_values[0]:.3g}, {self.im_values[-1]:.3g}], "
            f"lr_range=[{self.loss_ratios[0]:.3g}, {self.loss_ratios[-1]:.3g}])"
        )


def compute_expected_loss(
    im: float,
    loss_functions: List[LossFunction],
    replacement_values: Optional[List[float]] = None,
) -> float:
    """Total expected loss across a portfolio of assets at a single IM level.

    Parameters
    ----------
    im : float
        Intensity measure value.
    loss_functions : list of LossFunction
        One per asset.
    replacement_values : list of float, optional
        Replacement values for each asset.  Defaults to 1.0 for all.

    Returns
    -------
    float
        Sum of expected losses across the portfolio.
    """
    if replacement_values is None:
        replacement_values = [1.0] * len(loss_functions)
    if len(loss_functions) != len(replacement_values):
        raise ValueError(
            "loss_functions and replacement_values must have the same length."
        )
    return sum(
        lf.expected_loss(im, rv)
        for lf, rv in zip(loss_functions, replacement_values)
    )


def compute_loss_curve(
    loss_function: LossFunction,
    hazard_curve: HazardCurve,
    n_im_points: int = 100,
) -> tuple:
    """Compute an annual loss curve by convolving a loss function with a hazard curve.

    The annual loss curve gives the mean annual rate of exceeding a loss
    level *l*, computed as:

        λ(L > l) ≈ ∫ P(L > l | IM=im) |dH/dim| dim

    This implementation treats the loss function as a deterministic mapping
    (no uncertainty in loss given IM), so:

        P(L > l | IM=im) = 1  if E[L|im] > l,  else 0

    which simplifies to evaluating the hazard at the IM that corresponds to
    loss level *l*.

    Parameters
    ----------
    loss_function : LossFunction
        Expected loss ratio as a function of IM.
    hazard_curve : HazardCurve
        Site hazard curve.
    n_im_points : int
        Number of IM integration points (default 100).

    Returns
    -------
    tuple of (loss_ratios, annual_rates)
        ``loss_ratios`` – 1-D array of loss ratio levels.
        ``annual_rates`` – 1-D array of corresponding mean annual rates of
        exceedance.

    Examples
    --------
    >>> import numpy as np
    >>> from seismicresiliencekit import LossFunction, HazardCurve, compute_loss_curve
    >>> im = np.array([0.05, 0.10, 0.20, 0.40, 0.80])
    >>> rates = np.array([0.10, 0.04, 0.01, 0.002, 0.0002])
    >>> hc = HazardCurve(im, rates)
    >>> lf = LossFunction([0.05, 0.10, 0.20, 0.40, 0.80],
    ...                   [0.01, 0.05, 0.15, 0.40, 0.80])
    >>> lr, ar = compute_loss_curve(lf, hc)
    >>> len(lr) == len(ar)
    True
    """
    # Sample IM range common to both functions
    im_min = max(loss_function.im_values[0], hazard_curve.im_values[0])
    im_max = min(loss_function.im_values[-1], hazard_curve.im_values[-1])
    im_pts = np.logspace(np.log10(im_min), np.log10(im_max), n_im_points)

    lr_pts = loss_function.loss_ratios_array(im_pts)
    rate_pts = hazard_curve.annual_rates_array(im_pts)

    # Sort by loss ratio (ascending) to get the loss curve
    sort_idx = np.argsort(lr_pts)
    lr_sorted = lr_pts[sort_idx]
    rate_sorted = rate_pts[sort_idx]

    # Unique loss ratio values with corresponding rates (take first = highest rate)
    unique_lr, unique_idx = np.unique(lr_sorted, return_index=True)
    unique_rates = rate_sorted[unique_idx]

    return unique_lr, unique_rates


def expected_annual_loss(
    loss_function: LossFunction,
    hazard_curve: HazardCurve,
    n_im_points: int = 200,
) -> float:
    """Expected annual loss (EAL) by numerical integration.

    EAL = ∫ E[L|im] |dH/dim| dim

    Parameters
    ----------
    loss_function : LossFunction
        Expected loss ratio as a function of IM.
    hazard_curve : HazardCurve
        Site hazard curve.
    n_im_points : int
        Number of integration points.

    Returns
    -------
    float
        Expected annual loss ratio.
    """
    im_min = max(loss_function.im_values[0], hazard_curve.im_values[0])
    im_max = min(loss_function.im_values[-1], hazard_curve.im_values[-1])
    im_pts = np.logspace(np.log10(im_min), np.log10(im_max), n_im_points)

    lr_pts = loss_function.loss_ratios_array(im_pts)
    rate_pts = hazard_curve.annual_rates_array(im_pts)

    # EAL = -∫ E[L|im] dH(im)  (negative because H decreases with IM)
    eal = -np.trapezoid(lr_pts, rate_pts)
    return float(eal)
