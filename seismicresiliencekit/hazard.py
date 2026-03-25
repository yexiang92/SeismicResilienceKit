"""
hazard.py
---------
Seismic hazard analysis utilities.

Provides tools for:
- Representing and manipulating hazard curves (annual rate of exceedance vs IM)
- Computing mean annual rate of exceedance from a hazard model
- Basic seismic disaggregation
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.interpolate import interp1d


class HazardCurve:
    """Seismic hazard curve: annual rate of exceedance as a function of IM.

    A hazard curve *H(im)* gives the mean annual rate (or probability) of
    exceeding the intensity measure value *im* at a site.

    Parameters
    ----------
    im_values : array-like
        Intensity measure levels (must be strictly increasing, > 0).
    annual_rates : array-like
        Corresponding mean annual rates of exceedance (> 0).

    Examples
    --------
    >>> import numpy as np
    >>> im = np.array([0.05, 0.10, 0.20, 0.40, 0.80])
    >>> rates = np.array([0.10, 0.04, 0.01, 0.002, 0.0002])
    >>> hc = HazardCurve(im, rates)
    >>> round(hc.annual_rate(0.10), 4)
    0.04
    >>> round(hc.return_period(0.10), 2)
    25.0
    """

    def __init__(
        self,
        im_values: "array-like",
        annual_rates: "array-like",
    ) -> None:
        im = np.asarray(im_values, dtype=float)
        rates = np.asarray(annual_rates, dtype=float)
        if im.ndim != 1 or rates.ndim != 1:
            raise ValueError("im_values and annual_rates must be 1-D arrays.")
        if len(im) != len(rates):
            raise ValueError("im_values and annual_rates must have the same length.")
        if not np.all(np.diff(im) > 0):
            raise ValueError("im_values must be strictly increasing.")
        if np.any(im <= 0):
            raise ValueError("im_values must be positive.")
        if np.any(rates < 0):
            raise ValueError("annual_rates must be non-negative.")
        self.im_values = im
        self.annual_rates = rates
        # Build log-log interpolator for the valid (positive-rate) range
        mask = rates > 0
        if mask.sum() < 2:
            raise ValueError("At least two positive annual_rates are required.")
        self._interp = interp1d(
            np.log(im[mask]),
            np.log(rates[mask]),
            kind="linear",
            bounds_error=False,
            fill_value=(np.log(rates[mask][0]), np.log(rates[mask][-1])),
        )

    def annual_rate(self, im: float) -> float:
        """Return the mean annual rate of exceeding *im*.

        Parameters
        ----------
        im : float
            Intensity measure value (> 0).

        Returns
        -------
        float
            Mean annual rate of exceedance.
        """
        if im <= 0:
            raise ValueError("im must be positive.")
        return float(np.exp(self._interp(np.log(im))))

    def annual_rates_array(self, im_values: "array-like") -> np.ndarray:
        """Vectorised :meth:`annual_rate` over an array of IM values."""
        im_arr = np.asarray(im_values, dtype=float)
        if np.any(im_arr <= 0):
            raise ValueError("All im_values must be positive.")
        return np.exp(self._interp(np.log(im_arr)))

    def return_period(self, im: float) -> float:
        """Return period (years) for exceedance of *im*.

        Parameters
        ----------
        im : float
            Intensity measure value (> 0).

        Returns
        -------
        float
            Return period in years.
        """
        rate = self.annual_rate(im)
        if rate == 0:
            return np.inf
        return 1.0 / rate

    def im_for_return_period(self, return_period_years: float) -> float:
        """IM value corresponding to a given return period.

        Parameters
        ----------
        return_period_years : float
            Target return period in years (> 0).

        Returns
        -------
        float
            Intensity measure level.
        """
        if return_period_years <= 0:
            raise ValueError("return_period_years must be positive.")
        target_rate = 1.0 / return_period_years
        # Invert the interpolator: search for log(IM) given log(rate)
        mask = self.annual_rates > 0
        log_im = np.log(self.im_values[mask])
        log_rates = np.log(self.annual_rates[mask])
        # Rates are decreasing with increasing IM; invert the direction
        inv_interp = interp1d(
            log_rates[::-1],
            log_im[::-1],
            kind="linear",
            bounds_error=False,
            fill_value=(log_im[-1], log_im[0]),
        )
        return float(np.exp(inv_interp(np.log(target_rate))))

    def annual_loss_rate(self, loss_fn: "array-like", im_losses: "array-like") -> float:
        """Compute mean annual loss rate by integrating over the hazard curve.

        Uses the relationship:
            λ(loss) ≈ ∫ P(Loss > l | IM=im) |dH/dim| dim

        This method computes the expected annual loss (EAL):
            EAL = ∫ E[Loss | IM=im] |dH/dim| dim

        Parameters
        ----------
        loss_fn : array-like
            Expected loss (or loss ratio) at each IM level in *im_losses*.
        im_losses : array-like
            IM levels corresponding to *loss_fn*.

        Returns
        -------
        float
            Expected annual loss.
        """
        im_arr = np.asarray(im_losses, dtype=float)
        loss_arr = np.asarray(loss_fn, dtype=float)
        rates = self.annual_rates_array(im_arr)
        # Numerical integration: EAL = -∫ E[L|im] dH(im)
        eal = -np.trapezoid(loss_arr, rates)
        return float(eal)

    def probability_of_exceedance(self, im: float, exposure_years: float = 50.0) -> float:
        """Probability of exceedance of *im* over a given exposure period.

        Uses the Poisson model: P = 1 - exp(-λ * t).

        Parameters
        ----------
        im : float
            Intensity measure value.
        exposure_years : float
            Exposure period in years (default 50).

        Returns
        -------
        float
            Probability of exceedance in [0, 1].
        """
        rate = self.annual_rate(im)
        return float(1.0 - np.exp(-rate * exposure_years))

    def __repr__(self) -> str:
        return (
            f"HazardCurve(im_range=[{self.im_values[0]:.3g}, {self.im_values[-1]:.3g}], "
            f"rate_range=[{self.annual_rates[-1]:.3g}, {self.annual_rates[0]:.3g}])"
        )


def compute_mean_annual_rate(
    sources: List[Dict],
    site_im_values: np.ndarray,
    attenuation_fn,
) -> np.ndarray:
    """Compute mean annual rate of exceedance using a simple PSHA summation.

    This is a simplified PSHA implementation that sums contributions from
    multiple seismic sources. Each source is described by a dictionary with
    keys:

    - ``activity_rate`` : mean annual rate of earthquakes on the source.
    - ``magnitude_pmf`` : list of ``(magnitude, probability)`` pairs that
      together form a discrete magnitude PMF.
    - ``distance`` : representative source-to-site distance (km).

    Parameters
    ----------
    sources : list of dict
        Seismic sources; see above for required keys.
    site_im_values : numpy.ndarray
        IM levels at which to evaluate the hazard (must be increasing).
    attenuation_fn : callable
        Ground motion prediction equation (GMPE) with signature
        ``attenuation_fn(magnitude, distance) -> (median_im, sigma_ln)``.

    Returns
    -------
    numpy.ndarray
        Mean annual rates of exceedance corresponding to *site_im_values*.

    Notes
    -----
    The computation follows the standard PSHA integral::

        λ(IM > im) = Σ_src  ν_src  Σ_m  P(M=m)  P(IM > im | M=m, R=r_src)

    where IM > im is evaluated using the lognormal distribution assumed by
    most GMPEs.
    """
    from scipy.stats import norm

    im_arr = np.asarray(site_im_values, dtype=float)
    rates = np.zeros_like(im_arr)

    for src in sources:
        activity_rate: float = src["activity_rate"]
        mag_pmf: List[Tuple[float, float]] = src["magnitude_pmf"]
        distance: float = src["distance"]
        for mag, p_mag in mag_pmf:
            median_im, sigma_ln = attenuation_fn(mag, distance)
            # P(IM > im | M, R) via lognormal exceedance
            with np.errstate(divide="ignore"):
                z = (np.log(im_arr) - np.log(median_im)) / sigma_ln
            p_exceed = 1.0 - norm.cdf(z)
            rates += activity_rate * p_mag * p_exceed

    return rates


def disaggregate(
    hazard_curve: HazardCurve,
    target_im: float,
    sources: List[Dict],
    attenuation_fn,
) -> List[Dict]:
    """Seismic disaggregation at a target IM level.

    Decomposes the hazard at *target_im* into contributions from each
    magnitude–distance bin, normalised to sum to 1.

    Parameters
    ----------
    hazard_curve : HazardCurve
        Site hazard curve (retained for API consistency; not used in
        normalisation — contributions are normalised by the sum of source
        rates at *target_im*).
    target_im : float
        Intensity measure level for disaggregation.
    sources : list of dict
        Same format as :func:`compute_mean_annual_rate`.
    attenuation_fn : callable
        Same format as :func:`compute_mean_annual_rate`.

    Returns
    -------
    list of dict
        Each entry has keys ``"magnitude"``, ``"distance"``,
        ``"contribution"`` (fractional).
    """
    from scipy.stats import norm

    contributions = []
    bin_rates = []

    for src in sources:
        activity_rate: float = src["activity_rate"]
        mag_pmf: List[Tuple[float, float]] = src["magnitude_pmf"]
        distance: float = src["distance"]
        for mag, p_mag in mag_pmf:
            median_im, sigma_ln = attenuation_fn(mag, distance)
            with np.errstate(divide="ignore"):
                z = (np.log(target_im) - np.log(median_im)) / sigma_ln
            p_exceed = 1.0 - norm.cdf(z)
            bin_rate = activity_rate * p_mag * p_exceed
            bin_rates.append(bin_rate)
            contributions.append(
                {
                    "magnitude": mag,
                    "distance": distance,
                    "contribution": bin_rate,  # normalized below
                }
            )

    total_source_rate = sum(bin_rates)
    for i, entry in enumerate(contributions):
        entry["contribution"] = (
            bin_rates[i] / total_source_rate if total_source_rate > 0 else 0.0
        )

    return contributions
