"""
fragility.py
------------
Fragility curves and damage state probability calculations.

A fragility curve gives the conditional probability that a structure reaches
or exceeds a given damage state (DS) for a specific ground-motion intensity
measure (IM).  The most common parametric form is the lognormal CDF:

    P(DS >= ds_i | IM = im) = Φ( (ln(im) - ln(θ_i)) / β_i )

where
    θ_i  – median IM capacity for damage state i
    β_i  – logarithmic standard deviation (dispersion) for damage state i
    Φ    – standard normal CDF
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.stats import norm


class FragilityCurve:
    """Lognormal fragility curve for a single damage state.

    Parameters
    ----------
    damage_state : str
        Name / label of the damage state (e.g. ``"slight"``, ``"moderate"``).
    median : float
        Median intensity measure capacity *θ* (same units as IM).
    beta : float
        Total logarithmic standard deviation *β* (> 0).
    im_name : str, optional
        Name of the intensity measure (e.g. ``"PGA"``, ``"Sa(1.0s)"``).

    Examples
    --------
    >>> fc = FragilityCurve("slight", median=0.1, beta=0.4, im_name="PGA")
    >>> round(fc.probability(0.1), 4)
    0.5
    >>> round(fc.probability(0.2), 4)
    0.8595
    """

    def __init__(
        self,
        damage_state: str,
        median: float,
        beta: float,
        im_name: str = "IM",
    ) -> None:
        if median <= 0:
            raise ValueError("median must be positive.")
        if beta <= 0:
            raise ValueError("beta must be positive.")
        self.damage_state = damage_state
        self.median = float(median)
        self.beta = float(beta)
        self.im_name = im_name

    def probability(self, im: float) -> float:
        """Return P(DS >= damage_state | IM = im).

        Parameters
        ----------
        im : float
            Intensity measure value (must be > 0).

        Returns
        -------
        float
            Exceedance probability in [0, 1].
        """
        if im <= 0:
            return 0.0
        z = math.log(im / self.median) / self.beta
        return float(norm.cdf(z))

    def probabilities(self, im_values: Sequence[float]) -> np.ndarray:
        """Vectorised version of :meth:`probability`.

        Parameters
        ----------
        im_values : array-like
            Sequence of IM values.

        Returns
        -------
        numpy.ndarray
            Array of exceedance probabilities.
        """
        im_arr = np.asarray(im_values, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(im_arr > 0, np.log(im_arr / self.median) / self.beta, -np.inf)
        return norm.cdf(z)

    def __repr__(self) -> str:
        return (
            f"FragilityCurve(damage_state={self.damage_state!r}, "
            f"median={self.median}, beta={self.beta}, im_name={self.im_name!r})"
        )


class FragilitySet:
    """A set of fragility curves covering multiple ordered damage states.

    Damage states must be provided in ascending order of severity (from the
    least severe to the most severe). Each curve represents the probability
    of *reaching or exceeding* that damage state.

    Parameters
    ----------
    curves : list of FragilityCurve
        Fragility curves ordered from least to most severe damage state.

    Examples
    --------
    >>> fs = FragilitySet([
    ...     FragilityCurve("slight",    median=0.10, beta=0.4),
    ...     FragilityCurve("moderate",  median=0.20, beta=0.4),
    ...     FragilityCurve("extensive", median=0.40, beta=0.4),
    ...     FragilityCurve("complete",  median=0.80, beta=0.4),
    ... ])
    >>> probs = fs.damage_state_probabilities(0.20)
    >>> abs(sum(probs.values()) - 1.0) < 1e-10
    True
    """

    def __init__(self, curves: List[FragilityCurve]) -> None:
        if not curves:
            raise ValueError("At least one FragilityCurve is required.")
        self.curves = list(curves)
        self._validate_medians()

    def _validate_medians(self) -> None:
        medians = [c.median for c in self.curves]
        for i in range(1, len(medians)):
            if medians[i] <= medians[i - 1]:
                raise ValueError(
                    "Fragility curve medians must be strictly increasing. "
                    f"Got {medians[i - 1]} then {medians[i]}."
                )

    @property
    def damage_states(self) -> List[str]:
        """Ordered list of damage state labels."""
        return [c.damage_state for c in self.curves]

    def exceedance_probabilities(self, im: float) -> Dict[str, float]:
        """P(DS >= ds_i | IM) for every damage state.

        Parameters
        ----------
        im : float
            Intensity measure value.

        Returns
        -------
        dict
            Mapping from damage-state label to exceedance probability.
        """
        return {c.damage_state: c.probability(im) for c in self.curves}

    def damage_state_probabilities(self, im: float) -> Dict[str, float]:
        """Probability of being in *exactly* each discrete damage state.

        States are: ``"none"``, then each labelled damage state.

        Parameters
        ----------
        im : float
            Intensity measure value.

        Returns
        -------
        dict
            Mapping from state label (including ``"none"``) to probability.
        """
        exceed = [c.probability(im) for c in self.curves]
        probs: Dict[str, float] = {}
        probs["none"] = 1.0 - exceed[0]
        for i, curve in enumerate(self.curves):
            next_p = exceed[i + 1] if i + 1 < len(exceed) else 0.0
            probs[curve.damage_state] = exceed[i] - next_p
        return probs

    def mean_damage_ratio(
        self, im: float, damage_ratios: Optional[Dict[str, float]] = None
    ) -> float:
        """Expected damage ratio given IM.

        Parameters
        ----------
        im : float
            Intensity measure value.
        damage_ratios : dict, optional
            Mapping from damage-state label (including ``"none"``) to a
            damage ratio in [0, 1].  If not provided, linearly spaced
            ratios ``[0, 1/(n), 2/(n), …, 1]`` are used where *n* is the
            number of damage states.

        Returns
        -------
        float
            Mean damage ratio in [0, 1].
        """
        n = len(self.curves)
        if damage_ratios is None:
            states = ["none"] + self.damage_states
            damage_ratios = {s: i / n for i, s in enumerate(states)}
        ds_probs = self.damage_state_probabilities(im)
        return sum(damage_ratios[s] * p for s, p in ds_probs.items())

    def __repr__(self) -> str:
        ds = ", ".join(self.damage_states)
        return f"FragilitySet([{ds}])"
