"""
SeismicResilienceKit
====================
A library for collecting practical scripts for seismic resilience assessment.

Modules
-------
- fragility   : Fragility curves and damage state probabilities
- hazard      : Seismic hazard analysis (PSHA)
- ground_motion: Ground motion intensity measure calculations
- loss        : Loss estimation from damage states
- resilience  : Resilience metrics and recovery modeling
"""

from .fragility import FragilityCurve, FragilitySet
from .hazard import HazardCurve, compute_mean_annual_rate, disaggregate
from .ground_motion import (
    peak_ground_acceleration,
    spectral_acceleration,
    arias_intensity,
    significant_duration,
    cumulative_absolute_velocity,
)
from .loss import LossFunction, compute_expected_loss, compute_loss_curve
from .resilience import (
    ResilienceFunction,
    trapezoid_resilience,
    compute_functionality,
)

__version__ = "0.1.0"
__all__ = [
    # fragility
    "FragilityCurve",
    "FragilitySet",
    # hazard
    "HazardCurve",
    "compute_mean_annual_rate",
    "disaggregate",
    # ground motion
    "peak_ground_acceleration",
    "spectral_acceleration",
    "arias_intensity",
    "significant_duration",
    "cumulative_absolute_velocity",
    # loss
    "LossFunction",
    "compute_expected_loss",
    "compute_loss_curve",
    # resilience
    "ResilienceFunction",
    "trapezoid_resilience",
    "compute_functionality",
]
