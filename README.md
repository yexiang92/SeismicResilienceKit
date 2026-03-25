# SeismicResilienceKit
A library for collecting practical scripts for seismic resilience assessment.

## Overview

**SeismicResilienceKit** is a Python library that provides practical tools for
seismic resilience assessment of buildings and infrastructure systems. It covers
the full workflow from ground-motion characterisation to resilience quantification:

| Module | Description |
|---|---|
| `fragility` | Lognormal fragility curves and damage-state probability calculations |
| `hazard` | Seismic hazard curves, PSHA summation, and disaggregation |
| `ground_motion` | Ground-motion intensity measure (IM) calculations from time-series |
| `loss` | Vulnerability functions and expected annual loss estimation |
| `resilience` | Functionality curves, resilience indices, and recovery modelling |

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Quick Start

### Fragility Curves

```python
from seismicresiliencekit import FragilityCurve, FragilitySet

# Single lognormal fragility curve
fc = FragilityCurve("slight", median=0.10, beta=0.4, im_name="PGA [g]")
print(fc.probability(0.10))   # 0.5  (at median)
print(fc.probability(0.20))   # ~0.86

# Multi-state fragility set
fs = FragilitySet([
    FragilityCurve("slight",    median=0.10, beta=0.4),
    FragilityCurve("moderate",  median=0.20, beta=0.4),
    FragilityCurve("extensive", median=0.40, beta=0.4),
    FragilityCurve("complete",  median=0.80, beta=0.4),
])

# Probability of being in each damage state at PGA = 0.20 g
probs = fs.damage_state_probabilities(0.20)
# {'none': ..., 'slight': ..., 'moderate': ..., 'extensive': ..., 'complete': ...}
```

### Seismic Hazard

```python
import numpy as np
from seismicresiliencekit import HazardCurve

im    = np.array([0.05, 0.10, 0.20, 0.40, 0.80])  # PGA [g]
rates = np.array([0.10, 0.04, 0.01, 0.002, 0.0002])  # annual rates

hc = HazardCurve(im, rates)
print(hc.return_period(0.10))                    # 25 years
print(hc.im_for_return_period(475))              # 475-year PGA
print(hc.probability_of_exceedance(0.10, 50))   # 50-year exceedance probability
```

### Ground Motion Intensity Measures

```python
import numpy as np
from seismicresiliencekit import (
    peak_ground_acceleration,
    spectral_acceleration,
    arias_intensity,
    significant_duration,
    cumulative_absolute_velocity,
)

dt  = 0.01          # time step [s]
acc = np.loadtxt("record.txt")  # acceleration [m/s²]

pga  = peak_ground_acceleration(acc)
sa1  = spectral_acceleration(acc, dt, period=1.0, damping_ratio=0.05)
ia   = arias_intensity(acc, dt)
d595 = significant_duration(acc, dt, pct_start=5, pct_end=95)
cav  = cumulative_absolute_velocity(acc, dt)
```

### Loss Estimation

```python
from seismicresiliencekit import LossFunction, compute_expected_loss, compute_loss_curve, HazardCurve
import numpy as np

# Direct vulnerability function
lf = LossFunction(
    im_values   = [0.05, 0.10, 0.20, 0.40, 0.80],
    loss_ratios = [0.01, 0.05, 0.15, 0.40, 0.80],
)
print(lf.loss_ratio(0.20))               # expected loss ratio at PGA=0.20 g
print(lf.expected_loss(0.20, 1_000_000)) # expected loss for $1M building

# Build from a FragilitySet
from seismicresiliencekit import FragilitySet, FragilityCurve
fs = FragilitySet([
    FragilityCurve("slight",    median=0.10, beta=0.4),
    FragilityCurve("moderate",  median=0.20, beta=0.4),
    FragilityCurve("extensive", median=0.40, beta=0.4),
    FragilityCurve("complete",  median=0.80, beta=0.4),
])
state_lr = {"none": 0.0, "slight": 0.05, "moderate": 0.20,
            "extensive": 0.50, "complete": 1.00}
lf2 = LossFunction.from_fragility_set(fs, state_lr)

# Annual loss curve and expected annual loss
hc = HazardCurve(
    np.array([0.05, 0.10, 0.20, 0.40, 0.80]),
    np.array([0.10, 0.04, 0.01, 0.002, 0.0002]),
)
from seismicresiliencekit.loss import expected_annual_loss
eal = expected_annual_loss(lf2, hc)
print(f"Expected Annual Loss ratio: {eal:.4f}")
```

### Resilience Metrics

```python
from seismicresiliencekit import ResilienceFunction, trapezoid_resilience, compute_functionality

# Functionality-recovery curve
rf = ResilienceFunction(
    t_event        = 0,
    q_post         = 0.6,     # 60 % functionality immediately after earthquake
    t_recovery     = 180,     # full recovery in 180 days
    recovery_shape = "trigonometric",
)
print(rf.functionality(90))             # functionality at day 90
print(rf.resilience_index(0, 180))      # normalised resilience index
print(rf.loss_of_resilience(0, 180))    # area of resilience triangle

# Quick trapezoidal estimate
delta_r = trapezoid_resilience(q_post=0.6, t_recovery=180)

# Map damage ratio to functionality
q = compute_functionality(damage_ratio=0.3, functionality_model="linear")
```

## Modules

### `seismicresiliencekit.fragility`
- `FragilityCurve` – single lognormal fragility curve
- `FragilitySet` – ordered set of fragility curves for multiple damage states

### `seismicresiliencekit.hazard`
- `HazardCurve` – site hazard curve with interpolation and return-period tools
- `compute_mean_annual_rate` – simplified PSHA summation over seismic sources
- `disaggregate` – seismic disaggregation by magnitude–distance bin

### `seismicresiliencekit.ground_motion`
- `peak_ground_acceleration` – PGA from acceleration time-series
- `spectral_acceleration` – Sa(T, ξ) via Newmark integration
- `arias_intensity` – Arias intensity Ia
- `significant_duration` – D5-75, D5-95, etc.
- `cumulative_absolute_velocity` – CAV and CAV_std

### `seismicresiliencekit.loss`
- `LossFunction` – vulnerability function (direct or fragility-based)
- `compute_expected_loss` – portfolio-level expected loss at a single IM
- `compute_loss_curve` – annual loss exceedance curve
- `expected_annual_loss` – EAL by numerical integration with hazard curve

### `seismicresiliencekit.resilience`
- `ResilienceFunction` – time-varying functionality curve with multiple recovery shapes
- `trapezoid_resilience` – quick triangular resilience-loss estimate
- `compute_functionality` – damage-ratio to functionality conversion
- `recovery_time_from_damage` – damage-ratio to recovery-time estimate

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/
```

## License

See [LICENSE](LICENSE).

