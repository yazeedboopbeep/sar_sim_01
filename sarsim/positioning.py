"""Antenna-position error models.

This module is the whole point of Week 5.

Synthetic aperture imaging works by adding echoes from many antenna positions
coherently. That sum only adds up if the position of each echo is KNOWN. The
Week 4 simulator assumed positions were known exactly, which is the one
assumption the product cannot buy off the shelf: a wearable knows where it is
from visual-inertial odometry (VIO), not from a rail.

The tolerance follows from the phase error a position error causes. A radial
position error d changes the two-way path by 2d, so

    phase error  phi = 4 * pi * d / lambda

and for zero-mean Gaussian phase errors of standard deviation sigma_phi the
coherent peak amplitude is scaled by exp(-sigma_phi^2 / 2), i.e.

    loss_dB = 4.343 * sigma_phi^2

The energy does not vanish; it moves out of the main lobe into a noise-like
pedestal, so the image loses contrast before it loses focus.

Two error types are modelled because they behave differently:

  * JITTER  -- independent random error at each position. Raises the pedestal,
               costs peak SNR, barely moves the target.
  * DRIFT   -- a slow systematic ramp across the aperture, which is what
               inertial navigation actually does. Displaces and smears the
               target rather than just dimming it.
"""

from __future__ import annotations

import numpy as np

from .config import RadarConfig


def phase_error_rad(position_error_m: float, radar: RadarConfig,
                    at_band_edge: bool = True) -> float:
    """Two-way phase error caused by a radial position error."""
    lam = radar.wavelength_min_m if at_band_edge else radar.wavelength_center_m
    return 4.0 * np.pi * position_error_m / lam


def predicted_loss_db(sigma_pos_m: float, radar: RadarConfig,
                      at_band_edge: bool = True) -> float:
    """Closed-form coherence loss for Gaussian position error.

    loss_dB = 4.343 * sigma_phi^2, from |E[exp(j*phi)]| = exp(-sigma_phi^2 / 2).
    Used to check the simulation, not to produce its results.
    """
    return 4.343 * phase_error_rad(sigma_pos_m, radar, at_band_edge) ** 2


def tolerance_m(radar: RadarConfig, max_loss_db: float = 1.0,
                at_band_edge: bool = True) -> float:
    """Invert the loss law: the position RMS error costing `max_loss_db`."""
    lam = radar.wavelength_min_m if at_band_edge else radar.wavelength_center_m
    sigma_phi = np.sqrt(max_loss_db / 4.343)
    return sigma_phi * lam / (4.0 * np.pi)


def apply_jitter(positions: np.ndarray, sigma_m: float,
                 rng: np.random.Generator) -> np.ndarray:
    """Independent zero-mean Gaussian error on each antenna position.

    Applied in both axes: along-track error mostly shifts the aperture sample,
    cross-track (range-direction) error is what dominates the phase term.
    """
    if sigma_m <= 0:
        return positions.copy()
    return positions + rng.normal(0.0, sigma_m, positions.shape)


def apply_drift(positions: np.ndarray, total_drift_m: float,
                axis: str = "y", order: int = 1) -> np.ndarray:
    """Systematic drift accumulated across the aperture.

    VIO error grows with distance travelled (typically 0.1-1% of path length),
    so over a 1.1 m aperture it is not white noise, it is a smooth curve.

    `order` decides which kind of smooth, and the two behave completely
    differently in the image:

      order=1  linear ramp -> a linear phase ramp across the aperture, which is
               a SHIFT. The target stays sharp and lands in the wrong place.
      order=2  quadratic bow of sagitta `total_drift_m` at the aperture centre
               -> a quadratic phase error, which is DEFOCUS. The target stays
               put and smears. A ramp with no curvature costs nothing but
               accuracy of position; curvature costs the image itself.

    This is why "how big is the drift" is the wrong question: what matters is
    how much of it is curvature.
    """
    out = positions.copy()
    n = positions.shape[0]
    u = np.linspace(0.0, 1.0, n)
    shape = u if order == 1 else (1.0 - 4.0 * (u - 0.5) ** 2)
    out[:, 1 if axis == "y" else 0] += total_drift_m * shape
    return out


def vio_error_over_aperture(aperture_m: float, drift_rate_percent: float) -> float:
    """Position error a VIO system accumulates over one aperture.

    Published visual-inertial odometry drift rates run roughly 0.1-1% of
    distance travelled, so this converts a spec sheet number into the quantity
    this module cares about.
    """
    return aperture_m * drift_rate_percent / 100.0
