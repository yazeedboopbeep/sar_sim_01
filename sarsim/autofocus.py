"""Autofocus: recover coherence when the assumed positions are wrong.

If a position error is what breaks the image, the image itself contains the
evidence needed to undo it. Autofocus is standard practice in airborne SAR for
exactly this reason -- aircraft navigation is never good enough on its own, and
nobody solves that by buying a better INS.

Method implemented here (a wideband variant of phase-gradient autofocus):

  1. Form an image with the assumed positions and find the brightest pixel.
  2. Extract each aperture position's individual contribution to that pixel.
     With perfect positions these are all in phase -- that is what makes a peak.
     Any relative phase between them is the position error, measured.
  3. Convert the measured phase to a RANGE error rather than a constant phase
     offset. The system is wideband, so a position error of d produces a phase
     error 4*pi*f*d/c that grows across the band by a factor of 1.7 from 3.1 to
     5.3 GHz. Correcting a single phase would fix mid-band and leave the edges
     wrong; correcting a delay fixes all of it.
  4. Apply the correction and repeat.

Limits worth stating plainly: this needs a dominant scatterer to lock onto, and
the phase measurement wraps once the error exceeds a quarter wavelength
(~18 mm at band centre), so it cannot rescue an arbitrarily bad trajectory.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .config import C, RadarConfig
from .imaging import _interp_complex, back_project, range_compress


def estimate_position_errors(data: np.ndarray, radar: RadarConfig,
                             positions_assumed: np.ndarray,
                             focus_xy: Tuple[float, float]) -> np.ndarray:
    """Measure the per-position range error from the phase at one bright pixel."""
    ranges, profiles = range_compress(data, radar)
    px, py = focus_xy
    r = np.hypot(positions_assumed[:, 0] - px, positions_assumed[:, 1] - py)

    contributions = np.empty(positions_assumed.shape[0], dtype=np.complex128)
    k_carrier = 4.0 * np.pi * radar.f_start_hz / C
    for n in range(positions_assumed.shape[0]):
        contributions[n] = (_interp_complex(ranges, profiles[n],
                                            np.array([r[n]]))[0]
                            * np.exp(1j * k_carrier * r[n]))

    # Phase relative to the aperture mean: a common phase is a global image
    # phase and carries no position information.
    reference = np.mean(contributions)
    if abs(reference) < 1e-15:
        return np.zeros(positions_assumed.shape[0])
    phase = np.angle(contributions * np.conj(reference))
    return -phase * radar.wavelength_center_m / (4.0 * np.pi)


def autofocus(data: np.ndarray, radar: RadarConfig,
              positions_assumed: np.ndarray,
              grid_x: np.ndarray, grid_y: np.ndarray,
              n_iterations: int = 4,
              gate_m: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Iteratively estimate and remove per-position range errors.

    `gate_m`: if the first estimate's RMS is below this, leave the data alone.
    On a scene with no single dominant scatterer the estimator picks up the
    interference between neighbouring body points and "corrects" a phase error
    that was never there, which costs about 1.7 dB. Gating on the estimator's
    own output is the cheap fix: it says how bad it thinks the trajectory is,
    and if that is small there is nothing to win.

    Returns (corrected_data, total_estimated_range_error_per_position).
    """
    if gate_m > 0.0:
        first = estimate_position_errors(
            data, radar,positions_assumed,
            back_project(data, radar, positions_assumed,
                         grid_x, grid_y).peak_position())
        if float(np.sqrt(np.mean(first ** 2))) < gate_m:
            return data.copy(), np.zeros(positions_assumed.shape[0])

    corrected = data.copy()
    total = np.zeros(positions_assumed.shape[0])
    freqs = radar.frequencies()

    for _ in range(n_iterations):
        image = back_project(corrected, radar, positions_assumed, grid_x, grid_y)
        delta = estimate_position_errors(corrected, radar, positions_assumed,
                                         image.peak_position())
        corrected = corrected * np.exp(
            2j * np.pi * 2.0 * np.outer(delta, freqs) / C)
        total += delta
        if np.max(np.abs(delta)) < 1e-5:   # 10 micrometres: converged
            break

    return corrected, total
