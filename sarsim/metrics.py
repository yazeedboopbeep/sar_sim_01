"""Resolution, separability and SNR metrics."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .config import ApertureConfig, RadarConfig
from .imaging import Image
from .targets import Target

RAYLEIGH_DIP_DB = 1.34   # 26.5% intensity dip -- the classical Rayleigh criterion


def theoretical_cross_range_res(radar: RadarConfig, range_m: float,
                                aperture_len_m: float) -> float:
    """d_cr = lambda_c * R / (2 L) for a monostatic synthetic aperture."""
    return radar.wavelength_center_m * range_m / (2.0 * aperture_len_m)


def required_aperture(radar: RadarConfig, range_m: float,
                      target_res_m: float) -> float:
    """Invert the resolution law: L = lambda_c * R / (2 d)."""
    return radar.wavelength_center_m * range_m / (2.0 * target_res_m)


def minus_3db_width(x: np.ndarray, profile: np.ndarray) -> float:
    """Half-power width of a single-point response."""
    p = profile / profile.max()
    peak_i = int(np.argmax(p))
    half = 1.0 / np.sqrt(2.0)

    def edge(direction: int) -> float:
        i = peak_i
        while 0 < i < p.size - 1:
            j = i + direction
            if p[j] < half:
                t = (half - p[j]) / (p[i] - p[j] + 1e-12)
                return x[j] + t * (x[i] - x[j])
            i = j
        return x[0] if direction < 0 else x[-1]

    return abs(edge(+1) - edge(-1))


def resolved(x: np.ndarray, profile: np.ndarray, sep_m: float,
             dip_db: float = RAYLEIGH_DIP_DB) -> Tuple[bool, float]:
    """Rayleigh two-target separability.

    Resolved if the profile shows two maxima near the true positions with a
    saddle at least `dip_db` below the weaker peak. The default is the classical
    Rayleigh criterion (saddle at 73.5% of peak intensity = 1.34 dB).
    """
    p = profile / profile.max()
    left = (x >= -sep_m) & (x <= 0.0)
    right = (x >= 0.0) & (x <= sep_m)
    if not left.any() or not right.any():
        return False, 0.0
    centre = (x >= -sep_m / 2.0) & (x <= sep_m / 2.0)
    valley = p[centre].min()
    dip = 20.0 * np.log10(min(p[left].max(), p[right].max()) / max(valley, 1e-12))
    return bool(dip >= dip_db), float(dip)


def image_snr_db(image: Image, targets: List[Target],
                 box_m: float = 0.15) -> float:
    """Peak target response over the RMS floor away from the targets.

    This is the contrast metric: it falls when energy leaks out of the main
    lobes into a pedestal, which is exactly what a position error does.
    """
    XX, YY = np.meshgrid(image.x, image.y, indexing="xy")
    mag = np.abs(image.values)
    mask = np.zeros_like(mag, dtype=bool)
    for t in targets:
        mask |= (np.abs(XX - t.x_m) < box_m) & (np.abs(YY - t.y_m) < box_m)
    if not mask.any() or mask.all():
        return float("nan")
    return float(20.0 * np.log10(mag[mask].max()
                                 / max(np.sqrt(np.mean(mag[~mask] ** 2)), 1e-12)))


def peak_amplitude_at(image: Image, target: Target) -> float:
    """Coherent response at the known target location."""
    XX, YY = np.meshgrid(image.x, image.y, indexing="xy")
    d = np.hypot(XX - target.x_m, YY - target.y_m)
    iy, ix = np.unravel_index(np.argmin(d), d.shape)
    return float(np.abs(image.values[iy, ix]))


def peak_displacement_m(image: Image, target: Target) -> float:
    """How far the brightest pixel has moved from the true target position.

    Jitter dims a target; drift moves it. This metric separates the two.
    """
    px, py = image.peak_position()
    return float(np.hypot(px - target.x_m, py - target.y_m))
