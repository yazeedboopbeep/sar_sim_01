"""Back-projection imaging.

Two implementations:

  back_project_direct  -- the v0.1 formulation, a coherent sum over every
                          (position, frequency) pair. O(P x F x N_pixels).
                          Kept as the reference the fast path is tested against.

  back_project         -- range-compress each position's spectrum ONCE, then
                          interpolate the compressed profile at each pixel's
                          range. O(P x N_pixels) plus one FFT per position.
                          Mathematically the same sum, about two orders of
                          magnitude faster, which is what makes the Monte Carlo
                          sweeps in this week's experiments practical.

The identity being exploited: writing f = f_start + m*df,

    sum_m s[n,m] exp(+j 4 pi f R / c)
        = exp(+j 4 pi f_start R / c) * sum_m s[n,m] exp(+j 4 pi m df R / c)

and the second factor is exactly an inverse DFT evaluated at range R. So
compress once per position, then all the imager does per pixel is look up a
range and rotate by the carrier phase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

from .config import C, ApertureConfig, RadarConfig


@dataclass
class Image:
    x: np.ndarray
    y: np.ndarray
    values: np.ndarray = field(default=None, repr=False)

    @property
    def magnitude_db(self) -> np.ndarray:
        mag = np.abs(self.values)
        peak = mag.max() if mag.max() > 0 else 1.0
        return 20.0 * np.log10(np.maximum(mag / peak, 1e-6))

    def peak_position(self) -> Tuple[float, float]:
        iy, ix = np.unravel_index(np.argmax(np.abs(self.values)),
                                  self.values.shape)
        return float(self.x[ix]), float(self.y[iy])


def range_compress(data: np.ndarray, radar: RadarConfig,
                   n_fft: int = 4096,
                   window: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Inverse-DFT each position's spectrum into a complex range profile.

    Returns (ranges, profiles) with profiles shaped (n_positions, n_fft).
    Windowing is OFF by default so the result matches the unweighted direct sum
    exactly; turn it on to trade resolution for lower range sidelobes.
    """
    x = data * np.hanning(radar.n_freq)[None, :] if window else data
    profiles = np.fft.ifft(x, n=n_fft, axis=1) * n_fft
    ranges = np.arange(n_fft) * C / (2.0 * radar.freq_step_hz * n_fft)
    return ranges, profiles


def _interp_complex(ranges: np.ndarray, profile: np.ndarray,
                    query: np.ndarray) -> np.ndarray:
    """Linear interpolation of a complex profile at arbitrary ranges."""
    dr = ranges[1] - ranges[0]
    idx = query / dr
    i0 = np.floor(idx).astype(np.int64)
    frac = idx - i0
    i0 = np.clip(i0, 0, profile.size - 2)
    return profile[i0] * (1.0 - frac) + profile[i0 + 1] * frac


def back_project(data: np.ndarray, radar: RadarConfig,
                 positions_assumed: np.ndarray,
                 grid_x: np.ndarray, grid_y: np.ndarray,
                 n_fft: int = 4096, window: bool = False) -> Image:
    """Fast back-projection over the assumed antenna positions.

    `positions_assumed` is what the navigation system THINKS the antenna
    positions were. Feeding it something different from the positions used to
    generate the data is how positioning error is simulated.
    """
    ranges, profiles = range_compress(data, radar, n_fft=n_fft, window=window)
    XX, YY = np.meshgrid(grid_x, grid_y, indexing="xy")
    fx, fy = XX.ravel(), YY.ravel()

    acc = np.zeros(fx.size, dtype=np.complex128)
    k_carrier = 4.0 * np.pi * radar.f_start_hz / C
    for n in range(positions_assumed.shape[0]):
        r = np.hypot(fx - positions_assumed[n, 0], fy - positions_assumed[n, 1])
        acc += _interp_complex(ranges, profiles[n], r) * np.exp(1j * k_carrier * r)

    return Image(grid_x, grid_y, acc.reshape(XX.shape))


def back_project_direct(data: np.ndarray, radar: RadarConfig,
                        positions_assumed: np.ndarray,
                        grid_x: np.ndarray, grid_y: np.ndarray) -> Image:
    """Reference implementation: the explicit double sum.

        I(p) = sum_n sum_f s[n,f] * exp(+j 4 pi f R_n(p) / c)

    Slow. Its only job is to be obviously correct so the fast path can be
    tested against it.
    """
    freqs = radar.frequencies()
    XX, YY = np.meshgrid(grid_x, grid_y, indexing="xy")
    fx, fy = XX.ravel(), YY.ravel()

    acc = np.zeros(fx.size, dtype=np.complex128)
    for n in range(positions_assumed.shape[0]):
        r = np.hypot(fx - positions_assumed[n, 0], fy - positions_assumed[n, 1])
        acc += np.exp(2j * np.pi * 2.0 * np.outer(r, freqs) / C) @ data[n]

    return Image(grid_x, grid_y, acc.reshape(XX.shape))


def cross_range_cut(data: np.ndarray, radar: RadarConfig,
                    positions_assumed: np.ndarray, range_m: float,
                    half_width_m: float = 0.45,
                    n_points: int = 601) -> Tuple[np.ndarray, np.ndarray]:
    """1-D image cut at fixed range -- the cheap way to measure cross-range
    resolution when sweeping a parameter."""
    x = np.linspace(-half_width_m, half_width_m, n_points)
    img = back_project(data, radar, positions_assumed, x, np.array([range_m]))
    return x, np.abs(img.values[0])


def range_profile(data: np.ndarray, radar: RadarConfig,
                  position_index: int | None = None,
                  n_fft: int = 2048) -> Tuple[np.ndarray, np.ndarray]:
    """Windowed range profile at one aperture position.

    Shows RANGE resolution (c/2B), which the aperture cannot change.
    """
    if position_index is None:
        position_index = data.shape[0] // 2
    ranges, profiles = range_compress(data[position_index][None, :], radar,
                                      n_fft=n_fft, window=True)
    return ranges, np.abs(profiles[0])


def default_grid(range_m: float = 3.0, half_width_m: float = 1.0,
                 depth_m: float = 2.0, step_m: float = 0.0125):
    """Standard imaging grid centred on the target."""
    gx = np.arange(-half_width_m, half_width_m + step_m, step_m)
    gy = np.arange(range_m - depth_m / 2, range_m + depth_m / 2 + step_m, step_m)
    return gx, gy
