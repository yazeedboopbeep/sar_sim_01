"""Forward model: synthesise the raw stepped-frequency data cube.

The key change from v0.1: echoes are generated from the TRUE antenna positions,
while the imager is given the ASSUMED positions. When the two differ, the
coherent sum degrades -- which is exactly the effect Week 5 measures.
"""

from __future__ import annotations

from typing import List

import numpy as np

from .config import C, RadarConfig, SceneConfig
from .targets import Target


def simulate_echoes(
    targets: List[Target],
    radar: RadarConfig,
    scene: SceneConfig,
    positions_true: np.ndarray,
    include_targets: bool = True,
    add_noise: bool = True,
    rng: np.random.Generator | None = None,
    noise_power: float | None = None,
) -> np.ndarray:
    """Raw data cube s[n_position, n_frequency].

    Monostatic. For a scatterer at range R from antenna position n the two-way
    phase is exp(-j*4*pi*f*R/c) and the amplitude falls as 1/R^2 (the amplitude
    form of the R^-4 power law used in the Week 3 radar equation).
    """
    if rng is None:
        rng = np.random.default_rng(scene.seed)

    freqs = radar.frequencies()
    ax, ay = positions_true[:, 0], positions_true[:, 1]
    s = np.zeros((positions_true.shape[0], freqs.size), dtype=np.complex128)
    wall_amp = scene.two_way_wall_amplitude()

    def accumulate(px: float, py: float, amp: complex, through_wall: bool) -> None:
        r = np.hypot(ax - px, ay - py)
        a = amp / np.maximum(r, 1e-3) ** 2
        if through_wall:
            a = a * wall_amp
        s[...] += a[:, None] * np.exp(-2j * np.pi * 2.0 * np.outer(r, freqs) / C)

    # Static wall front face, modelled as a line of specular scatterers.
    for wx in np.linspace(-1.6, 1.6, 33):
        accumulate(wx, scene.wall_y_m, scene.wall_return_amplitude / 33.0, False)

    # Static furniture behind the wall.
    for cx, cy, camp in scene.clutter_points:
        accumulate(cx, cy, camp, True)

    if include_targets:
        for t in targets:
            accumulate(t.x_m, t.y_m, t.complex_amplitude, True)

    if add_noise:
        if noise_power is None:
            noise_power = calibrate_noise_power(targets, radar, scene,
                                                positions_true)
        sigma = np.sqrt(noise_power / 2.0)
        s = s + rng.normal(0, sigma, s.shape) + 1j * rng.normal(0, sigma, s.shape)

    return s


def calibrate_noise_power(targets: List[Target], radar: RadarConfig,
                          scene: SceneConfig,
                          positions: np.ndarray) -> float:
    """Absolute receiver noise power implied by radar.raw_snr_db for this scene.

    Hold the returned value fixed across a sweep so that whatever is being
    swept (wall type, position error) actually shows up in the image SNR
    instead of being normalised away.
    """
    if not targets:
        return 1.0
    ax, ay = positions[:, 0], positions[:, 1]
    wall_amp = scene.two_way_wall_amplitude()
    total = 0.0
    for t in targets:
        r = np.hypot(ax - t.x_m, ay - t.y_m)
        a = t.amplitude / np.maximum(r, 1e-3) ** 2 * wall_amp
        total += float(np.mean(np.abs(a) ** 2))
    return total / (10.0 ** (radar.raw_snr_db / 10.0))


def background_subtract(scene_data: np.ndarray,
                        empty_data: np.ndarray) -> np.ndarray:
    """Empty-room background subtraction.

    The clutter-cancellation method bench test 0a can actually execute: capture
    the empty room, then capture with a person in it. It is the OPTIMISTIC
    bound -- it assumes the static scene is perfectly repeatable, and it doubles
    the noise power (+3 dB). Reference-free cancellation is Week 6 work.
    """
    return scene_data - empty_data
