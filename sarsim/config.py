"""Radar, wall, scene and aperture configuration.

Every constant carries a named source note -- same convention as the Week 3
link budget and the Week 4 v0.1 simulator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

C = 299_792_458.0  # speed of light, m/s


@dataclass(frozen=True)
class RadarConfig:
    """Radar front-end parameters.

    Defaults = TDSR P452 full-UWB dev kit, the platform selected in Week 4
    (3.1-5.3 GHz, 2.2 GHz occupied bandwidth).
    """

    name: str = "TDSR P452 (full-UWB)"
    f_start_hz: float = 3.1e9      # 47 CFR 15.510 UWB band lower edge
    f_stop_hz: float = 5.3e9       # P452 upper edge (BOM addendum)
    n_freq: int = 128
    raw_snr_db: float = 11.0
    # Per-position, per-frequency-sample SNR, derived from the Week 3 waterfall
    # (drywall, 3 m) so both models share one noise floor:
    #   single-pulse raw SNR                       -17.7 dB
    #   in-frame coherent pulse integration        +53.0 dB
    #   integration-efficiency allowance            -3.0 dB
    #   = per-frame (per aperture position)        +32.3 dB
    #   less range-compression gain this model
    #     performs itself (10log10 128)            -21.1 dB
    #   = per-sample input SNR                     +11.2 dB -> 11.0
    # Week 3's slow-time term (+23 dB) is NOT added: the aperture sum performs
    # exactly that integration.

    @property
    def bandwidth_hz(self) -> float:
        return self.f_stop_hz - self.f_start_hz

    @property
    def f_center_hz(self) -> float:
        return 0.5 * (self.f_start_hz + self.f_stop_hz)

    @property
    def wavelength_center_m(self) -> float:
        return C / self.f_center_hz

    @property
    def wavelength_min_m(self) -> float:
        """Shortest wavelength in the band -- the one that sets phase tolerance."""
        return C / self.f_stop_hz

    @property
    def range_resolution_m(self) -> float:
        """Two-way range resolution c / 2B."""
        return C / (2.0 * self.bandwidth_hz)

    @property
    def freq_step_hz(self) -> float:
        return self.bandwidth_hz / (self.n_freq - 1)

    @property
    def max_unambiguous_range_m(self) -> float:
        return C / (2.0 * self.freq_step_hz)

    def frequencies(self) -> np.ndarray:
        return np.linspace(self.f_start_hz, self.f_stop_hz, self.n_freq)


RADARS = {
    "tdsr_p452": RadarConfig(),
    "novelda_x7": RadarConfig(
        name="Novelda X7 Radar Direct (fallback)",
        # ~7.29 GHz centre, ~0.46 GHz effective bandwidth (BOM addendum).
        f_start_hz=7.06e9,
        f_stop_hz=7.52e9,
        n_freq=64,
    ),
}


# One-way wall loss at the operating band, from the Week 2 attenuation table
# (NIST IR 6055 plus 3-8 GHz survey data). The radar pays it twice.
WALLS = {
    "none":          dict(one_way_db=0.0,  label="free space (no wall)"),
    "drywall":       dict(one_way_db=2.0,  label="13 mm gypsum drywall"),
    "interior_wall": dict(one_way_db=4.0,  label="interior wall (2x drywall + studs)"),
    "wood_door":     dict(one_way_db=3.5,  label="solid wood door (~4 cm)"),
    "cmu_block":     dict(one_way_db=15.0, label="hollow CMU block 20 cm + render"),
    "brick":         dict(one_way_db=22.0, label="clay brick (~10 cm)"),
    "concrete_20cm": dict(one_way_db=30.0, label="20 cm poured concrete"),
}


@dataclass(frozen=True)
class SceneConfig:
    """Measurement geometry.

    The radar walks along the x axis at y = 0. The wall is a plane at
    y = wall_y_m. Targets sit behind it at y > wall_y_m.
    """

    wall: str = "drywall"
    wall_y_m: float = 0.5
    target_range_m: float = 3.0
    wall_return_amplitude: float = 30.0   # specular front face, ~30x a body echo
    clutter_points: Tuple[Tuple[float, float, float], ...] = (
        (-0.85, 2.30, 0.6),
        (0.95, 3.70, 0.5),
        (0.40, 4.20, 0.4),
    )
    seed: int = 20260701

    def two_way_wall_amplitude(self) -> float:
        return 10.0 ** (-2.0 * WALLS[self.wall]["one_way_db"] / 20.0)


@dataclass(frozen=True)
class ApertureConfig:
    """The synthetic aperture swept out by the walking operator."""

    length_m: float = 1.2
    samples_per_wavelength: float = 4.0
    # Grating lobes are avoided at lambda_min/4 (~14 mm for the P452). A 1 m/s
    # walk at 200 frames/s samples every 5 mm, so this is comfortable.

    def sample_spacing_m(self, radar: RadarConfig) -> float:
        return radar.wavelength_min_m / self.samples_per_wavelength

    def n_positions(self, radar: RadarConfig) -> int:
        return max(2, int(round(self.length_m / self.sample_spacing_m(radar))) + 1)

    def positions(self, radar: RadarConfig) -> np.ndarray:
        """Nominal (assumed) antenna positions, shape (n_pos, 2) as (x, y)."""
        x = np.linspace(-self.length_m / 2.0, self.length_m / 2.0,
                        self.n_positions(radar))
        return np.column_stack([x, np.zeros_like(x)])

    def duration_s(self, walking_speed_ms: float = 1.0) -> float:
        return self.length_m / walking_speed_ms
