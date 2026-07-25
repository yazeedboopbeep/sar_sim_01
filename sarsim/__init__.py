"""sarsim -- through-wall synthetic-aperture radar simulator.

v0.2 (Week 5). Changes from the Week 4 v0.1 single-file simulator:

  * split into modules with a pytest suite
  * fast range-compressed back-projection replacing the direct double sum
  * antenna-position error models (jitter and drift) -- the point of Week 5
  * autofocus, so a positioning shortfall has a route out rather than a wall

The open requirement this version exists to settle is SR-05: how accurately the
antenna position must be known for coherent aperture synthesis to work.
"""

from .config import (C, RADARS, WALLS, ApertureConfig, RadarConfig, SceneConfig)
from .targets import Target, body, single_point, two_points
from .forward import (background_subtract, calibrate_noise_power, simulate_echoes)
from .imaging import (Image, back_project, back_project_direct, cross_range_cut,
                      default_grid, range_compress, range_profile)
from .metrics import (RAYLEIGH_DIP_DB, image_snr_db, minus_3db_width,
                      peak_amplitude_at, peak_displacement_m, required_aperture,
                      resolved, theoretical_cross_range_res)
from .positioning import (apply_drift, apply_jitter, phase_error_rad,
                          predicted_loss_db, tolerance_m,
                          vio_error_over_aperture)
from .autofocus import autofocus, estimate_position_errors

__version__ = "0.2.0"

__all__ = [
    "C", "RADARS", "WALLS", "ApertureConfig", "RadarConfig", "SceneConfig",
    "Target", "body", "single_point", "two_points",
    "background_subtract", "calibrate_noise_power", "simulate_echoes",
    "Image", "back_project", "back_project_direct", "cross_range_cut",
    "default_grid", "range_compress", "range_profile",
    "RAYLEIGH_DIP_DB", "image_snr_db", "minus_3db_width", "peak_amplitude_at",
    "peak_displacement_m", "required_aperture", "resolved",
    "theoretical_cross_range_res",
    "apply_drift", "apply_jitter", "phase_error_rad", "predicted_loss_db",
    "tolerance_m", "vio_error_over_aperture",
    "autofocus", "estimate_position_errors",
]
