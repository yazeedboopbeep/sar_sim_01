"""Point-scatterer target models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class Target:
    """An isotropic point scatterer.

    `phase_rad` is the scatterer's own reflection phase. Real body parts are not
    phase-locked to each other, and two IN-PHASE equal scatterers merge into one
    lobe however long the aperture is -- coherent-imaging interference, not a
    resolution limit. Separability is therefore always evaluated by averaging
    over random scatterer phases.
    """

    x_m: float
    y_m: float
    amplitude: float = 1.0
    label: str = ""
    phase_rad: float = 0.0

    @property
    def complex_amplitude(self) -> complex:
        return self.amplitude * np.exp(1j * self.phase_rad)


def single_point(range_m: float = 3.0) -> List[Target]:
    """One scatterer -- measures the point-spread function directly."""
    return [Target(0.0, range_m, 1.0, "point")]


def two_points(range_m: float = 3.0, separation_m: float = 0.10,
               phases: np.ndarray | None = None) -> List[Target]:
    """Torso-versus-limb case: the 10 cm separability test from BR-03."""
    p = phases if phases is not None else np.zeros(2)
    return [
        Target(-separation_m / 2.0, range_m, 1.0, "torso", float(p[0])),
        Target(+separation_m / 2.0, range_m, 1.0, "arm", float(p[1])),
    ]


def body(range_m: float = 3.0, phases: np.ndarray | None = None) -> List[Target]:
    """Coarse five-point body (head, torso, two arms, legs).

    Amplitudes are relative, not calibrated RCS. Used for the qualitative
    imaging figures and for the autofocus test, where several scatterers of
    unequal strength make the problem realistic rather than trivial.
    """
    p = phases if phases is not None else np.zeros(5)
    spec = [
        (0.00, 0.00, 0.5, "head"),
        (0.00, 0.18, 1.0, "torso"),
        (-0.22, 0.20, 0.6, "arm_L"),
        (+0.22, 0.16, 0.6, "arm_R"),
        (+0.05, 0.45, 0.8, "legs"),
    ]
    return [Target(dx, range_m + dy, a, name, float(p[i]))
            for i, (dx, dy, a, name) in enumerate(spec)]
