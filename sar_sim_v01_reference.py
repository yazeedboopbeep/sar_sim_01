"""
sar_sim.py -- Through-wall synthetic-aperture resolution simulator.

Week 4 deliverable, PoseDetection / through-wall human pose estimation.

WHAT QUESTION THIS ANSWERS
--------------------------
The Week 3 link-budget model (link_budget_sim) answered "can the receiver HEAR a
person behind a wall on legal UWB power?"  -> yes, comfortably, through drywall
at a few metres.  It says nothing about whether the echo can be turned into a
SHAPE.  That is governed by a different law:

        angular resolution  ~  wavelength / aperture width

A shoulder-worn unit is ~0.2 m wide, which is 5-15x too small to separate body
parts ~10 cm apart at 3 m.  The proposed fix is synthetic aperture radar (SAR):
the wearer's own walking motion sweeps a small antenna through space, and the
echoes are combined coherently into one large VIRTUAL aperture.

This simulator tests that mechanism in software, before the ~$2K radar purchase:

    Q1  Does coherent back-projection over a walked aperture actually deliver
        the theoretical cross-range resolution  d_cr = lambda*R / (2*L) ?
    Q2  How long an aperture L is needed to resolve two scatterers 10 cm apart
        at 3 m through a wall?
    Q3  Does the radar under procurement (TDSR P452, 2.2 GHz BW) resolve in
        RANGE what the fallback (Novelda X7, ~0.46 GHz BW) cannot?

SCOPE OF THIS VERSION (v0.1, Week 4) -- deliberately idealised
-------------------------------------------------------------
Modelled:      free-space two-way spreading, wideband stepped-frequency signal
               model, isotropic point scatterers, homogeneous wall as a two-way
               transmission loss plus a specular front-face return, static
               furniture clutter, empty-room background subtraction, additive
               white Gaussian receiver noise scaled to the Week 3 raw SNR,
               time-domain delay-and-sum back-projection, -3 dB resolution and
               Rayleigh-dip separability metrics.

NOT modelled (Week 5 work -- see README / report):
               antenna position error (VIO/IMU drift) -- the real test 0c,
               refraction and multipath inside the wall slab,
               target motion / breathing during aperture synthesis (Risk 3),
               aspect-dependent (non-isotropic) body scattering,
               CFAR detection and point-cloud extraction,
               performance (this is a readable reference implementation, not a
               fast one).

Every physical constant carries a source note, same convention as Week 3.

Usage:
    python3 sar_sim.py image      --wall drywall --range 3.0 --aperture 1.2
    python3 sar_sim.py sweep      --wall drywall --range 3.0
    python3 sar_sim.py separation --wall drywall --range 3.0 --aperture 1.2
    python3 sar_sim.py radar-compare
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field, replace
from typing import List, Tuple

import numpy as np

C = 299_792_458.0  # speed of light, m/s


# ----------------------------------------------------------------------------
# 1. Configuration (every constant carries a source note)
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class RadarConfig:
    """Radar front-end parameters.

    Defaults = TDSR P452 full-UWB dev kit, the preferred platform in the
    Procurement/BOM addendum (2026-07-02): 3.1-5.3 GHz, 2.2 GHz occupied
    bandwidth. Novelda X7 Radar Direct is the fallback and is available via
    RADARS['novelda_x7'].
    """
    name: str = "TDSR P452 (full-UWB)"
    f_start_hz: float = 3.1e9      # 47 CFR 15.510 UWB band lower edge
    f_stop_hz: float = 5.3e9       # P452 upper edge (BOM addendum)
    n_freq: int = 128              # frequency samples across the band
    raw_snr_db: float = 11.0
    # Per-position, per-frequency-sample SNR entering the imaging chain,
    # DERIVED FROM THE WEEK 3 LINK BUDGET (drywall, 3 m) so the two models share
    # one noise floor:
    #     single-pulse raw SNR                    -17.7 dB
    #     in-frame coherent pulse integration     +53.0 dB  (2e5 pulses/frame)
    #     integration-efficiency allowance         -3.0 dB
    #     = per-frame (per aperture position)     +32.3 dB
    #     minus range-compression gain already
    #       counted by this model's coherent sum
    #       over 128 frequency bins (10log10 128) -21.1 dB
    #     = per-sample input SNR                  +11.2 dB  -> rounded to 11.0
    # Week 3's slow-time integration term (+23 dB over 200 frames) is NOT added
    # here: the aperture sum below performs exactly that integration, and
    # counting it twice would inflate the result.

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
        return C / self.f_stop_hz

    @property
    def range_resolution_m(self) -> float:
        """Classical two-way range resolution c / (2B)."""
        return C / (2.0 * self.bandwidth_hz)

    def frequencies(self) -> np.ndarray:
        return np.linspace(self.f_start_hz, self.f_stop_hz, self.n_freq)


RADARS = {
    "tdsr_p452": RadarConfig(),
    "novelda_x7": RadarConfig(
        name="Novelda X7 Radar Direct (fallback)",
        # X7 centre ~7.29 GHz with ~0.46 GHz effective bandwidth
        # (BOM addendum, 2026-07-02 -> ~33 cm range resolution).
        f_start_hz=7.06e9,
        f_stop_hz=7.52e9,
        n_freq=64,
    ),
}


# One-way wall loss at the operating band, from the Week 2 attenuation
# reference table (NIST IR 6055 + 3-8 GHz survey data). The radar pays it twice.
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
    """Geometry of the measurement.

    Coordinates: the radar walks along the x axis at y = 0. The wall is a plane
    at y = wall_y_m. Targets sit behind it at y > wall_y_m.
    """
    wall: str = "drywall"
    wall_y_m: float = 0.5          # standoff: operator 0.5 m from the wall face
    target_range_m: float = 3.0    # target range from the radar track
    wall_return_amplitude: float = 30.0   # specular front-face return; ~30x a
                                          # body scatterer. Static -> removed by
                                          # background subtraction.
    clutter_points: Tuple[Tuple[float, float, float], ...] = (
        # (x, y, amplitude) static furniture behind the wall
        (-0.85, 2.30, 0.6),
        (0.95, 3.70, 0.5),
        (0.40, 4.20, 0.4),
    )
    seed: int = 20260701


@dataclass(frozen=True)
class ApertureConfig:
    """The synthetic aperture swept out by the walking operator."""
    length_m: float = 1.2
    # Spatial sampling. Grating lobes are avoided at lambda_min/4; the P452 at
    # 5.3 GHz gives ~14 mm. A 1 m/s walk with a 200 fps frame rate samples every
    # 5 mm, so this is comfortably achievable.
    samples_per_wavelength: float = 4.0

    def positions(self, radar: RadarConfig) -> np.ndarray:
        step = radar.wavelength_min_m / self.samples_per_wavelength
        n = max(2, int(round(self.length_m / step)) + 1)
        return np.linspace(-self.length_m / 2.0, self.length_m / 2.0, n)


@dataclass
class Target:
    """An isotropic point scatterer. A body is modelled as several of these.

    `phase_rad` is the scatterer's own reflection phase. Real body parts are not
    phase-locked to each other, and two IN-PHASE equal scatterers merge into one
    lobe no matter how long the aperture is (coherent-imaging interference, not a
    resolution limit). Separability is therefore evaluated by averaging over
    random scatterer phases -- see `run_separation`.
    """
    x_m: float
    y_m: float
    amplitude: float = 1.0
    label: str = ""
    phase_rad: float = 0.0


def torso_and_arm(range_m: float, separation_m: float = 0.10) -> List[Target]:
    """Two scatterers separated in cross-range -- the canonical 'can I tell a
    limb from a torso?' test. 10 cm is the body-part separation the R&D briefing
    sets as the resolution requirement."""
    return [
        Target(-separation_m / 2.0, range_m, 1.0, "torso"),
        Target(+separation_m / 2.0, range_m, 1.0, "arm"),
    ]


def single_point(range_m: float) -> List[Target]:
    """One scatterer -- measures the point-spread function directly."""
    return [Target(0.0, range_m, 1.0, "point")]


def crouching_person(range_m: float) -> List[Target]:
    """Coarse five-point body model (head, torso, two arms, legs) for the
    qualitative imaging figure. Amplitudes are relative, not calibrated RCS."""
    return [
        Target(0.00, range_m + 0.00, 0.5, "head"),
        Target(0.00, range_m + 0.18, 1.0, "torso"),
        Target(-0.22, range_m + 0.20, 0.6, "arm_L"),
        Target(+0.22, range_m + 0.16, 0.6, "arm_R"),
        Target(+0.05, range_m + 0.45, 0.8, "legs"),
    ]


# ----------------------------------------------------------------------------
# 2. Forward signal model
# ----------------------------------------------------------------------------

def wall_two_way_amplitude(wall: str) -> float:
    """Linear amplitude factor for the two-way wall transit."""
    one_way_db = WALLS[wall]["one_way_db"]
    return 10.0 ** (-2.0 * one_way_db / 20.0)


def simulate_echoes(
    targets: List[Target],
    radar: RadarConfig,
    scene: SceneConfig,
    aperture: ApertureConfig,
    include_targets: bool = True,
    add_noise: bool = True,
    rng: np.random.Generator | None = None,
    noise_power: float | None = None,
) -> np.ndarray:
    """Stepped-frequency raw data cube s[n_position, n_frequency].

    Monostatic model. For a scatterer at range R from aperture position n, the
    two-way phase is exp(-j*4*pi*f*R/c) and the amplitude falls as 1/R^2
    (equivalent to the R^-4 power law of the radar equation used in Week 3).
    """
    if rng is None:
        rng = np.random.default_rng(scene.seed)

    freqs = radar.frequencies()
    xs = aperture.positions(radar)
    s = np.zeros((xs.size, freqs.size), dtype=np.complex128)


    def accumulate(px: float, py: float, amp: complex, through_wall: bool) -> None:
        r = np.hypot(xs - px, py)                      # (n_pos,)
        a = amp / np.maximum(r, 1e-3) ** 2
        if through_wall:
            a = a * wall_two_way_amplitude(scene.wall)
        # two-way propagation phase: exp(-j * 4 * pi * f * R / c)
        phase = np.exp(-2j * np.pi * 2.0 * np.outer(r, freqs) / C)
        s[...] += a[:, None] * phase

    # Static wall front face: a line of specular scatterers along the wall plane.
    for wx in np.linspace(-1.6, 1.6, 33):
        accumulate(wx, scene.wall_y_m, scene.wall_return_amplitude / 33.0, False)

    # Static furniture behind the wall.
    for cx, cy, camp in scene.clutter_points:
        accumulate(cx, cy, camp, True)

    # The people we actually want.
    if include_targets:
        for t in targets:
            accumulate(t.x_m, t.y_m,
                       t.amplitude * np.exp(1j * t.phase_rad), True)

    if add_noise:
        # Reference power = mean power of the target-only return, so that
        # raw_snr_db is the per-sample SNR of the signal of interest.
        # Pass `noise_power` explicitly to hold the receiver noise floor FIXED
        # while the wall changes -- that is what makes a wall-to-wall SNR
        # comparison meaningful (see calibrate_noise_power).
        if noise_power is None:
            ref = _target_only_power(targets, radar, scene, aperture)
            noise_power = ref / (10.0 ** (radar.raw_snr_db / 10.0))
        sigma = np.sqrt(noise_power / 2.0)
        s = s + rng.normal(0, sigma, s.shape) + 1j * rng.normal(0, sigma, s.shape)

    return s


def calibrate_noise_power(
    targets: List[Target],
    radar: RadarConfig,
    scene: SceneConfig,
    aperture: ApertureConfig,
) -> float:
    """Absolute receiver noise power implied by `radar.raw_snr_db` for THIS
    scene. Hold this fixed across walls so the wall loss actually shows up in
    the image SNR instead of being normalised away."""
    ref = _target_only_power(targets, radar, scene, aperture)
    return ref / (10.0 ** (radar.raw_snr_db / 10.0))


def _target_only_power(
    targets: List[Target],
    radar: RadarConfig,
    scene: SceneConfig,
    aperture: ApertureConfig,
) -> float:
    """Mean per-sample power of the target return alone (noise calibration)."""
    if not targets:
        return 1.0
    xs = aperture.positions(radar)
    total = 0.0
    for t in targets:
        r = np.hypot(xs - t.x_m, t.y_m)
        a = t.amplitude / np.maximum(r, 1e-3) ** 2 * wall_two_way_amplitude(scene.wall)
        total += float(np.mean(np.abs(a) ** 2))
    return total


# ----------------------------------------------------------------------------
# 3. Clutter cancellation
# ----------------------------------------------------------------------------

def background_subtract(scene_data: np.ndarray, empty_data: np.ndarray) -> np.ndarray:
    """Empty-room background subtraction.

    This is the clutter-cancellation method that test 0a can actually execute on
    the bench (capture the empty room, then capture with a person in it). It is
    the OPTIMISTIC bound: it assumes the static scene is perfectly repeatable.
    Week 5 replaces it with a reference-free method (slow-time high-pass or
    SVD/subspace removal) and measures the penalty.
    """
    return scene_data - empty_data


# ----------------------------------------------------------------------------
# 4. Imaging: delay-and-sum back-projection
# ----------------------------------------------------------------------------

@dataclass
class ImageGrid:
    x: np.ndarray
    y: np.ndarray
    values: np.ndarray = field(default=None, repr=False)

    @property
    def magnitude_db(self) -> np.ndarray:
        mag = np.abs(self.values)
        peak = mag.max() if mag.max() > 0 else 1.0
        return 20.0 * np.log10(np.maximum(mag / peak, 1e-6))


def back_project(
    data: np.ndarray,
    radar: RadarConfig,
    aperture: ApertureConfig,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
) -> ImageGrid:
    """Coherent delay-and-sum back-projection.

        I(p) = sum_n sum_f  s[n,f] * exp(+j 4 pi f R_n(p) / c)

    For each pixel and aperture position the model phase is removed and the
    result summed; energy adds coherently only where a scatterer actually is.
    This is the operation that turns a walked path into a virtual aperture.

    Straightforward triple loop collapsed to one matrix product per aperture
    position -- readable, not optimised. Week 5: vectorise / chunk / GPU.
    """
    freqs = radar.frequencies()
    xs = aperture.positions(radar)
    XX, YY = np.meshgrid(grid_x, grid_y, indexing="xy")
    flat_x, flat_y = XX.ravel(), YY.ravel()

    acc = np.zeros(flat_x.size, dtype=np.complex128)
    for n, ax in enumerate(xs):
        r = np.hypot(flat_x - ax, flat_y)                    # (n_pixel,)
        phase = np.exp(2j * np.pi * 2.0 * np.outer(r, freqs) / C)
        acc += phase @ data[n]

    return ImageGrid(grid_x, grid_y, acc.reshape(XX.shape))


def cross_range_cut(
    data: np.ndarray,
    radar: RadarConfig,
    aperture: ApertureConfig,
    range_m: float,
    half_width_m: float = 1.0,
    n_points: int = 801,
) -> Tuple[np.ndarray, np.ndarray]:
    """Back-project onto a 1-D cut at fixed range -- the cheap way to measure
    cross-range resolution when sweeping aperture length."""
    x = np.linspace(-half_width_m, half_width_m, n_points)
    img = back_project(data, radar, aperture, x, np.array([range_m]))
    return x, np.abs(img.values[0])


def range_profile(data: np.ndarray, radar: RadarConfig,
                  position_index: int | None = None,
                  n_fft: int = 2048) -> Tuple[np.ndarray, np.ndarray]:
    """Fast-time range profile at one aperture position.

    Inverse-DFT of the stepped-frequency response, windowed to suppress
    sidelobes. Shows RANGE resolution (c/2B) independently of the aperture.
    """
    if position_index is None:
        position_index = data.shape[0] // 2
    spec = data[position_index] * np.hanning(radar.n_freq)
    profile = np.abs(np.fft.ifft(spec, n=n_fft))
    df = radar.bandwidth_hz / (radar.n_freq - 1)
    max_range = C / (2.0 * df)
    r = np.linspace(0.0, max_range, n_fft, endpoint=False)
    return r, profile


# ----------------------------------------------------------------------------
# 5. Metrics
# ----------------------------------------------------------------------------

def theoretical_cross_range_res(radar: RadarConfig, range_m: float,
                                aperture_len_m: float) -> float:
    """d_cr = lambda_c * R / (2 * L)   (two-way / monostatic SAR)."""
    return radar.wavelength_center_m * range_m / (2.0 * aperture_len_m)


def required_aperture(radar: RadarConfig, range_m: float, target_res_m: float) -> float:
    """Invert the resolution law: L = lambda_c * R / (2 * d)."""
    return radar.wavelength_center_m * range_m / (2.0 * target_res_m)


def minus_3db_width(x: np.ndarray, profile: np.ndarray) -> float:
    """-3 dB (half-power) main-lobe width of a single-point response."""
    p = profile / profile.max()
    peak_i = int(np.argmax(p))
    half = 1.0 / np.sqrt(2.0)

    def edge(direction: int) -> float:
        i = peak_i
        while 0 < i < p.size - 1:
            j = i + direction
            if p[j] < half:
                # linear interpolation onto the crossing
                t = (half - p[j]) / (p[i] - p[j] + 1e-12)
                return x[j] + t * (x[i] - x[j])
            i = j
        return x[0] if direction < 0 else x[-1]

    return abs(edge(+1) - edge(-1))


RAYLEIGH_DIP_DB = 1.34   # 26.5% intensity dip -- the classical Rayleigh criterion


def resolved(x: np.ndarray, profile: np.ndarray, sep_m: float,
             dip_db: float = RAYLEIGH_DIP_DB) -> Tuple[bool, float]:
    """Rayleigh two-target separability.

    Two scatterers count as RESOLVED if the profile shows two maxima near the
    true positions with a saddle between them at least `dip_db` down from the
    weaker peak. The default is the classical Rayleigh criterion: the saddle
    falls to 73.5% of peak intensity, i.e. 1.34 dB. A dip above ~3 dB is
    comfortable rather than marginal.

    Returns (resolved, measured_dip_db).
    """
    p = profile / profile.max()
    left_mask = (x >= -sep_m) & (x <= 0.0)
    right_mask = (x >= 0.0) & (x <= sep_m)
    if not left_mask.any() or not right_mask.any():
        return False, 0.0
    l_peak = p[left_mask].max()
    r_peak = p[right_mask].max()
    centre_mask = (x >= -sep_m / 2.0) & (x <= sep_m / 2.0)
    valley = p[centre_mask].min()
    dip = 20.0 * np.log10(min(l_peak, r_peak) / max(valley, 1e-12))
    return bool(dip >= dip_db), float(dip)


def image_snr_at_target_db(image: ImageGrid, target: Target,
                           exclude_m: float = 0.3) -> float:
    """Matched-location image SNR: the coherent response AT the known target
    position, over the RMS floor measured away from it.

    Unlike a peak-in-a-box metric this degrades gracefully below 0 dB once the
    target is buried, so it is the right metric for a wall-by-wall comparison.
    """
    XX, YY = np.meshgrid(image.x, image.y, indexing="xy")
    mag = np.abs(image.values)
    d = np.hypot(XX - target.x_m, YY - target.y_m)
    iy, ix = np.unravel_index(np.argmin(d), d.shape)
    signal = mag[iy, ix]
    far = d > exclude_m
    if not far.any():
        return float("nan")
    floor = np.sqrt(np.mean(mag[far] ** 2))
    return float(20.0 * np.log10(max(signal, 1e-12) / max(floor, 1e-12)))


def image_snr_db(image: ImageGrid, targets: List[Target],
                 box_m: float = 0.15) -> float:
    """Peak target response over the noise/artefact floor away from targets."""
    XX, YY = np.meshgrid(image.x, image.y, indexing="xy")
    mag = np.abs(image.values)
    target_mask = np.zeros_like(mag, dtype=bool)
    for t in targets:
        target_mask |= (np.abs(XX - t.x_m) < box_m) & (np.abs(YY - t.y_m) < box_m)
    if not target_mask.any() or target_mask.all():
        return float("nan")
    peak = mag[target_mask].max()
    floor = np.sqrt(np.mean(mag[~target_mask] ** 2))
    return float(20.0 * np.log10(peak / max(floor, 1e-12)))


# ----------------------------------------------------------------------------
# 6. High-level experiments
# ----------------------------------------------------------------------------

def run_image(radar: RadarConfig, scene: SceneConfig, aperture: ApertureConfig,
              targets: List[Target]):
    """Full 2-D image of a scene, with and without clutter cancellation."""
    rng = np.random.default_rng(scene.seed)
    with_target = simulate_echoes(targets, radar, scene, aperture,
                                  include_targets=True, rng=rng)
    empty = simulate_echoes(targets, radar, scene, aperture,
                            include_targets=False,
                            rng=np.random.default_rng(scene.seed + 1))
    cleaned = background_subtract(with_target, empty)

    gx = np.linspace(-1.2, 1.2, 241)
    gy = np.linspace(0.3, 4.5, 211)
    raw_img = back_project(with_target, radar, aperture, gx, gy)
    clean_img = back_project(cleaned, radar, aperture, gx, gy)
    return with_target, cleaned, raw_img, clean_img


def run_aperture_sweep(radar: RadarConfig, scene: SceneConfig,
                       lengths_m: np.ndarray, range_m: float):
    """Measured vs theoretical cross-range resolution across aperture length."""
    rows = []
    for L in lengths_m:
        ap = ApertureConfig(length_m=float(L))
        targets = single_point(range_m)
        rng = np.random.default_rng(scene.seed)
        d = simulate_echoes(targets, radar, scene, ap, True, rng=rng)
        e = simulate_echoes(targets, radar, scene, ap, False,
                            rng=np.random.default_rng(scene.seed + 1))
        x, prof = cross_range_cut(background_subtract(d, e), radar, ap, range_m)
        rows.append(dict(
            aperture_m=float(L),
            n_positions=int(ap.positions(radar).size),
            measured_res_m=minus_3db_width(x, prof),
            theory_res_m=theoretical_cross_range_res(radar, range_m, float(L)),
        ))
    return rows


def run_separation(radar: RadarConfig, scene: SceneConfig,
                   aperture: ApertureConfig, range_m: float,
                   separation_m: float = 0.10,
                   n_realizations: int = 32,
                   half_width_m: float = 0.45,
                   n_points: int = 601):
    """Two-scatterer separability, averaged over random scatterer phases.

    Averaging matters: a single realisation of two equal, in-phase scatterers
    merges into one lobe (constructive interference) and a single realisation of
    two anti-phase scatterers shows a false null. Neither is a statement about
    resolution. The phase-averaged magnitude profile is.
    """
    phase_rng = np.random.default_rng(scene.seed + 99)
    acc = None
    x = None
    for k in range(n_realizations):
        ph = phase_rng.uniform(0, 2 * np.pi, 2)
        targets = [
            Target(-separation_m / 2.0, range_m, 1.0, "torso", float(ph[0])),
            Target(+separation_m / 2.0, range_m, 1.0, "arm", float(ph[1])),
        ]
        d = simulate_echoes(targets, radar, scene, aperture, True,
                            rng=np.random.default_rng(scene.seed + k))
        e = simulate_echoes(targets, radar, scene, aperture, False,
                            rng=np.random.default_rng(scene.seed + 1000 + k))
        x, prof = cross_range_cut(background_subtract(d, e), radar, aperture,
                                  range_m, half_width_m=half_width_m,
                                  n_points=n_points)
        acc = prof ** 2 if acc is None else acc + prof ** 2
    prof = np.sqrt(acc / n_realizations)
    ok, dip = resolved(x, prof, separation_m)
    return x, prof, ok, dip


# ----------------------------------------------------------------------------
# 7. CLI
# ----------------------------------------------------------------------------

def _banner(radar: RadarConfig, scene: SceneConfig, aperture: ApertureConfig) -> str:
    w = WALLS[scene.wall]
    return (
        f"radar        : {radar.name}\n"
        f"band         : {radar.f_start_hz/1e9:.2f}-{radar.f_stop_hz/1e9:.2f} GHz"
        f"  (B={radar.bandwidth_hz/1e9:.2f} GHz, fc={radar.f_center_hz/1e9:.2f} GHz,"
        f" lambda_c={radar.wavelength_center_m*100:.1f} cm)\n"
        f"range res    : {radar.range_resolution_m*100:.1f} cm  (c/2B)\n"
        f"wall         : {scene.wall} -- {w['label']}, {w['one_way_db']:.0f} dB one-way"
        f" / {2*w['one_way_db']:.0f} dB two-way\n"
        f"aperture     : {aperture.length_m:.2f} m, "
        f"{aperture.positions(radar).size} positions "
        f"(spacing {radar.wavelength_min_m/aperture.samples_per_wavelength*1000:.1f} mm)\n"
        f"raw SNR/samp : {radar.raw_snr_db:.1f} dB\n"
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["image", "sweep", "separation", "radar-compare"])
    p.add_argument("--radar", default="tdsr_p452", choices=sorted(RADARS))
    p.add_argument("--wall", default="drywall", choices=sorted(WALLS))
    p.add_argument("--range", type=float, default=3.0, dest="range_m")
    p.add_argument("--aperture", type=float, default=1.2, dest="aperture_m")
    p.add_argument("--separation", type=float, default=0.10)
    p.add_argument("--snr", type=float, default=None, help="override raw SNR (dB)")
    a = p.parse_args(argv)

    radar = RADARS[a.radar]
    if a.snr is not None:
        radar = replace(radar, raw_snr_db=a.snr)
    scene = SceneConfig(wall=a.wall, target_range_m=a.range_m)
    aperture = ApertureConfig(length_m=a.aperture_m)

    print(_banner(radar, scene, aperture))

    if a.mode == "image":
        targets = crouching_person(a.range_m)
        _, _, raw_img, clean_img = run_image(radar, scene, aperture, targets)
        print(f"image SNR before clutter cancellation : "
              f"{image_snr_db(raw_img, targets):6.1f} dB")
        print(f"image SNR after  clutter cancellation : "
              f"{image_snr_db(clean_img, targets):6.1f} dB")

    elif a.mode == "sweep":
        rows = run_aperture_sweep(radar, scene,
                                  np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.6, 2.0]),
                                  a.range_m)
        print(f"{'L (m)':>7} {'#pos':>6} {'measured -3dB (cm)':>20} "
              f"{'theory (cm)':>13} {'ratio':>7}")
        for r in rows:
            print(f"{r['aperture_m']:7.2f} {r['n_positions']:6d} "
                  f"{r['measured_res_m']*100:20.1f} {r['theory_res_m']*100:13.1f} "
                  f"{r['measured_res_m']/r['theory_res_m']:7.2f}")
        need = required_aperture(radar, a.range_m, 0.10)
        print(f"\naperture needed for 10 cm cross-range at {a.range_m:.1f} m : "
              f"{need:.2f} m  ({need/0.2:.1f}x a 0.2 m shoulder unit)")

    elif a.mode == "separation":
        x, prof, ok, dip = run_separation(radar, scene, aperture,
                                          a.range_m, a.separation)
        print(f"two scatterers {a.separation*100:.0f} cm apart at {a.range_m:.1f} m")
        print(f"theoretical cross-range resolution : "
              f"{theoretical_cross_range_res(radar, a.range_m, a.aperture_m)*100:.1f} cm")
        print(f"measured dip between peaks         : {dip:.1f} dB")
        print(f"VERDICT: {'RESOLVED' if ok else 'NOT RESOLVED (single blob)'}")

    elif a.mode == "radar-compare":
        print(f"{'radar':<34} {'B (GHz)':>8} {'range res':>11} "
              f"{'L for 10cm @3m':>16}")
        for key in ("tdsr_p452", "novelda_x7"):
            r = RADARS[key]
            print(f"{r.name:<34} {r.bandwidth_hz/1e9:8.2f} "
                  f"{r.range_resolution_m*100:9.1f} cm "
                  f"{required_aperture(r, 3.0, 0.10):14.2f} m")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
