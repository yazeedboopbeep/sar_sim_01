"""Test suite for sarsim.

Week 4's verify.py checked the model against closed-form physics. That is still
here (test_physics.py style cases below), but the reason this is now a real
pytest suite is the refactor: the fast imager and the position-error models are
new code, and a Monte Carlo sweep that silently images the wrong thing produces
confident nonsense.
"""

from __future__ import annotations

import numpy as np
import pytest

from sarsim import (ApertureConfig, RADARS, SceneConfig, apply_drift,
                    apply_jitter, autofocus, back_project, back_project_direct,
                    background_subtract, calibrate_noise_power,
                    cross_range_cut, image_snr_db, minus_3db_width,
                    peak_displacement_m, predicted_loss_db, range_profile,
                    required_aperture, resolved, simulate_echoes, single_point,
                    theoretical_cross_range_res, tolerance_m, two_points)

RADAR = RADARS["tdsr_p452"]
SCENE = SceneConfig(wall="drywall", target_range_m=3.0)


def clean_capture(targets, aperture=None, positions_true=None,
                  scene=SCENE, radar=RADAR, noise=False):
    """Scene capture minus empty-room capture, on given true positions."""
    ap = aperture or ApertureConfig(length_m=1.2)
    pos = ap.positions(radar) if positions_true is None else positions_true
    d = simulate_echoes(targets, radar, scene, pos, True, add_noise=noise,
                        rng=np.random.default_rng(1))
    e = simulate_echoes(targets, radar, scene, pos, False, add_noise=noise,
                        rng=np.random.default_rng(2))
    return background_subtract(d, e)


# --------------------------------------------------------------- geometry ---

def test_aperture_sampling_is_grating_lobe_free():
    ap = ApertureConfig(length_m=1.2)
    assert ap.sample_spacing_m(RADAR) <= RADAR.wavelength_min_m / 4 + 1e-12
    assert ap.positions(RADAR).shape == (ap.n_positions(RADAR), 2)


def test_range_resolution_matches_c_over_2b():
    assert RADAR.range_resolution_m == pytest.approx(0.068, abs=0.002)
    assert RADARS["novelda_x7"].range_resolution_m == pytest.approx(0.326, abs=0.01)


# ---------------------------------------------------------------- imaging ---

def test_fast_back_projection_matches_direct_sum():
    """The refactor's headline claim, tested rather than asserted."""
    ap = ApertureConfig(length_m=0.6)
    data = clean_capture(single_point(3.0), ap)
    gx = np.linspace(-0.3, 0.3, 41)
    gy = np.linspace(2.85, 3.15, 21)

    fast = back_project(data, RADAR, ap.positions(RADAR), gx, gy)
    slow = back_project_direct(data, RADAR, ap.positions(RADAR), gx, gy)

    scale = np.abs(slow.values).max()
    err = np.abs(fast.values - slow.values).max() / scale
    assert err < 0.02, f"fast/direct mismatch {20*np.log10(err):.1f} dB"


def test_image_peak_lands_on_the_target():
    ap = ApertureConfig(length_m=1.2)
    data = clean_capture([*single_point(3.0)], ap)
    gx = np.linspace(-0.5, 0.5, 81)
    gy = np.linspace(2.6, 3.4, 65)
    img = back_project(data, RADAR, ap.positions(RADAR), gx, gy)
    px, py = img.peak_position()
    assert abs(px) < 0.03 and abs(py - 3.0) < 0.03


def test_range_profile_separates_targets_20cm_apart():
    from sarsim import Target
    ap = ApertureConfig(length_m=0.4)
    data = clean_capture([Target(0.0, 3.0), Target(0.0, 3.2)], ap)
    r, prof = range_profile(data, RADAR)
    m = (r > 2.5) & (r < 3.7)
    ok, dip = resolved(r[m] - 3.1, prof[m], 0.2)
    assert ok, f"dip only {dip:.2f} dB"


# ------------------------------------------------------------- resolution ---

@pytest.mark.parametrize("length_m", [0.4, 0.8, 1.2, 2.0])
def test_cross_range_resolution_follows_the_law(length_m):
    """-3 dB width should be 0.886 * lambda*R/2L for a uniform aperture."""
    ap = ApertureConfig(length_m=length_m)
    data = clean_capture(single_point(3.0), ap)
    x, prof = cross_range_cut(data, RADAR, ap.positions(RADAR), 3.0)
    ratio = minus_3db_width(x, prof) / theoretical_cross_range_res(RADAR, 3.0,
                                                                  length_m)
    assert 0.78 < ratio < 0.98


def test_required_aperture_for_10cm_at_3m():
    assert required_aperture(RADAR, 3.0, 0.10) == pytest.approx(1.07, abs=0.02)


def test_two_points_unresolved_below_and_resolved_above_the_requirement():
    need = required_aperture(RADAR, 3.0, 0.10)
    dips = {}
    for factor in (0.8, 1.2):
        ap = ApertureConfig(length_m=need * factor)
        acc = None
        rng = np.random.default_rng(7)
        for _ in range(12):
            data = clean_capture(two_points(3.0, 0.10, rng.uniform(0, 2*np.pi, 2)), ap)
            x, prof = cross_range_cut(data, RADAR, ap.positions(RADAR), 3.0)
            acc = prof**2 if acc is None else acc + prof**2
        dips[factor] = resolved(x, np.sqrt(acc), 0.10)[1]
    assert dips[0.8] < 1.34 <= dips[1.2]


# ------------------------------------------------------------- positioning ---

def test_zero_jitter_is_a_no_op():
    ap = ApertureConfig(length_m=1.2)
    pos = ap.positions(RADAR)
    assert np.array_equal(apply_jitter(pos, 0.0, np.random.default_rng(0)), pos)


def test_jitter_loss_matches_closed_form():
    """Simulated coherence loss should track 4.343 * sigma_phi^2."""
    ap = ApertureConfig(length_m=1.2)
    nominal = ap.positions(RADAR)
    gx = np.linspace(-0.2, 0.2, 33)
    gy = np.linspace(2.9, 3.1, 17)

    reference = np.abs(back_project(clean_capture(single_point(3.0), ap),
                                    RADAR, nominal, gx, gy).values).max()
    for sigma in (0.002, 0.004):
        rng = np.random.default_rng(11)
        losses = []
        for _ in range(6):
            true_pos = apply_jitter(nominal, sigma, rng)
            data = clean_capture(single_point(3.0), ap, positions_true=true_pos)
            peak = np.abs(back_project(data, RADAR, nominal, gx, gy).values).max()
            losses.append(-20 * np.log10(peak / reference))
        measured = float(np.mean(losses))
        predicted = predicted_loss_db(sigma, RADAR)
        assert abs(measured - predicted) < 0.35 * max(predicted, 1.0), (
            f"sigma={sigma*1000:.0f} mm: measured {measured:.2f} dB, "
            f"predicted {predicted:.2f} dB")


def test_tolerance_inverts_the_loss_law():
    d = tolerance_m(RADAR, max_loss_db=1.0)
    assert predicted_loss_db(d, RADAR) == pytest.approx(1.0, abs=1e-6)


def test_drift_moves_the_target_but_jitter_does_not():
    """Jitter dims, drift displaces -- the distinction the week turns on."""
    ap = ApertureConfig(length_m=1.2)
    nominal = ap.positions(RADAR)
    gx = np.linspace(-0.4, 0.4, 65)
    gy = np.linspace(2.7, 3.3, 49)
    target = single_point(3.0)[0]

    jit = clean_capture([target], ap,
                        positions_true=apply_jitter(nominal, 0.004,
                                                    np.random.default_rng(3)))
    dri = clean_capture([target], ap,
                        positions_true=apply_drift(nominal, 0.020))

    jit_shift = peak_displacement_m(back_project(jit, RADAR, nominal, gx, gy),
                                    target)
    dri_shift = peak_displacement_m(back_project(dri, RADAR, nominal, gx, gy),
                                    target)
    assert jit_shift < 0.02
    assert dri_shift > jit_shift


# --------------------------------------------------------------- autofocus ---

def test_autofocus_recovers_most_of_the_jitter_loss():
    ap = ApertureConfig(length_m=1.2)
    nominal = ap.positions(RADAR)
    gx = np.linspace(-0.3, 0.3, 49)
    gy = np.linspace(2.8, 3.2, 33)

    reference = np.abs(back_project(clean_capture(single_point(3.0), ap),
                                    RADAR, nominal, gx, gy).values).max()
    true_pos = apply_jitter(nominal, 0.005, np.random.default_rng(5))
    data = clean_capture(single_point(3.0), ap, positions_true=true_pos)

    before = np.abs(back_project(data, RADAR, nominal, gx, gy).values).max()
    fixed, _ = autofocus(data, RADAR, nominal, gx, gy)
    after = np.abs(back_project(fixed, RADAR, nominal, gx, gy).values).max()

    loss_before = -20 * np.log10(before / reference)
    loss_after = -20 * np.log10(after / reference)
    assert loss_after < loss_before / 2.0, (
        f"before {loss_before:.2f} dB, after {loss_after:.2f} dB")


def test_autofocus_is_harmless_when_positions_are_already_right():
    ap = ApertureConfig(length_m=1.2)
    nominal = ap.positions(RADAR)
    gx = np.linspace(-0.3, 0.3, 49)
    gy = np.linspace(2.8, 3.2, 33)
    data = clean_capture(single_point(3.0), ap)

    before = np.abs(back_project(data, RADAR, nominal, gx, gy).values).max()
    fixed, delta = autofocus(data, RADAR, nominal, gx, gy)
    after = np.abs(back_project(fixed, RADAR, nominal, gx, gy).values).max()

    assert np.max(np.abs(delta)) < 0.001            # sub-millimetre
    assert after >= before * 0.99


# ------------------------------------------------------------------- noise ---

def test_noise_calibration_gives_the_requested_snr():
    ap = ApertureConfig(length_m=0.6)
    pos = ap.positions(RADAR)
    targets = single_point(3.0)
    power = calibrate_noise_power(targets, RADAR, SCENE, pos)

    signal_only = simulate_echoes(targets, RADAR, SCENE, pos, True,
                                  add_noise=False)
    empty_only = simulate_echoes(targets, RADAR, SCENE, pos, False,
                                 add_noise=False)
    target_power = float(np.mean(np.abs(signal_only - empty_only) ** 2))
    assert 10 * np.log10(target_power / power) == pytest.approx(
        RADAR.raw_snr_db, abs=0.5)


def test_clutter_cancellation_improves_image_snr():
    ap = ApertureConfig(length_m=1.2)
    pos = ap.positions(RADAR)
    targets = single_point(3.0)
    gx = np.linspace(-0.6, 0.6, 65)
    gy = np.linspace(0.3, 4.0, 121)

    raw = simulate_echoes(targets, RADAR, SCENE, pos, True,
                          rng=np.random.default_rng(1))
    empty = simulate_echoes(targets, RADAR, SCENE, pos, False,
                            rng=np.random.default_rng(2))
    before = image_snr_db(back_project(raw, RADAR, pos, gx, gy), targets)
    after = image_snr_db(back_project(background_subtract(raw, empty),
                                      RADAR, pos, gx, gy), targets)
    assert after - before > 20.0
