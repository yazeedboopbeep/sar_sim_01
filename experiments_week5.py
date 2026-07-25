"""Week 5 experiments: how accurately must the antenna position be known?

SIM-6   position-error sweep (jitter), with and without autofocus
SIM-7   images at three error levels -- what the degradation looks like
SIM-8   drift versus jitter
SIM-9   speed benchmark for the refactor

Run:  python3 experiments_week5.py
Writes figs/ and results_week5.json
"""

from __future__ import annotations

import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

from sarsim import (ApertureConfig, RADARS, SceneConfig, apply_drift,
                    apply_jitter, autofocus, back_project, back_project_direct,
                    background_subtract, calibrate_noise_power, cross_range_cut,
                    minus_3db_width, peak_displacement_m, predicted_loss_db,
                    required_aperture, simulate_echoes, single_point,
                    tolerance_m, vio_error_over_aperture, body)

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figs")
os.makedirs(FIGS, exist_ok=True)

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2df"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#fcfcfb", "#bcd6f2", "#2a78d6", "#123a68"])
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.size": 10,
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "lines.linewidth": 2.0,
})

RADAR = RADARS["tdsr_p452"]
SCENE = SceneConfig(wall="drywall", target_range_m=3.0)
APERTURE = ApertureConfig(length_m=1.2)
NOMINAL = APERTURE.positions(RADAR)
RANGE_M = 3.0
N_TRIALS = 12
results = {}


def capture(targets, positions_true, noise_power):
    d = simulate_echoes(targets, RADAR, SCENE, positions_true, True,
                        rng=np.random.default_rng(101), noise_power=noise_power)
    e = simulate_echoes(targets, RADAR, SCENE, positions_true, False,
                        rng=np.random.default_rng(202), noise_power=noise_power)
    return background_subtract(d, e)


def mean_response(image, targets):
    """Mean coherent response across the true target locations.

    Averaging over all five body points rather than reading the brightest pixel
    keeps the autofocus comparison honest: autofocus locks onto the dominant
    scatterer, so scoring only that one would flatter it.
    """
    XX, YY = np.meshgrid(image.x, image.y, indexing="xy")
    mag = np.abs(image.values)
    vals = []
    for t in targets:
        d = np.hypot(XX - t.x_m, YY - t.y_m)
        iy, ix = np.unravel_index(np.argmin(d), d.shape)
        vals.append(mag[iy, ix] / t.amplitude)
    return float(np.sqrt(np.mean(np.square(vals))))


# ---------------------------------------------------------------------------
# SIM-6 -- position-error sweep
# ---------------------------------------------------------------------------
def sim6():
    targets = body(RANGE_M)
    noise_power = calibrate_noise_power(targets, RADAR, SCENE, NOMINAL)
    gx = np.linspace(-0.6, 0.6, 97)
    gy = np.linspace(RANGE_M - 0.5, RANGE_M + 0.9, 113)

    reference = mean_response(
        back_project(capture(targets, NOMINAL, noise_power), RADAR, NOMINAL,
                     gx, gy), targets)

    sigmas_mm = np.array([0.5, 1.0, 2.0, 3.5, 5.0, 7.0, 10.0, 15.0])
    rows = []
    for s_mm in sigmas_mm:
        sigma = s_mm / 1000.0
        raw_losses, af_losses, widths, estimated = [], [], [], []
        rng = np.random.default_rng(int(s_mm * 17) + 5)
        for _ in range(N_TRIALS):
            true_pos = apply_jitter(NOMINAL, sigma, rng)
            data = capture(targets, true_pos, noise_power)

            img = back_project(data, RADAR, NOMINAL, gx, gy)
            raw_losses.append(mean_response(img, targets))

            fixed, delta = autofocus(data, RADAR, NOMINAL, gx, gy)
            af_losses.append(mean_response(
                back_project(fixed, RADAR, NOMINAL, gx, gy), targets))
            estimated.append(float(np.sqrt(np.mean(delta ** 2))))

            pt = single_point(RANGE_M)
            npow = calibrate_noise_power(pt, RADAR, SCENE, NOMINAL)
            x, prof = cross_range_cut(capture(pt, true_pos, npow), RADAR,
                                      NOMINAL, RANGE_M)
            widths.append(minus_3db_width(x, prof))

        to_db = lambda v: -20 * np.log10(np.sqrt(np.mean(np.square(v))) / reference)
        rows.append(dict(
            sigma_mm=float(s_mm),
            loss_db=round(float(to_db(raw_losses)), 2),
            loss_after_autofocus_db=round(float(to_db(af_losses)), 2),
            predicted_loss_db=round(float(predicted_loss_db(sigma, RADAR,
                                                            at_band_edge=False)), 2),
            resolution_cm=round(float(np.mean(widths)) * 100, 1),
            estimated_error_mm=round(float(np.mean(estimated)) * 1000, 2),
        ))
        print(f"  sigma={s_mm:5.1f} mm  loss {rows[-1]['loss_db']:6.2f} dB  "
              f"(theory {rows[-1]['predicted_loss_db']:6.2f})  "
              f"after autofocus {rows[-1]['loss_after_autofocus_db']:6.2f} dB  "
              f"res {rows[-1]['resolution_cm']:4.1f} cm  "
              f"estimator saw {rows[-1]['estimated_error_mm']:5.2f} mm")

    results["sim6_jitter_sweep"] = rows
    results["tolerance_1db_mm"] = round(tolerance_m(RADAR, 1.0) * 1000, 2)
    results["tolerance_3db_mm"] = round(tolerance_m(RADAR, 3.0) * 1000, 2)
    results["sr05_requirement_mm"] = round(RADAR.wavelength_min_m / 16 * 1000, 2)
    results["vio_error_mm"] = {
        "0.1_percent": round(vio_error_over_aperture(1.1, 0.1) * 1000, 1),
        "0.5_percent": round(vio_error_over_aperture(1.1, 0.5) * 1000, 1),
        "1.0_percent": round(vio_error_over_aperture(1.1, 1.0) * 1000, 1),
    }

    # ---- figure
    s = np.array([r["sigma_mm"] for r in rows])
    raw = np.array([r["loss_db"] for r in rows])
    af = np.array([r["loss_after_autofocus_db"] for r in rows])
    theory = np.array([r["predicted_loss_db"] for r in rows])

    fig, ax = plt.subplots(figsize=(7.6, 4.8), constrained_layout=True)
    ax.axvspan(1.1, 11.0, color=S2, alpha=0.10, lw=0)
    ax.text(3.5, 16.5, "where off-the-shelf VIO lands\n(0.1–1 % drift over 1.1 m)",
            color=S2, fontsize=9, ha="center")
    ax.plot(s, theory, ls="--", color=INK2, lw=1.5, label="theory  4.343·φ²")
    ax.plot(s, raw, color=S1, marker="o", ms=7, label="simulated, no correction")
    ax.plot(s, af, color=S3, marker="s", ms=6, label="simulated, after autofocus")
    ax.axhline(3.0, color=INK2, lw=1.0, ls=":")
    ax.text(0.62, 3.35, "3 dB — still usable", color=INK2, fontsize=9)
    ax.set_xscale("log")
    ax.set_xticks([0.5, 1, 2, 3.5, 5, 10, 15])
    ax.set_xticklabels(["0.5", "1", "2", "3.5", "5", "10", "15"])
    ax.set_xlabel("antenna position error, RMS (mm)")
    ax.set_ylabel("loss of coherent response (dB)")
    ax.set_ylim(-1, 20)
    ax.set_title("How accurately the wearer's position must be known",
                 fontsize=11, color=INK)
    ax.legend(loc="upper left")
    fig.savefig(f"{FIGS}/fig6_positioning_tolerance.png", dpi=170)
    plt.close(fig)


# ---------------------------------------------------------------------------
# SIM-7 -- what it looks like
# ---------------------------------------------------------------------------
def sim7():
    targets = body(RANGE_M)
    noise_power = calibrate_noise_power(targets, RADAR, SCENE, NOMINAL)
    gx = np.linspace(-0.6, 0.6, 145)
    gy = np.linspace(RANGE_M - 0.4, RANGE_M + 0.9, 157)

    cases = [(0.0, "positions known exactly"),
             (3.5, "3.5 mm random error — at the requirement"),
             (10.0, "10 mm random error — image lost")]

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.0), constrained_layout=True)
    out = []
    for ax, (s_mm, title) in zip(axes, cases):
        rng = np.random.default_rng(31)
        true_pos = apply_jitter(NOMINAL, s_mm / 1000.0, rng)
        img = back_project(capture(targets, true_pos, noise_power), RADAR,
                           NOMINAL, gx, gy)
        m = ax.pcolormesh(img.x, img.y, img.magnitude_db, cmap=SEQ,
                          vmin=-25, vmax=0, shading="auto", rasterized=True)
        for t in targets:
            ax.plot(t.x_m, t.y_m, "o", ms=7, mfc="none", mec=S2, mew=1.6)
        ax.set_xlabel("cross-range (m)")
        ax.set_title(title, fontsize=10, color=INK)
        ax.grid(False)
        out.append(dict(sigma_mm=s_mm, title=title))
    axes[0].set_ylabel("down-range (m)")
    cb = fig.colorbar(m, ax=axes, shrink=0.9, pad=0.02)
    cb.set_label("normalised magnitude (dB)", color=INK2)
    cb.outline.set_edgecolor(GRID)
    fig.suptitle("Five-point body behind drywall at 3 m, 1.2 m aperture  ·  "
                 "orange circles are the true positions",
                 fontsize=11, color=INK)
    fig.savefig(f"{FIGS}/fig7_position_error_images.png", dpi=170)
    plt.close(fig)
    results["sim7_cases"] = out


# ---------------------------------------------------------------------------
# SIM-8 -- drift versus jitter
# ---------------------------------------------------------------------------
def sim8():
    target = single_point(RANGE_M)
    noise_power = calibrate_noise_power(target, RADAR, SCENE, NOMINAL)
    gx = np.linspace(-0.5, 0.5, 161)
    gy = np.linspace(RANGE_M - 0.5, RANGE_M + 0.5, 161)

    ref_img = back_project(capture(target, NOMINAL, noise_power), RADAR,
                           NOMINAL, gx, gy)
    ref_peak = np.abs(ref_img.values).max()

    rows = []
    cases = [("random jitter", 5.0), ("linear drift", 5.0), ("curved drift", 5.0),
             ("random jitter", 10.0), ("linear drift", 10.0), ("curved drift", 10.0)]
    for kind, amount_mm in cases:
        if kind == "random jitter":
            true_pos = apply_jitter(NOMINAL, amount_mm / 1000.0,
                                    np.random.default_rng(19))
        else:
            true_pos = apply_drift(NOMINAL, amount_mm / 1000.0,
                                   order=1 if kind == "linear drift" else 2)
        img = back_project(capture(target, true_pos, noise_power), RADAR,
                           NOMINAL, gx, gy)
        rows.append(dict(
            error_type=kind,
            amount_mm=amount_mm,
            loss_db=round(float(-20 * np.log10(np.abs(img.values).max() / ref_peak)), 2),
            peak_shift_cm=round(peak_displacement_m(img, target[0]) * 100, 1),
            width_cm=round(minus_3db_width(*cross_range_cut(
                capture(target, true_pos, noise_power), RADAR, NOMINAL,
                RANGE_M)) * 100, 1),
        ))
        print(f"  {kind:14s} {amount_mm:5.1f} mm  loss {rows[-1]['loss_db']:6.2f} dB  "
              f"peak moved {rows[-1]['peak_shift_cm']:4.1f} cm  "
              f"width {rows[-1]['width_cm']:4.1f} cm")
    results["sim8_drift_vs_jitter"] = rows


# ---------------------------------------------------------------------------
# SIM-9 -- refactor speed benchmark
# ---------------------------------------------------------------------------
def sim9():
    target = single_point(RANGE_M)
    data = capture(target, NOMINAL,
                   calibrate_noise_power(target, RADAR, SCENE, NOMINAL))
    gx = np.linspace(-0.6, 0.6, 97)
    gy = np.linspace(2.4, 3.6, 97)

    t0 = time.perf_counter()
    fast = back_project(data, RADAR, NOMINAL, gx, gy)
    t_fast = time.perf_counter() - t0

    t0 = time.perf_counter()
    slow = back_project_direct(data, RADAR, NOMINAL, gx, gy)
    t_slow = time.perf_counter() - t0

    err = float(np.abs(fast.values - slow.values).max()
                / np.abs(slow.values).max())
    results["sim9_refactor"] = dict(
        pixels=int(gx.size * gy.size),
        positions=int(NOMINAL.shape[0]),
        frequencies=int(RADAR.n_freq),
        direct_seconds=round(t_slow, 2),
        fast_seconds=round(t_fast, 3),
        speedup=round(t_slow / t_fast, 1),
        agreement_db=round(float(20 * np.log10(max(err, 1e-12))), 1),
    )
    print(f"  direct {t_slow:.2f} s -> fast {t_fast:.3f} s "
          f"({t_slow/t_fast:.0f}x), agreement {20*np.log10(err):.0f} dB")


if __name__ == "__main__":
    print("SIM-6 position-error sweep")
    sim6()
    print("SIM-7 image panels")
    sim7()
    print("SIM-8 drift vs jitter")
    sim8()
    print("SIM-9 refactor benchmark")
    sim9()
    with open(os.path.join(HERE, "results_week5.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print("\nwrote figs/ and results_week5.json")
