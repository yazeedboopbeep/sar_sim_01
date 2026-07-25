# sarsim v0.2 — through-wall synthetic-aperture radar simulator

Week 5 deliverable for the **PoseDetection** venture. Successor to the Week 4
single-file `sar_sim.py` (v0.1), which is kept as `sar_sim_v01_reference.py`.

## The question this version answers

Week 4 showed the wearer's walk can synthesise an aperture long enough to
resolve body parts. It did that assuming the radar knew exactly where it was at
every step — which is the one thing a wearable cannot buy off the shelf. It
knows where it is from visual-inertial odometry, not from a rail.

**How wrong is that position allowed to be?**

## Headline results

| Question | Answer |
|---|---|
| Loss at 3.5 mm RMS random position error | 1.5 dB |
| Loss at 5 mm | 3.3 dB |
| Loss at 10 mm | 10.6 dB — image gone |
| Agreement with closed-form `4.343·φ²` | within 0.3 dB up to 7 mm |
| 10 mm **linear drift** | **no** focus loss; target lands 2.6 cm off |
| 10 mm **curved drift** (quadratic bow) | 1.2 dB |
| Autofocus at 10 mm random error | holds loss to 2.6 dB |
| Autofocus floor / measurement limit | 1.7 dB / ~3 mm on a five-point body |

**The finding that changed a requirement:** position error is not one quantity.
The random component blurs the image; smooth drift only moves it. Week 4's
SR-05 (3.5 mm on *total* error) was stricter than necessary and aimed at the
wrong thing.

- **SR-05 (revised)** — random component ≤ 5 mm RMS (≤ 3 dB loss)
- **SR-11 (new)** — smooth drift ≤ 40 mm, keeping location error under 10 cm

## What changed in the code

| | v0.1 (Week 4) | v0.2 (Week 5) |
|---|---|---|
| Structure | one 700-line file | seven-module package |
| Tests | none (7 manual checks) | 19 pytest tests, 3.4 s |
| One 9,409-pixel image | 7.33 s | 0.084 s |
| Positioning error | not modelled | jitter + linear/curved drift |
| Autofocus | none | wideband PGA variant |

The speed-up: range-compress each position **once**, then look up a range per
pixel, instead of summing all 128 frequencies at every pixel. Same sum, better
order. `test_fast_back_projection_matches_direct_sum` checks the two agree
(−59 dB), so it's a tested claim rather than an assertion.

## Layout

```
sarsim/
  config.py       radar / wall / scene / aperture parameters, each with a source note
  targets.py      point-scatterer models (single, pair, five-point body)
  forward.py      echo synthesis from TRUE positions + clutter subtraction
  imaging.py      fast and direct back-projection, range compression, cuts
  metrics.py      resolution, separability, SNR, displacement
  positioning.py  jitter and drift models, the phase-error law, tolerances
  autofocus.py    estimate and remove trajectory error from the image itself
tests/            19 pytest tests
experiments_week5.py   SIM-6 … SIM-9, writes figs/ and results_week5.json
```

## Usage

```bash
python3 -m pytest tests/ -q        # 19 tests
python3 experiments_week5.py       # regenerate figures + results_week5.json
```

Requires NumPy; Matplotlib for the experiments; pytest for the suite.

## Deliberate limitations → Week 6

| Limitation | Week 6 |
|---|---|
| Clutter cancellation still needs an empty-room reference | Reference-free removal (slow-time high-pass, SVD/subspace); measure the penalty |
| Target is perfectly still during the sweep | Add breathing and body motion; supply the prediction test 0d needs |
| Isotropic point scatterers | Aspect-dependent scattering |
| Autofocus needs a dominant scatterer | Multi-scatterer autofocus, or accept the 1.7 dB floor |

**What is settled:** how accurately the wearer must be positioned, which part of
the error matters, and that autofocus buys back most of a realistic shortfall.

**What is not:** whether a person holds still enough for any of this to work on
a living target. That is Week 6, and it is the last physics unknown before
hardware.
