"""
Validate the direct-VBMicrolensing magnification Op against the
MulensModel-backed Op (hpc_optimization.txt P2).

The direct path re-derives everything MulensModel does per call (parallax
projection, trajectory rotation, VBM dispatch), so any convention drift —
pi_E signs, alpha sense, frame origin — shows up here as a magnification
mismatch far above floating-point noise.
"""

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

pytestmark = pytest.mark.slow

from exozippy.components.mulensing.op import (
    BinaryLensMagOp,
    VBMDirectMagOp,
)

_COORDS = "268.0d -29.0d"
_T0_PAR = 2458554.89
# DC2018_128-like MAP: t_E~18.2d, s~0.98, q~0.0011, alpha~-52 deg
_MAP = dict(
    t_0=2458554.89,
    u_0=0.1,
    t_E=18.2,
    pi_E_N=0.02,
    pi_E_E=-0.01,
    rho=0.002,
    s=0.98,
    q=0.0011,
    alpha=-52.0,
    u1=0.5,
)
_ORDER = [
    "t_0",
    "u_0",
    "t_E",
    "pi_E_N",
    "pi_E_E",
    "rho",
    "s",
    "q",
    "alpha",
    "u1",
]
# Slice of a full _draw() vector for the point-source, no-LD param layout
# ([t_0, u_0, t_E, pi_E_N, pi_E_E, s, q, alpha]): drop rho and u1, keep the
# draw scatter conventions identical to the finite-source comparison.
_POINT_SOURCE_INDEX = [
    _ORDER.index(k) for k in _ORDER if k not in ("rho", "u1")
]


def _times_and_obs(n=400, span=150.0):
    """Times plus Skowron+2011 geocentric deviations for an L2-like observer.

    Both Ops consume the same deviation array, so a smooth annual-scale
    curve (Earth's orbit departs quadratically-then-worse from the linear
    reference, reaching O(1) AU over a season) plus a constant satellite
    offset exercises the parallax projection without any ephemeris or
    network dependency.

    The default span reaches u ~ 8 Einstein radii in the wings, past the
    far-field point-source guard boundary in VBMDirectMagOp._magnify, so the
    A/B comparison exercises both the near (BinaryMag2) and far (BinaryMag0)
    dispatch paths.
    """
    times = np.linspace(_T0_PAR - span, _T0_PAR + span, n)
    phase = 2.0 * np.pi * (times - _T0_PAR) / 365.25
    dev = np.column_stack(
        [
            0.5 * (1.0 - np.cos(phase)),
            0.5 * (np.sin(phase) - phase),
            0.2 * (1.0 - np.cos(phase)),
        ]
    )
    offset = np.array([0.009, 0.004, 0.002])
    return times, dev + offset[None, :]


def _compile(op):
    p = pt.dvector("p")
    t = pt.dvector("t")
    o = pt.dmatrix("o")
    return pytensor.function([p, t, o], op(p, t, o))


def _draw(rng, scale=1.0):
    p = dict(_MAP)
    p["t_0"] += rng.normal(0, 0.05) * scale
    p["u_0"] *= 1 + rng.normal(0, 0.05) * scale
    p["t_E"] *= 1 + rng.normal(0, 0.05) * scale
    p["pi_E_N"] += rng.normal(0, 0.02) * scale
    p["pi_E_E"] += rng.normal(0, 0.02) * scale
    p["rho"] *= 1 + rng.normal(0, 0.1) * scale
    p["s"] *= 1 + rng.normal(0, 0.02) * scale
    p["q"] *= 1 + rng.normal(0, 0.1) * scale
    p["alpha"] += rng.normal(0, 2.0) * scale
    return np.array([p[k] for k in _ORDER])


def test_vbm_direct_matches_mulensmodel_binary_with_parallax_and_ld():
    """
    Given a binary lens + finite source + LD + satellite parallax and random
      parameter draws around the DC128 MAP,
    When both the MulensModel Op (VBM everywhere) and the direct-VBM Op are
      evaluated,
    Then magnifications agree per-point to rtol 1e-8 (identical VBM kernel,
      only the trajectory plumbing differs).
    """
    times, obs = _times_and_obs()
    f_mm = _compile(
        BinaryLensMagOp(
            coords=_COORDS,
            mag_method=[times[0] - 1.0, "VBM", times[-1] + 1.0],
            use_rho=True,
            bandpass="Z087",
        )
    )
    f_dir = _compile(
        VBMDirectMagOp(
            coords=_COORDS, n_companions=1, use_rho=True, bandpass="Z087"
        )
    )

    rng = np.random.default_rng(42)
    worst = 0.0
    for _ in range(25):
        p = _draw(rng)
        A_mm = f_mm(p, times, obs)
        A_dir = f_dir(p, times, obs)
        worst = max(worst, np.max(np.abs(A_mm - A_dir) / np.abs(A_mm)))
    assert worst < 1e-8, f"direct path deviates from MulensModel: {worst:.2e}"


@pytest.mark.parametrize(
    "grid_name, n, span, min_peak",
    [
        ("wide", 400, 150.0, 5.0),
        ("caustic", 600, 2.0, 50.0),
    ],
)
def test_vbm_direct_matches_mulensmodel_binary_point_source(
    grid_name, n, span, min_peak
):
    """
    Given a POINT-source binary lens with satellite parallax and random
      parameter draws around the DC128 MAP,
    When both the MulensModel Op (auto_vbbl -> "point_source") and the
      direct-VBM Op (BinaryMag0) are evaluated,
    Then magnifications are finite and agree per-point to rtol 1e-9.

    This is the first independent check of vbm_direct's point-source binary
    magnification. It could not exist before: the MulensModel Op selected
    "VBBL" for a rho-less model, MulensModel refused, and perform() returned
    all-NaN -- so every A/B test in this file used use_rho=True.

    Two grids, because the wide one alone would prove nothing here: at 0.75 d
    sampling the narrow central caustic falls between epochs and the peak only
    reaches A ~ 11.  The "caustic" grid resolves it (A ~ 60-600 depending on
    the draw, up to ~63x the equivalent single-lens magnification), which is
    where two independent root solvers are most likely to part ways.
    min_peak asserts the grid really does sample that structure, so the test
    cannot silently decay into a flat-wings comparison.

    Both engines are exact analytic point-source solvers (MulensModel's
    BinaryLensPointSourceMagnification vs VBM's BinaryMag0), not the shared
    VBM kernel the finite-source test compares, so agreement is a genuine
    cross-implementation result rather than a tautology.
    """
    # Arrange
    times, obs = _times_and_obs(n=n, span=span)
    f_mm = _compile(
        BinaryLensMagOp(coords=_COORDS, mag_method="auto_vbbl", use_rho=False)
    )
    f_dir = _compile(
        VBMDirectMagOp(coords=_COORDS, n_companions=1, use_rho=False)
    )

    # Act
    rng = np.random.default_rng(42)
    worst = 0.0
    peak = 0.0
    for _ in range(25):
        p = _draw(rng)[_POINT_SOURCE_INDEX]
        A_mm = f_mm(p, times, obs)
        A_dir = f_dir(p, times, obs)
        assert np.all(np.isfinite(A_mm)), f"{grid_name}: MulensModel gave NaN"
        assert np.all(np.isfinite(A_dir)), f"{grid_name}: direct path gave NaN"
        worst = max(worst, np.max(np.abs(A_mm - A_dir) / np.abs(A_mm)))
        peak = max(peak, A_mm.max())

    # Assert
    assert peak > min_peak, (
        f"{grid_name} grid never reached the caustic (peak A={peak:.2f}); "
        "the comparison would be vacuous"
    )
    # Observed worst: 7.7e-15 (wide), 1.7e-12 (caustic); the caustic grid is
    # worse because point-source magnification diverges at a fold and the
    # 5th-order complex root solve is ill-conditioned there.  1e-9 keeps ~500x
    # headroom over that while staying far below any convention error (an
    # alpha or pi_E sign flip moves these by O(0.1-100), not O(1e-9)).
    assert worst < 1e-9, (
        f"{grid_name}: direct point-source path deviates from "
        f"MulensModel: {worst:.2e}"
    )


def test_multi_lens_frame_reduces_to_binary():
    """
    Given the direct Op with two companions, the second having negligible
      mass (q=1e-9) far from the caustics,
    When compared against the single-companion (BinaryMag2) path,
    Then magnifications agree to the VBM tolerance (1e-3 absolute; observed
      ~1e-4) — validating the trajectory-frame lens geometry construction.
    """
    times, obs = _times_and_obs()
    f_bin = _compile(
        VBMDirectMagOp(
            coords=_COORDS, n_companions=1, use_rho=True, bandpass="Z087"
        )
    )
    f_multi = _compile(
        VBMDirectMagOp(
            coords=_COORDS, n_companions=2, use_rho=True, bandpass="Z087"
        )
    )

    rng = np.random.default_rng(7)
    worst = 0.0
    for _ in range(5):
        p = _draw(rng)
        p_multi = np.concatenate([p[:-1], [9.0, 1e-9, 33.0], p[-1:]])
        A_bin = f_bin(p, times, obs)
        A_multi = f_multi(p_multi, times, obs)
        worst = max(worst, np.max(np.abs(A_bin - A_multi) / np.abs(A_bin)))
    assert worst < 1e-3, (
        f"multi-lens frame does not reduce to binary: {worst:.2e}"
    )


def test_triple_lens_magnification_evaluates():
    """
    Given the direct Op with two massive companions (a genuine triple lens),
    When evaluated at the MAP,
    Then it returns finite magnifications >= 1 everywhere (no NaN, no crash).
    """
    times, obs = _times_and_obs(n=60)
    f = _compile(
        VBMDirectMagOp(
            coords=_COORDS, n_companions=2, use_rho=True, bandpass="Z087"
        )
    )
    p = np.array(
        [_MAP[k] for k in _ORDER[:-1]] + [1.3, 0.002, 110.0, _MAP["u1"]]
    )
    A = f(p, times, obs)
    assert np.all(np.isfinite(A))
    assert np.all(A >= 1.0 - 1e-6)


def test_vbm_direct_point_source_is_finite_near_and_far():
    """
    Given use_rho=False (user's finite_source: False) and epochs spanning
      both near-caustic and far-field trajectory points,
    When the direct Op is evaluated,
    Then every output is finite. VBM's BinaryMag2 finite-source integrator
      returns NaN for an exactly-zero source radius, so rho=0 near-caustic
      calls must be dispatched to the point-source BinaryMag0 rather than
      gated on distance like the finite-source (rho>0) case.
    """
    times, obs = _times_and_obs(n=400)
    f = _compile(VBMDirectMagOp(coords=_COORDS, n_companions=1, use_rho=False))
    order_no_rho = [k for k in _ORDER if k not in ("rho", "u1")]
    p = np.array([_MAP[k] for k in order_no_rho])
    A = f(p, times, obs)
    assert np.all(np.isfinite(A)), "point-source path produced NaN"
    assert np.all(A >= 1.0 - 1e-6)


def test_vbm_direct_nan_params_yield_nan():
    """
    Given a parameter vector containing NaN (sampler exploring junk),
    When the direct Op is evaluated,
    Then every output is NaN (rejected proposal) rather than a crash or hang.
    """
    times, obs = _times_and_obs(n=30)
    f = _compile(
        VBMDirectMagOp(
            coords=_COORDS, n_companions=1, use_rho=True, bandpass="Z087"
        )
    )
    p = np.array([_MAP[k] for k in _ORDER])
    p[3] = np.nan
    assert np.all(np.isnan(f(p, times, obs)))
