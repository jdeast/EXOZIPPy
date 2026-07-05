"""
Validate the direct-VBMicrolensing magnification Op against the
MulensModel-backed Op (hpc_optimization.txt P2).

The direct path re-derives everything MulensModel does per call (parallax
projection, trajectory rotation, VBM dispatch), so any convention drift —
pi_E signs, alpha sense, frame origin — shows up here as a magnification
mismatch far above floating-point noise.
"""
import numpy as np
import pytest
import pytensor
import pytensor.tensor as pt

pytestmark = pytest.mark.slow

from exozippy.components.mulensing.op import (
    BinaryLensMagOp, VBMDirectMagOp, _earth_xyz_at)

_COORDS = "268.0d -29.0d"
_T0_PAR = 2458554.89
# DC2018_128-like MAP: t_E~18.2d, s~0.98, q~0.0011, alpha~-52 deg
_MAP = dict(t_0=2458554.89, u_0=0.1, t_E=18.2, pi_E_N=0.02, pi_E_E=-0.01,
            rho=0.002, s=0.98, q=0.0011, alpha=-52.0, u1=0.5)
_ORDER = ['t_0', 'u_0', 't_E', 'pi_E_N', 'pi_E_E', 'rho', 's', 'q', 'alpha', 'u1']


def _times_and_obs(n=400, span=150.0):
    """Times plus an L2-like satellite observer (absolute barycentric AU).

    The default span reaches u ~ 8 Einstein radii in the wings, past the
    far-field point-source guard boundary in VBMDirectMagOp._magnify, so the
    A/B comparison exercises both the near (BinaryMag2) and far (BinaryMag0)
    dispatch paths.
    """
    times = np.linspace(_T0_PAR - span, _T0_PAR + span, n)
    offset = np.array([0.009, 0.004, 0.002])
    return times, _earth_xyz_at(times) + offset[None, :]


def _compile(op):
    p = pt.dvector('p')
    t = pt.dvector('t')
    o = pt.dmatrix('o')
    return pytensor.function([p, t, o], op(p, t, o))


def _draw(rng, scale=1.0):
    p = dict(_MAP)
    p['t_0'] += rng.normal(0, 0.05) * scale
    p['u_0'] *= 1 + rng.normal(0, 0.05) * scale
    p['t_E'] *= 1 + rng.normal(0, 0.05) * scale
    p['pi_E_N'] += rng.normal(0, 0.02) * scale
    p['pi_E_E'] += rng.normal(0, 0.02) * scale
    p['rho'] *= 1 + rng.normal(0, 0.1) * scale
    p['s'] *= 1 + rng.normal(0, 0.02) * scale
    p['q'] *= 1 + rng.normal(0, 0.1) * scale
    p['alpha'] += rng.normal(0, 2.0) * scale
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
    f_mm = _compile(BinaryLensMagOp(
        coords=_COORDS, mag_method=[times[0] - 1.0, "VBM", times[-1] + 1.0],
        use_rho=True, bandpass="Z087"))
    f_dir = _compile(VBMDirectMagOp(
        coords=_COORDS, n_companions=1, use_rho=True, bandpass="Z087"))

    rng = np.random.default_rng(42)
    worst = 0.0
    for _ in range(25):
        p = _draw(rng)
        A_mm = f_mm(p, times, obs)
        A_dir = f_dir(p, times, obs)
        worst = max(worst, np.max(np.abs(A_mm - A_dir) / np.abs(A_mm)))
    assert worst < 1e-8, f"direct path deviates from MulensModel: {worst:.2e}"


def test_multi_lens_frame_reduces_to_binary():
    """
    Given the direct Op with two companions, the second having negligible
      mass (q=1e-9) far from the caustics,
    When compared against the single-companion (BinaryMag2) path,
    Then magnifications agree to the VBM tolerance (1e-3 absolute; observed
      ~1e-4) — validating the trajectory-frame lens geometry construction.
    """
    times, obs = _times_and_obs()
    f_bin = _compile(VBMDirectMagOp(
        coords=_COORDS, n_companions=1, use_rho=True, bandpass="Z087"))
    f_multi = _compile(VBMDirectMagOp(
        coords=_COORDS, n_companions=2, use_rho=True, bandpass="Z087"))

    rng = np.random.default_rng(7)
    worst = 0.0
    for _ in range(5):
        p = _draw(rng)
        p_multi = np.concatenate([p[:-1], [9.0, 1e-9, 33.0], p[-1:]])
        A_bin = f_bin(p, times, obs)
        A_multi = f_multi(p_multi, times, obs)
        worst = max(worst, np.max(np.abs(A_bin - A_multi) / np.abs(A_bin)))
    assert worst < 1e-3, f"multi-lens frame does not reduce to binary: {worst:.2e}"


def test_triple_lens_magnification_evaluates():
    """
    Given the direct Op with two massive companions (a genuine triple lens),
    When evaluated at the MAP,
    Then it returns finite magnifications >= 1 everywhere (no NaN, no crash).
    """
    times, obs = _times_and_obs(n=60)
    f = _compile(VBMDirectMagOp(
        coords=_COORDS, n_companions=2, use_rho=True, bandpass="Z087"))
    p = np.array([_MAP[k] for k in _ORDER[:-1]] + [1.3, 0.002, 110.0, _MAP['u1']])
    A = f(p, times, obs)
    assert np.all(np.isfinite(A))
    assert np.all(A >= 1.0 - 1e-6)


def test_vbm_direct_nan_params_yield_nan():
    """
    Given a parameter vector containing NaN (sampler exploring junk),
    When the direct Op is evaluated,
    Then every output is NaN (rejected proposal) rather than a crash or hang.
    """
    times, obs = _times_and_obs(n=30)
    f = _compile(VBMDirectMagOp(
        coords=_COORDS, n_companions=1, use_rho=True, bandpass="Z087"))
    p = np.array([_MAP[k] for k in _ORDER])
    p[3] = np.nan
    assert np.all(np.isnan(f(p, times, obs)))
