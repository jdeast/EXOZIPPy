"""Astrometric microlensing, stage 2: the centroid shift from
VBMicrolensing's astrox accumulators through ``VBMDirectMagOp(astrometry=True)``
(conventions.md C30; review 8.10.1).

What is pinned, and against what:

1. The Op's centroid for a BINARY lens against a first-principles solution
   of the lens equation (images found by 2-D root finding, weighted by
   their own magnifications) built in the TRAJECTORY frame from the same
   s, q, alpha, tau, u.  This is the origin test the review item made
   non-negotiable: it agrees with VBM's ``astrox`` only if VBM's origin is
   the lens centre of mass AND the Op's alpha rotation is undone correctly.
2. The same for THREE lens bodies (the MultiMag branch), at the Op level.
3. The single-lens finite-source shift against a direct disk integration
   of the point-source centroid, including the ``rho ~ u`` cancellation and
   the sign reversal beyond -- what stage 1 could only warn about.
4. A forced ``use_op`` point-source single lens through the Op equals the
   symbolic stage-1 shift, with real Earth deviations -- the frame
   rotation onto (N, E) is the same on both paths.
5. The photometric Op's A is untouched (``astrometry=False`` is the same
   code), and a NaN proposal yields three NaN outputs.
"""

import numpy as np
import pytensor.tensor as pt
import pytest
from scipy.optimize import fsolve

pytestmark = pytest.mark.slow

from test_astrometric_microlensing import (  # noqa: E402
    T0,
    U0,
    _astro_logp,
    _base_params,
    _build,
    _compile_at_start,
    _full_config,
    _mulens_config,
    _paczynski,
    _theta_E_truth,
)

from exozippy.components.mulensing.op import VBMDirectMagOp
from exozippy.system import System

COORDS = "266.4168d -29.0078d"


# ----------------------------------------------------------------------
# First-principles image solver (numpy, independent of VBM)
# ----------------------------------------------------------------------
def _images(zl, ml, zeta, n_grid=41, span=4.0):
    """All images of point source ``zeta`` for point lenses at ``zl``
    (complex) with masses ``ml`` (sum 1), by 2-D root finding from a grid
    of starts, deduplicated; returns [(z, |mu|), ...]."""

    def eq(p):
        z = p[0] + 1j * p[1]
        f = z - sum(m / np.conj(z - zz) for zz, m in zip(zl, ml)) - zeta
        return [f.real, f.imag]

    # Starts: a grid around the lenses (where the minor images sit) PLUS a
    # grid around the source (where the major image sits, at ~|zeta| for a
    # far-field epoch that the lens-centred grid would miss).
    starts = [
        (x0, y0)
        for x0 in np.linspace(-span, span, n_grid)
        for y0 in np.linspace(-span, span, n_grid)
    ] + [
        (zeta.real + dx, zeta.imag + dy)
        for dx in np.linspace(-2.0, 2.0, 21)
        for dy in np.linspace(-2.0, 2.0, 21)
    ]
    roots = []
    with np.errstate(all="ignore"):
        for x0, y0 in starts:
            sol, _, ier, _ = fsolve(eq, [x0, y0], full_output=True)
            if ier != 1:
                continue
            r = eq(sol)
            if abs(r[0]) + abs(r[1]) > 1e-10:
                continue
            z = sol[0] + 1j * sol[1]
            if all(abs(z - rr) > 1e-6 for rr in roots):
                roots.append(z)
    out = []
    for z in roots:
        g = sum(m / np.conj(z - zz) ** 2 for zz, m in zip(zl, ml))
        out.append((z, 1.0 / abs(1.0 - abs(g) ** 2)))
    return out


def _first_principles(companions, tau, u):
    """(A, dtau, dbeta) of the images' light centroid relative to the
    source, in the TRAJECTORY frame: source at (-tau, -u), companion j at
    s_j (cos alpha_j, -sin alpha_j) from the primary, origin at the centre
    of mass (op.py's N-lens construction, stated in its docstring)."""
    q_tot = sum(q for (_, q, _) in companions)
    m = [1.0 / (1.0 + q_tot)] + [q / (1.0 + q_tot) for (_, q, _) in companions]
    pos = [0j] + [
        s * np.cos(a) - 1j * s * np.sin(a) for (s, _, a) in companions
    ]
    com = sum(mi * zi for mi, zi in zip(m, pos))
    zl = [z - com for z in pos]
    zeta = -tau - 1j * u
    ims = _images(zl, m, zeta)
    A = sum(mu for _, mu in ims)
    cen = sum(mu * z for z, mu in ims) / A
    d = cen - zeta
    return A, d.real, d.imag


def _disk_average_shift(u0, rho, n_r=300, n_t=600):
    """Flux-weighted centroid (from the lens, along the axis) of a uniform
    disk of radius rho centred at u0 on the x axis, each element the exact
    point-lens pair of images; returns (shift = centroid - u0, A)."""
    r = (np.arange(n_r) + 0.5) / n_r * rho
    t = (np.arange(n_t) + 0.5) / n_t * 2 * np.pi
    R, T = np.meshgrid(r, t, indexing="ij")
    x = u0 + R * np.cos(T)
    y = R * np.sin(T)
    u = np.hypot(x, y)
    A = _paczynski(u)
    c_rad = u * (u * u + 3) / (u * u + 2)
    cx = c_rad * x / u
    w = A * R
    return (w * cx).sum() / w.sum() - u0, (A * R).sum() / R.sum()


def _op_eval(op, p, times, obs):
    outs = op(
        pt.as_tensor_variable(np.asarray(p, dtype=float)),
        pt.as_tensor_variable(np.asarray(times, dtype=float)),
        pt.as_tensor_variable(np.asarray(obs, dtype=float)),
    )
    if not isinstance(outs, (list, tuple)):
        outs = [outs]
    return [np.asarray(o.eval()) for o in outs]


# ----------------------------------------------------------------------
# 1 + 2: the origin pin, at the Op level
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "companions",
    [
        [(1.2, 0.3, np.radians(30.0))],
        [(0.8, 2.5, np.radians(-110.0))],  # q > 1 is legal (C14)
        [(1.2, 0.3, np.radians(30.0)), (0.6, 0.05, np.radians(200.0))],
    ],
)
def test_op_centroid_matches_first_principles_images(companions):
    """
    Given a binary (or triple) lens and a point source on several epochs of
      a trajectory with u_0 = 0.23,
    When VBMDirectMagOp(astrometry=True) evaluates (A, dtau, dbeta) and the
      images are solved from the lens equation in the trajectory frame,
    Then both A and the centroid shift agree to 1e-7 -- which holds only
      with VBM's origin at the centre of mass (primary at -s q/(1+q)) AND
      the Op's alpha rotation undone; any other origin is off by O(0.1).
    """
    t_e = 20.0
    times = T0 + np.array([-15.0, -4.0, 0.0, 3.0, 11.0])
    p = [T0, U0, t_e, 0.0, 0.0]
    for s, q, a in companions:
        p += [s, q, np.degrees(a)]
    op = VBMDirectMagOp(
        coords=COORDS,
        n_companions=len(companions),
        use_rho=False,
        astrometry=True,
    )
    A, dtau, dbeta = _op_eval(op, p, times, np.zeros((len(times), 3)))
    tau = (times - T0) / t_e
    for k in range(len(times)):
        A_fp, dt_fp, db_fp = _first_principles(companions, tau[k], U0)
        assert A[k] == pytest.approx(A_fp, rel=1e-7), (k, A[k], A_fp)
        assert dtau[k] == pytest.approx(dt_fp, abs=1e-7), (k, dtau[k], dt_fp)
        assert dbeta[k] == pytest.approx(db_fp, abs=1e-7), (k, dbeta[k], db_fp)
    # the shift is O(0.1) Einstein radii here, so 1e-7 has teeth
    assert np.max(np.hypot(dtau, dbeta)) > 0.05


def test_binary_far_field_branch_fills_the_centroid():
    """
    Given a finite-source binary and epochs far outside R_inf + 2 rho,
    When the Op takes its BinaryMag0 far-field branch,
    Then the shift there is finite, nonzero and within 5e-5 Einstein radii
      of the point-source first-principles centroid (the guard needs no
      exception).  The tolerance is VBM's, not this Op's: at u ~ 12-15 its
      BinaryMag0 carries ~1e-6 absolute error in A (3% of the A - 1 excess)
      and ~2e-5 in the centroid, while at u ~ 2 the same comparison holds
      to 1e-12 (the origin test above); op.py's class docstring records it.
    """
    companions = [(1.2, 0.3, np.radians(30.0))]
    t_e = 20.0
    times = T0 + np.array([-300.0, 250.0])  # u ~ 12-15
    p = [T0, U0, t_e, 0.0, 0.0, 0.01] + [1.2, 0.3, 30.0]
    op = VBMDirectMagOp(
        coords=COORDS, n_companions=1, use_rho=True, astrometry=True
    )
    A, dtau, dbeta = _op_eval(op, p, times, np.zeros((2, 3)))
    tau = (times - T0) / t_e
    for k in range(2):
        A_fp, dt_fp, db_fp = _first_principles(companions, tau[k], U0)
        assert np.isfinite(dtau[k]) and abs(dtau[k]) > 0.01
        assert dtau[k] == pytest.approx(dt_fp, abs=5e-5)
        assert dbeta[k] == pytest.approx(db_fp, abs=5e-5)
        assert A[k] == pytest.approx(A_fp, abs=5e-6)


# ----------------------------------------------------------------------
# 3: finite source, single lens
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "u0, rho",
    [(0.1, 0.01), (0.1, 0.1), (0.1, 0.5), (np.sqrt(2.0), 0.5)],
)
def test_espl_op_shift_matches_disk_integration(u0, rho):
    """
    Given a uniform finite source of radius rho at closest approach u_0,
    When the single-lens Op (ESPLMag2, astrox1 projected radially)
      evaluates the shift,
    Then it matches a direct integration of the point-source centroid over
      the disk to 3e-3 of theta_E (VBM Tol 1e-3) -- including the near-
      vanishing at rho ~ u_0 and the SIGN REVERSAL at rho > u_0 that make
      the point-source formula unsafe there.
    """
    p = [T0, u0, 20.0, 0.0, 0.0, rho]
    op = VBMDirectMagOp(
        coords=COORDS, n_companions=0, use_rho=True, astrometry=True
    )
    A, dtau, dbeta = _op_eval(op, p, [T0], np.zeros((1, 3)))
    shift_num, A_num = _disk_average_shift(u0, rho)
    # at t_0 the source sits at (0, -u0): the shift is along beta_hat only,
    # and "away from the lens" is -beta_hat there, so dbeta = -shift
    assert dtau[0] == pytest.approx(0.0, abs=1e-9)
    assert -dbeta[0] == pytest.approx(shift_num, abs=3e-3)
    assert A[0] == pytest.approx(A_num, rel=5e-3)
    point = u0 / (u0**2 + 2.0)
    if rho > u0:
        assert np.sign(-dbeta[0]) == -np.sign(point)  # reversed
    elif rho == pytest.approx(u0):
        assert abs(dbeta[0]) < 0.01 * point  # ~99.6% cancelled


def test_single_lens_op_never_reads_the_stale_astrox2():
    """
    Given one Op instance that has evaluated a BINARY epoch (filling
      astrox2) and is then used as a single lens,
    When ... it cannot: the single-lens Op is its own instance; but a
      single-lens instance evaluating a source off the beta axis must give
      a shift ALONG the lens->source axis exactly (radial projection of
      astrox1), which a stale or zero astrox2 would break.
    """
    op = VBMDirectMagOp(
        coords=COORDS, n_companions=0, use_rho=True, astrometry=True
    )
    t_e = 20.0
    times = T0 + np.array([-6.0, 9.0])
    p = [T0, U0, t_e, 0.0, 0.0, 0.05]
    A, dtau, dbeta = _op_eval(op, p, times, np.zeros((2, 3)))
    tau = (times - T0) / t_e
    for k in range(2):
        # source at (-tau, -u): the shift must be parallel to it
        cross = dtau[k] * (-U0) - dbeta[k] * (-tau[k])
        assert abs(cross) < 1e-12 * np.hypot(dtau[k], dbeta[k])
        # and pointing AWAY from the lens (rho << u here)
        assert dtau[k] * (-tau[k]) + dbeta[k] * (-U0) > 0


# ----------------------------------------------------------------------
# 4: Op path == symbolic path for a forced point-source single lens
# ----------------------------------------------------------------------
def test_forced_op_astrometric_terms_match_symbolic_with_parallax():
    """
    Given a point-source single-lens event with use_op: true (so
      uses_op is True and get_astrometric_terms takes the Op), real Earth
      deviations over a year, and a relative proper motion with both
      components nonzero,
    When (A, dN, dE) are evaluated on both paths at the start point,
    Then they agree to 1e-9 mas -- the Op's trajectory frame, its
      (dtau, dbeta) outputs and _shift_to_sky reproduce the symbolic
      closed form exactly, parallax included.
    """
    system = System(
        _mulens_config(use_op=True),
        user_params=_base_params(mu_lens=(4.0, -3.0)),
    )
    system.prepare()
    model = system.build_model()
    event = system.mulensevent
    assert event.uses_op()
    t = np.linspace(T0 - 200.0, T0 + 200.0, 9)
    from exozippy.ephemeris import get_observer_position

    dev = event.skowron_deviations(t, get_observer_position(t, "earth"))
    assert np.max(np.abs(dev)) > 1e-4  # the deviations are real
    with model:
        A_op, dN_op, dE_op = event.get_astrometric_terms(
            t, dev, system, index=0
        )
        A_sym = event.get_magnification(t, dev, system, index=0)
        dN_sym, dE_sym = event.get_centroid_shift(t, dev, system, index=0)
    vals = _compile_at_start(
        model, [A_op, dN_op, dE_op, A_sym, dN_sym, dE_sym]
    )
    A_op, dN_op, dE_op, A_sym, dN_sym, dE_sym = vals
    np.testing.assert_allclose(A_op, A_sym, rtol=1e-12)
    np.testing.assert_allclose(dN_op, dN_sym, atol=1e-9)
    np.testing.assert_allclose(dE_op, dE_sym, atol=1e-9)
    assert np.max(np.hypot(dN_sym, dE_sym)) > 0.05 * _theta_E_truth()


# ----------------------------------------------------------------------
# 5: photometry untouched; NaN arity
# ----------------------------------------------------------------------
def test_photometric_op_and_astrometric_op_agree_on_A():
    """
    Given the same binary parameters,
    When the photometric Op (astrometry=False, one output) and the
      astrometric Op (three outputs) evaluate A,
    Then A agrees to 1e-12 in both cases and not necessarily bitwise: VBM's
      astrometry flag moves the binary magnifications' last bit at some
      epochs (BinaryMag2 always at the probed point, BinaryMag0 on some of
      a trajectory's epochs), which is exactly why the two are separate
      instances and the light curve never sees the flag.
    """
    times = T0 + np.linspace(-10.0, 10.0, 7)
    obs = np.zeros((len(times), 3))
    for use_rho, p in (
        (False, [T0, U0, 20.0, 0.0, 0.0, 1.2, 0.3, 30.0]),
        (True, [T0, U0, 20.0, 0.0, 0.0, 0.02, 1.2, 0.3, 30.0]),
    ):
        phot = VBMDirectMagOp(coords=COORDS, n_companions=1, use_rho=use_rho)
        astro = VBMDirectMagOp(
            coords=COORDS, n_companions=1, use_rho=use_rho, astrometry=True
        )
        (A_phot,) = _op_eval(phot, p, times, obs)
        A_astro, dtau, dbeta = _op_eval(astro, p, times, obs)
        np.testing.assert_allclose(A_astro, A_phot, rtol=1e-12)
        assert np.all(np.isfinite(dtau)) and np.all(np.isfinite(dbeta))


def test_nan_proposal_gives_three_nan_outputs():
    op = VBMDirectMagOp(
        coords=COORDS, n_companions=1, use_rho=False, astrometry=True
    )
    p = [T0, U0, 20.0, 0.0, 0.0, np.nan, 0.3, 30.0]
    with pytest.warns(RuntimeWarning, match="non-finite"):
        outs = _op_eval(op, p, [T0, T0 + 1.0], np.zeros((2, 3)))
    assert len(outs) == 3
    for o in outs:
        assert o.shape == (2,) and np.all(np.isnan(o))


# ----------------------------------------------------------------------
# End to end: a binary-lens astrometric dataset through the instrument
# ----------------------------------------------------------------------
def test_binary_lens_astrometry_builds_evaluates_and_plots(
    tmp_path_factory,
):
    """
    Given a binary-lens event with a photometric light curve and an abs
      astrometric dataset of the source,
    When the system is built,
    Then the astrometric logp is finite, the compiled plot model runs on a
      grid (the Op accepts the plotter's tensor inputs), and the plotted
      shift differs from the point-lens closed form -- the binary centroid
      is really what is being used.
    """
    from test_astrometric_microlensing import (
        _astrometric_times,
        _write_abs_astrometry,
        _write_photometry,
    )

    tmp_dir = tmp_path_factory.mktemp("binastro")
    t_lc = np.linspace(T0 - 150.0, T0 + 150.0, 300)
    _write_photometry(tmp_dir / "lc.dat", t_lc, _theta_E_truth(), 5.0)
    t_a = _astrometric_times()
    _write_abs_astrometry(
        tmp_dir / "astrom.abs",
        t_a,
        np.zeros_like(t_a),
        np.zeros_like(t_a),
        0.1,
    )
    cfg = _full_config(tmp_dir)
    cfg["planet"] = [{"name": "Lb"}]
    cfg["lens"].append({"body": "planet.Lb", "name": "Lb"})
    params = _base_params()
    params["planet.Lb.mass"] = {"initval": 200.0}  # M_J: q ~ 0.4
    params["lens.Lb.s"] = {"initval": 1.1}
    params["lens.Lb.alpha"] = {"initval": 40.0}
    system, model = _build(cfg, params)
    assert system.mulensevent.uses_op()
    ip = model.initial_point()
    assert np.isfinite(_astro_logp(model, ip))

    inst = system.astrometryinstrument
    inst.compile_plotters(model, system)
    vals = _compile_at_start(model, [p.value for p in system.plot_params])
    vals = [np.asarray(v) if np.ndim(v) else float(v) for v in vals]
    dE_full, dN_full = inst._eval_lensed(system, 0, t_a, vals)
    assert np.all(np.isfinite(dE_full)) and np.all(np.isfinite(dN_full))
    point = {p.label: v for p, v in zip(system.plot_params, vals)}
    lE, lN = inst._linear_terms(inst.datasets[0], t_a, point, system)
    shift = np.hypot(dE_full - lE, dN_full - lN)
    assert shift.max() > 0.05 * _theta_E_truth()
