"""
Rossiter-McLaughlin (Hirano+2011) component -- unit tests.

The differentiable kernel (components/rm.py) is validated in isolation here.
Cross-checking the full RV distortion against allesfast's numpy reference
(itself validated vs IDL / EXOFASTv2) is the intended regression, mirroring
tests/test_torres.py's "pin the port against reference output" pattern.
"""

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from exozippy.components import rm


# --------------------------------------------------------------------------
# 1. The ported symbolic Bessel J0 must match scipy.special.j0.
# --------------------------------------------------------------------------
def test_bessj0_matches_scipy():
    """Given a range spanning both branches (|x|<8 and |x|>=8), When we
    evaluate the symbolic bessj0, Then it matches scipy.special.j0."""
    from scipy.special import j0 as scipy_j0

    x = pt.dvector("x")
    f = pytensor.function([x], rm.bessj0(x))
    xs = np.linspace(0.01, 60.0, 400)  # crosses the x=8 branch switch
    np.testing.assert_allclose(f(xs), scipy_j0(xs), atol=1e-6, rtol=1e-6)


def test_bessj0_is_differentiable():
    """bessj0 must have a finite gradient (needed for NUTS/numpyro)."""
    x = pt.dscalar("x")
    g = pytensor.function([x], pt.grad(rm.bessj0(x), x))
    for xv in (0.5, 5.0, 8.0, 20.0):
        assert np.isfinite(g(xv))


# --------------------------------------------------------------------------
# 2. rm_delta_v_core: finite output, and the correct qualitative shape.
# --------------------------------------------------------------------------
def _eval_delta_v(xv, yv, zv, fluxv, vsini=5000.0):
    x = pt.dvector("x")
    y = pt.dvector("y")
    z = pt.dvector("z")
    flx = pt.dvector("f")
    dv = rm.rm_delta_v_core(x, y, z, flx, vsini, 0.3, 0.2, n_sigma=101)
    fn = pytensor.function([x, y, z, flx], dv)
    return fn(xv, yv, zv, fluxv)


def test_delta_v_finite_and_zero_out_of_transit():
    """Out of transit (flux == 1) the RM anomaly is exactly zero; in transit
    it is finite and nonzero."""
    x = np.array([-0.6, -0.3, 0.0, 0.3, 0.6])
    y = np.zeros_like(x)
    z = np.ones_like(x)
    flux = np.array([1.0, 0.99, 0.99, 0.99, 1.0])  # dips only for the middle 3
    dv = _eval_delta_v(x, y, z, flux)
    assert np.all(np.isfinite(dv))
    assert dv[0] == pytest.approx(0.0) and dv[-1] == pytest.approx(0.0)
    assert np.any(np.abs(dv[1:4]) > 0.0)


def test_delta_v_antisymmetric_for_aligned_chord():
    """For lambda=0 (x already the spin-axis coordinate) a symmetric transit
    chord gives an antisymmetric RM curve: dv(-x) approx -dv(+x)."""
    x = np.array([-0.4, 0.4])
    y = np.zeros_like(x)
    z = np.ones_like(x)
    flux = np.array([0.99, 0.99])
    dv = _eval_delta_v(x, y, z, flux)
    assert dv[0] == pytest.approx(-dv[1], rel=1e-3)


def test_occultation_is_zeroed():
    """z < 0 (planet behind the star) contributes no RM."""
    x = np.array([0.3])
    y = np.array([0.0])
    z = np.array([-1.0])
    flux = np.array([0.99])
    assert _eval_delta_v(x, y, z, flux)[0] == pytest.approx(0.0)


# --------------------------------------------------------------------------
# 3. Cross-check vs allesfast numpy reference (skipped unless available).
# --------------------------------------------------------------------------
# Reference RM vector [m/s] from allesfast's exact Hirano+2011 kernel
# (rm_models/Hirano2011.py: _compute_m_array + _rm_delta_v, its own adaptive
# sigma grid), fed the geometry+flux below. allesfast itself is validated vs
# IDL / EXOFASTv2. Regenerate with scratchpad/gen_rm_fixture.py.
# Scenario mirrors allesfast tests/test_rm_beta_ip _RM_KWARGS:
#   ar=10, period=3, inc=87 deg, e=0, omega=90 deg, lambda=30 deg, Rp/Rstar=0.1,
#   vsini=5000, vzeta=3000, vbeta=4000, vgamma=1000 m/s, u1=0.3, u2=0.2.
_REF_X = np.array(
    [
        -6.4499677409e-01,
        -4.6390934721e-01,
        -2.8261843435e-01,
        -1.0120355571e-01,
        8.0255714157e-02,
        2.6167978121e-01,
        4.4298906687e-01,
        6.2410404288e-01,
        8.0494526623e-01,
        9.8543341399e-01,
        1.1654893181e00,
    ]
)
_REF_Y = np.array(
    [
        -9.7340208191e-01,
        -8.7004230718e-01,
        -7.6630090314e-01,
        -6.6222337419e-01,
        -5.5785537219e-01,
        -4.5324267638e-01,
        -3.4843117335e-01,
        -2.4346683690e-01,
        -1.3839570784e-01,
        -3.3263873856e-02,
        7.1882550766e-02,
    ]
)
_REF_Z = np.array(
    [
        9.9315893767e00,
        9.9512720042e00,
        9.9665896748e00,
        9.9775356698e00,
        9.9841051880e00,
        9.9862953475e00,
        9.9841051880e00,
        9.9775356698e00,
        9.9665896748e00,
        9.9512720042e00,
        9.9315893767e00,
    ]
)
_REF_FLUX = np.array(
    [
        1.0000000000e00,
        9.9784107518e-01,
        9.9238977628e-01,
        9.8916756464e-01,
        9.8730608364e-01,
        9.8669345272e-01,
        9.8730608364e-01,
        9.8916756464e-01,
        9.9238977628e-01,
        9.9784107518e-01,
        1.0000000000e00,
    ]
)
_REF_RM = np.array(
    [
        -0.0000000000e00,
        6.0033663312e00,
        1.3570359529e01,
        7.1275360413e00,
        -6.6564104943e00,
        -2.2235655788e01,
        -3.4118714209e01,
        -3.7969249130e01,
        -3.1132138186e01,
        -9.5929309991e00,
        -0.0000000000e00,
    ]
)


def test_matches_allesfast_reference():
    """The PyTensor port reproduces allesfast's numpy RM kernel (validated vs
    IDL/EXOFASTv2) on a fixed geometry+flux, to well under 1e-3 m/s on a
    ~38 m/s amplitude. Cross-validation, no runtime allesfast dependency."""
    x, y, z, flx = pt.dvectors("x", "y", "z", "flx")
    dv = rm.rm_delta_v_core(
        x,
        y,
        z,
        flx,
        5000.0,
        0.3,
        0.2,
        vzeta=3000.0,
        vbeta=4000.0,
        vgamma=1000.0,
        n_sigma=201,
    )
    port = pytensor.function([x, y, z, flx], dv)(
        _REF_X, _REF_Y, _REF_Z, _REF_FLUX
    )
    np.testing.assert_allclose(port, _REF_RM, atol=1e-3)


# --------------------------------------------------------------------------
# 4. End-to-end: the KELT-17 example (transit + RV with rm:) builds and yields
#    a finite initial logp (skipped only if the example files are absent).
# --------------------------------------------------------------------------
def test_rm_system_logp_finite():
    """The KELT-17 RM example (examples/kelt17) -- a transiting planet with
    two in-transit RV sequences tagged `rm: b` -- builds end to end and yields
    a finite initial logp."""
    import os

    import yaml

    from exozippy.system import System

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    exdir = os.path.join(root, "examples", "kelt17")
    if not os.path.exists(os.path.join(exdir, "kelt17.yaml")):
        pytest.skip("kelt17 example not present")
    with open(os.path.join(exdir, "kelt17.yaml")) as fh:
        cfg = yaml.safe_load(fh)
    cwd = os.getcwd()
    try:
        os.chdir(exdir)  # config data paths are relative
        s = System(cfg)
        s.prepare()
        model = s.build_model()
        lp = float(model.compile_logp()(model.initial_point()))
    finally:
        os.chdir(cwd)
    assert np.isfinite(lp), f"RM example logp not finite: {lp}"


def test_rm_system_with_linear_ld_builds():
    """Given the KELT-17 RM example with every band on `ld_law: linear` (so
    Band's manifest has no u2 at all), When the model is built, Then it builds
    and yields a finite logp AND a finite gradient.

    compute_rm_rv used to read band.u2 unconditionally and die with an
    AttributeError before the model existed (review 2.4.3). The gradient is
    asserted too: a linear-LD fit still has to reach the sampler.
    """
    import os

    import yaml

    from exozippy.system import System

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    exdir = os.path.join(root, "examples", "kelt17")
    if not os.path.exists(os.path.join(exdir, "kelt17.yaml")):
        pytest.skip("kelt17 example not present")
    with open(os.path.join(exdir, "kelt17.yaml")) as fh:
        cfg = yaml.safe_load(fh)
    for band_cfg in cfg["band"]:
        band_cfg["ld_law"] = "linear"

    cwd = os.getcwd()
    try:
        os.chdir(exdir)  # config data paths are relative
        s = System(cfg)
        s.prepare()
        assert "u2" not in s.band.manifest  # the configuration under test
        model = s.build_model()
        point = model.initial_point()
        lp = float(model.compile_logp()(point))
        grad = np.asarray(model.compile_dlogp()(point))
    finally:
        os.chdir(cwd)

    assert np.isfinite(lp), f"linear-LD RM example logp not finite: {lp}"
    assert np.all(np.isfinite(grad)), (
        "linear-LD RM example has a non-finite gradient"
    )


# --------------------------------------------------------------------------
# 5. Hirano+2010 fast closed-form kernel (rm_delta_v_hirano2010), selectable
#    via `rm_model: hirano2010`. ~25x faster than the H2011 disk integral; a
#    Taylor series that is accurate only for slow/moderate rotators.
# --------------------------------------------------------------------------
def _eval_h2010(xv, fluxv, vsini=5000.0, vbeta=4000.0):
    x = pt.dvector("x")
    flx = pt.dvector("f")
    dv = rm.rm_delta_v_hirano2010(x, flx, vsini, vbeta)
    return pytensor.function([x, flx], dv)(xv, fluxv)


def test_hirano2010_matches_closed_form():
    """The PyTensor H2010 kernel reproduces rmfit's closed-form series exactly
    (RMHirano.evaluate): v = -1000 vp F [(2b^2+2s^2)/(2b^2+s^2)]^1.5 *
    (1 - vp^2/D + vp^4/2D^2), with vp = vsini x, s = vsini/1.31, D = 2b^2+s^2."""
    x = np.linspace(-1.1, 1.1, 40)
    flux = 1.0 - 0.008 * np.exp(
        -((x / 0.5) ** 2)
    )  # a shallow transit-like dip
    vsini, vbeta = 6000.0, 4000.0
    port = _eval_h2010(x, flux, vsini, vbeta)
    vk, bk = vsini / 1e3, vbeta / 1e3
    sk = vk / 1.31
    vp = vk * x
    F = 1.0 - flux
    D = 2 * bk**2 + sk**2
    pref = ((2 * bk**2 + 2 * sk**2) / D) ** 1.5
    ref = -1000.0 * vp * F * pref * (1.0 - vp**2 / D + vp**4 / (2 * D**2))
    np.testing.assert_allclose(port, ref, atol=1e-9)


def test_hirano2010_zero_out_of_transit():
    """flux == 1 (F = 0) -> the RM anomaly is exactly zero."""
    x = np.array([-0.4, 0.0, 0.4])
    assert np.allclose(_eval_h2010(x, np.ones_like(x)), 0.0)


def test_hirano2010_is_differentiable():
    """Finite gradient wrt vsini (needed for NUTS/numpyro)."""
    x = pt.dvector("x")
    flx = pt.dvector("f")
    vs = pt.dscalar("vs")
    dv = rm.rm_delta_v_hirano2010(x, flx, vs, 4000.0)
    g = pytensor.function([x, flx, vs], pt.grad(pt.sum(dv), vs))
    assert np.isfinite(
        g(np.array([-0.3, 0.3]), np.array([0.99, 0.99]), 5000.0)
    )


def test_hirano2010_agrees_with_2011_at_low_vsini():
    """In its valid slow-rotator regime the fast H2010 series tracks the H2011
    disk integral: at vsini = 2 km/s the peak amplitudes agree to < 10% (they
    diverge for fast rotators, where the vp^2/D >~ 1 series is out of range)."""
    from exozippy.components.limbdark import quad_limb_darkened_flux

    x = pt.dvector("x")
    y = pt.dvector("y")
    z = pt.dvector("z")
    u1, u2, rprs = 0.4, 0.2, 0.0959
    rho = pt.sqrt(x * x + y * y)
    flux = pt.switch(
        pt.ge(z, 0.0), quad_limb_darkened_flux(rho, rprs, u1, u2), 1.0
    )
    f11 = pytensor.function(
        [x, y, z],
        rm.rm_delta_v_core(
            x,
            y,
            z,
            flux,
            2000.0,
            u1,
            u2,
            vzeta=4000.0,
            vbeta=4000.0,
            vgamma=1000.0,
            n_sigma=151,
        ),
    )
    f10 = pytensor.function(
        [x, y, z], rm.rm_delta_v_hirano2010(x, flux, 2000.0, 4000.0)
    )
    xv = np.linspace(-1.25, 1.25, 80)
    yv = np.full_like(xv, 0.53)
    zv = np.ones_like(xv)
    a11 = f11(xv, yv, zv)
    a10 = f10(xv, yv, zv)
    assert np.max(np.abs(a10 - a11)) / np.max(np.abs(a11)) < 0.10


# --------------------------------------------------------------------------
# 6. The RM term is INDEXED to its own instrument's rows, not switched over
#    every instrument's (review 3.6).
# --------------------------------------------------------------------------
def _kelt17_two_instrument_system(tmp_path, n_rm=40):
    """The KELT-17 example with its RV file split across two instruments:
    only the first is tagged `rm: b`.  Returns (system, model, n_rm, n_other).
    """
    import os
    import shutil

    import yaml

    from exozippy.system import System

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    exdir = os.path.join(root, "examples", "kelt17")
    if not os.path.exists(os.path.join(exdir, "kelt17.yaml")):
        pytest.skip("kelt17 example not present")
    work = str(tmp_path)
    for name in os.listdir(exdir):
        shutil.copy(os.path.join(exdir, name), work)

    rv = np.loadtxt(os.path.join(exdir, "KELT-17.TRES.rv"))
    np.savetxt(os.path.join(work, "rm_inst.rv"), rv[:n_rm])
    np.savetxt(os.path.join(work, "orb_inst.rv"), rv[n_rm:])

    with open(os.path.join(exdir, "kelt17.yaml")) as fh:
        cfg = yaml.safe_load(fh)
    cfg["rvinstrument"] = [
        {"name": "TRES_RM", "file": "rm_inst.rv", "rm": "b", "rm_band": "V"},
        {"name": "TRES_ORB", "file": "orb_inst.rv"},
    ]

    cwd = os.getcwd()
    try:
        os.chdir(work)
        system = System(cfg)
        system.prepare()
        model = system.build_model()
    finally:
        os.chdir(cwd)
    return system, model, n_rm, len(rv) - n_rm


def test_rm_is_evaluated_only_on_its_own_instrument_rows(
    tmp_path, monkeypatch
):
    """
    Given: two RV instruments, only one of them tagged `rm: b`
    When: the RV likelihood is built
    Then: compute_rm_rv is evaluated over exactly that instrument's rows

    A pt.switch over the branch VALUES evaluated the Hirano kernel at every
    instrument's timestamps and masked the result -- the JAX where-trap, and
    83% wasted work on this split (the H2011 kernel is a 201 x 64 quadrature
    per row).
    """
    # Arrange: spy on the shared kernel entry point
    from exozippy.components import rm as rm_module

    seen = []
    original = rm_module.compute_rm_rv

    def _spy(system, time, *args, **kwargs):
        seen.append(time)
        return original(system, time, *args, **kwargs)

    monkeypatch.setattr(rm_module, "compute_rm_rv", _spy)

    # Act
    _system, _model, n_rm, n_other = _kelt17_two_instrument_system(tmp_path)

    # Assert: the build_likelihood call (the one with a concrete length) sees
    # only the RM instrument's rows.  The plotting call takes a symbolic grid.
    concrete = []
    for t in seen:
        try:
            concrete.append(int(t.shape[0].eval()))
        except Exception:
            pass
    assert concrete, "compute_rm_rv was never called on concrete times"
    assert n_other > 0  # the split really does hold back rows
    assert concrete == [n_rm], (
        f"RM kernel evaluated over {concrete} rows; the RM instrument has "
        f"{n_rm} of {n_rm + n_other}"
    )


def test_rm_two_instrument_logp_and_gradient_finite_on_both_backends(tmp_path):
    """
    Given: the same two-instrument RM system
    When: logp and dlogp are evaluated on the C backend and on JAX
    Then: both are finite

    The JAX evaluation is the regression guard for the where-trap; the JAX
    SAMPLER path was verified separately by actually sampling this model with
    nuts_sampler="numpyro" (the standing house rule).
    """
    # Arrange
    _system, model, _n_rm, _n_other = _kelt17_two_instrument_system(tmp_path)
    point = model.initial_point()

    # Act
    lp_c = float(model.compile_logp()(point))
    grad_c = np.asarray(model.compile_dlogp()(point))
    lp_jax = float(np.asarray(model.compile_logp(mode="JAX")(point)))
    grad_jax = np.concatenate(
        [
            np.atleast_1d(np.asarray(g)).ravel()
            for g in [model.compile_dlogp(mode="JAX")(point)]
        ]
    )

    # Assert
    assert np.isfinite(lp_c) and np.all(np.isfinite(grad_c))
    assert np.isfinite(lp_jax) and np.all(np.isfinite(grad_jax))
    assert lp_jax == pytest.approx(lp_c, rel=1e-8)
