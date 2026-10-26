"""The microlensing magnification must honour the band's limb-darkening LAW.

Only the LINEAR coefficient used to reach the magnification (`vbm.a1`,
`set_limb_coeff_u`), while `Band`'s DEFAULT law is quadratic.  On a band whose
limb darkening only microlensing reads, that had two consequences:

  * the source profile was the wrong shape -- a linear law standing in for a
    quadratic one; and
  * the magnification was a function of u1 alone, so exactly one combination
    of the sampled Kipping pair (q1, q2) was LIKELIHOOD-FREE: sampled,
    reported, constrained by nothing but its prior.

Every shipped example hid this by writing `ld_law: "linear"` by hand.

The tests below pin the four claims the fix rests on, all by EVALUATING
magnifications rather than by reading the source:

  1. Turning the quadratic profile on is a no-op at u2 = 0, so an existing
     linear-band fit cannot move (test_quadratic_at_zero_u2_is_linear).
  2. A non-zero u2 measurably changes the curve, so the coefficient is really
     being consumed (test_u2_changes_the_finite_source_curve).
  3. The param-vector layout is self-describing, because u2 now sits where a
     naive `p[-1]` used to find u1 (test_param_labels_name_u2).
  4. Under the default backend every finite-source single lens goes through
     VBM's ESPL whatever the band's law, and the linear-only MulensModel
     backend stays available as the A/B reference
     (test_single_lens_finite_source_backend_is_the_backend).  The reason
     that used to keep a linear band on MulensModel -- a ~5 mmag
     disagreement blamed on Yoo04's tables -- is pinned against a direct
     integration of the disk (test_espl_accuracy_against_direct_integration).
"""

import logging

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest
from scipy.integrate import quad
from test_band_autopin_ld import _mulens_config, _mulens_params, _write_pspl_lc

from exozippy.components.mulensing.op import (
    MulensMagOp,
    VBMDirectMagOp,
)
from exozippy.system import System

COORDS = "264.0d -27.0d"
T0, TE = 2455000.0, 25.0

# A RESOLVED source.  With rho << |u_0| the source is a point as far as the
# magnification is concerned and every assertion below would pass for the
# wrong reason.
RHO = 0.02
U0 = 0.01


def _times(n=200, span_in_t_star=6.0):
    w = span_in_t_star * RHO * TE
    return np.linspace(T0 - w, T0 + w, n)


def _binary_params(u1, u2=None):
    p = [T0, U0, TE, 0.0, 0.0, RHO, 0.9, 0.3, 35.0, u1]
    if u2 is not None:
        p.append(u2)
    return np.array(p, dtype=float)


def _single_params(u1, u2=None):
    p = [T0, U0, TE, 0.0, 0.0, RHO, u1]
    if u2 is not None:
        p.append(u2)
    return np.array(p, dtype=float)


def _run(op, params, times):
    """Evaluate an Op outside a graph.

    VBMDirectMagOp exposes ``_compute``; the MulensModel Ops only have
    ``perform``, so go through the Op interface both understand.
    """
    obs = np.zeros((len(times), 3))
    out = [[None]]
    op.perform(None, [np.asarray(params, dtype=float), times, obs], out)
    return np.asarray(out[0][0], dtype=float)


@pytest.mark.parametrize("n_companions", [0, 1])
def test_quadratic_at_zero_u2_is_linear(n_companions):
    """
    Given a finite-source lens and a limb-darkening coefficient u1,
    When the quadratic profile is selected with u2 = 0,
    Then the magnification matches the linear profile to machine precision.

    This is what makes the fix safe to apply unconditionally rather than
    behind a flag: every existing `ld_law: linear` configuration keeps its
    answer, so the change can only move a fit that was already wrong.
    """
    # Arrange
    times = _times()
    make = _single_params if n_companions == 0 else _binary_params
    common = dict(
        coords=COORDS,
        n_companions=n_companions,
        use_rho=True,
        bandpass="I",
    )
    lin_op = VBMDirectMagOp(quadratic_ld=False, **common)
    quad_op = VBMDirectMagOp(quadratic_ld=True, **common)

    # Act
    a_lin = _run(lin_op, make(0.55), times)
    a_quad = _run(quad_op, make(0.55, 0.0), times)

    # Assert
    assert np.all(np.isfinite(a_lin)) and np.all(a_lin > 1.0)
    np.testing.assert_allclose(a_quad, a_lin, rtol=1e-13, atol=0.0)


@pytest.mark.parametrize("n_companions", [0, 1])
def test_u2_changes_the_finite_source_curve(n_companions):
    """
    Given the same u1,
    When u2 is given a physical non-zero value,
    Then the magnification changes by far more than the photometric precision
      these light curves are measured to.

    The point is not the exact size but that the coefficient is CONSUMED: the
    old code accepted a quadratic band and silently produced the u2 = 0 curve.
    """
    # Arrange
    times = _times()
    make = _single_params if n_companions == 0 else _binary_params
    op = VBMDirectMagOp(
        coords=COORDS,
        n_companions=n_companions,
        use_rho=True,
        bandpass="I",
        quadratic_ld=True,
    )

    # Act
    a_0 = _run(op, make(0.55, 0.0), times)
    a_2 = _run(op, make(0.55, 0.20), times)
    dmag_mmag = 2.5 * np.log10(a_2 / a_0) * 1000.0

    # Assert -- well above any plausible per-epoch error, and not a constant
    # offset (a rescaling of u1 could absorb that; a shape change cannot).
    assert np.max(np.abs(dmag_mmag)) > 1.0
    assert np.std(dmag_mmag) > 0.1


@pytest.mark.parametrize("n_companions", [0, 1])
def test_param_labels_name_u2(n_companions):
    """
    Given a quadratic-LD Op,
    When its parameter labels are listed,
    Then band.u2 is named and is last.

    `_compute` used to read u1 as `p[-1]`.  With u2 optionally following it,
    that index means a different parameter depending on the band's law, so the
    layout has to be self-describing -- this is what the non-finite guard
    prints when it names the offending parameter.
    """
    # Arrange
    op = VBMDirectMagOp(
        coords=COORDS,
        n_companions=n_companions,
        use_rho=True,
        bandpass="I",
        quadratic_ld=True,
    )
    make = _single_params if n_companions == 0 else _binary_params

    # Act
    labels = op._param_labels()

    # Assert
    assert labels[-2:] == ["band.u1", "band.u2"]
    assert len(labels) == len(make(0.55, 0.20))


def test_quadratic_ld_without_a_bandpass_is_inert():
    """
    Given quadratic_ld=True but no bandpass,
    When the Op is constructed,
    Then it reports no quadratic law.

    u2 only means anything alongside a u1; a quadratic law with no band would
    otherwise read u2 out of a param vector that never carried one.
    """
    op = VBMDirectMagOp(
        coords=COORDS, n_companions=1, use_rho=True, quadratic_ld=True
    )
    assert op.quadratic_ld is False
    assert "band.u2" not in op._param_labels()


def _disk_integral(u, a):
    """Integral of the point-lens A over a uniform disk of radius ``a`` whose
    center sits ``u`` from the lens: analytic in r from the lens
    (int A r dr = r sqrt(r^2 + 4) / 2), numerical in the polar angle."""

    def g(r):
        return 0.5 * r * np.sqrt(r * r + 4.0)

    if u < a:

        def inside(th):
            s = np.sin(th)
            return g(u * np.cos(th) + np.sqrt(a * a - u * u * s * s))

        return 2.0 * quad(inside, 0.0, np.pi, epsrel=1e-12, limit=200)[0]
    k = a / u

    # theta = arcsin(k sin psi) removes the square-root endpoint behavior.
    def outside(psi):
        st = k * np.sin(psi)
        ct = np.sqrt(1.0 - st * st)
        half = a * np.cos(psi)
        jac = k * np.cos(psi) / ct
        return (g(u * ct + half) - g(max(u * ct - half, 0.0))) * jac

    return 2.0 * quad(outside, 0.0, np.pi / 2, epsrel=1e-12, limit=200)[0]


def _espl_reference(u, rho, u1):
    """Linear-LD finite-source point-lens magnification by direct
    integration: annuli of the uniform disk weighted by I = 1 - u1 (1 - mu),
    with the radius written rho sin(phi) so the limb is not singular."""
    pts = [np.arcsin(u / rho)] if u < rho else None
    annuli = quad(
        lambda phi: np.sin(phi) * _disk_integral(u, rho * np.sin(phi)),
        0.0,
        np.pi / 2,
        epsrel=1e-10,
        limit=200,
        points=pts,
    )[0]
    flux = (1.0 - u1) * _disk_integral(u, rho) + u1 * annuli
    return flux / (np.pi * rho * rho * (1.0 - u1 / 3.0))


def test_espl_accuracy_against_direct_integration():
    """
    Given a linear-LD finite source on a single lens (rho = 0.01, u1 = 0.55)
      and a direct numerical integration of the disk as the reference,
    When VBM's ESPL, MulensModel's Yoo04 alone, and the mulensmodel backend's
      own method window are compared with it,
    Then VBM is within 0.25 mmag from u = 0.05 rho to 6 rho, Yoo04 alone is
      within 0.05 mmag, and the backend window is off by > 2 mmag just
      outside t_0 +/- 5 rho t_E.

    This is the measurement behind routing every finite-source single lens
    to VBM (MulensEvent._resolve_quadratic_ld).  The ~5 mmag VBM-vs-
    MulensModel difference that once kept a linear band on MulensModel was
    blamed on Yoo04's table interpolation; it is the backend's point-source
    window, not Yoo04, and VBM is the accurate side of it.  If the last
    assertion ever fails, the window was fixed -- update mulensing.md.
    """
    # Arrange
    rho, u1 = 0.01, 0.55
    x = np.array([0.05, 0.3, 0.7, 1.0, 1.3, 3.0, 5.2, 6.0])
    u = x * rho
    ref = np.array([_espl_reference(ui, rho, u1) for ui in u])
    # A trajectory with u_0 = 0.05 rho, timed to pass through those u.
    t0, te = 0.0, 20.0
    u0 = u[0]
    times = t0 + te * np.sqrt(np.maximum(u * u - u0 * u0, 0.0))
    params = [t0, u0, te, 0.0, 0.0, rho, u1]
    vbm_op = VBMDirectMagOp(
        coords=COORDS, n_companions=0, use_rho=True, bandpass="I"
    )
    window_op = MulensMagOp(
        coords=COORDS, mag_method="auto_vbbl", use_rho=True, bandpass="I"
    )
    yoo_op = MulensMagOp(
        coords=COORDS,
        mag_method="finite_source_LD_Yoo04",
        use_rho=True,
        bandpass="I",
    )

    # Act
    def mmag(a):
        return 2.5 * np.log10(a / ref) * 1000.0

    d_vbm = mmag(_run(vbm_op, params, times))
    d_window = mmag(_run(window_op, params, times))
    d_yoo = mmag(_run(yoo_op, params, times))

    # Assert
    assert np.max(np.abs(d_vbm)) < 0.25, d_vbm
    assert np.max(np.abs(d_yoo)) < 0.05, d_yoo
    outside = x > 5.0
    assert np.all(np.abs(d_window[outside]) > 2.0), d_window
    assert np.max(np.abs(d_window[~outside])) < 0.05, d_window


# ---------------------------------------------------------------------------
# Dispatch: which backend gets the second coefficient
# ---------------------------------------------------------------------------


def _fs_system(tmp_path, ld_law, backend=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    lc = _write_pspl_lc(tmp_path / "lc.dat")
    params = _mulens_params(finite_source=True)
    # rho and u_0 belong to the SOURCE's trajectory; the backend is an
    # event-level choice.
    params["source.Source.rho"] = {"initval": RHO}
    params["source.Source.u_0"] = {"initval": U0}
    config = _mulens_config(
        lc,
        bands=[{"name": "I", "filter": "I", "ld_law": ld_law}],
        finite_source=True,
    )
    if backend is not None:
        config["mulensevent"][0]["backend"] = backend
    system = System(config, user_params=params)
    system.prepare()
    system.build_model()
    return system


def test_resolver_returns_u2_only_for_a_quadratic_band(tmp_path):
    """
    Given a finite-source fit,
    When the shared LD resolver is asked,
    Then u2 comes back exactly when the band declares the quadratic law.

    The guard is on the MANIFEST, not the law string: with every band linear
    the parameter does not exist at all, so `"u2" in band.manifest` is the
    only safe test (the same one transit.py and rm.py use).
    """
    for law, expected in (("quadratic", True), ("linear", False)):
        system = _fs_system(tmp_path / law, law)
        u1, u2, bandpass = (
            system.mulensinstrument._finite_source_limb_darkening(system, 0)
        )
        assert bandpass == "I"
        assert u1 is not None
        assert (u2 is not None) is expected
        assert ("u2" in system.band.manifest) is expected


def test_mulensmodel_backend_drops_u2_and_says_so(tmp_path, caplog):
    """
    Given a quadratic band and the linear-only MulensModel backend,
    When the dispatcher resolves the LD law,
    Then u2 is dropped and a warning names both fixes -- once.

    Warn-and-continue rather than raise: it describes the topology, and
    raising would break the A/B reference backend for a band whose default law
    the user never chose.  But it must not be SILENT, which is the whole
    defect being fixed.
    """
    # Arrange
    system = _fs_system(tmp_path, "quadratic", backend="mulensmodel")
    event = system.mulensevent
    _, u2, bandpass = system.mulensinstrument._finite_source_limb_darkening(
        system, 0
    )
    assert u2 is not None

    # Act
    with caplog.at_level(logging.WARNING):
        first = event._resolve_quadratic_ld(u2, bandpass)
        second = event._resolve_quadratic_ld(u2, bandpass)

    # Assert
    assert first is False and second is False
    hits = [
        r for r in caplog.records if "quadratic limb-darkening" in r.message
    ]
    assert len(hits) == 1
    assert "backend: vbm_direct" in hits[0].message
    assert "ld_law: linear" in hits[0].message


def test_vbm_backend_honours_u2(tmp_path):
    """
    Given a quadratic band and the default vbm_direct backend,
    When the dispatcher resolves the LD law,
    Then u2 is accepted, with no warning to give.
    """
    system = _fs_system(tmp_path, "quadratic")
    _, u2, bandpass = system.mulensinstrument._finite_source_limb_darkening(
        system, 0
    )
    assert system.mulensevent.backend == "vbm_direct"
    assert system.mulensevent._resolve_quadratic_ld(u2, bandpass) is True


def test_no_u2_means_no_quadratic_law(tmp_path):
    """
    Given a LINEAR band,
    When the dispatcher resolves the LD law,
    Then it reports no quadratic law -- the VBM Op is built with LDlinear.
    """
    system = _fs_system(tmp_path, "linear")
    _, u2, bandpass = system.mulensinstrument._finite_source_limb_darkening(
        system, 0
    )
    assert u2 is None
    assert system.mulensevent._resolve_quadratic_ld(u2, bandpass) is False


@pytest.mark.parametrize(
    "law, backend",
    [
        ("linear", None),
        ("quadratic", None),
        ("linear", "mulensmodel"),
    ],
)
def test_single_lens_finite_source_backend_is_the_backend(
    tmp_path, law, backend
):
    """
    Given a finite-source SINGLE lens,
    When the magnification graph is built,
    Then the default backend produces the VBM ESPL Op whatever the band's
      law (LDquadratic exactly when the band is quadratic), and an explicit
      `backend: mulensmodel` still produces the MulensModel Op.

    A linear band used to keep MulensModel under the default backend; the
    rule recorded in MulensEvent._resolve_quadratic_ld is now the backend
    alone.  The mulensmodel case pins that the A/B reference survives.
    """
    # Arrange
    system = _fs_system(tmp_path / f"{law}_{backend}", law, backend=backend)
    u1, u2, bandpass = system.mulensinstrument._finite_source_limb_darkening(
        system, 0
    )
    t_in = pt.dvector("t")
    obs_in = pt.dmatrix("obs")

    # Act
    node = system.mulensevent.get_magnification_op(
        t_in, obs_in, system, index=0, u1=u1, u2=u2, bandpass=bandpass
    )
    op = node.owner.op

    # Assert
    if backend == "mulensmodel":
        assert isinstance(op, MulensMagOp)
        assert op.use_rho and op.bandpass == "I"
    else:
        assert isinstance(op, VBMDirectMagOp)
        assert op.n_companions == 0
        assert op.use_rho and op.bandpass == "I"
        assert op.quadratic_ld is (law == "quadratic")


def test_the_graph_actually_responds_to_u2(tmp_path):
    """
    Given a compiled finite-source magnification graph on a quadratic band,
    When u2 alone is varied,
    Then the magnification changes.

    The end-to-end version of test_u2_changes_the_finite_source_curve: it goes
    through the resolver, the dispatcher and the real PyMC nodes, so it would
    catch a u2 that is plumbed but never reaches the backend.
    """
    # Arrange
    system = _fs_system(tmp_path, "quadratic")
    u1, u2, bandpass = system.mulensinstrument._finite_source_limb_darkening(
        system, 0
    )
    t_in = pt.dvector("t")
    obs_in = pt.dmatrix("obs")
    u1_in = pt.dscalar("u1")
    u2_in = pt.dscalar("u2")
    node = system.mulensevent.get_magnification_op(
        t_in, obs_in, system, index=0, u1=u1_in, u2=u2_in, bandpass=bandpass
    )
    fn = pytensor.function(
        inputs=[t_in, obs_in, u1_in, u2_in]
        + [p.value for p in system.plot_params],
        outputs=node,
        on_unused_input="ignore",
    )
    inst = system.mulensinstrument
    times = inst.time[:60]
    obs = inst._abs_to_delta(times, inst.get_observer_position(times))
    # The one supported way to get the start values of the plot parameters:
    # Parameter.value draws from the PRIOR, so evaluating it here would
    # magnify a random trajectory (and, at these u_0, usually A = 1).
    values = inst._point_to_plot_params({}, system)

    # Act
    a_0 = np.asarray(fn(times, obs, 0.55, 0.0, *values), dtype=float)
    a_2 = np.asarray(fn(times, obs, 0.55, 0.20, *values), dtype=float)

    # Assert
    assert np.all(np.isfinite(a_0)) and np.all(np.isfinite(a_2))
    assert np.max(np.abs(2.5 * np.log10(a_2 / a_0) * 1000.0)) > 1.0
