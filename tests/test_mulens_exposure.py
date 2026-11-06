"""Exposure smearing of microlensing photometry (mulensing.md, "Exposure
smearing"; instrument.md, "Exposure smearing").

A light curve with ``ninterp > 1`` is modeled as the exposure-mean flux,
``f_s <A> + f_b``: the magnification is averaged on Gauss-Legendre nodes
spanning each exposure, with the observer at its ephemeris position at
every node.  Pinned on a point-source point-lens event against the
model's OWN instantaneous magnification (the unsmeared light curve's
compiled model, observer placed at each time, so whatever parallax the
start carries is in both sides):

  - (d) near the peak, with an exposure comparable to u0 t_E, the smeared
    model equals a dense numerical average of the instantaneous A(t) and
    drops measurably below the instantaneous peak;
  - an unsmeared light curve beside a smeared one keeps its instantaneous
    model exactly;
  - the plotted model is the likelihood's model (the builder contract);
  - ninterp 1 gives the same start logp as no exposure keys.
"""

import os

import numpy as np
import pytensor
import pytest

from exozippy.system import System

T0 = 2460025.0
U0 = 0.30
TE = 25.0
EXPTIME_S = 3.0 * 86400.0  # 3 d, comparable to u0 t_E = 7.5 d
NINTERP = 7


def _A(t):
    """Paczynski's A(t) at the seeded geometry (the synthetic data only)."""
    tau = (t - T0) / TE
    u = np.sqrt(tau**2 + U0**2)
    return (u**2 + 2.0) / (u * np.sqrt(u**2 + 4.0))


def _config(smeared_extra):
    return {
        "star": [{"name": "Lens"}, {"name": "Source"}],
        "mulensevent": [{"finite_source": False, "t0_par": T0}],
        "lens": [{"body": "star.Lens"}],
        "source": [{"body": "star.Source"}],
        "mulensinstrument": [
            {
                "name": "Long",
                "file": "long.dat",
                "data_format": "flux",
                **smeared_extra,
            },
            {"name": "Short", "file": "short.dat", "data_format": "flux"},
        ],
    }


def _params():
    p = {
        "source.Source.t_0": {"initval": T0},
        "source.Source.u_0": {"initval": U0},
        "mulensevent.t_E": {"initval": TE},
        "star.radius": {"sigma": 0.0},
        "star.teff": {"sigma": 0.0},
        "star.feh": {"sigma": 0.0},
    }
    for nm in ("Lens", "Source"):
        p[f"star.{nm}.ra"] = {"initval": 264.0, "sigma": 0}
        p[f"star.{nm}.dec"] = {"initval": -27.0, "sigma": 0}
    return p


def _build(tmpdir, smeared_extra):
    t = np.linspace(T0 - 40.0, T0 + 40.0, 81)
    for name in ("long.dat", "short.dat"):
        np.savetxt(
            os.path.join(str(tmpdir), name),
            np.column_stack([t, 100.0 * _A(t) + 20.0, np.full_like(t, 1.0)]),
            fmt="%.17g",
        )
    cwd = os.getcwd()
    os.chdir(str(tmpdir))
    try:
        system = System(_config(smeared_extra), user_params=_params())
        system.prepare()
        model = system.build_model()
    finally:
        os.chdir(cwd)
    return system, model


@pytest.fixture(scope="module")
def smeared(tmp_path_factory):
    d = tmp_path_factory.mktemp("mulens_exposure")
    return _build(d, {"exptime_s": EXPTIME_S, "ninterp": NINTERP})


def _at_vals(system, node, vals):
    """A likelihood node with ``system.plot_params`` set to ``vals`` -- the
    inputs the compiled plot functions take -- so both sides see one point."""
    fn = pytensor.function(
        [p.value for p in system.plot_params],
        node,
        on_unused_input="ignore",
    )
    return np.asarray(fn(*vals), dtype=float)


def _plot_vals(system):
    out = []
    for p in system.plot_params:
        v = np.asarray(p.initval, dtype=np.float64)
        out.append(
            float(np.squeeze(v))
            if getattr(p.value, "ndim", 0) == 0
            else np.atleast_1d(v)
        )
    return out


def _instant_A(mi, i, t, vals):
    """Light curve ``i``'s INSTANTANEOUS magnification at times ``t`` (any
    shape), from the unsmeared light curve's compiled model with the
    observer at each time -- the model's own geometry, parallax included."""
    fs, fb = mi._compiled_flux(*vals)
    flat = np.asarray(t, dtype=float).ravel()
    obs = mi._abs_to_delta(
        flat, mi.get_observer_position(flat, observer_location="earth")
    )
    F = mi._compiled_model_flux(flat, obs, 1, *vals)
    return ((np.asarray(F) - fb[1]) / fs[1]).reshape(np.shape(t))


def test_smeared_peak_is_the_averaged_magnification(smeared):
    """
    Given a PSPL event (u0 = 0.3, t_E = 25 d) and a light curve with 3-day
    exposures on 7 nodes,
    When the likelihood's model flux is evaluated,
    Then (F - f_b)/f_s equals a dense Simpson average of the model's own
    instantaneous magnification over each exposure (observer at each
    time), and at the peak it sits measurably below the instantaneous
    value -- the exposure averages over the curvature.
    """
    system, _ = smeared
    mi = system.mulensinstrument
    vals = _plot_vals(system)
    flux = _at_vals(system, mi._model_flux_data_node, vals)
    fs, fb = mi._compiled_flux(*vals)
    rows = mi.rows(0)
    t = mi.time[rows]
    A_smeared = (flux[rows] - fb[0]) / fs[0]

    u = np.linspace(-0.5, 0.5, 2001)
    w = np.ones_like(u)
    w[1:-1:2], w[2:-1:2] = 4.0, 2.0
    w /= w.sum()
    A_dense = (
        _instant_A(mi, 1, t[:, None] + u[None, :] * EXPTIME_S / 86400.0, vals)
        @ w
    )
    np.testing.assert_allclose(A_smeared, A_dense, rtol=2e-6)

    A_mid = _instant_A(mi, 1, t, vals)
    peak = np.argmax(A_mid)
    assert A_mid[peak] - A_smeared[peak] > 1e-2


def test_an_unsmeared_light_curve_keeps_its_instantaneous_model(
    smeared, tmp_path
):
    """
    Given a second light curve with no exposure keys beside the smeared one,
    When the likelihood's model flux is evaluated,
    Then its rows equal, to rounding, the same light curve's model in a
    system where nothing smears.
    """
    system, _ = smeared
    mi = system.mulensinstrument
    flux = _at_vals(system, mi._model_flux_data_node, _plot_vals(system))
    plain, _ = _build(tmp_path, {})
    pmi = plain.mulensinstrument
    ref = _at_vals(plain, pmi._model_flux_data_node, _plot_vals(plain))
    rows = mi.rows(1)
    # To rounding: in the smeared system these rows pass through the node
    # mean too (one node, weight 1), which pytensor may fuse differently.
    np.testing.assert_allclose(flux[rows], ref[pmi.rows(1)], rtol=1e-14)


def test_the_plotted_model_is_the_likelihood_model(smeared):
    """
    Given both light curves,
    When the plot path's compiled model (the one plot_data draws, and
    evaluates at the data epochs for a detrended GP file) is evaluated at
    each light curve's own epochs,
    Then it equals the likelihood's model flux on those rows.
    """
    system, _ = smeared
    mi = system.mulensinstrument
    vals = _plot_vals(system)
    flux = _at_vals(system, mi._model_flux_data_node, vals)
    for i in range(2):
        rows = mi.rows(i)
        plotted = mi._compiled_model_flux(
            mi.time[rows], mi.observer_pos[rows], i, *vals
        )
        np.testing.assert_allclose(plotted, flux[rows], rtol=1e-12)


def test_ninterp_one_is_the_instantaneous_model(tmp_path, smeared):
    """
    Given exptime_s stated with ninterp 1, and no exposure keys at all,
    When the start logps are compared,
    Then they are identical.
    """
    a_dir = tmp_path / "a"
    b_dir = tmp_path / "b"
    a_dir.mkdir()
    b_dir.mkdir()
    _, ma = _build(a_dir, {"exptime_s": EXPTIME_S, "ninterp": 1})
    _, mb = _build(b_dir, {})
    assert ma.compile_logp()(
        ma.initial_point(random_seed=0)
    ) == mb.compile_logp()(mb.initial_point(random_seed=0))
