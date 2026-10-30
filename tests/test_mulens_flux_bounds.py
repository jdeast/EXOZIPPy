"""f_source / f_blend carry no bounds, so the fit is unit-free (review 1.6.18).

A light curve's flux system is whatever its file is in: ~1e-8 for a magnitude
file (F = 10**(-0.4 m)), O(1e4) counts for difference imaging.  f_source and
f_blend are DERIVED (from log_f_total and q_source), so a defaults.yaml bound
on them became a soft barrier measured in those arbitrary units -- it charged
a magnitude file log(1/2) per light curve at the start (the f_source >= 0
barrier's turn-on width dwarfed the flux) and pushed a counts file's baseline
down from above.  Every physical limit those bounds could express is already
a unit-free limit on what is sampled (q_source in [0, 2]; log_f_total uniform
in log), so they were removed.  These tests pin the consequence: the flux
SCALE of a file does not enter the posterior.
"""

import os

import numpy as np
import pytensor
import pytest
import yaml
from mulens_acceptance import decompose, raw_start

from exozippy.system import System

K = 2.5 / np.log(10.0)  # d mag / d ln flux
T0 = 2460025.0
U0 = 0.30
TE = 25.0
SCALE = 1.0e10

DEFAULTS = os.path.join(
    os.path.dirname(__file__),
    "..",
    "src",
    "exozippy",
    "components",
    "mulensing",
    "defaults.yaml",
)


def _pspl_flux(t, f_source, f_blend):
    tau = (t - T0) / TE
    u = np.sqrt(tau**2 + U0**2)
    amp = (u**2 + 2.0) / (u * np.sqrt(u**2 + 4.0))
    return f_source * amp + f_blend


def _config(fmt):
    return {
        "star": [{"name": "Lens"}, {"name": "Source"}],
        "mulensevent": [{"finite_source": False, "t0_par": T0}],
        "lens": [{"body": "star.Lens"}],
        "source": [{"body": "star.Source"}],
        "mulensinstrument": [
            {"name": "LC", "file": "lc.dat", "data_format": fmt}
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


def _build(tmpdir, rows, fmt):
    np.savetxt(os.path.join(str(tmpdir), "lc.dat"), rows, fmt="%.17g")
    cwd = os.getcwd()
    os.chdir(str(tmpdir))
    try:
        system = System(_config(fmt), user_params=_params())
        system.prepare()
        model = system.build_model()
    finally:
        os.chdir(cwd)
    start = raw_start(system, model)
    parts, total, reconciles, _ = decompose(system, model, start)
    assert reconciles, "the logp decomposition no longer reconciles"
    return system, model, start, parts


def _value_at(model, node, start):
    """A node's value at the raw start (never Parameter.value read bare,
    which draws from the prior)."""
    fn = pytensor.function(
        model.value_vars,
        model.replace_rvs_by_values([node]),
        on_unused_input="ignore",
    )
    out = fn(*[np.asarray(start[v.name]) for v in model.value_vars])[0]
    return np.atleast_1d(np.asarray(out, dtype=float))


def _flux_bound_terms(parts):
    return sorted(
        k
        for k in parts
        if ("bound" in k)
        and (
            "mulensinstrument.f_source" in k or "mulensinstrument.f_blend" in k
        )
    )


def test_defaults_give_f_source_and_f_blend_no_bounds():
    """
    Given the mulensing defaults.yaml,
    When the f_source and f_blend entries are read,
    Then neither carries a lower or upper bound (they are derived, so a bound
    would be a barrier in the file's arbitrary flux units), while the
    sampled q_source and log_f_total -- where the unit-free limits live --
    keep theirs.
    """
    with open(DEFAULTS) as fh:
        params = yaml.safe_load(fh)["mulensinstrument"]

    for name in ("f_source", "f_blend"):
        assert "lower" not in params[name], name
        assert "upper" not in params[name], name
    assert (params["q_source"]["lower"], params["q_source"]["upper"]) == (
        0.0,
        2.0,
    )
    assert (
        params["log_f_total"]["lower"],
        params["log_f_total"]["upper"],
    ) == (-100, 10.0)


@pytest.fixture(scope="module")
def mag_and_scaled_flux(tmp_path_factory):
    """One light curve, written as magnitudes and as fluxes x 1e10."""
    rng = np.random.default_rng(16180)
    t = np.linspace(T0 - 90.0, T0 + 90.0, 400)
    f_true = _pspl_flux(t, f_source=8.0e-8, f_blend=2.0e-8)
    sigma_m = np.full(t.size, 0.01)
    flux = f_true * (1.0 + rng.normal(0.0, sigma_m / K))
    mag = -2.5 * np.log10(flux)
    # The flux twin is the magnitude file's OWN conversion (what load_data
    # does to it) times SCALE, so the two hold the same data to round-off.
    flux_m = 10.0 ** (-0.4 * mag)
    err_m = flux_m * sigma_m / K
    mag_rows = np.column_stack([t, mag, sigma_m])
    flux_rows = np.column_stack([t, SCALE * flux_m, SCALE * err_m])

    mag_build = _build(tmp_path_factory.mktemp("mag"), mag_rows, "magnitude")
    flux_build = _build(tmp_path_factory.mktemp("flux"), flux_rows, "flux")
    return mag_build, flux_build, t.size


@pytest.mark.slow
def test_no_flux_barrier_exists(mag_and_scaled_flux):
    """
    Given the magnitude build and the x 1e10 flux build,
    When their logp terms are listed,
    Then neither has a low_bound/up_bound potential on f_source or f_blend.
    """
    (_, _, _, mag_parts), (_, _, _, flux_parts), _ = mag_and_scaled_flux
    assert _flux_bound_terms(mag_parts) == []
    assert _flux_bound_terms(flux_parts) == []


@pytest.mark.slow
def test_scaled_flux_twin_gives_the_same_logp_up_to_the_jacobian(
    mag_and_scaled_flux,
):
    """
    Given the same light curve read as magnitudes and as fluxes x 1e10,
    When each model is decomposed at its own start,
    Then the starts agree (log_f_total shifted by exactly 10), every term
    is identical except two, and those two differ by exactly what a change
    of flux units must cost:
      * the data term by the constant Jacobian -N ln(1e10);
      * log_f_total's logit-uniform term by log(q(1-q)) at the shifted
        position -- the prior on log_f_total is uniform in both, so this is
        the raw coordinate's Jacobian, not a change of prior.
    """
    # ARRANGE
    (
        (mag_sys, mag_model, mag_start, mag_parts),
        (
            flux_sys,
            flux_model,
            flux_start,
            flux_parts,
        ),
        n,
    ) = mag_and_scaled_flux

    # ACT
    lf_mag = _value_at(
        mag_model, mag_sys.mulensinstrument.log_f_total.value, mag_start
    )
    lf_flux = _value_at(
        flux_model, flux_sys.mulensinstrument.log_f_total.value, flux_start
    )
    q_mag = _value_at(
        mag_model, mag_sys.mulensinstrument.q_source.value, mag_start
    )
    q_flux = _value_at(
        flux_model, flux_sys.mulensinstrument.q_source.value, flux_start
    )

    # ASSERT -- the same physical start, in two unit systems.
    np.testing.assert_allclose(lf_flux - lf_mag, 10.0, rtol=0, atol=1e-9)
    np.testing.assert_allclose(q_flux, q_mag, rtol=1e-9)

    assert set(mag_parts) == set(flux_parts)
    data = "RV:mulensinstrument.model"
    jac = "POT:logit_uniform_prior.mulensinstrument.log_f_total"
    assert data in mag_parts and jac in mag_parts

    np.testing.assert_allclose(
        flux_parts[data] - mag_parts[data],
        -n * np.log(SCALE),
        rtol=1e-9,
    )

    def log_jac(v):
        s = (v - (-100.0)) / 110.0
        return float(np.sum(np.log(s * (1.0 - s))))

    np.testing.assert_allclose(
        flux_parts[jac] - mag_parts[jac],
        log_jac(lf_flux) - log_jac(lf_mag),
        rtol=1e-9,
        atol=1e-12,
    )

    for term in sorted(set(mag_parts) - {data, jac}):
        assert flux_parts[term] == pytest.approx(
            mag_parts[term], rel=1e-9, abs=1e-12
        ), term


@pytest.mark.slow
def test_counts_scale_light_curve_pays_no_barrier(tmp_path):
    """
    Given a difference-imaging-like flux file with a ~6000-count baseline
    (the scale of a real MOA file whose start used to pay ~-1.1e4 nats to
    f_blend's [-1000, 1000] cap),
    When the model is built and decomposed at its start,
    Then no f_source/f_blend barrier exists, and no light-curve bound
    potential the model does carry pays anything there.
    """
    # ARRANGE
    rng = np.random.default_rng(6100)
    t = np.linspace(T0 - 90.0, T0 + 90.0, 300)
    f_true = _pspl_flux(t, f_source=1500.0, f_blend=4600.0)
    err = np.full(t.size, 30.0)
    rows = np.column_stack([t, f_true + rng.normal(0.0, err), err])

    # ACT
    system, model, start, parts = _build(tmp_path, rows, "flux")
    f_total = 10.0 ** _value_at(
        model, system.mulensinstrument.log_f_total.value, start
    )

    # ASSERT -- the start really is at the counts scale, and nothing there
    # pays for it.
    assert 5000.0 < float(f_total[0]) < 7000.0
    assert _flux_bound_terms(parts) == []
    # Only the light curve's own (flux-unit) barriers are this test's
    # business: this toy event's derived t_E/mu_rel barriers pay at the
    # start too, on master and here alike, through no flux scale.
    paying = {
        k: v
        for k, v in parts.items()
        if "bound" in k and "mulensinstrument" in k and v < -1e-6
    }
    assert not paying, paying
