"""
The transit baseline is a multiplicative flux NORMALIZATION, not an offset.

The light-curve model is ``flux = baseline * rel``, with ``rel`` the
unit-normalized astrophysical model (1 out of transit; the planet blocks a
FRACTION of the star's light), exactly EXOFASTv2's ``modelflux *= f0``
(exofast_chi2v2.pro) and ``f0*((model-1)*(1-dilute)+1)``
(exofast_tran.pro).  The model used to be ``baseline - dip``, which is
right only for a baseline of exactly 1: on HAT-3b (f0 = 1.0063) it biased
p high by 0.00035, and for a light curve in raw counts it is grossly
wrong (a 1% dip on 1e4 counts modelled as 0.01 counts).

Every check feeds ONE built system different baseline / dilution values
through its own compiled likelihood node, so nothing but the baseline and
the dilution changes between the evaluations.
"""

import numpy as np
import pytensor
import pytest

from exozippy.system import System

_T0 = 2459634.3
_PERIOD = 2.99
_COUNTS = 1.0e4


def _write_lc(path, n=160):
    """A light curve in raw COUNTS (~1e4), so the seeded baseline is far
    from 1 and the additive form would be visibly wrong."""
    rng = np.random.default_rng(7)
    t = np.linspace(_T0 - 0.2, _T0 + 0.2, n)
    flux = _COUNTS * (1.0 + rng.normal(0.0, 1e-3, n))
    np.savetxt(path, np.column_stack([t, flux, np.full(n, 1e-3 * _COUNTS)]))
    return str(path)


@pytest.fixture(scope="module")
def counts_system(tmp_path_factory):
    d = tmp_path_factory.mktemp("mult_baseline")
    config = {
        "star": [{"name": "A", "mist": False}],
        "planet": [{"name": "b"}],
        "orbit": [{"name": "b"}],
        "band": [{"name": "TESS", "filter": "TESS", "ld_law": "quadratic"}],
        "transit": [
            {
                "name": "inst0",
                "file": _write_lc(d / "lc.dat"),
                "band": "TESS",
                "fitdilute": True,
            }
        ],
    }
    params = {
        "star.0.radius": {"initval": 1.0, "sigma": 0.05},
        "star.0.mass": {"initval": 1.0, "sigma": 0.05},
        "star.0.teff": {"initval": 5800, "sigma": 100},
        "star.0.feh": {"initval": 0.0, "sigma": 0.08},
        "orbit.0.period": {"initval": _PERIOD},
        "orbit.0.tc": {"initval": _T0},
        "orbit.0.cosi": {"initval": 0.05},
        "planet.0.radius": {"initval": 1.7},
        "transit.0.dilute": {"initval": 0.0, "mu": 0.0, "sigma": 0.5},
    }
    system = System(config, user_params=params)
    system.prepare()
    model = system.build_model()
    with model:
        point = system.get_internal_point(model, system.get_raw_start(model))
    system.transit.compile_plotters(model, system)
    return system, model, point


def _inputs(system, point, baseline=None, dilute=None):
    """The plot-parameter values at ``point``, with the transit baseline
    and/or dilution replaced."""
    tr = system.transit
    values = list(tr._point_to_plot_params(point, system))
    labels = [p.label for p in system.plot_params]
    if baseline is not None:
        values[labels.index(tr.baseline.label)] = np.atleast_1d(
            np.asarray(baseline, dtype=float)
        )
    if dilute is not None:
        values[labels.index(tr.dilute.label)] = np.atleast_1d(
            np.asarray(dilute, dtype=float)
        )
    return values


def _mu(system, model, values):
    """The mu the likelihood scores (read off the observed node itself),
    at ``values``."""
    node = model.named_vars["transit_likelihood"].owner.inputs[2]
    fn = pytensor.function(
        [p.value for p in system.plot_params], node, on_unused_input="ignore"
    )
    return np.asarray(fn(*values))


def test_the_seeded_baseline_is_in_counts(counts_system):
    """
    Given a light curve in raw counts,
    When the model is built,
    Then the baseline starts at the data's ~1e4 level (the premise of the
    other tests: the additive form would be off by a factor ~1e4 there).
    """
    system, _, point = counts_system
    b = system.transit._point_value(point, system.transit.baseline, 0)
    assert b == pytest.approx(_COUNTS, rel=0.01)


@pytest.mark.parametrize("baseline", [2.0, _COUNTS])
def test_model_is_baseline_times_the_relative_model(counts_system, baseline):
    """
    Given the same parameters with the baseline at 1 and at B (2 or 1e4),
    When the likelihood's mu is evaluated,
    Then mu(B) == B * mu(1) to rounding: the transit depth is a FRACTION
    of the baseline, flux = B * (1 - dip).  The additive form
    (B - dip) would put a 1e4-count light curve's dip at ~0.01 count.
    """
    system, model, point = counts_system
    unit = _mu(system, model, _inputs(system, point, baseline=1.0, dilute=0))
    scaled = _mu(
        system, model, _inputs(system, point, baseline=baseline, dilute=0)
    )
    dip = 1.0 - unit
    assert dip.max() > 1e-3  # there is a transit
    assert dip.min() == pytest.approx(0.0, abs=1e-12)  # and out-of-transit
    np.testing.assert_allclose(scaled, baseline * (1.0 - dip), rtol=1e-13)
    # The depth in the data's units scales with the baseline: not B - dip.
    assert baseline - scaled.min() == pytest.approx(
        baseline * dip.max(), rel=1e-10
    )


def test_dilution_matches_exofastv2(counts_system):
    """
    Given dilute = 0.3 and a baseline of 2,
    When mu is evaluated,
    Then it is EXOFASTv2's f0*((model-1)*(1-dilute)+1) with model the
    undiluted unit-normalized light curve (exofast_tran.pro:157), i.e.
    f0*(model*(1-dilute)+dilute).
    """
    system, model, point = counts_system
    undiluted = _mu(
        system, model, _inputs(system, point, baseline=1.0, dilute=0.0)
    )
    f0, d = 2.0, 0.3
    mu = _mu(system, model, _inputs(system, point, baseline=f0, dilute=d))
    np.testing.assert_allclose(
        mu, f0 * (undiluted * (1.0 - d) + d), rtol=1e-13
    )
    np.testing.assert_allclose(
        mu, f0 * ((undiluted - 1.0) * (1.0 - d) + 1.0), rtol=1e-13
    )


def test_planet_terms_decompose_the_model_in_data_units(counts_system):
    """
    Given the counts light curve at its own start (baseline ~1e4),
    When the per-planet terms the plots use are read at the data,
    Then baseline + sum(terms) == the model, and each term is the
    baseline times the planet's change to the relative model.
    """
    system, model, point = counts_system
    tr = system.transit
    values = _inputs(system, point)
    phys, terms = tr._eval_lc_data(values)
    b = tr._point_value(point, tr.baseline, 0)
    np.testing.assert_allclose(b + terms.sum(axis=1), phys, rtol=1e-14)
    np.testing.assert_allclose(phys, _mu(system, model, values), rtol=1e-14)
    unit_phys, unit_terms = tr._eval_lc_data(
        _inputs(system, point, baseline=1.0)
    )
    np.testing.assert_allclose(terms, b * unit_terms, rtol=1e-12, atol=0)


def test_phased_panel_is_in_relative_flux(counts_system):
    """
    Given the counts light curve,
    When the phased panel's arrays are built,
    Then data, error bars and model are all divided by the baseline: the
    model is the planet's step of the UNIT-normalized model (a relative
    dip, not counts), the data are flux / baseline - 1, and every
    (data - model) / err is the likelihood's own normalized residual.
    """
    system, model, point = counts_system
    tr = system.transit
    b = tr._point_value(point, tr.baseline, 0)
    prep = tr._phased_lc_arrays(system, point, 0, 0)

    np.testing.assert_allclose(
        prep["y_data"], tr.flux / b - 1.0, rtol=0, atol=1e-15
    )
    np.testing.assert_allclose(prep["yerr_data"], tr.err / b, rtol=1e-14)
    assert -0.05 < prep["y_model"].min() < -1e-3  # a relative depth

    # The model at the data's own times, through the same normalization,
    # reproduces the likelihood's residuals exactly.
    values = _inputs(system, point)
    _, terms = tr._eval_lc_data(values)
    mu = _mu(system, model, values)
    np.testing.assert_allclose(
        (prep["y_data"] - terms[:, 0] / b) / prep["yerr_data"],
        (tr.flux - mu) / tr.err,
        rtol=1e-9,
        atol=1e-9,
    )
