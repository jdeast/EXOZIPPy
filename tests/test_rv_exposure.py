"""Exposure averaging of corrected radial velocities (rvinstrument.md,
"Corrected velocities are exposure-averaged too"; instrument.md, "Exposure
smearing").

A corrected RV file that states an exposure is modeled as ``gamma +
<reflex>``: the star's velocity averaged over each exposure on the base's
node table (Gauss-Legendre nodes for a uniform exposure).  Pinned here:

  - (c) a circular orbit (a pure sinusoid) with exptime = P/4 is attenuated
    by the analytic ``sinc(pi T / P)`` at every epoch, whether the exposure
    comes from the per-file ``exptime_s:`` key or the per-row ``exptime_s``
    column;
  - the plotted datum is the instantaneous ``gamma + reflex(tag)`` plus the
    likelihood's own residual from the exposure mean (the builder
    contract);
  - ``ninterp: 1`` is the instantaneous model exactly;
  - the retired ``exptime`` column role raises, and a component that reads
    no per-row description refuses ``exposure_meter:``.
"""

import os

import numpy as np
import pytensor
import pytest

from exozippy.system import System

P_D = 2.0
EXPTIME_S = P_D / 4.0 * 86400.0
TC = 2459000.3


def _write(tmpdir, with_column):
    t = np.linspace(TC - 10.0, TC + 10.0, 37)
    cols = [t, np.zeros_like(t), np.full_like(t, 5.0)]
    if with_column:
        cols.append(np.full_like(t, EXPTIME_S))
    np.savetxt(
        os.path.join(str(tmpdir), "rv.dat"), np.column_stack(cols), fmt="%.17g"
    )


def _system(tmpdir, entry_extra):
    _write(tmpdir, "columns" in entry_extra)
    config = {
        "star": [{"name": "A", "mist": False}],
        "planet": [{"name": "b"}],
        "orbit": [{"name": "b"}],
        "rvinstrument": [
            # The Roemer delay makes the observed reflex not quite a
            # sinusoid (~a/c over P); the sinc identity is for the
            # sinusoid, so the light-travel time is off here.
            {
                "name": "Spec",
                "file": "rv.dat",
                "light_travel_time": False,
                **entry_extra,
            },
        ],
    }
    params = {
        "star.A.mass": {"initval": 1.0, "sigma": 0.05},
        "star.A.radius": {"initval": 1.0, "sigma": 0.05},
        "planet.b.mass": {"initval": 5.0},
        "orbit.b.period": {"initval": P_D},
        "orbit.b.tc": {"initval": TC},
        "orbit.b.secosw": {"initval": 0.0, "sigma": 0},
        "orbit.b.sesinw": {"initval": 0.0, "sigma": 0},
    }
    cwd = os.getcwd()
    os.chdir(str(tmpdir))
    try:
        system = System(config, user_params=params)
        system.prepare()
        model = system.build_model()
    finally:
        os.chdir(cwd)
    return system, model


def _at_vals(system, nodes, vals):
    """Likelihood nodes with ``system.plot_params`` set to ``vals`` (the
    inputs the compiled plot functions take)."""
    fn = pytensor.function(
        [p.value for p in system.plot_params],
        list(nodes),
        on_unused_input="ignore",
    )
    return [np.asarray(x, dtype=float) for x in fn(*vals)]


KEY = {"exptime_s": EXPTIME_S, "ninterp": 5}
COLUMN = {
    "columns": {"time": 0, "rv": 1, "err": 2, "exptime_s": 3},
    "ninterp": 5,
}


@pytest.mark.parametrize("entry", [KEY, COLUMN], ids=["key", "column"])
def test_a_quarter_period_exposure_attenuates_a_sinusoid_by_sinc(
    tmp_path, entry
):
    """
    Given a circular orbit (P = 2 d) and 12-hour exposures (P/4), stated
    by the per-file key or the per-row column, on 5 Gauss nodes,
    When the likelihood's model is evaluated at the start,
    Then its reflex (model - gamma) is the instantaneous reflex at each tag
    times sinc(pi/4) = 0.9003, to 1e-9 of K.
    """
    system, _ = _system(tmp_path, entry)
    rv = system.rvinstrument
    vals = rv._point_to_plot_params({}, system)
    model, bary, matrix = _at_vals(
        system,
        [
            rv._rv_model_data_node,
            rv._rv_bary_data_node,
            rv._rv_matrix_data_node,
        ],
        vals,
    )
    inst = matrix.sum(axis=1)
    smeared = inst + bary
    x = np.pi * EXPTIME_S / 86400.0 / P_D
    K = np.max(np.abs(inst))
    np.testing.assert_allclose(
        smeared, np.sin(x) / x * inst, rtol=0, atol=1e-9 * K
    )
    assert np.max(np.abs(bary)) > 0.05 * K


def test_the_plotted_datum_carries_the_likelihood_residual(tmp_path):
    """
    Given an exposure-averaged corrected file,
    When the plots marshal its data (_plotted_rv) at the start,
    Then each plotted datum minus the plotted curve at its tag (gamma +
    the instantaneous reflex) equals the likelihood's residual, data minus
    the exposure-mean model.
    """
    system, _ = _system(tmp_path, KEY)
    rv = system.rvinstrument
    vals = rv._point_to_plot_params({}, system)
    model, matrix = _at_vals(
        system, [rv._rv_model_data_node, rv._rv_matrix_data_node], vals
    )
    gamma = model - (
        matrix.sum(axis=1) + _at_vals(system, [rv._rv_bary_data_node], vals)[0]
    )
    plotted = rv._plotted_rv(system, {})
    np.testing.assert_allclose(
        plotted - (gamma + matrix.sum(axis=1)),
        rv.rv - model,
        rtol=0,
        atol=1e-12,
    )


def test_ninterp_one_is_the_instantaneous_model(tmp_path):
    """
    Given exptime_s stated with ninterp 1, and no exposure keys at all,
    When the start logps are compared,
    Then they are identical (one node is the tag itself).
    """
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    _, ma = _system(a, {"exptime_s": EXPTIME_S, "ninterp": 1})
    _, mb = _system(b, {})
    assert ma.compile_logp()(
        ma.initial_point(random_seed=0)
    ) == mb.compile_logp()(mb.initial_point(random_seed=0))


def test_the_retired_exptime_column_role_raises():
    """
    Given a columns: spec naming the old 'exptime' role,
    When the system is constructed,
    Then it raises naming the replacement 'exptime_s'.
    """
    config = {
        "star": [{"name": "A", "mist": False}],
        "rvinstrument": [
            {
                "name": "Spec",
                "file": "unused.rv",
                "columns": {"time": 0, "rv": 1, "err": 2, "exptime": 3},
            }
        ],
    }
    with pytest.raises(ValueError, match=r"'exptime' role is now 'exptime_s'"):
        System(config, user_params={})


def test_a_child_without_per_row_exposures_refuses_the_meter():
    """
    Given a transit file with exposure_meter: (a per-row description only
    rvinstrument reads),
    When the system is constructed,
    Then it raises naming the component and pointing at exptime_s:.
    """
    config = {
        "star": [{"name": "A", "mist": False}],
        "planet": [{"name": "b"}],
        "orbit": [{"name": "b"}],
        "band": [{"name": "V", "filter": "V", "ld_law": "quadratic"}],
        "transit": [
            {
                "name": "LC",
                "file": "unused.dat",
                "band": "V",
                "exposure_meter": "m.txt",
            }
        ],
    }
    with pytest.raises(NotImplementedError, match=r"transit.*exposure_meter"):
        System(config, user_params={})
