"""
Integration tests for the light-travel-time (Roemer delay) wiring in
transit.py's build_likelihood (components/ltt.py is unit-tested on its own
in tests/test_ltt.py; these tests exercise the WIRED path -- real
orbits.a/m_primary/m_total accessors, real units -- rather than calling
ltt.py's functions directly with hand-typed values).
"""

import json
from pathlib import Path
from unittest import mock

import numpy as np
import pymc as pm
import pytensor
import pytest
from exoplanet_core.pymc.ops import Kepler

from exozippy.components import ltt
from exozippy.constants import C_LIGHT_RSUN_PER_DAY
from exozippy.system import System

_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

_TC = 2459000.0
_PERIOD = 365.25  # days; with star.mass ~ 1 Msun this gives a ~ 1 AU


def _write_two_row_lc(path):
    # Only the TIMES matter for the checks below (mu is independent of the
    # flux/err columns); one row at primary conjunction (tc, by definition
    # of the orbit's tc), one at secondary conjunction (tc + P/2, exact for
    # a circular orbit).
    time = np.array([_TC, _TC + _PERIOD / 2.0])
    flux = np.array([1.0, 1.0])
    err = np.array([1.0e-3, 1.0e-3])
    np.savetxt(path, np.column_stack([time, flux, err]))
    return str(path)


def _ltt_wiring_config(lc_file, light_travel_time=None):
    entry = {"name": "inst0", "file": lc_file, "band": "TESS"}
    if light_travel_time is not None:
        entry["light_travel_time"] = light_travel_time
    return {
        "star": [{"name": "A", "mist": False}],
        "planet": [{"name": "b"}],
        "orbit": [{"name": "b"}],
        "band": [{"name": "TESS", "filter": "TESS", "ld_law": "quadratic"}],
        "transit": [entry],
    }


def _ltt_wiring_params():
    return {
        "star.0.radius": {
            "initval": 2.0,
            "sigma": 0.05,
        },  # deliberately != 1 Rsun: catches an a/R* (dimensionless)
        # vs physical a [Rsun] mixup, which R* = 1 would hide numerically.
        "star.0.mass": {"initval": 1.0, "sigma": 0.05},
        "star.0.teff": {"initval": 5800, "sigma": 100},
        "star.0.feh": {"initval": 0.0, "sigma": 0.08},
        "orbit.0.period": {"initval": _PERIOD},
        "orbit.0.tc": {"initval": _TC},
        "orbit.0.cosi": {"initval": 0.0, "sigma": 0.0},  # edge-on, sin_i = 1
        "orbit.0.secosw": {"initval": 0.0, "sigma": 0.0},  # circular, omega=0
        "orbit.0.sesinw": {"initval": 0.0, "sigma": 0.0},
        "planet.0.radius": {"initval": 1.0},
        # log_q pinned at its floor: mass ratio 1e-9, so the barycentric
        # factor m_primary/m_total is 1 to 9 significant figures -- doesn't
        # perturb the a/c comparison below.
        "planet.0.log_q": {"initval": -9.0, "sigma": 0.0},
    }


def _eval_at_point(node, model, point):
    fn = pytensor.function(
        [],
        node,
        givens=[
            (v, np.asarray(point[v.name]))
            for v in model.free_RVs
            if v.name in point
        ],
        on_unused_input="ignore",
        mode="FAST_COMPILE",
    )
    return fn()


def _count_kepler_solves(node):
    """Number of DISTINCT Kepler-solve Apply nodes among node's ancestors
    (sinf/cosf share one Apply node, so dedupe on the node, not the op)."""
    seen = set()
    for anc in pytensor.graph.ancestors([node]):
        if anc.owner is not None and isinstance(anc.owner.op, Kepler):
            seen.add(anc.owner)
    return len(seen)


def test_wired_ltt_delay_matches_a_over_c_through_real_accessors(tmp_path):
    """
    Given a transit-only system (star mass ~1 Msun, star RADIUS deliberately
    2 Rsun -- not 1, so a physical a [Rsun] and a dimensionless a/R* cannot
    be numerically confused --, period 365.25 d giving a ~ 1 AU via Kepler's
    third law, edge-on (cosi=0) circular (e=0) orbit, negligible planet
    mass) built through the real System/orbit/planet pipeline (NOT hand-
    typed a_rel/factor values),
    When build_likelihood's actual wiring (transit.py's call into
    ltt.retarded_time, using orbits.a/orbits.m_primary/orbits.m_total) is
    exercised by building the model, with a spy capturing the real delay
    tensor it produced,
    Then the delay at primary and secondary conjunction matches
    factor*a_rel/c (the SAME orbits.a/m_primary/m_total nodes the wiring
    itself used, read back independently) to 1e-6 relative -- confirming
    the units/accessor contract end to end -- and separately is close to
    the known ~499 s/AU light time in absolute terms, guarding against a
    shared-formula bug that an internal-consistency check alone would miss.
    Also confirms retarded_time is invoked exactly twice per build_model():
    once from build_likelihood's group loop (one oversample group here,
    ninterp=1) and once from compile_plotters -- both paths are wired, live,
    not a dead branch.
    """
    lc = _write_two_row_lc(tmp_path / "lc.dat")
    real_retarded_time = ltt.retarded_time
    calls = []

    def _spy(*args, **kwargs):
        result = real_retarded_time(*args, **kwargs)
        calls.append((args[0], result[1]))  # (t_grid, delay)
        return result

    with mock.patch(
        "exozippy.components.transit.transit.ltt.retarded_time",
        side_effect=_spy,
    ) as spy:
        system = System(
            _ltt_wiring_config(lc), user_params=_ltt_wiring_params()
        )
        system.prepare()
        model = system.build_model()
        # One from build_likelihood's group loop (3-D t_grid, has the
        # sub-exposure axis), one from compile_plotters (2-D, no
        # sub-exposure axis -- smearing is applied outside by
        # _smeared_full_lc instead).
        assert spy.call_count == 2

    with model:
        point = system.get_internal_point(model, system.get_raw_start(model))

    build_likelihood_calls = [
        delay for t_grid, delay in calls if t_grid.ndim == 3
    ]
    assert len(build_likelihood_calls) == 1
    delay = _eval_at_point(build_likelihood_calls[0], model, point)
    delay_primary = float(delay[0, 0, 0])
    delay_secondary = float(delay[1, 0, 0])

    orbit_map = system.planet.orbit_map
    a_rel_val = float(
        _eval_at_point(system.orbit.a.value[orbit_map][0], model, point)
    )
    m_primary_val = float(
        _eval_at_point(
            system.orbit.m_primary.value[orbit_map][0], model, point
        )
    )
    m_total_val = float(
        _eval_at_point(system.orbit.m_total.value[orbit_map][0], model, point)
    )
    factor_val = m_primary_val / m_total_val

    expected_amp_days = factor_val * a_rel_val / C_LIGHT_RSUN_PER_DAY

    assert abs(delay_primary) == pytest.approx(expected_amp_days, rel=1e-6)
    assert abs(delay_primary) * 86400.0 == pytest.approx(499.0, abs=2.0)
    assert (delay_secondary - delay_primary) == pytest.approx(
        2.0 * expected_amp_days, rel=1e-6
    )


def test_ltt_off_matches_pre_ltt_structure(tmp_path):
    """
    Given the same system, but with light_travel_time: False set on the
    transit file (the OFF branch, reached the real way now -- a per-file
    config key, not a monkeypatched flag),
    When the model is built,
    Then ltt.retarded_time is never called and the transit flux node's
    graph contains exactly ONE Kepler-solve Apply node -- i.e. the OFF
    branch is algebraically the pre-LTT computation (M = (t_grid - tp) * n
    fed straight to the group loop's own, sole, Kepler solve), not a
    second solve that happens to cancel out. The same system built with
    LTT ON (the default -- light_travel_time left unset) is confirmed to
    contain exactly TWO (the delay solve plus the corrected-time geometry
    solve), so this is a real structural difference, not a vacuous count.
    """
    lc = _write_two_row_lc(tmp_path / "lc.dat")

    with mock.patch(
        "exozippy.components.transit.transit.ltt.retarded_time"
    ) as off_spy:
        system_off = System(
            _ltt_wiring_config(lc, light_travel_time=False),
            user_params=_ltt_wiring_params(),
        )
        system_off.prepare()
        system_off.build_model()
        off_spy.assert_not_called()

    assert _count_kepler_solves(system_off.transit._model_flux_node) == 1

    system_on = System(
        _ltt_wiring_config(lc), user_params=_ltt_wiring_params()
    )
    system_on.prepare()
    system_on.build_model()
    assert _count_kepler_solves(system_on.transit._model_flux_node) == 2


def test_ltt_off_build_likelihood_reproduces_pre_ltt_model_values(tmp_path):
    """
    Given a STORED golden fixture (tests/fixtures/transit_ltt_off_reference.
    json) of transit._model_flux_node's output from the actual pre-LTT
    Transit class -- generated once from git commit 3ae5466 (the parent of
    df96bd9, the transit light-travel-time wiring commit; confirmed via grep
    at generation time to have no ltt/LIGHT_TRAVEL references), not a hand
    reimplementation of the (fairly involved) group loop that could itself
    drift from what the real old code did,
    When the SAME config/params (_ltt_wiring_config/_ltt_wiring_params, with
    light_travel_time: False set on the transit file -- the real, per-file
    config-key OFF path, not a monkeypatched flag) build a model with the
    CURRENT Transit,
    Then transit._model_flux_node evaluates to the SAME array the fixture
    recorded, to floating-point precision -- the structural "one Kepler
    solve, ltt never called" check in test_ltt_off_matches_pre_ltt_structure
    confirms the graph SHAPE didn't grow; this confirms the VALUES didn't
    move either, catching a stray bug in surrounding code (e.g. an
    accidental reorder of sin_i/cos_i, a mis-sliced index) that a shape-only
    check would miss. This is the backward-compat guard: with LTT off, the
    model is byte-for-byte the pre-LTT one, config key or not.

    Deliberately does NOT read git history (unlike an earlier version of
    this test, which shelled out to `git show HEAD:...` as its oracle): that
    self-invalidates the moment the LTT commit lands on HEAD, since HEAD
    then stops being the pre-LTT answer -- exactly what happened here. A
    frozen fixture has no such expiry.
    """
    lc = _write_two_row_lc(tmp_path / "lc.dat")
    config = _ltt_wiring_config(lc, light_travel_time=False)
    params = _ltt_wiring_params()

    with open(_FIXTURES_DIR / "transit_ltt_off_reference.json") as f:
        fixture = json.load(f)
    mu_reference = np.asarray(fixture["model_flux"])

    system_off = System(config, user_params=params)
    system_off.prepare()
    model_off = system_off.build_model()
    with model_off:
        point_off = system_off.get_internal_point(
            model_off, system_off.get_raw_start(model_off)
        )
    mu_off = _eval_at_point(
        system_off.transit._model_flux_node, model_off, point_off
    )

    np.testing.assert_array_equal(mu_reference, mu_off)


def _mixed_group_config(lc0, lc1):
    """Two transit files sharing a band/orbit/planet, both default ninterp=1
    (so _oversample_groups puts their rows in the SAME group) -- inst0 left
    at the light_travel_time default (True), inst1 explicitly False. This is
    the genuinely-mixed case the group loop's pt.where branch exists for.
    """
    return {
        "star": [{"name": "A", "mist": False}],
        "planet": [{"name": "b"}],
        "orbit": [{"name": "b"}],
        "band": [{"name": "TESS", "filter": "TESS", "ld_law": "quadratic"}],
        "transit": [
            {"name": "inst0", "file": lc0, "band": "TESS"},
            {
                "name": "inst1",
                "file": lc1,
                "band": "TESS",
                "light_travel_time": False,
            },
        ],
    }


def test_mixed_group_ltt_gradient_is_finite(tmp_path):
    """
    Given two transit files in the SAME oversample group (both ninterp=1,
    the default) but DIFFERENT light_travel_time settings -- inst0 on,
    inst1 off -- so the group loop must take its pt.where(lt_mask, ...)
    branch (test_ltt_off_matches_pre_ltt_structure's all-on/all-off cases
    never exercise this line at all),
    When the model's logp and its gradient are evaluated at the initial
    point, and a short NUTS chain is sampled with nuts_sampler="numpyro",
    Then both are finite throughout.

    This is a real check, not an assumption: the where-trap risk this
    pt.where COULD carry is that pytensor differentiates BOTH branches even
    where only one is selected, so if either branch had a singularity (like
    solve_delay's az=0 quadratic branch, which needed az_safe specifically
    because of this), the discarded branch's bad gradient can poison the
    selected one. Here both branches are just t_grid_corrected and t_grid --
    ordinary, everywhere-finite time arrays with no division, sqrt, or log
    anywhere near them -- so there is no candidate singularity for either
    branch to carry. This test confirms that holds in practice rather than
    asserting it from the formula alone.
    """
    lc0 = _write_two_row_lc(tmp_path / "lc0.dat")
    lc1 = _write_two_row_lc(tmp_path / "lc1.dat")
    config = _mixed_group_config(lc0, lc1)
    params = _ltt_wiring_params()

    with mock.patch(
        "exozippy.components.transit.transit.ltt.retarded_time",
        side_effect=ltt.retarded_time,
    ) as spy:
        system = System(config, user_params=params)
        system.prepare()
        model = system.build_model()
        # One call for the (single, shared) oversample group in
        # build_likelihood, one for compile_plotters -- confirms the mixed
        # group actually took the "any() but not all()" branch and called
        # ltt.retarded_time at all (a bug that skipped the correction
        # entirely for a mixed group would also show 0 calls here).
        assert spy.call_count == 2

    assert _count_kepler_solves(system.transit._model_flux_node) == 2

    with model:
        point = model.initial_point()
        logp_val = model.compile_logp()(point)
        grad_val = model.compile_dlogp()(point)

    assert np.isfinite(logp_val), f"non-finite logp: {logp_val}"
    assert np.all(np.isfinite(grad_val)), (
        f"non-finite gradient in mixed-group pt.where: {grad_val}"
    )

    with model:
        idata = pm.sample(
            draws=25,
            tune=25,
            chains=1,
            cores=1,
            nuts_sampler="numpyro",
            progressbar=False,
            random_seed=0,
        )

    for var in idata.posterior.data_vars:
        assert np.all(np.isfinite(idata.posterior[var].values)), (
            f"non-finite draws for {var}"
        )
