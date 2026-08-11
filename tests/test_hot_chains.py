"""Hot-chain retention (ptde_async store_hot_chains) and posterior-
suppressed-mode discovery (outputs.ledger.discover_hot_modes)."""

import numpy as np
import pymc as pm
import pytest
import xarray as xr

from exozippy.components.parameter import Parameter
from exozippy.outputs.ledger import (
    build_seed_ledger,
    discover_hot_modes,
    ledger_to_text,
    match_ledger_to_modes,
)
from exozippy.samplers.ptde_async import ptde_async_sample


class _MinimalSystem:
    active_components = {}

    def get_raw_start(self, model):
        return model.initial_point()


def test_store_hot_chains_writes_the_posterior_hot_group():
    """
    Given ptde_async with store_hot_chains=5 and 3 temperatures,
    When sampling completes,
    Then the InferenceData carries a posterior_hot group: (n_temps-1) x
      n_chains hot chains, a per-chain temperature coordinate on the hot
      rungs' values, finite untempered lp, and ~draws/5 thinned draws.
    """
    # ARRANGE
    with pm.Model() as model:
        pm.Normal("x", mu=0.0, sigma=1.0)
        pm.Normal("y", mu=3.0, sigma=0.5)

    # ACT
    idata = ptde_async_sample(
        model,
        _MinimalSystem(),
        draws=60,
        tune=20,
        n_temps=3,
        T_max=8.0,
        n_chains=4,
        cores=1,
        seed=7,
        log_interval=10000,
        store_hot_chains=5,
    )

    # ASSERT
    assert hasattr(idata, "posterior_hot")
    hot = idata.posterior_hot
    assert hot.sizes["chain"] == (3 - 1) * 4
    temps = np.asarray(hot["temperature"].values)
    assert np.all(temps > 1.0)
    assert len(np.unique(temps)) == 2
    lp = np.asarray(hot["lp"].values)
    assert np.all(np.isfinite(lp))
    assert 1 <= hot.sizes["draw"] <= 60 // 5


def test_store_hot_chains_off_by_default():
    """
    Given ptde_async without store_hot_chains,
    When sampling completes,
    Then no posterior_hot group exists (byte-identical legacy output).
    """
    with pm.Model() as model:
        pm.Normal("x", mu=0.0, sigma=1.0)
    idata = ptde_async_sample(
        model,
        _MinimalSystem(),
        draws=30,
        tune=10,
        n_temps=2,
        T_max=4.0,
        n_chains=3,
        cores=1,
        seed=1,
        log_interval=10000,
    )
    assert not hasattr(idata, "posterior_hot")


def _two_basin_model():
    """Bounded parameter with basins at x=2 (deep) and x=7 (6 nats down)."""
    p = Parameter(label="toy.x", initval=2.0, lower=0.0, upper=10.0)
    with pm.Model() as model:
        xv = p.build_pymc()
        lp1 = -0.5 * ((xv - 2.0) / 0.05) ** 2
        lp2 = -0.5 * ((xv - 7.0) / 0.05) ** 2 - 6.0
        pm.Potential("like", pm.math.logsumexp(pm.math.stack([lp1, lp2])))
    return model, p


class _StubSystem:
    def __init__(self, params):
        self._params = params

    def get_all_parameters(self):
        return self._params


def _fake_hot_group(model, p, rng):
    """Synthetic posterior_hot: hot draws around BOTH basins with real
    (untempered) logp values, like a T ~ 6 rung would produce."""
    logp = model.compile_logp()
    raw2 = float(np.asarray(p.raw_from_initval(np.array([2.0])))[0])
    raw7 = float(np.asarray(p.raw_from_initval(np.array([7.0])))[0])
    # spread ~ sqrt(T) x basin sigma in raw units; basins are far apart
    draws = np.concatenate(
        [
            raw2 + 0.5 * rng.standard_normal(120),
            raw7 + 0.5 * rng.standard_normal(120),
        ]
    )
    lp = np.array([float(logp({"toy.x_raw": np.array([d])})) for d in draws])
    return xr.Dataset(
        {
            "toy.x_raw": (("chain", "draw"), draws.reshape(2, 120)),
            "lp": (("chain", "draw"), lp.reshape(2, 120)),
        },
        coords={
            "chain": [0, 1],
            "draw": np.arange(120),
            "temperature": ("chain", [6.0, 6.0]),
        },
    )


def test_discovery_finds_polishes_and_dedups_the_suppressed_basin():
    """
    Given a seed ledger holding only the deep basin and hot draws visiting
      both basins,
    When discover_hot_modes runs,
    Then exactly one new record appears (source 'hot-chain'), polished to
      the suppressed basin's optimum with delta_lp ~ 6, and a second
      discovery pass adds nothing (dedup).
    """
    # ARRANGE
    model, p = _two_basin_model()
    stub = _StubSystem([p])
    raw2 = np.asarray(p.raw_from_initval(np.array([2.0])))
    ledger0 = build_seed_ledger(stub, model, [{"toy.x_raw": raw2}], [0])
    hot = _fake_hot_group(model, p, np.random.default_rng(11))

    # ACT
    ledger = discover_hot_modes(stub, model, hot, ledger0, min_points=20)

    # ASSERT
    assert len(ledger) == 2
    rec = ledger[1]
    assert rec.source == "hot-chain"
    assert rec.phys["toy.x"][0] == pytest.approx(7.0, abs=0.05)
    assert rec.delta_lp == pytest.approx(5.73, abs=0.3)

    # ACT again: rediscovery must dedup
    ledger2 = discover_hot_modes(stub, model, hot, ledger, min_points=20)

    # ASSERT
    assert len(ledger2) == 2


def test_hot_record_reports_as_hot_chain_in_the_ledger_text():
    """
    Given a ledger extended by discovery,
    When it is matched against a report holding only the deep basin and
      rendered,
    Then the hot-chain record appears as REJECTED with the '(hot-chain)'
      source tag.
    """
    from test_mode_ledger import _fake_report

    model, p = _two_basin_model()
    stub = _StubSystem([p])
    raw2 = np.asarray(p.raw_from_initval(np.array([2.0])))
    ledger0 = build_seed_ledger(stub, model, [{"toy.x_raw": raw2}], [0])
    hot = _fake_hot_group(model, p, np.random.default_rng(3))
    ledger = discover_hot_modes(stub, model, hot, ledger0, min_points=20)
    match_ledger_to_modes(ledger, _fake_report([float(raw2[0])], "toy.x_raw"))

    text = ledger_to_text(ledger)
    assert "(hot-chain): REJECTED" in text


# ---------------------------------------------------------------------------
# Hot-chain search OUTCOME reporting: "never searched", "searched and found
# nothing", "the search crashed" and "found N" must be distinguishable in the
# report a user reads, not only in a log file.
# ---------------------------------------------------------------------------


def _idata_with(hot=None):
    """Minimal InferenceData, optionally carrying a posterior_hot group."""
    import arviz as az

    idata = az.from_dict(
        {"posterior": {"toy.x_raw": np.zeros((2, 5))}},
    )
    if hot is not None:
        idata["posterior_hot"] = hot
    return idata


def test_four_hot_search_outcomes_render_differently():
    """
    Given the four possible outcomes of the hot-chain suppressed-mode
      search,
    When each is rendered for the report,
    Then all four texts are distinct and each names its own state, so a
      reader can never mistake "never searched" or "the search failed" for
      the reassurance "searched, nothing found".
    """
    # ARRANGE
    from exozippy.outputs.ledger import (
        HOT_FAILED,
        HOT_FOUND,
        HOT_NONE_FOUND,
        HOT_NOT_SEARCHED,
        hot_status_to_text,
    )

    states = {
        HOT_NOT_SEARCHED: "NOT PERFORMED",
        HOT_NONE_FOUND: "PERFORMED",
        HOT_FAILED: "FAILED",
        HOT_FOUND: "PERFORMED",
    }

    # ACT
    texts = {s: hot_status_to_text({"state": s, "n_new": 2}) for s in states}

    # ASSERT
    assert len(set(texts.values())) == 4
    for state, marker in states.items():
        assert marker in texts[state]
    # the two "did not happen" states must not read as a clean negative
    assert "no additional mode was found" not in texts[HOT_NOT_SEARCHED]
    assert "no additional mode was found" not in texts[HOT_FAILED]
    assert "not the" in texts[HOT_FAILED].lower()
    # and an empty status renders nothing at all
    assert hot_status_to_text({}) == ""


def test_missing_hot_group_reports_not_searched():
    """
    Given a trace with no posterior_hot group (store_hot_chains defaults
      to off, so this is the common case),
    When run_hot_mode_discovery runs,
    Then the status says the search never happened -- NOT that nothing was
      found.
    """
    from exozippy.outputs.ledger import (
        HOT_NOT_SEARCHED,
        run_hot_mode_discovery,
    )

    ledger, status = run_hot_mode_discovery(None, None, _idata_with(None), [])

    assert status["state"] == HOT_NOT_SEARCHED
    assert ledger == []


def test_a_crashing_search_reports_failed_not_none_found(monkeypatch):
    """
    Given hot draws present but the discovery raising,
    When run_hot_mode_discovery runs,
    Then the ledger is returned unchanged (non-fatal, a wrap-up diagnostic
      must not kill a finished fit) AND the status is 'failed' carrying the
      exception type and message -- distinguishable from a clean
      no-candidates run, which is the confusion being fixed.
    """
    # ARRANGE
    from exozippy.outputs import ledger as L

    model, p = _two_basin_model()
    stub = _StubSystem([p])
    hot = _fake_hot_group(model, p, np.random.default_rng(11))

    def _boom(*a, **k):
        raise RuntimeError("clustering blew up")

    monkeypatch.setattr(L, "discover_hot_modes", _boom)

    # ACT
    out_ledger, status = L.run_hot_mode_discovery(
        stub, model, _idata_with(hot), ["sentinel"]
    )

    # ASSERT
    assert out_ledger == ["sentinel"]
    assert status["state"] == L.HOT_FAILED
    assert "RuntimeError" in status["detail"]
    assert "clustering blew up" in status["detail"]
    text = L.hot_status_to_text(status)
    assert "FAILED" in text
    assert text != L.hot_status_to_text({"state": L.HOT_NONE_FOUND})


def test_a_clean_search_that_finds_a_basin_reports_found():
    """
    Given hot draws visiting a basin the ledger does not hold,
    When run_hot_mode_discovery runs,
    Then the status is 'found' with the new-record count.
    """
    from exozippy.outputs import ledger as L

    model, p = _two_basin_model()
    stub = _StubSystem([p])
    raw2 = np.asarray(p.raw_from_initval(np.array([2.0])))
    ledger0 = build_seed_ledger(stub, model, [{"toy.x_raw": raw2}], [0])
    hot = _fake_hot_group(model, p, np.random.default_rng(11))

    out_ledger, status = L.run_hot_mode_discovery(
        stub, model, _idata_with(hot), ledger0, min_points=20
    )

    assert status["state"] == L.HOT_FOUND
    assert status["n_new"] == 1
    assert len(out_ledger) == 2


def test_a_search_finding_only_rediscoveries_reports_none_found():
    """
    Given hot draws whose every cluster is a basin the ledger already
      holds,
    When run_hot_mode_discovery runs,
    Then the status is 'none-found' -- the reassurance the user wants,
      earned by an actual search.
    """
    from exozippy.outputs import ledger as L

    model, p = _two_basin_model()
    stub = _StubSystem([p])
    hot = _fake_hot_group(model, p, np.random.default_rng(11))
    # seed the ledger with BOTH basins so every cluster dedups
    starts = [
        {"toy.x_raw": np.asarray(p.raw_from_initval(np.array([v])))}
        for v in (2.0, 7.0)
    ]
    ledger0 = build_seed_ledger(stub, model, starts, [0, 1])

    out_ledger, status = L.run_hot_mode_discovery(
        stub, model, _idata_with(hot), ledger0, min_points=20
    )

    assert status["state"] == L.HOT_NONE_FOUND
    assert status["n_new"] == 0
    assert len(out_ledger) == 2


def test_an_empty_hot_group_reports_failed_not_none_found():
    """
    Given a posterior_hot group that reached discovery with zero draws --
      what a mis-sliced group looks like (review 2.9.2),
    When discovery runs,
    Then it is reported as a FAILED search, never as "searched and found
      nothing".
    """
    import xarray as xr

    from exozippy.outputs import ledger as L

    empty = xr.Dataset(
        {
            "toy.x_raw": (("chain", "draw"), np.zeros((0, 0))),
            "lp": (("chain", "draw"), np.zeros((0, 0))),
        }
    )
    _ledger, status = L.run_hot_mode_discovery(
        None, None, _idata_with(empty), []
    )

    assert status["state"] == L.HOT_FAILED
    assert "0 draws" in status["detail"]


# ---------------------------------------------------------------------------
# store_hot_chains: 'auto' resolves from the TOPOLOGY -- on where suppressed
# modes are the norm (microlensing), off otherwise.
# ---------------------------------------------------------------------------


class _FakeComponent:
    def __init__(self, expects):
        self.expects_suppressed_modes = expects


class _TopologySystem:
    """Duck-typed System exposing only what the resolver reads."""

    def __init__(self, **components):
        self.active_components = {
            k: _FakeComponent(v) for k, v in components.items()
        }


def _resolve(spec, system, caplog=None, n_temps=8, n_raw=10, n_out=20):
    import logging

    from exozippy.samplers._common import resolve_store_hot_chains

    log = logging.getLogger("exozippy.test.hot")
    return resolve_store_hot_chains(
        spec, system, n_temps, n_raw, n_out, "PTDE-async", log
    )


def test_auto_stores_hot_chains_for_a_microlensing_topology():
    """
    Given a system whose lens component declares expects_suppressed_modes,
    When store_hot_chains is left at its 'auto' default,
    Then hot-rung retention turns ON at the default thinning.

    Microlensing degeneracies (u_0 sign, close/wide s, parallax families)
    are structural, so a solution the T=1 posterior abandons still has to be
    reported -- and hot draws are the only detector for it.
    """
    from exozippy.samplers._common import DEFAULT_HOT_THIN

    system = _TopologySystem(lens=True, mulensinstrument=False, star=False)

    assert _resolve("auto", system) == DEFAULT_HOT_THIN


def test_auto_skips_hot_chains_for_a_transit_or_rv_topology():
    """
    Given a system of components that expect no suppressed modes,
    When store_hot_chains is left at 'auto',
    Then hot-rung retention stays OFF -- the alternative-mode question
      rarely arises there and the storage is not worth it.
    """
    system = _TopologySystem(
        star=False, planet=False, transit=False, rvinstrument=False
    )

    assert _resolve("auto", system) == 0


def test_explicit_false_beats_the_microlensing_default():
    """
    Given a microlensing topology,
    When the user sets store_hot_chains false (or 'off'),
    Then the explicit value wins -- only the UNSET case is topology-driven.
    """
    system = _TopologySystem(lens=True)

    assert _resolve(False, system) == 0
    assert _resolve("off", system) == 0


def test_explicit_integer_thinning_is_honored_in_both_topologies():
    """
    Given an explicit integer thinning factor,
    When it is resolved under either topology,
    Then that exact factor is used -- and in particular the integer 1 means
      "keep every hot draw", not the default thinning (the 1 == True
      collision that cost seed_polish a whole value, review 2.9.1).
    """
    lensy = _TopologySystem(lens=True)
    planar = _TopologySystem(transit=False)

    assert _resolve(5, lensy) == 5
    assert _resolve(5, planar) == 5
    assert _resolve(1, lensy) == 1
    assert _resolve(1, planar) == 1
    # ... while the BOOLEAN True still means the default thinning
    from exozippy.samplers._common import DEFAULT_HOT_THIN

    assert _resolve(True, planar) == DEFAULT_HOT_THIN


def test_the_auto_decision_and_its_cost_are_logged(caplog):
    """
    Given either topology,
    When 'auto' resolves,
    Then the decision, the reason and the trace-size cost are logged -- a
      silent topology-dependent default is what makes behavior feel
      unpredictable.
    """
    import logging

    # ON: names the component that asked for it and quotes the cost
    with caplog.at_level(logging.INFO, logger="exozippy.test.hot"):
        _resolve("auto", _TopologySystem(lens=True))
    assert "auto -> ON" in caplog.text
    assert "lens" in caplog.text
    assert "% of the trace file" in caplog.text

    # OFF: says no search will run, and what enabling it would cost
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="exozippy.test.hot"):
        _resolve("auto", _TopologySystem(transit=False))
    assert "auto -> OFF" in caplog.text
    assert "NO search for suppressed modes will run" in caplog.text
    assert "store_hot_chains: true" in caplog.text


def test_trace_share_matches_the_ob140939_arithmetic():
    """
    Given ob140939's shape -- 24 rungs, thin 20, 19 raw and 39 output
      elements,
    When the predicted hot-group share is computed,
    Then it is ~37% of the trace, matching the measured file.

    n_chains and draws cancel exactly (both groups scale with them), so the
    share is set by the ladder height, the thinning, and the DERIVED-variable
    count -- not by n_temps alone.
    """
    from exozippy.samplers._common import hot_chain_trace_share

    share = hot_chain_trace_share(
        n_temps=24, hot_thin=20, n_raw_elements=19, n_out_elements=39
    )

    assert share == pytest.approx(23.0 / 63.0, rel=1e-9)
    assert 0.36 < share < 0.37
    # a two-rung ladder or no thinning at all degrade sensibly
    assert hot_chain_trace_share(1, 20, 19, 39) == 0.0
    assert hot_chain_trace_share(24, 0, 19, 39) == 0.0


def test_an_unrecognized_store_hot_chains_string_raises():
    """
    Given a typo in the sampler block,
    When it is resolved,
    Then it raises rather than silently meaning "off" -- a misspelled
      opt-in that quietly does nothing is how a search stops running
      without anyone noticing.
    """
    with pytest.raises(ValueError, match="not recognized"):
        _resolve("yes-please", _TopologySystem(lens=True))


def test_lens_component_declares_that_it_expects_suppressed_modes():
    """
    Given the shipped Lens component,
    When its capability flag is read,
    Then it declares expects_suppressed_modes, and the base Component does
      not -- this is the whole topology signal, so pin it here rather than
      only through a duck-typed stand-in.
    """
    from exozippy.components.component import Component
    from exozippy.components.mulensing.lens import Lens

    assert Lens.expects_suppressed_modes is True
    assert Component.expects_suppressed_modes is False


def test_default_store_hot_chains_is_auto_end_to_end():
    """
    Given a model with no microlensing component and no explicit setting,
    When the async sampler runs,
    Then no posterior_hot group is written -- the 'auto' default resolves
      to off for a non-microlensing topology all the way through the
      sampler, not just in the resolver.
    """
    with pm.Model() as model:
        pm.Normal("x", mu=0.0, sigma=1.0)

    idata = ptde_async_sample(
        model,
        _MinimalSystem(),
        draws=30,
        tune=10,
        n_temps=3,
        T_max=8.0,
        n_chains=4,
        cores=1,
        seed=3,
        log_interval=10000,
    )

    assert not hasattr(idata, "posterior_hot")


def test_auto_stores_hot_chains_end_to_end_for_a_lensy_system():
    """
    Given a system whose components declare expects_suppressed_modes and no
      explicit store_hot_chains,
    When the async sampler runs,
    Then the posterior_hot group IS written -- the microlensing default
      reaches the sampler, which is the behavior change users will see.
    """
    with pm.Model() as model:
        pm.Normal("x", mu=0.0, sigma=1.0)

    class _LensySystem(_MinimalSystem):
        active_components = {"lens": _FakeComponent(True)}

    idata = ptde_async_sample(
        model,
        _LensySystem(),
        draws=60,
        tune=20,
        n_temps=3,
        T_max=8.0,
        n_chains=4,
        cores=1,
        seed=7,
        log_interval=10000,
    )

    assert hasattr(idata, "posterior_hot")
    assert idata.posterior_hot.sizes["chain"] == (3 - 1) * 4
