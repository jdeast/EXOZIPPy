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
