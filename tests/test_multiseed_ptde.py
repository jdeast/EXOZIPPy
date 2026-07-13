# tests/test_multiseed_ptde.py
"""Tests for multi-seed chain distribution in the PTDE sampler (P4 layer c).

_make_starts is the SHARED helper imported by both ptde.py and ptde_async.py,
so testing it here covers both samplers' chain-assignment logic. Chains are
assigned to seeds round-robin (chain j -> seed j % K); the first chain of
each seed group starts exactly at that seed, later chains in the group jitter
around it.
"""
import numpy as np
import pymc as pm
import pytest
import xarray as xr

from exozippy.samplers.ptde import _make_starts, ptde_sample


class _MinimalSystem:
    active_components = {}

    def get_raw_start(self, model):
        return model.initial_point()


def _simple_model():
    with pm.Model() as model:
        pm.Normal("x", mu=0.0, sigma=1.0)
        pm.Normal("y", mu=3.0, sigma=0.5)
    return model


def test_make_starts_round_robins_chains_across_two_seeds():
    """
    Given 2 raw starts (seed indices [0, 7]) and n_chains=4,
    When _make_starts runs,
    Then chains are assigned round-robin (0->seed0, 1->seed7, 2->seed0,
    3->seed7); the first chain of each seed group starts EXACTLY at that
    seed's raw values (no jitter).
    """
    model = _simple_model()
    with model:
        logp_fn = model.compile_logp()
    rng = np.random.default_rng(0)

    start0 = {"x": np.array(0.0), "y": np.array(3.0)}
    start1 = {"x": np.array(1.5), "y": np.array(-2.0)}

    starts, chain_seed_index = _make_starts(
        4, [start0, start1], logp_fn, rng, seed_indices=[0, 7])

    assert len(starts) == 4
    assert chain_seed_index == [0, 7, 0, 7]
    # First chain of each seed group is the exact (unjittered) seed center.
    assert np.allclose(starts[0]["x"], start0["x"])
    assert np.allclose(starts[0]["y"], start0["y"])
    assert np.allclose(starts[1]["x"], start1["x"])
    assert np.allclose(starts[1]["y"], start1["y"])


def test_make_starts_single_start_backward_compatible():
    """
    Given a single raw-start dict (not a list; legacy call signature),
    When _make_starts runs,
    Then it behaves exactly as before: every chain seeded from that one
    start, chain_seed_index is all zeros.
    """
    model = _simple_model()
    with model:
        logp_fn = model.compile_logp()
    rng = np.random.default_rng(0)

    start0 = model.initial_point()
    starts, chain_seed_index = _make_starts(4, start0, logp_fn, rng)

    assert len(starts) == 4
    assert chain_seed_index == [0, 0, 0, 0]


def test_ptde_sample_records_chain_seed_index_in_posterior_attrs():
    """
    Given ptde_sample called with raw_starts=[start0, start1] and
    seed_indices=[0, 1] (2 seeds, round-robinned over 4 chains),
    When sampling completes,
    Then idata.posterior.attrs['chain_seed_index'] records which seed each
    T=1 chain was drawn from -- because with seeded starts, occupancy weights
    are initialization artifacts by design unless chains mix, downstream
    reporting needs this provenance.
    """
    model = _simple_model()
    system = _MinimalSystem()

    start0 = {"x": np.array(0.0), "y": np.array(3.0)}
    start1 = {"x": np.array(0.2), "y": np.array(2.8)}

    idata = ptde_sample(
        model, system, draws=20, tune=20,
        n_temps=2, T_max=2.0, n_chains=4, cores=1, seed=0,
        raw_starts=[start0, start1], seed_indices=[0, 1],
        log_interval=100,
    )

    assert isinstance(idata, xr.DataTree)
    assert "chain_seed_index" in idata.posterior.attrs
    recorded = idata.posterior.attrs["chain_seed_index"]
    assert len(recorded) == 4
    assert set(recorded) <= {0, 1}
    # Round-robin: chains 0,2 -> seed 0; chains 1,3 -> seed 1.
    assert recorded == [0, 1, 0, 1]
