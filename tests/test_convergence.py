"""Tests for post-hoc burn-in detection and convergence diagnostics
(src/exozippy/samplers/convergence.py).

The pathology these guard against: a run whose reported statistics discard
zero burn-in even though a slow, likelihood-flat degenerate direction drifts
for a large fraction of the trace (DC2018_128; see notes/todo.txt). Detection
must happen in parameter space (rank Rhat/bulk-ESS), the burn-in must land at
the ESS knee, and stuck chains must be dropped before any statistic.
"""

import numpy as np
import pytest

from exozippy.samplers import convergence as C


def _stationary(nc, nd, seed=0):
    return np.random.default_rng(seed).standard_normal((nc, nd))


def _transient(nc, nd, frac, seed=0):
    """A trace with a between-chain-divergent drift over the first `frac`,
    stationary thereafter -- the shape a burn-in trimmer must catch."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((nc, nd))
    k = int(frac * nd)
    offsets = np.linspace(-4, 4, nc)[:, None]
    x[:, :k] += offsets * np.linspace(1.0, 0.0, k)[None, :]
    return x


# --- good_chain_mask ------------------------------------------------------

def test_good_chain_mask_drops_stuck_chain():
    # Given six chains where one never reaches the good-likelihood region
    rng = np.random.default_rng(3)
    lp = rng.standard_normal((6, 500)) - 2.0
    lp[3] -= 50.0
    # When we compute the good-chain mask
    mask, reliable = C.good_chain_mask(lp)
    # Then only the stuck chain is dropped and the verdict is reliable
    assert reliable is True
    assert mask[3] == False  # noqa: E712
    assert mask.sum() == 5


def test_good_chain_mask_keeps_all_when_too_few_good():
    # Given a trace where fewer than three chains reach the good region
    lp = np.full((5, 200), -100.0)
    lp[0] = 0.0  # only one good chain
    # When we compute the mask
    mask, reliable = C.good_chain_mask(lp)
    # Then all chains are kept and the fallback is flagged unreliable
    assert mask.all()
    assert reliable is False


# --- find_burnin ----------------------------------------------------------

def test_find_burnin_stationary_trims_nothing():
    # Given an already-stationary trace
    x = {"p.x": _stationary(8, 4000)}
    # When we search for burn-in
    diag = C.find_burnin(x)
    # Then no burn-in is trimmed and Rhat is ~1
    assert diag["burnin"] == 0
    assert diag["max_rhat"] < 1.02


def test_find_burnin_locates_transient_knee():
    # Given a trace with a divergent drift over its first 40%
    x = {"p.x": _transient(8, 4000, frac=0.4)}
    # When we search for burn-in
    diag = C.find_burnin(x)
    # Then the burn-in lands at the end of the transient (not zero, not
    # deep into the stationary tail) and Rhat drops to ~1 after trimming
    assert 0.25 <= diag["burnin"] / diag["n_draws"] <= 0.55
    assert diag["max_rhat"] < 1.02


def test_find_burnin_transient_is_worse_before_trimming():
    # Given the same drifting trace
    x = {"p.x": _transient(8, 4000, frac=0.4)}
    # When we compare Rhat on the full trace vs the trimmed trace
    full = C._rhat_ess({"p.x": x["p.x"]})
    diag = C.find_burnin(x)
    # Then the untrimmed Rhat is clearly non-converged and trimming fixes it
    assert full[0] > 1.1
    assert diag["max_rhat"] < full[0]


# --- converged_on_tail (the live auto-stop test) --------------------------

def test_converged_on_tail_true_for_stationary():
    # Given a stationary trace
    x = {"p.x_raw": _stationary(6, 3000)}
    # When we run the cheap tail check
    converged, rhat, ess = C.converged_on_tail(x, None, min_ess=100,
                                               max_rhat=1.01)
    # Then it reports converged
    assert converged is True
    assert rhat < 1.01


def test_converged_on_tail_false_when_transient_exceeds_tail():
    # Given a transient occupying the first 70% -- longer than the 50% tail
    x = {"p.x_raw": _transient(6, 3000, frac=0.7)}
    # When we run the cheap tail check (keeps the last half)
    converged, rhat, ess = C.converged_on_tail(x, None, min_ess=100,
                                               max_rhat=1.01)
    # Then the still-contaminated tail keeps it from declaring convergence
    assert converged is False


# --- default_var_names ----------------------------------------------------

def test_default_var_names_drops_raw_when_physical_present():
    # Given a posterior carrying both physical and raw copies plus mode
    posterior = {"p.x": None, "p.x_raw": None, "mode": None}
    # When we pick the variables to judge
    names = C.default_var_names(posterior)
    # Then only the physical copy survives
    assert names == ["p.x"]


def test_default_var_names_keeps_raw_only_store():
    # Given a raw-only store (PTDE's live trace: every key ends in _raw)
    posterior = {"p.x_raw": None, "p.y_raw": None}
    # When we pick the variables to judge
    names = C.default_var_names(posterior)
    # Then the raw variables are kept (otherwise nothing to judge)
    assert names == ["p.x_raw", "p.y_raw"]


# --- analyze_idata (the reporting entry point) ----------------------------

def test_analyze_idata_trims_and_flags(monkeypatch):
    az = pytest.importorskip("arviz")
    # Given an InferenceData with a transient, a raw duplicate, and lp
    raw = _transient(6, 2000, frac=0.3, seed=5)
    idata = az.from_dict({
        "posterior": {"p.x": raw * 2.0 + 10.0, "p.x_raw": raw,
                      "mode": np.zeros((6, 2000), int)},
        "sample_stats": {"lp": -0.5 * raw ** 2},
    })
    # When we analyze it
    trimmed, diag = C.analyze_idata(idata, min_ess=100, max_rhat=1.01)
    # Then draws are trimmed, chains preserved, and the diag reports a verdict
    assert diag["burnin"] > 0
    assert trimmed.posterior.sizes["draw"] == 2000 - diag["burnin"]
    assert trimmed.sample_stats.sizes["draw"] == 2000 - diag["burnin"]
    assert isinstance(diag["converged"], bool)
    assert diag["max_rhat"] < 1.02
