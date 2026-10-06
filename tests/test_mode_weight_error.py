"""
Calibration of the mode-occupancy weight error (review 1.11.3).

identify_modes quotes each mode's occupancy weight with a 1-sigma.  Until
1.11.3 that sigma came from the per-chain mode-indicator IACT alone
(weight_ess), which treats the chains as independent.  On ob140939 it was
8-23x smaller than the run-to-run scatter of the weight.  The mechanism,
measured on the traces on disk: individual chains exchange mode labels
quickly, so every per-chain indicator decorrelates in a few draws, while the
FRACTION of the population in a mode drifts slowly as one collective
variable that every chain shares.  population_weight_error measures that
collective series directly.

These tests run the estimators on synthetic indicator processes whose true
occupancy is known, over many independent repeats, and score the quoted
1-sigma against the actual scatter of the estimate about the truth:

  * independent chains (fast and rare switching) -- the old estimator was
    already right here, and the new quote must not break it;
  * a coupled population: a latent population fraction g(t) (an AR(1) about
    the true weight w with correlation time tau_s), with each chain keeping
    its label or redrawing it from Bernoulli(g(t)) every draw -- fast
    per-chain switching on top of slow common-mode drift, which is what the
    ob140939 traces show.  The ob140939-like case uses the measured numbers:
    34 chains, per-chain IACT ~6 draws, population drift of sd ~0.075 on a
    few hundred draws, and the 1548 draws of the shortest converged run.
"""

import arviz as az
import numpy as np
import pytest

from exozippy.outputs.modes import (
    identify_modes,
    occupancy_weight_error,
    population_weight_error,
)

N_REPEATS = 200
N_CHAINS = 34
TRUE_W = 0.7


def _independent_chains(rng, n_draws, tau):
    """Independent two-state chains with indicator IACT ~tau, stationary."""
    s = 2.0 / (tau + 1.0)
    p01, p10 = TRUE_W * s, (1.0 - TRUE_W) * s
    x = np.empty((N_REPEATS, N_CHAINS, n_draws), dtype=np.int8)
    cur = rng.random((N_REPEATS, N_CHAINS)) < TRUE_W
    for t in range(n_draws):
        u = rng.random((N_REPEATS, N_CHAINS))
        cur = np.where(cur, u >= p10, u < p01)
        x[:, :, t] = cur
    return x


def _coupled_population(rng, n_draws, tau_s, sd=0.075, keep=0.85):
    """Fast per-chain switching around a slow shared population fraction."""
    phi = np.exp(-1.0 / tau_s)
    g = TRUE_W + sd * rng.standard_normal(N_REPEATS)
    cur = rng.random((N_REPEATS, N_CHAINS)) < g[:, None]
    x = np.empty((N_REPEATS, N_CHAINS, n_draws), dtype=np.int8)
    for t in range(n_draws):
        g = (
            TRUE_W
            + phi * (g - TRUE_W)
            + sd * np.sqrt(1.0 - phi**2) * rng.standard_normal(N_REPEATS)
        )
        redraw = rng.random((N_REPEATS, N_CHAINS)) >= keep
        new = rng.random((N_REPEATS, N_CHAINS)) < np.clip(g, 0, 1)[:, None]
        cur = np.where(redraw, new, cur)
        x[:, :, t] = cur
    return x


def _quotes(labels_2d):
    """(weight, old per-chain sigma, quoted sigma, unresolved) for mode 1,
    from the same function identify_modes calls."""
    occ = occupancy_weight_error(labels_2d, 1)
    return occ.weight, occ.sigma_chain, occ.sigma, occ.unresolved


def _score(x):
    """Scatter of the weight about the truth vs each quoted 1-sigma."""
    rows = [_quotes(x[r].astype(int)) for r in range(x.shape[0])]
    w, old, new, unresolved = (np.array(c) for c in zip(*rows))
    err = np.abs(w - TRUE_W)
    rms = float(np.sqrt(np.mean(err**2)))
    return {
        "rms_over_old": rms / float(np.median(old)),
        "rms_over_new": rms / float(np.median(new)),
        "cover1_new": float(np.mean(err < new)),
        "cover2_new": float(np.mean(err < 2 * new)),
        "cover2_old": float(np.mean(err < 2 * old)),
        "frac_unresolved": float(np.mean(unresolved)),
    }


def test_independent_fast_chains_stay_calibrated():
    """
    Given independent chains that switch modes every ~20 draws -- the case
      the per-chain estimator was built for and already got right,
    When the weight is estimated over 200 repeats with known truth,
    Then the new quote covers the truth at the nominal 1- and 2-sigma rates
      (it is not inflated into uselessness), and the unresolved flag stays
      quiet.
    """
    s = _score(_independent_chains(np.random.default_rng(1), 2000, 20))

    assert 0.65 < s["rms_over_new"] < 1.3
    assert 0.6 < s["cover1_new"] < 0.9
    assert s["cover2_new"] > 0.9
    assert s["frac_unresolved"] < 0.15


def test_independent_rare_switching_chains_stay_calibrated():
    """
    Given independent chains with RARE switches (indicator IACT 400 draws in
      a 1500-draw run, a handful of transitions per chain) -- the "short
      series with few transitions" half of 1.11.3's diagnosis,
    When the weight is estimated over 200 repeats,
    Then the quote still covers the truth (the per-chain estimator's
      between-chain term already sees this case; the new quote keeps it).
    """
    s = _score(_independent_chains(np.random.default_rng(2), 1500, 400))

    assert s["rms_over_new"] < 1.2
    assert s["cover2_new"] > 0.9


def test_coupled_population_ob140939_like_old_fails_new_covers():
    """
    Given the ob140939-like coupled population (34 chains, fast per-chain
      switching, a shared population fraction drifting with sd 0.075 on a
      300-draw timescale, 1548 draws),
    When the weight is estimated over 200 repeats with known truth,
    Then the per-chain estimate is several times too small -- 1.11.3's
      failure, reproduced -- while the new quote matches the real scatter
      to within ~30% and covers the truth at close to the 2-sigma rate, and
      the run is flagged as not resolving the drift.
    """
    s = _score(_coupled_population(np.random.default_rng(3), 1548, 300))

    assert s["rms_over_old"] > 4.0
    assert s["cover2_old"] < 0.4
    assert s["rms_over_new"] < 1.4
    assert s["cover1_new"] > 0.5
    assert s["cover2_new"] > 0.75
    assert s["frac_unresolved"] > 0.7


def test_coupled_population_resolved_run_is_calibrated():
    """
    Given a coupled population whose drift is fast compared with the run
      (tau_s = 50 draws in a 5000-draw run),
    When the weight is estimated over 200 repeats,
    Then the per-chain estimate is still ~3x too small (the coupling alone
      defeats it), while the new quote covers the truth at the nominal
      rates.
    """
    s = _score(_coupled_population(np.random.default_rng(4), 5000, 50))

    assert s["rms_over_old"] > 2.0
    assert 0.6 < s["rms_over_new"] < 1.2
    assert s["cover1_new"] > 0.6
    assert s["cover2_new"] > 0.9


def test_population_weight_error_drops_unassigned_draws():
    """
    Given a label array whose (-1) unassigned draws sit in a few columns,
    When population_weight_error measures it,
    Then the result is finite, and a mode with no assigned draws at all
      RAISES naming the mode rather than returning a fabricated zero.
    """
    rng = np.random.default_rng(5)
    labels = (rng.random((8, 400)) < 0.3).astype(int)
    labels[:, 10:20] = -1
    labels[2, 50:60] = -1

    pop = population_weight_error(labels, 1)

    assert np.isfinite(pop.sigma_iact) and pop.sigma_iact > 0
    assert np.isfinite(pop.sigma_batch) and np.isfinite(pop.sigma_scaled)
    with pytest.raises(ValueError, match="mode 1"):
        population_weight_error(np.full((4, 50), -1), 1)


def test_identify_modes_quotes_the_population_error_and_warns(caplog):
    """
    Given a two-mode trace whose labels come from the ob140939-like coupled
      population,
    When identify_modes runs,
    Then the quoted weight error is the population one (larger than the
      per-chain estimate it used to quote), the per-chain value is kept for
      comparison, the report names which estimate set the error, and an
      advisory note + WARNING says the error is approximate -- without
      blocking anything.
    """
    rng = np.random.default_rng(6)
    lab = _coupled_population(rng, 1548, 300)[0].astype(int)
    n_chain, n_draw = lab.shape
    a = rng.normal(0, 1, lab.shape) + 10 * lab
    idata = az.from_dict(
        {
            "posterior": {"a_raw": a},
            "sample_stats": {"lp": rng.normal(0, 1, (n_chain, n_draw))},
        }
    )

    with caplog.at_level("WARNING", logger="exozippy.outputs.modes"):
        rep = identify_modes(idata)

    assert rep.n_modes == 2
    m = rep.modes[0]
    assert m.weight_err == m.occ_weight_err
    assert m.weight_err > 2 * m.weight_err_chain
    assert m.weight_err_source.startswith("population")
    assert m.weight_ess == pytest.approx(
        m.occ_weight * (1 - m.occ_weight) / m.weight_err**2
    )
    assert m.weight_err_unresolved
    text = rep.to_text()
    assert "error set by: population" in text
    assert "per-chain IACT, chains as independent" in text
    assert any("mode-weight error is APPROXIMATE" in n for n in rep.notes)
    assert "mode-weight error is APPROXIMATE" in caplog.text
