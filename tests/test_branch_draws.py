"""Reporting a branch-marginalized posterior (exozippy/branches.py).

The likelihood of a many-to-one parameterization is a mixture over a discrete
branch indicator the sampler never sees (System._add_branch_mixtures), so the
trace's Deterministics are all the PRIMARY branch.  The report draws each
draw's branch combination from the mixture's own per-combination weights and
re-derives every branch-dependent quantity under it -- one draw per draw,
never an average, and the JOINT combination when there are several branches
(JDE ruling 2026-10-01, review 1.8.14).

The toy models here are small enough that the exact posterior is a quadrature,
so the reported histograms are checked against the answer rather than against
another sampler.
"""

import arviz as az
import numpy as np
import pymc as pm
import pytest
import scipy.stats as st

from exozippy.branches import (
    branch_probabilities,
    branch_summary_lines,
    resolve_branch_draws,
)
from exozippy.system import System

SIGMA_X = 0.1  # prior width on x: both branches compete at every draw
SIGMA_Y = 1.0  # likelihood width on the branch-dependent d


def _toy_system():
    """x ~ N(0, 0.1); d = x + 1 (primary) or x - 1 (alternative); y=0 ~ N(d, 1).

    Both branches are comparably supported at every x, so the exact marginal
    posterior of the reported d is BIMODAL at +/-1 -- and a per-draw
    responsibility-weighted average would put it near 0, where neither branch
    is.  `g = 2 d` is a second Deterministic downstream of d, for the
    consistency claim.
    """
    system = System.__new__(System)
    system._branch_alternatives = []
    with pm.Model() as model:
        x = pm.Normal("x", 0.0, SIGMA_X)
        d = pm.Deterministic("d", x + 1.0)
        pm.Deterministic("g", 2.0 * d)
        pm.Normal("y", mu=d, sigma=SIGMA_Y, observed=np.array([0.0]))
        system.register_branch_alternative("toy", {d: d - 2.0})
        system._add_branch_mixtures(model)
    return system, model


def _exact_joint(xgrid):
    """Unnormalized p(x, z) on a grid, z = 0 (d = x+1) and z = 1 (d = x-1)."""
    prior = st.norm.pdf(xgrid, 0.0, SIGMA_X)
    return np.stack(
        [
            prior * 0.5 * st.norm.pdf(0.0, xgrid + 1.0, SIGMA_Y),
            prior * 0.5 * st.norm.pdf(0.0, xgrid - 1.0, SIGMA_Y),
        ]
    )


def _exact_x_draws(n_chains, n_draws, seed):
    """Exact draws of x from its MARGINAL posterior (what the sampler sees)."""
    xgrid = np.linspace(-1.0, 1.0, 200_001)
    marg = _exact_joint(xgrid).sum(axis=0)
    cdf = np.cumsum(marg)
    cdf /= cdf[-1]
    u = np.random.default_rng(seed).uniform(size=(n_chains, n_draws))
    return np.interp(u, cdf, xgrid)


def _toy_idata(x, seed=1234):
    idata = az.from_dict(
        {"posterior": {"x": x, "d": x + 1.0, "g": 2.0 * (x + 1.0)}}
    )
    idata.posterior.attrs["random_seed"] = seed
    return idata


def _exact_d_cdf():
    """CDF of the reported d under the exact joint posterior p(x, z)."""
    dgrid = np.linspace(-3.0, 3.0, 600_001)
    joint_on_d = (
        _exact_joint(dgrid - 1.0)[0] + _exact_joint(dgrid + 1.0)[1]
    )  # z = 0 at x = d - 1, z = 1 at x = d + 1
    cdf = np.cumsum(joint_on_d)
    return dgrid, cdf / cdf[-1]


def test_the_reported_draws_follow_the_exact_marginal_posterior():
    """
    Given exact marginal-posterior draws of a two-branch toy model,
    When each draw's branch is drawn from the mixture's own weights,
    Then the reported branch-dependent quantity follows its EXACT marginal
      posterior (bimodal at +/-1), every downstream Deterministic is evaluated
      on the same branch, and a per-draw weighted average would NOT have
      followed it.
    """
    system, model = _toy_system()
    x = _exact_x_draws(4, 5000, seed=0)
    idata = _toy_idata(x)

    regenerated = resolve_branch_draws(system, model, idata, cores=1)

    assert regenerated == {"d", "g"}
    d = idata.posterior["d"].values
    z = idata.sample_stats["branch_combination"].values
    np.testing.assert_allclose(d, np.where(z == 1, x - 1.0, x + 1.0))
    np.testing.assert_allclose(idata.posterior["g"].values, 2.0 * d)

    dgrid, cdf = _exact_d_cdf()
    ks = st.kstest(d.ravel(), lambda v: np.interp(v, dgrid, cdf))
    assert ks.pvalue > 1e-3, ks

    # The tempting alternative: average the two branches by responsibility.
    log_w = idata.sample_stats["branch_log_weight"].values
    p_alt = np.exp(log_w[..., 1] - np.logaddexp(log_w[..., 0], log_w[..., 1]))
    averaged = (1 - p_alt) * (x + 1.0) + p_alt * (x - 1.0)
    ks_avg = st.kstest(averaged.ravel(), lambda v: np.interp(v, dgrid, cdf))
    assert ks_avg.pvalue < 1e-30
    # ...because it lands between the modes, where the posterior has no mass.
    assert np.mean(np.abs(averaged) < 0.5) > 0.9
    assert np.mean(np.abs(d) < 0.5) < 0.05


def test_the_branch_probability_is_the_rao_blackwellized_mean():
    """
    Given the toy model's per-draw combination weights,
    When the branch probability is reported,
    Then it is the mean responsibility (an expectation -- the one place a
      weighted average is right), it matches the exact posterior probability
      of the alternative branch, and the summary file carries it.
    """
    system, model = _toy_system()
    idata = _toy_idata(_exact_x_draws(4, 5000, seed=1))
    resolve_branch_draws(system, model, idata, cores=1)

    xgrid = np.linspace(-1.0, 1.0, 200_001)
    joint = _exact_joint(xgrid)
    exact = joint[1].sum() / joint.sum()
    attrs = idata.sample_stats["branch_combination"].attrs
    (p_alt,) = np.atleast_1d(attrs["branch_probability"])
    assert p_alt == pytest.approx(exact, abs=0.005)
    assert any("toy: " in line for line in branch_summary_lines(idata))


def test_the_branch_draw_is_reproducible_from_the_trace_seed():
    """
    Given the same trace twice, and once with a different stamped seed,
    When the branches are drawn,
    Then the same seed gives identical combinations and a different seed
      different ones -- the report is a function of the trace alone.
    """
    x = _exact_x_draws(2, 2000, seed=2)
    out = []
    for seed in (7, 7, 8):
        system, model = _toy_system()
        idata = _toy_idata(x.copy(), seed=seed)
        resolve_branch_draws(system, model, idata, cores=1)
        out.append(idata.sample_stats["branch_combination"].values.copy())
    np.testing.assert_array_equal(out[0], out[1])
    assert np.mean(out[0] != out[2]) > 0.2


def _two_branch_system():
    """Two branches whose choices the likelihood CORRELATES.

    d1 = x1 + 1 or x1 - 1, d2 = x2 + 1 or x2 - 1, and the data measure only
    d1 - d2 = 0 (tightly): (hi, hi) and (lo, lo) fit, the mixed pairs miss by
    2.  Each branch alone is 50/50, so drawing them independently from their
    marginal responsibilities pairs them wrongly half the time.
    """
    system = System.__new__(System)
    system._branch_alternatives = []
    with pm.Model() as model:
        x1 = pm.Normal("x1", 0.0, SIGMA_X)
        x2 = pm.Normal("x2", 0.0, SIGMA_X)
        d1 = pm.Deterministic("d1", x1 + 1.0)
        d2 = pm.Deterministic("d2", x2 + 1.0)
        pm.Normal("y", mu=d1 - d2, sigma=0.05, observed=np.array([0.0]))
        system.register_branch_alternative("one", {d1: d1 - 2.0})
        system.register_branch_alternative("two", {d2: d2 - 2.0})
        system._add_branch_mixtures(model)
    return system, model


def test_several_branches_are_drawn_jointly_not_independently():
    """
    Given two branches the likelihood correlates,
    When the combinations are drawn,
    Then every draw pairs them the way the posterior does (d1 - d2 ~ 0), each
      branch is still 50/50 on its own, and independent per-branch draws from
      those same marginals would have mispaired about half the draws.
    """
    system, model = _two_branch_system()
    rng = np.random.default_rng(3)
    x1 = rng.normal(0.0, 0.03, size=(4, 1000))
    x2 = x1 + rng.normal(0.0, 0.03, size=x1.shape)
    idata = az.from_dict(
        {"posterior": {"x1": x1, "x2": x2, "d1": x1 + 1.0, "d2": x2 + 1.0}}
    )
    idata.posterior.attrs["random_seed"] = 11

    resolve_branch_draws(system, model, idata, cores=1)

    d1, d2 = idata.posterior["d1"].values, idata.posterior["d2"].values
    assert np.all(np.abs(d1 - d2) < 0.5)
    z = idata.sample_stats["branch_combination"].values
    assert set(np.unique(z)) <= {0, 3}  # (hi, hi) or (lo, lo), never mixed

    log_w = idata.sample_stats["branch_log_weight"].values
    prob = np.exp(log_w - log_w.max(axis=-1, keepdims=True))
    prob /= prob.sum(axis=-1, keepdims=True)
    p_one, p_two = branch_probabilities(prob, 2)
    assert p_one == pytest.approx(0.5, abs=0.05)
    assert p_two == pytest.approx(0.5, abs=0.05)
    # Independent draws from those marginals: mispaired ~ 2 p (1 - p).
    mispaired = 2 * p_one * (1 - p_two)
    assert mispaired > 0.4
