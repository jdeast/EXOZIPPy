"""The galactic-model microlensing prior against genulens and OGLE (7.16.3, 1.7.6).

Harness: scripts/galactic_prior_validation/validate.py (package
scripts/galactic_prior_validation/galactic_prior/); method and results:
docs/galactic_prior_validation.md.

FAST TESTS hold the pure-numpy machinery the comparison rests on: weighted
quantiles, KS distances, the weighted bootstrap, the importance-sampling
diagnostics (Kish ESS, Pareto k), the proposal densities, and the
reproduction of genulens's line-of-sight support on the port's draws.

THE SLOW TEST evaluates the prior (lens + source + galacticmodel, no light
curve) toward Baade's window by importance sampling -- NOT by NUTS, which
never visits the black-hole lens mode (0 of 192000 draws above 3 Msun where
the prior puts 1.5% of the event rate; that alone shortens t_E's q90 / q95
by ~4% / 7%) -- and compares it with the stored genulens 2.0.0a4 run
(unmodified, runtime option Dmax 20000, REMNANT 1, heliocentric), the
port's draws restricted to genulens's lens-in-front-of-the-source's-bin
support.

TOLERANCE, per quantile (q05 ... q95 of t_E, D_L, D_S, mu_rel, M_L):

    |port / genulens - 1| <= 4 sqrt(se_port^2 + se_gen^2) / q_gen + 0.01

* se_port: the weighted-bootstrap error of THIS run's quantile (a smaller
  run than the stored reference, so it is the larger term);
* se_gen: the stored weighted-bootstrap error of genulens's quantile;
* 0.01: the agreement floor the full-size comparison measures on 15 sight
  lines -- the stated approximations of the port (the Shu distribution
  function's tabulated normalization, the distribution function at a star's
  own (R, z) rather than genulens's table node, isotropic-base kicks) each
  move a tail quantile by a few tenths of a percent.

The black-hole share of the rate (M_L > 3 Msun) must be within 4 combined
sigma of genulens's, which a sampler that misses that mode fails at ~14
sigma.
"""

import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

VALIDATION = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "galactic_prior_validation"
)


@pytest.fixture(scope="module")
def gp():
    """The galactic_prior package (its numpy-only modules)."""
    sys.path.insert(0, str(VALIDATION))
    try:
        yield {
            name: importlib.import_module(f"galactic_prior.{name}")
            for name in ("stats", "sightlines", "importance", "refs")
        }
    finally:
        sys.path.remove(str(VALIDATION))


# ----------------------------------------------------------------------
# fast: statistics helpers
# ----------------------------------------------------------------------
def test_weighted_quantiles_match_numpy_and_repetition(gp):
    """
    Given equal weights, the weighted quantile is numpy's hazen quantile;
    given integer weights, it is close to the quantile of the repeated
    sample.
    """
    st = gp["stats"]
    rng = np.random.default_rng(1)
    x = rng.normal(size=1001)
    q = (0.05, 0.5, 0.95)
    np.testing.assert_allclose(
        st.weighted_quantiles(x, None, q),
        np.quantile(x, q, method="hazen"),
        rtol=0,
        atol=1e-12,
    )
    y = rng.normal(size=200)
    k = rng.integers(1, 5, size=200)
    np.testing.assert_allclose(
        st.weighted_quantiles(y, k.astype(float), q),
        np.quantile(np.repeat(y, k), q, method="hazen"),
        atol=0.05,
    )


def test_weighted_ks_distance(gp):
    """
    Given two weighted samples, the KS distance is 0 for the same sample,
    and for N(0,1) vs N(0.5,1) it approaches the analytic
    2 Phi(0.25) - 1 = 0.197.
    """
    st = gp["stats"]
    rng = np.random.default_rng(2)
    a = rng.normal(size=50000)
    assert st.weighted_ks(a, None, a, np.full(a.size, 3.0)) == 0.0
    b = rng.normal(0.5, 1.0, size=50000)
    d = st.weighted_ks(a, np.ones(a.size), b, np.ones(b.size))
    assert abs(d - 0.1974) < 0.015


def test_bootstrap_quantile_se_matches_asymptotic(gp):
    """
    Given 1e5 equal-weight N(0,1) draws, the bootstrap se of the median is
    the asymptotic sqrt(pi / 2) / sqrt(n) to within 20%.
    """
    st = gp["stats"]
    rng = np.random.default_rng(3)
    x = rng.normal(size=100_000)
    se = st.bootstrap_quantile_se({"x": x}, np.ones(x.size), (0.5,), n_boot=60)
    expect = np.sqrt(np.pi / 2) / np.sqrt(x.size)
    assert abs(se["x"][0] / expect - 1) < 0.2


def test_pareto_khat_recovers_tail_shape(gp):
    """
    Given weights with a Pareto tail of shape k = 0.5, k-hat is ~0.5;
    given bounded weights, k-hat is negative; the Kish ESS of equal
    weights is n.
    """
    st = gp["stats"]
    rng = np.random.default_rng(4)
    u = rng.uniform(size=100_000)
    assert abs(st.pareto_khat(-0.5 * np.log(u)) - 0.5) < 0.1
    assert st.pareto_khat(np.log(rng.uniform(1, 2, size=100_000))) < 0.0
    assert st.kish_ess(np.ones(500)) == pytest.approx(500)


def test_ogle_reconstruction_matches_stored_reference(gp):
    """
    Given the committed Mroz et al. 2019 BLG513 events and Mroz et al. 2017
    BLG505 table, the reconstructed efficiency-corrected median t_E equals
    the stored mroz_tE_reference.json value.
    """
    refs = gp["refs"]
    stored = json.loads((refs.REFS / "mroz_tE_reference.json").read_text())
    for line, (paper, field) in (
        ("baade", ("mroz2019", "BLG513")),
        ("blg505", ("mroz2017", "BLG505")),
    ):
        kind, data = refs.load_ogle(line)
        q, se = refs.ogle_quantiles(kind, data, (0.5,), n_boot=50)
        assert q[0] == pytest.approx(stored[paper][field]["q50"], rel=1e-9)
        assert 0 < se[0] < 0.2 * q[0]


# ----------------------------------------------------------------------
# fast: genulens's support, reproduced on the port's draws
# ----------------------------------------------------------------------
def test_genulens_bin_width_and_nsd_box(gp):
    """
    Given genulens's grid rule, dD is 100 pc off its nuclear-disk box and
    25 pc inside it for Dmax 16 and 20 kpc, and the box is the same as the
    port's (koshimoto21.nsd_active).
    """
    from exozippy.components.galacticmodel import koshimoto21

    sl = gp["sightlines"]
    for dmax in (16000.0, 20000.0):
        assert sl.genulens_bin_width(1.0, -3.9, dmax) == 100.0
        assert sl.genulens_bin_width(0.3, -1.6, dmax) == 25.0
    for l_deg in np.linspace(-8, 8, 33):
        for b_deg in np.linspace(-4, 4, 33):
            assert sl.genulens_nsd_active(
                l_deg, b_deg
            ) == koshimoto21.nsd_active(l_deg, b_deg)


def test_genulens_support_mask(gp):
    """
    Given a source at 8050 pc and dD = 100 pc, a lens is inside genulens's
    support only in front of 8000 pc; with Dmax the source must be inside it.
    """
    sl = gp["sightlines"]
    d_l = np.array([7999.0, 8001.0, 7000.0, 7000.0])
    d_s = np.array([8050.0, 8050.0, 15999.0, 16001.0])
    np.testing.assert_array_equal(
        sl.genulens_support_mask(d_l, d_s, 100.0), [True, False, True, True]
    )
    np.testing.assert_array_equal(
        sl.genulens_support_mask(d_l, d_s, 100.0, 16000.0),
        [True, False, True, False],
    )


# ----------------------------------------------------------------------
# fast: proposal densities and an importance estimate of a hidden mode
# ----------------------------------------------------------------------
def test_tmixture_logpdf_matches_scipy(gp):
    """
    Given one Student-t component, the mixture's logpdf is scipy's
    multivariate_t.
    """
    from scipy.stats import multivariate_t

    imp = gp["importance"]
    cov = np.array([[2.0, 0.3], [0.3, 0.5]])
    mix = imp.TMixture([1.0], [[1.0, -2.0]], [cov], df=5.0)
    x = np.random.default_rng(5).normal(size=(50, 2))
    np.testing.assert_allclose(
        mix.logpdf(x),
        multivariate_t(loc=[1.0, -2.0], shape=cov, df=5.0).logpdf(x),
        rtol=1e-10,
    )


def test_importance_sampling_recovers_an_isolated_mode(gp):
    """
    Given a 2-D target whose second coordinate has a 1.5% mode far from the
    main one (the black-hole lenses' situation), and a proposal that is a
    Student-t in the first coordinate times a grid density in the second,
    the importance estimate of the minor mode's share is 1.5% within its
    error, and the grid density integrates to one.
    """
    from scipy.stats import norm

    imp = gp["importance"]
    st = gp["stats"]
    rng = np.random.default_rng(6)

    def logp(x):
        a = norm.logpdf(x[:, 0])
        m = np.logaddexp(
            np.log(0.985) + norm.logpdf(x[:, 1], 0.0, 0.3),
            np.log(0.015) + norm.logpdf(x[:, 1], 6.0, 0.3),
        )
        return a + m

    edges = np.linspace(-3.0, 9.0, 241)
    centers = 0.5 * (edges[1:] + edges[:-1])
    prof = np.exp(logp(np.column_stack([np.zeros_like(centers), centers])))
    grid = imp.GridProposal(edges, 0.85 * prof / prof.sum() + 0.15 / prof.size)
    assert np.sum(np.exp(grid.logdens) * np.diff(edges)) == pytest.approx(1.0)
    prop = imp.ProductProposal(
        imp.TMixture([1.0], [[0.0]], [[[2.25]]]), grid, 1, 2
    )
    x = prop.sample(200_000, rng)
    lw = logp(x) - prop.logpdf(x)
    w = np.exp(lw - lw.max())
    f, se = st.fraction_with_se(x[:, 1] > 3.0, w)
    assert abs(f - 0.015) < 4 * se
    d = st.importance_diagnostics(lw)
    assert d["pareto_khat"] < 0.7 and d["ess_fraction"] > 0.3


# ----------------------------------------------------------------------
# slow: the prior itself, toward Baade's window
# ----------------------------------------------------------------------
@pytest.mark.slow
def test_prior_matches_genulens_baade_window(gp, tmp_path):
    """
    Given the prior importance-sampled toward Baade's window (2 x 500 pilot
    NUTS draws, three adaptive rounds of 1e5, 4e5 final draws), restricted to
    genulens's lens-in-front-of-the-source's-bin support,
    When its t_E, D_L, D_S, mu_rel and M_L quantiles are compared with the
    stored genulens (Dmax 20 kpc) run,
    Then every quantile q05 ... q95 agrees within the stated tolerance and
      the black-hole share of the rate within 4 combined sigma.
    """
    sys.path.insert(0, str(VALIDATION))
    try:
        compare = importlib.import_module("galactic_prior.compare")
        model = importlib.import_module("galactic_prior.model")
    finally:
        sys.path.remove(str(VALIDATION))
    st, sl, imp, refs = (
        gp["stats"],
        gp["sightlines"],
        gp["importance"],
        gp["refs"],
    )
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        prior = model.PriorModel(1.0, -3.9)
        cols, log_w, info = imp.run(
            prior,
            n_final=400_000,
            n_adapt=100_000,
            rounds=3,
            pilot_draws=500,
            pilot_tune=500,
            pilot_chains=2,
            cores=min(8, os.cpu_count() or 1),
            seed=7163,
            log=lambda *a: None,
        )
    finally:
        os.chdir(cwd)
    assert info["final"]["n_nan_logp"] == 0
    assert info["final"]["pareto_khat"] < 0.7, info["final"]
    cols["D_L"] = cols["D_L"] / 1000.0
    cols["D_S"] = cols["D_S"] / 1000.0
    w = np.exp(log_w - log_w[np.isfinite(log_w)].max())
    w[~np.isfinite(log_w)] = 0.0
    dd = sl.genulens_bin_width(1.0, -3.9, 20000.0)
    w = w * sl.genulens_support_mask(cols["D_L"] * 1e3, cols["D_S"] * 1e3, dd)
    port = compare.summarize(cols, w, n_boot=60)
    gen = json.loads((refs.REFS / "genulens_reference.json").read_text())
    g = gen["runs"]["baade"]["dmax20"]
    bad = []
    for qty in refs.QUANTITIES:
        for key in compare.KEYS:
            qp, qg = port[qty]["q"][key], g[qty]["q"][key]
            tol = 4 * np.hypot(port[qty]["se"][key], g[qty]["se"][key]) / qg
            tol += 0.01
            if abs(qp / qg - 1) > tol:
                bad.append(
                    f"{qty} {key}: {qp:.4g} vs {qg:.4g} (tol {tol:.3f})"
                )
    assert not bad, "; ".join(bad)
    assert st.kish_ess(w[w > 0]) > 2e4
    z = (port["bh_fraction"] - g["bh_fraction"]) / np.hypot(
        port["bh_fraction_se"], g["bh_fraction_se"]
    )
    assert abs(z) < 4, (port["bh_fraction"], g["bh_fraction"])
