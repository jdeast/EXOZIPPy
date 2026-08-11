"""
Tests for per-mode local evidence estimation via warp bridge sampling
(outputs/evidence.py).

The bridge estimator returns each mode's local log-evidence relative to a
Gaussian proposal fit in raw (unconstrained) space; softmax of the lnZ values
gives evidence-based mode weights that -- unlike occupancy weights -- recover
the true posterior-mass ratio even when the sampler did not mix between modes.
The estimator is self-diagnosing: when its relative-MSE diagnostic says the
proposal poorly supports the target (e.g. a bound pileup / fat raw-space tail)
it REFUSES the mode rather than emit a shaky number, and the report falls back
to occupancy weights.

Validation strategy:
  - bridge_lnZ is checked against an analytically known Gaussian evidence
    (constant C): all log-ratios equal log C, so the estimator must return
    log C with ~zero error.
  - estimate_mode_evidences is checked end-to-end on a real 2-mode mixture
    PyMC model whose per-mode local evidence is analytically w_k: with equal
    occupancy but unequal true weights, the recovered softmax weights track the
    true weights (not the 50/50 occupancy) within the propagated error bars.
  - Refusal is checked both at the core (a proposal far narrower than the
    target -> huge relative MSE) and end-to-end (a heavy-tailed Cauchy target
    whose Gaussian proposal cannot support the tails).

The "realistic scale" block near the bottom covers review item 1.13, whose
three defects all lived where the tests above never went -- |lnZ| of order
1e4..1e6, i.e. the raw model-logp scale of any actual dataset, and posterior
draws that are actually autocorrelated:

  (a) the relative-MSE diagnostic exponentiated unrecentered log-ratios, so
      anywhere |lnZ| > ~709 it returned nan/inf and EVERY mode was refused
      with the misleading reason "proposal poorly supports the target";
  (b) the fixed-point convergence test was ABSOLUTE at 1e-10, which at
      |lnZ| = 1e6 sits inside the float64 ULP spacing of lr itself, so ~25%
      of otherwise perfect estimates reported spurious non-convergence;
  (c) the IACT that inflates the posterior-side error term was measured on a
      series whose order had been destroyed by an unsorted rng.choice
      subsample, so it read ~1 whatever the sampler did and the error bar was
      understated by sqrt(tau) -- up to ~20x on a strongly correlated chain.
"""

import arviz as az
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest

from conftest import requires_fork
from exozippy.outputs.autocorr import iact
from exozippy.outputs.evidence import (
    EvidenceResult,
    _mode_draw_index,
    _segments,
    apply_evidence_weighting,
    bridge_lnZ,
    estimate_mode_evidences,
    softmax_weights,
)
from exozippy.outputs.latex import build_csv_output, build_latex_output
from exozippy.outputs.modes import ModeInfo, ModeReport, identify_modes


def _normal_logpdf(x, mu, sigma):
    return (
        -0.5 * ((x - mu) / sigma) ** 2
        - np.log(sigma)
        - 0.5 * np.log(2 * np.pi)
    )


# ----------------------------------------------------------------------
# pure bridge-sampling core
# ----------------------------------------------------------------------


def test_bridge_recovers_analytic_gaussian_constant():
    """
    Given a target that is a standard normal scaled by a known constant C
      (so log target - log proposal = log C for every draw), with a matching
      standard-normal proposal,
    When bridge_lnZ runs,
    Then it returns lnZ = log C essentially exactly, with ~zero error.
    """
    rng = np.random.default_rng(0)
    logC = 1.7
    x1 = rng.normal(0, 1, 4000)
    y2 = rng.normal(0, 1, 4000)
    # target unnormalized = C * N(x;0,1); proposal = N(x;0,1)
    l1 = logC + _normal_logpdf(x1, 0, 1) - _normal_logpdf(x1, 0, 1)
    l2 = logC + _normal_logpdf(y2, 0, 1) - _normal_logpdf(y2, 0, 1)

    lnZ, err, re2, converged = bridge_lnZ(l1, l2)

    assert converged
    assert lnZ == pytest.approx(logC, abs=1e-6)
    assert re2 == pytest.approx(0.0, abs=1e-9)


def test_bridge_recovers_shifted_gaussian_evidence():
    """
    Given a normal target N(0,1.3) with true evidence C and a (correct)
      normal proposal, using genuinely random draws,
    When bridge_lnZ runs,
    Then the recovered lnZ matches log C within a few sigma of its own error.
    """
    rng = np.random.default_rng(1)
    logC = -0.8
    sig = 1.3
    x1 = rng.normal(0, sig, 6000)  # posterior draws ~ target/C
    y2 = rng.normal(0, sig, 6000)  # proposal draws
    l1 = logC + _normal_logpdf(x1, 0, sig) - _normal_logpdf(x1, 0, sig)
    l2 = logC + _normal_logpdf(y2, 0, sig) - _normal_logpdf(y2, 0, sig)

    lnZ, err, re2, converged = bridge_lnZ(l1, l2)

    assert converged
    assert abs(lnZ - logC) <= max(1e-6, 5 * err)


def test_bridge_refuses_on_poor_overlap():
    """
    Given a target and a proposal that barely overlap (the proposal is shifted
      far off the target -- the raw-space signature of a misplaced proposal
      fit, as happens when draws pile against a bound and the fitted Gaussian
      sits off in a poorly-supported tail),
    When bridge_lnZ runs,
    Then the relative-MSE diagnostic is large, flagging an untrustworthy
      estimate.
    """
    rng = np.random.default_rng(2)
    x1 = rng.normal(0, 1.0, 4000)  # target ~ N(0,1)
    y2 = rng.normal(8.0, 1.0, 4000)  # proposal ~ N(8,1), barely overlaps
    l1 = _normal_logpdf(x1, 0, 1.0) - _normal_logpdf(x1, 8.0, 1.0)
    l2 = _normal_logpdf(y2, 0, 1.0) - _normal_logpdf(y2, 8.0, 1.0)

    lnZ, err, re2, converged = bridge_lnZ(l1, l2)

    assert re2 > 0.25


def test_softmax_weights_and_uncertainties():
    """
    Given two log-evidences with error bars,
    When softmax_weights runs,
    Then the weights sum to 1, order matches the lnZ order, and each weight
      carries a finite propagated uncertainty.
    """
    w, dw = softmax_weights([np.log(0.75), np.log(0.25)], [0.05, 0.08])

    assert w.sum() == pytest.approx(1.0)
    assert w[0] == pytest.approx(0.75, abs=1e-9)
    assert np.all(dw > 0) and np.all(np.isfinite(dw))


# ----------------------------------------------------------------------
# end-to-end: real 2-mode mixture model + synthetic trace
# ----------------------------------------------------------------------

N_CHAIN, N_DRAW = 4, 1000
N = N_CHAIN * N_DRAW


def _two_bump_mixture_model(mu0, mu1, w0, w1):
    """A 2-D model whose raw-space density is a two-Gaussian mixture with
    unit-covariance bumps and known mixture weights; the local evidence of
    bump k is exactly w_k when the bumps are well separated."""
    mu0 = np.asarray(mu0, float)
    mu1 = np.asarray(mu1, float)
    with pm.Model() as model:
        x = pm.Flat("x_raw", shape=2)
        lp0 = np.log(w0) - 0.5 * pt.sum((x - mu0) ** 2) - np.log(2 * np.pi)
        lp1 = np.log(w1) - 0.5 * pt.sum((x - mu1) ** 2) - np.log(2 * np.pi)
        pm.Potential("mix", pt.logaddexp(lp0, lp1))
    return model


def _mixture_lp(x, mu0, mu1, w0, w1):
    lp0 = np.log(w0) - 0.5 * ((x - mu0) ** 2).sum(-1) - np.log(2 * np.pi)
    lp1 = np.log(w1) - 0.5 * ((x - mu1) ** 2).sum(-1) - np.log(2 * np.pi)
    return np.logaddexp(lp0, lp1)


@requires_fork
def test_evidence_weights_recover_true_mixture_weights():
    """
    Given a well-separated two-mode mixture with equal draw occupancy (~50/50)
      but true mixture weights 0.75 / 0.25,
    When estimate_mode_evidences runs on the model and a matching synthetic
      trace,
    Then the softmax evidence weights recover the true 0.75 / 0.25 mass ratio
      (not the 50/50 occupancy) within the propagated error bars.
    """
    rng = np.random.default_rng(7)
    mu0, mu1 = np.array([0.0, 0.0]), np.array([8.0, 0.0])
    w0, w1 = 0.75, 0.25

    # Equal occupancy: half the draws from each bump, interleaved in every
    # chain so identify_modes labels both modes.
    origin = (rng.random(N) < 0.5).astype(int)
    centers = np.where(origin[:, None] == 0, mu0, mu1)
    x = rng.normal(centers, 1.0)
    lp = _mixture_lp(x, mu0, mu1, w0, w1)
    idata = az.from_dict(
        {
            "posterior": {"x_raw": x.reshape(N_CHAIN, N_DRAW, 2)},
            "sample_stats": {"lp": lp.reshape(N_CHAIN, N_DRAW)},
        }
    )
    report = identify_modes(idata)
    assert report.n_modes == 2

    model = _two_bump_mixture_model(mu0, mu1, w0, w1)
    results = estimate_mode_evidences(
        model, idata, report, max_posterior_draws=800, n_proposal=800
    )

    assert all(not r.refused for r in results)
    # Map each mode index to its bump via the mode's raw x[0] center.
    expected = {}
    for m in report.modes:
        c0 = m.center.get("x_raw[0]", np.nan)
        expected[m.index] = w0 if abs(c0 - 0.0) < abs(c0 - 8.0) else w1
    lnZ = np.array([r.lnZ for r in results])
    weights, dweights = softmax_weights(
        lnZ, np.array([r.lnZ_err for r in results])
    )
    for r in results:
        exp = expected[r.mode] / (w0 + w1)
        assert abs(weights[r.mode] - exp) <= max(0.05, 5 * dweights[r.mode])


def test_apply_evidence_weighting_replaces_occupancy():
    """
    Given a 2-mode report whose occupancy weights are ~50/50 and successful
      bridge results implying 0.75 / 0.25,
    When apply_evidence_weighting runs,
    Then the mode weights and provenance are replaced by the evidence values
      and each mode carries a propagated weight uncertainty.
    """
    report = _fake_two_mode_report(w0_occ=0.5, w1_occ=0.5)
    results = [
        EvidenceResult(0, np.log(0.75), 0.03, 0.001, 800, 800, False),
        EvidenceResult(1, np.log(0.25), 0.05, 0.002, 800, 800, False),
    ]

    applied = apply_evidence_weighting(report, results)

    assert applied
    assert report.provenance.startswith("evidence (bridge sampling")
    assert report.modes[0].weight == pytest.approx(0.75, abs=1e-6)
    assert report.modes[1].weight == pytest.approx(0.25, abs=1e-6)
    assert report.modes[0].weight_err > 0


def test_apply_evidence_weighting_falls_back_on_refusal():
    """
    Given a 2-mode report where one mode's bridge estimate is refused,
    When apply_evidence_weighting runs,
    Then it does NOT replace the occupancy weights (a single refusal
      invalidates the softmax set) and records the refusal in the provenance.
    """
    report = _fake_two_mode_report(w0_occ=0.6, w1_occ=0.4)
    results = [
        EvidenceResult(0, np.log(0.7), 0.03, 0.001, 800, 800, False),
        EvidenceResult(
            1,
            np.nan,
            np.inf,
            np.inf,
            800,
            800,
            True,
            "relative-MSE diagnostic re2=1.2 exceeds 0.25",
        ),
    ]

    applied = apply_evidence_weighting(report, results)

    assert not applied
    assert report.modes[0].weight == pytest.approx(0.6)  # occupancy kept
    assert "refused" in report.provenance


@requires_fork
def test_evidence_refuses_heavy_tailed_mode():
    """
    Given a single mode whose raw-space target is heavy-tailed (Cauchy) so a
      Gaussian proposal cannot support the tails (the raw-space signature of a
      bound pileup),
    When estimate_mode_evidences runs,
    Then it refuses the mode rather than reporting a number.
    """
    rng = np.random.default_rng(3)
    x = rng.standard_cauchy(N)
    # keep values finite/representable but retain the fat tail
    x = np.clip(x, -1e6, 1e6)
    lp = -np.log1p(x**2)
    idata = az.from_dict(
        {
            "posterior": {"x_raw": x.reshape(N_CHAIN, N_DRAW)},
            "sample_stats": {"lp": lp.reshape(N_CHAIN, N_DRAW)},
        }
    )
    report = _fake_one_mode_report(idata)

    with pm.Model() as model:
        xt = pm.Flat("x_raw")
        pm.Potential("cauchy", -pt.log1p(xt**2))

    results = estimate_mode_evidences(
        model, idata, report, max_posterior_draws=800, n_proposal=800
    )

    assert len(results) == 1
    assert results[0].refused
    assert not apply_evidence_weighting(report, results)


# ----------------------------------------------------------------------
# realistic model-logp scale (review 1.13)
# ----------------------------------------------------------------------


def _gaussian_bridge_inputs(logC, sigma_target, n, seed, shift=0.0):
    """Log-ratios for a N(shift, sigma_target) target of evidence exp(logC)
    against a standard-normal proposal.  logC enters as a pure additive
    offset, so the analytic answer is lnZ = logC at any scale."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(shift, sigma_target, n)
    y2 = rng.normal(0.0, 1.0, n)
    l1 = (
        logC
        + _normal_logpdf(x1, shift, sigma_target)
        - _normal_logpdf(x1, 0.0, 1.0)
    )
    l2 = (
        logC
        + _normal_logpdf(y2, shift, sigma_target)
        - _normal_logpdf(y2, 0.0, 1.0)
    )
    return l1, l2


def _unrecentered_re2(l1, l2, lr):
    """The pre-fix relative-MSE expression, verbatim, for the equivalence
    check.  Exponentiates l1/l2/lr directly, which is exactly why it
    overflows once the log-ratios reach real model-logp scale."""
    n1, n2 = l1.size, l2.size
    s1, s2 = n1 / (n1 + n2), n2 / (n1 + n2)
    # The overflow IS the defect being reproduced, so silence the warnings.
    with np.errstate(over="ignore", invalid="ignore"):
        r = np.exp(lr)
        f1 = np.exp(l2) / (s1 * np.exp(l2) + s2 * r)
        f2 = 1.0 / (s1 * np.exp(l1) + s2 * r)
        term1 = (f1.var() / f1.mean() ** 2) / n2
        term2 = iact(f2) * (f2.var() / f2.mean() ** 2) / n1
    return term1 + term2


@pytest.mark.parametrize("logC", [7.0e2, 1.0e3, 1.0e4, 1.0e5, 1.0e6])
def test_diagnostic_is_finite_at_realistic_logp_scale(logC):
    """
    Given bridge inputs at the raw model-logp scale of a real dataset
      (|lnZ| from 700 to 1e6 nats), where exp(lnZ) is not representable,
    When bridge_lnZ runs,
    Then it returns a finite relative-MSE diagnostic and error bar (the
      unrecentered expression returns nan/inf here, which refused every mode
      on every real fit).
    """
    l1, l2 = _gaussian_bridge_inputs(logC, 1.3, 3000, seed=0)

    lnZ, err, re2, converged = bridge_lnZ(l1, l2)

    assert converged
    assert abs(lnZ - logC) <= max(1e-6, 5 * err)  # Monte Carlo error only
    assert np.isfinite(re2) and re2 > 0
    assert np.isfinite(err) and err > 0
    assert re2 < 0.25  # a well-matched proposal must NOT be refused
    # the expression this replaced could not even be evaluated here
    assert not np.isfinite(_unrecentered_re2(l1, l2, lnZ))


def test_recentered_diagnostic_equals_the_old_one_where_the_old_one_worked():
    """
    Given bridge inputs across the range of |lnZ| where the unrecentered
      relative-MSE expression was still representable in float64,
    When bridge_lnZ's recentered diagnostic is compared with it,
    Then the two agree to float precision -- only the scale-free ratios
      var/mean^2 enter re2, so dividing f1 through by exp(l2) and f2 by
      exp(lr) cannot change the answer.
    """
    worst_rel = 0.0
    n_compared = 0
    for logC in (-5.0, -0.5, 0.0, 2.0, 50.0, 200.0, 500.0):
        for sigma in (1.05, 1.3, 2.0, 3.0):
            l1, l2 = _gaussian_bridge_inputs(
                logC, sigma, 3000, seed=int(abs(logC) + 100 * sigma)
            )
            lnZ, _err, re2, converged = bridge_lnZ(l1, l2)
            old = _unrecentered_re2(l1, l2, lnZ)
            assert converged
            # Skip the degenerate target == proposal case: there re2 is
            # exactly zero and both forms return ~1e-32 of pure round-off,
            # so a RELATIVE comparison of two noise floors is meaningless.
            if not np.isfinite(old) or old <= 1e-20:
                assert abs(re2 - old) < 1e-25 if np.isfinite(old) else True
                continue
            n_compared += 1
            worst_rel = max(worst_rel, abs(re2 - old) / old)

    assert n_compared >= 20
    assert worst_rel < 1e-12, f"recentering changed re2 by {worst_rel:.3e}"


@pytest.mark.parametrize("logC", [1.0e4, 1.0e6, 1.0e8])
def test_relative_tolerance_converges_at_large_lnZ(logC):
    """
    Given a well-matched proposal at |lnZ| far above the float64 ULP scale of
      the absolute 1e-10 tolerance this replaced,
    When bridge_lnZ runs over many independent realizations,
    Then every one of them converges (the absolute test failed ~25% of them
      at |lnZ| >= 1e6, purely because the fixed point cannot be pinned to
      1e-10 when one ULP of lr is already 1.2e-10).
    """
    failures = [
        seed
        for seed in range(50)
        if not bridge_lnZ(
            *_gaussian_bridge_inputs(logC, 2.0, 2000, seed=seed)
        )[3]
    ]

    assert failures == []


@pytest.mark.parametrize("logC", [0.0, 1.0e6])
def test_relative_tolerance_still_reports_genuine_non_convergence(logC):
    """
    Given a proposal that barely overlaps the target, so the optimal-bridge
      fixed point never settles,
    When bridge_lnZ runs to maxiter at both zero and realistic |lnZ|,
    Then it reports converged=False -- the relative tolerance loosens the
      test by the float64 resolution of lr, not by anything a genuinely
      unconverged iteration (still moving O(0.01-1) nat per step) could hide
      under.
    """
    l1, l2 = _gaussian_bridge_inputs(logC, 1.0, 3000, seed=1, shift=0.0)
    rng = np.random.default_rng(1)
    y2 = rng.normal(9.0, 1.0, 3000)
    l2 = logC + _normal_logpdf(y2, 0.0, 1.0) - _normal_logpdf(y2, 9.0, 1.0)
    x1 = rng.normal(0.0, 1.0, 3000)
    l1 = logC + _normal_logpdf(x1, 0.0, 1.0) - _normal_logpdf(x1, 9.0, 1.0)

    lnZ, err, re2, converged = bridge_lnZ(l1, l2)

    assert not converged
    assert re2 > 0.25  # and it is refused on the diagnostic too


def _ar1(n, phi, rng):
    """AR(1) series with integrated autocorrelation time (1+phi)/(1-phi)."""
    x = np.empty(n)
    x[0] = rng.normal(0.0, 1.0 / np.sqrt(1.0 - phi**2))
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.normal()
    return x


@pytest.mark.parametrize("phi,true_tau", [(0.9, 19.0), (0.98, 99.0)])
def test_iact_survives_the_real_subsampling_path(phi, true_tau):
    """
    Given an AR(1) series of KNOWN integrated autocorrelation time, laid out
      exactly as estimate_mode_evidences lays out a mode's posterior draws
      (chain-major rows of the (chain, draw) posterior matrix),
    When it is reduced to max_posterior_draws through the real index helper
      and measured with the shared IACT estimator,
    Then the IACT reflects the autocorrelation of the retained series
      (roughly true_tau/stride), whereas the unsorted rng.choice subsample it
      replaced reads ~1 and understates the bridge error bar by sqrt(tau).
    """
    rng = np.random.default_rng(11)
    n_chain, n_draw, max_draws = 4, 5000, 4000
    chains = np.array([_ar1(n_draw, phi, rng) for _ in range(n_chain)])
    flat = chains.reshape(-1)  # (chain, draw) row-major, as _posterior_matrix
    labels = np.zeros(flat.size, dtype=int)

    index, chain_lengths = _mode_draw_index(
        labels, 0, n_chain, n_draw, max_draws
    )
    kept = flat[index]
    tau_new = iact(_segments(kept, np.ones(kept.size, bool), chain_lengths))

    shuffled = np.random.default_rng(1).choice(
        flat.size, max_draws, replace=False
    )
    tau_old = iact(flat[shuffled])

    stride = int(np.ceil(flat.size / max_draws))
    tau_expected = (1 + phi**stride) / (1 - phi**stride)

    assert index.size <= max_draws + n_chain
    assert sum(chain_lengths) == index.size
    # order preserved: indices ascend within each chain's block
    start = 0
    for length in chain_lengths:
        block = index[start : start + length]
        assert np.all(np.diff(block) > 0)
        start += length
    assert tau_old < 1.5  # the defect: order destroyed -> tau reads ~1
    assert tau_new > 0.4 * tau_expected
    assert tau_new > 2.0 * tau_old


def test_iact_recovers_a_known_ar1_autocorrelation():
    """
    Given AR(1) series with analytically known IACT (1+phi)/(1-phi),
    When the shared estimator measures them in order,
    Then it recovers the true value to within the estimator's own noise.
    """
    rng = np.random.default_rng(3)
    for phi in (0.5, 0.9, 0.98):
        true_tau = (1 + phi) / (1 - phi)
        chains = np.array([_ar1(40000, phi, rng) for _ in range(4)])

        tau = iact(chains)

        assert 0.7 * true_tau < tau < 1.5 * true_tau, (phi, tau, true_tau)


def test_bridge_error_bar_grows_with_posterior_autocorrelation():
    """
    Given two bridge problems with identical marginal log-ratios but one
      drawn i.i.d. and one drawn as a strongly autocorrelated series,
    When bridge_lnZ measures each with its chain structure declared,
    Then the autocorrelated one carries the larger error bar -- the whole
      point of the IACT term, and exactly what the destroyed ordering hid.
    """
    rng = np.random.default_rng(21)
    n = 4000
    y2 = rng.normal(0.0, 1.0, n)
    l2 = _normal_logpdf(y2, 0.0, 1.3) - _normal_logpdf(y2, 0.0, 1.0)

    iid = rng.normal(0.0, 1.3, n)
    correlated = 1.3 * _ar1(n, 0.98, rng)
    l1_iid = _normal_logpdf(iid, 0.0, 1.3) - _normal_logpdf(iid, 0.0, 1.0)
    l1_cor = _normal_logpdf(correlated, 0.0, 1.3) - _normal_logpdf(
        correlated, 0.0, 1.0
    )

    _z, err_iid, _re2, _c = bridge_lnZ(l1_iid, l2, l1_chains=[n])
    _z, err_cor, _re2, _c = bridge_lnZ(l1_cor, l2, l1_chains=[n])

    assert err_cor > 3.0 * err_iid


# ----------------------------------------------------------------------
# output wiring smoke test
# ----------------------------------------------------------------------


class _StubSystem:
    name = "toy"

    def get_all_components(self):
        return []


def test_evidence_provenance_replaces_occupancy_in_output(tmp_path):
    """
    Given a 2-mode report re-weighted by evidence,
    When build_latex_output renders it,
    Then the evidence provenance string (not the occupancy one) appears in the
      table comments and the mode-weight macros carry the evidence weights.
    """
    report = _fake_two_mode_report(w0_occ=0.5, w1_occ=0.5)
    results = [
        EvidenceResult(0, np.log(0.75), 0.03, 0.001, 800, 800, False),
        EvidenceResult(1, np.log(0.25), 0.05, 0.002, 800, 800, False),
    ]
    assert apply_evidence_weighting(report, results)

    var_file = tmp_path / "defs.tex"
    tmpl_file = tmp_path / "table.tex"
    build_latex_output(
        _StubSystem(),
        var_filename=str(var_file),
        template_filename=str(tmpl_file),
        caption="toy",
        mode_report=report,
    )

    defs = var_file.read_text()
    tmpl = tmpl_file.read_text()
    assert "0.750" in defs and "0.250" in defs
    assert "evidence (bridge sampling" in tmpl
    assert "occupancy" not in tmpl


def test_weight_err_reaches_text_latex_and_csv(tmp_path):
    """
    Given a 2-mode report re-weighted by evidence, so every mode carries a
      propagated weight uncertainty,
    When the text report, the LaTeX definitions/table and the CSV are built,
    Then all three carry the uncertainty next to the weight -- it used to be
      computed and then printed nowhere at all.
    """
    report = _fake_two_mode_report(w0_occ=0.5, w1_occ=0.5)
    results = [
        EvidenceResult(0, np.log(0.75), 0.03, 0.001, 800, 800, False),
        EvidenceResult(1, np.log(0.25), 0.05, 0.002, 800, 800, False),
    ]
    assert apply_evidence_weighting(report, results)
    dw0 = report.modes[0].weight_err
    assert np.isfinite(dw0) and dw0 > 0

    text = report.to_text()
    var_file = tmp_path / "defs.tex"
    tmpl_file = tmp_path / "table.tex"
    csv_file = tmp_path / "results.csv"
    build_latex_output(
        _StubSystem(),
        var_filename=str(var_file),
        template_filename=str(tmpl_file),
        mode_report=report,
    )
    build_csv_output(_StubSystem(), str(csv_file), mode_report=report)

    assert f"0.7500 +/- {dw0:.4f}" in text
    defs = var_file.read_text()
    assert r"\ezmodeweighterrone" in defs
    assert f"{dw0:.3f}" in defs
    assert r"\ezmodeweighterrone" in tmpl_file.read_text()
    header = csv_file.read_text().splitlines()[0]
    assert "weight_err" in header


def test_evidence_weighting_updates_the_attached_mode_attrs():
    """
    Given an idata carrying identify_modes' occupancy weights in
      posterior['mode'].attrs,
    When apply_evidence_weighting replaces the weights and is handed the idata,
    Then the attrs hold the EVIDENCE weights and provenance, not the stale
      occupancy values the DataArray was built with, and the occupancy values
      remain available separately for the cross-check.
    """
    rng = np.random.default_rng(17)
    x = np.concatenate(
        [rng.normal(0.0, 1.0, N // 2), rng.normal(8.0, 1.0, N // 2)]
    )
    lp = _mixture_lp(x[:, None], np.array([0.0]), np.array([8.0]), 0.5, 0.5)
    idata = az.from_dict(
        {
            "posterior": {"x_raw": x.reshape(N_CHAIN, N_DRAW)},
            "sample_stats": {"lp": lp.reshape(N_CHAIN, N_DRAW)},
        }
    )
    report = identify_modes(idata)
    assert report.n_modes == 2
    occupancy = list(idata.posterior["mode"].attrs["weights"])
    results = [
        EvidenceResult(0, np.log(0.9), 0.03, 0.001, 800, 800, False),
        EvidenceResult(1, np.log(0.1), 0.05, 0.002, 800, 800, False),
    ]

    assert apply_evidence_weighting(report, results, idata=idata)

    attrs = idata.posterior["mode"].attrs
    assert attrs["weights"] == pytest.approx([0.9, 0.1], abs=1e-9)
    assert attrs["weights"] != pytest.approx(occupancy, abs=1e-3)
    assert "evidence (bridge sampling" in attrs["provenance"]
    assert attrs["weight_errs"][0] > 0
    # occupancy survives for the cross-check
    assert attrs["occupancy_weights"] == pytest.approx(occupancy, abs=1e-9)


# ----------------------------------------------------------------------
# helpers to build synthetic ModeReports
# ----------------------------------------------------------------------


def _mode_info(index, weight, n=800):
    return ModeInfo(
        index=index,
        weight=weight,
        n_draws=n,
        lp_med=0.0,
        lp_max=0.0,
        delta_lp_max=0.0,
        per_chain_weight=np.array([weight]),
    )


def _fake_two_mode_report(w0_occ, w1_occ):
    labels = np.zeros((N_CHAIN, N_DRAW), dtype=int)
    return ModeReport(
        labels=labels,
        modes=[_mode_info(0, w0_occ), _mode_info(1, w1_occ)],
        n_valid=N,
        n_invalid=0,
        n_unassigned=0,
        provenance="occupancy (UNRELIABLE: chains do not mix between modes)",
        weights_reliable=False,
        n_transitions=0,
        feature_vars=["x_raw"],
        notes=[],
    )


def _fake_one_mode_report(idata):
    labels = np.zeros(
        (idata.posterior.sizes["chain"], idata.posterior.sizes["draw"]),
        dtype=int,
    )
    rep = ModeReport(
        labels=labels,
        modes=[_mode_info(0, 1.0, n=labels.size)],
        n_valid=labels.size,
        n_invalid=0,
        n_unassigned=0,
        provenance="unimodal",
        weights_reliable=True,
        n_transitions=0,
        feature_vars=["x_raw"],
        notes=[],
    )
    rep.attach(idata)
    return rep
