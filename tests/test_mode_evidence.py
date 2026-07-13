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
"""

import numpy as np
import arviz as az
import pytest

import pymc as pm
import pytensor.tensor as pt

from exozippy.outputs.evidence import (
    bridge_lnZ, softmax_weights, estimate_mode_evidences,
    apply_evidence_weighting, EvidenceResult,
)
from exozippy.outputs.modes import identify_modes, ModeInfo, ModeReport
from exozippy.outputs.latex import build_latex_output


def _normal_logpdf(x, mu, sigma):
    return (-0.5 * ((x - mu) / sigma) ** 2
            - np.log(sigma) - 0.5 * np.log(2 * np.pi))


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
    x1 = rng.normal(0, sig, 6000)              # posterior draws ~ target/C
    y2 = rng.normal(0, sig, 6000)              # proposal draws
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
    x1 = rng.normal(0, 1.0, 4000)              # target ~ N(0,1)
    y2 = rng.normal(8.0, 1.0, 4000)            # proposal ~ N(8,1), barely overlaps
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
    idata = az.from_dict({
        "posterior": {"x_raw": x.reshape(N_CHAIN, N_DRAW, 2)},
        "sample_stats": {"lp": lp.reshape(N_CHAIN, N_DRAW)},
    })
    report = identify_modes(idata)
    assert report.n_modes == 2

    model = _two_bump_mixture_model(mu0, mu1, w0, w1)
    results = estimate_mode_evidences(model, idata, report,
                                      max_posterior_draws=800, n_proposal=800)

    assert all(not r.refused for r in results)
    # Map each mode index to its bump via the mode's raw x[0] center.
    expected = {}
    for m in report.modes:
        c0 = m.center.get("x_raw[0]", np.nan)
        expected[m.index] = w0 if abs(c0 - 0.0) < abs(c0 - 8.0) else w1
    lnZ = np.array([r.lnZ for r in results])
    weights, dweights = softmax_weights(lnZ, np.array([r.lnZ_err for r in results]))
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
    results = [EvidenceResult(0, np.log(0.75), 0.03, 0.001, 800, 800, False),
               EvidenceResult(1, np.log(0.25), 0.05, 0.002, 800, 800, False)]

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
    results = [EvidenceResult(0, np.log(0.7), 0.03, 0.001, 800, 800, False),
               EvidenceResult(1, np.nan, np.inf, np.inf, 800, 800, True,
                              "relative-MSE diagnostic re2=1.2 exceeds 0.25")]

    applied = apply_evidence_weighting(report, results)

    assert not applied
    assert report.modes[0].weight == pytest.approx(0.6)  # occupancy kept
    assert "refused" in report.provenance


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
    lp = -np.log1p(x ** 2)
    idata = az.from_dict({
        "posterior": {"x_raw": x.reshape(N_CHAIN, N_DRAW)},
        "sample_stats": {"lp": lp.reshape(N_CHAIN, N_DRAW)},
    })
    report = _fake_one_mode_report(idata)

    with pm.Model() as model:
        xt = pm.Flat("x_raw")
        pm.Potential("cauchy", -pt.log1p(xt ** 2))

    results = estimate_mode_evidences(model, idata, report,
                                      max_posterior_draws=800, n_proposal=800)

    assert len(results) == 1
    assert results[0].refused
    assert not apply_evidence_weighting(report, results)


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
    results = [EvidenceResult(0, np.log(0.75), 0.03, 0.001, 800, 800, False),
               EvidenceResult(1, np.log(0.25), 0.05, 0.002, 800, 800, False)]
    assert apply_evidence_weighting(report, results)

    var_file = tmp_path / "defs.tex"
    tmpl_file = tmp_path / "table.tex"
    build_latex_output(_StubSystem(), var_filename=str(var_file),
                       template_filename=str(tmpl_file), caption="toy",
                       mode_report=report)

    defs = var_file.read_text()
    tmpl = tmpl_file.read_text()
    assert "0.750" in defs and "0.250" in defs
    assert "evidence (bridge sampling" in tmpl
    assert "occupancy" not in tmpl


# ----------------------------------------------------------------------
# helpers to build synthetic ModeReports
# ----------------------------------------------------------------------

def _mode_info(index, weight, n=800):
    return ModeInfo(index=index, weight=weight, n_draws=n, lp_med=0.0,
                    lp_max=0.0, delta_lp_max=0.0,
                    per_chain_weight=np.array([weight]))


def _fake_two_mode_report(w0_occ, w1_occ):
    labels = np.zeros((N_CHAIN, N_DRAW), dtype=int)
    return ModeReport(
        labels=labels,
        modes=[_mode_info(0, w0_occ), _mode_info(1, w1_occ)],
        n_valid=N, n_invalid=0, n_unassigned=0,
        provenance="occupancy (UNRELIABLE: chains do not mix between modes)",
        weights_reliable=False, n_transitions=0,
        feature_vars=["x_raw"], notes=[])


def _fake_one_mode_report(idata):
    labels = np.zeros(
        (idata.posterior.sizes["chain"], idata.posterior.sizes["draw"]),
        dtype=int)
    rep = ModeReport(
        labels=labels, modes=[_mode_info(0, 1.0, n=labels.size)],
        n_valid=labels.size, n_invalid=0, n_unassigned=0,
        provenance="unimodal", weights_reliable=True, n_transitions=0,
        feature_vars=["x_raw"], notes=[])
    rep.attach(idata)
    return rep
