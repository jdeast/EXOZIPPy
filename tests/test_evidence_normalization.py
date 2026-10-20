"""
The bridge-sampling lnZ is the NORMALIZED log evidence (outputs/evidence.py,
"Normalization"; Parameter.evidence_log_norm_terms; System.evidence_log_norm).

The model's sampled logp is a prior density only up to a constant: on every
bounded (logit-path) element section C of build_pymc leaves the raw density
at q(1-q)/sqrt(2 pi), which integrates to 1/(sqrt(2 pi) s) with s the
element's whitening scale, and a sigma on a bounded element is an
unnormalized exp(-z^2/2) on top of the uniform.  The constant is computed,
not built into the model, so the posterior and every start logp are
untouched; these tests pin that it is the RIGHT constant:

  (a) prior only -> normalized lnZ = 0, on a model that has every term kind
      the constant covers: plain uniforms at different whitening scales, a
      truncated normal whose truncation mass is far from 1, and the
      xalpha/yalpha circular pair;
  (b) the same model under two different whitening scales gives the same
      normalized lnZ (while the unnormalized one moves by sum ln m);
  (c) analytic evidences: a 1-D Gaussian likelihood on a uniform prior, and
      a von Mises likelihood on the angle of the circular pair, whose
      evidence is I0(kappa) only if the embedding is normalized correctly.

The posterior draws are drawn EXACTLY (from the prior, or from the analytic
posterior) and mapped to raw space with the build's own transform, so the
only estimator in play is the bridge itself.
"""

import arviz as az
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest

from conftest import requires_fork
from exozippy.components.parameter import Parameter
from exozippy.outputs.evidence import estimate_mode_evidences
from exozippy.outputs.modes import ModeInfo, ModeReport
from exozippy.system import System

N_CHAIN, N_DRAW = 4, 1000
N = N_CHAIN * N_DRAW


class _StubSystem:
    """Just enough System for System.evidence_log_norm."""

    def __init__(self, params):
        self._params = params

    def get_all_parameters(self):
        return self._params


def _params():
    """Every kind of prior term the normalization covers, one Parameter
    each.  c's truncation mass is Phi(6) - Phi(-2/3) ~ 0.75, so a missing
    ln Z would show as ~0.29 nat."""
    return {
        "a": Parameter(
            label="toy.a", initval=2.0, lower=0.0, upper=10.0, init_scale=1.0
        ),
        "b": Parameter(
            label="toy.b", initval=-1.0, lower=-5.0, upper=5.0, init_scale=0.3
        ),
        "c": Parameter(
            label="toy.c",
            initval=3.0,
            lower=0.0,
            upper=20.0,
            mu=2.0,
            sigma=3.0,
            init_scale=0.5,
        ),
        "x": Parameter(
            label="toy.xalpha",
            initval=1.0,
            lower=-100.0,
            upper=100.0,
            mu=0.0,
            sigma=1.0,
            init_scale=0.01,
        ),
        "y": Parameter(
            label="toy.yalpha",
            initval=0.0,
            lower=-100.0,
            upper=100.0,
            mu=0.0,
            sigma=1.0,
            init_scale=0.01,
        ),
    }


def _build(params, likelihood=None):
    with pm.Model() as model:
        vals = {k: p.build_pymc() for k, p in params.items()}
        if likelihood is not None:
            pm.Potential("like", likelihood(vals))
    return model


def _rescale(params, multipliers):
    for k, m in multipliers.items():
        params[k].set_whitening(np.array([m]))


def _prior_draws(rng, n):
    """Exact draws of the physical prior of _params()."""
    c = rng.normal(2.0, 3.0, 4 * n)
    c = c[(c >= 0.0) & (c <= 20.0)][:n]
    assert c.size == n
    return {
        "a": rng.uniform(0.0, 10.0, n),
        "b": rng.uniform(-5.0, 5.0, n),
        "c": c,
        "x": rng.normal(0.0, 1.0, n),
        "y": rng.normal(0.0, 1.0, n),
    }


def _idata(params, phys):
    """Map physical draws to raw through each build's OWN transform."""
    post = {}
    for k, v in phys.items():
        raw = params[k].element_raw_from_phys(0, v)
        post[f"{params[k].label}_raw"] = raw.reshape(N_CHAIN, N_DRAW)
    return az.from_dict(
        {
            "posterior": post,
            "sample_stats": {"lp": np.zeros((N_CHAIN, N_DRAW))},
        }
    )


def _one_mode_report(idata):
    labels = np.zeros((N_CHAIN, N_DRAW), dtype=int)
    rep = ModeReport(
        labels=labels,
        modes=[
            ModeInfo(
                index=0,
                weight=1.0,
                n_draws=labels.size,
                lp_med=0.0,
                lp_max=0.0,
                delta_lp_max=0.0,
                per_chain_weight=np.array([1.0]),
            )
        ],
        n_valid=labels.size,
        n_invalid=0,
        n_unassigned=0,
        provenance="unimodal",
        weights_reliable=True,
        n_transitions=0,
        feature_vars=list(idata.posterior.data_vars),
        notes=[],
    )
    rep.attach(idata)
    return rep


def _normalized_lnZ(model, params, idata):
    log_norm = System.evidence_log_norm(_StubSystem(list(params.values())))
    (r,) = estimate_mode_evidences(
        model,
        idata,
        _one_mode_report(idata),
        max_posterior_draws=N,
        n_proposal=N,
        log_norm=log_norm,
    )
    assert not r.refused, r.reason
    assert r.status["log_norm"] == pytest.approx(log_norm)
    return r


def _tol(*results):
    return max(
        0.03, 5.0 * float(np.hypot.reduce([r.lnZ_err for r in results]))
    )


def test_the_constant_tracks_the_whitening_scale_exactly():
    """
    Given a bounded uniform element,
    When set_whitening multiplies its logit scale by m,
    Then its normalization constant moves by exactly ln m, and its value is
      ln s + ln sqrt(2 pi) with s the CURRENT scale_logit.
    """
    p = Parameter(
        label="toy.a", initval=2.0, lower=0.0, upper=10.0, init_scale=1.0
    )
    with pm.Model():
        p.build_pymc()
    before = p.evidence_log_norm()
    s0 = float(p._whiten_state["sv_scale_logits"].get_value()[0])

    p.set_whitening(np.array([37.0]))

    assert before == pytest.approx(np.log(s0) + 0.5 * np.log(2 * np.pi))
    assert p.evidence_log_norm() - before == pytest.approx(np.log(37.0))
    assert [k for _i, k, _v in p.evidence_log_norm_terms()] == ["uniform"]


def test_a_gaussian_on_a_bounded_element_owes_its_truncated_normal_norm():
    """
    Given a bounded element with mu/sigma (a truncated normal prior),
    When its terms are listed,
    Then it owes the uniform term AND ln span - ln(sigma sqrt(2 pi)) - ln Z,
      Z the truncation mass.
    """
    from scipy.stats import norm

    p = _params()["c"]
    with pm.Model():
        p.build_pymc()

    terms = {k: v for _i, k, v in p.evidence_log_norm_terms()}

    z = norm.cdf((20.0 - 2.0) / 3.0) - norm.cdf((0.0 - 2.0) / 3.0)
    expect = np.log(20.0) - np.log(3.0 * np.sqrt(2 * np.pi)) - np.log(z)
    assert set(terms) == {"uniform", "truncated_normal"}
    assert terms["truncated_normal"] == pytest.approx(expect, rel=1e-12)


@requires_fork
def test_prior_only_model_has_zero_normalized_log_evidence():
    """
    Given a likelihood-free model of two uniforms, a truncated normal and the
      xalpha/yalpha pair, whitened to five very different scales,
    When the bridge estimates its evidence from exact prior draws,
    Then the NORMALIZED lnZ is 0 (a normalized prior integrates to one), while
      the constant itself is far from 0 -- so the constant is what is being
      tested.
    """
    rng = np.random.default_rng(11)
    params = _params()
    model = _build(params)
    _rescale(params, {"a": 0.05, "b": 20.0, "c": 3.0, "x": 50.0, "y": 0.2})
    idata = _idata(params, _prior_draws(rng, N))

    r = _normalized_lnZ(model, params, idata)

    assert abs(r.lnZ) < _tol(r)
    assert abs(r.status["log_norm"]) > 1.0


@requires_fork
def test_normalized_log_evidence_is_invariant_to_the_whitening():
    """
    Given the same prior-only model built twice, one copy rescaled by
      multipliers spanning four decades,
    When each is bridged from the SAME physical draws,
    Then the normalized lnZ agree within the bridge error, while the
      unnormalized ones differ by -sum(ln m) -- the run-to-run drift the
      constant removes.
    """
    rng = np.random.default_rng(12)
    phys = _prior_draws(rng, N)
    p1, p2 = _params(), _params()
    m1, m2 = _build(p1), _build(p2)
    mult = {"a": 0.01, "b": 100.0, "c": 0.1, "x": 30.0, "y": 3.0}
    _rescale(p2, mult)

    r1 = _normalized_lnZ(m1, p1, _idata(p1, phys))
    r2 = _normalized_lnZ(m2, p2, _idata(p2, phys))

    assert abs(r1.lnZ - r2.lnZ) < _tol(r1, r2)
    raw_shift = r2.status["lnZ_bridge"] - r1.status["lnZ_bridge"]
    assert raw_shift == pytest.approx(
        -np.sum(np.log(list(mult.values()))), abs=_tol(r1, r2)
    )


@requires_fork
def test_gaussian_likelihood_on_a_uniform_prior_recovers_the_analytic_evidence():
    """
    Given x ~ U(0, 10) and a normalized Gaussian likelihood N(4 | x, 0.5),
      at a non-trivial whitening scale,
    When the bridge runs on exact posterior draws,
    Then the normalized lnZ is ln[(Phi(12) - Phi(-8)) / 10] = -ln 10.
    """
    rng = np.random.default_rng(13)
    params = {"a": _params()["a"]}

    def like(v):
        return -0.5 * ((v["a"] - 4.0) / 0.5) ** 2 - np.log(
            0.5 * np.sqrt(2 * np.pi)
        )

    model = _build(params, like)
    _rescale(params, {"a": 0.2})
    idata = _idata(params, {"a": rng.normal(4.0, 0.5, N)})

    r = _normalized_lnZ(model, params, idata)

    assert abs(r.lnZ - (-np.log(10.0))) < _tol(r)


@requires_fork
def test_von_mises_likelihood_on_the_circular_pair_gives_i0_kappa():
    """
    Given the xalpha/yalpha pair (angle alpha = arctan2(y, x) uniform) and a
      likelihood exp(kappa cos(alpha - alpha0)) that reads only the angle,
    When the bridge runs on exact posterior draws,
    Then the normalized lnZ is ln I0(kappa) -- the evidence of a von Mises
      likelihood under U(0, 2 pi) -- so the raw-space volume and the
      embedding's radial direction are accounted for exactly.
    """
    from scipy.special import i0

    kappa, a0 = 2.0, 0.7
    rng = np.random.default_rng(14)
    p = _params()
    params = {"x": p["x"], "y": p["y"]}

    def like(v):
        return kappa * pt.cos(pt.arctan2(v["y"], v["x"]) - a0)

    model = _build(params, like)
    _rescale(params, {"x": 40.0, "y": 0.3})
    # posterior: radius Rayleigh(1) (the N(0, I) pair), angle von Mises
    r_ = np.sqrt(-2.0 * np.log(rng.uniform(size=N)))
    ang = rng.vonmises(a0, kappa, N)
    idata = _idata(params, {"x": r_ * np.cos(ang), "y": r_ * np.sin(ang)})

    r = _normalized_lnZ(model, params, idata)

    assert abs(r.lnZ - np.log(i0(kappa))) < _tol(r)
