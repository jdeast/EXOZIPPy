"""The raw coordinate of a logit element is FLAT (review 1.2.9).

Parameter.build_pymc used to give every raw coordinate a pm.Normal(0, 1)
density and cancel it on the logit (two finite bounds) elements with a
+0.5*raw**2 term clipped at |raw| = 1e4.  The cancellation's float64 residual
was what the DC2018_128 PTDE runaway fed on, and the clip that fixed it put a
-0.5*(raw**2 - 1e8) wall in the "uniform" prior at 1e4 whitened widths from
seed 0's anchor -- -1.5e8 nats at the close seeds of a real multi-seed
microlensing start.  Now the logit raw density is flat (parameter.py,
_raw_logp) and nothing is cancelled.  These tests pin the three properties
that make that the fix rather than a loosening:

  * the prior is uniform in the physical value wherever the raw coordinate
    sits -- inside the old clip, at 2e4 widths, at 1e7 -- and a seed 2e4
    widths from seed 0 pays no penalty beyond the Jacobian;
  * far from the anchor the logp only ever FALLS with |raw| and stays below
    the start, so a proposal-accepting-increases sampler cannot ratchet it;
  * the non-logit elements keep their N(0, 1) prior, the RV keeps its
    `<label>_raw` name, and a NaN still propagates to logp.
"""

import numpy as np
import pymc as pm
import pytensor
import pytest

from exozippy.components.parameter import Parameter

LOWER, UPPER = 0.0, 10.0
INIT = 2.0
INIT_SCALE = 1.0e-6  # one raw unit ~ 1e-6 in the physical value near INIT
LOG_SQRT_2PI = 0.5 * np.log(2.0 * np.pi)


@pytest.fixture(scope="module")
def flat_leaf():
    """A bounded, likelihood-free leaf: its logp is its prior alone."""
    p = Parameter(
        label="toy.x",
        initval=INIT,
        lower=LOWER,
        upper=UPPER,
        init_scale=INIT_SCALE,
    )
    with pm.Model() as model:
        val = p.build_pymc()
    logp = model.compile_logp()
    (vv,) = model.value_vars
    (val_v,) = model.replace_rvs_by_values([val])
    val_fn = pytensor.function(
        [vv], [val_v, pytensor.grad(pytensor.tensor.sum(val_v), vv)]
    )
    return p, model, logp, val_fn


def _lp(logp, raw):
    return float(logp({"toy.x_raw": np.array([raw], dtype=float)}))


def _log_physical_density(logp, val_fn, raw):
    """log p(val): the raw-space logp minus log|dval/draw|."""
    _val, dval = val_fn(np.array([raw], dtype=float))
    return _lp(logp, raw) - float(np.log(np.abs(np.ravel(dval)[0])))


def test_rv_keeps_its_name_and_its_place_in_the_layout(flat_leaf):
    """
    Given a logit leaf,
    When it is built,
    Then its one free RV is still `<label>_raw` with an untransformed value
      variable of the same name -- the name trace reuse, restart files, the
      polish and the persisted whitening all key on.
    """
    _p, model, _logp, _val_fn = flat_leaf

    assert [rv.name for rv in model.free_RVs] == ["toy.x_raw"]
    assert [v.name for v in model.value_vars] == ["toy.x_raw"]
    assert model.rvs_to_transforms[model.free_RVs[0]] is None


@pytest.mark.parametrize("raw", [5.0e3, 2.0e4, 1.0e6, 1.0e7, -2.0e4, -1.0e7])
def test_prior_is_uniform_far_from_the_anchor(flat_leaf, raw):
    """
    Given a logit leaf whose raw coordinate sits at `raw` whitened widths
      from seed 0's anchor -- inside the old 1e4 clip, past it, and far past
      it (+/-1e7 moves logit(q) by ~6, i.e. most of the way to a bound),
    When the implied PHYSICAL log density is computed (logp minus
      log|dval/draw|),
    Then it equals the one at the anchor to 1e-9: the prior is uniform in
      the value, with no wall anywhere inside the bounds.  Under the old clip
      the 2e4 point was 1.5e8 nats down.
    """
    _p, _model, logp, val_fn = flat_leaf

    at_anchor = _log_physical_density(logp, val_fn, 0.0)
    far = _log_physical_density(logp, val_fn, raw)

    assert far == pytest.approx(at_anchor, abs=1e-9)
    # The point is a genuine interior value, not one pinned at a bound.
    val, _ = val_fn(np.array([raw]))
    assert LOWER + 1e-3 < float(np.ravel(val)[0]) < UPPER - 1e-3


def test_a_seed_2e4_widths_away_pays_only_the_jacobian(flat_leaf):
    """
    Given two seeds of one fit, 2e4 whitened widths apart (seed 0 is the
      anchor; the second goes through raw_from_initval, the same frozen
      transform multi-seed starts use),
    When the logp of the two starts is compared,
    Then the difference is exactly the logit Jacobian's
      log(q(1-q)) difference -- a few hundredths of a nat -- and not the
      -0.5*(raw**2 - 1e8) = -1.5e8 the old clip charged.
    """
    p, _model, logp, val_fn = flat_leaf
    alt_val = INIT + 2.0e4 * INIT_SCALE
    raw_alt = float(np.ravel(p.raw_from_initval(np.array([alt_val])))[0])
    assert raw_alt > 1.0e4  # it really is past the old clip

    delta = _lp(logp, raw_alt) - _lp(logp, 0.0)

    def log_q1q(v):
        q = (v - LOWER) / (UPPER - LOWER)
        return np.log(q) + np.log1p(-q)

    val_alt, _ = val_fn(np.array([raw_alt]))
    expected = log_q1q(float(np.ravel(val_alt)[0])) - log_q1q(INIT)
    assert delta == pytest.approx(expected, abs=1e-9)
    assert abs(delta) < 0.1


def test_logp_only_falls_far_from_the_anchor(flat_leaf):
    """
    Given the leaf, and raw excursions out to the |raw| ~ 1e17 the
      DC2018_128 runaway reached,
    When logp is evaluated along both rays,
    Then it never exceeds the prior's own maximum (log(1/4) at q = 1/2, plus
      the kept -ln sqrt(2 pi)), it only FALLS with |raw| once logit(q) is
      past that peak (|raw| >= 1e7 here), and at 1e17 it is overwhelmingly
      negative.  That is the no-runaway property: PTDE accepts increases,
      and with no large term left to cancel there is no float64 residual for
      it to select -- the logit Jacobian and the saturation guard both fall
      monotonically outside the peak.
    """
    _p, _model, logp, _val_fn = flat_leaf
    ceiling = np.log(0.25) - LOG_SQRT_2PI + 1e-12
    mags = 10.0 ** np.arange(0, 18)

    for sign in (1.0, -1.0):
        lps = np.array([_lp(logp, sign * m) for m in mags])
        assert np.all(np.isfinite(lps)), lps
        assert np.all(lps <= ceiling), lps
        assert np.all(np.diff(lps[mags >= 1e7]) < 0.0), lps
        assert lps[-1] < -1.0e10, lps[-1]

    # The same with a dense random sample of large excursions, which is how
    # the old residual showed up (positive about half the time).
    rng = np.random.default_rng(1229)
    for r in rng.uniform(4.0, 17.0, 2000):
        val = rng.choice([-1.0, 1.0]) * 10.0**r
        assert _lp(logp, val) <= ceiling


def test_a_nan_raw_still_poisons_logp(flat_leaf):
    """
    Given the flat logit density,
    When the raw coordinate is NaN,
    Then logp is NaN -- the sampler's own reject signal, which the
      microlensing NaN-scrub removals (mulensing/physics.py) rely on.
    """
    _p, _model, logp, _val_fn = flat_leaf

    assert np.isnan(_lp(logp, np.nan))


def test_a_mixed_vector_keeps_the_normal_on_its_non_logit_element():
    """
    Given one parameter vector with a logit element and an unbounded element
      whose prior is its raw N(0, 1) (no sigma: N(initval, init_scale)),
    When the RV's own log density is evaluated,
    Then it is -0.5*raw**2 on the unbounded element only, plus the kept
      -ln sqrt(2 pi) per element; the logit element's raw contributes
      nothing (its prior is the logit_uniform_prior potential).
    """
    # Arrange
    p = Parameter(
        label="toy.v",
        shape=(2,),
        initval=[2.0, 5.0],
        lower=[0.0, -np.inf],
        upper=[10.0, np.inf],
        init_scale=[0.1, 0.3],
    )
    with pm.Model() as model:
        p.build_pymc()
    (rv,) = model.free_RVs
    rv_logp = model.compile_logp(vars=[rv], sum=True)
    raw = np.array([3.0e4, 1.7])

    # Act
    got = float(rv_logp({"toy.v_raw": raw}))

    # Assert
    assert rv.name == "toy.v_raw"
    assert got == pytest.approx(-0.5 * 1.7**2 - 2.0 * LOG_SQRT_2PI, abs=1e-12)


def test_an_all_non_logit_vector_is_still_a_plain_normal():
    """
    Given a parameter with no logit element,
    When it is built,
    Then its raw RV is the same pm.Normal(0, 1) it always was, so such
      parameters are unchanged bit for bit.
    """
    p = Parameter(
        label="toy.u",
        initval=1.0,
        init_scale=0.5,
        lower=-np.inf,
        upper=np.inf,
    )
    with pm.Model() as model:
        p.build_pymc()
    (rv,) = model.free_RVs

    assert type(rv.owner.op).__name__ == "NormalRV"
    raw = {"toy.u_raw": np.array([2.5])}
    assert float(model.compile_logp()(raw)) == pytest.approx(
        -0.5 * 2.5**2 - LOG_SQRT_2PI, abs=1e-12
    )


@pytest.mark.slow
@pytest.mark.parametrize("backend", ["pymc", "nutpie", "numpyro", "blackjax"])
def test_every_nuts_backend_samples_the_flat_raw(backend):
    """
    Given a logit element under a N(3, 0.5) likelihood (its flat raw plus
      Jacobian is a uniform prior, far wider than the likelihood) and an
      unbounded element whose prior is its raw N(0, 1), i.e. N(1, 0.5),
    When it is sampled with each NUTS backend (pymc with two worker
      processes, so the model is shipped to them; nutpie; numpyro and
      blackjax through JAX),
    Then each recovers both marginals.  The raw RV is a pm.CustomDist now,
      and "compiles" is not proof a sampler works with it -- sampling is.
    """
    # Arrange
    if backend != "pymc":
        pytest.importorskip(backend)
    if backend == "blackjax":
        from exozippy.compat import patch_blackjax_progress_bar

        patch_blackjax_progress_bar()
    px = Parameter(
        label="toy.x", initval=2.0, lower=0.0, upper=10.0, init_scale=0.5
    )
    pu = Parameter(
        label="toy.u",
        initval=1.0,
        lower=-np.inf,
        upper=np.inf,
        init_scale=0.5,
    )
    with pm.Model():
        x = px.build_pymc()
        u = pu.build_pymc()
        pm.Deterministic("xd", x)
        pm.Deterministic("ud", u)
        pm.Normal("obs", mu=x, sigma=0.5, observed=np.array([3.0]))
        kw = dict(
            draws=600, tune=600, chains=2, random_seed=3, progressbar=False
        )

        # Act
        if backend == "pymc":
            idata = pm.sample(cores=2, **kw)
        else:
            # blackjax: vectorized, because parallel is one jax.pmap device
            # per chain and an earlier test on this worker may already have
            # started JAX with one device (#506; run.py warns and switches).
            if backend == "blackjax":
                kw["nuts_sampler_kwargs"] = {"chain_method": "vectorized"}
            idata = pm.sample(nuts_sampler=backend, **kw)

    # Assert
    xd = np.ravel(idata.posterior["xd"].values)
    ud = np.ravel(idata.posterior["ud"].values)
    assert np.all(np.isfinite(xd)) and np.all(np.isfinite(ud))
    assert abs(xd.mean() - 3.0) < 0.15 and abs(xd.std() - 0.5) < 0.1
    assert abs(ud.mean() - 1.0) < 0.15 and abs(ud.std() - 0.5) < 0.1
