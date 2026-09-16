"""`Model.set_initval` is load-bearing, so its contract is pinned (4.3.1).

After review 4.3.1 no sampler branch is handed a start: the chains begin at
`Model.initial_point()`, and `System.recenter_whitening_anchor` makes that
the polished start -- by folding the displacement into the logit anchor, and,
for Gaussian-path elements whose center IS their prior mean and therefore may
not move, by `Model.set_initval`.

**That makes correct starts STRUCTURAL, which is exactly why this file has to
exist.**  A silent upstream regression in `set_initval` (or in how pymc reads
the model's initial point) would now mis-start every fit rather than raising,
and a wrong start is not visible in any output: the run completes, the trace
looks normal, and only the chains' first adaptation windows know.  A test that
fails loudly is the only thing that turns that into a diagnosis.  CI resolves
exactly one pymc version per run, and `pytest-latest-deps` (nightly) then
exercises this against the newest resolvable pymc for free.

MEASURED, not inferred.  JDE flagged 2026-09-11 that "back on pymc 5.x there
seemed to be a lot of buggy behavior with setting/querying a specific point.
i think it was silently dropping my requested point and drawing from a wide
prior".  Rather than assume, every claim below was probed on the installed
pymc 6.3.2 / pytensor 3.3.1 AND, in a throwaway venv on /pool, on **pymc
6.0.0** -- `pyproject`'s declared floor, which resolved pytensor 3.0.7 and
numpy 2.4.6.  Every number was IDENTICAL on both endpoints of
`pymc>=6.0.0,<7.0.0`, including the draw vectors, so the range needs no
change.

THE REMEMBERED SYMPTOM HAS A MECHANISM and it is most likely ours rather than
pymc's.  Two traps produce exactly "my point was dropped and replaced by a
wide-prior draw", and both return plausible NUMBERS rather than raising:

* `Parameter.value` is built on the RV, not the value var, so evaluating it
  directly DRAWS FROM THE PRIOR -- every call returns a different
  plausible-looking start.  (It cost about an hour on the mann branch,
  chasing a nonexistent "start does not match initval" bug.)
* **The first stored draw is not the start.**  A `draws=1, tune=0` NUTS run
  from `[1.5, -2.5, 0.25]` stored `[-1.408, 0.789, 0.364]` in the session's
  own probe and looked exactly like a dropped initval.  It was one NUTS step
  at the default step size.

So "did the chain start here" is asked with `pm.Metropolis(scaling=1e-12)`
below, never with NUTS.
"""

import numpy as np
import pymc as pm
import pytest

# The requested start, in raw (whitened) units where the prior is N(0, 1) --
# so these are displacements in sigma, far outside any rounding tolerance and
# nowhere near the anchor a dropped start would fall back to.
REQUESTED = np.array([1.5, -2.5, 0.25])


def _build():
    """A model shaped like ours: one untransformed `pm.Normal` raw vector
    (exactly what `Parameter.build_pymc` creates) plus an observation."""
    with pm.Model() as model:
        raw = pm.Normal("raw", 0.0, 1.0, shape=3, initval=np.zeros(3))
        pm.Normal("obs", mu=raw.sum(), sigma=1.0, observed=0.0)
    return model, raw


def _nuts_draws(model, **kwargs):
    with model:
        idata = pm.sample(
            draws=2,
            tune=0,
            chains=1,
            step=pm.NUTS(target_accept=0.9),
            cores=1,
            random_seed=42,
            progressbar=False,
            compute_convergence_checks=False,
            **kwargs,
        )
    return idata.posterior["raw"].values[0]


def test_set_initval_is_honored_bit_for_bit_like_initvals():
    """
    Given the same model, the same seed and the same requested start,
    When one build gets it through `Model.set_initval` and another through
      `pm.sample(initvals=)`,
    Then the draws agree EXACTLY, and both differ from a run given no start.

    This is the decisive comparison, promoted from the session's own probe
    rather than rewritten.  Agreement alone would not be enough -- two runs
    that both IGNORED the start would also agree -- so the third arm is what
    proves the start is genuinely consumed.  Measured identical on pymc 6.3.2
    and on the 6.0.0 floor.
    """
    # ARRANGE
    model_a, raw_a = _build()
    model_a.set_initval(raw_a, REQUESTED)
    model_b, _raw_b = _build()
    model_c, _raw_c = _build()

    # ACT
    via_set_initval = _nuts_draws(model_a)
    via_initvals = _nuts_draws(model_b, initvals={"raw": REQUESTED})
    no_start = _nuts_draws(model_c)

    # ASSERT
    np.testing.assert_array_equal(
        via_set_initval,
        via_initvals,
        err_msg=(
            "set_initval and initvals= no longer produce identical chains; "
            "after review 4.3.1 set_initval is the ONLY channel carrying a "
            "Gaussian-path element's polished start, so a divergence here "
            "mis-starts every fit silently"
        ),
    )
    # The control must fire: a run that ignored the start would match this.
    assert not np.allclose(via_set_initval, no_start), (
        "the requested start made no difference to the draws, so this test "
        "cannot distinguish 'honored' from 'silently dropped'"
    )


def test_a_tiny_step_sampler_starts_exactly_at_the_requested_point():
    """
    Given `set_initval` on the raw vector,
    When a sampler that can barely move takes one draw with no tuning,
    Then that draw IS the requested point.

    `Metropolis(scaling=1e-12)` is the clean way to ask "did the chain start
    here": its proposal is numerically zero, so the first stored draw is the
    start.  NUTS is NOT usable for this question -- its first stored draw is
    post-step, which is the red herring recorded in this module's docstring.
    """
    # ARRANGE
    model, raw = _build()
    model.set_initval(raw, REQUESTED)

    # ACT
    with model:
        idata = pm.sample(
            draws=1,
            tune=0,
            chains=1,
            step=pm.Metropolis(scaling=1e-12),
            cores=1,
            random_seed=0,
            progressbar=False,
            compute_convergence_checks=False,
        )
    first = idata.posterior["raw"].values[0, 0]

    # ASSERT
    np.testing.assert_allclose(first, REQUESTED, rtol=0, atol=1e-5)


def test_set_initval_reaches_initial_point_and_is_deterministic():
    """
    Given `set_initval` on the raw vector,
    When `Model.initial_point()` is read repeatedly,
    Then it returns the requested point, unchanged on every call.

    Determinism is the half that matters for us: `get_raw_start`, the
    whitening probe, the seed ledger, the startup table and the sampler all
    read the start separately, and a point that varied per call would put
    them at different physical places while every one of them reported a
    plausible number.
    """
    # ARRANGE
    model, raw = _build()
    assert np.allclose(model.initial_point()["raw"], 0.0)  # the anchor

    # ACT
    model.set_initval(raw, REQUESTED)
    points = [model.initial_point()["raw"] for _ in range(3)]

    # ASSERT
    for point in points:
        np.testing.assert_array_equal(point, REQUESTED)


def test_set_initval_goes_through_a_bounded_rvs_forward_transform():
    """
    Given a bounded RV, whose initial point lives in TRANSFORMED space,
    When `set_initval` is given a physical value,
    Then `initial_point()` holds its forward transform, which inverts back to
      exactly that value.

    Our own raw variables are untransformed `pm.Normal`s, so this is not a
    path production takes.  It is pinned because it is the claim that
    `set_initval` means "start at this VALUE" rather than "write these bytes
    into the point dict" -- and a pymc that silently changed which space the
    argument is in would corrupt any future parameterization that does carry
    a transform, in the direction of a plausible wrong number.
    """
    # ARRANGE
    with pm.Model() as model:
        b = pm.Uniform("b", lower=0.0, upper=10.0, initval=5.0)
        pm.Normal("obs", mu=b, sigma=1.0, observed=5.0)

    # ACT
    model.set_initval(b, 9.0)
    point = model.initial_point()

    # ASSERT
    assert list(point) == ["b_interval__"], (
        f"the bounded RV's value variable is no longer the interval "
        f"transform: {list(point)}"
    )
    assert float(point["b_interval__"]) == pytest.approx(2.19722458, abs=1e-8)
    transform = model.rvs_to_transforms[b]
    back = transform.backward(point["b_interval__"], *b.owner.inputs).eval()
    assert float(back) == pytest.approx(9.0, rel=0, abs=1e-12)


# ----------------------------------------------------------------------
# The two branches whose explicit start was DELETED in 4.3.1
# ----------------------------------------------------------------------


def test_the_jax_sampler_consumes_the_models_own_start():
    """
    Given pymc's JAX NUTS entry point with no `initvals=`,
    When the model carries a `set_initval` start,
    Then it draws exactly what it drew when handed `initvals=`, and not what
      it draws with no start at all.

    run.py's jax branch used to pass `initvals=internal_start`; 4.3.1 deleted
    it because `Model.initial_point()` is now the polished start.  This is
    the measurement that makes that deletion safe rather than a guess, and it
    is the only thing in the suite that exercises the jax branch's start.
    """
    pytest.importorskip("numpyro")
    import jax
    from pymc.sampling.jax import sample_jax_nuts

    jax.config.update("jax_enable_x64", True)

    def run(model, **kwargs):
        with model:
            idata = sample_jax_nuts(
                draws=1,
                tune=0,
                chains=1,
                jitter=False,
                nuts_sampler="numpyro",
                random_seed=7,
                progressbar=False,
                **kwargs,
            )
        return idata.posterior["raw"].values[0, 0]

    # ARRANGE
    model_a, raw_a = _build()
    model_a.set_initval(raw_a, REQUESTED)
    model_b, _ = _build()
    model_c, _ = _build()

    # ACT
    via_model = run(model_a)
    via_initvals = run(model_b, initvals={"raw": REQUESTED})
    no_start = run(model_c)

    # ASSERT
    np.testing.assert_array_equal(via_model, via_initvals)
    assert not np.allclose(via_model, no_start)


def test_nutpie_consumes_the_models_own_start_and_ignores_init_mean():
    """
    Given nutpie through pm.sample,
    When the start is given as `set_initval` and, separately, as the
      `init_mean` run.py used to pass,
    Then only the first is consumed -- `init_mean` leaves the draw exactly
      where a no-start run leaves it.

    This is not a curiosity: it means run.py's nutpie branch was a SECOND
    live instance of 1.3.6's bug class, and 4.3.1's deletion FIXES it rather
    than merely simplifying it.  The mechanism is in nutpie's own source --
    `CompiledPyMCModel._make_model(init_mean)` accepts the argument and never
    uses it, handing `self.initial_point_func` to `PyMcModel` instead -- so
    for a pymc-compiled model the start comes from the MODEL.  Measured
    against nutpie 0.16.11 + pymc 6.3.2 through both spellings
    (`nuts_sampler_kwargs=` and the newer `nuts=`).

    If a future nutpie starts honoring `init_mean`, this test fails and the
    comment at run.py's nutpie branch has to be re-derived -- which is the
    point of pinning an upstream fact rather than quoting it.
    """
    pytest.importorskip("nutpie")

    def run(model, **kwargs):
        with model:
            idata = pm.sample(
                draws=1,
                tune=0,
                chains=1,
                nuts_sampler="nutpie",
                cores=1,
                random_seed=7,
                return_inferencedata=True,
                progressbar=False,
                **kwargs,
            )
        return idata.posterior["raw"].values[0, 0]

    # ARRANGE
    model_a, raw_a = _build()
    model_a.set_initval(raw_a, REQUESTED)
    model_b, _ = _build()
    model_c, _ = _build()

    # ACT
    via_model = run(model_a)
    via_init_mean = run(model_b, nuts={"init_mean": REQUESTED.copy()})
    no_start = run(model_c)

    # ASSERT
    assert not np.allclose(via_model, no_start), (
        "nutpie no longer reads the model's initial point, so run.py's "
        "nutpie branch passes no start it consumes (review 4.3.1 deleted "
        "init_mean on the measured grounds that it was inert)"
    )
    np.testing.assert_array_equal(
        via_init_mean,
        no_start,
        err_msg=(
            "nutpie now honors init_mean, which it did not at 0.16.11; "
            "re-derive the comment at run.py's nutpie branch"
        ),
    )
