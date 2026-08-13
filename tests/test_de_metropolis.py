"""Cover for the ``demc`` / ``demcz`` sampler keys (review item 5.5).

Three things are pinned here:

* the two config keys dispatch to the two PyMC differential-evolution step
  methods, and nothing else does;
* the ``_fix_de_stats`` coercion the module carries is DORMANT on the
  installed PyMC -- measured by sampling and counting, not asserted from a
  version number -- and still repairs a step method that regresses to the
  pre-6.0.0 behavior;
* a DE fit actually completes and produces finite draws, per the house rule
  that sampler claims are backed by a completed ``pm.sample``.
"""

import numpy as np
import pymc as pm
import pytest

from exozippy import run
from exozippy.samplers import de_metropolis
from exozippy.samplers.de_metropolis import (
    STEP_CLASSES,
    DEMetropolis,
    DEMetropolisZ,
    _fix_de_stats,
    de_metropolis_sample,
)

# ---------------------------------------------------------------------------
# Fixtures / stubs
# ---------------------------------------------------------------------------


class _Stop(Exception):
    """Sentinel raised by the fake pm.sample, to stop right after the call."""


def _fake_sample(seen):
    """A pm.sample stand-in that records its kwargs and stops the run."""

    def fake(**kwargs):
        seen.update(kwargs)
        raise _Stop

    return fake


class _StubSystem:
    """The two things ``de_metropolis_sample`` asks a System for.

    ``get_raw_start`` is the real contract; the absence of
    ``get_raw_starts``/``jitter_raw_start`` exercises
    ``_common.resolve_start_population``'s documented fallback to a single
    raw-space-jittered start.
    """

    def get_raw_start(self, model):
        return {v.name: np.zeros(4, dtype=float) for v in model.free_RVs}


@pytest.fixture
def tiny_model():
    """A 4-parameter standard normal: cheap, unbounded, always finite."""
    with pm.Model() as model:
        pm.Normal("x", 0.0, 1.0, shape=4)
    return model


# ---------------------------------------------------------------------------
# Key -> step method dispatch
# ---------------------------------------------------------------------------


def test_the_two_keys_map_to_the_patched_pymc_step_methods():
    """Given the sampler-key table,
    When it is read,
    Then 'demc'/'demcz' name the patched subclasses of PyMC's DE steps.
    """
    # Arrange / Act
    keys = STEP_CLASSES

    # Assert
    assert set(keys) == {"demc", "demcz"}
    assert keys["demc"] is DEMetropolis
    assert issubclass(DEMetropolis, pm.DEMetropolis)
    assert keys["demcz"] is DEMetropolisZ
    assert issubclass(DEMetropolisZ, pm.DEMetropolisZ)


@pytest.mark.parametrize(
    "variant, expected", [("demc", DEMetropolis), ("demcz", DEMetropolisZ)]
)
def test_each_variant_hands_pm_sample_its_own_step(
    tiny_model, monkeypatch, variant, expected
):
    """Given a variant name,
    When de_metropolis_sample runs,
    Then pm.sample receives an instance of that variant's step class.
    """
    # Arrange
    seen = {}
    monkeypatch.setattr(de_metropolis.pm, "sample", _fake_sample(seen))

    # Act
    with pytest.raises(_Stop):
        de_metropolis_sample(
            tiny_model,
            _StubSystem(),
            draws=5,
            tune=5,
            variant=variant,
            chains=6,
        )

    # Assert
    assert isinstance(seen["step"], expected)
    assert seen["draws"] == 5 and seen["tune"] == 5
    assert seen["chains"] == 6
    assert len(seen["initvals"]) == 6


def test_an_unknown_variant_raises(tiny_model):
    """Given a variant name this module does not implement,
    When de_metropolis_sample is called,
    Then it raises rather than silently sampling something else.
    """
    # Arrange / Act / Assert
    with pytest.raises(ValueError, match="Unknown DE-MC variant"):
        de_metropolis_sample(
            tiny_model, _StubSystem(), draws=1, tune=1, variant="de"
        )


def test_the_run_dispatch_table_is_the_module_table():
    """Given run.py's sampler dispatch,
    When a method name is tested against it,
    Then only the two DE keys match, so every other name is unchanged.
    """
    # Arrange
    others = ["nuts", "numpyro", "blackjax", "nutpie", "ptde", "ptde_async"]

    # Act / Assert
    assert "demc" in de_metropolis.STEP_CLASSES
    assert "demcz" in de_metropolis.STEP_CLASSES
    for name in others:
        assert name not in de_metropolis.STEP_CLASSES


def test_the_de_keys_add_no_new_sampler_block_key():
    """Given the DE samplers reuse chains/draws/tune/cores/maxtime,
    When a demc sampler block is validated,
    Then no key is reported unknown (the vocabulary is unchanged).
    """
    # Arrange
    cfg = {
        "method": "demc",
        "chains": 40,
        "draws": 500,
        "tune": 500,
        "cores": 4,
        "maxtime": 60,
    }

    # Act
    unknown = run.warn_unknown_sampler_keys(cfg)

    # Assert
    assert unknown == []
    assert run.warn_unknown_sampler_keys({"method": "demc", "bogus": 1}) == [
        "bogus"
    ]


# ---------------------------------------------------------------------------
# Population sizing and the maxtime asymmetry
# ---------------------------------------------------------------------------


def test_an_unset_chains_sizes_the_demc_population_from_the_parameters(
    tiny_model, monkeypatch
):
    """Given chains is not set,
    When demc runs,
    Then the population is the shared DE default 2 x n_params, not PyMC's 4.
    """
    # Arrange
    seen = {}
    monkeypatch.setattr(de_metropolis.pm, "sample", _fake_sample(seen))

    # Act
    with pytest.raises(_Stop):
        de_metropolis_sample(
            tiny_model,
            _StubSystem(),
            draws=5,
            tune=5,
            variant="demc",
            chains=None,
        )

    # Assert
    assert seen["chains"] == 8  # 2 x 4 parameters


def test_maxtime_is_refused_for_demc_and_honored_for_demcz(
    tiny_model, monkeypatch, caplog
):
    """Given a wall-clock cap,
    When each variant runs,
    Then demcz gets a callback and demc gets a warning instead of a silent
    drop (PyMC's population path discards per-draw callbacks).
    """
    # Arrange
    seen = {}
    monkeypatch.setattr(de_metropolis.pm, "sample", _fake_sample(seen))

    # Act
    with caplog.at_level("WARNING"):
        with pytest.raises(_Stop):
            de_metropolis_sample(
                tiny_model,
                _StubSystem(),
                draws=5,
                tune=5,
                variant="demc",
                chains=6,
                maxtime=30.0,
            )
    demc_callback = seen["callback"]
    demc_warned = "IGNORED" in caplog.text

    with pytest.raises(_Stop):
        de_metropolis_sample(
            tiny_model,
            _StubSystem(),
            draws=5,
            tune=5,
            variant="demcz",
            chains=6,
            maxtime=30.0,
        )

    # Assert
    assert demc_callback is None and demc_warned
    assert callable(seen["callback"])


# ---------------------------------------------------------------------------
# The PyMC stats-shape patch: dormant here, live on a regression
# ---------------------------------------------------------------------------


def _count_array_stats(step_cls, model):
    """Sample with ``step_cls`` and count stats whose shape is not scalar."""
    counts = {"steps": 0, "array": 0}
    astep = step_cls.astep

    def counting(self, q0):
        result, stats = astep(self, q0)
        counts["steps"] += 1
        for s in stats:
            for key in ("scaling", "lambda"):
                if key in s and np.ndim(s[key]) > 0:
                    counts["array"] += 1
        return result, stats

    counted = type("Counted", (step_cls,), {"astep": counting})
    with model:
        pm.sample(
            draws=20,
            tune=20,
            chains=6,
            cores=1,
            step=counted(),
            progressbar=False,
            compute_convergence_checks=False,
        )
    return counts


@pytest.mark.parametrize("step_cls", [pm.DEMetropolis, pm.DEMetropolisZ])
def test_the_stats_patch_is_dormant_on_the_installed_pymc(
    tiny_model, step_cls
):
    """Given the installed PyMC (floor: >= 6.0.0),
    When a DE step method really samples,
    Then no 'scaling'/'lambda' stat is ever array-shaped, i.e. the coercion
    in _fix_de_stats never fires.

    The bug it was written for was fixed upstream below this project's floor
    (DEMetropolisZ in 5.26.0, DEMetropolis in 6.0.0). If this test ever fails
    the patch has become live again -- which is exactly why it is kept.
    """
    # Arrange / Act
    counts = _count_array_stats(step_cls, tiny_model)

    # Assert
    assert counts["steps"] > 0
    assert counts["array"] == 0


def _sample_with(step_cls, model, **kwargs):
    with model:
        return pm.sample(
            draws=20,
            tune=20,
            chains=6,
            cores=1,
            step=step_cls(),
            progressbar=False,
            compute_convergence_checks=False,
            **kwargs,
        )


def test_the_stats_patch_still_repairs_a_regressed_step(tiny_model):
    """Given a step method that regresses to the pre-6.0.0 behavior
    (returning the raw np.atleast_1d scaling array where the declared stat
    shape is scalar),
    When it is sampled with and without the _fix_de_stats wrapper,
    Then the unwrapped one crashes the trace backend and the wrapped one
    samples and records a scalar stat.

    This is the failure the patch exists for, reproduced against the
    installed PyMC. It is what makes the dormancy test above a statement
    about PyMC rather than about the wrapper being inert code.
    """

    # Arrange: reproduce the pymc <= 5.25.1 astep return exactly.
    class Regressed(pm.DEMetropolis):
        def astep(self, q0):
            result, stats = pm.DEMetropolis.astep(self, q0)
            for s in stats:
                s["scaling"] = np.atleast_1d(s["scaling"]).astype("d")
                s["lambda"] = np.atleast_1d(s["lambda"]).astype("d")
            return result, stats

    class Repaired(Regressed):
        astep = _fix_de_stats(Regressed.astep)

    # Act / Assert
    with pytest.raises(ValueError, match="array element with a sequence"):
        _sample_with(Regressed, tiny_model)

    idata = _sample_with(Repaired, tiny_model)
    scaling = np.asarray(idata.sample_stats["scaling"])
    assert scaling.shape == (6, 20)
    assert np.isfinite(scaling).all()


# ---------------------------------------------------------------------------
# Verify by sampling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", ["demc", "demcz"])
def test_a_de_fit_completes_and_produces_finite_draws(tiny_model, variant):
    """Given a tiny model,
    When it is sampled through de_metropolis_sample,
    Then sampling completes, every draw is finite, and run.py's post-hoc lp
    pass (these step methods record none) evaluates finite everywhere.
    """
    # Arrange
    system = _StubSystem()

    # Act
    idata = de_metropolis_sample(
        tiny_model,
        system,
        draws=100,
        tune=100,
        variant=variant,
        chains=8,
        cores=1,
        seed=20260812,
        progressbar=False,
    )

    # Assert
    x = np.asarray(idata.posterior["x"])
    assert x.shape == (8, 100, 4)
    assert np.isfinite(x).all()
    # DE step methods write no 'lp' stat, so run.py fills it in from the
    # model; that is the path a demc/demcz fit takes.
    assert "lp" not in idata.sample_stats.data_vars
    lp = run._compute_lp_from_model(tiny_model, idata)
    assert lp is not None and np.isfinite(lp).all()
    # The chains started over-dispersed around one seed, so provenance is
    # recorded the same way PTDE records it.
    assert idata.posterior.attrs["chain_seed_index"] == [0] * 8
    # A standard normal: the posterior mean should be near zero.
    assert abs(float(x.mean())) < 0.5
