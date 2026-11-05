"""Checkpointed numpyro NUTS (samplers/nuts_checkpoint.py).

A fit stopped mid-sampling used to lose every draw: pymc's sample_jax_nuts
returns nothing until the end.  These tests stop a small fit after two of
three chunks, rerun it, and check that it resumes from the checkpoint, draws
only what was missing, and returns the full posterior.
"""

import os

import numpy as np
import pymc as pm
import pytest

from exozippy.samplers import nuts_checkpoint as nc

pytest.importorskip("numpyro")


def _model():
    with pm.Model() as model:
        pm.Normal("x", 1.5, 0.5)
    return model


def _sample(model, draws=60, seed=7):
    from pymc.sampling.jax import sample_jax_nuts

    with model:
        return sample_jax_nuts(
            draws=draws,
            tune=50,
            chains=2,
            chain_method="vectorized",
            random_seed=seed,
            progressbar=False,
            compute_convergence_checks=False,
            nuts_sampler="numpyro",
        )


class _Stop(Exception):
    pass


@pytest.mark.slow
def test_a_stopped_fit_resumes_and_draws_only_what_was_missing(
    tmp_path, monkeypatch
):
    """
    Given a checkpointed numpyro fit stopped after the warmup and two of three
    draw chunks,
    When the same fit is run again, with another seed,
    Then it resumes from the checkpoint, draws only the last chunk, returns all
    draws, and keeps the starting seed.
    """
    path = str(tmp_path / "fit_nutsckpt.pkl")
    fp = nc.fingerprint({"model": "x"}, draws=60)
    model = _model()

    saves = []
    real_save = nc._save

    def stop_after_two_chunks(p, payload):
        real_save(p, payload)
        saves.append((payload["warm"], payload["done"]))
        if payload["done"] == 40:
            raise _Stop()

    monkeypatch.setattr(nc, "_save", stop_after_two_chunks)
    with nc.checkpointed_numpyro(path, 20, fp), pytest.raises(_Stop):
        _sample(model)
    assert [d for _, d in saves] == [
        0,
        0,
        0,
        20,
        40,
    ]  # warmup 20, 40, 50, then two chunks

    saves.clear()
    with nc.checkpointed_numpyro(path, 20, fp):
        idata = _sample(model, seed=999)  # a seedless rerun draws a new seed
    assert [d for _, d in saves] == [60]  # only the last chunk was drawn
    x = idata.posterior["x"].values
    assert x.shape == (2, 60)
    assert abs(x.mean() - 1.5) < 0.3
    assert nc.checkpoint_seed(path) == 7  # the seed the chains started with


def test_a_checkpoint_of_another_fit_is_refused(tmp_path):
    """
    Given a checkpoint written under one fingerprint,
    When a fit with another fingerprint finds it,
    Then it raises rather than splice two fits.
    """
    path = str(tmp_path / "fit_nutsckpt.pkl")
    model = _model()
    with nc.checkpointed_numpyro(
        path, 30, nc.fingerprint({"model": "x"}, draws=60)
    ):
        _sample(model)
    with nc.checkpointed_numpyro(
        path, 30, nc.fingerprint({"model": "y"}, draws=60)
    ):
        with pytest.raises(RuntimeError, match="different fit"):
            _sample(model)


def test_pymc_gets_its_own_function_back(tmp_path):
    """
    Given pymc's numpyro sampling function,
    When a checkpointed block is entered and left,
    Then the function is swapped inside the block and restored after it.
    """
    import pymc.sampling.jax as pj

    original = pj._sample_numpyro_nuts
    with nc.checkpointed_numpyro(str(tmp_path / "c.pkl"), 10, "fp"):
        assert pj._sample_numpyro_nuts is not original
    assert pj._sample_numpyro_nuts is original


@pytest.mark.slow
def test_a_fit_stopped_mid_warmup_resumes_the_same_adaptation(
    tmp_path, monkeypatch
):
    """
    Given a checkpointed fit stopped after 20 of 50 warmup steps,
    When it is run again,
    Then the warmup resumes at step 20 under the same schedule, the step size
    ends adapted, and the posterior is right.

    A slow model's warmup alone can outlast a wall-clock limit, so the
    warmup is chunked too.  Stopped after 20 of 50 warmup steps, the rerun
    takes the other 30 under the same schedule: the step counter reaches
    tune, the step size is adapted (not the first-chunk value), and the
    posterior is right.
    """
    path = str(tmp_path / "fit_nutsckpt.pkl")
    fp = nc.fingerprint({"model": "x"}, draws=60)
    model = _model()

    saves = []
    states = []
    real_save = nc._save

    def stop_in_warmup(p, payload):
        real_save(p, payload)
        saves.append((payload["warm"], payload["done"]))
        states.append(payload["state"])
        if payload["warm"] == 20:
            raise _Stop()

    monkeypatch.setattr(nc, "_save", stop_in_warmup)
    with nc.checkpointed_numpyro(path, 20, fp), pytest.raises(_Stop):
        _sample(model)
    assert saves == [(20, 0)]

    saves.clear()
    with nc.checkpointed_numpyro(path, 20, fp):
        idata = _sample(model)
    # warmup 40, 50 (a short last chunk lands exactly on tune), then draws
    assert saves == [(40, 0), (50, 0), (50, 20), (50, 40), (50, 60)]
    assert np.all(np.asarray(states[-1].i) == 50 + 60)
    # adaptation ran to its end: the step size after warmup is fixed and
    # differs from the one the first chunk left
    step = idata.sample_stats["step_size"].values
    assert np.allclose(step, step[:, :1])
    assert not np.allclose(
        step[:, 0], np.asarray(states[0].adapt_state.step_size)
    )
    x = idata.posterior["x"].values
    assert x.shape == (2, 60)
    assert abs(x.mean() - 1.5) < 0.3
    assert abs(x.std() - 0.5) < 0.2
    # the warmup's own stats are kept (pymc's path discards them), across
    # the stop: one row per warmup step, chains first
    import pickle

    with open(path, "rb") as fh:
        ck = pickle.load(fh)
    widths = [np.asarray(f["num_steps"]).shape for f in ck["warm_fields"]]
    assert [w[0] for w in widths] == [2, 2, 2]
    assert sum(w[1] for w in widths) == 50


def test_the_fingerprint_carries_the_models_value_at_the_start():
    """
    Given two models with the same structure and different values at the start,
    When their start-logp stamps and fingerprints are taken,
    Then they differ, and the stamp is deterministic.

    Two models with the same structure but a different value at the start
    (a potential's width changed, a light curve edited) must not share a
    checkpoint.
    """
    with pm.Model() as a:
        pm.Normal("x", 1.5, 0.5)
    with pm.Model() as b:
        pm.Normal("x", 1.5, 0.6)
    sa, sb = nc.start_logp_stamp(a), nc.start_logp_stamp(b)
    assert sa != sb
    assert nc.fingerprint({"m": 1}, start_logp=sa) != nc.fingerprint(
        {"m": 1}, start_logp=sb
    )
    assert nc.start_logp_stamp(a) == sa  # deterministic


def test_the_fingerprint_covers_every_key_that_can_change_a_chain():
    """
    Given sampler configs that differ only in keys that cannot change a chain,
    or in one that can,
    When their chain settings are taken,
    Then the first pair match and the second differ.
    """
    base = {
        "method": "numpyro",
        "tune": 500,
        "seed": 3,
        "checkpoint_every": 10,
    }
    same = dict(base, seed=99, checkpoint_every=50, recompute_trace=False)
    other = dict(base, jitter=True)
    assert nc.chain_settings(base) == nc.chain_settings(same)
    assert nc.chain_settings(base) != nc.chain_settings(other)


@pytest.mark.slow
def test_a_stopped_and_resumed_fit_draws_what_pymc_itself_would_have(
    tmp_path, monkeypatch
):
    """
    Given the same model and seed,
    When it is sampled by pymc's own numpyro path, and checkpointed in chunks
    of 30 with a stop after the warmup's second chunk and a resume,
    Then the draws and step sizes are bit-identical: each chunk continues from
    the saved state's own random key, so chunking and resuming change nothing.
    """
    from pymc.sampling.jax import sample_jax_nuts

    def model():
        with pm.Model() as m:
            cov = np.array([[1.0, 0.9, 0.0], [0.9, 1.0, 0.0], [0.0, 0.0, 4.0]])
            pm.MvNormal("x", mu=np.zeros(3), cov=cov)
        return m

    kw = dict(
        draws=100,
        tune=150,
        chains=2,
        chain_method="vectorized",
        random_seed=11,
        progressbar=False,
        compute_convergence_checks=False,
        nuts_sampler="numpyro",
    )
    with model():
        ref = sample_jax_nuts(**kw)

    path = str(tmp_path / "fit_nutsckpt.pkl")
    real_save = nc._save

    def stop_at_60(p, payload):
        real_save(p, payload)
        if payload["warm"] == 60:
            raise _Stop()

    monkeypatch.setattr(nc, "_save", stop_at_60)
    with (
        nc.checkpointed_numpyro(path, 30, "fp"),
        model(),
        pytest.raises(_Stop),
    ):
        sample_jax_nuts(**kw)
    monkeypatch.setattr(nc, "_save", real_save)
    with nc.checkpointed_numpyro(path, 30, "fp"), model():
        got = sample_jax_nuts(**kw)

    assert np.array_equal(got.posterior["x"].values, ref.posterior["x"].values)
    assert np.array_equal(
        got.sample_stats["step_size"].values,
        ref.sample_stats["step_size"].values,
    )


class _StopAtSampler(Exception):
    def __init__(self, initial_point):
        super().__init__("sampler reached")
        self.initial_point = initial_point


def _drive_to_the_sampler(config, user_params, polish_engine, measure):
    """run_fit up to pymc's sample_jax_nuts; the start it would sample."""
    import os

    from exozippy import polish as polish_module
    from exozippy import run as run_module
    from exozippy import whitening

    def stop(*args, **kwargs):
        model = pm.modelcontext(None)
        point = model.initial_point()
        # raw = 0 is the start on every path (the anchor is re-centered on
        # it), so the raw point cannot tell two anchors apart: the logp there
        # does, and it is what the checkpoint's fingerprint stamps.
        raise _StopAtSampler((point, float(model.compile_logp()(point))))

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(polish_module, "polish_raw_starts", polish_engine)
        mp.setattr(whitening, "measure_and_whiten", measure)
        mp.setattr("pymc.sampling.jax.sample_jax_nuts", stop)
        cwd = os.getcwd()
        os.chdir(config["prefix"].rsplit("/", 1)[0])
        try:
            with pytest.raises(_StopAtSampler) as stopped:
                run_module.run_fit(config, user_params=user_params)
        finally:
            os.chdir(cwd)
    return stopped.value.initial_point


def test_a_resume_restores_the_saved_start_without_polishing_again(tmp_path):
    """
    Given a fresh numpyro fit whose polish moved the start and whose
      whitening was saved, then a NUTS checkpoint left beside it,
    When the same config is run again,
    Then it reaches the sampler without polishing or probing, at the very
      start the fresh run sampled from.

    A resume used to polish and probe again (4 min of W1's 6.4 min restart),
    and only the checkpoint's start-logp check caught a polish that landed
    elsewhere; now the start comes from the saved whitening file.
    """
    # ARRANGE
    from exozippy import whitening

    config = {
        "name": "resume_start",
        "orbit": [{"name": "o"}],
        "prefix": str(tmp_path / "resume"),
        "sampler": {
            "method": "numpyro",
            "tune": 10,
            "draws": 10,
            "chains": 1,
            "cores": 1,
            "seed_polish": 50,
            "checkpoint_every": 5,
            "recompute_trace": True,
        },
    }
    user_params = {
        "orbit.o.logP": {"initval": float(np.log10(10.0))},
        "orbit.o.tc": {"initval": 0.0},
        "orbit.o.secosw": {"initval": 0.0},
        "orbit.o.sesinw": {"initval": 0.0},
    }

    def moved(model, raw_starts, **kwargs):
        polished = [
            {k: np.asarray(v, dtype=float) + 0.75 for k, v in s.items()}
            for s in raw_starts
        ]
        kwargs["stops_out"].extend(("converged", "stub") for _ in polished)
        return polished, [0.0] * len(polished), "stub"

    def must_not_run(*args, **kwargs):
        raise AssertionError("a resume polished or probed again")

    def unmoved(model, raw_starts, **kwargs):
        kwargs["stops_out"].extend(("converged", "stub") for _ in raw_starts)
        return [dict(s) for s in raw_starts], [0.0] * len(raw_starts), "stub"

    _, lp_unpolished = _drive_to_the_sampler(
        config, user_params, unmoved, whitening.measure_and_whiten
    )
    fresh, lp_fresh = _drive_to_the_sampler(
        config, user_params, moved, whitening.measure_and_whiten
    )
    # the precondition: the polish moved the start, so a resume that kept
    # the build's own anchor would be told apart
    assert abs(lp_fresh - lp_unpolished) > 1e-3
    assert (tmp_path / "resume_whitening.json").exists()
    with open(tmp_path / "resume_nutsckpt.pkl", "wb") as fh:
        import pickle

        pickle.dump({"seed": 123}, fh)

    # ACT
    resumed, lp_resumed = _drive_to_the_sampler(
        config, user_params, must_not_run, must_not_run
    )

    # ASSERT
    assert lp_resumed == lp_fresh
    assert sorted(resumed) == sorted(fresh)
    for name in fresh:
        np.testing.assert_array_equal(resumed[name], fresh[name])


def _maps():
    with open("/proc/self/maps") as fh:
        return sum(1 for _ in fh)


@pytest.mark.skipif(
    not os.path.exists("/proc/self/maps"), reason="needs /proc"
)
@pytest.mark.slow
def test_chunks_do_not_pile_up_compiled_code(tmp_path, monkeypatch):
    """
    Given a fit checkpointed in 30 chunks,
    When it runs,
    Then the process's memory mappings stay flat from chunk to chunk.

    numpyro compiles its sampler again on every MCMC.run, and JAX keeps every
    compiled program: on the four-sector HAT-P-3 transit fit, 38 chunks of 10
    steps in one process exhausted the kernel's map limit (65530) and LLVM
    aborted the run ("Unable to allocate section memory").  The checkpointed
    sampler now clears JAX's caches after each chunk.
    """
    # ARRANGE
    counts = []
    real_save = nc._save

    def save_and_count(path, ck):
        real_save(path, ck)
        counts.append(_maps())

    monkeypatch.setattr(nc, "_save", save_and_count)
    path = str(tmp_path / "ck.pkl")

    # ACT
    with pm.Model():
        pm.MvNormal("x", np.zeros(5), np.eye(5))
        with nc.checkpointed_numpyro(path, 5, {"fp": 1}):
            _sample(pm.modelcontext(None), draws=100)

    # ASSERT
    growth = (counts[-1] - counts[5]) / (len(counts) - 1 - 5)
    assert len(counts) == 30
    assert growth < 20


def _write_rel(path, t0):
    """Relative astrometry whose errors make its likelihood ~flat
    (tests/test_direct_angles.py's)."""
    t = np.linspace(t0, t0 + 800.0, 12)
    phase = 2 * np.pi * (t - t0) / 400.0
    sep = 20.0 + 5.0 * np.cos(phase)
    pa = np.degrees(np.arctan2(np.sin(phase), 0.6 * np.cos(phase))) % 360.0
    np.savetxt(
        path,
        np.column_stack(
            [t, sep, np.full(t.size, 1e4), pa, np.full(t.size, 1e4)]
        ),
    )
    return str(path)


def _write_rv(path, t0):
    rng = np.random.default_rng(5)
    t = np.sort(rng.uniform(t0, t0 + 400.0, 20))
    np.savetxt(
        path,
        np.column_stack(
            [t, 50.0 * np.sin(2 * np.pi * t / 31.0), np.full(t.size, 1e4)]
        ),
    )
    return str(path)


@pytest.mark.slow
def test_a_resume_restores_the_direct_angle_choice(tmp_path):
    """
    Given a fresh numpyro fit whose ascending node the data pin to 0.1 deg,
      so the automatic switch samples it directly (exozippy.angles), and a
      NUTS checkpoint of that fit left beside its whitening,
    When the same config is run again,
    Then it reaches the sampler without polishing or probing, with the node
      still sampled directly, at the very start the fresh run sampled from.

    The switch is decided from the polished start, which a resume does not
    recompute, and it rebuilds the model, so the saved whitening describes
    the direct model: restoring it onto the latent build is refused.
    """
    # ARRANGE
    from exozippy import angles, whitening

    t0 = 2458000.0
    config = {
        "name": "resume_angles",
        "star": [{"name": "A", "mist": False}],
        "planet": [{"name": "BH"}],
        "orbit": [{"name": "BH"}],
        "astrometryinstrument": [
            {
                "name": "Rel",
                "file": _write_rel(tmp_path / "rel.astrom", t0),
                "mode": "rel",
            }
        ],
        "rvinstrument": [
            {"name": "inst", "file": _write_rv(tmp_path / "rv.dat", t0)}
        ],
        "prefix": str(tmp_path / "resume"),
        "sampler": {
            "method": "numpyro",
            "tune": 10,
            "draws": 10,
            "chains": 1,
            "cores": 1,
            "seed_polish": 50,
            "checkpoint_every": 5,
            "recompute_trace": True,
        },
    }
    user_params = {
        "star.A.mass": {"initval": 1.0, "sigma": 0.05},
        "star.A.radius": {"initval": 1.0, "sigma": 0.1},
        "star.A.distance": {"initval": 100.0},
        "planet.BH.mass": {"initval": 300.0},
        "planet.BH.radius": {"initval": 1.0, "sigma": 0},
        "orbit.BH.period": {"initval": 400.0},
        "orbit.BH.tc": {"initval": t0 + 100.0},
        "orbit.BH.secosw": {"initval": 0.2},
        "orbit.BH.sesinw": {"initval": 0.5},
        "orbit.BH.cosi": {"initval": 0.4},
        "orbit.BH.bigomega": {"initval": 150.0, "mu": 150.0, "sigma": 0.1},
    }

    def unmoved(model, raw_starts, **kwargs):
        kwargs["stops_out"].extend(("converged", "stub") for _ in raw_starts)
        return [dict(s) for s in raw_starts], [0.0] * len(raw_starts), "stub"

    def must_not_run(*args, **kwargs):
        raise AssertionError("a resume polished or probed again")

    adopted = []
    real_adopt = angles.adopt_direct_angles

    def recording_adopt(system, model, raw, choices):
        adopted.append(choices)
        return real_adopt(system, model, raw, choices)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("exozippy.run.adopt_direct_angles", recording_adopt)
        fresh, lp_fresh = _drive_to_the_sampler(
            config, user_params, unmoved, whitening.measure_and_whiten
        )
    # the precondition: the fresh run switched the node
    assert len(adopted) == 1 and adopted[0]
    assert not any("xbigomega" in name for name in fresh)
    with open(tmp_path / "resume_nutsckpt.pkl", "wb") as fh:
        import pickle

        pickle.dump(
            {
                "seed": 123,
                "direct_angles": angles.choices_to_json(adopted[0]),
            },
            fh,
        )

    # ACT
    resumed, lp_resumed = _drive_to_the_sampler(
        config, user_params, must_not_run, must_not_run
    )

    # ASSERT
    assert lp_resumed == lp_fresh
    assert sorted(resumed) == sorted(fresh)


def test_the_checkpoint_carries_the_fits_direct_angle_record(
    tmp_path, monkeypatch
):
    """
    Given a checkpointed numpyro fit handed its direct-angle record, and one
      handed none,
    When each saves its first chunk,
    Then the checkpoint returns the record it was handed, and None.
    """
    record = '{"orbit.bigomega": {"center": [2.6], "elements": [0]}}'
    real_save = nc._save

    def stop_at_first_save(p, payload):
        real_save(p, payload)
        raise _Stop()

    monkeypatch.setattr(nc, "_save", stop_at_first_save)
    for name, given in (("with", record), ("without", None)):
        path = str(tmp_path / f"{name}_nutsckpt.pkl")
        fp = nc.fingerprint({"model": "x"}, draws=60)
        with (
            nc.checkpointed_numpyro(path, 20, fp, direct_angles=given),
            pytest.raises(_Stop),
        ):
            _sample(_model())
        assert nc.checkpoint_direct_angles(path) == given
    assert nc.checkpoint_direct_angles(str(tmp_path / "absent.pkl")) is None
