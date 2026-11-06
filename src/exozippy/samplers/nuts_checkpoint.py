"""Checkpointed numpyro NUTS: a fit that stops mid-sampling resumes where it
stopped.

`pymc.sampling.jax.sample_jax_nuts` runs numpyro's warmup and draws in one call
and returns nothing until the end, so a fit stopped during sampling -- a
scheduler's wall-clock limit, a crash, a Ctrl-C -- loses every draw it took
(run.md: the trace is written only after sampling).  This module replaces the
one function that does the sampling, `pymc.sampling.jax._sample_numpyro_nuts`,
for the duration of a `with checkpointed_numpyro(...)` block, by a version with
the same signature and return value that:

* runs the warmup in chunks of `every` too, each chunk continuing the last
  (`MCMC.post_warmup_state`).  Every chunk's kernel is built with the FULL
  warmup length, so numpyro lays its adaptation windows out over all of it
  and keeps adapting while the state's step counter `i` is below it: the
  chunked warmup takes the same steps under the same schedule as one piece
  (a slow model's warmup alone can outlast a scheduler's limit);
* then draws in chunks of `every`, continuing each from the last state,
  and after each chunk saves the state, the draws and the sample stats so far.

On a rerun with the same fingerprint (the System's structural metadata and
every sampler setting that changes the chain) it loads the checkpoint, skips
what is done, and finishes; with a different fingerprint it raises rather than
splice two runs.  Everything around the sampling -- the start points, the
postprocessing into an InferenceData -- is still pymc's, so a checkpointed fit
and an uninterrupted one go through the same code on either side.

Chunked and resumed fits draw exactly what an uninterrupted pymc run with the
same seed draws: every chunk continues from the saved state's own random key,
and pymc splits the seed into chain keys the same way.  Measured bit for bit,
draws and step sizes, under both chain methods, with a stop and resume in the
warmup (tests/test_nuts_checkpoint.py).
"""

import contextlib
import hashlib
import json
import logging
import os
import pickle

import numpy as np

logger = logging.getLogger(__name__)

# The stats pymc asks numpyro for, in pymc's order (pymc/sampling/jax.py).
_EXTRA_FIELDS = (
    "num_steps",
    "potential_energy",
    "energy",
    "adapt_state.step_size",
    "accept_prob",
    "diverging",
)

CHECKPOINT_VERSION = 3


def fingerprint(structural, **settings):
    """A stable hash of what a checkpoint may be resumed under."""
    blob = json.dumps(
        {"structural": structural, "settings": settings},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(blob.encode()).hexdigest()


# Sampler keys that cannot change a chain's draws: where it is saved, whether
# it is resumed, how it is reported, and the seed (a resume adopts the one the
# checkpoint was started with; run.py).
_NOT_CHAIN_KEYS = frozenset(
    {
        "checkpoint_every",
        "recompute_trace",
        "profile",
        "cores",
        "seed",
        "polish_timeout",
        "eval_timeout",
        "maxtime",
    }
)


def chain_settings(sampler_cfg):
    """The sampler config minus the keys that cannot change a chain, sorted --
    for the fingerprint."""
    return sorted(
        (k, sampler_cfg[k]) for k in sampler_cfg if k not in _NOT_CHAIN_KEYS
    )


def start_logp_stamp(model):
    """The model's log posterior at Model.initial_point(), as a string to 9
    significant digits -- for the fingerprint."""
    lp = float(model.compile_logp()(model.initial_point()))
    return f"{lp:.9g}"


def _save(path, payload):
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def _load(path, fp):
    with open(path, "rb") as fh:
        payload = pickle.load(fh)
    if (
        payload.get("version") != CHECKPOINT_VERSION
        or payload.get("fingerprint") != fp
    ):
        raise RuntimeError(
            f"NUTS checkpoint {path} was written by a different fit (its "
            f"fingerprint does not match this config, params file or sampler "
            f"settings).  Delete it to start sampling afresh."
        )
    return payload


def _stats_to_dict(fields):
    """pymc's _numpyro_stats_to_dict, on extra fields gathered by chunk."""
    rename_key = {
        "potential_energy": "lp",
        "adapt_state.step_size": "step_size",
        "num_steps": "n_steps",
        "accept_prob": "acceptance_rate",
    }
    data = {}
    for stat, value in fields.items():
        name = rename_key.get(stat, stat)
        data[name] = np.asarray(value).copy()
        if stat == "num_steps":
            data["tree_depth"] = np.log2(value).astype(int) + 1
    return data


def _concat(parts):
    """Join chunks along the draw axis (axis 1; chains first).  Samples come
    as a list, one array per value variable (pymc's potential_fn form); the
    extra fields as a dict."""
    if isinstance(parts[0], dict):
        return {
            k: np.concatenate([np.asarray(p[k]) for p in parts], axis=1)
            for k in parts[0]
        }
    return [
        np.concatenate([np.asarray(p[i]) for p in parts], axis=1)
        for i in range(len(parts[0]))
    ]


def _fields(jax, run):
    """The scalar sample stats of a chunk, chains first."""
    return {
        k: v
        for k, v in jax.device_get(
            run.get_extra_fields(group_by_chain=True)
        ).items()
        if not isinstance(v, dict | tuple)
    }


def _chunk_summary(fields):
    """One line per chunk: each chain's mean and max leapfrogs per step, its
    last step size, and its divergences -- which chain is slow, and why."""
    steps = np.asarray(fields.get("num_steps"))
    size = np.asarray(fields.get("adapt_state.step_size"))
    div = np.asarray(fields.get("diverging"))
    if steps.ndim != 2:
        return ""
    parts = []
    for c in range(steps.shape[0]):
        parts.append(
            "chain %d: leapfrogs mean %.0f max %d, step %.3g, %d div"
            % (
                c,
                steps[c].mean(),
                steps[c].max(),
                size[c, -1],
                int(div[c].sum()),
            )
        )
    return "; ".join(parts)


def _drop_compiled(jax):
    """Release the programs JAX compiled for the chunk just run.

    numpyro compiles its sampler again on every MCMC.run (a fresh partial
    under pmap, a fresh loop body under fori_collect), and JAX keeps every
    compiled program.  One run per process never notices; a fit in chunks
    does: on the four-sector HAT-P-3 transit fit, 38 chunks of 10 steps in
    one process exhausted the kernel's map limit (65530) and LLVM aborted
    the run.  The next chunk compiles again either way, so nothing is lost.
    """
    jax.clear_caches()


def make_sampler(path, every, fp, direct_angles=None):
    """pymc.sampling.jax._sample_numpyro_nuts, checkpointed (a drop-in).

    ``direct_angles`` is the fit's direct-angle record (exozippy.angles,
    JSON text, or None when every angle keeps its latents), saved with the
    checkpoint so a resume can rebuild the model it was sampled in."""

    def _sample_numpyro_nuts(
        model,
        target_accept,
        tune,
        draws,
        chains,
        chain_method,
        progressbar,
        random_seed,
        initial_points,
        nuts_kwargs,
        logp_fn=None,
    ):
        import jax
        import numpyro
        from numpyro.infer import MCMC, NUTS
        from pymc.sampling.jax import get_jaxified_logp

        if logp_fn is None:
            logp_fn = get_jaxified_logp(model, negative_logp=False)
        nuts_kwargs = dict(nuts_kwargs)
        nuts_kwargs.setdefault("adapt_step_size", True)
        nuts_kwargs.setdefault("adapt_mass_matrix", True)
        nuts_kwargs.setdefault("dense_mass", False)
        kernel = NUTS(
            potential_fn=logp_fn,
            target_accept_prob=target_accept,
            **nuts_kwargs,
        )

        def _mcmc(n_samples):
            # num_warmup is always the full tune: a kernel initialized in a
            # fresh process takes its adaptation schedule from it.
            return MCMC(
                kernel,
                num_warmup=tune,
                num_samples=n_samples,
                num_chains=chains,
                postprocess_fn=None,
                chain_method=chain_method,
                progress_bar=progressbar,
            )

        if os.path.exists(path):
            ck = _load(path, fp)
            logger.info(
                "NUTS checkpoint %s: resuming after %d of %d warmup steps and "
                "%d of %d draws.",
                path,
                ck["warm"],
                tune,
                ck["done"],
                draws,
            )
        else:
            ck = {
                "version": CHECKPOINT_VERSION,
                "fingerprint": fp,
                "seed": int(random_seed),
                "direct_angles": direct_angles,
                "state": None,
                "warm": 0,
                "done": 0,
                "samples": [],
                "fields": [],
                # per-chunk sample stats of the warmup, which pymc's own
                # path discards: where a slow chain spent its leapfrogs
                "warm_fields": [],
            }

        runners = {}  # one MCMC per chunk length, so each compiles once
        if ck["state"] is None:
            # The first chunk starts the chains from pymc's initial points.
            # numpyro's own warmup() runs without collecting through this
            # same private setter; here it stops after n steps, not tune.
            n = int(min(every, tune))
            key = jax.random.PRNGKey(random_seed)
            if chains > 1:
                key = jax.random.split(key, chains)
            first = _mcmc(n)
            # Collect its n steps (as warmup(collect_warmup=True) would), for
            # the warmup stats below.
            first._set_collection_params(0, n, n, "warmup")
            first.run(
                key, init_params=initial_points, extra_fields=_EXTRA_FIELDS
            )
            ck["state"] = jax.device_get(first.last_state)
            ck["warm"] = n
            ck["warm_fields"].append(_fields(jax, first))
            _save(path, ck)
            _drop_compiled(jax)
            logger.info(
                "NUTS checkpoint %s: %d of %d warmup steps saved. %s",
                path,
                n,
                tune,
                _chunk_summary(ck["warm_fields"][-1]),
            )

        state = jax.tree_util.tree_map(jax.numpy.asarray, ck["state"])
        while ck["warm"] < tune or ck["done"] < draws:
            warming = ck["warm"] < tune
            if warming:
                n = int(min(every, tune - ck["warm"]))
            else:
                n = int(min(every, draws - ck["done"]))
            run = runners.setdefault(n, _mcmc(n))
            run.post_warmup_state = state
            # init_params = the saved positions: a kernel built in a fresh
            # process has no sample function until it is initialized, and
            # numpyro initializes it from init_params even when it then
            # continues from the warm state.
            run.run(
                state.rng_key, init_params=state.z, extra_fields=_EXTRA_FIELDS
            )
            state = run.last_state
            fields = _fields(jax, run)
            if warming:
                ck["warm"] += n
                ck["warm_fields"].append(fields)
            else:
                ck["samples"].append(
                    jax.device_get(run.get_samples(group_by_chain=True))
                )
                ck["fields"].append(fields)
                ck["done"] += n
            ck["state"] = jax.device_get(state)
            _save(path, ck)
            _drop_compiled(jax)
            logger.info(
                "NUTS checkpoint %s: %d of %d warmup steps, %d of %d draws "
                "saved. %s",
                path,
                ck["warm"],
                tune,
                ck["done"],
                draws,
                _chunk_summary(fields),
            )

        samples = _concat(ck["samples"])
        stats = _stats_to_dict(_concat(ck["fields"]))
        raw = [jax.numpy.asarray(v) for v in samples]
        return raw, stats, numpyro

    return _sample_numpyro_nuts


def checkpoint_seed(path):
    """The seed a saved checkpoint was started with, or None (no checkpoint).

    A resumed fit must stamp the seed its chains were STARTED with, not the
    one a seedless rerun draws afresh (run.py adopts this one)."""
    if not os.path.exists(path):
        return None
    with open(path, "rb") as fh:
        return int(pickle.load(fh)["seed"])


def checkpoint_direct_angles(path):
    """The direct-angle record (JSON text) a saved checkpoint was sampled
    under, or None (no checkpoint, or every angle kept its latents).

    run.py applies it BEFORE the build on a resume, as the reuse path
    applies a trace's (angles.restore_from_trace): the choice is made from
    the polished start, which a resume does not recompute."""
    if not os.path.exists(path):
        return None
    with open(path, "rb") as fh:
        return pickle.load(fh).get("direct_angles")


@contextlib.contextmanager
def checkpointed_numpyro(path, every, fp, direct_angles=None):
    """Swap pymc's numpyro sampling function for the checkpointed one."""
    import pymc.sampling.jax as pj

    original = pj._sample_numpyro_nuts
    pj._sample_numpyro_nuts = make_sampler(path, int(every), fp, direct_angles)
    try:
        yield
    finally:
        pj._sample_numpyro_nuts = original
