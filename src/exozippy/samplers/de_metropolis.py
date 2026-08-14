"""Differential-evolution MCMC: the ``demc`` and ``demcz`` sampler keys.

Two gradient-free samplers, selected from the config's sampler block:

  * ``method: demc``  -- PyMC's population DE-MC (ter Braak 2006).  The
    chains ARE the differential-evolution population: each proposal is
    ``x_i + lambda*(x_j1 - x_j2) + eps``, so the population's own spread is
    the proposal generator and the population size must exceed the parameter
    count (see ``_common.resolve_n_chains``, shared with PTDE).
  * ``method: demcz`` -- PyMC's DEMetropolisZ (ter Braak & Vrugt 2008), which
    draws its difference vectors from each chain's OWN past history instead
    of from sibling chains, so it needs only a handful of chains -- but needs
    LONG ones, since that history is what has to fill.  Measured on
    ``examples/kelt4`` RV-only (15 parameters): at 500 tune + 500 draws it is
    still visibly unconverged (Rhat 3.9, chains stuck near their starts)
    where ``demc`` at the same length reaches Rhat 1.43; budget tens of
    thousands of iterations, and thin with ``nthin``.

Both are single-temperature.  ``ptde``/``ptde_async`` are the same DE move
run on a parallel-tempered ladder and remain the recommended non-HMC default;
these two exist because a plain DE-MC is the cheap, dependency-free control
to run a tempered result against, and because PyMC ships them.

Starts are the shared EXOZIPPy start population (``_common``), not PyMC's
default: without it every chain starts at the same relaxation-engine point,
every difference vector is exactly zero, and a DE population can only crawl
apart on its ``scaling``-sized jitter (0.001 raw units by default).

PyMC's own step classes are used directly.  There was a local subclass of
each carrying a ``scaling``/``lambda`` sampler-stat shape fix; the bug it
worked around was fixed upstream below this project's ``pymc>=6.0.0`` floor
(see the comment on that pin in ``pyproject.toml``), so the subclasses are
gone.  ``tests/test_de_metropolis.py`` samples with both variants and checks
those stats are scalar in the written trace, which is the thing the fix
protected.
"""

import logging
import time

import numpy as np
import pymc as pm

from . import _common

logger = logging.getLogger(__name__)

# The `method:` values this module answers to, mapped to their step class.
STEP_CLASSES = {"demc": pm.DEMetropolis, "demcz": pm.DEMetropolisZ}

# Chains for `demcz`, whose difference vectors come from its own history and
# so impose no population-size requirement. 4 is PyMC's own default and gives
# R-hat something to compare.
DEFAULT_DEMCZ_CHAINS = 4


def _wallclock_callback(maxtime, label):
    """A per-draw callback that interrupts sampling after ``maxtime`` s.

    ``pm.sample`` turns a ``KeyboardInterrupt`` raised in a callback into a
    graceful stop that keeps the draws taken so far -- the same mechanism
    run.py's NUTS branch uses.
    """
    started = time.time()

    def callback(trace, draw):
        if time.time() - started > maxtime:
            logger.info(f"{label}: wall-clock limit {maxtime:.0f}s reached")
            raise KeyboardInterrupt

    return callback


def de_metropolis_sample(
    model,
    system,
    draws,
    tune,
    *,
    variant="demc",
    chains=None,
    cores=None,
    raw_starts=None,
    seed_indices=None,
    raw_scales=None,
    maxtime=None,
    seed=None,
    plot_prefix=None,
    progressbar=True,
):
    """Sample ``model`` with a differential-evolution step method.

    Parameters
    ----------
    model : PyMC model (from ``system.build_model()``)
    system : EXOZIPPy System (start point + raw -> physical conversion)
    draws, tune : int
    variant : "demc" | "demcz"
    chains : int | None
        ``None`` means "size it for me": for ``demc`` that is the shared DE
        default ``2 * n_params`` (``_common.resolve_n_chains``, which also
        warns when an explicit population cannot span parameter space); for
        ``demcz`` it is ``DEFAULT_DEMCZ_CHAINS``.
    cores : int | None
        Passed to ``pm.sample``.  For ``demc`` (a PyMC *population* sampler)
        this parallelizes the population steppers rather than the chains.
    raw_starts, seed_indices, raw_scales
        Multi-seed starts and the measured whitening scales, forwarded to
        ``_common.resolve_start_population`` exactly as PTDE does.
    maxtime : float | None
        Wall-clock cap.  Honored for ``demcz``; PyMC's population path
        discards per-draw callbacks, so for ``demc`` it is warned about and
        ignored rather than silently dropped.
    seed : int | None
    plot_prefix : str | None
        When given, ensemble start plots are written under this prefix.
    progressbar : bool

    Returns
    -------
    arviz.InferenceData
    """
    variant = str(variant).lower()
    if variant not in STEP_CLASSES:
        raise ValueError(
            f"Unknown DE-MC variant {variant!r}; expected one of "
            f"{sorted(STEP_CLASSES)}."
        )
    label = variant.upper()

    rng = np.random.default_rng(seed)
    logp_fn = model.compile_logp()
    raw_start = system.get_raw_start(model)
    n_params = int(sum(np.size(v) for v in raw_start.values()))

    if variant == "demc":
        # The chains are the DE population, so this is the same sizing
        # question PTDE asks per rung -- one implementation, in _common.
        n_chains = _common.resolve_n_chains(chains, n_params, label, logger)
    else:
        n_chains = (
            DEFAULT_DEMCZ_CHAINS if chains is None else max(int(chains), 1)
        )
    logger.info(f"{label}: {n_params} params, {n_chains} chains")

    # Over-dispersed starts around the relaxation-engine solution (and around
    # every extra solved seed, round-robin) -- the same population PTDE's T=1
    # rung starts from.
    starts, chain_seed_index = _common.resolve_start_population(
        model,
        system,
        n_chains,
        logp_fn,
        rng,
        raw_start,
        raw_starts=raw_starts,
        seed_indices=seed_indices,
        raw_scales=raw_scales,
    )

    if plot_prefix is not None:
        _, raw_to_phys_batched, raw_var_names, out_var_names = (
            _common.compile_conversions(model)
        )
        _common.plot_start_ensemble(
            system,
            starts,
            raw_to_phys_batched,
            raw_var_names,
            out_var_names,
            plot_prefix,
            logger,
        )

    callback = None
    if maxtime is not None:
        if variant == "demc":
            logger.warning(
                f"{label}: maxtime={maxtime:.0f}s is IGNORED -- PyMC routes "
                f"population samplers through _sample_population, which "
                f"discards per-draw callbacks. Use method: demcz or "
                f"ptde_async for a wall-clock cap."
            )
        else:
            callback = _wallclock_callback(maxtime, label)

    with model:
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=n_chains,
            cores=cores,
            step=STEP_CLASSES[variant](),
            initvals=starts,
            callback=callback,
            progressbar=progressbar,
            random_seed=seed,
            return_inferencedata=True,
        )

    # Multi-seed provenance, same attr PTDE writes (_common).
    idata.posterior.attrs["chain_seed_index"] = list(chain_seed_index)
    if len(set(chain_seed_index)) > 1:
        logger.info(
            f"{label} multi-seed provenance (chain -> seed): "
            f"{list(chain_seed_index)}"
        )
    return idata
