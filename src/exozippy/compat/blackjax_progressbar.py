"""Workaround making ``nuts_sampler="blackjax"`` usable with pymc 6.1/6.2.

``pymc/sampling/jax.py::_blackjax_inference_loop`` has two independent defects
against blackjax 1.6.x, and both fire for *every* model -- a bare ``pm.Normal``
fails identically, which is why this lives in ``compat`` rather than in any
component's ``sampler_requirements()``.

1.  ``progress_bar`` leaks into the kernel::

        adapt = blackjax.window_adaptation(..., **adaptation_kwargs)   # (a)
        ...
        progress_bar = adaptation_kwargs.pop("progress_bar", False)    # (b)

    ``_sample_blackjax_nuts`` unconditionally sets
    ``nuts_kwargs["progress_bar"] = progressbar``, so it arrives in
    ``adaptation_kwargs``.  The pop at (b) is meant to remove it before
    blackjax sees it, but runs *after* (a) already forwarded it;
    ``window_adaptation`` funnels unknown kwargs through
    ``**extra_parameters`` into the algorithm's kernel::

        TypeError: build_kernel.<locals>.kernel() got an unexpected keyword
                   argument 'progress_bar'

2.  Once (1) is out of the way, pymc calls
    ``blackjax.progress_bar.gen_scan_fn(draws, progress_bar)``.  In blackjax
    1.6.2 ``blackjax.progress_bar`` is a *decorator function*, not a module::

        AttributeError: 'function' object has no attribute 'gen_scan_fn'

Rather than shim two upstream call sites separately, this module substitutes
one corrected copy of the function: identical to pymc's, minus the leak, with
``jax.lax.scan`` in place of the missing ``gen_scan_fn``.  The only behavioral
difference is that blackjax runs without a progress bar.

Retiring this: ``_is_affected`` detects the defects themselves, not a pymc
version, so the patch disables itself the moment either is fixed upstream.
Delete the module once both are.
"""

import inspect
import logging

logger = logging.getLogger(__name__)

_PATCH_FLAG = "_exozippy_blackjax_patched"


def _is_affected(func):
    """True when pymc's blackjax loop has either defect described above.

    Reads the *source*, so a fixed upstream stops matching without anyone
    having to bump a version check here.  Source that cannot be read (already
    wrapped, or compiled) is treated as unaffected: leaving a working sampler
    alone beats patching blind.
    """
    try:
        src = inspect.getsource(func)
    except (OSError, TypeError):
        return False

    # Defect 1: the pop that removes progress_bar runs after the call that
    # would forward it.
    adapt_at = src.find("window_adaptation(")
    pop_at = src.find('pop("progress_bar"')
    if pop_at < 0:
        pop_at = src.find("pop('progress_bar'")
    leaks_progress_bar = adapt_at >= 0 and pop_at >= 0 and pop_at > adapt_at

    # Defect 2: pymc reaches for blackjax.progress_bar.gen_scan_fn, which this
    # blackjax does not have.
    missing_gen_scan_fn = False
    if "progress_bar.gen_scan_fn" in src:
        try:
            import blackjax
            missing_gen_scan_fn = not hasattr(
                blackjax.progress_bar, "gen_scan_fn")
        except Exception:                  # pragma: no cover - optional dep
            missing_gen_scan_fn = False

    return leaks_progress_bar or missing_gen_scan_fn


def _inference_loop(seed, init_position, logp_fn, draws, tune, target_accept,
                    **adaptation_kwargs):
    """Corrected copy of pymc's ``_blackjax_inference_loop``.

    Same contract: returns ``(samples, stats)`` for one chain.
    """
    import jax
    import jax.numpy as jnp
    import blackjax
    from blackjax.adaptation.base import get_filter_adapt_info_fn

    # Defect 1: strip it before window_adaptation can forward it. We do not
    # render a progress bar at all, which is what defect 2 costs us.
    adaptation_kwargs.pop("progress_bar", None)

    algorithm_name = adaptation_kwargs.pop("algorithm", "nuts")
    if algorithm_name == "nuts":
        algorithm = blackjax.nuts
    elif algorithm_name == "hmc":
        algorithm = blackjax.hmc
    else:
        raise ValueError(
            "Only supporting 'nuts' or 'hmc' as algorithm to draw samples.")

    adapt = blackjax.window_adaptation(
        algorithm=algorithm,
        logdensity_fn=logp_fn,
        target_acceptance_rate=target_accept,
        adaptation_info_fn=get_filter_adapt_info_fn(),
        **adaptation_kwargs,
    )
    (last_state, tuned_params), _ = adapt.run(seed, init_position,
                                              num_steps=tune)
    kernel = algorithm(logp_fn, **tuned_params).step

    def _one_step(state, xs):
        _, rng_key = xs
        state, info = kernel(rng_key, state)
        stats = {
            "diverging": info.is_divergent,
            "energy": info.energy,
            "tree_depth": info.num_trajectory_expansions,
            "n_steps": info.num_integration_steps,
            "acceptance_rate": info.acceptance_rate,
            "lp": state.logdensity,
        }
        return state, (state.position, stats)

    keys = jax.random.split(seed, draws)
    # Defect 2: pymc would call blackjax.progress_bar.gen_scan_fn here; with
    # no progress bar that function is exactly jax.lax.scan.
    _, (samples, stats) = jax.lax.scan(
        _one_step, last_state, (jnp.arange(draws), keys))

    return samples, stats


def patch_blackjax_progress_bar():
    """Make ``nuts_sampler="blackjax"`` usable. Returns True if patched.

    Idempotent, and a no-op when pymc's JAX sampling module cannot be
    imported or when neither defect is present.
    """
    try:
        from pymc.sampling import jax as pmjax
    except Exception:                      # pragma: no cover - optional dep
        return False

    original = getattr(pmjax, "_blackjax_inference_loop", None)
    if original is None or getattr(original, _PATCH_FLAG, False):
        return False
    if not _is_affected(original):
        return False

    setattr(_inference_loop, _PATCH_FLAG, True)
    pmjax._blackjax_inference_loop = _inference_loop

    logger.info(
        "Applied the pymc/blackjax compatibility patch "
        "(exozippy.compat.blackjax_progressbar): blackjax runs without a "
        "progress bar.")
    return True
