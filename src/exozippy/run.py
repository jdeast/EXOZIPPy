import contextlib
import gc
import importlib
import itertools
import logging
import multiprocessing as mp
import os
import signal
import time
import traceback
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pytensor
from matplotlib.backends.backend_pdf import PdfPages

from exozippy.samplers import convergence, de_metropolis
from exozippy.samplers.ptde import ptde_sample
from exozippy.samplers.ptde_async import ptde_async_sample
from exozippy.system import System

from .corner_utils import collect_corner_samples, save_corner_plot
from .diagnostics import ModelAuditor
from .logger import setup_logging
from .mkparam import write_param_file
from .outputs.modeling import build_modeling_output, compile_modeling_pdf
from .outputs.modes import DEFAULT_MAX_INVALID_FRAC, mode_suffix
from .outputs.report_pipeline import build_mode_reports
from .polish import polish_raw_starts, resolve_polish_steps
from .trace_meta import check_trace_freshness, stamp_structural_metadata
from .whitening import prepare_whitening

logger = logging.getLogger(__name__)

# Every key `_run_fit` reads off the `sampler:` block. Anything else is
# warned about and ignored, so this set must stay a superset of the keys the
# code actually consumes: a key missing here produces a warning saying it will
# be IGNORED about a key that is in fact HONORED, which is worse than no
# warning at all (that is how 'jitter' -- the very key the sample_jax_nuts
# comment tells users to opt back in with -- got flagged as unknown).
# tests/test_known_keys.py cross-checks this set against the `sampler_cfg`
# accesses in this module's own source, in both directions, so it cannot
# silently drift again. Add the key here in the same edit that consumes it.
KNOWN_SAMPLER_KEYS = {
    "init",
    "tune",
    "draws",
    "chains",
    "cores",
    "target_accept",
    "method",
    "n_temps",
    "T_max",
    "n_chains",
    "adapt_ladder",
    "recompute_trace",
    "nthin",
    "measure_scales",
    "profile",
    "min_ess",
    "max_rhat",
    "maxtime",
    "chain_method",
    "jitter",
    "eval_timeout",
    "rung_thin_factor",
    "rung_thin_start",
    "collect_rung_timing",
    "swap_schedule",
    "seed_polish",
    "store_hot_chains",
}


@contextlib.contextmanager
def sigterm_as_interrupt():
    """Map SIGTERM to Python's default SIGINT handler while sampling.

    A batch scheduler (`qsig -s SIGTERM <job_id>` / `kill -TERM <pid>`) can
    then interrupt sampling the same way a terminal Ctrl+C already does,
    instead of Python's default SIGTERM action (immediate termination with no
    partial trace saved). ``pm.sample`` already handles a KeyboardInterrupt
    raised mid-sampling gracefully -- that's exactly how the maxtime cutoffs
    work.

    Used by every branch that calls ``pm.sample`` directly: PyMC NUTS, the
    DE-Metropolis variants, and **nutpie**, which was missed until 2026-08
    and is the one that most needed saying out loud -- it reaches
    ``pm.sample`` through an EXTERNAL sampler, so whether the interrupt
    survives is nutpie's decision rather than PyMC's. It does:
    ``nutpie.sample`` catches ``KeyboardInterrupt`` and returns
    ``background_sampler.abort()``, i.e. the draws taken so far (verified
    against nutpie 0.16.11).

    The numpyro/blackjax branch is deliberately NOT wrapped, and that is not
    an oversight of the same kind: it goes through ``sample_jax_nuts``, whose
    chain runs inside one jitted scan with no Python frame to raise in --
    the same reason ``maxtime`` cannot be honored there (see
    ``warn_maxtime_unsupported``). The PTDE and async-PTDE samplers manage
    their own signal handling in ``samplers/_common.py``.
    """
    old_sigterm = signal.signal(signal.SIGTERM, signal.default_int_handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, old_sigterm)


@contextlib.contextmanager
def nonfatal_wrapup(what):
    """Run one wrap-up step; a crash inside it warns instead of aborting.

    Everything after ``pm.sample`` returns is a REPORT on a fit that already
    finished, and the fit's irreplaceable artifacts -- the trace, the mode
    report, the restart file -- are cheap to lose and expensive to recreate.
    The plotting block between the tables and ``write_param_file`` was the
    one stretch of bare calls in an otherwise wrapped wrap-up, so a
    degenerate-KDE crash inside ``save_multipage_trace`` (which any short or
    stopped run can provoke) skipped the restart file and the final
    paper.tex regeneration of a multi-day fit.

    Deliberately a broad ``except``: the point is that no diagnostic, from
    any component, may kill a finished fit, and enumerating the exception
    types a third-party plotting stack can raise is exactly the list that
    goes stale.  ``exc_info=True`` keeps the traceback in the log (and so in
    the GUI's status.json) rather than reducing the failure to one line --
    the alternative, and the reason this is warn-and-continue rather than
    swallow, is a wrap-up that silently produces fewer files than it should.

    KeyboardInterrupt and SystemExit are NOT caught (they are not
    ``Exception``): a user interrupting wrap-up wants it to stop.
    """
    try:
        yield
    except Exception:
        logger.warning("%s failed (non-fatal)", what, exc_info=True)


# Samplers that cannot honor `maxtime`, and why.  The three external NUTS
# backends run their whole chain outside Python's per-draw loop -- the JAX
# ones inside one jitted scan, nutpie inside Rust -- so there is no point at
# which a wall-clock check could raise the KeyboardInterrupt that the maxtime
# mechanism turns into a graceful stop.  PyMC agrees and says so out loud:
# pm.sample RAISES for a `callback` with any `nuts_sampler` but its own.
MAXTIME_UNSUPPORTED_METHODS = ("numpyro", "blackjax", "nutpie")


def warn_maxtime_unsupported(method, maxtime):
    """Say so when `maxtime:` cannot be honored by the selected sampler.

    A key that is silently ignored is worse than one that is refused: the
    whole point of `maxtime` is that a scheduler-bound job stops itself
    before the queue kills it, so a user who sets it and gets nothing has no
    partial trace AND no idea why.  demc already warns for exactly this
    reason (PyMC's population path discards per-draw callbacks); these three
    were the remaining silent ones.

    Returns True when a warning was emitted, so the check is exercisable
    without running a fit -- same shape as ``warn_unknown_sampler_keys``.
    """
    if maxtime is None or method not in MAXTIME_UNSUPPORTED_METHODS:
        return False
    logger.warning(
        f"{method}: maxtime={float(maxtime):.0f}s is IGNORED -- external NUTS "
        f"samplers run the chain outside Python's per-draw loop and invoke no "
        f"callback, so there is nothing to interrupt. Use method: nuts, "
        f"ptde_async or demcz for a wall-clock cap."
    )
    return True


# Sampler keys that only ONE method consumes.  A key here is silently inert
# under any other method: it is in KNOWN_SAMPLER_KEYS, so warn_unknown_sampler_keys
# says nothing, and the branch that would read it is never taken.
#
# That is the whole defect (review 2.4.2).  store_hot_chains is forwarded only
# to ptde_async, so under method: ptde the hot-chain mode discovery simply
# never runs and the user is told nothing; rung_thin_factor / rung_thin_start
# are the same thing mirrored -- ptde-only, silently ignored by ptde_async.
#
# Values are the methods that DO consume the key.
METHOD_ONLY_SAMPLER_KEYS = {
    "store_hot_chains": ("ptde_async",),
    "rung_thin_factor": ("ptde",),
    "rung_thin_start": ("ptde",),
}


def warn_method_only_sampler_keys(sampler_cfg, method):
    """Warn about keys the CHOSEN method does not consume.

    Only keys the user EXPLICITLY set are reported -- these all have defaults,
    and warning about a default nobody wrote would fire on every run and teach
    people to ignore the log.

    Must be called AFTER `method` is resolved and lowercased.  It deliberately
    does not live beside warn_unknown_sampler_keys, which runs early enough
    that `method` may still be None (auto-selection has not happened yet).

    Returns the sorted list of (key, method) pairs warned about, so the check
    is exercisable without running a fit -- same shape as its two siblings.
    """
    warned = []
    for key, consumers in sorted(METHOD_ONLY_SAMPLER_KEYS.items()):
        if key not in sampler_cfg or method in consumers:
            continue
        warned.append((key, method))
        logger.warning(
            f"{method}: sampler key '{key}' is IGNORED -- only "
            f"{' / '.join(consumers)} consume(s) it. Remove it, or switch "
            f"method to one of: {', '.join(consumers)}."
        )
    return warned


def warn_unknown_sampler_keys(sampler_cfg):
    """Warn about `sampler:` keys this module does not consume.

    Returns the sorted list of unrecognized keys (empty when all are known),
    so the check is exercisable without running a fit.
    """
    unknown = sorted(set(sampler_cfg) - KNOWN_SAMPLER_KEYS)
    if unknown:
        logger.warning(
            f"Unrecognized key(s) in the sampler block will be ignored: "
            f"{unknown}. "
            f"Did you mean 'method'? Valid sampler keys: {sorted(KNOWN_SAMPLER_KEYS)}"
        )
    return unknown


def run_fit(config, user_params=None):
    """The main library entry point to run an orbital fit.

    ``config`` is the system-config dict (what the CLI loads from the YAML
    file). ``user_params`` optionally supplies the parameter overrides as an
    in-memory dict; when omitted, ``config["parameter_file"]`` is read from
    disk as before. Data-file paths inside ``config`` are resolved relative
    to the current working directory, so a dict caller should chdir first
    (the same contract as solve_api).

    Thin wrapper around ``_run_fit`` that guarantees the GUI status file (when
    enabled via config["gui"]["snapshot"] or EXOZIPPY_GUI_SNAPSHOT=1) is left
    on a terminal phase on EVERY exit path -- a normal completion writes
    "done", a Ctrl+C / SIGTERM graceful abort writes "stopped", any other
    exception writes "error". A monitoring GUI therefore never sees the file
    stranded on a non-terminal phase after the process is gone. The reporter
    is a no-op when the flag is off, so ordinary non-GUI runs write nothing
    extra.
    """
    from exozippy.gui.status import GuiReporter
    from exozippy.pytensor_fallback import ensure_usable_backend

    # Probe the C toolchain before anything compiles: a missing g++ or
    # missing Python.h otherwise surfaces as a CompileError deep inside the
    # first pytensor.function call. Falls back to the (much slower)
    # pure-Python backend with a loud banner naming the fix.
    ensure_usable_backend()

    gui = GuiReporter.from_config(config)
    gui.phase("preparing")
    try:
        result = _run_fit(config, gui, user_params=user_params)
    except KeyboardInterrupt:
        # PTDE/NUTS raise KeyboardInterrupt for a during-tune or second-signal
        # abort; a graceful during-draws stop instead returns partial draws and
        # completes normally (-> "done" below).
        gui.terminal("stopped")
        raise
    except BaseException:
        # Record the traceback in the status file: the interpreter is gone by
        # the time a monitor sees phase "error", so this is the only place the
        # cause survives (e.g. a wrap-up crash after a graceful stop).
        gui.terminal("error", error=traceback.format_exc())
        raise
    gui.terminal("done")
    return result


def _run_fit(config, gui, user_params=None):
    """
    The main library entry point to run an orbital fit.
    """

    # 1. Prepare output directory
    prefix = Path(config.get("prefix", "fitresults/planet"))
    parent_dir = prefix.parent
    parent_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(prefix, config.get("logger_level", "INFO"))

    # 2. Load the sampler settings (flat under sampler:)
    sampler_cfg = config.get("sampler", {})
    init = sampler_cfg.get("init", "adapt_diag")
    tune = int(sampler_cfg.get("tune", 2000))
    draws = int(sampler_cfg.get("draws", 2000))
    chains = int(sampler_cfg.get("chains", 4))
    _cores_raw = sampler_cfg.get("cores", None)
    if _cores_raw is not None:
        cores = int(_cores_raw)
    else:
        _phys = mp.cpu_count()
        cores = max(1, min(int(_phys * 0.75), _phys - 1))
    target_accept = sampler_cfg.get("target_accept", 0.9)
    method = sampler_cfg.get(
        "method", None
    )  # None → auto-select after system is built
    # "auto" passes through to the sampler, which sizes the ladder from the
    # parameter count once the model is built (ptde.resolve_n_temps).
    # Barrier-equalizing ladder re-spacing during tune (Syed et al. 2022).
    # Off by default, matching the synchronous sampler; worth turning on when
    # ladder_health_report shows NON-UNIFORM per-rung swap acceptance, since a
    # round trip must cross every pair and more rungs cannot fix a badly
    # SHAPED ladder.
    adapt_ladder = bool(sampler_cfg.get("adapt_ladder", False))
    _n_temps_raw = sampler_cfg.get("n_temps", 8)
    n_temps = (
        _n_temps_raw if isinstance(_n_temps_raw, str) else int(_n_temps_raw)
    )
    T_max = float(sampler_cfg.get("T_max", 200.0))
    _n_chains_raw = sampler_cfg.get("n_chains", None)
    n_chains = int(_n_chains_raw) if _n_chains_raw is not None else None
    recompute_trace = sampler_cfg.get("recompute_trace", False)
    nthin = int(sampler_cfg.get("nthin", 1))
    # Data-driven whitening: probe each raw element's true local scale from
    # the start and rescale the model's whitening in place before sampling.
    # On by default; 'measure_scales: false' keeps the preliminary scales
    # (defaults.yaml init_scale or the span-fraction fallback).  It gates the
    # MEASUREMENT only -- reusing a trace still restores the whitening that
    # trace was sampled under, whatever this says.
    measure_scales = sampler_cfg.get("measure_scales", True)
    profile = sampler_cfg.get("profile", False)
    _min_ess_raw = sampler_cfg.get("min_ess", 1000)
    min_ess = int(_min_ess_raw) if _min_ess_raw is not None else None
    _max_rhat_raw = sampler_cfg.get("max_rhat", 1.01)
    max_rhat = float(_max_rhat_raw) if _max_rhat_raw is not None else None
    _maxtime_raw = sampler_cfg.get("maxtime", None)
    maxtime = float(_maxtime_raw) if _maxtime_raw is not None else None
    _eval_timeout_raw = sampler_cfg.get("eval_timeout", None)
    eval_timeout = (
        float(_eval_timeout_raw) if _eval_timeout_raw is not None else None
    )
    # Thinned hot-rung retention (ptde_async only): detector data for
    # post-hoc discovery of posterior-suppressed modes; see
    # outputs.ledger.discover_hot_modes.  "auto" | False | True (thin 20) |
    # int thin.  "auto" (the default) is resolved from the TOPOLOGY -- on for
    # microlensing, off otherwise -- in
    # samplers._common.resolve_store_hot_chains, which logs the decision and
    # its trace-size cost.  Passed through unresolved on purpose: the
    # component list does not exist yet at this point in run_fit.
    store_hot_chains = sampler_cfg.get("store_hot_chains", "auto")
    rung_thin_factor = int(sampler_cfg.get("rung_thin_factor", 1))
    _rung_thin_start_raw = sampler_cfg.get("rung_thin_start", None)
    rung_thin_start = (
        int(_rung_thin_start_raw) if _rung_thin_start_raw is not None else None
    )
    collect_rung_timing = bool(sampler_cfg.get("collect_rung_timing", False))
    swap_schedule = sampler_cfg.get("swap_schedule", "deo")
    if profile:
        pytensor.config.profile = True

    # Warn about unrecognized keys in the sampler block so they are never silently ignored.
    warn_unknown_sampler_keys(sampler_cfg)

    # 3. Build the stellar system into a PyMC Graph
    system = System(config, user_params=user_params)
    system.prepare()  # this triggers I/O
    gui.phase("compiling")  # build_model + get_mcmc_init compile the graph
    model = system.build_model()

    # Aggregate sampler requirements from all active components.
    # Components advertise incompatible/recommended samplers via sampler_requirements();
    # run.py stays agnostic about which component imposes the constraint.
    _incompatible, _recommended, _reasons = set(), set(), []
    for comp in system.active_components.values():
        reqs = comp.sampler_requirements()
        _incompatible.update(reqs.get("incompatible", set()))
        if "recommended" in reqs:
            _recommended.add(reqs["recommended"])
        if "reason" in reqs:
            _reasons.append(reqs["reason"])

    # sorted(), not next(iter(...)): _recommended is a set, so with two
    # components recommending different samplers the choice would be a
    # PYTHONHASHSEED coin flip -- i.e. a different sampler per run.  Only one
    # component recommends anything today (mulensing's Lens), so this is
    # inert; it stops being inert silently.
    if method is None:
        method = sorted(_recommended)[0] if _recommended else "nuts"
    elif method.lower() in _incompatible:
        rec_str = sorted(_recommended)[0] if _recommended else "ptde_async"
        reason_str = (
            "; ".join(_reasons) if _reasons else "incompatible with this model"
        )
        logger.warning(
            f"Sampler '{method}' cannot be used with this model ({reason_str}). "
            f"Set 'method: {rec_str}' in the sampler block."
        )
    method = method.lower()
    warn_method_only_sampler_keys(sampler_cfg, method)

    # First modeling-draft checkpoint: the components declared their prose
    # during stages 1-7 and the sampler is now resolved, so the citation
    # scaffold (<prefix>_paper.tex) can be written BEFORE sampling --
    # the user keeps it even if the fit dies hours in.  Regenerated (not
    # appended) at wrap-up with the results/convergence/figures/table
    # sections. Never fatal: the draft is a bonus deliverable.
    _add_sampler_prose(system, method, swap_schedule=swap_schedule)
    try:
        build_modeling_output(system, prefix)
    except Exception:
        logger.warning(
            "modeling-draft generation failed (non-fatal)", exc_info=True
        )

    # 4. Sample
    # We use adapt_diag to start exactly at our estimated means
    with model:
        # Build the raw starting point explicitly: 0 for logit params,
        # (initval - mu)/sigma for Gaussian-path params, so the physical
        # start is always our initval.
        raw_start = system.get_raw_start(model)

        # Data-driven whitening: measure every raw element's true local
        # scale from the relaxation-engine start and rescale the model's
        # whitening in place (Parameter.set_whitening), then measure the
        # soft-bound barrier steepness scales the same way.  After this, a
        # unit step along any raw direction costs ~0.5 nats -- the
        # "curvature = -1" conditioning the retired curvature check used to
        # ask users to approximate by hand-tuning init_scale.
        #
        # The state is persisted next to the trace: a reload
        # (recompute_trace: false with an existing trace) restores the exact
        # scales the trace was sampled with instead of re-probing -- raw
        # draws only decode correctly under the whitening they were sampled
        # under.  On a FRESH run a model mismatch falls back to a fresh
        # measurement (nothing is sampled yet, so the coordinates are still
        # a free choice); on the reuse path they are not, and a mismatch
        # raises instead -- see whitening.restore_whitening_for_trace.
        trace_path = str(prefix) + "_trace.nc"
        whitening_path = str(prefix) + "_whitening.json"
        reusing_trace = os.path.exists(trace_path) and not recompute_trace

        # Pre-whitening seed polish (sampler-config `seed_polish`): promote
        # each solution-estimate start to its basin's optimum BEFORE the
        # whitening probe measures scales around it.  A start far below its
        # optimum makes the probe gradient-dominated (scales come out
        # orders of magnitude too tight -- examples/ob140939 measured
        # ~1000x tight from a start ~5900 nats low and diverged on 86% of
        # its draws), freezes the barrier steepness against dishonest unit
        # steps, and starts every sampler outside the typical set.  'auto'
        # (default) polishes the canonical single start and MMEXOFAST seed
        # sets, never multi-seed posterior-draw restart sets (already at
        # equilibrium; polishing would collapse their overdispersion);
        # on/off/int override (int = step count).  L-BFGS on logp+grad
        # when the model is differentiable, the PR #56 T=1 DE polish
        # otherwise.  Skipped when reusing an existing trace (nothing will
        # be sampled).  This supersedes the PTDE-internal seed polish,
        # which ran after the probe and only under PTDE.
        if not reusing_trace:
            raw_starts_pre, seed_indices_pre = system.get_raw_starts(model)
            polish_steps = resolve_polish_steps(
                sampler_cfg.get("seed_polish", "auto"),
                n_seeds=len(raw_starts_pre),
                has_seed_hints=bool(
                    getattr(system.config_manager, "seed_hint_sets", None)
                ),
            )
            if polish_steps:
                polished, _dlps, _pmethod = polish_raw_starts(
                    model,
                    raw_starts_pre,
                    n_steps=polish_steps,
                    seed_indices=seed_indices_pre,
                    # Same core grant the sampler is about to use: the DE
                    # engine ran serial here while every one of them sat
                    # idle, which on a 64-core microlensing job was the
                    # whole polish stage at 1/64 throughput.
                    cores=cores,
                )
                system.apply_polished_starts(polished, seed_indices_pre)
                raw_start = system.get_raw_start(model)

        whiten_report = None
        # Fresh run -> measure + persist.  Reuse -> restore only: the
        # whitening is a property of the draws being decoded, so it is
        # never re-measured and never rewritten there (a mismatch
        # raises StaleWhiteningError, a missing file warns and keeps
        # the preliminary scales).  The old code fell back to
        # measure + save on BOTH, silently re-coordinating the trace it
        # was reusing and overwriting the only record of how to read it.
        #
        # `measure_scales: false` gates only the MEASUREMENT, and cannot gate
        # the restore: it asks "do not probe this run's start", which is a
        # statement about a run that is about to sample.  A trace sampled
        # under measured scales and reloaded with the key off would otherwise
        # decode its raw draws under preliminary scales, silently and with no
        # message -- the exact failure restore_whitening_for_trace exists to
        # prevent.  The reuse path's honest answer when the trace really was
        # sampled without measurement is already built in: no whitening file
        # exists, so the restore warns and keeps the preliminary scales, which
        # for that trace ARE the sampled coordinates.
        if reusing_trace or measure_scales:
            whiten_report = prepare_whitening(
                system,
                model,
                raw_start,
                whitening_path,
                trace_path,
                reusing_trace=reusing_trace,
            )
            if whiten_report is not None:
                # The rescale re-expressed a polished (nonzero) start in
                # the new raw coordinates; re-read it for every consumer
                # below (a no-op for an unpolished all-zeros start).
                raw_start = system.get_raw_start(model)

        # 1. Get your starting dictionaries (after the rescale, so the
        # diagnostic table reports the measured scales)
        transformed_inits = system.get_mcmc_init(model)
        inspect_start(
            model,
            system,
            transformed_inits,
            whiten_report=whiten_report,
        )

        # Multi-seed starts (P4): a list of raw start dicts (one per solved
        # seed) plus their original seed indices. seed 0 == raw_start above;
        # get_raw_starts returns just [raw_start], [0] for the ordinary case.
        # After a polish, seed 0 comes from the polished raw_initval and
        # seeds k>0 are re-derived from their polished physical values.
        raw_starts, seed_indices = system.get_raw_starts(model)

        # Seeded-solution ledger (multi-seed fits only): a Laplace record
        # of every polished seed -- peak logp and curvature widths at the
        # basin optimum -- measured NOW, while the seeds exist, so the
        # final report can distinguish "considered and rejected at
        # delta lp = X" from "never looked" even for modes the T=1
        # posterior abandons entirely. Costs n_seeds x n_params x O(15)
        # logp calls on the freshly whitened model. Disable with config
        # `modes: {ledger: false}`.
        # (Skipped when reusing an existing trace: the polish was skipped
        # there too, so seed lp would be a start value, not a basin peak.)
        seed_ledger = None
        _ledger_on = (config.get("modes", {}) or {}).get("ledger", True)
        if len(raw_starts) > 1 and _ledger_on and not reusing_trace:
            from .outputs.ledger import build_seed_ledger

            try:
                seed_ledger = build_seed_ledger(
                    system, model, raw_starts, seed_indices
                )
            except Exception:
                logger.warning(
                    "Seed ledger measurement failed; final report will "
                    "not carry rejected-mode records",
                    exc_info=True,
                )

        # convert raw starting point to the internal starting point
        internal_start = system.get_internal_point(model, raw_start)

        # make all the component plots
        for comp in system.active_components.values():
            comp.plot(
                system,
                [internal_start],
                filename_prefix=str(prefix) + "_start",
            )

        if profile:
            func = model.logp_dlogp_function(profile=True)
            func.profile.summary()

        if reusing_trace:
            # if we've already done the sampling and don't want to redo it, load it
            idata = az.from_netcdf(trace_path)
            # ...but only if it was sampled from THIS model. The raw draws
            # decode through this build's bounds/links/whitening, so a trace
            # from an edited config would be relabeled and reported as if it
            # belonged here. Unlike the whitening reload above, which can
            # honestly re-measure on a mismatch, there is no load-time repair
            # for foreign draws: a mismatch raises (trace_meta).
            check_trace_freshness(idata, system, trace_path)
        else:
            # do the sampling and save the results
            gui.phase("sampling")
            if method in ("numpyro", "blackjax", "nutpie"):
                try:
                    importlib.import_module(method)
                except ImportError:
                    logger.warning(
                        f"{method} is not installed — falling back to PyMC NUTS. "
                        f"Install with: poetry install --extras jax"
                    )
                    method = "nuts"

            # Placed AFTER the import fallback above, so a config asking for
            # numpyro on a box without it -- which lands on PyMC NUTS, where
            # maxtime IS honored -- is not warned about a limit that will be
            # applied.
            warn_maxtime_unsupported(method, maxtime)

            if method == "ptde":
                idata = ptde_sample(
                    model,
                    system,
                    draws,
                    tune,
                    n_temps=n_temps,
                    T_max=T_max,
                    n_chains=n_chains,
                    cores=cores,
                    raw_starts=raw_starts,
                    seed_indices=seed_indices,
                    raw_scales=(
                        whiten_report["raw_scales"] if whiten_report else None
                    ),
                    plot_prefix=str(prefix),
                    min_ess=min_ess,
                    max_rhat=max_rhat,
                    maxtime=maxtime,
                    eval_timeout=eval_timeout,
                    rung_thin_factor=rung_thin_factor,
                    rung_thin_start=rung_thin_start,
                    collect_rung_timing=collect_rung_timing,
                    swap_schedule=swap_schedule,
                    adapt_ladder=adapt_ladder,
                    progress_callback=gui.progress_callback,
                )
            elif method == "ptde_async":
                # The non-blocking PTDE dispatch loop (see samplers.md; the
                # hpc_optimization.txt prompt it used to cite was pruned
                # PROMPT 13) -- the recommended default for Op-based models;
                # see exozippy/samplers/ptde_async.py's module docstring for
                # the stale-DE-partner caveat and how swaps stay rigorous.
                # rung_thin_factor/rung_thin_start are ptde-only (thinning
                # addresses the blocking problem that async dispatch removes
                # outright) and are not forwarded here.
                idata = ptde_async_sample(
                    model,
                    system,
                    draws,
                    tune,
                    store_hot_chains=store_hot_chains,
                    n_temps=n_temps,
                    T_max=T_max,
                    n_chains=n_chains,
                    cores=cores,
                    raw_starts=raw_starts,
                    seed_indices=seed_indices,
                    raw_scales=(
                        whiten_report["raw_scales"] if whiten_report else None
                    ),
                    plot_prefix=str(prefix),
                    min_ess=min_ess,
                    max_rhat=max_rhat,
                    maxtime=maxtime,
                    eval_timeout=eval_timeout,
                    collect_rung_timing=collect_rung_timing,
                    swap_schedule=swap_schedule,
                    progress_callback=gui.progress_callback,
                )
            elif method in ("numpyro", "blackjax"):
                import jax

                jax.config.update("jax_enable_x64", True)
                if method == "blackjax":
                    # pymc forwards its own progress_bar kwarg into blackjax's
                    # kernel, which raises for every model. See
                    # exozippy/compat/blackjax_progressbar.py; self-retiring.
                    from .compat import patch_blackjax_progress_bar

                    patch_blackjax_progress_bar()
                from pymc.sampling.jax import sample_jax_nuts

                chain_method = sampler_cfg.get("chain_method", "parallel")
                # jitter=False: the JAX samplers default to jittering each
                # chain by U(-1, 1) in raw (whitened) space, i.e. +/- one
                # whitening scale per parameter.  We deliberately construct
                # the start from the relaxation-engine solution.  With
                # measured scales the historical failure mode (a scale much
                # wider than the posterior launching chains at logp ~ -1e6,
                # collapsing the step size to zero) mostly goes away, but
                # with 'measure_scales: false' the preliminary scales bring
                # it right back -- so the default stays off.  Opt back in
                # with 'jitter: true'.
                idata = sample_jax_nuts(
                    draws=draws,
                    tune=tune,
                    chains=chains,
                    target_accept=target_accept,
                    initvals=internal_start,
                    jitter=sampler_cfg.get("jitter", False),
                    chain_method=chain_method,
                    nuts_sampler=method,
                )
            elif method == "nutpie":
                # nutpie ignores initvals; it uses init_mean: a flat float64
                # array in model.free_RVs order (raw/unconstrained space).
                nutpie_init_mean = np.concatenate(
                    [
                        np.asarray(raw_start[v.name], dtype=float).ravel()
                        for v in model.free_RVs
                    ]
                )
                # Wrapped like the other two pm.sample branches, and it pays
                # off here: nutpie.sample catches a KeyboardInterrupt and
                # returns `background_sampler.abort()`, i.e. the draws taken
                # so far (verified against nutpie 0.16.11).  So a scheduler
                # SIGTERM buys a partial trace instead of an immediate kill,
                # which is exactly what this context manager is for.
                with sigterm_as_interrupt():
                    idata = pm.sample(
                        draws=draws,
                        tune=tune,
                        chains=chains,
                        nuts_sampler="nutpie",
                        target_accept=target_accept,
                        nuts_sampler_kwargs={"init_mean": nutpie_init_mean},
                        cores=cores,
                        return_inferencedata=True,
                    )
            elif method in de_metropolis.STEP_CLASSES:
                # Gradient-free differential-evolution MCMC (ter Braak 2006 /
                # ter Braak & Vrugt 2008) on PyMC's own step methods, started
                # from the same over-dispersed population PTDE's T=1 rung
                # uses -- see samplers/de_metropolis.py.  `chains` is read
                # off the raw config rather than the resolved default so an
                # unset key can mean "size the DE population from the
                # parameter count" (2 x n_params) instead of silently
                # accepting the generic 4.
                with sigterm_as_interrupt():
                    idata = de_metropolis.de_metropolis_sample(
                        model,
                        system,
                        draws,
                        tune,
                        variant=method,
                        chains=sampler_cfg.get("chains", None),
                        cores=cores,
                        raw_starts=raw_starts,
                        seed_indices=seed_indices,
                        raw_scales=(
                            whiten_report["raw_scales"]
                            if whiten_report
                            else None
                        ),
                        maxtime=maxtime,
                        plot_prefix=str(prefix),
                    )
            else:
                nuts_callback = None
                if maxtime is not None:
                    _nuts_start = time.time()

                    def nuts_callback(trace, draw):
                        if time.time() - _nuts_start > maxtime:
                            logger.info(
                                f"NUTS: wall-clock limit {maxtime:.0f}s reached"
                            )
                            raise KeyboardInterrupt

                step = pm.NUTS(target_accept=target_accept)
                with sigterm_as_interrupt():
                    idata = pm.sample(
                        draws=draws,
                        tune=tune,
                        chains=chains,
                        init=init,
                        step=step,
                        cores=cores,
                        return_inferencedata=True,
                        callback=nuts_callback,
                    )
            if nthin > 1:
                idata = idata.sel(draw=slice(None, None, nthin))
            # Record the storage thinning on the trace.  Consecutive stored
            # draws that are really nthin sampler steps apart make mode
            # changes look more independent than they are, so outputs.modes
            # must be told rather than left to assume 1 (see
            # ModeReport.thin_factor / thin_known).
            idata.posterior.attrs["nthin"] = int(nthin)
            # Ensure lp is in sample_stats; compute and persist if missing,
            # so the archived trace carries it and no later reader (modes,
            # mkparam, the plotters) has to recompute it.
            _ensure_lp(idata, model)
            # Convert sampled variables to user-facing units before archiving.
            # This makes the trace file, trace plots, ArviZ summary, and
            # mkparam output all use the same units the user specified.
            _convert_posterior_to_user_units(
                idata, system.get_parameter_lookup()
            )
            _sanitize_netcdf_attrs(idata)
            # Stamp the structural fingerprint of the config + params that
            # produced these draws, so any later reload can verify it.
            stamp_structural_metadata(idata, system)
            idata.to_netcdf(trace_path)

    # Sampling is done; the rest is post-processing + report/plot output.
    gui.phase("writing")

    # Collapse any exact label degeneracy a component declares (review
    # 1.8.3's ascending node is the one case today).  HERE, and exactly once:
    # this is the seam both paths -- fresh draws and a reloaded trace --
    # arrive at, and everything below consumes what it leaves.  Folding per
    # consumer is how the convergence check, the mode reporter and the seed
    # ledger come to disagree about how many solutions a chain found.
    refolded = system.fold_degenerate_draws(idata, model)
    if refolded:
        # The regenerated deterministics come back in INTERNAL units --
        # PyMC recomputes them from the model graph, which knows nothing
        # about the conversion already applied to the rest of the
        # posterior -- so put just those back in the user's units.
        _convert_posterior_to_user_units(
            idata, system.get_parameter_lookup(), only=refolded
        )

    # Post-hoc burn-in + stuck-chain trimming (samplers/convergence.py). We
    # keep the FULL, untrimmed trace on disk (idata.to_netcdf above / the
    # loaded .nc) so any reanalysis can recompute this, but every downstream
    # report -- mode ID, medians/CIs, corner, trace plots -- runs on the
    # trimmed view so the initial transient never biases the science. This is
    # the fix for the DC2018_128 pathology (notes/todo.txt): the reported
    # summary previously discarded zero burn-in even though a likelihood-flat
    # degenerate direction drifted for ~half the run.
    # Hot-chain mode discovery (store_hot_chains): cluster the thinned
    # hot-rung draws, polish each new basin's best point, and append
    # Laplace records to the seed ledger -- so a mode the T=1 posterior
    # never held still shows up as "considered and rejected" in the final
    # report. Runs BEFORE burn-in trimming (hot draws are detectors; no
    # burn-in semantics) and tolerates a missing/empty group.
    # The outcome is recorded in `hot_status` and rendered into
    # <prefix>_modes.txt: "searched and found nothing", "never searched"
    # (what a non-microlensing topology gets by default) and "the search
    # crashed" used to be indistinguishable in every output the user reads,
    # which turns a silent failure into false assurance that a candidate
    # mode was considered.  The catch stays broad and stays
    # NON-FATAL -- a wrap-up diagnostic must not kill a finished multi-day
    # fit -- but the exception type and message now reach the report.
    from .outputs.ledger import run_hot_mode_discovery

    seed_ledger, hot_status = run_hot_mode_discovery(
        system, model, idata, seed_ledger
    )

    idata, burn_diag = convergence.analyze_idata(
        idata, min_ess=min_ess, max_rhat=max_rhat
    )
    convergence.log_convergence(burn_diag, logger)

    # Identify posterior modes, distribute the posterior onto the Parameter
    # objects, and write the mode report + LaTeX/CSV tables. Shared with the
    # exozippy-modes CLI (outputs/report_pipeline.py) so reprocessing a saved
    # trace can never drift from what a live fit produces. A live fit must
    # not silently emit final tables from a numerically broken run (hence
    # raise_on_invalid=True, overridable via config
    # `modes: {max_invalid_frac: ..., force: true}`), and may opt into
    # per-mode evidence weighting via `modes: {weights: evidence}`.
    modes_cfg = config.get("modes", {}) or {}
    mode_report = build_mode_reports(
        system,
        idata,
        prefix,
        model=model,
        trace_path=trace_path,
        max_invalid_frac=modes_cfg.get(
            "max_invalid_frac", DEFAULT_MAX_INVALID_FRAC
        ),
        force=modes_cfg.get("force", False),
        raise_on_invalid=True,
        evidence_weights=str(modes_cfg.get("weights", "")).lower()
        == "evidence",
        seed_ledger=seed_ledger,
        hot_status=hot_status,
    )

    summary_path = Path(str(prefix) + "_summary.txt")
    summary_path.write_text(
        _format_summary(idata, burn_diag), encoding="utf-8"
    )

    # Every plot below is wrapped, and per COMPONENT rather than per loop, so
    # one component's broken diagnostic costs its own figure and nothing else
    # -- neither its siblings' figures nor, further down, the restart file.
    # make a corner plot of fitted parameters (similar to EXOFASTv2 covar plot)
    with nonfatal_wrapup("corner plot"):
        make_corner(model, idata, str(prefix) + "_corner.png")

    # Component-specific corner plots (e.g. mulensing geometry). Unlike
    # comp.plot(), which also runs pre-flight on a single point, this only
    # runs here, once, when the full posterior (idata) actually exists.
    for comp in system.active_components.values():
        with nonfatal_wrapup(f"corner plot for {comp.label}"):
            comp.plot_corner(idata, filename_prefix=str(prefix))

    # Save a 1D trace plot (similar to EXOFASTv2 chain file)
    with nonfatal_wrapup("detailed trace plot"):
        all_params = system.get_all_parameters()
        plot_vars = [
            p.label for p in all_params if p.label in idata["posterior"]
        ]
        save_multipage_trace(
            idata, plot_vars, str(prefix) + "_trace_detailed.pdf", model=model
        )

    # Pick the suspected troublemakers
    # List every tracked parameter in the posterior
    # available_vars = list(idata.posterior.data_vars)
    # print("All available variables:\n", available_vars)

    # Automatically filter for the ones we care about
    # vars_to_check = [v for v in available_vars if any(sub in v for sub in ['secosw', 'sesinw', 'ecc', 'omega', 'mass'])]
    # print("\nFiltered variables to plot:\n", vars_to_check)
    # az.plot_pair(
    #    idata,
    #    var_names=vars_to_check,
    #    kind='scatter',
    #    divergences=True,
    #    divergences_kwargs={'color': 'C3', 'alpha': 0.5, 'markersize': 5}  # C3 is usually red
    # )
    # plt.show()

    # Generate final plots.  `draws` outlives this block -- the modeling
    # draft reads draws[0] for its model-bearing figures -- so it is seeded
    # empty first: a get_draws failure must degrade the draft to its
    # data-only specs, not NameError past the wrap.
    draws = []
    with nonfatal_wrapup("posterior draw extraction"):
        draws = get_draws(idata, param_lookup=system.get_parameter_lookup())
    for comp in system.active_components.values():
        with nonfatal_wrapup(f"posterior plots for {comp.label}"):
            comp.plot(system, draws, filename_prefix=str(prefix) + "_mcmc")

    # Multimodal posteriors: re-emit the same corner + component plots once
    # per mode, restricted to that mode's draws (interim solution -- see
    # notes/multimode_implementation.txt P7; a recolored/stratified single
    # figure is deferred). Per-mode LaTeX columns and CSV rows are already
    # produced above via mode_report=mode_report; this loop only covers the
    # plot outputs, which have no such mechanism. Single-mode runs take this
    # branch never, so they emit zero new files.
    if mode_report is not None and mode_report.n_modes > 1:
        try:
            _emit_per_mode_outputs(system, model, idata, mode_report, prefix)
        except Exception:
            logger.warning(
                "Per-mode output generation failed; the combined "
                "posterior outputs above are unaffected",
                exc_info=True,
            )

    # Final modeling-draft checkpoint: the table fragments and posterior
    # plots now exist on disk and the convergence/mode facts are known, so
    # regenerate <prefix>_paper.tex with its Results sections and
    # (config `modeling: {compile: false}` to opt out) compile the draft
    # PDF.  Compile failure or missing TeX never fails the fit.
    modeling_cfg = config.get("modeling", {}) or {}
    for _key in modeling_cfg:
        if _key != "compile":
            logger.warning(
                f"Unrecognized key '{_key}' in the modeling block will be "
                f"ignored (known: compile)."
            )
    try:
        _add_wrapup_prose(system, burn_diag, mode_report)
        # One posterior draw unlocks the model-bearing plot specs (phased
        # panels), whose figures otherwise never enter the draft.
        tex_path = build_modeling_output(
            system, prefix, point=draws[0] if draws else None
        )
        if modeling_cfg.get("compile", True):
            compile_modeling_pdf(tex_path)
    except Exception:
        logger.warning(
            "modeling-draft generation failed (non-fatal)", exc_info=True
        )

    try:
        # mkparam re-derives the structural fingerprint from this config and
        # the params, not from the live System; measured to reproduce the
        # System snapshot exactly (see the note at the check inside
        # mkparam.write_param_file).  The params half has to be handed over
        # when run_fit was called with an in-memory dict: the file at
        # config['parameter_file'] is then not what was fitted (it may be
        # stale, or absent), and write_param_file would merge ITS priors and
        # bounds into the restart file.  Left None for a file-driven run, so
        # that path still reads the file itself and its error messages still
        # name it.
        write_param_file(
            config, trace_path=trace_path, user_params=user_params
        )
    except Exception:
        logger.exception("mkparam failed (non-fatal)")


def _user_initval(config_manager, par, index):
    """Start value (INTERNAL units) for element ``index`` of ``par`` as spelled
    in the user/solved parameter table, or None if the table does not set one.

    The key lookup mirrors ``ConfigManager.resolve``: the same three forms
    (``comp.param``, ``comp.<index>.param``, ``comp.<name>.param``) in the same
    precedence (later wins), each canonicalized through
    ``ConfigManager.canonical_key``.  Without that canonicalization a NAMED
    instance never matches, because ``standardize_param_names`` stores every
    entry in index form -- so the lookup used to succeed or fail depending on
    whether the user had named their components.
    """
    prefix = par.label.split(".")[0]
    attr = par.label.split(".")[-1]
    candidates = [
        f"{prefix}.{attr}",
        f"{prefix}.{index}.{attr}",
        par.get_display_label(index),
    ]

    val = None
    for key in candidates:
        entry = config_manager.user_params.get(
            config_manager.canonical_key(key)
        )
        if entry is None:
            continue
        if not isinstance(entry, dict):
            entry = {"initval": entry}  # bare scalar, as resolve() treats it
        if "initval" not in entry:
            continue
        v = entry["initval"]
        # List-valued initvals are per-seed starts (P4); seed 0 is canonical.
        if isinstance(v, (list, tuple)):
            v = v[0] if len(v) else None
        if v is not None:
            val = v

    if val is None:
        return None
    try:
        return float(par.to_internal(float(val), index=index))
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _is_unset(x):
    """True when a start-value element carries no usable number."""
    if x is None:
        return True
    try:
        return bool(np.isnan(float(x)))
    except (TypeError, ValueError):
        return False


def inspect_start(
    model,
    system,
    transformed_inits,
    whiten_report=None,
):
    # No physical inits/scales arguments: this reads p.initval / p.init_scale
    # off the Parameters below, so the two dicts get_mcmc_init used to build
    # and hand over were never read.
    auditor = ModelAuditor(model, system, transformed_inits)
    param_logps, other_nodes = auditor.get_aggregated_logps()
    unused_yaml = auditor.check_unused_yaml()

    # Map the whitening probe's measured multipliers (one per SAMPLED element,
    # keyed by raw-variable name) back to full per-parameter element vectors.
    # Used only for flat detection: NaN marks a direction the probe found
    # flat (logp ignores it), which keeps its preliminary scale and earns a
    # warning after the table.
    mult_map = {}
    if whiten_report is not None:
        multipliers = whiten_report.get("multipliers", {})
        for p in auditor.all_params:
            m = multipliers.get(f"{p.label}_raw")
            if m is None:
                continue
            n_elements = np.prod(p.shape).astype(int) if p.shape != () else 1
            full = np.full(n_elements, np.nan)
            m = np.asarray(m, dtype=float).reshape(-1)
            is_sampled = np.atleast_1d(getattr(p, "is_sampled", False))
            if is_sampled.size == n_elements and np.sum(is_sampled) == m.size:
                full[is_sampled] = m
            elif m.size <= n_elements:
                full[: m.size] = m
            mult_map[p.label] = full

    # Dynamic Width Logic
    display_labels = [
        p.get_display_label(i)
        for p in auditor.all_params
        for i in range(np.prod(p.shape).astype(int) if p.shape != () else 1)
    ]
    max_label_len = max(
        [len(l) for l in display_labels]
        + [len(k) for k in other_nodes.keys()]
        + [24]
    )

    header = f"{'Parameter':>{max_label_len}} | {'Value':>15} | {'Scale':>10} | {'Units':>12} | {'Log-Prob':>10} | Priors & Bounds (*=user) |"
    table_width = len(header)

    def _banner(text):
        # Pad each caption line so its trailing dashes line up with the
        # table's horizontal rules, whatever max_label_len came out to.
        line = f"-----   {text}"
        return line + " " * max(table_width - len(line) - 5, 1) + "-----"

    logger.info("-" * table_width)
    logger.info(
        _banner(
            "Starting points and penalties (Physical Space) with measured whitening scales"
        )
    )
    logger.info(
        _banner(
            "Scale is the data-driven 1-sigma step measured from the start (the 0.5-nat logp contour);"
        )
    )
    logger.info(
        _banner(
            "the sampler steps in units of it.  N/A marks parameters that are not sampled (fixed/derived)."
        )
    )
    logger.info(
        _banner(
            "Log-Prob for parameters includes summed penalties from bounds and priors."
        )
    )
    logger.info("-" * table_width)
    logger.info(header)
    logger.info("-" * table_width)

    flat_warnings = []

    # --- PART 1: CORE PARAMETERS ---
    for p in auditor.all_params:
        should_print = getattr(p, "debug_print", None)
        if should_print is None:
            should_print = np.any(getattr(p, "is_sampled", False))
            # Handle vectorized boolean flags
            if isinstance(should_print, np.ndarray):
                should_print = np.any(should_print)
        if not should_print:
            continue

        raw_v = p.initval
        raw_s = p.init_scale

        n_elements = int(np.prod(p.shape)) if p.shape not in ((), None) else 1
        cfg_mgr = auditor.system.config_manager

        if raw_v is None:
            # 1. Try the user/solved parameter table, one element at a time so
            #    a vector parameter still reports every element.
            try:
                vals = np.full(n_elements, np.nan)
                for i in range(n_elements):
                    uv = _user_initval(cfg_mgr, p, i)
                    if uv is not None:
                        vals[i] = uv
                if np.any(np.isfinite(vals)):
                    raw_v = vals if n_elements > 1 else float(vals[0])
            except Exception:
                pass

            # 2. Last resort: Eval the expression if it exists
            if raw_v is None and p.expression is not None:
                try:
                    # 'deps' often need to be resolved. This is a hacky but effective way
                    # to visualize the starting point of a deterministic.
                    raw_v = (
                        p.expression().eval()
                        if hasattr(p.expression(), "eval")
                        else p.expression()
                    )
                except:
                    pass

        if raw_v is None:
            continue

        # inspect_start is a READ-ONLY diagnostic.  np.atleast_1d returns the
        # SAME object for a 1-D array, so without the copy every write below
        # lands in Parameter.initval itself -- which used to revert the
        # polished starts apply_polished_starts had just stored there, so the
        # table lied AND get_raw_starts/_seed_initvals_for rebuilt the later
        # seeds from the reverted values.
        v_phys = np.atleast_1d(raw_v).copy()
        s_phys = np.atleast_1d(raw_s if raw_s is not None else np.nan)
        m_phys = np.atleast_1d(mult_map.get(p.label, [np.nan] * len(v_phys)))

        user_flag = "*" if getattr(p, "user_prior_modified", False) else ""

        for i in range(len(v_phys)):
            # An INACTIVE element is not a parameter of its instance's
            # parameterization (a non-MIST star's EEP, a linear-law band's u2):
            # it is held at a bookkeeping value nothing reads, so a row for it
            # would report a start value the fit does not have.
            if not p.element_is_active(i):
                continue

            row_label = p.get_display_label(i)

            # Parameter.initval is the AUTHORITATIVE start: it is what
            # get_raw_start encodes and what the sampler actually begins from.
            # ConfigManager.resolve already layered the solver-reconciled
            # user_params value into it at construction, and a pre-whitening
            # seed polish (apply_polished_starts) may legitimately have moved
            # it since.  So the user_params entry is only a FALLBACK, for an
            # element the Parameter never got a number for -- re-asserting it
            # over a live initval would make this table report a start the
            # sampler is not going to use.
            if _is_unset(v_phys[i]):
                uv = _user_initval(cfg_mgr, p, i)
                if uv is not None:
                    v_phys[i] = uv

            def safe_float(x):
                if x is None or (hasattr(x, "size") and x.size == 0):
                    return np.nan
                if hasattr(x, "eval"):
                    x = x.eval()
                    # Extract scalar from numpy arrays/scalars
                val = x.item() if hasattr(x, "item") else x
                try:
                    return float(val)
                except (TypeError, ValueError):
                    return np.nan

            # Internal -> user for THIS element.  index=i matters as soon as
            # a parameter carries per-element units (a `unit:` override on
            # one instance of a vector): the whole-vector call returned an
            # n-element array for a scalar input, and the .item() below then
            # raised "can only convert an array of size 1", killing the fit
            # in its startup table.
            val_out = float(p.from_internal(safe_float(v_phys[i]), index=i))
            scale_out = float(p.from_internal(safe_float(s_phys[i]), index=i))

            # Float/Scientific formatting logic ---
            def smart_format(val, width):
                # 2. Print a clean N/A instead of 'nan'
                if np.isnan(val):
                    return f"{'N/A':>{width}}"
                if val == 0:
                    return f"{0.0:>{width}.6f}"

                abs_v = abs(val)
                # Use scientific notation if it's outside the "clean" range
                # (below 1e-3 the Scale column's 3 decimals round to 0.000)
                if abs_v < 1e-3 or abs_v > 1e6:
                    precision = max(0, width - 7)
                    return f"{val:>{width}.{precision}e}"

                # Otherwise, use standard fixed-point
                precision = max(0, width - 7)
                return f"{val:>{width}.{precision}f}"

            val_str = smart_format(val_out, width=15)

            # Only sampled elements were probed; everything else (fixed,
            # derived) has no whitening scale to report.
            sampled_arr = np.atleast_1d(getattr(p, "is_sampled", False))
            elem_sampled = (
                bool(sampled_arr[i]) if i < sampled_arr.size else False
            )
            scale_str = (
                smart_format(scale_out, width=10)
                if elem_sampled
                else f"{'N/A':>10}"
            )

            # A NaN multiplier on a sampled element means the probe found
            # logp flat along it (it keeps its preliminary scale) -- warn
            # after the table.
            raw_m = m_phys[i] if i < len(m_phys) else np.nan
            if (
                whiten_report is not None
                and elem_sampled
                and (np.isnan(raw_m) or np.isinf(raw_m))
            ):
                flat_warnings.append(row_label)

            prior_str = p.get_prior_str(i, latex=False)

            logger.info(
                f"{row_label:>{max_label_len}} | {val_str} | {scale_str} | {p.get_unit_str(i):>12} | {param_logps.get(p.label, 0.0):10.2f} | {prior_str}{user_flag}"
            )

    # --- 2. Potentials & Likelihoods ---
    for node, lp in other_nodes.items():
        # logit_uniform_prior nodes are constant log-volume factors (−log range) that
        # never change during sampling and add nothing informative to the table.
        if node.startswith("logit_uniform_prior"):
            continue

        clean_node = (
            node.replace("up_bound.", "")
            .replace("low_bound.", "")
            .replace("prior.", "")
            .replace("user_prior.", "")
        )
        parent = auditor.param_lookup.get(clean_node)
        is_bound = "low_bound" in node or "up_bound" in node

        # Bug fix: skip inactive bounds — lp≈0 means we're well within the bound.
        # They clutter the table without conveying useful information.
        if is_bound and abs(lp) < 1e-6:
            continue

        # Bug fix: for bound nodes mark * only when the user explicitly set the
        # prior/bounds (sigma, lower, upper), NOT merely because they set initval.
        # user_modified is True for any user touch; user_prior_modified requires
        # an explicit physics override (sigma, lower, upper, mu).
        if is_bound:
            is_user = parent and getattr(parent, "user_prior_modified", False)
        else:
            is_user = (parent and parent.user_modified) or (
                clean_node in auditor.user_params
            )

        if abs(lp) > 1e-6 or is_user:
            p_info = "Likelihood/Det."

            if is_user:
                if "up_bound" in node and parent:
                    val = (
                        parent.upper[0] if parent.upper is not None else "N/A"
                    )
                    p_info = f"< {val}"
                elif "low_bound" in node and parent:
                    val = (
                        parent.lower[0] if parent.lower is not None else "N/A"
                    )
                    p_info = f"> {val}"
                elif parent:
                    p_info = parent.get_prior_str(latex=False)

            logger.info(
                f"{node:>{max_label_len}} | {'N/A':>15} | {'N/A':>10} | {'---':>12} | {lp:10.2f} | {p_info}{' *' if is_user else ''}"
            )
    logger.info("-" * table_width)

    # --- 3. THE FATAL CHECK ---
    bad_params = {k: v for k, v in param_logps.items() if not np.isfinite(v)}
    bad_nodes = {k: v for k, v in other_nodes.items() if not np.isfinite(v)}

    # if we start at a bad spot, PyMC will draw randomly from the prior, which will never work
    # raise an error here
    if bad_params or bad_nodes:
        bad_list = "\n".join(
            f"  -> {k}: {v}" for k, v in {**bad_params, **bad_nodes}.items()
        )
        logger.error(
            "!" * 40 + "\n"
            "Fatal error: the starting model returned an infinite/NaN penalty!\n"
            "The following nodes have Infinite or NaN Log-Probability:\n"
            f"{bad_list}\n"
            "Check your initial values against your bounds/priors!\n"
            + "!"
            * 40
        )
        raise ValueError(
            "Initialization failed due to non-finite Log-Probability."
        )

    if flat_warnings:
        logger.warning(
            "?" * 60 + "\n"
            f"WARNING: logp is flat along: {flat_warnings}. Check your bounds/initialization.\n"
            "These parameters keep their preliminary whitening scale.\n"
            "Even a single unconstrained parameter will destroy HMC efficiency.\n"
            + "?"
            * 60
        )

    if unused_yaml:
        logger.warning(
            f"The following parameters in the parameter.yaml file did not match any model parameter "
            f"and were not applied: {unused_yaml}\n"
            "This can be safely ignored if intentional, but check for typos."
        )


def _add_sampler_prose(system, method, swap_schedule="deo"):
    """Declare the run-level modeling prose (intro + sampler paragraph).

    Config facts only, per the prose contract (outputs/prose.py): the
    sampler's identity and citations.  The actual draw/burn-in counts are
    measured facts and belong to ``_add_wrapup_prose``.
    """
    prose = system.prose
    prose.add(
        r"This analysis used the EXOZIPPy modeling suite (Eastman et al., "
        r"in preparation), a successor to EXOFAST \citep{Eastman:2013} and "
        r"EXOFASTv2 \citep{Eastman:2019} built on PyMC "
        r"\citep{AbrilPla:2023}.",
        section="intro",
        key="run.exozippy",
        rank=5,
    )
    if method in ("ptde", "ptde_async"):
        swap_cite = (
            r" and the non-reversible deterministic even--odd (DEO) swap "
            r"schedule \citep{Syed:2022}"
            if str(swap_schedule).lower() == "deo"
            else ""
        )
        prose.add(
            r"We sampled the posterior with a parallel-tempered "
            r"differential-evolution MCMC "
            r"\citep{terBraak:2006, terBraak:2008} with adaptive "
            r"temperature-ladder placement \citep{Vousden:2016}"
            + swap_cite
            + ", as implemented in EXOZIPPy.",
            section="sampling",
            key="run.sampler",
            rank=10,
        )
    elif method == "demc":
        prose.add(
            r"We sampled the posterior with differential-evolution MCMC "
            r"\citep{terBraak:2006} as implemented in PyMC "
            r"\citep{AbrilPla:2023}.",
            section="sampling",
            key="run.sampler",
            rank=10,
        )
    elif method == "demcz":
        prose.add(
            r"We sampled the posterior with the DE-MC$_Z$ variant of "
            r"differential-evolution MCMC, which draws its proposal vectors "
            r"from each chain's own past states \citep{terBraak:2008}, as "
            r"implemented in PyMC \citep{AbrilPla:2023}.",
            section="sampling",
            key="run.sampler",
            rank=10,
        )
    else:
        # nuts / numpyro / blackjax are all NUTS implementations.
        prose.add(
            r"We sampled the posterior with the No-U-Turn Sampler "
            r"\citep{Hoffman:2014} as implemented in PyMC "
            r"\citep{AbrilPla:2023}.",
            section="sampling",
            key="run.sampler",
            rank=10,
        )


def _add_wrapup_prose(system, diag, mode_report):
    """Declare the post-fit prose: burn-in, convergence criteria, modes.

    These are diagnostics of the run (the convergence criteria the user
    asked the draft to record), not fitted values -- posterior numbers stay
    in the table, whose macros are the mechanism for citing them in prose.
    """
    prose = system.prose
    prose.add(
        r"The median values and 68\% confidence intervals of the "
        r"posterior are listed in Table~\ref{tab:"
        + str(getattr(system, "name", "system"))
        + r"}.",
        section="results",
        key="run.table_ref",
        rank=10,
    )
    if diag:
        prose.add(
            f"We discarded the first {diag.get('burnin', 0)} draws "
            f"({100 * diag.get('burnin_frac', 0.0):.0f}\\% of "
            f"{diag.get('n_draws', 0)}) of each chain as burn-in, keeping "
            f"{diag.get('n_chains_used')} chains.",
            section="convergence",
            key="run.burnin",
            rank=10,
        )
        criteria = []
        if diag.get("max_rhat_threshold") is not None:
            criteria.append(rf"$\hat{{R}} \le {diag['max_rhat_threshold']}$")
        if diag.get("min_ess_threshold") is not None:
            criteria.append(rf"ESS $\ge {diag['min_ess_threshold']}$")
        verdict = "met" if diag.get("converged", False) else "NOT met"
        measured = []
        if diag.get("max_rhat") is not None:
            measured.append(rf"maximum $\hat{{R}} = {diag['max_rhat']:.3f}$")
        if diag.get("min_ess") is not None:
            measured.append(f"minimum ESS $= {diag['min_ess']:.0f}$")
        prose.add(
            r"Convergence was assessed with the rank-normalized "
            r"Gelman--Rubin statistic and the effective sample size "
            r"\citep{Gelman:1992, Vehtari:2021} as implemented in ArviZ "
            r"\citep{Kumar:2019}"
            + (
                ", requiring "
                + " and ".join(criteria)
                + " for every sampled parameter"
                if criteria
                else ""
            )
            + f"; these criteria were {verdict}"
            + (" (" + ", ".join(measured) + ")." if measured else "."),
            section="convergence",
            key="run.convergence",
            rank=20,
        )
    if mode_report is not None and getattr(mode_report, "n_modes", 1) > 1:
        # The provenance is plain text (N_eff, >=): escape it for LaTeX
        # text mode, exactly as latex.py does for \tablecomments.
        from .outputs.texutils import latex_escape_prose

        provenance = latex_escape_prose(
            getattr(mode_report, "provenance", "see the table notes")
        )
        prose.add(
            f"The posterior is multimodal: {mode_report.n_modes} distinct "
            "modes were identified and are reported separately in the "
            f"parameter table. Mode weights: {provenance}.",
            section="modes",
            key="run.modes",
            rank=10,
        )


def _format_summary(idata, diag):
    """Build the *_summary.txt body: physical params only, worst Rhat first.

    Drops the ``*_raw`` unconstrained duplicates (rank-identical to their
    physical partners -- the confusing rows whose raw means blow up to ~1000
    on a degenerate direction) and sorts by r_hat descending so any
    convergence trouble sits at the top. A burn-in note and a loud NOT-
    CONVERGED banner (when applicable) are prepended so the file is honest
    about what was trimmed and whether thresholds were met.
    """
    post = idata.posterior
    var_names = [
        v for v in post.data_vars if not v.endswith("_raw") and v != "mode"
    ]
    df = az.summary(idata, var_names=var_names)
    if "r_hat" in df.columns:
        df = df.sort_values("r_hat", ascending=False)

    header = [
        f"# burn-in discarded: {diag['burnin']} draws "
        f"({100 * diag.get('burnin_frac', 0.0):.0f}% of {diag.get('n_draws', 0)}); "
        f"chains kept: {diag.get('n_chains_used')}",
    ]
    if not diag.get("good_reliable", True):
        header.append(
            "# NOTE: <3 chains reached the good-likelihood region; "
            "all chains kept (possible stuck-chain contamination)"
        )
    if not diag.get("converged", False):
        header.append(
            f"# WARNING: convergence NOT reached -- max Rhat={diag['max_rhat']:.3f} "
            f"({diag.get('worst_rhat_var')}), min ESS={diag['min_ess']:.0f} "
            f"({diag.get('worst_ess_var')}); thresholds "
            f"Rhat<={diag.get('max_rhat_threshold')}, "
            f"ESS>={diag.get('min_ess_threshold')}"
        )
    # to_string(), not str(): str(df) elides middle columns ("...") at
    # narrow terminal widths, silently dropping ess_bulk/ess_tail/r_hat --
    # the columns downstream tooling (e.g. examples/DC2018's collector)
    # parses from this file.
    return "\n".join(header) + "\n" + df.to_string() + "\n"


def make_corner(model, idata, filename, max_samples=1000):
    all_vars = list(idata["posterior"].data_vars)
    physical_vars = [
        v
        for v in all_vars
        if "_raw" not in v and "_interval" not in v and v != "mode"
    ]
    var_specs = [(v, None) for v in physical_vars]
    samples, labels = collect_corner_samples(idata, var_specs)
    save_corner_plot(samples, labels, filename, max_samples=max_samples)


# Module-level globals for fork-based parallel lp evaluation.
# PyTensor compiled functions can't be pickled, so they're set here before
# forking; child processes inherit them via copy-on-write without IPC.
_LP_FN = None
_LP_POINT_MAP = None


def _lp_eval_chain(args):
    """Evaluate logp for every draw in one chain (runs in a forked child)."""
    chain_data, chain_idx, n_draws = args
    lp_chain = np.full(n_draws, np.nan)
    for d in range(n_draws):
        point = {
            _LP_POINT_MAP[tname]: np.atleast_1d(chain_data[tname][d])
            for tname in chain_data
        }
        lp_chain[d] = float(_LP_FN(point))
    return chain_idx, lp_chain


def _ensure_lp(idata, model=None):
    """Make sure ``idata.sample_stats["lp"]`` exists; return whether it does.

    NUTS writes lp itself; the Metropolis/DE families and PTDE do not, so it
    is computed from the model and persisted -- once, right after sampling,
    so the saved trace carries it and no reader has to recompute it.  The
    fallback also serves old trace files written before that was done.

    One function for two call sites that had drifted, and the merge turned up
    that BOTH were broken for a trace carrying no ``sample_stats`` group at
    all -- just differently.  The save path checked
    ``hasattr(idata, "sample_stats")`` and then assigned into
    ``idata.sample_stats`` regardless, so the assignment raised; the plotting
    path guarded it with ``idata.add_groups(...)``, which is arviz 0.x API
    that no supported arviz has (``pyproject.toml`` floors it at 1.1.0, where
    ``InferenceData`` IS an ``xarray.DataTree``), so it raised too. Adding a
    group on a DataTree is ``idata["sample_stats"] = xr.Dataset()``.

    ``model=None`` means "report, do not compute": the plotting path can be
    handed a trace with no model, and there is nothing to fall back to.
    """
    ss = getattr(idata, "sample_stats", None)
    if ss is not None and "lp" in ss.data_vars:
        return True
    if model is None:
        return False

    logger.info("lp is not in the trace -- computing it from the model")
    lp_vals = _compute_lp_from_model(model, idata)
    if lp_vals is None:
        return False

    import xarray as xr

    if getattr(idata, "sample_stats", None) is None:
        idata["sample_stats"] = xr.Dataset()
    idata.sample_stats["lp"] = xr.DataArray(
        lp_vals,
        dims=["chain", "draw"],
        coords={
            "chain": idata.posterior.chain,
            "draw": idata.posterior.draw,
        },
    )
    return True


def _compute_lp_from_model(model, idata):
    """Compute log posterior at each draw by evaluating the compiled model logp.

    Used when the sampler (Metropolis) doesn't write lp to sample_stats.
    Chains are processed in parallel via fork so the PyTensor compiled function
    is inherited without pickling (numpy chain data is all that's sent over IPC).
    Returns an (n_chains, n_draws) float64 array, or None on failure.
    """
    try:
        n_chains = idata.posterior.sizes["chain"]
        n_draws = idata.posterior.sizes["draw"]

        with model:
            logp_fn = model.compile_logp(jacobian=False)

        # In EXOZIPPy's non-centered parameterization, the free RVs ARE the raw
        # unconstrained variables (e.g. "star.logmass_raw"). ArviZ stores them in
        # the posterior under the same name. Do NOT append another "_raw" here.
        point_map = {}  # trace var name → logp_fn input name
        for rv in model.free_RVs:
            vv = model.rvs_to_values.get(rv)
            if vv is None:
                continue
            if rv.name in idata.posterior.data_vars:
                point_map[rv.name] = vv.name

        if not point_map:
            logger.warning(
                "_compute_lp_from_model: no unconstrained vars found in trace"
            )
            return None

        logger.info(
            f"Computing lp for {n_chains}×{n_draws} draws "
            f"({len(point_map)} unconstrained vars)"
        )

        # Extract per-chain numpy arrays (picklable; logp_fn is NOT pickled —
        # it's inherited by child processes via fork).
        chain_arrays = []
        for c in range(n_chains):
            chain_arrays.append(
                {
                    tname: idata.posterior[tname].values[c]
                    for tname in point_map
                }
            )

        # Set module-level globals so forked workers inherit them without pickling.
        global _LP_FN, _LP_POINT_MAP
        _LP_FN = logp_fn
        _LP_POINT_MAP = point_map

        n_workers = min(n_chains, mp.cpu_count())
        ctx = mp.get_context("fork")
        with ctx.Pool(n_workers) as pool:
            results = pool.map(
                _lp_eval_chain,
                [(arr, c, n_draws) for c, arr in enumerate(chain_arrays)],
            )

        lp_vals = np.full((n_chains, n_draws), np.nan)
        for chain_idx, chain_lp in results:
            lp_vals[chain_idx] = chain_lp
            logger.info(
                f"  chain {chain_idx}: lp range "
                f"[{chain_lp.min():.1f}, {chain_lp.max():.1f}]"
            )

        return lp_vals

    except Exception as e:
        logger.warning(f"Could not compute lp from model: {e}")
        return None


def _chunk_by_rows(specs, rows_per_page):
    """Yield (chunk, n_rows) pairs sized so each page needs <= rows_per_page rows.

    ``specs`` is a list of (var_name, coords, n_rows) triples as produced by
    _split_degenerate_vars.  A spec carrying a non-None ``coords`` element
    selection gets a page to itself: ArviZ takes ONE coords mapping for the
    whole call, so two variables sharing a dimension name but needing
    different element subsets would silently cross-contaminate.
    """
    chunk, chunk_rows = [], 0
    for name, coords, r in specs:
        solo = coords is not None
        if chunk and (solo or chunk_rows + r > rows_per_page):
            yield chunk, chunk_rows
            chunk, chunk_rows = [], 0
        chunk.append((name, coords))
        chunk_rows += r
        if solo:
            yield chunk, chunk_rows
            chunk, chunk_rows = [], 0
    if chunk:
        yield chunk, chunk_rows


# arviz_stats builds its KDE on a fixed 512-interval grid spanning exactly
# [min, max] of the finite draws (bound_correction=True, so the grid is NOT
# widened by a bandwidth).  np.histogram then raises
#   ValueError: Too many bins for data range. Cannot create 512 finite-sized bins.
# whenever np.linspace(min, max, 513) cannot produce 513 strictly increasing
# float64 edges -- i.e. whenever the entire range spans fewer than ~512 ULPs.
# That kills the whole page, and with it the rest of the PDF and all of
# wrap-up, for one variable that did not move.
_KDE_GRID_LEN = 512


def _dist_degeneracy(values):
    """Why no density can be drawn for ``values``, or None if one can.

    Returns a short human-readable reason string.  Three shapes qualify, and
    all three are things a real fit produces:

    * no finite draws at all (a Deterministic whose physics went NaN);
    * exactly constant (an element pinned with ``sigma: 0`` inside an
      otherwise-sampled vector -- GP and robust-likelihood hyperparameters
      are full-length vectors with the non-opted-in files pinned, and such a
      vector IS tracked as a Deterministic, so its pinned elements reach the
      plot as constants);
    * finite but spanning fewer than _KDE_GRID_LEN float64 steps -- a chain
      that never moved except in the last few bits, which is what a
      gracefully stopped, unmixed run produces and what actually crashed CI.

    The third test is the exact condition numpy itself raises on, so it
    tracks the failure rather than approximating it.
    """
    x = np.asarray(values, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return "no finite draws"
    lo, hi = float(x.min()), float(x.max())
    if lo == hi:
        return f"constant at {lo:.10g}"
    if np.any(np.diff(np.linspace(lo, hi, _KDE_GRID_LEN + 1)) <= 0):
        return (
            f"range {hi - lo:.3g} around {lo:.10g} spans fewer than "
            f"{_KDE_GRID_LEN} float64 steps"
        )
    return None


class _DegenerateRow:
    """One (variable, element) row whose density cannot be drawn."""

    __slots__ = ("label", "reason", "values")

    def __init__(self, label, reason, values):
        self.label = label
        self.reason = reason
        self.values = values


def _element_rows(dataset, var):
    """Yield (selection, label, values) for every plotted row of ``var``.

    One row per element, matching _render_trace_page's compact=False layout
    (the row count that used to come from a separate _n_trace_rows helper is
    just len(list(_element_rows(...))) now).  ``selection`` is the label
    mapping that isolates the element -- ``{}`` for a scalar variable, and
    None when a dimension carries no coordinates and so cannot be
    label-selected at all; ``values`` is the element's (chain, draw) array.
    """
    da = dataset[var]
    extra = [d for d in da.dims if d not in ("chain", "draw")]
    if not extra:
        yield {}, var, np.asarray(da.values)
        return
    labelled = all(d in da.coords for d in extra)
    for combo in itertools.product(*(range(da.sizes[d]) for d in extra)):
        pos = dict(zip(extra, combo))
        # .isel, not .sel: a dimension without coordinates cannot be
        # label-selected, and only a label-selectable one can be handed back
        # to ArviZ as a ``coords`` mapping (see _split_degenerate_vars).
        vals = np.asarray(da.isel(pos).values)
        if labelled:
            sel = {
                d: np.asarray(da.coords[d].values)[i] for d, i in pos.items()
            }
        else:
            sel = None
        shown = sel if sel is not None else pos
        label = f"{var}[{', '.join(str(shown[d]) for d in extra)}]"
        yield sel, label, vals


def _split_degenerate_vars(idata, var_names, group="posterior"):
    """Split plot rows into ArviZ-renderable ones and density-less ones.

    Returns ``(specs, degenerate)``:

    * ``specs`` -- list of (var_name, coords, n_rows) for _chunk_by_rows.
      ``coords`` is None when the whole variable renders normally (so a fit
      with no degenerate element takes byte-identical code to before), and a
      {dim: [kept values]} mapping when only some elements of a vector do.
    * ``degenerate`` -- list of _DegenerateRow, rendered on their own pages.

    Each degenerate row is logged as a warning naming the variable element
    and the reason: a missing density is reported, never swallowed.
    """
    dataset = idata[group]
    specs, degenerate = [], []
    for v in var_names:
        if v not in dataset.data_vars:
            # Unknown name: pass it straight through so ArviZ raises the same
            # error it always did rather than having it silently disappear.
            specs.append((v, None, 1))
            continue
        rows = list(_element_rows(dataset, v))
        bad = [
            i
            for i, (_, _, vals) in enumerate(rows)
            if _dist_degeneracy(vals) is not None
        ]
        if not bad:
            specs.append((v, None, len(rows)))
            continue
        for i in bad:
            _, label, vals = rows[i]
            reason = _dist_degeneracy(vals)
            logger.warning(
                f"trace plot: no density for {label} ({reason}); its trace "
                "panel is still drawn"
            )
            degenerate.append(_DegenerateRow(label, reason, vals))
        good = [i for i in range(len(rows)) if i not in set(bad)]
        if not good:
            continue
        extra = [d for d in dataset[v].dims if d not in ("chain", "draw")]
        if len(extra) == 1 and rows[good[0]][0] is not None:
            dim = extra[0]
            kept = [rows[i][0][dim] for i in good]
            specs.append((v, {dim: kept}, len(kept)))
        else:
            # >= 2 element dimensions (a coords mapping would select the
            # CROSS PRODUCT of the per-dimension survivors, which can readmit
            # a degenerate element) or an unlabelled dimension ArviZ cannot be
            # given a coords mapping for.  No EXOZIPPy Parameter is shaped
            # either way (Parameter vectors are 1-D and ArviZ always names
            # their dimension), so rather than risk a re-crash the surviving
            # elements go on the annotated pages too, saying so.
            for i in good:
                _, label, vals = rows[i]
                degenerate.append(
                    _DegenerateRow(
                        label,
                        f"density omitted: other elements of {v} are "
                        "degenerate and this variable's element layout "
                        "cannot be split",
                        vals,
                    )
                )
    return specs, degenerate


def _render_degenerate_page(idata, rows, title, group="posterior"):
    """One page of rows whose density cannot be drawn.

    Same two-column geometry as _render_trace_page (dist column then trace
    column, one row per element), so the page reads continuously with the
    ArviZ ones and _shade_trace_axes_by_mode's even/odd axis convention still
    holds.  The trace panel is drawn for real -- a flat chain is exactly what
    the reader needs to see -- and the dist panel carries the reason and the
    value instead of a density.

    Annotating beats the alternatives: a skipped panel looks like a plotting
    bug, and a single-bin histogram looks like a real (if uninformative)
    density while conveying neither the value nor why it is the only bin.
    """
    dataset = idata[group]
    draw_coord = np.asarray(dataset["draw"].values)
    n = len(rows)
    fig, axes = plt.subplots(n, 2, figsize=(12, 3 * n), squeeze=False)
    for r, row in enumerate(rows):
        ax_dist, ax_trace = axes[r]
        ax_dist.set_title(row.label)
        ax_dist.text(
            0.5,
            0.5,
            f"no density\n{row.reason}",
            ha="center",
            va="center",
            transform=ax_dist.transAxes,
            fontsize=9,
            color="0.25",
        )
        ax_dist.set_xticks([])
        ax_dist.set_yticks([])

        vals = np.atleast_2d(np.asarray(row.values, dtype=float))
        finite = np.isfinite(vals)
        if finite.any():
            for c in range(vals.shape[0]):
                ax_trace.plot(draw_coord, vals[c], lw=0.8)
        else:
            ax_trace.text(
                0.5,
                0.5,
                "no finite draws",
                ha="center",
                va="center",
                transform=ax_trace.transAxes,
                fontsize=9,
                color="0.25",
            )
        ax_trace.set_xlabel("Draw")
        ax_trace.set_ylabel(row.label)
    fig.suptitle(title, fontsize=14)
    _shade_trace_axes_by_mode(fig, idata)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


def save_multipage_trace(
    idata,
    var_names,
    filename,
    rows_per_page=4,
    draws_per_chain=100,
    model=None,
):
    n_draws = idata.posterior.draw.size

    # Thin to cap matplotlib memory and render time.  The target is per-CHAIN.
    # Deriving thin_factor from the TOTAL sample count (n_chains * n_draws) but
    # applying it to the draw axis alone left every chain with only
    # max_samples/n_chains points -- ~28 for a 70-chain run, no matter how long
    # it actually sampled.  That is too few to read anything off the trace
    # column, and it silently pushed arviz's kind="auto" under its
    # 100-draws-per-chain threshold, so the dist column rendered ECDFs (which
    # saturate at 1.0 and look like the posterior is clipped) instead of
    # densities.
    #
    # isel on the InferenceData -- not az.from_dict, which expects dicts of
    # arrays rather than Datasets -- thins every group carrying a draw dim and
    # preserves the original `draw` coordinate, so the trace x axis stays in
    # true, unthinned draw numbers.
    if n_draws > draws_per_chain:
        thin_factor = max(1, n_draws // draws_per_chain)
        idata = idata.isel(draw=slice(None, None, thin_factor))

    # lp is in sample_stats for NUTS traces and for every trace saved after
    # the fix that computes and persists it right after pm.sample().  Fall
    # back to computing it for old trace files; a trace with no model to
    # fall back to simply gets no lp page.
    if _ensure_lp(idata, model):
        lp_idata, lp_var = idata, "lp"
    else:
        lp_idata, lp_var = None, None

    with PdfPages(filename) as pdf:
        # lp gets its own first page when available — mixing two different
        # datasets (sample_stats + posterior) in a pre-allocated axes grid
        # caused ArviZ 0.19 to silently ignore the passed axes and render
        # into its own floating figure, leaving our fig blank.  Let ArviZ
        # own the figure and retrieve it from the returned axes instead.
        if lp_var and lp_idata is not None:
            for fig in _trace_page_figures(
                lp_idata,
                [lp_var],
                rows_per_page=1,
                group="sample_stats",
                title_fn=lambda _i: "Trace Plots: log-posterior (lp)",
            ):
                pdf.savefig(fig)
                plt.close(fig)
                gc.collect()

        for fig in _trace_page_figures(
            idata,
            var_names,
            rows_per_page=rows_per_page,
            group="posterior",
            title_fn=lambda i: f"Trace Plots: Page {i}",
        ):
            pdf.savefig(fig)
            plt.close(fig)
            gc.collect()


def _trace_page_figures(idata, var_names, rows_per_page, group, title_fn):
    """Yield one figure per page, ArviZ pages first then annotated ones.

    Variables (or single vector elements) whose draws admit no density are
    routed to _render_degenerate_page instead of being handed to ArviZ, whose
    KDE grid is what raises on them.  With no such element the split is the
    identity and the ArviZ call is exactly the one made before this existed.
    """
    specs, degenerate = _split_degenerate_vars(idata, var_names, group=group)

    page_num = 0
    for chunk, n_rows in _chunk_by_rows(specs, rows_per_page):
        page_num += 1
        names = [name for name, _ in chunk]
        coords = next((c for _, c in chunk if c is not None), None)
        yield _render_trace_page(
            idata,
            names,
            n_rows,
            title=title_fn(page_num),
            group=group,
            coords=coords,
        )

    for start in range(0, len(degenerate), max(1, rows_per_page)):
        page_num += 1
        rows = degenerate[start : start + max(1, rows_per_page)]
        yield _render_degenerate_page(
            idata,
            rows,
            title=f"{title_fn(page_num)} (no density)",
            group=group,
        )


def _render_trace_page(
    idata, var_names, n_rows, title, group="posterior", coords=None
):
    """One trace-plot page: dist column + trace column, one row per element.

    plot_trace_dist (not plot_trace) is the ArviZ 1.0 equivalent of the old
    dist + trace two-column layout; plain plot_trace now renders only the
    trace lines.  compact=False keeps one row per vector element, matching
    the rows_per_page pagination math.

    kind is pinned to "kde" rather than left to default.  The default is
    rcParams["plot.density_kind"] = "auto", which silently switches to an ECDF
    whenever a chain carries fewer than 100 draws -- the dist column then plots
    a cumulative curve that plateaus at 1.0, which reads as a posterior clipped
    at its maximum.  A density is always what this column is meant to show.

    ``coords`` restricts which elements of a vector variable are drawn; it is
    set only when some of that vector's elements have no density (see
    _split_degenerate_vars) and is None for every ordinary page.
    """
    pc = az.plot_trace_dist(
        idata,
        var_names=var_names,
        group=group,
        coords=coords,
        compact=False,
        kind="kde",
        figure_kwargs={"figsize": (12, 3 * n_rows)},
    )
    fig = pc.viz["figure"].item()
    fig.suptitle(title, fontsize=14)
    _shade_trace_axes_by_mode(fig, idata)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


# Qualitative colormap for per-draw mode markers; wraps at 10 modes (a
# sensible ceiling given identify_modes' max_modes default of 8).
_MODE_CMAP = plt.get_cmap("tab10")
_MODE_INVALID_COLOR = (0.6, 0.6, 0.6, 0.9)


def _shade_trace_axes_by_mode(fig, idata):
    """Overlay per-draw markers colored by mode label on each trace axis.

    Makes mode-hopping (or its absence) visible directly in the chain trace
    plot: draws from the same chain keep their line but gain a dot colored
    by posterior['mode'] at that draw (gray for invalid/unassigned draws).
    No-op when idata has no 'mode' variable (older trace files, or mode
    identification failed/found only one mode) so unimodal runs render
    exactly as before this feature was added.

    Relies on _render_trace_page's fixed layout: compact=False with a dist
    column then a trace column, one row per variable element, so trace axes
    are every other axis (odd index) in fig.axes, and each trace axis's
    per-chain Line2D objects are in chain order (arviz's default behavior).
    """
    if not hasattr(idata, "posterior") or "mode" not in idata.posterior:
        return
    mode_vals = np.asarray(idata.posterior["mode"].values)  # (chain, draw)
    if mode_vals.size == 0 or mode_vals.max() < 1:
        return  # unimodal (single label 0) or nothing valid: nothing to show
    n_chain = mode_vals.shape[0]
    # x values are true (unthinned) draw numbers taken from the preserved
    # `draw` coordinate, so they index mode_vals only after being mapped back
    # to positions along the thinned axis.
    draw_coord = np.asarray(idata.posterior["draw"].values)

    for i, ax in enumerate(fig.axes):
        if i % 2 == 0:
            continue  # dist column; only shade the trace column
        for c, line in enumerate(list(ax.lines)):
            if c >= n_chain:
                break
            xd = np.asarray(line.get_xdata())
            yd = np.asarray(line.get_ydata())
            if xd.size == 0:
                continue
            idx = np.searchsorted(draw_coord, xd)
            idx = np.clip(idx, 0, mode_vals.shape[1] - 1)
            labels = mode_vals[c, idx]
            colors = [
                _MODE_CMAP(int(l) % 10) if l >= 0 else _MODE_INVALID_COLOR
                for l in labels
            ]
            ax.scatter(xd, yd, c=colors, s=8, zorder=5, linewidths=0)


def _sanitize_netcdf_attrs(idata):
    """Flatten dict-valued attrs to JSON strings so xarray can serialize to netCDF.

    nutpie stores rich metadata (dicts) in sample_stats attrs; netCDF only allows
    scalars/strings/arrays.
    """
    import json

    for group in idata.children:
        ds = getattr(idata, group, None)
        if ds is None or not hasattr(ds, "attrs"):
            continue
        for k, v in list(ds.attrs.items()):
            if isinstance(v, dict):
                ds.attrs[k] = json.dumps(v)


def _convert_posterior_to_user_units(idata, param_lookup, only=None):
    """Convert idata.posterior in-place from internal math units to user units.

    Each non-raw variable in the posterior whose Parameter has a non-trivial
    unit conversion is multiplied by the internal→user factor.  This is called
    once after sampling so that the saved trace, trace plots, ArviZ summary,
    and mkparam output are all in user-facing units (e.g. jupiterMass, m/s).

    This and ``get_draws`` below are the deliberate exceptions to "convert
    through Parameter.to_internal / from_internal": the operand is a
    (chain, draw, element) DataArray, so the factor has to broadcast against
    the TRAILING axis and the owner's element-count check -- which sees the
    total size -- would reject it.  The direction is the only thing that
    matters here, and it is stated: internal -> user multiplies, and
    get_draws (user trace -> internal for the physics) divides.

    `only` restricts the pass to a subset, which the degeneracy fold
    (review 1.8.3) needs: it runs AFTER this conversion and regenerates
    some deterministics from PyMC, so those few come back in internal
    units and have to be converted again -- while re-converting the whole
    posterior would multiply everything else a second time.
    """
    names = list(idata.posterior.data_vars) if only is None else list(only)
    for var_name in names:
        if var_name.endswith("_raw") or var_name not in param_lookup:
            continue
        factor = np.squeeze(
            np.asarray(
                param_lookup[var_name]._get_conversion_factors(), dtype=float
            )
        )
        if np.all(factor == 1.0):
            continue
        idata.posterior[var_name] = idata.posterior[var_name] * factor


def get_draws(idata, n_draws=50, param_lookup=None, mode=None):
    """
    Extracts a random subset of draws from the posterior for plotting.

    The trace is stored in user units.  Component physics functions expect
    internal units, so each variable is divided by its conversion factor
    before being returned when ``param_lookup`` is provided.

    ``mode``: if given, restrict to draws whose ``posterior['mode']`` label
    equals this integer (used by the per-mode output loop in run_fit to
    build a mode-specific draw set). If omitted (default), every valid draw
    (mode >= 0) is eligible, matching the combined-posterior behavior.
    """
    # 1. Flatten chains/draws into a single 'sample' dimension
    post = az.extract(idata, combined=True, keep_dataset=True)

    # Never plot draws flagged invalid by mode identification (mode == -1:
    # runaway/stuck-chain draws with broken lp). With an explicit `mode`,
    # restrict to exactly that mode's draws instead.
    if "mode" in post:
        labels = np.asarray(post["mode"].values, dtype=int)
        keep = (labels == mode) if mode is not None else (labels >= 0)
        post = post.isel(sample=keep)
    elif mode is not None:
        raise ValueError(
            "get_draws: mode=%r requested but idata has no "
            "posterior['mode'] variable (identify_modes was "
            "not run or failed)" % (mode,)
        )

    total_available = post.sample.size
    n_to_extract = min(n_draws, total_available)

    # 2. Pick random indices
    indices = np.random.choice(
        total_available, size=n_to_extract, replace=False
    )

    draw_list = []
    for idx in indices:
        point = {}
        for var in post.data_vars:
            if var == "mode":
                continue
            val = post[var].isel(sample=idx).values
            if (
                param_lookup is not None
                and var in param_lookup
                and not var.endswith("_raw")
            ):
                factor = np.squeeze(
                    np.asarray(
                        param_lookup[var]._get_conversion_factors(),
                        dtype=float,
                    )
                )
                val = val / factor
            point[var] = val
        draw_list.append(point)

    return draw_list


def _idata_for_mode(idata, mode_k):
    """Build a synthetic single-chain InferenceData holding only mode_k's draws.

    ``make_corner`` (and anything else that reads idata.posterior directly,
    as opposed to Parameter.posterior) needs a real InferenceData with
    (chain, draw) dims to stack over; the mode label lives at the (chain,
    draw) granularity of the original trace, so the cheapest way to hand it
    a mode-restricted view is to flatten the selected draws into one
    synthetic chain (chain identity doesn't matter for a corner plot).
    """
    post = az.extract(idata, combined=True, keep_dataset=True)
    labels = np.asarray(post["mode"].values, dtype=int)
    sub = post.isel(sample=(labels == mode_k))

    data = {}
    for var in sub.data_vars:
        if var == "mode":
            continue
        arr = np.asarray(sub[var].values)  # dims: (*extra, sample)
        arr = np.moveaxis(arr, -1, 0)  # -> (sample, *extra)
        data[var] = arr[np.newaxis, ...]  # -> (1, sample, *extra)
    return az.from_dict({"posterior": data})


def _emit_per_mode_outputs(system, model, idata, mode_report, prefix):
    """Re-emit the combined-posterior corner + component plots once per mode.

    Interim (P7) multimodal reporting: loop the existing single-posterior
    plot calls once per detected mode instead of building a new stratified
    figure. Only called when mode_report.n_modes > 1 (see the guard at the
    call site in run_fit); per-mode LaTeX columns and CSV rows already exist
    via build_latex_output/build_csv_output's mode_report kwarg and are not
    duplicated here.

    Model plots (comp.plot) can be expensive (e.g. VBM microlensing
    evaluations), so each mode's wall-clock cost is logged -- a slow
    per-mode loop should be visible in logs, not silent.
    """
    prefix = str(prefix)
    param_lookup = system.get_parameter_lookup()
    for k, m in enumerate(mode_report.modes):
        suffix = mode_suffix(k)
        t0 = time.time()

        idata_k = _idata_for_mode(idata, k)
        make_corner(model, idata_k, f"{prefix}_corner_{suffix}.png")

        # Same draw-count knob as the combined-posterior plots (get_draws'
        # n_draws default) -- no extra stratification needed here since each
        # mode draws from its own full, already-labeled set of samples.
        draws_k = get_draws(idata, param_lookup=param_lookup, mode=k)
        for comp in system.active_components.values():
            comp.plot(
                system, draws_k, filename_prefix=f"{prefix}_mcmc_{suffix}"
            )

        logger.info(
            f"Per-mode outputs for {suffix} (weight={m.weight:.3f}, "
            f"n_draws={m.n_draws}) written in {time.time() - t0:.1f}s"
        )
