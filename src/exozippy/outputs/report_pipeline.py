"""Shared post-sampling reporting pipeline: mode identification ->
posterior distribution -> LaTeX/CSV table generation.

This is the block that used to live inline in run.run_fit() right after
sampling finished. It is now a single function so that run.py (the live
fitting path) and the exozippy-modes CLI (outputs/cli_modes.py, which
reprocesses a previously saved trace file without re-sampling) can never
drift apart: both call sites import build_mode_reports from this module.
"""

import logging
from pathlib import Path

from .modes import identify_modes, check_invalid_frac, DEFAULT_MAX_INVALID_FRAC
from .latex import build_latex_output, build_csv_output

logger = logging.getLogger(__name__)


def build_mode_reports(system, idata, prefix, min_weight=None, max_modes=None,
                       feature_vars=None, seed=None, model=None,
                       trace_path=None,
                       max_invalid_frac=DEFAULT_MAX_INVALID_FRAC, force=False,
                       raise_on_invalid=True, evidence_weights=False):
    """Identify posterior modes, distribute the posterior, write tables.

    Writes ``<prefix>_modes.txt``, ``<prefix>_definitions.tex``,
    ``<prefix>_template.tex``, and ``<prefix>_results.csv``.

    Mode identification is wrapped in a broad try/except: a broken mode
    pass must never take down the rest of a fit's outputs, so a failure
    here is logged as a warning and the tables fall back to describing the
    combined (unimodal) posterior.

    Parameters
    ----------
    system : exozippy.system.System
        Must already have had ``prepare()`` and ``build_model()`` called
        (so every Parameter has its expression/metadata wired up) --
        ``distribute_posterior`` and the table builders below need that.
    idata : arviz.InferenceData
        Must have a posterior group (and ideally sample_stats["lp"]).
        Mutated in place: identify_modes attaches an integer 'mode'
        variable to idata.posterior.
    prefix : str or pathlib.Path
        Output files are written as '<prefix>_modes.txt', etc.
    min_weight, max_modes, feature_vars, seed : optional
        Passed through to identify_modes when not None; identify_modes's
        own defaults apply otherwise.
    model : pymc.Model, optional
        Required only when ``evidence_weights=True`` (bridge-sampling
        evidence estimation evaluates the model's logp).
    trace_path : optional
        Included in the ``check_invalid_frac`` error message (the trace and
        mode report are already written to disk by the time it can raise,
        so this just tells the caller where to look).
    max_invalid_frac, force : passed through to check_invalid_frac.
    raise_on_invalid : bool
        Live fits (run.py) must not silently emit final tables from a
        numerically broken run, so this defaults to True. The forensic
        exozippy-modes CLI reprocesses a saved trace and always completes,
        so it passes False and reports invalid-draw problems as a warning
        banner of its own instead.
    evidence_weights : bool
        Opt-in per-mode evidence weighting (bridge sampling, a fallback /
        cross-check path -- see outputs.evidence). On success it replaces
        the occupancy weights and provenance on ``mode_report`` in place, so
        the LaTeX/CSV output below picks the new weights up automatically.
        Self-diagnosing: a single refused mode falls back to occupancy.

    Returns
    -------
    outputs.modes.ModeReport, or None if mode identification failed or
    found no valid draws (see the warning logged in that case).
    """
    prefix = Path(prefix)
    mode_kwargs = {}
    if min_weight is not None:
        mode_kwargs["min_weight"] = min_weight
    if max_modes is not None:
        mode_kwargs["max_modes"] = max_modes
    if feature_vars is not None:
        mode_kwargs["feature_vars"] = feature_vars
    if seed is not None:
        mode_kwargs["seed"] = seed

    # Identify posterior modes and label every draw: idata gains an integer
    # posterior['mode'] variable (-1 = invalid/unassigned) that
    # distribute_posterior and the table builders below key off of.  Mode
    # detection must never take down a finished fit's outputs, hence the
    # broad catch.
    mode_report = None
    modes_path = None
    try:
        mode_report = identify_modes(idata, **mode_kwargs)
        modes_path = Path(str(prefix) + "_modes.txt")
        modes_path.write_text(mode_report.to_text(), encoding="utf-8")
        if mode_report.n_modes > 1:
            logger.info(
                f"Posterior is multimodal: {mode_report.n_modes} modes, "
                f"weights {[f'{w:.3f}' for w in mode_report.weights]} "
                f"({'weights validated' if mode_report.weights_reliable else 'weights UNRELIABLE'}); "
                f"see {modes_path}")
    except Exception:
        logger.warning("Mode identification failed; reporting the combined "
                       "posterior only", exc_info=True)

    if raise_on_invalid:
        # The trace and mode report are already written at this point, so
        # evidence survives this raise; override via config
        # `modes: {max_invalid_frac: ..., force: true}` for forensic
        # re-processing of old/known-bad traces.
        check_invalid_frac(mode_report, max_invalid_frac=max_invalid_frac,
                           force=force, trace_path=trace_path,
                           modes_path=modes_path)

    # Optional per-mode evidence weighting (fallback / cross-check path).
    # On success it replaces the occupancy weights and provenance in place,
    # so the LaTeX weight row and CSV weight column below pick the evidence
    # weights up automatically.
    if (evidence_weights and mode_report is not None
            and mode_report.n_modes > 1):
        try:
            from .evidence import estimate_mode_evidences, apply_evidence_weighting
            evidences = estimate_mode_evidences(model, idata, mode_report)
            applied = apply_evidence_weighting(mode_report, evidences)
            # Refresh the human-readable mode report with the new weights.
            modes_path.write_text(mode_report.to_text(), encoding="utf-8")
            logger.info("Evidence weighting %s: weights %s (%s)",
                        "applied" if applied else "refused (kept occupancy)",
                        [f"{w:.3f}" for w in mode_report.weights],
                        mode_report.provenance)
        except Exception:
            logger.warning("Evidence weighting failed; keeping occupancy "
                           "weights", exc_info=True)

    # populate the parameters with the posteriors
    system.distribute_posterior(idata)

    # Generate latex table and machine-readable CSV
    build_latex_output(system,
                       var_filename=str(prefix) + '_definitions.tex',
                       template_filename=str(prefix) + '_template.tex',
                       caption=r"Median and 68\% Confidence intervals for " + prefix.stem,
                       mode_report=mode_report)
    build_csv_output(system, csv_filename=str(prefix) + '_results.csv',
                     mode_report=mode_report)

    return mode_report
