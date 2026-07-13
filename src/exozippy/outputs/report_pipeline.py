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

from .modes import identify_modes
from .latex import build_latex_output, build_csv_output

logger = logging.getLogger(__name__)


def build_mode_reports(system, idata, prefix, min_weight=None, max_modes=None,
                       feature_vars=None, seed=None):
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
