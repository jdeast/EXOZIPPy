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

from .latex import build_csv_output, build_latex_output
from .modes import (
    DEFAULT_MAX_INVALID_FRAC,
    MODE_FAILED,
    MODE_NO_VALID_DRAWS,
    MODE_OK,
    NoValidDrawsError,
    check_invalid_frac,
    identify_modes,
    mode_status_to_text,
)
from .texutils import latex_escape

logger = logging.getLogger(__name__)


def build_mode_reports(
    system,
    idata,
    prefix,
    min_weight=None,
    max_modes=None,
    feature_vars=None,
    seed=None,
    model=None,
    trace_path=None,
    max_invalid_frac=DEFAULT_MAX_INVALID_FRAC,
    force=False,
    raise_on_invalid=True,
    evidence_weights=False,
    seed_ledger=None,
    hot_status=None,
    mode_status=None,
):
    """Identify posterior modes, distribute the posterior, write tables.

    Writes ``<prefix>_modes.txt``, ``<prefix>_definitions.tex``,
    ``<prefix>_template.tex``, and ``<prefix>_results.csv``.

    Mode identification is wrapped in a broad try/except: a broken mode
    pass must never take down the rest of a fit's outputs, so a failure
    here is logged as a warning and the tables fall back to describing the
    combined (unimodal) posterior.  The one carve-out is
    ``NoValidDrawsError`` -- every draw rejected by the numerical-validity
    filter -- which is not a broken mode pass but a broken trace, and is
    routed to ``check_invalid_frac`` instead of being absorbed.

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
    seed_ledger : list of outputs.ledger.SeedRecord, optional
        The multi-seed "considered and rejected" ledger built by run.py
        after the seed polish (outputs.ledger.build_seed_ledger). When
        given, each record is matched to a surviving posterior mode or
        marked rejected; the ledger section is appended to
        <prefix>_modes.txt, rejected solutions get Laplace rows in the
        CSV and a standalone <prefix>_rejected_modes.tex table. Absent for
        single-seed fits and for the trace-reprocessing CLI (the seeds no
        longer exist there).
    hot_status : dict, optional
        Outcome of the hot-chain suppressed-mode search
        (outputs.ledger.discover_hot_modes / hot_status_to_text), as built
        by run.py.  Rendered into <prefix>_modes.txt INDEPENDENTLY of
        ``seed_ledger``: "never searched" and "search failed" are exactly
        the states in which no ledger records exist, and they are precisely
        the states the report must not render as silence.
    mode_status : dict, optional
        In/out: filled in with the machine-readable outcome of the mode
        pass -- a ``state`` from the outputs.modes MODE_* vocabulary plus
        the invalid-draw bookkeeping -- following the status-dict pattern
        of outputs/ledger.py's hot-chain search and outputs/evidence.py's
        per-mode results.  It is what lets a caller tell a returned None
        that means "the draws are unusable" (MODE_NO_VALID_DRAWS) from one
        that means "the mode pass could not tell you anything"
        (MODE_FAILED); the validity gate below reads exactly that
        distinction.  A fresh dict is used when none is passed.

    Returns
    -------
    outputs.modes.ModeReport, or None if mode identification failed or
    found no valid draws (see the warning logged in that case, and
    ``mode_status`` for which of the two it was).
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
    if mode_status is None:
        mode_status = {}
    try:
        mode_report = identify_modes(idata, **mode_kwargs)
        mode_status.update(
            state=MODE_OK,
            n_draws=int(mode_report.labels.size),
            n_invalid=int(mode_report.n_invalid),
            invalid_frac=float(mode_report.invalid_frac),
            reasons=dict(mode_report.invalid_reason_counts or {}),
        )
        modes_path = Path(str(prefix) + "_modes.txt")
        modes_path.write_text(mode_report.to_text(), encoding="utf-8")
        if mode_report.n_modes > 1:
            logger.info(
                f"Posterior is multimodal: {mode_report.n_modes} modes, "
                f"weights {[f'{w:.3f}' for w in mode_report.weights]} "
                f"({'weights validated' if mode_report.weights_reliable else 'weights UNRELIABLE'}); "
                f"see {modes_path}"
            )
    except NoValidDrawsError as exc:
        # EVERY draw failed the numerical-validity filter.  This is the one
        # mode-pass failure that IS a statement about the draws, so it gets
        # its own state and is handed to the gate below -- catching it in
        # the broad clause is what let a 100%-invalid trace emit a clean
        # -looking table while a 1.1%-invalid one refused (review 3.17).
        mode_status.update(
            state=MODE_NO_VALID_DRAWS,
            n_draws=exc.n_draws,
            n_invalid=exc.n_invalid,
            invalid_frac=exc.invalid_frac,
            reasons=exc.reason_counts,
            per_chain_invalid=exc.per_chain_invalid,
            detail=str(exc),
        )
        logger.warning(
            "Mode identification found NO VALID DRAWS: %s. Nothing computed "
            "from this trace describes a posterior.",
            exc,
        )
        # Leave the forensic record in the file the user is pointed at.
        # Written before the gate raises (the raise is the point, but the
        # evidence has to survive it) and before the hot-status/ledger
        # sections append -- otherwise, on the non-raising paths, the only
        # <prefix>_modes.txt a user gets is one whose visible content reads
        # entirely normal.
        try:
            modes_path = Path(str(prefix) + "_modes.txt")
            modes_path.write_text(
                mode_status_to_text(mode_status), encoding="utf-8"
            )
        except Exception:
            modes_path = None
            logger.warning(
                "Could not write the no-valid-draws mode report; the "
                "failure is reported here and by the check below",
                exc_info=True,
            )
    except Exception as exc:
        # Any other mode-pass failure says nothing about whether the draws
        # are usable, so it stays a warning and the gate below stays quiet.
        mode_status.update(state=MODE_FAILED, detail=repr(exc))
        logger.warning(
            "Mode identification failed; reporting the combined "
            "posterior only",
            exc_info=True,
        )

    if raise_on_invalid:
        # The trace and mode report are already written at this point, so
        # evidence survives this raise; override via config
        # `modes: {max_invalid_frac: ..., force: true}` for forensic
        # re-processing of old/known-bad traces.  mode_status is what makes
        # the all-invalid case (no report at all) reach this gate instead
        # of slipping through the None check.
        check_invalid_frac(
            mode_report,
            max_invalid_frac=max_invalid_frac,
            force=force,
            trace_path=trace_path,
            modes_path=modes_path,
            status=mode_status,
        )

    # Optional per-mode evidence weighting (fallback / cross-check path).
    # On success it replaces the occupancy weights and provenance in place,
    # so the LaTeX weight row and CSV weight column below pick the evidence
    # weights up automatically.
    if (
        evidence_weights
        and mode_report is not None
        and mode_report.n_modes > 1
    ):
        try:
            from .evidence import (
                apply_evidence_weighting,
                estimate_mode_evidences,
            )

            evidences = estimate_mode_evidences(model, idata, mode_report)
            # idata= keeps posterior['mode'].attrs in step with the rewritten
            # weights/provenance instead of leaving the occupancy values
            # attached to the trace that ships to disk.
            applied = apply_evidence_weighting(
                mode_report, evidences, idata=idata
            )
            # Refresh the human-readable mode report with the new weights.
            modes_path.write_text(mode_report.to_text(), encoding="utf-8")
            logger.info(
                "Evidence weighting %s: weights %s (%s)",
                "applied" if applied else "refused (kept occupancy)",
                [f"{w:.3f}" for w in mode_report.weights],
                mode_report.provenance,
            )
        except Exception:
            logger.warning(
                "Evidence weighting failed; keeping occupancy weights",
                exc_info=True,
            )

    # Hot-chain suppressed-mode search outcome.  Written whether or not a
    # ledger exists: the states worth distinguishing most ("never searched",
    # "search failed") are the ones that produce no ledger records at all,
    # so gating this on `seed_ledger` would silence exactly the cases it is
    # here to surface.
    if hot_status:
        try:
            from .ledger import hot_status_to_text

            if modes_path is None:
                modes_path = Path(str(prefix) + "_modes.txt")
            with open(modes_path, "a", encoding="utf-8") as f:
                f.write(hot_status_to_text(hot_status))
        except Exception:
            logger.warning(
                "Hot-chain status reporting failed; continuing without it",
                exc_info=True,
            )

    # Seeded-solution ledger: match every (polished) seed to a surviving
    # mode or mark it rejected, and report the rejected ones -- the
    # "considered and rejected" record that pure T=1 occupancy loses.
    # Appended AFTER any evidence-weighting rewrite of the mode report.
    if seed_ledger:
        try:
            from .ledger import (
                ledger_to_text,
                match_ledger_to_modes,
                rejected_records,
                write_rejected_latex,
            )

            match_ledger_to_modes(seed_ledger, mode_report)
            text = ledger_to_text(seed_ledger)
            if modes_path is None:
                modes_path = Path(str(prefix) + "_modes.txt")
            with open(modes_path, "a", encoding="utf-8") as f:
                f.write(text)
            n_rej = len(rejected_records(seed_ledger))
            if n_rej:
                write_rejected_latex(
                    seed_ledger,
                    str(prefix) + "_rejected_modes.tex",
                    hot_status=hot_status,
                )
                logger.info(
                    f"Seed ledger: {n_rej} seeded solution(s) rejected by "
                    f"the posterior; Laplace characterization in "
                    f"{modes_path} and {prefix}_rejected_modes.tex"
                )
        except Exception:
            logger.warning(
                "Seed-ledger reporting failed; continuing without it",
                exc_info=True,
            )

    # populate the parameters with the posteriors
    system.distribute_posterior(idata)

    # The ledger's rejected-seed rows are mode-keyed ('rejected-seed<k>'),
    # so the CSV must carry the mode columns even when the surviving
    # posterior is unimodal -- otherwise a 4-column header would sit over a
    # mix of 4- and 7-column rows.  Decided with the same predicate
    # append_ledger_csv uses, so the two can never disagree.
    ledger_rows = False
    if seed_ledger:
        from .ledger import rejected_records

        ledger_rows = bool(rejected_records(seed_ledger))

    # latex.py already appends an invalid-draw note to \tablecomments, but
    # it reads it off the mode report -- which does not exist when EVERY
    # draw was invalid, so the one table that is entirely untrustworthy was
    # the one table carrying no note at all (the same inversion as the gate
    # itself).  Supply it from the status instead.  Only reachable on the
    # non-raising paths (exozippy-modes, or `modes: {force: true}`): a live
    # fit has already refused above.  The percentage MUST carry \%, or the
    # bare % comments out the rest of the \tablecomments{} line.
    table_comments = None
    if mode_status.get("state") == MODE_NO_VALID_DRAWS:
        table_comments = (
            rf"ALL {mode_status.get('n_invalid', 0)} draws "
            rf"({100 * mode_status.get('invalid_frac', 0.0):.2f}\%) in this "
            "trace were rejected as numerically invalid, so no posterior "
            "mode could be identified: every value in this table summarizes "
            "REJECTED draws and none of it is meaningful. This indicates a "
            "model or sampler bug."
        )

    # Generate latex table and machine-readable CSV.  prefix.stem is a user
    # string on its way into \tablecaption{}: 'DC2018_128' and
    # 'KMT-2019-BLG-1806_nt8long' are both real prefixes here, and a raw
    # underscore is a LaTeX compile error at the end of a long fit.
    build_latex_output(
        system,
        var_filename=str(prefix) + "_definitions.tex",
        template_filename=str(prefix) + "_template.tex",
        caption=r"Median and 68\% Confidence intervals for "
        + latex_escape(prefix.stem),
        tablecomments=table_comments,
        mode_report=mode_report,
    )
    build_csv_output(
        system,
        csv_filename=str(prefix) + "_results.csv",
        mode_report=mode_report,
        mode_columns=ledger_rows,
    )
    if ledger_rows:
        try:
            from .ledger import append_ledger_csv

            append_ledger_csv(seed_ledger, str(prefix) + "_results.csv")
        except Exception:
            logger.warning(
                "Seed-ledger CSV rows failed; continuing without them",
                exc_info=True,
            )

    return mode_report
