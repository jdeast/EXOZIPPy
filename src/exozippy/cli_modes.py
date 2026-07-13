"""Console entry point: exozippy-modes <config.yaml>

Reprocesses a previously saved trace (``<prefix>_trace.nc``) through the
posterior mode-identification + reporting pipeline without re-sampling. This
is a forensic/offline tool: it rebuilds the System from the same config and
parameter_file YAML used for the original fit (needed for Parameter units,
expressions, and derived-parameter posteriors -- see CLAUDE.md's six-stage
lifecycle), loads the saved trace, runs outputs.modes.identify_modes and
System.distribute_posterior, and rewrites <prefix>_modes.txt,
<prefix>_definitions.tex, <prefix>_template.tex, and <prefix>_results.csv.

It shares the exact identify_modes -> distribute_posterior -> LaTeX/CSV
pipeline with run.run_fit() via outputs.report_pipeline.build_mode_reports,
so this CLI can never drift from what a live fit produces.

It also persists the mode labels back into the trace file itself (atomic
write via a temp file + rename in the same directory, so a reader that has
the old file open never sees a partial write), so a single .nc file ends up
carrying the full multimodal solution.

identify_modes is deterministic under its default seed (20260711), so
re-running this CLI on the same trace file with the same options reproduces
identical mode labels and reports -- it is idempotent.
"""

import logging
import os
from pathlib import Path

import click
import yaml
import arviz as az

from .logger import setup_logging
from .system import System
from .outputs.report_pipeline import build_mode_reports

logger = logging.getLogger(__name__)


def _persist_trace(idata, trace_path):
    """Atomically rewrite trace_path with idata (mode labels attached).

    Writes to a temp file in the same directory first, then renames over
    the original -- a reader with the old file open, or a crash mid-write,
    never sees a partially-written trace.
    """
    trace_path = Path(trace_path)
    tmp_path = trace_path.with_name(trace_path.name + ".tmp")
    idata.to_netcdf(str(tmp_path))
    os.replace(str(tmp_path), str(trace_path))


@click.command()
@click.argument("config_file")
@click.option("--min-weight", type=float, default=None,
              help="Modes below this fraction of valid draws are dropped "
                   "and their draws left unassigned "
                   "(identify_modes default: 0.005).")
@click.option("--max-modes", type=int, default=None,
              help="Upper limit for the BIC mode-count scan "
                   "(identify_modes default: 8).")
@click.option("--feature-vars", default=None,
              help="Comma-separated posterior variable names to cluster on "
                   "(identify_modes default: every '*_raw' unconstrained "
                   "sampled variable in the trace).")
@click.option("--seed", type=int, default=None,
              help="Random seed for k-means clustering (identify_modes "
                   "default: 20260711). identify_modes is deterministic "
                   "for a fixed seed, so re-running this CLI on the same "
                   "trace file with the same seed reproduces identical "
                   "mode labels and reports every time.")
@click.option("--logger-level", default=None,
              type=click.Choice(["DEBUG", "INFO", "WARNING"], case_sensitive=False),
              help="Logging level (overrides logger_level in config file).")
def main(config_file, min_weight, max_modes, feature_vars, seed, logger_level):
    """Reprocess a saved trace through posterior mode identification.

    CONFIG_FILE is the same system YAML passed to `exozippy`; its `prefix:`
    key locates the previously saved trace (<prefix>_trace.nc). Rewrites
    <prefix>_modes.txt, <prefix>_definitions.tex, <prefix>_template.tex,
    <prefix>_results.csv, and the trace file itself (with
    idata.posterior['mode'] attached) -- without re-running the sampler.

    This tool always completes: unlike a live fit, mode-identification
    problems here (a high invalid-draw fraction, or unreliable mode
    weights) are reported as a prominent warning banner, not a fatal error,
    since the whole point of this command is to be able to inspect an
    already-finished trace.
    """
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    if logger_level:
        config["logger_level"] = logger_level.upper()

    prefix = Path(config.get("prefix", "fitresults/planet"))
    setup_logging(prefix, config.get("logger_level", "INFO"))

    trace_path = Path(str(prefix) + "_trace.nc")
    if not trace_path.exists():
        raise FileNotFoundError(
            f"No saved trace found at {trace_path}. exozippy-modes reprocesses "
            f"an existing trace produced by a live fit; run "
            f"`exozippy {config_file}` first.")

    # Build the System (needed for Parameter units/expressions and
    # derived-parameter posteriors) but never sample -- prepare() +
    # build_model() only, matching the lifecycle documented in CLAUDE.md.
    system = System(config)
    system.prepare()
    system.build_model()

    idata = az.from_netcdf(str(trace_path))

    feature_var_list = feature_vars.split(",") if feature_vars else None

    mode_report = build_mode_reports(
        system, idata, prefix,
        min_weight=min_weight, max_modes=max_modes,
        feature_vars=feature_var_list, seed=seed)

    if mode_report is None:
        logger.warning(
            "!" * 60 + "\n"
            "MODE IDENTIFICATION FAILED: wrote combined-posterior tables "
            f"only. See the warning above and {prefix}_modes.txt (if "
            "present) for details.\n"
            + "!" * 60)
    else:
        n_total = mode_report.labels.size
        invalid_frac = mode_report.n_invalid / n_total if n_total else 0.0
        loud = (not mode_report.weights_reliable) or invalid_frac > 0.05
        if loud:
            logger.warning(
                "!" * 60 + "\n"
                f"MODE REPORT FLAGGED: {mode_report.n_modes} mode(s), "
                f"{mode_report.n_invalid}/{n_total} draws ({invalid_frac:.1%}) "
                "rejected as invalid, weights "
                f"{'validated' if mode_report.weights_reliable else 'UNRELIABLE'}. "
                f"See {prefix}_modes.txt before trusting these tables.\n"
                + "!" * 60)
        else:
            logger.info(
                f"Mode report: {mode_report.n_modes} mode(s), "
                f"{mode_report.n_invalid}/{n_total} draws rejected as "
                f"invalid; see {prefix}_modes.txt")

    _persist_trace(idata, trace_path)
    logger.info(f"Rewrote {trace_path} with posterior mode labels attached.")


if __name__ == "__main__":
    main()
