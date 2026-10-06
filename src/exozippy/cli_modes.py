"""Console entry point: exozippy-modes <config.yaml>

Reprocesses a previously saved trace (``<prefix>_trace.nc``) WITHOUT
re-sampling, by running the fit itself with ``sampler: {recompute_trace:
false}`` (review 1.3.9).  That IS the supported reprocessing path under the
three-phase ruling (review 2.14.12, JDE 2026-09-25): it reloads the trace,
checks it against the model this config builds, finishes it if its run
died before the post-save step, and then runs the whole live wrap-up --
the degeneracy fold, branch resolution, the burn-in / stuck-chain trim,
mode identification, tables, plots, the modeling draft and the restart
file -- through ``run._wrap_up``, the code a live fit runs.

This CLI used to be a second, partial copy of that wrap-up: it called
``report_pipeline.build_mode_reports`` directly and so skipped the fold and
the burn-in trim, and its tables included the transient and both node
labels while its docstring claimed it "can never drift from what a live fit
produces".  A thin wrapper over the live path cannot drift.

Differences from typing the same thing yourself are deliberately few:

* it refuses outright when no trace exists (``recompute_trace: false``
  with no trace on disk would SAMPLE), and
* its options set the matching ``modes:`` keys for this run only
  (``--force`` is ``modes: {force: true}``: emit the reports from a trace
  past the invalid-draw gate, for forensics).

It no longer "always completes": a trace past the invalid-draw gate raises
exactly as a live fit does unless ``--force``, and a wrap-up failure raises
-- after which the same command reruns from the same saved trace.
"""

import copy
from pathlib import Path

import click

from .yamlio import load_system_config


@click.command()
@click.argument("config_file")
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Write the reports even from a trace past the invalid-draw gate "
    "(sets modes: {force: true} for this run).",
)
@click.option(
    "--logger-level",
    default=None,
    type=click.Choice(["DEBUG", "INFO", "WARNING"], case_sensitive=False),
    help="Logging level (overrides logger_level in config file).",
)
def main(config_file, force, logger_level):
    """Re-run a fit's whole wrap-up from its saved trace, without sampling.

    CONFIG_FILE is the same system YAML passed to `exozippy`; its `prefix:`
    key locates the saved trace (<prefix>_trace.nc).  Equivalent to running
    `exozippy CONFIG_FILE` with `sampler: {recompute_trace: false}`.
    """
    from .run import run_fit

    # An empty or non-mapping config is refused by name (review 2.3.11).
    config = copy.deepcopy(load_system_config(config_file))

    if logger_level:
        config["logger_level"] = logger_level.upper()

    prefix = Path(config.get("prefix", "fitresults/planet"))
    trace_path = Path(str(prefix) + "_trace.nc")
    if not trace_path.exists():
        raise FileNotFoundError(
            f"No saved trace found at {trace_path}. exozippy-modes reprocesses "
            f"an existing trace produced by a live fit; run "
            f"`exozippy {config_file}` first."
        )

    config["sampler"] = dict(config.get("sampler") or {})
    config["sampler"]["recompute_trace"] = False
    if force:
        config["modes"] = dict(config.get("modes") or {})
        config["modes"]["force"] = True

    run_fit(config)


if __name__ == "__main__":
    main()
