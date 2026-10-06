"""Console entry point: exozippy-summary <config.yaml>

Draws a finished fit's one-page summary figure -- transits, RVs, SED and Kiel
diagram at the best-fit draw, under a header of each planet's P, R_P, M_P and
e -- from the config it ran with and its saved ``<prefix>_trace.nc``, without
re-sampling.  The figure is ``outputs/summary_plot.create_summary_plot``; this
is its command line.

Nothing beyond the config is needed: the title is its ``run: name:``, each
instrument is shown by its ``label:`` (else its ``name:``), and the TESS files
are grouped by cadence.  To override any of that without editing the config,
an optional ``--options`` YAML file takes ``create_summary_plot``'s keywords
(the mappings are awkward as flags), e.g.::

    labels:
      TCS_MuSCAT2_UT20251116_9: "MuSCAT2 ($i'$)"
    transit_bin:
      "TESS 120 s": 10

A flag given on the command line wins over the same key in the file.

Logging goes to the console only.  ``setup_logging`` would open
``<prefix>.log`` for writing, and that file is the fit's own log.
"""

import logging
import sys

import click

from .outputs.summary_plot import create_summary_plot
from .yamlio import load_yaml

#: The --options file's vocabulary: create_summary_plot's keywords.
OPTION_KEYS = (
    "title",
    "labels",
    "transit_groups",
    "transit_bin",
    "transit_spacing",
    "figsize",
)


def _console_logging(level):
    log = logging.getLogger("exozippy")
    log.setLevel(getattr(logging, level.upper()))
    if not any(isinstance(h, logging.StreamHandler) for h in log.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(handler)


def _read_options(path):
    if path is None:
        return {}
    options = load_yaml(path) or {}
    if not isinstance(options, dict):
        raise click.UsageError(
            f"--options file {path} must be a mapping of keywords."
        )
    unknown = sorted(set(options) - set(OPTION_KEYS))
    if unknown:
        raise click.UsageError(
            f"--options file {path} has unknown key(s) {unknown}; the "
            f"keywords are {list(OPTION_KEYS)}."
        )
    return options


@click.command()
@click.argument("config_file")
@click.option(
    "--output",
    "-o",
    default=None,
    help="Output file (default <prefix>_mcmc_summary.pdf); the format "
    "follows the extension.",
)
@click.option(
    "--title",
    default=None,
    help="Figure title (default: the config's run: name:).",
)
@click.option(
    "--transit-bin",
    type=float,
    default=None,
    help="Bin every phased transit to this many minutes.",
)
@click.option(
    "--options",
    "options_file",
    default=None,
    help="YAML file of create_summary_plot keywords: "
    + ", ".join(OPTION_KEYS)
    + ".",
)
@click.option(
    "--logger-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING"], case_sensitive=False),
    help="Console logging level.",
)
def main(config_file, output, title, transit_bin, options_file, logger_level):
    """Draw the one-page summary figure of a finished fit.

    CONFIG_FILE is the same system YAML passed to `exozippy`; its `prefix:`
    locates the saved trace (<prefix>_trace.nc), and the figure is written
    to <prefix>_mcmc_summary.pdf unless --output says otherwise.
    """
    _console_logging(logger_level)
    options = _read_options(options_file)
    if title is not None:
        options["title"] = title
    if transit_bin is not None:
        options["transit_bin"] = transit_bin
    if options.get("figsize") is not None:
        options["figsize"] = tuple(options["figsize"])
    out = create_summary_plot(config_file, output, **options)
    click.echo(f"Wrote {out}")


if __name__ == "__main__":
    main()
