"""
Tests for the system summary figure (outputs/summary_plot.py) and its
console script (cli_summary.py).

The figure is a third renderer of the components' Charts: it adds no
physics, so what is tested here is what it OWNS -- the header's number
formatting, the choice of the best-fit draw, the page layout, the name
checks on its keywords, and the end-to-end path from a real trace on disk.
The O-C it draws is the components' meta["residuals"], whose numbers are
pinned in tests/test_plot_data.py.

They follow AAA with Given/When/Then docstrings.
"""

import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pytest
import yaml
from click.testing import CliRunner

from exozippy import cli_summary
from exozippy.components.parameter import PosteriorSummary
from exozippy.outputs import summary_plot as sp

_KELT4_DIR = Path(__file__).parent.parent / "examples" / "kelt4"


# ---------------------------------------------------------------------------
# The header's numbers
# ---------------------------------------------------------------------------


def test_format_value_writes_the_errors_at_the_median_precision():
    """
    Given a median of 0.0841 with errors -0.0612/+0.1104, which the results
      table writes 0.084^{+0.11}_{-0.061},
    When format_value renders it,
    Then all three numbers carry the median's three decimal places, the
      trailing zero of the upper error kept.
    """
    summary = PosteriorSummary(
        median=0.0841, err_minus=0.0612, err_plus=0.1104
    )

    assert sp.format_value(summary) == "0.084^{+0.110}_{-0.061}"


@pytest.mark.parametrize(
    "median, err_minus, err_plus",
    [
        (2.95248098, 9.57e-7, 9.55e-7),
        (10.834, 17.256, 14.410),
        (1234.5, 560.0, 780.0),
        (0.74727, 0.01445, 0.03827),
    ],
)
def test_format_value_rounds_the_median_as_the_table_does(
    median, err_minus, err_plus
):
    """
    Given posterior summaries across many orders of magnitude,
    When format_value renders each,
    Then its median is the number PosteriorSummary.format puts in the table,
      and each error has exactly the median's number of decimal places.
    """
    summary = PosteriorSummary(median, err_minus, err_plus)

    text = sp.format_value(summary)

    if r" \pm " in text:
        med, err = text.split(r" \pm ")
        errs = [err]
    else:
        med, rest = text.split("^{+")
        hi, lo = rest.rstrip("}").split("}_{-")
        errs = [hi, lo]
    assert float(med) == float(summary.format().median)
    places = len(med.split(".")[1]) if "." in med else 0
    for err in errs:
        assert (len(err.split(".")[1]) if "." in err else 0) == places


def test_format_value_collapses_equal_errors_and_marks_fixed_values():
    """
    Given a symmetric posterior, a zero-spread (fixed) one and a NaN one,
    When format_value renders them,
    Then the first is written with \\pm, the second with \\equiv as the table
      writes a fixed value, and the third has nothing to report (None).
    """
    assert sp.format_value(PosteriorSummary(5.0, 0.1, 0.1)) == r"5.00 \pm 0.10"
    assert sp.format_value(PosteriorSummary(0.0, 0.0, 0.0)) == r"\equiv 0"
    assert sp.format_value(PosteriorSummary(float("nan"), 1.0, 1.0)) is None


# ---------------------------------------------------------------------------
# The best-fit draw
# ---------------------------------------------------------------------------


class _Param:
    """A Parameter stand-in: user -> internal divides by ``factor``."""

    def __init__(self, factor):
        self.factor = factor

    def to_internal(self, val):
        return np.asarray(val) / self.factor


def _stub_system():
    return SimpleNamespace(
        get_parameter_lookup=lambda: {
            "orbit.period": _Param(2.0),
            "orbit.logP": _Param(1.0),
        },
        report_only_labels=lambda: ["planet.t14"],
    )


def _stub_idata(lp, mode=None):
    lp = np.asarray(lp, dtype=float)
    period = np.arange(lp.size, dtype=float).reshape(lp.shape) + 10.0
    posterior = {
        "orbit.period": period,
        "orbit.logP_raw": period * 100.0,
        "planet.t14": period * 0.0,
    }
    if mode is not None:
        posterior["mode"] = np.asarray(mode)
    return az.from_dict({"posterior": posterior, "sample_stats": {"lp": lp}})


def test_best_fit_point_skips_a_draw_the_mode_pass_rejected():
    """
    Given a trace whose highest lp belongs to a draw labelled -1 (the
      runaway-lp failure: a finite, enormous lp on a numerically invalid
      draw),
    When best_fit_point picks the draw,
    Then it takes the best VALID draw, converts its physical values to
      internal units, passes the *_raw variable through untouched, and leaves
      out the mode label and the report-only Deterministics.
    """
    idata = _stub_idata(
        lp=[[0.0, 5.0, 3.0], [1e6, 4.0, 1.0]],
        mode=[[0, 0, 0], [-1, 0, 0]],
    )

    point, (chain, draw), lp = sp.best_fit_point(_stub_system(), idata)

    assert (chain, draw) == (0, 1)
    assert lp == 5.0
    period = idata.posterior["orbit.period"].values[0, 1]
    assert point["orbit.period"] == pytest.approx(period / 2.0)
    assert point["orbit.logP_raw"] == pytest.approx(period * 100.0)
    assert "mode" not in point
    assert "planet.t14" not in point


def test_best_fit_point_refuses_a_trace_with_no_rankable_draw():
    """
    Given a trace with no sample_stats lp, and one whose every draw was
      rejected by the mode pass,
    When best_fit_point is asked for the best-fit draw,
    Then it raises rather than guessing one.
    """
    no_lp = az.from_dict({"posterior": {"orbit.period": np.ones((1, 3))}})
    with pytest.raises(ValueError, match="no sample_stats"):
        sp.best_fit_point(_stub_system(), no_lp)

    rejected = _stub_idata(lp=[[1.0, 2.0]], mode=[[-1, -1]])
    with pytest.raises(ValueError, match="No valid draw"):
        sp.best_fit_point(_stub_system(), rejected)


# ---------------------------------------------------------------------------
# Layout and keyword checks
# ---------------------------------------------------------------------------


def test_place_reproduces_the_classic_one_planet_page():
    """
    Given the panels of a one-planet transit + RV + SED + MIST fit, in
      reading order,
    When _place lays them out,
    Then the transits sit over the SED on the left, and the RVs against
      time, the folded RVs and the Kiel diagram run down the right, on a
      two-column, three-row grid.
    """
    panels = [
        sp._Panel("transit", [], 2),
        sp._Panel("rv_time", [], 1),
        sp._Panel("rv_phase", [], 1),
        sp._Panel("sed", [], 1),
        sp._Panel("kiel", [], 1),
    ]

    placed, ncols, nrows = sp._place(panels)

    assert (ncols, nrows) == (2, 3)
    assert [(p.kind, col, row) for p, col, row in placed] == [
        ("transit", 0, 0),
        ("rv_time", 1, 0),
        ("rv_phase", 1, 1),
        ("sed", 0, 2),
        ("kiel", 1, 2),
    ]
    _, ncols, _ = sp._place([sp._Panel("sed", [], 1)])
    assert ncols == 1


def test_bin_averages_within_each_bin_and_drops_empty_ones():
    """
    Given points falling in bins 0, 1 and 3 of width 1 (bin 2 empty),
    When _bin averages them,
    Then it returns the three occupied bins' centers and means.
    """
    x = np.array([0.0, 0.1, 0.2, 1.05, 1.1, 3.0])
    y = np.array([1.0, 2.0, 3.0, 4.0, 6.0, 10.0])

    centers, means = sp._bin(x, y, 1.0)

    np.testing.assert_allclose(centers, [0.5, 1.5, 3.5])
    np.testing.assert_allclose(means, [2.0, 5.0, 10.0])


def test_an_unknown_instrument_name_raises_with_a_suggestion():
    """
    Given a keyword naming an instrument with the wrong case,
    When the summary checks the names against the fit's instruments,
    Then it raises, naming the near miss -- suggested, never accepted, as
      every user-facing name in EXOZIPPy is case-sensitive.
    """
    with pytest.raises(
        ValueError, match=r"'TESS_s16' \(did you mean 'TESS_S16'"
    ):
        sp._check_names("labels", {"TESS_s16": "x"}, ["TESS_S16", "HIRES"])


def test_cli_refuses_an_unknown_options_key(tmp_path):
    """
    Given an --options file with a misspelled keyword,
    When exozippy-summary runs,
    Then it exits with a usage error naming the key, before building
      anything.
    """
    options = tmp_path / "options.yaml"
    options.write_text("titel: TOI-1234\n")

    result = CliRunner().invoke(
        cli_summary.main,
        [str(tmp_path / "no_such_fit.yaml"), "--options", str(options)],
    )

    assert result.exit_code == 2
    assert "titel" in result.output


# ---------------------------------------------------------------------------
# End to end, from a real trace on disk (slow)
# ---------------------------------------------------------------------------


def _kelt4_config():
    """kelt4: two RV instruments plus one TESS sector, the smallest fit that
    makes a transit stack and both RV panels."""
    return {
        "run": {"name": "KELT-4A"},
        "prefix": "fitresults/KELT-4A",
        "star": [{"name": "A", "mist": False}],
        "planet": [{"name": "b"}],
        "orbit": [{"name": "b", "primary": ["A"], "companion": ["b"]}],
        "transit": [
            {
                "name": "TESS_S48",
                "file": "n20220130.TESS.TESS.TIC165297570.S48.0120.SPOC.dat",
                "band": "TESS",
                "exptime": 2.0,
                "ninterp": 1.0,
            }
        ],
        "band": [{"name": "TESS", "filter": "TESS"}],
        "rvinstrument": [
            {"name": "HIRES", "file": "KELT-4b.HIRES.rv"},
            {"name": "TRES", "file": "KELT-4b.TRES.rv"},
        ],
        "parameter_file": "summary_test.params.yaml",
        # nutpie, not PyMC NUTS: it compiles through numba, while the C
        # backend's code for this model's gradient exceeds Apple clang's
        # 256-deep bracket nesting limit.  A core dependency on every
        # platform, like the hat3 example's sampler.
        "sampler": {
            "method": "nutpie",
            "tune": 5,
            "draws": 4,
            "chains": 1,
            "cores": 1,
            "measure_scales": False,
            "recompute_trace": True,
        },
        "modeling": {"compile": False},
    }


# Star A's literature values as priors (there is no SED or relation to
# constrain it here) and the transit geometry of the example's own params.
# Not kelt4_rv+transit+sed.params.yaml, which also names stars B and C.
_KELT4_PARAMS = {
    "star.A.radius": {"initval": 1.610, "mu": 1.610, "sigma": 0.05},
    "star.A.mass": {"initval": 1.204, "mu": 1.204, "sigma": 0.05},
    "star.A.teff": {"initval": 6207, "mu": 6207, "sigma": 100},
    "star.A.feh": {"initval": -0.116, "mu": -0.116, "sigma": 0.08},
    "planet.b.radius": {"initval": 1.706},
    "orbit.b.period": {"initval": 2.9895933},
    "orbit.b.tc": {"initval": 2459634.3},
    "orbit.b.cosi": {"initval": 0.11996},
}


@pytest.fixture(scope="module")
def kelt4_fit(tmp_path_factory):
    """A finished (tiny) kelt4 fit in a copy of the example directory:
    returns the path of its config.  run_fit is called once per module."""
    if not _KELT4_DIR.is_dir():
        pytest.skip("kelt4 example not present")
    from exozippy.run import run_fit

    work_dir = tmp_path_factory.mktemp("summary") / "kelt4"
    shutil.copytree(
        _KELT4_DIR,
        work_dir,
        ignore=shutil.ignore_patterns("fitresults*", ".#*", "#*#"),
    )
    config = _kelt4_config()
    config_path = work_dir / "summary_test.yaml"
    config_path.write_text(yaml.safe_dump(config))
    (work_dir / config["parameter_file"]).write_text(
        yaml.safe_dump(_KELT4_PARAMS)
    )

    cwd = os.getcwd()
    os.chdir(work_dir)
    try:
        run_fit(config)
    finally:
        os.chdir(cwd)
    return config_path


@pytest.fixture(scope="module")
def kelt4_posterior(kelt4_fit):
    """The kelt4 fit's System, rebuilt, with its reported posterior and
    best-fit point."""
    from exozippy.system import System

    cwd = os.getcwd()
    os.chdir(kelt4_fit.parent)
    try:
        system = System(_kelt4_config())
        system.prepare()
        system.build_model()
        idata = az.from_netcdf("fitresults/KELT-4A_trace.nc")
        posterior = sp.reported_posterior(system, idata)
        point, where, lp = sp.best_fit_point(system, posterior)
    finally:
        os.chdir(cwd)
    return system, posterior, point, where


@pytest.mark.slow
def test_create_summary_plot_writes_the_default_file(kelt4_fit):
    """
    Given a finished fit, and a working directory that is NOT the config's,
    When create_summary_plot is called with just the config path,
    Then it resolves the fit's relative paths against the config's
      directory and writes <prefix>_mcmc_summary.pdf there.
    """
    assert Path.cwd() != kelt4_fit.parent

    out = sp.create_summary_plot(kelt4_fit)

    expected = kelt4_fit.parent / "fitresults" / "KELT-4A_mcmc_summary.pdf"
    assert out == expected.resolve()
    assert out.stat().st_size > 0


@pytest.mark.slow
def test_best_fit_point_is_the_best_reported_draw_in_internal_units(
    kelt4_posterior,
):
    """
    Given the fit's reported (post-burn-in) posterior,
    When best_fit_point picks the draw,
    Then it is the highest-lp draw of that posterior, and its physical
      values convert back through the Parameters to exactly the trace's
      user-unit values at that draw.
    """
    system, posterior, point, (chain, draw) = kelt4_posterior
    lp = posterior.sample_stats["lp"].values
    labels = posterior.posterior["mode"].values

    assert lp[chain, draw] == np.max(np.where(labels < 0, -np.inf, lp))
    lookup = system.get_parameter_lookup()
    for label in ("orbit.period", "planet.mass", "rvinstrument.gamma"):
        np.testing.assert_allclose(
            lookup[label].from_internal(point[label]),
            posterior.posterior[label].values[chain, draw],
            rtol=1e-12,
        )


@pytest.mark.slow
def test_summary_figure_draws_one_panel_per_chart_kind(kelt4_posterior):
    """
    Given the fit at its best-fit point,
    When summary_figure draws it with instrument labels and a binned transit,
    Then the page has the transit stack, the RVs against time and the
      folded RVs (each of the last two with an O-C axis), the header lines
      it was given, the transit labelled as asked, and the folded panel's
      legend carrying the relabelled instruments.
    """
    system, _, point, _ = kelt4_posterior
    header = sp.planet_header_lines(system)

    fig = sp.summary_figure(
        system,
        point,
        title="KELT-4A",
        header_lines=header,
        labels={"TESS_S48": "TESS S48", "HIRES": "Keck/HIRES"},
        transit_bin=10,
    )
    try:
        texts = [t.get_text() for t in fig.texts]
        axes = fig.get_axes()
        ylabels = [ax.get_ylabel() for ax in axes]
        transit_ax = axes[ylabels.index("Normalized Flux + Constant")]
        legend_texts = [
            t.get_text()
            for ax in axes
            if ax.get_legend() is not None
            for t in ax.get_legend().get_texts()
        ]
    finally:
        plt.close(fig)

    assert len(axes) == 5
    assert ylabels.count("O-C [m/s]") == 2
    assert texts[0] == "KELT-4A" and texts[1:] == header
    assert len(header) == 1
    for symbol in ("$P = ", "$R_P = ", "$M_P = ", "$e = "):
        assert symbol in header[0]
    assert "TESS S48" in [t.get_text() for t in transit_ax.texts]
    assert {"Keck/HIRES", "TRES"} <= set(legend_texts)


@pytest.mark.slow
def test_summary_figure_rejects_keywords_that_do_not_fit_the_fit(
    kelt4_posterior,
):
    """
    Given the fit (transit TESS_S48; RVs HIRES and TRES),
    When the summary is asked to group an RV instrument as a transit, to
      bin a row that does not exist, or to label an unknown instrument,
    Then each raises, naming the offending name.
    """
    system, _, point, _ = kelt4_posterior

    with pytest.raises(ValueError, match="'HIRES'"):
        sp.summary_figure(system, point, transit_groups={"x": ["HIRES"]})
    with pytest.raises(ValueError, match="'TESS'"):
        sp.summary_figure(system, point, transit_bin={"TESS": 10})
    with pytest.raises(ValueError, match="'FIES'"):
        sp.summary_figure(system, point, labels={"FIES": "FIES"})
    plt.close("all")


@pytest.mark.slow
def test_a_trace_from_another_model_is_refused(kelt4_fit):
    """
    Given the fit's trace and a config that builds a DIFFERENT model (one RV
      instrument dropped) with the same prefix,
    When create_summary_plot is called on that config,
    Then it raises StaleTraceError, as every trace reload does.
    """
    from exozippy.trace_meta import StaleTraceError

    config = _kelt4_config()
    config["rvinstrument"] = config["rvinstrument"][:1]
    other = kelt4_fit.parent / "summary_test_other.yaml"
    other.write_text(yaml.safe_dump(config))

    with pytest.raises(StaleTraceError):
        sp.create_summary_plot(other, kelt4_fit.parent / "never.pdf")


@pytest.mark.slow
def test_cli_writes_the_figure_with_an_options_file(kelt4_fit, tmp_path):
    """
    Given the finished fit and an --options file of labels and a per-row bin,
    When exozippy-summary runs with --output and a --title flag,
    Then it exits cleanly and writes the requested file.
    """
    options = tmp_path / "options.yaml"
    options.write_text(
        yaml.safe_dump(
            {
                "title": "overridden by the flag",
                "labels": {"HIRES": "Keck/HIRES"},
                "transit_bin": {"TESS_S48": 10},
                "figsize": [17, 14],
            }
        )
    )
    out = tmp_path / "summary.png"

    result = CliRunner().invoke(
        cli_summary.main,
        [
            str(kelt4_fit),
            "--output",
            str(out),
            "--title",
            "KELT-4A",
            "--options",
            str(options),
        ],
    )

    assert result.exit_code == 0, result.output + repr(result.exception)
    assert out.stat().st_size > 0
