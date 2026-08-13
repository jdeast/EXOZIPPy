"""
Tests for posterior mode identification and reporting (outputs/modes.py).

identify_modes clusters posterior draws in the raw (unconstrained) sampled
space, rejects invalid draws (runaway/stuck chains pinned at bounds with
non-finite or implausible lp), merges over-split clusters with a density-dip
test, and labels every draw with a mode index (-1 = invalid/unassigned).
Draw-count occupancy weights are only trusted when the sampler demonstrably
mixed between modes; otherwise the report flags them UNRELIABLE.

Parameter.compute_mode_summaries and the latex/csv builders consume the
labels to produce per-mode summaries and one table column per mode.
"""

import arviz as az
import numpy as np
import pytest

from exozippy.components.parameter import Parameter
from exozippy.outputs.latex import build_csv_output, build_latex_output
from exozippy.outputs.modes import (
    DEFAULT_MAX_INVALID_FRAC,
    MODE_FAILED,
    MODE_NO_VALID_DRAWS,
    MODE_OK,
    ModeInfo,
    ModeReport,
    NoValidDrawsError,
    check_invalid_frac,
    identify_modes,
    markov_indicator_iact,
    mode_status_to_text,
    mode_suffix,
    transition_stats,
    weight_ess,
)
from exozippy.outputs.report_pipeline import build_mode_reports

N_CHAIN, N_DRAW = 8, 1500
N = N_CHAIN * N_DRAW


class _StubSystem:
    """Component-free system, for exercising the table builders alone."""

    name = "toy"

    def get_all_components(self):
        return []


def _make_idata(posterior, lp):
    """Build a minimal InferenceData from flat (N,) arrays."""
    return az.from_dict(
        {
            "posterior": {
                k: np.asarray(v).reshape(N_CHAIN, N_DRAW)
                for k, v in posterior.items()
            },
            "sample_stats": {"lp": np.asarray(lp).reshape(N_CHAIN, N_DRAW)},
        }
    )


def _two_mode_idata(rng, w2=0.3, sep=8.0, garbage=0):
    """Two Gaussian modes mixed within every chain, plus optional runaway
    draws with absurd raw values and lp; returns (idata, true_labels)."""
    labels = (rng.random(N) < w2).astype(int)
    a = rng.normal(0, 1, N) + sep * labels
    b = rng.normal(0, 1, N) - 0.6 * sep * labels
    c = rng.normal(0, 1, N)
    lp = rng.normal(1000, 3, N) - 5 * labels
    if garbage:
        bad = rng.choice(N, garbage, replace=False)
        a[bad] = 1e20
        lp[bad] = 1e30
        labels[bad] = -1
    return _make_idata({"a_raw": a, "b_raw": b, "c_raw": c}, lp), labels


# ----------------------------------------------------------------------
# identify_modes core behavior
# ----------------------------------------------------------------------


def test_two_modes_weights_and_labels_recovered():
    """
    Given a trace with two well-separated modes (70/30) mixed within every
      chain plus 200 runaway draws,
    When identify_modes runs,
    Then it finds 2 modes with weights near 0.7/0.3, rejects the runaway
      draws as invalid, matches the true labels, and validates the weights.
    """
    rng = np.random.default_rng(42)
    idata, truth = _two_mode_idata(rng, garbage=200)

    rep = identify_modes(idata)

    assert rep.n_modes == 2
    assert rep.weights[0] == pytest.approx(0.7, abs=0.03)
    assert rep.weights[1] == pytest.approx(0.3, abs=0.03)
    assert rep.n_invalid == 200
    assert rep.weights_reliable
    found = rep.labels.ravel()
    ok = (found >= 0) & (truth >= 0)
    assert ((found[ok] == 1) == (truth[ok] == 1)).mean() > 0.99


def test_single_curved_mode_not_split():
    """
    Given a unimodal but banana-shaped (curved, correlated) posterior,
    When identify_modes runs,
    Then it reports exactly one mode (the dip-merge pass undoes any k-means
      fragmentation).
    """
    rng = np.random.default_rng(7)
    t = rng.normal(0, 1.5, N)
    idata = _make_idata(
        {"a_raw": t, "b_raw": t**2 + rng.normal(0, 0.3, N)},
        rng.normal(0, 1, N),
    )

    rep = identify_modes(idata)

    assert rep.n_modes == 1
    assert rep.provenance == "unimodal"
    assert rep.weights_reliable


def test_stuck_chains_flagged_unreliable():
    """
    Given two modes where each chain sits in only one mode for its whole
      length (no inter-mode transitions),
    When identify_modes runs,
    Then both modes are found but the occupancy weights are flagged
      UNRELIABLE (they reflect initialization, not posterior mass).
    """
    rng = np.random.default_rng(3)
    chain_mode = np.repeat([0] * 6 + [1] * 2, N_DRAW)
    a = rng.normal(0, 1, N) + 10 * chain_mode
    idata = _make_idata({"a_raw": a}, rng.normal(0, 1, N))

    rep = identify_modes(idata)

    assert rep.n_modes == 2
    assert not rep.weights_reliable
    assert "UNRELIABLE" in rep.provenance


def test_mode_variable_attached_to_idata():
    """
    Given a multimodal trace,
    When identify_modes runs with attach=True (default),
    Then idata.posterior['mode'] holds the per-draw labels with n_modes,
      weights, and provenance in its attrs, and survives az.extract aligned
      with the other variables.
    """
    rng = np.random.default_rng(11)
    idata, _ = _two_mode_idata(rng)

    rep = identify_modes(idata)

    assert "mode" in idata.posterior
    da = idata.posterior["mode"]
    assert da.dims == ("chain", "draw")
    assert da.attrs["n_modes"] == rep.n_modes
    assert np.array_equal(da.values, rep.labels)

    extracted = az.extract(idata, keep_dataset=True)
    assert "mode" in extracted
    assert extracted["mode"].values.shape == (N,)


def test_report_text_contains_key_facts():
    """
    Given a multimodal report,
    When to_text renders it,
    Then the human-readable report states the mode count, weights, and
      invalid-draw count.
    """
    rng = np.random.default_rng(5)
    idata, _ = _two_mode_idata(rng, garbage=100)

    rep = identify_modes(idata)
    text = rep.to_text()

    assert "modes found: 2" in text
    assert "100 invalid" in text
    assert "mode 1:" in text and "mode 2:" in text


def test_all_invalid_raises():
    """
    Given a trace where every draw has non-finite lp,
    When identify_modes runs,
    Then it raises rather than clustering garbage.
    """
    rng = np.random.default_rng(1)
    idata = _make_idata({"a_raw": rng.normal(0, 1, N)}, np.full(N, np.inf))

    with pytest.raises(ValueError, match="no valid draws"):
        identify_modes(idata)


def test_fallback_to_physical_vars_without_raw():
    """
    Given a trace with no *_raw variables,
    When identify_modes runs,
    Then it clusters on the physical variables and notes the fallback.
    """
    rng = np.random.default_rng(9)
    labels = (rng.random(N) < 0.5).astype(int)
    idata = _make_idata(
        {"x": rng.normal(0, 1, N) + 12 * labels}, rng.normal(0, 1, N)
    )

    rep = identify_modes(idata)

    assert rep.n_modes == 2
    assert any("no *_raw" in n for n in rep.notes)


# ----------------------------------------------------------------------
# Parameter per-mode summaries
# ----------------------------------------------------------------------


def _sampled_param(posterior):
    p = Parameter(
        label="star.teff",
        latex=r"T_{\rm eff}",
        description="Effective temperature",
        initval=5000.0,
        lower=3000.0,
        upper=7000.0,
        init_scale=100.0,
    )
    p.posterior = posterior
    return p


def test_parameter_mode_summaries_split_by_label():
    """
    Given a scalar parameter whose posterior is two blocks of constant
      values (1.0 for mode 0, 3.0 for mode 1),
    When compute_mode_summaries runs with the matching labels,
    Then each mode's summary is the median of its own block only.
    """
    post = np.array([1.0] * 700 + [3.0] * 300)
    labels = np.array([0] * 700 + [1] * 300)
    p = _sampled_param(post)

    summaries = p.compute_mode_summaries(labels, 2)

    assert summaries[0].median == pytest.approx(1.0)
    assert summaries[1].median == pytest.approx(3.0)
    assert p.mode_summaries is summaries


def test_parameter_mode_summaries_vector_and_empty_mode():
    """
    Given a vector parameter (2 elements) and a mode with zero assigned
      draws,
    When compute_mode_summaries runs,
    Then per-element summaries are returned per mode and the empty mode
      yields NaN summaries instead of raising.
    """
    post = np.vstack([np.linspace(0, 1, 100), np.linspace(10, 11, 100)])
    labels = np.zeros(100, dtype=int)
    p = _sampled_param(post)

    summaries = p.compute_mode_summaries(labels, 2)

    assert isinstance(summaries[0], list) and len(summaries[0]) == 2
    assert summaries[0][1].median == pytest.approx(10.5)
    empty = (
        summaries[1] if not isinstance(summaries[1], list) else summaries[1][0]
    )
    assert np.isnan(empty.median)


def test_mode_latex_defs_use_suffixed_macros():
    """
    Given a sampled parameter with two mode summaries,
    When to_latex_mode_defs renders,
    Then it defines one macro per mode with the modeone/modetwo suffixes.
    """
    post = np.array([1.0] * 50 + [3.0] * 50)
    labels = np.array([0] * 50 + [1] * 50)
    p = _sampled_param(post)
    p.compute_mode_summaries(labels, 2)

    defs = p.to_latex_mode_defs()

    assert mode_suffix(0) == "modeone" and mode_suffix(1) == "modetwo"
    assert f"\\{p.latex_varname}modeone" in defs
    assert f"\\{p.latex_varname}modetwo" in defs


def test_table_line_one_value_cell_per_mode():
    """
    Given a sampled parameter and two mode suffixes,
    When to_table_line renders with mode_suffixes,
    Then the row contains one value cell per mode (5 columns total), while
      the default call keeps the original 4-column layout.
    """
    post = np.array([1.0] * 50 + [3.0] * 50)
    p = _sampled_param(post)
    p.compute_summary()

    row_multi = p.to_table_line(mode_suffixes=["modeone", "modetwo"])
    row_single = p.to_table_line()

    assert row_multi.count("&") == 4  # param & desc & val1 & val2 & prior
    assert f"\\{p.latex_varname}modeone" in row_multi
    assert f"\\{p.latex_varname}modetwo" in row_multi
    assert row_single.count("&") == 3  # param & desc & val & prior
    assert f"\\{p.latex_varname}modeone" not in row_single


# ----------------------------------------------------------------------
# PROMPT 1: loud invalid-draw reporting
# ----------------------------------------------------------------------


def test_invalid_draws_warn_loudly(caplog):
    """
    Given a trace with invalid draws spanning every rejection reason
      (non-finite raw values, non-finite lp, lp over the ceiling),
    When identify_modes runs,
    Then it logs a warning naming the count and reason breakdown, and
      ModeReport.to_text() carries a prominent WARNING banner.
    """
    rng = np.random.default_rng(21)
    a = rng.normal(0, 1, N)
    lp = rng.normal(1000, 3, N)
    a[0:5] = np.nan  # nonfinite-raw
    lp[5:10] = np.nan  # nonfinite-lp
    lp[10:15] = 1e20  # lp-ceiling
    idata = _make_idata({"a_raw": a}, lp)

    with caplog.at_level("WARNING", logger="exozippy.outputs.modes"):
        rep = identify_modes(idata)

    assert rep.n_invalid == 15
    assert rep.invalid_reason_counts == {
        "nonfinite-raw": 5,
        "nonfinite-lp": 5,
        "lp-ceiling": 5,
    }
    assert any(
        "rejected as numerically invalid" in r.message for r in caplog.records
    )

    text = rep.to_text()
    assert "WARNING" in text
    assert "model or sampler bug" in text


def test_no_warning_when_no_invalid_draws(caplog):
    """
    Given a clean trace with no invalid draws,
    When identify_modes runs,
    Then no invalid-draw warning is logged and to_text() has no banner.
    """
    rng = np.random.default_rng(23)
    a = rng.normal(0, 1, N)
    idata = _make_idata({"a_raw": a}, rng.normal(1000, 3, N))

    with caplog.at_level("WARNING", logger="exozippy.outputs.modes"):
        rep = identify_modes(idata)

    assert rep.n_invalid == 0
    assert not any(
        "rejected as numerically invalid" in r.message for r in caplog.records
    )
    assert "WARNING" not in rep.to_text()


def _fake_report(n_invalid, n_total=1000):
    labels = np.zeros(n_total, dtype=int)
    return ModeReport(
        labels=labels,
        modes=[],
        n_valid=n_total - n_invalid,
        n_invalid=n_invalid,
        n_unassigned=0,
        provenance="unimodal",
        weights_reliable=True,
        n_transitions=0,
        feature_vars=[],
    )


def test_check_invalid_frac_raises_above_threshold():
    """
    Given a mode report whose invalid fraction exceeds max_invalid_frac,
    When check_invalid_frac runs,
    Then it raises, naming the threshold and the already-written paths.
    """
    rep = _fake_report(n_invalid=20, n_total=1000)  # 2%

    with pytest.raises(RuntimeError, match="exceeding max_invalid_frac"):
        check_invalid_frac(
            rep,
            max_invalid_frac=0.01,
            trace_path="foo_trace.nc",
            modes_path="foo_modes.txt",
        )


def test_check_invalid_frac_below_threshold_is_noop():
    """
    Given a mode report whose invalid fraction is below max_invalid_frac,
    When check_invalid_frac runs,
    Then it does not raise.
    """
    rep = _fake_report(n_invalid=1, n_total=1000)  # 0.1%
    check_invalid_frac(rep, max_invalid_frac=DEFAULT_MAX_INVALID_FRAC)


def test_check_invalid_frac_force_overrides():
    """
    Given a mode report whose invalid fraction exceeds the threshold,
    When check_invalid_frac runs with force=True,
    Then it does not raise, enabling forensic re-processing.
    """
    rep = _fake_report(n_invalid=20, n_total=1000)
    check_invalid_frac(rep, max_invalid_frac=0.01, force=True)


def test_check_invalid_frac_noop_when_no_invalid_draws():
    """
    Given a mode report with zero invalid draws,
    When check_invalid_frac runs,
    Then it never raises regardless of threshold.
    """
    rep = _fake_report(n_invalid=0, n_total=1000)
    check_invalid_frac(rep, max_invalid_frac=0.0)


# ----------------------------------------------------------------------
# PROMPT 1: combined-def suppression in the multimodal LaTeX table
# ----------------------------------------------------------------------


class _FakeComp:
    label = "star"


def _fake_system(comp):
    class _Sys:
        name = "test"

        def get_all_components(self):
            return [comp]

    return _Sys()


def _two_mode_report(n_invalid=0):
    modes = [
        ModeInfo(
            index=0,
            weight=0.5,
            n_draws=50,
            lp_med=0.0,
            lp_max=0.0,
            delta_lp_max=0.0,
            per_chain_weight=np.array([0.5]),
        ),
        ModeInfo(
            index=1,
            weight=0.5,
            n_draws=50,
            lp_med=0.0,
            lp_max=0.0,
            delta_lp_max=0.0,
            per_chain_weight=np.array([0.5]),
        ),
    ]
    return ModeReport(
        labels=np.zeros((1, 100), dtype=int),
        modes=modes,
        n_valid=100 - n_invalid,
        n_invalid=n_invalid,
        n_unassigned=0,
        provenance="occupancy (validated: ...)",
        weights_reliable=True,
        n_transitions=20,
        feature_vars=["a_raw"],
    )


def test_multimode_latex_suppresses_combined_defs(tmp_path):
    """
    Given a sampled parameter and a two-mode report,
    When build_latex_output runs,
    Then the unsuffixed (pooled-across-modes) macro def is absent, the
      per-mode defs are present, and tablecomments explains the suppression.
    """
    post = np.array([1.0] * 50 + [3.0] * 50)
    labels = np.array([0] * 50 + [1] * 50)
    p = _sampled_param(post)
    comp = _FakeComp()
    comp.teff = p
    sys_obj = _fake_system(comp)
    sys_obj.mode_labels = labels

    mode_report = _two_mode_report()

    var_path = tmp_path / "vars.tex"
    tmpl_path = tmp_path / "tmpl.tex"
    build_latex_output(
        sys_obj,
        var_filename=str(var_path),
        table_filename=str(tmpl_path),
        mode_report=mode_report,
    )

    var_text = var_path.read_text()
    assert f"\\providecommand{{\\{p.latex_varname}}}" not in var_text
    assert f"\\{p.latex_varname}modeone" in var_text
    assert f"\\{p.latex_varname}modetwo" in var_text
    assert "suppressed" in tmpl_path.read_text()


def test_single_mode_latex_output_unchanged(tmp_path):
    """
    Given a sampled parameter,
    When build_latex_output runs with mode_report=None vs a unimodal
      mode_report (n_modes == 1, no invalid draws),
    Then the emitted variable definitions are byte-identical.
    """
    post = np.array([1.0] * 100)

    def _sys_with_param():
        comp = _FakeComp()
        comp.teff = _sampled_param(post)
        return _fake_system(comp)

    var1, tmpl1 = tmp_path / "v1.tex", tmp_path / "t1.tex"
    build_latex_output(
        _sys_with_param(), var_filename=str(var1), table_filename=str(tmpl1)
    )

    unimodal_report = ModeReport(
        labels=np.zeros((1, 100), dtype=int),
        modes=[
            ModeInfo(
                index=0,
                weight=1.0,
                n_draws=100,
                lp_med=0.0,
                lp_max=0.0,
                delta_lp_max=0.0,
                per_chain_weight=np.array([1.0]),
            )
        ],
        n_valid=100,
        n_invalid=0,
        n_unassigned=0,
        provenance="unimodal",
        weights_reliable=True,
        n_transitions=0,
        feature_vars=["a_raw"],
    )

    var2, tmpl2 = tmp_path / "v2.tex", tmp_path / "t2.tex"
    build_latex_output(
        _sys_with_param(),
        var_filename=str(var2),
        table_filename=str(tmpl2),
        mode_report=unimodal_report,
    )

    assert var1.read_text() == var2.read_text()


def test_latex_tablecomments_notes_invalid_draws(tmp_path):
    """
    Given a mode report with invalid draws,
    When build_latex_output runs,
    Then the table template's tablecomments names the invalid count and
      warns to investigate before trusting the table.
    """
    post = np.array([1.0] * 100)
    comp = _FakeComp()
    comp.teff = _sampled_param(post)
    sys_obj = _fake_system(comp)

    mode_report = ModeReport(
        labels=np.zeros((1, 100), dtype=int),
        modes=[
            ModeInfo(
                index=0,
                weight=1.0,
                n_draws=95,
                lp_med=0.0,
                lp_max=0.0,
                delta_lp_max=0.0,
                per_chain_weight=np.array([1.0]),
            )
        ],
        n_valid=95,
        n_invalid=5,
        n_unassigned=0,
        provenance="unimodal",
        weights_reliable=True,
        n_transitions=0,
        feature_vars=["a_raw"],
    )

    tmpl_path = tmp_path / "t.tex"
    build_latex_output(
        sys_obj,
        var_filename=str(tmp_path / "v.tex"),
        table_filename=str(tmpl_path),
        mode_report=mode_report,
    )

    text = tmpl_path.read_text()
    assert "5 draws" in text
    assert "model or sampler bug" in text
    # A bare % starts a LaTeX comment and would swallow the closing brace
    # of \tablecomments{...}: the percentage must arrive escaped.
    comments = next(
        ln for ln in text.splitlines() if ln.startswith(r"\tablecomments")
    )
    assert r"5.00\%" in comments
    assert "%" not in comments.replace(r"\%", "")


# ----------------------------------------------------------------------
# Non-LaTeX strings reaching LaTeX (review 2.8.2, generalized)
#
# Every user- or YAML-supplied string in the table -- the output prefix,
# instance (instrument/band/star) names, component section labels,
# parameter descriptions -- is data, not markup.  Real generated tables
# under examples/ carry `\textit{MEARTH_20090513:}` and a description
# column reading `Binary lens mass ratio (M_2 / M_1)`; neither compiles.
# `Parameter.latex`, `unit_latex` and `table_note` ARE markup and must stay
# unescaped.
# ----------------------------------------------------------------------


def test_instance_names_and_descriptions_are_escaped(tmp_path):
    """
    Given a vector parameter whose instance names contain underscores
      (MEARTH_20090513) and whose description carries deliberate math,
    When build_latex_output writes the table,
    Then the instance sub-header and component label are escaped (they are
      DATA -- user-chosen names), while the description and the latex
      symbol pass through untouched: descriptions are trusted LaTeX now
      (same contract as table_note; the defaults.yaml audit in
      tests/test_prose.py guards raw specials in shipped descriptions).
    """
    # ARRANGE
    p = Parameter(
        label="transit.depth",
        latex=r"\delta_{\rm t}",
        description=r"mass ratio ($M_2/M_1$)",
        initval=np.array([0.01, 0.02]),
        lower=0.0,
        upper=1.0,
        names=["MEARTH_20090513", "FLWO_20090601"],
        shape=(2,),
    )
    comp = _FakeComp()
    comp.label = "Transit_Parameters"
    comp.depth = p
    tmpl_path = tmp_path / "t.tex"

    # ACT
    build_latex_output(
        _fake_system(comp),
        var_filename=str(tmp_path / "v.tex"),
        table_filename=str(tmpl_path),
    )

    # ASSERT
    text = tmpl_path.read_text()
    assert r"\textit{MEARTH\_20090513:}" in text
    assert r"mass ratio ($M_2/M_1$)" in text  # trusted LaTeX, unescaped
    assert r"\sidehead{Transit\_Parameters:}" in text
    assert r"\delta_{\rm t}" in text  # the symbol is markup: untouched
    body = [
        ln
        for ln in text.splitlines()
        if ln.startswith(("~~~~", r"\multicolumn", r"\sidehead"))
    ]
    for ln in body:
        # drop the math spans, then no bare underscore may remain
        stripped = "".join(ln.split("$")[::2]).replace(r"\_", "")
        assert "_" not in stripped, ln


# ----------------------------------------------------------------------
# raw-z lp exemption: displaced high-lp basins are modes, not runaways
# ----------------------------------------------------------------------


def test_displaced_high_lp_minority_is_a_mode_not_invalid():
    """
    Given a minority cluster (one chain's worth) displaced far beyond the
      raw-z threshold but with lp ABOVE the bulk's median (the DC2018 event
      128 scenario: 2/54 chains found the true s-branch at +500 nats and
      were discarded as 'raw-z invalid'),
    When identify_modes runs,
    Then those draws are exempted from invalidation and reported as a
      second mode, with no invalid draws at all.
    """
    rng = np.random.default_rng(7)
    a = rng.normal(
        0.0, 0.001, N
    )  # razor-tight bulk: huge robust z for any offset
    lp = rng.normal(1000.0, 3.0, N)
    minority = slice(0, N_DRAW)  # chain 0 entirely in the displaced basin
    a[minority] = rng.normal(5.0, 0.001, N_DRAW)  # z ~ 5000 sigma_bulk
    lp[minority] = rng.normal(1500.0, 3.0, N_DRAW)  # +500 nats: better fit

    rep = identify_modes(_make_idata({"a_raw": a}, lp))

    assert rep.n_invalid == 0
    assert rep.n_modes == 2
    assert any("candidate modes" in n for n in rep.notes)
    # the displaced basin is the best mode (delta_lp_max = 0 by definition
    # of the best); the majority mode trails by ~500 nats
    lp_maxes = sorted((m.lp_max for m in rep.modes), reverse=True)
    assert lp_maxes[0] - lp_maxes[1] > 400


def test_displaced_low_lp_cluster_stays_invalid():
    """
    Given a cluster equally far beyond the raw-z threshold but with lp
      ~1000 nats BELOW the bulk (the runaway/saturated-plateau signature),
    When identify_modes runs,
    Then the lp exemption does not fire and the draws stay invalid.
    """
    rng = np.random.default_rng(11)
    a = rng.normal(0.0, 0.001, N)
    lp = rng.normal(1000.0, 3.0, N)
    runaway = slice(0, 300)
    a[runaway] = rng.normal(5.0, 0.001, 300)
    lp[runaway] = rng.normal(0.0, 3.0, 300)  # -1000 nats: degraded

    rep = identify_modes(_make_idata({"a_raw": a}, lp))

    assert rep.n_invalid == 300
    assert rep.invalid_reason_counts == {"raw-z": 300}
    assert rep.n_modes == 1


# ----------------------------------------------------------------------
# Mode-transition bookkeeping and the occupancy weights' error bar
#
# Occupancy weighting is unbiased when the sampler mixes between modes, but
# its PRECISION is set by the number of independent mode transitions, not by
# the number of draws: a 50000-draw run that switched five times knows the
# weight to roughly 40%.  Nothing reported that, so "0.7" and "0.7 +/- 0.3"
# were indistinguishable in every output format.
# ----------------------------------------------------------------------


def _two_state_chain(n, p01, p10, rng):
    """Two-state Markov chain; the indicator's IACT is 2/(p01+p10) - 1."""
    s = np.zeros(n, dtype=int)
    s[0] = rng.random() < p01 / (p01 + p10)
    for i in range(1, n):
        s[i] = (rng.random() >= p10) if s[i - 1] else (rng.random() < p01)
    return s.astype(int)


def test_transition_stats_counts_per_chain_and_round_trips():
    """
    Given label sequences with hand-countable mode changes -- one chain that
      never leaves mode 0, one that crosses once, one that goes 0 -> 1 -> 0,
    When transition_stats runs,
    Then the total, the per-chain counts and the round trips (k -> j -> k)
      are each exactly what the sequences show, and unassigned (-1) draws are
      skipped rather than counted as a visit to a third state.
    """
    labels = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1],
            [0, 0, 1, 1, 0, 0],
            [0, -1, -1, 0, 0, 0],
        ]
    )

    total, per_chain, round_trips = transition_stats(labels)

    assert per_chain.tolist() == [0, 1, 2, 0]
    assert total == 3
    assert round_trips == 1  # only chain 2 returns to where it started


def test_weight_ess_recovers_a_known_transition_rate():
    """
    Given a two-state chain whose mode-label transition probabilities are
      known, so the indicator's IACT is exactly 2/(p01+p10) - 1,
    When weight_ess measures it as a time series,
    Then the recovered IACT matches the analytic Markov value, and N_eff and
      sigma_w follow -- N_eff is thousands of draws below the raw draw count.
    """
    rng = np.random.default_rng(5)
    p01, p10 = 0.005, 0.015
    labels = np.array(
        [_two_state_chain(20000, p01, p10, rng) for _ in range(4)]
    )
    analytic_tau = 2.0 / (p01 + p10) - 1.0  # = 99

    n_eff, tau = weight_ess(labels, 1)

    assert tau == pytest.approx(analytic_tau, rel=0.25)
    assert n_eff == pytest.approx(labels.size / analytic_tau, rel=0.3)
    w = float((labels == 1).mean())
    sigma_w = np.sqrt(w * (1 - w) / n_eff)
    # the naive "sqrt(w(1-w)/N)" over all 80000 draws would be ~10x tighter
    assert sigma_w > 5 * np.sqrt(w * (1 - w) / labels.size)


def test_time_series_and_markov_iact_agree():
    """
    Given synthetic two-state chains spanning fast and slow mixing,
    When the time-series IACT estimate and the closed-form two-state Markov
      value 2/(p01+p10) - 1 are compared,
    Then they agree -- an independent cross-check that the time-series
      estimator is measuring what it claims to.
    """
    rng = np.random.default_rng(9)
    for p01, p10 in [(0.3, 0.3), (0.02, 0.02), (0.005, 0.015)]:
        labels = np.array(
            [_two_state_chain(20000, p01, p10, rng) for _ in range(4)]
        )

        _n_eff, tau = weight_ess(labels, 1)
        tau_markov = markov_indicator_iact(labels, 1)

        assert tau_markov == pytest.approx(2.0 / (p01 + p10) - 1.0, rel=0.3)
        assert tau == pytest.approx(tau_markov, rel=0.3)


def test_thinning_does_not_manufacture_independence():
    """
    Given one two-state chain set, and the SAME series thinned by 5, 10 and
      50 -- thinning makes consecutive stored draws further apart, so the
      thinned series genuinely looks less autocorrelated,
    When weight_ess measures each,
    Then N_eff (and therefore sigma_w) is unchanged: the IACT falls by the
      thinning factor exactly as the draw count does, so no thinning factor
      can be mistaken for extra information.
    """
    rng = np.random.default_rng(13)
    labels = np.array(
        [_two_state_chain(20000, 0.005, 0.005, rng) for _ in range(4)]
    )
    n_eff_full, tau_full = weight_ess(labels, 1)

    for thin in (5, 10, 50):
        n_eff, tau = weight_ess(labels[:, ::thin], 1)

        assert tau == pytest.approx(tau_full / thin, rel=0.5)
        assert n_eff == pytest.approx(n_eff_full, rel=0.25)


def test_chains_that_never_switch_get_no_spuriously_tight_weight():
    """
    Given chains that each sit in one mode for their entire length -- the
      case evidence weighting exists for -- so there is no information at all
      about the relative masses beyond "each chain landed somewhere",
    When identify_modes runs,
    Then the report says zero transitions and every chain stuck, N_eff is of
      order the number of chains rather than the number of draws, and the
      weight's 1-sigma is comparable to the weight itself.
    """
    rng = np.random.default_rng(3)
    chain_mode = np.repeat([0] * 4 + [1] * 4, N_DRAW)
    a = rng.normal(0, 1, N) + 10 * chain_mode
    idata = _make_idata({"a_raw": a}, rng.normal(0, 1, N))

    rep = identify_modes(idata)

    assert rep.n_modes == 2
    assert rep.n_transitions == 0
    assert rep.n_round_trips == 0
    assert rep.transitions_per_chain.tolist() == [0] * N_CHAIN
    assert rep.n_chains_no_switch == N_CHAIN
    assert rep.modes[0].weight_ess < 4 * N_CHAIN
    assert rep.modes[0].weight_err > 0.2
    assert not rep.weights_reliable
    text = rep.to_text()
    assert "chains that never changed mode: 8/8" in text
    assert "mode-weight precision" in " ".join(rep.notes)


def test_mixing_transitions_give_a_tight_weight():
    """
    Given two modes that every chain visits repeatedly (draw-by-draw mixing),
    When identify_modes runs,
    Then the transition count is large, the weights are validated, and the
      weight uncertainty is small -- the advisory low-N_eff note does not
      fire on a run that genuinely mixed.
    """
    rng = np.random.default_rng(21)
    labels = (rng.random(N) < 0.3).astype(int)
    a = rng.normal(0, 1, N) + 10 * labels
    idata = _make_idata({"a_raw": a}, rng.normal(0, 1, N))

    rep = identify_modes(idata)

    assert rep.n_modes == 2
    assert rep.n_transitions > 1000
    assert rep.n_round_trips > 500
    assert rep.n_chains_no_switch == 0
    assert rep.weights_reliable
    assert rep.modes[0].weight_err < 0.02
    assert not any("mode-weight precision" in n for n in rep.notes)


def test_transition_diagnostics_reach_latex_and_csv(tmp_path):
    """
    Given a report whose chains never switched mode,
    When the LaTeX table comments and the CSV header block are built,
    Then both carry the weight uncertainty and the transition/no-switch
      counts, so the numbers a reader needs to judge the weights travel with
      them into every output format.
    """
    rng = np.random.default_rng(3)
    chain_mode = np.repeat([0] * 6 + [1] * 2, N_DRAW)
    a = rng.normal(0, 1, N) + 10 * chain_mode
    rep = identify_modes(_make_idata({"a_raw": a}, rng.normal(0, 1, N)))

    var_file = tmp_path / "defs.tex"
    tmpl_file = tmp_path / "table.tex"
    csv_file = tmp_path / "results.csv"
    build_latex_output(
        _StubSystem(),
        var_filename=str(var_file),
        table_filename=str(tmpl_file),
        mode_report=rep,
    )
    build_csv_output(_StubSystem(), str(csv_file), mode_report=rep)

    tmpl = tmpl_file.read_text()
    assert r"\ezmodeweighterrone" in var_file.read_text()
    assert r"\ezmodeweighterrone" in tmpl
    assert "Mode changes in the stored draws: 0" in tmpl
    assert "2 chains never changed mode" not in tmpl  # it is 8 of 8
    assert "8 of 8 chains never changed mode" in tmpl
    csv_text = csv_file.read_text()
    assert "weight_err" in csv_text.splitlines()[0]
    assert "Mode changes in the stored draws: 0" in csv_text


def test_thin_factor_read_from_the_trace_and_reported():
    """
    Given a trace stamped by run.py with its storage thinning,
    When identify_modes runs,
    Then the report carries the factor and says out loud that the transition
      count is a lower bound; an unstamped trace says the thinning is unknown
      rather than quietly asserting it was 1.
    """
    rng = np.random.default_rng(31)
    labels = (rng.random(N) < 0.3).astype(int)
    a = rng.normal(0, 1, N) + 10 * labels

    unstamped = _make_idata({"a_raw": a}, rng.normal(0, 1, N))
    rep_unstamped = identify_modes(unstamped)

    stamped = _make_idata({"a_raw": a}, rng.normal(0, 1, N))
    stamped.posterior.attrs["nthin"] = 10
    rep_stamped = identify_modes(stamped)

    assert rep_unstamped.thin_factor == 1
    assert not rep_unstamped.thin_known
    assert "does not record its storage thinning" in rep_unstamped.to_text()
    assert rep_stamped.thin_factor == 10
    assert rep_stamped.thin_known
    assert "LOWER BOUND" in rep_stamped.to_text()


def test_ladder_round_trips_quoted_separately_from_mode_changes():
    """
    Given a PTDE trace stamped with the ladder's own temperature round trips,
    When the mode report renders,
    Then it quotes them as sampler context and says explicitly that they are
      temperature round trips and NOT mode changes -- "swap" is ambiguous
      between the two and conflating them would misstate the weights.
    """
    rng = np.random.default_rng(41)
    labels = (rng.random(N) < 0.3).astype(int)
    a = rng.normal(0, 1, N) + 10 * labels
    idata = _make_idata({"a_raw": a}, rng.normal(0, 1, N))
    idata.posterior.attrs["ptde_ladder_round_trips"] = 77
    idata.posterior.attrs["ptde_swap_rounds"] = 5000

    rep = identify_modes(idata)
    text = rep.to_text()

    assert rep.ladder_round_trips == 77
    assert rep.ladder_swap_rounds == 5000
    assert "temperature round trips" in text
    assert "NOT mode changes" in text
    assert "77" in text


# ----------------------------------------------------------------------
# Review 3.17: the validity gate must not be bypassed by an ALL-invalid
# trace.  identify_modes cannot return a report when every draw is
# rejected, so before the fix the failure arrived at build_mode_reports as
# a bare exception, was absorbed by the broad catch, and
# check_invalid_frac(None) returned immediately -- a 1.1%-invalid trace
# refused to emit tables while a 100%-invalid one emitted a clean-looking
# set.
# ----------------------------------------------------------------------


class _PipelineStubSystem(_StubSystem):
    """_StubSystem plus the one extra hook build_mode_reports calls."""

    def __init__(self):
        self.distributed = 0

    def distribute_posterior(self, idata):
        self.distributed += 1


def _all_lp_nan_idata(rng):
    """Ordinary-looking draws whose stored lp is entirely non-finite.

    The dangerous shape: the parameter values are finite and perfectly
    plausible, so every table built from them reads as a healthy fit.
    """
    return _make_idata({"a_raw": rng.normal(0, 1, N)}, np.full(N, np.nan))


def test_no_valid_draws_error_carries_the_counts():
    """
    Given a trace in which every draw fails the validity filter,
    When identify_modes runs,
    Then it raises NoValidDrawsError (a ValueError) carrying the invalid
      count, fraction, per-reason breakdown and per-chain counts -- the
      only channel through which the all-invalid case can reach the gate,
      since no ModeReport can exist to read them off.
    """
    rng = np.random.default_rng(1)
    idata = _all_lp_nan_idata(rng)

    with pytest.raises(NoValidDrawsError) as excinfo:
        identify_modes(idata)

    exc = excinfo.value
    assert isinstance(exc, ValueError)  # pre-existing callers still work
    assert exc.n_invalid == N
    assert exc.n_draws == N
    assert exc.invalid_frac == 1.0
    assert exc.reason_counts == {"nonfinite-lp": N}
    assert len(exc.per_chain_invalid) == N_CHAIN


def _all_invalid_status(n=1000, n_chain=4):
    return {
        "state": MODE_NO_VALID_DRAWS,
        "n_draws": n,
        "n_invalid": n,
        "invalid_frac": 1.0,
        "reasons": {"nonfinite-lp": n},
        "per_chain_invalid": [n // n_chain] * n_chain,
    }


def test_check_invalid_frac_raises_when_every_draw_is_invalid():
    """
    Given no mode report because EVERY draw was rejected as invalid,
    When check_invalid_frac runs,
    Then it raises, says all the draws were rejected, and tells the user
      what to do next.

    Regression for review 3.17: the absent report used to return early,
    so 100% invalid was the one fraction the gate let through.
    """
    with pytest.raises(RuntimeError) as excinfo:
        check_invalid_frac(
            None,
            max_invalid_frac=DEFAULT_MAX_INVALID_FRAC,
            trace_path="foo_trace.nc",
            modes_path="foo_modes.txt",
            status=_all_invalid_status(),
        )

    msg = str(excinfo.value)
    assert "ALL 1000 draws (100.00%)" in msg
    assert "nonfinite-lp" in msg
    assert "foo_trace.nc" in msg
    assert "exozippy-modes" in msg  # what to do next
    assert "modes: {force: true}" in msg


def test_check_invalid_frac_silent_when_report_absent_for_other_reasons():
    """
    Given no mode report for a reason that says nothing about the draws'
      numerical validity (the mode pass crashed, or no status was kept),
    When check_invalid_frac runs,
    Then it does not raise -- the gate fires on "the draws are unusable",
      never on "there was nothing to report".
    """
    check_invalid_frac(None, max_invalid_frac=DEFAULT_MAX_INVALID_FRAC)
    check_invalid_frac(None, max_invalid_frac=0.0, status={})
    check_invalid_frac(
        None, max_invalid_frac=0.0, status={"state": MODE_FAILED}
    )
    check_invalid_frac(None, max_invalid_frac=0.0, status={"state": MODE_OK})


def test_check_invalid_frac_all_invalid_honours_the_same_overrides():
    """
    Given no mode report because every draw was rejected,
    When check_invalid_frac runs with force=True, or with a
      max_invalid_frac of 1.0,
    Then it does not raise: the all-invalid case is the extreme end of the
      same continuum and obeys the same documented escape hatches, rather
      than becoming a second, stricter gate at 100%.
    """
    status = _all_invalid_status()
    check_invalid_frac(None, max_invalid_frac=0.01, force=True, status=status)
    check_invalid_frac(None, max_invalid_frac=1.0, status=status)


def test_mode_status_to_text_renders_only_the_all_invalid_state():
    """
    Given a mode-pass status dict,
    When mode_status_to_text renders it,
    Then the all-invalid state produces a report saying so, and every
      other state (including no status at all) renders nothing -- the
      innocent cases keep writing exactly the files they wrote before.
    """
    text = mode_status_to_text(_all_invalid_status())

    assert "NO VALID DRAWS" in text
    assert "1000 draws (100.00%)" in text
    assert "nonfinite-lp" in text
    assert mode_status_to_text(None) == ""
    assert mode_status_to_text({}) == ""
    assert mode_status_to_text({"state": MODE_FAILED}) == ""
    assert mode_status_to_text({"state": MODE_OK}) == ""


def test_pipeline_refuses_to_write_tables_when_every_draw_is_invalid(
    tmp_path,
):
    """
    Given a live fit whose draws were all rejected as numerically invalid,
    When build_mode_reports runs the reporting pipeline,
    Then it raises before writing the LaTeX/CSV tables, and the
      <prefix>_modes.txt it leaves behind says the draws were all rejected.

    Regression for review 3.17: this run used to complete, writing a full
    set of tables carrying finite values and error bars, with nothing on
    disk saying the trace had been rejected in its entirety.
    """
    rng = np.random.default_rng(2)
    prefix = tmp_path / "broken"
    status = {}

    with pytest.raises(RuntimeError, match="ALL 12000 draws"):
        build_mode_reports(
            _PipelineStubSystem(),
            _all_lp_nan_idata(rng),
            str(prefix),
            trace_path=str(prefix) + "_trace.nc",
            raise_on_invalid=True,
            mode_status=status,
        )

    assert status["state"] == MODE_NO_VALID_DRAWS
    assert status["invalid_frac"] == 1.0
    assert not (tmp_path / "broken_results.csv").exists()
    assert not (tmp_path / "broken_definitions.tex").exists()
    assert not (tmp_path / "broken_table.tex").exists()
    assert "NO VALID DRAWS" in (tmp_path / "broken_modes.txt").read_text()


def test_pipeline_reports_all_invalid_without_raising_for_forensics(
    tmp_path,
):
    """
    Given the same all-invalid trace and the forensic reprocessing path
      (raise_on_invalid=False, what exozippy-modes uses),
    When build_mode_reports runs,
    Then it completes as that tool's contract requires, but records the
      all-invalid state in the status dict and writes a <prefix>_modes.txt
      that says so -- the tables it goes on to write are never the only
      thing on disk describing this trace.
    """
    rng = np.random.default_rng(3)
    prefix = tmp_path / "forensic"
    status = {}

    report = build_mode_reports(
        _PipelineStubSystem(),
        _all_lp_nan_idata(rng),
        str(prefix),
        raise_on_invalid=False,
        mode_status=status,
    )

    assert report is None
    assert status["state"] == MODE_NO_VALID_DRAWS
    assert "NO VALID DRAWS" in (tmp_path / "forensic_modes.txt").read_text()
    # latex.py's own invalid-draw note reads off the mode report, which does
    # not exist here, so the pipeline supplies it: the table it does write
    # must not read as a clean result.
    template = (tmp_path / "forensic_table.tex").read_text()
    assert "rejected as numerically invalid" in template
    assert r"100.00\%" in template


def test_pipeline_unchanged_when_the_mode_pass_fails_for_another_reason(
    tmp_path, monkeypatch
):
    """
    Given a mode pass that fails for a reason unrelated to draw validity,
    When build_mode_reports runs with raise_on_invalid=True,
    Then it warns, returns None, and writes the combined-posterior tables
      exactly as before -- no raise, and no <prefix>_modes.txt invented
      for a state that carries no evidence about the draws.
    """
    import exozippy.outputs.report_pipeline as rp

    def _boom(*args, **kwargs):
        raise RuntimeError("clustering exploded")

    monkeypatch.setattr(rp, "identify_modes", _boom)

    rng = np.random.default_rng(4)
    idata = _make_idata({"a_raw": rng.normal(0, 1, N)}, rng.normal(0, 1, N))
    prefix = tmp_path / "innocent"
    status = {}

    report = build_mode_reports(
        _PipelineStubSystem(),
        idata,
        str(prefix),
        trace_path=str(prefix) + "_trace.nc",
        raise_on_invalid=True,
        mode_status=status,
    )

    assert report is None
    assert status["state"] == MODE_FAILED
    assert not (tmp_path / "innocent_modes.txt").exists()
    assert (tmp_path / "innocent_results.csv").exists()
    assert (tmp_path / "innocent_definitions.tex").exists()


def test_pipeline_status_records_a_healthy_mode_pass(tmp_path):
    """
    Given a healthy trace,
    When build_mode_reports runs,
    Then the status dict reports MODE_OK with zero invalid draws, so a
      caller reading the status can tell success from either failure mode.
    """
    rng = np.random.default_rng(5)
    idata = _make_idata({"a_raw": rng.normal(0, 1, N)}, rng.normal(0, 1, N))
    status = {}

    report = build_mode_reports(
        _PipelineStubSystem(),
        idata,
        str(tmp_path / "healthy"),
        raise_on_invalid=True,
        mode_status=status,
    )

    assert report is not None
    assert status["state"] == MODE_OK
    assert status["n_invalid"] == 0
