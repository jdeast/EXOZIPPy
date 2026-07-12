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

import numpy as np
import arviz as az
import pytest

from exozippy.outputs.modes import identify_modes, mode_suffix
from exozippy.components.parameter import Parameter


N_CHAIN, N_DRAW = 8, 1500
N = N_CHAIN * N_DRAW


def _make_idata(posterior, lp):
    """Build a minimal InferenceData from flat (N,) arrays."""
    return az.from_dict({
        "posterior": {k: np.asarray(v).reshape(N_CHAIN, N_DRAW)
                      for k, v in posterior.items()},
        "sample_stats": {"lp": np.asarray(lp).reshape(N_CHAIN, N_DRAW)},
    })


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
    idata = _make_idata({"a_raw": t, "b_raw": t ** 2 + rng.normal(0, 0.3, N)},
                        rng.normal(0, 1, N))

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
    idata = _make_idata({"x": rng.normal(0, 1, N) + 12 * labels},
                        rng.normal(0, 1, N))

    rep = identify_modes(idata)

    assert rep.n_modes == 2
    assert any("no *_raw" in n for n in rep.notes)


# ----------------------------------------------------------------------
# Parameter per-mode summaries
# ----------------------------------------------------------------------

def _sampled_param(posterior):
    p = Parameter(label="star.teff", latex=r"T_{\rm eff}",
                  description="Effective temperature", initval=5000.0,
                  lower=3000.0, upper=7000.0, init_scale=100.0)
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
    empty = summaries[1] if not isinstance(summaries[1], list) else summaries[1][0]
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

    assert row_multi.count("&") == 4   # param & desc & val1 & val2 & prior
    assert f"\\{p.latex_varname}modeone" in row_multi
    assert f"\\{p.latex_varname}modetwo" in row_multi
    assert row_single.count("&") == 3  # param & desc & val & prior
    assert f"\\{p.latex_varname}modeone" not in row_single
