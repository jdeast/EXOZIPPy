"""Seeded-solution ledger (outputs/ledger.py): Laplace records for every
polished seed, matched to surviving posterior modes or reported as
"considered and rejected"."""

import csv

import numpy as np
import pymc as pm
import pytest

from exozippy.components.parameter import Parameter
from exozippy.outputs.latex import build_csv_output
from exozippy.outputs.ledger import (
    SeedRecord,
    _delta_lp_cell,
    append_ledger_csv,
    build_seed_ledger,
    ledger_to_text,
    match_ledger_to_modes,
    rejected_records,
    write_rejected_latex,
)
from exozippy.outputs.modes import ModeInfo, ModeReport

# The mode-keyed results.csv layout, spelled out rather than imported so
# these tests pin the contract instead of whatever the code defines.
MODE_COLUMNS = (
    "parname",
    "mode",
    "weight",
    "weight_err",
    "value",
    "up_err",
    "low_err",
)


def _two_basin_model():
    """One bounded parameter with two Gaussian basins: x=2 (deep) and x=7
    (shallow, delta lp = 6)."""
    p = Parameter(label="toy.x", initval=2.0, lower=0.0, upper=10.0)
    with pm.Model() as model:
        xv = p.build_pymc()
        lp1 = -0.5 * ((xv - 2.0) / 0.05) ** 2
        lp2 = -0.5 * ((xv - 7.0) / 0.05) ** 2 - 6.0
        pm.Potential("like", pm.math.logsumexp(pm.math.stack([lp1, lp2])))
    return model, p


class _StubSystem:
    def __init__(self, params):
        self._params = params

    def get_all_parameters(self):
        return self._params


def _fake_report(centers, feature_name):
    """Minimal ModeReport with the given raw-space centers."""
    modes = [
        ModeInfo(
            index=k,
            weight=1.0 / len(centers),
            n_draws=100,
            lp_med=0.0,
            lp_max=0.0,
            delta_lp_max=0.0,
            per_chain_weight=np.ones(2),
            center={feature_name: float(c)},
        )
        for k, c in enumerate(centers)
    ]
    return ModeReport(
        labels=np.zeros((2, 100), dtype=int),
        modes=modes,
        n_valid=200,
        n_invalid=0,
        n_unassigned=0,
        provenance="occupancy",
        weights_reliable=True,
        n_transitions=10,
        feature_vars=[feature_name],
    )


def test_ledger_measures_peak_and_width_per_seed():
    """
    Given two seeds at the two basin optima of a known double-Gaussian,
    When build_seed_ledger measures them,
    Then each record carries the basin's peak lp (delta_lp = 6 between
      them) and a physical width ~ the basin sigma (0.05).
    """
    # ARRANGE
    model, p = _two_basin_model()
    seeds = [
        {"toy.x_raw": np.asarray(p.raw_from_initval(np.array([2.0])))},
        {"toy.x_raw": np.asarray(p.raw_from_initval(np.array([7.0])))},
    ]
    stub = _StubSystem([p])

    # ACT
    ledger = build_seed_ledger(stub, model, seeds, [0, 1])

    # ASSERT: the basin depth difference is 6 nats in the likelihood, minus
    # the logit-Jacobian ratio between the two locations -- logp is a
    # density in the sampled (raw) space, so the flat-in-x prior carries
    # log[q(1-q)] with it: log(0.7*0.3) - log(0.2*0.8) = +0.272 at x=7.
    expected = 6.0 - (np.log(0.7 * 0.3) - np.log(0.2 * 0.8))
    assert ledger[0].delta_lp == pytest.approx(0.0, abs=0.01)
    assert ledger[1].delta_lp == pytest.approx(expected, abs=0.05)
    for rec in ledger:
        assert rec.phys_sigma["toy.x"][0] == pytest.approx(0.05, rel=0.25)
    assert ledger[0].phys["toy.x"][0] == pytest.approx(2.0, abs=0.01)
    assert ledger[1].phys["toy.x"][0] == pytest.approx(7.0, abs=0.01)


def test_matching_assigns_survivor_and_rejects_the_missing_mode():
    """
    Given a mode report containing only the deep basin,
    When the ledger is matched against it,
    Then the seed at that basin matches mode 0 and the other seed is
      rejected (matched_mode None), with the distance recorded.
    """
    # ARRANGE
    model, p = _two_basin_model()
    raw0 = np.asarray(p.raw_from_initval(np.array([2.0])))
    raw1 = np.asarray(p.raw_from_initval(np.array([7.0])))
    seeds = [{"toy.x_raw": raw0}, {"toy.x_raw": raw1}]
    ledger = build_seed_ledger(_StubSystem([p]), model, seeds, [0, 1])
    report = _fake_report([float(raw0[0])], "toy.x_raw")

    # ACT
    match_ledger_to_modes(ledger, report)

    # ASSERT
    assert ledger[0].matched_mode == 0
    assert ledger[0].match_distance < 1.0  # fraction of threshold
    assert ledger[1].matched_mode is None
    assert ledger[1].match_distance > 1.0  # beyond every criterion
    assert rejected_records(ledger) == [ledger[1]]


def test_text_csv_and_latex_report_the_rejected_solution(tmp_path):
    """
    Given a ledger with one rejected seed,
    When the text/CSV/LaTeX emitters run,
    Then the rejected solution appears with its delta lp and Laplace
      value +/- sigma, labeled as rejected.
    """
    # ARRANGE
    model, p = _two_basin_model()
    raw0 = np.asarray(p.raw_from_initval(np.array([2.0])))
    raw1 = np.asarray(p.raw_from_initval(np.array([7.0])))
    ledger = build_seed_ledger(
        _StubSystem([p]),
        model,
        [{"toy.x_raw": raw0}, {"toy.x_raw": raw1}],
        [0, 1],
    )
    match_ledger_to_modes(ledger, _fake_report([float(raw0[0])], "toy.x_raw"))

    # ACT
    text = ledger_to_text(ledger)
    csv_path = tmp_path / "results.csv"
    csv_path.write_text("# " + ", ".join(MODE_COLUMNS) + "\n")
    append_ledger_csv(ledger, str(csv_path))
    tex_path = tmp_path / "rejected.tex"
    wrote = write_rejected_latex(ledger, str(tex_path))

    # ASSERT
    assert "seed 0 (seed): survived as mode 1" in text
    assert "seed 1 (seed): REJECTED" in text
    assert "delta vs best seed = 5.7" in text
    csv = csv_path.read_text()
    assert "toy.x,rejected-seed1," in csv
    assert wrote and tex_path.exists()
    tex = tex_path.read_text()
    assert "considered and rejected" in tex.lower()
    assert "seed 1" in tex


def test_no_rejected_seeds_writes_nothing(tmp_path):
    """
    Given every seed matches a surviving mode,
    When the CSV/LaTeX emitters run,
    Then nothing is appended or written (the main tables already cover
      every mode).
    """
    # ARRANGE
    model, p = _two_basin_model()
    raw0 = np.asarray(p.raw_from_initval(np.array([2.0])))
    ledger = build_seed_ledger(
        _StubSystem([p]), model, [{"toy.x_raw": raw0}], [0]
    )
    match_ledger_to_modes(ledger, _fake_report([float(raw0[0])], "toy.x_raw"))

    # ACT
    csv_path = tmp_path / "results.csv"
    csv_path.write_text("header\n")
    append_ledger_csv(ledger, str(csv_path))
    wrote = write_rejected_latex(ledger, str(tmp_path / "rejected.tex"))

    # ASSERT
    assert csv_path.read_text() == "header\n"
    assert not wrote


def test_matching_uses_the_modes_marginal_scale():
    """
    Given a mode whose center (a posterior MEDIAN) sits ~20 of the seed's
      conditional widths from the basin peak but within the mode's own
      marginal spread,
    When the ledger is matched,
    Then the seed still matches (no false rejection) -- while a seed
      genuinely far away in BOTH scales stays rejected.

    Regression: on ob140939 all four seeds were falsely rejected because
    correlated posteriors put marginal medians tens of conditional sigmas
    from the polished peaks.
    """
    # ARRANGE
    model, p = _two_basin_model()
    raw0 = np.asarray(p.raw_from_initval(np.array([2.0])), dtype=float)
    raw7 = np.asarray(p.raw_from_initval(np.array([7.0])), dtype=float)
    ledger = build_seed_ledger(
        _StubSystem([p]),
        model,
        [{"toy.x_raw": raw0}, {"toy.x_raw": raw7}],
        [0, 1],
    )
    sep = abs(float(raw7[0]) - float(raw0[0]))
    # mode center displaced sep/6 from the peak (tens of the seed's tiny
    # conditional widths) with a marginal scale sep/12 -- the median sits
    # 2 marginal sigmas from the peak, and the other basin 10 away
    center = (
        float(raw0[0]) + np.sign(float(raw7[0]) - float(raw0[0])) * sep / 6.0
    )
    report = _fake_report([center], "toy.x_raw")
    report.modes[0].center_scale = {"toy.x_raw": sep / 12.0}

    # ACT
    match_ledger_to_modes(ledger, report)

    # ASSERT
    assert ledger[0].matched_mode == 0  # matched despite 20 seed-widths
    assert ledger[1].matched_mode is None  # other basin: still rejected


# ----------------------------------------------------------------------
# The results CSV stays rectangular when the ledger writes into it
#
# Review item 2.8.1: append_ledger_csv always writes mode-keyed rows, but
# build_csv_output only emitted the mode columns for a MULTIMODAL report.
# A multi-seed fit whose surviving posterior is unimodal (the ob140939
# setup) therefore got a 4-column header over a mix of 4- and 7-column
# rows -- unparseable by pandas, csv.DictReader or any spreadsheet.
# ----------------------------------------------------------------------


class _FakeComp:
    label = "toy"


def _fake_system(comp):
    class _Sys:
        name = "test"

        def get_all_components(self):
            return [comp]

    return _Sys()


def _one_rejected_ledger():
    """A two-seed ledger against a unimodal report: seed 1 is rejected."""
    model, p = _two_basin_model()
    raw0 = np.asarray(p.raw_from_initval(np.array([2.0])))
    raw1 = np.asarray(p.raw_from_initval(np.array([7.0])))
    ledger = build_seed_ledger(
        _StubSystem([p]),
        model,
        [{"toy.x_raw": raw0}, {"toy.x_raw": raw1}],
        [0, 1],
    )
    match_ledger_to_modes(ledger, _fake_report([float(raw0[0])], "toy.x_raw"))
    return ledger


def _data_rows(path):
    """Every non-comment row of a CSV, parsed (not eyeballed)."""
    with open(path, newline="") as f:
        return [
            row
            for row in csv.reader(f)
            if row and not row[0].lstrip().startswith("#")
        ]


def test_unimodal_posterior_plus_ledger_writes_a_rectangular_csv(tmp_path):
    """
    Given a UNIMODAL surviving posterior and a seed ledger with one
      rejected seed (a multi-seed fit whose rejected seeds are the only
      mode-keyed rows),
    When build_csv_output writes the results CSV and append_ledger_csv
      adds the rejected-seed rows,
    Then every row in the file has the same width, the header comment names
      exactly those columns, and csv.DictReader reads the rejected row back
      with its mode key and Laplace weight.
    """
    # ARRANGE
    ledger = _one_rejected_ledger()
    comp = _FakeComp()
    comp.x = Parameter(
        label="toy.x",
        latex="x",
        description="toy parameter",
        initval=2.0,
        lower=0.0,
        upper=10.0,
    )
    csv_path = tmp_path / "results.csv"

    # ACT
    build_csv_output(_fake_system(comp), str(csv_path), mode_columns=True)
    append_ledger_csv(ledger, str(csv_path))

    # ASSERT
    rows = _data_rows(csv_path)
    assert {len(r) for r in rows} == {len(MODE_COLUMNS)}

    header = csv_path.read_text().splitlines()[0]
    assert header.startswith("# ")
    assert [c.strip() for c in header[2:].split(",")] == list(MODE_COLUMNS)

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(
            (ln for ln in f if not ln.startswith("#")),
            fieldnames=MODE_COLUMNS,
        )
        parsed = list(reader)
    assert all(None not in d and "" not in d.keys() for d in parsed)
    rejected = [d for d in parsed if d["mode"] == "rejected-seed1"]
    assert len(rejected) == 1
    assert float(rejected[0]["value"]) == pytest.approx(7.0, abs=0.01)
    assert float(rejected[0]["weight"]) < 1.0
    assert [d for d in parsed if d["mode"] == "all"]  # posterior rows kept


def test_ledger_refuses_to_append_to_a_plain_layout_csv(tmp_path):
    """
    Given a results CSV written WITHOUT the mode columns,
    When append_ledger_csv tries to add its mode-keyed rows,
    Then it raises rather than making the file ragged, and names the fix.
    """
    # ARRANGE
    ledger = _one_rejected_ledger()
    comp = _FakeComp()
    comp.x = Parameter(
        label="toy.x",
        latex="x",
        description="toy parameter",
        initval=2.0,
        lower=0.0,
        upper=10.0,
    )
    csv_path = tmp_path / "results.csv"
    build_csv_output(_fake_system(comp), str(csv_path))
    before = csv_path.read_text()

    # ACT / ASSERT
    with pytest.raises(ValueError, match="mode_columns=True"):
        append_ledger_csv(ledger, str(csv_path))
    assert csv_path.read_text() == before  # untouched, still rectangular


# ----------------------------------------------------------------------
# the rejected-modes table's delta is a log-POSTERIOR (review 3.11.1)
# ----------------------------------------------------------------------


def test_rejected_table_does_not_call_the_posterior_a_likelihood(tmp_path):
    """
    Given a ledger with one rejected seed,
    When the rejected-solutions LaTeX table is written,
    Then its delta column is labelled as a log-POSTERIOR difference and the
      caption says so.  lp comes from model.compile_logp(), so it carries the
      priors, the component potentials and the reparameterization Jacobians
      -- this very file's own test measures the +0.272 logit-Jacobian term
      inside delta_lp -- and labelling it a Delta ln L propagated a
      likelihood-ratio claim into papers.  ledger_to_text has always said
      "lp at optimum".
    """
    # ARRANGE
    model, p = _two_basin_model()
    raw0 = np.asarray(p.raw_from_initval(np.array([2.0])))
    raw1 = np.asarray(p.raw_from_initval(np.array([7.0])))
    ledger = build_seed_ledger(
        _StubSystem([p]),
        model,
        [{"toy.x_raw": raw0}, {"toy.x_raw": raw1}],
        [0, 1],
    )
    match_ledger_to_modes(ledger, _fake_report([float(raw0[0])], "toy.x_raw"))
    tex_path = tmp_path / "rejected.tex"

    # ACT
    write_rejected_latex(ledger, str(tex_path))

    # ASSERT
    tex = tex_path.read_text()
    assert r"\Delta \ln \mathcal{P}" in tex
    assert r"\mathcal{L}" not in tex
    assert "log-POSTERIOR" in tex


def test_a_zero_delta_prints_unsigned():
    """
    Given a seed whose delta against the best is zero (the best seed itself,
      or any seed within rounding of it),
    When its table cell is formatted,
    Then it prints an unsigned zero, not "-0.0" -- a signed zero reads as a
      real, if tiny, difference from the best solution.
    """
    # ARRANGE / ACT / ASSERT
    assert _delta_lp_cell(0.0) == "$0.0$"
    assert _delta_lp_cell(0.01) == "$0.0$"
    assert _delta_lp_cell(5.72) == "$-5.7$"


def _hand_record(k, laplace_logw, lp_max=None):
    """One SeedRecord with a chosen Laplace log-weight.

    Hand-built so the weight-reporting tests can set the weight directly
    instead of engineering a model whose curvature happens to produce it.
    """
    return SeedRecord(
        seed_index=k,
        lp_max=100.0 if lp_max is None else lp_max,
        delta_lp=0.0,
        laplace_logw=laplace_logw,
        raw_point={"toy.x_raw": np.array([float(k)])},
        raw_scales={"toy.x_raw": np.array([1.0])},
        phys={"toy.x": np.array([float(k)])},
        phys_sigma={"toy.x": np.array([0.1])},
        sampled_idx={"toy.x": [0]},
    )


def test_the_ledger_prints_the_computed_laplace_gap_not_boilerplate():
    """
    Given two seeds whose Laplace log-weights differ by 106 nats,
    When the ledger text is written,
    Then each entry reports its OWN computed gap against the best -- 0.00
      for the best seed, -106.00 for the other -- and the old boilerplate
      claim that the weights are "comparable at the ~1-nat level" appears
      nowhere.  That sentence was printed verbatim on every entry, so the
      ob140939 ledger asserted comparability on a delta lp = 106
      rejection.
    """
    # ARRANGE
    ledger = [_hand_record(0, -50.0), _hand_record(1, -156.0)]

    # ACT
    text = ledger_to_text(ledger)

    # ASSERT
    assert "Laplace log-weight vs best = 0.00" in text
    assert "Laplace log-weight vs best = -106.00" in text
    assert "comparable" not in text
    assert "1-nat" not in text


def test_a_non_finite_laplace_weight_is_omitted_not_printed_as_nan(tmp_path):
    """
    Given a ledger in which no seed has a finite Laplace log-weight (an
      invalid start the polish never rescued: lp is -inf, so the weight is
      too),
    When the text and CSV emitters run,
    Then the gap is left out of the text entirely and the CSV weight cell
      is blank -- not "nan".  The reference used to be a bare max() over
      the ledger, which propagated one unusable entry into EVERY other
      seed's reported weight.
    """
    # ARRANGE
    ledger = [
        _hand_record(0, float("-inf"), lp_max=float("-inf")),
        _hand_record(1, float("-inf"), lp_max=float("-inf")),
    ]
    csv_path = tmp_path / "results.csv"
    csv_path.write_text("# " + ", ".join(MODE_COLUMNS) + "\n")

    # ACT
    text = ledger_to_text(ledger)
    append_ledger_csv(ledger, str(csv_path))

    # ASSERT
    assert "Laplace log-weight" not in text
    assert "nan" not in text.lower()
    rows = [
        r
        for r in csv.reader(csv_path.read_text().splitlines())
        if r and not r[0].startswith("#")
    ]
    assert rows, "the rejected seeds should still be reported"
    weight_col = MODE_COLUMNS.index("weight")
    assert all(r[weight_col] == "" for r in rows)
