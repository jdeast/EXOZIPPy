"""Seeded-solution ledger (outputs/ledger.py): Laplace records for every
polished seed, matched to surviving posterior modes or reported as
"considered and rejected"."""

import numpy as np
import pymc as pm
import pytest

from exozippy.components.parameter import Parameter
from exozippy.outputs.ledger import (
    append_ledger_csv,
    build_seed_ledger,
    ledger_to_text,
    match_ledger_to_modes,
    rejected_records,
    write_rejected_latex,
)
from exozippy.outputs.modes import ModeInfo, ModeReport


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
    assert ledger[0].match_distance < 1.0
    assert ledger[1].matched_mode is None
    assert ledger[1].match_distance > 10.0
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
    csv_path.write_text("name,mode,weight,value,hi,lo\n")
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
