"""The blind 2L1S pipeline's decisions (examples/DC2018_069/blind.py).

Stages 2, 4 and 5 of the shipped example decide which grid basins become
modes, when two sampling runs found the same basin, how a run's sub-mode
evidences combine, and how normalized lnZ becomes a relative probability.
Those decisions are plain functions of numbers, so these tests drive them on
synthetic grids and synthetic per-run records: no sampler, no trace, no
challenge data.  They are also the acceptance spec for the blind-pipeline
driver.
"""

import importlib
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from exozippy.components.mulensing.gridsearch import GridResult
from exozippy.outputs.evidence import softmax_weights

EX = Path(__file__).resolve().parents[1] / "examples" / "DC2018_069"


@pytest.fixture(scope="module")
def B():
    sys.path.insert(0, str(EX))
    try:
        yield importlib.import_module("blind")
    finally:
        sys.path.remove(str(EX))


def _grid(basins):
    """A 21 x 8 grid (the shipped box) with Gaussian wells at `basins`,
    each (log_s, log_q, depth)."""
    ls = np.linspace(-0.7, 0.7, 21)
    lq = np.linspace(-6.0, -1.0, 8)
    S, Q = np.meshgrid(ls, lq, indexing="ij")
    chi2 = np.full(S.shape, 40000.0)
    for s0, q0, depth in basins:
        chi2 -= depth * np.exp(
            -(((S - s0) / 0.05) ** 2 + ((Q - q0) / 0.5) ** 2)
        )
    full = np.ones(S.shape)
    return GridResult(
        log_s=ls, log_q=lq, chi2=chi2, alpha=290.0 * full,
        t_0=2460035.49 * full, u_0=0.0024 * full, t_E=7.1 * full,
        rho=np.nan * full, n_done=S.size, n_tasks=S.size, complete=True,
    )  # fmt: skip


def test_candidates_include_the_mirror_and_are_deduped(B):
    """
    Given a grid whose best basin is at s < 1, a shallower one at its exact
    mirror, and a far, much worse basin,
    When candidate_modes picks the modes,
    Then the close/wide pair are both modes (the mirror is not dropped, nor
    listed twice as 'minimum' AND 'mirror'), the far basin is cut by dchi2,
    and the modes are named m0, m1, ... best first.
    """
    g = _grid([(-0.07, -3.86, 600.0), (0.07, -3.86, 500.0),
               (0.56, -2.43, 20.0)])  # fmt: skip
    cands = B.candidate_modes(g, dchi2_max=200.0)
    assert [c["mode"] for c in cands] == ["m0", "m1"]
    assert [round(c["log_s"], 2) for c in cands] == [-0.07, 0.07]
    keys = {(round(c["log_s"], 2), round(c["log_q"], 2)) for c in cands}
    assert len(keys) == len(cands)


def test_mirror_is_added_even_when_not_a_local_minimum(B):
    """
    Given a single basin at s < 1 whose mirror column has no well,
    When candidate_modes runs with a loose dchi2 cut,
    Then the mirror column's best node is still a candidate, labelled as
    a mirror -- the grid's preferred side is not trusted to be the truth.
    """
    g = _grid([(-0.21, -3.86, 300.0)])
    cands = B.candidate_modes(g, dchi2_max=1000.0, max_modes=5)
    origins = [c["origin"] for c in cands]
    assert any(o.startswith("mirror of") for o in origins)
    mir = next(c for c in cands if c["origin"].startswith("mirror of"))
    assert mir["log_s"] == pytest.approx(0.21, abs=0.036)


def _run(B, mode, mean, lnz, re2, sign=1, err=0.3, sd=(0.002, 0.02, 0.2)):
    cov = np.diag(np.square(sd)).tolist()
    return {
        "mode": mode, "arch": "2L1S", "lnZ": lnz, "lnZ_err": err,
        "re2": re2, "bridge_status": "ok", "stuck_frac": 0.0,
        "lp_rhat": 1.0, "drift": "", "origin": "minimum 0",
        "summary": {"s": [10 ** mean[0], 0, 0], "q": [10 ** mean[1], 0, 0]},
        "features": {"mean": list(mean), "cov": cov, "u0_sign": sign,
                     "u0_sign_purity": 1.0},
    }  # fmt: skip


def test_dedupe_keeps_lower_re2_and_never_adds_lnz(B):
    """
    Given two runs that converged to one basin (1 sigma apart), a third in
    the mirror basin and a fourth at the first's (log s, log q, alpha) but
    with u_0 of the other sign,
    When the report is built,
    Then the first pair collapses to the lower-re2 run, whose lnZ alone
    enters the softmax, and the u_0-sign twin stays a separate mode.
    """
    a = _run(B, "m0", (-0.062, -3.73, -71.8), 100.0, 0.05)
    a_dup = _run(B, "m2", (-0.060, -3.72, -71.6), 100.4, 0.01)
    b = _run(B, "m1", (0.056, -3.75, -71.9), 99.0, 0.02)
    c = _run(B, "m3", (-0.062, -3.73, -71.8), 98.0, 0.02, sign=-1)
    rep = B.build_report([a, a_dup, b, c])
    kept = [m["mode"] for m in rep["modes"]]
    assert "m0" not in kept and "m2" in kept
    assert set(kept) == {"m1", "m2", "m3"}
    assert rep["duplicates"][0]["mode"] == "m0"
    assert rep["duplicates"][0]["duplicate_of"] == "m2"
    w = {m["mode"]: m["weight_2L1S"] for m in rep["modes"]}
    ref, _ = softmax_weights([100.4, 99.0, 98.0], [0.3, 0.3, 0.3])
    assert [w["m2"], w["m1"], w["m3"]] == pytest.approx(list(ref))
    assert sum(w.values()) == pytest.approx(1.0)


def test_alpha_wraps_in_the_basin_distance(B):
    """
    Given two runs at alpha = 179.9 and -179.9 deg, otherwise identical,
    When they are compared,
    Then they are one basin: alpha is circular.
    """
    a = _run(B, "m0", (0.05, -3.7, 179.9), 1.0, 0.1)
    b = _run(B, "m1", (0.05, -3.7, -179.9), 1.0, 0.1)
    assert B.mahalanobis(a["features"], b["features"]) < 3.0


def test_refused_bridge_gets_no_weight_and_is_flagged(B):
    """
    Given one run with a refused bridge (lnZ nan) and one accepted,
    When the report is built,
    Then the refused run has weight 0, is listed, and carries its flag.
    """
    a = _run(B, "m0", (-0.06, -3.7, -72.0), 50.0, 0.05)
    b = _run(B, "m1", (0.06, -3.7, -72.0), float("nan"), float("inf"))
    b["bridge_status"] = "refused"
    b["lnZ_err"] = float("inf")
    rep = B.build_report([a, b])
    w = {m["mode"]: m for m in rep["modes"]}
    assert w["m0"]["weight_2L1S"] == pytest.approx(1.0)
    assert w["m1"]["weight_2L1S"] == 0.0
    assert any("bridge refused" in f for f in w["m1"]["flags"])


def test_submodes_combine_by_logsumexp(B):
    """
    Given a run that identify_modes split into two accepted sub-modes and
    one refused one,
    When the run's lnZ is formed,
    Then it is the logsumexp of the accepted two and the status is
    'partial' (the refused sub-mode's mass is missing, and says so).
    """
    sub = [
        {"lnZ": 10.0, "lnZ_err": 0.2, "re2": 0.01, "refused": False},
        {"lnZ": 9.0, "lnZ_err": 0.4, "re2": 0.03, "refused": False},
        {"lnZ": float("nan"), "lnZ_err": float("inf"), "re2": float("inf"),
         "refused": True},
    ]  # fmt: skip
    t = B.lnz_from_submodes(sub)
    assert t["lnZ"] == pytest.approx(10.0 + math.log(1 + math.exp(-1)))
    assert t["status"] == "partial"
    assert t["re2"] == pytest.approx(0.03)
    assert B.lnz_from_submodes(sub[2:])["status"] == "refused"


def test_drift_and_stuck_flags(B):
    """
    Given a seed on the close side and a run whose median crossed s = 1,
    and a run with 20% of chains stuck 50 nats low,
    Then both are flagged.
    """
    seed = {"log_s": -0.07, "log_q": -3.86, "u_0": 0.0024}
    why = B.drift(seed, 0.05, -3.8, 1, 0.07, 0.714)
    assert "crossed s=1" in why
    assert B.drift(seed, -0.06, -3.8, 1, 0.07, 0.714) == ""
    lp = np.zeros((10, 100))
    lp[:2] -= 50.0
    assert B.stuck_fraction(lp) == pytest.approx(0.2)
    flags = B.reliability_flags({"stuck_frac": 0.2, "drift": why})
    assert any(f.startswith("stuck") for f in flags)
    assert any(f.startswith("drift") for f in flags)


def test_report_baseline_and_truth_section(B):
    """
    Given two modes and a no-companion baseline far below them,
    When the text report is rendered without and with the answer key,
    Then the baseline is in the 'all architectures' softmax with ~0 weight,
    and the truth appears ONLY in the labelled validation section, which
    is the last part of the report and absent without --truth.
    """
    a = _run(B, "m0", (-0.062, -3.73, -71.8), 100.0, 0.05)
    b = _run(B, "m1", (0.056, -3.75, -71.9), 99.0, 0.02)
    base = {"mode": "1L1S_fs", "arch": "1L1S_fs", "lnZ": -3000.0,
            "lnZ_err": 0.5, "re2": 0.01, "bridge_status": "ok",
            "drift": "", "summary": {}}  # fmt: skip
    rep = B.build_report([a, b], base)
    assert rep["baseline"]["weight_all"] < 1e-100
    assert rep["baseline"]["dlnZ_best_2L1S_minus_baseline"] == 3100.0
    assert sum(m["weight_all"] for m in rep["modes"]) == pytest.approx(1.0)
    blind = B.report_text(rep)
    assert "VALIDATION ONLY" not in blind
    txt = B.report_text(rep, truth={"s": 1.1384, "q": 10**-3.76})
    head, tail = txt.split("VALIDATION ONLY")
    assert "1.1384" not in head and "s=1.1384" in tail
    assert "mirror side" in tail and "truth side" in tail


def test_all_refused_gives_no_weights_only_labelled_indicative(B):
    """
    Given every bridge refused (re2 above the limit) -- what 069's runs
    produced -- with two runs in one basin,
    When the report is built,
    Then no mode gets an evidence weight, the status says so, the refused
    estimates appear only as INDICATIVE weights, and the duplicate is
    resolved by the sub-mode re2 (the lower one is kept).
    """

    def refused(mode, mean, lnz_ind, re2):
        r = _run(B, mode, mean, float("nan"), float("inf"), err=float("inf"))
        r["bridge_status"] = "refused"
        r["lnZ_indicative"] = lnz_ind
        r["submodes"] = [{"re2": re2, "state": "re2-too-large"}]
        return r

    a = refused("m0", (-0.07, -3.71, -71.9), 10.0, 0.63)
    a2 = refused("m2", (-0.0705, -3.712, -71.92), 9.0, 0.37)
    b = refused("m1", (0.067, -3.73, -71.9), 10.0, 0.37)
    rep = B.build_report([a, a2, b])
    assert rep["weights_status"].startswith("NO mode accepted")
    assert all(m["weight_2L1S"] == 0.0 for m in rep["modes"])
    assert [d["mode"] for d in rep["duplicates"]] == ["m0"]
    iw = {m["mode"]: m["weight_2L1S_indicative"] for m in rep["modes"]}
    assert iw["m1"] == pytest.approx(1 / (1 + math.exp(-1.0)))
    txt = B.report_text(rep)
    assert "INDICATIVE" in txt and "NOT a weight" in txt
