"""The RMDC26 evidence report's decisions (examples/RMDC26/rmdc26.py).

How per-mode lnZ becomes a submitted relative_probability, which solutions
are flagged UNRELIABLE or inactive, and what the microlens-submit files
contain are plain functions of numbers and dicts, so these tests drive them
on synthetic records: no sampler, no trace, no challenge data.
"""

import importlib
import math
import sys
from pathlib import Path

import numpy as np
import pytest

EX = Path(__file__).resolve().parents[1] / "examples" / "RMDC26"

#: Every field of microlens_submit 0.17.9's Solution model
#: (models/solution.py) a solution file may carry; `saved` is excluded from
#: its JSON and `hardware_info`/`physical_parameter_uncertainties` are
#: optional ones this writer leaves out.
SOLUTION_FIELDS = {
    "solution_id", "model_type", "bands", "higher_order_effects", "t_ref",
    "parameters", "is_active", "alias", "compute_info", "notes_path",
    "used_astrometry", "used_postage_stamps", "limb_darkening_model",
    "limb_darkening_coeffs", "parameter_uncertainties", "uncertainty_method",
    "confidence_level", "physical_parameters", "log_likelihood",
    "relative_probability", "n_data_points", "hardware_info",
    "posterior_path", "lightcurve_plot_path", "lens_plane_plot_path",
    "physical_parameter_uncertainties", "creation_timestamp",
}  # fmt: skip


@pytest.fixture(scope="module")
def R():
    sys.path.insert(0, str(EX))
    try:
        yield importlib.import_module("rmdc26")
    finally:
        sys.path.remove(str(EX))


# --- probabilities ----------------------------------------------------------


def test_modes_within_the_floor_share_their_group_total_equally(R):
    """Given two modes 1 nat apart (inside the +/-2 nat floor) and one 20
    nats below, when the robust probabilities are formed, then the pair
    splits its raw total equally, the far mode keeps its raw value, and
    the sum is still 1."""
    lnz = [100.0, 99.0, 80.0]
    out = R.robust_probabilities(lnz, [0.1, 0.1, 0.1])
    p_raw, p = out["p_raw"], out["p_robust"]
    assert out["groups"] == [[0, 1], [2]]
    pair = p_raw[0] + p_raw[1]
    assert p[0] == pytest.approx(pair / 2) and p[1] == pytest.approx(pair / 2)
    assert p_raw[0] / p_raw[1] == pytest.approx(math.e)
    assert p[2] == pytest.approx(p_raw[2])
    assert p.sum() == pytest.approx(1.0)


def test_distinguishable_modes_keep_the_plain_softmax(R):
    """Given modes 10 nats apart, then nothing is grouped and the robust
    probabilities ARE the softmax of lnZ."""
    out = R.robust_probabilities([10.0, 0.0], [0.2, 0.2])
    assert out["groups"] == [[0], [1]]
    np.testing.assert_allclose(
        out["p_robust"], [1 / (1 + math.exp(-10)), 1 / (1 + math.exp(10))]
    )


def test_the_floor_is_added_in_quadrature_to_the_bridge_error(R):
    """Given a bridge error of 1.5 nats, then the total is sqrt(1.5^2+2^2)
    = 2.5, and two such modes 3.5 nats apart (< 2.5*sqrt 2 = 3.54) are
    one group while 3.6 apart are not."""
    assert R.total_error([1.5])[0] == pytest.approx(2.5)
    assert R.indistinguishable_groups([0.0, -3.5], [2.5, 2.5]) == [[0, 1]]
    assert R.indistinguishable_groups([0.0, -3.6], [2.5, 2.5]) == [[0], [1]]


def test_groups_do_not_chain(R):
    """Given modes at 0, -2.5, -5 (each pair of neighbours within the
    floor), then membership is measured from the leader only: {0, 1}, {2}."""
    sig = R.total_error([0.0, 0.0, 0.0])
    assert R.indistinguishable_groups([0.0, -2.5, -5.0], sig) == [[0, 1], [2]]


def test_a_mode_without_lnz_gets_no_probability(R):
    """Given a refused mode (nan lnZ), then it gets 0 and joins no group."""
    out = R.robust_probabilities([5.0, np.nan], [0.1, np.inf])
    assert out["p_robust"][1] == 0.0
    assert out["group_of"] == [0, -1]
    assert out["p_robust"][0] == pytest.approx(1.0)


def test_active_solutions_are_renormalized_to_one(R):
    """Given probabilities 0.6, 0.3999, 1e-4, then the last is inactive,
    the two active ones sum to exactly 1, and the inactive keeps its raw
    value (the microlens-submit validator sums ACTIVE solutions only)."""
    act, rel = R.active_and_renormalized(np.array([0.6, 0.3999, 1e-4]))
    assert act.tolist() == [True, True, False]
    assert rel[:2].sum() == pytest.approx(1.0)
    assert rel[2] == pytest.approx(1e-4)


# --- flags --------------------------------------------------------------------


def _sol(**kw):
    base = {
        "lpcheck": {"dlp_max_abs": 1e-9},
        "bridge_refused": False,
        "bridge_state": "ok",
        "equilibrated": True,
        "converged": True,
        "err_scale": {"F146": 1.0, "F087": 1.02, "F213": 0.97},
        "outlier_kinds": {},
        "stuck_frac": 0.0,
        "lp_rhat": 1.01,
        "invalid_frac": 0.0,
        "re2": 0.03,
        "out_scale_rhat": {"F146": 1.0, "F087": 1.0, "F213": 1.0},
    }
    base.update(kw)
    return base


def test_a_clean_solution_has_no_flags(R):
    assert R.reliability_flags(_sol()) == ([], [])


@pytest.mark.parametrize(
    "band,value", [("F213", 21.0), ("F213", 8.1), ("F087", 3.2), ("F146", 0.7)]
)
def test_err_scale_out_of_range_is_unreliable(R, band, value):
    """The known cases (016 F213 21.0, 173 F213 8.1, 163 F087 3.2) and a
    deflated 0.7 are UNRELIABLE whatever the lnZ."""
    es = {"F146": 1.0, "F087": 1.0, "F213": 1.0, band: value}
    bad, _ = R.reliability_flags(_sol(err_scale=es))
    assert len(bad) == 1 and band in bad[0] and ("%.3g" % value) in bad[0]


def test_err_scale_edges_are_inside(R):
    bad, _ = R.reliability_flags(_sol(err_scale={"F146": 0.8, "F087": 1.25}))
    assert bad == []


def test_lp_not_reproduced_and_refused_are_unreliable_trending_warns(R):
    bad, warn = R.reliability_flags(
        _sol(lpcheck={"dlp_max_abs": 0.5}, bridge_refused=True,
             bridge_state="re2-too-large", equilibrated=False)
    )  # fmt: skip
    assert len(bad) == 2
    assert any("lp NOT reproduced" in b for b in bad)
    assert any("re2-too-large" in b for b in bad)
    assert warn[0] == "NOT EQUILIBRATED (lp trending)"


def test_outlier_kinds_split_between_unreliable_and_warning(R):
    """ANOMALY (the mixture absorbing points AT the event) is UNRELIABLE for
    a single lens and a warning for a binary; VARIABILITY/FORGIVEN are
    warnings; AT_CAP carries the band's out_scale Rhat beside it."""
    kinds = {
        "Roman_F146": ["ANOMALY", "FORGIVEN"],
        "Roman_F213": ["AT_CAP", "VARIABILITY"],
    }
    bad, warn = R.reliability_flags(
        _sol(outlier_kinds=kinds, out_scale_rhat={"F213": 1.70}, arch="1S1L")
    )
    assert bad == ["ANOMALY outliers in Roman_F146"]
    assert "FORGIVEN in Roman_F146" in warn
    assert "VARIABILITY in Roman_F213" in warn
    assert "AT_CAP in Roman_F213 (out_scale Rhat 1.7)" in warn
    # the high out_scale Rhat is reported once, next to AT_CAP
    assert sum("1.7" in w for w in warn) == 1
    bad2, warn2 = R.reliability_flags(_sol(outlier_kinds=kinds, arch="1S2L"))
    assert bad2 == [] and "ANOMALY outliers in Roman_F146" in warn2


def test_sampler_health_warnings(R):
    _, warn = R.reliability_flags(
        _sol(stuck_frac=0.3, lp_rhat=1.5, invalid_frac=0.27, re2=0.15,
             converged=False, max_rhat=1.69, worst_rhat_var="x", min_ess=58,
             n_chains=1)
    )  # fmt: skip
    joined = " | ".join(warn)
    for s in ("stuck chains 0.30", "lp Rhat 1.5", "invalid draws 27.0%",
              "re2 0.15", "unconverged", "only 1 chain"):  # fmt: skip
        assert s in joined


def test_outlier_kinds_from_audit_rows(R):
    rows = [
        {"name": "Roman_F146", "flags": [["AT_CAP", "..."]]},
        {"name": "Roman_F146", "flags": [["FORGIVEN", "..."]]},
        {"name": "Roman_F087", "flags": []},
    ]
    assert R.outlier_kinds(rows) == {
        "Roman_F146": ["AT_CAP", "FORGIVEN"],
        "Roman_F087": [],
    }


# --- fit discovery ------------------------------------------------------------


def test_architecture_and_ld_pin(R):
    assert R.architecture({"lens": [{"body": "a"}, {"body": "b"}]}) == "1S2L"
    assert R.architecture({"lens": [{"body": "a"}]}) == "1S1L"
    pinned = {"band.F146.u1": {"initval": 0.0, "sigma": 0}}
    free = {"band.F146.u1": {"initval": 0.3}}
    assert R.ld_pinned_zero(pinned)
    assert not R.ld_pinned_zero(free)
    assert not R.ld_pinned_zero({})


def test_circular_percentiles_straddle_zero(R):
    """Given an angle scattered around 0 rad, the median is reported near 0
    (mod 2 pi) with a small width, not near pi."""
    a = np.array([-0.02, -0.01, 0.0, 0.01, 0.02]) % (2 * math.pi)
    med, lo, hi = R.circ_percentiles(a)
    assert min(med, 2 * math.pi - med) < 1e-9
    assert 0 < lo < 0.03 and 0 < hi < 0.03


# --- the microlens-submit files -----------------------------------------------


def _binary_solution(R):
    s = {
        "solution_id": R.solution_id(94, "1S2L", 0),
        "arch": "1S2L",
        "mode": 0,
        "bands": ["F146", "F087", "F213"],
        "t0_par": 2462793.5,
        "finite_source": True,
        "is_active": True,
        "relative_probability": 0.75,
        "log_likelihood": 1.06e6,
        "n_data_points": 49093,
        "summary": {
            "t_0": (2462793.51, 0.001, 0.002),
            "u_0": (0.0515, 0.0008, 0.0008),
            "t_E": (8.6, 0.1, 0.1),
            "rho": (0.0027, 0.0001, 0.0001),
            "pi_E_N": (0.1, 0.05, 0.05),
            "pi_E_E": (-0.02, 0.03, 0.03),
            "s": (1.82, 0.02, 0.02),
            "q": (0.0028, 0.0002, 0.0002),
            "alpha_rad": (5.9, 0.01, 0.01),
            "alpha_deg_user": (338.0, 0.6, 0.6),
        },
    }
    for b, f in (("F146", 1e-8), ("F087", 3e-9), ("F213", 2e-8)):
        s["summary"]["f_source_" + b] = (f, f / 100, f / 100)
        s["summary"]["f_blend_" + b] = (f / 3, f / 50, f / 50)
    return s


def test_solution_json_fields_and_units(R):
    """Given a 1S2L mode, the solution file uses only microlens-submit's
    field names, the challenge's parameter names, alpha in RADIANS (not the
    trace's degrees), [lower, upper] uncertainties, F<band index>_S/_B with
    0 = F146, 1 = F087, 2 = F213, and a t_ref because parallax is in."""
    s = _binary_solution(R)
    j = R.solution_json(s, "events/x/solutions/y.md")
    assert set(j) <= SOLUTION_FIELDS
    assert j["solution_id"] == "RMDC26_000094_1S2L_m1"
    assert j["model_type"] == "1S2L"
    assert j["bands"] == ["0", "1", "2"]
    assert set(j["higher_order_effects"]) == {"finite-source", "parallax"}
    assert j["t_ref"] == 2462793.5
    p = j["parameters"]
    for k in ("t0", "u0", "tE", "s", "q", "alpha", "rho", "piEN", "piEE"):
        assert k in p and k in j["parameter_uncertainties"]
        assert j["parameter_uncertainties"][k] > 0
    assert p["alpha"] == pytest.approx(5.9)
    assert 0.0 <= p["alpha"] < 2 * math.pi
    assert p["F0_S"] == pytest.approx(1e-8)
    assert p["F1_S"] == pytest.approx(3e-9)
    assert p["F2_B"] == pytest.approx(2e-8 / 3)
    assert {f"F{i}_{k}" for i in range(3) for k in "SB"} <= set(p)
    # one number, the larger half-width: the validator rejects [lo, hi]
    # with lo > hi, which a skewed posterior produces half the time
    assert j["parameter_uncertainties"]["t0"] == 0.002
    assert p["t_ref"] == j["t_ref"]
    assert j["limb_darkening_coeffs"] == {"0": [0.0], "1": [0.0], "2": [0.0]}
    assert j["is_active"] is True and j["relative_probability"] == 0.75
    assert isinstance(j["n_data_points"], int)


def test_pspl_solution_has_no_binary_parameters(R):
    s = _binary_solution(R)
    s["arch"] = "1S1L"
    s["solution_id"] = R.solution_id(94, "1S1L", 1)
    j = R.solution_json(s, "n.md")
    assert j["solution_id"] == "RMDC26_000094_1S1L_m2"
    assert not {"s", "q", "alpha"} & set(j["parameters"])
    assert {"t0", "u0", "tE"} <= set(j["parameters"])


def test_no_parallax_means_no_t_ref(R):
    s = _binary_solution(R)
    del s["summary"]["pi_E_N"], s["summary"]["pi_E_E"]
    j = R.solution_json(s, "n.md")
    assert "parallax" not in j["higher_order_effects"]
    assert j["t_ref"] is None and "t_ref" not in j["parameters"]


def test_event_json_and_notes(R):
    assert R.event_json(94) == {"event_id": "RMDC26_000094"}
    s = _binary_solution(R)
    s.update(unreliable=["err_scale F213 = 21 outside [0.8, 1.25]"],
             warnings=[], lnZ=1.0, lnZ_err=0.2, re2=0.03, n_draws=100,
             p_raw=0.7, group="G1", group_members=[s["solution_id"]])  # fmt: skip
    txt = R.notes_markdown(s, {"arch_prob": {"1S2L": 1.0}, "missing": []})
    assert "**FIT_UNRELIABLE:** err_scale F213" in txt
    assert "EVIDENCE_UNRELIABLE" not in txt
    s["unreliable"].append("bridge refused (re2-too-large)")
    txt = R.notes_markdown(s, {"arch_prob": {"1S2L": 1.0}, "missing": []})
    assert "**EVIDENCE_UNRELIABLE:** bridge refused (re2-too-large)" in txt
    assert R.split_unreliable(s["unreliable"]) == (
        ["bridge refused (re2-too-large)"],
        ["err_scale F213 = 21 outside [0.8, 1.25]"],
    )
    assert R.solution_flags(dict(s, warnings=[])) == [
        "FIT_UNRELIABLE",
        "EVIDENCE_UNRELIABLE",
    ]
    assert R.CONVENTIONS in txt
    assert "RADIANS" in txt and "1S2L: 1" in txt
    assert "| t_0 | 2462793.51 | 0.001 | 0.002 |" in txt


def test_wrapup_banners_are_parsed(R):
    """The fit's own summary.txt banner (burn-in, convergence WARNING) and
    its log's NOT EQUILIBRATED warning, as run.py writes them."""
    summ = (
        "# burn-in discarded: 7500 draws (75% of 10000); chains kept: 36\n"
        "# WARNING: convergence NOT reached -- max Rhat=1.695 "
        "(mulensinstrument.out_scale), min ESS=58 "
        "(mulensinstrument.out_scale); thresholds Rhat<=1.01, ESS>=1000\n"
    )
    log = "... WARNING  exozippy.run: NOT EQUILIBRATED -- Rhat and ESS ..."
    w = R.parse_wrapup_text(summ, log)
    assert w["burnin"] == 7500 and w["n_draw"] == 10000
    assert w["chains_kept"] == 36
    assert w["unconverged"] and w["max_rhat"] == pytest.approx(1.695)
    assert w["worst_rhat_var"] == "mulensinstrument.out_scale"
    assert w["min_ess"] == 58
    assert w["not_equilibrated"] is True
    clean = R.parse_wrapup_text(
        "# burn-in discarded: 0 draws (0% of 8000); chains kept: 36\n", ""
    )
    assert clean["burnin"] == 0 and not clean["unconverged"]
    assert clean["not_equilibrated"] is False
    assert "not_equilibrated" not in R.parse_wrapup_text("", None)


# --- POSSIBLE_UNLISTED_MODE and the forced set --------------------------------

_DIMS = ["source.t_0_raw", "source.u_0_raw", "star.pm_ra_raw[0]",
         "star.pm_ra_raw[1]", "mulensinstrument.err_scale_raw[2]"]  # fmt: skip


def _beyond(rows):
    return np.array([[d in r for d in _DIMS] for r in rows], dtype=bool)


def test_a_rejection_concentrated_in_t0_is_a_possible_unlisted_mode(R):
    """173's shape: 28% of the draws rejected, all out of range in
    source.t_0_raw only -> POSSIBLE_UNLISTED_MODE (source.t_0_raw)."""
    B = _beyond([{"source.t_0_raw"}] * 28)
    assert R.unlisted_mode_params(B, _DIMS, n_total=100) == ["source.t_0_raw"]
    bad, warn = R.reliability_flags(
        _sol(unlisted_mode_params=["source.t_0_raw"], rawz_frac=0.28,
             invalid_frac=0.28)
    )  # fmt: skip
    assert bad == []
    assert any(
        w.startswith("POSSIBLE_UNLISTED_MODE (source.t_0_raw)") for w in warn
    )
    # it replaces, not duplicates, the generic invalid-fraction warning
    assert not any(w.startswith("invalid draws") for w in warn)


def test_two_elements_of_one_vector_count_as_one_parameter(R):
    B = _beyond([{"star.pm_ra_raw[0]", "star.pm_ra_raw[1]"}] * 10)
    assert R.unlisted_mode_params(B, _DIMS, n_total=100) == ["star.pm_ra_raw"]


def test_a_small_or_diffuse_or_noise_rejection_is_not(R):
    """016's shape (2.8%, below 5%) does not fire; nor does a rejection
    spread over three parameters, nor one in a NOISE parameter."""
    one = [{"source.t_0_raw"}]
    assert R.unlisted_mode_params(_beyond(one * 3), _DIMS, 100) == []
    diffuse = [
        {"source.t_0_raw"},
        {"source.u_0_raw"},
        {"star.pm_ra_raw[0]"},
    ] * 4
    assert R.unlisted_mode_params(_beyond(diffuse), _DIMS, 100) == []
    noise = [{"mulensinstrument.err_scale_raw[2]"}] * 20
    assert R.unlisted_mode_params(_beyond(noise), _DIMS, 100) == []
    _, warn = R.reliability_flags(_sol(invalid_frac=0.12))
    assert warn == ["invalid draws 12.0%"]


def test_two_geometric_parameters_can_carry_it(R):
    rows = [{"source.t_0_raw"}] * 6 + [
        {"source.t_0_raw", "source.u_0_raw"}
    ] * 4
    got = R.unlisted_mode_params(_beyond(rows), _DIMS, 100)
    assert got == ["source.t_0_raw", "source.u_0_raw"] or got == [
        "source.t_0_raw"
    ]
    # 6/10 covered by t_0 alone (< 0.8), so the pair is needed
    assert got == ["source.t_0_raw", "source.u_0_raw"]


@pytest.mark.parametrize("ev", [16, 163, 173])
def test_forced_events_are_unreliable_whatever_else(R, ev):
    bad, _ = R.reliability_flags(_sol(event=ev))
    assert len(bad) == 1 and bad[0].startswith("forced:")
    assert R.reliability_flags(_sol(event=94)) == ([], [])


def test_a_refused_mode_near_the_top_is_reported(R):
    """Given an accepted mode at 100 and refused ones at 103 and 90, the
    refused 103 is reported (+3 nats), the one 10 below (beyond 3 floors =
    6 nats) is not; with nothing accepted, nothing is reported."""
    lnz = [100.0, 103.0, 90.0]
    assert R.refused_competitors(lnz, [True, False, False]) == [(1, 3.0)]
    assert R.refused_competitors(lnz, [False, False, False]) == []


def test_a_trace_older_than_its_config_or_whitening_is_stale(R):
    """015/022/187's shape: configs regenerated after the fit; 015 also has
    a re-fit under way that has overwritten its whitening."""
    assert R.staleness_from_mtimes(100.0, 50.0, 50.0, 40.0) == ""
    why = R.staleness_from_mtimes(100.0, 200.0, 200.0, 40.0)
    assert "config is newer" in why and "params file is newer" in why
    assert "re-fit" in R.staleness_from_mtimes(100.0, 50.0, 50.0, 500.0)
    # a whitening written in the same minute as the trace is not a re-fit
    assert R.staleness_from_mtimes(100.0, 50.0, 50.0, 130.0) == ""
    assert R.staleness_from_mtimes(None, 1.0, 1.0, 1.0) == "no trace"


# --- the manifest-driven submission (SED-tied fits, 2026-10) ---------------

#: microlens_submit 0.17.9 validate_parameters.PHYSICAL_PARAMETERS.
PHYSICAL_KEYS = {
    "Mtot", "M1", "M2", "M3", "M4", "D_L", "D_S", "thetaE", "piE", "piE_N",
    "piE_E", "piE_parallel", "piE_perpendicular", "piE_l", "piE_b",
    "mu_rel", "mu_rel_N", "mu_rel_E", "mu_rel_l", "mu_rel_b", "phi",
}  # fmt: skip

MANIFEST = """RMDC26 submission manifest
Regenerate: python make_manifest.py

FULL   = required fields + optional physical_parameters (2 events)

009  FULL  arm=t1  dir=out_long
    trace   /c/sed/t1/009/out_long/009_trace.nc
    results /c/sed/t1/009/out_long/009_results.csv
    config  /c/sed/t1/009/run_long.yaml
    params  /c/sed/t1/009/run.params.yaml
    verdict 009   OK                                 OK                             1.040    1.00     1.78   0.190

030  SCORED  arm=t2  dir=out
    trace   /c/sed/t2/030/out/030_trace.nc
    config  /c/sed/t2/030/run.yaml   [MISSING]
    verdict 030   OK                                 A_V vs stated,theta* off       1.315    1.00     0.39   0.106

044  FULL  *** NO RESULTS ***

"""


def test_the_manifest_is_parsed_block_by_block(R):
    """Given the campaign's manifest text, each block gives the event, tier,
    arm, run dir, paths, MISSING markers and the verdict row; a NO RESULTS
    block is kept and marked."""
    e = R.parse_manifest(MANIFEST)
    assert [x["event"] for x in e] == [9, 30, 44]
    assert e[0]["tier"] == "FULL" and e[0]["arm"] == "t1"
    assert e[0]["dir"] == "out_long"
    assert e[0]["paths"]["config"] == "/c/sed/t1/009/run_long.yaml"
    assert e[1]["missing"] == ["config"] and e[1]["tier"] == "SCORED"
    assert e[0]["verdict"].startswith("009   OK")
    assert e[2]["no_results"] and e[2]["arm"] is None
    with pytest.raises(ValueError, match="twice"):
        R.parse_manifest(MANIFEST + "009  FULL  arm=t2  dir=out\n")


def test_the_tier_lists_decide_what_is_submitted(R):
    """Given the manifest and tier lists, only events in a list AND with
    results are submitted; the lists win over the manifest's tier, and an
    event in neither list, in both, or missing from the manifest is a
    reported problem, never silently dropped or silently submitted."""
    e = R.parse_manifest(MANIFEST + "077  FULL  arm=t2  dir=out\n")
    sub, probs = R.select_submitted(e, full=[9, 44, 101], scored=[30, 77])
    assert [(x["event"], x["tier"]) for x in sub] == [
        (9, "FULL"),
        (30, "SCORED"),
        (77, "SCORED"),
    ]
    text = "\n".join(probs)
    assert "044 (FULL): manifest has NO RESULTS" in text
    assert "077: manifest says FULL, tier lists say SCORED" in text
    assert "101 is in a tier list but not in the manifest" in text
    sub, probs = R.select_submitted(e, full=[9], scored=[9, 30])
    assert [x["event"] for x in sub] == [30]
    assert any("BOTH tier lists" in p for p in probs)
    assert any(
        "077 is in the manifest (FULL) but in neither" in p for p in probs
    )
    assert R.read_event_list("006\n# c\n 013  # x\n\n") == [6, 13]


def _probe(**dirs):
    def probe(name):
        return dict({"trace": False, "results": False, "stale": ""},
                    **dirs.get(name, {}))  # fmt: skip

    return probe


def test_out_long_is_always_preferred_when_usable(R):
    """Given a finished out_long/ run, it is used even when the manifest
    names out/; a long rerun in flight (whitening newer than the trace, or
    no trace yet) is not usable and out/ is used with a note."""
    done = {"trace": True, "results": True}
    run, note = R.prefer_out_long("out", _probe(out=done, out_long=done))
    assert run == "out_long" and "out_long/ holds a finished run" in note
    assert R.prefer_out_long("out_long", _probe(out=done, out_long=done)) == (
        "out_long",
        "",
    )
    inflight = {"trace": True, "results": True,
                "stale": "whitening is newer than the trace (a re-fit is under way)"}  # fmt: skip
    run, note = R.prefer_out_long(
        "out_long", _probe(out=done, out_long=inflight)
    )
    assert run == "out" and "re-fit" in note
    run, note = R.prefer_out_long("out", _probe(out=done))
    assert (run, note) == ("out", "")
    run, note = R.prefer_out_long("out", _probe(out={"trace": True}))
    assert run is None and "no results" in note


def test_the_config_that_wrote_the_trace_is_found(R):
    """run_long.yaml and run_long2.yaml both write out_long/: the newest one
    that predates the trace is the trace's; a newer config (a later rerun)
    never is; the manifest's choice wins when it qualifies."""
    cands = [("run.yaml", "out/009", 10.0), ("run_long.yaml", "out_long/009", 20.0),
             ("run_long2.yaml", "out_long/009", 30.0)]  # fmt: skip
    assert R.pick_config(cands, "out_long", 9, 35.0) == "run_long2.yaml"
    assert R.pick_config(cands, "out_long", 9, 25.0) == "run_long.yaml"
    assert (
        R.pick_config(cands, "out_long", 9, 35.0, "run_long.yaml")
        == "run_long.yaml"
    )
    assert R.pick_config(cands, "out", 9, 35.0) == "run.yaml"
    assert R.pick_config(cands, "out", 9, 5.0) is None


def test_the_architecture_rule(R):
    """1S2L: dchi2_np > 10 on the blind grid -> binary favoured, no flag;
    <= 10 or no grid -> ARCHITECTURE_UNCERTAIN.  A pinned q under a small
    dchi2 (105: dchi2 8, q 2.57e-5 +/- 2.6e-6) spells out the tension.
    1S1L: the audit decides; ANOMALY flags a possible missed companion."""
    v = R.architecture_verdict("1S2L", 3000.0)
    assert v["flag"] is None
    assert (
        "binary lens favoured by dchi2_np = 3000.0 on the blind grid"
        in v["statement"]
    )
    assert (
        R.architecture_verdict("1S2L", 10.0)["flag"]
        == "ARCHITECTURE_UNCERTAIN"
    )
    assert R.architecture_verdict("1S2L", 10.01)["flag"] is None
    v = R.architecture_verdict("1S2L", 8.0, q=(2.57e-5, 2.6e-6, 2.5e-6))
    assert v["flag"] == "ARCHITECTURE_UNCERTAIN" and v["dchi2_np"] == 8.0
    assert (
        "prefers a companion by only dchi2 = 8 while the posterior pins q "
        "to 10%; the coarse grid can under-resolve a narrow solution, or "
        "the fit may be overconfident; unresolved"
    ) in v["statement"]
    # a loose q gets no tension sentence
    v = R.architecture_verdict("1S2L", 8.0, q=(1e-3, 9e-4, 2e-3))
    assert "pins q" not in v["statement"]
    v = R.architecture_verdict("1S2L", None)
    assert v["flag"] == "ARCHITECTURE_UNCERTAIN" and v["rule"] == "no-grid"
    v = R.architecture_verdict("1S1L", 1e6)  # a grid never applies to 1S1L
    assert v["flag"] is None and v["rule"] == "audit" and v["dchi2_np"] is None
    v = R.architecture_verdict("1S1L", anomaly_files=["Roman_F146"])
    assert (
        v["flag"] == "ANOMALY_POSSIBLE_COMPANION"
        and "Roman_F146" in v["statement"]
    )


def test_a_direct_comparison_supersedes_the_grid(R, tmp_path):
    """105: the grid gave dchi2_np = 8 but a direct 1L1S arm is worse by 513
    -> no flag, rule 'direct', the grid quoted for the record.  A direct
    dchi2 under the threshold still flags; 1S1L ignores it; the loader
    merges every arch*/arch_direct.json."""
    rec = {"dchi2": 513.0, "n": 49096, "k_extra": 14}
    v = R.architecture_verdict("1S2L", 8.0, q=(2.57e-5, 2.6e-6, 2.5e-6),
                               direct=rec)  # fmt: skip
    assert v["flag"] is None and v["rule"] == "direct"
    assert v["dchi2_direct"] == 513.0
    assert (
        "worse than the binary by dchi2 = 513.0 for 14 extra"
        in (v["statement"])
    )
    assert "the blind grid gave dchi2_np = 8.0" in v["statement"]
    assert "pins q" not in v["statement"]
    v = R.architecture_verdict("1S2L", 3000.0, direct=dict(rec, dchi2=5.0))
    assert v["flag"] == "ARCHITECTURE_UNCERTAIN" and v["rule"] == "direct"
    v = R.architecture_verdict("1S1L", direct=rec)
    assert v["rule"] == "audit"
    for sub, ev in (("arch105", "105"), ("arch200", "200")):
        (tmp_path / sub).mkdir()
        (tmp_path / sub / "arch_direct.json").write_text(
            '{"%s": {"dchi2": 99.0}}' % ev
        )
    assert set(R.load_direct_arch(tmp_path)) == {"105", "200"}
    assert R.load_direct_arch(tmp_path / "none") == {}


def test_physical_parameters_are_formed_per_draw(R):
    """Given anti-correlated draws, each physical parameter is the median of
    the per-draw quantity, not a product of medians; M2 = Mtot - M1 per
    draw, distances pc -> kpc, piE = hypot per draw, and only schema keys
    reach the JSON."""
    rng = np.random.default_rng(3)
    n = 4001
    m1 = rng.lognormal(np.log(0.5), 0.5, n)
    q = rng.lognormal(np.log(1e-3), 1.0, n)
    d = {
        "mulensevent.theta_E": rng.lognormal(np.log(0.3), 0.3, (n, 1)),
        "star.mass": np.column_stack(
            [rng.uniform(0.5, 1, n), m1]
        ),  # lens is index 1
        "mulensevent.mlens_total": (m1 * (1 + q))[:, None],
        "star.distance": np.column_stack(
            [np.full(n, 8000.0), np.full(n, 6000.0)]
        ),
        "mulensevent.pi_E_N": rng.normal(0.1, 0.2, (n, 1)),
        "mulensevent.pi_E_E": rng.normal(-0.1, 0.2, (n, 1)),
        "mulensevent.mu_rel_mag": np.full((n, 1), 5.0),
        "mulensevent.mu_rel_geo_mag": np.full((n, 1), 4.8),
    }
    dr = R.physical_draws(d, "1S2L", lens_idx=1, src_idx=0)
    np.testing.assert_allclose(dr["M2"], q * m1)  # per draw
    np.testing.assert_allclose(dr["Mtot"], dr["M1"] + dr["M2"])
    np.testing.assert_allclose(
        dr["piE"],
        np.hypot(d["mulensevent.pi_E_N"][:, 0], d["mulensevent.pi_E_E"][:, 0]),
    )
    summ = R.physical_summary(dr)
    assert summ["D_L"][0] == pytest.approx(6.0) and summ["D_S"][
        0
    ] == pytest.approx(8.0)
    assert summ["M2"][0] == pytest.approx(np.median(q * m1))
    # ... which is not the product of the medians
    assert summ["M2"][0] != pytest.approx(
        np.median(q) * np.median(m1), rel=0.01
    )
    piE_of_medians = math.hypot(
        np.median(d["mulensevent.pi_E_N"]), np.median(d["mulensevent.pi_E_E"])
    )
    assert summ["piE"][0] != pytest.approx(piE_of_medians, rel=0.05)
    par, unc = R.submit_physical(summ)
    assert set(par) <= PHYSICAL_KEYS and set(par) == set(unc)
    assert "mu_rel_geo" not in par and par["mu_rel"] == 5.0
    single = R.physical_draws(d, "1S1L", lens_idx=1, src_idx=0)
    assert "M2" not in single and np.array_equal(single["Mtot"], m1)


def test_body_indices_follow_the_config(R):
    cfg = {"star": [{"name": "Source"}, {"name": "Lens"}],
           "lens": [{"body": "star.Lens"}, {"body": "planet.Companion"}],
           "source": [{"body": "star.Source"}]}  # fmt: skip
    assert R.body_indices(cfg) == (1, 0)


def test_the_extinction_caveat_only_above_rhat_1p3(R):
    assert R.extinction_caveat(1.29) is None
    assert R.extinction_caveat(1.3) is None
    assert R.extinction_caveat(float("nan")) is None
    assert "1.392 > 1.3" in R.extinction_caveat(1.392)


def test_alpha_round_trips_degrees_to_radians_with_no_offset(R):
    """013's results.csv alpha is -51.04 deg.  The submitted value is a pure
    deg -> rad conversion wrapped to [0, 2 pi): never negated, no +180 (the
    C21 source-trajectory shift JDE ruled out), and back to degrees it is
    the same angle."""
    deg = -51.04
    rad = float(R.alpha_submit(deg))
    assert 0.0 <= rad < 2 * math.pi
    assert rad == pytest.approx(2 * math.pi - math.radians(51.04))
    back = math.degrees(rad)
    assert (back - deg) % 360.0 == pytest.approx(0.0, abs=1e-9)
    assert abs((back - deg - 180.0) % 360.0) > 1.0  # not the C21 +180
    assert float(R.alpha_submit(139.56623453)) == pytest.approx(
        math.radians(139.56623453)
    )


def test_a_t0_with_an_offset_subtracted_is_refused(R):
    """013's t0 is 2463102.43884 (full BJD); 3102.43884 means an offset
    was subtracted, and the writer refuses it loudly."""
    s = _binary_solution(R)
    s["summary"]["t_0"] = (3102.43884, 1e-4, 1e-4)
    with pytest.raises(ValueError, match="full BJD"):
        R.solution_json(s, "n.md")
    s = _binary_solution(R)
    s["summary"]["alpha_rad"] = (-0.5, 0.01, 0.01)
    with pytest.raises(ValueError, match="alpha"):
        R.solution_json(s, "n.md")


def test_physical_parameters_only_when_the_caller_sets_them(R):
    """FULL sets physical_submit; SCORED (and legacy) never carries the
    block."""
    s = _binary_solution(R)
    assert R.solution_json(s, "n.md")["physical_parameters"] is None
    s["physical_submit"] = (
        {"thetaE": 0.3, "Mtot": 0.5},
        {"thetaE": 0.01, "Mtot": 0.1},
    )
    j = R.solution_json(s, "n.md")
    assert set(j) <= SOLUTION_FIELDS
    assert j["physical_parameters"] == {"thetaE": 0.3, "Mtot": 0.5}
    assert j["physical_parameter_uncertainties"]["Mtot"] == 0.1


def test_the_verdict_row_is_parsed(R):
    v = R.parse_verdict_line(
        "138   OK                                 MIST wall,M-R locus,theta* off   1.451    1.02     8.73   0.120",
        header=R.VERDICT_HEADER_LEGACY,
    )
    assert v["lc"] == "OK" and v["phy"] == "MIST wall,M-R locus,theta* off"
    assert v["rhat"] == pytest.approx(1.451) and v["outfrac"] == pytest.approx(
        0.12
    )
    assert R.parse_verdict_line(None) == {}
    # 2026-10-24 format: modes/stuck appended, an LC verdict with spaces and
    # a PHY verdict overflowing its column (the rows that broke tok[-4:])
    hdr = (
        "ev    LC verdict                               PHY verdict          "
        "           Rhat    chi2    ratio  outfrc  modes  stuck"
    )
    v = R.parse_verdict_line(
        "094   OK                                       MIST wall,M-R locus,"
        "theta* off (0.037)   1.255     nan     0.04   0.054      1   1.00",
        header=hdr,
    )
    assert v["phy"] == "MIST wall,M-R locus,theta* off (0.037)"
    assert v["rhat"] == pytest.approx(1.255) and np.isnan(v["chi2"])
    assert v["ratio"] == pytest.approx(0.04) and v["stuck"] == 1.0
    v = R.parse_verdict_line(
        "006   modes stuck 2/98%                        theta* off (13)      "
        "          1.230    1.08    13.42   0.150      2   0.98",
        header=hdr,
    )
    assert v["lc"] == "modes stuck 2/98%" and v["phy"] == "theta* off (13)"
    assert v["rhat"] == pytest.approx(1.23) and v["modes"] == 2.0
    # 2026-10-25: Rhat split into LC / physical / global columns
    hdr3 = (
        "ev    LC verdict                               PHY verdict          "
        "           Rhat  Rhat_phys  Rhat_all    chi2    ratio  outfrc  modes"
        "  stuck"
    )
    v = R.parse_verdict_line(
        "106   OK                                       Rhat_phys=1.53(mann."
        "absks)     1.440      1.530     1.530    1.17     0.90   0.130      1"
        "   1.00",
        header=hdr3,
    )
    assert v["phy"] == "Rhat_phys=1.53(mann.absks)"
    assert v["rhat"] == pytest.approx(1.44)
    assert v["rhat_phys"] == pytest.approx(1.53)
    assert v["rhat_all"] == pytest.approx(1.53) and v["stuck"] == 1.0
    assert "FULL" in R.tier_reason("FULL", v) and "withheld" in R.tier_reason(
        "SCORED", v
    )


def test_the_event_note_is_the_campaign_note_verbatim_plus_one_section(R):
    """notes_path = the campaign's generated note, byte for byte, then ONE
    appended section with the per-mode evidence, probabilities, flags, the
    architecture statement and the conventions (the ruled alpha sentence)."""
    camp = "# RMDC26_105\n\nModel: **1S2L**.\n\n## Architecture\n\ntheirs\n"
    s = _binary_solution(R)
    s.update(unreliable=[], warnings=["lp Rhat 1.2"], lnZ=12.5, lnZ_err=0.3,
             sigma_total=2.02, re2=0.05, bridge_state="ok",
             bridge_refused=False, group="G1")  # fmt: skip
    r = dict(s, solution_id=s["solution_id"].replace("m1", "m2"),
             bridge_refused=True, bridge_state="re2-too-large", lnZ=3.0,
             is_active=False, relative_probability=1e-4)  # fmt: skip
    ev = {"solutions": [s, r], "basis": "accepted",
          "flags": ["ARCHITECTURE_UNCERTAIN"],
          "architecture": R.architecture_verdict("1S2L", 8.0, q=(2.57e-5, 2.6e-6, 2.6e-6))}  # fmt: skip
    txt = R.event_notes_markdown(camp, ev)
    assert txt.startswith(camp)
    assert txt.count(R.NOTES_SECTION) == 1 and txt.count("\n## ") == 2
    tail = txt[len(camp) :]
    assert (
        "| RMDC26_000094_1S2L_m1 | 12.50 | 0.30 | 2.02 | ok | G1 | 0.75 | yes | WARN(1) |"
        in tail
    )
    assert "REFUSED (re2-too-large): lnZ indicative only" in tail
    assert "ARCHITECTURE_UNCERTAIN" in tail and "pins q to 10%" in tail
    assert R.ALPHA_SENTENCE in tail
    assert (
        "converted from degrees to radians with no other transformation"
        in tail
    )
    assert "theta*" not in tail  # never re-described here
    assert "lp Rhat 1.2" in tail


def test_results_csv_is_read_in_both_schemas(tmp_path):
    """A fit that settles into one mode writes the 4-column schema; the
    7-column-only reader dropped it (029, 040), and the alpha/t0 round trip
    then compared nothing.  Both schemas must give the same rows."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "rmdc26_validate", EX / "04_validate.py"
    )
    V = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(V)
    multi = tmp_path / "m.csv"
    multi.write_text(
        "# parname, mode, weight, weight_err, value, up_err, low_err\n"
        "lens.Companion.alpha,all,1.0,,118.853,0.098,0.1\n"
        "source.t_0,all,1.0,,2461821.8299,0.0022,0.0021\n"
        "source.t_0,rejected-seed1,0.0,,2461800.0,1.0,1.0\n"
    )
    single = tmp_path / "s.csv"
    single.write_text(
        "# parname, value, up_err, low_err\n"
        "lens.Companion.alpha,118.853,0.098,0.1\n"
        "source.t_0,2461821.8299,0.0022,0.0021\n"
    )
    rm, rs = V.results_rows(multi), V.results_rows(single)
    assert rm == rs
    assert rs["source.t_0"] == [("all", 2461821.8299, 0.0022, 0.0021)]
    sol = {"parameters": {"alpha": math.radians(118.853), "t0": 2461821.83}}
    da, dt = V.roundtrip(sol, rs)
    assert da == pytest.approx(0.0, abs=1e-9) and dt < 1e-3
    assert V.roundtrip(sol, {}) == (None, None)
