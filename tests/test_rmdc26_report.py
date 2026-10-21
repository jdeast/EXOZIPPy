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
    assert p["alpha"] == pytest.approx(5.9)
    assert 0.0 <= p["alpha"] < 2 * math.pi
    assert p["F0_S"] == pytest.approx(1e-8)
    assert p["F1_S"] == pytest.approx(3e-9)
    assert p["F2_B"] == pytest.approx(2e-8 / 3)
    assert {f"F{i}_{k}" for i in range(3) for k in "SB"} <= set(p)
    assert j["parameter_uncertainties"]["t0"] == [0.001, 0.002]
    assert j["limb_darkening_coeffs"] == {"u_0": 0.0, "u_1": 0.0, "u_2": 0.0}
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
    assert j["t_ref"] is None


def test_event_json_and_notes(R):
    assert R.event_json(94) == {"event_id": "RMDC26_000094"}
    s = _binary_solution(R)
    s.update(unreliable=["err_scale F213 = 21 outside [0.8, 1.25]"],
             warnings=[], lnZ=1.0, lnZ_err=0.2, re2=0.03, n_draws=100,
             p_raw=0.7, group="G1", group_members=[s["solution_id"]])  # fmt: skip
    txt = R.notes_markdown(s, {"arch_prob": {"1S2L": 1.0}, "missing": []})
    assert "**UNRELIABLE:**" in txt and "err_scale F213" in txt
    assert R.CONVENTIONS in txt
    assert "RADIANS" in txt and "1S2L: 1" in txt


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
