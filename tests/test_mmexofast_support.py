"""Tests for components/mulensing/mmexofast_support.py (auto-init layer).

Covers the pieces added for the MMEXOFAST data-driven-hints integration:
start-value sufficiency detection, jd_offset handling in seed extraction,
excluded_points -> Instrument mask translation, and errfacs -> err_scale
seeding. The actual MMEXOFAST run (run_or_load's fitter branch) is exercised
by the DC2018 example workflow, not here -- it needs the optional package
and minutes of CPU.
"""

import numpy as np
import pytest

from exozippy.components.mulensing import mmexofast_support as mmx


class _RecordingConfigManager:
    def __init__(self):
        self.hints = {}
        self.hint_ranks = {}
        self.scale_hints = {}
        self.seed_hint_sets = []

    def add_hint(self, path, value, rank=None):
        self.hints[path] = value
        self.hint_ranks[path] = rank

    def add_scale_hint(self, path, scale):
        self.scale_hints[path] = scale

    def add_seed_hints(self, seed_dicts, rank=None):
        self.seed_hint_sets = seed_dicts


# ---------------------------------------------------------------------------
# user_hints_sufficient
# ---------------------------------------------------------------------------


# The check now asks the relaxation engine whether each observable can be
# DERIVED, not whether the params file happens to name it, so these tests
# drive a real ConfigManager over a real lens topology.
_PSPL_CONFIG = {
    "star": [{"name": "Lens"}, {"name": "Source"}],
    "lens": [{"name": "Lens", "lenses": ["star.0"], "sources": ["star.1"]}],
}
_BINARY_CONFIG = {
    "star": [{"name": "Lens"}, {"name": "Source"}],
    "planet": [{"name": "b"}],
    "lens": [
        {
            "name": "Lens",
            "lenses": ["star.0", "planet.0"],
            "sources": ["star.1"],
        }
    ],
}


def _cm(params, config=None):
    from exozippy.config import ConfigManager

    return ConfigManager(params, system_config=config or _PSPL_CONFIG)


def _full_pspl_params():
    return {
        "lens.Lens.t_0": {"initval": 2458554.9},
        "lens.Lens.u_0": {"initval": 0.14},
        "lens.Lens.t_E": {"initval": 18.2},
    }


def _binary_geometry():
    return {
        "lens.Lens.alpha": {"initval": -52.0},
        "lens.Lens.q": {"initval": 1.1e-3},
        "lens.Lens.s": {"initval": 0.98},
    }


def test_sufficiency_pspl_complete():
    """
    Given user initvals for t_0/u_0/t_E on a point lens without finite source,
    When sufficiency is checked,
    Then the hints are sufficient (no MMEXOFAST run needed).
    """
    assert mmx.user_hints_sufficient(
        _cm(_full_pspl_params()), is_binary=False, want_rho=False
    )


def test_sufficiency_missing_t_E_is_insufficient():
    """
    Given user initvals lacking t_E and nothing to derive it from,
    When sufficiency is checked,
    Then the hints are insufficient.
    """
    params = _full_pspl_params()
    del params["lens.Lens.t_E"]
    assert not mmx.user_hints_sufficient(
        _cm(params), is_binary=False, want_rho=False
    )


def test_sufficiency_binary_needs_geometry():
    """
    Given a binary lens whose params cover only the PSPL trio,
    When sufficiency is checked,
    Then the hints are insufficient until s, alpha and q appear.
    """
    params = _full_pspl_params()
    assert not mmx.user_hints_sufficient(
        _cm(params, _BINARY_CONFIG), is_binary=True, want_rho=False
    )
    params.update(_binary_geometry())
    assert mmx.user_hints_sufficient(
        _cm(params, _BINARY_CONFIG), is_binary=True, want_rho=False
    )


def test_sufficiency_accepts_log_s_for_s():
    """
    Given a binary lens seeded with log_s instead of s,
    When sufficiency is checked,
    Then log_s satisfies the separation requirement.
    """
    params = _full_pspl_params()
    params.update(_binary_geometry())
    params["lens.Lens.log_s"] = params.pop("lens.Lens.s")
    params["lens.Lens.log_s"] = {"initval": -0.01}
    assert mmx.user_hints_sufficient(
        _cm(params, _BINARY_CONFIG), is_binary=True, want_rho=False
    )


def test_sufficiency_bounds_only_entry_does_not_count():
    """
    Given a t_E entry carrying only bounds (no initval or mu),
    When sufficiency is checked,
    Then the entry does not count as a start value.
    """
    params = _full_pspl_params()
    params["lens.Lens.t_E"] = {"lower": 1.0, "upper": 100.0}
    assert not mmx.user_hints_sufficient(
        _cm(params), is_binary=False, want_rho=False
    )


def test_sufficiency_finite_source_needs_rho():
    """
    Given finite_source is on and rho has no start value,
    When sufficiency is checked,
    Then the hints are insufficient.
    """
    assert not mmx.user_hints_sufficient(
        _cm(_full_pspl_params()), is_binary=False, want_rho=True
    )


def test_sufficiency_accepts_a_derived_q_from_body_masses():
    """
    Given a binary lens whose params name the BODY MASSES rather than q --
    the shape every mkparam restart file has, since lens.q is derived and so
    is never written --
    When sufficiency is checked,
    Then q is recognized as derivable and MMEXOFAST is not re-run.

    This is the regression: the old literal-key scan called a complete
    restart file insufficient, re-ran MMEXOFAST on every second-iteration
    fit, and then died inside it with "Parameter q has to be larger than 0".
    """
    params = _full_pspl_params()
    params.update(_binary_geometry())
    del params["lens.Lens.q"]
    params["planet.b.mass"] = {"initval": 0.35}
    params["star.Lens.logmass"] = {"initval": -0.4}

    assert mmx.user_hints_sufficient(
        _cm(params, _BINARY_CONFIG), is_binary=True, want_rho=False
    )


def test_literal_params_skip_the_engine_probe():
    """
    Given a params file that names every required observable outright,
    When sufficiency is checked,
    Then the answer comes from the cheap scan and the relaxation engine is
    never run -- naming a value makes it RANK_USER, so the probe could not
    change the answer, and every ordinary mulens fit would otherwise pay for
    an extra solve.
    """
    params = _full_pspl_params()
    params.update(_binary_geometry())
    cm = _cm(params, _BINARY_CONFIG)

    calls = []
    real = cm.probe_derivable
    cm.probe_derivable = lambda *a, **k: (calls.append(1), real(*a, **k))[1]

    assert mmx.user_hints_sufficient(cm, is_binary=True, want_rho=False)
    assert calls == [], "the probe ran despite every observable being named"


def test_probe_derivable_leaves_no_trace():
    """
    Given a ConfigManager,
    When the derivability probe runs,
    Then user_params, diagnostics and the export snapshots are unchanged --
    the probe must not pre-empt the real solve at stage 3.
    """
    import copy

    cm = _cm(_full_pspl_params(), _BINARY_CONFIG)
    before = (
        copy.deepcopy(cm.user_params),
        list(cm.diagnostics),
        dict(cm._last_resolved),
    )

    cm.probe_derivable(["lens.0.t_E"])

    assert cm.user_params == before[0]
    assert cm.diagnostics == before[1]
    assert cm._last_resolved == before[2]


# ---------------------------------------------------------------------------
# push_seed_hints: jd_offset
# ---------------------------------------------------------------------------


def test_seed_hints_subtract_jd_offset_from_t_0_only():
    """
    Given a JSON whose epochs carry jd_offset = 2450000 (full-JD convention),
    When seeds are pushed,
    Then t_0 is shifted back into the data's own time system while the
    duration t_E is untouched.
    """
    data = {
        "fits": [
            {
                "parameters": {"t_0": 2458554.9, "u_0": 0.14, "t_E": 18.2},
                "sigmas": {"t_0": 0.02, "u_0": 0.004, "t_E": 0.3},
            }
        ],
        "jd_offset": 2450000.0,
    }
    cm = _RecordingConfigManager()
    n = mmx.push_seed_hints(data, cm, want_rho=False, is_binary=False)
    assert n == 1
    seed = cm.seed_hint_sets[0]
    assert np.isclose(seed["lens.0.t_0"], 8554.9)
    assert np.isclose(seed["lens.0.t_E"], 18.2)


def test_seed_hints_no_jd_offset_key_is_zero_shift():
    """
    Given an older JSON without the jd_offset key,
    When seeds are pushed,
    Then t_0 passes through unshifted (backward compatibility).
    """
    data = {
        "fits": [{"parameters": {"t_0": 2458554.9, "u_0": 0.14, "t_E": 18.2}}]
    }
    cm = _RecordingConfigManager()
    mmx.push_seed_hints(data, cm, want_rho=False, is_binary=False)
    assert np.isclose(cm.seed_hint_sets[0]["lens.0.t_0"], 2458554.9)


# ---------------------------------------------------------------------------
# apply_excluded_points
# ---------------------------------------------------------------------------

FILES = ["data/n1.W149.WFIRST18.128.txt", "data/n1.Z087.WFIRST18.128.txt"]


def test_excluded_points_fill_matching_files_by_basename():
    """
    Given excluded_points keyed by file basename,
    When applied to a component whose files carry directory prefixes,
    Then the matching file's mask spec becomes the index list and the other
    file stays unmasked.
    """
    data = {
        "excluded_points": {
            "n1.W149.WFIRST18.128.txt": {
                "n_data": 100,
                "indices": [3, 17, 99],
                "times": [1.0, 2.0, 3.0],
            }
        }
    }
    specs = mmx.apply_excluded_points(data, FILES, [None, None], "mulens")
    assert specs[0] == [3, 17, 99]
    assert specs[1] is None


def test_excluded_points_respect_existing_user_mask():
    """
    Given a file that already has a user mask spec,
    When excluded_points name the same file,
    Then the user's mask wins and the mmexofast indices are ignored.
    """
    data = {
        "excluded_points": {
            "n1.W149.WFIRST18.128.txt": {"indices": [3], "times": [1.0]}
        }
    }
    specs = mmx.apply_excluded_points(data, FILES, [[7], None], "mulens")
    assert specs[0] == [7]


def test_excluded_points_unknown_label_warns_and_skips(caplog):
    """
    Given an excluded_points label matching none of the component's files,
    When applied,
    Then a warning is logged and no spec changes.
    """
    data = {"excluded_points": {"other.txt": {"indices": [1], "times": []}}}
    with caplog.at_level("WARNING"):
        specs = mmx.apply_excluded_points(data, FILES, [None, None], "mulens")
    assert specs == [None, None]
    assert any("other.txt" in rec.message for rec in caplog.records)


def test_excluded_points_skip_files_with_a_robust_likelihood(caplog):
    """
    Given one file that opted into the hogg mixture and one that did not,
    When excluded_points name both,
    Then the robust file's points stay unmasked (the mixture supersedes the
    frozen hard mask, with a log line saying so) while the plain file's mask
    fills as before.
    """
    data = {
        "excluded_points": {
            "n1.W149.WFIRST18.128.txt": {"indices": [3, 17], "times": []},
            "n1.Z087.WFIRST18.128.txt": {"indices": [5], "times": []},
        }
    }
    with caplog.at_level("INFO"):
        specs = mmx.apply_excluded_points(
            data,
            FILES,
            [None, None],
            "mulens",
            robust_kinds=["hogg", ""],
        )
    assert specs[0] is None  # hogg file: left unmasked
    assert specs[1] == [5]  # plain file: masked as before
    assert any(
        "likelihood: hogg" in rec.message and "unmasked" in rec.message
        for rec in caplog.records
    )


def test_excluded_points_empty_indices_leave_spec_none():
    """
    Given a dataset entry whose index list is empty (nothing excluded),
    When applied,
    Then the file's spec stays None (no pointless mask machinery).
    """
    data = {
        "excluded_points": {
            "n1.W149.WFIRST18.128.txt": {"indices": [], "times": []}
        }
    }
    specs = mmx.apply_excluded_points(data, FILES, [None, None], "mulens")
    assert specs == [None, None]


# ---------------------------------------------------------------------------
# push_errfac_hints
# ---------------------------------------------------------------------------


def test_errfacs_seed_err_scale_by_element_index():
    """
    Given errfacs for one of two files,
    When pushed,
    Then only that element's err_scale initval hint appears, at
    RANK_DERIVED_DATA so a user override still wins.
    """
    from exozippy.config import RANK_DERIVED_DATA

    data = {"errfacs": {"n1.Z087.WFIRST18.128.txt": 1.062}}
    cm = _RecordingConfigManager()
    mmx.push_errfac_hints(data, FILES, "mulensinstrument", cm)
    assert cm.hints == {"mulensinstrument.1.err_scale": 1.062}
    assert cm.hint_ranks["mulensinstrument.1.err_scale"] == RANK_DERIVED_DATA


def test_errfacs_nonpositive_factor_is_skipped():
    """
    Given a non-finite or non-positive error factor,
    When pushed,
    Then it is skipped rather than seeding a nonsensical err_scale.
    """
    data = {
        "errfacs": {
            "n1.W149.WFIRST18.128.txt": 0.0,
            "n1.Z087.WFIRST18.128.txt": float("nan"),
        }
    }
    cm = _RecordingConfigManager()
    mmx.push_errfac_hints(data, FILES, "mulensinstrument", cm)
    assert cm.hints == {}


# ---------------------------------------------------------------------------
# run_or_load: cache behavior (no mmexofast package involved)
# ---------------------------------------------------------------------------


def test_run_or_load_uses_cached_json(tmp_path):
    """
    Given an existing cached JSON at the target path,
    When run_or_load is called,
    Then the cache is returned without importing or running mmexofast.
    """
    cached = tmp_path / "event_mmexofast.json"
    cached.write_text('{"fits": [], "errfacs": {}}')
    data = mmx.run_or_load(cached, ["a.txt"])
    assert data == {"fits": [], "errfacs": {}}


# ---------------------------------------------------------------------------
# ConfigManager.seed_start_value: user-unit view of the seed hints
# ---------------------------------------------------------------------------


def test_seed_start_value_returns_user_units():
    """
    Given seed hints pushed through the real ConfigManager (which stores
    internal units -- alpha in radians),
    When seed_start_value reads them back,
    Then the values come back in USER units (alpha in degrees), matching what
    a raw user_params entry would hold, so stage-1 consumers (the mulens flux
    bootstrap) can use either source interchangeably.
    """
    from exozippy.config import ConfigManager

    cm = ConfigManager({}, system_config={"lens": [{"name": "Lens"}]})
    cm.add_seed_hints(
        [
            {
                "lens.0.t_0": 2458554.82,
                "lens.0.u_0": 0.131,
                "lens.0.t_E": 19.16,
                "lens.0.log_s": -0.066,
                "lens.0.alpha": -50.37,
                "lens.0.q": 9.26e-4,
            },
            {"lens.0.alpha": -50.47},
        ]
    )

    stored = cm.seed_hint_sets[0]["lens.0.alpha"]
    assert np.isclose(stored, np.deg2rad(-50.37))  # internal storage: rad

    assert np.isclose(cm.seed_start_value("lens.0.alpha"), -50.37)
    assert np.isclose(cm.seed_start_value("lens.0.t_0"), 2458554.82)
    assert np.isclose(cm.seed_start_value("lens.0.log_s"), -0.066)
    assert np.isclose(cm.seed_start_value("lens.0.alpha", seed=1), -50.47)
    assert cm.seed_start_value("lens.0.rho") is None  # never pushed
    assert cm.seed_start_value("lens.0.t_0", seed=5) is None  # no such seed
