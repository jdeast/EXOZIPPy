"""
The automatic exposure node count (instrument.md, "Choosing ninterp"):

  - a file with exptime_s and no ninterp gets the SMALLEST n whose smearing
    error, as an aggregate Delta chi2 against its error bars at the start,
    is below exposure.AUTO_DCHI2 -- n meets it and n - 1 does not;
  - an exposure too short to matter keeps n = 1, bit-identical to no keys;
  - a user's ninterp is never changed, and is warned about when too coarse
    and when at least FINE_WARN_FACTOR times finer than needed;
  - the cap warns, naming the file;
  - after the polish an automatic n is RAISED when the polished model is
    sharper than the start (a deeper transit), and the model rebuilt with
    the start carried across.

The light curves are synthetic, on the hot-Jupiter geometry of
test_transit_exptime_ninterp.py (ingress ~11.5 min, T14 ~100 min).
"""

import logging
import re

import numpy as np
import pytest
from test_transit_exptime_ninterp import T14, TC, _config, _params, _write_lc

from exozippy.components import exposure as exposure_rules
from exozippy.system import System

LOGGER = "exozippy.components.instrument"


# ---------------------------------------------------------------------------
# The search, on its own (no model)
# ---------------------------------------------------------------------------


def _scaling(coeffs, calls=None):
    """An evaluator whose Delta chi2 falls as n^-4 (a midpoint rule's
    n^-2 error, squared), per file."""

    def evaluate(trial):
        if calls is not None:
            calls.append(dict(trial))
        return {f: coeffs[f] / n**4 for f, n in trial.items()}

    return evaluate


def test_search_returns_the_smallest_passing_count_for_every_file():
    """
    Given three files whose Delta chi2 is C / n^4 with different C,
    When the smallest count is searched from nothing known,
    Then each file's n passes and n - 1 fails, and the files share one
      evaluation per trial (the searches are batched).
    """
    coeffs = {0: 0.05, 1: 30.0, 2: 4000.0}
    calls = []
    found = exposure_rules.smallest_ninterp(
        _scaling(coeffs, calls), {f: 0 for f in coeffs}
    )
    thr = exposure_rules.AUTO_DCHI2
    for f, (n, d, capped, need) in found.items():
        assert need == n
        assert not capped
        assert d < thr
        assert d == pytest.approx(coeffs[f] / n**4)
        if n > 1:
            assert coeffs[f] / (n - 1) ** 4 >= thr
    assert found[0][0] == 1
    assert found[1][0] == 5  # 30/5^4 = 0.048, 30/4^4 = 0.117
    assert found[2][0] == 15  # 4000/15^4 = 0.079, 4000/14^4 = 0.104
    # One evaluation call per doubling or bisection step, all files at once.
    assert len(calls) < 10


def test_search_caps_and_says_so():
    """
    Given a file no count up to the cap brings under the threshold,
    When searched,
    Then it comes back at the cap, flagged capped.
    """
    found = exposure_rules.smallest_ninterp(_scaling({0: 1e9}), {0: 0}, cap=8)
    assert found[0][0] == 8
    assert found[0][2] is True
    # The count it needs is ESTIMATED from the convergence between 4 and 8
    # (C / n^4 < 0.1 first at n = 317), never evaluated there.
    assert found[0][3] == 317


def test_estimate_refuses_a_count_that_does_not_converge():
    assert exposure_rules.estimate_needed({4: 1.0, 8: 2.0}) is None
    assert exposure_rules.estimate_needed({8: 2.0}) is None


def test_search_from_a_known_bracket_only_bisects():
    """A user count known to pass (hi) is searched downward without
    doubling past it; one known to fail (lo) upward from it."""
    calls = []
    found = exposure_rules.smallest_ninterp(
        _scaling({0: 30.0}, calls), {0: 0}, {0: 16}
    )
    assert found[0][0] == 5
    assert all(c[0] < 16 for c in calls)
    found = exposure_rules.smallest_ninterp(_scaling({0: 30.0}), {0: 2})
    assert found[0][0] == 5


# ---------------------------------------------------------------------------
# On a built model (slow)
# ---------------------------------------------------------------------------


def _build(tmp_path, extra, err=2e-4, params=None):
    t = np.linspace(TC - 0.8 * T14, TC + 0.8 * T14, 300)
    lc = _write_lc(tmp_path / "lc.dat", t, err=np.full(t.size, err))
    system = System(_config([lc], [extra]), user_params=params or _params())
    system.prepare()
    model = system.build_model()
    return system, model


@pytest.fixture(scope="module")
def auto_system(tmp_path_factory):
    """A 30-minute cadence light curve with no ninterp: the count is
    chosen at the build start."""
    return _build(tmp_path_factory.mktemp("auto"), {"exptime_s": 1800.0})


@pytest.mark.slow
def test_auto_count_meets_the_threshold_and_one_fewer_does_not(auto_system):
    """
    Given a 30-min cadence light curve with 0.2 ppt errors and no ninterp,
    When the model is built,
    Then the chosen n is > 1, its Delta chi2 at the start is below
      AUTO_DCHI2, n - 1's is not, and the likelihood smears on n nodes.
    """
    system, model = auto_system
    tr = system.transit
    n = tr.ninterp[0]
    assert tr.ninterp_auto[0] and n > 1
    start = system.get_raw_start(model)
    below, at = tr.exposure_dchi2(system, model, start, [{0: n - 1}, {0: n}])
    assert at[0] < exposure_rules.AUTO_DCHI2 <= below[0]
    (rows, grid, weights) = tr._exposure_groups[0]
    assert grid.shape[1] == n and weights.size == n


@pytest.mark.slow
def test_auto_count_is_stamped_as_a_build_choice(auto_system):
    """The chosen count travels with the trace (build_choices), and a
    fresh System restoring it builds at that count without choosing."""
    system, _ = auto_system
    n = system.transit.ninterp[0]
    assert system.build_choices() == {"transit": {"ninterp": {"inst0": n}}}


@pytest.mark.slow
def test_polished_deeper_transit_raises_the_auto_count(auto_system, caplog):
    """
    Given the auto-chosen model and a "polished" point whose planet is
    twice as large (a transit four times deeper, so a sharper model),
    When the build choices are re-checked there,
    Then the automatic n is RAISED (a log line naming the file and the old
      and new counts), a rebuild is asked for, and the rebuilt model smears
      on the new count with the polished point carried across exactly.
    """
    system, model = auto_system
    tr = system.transit
    n0 = tr.ninterp[0]
    point = system.get_raw_start(model)
    rad = system.planet.radius
    phys = rad.phys_from_raw(point[f"{rad.label}_raw"])
    point[f"{rad.label}_raw"] = rad.raw_from_initval(phys * 2.0)
    with caplog.at_level(logging.INFO, logger=LOGGER):
        changed = system.check_at_point(model, point, "polished")
    assert changed
    n1 = tr.ninterp[0]
    assert n1 > n0
    assert any(
        f"raised from {n0} to {n1}" in r.getMessage()
        and "inst0" in r.getMessage()
        for r in caplog.records
    )
    new_model = system.rebuild_carrying_start(point)
    assert system.transit.ninterp[0] == n1
    assert system.transit._exposure_groups[0][1].shape[1] == n1
    new_start = system.get_raw_start(new_model)
    rad = system.planet.radius
    assert rad.phys_from_raw(new_start[f"{rad.label}_raw"])[
        0
    ] == pytest.approx(2.0 * phys[0], rel=1e-9)
    # At the posterior median an automatic count is only ever warned about.
    system.transit.ninterp[0] = n0
    assert not system.check_at_point(new_model, new_start, "median")
    assert system.transit.ninterp[0] == n0


@pytest.mark.slow
def test_negligible_exposure_is_one_node_and_bit_identical(tmp_path):
    """
    Given a 1-s exposure (nothing changes within it) and no ninterp,
    When built,
    Then n = 1 and the start logp is bit-identical to the same light curve
      with no exposure keys at all.
    """
    system, model = _build(tmp_path, {"exptime_s": 1.0})
    assert system.transit.ninterp == [1]
    bare, bare_model = _build(tmp_path, {})
    a = model.compile_logp()(system.get_raw_start(model))
    b = bare_model.compile_logp()(bare.get_raw_start(bare_model))
    assert float(a).hex() == float(b).hex()


@pytest.mark.slow
def test_user_count_too_coarse_is_kept_and_warned(tmp_path, caplog):
    """
    Given ninterp: 1 on a 30-min cadence (far too coarse),
    When built,
    Then the count stays 1 and a warning names the file, the value, the
      threshold and the count that would meet it.
    """
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        system, _ = _build(tmp_path, {"exptime_s": 1800.0, "ninterp": 1})
    assert system.transit.ninterp == [1]
    msgs = [
        r.getMessage()
        for r in caplog.records
        if "too coarse" in r.getMessage()
    ]
    assert len(msgs) == 1
    assert "inst0" in msgs[0] and "ninterp: 1" in msgs[0]
    assert str(exposure_rules.AUTO_DCHI2) in msgs[0]
    assert "would meet it" in msgs[0]


@pytest.mark.slow
def test_user_count_too_fine_is_kept_and_warned(tmp_path, caplog):
    """
    Given ninterp: 20 on a 2-min cadence (one node already suffices),
    When built,
    Then the count stays 20 and a warning names the file, the value, the
      smallest sufficient count and the runtime factor.  A count under
      FINE_WARN_FACTOR times the needed one is not warned about.
    """
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        system, _ = _build(tmp_path, {"exptime_s": 120.0, "ninterp": 20})
    assert system.transit.ninterp == [20]
    msgs = [
        r.getMessage()
        for r in caplog.records
        if "finer than" in r.getMessage()
    ]
    assert len(msgs) == 1
    assert "inst0" in msgs[0] and "ninterp: 20" in msgs[0]
    need = int(re.search(r"ninterp: (\d+) already", msgs[0]).group(1))
    assert 20 >= exposure_rules.FINE_WARN_FACTOR * need
    assert f"~{20 / need:.1f}x" in msgs[0]

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        system, _ = _build(tmp_path, {"exptime_s": 1800.0, "ninterp": 20})
    assert not [r for r in caplog.records if "finer than" in r.getMessage()]


@pytest.mark.slow
def test_cap_warns_naming_the_file(tmp_path, caplog, monkeypatch):
    """
    Given a cap below what the file needs,
    When the count is chosen,
    Then it is the cap and a warning names the file.
    """
    monkeypatch.setattr(exposure_rules, "AUTO_NINTERP_CAP", 2)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        system, _ = _build(tmp_path, {"exptime_s": 1800.0})
    assert system.transit.ninterp == [2]
    msgs = [
        r.getMessage() for r in caplog.records if "even 2" in r.getMessage()
    ]
    assert len(msgs) == 1 and "inst0" in msgs[0]
    # ... and names the count it needs, saying it is an estimate.
    assert "would need ~" in msgs[0] and "ESTIMATED" in msgs[0]


# ---------------------------------------------------------------------------
# check_exposure: false -- the user's choice, not checked
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_check_exposure_false_skips_the_user_check_entirely(
    tmp_path, caplog, monkeypatch
):
    """
    Given ninterp: 1 on a 30-min cadence (too coarse) with
    check_exposure: false,
    When built,
    Then no check is compiled, no too-coarse warning is logged, the count
      stays 1, and one INFO line says the check is off.
    """
    from exozippy.components.instrument import Instrument

    calls = []
    monkeypatch.setattr(
        Instrument,
        "_pairs_dchi2",
        lambda self, *a, **k: calls.append(1) or [],
    )
    with caplog.at_level(logging.INFO, logger=LOGGER):
        system, _ = _build(
            tmp_path,
            {"exptime_s": 1800.0, "ninterp": 1, "check_exposure": False},
        )
    assert system.transit.ninterp == [1]
    assert calls == []
    msgs = [r.getMessage() for r in caplog.records]
    assert not [m for m in msgs if "too coarse" in m]
    off = [m for m in msgs if "check_exposure: false" in m]
    assert len(off) == 1 and "inst0" in off[0] and "ninterp: 1" in off[0]


@pytest.mark.parametrize(
    "extra, match",
    [
        ({"ninterp": 1, "check_ninterp": False}, "check_exposure: false"),
        ({"ninterp": 1, "check_exposure": "no"}, "true or false"),
        ({"check_exposure": False}, "does nothing"),
    ],
    ids=["variant-spelling", "not-a-boolean", "nothing-to-switch-off"],
)
def test_check_exposure_has_one_spelling_and_one_meaning(
    tmp_path, extra, match
):
    t = np.linspace(TC - 0.8 * T14, TC + 0.8 * T14, 30)
    lc = _write_lc(tmp_path / "lc.dat", t)
    with pytest.raises(ValueError, match=match):
        System(
            _config([lc], [{"exptime_s": 1800.0, **extra}]),
            user_params=_params(),
        )
