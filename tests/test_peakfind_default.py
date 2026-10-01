"""The built-in peak finder is the DEFAULT microlensing seeder (review 8.6.25).

WHAT THESE GUARD, end to end through ``System.prepare()`` on the shipped
DC2018_128 example (one Z087 light curve, binary lens):

  * with no ``mmexofast:`` key and starts the engine cannot derive, the peak
    finder seeds t_0/u_0/t_E and the mmexofast package is never touched
    (MMEXOFAST runs only when asked for: ``mmexofast: true`` or a JSON path);
  * the retired ``auto`` spellings of both keys raise, naming the new ones;
  * a user-named JSON that does not exist raises (review 1.6.15);
  * an informed t_0 is HELD and only the missing u_0/t_E are found and
    pushed; an informed t_E -- here one the kinematics derive -- is held
    too, never refit (8.6.25 part 2).

Each case runs prepare() on a scratch copy (prepare writes under the
prefix), so nothing in examples/ is touched.
"""

import json
import shutil
import sys
from pathlib import Path

import pytest

from exozippy.components.mulensing import mmexofast_support

EXAMPLE = Path(__file__).parent.parent / "examples" / "DC2018_128"
MMX_JSON = EXAMPLE / "mmexofast.json"
DATA = "n20180816.Z087.WFIRST18.128.txt"

pytestmark = pytest.mark.skipif(
    not (EXAMPLE / DATA).exists(), reason="DC2018_128 example data not present"
)

GEOMETRY = {"source.0.t_0", "source.0.u_0", "mulensevent.0.t_E"}


@pytest.fixture
def no_mmexofast(monkeypatch):
    """Make the lazy mmexofast import impossible and record any attempt to
    reach it, so a test can assert MMEXOFAST was never asked for."""
    calls = []
    real = mmexofast_support.run_or_load

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return real(*args, **kwargs)

    monkeypatch.setitem(sys.modules, "mmexofast", None)
    monkeypatch.setattr(mmexofast_support, "run_or_load", spy)
    return calls


def _prepare(tmp_path, user_params=None, **event_keys):
    from exozippy.system import System
    from exozippy.yamlio import load_system_config

    shutil.copy(EXAMPLE / "DC2018_128.yaml", tmp_path / "DC2018_128.yaml")
    (tmp_path / DATA).symlink_to(EXAMPLE / DATA)
    config = load_system_config(str(tmp_path / "DC2018_128.yaml"))
    config.pop("parameter_file", None)
    config["prefix"] = str(tmp_path / "fitresults" / "DC2018_128")
    config["mulensinstrument"][0]["file"] = str(tmp_path / DATA)
    config["mulensevent"][0].update(event_keys)
    system = System(config, user_params=dict(user_params or {}))
    system.prepare()
    return system.config_manager


def test_absent_key_with_no_starts_runs_the_peak_finder_not_mmexofast(
    tmp_path, no_mmexofast
):
    """
    Given the DC2018_128 example with no params file and no `mmexofast:`
    key (which used to run MMEXOFAST automatically),
    When prepare() runs,
    Then ONE seed set from the peak finder holds t_0/u_0/t_E near the
    MMEXOFAST solution, and the mmexofast package was never reached.
    """
    cm = _prepare(tmp_path)

    assert no_mmexofast == []
    assert len(cm.seed_hint_sets) == 1
    assert set(cm.seed_hint_sets[0]) == GEOMETRY
    assert "peak finder" in cm.seed_hint_source
    # 8.4.9: the finder reproduces MMEXOFAST's PSPL on event 128.
    assert cm.seed_start_value("source.0.t_0") == pytest.approx(
        2458554.82, abs=0.1
    )
    assert cm.seed_start_value("mulensevent.0.t_E") == pytest.approx(
        19.2, rel=0.1
    )


def test_mmexofast_true_is_the_explicit_opt_in(tmp_path, monkeypatch):
    """
    Given `mmexofast: true` and no starts,
    When prepare() runs,
    Then MMEXOFAST's run_or_load is called (mocked: it returns the shipped
    JSON) and its two solutions are the seed sets -- the peak finder, one
    seeder per fit, stays out.
    """
    with open(MMX_JSON) as f:
        data = json.load(f)
    calls = []

    def fake_run_or_load(json_path, files, **kwargs):
        calls.append(json_path)
        return data

    monkeypatch.setattr(mmexofast_support, "run_or_load", fake_run_or_load)
    cm = _prepare(tmp_path, mmexofast=True)

    assert len(calls) == 1
    assert calls[0].endswith("DC2018_128_mmexofast.json")
    assert len(cm.seed_hint_sets) == len(data["fits"]) == 2
    assert all("lens.1.q" in s for s in cm.seed_hint_sets)
    assert cm.seed_hint_source.startswith("MMEXOFAST")


@pytest.mark.parametrize(
    "keys, match",
    [
        ({"mmexofast": "auto"}, "mmexofast: true"),
        ({"peak_find": "auto"}, "omit the key"),
        ({"peak_find": "yes please"}, "not a spelling"),
    ],
)
def test_retired_and_unknown_spellings_raise(tmp_path, keys, match):
    with pytest.raises(ValueError, match=match):
        _prepare(tmp_path, **keys)


def test_a_missing_user_named_json_raises_naming_the_path(tmp_path):
    """Review 1.6.15: a typo in an explicit path used to warn and run
    unseeded."""
    missing = tmp_path / "no_such_mmexofast.json"
    with pytest.raises(FileNotFoundError, match="no_such_mmexofast.json"):
        _prepare(tmp_path, mmexofast=str(missing))


def test_an_informed_t_0_is_held_and_only_u_0_t_E_are_found(
    tmp_path, no_mmexofast
):
    """
    Given a params file that names t_0 only,
    When prepare() runs,
    Then the finder pushes u_0 and t_E (and nothing else), fit around the
    user's t_0, which keeps its user value.  It used to skip the finder and
    leave u_0/t_E at defaults.yaml (8.6.25 part 2).
    """
    t_0 = 2458554.9
    cm = _prepare(tmp_path, {"source.Source.t_0": {"initval": t_0}})

    assert no_mmexofast == []
    assert len(cm.seed_hint_sets) == 1
    assert set(cm.seed_hint_sets[0]) == {"source.0.u_0", "mulensevent.0.t_E"}
    assert cm.seed_start_value("source.0.u_0") == pytest.approx(0.14, rel=0.2)
    assert cm.seed_start_value("mulensevent.0.t_E") == pytest.approx(
        19.0, rel=0.15
    )
    probed = cm.probe_start(["source.0.t_0"])["source.0.t_0"]
    assert probed.user_value == pytest.approx(t_0)
    assert probed.source == "user"


def test_a_t_E_the_kinematics_derive_is_not_overridden(tmp_path, no_mmexofast):
    """
    Given t_0 plus theta_E and mu_rel (so the engine DERIVES t_E),
    When prepare() runs,
    Then only u_0 is pushed: the kinematic t_E is held, never replaced by a
    PSPL one -- the reason the old t_0-only gate existed.
    """
    cm = _prepare(
        tmp_path,
        {
            "source.Source.t_0": {"initval": 2458554.9},
            "mulensevent.theta_E": {"initval": 0.5},
            "mulensevent.mu_rel_mag": {"initval": 10.0},
        },
    )
    assert len(cm.seed_hint_sets) == 1
    assert set(cm.seed_hint_sets[0]) == {"source.0.u_0"}


def test_a_user_t_E_without_t_0_is_held_and_t_0_u_0_are_found(
    tmp_path, no_mmexofast
):
    """
    Given a params file that names t_E only (JDE 2026-10-01: "when the user
    supplies [t_E], it should respect it"),
    When prepare() runs,
    Then the finder pushes t_0 and u_0 only, and t_E keeps the user value.
    """
    t_E = 17.0
    cm = _prepare(tmp_path, {"mulensevent.t_E": {"initval": t_E}})

    assert no_mmexofast == []
    assert len(cm.seed_hint_sets) == 1
    assert set(cm.seed_hint_sets[0]) == {"source.0.t_0", "source.0.u_0"}
    assert cm.seed_start_value("source.0.t_0") == pytest.approx(
        2458554.82, abs=0.3
    )
    probed = cm.probe_start(["mulensevent.0.t_E"])["mulensevent.0.t_E"]
    assert probed.user_value == pytest.approx(t_E)
    assert probed.source == "user"


def test_all_three_given_runs_nothing(tmp_path, no_mmexofast):
    cm = _prepare(
        tmp_path,
        {
            "source.Source.t_0": {"initval": 2458554.9},
            "source.Source.u_0": {"initval": 0.14},
            "mulensevent.t_E": {"initval": 18.2},
        },
    )
    assert no_mmexofast == []
    assert cm.seed_hint_sets == []
