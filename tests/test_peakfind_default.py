"""The built-in peak finder is THE microlensing seeder (review 8.6.25).

WHAT THESE GUARD, end to end through ``System.prepare()`` on the shipped
DC2018_128 example (one Z087 light curve, binary lens):

  * with starts the engine cannot derive, the peak finder seeds
    t_0/u_0/t_E;
  * the retired ``auto`` spelling of ``peak_find`` raises, naming the new
    ones;
  * the REMOVED ``mmexofast:`` / ``mmexofast_options:`` keys raise at the
    user boundary with the migration (JDE 2026-10-01: the MMEXOFAST
    hand-off is gone; its JSON's content belongs in the params file);
  * an informed t_0 is HELD and only the missing u_0/t_E are found and
    pushed; an informed t_E -- here one the kinematics derive -- is held
    too, never refit (8.6.25 part 2).

Each case runs prepare() on a scratch copy (prepare writes under the
prefix), so nothing in examples/ is touched.
"""

import shutil
from pathlib import Path

import pytest

EXAMPLE = Path(__file__).parent.parent / "examples" / "DC2018_128"
DATA = "n20180816.Z087.WFIRST18.128.txt"

pytestmark = pytest.mark.skipif(
    not (EXAMPLE / DATA).exists(), reason="DC2018_128 example data not present"
)

GEOMETRY = {"source.0.t_0", "source.0.u_0", "mulensevent.0.t_E"}


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


def test_no_starts_runs_the_peak_finder(tmp_path):
    """
    Given the DC2018_128 example with no params file,
    When prepare() runs,
    Then ONE seed set from the peak finder holds t_0/u_0/t_E near the
    published light-curve solution.
    """
    cm = _prepare(tmp_path)

    assert len(cm.seed_hint_sets) == 1
    assert set(cm.seed_hint_sets[0]) == GEOMETRY
    assert "peak finder" in cm.seed_hint_source
    # 8.4.9: the finder reproduces the published PSPL on event 128.
    assert cm.seed_start_value("source.0.t_0") == pytest.approx(
        2458554.82, abs=0.1
    )
    assert cm.seed_start_value("mulensevent.0.t_E") == pytest.approx(
        19.2, rel=0.1
    )


@pytest.mark.parametrize(
    "keys",
    [
        {"mmexofast": "fits.json"},
        {"mmexofast": True},
        {"mmexofast": False},
        {"mmexofast_options": {"no_parallax": False}},
    ],
)
def test_the_removed_mmexofast_keys_raise_with_the_migration(tmp_path, keys):
    """
    Given a config that still names a removed MMEXOFAST key -- with ANY
    value, `false` included,
    When prepare() runs,
    Then it raises at the boundary, naming the key and saying where its
    content goes now (params-file initvals, the instrument mask:, err_scale)
    and which script converts an existing JSON.  Silently ignoring it would
    start a fit the user configured around a JSON from a different place.
    """
    with pytest.raises(ValueError) as exc:
        _prepare(tmp_path, **keys)
    msg = str(exc.value)
    assert repr([next(iter(keys))]) in msg
    assert "removed" in msg
    assert "initval" in msg and "mask:" in msg and "err_scale" in msg
    assert "convert_mmexofast_json.py" in msg


@pytest.mark.parametrize(
    "keys, match",
    [
        ({"peak_find": "auto"}, "omit the key"),
        ({"peak_find": "yes please"}, "not a spelling"),
    ],
)
def test_retired_and_unknown_spellings_raise(tmp_path, keys, match):
    with pytest.raises(ValueError, match=match):
        _prepare(tmp_path, **keys)


def test_an_informed_t_0_is_held_and_only_u_0_t_E_are_found(tmp_path):
    """
    Given a params file that names t_0 only,
    When prepare() runs,
    Then the finder pushes u_0 and t_E (and nothing else), fit around the
    user's t_0, which keeps its user value.  It used to skip the finder and
    leave u_0/t_E at defaults.yaml (8.6.25 part 2).
    """
    t_0 = 2458554.9
    cm = _prepare(tmp_path, {"source.Source.t_0": {"initval": t_0}})

    assert len(cm.seed_hint_sets) == 1
    assert set(cm.seed_hint_sets[0]) == {"source.0.u_0", "mulensevent.0.t_E"}
    assert cm.seed_start_value("source.0.u_0") == pytest.approx(0.14, rel=0.2)
    assert cm.seed_start_value("mulensevent.0.t_E") == pytest.approx(
        19.0, rel=0.15
    )
    probed = cm.probe_start(["source.0.t_0"])["source.0.t_0"]
    assert probed.user_value == pytest.approx(t_0)
    assert probed.source == "user"


def test_a_t_E_the_kinematics_derive_is_not_overridden(tmp_path):
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


def test_a_user_t_E_without_t_0_is_held_and_t_0_u_0_are_found(tmp_path):
    """
    Given a params file that names t_E only (JDE 2026-10-01: "when the user
    supplies [t_E], it should respect it"),
    When prepare() runs,
    Then the finder pushes t_0 and u_0 only, and t_E keeps the user value.
    """
    t_E = 17.0
    cm = _prepare(tmp_path, {"mulensevent.t_E": {"initval": t_E}})

    assert len(cm.seed_hint_sets) == 1
    assert set(cm.seed_hint_sets[0]) == {"source.0.t_0", "source.0.u_0"}
    assert cm.seed_start_value("source.0.t_0") == pytest.approx(
        2458554.82, abs=0.3
    )
    probed = cm.probe_start(["mulensevent.0.t_E"])["mulensevent.0.t_E"]
    assert probed.user_value == pytest.approx(t_E)
    assert probed.source == "user"


def test_all_three_given_runs_nothing(tmp_path):
    cm = _prepare(
        tmp_path,
        {
            "source.Source.t_0": {"initval": 2458554.9},
            "source.Source.u_0": {"initval": 0.14},
            "mulensevent.t_E": {"initval": 18.2},
        },
    )
    assert cm.seed_hint_sets == []
