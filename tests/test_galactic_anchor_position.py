"""GalacticModel refuses an anchor star nobody positioned.

The prior is one function of one line of sight -- the anchor star's -- and
a star left at defaults.yaml's ra/dec (180, 0) points at Galactic (l 276,
b +60).  The 2026-09 DC2018 sweep ran every event that way: its configs
pinned the Lens's coordinates and anchored the galactic model on the Source,
so the density fell exponentially with distance along a sight line 60 deg
out of the plane and every source came back at half its distance and
radius.  Nobody wants a Galactic prior without saying where it is, so
building one is an error.

Built on the shipped DC2018_128 example because it exercises the real
provenance ledger: the refusal keys on ConfigManager.initval_source saying
"default", not on the numerical value, so a user who really means (180, 0)
is not refused and a mock Parameter with no ledger is left alone.
"""

import copy
import pathlib

import pytest
import yaml

from exozippy.system import System

EXAMPLE_DIR = pathlib.Path(__file__).parent / ".." / "examples" / "DC2018_128"

pytestmark = pytest.mark.slow


def _inputs():
    with open(EXAMPLE_DIR / "DC2018_128.yaml") as f:
        config = yaml.safe_load(f)
    with open(EXAMPLE_DIR / "DC2018_128.params.yaml") as f:
        user_params = yaml.safe_load(f)
    for k in ("run", "prefix", "parameter_file", "sampler"):
        config.pop(k, None)
    return config, user_params


def _anchor_name(config):
    idx = config["galacticmodel"][0].get("anchor_idx", 0)
    return config["star"][idx]["name"]


def test_shipped_example_positions_its_anchor(monkeypatch):
    monkeypatch.chdir(EXAMPLE_DIR)
    config, user_params = _inputs()
    anchor = _anchor_name(config)
    assert f"star.{anchor}.ra" in user_params
    assert f"star.{anchor}.dec" in user_params
    system = System(
        copy.deepcopy(config), user_params=copy.deepcopy(user_params)
    )
    system.prepare()
    system.build_model()  # must not raise


def test_unpositioned_anchor_is_refused(monkeypatch):
    monkeypatch.chdir(EXAMPLE_DIR)
    config, user_params = _inputs()
    anchor = _anchor_name(config)
    user_params = copy.deepcopy(user_params)
    del user_params[f"star.{anchor}.ra"]
    del user_params[f"star.{anchor}.dec"]
    system = System(copy.deepcopy(config), user_params=user_params)
    system.prepare()
    with pytest.raises(ValueError, match=r"has no sky position") as exc:
        system.build_model()
    msg = str(exc.value)
    assert f"star.{anchor}.ra" in msg and f"star.{anchor}.dec" in msg
    assert "anchor_idx" in msg


def test_one_missing_coordinate_is_enough_to_refuse(monkeypatch):
    monkeypatch.chdir(EXAMPLE_DIR)
    config, user_params = _inputs()
    anchor = _anchor_name(config)
    user_params = copy.deepcopy(user_params)
    del user_params[f"star.{anchor}.dec"]
    system = System(copy.deepcopy(config), user_params=user_params)
    system.prepare()
    with pytest.raises(ValueError, match=r"has no sky position"):
        system.build_model()
