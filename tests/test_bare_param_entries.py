# tests/test_bare_param_entries.py
"""A bare params value is translated to ``{initval: ...}`` at the boundary.

A params file may write a parameter as a dict of fields, a bare number
(``star.0.teff: 5800``) or a bare per-seed list (``star.0.teff: [5000,
6000]``).  The two bare forms mean exactly ``{initval: <value>}``.  Before
review 1.1.7 the bare shape survived ``standardize_param_names`` and each
downstream reader re-guessed it; ``_build_seed_overrides`` skipped any
non-dict entry, so a bare per-seed list silently kept only seed 0 -- no
warning, and ``resolve()`` took element 0 so the run looked normal.

Now ``ConfigManager.user_params`` holds exactly one entry shape, readers go
through ``config.user_entry``, and a non-dict entry that reaches one RAISES,
naming the key.

Tests follow AAA (Arrange / Act / Assert) with Given/When/Then docstrings.
"""

import copy
import io
import logging
from pathlib import Path

import numpy as np
import pytest
import yaml
from ruamel.yaml import YAML

from exozippy.config import ConfigManager, user_entry
from exozippy.system import System

EXAMPLE_DIR = Path(__file__).parent.parent / "examples" / "kelt4"


def _kelt4_rvonly_config():
    with open(EXAMPLE_DIR / "kelt4_rvonly.yaml") as f:
        return yaml.safe_load(f)


def _prepared(user_params, monkeypatch):
    monkeypatch.chdir(EXAMPLE_DIR)
    system = System(_kelt4_rvonly_config(), user_params=user_params)
    system.prepare()
    return system


# ---------------------------------------------------------------------------
# The reproduction: a bare per-seed list
# ---------------------------------------------------------------------------


def test_a_bare_per_seed_list_solves_one_start_per_seed(monkeypatch):
    """
    Given kelt4_rvonly with the documented bare per-seed list
      `star.0.teff: [5000, 6000]`,
    When System.prepare() runs the relaxation engine,
    Then it solves K = 2 seeds carrying 5000 and 6000 -- exactly what the
    `{initval: [5000, 6000]}` spelling gives.

    On master the bare list gave K = None (one seed, at 5000) with no
    warning: _build_seed_overrides skipped every non-dict entry.
    """
    # Arrange / Act
    bare = _prepared({"star.0.teff": [5000, 6000]}, monkeypatch)
    spelled = _prepared(
        {"star.0.teff": {"initval": [5000, 6000]}}, monkeypatch
    )

    # Assert
    seeds = bare.config_manager.seed_resolved
    assert seeds is not None and len(seeds) == 2
    assert [s["star.0.teff"] for s in seeds] == [5000.0, 6000.0]
    assert [s["star.0.teff"] for s in seeds] == [
        s["star.0.teff"] for s in spelled.config_manager.seed_resolved
    ]


def test_a_bare_scalar_resolves_exactly_as_its_dict_spelling(monkeypatch):
    """
    Given kelt4_rvonly with `star.0.teff: 5800` and, separately, with
    `star.0.teff: {initval: 5800}` (5800, not the 5778 default, so a dropped
    value could not hide behind the backstop),
    When both are prepared,
    Then resolve(), initval_source, the provenance ledger and the stored
    entry agree field for field.
    """
    # Arrange / Act
    bare = _prepared({"star.0.teff": 5800}, monkeypatch).config_manager
    spelled = _prepared(
        {"star.0.teff": {"initval": 5800}}, monkeypatch
    ).config_manager

    # Assert
    rb = bare.resolve("star", "teff", element=0)
    rs = spelled.resolve("star", "teff", element=0)
    assert rb["initval"][0] == 5800.0
    assert rb.keys() == rs.keys()
    for key in rb:
        if isinstance(rb[key], np.ndarray):
            np.testing.assert_array_equal(rb[key], rs[key])
        else:
            assert rb[key] == rs[key], key
    assert rb["user_modified"] is True
    assert rb["user_prior_modified"] is False
    assert bare.initval_source("star", "teff", element=0) == "user"
    assert (
        bare._last_provenance["star.0.teff"]
        == (spelled._last_provenance["star.0.teff"])
    )
    assert (
        bare.user_params["star.0.teff"] == spelled.user_params["star.0.teff"]
    )


# ---------------------------------------------------------------------------
# The boundary: every pass emits one entry shape
# ---------------------------------------------------------------------------


def test_every_standardizer_pass_emits_a_field_dict():
    """
    Given bare values in every spelling the standardizer handles -- a 3-part
    name key, a 3-part index key, a 2-part list-component broadcast, a
    broadcast under a bare specific entry, and a flat-dict component's 2-
    and 3-part keys,
    When a ConfigManager is built,
    Then every user_params entry is a field dict, each broadcast copy is its
    own object, and the specific bare entry still beats the broadcast.
    """
    # Arrange
    system_config = {
        "star": [{"name": "A"}, {"name": "B"}],
        "sed": {},
    }
    user_params = {
        "star.A.teff": 5800,
        "star.1.radius": 1.2,
        "star.mass": [0.9, 1.1],
        "star.feh": 0.1,
        "star.B.feh": -0.2,
        "sed.av": 0.1,
        "sed.0.errscale": 2.0,
    }
    original = copy.deepcopy(user_params)

    # Act
    cm = ConfigManager(user_params, system_config=system_config)

    # Assert
    up = cm.user_params
    assert all(isinstance(v, dict) for v in up.values()), up
    assert up["star.0.teff"] == {"initval": 5800}
    assert up["star.1.radius"] == {"initval": 1.2}
    assert up["star.0.mass"] == {"initval": [0.9, 1.1]}
    assert up["star.1.mass"] == {"initval": [0.9, 1.1]}
    assert up["star.0.mass"] is not up["star.1.mass"]
    assert up["star.0.feh"] == {"initval": 0.1}
    assert up["star.1.feh"] == {"initval": -0.2}
    assert up["sed.av"] == {"initval": 0.1}
    assert up["sed.0.errscale"] == {"initval": 2.0}
    # The translation is internal: the caller's dict (the user's file) keeps
    # its own spelling.
    assert user_params == original


def test_the_no_config_branch_emits_a_field_dict_too():
    """
    Given a bare value and no system_config (the tests-and-direct-drivers
    branch, which skips key standardization),
    When a ConfigManager is built,
    Then the entry is still a field dict and the caller's dict is untouched.
    """
    # Arrange
    user_params = {"star.0.teff": 5800, "star.0.mass": [0.3, 0.7]}

    # Act
    cm = ConfigManager(user_params)

    # Assert
    assert cm.user_params == {
        "star.0.teff": {"initval": 5800},
        "star.0.mass": {"initval": [0.3, 0.7]},
    }
    assert user_params == {"star.0.teff": 5800, "star.0.mass": [0.3, 0.7]}


def test_a_null_entry_is_dropped_with_a_warning(caplog):
    """
    Given `star.0.teff:` with no value (YAML null) next to a real entry,
    When a ConfigManager is built,
    Then the null entry is gone from user_params -- it states no field, and
    resolve()/finalize already ignored it while the key-presence checks
    counted it -- and the user is told which line was ignored.
    """
    # Arrange
    user_params = {"star.0.teff": None, "star.0.mass": 1.0}

    # Act
    with caplog.at_level(logging.WARNING, logger="exozippy.config"):
        cm = ConfigManager(user_params, system_config={"star": [{}]})

    # Assert
    assert "star.0.teff" not in cm.user_params
    assert cm.user_params["star.0.mass"] == {"initval": 1.0}
    assert "star.0.teff" in caplog.text


def test_the_gui_document_keeps_its_bare_spelling():
    """
    Given a ruamel round-trip document holding a bare value with a comment
    (what the GUI hands to System),
    When a ConfigManager is built from it,
    Then dumping the document reproduces the user's text byte for byte: the
    translation is a copy, never an edit of the user's file.
    """
    # Arrange
    text = "star.A.teff: 5800  # from the spectrum\nstar.A.mass: [0.9, 1.1]\n"
    ryaml = YAML()
    doc = ryaml.load(text)

    # Act
    cm = ConfigManager(doc, system_config={"star": [{"name": "A"}]})
    out = io.StringIO()
    ryaml.dump(doc, out)

    # Assert
    assert cm.user_params["star.0.teff"] == {"initval": 5800}
    assert out.getvalue() == text


# ---------------------------------------------------------------------------
# Past the boundary: a bare entry is a bookkeeping bug, and it raises
# ---------------------------------------------------------------------------


def test_user_entry_raises_on_a_bare_value_naming_the_key():
    """
    Given a mapping holding a bare value (something wrote past the boundary),
    When user_entry reads it,
    Then it raises TypeError naming the key; an absent key is None and a
    field dict is returned as is.
    """
    # Arrange
    params = {"star.0.teff": 5800, "star.0.mass": {"initval": 1.0}}

    # Act / Assert
    with pytest.raises(TypeError, match=r"star\.0\.teff"):
        user_entry(params, "star.0.teff")
    assert user_entry(params, "star.0.mass") == {"initval": 1.0}
    assert user_entry(params, "star.0.radius") is None


def test_finalize_raises_on_a_bare_entry_written_after_construction():
    """
    Given a ConfigManager whose user_params had a bare value written into it
    AFTER construction,
    When finalize_user_params runs,
    Then it raises TypeError naming the key instead of re-guessing the shape
    (the re-guessing is what let a bare per-seed list lose its seeds).
    """
    # Arrange
    cm = ConfigManager({}, system_config={"star": [{"name": "Lens"}]})
    cm.user_params["star.0.mass"] = [0.3, 0.7]

    # Act / Assert
    with pytest.raises(TypeError, match=r"star\.0\.mass"):
        cm.finalize_user_params()
