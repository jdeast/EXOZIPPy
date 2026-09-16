"""The `parameter_file:` key is OMITTABLE (review 3.14.5).

A params file may already be empty: defaults.yaml, the components'
data-derived hints and the relaxation engine can start a fit between them,
and since the global search (8.3.1) a blind fit seeds its own period and
epoch from BLS/Lomb-Scargle.  Requiring the KEY while the FILE it names may
be empty was a distinction with no content -- and exactly the friction a
blind fit should not have to pay.

What stays fatal is a key that IS present and names a file that is not
there: that is a typo, not a choice.
"""

import logging
import os

import pytest

from exozippy.system import System

_CONFIG = {
    "run": {"name": "no_params"},
    "star": [{"name": "A", "mist": False}],
    "planet": [{"name": "b"}],
    "orbit": [{"name": "b", "primary": ["A"], "companion": ["b"]}],
}


def test_a_config_without_a_parameter_file_builds(caplog):
    """
    Given a config with no parameter_file key,
    When a System is built from it,
    Then it succeeds with empty user_params and says so at INFO.
    """
    with caplog.at_level(logging.INFO, logger="exozippy.system"):
        system = System(dict(_CONFIG))

    assert system.user_params == {}
    assert "No 'parameter_file'" in caplog.text
    assert "defaults.yaml" in caplog.text  # names where the values come from


def test_no_parameter_file_is_info_not_a_warning(caplog):
    """
    Given the same config,
    When a System is built,
    Then nothing is logged at WARNING -- this is a supported way to write a
      config, not a mistake to apologize for.
    """
    with caplog.at_level(logging.WARNING, logger="exozippy.system"):
        System(dict(_CONFIG))

    assert "parameter_file" not in caplog.text


def test_no_parameter_file_leaves_the_provenance_file_unset():
    """
    Given a config with no parameter_file,
    When a System is built,
    Then ConfigManager.param_file is not set to anything.

    Error messages quote that path to say where a bad value came from, and
    a config naming no file has nowhere on disk to point at.  Naming one
    anyway would send the user looking for a line in a file that does not
    exist -- the same failure `initval_source` exists to avoid.
    """
    system = System(dict(_CONFIG))

    assert not getattr(system.config_manager, "param_file", None)


def test_a_named_parameter_file_that_is_missing_still_raises(tmp_path):
    """
    Given a config naming a parameter_file that does not exist,
    When a System is built,
    Then FileNotFoundError still names the path and the resolution rule.

    Omitting the key is a choice; naming a file that is not there is a typo.
    """
    cfg = dict(_CONFIG, parameter_file=str(tmp_path / "nope.params.yaml"))

    with pytest.raises(FileNotFoundError) as exc:
        System(cfg)

    assert "nope.params.yaml" in str(exc.value)


def test_a_named_empty_parameter_file_is_equivalent(tmp_path, monkeypatch):
    """
    Given an EMPTY params file that the config does name,
    When a System is built,
    Then its user_params match the no-key case -- which is the argument for
      making the key optional in the first place.
    """
    empty = tmp_path / "empty.params.yaml"
    empty.write_text("")
    monkeypatch.chdir(tmp_path)

    with_file = System(dict(_CONFIG, parameter_file=str(empty)))
    without_file = System(dict(_CONFIG))

    assert with_file.user_params == without_file.user_params == {}
    # ...but the one that read a file still records which file it read.
    assert os.path.basename(with_file.config_manager.param_file) == (
        "empty.params.yaml"
    )


def test_an_in_memory_dict_still_wins_over_an_absent_key():
    """
    Given no parameter_file key AND an in-memory user_params dict,
    When a System is built,
    Then the dict is used -- the new branch must not shadow run_fit's
      in-memory entry point.
    """
    params = {"star.A.teff": {"initval": 5800.0}}

    system = System(dict(_CONFIG), user_params=params)

    assert system.user_params == params
