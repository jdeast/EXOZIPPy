"""An empty SYSTEM config is refused by name; an empty PARAMS file is not.

Review 2.3.11.  ``load_yaml`` returns ``None`` for an empty file, and both
CLIs used to walk straight off the end of it: ``exozippy`` died with
``AttributeError: 'NoneType' object has no attribute 'get'`` from inside
``run_fit``, ``exozippy-modes`` with the same thing from
``config.get("prefix", ...)``, and neither message named the file the user
had just passed.  An empty system config cannot mean anything -- it names no
component, no prefix and no parameter file -- so ``load_system_config``
refuses it and says which file.

The asymmetry is JDE's ruling (2026-09-11): "an empty system config file
should fail.  an empty param file might be ok (it might fail later because
some component required a specific starting value, but it shouldn't fail by
construction)."  The params side is normalized to ``{}`` by construction
(``System.__init__``, and ``user_params or {}`` throughout config.py) and is
a SUPPORTED way to write a fit -- see run.md's "The `parameter_file:` key is
omittable, and an empty params file means the same thing".  So the guard is
in ``load_system_config`` and NOT in ``load_yaml``, which both paths share,
and the last two tests here are what pins that.
"""

from pathlib import Path

import pytest
from click.testing import CliRunner

from exozippy import cli, cli_modes
from exozippy.system import System
from exozippy.yamlio import load_system_config, load_yaml

CLIS = [("exozippy", cli.main), ("exozippy-modes", cli_modes.main)]


@pytest.mark.parametrize("name,command", CLIS, ids=[c[0] for c in CLIS])
def test_a_cli_refuses_an_empty_config_and_names_the_file(
    tmp_path, name, command
):
    """
    Given an empty system config file,
    When either console script is invoked on it,
    Then it fails with a message that NAMES the file and says it is empty.

    Both CLIs, because they had the identical defect at the identical line
    and a fix to one would have looked complete.
    """
    # ARRANGE
    empty = tmp_path / "empty.yaml"
    empty.write_text("")

    # ACT
    result = CliRunner().invoke(command, [str(empty)])

    # ASSERT
    assert result.exit_code != 0
    message = str(result.exception)
    assert str(empty) in message, message
    assert "empty" in message, message


@pytest.mark.parametrize("name,command", CLIS, ids=[c[0] for c in CLIS])
def test_a_cli_refuses_a_comments_only_config(tmp_path, name, command):
    """
    Given a config file holding only comments (which parses to None too),
    When either console script is invoked on it,
    Then it is refused the same way.

    The realistic shape of the empty case: a file someone commented out
    wholesale, rather than a zero-byte one.
    """
    # ARRANGE
    commented = tmp_path / "commented.yaml"
    commented.write_text("# star:\n#   - name: A\n")

    # ACT
    result = CliRunner().invoke(command, [str(commented)])

    # ASSERT
    assert result.exit_code != 0
    assert str(commented) in str(result.exception)


def test_a_non_mapping_config_is_refused_by_name(tmp_path):
    """
    Given a config file that parses to a list rather than a mapping,
    When load_system_config reads it,
    Then it is refused, naming the file and what it parsed as.

    The neighbouring failure: `config.get(...)` dies one frame later on a
    list exactly as it does on None, and the pre-fix message named neither
    the file nor the shape.
    """
    # ARRANGE
    listy = tmp_path / "listy.yaml"
    listy.write_text("- star\n- planet\n")

    # ACT / ASSERT
    with pytest.raises(ValueError) as exc:
        load_system_config(str(listy))
    assert str(listy) in str(exc.value)
    assert "list" in str(exc.value)


def test_a_real_config_still_loads(tmp_path):
    """
    Given an ordinary one-component config,
    When load_system_config reads it,
    Then it comes back as the parsed mapping.

    The control: the guard must refuse only what cannot mean anything.
    """
    path = tmp_path / "ok.yaml"
    path.write_text("star:\n  - name: A\nprefix: out/x\n")

    assert load_system_config(str(path)) == {
        "star": [{"name": "A"}],
        "prefix": "out/x",
    }


def test_load_yaml_still_returns_none_for_an_empty_file(tmp_path):
    """
    Given an empty file read through the SHARED loader,
    When load_yaml reads it,
    Then it still returns None.

    The params side goes through this function, and an empty params file is
    legal.  A guard here rather than in load_system_config would have made
    JDE's ruling impossible to honor -- it is the one thing this change must
    not do.
    """
    empty = tmp_path / "empty.params.yaml"
    empty.write_text("")

    assert load_yaml(str(empty)) is None


def test_an_empty_params_file_is_still_a_legal_fit(tmp_path):
    """
    Given a config naming a params file that is EMPTY,
    When the System is constructed,
    Then user_params is {} and nothing raises -- the file is normalized by
      construction, not refused.

    The other half of the ruling, asserted end-to-end rather than at the
    loader: `{}` is what every consumer in config.py already spells
    `user_params or {}` for.
    """
    # ARRANGE
    params = tmp_path / "p.params.yaml"
    params.write_text("")

    # ACT
    system = System(
        {"star": [{"name": "A"}], "parameter_file": str(params)},
    )

    # ASSERT
    assert system.user_params == {}
    assert Path(params).exists()
