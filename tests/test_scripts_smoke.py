"""Every file in scripts/ imports, and every CLI in scripts/ answers --help.

Nothing in the suite exercised scripts/ at all, and it showed: mkprior.py
(now mkparam.py) did ``from exozippy.mkprior import backup_params, mkprior``
against a module that had never existed under that name, and a function that
had been deleted from the tree entirely.  It raised ModuleNotFoundError on
every invocation it ever had, and no test noticed for as long as it existed.

This file is the cheap guard against that recurring.  Deliberately narrow:

  * Importing a module executes its top level -- its imports, its constants,
    its decorators -- which is exactly where that class of rot lives.  It does
    NOT run the script, so this stays side-effect free.  Every script in
    scripts/ therefore has to keep its work behind ``main()``; two did not
    (make_test_fixtures.py wrote a fixture at import, validate_vbm_safedist.py
    ran a multi-minute VBM scan), and they were wrapped when this went in.
  * --help constructs the argparse parser and exits, which catches a flag
    referring to a parameter that no longer exists and a help string
    describing behaviour that changed.  It is driven in-process rather than
    as a subprocess so the whole file costs about as much as one import of
    pytensor rather than one per script.

What it does NOT do is run any script for real.  Several need IDL, a
multi-hour VBM scan, network access, or a finished fit's trace; that is why
the review this file came out of had to exercise those by hand.
"""

import importlib.util
import runpy
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"

# Scripts that deliberately expose no command line.  Everything NOT listed
# here must answer --help.  The list is checked against the tree below, so a
# rename or a deletion fails loudly instead of silently shrinking coverage.
NO_CLI = {
    # Diagnostics with no arguments: they pin one example and one raw point.
    "diag_mulens.py",
    "diag_rotation.py",
    # One-shot generators, hard-coded scenario, no knobs.
    "make_exofast_tran_reference.py",  # also: shells out to IDL
    "make_test_fixtures.py",
    "validate_vbm_safedist.py",
}


def _script_paths():
    return sorted(
        p for p in SCRIPTS_DIR.glob("*.py") if p.name != "__init__.py"
    )


SCRIPTS = [p.name for p in _script_paths()]


def _import_script(path):
    """Import scripts/<name>.py under a private module name.

    sys.path is snapshotted and restored: a script that inserts its own
    src/ at import (repro_vbm_timeout.py did) would otherwise mutate the
    worker for every test file that runs after this one.
    """
    mod_name = f"_scripts_smoke_{path.stem}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    saved_path = list(sys.path)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(mod_name, None)
        sys.path[:] = saved_path
    return module


def test_scripts_directory_is_not_empty():
    """
    Given the repository,
    When scripts/ is globbed for .py files,
    Then at least one is found.

    Guards the parametrizations below: an empty glob would make every other
    test in this file vacuously pass.
    """
    # Assert
    assert SCRIPTS, f"no .py files found under {SCRIPTS_DIR}"


def test_no_cli_list_matches_the_tree():
    """
    Given the NO_CLI opt-out list,
    When it is compared with the files actually in scripts/,
    Then every name in it still exists.

    Without this, deleting or renaming a script would silently drop it from
    the --help check rather than failing.
    """
    # Assert
    missing = sorted(NO_CLI - set(SCRIPTS))
    assert not missing, (
        f"NO_CLI names scripts that no longer exist: {missing}. Remove them "
        f"(or fix the rename) -- a stale entry silently exempts nothing."
    )


@pytest.mark.parametrize("name", SCRIPTS)
def test_script_imports(name):
    """
    Given a file in scripts/,
    When it is imported,
    Then its top level executes without raising.

    This is the check that mkprior.py failed for its entire lifetime.
    """
    # Act / Assert
    _import_script(SCRIPTS_DIR / name)


@pytest.mark.parametrize("name", [n for n in SCRIPTS if n not in NO_CLI])
def test_script_help(name, monkeypatch, capsys):
    """
    Given a script in scripts/ that exposes a command line,
    When it is run with --help,
    Then argparse builds its parser, prints usage and exits 0.

    Catches a --help text or a flag that outlived the API behind it, which is
    the second and third thing that was wrong with mkprior.py.
    """
    # Arrange
    module = _import_script(SCRIPTS_DIR / name)
    assert hasattr(module, "main"), (
        f"scripts/{name} has no main(); either give it one or add it to "
        f"NO_CLI with the reason."
    )
    monkeypatch.setattr(sys, "argv", [name, "--help"])

    # Act
    with pytest.raises(SystemExit) as exc:
        module.main()

    # Assert
    assert exc.value.code == 0, (
        f"scripts/{name} --help exited {exc.value.code}"
    )
    assert "usage" in capsys.readouterr().out.lower(), (
        f"scripts/{name} --help printed no usage line"
    )


@pytest.mark.parametrize("name", SCRIPTS)
def test_script_guards_its_main(name):
    """
    Given a file in scripts/,
    When its source is inspected,
    Then it guards execution behind ``if __name__ == "__main__":``.

    The import test above is only side-effect free while this holds.  Two
    scripts violated it: make_test_fixtures.py wrote into tests/fixtures/ at
    import time and validate_vbm_safedist.py started a multi-minute
    VBMicrolensing scan.
    """
    # Assert
    source = (SCRIPTS_DIR / name).read_text()
    assert '__name__ == "__main__"' in source, (
        f"scripts/{name} runs work at import time; move it behind "
        f'if __name__ == "__main__": main().'
    )


def test_mkparam_actually_runs_its_main_when_executed():
    """
    Given scripts/mkparam.py executed as __main__ with no arguments,
    When runpy runs it,
    Then argparse rejects the missing config and exits 2.

    The other half of the contract the tests above pin: the ``__main__``
    guard exists so that IMPORTING is inert, not so that RUNNING is.  A
    script whose main() was never wired to the guard would import fine and
    do nothing when invoked, which no other test here would notice.
    mkparam.py is the concrete case this review started from.
    """
    # Arrange
    argv = sys.argv
    sys.argv = ["mkparam.py"]

    # Act / Assert
    try:
        with pytest.raises(SystemExit) as exc:
            runpy.run_path(
                str(SCRIPTS_DIR / "mkparam.py"), run_name="__main__"
            )
        # argparse exits 2 on a missing required positional.
        assert exc.value.code == 2
    finally:
        sys.argv = argv
