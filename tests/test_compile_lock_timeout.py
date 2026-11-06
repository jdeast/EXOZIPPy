"""Two fits at once wait for the compile lock instead of crashing.

PyTensor's compile lock is one file per compiledir, shared by every process
of the user, and a waiter gives up after `compile__timeout` (120 s by
default) with filelock's Timeout.  `exozippy/__init__.py` raises that bound
to COMPILE_LOCK_TIMEOUT_S before pytensor is imported, unless the user set
it (PYTENSOR_FLAGS or a .pytensorrc).  The setting is fixed at pytensor
import, so each case runs in a fresh interpreter.
"""

import os
import subprocess
import sys

import pytest

_PROBE = (
    "import exozippy, pytensor; "
    "print(pytensor.config.compile__timeout, exozippy.COMPILE_LOCK_TIMEOUT_S)"
)


def _timeout_in_fresh_interpreter(tmp_path, flags=None, rc_text=None):
    env = dict(os.environ)
    if flags is None:
        env.pop("PYTENSOR_FLAGS", None)
    else:
        env["PYTENSOR_FLAGS"] = flags
    rc = tmp_path / "pytensorrc"
    rc.write_text(rc_text or "")
    env["PYTENSORRC"] = str(rc)
    out = subprocess.run(
        [sys.executable, "-c", _PROBE],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    got, ours = out.stdout.split()
    return int(got), int(ours)


@pytest.mark.parametrize("flags", [None, "floatX=float64"])
def test_the_default_is_raised_and_other_flags_kept(tmp_path, flags):
    """
    Given no user setting for compile__timeout (with or without other flags),
    When exozippy is imported,
    Then pytensor's compile lock timeout is EXOZIPPy's, not 120 s.
    """
    got, ours = _timeout_in_fresh_interpreter(tmp_path, flags=flags)
    assert got == ours
    assert ours > 120


def test_a_pytensor_flags_setting_is_left_alone(tmp_path):
    got, _ = _timeout_in_fresh_interpreter(
        tmp_path, flags="compile__timeout=7"
    )
    assert got == 7


def test_a_pytensorrc_setting_is_left_alone(tmp_path):
    got, _ = _timeout_in_fresh_interpreter(
        tmp_path, rc_text="[compile]\ntimeout = 9\n"
    )
    assert got == 9


def test_an_rc_spelling_pytensor_ignores_does_not_disable_ours(tmp_path):
    """
    Given `[global] compile__timeout`, which PyTensor does not read,
    When exozippy is imported,
    Then EXOZIPPy's timeout applies -- treating it as the user's would
    leave PyTensor's 120 s in force.
    """
    got, ours = _timeout_in_fresh_interpreter(
        tmp_path, rc_text="[global]\ncompile__timeout = 9\n"
    )
    assert got == ours
