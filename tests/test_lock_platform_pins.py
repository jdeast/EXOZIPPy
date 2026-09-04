"""The lock must honor pyproject's platform-conditional pins.

WHY THIS EXISTS. Dependabot PR #232 ("bump the python-dependencies group with
9 updates") silently DOWNGRADED numba 0.65.1 -> 0.62.1 and llvmlite
0.47.0 -> 0.45.1 on every platform, by deleting the non-Intel-Mac entries and
stripping the `sys_platform == "darwin" and platform_machine == "x86_64"`
marker off the Intel-Mac ones. Nothing upstream forced it -- both versions are
live and unyanked on PyPI with cp314 wheels, and none of the nine bumps names
numba. It was a bad resolution.

What makes it worth a test is HOW IT WAS CAUGHT. Fifteen CI checks went GREEN
while installing a numba that pyproject forbids; only Python 3.14 failed, and
only because llvmlite 0.45.1 happens to ship no cp314 wheel, so it fell back
to a source build that died in setuptools' vendored distutils. Take 3.14 out
of the matrix and this merges clean and silently downgrades numba for every
Linux user.

So the check is: for each package pyproject pins per platform, the lock must
carry ONE ENTRY PER MARKER at a version satisfying that marker's constraint.
That is cheap, needs no network, and fails on the resolution rather than on a
downstream build.

Tests follow AAA with Given/When/Then docstrings.
"""

import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
LOCKFILE = ROOT / "poetry.lock"


def _lock_entries(name):
    """Every [[package]] block for `name`: (version, markers-or-None)."""
    text = LOCKFILE.read_text()
    out = []
    for block in text.split("[[package]]"):
        m = re.search(r'^name = "%s"\s*$' % re.escape(name), block, re.M)
        if not m:
            continue
        ver = re.search(r'^version = "([^"]+)"', block, re.M)
        mark = re.search(r"^markers = (.+)$", block, re.M)
        out.append(
            (ver.group(1) if ver else None, mark.group(1) if mark else None)
        )
    return out


def _pyproject_constraints(name):
    """Every dependency string in [project].dependencies naming `name`."""
    data = tomllib.loads(PYPROJECT.read_text())
    deps = data["project"]["dependencies"]
    return [
        d for d in deps if re.match(r"^%s\s*[><=!\[]" % re.escape(name), d)
    ]


def _floor(constraint):
    """The `>=X.Y.Z` floor in one dependency string, as a tuple."""
    m = re.search(r">=\s*([0-9]+(?:\.[0-9]+)*)", constraint)
    if not m:
        return None
    return tuple(int(p) for p in m.group(1).split("."))


def _as_tuple(version):
    return tuple(int(p) for p in re.findall(r"\d+", version)[:3])


# The packages whose pins are platform-conditional today. numba is the one
# that actually broke; llvmlite follows it and is listed because it is the
# transitive half that carries the wheels.
@pytest.mark.parametrize("package", ["numba"])
def test_every_platform_conditional_pin_has_a_lock_entry(package):
    """
    Given a package pyproject pins twice, once per platform marker,
    When the lock is read,
    Then it carries at least as many entries as there are pins.

    Collapsing two conditional pins into one entry is exactly what PR #232
    did, and it is invisible on any platform whose wheels happen to exist for
    the surviving version.
    """
    # Arrange
    constraints = _pyproject_constraints(package)

    # Act
    entries = _lock_entries(package)

    # Assert
    assert len(constraints) >= 2, (
        f"{package} is no longer pinned per platform in pyproject; if that is "
        f"deliberate, this test should be updated or dropped rather than "
        f"silently passing"
    )
    assert len(entries) >= len(constraints), (
        f"pyproject pins {package} {len(constraints)} times (per platform) but "
        f"poetry.lock has {len(entries)} entry/entries: {entries}. A collapsed "
        f"entry applies one platform's version everywhere -- see this module's "
        f"docstring."
    )


@pytest.mark.parametrize("package", ["numba"])
def test_no_lock_entry_falls_below_its_pyproject_floor(package):
    """
    Given the floors pyproject declares for a package,
    When the locked versions are compared to them,
    Then the highest locked version reaches the highest floor.

    The precise failure in PR #232: pyproject's non-Intel-Mac floor is
    numba>=0.65.1, and the lock offered only 0.62.1. Checking the MAXIMUM
    against the MAXIMUM floor is deliberately loose -- it does not try to
    re-evaluate environment markers, which is poetry's job, only to catch a
    lock that cannot satisfy the strictest pin on any platform.
    """
    # Arrange
    floors = [
        f for f in (_floor(c) for c in _pyproject_constraints(package)) if f
    ]
    assert floors, f"no >= floor found for {package} in pyproject"

    # Act
    locked = [_as_tuple(v) for v, _ in _lock_entries(package) if v]

    # Assert
    assert locked, f"{package} is absent from poetry.lock"
    assert max(locked) >= max(floors), (
        f"poetry.lock's highest {package} is {max(locked)} but pyproject "
        f"requires >= {max(floors)} on at least one platform. A resolution "
        f"that drops the higher pin downgrades every platform that needed it."
    )


def test_llvmlite_keeps_a_cp314_wheel():
    """
    Given the locked llvmlite,
    When its wheel list is inspected,
    Then a cp314 wheel is present.

    This is the concrete thing that broke: without a cp314 wheel, Python 3.14
    builds llvmlite from source and dies in setuptools' vendored distutils
    (`Popen.__init__() got an unexpected keyword argument 'dry_run'`). The
    project tests 3.14, so a lock without that wheel cannot install there --
    and asserting the wheel says why in one line instead of 400 lines of
    build log.
    """
    # Arrange / Act
    text = LOCKFILE.read_text()

    # Assert
    assert "cp314" in text, "no cp314 wheels anywhere in the lock"
    assert re.search(r"llvmlite-[0-9.]+-cp314", text), (
        "the locked llvmlite ships no cp314 wheel, so Python 3.14 will try to "
        "build it from source -- see this module's docstring"
    )
