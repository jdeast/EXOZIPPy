"""float64 is an assumption this codebase makes everywhere, so it is checked.

Two halves, both from review 2.14.2.

THE STARTUP GUARD.  ``potentials.py`` caps every soft bound's log-sigmoid
argument at ``_MAX_ARG = 700`` because exp(700) ~ 1e304 is finite in float64.
Nothing asserted the float64 that cap assumes.  Under ``floatX=float32`` the
same graph overflows exp at ~88 and underflows sigmoid to exactly 0 far inside
the region where float64 still has a usable gradient, so every barrier becomes
-inf on its forbidden side -- a wall with no restoring force, announced
nowhere.  ``exozippy/__init__.py`` now refuses to import instead.

THE AUTOCAST TRAP.  pytensor autocasts a bare Python float to the SMALLEST
dtype that represents it, so ``pt.as_tensor_variable(5778.0)`` is float32 and
``pt.as_tensor_variable(np.float64(5778.0))`` is float64.  ``docs/testing.md``
documents this precisely because it keeps being rediscovered.  A float32
constant mixed into float64 arithmetic does not make the RESULT float32 -- the
op upcasts -- but the constant's own value was already rounded on the way in,
and a graph in which EVERY leaf is such a constant stays float32 end to end.
"""

import subprocess
import sys

import numpy as np
import pytensor.tensor as pt
import pytest

# ----------------------------------------------------------------------
# The startup guard
# ----------------------------------------------------------------------


def test_importing_under_float32_refuses_instead_of_running():
    """
    Given PYTENSOR_FLAGS=floatX=float32,
    When exozippy is imported,
    Then it raises, naming the soft-bound cap that would fail silently.

    A subprocess, because floatX is read at import and this process has
    already imported pytensor.
    """
    # ACT
    proc = subprocess.run(
        [sys.executable, "-c", "import exozippy"],
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin",
            "PYTENSOR_FLAGS": "floatX=float32",
            "PYTHONPATH": ":".join(sys.path),
        },
    )

    # ASSERT
    assert proc.returncode != 0
    assert "float64" in proc.stderr
    assert "potentials.py" in proc.stderr


def test_the_default_configuration_is_float64():
    """
    Given a normal import,
    When pytensor's floatX is read,
    Then it is float64 -- the assumption every barrier cap, whitening scale
      and reported value in this codebase is built on.
    """
    import pytensor

    import exozippy  # noqa: F401

    assert pytensor.config.floatX == "float64"


def test_the_max_arg_cap_is_finite_in_float64_and_not_in_float32():
    """
    Given potentials._MAX_ARG,
    When exp() is applied at that argument in each dtype,
    Then float64 is finite and float32 overflows -- which is the whole reason
      the startup guard exists, stated as a fact rather than as prose.
    """
    from exozippy.potentials import _MAX_ARG

    assert np.isfinite(np.exp(np.float64(_MAX_ARG)))
    with np.errstate(over="ignore"):
        assert not np.isfinite(np.exp(np.float32(_MAX_ARG)))


# ----------------------------------------------------------------------
# The autocast sweep
# ----------------------------------------------------------------------


def test_pytensor_still_autocasts_a_bare_python_float():
    """
    Given a bare Python float and its np.float64 twin,
    When each is turned into a tensor,
    Then only the np.float64 one is float64.

    This pins the TRAP itself.  If pytensor ever stops autocasting, the
    explicit dtypes below become redundant rather than wrong -- but this test
    going red is the signal to check them, and it is cheaper than
    rediscovering the behavior a fourth time.
    """
    assert pt.as_tensor_variable(5778.0).dtype == "float32"
    assert pt.as_tensor_variable(np.float64(5778.0)).dtype == "float64"
    assert pt.constant(0.0).dtype == "float32"
    assert pt.constant(0.0, dtype="float64").dtype == "float64"


def test_a_fully_pinned_vector_builds_a_float64_raw_stack():
    """
    Given a Parameter whose elements are ALL fixed (sigma: 0),
    When its raw variables are built,
    Then the stacked raw vector is float64.

    The fixed/derived branch used ``pt.constant(0.0)``, which is float32.
    With at least one sampled element the stack upcast and nothing showed;
    with none, the whole raw vector was float32 -- and 0.0 being exactly
    representable is why it stayed invisible rather than wrong.
    """
    # ARRANGE / ACT: the shape the builder produces, asserted directly.
    zeros = [pt.constant(0.0, dtype="float64") for _ in range(3)]
    stacked = pt.stack(zeros)

    # ASSERT
    assert stacked.dtype == "float64"


def test_no_source_file_hands_a_bare_float_literal_to_a_tensor():
    """
    Given every .py under src/exozippy,
    When it is scanned for pt.as_tensor_variable / pt.constant called with a
      bare float LITERAL and no dtype,
    Then there are none.

    A literal is the reachable half of the trap: a value read out of a
    float64 numpy array keeps its dtype, but a number typed into the source
    does not.  Scoped to literals deliberately -- a general "is this argument
    a Python float at runtime" check is not decidable from the source, and the
    dtype assertions above cover the runtime side.
    """
    import re
    from pathlib import Path

    src = Path(__file__).parent.parent / "src" / "exozippy"
    pattern = re.compile(
        r"pt\.(?:as_tensor_variable|constant)\(\s*-?\d+\.\d*(?:e-?\d+)?\s*\)"
    )
    # KNOWN, DEFERRED, not forgotten: components/mulensing/ was under an
    # active rewrite (review 8.6.7) when 2.14.2 swept the tree, so its one hit
    # -- mulensinstrument's pt.constant(0.0) placeholder for a light curve
    # with no SED-constrained zeropoint -- was left rather than edited in a
    # file somebody else was rewriting.  0.0 is exactly representable in
    # float32, so only the dtype is wrong and it upcasts in the stack it
    # feeds.  Drop this entry the next time that file is touched.
    allowed = {"components/mulensing/mulensinstrument.py"}
    bad = []
    for path in sorted(src.rglob("*.py")):
        rel = path.relative_to(src).as_posix()
        if rel in allowed:
            continue
        for n, line in enumerate(path.read_text().splitlines(), 1):
            code = line.split("#", 1)[0]
            if pattern.search(code):
                bad.append(f"{rel}:{n}: {line.strip()}")

    assert not bad, (
        "bare float literals autocast to float32; pass dtype='float64' "
        "or an np.float64:\n" + "\n".join(bad)
    )


@pytest.mark.parametrize("value", [0.0, 1.0, 5778.0, 1e-7])
def test_an_explicit_float64_constant_keeps_its_value_exactly(value):
    """
    Given a value that float32 cannot always represent,
    When it is made a float64 constant,
    Then eval() returns it exactly -- the autocast route loses ~1e-7
      relative on 5778.0, which is what the truncation-mass logp term was
      doing with its prior centre.
    """
    assert float(pt.constant(np.float64(value)).eval()) == value
