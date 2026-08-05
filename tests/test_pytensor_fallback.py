"""Tests for the PyTensor pure-Python backend fallback (pytensor_fallback.py)."""

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest
from pytensor.compile.mode import Mode

from exozippy import pytensor_fallback
from exozippy.pytensor_fallback import (
    _split_variadic,
    activate_python_fallback,
    check_c_backend,
    ensure_usable_backend,
    register_wide_elemwise_split,
)


@pytest.fixture
def restore_pytensor_config():
    """Save and restore the global PyTensor config the fallback mutates."""
    cxx = pytensor.config.cxx
    blas = pytensor.config.blas__ldflags
    yield
    pytensor.config.cxx = cxx
    pytensor.config.blas__ldflags = blas


def test_split_variadic_builds_legal_tree():
    """Given many more inputs than the operand limit, when _split_variadic
    rebuilds the sum, then every node in the result is narrow enough for the
    Python backend and the value is unchanged."""
    # Arrange: two levels of splitting required (1000 > 31*31 is false, but
    # 1000 inputs -> 33 chunks -> a second pass is still needed).
    n = 1000
    xs = [pt.constant(float(i), dtype="float64") for i in range(n)]

    # Act
    out = _split_variadic(pt.add, xs)

    # Assert: correct value, and no apply node anywhere exceeds the limit.
    assert out.eval() == pytest.approx(n * (n - 1) / 2.0)
    from pytensor.graph.traversal import ancestors

    for var in ancestors([out]):
        node = var.owner
        if node is not None:
            assert len(node.inputs) + len(node.outputs) <= 32


@pytest.mark.parametrize("fn,np_fn", [(pt.add, np.sum), (pt.mul, np.prod)])
def test_wide_variadic_runs_in_python_mode(fn, np_fn, restore_pytensor_config):
    """Given a variadic Add/Mul wider than numpy's 32-operand ufunc limit,
    when the splitting rewrite is registered and the graph runs on the
    pure-Python linker, then it evaluates instead of raising
    NotImplementedError and matches numpy."""
    # Arrange. cxx="" mirrors the real fallback environment; with cxx set,
    # pytensor's fusion pass assumes the C backend (1024-operand cap) and
    # would fuse the split tree back into one too-wide Composite.
    pytensor.config.cxx = ""
    register_wide_elemwise_split()
    n = 46  # what PyMC's joint logp produced for examples/ob140939
    xs = [pt.dscalar(f"x{i}") for i in range(n)]
    vals = 1.0 + np.arange(n) / float(n)

    # Act
    f = pytensor.function(
        xs, fn(*xs), mode=Mode(linker="py", optimizer="fast_run")
    )

    # Assert
    assert f(*vals) == pytest.approx(np_fn(vals))


def test_check_c_backend_passes_on_healthy_toolchain():
    """Given the CI/dev toolchain (which compiles C fine), when the probe
    runs, then it reports no error."""
    assert check_c_backend() is None


def test_check_c_backend_reports_missing_cxx(restore_pytensor_config):
    """Given PyTensor configured with no C compiler, when the probe runs,
    then it returns a reason instead of raising."""
    # Arrange
    pytensor.config.cxx = ""

    # Act
    reason = check_c_backend()

    # Assert
    assert reason is not None
    assert "no C compiler" in reason


def test_activate_python_fallback_sets_config_and_warns(
    restore_pytensor_config, capsys
):
    """Given a Python.h-flavored compile error, when the fallback activates,
    then the C backend is disabled in-process, the split rewrite is
    registered, and the banner names both the slowdown and the header fix."""
    # Act
    activate_python_fallback(
        "fatal error: Python.h: No such file or directory"
    )

    # Assert
    out = capsys.readouterr().out
    assert pytensor.config.cxx == ""
    assert pytensor_fallback._REWRITE_REGISTERED
    assert "ORDERS OF MAGNITUDE slower" in out
    assert "python3.X-devel" in out


def test_ensure_usable_backend_is_silent_when_healthy(capsys):
    """Given a working C toolchain, when ensure_usable_backend runs, then it
    returns True and prints nothing."""
    # Act
    ok = ensure_usable_backend()

    # Assert
    assert ok is True
    assert capsys.readouterr().out == ""
