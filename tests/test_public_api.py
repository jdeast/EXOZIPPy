"""The declared public API surface of the `exozippy` package (review 8.13.6).

`__all__` used to be `["__version__", "System"]`, so `run_fit` -- the in-memory
entry point src/exozippy/run.md documents -- `write_param_file`, and the
`solve_api`/`introspect` modules the GUI treats as contracts were unexported
and unmarked, indistinguishable from internals. These tests pin the set, pin
that every name in it actually resolves, and pin that resolving it is LAZY:
the lazy path is not a micro-optimization here, it is what keeps
`exozippy/__init__` and `exozippy.run`/`exozippy.mkparam` (which import from
this package) out of a circular import.
"""

import subprocess
import sys

import exozippy

EXPECTED = {
    "System",
    "__version__",
    "introspect",
    "run_fit",
    "solve_api",
    "write_param_file",
}


def test_the_public_surface_is_exactly_what_is_declared():
    """
    Given the package's __all__,
    When it is compared against the intended public set,
    Then they match exactly.

    Spelled out here rather than imported so the test pins the decision
    instead of echoing whatever the code currently says. Widening the surface
    is a decision, so it should take an edit in two places.
    """
    assert set(exozippy.__all__) == EXPECTED
    # Sorted, so additions land in a stable place and never conflict on order.
    assert list(exozippy.__all__) == sorted(exozippy.__all__)


def test_every_exported_name_resolves():
    """
    Given each name in __all__,
    When it is looked up on the package,
    Then it exists -- an __all__ entry with nothing behind it is worse than no
      entry, because `from exozippy import *` raises on it.
    """
    for name in exozippy.__all__:
        assert getattr(exozippy, name) is not None, name

    assert callable(exozippy.run_fit)
    assert callable(exozippy.write_param_file)
    assert callable(exozippy.solve_api.solve)
    assert callable(exozippy.introspect.full_schema)


def test_an_unknown_attribute_still_raises_attribute_error():
    """
    Given a name the package does not define,
    When it is accessed,
    Then AttributeError is raised, not KeyError.

    The lazy __getattr__ looks the name up in a dict; letting that KeyError
    escape would break hasattr(), copy, pickle and every other protocol that
    probes for optional attributes.
    """
    try:
        exozippy.no_such_name
    except AttributeError as exc:
        assert "no_such_name" in str(exc)
    else:
        raise AssertionError("expected AttributeError")


def test_dir_lists_the_lazy_names():
    """
    Given the lazily-resolved exports,
    When dir() is called on the package,
    Then they are listed.

    Without __dir__, tab completion and help() show a public surface smaller
    than __all__ claims -- the names work but are undiscoverable.
    """
    listed = set(dir(exozippy))

    assert EXPECTED <= listed


def test_importing_the_package_does_not_import_run_or_mkparam():
    """
    Given a fresh interpreter,
    When `import exozippy` runs,
    Then neither exozippy.run nor exozippy.mkparam is in sys.modules, and both
      still import on demand.

    A subprocess because this process has almost certainly imported them
    already. This is the property that keeps the lazy indirection honest: both
    modules import from this package, so making these eager would have
    __init__ and them initializing each other.
    """
    code = (
        "import sys, exozippy;"
        "assert 'exozippy.run' not in sys.modules, 'run imported eagerly';"
        "assert 'exozippy.mkparam' not in sys.modules,"
        " 'mkparam imported eagerly';"
        "exozippy.run_fit; exozippy.write_param_file;"
        "assert 'exozippy.run' in sys.modules;"
        "assert 'exozippy.mkparam' in sys.modules;"
        "print('ok')"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"


def test_the_contract_modules_declare_their_own_surfaces():
    """
    Given solve_api and introspect,
    When their __all__ is read,
    Then it names the functions the GUI calls.

    These two are exported as modules, so `__all__` on the package says
    nothing about what inside them is public; each has to say so itself.
    """
    assert set(exozippy.solve_api.__all__) == {
        "SolveResult",
        "solve",
        "validate",
    }
    assert set(exozippy.introspect.__all__) == {
        "boolean_option_keys",
        "component_schema",
        "full_schema",
        "list_components",
    }
