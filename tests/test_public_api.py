"""The declared public API surface of the `exozippy` package (review 8.13.6).

`__all__` was `["__version__", "System"]`, so `run_fit` -- the in-memory entry
point src/exozippy/run.md documents, and the batch path that is the primary use
case -- was unexported and unmarked, indistinguishable from an internal.

The scope is a ruling, not a judgement call, so these tests pin BOTH halves of
it: the three names that are public, and the names deliberately left OUT.
`write_param_file` stays internal because promoting it later is one
non-breaking line while demoting it is breaking; `solve_api` and `introspect`
stay internal because the GUI is first-party and has no third-party consumer;
`Component`/`Parameter` are a declared extension API documented in
components/components.md rather than part of the run-a-fit surface. An
enthusiastic future edit that "completes" `__all__` is the failure mode here,
which is why the exclusions get a test of their own.

Laziness is pinned too. It is not a micro-optimization: `exozippy.run` imports
from this package, so an eager import in `__init__` would have the two modules
initializing each other.
"""

import subprocess
import sys

import exozippy

PUBLIC = {"System", "__version__", "run_fit"}

# Deliberately internal -- see the module docstring. Split by what can
# honestly be asserted about each.
#
# Ordinary names: absent from __all__ AND from the package root.
INTERNAL_NAMES = {"write_param_file", "Component", "Parameter"}
# SUBMODULES: absent from __all__ only. `import exozippy.solve_api` anywhere in
# the process binds it as an attribute of the package -- that is an import
# mechanic, not a declaration -- so `not hasattr(exozippy, "solve_api")` is
# true or false depending on which test file ran first in the worker. Asserting
# it would be an order-dependent test that passes on its own and fails behind
# tests/test_solve_api.py (verified, not assumed). Membership in __all__ is the
# assertion that means anything here.
INTERNAL_SUBMODULES = {"solve_api", "introspect"}


def test_the_public_surface_is_exactly_the_three_names_ruled_public():
    """
    Given the package's __all__,
    When it is compared against the ruled public set,
    Then they match exactly.

    Spelled out here rather than imported from the module, so the test pins the
    decision instead of echoing whatever the code currently says.
    """
    assert set(exozippy.__all__) == PUBLIC


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


def test_the_internal_names_are_not_on_the_package_root():
    """
    Given the names the ruling keeps internal,
    When they are looked up on the package root,
    Then they are absent -- while still importable from their own modules.

    The point of the second half: "internal" here is documentation, not
    enforcement. Nothing stops an external caller, and that is what makes
    internal the cheap default -- promoting a name later is non-breaking,
    demoting one is not.
    """
    for name in INTERNAL_NAMES | INTERNAL_SUBMODULES:
        assert name not in exozippy.__all__, name
    for name in INTERNAL_NAMES:
        assert not hasattr(exozippy, name), (
            f"{name} is on the package root; the ruling keeps it internal"
        )

    from exozippy import introspect, solve_api  # noqa: F401
    from exozippy.components.component import Component  # noqa: F401
    from exozippy.components.parameter import Parameter  # noqa: F401
    from exozippy.mkparam import write_param_file  # noqa: F401


def test_an_unknown_attribute_still_raises_attribute_error():
    """
    Given a name the package does not define,
    When it is accessed,
    Then AttributeError is raised, not KeyError.

    The lazy __getattr__ looks the name up in a dict; letting that KeyError
    escape would break hasattr(), copy, pickle and every other protocol that
    probes for optional attributes -- including the hasattr calls in the test
    above, which would error instead of returning False.
    """
    try:
        exozippy.no_such_name
    except AttributeError as exc:
        assert "no_such_name" in str(exc)
    else:
        raise AssertionError("expected AttributeError")


def test_dir_lists_the_lazy_name():
    """
    Given run_fit, which resolves lazily,
    When dir() is called on the package,
    Then it is listed.

    Without __dir__, tab completion and help() show a public surface smaller
    than __all__ claims until something has already touched the name.
    """
    assert PUBLIC <= set(dir(exozippy))


def test_importing_the_package_does_not_import_run():
    """
    Given a fresh interpreter,
    When `import exozippy` runs,
    Then exozippy.run is not in sys.modules, and it still imports on demand.

    A subprocess because this process has almost certainly imported it already.
    This is the property that makes the lazy indirection load-bearing rather
    than decorative: `run` imports from this package, so an eager import here
    would have the two initializing each other. It also keeps `import exozippy`
    off the whole outputs tree for callers that only want System.
    """
    code = (
        "import sys, exozippy;"
        "assert 'exozippy.run' not in sys.modules, 'run imported eagerly';"
        "exozippy.run_fit;"
        "assert 'exozippy.run' in sys.modules;"
        "print('ok')"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"
