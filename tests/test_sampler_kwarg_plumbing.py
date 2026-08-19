"""Every sampler knob both PTDE samplers accept must reach BOTH of them.

`adapt_ladder` was forwarded to `ptde_sample` and NOT to `ptde_async_sample`,
which accepts it.  So `adapt_ladder: true` in a config was silently inert for
ptde_async -- the recommended sampler for Op-based (microlensing) models, and
the one the DC2018 fits actually use.  The knob had never once been reachable
from a config, and nothing said so: no warning, no error, and the sampler
simply took the parameter's own `False` default.

That cost real time.  A 21-hour DC2018 event-128 run and a second one after it
were launched specifically to test ladder re-spacing, both reported zero
re-spacing events, and the absence was read as evidence about the adaptation
ALGORITHM when it was really evidence that the algorithm was never switched on.

The invariant pinned here is deliberately the SYMMETRY rather than a list of
expected kwargs: a key that both functions accept but only one call site passes
is exactly this bug's shape, needs no judgement about what "should" be
forwarded, and stays correct as knobs are added.  Keys only one function
accepts are excluded automatically, which is what makes the two documented
asymmetries -- rung_thin_factor/rung_thin_start (ptde-only: thinning addresses
the blocking problem async dispatch removes outright) and store_hot_chains
(async-only) -- pass without an allowlist to maintain.
"""

import ast
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_PY = REPO_ROOT / "src" / "exozippy" / "run.py"
SAMPLERS = {
    "ptde_sample": REPO_ROOT / "src" / "exozippy" / "samplers" / "ptde.py",
    "ptde_async_sample": (
        REPO_ROOT / "src" / "exozippy" / "samplers" / "ptde_async.py"
    ),
}


def _accepted_params(path, func_name):
    """Parameter names of ``func_name`` as defined in ``path``."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return {a.arg for a in node.args.args + node.args.kwonlyargs} - {
                "model",
                "system",
            }
    raise AssertionError(f"{func_name} not found in {path}")


def _forwarded_kwargs(func_name):
    """Keyword names run.py passes at its ``func_name(...)`` call site."""
    src = RUN_PY.read_text(encoding="utf-8")
    # The call site, not the import: match "name(" preceded by "= " so the
    # `from ... import name` line cannot be picked up.
    start = src.index(f"= {func_name}(") + len(f"= {func_name}")
    depth = 0
    for end in range(start, len(src)):
        if src[end] == "(":
            depth += 1
        elif src[end] == ")":
            depth -= 1
            if depth == 0:
                break
    return set(re.findall(r"^\s*(\w+)=", src[start:end], re.M))


def test_both_ptde_samplers_are_defined_and_called():
    """Guards the tests below from passing vacuously on a rename."""
    for name, path in SAMPLERS.items():
        assert _accepted_params(path, name)
        assert _forwarded_kwargs(name)


@pytest.mark.parametrize("func_name", sorted(SAMPLERS))
def test_run_py_forwards_adapt_ladder_to_this_sampler(func_name):
    """
    Given a PTDE sampler that accepts adapt_ladder,
    When run.py dispatches to it,
    Then adapt_ladder is forwarded.

    The specific regression: ptde_async accepted it and never received it.
    """
    # Arrange
    accepted = _accepted_params(SAMPLERS[func_name], func_name)
    assert "adapt_ladder" in accepted, (
        f"{func_name} no longer accepts adapt_ladder; if that is deliberate, "
        f"delete this test rather than weakening it."
    )

    # Act / Assert
    assert "adapt_ladder" in _forwarded_kwargs(func_name), (
        f"run.py does not pass adapt_ladder to {func_name}, so the sampler "
        f"config key is silently inert for it."
    )


def test_no_shared_sampler_knob_reaches_only_one_of_them():
    """
    Given a keyword BOTH PTDE samplers accept,
    When run.py's two call sites are compared,
    Then neither passes it while the other does not.

    The general form of the bug.  A knob only one function accepts is not an
    asymmetry (rung_thin_* is ptde-only, store_hot_chains async-only) and is
    excluded by the intersection, so this needs no maintained allowlist.
    """
    # Arrange
    sync_accepts = _accepted_params(SAMPLERS["ptde_sample"], "ptde_sample")
    async_accepts = _accepted_params(
        SAMPLERS["ptde_async_sample"], "ptde_async_sample"
    )
    shared = sync_accepts & async_accepts

    # Act
    sync_passed = _forwarded_kwargs("ptde_sample") & shared
    async_passed = _forwarded_kwargs("ptde_async_sample") & shared

    # Assert
    assert sync_passed == async_passed, (
        "asymmetric plumbing for keywords BOTH samplers accept -- "
        f"ptde only: {sorted(sync_passed - async_passed)}; "
        f"ptde_async only: {sorted(async_passed - sync_passed)}"
    )
