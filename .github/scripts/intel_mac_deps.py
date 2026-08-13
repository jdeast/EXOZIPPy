"""Emit the dependency list for an Intel-Mac (darwin + x86_64) install.

WHY THIS EXISTS, AND WHEN TO DELETE IT
--------------------------------------
``pyproject.toml`` currently declares ``jax>=0.5.0`` / ``jaxlib>=0.5.0``.
On macOS x86_64 that is not slow to install, it is IMPOSSIBLE: jaxlib's last
macOS x86_64 wheel is 0.4.38 and jaxlib publishes no sdist at all, so there
is nothing for pip to fall back to.  The same upgrade that raised that pin
(PyMC 6 / PyTensor 3) also raised ``numba`` past 0.62.1, the last release
with macOS x86_64 wheels -- and llvmlite's source build needs a matching
LLVM, which is not something a student laptop is going to produce.

Before committing those caps to ``pyproject.toml`` as environment markers we
want evidence that the resulting stack actually drives pymc 6, because the
newest numpyro and blackjax that accept jax 0.4.38 are 0.19.0 and 1.3 --
neither of which pymc declares a constraint against either way.  A pip
``--constraint`` file cannot answer that: constraints only narrow a
requirement, so ``jax==0.4.38`` against a declared ``jax>=0.5.0`` is a
ResolutionImpossible rather than an experiment.

So this script reads the declared dependencies and rewrites just the
overridden ones, leaving everything else -- including its environment
markers, which pip evaluates itself -- exactly as written.  CI installs the
result and then installs the project with ``--no-deps``.

WHEN THE MARKERS LAND IN pyproject.toml, DELETE THIS FILE and let CI install
the project normally.  Its whole purpose is to test a change that has not
been made yet; kept around afterwards it becomes a second, silently
diverging copy of the pin set.
"""

import argparse
import re
import sys
import tomllib
from pathlib import Path

# The proposed Intel-Mac pin set.  Keyed by canonical (PEP 503) distribution
# name; the value is the full replacement requirement string, or None to drop
# the dependency entirely.
#
# Every entry needs a reason, because each one is a ceiling imposed by wheel
# availability rather than a compatibility bound we discovered:
OVERRIDES = {
    # DROPPED, not downgraded -- and the difference is the whole finding of
    # CI run 31742944005.
    #
    # The obvious move is jax==0.4.38, the last version with a macOS x86_64
    # wheel (there is no sdist, so that is a hard ceiling). It does not
    # work, and it fails WORSE than having no jax at all:
    #
    #   exoplanet_core/pymc/__init__.py does
    #       try:    from exoplanet_core.pymc import jax_support
    #       except ImportError: pass
    #   and jax_support reaches jax.ffi.register_ffi_target at module
    #   scope. `jax.ffi` did not become public until jax 0.5.0 -- through
    #   0.4.38 it was jax.extend.ffi -- so on 0.4.38 that is an
    #   AttributeError, which that guard does NOT catch. `import exozippy`
    #   dies in components/orbit/orbit.py.
    #
    # With jax simply ABSENT the same guard catches a real ImportError and
    # the package imports fine. So exoplanet-core's `jax>=0.5.0` floor is a
    # genuine requirement rather than a conservative one, and on this
    # platform the only coherent configuration is no jax at all.
    #
    # numpyro and blackjax go with it: both are pure-Python but useless
    # without jaxlib, and the newest that even accept jax 0.4.38 are 0.19.0
    # and 1.3 respectively. Consequence for users: nuts / ptde / nutpie
    # work on Intel Mac, numpyro and blackjax do not.
    "jax": None,
    "jaxlib": None,
    "numpyro": None,
    "blackjax": None,
    # Last numba with macOS x86_64 wheels is 0.62.1 (llvmlite 0.45.1).
    # PyTensor 3 accepts numba>=0.58, so this stays inside its range.
    "numba": "numba>=0.62.1,<0.63",
    # numba 0.62.1 requires numpy<2.4.
    "numpy": "numpy>=2.0,<2.4",
    # NO celerite2 ENTRY, and the empty space is worth explaining because
    # two earlier CI runs put one here.
    #
    # celerite2 0.3.3 ships no macOS x86_64 wheel, so pip source-builds it,
    # and 0.3.3 added `jax==0.8.0` to its [build-system] requires. Build
    # isolation resolves that in a FRESH environment where our pins do not
    # apply, so it hunts for jaxlib 0.8.0, finds no Intel-Mac wheel, and the
    # install dies before a line of C++ is compiled (run 31742588763).
    # Nothing pinned at the top level reaches into a build environment, and
    # PIP_CONSTRAINT cannot help either: constraints only narrow, and
    # celerite2 asks for jax==0.8.0 exactly.
    #
    # Capping at 0.3.2 installs, and then does not work: `from
    # celerite2.pymc import terms` raises ImportError under PyMC 6 /
    # PyTensor 3, failing 9 GP tests (run 31743803311).
    #
    # The real answer is that the jax build dependency is not structural.
    # 0.3.3's CMakeLists declares
    #     option(BUILD_JAX "Build JAX extension (requires jaxlib headers)")
    # and, when `from jax import ffi` fails, prints "Skipping JAX extension"
    # and carries on -- the pymc and numpy backends do not need it. So the
    # workflow pre-builds 0.3.3 with --no-build-isolation and
    # cmake.define.BUILD_JAX=OFF, and this table leaves the version alone.
    #
    # Upstream fix, which would remove the need for that step entirely:
    # drop `jax==0.8.0` from celerite2's build requires, since its own CMake
    # already treats the extension as optional.
}


def canonical(name):
    """PEP 503 normalization, so `dm-tree` and `dm_tree` compare equal."""
    return re.sub(r"[-_.]+", "-", name).lower()


def requirement_name(req):
    """The distribution name at the head of a PEP 508 requirement string."""
    return canonical(re.split(r"[\s\[<>=!~;(]", req.strip(), 1)[0])


def build(pyproject_path, extras):
    with open(pyproject_path, "rb") as fh:
        data = tomllib.load(fh)

    project = data["project"]
    declared = list(project.get("dependencies", []))
    optional = project.get("optional-dependencies", {})
    for extra in extras:
        if extra not in optional:
            raise SystemExit(
                f"{pyproject_path}: no such extra {extra!r} "
                f"(have: {', '.join(sorted(optional)) or 'none'})"
            )
        declared += list(optional[extra])

    out, applied = [], set()
    for req in declared:
        name = requirement_name(req)
        if name in OVERRIDES:
            applied.add(name)
            replacement = OVERRIDES[name]
            if replacement is not None:
                out.append(replacement)
        else:
            out.append(req)

    # An override that matched nothing means the dependency was renamed or
    # removed upstream in pyproject.toml and this table was not updated --
    # i.e. the pin it describes is silently not being applied.  That is
    # exactly the failure mode this whole exercise is about, so it is an
    # error rather than a warning.
    missing = sorted(set(OVERRIDES) - applied)
    if missing:
        raise SystemExit(
            "intel_mac_deps.py: these overrides matched no declared "
            f"dependency: {', '.join(missing)}. Either pyproject.toml no "
            "longer declares them (drop the override) or they were renamed "
            "(update the key)."
        )
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pyproject",
        default=str(Path(__file__).resolve().parents[2] / "pyproject.toml"),
    )
    parser.add_argument(
        "--extra",
        action="append",
        default=[],
        dest="extras",
        help="also include this optional-dependency group (repeatable)",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="write here instead of stdout (a pip requirements file)",
    )
    args = parser.parse_args(argv)

    lines = build(args.pyproject, args.extras)
    text = "\n".join(lines) + "\n"
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        print(
            f"wrote {len(lines)} requirements to {args.output}",
            file=sys.stderr,
        )
        print(text, file=sys.stderr)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
