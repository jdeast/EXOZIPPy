#!/usr/bin/env bash
#
# Set up a working EXOZIPPy development environment on an Intel (x86_64) Mac.
#
# WHY THIS EXISTS
# ---------------
# `poetry install` cannot succeed unaided on this platform.  celerite2 ships
# no macOS x86_64 wheel from 0.3.3 on, so it is source-built -- and 0.3.3's
# [build-system] requires names `jax==0.8.0`, which build isolation resolves
# in a fresh environment our pins cannot reach.  jaxlib 0.8.0 has no Intel
# wheel and publishes NO sdist, so the build dependency is *unsatisfiable*
# rather than merely slow: the whole install dies before any C++ compiles,
# with the misleading "Cannot install build-system.requires for celerite2".
#
# That build dependency is not structural.  celerite2's own CMakeLists
# declares BUILD_JAX an option and the pymc and numpy backends do not use the
# JAX extension.  So we install the real build requirements ourselves, turn
# the JAX extension off explicitly, and build with --no-build-isolation.
# `poetry install` afterwards sees celerite2 already satisfied at the locked
# version and leaves it alone.
#
# This is the same recipe .github/workflows/intel-mac.yml runs, lifted out of
# a CI comment so a laptop can use it too -- which is the whole reason it is
# here.  KEEP THE TWO IN SYNC; if you change one, change the other.  Upstream
# fix that would delete both: drop `jax==0.8.0` from celerite2's build
# requires, since CMake already treats it as optional.
#
# WHAT YOU DO *NOT* GET ON THIS PLATFORM
# --------------------------------------
# pyproject.toml's environment markers deliberately install NO jax here
# (jaxlib's last Intel wheel is 0.4.38, and an old jax is measurably worse
# than none -- exoplanet-core reaches jax.ffi and `import exozippy` dies).
# Consequences, both expected and neither a defect:
#
#   * nuts / ptde / nutpie work; numpyro and blackjax do not.
#   * No GP support.  celerite2 builds and imports, but celerite2/pymc/ops.py
#     does `from pytensor.link.jax.dispatch import jax_funcify` at MODULE
#     scope, so its PyMC backend hard-requires jax at import time even though
#     CMake treats the JAX extension as optional at build time.  tests/
#     test_gp.py's kernel tests carry @needs_celerite2_pymc and skip.
#
# Intel macOS is a probe of a stack this project does not yet claim to
# support (see the header of .github/workflows/intel-mac.yml, which is
# deliberately outside the merge gate).  This script makes it workable, not
# supported.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "x86_64" ]]; then
    echo "This script is only needed on Intel macOS (found $(uname -s)/$(uname -m))."
    echo "Everywhere else, plain 'poetry install' works: celerite2 has a wheel."
    exit 1
fi

# Deliberately NO `poetry env use ...` here.  It is not needed -- poetry
# creates the environment on demand -- and naming an interpreter is actively
# harmful: `poetry env use python3` resolves whatever `python3` happens to be
# on PATH, which on a machine with several installed silently CREATES a
# second environment and switches the project to it.  That is how this script
# once moved a project from its py3.12 environment to a fresh py3.13 one, and
# then spent an install and a test run in the wrong place.  If you need a
# specific interpreter, choose it yourself before running this.
echo "==> Using poetry environment: $(poetry env info --path 2>/dev/null || echo '(will be created)')"

# The numpy range must MATCH what the bulk install resolves, not float free.
# Otherwise pip resolves the newest numpy here, celerite2's extension compiles
# against those headers, and `poetry install` then downgrades numpy under it --
# a compiled module running against an older numpy than it was built for,
# which fails at import in a way that looks nothing like its cause.
#
# setuptools_scm is in this list and is NOT in celerite2's [build-system]
# requires; that is our problem, not an upstream one.  celerite2 sets
# metadata.version.provider = "scikit_build_core.metadata.setuptools_scm",
# and that plugin declares its requirement dynamically through PEP 517's
# get_requires_for_build_wheel hook -- which --no-build-isolation skips, so we
# inherit the job of supplying it.
echo "==> Installing celerite2's real build requirements"
poetry run pip install \
    scikit-build-core "numpy>=2.0,<2.4" pybind11 cmake ninja setuptools setuptools_scm

echo "==> Building celerite2 without its jax build dependency (this is slow)"
poetry run pip install --no-build-isolation \
    --config-settings=cmake.define.BUILD_JAX=OFF \
    "celerite2>=0.3.3,<0.4.0"
poetry run python -c "import celerite2; print('celerite2', celerite2.__version__)"

# --all-extras, matching CI's `pip install ".[gui]"`. Without it `ruamel` is
# missing and tests/test_gui_document.py and tests/test_yaml_booleans.py fail
# to import -- which is a broken dev environment, not a platform finding.
echo "==> poetry install"
poetry install --all-extras

echo "==> Smoke test"
poetry run python -c "import exozippy; print('exozippy', exozippy.__file__)"
poetry run python -c "import exozippy.run, exozippy.system, exozippy.config"

cat <<'EOF'

Done. Expected on this platform (not defects -- see the header above):
  * `poetry run python -c "from celerite2.pymc import terms"` fails, so the
    `gp:` feature is unavailable and tests/test_gp.py's kernel tests skip.
  * sampler: {method: numpyro} and {method: blackjax} are unavailable;
    nuts, ptde, ptde_async, nutpie, demc and demcz all work.
EOF
