# Running EXOZIPPy on an Intel Mac (macOS x86_64)

Intel Macs are supported with **two documented gaps** and **one manual install
step**. Everything below is measured on GitHub's `macos-15-intel` runners by
the `Intel macOS` CI job (`.github/workflows/intel-mac.yml`), which installs
this way on every pull request -- so this document does not drift.

Apple Silicon Macs need none of this: they get wheels for everything and
install with a plain `poetry install`. Check which you have with `uname -m`
(`x86_64` = Intel, `arm64` = Apple Silicon).

## Summary

| | |
| --- | --- |
| Python | **3.12 or 3.13** (not 3.14 -- see below) |
| Works | RV, transit, SED, astrometry, microlensing; the `nuts`, `ptde` and `nutpie` samplers |
| Does not work | the `gp:` key (Gaussian-process noise); the `numpyro` and `blackjax` samplers |
| Manual step | one `pip install` for celerite2 before `poetry install` |

The gaps are not EXOZIPPy limitations. Both trace to a single upstream fact:
**jaxlib's last macOS x86_64 wheel is 0.4.38, and jaxlib publishes no sdist**,
so no newer jax can be installed on this platform at all -- not slowly, not
from source, not at all.

## Install

```bash
# 1. Check you are on Python 3.12 or 3.13.
python --version

# 2. Build celerite2 without its jax build dependency. See "Why" below.
pip install scikit-build-core numpy pybind11 cmake ninja setuptools setuptools_scm
pip install --no-build-isolation \
    --config-settings=cmake.define.BUILD_JAX=OFF \
    "celerite2>=0.3.3,<0.4.0"

# 3. Everything else, as normal.
poetry install --extras gui
```

Step 2 must come first. Without it, step 3 fails while *installing build
dependencies* for celerite2, before any of your machine's compiler is invoked.

You need the Xcode command line tools (`xcode-select --install`), because
`exoplanet-core` and `celerite2` have no Intel-Mac wheels and are compiled
from source. That part works fine -- CI does it on both 3.12 and 3.13.

## Why each restriction exists

### Python 3.12 or 3.13, not 3.14

`numba` is a hard dependency of PyTensor 3, and its last macOS x86_64 wheels
are 0.62.1 (with llvmlite 0.45.1), which ship cp310-cp313 only. There is no
cp314 wheel, and building llvmlite from source requires a matching LLVM
install. `pyproject.toml` caps numba on this platform accordingly -- PyTensor
accepts `numba>=0.58`, so 0.62.1 is comfortably inside its supported range.

`requires-python` still says `>=3.12,<3.15` because that field cannot be made
platform-conditional. On an Intel Mac, 3.14 will fail to resolve.

### No `gp:` (Gaussian-process noise)

`celerite2/pymc/ops.py` imports `pytensor.link.jax.dispatch` at module scope,
which imports jax. So celerite2's *PyMC* backend requires jax at import time,
even though nothing in it needs JAX to compute anything.

With no jax installable here, `gp:` raises a clear error naming the real
missing module. Nothing else is affected: `gp:` is off by default, EXOZIPPy
imports celerite2 lazily, and a fit without a `gp:` key never touches it.

Upstream fix in flight: [exoplanet-dev/celerite2#194](https://github.com/exoplanet-dev/celerite2/pull/194).

### No `numpyro` / `blackjax` samplers

Both are pure Python but inert without jaxlib. The newest releases that even
accept jax 0.4.38 are numpyro 0.19.0 and blackjax 1.3, and jax 0.4.38 is
itself unusable here for a second reason: `exoplanet-core` reaches `jax.ffi`,
which only became public in jax 0.5.0, and the resulting `AttributeError`
escapes its own `except ImportError` guard -- so installing an old jax breaks
`import exozippy` entirely, which is **worse than having no jax**.

`pyproject.toml` therefore omits the jax family on this platform rather than
pinning it down. Use `nuts`, `ptde` (the default) or `nutpie` instead; none of
them touches jax.

Upstream fix in flight: [exoplanet-dev/exoplanet-core#146](https://github.com/exoplanet-dev/exoplanet-core/pull/146),
which turns that crash into a clean degrade.

### Why celerite2 needs the manual step

celerite2 0.3.3 ships no macOS x86_64 wheel, so pip builds it from the sdist
-- and 0.3.3 lists `jax==0.8.0` in its `[build-system] requires`. pip resolves
build requirements in an *isolated* environment that our pins cannot reach, so
it goes looking for jaxlib 0.8.0, finds no Intel-Mac wheel, and the install
dies before any C++ is compiled. `PIP_CONSTRAINT` cannot help either:
constraints only narrow a requirement, so they cannot relax an `==` pin.

That build dependency is not structural -- celerite2's own `CMakeLists.txt`
declares `BUILD_JAX` an option and prints "Skipping JAX extension" when jax is
absent. `--no-build-isolation` plus `cmake.define.BUILD_JAX=OFF` is what lets
us say so explicitly. (`--no-build-isolation` also means *we* must supply the
build requirements, which is why the first command installs `setuptools_scm`:
scikit-build-core requests it dynamically, and skipping isolation skips that
request.)

Upstream fix in flight: [exoplanet-dev/celerite2#193](https://github.com/exoplanet-dev/celerite2/pull/193).
**When that is released, step 2 can be deleted** and `poetry install` will
work on its own.

## Verifying your install

```bash
python -c "import exozippy; print(exozippy.__file__)"
poetry run pytest tests/test_limbdark.py tests/test_gp.py -n0
```

`tests/test_gp.py` should report passes plus **10 skips** -- the skipped ones
are the kernel-building tests, skipped because celerite2's PyMC backend is
unimportable without jax, exactly as described above. Failures there (rather
than skips) mean something is wrong; skips are the expected result.

To see the whole picture at once, including which samplers are available:

```bash
poetry run python scripts/diag_jax_samplers.py
```

On an Intel Mac this reports `jax -- not installed` and `n/a` for both JAX
samplers. That is the correct output here, not a broken probe.

## The horizon

`macos-13`, the old free Intel CI runner, has already been removed by GitHub;
`macos-15-intel` and `macos-26-intel` replace it, and macOS 26 is expected to
be Apple's last Intel release. This platform has a finite life, and support
here is deliberately cheap to maintain -- a handful of environment markers in
`pyproject.toml` and one CI job -- rather than a parallel code path.
