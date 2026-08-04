# EXOZIPPy
[DeepWiki](https://www.deepwiki.com/jdeast/EXOZIPPy)

This will eventually be a python successor to EXOFASTv2, but it is not officially
released yet. Many features are missing, not tested, or not functional. If you'd
like to help with development, please contact me at jason.eastman@cfa.harvard.edu

## Installing

EXOZIPPy is on PyPI. Only pre-releases exist so far, so `--pre` is required --
without it pip reports that no matching version exists:

```
pip install --pre exozippy
```

All dependencies resolve from PyPI, so no compiler is required to *install* on
the supported platforms below (a compiler and the Python headers are still
needed at *runtime* -- see "Runtime requirements" under Supported platforms). A
nightly CI job installs exactly this way, with no lock file, to check that a
fresh install keeps working as upstream packages move.

### For development

Use Poetry, which installs the pinned `poetry.lock` and so reproduces a known
good dependency set:

```
conda create -n exozippy python=3.12
conda activate exozippy
git clone https://github.com/jdeast/EXOZIPPy.git
cd EXOZIPPy
poetry install --extras gui
poetry run pre-commit install
```

`--extras gui` is worth taking even if you never open the GUI: ruamel-yaml lives
in that extra, and without it roughly 30 tests fail at import.

See CONTRIBUTING.md for the workflow (`master` is protected; changes go through
a pull request with a passing test suite).

## Supported platforms

Every push and pull request runs the full test suite on:

| OS | Python |
|----|--------|
| Linux (ubuntu-latest) | 3.12, 3.13, 3.14 |
| macOS (arm64) | 3.12 |

Windows is **not currently tested and not supported.** It is not merely
unverified -- the test suite takes over 90 minutes there against ~16 minutes on
Linux and has never completed a CI run, so the job was removed rather than left
producing no signal. Three real Windows bugs were found and fixed along the way
(an unsatisfiable mkl pin, a POSIX-only `SIGALRM` in the symbolic solver, and
the GUI's cross-process stop signal), so basic use may well work. But the PTDE
sampler will not: it builds worker pools with multiprocessing's `fork` start
method, which does not exist on Windows. See `notes/todo.txt` if you want to
pick this up -- patches welcome.

Intel macOS is untested here and needs a C++ compiler: exoplanet-core publishes
wheels for CPython 3.12-3.14 on Linux (glibc 2.28+), Apple Silicon macOS and
Windows, but not Intel macOS, so it builds from source there.

### Runtime requirements

PyTensor compiles C code at runtime, so running a fit needs a C++ compiler
*and* the Python development headers -- even though `pip install` itself
succeeds without them. Missing headers show up as a `CompileError` ending in
`fatal error: Python.h: No such file or directory` the first time a model is
built. This is common on RHEL-family systems, where the headers ship in a
separate package from Python itself. To install both:

```
# RHEL / Rocky / Alma / CentOS / Fedora (match the -devel version to your Python)
sudo dnf install gcc-c++ python3.12-devel

# Debian / Ubuntu
sudo apt install g++ python3.12-dev
```

macOS's Xcode Command Line Tools (`xcode-select --install`) and any conda
Python (e.g. the Miniforge setup above) already include the headers. So if you
lack root on a Linux box, building your environment from a conda Python
instead of the system one sidesteps the problem entirely (conda can also
supply the compiler itself if the box has none):

```
conda create -n exozippy python=3.12
conda activate exozippy
pip install --pre exozippy       # or the Poetry development setup above
```

If you cannot get a compiler or headers at all, PyTensor can fall back to its
much slower pure-Python mode -- usable as a smoke test, not for a real fit:

```
PYTENSOR_FLAGS="blas__ldflags=,cxx="
```

## Running a fit

```
cd examples/ob140939
exozippy ob140939.yaml
```

The optional browser GUI (installed by the `gui` extra) starts with:

```
exozippy-gui
```
