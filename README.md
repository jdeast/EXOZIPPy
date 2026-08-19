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

`--extras gui` is worth taking even if you never open the GUI (and you probably
should not -- see "The GUI is experimental" below): ruamel-yaml lives in that
extra, and without it roughly 30 tests fail at import.

See CONTRIBUTING.md for the workflow (`master` is protected; changes go through
a pull request with a passing test suite).

## Supported platforms

Every push and pull request runs the full test suite on:

| OS | Python |
|----|--------|
| Linux (ubuntu-latest) | 3.12, 3.13, 3.14 |
| macOS (arm64) | 3.12 |

### Intel Macs: supported, with two gaps and an extra install step

**macOS x86_64 works**, on Python **3.12 or 3.13**, and a separate CI job
(`macos-15-intel`, nightly and on every pull request) keeps it that way. Two
features are unavailable there, and the install needs one extra command
before `poetry install`. Both gaps trace to a single upstream fact -- jaxlib's
last macOS x86_64 wheel is 0.4.38 and jaxlib ships no sdist, so no newer jax
can be installed on that hardware at all:

* the `gp:` key (Gaussian-process noise) is unavailable, because celerite2's
  PyMC backend imports jax at module scope;
* the `numpyro` and `blackjax` samplers are unavailable. Use `nuts`, `ptde`
  (the default) or `nutpie`, none of which touch jax.

Everything else -- RV, transit, SED, astrometry, microlensing -- works
normally. The runbook, the reasoning, and the upstream fixes that will remove
the extra step are in
[`MACOS_INTEL_INSTALL.md`](MACOS_INTEL_INSTALL.md). Apple Silicon Macs need
none of this; check with `uname -m`.

### Windows: supported through WSL2

**Windows is supported, via WSL2** (Windows Subsystem for Linux) -- not natively.
That is not a hedge: inside WSL2 you are running a real Linux kernel with a
genuine Ubuntu userspace, so it *is* the Linux platform in the table above,
`fork`-based samplers and all. A full development setup there passes **1272 of
1272 tests**. The step-by-step runbook, verified end to end on a real machine
(Windows 11 -> Ubuntu 26.04 -> Python 3.14 -> Poetry), is
[`WINDOWS_INSTALL.md`](WINDOWS_INSTALL.md).

#### Minimum Windows version

WSL2 requires **build 19041 or newer** (Windows 10 version 2004, the May 2020
Update) or any Windows 11. Check with `winver`, or from PowerShell:

```
[System.Environment]::OSVersion.Version
```

Note: Microsoft numbers Windows 10 releases `YYMM`, so "version 2004" means
April 2020, not the year 2004. Build numbers are less ambiguous and are what
this document uses.

Builds 18362 and 18363 (versions 1903 and 1909) can also run WSL2 on x64 if
fully patched, but `wsl --install` does not exist there and the manual setup is
not covered here.

Below that there is no way to run EXOZIPPy on Windows: native Windows cannot
run the samplers (below), and WSL1 is a syscall translation layer rather than a
Linux kernel. Windows 7, 8, 8.1 and early Windows 10 are therefore unsupported,
as is any Windows 10 after its October 2025 end of support. On hardware that
cannot take a current Windows, install Linux directly -- it is the platform
EXOZIPPy is developed and tested on, and it will outperform a VM on old
machines.

**Why not natively?** Two independent reasons:

1. **The PTDE sampler cannot work.** It builds worker pools with
   multiprocessing's `fork` start method so children inherit the compiled
   PyTensor logp function without pickling it -- cloudpickle cannot serialize
   one. Windows has no `fork`, only `spawn`.
2. **The suite never finished.** It took over 90 minutes against ~16 on Linux
   and never completed a CI run, so the job was removed rather than left
   producing no signal. Three real Windows bugs were found and fixed along the
   way (an unsatisfiable mkl pin, a POSIX-only `SIGALRM` in the symbolic solver,
   and a cross-process stop signal), so basic use may well work -- but nothing
   verifies it.

Native Windows would additionally need a GCC-style C++ compiler for PyTensor's
runtime compilation (MSVC will not do), which we have never tested.

Those are the headline reasons, not the whole list. If you want to pick native
Windows up, [`docs/windows-native.md`](docs/windows-native.md) is the running
inventory of known blockers -- each one with what breaks, why, and the minimum
work to clear it -- so you can size the job before starting it. Patches
welcome.

#### Quickstart

Open an administrator PowerShell (Start menu, type "powershell", then "Run as
administrator") and run:

```
wsl --install
```

**Reboot** -- the optional components only activate then, and without it
`wsl --status` claims virtualization is disabled even when it is not. Open the
Ubuntu app once to create a Unix user, then inside Ubuntu:

```
sudo apt update
sudo apt install -y g++ python3-dev python3-venv
python3 -m venv ~/exozippy-env
source ~/exozippy-env/bin/activate
pip install --pre exozippy
```

Note `python3`, not `python`: see the `PATH` note below. `python3-venv` is not
optional -- without it `python3 -m venv` fails on `ensurepip`.

#### What WSL2 costs you

One thing genuinely bites, and it is a resource default rather than a missing
feature:

- **Memory.** WSL2 takes **50% of host RAM** by default. On a 16 GB laptop that
  is ~7.6 GB, while `pyproject.toml` asks for `-n 6` test workers that each peak
  at 1-2 GB compiling PyTensor graphs. Measured on that machine, available
  memory bottomed out at **41 MB**: workers were killed (`[gwN] node down`),
  xdist died in its own scheduler, and one run hung for hours. Fix it with a
  `.wslconfig` (see the runbook) or run `-n 2` as CI does.

Smaller, setup-time only:

- Windows `PATH` is appended to the Linux one, so bare `python`/`pip`/`npm`
  resolve to *Windows* executables -- which is how `poetry install` can install
  nothing and still exit 0.
- There is no full systemd session, so `apt` prints harmless
  `Failed to connect to system scope bus` warnings.
- Keep the repo on the Linux filesystem, not `/mnt/c`, where I/O is far slower.

Not limitations: `fork` works (so PTDE and its tests run), PyTensor compiles C
at runtime, and the whole suite passes. GPU/CUDA for the JAX samplers is
untested.

The end-user path above (`pip install --pre exozippy` inside WSL) is enough to
run fits. For a development checkout -- git clone plus Poetry plus the test
suite -- follow [`WINDOWS_INSTALL.md`](WINDOWS_INSTALL.md), which covers the
traps above in order.

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

If the toolchain is broken anyway (no g++, or g++ without `Python.h`),
`exozippy` detects it at startup, prints a warning naming the fix, and falls
back automatically to PyTensor's much slower pure-Python mode -- usable as a
smoke test, not for a real fit. (Setting `PYTENSOR_FLAGS="cxx="` by hand is
not enough: models with more than ~31 likelihood terms then die on numpy's
32-operand ufunc limit; the automatic fallback also installs the graph
rewrite that works around it.)

## Running a fit

```
cd examples/ob140939
exozippy ob140939.yaml
```

### The GUI is experimental

There is an optional browser GUI (installed by the `gui` extra), started with:

```
exozippy-gui
```

**Treat it as experimental on every platform, including Linux and macOS.** It is
still buggy and has never been verified driving a real fit end to end, so it is
not part of what "supported" means above -- unlike the CLI, nothing in CI
exercises it beyond unit tests of its own modules. Use it to look around; do not
rely on it for science.

Note this is a statement about the GUI everywhere, not a WSL caveat.
