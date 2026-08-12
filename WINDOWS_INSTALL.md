# Running EXOZIPPy on Windows (via WSL2)

EXOZIPPy does not run well on native Windows and the recommended path is
**WSL2** (Windows Subsystem for Linux). This document is the setup runbook. It
is updated as each step is actually verified on a real machine -- see the
status tags: [VERIFIED], [EXPECTED] (not yet reproduced here), [TODO].

Verified end to end on: Windows 11 Pro -> WSL2 -> Ubuntu 26.04 LTS ("resolute")
-> system Python 3.14.4 -> Poetry 2.4.1 -> gcc 15.2.0, on 8 cores / 7.6 GiB.
Result: **1260 of 1263 tests pass**, and the 3 failures are memory-pressure
artifacts that pass in isolation -- see "Known test results on this machine".
This platform also turned up one genuine EXOZIPPy bug (a cross-machine
determinism failure, PR #92), documented there.

Note there is no WSL-specific code in this tree -- WSL2 is a real Linux kernel
with genuine Ubuntu userspace, so the `ubuntu-latest` CI matrix already covers
the *runtime*. Everything below is therefore setup and environment, which is
exactly what CI cannot cover. **Do not add a Windows CI job**: native Windows
cannot run PTDE at all (no `fork`), and per README the job was removed for
taking 90+ minutes without ever completing.

## Why not native Windows?

Two independent reasons, both pushing to WSL:

1. **The samplers require `fork`.** PTDE's worker pool is built with
   `multiprocessing.get_context("fork")` (`src/exozippy/samplers/_common.py`,
   `make_pool`/`recycle_pool`; also `run.py` and `outputs/evidence.py`) and
   relies on copy-on-write fork so child processes inherit the compiled
   PyTensor logp function *without pickling it* -- cloudpickle (what PyMC's
   `spawn` backend would use) cannot serialize a compiled PyTensor function.
   Windows has no `fork`, only `spawn`, so on native Windows these samplers
   fall back to serial execution (or fail). The test suite gates the relevant
   tests behind `tests/conftest.py`'s `requires_fork` marker, which skips on
   `not hasattr(os, "fork")`. [VERIFIED -- code inspection]

2. **Speed.** The observed native-Windows test suite ran 10x+ slower before it
   was killed, consistent with losing the parallel workers plus PyTensor's
   C-compile toolchain being slower/less reliable on Windows. [EXPECTED]

## Read this first: two WSL-specific traps

Both of these cost real time on this machine. They are not in any upstream
install guide, so they are up front rather than buried in "gotchas".

### Trap 1 -- Windows PATH interop shadows your Linux toolchain

WSL appends your entire Windows `PATH` to the Linux `PATH` by default. Because
Ubuntu ships `python3` but **no** `python`, and no `pip` or `node` at all, bare
commands silently resolve to Windows executables:

```bash
which -a python pip npm
# /mnt/c/Users/<you>/.pyenv/pyenv-win/shims/python
# /mnt/c/Users/<you>/.pyenv/pyenv-win/shims/pip
# /mnt/c/Program Files/nodejs/npm
```

A Windows miniconda install also landed on the path here. The failure mode is
ugly: `pyenv-win`'s shims are POSIX shell scripts saved with **CRLF** line
endings, so their shebang carries a trailing `\r` and Linux `exec` fails with

```
[Errno 2] No such file or directory: '/mnt/c/Users/<you>/.pyenv/pyenv-win/shims/python'
```

This is what `poetry install` does on a fresh clone -- and it **exits 0** while
installing nothing. See Step 5 for the fix. [VERIFIED -- hit here]

Rules of thumb:

- Always type `python3`, never `python`.
- Never `pip install` / `npm install -g` inside WSL expecting a Linux install.
- If a Linux tool reports a `/mnt/c/...` path in an error, this is why.

If you want the trap gone permanently rather than worked around, disable the
Windows path append. **This restarts WSL and kills any running shells**
(including a Claude Code session), so do it before you start work, not during:

```bash
sudo tee /etc/wsl.conf >/dev/null <<'EOF'
[interop]
appendWindowsPath = false
EOF
```

Then from PowerShell: `wsl --shutdown`, and reopen Ubuntu. You can still launch
Windows programs by full path (e.g. `/mnt/c/Windows/explorer.exe .`).
Status: [TODO -- not applied here; the Step 5 Poetry config was used instead]

### Trap 2 -- `sudo` needs a real terminal

Every `sudo` step below must be run in an ordinary Ubuntu shell. Claude Code's
`!` shell prefix runs commands without a TTY, so `sudo` cannot prompt:

```
sudo: A terminal is required to authenticate
```

Keep a second Ubuntu window open for the `sudo` steps. [VERIFIED -- hit here]

## Prerequisites

- **Windows build 19041 or newer.** That is Windows 10 version 2004 (the May
  2020 Update) or any Windows 11. Windows 11 Pro is what this runbook was
  verified on [VERIFIED]; Windows 10 at build 19041+ is expected to work but
  has not been reproduced here [EXPECTED].

  Note: Microsoft numbers Windows 10 releases `YYMM`, so "version 2004" is
  April 2020, not the year 2004. This document uses build numbers, which are
  unambiguous.

  Builds 18362 and 18363 (versions 1903 and 1909) can also run WSL2 on x64 if
  fully patched, but `wsl --install` (Step 1) does not exist there -- the
  manual setup is enabling two optional Windows features, installing the
  kernel update package, then `wsl --set-default-version 2`. Not covered here.

  Below 18362 there is no WSL2, and therefore no supported way to run EXOZIPPy
  on Windows: native Windows cannot run the samplers (no `fork` -- see "Why
  not native Windows?" above), and WSL1 is a syscall translation layer rather
  than a Linux kernel. Windows 7, 8 and 8.1 are out. Windows 10 reached end of
  support in October 2025, so an unpatched Windows 10 machine is a security
  question as well as a compatibility one. On hardware that cannot take a
  current Windows, install Linux directly -- it is the platform EXOZIPPy is
  developed and tested on, and it will outperform a VM on old machines.

  Check the build from PowerShell, or run `winver`:

  ```powershell
  [System.Environment]::OSVersion.Version    # want: Build 19041 or higher
  ```

- CPU virtualization enabled in firmware. On this machine
  `Win32_Processor.VirtualizationFirmwareEnabled` was already `True`, so **no
  BIOS trip was needed**. [VERIFIED]
- Admin access (for `wsl --install`) and one reboot.

Check virtualization from PowerShell (non-elevated is fine):

```powershell
(Get-CimInstance Win32_Processor).VirtualizationFirmwareEnabled   # want: True
```

If this is `False`, enable Intel VT-x / AMD-V in your BIOS/UEFI first.

## Step 1 -- Install WSL2 + Ubuntu

In an **elevated** PowerShell (Run as Administrator):

```powershell
wsl --install
```

This enables the "Virtual Machine Platform" and "Windows Subsystem for Linux"
optional components, installs WSL2, and installs Ubuntu by default.

**Then REBOOT.** The optional components only activate after a reboot. If you
skip it, `wsl --status` reports "WSL2 is unable to start since virtualization
is not enabled" even when your firmware flag is `True`. [VERIFIED -- this was
the exact symptom hit here]

After rebooting, confirm:

```powershell
wsl -l -v          # should list Ubuntu, VERSION 2
wsl --status       # Default Version: 2, no error
```

If no distro is listed after reboot:

```powershell
wsl --install -d Ubuntu
```

On first launch, Ubuntu prompts for a UNIX username and password (these are
local to the Linux environment, unrelated to your Windows login).

Status: [VERIFIED -- Ubuntu 26.04 LTS came up after the reboot]

## Step 2 -- System build dependencies (inside Ubuntu)

Open the Ubuntu shell (Start menu -> Ubuntu, or `wsl` from PowerShell) and:

```bash
sudo apt update
sudo apt install -y build-essential python3-dev python3-venv
```

- `build-essential` provides the C compiler PyTensor needs to compile its ops
  (installed gcc/g++ 15.2.0 here).
- `python3-dev` provides `Python.h`, which those compiled ops link against.
- `python3-venv` is **required and easy to miss**: the stock image has no
  `ensurepip`, so `python3 -m venv` fails with *"ensurepip is not available"*
  and the Poetry installer in Step 5 cannot bootstrap. There is also no `pip`
  module at all on a fresh image (`python3 -m pip` -> `No module named pip`);
  you do not need one if you use Poetry as described below.

`git` (2.53.0) and `curl` (8.18.0) were already present on the stock image.

Do not skip `sudo apt update` -- the freshly installed image had only a partial
package index (no `universe` lists), so `apt install` could not resolve
everything until the index was refreshed.

Status: [VERIFIED]

## Step 3 -- Claude Code (inside Ubuntu)

Run Claude Code as a native Linux process inside WSL -- do NOT drive WSL from
the Windows side, and do **not** use `npm install -g`, which per Trap 1 would
install into your *Windows* npm. Use the native installer, which needs no Node
or nvm at all:

```bash
curl -fsSL https://claude.ai/install.sh | bash
```

This lands in `~/.local/share/claude/versions/<version>` with a launcher at
`~/.local/bin/claude`, and adds `~/.local/bin` to your `PATH` in `~/.bashrc`.
Open a new shell (or `exec bash`) and confirm:

```bash
which claude    # want: /home/<you>/.local/bin/claude
```

Status: [VERIFIED -- 2.1.226 installed this way; no Node present on the machine]

## Step 4 -- Clone the repo onto the Linux filesystem

**Keep the repo on the Linux filesystem, NOT under `/mnt/c/`.**
Cross-filesystem I/O on `/mnt/c` is dramatically slower and would thrash
PyTensor's compile cache. Sync with your Windows checkout through GitHub
(push/pull), not by sharing a directory.

```bash
mkdir -p ~/python
git clone https://github.com/jdeast/EXOZIPPy.git ~/python/EXOZIPPy
cd ~/python/EXOZIPPy
```

(Any path under `~` works; `~/python/EXOZIPPy` is what this machine uses.)

A fresh WSL image has **no git identity**, and you only find out when a commit
fails with `Please tell me who you are`. Set it to whatever your other machines
use, so authorship stays consistent:

```bash
git config --global user.name  "Your Name"
git config --global user.email "you@example.com"
```

If you plan to push or open PRs, install the GitHub CLI too. It needs **no
sudo** -- the release tarball drops a single binary into `~/.local/bin`, which
is already on your `PATH` from Step 3:

```bash
V=$(curl -fsSL https://api.github.com/repos/cli/cli/releases/latest \
    | grep -m1 '"tag_name"' | sed 's/.*"v\([^"]*\)".*/\1/')
curl -fsSL "https://github.com/cli/cli/releases/download/v${V}/gh_${V}_linux_amd64.tar.gz" \
    | tar xz -C /tmp
install -m755 "/tmp/gh_${V}_linux_amd64/bin/gh" ~/.local/bin/gh
gh auth login      # interactive: needs a real terminal, see Trap 2
```

Choose HTTPS and let it authenticate git as well; that installs the credential
helper, without which `git push` has no way to authenticate.

Status: [VERIFIED -- gh 2.97.0 installed this way]

## Step 5 -- Poetry + dependencies

Install Poetry with the official installer. Note `python3`, not `python`:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

It installs into `~/.local/bin`, which Ubuntu's stock `~/.profile` already puts
on `PATH` when the directory exists -- so no manual `export PATH` was needed
here. Open a new shell if `poetry` is not yet found, then:

```bash
poetry --version    # 2.4.1 here
```

**Now apply the Trap 1 fix before installing anything.** Poetry defaults to
`virtualenvs.use-poetry-python = false`, meaning it builds project venvs with
whatever bare `python` is on `PATH` -- i.e. the broken Windows `pyenv-win`
shim. Tell it to use the interpreter it runs on instead:

```bash
poetry config virtualenvs.use-poetry-python true
```

Verify it now sees the Linux interpreter, and that the version satisfies
`requires-python = ">=3.12,<3.15"` in `pyproject.toml`:

```bash
poetry env info      # want Base Executable: /usr/bin/python3.14
```

Then install. Take `--extras gui` even if you never open the GUI -- `ruamel-yaml`
lives in that extra and ~30 tests fail at import without it (this is what CI
installs):

```bash
cd ~/python/EXOZIPPy
poetry install --extras gui
poetry run pre-commit install     # optional; matches the README dev setup
```

Without the config change this fails as described in Trap 1 **and still exits
0** -- so check `poetry env info` rather than trusting the exit status.
(Equivalent alternatives, if you prefer: disable `appendWindowsPath` per Trap 1,
or `sudo apt install python-is-python3`.)

The `gui` extra installs cleanly here (PyQt6 6.9.1 against glibc 2.43); its
`<6.10` cap in `pyproject.toml` is a glibc *floor*, so a newer distro is fine.
Installing the extra is NOT an endorsement of the GUI: it is experimental on
every platform, still buggy, and has never been verified driving a real fit --
see README. `exozippy-gui` was never launched on this machine, so nothing here
says whether it works under WSL (it would need WSLg for the desktop window, or
`--browser` plus localhost forwarding). The extra is taken for `ruamel-yaml`,
which ~30 tests import.

Status: [VERIFIED]

## Step 5b -- Give WSL more memory (do this before running the suite)

WSL2 defaults to **50% of host RAM**. On this machine that is 7.6 GiB of
15.75 GiB -- so a workstation gets handed roughly the memory ceiling of a
GitHub runner, which is exactly why `.github/workflows/tests.yml` overrides the
`-n 6` in `pyproject.toml` with `-n 2`.

That ceiling is real here: each worker peaks ~1-2 GB building a `System` and
compiling PyTensor graphs, and a full `-n 6` run drove **available memory down
to 292 MB**. The symptom is never an assertion failure -- it is

```
[gw4] node down: Not properly terminated
```

followed by an `F` for whichever heavy test that worker happened to hold, and
in the worst case xdist then dies in its own rescheduler with
`INTERNALERROR> KeyError: <WorkerController gw11>` (it respawns replacement
workers gw6..gw11, which is the tell) or simply **hangs** -- one run here sat
wedged at 94% for over three hours. `tests.yml` documents the same signature
upstream: "That is a memory ceiling, not a bug."

Create `%UserProfile%\.wslconfig` on the **Windows** side (not inside Ubuntu).
From WSL that path is `/mnt/c/Users/<you>/.wslconfig`:

```ini
[wsl2]
memory=12GB
processors=8
swap=4GB
```

Then `wsl --shutdown` from PowerShell and reopen Ubuntu; **this kills every
running shell**, so do not do it mid-run. Verify with `free -h`.

Size `memory` to leave Windows a few GB (12 GB of this machine's 15.75 GB
leaves ~3.75 GB). It is a ceiling, not a reservation -- WSL reclaims lazily --
so the cost of setting it too high is Windows swapping, not WSL wasting RAM.

If you would rather not raise the ceiling, run the suite at `-n 2` like CI
does. Either way, **always run the suite under a timeout** so a wedged xdist
cannot silently eat an afternoon:

```bash
timeout 3000 poetry run pytest -q -rf -n 6
```

Use `-rf` too: with a bare `-q` the failures print as bare `F` characters with
no test names, which is useless for diagnosis.

Status: [VERIFIED -- 292 MB and later a 41 MB floor measured at -n 6 on the
7.6 GiB default. `.wslconfig` is written on this machine but NOT yet applied:
applying it needs `wsl --shutdown`, so the numbers in "Known test results"
below are all still from the 7.6 GiB default.]

## Step 6 -- Launch Claude Code and verify

From inside the repo in the Ubuntu shell:

```bash
claude
```

Verify the environment, cheapest check first. A good smoke test is that
PyTensor can actually compile C -- the thing most likely to be broken:

```bash
poetry run python -c "
import pytensor, pytensor.tensor as pt, numpy as np, os
x = pt.dvector('x'); f = pytensor.function([x], (x**2).sum())
print('compiled ok ->', f(np.arange(5.0)), '| cxx =', pytensor.config.cxx)
print('has os.fork:', hasattr(os, 'fork'))
"
```

Expect `compiled ok -> 30.0 | cxx = /usr/bin/g++` and `has os.fork: True`.
Then a fork-dependent test module, then the full suite:

```bash
poetry run pytest tests/test_ptde.py -n0 -x        # 28 passed in ~25 s here
timeout 3000 poetry run pytest -q -rf -n 6         # full suite; see Step 5b
```

The suite takes **~18-20 minutes** on this machine (8 cores), not the ~10
minutes quoted in `CLAUDE.md` for a workstation. Do not set a short timeout,
but do set the long one from Step 5b.

Status: [VERIFIED -- see "Known test results on this machine" below]

## Known test results on this machine

Full suite, `timeout 3000 poetry run pytest -q -rf -n 6`, on master at
`24f85cf` with `--extras gui` installed and the fix from PR #92 applied:

```
3 failed, 1260 passed, 84 warnings in 1567.57s (0:26:07)
```

All three remaining failures are the **memory-pressure class** described in
Step 5b, not bugs: `test_multiplanet.py::test_two_planet_start_is_finite`,
`test_rossiter.py::test_rm_system_with_linear_ld_builds` and
`test_sed_flux_constraints.py::test_astrometry_fluxfrac_derived_from_sed`. All
three **pass in isolation** (`3 passed in 289.78s`). An earlier run also lost
`test_run_endpoints.py::test_endpoint_run_lifecycle_start_sampling_stop`, whose
child fit died with `returncode: -11` (SIGSEGV) after a 240 s graceful-stop
timeout while five other workers competed for the last few MB; it too passes
alone. Available memory bottomed out at **41 MB** in that run, so it survived
on luck. Do Step 5b, then re-run any named failure alone before believing it.
[VERIFIED both ways]

### The one real bug this platform found (fixed: PR #92)

`tests/test_runaway_logp_regression.py::test_good_draw_logp_matches_deterministic_build`
failed here with `got -113614.6510, expected -945.5716` while CI was green on
the same commit. It was **not** a WSL bug -- it was a latent determinism bug in
EXOZIPPy that this machine happened to expose:

`ConfigManager` discovered every `symbolic_physics.py` and `defaults.yaml` with
`Path.rglob`, which yields **filesystem directory order** -- stable on one
machine, different between machines (ext4's hashed btree vs xfs/NFS). That order
sets `all_relations` order, which sets the order the relaxation engine visits
equations, which decides which member of a *symmetric* relation pair it solves
for. Two `mulensing` relations are symmetric under swapping their pair, so
nothing in the equation breaks the tie. The result was three transposed pairs
-- `(pm_ra, pm_dec)`, `(mu_ra_rel, mu_dec_rel)`, `(pi_E_N, pi_E_E)` -- and
`t_E` coming out 11.54 d instead of 18.29 d: two different physical models from
one config.

Two things worth carrying forward:

- It **survived `PYTHONHASHSEED` randomization** (identical to four decimals
  across five seeds and two commits), because filesystem order is stable per
  machine. Reproducibility on one box proves nothing about determinism.
- **CI structurally cannot catch this class of bug**, because CI only ever runs
  one filesystem layout. Comparing two machines is the only detector, which is
  what `scripts/diag_mulens.py` exists for.

The arbitrary choice is now consistent but still arbitrary -- the seed carries
no `pi_E`, so the engine is inverting an underdetermined system. Tracked in
issue #93.

## Performance note -- PyTensor cannot find a BLAS

Every model build here emits:

```
UserWarning: PyTensor could not link to a BLAS installation. Operations that
might benefit from BLAS will be severely degraded.
```

**Why it happens** (`pytensor.config.blas__ldflags` is the empty string):

- numpy 2.4.6 from PyPI carries its BLAS *inside the wheel*, as
  `site-packages/numpy.libs/libscipy_openblas64_-32a4b2a6.so` -- a
  hash-mangled soname.
- numpy's recorded BLAS `lib directory` is the manylinux **build** path
  (`/opt/_internal/cpython-3.14.0/lib/python3.14/site-packages/scipy_openblas64/lib`),
  which does not exist on your machine.
- PyTensor's autodetection tries to reuse whatever BLAS numpy links. It cannot
  turn a mangled soname at a nonexistent path into usable `-L/-l` flags, so it
  gives up and emits the warning.
- There is no system BLAS to fall back on: a stock Ubuntu has no
  `libblas`/`libopenblas` in `/usr/lib/x86_64-linux-gnu`.

**This is not WSL-specific**, and it is not a correctness problem -- it is the
normal consequence of a pip/Poetry install (conda links a real shared BLAS,
which is why the warning names conda). By the same reasoning CI hits it too,
since CI installs via Poetry from the same lock file.

Two fixes. Preferred, durable -- install a system BLAS:

```bash
sudo apt install -y libopenblas-dev
```

If the warning persists after that, pin the flags in `~/.pytensorrc`:

```ini
[blas]
ldflags = -lopenblas
```

Alternative needing **no sudo** -- point PyTensor at numpy's own bundled
OpenBLAS (verified here: the warning goes from 1 to 0 and the test still
passes). Put it in `~/.pytensorrc`, substituting your venv path and the exact
filename from `ls $(poetry run python -c "import numpy,os;print(os.path.dirname(os.path.dirname(numpy.__file__)))")/numpy.libs`:

```ini
[blas]
ldflags = -L/path/to/venv/lib/python3.14/site-packages/numpy.libs -l:libscipy_openblas64_-32a4b2a6.so
```

Note the GNU-ld `-l:` exact-filename form -- a plain `-lscipy_openblas64_` will
not match the hashed name. The drawback is that the hash changes with every
numpy version, so this silently reverts to no-BLAS after an upgrade; prefer
`libopenblas-dev`. Either way you then have two OpenBLAS copies in one process
(numpy's and PyTensor's), so leave `conftest.py`'s thread pinning alone.

Status: [VERIFIED -- cause diagnosed; the no-sudo fix tested end to end
(`tests/test_chen.py::test_rv_only_system_gets_the_chen_potential`: warning
1 -> 0, still passes). Wall-clock benefit not yet measured.]

## Notes / gotchas

- Access your Windows files from WSL under `/mnt/c/...` if you ever need to
  copy something across, but do not run the project from there.
- Access your WSL files from Windows via the `\\wsl$\Ubuntu\home\<user>\` path
  or `explorer.exe .` from inside the repo.
- `apt` may print `Failed to connect to system scope bus` / `Transport endpoint
  is not connected` while processing `systemd` triggers. Harmless under WSL --
  there is no full systemd session; the packages install correctly.
- The default `-n 6` in `pyproject.toml`'s `addopts` wants ~6 usable cores.
  This machine reports 8 to WSL, so the default is fine; on a smaller WSL
  allocation, lower it or pass `-n0`. Memory here is 7.6 GiB total, and WSL
  sizes its VM from the Windows host -- tune via `%UserProfile%\.wslconfig` if
  the suite gets OOM-killed.
