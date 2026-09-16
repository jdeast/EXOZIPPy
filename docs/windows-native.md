# Native Windows: the known blockers

**Windows is supported through WSL2, and that is the supported path.** Inside
WSL2 you are running a real Linux kernel with a genuine Ubuntu userspace, so it
*is* the Linux platform -- `fork`-based samplers and all. See the "Windows"
section of `README.md` and the step-by-step runbook in `WINDOWS_INSTALL.md`.

This file is the other question: **what would it take to run EXOZIPPy on
Windows natively, without WSL2?** It exists so that anyone who wants to take
that on can see the minimum work up front rather than discovering it one
failure at a time. It is a running list -- add to it whenever you find or fix
something platform-specific, and delete an entry only when it is actually
fixed and verified on a Windows machine.

Nobody is working on this. Patches welcome.

## Status

Native Windows has never worked and is not tested. A CI job existed and was
removed on 2026-07-28 (see `.github/workflows/tests.yml`); it never once
produced a result, green or red. Three real bugs it *did* find were fixed and
stay fixed: an unsatisfiable mkl pin, a POSIX-only `SIGALRM` in the symbolic
solver, and the GUI's cross-process stop signal.

## Blockers

### 1. The PTDE samplers cannot start their workers (hard)

`src/exozippy/samplers/_common.py` builds its worker pool with
`mp.get_context("fork")`, and so do `run.py:1704` and
`outputs/evidence.py:575`. The `fork` start method is what lets a child
inherit the compiled PyTensor logp function *without pickling it* --
cloudpickle cannot serialize one. Windows has no `fork`, only `spawn`.

`ptde_async` is the default sampler, so this is not a corner case: it is the
main path.

Minimum work: a spawn-safe worker that rebuilds the model in each child from
the config (paying the compile cost per worker, once), or a threaded evaluator
that releases the GIL. Neither is a small change, and the second interacts
with the VBMicrolensing GIL question. Note that `gui/tune.py:354` already uses
`spawn` deliberately, so the pattern exists in the codebase.

### 2. PyTensor needs a GCC-style C++ compiler (hard, untested)

PyTensor compiles C modules at runtime. MSVC will not do; it wants a
GCC-style toolchain (`m2w64-toolchain` / mingw-w64). We have never tested
this. If it does not work, nothing else on this list matters -- every model
falls back to the slow Python path at best.

### 3. The test suite's timeout mechanism is POSIX-only (hard, suite only)

`pyproject.toml` pins `timeout_method = "signal"`, which is `SIGALRM`, which
Windows does not have. The pin is not incidental: pytest-timeout's `thread`
method ends in `os._exit()`, which discards the timeout banner and the stack
dump, so a thread-method timeout tells you nothing about *what* hung. It also
cannot interrupt a process blocked in a `gcc` subprocess -- which is the exact
failure the 600-second `timeout` is sized around.

Library code is already clean here: `config.py:123` gates every `SIGALRM` use
behind `_HAS_SIGALRM = hasattr(signal, "SIGALRM")` and no-ops where the signal
is absent. This blocker is the *suite*, not the package.

Minimum work: a Windows-only timeout strategy that still identifies the
hanging test, or accept an untimed suite there and lose that diagnostic.

### 4. The suite has never finished (hard, suite only)

Over 90 minutes against ~16 on Linux at the time, cancelled at the CI cap on
every attempt. Adding a Windows job back means first making the suite *finish*
there -- not raising the cap until it fits. See the comment block at
`.github/workflows/tests.yml:57`, which spells this out.

### 5. Cache and data paths assume XDG (easy)

`utilities/zenodo.py:137` `shared_cache_root()` resolves
`$XDG_CACHE_HOME/exozippy`, falling back to `~/.cache/exozippy`. **XDG is not
a Windows concept.** There is no `XDG_CACHE_HOME` there, and the fallback
creates a Unix-shaped `~/.cache` in a Windows home directory. The Windows
equivalent is `%LOCALAPPDATA%` (`%APPDATA%` for the roaming variant), so this
wants a `sys.platform` branch rather than another fallback.

`EXOZIPPY_CACHE_DIR` already overrides everything, so there is a workaround
today; the point is that the *default* is wrong.

The same applies to the test-suite compiledir in `conftest.py`, which uses
`Path.home() / ".pytensor-pytest"`. That one is merely un-Windows-like rather
than broken.

### 6. Two MIST download paths write inside the installed package (easy)

Wrong on Linux too, not only here.

This is tracked as review item 1.9.7 and is not Windows-specific, but Windows
makes it bite harder, because a `Program Files` install is read-only by
default in a way a Linux site-packages often is not.

- `models/MIST/download_MIST_EEPs.py:14` --
  `download_tarfiles(url, dest_folder="temp_files")` is a **relative** path,
  so the tarball lands in whatever directory the process happens to be
  standing in, while the extractor at line 71 looks under `EEP_PATH`
  *inside the package*.
- `models/MIST/eep_grid.py:100` fetches into `EEP_GRID_DIR =
  current_dir / "MISTv2.5" / "EEPs"`, also inside the package.

Both should resolve through `shared_cache_root()` -- which then makes item 5
above load-bearing rather than cosmetic.

### 7. Never audited (unknown)

Nothing below has been looked at even once. Listed so that "it is not on the
list" does not read as "it is fine".

- Path separators and `Path` round-tripping through the YAML config, and
  whether a user-written Windows path survives `ConfigManager`.
- Case-insensitive filesystems -- filter names, component names and element
  names are matched case-sensitively in places.
- The GUI's subprocess isolation and its stop signal, on the spawn path.
- Long-path limits (260 characters) against pytensor's content-hash
  compiledir names.
- Line endings in the data readers.
