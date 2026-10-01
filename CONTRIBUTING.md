# Contributing to EXOZIPPy

This guide is for changing EXOZIPPy itself. To only run fits, follow
"Installing" in [`README.md`](README.md) instead.

## 1. Development install

Developers use [Poetry](https://python-poetry.org), which installs the
pinned `poetry.lock` so everyone works against the same dependency set
(otherwise your contribution may pass locally but be rejected remotely
because of dependency mismatches).

1. Do Step 1 of "Installing" in [`README.md`](README.md) (Miniforge, and
   on macOS the Xcode tools). On Windows, first set up WSL2 with
   [`WINDOWS_INSTALL.md`](WINDOWS_INSTALL.md) and read
   [Developing under WSL2](#developing-under-wsl2) below; on an Intel Mac,
   read [Developing on an Intel Mac](#developing-on-an-intel-mac).

2. Create a Python 3.12 environment (on Linux, with the compiler; drop
   `gxx openblas` on macOS) and install Poetry. Keep this environment
   activated whenever you work: it is what puts the compiler on your PATH.

   ```bash
   conda create -n exozippy-dev python=3.12 pip gxx openblas
   conda activate exozippy-dev
   curl -sSL https://install.python-poetry.org | python3 -
   ```

3. Clone and install, then install the git hooks (section 2):

   ```bash
   git clone https://github.com/jdeast/EXOZIPPy.git
   cd EXOZIPPy
   poetry env use python3.12
   poetry install --extras gui
   poetry run pre-commit install
   ```

   Take `--extras gui` even if you never use the GUI: ruamel-yaml lives in
   that extra, and ~30 tests fail without it.

After every `git pull`, sync to the pulled lock file:

```bash
poetry install --extras gui
```

Do not use `poetry update` for that; it re-resolves everything and rewrites
`poetry.lock`. Use it only when you mean to upgrade dependencies, and commit
the new lock file.

Other everyday commands:

```bash
poetry run exozippy fit.yaml        # run anything inside the environment
poetry add <package>                # add a dependency
poetry add --group dev <package>    # add a development-only dependency
```

### Developing on an Intel Mac

Plain `poetry install` fails on an Intel (x86_64) Mac. Use Python 3.12 or
3.13, and in place of `poetry install --extras gui` run:

```bash
./scripts/bootstrap_intel_mac.sh
```

`tests/test_gp.py` reports 10 skips on this platform; that is expected.

### Developing under WSL2

- **Windows PATH shadows Linux tools.** WSL appends the Windows `PATH`, so a
  bare `python` or `pip` can be a Windows executable, and `poetry install`
  then installs nothing yet exits 0. Keep the conda environment activated
  and check that `poetry env info` shows a Linux path, not `/mnt/c/...`.
  Without conda, run `poetry config virtualenvs.use-poetry-python true`
  first.
- **Set a git identity**; a fresh WSL image has none:
  `git config --global user.name "..."` and `user.email "..."`.
- **Give the test suite memory.** Six test workers need more than WSL's
  default 50% of RAM on a 16 GB machine; the symptom is `[gwN] node down`
  or a hang, not a test failure. Raise it with a `.wslconfig`
  (`WINDOWS_INSTALL.md`, Step 2) or lower `-n`
  (`python3 scripts/pytest_workers.py --explain` suggests a value).

If every model build warns that PyTensor could not link to a BLAS (any
pip or Poetry install can), fits are slower but correct. Fix it with
`sudo apt install -y libopenblas-dev`.

## 2. Git hooks

`pre-commit install` sets up two hooks:

- **On commit:** ruff sorts imports and formats your staged files. If it
  changes anything the commit aborts; `git add` the result and commit again.
- **On push:** the full test suite (~20 minutes). A failure aborts the push.

If you installed the hooks before the push hook existed, run
`poetry run pre-commit install` again; without it the push hook never runs.

Do not run `pre-commit autoupdate`: the ruff version is pinned so everyone's
formatting is byte-identical. Let a hook finish rather than Ctrl-C it --
pre-commit stashes your unstaged changes while it runs.

```bash
poetry run pre-commit run --all-files                          # commit hooks, whole repo
poetry run pre-commit run --hook-stage pre-push --all-files    # the push hook
```

## 3. Style

- **Formatting and imports:** ruff, through the hooks, at line length 79.
  Run it through `pre-commit run`, never a bare `ruff format` -- ruff is
  deliberately not a project dependency, and a bare run also reformats the
  code blocks in the markdown docs.
- **Lint:** a deliberately narrow rule set (`[tool.ruff.lint]` in
  `pyproject.toml`, which also says why each excluded rule is off). Never
  widen the hook's `--fix` beyond import sorting: with unused-import checks
  on, ruff would delete the `from . import physics` lines that look unused
  but register each component's physics.
- PEP 8; no type hints; `CamelCase` classes, `snake_case` everything else.
- Constants live in `constants.py`, in `UPPER_SNAKE_CASE`.
- Google-style docstrings (`Args:`, `Returns:`) for public modules, classes
  and functions.
- Plain ASCII in code, comments and docs (see `CLAUDE.md`).

`git config blame.ignoreRevsFile .git-blame-ignore-revs` makes `git blame`
skip the two whole-tree reformatting commits.

## 4. Tests

Tests use long, descriptive names, Given/When/Then docstrings and
Arrange/Act/Assert layout. Every bug fix adds a test that fails before the
fix and passes after it. [`docs/testing.md`](docs/testing.md) is the suite's
runbook.

## 5. Workflow

`master` is protected for everyone, including the owner: every change goes
through a pull request, which merges only when the `test` CI check passes.
No force-pushes.

```bash
git checkout -b some-change
# ... work, commit ...
git push -u origin some-change
gh pr create --fill
gh pr merge --auto --squash      # merges by itself when CI goes green
```

Open a GitHub issue for anything worth a durable record (a design decision,
a reported bug, multi-step work); a trivial fix needs only a good commit
message. Without write access, work from a fork the same way.

## 6. Releases

The version comes from the git tag; there is no version string to edit
(leave the `version = "0.0.0"` placeholder in `pyproject.toml` alone). To
release, push a `v`-prefixed tag:

```bash
git tag v0.1.0
git push origin v0.1.0
```

`.github/workflows/publish.yml` then tests, builds and publishes to PyPI.
Run that workflow manually with the TestPyPI option to rehearse. Never put
two tags on one commit.

## 7. AI use

AI use is encouraged. Claude Code, using the Opus 5/5.5 and Fable 5.1
models, has written substantial parts of this repo; see `CLAUDE.md`. Any
pull request will be reviewed by Claude and must pass a rigorous suite of
3000+ unit tests before merging to master.
