Development Setup & Guidelines

Welcome to the project! This guide outlines the tools and standards we use to keep our codebase clean, consistent, and reliable. We use Poetry for dependency management and pre-commit to automate our style and quality checks.

# 1. Dependency Management with Poetry

We use Poetry to manage dependencies, virtual environments, and packaging. It replaces pip and requirements.txt with a more robust pyproject.toml file.
Installation & Setup

First, we need to be sure we're on the same version of python (we'll test version-related bugs separately and explicitly):

    pyenv install 3.12.13

You might need to install pyenv first:

MacOS:

    brew install pyenv


If you don't have Poetry installed, install it globally using the official installer:
Bash

    curl -sSL https://install.python-poetry.org | python3 -

Download EXOZIPPy (if you haven't already):

    git clone https://github.com/jdeast/EXOZIPPy.git

Navigate to the project root and install the dependencies. This will automatically create a virtual environment for you:
Bash

    poetry env use python3.12

    poetry install --extras gui

`--extras gui` is worth taking even if you never open the GUI: ruamel-yaml lives
in that extra, and without it roughly 30 tests fail at import. CI installs it
for exactly that reason.

When you git pull, be sure to install any new dependencies:

    git pull
    poetry update

To run a python script inside the poetry environment (without activating it):

    poetry run python main.py

If you need to add a new dependent package:

    poetry add <package>

To add a new development dependency:

    poetry add --group dev pytest

Update dependencies:

    poetry update

# 2. Automated Checks with Pre-commit

We use pre-commit to enforce style guidelines and catch common errors before
they reach the repository. It runs at two different moments, and the split is
deliberate.

Installation & Setup

Pre-commit is included in our development dependencies. Once you have run
"poetry install", you just need to install the git hooks:

    poetry run pre-commit install

You should see it report that hooks were installed at BOTH .git/hooks/pre-commit
and .git/hooks/pre-push.

IF YOU ALREADY HAD PRE-COMMIT INSTALLED, RUN THAT COMMAND AGAIN ANYWAY. The
config gained a pre-push hook, and which hook files exist on disk is decided
when you run `pre-commit install`, not when you pull. Pulling the new config
alone leaves you with no .git/hooks/pre-push, and the test hook silently never
fires. Re-running is harmless if you're already set up.

Do NOT run `pre-commit autoupdate`. The tool versions in
.pre-commit-config.yaml are pinned on purpose (see section 3).

How it Works

On `git commit`, over your staged files:

    isort, then black -- these rewrite your files rather than just complaining.
    If either changes anything the commit aborts; `git add` the result and
    commit again. Fast, a second or two.

On `git push`, over the whole repo:

    the full pytest suite, ~3 minutes locally. A failure aborts the push.

Tests run on push rather than on commit because paying three minutes per commit
is what pushes people toward giant commits and habitual --no-verify. Per push
it's affordable.

One hazard worth knowing: pre-commit stashes your unstaged changes while hooks
run and restores them afterwards. Killing a hook mid-run can leave that stash
unrestored. The work is recoverable from the patch it prints under
~/.cache/pre-commit/, but let a run finish rather than Ctrl-C it.

Common Pre-commit Commands (bash)

Run checks manually on all files:

    poetry run pre-commit run --all-files

Run checks on specific files:

    poetry run pre-commit run --files src/my_script.py

Run the push-stage hook without pushing:

    poetry run pre-commit run --hook-stage pre-push --all-files

# 3. Style Guidelines

We follow standard Python conventions to ensure readability and maintainability.

ruff does all three jobs

You rarely need to worry about formatting code manually. One tool -- ruff --
handles linting, import sorting and formatting, via two pre-commit hooks:

    ruff-check: sorts imports (rule `I`, replacing isort) and reports bugs.
    Its --fix is scoped to `--fixable I`, so it may reorder imports and
    nothing else.

    ruff-format: the formatter, replacing black. Note line-length = 79, NOT
    the default 88 -- it matches the width this codebase's comments and
    docstrings already wrap at. If ruff formats it, that is the standard.

ruff-check runs before ruff-format: the linter decides which lines exist, the
formatter decides how they wrap.

The rev is pinned exactly in .pre-commit-config.yaml and must stay that way.
Everyone has to produce byte-identical output, or merging a long-running branch
conflicts on formatting rather than on content. Keeping one tool aligned across
contributors instead of three is most of why ruff replaced black and isort.

`ruff format` is used, but there is deliberately NO second formatter. Do not
add [tool.black] or [tool.isort] back.

The lint rule set (pyproject.toml, [tool.ruff.lint]) is deliberately narrow:
undefined names, syntax errors, `is` against a literal, broken format strings,
pylint's error category, and import order. Every one had zero violations when
adopted, so it exists to stop regressions rather than to relitigate style.
pyproject.toml lists the rules that are switched OFF and why -- including
several that look useful but produce only false positives here -- so read that
before widening the set.

Unused imports (F401) are NOT checked yet. Enabling that rule needs care: the
`from . import physics` lines in eight components look unused but populate
PHYSICS_REGISTRY via @register_physics, and ruff considers them auto-fixable.
Never broaden the hook's --fix beyond `I`.

Note on `git blame`: two commits reformatted the tree wholesale -- 536c2da
(black and isort, 176 files) and the ruff-format adoption (49 files) -- so
naive blame attributes many lines to them. .git-blame-ignore-revs fixes this.
GitHub's blame view honours it automatically; locally it takes one opt-in per
clone:

    git config blame.ignoreRevsFile .git-blame-ignore-revs

General Conventions:

    PEP 8: We adhere to PEP 8 standards.

    No Type Hinting

Naming Conventions:

    CamelCase for Classes.

    snake_case for variables, functions, and methods.

    All constants should be defined in/imported from constants.py and use UPPER_SNAKE_CASE.

Docstrings: Write docstrings for all public modules, classes, and functions using the Google Style. Briefly explain what the function does, its arguments (Args:), and what it returns (Returns:).

# 4. Unit testing:

Following python conventions, this test suite
  - uses long, explicit function names to describe the test
  - Has Given/When/Then doc strings
  - follows the AAA (Arrange, Act, Assert) organizational scheme

Developers are forced to pass all unit tests before pushing a commit via git
hooks -- and, since the hook can be skipped with --no-verify or simply never
installed, again on the server before anything can reach `master` (section 5).

Anytime a developer fixes a bug, a new unit test following the above convention should be added.
The test should demonstrate failure before the fix and success after the fix.

# 5. Project stage & workflow

EXOZIPPy is pre-1.0 and not yet formally released. Expect breaking changes and
rough edges; that's normal at this stage, not a bug in the process.

`master` is protected by a GitHub ruleset. There are no bypass actors -- this
applies to every contributor including the repository owner. Concretely:

- You cannot push to `master`. Every change, including a one-line typo fix,
  goes through a pull request.
- A PR cannot merge until the `test` check passes. That check is pinned to
  GitHub Actions, so it can only be satisfied by a real CI run.
- No force-pushes and no deleting `master`. Anything that lands is undone by a
  revert commit, never by rewriting history.
- Reviews are NOT required (0 approvals). Self-merging your own PR is expected
  and fine. The gate is CI, not gatekeeping by a human.

The normal loop:

    git checkout -b some-change
    # ... work, commit ...
    git push -u origin some-change     # feature branches are unrestricted
    gh pr create --fill
    gh pr merge --auto --squash

`--auto` is the part that makes this cheap: it queues the merge and it fires by
itself the moment CI goes green, so you don't sit watching a 15-minute run. The
branch is deleted automatically on merge.

Other workflow notes:

- Open a GitHub issue for anything worth a durable record: a design decision,
  a bug a student/collaborator reports, or multi-step work you might pause or
  hand off. Don't feel obligated to file one for every change — a good commit
  message already covers most of what a trivial-fix issue would say.
- Every push and PR against `master` runs the test suite via GitHub Actions
  (`.github/workflows/tests.yml`). CI overrides the `-n 6` in pyproject's
  addopts with a count computed from the runner it landed on --
  `min(cores, memory_gb // 3)`, via `scripts/pytest_workers.py`. Six workers
  suit a workstation but exhaust a GitHub runner's memory, which shows up as
  "worker 'gwN' crashed" on whichever heavy test drew the short straw rather
  than as an honest failure. The count is computed rather than written down
  because the `-n 2` it replaced was right when a hosted runner had two cores
  and then quietly stopped being right; the job logs the cores and memory it
  saw, so check there rather than assuming a number.
- CI also splits the suite across **4 shards**, so `pytest (ubuntu-latest,
  3.12, 1)` runs a quarter of the test files and `... 2)`, `... 3)`, `... 4)`
  the rest. The required `test` check aggregates every combination, so nothing
  about that changes what you wait for. If you are reproducing one shard
  locally, `python scripts/pytest_shard.py --shard 1 --of 4` prints its file
  list.
- The shards are balanced from `tests/durations.json`. You do not need to
  update it when you add a test -- an unrecorded file is charged the median
  cost -- but if CI warns that a large share of files are unrecorded, it wants
  regenerating; `docs/testing-cache.md` has the command.
- Contributing from a fork works the same way, and is the right approach if you
  don't have write access. Note that auto-delete-on-merge cannot reach a fork's
  branches, so clean those up yourself after merge.

If CI is ever broken for reasons unrelated to your change (a runner outage, an
Actions incident), nothing can merge and no one can override it. The escape
hatch is to set the ruleset to `evaluate` or `disabled`, merge, and re-enable
it. That is a deliberate cost of having no bypass actor, and it should be rare.

# 6. Versioning & releases

There is no version string anywhere to bump. `pyproject.toml` declares
`dynamic = ["version"]`, and poetry-dynamic-versioning derives the version from
`git describe` at build time. (The `version = "0.0.0"` under `[tool.poetry]` is
a placeholder poetry-core insists on -- never edit it.) A release is a tag and
nothing else.

To cut a release, push a `v`-prefixed tag:

    git tag v0.1.0
    git push origin v0.1.0

That triggers .github/workflows/publish.yml, which runs the full test suite,
builds the sdist and wheel, checks that the built version matches the tag, and
publishes to PyPI, then opens a GitHub Release. Publishing uses PyPI Trusted
Publishing (OIDC) -- there is no API token in the repository secrets, and the
trust is registered against the workflow's filename, so renaming publish.yml
breaks releases until PyPI is updated to match.

To rehearse without releasing, run the workflow manually from the Actions tab
with the TestPyPI option; that builds and verifies without touching PyPI.

Never put two tags on one commit -- setuptools_scm and poetry-dynamic-versioning
both pick one arbitrarily and you get a build labelled with the wrong version.

Installed releases so far: 0.1.0rc1 on PyPI (`pip install --pre exozippy`).

# 7. AI use:

AI use is encouraged, but thorough review and testing is essential. Create unit tests that verify/confirm the output for all essential code (see above). Unit testing is especially critical with AI generated code, as it is often tunnel visioned and drops important features not relevant to the bug.

scripts/dump_code.py will collate the entire repo into a copy/pasteable file for AI review. Note that, anecdotally, ~1M tokens is suffificient to keep ~5000 lines of code in context sufficient for a deep, logically review. Beyond that, it tends to lose its focus and forget aspects of the code. The current code base is a bit larger than that, so you may wish to filter the repo dump for more targeted, relevant advice. 

Useful prompts: 
    
    here is the code dump. identify any inconsistent use of functions, style, variable names, etc. Identify any AI comments irrelevant to the final code base. Flag obsolete code. Suggest revisions for speed, clarity, standardization, and readability. Identify arbitrary fallbacks designed to make the code run at all costs, when that cost is producing garbage [paste code dump]

    here is the code dump. write unit tests to protect functionality not currently covered [paste code dump with -c all flag, which includes the unit test directory]