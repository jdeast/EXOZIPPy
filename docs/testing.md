# Testing conventions

Tests follow AAA (Arrange / Act / Assert) with Given/When/Then docstrings. All tests that use `System` must call `system.prepare()` before `system.build_model()`. RA/Dec user params are in **degrees** (the default unit); `Parameter.__post_init__` converts to radians internally.

Do not start the full suite with a timeout. Start it and poll.

Testing note: build relation inputs with `pt.dscalar`, **not** `pt.as_tensor_variable(<python float>)` -- pytensor autocasts a bare Python float to the smallest dtype that represents it (5778.0 -> float32), and a unary op like `pt.log10` on it then computes in float32, silently losing ~1e-7. The model always feeds float64. `tests/test_torres.py` pins the port against real IDL output from `massradius_torres.pro`.

## The pre-push hook, and why it does not say `poetry run pytest`

The full suite runs on push, wired in `.pre-commit-config.yaml` (install both hook
types with `poetry run pre-commit install`). The entry is
`scripts/pre_push_suite.sh` rather than `poetry run pytest`, because that spelling
**cannot work from a git worktree** -- and essentially all work here is developed in
one, so it was on the path of every push. Two failures, one of them silent:

- **Loud.** Poetry names a project's virtualenv from a hash of the project *path*. A
  worktree is a different path, so `poetry run` there does not find the populated venv;
  it creates a fresh empty one and dies on `ModuleNotFoundError: pytest`, leaving an
  orphan venv behind (these have reached tens of GB).
- **Silent, and the one that matters.** Even once pytest runs, the main venv's editable
  install is a plain path entry (`exozippy.pth`) pointing at the *shared* checkout's
  `src`. So the suite imports the other tree's code and reports green -- a hook that
  proves nothing about what you are pushing.

The script resolves the interpreter and exports `PYTHONPATH=<this tree>/src` itself, so
the hook tests the tree it was launched from **by construction** rather than by the
caller remembering to export something. It prints the tree, the interpreter and the
import path, because the bug it replaces was a hook that passed while testing the wrong
code. `tests/test_pre_push_hook.py` pins the worktree case, the main-tree case (git
reports `--git-common-dir` relative there and absolute from a worktree), and the wiring.

Two rejected alternatives, so they are not re-proposed:

- `POETRY_VIRTUALENVS_CREATE=false`, which makes `poetry run` resolve from its own base
  environment. It fixes the loud half and leaves the silent half live.
- A per-worktree symlink or a hand-exported `PYTHONPATH`. Both work and both are
  invisible: the failure mode is a green hook, so a mitigation nobody can see in the
  repository is not a mitigation.

Knobs: `EXOZIPPY_VENV_PYTHON` names the interpreter explicitly and skips the poetry
lookup (for a conda or hand-built environment, or CI); `EXOZIPPY_PREPUSH_DRYRUN=1`
prints the resolution and exits without running the suite.

Two properties of the hook that this did **not** change, and that still bite:

- It tests the **working tree**, not the commits being pushed. Pushing a branch from
  the main tree therefore tests whatever is checked out there, not the branch.
- `pre-commit` stashes unstaged changes while hooks run and restores them afterwards.
  Do not kill a run in progress; the work is recoverable from the patch it prints under
  `~/.cache/pre-commit/`, but only by hand.

## Suite runtime and the pytensor compile cache

The suite runs in **~16 minutes warm** on an idle 36-core box (`-n 6`, 3108 tests,
measured 2026-08-19 at 2977 tests / 11:37 and 2026-08-25 at 3052 / 15:47). A cold run is
**~25 minutes** and happens once per fresh checkout or worktree, and on CI until its
compiledir cache is populated. The runbook -- what the cache is, how it is bounded, how
to reclaim space, and how to measure a run honestly -- is in `docs/testing-cache.md`.
Read that before changing anything about the compile cache or the suite's timing.

### Where the time actually goes (measured 2026-08-25, review item 6.13.1)

One `--durations=0` run, 3105 passed / 3 skipped, **6715 worker-seconds** of measured
test time. The two numbers that decide what is worth optimizing:

| | worker-seconds | share |
|---|---|---|
| `call` (test bodies) | 5690 | **84.7%** |
| `setup` (fixtures, i.e. shared `System` builds) | 1025 | 15.3% |
| `teardown` | ~0 | 0.0% |

So the cost is in test BODIES, not in fixture construction. Two hypotheses that look
obvious and are wrong:

- *"Consolidate the files that share a config so a module fixture builds the System
  once."* The 13 kelt4 files, 8 KMT, 7 DC2018_128 and 5 ob161003 do each build their
  own -- but all `setup` everywhere totals 15%, so this bounds the whole prize at well
  under that, and it costs parallelism to collect it.
- *"The `--dist loadfile` tail serializes the run."* The slowest single file is
  `test_rm_ltt.py` at 338 s against an ideal `-n 6` wall of 1119 s, and **no file
  exceeds the ideal wall**. The distribution is not the constraint, and adding workers
  still scales nearly linearly -- which is the change CI got.

It is a long tail rather than a few hot spots: the top 30 of 202 files are 67% of the
total, the worst single file is 5.0%. That shape is why CI splits the suite across
**4 shards** (`scripts/pytest_shard.py`, packing longest-file-first from
`tests/durations.json`, measured at 1.00x of ideal balance): with no dominant file there
is nothing to cut, so the remaining lever is more machines.

Four is where that lever runs out. `--dist loadfile` pins a file to one worker, so a
shard cannot beat its slowest file's serial time -- past 4 shards the binding constraint
stops being the spread and becomes `test_rm_ltt.py` alone. Below ~8 minutes the next
move is splitting slow FILES, which is exactly why `test_runner_lifecycle.py` was split
out of `test_runner.py`. See `docs/testing-cache.md` for the full arithmetic, the
sharding, and the compiledir seeding that keeps its cache affordable.

### Looking for tests to cut: `scripts/find_redundant_tests.py`

Before proposing a deletion, run it, and read what it says about its own
limits. It ranks expensive tests by how little unique line coverage they
contribute and names the single other test that dominates each one -- the
strongest signal available from a machine, and still only a candidate list.

**Coverage overlap is not redundancy.** Two tests can execute exactly the same
lines and assert entirely different properties, and in this codebase that is
the norm. Run over the whole suite on 2026-08-25 it flagged 12 of 62 expensive
tests, and **none survived review**: 11 were dominated by a peer in the same
file (5 mutually, meaning identical coverage -- several tests building one
model and checking different things about it), the most expensive was the
known JAX blind spot where both backends run the same graph-construction lines,
and the one cross-file hit was a coincidence involving
`test_shipped_example_prepares`, which runs `prepare()` on every shipped config
and so dominates plenty while testing none of it.

Two traps it now guards, both of which produced confidently wrong answers
first:

- **Cost must be per TEST, not per file.** Weighted by file, every test in
  `test_vcve.py` was charged that file's 208 s and a ten-line, zero-second
  numeric check topped the list of things to delete.
- **Coverage cannot see a subprocess.**
  `test_run_lifecycle_status_snapshot_and_graceful_stop` costs 128 s and covers
  **74 lines**, because its work happens in a `python -m exozippy.cli` child
  that is not traced. It looked maximally redundant while being one of the most
  expensive tests in the suite. Such tests are now reported separately with no
  conclusion drawn.

It needs `coverage` and `pytest-cov`, both dev dependencies, and a whole-suite
run with per-test contexts (about +33% over a normal run). The invocation is in
the script's docstring.

**Two of the heaviest were reviewed and deliberately KEPT** (JDE, 2026-08-25):
"expensive but worth it". Recorded so a future runtime sweep does not
re-litigate them.

- `test_runner.py::test_run_without_flag_writes_no_status` (232 s) is the
  suite's single most expensive test and sets the CI shard floor at ~6.7 min.
  Its cost is **entirely fixed startup** -- it already runs 2 tune / 1 draw, so
  no sampler tuning touches it -- and it is deliberately doubling as the
  end-to-end whitening-probe test, with `measure_scales` left at its default
  so one real run exercises the startup probe and rescale.
- `test_examples_prepare.py::test_shipped_example_prepares` (238 s over 25
  cases) re-prepares every shipped example on every matrix combination. That
  breadth IS the point: it is the canary that a shipped config still works.

Cutting either is a coverage trade, not a cleanup. The remaining runtime levers
that cost no coverage are in `docs/testing-cache.md`, and they are nearly
exhausted -- see the floor arithmetic there.

The heaviest individual tests, for anyone looking for something to cut:

| seconds | test |
|---|---|
| 231.6 | `test_runner.py::test_run_without_flag_writes_no_status` |
| 175.7 | `test_robust_likelihood.py::test_outlier_prob_at_data_flags_a_planted_outlier` |
| 171.3 | `test_rm_ltt.py::test_wired_rm_ltt_delay_matches_a_over_c_through_real_accessors` |
| 167.1 | `test_rm_ltt.py::test_rm_ltt_off_reproduces_pre_ltt_output` |
| 144.9 | `test_rossiter.py::test_rm_two_instrument_logp_and_gradient_finite_on_both_backends` |
| 139.6 | `test_mkparam_in_memory.py` (fixture setup) |

Most of that is `build_model()` plus compiling logp/dlogp inside the test body, and the
`both_backends` cases pay it twice. Compilation is not what a WARM run spends its time
on, though: the same run added only 15-45 new compiledir entries per worker (~150 total)
against the 1564 a cold run creates, so the 7:36 -> 15:47 regression since July is added
work, not a degraded cache.

Two claims that used to live in `CLAUDE.md` are wrong and are recorded here only so nobody
reintroduces them:

- "The test suite takes ~10 minutes." Superseded by the measured numbers above.
- "`poetry run pytensor-cache cleanup`" as the remedy for a compile-cache-induced pytest
  Timeout. The diagnosis of the Timeout is right; the remedy is not. `pytensor-cache
  cleanup` only deletes entries older than 31 days and provably does nothing here
  (measured 4035 -> 4034 entries, 4.1 G -> 4.1 G).
