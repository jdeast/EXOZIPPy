# The test suite and the PyTensor compile cache

This is the runbook for the suite's runtime and for the compile cache that
dominates it. It replaces the advice in CLAUDE.md's Commands block, which
told you to run `pytensor-cache cleanup` when a test timed out inside
`cmodule.py`. **That command does not fix this and never did** -- see
"What actually reclaims cache space" below.

## The failure it explains

The symptom is a test going red on `Failed: Timeout (>300.0s)` with a
traceback inside `pytensor/link/c/cmodule.py`, blaming a test that has
nothing to do with compilation. It moves between tests run to run, and it
usually passes on a rerun with no source change.

The mechanism: PyTensor caches every compiled C module as one subdirectory
of its compiledir. The first compile in **any process** constructs the
`ModuleCache`, whose `refresh()` walks every subdirectory and unpickles
every `key.pkl`. That walk is `O(entries)` file reads, it happens once per
process, and it holds the compile lock while it runs -- so under `pytest -n
6` it happens six times, serialized. Whichever test triggers the first
compile in its worker is billed for the whole walk, and pytest-timeout's
300 s cap is what it hits.

Two measurements pin it down:

- The shared `~/.pytensor` had grown to **4035 entries / 4.1 GB**, fed by
  months of interactive fits and by every agent worktree on the box. A
  single-process `get_module_cache()` against it took **14 s** with the
  pages partly warm; the suite's six workers, cold and serialized, showed
  up as 13 tests taking 60-135 s each -- about 40% of the suite's CPU.
- Re-running the suite immediately afterwards, against the **same** 4035
  entries but with the pages now hot, took 9:13 with **zero** failures and
  a slowest test of 17 s.

So the cost is cold page-cache I/O over the entry files, and it is linear
in the **entry count**. That is the quantity to bound.

## What actually reclaims cache space

| | what it does | effect here |
|---|---|---|
| `pytensor-cache cleanup` | `compiledir.cleanup()` + `ModuleCache.clear_old()` | **nothing** |
| `scripts/pytensor_cache_budget.py` | LRU eviction down to an entry count | what you want |

`clear_old` deletes only entries older than
`cmodule__age_thresh_use + 7 days`, i.e. 31 days. On a repository whose
suite runs daily nothing is ever 31 days untouched -- the `refresh()` walk
itself bumps every entry's atime on every run, so age is precisely the knob
that cannot work here. Measured on the 4.1 GB cache: **4035 -> 4034
entries, 4.1 G -> 4.1 G, 44 s spent.**

Never `rm -rf` the `tmp*/` subdirectories individually thinking they are
scratch; they **are** the cached modules. Deleting the whole compiledir is
safe but costs a full cold recompile (see the numbers below).

To reclaim space by hand:

```bash
# What would go, without touching anything.
poetry run python scripts/pytensor_cache_budget.py --max-entries 2500 --dry-run

# Do it. Defaults to whatever pytensor.config resolves, so PYTENSOR_FLAGS
# is honoured; --compiledir names one explicitly.
poetry run python scripts/pytensor_cache_budget.py --max-entries 2500
```

It evicts least-recently-used on the `atime` of `key.pkl` -- the same stat
field PyTensor's own `last_access_time()` reads -- and also removes broken
entries (a `key.pkl` with no `.so` beside it). `--sweep-other-platforms`
additionally deletes sibling `compiledir_*` trees stranded by a kernel or
Python version bump; only use it on a base directory owned by one purpose.

## The test compiledir policy

Two changes in the root `conftest.py`, which is the only place that runs
before `pytensor` is imported. That timing is forced: `base_compiledir` is
declared `mutable=False`, so it can only be set through `PYTENSOR_FLAGS`
ahead of the import.

1. **The suite has its own `base_compiledir`, `~/.pytensor-pytest`.**
   Interactive fits and the suite stop inflating each other's startup walk.
   It lives under `$HOME` rather than in the checkout on purpose: every
   worktree of this repo then shares one **warm** cache, where an in-repo
   path would make each new agent worktree pay a full cold compile.
2. **It is bounded at 3000 entries**, pruned LRU on the xdist controller in
   `pytest_configure`, before any worker exists -- which is also what makes
   the prune safe without taking PyTensor's compile lock.
3. **The `ModuleCache` walk is forced in `pytest_configure`, in every
   worker.** pytest-timeout arms its per-test `SIGALRM` later, so however
   long the walk takes it can no longer fail a test. That is the actual fix
   for the red test; bounding the count is what makes it fast.

Escape hatches:

| variable | effect |
|---|---|
| `EXOZIPPY_TEST_COMPILEDIR=/some/path` | put the suite's cache somewhere else |
| `EXOZIPPY_TEST_COMPILEDIR=` (empty) | opt out; use whatever PyTensor would pick |
| `EXOZIPPY_TEST_COMPILEDIR_MAX_ENTRIES=N` | change the budget (CI uses 2500) |

The prune itself is guarded by a cheap pre-check, and that is load-bearing
rather than an optimization: its per-entry pass opens a directory and stats
a `key.pkl` for every entry, which measured **72 s over 4034 entries** on a
cold page cache. One `listdir` of the parent bounds the entry count from
above, so being under budget is provable without the pass, and steady state
is a single directory read. Without it, `pytest -n0 -x` on one test would
pay the very cost this change removes.

## The three numbers

All on the same 36-core box, `-n 6 --dist loadfile`, same commit.

| | wall | result | notes |
|---|---|---|---|
| **Cold compiledir** | **27:15** | 4 failed | empty cache, everything compiles. The best local proxy for a CI job. |
| **Contaminated warm** | **9:39** | 1 failed | the 4035-entry shared cache, cold pages. The failure is the cache-init victim. |
| **Clean warm** | **7:36** | all pass | the bounded private cache, page cache evicted first |

The spread is the finding: the same suite, same commit, differs by a factor
of three and a half depending only on the state of a directory nobody was
managing.

The sharpest single number is the test that started this. On the
contaminated cache `test_vcve.py::test_the_inversion_round_trips_the_forward_relation`
blew the 300 s cap and went red; on a fully hot page cache it still took
84 s; on the bounded private cache with the walk moved out of band it takes
**0.32 s**. It was always a 0.32 s test being billed for the cache walk.

The remaining slow tests -- 81 s for
`test_robust_likelihood::test_outlier_prob_at_data_flags_a_planted_outlier`,
79 s for the two `test_rm_ltt` cases -- are genuinely slow sampling tests
and cost the same on a hot cache before this change. They are the next
thing to look at if the suite needs to get faster; the cache is no longer
the bottleneck.

### A cold cache is still not a pleasant place, and that is worth knowing

**The cold run's four failures were a separate bug the measurement turned
up.** They were not the refresh-walk timeout; they were
`filelock._error.Timeout` on PyTensor's compile lock. All compilation for a
compiledir is serialized behind that one lock, and its default acquire
timeout is 120 s (`compile__wait * 24`), which six workers queueing on an
empty cache blow straight through. It hit `test_distance_volume_prior`,
`test_nsnl`, `test_rossiter` and `test_multiplanet` -- whichever happened
to queue behind a long compile, the same lottery as the refresh-walk
timeout. The conftest now sets `compile__timeout=600`, and a repeat cold
run confirms it: **4 lock timeouts -> 0**, in 24:57.

That repeat run still had **one** failure, and it is a third, independent
effect that this change does not fix and does not claim to.
`test_astrometry.py::test_finite_logp_and_gradient` hit pytest-timeout's
300 s cap -- not because of the cache walk, which is now paid out of band,
but because on an empty compiledir the gcc work that test triggers really
does take that long. It is a 17 s test warm.

The instructive part is the durations either side of it: on the cold run
`test_band_autopin_ld` took 365 s, `test_vcve` 337 s, `test_rossiter` 328 s
and `test_run_endpoints` 316 s, and all four **passed**. They overran the
same cap; the difference is only where the interpreter happened to be when
`SIGALRM` fired. pytest-timeout's signal method cannot interrupt a process
blocked in a C extension (see the note on `timeout_method` in
pyproject.toml), so a test waiting on a gcc subprocess simply ignores the
alarm until it returns to Python. Whether a cold run goes red is therefore
a coin flip, exactly as the original bug was, one level down.

This is not worth weakening the 300 s cap for. It only bites on a
genuinely empty compiledir, which is a state you now hit **once**, the
first time you run the suite after this change. Just re-run -- the second
run is warm and green -- or make the cold run deliberate with
`--timeout=1800`. CI never sees it at `-n 2`, where the lock queue is two
deep instead of six.

One full cold run creates **1564 entries / 825 MB**. That is the number the
budgets are sized against -- a budget below one run's working set would
evict entries that run still needs. A *second* run on a different branch
does not add another 1564: it reuses nearly all of them and adds only the
delta for graphs that changed.

Refresh cost scales with the entry count as advertised: 1561 entries with
the page cache explicitly evicted takes **6.4 s** in one process, against
14 s for 4034 entries partly warm.

Note on honesty of the "clean warm" number: re-running a suite immediately
after another one finds every `key.pkl` already resident in the page cache
and reports a time nobody will ever see in practice. The clean-warm run
below was taken after evicting the compiledir from the page cache with
`posix_fadvise(POSIX_FADV_DONTNEED)`, which is what `drop_caches` does
globally and needs root for.

## CI

`.github/workflows/tests.yml` now restores the compiledir in every matrix
job and saves it **only on master pushes**. The reasoning is in the comment
on the step; the two load-bearing points are that a cache saved on the
default branch is readable from every branch while a topic-branch one is
not, and that the 10 GB repository cache budget is evicted
least-recently-accessed -- so writing a few hundred MB per matrix
combination on every PR run would churn several GB a day and put the Zenodo
spectra (132 MB) and ephemeris kernel (115 MB) caches at risk of eviction.

The kernel version is the footgun. PyTensor names the compiledir after the
platform string, which includes the kernel release, so a runner image bump
silently strands the whole restored tree -- nothing reads it, and nothing
would ever delete it either, so it would ride along in the cache forever.
That is handled on the pytest side: the conftest prune sweeps stale sibling
`compiledir_*` trees. This is the same class of bug as the Zenodo cache path
that went stale in July 2026 and kept reporting "cache hit" while restoring
132 MB into a directory nothing opened.

## A source change can invalidate the whole cache

A change to the *structure* of a commonly built graph misses every cached
entry, not a few.  Measured 2026-08-18: turning `_RAW_CANCELLATION_CLIP` into a
`pytensor.shared` (review 1.2.1) altered the graph of every logit-transformed
element, so the first suite run after it took **14:40 instead of 8:26**, almost
all of it in `cc1plus`.

This is a one-time cost per such change, and it is expected -- but it looks
exactly like "the compile cache stopped working", including on CI, where every
matrix job pays it on the first run after the merge.  Before diagnosing a slow
run as a cache regression, check whether the diff touched a graph that
everything builds; the tell is `pgrep cc1plus` during the run, and an entry
count that climbs rather than holding steady.
