# Testing conventions

Tests follow AAA (Arrange / Act / Assert) with Given/When/Then docstrings. All tests that use `System` must call `system.prepare()` before `system.build_model()`. RA/Dec user params are in **degrees** (the default unit); `Parameter.__post_init__` converts to radians internally.

Do not start the full suite with a timeout. Start it and poll.

Testing note: build relation inputs with `pt.dscalar`, **not** `pt.as_tensor_variable(<python float>)` -- pytensor autocasts a bare Python float to the smallest dtype that represents it (5778.0 -> float32), and a unary op like `pt.log10` on it then computes in float32, silently losing ~1e-7. The model always feeds float64. `tests/test_torres.py` pins the port against real IDL output from `massradius_torres.pro`.

## Renames create tests that pass while testing nothing

A parameter or component rename does not usually make a test fail -- it makes
it VACUOUS, which is worse, because a green suite tells you to stop looking.
The 8.6.17 mulensevent split produced four distinct shapes of this in one
sweep, all found by porting the tests rather than by running them:

- **"The old name is absent."** `assert "lens.log_pi_rel_raw" not in
  [v.name for v in model.value_vars]` was the *off* half of a coordinate-flag
  test. After the rename it is trivially true whatever the flag does, so the
  test passed while asserting nothing. Any `not in` / `not hasattr` /
  `assert not ...` phrased against a name that MOVED is now free.
- **A filter keyed on the old prefix.** `{c for c in consumers if
  c.label.startswith("lens(")}` built the set under test. Renaming only the
  neighbouring equality assertion left the filter matching nothing, so the
  test asserted a property of the empty set.
- **A guard scanning a fixed list of modules.** A check that "no hard-coded
  u_0 clip survives" scanned the three modules that then held the clip. The
  code moved to two new modules; the guard kept passing while watching files
  that no longer contain what it polices.
- **A precondition that skips.** A test's arrange block did
  `if not lens_block.get("sources"): pytest.skip(...)`. Post-split that key
  never exists, so it skipped unconditionally -- a green `s` in the summary
  and zero execution, for a test whose assertion had also become wrong.

Practical rules, all of which paid for themselves here:

1. After a rename, grep the suite for the OLD name and check every hit that
   is inside a `not in`, a `startswith`, a module/symbol list, or a skip
   condition. Those four are where vacuity hides; a direct `assert x == y`
   fails loudly and needs no hunting.
2. Verify a renamed name against the code that GENERATES it, not against a
   plan or a summary. `op.py` spells the post-split companion index out as
   `f"lens.{j + 1}.s"`; `star.py` emits the literal
   `"mulensevent(finite_source)"`. Both settled index and label questions
   that prose had got wrong.
3. When an exemption is added to a test (this element is allowed to be NaN,
   this case is allowed to be missing), assert that the exemption FIRED and
   that the unexempted path was still exercised. Otherwise a later change
   that broadens the exemption silently empties the test.
4. When a bug becomes structurally impossible rather than fixed -- the
   event-rate double-count could only exist while `mu_rel` was per-source,
   and R1 makes it shape (1,) -- say so in the docstring and pin the SHAPE
   claim that makes it impossible. Do not leave a test implying it still
   guards the old failure.

## Covering a code path is not testing its numbers

Distinct from the six shapes above, because nothing here is stale and nothing
is vacuous: the test runs the code, and would catch a crash, a shape change
or a role change. It is simply blind to the arithmetic.

`tests/test_triple_lens_q_start.py` is the ONLY coverage of the
`n_companions >= 2` seeding branch. It asserts `q0.size == 3`, `isnan` on the
unseeded slot, `element_is_active` / `element_is_derived` per element, and
the text of two warnings. Every one of those is a structural property. So
when `_mass_initval` seeded a planet's mass in jupiterMass where solMass was
promised -- `mlens_total`, and through it `theta_E`, `t_E` and every
per-element `q`, 1047x too large -- that file went on passing, and it was the
only file that could have noticed.

It surfaced because a 3-body lens was built for an unrelated reason and one
number looked wrong by suspiciously close to a mass-unit ratio.

- If a test is the only thing exercising a branch, assert at least one VALUE
  from it, not only shapes and roles. Shapes catch wiring; values catch
  arithmetic, and unit errors are arithmetic.
- Prefer an assertion that needs no constant of its own. The regression test
  here compares the SEEDED value against what the BUILT graph computes, so it
  carries no jupiter/solar factor -- it cannot repeat the mistake it guards
  against, and it fails on a slip in either direction. A hand-computed
  expected value would have been a second place for the same error to live
  (CLAUDE.md: never hand-write a conversion).
- Suspect a clean ratio. 1047.6 is not noise; a factor that lands on
  jupiter/solar, 365.25, 206265 or a power of ten is a unit or a
  radians/degrees slip, and is worth chasing even when a test is green.

## A sampler budget too small to adapt cannot test a posterior

Same family as the section above, and the sharper case: the test runs real
code, asserts a real number, and the number it asserts is a coin flip.

`tests/test_integration_kelt4.py` drives `tune: 2, draws: 1, chains: 1` -- the
right budget for an end-to-end pipeline test, and the reason the file is
affordable at all. A test named `test_run_fit_kelt4_posterior_in_sane_range`
then asserted hard physical bounds on the resulting "posterior mean". That
cannot work, for a reason that has nothing to do with the model: at `tune: 2`
dual averaging has adapted nothing, so the step size is still the
`step_scale / size**0.25` heuristic and the single draw is **one random jump**
of that size, in raw space, away from the start. The jump's PHYSICAL size is
therefore set by the start's CONDITIONING, not by the physics. While the start
sat at `raw = 0` eight seeds agreed on `planet.mass` to 1% (0.8304-0.8721) and
the bounds looked solid; once review 1.3.6's fix moved the start off `raw = 0`
the same jump landed anywhere from 0.81 to 8.718 Mjup. Out-of-bound values
observed before it was rescoped: 2.870 and 5.7693 locally (the second on plain
master), 6.724 and 8.718 on CI. Each one read as a branch regression and cost
a triage.

**The remedy is to assert the START** (review 7.13.6, ruled by JDE), because
that is the one quantity such a budget actually determines: the pre-whitening
polish has no RNG, so the start is bit-reproducible run to run, while the draw
is not. `test_run_fit_kelt4_start_is_physical` reads the startup table out of
`<prefix>.log` -- the start the run ITSELF reported, which is what keeps it an
integration test; rebuilding the System from the same config would miss the
polish and the anchor re-centering, both of which happen inside `run_fit` and
both of which move the start. It is a deliberate **golden-value** test: an
intentional change to where the sampler begins has to edit those numbers, so
the move shows up in the diff and gets justified in the commit. Review 1.3.6
was a wrong default start that survived months precisely because nothing in
the suite asserted where the sampler begins. Measured over six consecutive
runs of the same fixture on one box (under the old polish stop, which is why
the value differs from today's golden): the three start values were **bit-
identical** every time (`planet.b.mass` 0.96736983 on all six), while the
draw those runs produced ranged over 0.8158-1.4158 Mjup, a factor of 1.7 --
and that is the well-behaved case, inside the old bounds.

**A golden start value cannot be pinned to the ~1e-9 of review 3.14.20, and
the first version of this test went red on CI for assuming it could.** Two
compounding mistakes, both worth knowing before you write another one.

*A relative tolerance on a log quantity measures the offset, not the error.*
`star.A.logmass` is 0.08 dex, so 2.65e-4 of absolute dex scatter reads as
3.3e-3 RELATIVE -- an `rtol=1e-3` fails on it -- while the SAME scatter in the
physical mass (1.203575 to 1.204309 Msun) is 6.1e-4 relative and passes.
Compare each quantity in its own domain: an absolute tolerance in dex for a
dex/log quantity, a relative one for a linear one.

*And the scatter is the optimizer, not float noise.* The value being asserted
is POST-POLISH, and the polish is an iterative optimizer terminating on
`|grad|_inf < polish._LBFGS_GTOL` nats/unit, so ANY difference in the
arithmetic that feeds it -- of any size -- moves the point where it stops.
Review 7.13.8 measured what does and does not count as such a difference.
NOT: machine load (to 28.9 on a 36-core box), BLAS thread count,
`PYTHONHASHSEED`, a cold or per-worker pytensor compiledir, heap churn, or
another `System` built first in the same process -- fifteen same-machine
runs varying all of those were bit-identical over all 177 evaluations. (An
earlier version of this section blamed "multithreaded BLAS partitioning by
load"; the compiled objective has no BLAS op, and that story was wrong.)
YES: a different OpenBLAS kernel inside scipy's OWN L-BFGS-B bookkeeping --
what a different CI runner CPU picks under `DYNAMIC_ARCH`, reproducible
locally with `OPENBLAS_CORETYPE=Haswell` -- which changes one iterate by ONE
ULP at evaluation 4; or an equivalent but differently-fused compiled graph
(`optimizer_excluding=fusion`), which differs at evaluation 0 by 7e-10
relative. Either lands the polish somewhere else on the same basin.
Cross-platform libm/compiler/SIMD differences are >= 1 ulp by construction,
so this cannot be removed; it can only be kept from being AMPLIFIED.

**The amplifier was the stop, and the fix is in `polish.py`, not in the
test.** scipy's gtol test is on the CURRENT iterate, so it fires on the first
evaluation to dip under the threshold. At the old `gtol = 0.01` that was a
first dip mid-climb on kelt4's tc/logP ridge (Hessian condition number
5.4e6): exactly one of 177 evaluations was under 0.01, it fired 0.507 nats
below the basin optimum, and on that shoulder one ulp moved `cosi` by 8.5%
and the polished logp by 0.24 nats -- more than the test's own 0.2-nat logp
tolerance, which the six observed environments passed only by chance. The
shipped `gtol = 1e-4` (cap 400) stops at the optimum, where the same
perturbation moves `cosi` by 8.6e-4; the numbers are in the header of
`tests/test_integration_kelt4.py` and on `polish._LBFGS_GTOL`.

**How to calibrate a post-optimizer golden value, then -- and the rule.**
NEVER from repeated runs of one environment, however many, and not from a
handful of environments either: seven bit-identical solo runs opened that PR
and proved nothing, and the six-environment table its second version
calibrated from (dev box solo and under `-n6`, four CI combinations) was a
six-point SAMPLE of a distribution whose full width turned out to be 5x their
spread. The calibration is the **ulp-perturbation harness**: run
`polish._lbfgs_polish_one` from the fixture's raw start with the compiled
`(lp, grad)` multiplied by `(1 + s * 2**-52)`, `s` in {-1, 0, +1} a hash of
(x, output component, seed) -- "the same function computed by a different
but equally correct arithmetic" -- for 16 seeds, and set each tolerance to
~5x the full width it reports; the golden values are the unperturbed seed.
`tests/test_polish.py::test_ulp_perturbation_does_not_move_the_polished_start`
runs a four-seed version on every CI run, so a stop that amplifies again
fails there rather than as an intermittent red in the integration test.
Measured 2026-09-14 under the shipped constants:

| quantity | golden | 16-seed full width | tolerance | at the old gtol 0.01 |
|---|---|---|---|---|
| `star.A.logmass` | 0.08052772 | 2.4e-6 dex | 2e-5 dex | 2.7e-4 dex |
| `planet.b.mass` | 0.96091637 | 3.0e-4 rel | 2e-3 rel | 2.1e-2 rel |
| `orbit.b.logP` | 0.47554391 | 1.1e-7 dex | 2e-5 dex | 3.1e-6 dex |
| `orbit.b.cosi` | 0.50012485 | 8.6e-4 rel | 5e-3 rel | 8.5e-2 rel |
| `m sin i` | 0.83210871 | 2.6e-5 rel | 2e-4 rel | 9.5e-3 rel |
| polished logp | 81.947279 | 9.1e-7 nats | 0.2 nats (print resolution) | 0.24 nats |
| iterations | 268 | 240-294, none of 16 capped | cap 400 | 134-150, 3 of 16 capped |

**The scatter is also not uniform across parameters, and that part is physics
rather than noise.** Ranked by the harness width: the start logp (stationary
at the optimum), `orbit.logP` (pinned by the data), `star.logmass` (pinned by
its Gaussian prior), `m sin i` (what the RVs constrain), `planet.mass` (which
is `m sin i / sin i` and so inherits `cosi`), and `orbit.cosi` (the flat
direction an RV-only fit says nothing about). One tolerance across that range
is either vacuous at the top or red at the bottom, so give the flat direction
its own -- and note that the hierarchy itself is informative: if `cosi` ever
stops being the loosest row, something has started constraining the
inclination.

**Prefer a golden START LOGP to golden parameter values**, and assert both --
but the stationarity argument holds only when the optimizer actually reaches
the optimum. logp is STATIONARY there, so optimizer scatter perturbs it only
at second order (9e-7 nats here, six orders below the 0.1 nat the polish line
prints), while a changed prior, a unit-conversion slip or a lost likelihood
term moves it by O(1) nats. On the old first-dip shoulder it was NOT
stationary (0.24 nats of harness width against a 0.2-nat tolerance), which
is why the perturbation test above is what makes the logp assertion the
discriminating one. The parameter values are the readable failure message;
the logp is the discriminating assertion. Pin BOTH ends of the polish: the
pre-polish value is a plain evaluation at the build start with no optimizer
in it at all, so it carries none of that scatter -- and the 7.13.8 stop
change is the worked example: build logp unchanged at -601.1, polished logp
81.4 -> 81.9, the half nat the first dip had been leaving on the table.

**One instance of the same shape is knowingly left in place**, so a later
reader does not think the sweep missed it: `..._posterior_in_user_units` in
the same file reads the same single draw, against a tighter `0.3 < logP <
0.65`. It is left because `orbit.logP`'s whitening scale is 1e-5, so the jump
moves it by ~1e-5 against a window of 0.35 -- measured spread over those six
runs, 0.475620-0.475637. The window is 20000x the noise, and the regression it
watches for (internal vs user units) is a factor of ~1000. Rescope it if that
scale ever changes.

Two options were considered and rejected, so they are not re-proposed.
*Widening the bound* encodes the noise rather than measuring it -- 6.724 is
~7x truth and the single-draw tail already reached the old 2.5 bound.
*Buying a real tuning budget* is honest but is a separate, `slow`-marked test
if it is wanted at all; it is not what this test is for.

**A golden value and a literature value are different claims, and a
parameter earns one or the other.** In `kelt4_rvonly.yaml` the RV data
constrains `m sin i`, not the mass: there is no inclination information, so
`cosi` is prior-dominated (the polish walks it from the params file's
transit-derived 0.11996 to 0.50012) and `mass = m sin i / sin i` inherits
that. So `m sin i` is the quantity comparable to a published value and
carries the LITERATURE check, while `planet.mass` is still perfectly
deterministic given the same code and priors and carries a GOLDEN-VALUE
REGRESSION check against our own recorded number. Assert both, and say in
the test which is which -- otherwise the next reader either "fixes" the mass
against the literature or deletes it as prior-dependent noise, and both are
wrong. What the comment must NOT say is that `m sin i` is the trustworthy
one: a mass marginalized over the inclination prior IS a posterior for the
mass, and its width says how much of it is prior, whereas `m sin i` is a
lower bound the field routinely quotes as a measurement. KELT-4Ab agrees
well because it transits (i ~ 83 deg, so it sits near its minimum mass) --
a property of that system, not of the statistic.

**What the old assertion was incidentally covering, and what replaces it.**
Reading `planet.mass` out of `idata.posterior` did exercise the path where a
quantity is computed during sampling, written to the trace and converted to
user units on the way out. Presence and units were already covered by sibling
tests; the IDENTITY was not.
`test_run_fit_kelt4_derived_parameters_are_self_consistent` recomputes a
derived value from its parents AT THE SAME DRAW -- `orbit.period == 10 **
orbit.logP`, and `orbit.vcve` from the `sqrt(e)cos/sin(omega)` pair -- and
one draw is not a compromise there, it is sufficient by construction: a
derived quantity is a deterministic function of its parents, so the identity
either holds everywhere or is broken. Two subjects that look obvious and are
not: `planet.mass` is SAMPLED in this config (the trace carries
`planet.mass_raw`; the relation runs the other way, K from the mass), and
`star.mass` is derived but never appears in `idata.posterior` at all, because
a pure-expression parameter never does.

When you write an end-to-end test, the question to ask is not "does this
exercise the sampler" but "does this budget DETERMINE the quantity I am about
to assert". A start value, a shape, a file, a variable name, a finiteness
check and an IDENTITY between a derived value and its parents all survive
`draws: 1`. A mean, a physical range, an Rhat and an ESS do not.

`tests/test_integration_ob09020.py` is the microlensing twin of the kelt4 test
(PTDE on a two-rung ladder, `seed_polish: false`, `measure_scales: false`,
three draws, on a COPY whose large photometry files keep every 40th data row
-- one logp of the full VBMicrolensing finite-source model costs ~3 s, and
every scale probe, whitening's or PTDE's own, is a serial loop of tens of
evaluations per raw element, so at full size the probe alone is an hour; the
config itself is untouched) and asserts
exactly that class of quantity: the trace and start-plot files, the posterior
variable names, the NUMERICAL validity of every draw (finite raw values and
lp, lp inside the runaway ceiling), and -- with the polish off -- the
user-start contract on the seeds the relaxation engine reproduces exactly.
The wrap-up's invalid-draw gate is forced there (`modes: {force: true}`,
test-local), because on a few dozen untuned draws its 1% threshold fails on
any single rejection and one of its four reasons, `raw-z`, is a z-score
against the trace's own median/MAD -- a property of the sample size (review
7.13.9); the three numerical reasons are asserted directly instead. It exists because the shipped
microlensing examples were otherwise only `prepare()`-tested or replayed for
their logp terms, and `examples/ob09020` shipped from 2026-09-07 to 09-25 with a
TypeError in its first start-point plot (the orbit-derived lens geometry listed
as a compiled-plotter input) that no test reached.

## Network tests and the Zenodo pins

Download tests monkeypatch the transport (`tests/test_zenodo_fetch.py`,
`tests/test_mist_eep_download.py`, most of `tests/test_bc_tables_zenodo.py`).
The few that read a real Zenodo record are marked `network`; they run in the
ordinary suite and on CI, and **skip only on a transport failure** (no
network, DNS, a 5xx). An HTTP 4xx fails, because it means a pinned record id
is wrong. Deselect them with `-m "not network"` on a machine without
network.

`tests/test_zenodo_assets.py::test_the_pins_match_the_live_zenodo_record`
re-reads `https://zenodo.org/api/records/<id>` for every record in
`src/exozippy/utilities/zenodo_assets.py` -- the ONE table of every Zenodo
record, file, size and md5 the code downloads (review 4.9.2) -- and asserts
the record id, concept id, DOIs and the full file list with every size and
md5 match. **To publish a new version of a record:** upload it on Zenodo as
a new version of the same record, update that record's ONE entry in
`zenodo_assets.py` (the new `record_id`, and the size and md5 of every
changed file), and run the network test:

```bash
pytest tests/test_zenodo_assets.py -n0
```

The offline tests in the same file guard the rest: the registry is
internally consistent, every consumer reads its pins from it, the migrated
pins are bit-identical to the literals they replaced, and no other module
under `src/` spells out a Zenodo record id or a pinned md5.

## The pre-push hook, and why it does not say `poetry run pytest`

A suite runs on push, wired in `.pre-commit-config.yaml` (install both hook
types with `poetry run pre-commit install`) -- since 2026-10-01 the FAST tier by
default, see "The pre-push tier" below. The entry is
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
prints the resolution and exits without running the suite; `EXOZIPPY_PREPUSH_NOWAIT=1`
skips the wait described next.

**One pre-push suite at a time per machine.** The hook takes an `flock` on
`/tmp/exozippy-prepush-<uid>.lock` before running pytest, so a second push on the same
machine WAITS -- printing a line every minute, never failing -- until the first suite
finishes. That is throughput, not correctness: two full suites at once measured 37
minutes against 8 for one. Correctness no longer depends on it, because since review
2.13.5 every pytest run compiles in its own directory (`docs/testing-cache.md`,
"Concurrent suites"), so a targeted `pytest` run beside a pre-push suite is safe too.
If a run nonetheless fails on a `compiledir_*/tmp...` path, the terminal summary says so
in one line: triage by error class -- compile/import errors on that path are
environmental, an assertion on a number still has to be explained.

Two properties of the hook that this did **not** change, and that still bite:

- It tests the **working tree**, not the commits being pushed. Pushing a branch from
  the main tree therefore tests whatever is checked out there, not the branch.
- `pre-commit` stashes unstaged changes while hooks run and restores them afterwards.
  Do not kill a run in progress; the work is recoverable from the patch it prints under
  `~/.cache/pre-commit/`, but only by hand.


### The pre-push tier: fast locally, everything in CI

**The hook runs `-m "not slow"`; CI runs every test; CI is the merge gate.**
(JDE 2026-10-01.) The repository ruleset "master: require green tests" requires a
pull request and the `test` check -- the aggregator over lint and every pytest
shard -- before anything reaches `master`, so a regression in a slow test cannot
merge. What the fast tier gives up is only *where you hear about it first*: a slow
test that breaks is reported by CI on the PR, about 25 minutes later, instead of
blocking the push.

Why not keep the full suite in the hook: with several sessions working the repo,
one 30-minute suite per push serialized behind the per-machine lock (above) was the
throughput bottleneck -- pushes waited one to two hours for their turn on
2026-09-30/10-01. The fast tier is measured at a few minutes at `-n6`.

**What `slow` means.** `@pytest.mark.slow` (registered in `pyproject.toml`) marks
end-to-end fits, sampling-heavy tests, and builds or compiles that cost more than
~15 s on this box. It was applied from measured `--durations` of two full runs
(every call phase or module fixture over 15 s; a module fixture marks the whole
module, since its tests share it). When a new test is that slow, mark it -- the CI
time-budget check (`scripts/check_test_budget.py`, above) lists anything over 60 s
on the job summary, which is the cue.

**Overrides.** `EXOZIPPY_PREPUSH_FULL=1 git push` runs the full suite locally, as
the hook did before. Arguments given to `scripts/pre_push_suite.sh` are passed to
pytest after the tier's `-m`, so an explicit `-m` of your own wins. Tests:
`tests/test_pre_push_hook.py` (default tier, the override, and that `slow` is
registered -- an unregistered marker only warns, so a typo would silently move a
slow test into the fast tier).

## Model data from Zenodo: the spectra and the BC tables

The NextGen spectra, the NextGen BC tables and the MIST EEP grid are not in the
repository; each is fetched from Zenodo on first use (`utilities/zenodo.py`). The
autouse `_no_shared_download_cache` fixture switches the machine cache
(`~/.cache/exozippy/downloads`) off inside every test, so the tables a test reads must
already be in the tree. `tests/conftest.py::_prewarm_bc_tables` puts every published BC
table there once, before the suite (served by the machine cache when it is warm, so a
fresh worktree costs hard links, not a download); a failure is reported as a warning and
then re-raised by each SED test that needs a table. CI restores both the spectra and the
tables with actions/cache (`.github/workflows/tests.yml`). Offline with a cold cache,
run `exozippy-fetch-bc-tables` once with network first.

## Suite runtime and the pytensor compile cache

The suite ran in **~16 minutes warm** on an idle 36-core box (`-n 6`, 3108 tests,
measured 2026-08-19 at 2977 tests / 11:37 and 2026-08-25 at 3052 / 15:47); it has grown
19% in worker-seconds since (**11987 ubuntu worker-seconds over 229 files** on CI run
34888667878, 2026-09-14, against 10055 over 204 files on 2026-08-25), so expect a warm
local run nearer 19-20 minutes today. A cold run is **~25 minutes** and happens once per
fresh checkout or worktree, and on CI until its compiledir cache is populated. The
runbook -- what the cache is, how it is bounded, how to reclaim space, and how to measure
a run honestly -- is in `docs/testing-cache.md`. Read that before changing anything
about the compile cache or the suite's timing.

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
shards (`scripts/pytest_shard.py`, packing longest-file-first from
`tests/durations.json`, measured at 1.00x of ideal balance): with no dominant file there
is nothing to cut, so the remaining lever is more machines -- **4 ubuntu shards and 3
macOS** (macOS is 3.2x faster per worker-second, and the free plan caps concurrent macOS
jobs at 5).

Where that lever runs out (re-measured 2026-09-14). `--dist loadfile` pins a file to
one worker, so a shard cannot beat its slowest file's serial time, and a CI job's wall
clock is `~100 s fixed + max(shard worker-seconds / workers, slowest file)` to within 4%.
When the 873 s `test_mulens_acceptance.py` landed it exceeded the ~750 s per-worker share
at 4 shards, and the worst job stayed at 16.2 min at 4, 5, 6 and 8 shards alike: more
machines bought nothing. Splitting the file (`test_mulens_acceptance_a.py` / `_b.py`, by
fixture-name partition in `tests/mulens_acceptance_replay.py`) is what brought it to
14.2 min. So below ~10 minutes the move is splitting slow FILES -- which is also why
`test_runner_lifecycle.py` was split out of `test_runner.py` -- and never adding jobs
past the concurrency cap. See `docs/testing-cache.md` for the full arithmetic, the
sharding, and the compiledir seeding that keeps its cache affordable.

### Keeping tests/durations.json current

`tests/durations.json` is the per-file weighting the shard split packs from. Two rules
and one mechanism, because the file silently drifted for 20 days and 25 new test files
before anyone looked (2026-09-14), and the cost was a shard running 17 minutes against
another's 11:

- **It is regenerated from CI artifacts, never from a workstation.** Every pytest job
  uploads its `--durations=0` transcript as a `durations-<os>-<python>-<shard>` artifact;
  the whole suite is the union of one os+python's shards. Workstation weights balance the
  recorded sums and still produce a 1.6x spread in real wall clock (the arithmetic is in
  `docs/testing-cache.md`).
- **Regenerate whenever a test file is added or a file's cost changes materially**
  (a new fixture in a parametrized replay, a sampler budget change, a split). A file
  absent from the map is charged the median, which is exactly how an 873 s file got packed
  as an 8 s one.

  ```bash
  gh run download <master-run-id> -p 'durations-ubuntu-latest-3.12-*' -D /tmp/dur
  poetry run python scripts/gen_durations.py /tmp/dur/*/durations.txt \
      --source 'CI run <master-run-id>, ubuntu-latest 3.12, 4 shards at -n4'
  ```

- **CI closes the loop and warns when it is stale.** `.github/workflows/refresh-durations.yml`
  runs weekly (and on `gh workflow run refresh-durations.yml`): it downloads the latest
  green master run's ubuntu-3.12 transcripts, regenerates the file, and proposes the result
  when the predicted worst shard improves by more than 60 s of wall clock or any test file
  was absent. It pushes and opens the pull request as a GitHub App (variable
  `DURATIONS_APP_CLIENT_ID`, secret `DURATIONS_APP_PRIVATE_KEY`; setup in the workflow's
  first step), whose pull_request event starts `tests.yml` like any PR's. Without the App
  it falls back to `GITHUB_TOKEN`: a pull request if the repository lets Actions open them,
  otherwise a pushed branch plus a tracking issue with the one-click compare link, and it
  then dispatches `tests.yml` on the proposal branch itself (`gh workflow run tests.yml
  --ref`), because events raised by `GITHUB_TOKEN` never start workflows -- PR #343 sat with
  no checks until a human closed and reopened it (2026-09-28). A `GITHUB_TOKEN` PR is also
  held by master's ruleset for an approving review (unattributed changes; PR #386,
  2026-10-05). Independently,
  every pytest job's `pytest_shard.py --verify` reports the weights' age and the absent
  files on its job summary page, and shard 1 of each leg raises a `::warning::` annotation
  when any file is absent (`scripts/pytest_shard.py --balance-json` prints the same
  numbers for a human). There is no per-test timing logger to add: the artifacts ARE the
  per-test timing on the hardware that runs the suite.

### Per-test time budget

**60 s per test phase (setup / call / teardown), 300 worker-seconds per
file**, checked on every CI job by `scripts/check_test_budget.py` from the
`--durations=0` transcript that job already writes. Over-budget phases and
files are listed on the job summary page of every job and raised as
`::warning::` annotations on the pull request from the ubuntu 3.12 leg only
(one set per run, not four). It is a WARNING, never a failure: runner speed
varies by +/-40% job to job and a cold compile cache can triple a build-only
test, so a red here would fire for reasons that are not in the diff and teach
people to re-run instead of read. The point is that a slow test shows up on
the pull request that made it slow, instead of weeks later in a durations
refresh.

Why these two numbers. A job's wall clock is `~100 s fixed + max(shard
worker-seconds / workers, slowest FILE serial)` (`docs/testing-cache.md`),
and `--dist loadfile` runs a whole file on one worker -- so the slowest file
is a FLOOR under every job that draws it, whatever the shard count.
`test_integration_ob09020.py` was that floor on 2026-10-01 at 1329 s; at 4
shards x 4 workers the per-worker share of ~18700 worker-seconds is ~1170 s,
so one file of that size sets the job by itself. 300 s is roughly a quarter
of that share: a file past it is on its way to setting the floor and should
be split or trimmed. 60 s per phase is where a structure test that samples,
or a fit fixture whose budget nobody re-checked, usually shows up first.

**What the over-budget list mostly is, and why it is not all actionable from
a test.** Profiled 2026-10-01 (a phase timer over `prepare`, `build_model`,
`compile_logp`/`compile_dlogp` and gcc): the heavy build-and-evaluate tests
are C-COMPILE-bound on CI, not test-bound. `test_band_autopin_ld`'s mixed-law
test costs 296 s locally with its graph cold and 32-76 s warm;
`test_vcve.py::test_each_vcve_orbit_adds_one_branch[modes0-2-True]` (413 s on
CI) is 220 s cold and 27 s warm, ~90% of it gcc on the two-branch mixture's
logp+dlogp, which is twice a plain model's graph; `test_rossiter.py` is 684 s
on CI and 106 s warm. Only shard 1's compiledir is saved
(`docs/testing-cache.md`), so a graph that lives in shards 2-4 is compiled
cold on every run. Those tests already do no sampling; what would cut them
is a cache that holds their graphs, which is a cache-budget decision, not a
test change. Compiling logp and dlogp in ONE function was measured and does
not help (vcve: 184 s cold, 23 s warm, against 19 s warm separately).

Real findings from the same pass, fixed in the PR that added the check:

| test | before (CI / local) | after (local) | what it was |
|---|---|---|---|
| `test_integration_ob09020.py` (fixture) | 1329 s / 992 s | 283 s | PTDE probed the start once PER RUNG (two ~400 s probes, identical); now once. Photometry thinned 40x instead of 10x (0.37 -> 0.16 s per logp) |
| `test_robust_likelihood.py::test_outlier_prob_at_data_flags_a_planted_outlier` | 352 s / 203 s | 4.5 s | 193 s of sympy: a pinned mass against a live K hint made the engine solve the K relation for `ecc`; seeding e = 0 skips it |

Deliberately left over budget, so a later sweep does not re-litigate them:
`test_runner.py::test_run_without_flag_writes_no_status` (JDE-kept, below),
the two GUI lifecycle tests (`test_runner_lifecycle.py`,
`test_run_endpoints.py`: a real subprocess fit that must reach 100 draws, so
the budget is the run, not the assertion), and `test_integration_kelt4.py`'s
fixture (`tune: 2, draws: 1, chains: 1` already; the time is the polish, the
NUTS compile and a 24 s wrap-up).

```bash
python scripts/check_test_budget.py durations.txt            # summary only
python scripts/check_test_budget.py /tmp/dur/*/durations.txt --annotate
```

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

The same list **on CI** (ubuntu-latest 3.12, run 34888667878, 2026-09-14; 10182 s `call`
against 1817 s `setup`, i.e. 85% / 15% -- the split has not moved). The two heaviest are
build-and-evaluate tests with no sampler, and they are heavy on CI because their graphs
never enter the compile cache: only shard 1's tree is saved, and both live in other shards
(`docs/testing-cache.md`). `test_rm_ltt.py`, the 2026-08 floor file, is 26 s now.

| seconds | test |
|---|---|
| 483.6 | `test_band_autopin_ld.py::test_two_transits_may_use_different_limb_darkening_laws` |
| 375.6 | `test_robust_likelihood.py::test_outlier_prob_at_data_flags_a_planted_outlier` |
| 329.8 | `test_vcve.py::test_each_vcve_orbit_adds_one_branch[modes0-2-True]` |
| 232.5 | `test_runner.py::test_run_without_flag_writes_no_status` |
| 221.2 | `test_rossiter.py::test_rm_system_with_linear_ld_builds` |
| 217.4 | `test_integration_kelt4.py::test_run_fit_kelt4_trace_file_written` |
| 199.8 | `test_run_endpoints.py::test_endpoint_run_lifecycle_start_sampling_stop` |
| 187.9 | `test_runner_lifecycle.py::test_run_lifecycle_status_snapshot_and_graceful_stop` |
| 176.2 | `test_mkparam_in_memory.py::test_restart_file_is_written_for_an_in_memory_run` |
| 141.2 | `test_mulens_acceptance_a.py::...recorded_decomposition[ob09020]` |

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
