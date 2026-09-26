# Sampler selection

`src/exozippy/samplers/`: which sampler `sampler: {method: ...}` selects, how the start
population is built, and what each method needs. PTDE (`ptde`/`ptde_async`) and the
gradient-free DE step methods (`samplers/de_metropolis.py`) share the start machinery in
`samplers/_common.py`.

Read this before adding a sampler, changing chain sizing, or declaring a method compatible
or incompatible with a component. Related: `src/exozippy/whitening.md` (where the measured
raw scales the samplers are handed come from), `src/exozippy/run.md`.

## Sampler selection (`sampler: {method: ...}`)

`method:` picks the sampler; `run.py` dispatches on it and everything else in the `sampler:` block is shared vocabulary (`KNOWN_SAMPLER_KEYS`). The values are `nuts` (PyMC NUTS), `numpyro` / `blackjax` (JAX NUTS via `sample_jax_nuts`), `nutpie`, `ptde` / `ptde_async` (EXOZIPPy's parallel-tempered differential evolution -- the recommended non-HMC default), and `demc` / `demcz`. An unrecognized value falls through to the `nuts` branch, as it always has. Omitting `method:` auto-selects from the components' `sampler_requirements()` (`recommended`, else `nuts`), and a method a component declares `incompatible` warns and names the recommendation -- microlensing's non-differentiable Op path is the one live case, and the gradient-free samplers, `demc` included, are not in its incompatible set.

**`demc` / `demcz` (`samplers/de_metropolis.py`)** are PyMC's own gradient-free differential-evolution step methods, wired to the same start machinery PTDE uses:

- `demc` is population DE-MC (ter Braak 2006): the chains **are** the DE population, so an unset `chains:` is sized by `_common.resolve_n_chains` (`2 * n_params`, the same helper and the same warnings PTDE gets per rung) rather than taking run.py's generic default of 4. `demcz` is DEMetropolisZ (ter Braak & Vrugt 2008), which draws difference vectors from each chain's own past states and so needs only a handful of chains (default 4) -- but needs **long** runs for that archive to fill: measured on `examples/kelt4` RV-only (15 params), 500 tune + 500 draws leaves it visibly unconverged (Rhat 3.9) where `demc` at the same length reaches Rhat 1.43.
- Starts come from `_common.resolve_start_population`, not from PyMC. Without it every chain starts at the identical relaxation-engine point, **every difference vector is exactly zero**, and the population can only crawl apart on the step's `scaling` jitter (0.001 raw units). Multi-seed starts and the measured whitening scales are forwarded exactly as PTDE forwards them, and `chain_seed_index` is stamped on the trace the same way.
- `maxtime:` is honored for `demcz` and **refused with a warning** for `demc`: PyMC routes population samplers through `_sample_population`, which swallows the per-draw `callback` in `**kwargs`. Silently accepting it would be the "warning says ignored about a key that is honored" failure in reverse.
- Neither step method writes an `lp` sample stat; run.py's `_compute_lp_from_model` fills it in post hoc, which is the path these fits already take.
- **PyMC's step classes are used directly -- `STEP_CLASSES` names `pm.DEMetropolis` / `pm.DEMetropolisZ`, with no local subclass.** There were two, carrying a `_fix_de_stats` coercion of the `scaling`/`lambda` sampler stats back to scalars: PyMC declared them scalar in `stats_dtypes_shapes` while `astep` returned the `np.atleast_1d` array `Metropolis.__init__` stores, and the trace backend rejected that with `ValueError: setting an array element with a sequence`. Upstream fixed it **below this project's floor** (DEMetropolisZ in pymc 5.26.0, DEMetropolis in 6.0.0; `pyproject.toml` requires `pymc>=6.0.0`), so the patch could not fire on any installable PyMC and was deleted in 2026-08. **The floor is what makes that safe** -- there is a comment on the `pymc` pin saying so; do not lower it. `tests/test_de_metropolis.py` samples both variants end to end and asserts the two stats are present and scalar in the written trace, which is exactly what the patch protected, so a re-regression fails a test rather than crashing a fit.


## Reproducibility (`sampler: {seed: ...}`)

`seed:` is the one seed for the whole run. It reaches `pm.sample`'s
`random_seed` (NUTS and nutpie), `sample_jax_nuts`'s `random_seed`
(numpyro/blackjax), and the `seed=` argument every in-house sampler already
had -- `ptde_sample`, `ptde_async_sample`, `de_metropolis_sample`,
`nested_sample`. Until review 2.14.4 run.py passed no seed to any of them, so
a user could not reproduce their own fit.

**Absent is not the same as unseeded.** With no `seed:` key, run.py DRAWS one
from the OS entropy pool, logs it, and stamps it on the trace
(`posterior.attrs["random_seed"]`); `mkparam` copies it into the restart
file's header. So the default stays "different every run" -- which is what you
want for a fresh fit -- while the run that actually happened stays
reproducible after the fact. A hardcoded default seed would be strictly worse
than none: it would correlate every user's chains while looking responsible.

**The nested backends take the seed two different ways, and ultranest's is a
process-global one.** dynesty has an `rstate=` argument and gets
`np.random.default_rng(seed)` handed to it. ultranest (4.5.0) has no such
argument on `ReactiveNestedSampler` OR on `popstepsampler.PopulationSliceSampler`,
and draws its live-point indices and slice positions from numpy's LEGACY GLOBAL
state -- so `nested._seed_ultranest` calls `np.random.seed(seed)` immediately
before the sampler is constructed, which is ultranest's own mechanism
(`ultranest.solvecompat.solve` does exactly that). Until review 2.4.7 it did
not, so `sampler: seed:` reached the dynesty branch and nothing else while
run.py's startup line promised the user a reproducible rerun. Perturbing the
process-global generator is acceptable only because `nested_sample` owns the
process there -- one sampler is dispatched, the forked workers only evaluate a
deterministic logp, and every draw EXOZIPPy makes for itself goes through an
explicit `default_rng`. If a later ultranest grows a real seed argument, use it
and delete the global.

**`ptde_async` is the one method a seed cannot make reproducible, and that is
a trade, not a defect.** It consumes worker results through
`result_q.get(timeout=...)` -- arrival order -- and whether a proposal is
accepted depends on which partners' states happen to be visible when it lands.
A seed fixes the draws, not the trajectory. Making it reproducible would mean
buffering arrivals into a canonical order, and that IS the synchronous
sampler: determinism costs exactly the asynchrony `ptde_async` exists for.
**If you need a bit-reproducible run, use `method: ptde`.** Synchronous PTDE
is parallel but deterministic on purpose -- it fans proposals out to a shared
pool and then applies accept/reject "in the same per-seed order"
(`polish_seed_starts`), so no worker's scheduling can reach the chain state.
NUTS, numpyro, blackjax and nutpie run independent chains and reproduce
outright. run.py's startup log says which of the two sentences applies rather
than emitting one that is true for one path and false for the other.

Two things a seed does NOT cover, both deliberate: the posterior-spaghetti
draws in the model plots (`run.get_draws`, unseeded so the overlay honestly
shows spread -- see its docstring), and floating-point non-associativity
across a different core count.

## The tuning window is not the logging window (2.4.21)

`gamma` and the ladder adapt during TUNE only, at window boundaries. Those
boundaries used to BE `log_every = (tune + draws) // 20`, so the number of
adaptation windows was `20 * tune/(tune+draws)` -- **asking for more draws
bought less step-size tuning.**

A DC2018 production run (`tune 5000, draws 50000`) got **one** window. It
applied a single sqrt-damped correction, `gamma 0.2695 -> 0.1046`, and then
sampled 50,000 draws at **4.7%** T=1 acceptance against a `target_accept` of
**0.20**, with every rung at 0.030-0.050. The 27-D transport bench
(`tune 385, draws 1158`) got five windows, reached 0.210 at `gamma 0.3239`,
and looked perfectly healthy -- which is exactly how this survived: every
short test gets 6-10 windows, and only production is starved.

`adaptation_window(tune, log_every)` owns the cadence now: at most
`log_every`, so no run ever adapts LESS often than it did, and at least
`MIN_ADAPT_WINDOW` (50) steps, because both consumers read a window's
acceptance or swap rates. That floor is load-bearing -- the first cut used
`tune//20` outright, which is a 15-step window on a 300-step tune, and the
ladder began re-spacing on noise:
`test_ptde_deo.py::test_deo_achieves_higher_round_trip_rate_than_random`
went from hundreds of round trips to zero. Production moves 1 window -> 20;
short runs keep their old cadence exactly.

**What it plausibly explains, and what is not yet measured.** A chain at 3.8%
acceptance barely decorrelates between swap attempts, and production attempts
one every step (`swap_interval` defaults to 1), so a replica carries a stale
`lp` up and down the ladder -- which is the assumption DEO's transport theory
makes and the shape of the 0-1 round trips per 55,000 swap rounds seen on
every sweep event. Whether fixing the cadence restores transport is an
open measurement, not a claim. Tests: `tests/test_ptde.py`'s three
`adaptation_window` / `gamma_adapts` cases.

## `swap_schedule`: DEO, and why `random` is still here

`sampler: {swap_schedule: deo}` is the default and is what you want. It is the
non-reversible Deterministic Even-Odd schedule (Syed et al. 2022), which turns
round-trip transport across the ladder from O(n_temps^2) to O(n_temps).

`swap_schedule: random` is the legacy random-pair schedule and is **diagnostic
only**. It was measured head-to-head against DEO on a real event (review 7.4.4
leg (a), 2026-08-27):

| | round trips | draws to converge |
|---|---|---|
| `deo` | 37/41 | 3,995 |
| `random` | 0/0 | 55,648, or never |

And the failure is not merely slow. **One random arm lost a mode**: it reported
1 mode at weight 1.000, zero inter-mode transitions, 34/34 chains in one basin
-- on an event where every DEO run finds the +/-u_0 pair, and where the -u_0
solution had been seeded and was sitting in that same run's rejected-seed
ledger. A schedule that can silently return half the posterior is not a
performance choice.

**And a ladder on DEO can still make zero round trips -- but round trips are
TEMPERATURE transport, and buying more of them does not buy mode mixing.**
Two things are measured, both on the 27-D Gaussian of
`examples/DC2018/pt_transport_bench.py`, at a FIXED ladder (`n_temps = 24`,
`n_chains = 54`) and 2M evaluations per configuration.

*Round trips are controlled by the path length.* Across `T_max` 4, 16, 50,
200, 1000, 8500 they go 1774, 249, 58, 6, 0-1, 0, with Lambda 2.8, 5.6, 7.8,
10.2, 12.8, 15.8. Lambda 5.6 and 7.8 transport perfectly well, so there is no
"ceiling" in Lambda; and since a longer path at fixed rungs IS a higher
barrier, this says "shorten the path and transport returns", not "T_max is
causal".

*But mode balance does not follow.* With the target bimodal, the cold chains'
far-mode fraction (0.5 is correct) is 0.27-0.35 at a 24-nat barrier and
0.05-0.09 at a 78-nat one, **at every `T_max` from 16 to 8500** -- while an
8-nat barrier equilibrates everywhere, including at `T_max = 8500` where there
are zero round trips, because the DE proposals cross it directly. (DC2018 062
is the same story in production: 41,674 inter-mode transitions at `T=1` with
zero round trips.) A low `T_max` transports and cannot cross; a high one
crosses and cannot transport.

**And the scanned range is the optimistic end.** Those barrier heights were
chosen from the mode report's `delta vs best seed`, which is peak-to-PEAK --
the gap between two optima. Measured peak-to-VALLEY on a real event
(`examples/DC2018/dc18_barrier_profile.py`, DC2018 152, straight line between
the two modes' best draws in raw coordinates with the whitening restored):
two modes **4.3 nats apart peak-to-peak sit either side of a 655-nat
valley**, a factor of 150. A straight line is one path, so 655 is an upper
bound -- but the scan above already fails at 78, so a true barrier anywhere
near this makes the conclusion stronger, not weaker.

So `ladder_health_report`'s rung recommendation is the remedy for the
CRITERION and not for what a reader usually wants it for, and it now says so.
Where basins are far apart the traffic comes from multi-seed starts, the
hot-rung suppressed-mode search (`store_hot_chains`), per-mode evidence
weighting or explicit mode jumps. **And do not shorten the ladder to buy round
trips**: the hot-rung search's reach is `10 x T_max` (2000 nats at the default
200, 500 at 50), and it is what found DC2018 223's truth basin. Full trail:
`notes/pt_round_trip_collapse.txt`. Tests: `tests/test_ptde.py`'s two
`ladder_health_report` cases.

**So why keep it?** One real use, and one cheap one. The real use is as the
CONTROL for diagnosing ladder transport: review 2.4.9 (`ptde_async`'s ladder
does not transport) was diagnosable precisely because async-on-DEO behaves
like sync-on-random, and that comparison needs a known-bad schedule to point
at. The cheap one is that keeping it spares the next investigator from
re-implementing it to rediscover that it is worse.

It is nearly free to keep. Since review 7.4.4's fold, both schedules share one
Metropolis test and one state exchange and differ only in which
`(rung, chain, chain)` triples they attempt -- so PT invariance cannot differ
between them, and `random` costs one generator rather than a duplicated swap
loop. Before the fold, sync carried ~25 lines of duplicated exchange while
`ptde_async` had always shared it.

## PTDE: the proposal path is BIT-IDENTICAL by construction

A rung's population is one `(n_chains, n_raw_elements)` float64 array
(`_common.RawLayout`), not a list of per-variable dicts, and the compiled logp is
called by position with `trust_input` (`_common.PositionalLogp`) rather than
through pymc's dict wrapper. Together those took a DC2018-shaped serial step
(27 elements over 20 raw variables, 8 rungs x 54 chains) from 72 to 34
ms/step, with an identical summed lp.

Three properties make the packing bit-identical rather than merely equivalent
in distribution, and **each is load-bearing** -- `tests/test_ptde.py` pins them:

- the DE partner draw stays **per member** (`_pick_two`), because a batched
  draw consumes the bit stream in a different order and moves every later
  number;
- one `standard_normal(total)` per proposal is the same SEQUENCE as one draw
  per key in key order (numpy's generator fills sequentially);
- the arithmetic is elementwise, so concatenating the operands changes no
  float operation and no rounding.

Two traps that follow. The populations are numpy **rows**, so the tuple-swap
idiom `a[i], b[j] = b[j], a[i]` holds views and the second assignment reads the
first one back -- both samplers copy first. And `PositionalLogp` **must**
coerce with `np.asarray(v, dtype)`: `trust_input` disables filtering, and
`pop[i] + gamma*(...)` on a 0-d parameter yields a numpy SCALAR, which the
numba backend rejects outright ("Vectorized inputs must be arrays") and a wrong
dtype would silently read as raw memory.

Measuring any of this: **warm the pytensor module cache first**. The first
`compile_logp()` in a process pays `cmodule.refresh()` (~20 s), and a naive
end-to-end timing divides that by the step count and reports pure noise -- it
briefly reported a 1.7x speedup for a change that had not been applied.

Remaining headroom, measured and deliberately not taken: the IPC payload is
still a dict (36 us to pickle, 21 us to unpickle per proposal, against 4.3 and
3.1 us for one packed vector). Sending the vector and slicing it in the worker
changes the contract `polish`, `_make_starts`, `describe_proposal` and the
tests all share, so it is its own PR.

## Sync and async share a lot -- and the parity rule for what they share

`ptde.py` and `ptde_async.py` are two loops around one sampler, and the
sharing is deliberate. Two channels carry it, and the second one is
invisible if you only grep for `_common`:

- **`_common.py`** owns the non-statistical scaffolding both call: the
  packing and the DE move (`RawLayout`), the positional logp
  (`PositionalLogp`), the worker pool (`create_pool`, `recycle_pool`,
  `_shutdown_pool`, `_worker_init`, `warn_serial_eval_timeout`), the start
  population (`resolve_n_chains`, `resolve_start_population`,
  `_make_starts`, `plot_start_ensemble`), the gamma rule (`next_gamma`),
  the stop handlers, `LpPlausibilityGuard`, the draw buffers
  (`grow_draw_storage` and its hot sibling `grow_hot_draw_storage`), the
  hot-rung retention (`HotChainRecorder`), the validation of the knobs
  run.py forwards to both (`validate_shared_ptde_args`), and
  the output (`assemble_inference_data`, `stamp_and_log_run_summary`).
- **`ladder.py`** owns the temperature ladder: `_geometric_ladder`,
  `resolve_n_temps`, `ladder_health_report`, `_deo_pairs`,
  `_deo_pair_sequence`, `_record_round_trips`, `_update_ladder_barrier`.
  These lived in `ptde.py`, and `ptde_async` imported nine names from it --
  a second sharing channel that made `ptde.py` simultaneously a sampler and
  the other sampler's library. **Neither sampler imports the other now.**
  Anything one sampler needs from its sibling is a shared module that has
  not been written yet. (`_convergence_check_schedule`, `_safe_progress` and
  `_check_convergence` went to `_common` in the same move; none of the three
  has anything to do with dispatch.)

**The prologue and the epilogue ARE shared**, as of the same change:
`_common.prepare_ptde_run` validates, resolves the ladder and the chain
count, compiles the logp and the conversions, builds the rung populations,
plots the start ensemble and arms the hot recorder; `_common.finish_ptde_run`
refuses an empty run, notes an early stop, assembles the trace, attaches the
hot group and emits the ladder statistics.  Neither is a loop.  The ORDER
inside `prepare` is load-bearing -- `rng` is built and then consumed by
`build_rung_populations`, so anything that reordered it would move every
subsequent random number and break the bit-identical proposal path -- which
is exactly why there must be one copy of it rather than two.  What varies
between callers is passed in: the label, an early-stop detail string, the
summary counters, and `notes` for async's "adapt_ladder never fired"
warning.  `_common.progress_state` likewise owns the nine-key GUI payload
that used to be written out verbatim in both files.

A caution for whoever extends this: the shared wrap-up calls
`ladder.ladder_health_report` through the MODULE, not a from-import, because
`tests/test_ptde.py::test_the_wrap_up_barrier_measures_the_draw_phase_only`
spies on it by patching that attribute.  A from-import here would silently
stop the spy from intercepting and the test would pass while measuring
nothing -- the vacuity shape `docs/testing.md` is about.

**What is STILL deliberately not shared** is the loop itself, and everything
whose shape follows from it: the stop/abort path (sync breaks inline, async runs a
`_maybe_stop` closure over a category state machine), `eval_timeout`
enforcement (sync blocks on a batch `_map_logp_timeout`, async scans
in-flight submissions on a wall clock), the ladder- and gamma-adaptation
windows (sync gets its window free from `log_every`; async has to count
proposals and freeze gamma when the first chain starts recording), the
progress line. Folding those into one function would mean re-deriving the
asynchrony ptde_async exists for -- do not try.

**The hot-rung storage used to be on that list, and it did not belong
there.** `store_hot_chains` was honored by `ptde_async` and IGNORED WITH A
WARNING by `ptde`, so a `method: ptde` fit had no suppressed-mode detector at
all -- `outputs.ledger.discover_hot_modes` had nothing to read -- and a
sync-vs-async comparison was unequal in a way that had nothing to do with
scheduling. Nothing about retaining a thinned copy of the hot rungs depends
on how proposals are dispatched. `_common.HotChainRecorder` now owns the
buffers, the store predicate and the `posterior_hot` assembly, and BOTH loops
call it; the key left `run.py`'s `METHOD_ONLY_SAMPLER_KEYS` asymmetry entry
and is classified as a PTDE-family key, since a ladderless sampler still
ignores it. What legitimately differs at the two call sites is one thing:
WHICH COUNTER THINS -- each chain's own iteration count for async, whose
chains advance independently, and the draw index for sync, which is
step-synchronous. That difference IS the asynchrony; the rest was not.
`tests/test_ptde.py::test_both_samplers_store_the_hot_rungs_they_were_asked_for`
is parametrized over both.

**The rule, which is what review 6.4.6 is about.** 6.4.5 stopped the T=1
draw buffers being preallocated at the full configured `draws` -- ~1.6 GB of
resident memory reserved and touched for draws an early-stopped run never
takes -- and closed, having fixed `ptde.py` only. `ptde_async`, the
production default, kept preallocating (~2.9 GB on a DC2018-shaped run,
before the hot group) because nothing failed when only one of the two was
fixed. So: **a storage or memory fix to one PTDE sampler is not done until
the parity test covers both.** That test is
`tests/test_ptde.py::test_an_early_stop_does_not_allocate_the_draws_it_never_takes`,
parametrized over `ptde_sample` and `ptde_async_sample` -- the same "one
rule, N callers" shape `tests/test_polish.py` uses for `next_gamma`. Add the
arm before the fix, not after.

The parallel code paths most likely to drift, honestly, after the prologue,
the epilogue and the progress payload have been folded: the two
`eval_timeout` mechanisms, the ladder-adaptation blocks (async's has already
learned a windowing fix sync's has not), and the stop/abort wording. None is
a bug today; each is two copies of one intention.

And one measurement worth keeping, because it predicts where the next
duplication will come from: folding logic into `_common` does NOT
monotonically reduce the verbatim overlap between the two files. It went
127 -> 146 lines when `HotChainRecorder` landed and only back to ~129 after
`prepare`/`finish`, because a shared helper needs its full argument list
written out at each of two call sites -- the 32-line `prepare_ptde_run(...)`
call is now the single largest duplicated block. The remedy for that one is
a shared settings object, not another extraction.

**6.4.6 is not the only instance, which is the point.** Reviews 1.4.3 and
2.4.16 were the same shape in the argument surface rather than the storage:
`de_mode_hop` was validated in `ptde_async_sample` (it raises outside
`[0, 1)`) and NOT in `ptde_sample`, while `run.py` feeds the identical
config value to both -- so `de_mode_hop: 1.5` raised under one method and
was accepted as a probability above 1 under the other, in a sampler that
reads it as one. FIXED: both now call `_common.validate_shared_ptde_args`,
which also absorbed the `swap_schedule` check that had been two copies of
one `if`, and
`tests/test_ptde.py::test_both_samplers_reject_the_same_bad_shared_knob` is
parametrized over both samplers and both knobs. So the useful question is never "is this knob
validated" -- it is **"which shared knobs does exactly one of the two
samplers validate, and which shared buffers does exactly one of them
manage?"** Ask it of anything `run.py` forwards to both.
## `de_partner_snapshot`: async takes DE partners from an archive, not from
## whatever is visible

`sampler: {de_partner_snapshot: true}` is the default and is a CORRECTNESS
setting, not a tuning one (review 2.4.20, JDE-proposed).

`ptde_async` chains advance as fast as their likelihood evaluates, and on the
microlensing Op path evaluation cost rises steeply with caustic proximity --
that heavy tail is the whole reason async exists. So a chain is slow BECAUSE
of where it is, and partner states taken "as available" are weighted toward
chains in cheap regions. The proposal kernel then depends on the proposing
chain's own cost, and the plain Metropolis ratio does not correct for it:
detailed balance breaks in a direction that correlates with the physics.

**Time-staleness by itself is NOT the argument**, and the module docstring's
wording invites that mistake. At stationarity `x_a(t1) - x_b(t2)` is the
difference of two draws from the same target, so the time indices do not
enter its distribution at all. What bites is the speed-POSITION correlation
above, and non-stationarity during burn-in, where "the chains represent the
posterior" is exactly what is false.

The fix is one archive per rung, refreshed when that rung's SLOWEST chain
advances, so every chain proposes from the same array. This is the
DEMetropolisZ construction (ter Braak & Vrugt 2008) -- difference vectors
from an archive rather than live states -- and `demcz` already ships it here,
so the validity argument is published rather than invented.

Three properties to preserve if you touch this:

- **The BASE stays `current_state[k][i]`.** `RawLayout.propose` returns
  `pop[i] + gamma*(partners[j1] - partners[j2])`; only the difference comes
  from the archive. Taking the base from the archive would propose from a
  position the chain is not at, against an acceptance test that compares the
  CURRENT lp.
- **The rng stream does not move.** One `_pick_two` over an equal-length
  population, one `standard_normal`, so `partners=pop` is bit-for-bit the old
  behaviour -- which the tests that pin the proposal path depend on.
  `tests/test_ptde.py::test_partners_change_the_difference_vector_but_not_the_base_or_the_stream`.
- **The key is ptde_async-ONLY, and classified as such** in run.py's
  `METHOD_ONLY_SAMPLER_KEYS`. It is the mirror of `rung_thin_factor`: `ptde`'s
  population is synchronized by construction, so there is no archive to take
  a snapshot of.

What it costs is lag, worst exactly where the population is not yet
stationary. One pathologically slow chain freezes its rung's archive; that is
unbounded today, and a max-lag forced refresh is the obvious extension.

**What the evidence does and does not say.** The testable prediction
"stranded chains are the slow ones" FAILED on ab194: the 3 good chains had
lag-1 lp autocorrelation 0.9998 against the 75 stranded ones' 0.9933 -- the
stranded chains moved faster. But lp autocorrelation is movement in lp, not
evaluation wall-time, and per-chain timing was never collected
(`collect_rung_timing` was off). So the mechanism is UNTESTED rather than
refuted, the fix rests on the correctness argument alone, and those numbers
must not be cited as support for it.

## `de_mode_hop`: the counters the adapter reads must share one window

ter Braak's gamma=1 mode hop (`sampler: {de_mode_hop: p}`, default 0.0 = off)
is deliberately over-sized and mostly rejected, so both samplers SUBTRACT the
hop accept/propose counts from the T=1 rate the gamma adapter reads. Letting
hops depress that rate would make the adapter shrink gamma, degrading
within-mode sampling as the price of attempting hops.

**A counter subtracted from a windowed counter must itself be windowed.**
`ptde` zeroed `n_accept`/`n_propose` at each `log_interval` during tune and
left the hop counters cumulative, so from the second window on the
denominator was short by every earlier window's hops: the measured rate ran
above 1.0 (1.222, gamma 0.872 -> 2.155, away from target) and, once the
numerator went negative, the `ar_T1 > 0` guard ended the adaptation for the
rest of tune in silence, freezing a garbage gamma into the draw phase
(review 1.4.3). `ptde_async` was never affected -- it keeps SEPARATE window
counters for the adapter and lets the hop counters run cumulatively for its
report. After `ptde`'s unconditional tune -> draw reset its hop counters span
the draw phase, which is the same window `n_accept`/`n_propose` report on.
Both samplers log the hop acceptance at wrap-up through one
`_common.log_mode_hop_summary`, so the message cannot drift again.

## `cores`: one rule, and `None` means AUTO

`_common.default_cores()` is the single definition of "how many cores does a
parallel stage take when nobody named a number": `CORE_FRACTION` (0.75, in
`constants.py`) of the physical cores, capped at `n_phys - 1` so the OS and the
user's shell keep one. `sampler: cores: N` overrides it outright, and `run.py`
forwards whatever it resolved to every stage that forks -- sampling, the
pre-sampling seed polish, and the hot-mode candidate polish.

**`cores=None` means AUTO in every one of them, never serial. Serial is
`cores=1`.** That is not a style preference; it is review 6.11.3. The rule had
three hand-written copies -- `run.py` from the named constant, `create_pool`
hardcoding 0.75 under a comment claiming it was the "same formula", and
`nested.py` hardcoding it while dropping the `n_phys - 1` arm -- and, worse,
`polish._resolve_polish_cores` read `None` as *serial* while `create_pool` read
it as *auto*. A caller that passed nothing therefore got the whole machine in
one function and a single core in another, which is exactly how a hot-mode
polish came to hold 1 of 36 cores for 38 minutes with nothing in the log. If
you add a stage that forks, call `default_cores()` for its fallback and accept
a `cores` argument that `run.py` can fill.

**`cores <= 0` is that same automatic grant** (review 2.4.8). It is the `None`
rule's loophole: `0` is not `None`, so the three resolvers each did their own
thing with it -- `create_pool` took `min(0, total_proposals)` and ran serial,
`_resolve_polish_cores` swept it into its `n <= 1` serial arm, and `nested.py`
read it as AUTO because `cores or default_cores()` treats 0 as falsy -- so one
written number produced two different behaviors within a single run, and a
negative value produced a negative pool size in the third. `run.py`'s
`resolve_cores_setting` now normalizes `<= 0` to the `None` sentinel at the
parse boundary and **warns** rather than raising or clamping silently (rope,
not gates: `cores: 0` most plausibly means "let the machine decide", so the run
takes that reading and the message says which reading it got and that
`cores: 1` is how to ask for serial). All three resolvers carry the same arm
anyway, so a direct caller -- a test, a script, the GUI -- lands where `run.py`
would have put it.

## eval_timeout: what it does, and where it is enforced

`eval_timeout` is opt-in (default None) and exists for a logp that can HANG --
the near-caustic VBM evaluation -- not for one that is merely slow. Semantics
worth knowing before changing it:

- **It has no effect without a worker pool** (`cores <= 1`), which
  `warn_serial_eval_timeout` says at startup: there is no process to time out
  against, and in serial mode the evaluation has already completed by the time
  anything could scan for it.
- **A timeout tears down the WHOLE pool.** There is no way to kill one hung
  worker in a `multiprocessing.Pool`, so every in-flight evaluation is written
  off and resubmitted with a fresh proposal. A written-off result that arrives
  anyway finds its submission id gone and is dropped, so nothing is ever
  processed twice.
- **ptde_async scans on BOTH the empty-queue and the result path**, paced by a
  wall clock. The scan used to live only inside `except queue.Empty`, a state a
  healthy run essentially never reaches -- so a hung slot froze for the whole
  run, and a hung T=1 slot froze the run itself, since `min(per_chain_draws)`
  never advanced (review 1.4.1). Do not move it back.
- **The async poll is bounded (1 s) even with no eval_timeout**, because
  `_maybe_stop` runs on the poll: otherwise a Ctrl+C, a scheduler SIGTERM or
  maxtime is only noticed when some slot happens to finish, and the user's
  second signal then throws away every draw already collected (2.4.3). If a
  stop finds NOTHING completing, the loop leaves the in-flight evaluations
  behind and goes to the save path rather than waiting on results that are not
  coming.
- Both samplers tear the pool down with `_common._shutdown_pool`, never
  `close()` + `join()`: the workers ignore SIGTERM by design, so `join()` on a
  wedged worker never returns (2.4.1). **`polish.polish_raw_starts` now does
  the same** -- it used to `close()` + `join()`, which was survivable only
  while nothing in that stage could wedge a worker.

**The seed polish enforces it too, but nothing configures it yet (3.4.4).**
`ptde.polish_seed_starts` and `polish.polish_raw_starts` take an
`eval_timeout` with the same semantics as above, plus a `pool_recycler`
callable -- `_common.recycle_pool`, supplied by whoever OWNS the pool, which
for this stage is `polish_raw_starts`. The recycler is not a style choice:
`polish_seed_starts` is handed a pool and does not know how many workers to
fork, so the owner has to learn which object it now owns or its own teardown
closes the corpse and leaks the live one. Both default to `None`, and
**`run.py` passes neither**, so today's behaviour is unchanged: `sampler:
eval_timeout:` is in `run.METHOD_ONLY_SAMPLER_KEYS` as a PTDE-family key, and
`warn_method_only_sampler_keys` tells a `demc`/`demcz`/`nested` user it is
IGNORED -- which honoring it in a stage that runs under every method would
turn into a lie for a gradient-free model. That partition is review 2.3.6's
ruling; re-opening it is its own change. What the polish gained regardless is
the mid-batch heartbeat (see `run.md`), which needs no timeout to tell
computing from hung.

## The seed polish is asynchronous on a pool (2.4.14)

`ptde.polish_seed_starts` is the DE engine behind `polish.polish_raw_starts`
on every model whose gradient graph does not build (finite-source and
binary-lens microlensing). It used to be a synchronous batch loop: one sweep =
one batch of `n_seeds * pop_size` proposals, one barrier. That is the defect
`ptde_async` was written to remove, and it was measured on DC2018-226 (32
workers, job 46562457): each ~23 s sweep had all 32 workers busy for ~6 s and
then drained 19, 15, 10, 8, 6, 4, 2, 1 while the batch's dearest VBM proposals
finished -- 15.3 of 32 workers computing on average, 43% per-worker CPU over
ten hours -- because a population that has partly migrated proposes a few
very expensive points per sweep and the whole node waited for them.

**On a real pool the polish now follows `ptde_async`'s procedure** (JDE
2026-09-15): every (seed, member) slot keeps ONE proposal in flight, results
are consumed in arrival order, each is accepted or rejected against its own
member's current lp, and the slot's next proposal is drawn from the population
as it is THEN. The budget is unchanged in meaning -- `n_steps` sweeps, counted
as `n_steps * pop_size` completed proposals per seed -- and the gamma
adaptation, the opt-in improvement window, the trust radius, best-visited
tracking and the wrap-up lines are shared with the synchronous engine through
`_accept` / `_end_of_sweep`, so the two cannot drift on the statistics.
`eval_timeout` is enforced the async way: stale in-flight submissions are
found on a wall clock, a stale one writes off EVERY in-flight submission (a
pool cannot lose one worker), the pool is recycled through `pool_recycler`,
and the written-off slots are resubmitted; a result that raced the write-off
finds its id gone and is dropped.

**What it costs is exactly what `ptde_async` costs under "Reproducibility"
above:** the trajectory depends on arrival order, so the DE-path START of a
pipeline run on `cores > 1` is no longer bit-identical run to run (the 44-event
survey's 150-sweep numbers for DC2018-226 were reproduced bit for bit by the
synchronous engine on a different node, and will not be by this one).
`asynchronous=False` on either function restores the synchronous engine; the
serial path and a bare-`map` pool always run it, and the L-BFGS path has no RNG
and is untouched. Nothing in the `sampler:` vocabulary selects it yet -- that
is a config-vocabulary decision like `eval_timeout`'s (2.3.6), and a caller
that needs determinism has `cores: 1`.

**Measured head to head on the event that motivated it** (DC2018-226, 32
workers, five 150-sweep restart legs from the same degenerate seed, jobs
46629651 synchronous and 46630701 asynchronous, 2026-09-15):

| leg | sync s | sync sweeps/min | async s | async sweeps/min |
|---|---|---|---|---|
| 1 | 175 | 52 | 121 | 74 |
| 2 | 533 | 17 | 50 | 180 |
| 3 | 59 | 154 | 45 | 201 |
| 4 | 103 | 87 | 38 | 236 |
| 5 | 70 | 128 | 38 | 240 |
| 750 sweeps | 940 | | 292 | |

Same lp plateau (-185.5k on both seeds either way), 3.2x less wall clock, and
the asynchronous rate CLIMBS as the population settles where the synchronous
one collapses whenever a few proposals wander into the expensive region
(legs 2 and 6 of the synchronous run: 17 and 20 sweeps/min). The expensive
proposals are still evaluated -- asynchrony removes the idling, not the cost.

The wrap-up now also reports, per seed, how many population members NEVER
accepted a move and the population's median and max distance from the best
point. At the fixed 2.38/sqrt(2D) step the T=1 acceptance is ~0.3%, so a
polish that ran for thousands of sweeps can be one migrant plus 63 members
still at their birth positions; that population proposes 70-unit throws for
the rest of the run, and until this line existed nothing said so.
Tests: `tests/test_polish.py`, the `_CallbackPool` block.

## Chain starts

`store_hot_chains` (ptde_async only) takes `auto` / true / false / an integer
thinning factor; **an integer <= 0 means OFF**, the same as `false`, and not
the maximum retention `max(1, ...)` used to produce (1.4.2). An unrecognized
STRING still raises -- a misspelled opt-in is how a mode search silently stops
running.

`_make_starts` measures the population it built against the probe scales and
warns when the between-chain spread is under 1.0x of them
(`warn_if_starts_underdispersed`). Rhat's between-chain term only means
something if the chains start FURTHER apart than the posterior is wide, and
`convergence.good_chain_mask`, `converged_on_tail` and the min_ess/max_rhat
early stop all inherit that assumption -- so a restart seeded from a previous
run's posterior draws can stop a fit that never mixed. It is reported, never
corrected: a run past D = 500 is under-dispersed by construction, since the
scatter factor is `min(sqrt(500/D), 3)`.

Explicit `initvals` are consumed positionally, one per chain, and a
wrong-length list RAISES (it used to be a bare `assert`, which `python -O`
compiles out, leaving chains paired with the wrong starts).

**The params file DECLARES whether its seeds want dispersing; the sampler does
not guess** (`overdisperse:`, review 8.3.3). K seeds mean one of exactly two
things and nothing at run time can tell them apart: the user is seeding at K
MODES, each of which still wants scattering, or they are iterating from a file
whose seeds are ALREADY properly dispersed -- joint posterior draws off a
finished fit, spread across its covariance by construction. So the writer says
which. `mkparam` writes `overdisperse: false` for a multi-seed restart file (a
true statement about the seeds' ORIGIN, not a claim that the fit converged) and
`true` for its default single-seed one; `utilities/mmexofast_to_params` writes
`true` (its seeds are single optima, one per solution). An **absent** key means
`true` -- the safe direction for a hand-written file, since over-dispersing a
good seed set costs some burn-in while under-dispersing a bad one makes Rhat
read ~1.00 on chains that never mixed. It is a reserved NON-parameter key and
is stripped by `ConfigManager` before anything downstream sees it
(`src/exozippy/config.md`).

`false` means "use these seeds exactly as they are": every chain starts at its
round-robin seed with no jitter, and the `max_exact = n_chains // 2` budget
does not apply. Two guards come with it, both at sampler start:

- **One seed plus `false` RAISES.** Every chain would start at the identical
  point, so every DE difference vector is exactly zero and the population can
  never move apart.
- **Few unique seeds WARN, in two tiers.** Below `2 * n_params` (the default
  chain count, and ter Braak's mixing recommendation, so the tier has headroom)
  it is a mixing complaint. Below `de_span_floor(n_params) = n_params + 2` it
  stops being one: a DE proposal for member i draws its difference vector from
  the OTHER members, so n-1 members span at most n-2 directions and it takes
  `n >= n_params + 2` to span parameter space at all. Below that the population
  sits in a proper subspace whose only escape is `DE_JITTER = 1e-4`, and
  off-hull diffusion goes as `jitter*sqrt(steps)`, so covering ONE whitened
  sigma takes ~1e8 accepted steps -- ergodic in principle, hopeless in
  practice. The escalated message says so. `de_span_floor` is the ONE owner of
  that number, shared with `warn_if_population_degenerate`, which asks the same
  question of the CHAIN count.

**The threshold check is here and not in `mkparam`**, and that is structural:
`n_params` belongs to the NEXT fit's model, which may differ from the one that
produced the seeds (added data, a changed parameterization), so the writer
cannot evaluate it. What the writer CAN do is record what it measured, and
`mkparam`'s header comment carries the min bulk-ESS and max split-Rhat of the
fit its seeds came from. Those are a RECORD and deliberately **not** a gate.
A plain "did it converge?" boolean would be the wrong gate because nobody
reruns a well-mixed fit: the population of fits `mkparam` processes is selected
for being unsatisfactory, so the flag would read False almost always and
`overdisperse: false` would be dead code. The two also gate different failures.
CONVERGENCE (max Rhat) asks whether the seeds come from the right distribution
at all; MIXING (min ESS) asks how many EFFECTIVELY INDEPENDENT seeds there are
-- K seeds off a chain with bulk-ESS `n_eff` are ~`n_eff` independent points,
so the affine-hull argument above really applies with `n_eff` and not with K,
and forty seeds off an ESS-5 chain span at most a 4-dimensional hull. And
seeds drawn from an unconverged or mode-stuck posterior are properly dispersed
with respect to what was SAMPLED while being badly under-dispersed with respect
to the TRUE posterior; no local test on the seed list can detect that, which is
exactly why the numbers are printed for a human. `warn_if_starts_underdispersed`
still runs on an `overdisperse: false` population and will usually fire: that is
correct, since Rhat is compromised either way, and it is measurement rather than
contradiction. Tests: `tests/test_overdisperse_declaration.py`.
