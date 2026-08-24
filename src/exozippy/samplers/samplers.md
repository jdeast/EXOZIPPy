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
  wedged worker never returns (2.4.1).

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
