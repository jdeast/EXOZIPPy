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

