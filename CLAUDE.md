# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install / update dependencies
poetry install
poetry update          # after git pull

# Run all tests (runs in parallel by default: -n 6 --dist loadfile, set in
# pyproject.toml addopts). --dist loadfile pins each file to one worker so
# module/session-scoped fixtures are shared, not rebuilt per worker.
poetry run pytest

# Run a single test -- add -n0 to disable the parallel workers (faster startup
# and readable output for one test); the default -n6 otherwise spawns 6 workers.
poetry run pytest tests/test_config_healing.py::test_name -n0 -x

# Run the fitter on an example
cd examples/kelt4rvonly && poetry run exozippy kelt4.yaml
cd examples/ob140939   && poetry run exozippy ob140939.yaml

# Dump the full repo for AI review (see scripts/dump_code.py for flags)
poetry run python scripts/dump_code.py

# pytest Timeout inside pytensor cmodule.py, blaming an innocent test: the
# compile cache grew until refresh(), which opens every entry, neared the 300s
# cap. Its tmp*/ dirs ARE the cached modules -- prune by age, never rm them.
poetry run pytensor-cache cleanup
```

`system.prepare()` must be called before `system.build_model()` in any script or test.

## Code style

Use plain ASCII in new code, comments, docstrings, and documentation -- avoid unicode punctuation and symbols. Use `->` not `→`, `--` or `-` not `—`/`–`, `...` not `…`, `sigma`/`Delta`/`chi2` not `σ`/`Δ`/`χ²`, `+/-` not `±`, `~` or `approx` not `≈`, `x` not `×`. Plain ASCII decorative separators (e.g. `# ---`) are fine; avoid box-drawing characters (`─` etc.) for these. This is a convention for new/edited content only -- do not go out of your way to rewrite unrelated existing unicode.

## Architecture

EXOZIPPy is a component-agnostic wrapper around PyMC. A user writes two YAML files — a system config (`*.yaml`) and a parameter override file (`*.params.yaml`) — and calls `exozippy <config.yaml>`. The high-level flow in `run.py` is:

All component-specific logic is handled within the components. The higher level codes (run, config, system, etc) rely solely on the generic component blueprint, not specific implementations.

```python
system = System(config, user_params)
system.prepare()           # I/O + registration + symbolic solving
model  = system.build_model()
pm.sample(...)
```

`run.run_fit(config, user_params=None)` accepts in-memory dicts for both arguments (no YAML files needed; data paths resolve relative to cwd). When `user_params` is omitted it falls back to reading `config["parameter_file"]` from disk.

### The six lifecycle stages

`System.prepare()` drives stages 1–3; `System.build_model()` drives stages 4–6.

| Stage | Method | What happens |
|-------|--------|--------------|
| 1a | `comp.load_data(system)` | File I/O; push data-derived hints to ConfigManager |
| 1b | `comp.build_maps()` | Build integer index arrays (`*_map`) linking children to parents |
| 2 | `comp.register_parameters(system)` | Declare `comp.manifest`; push component-level hints |
| 3 | `config_manager.finalize_user_params()` | Symbolic relaxation engine resolves all initvals/scales |
| 4 | `comp.build_tensor_maps()` | Auto-convert `*_map` numpy arrays → PyTensor int32 variables |
| 5 | `comp.add_parameter(model, name, system)` | Materialize PyMC nodes in topological order |
| 6 | `comp.build_likelihood(model, system)` | Add `pm.Normal`/`pm.Potential` observational constraints |

### ConfigManager and the relaxation engine

`src/exozippy/config.py` is the initialization backbone. It:

1. Loads all `defaults.yaml` files (one per component subdirectory) into `base_defaults`.
2. Loads all `symbolic_physics.py` files; calls `get_symbol_map()` to translate abstract symbols to indexed paths (`star.0.mass`, `lens.0.t_E`, …) and collects `RELATIONS` (SymPy equations).
3. Manages a **provenance ranking** system. Higher rank wins when values conflict:
   - `RANK_USER = 100` — from `params.yaml`
   - `RANK_DERIVED_DATA = 60` — from data (e.g., RV offset from median)
   - `RANK_DERIVED_MIXED = 40` — solved using a mix of user + defaults
   - `RANK_DEFAULT = 20` — from `defaults.yaml`
   - rank 30 is used for microlensing distance hints (overrides 10 pc default, yields to user)
4. Runs a **relaxation engine** (`resolve_and_validate_parameters`): iteratively substitutes known values into the SymPy equations to derive unknowns and detect contradictions.
5. Collects `init_scale` values from defaults, component hints, user sigmas, and the engine's own solved-value sync. These are **preliminary only** (see "Whitening" below): sampled parameters get their real scale from the startup probe, and soft-bound barrier steepness is measured numerically post-whitening (the old sympy forward/backward Jacobian scale passes are deleted). `init_scale` is **not user-facing**: entries in a params file are stripped with a warning at ConfigManager construction.

Components push hints via `config_manager.add_hint(path, value, rank)` during stage 1–2. The hint system is the correct way for components to propose data-driven initial guesses; they are layered in after defaults but before the solver runs.

Manifest entries take an `"overrides"` dict (`{field: value_or_per_element_list}`, forwarded to `resolve(internal_overrides=...)`) for component-computed per-element **defaults**. Unlike the manifest's other options, which are merged *over* the resolved config and so beat the user's params file, these are layered *under* it — use them whenever the user must still be able to override. `NaN` in a per-element list means "leave this element alone"; `±inf` is a real bound and is applied.

`src/exozippy/components/defaults.yaml` (no component key, straight parameter names) holds root-level defaults shared by several components; `resolve()` layers the component's own block over it. Today it carries only the GP blueprint.

### Component structure

Each component lives in `src/exozippy/components/<name>/` and contains:
- `<name>.py` — class inheriting `Component`; implements the 6-stage methods
- `defaults.yaml` — default values, bounds, units, and expression wiring for every parameter
- `symbolic_physics.py` — SymPy `RELATIONS` (equations) and `get_symbol_map()` (maps abstract symbols → indexed YAML paths); must set `comp_key = "<yaml_key>"` to match the YAML block name
- `physics.py` — PyTensor/numpy implementations decorated with `@register_physics`; function name must match `func_name` in `defaults.yaml`

The **factory** (`factory.py`) auto-discovers all `Component` subclasses by scanning subdirectories; the YAML key used to instantiate a component is the lowercase class name (or `yaml_key` class attribute if set). No registration step is required for new components. Abstract intermediate bases are skipped (`inspect.isabstract`), so a shared base can leave `Component`'s abstract methods unimplemented and never be instantiated.

The four data components (`rvinstrument`, `transit`, `mulensinstrument`, `astrometryinstrument`) subclass `Instrument(Component)` (`components/instrument.py`), which owns their shared scaffolding.

`Instrument._sort_by_time(df)` sorts each data file ascending by time at read time, before any column is split out or anything is derived from the times — so the observable, errors, detrend columns and per-epoch side arrays (mulens observer positions, astrometry parallax factors) stay aligned by construction. **Per file, never globally**: the concatenated arrays must stay contiguous per instrument or `_build_block_detrend`'s block-diagonal row ranges and mulens's row-aligned `observer_pos` both break silently.

`Instrument._read_data(i, roles, detrend=False)` is the single shared file reader every child calls in `load_data` (it replaced the per-child read/mask/sort triplet). Per file it applies, in order: the optional `columns:` layout, the optional `mask:` row exclusion, the optional time-system handling (`time_offset:` added to every raw time, then `time_scale:`/`time_frame:` conversion to BJD_TDB), and finally the per-file time sort. `roles` names the canonical columns (must start with `"time"`); the returned DataFrame is positionally indexed in that order, detrend columns after. With none of the keys set it is byte-for-byte the old read.

`columns:` maps role names to 0-based file column indices (`columns: {time: 3, rv: 0, err: 1, detrend: [4, 6]}`). Unnamed roles keep their canonical position; when `columns:` is present, detrend columns must be listed explicitly (no "rest of the columns" guess). Structure is validated at construction, role names at load (only the child knows its roles; astrometry's depend on the file's `mode`).

Time system: default is BJD_TDB in, untouched. `time_scale:` (utc/tai/tt/tdb/tcb/tcg/ut1, `ut` = ut1) and `time_frame:` (jd/hjd/bjd) convert to BJD_TDB at load via astropy (Eastman, Siverd & Gaudi 2010 algorithm: strip the input frame's light-travel term by fixed-point iteration, scale-convert, re-add the barycentric term). Conversion demands absolute JDs (raise if min < 2e6 -- point users at `time_offset`); jd/hjd frames demand **user-set** star ra/dec (`_time_coord` refuses defaults.yaml placeholders; `star_ndx` on the file entry picks the star). frame=bjd scale-only conversions need no coordinates (the barycentric term cancels exactly). Optional `time_location:` (observatory name or [lon, lat, height]; default geocenter, up to 21 ms unmodeled) and `time_ephemeris:` (default `builtin`, ~us; `de440` ~ns, needs jplephem). Not modeled -- all below the ~40 us float64 quantization of a single-float absolute JD: TT(BIPM), Shapiro delay, proper-motion/parallax evolution of the direction. MMEXOFAST seeding reads the raw files itself, so `MulensInstrument._reject_time_spec_with_mmexofast` hard-errors when both are active. Tests: `tests/test_instrument_time_columns.py`.

`Instrument._apply_mask(df, i)` (called by `_read_data` after column selection) drops excluded rows per an optional per-file `mask:` config key — a flag-file path (one 0/1 per data row, nonzero = exclude), a boolean list, or a list of 0-based row indices, all referring to the file's ON-DISK row order (mask before sort, so indices from external tools stay valid). All-points-masked and length/range mismatches raise. Tests: `tests/test_instrument_mask.py`.

Optional detrending against extra data columns (columns past the error column, one coefficient per column per instrument, block-diagonal so coefficients never mix) is supported by `rvinstrument`, `transit` and `mulensinstrument` (magnitude space there). `astrometryinstrument` has none — its 2-observable modes would need per-channel coefficients.

### MMEXOFAST integration (`components/mulensing/mmexofast_support.py`)

The lens block's `mmexofast:` key drives three behaviors: an explicit JSON path loads seed initvals/scales (stage 2, `Lens._load_mmexofast_seeds`) AND applies the JSON's bad-data mask (`excluded_points` → the instrument `mask:` machinery; skipped for files with a robust `likelihood:` -- see that section) and error factors (`errfacs` → `err_scale` initval hints) in `MulensInstrument._resolve_mmexofast` (stage 1a, before the photometry is read); absent-key-with-insufficient-start-values (or `mmexofast: auto`) runs MMEXOFAST itself on the raw light curves (`renormalize_errors=True`, cached at `<prefix>_mmexofast.json`, extra fitter kwargs via `mmexofast_options:`) and consumes all of the above; `mmexofast: false` opts out. "Insufficient" = the params file lacks a start value for any required observable (t_0/u_0/t_E; +rho if finite_source; +s-or-log_s/alpha/q if binary). The mmexofast package is imported lazily inside `run_or_load` only — nothing else may import it (PyPI publishability; see the microlensing group note in pyproject.toml). Newer MMEXOFAST adds `jd_offset` to every epoch in the JSON; seed extraction subtracts it so t_0 lands in the data's own time system. `examples/DC2018` is the end-to-end workflow (one cluster job per Roman Data Challenge event); tests: `tests/test_mmexofast_support.py`, `tests/test_multiseed_mmexofast.py`.

### Gaussian-process noise (`components/gp.py`)

Optional, per data file, off by default. A file gets correlated noise by naming a celerite2 kernel on its config entry — `gp: rotation` (`RotationTerm`, spot modulation), `gp: sho` (`SHOTerm`, granulation/generic red noise), or `gp: [rotation, sho]` for their sum. Absent or `gp: none` keeps the independent-Gaussian likelihood, byte for byte.

`Instrument` owns the lifecycle in three hooks the wired children call: `_prepare_gp(time, err, inst_map, user_factor)` at the end of `load_data`, `_register_gp(manifest)` in `register_parameters`, and `add_observation_likelihood(name, mu, sigma, observed)` in place of the final `pm.Normal`. `components/gp.py` owns the `gp:` vocabulary, the per-term parameter tables, and the kernel constructors, so a new kernel is a table entry there plus a `defaults.yaml` block.

Design points worth not rediscovering:
- **celerite2 requires ascending times and does not check.** `_prepare_gp` records one sort permutation per GP file; the times, `mu`, `sigma` and data all go through it in `add_observation_likelihood`. `tests/test_gp.py` pins order-invariance.
- Hyperparameters are **full-length (`n_elements`) vectors**, so a user path resolves by instrument name the same way every other instrument parameter does (`standardize_param_names` rewrites names to *global* indices, so a compacted GP-only vector would mis-address). Files that did not opt in are pinned fixed via the `"overrides"` channel — free to the sampler, still user-overridable.
- Quality factors are sampled as `gp_*_log_q*` (base 10, like `log_s` in mulensing); amplitudes and periods are sampled linearly in the data's own units so literature priors go in as written. The linear `Q0`/`dQ`/`Q` are `pm.Deterministic`s. SHO is a stochastically-driven damped **simple harmonic oscillator** (not shot noise); `Q > 1/2` underdamped, `< 1/2` overdamped, and the kernel switches formula at `Q = 1/2` (a logp kink, though both branches are `maximum(..., eps)`-guarded so gradients stay finite).
- The **amplitude seed is `median(err)`**, the white-noise level. Anything derived from the observations (their scatter, or the point-to-point scatter of sparsely sampled RVs) measures the signal the physical model is supposed to explain, and seeding there invites the GP to eat it.
- `astrometryinstrument` sets `supports_gp = False`: two observables per epoch (dE/dN or sep/PA) in different units cannot share one amplitude, so a `gp:` key there raises instead of being ignored.
- **Transit imposes no sampler constraint.** It used to exclude `numpyro`/`blackjax`: exoplanet-core through 0.4.0rc1 wired the limb-darkening Op's JAX conversion to the raw FFI call, so the funcified logp evaluated but `jax.grad` of it raised and the JAX samplers died at HMC init. Fixed in exoplanet-core 0.4.0rc2 (exoplanet-dev/exoplanet-core#144), which is the floor in `pyproject.toml` — **do not lower it**, or the JAX samplers silently break again. The only exoplanet-core ops used are `kepler` and `quad_solution_vector`, both now JAX-differentiable.
- **A GP imposes no sampler constraint** — celerite2 registers a JAX conversion for its PyTensor ops (`@jax_funcify.register(_CeleriteOp)`), so `numpyro` works (verified end to end on 1/2/4 chains). Do not copy transit's numpyro exclusion here by analogy: that one is about exoplanet_core's LD op, which genuinely has no differentiable JAX path.
- **Plots**: `_compile_gp_plotters(system)` (called from each child's `compile_plotters`) compiles two per-file evaluators of the *pure* GP conditional mean (`include_mean=False`, so celerite2 subtracts the physical model it conditioned on): `gp_mean_at_data(system, point)` returns an `(n_total_obs,)` vector, zero for non-GP files, and `gp_mean_on_grid(system, point, i, t_grid)` evaluates file `i` on a sorted grid. Convention: **unphased panels show the GP** (it is part of the model the likelihood fits), **phased panels remove it from the data** (otherwise the fold smears). RV draws one extra physical+GP curve per GP instrument, spanning only that instrument's own data range; transit's unphased curve is already per-instrument so the GP goes straight in; mulens adds the GP in the instrument's own magnitudes *before* `_align_mag` maps it to the reference flux system, since that map is nonlinear in magnitude. Both evaluators are re-run per posterior draw, so the spaghetti shows the GP's uncertainty too.

### Robust observation likelihoods (`components/likelihood.py`)

Optional, per data file, off by default -- with no `likelihood:` key the model is byte-for-byte the plain `pm.Normal`. A file opts in with `likelihood: hogg` (marginalized inlier/outlier Normal mixture, Hogg+2010: `(1-out_frac)*N(mu,sigma) + out_frac*N(mu, sqrt(sigma^2+out_scale^2))`, added as a `pm.Potential`) or `likelihood: studentt` (observed `pm.StudentT`; `t_log_nu` sampled base-10, linear `t_nu` reported as a Deterministic -- exactly the analytic marginal of a per-point hierarchical error model, so don't add an explicit per-point-variance variant). The plumbing deliberately mirrors the GP feature: `components/likelihood.py` owns the vocabulary/param tables/logp builders, `Instrument` owns three hooks (`_prepare_robust` in load_data, `_register_robust` in register_parameters, and the shared `add_observation_likelihood` dispatcher), parameters are full-length `n_elements` vectors with non-opted files pinned via the `"overrides"` channel, and `astrometryinstrument` sets `supports_robust_likelihood = False` (two observables, one out_scale unit -- same reason as its GP opt-out). Design points:

- The instrument's `err_scale`/`jitter_variance` sigma stays the **inlier** scale; `out_scale` (data units, per-component defaults.yaml override like the GP amplitudes) adds in quadrature only in the background component. Its data-driven hint is `10 x median(err)` so the components start separated and cannot swap roles; `out_frac` is capped at 0.5 for the same identifiability reason.
- `gp:` + `likelihood:` on one file raises: celerite2's closed-form marginal is Gaussian-only.
- The mixture logp is composed with `pt.logaddexp`, never `where` over branch logps (the JAX where-trap). Verified by actually sampling with `nuts_sampler="numpyro"`, per the standing rule.
- `Instrument.outlier_prob_at_data(system, point)` returns per-point posterior outlier probabilities (`(n_total_obs,)`, zero for non-hogg files; lazily compiled against `system.plot_params`, evaluate per posterior draw) -- the auditable replacement for a hard bad-data mask. `examples/DC2018` turns `likelihood: hogg` on for every light curve.
- **A robust file does not consume the MMEXOFAST mask.** `apply_excluded_points` skips files whose `likelihood_kinds[i]` is set: the robust likelihood supersedes the frozen hard mask (and keeps the points auditable). MMEXOFAST still renormalizes internally -- its chi2-based anomaly/grid search has no robust likelihood of its own and needs the rejection to find the right basin -- and its errfacs still seed `err_scale` (they are inlier scales, consistent with the mixture's). A user's explicit `mask:` always applies.

Tests: `tests/test_robust_likelihood.py`.

### Parameter system (`parameter.py`)

`Parameter` is the universal node wrapper. Key points:
- All numeric fields (`initval`, `init_scale`, `lower`, `upper`, `mu`, `sigma`) are stored in **internal units** after `__post_init__` applies the unit conversion factor.
- `unit` is the user-facing unit (from `defaults.yaml` or user override); `internal_unit` is the math unit.
- `sigma = 0` → parameter is fixed. `sigma > 0` → Gaussian potential applied. No sigma → uniform prior on `[lower, upper]` via logit transform.
- Symbolic PyTensor nodes passed as `initval` are preserved as-is (no unit conversion applied).
- `build_pymc()` uses non-centered parameterization: raw `N(0, 1)` mapped to physical space via logit or linear scale + shift.

### Whitening (`whitening.py`)

The whitening scale of every sampled element is **measured from the data at startup**, not hand-tuned. `build_pymc()` builds the model with *preliminary* scales (`defaults.yaml` `init_scale`, optional; missing values fall back to a fraction of the bound span) and stores the whitening constants in `pytensor.shared` variables. `run.py` then compiles logp once and runs `whitening.measure_and_whiten(system, model, raw_start)`: the same bracket+bisect probe PTDE uses for chain initialization, at wider dynamic range, and each `Parameter.set_whitening()` multiplies its shared logit-space scale by the measured 0.5-nat step **in place** — no rebuild, no recompile, and the posterior is provably unchanged (the logit-uniform correction potential cancels the raw N(0,1) prior symbolically for any scale). After the rescale a unit step along any raw direction costs ~0.5 nats — the "curvature = -1" conditioning the retired curvature check asked users to approximate by editing `init_scale`. Sampler-block key `measure_scales: false` skips all of it (keeps preliminary scales). Design points:
- `set_whitening` deliberately does **not** recompute `logit_q_inits`/`q_floors`: raw = 0 must keep mapping to the exact start the probe measured around; the update is a pure scale change in logit space (and `_raw_transform` is synced for multi-seed starts). It returns the post-rescale raw-unit scales, which run.py hands to PTDE (`raw_scales=`) so `_make_starts` skips its own probe.
- Elements whose raw N(0,1) **is** the prior (unbounded with sigma) are never rescaled; a flat probe direction (NaN multiplier) keeps its preliminary scale and is flagged after the startup table. A multiplier clipped at the probe's dynamic range (preliminary scale off by >~9 orders of magnitude) is **escalated**: the clipped correction is applied and just those elements are re-probed in the new coordinates; still-clipped elements get a warning naming the defaults.yaml scale to fix.
- **Soft-bound barrier steepness is measured too** (`whitening.measure_barrier_scales`): with the model whitened, each barrier-carrying element's natural scale is its quadrature response to unit raw steps — n_sampled+1 forward evals of the parameter transform graphs, replacing the old sympy forward-Jacobian scale pass (deleted). Barrier scales live in a `pytensor.shared` per parameter; a user `bound_scale:` (params.yaml or defaults.yaml, physical units, barrier transition width = 0.01 x scale) pins an element against the measured update — this is the *only* user-facing scale knob left, and unlike the old init_scale it is honestly a posterior choice, not conditioning. If updating the barriers moves the start's logp (a bound is active at the start), `measure_and_whiten` re-measures once against the final barriers — otherwise the probe/barrier feedback is exactly zero.
- The **absolute** whitening + barrier state is persisted to `<prefix>_whitening.json` next to the trace. A reload (`recompute_trace: false` with an existing trace) restores it instead of re-probing — raw draws only decode correctly under the whitening they were sampled with; any model mismatch is detected up front and falls back to a fresh measurement. Physical Deterministics in the idata are unaffected either way.
- User-side `init_scale` (params.yaml) is stripped with a warning; `mkparam`/`mmexofast_to_params` no longer write it; it is not linkable.

### User-defined parameter links (`linking.py`)

Any of the six numeric fields in a `params.yaml` entry may be a string expression referencing other parameters (`star.A.age: {initval: star.B.age, sigma: 0}`, `orbit.b.omega: {initval: "orbit.c.omega + 180", sigma: 0}`, `star.A.av: {lower: star.B.av}`). Semantics:
- `initval` link + `sigma: 0` → hard link: the element is never sampled and deterministically tracks the expression.
- `initval` link + `sigma > 0` (or a `mu` link) → soft link: sampled normally plus a Gaussian `pm.Potential` on the difference.
- `initval` link, no sigma → initialization seeding only (relaxation-engine snapshot, no runtime tie).
- `lower`/`upper` link → dynamic hard bound: the logit transform maps into the tensor-valued interval; a `-log(span)` potential keeps the conditional prior normalized.
- `sigma` link → static numeric snapshot from the relaxation-engine solution. (`init_scale` is not linkable — it is stripped from user params.)

Referenced parameters contribute their values in **their own user units**; the result is read in the **target's user unit**. `ConfigManager` extracts links at construction (`extract_links` strips the strings from `user_params`), the relaxation engine asserts `initval`/`mu` links as directed RANK_USER assignments each iteration, `Component._wire_user_links` builds the PyTensor closures (same-parameter element references are resolved inside `build_pymc` via `set_subtensor`; cross-parameter references use the lazy `add_parameter` recursion), and `graph.py` adds cross-parameter build-order edges. Tests: `tests/test_linked_params.py`.

### Physics registry

`@register_physics` (in `physics_registry.py`) populates `PHYSICS_REGISTRY` at import time. The `add_parameter` method in `Component` looks up `func_name` from `defaults.yaml` in this registry to wire up PyTensor expression lambdas. Any new physics function must use this decorator.

The registry is a **flat namespace keyed by bare function name** -- there is no component scoping, so two components registering the same name would shadow each other, last import wins. `register_physics` now raises on a duplicate rather than allowing that. If two components need the same physics, give it one owner and import it (see `components/planet/physics.py`'s `calc_density`). Name functions after what they *take* when the same quantity has several forms: `calc_logg_from_logmass` (star) vs `calc_logg_from_mass` (planet). These two used to collide as `calc_logg`, and planet's won -- so `star.logg` was silently computed as `LOGG_CONST + log10(logmass) - 2*log10(radius)`: wrong for every star, NaN below 1 solMass. It went unnoticed because nothing consumed `star.logg` (the SED builds its own `loggsed` via a direct import, which is why fits were unaffected) until `components/torres` needed it. Cover: `tests/test_physics_registry.py`. Note a direct `from ..star.physics import x` binds the function object and bypasses the registry entirely -- only `func_name:` lookups go through it.

### Graph and build order

`graph.py:determine_pymc_build_order()` reads every manifest entry's `expressions.deps` list and performs a topological sort. Dependencies referencing other components use the `"comp.param[map_name]"` syntax (e.g., `"star.mass[lens_map]"`); the brackets name the integer map attribute on the requesting component that provides the index slice.

### Adding a new component

1. Create `src/exozippy/components/<name>/` with the four standard files.
2. Set `comp_key` in `symbolic_physics.py` and `prefix` property in the class to match the YAML key.
3. Declare `self.manifest` in `register_parameters()`. Manifest values: `None` (free parameter, no expression), `"default"` (use `expressions.default` from `defaults.yaml`), or a dict with `"expr_key"` and optional overrides.
4. Every sampled (non-derived, non-fixed) parameter **must** have `lower` and `upper` in `defaults.yaml`. `init_scale` is recommended but optional (it seeds the whitening probe; missing values fall back to a fraction of the span).
5. Add the YAML key to example configs to test.

### Plotting: one description, two renderers (`plot_data` + `plotrender`)

Each plot is described ONCE, as the `PlotSpec` list returned by `Component.plot_data(system, point=None)` (see `src/exozippy/plotspec.py`). Two renderers consume the same specs: `src/exozippy/plotrender.py` draws them with matplotlib for the saved PDFs (`Component.plot()` is a one-liner calling `plotrender.plot_via_specs`, which handles the points-list spaghetti: data/decorations from `points[0]`, later draws overlay their model traces at alpha 0.1), and the GUI draws them with plotly (`gui/frontend/src/plotly-adapter.ts`). The two renderers share a meta/style vocabulary (documented in plotrender.py's module docstring: `file_tag` = the PDF filename tag, `figsize`, `hline_y`, `x/y_range`, `x/y_log`, `x/y_inverted`, `aspect_equal`; trace style: `series_index` fixed categorical color, `color`/`marker`/`lw`/`legend`) -- extend BOTH when adding a key. Never hand-draw a data plot in a component again; genuinely bespoke diagnostics (astrometry's `plot_sky`, lens caustics, corner plots) may stay matplotlib-only.

`plot_data` rules: with `point=None` return data-only specs (usable after `load_data()`, before `build_model()`); with a point, add model traces evaluated at that point by reusing the functions from `compile_plotters()` -- do not duplicate physics. Set each spec's `param_deps` via `_model_trace_param_deps(node, system)` on a symbolic node retained by `compile_plotters` -- empty `param_deps` makes the Evaluator's `changed_label` filter skip the component, freezing its charts in GUI live mode. Keep model traces' symbolic nodes on `Trace.node`. `meta.file_tag` must reproduce the component's historical PDF filenames (`{prefix}_{file_tag}.pdf`).

### The GUI (`src/exozippy/gui/`)

The optional browser GUI (`exozippy-gui` console script, `gui` extra) is a component-agnostic FastAPI + React wrapper around the backend contracts (`introspect`, `utilities/registry`, `solve_api`, `plotspec`, `evaluator`, `gui/runner`). Its full architecture -- server modules, the HTTP/WebSocket API, the frontend tabs, the Solve-then-live-sliders interaction, and the invariants (component-agnostic, ruamel round-trip, process isolation, local-only) -- is documented in `src/exozippy/gui/gui.md`. Read that before adding a tab, endpoint, or utility to the GUI, or before changing how the GUI consumes a component.

### Parameter naming convention

User-facing paths always use three dot-separated parts: `<component>.<instance_name>.<param>` (e.g., `star.Lens.distance`). Internally, instance names are standardized to indices (`star.0.distance`). `ConfigManager.resolve()` checks all three forms (`comp.param`, `comp.0.param`, `comp.Name.param`).

### Hierarchical orbits and body groups

Every orbit is a two-body Keplerian arc between a `primary:` and a `companion:` **body group** (lists of star/planet instance names or `star.X`/`planet.X` paths; parsing in `components/orbit/bodies.py`). Multi-body groups are point masses at their barycenter, which is how hierarchical systems are built (KELT-4: `b` orbits `A`; `B` orbits `C`; `[B, C]` orbits `[A, b]` -- see `examples/kelt4`). Omitting both keys reproduces the legacy implicit topology (companion = planets whose `orbit_ndx` points at the orbit, primary = their host stars). Each orbit derives `m_primary`/`m_companion`/`m_total`/`arsun`/`K` from the member bodies' `star.mass`/`planet.mass` nodes (weighted-sum context nodes injected by `Orbit.add_parameter`), so orbits sharing a body are automatically mass-consistent. The relaxation engine gets per-orbit `m_total` initvals from a custom solver (sum of member masses) feeding the instance-scoped Kepler relation; relation symbols must stay mapped in `get_symbol_map` (`a` -> `arsun`, `m_total`) or they instantiate as symbols shared across orbits.

Consumers are membership-based: `rvinstrument` (per-instrument `star_ndx`, default 0) sums `orbit.K` over every orbit containing the observed star (`Orbit.star_membership`; companion-side membership flips sign and rescales by `m_primary/m_companion`); `astrometryinstrument` rel mode references an orbit by name (`orbit:` key; legacy `planet_ndx` still resolves) and models the companion group relative to the primary group, adding the photocenter wobble of any orbit nested inside a group (SED-weighted when `band:` is given, dark-companion beta=0 otherwise, barycenter fallback with a warning); gaia/abs photocenter wobble sums the orbits whose primary group contains the target star. Stars only get sampled `ra/dec/pm_ra/pm_dec` when a gaia/abs dataset exists -- rel data are differential and need only the parallax scale.

### Bands, filters, and SED flux constraints

The Band component is the single carrier of filter identity and limb darkening: instruments (`transit`, `mulensinstrument`, optionally `astrometryinstrument`) reference a `band:` block by name; each band's `filter:` string is resolved through the SED alias table (`components/sed/filters/filternames.txt`, columns Keivan/MIST/Claret/SVO/VOID) into canonical names at load time. Transit LD (q1/q2/u1/u2) lives on Band, not on the transit component.

The SED component supports n stars: each `.sed` filter row's `photType: {pos: [...], neg: [...]}` (with `blend:` as an alias for `pos`; entries are star names or indices) builds a +1/0/-1 blend matrix; blended rows are flux sums, `neg` makes the row a differential magnitude (`-2.5*log10(F_pos/F_neg)`). An empty `filters:` list is legal — the SED then only serves cross-component flux predictions (`predict_star_appmag`, `predict_blend_appmag`, `predict_flux_fraction`). BC tables for missing filters auto-generate from the model spectra (`components/sed/make_bc.py`, CLI `scripts/make_bc_tables.py`).

Cross-component hooks when a `sed:` block exists: `mulensinstrument` ties each light curve's `f_source` to the SED-predicted source mag through a per-lightcurve `zeropoint` (Deterministic + Gaussian potential, default 0 +/- 0.2 mag; `sed_constrain_blend: true` opts f_blend in); `transit` dilutes depths by the host's SED flux fraction in the band; `astrometryinstrument` derives its photocenter `fluxfrac` from the SED when given `band:` + `companion_star_ndx:`.

### Empirical stellar relations

Relations that constrain `star.mass`/`star.radius` from other stellar properties are **one component per relation**, one instance per constrained star, each naming its target with a `star:` key (`mann: [{star: "B", constrain: [mass, radius]}]`; see `examples/kelt4`). They add Gaussian potentials on the star's existing nodes -- they never make mass/radius derived, so other data (transit, RV) still constrain them jointly. `star.py` adds `distance` to its manifest when such a component is in the topology.

`mann` (`components/mann/`) implements Mann+2015 (radius) and Mann+2019 (mass) from absolute Ks, ported from EXOFASTv2's `massradius_mann.pro`. Two Ks pathways, matching EXOFASTv2's mannrad/mannmass vs mannsynrad/mannsynmass: `ks: synthetic` takes the star's individual Ks from `sed.predict_star_appmag(star_idx, "2MASS/2MASS.Ks")` (so it works when the observed Ks is a blend of several modeled stars), and `ks: <number>` + `ks_err:` takes a direct observation. Either way the relation's input is **non-centered** -- `mann.appks = ks_source + ks_err * ks_offset` with `ks_offset ~ N(0,1)` -- which is the same posterior as EXOFASTv2's free `appks` plus a `((appks - synk)/0.02)^2` penalty, without the rotated stiff pair that a diagonal mass matrix cannot whiten. sigma on each penalty is the relation's published **fractional** scatter times its own *prediction*, so unlike the fixed-sigma priors elsewhere the `-log(sigma)` normalization is kept (EXOFASTv2 accumulates chi2 only and drops it). Applies to 0.075-0.7 solMass.

`torres` (`components/torres/`) implements Torres+2010 from `star.teff`/`star.logg`/`star.feh`, ported from `massradius_torres.pro`, and applies above ~0.6 solMass -- the complement of `mann`. It needs no new parameters at all (empty manifest, potentials only). Its relations predict **log10(M)** and **log10(R)** and its scatter is in **dex** (0.027/0.014, overridable per instance as `logm_floor`/`logr_floor` -- deliberately named differently from mann's *fractional* `mstar_floor`/`rstar_floor`), so the mass penalty acts directly on `star.logmass` with no exponentiation round trip, and the constant `-log(sigma)` is dropped exactly as EXOFASTv2 does. These two structural differences (log vs linear, dex vs fractional, latent vs no latent) are why the relations are separate components rather than sharing a base class.

For both, calibration-range violations are **startup warnings only** -- EXOFASTv2 re-checks every likelihood call and hard-rejects on out-of-range [Fe/H], but a `-inf` wall has no gradient for NUTS to follow, so nothing here bounds the posterior.

Testing note: build relation inputs with `pt.dscalar`, **not** `pt.as_tensor_variable(<python float>)` -- pytensor autocasts a bare Python float to the smallest dtype that represents it (5778.0 -> float32), and a unary op like `pt.log10` on it then computes in float32, silently losing ~1e-7. The model always feeds float64. `tests/test_torres.py` pins the port against real IDL output from `massradius_torres.pro`.

## Tests

Tests follow AAA (Arrange / Act / Assert) with Given/When/Then docstrings. All tests that use `System` must call `system.prepare()` before `system.build_model()`. RA/Dec user params are in **degrees** (the default unit); `Parameter.__post_init__` converts to radians internally.

The test suite takes ~10 minutes. Do not start it with a timeout. Start it and poll.