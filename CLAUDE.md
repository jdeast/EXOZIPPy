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
5. Propagates `init_scale` via symbolic Jacobians of the solution expressions.

Components push hints via `config_manager.add_hint(path, value, rank)` during stage 1–2. The hint system is the correct way for components to propose data-driven initial guesses; they are layered in after defaults but before the solver runs.

### Component structure

Each component lives in `src/exozippy/components/<name>/` and contains:
- `<name>.py` — class inheriting `Component`; implements the 6-stage methods
- `defaults.yaml` — default values, bounds, units, and expression wiring for every parameter
- `symbolic_physics.py` — SymPy `RELATIONS` (equations) and `get_symbol_map()` (maps abstract symbols → indexed YAML paths); must set `comp_key = "<yaml_key>"` to match the YAML block name
- `physics.py` — PyTensor/numpy implementations decorated with `@register_physics`; function name must match `func_name` in `defaults.yaml`

The **factory** (`factory.py`) auto-discovers all `Component` subclasses by scanning subdirectories; the YAML key used to instantiate a component is the lowercase class name (or `yaml_key` class attribute if set). No registration step is required for new components.

### Parameter system (`parameter.py`)

`Parameter` is the universal node wrapper. Key points:
- All numeric fields (`initval`, `init_scale`, `lower`, `upper`, `mu`, `sigma`) are stored in **internal units** after `__post_init__` applies the unit conversion factor.
- `unit` is the user-facing unit (from `defaults.yaml` or user override); `internal_unit` is the math unit.
- `sigma = 0` → parameter is fixed. `sigma > 0` → Gaussian potential applied. No sigma + `init_scale` → uniform prior on `[lower, upper]` via logit transform.
- Symbolic PyTensor nodes passed as `initval` are preserved as-is (no unit conversion applied).
- `build_pymc()` uses non-centered parameterization: raw `N(0, 1)` mapped to physical space via logit or linear scale + shift.

### User-defined parameter links (`linking.py`)

Any of the six numeric fields in a `params.yaml` entry may be a string expression referencing other parameters (`star.A.age: {initval: star.B.age, sigma: 0}`, `orbit.b.omega: {initval: "orbit.c.omega + 180", sigma: 0}`, `star.A.av: {lower: star.B.av}`). Semantics:
- `initval` link + `sigma: 0` → hard link: the element is never sampled and deterministically tracks the expression.
- `initval` link + `sigma > 0` (or a `mu` link) → soft link: sampled normally plus a Gaussian `pm.Potential` on the difference.
- `initval` link, no sigma → initialization seeding only (relaxation-engine snapshot, no runtime tie).
- `lower`/`upper` link → dynamic hard bound: the logit transform maps into the tensor-valued interval; a `-log(span)` potential keeps the conditional prior normalized.
- `sigma`/`init_scale` link → static numeric snapshot from the relaxation-engine solution.

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
4. Every sampled (non-derived, non-fixed) parameter **must** have `lower`, `upper`, and `init_scale` in `defaults.yaml`.
5. Add the YAML key to example configs to test.

### Plotting for the GUI (`plot_data`)

`Component.plot()` renders matplotlib figures for the CLI; the browser GUI instead consumes `Component.plot_data(system, point=None) -> list[PlotSpec]` (see `src/exozippy/plotspec.py`), which returns the arrays and labels (not rendered figures) so it can draw interactive charts and re-render model curves when sliders move. Override it in components that own observational data: with `point=None` return data-only specs (usable after `load_data()`, before `build_model()`); with a point, add model traces evaluated at that point by reusing the functions from `compile_plotters()` -- do not duplicate physics. Extract the shared array preparation out of `plot()` so both paths draw identical data, set each spec's `param_deps` (use `_model_trace_param_deps(node, system)`), and keep the model traces' symbolic nodes on the `Trace.node` field for later compiled re-evaluation.

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