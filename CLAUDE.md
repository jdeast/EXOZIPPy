# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

It is deliberately a **trunk**: the commands, the architecture, the invariants that must survive even if you never follow a pointer, and a map of the deeper docs. Each subsystem's rulings live in a doc next to its code -- read the mapped doc BEFORE editing that subsystem. Those docs are not optional background: nearly every paragraph in them is a "this looks wrong, here is why it is deliberate, do not revert it", and losing one causes a real regression.

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

The component contract -- the four standard files per component, the auto-discovering factory, the manifest vocabulary and the build order -- is in `src/exozippy/components/components.md`.

## Invariants

These hold everywhere and are repeated here because a reader who never follows a pointer still has to know them.

- `system.prepare()` must be called before `system.build_model()` (above), and the plain-ASCII rule (above) governs everything anyone writes.
- **Never write into `config_manager.user_params` from a component.** `setdefault` there looks like it preserves user precedence, and does -- but the entry then *is* a user entry as far as everything downstream can tell: the provenance ledger, `export_solution`, `initval_source`, `probe_derivable`'s flat, `resolve`'s `user_modified`/`user_prior_modified` flags (the `*` in run.py's startup table) and the GUI all report a value the user never wrote. It also has a second effect nobody intends: `finalize_user_params` registers every unmapped `user_params` key as a **leaf symbol** in the relaxation engine, so an injected `star.av` became an engine symbol with its own ledger row, its own default-armor value and its own inject-back initval, for a parameter instance that does not exist -- the "orphaned 2-part rows" `export_solution` filters out by hand. Both remaining injections (the SED grid bounds; `AstrometryInstrument`'s `sigma: 0` pin on a `fluxfrac` the SED supplies) were removed in 2026-08 with **bit-identical start logp on all ten shipped examples** that exercise either component. Neither was ever a candidate for `add_hint`: a hint is a *ranked start value*, one scalar feeding `initval`, whereas `lower`/`upper`/`sigma` never enter the provenance ledger at all -- which is also why the fix carries no rank. Tests: `tests/test_component_override_channel.py`.
- **There are TWO conversion factors in the codebase and they are RECIPROCALS.** `Parameter._get_conversion_factors` (parameter.py) is the **internal -> user** multiplier; `ConfigManager.get_conversion_factor` (config.py, via the `unit_conversion` helper) is the **user -> internal** one. So `* factor` means opposite things in the two modules, and confusing them is silent -- `outputs/ledger.py` divided where it had to multiply and reported **every** converted parameter in `<prefix>_results.csv`, the rejected-seeds text and the rejected-modes LaTeX table wrong by `factor**2` (`examples/hd80606`'s planet mass: 1.45e-06 jupiterMass for a start of 1.596; `examples/GaiaBH1`'s `star.ra`: 0.0799 deg for 262.171). Nothing outside `Parameter` may reach for a factor: call `to_internal(val, index=None)` / `from_internal(val, index=None)` and the direction is in the name.
- **Never compose branch log-densities with `pt.where`** (the JAX where-trap): `where`'s VJP multiplies the unselected branch by zero, and `0 * NaN` or `0 * inf` poisons the gradient of the whole expression on every backend, while the C backend's finite dlogp hides it. Combine branch logps with `pt.logaddexp`, keep a floor on a radicand rather than on the result, and use the soft-bound helpers in `src/exozippy/potentials.py` for a penalty that needs a gradient pointing back. Likewise assemble a mixed value vector with `pt.set_subtensor`, not `pt.where`.

## Subsystem map

Each line names when to read the doc. Read it first; do not reconstruct its reasoning from the code.

- Before adding a component, or touching the manifest vocabulary (`manifest.py`), the build graph (`graph.py`) or the physics registry (`physics_registry.py`), read `src/exozippy/components/components.md`.
- Before changing how a config value, hint, override, bound or link is resolved -- or the three user-facing spellings of an element name -- read `src/exozippy/config.md`.
- Before changing how a `Parameter` becomes a PyMC node (units, start values, pins, per-element roles, how a component-added prior is reported), read `src/exozippy/components/parameter.md`.
- Before touching `init_scale`, `bound_scale`, the startup probe or the persisted whitening state, read `src/exozippy/whitening.md`.
- Before projecting anything onto the sky, or "fixing" a sign in orbit, astrometry or microlensing geometry, read `src/exozippy/skyframe.md`.
- Before changing the run flow, the restart-file writer (`mkparam.py`) or trace reuse (`trace_meta.py`), read `src/exozippy/run.md`.
- Before touching how data files are read (columns, masks, time systems), detrending, the noise model, Gaussian-process noise or robust likelihoods, read `src/exozippy/components/instrument.md`.
- Before touching the microlensing likelihood, MMEXOFAST seeding, or the lens/source body maps, read `src/exozippy/components/mulensing/mulensing.md`.
- Before changing a stellar prior (IMF, the FFP mass function, distance), an empirical stellar or planetary relation, or the planet mass coordinate, read `src/exozippy/components/star/star.md`.
- Before changing orbit topology, the `tc` window, or the eccentricity/inclination coordinates (`fitvcve`, `fitchord`), read `src/exozippy/components/orbit/orbit.md`.
- Before touching limb darkening, filter identity, the bolometric-correction grid or an SED flux hook, read `src/exozippy/components/sed/sed.md`.
- Before adding a sampler, or changing chain sizing or start populations, read `src/exozippy/samplers/samplers.md`.
- Before adding a table column, a LaTeX macro, a prose sentence or a plot, read `src/exozippy/outputs/outputs.md`.
- Before writing or running tests, read `docs/testing.md`.
- The GUI (`src/exozippy/gui/`): the optional browser GUI (`exozippy-gui` console script, `gui` extra) is a component-agnostic FastAPI + React wrapper around the backend contracts (`introspect`, `utilities/registry`, `solve_api`, `plotspec`, `evaluator`, `gui/runner`). Its full architecture -- server modules, the HTTP/WebSocket API, the frontend tabs, the Solve-then-live-sliders interaction, and the invariants (component-agnostic, ruamel round-trip, process isolation, local-only) -- is documented in `src/exozippy/gui/gui.md`. Read that before adding a tab, endpoint, or utility to the GUI, or before changing how the GUI consumes a component.

## Tests

Conventions and the suite's runbook are in `docs/testing.md`.
