# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install / update dependencies
poetry install
poetry update          # after git pull

# Run all tests
poetry run pytest

# Run a single test
poetry run pytest tests/test_config_healing.py::test_name -x

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

### Graph and build order

`graph.py:determine_pymc_build_order()` reads every manifest entry's `expressions.deps` list and performs a topological sort. Dependencies referencing other components use the `"comp.param[map_name]"` syntax (e.g., `"star.mass[lens_map]"`); the brackets name the integer map attribute on the requesting component that provides the index slice.

### Adding a new component

1. Create `src/exozippy/components/<name>/` with the four standard files.
2. Set `comp_key` in `symbolic_physics.py` and `prefix` property in the class to match the YAML key.
3. Declare `self.manifest` in `register_parameters()`. Manifest values: `None` (free parameter, no expression), `"default"` (use `expressions.default` from `defaults.yaml`), or a dict with `"expr_key"` and optional overrides.
4. Every sampled (non-derived, non-fixed) parameter **must** have `lower`, `upper`, and `init_scale` in `defaults.yaml`.
5. Add the YAML key to example configs to test.

### Parameter naming convention

User-facing paths always use three dot-separated parts: `<component>.<instance_name>.<param>` (e.g., `star.Lens.distance`). Internally, instance names are standardized to indices (`star.0.distance`). `ConfigManager.resolve()` checks all three forms (`comp.param`, `comp.0.param`, `comp.Name.param`).

## Tests

Tests follow AAA (Arrange / Act / Assert) with Given/When/Then docstrings. All tests that use `System` must call `system.prepare()` before `system.build_model()`. RA/Dec user params are in **degrees** (the default unit); `Parameter.__post_init__` converts to radians internally.
