# Components: structure, manifest, build order

What a component is, how the factory finds it, the manifest vocabulary and its single
interpreter, the physics registry, the topological build order, and how to declare a
per-instance parameterization.

Read this before adding a component, before touching the manifest vocabulary
(`src/exozippy/manifest.py`), the build graph (`src/exozippy/graph.py`) or the physics
registry (`src/exozippy/physics_registry.py`). Related: the four element roles a manifest
entry can declare are in `src/exozippy/components/parameter.md`; the shared data-component
scaffolding is `src/exozippy/components/instrument.md`.

## The declared extension API

**Writing a new component is a supported extension point, and the contract is this document plus `parameter.md`.** The operative case is not a contributor who opens a pull request; it is someone who forks EXOZIPPy, writes a component for their own instrument or their own physics, and never pushes it back. That author never negotiates the contract with anyone, so it has to be written down rather than agreed. The declared surface is:

- **`Component`** (`components/component.py`) -- the seven lifecycle methods (the table in `CLAUDE.md`), `manifest`, `label`, and the `add_hint` / `add_override` channels documented in `src/exozippy/config.md`.
- **`Parameter`** (`components/parameter.py`) -- its fields, units contract, and the four per-element roles. Read `parameter.md` before relying on any of it.
- **The manifest vocabulary** -- `manifest.py`'s `interpret_manifest_entry` and the entry shapes it accepts, described above.
- **The four-file component layout** and the auto-discovering factory, described immediately below: a component in a fork's own directory is found the same way a shipped one is.

These are NOT in the package root's `__all__`, and that is deliberate rather than an oversight: `exozippy.__all__` is the run-a-fit surface (`System`, `run_fit`, `__version__`), and writing a component is a different job from running a fit. Import them from their modules.

**Everything in `exozippy` outside those two surfaces is internal** -- importable, since none of this is enforcement, but free to change signature without a major version bump. That includes the GUI's backend modules (`introspect`, `solve_api`, `plotspec`, `evaluator`, `utilities/registry`, `gui/runner`): the GUI ships in this distribution, is versioned with it, and has no third-party consumer, so those stay refactorable as long as both sides move together (`gui/gui.md`). Review 8.13.6; tests: `tests/test_public_api.py`.

## Component structure

Each component lives in `src/exozippy/components/<name>/` and contains:
- `<name>.py` — class inheriting `Component`; implements the 7-stage methods
- `defaults.yaml` — default values, bounds, units, and expression wiring for every parameter
- `symbolic_physics.py` — SymPy `RELATIONS` (equations) and `get_symbol_map()` (maps abstract symbols → indexed YAML paths); must set `comp_key = "<yaml_key>"` to match the YAML block name
- `physics.py` — PyTensor/numpy implementations decorated with `@register_physics`; function name must match `func_name` in `defaults.yaml`

The **factory** (`factory.py`) auto-discovers all `Component` subclasses by scanning subdirectories; the YAML key used to instantiate a component is the lowercase class name (or a `yaml_key` class attribute, which nothing in the tree sets -- three components restated the fallback as an explicit `yaml_key` until 2026-08, implying they differed from their class names when they did not; review 4.2.4). No registration step is required for new components. Abstract intermediate bases are skipped (`inspect.isabstract`), so a shared base can leave `Component`'s abstract methods unimplemented and never be instantiated. A module that fails to import is recorded rather than fatal -- **every** exception, not only `ImportError`, since a SyntaxError in one unused component used to abort discovery for every fit and every GUI open (review 2.2.3) -- and `System.__init__` consults `import_failures()` so a config that names the broken component still fails loudly.

Every component carries a **`label`**, declared on `Component` and defaulted to the class name: it is the heading of that component's block in the results table (`outputs/latex.py`'s `\sidehead`). Ten components assigned one while the base class declared nothing, so the single consumer had to `getattr`-guard -- the tell that it was not part of the contract (review 4.2.3).

## Physics registry

`@register_physics` (in `physics_registry.py`) populates `PHYSICS_REGISTRY` at import time. The `add_parameter` method in `Component` looks up `func_name` from `defaults.yaml` in this registry to wire up PyTensor expression lambdas. Any new physics function must use this decorator.

The registry is a **flat namespace keyed by bare function name** -- there is no component scoping, so two components registering the same name would shadow each other, last import wins. `register_physics` now raises on a duplicate rather than allowing that. If two components need the same physics, give it one owner and import it (see `components/planet/physics.py`'s `calc_density`). Name functions after what they *take* when the same quantity has several forms: `calc_logg_from_logmass` (star) vs `calc_logg_from_mass` (planet). These two used to collide as `calc_logg`, and planet's won -- so `star.logg` was silently computed as `LOGG_CONST + log10(logmass) - 2*log10(radius)`: wrong for every star, NaN below 1 solMass. It went unnoticed because nothing consumed `star.logg` (the SED built its own `loggsed` inline via a direct import, which is why fits were unaffected -- `loggsed` is a derived Parameter now, and reaches the same function through `func_name:`, i.e. through the registry) until `components/torres` needed it. Cover: `tests/test_physics_registry.py`. Note a direct `from ..star.physics import x` binds the function object and bypasses the registry entirely -- only `func_name:` lookups go through it.

## Graph and build order

`graph.py:determine_pymc_build_order()` reads every manifest entry's `expressions.deps` list and performs a topological sort. Dependencies referencing other components use the `"comp.param[map_name]"` syntax (e.g., `"star.mass[lens_map]"`); the brackets name the integer map attribute on the requesting component that provides the index slice.

**The manifest vocabulary has exactly one interpreter, `manifest.py` (`interpret_manifest_entry` -> `ManifestEntry`), and all three of its consumers go through it**: `graph.determine_pymc_build_order` (the build order), `Component.add_parameter` (stage 6) and `System.derived_params`. It answers three questions -- does this entry name an expression (`names_expression`), which `expressions:` block does it select given a resolved config (`expression_config`), and what deps does that expression take (`dep_names`, where a manifest `deps` list beats the block's) -- and hands back everything else as `options`. It imports nothing from the package, so any consumer can use it without a cycle, and it holds no component-specific knowledge.

**An `expr_key` the resolved config does not define RAISES** (`MissingExpressionError`, naming the component, the parameter, the missing key, the keys that are available, and the two legal spellings of a free parameter). It used to answer "free", in all three readers at once -- consistent, and silently wrong: a typo, a renamed block or a deleted `expressions:` section demoted a derived parameter to a sampled one with no message anywhere. Breaking `mulensinstrument.f_source`'s expr_key on `examples/ob08092` that way put an `f_source_raw` in `model.free_RVs` and moved the start logp from +6187.7 to -6.46e9, and the fit still ran and still reported. There is nothing legitimate to lose, because a free parameter has two explicit spellings (`None`, or an options-only dict) and neither is affected. Instrumenting the resolution across every shipped config found exactly one entry relying on the fallback -- `rvinstrument`'s `"gamma": "default"`, whose `expressions:` block was deleted in June 2026 -- and it is now `None`.

That raise is also what keeps the structural `names_expression` and the config-aware `expression_config` from disagreeing: the one state in which they could differ now raises instead of answering. `System.derived_params` asks through `expression_config` (resolving each entry, exactly as graph.py does) rather than structurally, because only the *build* path would ever reach that raise and `derived_params`' callers -- `solve_api`, the GUI's Tune tab -- never build a model. A structural answer there is precisely the silence this raise removes: `solve_api._bounds_diagnostics` skips parameters reported derived, so mislabelled `gamma` meant no RV offset was ever bounds-checked in the GUI, out-of-bounds start and all.

A manifest entry is derived **only** when it is a string or a dict carrying `expr_key` -- a dict holding just options (an `"overrides"` pin, a `shape`, a `table_note`) is a free parameter. That rule was `add_parameter`'s all along; the other two readers were hand-written copies of it and drifted. graph.py fell back to the `default` expression for *any* dict entry, which was inert only while no pinned free parameter had an **unused** `expressions:` block in its defaults.yaml, and became a hard "Dependency Error" the moment one did (Band's linear-law `u1`, whose Kipping expression the manifest deliberately ignores; planet's `beam` "off" entry, which made every orbit-less config demand an RV semi-amplitude). The fallback could only ever add edges `add_parameter` does not use, never supply a needed one -- the parameters it applied to are free by definition -- so adopting the shared reader strictly removes spurious edges. Verified: byte-identical build order and `derived_params()` on all 19 shipped example configs that build. Do not re-derive the rule at a call site; add the question to `ManifestEntry`. Tests: `tests/test_manifest_interpreter.py` (contract, the graph-level regression, and the real Band `u1` reproduction end to end).

## Building a model twice on one System

`System.build_model()` may be called more than once on a prepared System, and the second model must be a complete, independent graph. Two mechanisms make that true, and both had to be fixed for either to matter (reviews 1.5.2, 3.14.12):

- **The already-built guard is per BUILD, not per System.** `Component.add_parameter`'s "do not build this twice" check exists so a recursive dependency resolves once; a component persists on the System, so asking only "is there a `Parameter` here" handed the second build the *first* model's nodes and the second model's logp compile raised "Random variables detected in the logp graph". `Component._parameter_is_current(comp, name, model)` is the predicate every such site now asks: `add_parameter`'s own guard, the local and cross-component dependency resolvers, the user-link resolver, and the three component-side copies (`Orbit._chord_context`'s planet `p`/`ar`, `Orbit.add_parameter`'s group masses, `SED._ensure_star_nodes`). It stamps a weakref to the model in `Component._built_for_model` at `setattr` time; **absent provenance counts as current**, so a Parameter set by hand or by a test double behaves exactly as it always did. `_has_built_parameter` survives as the narrower type-only question it documents.
- **A cached node belongs to its build.** Declare its attribute name in the class-level `per_build_caches` tuple and `System.build_model` clears it before stage 5. Use this rather than another ad-hoc reset: `Transit._dilution_node` and `Instrument`'s outlier evaluators are cleared at the top of stage 7, which is early enough only because nothing reads them before then -- `SED._m_pred_matrix` is built at stage 6 (the mulensing zeropoint expression asks for it while parameters are still being materialized) and a stage-7 reset would come too late.

Tests: `tests/test_rebuild_caches.py`, whose graph walk asserts that no node of the second model descends from a random variable of the first -- the property, rather than the logp-compile symptom, so a leak that lands outside the logp graph is still caught.

## Declaring a parameterization

The per-element roles these tables expand into are documented in
`src/exozippy/components/parameter.md`.

**Declaring a parameterization is a TABLE, not a hand-built mask** (`components/parameterization.py`). A component holds a per-instance choice read from its own config and a statement of what each choice uses; `mode_manifest(modes, table, options=...)` turns that into manifest entries, so the four consumers share one expansion instead of four hand-rolled ones. A parameter a mode does not name is inactive on that mode's elements -- that is how `linear` says it has no `q2`. Two properties make it safe to adopt: a single-mode system expands to **exactly** the manifest the component used to hand-write (bare-string `expr_key`, no `mask`, and an inert `inactive_value` dropped), and a parameter **no** instance uses is omitted entirely rather than declared wholly inactive (an all-linear band set has no `u2`, which is what its consumers' `"u2" in band.manifest` guard reads).

The sibling helper `pin_unselected(n_elements, selected)` is the **opt-in** pin, and the difference from an inactive element is the reason both exist: it pins through `"overrides"`, which layers *under* the params file, for a parameter that exists for every instance but is only wanted on some (a GP hyperparameter on the files that asked for one, an LD coefficient on the bands something reads, the BEER terms on the bands that fit them) -- so a user who explicitly wants one back still wins. An inactive element's pin is structural and unreported, because freeing it would add a dimension no likelihood term reads. `Instrument._register_gp`, `Instrument._register_robust` and `Band` (BEER terms + the unread-LD autopin) had that loop written out line for line; it is one function now (review 4.5.2, 4.5.3).

## Config flag vocabulary

Boolean config flags come in exactly three kinds, and the kinds do not share a
spelling.  Before adding a flag, pick its kind and follow that template.

- **Include-this-signal toggles, `fit<x>`** (`fitbeam`, `fitellip`, `fitthermal`,
  `fitreflect`): add a signal and its free amplitude to the model.  `finite_source`
  is this kind in spirit (physics inclusion) and is deliberately NOT spelled
  `fit_finite_source`: the finite-source effect is a direct measurement of rho and
  is always on when the physics is.
- **Coordinate choices, `fit<coord>`** (`fitvcve`, `fitchord`): sample in this
  parameterization rather than the default.  Nothing becomes more or less
  constrained; only the coordinates change.
- **Tie toggles, `X_constrains_Y`** (`beam_constrains_mass`, `sed_constrains_blend`,
  `star_constrains_rho`): supply or sever a physics link between two quantities.
  The name states the INTENT -- the reason a user turns the tie on is that they
  want X's information to sharpen Y.  The mechanism is symmetric: a tie is a link
  in a joint density, not a one-way assignment, and information flows toward
  whichever side is less constrained by everything else.  That reversal is not
  hypothetical: on DC2018 event 128 the rho = theta_star/theta_E identity (the tie
  `star_constrains_rho` now controls) ran BACKWARD -- the priors on theta_E's side
  overrode the light curve's 2.5%-precise rho -- which is why each tie flag's
  severed mode reports the untied prediction (`rho_pred`) so the pull is visible.
  A `Z_links_X_and_Y` spelling would name the mechanism more honestly, but buries
  the searchable quantity mid-name and reads worse at the call site; the ruling is
  intent-named flags, mechanism documented here and in each flag's schema doc.

## Adding a new component

1. Create `src/exozippy/components/<name>/` with the four standard files.
2. Set `comp_key` in `symbolic_physics.py` and `prefix` property in the class to match the YAML key.
3. Declare `self.manifest` in `register_parameters()`. Manifest values: `None` (free parameter, no expression), `"default"` (use `expressions.default` from `defaults.yaml`), or a dict with `"expr_key"` and optional overrides. A dict with no `"expr_key"` is a free parameter carrying options; an `"expr_key"` naming a block the defaults.yaml does not define raises.
4. Every sampled (non-derived, non-fixed) parameter **must** have `lower` and `upper` in `defaults.yaml`. `init_scale` is recommended but optional (it seeds the whitening probe; missing values fall back to a fraction of the span).
5. Add the YAML key to example configs to test.

## Referring to a star

Every user-facing star reference -- an instrument's or a band's `star_ndx:`, an SED
`photType` entry, a relation component's `star:` key -- goes through the one translator
`components.component.resolve_star_ref(ref, star_names, where)`. It accepts an index
(`1`), a name (`"B"`) and either as a path (`"star.B"`, `"star.1"`), rejects a bool
(`star_ndx: true` is a typo, not star 1), range-checks the result, and puts the
caller-supplied `where` plus the known star names in every message.

There were three copies of this and, worse, two schemas that advertised the behaviour
without having it: `rvinstrument` and `band` both documented `star_ndx` as "Index or name"
while every consumer called `int()` on it, so a name crashed with a raw
`invalid literal for int()` (review 3.5.1). `Component.resolve_star_ndx(ref, where)` is the
component-side wrapper (`None` -> the default index), and it reads the star names off
`config_manager.system_config` rather than `system.star`, so a NAME resolves at
construction and at stage 1 -- before the Star component exists. With no system config
(a test stub) only indices resolve, which is the historical behaviour rather than a
spurious failure.
