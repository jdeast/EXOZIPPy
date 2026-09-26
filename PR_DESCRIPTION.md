## What this adds

An `evolutionarymodel:` component that ties a star's feh/radius/teff/age to the
MIST evolutionary tracks, in the shape `components/torres` already established:
**no parameters of its own** (empty manifest, potentials only), one instance per
constrained star, `star:` + `constrain:`, via the `StellarRelation` mixin.

It samples `star.logmass`, `star.initfeh`, `star.eep`, `star.feh`, `star.radius`,
`star.teff`, `star.age`; interpolates `(logmass, initfeh, eep)` over the MIST grid;
and ties each sampled quantity to its prediction with a Gaussian potential floored
at EXOFASTv2's mass-dependent `percenterror`.

The star side needed **no edit** and has none: `star/defaults.yaml` already carried
`initfeh`/`eep`/`age` and the `in_system("evolutionarymodel")` branch already
declared them. This is the component that branch was waiting for.

Reasoning that is easy to get wrong is in the new
`src/exozippy/components/evolutionarymodel/evolutionarymodel.md` (mapped from
CLAUDE.md's subsystem map), notably: the EEP -> age Jacobian needs a two-sided
**clip**, not a floor (the potential rewards a small derivative, so a bare floor is
+69 nats of attraction toward a table defect); `feh_mist == 30.0` is **data**, not an
error sentinel; and the `here_be_dragons` flag is extended over each track's
unresolved-age tail at load.

Also included: `scripts/bootstrap_intel_mac.sh`, which automates the one manual
celerite2 step `MACOS_INTEL_INSTALL.md` documents, and a shipped example
(`examples/hat3/hat3_mist.yaml`).

## Merging master

The branch was 532 commits behind. Four conflicts, all from master's doc
restructure and its new per-element parameter roles:

- **CLAUDE.md** -- master turned it into a trunk plus a subsystem map, so the
  component's prose moved to a doc next to its code and CLAUDE.md got a map entry.
  The Intel-mac block became a pointer to `MACOS_INTEL_INSTALL.md`.
- **`components/relations.py`** -- kept both sides: master's `normalize_config_block`
  classmethod (its ConfigManager-timing fix) and this branch's `constrainable` class
  attribute, which the auto-merged `_parse_constrain` already calls.
- **`evolutionarymodel/__init__.py`** -- master's "landing pad" placeholder is
  obsolete now the component exists.
- **`tests/conftest.py`** -- both sides appended unrelated helpers; kept both.

Two things the textual merge would otherwise have hidden:

- The manifest `mask` field is **no longer "declared but unconsumed"** -- master's
  per-element roles consume it. `_pin_unmodeled_stars` is still needed and its
  docstring now says why: `mist:` **defaults to True**, so the mask opts every star
  in and is a weaker test than "an evolutionarymodel instance names this star". The
  two mechanisms compose rather than overlap.
- Master renamed `plotspec.PlotSpec` to `chart.Chart` and **promoted the six
  axis-geometry fields out of `meta`** into dataclass fields; `plotrender._apply_meta`
  became `_apply_axes(ax, spec)`. This branch was the last user of all three old
  names, and is ported.

## One change to `star.py`

`Star.structure_consumers` decided whether a star's radius/teff/feh is read by an
evolutionary model from the per-star `mist:`/`parsec:` **switches**. Those default to
opted-in, so they answer "did this star ask for a track" rather than "did a block give
it one" -- every star in a config carrying an `evolutionarymodel:` block counted as
read, including stars no instance names.

Measured on two stars with a block naming only A and nothing else reading them: star
B's radius/teff/feh went **active and sampled**, and B's feh had a logp gradient of
**exactly 0.0** -- three likelihood-free dimensions on a star the model never touches.
Without the block those three are inactive, so adding a block for A was activating B.

The fix is the idiom already used a dozen lines below: ask the component for its
`star_indices`, as `mann` and `torres` are asked. `star_indices` is set in `load_data`
(stage 1), which satisfies that method's "nothing built in stage 3" rule. `in_topology`
returns the instance where a component backs the key and the raw config where nothing
does, so an absent `star_indices` selects the premature-block fallback, which still
marks on the switches rather than deactivating what a component about to land will want.

This is the only edit to `star.py` in the PR; the branch's claim that the star side
needed no edit to *land* the component still holds -- this corrects a predicate master
wrote in anticipation, which shipping the component makes routinely reachable.

Not addressed, deliberately: `constrain:` is still ignored, so `constrain: [teff]`
leaves that star's feh sampled with a flat logp. `mann` is marked for radius+feh and
`torres` for radius+teff+feh regardless of their own `constrain:` lists, so that
approximation is uniform across all three relations and narrowing it is one decision
about all of them.

## Pre-existing test failures, fixed here

Five tests in `test_evolutionary_model.py` already failed on `mist-evol` before this
merge -- verified by running the pre-merge commit in a worktree against the same
interpreter (same five, same errors). They are drift from the later plotting commits:
label/prose/trace-name changes, plus the move of the systematic-floor error bars from
the fit mark to the MIST mark. Fixed in their own commit so they read separately from
the merge work.

One deliberate call worth flagging: at a **start point** (no posterior) neither mark
draws error bars, and that stays. The fit mark's bars are posterior summaries that do
not exist yet; the MIST mark's systematic floor does exist but is suppressed with them
so a pre-flight chart does not imply a measured uncertainty. The floor still sizes both
axis windows either way, so adding the bars later cannot push a mark off-chart.

Three master tests reached this component for the first time and needed real grids:
they now request the synthetic-grid fixture instead of fetching 128 MB from Zenodo, and
a YAML `model_root:` with no value now means "the default" rather than crashing in
`Path(None)`.

The premature-block warning in `Star.register_parameters` is gated on the component
being **absent**, so it cannot fire any more. The test that asserted it fires now pins
the live branch -- that it must NOT -- which is what keeps the gate from being deleted
as dead code and warning on every ordinary fit.

## Testing

Full suite on macOS x86_64 / Python 3.12.13: **3923 passed, 11 failed, 2 errors,
17 skipped** (1 h 13 m).

**12 of the 13 failures reproduce identically on clean `origin/master`** -- checked by
running each one in a worktree at `bb32967a` against the same interpreter. None is
introduced by this branch. They are this machine, not this PR:

- **11** need jax, which is deliberately not installed on Intel macOS
  (`MACOS_INTEL_INSTALL.md`): the `numpyro`/`blackjax` sampler tests, the
  `jax_funcify` gradient checks, and the two mulens GP errors (`celerite2.pymc`
  imports jax at module scope).
- **1** is a last-ULP float equality in
  `test_galactic_model.py::test_chabrier_is_the_default_and_carries_the_truncation_normalizer`
  (`10.093300692915243 == 10.09330069291524`, asserted with bare `==`).
- **1** is a local clang limit in `test_band_autopin_ld.py`
  (`bracket nesting level exceeded maximum of 256`; wants `-fbracket-depth`).

The thirteenth is `test_runner.py::test_run_without_flag_writes_no_status`, which is
load-sensitive rather than broken: it allows its subprocess `timeout=600` and takes
**393 s running alone** on this machine, so under six parallel workers it exceeds the
budget. It passes in isolation, and the code this PR changes is not on its path (that
branch runs only for a config with an `evolutionarymodel:` key; the test fits
`kelt4_rvonly`, which has none).

A further failure **was** this branch's and is fixed here:
`test_run_dead_residue.py` caught `examples/hat3/hat3_mist.yaml` carrying
`init: adapt_diag`, a sampler key master retired and stripped from the other fifteen
configs. The example predates that change.

The tests this PR touches pass: 229 passed, 1 skipped across
`test_evolutionary_model`, `test_star_evolutionary_model`, `test_run_dead_residue`,
`test_stellar_relations`, `test_torres`, `test_examples_prepare` and
`test_component_override_channel`. The one skip is `hat3_mist.yaml` declining to
fetch the 128 MB grid, which is the conditional skip this branch adds for exactly
that.

CI should be the real check for the jax-dependent tests, since it installs jax.
