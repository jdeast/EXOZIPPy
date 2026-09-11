# Pharmacokinetics: the components, and what building them found

## READ THIS FIRST

**This component set was written by an astrophysicist and an LLM. No biologist,
pharmacologist, clinician, or pharmacometrician has reviewed it.** It exists to
demonstrate and enforce the component-agnostic architecture and to be a
starting point for non-astronomy development. Reproducing a published fit
validates that the code computes the model it claims to; it does not validate
that the model or its priors suit anyone's data. The full caveat, and what a
domain expert would need to check, is in `README.md` next to this file -- read
that before changing anything here.

Read this document before editing the components. It is the ruling record: the
design decisions, and -- because that is half the point of the exercise -- what
building a non-astronomy component actually ran into.

## The components

| Component | Instances | Owns | Astronomy analogue |
|-----------|-----------|------|--------------------|
| `subject` | one per individual | the PK parameters | `star` |
| `assay` | one per data file | observations, residual error model, likelihood | `rvinstrument` |

`population` (between-subject variability, allometric covariate) is P4 and does
not exist; every subject here is independent, so this is not yet "population"
PK in the field's sense. `physics.py` holds the forward model and is where the
numerically interesting part lives.

## The forward model, and its two numerical traps

One compartment, first-order absorption, single oral dose. `F` is pinned at 1
because it is exactly confounded with `V` and `CL` on oral-only data, so every
estimate is an *apparent* `CL/F` or `V/F` -- the labels say so, and a reader who
takes them for `CL` and `V` is wrong by `1/F`.

**The `ka == ke` singularity is removable and must not be branched.** Folding
`exp(-ke t)` into the bracket turns the curve into `exp(-m) sinh(s)/s`, with
`m` and `s` the mean and half-difference of `ke*t` and `ka*t`. `sinh(s)/s` is
even, so the floor goes on `s**2` -- no sign to lose -- and the floor is
**strictly positive**, the `CHORD_RADICAND_FLOOR` rule.

**How that term is spelled cost two wrong answers, both found by numerical test
and both now pinned:**

- `exp(-m) * sinh(y)/y` **overflows**: `sinh` reaches `inf` near `y = 710` while
  `exp(-m)` has underflowed to `0`, giving `0 * inf` -> NaN. That is the exact
  gradient poisoning the floor exists to prevent, reintroduced at the other end
  of the range, and it is reachable by an ordinary excursion to a large `ka`
  over a 25-hour window.
- `(exp(y-m) - exp(-y-m)) / 2y` fixes the overflow and loses **six digits** to
  cancellation at small `y` (measured: 1.1e-10 relative error at the limit).

The working form is `exp(y-m) * (-expm1(-2y)) / (2y)`. Do not "simplify" it
back; `_damped_sinhc` carries the measurements.

**The flip-flop degeneracy is exact.** Swapping `ka <-> ke` and rescaling
`V -> V*ke/ka` leaves every prediction bit-identical, with `CL` invariant and
`V` not. Nothing breaks the symmetry: it is a real property of oral-only data,
and truncating one mode away would be a hard bound on a posterior that hugs it
(the failure `_restrict_bigomega_halfplane`'s removal documents). `Subject`
sets `expects_suppressed_modes = True`, which turns on hot-chain retention
generically.

## The coordinate choice: NONMEM TRANS1 and TRANS2

NONMEM ships two spellings of this one model and users hold opinions about
which: **TRANS2** samples `(CL, V)` and derives `ke = CL/V`; **TRANS1** samples
`(ke, V)` and derives `CL = ke*V`. Same model, different coordinates -- so it
is a *coordinate choice*: `parameterization: cl_v` (the default) or
`parameterization: ke_v`, per subject, and a system may mix them.

**It is an enum, not a `fit<coord>` boolean, and `components.md` now says
why.** That flag vocabulary is explicitly about BOOLEAN flags and had no entry
for an n-way choice, so `fitke: true` looked like the house style. It is not:
the house style for an n-way per-instance choice is `band.ld_law` and
`planet.mass_parameterization` -- an enum whose values name the alternatives.
Here they name the sampled pair, so a config says what it selects without a
lookup, and P4's third basis is a new VALUE rather than a second flag that
would have to be checked against the first. (The design note called for
`pk_trans:`, reasoning that a coordinate choice is "spelled after the
coordinate, not as a `fit<x>` toggle"; the first half was right.)

`Subject.COORD_MODE_TABLE` is the whole implementation -- a `mode_manifest`
table, not a hand-built mask -- and **the interesting half is which role each
coordinate takes, because both of the interesting roles appear:**

| under `cl_v` (TRANS2) | under `ke_v` (TRANS1) | |
|---|---|---|
| `log_cl` sampled | `log_cl` **reported** | |
| `cl` derived, consumed by `ke` | `cl` **reported** | the likelihood needs only `ka`, `ke`, `V` |
| `ke` derived, consumed by the model | `ke` derived, from `log_ke` | |
| `log_ke` **reported** | `log_ke` sampled | |

`cl` under `ke_v` is the textbook case for `reported`: consumed by nothing,
and still the one quantity a PK table exists to show.

**Three traps, all of them load-bearing:**

1. **`auc` must not be written `dose/cl`.** That is the identity it is, and it
   raises under `ke_v` -- nothing may consume a reported element, and
   `System._validate_reported_not_consumed` refuses the manifest. It is
   written `dose/(ke*v)`, which is the same number and consumes only
   quantities that are sampled-or-derived in both.
2. **A mixed system would deadlock a naive wiring.** `cl` is derived from
   `ke` on the `ke_v` subjects while `ke` is derived from `cl` on the `cl_v`
   ones, so the per-parameter build order would see `cl -> ke -> cl`. It does
   not, because a *reported* selection contributes no edge to `graph.py`.
   That is not a coincidence to be re-derived per component: it is the
   property the role was designed around (`parameter.md`), and a table whose
   modes derive a quantity from something the other mode *derives* rather
   than reports would cycle.
3. **`t_half`, `tmax`, `cmax` and `auc` stay `derived` and must not be
   flipped to `reported`,** which the design note said P3 would do. "Nothing
   consumes it" is the role's description, not its purpose. `reported` is for
   a coordinate a parameterization *masked out*, and it deliberately carries
   no potential -- so `subject.S1.t_half: {mu: 8.0, sigma: 0.5}`, the way the
   literature quotes these, would be silently discarded. As a derived element
   it becomes a Gaussian on `ke`, which is what the user asked for.

## What building this found about the core

The point of the exercise. Measured, not speculated.

**The core is genuinely component-agnostic.** `src/exozippy/*.py` and
`outputs/*.py` contain zero hardcoded component names (the only hit for "star"
is a plot marker glyph in `plot_theme.py`), and `system.py` has no reference to
any component. Both components were written against `Component` + `Parameter` +
the manifest vocabulary + the four-file layout, as `components.md` declares,
and needed no change to the core to build, sample, and report.

Five places where the fit was not frictionless:

1. **The reporting convention was hardcoded, and it mattered most.** Median +
   68.3% (1-sigma) is astronomy's; pharmacometrics reports 95%. Fixed before
   any of this component existed -- `exozippy.reporting`, and the
   "credible-interval width" section of `outputs/outputs.md`. Note the two
   fields do not merely use different widths but different *kinds* of number:
   sigma multiples versus exact round probabilities.
2. **The prose layer's ORDERING vocabulary is astronomy-shaped** -- but only
   the ordering, and this is milder than it first looks. Two mechanisms are
   easy to conflate and only one has a problem:

   - The **results-table side headings** come from `Component.label` and are
     component-generated already (`outputs/latex.py`'s `\sidehead`). `Subject`
     and `Assay` supply their own; no astronomy vocabulary is involved. This
     is exactly the firewall it was designed to be, and it works.
   - The **modeling-draft prose** (`outputs/prose.py`) keys each sentence to
     one of thirteen `SECTION_ORDER` slots, four of which are `stellar`,
     `planetary`, `orbits`, `microlensing`. An unknown slot *raises*, rightly.

   Crucially those slot names are **not printed**: `modeling._DOC_SECTIONS`
   routes all thirteen into just three `\section{}` headings -- Observations,
   Modeling, Results -- which are already field-neutral. So a reader never sees
   the word "stellar", and the astronomy vocabulary is purely an internal
   ordering key.

   The residual problem is therefore ordering, not text: a non-astronomy
   component has no slot that describes it, so `Subject`'s "what we fitted"
   sentence is filed under `data`, where it will be ordered among
   data-inventory sentences rather than after them.

   **FIXED.** The band is now extensible: a component declares `prose_topic` and `System` registers it in build-graph order (`outputs/outputs.md`). `subject` declares `pharmacokinetics`, so its model sentence has its own paragraph under Modeling instead of sitting among the data inventory. An all-astronomy system's order is unchanged. The originally proposed fix was WRONG and worth recording as such: collapse the four physics slots into
   one `model` slot and order sentences within it by the component's position
   in `graph.determine_pymc_build_order`'s topological sort. Dependency order
   is the right editorial order -- inputs before things derived from them --
   and it reproduces stellar -> planetary -> orbits for astronomy while giving
   subject -> assay here, with no vocabulary to extend when a new field
   arrives -- but it would have MERGED FOUR PARAGRAPHS INTO ONE and destroyed
   the topical grouping, which is real information (an orbit-ish sentence from
   `rvinstrument` belongs with `orbit`'s, not with `rvinstrument`'s other
   sentences). Keeping topics as named subjects and deriving only their ORDER
   from the graph is what preserves both. The ordering half of the instinct
   was right and is what shipped.
3. **Numeric instance names are rejected, and clinical data are numerically
   labelled.** `validate_instance_names` refuses a purely numeric name because
   it would be ambiguous with the internal `subject.0` index form -- correct,
   and it collides head-on with a field where subjects are `1..12`. Handled in
   the component with an explicit `subject_prefix` on `assay` rather than a
   fallback that tries the bare name and then a prefixed one, since a silent
   second attempt would pair the wrong rows whenever both spellings exist.
4. **`Instrument` is the astronomy-coupled scaffold, not the core.** Its data
   machinery -- columns, masks, detrending, GP, robust likelihoods, jitter -- is
   field-neutral and would fit `assay`. Its vocabulary is not: `time_frame`
   defaults to `bjd`, `time_offset` is in days, `_to_bjd_tdb`, `time_location`
   as an observatory. Those defaults pass a blood draw through **untouched**,
   which is the trap: inheriting would have worked while documenting a
   barycentric correction on a plasma sample.

   `assay` therefore inherits `Component` directly, and **what it
   re-implements is the finding**: CSV reading, a column-role mapping
   (extended to accept header names), non-finite row dropping, a data-side
   unit conversion, and a per-file noise seed pushed through `add_hint`.

   **ACTED ON, partly.** The time layer is now a separable mixin,
   `components/timesystem.py`, extracted verbatim from `Instrument` -- see
   `components/instrument.md`. That is the field-coupled half; the remaining
   list above is the field-neutral half, and is what a future `DataComponent`
   base would carry. `tests/test_timesystem_split.py` pins that `assay` has no
   time vocabulary, which is the property the split exists to guarantee rather
   than merely intend.

5. **A user constraint on a `reported` element was dropped in silence, and
   now warns.** `parameter.md` said `inactive` was the one lossy role in a
   parameterization flip. It is not: `gaussian_prior_mask` excludes reported
   elements and so does the soft barrier, and *necessarily* -- a reported
   value is a placeholder until `finalize_deferred` patches it after stage 7,
   so a potential built in phase 1 would penalize the pre-patch vector. The
   exclusion is right; the silence was not. `subject.S1.cl: {mu, sigma}` is
   the single most natural prior a user of this component writes, and under
   `parameterization: ke_v` it vanished without a word. `build_pymc` now warns for a
   reported element exactly as it does for an inactive one, keyed on what the
   user wrote. **This is a shipped astronomy bug too**, reached by
   `orbit.b.secosw: {mu, sigma}` under `fitvcve` -- found from a field with
   no stars in it, which is the sort of thing this exercise is for.

Two smaller notes: `utilities/zenodo.fetch_assets` is generic despite its name
and is reused here for a non-Zenodo URL, but it prints "Downloading ... from
Zenodo", which is now inaccurate for one caller; and `add_hint` correctly
refuses a 2-part broadcast path, which is worth knowing before writing a
component's first seed.

## Data, units, and why the data are fetched

Internal units are hours, mg, L and mg/L. Parameters convert through the
`Parameter` layer; the two *data* columns have no `Parameter` and are converted
in `Assay.load_data`, loudly -- `conc_unit: ng/mL` is a factor of 1000, the size
of error this codebase has hidden before.

The Theophylline data are **not redistributed**. EXOZIPPy is BSD-3-Clause and
the table reaches most people through R's GPL-2 `datasets` package; bundling
GPL-2 material in a BSD distribution is a compatibility problem for this
project, even though GPL-2 plainly permits redistribution -- it is copyleft, not
a ban, and the table is in any case measured factual data whose
copyrightability is doubtful. Rather than make that judgement call on a user's
behalf, `exozippy-fetch-theoph` downloads it on request, md5-pinned.
`examples/theophylline/.gitignore` is the enforcement; do not `git add -f` the
CSV.

## Phases

P0 (reporting width), P1 (these two components) and P3 (the TRANS1/TRANS2
coordinate choice, above) are done. P2 -- still open -- adds
`symbolic_physics.py` so the relaxation engine can accept a half-life where the
model wants a clearance. P4 adds `population`; P5 does the degeneracy
reporting. See `README.md` for the table and for which caveat copies each phase
owes.

P3 was done before P2 deliberately: they are independent, and P3 is the one
that tests a documented core contract (the element roles) rather than adding a
convenience. Note what P2 would change about P3: today a user's `cl` in the
`ke_v` basis reaches nothing at all, because with no relations the engine cannot
translate a clearance into a `log_ke` start either.

Tests: `tests/test_pharmacokinetics_physics.py` (the forward model),
`tests/test_pharmacokinetics_components.py` (config, maps, units, the built
model).
