# Blind global period searches (`components/globalsearch.py`)

How a fit gets an orbital period and a time of conjunction when nobody typed them in:
Box Least Squares over transit photometry, Lomb-Scargle over radial velocities, and the
ranked-hint channel that turns a detection into a start value. Read this before changing
what the searches seed, when they run, or how a component pushes a start value that a
stage-2 consumer has to see.

Related: `src/exozippy/config.md` (hints, overrides, provenance ranks and the relaxation
engine), the MMEXOFAST integration this copies end to end
(`components/mulensing/mmexofast_support.py`), and the `Instrument` family
(`components/instrument.py`).

## What it is for

`params.yaml` becomes **optional** for the simple blind case: a single-planet system with a
light curve, or with radial velocities, and no start values for the orbit. Point EXOZIPPy at
the data and the period and conjunction epoch are measured rather than typed. Nothing here
enters a likelihood -- these are start values, and a start value cannot move a posterior.

Both algorithms are `astropy.timeseries`' (`BoxLeastSquares`, `LombScargle`). astropy is
already a hard dependency and these are the reference implementations; nothing in this
module reimplements a periodogram, only the grid limits, the detection thresholds and the
translation into EXOZIPPy parameters.

## What is seeded, and what deliberately is not

| quantity | from | parameter |
|---|---|---|
| orbital period | BLS peak / Lomb-Scargle peak | `orbit.<n>.period` (the engine derives `logP`) |
| time of conjunction | BLS transit time / LS sinusoid phase | `orbit.<n>.tc` |
| radius ratio | `sqrt(BLS depth)` | `planet.<n>.p` (the engine derives `radius`) |
| RV semi-amplitude | the LS sinusoid's amplitude | `planet.<n>.K`, via `RVInstrument.k_init` |

Not seeded, each for a reason:

- **a/R\* from the transit duration.** `planet.ar` carries `rank: 5` in its defaults.yaml
  precisely so the relaxation engine's Condition B always rewrites *it* rather than the
  period, mass or radius it is derived from -- so a hint there is discarded by design.
  Forcing it through would invert exactly the direction the ranking system exists to
  protect: a duration-derived a/R\* silently moving a stellar mass. The duration is
  reported in the log and by the CLI utility instead.
- **K as a hint.** `planet.K` already gets a data-driven hint from
  `RVInstrument._estimate_k_init`, and for a single circular signal `sqrt(2) * std` IS the
  semi-amplitude exactly. Rather than add a second competing hint on the same parameter
  (planet's is pushed at stage 2 and would overwrite a stage-1a one regardless), the LS
  fit's amplitude replaces `self.k_init` at the source and flows through the existing
  channel. The gain is confined to the low-SNR case, where `sqrt(2) * std` counts the noise
  variance as signal.
- **Astrometry.** Explicitly out of scope; see `AstrometryInstrument.get_utilities`, whose
  docstring records why a periodogram is the wrong shape for both 1-D scan-angle astrometry
  and 2-D (ground / Roman) astrometry, and that the two would need different algorithms.
  The placeholder is declared and `available=False` rather than filled with whichever method
  happened to be written down.

## Whether the search runs at all

Derivability, never a literal-key scan -- the MMEXOFAST rule, for the same reason.
`orbit.period` is a *derived* parameter (`10**logP`), so a restart file written by `mkparam`
never names it, and a scan would re-run the search on every second-iteration fit.
`starts_satisfied` therefore asks `ConfigManager.probe_derivable`, which runs the relaxation
engine on a snapshot and rolls every mutation back, and tests on **provenance above
`RANK_DEFAULT`**. A group whose alternatives are all literally named short-circuits the
probe, so the ordinary hand-written params file pays nothing.

Two refinements of the MMEXOFAST pattern:

- **Only the missing quantities are seeded.** A params file that gives the period but not
  the epoch keeps its period and gains an epoch. That is what removes every precedence
  question: where a search seeded something, nobody else had.
- **The answer is cached per group**, and that is load-bearing rather than an optimization.
  `probe_derivable` runs the engine, which layers `config_manager.hints` -- so once any
  search has seeded a period, every later caller is told the period is already derivable.
  On a system with both photometry and RVs that made the answer depend on config key order.
  The question being asked is "did the INPUTS supply this?", which has exactly one answer
  per run.

`global_search:` on the **orbit** block is the switch: absent (default) runs the search only
when something is missing, `true` forces it, `false` opts out. It lives on the orbit block
for the same reason `mmexofast:` lives on the lens block -- that is the thing being seeded;
the instrument running the search only borrows the switch, and `Orbit` itself never reads
it.

## Stage 1a, and why not the usual 1a/2 split

`Transit._hint_baseline` measures in `load_data` (stage 1a) and pushes in
`register_parameters` (stage 2). This feature does **both** in stage 1a, and the reason is
concrete: `Orbit.register_parameters` builds `tc`'s HARD bounds as `tc_init +/- P/2` from
the start values it can see at stage 2. A seed pushed from another component's stage 2 would
race against orbit's -- `System.prepare` walks `active_components` in config key order --
and if orbit went first, the searched epoch would land outside a window built around the
defaults.yaml 2460000, which `Parameter.build_pymc` correctly treats as fatal. Every
component's stage 1a precedes every component's stage 2, so stage 1a is the only placement
that is right for both orderings. MMEXOFAST pushes at stage 1a for the same class of reason
(its own flux bootstrap, later in the same `load_data`).

## Two channels for one number

`globalsearch.seed_start` writes each seed twice, and both are needed:

- **`ConfigManager.add_hint`** at `RANK_DERIVED_DATA` is the ranked start value -- what the
  relaxation engine solves from, what the provenance ledger, `initval_source` ("data") and
  `export_solution` report, and what every user entry outranks. This is the channel a start
  value must use.
- **`ConfigManager.add_override(path, initval=...)`** exists here only because
  `ConfigManager.resolve()` does not layer `self.hints` -- so the stage-2 readers that ask
  `resolve()` for a start value (`Orbit`'s `tc` window, `Orbit._seeded_period`) cannot see a
  hint at all. Overrides ARE layered by `resolve()`, before the user's params, so this makes
  the same number visible to them with the same precedence.

They carry one value computed once, so they cannot disagree, and both lose to the params
file. **The cleaner fix is for `resolve()` to layer hints as well**; that is `config.py`, and
until it happens the override half is what keeps the `tc` window honest. Note the two
channels take different units -- `add_hint` the parameter's *user* unit, `add_override` its
*defaults.yaml* unit -- and `seed_start` takes the internal unit and converts for both,
warning and skipping the override half if the defaults unit is not the internal one (it is,
for all three parameters seeded today: days, days, dimensionless).

`seed_start` also breaks the one tie the rank system cannot: a transit epoch and period beat
an RV one by orders of magnitude, but both are honestly `RANK_DERIVED_DATA`
(`config._provenance_label` reports any other value as "solved"), and `add_hint` is
last-writer-wins. A per-ConfigManager registry records `(quality, value, source)` per path,
so `QUALITY_TRANSIT > QUALITY_RV` decides it whatever order the components ran in.

## Refusing to guess

A wrong confident seed is worse than no seed -- `orbit.tc`'s hard window is `tc +/- P/2`, so
a bad epoch does not merely start the chain in the wrong place, it makes the right place
unreachable. Three refusals:

- **Below threshold, nothing is seeded.** BLS needs SDE >= 7 *and* depth SNR >= 7.5 (the
  peak inflates the SDE's own denominator, so the statistic errs conservative);
  Lomb-Scargle needs a Baluev false-alarm probability below 1e-3. A miss is a warning that
  names what was searched and by how much it fell short, and the defaults.yaml starts stand.
- **More than one orbit and the search does not run.** A periodogram returns peaks, not
  attributions: assigning the strongest peak to `orbit[0]` and the next to `orbit[1]` is a
  guess, and harmonics and aliases make it a guess that fails in the way that hurts most --
  one planet's period seeded twice, confidently, into two different orbits. A multi-planet
  blind fit needs a params file; the standalone CLI utilities report the peaks for one.
- **Too few points, no baseline, an empty grid**: declined, with the numbers.

## The conjunction convention

For a circular orbit EXOZIPPy's own RV model reduces to
`RV(t) = -K sin(2 pi (t - T_C) / P)` -- read off `Orbit.get_radial_velocity` and
`physics.calc_tp_from_ecc`, where the true anomaly at conjunction is `f = pi/2 - omega` and
the omega dependence cancels exactly. The star therefore recedes before conjunction and
approaches after it, which is what makes this the same `T_C` a transit measures.
`_conjunction_from_sinusoid` inverts that. A sign slip there is a half-period phase error --
a confidently wrong seed -- so it is pinned twice: against the component's own
`calc_tp_from_ecc` + Kepler chain at several omega, and end to end by requiring the seeded
start's logp to beat the unseeded one (a half-period error would make it worse).

Both searches report the conjunction in the cycle nearest the middle of the data
(`fold_epoch`): the further the seed sits from the observations, the more a period error
leverages into a phase error, and the `tc` window is only one period wide.

## The standalone utilities

`python -m exozippy.utilities.bls <light curves...>` and
`python -m exozippy.utilities.lomb_scargle <rv files...>` run the same searches over raw
files and print a params.yaml snippet. They are the component-declared utilities the GUI
surfaces (`Transit.get_utilities`, `RVInstrument.get_utilities`), and they are how to look
at a system the automatic path refuses.

## How far "params.yaml is optional" actually goes

A single-planet transit fit or a single-planet RV fit, with a `parameter_file:` naming an
empty (or minimal) params file. The config key itself is still required -- `System.__init__`
raises without it -- so making it fully optional is a separate change in `system.py`.
Everything else the fit needs (stellar parameters, limb darkening, noise) already has a
defaults.yaml start.

Tests: `tests/test_global_search.py`.
