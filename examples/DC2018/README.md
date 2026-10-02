# DC2018: the 2018 Roman (WFIRST) Data Challenge, end to end

Fits the challenge's 2L1S sample -- 30 of its 44 events, see "The static
sweep's event list" below -- with the full
pipeline, one cluster job per event:

1. **Seeding**: the params file `run_event.py` writes deliberately carries
   no microlensing start values, so EXOZIPPy's built-in PSPL peak finder
   seeds `t_0`, `u_0` and `t_E` from the light curves (primary-first: a
   planetary anomaly does not capture the seed). `s`, `q` and `alpha` keep
   their defaults.yaml starts -- there is deliberately no binary-lens
   estimator anywhere; the sampler finds the companion (JDE 2026-09-17).
   (Until 2026-10-01 this step ran MMEXOFAST on both bands and seeded the
   fit from its JSON; that hand-off was removed -- see "MMEXOFAST JSONs ->
   params files" below.)
2. **EXOZIPPy** samples the 2L1S system (PTDE, EXOFASTv2-parity settings)
   and writes the usual artifacts under `events/<NNN>/fitresults/`.
   Every light curve fits with `likelihood: hogg` -- the marginalized
   inlier/outlier mixture -- instead of a hard bad-data mask: every point
   stays in the fit, junk lands in the wide background component instead
   of dragging the solution, and per-point posterior outlier probabilities
   are available afterwards via `Instrument.outlier_prob_at_data` -- at the
   cost of two extra parameters per curve (`out_frac`, `out_scale`). Each
   curve's `err_scale` is a fitted parameter starting at 1.
3. **Comparison** against the challenge's answer key
   (`Answers/master_file.txt`, positional lookup, t_0 origin JD 2458234)
   is written to `events/<NNN>/comparison.csv`.
4. **Collection**: `collect_results.py` gathers everything into
   `dc2018_summary.csv` -- one light curve per row; per-parameter value,
   errors, truth, sigma pull, r_hat and ess columns, plus overall
   convergence and err_scale.

## Prerequisites

- The data tree (`n20180816.{W149,Z087}.WFIRST18.<NNN>.txt`,
  `event_info.txt`, `Answers/`). Default location is the MMEXOFAST source
  checkout, `~/python/MMEXOFAST/data/2018DataChallenge`; override with
  `--data-dir` or `$DC18_DATA`.
- An EXOZIPPy environment (`poetry install`); nothing else.

## Quick single-event test (local)

    cd examples/DC2018
    poetry run python run_event.py 128 --quick

`--quick` fits only the ~880-point Z087 curve with tune=500/draws=1000 --
a smoke test of the whole pipeline, not science. Drop `--quick` for the
real thing (both bands, tune=5000/draws=50000).

## On the supercomputer

Test with a single event first:

    cd ~/python/EXOZIPPy/examples/DC2018
    qsub -v EVENT=128 dc2018.job

then run them all (30 tasks, one per line of `events.txt`):

    qsub -t 1-30 dc2018.job

Knobs: `qsub -v EVENT=128,BANDS=Z087,EXTRA="--quick" dc2018.job`. The
sampler core count follows the job's `$NSLOTS`, so it always matches the
`-pe mthread` grant.

After the jobs finish:

    python collect_results.py            # -> dc2018_summary.csv + stdout table

## The static sweep's event list (JDE 2026-09-22)

`events.txt` holds the **30** events a static 2L1S model can fit honestly;
`events_all44.txt` is the challenge's full sample and `events_moving.txt`
the 14 removed, with the measurement behind each.

Every bound planet in the answer key **orbits** (`a`, `inc`, `phase`,
`period` columns) and the simulator moved the lens. A static binary fit
pays for that in `s` and `q`: on event 128, whose planet turns 85 degrees
of orbital phase across the fitted window, the static posterior landed 1.6%
low in `s` and 11% low in `q` at 40 and 21 sigma, and the prior-free
likelihood preferred that biased solution to the truth by 5,900 chi2 --
while a linear-orbital-motion fit recovers `s`, `q` and `rho` to under 1%
(`dc18_orbital_motion_signal.py` docstring, review 2.4.14). Rather than
guess from the period, each event's signal is **measured**: chi2 at the
key's own parameters, static versus moving at the rates the key's orbit
predicts (the orbit reproduces the key's `s` to four digits on all 43
planets, so the convention is exact). Above 25 -- about 5 sigma for two
parameters -- the event leaves the static sweep. Thirteen did, from 29.9
(008) to 9,570 (186); four sit between 10 and 25 and stay flagged (163,
214, 218, 289); event **001 is a cataclysmic variable** and leaves on that
ground. Event **131** stays with a different flag: its static truth fits at
chi2/N = 1.12 and motion does not help, so something else is in that curve.

These events come back when the orbital-motion rung exists (the model does:
`orbital_motion: linear` / `keplerian` on the lens block, conventions.md
C24). Architecture selection -- orbital motion, binary-star lenses, binary
sources, the challenge's CV and free-floating-planet classes -- is the
roadmap item after static 2L1S works (`notes/todo.txt`, microlensing).

## MMEXOFAST JSONs -> params files (2026-10-01)

The `mmexofast:` (and `mmexofast_options:`) config key was removed with the
MMEXOFAST hand-off, and a config that still names it RAISES at load time
with the migration. `convert_mmexofast_json.py CONFIG.yaml ...` does that
migration for an existing config and its JSON, reproducing exactly what the
removed loader did, but as user input:

- `fits` -> per-seed `initval: [...]` lists in the params file (one entry
  per fit; `t_0` has the JSON's `jd_offset` subtracted; `s` becomes
  `log_s`); a path the params file already starts is left alone, since a
  user entry always outranked the seeds;
- `excluded_points` -> the instrument entry's `mask:` (0-based row
  indices), skipped for files with a robust `likelihood:` or their own
  `mask:`, as before;
- `errfacs` -> `mulensinstrument.<name>.err_scale: {initval: ...}`;
- more than one fit -> `sampler: {seed_polish: true}` (`auto` polishes a
  multi-seed set only when a component seeded it, and would read user
  lists as posterior-draw restarts);
- fit 0's `sigmas` are dropped: they only seeded the whitening probe, which
  measures the scales itself.

The `sweep/`, `sweep_v7_avW149/` and `ab194/` configs were converted, with
start logp bit-identical to the JSON path; their seed JSONs became
`events/<NNN>/DC2018_<NNN>_seed.params.yaml` fragments, which
`dc18_seed.py` now writes and `dc18_sweep_config.py` merges into the params
file it generates. Twelve configs name a `*_mmexofast.json` that exists
only on the cluster (those caches are gitignored); they still name the key,
carry a "REMOVED KEY" comment above it, and raise until the converter is run
there:

    configs/DC2018_128_severed_v3.yaml ... configs/DC2018_128_severed_v7.yaml
    configs/DC2018_128_tightpriors.yaml
    events/062/DC2018_062.yaml
    events/152/DC2018_152.yaml
    events/152/base/DC2018_152_base.yaml
    events/152/obs/DC2018_152_obs.yaml
    events/194/DC2018_194.yaml
    events/223/DC2018_223.yaml

The 22 pre-v0.1.0 configs (event options on the `lens:` block) under
`configs/` and `events*/128/` already did not build and were left
untouched; `configs/README.md` says how to port one.

## Caveats

- **alpha**: the answer key measures alpha to the SOURCE's motion from the
  planet orbit's LINE OF NODES, so it maps onto EXOZIPPy's (= MulensModel's =
  MMEXOFAST's) by a per-event rule,
  `alpha_EXZ = alpha_key + 180 - atan2(sin(phase) cos(inc), cos(phase))`
  (`key_alpha_to_exozippy` in `dc18_common.py`; claim C22 of
  `src/exozippy/components/mulensing/conventions.md`). Measured 2026-10-02:
  19/19 events whose anomaly pins alpha (chi2 contrast >= 1000) match the
  light curve's own alpha to a median 0.10 deg, max 1.53 deg. The comparison
  reports a truth and a circular pull. Orbits shorter than 2 yr are FLAGGED:
  the simulator moves the lens, the key's alpha is the t_0 geometry, and a
  static fit sits a few degrees off it (event 128: -1.7 deg, which at a
  0.03 deg error bar is a ~50-sigma pull that is physics, not a fitting
  failure). Until 2026-10-02 the key was recorded as unmappable, because
  only global offsets had been tested; the old sign/offset search stays
  deleted (it always returned its closest candidate).
- **u_0** is compared SIGNED: the key's sign maps by the identity. The only
  ambiguity is the exact no-parallax mirror (u_0, alpha) -> -(u_0, alpha)
  (C23): a fit in the key's mirror branch is scored against the mirror
  image and its u_0/alpha rows say "mirror tie" -- reported, not folded.
- Event 1 is a cataclysmic variable, not a lensing event; expect the 2L1S
  fit to fail or diverge on it (the collector will show it as such).
- Each event directory gets a dumped `DC2018_<NNN>.yaml` +
  `DC2018_<NNN>.params.yaml`, so any event can be re-run or debugged with
  the plain CLI: `cd events/<NNN> && exozippy DC2018_<NNN>.yaml`.
