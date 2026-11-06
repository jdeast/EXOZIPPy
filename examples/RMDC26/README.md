# RMDC26: per-event evidence, relative probabilities and the submission

This directory turns finished RMDC26 (2026 Roman Microlensing Data
Challenge, beginner tier) fits into `microlens-submit` files. It is a
port of `examples/DC2018_069`'s stages 4-5 to the RMDC26 trace layout.

The campaign itself lives OUTSIDE the repo, at
`/n/holylabs/jjohnson_lab/Lab/RMDC26_run` (configs, traces, the triage).
That directory is another agent's live run: these scripts only READ it.
Everything they write goes under `$RMDC26_REPORT`, which defaults to
`/n/holylabs/jjohnson_lab/Lab/rmdc26_report`. Never point it at `$HOME`.

## The layout being read

- Per event, `events/<NNN>/RMDC26_<NNN>.yaml` plus its `.params.yaml`, and
  `events/<NNN>/fitresults/` holding `RMDC26_<NNN>_trace.nc`,
  `_whitening.json`, `_summary.txt`, `.log`, `_modes.txt` and, where the
  wrap-up got that far, `_outliers.txt` and `_results.csv`.
- **There is one fit per event and architecture.** The 113 anomaly
  candidates are fitted as 2L1S (config has `planet:` and a two-body
  `lens:`), and the 40 single lenses as PSPL. The architecture is read
  from the config, not from the event list.
- **A 2L1S run carries several grid seeds** (`gridsearch.seed_fragment`,
  up to 5 `(s, q, alpha)` basins in one PTDE run). `identify_modes` splits
  the run into the event's modes, so there is no cross-run dedupe here,
  unlike DC2018_069. Every identified mode of every fit is one solution.
- Extra fits of an event sit in `events/<NNN>_<tag>/` and are picked up
  automatically. A fit whose `band.*.u1` is not pinned at 0 is SKIPPED,
  and the skip is recorded in the event's notes (the PI ruled zero limb
  darkening). `events/094_pspl_backup` is such a fit: a free-u1 PSPL.
  Pass `--allow-free-ld` to include it anyway.

## The check that comes first: does master reproduce the sampler's lp?

The traces were sampled by an older tree (`wt/launch1015`, commit
`0a7f7ade` stamped in every trace). The evidence normalization
(`System.evidence_log_norm`, PR #465) exists only on master, whose config
engine was rewritten in between. A bridge lnZ integrates whatever logp the
REPORTING tree compiles, so it is the sampled posterior's evidence only if
that logp equals the sampler's at the stored draws.

`lpcheck.py` tests exactly that. It rebuilds the model from the fit's own
config, restores the trace's whitening (`restore_whitening_for_trace`,
never re-measured), evaluates `model.compile_logp(jacobian=True)` at 50
random stored draws, and compares the result with `sample_stats.lp`:

    sbatch lpcheck.sbatch <tree> <tag> <fit_dir> ...   # see the header

Results, 2026-10-21:

| fit | tree | max abs(dlp) over 50 draws |
|---|---|---|
| 094 (2L1S, 17 value vars) | master (this branch) | 4.7e-10 nat |
| 094 | wt/hogg467 (launch1015 + #467 + #472) | 4.7e-10 nat |
| 012 (PSPL, 13 value vars) | master | 0 |
| 012 | wt/hogg467 | 0 |

Master reproduces the sampled logp to float rounding, so the bridge can run
on master and the normalization applies directly. Neither the old tree nor
a ported constant is needed. `01_evidence.py` repeats the same 50-draw
check on EVERY fit it processes. If a fit fails it (max abs(dlp) >= 0.01
nat), that fit's solutions are UNRELIABLE and none of its lnZ enter a
probability.

## The two stages

    cd examples/RMDC26
    sbatch run_evidence.sbatch 94 173 16          # stage 1, test partition
    sbatch run_evidence.sbatch --events-file finished.txt
    srun -p test -c 1 --mem=8G -t 30 python 02_report.py     # stage 2

### Stage 1 -- `01_evidence.py` (per fit; heavy)

For each fit of each listed event:

0. **Stale traces are skipped** (`rmdc26.trace_staleness`). A config or
   params file newer than the trace means the configs were regenerated
   after the fit ran. On 2026-10-20 03:46 they were rewritten for zero
   limb darkening, so a trace from before then is the free-u1 model's. A
   whitening file newer than the trace means a re-fit has started and
   overwritten it. On 2026-10-21 this skipped 015 (both reasons), 022 and
   187. Before this rule existed, the lp check caught 015 on its own
   (max abs(dlp) = 200 nats, against its re-fit's new whitening), and 022
   raised `StaleWhiteningError` ("'band.u1' is no longer whitened").
1. **Stage and rebuild.** The config and params are copied into
   `$RMDC26_REPORT/work/` with `prefix:` repointed there, and the trace
   and whitening files are symlinked. The model is rebuilt on this tree
   with the whitening restored.
2. **lp check** at 50 draws (above).
3. **Valid draws only.** `fold_degeneracies` runs first. Then the burn-in
   is applied: the larger of the wrap-up's own (read from `_summary.txt`)
   and master's `find_burnin`. The two can differ: on 094 the wrap-up said
   75% and master's max-ESS rule says 0%, and the early draws of a
   multi-seed run carry the seeds' transient. Master's stuck-chain mask
   then drops chains, and `identify_modes` labels its rejected draws -1,
   which belong to no mode. Rejections are common on this data (173:
   25% raw-z rejected on the old tree); the invalid fraction is recorded
   and flagged above 5%.
4. **Normalized lnZ per mode:** `estimate_mode_evidences(...,
   max_posterior_draws=20000, log_norm=system.evidence_log_norm())`.
   A single-mode run is one mode.
5. **Per mode:** the posterior median and 68% interval of every
   submission parameter; the robust-likelihood audit
   (`outputs/outliers.py`, the code that writes `_outliers.txt`) at the
   mode's highest-lp draw and at its median draw; err_scale per band; and
   the hogg and Gaussian log-likelihood and the point count at the best
   draw.
6. **Per fit:** the stuck-chain fraction, lp split-Rhat, the wrap-up's
   convergence and NOT EQUILIBRATED verdicts, the invalid fraction, and
   out_scale split-Rhat per band.

Output: `$RMDC26_REPORT/evidence/<NNN>/<fit>.json`. An existing record is
skipped unless `--force`. An exception is recorded in the JSON, never
dropped.

### Stage 2 -- `02_report.py` (per event; light)

This stage writes the probabilities, flags, the submission files and the
summary tables. Every rule it applies is a function in `rmdc26.py`, and
each is pinned by `tests/test_rmdc26_report.py`.

## The rules

### Relative probability (`rmdc26.robust_probabilities`)

1. **Which modes count.** Only modes whose bridge was ACCEPTED and whose
   fit passed the lp check contribute an lnZ. A refused mode gets 0 and is
   inactive. If no mode of the event is accepted, the refused estimates
   are used instead, and everything for that event is labelled
   `INDICATIVE`. A refused mode whose indicative lnZ is above the best
   accepted mode, or less than 3 floors (6 nats) below it, adds a warning
   to every solution of the event (`rmdc26.refused_competitors`). Its mass
   is then missing from the accepted probabilities.
2. **Raw probabilities.** `p_raw` is a softmax of lnZ over ALL modes of
   ALL architectures of the event (1S1L and 1S2L together). lnZ is
   normalized, so the architectures share one scale.
3. **Systematic floor.** Each mode's total error is
   `sigma = sqrt(err_bridge**2 + 2**2)`. The 2-nat floor is the
   run-to-run scatter measured on DC2018_069: two runs of one basin
   differed by 2.2 nats against bridge errors of 0.2.
4. **Indistinguishable groups.** Groups are formed greedily from the top.
   The best remaining mode leads a group, and any remaining mode with
   `lnZ_lead - lnZ_k <= sqrt(sigma_lead**2 + sigma_k**2)` joins it. The
   rule does not chain: membership is always measured against the leader.
5. **Robust probabilities.** Each group's total `p_raw` is shared EQUALLY
   among its members. This replaces only the split the evidence cannot
   make, and keeps the group totals.
6. **Architecture probability.** The sum of the robust probabilities of
   one `model_type`. It is written into every solution's notes, because
   microlens-submit has no architecture field. When an event has no
   usable fit of an architecture, the notes say that the probabilities
   are CONDITIONAL on the architectures that were fitted. Today that
   applies to every 2L1S event: no zero-LD PSPL exists for any of them.
7. **`is_active` and normalization.** A solution is `is_active` when its
   robust probability is >= 1e-3. Under the floor, two modes are only
   distinguishable beyond ~2.8 nats. A cut at 1e-3 sits at dlnZ = 6.9,
   about 2.4 combined sigma, so everything plausibly competitive stays in
   the export. The microlens-submit validator requires the active
   solutions to sum to 1, so they are renormalized. An inactive solution
   keeps its unrenormalized probability.

### Reliability flags (`rmdc26.reliability_flags`)

**UNRELIABLE** means "do not trust this lnZ or probability". It is set
when any of these holds:

- the event is in `FORCED_UNRELIABLE` (016, 163, 173, set by the campaign
  agent 2026-10-21). 173 also flags VARIABILITY + FORGIVEN + AT_CAP in all
  three bands, and a third of its chains sit in a worse t_0 mode;
- the lp was not reproduced;
- the bridge was refused, so there is no accepted lnZ;
- any err_scale lies outside [0.8, 1.25]. The noise model, not the lens
  model, is then carrying the fit. The known cases are 016 F213 21.0,
  173 F213 8.1 and 163 F087 3.2;
- a **1S1L** solution has an ANOMALY outlier flag: the mixture is
  absorbing points AT the event, so the single lens is missing a signal
  and its lnZ is subsidized.

**Warnings** cover:

- NOT EQUILIBRATED, from the wrap-up log or from master's check on the
  trimmed lp. It is listed first, because it makes Rhat and ESS
  meaningless.
- Unconverged: Rhat/ESS against the config's own thresholds.
- A mode held by fewer than 3 chains.
- Stuck chains: more than 10% of chains with a second-half mean lp more
  than 20 nats below the best chain, as in `diag30.py`.
- lp Rhat > 1.1.
- POSSIBLE_UNLISTED_MODE, or else more than 5% invalid draws.
- Bridge re2 > 0.1.
- ANOMALY on a 1S2L solution. On 2026-10-21 it fired on most binary
  modes, so on its own it does not condemn one.
- The VARIABILITY and FORGIVEN outlier kinds.
- AT_CAP, with that band's out_scale Rhat beside it.
- An out_scale Rhat > 1.1.

**POSSIBLE_UNLISTED_MODE (<param>)** is its own warning, meaning the mode
list may be incomplete. It fires when more than 5% of the draws are raw-z
rejected AND at least 80% of the rejected draws are out of range ONLY in
one or two GEOMETRIC parameters: the trajectory, the lens, or the physical
parameters that t_E, pi_E and rho derive from. Noise and flux parameters
do not count. The shape it catches is 173, where 28% of the draws are
rejected, all in `source.t_0_raw`, and come from 10 of 36 chains sitting
~25,000 nats below the bulk: a coherent displaced basin, not noise.
identify_modes reports its rejections only by reason, so `01_evidence.py`
(`rawz_info`) recomputes the robust z of the -1 draws the same way the
mode pass does. When the flag fires it REPLACES the generic
invalid-fraction warning, which is kept for diffuse, multi-parameter
rejection. 016's 2.8% (one chain, all bound pileup) does not fire. To
patch records written before this rule existed, run
`01_evidence.py --update-rawz`, which skips the bridge.

**Warnings** cover: unconverged (Rhat/ESS against the config's own
thresholds), stuck chains (> 10% of chains with a second-half mean lp more
than 20 nats below the best chain, as in `diag30.py`), lp Rhat > 1.1, more
than 5% invalid draws, bridge re2 > 0.1, the VARIABILITY and FORGIVEN
outlier kinds, AT_CAP with that band's out_scale Rhat beside it, and an
out_scale Rhat > 1.1.

Stuck chains are a run-level number. In a multi-seed run, chains parked
in a worse seed's basin count as stuck by construction.

## The submission files

Field names follow `microlens_submit` 0.17.9 (`models/solution.py`,
`models/event.py`, `validate_parameters.py`), read from the wheel, which
is not installed anywhere:

    submission/submission.json                 team_name / tier BLANK for the PI
    submission/events/RMDC26_000094/event.json {"event_id": ...}
    submission/events/RMDC26_000094/solutions/RMDC26_000094_1S2L_m1.json
    submission/events/RMDC26_000094/solutions/RMDC26_000094_1S2L_m1.md   (notes_path)

Each solution file carries:

- `model_type`: 1S1L or 1S2L.
- `parameters`:
  - `t0`, `u0`, `tE`;
  - `s`, `q` and `alpha` for 1S2L;
  - `rho` and `piEN`/`piEE`;
  - `F0_S/F0_B`, `F1_S/F1_B` and `F2_S/F2_B`, with F0 = F146, F1 = F087
    and F2 = F213.
- `parameter_uncertainties`: ONE number per parameter, max(minus, plus),
  the larger 68% half-width. The package's validator rejects an
  asymmetric `[lower, upper]` pair with lower > upper, which a skewed
  posterior gives about half the time. The exact interval is tabulated in
  the notes.
- `higher_order_effects`: `finite-source` and `parallax`. `t_ref` is
  written both as the solution field and inside `parameters`, where the
  validator looks for it. Every RMDC26
  model is physical (masses, distances, proper motions), so `pi_E` and
  `rho` are always derived. When parallax is included, `t_ref` is the
  fit's `t0_par`.
- `bands`: `["0", "1", "2"]`.
- `log_likelihood`: the hogg-mixture data log-likelihood at the mode's
  highest-lp draw. The Gaussian value is in the notes.
- `n_data_points`.
- `relative_probability` and `is_active`.
- `limb_darkening_coeffs`: all zero.

The notes Markdown carries the flags, lnZ, the group, the architecture
probabilities and the conventions paragraph below.
microlens-submit has no inline notes field; notes live in the file that
`notes_path` names, at the canonical location the package itself uses.

### Conventions (PI rulings, written into every solution's notes)

- **`t0`** is in BJD_TDB, the challenge's `bjd` column as given, with no
  offset subtracted.
- **`alpha`** is in RADIANS. It is our fitted binary-axis angle with NO
  conversion: the angle from the primary-to-companion axis to the LENS
  motion relative to the source, counter-clockwise (North through East).
  This is identical to MulensModel and to Skowron+2011 as written
  (`conventions.md` C15 and C18). The trace stores degrees, so the
  conversion goes through `Parameter.to_internal`, whose internal unit for
  `lens.alpha` is radians (asserted).
- **`piEN`/`piEE`** are the geocentric components in the
  `conventions.md` frame (C8-C10), anchored at `t0_par`.
- **`u0 > 0`** means the lens passes the source on its right.
- **`q`** is m_companion / m_primary. A `q > 1` is left as fitted and is
  NOT relabelled (C19 would also shift alpha by 180 degrees).
- **Fluxes** are `F = 10**(-0.4 m)` on the challenge magnitudes, which
  are taken to be AB, with zero point 0. This is how the fit itself
  converts them.
- **Limb darkening** is zero, and 2S1L is out of scope.

## Open questions for the PI

- **`alpha`'s definition.** microlens-submit's own description of `alpha`
  is "Angle of source trajectory relative to binary axis". Per
  `conventions.md` C21, measuring to the SOURCE's direction of travel is
  180 degrees from ours. We submit ours unconverted, as ruled. The manual
  also says radians while its example CSV uses `45.0`.
- **`t0`'s time system.** The package's `PARAMETER_PROPERTIES` gives
  `t0` in "HJD", the manual gives BJD, and the data's `bjd` column is
  BJD. We submit the data's own time system.
- **Flux unit.** "AB" fixes the magnitude system but not the zero point.
  We give `10**(-0.4 m_AB)`, which is 1 at m = 0. If the organizers
  expect, for example, a zero point of 25 or Jy, it is a single scale
  factor.
- **No zero-LD PSPL fit exists for the 2L1S events.** Their probabilities
  are therefore conditional on 1S2L. A PSPL arm per anomaly candidate
  would make the architecture comparison real.
- **Asymmetric errors.** We use symmetric max-half-widths because of the
  validator's lower <= upper check (see above). If the organizers read
  `[lower, upper]` as offsets, the check is a validator bug worth
  reporting; the asymmetric values are in every solution's notes.
- **Team name, tier and hardware info** in `submission.json` are blank.

## Results on the 76 events finished by 2026-10-21

The outputs are under `/n/holylabs/jjohnson_lab/Lab/rmdc26_report/`:
`report/campaign_summary.{txt,csv}`, `report/<NNN>_report.txt` and
`submission/`. The run covered 74 events with solutions, 128 solutions in
all. 022 and 187 were skipped as stale; 015 was processed on its
finished re-fit.

- **lp check: all 74 fits PASS** (max abs(dlp) <= 3.3e-7 nat), on master.
- **1S1L fits: all 39 bridges accepted** (re2 small, errors 0.1-0.5 nat).
- **1S2L fits: 70 of 83 mode bridges refused.** The counts are 46
  re2-too-large, 18 not-converged and 6 unsupported-proposal-mass. Only 9
  of 35 binary events have an accepted mode: 006, 015, 027, 036, 043,
  052, 053, 103 and 114. The other 26 binary events are INDICATIVE. The
  cause is the posteriors, not the bridge size: most binary runs are NOT
  EQUILIBRATED and lp Rhat > 1.1. Many modes are one or two chains
  parked in a seed's basin ("mode held by only N chain(s)"), and their
  draws are not a Gaussian-coverable population.
- **No event has both architectures**, so every probability is
  conditional on the one architecture fitted. Each 1S1L event is
  P(1S1L) = 1, and each binary event is P(1S2L) = 1 split over its modes.
- **1S1L UNRELIABLE (ANOMALY at the event):** 042, 063, 079, 081, 127,
  131, 160 and 166. These are single lenses whose outlier mixture absorbs
  points at the event, which makes them candidates for a missed companion.
- **err_scale out of range:** 016 (F213 ~14 in the top mode), 163 (F087
  2.8, F213 0.62), 173 (F213 8.1, F087 1.67) and 177 (F213 0.077).
- **POSSIBLE_UNLISTED_MODE:** 173 (`source.t_0_raw`, 25.3% rejected) and
  176 (`lens.log_s_raw` + `planet.log_q_raw`, 15.4%).
- **Runtime of stage 1 on 32 test-partition cores:** median 53 s per PSPL
  fit (max 63 s), and median 373 s per binary fit. The cost grows with the
  mode count: 163, with 4 modes on 49k epochs, took 5746 s, and 016, with
  8 modes, about as long. Stage 2 takes seconds.

## Validation against the package

`microlens_submit` 0.17.9 (the wheel, unpacked, not installed; pydantic
is absent from the conda env) was used to check the 2026-10-21
submission of 76 events and 128 solutions. Its pure validators
(`check_solution_completeness`, `validate_parameter_uncertainties`,
`validate_solution_consistency`) were loaded straight from
`validate_parameters.py`, and the Solution field names and the
`higher_order_effects` vocabulary were checked against
`models/solution.py`. Every event's active probabilities sum to 1. What
remains are the package's advisory "uncertainty very large/small"
warnings: for example, t0 is known to better than 0.1% of a BJD, and piE
and rho are often prior-dominated.

## The 2026-10 submission: the campaign MANIFEST (SED-tied fits)

The submitted fits are now the campaign's SED-tied runs under
`RMDC26_run/sed/<arm>/<NNN>/` (`out/` or `out_long/`, configs `run.yaml`,
`run_long.yaml`, `run_long2.yaml`), sampled on `wt/master1021` (b1b41d8e).
What is submitted, and in which tier, is decided by the campaign agent.
This tool reads those decisions and never hardcodes an event list.

### Regenerate (one command)

The campaign side runs first, in this order:

    python RMDC26_run/sed/verdict.py          # verdict.out, tier_full.txt, tier_scored.txt
    python RMDC26_run/sed/make_manifest.py    # SUBMISSION_MANIFEST.txt
    python RMDC26_run/sed/make_notes.py       # sed/notes/<NNN>.md

Then, from the root of this worktree:

    sbatch examples/RMDC26/run_submission.sbatch

It runs on the test partition as ONE job with internal parallelism
(96 cores; `WORKERS` fits at a time on disjoint core slices, default 6).
Everything goes under `$RMDC26_REPORT`, which defaults to
`/n/holylabs/jjohnson_lab/Lab/rmdc26_report/submission_v2`. The job:

1. `01_evidence.py --manifest`: the stage-1 record of every SUBMITTED
   fit. A record is reused only while its trace's (mtime, size) and the
   evaluating tree's commit are unchanged, so a rerun recomputes only the
   events whose trace changed. The resolved list goes to
   `evidence/manifest_resolved.json`.
2. `lpcheck.py` with THIS tree (master) on every submitted fit, into
   `lpcheck_master/`.
3. `03_submission.py`: rebuilds `submission/` from scratch, so an event
   that leaves the tier lists also leaves the submission. Then it runs
   `04_validate.py` and writes `report/submission_summary.{txt,csv,json}`
   and `report/validation.json`.

`--only 13,68` restricts stage 1 to some events, for a trial.

### What is submitted (`rmdc26.select_submitted`, `prefer_out_long`)

- **Tier filter.** `tier_full.txt` and `tier_scored.txt` are read on
  every run and are authoritative. FULL submits the required fields plus
  `physical_parameters`. SCORED submits the required fields only. Every
  other event is "not ok" and is not submitted. An event in the manifest
  but in neither list, an event in both lists, an event listed but missing
  from the manifest, and a NO RESULTS block are all reported as problems
  in the summary.
- **out_long/ is always preferred** when it holds a finished run (trace
  and results.csv) that is not stale, whatever the manifest names. A long
  rerun still in flight is not usable, and out/ is used with a note. That
  covers a whitening newer than the trace, or no trace yet.
- **The config is the one that wrote the trace** (`pick_config`): the
  newest `run*.yaml` whose `prefix:` is that run directory and that
  predates the trace. `make_manifest.py` always names `run_long.yaml` for
  out_long/, but the 70-event `long2` rerun writes out_long/ from
  `run_long2.yaml`.

### Which tree evaluates logp

The bridge integrates whatever logp the reporting tree compiles. Master
differs from the sampling tree by exactly one model change: #478, which
places the observer at the instant it received the light. That moves lp
by ~0.01-0.02 nat, against a 0.01 nat tolerance. Stage 1 therefore runs on
the SAMPLING tree (`EVIDENCE_TREE`, default `wt/master1021`), where the
50-draw lp check reproduces the sampler. Step 2 is the separate check of
master against every stored lp, as before. Its numbers are in the summary.

### Evidence and probabilities

These are unchanged from stage 2 above (`02_report.build_event`):

- valid draws only and 20000 proposal draws;
- the +/-2 nat floor;
- equal shares inside indistinguishable groups;
- refused bridges reported as INDICATIVE only.

`relative_probability` is normalized over the event's own submitted
solutions, which all share one architecture.

### Architecture (PI-approved; decided by Cannon Dev as owner)

Each event is fitted with one architecture, chosen by triage. The model
type comes from the fit itself (a `planet:` block, and s and q in the
posterior; cross-checked), never from the tier.

- **1S2L:** the rule uses the campaign's blind (s, q) grid,
  `dchi2_np = chi2(no-planet row) - chi2(best node)`. It is read from
  `grid/dchi2_noplanet.json`, which uses the same definition as
  `grid/gridcheck26.py`, with the no-planet row at log_q = -5.5.
  - If `dchi2_np > 50`, the note reads "binary lens favoured by
    dchi2_np = X on the blind grid" and P(1S1L) ~ 0.
  - Otherwise, or when there is no grid, the event is
    ARCHITECTURE_UNCERTAIN. When the posterior nevertheless pins q (68%
    half-width < 25% of the median), the tension is spelled out. 105 is
    the case: dchi2 = 8, with q pinned to 10%.
  - Event-specific findings are in `rmdc26.ARCH_EVENT_NOTES`.
  - The campaign's dchi2 is bimodal, with nothing between 10 and 100, so
    the exact cut moves nothing.
- **1S1L:** single lenses were not gridded. The outlier audit gives the
  verdict: an ANOMALY flag means a possible missed companion. A FULL
  single lens's status rests on t_E, rho and Rhat only.

### physical_parameters (FULL only; `rmdc26.physical_draws`)

The values are computed PER DRAW within each mode, then summarized as the
median and max(minus, plus). The keys are restricted to
`validate_parameters.PHYSICAL_PARAMETERS`:

| key | definition |
|---|---|
| `thetaE` | `mulensevent.theta_E` (mas) |
| `M1` | `star.mass[lens]` (Msun) |
| `Mtot` | `mulensevent.mlens_total` (1S2L), or M1 (1S1L) |
| `M2` | Mtot - M1, per draw (1S2L only; no planet-mass unit involved) |
| `D_L`, `D_S` | `star.distance` / 1000 (kpc) |
| `piE` | hypot(pi_E_N, pi_E_E), per draw |
| `mu_rel` | `mulensevent.mu_rel_mag` (heliocentric, mas/yr) |

### Notes (`notes_path`)

Each event has one notes file, `events/<id>/notes.md`, and all of the
event's solutions point to it. It is the campaign's generated note
(`sed/notes/<NNN>.md`) copied VERBATIM, followed by ONE appended,
machine-generated section, "## Evidence, flags and conventions (exozippy
report stage)". That section holds:

- the per-mode lnZ, bridge error and total error with the floor;
- the bridge status (a refused bridge is "indicative only");
- relative_probability;
- the reliability flags;
- the architecture statement;
- the conventions.

The alpha sentence is the ruled one (`rmdc26.ALPHA_SENTENCE`): degrees to
radians, no other transformation, wrapped to [0, 2 pi) and never negated.
`alpha_submit` does this, and `01_evidence.mode_summary` asserts that the
tree's `to_internal` agrees with it. Every written t0 must be a full BJD
(> 2.4e6); `check_submitted_values` raises otherwise. The extinction-law
caveat is added only when max Rhat > 1.3, read from the verdict row.
Converged tied fits recover the stated extinction (29/29).

### submission.json and validation

`team_name` is the placeholder `TEAM_NAME_TBD`, for the PI to replace.
`tier` is `beginner`. `hardware_info` names the cluster.

`04_validate.py` loads the tree with microlens-submit 0.17.9's OWN
`load()` and models, from the unpacked wheel. pydantic and psutil are
installed into `$RMDC26_REPORT/deps` by the sbatch. It then runs:

- `Submission.run_validation()`;
- `Solution.run_validation()` on every inactive solution as well;
- any warning the models raise at construction, which it counts as an
  ERROR;
- the physical-parameter key and consistency checks;
- the t0/alpha guards;
- an alpha and t0 ROUND TRIP against each fit's results.csv: radians
  back to degrees, with no offset.

### Results, 2026-10-24 (manifest of 25 FULL + 7 SCORED)

The outputs are under `/n/holylabs/jjohnson_lab/Lab/rmdc26_report/submission_v2/`.
The run took 48 min on 96 test-partition cores: stage 1 45 min, 5-10 min
per fit with 16 cores each, and the master lp check 3 min.

- **Submitted:** 32 events and 38 solutions (32 active). FULL is 25
  events with 28 solutions; SCORED is 7 events with 10 solutions. 29
  events are 1S2L and 3 are 1S1L (068, 083, 182). The manifest raised no
  problems.
- **lp reproduction:** the sampling tree reproduces every fit to
  <= 4.7e-10 nat. Master fails 15 of 32 fits at the 0.01 nat tolerance
  (median 0.009, max 0.18 nat on 162). Master's only model change since
  b1b41d8e is #478.
- **Bridges:** 30 of 38 mode bridges are refused, 19 for re2-too-large
  and 11 for not-converged. That makes 24 events INDICATIVE. With 40-odd
  dimensions, the SED-tied posteriors are poorly covered by the Gaussian
  proposal at 20000 draws. Single-mode events report
  relative_probability 1 regardless.
- **Architecture:** 28 of 29 binaries have dchi2_np > 50. They range from
  179 (110) to 1.0e6 (096). 105 is ARCHITECTURE_UNCERTAIN (dchi2_np =
  8.0, q pinned to 10%). None of the three single lenses has an ANOMALY.
- **Extinction caveat (max Rhat > 1.3):** 008, 030, 051, 053, 106, 138
  and 182.
- **Validation:** 0 errors. The 19 warnings are the package's advisory
  "uncertainty very small/large" warnings and the TEAM_NAME_TBD
  placeholder.

## Files

| file | what |
|---|---|
| `rmdc26.py` | paths, thresholds and every decision (pure functions) |
| `lpcheck.py`, `lpcheck.sbatch` | the lp-reproduction check |
| `01_evidence.py`, `run_evidence.sbatch` | stage 1 (`--manifest` for the SED-tied submission) |
| `02_report.py` | stage 2 (legacy events/ layout) |
| `03_submission.py` | the manifest submission tree and its summary |
| `04_validate.py` | microlens-submit's own validation of a tree |
| `run_submission.sbatch` | the one regenerate command |
| `tests/test_rmdc26_report.py` | the stage 2 rules and the writer, without a sampler |
