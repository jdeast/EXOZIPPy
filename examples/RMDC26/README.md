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

- the lp was not reproduced;
- the bridge was refused;
- the run was NOT EQUILIBRATED, according to the wrap-up log or to
  master's check on the trimmed lp;
- any err_scale lies outside [0.8, 1.25]. The noise model, not the lens
  model, is then carrying the fit. The known cases are 016 F213 21.0,
  173 F213 8.1 and 163 F087 3.2;
- there is an ANOMALY outlier flag: the mixture is absorbing points AT the
  event, which means this model is missing a signal;
- the event is in `FORCED_UNRELIABLE` (016, 163, 173, set by the campaign
  agent 2026-10-21). 173 also flags VARIABILITY + FORGIVEN + AT_CAP in all
  three bands, and a third of its chains sit in a worse t_0 mode.

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
- `parameter_uncertainties`: `[lower, upper]` 68% offsets.
- `higher_order_effects`: `finite-source` and `parallax`. Every RMDC26
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
- **Team name, tier and hardware info** in `submission.json` are blank.

## Files

| file | what |
|---|---|
| `rmdc26.py` | paths, thresholds and every decision (pure functions) |
| `lpcheck.py`, `lpcheck.sbatch` | the lp-reproduction check |
| `01_evidence.py`, `run_evidence.sbatch` | stage 1 |
| `02_report.py` | stage 2 |
| `tests/test_rmdc26_report.py` | the stage 2 rules and the writer, without a sampler |
