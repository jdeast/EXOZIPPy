# DC2018: the 2018 Roman (WFIRST) Data Challenge, end to end

Fits every light curve of the challenge's 44-event sample with the full
pipeline, one cluster job per event:

1. **MMEXOFAST** runs on both bands (`renormalize_errors=True`,
   `binary_lens`), cached as `events/<NNN>/DC2018_<NNN>_mmexofast.json`.
   It always sees both bands even when EXOZIPPy later fits only one: the
   15-min W149 cadence is what localizes the anomaly. Its solutions seed
   the fit (multi-seed sampling), its bad-data mask (`excluded_points`)
   drops the flagged points via the generic instrument `mask:` feature,
   and its error factors (`errfacs`) seed each instrument's `err_scale`.
   By default the run stops after the fast binary-parameter ESTIMATION:
   EXOZIPPy's tempered multi-seed PTDE is itself the polish, and with
   the renormalization peak-protection fix the raw estimator seeds land
   in the right basin (on event 128: s = 0.977, rho = 0.0075 vs the
   polished 0.979/0.0054 -- the once-alarming "rho collapses to ~1e-6"
   was an artifact of outlier rejection eating the finite-source peak).
   Pass `--mmx-emcee` (job: `EXTRA="--mmx-emcee"`) to turn MMEXOFAST's
   hours-long emcee polish back on. (This explicit step is optional in
   general: a config whose params file lacks microlensing start values
   triggers the same MMEXOFAST run automatically inside EXOZIPPy's
   data-driven-hints layer.)
2. **EXOZIPPy** samples the 2L1S system (PTDE, EXOFASTv2-parity settings)
   and writes the usual artifacts under `events/<NNN>/fitresults/`.
   Every light curve fits with `likelihood: hogg` -- the marginalized
   inlier/outlier mixture -- which supersedes MMEXOFAST's hard bad-data
   mask: `excluded_points` are NOT propagated for hogg files (MMEXOFAST
   still rejects internally to protect its own anomaly search, and its
   errfacs still seed `err_scale`), so every point stays in the fit,
   junk lands in the wide background component instead of dragging the
   solution, and per-point posterior outlier probabilities are available
   afterwards via `Instrument.outlier_prob_at_data` -- at the cost of
   two extra parameters per curve (`out_frac`, `out_scale`).
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
- The `mmexofast` package in the environment (git-only, so it is not part
  of a plain install):

      poetry install --with microlensing            # dev machine
      # or, in the cluster's conda env:
      pip install git+https://github.com/jenniferyee/MMEXOFAST.git
      pip install git+https://github.com/jenniferyee/sfit_minimizer.git emcee

  NOTE: the automatic mask/err_scale consumption needs an MMEXOFAST recent
  enough to write `excluded_points`/`jd_offset` in its exozippy-init JSON
  (2026-07 or later). Older versions still work -- seeds only, no masking.
  This workflow also relies on three 2026-07-31 MMEXOFAST performance/
  correctness patches (fast outlier rejection, emcee pool support, and the
  renormalize-after-anomaly-search reorder so the outlier rejection cannot
  eat the planetary anomaly); until they are merged upstream, install
  MMEXOFAST from the patched checkout.

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

then run them all (44 tasks, one per line of `events.txt`):

    qsub -t 1-44 dc2018.job

Knobs: `qsub -v EVENT=128,BANDS=Z087,EXTRA="--quick" dc2018.job`. The
sampler core count follows the job's `$NSLOTS`, so it always matches the
`-pe mthread` grant.

After the jobs finish:

    python collect_results.py            # -> dc2018_summary.csv + stdout table

## Caveats

- **alpha**: the answer key's alpha convention differs from
  EXOZIPPy/MMEXOFAST's (center-of-mass origin). The comparison maps truth
  through a sign/offset search against the fitted value and records the
  choice in the `note` column -- treat alpha pulls as indicative.
- **u_0** is compared in absolute value (the truth carries a
  trajectory-side sign the fits do not).
- Event 1 is a cataclysmic variable, not a lensing event; expect the 2L1S
  fit to fail or diverge on it (the collector will show it as such).
- Each event directory gets a dumped `DC2018_<NNN>.yaml` +
  `DC2018_<NNN>.params.yaml`, so any event can be re-run or debugged with
  the plain CLI: `cd events/<NNN> && exozippy DC2018_<NNN>.yaml`.
