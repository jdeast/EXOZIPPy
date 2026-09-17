THE v7 SWEEP, RUN WITH av = A_W149 AND A GRID THAT STOPPED AT Av = 6.
Archived 2026-09-17 before the re-run with corrected priors.  Configs are
COPIED here (the three per event); everything else was MOVED, so this is the
only copy of these traces and result tables.

WHY IT WAS RE-RUN.  Two defects that masked each other (7.7.3):
  * dc18_sweep_config.py's AV_COLS wrote the clump's A_W149 into
    `star.Source.av`.  A_W149 is roughly a FIFTH of A_V, so every event ran
    with a prior ~5x too small.
  * The BC grid's Av axis stopped at 6.0 mag and sed.py makes the grid
    extents the parameter's exact support, so the CORRECT prior would have
    been refused at startup on three of these six events.  The wrong one sat
    comfortably inside.

The corrected, colour-anchored priors (and the widened zeropoint that now
absorbs the C29 convention residual) are:

    event   was                   now                    zeropoint sigma
    008     0.66 +/- 0.01         2.26 +/- 0.14          0.07  (was 0.02)
    062     2.20 +/- 0.20         7.26 +/- 0.79          0.41
    128     1.94 +/- 0.26         6.44 +/- 0.94          0.34
    152     1.10 +/- 0.02         3.70 +/- 0.23          0.15
    194     2.75 +/- 0.33         9.01 +/- 1.20          0.58
    223     1.64 +/- 0.15         5.46 +/- 0.61          0.26

WHAT THIS ARCHIVE IS STILL GOOD FOR, and it is not nothing:
  * It is the scored baseline the truth table was built from -- 152 (5/7) and
    194 (6/7) self-reported; 008 (2/7), 062 (2/7), 128 (1/7) and 223 (1/7)
    salvaged from traces after h_rss kills.  Any claim that the corrected
    priors improved or degraded recovery is measured AGAINST THIS.
  * Every one of these runs used the SAME sampler configuration as the
    re-run: ptde_async, 64 cores, n_temps auto, T_max 200, tune 5000,
    draws 50000, nthin 1, eval_timeout 10, store_hot_chains true.

    BUT THE PRIORS ARE **NOT** THE ONLY DIFFERENCE, and pretending otherwise
    would make the comparison dishonest.  Regenerating the configs also
    picked up a change that postdates them: `err_scale` is now bounded to
    [0.5, 2.0] on both instruments, where these archived runs used
    defaults.yaml's [0.01, 100].  That bound is JDE's call of 2026-09-15
    (review 8.2.2) -- these are simulated curves with honest error bars, so
    err_scale is a check and not a fit -- and it was added because on event
    226 a point-lens basin inflated both bands 300-460x and turned the
    anomaly into noise.  Re-running WITHOUT it would deliberately reinstate
    a known defect, so it stays.

    SO THE RE-RUN CHANGES TWO THINGS: the av/zeropoint priors above, and the
    err_scale bound.  If a recovery changes, that is the pair of candidates,
    and err_scale's posterior (or its near-bound warning) says immediately
    whether the second one bit on that event.
  * Four of the six died to the h_rss accounting artefact and were scored
    from their traces.  That is fixed by the queue change (lThM.q +
    use_himem), so the re-run should self-report all six.  If it does not,
    the queue is not the whole story after all.

DO NOT RERUN FROM THIS DIRECTORY.  The copied configs' internal paths
(prefix, parameter_file, sed.file) still point at ../sweep.
