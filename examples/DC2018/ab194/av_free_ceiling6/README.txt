av_free, RUN ON THE OLD GRID WHOSE Av AXIS STOPPED AT 6.0 MAG.  Archived
2026-09-17 so a relaunch on the extended axis cannot overwrite it, because
this run IS the measurement that the ceiling was binding.

  star.Source.av = 5.26 +0.33 -0.40     (bound: [0, 6])

The colour-anchored truth for this event is 9.01 (see code_review_20260824.txt,
7.7.3, and conventions.md C29), so this posterior is not a measurement of the
extinction -- it is the grid running out, 2.3 sigma from a hard bound it
cannot cross, with no warning printed.  That missing warning is the open
sub-item of 2.9.16.

Its two sibling arms never started at all: av_clump (prior 8.61) and av_band
(prior 11.65) were both REFUSED at startup for a start value outside [0, 6],
which is the bound guard behaving exactly as designed -- see
dc18_ab194.15450902.11.log and .12.log.  av_true died earlier still, on a
0-byte config (fixed, and dc18_set_cores.py now says so in one line instead
of an AttributeError traceback).

The configs are copied here for provenance; their internal paths still point
at ../av_free, so do NOT rerun from this directory.
