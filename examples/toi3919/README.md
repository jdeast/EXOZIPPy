# TOI-3919: a transiting giant planet plus a linear RV trend

TOI-3919 b is a 7.43-day, eccentric (e ~ 0.26) giant planet observed in four
TESS sectors (16, 23, 49 and 50) and four ground-based transits (CALOU-0m4
in R, KeplerCam and LCO-McD-1m0 in Sloan i', WCO in Sloan r'), with 21 TRES
radial velocities.  The RVs also drift: the EXOFASTv2 fit this example was
converted from used `/fitslope` and measured a slope of -0.79 +0.31/-0.30
m/s/day.

**This is the first shipped example with an RV trend**, and the trend is
written the way EXOZIPPy models one: as an ORBIT.

```yaml
orbit:
  - name: "b"
    primary: ["A"]
    companion: ["b"]
  - name: "trend"
    type: linear
    primary: ["A"]
```

A `type: linear` orbit is a Taylor orbit -- the reflex of a companion too
long-period for the data to resolve, written as the derivative of the star's
radial velocity about a reference epoch, `orbit.trend.gammadot` (m/s/day).
It has no period, eccentricity, mass or K, and none are reported; its
companion is unseen, so none is named.  What the slope DOES say about that
companion is reported as a bound: `orbit.trend.mc_over_r2_min` = |gammadot|/G,
the companion's mass over its squared separation (1 M_J at 1 AU pulls the
star at 0.489 m/s/day), ~1.6 M_J/AU^2 for this slope -- e.g. at least
~1.6 M_J at 1 AU, or ~6.4 M_J at 2 AU.  `type: quadratic` adds the curvature
`gammaddot` (m/s/day^2, a second derivative -- twice EXOFASTv2's QUAD).  The
same orbit type carries a microlensing lens's linear orbital motion
(`examples/ob09020/ob09020_linear.yaml`).  The design, the epoch rule and
what is not implemented yet are in `src/exozippy/components/orbit/orbit.md`,
"Taylor orbits".

The reference epoch is not set, so it defaults to EXOFASTv2's RVEPOCH: the
midpoint of the RVs, 2459687.926215 BJD_TDB (logged at startup, and stated
in the results table's note on `gammadot`).

## How it was made

```bash
cd examples/toi3919
exozippy-exofast2exozippy ~/modeling/toi3919/fittoi3919.pro \
    --priorfile ~/modeling/toi3919/toi3919.priors.final -o .
```

then, by hand: the `prefix:` set to `fitresults/toi3919` (the driver's own
`./fitresults_final_4/...` was kept verbatim), the R band's filter written
out as `Generic/Cousins.R`, and the header comments at the top of
`toi3919.yaml`.

Two things about the EXOFASTv2 inputs are worth knowing, because they are
why the converter grew the features it did in the same pull request:

- **The driver's `priorfile='toi3919.priors.3'` does not exist** in the
  modeling directory; the run's own `toi3919.priors.final` does.
  `--priorfile` replaces the driver's.
- **`toi3919.priors.final` was written by a run on FEWER data files** than
  the driver now globs (TESS plus KeplerCam only).  EXOFASTv2 numbers
  transits by sorted filename and bands by sorted name, so its `variance_4`
  (KeplerCam there) is the TESS S50 file here, and its `u1_0` (i' there) is
  the R band here.  The converter now reads EXOFASTv2's own section headers
  (`# KeplerCam UT 2022-04-28 (i')`, `# TESS`) and re-points each prior at
  the instance the header names, saying so in its notes.

What did not translate (listed at the top of `toi3919.yaml`): the Claret
limb-darkening priors (the LD table prior is not implemented, so the limb
darkening is constrained by the transits alone), and `fitdilute=['TESS']`
with its `dilute` prior (a 0 +/- 0.00047 dilution, i.e. essentially none).
The KeplerCam detrending coefficients in the priors file were start values
only; the extra columns of every ground-based file are still detrended
additively, as in EXOFASTv2.

## EXOFASTv2 values to compare against

EXOFASTv2 from `TOI-3919.MIST.SED.median.csv`; EXOZIPPy from a SHORT check
fit (numpyro, 4 chains x 500 tune + 500 draws, 2026-10-07; max Rhat 1.011,
min ESS 482 -- below the convergence thresholds, so treat the last digit
loosely).  Median, +/- 1 sigma:

| parameter | EXOFASTv2 | EXOZIPPy (check fit) |
|---|---|---|
| RV slope (m/s/day) | -0.79 +0.31/-0.30 | -0.80 +/- 0.31 |
| P (days) | 7.433234 +/- 0.000014 | 7.433232 +/- 0.000014 |
| T_C (BJD_TDB) | 2458954.3740 +/- 0.0013 | 2458954.3744 +0.0013/-0.0014 |
| e | 0.259 +0.033/-0.036 | 0.264 +0.033/-0.035 |
| K (m/s) | 367 +18/-17 | 369 +18/-17 |
| M_P (M_J) | 3.88 +/- 0.23 | 3.83 +0.23/-0.24 |
| R_P (R_J) | 1.099 +0.052/-0.050 | 1.113 +0.050/-0.045 |
| M_* (M_sun) | 1.208 +0.067/-0.070 | 1.185 +0.067/-0.076 |
| R_* (R_sun) | 1.319 +0.052/-0.048 | 1.325 +0.052/-0.046 |
