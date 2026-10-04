# Microlensing `alpha` conventions -- the 2026-10-02 measurement

This is the measurement record behind conventions.md **C18, C21 and C22**
(`src/exozippy/components/mulensing/conventions.md`). The rules themselves live
there; this file keeps the chi2 tables they were read from, so a reader can
check a claim without re-running anything. Landed in #371 (KMT-2019-BLG-1806
start sign) and #372 (C21/C22 and the DC2018 truth comparison).

Everything below is MEASURED (chi2 or `A(t)` against the shipped data), not read
off a docstring. MulensModel as installed in the project venv, VBMicrolensing
>= 5.6. The DC2018 scan is `scripts/dc18_alpha_convention.py`; the per-event
pairs it produced are copied into `tests/test_dc18_alpha_mapping.py`.

## Summary

0. **EXOZIPPy == MulensModel == MMEXOFAST.** Same convention for every
   trajectory parameter, measured end to end. No transformation.
1. **KMT-2019-BLG-1806** (`examples/KMT-2019-BLG-1806`): the start
   `alpha = +57.03` (`u_0 > 0`) shipped before #371 was wrong; the MMEXOFAST
   fit's own `-57.03` (`u_0 > 0`) is right as written. The negation cost
   Delta chi2 = 2625 on the 2441 shipped points (+539.5 nats of start logp).
2. **Zang et al. (2023)**, KMT-2019-BLG-1806: `alpha_EXZ = alpha_paper + 180`,
   `u_0` and `pi_E` kept.
3. **Jung et al. (2017)**, OGLE-2016-BLG-1003: `alpha_EXZ = 180 - alpha_paper`,
   `u_0` kept (equivalently `alpha_paper + 180` with `u_0` flipped -- no
   parallax, so the two are the same statement). The two KMTNet-era papers
   therefore differ from each other by the sense of rotation; there is no single
   "KMTNet convention" (C21).
4. **DC2018 answer key**: mappable, by an event-dependent rule (C22):

       alpha_EXZ  = alpha_key + 180 - theta_axis(t_0)      (u_0 kept)
       theta_axis = atan2(sin(phase) cos(inc), cos(phase))

   19/19 events with chi2 contrast >= 1000 match to a median 0.10 deg, max
   1.53 deg.

## 0. Same convention? (EXOZIPPy vs MulensModel at identical parameter values)

Build the shipped example, evaluate at the raw start the exact inputs the light
curve's Op receives and its `A(t)` on a 10,000-epoch grid (4000 over the season
plus 6000 over `t0_par +/- 3 d`, through the anomaly), with EXOZIPPy's own
ephemeris and Skowron deviations; give the same numbers to an independent
`MulensModel.Model` (same coordinates, `t_0_par` = the event's `t0_par`,
MulensModel's own annual parallax, VBBL, `set_limb_coeff_u`).

| case | backend | max abs(A_EXZ / A_MM - 1) |
|---|---|---|
| KMT-2019-BLG-1806 start: 2L1S + finite source + LD + parallax, `A_max = 54` | `vbm_direct` | 1.6e-4 (median 2.9e-6; the outliers are all in the anomaly, VBM accuracy 1e-3 vs MM default). Against MM at each of the 7 other `(u_0, alpha)` maps: 0.79 - 1.06 |
| ob140939 start: PSPL + annual parallax | symbolic | 4.9e-7 (vs MM at `-u_0`: 2.3e-3 -- parallax breaks the sign) |
| static 2L1S, 6 random draws, caustic crossings to `A = 145`, no parallax | `VBMDirectMagOp` vs MM VBBL, both accuracy 1e-5 | <= 1.7e-15; vs MM at `(-u_0, -alpha)`: <= 1.7e-15 (the exact mirror, Skowron A12); vs MM at `(u_0, -alpha)`: 0.09 - 0.87 |
| linear lens orbital motion, 3 draws, `dalpha_dt` 50-84 deg/yr | `VBMDirectMagOp(orbital_motion)` vs MM `ds_dt`/`dalpha_dt`/`t_0_kep` | <= 6.7e-16 |

The MMEXOFAST JSON for KMT-2019-BLG-1806, evaluated directly in MulensModel on
the raw files:

| solution | as written | `(u_0, alpha)` mirrored | `alpha` negated only |
|---|---|---|---|
| sol0, raw chi2 | 2352.8 | 2352.8 | 4978.1 |
| sol1, raw chi2 | 2351.9 | 2351.9 | 2718.2 |

So MMEXOFAST's convention is MulensModel's, which is EXOZIPPy's.

## 1-2. KMT-2019-BLG-1806 (Zang et al. 2023, AJ 165, 103, arXiv:2210.12344)

Zang+2023 Table 4, Outer `u_0 > 0`: `t_0 = 8715.453`, `u_0 = +0.0257`,
`t_E = 134.1`, `alpha = +2.151 rad = 123.24 deg`, `s = 1.0339`,
`log q = -4.717`, `pi_E = (-0.060, -0.057)`. `rho` is an upper limit only and
was evaluated at `5e-4` with no LD; `t_0_par = t_0` (not stated in the paper).

chi2 on the shipped 2441 KMTNet points, fluxes linear per file, raw errors.
"EXZ" is the shipped `VBMDirectMagOp` on EXOZIPPy's own data arrays; "MM" is
MulensModel on the raw files.

| mapping | `u_0` | `alpha` [deg] | chi2 EXZ | Delta | chi2 MM |
|---|---|---|---|---|---|
| 180 + alpha, u_0 kept | +0.0257 | 303.24 | 2316.6 | 0.0 | 2316.6 |
| 180 - alpha, u_0 flipped | -0.0257 | 56.76 | 2316.9 | 0.3 | 2316.9 |
| -alpha, u_0 kept | +0.0257 | 236.76 | 2446.0 | 129.5 | 2446.0 |
| alpha, u_0 flipped | -0.0257 | 123.24 | 2446.9 | 130.3 | 2446.8 |
| 180 + alpha, u_0 flipped | -0.0257 | 303.24 | 3158.6 | 842.1 | 3157.8 |
| 180 - alpha, u_0 kept | +0.0257 | 56.76 | 3250.3 | 933.8 | 3246.7 |
| -alpha, u_0 flipped | -0.0257 | 236.76 | 3615.4 | 1298.9 | 3613.3 |
| alpha, u_0 kept | +0.0257 | 123.24 | 3663.1 | 1346.6 | 3662.7 |

The Outer `u_0 < 0` and Inner `u_0 > 0` solutions give the same ranking. The
`(u_0, alpha) -> -(u_0, alpha)` mirror is exact without parallax and costs only
0.3-0.5 with Zang's small parallax (C23's ecliptic degeneracy), so this event
pins Zang's `alpha` only modulo that mirror; what it excludes, by 842-1347
chi2, are the two rules that do not pair with the mirror. Zang Outer
`u_0 > 0` -> 303.24 deg = -56.76 deg agrees with MMEXOFAST's independent
-57.03 deg, `u_0 > 0` in both.

Full-pipeline check on the shipped config and params, only
`lens.Companion.alpha` changed: start logp 44956.107 at `+57.03` vs 45495.619
at `-57.03` (+539.51, of which the light curve is +539.64).

## 3. OGLE-2016-BLG-1003 (Jung et al. 2017, ApJ 841, 75) -- 2S2L, no parallax

Published values as in `examples/ob161003`'s params file (`alpha = 48.243`).
Both sources' `u_0` flipped together; fluxes `(f_s1, f_s2, f_b)` linear per
file; 4300 points in 7 files.

| mapping | `alpha` [deg] | chi2 EXZ | Delta | chi2 MM |
|---|---|---|---|---|
| 180 + alpha, u_0 flipped | 228.243 | 7222.3 | 0.0 | 7222.3 |
| 180 - alpha, u_0 kept (shipped) | 131.757 | 7222.3 | 0.0 | 7222.3 |
| alpha, u_0 flipped | 48.243 | 24567.6 | 17345.3 | 24567.6 |
| -alpha, u_0 kept | 311.757 | 24567.6 | 17345.3 | 24567.6 |
| 180 + alpha, u_0 kept | 228.243 | 49419.9 | 42197.5 | 49419.9 |
| 180 - alpha, u_0 flipped | 131.757 | 49419.9 | 42197.5 | 49419.9 |
| alpha, u_0 kept | 48.243 | 68159.9 | 60937.5 | 68159.9 |
| -alpha, u_0 flipped | 311.757 | 68159.9 | 60937.5 | 68159.9 |

Under Zang's rule (`180 + alpha`, `u_0` kept) this event costs +42,197; under
Jung's rule Zang's event costs +842 or +934. Neither paper's rule may be
applied to the other's. `(-u_0, 180 - alpha)` is the +42,197 row, which is why
C21's earlier explanation of ob161003 ("the source-trajectory shift composed
with the opposite `u_0` branch") was wrong.

## 4. DC2018 (the 2018 Roman/WFIRST data challenge answer key)

`scripts/dc18_alpha_convention.py` on 36 events (the 30 of
`examples/DC2018/events.txt` plus 1, 8, 12, 32, 40, 128) finds the light
curve's own preferred `alpha` at the key's `(t_0, u_0, t_E, rho, s, q)`, fluxes
linear. Every GLOBAL offset tested against it (fit -/+ key, optionally minus
the position angle of Galactic North or of `mu_rel`) scatters (circular
`R <= 0.19`), because the key measures the source trajectory against the
planet orbit's **line of nodes**, not the binary axis, and the binary axis sits
at a different `theta_axis` from that line for every event. The node angle
itself is not needed: `alpha_key` and `theta_axis` share the node line, so it
cancels.

| ev | key alpha | theta_axis | predicted | fit alpha | fit - pred | chi2 contrast | P [yr] |
|---|---|---|---|---|---|---|---|
| 4 | 38.67 | -84.19 | 302.85 | 302.95 | +0.10 | 7968 | 686.81 |
| 8 | 316.98 | 3.30 | 133.68 | 133.50 | -0.18 | 362 | 26.06 |
| 12 | 274.15 | 75.68 | 18.47 | 18.70 | +0.23 | 5997 | 10.78 |
| 32 | 69.62 | 43.47 | 206.15 | 208.35 | +2.20 | 993 | 3.28 |
| 40 | 351.96 | 170.44 | 1.53 | 330.95 | -30.58 | 635 | 1.27 |
| 47 | 248.16 | 178.32 | 249.84 | 250.00 | +0.16 | 591 | 3.31 |
| 53 | 25.38 | 164.36 | 41.03 | 41.15 | +0.12 | 381 | 17.90 |
| 66 | 68.82 | 5.28 | 243.54 | 243.55 | +0.01 | 1258871 | 82.00 |
| 69 | 320.00 | -147.56 | 287.56 | 288.10 | +0.54 | 17500 | 13.50 |
| 74 | 109.71 | -9.65 | 299.36 | 299.30 | -0.06 | 3195 | 4.61 |
| 78 | 132.22 | 21.72 | 290.50 | 290.50 | +0.00 | 536851 | 39.65 |
| 92 | 149.59 | 38.18 | 291.41 | 291.35 | -0.06 | 206 | 12.93 |
| 99 | 178.32 | 176.90 | 181.42 | 181.80 | +0.38 | 2376 | 0.98 |
| 103 | 288.76 | 2.05 | 106.72 | 106.65 | -0.07 | 43752 | 2.98 |
| 128 | 348.36 | -141.33 | 309.68 | 308.15 | -1.53 | 325858 | 1.27 |
| 131 | 78.91 | 147.75 | 111.15 | 110.25 | -0.90 | 738 | 13.88 |
| 139 | 94.37 | 178.26 | 96.11 | 96.00 | -0.11 | 527 | 6.83 |
| 163 | 71.25 | -64.48 | 315.73 | 315.70 | -0.03 | 19472 | 31.11 |
| 193 | 29.33 | -37.66 | 246.99 | 247.00 | +0.01 | 11224 | 15.51 |
| 199 | 82.16 | -134.17 | 36.33 | 36.20 | -0.13 | 25630 | 13.34 |
| 208 | 117.95 | 3.03 | 294.92 | 299.10 | +4.18 | 358 | 1.51 |
| 214 | 88.48 | 117.25 | 151.23 | 150.65 | -0.58 | 534 | 38.14 |
| 217 | 254.92 | -16.42 | 91.34 | 91.30 | -0.04 | 37858 | 4.37 |
| 218 | 180.74 | 60.49 | 300.25 | 300.30 | +0.05 | 1027 | 8.33 |
| 223 | 300.31 | 54.01 | 66.30 | 64.95 | -1.35 | 15476 | 8.81 |
| 226 | 3.48 | 19.58 | 163.90 | 163.45 | -0.45 | 6523 | 148.19 |
| 227 | 80.89 | -4.56 | 265.45 | 265.75 | +0.30 | 779 | 22.81 |
| 250 | 51.39 | 4.05 | 227.34 | 227.70 | +0.36 | 2239 | 8.50 |
| 253 | 351.84 | -45.40 | 217.24 | 217.15 | -0.09 | 3721 | 38.37 |
| 258 | 259.54 | 108.98 | 330.56 | 330.45 | -0.11 | 5534 | 17.84 |
| 289 | 292.03 | -15.92 | 127.95 | 127.15 | -0.80 | 484 | 10.68 |

All angles in degrees. Events whose anomaly does not pin `alpha` (contrast < 20
chi2) scatter, as they must: 25 (-161.0), 107 (+41.2), 152 (-0.5), 194 (+65.4),
and event 1 (a cataclysmic variable, not a lens; -67.8).

- contrast >= 1000: n = 19, abs(fit - pred) median 0.10 deg, max 1.53 deg.
- contrast >= 100: n = 31, median 0.16 deg, max 30.58 deg (event 40).
- circular `R` of `(fit - key + theta_axis)` on the original 30 events: 0.908
  (all), 1.000 (the 17 with contrast >= 1000), mean 179.95 deg.

The residuals above ~1 deg are the shortest periods -- 40 (P = 1.27 yr), 208
(1.51), 32 (3.28), 128 (1.27). The simulator moves the lens (85 deg of orbital
phase over `+/- 3 t_E` on event 128), the key's `alpha` is the `t_0` geometry,
and a static fit finds the anomaly-epoch compromise. This is the same reason
lens-orbital-motion events are excluded from the static 2L1S sweep
(`examples/DC2018/dc18_common.py`, `SHORT_PERIOD_YR`).

What stays unidentifiable is only what is unidentifiable for every
no-parallax event: the `(u_0, alpha) -> -(u_0, alpha)` mirror (C23).
