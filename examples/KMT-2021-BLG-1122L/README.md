# KMT-2021-BLG-1122L: the triple-lens validation example

Han et al. 2023, A&A 672, A8 (arXiv:2302.05613): "KMT-2021-BLG-1122L: the
first microlensing triple stellar system".  3L1S: a 0.47 Msun primary with
two M-dwarf companions (q_2 = 0.53, q_3 = 0.24) both near the Einstein ring,
a finite source (rho = 2.5e-3) and a 14.7-day event.  This directory is the
triple-lens validation vehicle of review 7.6.4: the FIRST real 3-body lens
to go through the pipeline, so every untested claim about the N >= 3 path
(mulensing.md, "Three or more lens bodies"; reviews 8.6.13, 8.6.16) gets
exercised here.

## Data

KMTNet pySIS photometry of the three sites, I and V, from the KMTNet event
page (`KMT{A,C,S}14_{I,V}.pysis`).  `prepare_data.py` writes the `.dia`
files the config reads with `data_format: dia`: the NATIVE difference flux,
not the pySIS mag column.  That column satisfies
`mag = 28.0 - 2.5 log10(1584.9 - dflux)` at every site to 3e-5 mag: the
reference 1584.9 is 20.0 mag at zp 28, a fixed pipeline reference rather
than this star's baseline, so a magnitude file has a fictitious total flux
(review 2.2.22).  Each instrument instead states an explicit
`reference_mag` (I 16.0, V 19.0) above every epoch's flux; the reference
only shifts the blend, which nothing reads.  Rows with `fwhm <= 0` (pySIS's
failed-PSF flag) are dropped: they included points 1-9 mag above baseline
with small errors, one at 11.4 mag with sigma 0.16.

| file | site | band | points |
|---|---|---|---|
| n20210603.I.KMTC14.dia | CTIO | I | 1303 |
| n20210603.I.KMTS14.dia | SAAO | I | 663 |
| n20210603.I.KMTA14.dia | SSO | I | 385 |
| n20210603.V.KMTC14.dia | CTIO | V | 114 |
| n20210603.V.KMTS14.dia | SAAO | V | 50 |
| n20210603.V.KMTA14.dia | SSO | V | 26 |

Han+2023 modelled ~1435 pySIS I points (their cuts are not stated) and
used pyDIA only for the source colour; with 2351 I points we fit about
60% more, which is why our error scales sit near 1.5-2.3 rather than 1.

The V curves give the source colour; with the A_V prior (5.3 +/- 0.3 from
Han's A_I = 3.3) that fixes theta_*, and with it theta_E = theta_*/rho.
With I alone theta_* was free and the distances drifted to ~100 pc.

Coordinates (J2000) 17:35:51.10 -28:26:47.22, (l, b) = (-0.720, +2.076),
from the event page, matching Han Sect. 2; pinned on all four stars.

## Table of record

Table 2 of arXiv:2302.05613v1 (11 Feb 2023), "Lensing parameters of the
best-fit 3L1S solution", chi2/dof = 1399.8/1425.  The A&A version of record
(672, A8) could not be fetched from this machine (HTTP 403), so the 3.6.4
version-skew check against it is STILL OWED: compare the journal's Table 2
with the numbers in the params file before quoting any recovery.

| parameter | value | | parameter | value |
|---|---|---|---|---|
| t_0 (HJD') | 9370.226 +/- 0.061 | | s_2 | 1.386 +/- 0.027 |
| u_0 | 0.126 +/- 0.008 | | q_2 | 0.526 +/- 0.039 |
| t_E (d) | 14.74 +/- 0.93 | | alpha (rad) | 2.292 +/- 0.013 |
| rho (1e-3) | 2.50 +/- 0.21 | | s_3 | 1.601 +/- 0.043 |
| | | | q_3 | 0.241 +/- 0.027 |
| | | | psi (rad) | 1.383 +/- 0.010 |

Physical (Sect. 5): M_1 = 0.47 Msun, M_2 = 0.24, M_3 = 0.11, D_L = 6.57 kpc,
theta_E = 0.50 mas, mu_rel = 12.4 mas/yr, theta_* = 1.26 uas.

## Mapping the published solution onto EXOZIPPy's coordinates

Han's text does not state the sense of alpha or psi or the origin of
(t_0, u_0) (Sect. 3; Fig. 4 centres the coordinates on the M1-M2
"effective position").  The mapping was therefore decided by an OPTIMIZED
scan on the cleaned model (`scan_conventions_opt.py`): 48 candidates =
alpha rule {alpha + 180, 180 - alpha, alpha, -alpha} x u_0 sign x
alpha_3 = alpha_2 +/- psi x s normalization {s, s*k, s/k} with
k = sqrt(1 + q_2 + q_3).  Each candidate's geometry is pinned and every
other parameter Powell-optimized; the light-curve term decides.

    alpha_EXOZIPPy = -alpha_Han,   alpha_3 = alpha_2 - psi,   u_0 as published

(or its exact mirror: +alpha_Han, alpha_2 + psi, -u_0), with s, q, t_E
and rho as published.  Light-curve term 9424; the best other candidate
reaches 7659.  At the optimum: t_E 14.73 d, theta_E 0.491 mas,
rho 0.0025, D_L 6.57 kpc, D_S 8.68 kpc, masses 0.45 / 0.24 / 0.11 Msun --
Table 2 and Sect. 5 recovered.  t_0 sits at 2459374.59, +4.37 d from
Han's, the translation from the M1-M2 effective-position origin to
EXOZIPPy's centre of mass of all three bodies.

The earlier `check_start.py` scan picked 180 - alpha with +psi and -u_0.
It scored STARTING POINTS on outlier-laden magnitude files with a broken
flux bootstrap, and its winner was an artefact; every fit launched from it
(the first acceptance fit included) sat in the wrong solution, with
unphysical distances and error scales of 3-7.  `check_start.py` and
`check_mirror*.py` are kept as the record.

## What the triple path needed that a binary does not (review 8.6.13)

All three bit, exactly as 7.6.4 predicted, and each cost one scan iteration:

- **Companion slot 1 ignores `s:` and `alpha:`.**  The engine's
  `s <-> log_s` and `alpha <-> (xalpha, yalpha)` relations exist for
  companion slot 0 only, so `lens.LensC.s` / `lens.LensC.alpha` initvals
  were accepted (no warning) and silently dropped: LensC started at the
  defaults (s = 1, alpha = 0) and every scan combination gave the same
  total.  LensC is seeded through its sampled coordinates `log_s`,
  `xalpha`, `yalpha` instead.
- **Explicit per-body masses are REQUIRED** (documented): q_2, q_3 are
  derived from `star.{Lens,LensB,LensC}.mass`, set so the derived ratios
  reproduce Table 2.
- **A bare `mulensevent.t_E` initval loses to the galactic hints** (the
  8.6.15 recipe): the first scan started at t_E = 36.3 d (mu_rel 4.87
  mas/yr, the prior mean) with the published 14.74 silently dropped.
  Seeding the leaves (masses, distances, proper motions) gives
  theta_E = 0.50 mas and |mu_rel| = 12.39 mas/yr, hence t_E = 14.74 d.
- **rho is derived** from the source radius and theta_E; the source-flux
  seed alone gave 1.9e-3, so the radius is seeded (2.35 Rsun at 8.68 kpc,
  the paper's theta_* = 1.26 uas), giving the published 2.5e-3.

## Mirror check (`check_mirror.py`)

Without parallax the reflection (u_0, alpha_j) -> (-u_0, -alpha_j) for
EVERY companion is exact.  The direct Op reproduces this: the triple-lens
light curve and its mirror agree to 2.7e-5 in A (VBM's tolerance), the
binary to 3e-15, and so does MulensModel's binary used by the flux
bootstrap.  Yet the two BUILDS start 4.8e6 nats apart (`check_mirror.py`),
and the gap is entirely in the stage-1 seeding (`check_mirror4.py`,
`check_mirror5.py`: deterministic, sign-only, not order-dependent):

- **Review 2.6.33.**  The peak finder's chi2 returns inf unless u_0 > 0,
  and a HELD u_0 is passed as written, so with the u_0 = -0.4213 of the
  EARLIER (wrong) mapping every candidate was rejected ("found no PSPL
  solution") while +0.4213 fit.  That start was sane only because the
  finder failed on it.
- **Review 2.6.34.**  For three lens bodies the total-mass relation is not
  registered, so at stage 1 `probe_start` answers t_E / theta_E / rho
  "not derivable" even with every leaf seeded (pi_rel and mu_rel ARE
  solved).  The finder therefore refits t_E with a point lens on a
  caustic-crossing light curve and pushes 2.65 d (vs 14.74) at data rank;
  the engine back-solves mlens_total 0.027 Msun, theta_E 0.090 mas, rho
  0.0140, pi_E 0.41 from it for the stage-1 readers, and the flux
  bootstrap fits that to each file: negative blends, q_source 3.2 and 5.0
  clipped to the bound 2, zeropoints 2 mag off.  The stage-4 start is
  unaffected (the leaves' t_E outranks the hint).

The shipped params now carry the correct mapping with u_0 = +0.138, and
they seed q_source and err_scale explicitly (from the convention scan's
optimum), so neither defect reaches the start: the build starts at
lp 9285, within 10 nats of the optimum.

## Fit

`KMT-2021-BLG-1122L.yaml`: sync PTDE (`method: ptde`, `n_temps: auto`,
T_max 200, tune 5000, up to 20000 draws, hot chains stored), the direct
VBM N-body Op (no gradient), finite source with a linear Cousins-I limb
darkening law, NextGen SED on the source with the pure f_source constraint
(empty `filters:` list, as in KMT-2019-BLG-1806), wide zeropoint priors
(the pySIS magnitudes are instrumental).

**The first launch hung** (job 15506216, 64 cores): the DE seed polish
stopped at "sweep 400/400 IN PROGRESS (60/64 proposals back)" with the
job's CPU time frozen, and stderr carried three `double free or corruption
(!prev)` aborts from VBMicrolensing's three-body MultiMag2.  A worker that
aborts takes its proposal with it, and the polish has no `eval_timeout`
by design (run.md), so it waited forever (review 2.6.35).  The sampling
phase scores a lost call -inf after `eval_timeout` and recycles the pool,
so the shipped config sets `seed_polish: false` and lets PTDE do the
polishing.  `check_vbm_crash.py` measures the abort rate and names the
proposals that trigger it, in child processes that journal each proposal.

**The kernel fails on ~1.8e-3 of proposals** (`check_vbm_crash.py`, 6000
DE-style proposals around the start, each in a journaling child process):
11 failures under Multipoly, the method op.py selects for three bodies --
4 glibc aborts ("double free or corruption", "corrupted size vs.
prev_size"), 2 segfaults and 5 hangs (no return in 120 s; typical calls
take milliseconds).  Every one of the 11 has a companion at s < 0.2 or
s > 2.4, and all 11 evaluate cleanly under Nopoly -- but Nopoly is no
escape: the same 6000 proposals under Nopoly fail 13 times (12 segfaults,
1 hang), all clean under Multipoly, and 12 of the 13 sit inside
s in [0.4, 2.3] on both companions, where Multipoly failed 0 of 3075.
The two failure sets are disjoint; the reproducers are in
`check_vbm_crash_{Multipoly,Nopoly}_1.json`.  The sampler's start
population is scored serially in the main process with no timeout (review
2.6.36), so during that phase one such call kills or freezes the fit; the
second launch (job 15506222) spent 45 minutes there on one core before it
was stopped.  The params file therefore bounds both separations to
s in [0.4, 2.3] (published 1.386 and 1.601, both near the Einstein ring),
which keeps every failing geometry out of the kernel; for LensC the bound
goes on `log_s` directly (slot-1 relations are absent, 8.6.13).

**The bounded launch hung too** (job 15506232), in the serial start
scoring: "PTDE init rung 9" at 09:40 and nothing after, 95 minutes of one
core on a single VBM call reached by the auto start dispersion (factor 9
at that rung) in some parameter the stress test did not scan.  Rungs 1-8
at factors 3-7 had scored ~2000 proposals without one.  So the config
also sets `start_dispersion: 3.0`: every rung starts at the T=1
dispersion that those rungs proved.

**The T_max 200 ladder lives in the kernel's failure tail** (job 15506690):
with the init fixed the fit sampled, and in its first 70 steps hit a hung
VBM call 34 times, all on rungs 15-24 of 25 (T above ~25), each costing
the 10 s `eval_timeout` plus a pool recycle -- two thirds of the wall
clock.  So the config sets `T_max: 20`: the ladder stops where the hangs
start, and the start is the scan winner, so deep tempering is not what
this fit needs.

**T_max 20 did not help** (job 15507120): 10 timeouts in the first 29
steps, spread over rungs 4-13, so the hang is not a hot-rung phenomenon
but the kernel's retry bug firing on ordinary proposals at any
temperature; stopped.  **The fit is blocked on the kernel fix** -- and the
fix exists: the Radish agent root-caused both bugs and opened
valboz/VBMicrolensing#74 (a `flagbad` never reset after a retry, and
stale pairing indices in the image ordering).  Built from that branch
into a scratch venv (`vbm_fix_test.job`, VBMicrolensing 5.6), the 24
reproducers all pass and the 6000-proposal stress test shows 0 crashes
under either method, one multi-minute call per 6000 (rejected by the
sampler's eval timeout).  Results in `vbmfix_results/`.

**Status 2026-10-12.**  VBMicrolensing 5.6.1 (PyPI) contains the crash
fix (valboz/VBMicrolensing#74) and EXOZIPPy pins it (#428).  The Multipoly
slow tail (minutes per call at s < 0.2 or > 3) is fixed by
valboz/VBMicrolensing#75, not yet released.  The acceptance fit
(`KMT-2021-BLG-1122L_sync_slowtail.job`) therefore runs the SYNCHRONOUS
`ptde` sampler under a venv with #75 (`~/vbm_slowtail_venv`), T_max 200,
start dispersion 3.0, no seed polish; the async job and its fork-build
venv are retired.  The rejected-call guard in the magnification Op (#415)
removes the collapsed-theta_E proposals that used to stall it.

An outlier mixture (`likelihood: hogg`) was tried as a diagnostic and is
NOT used: on the wrong geometries it wrote off about half the I points in
the anomaly window, which would hide exactly the misfit that exposed the
mapping error.

RESULTS: (filled in when the fit lands -- posterior vs Table 2, and the
lens masses under the IMF and galactic priors).
