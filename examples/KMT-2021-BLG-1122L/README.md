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

KMTNet pySIS photometry of the three sites, I band only, from the KMTNet
event page (`KMT{A,C,S}14_I.pysis`; the V-band files are kept for the
record but not fitted).  `prepare_data.py` writes the `.pys` files the
config reads: HJD' + 2450000, magnitude, error.

| file | site | points |
|---|---|---|
| n20210603.I.KMTC14.pys | CTIO | 1346 |
| n20210603.I.KMTS14.pys | SAAO | 676 |
| n20210603.I.KMTA14.pys | SSO | 412 |

Coordinates (J2000) 17:35:51.10 -28:26:47.22, (l, b) = (-0.72, +2.08), from
the event page; pinned on all four stars in the params file.

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

Han's parameterization and EXOZIPPy's (conventions.md) differ in three
places, and the paper states the sense of none of them, so all three were
decided empirically by `check_start.py`: 32 combinations of
(alpha rule in {alpha + 180, 180 - alpha, alpha, -alpha}) x (u_0 sign) x
(alpha_3 = alpha_2 +/- psi) x (origin shift off/on), each evaluated as a
start state on the shipped data and ranked by the data logp
(`check_start.json`, job 15506210 on commit a05449bf).

| rank | alpha rule | u_0 sign | alpha_3 | origin | alpha_2 (deg) | alpha_3 (deg) | t_0 (HJD') | u_0 | data logp - best |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 180 - a | - | alpha_2 + psi | COM | 48.68 | 127.92 | 9370.609 | -0.4213 | 0 |
| 2 | a | - | alpha_2 - psi | COM | 131.32 | 52.08 | 9369.843 | -0.4213 | -12,770 |
| 3 | -a | - | alpha_2 - psi | COM | -131.32 | -210.56 | 9374.593 | -0.1139 | -19,564 |
| 4 | a + 180 | - | alpha_2 + psi | COM | 311.32 | 390.56 | 9365.859 | -0.1139 | -20,418 |
| 5 | -a | - | alpha_2 - psi | none | -131.32 | -210.56 | 9370.226 | -0.1260 | -46,520 |
| 10 | 180 - a | - | alpha_2 + psi | none | 48.68 | 127.92 | 9370.226 | -0.1260 | -120,153 |
| 32 | (worst) | | | | | | | | -20,875,006 |

The winner is unambiguous, with the runner-up 12,770 nats behind.  But
it is a convention discriminator, not a fit: the likelihood is Gaussian in
FLUX, so its 2434-point normalization is +47,273 nats and the winner's data
logp of -417 at the committed start (`check_chi2.py`) is chi2 = 95,381, 39
per point (CTIO 48, SAAO 39, SSO 11), against the paper's 1399.8/1425 on
its own, larger, data set.  Rounded published values, a caustic-crossing
source at rho = 2.5e-3 and the two-sigma-rounded origin translation are
enough to put the crossings off by more than their own width; the DE seed
polish at the start of the fit gained 43,742 nats in 400 sweeps.  The
three rules:

1. **alpha_EXOZIPPy = 180 deg - alpha_Han** (the Jung+2017 sense, NOT the
   alpha + 180 that conventions.md C21 records for Zang+2023), combined
   with **u_0 < 0**.  The pair is a convention choice, not a measurement:
   the exact mirror (+u_0, both alphas negated) is the same light curve.
2. **alpha_3 = alpha_2 + psi**: psi is counter-clockwise in the same sense
   as alpha.
3. **The origin must be translated.**  Han's origin is the "effective
   position" of the M1-M2 pair (the binary centre of magnification: the
   primary displaced by q_2 / ((1 + q_2) s_2) toward M2); EXOZIPPy's is the
   centre of mass of all three bodies (C13; `op.py`, `pos -= m @ pos`).
   The translation D = COM - EFF = (-0.026, -0.295) theta_E in the
   trajectory frame moves the closest approach to t_0' = t_0 - D_tau t_E
   (+0.383 d) and u_0' = u_0 + D_u (-0.126 -> -0.421).  Without it the
   best convention is 46,500 nats worse (rank 5), and the winner's own
   angles sit at rank 10.

The params file carries the winner with every mapping commented.

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
  and a HELD u_0 is passed as written, so with the shipped u_0 = -0.4213
  every candidate is rejected ("found no PSPL solution") while +0.4213
  fits.  The shipped start is sane only because the finder fails on it.
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

Acceptance fit: job 15507641 under the fixed kernel
(`KMT-2021-BLG-1122L_vbmfix.job`: the venv's python, T_max 200 and the
auto start dispersion restored, no seed polish, bounded separations),
started 2026-10-06.  The shipped config keeps the T_max 20 /
start_dispersion 3.0 workarounds until the fix is released.  RESULTS:
(filled in when it lands -- posterior vs Table 2 through the mappings
above, and the lens masses under the IMF and galactic priors).
