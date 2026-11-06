# Validation of the galactic-model microlensing prior

EXOZIPPy's `galacticmodel` component (`model: koshimoto21`, the default) is a
port of genulens (Koshimoto, Baba & Bennett 2021, ApJ 917, 78, "K21";
https://github.com/nkoshimoto/genulens, version 2.0.0a4, commit 2cc5870).
Its point-level pieces (densities, kinematics, mass function, source
weighting, binaries) are tested against genulens's own functions in
`tests/test_koshimoto21.py`.  This document is the END-TO-END check: the
prior a microlensing fit actually samples, compared event by event with
genulens and with the timescales OGLE-IV measured.

Everything here is reproduced by `scripts/galactic_prior_validation/validate.py`
(the commands are below and in `scripts/galactic_prior_validation/README.md`);
the stored summaries are in `scripts/galactic_prior_validation/refs/`;
`tests/test_galactic_prior_validation.py` guards the result.  Review tags: 7.16.3,
1.7.6.

## What is compared

**The prior.**  A lens + source model with `galacticmodel` and NO light curve
-- the prior a fit uses when its data do not constrain the lens.  Its event
rate weighting is the model's own; nothing is added for this check:

- `galacticmodel.kinematic_prior` gives each star the number density of the
  K21 populations (seven thin-disk age bins, the thick disk, the E+E_X bar,
  the stellar halo, and the nuclear stellar disk (NSD) on sight lines inside
  genulens's box, |l| < 5 deg, |b| < 2 deg) times its velocity distribution.
  The lens takes the density of all objects times D^2; the source, which is
  seen through a flux limit, takes genulens's default source weight, the
  main-sequence density times sqrt(D_S / 8 kpc) (genulens `gammaDs 0.5`).
  The lens mass prior is the present-day mass function: K21's IMF below each
  population's turnoff, white dwarfs, neutron stars and black holes above it,
  with genulens's natal kicks (350 km/s neutron stars, 100 km/s black holes).
- `mulensevent.event_rate_prior` adds the rate factor theta_E mu_rel.
- `mulensevent.source_behind_lens` is a soft barrier on D_S - D_L.

The supports are 100 pc - 20 kpc in distance and +/-50 mas/yr in proper
motion for both stars.  With no light curve there is no Earth velocity, so
mu_rel and t_E are heliocentric.

**genulens**, unmodified, `REMNANT 1` (remnants on, as the port), default
source weighting, `vEarthlb 0 0` (heliocentric, as the port), 2e6 simulated
events per line, weighted by its event weight `wtj`, in two configurations:

- **plain** -- every other option at genulens's default;
- **Dmax 20 kpc** -- the runtime option `Dmax 20000`, which extends genulens's
  line-of-sight grid to the port's distance reach.

**OGLE-IV**: the efficiency-corrected t_E distributions of Mroz et al. 2017
(Nature 548, 183; nine high-cadence fields, binned, Extended Data Tables 4
and 5) and Mroz et al. 2019 (ApJS 244, 29; per event with the inverse
detection efficiency, Table 3), for the fields the sight lines sit in.

**Sight lines** (Galactic l, b in degrees): Baade's window (1.0, -3.9),
inside OGLE field BLG513; the nine Mroz et al. 2017 high-cadence fields at
their centers -- BLG500 (1.00, -1.03), BLG501 (-0.06, -1.64), BLG504 (2.15,
-1.77), BLG505 (1.09, -2.39), BLG506 (0.01, -3.00), BLG511 (3.28, -2.52),
BLG512 (2.22, -3.14), BLG534 (-1.14, -2.25), BLG611 (0.33, +2.82); the
low-cadence field BLG521 (1.19, -5.11); OGLE-2014-BLG-0939's line (6.65,
+3.61), in BLG630; and three more NSD-box lines in the Roman Galactic Bulge
Time Domain Survey footprint, nsd_pt (0.3, -1.6), gbtds_w (-0.4, -1.2) and
nsd_lowb (0.0, -0.6).  Six of the fifteen (blg500, blg501, blg504 and the
last three) are inside genulens's NSD box.
Baade's window and nsd_pt are also run with unresolved binary lenses on both
sides (`binaries: true`, genulens `BINARY 1`; M_L is then the effective lens
mass).

## Method: importance sampling, not MCMC

**NUTS does not sample this prior correctly.**  The rate-weighted lens mass
function has two modes in log mass -- luminous stars, white dwarfs and
neutron stars below ~2 Msun, black holes above ~4 Msun -- separated by a
trough near 3 Msun where the present-day mass function has no objects at
all: the per-dex density is 10^-3 of its peak at 2 Msun, 10^-9 at 2.5 Msun,
10^-19.5 at 3 Msun (the floor), and back to 10^-3 at 5 Msun.  NUTS cannot
cross a 19-dex wall.  On Baade's window, 64 chains x 3000 draws (R-hat
1.015, bulk ESS >= 3900) contained NOT ONE draw above 3 Msun, where the prior
puts 1.5% of the event rate; and since those are the longest events, the
NUTS t_E q90 / q95 came out ~4% / 7% short (50.9 / 69.7 d against 53.2 /
74.8 d below).  R-hat cannot detect a mode no chain finds.  **This is a sampler
caveat for fits, too**: a microlensing fit whose light curve does not pin
the lens mass (weak pi_E and theta_E) and that samples with NUTS samples a
prior without black-hole lenses.  Parallel tempering (`ptde`) with its
default ladder does reach the mode: on the same model two runs gave a cold-
chain black-hole share of 1.92% and 1.05% (mean 1.49%), with the share
arriving in bursts, so its Monte Carlo error is set by the number of
excursions (~0.3-0.5% absolute per default-length run) rather than by the
number of draws.

So the prior is evaluated by **importance sampling** (IS;
`scripts/galactic_prior_validation/galactic_prior/importance.py`).  The target is the
model's own compiled logp in its raw coordinates -- every prior term,
potential and Jacobian, exactly what a sampler sees -- and the reported
quantities are the model's own derived nodes (t_E, mu_rel, D_L, D_S, M_L).
The proposal is a product of

- a mixture of up to 40 multivariate Student-t components (5 degrees of
  freedom, covariances inflated by 1.5^2) over every coordinate except the
  lens log mass, fit by EM to a short NUTS run (4 chains x 500 draws; it
  only seeds the proposal) together with one broad component (the pilot's
  covariance x 3^2, 20%), and refit to the importance-resampled draws over
  three adaptive rounds of 2e5, the pilot's fit keeping a 20% share; and
- for the lens log mass, a piecewise-constant density on a 0.01-dex grid
  from 10^-3.3 to 10^2.3 Msun: 85% the conditional profile p(log M | rest)
  evaluated ON THE GRID at 256 pilot points and averaged (so the black-hole
  mode is in it although the pilot never visited it), 15% uniform in log M.

After the first round half the proposal is instead a mixture over ALL
coordinates, which carries the mass-velocity correlation of kicked neutron
stars that the product cannot, and in every round 20% of the proposal draws
the lens's distance and velocity three times wider (kicked lenses; the
nearest lenses).  The final draw (2e6 per line, 6e6 on nsd_lowb) uses the
round's best proposal.  One limit of importance sampling is worth stating:
a region the proposal under-covers shows up in no diagnostic -- it simply
gets no draws -- so the defensive shares above, not the ESS or k-hat, are
what protect the tails, and the line-by-line agreement with genulens below
is the evidence that they do.  Quantile errors are a weighted (Poisson) bootstrap,
100 replicates, applied identically to genulens's weighted events; a z-score
uses the combined error sqrt(se_port^2 + se_genulens^2).  The diagnostics
reported per line are the Kish effective sample size, the largest
normalized weight, and the Pareto k-hat of the weights (Vehtari et al. 2024,
JMLR 25, 72; k < 0.7 means the estimate's error is trustworthy).

```bash
cd scripts/galactic_prior_validation
PYTHONPATH=<repo>/src python validate.py sample \
    --run baade --out-dir <scratch>/port --cores 24        # one line
python validate.py genulens --genulens <path>/genulens \
    --out-dir <scratch>/genulens                           # all lines
python validate.py compare --port-dir <scratch>/port \
    --genulens-dir <scratch>/genulens --store
python validate.py figures --port-dir <scratch>/port \
    --genulens-dir <scratch>/genulens --out-dir <scratch>/figures
```

`compare` writes `comparison.json` and the tables below (`comparison.md`)
next to the draws; `figures` writes `cdf_<line>.pdf` (the five CDFs, port
vs plain genulens, with residual panels for the plain and matched
comparisons), `summary_quantiles.pdf` (every line's fractional quantile
differences with error bars) and `ogle_tE.pdf` (t_E CDFs against OGLE per
field), each also as PNG.

## Results

### Headline

Fifteen sight lines, two of them also with binary lenses: 17 comparisons x 5
quantities x 5 quantiles (q05, q10, q50, q90, q95) = 425 quantiles each.

| comparison | quantiles beyond 1% | beyond 1%, M_L q95 excluded | largest KS D (t_E) | largest KS D (any other than M_L) | KS D of M_L |
|---|---|---|---|---|---|
| port vs PLAIN genulens | 114 / 425 | 105 / 408 | 0.0132 | 0.0328 (D_S) | 0.009-0.013 |
| port restricted to plain genulens's support vs plain genulens | 28 / 425 | 18 / 408 | 0.0032 | 0.0042 | 0.009-0.012 |
| port with the bin-edge support vs genulens Dmax 20 kpc ("matched") | 34 / 425 | 23 / 408 | 0.0037 | 0.0054 | 0.008-0.014 |

- **Against plain genulens** the port differs in two systematic ways: t_E's
  q05 is 1.8-3.3% SHORT on every line off genulens's NSD box (baade,
  blg505/506/511/512/534/611, blg521, ob140939), and on the NSD-box lines
  t_E's q50-q95 run 1.4-5.5% LONG with D_S's q90/q95 3.4-8.0% FAR (worst
  nsd_lowb, blg500, gbtds_w; D_S q95 also runs 2.0-3.5% far off the box).
  Otherwise mu_rel agrees to <= 1.7%, M_L's q05-q90 to <= 1.4%, and D_L to
  <= 2.0% (q10-q95) and 3.1% (q05).
- **Both are genulens discretizations**, and applying them to the port's
  draws at analysis time removes them (next section): with genulens's
  support imposed on the port, 18 of 408 quantiles (M_L q95 excluded) are
  beyond 1%, none beyond 2.7%, and every KS distance other than M_L's is
  <= 0.0042 (t_E <= 0.0032).  What is left is listed under "Residuals".
- **The black-hole share of the rate** (M_L > 3 Msun) is 1.47-1.62% in the
  port and 1.44-1.62% in genulens on every line; NUTS gives 0.
- **Against OGLE** the port's t_E and plain genulens's are indistinguishable
  on every field (KS D equal to within 0.004), and both are consistent with
  OGLE at p >= 0.02 on all twelve fields -- which K21's fit to these data
  makes the expected outcome, not an independent confirmation.

`figures` draws all of this: `summary_quantiles.pdf` (left column: plain;
right column: matched), one `cdf_<line>.pdf` per line, and `ogle_tE.pdf`.

### The two genulens discretizations

Neither is a modelling choice; both are numerical features of genulens's
event generator, identified in its source (`los_density_grid.cpp`).  This
validation reproduces each one on the PORT's side instead
(`sightlines.genulens_support_mask`), so genulens is only ever run
unmodified.

**(e) The lens is drawn only in front of the source's distance bin.**
genulens tabulates each line of sight on a distance grid of step dD (100 pc;
25 pc inside its NSD box), draws the source distance, and then draws the
lens only from the bins wholly in front of the source's bin
(`get_lens_importance_bins`: bins 0 .. int(D_S / dD) - 1).  So no lens sits
within one grid step in front of its source.  Those lenses have the
smallest pi_rel and so the SHORTEST events: they are 0.3-0.6% of the port's
rate off the NSD box and 0.04-0.12% inside it, and they set t_E's q05 --
cutting them from the port lengthens its q05 by 2.2-3.7% (100-pc grid) or
0.4-0.9% (25-pc grid) and moves q50 by <= 0.7% and q95 by <= 0.3%.  This is
the whole short-t_E "deficit" against plain genulens.

**(f) The grid ends at Dmax = 16 kpc.**  genulens's default line-of-sight
grid stops at 16 kpc, for sources and lenses alike; the port, like a default
fit, reaches the distance bound (20 kpc here).  0.5-1.3% of the port's rate
has its source beyond 16 kpc off the NSD box, and 1.9-3.3% on the NSD-box
lines, where |b| is small and the far disk behind the bulge is seen through.
Those far sources make longer events: removing them from the port shortens
t_E's q95 by 0.4-1.4% off the box and 1.9-4.7% on it, and D_S's q95 by
1.6-3.4% and 4.5-7.5%.  This is the whole NSD-line excess.  `Dmax` is a
genulens runtime option, so the matched comparison simply runs genulens with
`Dmax 20000`.

### Residuals

After both, what remains is at the level of the port's stated
approximations (`src/exozippy/components/galacticmodel/galacticmodel.md`):
the Shu distribution function evaluated at a star's own (R, z) rather than
at genulens's table node, its tabulated normalization (within 0.7% of
genulens's own), the untruncated velocity tails, isotropic-base natal kicks,
and the 0.02-dex smoothing of the remnant masses:

- **D_L q05 at the higher-|b| lines**: +1.6 to +2.6% (2.8-6.8 sigma) at
  baade, blg511, blg512, blg534, blg521 and ob140939 (and baade with
  binaries); within +/-1.2% elsewhere.  The port has slightly fewer of the
  nearest (disk) lenses, D_L ~ 2-3 kpc, on lines that look out of the plane
  -- where the distribution function at a star's own (R, z) and genulens's
  table-node lookup differ most.  It does not reach t_E (t_E q05 within
  0.3% on those lines).  The same sign and size were measured on an
  independent importance-sampling run with a different proposal.
- **M_L** -- KS D 0.008-0.014 on every line, against <= 0.0054 for the
  other quantities: a narrow spike in the CDF residual at the white-dwarf /
  neutron-star boundary (~1.2-1.4 Msun) where the port's kernel-smoothed
  initial-final mass relation meets genulens's point masses.  M_L's q95
  sits on that step, so it is the noisiest quantile; the three largest
  differences are blg505 +3.9% (1.5 sigma), gbtds_w +3.5% (2.9 sigma) and
  ob140939 -2.7% (2.6 sigma).
- **The black-hole share** is higher in the port on 16 of 17 lines, by up
  to 0.17 absolute percent (median 0.04, i.e. ~3% of itself; at most 3.5
  sigma, gbtds_w).
- **gbtds_w t_E q95** +2.9% (2.8 sigma), with q90 +1.1%: this line's pilot
  NUTS run was poor (its first importance round had an ESS of 90) and the
  adaptive rounds recovered, so its tail is the least certain of the set.

`nsd_lowb` (|b| = 0.6) is the hardest line for the importance sampler (the
smallest ESS fraction per draw), so it was run with 6e6 final draws; its
matched t_E agrees to <= 0.7% at every quantile.

### OGLE-IV

The port's efficiency-corrected-t_E consistency with OGLE is the same as
genulens's on every field (table below; `ogle_tE.pdf`), because the port
reproduces genulens.  Three caveats keep this a consistency check:

1. **Circularity.**  K21 fit their mass function to these same OGLE-IV t_E
   distributions (Mroz et al. 2017, 2019), so genulens -- and therefore the
   port -- agreeing with them holds by construction.  Do not tune the prior
   to OGLE's t_E; that would count the data twice.
2. **Survey selection.**  The port weights sources by genulens's default
   sqrt(D_S) survey approximation; no luminosity function or
   field-specific magnitude limit is applied, and OGLE's t_E is geocentric
   (a season average, ~1-2% longer than heliocentric at q90).
3. **Short events.**  The Mroz et al. 2017 high-cadence fields (BLG500 and
   BLG501 most) have an efficiency-corrected excess below t_E ~ 0.5 d --
   the free-floating-planet candidates of that paper -- that a mass function
   ending at 1e-3 Msun does not produce; the binned KS compares only the
   tabulated 0.1-316 d range and is dominated by it on those two fields.

The low-cadence fields BLG521 (69 events) and BLG630 (30 events) have
too few events to say more than p ~ 0.04-0.09.

### Sampler caveat for fits

A fit samples this prior wherever its light curve leaves the lens
unconstrained.  With NUTS, such a fit has no black-hole lenses (above), so
its t_E tail and any lens-mass or remnant-class probability it reports are
biased low; the parallel-tempering sampler reaches the black-hole mode on
average (above).  This is a property of the sampler on this prior, not of
the prior.

## What the test guards

`tests/test_galactic_prior_validation.py` (slow) importance-samples Baade's window
with a smaller budget (4e5 final draws) and requires every q05-q95 of t_E,
D_L, D_S, mu_rel and M_L, after the bin-edge cut, to agree with the stored
genulens Dmax 20 kpc run within 4 x the combined bootstrap error + 1%, and
the black-hole share within 4 combined sigma.  The fast tests check the
statistics (weighted quantiles, KS, bootstrap, Kish ESS, Pareto k), the
proposal densities, an importance estimate of an isolated 1.5% mode, the
genulens support helpers, and the OGLE reconstruction.

## Appendix: the full tables

Produced by `compare` (`comparison.md`).  Percentages are port / genulens -
1, z the difference over the combined bootstrap error; D in kpc, mu_rel in
mas/yr, M_L in Msun.

#### Port vs plain genulens: port / genulens - 1 in % (z)

| line | quantity | q05 | q10 | q50 | q90 | q95 | KS D |
|---|---|---|---|---|---|---|---|
| baade | t_E | -2.7 (-7.1) | -1.5 (-5.2) | +0.1 (+0.5) | +0.2 (+0.6) | +0.4 (+1.0) | 0.0028 |
| baade | D_L | +1.9 (+3.8) | +0.9 (+2.4) | +0.1 (+3.3) | +0.1 (+2.7) | +0.1 (+2.5) | 0.0046 |
| baade | D_S | +0.1 (+1.3) | +0.0 (+0.6) | +0.0 (+1.0) | +0.7 (+6.5) | +2.3 (+8.1) | 0.0055 |
| baade | mu_rel | -0.4 (-1.6) | -0.3 (-1.5) | -0.2 (-1.4) | -0.3 (-2.0) | -0.4 (-1.6) | 0.0021 |
| baade | M_L | -0.3 (-0.5) | -0.3 (-0.8) | -0.2 (-0.6) | +0.2 (+1.1) | +0.9 (+0.9) | 0.0105 |
| blg500 | t_E | +1.5 (+3.7) | +1.8 (+4.8) | +2.5 (+7.6) | +3.3 (+4.8) | +2.6 (+5.6) | 0.0125 |
| blg500 | D_L | +1.4 (+2.2) | +1.5 (+2.1) | +0.2 (+7.2) | +0.2 (+9.0) | +0.3 (+7.1) | 0.0100 |
| blg500 | D_S | +0.1 (+4.9) | +0.1 (+7.4) | +0.3 (+12.0) | +5.2 (+18.4) | +7.8 (+18.6) | 0.0328 |
| blg500 | mu_rel | -0.7 (-2.0) | -0.9 (-3.6) | -1.0 (-5.9) | -0.2 (-1.1) | -0.2 (-1.0) | 0.0083 |
| blg500 | M_L | +0.6 (+1.1) | +0.2 (+0.4) | -0.4 (-0.9) | +0.1 (+0.6) | +0.4 (+0.5) | 0.0107 |
| blg501 | t_E | +0.6 (+1.5) | +0.9 (+2.7) | +1.4 (+3.9) | +1.8 (+4.5) | +1.6 (+3.3) | 0.0074 |
| blg501 | D_L | +0.6 (+0.7) | +0.5 (+0.9) | +0.1 (+4.3) | +0.2 (+5.4) | +0.2 (+4.5) | 0.0080 |
| blg501 | D_S | -0.0 (-0.3) | +0.0 (+0.5) | +0.2 (+5.6) | +4.5 (+8.5) | +5.2 (+8.5) | 0.0200 |
| blg501 | mu_rel | -1.0 (-2.6) | -0.6 (-2.1) | -0.3 (-1.6) | -0.3 (-2.0) | -0.2 (-1.3) | 0.0040 |
| blg501 | M_L | +0.3 (+0.4) | -0.1 (-0.1) | -0.9 (-1.9) | +0.0 (+0.2) | +0.4 (+0.6) | 0.0126 |
| blg504 | t_E | -0.0 (-0.0) | +0.3 (+1.0) | +1.4 (+4.8) | +1.5 (+3.6) | +1.6 (+2.9) | 0.0067 |
| blg504 | D_L | -0.6 (-0.5) | +0.4 (+0.7) | +0.1 (+4.0) | +0.2 (+6.4) | +0.3 (+6.1) | 0.0066 |
| blg504 | D_S | +0.1 (+1.0) | +0.0 (+1.2) | +0.2 (+8.9) | +3.4 (+20.9) | +4.5 (+25.5) | 0.0194 |
| blg504 | mu_rel | -0.4 (-1.2) | -0.2 (-1.0) | -0.5 (-3.9) | -0.1 (-0.6) | +0.0 (+0.1) | 0.0049 |
| blg504 | M_L | -0.5 (-0.9) | -0.9 (-1.9) | -0.5 (-1.5) | +0.2 (+0.5) | +1.2 (+0.7) | 0.0120 |
| blg505 | t_E | -3.3 (-9.7) | -1.7 (-5.7) | +0.5 (+1.7) | +1.4 (+4.3) | +1.7 (+4.2) | 0.0040 |
| blg505 | D_L | -0.5 (-0.4) | +0.0 (+0.0) | +0.2 (+6.6) | +0.2 (+6.8) | +0.3 (+6.7) | 0.0088 |
| blg505 | D_S | +0.1 (+2.3) | +0.0 (+0.8) | +0.1 (+2.2) | +2.9 (+12.8) | +3.3 (+18.1) | 0.0114 |
| blg505 | mu_rel | -0.5 (-2.0) | -0.6 (-3.1) | -0.2 (-1.7) | +0.1 (+0.4) | +0.6 (+1.0) | 0.0039 |
| blg505 | M_L | +0.4 (+0.8) | -0.2 (-0.7) | -0.3 (-0.7) | +0.5 (+1.3) | +3.8 (+1.7) | 0.0104 |
| blg506 | t_E | -2.2 (-6.2) | -1.6 (-5.2) | -0.0 (-0.1) | +1.2 (+1.4) | +0.5 (+1.2) | 0.0042 |
| blg506 | D_L | +0.9 (+1.0) | +0.6 (+0.9) | +0.1 (+3.2) | +0.2 (+2.1) | +0.2 (+1.6) | 0.0053 |
| blg506 | D_S | +0.0 (+0.8) | -0.0 (-0.4) | +0.0 (+1.0) | +1.4 (+3.6) | +3.3 (+4.2) | 0.0088 |
| blg506 | mu_rel | -1.7 (-1.5) | -0.8 (-1.2) | -0.4 (-2.2) | -0.1 (-0.4) | +0.2 (+0.5) | 0.0036 |
| blg506 | M_L | -0.5 (-1.0) | -0.9 (-2.0) | -0.5 (-1.0) | +0.1 (+0.3) | +0.9 (+0.6) | 0.0110 |
| blg511 | t_E | -2.4 (-6.8) | -1.3 (-4.8) | +0.2 (+0.7) | +0.2 (+0.7) | -0.1 (-0.2) | 0.0041 |
| blg511 | D_L | +2.0 (+3.8) | +1.2 (+2.4) | +0.2 (+6.3) | +0.2 (+5.0) | +0.3 (+4.0) | 0.0071 |
| blg511 | D_S | +0.1 (+2.5) | +0.1 (+2.4) | +0.1 (+3.8) | +2.9 (+11.2) | +3.4 (+11.8) | 0.0131 |
| blg511 | mu_rel | -0.0 (-0.1) | -0.4 (-1.2) | -0.3 (-2.4) | -0.4 (-1.9) | -0.7 (-2.4) | 0.0035 |
| blg511 | M_L | -0.3 (-0.5) | -0.1 (-0.2) | -0.4 (-1.2) | -0.2 (-0.9) | -0.6 (-0.5) | 0.0120 |
| blg512 | t_E | -2.9 (-8.7) | -1.9 (-7.1) | +0.3 (+1.3) | +0.8 (+2.6) | +0.8 (+1.9) | 0.0037 |
| blg512 | D_L | +2.0 (+3.7) | +1.2 (+2.7) | +0.2 (+6.6) | +0.1 (+4.9) | +0.2 (+4.2) | 0.0078 |
| blg512 | D_S | +0.1 (+2.6) | +0.1 (+2.1) | +0.1 (+3.6) | +1.7 (+9.4) | +2.7 (+12.1) | 0.0077 |
| blg512 | mu_rel | -0.8 (-2.8) | -0.7 (-3.2) | -0.2 (-1.7) | -0.4 (-2.1) | -0.6 (-2.1) | 0.0026 |
| blg512 | M_L | -0.1 (-0.1) | +0.1 (+0.3) | +0.0 (+0.0) | -0.0 (-0.2) | -0.4 (-0.4) | 0.0102 |
| blg534 | t_E | -2.7 (-8.1) | -1.4 (-5.2) | -0.0 (-0.0) | +1.0 (+3.3) | +0.7 (+1.5) | 0.0036 |
| blg534 | D_L | +2.4 (+6.3) | +2.0 (+5.6) | +0.2 (+8.8) | +0.1 (+5.5) | +0.1 (+4.0) | 0.0096 |
| blg534 | D_S | +0.1 (+3.0) | +0.1 (+1.6) | +0.1 (+3.5) | +2.8 (+10.1) | +3.5 (+13.6) | 0.0119 |
| blg534 | mu_rel | -0.4 (-1.4) | -0.3 (-1.3) | -0.5 (-3.9) | -0.4 (-2.9) | -0.5 (-2.7) | 0.0044 |
| blg534 | M_L | +0.6 (+1.1) | +0.1 (+0.3) | +0.2 (+0.5) | -0.1 (-0.8) | -0.3 (-0.5) | 0.0096 |
| blg611 | t_E | -3.1 (-8.7) | -2.0 (-7.0) | +0.4 (+1.9) | +1.3 (+4.1) | +1.1 (+2.8) | 0.0040 |
| blg611 | D_L | +0.5 (+0.8) | +0.4 (+1.0) | +0.1 (+4.9) | +0.2 (+6.3) | +0.2 (+5.5) | 0.0063 |
| blg611 | D_S | -0.0 (-0.1) | -0.0 (-0.8) | +0.0 (+1.7) | +1.8 (+8.8) | +3.0 (+12.8) | 0.0098 |
| blg611 | mu_rel | -0.7 (-2.5) | -0.4 (-2.1) | -0.3 (-2.9) | -0.3 (-1.9) | -0.2 (-0.9) | 0.0036 |
| blg611 | M_L | -0.3 (-0.6) | -0.4 (-0.9) | -0.1 (-0.3) | +0.2 (+0.7) | +1.2 (+1.4) | 0.0103 |
| blg521 | t_E | -1.8 (-5.2) | -0.9 (-3.0) | +0.7 (+2.6) | +0.8 (+2.3) | +0.6 (+1.2) | 0.0033 |
| blg521 | D_L | +2.6 (+4.8) | +1.4 (+4.0) | +0.3 (+4.3) | +0.2 (+2.1) | +0.2 (+3.8) | 0.0061 |
| blg521 | D_S | +0.1 (+2.0) | +0.1 (+1.4) | +0.1 (+2.9) | +0.7 (+3.7) | +2.0 (+3.3) | 0.0064 |
| blg521 | mu_rel | -0.4 (-1.2) | -0.3 (-1.3) | -0.4 (-2.9) | -0.5 (-3.2) | -0.7 (-2.3) | 0.0056 |
| blg521 | M_L | +0.3 (+0.5) | +0.5 (+1.1) | +0.3 (+0.8) | -0.1 (-0.4) | -1.3 (-1.2) | 0.0086 |
| ob140939 | t_E | -1.8 (-5.5) | -0.7 (-2.4) | +0.8 (+3.6) | +1.5 (+4.1) | +1.6 (+3.7) | 0.0055 |
| ob140939 | D_L | +3.1 (+7.6) | +1.6 (+5.6) | +0.4 (+7.6) | +0.3 (+8.0) | +0.4 (+6.2) | 0.0081 |
| ob140939 | D_S | +0.3 (+3.3) | +0.2 (+3.0) | +0.3 (+6.5) | +2.8 (+17.2) | +3.1 (+17.8) | 0.0116 |
| ob140939 | mu_rel | -0.7 (-2.5) | -0.7 (-3.1) | -0.8 (-6.1) | -0.9 (-5.3) | -1.3 (-4.8) | 0.0065 |
| ob140939 | M_L | +0.1 (+0.1) | -0.2 (-0.5) | +0.0 (+0.0) | -0.4 (-1.7) | -3.5 (-3.5) | 0.0093 |
| nsd_pt | t_E | +0.2 (+0.6) | +0.4 (+1.3) | +1.7 (+6.3) | +2.7 (+4.8) | +2.4 (+4.2) | 0.0090 |
| nsd_pt | D_L | +1.5 (+2.5) | +0.9 (+1.8) | +0.1 (+4.0) | +0.2 (+5.1) | +0.3 (+6.1) | 0.0064 |
| nsd_pt | D_S | +0.1 (+2.7) | +0.1 (+4.3) | +0.2 (+8.9) | +4.8 (+19.0) | +5.5 (+17.2) | 0.0210 |
| nsd_pt | mu_rel | -0.5 (-1.7) | -0.7 (-3.1) | -0.7 (-5.7) | -0.2 (-1.9) | -0.2 (-1.3) | 0.0063 |
| nsd_pt | M_L | -0.0 (-0.0) | -0.2 (-0.5) | -0.7 (-1.9) | +0.1 (+0.6) | +1.1 (+1.6) | 0.0121 |
| gbtds_w | t_E | +0.4 (+1.0) | +0.8 (+2.2) | +2.5 (+7.9) | +4.1 (+7.6) | +5.5 (+5.6) | 0.0130 |
| gbtds_w | D_L | -0.3 (-0.2) | +0.6 (+1.1) | +0.1 (+5.9) | +0.1 (+5.0) | +0.2 (+6.5) | 0.0076 |
| gbtds_w | D_S | +0.1 (+2.3) | +0.1 (+4.0) | +0.2 (+9.4) | +4.7 (+19.3) | +6.4 (+28.6) | 0.0260 |
| gbtds_w | mu_rel | -1.2 (-3.7) | -1.1 (-4.0) | -0.8 (-4.6) | -0.1 (-0.7) | +0.1 (+0.3) | 0.0075 |
| gbtds_w | M_L | -1.1 (-1.6) | -0.4 (-1.0) | +0.4 (+1.0) | +0.7 (+2.7) | +1.9 (+1.8) | 0.0090 |
| nsd_lowb | t_E | +0.3 (+1.0) | +1.0 (+3.9) | +3.1 (+12.6) | +3.9 (+8.6) | +4.2 (+5.6) | 0.0132 |
| nsd_lowb | D_L | +1.4 (+2.7) | +1.3 (+4.3) | +0.1 (+10.5) | +0.1 (+9.2) | +0.2 (+8.6) | 0.0127 |
| nsd_lowb | D_S | +0.1 (+7.7) | +0.1 (+8.8) | +0.3 (+14.3) | +5.8 (+24.1) | +8.0 (+22.0) | 0.0322 |
| nsd_lowb | mu_rel | -1.1 (-3.4) | -0.7 (-3.2) | -0.5 (-3.8) | +0.0 (+0.1) | +0.1 (+0.8) | 0.0042 |
| nsd_lowb | M_L | -0.0 (-0.1) | -0.5 (-1.1) | +0.0 (+0.1) | +0.3 (+1.8) | +2.1 (+3.7) | 0.0095 |
| baade_bin | t_E | -2.5 (-6.9) | -1.6 (-5.3) | -0.4 (-2.0) | +0.1 (+0.3) | -0.1 (-0.1) | 0.0031 |
| baade_bin | D_L | +1.8 (+3.3) | +0.9 (+2.3) | +0.2 (+4.1) | +0.2 (+4.3) | +0.2 (+3.6) | 0.0052 |
| baade_bin | D_S | -0.0 (-0.3) | -0.0 (-1.1) | +0.0 (+0.5) | +0.6 (+4.8) | +2.0 (+6.0) | 0.0054 |
| baade_bin | mu_rel | -0.2 (-0.5) | -0.0 (-0.0) | +0.1 (+0.6) | -0.2 (-1.1) | -0.2 (-0.7) | 0.0017 |
| baade_bin | M_L | +0.2 (+0.3) | +0.0 (+0.0) | -1.0 (-3.1) | +0.6 (+2.0) | -0.8 (-1.8) | 0.0117 |
| nsd_pt_bin | t_E | +0.7 (+1.8) | +1.3 (+4.0) | +1.7 (+6.1) | +2.8 (+5.6) | +2.5 (+4.0) | 0.0106 |
| nsd_pt_bin | D_L | -0.0 (-0.0) | +0.1 (+0.2) | +0.1 (+5.0) | +0.2 (+6.0) | +0.2 (+5.9) | 0.0078 |
| nsd_pt_bin | D_S | +0.1 (+1.2) | +0.1 (+2.0) | +0.2 (+6.6) | +4.4 (+14.1) | +5.3 (+16.3) | 0.0210 |
| nsd_pt_bin | mu_rel | -0.7 (-2.2) | -0.8 (-2.8) | -0.4 (-2.7) | +0.2 (+1.1) | +0.4 (+1.2) | 0.0043 |
| nsd_pt_bin | M_L | -0.9 (-1.6) | +0.2 (+0.5) | +0.0 (+0.1) | +1.4 (+4.5) | +1.5 (+2.0) | 0.0095 |

#### Port restricted to plain genulens's support vs plain genulens

| line | quantity | q05 | q10 | q50 | q90 | q95 | KS D |
|---|---|---|---|---|---|---|---|
| baade | t_E | -0.0 (-0.0) | +0.1 (+0.4) | +0.2 (+1.0) | -0.0 (-0.1) | +0.2 (+0.6) | 0.0016 |
| baade | D_L | +1.6 (+3.1) | +0.5 (+1.3) | -0.0 (-0.1) | -0.0 (-0.7) | -0.0 (-0.5) | 0.0023 |
| baade | D_S | +0.1 (+1.4) | +0.0 (+0.5) | +0.0 (+0.3) | +0.1 (+1.5) | +0.4 (+1.8) | 0.0016 |
| baade | mu_rel | -0.2 (-0.6) | -0.1 (-0.4) | -0.0 (-0.2) | -0.2 (-1.2) | -0.3 (-1.2) | 0.0019 |
| baade | M_L | +0.1 (+0.2) | -0.2 (-0.6) | -0.2 (-0.5) | +0.3 (+1.1) | +0.9 (+0.8) | 0.0103 |
| blg500 | t_E | +0.3 (+0.8) | +0.2 (+0.7) | -0.6 (-2.4) | -1.0 (-3.0) | -0.9 (-2.1) | 0.0029 |
| blg500 | D_L | +0.3 (+0.4) | +0.7 (+1.6) | +0.0 (+1.1) | -0.0 (-0.4) | -0.1 (-2.0) | 0.0018 |
| blg500 | D_S | +0.0 (+0.7) | +0.0 (+1.2) | -0.0 (-1.4) | -0.3 (-2.3) | -0.2 (-1.5) | 0.0029 |
| blg500 | mu_rel | +0.9 (+3.0) | +0.4 (+1.5) | +0.0 (+0.4) | +0.4 (+2.0) | +0.3 (+1.0) | 0.0025 |
| blg500 | M_L | +0.4 (+0.8) | +0.1 (+0.2) | -0.2 (-0.4) | +0.2 (+0.9) | +0.7 (+0.7) | 0.0098 |
| blg501 | t_E | +0.1 (+0.2) | -0.0 (-0.1) | +0.1 (+0.5) | -0.2 (-0.4) | -0.3 (-0.6) | 0.0011 |
| blg501 | D_L | +0.0 (+0.0) | -0.3 (-0.6) | +0.0 (+0.2) | +0.0 (+0.9) | +0.0 (+0.6) | 0.0011 |
| blg501 | D_S | -0.1 (-1.3) | -0.1 (-1.4) | -0.0 (-1.4) | -0.2 (-0.8) | -0.0 (-0.0) | 0.0027 |
| blg501 | mu_rel | -0.4 (-0.9) | +0.0 (+0.2) | +0.1 (+1.0) | +0.0 (+0.3) | +0.1 (+0.3) | 0.0013 |
| blg501 | M_L | +0.2 (+0.4) | -0.2 (-0.4) | -0.5 (-1.2) | +0.1 (+0.5) | +0.5 (+0.8) | 0.0115 |
| blg504 | t_E | -0.6 (-1.7) | -0.6 (-1.8) | -0.2 (-0.9) | -0.6 (-2.0) | -0.6 (-1.2) | 0.0028 |
| blg504 | D_L | -1.2 (-1.0) | -0.4 (-0.5) | -0.0 (-0.8) | -0.0 (-0.7) | -0.0 (-0.3) | 0.0019 |
| blg504 | D_S | -0.0 (-0.1) | -0.0 (-0.8) | -0.0 (-0.4) | -0.4 (-2.2) | -0.2 (-1.4) | 0.0038 |
| blg504 | mu_rel | +0.4 (+1.1) | +0.5 (+2.0) | +0.1 (+0.6) | +0.2 (+0.7) | +0.3 (+0.6) | 0.0019 |
| blg504 | M_L | -0.4 (-0.8) | -0.9 (-2.0) | -0.4 (-1.0) | +0.2 (+0.7) | +1.3 (+0.7) | 0.0114 |
| blg505 | t_E | -0.5 (-1.4) | +0.2 (+0.7) | +0.2 (+0.8) | +0.4 (+1.1) | +0.7 (+1.7) | 0.0017 |
| blg505 | D_L | -1.1 (-0.7) | -0.8 (-0.8) | +0.0 (+1.0) | +0.0 (+0.8) | +0.1 (+1.4) | 0.0025 |
| blg505 | D_S | +0.1 (+1.6) | +0.0 (+0.3) | -0.0 (-0.5) | +0.2 (+1.1) | +0.0 (+0.0) | 0.0015 |
| blg505 | mu_rel | -0.1 (-0.4) | -0.2 (-0.9) | +0.1 (+0.4) | +0.3 (+0.8) | +0.7 (+1.0) | 0.0022 |
| blg505 | M_L | +0.4 (+0.7) | -0.2 (-0.4) | -0.3 (-0.6) | +0.5 (+1.1) | +3.9 (+1.4) | 0.0104 |
| blg506 | t_E | +0.5 (+1.4) | -0.1 (-0.3) | -0.3 (-0.9) | -0.1 (-0.3) | +0.1 (+0.2) | 0.0024 |
| blg506 | D_L | +0.4 (+0.4) | -0.1 (-0.1) | -0.0 (-0.3) | -0.0 (-0.4) | -0.0 (-0.2) | 0.0023 |
| blg506 | D_S | +0.0 (+0.5) | -0.0 (-0.8) | -0.0 (-1.0) | +0.0 (+0.1) | +0.0 (+0.0) | 0.0032 |
| blg506 | mu_rel | -0.2 (-0.6) | +0.1 (+0.5) | -0.1 (-0.5) | +0.1 (+0.2) | +0.4 (+0.7) | 0.0018 |
| blg506 | M_L | -0.8 (-1.5) | -1.0 (-2.3) | -0.3 (-0.8) | +0.2 (+0.5) | +1.1 (+0.6) | 0.0106 |
| blg511 | t_E | +0.1 (+0.3) | +0.1 (+0.5) | -0.3 (-1.4) | -0.6 (-2.0) | -1.0 (-2.4) | 0.0032 |
| blg511 | D_L | +1.2 (+2.5) | +0.5 (+1.0) | +0.0 (+0.8) | -0.0 (-0.2) | -0.0 (-0.4) | 0.0019 |
| blg511 | D_S | +0.1 (+1.4) | +0.0 (+1.0) | -0.0 (-0.2) | -0.1 (-0.7) | -0.1 (-0.4) | 0.0015 |
| blg511 | mu_rel | +0.4 (+1.4) | +0.4 (+2.1) | +0.1 (+0.6) | -0.1 (-0.5) | -0.5 (-1.5) | 0.0019 |
| blg511 | M_L | -0.2 (-0.4) | -0.2 (-0.4) | -0.4 (-1.2) | -0.1 (-0.5) | -0.5 (-0.4) | 0.0112 |
| blg512 | t_E | +0.1 (+0.4) | -0.1 (-0.2) | +0.3 (+1.4) | +0.2 (+0.5) | +0.1 (+0.3) | 0.0025 |
| blg512 | D_L | +1.5 (+3.0) | +0.6 (+1.6) | +0.1 (+2.1) | +0.0 (+0.4) | -0.0 (-0.2) | 0.0033 |
| blg512 | D_S | +0.1 (+2.8) | +0.1 (+1.9) | +0.1 (+2.0) | +0.3 (+1.4) | +0.3 (+1.1) | 0.0029 |
| blg512 | mu_rel | -0.3 (-1.2) | -0.3 (-1.5) | +0.0 (+0.3) | -0.2 (-1.3) | -0.4 (-1.5) | 0.0021 |
| blg512 | M_L | +0.1 (+0.1) | +0.1 (+0.3) | +0.0 (+0.1) | -0.0 (-0.1) | -0.4 (-0.4) | 0.0101 |
| blg534 | t_E | +0.2 (+0.7) | +0.4 (+1.5) | -0.3 (-1.4) | -0.0 (-0.0) | -0.4 (-1.1) | 0.0019 |
| blg534 | D_L | +1.7 (+4.8) | +1.2 (+3.3) | +0.1 (+3.2) | +0.0 (+0.2) | -0.0 (-0.8) | 0.0038 |
| blg534 | D_S | +0.1 (+2.2) | +0.0 (+1.0) | -0.0 (-0.1) | +0.0 (+0.1) | +0.1 (+0.4) | 0.0020 |
| blg534 | mu_rel | +0.2 (+0.6) | +0.2 (+1.3) | -0.1 (-1.0) | -0.2 (-1.8) | -0.4 (-2.0) | 0.0025 |
| blg534 | M_L | +0.5 (+1.0) | +0.1 (+0.2) | +0.2 (+0.4) | -0.1 (-0.6) | -0.3 (-0.5) | 0.0095 |
| blg611 | t_E | -0.1 (-0.3) | -0.3 (-1.0) | +0.3 (+1.1) | +0.6 (+1.8) | +0.6 (+1.4) | 0.0019 |
| blg611 | D_L | -0.1 (-0.3) | -0.3 (-0.7) | +0.0 (+0.2) | +0.0 (+0.7) | +0.0 (+1.3) | 0.0014 |
| blg611 | D_S | -0.0 (-0.3) | -0.0 (-0.8) | -0.0 (-0.8) | -0.0 (-0.2) | +0.0 (+0.1) | 0.0015 |
| blg611 | mu_rel | -0.3 (-1.2) | -0.1 (-0.3) | -0.0 (-0.3) | -0.2 (-1.2) | -0.1 (-0.6) | 0.0013 |
| blg611 | M_L | -0.2 (-0.3) | -0.3 (-0.9) | -0.1 (-0.2) | +0.2 (+0.7) | +1.1 (+1.4) | 0.0102 |
| blg521 | t_E | +0.1 (+0.4) | +0.3 (+0.9) | +0.6 (+2.3) | +0.6 (+1.8) | +0.3 (+0.7) | 0.0031 |
| blg521 | D_L | +2.3 (+4.2) | +1.0 (+2.6) | +0.1 (+1.5) | -0.0 (-0.4) | +0.1 (+1.1) | 0.0028 |
| blg521 | D_S | +0.1 (+1.3) | +0.0 (+0.8) | +0.1 (+2.1) | +0.1 (+1.9) | +0.4 (+2.6) | 0.0027 |
| blg521 | mu_rel | -0.0 (-0.1) | -0.1 (-0.3) | -0.3 (-3.3) | -0.4 (-2.2) | -0.5 (-1.6) | 0.0042 |
| blg521 | M_L | +0.2 (+0.4) | +0.4 (+0.8) | -0.0 (-0.1) | -0.1 (-0.2) | -1.1 (-1.1) | 0.0097 |
| ob140939 | t_E | -0.1 (-0.4) | +0.1 (+0.5) | +0.4 (+1.9) | +0.6 (+1.9) | +0.8 (+2.0) | 0.0030 |
| ob140939 | D_L | +2.7 (+6.3) | +1.2 (+4.5) | +0.2 (+2.8) | +0.1 (+1.8) | +0.1 (+1.1) | 0.0039 |
| ob140939 | D_S | +0.2 (+2.5) | +0.1 (+2.0) | +0.1 (+2.1) | +0.6 (+3.2) | +0.5 (+2.6) | 0.0034 |
| ob140939 | mu_rel | -0.2 (-1.0) | -0.2 (-0.9) | -0.4 (-3.0) | -0.6 (-4.1) | -1.1 (-4.4) | 0.0035 |
| ob140939 | M_L | +0.1 (+0.1) | -0.3 (-0.8) | -0.0 (-0.1) | -0.4 (-2.1) | -3.6 (-4.0) | 0.0094 |
| nsd_pt | t_E | -0.4 (-1.0) | -0.5 (-1.9) | -0.1 (-0.5) | +0.2 (+0.4) | +0.5 (+0.8) | 0.0012 |
| nsd_pt | D_L | +0.8 (+1.4) | +0.3 (+0.8) | -0.0 (-0.7) | +0.0 (+0.2) | +0.1 (+2.3) | 0.0014 |
| nsd_pt | D_S | +0.0 (+0.6) | +0.0 (+1.3) | +0.0 (+0.4) | +0.3 (+1.7) | +0.3 (+1.7) | 0.0018 |
| nsd_pt | mu_rel | +0.1 (+0.3) | +0.1 (+0.3) | -0.2 (-1.3) | +0.1 (+0.7) | +0.1 (+0.5) | 0.0016 |
| nsd_pt | M_L | -0.0 (-0.1) | -0.3 (-0.7) | -0.7 (-2.1) | +0.2 (+0.9) | +1.2 (+1.6) | 0.0118 |
| gbtds_w | t_E | -0.4 (-0.8) | -0.4 (-1.1) | -0.1 (-0.3) | +0.7 (+1.4) | +1.6 (+1.8) | 0.0024 |
| gbtds_w | D_L | -1.2 (-1.1) | -0.4 (-0.6) | +0.0 (+0.0) | -0.1 (-2.6) | -0.1 (-2.2) | 0.0028 |
| gbtds_w | D_S | +0.0 (+0.2) | +0.0 (+0.4) | -0.0 (-1.1) | -0.5 (-2.4) | -0.3 (-1.4) | 0.0040 |
| gbtds_w | mu_rel | -0.2 (-0.7) | +0.3 (+1.2) | +0.0 (+0.1) | +0.3 (+1.5) | +0.5 (+1.7) | 0.0020 |
| gbtds_w | M_L | -1.0 (-1.7) | -0.5 (-1.2) | +0.4 (+0.9) | +0.7 (+2.4) | +2.0 (+1.7) | 0.0092 |
| nsd_lowb | t_E | -0.6 (-2.0) | -0.5 (-2.1) | -0.2 (-1.2) | -0.4 (-1.3) | -0.6 (-1.4) | 0.0021 |
| nsd_lowb | D_L | +0.3 (+0.7) | +0.1 (+0.5) | +0.0 (+1.7) | -0.0 (-0.7) | -0.0 (-1.3) | 0.0025 |
| nsd_lowb | D_S | +0.0 (+2.0) | +0.0 (+1.2) | +0.0 (+0.5) | -0.2 (-1.7) | -0.2 (-1.8) | 0.0018 |
| nsd_lowb | mu_rel | +0.3 (+1.1) | +0.7 (+3.2) | +0.3 (+3.3) | +0.5 (+5.0) | +0.6 (+4.2) | 0.0033 |
| nsd_lowb | M_L | +0.1 (+0.2) | -0.0 (-0.1) | +0.1 (+0.3) | +0.3 (+1.9) | +2.1 (+4.2) | 0.0093 |
| baade_bin | t_E | -0.1 (-0.2) | +0.1 (+0.4) | -0.3 (-1.3) | -0.1 (-0.2) | -0.3 (-0.8) | 0.0021 |
| baade_bin | D_L | +1.4 (+2.5) | +0.6 (+1.5) | +0.0 (+1.2) | +0.0 (+0.9) | +0.0 (+0.7) | 0.0021 |
| baade_bin | D_S | -0.0 (-0.4) | -0.0 (-1.3) | -0.0 (-0.4) | -0.0 (-0.3) | -0.2 (-0.7) | 0.0020 |
| baade_bin | mu_rel | +0.2 (+1.1) | +0.2 (+1.2) | +0.2 (+1.8) | -0.1 (-0.4) | -0.0 (-0.1) | 0.0022 |
| baade_bin | M_L | +0.0 (+0.0) | -0.0 (-0.0) | -0.8 (-2.8) | +0.7 (+2.4) | -0.8 (-1.6) | 0.0113 |
| nsd_pt_bin | t_E | +0.1 (+0.3) | +0.3 (+0.8) | -0.1 (-0.5) | +0.4 (+1.1) | +0.3 (+0.5) | 0.0016 |
| nsd_pt_bin | D_L | -0.8 (-1.0) | -0.8 (-1.4) | -0.0 (-0.8) | +0.0 (+0.1) | +0.0 (+0.4) | 0.0019 |
| nsd_pt_bin | D_S | -0.0 (-0.2) | -0.0 (-0.3) | -0.0 (-2.1) | -0.5 (-3.1) | -0.2 (-1.6) | 0.0026 |
| nsd_pt_bin | mu_rel | +0.1 (+0.3) | +0.1 (+0.3) | +0.2 (+1.9) | +0.5 (+2.6) | +0.7 (+2.1) | 0.0028 |
| nsd_pt_bin | M_L | -0.7 (-1.3) | +0.4 (+1.0) | +0.0 (+0.1) | +1.5 (+4.3) | +1.5 (+1.8) | 0.0094 |

#### Port with the bin-edge support vs genulens Dmax 20 kpc

| line | quantity | q05 | q10 | q50 | q90 | q95 | KS D |
|---|---|---|---|---|---|---|---|
| baade | t_E | -0.2 (-0.6) | -0.0 (-0.1) | +0.2 (+0.7) | -0.3 (-0.9) | +0.3 (+0.6) | 0.0013 |
| baade | D_L | +1.6 (+3.2) | +0.5 (+1.3) | +0.0 (+0.9) | -0.0 (-0.1) | +0.0 (+0.3) | 0.0023 |
| baade | D_S | +0.1 (+1.5) | +0.0 (+0.8) | +0.0 (+1.4) | +0.1 (+0.9) | +0.1 (+0.4) | 0.0027 |
| baade | mu_rel | +0.0 (+0.2) | +0.2 (+0.9) | +0.1 (+0.5) | -0.2 (-1.0) | -0.4 (-1.7) | 0.0020 |
| baade | M_L | -0.5 (-0.8) | -0.8 (-2.1) | -0.3 (-0.8) | +0.3 (+1.2) | +0.7 (+0.7) | 0.0103 |
| blg500 | t_E | +0.4 (+1.0) | +0.3 (+0.8) | -0.4 (-1.3) | -0.2 (-0.3) | -0.4 (-0.8) | 0.0025 |
| blg500 | D_L | +0.3 (+0.5) | +0.1 (+0.1) | -0.0 (-0.4) | -0.0 (-1.1) | -0.1 (-2.7) | 0.0021 |
| blg500 | D_S | +0.0 (+1.1) | +0.0 (+1.5) | -0.0 (-1.5) | -0.1 (-0.5) | +0.2 (+0.5) | 0.0036 |
| blg500 | mu_rel | +0.9 (+2.4) | +0.3 (+1.0) | +0.1 (+0.4) | +0.3 (+2.0) | +0.3 (+1.5) | 0.0021 |
| blg500 | M_L | +0.6 (+1.1) | +0.2 (+0.5) | -0.3 (-0.8) | +0.4 (+2.2) | +1.8 (+2.1) | 0.0105 |
| blg501 | t_E | +0.2 (+0.4) | +0.0 (+0.1) | -0.7 (-1.8) | -0.5 (-1.3) | -0.4 (-0.9) | 0.0032 |
| blg501 | D_L | -0.1 (-0.1) | -0.4 (-0.6) | -0.0 (-0.1) | -0.0 (-0.5) | +0.0 (+0.1) | 0.0016 |
| blg501 | D_S | -0.1 (-1.1) | -0.0 (-0.9) | -0.0 (-1.1) | -0.1 (-0.1) | +0.1 (+0.2) | 0.0026 |
| blg501 | mu_rel | -0.1 (-0.1) | +0.2 (+0.8) | +0.3 (+1.9) | -0.0 (-0.2) | -0.1 (-0.5) | 0.0032 |
| blg501 | M_L | +0.5 (+0.8) | -0.0 (-0.1) | -1.5 (-2.7) | +0.1 (+0.5) | +1.0 (+1.7) | 0.0142 |
| blg504 | t_E | -0.5 (-1.4) | -0.6 (-2.0) | -0.4 (-1.3) | -0.9 (-2.3) | -0.9 (-1.7) | 0.0037 |
| blg504 | D_L | -1.1 (-1.0) | -0.2 (-0.3) | -0.0 (-0.3) | -0.0 (-1.2) | -0.0 (-0.2) | 0.0015 |
| blg504 | D_S | -0.0 (-0.3) | -0.0 (-0.8) | +0.0 (+0.0) | -0.4 (-2.4) | -0.3 (-1.7) | 0.0031 |
| blg504 | mu_rel | +0.6 (+2.2) | +0.6 (+3.0) | +0.1 (+0.9) | +0.3 (+1.0) | +0.3 (+0.6) | 0.0022 |
| blg504 | M_L | -0.4 (-0.8) | -1.1 (-2.4) | -0.5 (-1.5) | +0.3 (+0.9) | +1.3 (+0.8) | 0.0119 |
| blg505 | t_E | -0.2 (-0.6) | +0.4 (+1.4) | +0.1 (+0.5) | -0.0 (-0.1) | +0.1 (+0.3) | 0.0019 |
| blg505 | D_L | -1.1 (-0.8) | -0.9 (-1.0) | -0.0 (-0.1) | +0.0 (+0.0) | +0.0 (+0.7) | 0.0021 |
| blg505 | D_S | +0.1 (+1.4) | -0.0 (-0.1) | -0.0 (-0.5) | +0.1 (+0.5) | -0.0 (-0.2) | 0.0020 |
| blg505 | mu_rel | +0.1 (+0.3) | -0.0 (-0.2) | +0.2 (+1.0) | +0.3 (+0.9) | +0.6 (+0.9) | 0.0021 |
| blg505 | M_L | +0.4 (+0.8) | -0.2 (-0.5) | -0.4 (-0.9) | +0.6 (+1.3) | +3.9 (+1.5) | 0.0105 |
| blg506 | t_E | +0.8 (+2.3) | +0.1 (+0.4) | +0.0 (+0.0) | +0.7 (+0.8) | -0.2 (-0.3) | 0.0021 |
| blg506 | D_L | +0.6 (+0.7) | +0.2 (+0.3) | +0.0 (+0.2) | +0.1 (+0.9) | +0.1 (+1.0) | 0.0020 |
| blg506 | D_S | +0.1 (+1.0) | +0.0 (+0.2) | +0.0 (+0.5) | +0.4 (+0.9) | +0.6 (+0.7) | 0.0024 |
| blg506 | mu_rel | -1.3 (-1.0) | -0.4 (-0.6) | -0.2 (-1.1) | +0.0 (+0.1) | +0.3 (+0.7) | 0.0020 |
| blg506 | M_L | -0.4 (-0.7) | -0.7 (-1.7) | -0.4 (-0.7) | +0.3 (+0.8) | +1.4 (+0.9) | 0.0104 |
| blg511 | t_E | -0.2 (-0.4) | -0.1 (-0.2) | -0.3 (-1.3) | -0.6 (-1.7) | -0.5 (-1.1) | 0.0036 |
| blg511 | D_L | +1.2 (+2.8) | +0.6 (+1.3) | +0.0 (+0.9) | +0.0 (+0.5) | +0.0 (+0.5) | 0.0020 |
| blg511 | D_S | +0.1 (+1.4) | +0.0 (+1.1) | +0.0 (+0.5) | +0.0 (+0.1) | +0.0 (+0.1) | 0.0014 |
| blg511 | mu_rel | +0.4 (+1.6) | +0.1 (+0.4) | +0.0 (+0.4) | -0.1 (-0.7) | -0.3 (-1.0) | 0.0017 |
| blg511 | M_L | -0.5 (-0.9) | -0.2 (-0.4) | -0.4 (-1.0) | -0.1 (-0.5) | +0.3 (+0.2) | 0.0118 |
| blg512 | t_E | +0.1 (+0.4) | +0.0 (+0.2) | +0.2 (+0.8) | -0.1 (-0.2) | -0.2 (-0.5) | 0.0021 |
| blg512 | D_L | +1.6 (+2.9) | +0.6 (+1.4) | +0.1 (+2.1) | +0.0 (+0.5) | -0.0 (-0.9) | 0.0035 |
| blg512 | D_S | +0.1 (+3.4) | +0.1 (+2.9) | +0.0 (+1.7) | +0.0 (+0.2) | -0.0 (-0.1) | 0.0023 |
| blg512 | mu_rel | -0.4 (-1.3) | -0.4 (-1.8) | -0.0 (-0.1) | -0.1 (-0.9) | -0.4 (-1.5) | 0.0022 |
| blg512 | M_L | +0.0 (+0.0) | +0.1 (+0.2) | -0.0 (-0.1) | -0.1 (-0.3) | -0.7 (-0.8) | 0.0102 |
| blg534 | t_E | +0.1 (+0.4) | +0.1 (+0.5) | -0.4 (-1.6) | -0.5 (-1.7) | -0.6 (-1.5) | 0.0023 |
| blg534 | D_L | +2.0 (+4.9) | +1.4 (+3.9) | +0.1 (+3.4) | +0.0 (+0.4) | -0.0 (-1.3) | 0.0054 |
| blg534 | D_S | +0.1 (+2.3) | +0.0 (+1.4) | +0.0 (+0.1) | -0.0 (-0.2) | -0.1 (-0.8) | 0.0022 |
| blg534 | mu_rel | +0.1 (+0.4) | +0.1 (+0.6) | -0.0 (-0.0) | -0.3 (-2.2) | -0.4 (-2.4) | 0.0018 |
| blg534 | M_L | +0.5 (+1.0) | -0.0 (-0.1) | +0.1 (+0.1) | -0.1 (-0.6) | -0.6 (-0.9) | 0.0099 |
| blg611 | t_E | -0.1 (-0.2) | -0.4 (-1.3) | +0.1 (+0.5) | +0.2 (+0.6) | +0.1 (+0.3) | 0.0011 |
| blg611 | D_L | +0.1 (+0.2) | -0.3 (-0.7) | +0.0 (+0.3) | +0.0 (+0.0) | -0.0 (-0.0) | 0.0018 |
| blg611 | D_S | -0.0 (-0.2) | -0.0 (-1.1) | -0.0 (-0.0) | -0.2 (-1.2) | -0.2 (-0.8) | 0.0016 |
| blg611 | mu_rel | +0.0 (+0.0) | +0.3 (+1.2) | -0.0 (-0.3) | -0.1 (-0.9) | -0.2 (-0.6) | 0.0014 |
| blg611 | M_L | -0.8 (-1.5) | -0.4 (-1.0) | -0.3 (-0.7) | +0.3 (+1.2) | +0.9 (+1.1) | 0.0106 |
| blg521 | t_E | -0.0 (-0.1) | +0.1 (+0.3) | +0.5 (+1.9) | +0.5 (+1.4) | +0.3 (+0.6) | 0.0028 |
| blg521 | D_L | +2.5 (+5.3) | +1.4 (+3.9) | +0.1 (+2.4) | +0.1 (+1.2) | +0.1 (+1.2) | 0.0039 |
| blg521 | D_S | +0.1 (+1.8) | +0.1 (+1.6) | +0.1 (+1.8) | +0.3 (+1.5) | +0.8 (+1.3) | 0.0031 |
| blg521 | mu_rel | -0.1 (-0.5) | -0.2 (-0.8) | -0.3 (-1.9) | -0.4 (-2.3) | -0.5 (-1.8) | 0.0043 |
| blg521 | M_L | -0.2 (-0.3) | -0.1 (-0.2) | +0.1 (+0.4) | -0.2 (-0.8) | -1.7 (-1.5) | 0.0089 |
| ob140939 | t_E | +0.1 (+0.2) | +0.2 (+0.5) | +0.4 (+1.7) | +0.9 (+3.0) | +1.5 (+3.7) | 0.0029 |
| ob140939 | D_L | +2.6 (+6.8) | +1.4 (+5.2) | +0.2 (+3.8) | +0.1 (+1.1) | -0.0 (-0.3) | 0.0041 |
| ob140939 | D_S | +0.2 (+2.7) | +0.1 (+2.3) | +0.1 (+2.5) | +0.5 (+2.9) | +0.4 (+2.2) | 0.0030 |
| ob140939 | mu_rel | -0.4 (-1.5) | -0.4 (-1.9) | -0.4 (-3.3) | -0.5 (-3.1) | -1.0 (-3.9) | 0.0034 |
| ob140939 | M_L | +0.6 (+1.1) | +0.1 (+0.3) | +0.1 (+0.3) | -0.3 (-1.4) | -2.7 (-2.6) | 0.0092 |
| nsd_pt | t_E | -0.3 (-0.8) | -0.3 (-0.9) | -0.2 (-0.6) | +0.3 (+0.5) | -0.2 (-0.3) | 0.0012 |
| nsd_pt | D_L | +0.8 (+1.2) | +0.1 (+0.1) | -0.0 (-0.8) | -0.0 (-0.3) | +0.0 (+0.4) | 0.0018 |
| nsd_pt | D_S | +0.0 (+0.4) | +0.0 (+1.0) | -0.0 (-0.6) | +0.1 (+0.5) | +0.3 (+1.2) | 0.0015 |
| nsd_pt | mu_rel | +0.8 (+3.0) | +0.3 (+1.2) | -0.0 (-0.2) | +0.1 (+0.4) | -0.0 (-0.0) | 0.0015 |
| nsd_pt | M_L | -0.0 (-0.1) | +0.0 (+0.1) | -0.4 (-1.1) | +0.4 (+1.9) | +1.7 (+2.4) | 0.0113 |
| gbtds_w | t_E | -0.4 (-0.9) | -0.6 (-1.6) | -0.2 (-0.6) | +1.1 (+1.8) | +2.9 (+2.8) | 0.0035 |
| gbtds_w | D_L | -1.2 (-1.2) | -0.4 (-0.6) | -0.0 (-0.3) | -0.0 (-1.1) | -0.0 (-1.3) | 0.0024 |
| gbtds_w | D_S | +0.0 (+0.2) | +0.0 (+0.8) | -0.0 (-1.6) | -0.5 (-2.4) | -0.4 (-1.8) | 0.0045 |
| gbtds_w | mu_rel | +0.1 (+0.4) | +0.0 (+0.1) | +0.0 (+0.3) | +0.3 (+1.7) | +0.5 (+2.1) | 0.0028 |
| gbtds_w | M_L | -0.4 (-0.5) | -0.2 (-0.4) | +0.9 (+1.8) | +1.0 (+3.7) | +3.5 (+2.9) | 0.0081 |
| nsd_lowb | t_E | -0.3 (-1.0) | -0.4 (-1.5) | -0.6 (-2.3) | -0.7 (-1.4) | -0.4 (-0.6) | 0.0033 |
| nsd_lowb | D_L | +0.3 (+0.7) | +0.3 (+1.1) | +0.0 (+0.5) | -0.0 (-2.2) | -0.0 (-1.8) | 0.0029 |
| nsd_lowb | D_S | +0.0 (+1.9) | +0.0 (+0.8) | -0.0 (-1.0) | -0.3 (-1.3) | -0.3 (-1.0) | 0.0028 |
| nsd_lowb | mu_rel | +0.6 (+1.7) | +0.7 (+2.4) | +0.3 (+2.4) | +0.4 (+3.3) | +0.4 (+3.0) | 0.0031 |
| nsd_lowb | M_L | -0.1 (-0.2) | -0.4 (-0.8) | -0.0 (-0.1) | +0.1 (+0.8) | +1.3 (+2.1) | 0.0098 |
| baade_bin | t_E | +0.0 (+0.0) | +0.2 (+0.5) | -0.2 (-0.9) | -0.1 (-0.3) | -0.4 (-1.1) | 0.0018 |
| baade_bin | D_L | +1.6 (+2.8) | +0.6 (+1.6) | +0.1 (+1.9) | +0.1 (+1.4) | +0.0 (+0.7) | 0.0025 |
| baade_bin | D_S | -0.0 (-0.5) | -0.0 (-1.1) | +0.0 (+0.3) | +0.1 (+0.8) | +0.2 (+0.5) | 0.0011 |
| baade_bin | mu_rel | -0.1 (-0.2) | +0.1 (+0.2) | +0.2 (+2.2) | +0.0 (+0.1) | -0.1 (-0.3) | 0.0025 |
| baade_bin | M_L | +0.4 (+0.8) | -0.0 (-0.1) | -0.9 (-3.2) | +0.6 (+2.2) | -0.8 (-1.8) | 0.0115 |
| nsd_pt_bin | t_E | +0.0 (+0.1) | +0.3 (+1.1) | -0.1 (-0.4) | +0.2 (+0.5) | -0.1 (-0.1) | 0.0011 |
| nsd_pt_bin | D_L | -0.8 (-0.9) | -0.7 (-1.1) | -0.0 (-0.1) | -0.0 (-0.7) | -0.0 (-1.1) | 0.0015 |
| nsd_pt_bin | D_S | -0.0 (-0.1) | -0.0 (-0.3) | -0.0 (-1.2) | -0.1 (-0.4) | +0.1 (+0.2) | 0.0020 |
| nsd_pt_bin | mu_rel | +0.1 (+0.3) | +0.0 (+0.2) | +0.2 (+1.4) | +0.7 (+3.4) | +0.7 (+2.0) | 0.0034 |
| nsd_pt_bin | M_L | -1.3 (-2.1) | +0.1 (+0.2) | -0.1 (-0.2) | +1.2 (+3.1) | +1.6 (+1.8) | 0.0101 |

#### Quantiles (port as it is | plain genulens)

| line | quantity | port q05/q10/q50/q90/q95 | genulens q05/q10/q50/q90/q95 |
|---|---|---|---|
| baade | t_E | 3.63/5.28/17.4/52.9/74.3 | 3.73/5.36/17.4/52.8/74 |
| baade | D_L | 2.59/3.47/7.18/8.12/8.43 | 2.54/3.44/7.17/8.11/8.42 |
| baade | D_S | 7.21/7.47/8.51/10.3/11.7 | 7.2/7.47/8.51/10.2/11.4 |
| baade | mu_rel | 2.35/3.05/6.26/10.6/12.1 | 2.36/3.06/6.27/10.6/12.2 |
| baade | M_L | 0.0417/0.0701/0.454/0.954/1.18 | 0.0419/0.0703/0.455/0.952/1.17 |
| blg500 | t_E | 2.57/3.8/14/46.6/65.1 | 2.53/3.73/13.6/45.1/63.4 |
| blg500 | D_L | 3.69/5.09/7.74/8.31/8.51 | 3.64/5.02/7.73/8.3/8.49 |
| blg500 | D_S | 7.73/7.9/8.55/13.4/15.1 | 7.72/7.89/8.52/12.8/14 |
| blg500 | mu_rel | 2.42/3.14/6.58/11.3/12.8 | 2.43/3.17/6.64/11.3/12.9 |
| blg500 | M_L | 0.0422/0.0706/0.452/0.954/1.16 | 0.0419/0.0705/0.454/0.952/1.16 |
| blg501 | t_E | 2.63/3.86/13.7/44.9/63.1 | 2.61/3.83/13.5/44.1/62.1 |
| blg501 | D_L | 3.5/4.87/7.76/8.41/8.6 | 3.48/4.84/7.75/8.39/8.59 |
| blg501 | D_S | 7.75/7.94/8.63/12.4/14.1 | 7.75/7.94/8.62/11.9/13.4 |
| blg501 | mu_rel | 2.43/3.16/6.58/11.1/12.7 | 2.45/3.18/6.59/11.2/12.7 |
| blg501 | M_L | 0.0419/0.0702/0.448/0.949/1.14 | 0.0418/0.0703/0.452/0.948/1.14 |
| blg504 | t_E | 3.06/4.49/15.7/49.5/69 | 3.06/4.48/15.4/48.8/68 |
| blg504 | D_L | 3.17/4.38/7.46/8.21/8.46 | 3.19/4.36/7.45/8.19/8.44 |
| blg504 | D_S | 7.46/7.67/8.49/12.6/14.2 | 7.46/7.66/8.47/12.2/13.6 |
| blg504 | mu_rel | 2.37/3.08/6.36/10.8/12.4 | 2.38/3.09/6.4/10.8/12.4 |
| blg504 | M_L | 0.0417/0.0698/0.452/0.956/1.18 | 0.0419/0.0705/0.454/0.954/1.17 |
| blg505 | t_E | 3.01/4.41/15.1/47.8/67.2 | 3.11/4.48/15/47.1/66.1 |
| blg505 | D_L | 3.08/4.26/7.5/8.25/8.51 | 3.1/4.26/7.48/8.23/8.48 |
| blg505 | D_S | 7.51/7.71/8.54/11.4/13.3 | 7.5/7.71/8.54/11.1/12.8 |
| blg505 | mu_rel | 2.4/3.11/6.42/10.9/12.4 | 2.41/3.13/6.44/10.8/12.3 |
| blg505 | M_L | 0.0418/0.0701/0.452/0.955/1.19 | 0.0416/0.0703/0.453/0.95/1.15 |
| blg506 | t_E | 3.25/4.71/15.8/49.2/68.4 | 3.32/4.79/15.8/48.6/68 |
| blg506 | D_L | 2.93/4/7.47/8.38/8.65 | 2.91/3.98/7.47/8.36/8.63 |
| blg506 | D_S | 7.5/7.74/8.69/10.7/12.5 | 7.49/7.75/8.69/10.5/12.1 |
| blg506 | mu_rel | 2.34/3.06/6.33/10.7/12.3 | 2.38/3.08/6.35/10.7/12.2 |
| blg506 | M_L | 0.0416/0.0698/0.452/0.952/1.17 | 0.0418/0.0704/0.454/0.951/1.16 |
| blg511 | t_E | 3.6/5.22/17.4/53/73.8 | 3.68/5.29/17.4/52.8/73.8 |
| blg511 | D_L | 2.85/3.82/7.16/8.05/8.35 | 2.8/3.77/7.14/8.04/8.33 |
| blg511 | D_S | 7.18/7.43/8.37/11.8/13.5 | 7.17/7.42/8.36/11.4/13 |
| blg511 | mu_rel | 2.32/3/6.2/10.5/12 | 2.32/3.01/6.22/10.5/12.1 |
| blg511 | M_L | 0.0419/0.0705/0.455/0.957/1.2 | 0.042/0.0706/0.457/0.958/1.2 |
| blg512 | t_E | 3.44/5/16.7/51.4/71.9 | 3.54/5.1/16.7/51/71.3 |
| blg512 | D_L | 2.8/3.76/7.22/8.04/8.34 | 2.74/3.71/7.21/8.03/8.32 |
| blg512 | D_S | 7.24/7.48/8.39/10.7/12.6 | 7.23/7.47/8.38/10.5/12.2 |
| blg512 | mu_rel | 2.34/3.04/6.26/10.6/12.1 | 2.36/3.06/6.27/10.6/12.2 |
| blg512 | M_L | 0.0419/0.0706/0.455/0.953/1.17 | 0.042/0.0705/0.455/0.953/1.17 |
| blg534 | t_E | 3.02/4.4/14.9/47.4/66.2 | 3.1/4.47/14.9/46.9/65.8 |
| blg534 | D_L | 3.26/4.47/7.72/8.56/8.76 | 3.18/4.38/7.71/8.54/8.75 |
| blg534 | D_S | 7.75/8.01/8.79/11.4/13.3 | 7.74/8/8.79/11.1/12.9 |
| blg534 | mu_rel | 2.33/3.03/6.28/10.7/12.1 | 2.34/3.04/6.31/10.7/12.2 |
| blg534 | M_L | 0.042/0.0704/0.454/0.949/1.14 | 0.0418/0.0703/0.453/0.95/1.15 |
| blg611 | t_E | 3.18/4.62/15.6/48.4/67.5 | 3.28/4.72/15.5/47.7/66.8 |
| blg611 | D_L | 3.07/4.22/7.49/8.35/8.62 | 3.05/4.2/7.48/8.33/8.6 |
| blg611 | D_S | 7.5/7.74/8.67/11/13 | 7.5/7.74/8.67/10.8/12.6 |
| blg611 | mu_rel | 2.36/3.06/6.32/10.7/12.2 | 2.37/3.07/6.34/10.7/12.2 |
| blg611 | M_L | 0.0416/0.0701/0.452/0.949/1.15 | 0.0417/0.0704/0.452/0.947/1.14 |
| blg521 | t_E | 4.18/6.05/19.8/59.4/83 | 4.26/6.11/19.7/59/82.5 |
| blg521 | D_L | 2.14/2.91/6.84/7.93/8.29 | 2.09/2.87/6.82/7.92/8.27 |
| blg521 | D_S | 6.88/7.21/8.37/10.2/11.2 | 6.87/7.2/8.36/10.1/11 |
| blg521 | mu_rel | 2.34/3.03/6.21/10.6/12.3 | 2.34/3.04/6.23/10.7/12.4 |
| blg521 | M_L | 0.0419/0.0707/0.458/0.958/1.21 | 0.0417/0.0704/0.456/0.959/1.23 |
| ob140939 | t_E | 4.9/7.06/22.2/64.2/89.3 | 4.99/7.11/22.1/63.3/87.9 |
| ob140939 | D_L | 2.24/2.99/6.41/7.71/8.1 | 2.18/2.94/6.39/7.68/8.07 |
| ob140939 | D_S | 6.47/6.82/8.15/11.6/13.2 | 6.45/6.81/8.12/11.2/12.8 |
| ob140939 | mu_rel | 2.21/2.86/5.95/10.3/12 | 2.22/2.88/5.99/10.4/12.2 |
| ob140939 | M_L | 0.0422/0.0707/0.459/0.964/1.24 | 0.0422/0.0709/0.459/0.968/1.28 |
| nsd_pt | t_E | 2.61/3.84/13.7/45.4/63.5 | 2.61/3.82/13.5/44.2/62 |
| nsd_pt | D_L | 3.54/4.89/7.73/8.37/8.57 | 3.49/4.85/7.72/8.35/8.55 |
| nsd_pt | D_S | 7.73/7.91/8.6/12.6/14.2 | 7.72/7.9/8.58/12/13.5 |
| nsd_pt | mu_rel | 2.45/3.17/6.58/11.2/12.7 | 2.46/3.19/6.63/11.2/12.7 |
| nsd_pt | M_L | 0.0417/0.0702/0.45/0.95/1.15 | 0.0417/0.0703/0.453/0.949/1.14 |
| gbtds_w | t_E | 2.47/3.64/13.4/45.3/64.7 | 2.46/3.61/13/43.5/61.3 |
| gbtds_w | D_L | 3.67/5.14/7.86/8.43/8.61 | 3.68/5.1/7.85/8.42/8.59 |
| gbtds_w | D_S | 7.85/8.03/8.65/13/14.6 | 7.85/8.02/8.63/12.4/13.8 |
| gbtds_w | mu_rel | 2.41/3.14/6.59/11.3/12.9 | 2.44/3.18/6.64/11.3/12.8 |
| gbtds_w | M_L | 0.0415/0.0701/0.455/0.957/1.17 | 0.0419/0.0704/0.453/0.95/1.15 |
| nsd_lowb | t_E | 2.19/3.26/12.6/45.9/65.9 | 2.19/3.23/12.2/44.2/63.2 |
| nsd_lowb | D_L | 4.16/5.74/7.96/8.31/8.46 | 4.1/5.67/7.95/8.3/8.44 |
| nsd_lowb | D_S | 7.95/8.07/8.49/13.4/15 | 7.95/8.06/8.46/12.6/13.9 |
| nsd_lowb | mu_rel | 2.16/2.85/6.24/11/12.7 | 2.19/2.87/6.27/11/12.7 |
| nsd_lowb | M_L | 0.0418/0.0701/0.454/0.961/1.18 | 0.0418/0.0705/0.454/0.958/1.15 |
| baade_bin | t_E | 3.6/5.26/17.7/54.5/76.3 | 3.69/5.34/17.8/54.4/76.3 |
| baade_bin | D_L | 2.58/3.46/7.18/8.12/8.44 | 2.54/3.43/7.17/8.11/8.42 |
| baade_bin | D_S | 7.2/7.46/8.51/10.3/11.7 | 7.2/7.47/8.51/10.2/11.4 |
| baade_bin | mu_rel | 2.36/3.05/6.28/10.6/12.2 | 2.36/3.05/6.28/10.6/12.2 |
| baade_bin | M_L | 0.0413/0.0701/0.475/1.02/1.45 | 0.0412/0.0701/0.48/1.02/1.46 |
| nsd_pt_bin | t_E | 2.58/3.83/13.9/46.7/65.6 | 2.56/3.79/13.7/45.4/64 |
| nsd_pt_bin | D_L | 3.5/4.86/7.73/8.37/8.57 | 3.5/4.85/7.72/8.35/8.55 |
| nsd_pt_bin | D_S | 7.73/7.91/8.6/12.5/14.2 | 7.72/7.9/8.58/12/13.5 |
| nsd_pt_bin | mu_rel | 2.44/3.17/6.6/11.2/12.8 | 2.45/3.19/6.62/11.2/12.7 |
| nsd_pt_bin | M_L | 0.0406/0.0699/0.474/1.02/1.45 | 0.041/0.0697/0.474/1.01/1.43 |

#### Black-hole share of the rate (M_L > 3 Msun) and diagnostics

| line | port | genulens plain | genulens Dmax 20 | IS draws | Kish ESS | max weight | Pareto k |
|---|---|---|---|---|---|---|---|
| baade | 1.51 +/- 0.02% | 1.48 +/- 0.02% | 1.46 +/- 0.02% | 2e+06 | 2.87e+05 | 0.037% | 0.54 |
| blg500 | 1.50 +/- 0.02% | 1.49 +/- 0.02% | 1.46 +/- 0.03% | 2e+06 | 1.8e+05 | 0.133% | 0.46 |
| blg501 | 1.52 +/- 0.02% | 1.48 +/- 0.02% | 1.41 +/- 0.02% | 2e+06 | 1.11e+05 | 0.213% | 0.53 |
| blg504 | 1.51 +/- 0.03% | 1.46 +/- 0.02% | 1.45 +/- 0.02% | 2e+06 | 1.97e+05 | 0.080% | 0.51 |
| blg505 | 1.49 +/- 0.02% | 1.47 +/- 0.02% | 1.48 +/- 0.02% | 2e+06 | 1.75e+05 | 0.154% | 0.54 |
| blg506 | 1.48 +/- 0.02% | 1.44 +/- 0.02% | 1.44 +/- 0.02% | 2e+06 | 1.28e+05 | 0.169% | 0.54 |
| blg511 | 1.52 +/- 0.02% | 1.51 +/- 0.02% | 1.50 +/- 0.02% | 2e+06 | 2.38e+05 | 0.065% | 0.56 |
| blg512 | 1.48 +/- 0.02% | 1.47 +/- 0.02% | 1.49 +/- 0.02% | 2e+06 | 3.06e+05 | 0.030% | 0.51 |
| blg534 | 1.51 +/- 0.02% | 1.46 +/- 0.02% | 1.47 +/- 0.03% | 2e+06 | 3.02e+05 | 0.040% | 0.45 |
| blg611 | 1.50 +/- 0.02% | 1.46 +/- 0.02% | 1.47 +/- 0.02% | 2e+06 | 3.01e+05 | 0.035% | 0.46 |
| blg521 | 1.54 +/- 0.02% | 1.51 +/- 0.02% | 1.52 +/- 0.02% | 2e+06 | 1.66e+05 | 0.160% | 0.53 |
| ob140939 | 1.61 +/- 0.03% | 1.55 +/- 0.02% | 1.55 +/- 0.02% | 2e+06 | 3.04e+05 | 0.018% | 0.40 |
| nsd_pt | 1.47 +/- 0.02% | 1.44 +/- 0.02% | 1.42 +/- 0.02% | 2e+06 | 2.7e+05 | 0.044% | 0.55 |
| gbtds_w | 1.57 +/- 0.04% | 1.46 +/- 0.02% | 1.40 +/- 0.02% | 2e+06 | 1.74e+05 | 0.044% | 0.46 |
| nsd_lowb | 1.51 +/- 0.02% | 1.46 +/- 0.02% | 1.45 +/- 0.02% | 6e+06 | 3.44e+05 | 0.062% | 0.42 |
| baade_bin | 1.62 +/- 0.02% | 1.62 +/- 0.02% | 1.60 +/- 0.02% | 2e+06 | 2.71e+05 | 0.041% | 0.47 |
| nsd_pt_bin | 1.62 +/- 0.02% | 1.57 +/- 0.03% | 1.58 +/- 0.03% | 2e+06 | 2.18e+05 | 0.075% | 0.56 |

#### The two genulens discretizations, measured on the port's draws

| line | dD (pc) | rate share removed by (e) | (e): t_E q05 / q50 / q95 shift | rate share beyond 16 kpc | (f): t_E q95 / D_S q95 shift |
|---|---|---|---|---|---|
| baade | 100 | 0.45% | +3.02% / +0.48% / +0.20% | 0.50% | -0.39% / -2.03% |
| blg500 | 25 | 0.06% | +0.53% / +0.10% / +0.03% | 3.27% | -3.49% / -7.43% |
| blg501 | 25 | 0.06% | +0.55% / +0.06% / +0.02% | 1.99% | -1.88% / -4.97% |
| blg504 | 25 | 0.04% | +0.37% / +0.05% / +0.00% | 1.94% | -2.09% / -4.53% |
| blg505 | 100 | 0.62% | +3.51% / +0.67% / +0.31% | 1.13% | -1.25% / -3.34% |
| blg506 | 100 | 0.51% | +3.26% / +0.57% / +0.24% | 0.88% | -0.68% / -3.29% |
| blg511 | 100 | 0.50% | +3.20% / +0.53% / +0.27% | 1.31% | -1.14% / -3.36% |
| blg512 | 100 | 0.56% | +3.54% / +0.60% / +0.25% | 0.73% | -0.93% / -2.49% |
| blg534 | 100 | 0.61% | +3.70% / +0.68% / +0.31% | 1.19% | -1.41% / -3.42% |
| blg611 | 100 | 0.56% | +3.39% / +0.66% / +0.24% | 0.92% | -0.80% / -2.97% |
| blg521 | 100 | 0.35% | +2.26% / +0.35% / +0.15% | 0.54% | -0.48% / -1.64% |
| ob140939 | 100 | 0.31% | +2.15% / +0.35% / +0.14% | 1.00% | -0.91% / -2.55% |
| nsd_pt | 25 | 0.05% | +0.46% / +0.06% / +0.02% | 1.97% | -1.86% / -4.94% |
| gbtds_w | 25 | 0.07% | +0.51% / +0.08% / +0.03% | 2.59% | -3.73% / -6.22% |
| nsd_lowb | 25 | 0.12% | +0.87% / +0.15% / +0.06% | 3.22% | -4.68% / -7.53% |
| baade_bin | 100 | 0.45% | +2.78% / +0.47% / +0.21% | 0.53% | -0.44% / -2.22% |
| nsd_pt_bin | 25 | 0.06% | +0.39% / +0.06% / +0.04% | 2.10% | -2.18% / -5.29% |

#### t_E against OGLE-IV (efficiency corrected)

| line | OGLE field | OGLE q05/q10/q50/q90/q95 (d) | +/- q50 | port | KS D (p) | genulens plain KS D (p) |
|---|---|---|---|---|---|---|
| baade | mroz2019 BLG513 | 3.27/4.22/18.9/51.6/65.9 | 1.6 | 3.63/5.28/17.4/52.9/74.3 | 0.068 (0.31) | 0.071 (0.27) |
| blg500 | mroz2017 BLG500 | 0.23/0.882/11.3/44.2/65.6 | 1.7 | 2.57/3.8/14/46.6/65.1 | 0.105 (0.29) | 0.105 (0.3) |
| blg501 | mroz2017 BLG501 | 0.133/0.606/11.8/44.6/61.4 | 1.7 | 2.63/3.86/13.7/44.9/63.1 | 0.119 (0.019) | 0.115 (0.026) |
| blg504 | mroz2017 BLG504 | 2.93/4.08/15.2/49.7/69.5 | 0.84 | 3.06/4.49/15.7/49.5/69 | 0.023 (1) | 0.020 (1) |
| blg505 | mroz2017 BLG505 | 2.75/4.07/12.8/51.2/68.9 | 0.81 | 3.01/4.41/15.1/47.8/67.2 | 0.071 (0.036) | 0.070 (0.039) |
| blg506 | mroz2017 BLG506 | 3.61/5/16.1/59/108 | 1 | 3.25/4.71/15.8/49.2/68.4 | 0.038 (0.8) | 0.042 (0.7) |
| blg511 | mroz2017 BLG511 | 2.75/3.75/16.3/55.8/77.8 | 1.2 | 3.6/5.22/17.4/53/73.8 | 0.057 (0.44) | 0.060 (0.38) |
| blg512 | mroz2017 BLG512 | 1.96/4.07/15.6/54.5/71.8 | 1.2 | 3.44/5/16.7/51.4/71.9 | 0.041 (0.86) | 0.041 (0.86) |
| blg534 | mroz2017 BLG534 | 2.45/3.28/14.2/47/59.6 | 1.4 | 3.02/4.4/14.9/47.4/66.2 | 0.054 (0.58) | 0.056 (0.55) |
| blg611 | mroz2017 BLG611 | 2.53/4.37/17.2/45.3/67.3 | 1.4 | 3.18/4.62/15.6/48.4/67.5 | 0.046 (0.77) | 0.048 (0.72) |
| blg521 | mroz2019 BLG521 | 8.34/10.6/22.7/64.4/72.9 | 2.4 | 4.18/6.05/19.8/59.4/83 | 0.169 (0.044) | 0.168 (0.045) |
| ob140939 | mroz2019 BLG630 | 10.9/11.8/27.5/67.8/118 | 3.6 | 4.9/7.06/22.2/64.2/89.3 | 0.225 (0.089) | 0.228 (0.083) |
