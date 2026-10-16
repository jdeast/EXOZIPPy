# The KS relaxation engine: measurements (review 3.14.22)

The rule and its reasoning are in `src/exozippy/config.md`, "Precedence, and
which value absorbs a discrepancy".  This file holds the measurements behind
it, so the rule's doc does not have to carry tables.

How each number was taken: `System.prepare()` then `build_model()`, the start
logp at the raw start (`model.compile_logp()(system.get_raw_start(model))`),
`ModelAuditor.check_user_starts()` for the misses -- the way
`scripts/start_value_report.py` reads a start.  Every shipped top-level
config except the `.sed`/`.2`/`_hpc` fragments (the `ob09020_*` sampler
variants share `ob09020`'s model and params and are omitted), plus four
scenarios: `rank22` (DC2018_128 with the lens-mass hint dropped to
precedence 22, review 3.14.22(2)), `tp` (ob170114 seeded by its time of
periastron, review 1.1.9), `overdet` (ob170114 with its source proper motions
removed and t_E/pi_E pinned, review 8.6.15), `density` (hat3_mist with its
mass and radius pins replaced by a density pin of 2.6 g/cm3).

## How the engine is judged

The relaxation engine is judged by how well it RECOVERS THE USER'S START
-- whether the model is built at the values the params file states --
not by how well that start fits the data (JDE 2026-10-15).  The measure
is `ModelAuditor.check_user_starts`: every user value compared with the
value the compiled model starts at (requested vs produced); a miss is a
user value the start does not reproduce.  A low start logp from faithful
seeds is a question about the seeds or the fit, not about the engine; the
start logp tables further down are kept as secondary context only.

### Missed user starts, master vs this PR

Re-measured 2026-10-16 against master c0d4ec25 (identical to the counts
against 4e123970).

Each config is measured with its own branch's `examples/`.  This PR
changes the params files of `ob09020`, `ob09020_linear`,
`KMT-2019-BLG-1806` and `kelt17` (published values, see their READMEs);
with ob09020's old pins the rule missed 2 (it dropped the theta_E and
pi_E_E the masses and distances cannot give), against master's 5.

| config | master: missed user starts | this PR |
|---|---|---|
| astrometry_sim/simbinary | 0 | 0 |
| DC2018_128/DC2018_128 (rank22) | 0 | 0 |
| DC2018_128/DC2018_128 | 0 | 0 |
| GaiaBH1/gaiabh1 | 0 | 0 |
| galactic_model/gm | 0 | 0 |
| gj1214/gj1214 | 0 | 0 |
| hat3/hat3 | 2 (TESS.u1, TESS.u2) | 2 (TESS.u1, TESS.u2) |
| hat3/hat3_mist (density) | 1 (A.density) | 0 |
| hat3/hat3_mist | 0 | 0 |
| hat3/hat3_mist_claret | 0 | 0 |
| hat3/hat3_staronly | 0 | 0 |
| hat3/hat3_transitonly | 0 | 0 |
| hd80606/hd80606 | 0 | 0 |
| hd80606/hd80606_rvonly | 0 | 0 |
| hd80606/hd80606_transitonly | 0 | 0 |
| HIP1349/hip1349 | 0 | 0 |
| kb180087_obj3/kb180087_obj3 | 0 | 0 |
| kelt17/kelt17 | 0 | 0 |
| kelt17/kelt17_all | 1 (b.p) | 1 (b.p) |
| kelt17/kelt17_dt | 1 (b.p) | 1 (b.p) |
| kelt17/kelt17_fast | 0 | 0 |
| kelt17/kelt17_rm | 1 (b.p) | 1 (b.p) |
| kelt4/kelt4 | 0 | 0 |
| kelt4/kelt4_rv+transit+sed | 0 | 0 |
| kelt4/kelt4_rvonly | 0 | 0 |
| KMT-2019-BLG-1806/KMT-2019-BLG-1806 | 1 (t_E) | 0 |
| KMT-2021-BLG-1122L/KMT-2021-BLG-1122L | 0 | 0 |
| ob07224/ob07224 | 3 (pi_E_E, pi_E_N, Source.rho) | 2 (pi_E_E, pi_E_N) |
| ob08092/ob08092 | 1 (t_E) | 0 |
| ob09020/ob09020 | 5 (pi_E_E, pi_E_N, t_E, theta_E, Source.rho) | 0 |
| ob09020/ob09020_linear | 5 (pi_E_E, pi_E_N, t_E, theta_E, Source.rho) | 0 |
| ob140939/ob140939 | 0 | 0 |
| ob140939/ob140939_today | 0 | 0 |
| ob161003/ob161003 | 0 | 0 |
| ob161045/ob161045 | 0 | 0 |
| ob170114/ob170114 (overdet) | 3 (pi_E_E, pi_E_N, t_E) | 0 |
| ob170114/ob170114 | 0 | 0 |
| ob170114/ob170114 (tp) | 0 | 0 |
| ogle0383/OGLE_0383LD | 1 (t_E) | 0 |
| theophylline/theophylline | 0 | 0 |
| toi3919/toi3919 | 0 | 0 |
| toi3919/toi3919_claret | 0 | 0 |
| wasp18/wasp18 | 0 | 0 |

The misses left on this PR are the params files' own: hat3's TESS
limb-darkening starts contradict master's new u1/u2 priors (both
branches); kelt17_all/_dt/_rm pin a radius ratio their other pins do not
give; ob07224 pins a pi_E its masses and distances do not give.  Each is
a statement about the seeds, reported by name.

### Derived vs published, ob09020

At the shipped starts (`examples/ob09020/README.md` gives every seed's
source):

- Keplerian: t_0, u_0, rho, q, t_E = 76.900 d, theta_E, pi_E and
  alpha_0 = 189.080 deg are all reproduced exactly.  The one difference
  is s_0 = 0.42633 against Skowron+2011's 0.4294 (-0.7%): in this model
  s_0 is derived from Yee+2016's orbit and D_L with theta_E, so it is an
  inconsistency between the two papers, not an unsolved relation.
- Linear: s_0 = 0.4294, alpha_0 = 189.08 deg, ds/dt = 0.04294 r_E/yr,
  dalpha/dt = -131.78 deg/yr, t_E = 76.9 d and q = 0.273 are all exact;
  start logp +47,696.

## Start logp (secondary context)

## A. Start logp, one column per state

Cells: start logp (change from master), `m` = check_user_starts misses,
`neg` = negative masses in the ledger, `d` = user values on derived elements
DROPPED, `c` = relations the rigid user values contradict, `u` = relations a
tier could not settle.  `master` is commit 01d6a8cc, the base these were measured on (the branch
was rebased onto a later master afterwards, which changed `examples/hat3`'s
limb-darkening priors); `rule` is the KS
rule with the HELIOCENTRIC t_E / pi_E relations (the first commit);
`rule+geo` adds the GEOCENTRIC relations (review 8.6.23, the second
commit).

| config | scenario | master | rule | rule+geo |
|---|---|---|---|---|
| astrometry_sim/simbinary | shipped | -247.616, m0 | -247.616, m0 | -247.616, m0 |
| DC2018_128/DC2018_128 | rank22 | -912820, m0, **neg1** | 3065.37 (+9.16e+05), m0 | 3065.94 (+9.16e+05), m0 |
| DC2018_128/DC2018_128 | shipped | 3373.88, m0 | 3374.55 (+0.669), m0 | 3374.96 (+1.08), m0 |
| GaiaBH1/gaiabh1 | shipped | -280.186, m0 | -280.186, m0 | -280.186, m0 |
| galactic_model/gm | shipped | -1.14765, m0 | -1.14765, m0 | -1.14765, m0 |
| gj1214/gj1214 | shipped | 108478, m0 | 108478, m0 | 108478, m0 |
| hat3/hat3 | shipped | 272792, m0 | 272792, m0, d2, c2 | 272792, m0, d2, c2 |
| hat3/hat3_mist | density | -1706.07, m1 | -790.418 (+916), m0 | -790.418 (+916), m0 |
| hat3/hat3_mist | shipped | -49.3124, m0 | -49.3124, m0 | -49.3124, m0 |
| hat3/hat3_staronly | shipped | -63.6153, m0 | -63.6153, m0 | -63.6153, m0 |
| hat3/hat3_transitonly | shipped | 257824, m0 | 258432 (+608), m0 | 258432 (+608), m0 |
| hd80606/hd80606 | shipped | -132978, m0 | -132978, m0 | -132978, m0 |
| hd80606/hd80606_rvonly | shipped | -125231, m0 | -125231, m0 | -125231, m0 |
| hd80606/hd80606_transitonly | shipped | -7861.49, m0 | -7861.49, m0 | -7861.49, m0 |
| HIP1349/hip1349 | shipped | -342.896, m0 | -342.896, m0 | -342.896, m0 |
| kb180087_obj3/kb180087_obj3 | shipped | 56830.6, m0 | 56831.2 (+0.642), m0 | 56832.2 (+1.61), m0 |
| kelt17/kelt17 | shipped | -1458.24, m0 | -1458.24, m0 | -1458.24, m0 |
| kelt17/kelt17_all | shipped | 8799.74, m1 | 8799.74, m1, d1 | 8799.74, m1, d1 |
| kelt17/kelt17_dt | shipped | 8649.97, m1 | 8649.97, m1, d1 | 8649.97, m1, d1 |
| kelt17/kelt17_fast | shipped | 16948, m0 | 16948, m0, d1 | 16948, m0, d1 |
| kelt17/kelt17_rm | shipped | -23777, m1 | -23777, m1, d1 | -23777, m1, d1 |
| kelt4/kelt4 | shipped | 83423.2, m0 | 83423.2, m0 | 83423.2, m0 |
| kelt4/kelt4_rv+transit+sed | shipped | 84384.5, m0 | 84384.5, m0 | 84384.5, m0 |
| kelt4/kelt4_rvonly | shipped | -601.08, m0 | -601.08, m0 | -601.08, m0 |
| KMT-2019-BLG-1806/KMT-2019-BLG-1806 | shipped | 45503.4, m1 | 45504.1 (+0.693), m1 | 45420.9 (-82.6), m0 |
| KMT-2021-BLG-1122L/KMT-2021-BLG-1122L | shipped | -10807.8, m0 | -10807.9 (-0.106), m0 | -10807.9 (-0.074), m0 |
| ob07224/ob07224 | shipped | 23136.8, m3 | 23145.4 (+8.61), m2, d3 | 23145.4 (+8.61), m2, d2 |
| ob08092/ob08092 | shipped | 6243.19, m1 | 6243.19, m1 | 6252.18 (+8.99), m0 |
| ob09020/ob09020 | shipped | 36499.8, m5 | -973974 (-1.01e+06), m4, d2 | -892087 (-9.29e+05), m2, d2 |
| ob09020/ob09020_linear | shipped | 33249.5, m5 | 2259.65 (-3.1e+04), m4, d2 | 15311.6 (-1.79e+04), m2, d2 |
| ob140939/ob140939 | shipped | -1549.53, m0 | -1549.53, m0 | -1233.58 (+316), m0 |
| ob140939/ob140939_today | shipped | -2294.27, m0 | -2294.27, m0, d1 | -1978.42 (+316), m0, d1 |
| ob161003/ob161003 | shipped | 79403.2, m0 | 79403.2, m0 | 79400.8 (-2.39), m0 |
| ob161045/ob161045 | shipped | 21063.3, m0 | 21063.3, m0 | 21063.4 (+0.0453), m0 |
| ob170114/ob170114 | overdet | -74625.8, m3 | -74625.6 (+0.194), m3, d1 | 5939.96 (+8.06e+04), m0, d1 |
| ob170114/ob170114 | shipped | 5939.36, m0 | 5939.68 (+0.323), m0 | 5940.06 (+0.695), m0 |
| ob170114/ob170114 | tp | 5939.08, m0 | 5939.4 (+0.323), m0 | 5939.78 (+0.695), m0 |
| ogle0383/OGLE_0383LD | shipped | 26734.5, m1 | 26735.2 (+0.688), m1 | 26739.7 (+5.15), m0 |
| theophylline/theophylline | shipped | -619.722, m0 | -619.722, m0 | -619.722, m0 |
| toi3919/toi3919 | shipped | 8217.45, m0 | 8217.45, m0, c2 | 8217.45, m0, c2 |
| wasp18/wasp18 | shipped | 83488.9, m0 | 83488.9, m0 | 83488.9, m0 |

Reading it (the rule alone):

- The negative companion mass is gone (`rank22`: -912820 -> 3065; the host's
  logmass absorbs, as the tier order says).
- 27 of the 38 shipped-config rows are bit-identical.  The +0.32..0.69 nat rows
  (DC2018_128, KMT-2019-BLG-1806, kb180087_obj3, OGLE_0383LD, ob170114) are one
  constant: the soft lower bound on the DERIVED `lens.q` takes its preliminary
  steepness from q's init_scale, which is now propagated from the body masses
  where there was none; the start sits far from the bound, so that term goes
  from -log 2 to 0.  `whitening.measure_barrier_scales` re-measures it before
  any sampling.
- ob07224 +8.6 nats, misses 3 -> 2: its pinned rho is now delivered.
- hat3_transitonly +608: the sampled V_c/V_e and the omega direction start at
  their own defaults instead of at values back-solved from the derived
  sqrt(e) pair's defaults.
- hat3 and toi3919 report their two `teffsed = teff` / `radiussed = radius`
  pins as contradictions; hat3's `star.mass` and `star.parallax` (derived) are
  dropped -- both used to be held in the ledger while the model started
  elsewhere.  The kelt17 configs drop `planet.p` (derived; the radius and the
  stellar radius it is the ratio of are both pinned).
- ob09020 (and its linear-orbit sibling) regress by ~1e6 nats at this state,
  as the engine proposal predicted: the rule honors its pinned t_E and pi_E_N
  exactly, through HELIOCENTRIC relations the model does not evaluate.  The
  second commit (8.6.23) makes the relations geocentric; ob09020's pins are
  for JDE to rule on after the ordering sweep.
- The `density` scenario: the radius absorbs and the MIST-backed mass keeps its
  seed; the density is delivered (m 1 -> 0) because its relation now carries
  the 3/(4 pi) of a mean density, as the model's `calc_density` does.


Reading it (with the geocentric relations, the second commit):

- Pinned t_E values are now delivered exactly: ob08092 +9.0 nats (m 1 -> 0),
  ob140939 and ob140939_today +316, OGLE_0383LD +5.2 (m 1 -> 0), and the
  `overdet` scenario +8.1e4 (m 3 -> 0): ob170114 with its source proper
  motions removed and t_E/pi_E pinned now starts within 0.1 nat of the
  shipped config.
- KMT-2019-BLG-1806 -82.6 (m 1 -> 0): its params file pins t_E = 118.2 d,
  which the old engine never delivered (the model started at 122.3 d).  The
  rule now delivers it, and the data prefer the old 122.3 d by 82.6 nats --
  a params-file value to re-tune, not an engine effect.
- ob161003 -2.4 nats, DC2018_128 +0.4, kb180087_obj3 +1.0, ob170114 +0.4:
  the same seeds evaluated in the frame the model uses.
- ob09020 (and `_linear`): t_E and pi_E_N are kept, theta_E = 2.95 and
  pi_E_E = 0.149 are dropped; the start is still ~9e5 nats below master,
  which delivered none of its five pins.  Its pins are mutually
  inconsistent with its masses and distances under every ordering in
  section C; they need a ruling, not an engine change.

## B. Sensitivity to init_scale

Every DECLARED init_scale the engine sees multiplied by its own factor
10**U(-3, 3) (one factor per path, three independent trials; propagated
scales follow from the perturbed ones), the solved starts compared with the
unperturbed run.  Cells: how many start values moved by more than 1%, and
the largest relative move.  Every row not listed moved nothing in any trial.

| config | trial 0 | trial 1 | trial 2 |
|---|---|---|---|
| hat3_mist density | 0 | 1 (max 0.28 star.0.eep) | 1 (max 0.28 star.0.eep) |
| kelt4_rvonly | 0 | 0 | 8 (max 13 orbit.0.esinw) |
| ob09020 | 4 (max 3.1 star.2.pm_dec) | 4 (max 1.6 star.2.pm_ra) | 4 (max 1.1 star.2.pm_ra) |
| ob09020_linear | 4 (max 3.1 star.2.pm_dec) | 4 (max 1.6 star.2.pm_ra) | 4 (max 1.1 star.2.pm_ra) |

On 37 of the 41 rows no start moves.  Three of the four that do split a
correction INSIDE one tier -- the only place the rule lets a scale act.
ob09020's four DEFAULT_WEAK proper motions share the t_E/pi_E_N
correction (min-norm in scale units, so the split follows the scales); in
kelt4_rvonly trial 2 the DEFAULT msini start's scale fell by ~1e3 and the
secosw/sesinw defaults (0.01) took a share of its correction (e 0.0002 ->
0.002).  The fourth is not the rule's: in the hat3_mist `density` scenario
the EEP (702 -> 505) is the MIST standalone solver's answer, and that
solver's own search starts from a scale-dependent guess.  Which value
absorbs never changed.  The old engine's choice of
absorber was itself scale-weighted, so the same perturbation re-routed
whole corrections there; this is the measurement behind "no WLS" (Q4).

## C. Ordering sweep

Start logp with both commits under each alternative ordering, as the change
from the current ladder (`=` unchanged), and the check_user_starts misses.
O0 current; O1 lens and source distance levels swapped (lens 30, source
25); O2 both distances above the data hints (62/65); O3 DEFAULT_WEAK merged
into DEFAULT (20); O4/O5/O6 LIKELIHOOD_BACKED just above DEFAULT (21), equal
to DERIVED_DATA (60), just below USER (99); O7 keep-order by params-file
position only; O8 keep-order by REVERSE depth.
| config | O0 | O1 | O2 | O3 | O4 | O5 | O6 | O7 | O8 |
|---|---|---|---|---|---|---|---|---|---|
| DC2018_128 rank22 | 3065.94 m0 | = m0 | = m0 | = m0 | = m0 | = m0 | = m0 | = m0 | = m0 |
| DC2018_128 | 3374.96 m0 | -8.23 m0 | -50.1 m0 | = m0 | = m0 | = m0 | = m0 | = m0 | = m0 |
| gm | -1.14765 m0 | = m0 | = m0 | = m0 | = m0 | = m0 | = m0 | = m0 | = m0 |
| hat3 | 272792 m0 | = m0 | = m0 | = m0 | = m0 | = m0 | = m0 | = m0 | = m0 |
| hat3_mist density | -790.418 m0 | = m0 | = m0 | = m0 | = m0 | = m0 | = m0 | = m0 | = m0 |
| hat3_mist | -49.3124 m0 | = m0 | = m0 | = m0 | = m0 | = m0 | = m0 | = m0 | = m0 |
| hat3_staronly | -63.6153 m0 | = m0 | = m0 | = m0 | = m0 | = m0 | = m0 | = m0 | = m0 |
| kb180087_obj3 | 56832.2 m0 | -8.77 m0 | -45 m0 | = m0 | = m0 | = m0 | = m0 | = m0 | = m0 |
| kelt4 | 83423.2 m0 | = m0 | = m0 | = m0 | = m0 | = m0 | = m0 | = m0 | = m0 |
| KMT-2019-BLG-1806 | 45420.9 m0 | -3.47e+04 m1 | +102 m0 | = m0 | = m0 | = m0 | = m0 | = m0 | = m0 |
| KMT-2021-BLG-1122L | -10807.9 m0 | = m0 | = m0 | = m0 | = m0 | = m0 | = m0 | = m0 | = m0 |
| ob07224 | 23145.4 m2 | = m2 | = m2 | = m2 | = m2 | = m2 | = m2 | = m2 | = m2 |
| ob08092 | 6252.18 m0 | -5.76 m0 | -3.26 m0 | = m0 | = m0 | = m0 | = m0 | = m0 | = m0 |
| ob09020 | -892087 m2 | = m2 | = m2 | = m2 | = m2 | = m2 | = m2 | = m2 | +3.42e+05 m2 |
| ob09020_linear | 15311.6 m2 | = m2 | = m2 | = m2 | = m2 | = m2 | = m2 | = m2 | +2.9e+03 m2 |
| ob140939 | -1233.58 m0 | -4.62e+03 m0 | +2.02 m0 | = m0 | = m0 | = m0 | = m0 | = m0 | = m0 |
| ob140939_today | -1978.42 m0 | = m0 | +2.65 m0 | = m0 | = m0 | = m0 | = m0 | = m0 | = m0 |
| ob161003 | 79400.8 m0 | = m0 | = m0 | = m0 | = m0 | = m0 | = m0 | = m0 | = m0 |
| ob161045 | 21063.4 m0 | = m0 | = m0 | = m0 | = m0 | = m0 | = m0 | = m0 | = m0 |
| ob170114 overdet | 5939.96 m0 | = m0 | = m0 | = m0 | = m0 | = m0 | = m0 | = m0 | = m0 |
| ob170114 | 5940.06 m0 | = m0 | = m0 | = m0 | = m0 | = m0 | = m0 | = m0 | = m0 |
| OGLE_0383LD | 26739.7 m0 | -7.75 m0 | -242 m0 | = m0 | = m0 | = m0 | = m0 | = m0 | = m0 |
| toi3919 | 8217.45 m0 | = m0 | = m0 | = m0 | = m0 | = m0 | = m0 | = m0 | = m0 |

Reading it:

- The current ladder (O0) is best or tied on every row but one.  Swapping
  the two distance levels (O1) costs up to 3.5e4 nats (KMT-2019-BLG-1806,
  which then misses its t_E) and 4.6e3 (ob140939); lifting both distances
  above the data hints (O2) costs up to 242 (OGLE_0383LD), but buys +102 on
  KMT-2019-BLG-1806.
- Merging DEFAULT_WEAK into DEFAULT (O3) changes nothing measured here: no
  shipped config pins a quantity whose only absorbers are a weak and a plain
  default.
- The LIKELIHOOD_BACKED slot (O4-O6) changes nothing on any row: no
  shipped config, nor the `density` scenario, has a backed seed in the
  absorbing tier for any slot from 21 to 99.  The slot is not decided by
  these data.
- Keep-order: params-file order (O7) keeps the same pins as depth order
  everywhere.  Reverse depth (O8) is the only alternative that changes
  ob09020: it keeps all three of t_E, pi_E_N, pi_E_E and drops theta_E and
  the lens mass instead (+3.4e5 nats over O0, still ~5.5e5 below master);
  with depth order (O0) it keeps the mass and t_E/pi_E_N and drops theta_E
  and pi_E_E.

## After JDE's 2026-10-14 rulings

Re-measured on master 4e123970 and on this PR's head, each with its own
`examples/` (this PR changes `ob09020`, `ob09020_linear` and
`KMT-2019-BLG-1806`'s params files; master had meanwhile changed `hat3`'s
limb-darkening priors).  The three rulings:

- **K and the planet mass are one quantity.**  The RV semi-amplitude
  estimate stays a hint on `planet.K`, and the sampled mass coordinate
  (`planet.mass`, or `planet.msini` under m sin i coordinates, now
  DEFAULT_WEAK like the mass) is the lowest seed K reads, so it absorbs the
  hint alone; a user value on any mass coordinate supersedes the hint
  (solve diagnostics, kind "superseded").  Every one of the 22 shipped
  configs with an RV planet starts at master's cos i, and so do the
  `hierarchical_orbit` and `taylor_orbit` test systems, whose cos i / e
  pins are gone again.
- **ob09020**: Skowron's `t_E`, `theta_E` and `pi_E` are no longer pinned;
  the lens primary's proper motion is seeded (README).
- **KMT-2019-BLG-1806**: Zang+2023 Table 4 "Outer" u_0 > 0 replaces our
  own static MMEXOFAST solution.  With it the settle was found to leave a
  correction's component no check sees (the common motion of the lens and
  source proper motions, +25,027 mas/yr); it now projects that out.

### A. Start logp


| config | scenario | master (4e123970) | this PR |
|---|---|---|---|
| astrometry_sim/simbinary | shipped | -247.616, m0 | -247.616, m0 |
| DC2018_128/DC2018_128 | rank22 | -912820, m0, **neg1** | 3065.94 (+9.16e+05), m0 |
| DC2018_128/DC2018_128 | shipped | 3373.88, m0 | 3374.96 (+1.08), m0 |
| GaiaBH1/gaiabh1 | shipped | -280.186, m0 | -280.186, m0 |
| galactic_model/gm | shipped | -1.14765, m0 | -1.14765, m0 |
| gj1214/gj1214 | shipped | 108478, m0 | 108478, m0 |
| hat3/hat3 | shipped | 272748, m2 | 272748, m2, d2, c2 |
| hat3/hat3_mist | density | -1706.07, m1 | -790.418 (+916), m0 |
| hat3/hat3_mist | shipped | -49.3124, m0 | -49.3124, m0 |
| hat3/hat3_staronly | shipped | -63.6153, m0 | -63.6153, m0 |
| hat3/hat3_transitonly | shipped | 257824, m0 | 258432 (+608), m0 |
| hd80606/hd80606 | shipped | -132978, m0 | -132978, m0 |
| hd80606/hd80606_rvonly | shipped | -125231, m0 | -125231, m0 |
| hd80606/hd80606_transitonly | shipped | -7861.49, m0 | -7861.49, m0 |
| HIP1349/hip1349 | shipped | -342.896, m0 | -342.896, m0 |
| kb180087_obj3/kb180087_obj3 | shipped | 56830.6, m0 | 56832.2 (+1.61), m0 |
| kelt17/kelt17 | shipped | -1458.24, m0 | -1458.24, m0 |
| kelt17/kelt17_all | shipped | 8799.74, m1 | 8799.74, m1, d1 |
| kelt17/kelt17_dt | shipped | 8649.97, m1 | 8649.97, m1, d1 |
| kelt17/kelt17_fast | shipped | 16948, m0 | 16948, m0, d1 |
| kelt17/kelt17_rm | shipped | -23777, m1 | -23777, m1, d1 |
| kelt4/kelt4 | shipped | 83423.2, m0 | 83423.2, m0 |
| kelt4/kelt4_rv+transit+sed | shipped | 84384.5, m0 | 84384.5, m0 |
| kelt4/kelt4_rvonly | shipped | -601.08, m0 | -601.08, m0 |
| KMT-2019-BLG-1806/KMT-2019-BLG-1806 | shipped | 45503.4, m1 | 45541.9 (+38.4), m0 |
| KMT-2021-BLG-1122L/KMT-2021-BLG-1122L | shipped | -10807.8, m0 | -10807.9 (-0.074), m0 |
| ob07224/ob07224 | shipped | 23136.8, m3 | 23145.4 (+8.61), m2, d2 |
| ob08092/ob08092 | shipped | 6243.19, m1 | 6252.18 (+8.99), m0 |
| ob09020/ob09020 | shipped | 36499.8, m5 | 36530.5 (+30.7), m0 |
| ob09020/ob09020_linear | shipped | 33249.5, m5 | 39048.7 (+5.8e+03), m0 |
| ob140939/ob140939 | shipped | -1549.53, m0 | -1233.58 (+316), m0 |
| ob140939/ob140939_today | shipped | -2294.27, m0 | -1978.42 (+316), m0, d1 |
| ob161003/ob161003 | shipped | 79403.2, m0 | 79400.8 (-2.39), m0 |
| ob161045/ob161045 | shipped | 21063.3, m0 | 21063.4 (+0.0453), m0 |
| ob170114/ob170114 | overdet | -74625.8, m3 | 5939.96 (+8.06e+04), m0, d1 |
| ob170114/ob170114 | shipped | 5939.36, m0 | 5940.06 (+0.695), m0 |
| ob170114/ob170114 | tp | 5939.08, m0 | 5939.78 (+0.695), m0 |
| ogle0383/OGLE_0383LD | shipped | 26734.5, m1 | 26739.7 (+5.15), m0 |
| theophylline/theophylline | shipped | -619.722, m0 | -619.722, m0 |
| toi3919/toi3919 | shipped | 8217.45, m0 | 8217.45, m0, c2 |
| wasp18/wasp18 | shipped | 83488.9, m0 | 83488.9, m0 |

The ob09020 rows: master delivered none of its five pins (m5); the rule
with them pinned started at -892,087 / +15,312.  KMT-2019-BLG-1806:
+45,420.9 with the MMEXOFAST set delivered exactly, +45,425.9 with its t_E
unpinned, +45,541.9 with Zang's.  hat3: the two misses are master's new
limb-darkening priors against the params file's own starts, on both
sides; the d2/c2 are the same `star.mass`/`star.parallax` drops and
`teffsed`/`radiussed` contradictions as in section A above.

### B. Sensitivity to init_scale (rows that move)

| config | trial 0 | trial 1 | trial 2 |
|---|---|---|---|
| KMT-2019-BLG-1806 | 3 (max 0.06 star.1.pm_ra) | 4 (max 0.079 star.1.pm_dec) | 4 (max 0.092 star.1.pm_dec) |
| hat3_mist density | 0 | 1 (max 0.28 star.0.eep) | 1 (max 0.28 star.0.eep) |

KMT-2019-BLG-1806's four DEFAULT_WEAK proper motions share their
correction in proportion to their scales, as the rule says (at most 0.09
relative, i.e. ~0.3 mas/yr); the kelt4_rvonly msini/e split of section B
above is gone (the K hint's only absorber is now the mass).  Every other
row moves nothing.

### C. Ordering sweep (rows the rulings changed)

| config | O0 | O1 | O2 | O3 | O4 | O5 | O6 | O7 | O8 |
|---|---|---|---|---|---|---|---|---|---|
| KMT-2019-BLG-1806 | 45541.9 m0 | -581 m0 | +4.8 m0 | = m0 | = m0 | = m0 | = m0 | = m0 | = m0 |
| ob09020 | 36530.5 m0 | = m0 | = m0 | = m0 | = m0 | = m0 | = m0 | = m0 | = m0 |
| ob09020_linear | 39048.7 m0 | = m0 | = m0 | = m0 | = m0 | = m0 | = m0 | = m0 | = m0 |

ob09020 keeps every pin under every ordering now (it no longer pins the
contradictory values).  KMT-2019-BLG-1806 still prefers the current
ladder except for O2 (+4.8 nats with both distances above the data
hints), and swapping the lens and source distance levels costs 581.


## After JDE's 2026-10-15 rulings (paper values)

Re-measured at the new head against master 4e123970.  Only three rows
changed; every other row of table A, every row of table B and every
other row of table C is identical to the section above.

| config | master | before | now |
|---|---|---|---|
| ob09020 | +36,499.8, m5 | +36,530.5, m0 | +33,540.2, m0 |
| ob09020_linear | +33,249.5, m5 | +39,048.7, m0 | +47,695.7, m0 |
| kelt17 | -1,458.2, m0 | -1,458.2, m0 | +665.4, m0 |

- ob09020: every seed is now printed in Skowron+2011, Yee+2016 or Gaia
  DR3, or derived from those with the arithmetic written out (README).
  The keplerian start is 2,960 nats below master's, whose seeds were
  proper motions the old engine derived; the linear one is 14,450 above
  (both files now take every Skowron+2011 value from the one "parallax +
  full orbit (with priors)" column; the linear file's earlier "2 par.
  motion" seeds gave +44,852.6).  At the keplerian start every printed
  value is reproduced exactly (t_0, u_0, rho, q, t_E = 76.900, theta_E,
  pi_E, alpha_0 = 189.080 deg); the only miss is s_0 = 0.42633 against
  S11's 0.4294 (-0.7%), which comes from combining Y16's orbit and D_L
  with S11's theta_E, not from an unsolved relation.
- An ob09020 fit from the keplerian start (scratch copy, 34 cores, PTDE)
  was stopped by request during the seed polish: round 1 of the DE polish
  took the start from +33,540.2 to +33,818.6 (579 sweeps, ~9 min, stopped
  on rate); round 2 was re-whitening at the polished point when it was
  stopped, ~25 min in.  No params.2 file is shipped (deferred).
- kelt17: the seeds are Zhou+2016's (the stellar mass was missing).
- The stage 1-3 probe now reads registered components' roles; no shipped
  config's start changed with it.
- Ordering sweep, kelt17 and both ob09020 rows: identical under all nine
  orderings.
