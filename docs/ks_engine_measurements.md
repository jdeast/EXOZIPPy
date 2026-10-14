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

## A. Start logp, one column per state

Cells: start logp (change from master), `m` = check_user_starts misses,
`neg` = negative masses in the ledger, `d` = user values on derived elements
DROPPED, `c` = relations the rigid user values contradict, `u` = relations a
tier could not settle.  `master` is the base of this PR; `rule` is the KS
rule with the HELIOCENTRIC t_E / pi_E relations (the first commit).

| config | scenario | master | rule |
|---|---|---|---|
| astrometry_sim/simbinary | shipped | -247.616, m0 | -247.616, m0 |
| DC2018_128/DC2018_128 | rank22 | -912820, m0, **neg1** | 3065.37 (+9.16e+05), m0 |
| DC2018_128/DC2018_128 | shipped | 3373.88, m0 | 3374.55 (+0.669), m0 |
| GaiaBH1/gaiabh1 | shipped | -280.186, m0 | -280.186, m0 |
| galactic_model/gm | shipped | -1.14765, m0 | -1.14765, m0 |
| gj1214/gj1214 | shipped | 108478, m0 | 108478, m0 |
| hat3/hat3 | shipped | 272792, m0 | 272792, m0, d2, c2 |
| hat3/hat3_mist | density | -1706.07, m1 | -790.418 (+916), m0 |
| hat3/hat3_mist | shipped | -49.3124, m0 | -49.3124, m0 |
| hat3/hat3_staronly | shipped | -63.6153, m0 | -63.6153, m0 |
| hat3/hat3_transitonly | shipped | 257824, m0 | 258432 (+608), m0 |
| hd80606/hd80606 | shipped | -132978, m0 | -132978, m0 |
| hd80606/hd80606_rvonly | shipped | -125231, m0 | -125231, m0 |
| hd80606/hd80606_transitonly | shipped | -7861.49, m0 | -7861.49, m0 |
| HIP1349/hip1349 | shipped | -342.896, m0 | -342.896, m0 |
| kb180087_obj3/kb180087_obj3 | shipped | 56830.6, m0 | 56831.2 (+0.642), m0 |
| kelt17/kelt17 | shipped | -1458.24, m0 | -1458.24, m0 |
| kelt17/kelt17_all | shipped | 8799.74, m1 | 8799.74, m1, d1 |
| kelt17/kelt17_dt | shipped | 8649.97, m1 | 8649.97, m1, d1 |
| kelt17/kelt17_fast | shipped | 16948, m0 | 16948, m0, d1 |
| kelt17/kelt17_rm | shipped | -23777, m1 | -23777, m1, d1 |
| kelt4/kelt4 | shipped | 83423.2, m0 | 83423.2, m0 |
| kelt4/kelt4_rv+transit+sed | shipped | 84384.5, m0 | 84384.5, m0 |
| kelt4/kelt4_rvonly | shipped | -601.08, m0 | -601.08, m0 |
| KMT-2019-BLG-1806/KMT-2019-BLG-1806 | shipped | 45503.4, m1 | 45504.1 (+0.693), m1 |
| KMT-2021-BLG-1122L/KMT-2021-BLG-1122L | shipped | -10807.8, m0 | -10807.9 (-0.106), m0 |
| ob07224/ob07224 | shipped | 23136.8, m3 | 23145.4 (+8.61), m2, d3 |
| ob08092/ob08092 | shipped | 6243.19, m1 | 6243.19, m1 |
| ob09020/ob09020 | shipped | 36499.8, m5 | -973974 (-1.01e+06), m4, d2 |
| ob09020/ob09020_linear | shipped | 33249.5, m5 | 2259.65 (-3.1e+04), m4, d2 |
| ob140939/ob140939 | shipped | -1549.53, m0 | -1549.53, m0 |
| ob140939/ob140939_today | shipped | -2294.27, m0 | -2294.27, m0, d1 |
| ob161003/ob161003 | shipped | 79403.2, m0 | 79403.2, m0 |
| ob161045/ob161045 | shipped | 21063.3, m0 | 21063.3, m0 |
| ob170114/ob170114 | overdet | -74625.8, m3 | -74625.6 (+0.194), m3, d1 |
| ob170114/ob170114 | shipped | 5939.36, m0 | 5939.68 (+0.323), m0 |
| ob170114/ob170114 | tp | 5939.08, m0 | 5939.4 (+0.323), m0 |
| ogle0383/OGLE_0383LD | shipped | 26734.5, m1 | 26735.2 (+0.688), m1 |
| theophylline/theophylline | shipped | -619.722, m0 | -619.722, m0 |
| toi3919/toi3919 | shipped | 8217.45, m0 | 8217.45, m0, c2 |
| wasp18/wasp18 | shipped | 83488.9, m0 | 83488.9, m0 |

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
