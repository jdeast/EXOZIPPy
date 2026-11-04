# Galactic-model microlensing prior vs genulens and OGLE (reviews 7.16.3, 1.7.6)

Evaluates the prior a microlensing fit uses when the light curve is
uninformative (lens + source + `galacticmodel`, `model: koshimoto21`, no
data) on bulge sight lines, and compares its rate-weighted t_E, D_L, D_S,
|mu_rel| and M_L with genulens (Koshimoto, Baba & Bennett 2021) run
unmodified, and its t_E with the efficiency-corrected OGLE-IV distributions
(Mroz et al. 2017, 2019).  The prior is evaluated by **importance
sampling**, not NUTS: NUTS never visits the black-hole lens mode.  The code
is the package `galactic_prior/` (statistics, sight lines, genulens and
OGLE readers, the model, the importance sampler, the comparison, the
figures); `validate.py` is its command line.  Method, results, tables and the
known differences: `docs/galactic_prior_validation.md`.

Runs on a workstation (one sight line: ~2-3 min to compile the model cold,
~2 min of pilot NUTS, then the importance draws over `--cores`).

```bash
cd scripts/galactic_prior_validation
# genulens 2.0.0a4 (github.com/nkoshimoto/genulens, commit 2cc5870), built
# by the user; run UNMODIFIED, default options and Dmax 20000.
python validate.py genulens --genulens <path>/genulens \
    --out-dir <scratch>/genulens
# the EXOZIPPy prior, one sight line per call (all: see --help)
PYTHONPATH=<repo>/src python validate.py sample \
    --run baade --out-dir <scratch>/port --cores 24
python validate.py compare --port-dir <scratch>/port \
    --genulens-dir <scratch>/genulens [--store]
python validate.py figures --port-dir <scratch>/port \
    --genulens-dir <scratch>/genulens --out-dir <scratch>/figures
# OGLE references (only to rebuild them), from a download of
# https://ftp.astrouw.edu.pl/ogle/ogle4/microlensing_maps/
python validate.py mroz --mroz2019-dir <download>
```

`refs/` holds the stored references:

- `mroz2017_edt4_counts.dat`, `mroz2017_edt5_eff.dat` -- Mroz et al. 2017
  (Nature 548, 183; arXiv:1707.07634) Extended Data Tables 4 and 5:
  events and detection efficiency per high-cadence field (BLG500, 501,
  504, 505, 506, 511, 512, 534, 611) in 25 bins of log t_E, 2010-2015.
  There is no per-event list, so these per-field distributions are binned.
- `mroz2019_table3_<field>.dat` -- Mroz et al. 2019 (ApJS 244, 29) Table 3
  rows (best-fit t_E and the inverse-efficiency weight) for BLG513 (Baade's
  window), BLG521 and BLG630 (OGLE-2014-BLG-0939's field).  Reconstructing
  each field's efficiency-corrected mean t_E from Table 3 reproduces the
  paper's Table 7 `t_E_mean` to 0.001 +/- 0.011 of its quoted error over
  the 64 fields with >= 30 events, so the weighting is the paper's.
- `mroz_tE_reference.json` -- per-field efficiency-corrected t_E
  quantiles and means for every field of both papers.
- `genulens_reference.json` -- genulens's weighted quantiles and their
  bootstrap errors, black-hole and lens-type shares, per sight line, for
  the `plain` (default options, Dmax 16 kpc) and `dmax20` (runtime option
  Dmax 20000) runs; REMNANT 1, default source weighting, `vEarthlb 0 0`.
- `port_is_reference.json` -- the EXOZIPPy prior's importance-sampled
  quantiles with errors and diagnostics, and the comparisons, per sight
  line.  It describes the code at the commit that wrote it; regenerate it
  with `compare --store` after any change to the galactic model.

The slow test `tests/test_galactic_prior_validation.py` importance-samples Baade's
window and holds it to the stored genulens run at a stated tolerance; its
fast tests cover the statistics and the genulens-support helpers.
