# EXOZIPPy examples

Each subdirectory is a worked fit: a system config (`<name>.yaml`), a
parameter file (`<name>.params.yaml`, named by the config's
`parameter_file:` key), the data, and usually a README with the
provenance of every number. They double as the regression targets of the
test suite (`docs/testing.md`), so they are kept building on master.

## Running one

```bash
cd examples/<dir> && poetry run exozippy <config>.yaml
# e.g.
cd examples/kelt4 && poetry run exozippy kelt4_rvonly.yaml
```

**A fit run in place writes into the example directory**: the trace,
tables, plots and logs under the config's `prefix:` (usually
`fitresults/`), and `mkparam` writes a restart parameter file next to the
config (`<name>.params.2.yaml`, overwriting any earlier one). For a
throwaway run, copy the directory somewhere scratch and run it there:

```bash
cp -r examples/hat3 /path/to/scratch/ && cd /path/to/scratch/hat3
poetry run exozippy hat3.yaml
```

A parameter file is a template: published values with their citations,
not numbers read off an earlier run of the same fit. The reasoning behind
a value (a unit conversion, a convention mapping, why a start is seeded
the way it is) lives in the example's README.

## Exoplanets, stars and binaries

| directory | system | what it demonstrates |
|---|---|---|
| `hat3` | HAT-P-3 b, a hot Jupiter with four TESS sectors and HIRES RVs | The standard transit + RV + SED fit with the Torres relations (`hat3.yaml`), and its pieces: SED-only star (`hat3_staronly`), star + MIST evolutionary tracks (`hat3_mist`), transit-only (`hat3_transitonly`) and the full fit with MIST plus the Claret limb-darkening tie (`hat3_mist_claret`). |
| `hd80606` | HD 80606 b, e ~ 0.93 on a 111-day orbit | The high-eccentricity validation: transit + RV (`hd80606.yaml`), RV-only with the `fitmsini` (m sin i, cos i) default (`hd80606_rvonly`), and transit-only with the V_c/V_e transit-chord (photo-eccentric) coordinates and a published period prior (`hd80606_transitonly`); MIST + SED throughout. |
| `kelt4` | KELT-4A b, a hot Jupiter in a hierarchical triple with the KELT-4BC pair | RV-only (`kelt4_rvonly`); transit + RV + three-star SED with the Torres/Mann relations (`kelt4_rv+transit+sed`); and the full hierarchy (`kelt4.yaml`) -- nested orbits (b around A, B around C, BC around A) with relative astrometry of both pairs. |
| `kelt17` | KELT-17 b, a retrograde hot Jupiter around a fast-rotating A star | The Rossiter-McLaughlin effect (`kelt17.yaml`, `kelt17_rm`), Doppler tomography (`kelt17_fast`, `kelt17_dt`) and both jointly (`kelt17_all`), converted 1:1 from the EXOFASTv2 example. |
| `toi3919` | TOI-3919 b, an eccentric 7.4-day giant with TESS + ground transits and TRES RVs | An RV trend written as a `type: linear` orbit (`orbit.trend.gammadot`); `toi3919_claret` adds the Claret limb-darkening tie. Converted from an EXOFASTv2 fit. |
| `wasp18` | WASP-18 b, a ~10 M_J planet on a 0.94-day orbit | A TESS phase curve: Doppler beaming (`fitbeam`), thermal emission, reflection and ellipsoidal variation, with CORALIE RVs, SED and Torres. |
| `gj1214` | GJ 1214 b, a sub-Neptune around an M dwarf | Recreates Mahajan & Eastman (2024): JWST MIRI LRS transits and occultations (thermal emission, spline detrending), ground transits, HARPS RVs, SED and the Mann M-dwarf mass relation. |
| `GaiaBH1` | Gaia BH1, a G star orbiting a dark 9.6 M_sun companion | Joint real RVs + simulated Gaia epoch astrometry (`astrometryinstrument`, `mode: gaia`) with a dark companion. |
| `HIP1349` | HIP 1349, a 411-day astrometric binary | Real Hipparcos intermediate astrometric data (1-D along-scan abscissae, `mode: gaia`). |
| `astrometry_sim` | A simulated nearby G + M binary | The two ground-based astrometry modes: absolute photocenter astrometry with a light-contributing companion (`mode: abs`) and relative separation/position angle (`mode: rel`). |
| `galactic_model` | A single bulge star, no data | Sampling the galactic-model prior alone (distance, proper motion, RV) -- a prior-only check. |

## Microlensing

| directory | event | what it demonstrates |
|---|---|---|
| `ob08092` | OGLE-2008-BLG-092 (late-time subevent) | The simplest fit: a point-source point-lens (PSPL) curve on one OGLE light curve. |
| `ob140939` | OGLE-2014-BLG-0939 | Satellite parallax (Earth + Spitzer, real Horizons ephemeris) on a PSPL event, all four Refsdal-degenerate solutions as per-seed starts sampled with PTDE, and a source proper-motion prior in the bulge frame; `ob140939_today` adds a source SED, a zeropoint tie and Gaia DR3 astrometry. |
| `ob07224` | OGLE-2007-BLG-224 | Terrestrial parallax on a finite-source point lens (a thick-disk brown dwarf). Work in progress: the setup builds, but the shipped data cannot recover the parallax (see its README). |
| `ob161045` | OGLE-2016-BLG-1045 | A finite-source point lens (FSPL) with linear limb darkening: the source transits the lens, so theta_E is measured. |
| `ob161003` | OGLE-2016-BLG-1003 | A binary source and a binary lens (2S2L): two sources with their own trajectories share one lens system. |
| `kb180087_obj3` | KMT-2018-BLG-0087 | A binary-lens point-source (2L1S) fit with two competing solutions as per-seed starts. |
| `KMT-2019-BLG-1806` | KMT-2019-BLG-1806 / OGLE-2019-BLG-1250 | A 2L1S planetary event with parallax, a finite source and a source SED tied to the light curves through calibrated zeropoints (Zang et al. 2023). |
| `MOA-2008-BLG-379` | MOA-2008-BLG-379 | An image-constrained 2L1S planetary fit (Bennett et al. 2024): Keck AO and Hubble imaging taken 5-10 years later resolve the lens from the source, and the lens-source separation (`astrometryinstrument` `mode: lensrel`), the lens and source magnitudes (SED rows) and the OGLE zeropoint tie are fit jointly with the public OGLE + MOA light curves, the galactic model and MIST on both stars, over the four close/wide x u_0 seeds. |
| `KMT-2021-BLG-1122L` | KMT-2021-BLG-1122L | A triple lens (3L1S): the validation example for three or more lens bodies, I + V difference-flux photometry. |
| `ogle0383` | A KMTNet 2L1S event | A binary-lens finite-source fit with linear limb darkening on six KMTNet light curves. |
| `ob09020` | OGLE-2009-BLG-020 | A binary lens whose orbit is also measured by RVs: the companion's Keplerian orbit drives s(t), alpha(t) (`ob09020.yaml`, joint light curve + RV), and the linear orbital-motion alternative (`ob09020_linear`). Work in progress. |
| `ob170114` | OGLE-2017-BLG-0114 | A planetary 2L1S event with parallax and xallarap: the source's binary orbit as a Keplerian orbit. Work in progress. |
| `DC2018_128` | 2018 Roman (WFIRST) Data Challenge event 128 | A simulated 2L1S event observed from a Roman ephemeris, with two per-seed starts. |
| `DC2018_069` | 2018 Roman Data Challenge event 069 | The blind procedure end to end: grid search over (s, q, alpha), one fit per candidate mode, evidence and the mode report. Scripts, no shipped config. |
| `DC2018` | The 2018 Roman Data Challenge 2L1S sample | The whole challenge, one cluster job per event, seeded by the built-in peak finder, with robust (`likelihood: hogg`) light curves. Its subdirectories are generated configs for sweeps and ablations. |
| `RMDC26` | The 2026 Roman Microlensing Data Challenge | Scripts that turn finished fits into per-event evidence, relative probabilities and the challenge submission. No shipped config. |

## Other

| directory | what it is |
|---|---|
| `theophylline` | Population pharmacokinetics of theophylline: a non-astronomy component set running on the same machinery. A demonstration of the component-agnostic architecture, not a validated pharmacokinetic analysis -- read its README first. |
| `validation` | Whole-pipeline validation harnesses (inject-and-recover pull tables and diagnostics). Not a shipped example: each realization generates its own config. |

`run_example.job` runs any shipped example through the CLI on an SGE
cluster (`qsub -v DIR=<dir>,CFG=<config>.yaml run_example.job`).
