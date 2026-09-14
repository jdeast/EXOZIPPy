"""Generate ONE identically-shaped fit config per DC2018 event.

THE POINT IS THAT NOTHING HERE IS PER-EVENT TUNED.  Every number this
writes comes from a rule applied the same way to every event, and the rules
are named below.  A sweep whose configs were each nudged into working is a
sweep whose results mean nothing, so the only per-event inputs are
measurements: the coordinates, the line-of-sight extinction, the data
files, and the PSPL seed our own peak finder produced.

THE RECIPE IS v7's, which is the only microlensing configuration in this
repo that has been validated against truth end to end (review 8.6.7): the
three observable coordinate swaps (fitmurel / fitpirel / fitthetae) on the
event, the SED relinked to rho (`star_constrains_rho: True`), stellar teff
and radius UNPINNED with feh at N(0, 0.5), and the Hogg mixture on both
light curves.  On DC2018-128 that combination moved mu_rel from 10.83 to
1.469 against a truth of 1.814 and put five of seven physical quantities
inside their 95% intervals.

WHAT DELIBERATELY IS NOT CARRIED OVER FROM v7:
  * `mulensevent.log_theta_E: {initval: -1.27}`.  That start was read off
    v5's fitted theta_star / rho -- i.e. it used the answer.  A blind sweep
    cannot, so it is dropped and the relaxation engine plus the seed polish
    have to find theta_E themselves.  b88d791a (whitening re-centred on the
    polished start) landed for exactly this kind of case, so this run also
    tests it.  IF THE SWEEP FAILS ON 128, WHERE v7 SUCCEEDED, THE START IS
    THE FIRST SUSPECT.
  * MMEXOFAST.  `mmexofast:` points at dc18_seed.py's output instead --
    same JSON contract, a PSPL peak fit rather than a binary-lens search,
    and s/q/alpha left at defaults.yaml generics so the sampler has to find
    the planet on its own.
  * The solar pins run_event.py still writes (`star.Lens.teff: sigma: 0`
    and friends).  2.9.10 showed those force the lens mass through the SED
    floor potentials; v5 removed them and the lens mass went from 1.98x
    truth to 0.85x.

THE RULES, one line each:
  ra, dec           event_info.txt columns 3-4.
  star.Source.av    event_info.txt columns 6-7 (the red-clump A_W149 and
                    its dispersion along that line of sight).  NOTE: this
                    is what v7 does and it is kept for comparability, but
                    A_W149 is not A_V -- see the caveat by AV_COLS below.
  out_scale         upper = 10x the per-LC median flux error, initval = 1x,
                    in the file's flux system (review 8.6.3, RULED).
  zeropoint         N(22.0, 0.02) -- the simulation's, identical for every
                    event because it is the same instrument.
  u1                pinned to 0: no limb darkening is simulated.
  everything else   defaults.yaml.
"""

import argparse
import io
import json
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dc18_common as C  # noqa: E402

DATA = "/home/jeastman/python/MMEXOFAST/data/2018DataChallenge"
BANDS = [("W149", "Roman/WFI.F146"), ("Z087", "Roman/WFI.F087")]

# event_info.txt columns, 0-based: 5 = A_W149, 6 = its dispersion.
# CAVEAT, stated so nobody has to rediscover it: the SED's parameter is
# `av`, i.e. A_V, and A_W149 is roughly a fifth of it.  v7 nonetheless puts
# A_W149 there, and v7 is the arm that was validated against truth, so the
# sweep does the same rather than changing two things at once.  A shared
# systematic across all six events is something the truth table can find;
# six differently-extincted events cannot be compared at all.
AV_COLS = (5, 6)


def event_info_row(event):
    rows = np.genfromtxt(
        os.path.join(DATA, "event_info.txt"), dtype=None, encoding="utf-8"
    )
    for r in rows:
        if int(r[1]) == int(event):
            return r
    raise ValueError("event %s not in event_info.txt" % event)


def median_flux_err(path):
    e = np.loadtxt(path)[:, 2]
    return float(np.median(e[np.isfinite(e) & (e > 0)]))


def build(event, outdir, draws, tune, cores, t_max):
    ev3 = "%03d" % int(event)
    name = "DC2018_%s" % ev3
    row = event_info_row(event)
    ra, dec = float(row[2]), float(row[3])
    av_mu, av_sd = float(row[AV_COLS[0]]), float(row[AV_COLS[1]])

    files = {
        b: os.path.join(DATA, "n20180816.%s.WFIRST18.%s.txt" % (b, ev3))
        for b, _ in BANDS
    }
    for b, p in files.items():
        if not os.path.exists(p):
            raise SystemExit("missing light curve: %s" % p)

    seed = os.path.abspath("events/%s/%s_seed.json" % (ev3, name))
    if not os.path.exists(seed):
        raise SystemExit(
            "no seed for %s -- run: python dc18_seed.py %s" % (ev3, int(event))
        )

    base = os.path.abspath(os.path.join(outdir, ev3))
    os.makedirs(base, exist_ok=True)

    # The SED block: no catalog photometry.  These are simulated light
    # curves and the challenge released none, so the SED constrains the
    # source ONLY through the calibrated microlensing baseline flux (the
    # per-lightcurve zeropoint hook) with the Roman BC grids pulled in via
    # each band's `filter:`.  This is JDE's axis 3 -- the absolute source
    # flux constraining its radius -- and it is what v7 validated.
    sed_path = os.path.join(base, "%s.sed.yaml" % name)
    io.open(sed_path, "w", encoding="utf-8").write(
        "model: NextGen\nfilters: []\n"
    )

    cfg = {
        "run": {"name": name},
        "star": [{"name": "Lens"}, {"name": "Source"}],
        "planet": [{"name": "Companion"}],
        "mulensevent": [
            {
                "finite_source": True,
                "fitmurel": True,
                "fitpirel": True,
                "fitthetae": True,
                "mmexofast": seed,
            }
        ],
        "lens": [{"body": "star.Lens"}, {"body": "planet.Companion"}],
        "source": [{"body": "star.Source", "star_constrains_rho": True}],
        "galacticmodel": [{"name": name, "anchor_idx": 1}],
        "band": (
            [{"name": b, "filter": f, "ld_law": "linear"} for b, f in BANDS]
            + [
                {
                    "name": "Ks_bcgrid",
                    "filter": "2MASS/2MASS.Ks",
                    "ld_law": "linear",
                }
            ]
        ),
        "mulensinstrument": [
            {
                "name": "Roman_%s" % b,
                "file": files[b],
                "data_format": "flux",
                "observer_location": "roman_simulated_2018dc",
                "band": b,
                "likelihood": "hogg",
            }
            for b, _ in BANDS
        ],
        "prefix": os.path.join(base, name),
        "sed": {"file": sed_path},
        "torres": [{"star": "Source", "constrain": ["mass", "radius"]}],
        "mann": [
            {
                "star": "Lens",
                "ks": "synthetic",
                "constrain": ["mass", "radius"],
            }
        ],
        "sampler": {
            "method": "ptde_async",
            "cores": cores,
            "n_temps": "auto",
            "T_max": t_max,
            "tune": tune,
            "draws": draws,
            "nthin": 1,
            "recompute_trace": True,
            "eval_timeout": 10,
            # Retain the hot rungs: outputs.ledger.discover_hot_modes
            # clusters them, and finding ALL the modes -- not just the one
            # the cold chains settled in -- is half of what this sweep is
            # for.  Explicit rather than "auto" so the sweep does not
            # silently lose mode discovery on an event where auto declines.
            "store_hot_chains": True,
        },
        "modes": {"weights": "evidence"},
        "parameter_file": os.path.join(base, "%s.params.yaml" % name),
    }

    params = {
        "star.Lens.ra": {"initval": ra, "sigma": 0},
        "star.Lens.dec": {"initval": dec, "sigma": 0},
        # Generic, NOT solar: teff and radius stay FREE (2.9.10), and feh
        # gets the broad prior JDE specified rather than a pin.
        "star.Lens.feh": {"mu": 0.0, "sigma": 0.5},
        # Not a microlensing observable and not used by anything here;
        # pinning it keeps it out of the sampled dimension count.
        "planet.Companion.radius": {"sigma": 0},
        "star.Source.av": {"mu": av_mu, "sigma": av_sd},
        # No limb darkening is present in the simulation.
        "band.W149.u1": {"initval": 0.0, "sigma": 0},
        "band.Z087.u1": {"initval": 0.0, "sigma": 0},
        # 2.9.11: with `filters: []` there is no photometry for errscale to
        # scale, so left free it samples its U(0.001, 1000) prior and
        # reports a median of ~500 that reads like a measurement.  Pin it.
        "sed.errscale": {"initval": 1.0, "sigma": 0},
    }
    for b, _ in BANDS:
        inst = "mulensinstrument.Roman_%s" % b
        med = median_flux_err(files[b])
        params["%s.zeropoint" % inst] = {"mu": 22.0, "sigma": 0.02}
        params["%s.out_scale" % inst] = {"upper": 10.0 * med, "initval": med}

    io.open(cfg["parameter_file"], "w", encoding="utf-8").write(
        yaml.safe_dump(params, sort_keys=True, default_flow_style=False)
    )
    cfg_path = os.path.join(base, "%s.yaml" % name)
    io.open(cfg_path, "w", encoding="utf-8").write(
        yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False)
    )

    s = json.load(io.open(seed))["fits"][0]["parameters"]
    print(
        "%s  ra=%.4f dec=%.4f  av=%.2f+/-%.2f  seed t_0=%.3f u_0=%.4f "
        "t_E=%.2f\n    -> %s"
        % (name, ra, dec, av_mu, av_sd, s["t_0"], s["u_0"], s["t_E"], cfg_path)
    )
    return cfg_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("events", nargs="+", type=int)
    ap.add_argument("--outdir", default="sweep")
    ap.add_argument("--draws", type=int, default=50000)
    ap.add_argument("--tune", type=int, default=5000)
    ap.add_argument("--cores", type=int, default=64)
    ap.add_argument("--t-max", type=float, default=200.0)
    a = ap.parse_args()
    for ev in a.events:
        build(ev, a.outdir, a.draws, a.tune, a.cores, a.t_max)


if __name__ == "__main__":
    main()
