#!/usr/bin/env python3
"""Run the full DC2018 pipeline for one event.

    poetry run python run_event.py 128 [options]

Pipeline (one process, suitable for one cluster job):
  1. MMEXOFAST runs on BOTH bands (renormalize_errors on, binary_lens) --
     always both, even when exozippy later fits only one: the 15-min W149
     cadence is what localizes the anomaly; the sparse Z087 curve alone
     sends the binary grid search off a cliff (q -> 0). The exozippy-init
     JSON is cached at <event_dir>/<name>_mmexofast.json (delete to
     re-run).
  2. EXOZIPPy fits the selected bands, seeded by that JSON via the lens
     block's `mmexofast:` key: its solutions seed the fit (multi-seed),
     its bad-data mask (excluded_points) drops the flagged points, and its
     error factors seed err_scale. Sampling is PTDE by default; artifacts
     land under <out-dir>/<NNN>/fitresults/. (Configs that skip step 1
     still work: when a params file has no microlensing start values,
     EXOZIPPy's data-driven-hints layer runs MMEXOFAST itself -- this
     driver prefers the explicit two-step so the MMEXOFAST fit always
     sees both bands.)
  3. The posterior is compared against the challenge truth
     (comparison.csv + stdout table).

The generated config and params are also dumped to YAML in the event
directory, so any event can be re-run or debugged with the plain CLI:
    cd <out-dir>/<NNN> && exozippy DC2018_<NNN>.yaml

Everything is idempotent per event directory: the MMEXOFAST JSON is a
cache, and --recompute controls whether an existing trace is resampled.
"""

import argparse
import os
import sys
import traceback
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dc18_common as dc


def run_mmexofast_step(
    data_dir, event, name, event_dir, cores=None, quick=False, emcee=False
):
    """Step 1: MMEXOFAST on both bands; returns the cached JSON path.

    ``cores`` caps the emcee pool for the binary-lens fits -- the dominant
    cost of the MMEXOFAST stage on DC18 cadence (40 walkers x 1000 steps x
    a 39k-epoch binary-lens chi2 per evaluation is hours serial). On the
    cluster this is the job's slot grant ($NSLOTS); locally it defaults to
    a quarter of the machine.
    """
    import multiprocessing as mp

    from astropy.coordinates import SkyCoord

    from exozippy.components.mulensing import mmexofast_support

    files = dc.light_curve_files(data_dir, event, ("W149", "Z087"))
    ra, dec = dc.event_coords(data_dir, event)
    coords = SkyCoord(ra, dec, unit="deg").to_string(style="hmsdms")
    json_path = Path(event_dir) / f"{name}_mmexofast.json"
    pool = int(cores) if cores else max(2, mp.cpu_count() // 4)
    # Zero limb darkening matches MMEXOFAST's own DC18 example.
    options = {
        "limb_darkening_coeffs_gamma": {"W149": 0.0, "Z087": 0.0},
        "pool": pool,
    }
    if not emcee:
        # Default: stop after the binary-parameter ESTIMATION. The emcee
        # polish (fit_binary_lens_models) costs hours on DC18 cadence -- 40
        # walkers x 1000 steps x a 39k-epoch binary chi2, straggler-bound at
        # the ensemble barrier -- and EXOZIPPy's tempered multi-seed PTDE is
        # itself the polish: the seeds only have to land in the right basin.
        # They do, now that renormalization no longer eats the peak: the
        # once-alarming "estimator gives rho ~ 1e-6 where the polished fit
        # gives 0.0054" was an artifact of the outlier-rejection bug clipping
        # the finite-source peak (fixed in MMEXOFAST PR#7); on the fixed code
        # the raw estimator's primary solution for event 128 is s = 0.977,
        # rho = 0.0075, q = 9.4e-4 vs the polished 0.979/0.0054/1.1e-3, with
        # the alternate-basin s = 0.86 solution second in the multi-seed
        # list. initialize_exozippy() falls back to the estimator's raw
        # solutions (parameters only, no sigmas -- EXOZIPPy skips the
        # optional scale hints). Masks and error factors still come out: the
        # renormalize stage runs before this cut.
        options["stop_before"] = "fit_binary_lens:fit_binary_lens_models"
    elif quick:
        # Smoke-test emcee: upstream defaults are 40 x 1000 + 500 burn.
        # NOTE: emcee's stretch move evaluates half the ensemble per batch,
        # so parallelism beyond n_walkers/2 processes buys nothing.
        options["emcee_settings"] = {
            "n_walkers": 40,
            "n_burn": 100,
            "n_steps": 200,
        }
    mmexofast_support.run_or_load(
        json_path,
        list(files.values()),
        coords=coords,
        fit_type="binary_lens",
        renormalize_errors=True,
        no_parallax=True,
        options=options,
    )
    return json_path


def build_config(name, files, prefix, mmx_json, args):
    bands = list(files)
    config = {
        "run": {"name": name},
        "prefix": str(prefix),
        "star": [{"name": "Lens"}, {"name": "Source"}],
        "planet": [{"name": "Companion"}],
        "lens": [
            {
                "name": "Lens",
                "lenses": ["star.0", "planet.0"],
                "sources": ["star.1"],
                "finite_source": bool(args.finite_source),
                # Explicit step-1 output: seeds (stage 2, Lens) + bad-data
                # mask and error factors (stage 1a, MulensInstrument).
                "mmexofast": str(mmx_json),
            }
        ],
        "galacticmodel": [{"name": name, "anchor_idx": 1}],
        "band": [
            {
                "name": b,
                "filter": f"Roman.WFI.{b}",
                "ld_law": "linear",
            }
            for b in bands
        ],
        "mulensinstrument": [
            {
                "name": f"Roman_{b}",
                "file": files[b],
                "data_format": "flux",
                "observer_location": "roman_simulated_2018dc",
                "band": b,
                # Hogg inlier/outlier mixture on every light curve. This
                # supersedes MMEXOFAST's hard mask (excluded_points are not
                # propagated for hogg files): junk lands in the wide
                # background component instead of dragging the fit, and the
                # per-point outlier probabilities are auditable afterwards
                # (Instrument.outlier_prob_at_data).
                "likelihood": "hogg",
            }
            for b in bands
        ],
        "sampler": {
            "method": args.sampler,
            "n_temps": 8,
            "T_max": 200,
            "cores": args.cores,
            "tune": args.tune,
            "draws": args.draws,
            "nthin": 1,
            "recompute_trace": bool(args.recompute),
            "eval_timeout": 10,
        },
    }
    return config


def build_user_params(ra, dec):
    """Fixed-parameter overrides, mirroring examples/DC2018_128.

    Coordinates come from event_info.txt. The Lens's radius/teff/feh fixes
    are the same 'not constrainable without SED data' hack the DC2018_128
    example documents; the microlensing start values are deliberately
    ABSENT so the MMEXOFAST auto-initialization triggers. The Source star
    needs none of this: star.py pins its mass/teff/feh/radius/ra/dec
    automatically for any star that is purely a microlensing source (never
    also a lens body), falling back to the Lens's coordinates here.
    """
    params = {
        "star.Lens.ra": {"initval": ra, "sigma": 0},
        "star.Lens.dec": {"initval": dec, "sigma": 0},
        "star.Lens.teff": {"sigma": 0.0},
        "star.Lens.feh": {"sigma": 0.0},
        "star.Lens.radius": {"sigma": 0.0},
        "planet.Companion.radius": {"sigma": 0},
    }
    return params


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("event", type=int, help="DC18 event number (e.g. 128)")
    ap.add_argument(
        "--data-dir",
        default=None,
        help="2018DataChallenge tree (default $DC18_DATA or the MMEXOFAST "
        "source checkout)",
    )
    ap.add_argument(
        "--out-dir",
        default="events",
        help="Parent directory for per-event output (default ./events)",
    )
    ap.add_argument(
        "--bands",
        default="W149,Z087",
        help="Comma-separated bands to fit (default W149,Z087; use Z087 "
        "for a much faster fit -- ~880 points vs ~38600)",
    )
    ap.add_argument(
        "--cores",
        type=int,
        default=int(os.environ.get("NSLOTS", 0)) or None,
        help="Sampler cores; MUST match the job's slot grant on a cluster "
        "(defaults to $NSLOTS when set)",
    )
    ap.add_argument("--tune", type=int, default=5000)
    ap.add_argument("--draws", type=int, default=50000)
    ap.add_argument(
        "--sampler",
        default="ptde_async",
        help="Sampler method (default ptde_async)",
    )
    ap.add_argument(
        "--finite-source",
        action="store_true",
        help="Model finite-source effects (default off, as in DC2018_128)",
    )
    ap.add_argument(
        "--recompute",
        action="store_true",
        default=True,
        help="Resample even if a trace exists (default on)",
    )
    ap.add_argument(
        "--no-recompute",
        dest="recompute",
        action="store_false",
        help="Reuse an existing trace and only redo reports/comparison",
    )
    ap.add_argument(
        "--quick",
        action="store_true",
        help="Smoke test: Z087 only, tune=500, draws=1000",
    )
    ap.add_argument(
        "--compare-only",
        action="store_true",
        help="Skip the fit; just rebuild comparison.csv from existing output",
    )
    ap.add_argument(
        "--mmx-emcee",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run MMEXOFAST's emcee binary-lens polish (default: off -- "
        "EXOZIPPy's tempered multi-seed PTDE is the polish, and with "
        "the peak-protection fix the raw estimator seeds land in the "
        "right basin; the old rho-collapse was a renormalization "
        "clipping artifact). --mmx-emcee turns the hours-long polish "
        "back on",
    )
    args = ap.parse_args(argv)

    if args.quick:
        args.bands = "Z087"
        args.tune = min(args.tune, 500)
        args.draws = min(args.draws, 1000)

    data_dir = dc.data_dir_or_raise(args.data_dir)
    bands = [b.strip() for b in args.bands.split(",") if b.strip()]
    files = dc.light_curve_files(data_dir, args.event, bands)
    ra, dec = dc.event_coords(data_dir, args.event)

    name = f"DC2018_{args.event:03d}"
    event_dir = (Path(args.out_dir) / f"{args.event:03d}").resolve()
    event_dir.mkdir(parents=True, exist_ok=True)
    prefix = event_dir / "fitresults" / name

    status_file = event_dir / "status.txt"

    # Step 1: MMEXOFAST on both bands (cached).
    if not args.compare_only:
        status_file.write_text("running mmexofast\n")
        try:
            mmx_json = run_mmexofast_step(
                data_dir,
                args.event,
                name,
                event_dir,
                cores=args.cores,
                quick=args.quick,
                emcee=args.mmx_emcee,
            )
        except BaseException:
            status_file.write_text(
                "failed (mmexofast)\n" + traceback.format_exc()
            )
            raise
    else:
        mmx_json = event_dir / f"{name}_mmexofast.json"

    config = build_config(name, files, prefix, mmx_json, args)
    user_params = build_user_params(ra, dec)

    # Reproducibility dump: `cd <event_dir> && exozippy DC2018_NNN.yaml`
    params_yaml = event_dir / f"{name}.params.yaml"
    with open(params_yaml, "w") as f:
        yaml.safe_dump(user_params, f, sort_keys=False)
    with open(event_dir / f"{name}.yaml", "w") as f:
        yaml.safe_dump(
            {**config, "parameter_file": str(params_yaml)}, f, sort_keys=False
        )

    results_csv = Path(str(prefix) + "_results.csv")

    if not args.compare_only:
        status_file.write_text("running exozippy\n")
        try:
            from exozippy.run import run_fit

            run_fit(config, user_params=user_params)
        except BaseException:
            status_file.write_text("failed\n" + traceback.format_exc())
            raise

    if not results_csv.exists():
        status_file.write_text("failed\nno results CSV produced\n")
        raise FileNotFoundError(results_csv)

    rows = dc.compare_event(
        args.event,
        data_dir,
        results_csv,
        mmx_json=mmx_json if Path(mmx_json).exists() else None,
        out_csv=event_dir / "comparison.csv",
    )
    print()
    print(dc.format_comparison(rows))
    print(f"\nWrote {event_dir / 'comparison.csv'}")
    status_file.write_text("ok\n")


if __name__ == "__main__":
    main()
