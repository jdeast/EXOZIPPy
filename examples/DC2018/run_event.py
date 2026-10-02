#!/usr/bin/env python3
"""Run the full DC2018 pipeline for one event.

    poetry run python run_event.py 128 [options]

Pipeline (one process, suitable for one cluster job):
  1. EXOZIPPy fits the selected bands.  The params file this driver writes
     deliberately carries NO microlensing start values, so the built-in
     PSPL peak finder seeds t_0/u_0/t_E from the light curves and s, q and
     alpha keep their defaults.yaml starts: no binary-lens estimator runs
     anywhere, the sampler finds the companion (JDE 2026-09-17).  Sampling
     is PTDE by default; artifacts land under <out-dir>/<NNN>/fitresults/.
     (Until 2026-10-01 a step before this ran MMEXOFAST on both bands and
     seeded the fit from its JSON; that hand-off was removed from
     EXOZIPPy.  convert_mmexofast_json.py moves an existing JSON's seeds,
     mask and error factors into a config's own files.)
  2. The posterior is compared against the challenge truth
     (comparison.csv + stdout table).

The generated config and params are also dumped to YAML in the event
directory, so any event can be re-run or debugged with the plain CLI:
    cd <out-dir>/<NNN> && exozippy DC2018_<NNN>.yaml

Everything is idempotent per event directory: --recompute controls
whether an existing trace is resampled.
"""

import argparse
import os
import sys
import traceback
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dc18_common as dc


def build_config(name, files, prefix, args):
    bands = list(files)
    config = {
        "run": {"name": name},
        "prefix": str(prefix),
        "star": [{"name": "Lens"}, {"name": "Source"}],
        "planet": [{"name": "Companion"}],
        # Event-level keys live on the event; the lens is one entry per
        # lens BODY (primary first) and the source one per source body.
        "mulensevent": [{"finite_source": bool(args.finite_source)}],
        "lens": [{"body": "star.Lens"}, {"body": "planet.Companion"}],
        "source": [{"body": "star.Source"}],
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
                # Hogg inlier/outlier mixture on every light curve, in
                # place of a hard bad-data mask: junk lands in the wide
                # background component instead of dragging the fit, and the
                # per-point outlier probabilities are auditable afterwards
                # (Instrument.outlier_prob_at_data).
                "likelihood": "hogg",
            }
            for b in bands
        ],
        "sampler": {
            "method": args.sampler,
            # The DE polish is the only one available here (the binary-lens
            # magnification Op has no gradient), and on DC18 cadence it
            # routinely spends the whole default 150-sweep cap still
            # climbing -- the log says "hit the 150-step cap" and the
            # whitening probe then reports a gradient-dominated start.  A
            # seed left short of its basin optimum is a seed the sampler
            # abandons, so this is exposed per run rather than buried.
            "seed_polish": args.seed_polish,
            "n_temps": (
                args.n_temps
                if str(args.n_temps).lower() == "auto"
                else int(args.n_temps)
            ),
            # T_max sets what barrier the ladder can cross: the hottest rung
            # sees a barrier of B/T_max nats, so crossing needs that down to
            # a few.  Event 128's measured close/wide barrier is 42563 nats,
            # i.e. 213 nats even at the
            # default T_max=200 -- unreachable -- while T_max=8500 brings it
            # to 5.  With `n_temps: auto` the rung count follows
            # sqrt(D/2)*ln(T_max), so that costs 34 rungs against 20, not a
            # different algorithm.
            "T_max": args.t_max,
            # n_temps must satisfy the DEO criterion n_temps >= 2*Lambda+1
            # (Lambda = the measured communication barrier), and `auto` --
            # sqrt(D/2)*ln(T_max) -- does NOT know Lambda, so it
            # under-provisions by ~20% on this model at every T_max.  When
            # parallax became a real likelihood direction Lambda doubled
            # (6.0 -> 11.5 at T_max=200) and round trips collapsed from 1427
            # to 8 on an unchanged 20-rung ladder.  Set n_chains down when
            # setting n_temps up to hold the slot count roughly fixed.
            "n_chains": args.n_chains,
            # Re-space the ladder during tune to equalize the per-pair
            # communication barrier (Syed+2022).  Worth turning on when the
            # per-rung swap acceptances are non-uniform: a round trip must
            # cross EVERY pair, so transport is throttled by the worst
            # stretch and a geometric ladder cannot fix that by getting
            # longer.  Measured on event 128 at n_temps=48 (a correctly
            # provisioned ladder: Lambda=18.9 vs the 39 the DEO criterion
            # asks): acceptance 0.46-0.52 cold against 0.66-0.70 hot, and
            # zero round trips in 21 h.
            "adapt_ladder": bool(args.adapt_ladder),
            "cores": args.cores,
            "tune": args.tune,
            "draws": args.draws,
            "nthin": 1,
            "recompute_trace": bool(args.recompute),
            "eval_timeout": 10,
        },
        # Degenerate events routinely leave chains split across solution
        # branches with no inter-mode mixing; occupancy weights then reflect
        # initialization, not posterior mass (event 128: 52/54 chains in a
        # branch 500 nats WORSE than the one the other 2 found). Per-mode
        # bridge-sampling evidence weights are the designed remedy.
        "modes": {"weights": "evidence"},
    }
    return config


def build_user_params(ra, dec, fix_u1=False, bands_for_u1=()):
    """Fixed-parameter overrides, mirroring examples/DC2018_128.

    Coordinates come from event_info.txt. The Lens's radius/teff/feh fixes
    are the same 'not constrainable without SED data' hack the DC2018_128
    example documents; the microlensing start values are deliberately
    ABSENT so the built-in peak finder seeds them from the light curves (a
    user entry would outrank it). The Source star
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
    if fix_u1:
        # The DC2018 light curves were simulated with gamma = 0 -- NO limb
        # darkening -- in all 44 events (the master file's `gamma` column,
        # and why the pre-2026-10 seeding step ran its fitter with
        # gamma = 0).  band.u1 is the
        # linear u (op.py applies it via set_limb_coeff_u; gamma = 2u/(3-u),
        # so gamma = 0 <=> u = 0 exactly).
        #
        # Left free it does not merely waste a parameter, it corrupts rho:
        # limb darkening and source size both shape the finite-source peak
        # and trade against each other.  Measured across five event-128 runs,
        # u1(W149) ran 0.14 -> 1.87 (two of them physically impossible) and
        # rho tracked it monotonically, 0.0050 -> 0.0111 against a truth of
        # 0.00607.  For REAL data a prior is the right answer; for a dataset
        # generated with zero limb darkening, zero is.
        for b in bands_for_u1:
            params[f"band.{b}.u1"] = {"initval": 0.0, "sigma": 0}
    return params


def build_parser():
    """The CLI, as a parser other drivers can borrow.

    EXTRACTED so a caller can get every default without hand-stubbing an
    args object.  build_config reads eleven attributes off `args`
    (adapt_ladder, cores, draws, finite_source, n_chains, n_temps,
    recompute, sampler, seed_polish, t_max, tune), and two helper scripts
    crashed with AttributeError on a stub that had two of them -- a bug that
    recurs every time this parser gains an option.  Borrow the parser
    instead:

        args = build_parser().parse_args([str(event)])
        args.finite_source = True        # then override what you mean to
    """
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("event", type=int, help="DC18 event number (e.g. 128)")
    ap.add_argument(
        "--data-dir",
        default=None,
        help="2018DataChallenge tree (default $DC18_DATA or "
        "~/python/MMEXOFAST/data/2018DataChallenge)",
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
        "--n-temps",
        default="8",
        help="PT temperature rungs: an integer or 'auto' "
        "(max(8, ceil(sqrt(D/2)*ln(T_max))); use when the ladder-health "
        "warning reports a communication-limited ladder)",
    )
    ap.add_argument(
        "--fix-u1",
        action="store_true",
        help="Pin every band's linear limb-darkening u1 at 0. Correct for "
        "DC2018, whose 44 events were all simulated with gamma = 0; leaving "
        "u1 free lets it trade against rho through the finite-source profile",
    )
    ap.add_argument(
        "--adapt-ladder",
        action="store_true",
        help="Re-space the PT ladder during tuning to equalize the per-pair "
        "communication barrier (sampler key adapt_ladder). Use when the "
        "per-rung swap acceptances reported by the ladder-health line are "
        "non-uniform",
    )
    ap.add_argument(
        "--n-chains",
        type=int,
        default=None,
        help="Chains per temperature rung (default: the sampler's own "
        "resolve_n_chains). Lower this when raising --n-temps so the total "
        "slot count, and so the cost, stays comparable",
    )
    ap.add_argument(
        "--t-max",
        type=float,
        default=200.0,
        help="Hottest PT rung (default 200). The hottest rung sees a "
        "basin barrier of B/T_max nats, so raise this when modes are "
        "suspected to be separated by more than a few hundred nats; with "
        "--n-temps auto the rung count follows sqrt(D/2)*ln(T_max)",
    )
    ap.add_argument(
        "--seed-polish",
        default="auto",
        help="Pre-whitening seed polish: 'auto' (default), 'off', or a max "
        "sweep count. The gradient-free DE polish stops on the cap, so a "
        "seed that reports 'hit the N-step cap' was still improving; raise "
        "this when the whitening probe warns of a gradient-dominated start",
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
    return ap


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)

    if args.quick:
        args.bands = "Z087"
        args.tune = min(args.tune, 500)
        args.draws = min(args.draws, 1000)

    if str(args.seed_polish).lstrip("-").isdigit():
        args.seed_polish = int(args.seed_polish)

    data_dir = dc.data_dir_or_raise(args.data_dir)
    bands = [b.strip() for b in args.bands.split(",") if b.strip()]
    files = dc.light_curve_files(data_dir, args.event, bands)
    ra, dec = dc.event_coords(data_dir, args.event)

    name = f"DC2018_{args.event:03d}"
    event_dir = (Path(args.out_dir) / f"{args.event:03d}").resolve()
    event_dir.mkdir(parents=True, exist_ok=True)
    prefix = event_dir / "fitresults" / name

    status_file = event_dir / "status.txt"

    config = build_config(name, files, prefix, args)
    user_params = build_user_params(
        ra, dec, fix_u1=args.fix_u1, bands_for_u1=bands
    )

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
        mmx_json=None,
        out_csv=event_dir / "comparison.csv",
    )
    print()
    print(dc.format_comparison(rows))
    print(f"\nWrote {event_dir / 'comparison.csv'}")
    status_file.write_text("ok\n")


if __name__ == "__main__":
    main()
