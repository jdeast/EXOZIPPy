"""Validation of EXOZIPPy's galactic-model microlensing prior.

WHAT IS COMPUTED.  For a bulge sight line, the prior a microlensing fit
uses when its light curve is uninformative -- a lens + source model with
``galacticmodel`` (``model: koshimoto21``, the port of genulens; Koshimoto,
Baba & Bennett 2021, ApJ 917, 78, "K21") and NO data -- and from it the
rate-weighted distributions of the Einstein timescale t_E, the lens and
source distances D_L and D_S, the heliocentric relative proper motion
|mu_rel| and the lens mass M_L.  These are compared with

* genulens itself (2.0.0a4), run unmodified for the same sight lines; and
* the efficiency-corrected OGLE-IV t_E distributions (Mroz et al. 2017,
  Nature 548, 183; Mroz et al. 2019, ApJS 244, 29), field by field.

HOW THE EVENT-RATE WEIGHTING ARISES (nothing is added here; this is what
the model does in every fit):

* ``galacticmodel.kinematic_prior``, per star: each K21 population's
  number density (seven thin-disk age bins, thick disk, bar, halo, and the
  nuclear stellar disk inside genulens's box) times its velocity
  distribution and the Jacobian to the sampled proper motions.  The lens
  takes the density of all objects times the D^2 volume element; the source,
  observed through a flux limit, takes genulens's default source weight
  (main-sequence density x sqrt(D_S / 8 kpc)).  The lens mass prior is the
  population's present-day mass function (K21's IMF below the turnoff,
  white dwarfs, neutron stars and black holes above it, with natal kicks),
  optionally with unresolved binary lenses (``binaries: true``).
* ``mulensevent.event_rate_prior``: log(mu_rel) + log(theta_E), the rate
  factor Gamma ~ theta_E mu_rel.
* ``mulensevent.source_behind_lens``: a soft barrier on D_S - D_L.

THE PRIOR IS EVALUATED BY IMPORTANCE SAMPLING, not MCMC: NUTS never visits
the black-hole lens mode (``galactic_prior/importance.py`` says why and how
the proposal is built).  Every quantile is a weighted quantile with a
weighted-bootstrap error, and the diagnostics (Kish ESS, largest weight,
Pareto k) are stored with the result.

KNOWN DIFFERENCES FROM WHAT GENULENS AND A SURVEY REPORT.

* genulens draws the lens distance only from the line-of-sight bins wholly
  in front of the source's bin (dD = 100 pc, 25 pc in its nuclear-disk
  box), so no lens sits just in front of its source; those are the very
  shortest events.  And its grid ends at Dmax (16 kpc by default; the port
  reaches 20 kpc).  ``compare`` applies both supports to the PORT's draws
  (``sightlines.genulens_support_mask``) to show they account for the
  difference; genulens is never modified.
* With no light curve there is no Earth velocity: the port's mu_rel and
  t_E are heliocentric, and genulens runs with ``vEarthlb 0 0`` to match.
  OGLE's t_E is geocentric (a season average; ~1-2% longer at q90).
* OGLE is a consistency check, NOT an independent calibration: K21 fit
  their mass function to these same OGLE-IV t_E distributions.  The
  port's source weight is a survey approximation (no luminosity function).

Subcommands::

  sample    importance-sample the EXOZIPPy prior on one sight line
  genulens  run genulens (unmodified) for the sight lines
  mroz      rebuild the OGLE reference quantiles from a Mroz 2019 download
  compare   port vs genulens and port vs OGLE: tables and JSON
  figures   CDF overlays, the multi-line summary and the OGLE panels

``sample`` needs exozippy on PYTHONPATH; the rest need numpy/scipy (and
matplotlib for ``figures``).  See docs/galactic_prior_validation.md.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from galactic_prior import refs  # noqa: E402
from galactic_prior.sightlines import (  # noqa: E402
    BINARY_LINES,
    SIGHTLINES,
    parse_run,
    run_name,
)

REFS = refs.REFS
ALL_RUNS = list(SIGHTLINES) + [run_name(n, True) for n in BINARY_LINES]


def cmd_sample(args):
    from galactic_prior import importance
    from galactic_prior.model import PriorModel

    line, binaries = parse_run(args.run)
    sl = SIGHTLINES[line]
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    # PyMC / exozippy may write side files into the cwd.
    work = out_dir / f"work_{args.run}"
    work.mkdir(exist_ok=True)
    os.chdir(work)
    t0 = time.time()
    prior = PriorModel(sl["l"], sl["b"], binaries=binaries)
    build_s = time.time() - t0
    cols, log_w, info = importance.run(
        prior,
        n_final=int(args.draws),
        n_adapt=int(args.adapt_draws),
        rounds=args.rounds,
        n_components=args.components,
        pilot_draws=args.pilot_draws,
        pilot_tune=args.pilot_tune,
        pilot_chains=args.pilot_chains,
        cores=args.cores,
        seed=args.seed,
    )
    cols["D_L"] = cols["D_L"] / 1000.0  # pc -> kpc
    cols["D_S"] = cols["D_S"] / 1000.0
    info.update(
        run=args.run,
        l=sl["l"],
        b=sl["b"],
        ra=prior.ra,
        dec=prior.dec,
        binaries=binaries,
        build_s=build_s,
    )
    np.savez(
        out_dir / f"port_{args.run}_is.npz",
        log_w=log_w,
        **{k: v.astype(np.float64) for k, v in cols.items()},
    )
    refs.write_json(out_dir / f"port_{args.run}_is.json", info)


def cmd_genulens(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for k, run in enumerate(args.runs or ALL_RUNS):
        for j, variant in enumerate(args.variants):
            done = out_dir / f"genulens_{run}_{variant}.npz"
            if done.exists() and not args.force:
                continue
            seed = args.seed + 100 * ALL_RUNS.index(run) + j
            refs.run_genulens(
                args.genulens, run, variant, args.nsimu, seed, out_dir
            )


def cmd_mroz(args):
    refs.build_ogle_reference(args.mroz2019_dir)


def cmd_compare(args):
    from galactic_prior import compare

    compare.main(
        Path(args.port_dir),
        Path(args.genulens_dir),
        runs=args.runs or ALL_RUNS,
        store=args.store,
        commit=args.genulens_commit,
        processes=args.processes,
    )


def cmd_figures(args):
    from galactic_prior import figures

    figures.main(
        Path(args.port_dir),
        Path(args.genulens_dir),
        Path(args.out_dir),
        runs=args.runs or ALL_RUNS,
    )


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sample", help="importance-sample the EXOZIPPy prior")
    s.add_argument(
        "--run",
        required=True,
        choices=ALL_RUNS,
        help="sight line; <line>_bin turns binary lenses on",
    )
    s.add_argument("--out-dir", required=True)
    s.add_argument(
        "--draws", type=float, default=2e6, help="final importance draws"
    )
    s.add_argument(
        "--adapt-draws", type=float, default=2e5, help="draws per round"
    )
    s.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="adaptive rounds before the final",
    )
    s.add_argument("--components", type=int, default=40)
    s.add_argument("--pilot-draws", type=int, default=500)
    s.add_argument("--pilot-tune", type=int, default=500)
    s.add_argument("--pilot-chains", type=int, default=4)
    s.add_argument("--cores", type=int, default=os.cpu_count() or 1)
    s.add_argument("--seed", type=int, default=7163)

    g = sub.add_parser("genulens", help="run genulens for the sight lines")
    g.add_argument("--genulens", required=True, help="path to the binary")
    g.add_argument("--out-dir", required=True)
    g.add_argument("--nsimu", type=float, default=2e6)
    g.add_argument("--seed", type=int, default=2021)
    g.add_argument(
        "--variants",
        nargs="*",
        default=list(refs.GENULENS_VARIANTS),
        choices=list(refs.GENULENS_VARIANTS),
    )
    g.add_argument("--runs", nargs="*", choices=ALL_RUNS)
    g.add_argument("--force", action="store_true")

    m = sub.add_parser("mroz", help="rebuild the OGLE references")
    m.add_argument("--mroz2019-dir", required=True)

    c = sub.add_parser("compare", help="port vs genulens and vs OGLE")
    c.add_argument("--port-dir", required=True)
    c.add_argument("--genulens-dir", required=True)
    c.add_argument("--runs", nargs="*", choices=ALL_RUNS)
    c.add_argument(
        "--store",
        action="store_true",
        help="also write the stored summaries in refs/",
    )
    c.add_argument("--genulens-commit", default="2cc5870")
    c.add_argument(
        "--processes", type=int, default=6, help="sight lines in parallel"
    )

    f = sub.add_parser("figures", help="make the figures")
    f.add_argument("--port-dir", required=True)
    f.add_argument("--genulens-dir", required=True)
    f.add_argument("--out-dir", required=True)
    f.add_argument("--runs", nargs="*", choices=ALL_RUNS)

    args = ap.parse_args(argv)
    {
        "sample": cmd_sample,
        "genulens": cmd_genulens,
        "mroz": cmd_mroz,
        "compare": cmd_compare,
        "figures": cmd_figures,
    }[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
