#!/usr/bin/env python3
"""Is the 2018 Roman Data Challenge answer key's alpha convertible to ours?

Answer, measured over all 44 events: NO.  This script is the measurement,
kept because the conclusion is a negative one and negatives rot -- the next
person to notice a 2000-sigma alpha pull should be able to re-run this in an
hour rather than re-derive it.

    poetry run python scripts/dc18_alpha_convention.py

WHAT IT DOES.  For every 2L1S event it holds the answer key's own t_0, u_0,
t_E, rho, s and q fixed, scans alpha over [0, 360) in MulensModel's
convention (which is EXOZIPPy's -- see components/mulensing/op.py, and
mmexofast_support maps MMEXOFAST's alpha to ours by the identity), fits the
source fluxes linearly at each step, and takes the alpha the light curve
itself prefers.  It then asks whether ANY global transformation carries the
master file's alpha onto that.

WHY THE u_0 SIGN IS TAKEN FROM THE MASTER FILE AND NOT SCANNED.  These
events have |pi_E| ~ 0.02, so parallax is negligible, and without it
(u_0, alpha) -> (-u_0, -alpha) is an EXACT mirror symmetry of the light
curve: event 128 gives byte-identical chi2 at (+0.1418, 308.15) and
(-0.1418, 51.85).  Scanning both signs therefore adds no information and
actively hurts -- `min` picks arbitrarily between two exactly tied minima,
so the recovered alpha flips between branches at random and washes out any
constant that might be there.  An earlier version of this script did scan
both signs and produced a strictly noisier answer.  Alpha is only ever
determined up to that reflection, which is why the reflected hypothesis is
tested explicitly below rather than by flipping u_0.

THE RESULT (44 events; circular concentration R, where 1.0 would mean "this
IS the rule" and ~0.15 is what 44 random angles give):

    fit - alpha_key                                       R = 0.09
    fit + alpha_key   (a reflection)                      R = 0.10
    either, with the galactic->equatorial PA removed      R = 0.11 - 0.19
    either, with PA(mu_rel) removed                       R = 0.03 - 0.19

That last one was the best physical guess: that the key's alpha is a sky
POSITION ANGLE of the binary axis (which is how BAGLE defines its alpha)
rather than an angle relative to the source trajectory, in which case the
two differ by the position angle of the relative proper motion -- and that
IS computable from the key's own galactic-frame proper motions.  It does
not concentrate either.

Restricting to the twelve events where the anomaly pins alpha hardest does
not help: R = 0.22 and 0.41, against ~0.29 expected from twelve random
angles.  So this is a property of the answer key, not of a weak constraint
or of the wrong sign branch.

CONSEQUENCE, already applied in examples/DC2018/dc18_common.py: alpha is
reported with NO truth value and NO pull.  The sign/+-180 search that used
to map it could not fail visibly -- it always returned its nearest
candidate, so an unmappable truth came back as a confident number.  On
event 128 it printed a 2034-sigma pull while the fitted alpha sat 0.3 deg
from the light curve's own optimum.

A plausible explanation, unconfirmed: the key carries a, inc, phase and
period -- a full orbit -- but no node angle, so if alpha is defined in the
orbital frame there is no way to reach a sky angle without Omega.  Matt
Penny (who generated the simulations) has been asked.
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import astropy.units as u
import MulensModel as mm
import numpy as np
from astropy.coordinates import SkyCoord

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "examples" / "DC2018"))
import dc18_common as dc  # noqa: E402

DATA = Path(os.environ.get("DC18_DATA", dc.DEFAULT_DATA_DIR))
BANDS = ("W149", "Z087")
COARSE = 360  # 1 deg
FINE_HALFWIDTH = 1.5  # deg either side of the coarse minimum
FINE_N = 61


def wrap(x):
    return (np.asarray(x, float) + 180) % 360 - 180


def concentration(v):
    """Circular mean, scatter and resultant length R.

    A plain std is meaningless near the 0/360 wrap, and the whole question
    is whether a set of angles clusters.
    """
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan, np.inf, 0.0
    c, s = np.mean(np.cos(np.radians(v))), np.mean(np.sin(np.radians(v)))
    R = float(np.hypot(c, s))
    mean = float(np.degrees(np.arctan2(s, c)))
    scat = float(np.degrees(np.sqrt(-2 * np.log(R)))) if R > 0 else np.inf
    return mean, scat, R


def _frame_basis(ra, dec):
    """Galactic (l, b) unit vectors at (ra, dec), in equatorial (E, N)."""
    c = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    g = c.galactic
    eps = 1e-5
    cosd = np.cos(np.radians(dec))

    def icrs_of(lon, lat):
        s = SkyCoord(l=lon * u.deg, b=lat * u.deg, frame="galactic").icrs
        return s.ra.deg, s.dec.deg

    r1, d1 = icrs_of(g.l.deg + eps / np.cos(g.b.rad), g.b.deg)
    r2, d2 = icrs_of(g.l.deg, g.b.deg + eps)
    e_l = np.array([(r1 - ra) * cosd, d1 - dec]) / eps
    e_b = np.array([(r2 - ra) * cosd, d2 - dec]) / eps
    return e_l / np.hypot(*e_l), e_b / np.hypot(*e_b)


def galactic_north_pa(ra, dec):
    """Position angle of galactic north, east of equatorial north."""
    _, e_b = _frame_basis(ra, dec)
    return float(np.degrees(np.arctan2(e_b[0], e_b[1])))


def pa_mu_rel(row, ra, dec):
    """PA of mu_rel (lens - source) in galactic and equatorial frames.

    The key gives proper motions in galactic (l, b), so the equatorial PA
    needs the frame basis above.  This is the quantity that would separate
    a trajectory-relative alpha from a sky position angle.
    """
    mu_l = float(row["lmu_l"]) - float(row["smu_l"])
    mu_b = float(row["lmu_b"]) - float(row["smu_b"])
    pa_gal = float(np.degrees(np.arctan2(mu_l, mu_b)))
    e_l, e_b = _frame_basis(ra, dec)
    v = mu_l * e_l + mu_b * e_b
    return pa_gal, float(np.degrees(np.arctan2(v[0], v[1])))


def one_event(event):
    """Scan alpha for one event at the key's own other parameters."""
    try:
        truth, cls = dc.load_truth(DATA, event)
        if (
            not np.isfinite(truth["s"])
            or not np.isfinite(truth["q"])
            or truth["q"] <= 0
        ):
            return dict(event=event, cls=cls, status="no 2L1S truth")

        datasets = []
        for b in BANDS:
            f = DATA / f"n20180816.{b}.WFIRST18.{event:03d}.txt"
            if not f.exists():
                continue
            t, flux, err = np.loadtxt(f, unpack=True)
            datasets.append(
                mm.MulensData(data_list=[t, flux, err], phot_fmt="flux")
            )
        if not datasets:
            return dict(event=event, cls=cls, status="no data")

        window = (
            truth["t_0"] - 3 * truth["t_E"],
            truth["t_0"] + 3 * truth["t_E"],
        )
        # u_0 keeps the key's own SIGN -- see the module docstring.
        base = {k: truth[k] for k in ("t_0", "u_0", "t_E", "rho", "s", "q")}

        def chi2(alpha):
            try:
                model = mm.Model(dict(base, alpha=float(alpha)))
                model.set_magnification_methods([window[0], "VBBL", window[1]])
                return float(
                    mm.Event(datasets=datasets, model=model).get_chi2()
                )
            except Exception:
                return np.inf

        grid = np.linspace(0.0, 360.0, COARSE, endpoint=False)
        curve = np.array([chi2(a) for a in grid])
        if not np.isfinite(curve).any():
            return dict(event=event, cls=cls, status="all chi2 non-finite")
        a0 = grid[int(np.nanargmin(curve))]
        fine = np.linspace(a0 - FINE_HALFWIDTH, a0 + FINE_HALFWIDTH, FINE_N)
        fcurve = np.array([chi2(a) for a in fine])
        best = float(fine[int(np.nanargmin(fcurve))])
        best_chi2 = float(np.nanmin(fcurve))

        ra, dec = dc.event_coords(DATA, event)
        master, _ = dc.load_master_row(DATA, event)
        pg, pe = pa_mu_rel(master, ra, dec)
        n = sum(len(d.time) for d in datasets)
        return dict(
            event=event,
            cls=cls,
            status="ok",
            alpha_master=truth["alpha"],
            alpha_fit=best,
            offset=float(wrap(best - truth["alpha"])),
            offset_reflected=float(wrap(best + truth["alpha"])),
            # How sharply the light curve pins alpha at all: a weak anomaly
            # leaves the whole grid within a few chi2 and its "best alpha"
            # is noise that must not be allowed to vote.
            contrast=float(np.nanmedian(curve) - best_chi2),
            pa_gal_north=galactic_north_pa(ra, dec),
            pa_murel_gal=pg,
            pa_murel_eq=pe,
            chi2=best_chi2,
            chi2_per_n=best_chi2 / n,
            n_points=n,
            u_0=truth["u_0"],
            s=truth["s"],
            q=truth["q"],
        )
    except Exception as exc:
        return dict(event=event, status=f"failed: {type(exc).__name__}: {exc}")


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--events",
        default=None,
        help="comma-separated event numbers (default: every line of "
        "examples/DC2018/events.txt)",
    )
    ap.add_argument(
        "--data-dir",
        default=None,
        help="2018DataChallenge tree (default $DC18_DATA, else the "
        "MMEXOFAST source checkout)",
    )
    ap.add_argument(
        "--ncpu",
        type=int,
        default=int(os.environ.get("NSLOTS", 0)) or None,
        help="worker processes (default $NSLOTS, else half the machine)",
    )
    ap.add_argument(
        "--contrast-cuts",
        default="0,1000",
        help="comma-separated minimum chi2 contrast for an event to vote in "
        "the hypothesis tests: a weak anomaly leaves alpha unconstrained and "
        "its best-fit value is noise (default 0,1000)",
    )
    ap.add_argument(
        "--out",
        default=str(REPO_ROOT / "dc18_alpha_convention.json"),
        help="where to write the per-event JSON",
    )
    return ap.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    global DATA
    if args.data_dir:
        DATA = Path(args.data_dir)
    if args.events:
        events = [int(x) for x in args.events.split(",") if x.strip()]
    else:
        events = [
            int(x)
            for x in open(REPO_ROOT / "examples" / "DC2018" / "events.txt")
            if x.strip()
        ]
    ncpu = args.ncpu or max(1, mp.cpu_count() // 2)
    print(
        f"scanning alpha for {len(events)} events on {ncpu} workers\n",
        flush=True,
    )
    with mp.Pool(ncpu) as pool:
        rows = pool.map(one_event, events)

    ok = [r for r in rows if r.get("status") == "ok"]
    print(f"{len(ok)}/{len(rows)} events gave a 2L1S alpha scan\n")
    print(
        f"{'ev':>4s} {'cls':>10s} {'master':>9s} {'fit':>9s} "
        f"{'offset':>9s} {'refl':>9s} {'contrast':>10s} {'chi2/N':>8s}"
    )
    for r in rows:
        if r.get("status") != "ok":
            print(
                f"{r['event']:>4d} {str(r.get('cls', '?')):>10s}   "
                f"{r['status']}"
            )
            continue
        print(
            f"{r['event']:>4d} {r['cls']:>10s} {r['alpha_master']:9.3f} "
            f"{r['alpha_fit']:9.3f} {r['offset']:+9.3f} "
            f"{r['offset_reflected']:+9.3f} {r['contrast']:10.1f} "
            f"{r['chi2_per_n']:8.4f}"
        )

    def report(name, v):
        mean, scat, R = concentration(v)
        flag = "   <== THIS IS THE RULE" if R > 0.7 else ""
        print(
            f"  {name:36s} n={np.size(v):3d}  mean {mean:+8.2f}  "
            f"scatter {scat:7.2f}  R={R:.4f}{flag}"
        )

    for cut in [float(c) for c in args.contrast_cuts.split(",")]:
        sub = [r for r in ok if r["contrast"] >= cut]
        print("\n" + "=" * 74)
        print(
            f"CONVENTION HYPOTHESES   (contrast >= {cut:g}; {len(sub)} events)"
        )
        print("=" * 74)
        if not sub:
            continue
        off = np.array([r["offset"] for r in sub])
        refl = np.array([r["offset_reflected"] for r in sub])
        pgn = np.array([r["pa_gal_north"] for r in sub])
        pmg = np.array([r["pa_murel_gal"] for r in sub])
        pme = np.array([r["pa_murel_eq"] for r in sub])
        report("fit - master", off)
        report("fit + master  (reflection)", refl)
        report("fit - master -/+ PA(gal north)", wrap(off - pgn))
        report("fit + master -/+ PA(gal north)", wrap(refl + pgn))
        if np.isfinite(pmg).any():
            report("fit - master - PA_gal(mu_rel)", wrap(off - pmg))
            report("fit + master - PA_gal(mu_rel)", wrap(refl - pmg))
            report("fit - master - PA_eq(mu_rel)", wrap(off - pme))
            report("fit + master - PA_eq(mu_rel)", wrap(refl - pme))

    print("\n  R near 1 (scatter << 20 deg) would identify the convention.")
    print("  Nothing concentrates, at any contrast cut -- which is why")
    print("  dc18_common reports alpha with no truth and no pull.")

    out = Path(args.out)
    json.dump(rows, open(out, "w"), indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
