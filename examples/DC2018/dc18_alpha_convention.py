#!/usr/bin/env python3
"""Calibrate the DC2018 answer key's alpha convention, empirically.

`dc18_common.compare_event` currently maps the master file's alpha through
a search over sign flips and +-180/360 offsets and keeps whatever lands
closest to the fit, which on event 128 still leaves a 40 deg residual and
reports a 2000-sigma pull.  That is a defect in the COMPARISON, not in the
fit: at the truth values of the other parameters the light curve's own
best-fit alpha is 308.0 deg while the master file says 348.357.

Neither BAGLE nor MulensModel defines this convention -- it belongs to the
simulation that generated the challenge -- so the only sound way to pin it
down is to measure it.  For every 2L1S event this scans alpha (in
MulensModel's convention, which is EXOZIPPy's) at the truth values of
t_0, u_0, t_E, rho, s and q, with source fluxes fit linearly, and records
the offset from the master file's value.  A convention that is a fixed
transformation shows up as a constant across 40+ independent geometries;
anything else means there is no single mapping and the alpha column should
be dropped rather than fudged.

Also records each field's galactic-north position angle, so a frame
rotation (equatorial vs galactic) can be tested as the explanation.
"""

import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dc18_common as dc
import MulensModel as mm

DATA = Path(
    os.environ.get(
        "DC18_DATA", Path.home() / "python/MMEXOFAST/data/2018DataChallenge"
    )
)
BANDS = ("W149", "Z087")
COARSE = 360  # 1 deg
FINE_HALFWIDTH = 1.5  # deg around the coarse minimum
FINE_N = 61


def galactic_north_pa(ra, dec):
    """Position angle of galactic north at (ra, dec), east of equatorial N."""
    c = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    g = c.galactic
    eps = 1e-5
    s = SkyCoord(
        l=g.l.deg * u.deg, b=(g.b.deg + eps) * u.deg, frame="galactic"
    ).icrs
    de = (s.ra.deg - c.ra.deg) * np.cos(np.radians(dec))
    dn = s.dec.deg - c.dec.deg
    return float(np.degrees(np.arctan2(de, dn)))


def one_event(event):
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

        t0 = truth["t_0"]
        window = (t0 - 3 * truth["t_E"], t0 + 3 * truth["t_E"])
        base = {k: truth[k] for k in ("t_0", "t_E", "rho", "s", "q")}

        def chi2(alpha, u0):
            try:
                model = mm.Model(dict(base, alpha=float(alpha), u_0=float(u0)))
                model.set_magnification_methods([window[0], "VBBL", window[1]])
                return float(
                    mm.Event(datasets=datasets, model=model).get_chi2()
                )
            except Exception:
                return np.inf

        # The u_0 sign and alpha are NOT independent conventions: flipping the
        # trajectory side reflects the geometry, so both signs must be on the
        # table or a sign difference is forced to masquerade as an alpha
        # offset (which for a binary lens is not the same solution at all).
        grid = np.linspace(0.0, 360.0, COARSE, endpoint=False)
        per_sign = {}
        for sgn in (+1.0, -1.0):
            u0 = sgn * abs(truth["u_0"])
            curve = np.array([chi2(a, u0) for a in grid])
            if not np.isfinite(curve).any():
                continue
            a0 = grid[int(np.nanargmin(curve))]
            fine = np.linspace(
                a0 - FINE_HALFWIDTH, a0 + FINE_HALFWIDTH, FINE_N
            )
            fcurve = np.array([chi2(a, u0) for a in fine])
            per_sign[sgn] = dict(
                alpha=float(fine[int(np.nanargmin(fcurve))]),
                chi2=float(np.nanmin(fcurve)),
                # How sharply the light curve determines alpha at all: the
                # gap to the median of the grid.  A weak anomaly leaves alpha
                # unconstrained and such an event must not vote.
                contrast=float(np.nanmedian(curve) - np.nanmin(fcurve)),
            )
        if not per_sign:
            return dict(event=event, cls=cls, status="all chi2 non-finite")
        sgn = min(per_sign, key=lambda k: per_sign[k]["chi2"])
        best = per_sign[sgn]["alpha"]
        best_chi2 = per_sign[sgn]["chi2"]
        other = per_sign.get(-sgn, {})

        ra, dec = dc.event_coords(DATA, event)
        n = sum(len(d.time) for d in datasets)
        wrap = lambda x: (x + 180) % 360 - 180  # noqa: E731
        return dict(
            event=event,
            cls=cls,
            status="ok",
            alpha_master=truth["alpha"],
            alpha_fit=best,
            u0_sign_fit=sgn,
            u0_sign_master=float(np.sign(truth["u_0"])),
            offset=wrap(best - truth["alpha"]),
            offset_reflected=wrap(best + truth["alpha"]),
            contrast=per_sign[sgn]["contrast"],
            dchi2_u0_sign=float(other.get("chi2", np.nan) - best_chi2),
            pa_gal_north=galactic_north_pa(ra, dec),
            chi2=best_chi2,
            chi2_per_n=best_chi2 / n,
            n_points=n,
            u_0=truth["u_0"],
            s=truth["s"],
            q=truth["q"],
            t_E=truth["t_E"],
            rho=truth["rho"],
            ra=ra,
            dec=dec,
        )
    except Exception as exc:
        return dict(event=event, status=f"failed: {type(exc).__name__}: {exc}")


def main():
    events = [
        int(x)
        for x in open(Path(__file__).resolve().parent / "events.txt")
        if x.strip()
    ]
    ncpu = int(os.environ.get("NSLOTS", 0)) or max(1, mp.cpu_count() // 2)
    print(
        f"scanning alpha for {len(events)} events on {ncpu} workers\n",
        flush=True,
    )
    with mp.Pool(ncpu) as pool:
        rows = pool.map(one_event, events)

    ok = [r for r in rows if r.get("status") == "ok"]
    print(f"{len(ok)}/{len(rows)} events gave a 2L1S alpha scan\n")
    print(
        f"{'ev':>4s} {'cls':>10s} {'master':>9s} {'fit':>9s} {'offset':>9s} "
        f"{'refl':>9s} {'u0fit':>6s} {'u0mas':>6s} {'contrast':>10s} "
        f"{'chi2/N':>8s}"
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
            f"{r['offset_reflected']:+9.3f} {r['u0_sign_fit']:+6.0f} "
            f"{r['u0_sign_master']:+6.0f} {r['contrast']:10.1f} "
            f"{r['chi2_per_n']:8.4f}"
        )

    wrap = lambda x: (x + 180) % 360 - 180  # noqa: E731

    def report(name, v):
        # circular statistics -- a plain std is meaningless near the wrap
        v = np.asarray(v, float)
        if v.size == 0:
            print(f"  {name:36s} (no events)")
            return
        r = np.hypot(
            np.mean(np.cos(np.radians(v))), np.mean(np.sin(np.radians(v)))
        )
        mean = np.degrees(
            np.arctan2(
                np.mean(np.sin(np.radians(v))), np.mean(np.cos(np.radians(v)))
            )
        )
        scat = np.degrees(np.sqrt(-2 * np.log(r))) if r > 0 else np.inf
        print(
            f"  {name:36s} n={v.size:3d}  mean {mean:+8.3f}  "
            f"scatter {scat:7.3f} deg   R={r:.4f}"
        )

    # Only events whose light curve actually pins alpha may vote.  A weak
    # anomaly leaves the whole alpha grid within a few chi2 of the minimum,
    # and its "best alpha" is noise that would swamp a real constant.
    for cut in (0.0, 100.0, 1000.0):
        sub = [r for r in ok if r["contrast"] >= cut]
        print("\n" + "=" * 70)
        print(
            f"CONVENTION HYPOTHESES   (contrast >= {cut:g}; {len(sub)} events)"
        )
        print("=" * 70)
        if not sub:
            continue
        off = np.array([r["offset"] for r in sub])
        refl = np.array([r["offset_reflected"] for r in sub])
        pa = np.array([r["pa_gal_north"] for r in sub])
        report("fit - master", off)
        report("fit + master  (reflection)", refl)
        report("fit - master - PA(gal N)", wrap(off - pa))
        report("fit - master + PA(gal N)", wrap(off + pa))
        report("fit + master - PA(gal N)", wrap(refl - pa))
        report("fit + master + PA(gal N)", wrap(refl + pa))
        # Split by which u_0 sign the light curve chose: if the master file's
        # u_0 sign convention differs, the two subsets obey different rules.
        for s in (+1.0, -1.0):
            idx = [i for i, r in enumerate(sub) if r["u0_sign_fit"] == s]
            report(f"  fit - master | u0_fit={s:+.0f}", off[idx])
            report(f"  fit + master | u0_fit={s:+.0f}", refl[idx])

    print(
        "\n  R near 1 (scatter << 10 deg) means that hypothesis IS the "
        "convention.\n  If nothing concentrates even at high contrast, "
        "there is no single mapping:\n  drop the alpha column rather than "
        "fudge it."
    )

    json.dump(rows, open("dc18_alpha_convention.json", "w"), indent=2)
    print("\nwrote dc18_alpha_convention.json")


if __name__ == "__main__":
    main()
