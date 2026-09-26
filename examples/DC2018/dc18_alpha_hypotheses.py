#!/usr/bin/env python3
"""Test what the DC2018 master file's alpha actually means.

Input is dc18_alpha_convention.run1-u0fixed.json: per event, the alpha that
best fits the light curve in MulensModel's convention (trajectory-relative,
counterclockwise from the source trajectory to the primary->companion axis)
at the truth values of everything else, with u_0 held at the master file's
own signed value.

A plain offset does not work (scatter ~100 deg over 44 events), which rules
out "same definition, different zero point".  The obvious alternative is
that the master file's alpha is a SKY POSITION ANGLE of the binary axis --
BAGLE defines its alpha that way -- rather than an angle relative to the
trajectory.  Those two differ by the position angle of the relative proper
motion, which is event-specific and recoverable from the answer key's own
galactic-frame proper motions.  That would produce exactly the observed
scatter, and it is testable.

Because parallax is negligible for these events, (u_0, alpha) ->
(-u_0, -alpha) is an exact mirror symmetry of the light curve, so every
hypothesis is only determined up to an overall reflection; each is
therefore scored in both handednesses.
"""

import json
from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord

HERE = Path(__file__).resolve().parent
DATA = Path.home() / "python/MMEXOFAST/data/2018DataChallenge"


def wrap(x):
    return (np.asarray(x, float) + 180) % 360 - 180


def concentration(v):
    """Circular mean and scatter; R -> 1 means 'this is a constant'."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan, np.inf, 0.0
    c, s = np.mean(np.cos(np.radians(v))), np.mean(np.sin(np.radians(v)))
    R = float(np.hypot(c, s))
    mean = float(np.degrees(np.arctan2(s, c)))
    scat = float(np.degrees(np.sqrt(-2 * np.log(R)))) if R > 0 else np.inf
    return mean, scat, R


def load_master():
    ans = DATA / "Answers"
    cols = np.genfromtxt(
        ans / "wfirstColumnNumbers.txt",
        dtype=None,
        encoding="utf-8",
        usecols=[0, 1],
        skip_header=2,
        names=["index", "name"],
    )
    names = [
        f"col{i}" if nm == "|" else nm for i, nm in enumerate(cols["name"])
    ]
    return pd.read_csv(
        ans / "master_file.txt",
        names=names,
        usecols=range(len(names)),
        sep=r"\s+",
        skiprows=1,
    )


def pa_murel(row):
    """Position angle of mu_rel (lens - source), galactic and equatorial.

    Galactic: measured from galactic north toward increasing l.
    Equatorial: measured from equatorial north toward east.
    """
    mu_l = float(row["lmu_l"]) - float(row["smu_l"])
    mu_b = float(row["lmu_b"]) - float(row["smu_b"])
    pa_gal = np.degrees(np.arctan2(mu_l, mu_b))

    ra, dec = float(row["ra"]), float(row["dec"])
    c = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    g = c.galactic
    eps = 1e-5
    cosd = np.cos(np.radians(dec))

    def icrs_of(l, b):
        s = SkyCoord(l=l * u.deg, b=b * u.deg, frame="galactic").icrs
        return s.ra.deg, s.dec.deg

    r1, d1 = icrs_of(g.l.deg + eps / np.cos(g.b.rad), g.b.deg)
    r2, d2 = icrs_of(g.l.deg, g.b.deg + eps)
    e_l = np.array([(r1 - ra) * cosd, d1 - dec]) / eps
    e_b = np.array([(r2 - ra) * cosd, d2 - dec]) / eps
    v = mu_l * e_l / np.hypot(*e_l) + mu_b * e_b / np.hypot(*e_b)
    pa_eq = np.degrees(np.arctan2(v[0], v[1]))
    return float(pa_gal), float(pa_eq)


def main():
    rows = json.load(open(HERE / "dc18_alpha_convention.run1-u0fixed.json"))
    master = load_master()
    recs = []
    for r in rows:
        if r.get("status") != "ok":
            continue
        pg, pe = pa_murel(master.iloc[r["event"] - 1])
        recs.append(
            dict(
                event=r["event"],
                fit=r["alpha_fit"],
                dc=r["alpha_master"],
                pa_gal=pg,
                pa_eq=pe,
                chi2_per_n=r["chi2_per_n"],
            )
        )
    print(f"{len(recs)} events\n")
    print(
        f"{'ev':>4s} {'alpha_DC':>9s} {'alpha_fit':>9s} "
        f"{'PA(murel)gal':>13s} {'PA(murel)eq':>12s} {'chi2/N':>7s}"
    )
    for x in recs:
        print(
            f"{x['event']:>4d} {x['dc']:9.3f} {x['fit']:9.3f} "
            f"{x['pa_gal']:13.3f} {x['pa_eq']:12.3f} "
            f"{x['chi2_per_n']:7.4f}"
        )

    fit = np.array([x["fit"] for x in recs])
    dc = np.array([x["dc"] for x in recs])
    pg = np.array([x["pa_gal"] for x in recs])
    pe = np.array([x["pa_eq"] for x in recs])

    tests = {
        "fit - dc                        ": fit - dc,
        "fit - dc + PA_gal(murel)        ": fit - dc + pg,
        "fit - dc - PA_gal(murel)        ": fit - dc - pg,
        "fit - dc + PA_eq(murel)         ": fit - dc + pe,
        "fit - dc - PA_eq(murel)         ": fit - dc - pe,
        "fit + dc                        ": fit + dc,
        "fit + dc + PA_gal(murel)        ": fit + dc + pg,
        "fit + dc - PA_gal(murel)        ": fit + dc - pg,
        "fit + dc + PA_eq(murel)         ": fit + dc + pe,
        "fit + dc - PA_eq(murel)         ": fit + dc - pe,
        "dc - PA_gal(murel)  [ignore fit]": dc - pg,
    }
    print("\n" + "=" * 74)
    print(
        "HYPOTHESES  -- R near 1 (scatter << 20 deg) identifies the convention"
    )
    print("=" * 74)
    for name, v in sorted(
        tests.items(), key=lambda kv: -concentration(wrap(kv[1]))[2]
    ):
        mean, scat, R = concentration(wrap(v))
        flag = "  <== " if R > 0.7 else ""
        print(
            f"  {name}  mean {mean:+8.2f}  scatter {scat:7.2f}  "
            f"R={R:.4f}{flag}"
        )

    # Same, restricted to events the light curve actually constrains well.
    good = np.array([x["chi2_per_n"] < 1.05 for x in recs])
    print(f"\n  (restricted to {good.sum()} events with chi2/N < 1.05)")
    for name, v in sorted(
        tests.items(),
        key=lambda kv: -concentration(wrap(np.asarray(kv[1])[good]))[2],
    ):
        mean, scat, R = concentration(wrap(np.asarray(v)[good]))
        flag = "  <== " if R > 0.7 else ""
        print(
            f"  {name}  mean {mean:+8.2f}  scatter {scat:7.2f}  "
            f"R={R:.4f}{flag}"
        )


if __name__ == "__main__":
    main()
