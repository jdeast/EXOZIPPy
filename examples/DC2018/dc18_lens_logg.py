"""How hard does the lens posterior press against the BC grid's logg ceiling?

For each run: the lens's logg distribution (fraction above 4.9 and 5.0),
the soft-bound potential on star.loggsed at each draw if stored, and the
lens mass conditional on logg.  Why: the NextGen BC grid stops at
logg = 5.0, a 0.13 solMass dwarf has logg 5.1 and a 0.09 solMass dwarf
5.25, and the SED soft-bounds loggsed to the grid -- so a light lens is
excluded by the atmosphere grid, not by the data.
"""

import argparse
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run-dir/event pairs as dir:event")
    args = ap.parse_args()
    import arviz as az

    G = 4.438
    for spec in args.runs:
        d, ev = spec.split(":")
        pre = os.path.join(os.path.abspath(d), f"DC2018_{ev}")
        idata = az.from_netcdf(pre + "_trace.nc")
        post = idata.posterior
        lm = post["star.logmass"].values[..., 0].reshape(-1)
        r = post["star.radius"].values[..., 0].reshape(-1)
        logg = G + lm - 2 * np.log10(r)
        lgs = (
            post["star.loggsed"].values[..., 0].reshape(-1)
            if "star.loggsed" in post
            else logg
        )
        m = 10**lm
        print(f"== {spec}: {lm.size} draws")
        print(
            f"  lens logg  median {np.median(logg):.3f}  [16,84] [{np.percentile(logg, 16):.3f}, {np.percentile(logg, 84):.3f}]  "
            f"max {logg.max():.3f};  frac > 4.9: {np.mean(logg > 4.9):.3f}  > 5.0: {np.mean(logg > 5.0):.4f}"
        )
        print(
            f"  lens loggsed median {np.median(lgs):.3f}  max {lgs.max():.3f}  frac > 5.0: {np.mean(lgs > 5.0):.4f}"
        )
        for lo, hi in (
            (4.5, 4.7),
            (4.7, 4.8),
            (4.8, 4.9),
            (4.9, 5.0),
            (5.0, 5.3),
        ):
            sel = (logg >= lo) & (logg < hi)
            if sel.sum():
                print(
                    f"  logg in [{lo},{hi}): {sel.mean():6.3f} of draws, lens mass median {np.median(m[sel]):.3f}"
                )
        # mass that the grid edge implies for a Mamajek dwarf: logg = 5.0
        print(
            f"  lens mass median {np.median(m):.3f}; a dwarf at logg 5.0 on the sequence is ~0.19 solMass"
        )


if __name__ == "__main__":
    main()
