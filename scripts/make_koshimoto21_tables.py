"""Write the tables of galacticmodel's genulens port (koshimoto21.py).

genulens: Koshimoto, Baba & Bennett 2021, ApJ 917, 78;
https://github.com/nkoshimoto/genulens (GPLv3).

- ``koshimoto21_shu_lognorm.npz`` (always): the normalization of genulens's
  Shu distribution function, Z(R, z) = int P(f_g | R, z) df_g, on a 100-pc
  (R, |z|) grid, from ``koshimoto21.log_shu_unnormalized`` (the port of
  calc_PRRg).  genulens computes the same integral (a trapezoid over its
  truncated support) at its own table nodes.
- with ``--genulens ROOT`` (a genulens checkout built with CMake, as
  scripts/make_genulens_fixture.py describes):
  - ``koshimoto21_nsd.npz``: genulens's input_files/NSD_moments.dat (the
    nuclear stellar disk's density and velocity moments from the
    Sormani et al. 2021 distribution function), as arrays on its (|z|, R)
    grid, in the units genulens reads them in;
  - ``koshimoto21_lf.npz``: genulens's I-band luminosity functions per
    component (make_LFs: the PARSEC isochrones iso-*.dat, K21's IMF and
    the population ages), printed by its own code through
    scripts/genulens_reference_driver.cpp.

    python scripts/make_koshimoto21_tables.py [--genulens ROOT]
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

from exozippy.components.galacticmodel import koshimoto21 as k21

R_GRID = np.arange(0.0, 20000.0 + 1.0, 100.0)  # pc
Z_GRID = np.arange(0.0, 4000.0 + 1.0, 100.0)  # pc, |z|


def nsd_table(root):
    data = np.loadtxt(os.path.join(root, "input_files", "NSD_moments.dat"))
    r = np.unique(data[:, 0]) * 1000.0
    z = np.unique(data[:, 1]) * 1000.0
    shape = (z.size, r.size)  # genulens: [iz][iR], R fastest
    cols = {
        "log10_rho": np.log10(data[:, 2]),
        "vphi": data[:, 3],
        "log10_sig_phi": np.log10(data[:, 4]),
        "log10_sig_r": np.log10(data[:, 5]),
        "log10_sig_z": np.log10(data[:, 6]),
        "cor_rz": data[:, 7],
    }
    return {"r": r, "z": z, **{k: v.reshape(shape) for k, v in cols.items()}}


def lf_table(root):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import make_genulens_points as mgp

    with tempfile.TemporaryDirectory() as tmp:
        exe = mgp.build_driver(root, tmp)
        out = subprocess.run(
            [
                exe,
                "l",
                "1.0",
                "b",
                "-3.9",
                "DLIST",
                "8000",
                "NMC",
                "10",
                "AIrc",
                "1.0",
            ],
            check=True,
            capture_output=True,
            text=True,
            cwd=root,
        ).stdout
    lf = json.loads(out)["lf"]
    return {"mi": np.array(lf["MIs"]), "cumu": np.array(lf["CumuN_MIs"])}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--genulens", help="genulens checkout root")
    args = ap.parse_args()
    log_norm = k21.shu_log_norm_grid(R_GRID, Z_GRID)
    np.savez_compressed(
        k21.SHU_NORM_FILE, r=R_GRID, z=Z_GRID, log_norm=log_norm
    )
    print(f"wrote {k21.SHU_NORM_FILE} {log_norm.shape}")
    if args.genulens:
        root = os.path.abspath(args.genulens)
        np.savez_compressed(k21.NSD_FILE, **nsd_table(root))
        print(f"wrote {k21.NSD_FILE}")
        np.savez_compressed(k21.LF_FILE, **lf_table(root))
        print(f"wrote {k21.LF_FILE}")


if __name__ == "__main__":
    main()
