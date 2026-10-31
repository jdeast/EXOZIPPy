"""Write galacticmodel/koshimoto21_shu_lognorm.npz: the normalization of the
genulens Shu distribution function on an (R, |z|) grid.

The port of genulens (Koshimoto, Baba & Bennett 2021, ApJ 917, 78;
https://github.com/nkoshimoto/genulens) evaluates the disk's v_phi density
at a star's own (R, z), so it needs Z(R, z) = int P(f_g | R, z) df_g there.
genulens computes the same integral (a trapezoid over its truncated support)
at its table nodes; this computes it from ``koshimoto21.log_shu_unnormalized``
(the port of calc_PRRg) on a 100-pc grid that the model interpolates
bilinearly in log Z.

    python scripts/make_koshimoto21_tables.py
"""

import argparse

import numpy as np

from exozippy.components.galacticmodel import koshimoto21 as k21

R_GRID = np.arange(0.0, 20000.0 + 1.0, 100.0)  # pc
Z_GRID = np.arange(0.0, 4000.0 + 1.0, 100.0)  # pc, |z|


def main():
    argparse.ArgumentParser(description=__doc__.split("\n\n")[0]).parse_args()
    log_norm = k21.shu_log_norm_grid(R_GRID, Z_GRID)
    np.savez_compressed(
        k21.SHU_NORM_FILE, r=R_GRID, z=Z_GRID, log_norm=log_norm
    )
    print(f"wrote {k21.SHU_NORM_FILE} {log_norm.shape}")


if __name__ == "__main__":
    main()
