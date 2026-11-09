"""The R = 3000 runtime grid against the shipped BC tables, at the tables'
own nodes: SpectralBC with the tables' extinction law must reproduce every
column to the R = 3000 sampling error (<= 0.43 mmag in the feasibility
study).  The four interior holes whose SVO files are empty are reported
apart: the tables hold values there from the original spectra, the runtime
grid an interpolation along Teff."""

import argparse
from pathlib import Path

import numpy as np

from exozippy.components.sed.bc_grid import (
    DEFAULT_MODEL_ROOT,
    bc_table_path,
    read_bc_table,
)
from exozippy.components.sed.spectral_bc import SpectralBC
from exozippy.models.NextGen import generate_NextGen_BC_Tables as gen
from exozippy.models.NextGen.runtime_spectra import RuntimeGrid

HOLES = {
    (4400.0, 3.0, -3.0),
    (4700.0, 5.0, -2.0),
    (7000.0, 0.0, -1.0),
    (7800.0, 2.0, 0.3),
}
AVS = (0.0, 1.0, 8.0, 20.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "grid_dir",
        help="directory holding NextGen.R3000.logflux.npy and its meta.json",
    )
    grid = RuntimeGrid(Path(ap.parse_args().grid_dir))

    rng = np.random.default_rng(3)
    nodes = [
        (
            float(rng.choice(grid.axes["teff"])),
            float(rng.choice(grid.axes["logg"])),
            float(rng.choice(grid.axes["feh"])),
        )
        for _ in range(60)
    ]
    nodes += [
        (t, 4.5, 0.0)
        for t in (2600.0, 3000.0, 4000.0, 5800.0, 8000.0, 10000.0)
    ] + sorted(HOLES)
    worst, worst_hole = {}, {}
    for fac, names in gen.FILTER_SETS.items():
        tab = (
            read_bc_table(bc_table_path(DEFAULT_MODEL_ROOT, "NextGen", fac))
            .set_index(["teff", "logg", "feh", "Av"])
            .sort_index()
        )
        s = SpectralBC(grid, names, law="shipped")
        for t, g, z in nodes:
            bucket = worst_hole if (t, g, z) in HOLES else worst
            for av in AVS:
                if (t, g, z, av) not in tab.index:
                    continue
                ref = tab.loc[(t, g, z, av), s.filters_mist].to_numpy(float)
                got = s.bc_np(t, g, z, av, clip=False)
                for c, v in zip(s.filters_mist, np.abs(got - ref) * 1000):
                    bucket[c] = max(bucket.get(c, 0.0), float(v))
        print(f"{fac}: done", flush=True)
    v = np.array(list(worst.values()))
    over = [k for k, x in worst.items() if x > 1]
    print(
        f"ordinary nodes: {len(worst)} filters, median of per-filter max "
        f"{np.median(v):.3f} mmag, worst {max(worst, key=worst.get)} "
        f"{v.max():.3f} mmag, filters over 1 mmag: {over}"
    )
    h = np.array(list(worst_hole.values()))
    print(
        f"the 4 interior holes: median {np.median(h):.1f} mmag, max "
        f"{h.max():.1f} ({max(worst_hole, key=worst_hole.get)})"
    )
    print("DONE")


if __name__ == "__main__":
    main()
