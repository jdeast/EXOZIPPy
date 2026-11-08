"""The R = 3000 runtime grid against the shipped BC tables, at the tables'
own nodes: SpectralBC with the tables' extinction law must reproduce every
column to the R = 3000 sampling error (<= 0.43 mmag in the feasibility
study).  The four interior holes whose SVO files are empty are reported
apart: the tables hold values there from the original spectra, the runtime
grid an interpolation along Teff."""

import sys
from pathlib import Path

import numpy as np

from exozippy.components.sed.bc_grid import (
    DEFAULT_MODEL_ROOT,
    bc_table_filter_columns,
    bc_table_path,
    read_bc_table,
)
from exozippy.components.sed.spectral_bc import SpectralBC
from exozippy.models.NextGen import generate_NextGen_BC_Tables as gen
from exozippy.models.NextGen.runtime_spectra import RuntimeGrid

grid = RuntimeGrid(Path(sys.argv[1]))
holes = {
    (4400.0, 3.0, -3.0),
    (4700.0, 5.0, -2.0),
    (7000.0, 0.0, -1.0),
    (7800.0, 2.0, 0.3),
}
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
    (t, 4.5, 0.0) for t in (2600.0, 3000.0, 4000.0, 5800.0, 8000.0, 10000.0)
] + sorted(holes)
avs = (0.0, 1.0, 8.0, 20.0)
worst, worst_hole = {}, {}
for fac, names in gen.FILTER_SETS.items():
    tab = (
        read_bc_table(bc_table_path(DEFAULT_MODEL_ROOT, "NextGen", fac))
        .set_index(["teff", "logg", "feh", "Av"])
        .sort_index()
    )
    s = SpectralBC(grid, names, law="shipped")
    for t, g, z in nodes:
        pad = (t, g, z) in holes
        for av in avs:
            if (t, g, z, av) not in tab.index:
                continue
            ref = tab.loc[(t, g, z, av), s.filters_mist].to_numpy(float)
            got = s.bc_np(t, g, z, av, clip=False)
            d = np.abs(got - ref) * 1000
            for c, v in zip(s.filters_mist, d):
                bucket = worst_hole if pad else worst
                bucket[c] = max(bucket.get(c, 0.0), float(v))
    print(f"{fac}: done", flush=True)
v = np.array(list(worst.values()))
print(
    f"ordinary nodes: {len(worst)} filters, median of per-filter max {np.median(v):.3f} mmag, worst {max(worst, key=worst.get)} {v.max():.3f} mmag, filters over 1 mmag: {[k for k, x in worst.items() if x > 1]}"
)
h = np.array(list(worst_hole.values()))
print(
    f"the 4 interior holes: median {np.median(h):.1f} mmag, max {h.max():.1f} ({max(worst_hole, key=worst_hole.get)})"
)
print("DONE")
