"""Record genulens's own model functions at chosen points, for the
point-by-point validation of EXOZIPPy's port of the genulens Galactic model.

genulens (Koshimoto, Baba & Bennett 2021, ApJ 917, 78, "K21";
https://github.com/nkoshimoto/genulens, GPLv3) is the reference.  This
script compiles ``scripts/genulens_reference_driver.cpp`` against a genulens
build, runs it on a few sight lines, and writes what it printed -- genulens's
numbers, evaluated by genulens's functions -- to
``tests/fixtures/genulens_points.json``.  No genulens source is copied into
EXOZIPPy.

Where ``scripts/make_genulens_fixture.py`` records genulens's Monte Carlo
EVENT distributions (a distribution-level check), this records the model
itself at fixed points, so a port can be compared term by term:

- ``constants``: the population normalizations genulens derives at start-up
  (disk ``rho0d``/``n0d``/``n0MSd``, bar ``rho0b``/``n0MSb``/``n0b`` from its
  VVV-box integral, halo ``n0SH``, ...), the Sgr A* offset, the solar motion,
  the rotation curve;
- ``density``: per distance, the Galactocentric position and the number
  density of each of the 11 components (7 thin-disk age bins, thick disk,
  bar, NSD, stellar halo), all stars and main-sequence stars, and -- on a
  sight line with an ``AIrc`` source selection -- the luminosity-function
  detection fraction ``fLF`` of each component;
- ``velocity``: per distance and component, the analytic building blocks
  (disk dispersions; bar mean and dispersions; halo dispersions) and the
  moments and quantiles of ``NMC`` draws from genulens's own velocity
  sampler ``get_vxyz_ran`` (escape-speed rejection included);
- ``shu``: the unnormalized Shu distribution P(f_g | R, z) on a fixed f_g
  grid at the table nodes the sight line visits, with genulens's
  normalization and its sampling support [f_g,min, f_g,max];
- ``mass`` (once, top level): the broken-power-law IMF grid, the PARSEC turnoff tables
  (``Minidie``), and the initial-final mass relation (mean and Monte Carlo).

Build genulens with CMake as ``scripts/make_genulens_fixture.py`` describes,
then

    python scripts/make_genulens_points.py /path/to/genulens

(the genulens checkout root, containing ``build/libgenulens_core.a`` and
``input_files/``).  Needs a C++17 compiler and GSL.  Recorded with genulens
2.0.0a4 (commit 2cc5870).
"""

import argparse
import json
import os
import subprocess
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
DRIVER_SRC = os.path.join(HERE, "genulens_reference_driver.cpp")
OUT = os.path.join(HERE, "..", "tests", "fixtures", "genulens_points.json")

# name -> (l, b, extra driver options)
SIGHT_LINES = {
    # examples/ob140939's catalog position
    "ob140939": (6.647997, 3.611349, []),
    # Baade's window, genulens's default sight line
    "baade": (1.0, -3.9, []),
    # Baade's window with the classic luminosity-function source selection
    # (I_s in 14-21, A_I = 1.0 at the red clump)
    "baade_lf": (1.0, -3.9, ["AIrc", "1.0", "Isrange", "14", "21"]),
    # inside |l| < 5, |b| < 2: genulens switches the nuclear stellar disk on
    "nsd": (0.3, -1.6, []),
}
DISTANCES_PC = [
    300,
    1000,
    2000,
    3000,
    4000,
    5000,
    6000,
    7000,
    7600,
    8000,
    8200,
    8500,
    9000,
    10000,
    11000,
    12000,
]
# the velocity Monte Carlo and the Shu tables are large; record them at a
# subset of the distances
VELOCITY_DISTANCES_PC = (1000, 3000, 5000, 7000, 8000, 8500, 9000, 11000)
SHU_DISTANCES_PC = (1000, 4000, 7000, 8200)
# sight lines that differ from another only in the source selection record
# the density block alone
DENSITY_ONLY = ("baade_lf",)
NMC = 20000
SIG = 12  # significant digits kept in the fixture


def build_driver(genulens_root, out_dir):
    exe = os.path.join(out_dir, "genulens_reference_driver")
    cmd = [
        "c++",
        "-std=c++17",
        "-O2",
        "-I" + os.path.join(genulens_root, "src"),
        DRIVER_SRC,
        os.path.join(genulens_root, "build", "libgenulens_core.a"),
        "-lgsl",
        "-lgslcblas",
        "-lm",
        "-ldl",
        # GCC 8 keeps std::filesystem in a separate library
        "-lstdc++fs",
        "-o",
        exe,
    ]
    subprocess.run(cmd, check=True)
    return exe


def run_driver(exe, genulens_root, l_deg, b_deg, extra, seed):
    cmd = [
        exe,
        "l",
        repr(l_deg),
        "b",
        repr(b_deg),
        "DLIST",
        ",".join(str(d) for d in DISTANCES_PC),
        "NMC",
        str(NMC),
        "seed",
        str(seed),
        *extra,
    ]
    out = subprocess.run(
        cmd, check=True, capture_output=True, text=True, cwd=genulens_root
    ).stdout
    return json.loads(out)


def _round(obj):
    if isinstance(obj, float):
        return float(f"{obj:.{SIG}g}")
    if isinstance(obj, list):
        return [_round(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _round(v) for k, v in obj.items()}
    return obj


def slim(rec, density_only):
    """Keep what a port is checked against; drop genulens's sampling tables."""
    rec.pop("lf", None)
    rec.pop("mass")
    if density_only:
        for k in ("velocity", "shu"):
            rec.pop(k)
        return _round(rec)
    shu = []
    for node in rec["shu"]:
        dist = [
            v["D"]
            for v in rec["velocity"]
            if v["comp"] == node["comp"]
            and v.get("R_node") == node["R_node"]
            and v.get("z_node") == node["z_node"]
        ]
        if not any(d in SHU_DISTANCES_PC for d in dist):
            continue
        key = (node["comp"], node["R_node"], node["z_node"])
        if any((n["comp"], n["R_node"], n["z_node"]) == key for n in shu):
            continue
        fg = node.pop("fg")
        node.pop("P_norm")
        node["fg_min"], node["fg_max"], node["n_fg"] = fg[0], fg[-1], len(fg)
        # every 5th point of the 0.01-step grid: 0.05 .. 3.00
        node["fg_grid"] = node["fg_grid"][4::5]
        node["P_unnorm_grid"] = node["P_unnorm_grid"][4::5]
        shu.append(node)
    rec["shu"] = shu
    rec["velocity"] = [
        v for v in rec["velocity"] if v["D"] in VELOCITY_DISTANCES_PC
    ]
    return _round(rec)


def slim_mass(mass):
    """The mass block is the same on every sight line; keep one copy."""
    mass["PlogM_every10"] = mass.pop("PlogM")[::10]
    for k in ("Mini", "Mrem_mean", "type_mean"):
        mass[k] = mass[k][::4]
    return _round(mass)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "genulens_root", help="genulens checkout, built with CMake"
    )
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    root = os.path.abspath(args.genulens_root)
    with tempfile.TemporaryDirectory() as tmp:
        exe = build_driver(root, tmp)
        fixture = {
            "genulens_version": "2.0.0a4 (2cc5870)",
            "driver": "scripts/genulens_reference_driver.cpp",
            "distances_pc": DISTANCES_PC,
            "sight_lines": {},
        }
        for seed, (name, (l_deg, b_deg, extra)) in enumerate(
            SIGHT_LINES.items()
        ):
            rec = run_driver(exe, root, l_deg, b_deg, extra, 1000 + seed)
            if "mass" not in fixture:
                fixture["mass"] = slim_mass(rec["mass"])
            fixture["sight_lines"][name] = slim(rec, name in DENSITY_ONLY)
    with open(args.out, "w") as fh:
        json.dump(fixture, fh, separators=(",", ":"))
        fh.write("\n")
    print(f"wrote {args.out} ({os.path.getsize(args.out) / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
