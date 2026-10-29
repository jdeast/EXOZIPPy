"""Record genulens's microlensing-event distributions for the galactic-model
distribution test (tests/test_galactic_genulens.py).

genulens (Koshimoto, Baba & Bennett 2021, ApJ 917, 78;
https://github.com/nkoshimoto/genulens) is the public C++ simulator of the
Galactic model EXOZIPPy's galacticmodel component simplifies.  This script
runs its command-line simulator on two sight lines and stores, per sight
line:

- ``source_distance_kpc``: equal-weight resampled source distances (genulens
  draws D_S with its own source selection; the test feeds these to
  EXOZIPPy's prior so both codes see the same sources),
- ``lens_mass_msun``: equal-weight draws from genulens's IMF (its output
  masses are rate-weighted by sqrt(M); that weight is divided back out),
- the rate-weighted quantiles of the lens distance D_L (kpc), the
  heliocentric relative proper motion mu_rel (mas/yr, genulens's ``muhel``)
  and t_E = theta_E / mu_rel_hel (days), at ``probabilities``.

Build genulens (C++ compiler, CMake, GSL; the Python extension is not
needed), then:

    cmake -S genulens -B genulens/build -DGENULENS_BUILD_PYTHON=OFF
    cmake --build genulens/build --target genulens
    python scripts/make_genulens_fixture.py genulens/build/genulens

(With GCC 8, add ``-DCMAKE_CXX_STANDARD_LIBRARIES=-lstdc++fs``.)  Recorded
2026-10-28 with genulens 2.0.0a4 (commit 2cc5870), default model options
(CenSgrA = 1, REMNANT = 0, BINARY = 0).
"""

import argparse
import json
import os
import subprocess

import numpy as np

SIGHT_LINES = {
    # examples/ob140939's catalog position
    "ob140939": (6.647997, 3.611349, 11),
    # Baade's window
    "baade": (1.0, -3.9, 12),
}
PROBABILITIES = [round(0.05 * k, 2) for k in range(1, 20)]
COLUMNS = (
    "wtj M_L D_L D_S t_E theta_E pi_E pi_EN pi_EE mu_rel mu_Sl mu_Sb "
    "I_L K_L iS iL fREM muhel_N muhel_E muhel"
).split()
N_SIMU = 40000
N_KEEP = 2000

OUT = os.path.join(
    os.path.dirname(__file__), "..", "tests", "fixtures", "genulens.json"
)


def weighted_quantiles(x, w, probs):
    o = np.argsort(x)
    c = np.cumsum(w[o]) / w.sum()
    return [float(np.interp(p, c, x[o])) for p in probs]


def run(binary, l_deg, b_deg, seed):
    cmd = [
        binary,
        "l",
        str(l_deg),
        "b",
        str(b_deg),
        "Nsimu",
        str(N_SIMU),
        "seed",
        str(seed),
        "VERBOSITY",
        "4",
    ]
    txt = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(binary))),
    ).stdout
    rows = [ln.split() for ln in txt.splitlines() if ln and ln[0] != "#"]
    a = np.array(rows, dtype=float)
    return " ".join(cmd[1:]), {c: a[:, i] for i, c in enumerate(COLUMNS)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("binary", help="path to the genulens executable")
    args = ap.parse_args()
    rng = np.random.default_rng(171)
    out = {
        "source": "genulens 2.0.0a4 (commit 2cc5870), Koshimoto+2021",
        "probabilities": PROBABILITIES,
        "sight_lines": {},
    }
    for name, (l_deg, b_deg, seed) in SIGHT_LINES.items():
        cmd, g = run(os.path.abspath(args.binary), l_deg, b_deg, seed)
        w = g["wtj"]
        p_src = w / w.sum()
        p_imf = w / np.sqrt(g["M_L"])
        p_imf /= p_imf.sum()
        t_e = g["theta_E"] / g["muhel"] * 365.25
        out["sight_lines"][name] = {
            "l_deg": l_deg,
            "b_deg": b_deg,
            "command": cmd,
            "source_distance_kpc": np.round(
                g["D_S"][rng.choice(w.size, N_KEEP, p=p_src)] / 1e3, 4
            ).tolist(),
            "lens_mass_msun": np.round(
                g["M_L"][rng.choice(w.size, N_KEEP, p=p_imf)], 6
            ).tolist(),
            "quantiles": {
                "D_L": weighted_quantiles(g["D_L"] / 1e3, w, PROBABILITIES),
                "mu_rel": weighted_quantiles(g["muhel"], w, PROBABILITIES),
                "t_E": weighted_quantiles(t_e, w, PROBABILITIES),
            },
        }
    with open(OUT, "w") as fh:
        json.dump(out, fh)
        fh.write("\n")
    print(f"wrote {os.path.normpath(OUT)}")


if __name__ == "__main__":
    main()
