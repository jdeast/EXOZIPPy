"""Review 7.6.4: which mapping of Han+2023 Table 2 onto EXOZIPPy's lens
convention do the CLEANED data (I+V difference flux, A_V prior, companions
tied) actually support?  check_start.py scored 32 conventions at their
STARTS on the old outlier-laden magnitude files; here every candidate is
OPTIMIZED (profile_geometry.main: s, q and the companion angles held at the
candidate, everything else free, Powell) and compared on the light-curve
term.

Candidates (48): alpha rule {a+180, 180-a, a, -a} x u_0 sign x psi sense
{alpha_3 = alpha_2 -/+ psi} x s normalization {as published; x k; / k},
k = sqrt(1 + q2 + q3), the ratio of the total-mass to the primary-mass
Einstein radius (in case Table 2's s is in M1's theta_E).  t_0 / u_0 start
at the centre-of-mass-shifted values (check_start.py) and are then free.

    python scan_conventions_opt.py <task 1..48> [maxfev]
"""

import copy
import json
import math
import os
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import profile_geometry as PG  # noqa: E402

A_HAN = math.degrees(2.292)
PSI = math.degrees(1.383)
Q2, Q3, S2, S3, TE, T0, U0 = (
    0.526,
    0.241,
    1.386,
    1.601,
    14.74,
    2459370.226,
    0.126,
)
K = math.sqrt(1.0 + Q2 + Q3)


def candidates():
    out = []
    rules = {
        "a+180": A_HAN + 180.0,
        "180-a": 180.0 - A_HAN,
        "a": A_HAN,
        "-a": -A_HAN,
    }
    for norm, f in (("s", 1.0), ("s*k", K), ("s/k", 1.0 / K)):
        for rule, a2 in rules.items():
            for u0 in (U0, -U0):
                for psi_sign, ptag in ((-1, "a2-psi"), (+1, "a2+psi")):
                    a3 = a2 + psi_sign * PSI
                    s2, s3 = S2 * f, S3 * f
                    e2 = np.array(
                        [
                            math.cos(math.radians(a2)),
                            -math.sin(math.radians(a2)),
                        ]
                    )
                    e3 = np.array(
                        [
                            math.cos(math.radians(a3)),
                            -math.sin(math.radians(a3)),
                        ]
                    )
                    eff = Q2 / ((1.0 + Q2) * s2)
                    com = (Q2 * s2 * e2 + Q3 * s3 * e3) / (1.0 + Q2 + Q3)
                    D = com - eff * e2
                    out.append(
                        {
                            "label": f"{norm} {rule} u0{'+' if u0 > 0 else '-'} {ptag}",
                            "s2": s2,
                            "s3": s3,
                            "a2": a2,
                            "a3": a3,
                            "t0": T0 - D[0] * TE,
                            "u0": u0 + D[1],
                        }
                    )
    return out


def main(task, maxfev):
    c = candidates()[task - 1]
    os.chdir(HERE)
    cfg = yaml.safe_load(open("KMT-2021-BLG-1122L.yaml"))
    prm = yaml.safe_load(open(cfg["parameter_file"]))
    p = copy.deepcopy(prm)
    for k in (
        "lens.LensB.s",
        "lens.LensB.alpha",
        "star.LensB.mass",
        "star.LensC.mass",
    ):
        p.pop(k, None)
    p.update(PG.geometry_pins(c["s2"], Q2, c["a2"], c["s3"], Q3, c["a3"]))
    p["source.Source.t_0"] = {"initval": float(c["t0"])}
    p["source.Source.u_0"] = {"initval": float(c["u0"])}
    print("CANDIDATE", json.dumps(c), flush=True)
    PG.main(
        "pinned",
        maxfev,
        params=p,
        tag=f"task{task}",
        out_npy=os.path.join(HERE, "scan_opt", f"x_{task:02d}.npy"),
    )


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 6000)
