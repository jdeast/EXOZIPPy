"""Minimal, EXOZIPPy-free reproducer for the VBMicrolensing three-lens
failures found by check_vbm_crash.py: aborts ("double free or corruption",
"corrupted size vs. prev_size"), segfaults and hangs in MultiMag2 /
MultiMag0 under Multipoly and Nopoly.  Each case runs in its own child
process so one failure cannot take the others down; the parent reports
the exit status per case.

Geometry is exactly what EXOZIPPy hands VBM: the source moves in the
trajectory frame at (-tau, -u); companion j sits at s_j (cos a_j, -sin a_j)
from the primary; the origin is the centre of mass; mass fractions sum to
one.  Epochs are the 2434 KMTNet epochs of KMT-2021-BLG-1122L (June 2021),
but the failures reproduce on a uniform grid too (set EPOCHS = "grid").

    python vbm_reproducer.py            # all cases, both methods
    python vbm_reproducer.py child <method> <case index>   # one case, in-process
"""

import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EPOCHS = "kmt"  # or "grid"
HANG_S = 120.0

# (method under which it fails, parameters).  p = [t_0, u_0, t_E, pi_E_N,
# pi_E_E, rho, s2, q2, alpha2_deg, s3, q3, alpha3_deg, u1]
CASES = []
for meth in ("Multipoly", "Nopoly"):
    f = os.path.join(HERE, f"check_vbm_crash_{meth}_1.json")
    if os.path.exists(f):
        for c in json.load(open(f))["crashes"]:
            CASES.append((meth, c["idx"], c["err"], c["p"]))


def epochs():
    if EPOCHS == "grid":
        return 2459370.609 + np.linspace(-6.0, 6.0, 2434)
    t = np.concatenate(
        [
            np.loadtxt(os.path.join(HERE, f"n20210603.I.KMT{s}14.pys"))[:, 0]
            for s in ("C", "S", "A")
        ]
    )
    return np.sort(t)


def run_case(method, p):
    import VBMicrolensing

    vbm = VBMicrolensing.VBMicrolensing()
    vbm.Tol = 1e-3
    vbm.RelTol = 0.0
    vbm.a1 = float(p[12])
    vbm.SetMethod(getattr(VBMicrolensing.VBMicrolensing.Method, method))
    t0, u0, tE, _, _, rho = p[:6]
    comps = [(p[6], p[7], np.radians(p[8])), (p[9], p[10], np.radians(p[11]))]
    q_tot = sum(q for _, q, _ in comps)
    m = np.empty(3)
    pos = np.zeros((3, 2))
    m[0] = 1.0 / (1.0 + q_tot)
    for j, (s, q, a) in enumerate(comps):
        m[j + 1] = q * m[0]
        pos[j + 1] = (s * np.cos(a), -s * np.sin(a))
    pos -= m @ pos
    vbm.SetLensGeometry(np.column_stack([pos, m]).ravel().tolist())
    t = epochs()
    tau = (t - t0) / tE
    x, y = -tau, -u0 * np.ones_like(tau)
    r_inf = max(s + 1.0 / s for s, _, _ in comps) + 2.0
    far = (x * x + y * y) > (r_inf + 2.0 * rho) ** 2
    out = np.empty(t.size)
    for i in range(t.size):
        out[i] = (
            vbm.MultiMag0(x[i], y[i])
            if far[i]
            else vbm.MultiMag2(x[i], y[i], rho)
        )
    return out


def main():
    print(
        f"VBMicrolensing {__import__('VBMicrolensing').__version__ if hasattr(__import__('VBMicrolensing'), '__version__') else '?'}; {len(CASES)} cases"
    )
    for k, (meth, idx, err, p) in enumerate(CASES):
        for method in ("Multipoly", "Nopoly"):
            t0 = time.time()
            try:
                r = subprocess.run(
                    [sys.executable, __file__, "child", method, str(k)],
                    capture_output=True,
                    text=True,
                    timeout=HANG_S,
                )
                tail = (r.stderr.strip().splitlines() or [""])[-1][:60]
                status = (
                    "ok" if r.returncode == 0 else f"rc={r.returncode} {tail}"
                )
            except subprocess.TimeoutExpired:
                status = f"HANG > {HANG_S:.0f} s"
            print(
                f"case {k:2d} (crash #{idx} under {meth}: {err[:28]:28s}) {method:9s}: {status}  [{time.time() - t0:.1f} s]",
                flush=True,
            )
        print(
            f"        p = {np.array2string(np.array(p), precision=4, max_line_width=200)}"
        )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "child":
        A = run_case(sys.argv[2], CASES[int(sys.argv[3])][3])
        print(f"max A {np.nanmax(A):.4g}, nan {int(np.isnan(A).sum())}")
    else:
        main()
