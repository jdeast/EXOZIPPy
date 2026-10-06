"""How often does VBMicrolensing's three-body MultiMag2 kill its process,
and on what geometry?  The first acceptance fit (job 15506216) hung in the
seed polish with 4 of 64 proposals never returning and three "double free
or corruption (!prev)" aborts on stderr: a worker that aborts takes its
task with it, and the polish (no eval_timeout by design, run.md) waits
forever.  This evaluates DE-polish-like proposals around the shipped start
in CHILD processes that journal each proposal before evaluating it, so a
dead child names the proposal that killed it; the parent restarts from the
next index.  Each crash is then re-tried under VBM's Nopoly method.
"""

import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HANG_S = (
    120.0  # a three-body call not back in two minutes is hung (typical: ms)
)
COORDS = "263.96292 -28.44645"
T0, U0, TE, RHO = 2459370.609, -0.4213, 14.74, 0.0025
S2, Q2, A2 = 1.386, 0.526, 48.68
S3, Q3, A3 = 1.601, 0.241, 127.92


def proposals(n, seed):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        z = rng.standard_normal(12)
        out.append(
            [
                T0 + 0.5 * z[0],
                U0 + 0.3 * z[1],
                max(TE * 10 ** (0.15 * z[2]), 0.5),
                0.0,
                0.0,
                RHO * 10 ** (0.5 * z[3]),
                S2 * 10 ** (0.3 * z[4]),
                Q2 * 10 ** (0.5 * z[5]),
                A2 + 60.0 * z[6],
                S3 * 10 ** (0.3 * z[7]),
                Q3 * 10 ** (0.5 * z[8]),
                A3 + 60.0 * z[9],
                0.5,
            ]
        )
    return out


def epochs():
    t = np.concatenate(
        [
            np.loadtxt(os.path.join(HERE, f"n20210603.I.KMT{s}14.pys"))[:, 0]
            for s in ("C", "S", "A")
        ]
    )
    return np.sort(t)


def child(start, stop, seed, journal, method):
    import pytensor
    import pytensor.tensor as pt
    import VBMicrolensing

    from exozippy.components.mulensing.op import VBMDirectMagOp

    op = VBMDirectMagOp(
        coords=COORDS, n_companions=2, use_rho=True, bandpass="I"
    )
    if method == "Nopoly":
        op._vbm.SetMethod(VBMicrolensing.VBMicrolensing.Method.Nopoly)
    p = pt.dvector("p")
    tt = pt.dvector("t")
    o = pt.dmatrix("o")
    f = pytensor.function([p, tt, o], op(p, tt, o))
    t = epochs()
    obs = np.zeros((t.size, 3))
    props = proposals(stop, seed)
    with open(journal, "a") as j:
        for i in range(start, stop):
            j.write(f"START {i} {time.time():.1f}\n")
            j.flush()
            A = f(np.array(props[i]), t, obs)
            j.write(f"DONE {i} {np.nanmax(A):.4g} {int(np.isnan(A).sum())}\n")
            j.flush()


def parent(n, seed, method, workers):
    import concurrent.futures as cf

    journal = os.path.join(HERE, f"check_vbm_crash_{method}_{seed}.journal")
    if os.path.exists(journal):
        os.remove(journal)
    open(journal, "a").close()
    crashes = []

    def run_range(lo, hi):
        start = lo
        while start < hi:
            proc = subprocess.Popen(
                [
                    sys.executable,
                    __file__,
                    "child",
                    str(start),
                    str(hi),
                    str(seed),
                    journal,
                    method,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            hung = None
            while proc.poll() is None:
                time.sleep(2.0)
                started, done = {}, set()
                for ln in open(journal):
                    w = ln.split()
                    if w[0] == "START" and start <= int(w[1]) < hi:
                        started[int(w[1])] = float(w[2])
                    elif w[0] == "DONE":
                        done.add(int(w[1]))
                live = [(i, t) for i, t in started.items() if i not in done]
                if live and time.time() - max(t for _, t in live) > HANG_S:
                    hung = max(i for i, _ in live)
                    proc.kill()
                    break
            _, stderr = proc.communicate()
            if proc.returncode == 0:
                return
            lines = [ln for ln in open(journal) if ln.startswith("START")]
            mine = [
                int(ln.split()[1])
                for ln in lines
                if start <= int(ln.split()[1]) < hi
            ]
            err = (stderr.strip().splitlines() or ["?"])[-1][:160]
            if hung is not None:
                err = f"HANG > {HANG_S:.0f} s (killed)"
            if not mine or max(mine) < start:
                print(
                    f"child for [{start}, {hi}) failed before its first proposal: {err}",
                    flush=True,
                )
                return
            last = max(mine)
            crashes.append((last, proc.returncode, err))
            print(f"   #{last}: {err}", flush=True)
            start = last + 1

    chunk = int(np.ceil(n / workers))
    with cf.ThreadPoolExecutor(workers) as ex:
        list(
            ex.map(
                lambda k: run_range(k * chunk, min(n, (k + 1) * chunk)),
                range(workers),
            )
        )
    done = sum(1 for ln in open(journal) if ln.startswith("DONE"))
    props = proposals(n, seed)
    print(
        f"[{method}] {n} proposals, {done} completed, {len(crashes)} crashed  (rate {len(crashes) / n:.2e})"
    )
    for idx, rc, err in sorted(crashes):
        p = props[idx]
        print(f"   crash #{idx} rc={rc} {err}")
        print(
            f"      t0-T0 {p[0] - T0:+.3f} u0 {p[1]:+.3f} tE {p[2]:.2f} rho {p[5]:.2e} | s2 {p[6]:.3f} q2 {p[7]:.3f} a2 {p[8]:.1f} | s3 {p[9]:.3f} q3 {p[10]:.3f} a3 {p[11]:.1f}"
        )
    json.dump(
        {
            "method": method,
            "n": n,
            "crashes": [
                {"idx": i, "rc": rc, "err": e, "p": props[i]}
                for i, rc, e in crashes
            ],
        },
        open(os.path.join(HERE, f"check_vbm_crash_{method}_{seed}.json"), "w"),
        indent=1,
    )
    return crashes


def retry_under(crashes, seed, method):
    """Re-evaluate the crashing / hanging proposals under another method,
    each in its own child with the same hang budget."""
    n_ok = 0
    for idx, _, _ in crashes:
        journal = os.path.join(HERE, f"check_vbm_crash_retry_{method}.journal")
        cmd = [
            sys.executable,
            __file__,
            "child",
            str(idx),
            str(idx + 1),
            str(seed),
            journal,
            method,
        ]
        try:
            r = subprocess.run(
                cmd, capture_output=True, text=True, timeout=HANG_S
            )
            verdict = (
                "ok" if r.returncode == 0 else f"CRASHED rc={r.returncode}"
            )
            n_ok += r.returncode == 0
        except subprocess.TimeoutExpired:
            verdict = "HANG (killed)"
        print(f"   retry #{idx} under {method}: {verdict}", flush=True)
    print(f"[{method}] retried {len(crashes)} crashers: {n_ok} survive")


if __name__ == "__main__":
    if sys.argv[1] == "child":
        child(
            int(sys.argv[2]),
            int(sys.argv[3]),
            int(sys.argv[4]),
            sys.argv[5],
            sys.argv[6],
        )
    else:
        n, seed, workers = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
        method = sys.argv[4] if len(sys.argv) > 4 else "Multipoly"
        other = "Nopoly" if method == "Multipoly" else "Multipoly"
        crashes = parent(n, seed, method, workers)
        if crashes:
            retry_under(crashes, seed, other)
