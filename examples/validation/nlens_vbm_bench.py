"""N-lens VBMicrolensing benchmark and validation (review 7.6.2, queue TASK 5).

Answers three questions the shipped configuration currently answers by
reasoning rather than by measurement:

1. COST.  Wall time per magnification call as a function of the number of
   lens bodies N and of the source's distance from the nearest caustic.
   "MultiMag2 at N >= 3 is ~3x a binary call" is the claim on record; this
   measures it, resolved by caustic proximity, because the cost of an
   adaptive contour integrator is dominated by exactly that distance.

2. THE METHOD DEFAULT.  op.py picks VBM's Multipoly for 3 lenses and Nopoly
   for 4+, citing the VBM docs (Bozza+2025).  That is a shipped default set
   from a paper, not from a measurement on our geometries.  All three
   methods (Multipoly, Nopoly, Singlepoly) are timed at N = 3 and N = 4.

3. ACCURACY, against a reference that is actually independent.  Comparing
   against MulensModel is NOT independent: tests/test_vbm_direct_vs_
   mulensmodel.py agrees to 1e-8 precisely because MulensModel calls the
   same VBM kernel (and review 7.6.3 records that the parity suite's
   finite-source coverage was hollow for the same reason).  So this script
   uses two references that do not involve VBM's contour integrator:

     (a) self-convergence -- the shipped Tol = 1e-3 against Tol = 1e-6;
     (b) inverse ray shooting -- a brute-force numerical solution of the
         lens equation, run at a large source radius (rho = 0.03, a bulge
         giant) where IRS is affordable, on a small subset of positions.

   (b) is the trusted reference.  (a) tells us whether the shipped tolerance
   is the accuracy limit; (b) tells us whether the integrator is right at
   all for N >= 3, which nothing in the suite currently establishes.

Pure compute, no EXOZIPPy model build: it calls VBMicrolensing directly with
the same setup op.py's _build_vbm uses (Tol, RelTol, LD profile, Method), so
what is measured is the kernel the fitter actually pays for.

Run:  python3 nlens_vbm_bench.py [--quick] [--out nlens_vbm_bench.json]
"""

import argparse
import json
import sys
import time

import numpy as np
import VBMicrolensing

# op.py's shipped defaults (VBMDirectMagOp.__init__).
TOL_SHIPPED = 1e-3
RELTOL_SHIPPED = 0.0
TOL_REFERENCE = 1e-6

# Production-scale source size for the timing and self-convergence legs.
RHO_PROD = 1e-3
# Inverse-ray-shooting leg: a source big enough that brute force is affordable.
RHO_IRS = 0.03

METHODS = {
    "Multipoly": VBMicrolensing.VBMicrolensing.Method.Multipoly,
    "Nopoly": VBMicrolensing.VBMicrolensing.Method.Nopoly,
    "Singlepoly": VBMicrolensing.VBMicrolensing.Method.Singlepoly,
}

# Caustic-proximity bins, in units of the source radius rho.  The integrator's
# cost and error both turn on whether the source limb is near a fold.
BIN_EDGES = [0.0, 0.5, 2.0, 10.0, 100.0, np.inf]
BIN_LABELS = ["<0.5", "0.5-2", "2-10", "10-100", ">100"]


# ---------------------------------------------------------------------------
# Geometries.  Positions are (x, y) in Einstein radii of the TOTAL mass and
# masses are fractions summing to 1 -- the same convention op.py builds in
# _magnify (m[0] = 1/(1+q_tot), pos recentred on the centre of mass).
# ---------------------------------------------------------------------------
def _binary(s, q):
    """Two bodies, separation s, mass ratio q, on the x axis, CoM at origin."""
    m0 = 1.0 / (1.0 + q)
    pos = np.array([[0.0, 0.0], [s, 0.0]])
    m = np.array([m0, q * m0])
    pos = pos - m @ pos
    return pos, m


def _nbody(companions):
    """companions: list of (s, q, alpha_deg) relative to the primary."""
    q_tot = sum(q for (_, q, _) in companions)
    m0 = 1.0 / (1.0 + q_tot)
    pos = np.zeros((len(companions) + 1, 2))
    m = np.empty(len(companions) + 1)
    m[0] = m0
    for j, (s, q, alpha_deg) in enumerate(companions):
        a = np.radians(alpha_deg)
        m[j + 1] = q * m0
        pos[j + 1] = (s * np.cos(a), -s * np.sin(a))
    pos = pos - m @ pos
    return pos, m


# Synthetic but physically ordinary geometries; NOT literature solutions.
# Labelled so nobody mistakes them for published events.
#
# The 4th field is (s, q) for the two-body geometries, so the BINARY call path
# (BinaryMag2, which is what op.py actually dispatches at n_companions == 1)
# can be timed alongside MultiMag2 on the same geometry.  That separation is
# the whole point: "MultiMag2 at N >= 3 is ~3x a binary call" conflates the
# cost of MORE BODIES with the cost of leaving the specialised binary kernel,
# and only a same-geometry Binary-vs-Multi pair can tell them apart.
GEOMETRIES = [
    # N=2 references for the "3x a binary call" claim.
    ("N2-planetary", 2, lambda: _binary(1.0, 1e-3), (1.0, 1e-3)),
    ("N2-resonant", 2, lambda: _binary(1.0, 1e-2), (1.0, 1e-2)),
    ("N2-stellar", 2, lambda: _binary(1.0, 0.5), (1.0, 0.5)),
    # N=3: a star with two planets, and a stellar binary with one planet.
    ("N3-two-planets", 3,
     lambda: _nbody([(1.0, 1e-3, 0.0), (1.6, 3e-4, 70.0)]), None),
    ("N3-binary+planet", 3,
     lambda: _nbody([(0.9, 0.4, 0.0), (1.8, 1e-3, 40.0)]), None),
    # N=4: three companions, the regime where op.py switches to Nopoly.
    (
        "N4-three-companions",
        4,
        lambda: _nbody(
            [(0.9, 0.3, 0.0), (1.7, 1e-3, 55.0), (2.4, 5e-4, 200.0)]
        ),
        None,
    ),
]


# ---------------------------------------------------------------------------
def make_vbm(pos, m, method_name, tol):
    """A VBM instance configured exactly as op.py's _build_vbm configures it."""
    v = VBMicrolensing.VBMicrolensing()
    v.Tol = tol
    v.RelTol = RELTOL_SHIPPED
    v.SetLDprofile(VBMicrolensing.VBMicrolensing.LDlinear)
    v.a1 = 0.0
    # SetMethod must precede SetLensGeometry (op.py says so, and it is load
    # bearing).  op.py sets it only for n_companions >= 2 because it never
    # calls MultiMag2 on a two-body geometry -- it dispatches BinaryMag2.
    # MEASURED HERE 2026-08-27: MultiMag2 on a two-body SetLensGeometry with
    # no prior SetMethod SEGFAULTS the interpreter (core dump, not an
    # exception).  Harmless today, but it is a live trap for any future
    # refactor that unifies the binary and N-lens dispatch, so this script
    # always sets it.
    v.SetMethod(METHODS[method_name])
    v.SetLensGeometry(np.column_stack([pos, m]).ravel().tolist())
    return v


def caustic_points(pos, m, method_name):
    """All caustic curve points for this geometry, as an (npts, 2) array."""
    v = make_vbm(pos, m, method_name, TOL_SHIPPED)
    curves = v.Multicaustics()
    xs, ys = [], []
    for curve in curves:
        xs.extend(curve[0])
        ys.extend(curve[1])
    return np.column_stack([np.asarray(xs), np.asarray(ys)])


def min_caustic_distance(points, caustics, chunk=2000):
    """Minimum Euclidean distance from each point to the caustic point set."""
    out = np.empty(len(points))
    for i in range(0, len(points), chunk):
        blk = points[i : i + chunk]
        d = np.hypot(
            blk[:, 0, None] - caustics[None, :, 0],
            blk[:, 1, None] - caustics[None, :, 1],
        )
        out[i : i + chunk] = d.min(axis=1)
    return out


def sample_positions(caustics, rho, n_per_bin, rng):
    """Positions binned by distance-to-caustic in units of rho.

    Sampled from an annular pool around the caustic structure: a plain
    uniform grid puts almost nothing in the innermost bins, which are the
    ones that cost.
    """
    lo = caustics.min(axis=0)
    hi = caustics.max(axis=0)
    span = float(np.max(hi - lo))
    centre = 0.5 * (lo + hi)

    pool = []
    # Near-caustic pool: jitter off actual caustic points.  This is the only
    # practical way to populate the <0.5 rho and 0.5-2 rho bins.
    idx = rng.integers(0, len(caustics), size=20000)
    for scale in (0.3 * rho, rho, 5.0 * rho, 50.0 * rho):
        ang = rng.uniform(0.0, 2.0 * np.pi, size=len(idx))
        r = rng.uniform(0.0, scale, size=len(idx))
        pool.append(
            caustics[idx] + np.column_stack([r * np.cos(ang), r * np.sin(ang)])
        )
    # Far pool: uniform over a box several caustic-spans wide, for >100 rho.
    box = max(3.0 * span, 4.0)
    pool.append(
        centre + rng.uniform(-box, box, size=(20000, 2))
    )
    pool = np.vstack(pool)

    d = min_caustic_distance(pool, caustics)
    dr = d / rho

    out = {}
    for k, label in enumerate(BIN_LABELS):
        sel = np.where((dr >= BIN_EDGES[k]) & (dr < BIN_EDGES[k + 1]))[0]
        if len(sel) == 0:
            continue
        take = rng.choice(sel, size=min(n_per_bin, len(sel)), replace=False)
        out[label] = (pool[take], d[take])
    return out


def time_calls(v, points, rho, budget_s, binary_sq=None):
    """Per-call wall times over these points.

    binary_sq=(s, q) times BinaryMag2 -- the kernel op.py dispatches at
    n_companions == 1.  binary_sq=None times MultiMag2.

    Returns (times, values, n_done).  Stops early if budget_s is exhausted so
    one pathological bin cannot hang the whole sweep.
    """
    times = []
    vals = []
    t_start = time.perf_counter()
    for x, y in points:
        if binary_sq is None:
            t0 = time.perf_counter()
            a = v.MultiMag2(float(x), float(y), rho)
            t1 = time.perf_counter()
        else:
            s, q = binary_sq
            t0 = time.perf_counter()
            a = v.BinaryMag2(s, q, float(x), float(y), rho)
            t1 = time.perf_counter()
        times.append(t1 - t0)
        vals.append(a)
        if t1 - t_start > budget_s:
            break
    return np.asarray(times), np.asarray(vals), len(times)


def stats(times):
    if len(times) == 0:
        return {}
    return {
        "n": int(len(times)),
        "median_us": float(np.median(times) * 1e6),
        "mean_us": float(np.mean(times) * 1e6),
        "p90_us": float(np.percentile(times, 90) * 1e6),
        "max_us": float(np.max(times) * 1e6),
    }


# ---------------------------------------------------------------------------
# Inverse ray shooting: the independent reference.
#
# Lens equation for N point masses, all positions in Einstein radii of the
# total mass and masses as fractions summing to 1:
#     beta = theta - sum_j m_j (theta - theta_j) / |theta - theta_j|^2
# Shoot a dense grid of image-plane points theta, map to beta, and count the
# fraction landing inside the source disk.  A = (rays in disk) * h^2 /
# (pi rho^2).  No contour integration, no VBM.
# ---------------------------------------------------------------------------
def irs_magnification(pos, m, src_x, src_y, rho, rays_per_rho=12.0,
                      half_width=None, chunk=4_000_000):
    h = rho / rays_per_rho
    if half_width is None:
        # Images lie within ~1 Einstein radius of the lenses plus the source
        # offset; pad generously.
        half_width = float(
            np.max(np.hypot(pos[:, 0], pos[:, 1]))
            + np.hypot(src_x, src_y)
            + 2.0
        )
    n = int(2.0 * half_width / h)
    if n > 60000:
        raise ValueError(
            "IRS grid too large (%d per side); raise rho or lower rays_per_rho"
            % n
        )
    axis = -half_width + h * (np.arange(n) + 0.5)
    rho2 = rho * rho
    hit = 0
    rows_per_chunk = max(1, int(chunk // n))
    for r0 in range(0, n, rows_per_chunk):
        ys = axis[r0 : r0 + rows_per_chunk]
        tx = np.broadcast_to(axis[None, :], (len(ys), n))
        ty = np.broadcast_to(ys[:, None], (len(ys), n))
        bx = tx.astype(np.float64).copy()
        by = ty.astype(np.float64).copy()
        for j in range(len(m)):
            dx = tx - pos[j, 0]
            dy = ty - pos[j, 1]
            r2 = dx * dx + dy * dy
            np.maximum(r2, 1e-300, out=r2)
            bx -= m[j] * dx / r2
            by -= m[j] * dy / r2
        d2 = (bx - src_x) ** 2 + (by - src_y) ** 2
        hit += int(np.count_nonzero(d2 <= rho2))
    return hit * h * h / (np.pi * rho2)

# ---------------------------------------------------------------------------
# Block driver.
#
# VBM's kernels can SEGFAULT at particular source positions (measured
# 2026-08-27: Nopoly at N=3 near a caustic takes down the interpreter, and
# MultiMag2 on a 2-body geometry with no prior SetMethod does the same).  A
# crash is a RESULT here, not an accident, so every (geometry, method, leg)
# block runs in its own subprocess: the driver records the signal and carries
# on instead of losing the sweep.  Run with --drive to get the whole matrix.
# ---------------------------------------------------------------------------
def enumerate_blocks():
    blocks = []
    for name, nbodies, _build, binary_sq in GEOMETRIES:
        methods = ["Multipoly"] if nbodies == 2 else list(METHODS)
        for mth in methods:
            blocks.append((name, mth, "timing"))
        if binary_sq is not None:
            blocks.append((name, "BinaryMag2", "timing"))
        shipped = "BinaryMag2" if nbodies == 2 else (
            "Multipoly" if nbodies == 3 else "Nopoly")
        blocks.append((name, shipped, "selfconv"))
        blocks.append((name, shipped, "irs"))
    return blocks


def geometry_by_name(name):
    for entry in GEOMETRIES:
        if entry[0] == name:
            return entry
    raise KeyError(name)


def run_block(name, method, leg, args, rng):
    """One (geometry, method, leg) measurement.  Returns a list of rows."""
    _name, nbodies, build, binary_sq = geometry_by_name(name)
    pos, m = build()
    sq = binary_sq if method == "BinaryMag2" else None
    vbm_method = "Multipoly" if method == "BinaryMag2" else method

    caustics = caustic_points(pos, m, vbm_method)
    rho = RHO_IRS if leg == "irs" else RHO_PROD
    binned = sample_positions(caustics, rho, args.n_per_bin, rng)
    rows = []

    if leg == "timing":
        v = make_vbm(pos, m, vbm_method, TOL_SHIPPED)
        for label in BIN_LABELS:
            if label not in binned:
                continue
            pts, dists = binned[label]
            t, vals, ndone = time_calls(v, pts, RHO_PROD, args.bin_budget,
                                        binary_sq=sq)
            row = {
                "leg": "timing", "geometry": name, "n_bodies": nbodies,
                "method": method, "bin_rho": label,
                "d_caustic_median": float(np.median(dists[:ndone]))
                if ndone else None,
                "A_median": float(np.median(vals)) if ndone else None,
                "truncated": bool(ndone < len(pts)),
            }
            row.update(stats(t))
            rows.append(row)
            print("  %-11s %-7s median %9.1f us  p90 %9.1f us  n=%3d  A~%.3g%s"
                  % (method, label, row.get("median_us", float("nan")),
                     row.get("p90_us", float("nan")), ndone,
                     row.get("A_median") or float("nan"),
                     "  [TRUNCATED]" if row["truncated"] else ""), flush=True)
        return rows

    if leg == "selfconv":
        v_lo = make_vbm(pos, m, vbm_method, TOL_SHIPPED)
        v_hi = make_vbm(pos, m, vbm_method, TOL_REFERENCE)

        def call(v, x, y):
            if sq is None:
                return v.MultiMag2(float(x), float(y), RHO_PROD)
            return v.BinaryMag2(sq[0], sq[1], float(x), float(y), RHO_PROD)

        for label in BIN_LABELS:
            if label not in binned:
                continue
            pts, _d = binned[label]
            sub = pts[: min(40, len(pts))]
            a_lo, a_hi, t_hi = [], [], []
            t_start = time.perf_counter()
            for x, y in sub:
                a_lo.append(call(v_lo, x, y))
                t0 = time.perf_counter()
                a_hi.append(call(v_hi, x, y))
                t_hi.append(time.perf_counter() - t0)
                if time.perf_counter() - t_start > 4.0 * args.bin_budget:
                    break
            a_lo = np.asarray(a_lo[: len(t_hi)])
            a_hi = np.asarray(a_hi[: len(a_lo)])
            good = np.isfinite(a_lo) & np.isfinite(a_hi) & (a_hi != 0)
            rel = np.abs(a_lo[good] - a_hi[good]) / np.abs(a_hi[good])
            row = {
                "leg": "selfconv", "geometry": name, "n_bodies": nbodies,
                "method": method, "bin_rho": label, "n": int(good.sum()),
                "rel_err_median": float(np.median(rel)) if rel.size else None,
                "rel_err_max": float(np.max(rel)) if rel.size else None,
                "ref_median_us": float(np.median(t_hi) * 1e6) if t_hi else None,
            }
            rows.append(row)
            def _f(v):
                # `v or nan` would map an exact 0.0 to nan, and 0.0 is a REAL
                # result here: far from the caustic the point-source shortcut
                # makes the answer tolerance-independent.
                return float("nan") if v is None else v

            print("  selfconv %-7s rel err median %.2e  max %.2e  "
                  "(Tol=1e-6 cost %.1f us)  n=%d"
                  % (label, _f(row["rel_err_median"]),
                     _f(row["rel_err_max"]),
                     _f(row["ref_median_us"]), row["n"]),
                  flush=True)
        return rows

    # leg == "irs": the independent reference.
    anchor = caustics[len(caustics) // 3]
    probes = [
        ("on-caustic", anchor),
        ("2rho-off", anchor + np.array([2.0 * RHO_IRS, 0.0])),
        ("far", anchor + np.array([1.5, 1.5])),
    ]
    v = make_vbm(pos, m, vbm_method, TOL_SHIPPED)
    for tag, (px, py) in probes:
        t0 = time.perf_counter()
        if sq is None:
            a_vbm = v.MultiMag2(float(px), float(py), RHO_IRS)
        else:
            a_vbm = v.BinaryMag2(sq[0], sq[1], float(px), float(py), RHO_IRS)
        t_vbm = time.perf_counter() - t0
        t0 = time.perf_counter()
        a_irs = irs_magnification(pos, m, float(px), float(py), RHO_IRS)
        t_irs = time.perf_counter() - t0
        rel = abs(a_vbm - a_irs) / abs(a_irs) if a_irs else float("nan")
        rows.append({
            "leg": "irs", "geometry": name, "n_bodies": nbodies,
            "method": method, "probe": tag,
            "x": float(px), "y": float(py), "rho": RHO_IRS,
            "A_vbm": float(a_vbm), "A_irs": float(a_irs),
            "rel_diff": float(rel),
            "t_vbm_us": float(t_vbm * 1e6), "t_irs_s": float(t_irs),
        })
        print("  IRS %-11s A_vbm=%.6f  A_irs=%.6f  rel=%.2e  "
              "(vbm %.1f us, irs %.1f s)"
              % (tag, a_vbm, a_irs, rel, t_vbm * 1e6, t_irs), flush=True)
    return rows


def drive(args):
    """Run every block as a subprocess; record crashes as data."""
    import os
    import subprocess
    import tempfile

    blocks = enumerate_blocks()
    if args.quick:
        blocks = [b for b in blocks if b[2] != "irs"]
    rows, crashes = [], []
    print("driving %d blocks" % len(blocks), flush=True)
    for (name, method, leg) in blocks:
        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        tmp.close()
        cmd = [sys.executable, os.path.abspath(__file__),
               "--block", "%s/%s/%s" % (name, method, leg),
               "--out", tmp.name,
               "--n-per-bin", str(args.n_per_bin),
               "--bin-budget", str(args.bin_budget),
               "--seed", str(args.seed)]
        if args.quick:
            cmd.append("--quick")
        print("\n=== %s / %s / %s ===" % (name, method, leg), flush=True)
        try:
            pr = subprocess.run(cmd, timeout=args.block_timeout)
            rc = pr.returncode
        except subprocess.TimeoutExpired:
            rc = None
        if rc == 0:
            try:
                with open(tmp.name) as fh:
                    rows.extend(json.load(fh)["rows"])
            except Exception as exc:  # noqa: BLE001
                crashes.append({"geometry": name, "method": method,
                                "leg": leg, "failure": "unreadable output: %r"
                                % (exc,)})
        else:
            how = ("TIMEOUT after %gs" % args.block_timeout if rc is None
                   else ("SIGNAL %d (crash)" % -rc if rc < 0
                         else "exit %d" % rc))
            crashes.append({"geometry": name, "method": method, "leg": leg,
                            "failure": how})
            print("  *** BLOCK FAILED: %s ***" % how, flush=True)
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
    return rows, crashes


def summarize(rows, crashes):
    timing = [r for r in rows if r.get("leg") == "timing"]
    print("\n=== COST vs N at the SHIPPED dispatch, median us/call ===",
          flush=True)
    shipped_of = {2: "BinaryMag2", 3: "Multipoly", 4: "Nopoly"}
    print("%-22s %-11s %s" % ("geometry", "method",
                              "  ".join("%9s" % b for b in BIN_LABELS)),
          flush=True)
    for name, nbodies, _b, _sq in GEOMETRIES:
        meth = shipped_of.get(nbodies, "Multipoly")
        cells = []
        for label in BIN_LABELS:
            hit = [r for r in timing if r["geometry"] == name
                   and r["method"] == meth and r["bin_rho"] == label]
            cells.append("%9.1f" % hit[0]["median_us"]
                         if hit and "median_us" in hit[0] else "%9s" % "-")
        print("%-22s %-11s %s" % (name, meth, "  ".join(cells)), flush=True)

    print("\n=== METHOD A/B at N>=3, median us/call ===", flush=True)
    for name, nbodies, _b, _sq in GEOMETRIES:
        if nbodies < 3:
            continue
        for mth in METHODS:
            cells = []
            for label in BIN_LABELS:
                hit = [r for r in timing if r["geometry"] == name
                       and r["method"] == mth and r["bin_rho"] == label]
                cells.append("%9.1f" % hit[0]["median_us"]
                             if hit and "median_us" in hit[0]
                             else "%9s" % "-")
            print("%-22s %-11s %s" % (name, mth, "  ".join(cells)), flush=True)

    if crashes:
        print("\n=== BLOCKS THAT FAILED (these are findings) ===", flush=True)
        for c in crashes:
            print("  %-22s %-11s %-9s %s" % (c["geometry"], c["method"],
                                             c["leg"], c["failure"]),
                  flush=True)
    else:
        print("\nno blocks failed.", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="small n_per_bin and no IRS leg (smoke test)")
    ap.add_argument("--drive", action="store_true",
                    help="run every block in its own subprocess (the real run)")
    ap.add_argument("--block", default=None,
                    help="run ONE block: GEOMETRY/METHOD/LEG")
    ap.add_argument("--out", default="nlens_vbm_bench.json")
    ap.add_argument("--n-per-bin", type=int, default=120)
    ap.add_argument("--bin-budget", type=float, default=60.0,
                    help="seconds per (geometry, method, bin) timing block")
    ap.add_argument("--block-timeout", type=float, default=3600.0,
                    help="seconds before the driver kills one block")
    ap.add_argument("--seed", type=int, default=20260827)
    args = ap.parse_args()

    if args.quick:
        args.n_per_bin = min(args.n_per_bin, 12)
        args.bin_budget = min(args.bin_budget, 5.0)

    meta = {
        "tol_shipped": TOL_SHIPPED, "tol_reference": TOL_REFERENCE,
        "reltol": RELTOL_SHIPPED, "rho_production": RHO_PROD,
        "rho_irs": RHO_IRS, "n_per_bin": args.n_per_bin,
        "bin_budget_s": args.bin_budget, "bin_labels": BIN_LABELS,
        "bin_edges_in_rho": [float(e) for e in BIN_EDGES],
        "seed": args.seed, "vbm_file": VBMicrolensing.__file__,
    }

    if args.block:
        name, method, leg = args.block.split("/")
        rng = np.random.default_rng(args.seed)
        rows = run_block(name, method, leg, args, rng)
        with open(args.out, "w") as fh:
            json.dump({"meta": meta, "rows": rows}, fh, indent=1)
        return

    if not args.drive:
        print("Nothing to do: pass --drive for the full sweep, or --block "
              "GEOMETRY/METHOD/LEG for one measurement.", flush=True)
        return

    rows, crashes = drive(args)
    with open(args.out, "w") as fh:
        json.dump({"meta": meta, "rows": rows, "crashes": crashes}, fh,
                  indent=1)
    print("\nwrote %s (%d rows, %d failed blocks)"
          % (args.out, len(rows), len(crashes)), flush=True)
    summarize(rows, crashes)


if __name__ == "__main__":
    main()
