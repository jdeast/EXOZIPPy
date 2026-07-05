"""Scientific validation of the VBM safedist fix / EXOZIPPy far-field guard.

Scan 1 (containment): max radius of any binary caustic point from the
  coordinate origin over a dense (s, q) grid, compared with
  R_inf = s + 1/s + 2. The guard is geometrically sound iff caustics never
  reach R_inf.

Scan 2 (boundary accuracy): |A_pointsource - A_finitesource| for source
  centers exactly ON the guard boundary d = R_inf + 2*rho (the worst
  admissible case), with strong limb darkening (a1 = 1), against
  BinaryMagDark at Tol = 1e-5 (never shortcuts). The guard is
  photometrically sound iff this is << 1e-3 (VBM's default Tol).
"""
import time
import numpy as np
import VBMicrolensing

vbm = VBMicrolensing.VBMicrolensing()

# ---------------- Scan 1: caustic containment --------------------------------
print("=== Scan 1: caustic containment vs R_inf = s + 1/s + 2 ===")
s_grid = np.logspace(np.log10(0.05), np.log10(20.0), 41)
q_grid = np.logspace(-9, 0, 28)   # VBM internal q <= 1 (swaps for q > 1)
worst_ratio = 0.0
worst = None
rows = []
for s in s_grid:
    for q in q_grid:
        caus = vbm.Caustics(float(s), float(q))
        rmax = 0.0
        for cau in caus:
            x = np.asarray(cau[0]); y = np.asarray(cau[1])
            rmax = max(rmax, float(np.sqrt(x*x + y*y).max()))
        Rinf = s + 1.0/s + 2.0
        ratio = rmax / Rinf
        rows.append((s, q, rmax, Rinf, ratio))
        if ratio > worst_ratio:
            worst_ratio = ratio
            worst = (s, q, rmax, Rinf)
rows = np.array(rows)
print(f"grid: {len(s_grid)} s-values x {len(q_grid)} q-values = {len(rows)} lens geometries")
print(f"max caustic radius / R_inf = {worst_ratio:.4f}  "
      f"at s={worst[0]:.4g}, q={worst[1]:.3g} (r_caustic={worst[2]:.4g}, R_inf={worst[3]:.4g})")
print(f"=> minimum clearance between any caustic and the guard circle: "
      f"{(1-worst_ratio)*100:.1f}% of R_inf")
# where is the bound tightest as a function of s?
for smask, lbl in [(s_grid < 0.5, "close (s<0.5)"),
                   ((s_grid >= 0.5) & (s_grid <= 2), "resonant (0.5<=s<=2)"),
                   (s_grid > 2, "wide (s>2)")]:
    sel = np.isin(rows[:, 0], s_grid[smask])
    print(f"  {lbl:22s}: max ratio {rows[sel, 4].max():.4f}")

# ---------------- Scan 2: boundary accuracy ----------------------------------
print("\n=== Scan 2: |A_ps - A_fs| on the guard boundary d = R_inf + 2*rho ===")
s_vals = [0.1, 0.3, 1.0, 3.0, 10.0]
q_vals = [1e-6, 1e-4, 1e-2, 0.1, 1.0]
rho_vals = [0.01, 0.1, 1.0, 3.0, 10.0]
angles = np.linspace(0, 2*np.pi, 13)[:-1]
worst_abs = 0.0
worst_cfg = None
n = 0
t0 = time.time()
for s in s_vals:
    Rinf = s + 1.0/s + 2.0
    for q in q_vals:
        for rho in rho_vals:
            d = Rinf + 2.0*rho
            for th in angles:
                x, y = float(d*np.cos(th)), float(d*np.sin(th))
                A_ps = vbm.BinaryMag0(s, q, x, y)
                vbm.a1 = 1.0   # strongest limb darkening (worst case)
                A_fs = vbm.BinaryMagDark(s, q, x, y, rho, 1e-5)
                err = abs(A_ps - A_fs)
                n += 1
                if err > worst_abs:
                    worst_abs = err
                    worst_cfg = (s, q, rho, x, y, A_ps, A_fs)
print(f"{n} boundary configurations in {time.time()-t0:.0f}s")
s, q, rho, x, y, A_ps, A_fs = worst_cfg
print(f"max |A_ps - A_fs| = {worst_abs:.3e}  "
      f"(at s={s}, q={q}, rho={rho}, x={x:.3f}, y={y:.3f}: "
      f"A_ps={A_ps:.8f}, A_fs={A_fs:.8f})")
print(f"=> worst boundary error is {worst_abs/1e-3:.3f} x VBM's default Tol (1e-3)")
