"""Our Galactic model's implied mu_rel distribution, against the answer key's.

The first version of this test assumed BOTH stars were bulge members, which
understates our own model's width: the simulation mixes bulge sources with
disk lenses, and a disk lens carries the 218 km/s rotation, producing large
relative motions two co-rotating bulge stars never make.  JDE called that
out.  This draws each star's component from OUR OWN density mixture along
the event's sight line instead.

Everything here is transcribed from galacticmodel.build_likelihood's three
branches (log_dens_thin / log_dens_thick / log_dens_bulge and their
log_vel_* partners), so it is the same model, evaluated in numpy:

  * mixture weight per component ~ its NUMBER DENSITY at that (l, b, d).
    Legitimate because each branch's velocity Gaussian is normalized by
    -log(s1*s2*s3) and the shared (2*pi)^(3/2) is dropped identically, so
    marginalizing velocity leaves the density.
  * velocities are drawn in the POLAR frame (v_r, v_phi, v_z) with the
    branch's diagonal sigmas and its rotation centre -- thin 218, thick 170
    (asymmetric drift), bulge Omega*r.
  * the draw is mapped to (pm_ra, pm_dec) through the SAME affine map the
    likelihood inverts, line_of_sight_basis's (M_rot, v0).

mu_rel is then |mu_lens - mu_source|, which is what the answer key's murel
column holds.  The Sun's motion cancels in the difference; it is carried
anyway because the map is shared.

THE RAW DRAWS ARE NOT COMPARABLE TO THE KEY.  The key's rows are DETECTED
events, and the microlensing event rate carries a factor of mu_rel, so the
key's murel distribution is the underlying one tilted toward fast events.
The fit applies that tilt too -- mulensevent.py's event_rate_prior is
log(mu_rel) + log(theta_E) -- so leaving it out of this script compares a
rate-selected sample against an unselected one and makes our model look
~1.3x colder than it is (median 4.30 unweighted vs 5.66 weighted, against
the key's 7.40).  --rate (the default) reweights each event's draws by
mu_rel; theta_E is constant per event once the geometry is the key's own,
so it drops out of the within-event weight.  --no-rate reproduces the old,
misleading numbers and is kept only for comparison.

The geometry is the simulation's own (D_s, D_l, l, b per event), so only the
KINEMATICS are ours -- the point of the test.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "src"))
import dc18_common as C  # noqa: E402

from exozippy.components.galacticmodel.physics import (  # noqa: E402
    galactic_xyz,
    line_of_sight_basis,
)
from exozippy.constants import (  # noqa: E402
    BULGE_BAR_ANGLE,
    BULGE_CENTRAL_NUMBER_DENSITY,
    BULGE_DENSITY_X_0,
    BULGE_DENSITY_Y_0,
    BULGE_DENSITY_Z_0,
    BULGE_RC,
    BULGE_RC_WIDTH,
    BULGE_ROTATION_ANGULAR_VELOCITY,
    BULGE_VELOCITY_SIGMA_1,
    BULGE_VELOCITY_SIGMA_2,
    BULGE_VELOCITY_SIGMA_3,
    DISK_LOCAL_NUMBER_DENSITY,
    DISK_RDBREAK,
    DISK_ROTATION_VELOCITY,
    DISK_SCALE_HEIGHT,
    DISK_SCALE_LENGTH,
    DISK_VELOCITY_SIGMA_U,
    DISK_VELOCITY_SIGMA_V,
    DISK_VELOCITY_SIGMA_W,
    K_VEL_CONVERSION,
    SUN_GC_DISTANCE,
    THICK_DISK_LOCAL_NUMBER_DENSITY,
    THICK_DISK_ROTATION_VELOCITY,
    THICK_DISK_SCALE_HEIGHT,
    THICK_DISK_SCALE_LENGTH,
    THICK_DISK_VELOCITY_SIGMA_U,
    THICK_DISK_VELOCITY_SIGMA_V,
    THICK_DISK_VELOCITY_SIGMA_W,
)


def _hinge(t):
    return 0.5 * (t + np.sqrt(t * t + 0.0025))


def branch_log_density(x, y, z):
    """log number density of each branch, transcribed from the likelihood."""
    r = np.hypot(x, y)
    z_smooth = np.sqrt(z**2 + 1e-6)
    r_beyond_sun = SUN_GC_DISTANCE - DISK_RDBREAK
    thin = (
        np.log(DISK_LOCAL_NUMBER_DENSITY)
        - (_hinge(r - DISK_RDBREAK) - r_beyond_sun) / DISK_SCALE_LENGTH
        - z_smooth / DISK_SCALE_HEIGHT
    )
    thick = (
        np.log(THICK_DISK_LOCAL_NUMBER_DENSITY)
        - (_hinge(r - DISK_RDBREAK) - r_beyond_sun) / THICK_DISK_SCALE_LENGTH
        - z_smooth / THICK_DISK_SCALE_HEIGHT
    )
    x_bar = x * np.cos(BULGE_BAR_ANGLE) + y * np.sin(BULGE_BAR_ANGLE)
    y_bar = -x * np.sin(BULGE_BAR_ANGLE) + y * np.cos(BULGE_BAR_ANGLE)
    r_b = np.sqrt(
        (x_bar / BULGE_DENSITY_X_0) ** 2
        + (y_bar / BULGE_DENSITY_Y_0) ** 2
        + (z / BULGE_DENSITY_Z_0) ** 2
    )
    bulge = (
        np.log(BULGE_CENTRAL_NUMBER_DENSITY)
        - 0.5 * r_b
        - 0.5 * (_hinge(r - BULGE_RC) / BULGE_RC_WIDTH) ** 2
    )
    return np.stack([thin, thick, bulge], axis=-1)


_SIG = np.array(
    [
        [DISK_VELOCITY_SIGMA_U, DISK_VELOCITY_SIGMA_V, DISK_VELOCITY_SIGMA_W],
        [
            THICK_DISK_VELOCITY_SIGMA_U,
            THICK_DISK_VELOCITY_SIGMA_V,
            THICK_DISK_VELOCITY_SIGMA_W,
        ],
        [
            BULGE_VELOCITY_SIGMA_1,
            BULGE_VELOCITY_SIGMA_2,
            BULGE_VELOCITY_SIGMA_3,
        ],
    ]
)


def draw_pm(rng, dist_kpc, basis, comp):
    """Draw (pm_ra, pm_dec) mas/yr for stars of component `comp`."""
    m_rot, v0, cl, sl, sb = basis
    x, y, z = galactic_xyz(dist_kpc, cl, sl, sb)
    r = np.hypot(x, y)
    centre = np.where(
        comp == 0,
        DISK_ROTATION_VELOCITY,
        np.where(
            comp == 1,
            THICK_DISK_ROTATION_VELOCITY,
            BULGE_ROTATION_ANGULAR_VELOCITY * r,
        ),
    )
    sig = _SIG[comp]  # (N, 3)
    v_r = rng.normal(0.0, sig[:, 0])
    v_phi = centre + rng.normal(0.0, sig[:, 1])
    v_z = rng.normal(0.0, sig[:, 2])
    cos_phi, sin_phi = x / r, y / r
    v_x = v_r * cos_phi - v_phi * sin_phi
    v_y = v_r * sin_phi + v_phi * cos_phi
    v_gal = np.stack([v_x, v_y, v_z], axis=-1) - v0  # (N, 3)
    v_icrs = np.linalg.solve(m_rot, v_gal.T).T
    scale = K_VEL_CONVERSION * dist_kpc
    return v_icrs[:, 0] / scale, v_icrs[:, 1] / scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=400)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument(
        "--rate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="weight draws by the event rate's mu_rel factor (see module "
        "docstring); --no-rate reproduces the unselected comparison",
    )
    args = ap.parse_args()
    d = Path(C.data_dir_or_raise(None))
    cols = np.genfromtxt(
        d / "Answers" / "wfirstColumnNumbers.txt",
        dtype=None,
        encoding="utf-8",
        usecols=[0, 1],
        skip_header=2,
        names=["i", "name"],
    )
    names = [
        f"col{i}" if nm == "|" else nm for i, nm in enumerate(cols["name"])
    ]
    df = pd.read_csv(
        d / "Answers" / "master_file.txt",
        names=names,
        usecols=range(len(names)),
        sep=r"\s+",
        skiprows=1,
    )
    rng = np.random.default_rng(args.seed)
    mus, comps = [], []
    for _, e in df.iterrows():
        basis = line_of_sight_basis(np.radians(e.ra), np.radians(e.dec))
        out = []
        for dist in (e.Ds, e.Dl):
            dk = np.full(args.draws, float(dist))
            x, y, z = galactic_xyz(dk, *basis[2:])
            w = branch_log_density(x, y, z)
            w = np.exp(w - w.max(axis=-1, keepdims=True))
            w /= w.sum(axis=-1, keepdims=True)
            u = rng.random((args.draws, 1))
            comp = (u > np.cumsum(w, axis=-1)).sum(axis=-1).clip(0, 2)
            out.append(draw_pm(rng, dk, basis, comp))
            comps.append(comp)
        mu = np.hypot(out[1][0] - out[0][0], out[1][1] - out[0][1])
        mus.append(mu)
    mu = np.concatenate(mus)
    comps = np.concatenate(comps)
    # One event, one unit of weight: normalize within the event, so a
    # sight line does not count more just because it runs fast.
    if args.rate:
        wt = np.concatenate([m / m.sum() for m in mus])
    else:
        wt = np.full(mu.size, 1.0 / mu.size)
    wt = wt / wt.sum()
    # Each drawn STAR carries its pair's weight; comps holds source and
    # lens separately, so each weight is used twice.
    wt_star = np.repeat(wt, 2) if comps.size == 2 * mu.size else None
    sim = df.murel.values
    print("OUR GALACTIC MODEL'S IMPLIED mu_rel, drawing each star's component")
    print("from our OWN density mixture at the simulation's own geometry.\n")
    frac = [np.mean(comps == k) for k in range(3)]
    print(
        f"component mix over all drawn stars: thin {frac[0]:.1%}, "
        f"thick {frac[1]:.1%}, bulge {frac[2]:.1%}"
    )
    print(
        "event-rate weighting: "
        + (
            "ON (comparable to the key)"
            if args.rate
            else "OFF (not comparable)"
        )
        + "\n"
    )

    def wq(a, w, ps):
        i = np.argsort(a)
        c = np.cumsum(w[i])
        return [float(np.interp(p, c / c[-1], a[i])) for p in ps]

    ps = [0.5, 0.68, 0.90, 0.95, 0.99]
    print(f"{'':<14}{'median':>9}{'68th':>8}{'90th':>8}{'95th':>8}{'99th':>8}")
    print(
        f"{'ours (mixture)':<14}"
        + "".join(f"{x:8.2f}" for x in wq(mu, wt, ps))
    )
    print(
        f"{'simulation':<14}"
        + "".join(
            f"{x:8.2f}" for x in np.percentile(sim, [p * 100 for p in ps])
        )
    )
    for thr, why in ((11.49, "event 194's truth"), (17.38, "the sim's 95th")):
        print(f"\nfraction above mu_rel = {thr} ({why}):")
        print(f"  ours       {wt[mu > thr].sum() * 100:6.2f}%")
        print(f"  simulation {np.mean(sim > thr) * 100:6.2f}%")
    if wt_star is not None:
        print(
            f"\nbulge fraction of drawn stars, rate-weighted: "
            f"{wt_star[comps == 2].sum() / wt_star.sum():.1%}"
        )


if __name__ == "__main__":
    main()
