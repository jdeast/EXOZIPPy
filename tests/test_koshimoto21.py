"""The genulens port (galacticmodel/koshimoto21.py) against genulens itself.

tests/fixtures/genulens_points.json holds genulens's OWN functions evaluated
at fixed points (scripts/make_genulens_points.py, which compiles
scripts/genulens_reference_driver.cpp against a genulens build).  Pieces the
port reproduces analytically must agree to the fixture's 12 significant
digits; the tolerance of every other piece is stated where it is tested.
"""

import json
import os

import numpy as np
import pytest

from exozippy.components.galacticmodel import koshimoto21 as k21

FIXTURE = os.path.join(
    os.path.dirname(__file__), "fixtures", "genulens_points.json"
)
RTOL = 1e-11  # the fixture keeps 12 significant digits


@pytest.fixture(scope="module")
def points():
    with open(FIXTURE) as fh:
        return json.load(fh)


def _sight_lines(points):
    return [(name, s) for name, s in points["sight_lines"].items()]


def test_sgra_offset_matches_genulens(points):
    for _, s in _sight_lines(points):
        np.testing.assert_allclose(
            k21.SGRA_OFFSET, s["constants"]["xyzSgrA"], rtol=RTOL, atol=1e-12
        )


def test_positions_match_dlb2xyz(points):
    for name, s in _sight_lines(points):
        l_deg, b_deg = s["sightline"]["l"], s["sightline"]["b"]
        for row in s["density"]:
            xyz = k21.dlb_to_xyz(row["D"], l_deg, b_deg)
            np.testing.assert_allclose(
                xyz, row["xyz"], rtol=RTOL, atol=1e-9, err_msg=name
            )


def test_component_shapes_match_calc_rho_each(points):
    """Every analytic density shape (thin-disk age bins, thick disk, the E+E_X
    bar, the stellar halo) equals genulens's calc_rho_each at genulens's own
    positions."""
    n_checked = 0
    for name, s in _sight_lines(points):
        for row in s["density"]:
            x, y, z = row["xyz"]
            shapes = k21.log_component_shapes(x, y, z)
            for i, log_rho in enumerate(shapes):
                if log_rho is None:  # the NSD is tabulated
                    continue
                ref = row["rhos"][i]
                assert ref > 0.0, (name, row["D"], i)
                # compared in logs: far from a component (rhos ~ 1e-30) the
                # 12-digit rounding of the fixture's positions moves
                # log rho by ~1e-12 * |log rho|, i.e. rho by 4e-11 relative
                np.testing.assert_allclose(
                    log_rho,
                    np.log(ref),
                    rtol=RTOL,
                    atol=RTOL,
                    err_msg=f"{name} D={row['D']} {k21.COMPONENTS[i]}",
                )
                n_checked += 1
    assert n_checked > 500


def test_number_densities_match_genulens(points):
    """n0[i] rhos[i] (all objects: genulens's lens density) and n0MS[i]
    rhos[i] (main sequence: its source density), every component but the
    NSD, against the driver's ``n`` and ``nMS``."""
    for name, s in _sight_lines(points):
        c = s["constants"]
        np.testing.assert_allclose(k21.N0[:8], c["n0d"], rtol=RTOL)
        np.testing.assert_allclose(k21.N0_MS[:8], c["n0MSd"], rtol=RTOL)
        np.testing.assert_allclose(
            [k21.N0[8], k21.N0[10]], [c["n0b"], c["n0SH"]], rtol=RTOL
        )
        np.testing.assert_allclose(
            [k21.N0_MS[8], k21.N0_MS[10]], [c["n0MSb"], c["n0MSSH"]], rtol=RTOL
        )
        for row in s["density"]:
            x, y, z = row["xyz"]
            for lum, key in ((False, "n"), (True, "nMS")):
                logs = k21.log_number_densities(x, y, z, luminous_only=lum)
                for i, ln in enumerate(logs):
                    if ln is None:
                        continue
                    np.testing.assert_allclose(
                        ln,
                        np.log(row[key][i]),
                        rtol=RTOL,
                        atol=RTOL,
                        err_msg=f"{name} D={row['D']} {key}[{i}]",
                    )


def test_rotation_curve_matches_genulens(points):
    c = points["sight_lines"]["baade"]["constants"]
    np.testing.assert_allclose(k21.ROTCURVE_R, c["Rcs"], rtol=RTOL)
    np.testing.assert_allclose(k21.ROTCURVE_V, c["Vcs"], rtol=RTOL)
    # the exact inverse of L = v_c R
    rg = np.linspace(1.0, 25000.0, 4001)
    back, _ = k21._guiding_radius(k21.rotation_curve(rg) * rg, np)
    np.testing.assert_allclose(back, rg, rtol=1e-12)


def test_shu_distribution_matches_calc_prrg(points):
    """The unnormalized Shu DF equals genulens's calc_PRRg on its f_g grid
    (analytic: fixture precision); where genulens returns 0 the port is
    below 1e-6 of the peak; the normalization agrees with genulens's
    trapezoid over its truncated support to 1%."""
    n_checked = 0
    for name, s in _sight_lines(points):
        for n in s.get("shu", []):
            fg = np.array(n["fg_grid"])
            ref = np.array(n["P_unnorm_grid"])
            lp = k21.log_shu_unnormalized(
                fg, n["R_node"], n["z_node"], n["comp"]
            )
            ok = ref > 1e-250
            np.testing.assert_allclose(
                lp[ok],
                np.log(ref[ok]),
                rtol=1e-10,
                atol=1e-10,
                err_msg=f"{name} comp {n['comp']}",
            )
            assert np.all(np.exp(lp[~ok]) < 1e-6 * ref.max())
            lz = k21.shu_log_norm(n["R_node"], n["z_node"])[n["comp"]]
            assert abs(np.exp(lz) / n["norm"] - 1.0) < 0.01, (name, n)
            n_checked += 1
    assert n_checked >= 50


def test_disk_and_bar_kinematic_blocks_match_genulens(points):
    """Analytic velocity blocks: disk dispersions (age-dependent, radial
    gradients) and the bar's mean and dispersions, to fixture precision."""
    for name, s in _sight_lines(points):
        if "velocity" not in s:
            continue
        pos = {row["D"]: row["xyz"] for row in s["density"]}
        for v in s["velocity"]:
            x, y, z = pos[v["D"]]
            if v["comp"] < 8:
                su, sw = k21.disk_dispersions(v["R"], v["comp"])
                np.testing.assert_allclose(
                    [su, sw], [v["sigU"], v["sigW"]], rtol=RTOL
                )
            elif v["comp"] == 8:
                (mx, my, _), sig = k21.bar_mean_and_sigma(x, y, z)
                # the 1 pc^2 radius floor moves these by < 1e-8 relative
                np.testing.assert_allclose(
                    [mx, my], v["mean_xyz"][:2], rtol=1e-7, atol=1e-7
                )
                np.testing.assert_allclose(sig, v["sigma_xyz"], rtol=RTOL)


def test_vphi_density_against_genulens_sampler(points):
    """The disk v_phi density is normalized (to 0.5%: the bilinear log Z
    table, worst at R < 1 kpc) and its central quantiles (16,
    25, 50, 75, 84%) match genulens's own sampler (20000 draws of
    get_vxyz_ran) to 8% of the 16-84% width.  The remaining gap is the
    port's stated deviation: genulens draws f_g from the table node BELOW a
    star (100 pc in R, 200 pc in |z|), the port at the star's own (R, z);
    largest (5.5% measured) at R ~ 1 kpc.  The tails differ more because
    genulens truncates its table where P < 1% of the peak."""
    vg = np.linspace(-400.0, 700.0, 22001)
    for name, s in _sight_lines(points):
        if "velocity" not in s:
            continue
        pos = {row["D"]: row["xyz"] for row in s["density"]}
        for v in s["velocity"]:
            if v["comp"] >= 8:
                continue
            x, y, z = pos[v["D"]]
            p = np.exp(k21.log_vphi_density(vg, v["R"], z, v["comp"]))
            assert abs(np.trapezoid(p, vg) - 1.0) < 5e-3, (
                name,
                v["D"],
                v["comp"],
            )
            cdf = np.cumsum(p)
            cdf /= cdf[-1]
            probs = np.array(v["probs"])
            mc = np.array(v["mc_q_vphi"])
            q = np.interp(probs, cdf, vg)
            width = mc[7] - mc[3]
            central = slice(3, 8)
            err = np.max(np.abs(q[central] - mc[central])) / width
            assert err < 0.08, (name, v["D"], v["comp"], err)


@pytest.mark.slow  # compiles the gradient graph (~2 min on a cold cache)
def test_pytensor_graph_matches_numpy_with_finite_gradient(points):
    """The same expressions as a pytensor graph: equal to numpy and with a
    finite gradient in position and velocity at every fixture point (and on
    the bar's axis and at counter-rotation, where genulens has zeros)."""
    import pytensor
    import pytensor.tensor as pt

    s = points["sight_lines"]["baade"]
    xyz = np.array([row["xyz"] for row in s["density"]])
    rng = np.random.default_rng(1)
    vel = rng.normal(
        [0.0, 200.0, 0.0], [80.0, 120.0, 60.0], size=(len(xyz), 3)
    )
    # the bar's minor axis and a counter-rotating star
    xyz = np.vstack([xyz, [[0.0, 0.0, 50.0], [3000.0, 100.0, -200.0]]])
    vel = np.vstack([vel, [[10.0, 5.0, 3.0], [0.0, -150.0, 0.0]]])
    xs, vs = pt.dmatrix("xs"), pt.dmatrix("vs")
    dens = k21.log_number_densities(xs[:, 0], xs[:, 1], xs[:, 2], xp=pt)
    velo = k21.log_velocity_densities(
        xs[:, 0], xs[:, 1], xs[:, 2], vs[:, 0], vs[:, 1], vs[:, 2], xp=pt
    )
    terms = [d + v for d, v in zip(dens, velo) if d is not None]
    total = pt.sum(pt.logsumexp(pt.stack(terms), axis=0))
    f = pytensor.function(
        [xs, vs], [pt.stack(terms), *pt.grad(total, [xs, vs])]
    )
    got, gx, gv = f(xyz, vel)
    dn = k21.log_number_densities(xyz[:, 0], xyz[:, 1], xyz[:, 2])
    vn = k21.log_velocity_densities(*xyz.T, *vel.T)
    want = np.array([d + v for d, v in zip(dn, vn) if d is not None])
    np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-8)
    assert np.all(np.isfinite(gx)) and np.all(np.isfinite(gv))


# --- the mass function (port PR 3) -----------------------------------------


def test_imf_is_k21s_broken_power_law(points):
    """K21's IMF in log10 M equals genulens's grid density (store_IMF_nBs:
    1000 log bins on [0.001, 120]) up to its normalization."""
    m = points["mass"]
    x = m["logMst"] + m["dlogM"] * 10 * np.arange(len(m["PlogM_every10"]))
    diff = k21.log_imf(x) - np.log(m["PlogM_every10"])
    np.testing.assert_allclose(diff - diff[0], 0.0, atol=1e-10)


def test_turnoff_masses_are_genulens_rows(points):
    """Each branch's turnoff is the Minidie.dat row genulens reads
    (event_sampler.cpp): thin bins at their median age rounded to the
    nearest 50 Myr row, thick disk 12 Gyr, halo 14 Gyr; the bar's rows."""
    m = points["mass"]
    ages, die = m["agesD"], m["MinidieD"]
    for i in range(7):
        ia = int(k21.MEDTAUDS[i] * 100 + 0.5)
        ia = ia + (5 - ia % 5) if ia % 5 > 2 else ia - ia % 5
        idx = (max(ia, 5) - ages[0]) // (ages[1] - ages[0])
        assert k21.MINIDIE[i] == pytest.approx(die[idx], rel=RTOL)
    assert k21.MINIDIE[7] == pytest.approx(die[-2], rel=RTOL)
    assert k21.MINIDIE[10] == pytest.approx(die[-1], rel=RTOL)
    np.testing.assert_allclose(
        [a for a, _ in k21.MINIDIE_BAR], np.array(m["agesB"]) / 100.0
    )
    np.testing.assert_allclose(
        [d for _, d in k21.MINIDIE_BAR], m["MinidieB"], rtol=RTOL
    )


def _imf_integral(lo, hi):
    from scipy.integrate import quad

    return quad(lambda x: np.exp(k21.log_imf(x)), lo, hi, limit=200)[0]


@pytest.mark.parametrize("remnants", [True, False])
def test_mass_function_counts(remnants):
    """Per branch, over the default logmass support: with remnants every
    born star is one object (the table integrates to the IMF's integral);
    without, only the alive ones remain (F_alive, genulens's REMNANT 0),
    to the 0.02-dex smoothing's precision."""
    mf = k21.MassFunction([-9.0], [2.5], remnants=remnants)
    lo, hi = np.log10(k21.IMF_MIN), np.log10(k21.IMF_MAX)
    z = _imf_integral(lo, hi)
    for j, b in enumerate(k21.BRANCHES):
        total = sum(np.exp(mf.log_table[g][j]) for g in mf.GROUPS)
        sel = (mf.x >= -9.0) & (mf.x <= 2.5)
        count = np.trapezoid(total[sel], mf.x[sel]) / np.exp(mf.log_norm[j, 0])
        if remnants:
            assert count == pytest.approx(1.0, abs=2e-3), b
        elif b != 8:
            f_alive = _imf_integral(lo, np.log10(k21.MINIDIE[b])) / z
            assert count == pytest.approx(f_alive, abs=2e-3), b


def test_kicked_velocity_density_is_normalized():
    r = np.linspace(1e-3, 1500.0, 300001)
    for sig, kick in ((20.0, 350.0), (80.0, 100.0), (150.0, 350.0)):
        lp = k21.log_kicked_velocity(r, 0.0 * r, 0.0 * r, (0, 0, 0), sig, kick)
        assert np.trapezoid(4 * np.pi * r**2 * np.exp(lp), r) == pytest.approx(
            1.0, abs=1e-4
        )
