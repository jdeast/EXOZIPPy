"""Catalog astrometric solutions and the blended photocenter (review 8.10.3;
mulensing.md "The blended photocenter moves").

A catalog proper motion of a microlensing target (Gaia, VIRAC2) is the
flux-weighted motion of everything in the PSF -- the source's images, the
luminous lens moving at mu_S + mu_rel, a neighbor on its own track -- fit
by the catalog with a five-parameter line.  Pinned here:

1. The projection: a track that IS a five-parameter line comes back as its
   own parameters; a two-parameter solution is the weighted mean position;
   epochs that cannot determine the solution raise.
2. An UNLENSED star's catalog model is exactly its own ra/dec/pm/parallax
   (the frame of the solution vector is the abs-mode frame).
3. The simulation the item asks for: source + lens + neighbor tracks, with
   the microlensing centroid shift inside the Gaia window, are simulated
   in numpy from their own tracks; two catalog solutions (a G-like and a
   Ks-like band, with different lens shares) are formed by an independent
   least-squares fit; the model reproduces both to 1e-6 mas; a maximum-likelihood fit
   from a displaced start recovers mu_rel and the lens flux in both bands;
   and the naive reading -- the catalog pm as a prior on the SOURCE's pm --
   is biased by many sigma.
4. The C and JAX gradients of the catalog likelihood are finite and agree.
5. Configuration errors raise, naming the fix.
"""

import numpy as np
import pymc as pm
import pytensor
import pytest

from exozippy.components.astrometryinstrument.astrometryinstrument import (
    CATALOG_ROLES,
    catalog_design_matrix,
    catalog_projection,
)
from exozippy.constants import DAYS_PER_YEAR, KAPPA
from exozippy.ephemeris import observer_position_at_reception
from exozippy.skyframe import parallax_factors
from exozippy.system import System

RAD2MAS = 180.0 / np.pi * 3600.0 * 1000.0
RA_DEG, DEC_DEG = 266.4168, -29.0078
RA, DEC = np.radians(RA_DEG), np.radians(DEC_DEG)

# The event peaks inside Gaia DR3's window, so the centroid shift is in the
# Gaia track; the reference epoch is the Gaia catalog epoch (J2016.0).
T0 = 2457300.0
U0 = 0.23
EPOCH = 2457389.0
GAIA_T = np.linspace(2456870.0, 2457890.0, 70)  # 2014.6 - 2017.4
VIRAC_T = np.linspace(2455200.0, 2458500.0, 90)  # 2010 - 2019

MU_S = (-1.0, -2.0)
MU_L = (3.0, 4.0)  # mu_rel = (4, 6) mas/yr, not along an axis
MU_N = (1.0, -1.0)
D_L, D_S, D_N = 4000.0, 8000.0, 6000.0
NEIGHBOR_OFFSET = (150.0, -80.0)  # mas from the source at EPOCH
# flux relative to the source, per band: the M-dwarf lens gains in Ks
R_L = {"Gaia": 0.3, "VIRAC": 1.2}
R_N = {"Gaia": 0.5, "VIRAC": 0.6}
# the sightline's unresolved background: flux per band, photocenter offset
# from the source at EPOCH (mas) and its own proper motion (mas/yr)
R_BG = {"Gaia": 0.2, "VIRAC": 0.35}
BG_BAND = {"Gaia": "G", "VIRAC": "Ks"}
BG_OFFSET = (30.0, -20.0)
MU_BG = (1.5, 1.0)
# catalog errors: (pos, plx, pm) in mas, mas, mas/yr
ERRS = {"Gaia": (0.3, 0.4, 0.25), "VIRAC": (1.0, 1.0, 0.4)}


# ----------------------------------------------------------------------
# 1. The projection
# ----------------------------------------------------------------------
def _pfac(t):
    xyz = observer_position_at_reception(t, RA, DEC, observer_location="earth")
    return parallax_factors(xyz, RA, DEC)


def test_projection_returns_a_linear_track_exactly():
    """
    Given a track that is exactly a five-parameter line (x0, plx, pm) at
      Gaia-like epochs,
    When it is projected with catalog_projection,
    Then the five parameters come back to 1e-9 (unit and random weights
      alike); a two-parameter projection returns the weighted mean
      position of a static star; and three epochs at one instant (no
      proper-motion lever arm) raise.
    """
    t = GAIA_T
    P_E, P_N = _pfac(t)
    truth = np.array([1.3, -2.1, 0.7, 4.2, -3.3])
    M = catalog_design_matrix(t, EPOCH, P_E, P_N, 5)
    track = M @ truth
    for w in (None, np.random.default_rng(1).uniform(0.5, 2.0, len(t))):
        proj = catalog_projection(t, EPOCH, P_E, P_N, 5, weights=w)
        np.testing.assert_allclose(proj @ track, truth, rtol=0, atol=1e-9)

    w = np.random.default_rng(2).uniform(0.5, 2.0, len(t))
    rng = np.random.default_rng(3)
    dE, dN = rng.normal(0, 1, len(t)), rng.normal(0, 1, len(t))
    p2 = catalog_projection(t, EPOCH, P_E, P_N, 2, weights=w)
    np.testing.assert_allclose(
        p2 @ np.concatenate([dE, dN]),
        [np.sum(w * dE) / np.sum(w), np.sum(w * dN) / np.sum(w)],
        rtol=1e-12,
    )
    same = np.full(3, EPOCH)
    pe, pn = _pfac(same)
    with pytest.raises(ValueError, match="cannot determine"):
        catalog_projection(same, EPOCH, pe, pn, 5)


# ----------------------------------------------------------------------
# File writers
# ----------------------------------------------------------------------
def _write_catalog(path, sol, errs, corr=None, n_par=5):
    """One catalog row in the Gaia archive's order from a solution vector
    (dE0, dN0, plx, pmE, pmN) in the tangent-plane frame of (RA, DEC)."""
    ra = RA_DEG + np.degrees(sol[0] / RAD2MAS / np.cos(DEC))
    dec = DEC_DEG + np.degrees(sol[1] / RAD2MAS)
    pos, plx_e, pm_e = errs
    if n_par == 2:
        row = [EPOCH, ra, dec, pos, pos, 0.0 if corr is None else corr]
    else:
        c = np.zeros(10) if corr is None else np.asarray(corr)
        row = [EPOCH, ra, dec, sol[2], sol[3], sol[4]]
        row += [pos, pos, plx_e, pm_e, pm_e] + list(c)
    assert len(row) == len(CATALOG_ROLES[n_par])
    np.savetxt(path, np.atleast_2d(row), fmt="%.12f")


def _cov(errs):
    pos, plx_e, pm_e = errs
    return np.diag([pos, pos, plx_e, pm_e, pm_e]) ** 2


# ----------------------------------------------------------------------
# 2. An unlensed star
# ----------------------------------------------------------------------
def test_unlensed_catalog_model_is_the_star_parameters(tmp_path):
    """
    Given a single star (no orbit, no event) with a five- and a
      two-parameter catalog dataset whose solutions are the star's own
      parameters, epochs from a span (so the uniform-epochs warning path
      runs) and from a list,
    When the system is built,
    Then the five-parameter solution equals the star's (position offset at
      the epoch, parallax, pm) to 1e-6 and the two-parameter one its mean
      position over the epochs, the catalog log-likelihood is the
      zero-residual multivariate normal, and the jitter is INACTIVE on
      the catalog datasets.
    """
    sol = np.array([0.0, 0.0, 2.5, 7.0, -11.0])
    _write_catalog(tmp_path / "c5.dat", sol, (0.2, 0.3, 0.25))
    _write_catalog(tmp_path / "c2.dat", sol[:2], (0.5, 0, 0), n_par=2)
    cfg = {
        "name": "catstar",
        "star": [{"name": "A"}],
        "astrometryinstrument": [
            {
                "name": "G5",
                "file": str(tmp_path / "c5.dat"),
                "mode": "catalog",
                "epoch": EPOCH,
                "catalog_span": [float(GAIA_T[0]), float(GAIA_T[-1])],
                "catalog_n_epochs": 40,
            },
            {
                "name": "G2",
                "file": str(tmp_path / "c2.dat"),
                "mode": "catalog",
                "catalog_parameters": 2,
                "epoch": EPOCH,
                "catalog_epochs": [float(x) for x in GAIA_T[::5]],
            },
        ],
    }
    params = {
        "star.A.ra": {"initval": RA_DEG},
        "star.A.dec": {"initval": DEC_DEG},
        "star.A.distance": {"initval": 400.0},
        "star.A.pm_ra": {"initval": 7.0},
        "star.A.pm_dec": {"initval": -11.0},
    }
    system = System(cfg, user_params=params)
    system.prepare()
    model = system.build_model()
    inst = system.astrometryinstrument
    assert not inst.jitter_variance.element_is_active(0)
    with model:
        nodes = []
        for i in (0, 1):
            d = inst.datasets[i]
            dE, dN = inst._absolute_model(system, d, d["time"], 0.0)
            nodes.append(inst._catalog_solution(d, dE, dN))
    fn = pytensor.function(model.free_RVs, nodes, on_unused_input="ignore")
    ip = model.initial_point()
    s5, s2 = fn(*[ip[v.name] for v in model.free_RVs])
    np.testing.assert_allclose(s5, sol, rtol=0, atol=1e-6)
    # two parameters: the MEAN position of the moving star over the epochs
    t2 = inst.datasets[1]["time"]
    pe, pn = _pfac(t2)
    dt = (t2 - EPOCH) / DAYS_PER_YEAR
    mean_track = [
        np.mean(sol[3] * dt + sol[2] * pe),
        np.mean(sol[4] * dt + sol[2] * pn),
    ]
    np.testing.assert_allclose(s2, mean_track, rtol=0, atol=1e-6)
    obs = [v for v in model.observed_RVs if v.name.endswith("model_G5")]
    lp = float(model.compile_logp(vars=obs, sum=True)(ip))
    expect = -0.5 * np.linalg.slogdet(2 * np.pi * inst.datasets[0]["cov"])[1]
    assert lp == pytest.approx(expect, abs=1e-6)


# ----------------------------------------------------------------------
# 3. Source + lens + neighbor
# ----------------------------------------------------------------------
def _theta_E():
    return float(np.sqrt(KAPPA * 0.5 * (1000.0 / D_L - 1000.0 / D_S)))


def _config(tmp, extra_gaia=None):
    def cat(name, epochs):
        return {
            "name": name,
            "file": str(tmp / f"{name}.dat"),
            "mode": "catalog",
            "observer_location": "earth",
            "epoch": EPOCH,
            "star_ndx": 1,
            "catalog_epochs": [float(x) for x in epochs],
            "psf_stars": ["Lens", "Neighbor"],
            "fitpsfflux": True,
            "background": "Bg",
            "band": BG_BAND[name],
        }

    gaia = {**cat("Gaia", GAIA_T), **(extra_gaia or {})}
    return {
        "name": "catmulens",
        "star": [{"name": "Lens"}, {"name": "Source"}, {"name": "Neighbor"}],
        "mulensevent": [{}],
        "lens": [{"body": "star.Lens", "name": "Lens"}],
        "source": [{"body": "star.Source", "name": "Source"}],
        "mulensinstrument": [{"name": "OGLE", "file": str(tmp / "lc.dat")}],
        "astrometryinstrument": [gaia, cat("VIRAC", VIRAC_T)],
        "background": [{"name": "Bg", "bands": ["G", "Ks"]}],
    }


def _params(mu_s=MU_S, mu_l=MU_L, r_l=None):
    r_l = r_l or R_L
    p = {
        "source.Source.t_0": {"initval": T0, "sigma": 0},
        "source.Source.u_0": {"initval": U0, "sigma": 0},
        "star.Lens.distance": {"initval": D_L, "sigma": 0},
        "star.Source.distance": {"initval": D_S, "sigma": 0},
        "star.Neighbor.distance": {"initval": D_N, "sigma": 0},
        "star.Neighbor.logmass": {"initval": 0.0, "sigma": 0},
        # the light curve is noiseless; hold its error scale
        "mulensinstrument.OGLE.err_scale": {"initval": 1.0, "sigma": 0},
        "star.Lens.logmass": {"initval": float(np.log10(0.5)), "sigma": 0},
        "star.Lens.pm_ra": {"initval": mu_l[0]},
        "star.Lens.pm_dec": {"initval": mu_l[1]},
        "star.Source.pm_ra": {"initval": mu_s[0]},
        "star.Source.pm_dec": {"initval": mu_s[1]},
        "star.Source.ra": {"initval": RA_DEG},
        "star.Source.dec": {"initval": DEC_DEG},
        # the modeled neighbor: position and motion pinned (review 8.9.7)
        "star.Neighbor.ra": {
            "initval": RA_DEG
            + np.degrees(NEIGHBOR_OFFSET[0] / RAD2MAS / np.cos(DEC)),
            "sigma": 0,
        },
        "star.Neighbor.dec": {
            "initval": DEC_DEG + np.degrees(NEIGHBOR_OFFSET[1] / RAD2MAS),
            "sigma": 0,
        },
        "star.Neighbor.pm_ra": {"initval": MU_N[0], "sigma": 0},
        "star.Neighbor.pm_dec": {"initval": MU_N[1], "sigma": 0},
    }
    for name in ("Gaia", "VIRAC"):
        pre = f"astrometryinstrument.{name}"
        p[f"{pre}.psf_fluxratio_0"] = {"initval": r_l[name]}
        # the neighbor's flux is known (resolved imaging): pinned
        p[f"{pre}.psf_fluxratio_1"] = {"initval": R_N[name], "sigma": 0}
    # the unresolved background, known here (pinned at the truth), so the
    # fit's unknowns are the two pms, the source position and the lens flux
    for k, name in enumerate(("Gaia", "VIRAC")):
        p[f"background.Bg.fluxratio_{k}"] = {"initval": R_BG[name], "sigma": 0}
    p["background.Bg.offset_e"] = {"initval": BG_OFFSET[0], "sigma": 0}
    p["background.Bg.offset_n"] = {"initval": BG_OFFSET[1], "sigma": 0}
    p["background.Bg.pm_ra"] = {"initval": MU_BG[0], "sigma": 0}
    p["background.Bg.pm_dec"] = {"initval": MU_BG[1], "sigma": 0}
    return p


def _build(cfg, params):
    system = System(cfg, user_params=params)
    system.prepare()
    return system, system.build_model()


def _start_values(model, nodes):
    fn = pytensor.function(model.free_RVs, nodes, on_unused_input="ignore")
    ip = model.initial_point()
    return fn(*[ip[v.name] for v in model.free_RVs])


def _simulate(lens_offset, t, band):
    """The blended photocenter track (dE, dN) mas at ``t``, from the
    three stars' own five-parameter tracks, the background's linear track
    (no parallax) and the C30 centroid shift of the source's images
    computed from the lens-source separation -- no model code but the
    ephemeris."""
    P_E, P_N = _pfac(t)
    dt = (t - EPOCH) / DAYS_PER_YEAR
    plx_s, plx_l, plx_n = 1000.0 / D_S, 1000.0 / D_L, 1000.0 / D_N
    xs = np.array([MU_S[0] * dt + plx_s * P_E, MU_S[1] * dt + plx_s * P_N])
    xl = np.array(
        [
            lens_offset[0] + MU_L[0] * dt + plx_l * P_E,
            lens_offset[1] + MU_L[1] * dt + plx_l * P_N,
        ]
    )
    xn = np.array(
        [
            NEIGHBOR_OFFSET[0] + MU_N[0] * dt + plx_n * P_E,
            NEIGHBOR_OFFSET[1] + MU_N[1] * dt + plx_n * P_N,
        ]
    )
    theta_e = _theta_E()
    dth = (xl - xs) / theta_e
    u2 = np.sum(dth**2, axis=0)
    shift = -theta_e * dth / (u2 + 2.0)
    A = (u2 + 2.0) / np.sqrt(u2 * (u2 + 4.0))
    xb = np.array([BG_OFFSET[0] + MU_BG[0] * dt, BG_OFFSET[1] + MU_BG[1] * dt])
    r_l, r_n, r_b = R_L[band], R_N[band], R_BG[band]
    xc = (A * (xs + shift) + r_l * xl + r_n * xn + r_b * xb) / (
        A + r_l + r_n + r_b
    )
    return xc, np.sqrt(np.min(u2))


def _lstsq_solution(t, track):
    """An independent catalog fit: lstsq of the five-parameter line."""
    P_E, P_N = _pfac(t)
    dt = (t - EPOCH) / DAYS_PER_YEAR
    z, o = np.zeros_like(t), np.ones_like(t)
    M = np.vstack(
        [
            np.column_stack([o, z, P_E, dt, z]),
            np.column_stack([z, o, P_N, z, dt]),
        ]
    )
    return np.linalg.lstsq(M, np.concatenate(track), rcond=None)[0]


@pytest.fixture(scope="module")
def truth_files(tmp_path_factory):
    """The light curve generated by the model's own magnification at the
    truth (so the truth is its exact optimum), and the two catalogs formed
    from the numpy-simulated photocenter tracks."""
    tmp = tmp_path_factory.mktemp("catmulens")
    t_lc = np.linspace(T0 - 120.0, T0 + 120.0, 300)
    np.savetxt(
        tmp / "lc.dat",
        np.column_stack(
            [t_lc, np.full_like(t_lc, 18.0), np.full_like(t_lc, 0.01)]
        ),
    )
    zero = np.zeros(5)
    for name in ("Gaia", "VIRAC"):
        _write_catalog(tmp / f"{name}.dat", zero, ERRS[name])
    system, model = _build(_config(tmp), _params())
    ev, mi = system.mulensevent, system.mulensinstrument
    xyz = mi.get_observer_position(t_lc)
    with model:
        A = ev.get_magnification(
            t_lc, ev.skowron_deviations(t_lc, xyz), system, 0
        )
        off = [ev.lens_offset_e.value[0], ev.lens_offset_n.value[0]]
    A, off_e, off_n = _start_values(model, [A, *off])
    f_s, f_b = 1.0e-8, 0.4e-8
    mag = -2.5 * np.log10(f_s * A + f_b)
    np.savetxt(
        tmp / "lc.dat", np.column_stack([t_lc, mag, np.full_like(t_lc, 0.2)])
    )

    sols, u_min = {}, {}
    for name, t in (("Gaia", GAIA_T), ("VIRAC", VIRAC_T)):
        track, u_min[name] = _simulate((float(off_e), float(off_n)), t, name)
        sols[name] = _lstsq_solution(t, track)
        _write_catalog(tmp / f"{name}.dat", sols[name], ERRS[name])
    return tmp, sols, (float(off_e), float(off_n)), u_min


@pytest.fixture(scope="module")
def truth_system(truth_files):
    tmp = truth_files[0]
    return _build(_config(tmp), _params())


@pytest.mark.slow
def test_model_reproduces_the_simulated_catalog_solutions(
    truth_files, truth_system
):
    """
    Given the two catalogs formed from the simulated source + lens +
      neighbor photocenter tracks (the event inside the Gaia window: its
      closest approach is among the Gaia epochs' span),
    When the model's catalog solutions are evaluated at the truth,
    Then each equals the independent lstsq solution to 1e-6 (mas, mas/yr),
      the derived lens position is at the simulated lens offset, and the
      blended photocenter's pm differs from the source's by many catalog
      sigma -- in both bands, by different amounts.
    """
    tmp, sols, off, u_min = truth_files
    system, model = truth_system
    inst = system.astrometryinstrument
    assert inst.psf_stars == [[0, 2], [0, 2]]
    assert system.mulensevent.lens_track_star(system) == 0
    assert u_min["Gaia"] < 1.0  # the event is sampled by the Gaia epochs
    with model:
        nodes = []
        for i in (0, 1):
            d = inst.datasets[i]
            dev = system.mulensevent.skowron_deviations(d["time"], d["xyz"])
            dE, dN = inst._absolute_model(system, d, d["time"], 0.0, dev=dev)
            nodes.append(inst._catalog_solution(d, dE, dN))
    s_gaia, s_virac = _start_values(model, nodes)
    np.testing.assert_allclose(s_gaia, sols["Gaia"], rtol=0, atol=1e-6)
    np.testing.assert_allclose(s_virac, sols["VIRAC"], rtol=0, atol=1e-6)
    for name, s in (("Gaia", s_gaia), ("VIRAC", s_virac)):
        pm_sigma = ERRS[name][2]
        bias = np.hypot(s[3] - MU_S[0], s[4] - MU_S[1])
        assert bias > 5.0 * pm_sigma, (name, bias)
    assert np.hypot(*(sols["VIRAC"][3:] - sols["Gaia"][3:])) > 1.0


def _naive_source_pm_bias(sol, errs):
    """The catalog pm read as a measurement of the SOURCE's pm: the bias,
    in sigma, of the MAP source pm it implies."""
    pm_sigma = errs[2]
    return np.hypot(sol[3] - MU_S[0], sol[4] - MU_S[1]) / pm_sigma


def _max_likelihood(model):
    """The maximum of the summed observed log-likelihood over the model's
    free (raw) variables, by L-BFGS-B with an analytic gradient."""
    import pytensor.tensor as pt
    from scipy.optimize import minimize

    obs = model.observed_RVs
    logl = model.logp(vars=obs, sum=True)
    free = list(model.free_RVs)
    values = [model.rvs_to_values[v] for v in free]
    grads = pt.grad(logl, values)
    f = pytensor.function(values, [logl] + grads, on_unused_input="ignore")
    ip = model.initial_point()
    names = [val.name for val in values]
    shapes = [np.shape(ip[n]) for n in names]
    sizes = [int(np.prod(s)) for s in shapes]

    def unpack(x):
        out, k = [], 0
        for s, n in zip(shapes, sizes):
            out.append(x[k : k + n].reshape(s))
            k += n
        return out

    def neg(x):
        lg, *g = f(*unpack(x))
        return -float(lg), -np.concatenate([np.ravel(gi) for gi in g])

    x0 = np.concatenate([np.ravel(ip[n]) for n in names])
    res = minimize(
        neg,
        x0,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 20000, "ftol": 1e-15, "gtol": 1e-10},
    )
    return {v.name: a for v, a in zip(free, unpack(res.x))}


@pytest.mark.slow
def test_max_likelihood_recovers_mu_rel_and_the_lens_flux(truth_files):
    """
    Given the two catalogs, the light curve and every geometry parameter
      pinned except the source position, both proper motions and the
      lens's flux ratio in each band,
    When a maximum-likelihood fit starts ~1 mas/yr and 0.15 in flux ratio away (inside the
      basin: from 2-3 mas/yr away pm.find_MAP stopped in another optimum of
      the mu_rel direction, 22 nats worse in the catalogs),
    Then it recovers mu_rel = mu_L - mu_S to 0.01 mas/yr, the source pm to
      0.01 mas/yr and the lens flux ratio in both bands to 0.005, while
      the naive source-pm reading of the same catalogs is biased by > 5
      sigma (Gaia) and > 8 sigma (VIRAC).
    """
    tmp, sols, _off, _u = truth_files
    start = _params(
        mu_s=(MU_S[0] + 1.0, MU_S[1] - 0.7),
        mu_l=(MU_L[0] - 0.8, MU_L[1] + 0.9),
        r_l={"Gaia": 0.45, "VIRAC": 1.05},
    )
    system, model = _build(_config(tmp), start)
    star, inst = system.star, system.astrometryinstrument
    # Maximize the LIKELIHOOD (the observed terms), not the posterior: with
    # noiseless data its maximum is the truth exactly, while the priors'
    # logit-uniform and soft-barrier terms would move a MAP by their own
    # gradients.
    mp = _max_likelihood(model)
    fn = pytensor.function(
        model.free_RVs,
        [star.pm_ra.value, star.pm_dec.value, inst.psf_fluxratio.value],
        on_unused_input="ignore",
    )
    pmra, pmdec, ratios = fn(*[mp[v.name] for v in model.free_RVs])
    mu_rel = (pmra[0] - pmra[1], pmdec[0] - pmdec[1])
    np.testing.assert_allclose(
        mu_rel, (MU_L[0] - MU_S[0], MU_L[1] - MU_S[1]), atol=0.01
    )
    np.testing.assert_allclose((pmra[1], pmdec[1]), MU_S, atol=0.01)
    # psf_fluxratio elements: Gaia lens, Gaia neighbor, VIRAC lens, ...
    np.testing.assert_allclose(
        [ratios[0], ratios[2]], [R_L["Gaia"], R_L["VIRAC"]], atol=5e-3
    )
    assert _naive_source_pm_bias(sols["Gaia"], ERRS["Gaia"]) > 5.0
    assert _naive_source_pm_bias(sols["VIRAC"], ERRS["VIRAC"]) > 8.0


# ----------------------------------------------------------------------
# 4. Gradients, C and JAX
# ----------------------------------------------------------------------
@pytest.mark.slow
def test_catalog_gradients_are_finite_in_c_and_jax(truth_system):
    """
    Given the lensed, split catalog model,
    When the logp and its gradient are evaluated at the start in the C
      backend and through JAX (the numpyro/blackjax path),
    Then all are finite and the two backends agree.
    """
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    from pymc.sampling.jax import get_jaxified_logp

    _system, model = truth_system
    ip = model.initial_point()
    lp_c = float(model.compile_logp()(ip))
    g_c = model.compile_dlogp()(ip)
    assert np.isfinite(lp_c) and np.all(np.isfinite(g_c))
    logp_fn = get_jaxified_logp(model)
    vals = [ip[v.name] for v in model.value_vars]
    lp_j = float(logp_fn(vals))
    grads = jax.grad(
        lambda *v: logp_fn(list(v)), argnums=tuple(range(len(vals)))
    )(*vals)
    assert lp_j == pytest.approx(lp_c, rel=1e-9, abs=1e-6)
    bad = [
        v.name
        for v, g in zip(model.value_vars, grads)
        if not np.all(np.isfinite(np.asarray(g)))
    ]
    assert not bad, f"non-finite JAX gradients: {bad}"


# ----------------------------------------------------------------------
# 5. Configuration errors
# ----------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.parametrize(
    "extra, match",
    [
        ({"fitpsfflux": False}, "fitpsfflux: true"),
        ({"catalog_span": [2456870.0, 2457890.0]}, "exactly one of"),
        ({"psf_stars": ["Source"]}, "microlensing SOURCE"),
        ({"band": "J"}, "does not\\s+list"),
        ({"background": "Nope"}, "unknown background"),
        (
            {"background": None, "psf_stars": [], "fitpsfflux": False},
            "fixed on the sky",
        ),
    ],
)
def test_misconfigured_split_raises(truth_files, extra, match):
    tmp = truth_files[0]
    with pytest.raises(ValueError, match=match):
        _build(_config(tmp, extra_gaia=extra), _params())


@pytest.mark.slow
def test_background_is_one_object_with_a_field_pm_prior(truth_files):
    """
    Given the two catalogs reading ONE background instance in two bands,
      and no params entry on its proper motion,
    When the system is built,
    Then the background owns exactly one fluxratio element per listed band
      (shared by every dataset that reads that band), one photocenter
      offset and one proper motion, and on this bulge sightline the pm
      carries the field prior -- the galactic model's bulge mean at the
      Galactic center's distance and a ~2.6 mas/yr dispersion.
    """
    from exozippy.components.background.background import (
        bulge_field_pm_prior,
    )

    tmp = truth_files[0]
    params = _params()
    del params["background.Bg.pm_ra"], params["background.Bg.pm_dec"]
    system, _model = _build(_config(tmp), params)
    bg = system.background
    assert bg.names == ["Bg"] and bg.bands == [["G", "Ks"]]
    assert np.shape(bg.fluxratio.initval) == (2,)
    assert system.astrometryinstrument._background == [0, 0]
    (mu_ra, mu_dec), sigma = bulge_field_pm_prior(RA, DEC)
    assert 2.0 < sigma < 3.2
    assert np.atleast_1d(bg.pm_ra.mu)[0] == pytest.approx(mu_ra)
    assert np.atleast_1d(bg.pm_dec.mu)[0] == pytest.approx(mu_dec)
    assert np.atleast_1d(bg.pm_ra.sigma)[0] == pytest.approx(sigma)
    # the bulge moves toward negative galactic longitude: not ~0
    assert np.hypot(mu_ra, mu_dec) > 2.0


def test_catalog_keys_on_an_abs_dataset_raise(tmp_path):
    cfg = {
        "name": "x",
        "star": [{"name": "A"}],
        "astrometryinstrument": [
            {
                "name": "a",
                "file": str(tmp_path / "a.dat"),
                "mode": "abs",
                "catalog_span": [1.0, 2.0],
            }
        ],
    }
    with pytest.raises(ValueError, match="catalog-mode keys"):
        System(cfg, user_params={})


def test_psf_stars_on_an_unlensed_dataset_raise(tmp_path):
    _write_catalog(tmp_path / "c.dat", np.zeros(5), (0.2, 0.3, 0.3))
    cfg = {
        "name": "x",
        "star": [{"name": "A"}, {"name": "B"}],
        "astrometryinstrument": [
            {
                "name": "c",
                "file": str(tmp_path / "c.dat"),
                "mode": "catalog",
                "catalog_span": [float(GAIA_T[0]), float(GAIA_T[-1])],
                "psf_stars": ["B"],
                "fitpsfflux": True,
            }
        ],
    }
    params = {
        "star.A.ra": {"initval": RA_DEG},
        "star.A.dec": {"initval": DEC_DEG},
        "star.B.ra": {"initval": RA_DEG},
        "star.B.dec": {"initval": DEC_DEG},
    }
    with pytest.raises(ValueError, match="no lens term"):
        system = System(cfg, user_params=params)
        system.prepare()
