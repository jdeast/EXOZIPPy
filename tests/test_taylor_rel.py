"""Relative astrometry of an ACCELERATING Taylor orbit (review 8.8.14):
`rel` data naming a `type: constant_acceleration | constant_jerk` orbit.

One model, many perspectives (orbit.md "Taylor orbits"): the orbit has ONE
state, the primary group's reflex acceleration (and jerk), and each consumer
projects it --

  * the RVs read its line-of-sight component (gammadot, gammaddot);
  * the absolute astrometry of a primary-group star reads its sky components
    (accel_ra/accel_dec, jerk_ra/jerk_dec) as the photocenter's;
  * a rel dataset reads the RELATIVE vector x_C - x_P = -(M/M_c) x_P, the
    same sky components scaled by the member bodies' masses, on top of the
    relative position and velocity at the epoch (rel_dE/dN/vE/vN), which
    nothing absorbs in relative data.

Pinned here: joint recovery of one injected acceleration from simulated rel
sep/PA + Gaia epochs + RVs, against an independent numpy least-squares fit
of the same linear model; a jerk recovered from rel + RVs alone; the group
masses kept active on the orbit; finite C and JAX gradients; the refusals.
"""

import astropy.units as u
import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from exozippy.constants import DAYS_PER_YEAR
from exozippy.ephemeris import observer_position_at_reception
from exozippy.skyframe import parallax_factors
from exozippy.system import System

pytestmark = pytest.mark.slow

RA0, DEC0 = 150.0, 20.0
PLX = 20.0  # mas (50 pc)
PM = (30.0, -15.0)  # mas/yr
M_A, M_B = 1.0, 0.5  # Msun; pinned
K = -(M_A + M_B) / M_B  # relative / primary reflex = -M/M_c = -3
EPOCH = 2457900.0
SPAN = (2456900.0, 2458900.0)
TRUTH = dict(
    accel_ra=0.5,  # mas/yr^2, the PRIMARY's reflex
    accel_dec=-0.3,
    jerk_ra=0.12,  # mas/yr^3
    jerk_dec=-0.08,
    rel_dE=180.0,  # mas at EPOCH
    rel_dN=-60.0,
    rel_vE=-6.0,  # mas/yr
    rel_vN=2.5,
    gammadot=-0.4,  # m/s/day
    gammaddot=0.001,  # m/s/day^2
)
GAIA_ERR, REL_ERR, RV_ERR = 0.1, 0.4, 3.0


def _pfac(t):
    ra, dec = np.radians(RA0), np.radians(DEC0)
    xyz = observer_position_at_reception(t, ra, dec, observer_location="earth")
    return parallax_factors(xyz, ra, dec)


def _poly(t, c0, c1, a, j=0.0):
    dt = (t - EPOCH) / DAYS_PER_YEAR
    return c0 + c1 * dt + 0.5 * a * dt**2 + j * dt**3 / 6.0


def _write_data(tmp, jerk, gaia, seed=17):
    rng = np.random.default_rng(seed)
    jE = TRUTH["jerk_ra"] if jerk else 0.0
    jN = TRUTH["jerk_dec"] if jerk else 0.0
    out = {}
    # rel: companion minus primary, the relative vector with K x reflex
    t = np.sort(rng.uniform(*SPAN, 30))
    dE = _poly(
        t, TRUTH["rel_dE"], TRUTH["rel_vE"], K * TRUTH["accel_ra"], K * jE
    )
    dN = _poly(
        t, TRUTH["rel_dN"], TRUTH["rel_vN"], K * TRUTH["accel_dec"], K * jN
    )
    dE = dE + rng.normal(0, REL_ERR, t.size)
    dN = dN + rng.normal(0, REL_ERR, t.size)
    sep = np.hypot(dE, dN)
    pa = np.degrees(np.arctan2(dE, dN)) % 360.0
    np.savetxt(
        tmp / "ab.rel",
        np.column_stack(
            [t, sep, np.full(t.size, REL_ERR), pa, np.degrees(REL_ERR / sep)]
        ),
    )
    out["rel"] = (t, dE, dN)
    if gaia:
        tg = np.sort(rng.uniform(*SPAN, 60))
        psi = rng.uniform(0, 2 * np.pi, tg.size)
        P_E, P_N = _pfac(tg)
        dtg = (tg - EPOCH) / DAYS_PER_YEAR
        gE = (
            PM[0] * dtg
            + PLX * P_E
            + _poly(tg, 0.0, 0.0, TRUTH["accel_ra"], jE)
        )
        gN = (
            PM[1] * dtg
            + PLX * P_N
            + _poly(tg, 0.0, 0.0, TRUTH["accel_dec"], jN)
        )
        w = (
            gE * np.sin(psi)
            + gN * np.cos(psi)
            + rng.normal(0, GAIA_ERR, tg.size)
        )
        np.savetxt(
            tmp / "g.astrom",
            np.column_stack(
                [tg, w, np.full(tg.size, GAIA_ERR), np.degrees(psi)]
            ),
        )
        out["gaia"] = (tg, psi, w)
    t_rv = np.sort(rng.uniform(EPOCH - 600.0, EPOCH + 600.0, 30))
    t_rv[0], t_rv[-1] = EPOCH - 600.0, EPOCH + 600.0
    g2 = TRUTH["gammaddot"] if jerk else 0.0
    rv = (
        20.0
        + TRUTH["gammadot"] * (t_rv - EPOCH)
        + 0.5 * g2 * (t_rv - EPOCH) ** 2
        + rng.normal(0, RV_ERR, t_rv.size)
    )
    np.savetxt(
        tmp / "a.rv", np.column_stack([t_rv, rv, np.full(t_rv.size, RV_ERR)])
    )
    out["rv"] = (t_rv, rv)
    return out


def _config(tmp, otype, gaia=True, companion=True):
    insts = [
        {
            "name": "R",
            "file": str(tmp / "ab.rel"),
            "mode": "rel",
            "orbit": "AB",
            "epoch": EPOCH,
        }
    ]
    if gaia:
        insts.append(
            {
                "name": "G",
                "file": str(tmp / "g.astrom"),
                "mode": "gaia",
                "observer_location": "earth",
                "epoch": EPOCH,
            }
        )
    orbit = {"name": "AB", "type": otype, "primary": ["A"]}
    orbit["companion"] = ["B"] if companion else []
    config = {
        "name": "relacc",
        "star": [{"name": "A", "mist": False}, {"name": "B", "mist": False}],
        "orbit": [orbit],
        "astrometryinstrument": insts,
        "rvinstrument": [{"name": "RV", "file": str(tmp / "a.rv")}],
    }
    # The jitters pinned at 0: free, each dataset's would reweigh Gaia
    # against rel, and the shared acceleration would no longer be the
    # nominal-error least-squares one this test compares with.
    up = {
        f"astrometryinstrument.{n}.jitter_variance": {
            "initval": 0.0,
            "sigma": 0.0,
        }
        for n in (["R", "G"] if gaia else ["R"])
    }
    for s, m in (("A", M_A), ("B", M_B)):
        up[f"star.{s}.logmass"] = {"initval": np.log10(m), "sigma": 0.0}
        up[f"star.{s}.radius"] = {"initval": 1.0, "sigma": 0.1}
        up[f"star.{s}.teff"] = {"initval": 5800, "sigma": 100}
        up[f"star.{s}.feh"] = {"initval": 0.0, "sigma": 0.1}
        up[f"star.{s}.ra"] = {"initval": RA0}
        up[f"star.{s}.dec"] = {"initval": DEC0}
        up[f"star.{s}.pm_ra"] = {"initval": PM[0]}
        up[f"star.{s}.pm_dec"] = {"initval": PM[1]}
        up[f"star.{s}.distance"] = {"initval": 1000.0 / PLX}
    return config, up


def _values(model, nodes, point):
    nodes = model.replace_rvs_by_values(nodes)
    fn = pytensor.function(model.value_vars, nodes, on_unused_input="ignore")
    return [
        float(np.ravel(x)[0])
        for x in fn(*[point[v.name] for v in model.value_vars])
    ]


def _max_likelihood(model, start):
    from scipy.optimize import minimize

    logl = model.logp(vars=model.observed_RVs, sum=True)
    values = list(model.value_vars)
    f = pytensor.function(
        values,
        [logl] + pt.grad(logl, values, disconnected_inputs="ignore"),
        on_unused_input="ignore",
    )
    shapes = [np.shape(start[v.name]) for v in values]
    sizes = [int(np.prod(sh)) for sh in shapes]

    def unpack(x):
        out, k = [], 0
        for sh, n in zip(shapes, sizes):
            out.append(np.asarray(x[k : k + n]).reshape(sh))
            k += n
        return out

    def neg(x):
        lg, *g = f(*unpack(x))
        return -float(lg), -np.concatenate([np.ravel(gi) for gi in g])

    x0 = np.concatenate([np.ravel(start[v.name]) for v in values])
    res = minimize(
        neg,
        x0,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 30000, "ftol": 1e-15, "gtol": 1e-10},
    )
    return {v.name: a for v, a in zip(values, unpack(res.x))}


def _lsq(data, jerk, gaia):
    """Independent weighted least squares of the SAME linear model: Gaia
    along-scan rows (dE0, dN0, plx, pmE, pmN, aE, aN[, jE, jN]) and rel
    E/N rows (rel_dE, rel_dN, rel_vE, rel_vN, K aE, K aN[, K jE, K jN]),
    the acceleration (and jerk) SHARED.  Returns ({name: value}, {name:
    sigma})."""
    names = ["accel_ra", "accel_dec"] + (
        ["jerk_ra", "jerk_dec"] if jerk else []
    )
    names += ["rel_dE", "rel_dN", "rel_vE", "rel_vN"]
    extra = ["dE0", "dN0", "plx", "pmE", "pmN"] if gaia else []
    cols = names + extra
    rows, y = [], []
    t, dE, dN = data["rel"]
    dt = (t - EPOCH) / DAYS_PER_YEAR
    for comp, obs in (("E", dE), ("N", dN)):
        A = np.zeros((t.size, len(cols)))
        A[:, cols.index("accel_ra" if comp == "E" else "accel_dec")] = (
            K * 0.5 * dt**2
        )
        if jerk:
            A[:, cols.index("jerk_ra" if comp == "E" else "jerk_dec")] = (
                K * dt**3 / 6
            )
        A[:, cols.index(f"rel_d{comp}")] = 1.0
        A[:, cols.index(f"rel_v{comp}")] = dt
        rows.append(A / REL_ERR)
        y.append(obs / REL_ERR)
    if gaia:
        tg, psi, w = data["gaia"]
        P_E, P_N = _pfac(tg)
        d = (tg - EPOCH) / DAYS_PER_YEAR
        s, c = np.sin(psi), np.cos(psi)
        A = np.zeros((tg.size, len(cols)))
        A[:, cols.index("dE0")] = s
        A[:, cols.index("dN0")] = c
        A[:, cols.index("plx")] = P_E * s + P_N * c
        A[:, cols.index("pmE")] = d * s
        A[:, cols.index("pmN")] = d * c
        A[:, cols.index("accel_ra")] = 0.5 * d**2 * s
        A[:, cols.index("accel_dec")] = 0.5 * d**2 * c
        if jerk:
            A[:, cols.index("jerk_ra")] = d**3 / 6 * s
            A[:, cols.index("jerk_dec")] = d**3 / 6 * c
        rows.append(A / GAIA_ERR)
        y.append(w / GAIA_ERR)
    A, y = np.vstack(rows), np.concatenate(y)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    sig = np.sqrt(np.diag(np.linalg.inv(A.T @ A)))
    return dict(zip(cols, coef)), dict(zip(cols, sig))


@pytest.fixture(
    scope="module",
    params=[("constant_acceleration", True), ("constant_jerk", False)],
    ids=["accel-rel+gaia+rv", "jerk-rel+rv"],
)
def fit(request, tmp_path_factory):
    otype, gaia = request.param
    jerk = otype == "constant_jerk"
    tmp = tmp_path_factory.mktemp(otype)
    data = _write_data(tmp, jerk, gaia)
    config, up = _config(tmp, otype, gaia=gaia)
    system = System(config, user_params=up)
    system.prepare()
    model = system.build_model()
    start = system.get_raw_starts(model)[0][0]
    best = _max_likelihood(model, start)
    return jerk, gaia, system, model, data, start, best


def test_one_acceleration_is_recovered_through_every_perspective(fit):
    """
    Given rel sep/PA epochs of B around A, Gaia epochs of A (constant
      acceleration case) and RVs of A, all simulated in numpy from ONE
      reflex acceleration of A (and jerk) -- the rel data seeing it as the
      relative vector, -(M_A + M_B)/M_B = -3 times A's reflex, on top of
      their own relative position and velocity -- with the masses pinned,
    When one Taylor orbit read by all of them is fit by maximum likelihood,
    Then accel_ra/dec (jerk_ra/dec), rel_dE/dN/vE/vN agree with an
      independent numpy least-squares fit of the same linear model to 0.1
      sigma and with the injection to 4 sigma; gammadot (gammaddot) with the
      injection to 4 sigma; and the start (data seeds) is finite.
    """
    # Arrange
    jerk, gaia, system, model, data, start, best = fit
    orbit = system.orbit
    lsq, sig = _lsq(data, jerk, gaia)
    names = [n for n in lsq if n in TRUTH]

    # Act
    got = dict(
        zip(
            names,
            _values(model, [getattr(orbit, n).value for n in names], best),
        )
    )

    # Assert
    for n in names:
        assert abs(got[n] - lsq[n]) < 0.1 * sig[n], (n, got[n], lsq[n], sig[n])
        assert abs(got[n] - TRUTH[n]) < 4 * sig[n], (
            n,
            got[n],
            TRUTH[n],
            sig[n],
        )
    # The line of sight: an RV slope (and curvature) about the RV midpoint,
    # which is the shared epoch here.
    g = _values(model, [orbit.gammadot.value], best)[0]
    msd = float((u.solRad / u.d**2).to(u.m / u.s / u.d))
    assert g * msd == pytest.approx(TRUTH["gammadot"], abs=0.02)
    if jerk:
        g2 = _values(model, [orbit.gammaddot.value], best)[0]
        g2 *= float((u.solRad / u.d**3).to(u.m / u.s / u.d**2))
        assert g2 == pytest.approx(TRUTH["gammaddot"], abs=3e-4)
    assert orbit.taylor_epoch[0] == EPOCH
    assert np.isfinite(model.compile_logp()(start))


def test_the_group_masses_live_on_the_rel_read_orbit(fit):
    """
    Given the rel-read accelerating Taylor orbit,
    When its manifest and the relative factor are read,
    Then m_primary, m_companion and m_total are declared (the one place a
      Taylor orbit has masses), equal the member stars' (1.0, 0.5, 1.5
      Msun), and the relative factor is -M/M_c = -3.
    """
    # Arrange
    _jerk, _gaia, system, model, _data, start, _best = fit
    orbit = system.orbit

    # Act
    mp, mc, mt, k = _values(
        model,
        [
            orbit.m_primary.value,
            orbit.m_companion.value,
            orbit.m_total.value,
            orbit._relative_reflex_factor(0),
        ],
        start,
    )

    # Assert
    assert {"m_primary", "m_companion", "m_total"} <= set(orbit.manifest)
    np.testing.assert_allclose(
        [mp, mc, mt, k], [M_A, M_B, M_A + M_B, K], rtol=1e-9
    )


def test_the_gradient_is_finite_on_both_backends(fit):
    """
    Given the joint model,
    When logp and its gradient are evaluated at the start in the C backend
      and through JAX,
    Then all are finite.
    """
    # Arrange
    _jerk, _gaia, system, model, _data, start, _best = fit

    # Act
    dlogp = model.compile_dlogp()(start)

    # Assert
    assert np.all(np.isfinite(dlogp))
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    from pymc.sampling.jax import get_jaxified_logp

    logp_fn = get_jaxified_logp(model)
    vals = [start[v.name] for v in model.value_vars]
    grads = jax.grad(lambda vs: logp_fn(vs))(vals)
    bad = [
        v.name
        for v, g in zip(model.value_vars, grads)
        if not np.all(np.isfinite(np.asarray(g)))
    ]
    assert not bad, f"non-finite JAX gradients: {bad}"


def test_a_rel_read_accelerating_orbit_needs_a_companion(tmp_path):
    """
    Given a rel dataset naming a constant_acceleration orbit whose
      companion group is empty,
    When the system is prepared,
    Then it raises: the relative vector is -(M/M_c) times the reflex, so the
      companion needs a mass (a rel dataset traces a companion anyway).
    """
    # Arrange
    _write_data(tmp_path, jerk=False, gaia=False)
    config, up = _config(
        tmp_path, "constant_acceleration", gaia=False, companion=False
    )

    # Act / Assert
    with pytest.raises((ValueError, NotImplementedError), match="companion"):
        System(config, user_params=up).prepare()
