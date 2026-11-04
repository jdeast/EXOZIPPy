"""Sky-plane Taylor terms: absolute astrometry of a star on a
`type: constant_acceleration | constant_jerk` orbit, and the one 3-D
acceleration the RVs and the astrometry share (review 8.8.14; orbit.md
"Taylor orbits").

A long-period companion too slow for the data to resolve accelerates its
primary.  A Taylor orbit's state is the primary group's reflex as a
polynomial about the orbit's epoch, served by `Orbit.state_vectors` like a
Keplerian orbit's: the RVs read its line-of-sight component (`gammadot`,
`gammaddot`), the gaia/abs/catalog photocenter its sky components
(`accel_ra`/`accel_dec` in mas/yr^2, `jerk_ra`/`jerk_dec` in mas/yr^3,
Gaia DR3's g t^2/2 + gdot t^3/6 convention), each consumer absorbing the
constant and linear terms in its own constants (gamma; ra/dec/pm).  A
constant velocity is invisible to both.

Pinned here:
1. The model term against the closed form, through the ONE absolute
   builder, and the default epoch (the astrometric reference epoch).
2. Recovery from simulated Gaia epochs by maximum likelihood, against an
   independent numpy least-squares fit.
3. JOINT recovery: one constant acceleration vector injected into RVs and
   Gaia epochs together comes back as gammadot + accel_ra/dec, and the
   derived |A| and M_c/r^2 = |A|/G match the injected vector; with a jerk,
   gammaddot and jerk_ra/dec come back too, the RV slope mapped from the RV
   epoch to the shared one.
4. The catalog projection: a catalog solution referred to a later epoch
   sees the acceleration as a proper-motion shift.
5. The photocenter weighting with a luminous companion.
6. The plotted along-scan model is the likelihood's; finite C and JAX
   gradients.
7. What raises: the removed `linear`/`quadratic` spellings, a
   companion-group target, rel astrometry, a constant_velocity orbit only
   RVs/absolute astrometry read, RVs of a uniform_rotation orbit, a sampled
   light the Taylor orbit cannot place.
"""

import astropy.constants as const
import astropy.units as u
import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from exozippy.components.astrometryinstrument.astrometryinstrument import (
    CATALOG_ROLES,
    catalog_design_matrix,
)
from exozippy.ephemeris import observer_position_at_reception
from exozippy.skyframe import parallax_factors
from exozippy.system import System

pytestmark = pytest.mark.slow

RAD2MAS = 180.0 / np.pi * 3600e3
YR = 365.25

# A nearby star accelerated by an unseen outer companion.  The epoch is
# deliberately NOT the middle of the data, so the quadratic has a linear
# part over the span that only the acceleration explains.
_T = dict(
    ra0=150.0,
    dec0=20.0,
    plx=20.0,
    pmra=30.0,
    pmdec=-15.0,
    accel_ra=0.8,
    accel_dec=-0.5,
)
_SPAN = (2456900.0, 2458900.0)  # ~5.5 yr, DR4-like
_EPOCH = 2457900.0
_ERR = 0.1
# The fit's start: ~50 sigma from the truth in each component.
_SEED = (0.3, -0.2)


def _pfac(t, ra_deg=_T["ra0"], dec_deg=_T["dec0"]):
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    xyz = observer_position_at_reception(t, ra, dec, observer_location="earth")
    return parallax_factors(xyz, ra, dec)


def _truth_track(t, epoch=_EPOCH, accel=(_T["accel_ra"], _T["accel_dec"])):
    """(dE, dN) mas of the accelerated star, from the definition."""
    P_E, P_N = _pfac(t)
    dt = (t - epoch) / YR
    dE = _T["pmra"] * dt + _T["plx"] * P_E + 0.5 * accel[0] * dt**2
    dN = _T["pmdec"] * dt + _T["plx"] * P_N + 0.5 * accel[1] * dt**2
    return dE, dN


def _write_gaia(path, n=60, seed=5, noise=True, epoch=_EPOCH):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(*_SPAN, n))
    psi = rng.uniform(0.0, 2.0 * np.pi, n)
    dE, dN = _truth_track(t, epoch=epoch)
    w = dE * np.sin(psi) + dN * np.cos(psi)
    if noise:
        w = w + rng.normal(0.0, _ERR, n)
    np.savetxt(
        path, np.column_stack([t, w, np.full(n, _ERR), np.degrees(psi)])
    )
    return t, psi, w


def _star_params(names, masses=None):
    up = {}
    for k, s in enumerate(names):
        m = 1.0 if masses is None else masses[k]
        up[f"star.{s}.logmass"] = {"initval": np.log10(m), "sigma": 0.0}
        up[f"star.{s}.radius"] = {"initval": 1.0, "sigma": 0.1}
        up[f"star.{s}.teff"] = {"initval": 5800, "sigma": 100}
        up[f"star.{s}.feh"] = {"initval": 0.0, "sigma": 0.1}
        up[f"star.{s}.ra"] = {"initval": _T["ra0"]}
        up[f"star.{s}.dec"] = {"initval": _T["dec0"]}
        up[f"star.{s}.pm_ra"] = {"initval": _T["pmra"]}
        up[f"star.{s}.pm_dec"] = {"initval": _T["pmdec"]}
        up[f"star.{s}.distance"] = {"initval": 1000.0 / _T["plx"]}
    return up


def _gaia_inst(path, name="G", star_ndx=0, epoch=None):
    inst = {
        "name": name,
        "file": str(path),
        "mode": "gaia",
        "observer_location": "earth",
        "star_ndx": star_ndx,
    }
    if epoch is not None:
        inst["epoch"] = epoch
    return inst


def _accel_params(orbit="outer", accel=(_T["accel_ra"], _T["accel_dec"])):
    return {
        f"orbit.{orbit}.accel_ra": {"initval": accel[0]},
        f"orbit.{orbit}.accel_dec": {"initval": accel[1]},
    }


def _build(config, up):
    system = System(config, user_params=up)
    system.prepare()
    model = system.build_model()
    return system, model


def _start(system, model):
    """The sampler's raw start (run.py's), NOT model.initial_point()."""
    return system.get_raw_starts(model)[0][0]


def _at(model, nodes, point):
    """``nodes`` at a raw-space ``point``, every RV replaced by its value
    variable (evaluated on the RVs, a node draws from the prior)."""
    nodes = model.replace_rvs_by_values(nodes)
    fn = pytensor.function(model.value_vars, nodes, on_unused_input="ignore")
    return [
        np.asarray(x) for x in fn(*[point[v.name] for v in model.value_vars])
    ]


def _max_likelihood(model, start):
    """The maximum of the summed OBSERVED log-likelihood over the model's
    free raw variables (L-BFGS-B, analytic gradient), from ``start``; the
    priors' soft terms would otherwise move it.  Returns a raw point."""
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
        options={"maxiter": 20000, "ftol": 1e-15, "gtol": 1e-10},
    )
    return {v.name: a for v, a in zip(values, unpack(res.x))}


def _taylor_only(system, i, t):
    """Dataset i's sky-plane Taylor term alone, as the likelihood builds it."""
    inst = system.astrometryinstrument
    beta = inst._sed_beta_node(system, i)
    return inst._photocenter_offset(
        system, i, beta, pt.as_tensor_variable(t), keplerian=False
    )


# ---------------------------------------------------------------------------
# 1-2, 5-6: one accelerated star with Gaia epochs
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def accel_case(tmp_path_factory):
    """Star A, Gaia epochs with the injected acceleration (noisy), one
    constant_acceleration orbit 'outer' with A as its primary and an unseen companion,
    the dataset's reference epoch _EPOCH and no orbit `epoch:`.  The
    acceleration is SEEDED at _SEED, not the truth, so the start is a point
    to check the closed form at and a displaced start to fit from."""
    tmp = tmp_path_factory.mktemp("accel")
    t, psi, w = _write_gaia(tmp / "g.astrom")
    config = {
        "name": "acc",
        "star": [{"name": "A", "mist": False}],
        "orbit": [
            {
                "name": "outer",
                "type": "constant_acceleration",
                "primary": ["A"],
            }
        ],
        "astrometryinstrument": [_gaia_inst(tmp / "g.astrom", epoch=_EPOCH)],
    }
    up = _star_params(["A"])
    up.update(_accel_params(accel=_SEED))
    system, model = _build(config, up)
    return system, model, t, psi, w


def test_the_sky_term_is_half_the_acceleration_times_dt_squared(accel_case):
    """
    Given a star on a constant_acceleration orbit, its acceleration seeded at
      (0.3, -0.2) mas/yr^2 and the orbit's epoch left to its default,
    When the absolute model and the Taylor term alone are evaluated at the
      start,
    Then the epoch is the astrometric reference epoch, the Taylor term is
      accel (t - epoch)**2 / 2 with t in Julian years, and the whole model
      is the definition's pm + parallax + acceleration track at the seed,
      both to 1e-6 mas.
    """
    # Arrange
    system, model, t, _psi, _w = accel_case
    inst = system.astrometryinstrument
    d = inst.datasets[0]
    beta = inst._sed_beta_node(system, 0)
    dt = (t - _EPOCH) / YR

    # Act
    tE, tN = _taylor_only(system, 0, t)
    dE, dN = inst._absolute_model(system, d, d["time"], beta)
    tE, tN, dE, dN = _at(model, [tE, tN, dE, dN], _start(system, model))

    # Assert
    assert system.orbit.taylor_epoch[0] == inst.epoch == _EPOCH
    np.testing.assert_allclose(tE, 0.5 * _SEED[0] * dt**2, atol=1e-6)
    np.testing.assert_allclose(tN, 0.5 * _SEED[1] * dt**2, atol=1e-6)
    eE, eN = _truth_track(t, accel=_SEED)
    np.testing.assert_allclose(dE, eE, atol=1e-6)
    np.testing.assert_allclose(dN, eN, atol=1e-6)
    # The orbit declares the two coefficients and no Keplerian element.
    assert {"accel_ra", "accel_dec"} <= set(system.orbit.manifest)
    assert "period" not in system.orbit.manifest


def test_the_acceleration_is_recovered_from_simulated_gaia_epochs(accel_case):
    """
    Given noisy (0.1 mas) Gaia along-scan epochs of a star with a known
      sky acceleration, simulated in numpy from the definition,
    When the model's likelihood is maximized from the seeded start (~50
      sigma away in each component),
    Then the fitted acceleration is within 4 sigma of the injected one in
      each component (sigma from an independent numpy least-squares fit of
      the same data), and agrees with that least-squares fit to 0.1 sigma.
    """
    # Arrange
    system, model, t, psi, w = accel_case
    # Independent linear least squares on (dE0, dN0, plx, pmE, pmN, aE, aN).
    P_E, P_N = _pfac(t)
    dt = (t - _EPOCH) / YR
    s, c = np.sin(psi), np.cos(psi)
    A = np.column_stack(
        [
            s,
            c,
            P_E * s + P_N * c,
            dt * s,
            dt * c,
            0.5 * dt**2 * s,
            0.5 * dt**2 * c,
        ]
    )
    coef, *_ = np.linalg.lstsq(A / _ERR, w / _ERR, rcond=None)
    cov = np.linalg.inv((A / _ERR).T @ (A / _ERR))
    lsq, sig = coef[5:], np.sqrt(np.diag(cov))[5:]
    orbit = system.orbit

    # Act
    best = _max_likelihood(model, _start(system, model))
    aE, aN = _at(model, [orbit.accel_ra.value, orbit.accel_dec.value], best)
    fitted = np.array([float(np.ravel(aE)[0]), float(np.ravel(aN)[0])])

    # Assert
    truth = np.array([_T["accel_ra"], _T["accel_dec"]])
    assert np.all(np.abs(fitted - truth) < 4.0 * sig), (fitted, truth, sig)
    np.testing.assert_allclose(fitted, lsq, atol=0.1 * sig.min())


def test_the_plotted_model_is_the_likelihood_model(accel_case):
    """
    Given the accelerated star at its start point,
    When plot_data builds the along-scan chart,
    Then its model trace carries the acceleration: it is the definition's
      along-scan model to 1e-6 mas, and plot() writes the sky plot with no
      Keplerian orbit to draw (its orbit panel is dropped, its path panel
      kept).
    """
    # Arrange
    system, model, t, psi, _w = accel_case
    inst = system.astrometryinstrument
    point = system.get_internal_point(model, _start(system, model))
    system.compile_plotter_functions(model)

    # Act
    spec = inst.plot_data(system, point)[0]
    trace = [tr for tr in spec.traces if tr.role == "model"][0]

    # Assert
    eE, eN = _truth_track(t, accel=_SEED)
    np.testing.assert_allclose(
        trace.y, eE * np.sin(psi) + eN * np.cos(psi), atol=1e-6
    )
    assert inst._compiled_photo[0] is not None
    assert inst._compiled_kepler_photo[0] is None


def test_the_gradient_is_finite_on_both_backends(accel_case):
    """
    Given the accelerated star,
    When logp and its gradient are evaluated at the start on the C backend
      and through JAX (the numpyro/blackjax path),
    Then all are finite, and the gradient in the acceleration is nonzero
      (the term reaches the likelihood).
    """
    # Arrange
    system, model, *_ = accel_case
    start = _start(system, model)

    # Act
    logp = model.compile_logp()(start)
    dlogp = model.compile_dlogp()(start)

    # Assert
    assert np.isfinite(logp)
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
    g_accel = [
        np.asarray(g)
        for v, g in zip(model.value_vars, grads)
        if v.name.startswith("orbit.accel_")
    ]
    assert g_accel and all(np.any(g != 0.0) for g in g_accel)


# ---------------------------------------------------------------------------
# 3. The catalog projection
# ---------------------------------------------------------------------------


def _write_catalog(path, sol, epoch, errs=(0.2, 0.3, 0.25)):
    ra = _T["ra0"] + np.degrees(
        sol[0] / RAD2MAS / np.cos(np.radians(_T["dec0"]))
    )
    dec = _T["dec0"] + np.degrees(sol[1] / RAD2MAS)
    pos, plx_e, pm_e = errs
    row = [epoch, ra, dec, sol[2], sol[3], sol[4]]
    row += [pos, pos, plx_e, pm_e, pm_e] + [0.0] * 10
    assert len(row) == len(CATALOG_ROLES[5])
    np.savetxt(path, np.atleast_2d(row), fmt="%.12f")


def _catalog_system(tmp_path, inst_epoch, cat_epoch, epochs, orbit_epoch):
    _write_catalog(tmp_path / "c.dat", np.zeros(5), cat_epoch)
    orbit = {
        "name": "outer",
        "type": "constant_acceleration",
        "primary": ["A"],
    }
    if orbit_epoch is not None:
        orbit["epoch"] = orbit_epoch
    config = {
        "name": "cat",
        "star": [{"name": "A", "mist": False}],
        "orbit": [orbit],
        "astrometryinstrument": [
            {
                "name": "DR3",
                "file": str(tmp_path / "c.dat"),
                "mode": "catalog",
                "observer_location": "earth",
                "epoch": inst_epoch,
                "catalog_epochs": [float(x) for x in epochs],
            }
        ],
    }
    up = _star_params(["A"])
    up.update(_accel_params())
    system, model = _build(config, up)
    inst = system.astrometryinstrument
    d = inst.datasets[0]
    beta = inst._sed_beta_node(system, 0)
    dE, dN = inst._absolute_model(system, d, d["time"], beta)
    (sol,) = _at(
        model, [inst._catalog_solution(d, dE, dN)], _start(system, model)
    )
    return system, sol


@pytest.mark.parametrize(
    "orbit_epoch", [None, 2456500.0], ids=["default", "user"]
)
def test_a_catalog_solution_sees_the_acceleration_as_a_pm_shift(
    tmp_path, orbit_epoch
):
    """
    Given a five-parameter catalog (J2016.0) of an accelerated star whose
      ra/dec/pm are referred to an astrometric epoch E 3 yr EARLIER, and the
      Taylor orbit's epoch left at its default (E) or set by the user to
      another instant,
    When the model's catalog solution is evaluated at the start,
    Then it equals an independent numpy least-squares projection of the
      definition's track -- pm (t - E) + 0.5 accel (t - E)^2, the constants
      absorbed at E whatever the orbit's own epoch -- onto the catalog's
      design matrix (1e-6), and its proper motion is the star's plus
      accel x 3 yr to 2%: the DR3-vs-DR4 proper-motion difference an
      unmodeled outer companion produces.
    """
    # Arrange
    cat_epoch = 2457389.0  # J2016.0
    inst_epoch = cat_epoch - 3.0 * YR
    epochs = np.linspace(cat_epoch - 510.0, cat_epoch + 510.0, 61)

    # Act
    system, sol = _catalog_system(
        tmp_path, inst_epoch, cat_epoch, epochs, orbit_epoch
    )

    # Assert
    expected_epoch = inst_epoch if orbit_epoch is None else orbit_epoch
    assert system.orbit.taylor_epoch[0] == expected_epoch
    P_E, P_N = _pfac(epochs)
    dt = (epochs - inst_epoch) / YR
    tE = _T["pmra"] * dt + _T["plx"] * P_E + 0.5 * _T["accel_ra"] * dt**2
    tN = _T["pmdec"] * dt + _T["plx"] * P_N + 0.5 * _T["accel_dec"] * dt**2
    M = catalog_design_matrix(epochs, cat_epoch, P_E, P_N, 5)
    expected, *_ = np.linalg.lstsq(M, np.concatenate([tE, tN]), rcond=None)
    np.testing.assert_allclose(sol, expected, atol=1e-6)
    shift = sol[3:] - np.array([_T["pmra"], _T["pmdec"]])
    np.testing.assert_allclose(
        shift, 3.0 * np.array([_T["accel_ra"], _T["accel_dec"]]), rtol=0.02
    )


# ---------------------------------------------------------------------------
# 4. The photocenter weighting with a luminous companion
# ---------------------------------------------------------------------------


def _binary_with_outer(tmp, outer_primary, beta):
    """A-B Keplerian binary (A primary), B luminous through a sampled
    fluxfrac ``beta`` in A's PSF, and a constant_acceleration orbit whose primary group
    is ``outer_primary``."""
    t, _psi, _w = _write_gaia(tmp / "g.astrom", noise=False)
    config = {
        "name": "bin",
        "star": [{"name": "A", "mist": False}, {"name": "B", "mist": False}],
        "orbit": [
            {"name": "AB", "primary": ["A"], "companion": ["B"]},
            {
                "name": "outer",
                "type": "constant_acceleration",
                "primary": outer_primary,
            },
        ],
        "astrometryinstrument": [_gaia_inst(tmp / "g.astrom", epoch=_EPOCH)],
    }
    up = _star_params(["A", "B"], masses=[1.0, 0.5])
    up.update(_accel_params())
    up.update(
        {
            "orbit.AB.period": {"initval": 400.0},
            "orbit.AB.tc": {"initval": 2457100.0},
            "orbit.AB.bigomega": {"initval": 60.0},
            "orbit.AB.cosi": {"initval": 0.5},
            "astrometryinstrument.G.fluxfrac": {"initval": beta},
        }
    )
    system, model = _build(config, up)
    return system, model, t


@pytest.mark.parametrize(
    "outer_primary, share",
    [(["A", "B"], lambda b: 1.0), (["A"], lambda b: 1.0 - b)],
    ids=["both-lights-accelerate", "target-only"],
)
def test_the_photocenter_accelerates_with_the_light_in_the_primary_group(
    tmp_path, outer_primary, share
):
    """
    Given a gaia target A with a luminous Keplerian companion B in its PSF
      (sampled fluxfrac b = 0.3), and an outer constant_acceleration orbit whose
      primary group is either both stars or A alone,
    When the Taylor term of the photocenter is evaluated at the start,
    Then it is accel (t - epoch)**2 / 2 times the share of the light that
      accelerates: all of it (b cancels) when both are in the primary
      group, (1 - b) when only the target is -- the Keplerian rows of the
      photocenter table with the group's acceleration in place of its
      orbit.
    """
    # Arrange
    b = 0.3
    system, model, t = _binary_with_outer(tmp_path, outer_primary, b)
    dt = (t - _EPOCH) / YR

    # Act
    tE, tN = _at(
        model, list(_taylor_only(system, 0, t)), _start(system, model)
    )

    # Assert
    k = share(b)
    np.testing.assert_allclose(
        tE, k * 0.5 * _T["accel_ra"] * dt**2, rtol=1e-9, atol=1e-9
    )
    np.testing.assert_allclose(
        tN, k * 0.5 * _T["accel_dec"] * dt**2, rtol=1e-9, atol=1e-9
    )


# ---------------------------------------------------------------------------
# 7. Refusals, and the legal mixed cases
# ---------------------------------------------------------------------------


def _single(tmp, orbit, star_names=("A",), star_ndx=0, mode="gaia", up=None):
    _write_gaia(tmp / "g.astrom", noise=False)
    inst = _gaia_inst(tmp / "g.astrom", star_ndx=star_ndx, epoch=_EPOCH)
    inst["mode"] = mode
    config = {
        "name": "ref",
        "star": [{"name": s, "mist": False} for s in star_names],
        "orbit": [orbit],
        "astrometryinstrument": [inst],
    }
    p = _star_params(list(star_names))
    p.update(up or {})
    return config, p


def test_a_companion_group_target_is_refused(tmp_path):
    """
    Given a gaia target B in a constant_acceleration orbit's COMPANION group,
    When the system is prepared,
    Then it raises naming the orbit and the mass ratio a Taylor orbit does
      not have.
    """
    # Arrange
    orbit = {
        "name": "outer",
        "type": "constant_acceleration",
        "primary": ["A"],
        "companion": ["B"],
    }
    config, up = _single(tmp_path, orbit, ("A", "B"), star_ndx=1)

    # Act / Assert
    with pytest.raises(NotImplementedError, match=r"outer.*COMPANION group"):
        System(config, user_params=up).prepare()


def test_relative_astrometry_of_a_uniform_rotation_orbit_is_refused(tmp_path):
    """
    Given a rel dataset naming a uniform_rotation orbit (the lens's polar
      frame; the Cartesian types ARE rel-readable, tests/test_taylor_rel.py),
    When the system is prepared,
    Then it raises naming the relative reader's limits.
    """
    # Arrange
    rel = tmp_path / "r.rel"
    np.savetxt(
        rel,
        np.column_stack(
            [
                np.linspace(*_SPAN, 5),
                np.full(5, 100.0),
                np.full(5, 1.0),
                np.full(5, 30.0),
                np.full(5, 1.0),
            ]
        ),
    )
    config = {
        "name": "rel",
        "star": [{"name": "A", "mist": False}, {"name": "B", "mist": False}],
        "orbit": [
            {
                "name": "outer",
                "type": "uniform_rotation",
                "primary": ["A"],
                "companion": ["B"],
            }
        ],
        "astrometryinstrument": [
            {"name": "R", "file": str(rel), "mode": "rel", "orbit": "outer"}
        ],
    }

    # Act / Assert
    with pytest.raises(NotImplementedError, match="relative"):
        System(config, user_params=_star_params(["A", "B"])).prepare()


def test_a_constant_velocity_orbit_absolute_astrometry_reads_is_unobservable(
    tmp_path,
):
    """
    Given a constant_velocity orbit whose only reader is gaia astrometry of
      its primary,
    When the system is prepared,
    Then it raises "nothing observable": a constant sky velocity is the
      proper motion, so the orbit would add free dimensions no likelihood
      constrains.
    """
    # Arrange
    orbit = {"name": "outer", "type": "constant_velocity", "primary": ["A"]}
    config, up = _single(tmp_path, orbit)

    # Act / Assert
    with pytest.raises(ValueError, match="nothing observable"):
        System(config, user_params=up).prepare()


@pytest.mark.parametrize(
    "old, new",
    [("linear", "constant_acceleration"), ("quadratic", "constant_jerk")],
)
def test_the_removed_taylor_spellings_raise_with_the_migration(
    tmp_path, old, new
):
    """
    Given an orbit block with the pre-2026-11 `type: linear` / `quadratic`,
    When the system is prepared,
    Then it raises naming the new spelling (the RV slope's
      constant_acceleration, the curvature's constant_jerk; the lens's old
      linear motion's uniform_rotation) -- one spelling, never a silent
      alias.
    """
    # Arrange
    config, up = _single(
        tmp_path, {"name": "outer", "type": old, "primary": ["A"]}
    )

    # Act / Assert
    with pytest.raises(ValueError, match=new) as exc:
        System(config, user_params=up).prepare()
    if old == "linear":
        assert "uniform_rotation" in str(exc.value)


def test_a_sampled_light_with_no_keplerian_orbit_to_place_it_raises(tmp_path):
    """
    Given a gaia target on a constant_acceleration orbit and NO Keplerian one, with a
      nonzero sampled fluxfrac,
    When the model is built,
    Then it raises naming the dataset: the light would stand on the Taylor
      orbit's other side, whose motion needs a mass ratio.
    """
    # Arrange
    orbit = {
        "name": "outer",
        "type": "constant_acceleration",
        "primary": ["A"],
    }
    config, up = _single(
        tmp_path,
        orbit,
        up={
            **_accel_params(),
            "astrometryinstrument.G.fluxfrac": {"initval": 0.2},
        },
    )
    system = System(config, user_params=up)
    system.prepare()

    # Act / Assert
    with pytest.raises(
        ValueError, match=r"\[astrometryinstrument\.G\].*no Keplerian"
    ):
        system.build_model()


def _rv_file(path):
    rng = np.random.default_rng(3)
    t = np.sort(rng.uniform(2457600.0, 2458200.0, 20))
    np.savetxt(
        path,
        np.column_stack([t, 300.0 - 0.5 * (t - t.mean()), np.full(20, 5.0)]),
    )
    return t


def test_an_orbit_read_by_rvs_and_astrometry_declares_one_vector(tmp_path):
    """
    Given a constant_acceleration orbit read by the RVs AND the gaia
      astrometry of its primary,
    When the system is built,
    Then it declares gammadot, accel_ra and accel_dec, the derived
      accel_total and mc_over_r2 (an equality) and NOT the RV-only bound
      mc_over_r2_min; the shared epoch is the astrometric one while each
      gamma stays at the RV midpoint; and logp is finite.
    """
    # Arrange
    t_rv = _rv_file(tmp_path / "a.rv")
    orbit = {
        "name": "outer",
        "type": "constant_acceleration",
        "primary": ["A"],
    }
    config, up = _single(tmp_path, orbit)
    config["rvinstrument"] = [{"name": "RV", "file": str(tmp_path / "a.rv")}]

    # Act
    system, model = _build(config, up)

    # Assert
    names = set(system.orbit.manifest)
    assert {
        "gammadot",
        "accel_ra",
        "accel_dec",
        "accel_total",
        "mc_over_r2",
    } <= names
    assert not {"gammaddot", "jerk_ra", "mc_over_r2_min"} & names
    assert system.orbit.taylor_epoch[0] == system.astrometryinstrument.epoch
    assert system.orbit.taylor_rv_epoch[0] == pytest.approx(
        0.5 * (t_rv.min() + t_rv.max()), abs=1e-6
    )
    assert np.isfinite(model.compile_logp()(_start(system, model)))


def test_rvs_of_a_uniform_rotation_orbit_are_refused(tmp_path):
    """
    Given a uniform_rotation orbit (a lens pair's relative rotation) whose
      primary has RVs,
    When the system is prepared,
    Then it raises: the state has no line-of-sight component.
    """
    # Arrange
    _rv_file(tmp_path / "a.rv")
    config = {
        "name": "ur",
        "star": [{"name": "A", "mist": False}],
        "orbit": [
            {"name": "outer", "type": "uniform_rotation", "primary": ["A"]}
        ],
        "rvinstrument": [{"name": "RV", "file": str(tmp_path / "a.rv")}],
    }

    # Act / Assert
    with pytest.raises(NotImplementedError, match="line-of-sight"):
        System(config, user_params=_star_params(["A"])).prepare()


# ---------------------------------------------------------------------------
# 3. One acceleration vector, RVs and Gaia epochs together
# ---------------------------------------------------------------------------

# 1 m/s/day of acceleration, in AU/yr^2 (times the parallax in mas, the
# angular sky acceleration in mas/yr^2).
_MSD_AU_YR2 = float((1.0 * u.m / u.s / u.d).to(u.AU / u.yr**2).value)
_RV_ERR = 3.0  # m/s
_JOINT = dict(
    gammadot=-0.6,  # m/s/day, line of sight
    accel_E=0.45,  # m/s/day, sky East
    accel_N=-0.30,  # m/s/day, sky North
    gammaddot=0.0015,  # m/s/day^2 (jerk case)
    jerk_ra=0.15,  # mas/yr^3 (jerk case)
    jerk_dec=-0.10,
    gamma=120.0,
)


def _joint_truth(jerk):
    aE = _JOINT["accel_E"] * _MSD_AU_YR2 * _T["plx"]  # mas/yr^2
    aN = _JOINT["accel_N"] * _MSD_AU_YR2 * _T["plx"]
    jE = _JOINT["jerk_ra"] if jerk else 0.0
    jN = _JOINT["jerk_dec"] if jerk else 0.0
    g2 = _JOINT["gammaddot"] if jerk else 0.0
    return aE, aN, jE, jN, g2


def _write_joint(tmp, jerk, seed=11):
    """Gaia epochs (about _EPOCH, the astrometric epoch) and RVs whose
    midpoint is 500 d LATER, from one primary-group acceleration
    a(t) = a0 + j (t - _EPOCH): the RV is gamma + the integral of a_z from
    the RV midpoint, the sky track pm (t - E) + plx P + a0 dt^2/2 +
    j dt^3/6."""
    aE, aN, jE, jN, g2 = _joint_truth(jerk)
    rng = np.random.default_rng(seed)
    n = 60
    t = np.sort(rng.uniform(*_SPAN, n))
    psi = rng.uniform(0.0, 2.0 * np.pi, n)
    P_E, P_N = _pfac(t)
    dt = (t - _EPOCH) / YR
    dE = _T["pmra"] * dt + _T["plx"] * P_E + 0.5 * aE * dt**2 + jE * dt**3 / 6
    dN = _T["pmdec"] * dt + _T["plx"] * P_N + 0.5 * aN * dt**2 + jN * dt**3 / 6
    w = dE * np.sin(psi) + dN * np.cos(psi) + rng.normal(0.0, _ERR, n)
    np.savetxt(
        tmp / "g.astrom",
        np.column_stack([t, w, np.full(n, _ERR), np.degrees(psi)]),
    )
    t_rv = np.sort(rng.uniform(_EPOCH + 100.0, _EPOCH + 900.0, 40))
    t_rv[0], t_rv[-1] = _EPOCH + 100.0, _EPOCH + 900.0
    mid = 0.5 * (t_rv[0] + t_rv[-1])
    g1 = _JOINT["gammadot"]
    rv = (
        _JOINT["gamma"]
        + g1 * (t_rv - mid)
        + 0.5 * g2 * ((t_rv - _EPOCH) ** 2 - (mid - _EPOCH) ** 2)
        + rng.normal(0.0, _RV_ERR, t_rv.size)
    )
    np.savetxt(
        tmp / "a.rv", np.column_stack([t_rv, rv, np.full(t_rv.size, _RV_ERR)])
    )
    return t, psi, w, t_rv, rv


def _joint_system(tmp, jerk):
    t, psi, w, t_rv, rv = _write_joint(tmp, jerk)
    otype = "constant_jerk" if jerk else "constant_acceleration"
    config = {
        "name": "joint",
        "star": [{"name": "A", "mist": False}],
        "orbit": [{"name": "outer", "type": otype, "primary": ["A"]}],
        "astrometryinstrument": [_gaia_inst(tmp / "g.astrom", epoch=_EPOCH)],
        "rvinstrument": [{"name": "RV", "file": str(tmp / "a.rv")}],
    }
    up = _star_params(["A"])
    # Start every Taylor coefficient far from the truth.
    up.update(_accel_params(accel=_SEED))
    up["orbit.outer.gammadot"] = {"initval": 0.0}
    if jerk:
        up["orbit.outer.gammaddot"] = {"initval": 0.0}
        up["orbit.outer.jerk_ra"] = {"initval": 0.0}
        up["orbit.outer.jerk_dec"] = {"initval": 0.0}
    system, model = _build(config, up)
    return system, model, (t, psi, w, t_rv, rv)


def _gaia_lsq(t, psi, w, jerk):
    """Independent weighted least squares of the along-scan epochs on
    (dE0, dN0, plx, pmE, pmN, aE, aN[, jE, jN]); returns (coef, sigma)."""
    P_E, P_N = _pfac(t)
    dt = (t - _EPOCH) / YR
    s, c = np.sin(psi), np.cos(psi)
    cols = [
        s,
        c,
        P_E * s + P_N * c,
        dt * s,
        dt * c,
        0.5 * dt**2 * s,
        0.5 * dt**2 * c,
    ]
    if jerk:
        cols += [dt**3 / 6 * s, dt**3 / 6 * c]
    A = np.column_stack(cols) / _ERR
    coef, *_ = np.linalg.lstsq(A, w / _ERR, rcond=None)
    return coef, np.sqrt(np.diag(np.linalg.inv(A.T @ A)))


def _rv_lsq(t_rv, rv, jerk):
    """Independent least squares of the RVs on (gamma, slope at the RV
    midpoint[, gammaddot]); returns (coef, sigma)."""
    mid = 0.5 * (t_rv[0] + t_rv[-1])
    dt = t_rv - mid
    cols = [np.ones_like(dt), dt] + ([0.5 * dt**2] if jerk else [])
    A = np.column_stack(cols) / _RV_ERR
    coef, *_ = np.linalg.lstsq(A, rv / _RV_ERR, rcond=None)
    return coef, np.sqrt(np.diag(np.linalg.inv(A.T @ A))), mid


@pytest.fixture(scope="module", params=[False, True], ids=["accel", "jerk"])
def joint_fit(request, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("joint")
    jerk = request.param
    system, model, data = _joint_system(tmp, jerk)
    best = _max_likelihood(model, _start(system, model))
    return jerk, system, model, data, best


def test_one_acceleration_vector_is_recovered_from_rvs_and_gaia(joint_fit):
    """
    Given RVs and Gaia along-scan epochs simulated in numpy from ONE
      primary-group acceleration (line of sight -0.6 m/s/day, sky
      (0.45, -0.30) m/s/day at 20 mas; with a jerk, also gammaddot
      0.0015 m/s/day^2 and sky jerk (0.15, -0.10) mas/yr^3), the RV
      midpoint 500 d after the astrometric epoch,
    When one Taylor orbit read by both is fit by maximum likelihood from a
      start with every coefficient far off,
    Then gammadot, accel_ra/dec (and gammaddot, jerk_ra/dec) agree with
      independent numpy least-squares fits of the two data sets to 0.1
      sigma and with the injected values to 4 sigma -- with a jerk the RV
      fit's slope at the RV midpoint maps to the shared epoch as
      gammadot = slope - gammaddot (rv_epoch - epoch).
    """
    # Arrange
    jerk, system, model, (t, psi, w, t_rv, rv), best = joint_fit
    orbit = system.orbit
    to_msd = float((u.solRad / u.d**2).to(u.m / u.s / u.d))
    names = ["gammadot", "accel_ra", "accel_dec"] + (
        ["gammaddot", "jerk_ra", "jerk_dec"] if jerk else []
    )

    # Act
    vals = _at(model, [getattr(orbit, n).value for n in names], best)
    fit = {n: float(np.ravel(v)[0]) for n, v in zip(names, vals)}
    fit["gammadot"] *= to_msd
    if jerk:
        fit["gammaddot"] *= float((u.solRad / u.d**3).to(u.m / u.s / u.d**2))

    # Assert
    assert orbit.taylor_epoch[0] == _EPOCH
    g, gs = _gaia_lsq(t, psi, w, jerk)
    r, rs, mid = _rv_lsq(t_rv, rv, jerk)
    assert orbit.taylor_rv_epoch[0] == pytest.approx(mid, abs=1e-6)
    aE, aN, jE, jN, g2 = _joint_truth(jerk)
    expect = {
        "accel_ra": (g[5], gs[5], aE),
        "accel_dec": (g[6], gs[6], aN),
    }
    if jerk:
        expect["jerk_ra"] = (g[7], gs[7], jE)
        expect["jerk_dec"] = (g[8], gs[8], jN)
        expect["gammaddot"] = (r[2], rs[2], g2)
        slope0 = r[1] - r[2] * (mid - _EPOCH)
        sig0 = np.hypot(rs[1], rs[2] * (mid - _EPOCH))
        # The injected a_z(t) = gammadot + g2 (t - E) is quoted at the
        # shared epoch E; the RV fit's slope is a_z at the RV midpoint.
        expect["gammadot"] = (slope0, sig0, _JOINT["gammadot"])
    else:
        expect["gammadot"] = (r[1], rs[1], _JOINT["gammadot"])
    for n, (lsq, sig, truth) in expect.items():
        assert abs(fit[n] - lsq) < 0.1 * sig, (n, fit[n], lsq, sig)
        assert abs(fit[n] - truth) < 4.0 * sig, (n, fit[n], truth, sig)


def test_the_derived_acceleration_magnitude_and_mass_bound(joint_fit):
    """
    Given the joint fit,
    When accel_total and mc_over_r2 are evaluated at the maximum,
    Then |A| is the quadrature sum of gammadot and the sky components made
      physical through the parallax (1e-9, from the fitted values), within
      5% of the injected |A|; and mc_over_r2 = |A| / G in M_J/AU^2
      (astropy, 1e-9).
    """
    # Arrange
    jerk, system, model, _data, best = joint_fit
    orbit, star = system.orbit, system.star
    nodes = [
        orbit.accel_total.value,
        orbit.mc_over_r2.value,
        orbit.gammadot.value,
        orbit.accel_ra.value,
        orbit.accel_dec.value,
        star.parallax.value,
    ]

    # Act
    A, mc, g1, aE, aN, plx = (
        float(np.ravel(v)[0]) for v in _at(model, nodes, best)
    )

    # Assert
    to_msd = float((u.solRad / u.d**2).to(u.m / u.s / u.d))
    sky = 1.0 / (_MSD_AU_YR2 * plx)  # mas/yr^2 -> m/s/day
    expected = np.sqrt((g1 * to_msd) ** 2 + (aE * sky) ** 2 + (aN * sky) ** 2)
    np.testing.assert_allclose(A * to_msd, expected, rtol=1e-9)
    injected = np.sqrt(
        _JOINT["gammadot"] ** 2
        + _JOINT["accel_E"] ** 2
        + _JOINT["accel_N"] ** 2
    )
    np.testing.assert_allclose(A * to_msd, injected, rtol=0.05)
    G_mc = float(
        ((A * u.solRad / u.d**2) / const.G).to(u.jupiterMass / u.AU**2).value
    )
    np.testing.assert_allclose(mc, G_mc, rtol=1e-9)
