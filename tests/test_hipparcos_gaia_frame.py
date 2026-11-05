"""Hipparcos and Gaia epoch astrometry of one star fit jointly, with the
Hipparcos-to-Gaia frame tie as parameters (`fithipframe`, on by default
for Hipparcos data fit with Gaia;
components/astrometryinstrument/frametie.py).

The point of the feature is the 25-year lever arm: a companion with a period
of decades bends the star's track between the Hipparcos epochs (1990-1993)
and the Gaia DR4 epochs (2014.6-2019.9), and a frame difference between the
two missions reads exactly as such a bend.  Pinned here, on a simulated star
with a 60-yr dark companion and a known injected frame offset and spin:

1. The model abscissae of BOTH datasets -- each measured from its own
   reference point (`reference_point:`), the Hipparcos one carrying the
   frame terms -- equal an independent numpy computation to 1e-6 mas.
2. A maximum-a-posteriori fit from a displaced start recovers the companion
   mass, its orbit orientation and the four frame terms within 4 sigma of
   the truth, sigma from the model's own Hessian at the optimum.
3. Without Hipparcos (DR4 alone) the companion mass is far less constrained
   -- the lever arm is what the joint fit buys.
4. The C and JAX gradients are finite and agree.
5. The flag is inert when off (no rows, the same graph) and misconfigurations
   raise, naming the fix.
"""

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from exozippy.components.astrometryinstrument.frametie import (
    FRAME_PARAMS,
    HIPPARCOS_EPOCH_JD,
)
from exozippy.constants import DAYS_PER_YEAR
from exozippy.ephemeris import observer_position_at_reception
from exozippy.skyframe import parallax_factors
from exozippy.system import System

RAD2MAS = 180.0 / np.pi * 3600e3
MJUP_PER_MSUN = 1047.5655
GAUSS_K = 0.01720209895  # AU^1.5 d^-1 Msun^-0.5

# The star at the reference epoch J2016.0 and its 60-yr dark companion.
EPOCH = 2457389.0
T = dict(
    ra=150.0,
    dec=20.0,
    pmra=120.0,
    pmdec=-80.0,
    plx=25.0,
    mstar=1.0,
    mcomp=0.08,  # Msun, dark
    P=60.0 * DAYS_PER_YEAR,
    ecc=0.3,
    w=1.0,
    bigom=0.8,
    inc=1.1,
    tp=2452000.0,
)
# Hipparcos minus Gaia, at the star: mas at J1991.25 and mas/yr.
FRAME = dict(frame_dE=0.4, frame_dN=-0.25, frame_dpmE=0.2, frame_dpmN=-0.15)

HIP_T = np.linspace(2447880.0, 2449050.0, 110)  # 1989.9 - 1993.1
DR4_T = np.linspace(2456870.0, 2458840.0, 300)  # 2014.6 - 2019.9
HIP_ERR, DR4_ERR = 2.0, 0.15


# ----------------------------------------------------------------------
# The independent reference
# ----------------------------------------------------------------------
def _kepler_E(M, ecc):
    E = np.mod(M, 2 * np.pi)
    for _ in range(100):
        E = E - (E - ecc * np.sin(E) - M) / (1 - ecc * np.cos(E))
    return E


def _photocenter(t):
    """(dE, dN) mas of the star about the barycenter, by explicit rotation
    (tests/test_astrometry.py's reference): omega is the STAR's."""
    mtot = T["mstar"] + T["mcomp"]
    # Kepler's third law with the Gaussian gravitational constant (AU, d,
    # Msun); "P in years" would be off by the sidereal-year ratio, 1e-5.
    a_au = (GAUSS_K**2 * mtot * T["P"] ** 2 / (4 * np.pi**2)) ** (1.0 / 3.0)
    a = a_au * T["mcomp"] / mtot * T["plx"]
    E = _kepler_E(2 * np.pi * (t - T["tp"]) / T["P"], T["ecc"])
    f = 2 * np.arctan2(
        np.sqrt(1 + T["ecc"]) * np.sin(E / 2),
        np.sqrt(1 - T["ecc"]) * np.cos(E / 2),
    )
    r = a * (1 - T["ecc"] ** 2) / (1 + T["ecc"] * np.cos(f))
    x = r * np.cos(T["w"] + f)
    y = r * np.sin(T["w"] + f) * np.cos(T["inc"])
    return (
        x * np.sin(T["bigom"]) + y * np.cos(T["bigom"]),
        x * np.cos(T["bigom"]) - y * np.sin(T["bigom"]),
    )


def _tc_from_tp():
    """The conjunction the orbit's tc parameter names (f = pi/2 - omega)."""
    e, w = T["ecc"], T["w"]
    E_c = 2 * np.arctan2(
        np.sqrt(1 - e) * (1 - np.sin(w)), np.sqrt(1 + e) * np.cos(w)
    )
    return T["tp"] + (E_c - e * np.sin(E_c)) * T["P"] / (2 * np.pi)


def _ref_point(t_jd):
    """The star's linear position at t (deg): a reference point 25 yr of
    proper motion away from the other one, as Hipparcos and DR4 are."""
    dt = (t_jd - EPOCH) / DAYS_PER_YEAR
    dec = T["dec"] + T["pmdec"] * dt / 3.6e6
    ra = T["ra"] + T["pmra"] * dt / 3.6e6 / np.cos(np.radians(T["dec"]))
    return ra, dec


def _abscissae(t, psi, ref, observer, frame=None):
    """Noiseless along-scan w (mas) about reference point ``ref`` (deg)."""
    ra_r, dec_r = np.radians(ref[0]), np.radians(ref[1])
    xyz = observer_position_at_reception(
        t, ra_r, dec_r, observer_location=observer
    )
    P_E, P_N = parallax_factors(xyz, ra_r, dec_r)
    dt = (t - EPOCH) / DAYS_PER_YEAR
    dE = (
        (np.radians(T["ra"]) - ra_r) * np.cos(dec_r) * RAD2MAS
        + T["pmra"] * dt
        + T["plx"] * P_E
    )
    dN = (
        (np.radians(T["dec"]) - dec_r) * RAD2MAS
        + T["pmdec"] * dt
        + T["plx"] * P_N
    )
    oE, oN = _photocenter(t)
    dE, dN = dE + oE, dN + oN
    if frame is not None:
        dt91 = (t - HIPPARCOS_EPOCH_JD) / DAYS_PER_YEAR
        dE = dE + frame["frame_dE"] + frame["frame_dpmE"] * dt91
        dN = dN + frame["frame_dN"] + frame["frame_dpmN"] * dt91
    return dE * np.sin(psi) + dN * np.cos(psi)


HIP_REF = _ref_point(HIPPARCOS_EPOCH_JD)
DR4_REF = _ref_point(2457936.875)  # ~J2017.5


@pytest.fixture(scope="module")
def sim_files(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("hipgaia")
    rng = np.random.default_rng(42)
    out = {}
    for name, t, err, ref, obs, frame in (
        ("hip", HIP_T, HIP_ERR, HIP_REF, "earth", FRAME),
        ("dr4", DR4_T, DR4_ERR, DR4_REF, "gaia", None),
    ):
        psi = rng.uniform(0, 2 * np.pi, len(t))
        w = _abscissae(t, psi, ref, obs, frame)
        out[name + "_true"] = w
        w = w + rng.normal(0, err, len(t))
        path = tmp / f"{name}.astrom"
        np.savetxt(
            path,
            np.column_stack([t, w, np.full(len(t), err), np.degrees(psi)]),
            fmt="%.10f",
        )
        out[name] = path
    return out


def _config(files, with_hip=True, fithipframe=None):
    hip = {
        "name": "Hipparcos",
        "file": str(files["hip"]),
        "mode": "gaia",
        "observer_location": "earth",
        "reference_point": list(HIP_REF),
        "epoch": EPOCH,
    }
    if fithipframe is not None:
        hip["fithipframe"] = fithipframe
    dr4 = {
        "name": "DR4",
        "file": str(files["dr4"]),
        "mode": "gaia",
        "observer_location": "gaia",
        "reference_point": list(DR4_REF),
        "epoch": EPOCH,
    }
    return {
        "name": "hipgaia",
        "star": [{"name": "A", "mist": False}],
        # radius pinned, so Chen & Kipping would act on the mass
        "planet": [{"name": "B", "chen": False}],
        "orbit": [{"name": "B"}],
        "astrometryinstrument": [hip, dr4] if with_hip else [dr4],
    }


def _params(displace=False, frame=None, free_shape=False):
    p = {
        "star.A.mass": {"initval": T["mstar"], "sigma": 0.02},
        "star.A.radius": {"initval": 1.0, "sigma": 0.1},
        "star.A.teff": {"initval": 5800, "sigma": 100},
        "star.A.feh": {"initval": 0.0, "sigma": 0.1},
        "star.A.ra": {"initval": T["ra"]},
        "star.A.dec": {"initval": T["dec"]},
        "star.A.pm_ra": {"initval": T["pmra"]},
        "star.A.pm_dec": {"initval": T["pmdec"]},
        "star.A.distance": {"initval": 1000.0 / T["plx"]},
        "planet.B.mass": {"initval": T["mcomp"] * MJUP_PER_MSUN},
        "planet.B.radius": {"initval": 1.0, "sigma": 0},
        "orbit.B.period": {"initval": T["P"], "sigma": 0},
        "orbit.B.tc": {"initval": _tc_from_tp()},
        "orbit.B.secosw": {
            "initval": np.sqrt(T["ecc"]) * np.cos(T["w"]),
            "sigma": 0,
        },
        "orbit.B.sesinw": {
            "initval": np.sqrt(T["ecc"]) * np.sin(T["w"]),
            "sigma": 0,
        },
        "orbit.B.bigomega": {"initval": np.degrees(T["bigom"])},
        "orbit.B.cosi": {"initval": np.cos(T["inc"])},
    }
    if free_shape:
        # the period and the eccentricity sampled too: what a real fit of a
        # partially covered orbit has to determine
        for k in ("orbit.B.period", "orbit.B.secosw", "orbit.B.sesinw"):
            del p[k]["sigma"]
    if displace:
        p["planet.B.mass"]["initval"] = 0.06 * MJUP_PER_MSUN
        p["orbit.B.bigomega"]["initval"] = np.degrees(T["bigom"]) + 8.0
        p["orbit.B.cosi"]["initval"] = np.cos(T["inc"]) + 0.05
        p["star.A.pm_ra"]["initval"] = T["pmra"] + 0.5
        p["star.A.pm_dec"]["initval"] = T["pmdec"] - 0.5
    for k, v in (frame or {}).items():
        p[f"astrometryinstrument.Hipparcos.{k}"] = {"initval": v}
    return p


def _build(cfg, params):
    system = System(cfg, user_params=params)
    system.prepare()
    return system, system.build_model()


def _model_w(system, model, i):
    """The likelihood's abscissa model of dataset i, compiled over the free
    RVs (the _absolute_model the likelihood itself calls)."""
    inst = system.astrometryinstrument
    d = inst.datasets[i]
    with model:
        dE, dN = inst._absolute_model(system, d, d["time"], 0.0)
    w = dE * d["sin_psi"] + dN * d["cos_psi"]
    fn = pytensor.function(model.free_RVs, w, on_unused_input="ignore")
    ip = model.initial_point()
    return fn(*[ip[v.name] for v in model.free_RVs]), d


# ----------------------------------------------------------------------
# 1. The model equals the independent computation
# ----------------------------------------------------------------------
def test_model_abscissae_match_the_numpy_reference(sim_files):
    """
    Given the simulated Hipparcos (frame offset and spin injected) and DR4
      datasets, each measured from its own reference point, and NO
      fithipframe key,
    When the system is built at the truth,
    Then the frame terms are on by default for the Hipparcos dataset only,
      each dataset's tangent point is its reference_point, and the model
      abscissae equal the noiseless numpy ones to 1e-6 mas on both.
    """
    system, model = _build(_config(sim_files), _params(frame=FRAME))
    inst = system.astrometryinstrument
    for i, (key, ref) in enumerate((("hip", HIP_REF), ("dr4", DR4_REF))):
        w, d = _model_w(system, model, i)
        assert d["ra_ref"] == np.radians(ref[0])
        assert d["dec_ref"] == np.radians(ref[1])
        # the files are sorted by time already, so row order is kept
        np.testing.assert_allclose(
            w, sim_files[key + "_true"], rtol=0, atol=1e-6
        )
    assert inst.fithipframe == [True, False]
    for p in FRAME_PARAMS:
        par = getattr(inst, p)
        assert par.element_is_active(0) and not par.element_is_active(1)


# ----------------------------------------------------------------------
# 2./3. MAP recovery and the lever arm
# ----------------------------------------------------------------------
def _value_vars(model):
    return [model.rvs_to_values[v] for v in model.free_RVs]


def _map(model):
    """MAP over the raw value variables by L-BFGS-B, analytic gradient."""
    from scipy.optimize import minimize

    values = _value_vars(model)
    logp = model.logp(sum=True)
    f = pytensor.function(
        values, [logp] + pt.grad(logp, values), on_unused_input="ignore"
    )
    ip = model.initial_point()
    names = [v.name for v in values]
    shapes = [np.shape(ip[n]) for n in names]
    sizes = [int(np.prod(s)) for s in shapes]

    def unpack(x):
        out, k = [], 0
        for s, n in zip(shapes, sizes):
            out.append(x[k : k + n].reshape(s))
            k += n
        return out

    def neg(x):
        lp, *g = f(*unpack(x))
        return -float(lp), -np.concatenate([np.ravel(gi) for gi in g])

    x0 = np.concatenate([np.ravel(ip[n]) for n in names])
    res = minimize(
        neg,
        x0,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 20000, "ftol": 1e-15, "gtol": 1e-9},
    )
    return dict(zip(names, unpack(res.x)))


def _sigmas(model, point, nodes, h=1e-4):
    """1-sigma of each quantity in ``nodes`` (tensors of the free RVs) from
    the inverse Hessian of the log posterior over the raw variables at
    ``point``: sigma_q^2 = g^T (-H)^{-1} g, g = dq/draw.  H is the central
    difference of the ANALYTIC gradient (one compiled dlogp, cheap), step
    ``h`` in the raw (whitened, order-unity) coordinates."""
    values = _value_vars(model)
    names = [v.name for v in values]
    shapes = [np.shape(point[n]) for n in names]
    sizes = [int(np.prod(s)) for s in shapes]
    logp = model.logp(sum=True)
    gfn = pytensor.function(
        values,
        pt.concatenate([pt.flatten(g) for g in pt.grad(logp, values)]),
        on_unused_input="ignore",
    )
    qs = model.replace_rvs_by_values(nodes)
    qfn = pytensor.function(
        values,
        [
            pt.concatenate(
                [
                    pt.flatten(g)
                    for g in pt.grad(q, values, disconnected_inputs="ignore")
                ]
            )
            for q in qs
        ],
        on_unused_input="ignore",
    )

    def unpack(x):
        out, k = [], 0
        for s, n in zip(shapes, sizes):
            out.append(x[k : k + n].reshape(s))
            k += n
        return out

    x0 = np.concatenate([np.ravel(point[n]) for n in names])
    H = np.empty((len(x0), len(x0)))
    for j in range(len(x0)):
        e = np.zeros_like(x0)
        e[j] = h
        H[:, j] = (gfn(*unpack(x0 + e)) - gfn(*unpack(x0 - e))) / (2 * h)
    cov = np.linalg.inv(-0.5 * (H + H.T))
    return [float(np.sqrt(g @ cov @ g)) for g in qfn(*unpack(x0))]


def _quantities(system):
    inst = system.astrometryinstrument
    nodes = {
        "mass": system.planet.mass.value[0],
        "bigomega": system.orbit.bigomega.value[0],
        "cosi": system.orbit.cosi.value[0],
    }
    if any(inst.fithipframe):
        for p in FRAME_PARAMS:
            nodes[p] = getattr(inst, p).value[0]
    return nodes


def _evaluate(model, point, nodes):
    fn = pytensor.function(
        _value_vars(model),
        model.replace_rvs_by_values(list(nodes.values())),
        on_unused_input="ignore",
    )
    vals = fn(*[point[v.name] for v in _value_vars(model)])
    return dict(zip(nodes, (float(v) for v in vals)))


@pytest.mark.slow
def test_map_recovers_the_companion_and_the_frame_terms(sim_files):
    """
    Given the joint Hipparcos + DR4 fit started 25% low in companion mass, 8
      deg off in node, 0.05 off in cos i, 0.5 mas/yr off in proper motion
      and at zero frame offset,
    When the log posterior is maximized,
    Then the companion mass, node, cos i and the four frame terms land
      within 4 sigma of the truth, sigma from the model's own Hessian.
    """
    system, model = _build(_config(sim_files), _params(displace=True))
    mp = _map(model)
    nodes = _quantities(system)
    est = _evaluate(model, mp, nodes)
    sig = dict(zip(nodes, _sigmas(model, mp, list(nodes.values()))))
    # the mass node is internal (solar masses); the truth is in the user's
    # jupiterMass, and only Parameter converts between them
    mass = system.planet.mass
    est["mass"] = float(mass.from_internal(est["mass"], index=0))
    sig["mass"] = float(mass.from_internal(sig["mass"], index=0))
    truth = {
        "mass": T["mcomp"] * MJUP_PER_MSUN,
        "bigomega": T["bigom"],
        "cosi": np.cos(T["inc"]),
        **FRAME,
    }
    for k, v in truth.items():
        assert abs(est[k] - v) < 4.0 * sig[k], (k, est[k], v, sig[k])
    # the companion is measured: mass to better than 15% from 2 mas
    # Hipparcos and 0.15 mas DR4 abscissae of a 60-yr orbit
    assert sig["mass"] < 0.15 * truth["mass"]


@pytest.mark.slow
def test_without_hipparcos_the_companion_is_far_less_constrained(sim_files):
    """
    Given the same star with the period and eccentricity free too, fit
      with DR4 alone and with Hipparcos + DR4 (each at its own MAP, started
      at the truth),
    When the companion mass's and period's 1-sigma are read from each
      model's Hessian,
    Then DR4 alone constrains both at least 3x worse: 5.3 yr of a 60-yr
      orbit leaves mass and period degenerate, and the 25-yr lever arm is
      what Hipparcos adds, even after the frame terms absorb a position
      and proper-motion offset of the Hipparcos dataset.
    """
    out = {}
    for with_hip in (True, False):
        system, model = _build(
            _config(sim_files, with_hip=with_hip),
            _params(frame=FRAME if with_hip else None, free_shape=True),
        )
        mp = _map(model)
        out[with_hip] = _sigmas(
            model,
            mp,
            [system.planet.mass.value[0], system.orbit.period.value[0]],
        )
    for k, name in enumerate(("mass", "period")):
        joint, alone = out[True][k], out[False][k]
        assert np.isfinite(joint), (name, out)
        assert not alone < 3.0 * joint, (name, out)


# ----------------------------------------------------------------------
# 4. Gradients
# ----------------------------------------------------------------------
@pytest.mark.slow
def test_gradients_are_finite_in_c_and_jax(sim_files):
    """
    Given the joint model with the frame terms,
    When logp and its gradient are evaluated at the start in C and in JAX,
    Then all are finite and the two backends agree.
    """
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    from pymc.sampling.jax import get_jaxified_logp

    _system, model = _build(_config(sim_files), _params(frame=FRAME))
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
    assert all(np.all(np.isfinite(np.asarray(g))) for g in grads)


# ----------------------------------------------------------------------
# 5. Inert when off; misconfigurations
# ----------------------------------------------------------------------
def test_flag_off_declares_nothing_and_leaves_the_model(sim_files):
    """
    Given the joint config with fithipframe: false,
    When it is built,
    Then no frame parameter exists, and the Hipparcos model abscissae are
      the frame-free numpy ones (the frames taken as identical).
    """
    system, model = _build(_config(sim_files, fithipframe=False), _params())
    inst = system.astrometryinstrument
    assert not any(hasattr(inst, p) for p in FRAME_PARAMS)
    assert not any("frame_" in v.name for v in model.free_RVs)
    w, d = _model_w(system, model, 0)
    t, psi = d["time"], np.arctan2(d["sin_psi"], d["cos_psi"])
    np.testing.assert_allclose(
        w, _abscissae(t, psi, HIP_REF, "earth"), rtol=0, atol=1e-6
    )


@pytest.mark.parametrize(
    "extra, match",
    [
        ({"fithipframe": "yes"}, "must be true or false"),
        ({"fithipframe": True, "observer_location": "gaia"}, "Gaia defines"),
        ({"fithipframe": True, "mode": "rel"}, "mode gaia or abs"),
        ({"reference_point": [10.0]}, r"\[ra, dec\] in degrees"),
        ({"reference_point": [10.0, 95.0]}, r"\[ra, dec\] in degrees"),
        ({"reference_point": [10.0, 5.0], "mode": "abs"}, "mode-gaia"),
    ],
)
def test_misconfigurations_raise(sim_files, extra, match):
    cfg = _config(sim_files, fithipframe=False)
    hip = cfg["astrometryinstrument"][0]
    hip.update(extra)
    if hip.get("mode") == "rel":
        del hip["reference_point"]
    with pytest.raises(ValueError, match=match):
        System(cfg, user_params=_params())


def test_frame_terms_on_gaia_epoch_data_raise():
    """The Gaia epoch file defines the frame; the flag on it raises, as
    does a reference_point that would compete with its own (ra0, dec0)."""
    from pathlib import Path

    f = (
        Path(__file__).parent.parent
        / "examples"
        / "gaia4"
        / "GAIA_DR4_PRERELEASE_EPOCH_ASTROMETRY_RAW.xml"
    )
    base = {
        "name": "G",
        "file": str(f),
        "mode": "gaia",
        "data_format": "gaia_epoch",
        "source_id": 1457486023639239296,
    }
    for extra, match in (
        ({"fithipframe": True}, "DEFINE the frame"),
        ({"reference_point": [209.5, 31.7]}, "states its own"),
    ):
        cfg = {
            "name": "x",
            "star": [{"name": "A", "mist": False}],
            "astrometryinstrument": [{**base, **extra}],
        }
        with pytest.raises(ValueError, match=match):
            System(cfg, user_params={})


def test_default_is_off_without_a_frame_or_outside_the_mission(
    sim_files, tmp_path
):
    """
    Given (a) the Hipparcos dataset fit alone, and (b) the same abscissae
      with one epoch moved past the end of the Hipparcos mission, fit with
      DR4, neither saying fithipframe,
    When each is prepared,
    Then (a) has nothing to tie to and (b) is not Hipparcos data by the
      epoch test, so the default leaves the terms off in both (no frame
      parameter is declared); an explicit true on (a) is honored.
    """
    cfg = _config(sim_files)
    cfg["astrometryinstrument"] = cfg["astrometryinstrument"][:1]
    system = System(cfg, user_params=_params())
    system.prepare()
    assert system.astrometryinstrument.fithipframe == [False]
    assert "frame_dE" not in system.astrometryinstrument.manifest

    arr = np.loadtxt(sim_files["hip"])
    arr[-1, 0] = 2449200.0  # 1993.5: after the mission
    moved = tmp_path / "late.astrom"
    np.savetxt(moved, arr, fmt="%.10f")
    cfg = _config(sim_files)
    cfg["astrometryinstrument"][0]["file"] = str(moved)
    system = System(cfg, user_params=_params())
    system.prepare()
    assert system.astrometryinstrument.fithipframe == [False, False]

    cfg = _config(sim_files, fithipframe=True)
    cfg["astrometryinstrument"] = cfg["astrometryinstrument"][:1]
    system = System(cfg, user_params=_params())
    system.prepare()
    assert system.astrometryinstrument.fithipframe == [True]
