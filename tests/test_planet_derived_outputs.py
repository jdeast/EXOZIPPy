"""
The planet's derived outputs ported from EXOFASTv2's derivepars.pro in
2026-09: teq, fave, msini, q, the four a priori transit/eclipse
probabilities pt/ptg/ps/psg, delta, tcirc and omegagr.

    T_eq = T_eff sqrt(R_*/(2a))
    <F>  = sigma_sb T_eff^4 / (a/R_* (1 + e^2/2))^2
    P    = (1 +/- p)/(a/R_*) (1 +/- e sin omega)/(1 - e^2)   (Winn 2010 eq 9)

EXOFASTv2 reports <F> in 10^9 erg s-1 cm-2; EXOZIPPy reports it in units
of Earth's insolation (1361 W m-2).  1e9 erg s-1 cm-2 is 1e6 W m-2, so the
two differ by 1e6/1361, NOT 1e9/1361 -- the Earth-around-the-Sun pin below
is what makes that factor unmistakable: it has to come out at 1.
"""

import astropy.units as u
import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from exozippy.components.planet.physics import (
    calc_fave,
    calc_omegagr,
    calc_tcirc,
    calc_teq,
)
from exozippy.constants import EARTH_INSOLATION_CGS, RSUN_TO_AU, SIGMA_SB_CGS
from exozippy.system import System

TEFF_SUN = 5772.0  # K, IAU 2015 nominal
AR_EARTH = 215.032  # au / R_sun
ECC = 0.3
OMEGA = np.radians(60.0)


def _physics_fn():
    teff, ar, ecc = pt.dscalar("teff"), pt.dscalar("ar"), pt.dscalar("ecc")
    return pytensor.function(
        [teff, ar, ecc], [calc_teq(teff, ar), calc_fave(teff, ar, ecc)]
    )


def test_earth_around_the_sun_is_one_earth_insolation():
    """
    Given the Sun's T_eff and Earth's a/R_sun on a circular orbit,
    When calc_fave and calc_teq are evaluated,
    Then <F> is 1 Earth insolation (to the 1e-4 the 1361 W m-2 nominal
      value differs from sigma_sb T_sun^4 (R_sun/au)^2) and T_eq is the
      textbook 278 K -- a pin on the UNIT, which a 1e9/1361 or 1e6 slip
      would miss by orders of magnitude.
    """
    teq, fave = _physics_fn()(TEFF_SUN, AR_EARTH, 0.0)
    assert float(fave) == pytest.approx(1.0, rel=1e-3)
    assert float(teq) == pytest.approx(278.33, rel=1e-4)


def test_fave_matches_exofastv2_formula_and_unit():
    """
    Given an eccentric hot Jupiter,
    When calc_fave is evaluated,
    Then it equals derivepars.pro's
      sigmab teff^4 / (ar (1 + e^2/2))^2 / 1d9  [10^9 erg s-1 cm-2]
      rescaled by 1e6/1361 to Earth units (EXOFASTv2's sigmab is CODATA
      2014, ours CODATA 2018; they agree to 1.3e-6).
    """
    teff, ar, ecc = 6200.0, 7.5, ECC
    _, fave = _physics_fn()(teff, ar, ecc)
    exofast_1e9_cgs = (
        5.670367e-5 * teff**4 / (ar * (1.0 + ecc**2 / 2.0)) ** 2 / 1e9
    )
    assert float(fave) == pytest.approx(
        exofast_1e9_cgs * 1e6 / 1361.0, rel=1e-5
    )
    # And the two constants really are what the docstring claims.
    assert SIGMA_SB_CGS == pytest.approx(5.670374e-5, rel=1e-6)
    assert EARTH_INSOLATION_CGS == pytest.approx(1.361e6, rel=1e-12)


def test_fave_eccentricity_factor_is_second_order_expansion():
    """
    Given a fixed T_eff and a/R_*,
    When e goes from 0 to 0.3,
    Then <F> drops by exactly (1 + e^2/2)^-2 -- EXOFASTv2's second-order
      expansion of the exact <(a/r)^2> = (1-e^2)^-1/2, kept so the two
      codes report the same number -- while T_eq, evaluated at the
      semi-major axis, does not move.
    """
    fn = _physics_fn()
    teq0, fave0 = fn(TEFF_SUN, 10.0, 0.0)
    teq1, fave1 = fn(TEFF_SUN, 10.0, ECC)
    assert float(fave1 / fave0) == pytest.approx(
        (1.0 + ECC**2 / 2.0) ** -2, rel=1e-12
    )
    assert float(teq1) == float(teq0)


def _params():
    return {
        "star.0.radius": {"initval": 1.0, "sigma": 0.05},
        "star.0.mass": {"initval": 1.0, "sigma": 0.05},
        "star.0.teff": {"initval": TEFF_SUN, "sigma": 100},
        "star.0.feh": {"initval": 0.0, "sigma": 0.08},
        "orbit.0.period": {"initval": 365.25},
        "orbit.0.tc": {"initval": 2459634.3},
        "orbit.0.cosi": {"initval": 0.05},
        "orbit.0.secosw": {"initval": np.sqrt(ECC) * np.cos(OMEGA)},
        "orbit.0.sesinw": {"initval": np.sqrt(ECC) * np.sin(OMEGA)},
        "planet.0.radius": {"initval": 1.0},
        "planet.0.mass": {"initval": 0.003},
    }


@pytest.fixture(scope="module")
def earth_system():
    """A star + planet + orbit System (no data) with an Earth-like orbit
    around a solar-twin host, its model, and (teff, ar, ecc, teq, fave)
    at the initial point."""
    config = {
        "star": [{"name": "A", "mist": False}],
        "planet": [{"name": "b"}],
        "orbit": [{"name": "b"}],
    }
    system = System(config, user_params=_params())
    system.prepare()
    model = system.build_model()
    ip = model.initial_point()
    givens = [
        (rv, np.asarray(ip[rv.name])) for rv in model.free_RVs if rv.name in ip
    ]
    names = (
        "teff",
        "ar",
        "ecc",
        "teq",
        "fave",
        "star_mass",
        "mass",
        "sini",
        "esinw",
        "p",
        "msini",
        "q",
        "pt",
        "ptg",
        "ps",
        "psg",
        "a",
        "period",
        "delta",
        "tcirc",
        "omegagr",
    )
    fn = pytensor.function(
        [],
        [
            system.star.teff.value[0],
            system.planet.ar.value[0],
            system.orbit.ecc.value[0],
            system.planet.teq.value[0],
            system.planet.fave.value[0],
            system.star.mass.value[0],
            system.planet.mass.value[0],
            system.orbit.sini.value[0],
            system.orbit.esinw.value[0],
            system.planet.p.value[0],
            system.planet.msini.value[0],
            system.planet.q.value[0],
            system.planet.pt.value[0],
            system.planet.ptg.value[0],
            system.planet.ps.value[0],
            system.planet.psg.value[0],
            system.planet.a.value[0],
            system.orbit.period.value[0],
            system.planet.delta.value[0],
            system.planet.tcirc.value[0],
            system.planet.omegagr.value[0],
        ],
        givens=givens,
        on_unused_input="ignore",
        mode="FAST_COMPILE",
    )
    return system, dict(zip(names, map(float, fn())))


def test_system_teq_fave_follow_the_model_geometry(earth_system):
    """
    Given a built System whose orbit has e = 0.3 at 1 au,
    When planet.teq and planet.fave are read from the model graph,
    Then they are the formulas evaluated on the MODEL's own teff, a/R_*
      and e (so star.teff really is routed to the planet through
      star_map, and orbit.ecc through orbit_map), and fave sits at
      (1 + e^2/2)^-2 of an Earth.
    """
    _, v = earth_system
    assert v["ar"] == pytest.approx(AR_EARTH, rel=1e-3)
    assert v["ecc"] == pytest.approx(ECC, rel=1e-9)
    assert v["teq"] == pytest.approx(
        v["teff"] * np.sqrt(0.5 / v["ar"]), rel=1e-10
    )
    expected = (
        SIGMA_SB_CGS
        * v["teff"] ** 4
        / (v["ar"] * (1.0 + v["ecc"] ** 2 / 2.0)) ** 2
        / EARTH_INSOLATION_CGS
    )
    assert v["fave"] == pytest.approx(expected, rel=1e-10)
    assert v["fave"] == pytest.approx((1.0 + ECC**2 / 2.0) ** -2, rel=1e-3)


def test_msini_and_q_follow_the_model_masses(earth_system):
    """
    Given the built System (a log_q-mode planet, since no data measure
      its orbit),
    When planet.msini and planet.q are read from the model graph,
    Then msini is the model's own mass times sin i (both internal solMass,
      so no unit factor intervenes) and q is mass over the host's mass --
      through star_map, as calc_mass_from_log_q's star.mass dep is.
    """
    _, v = earth_system
    assert v["sini"] < 1.0  # cosi = 0.05: sin i really enters
    assert v["msini"] == pytest.approx(v["mass"] * v["sini"], rel=1e-12)
    assert v["q"] == pytest.approx(v["mass"] / v["star_mass"], rel=1e-12)
    # The params-file start is 0.003 in planet.mass's USER unit (Jupiter
    # masses) around a 1 solMass host, so q is 0.003 M_J / M_sun.
    mjup_per_msun = (1.0 * u.M_jup).to(u.M_sun).value
    assert v["q"] == pytest.approx(0.003 * mjup_per_msun, rel=1e-6)


def _winn_eq9(v, secondary, grazing):
    sign = -1.0 if secondary else 1.0
    edge = 1.0 if grazing else -1.0
    return (
        (1.0 + edge * v["p"])
        / v["ar"]
        * (1.0 + sign * v["esinw"])
        / (1.0 - v["ecc"] ** 2)
    )


def test_transit_eclipse_probabilities_match_winn2010_eq9(earth_system):
    """
    Given the built System with e = 0.3 and omega = 60 deg (esinw ~ 0.26),
    When pt/ptg/ps/psg are read from the model graph,
    Then each is Winn 2010 eq 9 on the model's own p, a/R_*, e and esinw,
      with the transit pair carrying 1 + esinw and the eclipse pair
      1 - esinw -- and the ordering that geometry fixes holds: grazing
      >= full, and (esinw > 0, planet nearer at transit) transit > eclipse.
    """
    _, v = earth_system
    assert v["esinw"] > 0.2
    assert v["ptg"] == pytest.approx(_winn_eq9(v, False, True), rel=1e-10)
    assert v["pt"] == pytest.approx(_winn_eq9(v, False, False), rel=1e-10)
    assert v["psg"] == pytest.approx(_winn_eq9(v, True, True), rel=1e-10)
    assert v["ps"] == pytest.approx(_winn_eq9(v, True, False), rel=1e-10)
    assert v["pt"] < v["ptg"]
    assert v["ps"] < v["psg"]
    assert v["ptg"] > v["psg"]
    assert v["pt"] > v["ps"]


def test_mercury_precesses_43_arcsec_per_century():
    """
    Given Mercury's orbit around the Sun (a = 0.387 au, P = 87.969 d,
      e = 0.2056),
    When calc_omegagr is evaluated,
    Then it is Einstein's 43 arcsec per century -- in the internal rad/day,
      so this also pins the unit the defaults.yaml conversion starts from --
      and the declared user unit `deg/century` parses, converts and renders,
      because exozippy/units.py registers `century` with astropy (it ships
      none), which is what lets the table carry EXOFASTv2's spelling.
    """
    from exozippy.components.parameter import UnitTranslator
    from exozippy.config import unit_conversion

    per_century = unit_conversion("rad/d", "deg/century", where="test")
    assert per_century == pytest.approx(np.degrees(1.0) * 36525.0)
    assert "century" in UnitTranslator.get_latex(u.Unit("deg/century"))
    a, m, p, e = (
        pt.dscalar("a"),
        pt.dscalar("m"),
        pt.dscalar("p"),
        pt.dscalar("e"),
    )
    fn = pytensor.function([a, m, p, e], calc_omegagr(a, m, p, e))
    rad_per_day = float(fn(0.387098 / RSUN_TO_AU, 1.0, 87.969, 0.205630))
    arcsec_per_century = np.degrees(rad_per_day) * 3600.0 * 36525.0
    assert arcsec_per_century == pytest.approx(42.98, rel=2e-3)


def test_tcirc_matches_exofastv2_cgs_evaluation():
    """
    Given a hot Jupiter (1 M_J, 1 R_J at 0.05 au around the Sun, e = 0.1),
    When calc_tcirc is evaluated in the internal solar units,
    Then it equals derivepars.pro's cgs evaluation of Adams & Laughlin
      2006 eq 2 (Q_P = 1e6), in Gyr, to the precision of the constants --
      and lands at the ~1.5 Gyr the paper quotes for such an orbit.
    """
    import astropy.constants as const
    import astropy.units as u

    a_au, mstar, mp, rp, e = 0.05, 1.0, 1.0, 1.0, 0.1
    mp_sun = (mp * u.M_jup).to(u.M_sun).value
    rp_sun = (rp * u.R_jup).to(u.R_sun).value
    a_sun = a_au / RSUN_TO_AU
    ar = a_sun / 1.0
    p = rp_sun / 1.0

    sym = [pt.dscalar(n) for n in ("a", "ar", "p", "mass", "star_mass", "ecc")]
    fn = pytensor.function(sym, calc_tcirc(*sym))
    ours = float(fn(a_sun, ar, p, mp_sun, mstar, e))

    # derivepars.pro, cgs: 4 Qp/63 / (day * 365.25e9) * sqrt((a au)^3 /
    # (GMsun mstar)) * (mpsun/mstar) * (ar/p)^5 * (1-e^2)^(13/2) / (1+6e^2)
    au_cm = const.au.cgs.value
    gmsun = const.GM_sun.cgs.value
    exofast = (
        4.0e6
        / 63.0
        / (86400.0 * 365.25e9)
        * np.sqrt((a_au * au_cm) ** 3 / (gmsun * mstar))
        * (mp_sun / mstar)
        * (ar / p) ** 5
        * (1.0 - e**2) ** 6.5
        / (1.0 + 6.0 * e**2)
    )
    assert ours == pytest.approx(exofast, rel=1e-6)
    assert 0.5 < ours < 3.0


def test_delta_tcirc_omegagr_follow_the_model(earth_system):
    """
    Given the built System,
    When delta, tcirc and omegagr are read from the model graph,
    Then delta is p^2, and the other two are their formulas on the model's
      own a, masses, period and e (internal units: a in solRad, rad/day).
    """
    _, v = earth_system
    assert v["delta"] == pytest.approx(v["p"] ** 2, rel=1e-12)
    from exozippy.constants import C_LIGHT_RSUN_PER_DAY, G

    e2 = v["ecc"] ** 2
    tcirc = (
        4e6
        / 63.0
        * np.sqrt(v["a"] ** 3 / (G * v["star_mass"]))
        * (v["mass"] / v["star_mass"])
        * (v["ar"] / v["p"]) ** 5
        * (1.0 - e2) ** 6.5
        / (1.0 + 6.0 * e2)
        / 365.25e9
    )
    assert v["tcirc"] == pytest.approx(tcirc, rel=1e-10)
    omegagr = (
        3.0
        * G
        * v["star_mass"]
        * (2.0 * np.pi / v["period"])
        / (v["a"] * C_LIGHT_RSUN_PER_DAY**2 * (1.0 - e2))
    )
    assert v["omegagr"] == pytest.approx(omegagr, rel=1e-10)


def test_orbit_outputs_are_declared_only_with_an_orbit(earth_system):
    """
    Given the built System, and a second one with no orbit component,
    When each planet manifest is inspected,
    Then teq, fave, msini and the four probabilities are derived
      ("default") entries exactly when an orbit exists -- all need a/R_*
      or sin i, which a planet without an orbit does not have -- while q
      needs only the two masses and is declared either way.
    """
    system, _ = earth_system
    orbit_only = (
        "teq",
        "fave",
        "msini",
        "pt",
        "ptg",
        "ps",
        "psg",
        "delta",
        "tcirc",
        "omegagr",
    )
    for name in orbit_only + ("q",):
        assert system.planet.manifest[name] == "default"
        assert ("planet", name) in system.derived_params()

    orbitless = System(
        {"star": [{"name": "A", "mist": False}], "planet": [{"name": "b"}]},
        user_params={
            k: v for k, v in _params().items() if not k.startswith("orbit.")
        },
    )
    orbitless.prepare()
    for name in orbit_only:
        assert name not in orbitless.planet.manifest
    assert orbitless.planet.manifest["q"] == "default"


def test_linear_mode_planet_still_reports_q():
    """
    Given a planet whose mass coordinate is `linear` (the RV/astrometry
      case, where log_q is not a parameter of the planet at all),
    When its manifest is inspected,
    Then q is still a derived entry -- the linear mass ratio was
      otherwise unreported for every RV-measured planet, which is what a
      reader of an RV fit's table saw as "mp/mstar is missing".
    """
    system = System(
        {
            "star": [{"name": "A", "mist": False}],
            "planet": [{"name": "b", "mass_parameterization": "linear"}],
        },
        user_params={
            k: v for k, v in _params().items() if not k.startswith("orbit.")
        },
    )
    system.prepare()
    assert "log_q" not in system.planet.manifest
    assert system.planet.manifest["mass"] is None
    assert system.planet.manifest["q"] == "default"
