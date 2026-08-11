"""
Transit/occultation impact-parameter and duration Deterministics for
eccentric orbits, pinned against Winn 2010 (arXiv:1001.2010, eqs 7-8 and
14-16; same convention as EXOFASTv2 derivepars.pro).

The primary transit happens at true anomaly pi/2 - omega_star (see
calc_tp), where the star-planet separation is r = a(1-e^2)/(1 + esinw);
the secondary eclipse is at the opposite conjunction, r with (1 - esinw).
So for esinw != 0 the primary and secondary get DIFFERENT denominators,
and the 2026-08-08 audit (finding 1.2) caught them swapped: transit.b
(primary) used 1 - esinw and transit.bs used 1 + esinw, with every
duration Deterministic (t14, tfwhm, tau and their secondaries) and
orbit/planet calc_b inheriting the swap. The likelihood itself was always
correct (test_transit_exofast_parity) -- only the reported values were
wrong. These tests evaluate the Deterministics on a deliberately
eccentric system (e = 0.3, omega = 60 deg, so esinw ~ 0.26 splits the
two conjunctions by ~50%) and would fail under either single-sided or
consistently re-swapped denominators. They also pin finding 1.12: the
reported tfwhm must be (t14 + t23)/2, not Winn's t23 itself.
"""

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from exozippy.components.orbit.physics import calc_b
from exozippy.system import System

ECC = 0.3
OMEGA = np.radians(60.0)  # esinw = 0.3*sin(60 deg) ~ 0.2598


def _write_lc(path, t):
    np.savetxt(
        path, np.column_stack([t, np.ones_like(t), np.full_like(t, 1e-3)])
    )
    return str(path)


def _params():
    """A short-period hot Jupiter on a significantly eccentric orbit."""
    return {
        "star.0.radius": {"initval": 1.0, "sigma": 0.05},
        "star.0.mass": {"initval": 1.0, "sigma": 0.05},
        "star.0.teff": {"initval": 5800, "sigma": 100},
        "star.0.feh": {"initval": 0.0, "sigma": 0.08},
        "orbit.0.period": {"initval": 3.0},
        "orbit.0.tc": {"initval": 2459634.3},
        "orbit.0.cosi": {"initval": 0.05},
        "orbit.0.secosw": {"initval": np.sqrt(ECC) * np.cos(OMEGA)},
        "orbit.0.sesinw": {"initval": np.sqrt(ECC) * np.sin(OMEGA)},
        "planet.0.radius": {"initval": 1.2},
    }


def _initial_point_fn(model, tensors):
    ip = model.initial_point()
    givens = [
        (rv, np.asarray(ip[rv.name])) for rv in model.free_RVs if rv.name in ip
    ]
    return pytensor.function(
        [],
        tensors,
        givens=givens,
        on_unused_input="ignore",
        mode="FAST_COMPILE",
    )


@pytest.fixture(scope="module")
def eccentric_system(tmp_path_factory):
    """A one-planet eccentric transit System, its model, and the model's
    own geometry inputs (ar, cosi, sini, ecc, esinw, period, p) plus every
    impact-parameter/duration Deterministic and planet.b, all evaluated at
    the initial point."""
    d = tmp_path_factory.mktemp("transit_durations")
    t = 2459634.3 + np.linspace(-0.2, 0.2, 200)
    config = {
        "star": [{"name": "A", "mist": False}],
        "planet": [{"name": "b"}],
        "orbit": [{"name": "b"}],
        "band": [{"name": "V", "filter": "V", "ld_law": "quadratic"}],
        "transit": [
            {"name": "inst0", "file": _write_lc(d / "lc.dat", t), "band": "V"}
        ],
    }
    system = System(config, user_params=_params())
    system.prepare()
    model = system.build_model()

    orbits, planets = system.orbit, system.planet
    inputs = ["ar", "cosi", "sini", "ecc", "esinw", "period", "p"]
    dets = ["b", "bs", "t14", "t14s", "tfwhm", "tfwhms", "tau", "taus"]
    fn = _initial_point_fn(
        model,
        [
            planets.ar.value[0],
            orbits.cosi.value[0],
            pt.sin(orbits.inc.value[0]),
            orbits.ecc.value[0],
            orbits.esinw.value[0],
            orbits.period.value[0],
            planets.p.value[0],
        ]
        + [model[f"transit.{name}"][0] for name in dets]
        + [planets.b.value[0]],
    )
    values = [float(v) for v in fn()]
    geom = dict(zip(inputs, values[: len(inputs)]))
    reported = dict(zip(dets, values[len(inputs) : len(inputs) + len(dets)]))
    reported["planet.b"] = values[-1]
    return geom, reported


def _winn_durations(g, sign):
    """Winn 2010 eqs 7-8 and 14-16 in numpy. sign=+1 is the primary
    transit (conjunction at r = a(1-e^2)/(1+esinw)), sign=-1 the
    secondary. The (1-p)^2 arcsin quantity is Winn's t23 (the
    full-occultation duration); tfwhm = (t14+t23)/2 and
    tau = (t14-t23)/2, the EXOFASTv2 derivepars.pro convention (audit
    finding 1.12 had transit.tfwhm reporting t23 itself)."""
    denom = 1.0 + sign * g["esinw"]
    ecc_factor = np.sqrt(1.0 - g["ecc"] ** 2)
    b = g["ar"] * g["cosi"] * (1.0 - g["ecc"] ** 2) / denom

    def duration(p_offset):
        arg = np.sqrt(p_offset**2 - b**2) / (g["sini"] * g["ar"])
        return (g["period"] / np.pi) * np.arcsin(arg) * ecc_factor / denom

    t14 = duration(1.0 + g["p"])
    t23 = duration(1.0 - g["p"])
    return b, t14, (t14 + t23) / 2.0, (t14 - t23) / 2.0


def test_primary_deterministics_match_winn2010(eccentric_system):
    """
    Given an eccentric transiting system (e=0.3, omega=60 deg),
    When transit.b/t14/tfwhm/tau are evaluated at the initial point,
    Then they match Winn 2010's PRIMARY-transit formulas -- the ones with
    the 1 + esinw denominator -- computed in numpy from the model's own
    geometry (ar, cosi, ecc, esinw, period, p).
    """
    geom, reported = eccentric_system
    b, t14, tfwhm, tau = _winn_durations(geom, +1)
    assert reported["b"] == pytest.approx(b, rel=1e-10)
    assert reported["t14"] == pytest.approx(t14, rel=1e-10)
    assert reported["tfwhm"] == pytest.approx(tfwhm, rel=1e-10)
    assert reported["tau"] == pytest.approx(tau, rel=1e-10)


def test_secondary_deterministics_match_winn2010(eccentric_system):
    """
    Given the same eccentric system,
    When transit.bs/t14s/tfwhms/taus are evaluated at the initial point,
    Then they match Winn 2010's SECONDARY-eclipse formulas -- the ones
    with the 1 - esinw denominator.
    """
    geom, reported = eccentric_system
    bs, t14s, tfwhms, taus = _winn_durations(geom, -1)
    assert reported["bs"] == pytest.approx(bs, rel=1e-10)
    assert reported["t14s"] == pytest.approx(t14s, rel=1e-10)
    assert reported["tfwhms"] == pytest.approx(tfwhms, rel=1e-10)
    assert reported["taus"] == pytest.approx(taus, rel=1e-10)


def test_primary_secondary_asymmetry_direction(eccentric_system):
    """
    Given esinw > 0 (the planet is nearer the star at inferior
    conjunction than at superior conjunction),
    When the primary and secondary geometries are compared,
    Then the primary is the closer, faster conjunction: b < bs and
    t14 < t14s. This direction is fixed by the geometry itself, so it
    catches a consistent re-swap of both the code and the formula-based
    assertions above.
    """
    geom, reported = eccentric_system
    assert geom["esinw"] > 0.2  # the fixture really is eccentric
    assert reported["b"] < reported["bs"]
    assert reported["t14"] < reported["t14s"]
    assert reported["tfwhm"] < reported["tfwhms"]


def test_planet_b_matches_primary_transit_b(eccentric_system):
    """
    Given the same eccentric system,
    When the derived planet.b (orbit/physics.py calc_b, shared with
    orbit.b via defaults.yaml) is evaluated,
    Then it equals the primary-transit impact parameter -- both the
    transit.b Deterministic and the Winn 2010 formula.
    """
    geom, reported = eccentric_system
    b, _, _, _ = _winn_durations(geom, +1)
    assert reported["planet.b"] == pytest.approx(b, rel=1e-10)
    assert reported["planet.b"] == pytest.approx(reported["b"], rel=1e-12)


def test_calc_b_is_primary_convention():
    """
    Given scalar float64 inputs (pt.dscalar -- a bare Python float would
    autocast to float32 and lose precision),
    When calc_b is evaluated directly,
    Then it implements Winn 2010 eq 7: b = ar*cosi*(1-e^2)/(1 + esinw).
    """
    ar, cosi, ecc, esinw = (
        pt.dscalar(n) for n in ["ar", "cosi", "ecc", "esinw"]
    )
    fn = pytensor.function(
        [ar, cosi, ecc, esinw], calc_b(ar, cosi, ecc, esinw)
    )
    val = float(fn(8.0, 0.05, 0.3, 0.3 * np.sin(OMEGA)))
    expected = 8.0 * 0.05 * (1.0 - 0.3**2) / (1.0 + 0.3 * np.sin(OMEGA))
    assert val == pytest.approx(expected, rel=1e-12)
