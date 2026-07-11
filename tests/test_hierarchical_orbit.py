"""
Tests for hierarchical orbits: body groups on the orbit component,
per-orbit mass/scale parameters (m_primary, m_companion, m_total, arsun,
K), membership-based RV models, and orbit-referenced relative astrometry.

Reference topology (KELT-4-like): stars A, B, C and planet b, with
  orbit b    : b orbits A
  orbit BC   : B orbits C
  orbit A-BC : the B+C pair orbits A(+b)
"""

import numpy as np
import pymc as pm
import pytensor
import pytest

from exozippy.components.orbit.bodies import parse_body_ref, parse_orbit_bodies
from exozippy.config import ConfigManager
from exozippy.constants import KEPLER_CONST
from exozippy.system import System

import astropy.units as u

MJUP_PER_MSUN = (1.0 * u.solMass).to(u.jupiterMass).value

M_A, M_B, M_C = 1.2, 0.8, 0.7
M_b = 0.1                      # 0.1 Msun "planet" so mass sums are visible
LOGP = [1.0, 4.0, 6.0]
TC = 2456000.0


def _hier_config():
    return {
        "star": [{"name": "A", "mist": False},
                 {"name": "B", "mist": False},
                 {"name": "C", "mist": False}],
        "planet": [{"name": "b"}],
        "orbit": [
            {"name": "b", "primary": ["A"], "companion": ["b"]},
            {"name": "BC", "primary": ["C"], "companion": ["B"]},
            {"name": "A-BC", "primary": ["A", "b"], "companion": ["B", "C"]},
        ],
    }


def _hier_params():
    return {
        "star.A.mass": {"initval": M_A},
        "star.B.mass": {"initval": M_B},
        "star.C.mass": {"initval": M_C},
        "planet.b.mass": {"initval": M_b * MJUP_PER_MSUN},  # jupiterMass
        "orbit.b.logP": {"initval": LOGP[0]},
        "orbit.BC.logP": {"initval": LOGP[1]},
        "orbit.A-BC.logP": {"initval": LOGP[2]},
        "orbit.b.tc": {"initval": TC},
        "orbit.BC.tc": {"initval": TC},
        "orbit.A-BC.tc": {"initval": TC},
    }


def _eval_at_start(system, model, tensors):
    raw = system.get_raw_start(model)
    fn = pytensor.function(model.free_RVs, tensors, on_unused_input="ignore")
    return fn(*[raw[rv.name] for rv in model.free_RVs])


# ---------------------------------------------------------------------------
# Body-group parsing
# ---------------------------------------------------------------------------

def test_parse_body_ref_names_paths_and_ambiguity():
    """
    Given star and planet instance-name lists,
    When body references are parsed,
    Then bare names, star./planet. paths, and index paths all resolve, and
    ambiguous or unknown names raise.
    """
    stars, planets = ["A", "B"], ["b"]
    assert parse_body_ref("A", stars, planets) == ("star", 0)
    assert parse_body_ref("b", stars, planets) == ("planet", 0)
    assert parse_body_ref("star.B", stars, planets) == ("star", 1)
    assert parse_body_ref("planet.0", stars, planets) == ("planet", 0)
    with pytest.raises(ValueError, match="ambiguous"):
        parse_body_ref("X", ["X"], ["X"])
    with pytest.raises(ValueError, match="does not match"):
        parse_body_ref("Z", stars, planets)
    with pytest.raises(ValueError, match="out of range"):
        parse_body_ref("star.5", stars, planets)


def test_parse_orbit_bodies_explicit_groups():
    """
    Given the KELT-4-like config with explicit primary/companion lists,
    When the orbit body groups are parsed,
    Then each orbit gets the declared (comp_type, index) tuples.
    """
    cfg = _hier_config()
    prim, comp = parse_orbit_bodies(cfg["orbit"], cfg)
    assert prim[0] == [("star", 0)] and comp[0] == [("planet", 0)]
    assert prim[1] == [("star", 2)] and comp[1] == [("star", 1)]
    assert prim[2] == [("star", 0), ("planet", 0)]
    assert comp[2] == [("star", 1), ("star", 2)]


def test_parse_orbit_bodies_implicit_legacy_topology():
    """
    Given orbits without primary/companion keys,
    When the groups are parsed,
    Then each orbit pairs the planets whose orbit_ndx points at it with
    those planets' host stars (the historical implicit topology).
    """
    cfg = {
        "star": [{"name": "S1"}, {"name": "S2"}],
        "planet": [{"name": "p1", "orbit_ndx": 0},
                   {"name": "p2", "orbit_ndx": 1, "star_ndx": 1}],
        "orbit": [{"name": "o1"}, {"name": "o2"}],
    }
    prim, comp = parse_orbit_bodies(cfg["orbit"], cfg)
    assert prim[0] == [("star", 0)] and comp[0] == [("planet", 0)]
    assert prim[1] == [("star", 1)] and comp[1] == [("planet", 1)]


def test_parse_orbit_bodies_rejects_bad_groups():
    """
    Given group declarations that overlap, are one-sided, or use 'bodies',
    When parsed,
    Then a ValueError names the problem.
    """
    cfg = _hier_config()
    with pytest.raises(ValueError, match="both the primary and companion"):
        parse_orbit_bodies([{"name": "x", "primary": ["A"],
                             "companion": ["A"]}], cfg)
    with pytest.raises(ValueError, match="give both"):
        parse_orbit_bodies([{"name": "x", "primary": ["A"]}], cfg)
    with pytest.raises(ValueError, match="not a supported key"):
        parse_orbit_bodies([{"name": "x", "bodies": ["A"]}], cfg)


# ---------------------------------------------------------------------------
# Symbol-map regression: Kepler symbols must be per-instance
# ---------------------------------------------------------------------------

def test_kepler_relation_symbols_are_instance_scoped():
    """
    Given a multi-orbit config,
    When the relaxation-engine relations are instantiated,
    Then each orbit's Kepler relation uses its own arsun/m_total symbols
    (a bare shared 'a' or 'm_total' would let the engine equate different
    orbits' physics).
    """
    cm = ConfigManager(_hier_params(), system_config=_hier_config())
    sym_names = {s.name for rel in cm.all_relations for s in rel.free_symbols}
    assert "orbit.0.arsun" in sym_names
    assert "orbit.2.arsun" in sym_names
    assert "orbit.1.m_total" in sym_names
    assert "a" not in sym_names
    assert "m_total" not in sym_names


def test_m_total_custom_solver_sums_member_masses():
    """
    Given user mass initvals for all bodies,
    When the relaxation engine finalizes,
    Then each orbit's m_total initval is the sum of its member masses.
    """
    system = System(_hier_config(), user_params=_hier_params())
    system.prepare()
    cfg = system.config_manager.resolve("orbit", "m_total", shape=(3,),
                                        names=system.orbit.names)
    m_tot = np.atleast_1d(cfg["initval"]).astype(float)
    expected = [M_A + M_b, M_C + M_B, M_A + M_b + M_B + M_C]
    np.testing.assert_allclose(m_tot, expected, rtol=1e-3)


# ---------------------------------------------------------------------------
# Model-level: per-orbit mass/scale parameters
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def hier_system():
    """Prepared + built KELT-4-like hierarchical system (no data)."""
    with pytensor.config.change_flags(mode="FAST_COMPILE"):
        system = System(_hier_config(), user_params=_hier_params())
        system.prepare()
        model = system.build_model()
    return system, model


def test_orbit_group_masses_match_body_sums(hier_system):
    """
    Given the built hierarchical model,
    When m_primary/m_companion/m_total are evaluated at the start point,
    Then they equal the group sums of the star/planet mass initvals.
    """
    system, model = hier_system
    orb = system.orbit
    m_p, m_c, m_t = _eval_at_start(
        system, model, [orb.m_primary.value, orb.m_companion.value,
                        orb.m_total.value])
    np.testing.assert_allclose(m_p, [M_A, M_C, M_A + M_b], rtol=1e-4)
    np.testing.assert_allclose(m_c, [M_b, M_B, M_B + M_C], rtol=1e-4)
    np.testing.assert_allclose(m_t, m_p + m_c, rtol=1e-12)


def test_orbit_scale_follows_keplers_third_law(hier_system):
    """
    Given the built hierarchical model,
    When arsun and K are evaluated at the start point,
    Then arsun = KEPLER_CONST * m_total^(1/3) * P^(2/3) per orbit, and K
    matches the two-body semi-amplitude of the primary group.
    """
    system, model = hier_system
    orb = system.orbit
    m_t, arsun, K, period, sini, ecc = _eval_at_start(
        system, model, [orb.m_total.value, orb.arsun.value, orb.K.value,
                        orb.period.value, orb.sini.value, orb.ecc.value])
    np.testing.assert_allclose(period, 10.0 ** np.array(LOGP), rtol=1e-6)
    np.testing.assert_allclose(
        arsun, KEPLER_CONST * m_t ** (1 / 3) * period ** (2 / 3), rtol=1e-6)
    m_c = np.array([M_b, M_B, M_B + M_C])
    K_expected = (2 * np.pi * arsun * sini * (m_c / m_t)
                  / (period * np.sqrt(1 - ecc ** 2)))
    np.testing.assert_allclose(K, K_expected, rtol=1e-3)


# ---------------------------------------------------------------------------
# Membership: which orbits move which star
# ---------------------------------------------------------------------------

def test_star_membership_roles():
    """
    Given the hierarchical topology,
    When star_membership is queried per star,
    Then each star maps to the orbits (and roles) it participates in.
    """
    cm = ConfigManager(_hier_params(), system_config=_hier_config())
    from exozippy.components.orbit.orbit import Orbit
    orb = Orbit(_hier_config()["orbit"], cm)
    assert orb.star_membership(0) == [(0, "primary"), (2, "primary")]
    assert orb.star_membership(1) == [(1, "companion"), (2, "companion")]
    assert orb.star_membership(2) == [(1, "primary"), (2, "companion")]


def test_rv_terms_include_stellar_companion_orbit(tmp_path):
    """
    Given RVs of star A in the hierarchical system,
    When the RV model terms are assembled,
    Then they cover exactly the orbits with A in the primary group (b and
    A-BC) with the primary-reflex K, and the model logp is finite.
    """
    rv_file = tmp_path / "fake.rv"
    rng = np.random.default_rng(42)
    t = np.linspace(TC - 20.0, TC + 20.0, 12)
    np.savetxt(rv_file, np.column_stack(
        [t, rng.normal(0, 5, t.size), np.full(t.size, 3.0)]))

    config = _hier_config()
    config["rvinstrument"] = [{"name": "FAKE", "file": str(rv_file)}]
    with pytensor.config.change_flags(mode="FAST_COMPILE"):
        system = System(config, user_params=_hier_params())
        system.prepare()
        model = system.build_model()

    K_vec, omap = system.rvinstrument._orbit_rv_terms(system, 0)
    assert list(omap) == [0, 2]

    K_terms, K_all = _eval_at_start(system, model,
                                    [K_vec, system.orbit.K.value])
    np.testing.assert_allclose(K_terms, K_all[[0, 2]], rtol=1e-10)

    logp = model.compile_logp()(system.get_raw_start(model))
    assert np.isfinite(logp)


# ---------------------------------------------------------------------------
# Relative astrometry referencing an orbit by name
# ---------------------------------------------------------------------------

def _write_rel_file(path, sep_mas):
    rows = np.array([[TC + 10.0, sep_mas, sep_mas * 0.01, 45.0, 1.0],
                     [TC + 400.0, sep_mas, sep_mas * 0.01, 46.0, 1.0]])
    np.savetxt(path, rows)


def test_rel_astrometry_traces_named_orbit(tmp_path):
    """
    Given rel-mode astrometry blocks naming the BC and A-BC orbits,
    When the system is built,
    Then each dataset resolves to its orbit index, the A-BC dataset picks
    up the nested BC photocenter sub-orbit, and the model logp is finite.
    """
    f_bc = tmp_path / "bc.rel"
    f_abc = tmp_path / "abc.rel"
    _write_rel_file(f_bc, 50.0)
    _write_rel_file(f_abc, 1500.0)

    config = _hier_config()
    config["astrometryinstrument"] = [
        {"name": "BCrel", "file": str(f_bc), "mode": "rel", "orbit": "BC"},
        {"name": "ABCrel", "file": str(f_abc), "mode": "rel",
         "orbit": "A-BC"},
    ]
    with pytensor.config.change_flags(mode="FAST_COMPILE"):
        system = System(config, user_params=_hier_params())
        system.prepare()
        model = system.build_model()

    astrom = system.astrometryinstrument
    assert astrom.rel_orbit == [1, 2]

    # No abs/gaia data: the stars' ra/dec/pm must not be sampled
    assert not hasattr(system.star, "ra")

    # Nested photocenter: BC ({B,C}) is a sub-orbit of A-BC's companion
    # group; with no SED it falls back to the barycenter (beta None), and
    # the planet-b orbit nested in the primary group is dark (beta = 0)
    assert astrom._pair_beta(system, 1, 1) is None
    assert astrom._pair_beta(system, 1, 0) == 0.0

    logp = model.compile_logp()(system.get_raw_start(model))
    assert np.isfinite(logp)


def test_rel_astrometry_unknown_orbit_raises(tmp_path):
    """
    Given a rel-mode block naming a nonexistent orbit,
    When the component is instantiated,
    Then a ValueError lists the available orbit names.
    """
    f = tmp_path / "x.rel"
    _write_rel_file(f, 10.0)
    config = _hier_config()
    config["astrometryinstrument"] = [
        {"name": "bad", "file": str(f), "mode": "rel", "orbit": "nope"}]
    with pytest.raises(ValueError, match="unknown orbit"):
        System(config, user_params=_hier_params())
