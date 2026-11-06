"""Planet membership has ONE spelling: the orbit blocks' groups.

A planet's orbit and its host star are derived from the orbit blocks'
`primary:`/`companion:` groups by `orbit.bodies.planet_membership` -- the one
reader -- and never from a key on the planet.  `orbit_ndx:` and `star_ndx:`
were a second spelling of the same fact, with a silent default of 0 that put
a microlensing lens planet on whatever orbit happened to be first (a source's
xallarap orbit, or a stellar companion's linear lens orbit, which then refused
it).  They are removed and raise with the migration.  A planet with no
dynamics is paired by a `type: static` orbit.  Two unambiguous pairings may be
left implicit; everything else raises naming the body and the orbit.
"""

import pytest

from exozippy.components.orbit.bodies import planet_membership
from exozippy.components.orbit.orbit import Orbit
from exozippy.components.planet.planet import Planet
from exozippy.config import ConfigManager


def _lens_config(stars, lens, planets=("b",), orbits=None):
    cfg = {
        "star": [{"name": s} for s in stars],
        "planet": [{"name": p} for p in planets],
        "lens": [{"body": b} for b in lens],
        "source": [{"body": "star.Source"}],
    }
    if orbits is not None:
        cfg["orbit"] = orbits
    return cfg


# ---------------------------------------------------------------------------
# The removed keys
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("key", ["orbit_ndx", "star_ndx"])
def test_a_leftover_membership_key_raises_with_the_migration(key):
    """
    Given a planet that still carries orbit_ndx: or star_ndx:,
    When membership is derived, or the planet is constructed,
    Then it RAISES naming the key, the planet, and the replacement (the
      orbit block's groups, and type: static for a planet that does not
      move) -- never silently ignored.
    """
    cfg = {
        "star": [{"name": "A"}],
        "planet": [{"name": "b", key: 0}],
        "orbit": [{"name": "b", "primary": ["A"], "companion": ["b"]}],
    }
    with pytest.raises(ValueError, match=rf"planet.b\].*'{key}:'.*companion"):
        planet_membership(cfg)
    with pytest.raises(ValueError, match="type: static"):
        Planet(cfg["planet"], ConfigManager({}, system_config=cfg))


# ---------------------------------------------------------------------------
# Derived from the groups
# ---------------------------------------------------------------------------
def test_the_orbit_and_host_come_from_the_groups():
    """
    Given two planets around two stars, each named by its own orbit block
      (listed in the opposite order to the planets),
    Then each planet's orbit is the block naming it and its host the star
      in that block's primary group.
    """
    cfg = {
        "star": [{"name": "A"}, {"name": "B"}],
        "planet": [{"name": "b"}, {"name": "c"}],
        "orbit": [
            {"name": "oc", "primary": ["B"], "companion": ["c"]},
            {"name": "ob", "primary": ["A"], "companion": ["b"]},
        ],
    }
    assert planet_membership(cfg) == [(1, 0), (0, 1)]


def test_an_outer_orbit_may_hold_the_planet_inside_a_larger_group():
    """
    Given planet b on its own orbit around A, and [A, b] as the companion
      group of an outer orbit around C (a hierarchical system),
    Then b's orbit is the INNERMOST one naming it -- its own -- and its
      host is A.
    """
    cfg = {
        "star": [{"name": "A"}, {"name": "C"}],
        "planet": [{"name": "b"}],
        "orbit": [
            {"name": "outer", "primary": ["C"], "companion": ["A", "b"]},
            {"name": "b", "primary": ["A"], "companion": ["b"]},
        ],
    }
    assert planet_membership(cfg) == [(1, 0)]


def test_two_orbits_claiming_the_planet_raise():
    """
    Given two orbit blocks that both name planet b as their lone companion,
    Then it RAISES naming the planet and both orbits: a planet moves on one
      orbit (the old orbit_ndx default double-counted exactly this way).
    """
    cfg = {
        "star": [{"name": "A"}, {"name": "B"}],
        "planet": [{"name": "b"}],
        "orbit": [
            {"name": "o1", "primary": ["A"], "companion": ["b"]},
            {"name": "o2", "primary": ["B"], "companion": ["b"]},
        ],
    }
    with pytest.raises(ValueError, match=r"planet.b\].*'o1', 'o2'"):
        planet_membership(cfg)


def test_a_circumbinary_host_raises():
    """
    Given a planet whose orbit's primary group holds two stars,
    Then it RAISES: q = m_p/m_host and R_P/R_* need ONE host.
    """
    cfg = {
        "star": [{"name": "A"}, {"name": "B"}],
        "planet": [{"name": "b"}],
        "orbit": [{"name": "b", "primary": ["A", "B"], "companion": ["b"]}],
    }
    with pytest.raises(ValueError, match="circumbinary"):
        planet_membership(cfg)


def test_a_static_orbit_pairs_a_lens_planet_with_its_host():
    """
    Given a lens planet with a `type: static` orbit naming it,
    Then its orbit is that static orbit and its host the static orbit's
      primary.
    """
    cfg = _lens_config(
        ["Source", "L1", "L2"],
        ["star.L1", "star.L2", "planet.b"],
        orbits=[
            {
                "name": "s",
                "type": "static",
                "primary": ["L1"],
                "companion": ["b"],
            }
        ],
    )
    assert planet_membership(cfg) == [(0, 1)]


# ---------------------------------------------------------------------------
# The implicit pairings: only where unambiguous
# ---------------------------------------------------------------------------
def test_one_star_one_planet_one_bare_orbit_pair_implicitly():
    """The QoL default every one-planet transit/RV config relies on."""
    cfg = {
        "star": [{"name": "A"}],
        "planet": [{"name": "b"}],
        "orbit": [{"name": "b"}],
    }
    assert planet_membership(cfg) == [(0, 0)]


def test_a_lone_planet_with_no_orbit_is_paired_statically():
    """One star, one planet, no orbit block: an implicit static pairing."""
    cfg = {"star": [{"name": "A"}], "planet": [{"name": "b"}]}
    assert planet_membership(cfg) == [(None, 0)]


def test_a_lens_planet_is_hosted_by_the_lens_star_not_the_source():
    """
    Given a 2L1S lens [Lens, b] with the SOURCE listed first among the
      stars and no orbit block,
    Then b is paired (statically) with the lens star: the scope of a lens
      body is the lens, and a source star cannot host a lens companion.
    """
    cfg = _lens_config(["Source", "Lens"], ["star.Lens", "planet.b"])
    assert planet_membership(cfg) == [(None, 1)]


def test_a_lens_planet_beside_a_xallarap_orbit_is_not_put_on_it():
    """
    Given a lens planet with no orbit block of its own beside a source's
      Keplerian xallarap orbit (examples/ob170114's shape),
    Then the planet has NO orbit: the old orbit_ndx default of 0 put it on
      the source's orbit and derived its p, a, K from that orbit's elements.
    """
    cfg = _lens_config(
        ["L1", "Source", "SComp"],
        ["star.L1", "planet.b"],
        orbits=[
            {"name": "xal", "primary": ["Source"], "companion": ["SComp"]}
        ],
    )
    assert planet_membership(cfg) == [(None, 0)]


@pytest.mark.parametrize(
    "lens, planets",
    [
        # two lens stars: which hosts the planet?
        (["star.L1", "star.L2", "planet.b"], ("b",)),
        # two lens planets
        (["star.L1", "planet.b", "planet.c"], ("b", "c")),
    ],
)
def test_an_ambiguous_implicit_pairing_raises_asking_for_the_orbit_block(
    lens, planets
):
    stars = sorted(
        {b.split(".")[1] for b in lens if b.startswith("star.")} | {"Source"}
    )
    cfg = _lens_config(stars, lens, planets=planets)
    with pytest.raises(ValueError, match=r"planet.b\].*type: static"):
        planet_membership(cfg)


def test_a_bare_orbit_beside_two_stars_raises():
    """The implicit Keplerian pairing needs exactly one star: with two, the
    old star_ndx default of 0 was a guess."""
    cfg = {
        "star": [{"name": "A"}, {"name": "B"}],
        "planet": [{"name": "b"}],
        "orbit": [{"name": "b"}],
    }
    with pytest.raises(ValueError, match="primary:' and 'companion:'"):
        planet_membership(cfg)


# ---------------------------------------------------------------------------
# The static orbit block itself
# ---------------------------------------------------------------------------
def _orbit(blocks, cfg):
    return Orbit(blocks, ConfigManager({}, system_config=cfg))


def _static_cfg(**extra):
    block = {
        "name": "s",
        "type": "static",
        "primary": ["A"],
        "companion": ["b"],
    }
    block.update(extra)
    return [block], {
        "star": [{"name": "A"}],
        "planet": [{"name": "b"}],
        "orbit": [block],
    }


def test_a_static_orbit_is_neither_keplerian_nor_taylor():
    blocks, cfg = _static_cfg()
    orbit = _orbit(blocks, cfg)
    assert list(orbit.is_static) == [True]
    assert list(orbit.is_keplerian) == [False]
    assert list(orbit.is_taylor) == [False]
    # No instrument is moved by a pairing with no dynamics.
    assert orbit.star_membership(0) == []
    assert orbit.taylor_membership(0) == []


@pytest.mark.parametrize(
    "extra, match",
    [
        ({"fitvcve": True}, "only apply to a Keplerian orbit"),
        ({"epoch": 2450000.0}, "reference epoch of a Taylor orbit"),
    ],
)
def test_keys_a_static_orbit_cannot_use_raise(extra, match):
    blocks, cfg = _static_cfg(**extra)
    with pytest.raises(ValueError, match=match):
        _orbit(blocks, cfg)


@pytest.mark.parametrize("drop", ["primary", "companion"])
def test_a_static_orbit_must_name_both_groups(drop):
    blocks, cfg = _static_cfg()
    del blocks[0][drop]
    with pytest.raises(ValueError, match="static orbit is a pairing"):
        _orbit(blocks, cfg)
