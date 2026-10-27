"""A star in an orbit group shares its anchor's distance, proper motion,
systemic rv, initfeh, age and av (JDE 2026-10-26, rv 2026-10-27;
``Star._link_orbit_members``,
star.md "Linked stars", orbit.md "Hierarchical orbits and body groups").

Every STAR in either group of any orbit -- Keplerian, Taylor or static --
is hard-linked to that orbit's anchor (the first star of its primary
group), chained to the root anchor through a hierarchy, on each of those
parameters the topology declares for both.  Pinned here: each parameter on
each orbit type, the chain, the user's soft link breaking exactly one link,
the redundant hand link, and the raise when a parameter exists for only
one star of the pair.
"""

import logging

import numpy as np
import pytensor
import pytest

from exozippy.system import System

LINKED = ("distance", "pm_ra", "pm_dec", "rv", "initfeh", "age", "av")
RA_DEG, DEC_DEG = 266.4168, -29.0078


@pytest.fixture(scope="module")
def model_root(tmp_path_factory):
    """A tiny MIST grid; the shipped one is gitignored."""
    from conftest import write_synthetic_mist_grid

    return write_synthetic_mist_grid(tmp_path_factory.mktemp("mist_models"))


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("orbit_links")
    t = np.linspace(2459600.0, 2459760.0, 25)
    rv = tmp / "a.rv"
    np.savetxt(
        rv, np.column_stack([t, 300.0 - 1.0 * (t - t.mean()), 5.0 + 0 * t])
    )
    sed = tmp / "empty.sed"
    sed.write_text("model: NextGen\nfilters: []\n")
    return str(rv), str(sed)


def _config(files, model_root, orbits, stars=("A", "B")):
    """Every consumer that declares one of the shared parameters, for each
    star.  RVs of A only where an orbit reads them (a static orbit moves
    nothing, so an RV star on one alone has no RV model and raises)."""
    rv, sed = files
    cfg = {
        "star": [{"name": s} for s in stars],
        "orbit": orbits,
        "galacticmodel": [{"star_ndx": 0}],
        "evolutionarymodel": [
            {"star": s, "model_root": model_root} for s in stars
        ],
        "band": [{"name": "G", "filter": "GAIA2r.G", "ld_law": "linear"}],
        "sed": {"file": sed},
    }
    if any(o.get("type", "keplerian") != "static" for o in orbits):
        cfg["rvinstrument"] = [{"name": "TRES", "file": rv}]
    return cfg


def _params(extra=None):
    p = {
        "star.A.ra": {"initval": RA_DEG, "sigma": 0},
        "star.A.dec": {"initval": DEC_DEG, "sigma": 0},
        "star.A.distance": {"initval": 900.0},
        "star.A.pm_ra": {"initval": 2.5},
        "star.A.pm_dec": {"initval": -4.0},
        "star.A.av": {"initval": 0.3},
        "star.av": {"initval": 0.1},  # a start only: makes av read on all
        "star.A.mass": {"initval": 1.0},
        "star.B.mass": {"initval": 0.6},
    }
    p.update(extra or {})
    return p


KEPLER_STARTS = {
    "orbit.AB.period": {"initval": 40.0},
    "orbit.AB.tc": {"initval": 2459680.0},
}


def _prepare(cfg, params):
    system = System(cfg, user_params=params)
    system.prepare()
    return system


def _orbit_links(system):
    return {
        t: pl.expr_str
        for t, fields in system.config_manager.links.items()
        for pl in fields.values()
        if pl.origin == "orbit"
    }


ORBITS = {
    "keplerian": {"name": "AB", "primary": ["A"], "companion": ["B"]},
    "linear": {
        "name": "AB",
        "type": "linear",
        "primary": ["A"],
        "companion": ["B"],
    },
    "static": {
        "name": "AB",
        "type": "static",
        "primary": ["A"],
        "companion": ["B"],
    },
}


@pytest.mark.slow
@pytest.mark.parametrize("kind", sorted(ORBITS))
def test_every_shared_parameter_is_linked_on_every_orbit_type(
    files, model_root, kind
):
    """
    Given stars A and B in one orbit (Keplerian, linear Taylor or static),
    with distance, pm, rv, initfeh, age and av all declared for both,
    When the system is prepared,
    Then each of B's elements is hard-linked to A's through the component
    channel (origin "orbit", sigma 0 through the overrides channel), and
    nothing is written to user_params for B.
    """
    system = _prepare(
        _config(files, model_root, [ORBITS[kind]]),
        _params(KEPLER_STARTS if kind == "keplerian" else None),
    )
    links = _orbit_links(system)
    assert links == {f"star.1.{p}": f"star.0.{p}" for p in LINKED}
    star = system.star
    for p in LINKED:
        sigma = (star.manifest[p] or {})["overrides"]["sigma"]
        assert sigma[1] == 0.0, p
    assert not any(
        k.startswith("star.1.") or k.startswith("star.B.")
        for k in system.config_manager.user_params
        if k.split(".")[-1] in LINKED
        and not system.config_manager.user_params[k].get("derived")
    )


@pytest.mark.slow
def test_the_linked_values_track_the_anchor_in_the_built_model(
    files, model_root
):
    """
    Given A and B on a static orbit,
    When the model is built,
    Then B's linked elements are hard links whose start equals A's, and the
    galactic model counts B's distance as a member of A's position (one
    density term), as for a hand-written link.
    """
    system = _prepare(
        _config(files, model_root, [ORBITS["static"]]), _params()
    )
    model = system.build_model()
    star = system.star
    nodes = [getattr(star, p).value for p in LINKED]
    fn = pytensor.function(model.free_RVs, nodes, on_unused_input="ignore")
    ip = model.initial_point()
    vals = fn(*[ip[v.name] for v in model.free_RVs])
    for p, v in zip(LINKED, vals):
        assert 1 in (getattr(star, p).element_links or {}).get("hard", {}), p
        assert v[1] == v[0], p
    assert vals[0][0] == pytest.approx(900.0)
    free = {v.name for v in model.free_RVs}
    assert np.isfinite(model.compile_logp()(ip))
    assert free  # sanity: the model has coordinates at all


@pytest.mark.slow
def test_a_hierarchy_chains_to_the_root_anchor(files, model_root):
    """
    Given KELT-4's shape -- B orbits C, and [B, C] orbits A --
    When the system is prepared,
    Then both B and C link straight to A, the root anchor.
    """
    orbits = [
        {"name": "BC", "type": "static", "primary": ["C"], "companion": ["B"]},
        {
            "name": "ABC",
            "type": "static",
            "primary": ["A"],
            "companion": ["B", "C"],
        },
    ]
    system = _prepare(
        _config(files, model_root, orbits, stars=("A", "B", "C")),
        _params(),
    )
    assert system.star.orbit_anchors(system) == {1: 0, 2: 0}
    links = _orbit_links(system)
    for i in (1, 2):
        for p in LINKED:
            assert links[f"star.{i}.{p}"] == f"star.0.{p}"


@pytest.mark.slow
def test_a_user_soft_link_breaks_exactly_that_link(files, model_root, caplog):
    """
    Given a soft link the user wrote on B's age (sigma > 0),
    When the system is prepared,
    Then B's age is not orbit-linked -- the user's link stands -- every
    other parameter still is, and the startup output says which link was
    not made and why.
    """
    params = _params({"star.B.age": {"initval": "star.A.age", "sigma": 0.5}})
    with caplog.at_level(
        logging.WARNING, logger="exozippy.components.star.star"
    ):
        system = _prepare(
            _config(files, model_root, [ORBITS["static"]]), params
        )
    links = _orbit_links(system)
    assert "star.1.age" not in links
    assert set(links) == {f"star.1.{p}" for p in LINKED if p != "age"}
    assert system.config_manager.links["star.1.age"]["initval"].origin == (
        "user"
    )
    assert any(
        "NOT made" in r.message and "B.age" in r.message
        for r in caplog.records
    )


@pytest.mark.slow
def test_a_mu_sigma_prior_breaks_that_link(files, model_root):
    """
    Given a Gaussian prior the user put on B's av,
    Then B's av is not orbit-linked and keeps the user's prior.
    """
    params = _params({"star.B.av": {"mu": 0.5, "sigma": 0.1}})
    system = _prepare(_config(files, model_root, [ORBITS["static"]]), params)
    links = _orbit_links(system)
    assert "star.1.av" not in links
    assert "star.1.distance" in links


@pytest.mark.slow
def test_an_identical_hand_link_is_redundant_not_doubled(
    files, model_root, caplog
):
    """
    Given the user's own hard link of B's distance to A's (what the
    KELT-4 params file writes),
    When the system is prepared,
    Then the user's link is kept as the only link on that element, and a
    warning says it is redundant.
    """
    params = _params(
        {"star.B.distance": {"initval": "star.A.distance", "sigma": 0}}
    )
    with caplog.at_level(
        logging.WARNING, logger="exozippy.components.star.star"
    ):
        system = _prepare(
            _config(files, model_root, [ORBITS["static"]]), params
        )
    link = system.config_manager.links["star.1.distance"]
    assert set(link) == {"initval"} and link["initval"].origin == "user"
    assert "star.1.distance" not in _orbit_links(system)
    assert any(
        "redundant" in r.message or "can be deleted" in r.message
        for r in caplog.records
        if "B.distance" in r.message
    )


@pytest.mark.slow
def test_a_parameter_on_one_star_only_raises(files, model_root):
    """
    Given B opted out of the evolutionary model (mist: False) while A has a
    track,
    When the system is prepared,
    Then it raises naming both stars: one orbit system, one age.
    """
    cfg = _config(files, model_root, [ORBITS["static"]])
    cfg["star"][1]["mist"] = False
    cfg["evolutionarymodel"] = [{"star": "A", "model_root": model_root}]
    with pytest.raises(ValueError, match=r"B \(member\) and A \(anchor\)"):
        _prepare(cfg, _params())


@pytest.mark.slow
def test_a_member_linked_on_position_and_velocity_adds_no_kinematic_term(
    files, model_root
):
    """
    Given A and B on a static orbit under the galactic model,
    When the model is built,
    Then B shares A's position AND all three velocity coordinates
    (pm_ra, pm_dec, rv) -- the structure for which the kinematic prior
    adds nothing for B (``test_galactic_linked_stars.py``,
    ``test_fully_linked_member_contributes_nothing``) -- and B's rv starts
    at A's.
    """
    from exozippy.components.galacticmodel.galacticmodel import (
        linked_star_structure,
    )

    system = _prepare(
        _config(files, model_root, [ORBITS["static"]]),
        _params({"star.A.rv": {"initval": 12000.0}}),
    )
    model = system.build_model()
    star = system.star
    root_of, vel_linked, refused = linked_star_structure(star, 2)
    assert list(root_of) == [0, 0]
    assert vel_linked == {1: (0, 1, 2)}
    assert refused == []

    fn = pytensor.function(
        model.free_RVs, star.rv.value, on_unused_input="ignore"
    )
    ip = model.initial_point()
    vals = fn(*[ip[v.name] for v in model.free_RVs])
    assert vals[1] == vals[0] == pytest.approx(12000.0)


@pytest.mark.slow
def test_a_user_rv_prior_breaks_only_the_rv_link(files, model_root, caplog):
    """
    Given a Gaussian prior the user put on B's rv (a measured systemic
    velocity of its own), and a bare rv initval nowhere else,
    When the system is prepared,
    Then B's rv is not orbit-linked, every other parameter still is, and
    the startup output says the rv link was not made.
    """
    params = _params({"star.B.rv": {"mu": 3000.0, "sigma": 500.0}})
    with caplog.at_level(
        logging.WARNING, logger="exozippy.components.star.star"
    ):
        system = _prepare(
            _config(files, model_root, [ORBITS["static"]]), params
        )
    links = _orbit_links(system)
    assert set(links) == {f"star.1.{p}" for p in LINKED if p != "rv"}
    assert any(
        "NOT made" in r.message and "B.rv" in r.message for r in caplog.records
    )


@pytest.mark.slow
def test_a_bare_rv_initval_on_the_member_is_reported_unused(
    files, model_root, caplog
):
    """
    Given a bare initval on B's rv (a start, not a statement),
    When the system is prepared,
    Then B's rv is still orbit-linked and an INFO line says the initval is
    unused.
    """
    params = _params({"star.B.rv": {"initval": 5000.0}})
    with caplog.at_level(logging.INFO, logger="exozippy.components.star.star"):
        system = _prepare(
            _config(files, model_root, [ORBITS["static"]]), params
        )
    assert _orbit_links(system)["star.1.rv"] == "star.0.rv"
    assert any(
        "unused" in r.message and "B.rv" in r.message for r in caplog.records
    )
