"""The Galactic kinematic prior for LINKED stars (star.md, "Linked stars: one
position, one density term").

A star whose distance is hard-linked to another star's has no position of its
own, so the system gets ONE density and ONE d^2 volume element.  Before the
fix a binary-source (1L2S) model carried a second, non-constant density
factor on D_s through the linked Source2 and its ln Z was biased against
1L1S/2L1S.  These tests pin:

1. the static rule (``linked_star_structure``): who shares whose position,
   which velocity coordinates are linked, and what is refused;
2. under the genulens port (koshimoto21) a PARTLY linked velocity raises:
   its conditional has no closed form under the non-Gaussian disk
   kinematics.  (The removed 3-population model built it as a Gaussian
   conditional; review 1.7.6.)  A fully linked member adding only its mass
   term is tests/test_koshimoto21_model.py's
   ``test_fully_linked_member_adds_only_its_mass``;
3. ``evidence_log_norm`` counts independent positions;
4. end to end on a System: a 1L1S and a 1L2S model of one event carry the
   identical galactic contribution and the identical ``evidence_log_norm``.
"""

import numpy as np
import pymc as pm
import pytest

from conftest import _DummyConfigManager, _MockParam
from exozippy.components.galacticmodel import koshimoto21 as k21
from exozippy.components.galacticmodel.galacticmodel import (
    GalacticModel,
    linked_star_structure,
)

_RA_DEG, _DEC_DEG = 268.0, -29.5
_RA, _DEC = np.radians(_RA_DEG), np.radians(_DEC_DEG)

# Three stars: Lens (0), Source (1), Source2 (2, linked to Source).
_D = [4000.0, 8200.0, 8200.0]
_PM_RA = [-4.0, -6.1, -6.1]
_PM_DEC = [-1.5, -0.4, -0.4]
_RV = [12000.0, -35000.0, -35000.0]


def _link(target):
    """A hard-link spec as Component._wire_user_links records it."""
    return {
        "fn": None,
        "intra_deps": {target},
        "dep_paths": (f"star.{target}.distance",),
    }


def _pm_link(target, name):
    return {
        "fn": None,
        "intra_deps": {target},
        "dep_paths": (f"star.{target}.{name}",),
    }


class _Stars:
    def __init__(self, n, links=(), d=None, pm_ra=None, pm_dec=None, rv=None):
        """``links``: names of the parameters star 2 is hard-linked on."""
        sl = slice(0, n)
        self.names = ["Lens", "Source", "Source2"][sl]
        self.ra = _MockParam([_RA] * n)
        self.dec = _MockParam([_DEC] * n)
        self.logmass = _MockParam([np.log10(0.5)] * n, lower=-9.0, upper=2.5)
        self.distance = _MockParam((d or _D)[sl])
        self.pm_ra = _MockParam((pm_ra or _PM_RA)[sl])
        self.pm_dec = _MockParam((pm_dec or _PM_DEC)[sl])
        self.rv = _MockParam((rv or _RV)[sl])
        for p in (self.distance, self.pm_ra, self.pm_dec, self.rv):
            p.element_links = None
        if "distance" in links:
            self.distance.element_links = {"hard": {2: _link(1)}}
        for name in ("pm_ra", "pm_dec", "rv"):
            if name in links:
                getattr(self, name).element_links = {
                    "hard": {2: _pm_link(1, name)}
                }

    def position_root(self, i):
        # No derived lens positions here (Star.position_root).
        return i


class _Sys:
    def __init__(self, stars):
        self.star = stars
        # No component models a star's mass loss (Component.initial_logmass).
        self.active_components = {}


def _kin(stars):
    gm = GalacticModel([{"anchor_idx": 1}], _DummyConfigManager())
    with pm.Model() as model:
        gm.build_likelihood(model, _Sys(stars))
    return model, gm


def _kin_value(stars):
    model, gm = _kin(stars)
    return float(model["galacticmodel.kinematic_prior"].eval()), gm


# ----------------------------------------------------------------------
# 1. The static rule
# ----------------------------------------------------------------------
def test_linked_distance_joins_its_partners_group_and_chains_resolve():
    """
    Given star 2 hard-linked to star 1's distance and star 3 to star 2's,
    When the structure is read,
    Then both share star 1's position (the chain is followed to its root),
      star 0 is its own root, and nothing is refused.
    """
    stars = _Stars(3)
    stars.distance = _MockParam([1.0, 2.0, 2.0, 2.0])
    stars.distance.element_links = {"hard": {2: _link(1), 3: {**_link(2)}}}
    root_of, vel, refused = linked_star_structure(stars, 4)
    assert list(root_of) == [0, 1, 1, 1]
    assert vel == {}
    assert refused == []


def test_link_reading_anything_but_one_distance_is_refused():
    """
    Given a hard-linked distance whose expression also reads another
      parameter (or two stars' distances),
    When the structure is read,
    Then the star stays its own root and is reported as refused -- there is
      no single partner whose position it is.
    """
    stars = _Stars(3)
    stars.distance.element_links = {
        "hard": {
            2: {
                "fn": None,
                "intra_deps": {1},
                "dep_paths": ("star.1.distance", "mulensevent.0.pi_rel"),
            }
        }
    }
    root_of, _, refused = linked_star_structure(stars, 3)
    assert list(root_of) == [0, 1, 2]
    assert [k for k, _ in refused] == [2]

    stars.distance.element_links = {
        "hard": {
            2: {
                "fn": None,
                "intra_deps": {0, 1},
                "dep_paths": ("star.0.distance", "star.1.distance"),
            }
        }
    }
    root_of, _, refused = linked_star_structure(stars, 3)
    assert list(root_of) == [0, 1, 2]
    assert [k for k, _ in refused] == [2]


def test_velocity_links_are_recorded_per_coordinate():
    stars = _Stars(3, links=("distance", "pm_ra", "pm_dec"))
    root_of, vel, _ = linked_star_structure(stars, 3)
    assert list(root_of) == [0, 1, 1]
    assert vel == {2: (0, 1)}


# ----------------------------------------------------------------------
# 2. Linked members under the genulens port
# ----------------------------------------------------------------------
def test_unlinked_third_star_still_adds_its_own_full_term():
    """The control: without links, a third star is a third position."""
    two, _ = _kin_value(_Stars(2))
    three, _ = _kin_value(_Stars(3))
    # a second density factor on D_s, not a constant -- the 1L2S bias this
    # file is about
    assert abs(three - two) > 1.0


def test_partly_linked_velocity_raises():
    """
    Given Source2 hard-linked to Source on distance, pm_ra and pm_dec but
      not rv,
    When the kinematic prior is built,
    Then it raises: the conditional density of rv given the linked proper
      motions has no closed form under genulens's non-Gaussian disk
      kinematics, and a silently wrong density is worse than none.
    """
    with pytest.raises(NotImplementedError, match="partly linked"):
        _kin(_Stars(3, links=("distance", "pm_ra", "pm_dec")))


# ----------------------------------------------------------------------
# 3. The normalizer counts independent positions
# ----------------------------------------------------------------------
def test_evidence_log_norm_counts_independent_positions():
    """
    Given sampled distances on the default bounds,
    When evidence_log_norm is read,
    Then it is -n_root * (log Z_pos - log 1e-3) with the port's own
      position normalizer, a fully linked star adding nothing.
    """
    l_deg, b_deg = k21.line_of_sight_basis(_RA, _DEC)[:2]
    per_root = k21.log_position_norm(1e-3, 1e5, l_deg, b_deg, None) - np.log(
        1e-3
    )
    full = ("distance", "pm_ra", "pm_dec", "rv")
    for links, n_root in (((), 3), (full, 2)):
        stars = _Stars(3, links=links)
        stars.distance.lower, stars.distance.upper = 1e-3, 1e5
        stars.distance.is_sampled = [True, True, not links]
        _, gm = _kin(stars)
        assert gm.evidence_log_norm() == pytest.approx(-n_root * per_root)


# ----------------------------------------------------------------------
# 4. End to end: 1L1S vs 1L2S of the same event
# ----------------------------------------------------------------------
def _event_config(arch):
    stars = [{"name": "Lens"}, {"name": "Source"}]
    sources = [{"body": "star.Source", "name": "Source"}]
    if arch == "1L2S":
        # Source2 on the FFP mass function: the port gives a fully linked
        # member its own MASS term (a companion has a mass of its own), so
        # this isolates the position/velocity accounting this test is about.
        # The mass term is tests/test_koshimoto21_model.py's.
        stars.append({"name": "Source2", "mass_function": "ffp"})
        sources.append({"body": "star.Source2", "name": "Source2"})
    return {
        "name": "linked_" + arch,
        "star": stars,
        "mulensevent": [{}],
        "lens": [{"body": "star.Lens", "name": "Lens"}],
        "source": sources,
        "galacticmodel": [{"anchor_idx": 1}],
    }


def _event_params(arch, link_rv):
    p = {
        "source.Source.t_0": {"initval": 2460025.0},
        "source.Source.u_0": {"initval": 0.2},
        "star.Lens.distance": {"initval": _D[0], "sigma": 0},
        "star.Source.distance": {"initval": _D[1], "sigma": 0},
        "star.Lens.mass": {"initval": 0.5},
        "star.Lens.pm_ra": {"initval": _PM_RA[0], "sigma": 0},
        "star.Lens.pm_dec": {"initval": _PM_DEC[0], "sigma": 0},
        "star.Source.pm_ra": {"initval": _PM_RA[1], "sigma": 0},
        "star.Source.pm_dec": {"initval": _PM_DEC[1], "sigma": 0},
        "star.Lens.rv": {"initval": _RV[0], "sigma": 0},
        "star.Source.rv": {"initval": _RV[1], "sigma": 0},
    }
    for s in ("Lens", "Source", "Source2"):
        p[f"star.{s}.ra"] = {"initval": _RA_DEG, "sigma": 0}
        p[f"star.{s}.dec"] = {"initval": _DEC_DEG, "sigma": 0}
    if arch == "1L2S":
        p["source.Source2.t_0"] = {"initval": 2460030.0}
        p["source.Source2.u_0"] = {"initval": 0.5}
        for par in ("distance", "pm_ra", "pm_dec") + (
            ("rv",) if link_rv else ()
        ):
            p[f"star.Source2.{par}"] = {
                "initval": f"star.Source.{par}",
                "sigma": 0,
            }
    else:
        p = {k: v for k, v in p.items() if "Source2" not in k}
    return p


def _build_event(arch, link_rv=True):
    from exozippy.system import System

    system = System(
        _event_config(arch), user_params=_event_params(arch, link_rv)
    )
    system.prepare()
    model = system.build_model()
    return system, model


@pytest.mark.slow
def test_1l2s_and_1l1s_carry_the_same_galactic_contribution():
    """
    Given a 1L1S model and a 1L2S model of the same event, Source2's
      distance, proper motion and rv hard-linked to Source's (and Source2
      on the FFP mass function, so it owes no mass term here),
    When the galactic kinematic potential is evaluated at the same start,
    Then the two are identical, and so is evidence_log_norm: the 1L2S model
      has one source position, not two.  (Before the fix the 1L2S value
      carried a second density term on D_s.)
    """
    s1, m1 = _build_event("1L1S")
    s2, m2 = _build_event("1L2S", link_rv=True)

    def kin(model):
        # point_logps, not compile_fn on the potential: a compiled potential
        # still holds the RVs, which would be DRAWN, not read at the point.
        logps = model.point_logps(model.initial_point(), round_vals=14)
        return float(logps["galacticmodel.kinematic_prior"])

    np.testing.assert_allclose(kin(m2), kin(m1), rtol=0, atol=1e-8)
    assert s2.galacticmodel.evidence_log_norm() == pytest.approx(
        s1.galacticmodel.evidence_log_norm()
    )
    # The model's start values are the ones the comparison is about.
    d = s2.star.distance
    assert d.element_start(2) == pytest.approx(d.element_start(1))
