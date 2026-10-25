"""The Galactic kinematic prior for LINKED stars (star.md, "Linked stars: one
position, one density term").

A star whose distance is hard-linked to another star's has no position of its
own, so the system gets ONE density and ONE d^2 volume element.  Before the
fix a binary-source (1L2S) model carried a second, non-constant density
factor on D_s through the linked Source2 and its ln Z was biased against
1L1S/2L1S.  These tests pin:

1. the static rule (``linked_star_structure``): who shares whose position,
   which velocity coordinates are linked, and what is refused;
2. a fully linked member contributes NOTHING (the potential equals the
   unlinked model's bit for bit);
3. a member with linked proper motions and a free rv contributes an EXACTLY
   normalized conditional density in rv (its integral over rv is 1), and the
   one-linked-coordinate case likewise over its two free coordinates;
4. a distance-only link gives the shared-population joint density, checked
   against an independent scipy implementation;
5. the position normalizer against an independent quadrature;
6. end to end on a System: a 1L1S and a 1L2S model of one event carry the
   identical galactic contribution and the identical ``evidence_log_norm``.
"""

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest

from conftest import _DummyConfigManager, _MockParam
from exozippy.components.galacticmodel.galacticmodel import (
    BRANCH_SIGMAS,
    BRANCHES,
    GalacticModel,
    _log_total_number_density,
    branch_log_number_densities,
    linked_star_structure,
    log_position_norm,
)
from exozippy.components.galacticmodel.physics import (
    galactic_xyz,
    line_of_sight_basis,
    mean_polar_velocity,
)
from exozippy.constants import K_VEL_CONVERSION

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
# 2. A fully linked member adds nothing
# ----------------------------------------------------------------------
def test_fully_linked_member_contributes_nothing():
    """
    Given Lens + Source, and the same plus a Source2 hard-linked to Source on
      distance, pm_ra, pm_dec AND rv (one physical system),
    When the kinematic prior is evaluated,
    Then the two potentials are equal -- before the fix Source2 added a
      second full density term, ~log n(D_s) D_s^2 f(v_s) on top.
    """
    two, gm2 = _kin_value(_Stars(2))
    three, gm3 = _kin_value(
        _Stars(3, links=("distance", "pm_ra", "pm_dec", "rv"))
    )
    np.testing.assert_allclose(three, two, rtol=0, atol=1e-9)
    # ...and the same number of independent positions: identical constant.
    assert gm3.evidence_log_norm() == pytest.approx(gm2.evidence_log_norm())


def test_unlinked_third_star_still_adds_its_own_full_term():
    """The control: without links, a third star is a third position."""
    two, _ = _kin_value(_Stars(2))
    three, _ = _kin_value(_Stars(3))
    # The unnormalized per-star term is ~+8.8 nats here: a second density
    # factor on D_s, not a constant -- the 1L2S bias this file is about.
    assert abs(three - two) > 1.0


# ----------------------------------------------------------------------
# 3. A partially linked member: an exactly normalized conditional
# ----------------------------------------------------------------------
def _member_term_fn(links, free):
    """Compiled function of the free velocity coordinates of star 2 ->
    kinematic potential(3 stars) - potential(2 stars)."""
    base, _ = _kin_value(_Stars(2))
    stars = _Stars(3, links=links)
    syms = {}
    for name in free:
        sym = pt.dscalar(name)
        syms[name] = sym
        p = getattr(stars, name)
        p.value = pt.set_subtensor(p.value[2], sym)
    model, _ = _kin(stars)
    fn = pytensor.function(
        [syms[n] for n in free], model["galacticmodel.kinematic_prior"]
    )
    return lambda *args: fn(*args) - base


def test_pm_linked_member_contributes_a_normalized_rv_density():
    """
    Given Source2 hard-linked to Source on distance, pm_ra and pm_dec but
      with a FREE rv (the DC2018 1L2S configuration),
    When the potential is integrated over Source2's rv,
    Then exp(potential - potential without Source2) integrates to 1: the
      member contributes the exactly normalized conditional density of its
      rv given the shared position, proper motion and population -- and so
      leaves ln Z unchanged.
    """
    f = _member_term_fn(("distance", "pm_ra", "pm_dec"), ["rv"])
    rv = np.linspace(-1.2e6, 1.2e6, 6001)  # m/s; dispersions ~1e5
    vals = np.exp([f(v) for v in rv])
    integral = np.trapezoid(vals, rv)
    assert integral == pytest.approx(1.0, abs=1e-6)


def test_one_linked_coordinate_leaves_a_normalized_two_d_density():
    """Same, with only pm_ra linked: the conditional over (pm_dec, rv)."""
    f = _member_term_fn(("distance", "pm_ra"), ["pm_dec", "rv"])
    d_kpc = _D[1] / 1e3
    pmd = np.linspace(-20.0, 20.0, 161)  # 4.74*8.2*20 = 780 km/s
    rv = np.linspace(-9e5, 9e5, 161)
    vals = np.array([[np.exp(f(a, b)) for b in rv] for a in pmd])
    integral = np.trapezoid(np.trapezoid(vals, rv, axis=1), pmd)
    assert K_VEL_CONVERSION * d_kpc * 20 > 600  # grid spans the Gaussians
    assert integral == pytest.approx(1.0, abs=1e-4)


# ----------------------------------------------------------------------
# 4. Distance-only link: the shared-population joint density
# ----------------------------------------------------------------------
def _independent_joint(d_pc, vels):
    """Independent numpy/scipy implementation of the DOCUMENTED density of a
    group sharing one position with independent velocities:

      sum_b n_b(d) prod_k N_b(u_k) |dv/dx_k|      (no d^2 here)

    in the sampled coordinates, normalized Gaussians, (K d)^2 * 1e-3 per star
    (and, for the member, |det Q| of the smoothed polar map).
    """
    from scipy.stats import multivariate_normal

    m_rot, v0, clcb, slcb, sb = line_of_sight_basis(_RA, _DEC)
    d_kpc = d_pc / 1e3
    x, y, z = galactic_xyz(d_kpc, clcb, slcb, sb)
    r = np.sqrt(x**2 + y**2 + 1e-6)
    dens = branch_log_number_densities(x, y, z, r, xp=np)
    out = []
    for b, name in enumerate(BRANCHES):
        s = np.asarray(BRANCH_SIGMAS[name], dtype=float)
        mean_u = np.asarray(mean_polar_velocity(r, name))
        lp = dens[b]
        for n, (pm_ra, pm_dec, rv) in enumerate(vels):
            xp = np.array(
                [
                    K_VEL_CONVERSION * pm_ra * d_kpc,
                    K_VEL_CONVERSION * pm_dec * d_kpc,
                    rv / 1e3,
                ]
            )
            v = v0 + m_rot @ xp
            c, sn = x / r, y / r
            u_vec = np.array(
                [v[0] * c + v[1] * sn, v[1] * c - v[0] * sn, v[2]]
            )
            lp += multivariate_normal(mean_u, np.diag(s**2)).logpdf(u_vec)
            lp += 2 * np.log(K_VEL_CONVERSION * d_kpc) + np.log(1e-3)
            if n > 0:
                # |det Q| of the smoothed polar map: carried exactly for a
                # MEMBER (its term must integrate to 1), left out of the
                # root's (star.md) -- 4e-4 nats here, 50 pc from the GC.
                lp += np.log((x**2 + y**2) / r**2)
        out.append(lp)
    return float(np.logaddexp.reduce(out))


def test_distance_only_link_is_the_shared_population_joint_density():
    """
    Given Source2 hard-linked to Source on distance ONLY (its proper motion
      and rv free and different),
    When the potential is evaluated,
    Then the Source group's part equals sum_b n_b(d) N_b(v_1) N_b(v_2) d^2
      (one position, one population, two independent velocity draws) up to
      exactly the root's documented constant -- checked against an
      independent scipy implementation.
    """
    pm_ra = [_PM_RA[0], _PM_RA[1], _PM_RA[1] + 0.8]
    pm_dec = [_PM_DEC[0], _PM_DEC[1], _PM_DEC[1] - 0.5]
    rv = [_RV[0], _RV[1], _RV[1] + 20000.0]
    stars = _Stars(3, links=("distance",), pm_ra=pm_ra, pm_dec=pm_dec, rv=rv)
    full, _ = _kin_value(stars)
    lens_only, _ = _kin_value(
        _Stars(1, d=[_D[0]], pm_ra=[pm_ra[0]], pm_dec=[pm_dec[0]], rv=[rv[0]])
    )
    got = full - lens_only

    d = _D[1]
    want = _independent_joint(
        d, [(pm_ra[1], pm_dec[1], rv[1]), (pm_ra[2], pm_dec[2], rv[2])]
    ) + 2 * np.log(d)
    # The root leaves (2 pi)^(3/2) and the 1e-3 to evidence_log_norm.
    root_const = 1.5 * np.log(2 * np.pi) - np.log(1e-3)
    assert got == pytest.approx(want + root_const, abs=1e-8)


# ----------------------------------------------------------------------
# 5. The position normalizer
# ----------------------------------------------------------------------
def test_position_normalizer_matches_an_independent_quadrature():
    """
    Given the anchor's sight line and the default [1 mpc, 100 kpc] support,
    When log Z_pos = log int n(d) d^2 dd is computed,
    Then it agrees with scipy.integrate.quad (piecewise, in d) and is
      converged against 4x the points.
    """
    from scipy.integrate import quad

    from exozippy.components.galacticmodel import galacticmodel as gm_mod

    _, _, clcb, slcb, sb = line_of_sight_basis(_RA, _DEC)
    got = log_position_norm(1e-3, 1e5, clcb, slcb, sb)

    def integrand(d):
        return float(
            np.exp(_log_total_number_density(d, clcb, slcb, sb)) * d**2
        )

    edges = [1e-3, 100, 2000, 5000, 7000, 8000, 9000, 10000, 12000, 20000, 1e5]
    total = sum(
        quad(integrand, a, b, limit=400, epsrel=1e-11)[0]
        for a, b in zip(edges[:-1], edges[1:])
    )
    assert got == pytest.approx(np.log(total), abs=1e-7)

    n0 = gm_mod._POSITION_QUAD_POINTS
    try:
        gm_mod._POSITION_QUAD_POINTS = 4 * n0
        finer = log_position_norm(1e-3, 1e5, clcb, slcb, sb)
    finally:
        gm_mod._POSITION_QUAD_POINTS = n0
    assert abs(finer - got) < 1e-9


def test_evidence_log_norm_counts_independent_positions():
    """
    Given sampled distances on the default bounds,
    When evidence_log_norm is read,
    Then it is -n_root * (log Z_pos + 1.5 log 2pi - log 1e-3), with a
      distance-linked star adding nothing.
    """
    _, _, clcb, slcb, sb = line_of_sight_basis(_RA, _DEC)
    per_root = (
        log_position_norm(1e-3, 1e5, clcb, slcb, sb)
        + 1.5 * np.log(2 * np.pi)
        - np.log(1e-3)
    )
    for links, n_root in (((), 3), (("distance",), 2)):
        stars = _Stars(3, links=links)
        stars.distance.lower, stars.distance.upper = 1e-3, 1e5
        stars.distance.is_sampled = [True, True, "distance" not in links]
        _, gm = _kin(stars)
        assert gm.evidence_log_norm() == pytest.approx(-n_root * per_root)


# ----------------------------------------------------------------------
# 6. End to end: 1L1S vs 1L2S of the same event
# ----------------------------------------------------------------------
def _event_config(arch):
    stars = [{"name": "Lens"}, {"name": "Source"}]
    sources = [{"body": "star.Source", "name": "Source"}]
    if arch == "1L2S":
        stars.append({"name": "Source2"})
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
      distance, proper motion and rv hard-linked to Source's,
    When the galactic kinematic potential is evaluated at the same start,
    Then the two are identical, and so is evidence_log_norm: the 1L2S model
      has one source position, not two.  (Before the fix the 1L2S value
      carried a second log n(D_s) D_s^2 f(v_s) term.)
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
