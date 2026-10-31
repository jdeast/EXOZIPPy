"""galacticmodel `model: koshimoto21` (the genulens port) as a PyMC prior:
the wiring, the frame, and the guards.  The port's pieces against genulens
itself are tests/test_koshimoto21.py."""

import numpy as np
import pymc as pm
import pytensor
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord

from conftest import _DummyConfigManager, _MockParam
from exozippy.components.galacticmodel import koshimoto21 as k21
from exozippy.components.galacticmodel.galacticmodel import GalacticModel

# examples/ob140939's sight line, outside genulens's NSD box
_SKY = SkyCoord(l=6.647997 * u.deg, b=3.611349 * u.deg, frame="galactic").icrs
_RA, _DEC = float(_SKY.ra.rad), float(_SKY.dec.rad)


class _Stars:
    def __init__(self, d_pc, pm_ra, pm_dec, rv_ms, logmass):
        self.ra = _MockParam(_RA)
        self.dec = _MockParam(_DEC)
        self.distance = _MockParam(
            np.asarray(d_pc, float), lower=1e-3, upper=1e5
        )
        self.pm_ra = _MockParam(np.asarray(pm_ra, float))
        self.pm_dec = _MockParam(np.asarray(pm_dec, float))
        self.rv = _MockParam(np.asarray(rv_ms, float))
        self.logmass = _MockParam(
            np.asarray(logmass, float), lower=-9.0, upper=2.5
        )
        # shared values, so the gradient test can differentiate through them
        for p in (self.distance, self.pm_ra, self.pm_dec, self.rv):
            p.value = pytensor.shared(p.initval)

    def position_root(self, i):
        return i


class _System:
    def __init__(self, stars):
        self.star = stars
        self.active_components = {}


def _build(stars, **cfg):
    gm = GalacticModel(
        [{"model": "koshimoto21", **cfg}], _DummyConfigManager()
    )
    system = _System(stars)
    with pm.Model() as model:
        gm.build_likelihood(model, system)
    return gm, model


def _stars():
    # a disk lens and a bulge source, near genulens's typical event
    return _Stars(
        [4000.0, 8300.0],
        [-3.0, -6.0],
        [-1.0, -0.5],
        [10e3, -30e3],
        [-0.5, 0.0],
    )


def test_model_is_an_enum_with_one_spelling():
    with pytest.raises(ValueError, match="koshimoto21"):
        GalacticModel([{"model": "Koshimoto21"}], _DummyConfigManager())
    gm = GalacticModel([{}], _DummyConfigManager())
    assert gm.model == "koshimoto21"


def test_kinematic_prior_is_the_ported_mixture():
    """The potential is logsumexp_b [log n_b + log p_b(v)] + 2 log d +
    2 log(K d) per star, built from koshimoto21's numpy functions in
    genulens's frames.  Both stars are on the FFP mass function, so the
    kinematic potential carries no mass term."""
    stars = _stars()
    stars.ffp_mask = np.array([True, True])
    stars.ffp_alpha = np.array([0.96, 0.96])
    _, model = _build(stars)
    got = float(model["galacticmodel.kinematic_prior"].eval())

    l_deg, b_deg, m, v0 = k21.line_of_sight_basis(_RA, _DEC)
    d = stars.distance.initval
    x, y, z = k21.dlb_to_xyz(d, l_deg, b_deg)
    cols = np.stack(
        [
            k21.K_KMS * stars.pm_ra.initval * d / 1e3,
            k21.K_KMS * stars.pm_dec.initval * d / 1e3,
            stars.rv.initval / 1e3,
        ]
    )
    v = v0[:, None] + m @ cols
    dens = k21.log_number_densities(x, y, z)
    velo = k21.log_velocity_densities(x, y, z, *v)
    terms = np.stack([dens[i] + velo[i] for i in k21.BRANCHES])
    want = np.sum(
        np.logaddexp.reduce(terms, axis=0)
        + 2.0 * np.log(d)
        + 2.0 * np.log(k21.K_KMS * d / 1e3)
    )
    np.testing.assert_allclose(got, want, rtol=1e-9)


def test_velocity_frame_is_genulens_projection():
    """A star at velocity v has genulens's (mu_l, mu_b) = ((v - v_sun) . e_l,
    (v - v_sun) . e_b) / (K D) (event_sampler.cpp), and the map has |det| 1."""
    l_deg, b_deg, m, v0 = k21.line_of_sight_basis(_RA, _DEC)
    assert abs(abs(np.linalg.det(m)) - 1.0) < 1e-12
    lr, br = np.radians(l_deg), np.radians(b_deg)
    v = np.array([30.0, 150.0, -20.0])
    d_kpc = 5.0
    w = np.linalg.solve(m, v - v0)  # (K pm_ra d, K pm_dec d, rv)
    pm_ra, pm_dec = w[:2] / (k21.K_KMS * d_kpc)
    rel = v - v0
    mu_l = (
        (rel[0] * np.sin(lr) + rel[1] * np.cos(lr)) * k21.KS2MY / (d_kpc * 1e3)
    )
    mu_b = (
        (
            rel[0] * np.cos(lr) * np.sin(br)
            - rel[1] * np.sin(lr) * np.sin(br)
            + rel[2] * np.cos(br)
        )
        * k21.KS2MY
        / (d_kpc * 1e3)
    )
    g = SkyCoord(
        ra=_RA * u.rad,
        dec=_DEC * u.rad,
        pm_ra_cosdec=pm_ra * u.mas / u.yr,
        pm_dec=pm_dec * u.mas / u.yr,
    ).galactic
    np.testing.assert_allclose(
        [g.pm_l_cosb.to_value(u.mas / u.yr), g.pm_b.to_value(u.mas / u.yr)],
        [mu_l, mu_b],
        rtol=1e-9,
    )


@pytest.mark.slow
# remnants on: the costlier graph (kicked NS/BH terms) covers the other.
# FAST_COMPILE (no C compilation) keeps this well inside the CI timeout
# on a cold cache; finiteness does not depend on the backend.
@pytest.mark.parametrize("remnants", [True])
def test_prior_and_gradient_are_finite(remnants):
    stars = _stars()
    _, model = _build(stars, remnants=remnants)
    pot = model["galacticmodel.kinematic_prior"]
    assert np.isfinite(float(pot.eval()))
    # gradient with respect to the star's coordinates, through the mock's
    # shared variables
    params = [
        stars.distance.value,
        stars.pm_ra.value,
        stars.pm_dec.value,
        stars.rv.value,
    ]
    grads = pytensor.function(
        [], pytensor.grad(pot, params), mode="FAST_COMPILE"
    )()
    assert all(np.all(np.isfinite(g)) for g in grads)


def test_nsd_sight_line_carries_the_nsd_branch():
    """Inside genulens's NSD box the mixture has 11 branches; a star at the
    Galactic center's distance there is mostly NSD."""
    sky = SkyCoord(l=0.3 * u.deg, b=-0.3 * u.deg, frame="galactic").icrs
    stars = _Stars([8160.0], [-6.0], [-0.5], [0.0], [-0.3])
    stars.ra = _MockParam(float(sky.ra.rad))
    stars.dec = _MockParam(float(sky.dec.rad))
    gm, model = _build(stars)
    assert gm._k21_branches == k21.BRANCHES_NSD
    assert np.isfinite(float(model["galacticmodel.kinematic_prior"].eval()))


def test_position_normalizer_is_finite_and_ordered():
    l_deg, b_deg, _, _ = k21.line_of_sight_basis(_RA, _DEC)
    a = k21.log_position_norm(1.0, 8000.0, l_deg, b_deg)
    b = k21.log_position_norm(1.0, 16000.0, l_deg, b_deg)
    assert np.isfinite(a) and b > a


@pytest.mark.parametrize("key", ["IMF", "present_day"])
def test_simplified_mass_keys_raise_under_koshimoto21(key):
    with pytest.raises(ValueError, match="koshimoto21 carries K21"):
        GalacticModel(
            [
                {
                    "model": "koshimoto21",
                    key: "chabrier" if key == "IMF" else True,
                }
            ],
            _DummyConfigManager(),
        )


def test_remnants_is_a_koshimoto21_key():
    with pytest.raises(ValueError, match="remnants is a key"):
        GalacticModel(
            [{"model": "simplified", "remnants": False}], _DummyConfigManager()
        )


def test_heavy_bulge_lens_needs_remnants():
    """A 1.6 Msun object in the bulge: above the bar's turnoff at its
    likely ages (~1.07 Msun at 9 Gyr), so with remnants it is mostly a
    neutron star; genulens's REMNANT 0 discards it."""

    def lp(remnants):
        stars = _Stars([8300.0], [-6.0], [-0.5], [-30e3], [np.log10(1.6)])
        _, model = _build(stars, remnants=remnants)
        return float(model["galacticmodel.kinematic_prior"].eval())

    assert lp(True) > lp(False) + 2.0


class _Selector:
    """A component that observed star ``k`` through a flux limit (and, when
    ``binary`` is given, treats star ``binary`` as a single lens whose mass
    may be an unresolved binary's)."""

    prefix = "selector"

    def __init__(self, k, binary=None):
        self.k = k
        self.binary = binary

    def initial_logmass(self, system):
        return {}

    def selected_sources(self, system):
        return [self.k]

    def unresolved_binary_lenses(self, system):
        return {} if self.binary is None else {self.binary: self.k}


def _ffp(stars):
    n = stars.distance.initval.size
    stars.ffp_mask = np.ones(n, dtype=bool)
    stars.ffp_alpha = np.full(n, 0.96)
    return stars


def _kin(stars, selected=None, **cfg):
    gm = GalacticModel(
        [{"model": "koshimoto21", **cfg}], _DummyConfigManager()
    )
    system = _System(stars)
    if selected is not None:
        system.active_components = {"selector": _Selector(selected)}
    with pm.Model() as model:
        gm.build_likelihood(model, system)
    return float(model["galacticmodel.kinematic_prior"].eval())


def test_selected_source_takes_genulens_source_weight():
    """A survey-selected source is weighted n_MS sqrt(D / 8 kpc) per branch
    (genulens's default), not n D^2: the potential changes by exactly the
    swap of those factors inside the mixture."""
    stars = _ffp(_Stars([8300.0], [-6.0], [-0.5], [-30e3], [-0.3]))
    lp_field = _kin(stars)
    lp_src = _kin(stars, selected=0)
    l_deg, b_deg, m, v0 = k21.line_of_sight_basis(_RA, _DEC)
    d = 8300.0
    x, y, z = k21.dlb_to_xyz(d, l_deg, b_deg)
    cols = np.array(
        [k21.K_KMS * -6.0 * d / 1e3, k21.K_KMS * -0.5 * d / 1e3, -30.0]
    )
    v = v0 + m @ cols
    velo = k21.log_velocity_densities(x, y, z, *v)
    n_all = k21.log_number_densities(x, y, z)
    n_ms = k21.log_number_densities(x, y, z, luminous_only=True)
    w = k21.log_source_weight_sqrt(d)
    br = k21.BRANCHES
    field = np.logaddexp.reduce(
        [float(np.ravel(n_all[i] + velo[i])[0]) for i in br]
    )
    src = np.logaddexp.reduce(
        [float(np.ravel(n_ms[i] + w + velo[i])[0]) for i in br]
    )
    assert lp_src - lp_field == pytest.approx(src - field, abs=1e-8)


def test_luminosity_function_selection_needs_ai_rc():
    with pytest.raises(ValueError, match="source_ai_rc"):
        GalacticModel(
            [
                {
                    "model": "koshimoto21",
                    "source_selection": "luminosity_function",
                }
            ],
            _DummyConfigManager(),
        )
    with pytest.raises(ValueError, match="keys of model: koshimoto21"):
        GalacticModel(
            [{"model": "simplified", "source_selection": "sqrt_distance"}],
            _DummyConfigManager(),
        )
    stars = _ffp(_Stars([8300.0], [-6.0], [-0.5], [-30e3], [-0.3]))
    lp = _kin(
        stars,
        selected=0,
        source_selection="luminosity_function",
        source_ai_rc=1.0,
    )
    assert np.isfinite(lp)


def _link_all(stars, member, root):
    spec = {
        "hard": {
            member: {
                "intra_deps": [root],
                "dep_paths": [f"star.{root}.distance"],
            }
        }
    }
    stars.distance.element_links = spec
    for name in ("pm_ra", "pm_dec", "rv"):
        getattr(stars, name).element_links = {
            "hard": {member: {"intra_deps": [root], "dep_paths": []}}
        }
    return stars


def test_fully_linked_member_adds_only_its_mass():
    """A binary companion whose distance and velocity are its root's: one
    density, one volume element, one velocity density for the system --
    with both on the FFP mass function (no mass term), the prior is the
    root's alone."""
    one = _ffp(_Stars([4000.0], [-3.0], [-1.0], [10e3], [-0.5]))
    two = _ffp(
        _Stars(
            [4000.0, 4000.0],
            [-3.0, -3.0],
            [-1.0, -1.0],
            [10e3, 10e3],
            [-0.5, -0.7],
        )
    )
    _link_all(two, member=1, root=0)
    assert _kin(two) == pytest.approx(_kin(one), abs=1e-9)


def test_binaries_change_only_the_unresolved_lens_mass_term():
    """binaries: true (genulens's BINARY 1) swaps the lens's luminous mass
    function for the effective binary-lens density; it is a koshimoto21
    key and a boolean."""
    with pytest.raises(ValueError, match="binaries is a key"):
        GalacticModel(
            [{"model": "simplified", "binaries": True}], _DummyConfigManager()
        )
    with pytest.raises(ValueError, match="binaries must be"):
        GalacticModel([{"binaries": "yes"}], _DummyConfigManager())

    def lp(binaries):
        stars = _Stars(
            [6000.0, 8300.0],
            [-3.0, -6.0],
            [-1.0, -0.5],
            [10e3, -30e3],
            [-0.3, 0.0],
        )
        gm = GalacticModel([{"binaries": binaries}], _DummyConfigManager())
        system = _System(stars)
        system.active_components = {"sel": _Selector(1, binary=0)}
        with pm.Model() as model:
            gm.build_likelihood(model, system)
        return float(model["galacticmodel.kinematic_prior"].eval())

    on, off = lp(True), lp(False)
    assert np.isfinite(on) and on != off
    # a 0.5 Msun lens: binaries move its mass density by a modest amount
    assert abs(on - off) < 1.0
