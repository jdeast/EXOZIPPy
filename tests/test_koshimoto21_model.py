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
    assert gm.model == "simplified"


def test_kinematic_prior_is_the_ported_mixture():
    """The potential is logsumexp_b [log n_b + log p_b(v)] + 2 log d +
    2 log(K d) per star, built from koshimoto21's numpy functions in
    genulens's frames (present_day off, so no mass term)."""
    stars = _stars()
    _, model = _build(stars, present_day=False)
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


@pytest.mark.slow  # compiles the gradient graph (~80 s on a cold cache)
@pytest.mark.parametrize("present_day", [False, True])
def test_prior_and_gradient_are_finite(present_day):
    stars = _stars()
    _, model = _build(stars, present_day=present_day)
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
    grads = pytensor.function([], pytensor.grad(pot, params))()
    assert all(np.all(np.isfinite(g)) for g in grads)


def test_nsd_sight_line_raises():
    sky = SkyCoord(l=0.3 * u.deg, b=-1.6 * u.deg, frame="galactic").icrs
    stars = _stars()
    stars.ra = _MockParam(float(sky.ra.rad))
    stars.dec = _MockParam(float(sky.dec.rad))
    with pytest.raises(NotImplementedError, match="nuclear-stellar-disk"):
        _build(stars, present_day=False)


def test_position_normalizer_is_finite_and_ordered():
    l_deg, b_deg, _, _ = k21.line_of_sight_basis(_RA, _DEC)
    a = k21.log_position_norm(1.0, 8000.0, l_deg, b_deg)
    b = k21.log_position_norm(1.0, 16000.0, l_deg, b_deg)
    assert np.isfinite(a) and b > a
