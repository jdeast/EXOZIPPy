"""star.pm_ra_sgra / pm_dec_sgra: the proper motion relative to Sgr A*.

mulensing/conventions.md C31.  star.pm_ra/pm_dec are ABSOLUTE (ICRS,
heliocentric) proper motions.  OGLE/KMT relative astrometry quotes a source
proper motion relative to the field (bulge/red-clump) mean, so a prior from
such a paper is written on the derived pm_*_sgra -- the absolute pm minus
Sgr A*'s apparent motion (Reid & Brunthaler 2020) on the local axes at the
star's position -- and never silently on the absolute pair.

Pinned here:
  * the closed-form rotation against an independent astropy transform at
    the OGLE-2014-BLG-0939 field and at several other sky positions;
  * Yee+2015's Eq. 8 value round-trips to the absolute (-8.43, -6.25);
  * a mu/sigma on the derived parameter is a Gaussian potential in the
    model logp, centred where the conversion says, with no Jacobian term;
  * a mu on the derived parameter starts the sampled absolute pm at the
    converted value (Star._seed_pm_from_sgra_prior);
  * the parameters are declared only on a microlensing / Galactic-model
    topology, and are reported (a trace node).
"""

import astropy.units as u
import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest
from astropy.coordinates import SkyCoord

from exozippy.components.star.physics import (
    SGRA_PM_B,
    SGRA_PM_L_COSB,
    calc_pm_dec_sgra,
    calc_pm_ra_sgra,
    sgra_pm_equatorial,
)
from exozippy.system import System

# OGLE-2014-BLG-0939 (examples/ob140939)
OB140939_RA = 266.801041667
OB140939_DEC = -21.3829722222
# Yee et al. 2015, ApJ 802, 76, Eq. 8: (N, E) = (-0.64, -5.31) mas/yr,
# relative to the field (bulge) mean.
YEE_PM_E, YEE_PM_N = -5.31, -0.64


def _astropy_sgra_pm(ra_deg, dec_deg):
    """Sgr A*'s (mu_l*, mu_b) re-expressed on the ICRS axes at (ra, dec),
    by astropy's frame machinery -- independent of the closed form."""
    g = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg).galactic
    icrs = SkyCoord(
        l=g.l,
        b=g.b,
        pm_l_cosb=SGRA_PM_L_COSB * u.mas / u.yr,
        pm_b=SGRA_PM_B * u.mas / u.yr,
        frame="galactic",
    ).icrs
    return (
        float(icrs.pm_ra_cosdec.to_value(u.mas / u.yr)),
        float(icrs.pm_dec.to_value(u.mas / u.yr)),
    )


# The event field; Sgr A*; Baade's window; a disk field on the far side of
# the GC; a northern high-latitude field; RA near the 0/360 wrap; a southern
# field near the SGP.
POSITIONS = [
    (OB140939_RA, OB140939_DEC),
    (266.41683708, -29.00780556),
    (270.9, -30.03),
    (283.0, 5.0),
    (150.0, 60.0),
    (359.9, 10.0),
    (12.0, -40.0),
]


@pytest.mark.parametrize("ra_deg,dec_deg", POSITIONS)
def test_rotation_matches_astropy(ra_deg, dec_deg):
    exp_e, exp_n = _astropy_sgra_pm(ra_deg, dec_deg)
    got_e, got_n = sgra_pm_equatorial(
        np.radians(ra_deg), np.radians(dec_deg), xp=np
    )
    assert got_e == pytest.approx(exp_e, abs=1e-6)
    assert got_n == pytest.approx(exp_n, abs=1e-6)
    # A rotation: the magnitude is Sgr A*'s everywhere.
    assert np.hypot(got_e, got_n) == pytest.approx(
        np.hypot(SGRA_PM_L_COSB, SGRA_PM_B), rel=1e-12
    )


def test_ob140939_field_value():
    """The number the C31 text and the example comment quote."""
    e, n = sgra_pm_equatorial(
        np.radians(OB140939_RA), np.radians(OB140939_DEC), xp=np
    )
    assert e == pytest.approx(-3.116, abs=1e-3)
    assert n == pytest.approx(-5.607, abs=1e-3)


@pytest.mark.parametrize("ra_deg,dec_deg", POSITIONS)
def test_pytensor_graph_equals_numpy(ra_deg, dec_deg):
    pm_ra, pm_dec, ra, dec = pt.dscalars("pm_ra", "pm_dec", "ra", "dec")
    f = pytensor.function(
        [pm_ra, pm_dec, ra, dec],
        [calc_pm_ra_sgra(pm_ra, ra, dec), calc_pm_dec_sgra(pm_dec, ra, dec)],
    )
    r, d = np.radians(ra_deg), np.radians(dec_deg)
    got = f(1.5, -2.5, r, d)
    e, n = sgra_pm_equatorial(r, d, xp=np)
    assert float(got[0]) == pytest.approx(1.5 - e, abs=1e-12)
    assert float(got[1]) == pytest.approx(-2.5 - n, abs=1e-12)


def test_sgra_itself_maps_to_zero():
    """Sgr A*'s own absolute proper motion, at its own position, is zero
    relative to Sgr A* (astropy supplies the absolute value)."""
    ra, dec = POSITIONS[1]
    abs_e, abs_n = _astropy_sgra_pm(ra, dec)
    f_ra = calc_pm_ra_sgra(abs_e, np.radians(ra), np.radians(dec)).eval()
    f_dec = calc_pm_dec_sgra(abs_n, np.radians(ra), np.radians(dec)).eval()
    assert abs(float(f_ra)) < 1e-6
    assert abs(float(f_dec)) < 1e-6


def test_yee_round_trips_to_absolute():
    r, d = np.radians(OB140939_RA), np.radians(OB140939_DEC)
    e, n = sgra_pm_equatorial(r, d, xp=np)
    abs_e, abs_n = YEE_PM_E + e, YEE_PM_N + n
    assert abs_e == pytest.approx(-8.43, abs=0.02)
    assert abs_n == pytest.approx(-6.25, abs=0.02)
    assert float(calc_pm_ra_sgra(abs_e, r, d).eval()) == pytest.approx(
        YEE_PM_E, abs=1e-12
    )
    assert float(calc_pm_dec_sgra(abs_n, r, d).eval()) == pytest.approx(
        YEE_PM_N, abs=1e-12
    )


# --- through the System ---------------------------------------------------

SIGMA = 0.45
_CONFIG = {
    "star": [{"name": "Source"}],
    "galacticmodel": [{"anchor_idx": 0}],
}
_POSITION = {
    "star.Source.ra": {"initval": OB140939_RA, "sigma": 0},
    "star.Source.dec": {"initval": OB140939_DEC, "sigma": 0},
}


def _build(extra):
    import copy

    system = System(copy.deepcopy(_CONFIG), user_params={**_POSITION, **extra})
    system.prepare()
    model = system.build_model()
    return system, model


@pytest.fixture(scope="module")
def with_prior():
    return _build(
        {
            "star.Source.pm_ra_sgra": {"mu": YEE_PM_E, "sigma": SIGMA},
            "star.Source.pm_dec_sgra": {"mu": YEE_PM_N, "sigma": SIGMA},
        }
    )


@pytest.fixture(scope="module")
def without_prior(with_prior):
    """Same start for the absolute pm, written by hand, no derived prior."""
    system, _ = with_prior
    return _build(
        {
            "star.Source.pm_ra": {
                "initval": float(system.star.pm_ra.element_start(0))
            },
            "star.Source.pm_dec": {
                "initval": float(system.star.pm_dec.element_start(0))
            },
        }
    )


def _eval(model, nodes, point):
    nodes = model.replace_rvs_by_values(nodes)
    f = pytensor.function(model.value_vars, nodes, on_unused_input="ignore")
    return [
        np.asarray(v) for v in f(*[point[v.name] for v in model.value_vars])
    ]


def test_mu_on_derived_starts_the_absolute_pm_converted(with_prior):
    system, _ = with_prior
    assert system.star.pm_ra.element_start(0) == pytest.approx(
        -8.4264, abs=1e-3
    )
    assert system.star.pm_dec.element_start(0) == pytest.approx(
        -6.2469, abs=1e-3
    )


def test_derived_prior_is_a_gaussian_in_the_logp(with_prior, without_prior):
    """The two models share every term but the derived-parameter Gaussian
    and start from the same absolute pm, so at any raw point their logp
    differ by exactly -0.5 z^2 summed over the two components -- centred at
    Yee's value in the Sgr A* frame, with no Jacobian term."""
    sys_w, m_w = with_prior
    _, m_wo = without_prior
    names = [p.name for p in m_w.potentials]
    assert "gaussian_prior.star.pm_ra_sgra" in names
    assert "gaussian_prior.star.pm_dec_sgra" in names
    assert not any("sgra" in p.name for p in m_wo.potentials)

    logp_w = m_w.compile_logp()
    logp_wo = m_wo.compile_logp()
    rng = np.random.default_rng(140939)
    base = m_w.initial_point()
    assert {k: np.shape(v) for k, v in base.items()} == {
        k: np.shape(v) for k, v in m_wo.initial_point().items()
    }
    for _ in range(5):
        point = {
            k: v + rng.normal(0.0, 30.0, np.shape(v)) for k, v in base.items()
        }
        sg_ra, sg_dec, pm_ra, pm_dec = _eval(
            m_w,
            [
                m_w["star.pm_ra_sgra"],
                m_w["star.pm_dec_sgra"],
                m_w["star.pm_ra"],
                m_w["star.pm_dec"],
            ],
            point,
        )
        e, n = _astropy_sgra_pm(OB140939_RA, OB140939_DEC)
        assert float(sg_ra[0]) == pytest.approx(float(pm_ra[0]) - e, abs=1e-6)
        assert float(sg_dec[0]) == pytest.approx(
            float(pm_dec[0]) - n, abs=1e-6
        )
        expected = -0.5 * (
            ((float(sg_ra[0]) - YEE_PM_E) / SIGMA) ** 2
            + ((float(sg_dec[0]) - YEE_PM_N) / SIGMA) ** 2
        )
        assert float(logp_w(point)) - float(logp_wo(point)) == pytest.approx(
            expected, rel=1e-9, abs=1e-9
        )


def test_reported_both_frames(with_prior):
    """The derived pair is a trace node, so results report both frames."""
    system, model = with_prior
    det = {d.name for d in model.deterministics}
    assert {"star.pm_ra_sgra", "star.pm_dec_sgra"} <= det
    assert system.star.pm_ra_sgra.print_to_table
    assert system.star.pm_dec_sgra.print_to_table


def test_not_declared_off_the_galactic_topologies():
    """A plain star (no microlensing event, no Galactic model) has no
    Sgr A* frame parameters."""
    system = System({"star": [{"name": "A"}]}, user_params={})
    system.prepare()
    assert "pm_ra_sgra" not in system.star.manifest
    assert "pm_dec_sgra" not in system.star.manifest
