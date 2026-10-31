"""The present-day mass function of the galactic model (review 1.7.2).

galacticmodel/pdmf.py builds, per population, the IMF evolved to today
(MIST turnoff, Kalirai+2008 white dwarfs, Raithel+2018 neutron stars and
black holes, age marginalized).  These tests pin its normalization, its
remnant bookkeeping against independent quadratures, the white-dwarf mass
range, and finite gradients (C and JAX) through the turnoff and the R18
mass gaps; and, through a System build, that ``present_day: true`` moves
the mass prior inside the population mixture.
"""

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from conftest import _DummyConfigManager
from exozippy.components.galacticmodel import galacticmodel as gm
from exozippy.components.galacticmodel import pdmf, populations


def _chabrier(x):
    return gm.chabrier_logmass_logp(x, param=None, xp=np)


@pytest.fixture(scope="module")
def pd_default():
    return pdmf.PresentDayMassFunction(_chabrier, [-9.0, -3.0], [2.5, 1.0])


def test_each_population_normalizes_over_each_support(pd_default):
    """
    Given the PDMF built for two stars with different logmass supports,
    When each population's density is integrated over each star's support,
    Then every integral is 1.
    """
    for k, (lo, hi) in enumerate(((-9.0, 2.5), (-3.0, 1.0))):
        x = np.linspace(lo, hi, 400001)
        for b in range(3):
            z = np.trapezoid(np.exp(pd_default.log_density_np(b, x, k)), x)
            assert z == pytest.approx(1.0, abs=1e-5), (k, b, z)


def test_single_age_remnant_fraction_matches_turnoff_quadrature(monkeypatch):
    """
    Given a single-age 10 Gyr population ([Fe/H] = -0.5, the thick-disk
      branch with its age set to 10 Gyr),
    When the PDMF's remnant number and mass fractions are measured,
    Then they equal an independent quadrature of the Chabrier IMF above
      M_die(10 Gyr) = 0.942 Msun (design table: 0.94) mapped through
      Kalirai+2008 / Raithel+2018, to the smoothing's precision.
    """
    # A (near) single age through the shared definition (populations.py):
    # 10 +/- 0.01 Gyr is a step in age on the lifetime scale.
    monkeypatch.setattr(populations, "THICK_AGE_MEAN_GYR", 10.0)
    monkeypatch.setattr(populations, "THICK_AGE_SIGMA_GYR", 0.01)
    monkeypatch.setitem(populations.POPULATION_FEH, "thick_disk", -0.5)
    pd = pdmf.PresentDayMassFunction(_chabrier, [-9.0], [2.5])
    d = pd.dens["thick_disk"]
    sel = (pd.x >= -9.0) & (pd.x <= 2.5)
    x = pd.x[sel]
    num = {k: np.trapezoid(v[sel], x) for k, v in d.items()}
    dark = (num["WD"] + num["NS"] + num["BH"]) / sum(num.values())

    # Independent: the turnoff from the lifetime table by root finding, the
    # IMF integrated on its own grid.
    from scipy.optimize import brentq

    log_mdie = brentq(
        lambda lm: pdmf.log10_lifetime_gyr(lm, -0.5) - 1.0, -0.3, 0.5
    )
    assert 10**log_mdie == pytest.approx(0.942, abs=0.003)
    xi = np.linspace(-9.0, 2.5, 2_000_001)
    imf = np.exp(_chabrier(xi))
    expected = np.trapezoid(imf[xi > log_mdie], xi[xi > log_mdie]) / (
        np.trapezoid(imf, xi)
    )
    assert dark == pytest.approx(expected, rel=2e-3)

    # The remnant MASS: WDs from the K08 line, NS/BH from the R18 tables.
    m = 10**xi
    dead = xi > log_mdie
    wd = dead & (m < 9.0)
    hv = dead & (m >= 9.0)
    p_ns = pdmf._p_ns(m)
    mass_rem = np.trapezoid(
        imf * np.where(wd, 0.109 * m + 0.394, 0.0)
        + imf
        * np.where(
            hv, p_ns * pdmf._ns_mass(m) + (1 - p_ns) * pdmf._bh_mass(m), 0
        ),
        xi,
    )
    mass_lum = np.trapezoid(np.where(dead, 0.0, imf * m), xi)
    got = {k: np.trapezoid((v * 10**pd.x)[sel], x) for k, v in d.items()}
    got_frac = (got["WD"] + got["NS"] + got["BH"]) / sum(got.values())
    assert got_frac == pytest.approx(
        mass_rem / (mass_rem + mass_lum), rel=5e-3
    )
    # ~13% of the objects of an old population are remnants, holding ~1/3
    # of its present-day mass -- the familiar order (e.g. Gould 2000, ~20%
    # of bulge events from remnants).
    assert 0.08 < dark < 0.15
    assert 0.2 < got_frac < 0.35


def test_bulge_white_dwarf_mass_distribution(pd_default):
    """
    Given the bulge population (9 +/- 1 Gyr, [Fe/H] = 0),
    When its white-dwarf density is examined,
    Then it peaks at 0.109 M_die + 0.394 ~ 0.51-0.55 Msun and has no mass
      below K08's floor or above its 1.375 Msun ceiling beyond the 0.02 dex
      smoothing (3 sigma).
    """
    wd = pd_default.dens["bulge"]["WD"]
    m = 10**pd_default.x
    peak = m[np.argmax(wd)]
    assert 0.50 < peak < 0.56
    tail = 10 ** (3 * pdmf.SMOOTH_DEX)
    assert wd[m > 1.375 * tail].max() < 1e-2 * wd.max()
    assert wd[m < 0.49 / tail].max() < 1e-2 * wd.max()


def test_bulge_rate_weighted_remnant_fractions(pd_default):
    """
    Given the bulge PDMF weighted by the event rate's sqrt(M),
    When the rate-weighted lens type fractions and quantiles are computed,
    Then they reproduce the design's analytic preview (Chabrier system,
      PDMF + remnants: M_L 0.088 / 0.421 / 0.90, 5.9% above 1 Msun) and a
      remnant fraction near genulens's 21% (WD 17.0, NS 2.5, BH 1.4 with
      the K21 IMF).
    """
    b = pdmf.POPULATIONS.index("bulge")
    x = np.linspace(-9.0, 2.5, 115001)
    m = 10**x
    rate = np.exp(pd_default.log_density_np(b, x)) * np.sqrt(m)
    c = np.cumsum(rate) / rate.sum()
    q = [m[np.searchsorted(c, p)] for p in (0.1, 0.5, 0.9)]
    assert q == pytest.approx([0.088, 0.421, 0.90], rel=0.03)
    frac = pd_default.type_fractions(b, x)
    rf = {t: (f * rate).sum() / rate.sum() for t, f in frac.items()}
    assert 0.15 < rf["WD"] < 0.18
    assert 0.005 < rf["NS"] < 0.03
    assert 0.008 < rf["BH"] < 0.02
    assert 0.17 < 1 - rf["lum"] < 0.23


def _mixture_logp(pd, x):
    """A one-star branch mixture of the PDMF terms, as the model builds it."""
    terms = [
        float(lnd)
        + pdmf.interp_table(pd.log_table[b], pd.x0, pdmf.GRID_DEX, x)
        - pd.log_norm[b, 0]
        - pd.log_lum_frac[b, 0]
        for b, lnd in enumerate((-1.0, -3.0, 0.5))
    ]
    return pt.logsumexp(pt.stack(terms), axis=0)


def _grid_through_turnoff_and_gaps():
    # Turnoffs (~0.9-2.3 Msun), the WD ceiling (1.375), the NS band
    # (1.2-2.1), the NS/BH gap and the R18 BH-mass bins (2-16 Msun), and
    # the upper support edge.
    return np.concatenate(
        [
            np.linspace(-0.2, 0.45, 1301),
            np.linspace(0.45, 1.3, 851),
            np.linspace(2.3, 2.5, 51),
            [-9.0, -5.0],
        ]
    )


@pytest.mark.parametrize("mode", ["FAST_RUN", "JAX"])
def test_gradient_finite_through_turnoff_and_remnant_gaps(pd_default, mode):
    """
    Given the PDMF mixture as the model builds it (table interpolation,
      logsumexp over populations),
    When its gradient is evaluated on a dense logmass grid through the
      turnoffs, the white-dwarf edge and the R18 NS/BH gaps,
    Then every logp and gradient is finite on the C and the JAX backends
      (the where-trap would show up as NaN only on JAX), and the gradient
      agrees with a finite difference.
    """
    if mode == "JAX":
        pytest.importorskip("jax")
    xv = pt.dvector("x")
    lp = _mixture_logp(pd_default, xv)
    g = pytensor.grad(pt.sum(lp), xv)
    f = pytensor.function([xv], [lp, g], mode=mode)
    grid = _grid_through_turnoff_and_gaps()
    val, grad = (np.asarray(a) for a in f(grid))
    assert np.all(np.isfinite(val))
    assert np.all(np.isfinite(grad))
    # Smooth: a 2e-3 dex central difference matches the analytic gradient
    # away from table nodes to the table's own resolution.
    h = 1e-3
    fd = (f(grid + h)[0] - f(grid - h)[0]) / (2 * h)
    inside = (grid > -8.9) & (grid < 2.49)
    assert np.max(np.abs(fd - grad)[inside]) < 2.0


def _baade_system(present_day):
    import sys
    from pathlib import Path

    sys.path.insert(
        0, str(Path(__file__).resolve().parents[1] / "examples" / "validation")
    )
    import prior_predictive_tE as ppt

    from exozippy.system import System

    ra, dec = ppt._radec(1.0, -3.9)
    config, params = ppt.build_config(
        ra, dec, present_day=present_day, model="simplified"
    )
    system = System(config, user_params=params)
    system.prepare()
    return system, system.build_model()


def test_present_day_must_be_a_boolean():
    """
    Given present_day spelled as a string,
    When the galactic model is constructed,
    Then it raises rather than reading "false" as true.
    """
    with pytest.raises(ValueError, match="present_day must be true or false"):
        gm.GalacticModel(
            [{"model": "simplified", "present_day": "false"}],
            _DummyConfigManager(),
        )


@pytest.mark.slow
def test_present_day_moves_mass_prior_inside_the_mixture():
    """
    Given the Baade lens + source prior model with present_day on and off,
    When both are built and evaluated at the start,
    Then with it on the imf_prior potential is exactly zero (both stars'
      mass prior moved into the population mixture), the kinematic
      potential differs, p_remnant is a probability per star, the
      gradient is finite, and the evidence normalizer carries the
      object-density offsets (it moves, by less than 0.2 nats per star).
    """
    sys_on, m_on = _baade_system(True)
    sys_off, m_off = _baade_system(False)

    def pots(model):
        vals = model.compile_logp(vars=model.potentials, sum=False)(
            model.initial_point()
        )
        return {p.name: float(v) for p, v in zip(model.potentials, vals)}

    on, off = pots(m_on), pots(m_off)
    assert on["galacticmodel.imf_prior"] == 0.0
    assert off["galacticmodel.imf_prior"] != 0.0
    assert on["galacticmodel.kinematic_prior"] != pytest.approx(
        off["galacticmodel.kinematic_prior"]
    )
    assert "galacticmodel.p_remnant" not in m_off.named_vars
    p_rem = m_on.compile_fn(
        m_on["galacticmodel.p_remnant"],
        inputs=m_on.value_vars,
        on_unused_input="ignore",
    )(m_on.initial_point())
    p_rem = np.asarray(p_rem)
    assert p_rem.shape == (2,)
    assert np.all((p_rem >= 0.0) & (p_rem <= 1.0))
    assert np.all(np.isfinite(m_on.compile_dlogp()(m_on.initial_point())))
    d_ev = (
        sys_on.galacticmodel.evidence_log_norm()
        - sys_off.galacticmodel.evidence_log_norm()
    )
    assert -0.4 < d_ev < 0.0
