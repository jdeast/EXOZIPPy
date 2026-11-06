"""The present-day mass function building blocks (galacticmodel/pdmf.py).

``pdmf.build_population_tables`` turns an IMF and a per-branch turnoff
P(alive | M_ini) into the linear present-day densities (Kalirai+2008 white
dwarfs, Raithel+2018 neutron stars and black holes, 0.02 dex smoothing);
the genulens port (``koshimoto21.MassFunction``) feeds it genulens's PARSEC
turnoffs.  These tests pin the remnant bookkeeping against independent
quadratures, the white-dwarf mass range, and finite gradients (C and JAX)
through the turnoffs and the R18 mass gaps of the port's tables.

(Until review 1.7.6 this file also tested the removed 3-population model's
own present-day mass function, ``present_day: true``, with MIST turnoffs.)
"""

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from exozippy.components.galacticmodel import koshimoto21 as k21
from exozippy.components.galacticmodel import pdmf


def _k21_imf(x):
    return k21.log_imf(x)


def _single_age_tables(log_mdie, x_lo=-3.0, x_hi=2.5):
    """Tables of one branch whose stars die above a sharp turnoff."""
    return pdmf.build_population_tables(
        _k21_imf,
        x_lo,
        x_hi,
        alive={"old": lambda x: (x < log_mdie).astype(float)},
    )


def test_single_age_remnant_fraction_matches_turnoff_quadrature():
    """
    Given a single-age population dying above M_die = 0.94 Msun (a 10 Gyr
      turnoff), on K21's IMF,
    When the tables' remnant number and mass fractions are measured,
    Then they equal an independent quadrature of the IMF above M_die mapped
      through Kalirai+2008 / Raithel+2018, to the smoothing's precision.
    """
    log_mdie = np.log10(0.94)
    x_lo, x_hi = -3.0, 2.5
    grid, dens = _single_age_tables(log_mdie, x_lo, x_hi)
    d = dens["old"]
    sel = (grid >= x_lo) & (grid <= x_hi)
    x = grid[sel]
    num = {k: np.trapezoid(v[sel], x) for k, v in d.items()}
    dark = (num["WD"] + num["NS"] + num["BH"]) / sum(num.values())

    # Independent: the IMF integrated on its own grid.
    xi = np.linspace(x_lo, x_hi, 2_000_001)
    imf = np.exp(_k21_imf(xi))
    dead = xi > log_mdie
    expected = np.trapezoid(imf[dead], xi[dead]) / np.trapezoid(imf, xi)
    assert dark == pytest.approx(expected, rel=2e-3)

    # The remnant MASS: WDs from the K08 line, NS/BH from the R18 tables.
    m = 10**xi
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
    got = {k: np.trapezoid((v * 10**grid)[sel], x) for k, v in d.items()}
    got_frac = (got["WD"] + got["NS"] + got["BH"]) / sum(got.values())
    assert got_frac == pytest.approx(
        mass_rem / (mass_rem + mass_lum), rel=5e-3
    )


def test_white_dwarf_mass_distribution():
    """
    Given a population dying above M_die = 1.0 Msun (an old bulge),
    When its white-dwarf density is examined,
    Then it peaks at 0.109 M_die + 0.394 ~ 0.50-0.56 Msun and has no mass
      below K08's floor or above its 1.375 Msun ceiling beyond the 0.02 dex
      smoothing (3 sigma).
    """
    grid, dens = _single_age_tables(0.0)
    wd = dens["old"]["WD"]
    m = 10**grid
    peak = m[np.argmax(wd)]
    assert 0.50 < peak < 0.56
    tail = 10 ** (3 * pdmf.SMOOTH_DEX)
    assert wd[m > 1.375 * tail].max() < 1e-2 * wd.max()
    assert wd[m < 0.49 / tail].max() < 1e-2 * wd.max()


@pytest.fixture(scope="module")
def mass_function():
    return k21.MassFunction([-9.0], [2.5], remnants=True)


def _mixture_logp(mf, x):
    """A one-star mixture of the port's mass terms over branches and object
    groups, as the model combines them (table interpolation, logsumexp)."""
    weights = np.linspace(-3.0, 1.0, len(mf.branches))
    terms = [
        float(weights[j])
        + pdmf.interp_table(mf.log_table[g][j], mf.x0, pdmf.GRID_DEX, x)
        - mf.log_norm[j, 0]
        for j in range(len(mf.branches))
        for g in mf.GROUPS
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
def test_gradient_finite_through_turnoff_and_remnant_gaps(mass_function, mode):
    """
    Given the port's mass terms mixed as the model mixes them (table
      interpolation, logsumexp over branches and object groups),
    When their gradient is evaluated on a dense logmass grid through the
      turnoffs, the white-dwarf edge and the R18 NS/BH gaps,
    Then every logp and gradient is finite on the C and the JAX backends
      (the where-trap would show up as NaN only on JAX), and the gradient
      agrees with a finite difference.
    """
    if mode == "JAX":
        pytest.importorskip("jax")
    xv = pt.dvector("x")
    lp = _mixture_logp(mass_function, xv)
    g = pytensor.grad(pt.sum(lp), xv)
    f = pytensor.function([xv], [lp, g], mode=mode)
    grid = _grid_through_turnoff_and_gaps()
    val, grad = (np.asarray(a) for a in f(grid))
    assert np.all(np.isfinite(val))
    assert np.all(np.isfinite(grad))
    # Smooth: a 2e-3 dex central difference matches the analytic gradient
    # to the table's own resolution.  genulens's single-age turnoffs are
    # steps smoothed over 0.02 dex, so the slope reaches ~275 nats/dex on
    # their edges; there the 1e-3 dex table curvature costs ~1% (3 nats/dex
    # measured), hence the relative term.
    h = 1e-3
    fd = (f(grid + h)[0] - f(grid - h)[0]) / (2 * h)
    inside = (grid > -8.9) & (grid < 2.49)
    err = np.abs(fd - grad) - (2.0 + 0.02 * np.abs(grad))
    assert np.max(err[inside]) < 0.0
