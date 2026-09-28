"""The mamajek relation: Pecaut & Mamajek's mean dwarf sequence as a
Teff(M) (and optionally R(M)) prior, in the graph as a clamped table lookup.

Why it exists is in components/mamajek/mamajek.py; what is tested here is
that the graph reproduces the shipped table, clamps outside it, stays
differentiable, and that the component wires exactly what its config says
(Teff only by default; radius on request; floors overridable; actionable
config errors).
"""

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from exozippy.components.mamajek import physics
from exozippy.components.mamajek.mamajek import Mamajek
from exozippy.components.star.mamajek import read_mamajek


def _fn(fn):
    x = pt.dvector("x")
    return pytensor.function([x], fn(x))


def test_the_graph_reproduces_the_shipped_table_at_its_rows():
    t = read_mamajek(minmass=0.078)
    m = np.asarray(t["Msun"], float)
    teff = np.asarray(t["Teff"], float)
    r = np.asarray(t["R_Rsun"], float)
    ok = (
        np.isfinite(m)
        & np.isfinite(teff)
        & np.isfinite(r)
        & (m >= physics.MSTAR_MIN)
        & (m <= physics.MSTAR_MAX)
    )
    lm = np.log10(m[ok])
    got_t = 10 ** _fn(physics.calc_mamajek_logteff)(lm)
    got_r = 10 ** _fn(physics.calc_mamajek_logradius)(lm)
    # the uniform grid resamples the table, so agreement is to the grid's
    # linear-interpolation error, well under the 4% floor
    assert np.max(np.abs(got_t / teff[ok] - 1)) < 0.01
    assert np.max(np.abs(got_r / r[ok] - 1)) < 0.02


def test_it_clamps_outside_the_tabulated_mass_range():
    f = _fn(physics.calc_mamajek_logteff)
    lo, hi = np.log10(physics.MSTAR_MIN), np.log10(physics.MSTAR_MAX)
    assert f(np.array([lo - 1.0]))[0] == pytest.approx(f(np.array([lo]))[0])
    assert f(np.array([hi + 1.0]))[0] == pytest.approx(f(np.array([hi]))[0])


def test_teff_rises_with_mass_and_is_differentiable():
    x = pt.dscalar("x")
    y = physics.calc_mamajek_logteff(x)
    g = pytensor.function([x], pytensor.grad(y, x))
    f = pytensor.function([x], y)
    grid = np.linspace(np.log10(0.1), np.log10(1.4), 40)
    vals = np.array([f(v) for v in grid])
    assert np.all(np.diff(vals) > 0)
    assert all(np.isfinite(g(v)) and g(v) >= 0 for v in grid)


class _FakeStar:
    names = ["Lens", "Source"]


class _FakeSystem:
    star = _FakeStar()


def _mamajek(cfg):
    comp = Mamajek(cfg, None)
    comp.load_data(_FakeSystem())
    return comp


def test_teff_is_the_default_and_radius_is_opt_in():
    comp = _mamajek(
        [{"star": "Lens"}, {"star": "Source", "constrain": ["teff", "radius"]}]
    )
    assert comp.star_indices == [0, 1]
    assert comp.constrain == [{"teff"}, {"teff", "radius"}]
    assert comp.teff_floor == [physics.TEFF_FLOOR] * 2


def test_floors_are_overridable_and_must_be_positive():
    comp = _mamajek([{"star": "Lens", "teff_floor": 0.02}])
    assert comp.teff_floor == [0.02]
    with pytest.raises(ValueError, match="teff_floor"):
        _mamajek([{"star": "Lens", "teff_floor": 0.0}])


def test_unknown_constrain_entries_are_named():
    with pytest.raises(ValueError, match="unknown 'constrain:'"):
        _mamajek([{"star": "Lens", "constrain": ["mass"]}])


def test_register_parameters_declares_nothing():
    comp = _mamajek([{"star": "Lens"}])
    comp.register_parameters(_FakeSystem())
    assert comp.manifest == {}
