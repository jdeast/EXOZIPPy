"""Two rvinstrument load_data defects from the 2026-08-08 code review.

2.3.1 -- the per-file ``unit:`` key is documented in ``config_schema`` but
crashed: the YAML string was stored raw and ``load_data`` then called
``.to()`` on it.  ``astrometryinstrument`` handles its ``sep_unit`` correctly
with ``u.Unit(...)``; rvinstrument now does the same.

2.3.3 -- ``k_init`` (the seed for ``planet.K``) was ``sqrt(2) * std`` of every
instrument's RVs concatenated raw.  The scatter that dominates there is the
constant offset BETWEEN instruments, not any planet's amplitude: a single
absolute-RV instrument at a ~30 km/s systemic velocity seeded K at ~21 km/s
for a 20 m/s planet.  Each file's own mean (its ``gamma_init``) is removed
first now.

Both are load-time only, so these tests drive ``RVInstrument.load_data``
directly rather than a whole ``System``.
"""

import astropy.units as u
import numpy as np
import pytest

from exozippy.components.rvinstrument.rvinstrument import RVInstrument
from exozippy.config import ConfigManager

_MS_PER_INTERNAL = (u.solRad / u.d).to(u.m / u.s)


def _write_rv(path, times, rvs, errs):
    """Write a whitespace RV file at full float64 precision."""
    np.savetxt(
        path,
        np.column_stack(
            [
                np.asarray(times, float),
                np.asarray(rvs, float),
                np.asarray(errs, float),
            ]
        ),
        header="time rv err",
        fmt="%.17g",
    )
    return str(path)


def _load(config):
    inst = RVInstrument(config, ConfigManager({}))
    inst.load_data(system=None)
    return inst


# ---------------------------------------------------------------------------
# 2.3.1 -- the documented per-file `unit:` key
# ---------------------------------------------------------------------------
_T = np.linspace(2455000.0, 2455010.0, 21)
_RV_MS = 50.0 * np.sin(2 * np.pi * (_T - 2455000.0) / 10.0) + 12.0
_ERR_MS = np.full(_T.size, 5.0)


def test_km_per_s_unit_key_loads_and_converts(tmp_path):
    """
    Given one RV file written in km/s and tagged ``unit: km/s``,
    When it is loaded,
    Then its internal arrays equal those of the identical data in m/s.
    """
    # ARRANGE
    kms = _write_rv(tmp_path / "a.rv", _T, _RV_MS / 1000.0, _ERR_MS / 1000.0)
    ms = _write_rv(tmp_path / "b.rv", _T, _RV_MS, _ERR_MS)

    # ACT
    tagged = _load([{"name": "A", "file": kms, "unit": "km/s"}])
    plain = _load([{"name": "B", "file": ms}])

    # ASSERT -- the values, not merely the absence of a crash
    np.testing.assert_allclose(tagged.rv, plain.rv, rtol=1e-12)
    np.testing.assert_allclose(tagged.err, plain.err, rtol=1e-12)
    np.testing.assert_allclose(tagged.gamma_init, [12.0], rtol=1e-12)
    np.testing.assert_allclose(
        tagged.rv * _MS_PER_INTERNAL, _RV_MS, rtol=1e-12
    )
    np.testing.assert_allclose(
        tagged.err * _MS_PER_INTERNAL, _ERR_MS, rtol=1e-12
    )
    # The jitter floor is quoted in m/s^2 too, so it must not scale with the
    # file's unit either.
    np.testing.assert_allclose(
        tagged.jittervar_lower, plain.jittervar_lower, rtol=1e-12
    )


def test_default_unit_is_metres_per_second(tmp_path):
    """
    Given an RV file with no ``unit:`` key,
    When it is loaded,
    Then the columns are read as m/s, as config_schema documents.
    """
    f = _write_rv(tmp_path / "a.rv", _T, _RV_MS, _ERR_MS)
    inst = _load([{"name": "A", "file": f}])
    assert inst.units[0] == u.Unit("m/s")
    np.testing.assert_allclose(inst.rv * _MS_PER_INTERNAL, _RV_MS, rtol=1e-12)


@pytest.mark.parametrize("bad", ["bananas", "kg", "d"])
def test_unparseable_or_non_velocity_unit_raises(tmp_path, bad):
    """
    Given a ``unit:`` key that is not a velocity astropy understands,
    When the component is constructed,
    Then a ValueError naming the instrument and the offending string is
    raised (rather than an AttributeError deep inside load_data).
    """
    f = _write_rv(tmp_path / "a.rv", _T, _RV_MS, _ERR_MS)
    with pytest.raises(ValueError, match=r"unit: .*is not a velocity"):
        RVInstrument(
            [{"name": "HIRES", "file": f, "unit": bad}], ConfigManager({})
        )


def test_astropy_unit_object_still_accepted(tmp_path):
    """
    Given an in-memory config whose ``unit:`` is already an astropy Unit,
    When the component is constructed,
    Then it is used as-is (run_fit accepts dict configs, not only YAML).
    """
    f = _write_rv(tmp_path / "a.rv", _T, _RV_MS / 1000.0, _ERR_MS / 1000.0)
    inst = _load([{"name": "A", "file": f, "unit": u.km / u.s}])
    np.testing.assert_allclose(inst.rv * _MS_PER_INTERNAL, _RV_MS, rtol=1e-12)


# ---------------------------------------------------------------------------
# 2.3.3 -- k_init must not see the offsets between instruments
# ---------------------------------------------------------------------------
_T2 = np.linspace(2455000.0, 2455010.0, 41)
_PLANET_MS = 20.0 * np.sin(2 * np.pi * (_T2 - 2455000.0) / 10.0)
_GAMMA_ABS_MS = 30000.0  # a 30 km/s absolute-RV instrument
_TRUE_K = float(np.sqrt(2.0) * np.std(_PLANET_MS))  # 19.7484... m/s


@pytest.fixture
def rv_pair(tmp_path):
    """One relative-RV file and one at a 30 km/s systemic velocity."""
    rel = _write_rv(tmp_path / "rel.rv", _T2, _PLANET_MS, np.full(41, 2.0))
    abs_ = _write_rv(
        tmp_path / "abs.rv", _T2, _PLANET_MS + _GAMMA_ABS_MS, np.full(41, 2.0)
    )
    return rel, abs_


def test_single_instrument_k_seed_is_unchanged(rv_pair):
    """
    Given a single relative-RV instrument (the control),
    When it is loaded,
    Then k_init is sqrt(2)*std of the RVs -- removing a constant mean cannot
    change a standard deviation, so this case is bit-for-bit as before.
    """
    rel, _ = rv_pair
    inst = _load([{"name": "REL", "file": rel}])
    assert inst.k_init == pytest.approx(_TRUE_K, rel=1e-12)


def test_absolute_rv_instrument_does_not_inflate_k_seed(rv_pair):
    """
    Given two instruments observing the same 20 m/s planet, one of them
    reporting absolute RVs offset by 30 km/s,
    When they are loaded together,
    Then k_init still lands at the planet's amplitude (~19.7 m/s) instead of
    the ~21 km/s the raw concatenated scatter used to give.
    """
    rel, abs_ = rv_pair

    inst = _load([{"name": "REL", "file": rel}, {"name": "ABS", "file": abs_}])

    np.testing.assert_allclose(
        inst.gamma_init, [0.0, _GAMMA_ABS_MS], atol=1e-9
    )
    assert inst.k_init == pytest.approx(_TRUE_K, rel=1e-9)
    # The pre-fix value, for the record: sqrt(2)*std of the raw union.
    buggy = float(
        np.sqrt(2.0)
        * np.std(np.concatenate([_PLANET_MS, _PLANET_MS + _GAMMA_ABS_MS]))
    )
    assert buggy > 21000.0
    assert inst.k_init < 0.01 * buggy


def test_k_seed_survives_a_unit_tagged_absolute_instrument(tmp_path):
    """
    Given the absolute-RV instrument reported in km/s (both fixes at once),
    When the pair is loaded,
    Then the seed is still the planet's amplitude.
    """
    rel = _write_rv(tmp_path / "rel.rv", _T2, _PLANET_MS, np.full(41, 2.0))
    abs_ = _write_rv(
        tmp_path / "abs.rv",
        _T2,
        (_PLANET_MS + _GAMMA_ABS_MS) / 1000.0,
        np.full(41, 0.002),
    )
    inst = _load(
        [
            {"name": "REL", "file": rel},
            {"name": "ABS", "file": abs_, "unit": "km/s"},
        ]
    )
    assert inst.k_init == pytest.approx(_TRUE_K, rel=1e-9)


# ---------------------------------------------------------------------------
# 2.3.3 -- degenerate inputs leave no scatter to measure
# ---------------------------------------------------------------------------
def test_constant_rvs_fall_back_to_the_median_error(tmp_path):
    """
    Given a file whose RVs are all identical,
    When it is loaded,
    Then k_init falls back to the median error bar rather than 0 (which the
    relaxation engine turns into a ~1e-20 Mjup planet).
    """
    f = _write_rv(tmp_path / "c.rv", _T2, np.full(41, 5.0), np.full(41, 3.0))
    assert _load([{"name": "C", "file": f}]).k_init == pytest.approx(3.0)


def test_one_point_per_file_falls_back_to_the_median_error(tmp_path):
    """
    Given two files of one point each, far apart in absolute RV,
    When they are loaded,
    Then the mean-removed residuals are all zero and the median error bar
    seeds K -- the offset between the two points never leaks in.
    """
    a = _write_rv(tmp_path / "a.rv", [2455000.0], [4.0], [2.5])
    b = _write_rv(tmp_path / "b.rv", [2455001.0], [9000.0], [2.5])
    inst = _load([{"name": "A", "file": a}, {"name": "B", "file": b}])
    assert inst.k_init == pytest.approx(2.5)


def test_zero_variance_and_zero_errors_fall_back_to_one(tmp_path):
    """
    Given constant RVs AND zero error bars,
    When the file is loaded,
    Then k_init is a strictly positive last-resort 1 m/s.
    """
    f = _write_rv(tmp_path / "z.rv", _T2, np.full(41, 5.0), np.zeros(41))
    assert _load([{"name": "Z", "file": f}]).k_init == pytest.approx(1.0)
