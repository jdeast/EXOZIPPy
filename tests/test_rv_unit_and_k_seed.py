"""rvinstrument load_data defects from the 2026-08-08 code review.

2.3.1 -- the per-file ``unit:`` key is documented in ``config_schema`` but
crashed: the YAML string was stored raw and ``load_data`` then called
``.to()`` on it.  ``astrometryinstrument`` handles its ``sep_unit`` correctly
with ``u.Unit(...)``; rvinstrument now does the same.

This is load-time only, so these tests drive ``RVInstrument.load_data``
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
