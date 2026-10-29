"""RV unit boundaries: a user ``unit:`` on an RV quantity must not rescale
a start value the code computed in m/s.

Review 7.16.2.  ``ConfigManager.add_hint`` takes its value in the
parameter's USER unit and multiplies by ``get_conversion_factor``, which
honors a user ``unit:`` override (tests/test_config_units.py pins that
contract).  ``RVInstrument.register_parameters`` pushes ``gamma_init`` --
the mean of the file's RVs converted to m/s -- through that channel as if
it were in the user's unit, so with ``unit: km/s`` on gamma the start is
1000x too large.  ``Planet._push_rv_mass_hints`` does the same with K.

The known answer needs no constant of its own: the internal start must be
the same number whatever unit the user READS gamma in, because the data
file did not change.
"""

import astropy.units as u
import pytest

from exozippy.components.rvinstrument.rvinstrument import RVInstrument
from exozippy.config import ConfigManager

RVS_MS = [12000.0, 12001.0, 12002.0, 12003.0, 12004.0]


def _gamma_hint(tmp_path, user_params):
    data = tmp_path / "inst.rv"
    data.write_text(
        "".join(f"{2457000.0 + i} {rv} 1.0\n" for i, rv in enumerate(RVS_MS))
    )
    cm = ConfigManager(user_params)
    inst = RVInstrument([{"name": "Inst", "file": str(data)}], cm)
    inst.load_data(system=None)
    inst.register_parameters(system=None)
    return cm.hints["rvinstrument.0.gamma"]


def test_gamma_start_is_the_mean_rv_in_internal_units(tmp_path):
    """
    Given an RV file in m/s (the default unit) and no user unit,
    When the instrument hints gamma's start,
    Then the internal (solRad/d) start is the mean RV converted by astropy,
      and converting it back gives the mean in m/s (round trip).
    """
    mean_ms = sum(RVS_MS) / len(RVS_MS)
    internal = _gamma_hint(tmp_path, {})
    expected = (mean_ms * u.m / u.s).to_value(u.solRad / u.d)
    assert internal == pytest.approx(expected, rel=1e-14)
    assert (internal * u.solRad / u.d).to_value(u.m / u.s) == pytest.approx(
        mean_ms, rel=1e-14
    )


@pytest.mark.xfail(
    strict=True,
    reason="review 1.5.10: the m/s gamma hint is scaled by the user's "
    "unit: override, so unit: km/s starts gamma 1000x too large",
)
def test_gamma_start_does_not_depend_on_the_users_display_unit(tmp_path):
    """
    Given the same RV file, once with no override and once with
      `rvinstrument.0.gamma: {unit: km/s}`,
    When the instrument hints gamma's start,
    Then the internal start is identical -- the unit only changes how the
      user reads gamma, not where the data put it.
    """
    plain = _gamma_hint(tmp_path, {})
    in_kms = _gamma_hint(tmp_path, {"rvinstrument.0.gamma": {"unit": "km/s"}})
    assert in_kms == pytest.approx(plain, rel=1e-14)
