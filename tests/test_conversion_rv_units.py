"""RV unit boundaries: a user ``unit:`` on an RV quantity must not rescale
a start value the code computed in m/s.

Review 7.16.2.  ``ConfigManager.add_hint`` takes its value in the
parameter's USER unit and multiplies by ``get_conversion_factor``, which
honors a user ``unit:`` override (tests/test_config_units.py pins that
contract).  ``RVInstrument.register_parameters`` pushed ``gamma_init`` --
the mean of the file's RVs converted to m/s -- through that channel as if
it were in the user's unit, so with ``unit: km/s`` on gamma the start was
1000x too large; ``Planet._push_rv_mass_hints`` did the same with K
(review 1.5.10).  Both now state the unit their number is in
(``add_hint(..., unit="m/s")``).

The known answer needs no constant of its own: the internal start must be
the same number whatever unit the user READS gamma in, because the data
file did not change.
"""

from types import SimpleNamespace

import astropy.units as u
import pytest

from exozippy.components.planet.planet import Planet
from exozippy.components.rvinstrument.rvinstrument import RVInstrument
from exozippy.config import ConfigManager

RVS_MS = [12000.0, 12001.0, 12002.0, 12003.0, 12004.0]


def _gamma_hint(tmp_path, user_params, file_unit=None):
    """gamma's internal start from a file of RVS_MS, written in m/s, or in
    km/s (the same velocities / 1000) with the file's own `unit: km/s`."""
    data = tmp_path / "inst.rv"
    scale = 1.0 if file_unit is None else (u.m / u.s).to(u.Unit(file_unit))
    data.write_text(
        "".join(
            f"{2457000.0 + i} {rv * scale!r} {scale!r}\n"
            for i, rv in enumerate(RVS_MS)
        )
    )
    entry = {"name": "Inst", "file": str(data)}
    if file_unit is not None:
        entry["unit"] = file_unit
    cm = ConfigManager(user_params)
    inst = RVInstrument([entry], cm)
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


@pytest.mark.parametrize(
    "user_params",
    [{}, {"rvinstrument.0.gamma": {"unit": "km/s"}}],
    ids=["default-unit", "user-km/s"],
)
def test_gamma_start_from_a_km_s_file_is_the_m_s_files(tmp_path, user_params):
    """
    Given the same velocities written once in m/s and once in km/s (the
      file's own `unit: km/s`), with and without a user `unit: km/s` on gamma,
    When the instrument hints gamma's start,
    Then the internal start is the m/s file's, and it reads back as the mean
      in km/s through the user-unit helper.
    """
    plain = _gamma_hint(tmp_path, {})
    from_kms = _gamma_hint(tmp_path, user_params, file_unit="km/s")
    assert from_kms == pytest.approx(plain, rel=1e-12)

    cm = ConfigManager({"rvinstrument.0.gamma": {"unit": "km/s"}})
    mean_kms = sum(RVS_MS) / len(RVS_MS) / 1000.0
    assert cm.from_internal("rvinstrument.0.gamma", from_kms) == pytest.approx(
        mean_kms, rel=1e-12
    )


def _k_hint(user_params, k_ms):
    """planet.0.K's internal start from Planet._push_rv_mass_hints, on one
    linear-mass planet on a Keplerian orbit."""
    cm = ConfigManager(user_params)
    planet = SimpleNamespace(
        config_manager=cm,
        prefix="planet",
        n_elements=1,
        names=["b"],
        orbit_map=[0],
        mass_parameterizations=["linear"],
        _MASS_KEYS=Planet._MASS_KEYS,
    )
    system = SimpleNamespace(
        active_components={"orbit": SimpleNamespace(types=["keplerian"])}
    )
    Planet._push_rv_mass_hints(planet, system, k_ms)
    return cm.hints["planet.0.K"]


def test_k_start_does_not_depend_on_the_users_display_unit():
    """
    Given the RV semi-amplitude estimate (computed in m/s), once with no
      override and once with `planet.0.K: {unit: km/s}`,
    When the planet hints K's start,
    Then the internal start is identical, and is 25 m/s by astropy.
    """
    plain = _k_hint({}, 25.0)
    in_kms = _k_hint({"planet.0.K": {"unit": "km/s"}}, 25.0)
    assert plain == pytest.approx(
        (25.0 * u.m / u.s).to_value(u.solRad / u.d), rel=1e-14
    )
    assert in_kms == pytest.approx(plain, rel=1e-14)


def test_a_stated_unit_on_a_parameter_with_no_internal_unit_raises():
    """
    Given a parameter whose defaults.yaml declares no internal_unit,
    When a component pushes a hint on it stating a unit,
    Then add_hint raises naming the path -- the component's claim cannot be
      honored and must not be silently read as the internal unit.
    """
    cm = ConfigManager({})
    with pytest.raises(ValueError, match=r"planet\.0\.max_ecc.*internal_unit"):
        cm.add_hint("planet.0.max_ecc", 0.1, unit="m/s")
