import pytest
from exozippy.config import ConfigManager
from exozippy.components.rv_instrument.rv_instrument import RVInstrument
from exozippy.components.parameter import Parameter


def test_duplicate_component_names_raise_value_error():
    """
    Given a configuration with multiple components sharing the exact same name,
    When the component array is initialized,
    Then a ValueError should be raised to prevent silent PyMC node overwrites.
    """
    # ARRANGE
    user_params = {}
    config_manager = ConfigManager(user_params)
    bad_config = [{"name": "HIRES", "file": "data1.txt"}, {"name": "HIRES", "file": "data2.txt"}]

    # ACT & ASSERT
    with pytest.raises(ValueError, match="Duplicate names found"):
        RVInstrument(bad_config, config_manager)


def test_invalid_string_as_numeric_parameter_raises_value_error():
    """
    Given a Parameter initialized with a non-numeric string,
    When the internal unit conversion executes,
    Then a ValueError should be raised before PyMC compilation occurs.
    """
    # ARRANGE
    label = "bad_init"
    bad_value = "not_a_number"

    # ACT & ASSERT
    with pytest.raises(ValueError):
        Parameter(label=label, initval=bad_value, internal_unit="m/s")


def test_unrecognized_astropy_unit_string_raises_value_error():
    """
    Given a Parameter initialized with a fictitious string for its unit,
    When the string is parsed by the astropy registry,
    Then Astropy should raise a ValueError indicating it did not parse as a unit.
    """
    # ARRANGE
    label = "bad_unit"
    fake_unit = "fake_unit_that_doesnt_exist"

    # ACT & ASSERT
    with pytest.raises(ValueError, match="did not parse as unit"):
        Parameter(label=label, unit=fake_unit)


def test_missing_instrument_data_file_raises_file_not_found_error(tmp_path):
    """
    Given an RV Instrument configured to read from a non-existent filepath,
    When the instrument attempts to load its pandas dataframe,
    Then a standard FileNotFoundError should be raised.
    """
    # ARRANGE
    config_manager = ConfigManager({})
    bad_config = [{"name": "GhostInst", "file": "this_file_does_not_exist.dat"}]
    inst = RVInstrument(bad_config, config_manager)

    # ACT & ASSERT
    with pytest.raises(FileNotFoundError):
        inst.load_data()