import numpy as np
import pytensor.tensor as pt
import pytest

from exozippy.components.parameter import Parameter
from exozippy.components.rvinstrument.rvinstrument import RVInstrument
from exozippy.components.star.star import Star
from exozippy.config import ConfigManager, validate_instance_names


@pytest.mark.parametrize(
    "bad_name, match",
    [
        ("My.Star", "letters, digits, underscores, and hyphens"),
        ("My Star", "letters, digits, underscores, and hyphens"),
        ("star[0]", "letters, digits, underscores, and hyphens"),
        ("", "letters, digits, underscores, and hyphens"),
        ("1", "index notation"),
        ("128", "index notation"),
        (128, "must be strings"),
        (1.5, "must be strings"),
    ],
)
def test_invalid_instance_name_raises_value_error(bad_name, match):
    """
    Given a config whose instance name would corrupt parameter-path parsing
      (dots, spaces, brackets, empty, all-digit, or non-string),
    When the ConfigManager is constructed with that system config,
    Then a ValueError naming the offending component entry is raised.
    """
    config = {"star": [{"name": bad_name}]}
    with pytest.raises(ValueError, match=match):
        ConfigManager({}, system_config=config)


def test_valid_instance_names_are_accepted():
    """
    Given names using the full legal charset (letters, digits, _ and -),
    When validate_instance_names runs,
    Then no error is raised (hyphenated survey names must keep working).
    """
    config = {
        "star": [{"name": "Lens"}, {"name": "Source"}],
        "mulensinstrument": [{"name": "Roman_Z087"}],
        "galacticmodel": [{"name": "KMT-2019-BLG-1806"}],
    }
    validate_instance_names(config)


def test_unnamed_instances_and_flat_components_are_skipped():
    """
    Given entries without a 'name' key and non-list (flat dict) components,
    When validate_instance_names runs,
    Then they are ignored rather than crashing the validator.
    """
    config = {
        "star": [{"teff": 5778}],  # unnamed instance → defaults to index
        "sed": {"errscale": 1.0},  # flat-dict component
        "prefix": "fitresults/model",  # non-component scalar key
    }
    validate_instance_names(config)


def test_duplicate_component_names_raise_value_error():
    """
    Given a configuration with multiple components sharing the exact same name,
    When the component array is initialized,
    Then a ValueError should be raised to prevent silent PyMC node overwrites.
    """
    # ARRANGE
    user_params = {}
    config_manager = ConfigManager(user_params)
    bad_config = [
        {"name": "HIRES", "file": "data1.txt"},
        {"name": "HIRES", "file": "data2.txt"},
    ]

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
    bad_config = [
        {"name": "GhostInst", "file": "this_file_does_not_exist.dat"}
    ]
    inst = RVInstrument(bad_config, config_manager)

    # ACT & ASSERT
    with pytest.raises(FileNotFoundError):
        inst.load_data(system=None)


import pymc as pm


def test_sampled_parameter_missing_bounds_raises_developer_error():
    """
    Given a Parameter that is actively being sampled (no expression, sigma > 0),
    When the parameter is materialized in the PyMC graph without explicit upper/lower bounds,
    Then it should raise a ValueError to force the developer to define the physical limits.
    """
    # ARRANGE
    # A sampled parameter with an init_scale but missing physical bounds
    p = Parameter(
        label="bad.sampled_param",
        initval=1.0,
        init_scale=0.1,
        lower=None,  # Missing!
        upper=None,  # Missing!
        unit="",
        internal_unit="",
    )

    # ACT & ASSERT
    with pm.Model():
        with pytest.raises(
            ValueError, match="MUST have explicit 'lower' and 'upper' bounds"
        ):
            p.build_pymc()


def test_derived_parameter_missing_bounds_is_allowed():
    """
    Given a Parameter that is derived (has an expression),
    When it is materialized in the PyMC graph without explicit bounds,
    Then it should build successfully, as derived parameters inherit constraints from their parents.
    """
    import pytensor.tensor as pt

    # ARRANGE
    dummy_node = pt.dscalar("dummy")
    p = Parameter(
        label="good.derived_param",
        expression=dummy_node * 2.0,
        lower=None,
        upper=None,
        unit="",
        internal_unit="",
    )

    # ACT & ASSERT
    with pm.Model():
        try:
            p.build_pymc()
        except ValueError:
            pytest.fail(
                "Developer guardrail incorrectly blocked a derived parameter!"
            )


# ---------------------------------------------------------------------------
# A pin must say WHAT it pins to (Parameter.build_pymc).
#
# `sigma: 0` is the one way to fix an element, and the value it is fixed at is
# `initval` -- there is no other channel.  With no initval from ANY source
# to_vec's fill silently holds it at 0.0 in internal units, and the LaTeX
# emitter writes no macro for it while latex.py still references one.
# ---------------------------------------------------------------------------


def test_pinned_parameter_with_no_value_raises():
    """
    Given a parameter pinned with sigma=0 and no initval from any source,
    When it is materialized in the PyMC graph,
    Then a ValueError naming the parameter is raised rather than silently
    pinning it at 0.0.
    """
    # ARRANGE
    p = Parameter(
        label="star.teff",
        sigma=0,
        lower=1000.0,
        upper=10000.0,
        unit="",
        internal_unit="",
    )

    # ACT & ASSERT
    with pm.Model():
        with pytest.raises(
            ValueError, match="Pinned parameter with no value"
        ) as exc:
            p.build_pymc()
    assert "star.teff" in str(exc.value)
    assert "initval" in str(exc.value)


def test_pin_error_names_the_params_file_when_one_is_known():
    """
    Given a pinned-with-no-value parameter that came from a params FILE,
    When the error is raised,
    Then the message quotes that file so the user knows what to edit.
    """
    # ARRANGE
    p = Parameter(
        label="star.teff",
        sigma=0,
        lower=1000.0,
        upper=10000.0,
        unit="",
        internal_unit="",
        source_file="myfit.params.yaml",
    )

    # ACT & ASSERT
    with pm.Model():
        with pytest.raises(ValueError) as exc:
            p.build_pymc()
    assert "myfit.params.yaml" in str(exc.value)


def test_pin_error_is_per_element_and_spares_the_valued_elements():
    """
    Given a vector whose element 0 is pinned WITHOUT a value and whose
    element 1 is pinned WITH one,
    When the model is built,
    Then only element 0 is reported (the check is per element, like the
    "overrides" pins it has to coexist with).
    """
    # ARRANGE
    p = Parameter(
        label="gp.amp",
        initval=[np.nan, 2.0],
        sigma=[0.0, 0.0],
        lower=[0.0, 0.0],
        upper=[10.0, 10.0],
        shape=(2,),
        names=["fileA", "fileB"],
        unit="",
        internal_unit="",
    )

    # ACT & ASSERT
    with pm.Model():
        with pytest.raises(ValueError) as exc:
            p.build_pymc()
    assert "gp.fileA.amp" in str(exc.value)
    assert "gp.fileB.amp" not in str(exc.value)


def test_pinned_derived_parameter_with_no_value_warns_but_does_not_raise(
    caplog,
):
    """
    Given a DERIVED parameter pinned with sigma=0 and no initval,
    When the model is built,
    Then it builds -- its value comes from the expression, so nothing is
    undefined -- and keeps its own "sigma=0 has no effect" warning, which is
    a different mistake with a different fix.
    """
    import logging

    import pytensor.tensor as pt

    # ARRANGE
    expr_val = pt.as_tensor_variable(np.float64(3.0))
    p = Parameter(
        label="lens.t_E",
        sigma=0,
        expression=lambda: expr_val,
        unit="",
        internal_unit="",
    )

    # ACT
    with pm.Model():
        with caplog.at_level(
            logging.WARNING, logger="exozippy.components.parameter"
        ):
            p.build_pymc()

    # ASSERT
    assert any(
        "sigma=0 has no effect" in rec.message for rec in caplog.records
    )


def test_hard_linked_pinned_element_is_exempt():
    """
    Given an element pinned with sigma=0 whose value comes from a hard LINK
    (an initval expression referencing another parameter),
    When the model is built,
    Then no error is raised -- the link expression IS the value.
    """
    # ARRANGE
    p = Parameter(
        label="orbit.omega",
        sigma=0,
        lower=0.0,
        upper=360.0,
        unit="",
        internal_unit="",
        element_links={
            "hard": {0: {"fn": lambda v: pt.constant(180.0), "intra_deps": ()}}
        },
    )

    # ACT & ASSERT
    with pm.Model():
        p.build_pymc()  # must not raise


def test_pin_relying_on_a_defaults_yaml_initval_is_accepted():
    """
    Given a params file that pins star.A.radius with sigma: 0 and NO initval,
    When the parameter is built through the real component path,
    Then it builds and is held at the defaults.yaml initval -- a value the
    params file does not spell out is still a value, so this is not the
    error case.
    """
    # ARRANGE
    config_manager = ConfigManager({"star.A.radius": {"sigma": 0}})
    star = Star([{"name": "A"}], config_manager)

    # ACT
    with pm.Model(name="defaults_pin"):
        star.manifest = {"radius": {}}
        star.add_parameter(model=None, param_name="radius", system=None)

    # ASSERT -- defaults.yaml's star.radius initval, not 0.0
    assert np.isclose(star.radius.initval[0], 1.0)
    assert star.radius.is_sampled.tolist() == [False]


def test_pin_with_no_value_anywhere_raises_through_the_component_path():
    """
    Given a params file that pins a parameter whose defaults.yaml supplies no
    initval either (star.mass carries none -- it is normally derived),
    When the parameter is built through the real component path,
    Then the error fires and names the parameter as the user spelled it.
    """
    # ARRANGE
    config_manager = ConfigManager({"star.A.mass": {"sigma": 0}})
    star = Star([{"name": "A"}], config_manager)

    # ACT & ASSERT
    with pm.Model(name="valueless_pin"):
        star.manifest = {"mass": {}}
        with pytest.raises(
            ValueError, match="Pinned parameter with no value"
        ) as exc:
            star.add_parameter(model=None, param_name="mass", system=None)
    assert "star.A.mass" in str(exc.value)


def test_manifest_overrides_pins_are_unaffected():
    """
    Given the manifest "overrides" channel pinning element 0 of a vector
    (what Instrument._register_gp / _register_robust / Band's LD autopin do),
    When the parameter is built,
    Then no error is raised: a component-supplied pin always rides on a
    value, here the defaults.yaml initval every element inherits.
    """
    # ARRANGE
    config_manager = ConfigManager({})
    star = Star([{"name": "A"}, {"name": "B"}], config_manager)

    # ACT
    with pm.Model(name="override_pin"):
        star.manifest = {
            "radius": {"overrides": {"sigma": [0.0, np.nan]}},
        }
        star.add_parameter(model=None, param_name="radius", system=None)

    # ASSERT -- element 0 pinned by the override, element 1 still sampled
    assert star.radius.is_sampled.tolist() == [False, True]
    assert np.allclose(star.radius.initval, [1.0, 1.0])
