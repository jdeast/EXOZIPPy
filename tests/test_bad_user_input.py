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


# ---------------------------------------------------------------------------
# A SAMPLED ELEMENT MUST SAY WHERE IT STARTS (Parameter.build_pymc).
#
# The pin check's sibling, and the same argument: `initval` is the ONLY
# channel for a start value, and to_vec's `fill=0.0` turns "nobody said" into
# the number 0.0 in whatever internal unit the parameter carries.  For a
# sampled element that 0.0 is where the chains begin, where the whitening
# probe measures, and what every multi-seed start is derived from -- and it
# looks exactly like a start somebody chose.
#
# The first group pins that it fires; the second pins that it does NOT fire
# for any of the several channels a value legitimately arrives through, since
# stage 5 is where they have all landed.
# ---------------------------------------------------------------------------


def test_sampled_parameter_with_no_start_value_raises():
    """
    Given a sampled parameter with bounds but no initval from any source,
    When it is materialized in the PyMC graph,
    Then a ValueError naming the parameter is raised rather than silently
    starting the chains at 0.0 in internal units.
    """
    # ARRANGE
    p = Parameter(
        label="star.teff",
        lower=1000.0,
        upper=10000.0,
        unit="",
        internal_unit="",
    )

    # ACT & ASSERT
    with pm.Model():
        with pytest.raises(
            ValueError, match="Sampled parameter with no start value"
        ) as exc:
            p.build_pymc()
    msg = str(exc.value)
    assert "star.teff" in msg
    assert "initval" in msg
    # Not the pin error and not the out-of-bounds one: three neighbouring
    # checks, three different mistakes, three different fixes.
    assert "Pinned parameter with no value" not in msg
    assert "outside its hard bounds" not in msg


def test_no_start_error_names_the_params_file_when_one_is_known():
    """
    Given a start-less sampled parameter whose values came from a params FILE,
    When the error is raised,
    Then the message quotes that file so the user knows what to edit.
    """
    # ARRANGE
    p = Parameter(
        label="star.teff",
        lower=1000.0,
        upper=10000.0,
        unit="",
        internal_unit="",
        source_file="myfit.params.yaml",
    )

    # ACT & ASSERT
    with pm.Model():
        with pytest.raises(
            ValueError, match="Sampled parameter with no start value"
        ) as exc:
            p.build_pymc()
    assert "myfit.params.yaml" in str(exc.value)


def test_no_start_error_is_per_element_and_spares_the_valued_elements():
    """
    Given a vector whose element 0 has no start and whose element 1 does,
    When the model is built,
    Then only element 0 is reported -- the check is per element, like the
    "overrides" channel it has to coexist with.
    """
    # ARRANGE
    p = Parameter(
        label="gp.amp",
        initval=[np.nan, 2.0],
        lower=[0.0, 0.0],
        upper=[10.0, 10.0],
        shape=(2,),
        names=["fileA", "fileB"],
        unit="",
        internal_unit="",
    )

    # ACT & ASSERT
    with pm.Model():
        with pytest.raises(
            ValueError, match="Sampled parameter with no start value"
        ) as exc:
            p.build_pymc()
    msg = str(exc.value)
    assert "gp.fileA.amp" in msg
    assert "gp.fileB.amp" not in msg


def test_unbounded_sampled_element_with_no_start_raises_too():
    """
    Given a sampled element with no bounds and an explicit Gaussian sigma,
    When the model is built,
    Then it still raises: the Gaussian branch reads its start from initval
    too (raw = (initval - mu)/sigma), so a missing one starts the chain at
    0.0 rather than at the prior centre. The message says "unbounded" rather
    than quoting bounds it does not have.
    """
    # ARRANGE
    p = Parameter(
        label="star.teff",
        mu=5778.0,
        sigma=100.0,
        unit="",
        internal_unit="",
    )

    # ACT & ASSERT
    with pm.Model():
        with pytest.raises(
            ValueError, match="Sampled parameter with no start value"
        ) as exc:
            p.build_pymc()
    assert "unbounded" in str(exc.value)


def test_no_start_with_no_value_anywhere_raises_through_component_path():
    """
    Given a manifest that declares star.mass a FREE parameter (it carries no
    defaults.yaml initval -- it is normally derived from logmass) and nothing
    seeding it,
    When the parameter is built through the real component path,
    Then the error fires and names the parameter as the user spelled it.
    """
    # ARRANGE
    config_manager = ConfigManager({})
    star = Star([{"name": "A"}], config_manager)

    # ACT & ASSERT
    with pm.Model(name="valueless_sampled"):
        star.manifest = {"mass": {"lower": 0.1, "upper": 250.0}}
        with pytest.raises(
            ValueError, match="Sampled parameter with no start value"
        ) as exc:
            star.add_parameter(model=None, param_name="mass", system=None)
    assert "star.A.mass" in str(exc.value)


def test_no_start_error_tailors_its_advice_to_the_provenance():
    """
    Given a start-less element whose recorded provenance is the relaxation
    engine (some channel is on record while no number landed),
    When the error is raised,
    Then the advice names that channel rather than blaming a params-file line
    the user never wrote -- the same tailoring the out-of-bounds error does.
    """
    # ARRANGE
    p = Parameter(
        label="lens.t_E",
        lower=1.0,
        upper=500.0,
        unit="",
        internal_unit="",
        initval_source=lambda *a, **k: "solved",
    )

    # ACT & ASSERT
    with pm.Model():
        with pytest.raises(ValueError) as exc:
            p.build_pymc()
    msg = str(exc.value)
    assert "provenance: solved" in msg
    assert "relaxation engine" in msg
    assert "not your params file" not in msg  # the "default" advice


def _boom(*args, **kwargs):
    raise RuntimeError("provenance lookup exploded")


@pytest.mark.parametrize(
    "source", [_boom, lambda *a, **k: "somethingelse"], ids=["raises", "bogus"]
)
def test_no_start_error_survives_a_broken_provenance_lookup(source):
    """
    Given an initval_source callable that raises, or one that returns a label
    the advice table does not know,
    When the no-start error is rendered,
    Then the diagnosis still gets out (degraded to "default") -- a fault in
    the decoration must never replace the error it decorates, and a KeyError
    while rendering the message would do exactly that.
    """
    # ARRANGE
    p = Parameter(
        label="star.teff",
        lower=1000.0,
        upper=10000.0,
        unit="",
        internal_unit="",
        initval_source=source,
    )

    # ACT & ASSERT
    with pm.Model():
        with pytest.raises(
            ValueError, match="Sampled parameter with no start value"
        ) as exc:
            p.build_pymc()
    assert "provenance: default" in str(exc.value)


# --- the channels a value legitimately arrives through: none may fire -------


def test_derived_element_needs_no_start_value():
    """
    Given a DERIVED parameter with no initval at all,
    When the model is built,
    Then nothing is raised: its value is the expression, and it is not
    sampled.
    """
    # ARRANGE
    expr_val = pt.as_tensor_variable(np.float64(3.0))
    p = Parameter(
        label="lens.t_E",
        expression=lambda: expr_val,
        unit="",
        internal_unit="",
    )

    # ACT & ASSERT
    with pm.Model(name="derived_no_start"):
        p.build_pymc()  # must not raise


def test_pinned_element_with_no_start_still_gets_the_pin_error():
    """
    Given an element pinned with sigma=0 and no value,
    When the model is built,
    Then it is the PIN error that fires, not this one -- the two checks sit
    side by side and a pin has its own message and its own fix.
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
        with pytest.raises(ValueError, match="Pinned parameter with no value"):
            p.build_pymc()


def test_symbolic_initval_counts_as_a_start_value():
    """
    Given a sampled element whose initval is a symbolic node (what a linked
    initval resolves to),
    When the model is built,
    Then nothing is raised and the node is not evaluated to find out.
    """
    # ARRANGE
    p = Parameter(
        label="star.teff",
        initval=pt.as_tensor_variable(np.float64(5778.0)),
        lower=1000.0,
        upper=10000.0,
        unit="",
        internal_unit="",
    )

    # ACT & ASSERT
    with pm.Model(name="symbolic_start"):
        p.build_pymc()  # must not raise


def test_defaults_yaml_initval_satisfies_the_check():
    """
    Given a parameter whose only start value is the one in its component's
    defaults.yaml (star.radius: 1.0),
    When it is built through the real component path with empty user params,
    Then it builds and starts there.
    """
    # ARRANGE
    config_manager = ConfigManager({})
    star = Star([{"name": "A"}], config_manager)

    # ACT
    with pm.Model(name="defaults_start"):
        star.manifest = {"radius": {}}
        star.add_parameter(model=None, param_name="radius", system=None)

    # ASSERT
    assert star.radius.is_sampled.tolist() == [True]
    assert np.isclose(star.radius.initval[0], 1.0)


def test_component_hint_satisfies_the_check():
    """
    Given a component hint as the ONLY source of a start value (the channel
    load_data uses for a data-derived guess, layered in by the stage-3 solve),
    When the parameter is built,
    Then it builds and starts at the hinted value.
    """
    # ARRANGE
    config_manager = ConfigManager({}, system_config={"star": [{"name": "A"}]})
    config_manager.add_hint("star.A.mass", 1.3)
    config_manager.finalize_user_params()
    star = Star([{"name": "A"}], config_manager)

    # ACT
    with pm.Model(name="hint_start"):
        star.manifest = {"mass": {"lower": 0.1, "upper": 250.0}}
        star.add_parameter(model=None, param_name="mass", system=None)

    # ASSERT
    assert star.mass.is_sampled.tolist() == [True]
    assert np.isclose(star.mass.initval[0], 1.3)


def test_manifest_overrides_initval_satisfies_the_check():
    """
    Given the manifest "overrides" channel supplying the start value (what
    MulensInstrument._scale_flux_amplitudes and the GP amplitudes do),
    When the parameter is built,
    Then it builds -- the check has to see a channel applied inside its own
    stage, which is why it lives at stage 5.
    """
    # ARRANGE
    config_manager = ConfigManager({}, system_config={"star": [{"name": "A"}]})
    star = Star([{"name": "A"}], config_manager)

    # ACT
    with pm.Model(name="overrides_start"):
        star.manifest = {
            "mass": {
                "lower": 0.1,
                "upper": 250.0,
                "overrides": {"initval": 1.4},
            }
        }
        star.add_parameter(model=None, param_name="mass", system=None)

    # ASSERT
    assert np.isclose(star.mass.initval[0], 1.4)


def test_manifest_options_initval_satisfies_the_check():
    """
    Given a plain manifest option supplying the start value (what transit's
    per-file median baseline does),
    When the parameter is built,
    Then it builds.
    """
    # ARRANGE
    config_manager = ConfigManager({}, system_config={"star": [{"name": "A"}]})
    star = Star([{"name": "A"}], config_manager)

    # ACT
    with pm.Model(name="options_start"):
        star.manifest = {
            "mass": {"lower": 0.1, "upper": 250.0, "initval": 1.7}
        }
        star.add_parameter(model=None, param_name="mass", system=None)

    # ASSERT
    assert np.isclose(star.mass.initval[0], 1.7)


def test_relaxation_engine_solution_satisfies_the_check():
    """
    Given a start value that exists ONLY as the relaxation engine's solution
    (star.mass has no defaults.yaml initval; seeding logmass derives it
    through Eq(mass, 10**logmass)),
    When the parameter is built,
    Then it builds and starts at the derived value -- the engine's solution
    is a start value even though nothing spells it out anywhere.
    """
    # ARRANGE
    config = {"star": [{"name": "A"}]}
    config_manager = ConfigManager(
        {"star.A.logmass": {"initval": 0.3}}, system_config=config
    )
    config_manager.finalize_user_params()
    star = Star(config["star"], config_manager)

    # ACT
    with pm.Model(name="engine_start"):
        star.manifest = {"mass": {"lower": 0.1, "upper": 250.0}}
        star.add_parameter(model=None, param_name="mass", system=None)

    # ASSERT
    assert star.mass.is_sampled.tolist() == [True]
    assert np.isclose(star.mass.initval[0], 10**0.3)


# ---------------------------------------------------------------------------
# A START VALUE OUTSIDE ITS HARD BOUNDS IS FATAL (Parameter.build_pymc).
#
# Two finite bounds put the element on the logit transform, whose support IS
# [lower, upper] -- there is no raw coordinate for a value outside it. The old
# code clipped such a start onto the wall behind a warning, so a fit that
# started somewhere the user never asked for was indistinguishable from one
# that started where they did. It now raises.
#
# Three neighbours deliberately do NOT raise and are pinned below: a value
# exactly ON a bound (representable, just infinitely far in logit space --
# nudged inward by <= 1e-6 of the whitening scale, loudly), a one-sided SOFT
# bound (a penalty, not support), and a FIXED element (never transformed).
# ---------------------------------------------------------------------------


def test_start_outside_hard_bounds_raises():
    """
    Given a sampled parameter whose initval lies outside its [lower, upper],
    When it is materialized in the PyMC graph,
    Then a ValueError naming the parameter, the value and the bounds is
    raised rather than the start being clipped onto the wall.
    """
    # ARRANGE
    p = Parameter(
        label="star.teff",
        initval=12000.0,
        lower=1000.0,
        upper=10000.0,
        unit="",
        internal_unit="",
    )

    # ACT & ASSERT
    with pm.Model():
        with pytest.raises(
            ValueError, match="Start value outside its hard bounds"
        ) as exc:
            p.build_pymc()
    msg = str(exc.value)
    assert "star.teff" in msg
    assert "12000" in msg
    assert "1000" in msg and "10000" in msg


def test_start_below_lower_bound_raises_too():
    """
    Given a start value below the lower bound (the other side of the same
    error),
    When the model is built,
    Then it raises -- the check is two-sided.
    """
    # ARRANGE
    p = Parameter(
        label="star.teff",
        initval=-5.0,
        lower=1000.0,
        upper=10000.0,
        unit="",
        internal_unit="",
    )

    # ACT & ASSERT
    with pm.Model():
        with pytest.raises(ValueError, match="outside its hard bounds"):
            p.build_pymc()


def test_out_of_bounds_error_names_the_params_file_when_known():
    """
    Given an out-of-bounds start on a parameter that came from a params FILE,
    When the error is raised,
    Then the message quotes that file, so the user knows what to open --
    exactly as the "pinned with no value" error does.
    """
    # ARRANGE
    p = Parameter(
        label="star.teff",
        initval=12000.0,
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


def test_out_of_bounds_error_reports_every_offending_element():
    """
    Given a vector with TWO out-of-bounds elements and one good element,
    When the model is built,
    Then all of the offenders are named in one message and the in-bounds
    element is not -- a user fixing a vector should not have to rerun the
    build once per bad element.
    """
    # ARRANGE
    p = Parameter(
        label="gp.amp",
        initval=[-1.0, 5.0, 99.0],
        lower=[0.0, 0.0, 0.0],
        upper=[10.0, 10.0, 10.0],
        shape=(3,),
        names=["fileA", "fileB", "fileC"],
        unit="",
        internal_unit="",
    )

    # ACT & ASSERT
    with pm.Model():
        with pytest.raises(ValueError) as exc:
            p.build_pymc()
    msg = str(exc.value)
    assert "gp.fileA.amp" in msg
    assert "gp.fileC.amp" in msg
    assert "gp.fileB.amp" not in msg


def test_out_of_bounds_error_reports_values_in_user_units():
    """
    Given a parameter whose user unit differs from its internal unit (degrees
    in, radians internally) and an out-of-bounds start,
    When the error is raised,
    Then the value and bounds are quoted in the USER unit, so the numbers
    match what the user typed rather than the internal representation.
    """
    # ARRANGE -- 400 deg against a [0, 360] deg range, stored as radians
    p = Parameter(
        label="orbit.omega",
        initval=400.0,
        lower=0.0,
        upper=360.0,
        unit="deg",
        internal_unit="rad",
    )

    # ACT & ASSERT
    with pm.Model():
        with pytest.raises(ValueError) as exc:
            p.build_pymc()
    msg = str(exc.value)
    assert "400" in msg and "360" in msg
    assert "deg" in msg


def test_nan_start_raises_its_own_no_start_value_error():
    """
    Given a NaN start value on a bounded sampled element,
    When the model is built,
    Then it raises -- but with the "no start value" message, not the
    out-of-bounds one. NaN satisfies no bound either, yet "you asked to start
    outside the bounds" is the wrong diagnosis for "nothing gave this element
    a start at all", and the fixes differ. Previously np.clip returned the
    NaN unchanged, the code warned that it had "nudged" it, and the model was
    built around log(NaN/(1-NaN)).

    NaN is only ConfigManager.resolve's SPELLING of "this element was never
    set", so this is one case of the general rule pinned in the section
    below: a sampled element must be given a start value.
    """
    # ARRANGE
    p = Parameter(
        label="star.teff",
        initval=np.nan,
        lower=1000.0,
        upper=10000.0,
        unit="",
        internal_unit="",
    )

    # ACT & ASSERT
    with pm.Model():
        with pytest.raises(
            ValueError, match="Sampled parameter with no start value"
        ) as exc:
            p.build_pymc()
    msg = str(exc.value)
    assert "star.teff" in msg
    assert "outside its hard bounds" not in msg


def test_nan_element_of_a_vector_is_reported_per_element():
    """
    Given a vector where only element 1 has a start,
    When the model is built,
    Then the no-start error names element 0 and spares element 1 -- the check
    is per element, like the "pinned with no value" error it sits beside.
    """
    # ARRANGE
    p = Parameter(
        label="star.teff",
        initval=[np.nan, 5778.0],
        lower=[1000.0, 1000.0],
        upper=[10000.0, 10000.0],
        shape=(2,),
        names=["A", "B"],
        unit="",
        internal_unit="",
    )

    # ACT & ASSERT
    with pm.Model():
        with pytest.raises(
            ValueError, match="Sampled parameter with no start value"
        ) as exc:
            p.build_pymc()
    msg = str(exc.value)
    assert "star.A.teff" in msg
    assert "star.B.teff" not in msg


def test_in_bounds_start_is_untouched(caplog):
    """
    Given an ordinary in-bounds start,
    When the model is built,
    Then nothing is raised, nothing is nudged, and raw=0 maps back to exactly
    the requested value -- the check is inert on a well-posed model.
    """
    # ARRANGE
    import logging

    p = Parameter(
        label="star.teff",
        initval=5778.0,
        init_scale=100.0,
        lower=1000.0,
        upper=10000.0,
        unit="",
        internal_unit="",
    )

    # ACT
    with pm.Model():
        with caplog.at_level(
            logging.WARNING, logger="exozippy.components.parameter"
        ):
            p.build_pymc()

    # ASSERT -- no nudge, and raw = 0 recovers the start exactly
    assert not [r for r in caplog.records if "nudged" in r.getMessage()], [
        r.getMessage() for r in caplog.records
    ]
    assert np.allclose(p.phys_from_raw(np.zeros(1)), [5778.0])


def test_start_exactly_on_a_bound_is_nudged_inward_loudly(caplog):
    """
    Given a start value sitting EXACTLY on its lower bound (an angle
    defaulting to 0 on a [0, 360) range -- common and legitimate),
    When the model is built,
    Then it does NOT raise: the value is inside the support, it is merely at
    infinite distance in logit space, so it is nudged inward by at most
    1e-6 of the whitening scale and the move is logged.
    """
    # ARRANGE
    import logging

    p = Parameter(
        label="orbit.omega",
        initval=0.0,
        init_scale=1.0,
        lower=0.0,
        upper=360.0,
        unit="",
        internal_unit="",
    )

    # ACT
    with pm.Model():
        with caplog.at_level(
            logging.WARNING, logger="exozippy.components.parameter"
        ):
            p.build_pymc()

    # ASSERT -- loud, and the move is negligible in problem units
    assert any("nudged inward" in r.getMessage() for r in caplog.records)
    started_at = float(np.atleast_1d(p.phys_from_raw(np.zeros(1)))[0])
    assert 0.0 < started_at < 1e-5


def test_start_exactly_on_upper_bound_is_nudged_not_refused(caplog):
    """
    Given a start value sitting EXACTLY on its UPPER bound,
    When the model is built,
    Then the same nudge applies (the boundary case is symmetric) and no
    error is raised.
    """
    # ARRANGE
    import logging

    p = Parameter(
        label="orbit.omega",
        initval=360.0,
        init_scale=1.0,
        lower=0.0,
        upper=360.0,
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
    assert any("nudged inward" in r.getMessage() for r in caplog.records)
    started_at = float(np.atleast_1d(p.phys_from_raw(np.zeros(1)))[0])
    assert 360.0 - 1e-5 < started_at < 360.0


def test_one_sided_soft_bound_start_is_not_fatal():
    """
    Given a sampled element with only ONE finite bound (a soft barrier, not a
    hard support) and a start on the wrong side of it,
    When the model is built,
    Then it builds: a barrier is a penalty with a restoring gradient, and a
    start there is improbable rather than meaningless.
    """
    # ARRANGE -- a finite lower bound and an infinite upper one, so the
    # element takes the Gaussian branch and the bound becomes a soft barrier
    # rather than the logit transform's support.
    p = Parameter(
        label="planet.mass",
        initval=-3.0,
        init_scale=1.0,
        lower=0.0,
        upper=np.inf,
        sigma=1.0,
        unit="",
        internal_unit="",
    )

    # ACT & ASSERT
    with pm.Model():
        p.build_pymc()  # must not raise
    assert bool(np.atleast_1d(p.is_sampled)[0])


def test_fixed_element_outside_bounds_is_not_fatal():
    """
    Given an element PINNED with sigma=0 at a value outside its bounds,
    When the model is built,
    Then it builds. A pinned element is never sampled, so it gets neither the
    logit transform nor a barrier -- its bounds are inert and nothing is
    clipped. This is a deliberate exclusion, pinned so a later widening of
    the check is a conscious decision rather than an accident.
    """
    # ARRANGE
    p = Parameter(
        label="star.teff",
        initval=12000.0,
        sigma=0,
        lower=1000.0,
        upper=10000.0,
        unit="",
        internal_unit="",
    )

    # ACT & ASSERT
    with pm.Model():
        p.build_pymc()  # must not raise
    assert np.allclose(p.initval, [12000.0])


def test_derived_element_outside_bounds_is_not_fatal():
    """
    Given a DERIVED element whose expression value falls outside its bounds,
    When the model is built,
    Then it builds -- a derived parameter's bounds are soft barriers on the
    expression, not a start value anyone chose.
    """
    # ARRANGE
    expr_val = pt.as_tensor_variable(np.float64(12000.0))
    p = Parameter(
        label="star.teff",
        initval=12000.0,
        expression=lambda: expr_val,
        lower=1000.0,
        upper=10000.0,
        unit="",
        internal_unit="",
    )

    # ACT & ASSERT
    with pm.Model():
        p.build_pymc()  # must not raise


def test_out_of_bounds_error_blames_the_user_when_the_user_wrote_it():
    """
    Given a params file whose explicit initval for star.A.radius is outside
    the parameter's hard bounds,
    When the parameter is built through the real component path,
    Then the error fires, names the parameter as the user spelled it, and
    attributes the value to the user (not to a derivation they never made).
    """
    # ARRANGE -- star.radius bounds are [1e-4, 2000] solRad
    config_manager = ConfigManager({"star.A.radius": {"initval": 1.0e6}})
    star = Star([{"name": "A"}], config_manager)

    # ACT & ASSERT
    with pm.Model(name="oob_user"):
        star.manifest = {"radius": {}}
        with pytest.raises(
            ValueError, match="Start value outside its hard bounds"
        ) as exc:
            star.add_parameter(model=None, param_name="radius", system=None)
    msg = str(exc.value)
    assert "star.A.radius" in msg
    assert "start value from: user" in msg


def test_out_of_bounds_error_does_not_blame_the_user_for_a_default():
    """
    Given a params file that only TIGHTENS a bound, so that the untouched
    defaults.yaml start now falls outside it,
    When the parameter is built,
    Then the error fires but attributes the value to the default rather than
    to the user's params file -- there is no initval line for them to fix,
    and the advice must say so.
    """
    # ARRANGE -- star.radius defaults to 1.0 solRad; tighten above it
    config_manager = ConfigManager({"star.A.radius": {"lower": 5.0}})
    star = Star([{"name": "A"}], config_manager)

    # ACT & ASSERT
    with pm.Model(name="oob_default"):
        star.manifest = {"radius": {}}
        with pytest.raises(
            ValueError, match="Start value outside its hard bounds"
        ) as exc:
            star.add_parameter(model=None, param_name="radius", system=None)
    msg = str(exc.value)
    assert "start value from: default" in msg
    assert "defaults.yaml" in msg


def test_out_of_bounds_error_blames_the_engine_for_a_derived_value():
    """
    Given an out-of-bounds start that provenance attributes to the relaxation
    engine ("solved") rather than to the params file,
    When the error is raised,
    Then the advice says the value was DERIVED and that the inputs it was
    solved from are inconsistent -- it must NOT tell the user to fix an
    'initval' line they never wrote.
    """
    # ARRANGE
    p = Parameter(
        label="lens.t_E",
        initval=1e6,
        lower=0.0,
        upper=1000.0,
        unit="",
        internal_unit="",
        source_file="myfit.params.yaml",
        initval_source=lambda comp, param, element=None, name=None: "solved",
    )

    # ACT & ASSERT
    with pm.Model():
        with pytest.raises(ValueError) as exc:
            p.build_pymc()
    msg = str(exc.value)
    assert "start value from: solved" in msg
    assert "DERIVED by the relaxation engine" in msg
    assert "This start value is the 'initval' in your params file" not in msg


def test_out_of_bounds_error_survives_a_broken_provenance_lookup():
    """
    Given an initval_source callable that raises,
    When the out-of-bounds error is rendered,
    Then the error is still the bounds error (provenance is metadata; a fault
    there must never mask or replace the real diagnosis).
    """

    # ARRANGE
    def boom(*a, **k):
        raise RuntimeError("provenance is broken")

    p = Parameter(
        label="star.teff",
        initval=12000.0,
        lower=1000.0,
        upper=10000.0,
        unit="",
        internal_unit="",
        initval_source=boom,
    )

    # ACT & ASSERT
    with pm.Model():
        with pytest.raises(
            ValueError, match="Start value outside its hard bounds"
        ):
            p.build_pymc()


def test_out_of_bounds_through_the_solved_config_pipeline_says_user():
    """
    Given the FULL config pipeline (ConfigManager with a system config plus
    finalize_user_params, so the relaxation engine really runs and records
    provenance) and a user initval outside star.distance's hard bounds,
    When the parameter is built,
    Then it raises and the provenance really resolves to "user" -- the
    fallback scan is not what is being relied on here.
    """
    # ARRANGE -- star.distance bounds are [1e-3, 1e5] pc
    config = {"star": [{"name": "A"}]}
    cm = ConfigManager(
        {"star.A.distance": {"initval": 1.0e9}}, system_config=config
    )
    cm.finalize_user_params()
    star = Star(config["star"], cm)
    star.manifest = {"distance": None}

    # ACT & ASSERT
    with pm.Model(name="oob_pipeline"):
        with pytest.raises(
            ValueError, match="Start value outside its hard bounds"
        ) as exc:
            star.add_parameter(model=None, param_name="distance", system=None)
    msg = str(exc.value)
    assert "star.A.distance" in msg
    assert "start value from: user" in msg
