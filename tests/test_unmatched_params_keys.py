"""A params key that matches NO model parameter is reported to EVERY caller.

Review 2.3.16, and it cost a run.  ``DC2018_128_tightpriors.params.yaml``
keyed a bound as ``lens.DC2018_128.t_0`` -- the RUN name, where the config
declares ``lens: - name: "Lens"``.  The key matched nothing, the t_0
tightening (5.76 of that file's ~27.6 nats) was simply absent, and a
nested-sampling job ran for over an hour before the mistake was found by
reading the config.  ``examples/DC2018/dc18_fullns_pilot.py`` drives the
sampler directly, and the only warning about an unmatched key lived in
``run.inspect_start`` -- run.py's startup reporting -- so it never ran.

THE MECHANISM, which is why the config layer's existing strict check does
not cover this: ``config.standardize_param_names`` DOES refuse a 3-part key
naming an undeclared instance, but its accepted set is every ``name:``
declared by ANY list component, deliberately -- a component's per-element
names may borrow another component's (``source.<star name>.rho``).  The
tightpriors config declares ``galacticmodel: - name: "DC2018_128"``, so the
mis-keyed name WAS declared somewhere and passed.  The vehicle below
reproduces exactly that shape: the name is declared on the other component.

ONE AUTHORITY: ``diagnostics.ModelAuditor.check_unused_yaml`` still owns the
question (``check_user_starts`` defers to it in as many words).  What moved
is where it is REPORTED -- ``System.build_model``, the first point at which
the answer exists (Parameter objects are created in stage 6, so after
``prepare()`` alone ``get_all_parameters()`` is empty) and one every caller
that samples passes through.
"""

import logging

import numpy as np
import pytest

from exozippy.run import inspect_start
from exozippy.system import System

# The name is declared by `planet`, so the config layer's strict naming
# check accepts `star.DC2018_128.logmass` -- and nothing owns it.
CONFIG = {
    "star": [{"name": "A"}],
    "planet": [{"name": "DC2018_128"}],
    "orbit": [{"star": "star.A", "planet": "planet.DC2018_128"}],
}
BAD_KEY = "star.DC2018_128.logmass"
PARAMS = {
    "star.A.teff": {"initval": 5000.0},
    BAD_KEY: {"lower": -0.4, "upper": -0.2},
}

WARNING_MARKER = "did not match any model parameter"


@pytest.fixture(scope="module")
def built():
    """prepare() + build_model() and nothing else -- no run.py anywhere.

    The same two calls examples/DC2018/dc18_fullns_pilot.py makes before
    handing the model to a sampler itself.
    """
    system = System(CONFIG, user_params=dict(PARAMS))
    system.prepare()
    model = system.build_model()
    return system, model


def _warnings_for(records):
    return [r for r in records if WARNING_MARKER in r.getMessage()]


def test_parameters_do_not_exist_until_the_model_is_built():
    """
    Given a prepared System,
    When its parameters are listed after prepare() but before build_model(),
    Then there are none.

    This is why the check is reported from build_model and not from
    ConfigManager.finalize_user_params as the item proposed: at stage 4
    there is nothing to compare the user's keys against.  Pinned so the
    placement is not "simplified" back into the config layer.
    """
    system = System(CONFIG, user_params=dict(PARAMS))
    system.prepare()

    assert system.get_all_parameters() == []


def test_a_mis_keyed_params_entry_is_reported_without_run_py(caplog):
    """
    Given a params key whose instance name is declared by a DIFFERENT
      component (so the strict naming check accepts it) and which therefore
      owns no parameter,
    When the System is prepared and built -- with no run.py in the call path,
    Then a warning names the key, and it comes from the system layer.
    """
    # ARRANGE / ACT
    with caplog.at_level(logging.WARNING, logger="exozippy.system"):
        system = System(CONFIG, user_params=dict(PARAMS))
        system.prepare()
        system.build_model()

    # ASSERT
    hits = _warnings_for(caplog.records)
    assert hits, [r.getMessage() for r in caplog.records]
    assert BAD_KEY in hits[0].getMessage()
    assert hits[0].name == "exozippy.system"


def test_the_mis_keyed_bound_really_was_absent(built):
    """
    Given the same build,
    When star.logmass's bounds are read,
    Then they are the defaults.yaml ones, not the user's tightening.

    The warning is only worth emitting because the constraint is genuinely
    gone; asserted here so the two halves cannot drift apart (a future
    change that made the key WORK would leave the warning crying wolf).
    """
    system, _model = built

    logmass = system.star.logmass
    assert float(np.atleast_1d(logmass.lower)[0]) < -0.4
    assert float(np.atleast_1d(logmass.upper)[0]) > -0.2


def test_a_correctly_keyed_params_file_reports_nothing(caplog):
    """
    Given the same config with every key spelled correctly,
    When the System is prepared and built,
    Then no unmatched-key warning is emitted -- the check must not cry wolf.
    """
    # ARRANGE / ACT
    with caplog.at_level(logging.WARNING, logger="exozippy.system"):
        system = System(
            CONFIG, user_params={"star.A.teff": {"initval": 5000.0}}
        )
        system.prepare()
        system.build_model()

    # ASSERT
    assert _warnings_for(caplog.records) == []


def test_a_name_declared_nowhere_is_still_refused_outright():
    """
    Given a 3-part params key naming an instance declared NOWHERE,
    When the ConfigManager is constructed,
    Then it still raises the strict naming error.

    The control that says what this change did and did not do: the config
    layer's existing refusal is untouched and remains the first line of
    defence.  What build_model now reports is the RESIDUE that check
    deliberately lets through -- a name declared by some other component.
    """
    with pytest.raises(ValueError, match="STRICT NAMING ERROR"):
        System(CONFIG, user_params={"star.Nowhere.logmass": {"lower": 1.0}})


def test_the_startup_table_does_not_report_it_a_second_time(built, caplog):
    """
    Given a built model whose params file holds an unmatched key,
    When run.inspect_start renders the startup table,
    Then it does NOT emit the unmatched-key warning again.

    One authority, one emission: inspect_start used to be the only place
    this was reported, and leaving its copy behind would have meant every
    `exozippy <config>` run saying it twice.
    """
    # ARRANGE
    system, model = built
    start = system.get_raw_start(model)

    # ACT
    with caplog.at_level(logging.INFO, logger="exozippy.run"):
        inspect_start(model, system, start)

    # ASSERT
    assert [r.getMessage() for r in _warnings_for(caplog.records)] == [], (
        "inspect_start still emits its own copy"
    )


def test_a_rebuild_does_not_repeat_the_warning(caplog):
    """
    Given a System built twice (legal -- reviews 1.5.2 and 3.14.12),
    When the second build_model runs,
    Then the warning is not repeated.

    The params file cannot change between two builds of one System, and the
    GUI rebuilds on interaction, so a per-build warning would be noise.
    """
    # ARRANGE
    system = System(CONFIG, user_params=dict(PARAMS))
    system.prepare()
    system.build_model()
    assert _warnings_for(caplog.records), "the first build must warn"
    caplog.clear()

    # ACT
    with caplog.at_level(logging.WARNING, logger="exozippy.system"):
        system.build_model()

    # ASSERT
    assert _warnings_for(caplog.records) == []
