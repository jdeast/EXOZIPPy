"""The astronomical time layer is separable from a data component.

``components/timesystem.py`` was extracted from ``Instrument`` so that a data
component from a field with no Julian Dates in it can reuse the rest of the
data scaffolding without inheriting a time vocabulary it cannot honour.

The failure this guards is WORKING-BUT-WRONG, which is why it needs a test at
all. ``time_frame`` defaults to ``bjd`` and ``time_scale`` to ``tdb``, which
together make ``needs_conversion`` False and pass input straight through. A
non-astronomy component inheriting the whole of ``Instrument`` would therefore
run correctly while its config schema advertised a barycentric correction and
an observatory location for, say, a plasma sample. Nothing would fail; the
documentation would simply be false.
"""

import inspect

import pytest

from exozippy.components.component import Component
from exozippy.components.instrument import Instrument
from exozippy.components.timesystem import TimeSystem

# Everything the split moved.  Named explicitly rather than discovered, so
# moving one back into Instrument fails here instead of silently shrinking
# the guard.
TIME_MEMBERS = (
    "_parse_time_spec",
    "parse_time_specs",
    "has_nontrivial_time_spec",
    "_to_bjd_tdb",
    "_time_coord",
    "_time_location",
    "_time_config_schema",
)


@pytest.mark.parametrize("name", TIME_MEMBERS)
def test_every_time_member_lives_on_the_mixin(name):
    """Given a time member, Then TimeSystem owns it and Instrument does not.

    Checking the OWNER rather than mere availability: Instrument inherits all
    of these, so `hasattr` would pass whether or not the split happened.
    """
    assert name in TimeSystem.__dict__, f"{name} is not on TimeSystem"
    assert name not in Instrument.__dict__, (
        f"{name} is defined on Instrument again -- the split has been undone"
    )


def test_instrument_still_resolves_every_time_member():
    """Given Instrument, Then it still has all of them, via the mixin.

    The other half of the pair above: the split must not have removed
    behaviour, only relocated it.
    """
    for name in TIME_MEMBERS:
        assert hasattr(Instrument, name)
    assert TimeSystem in Instrument.__mro__


def test_the_mixin_is_not_a_component():
    """Given TimeSystem, Then it is not a Component and the factory ignores it.

    The factory discovers components by scanning for Component subclasses, so
    a mixin that inherited Component would be registered as a fake component
    with a `timesystem:` YAML key. `relations.StellarRelation` is a plain
    mixin for the same reason.
    """
    assert not issubclass(TimeSystem, Component)

    from exozippy.components.factory import discover_components

    assert "timesystem" not in discover_components()


def test_a_data_component_without_the_mixin_has_no_time_vocabulary():
    """Given a Component that does not mix TimeSystem in, Then it has none of it.

    THE POINT OF THE SPLIT. Before it, these members were on Instrument and a
    component inheriting Instrument got the whole astronomical time
    vocabulary whether or not it meant anything -- silently, because the
    defaults convert nothing. components/pharmacokinetics is the live case:
    it inherits Component directly, and this pins that the choice is real
    rather than merely intended.
    """
    from exozippy.components.pharmacokinetics.assay import Assay

    assert not issubclass(Assay, TimeSystem)
    for name in TIME_MEMBERS:
        assert not hasattr(Assay, name), (
            f"Assay has {name}: it is not an astronomical instrument and must "
            f"not advertise a time system it cannot honour"
        )


def test_the_time_config_schema_is_not_offered_by_a_non_instrument():
    """Given Assay's schema, Then it declares no astronomical time keys.

    The schema is what the GUI and the docs render, so an inherited
    `time_frame: bjd` would be published to users as a supported key on a
    component where it means nothing.
    """
    from exozippy.components.pharmacokinetics.assay import Assay

    keys = {entry["key"] for entry in Assay.config_schema()}
    astronomical = {
        "time_offset",
        "time_scale",
        "time_frame",
        "time_location",
        "time_ephemeris",
    }
    assert not (keys & astronomical), sorted(keys & astronomical)

    # It does still declare its own, field-appropriate time unit.
    assert "time_unit" in keys


def test_the_four_shipped_instruments_still_offer_the_time_schema():
    """Given each Instrument subclass, Then the time keys are still published.

    The split must be invisible to the components that legitimately want the
    vocabulary, and all four call `_time_config_schema()` from their own
    `config_schema`.
    """
    from exozippy.components.astrometryinstrument.astrometryinstrument import (
        AstrometryInstrument,
    )
    from exozippy.components.mulensing.mulensinstrument import MulensInstrument
    from exozippy.components.rvinstrument.rvinstrument import RVInstrument
    from exozippy.components.transit.transit import Transit

    for cls in (Transit, RVInstrument, MulensInstrument, AstrometryInstrument):
        keys = {entry["key"] for entry in cls.config_schema()}
        assert "time_frame" in keys, cls.__name__
        assert "time_scale" in keys, cls.__name__


def test_the_mixin_carries_no_import_edge_back_into_components():
    """Given timesystem.py, Then it imports nothing from the component tree.

    It is a leaf: a host supplies `prefix`, `names`, `config`,
    `config_manager` and `resolve_star_ndx` by duck typing. Keeping it free of
    a `Component` import is what lets a future field-neutral data base sit
    between them without an import cycle.
    """
    from exozippy.components import timesystem

    source = inspect.getsource(timesystem)
    for forbidden in ("from .component import", "from .instrument import"):
        assert forbidden not in source, forbidden
