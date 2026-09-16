"""One answer to "is component X in the topology?" (review 4.8.1).

The question was re-derived five ways -- three in star.py, two in orbit.py --
and they were NOT equivalent.  Each walked its own chain of holders, and the
disagreements sat exactly where a partially constructed system lives:

* ``Star.register_parameters``' local helper read ``system.config`` OR
  ``config_manager.system_config`` as an ``elif``, so a system carrying both
  but whose ``config`` lacked the key never consulted the second;
* ``Orbit._topology`` and ``Orbit.register_parameters``' inline check never
  consulted ``config_manager.system_config`` at all;
* only ``Star._galactic_imf`` looked at ``active_components``, the one holder
  carrying a BUILT instance.

``components.component.in_topology`` is the union, and these tests are about
the chain rather than about any one caller -- a sixth re-derivation is what
they exist to make pointless.
"""

import pytest

from exozippy.components.component import in_topology


class _Holder:
    """A stand-in system, with only the attributes each case is about."""

    def __init__(self, **attrs):
        for key, value in attrs.items():
            setattr(self, key, value)


class _ConfigManager:
    def __init__(self, system_config):
        self.system_config = system_config


# ----------------------------------------------------------------------
# The chain, in order of decreasing authority
# ----------------------------------------------------------------------
def test_a_built_instance_wins():
    """
    Given a system whose active_components holds the component,
    When in_topology is asked,
    Then it returns that instance.

    active_components is populated in System.__init__, so it is available
    from stage 1 -- and it is the only holder that carries the object, which
    is what a caller reading a config key off the component needs.
    """
    # Arrange
    built = object()
    system = _Holder(active_components={"sed": built})

    # Act / Assert
    assert in_topology(system, "sed") is built


def test_the_attribute_form_is_found():
    """
    Given a system that exposes the component only as an attribute,
    When in_topology is asked,
    Then it returns it.

    This is the shape the suite's mock systems use, and it is why the helper
    is a module function rather than only a System method: every caller is a
    component holding a `system` that may be a test double.
    """
    # Arrange
    comp = object()
    system = _Holder(sed=comp)

    # Act / Assert
    assert in_topology(system, "sed") is comp


def test_the_raw_config_block_counts_as_topology():
    """
    Given a config naming a component that has not been instantiated,
    When in_topology is asked,
    Then it returns the raw block.

    Deliberate: a premature `evolutionarymodel:` block that no component
    backs still IS topology, and Star.register_parameters depends on seeing
    it so it can warn about the likelihood-free dimensions it would create.
    """
    # Arrange
    block = [{"name": "x"}]
    system = _Holder(config={"evolutionarymodel": block})

    # Act / Assert
    assert in_topology(system, "evolutionarymodel") is block


def test_config_manager_system_config_is_consulted_even_when_config_exists():
    """
    Given a system carrying BOTH a config that lacks the key and a
      config_manager.system_config that has it,
    When in_topology is asked,
    Then it finds it.

    This is the concrete disagreement review 4.8.1 found: the star helper
    used an `elif`, so the presence of ANY `system.config` shadowed the
    second holder entirely, while the sibling helper two hundred lines away
    checked both.
    """
    # Arrange
    block = {"IMF": "Salpeter"}
    system = _Holder(
        config={"star": []},
        config_manager=_ConfigManager({"galacticmodel": block}),
    )

    # Act / Assert
    assert in_topology(system, "galacticmodel") is block


def test_an_absent_component_is_none():
    """
    Given a system with none of the holders naming the component,
    When in_topology is asked,
    Then it returns None.
    """
    # Arrange
    system = _Holder(
        active_components={}, config={}, config_manager=_ConfigManager({})
    )

    # Act / Assert
    assert in_topology(system, "sed") is None


def test_a_system_missing_every_holder_does_not_raise():
    """
    Given an object with no active_components, config or config_manager,
    When in_topology is asked,
    Then it returns None rather than raising.

    A topology-driven DEFAULT must not crash on a partially constructed
    system -- it should simply see nothing.  Orbit's deleted `_topology`
    helper carried that contract in its docstring and it is preserved here.
    """
    # Act / Assert
    assert in_topology(_Holder(), "transit") is None
    assert in_topology(None, "transit") is None


# ----------------------------------------------------------------------
# The one distinction callers have to know about
# ----------------------------------------------------------------------
@pytest.mark.parametrize("empty", [{}, []])
def test_an_empty_block_is_still_topology(empty):
    """
    Given a config naming a component with an EMPTY block,
    When in_topology is asked,
    Then it returns the block, which is not None.

    `sed: {}` is a system WITH an SED, so this is membership and not
    truthiness -- and it is tested because it is the one place callers can
    still get it wrong: they must compare against None, never test the
    return value's truthiness.  tests/test_star_evolutionary_model.py fakes
    its topology with exactly `evolutionarymodel: {}`, so the empty case is
    load-bearing rather than hypothetical.
    """
    # Arrange
    system = _Holder(config={"sed": empty})

    # Act
    found = in_topology(system, "sed")

    # Assert
    assert found is not None
    assert found == empty


# ----------------------------------------------------------------------
# The System method is the same implementation
# ----------------------------------------------------------------------
def test_the_system_method_delegates_to_the_function():
    """
    Given a real System,
    When System.in_topology is called,
    Then it agrees with the module function.

    The method exists for callers that hold a System (the GUI, scripts); it
    delegates rather than reimplementing, which is the entire point of the
    item.
    """
    # Arrange
    from exozippy.system import System

    config = {
        "star": [{"name": "A"}],
        "planet": [{"name": "b", "star_ndx": 0, "orbit_ndx": 0}],
        "orbit": [{"name": "b"}],
    }
    system = System(config, {})

    # Act / Assert
    for name in ("star", "planet", "orbit", "sed", "galacticmodel"):
        assert system.in_topology(name) is in_topology(system, name)
    assert system.in_topology("orbit") is not None
    assert system.in_topology("sed") is None
