"""Tests for components/factory.py: discover_components auto-discovery."""
from exozippy.components.factory import discover_components
from exozippy.components.component import Component


def test_discover_components_returns_core_yaml_keys():
    """
    Given the standard EXOZIPPy component tree,
    When discover_components is called,
    Then the registry contains the expected YAML keys for all core components.
    """
    registry = discover_components()
    expected = {"star", "band", "galacticmodel"}
    missing = expected - set(registry.keys())
    assert not missing, f"Missing component keys: {missing}"


def test_discover_components_uses_yaml_key_attribute_over_class_name():
    """
    Given Band sets yaml_key = 'band' (class name would map to 'band' either way,
      but the attribute lookup is explicit),
    When discover_components scans the components directory,
    Then registry['band'] resolves to the Band class.
    """
    from exozippy.components.band.band import Band
    registry = discover_components()
    assert registry.get("band") is Band, (
        f"Expected registry['band'] to be Band, got {registry.get('band')}"
    )


def test_discover_components_all_values_are_component_subclasses():
    """
    Given the discovered registry,
    When checking every value in it,
    Then each is a proper subclass of Component (not Component itself).
    """
    registry = discover_components()
    for key, cls in registry.items():
        assert issubclass(cls, Component) and cls is not Component, (
            f"Registry entry '{key}' ({cls}) is not a proper Component subclass"
        )


def test_discover_components_returns_dict():
    """
    Given the component directory exists,
    When discover_components is called,
    Then the return value is a non-empty dict.
    """
    registry = discover_components()
    assert isinstance(registry, dict)
    assert len(registry) > 0
