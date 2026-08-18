"""Tests for components/factory.py: discover_components auto-discovery."""

import importlib

from exozippy.components import factory
from exozippy.components.component import Component
from exozippy.components.factory import discover_components


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


def test_discover_components_keys_on_the_lowercase_class_name():
    """
    Given Band declares no yaml_key -- the factory's fallback is the
      lowercase class name, which is already 'band',
    When discover_components scans the components directory,
    Then registry['band'] resolves to the Band class.

    Review 4.2.4: three components restated that fallback as an explicit
    `yaml_key`, which read as "this one differs from its class name" when
    none of them did.  The override mechanism stays for a class that ever
    does; nothing in the tree uses it.
    """
    from exozippy.components.band.band import Band

    registry = discover_components()
    assert registry.get("band") is Band, (
        f"Expected registry['band'] to be Band, got {registry.get('band')}"
    )
    assert not hasattr(Band, "yaml_key")


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


def test_a_broken_component_module_does_not_abort_discovery(monkeypatch):
    """
    Given ONE component module that raises at import time with something
    other than ImportError (a SyntaxError, a NameError, a failed module-scope
    table build),
    When discover_components sweeps the tree,
    Then every other component is still discovered and the broken one is
    recorded in import_failures() -- so a config that names it still fails
    loudly, and every fit that does not is unaffected.

    Review 2.2.3: the sweep caught ImportError alone, so the comment's
    promise that an unused broken component "shouldn't break the code" held
    for a missing dependency and not for a typo -- one of those aborted
    discovery for every fit and every GUI open.
    """
    # ARRANGE
    real_import = importlib.import_module

    def _fake_import(name, *args, **kwargs):
        if name.endswith(".band.band"):
            raise SyntaxError("invalid syntax (band.py, line 1)")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(factory.importlib, "import_module", _fake_import)

    try:
        # ACT
        registry = discover_components()

        # ASSERT
        assert "star" in registry
        assert "band" not in registry
        failures = factory.import_failures()
        assert "band" in failures
        assert isinstance(failures["band"][1], SyntaxError)
    finally:
        # Leave the module-level failure record clean for the next test.
        monkeypatch.undo()
        discover_components()
