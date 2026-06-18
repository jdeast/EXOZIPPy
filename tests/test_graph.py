"""Tests for graph.py: determine_pymc_build_order topological sort."""
import pytest

from exozippy.graph import determine_pymc_build_order


class _FakeComp:
    def __init__(self, prefix, manifest):
        self.prefix = prefix
        self.manifest = manifest
        self.n_elements = 1


class _FakeConfigManager:
    """Returns dep lists for named params; everything else is dep-free."""

    def __init__(self, dep_map=None):
        self._dep_map = dep_map or {}

    def resolve(self, prefix, param_name, shape=None, names=None):
        key = f"{prefix}.{param_name}"
        deps = self._dep_map.get(key, [])
        if deps:
            return {"expressions": {"default": {"deps": deps}}}
        return {}


def test_independent_parameters_all_appear_in_result():
    """
    Given two components each with one independent parameter,
    When determine_pymc_build_order is called,
    Then both global keys appear in the result.
    """
    comps = {
        "compA": _FakeComp("compA", {"x": None}),
        "compB": _FakeComp("compB", {"y": None}),
    }
    order = determine_pymc_build_order(comps, _FakeConfigManager())
    assert "compA.x" in order
    assert "compB.y" in order


def test_dependency_precedes_dependent_parameter():
    """
    Given compA has 'x' (no deps) and 'y' that depends on 'x',
    When determine_pymc_build_order is called,
    Then 'compA.x' appears before 'compA.y' in the result.
    """
    comps = {"compA": _FakeComp("compA", {"x": None, "y": None})}
    cm = _FakeConfigManager(dep_map={"compA.y": ["x"]})  # intra-component dep
    order = determine_pymc_build_order(comps, cm)
    assert order.index("compA.x") < order.index("compA.y")


def test_cross_component_dependency_is_respected():
    """
    Given compB.y explicitly depends on compA.x (cross-component),
    When determine_pymc_build_order is called,
    Then 'compA.x' precedes 'compB.y' in the result.
    """
    comps = {
        "compA": _FakeComp("compA", {"x": None}),
        "compB": _FakeComp("compB", {"y": None}),
    }
    cm = _FakeConfigManager(dep_map={"compB.y": ["compA.x"]})
    order = determine_pymc_build_order(comps, cm)
    assert order.index("compA.x") < order.index("compB.y")


def test_missing_dependency_raises_value_error():
    """
    Given compA.y depends on 'compB.z' but compB has no 'z' in its manifest,
    When determine_pymc_build_order is called,
    Then a ValueError is raised naming the missing dependency.
    """
    comps = {
        "compA": _FakeComp("compA", {"y": None}),
        "compB": _FakeComp("compB", {"w": None}),
    }
    cm = _FakeConfigManager(dep_map={"compA.y": ["compB.z"]})
    with pytest.raises(ValueError, match="compB.z"):
        determine_pymc_build_order(comps, cm)


def test_circular_dependency_raises_value_error():
    """
    Given compA.x depends on compA.y and compA.y depends on compA.x,
    When determine_pymc_build_order is called,
    Then a ValueError naming a circular reference is raised.
    """
    comps = {"compA": _FakeComp("compA", {"x": None, "y": None})}
    cm = _FakeConfigManager(dep_map={"compA.x": ["y"], "compA.y": ["x"]})
    with pytest.raises(ValueError, match="[Cc]ircular"):
        determine_pymc_build_order(comps, cm)


def test_empty_active_components_returns_empty_list():
    """
    Given no active components,
    When determine_pymc_build_order is called,
    Then an empty list is returned without error.
    """
    order = determine_pymc_build_order({}, _FakeConfigManager())
    assert order == []
