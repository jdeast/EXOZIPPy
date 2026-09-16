"""One predicate answers "is this node already built?" (review 2.2.2).

``add_parameter``'s external-dependency branch asked ``hasattr(ext_comp,
name)`` alone, while its three siblings (the double-build guard, the local
dep lookup and the link-expression dep lookup) also required the attribute to
BE a ``Parameter``.  A component class attribute or method sharing a manifest
parameter's name would therefore have been mistaken for the built node, and
then either crashed on ``.value`` or wired the wrong object into the graph.
There are no such collisions in the tree today, which is why the odd one out
never showed; these tests pin the predicate itself and the branch that used
to skip it.
"""

import types

import numpy as np
import pymc as pm

from exozippy.components.component import Component
from exozippy.components.parameter import Parameter


class _Stub(Component):
    """A Component with no config plumbing -- only what the dep path reads."""

    def __init__(self):
        self.config = [{}]
        self.n_elements = 1

    @property
    def prefix(self):
        return "stub"

    def register_parameters(self, system):
        pass

    def build_likelihood(self, model, system):
        pass


class _Neighbour(_Stub):
    """A component whose CLASS defines `mass` -- a method, not a Parameter."""

    def __init__(self):
        super().__init__()
        self.built = []

    @property
    def prefix(self):
        return "neighbour"

    def mass(self):
        return "a method, not the node"

    def add_parameter(self, model, param_name, system, context_nodes=None):
        self.built.append(param_name)
        par = Parameter(
            label="neighbour.mass",
            initval=1.0,
            lower=0.0,
            upper=10.0,
            unit="",
            internal_unit="",
        )
        par.build_pymc()
        setattr(self, param_name, par)
        return par.value


def test_has_built_parameter_tests_the_type_not_the_name():
    """
    Given a component that carries a METHOD named like a manifest parameter
    and one that carries the built Parameter,
    When the shared predicate is asked,
    Then only the real Parameter counts as built.
    """
    # ARRANGE
    neighbour = _Neighbour()
    built = types.SimpleNamespace(
        mass=Parameter(label="x.mass", initval=1.0, unit="", internal_unit="")
    )

    # ACT / ASSERT
    assert not Component._has_built_parameter(neighbour, "mass")
    assert Component._has_built_parameter(built, "mass")
    assert not Component._has_built_parameter(neighbour, "nonexistent")


def test_external_dependency_builds_through_a_name_collision():
    """
    Given a cross-component dependency "neighbour.mass" on a component whose
    class already defines a `mass` METHOD,
    When the dependency node is resolved,
    Then the Parameter is built and its value node returned -- rather than
    the method being handed on as the node.
    """
    # ARRANGE
    stub = _Stub()
    neighbour = _Neighbour()
    system = types.SimpleNamespace(neighbour=neighbour)

    # ACT
    with pm.Model():
        name, node, aligned = stub._resolve_dep_node(
            None, system, {}, "neighbour.mass", where="stub.something"
        )

    # ASSERT
    assert neighbour.built == ["mass"]
    assert isinstance(neighbour.mass, Parameter)
    assert name == "neighbour.mass"
    assert not aligned  # a bare cross-component vector is never proven aligned
    assert (
        np.asarray(node.eval({"neighbour.mass_raw": np.zeros(1)})) is not None
    )
