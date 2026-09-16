"""Contract tests for the scaffolding the stellar-relation components share.

`components/relations.py` holds the wiring `components/mann` and
`components/torres` had a byte-identical copy of each (review item 4.3):
instance naming, `star:` resolution, `constrain:` parsing, the star_map, the
masked Gaussian potential, and the calibration-range warning.

What is deliberately NOT shared is the statistics -- Mann's fractional scatter
about a sampled prediction (which keeps its -log(sigma) normalization) versus
Torres's constant scatter in dex (which drops it).  `test_the_two_relations_*`
below pins both halves: that the scaffolding is one implementation, and that
the normalization stays an explicit per-relation argument with no default.
"""

import inspect
import logging

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest

from exozippy.components.mann.mann import Mann
from exozippy.components.relations import (
    CONSTRAINABLE,
    StellarRelation,
    constrain_schema_entry,
    star_initval,
    star_schema_entry,
)
from exozippy.components.torres.torres import Torres


class _FakeStar:
    names = ["A", "B", "C"]
    n_elements = 3


class _FakeSystem:
    star = _FakeStar()


class _Relation(StellarRelation):
    """A minimal user of the mixin, with no relation attached.

    Deliberately not a Component: the mixin's contract is exactly that it
    needs nothing from the lifecycle beyond `prefix`, `names`, `constrain`
    and `star_indices`.
    """

    prefix = "toy"

    def __init__(self, constrain=(), star_indices=()):
        self.constrain = [set(c) for c in constrain]
        self.star_indices = list(star_indices)
        self.names = [f"i{k}" for k in range(max(len(self.constrain), 1))]


def _bare():
    """A mixin instance with nothing configured, for the parsing helpers."""
    return _Relation()


# ----------------------------------------------------------------------
# Star resolution
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("A", 0),
        ("B", 1),
        ("star.C", 2),
        (1, 1),
        ("2", 2),
        ("star.0", 0),
    ],
)
def test_a_star_resolves_by_name_path_or_index(raw, expected):
    """
    Given a `star:` value written as a name, a star.X path or an index,
    When the shared resolver runs,
    Then it returns that star's index.
    """
    # Arrange
    comp = _bare()

    # Act
    idx = comp._resolve_star(_FakeSystem(), "i0", raw)

    # Assert
    assert idx == expected


def test_an_unknown_star_names_the_available_ones():
    """
    Given a `star:` naming a star that is not in the system,
    When the shared resolver runs,
    Then it raises and lists the stars the user could have meant.
    """
    # Arrange
    comp = _bare()

    # Act / Assert
    with pytest.raises(
        ValueError, match=r"unknown star 'Z'.*\['A', 'B', 'C'\]"
    ):
        comp._resolve_star(_FakeSystem(), "i0", "Z")


def test_an_out_of_range_star_index_is_rejected():
    """
    Given a numeric `star:` past the end of the star list,
    When the shared resolver runs,
    Then it raises rather than silently indexing out of range.

    The message says "out of range", not "unknown star": since the shared
    component.resolve_star_ref took over the translation, an index that
    resolves but does not exist is diagnosed as the range error it is.
    """
    # Arrange
    comp = _bare()

    # Act / Assert
    with pytest.raises(ValueError, match="out of range"):
        comp._resolve_star(_FakeSystem(), "i0", 7)


def test_a_missing_star_key_is_rejected():
    """
    Given a relation block with no `star:` key,
    When the shared resolver runs,
    Then it raises saying the key is required.
    """
    # Arrange
    comp = _bare()

    # Act / Assert
    with pytest.raises(ValueError, match="'star:' key is required"):
        comp._resolve_star(_FakeSystem(), "i0", None)


# ----------------------------------------------------------------------
# constrain: parsing
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, {"mass", "radius"}),
        (["mass"], {"mass"}),
        (["radius"], {"radius"}),
        (["mass", "radius"], {"mass", "radius"}),
        ("mass", {"mass"}),
        (["mass", "mass"], {"mass"}),
    ],
)
def test_constrain_parses_to_a_set_of_quantities(raw, expected):
    """
    Given a `constrain:` written as a list, a bare string or omitted,
    When the shared parser runs,
    Then it returns the set of quantities to constrain.
    """
    # Arrange
    comp = _bare()

    # Act
    con = comp._parse_constrain("i0", raw)

    # Assert
    assert con == expected


@pytest.mark.parametrize(
    "raw,match",
    [
        (["mass", "teff"], r"unknown 'constrain:' entries \['teff'\]"),
        (["logg"], r"unknown 'constrain:' entries \['logg'\]"),
        ([], "'constrain:' is empty"),
        (set(), "'constrain:' is empty"),
    ],
)
def test_bad_constrain_entries_are_rejected(raw, match):
    """
    Given a `constrain:` naming something the relations cannot predict, or
    nothing at all,
    When the shared parser runs,
    Then it raises a ValueError naming the problem.
    """
    # Arrange
    comp = _bare()

    # Act / Assert
    with pytest.raises(ValueError, match=match):
        comp._parse_constrain("i0", raw)


def test_the_error_message_carries_the_component_prefix():
    """
    Given two relation components sharing one parser,
    When each rejects the same bad config,
    Then the message names the component the user actually wrote.
    """
    # Arrange
    mann, torres = Mann([{"star": "B"}], None), Torres([{"star": "A"}], None)

    # Act / Assert
    with pytest.raises(ValueError, match="^mann 'B': unknown 'constrain:'"):
        mann._parse_constrain("B", ["teff"])
    with pytest.raises(ValueError, match="^torres 'A': unknown 'constrain:'"):
        torres._parse_constrain("A", ["teff"])


# ----------------------------------------------------------------------
# Instance naming and the star map
# ----------------------------------------------------------------------


def test_instances_are_named_after_their_star():
    """
    Given relation blocks with no explicit `name:`,
    When the component is constructed,
    Then each instance takes its star's bare name (so the base class's
    duplicate-name check enforces one instance per star).
    """
    # Arrange / Act
    comp = Mann([{"star": "B"}, {"star": "star.C"}], config_manager=None)

    # Assert
    assert comp.names == ["B", "C"]


def test_an_explicit_name_is_left_alone():
    """
    Given a relation block that sets `name:` itself,
    When the component is constructed,
    Then the naming step does not overwrite it.
    """
    # Arrange / Act
    comp = Torres([{"star": "A", "name": "custom"}], config_manager=None)

    # Assert
    assert comp.names == ["custom"]


def test_build_maps_makes_an_integer_star_map():
    """
    Given resolved star indices,
    When build_maps runs,
    Then star_map is an integer array in instance order.
    """
    # Arrange
    comp = _Relation(constrain=[{"mass"}, {"mass"}], star_indices=[2, 0])

    # Act
    comp.build_maps()

    # Assert
    assert comp.star_map.tolist() == [2, 0]
    assert np.issubdtype(comp.star_map.dtype, np.integer)


# ----------------------------------------------------------------------
# star_initval
# ----------------------------------------------------------------------


class _Param:
    def __init__(self, initval):
        self.initval = initval


@pytest.mark.parametrize(
    "param,idx,expected",
    [
        (_Param(np.array([1.0, 2.0, 3.0])), 2, 3.0),
        (_Param(np.array([7.0])), 2, 7.0),  # scalar broadcasts to every star
        (_Param(5.0), 0, 5.0),
        (_Param(None), 0, None),
        (_Param(np.array([1.0, np.nan])), 1, None),
        (object(), 0, None),  # no initval attribute at all
    ],
)
def test_star_initval_reads_one_element_or_gives_up(param, idx, expected):
    """
    Given a Parameter-like object holding an initval,
    When the shared reader asks for one element,
    Then it returns that element, or None when there is nothing usable.
    """
    # Act
    got = star_initval(param, idx)

    # Assert
    assert got == expected or (got is None and expected is None)


# ----------------------------------------------------------------------
# The calibration-range warning
# ----------------------------------------------------------------------


def test_a_star_outside_the_range_warns_and_does_not_raise(caplog):
    """
    Given a star starting outside a relation's calibration range,
    When the shared range check runs,
    Then it logs one warning per offending instance and returns normally --
    a -inf wall would have no gradient for NUTS, so nothing here bounds the
    posterior.
    """
    # Arrange
    comp = _Relation(constrain=[{"mass"}, {"mass"}], star_indices=[0, 1])
    param = _Param(np.array([0.5, 3.0]))

    # Act
    with caplog.at_level(logging.WARNING):
        comp._warn_outside_range(
            _FakeSystem(),
            param,
            0.075,
            0.7,
            message="star '{star}' starts at {value:.3f} solMass",
        )

    # Assert
    msgs = [r.message for r in caplog.records]
    assert msgs == ["toy 'i1': star 'B' starts at 3.000 solMass"]


def test_a_missing_or_nan_initval_is_skipped(caplog):
    """
    Given a star whose start value is unknown,
    When the shared range check runs,
    Then it says nothing rather than comparing against NaN.
    """
    # Arrange
    comp = _Relation(constrain=[{"mass"}], star_indices=[0])

    # Act
    with caplog.at_level(logging.WARNING):
        comp._warn_outside_range(
            _FakeSystem(),
            _Param(np.array([np.nan])),
            0.075,
            0.7,
            message="star '{star}' starts at {value:.3f}",
        )

    # Assert
    assert caplog.records == []


# ----------------------------------------------------------------------
# The masked penalty
# ----------------------------------------------------------------------


def _penalty_value(constrain, which, observed, predicted, sigma, normalize):
    """Build the masked potential alone and evaluate it."""
    comp = _Relation(constrain=constrain, star_indices=range(len(constrain)))
    with pm.Model() as model:
        comp._add_penalty(
            which,
            pt.as_tensor_variable(np.asarray(observed, dtype=float)),
            pt.as_tensor_variable(np.asarray(predicted, dtype=float)),
            pt.as_tensor_variable(np.asarray(sigma, dtype=float)),
            normalize=normalize,
        )
    return model


def test_only_the_instances_that_asked_contribute():
    """
    Given two instances, one constraining mass and one constraining radius,
    When the mass penalty is built,
    Then only the mass instance's residual is in it -- the other element is
    zeroed in place, so the vector stays aligned with star_map.
    """
    # Arrange / Act
    model = _penalty_value(
        constrain=[{"mass"}, {"radius"}],
        which="mass",
        observed=[1.0, 100.0],
        predicted=[0.0, 0.0],
        sigma=[1.0, 1.0],
        normalize=False,
    )

    # Assert
    (pot,) = model.potentials
    assert pot.name == "toy.mass_prior"
    assert float(pot.eval()) == pytest.approx(-0.5)


def test_no_potential_at_all_when_nobody_asked():
    """
    Given no instance constraining radius,
    When the radius penalty is built,
    Then no potential is registered -- not a zero one.
    """
    # Arrange / Act
    model = _penalty_value(
        constrain=[{"mass"}],
        which="radius",
        observed=[1.0],
        predicted=[0.0],
        sigma=[1.0],
        normalize=False,
    )

    # Assert
    assert list(model.potentials) == []


def test_normalize_toggles_exactly_the_log_sigma_term():
    """
    Given the same residual and sigma,
    When the penalty is built with and without normalization,
    Then the two differ by exactly -log(sigma) -- the one statistical choice
    the two relation components make differently.
    """
    # Arrange
    kwargs = dict(
        constrain=[{"mass"}],
        which="mass",
        observed=[1.0],
        predicted=[0.0],
        sigma=[2.0],
    )

    # Act
    plain = float(
        list(_penalty_value(normalize=False, **kwargs).potentials)[0].eval()
    )
    normed = float(
        list(_penalty_value(normalize=True, **kwargs).potentials)[0].eval()
    )

    # Assert
    assert plain == pytest.approx(-0.5 * (1.0 / 2.0) ** 2)
    assert normed - plain == pytest.approx(-np.log(2.0))


def test_normalize_has_no_default():
    """
    Given the -log(sigma) normalization is a per-relation decision,
    When a caller omits it,
    Then _add_penalty refuses rather than picking one silently.
    """
    # Arrange
    comp = _Relation(constrain=[{"mass"}], star_indices=[0])

    # Act / Assert
    with pm.Model():
        with pytest.raises(TypeError):
            comp._add_penalty(
                "mass",
                pt.as_tensor_variable(np.zeros(1)),
                pt.as_tensor_variable(np.zeros(1)),
                pt.as_tensor_variable(np.ones(1)),
            )


# ----------------------------------------------------------------------
# The dedup itself, and the line it must not cross
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "attr",
    [
        "__init__",
        "_resolve_star",
        "_parse_constrain",
        "build_maps",
        "_add_penalty",
        "_warn_outside_range",
        "compile_plotters",
        "plot",
    ],
)
def test_the_two_relations_share_one_copy_of_the_scaffolding(attr):
    """
    Given review item 4.3 (mann and torres carried identical scaffolding),
    When the shared attribute is looked up on each component,
    Then both resolve to the same function object on the mixin.
    """
    # Assert
    assert getattr(Mann, attr) is getattr(StellarRelation, attr)
    assert getattr(Torres, attr) is getattr(StellarRelation, attr)


@pytest.mark.parametrize(
    "name", ["load_data", "register_parameters", "build_likelihood", "prefix"]
)
def test_the_mixin_owns_no_statistics(name):
    """
    Given the locked decision that these stay separate components,
    When the mixin is inspected,
    Then it defines no lifecycle stage that produces a number, so the two
    sigma conventions cannot drift one inheritance level apart.
    """
    # Assert
    assert not hasattr(StellarRelation, name)


def test_each_component_states_its_own_normalization_at_the_call_site():
    """
    Given Mann's sigma scales with its prediction and Torres's is constant,
    When their build_likelihood sources are read,
    Then each passes `normalize=` explicitly, and they disagree.
    """

    # Arrange -- comments are stripped: each build_likelihood also explains in
    # prose why the *other* relation chooses differently, which is the point.
    def code(fn):
        return "\n".join(
            ln
            for ln in inspect.getsource(fn).splitlines()
            if not ln.strip().startswith("#")
        )

    mann_src = code(Mann.build_likelihood)
    torres_src = code(Torres.build_likelihood)

    # Assert
    assert mann_src.count("normalize=True") == 2
    assert "normalize=False" not in mann_src
    assert torres_src.count("normalize=False") == 2
    assert "normalize=True" not in torres_src


def test_the_schema_helpers_describe_the_shared_keys():
    """
    Given both components declare the same two config keys,
    When their schemas are built from the shared helpers,
    Then the keys and accepted values match, and only the doc differs.
    """
    # Arrange
    mann_keys = {e["key"]: e for e in Mann.config_schema()}
    torres_keys = {e["key"]: e for e in Torres.config_schema()}

    # Assert
    assert mann_keys["star"] == star_schema_entry("Mann")
    assert torres_keys["star"] == star_schema_entry("Torres")
    assert mann_keys["constrain"] == constrain_schema_entry("absolute Ks")
    assert torres_keys["constrain"] == constrain_schema_entry("teff/logg/feh")
    assert mann_keys["constrain"]["accepts"] == list(CONSTRAINABLE)
    assert torres_keys["constrain"]["accepts"] == list(CONSTRAINABLE)
