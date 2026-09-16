"""Element LINKS crossed with element ROLES: the seam neither file covered.

``tests/test_linked_params.py`` never builds an ``ElementExpression`` and
``tests/test_element_parameterization.py`` never sets ``element_links``, so the
cell where the two meet was untested -- and it held a defect.  Two different
mechanisms share the params-file words ``lower:``/``upper:``:

* ``lower: 3`` (a NUMBER) is a SOFT BARRIER.  It works on a derived element and
  is the right answer there: the penalty's gradient acts on the parameters the
  element is derived FROM, which is the only thing that can move a derived
  value.  ``tests/test_per_element_soft_bound.py`` owns that path end to end;
  the regression guard here is that refusing the LINK did not narrow it.
* ``lower: <expression naming another parameter>`` (a LINK) is NOT a bound.  It
  re-maps the element's own logit coordinate into the linked interval, so the
  value BECOMES the image of that coordinate.  Correct on a SAMPLED element;
  on a derived one there is no coordinate, the logit sits at its center, and
  the ``pt.set_subtensor`` handed back the interval's midpoint in place of the
  physics -- measured, before the fix, as a derived 7.77 evaluating to 6.5 with
  no error and no warning, and as ``orbit.ecc`` going to ``+inf`` on a mixed
  V_c/V_e system, where the static upper bound is infinite.  A hard link (an
  ``initval`` link with ``sigma: 0``) overwrote it outright.

The whole-vector case was already refused; roles are per ELEMENT, so the
refusal is too.  Both directions are pinned below: what must raise, and what
must still work (a numeric bound, a ``mu`` link, and a link on the SAMPLED
element of the same mixed vector).
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest

from exozippy.components.parameter import ElementExpression, Parameter
from exozippy.system import System

# The value the expression supplies for the derived element.  Deliberately
# outside the [0, 1] interval element 0 lives in, so "the physics" and "the
# transform" can never be confused for one another.
DERIVED = 7.77


def _link(value, index=0):
    """One element_links entry whose closure returns a constant tensor.

    The shape Component._wire_user_links produces: {field: {element index:
    {"fn": callable(phys_vector) -> scalar tensor, "intra_deps": set}}}.
    """
    return {index: {"fn": lambda _phys: pt.constant(value), "intra_deps": ()}}


def _mixed(links=None, sigma=None, lower=(0.0, 0.0), upper=(1.0, 7.0), **kw):
    """A 2-element vector: element 0 SAMPLED, element 1 DERIVED.

    A vector with one element of each role is the whole point of per-element
    roles, and it is what makes "the refusal is per element" testable.
    """
    kwargs = dict(
        initval=np.array([0.5, 0.5]),
        init_scale=np.array([0.1, 0.1]),
        lower=np.array(lower, dtype=float),
        upper=np.array(upper, dtype=float),
        sigma=sigma,
        unit="",
        internal_unit="",
        shape=(2,),
        names=["i0", "i1"],
        element_expressions=[
            ElementExpression(
                mask=[False, True],
                expr=lambda: pt.constant(DERIVED),
            )
        ],
        element_links=links,
    )
    kwargs.update(kw)
    return Parameter(label="comp.p", **kwargs)


def _values(param):
    """Build `param` in a fresh model and evaluate it at the start point."""
    with pm.Model() as model:
        val = param.build_pymc()
    fn = model.compile_fn(
        model.replace_rvs_by_values([val]),
        inputs=model.value_vars,
        point_fn=True,
        on_unused_input="ignore",
    )
    return np.atleast_1d(fn(model.initial_point())[0]), model


# ---------------------------------------------------------------------------
# 1. What must be REFUSED
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field", ["lower", "upper"])
def test_a_bound_link_on_a_derived_element_is_refused(field):
    """
    Given a mixed vector whose element 1 is derived by an expression,
    When the params file puts a `lower:`/`upper:` LINK on that element,
    Then the build raises, naming the element by index and by name, saying it
      is derived, and naming the two spellings that DO work.

    Before the fix this silently returned the midpoint of the linked interval
    (lower 6.0, static upper 7.0 -> 6.5) in place of the derived 7.77.
    """
    # Arrange
    p = _mixed(links={field: _link(6.0, index=1)})

    # Act / Assert
    with pytest.raises(ValueError) as excinfo:
        _values(p)

    msg = str(excinfo.value)
    assert "comp.p'[1] ('i1')" in msg
    assert "DERIVED" in msg
    assert "MIDPOINT" in msg
    # ...and it teaches the two things that work, since the user's instinct
    # (constrain a derived quantity) is reasonable and only the MECHANISM is
    # wrong.
    assert "NUMERIC" in msg
    assert "'mu' link" in msg


def test_a_hard_link_on_a_derived_element_is_refused():
    """
    Given a mixed vector whose element 1 is derived,
    When the params file hard-links it (an `initval` link with `sigma: 0`),
    Then the build raises, naming the element and the spelling.

    Before the fix the link expression's value simply replaced the physics.
    """
    # Arrange
    p = _mixed(links={"hard": _link(3.25, index=1)})

    # Act / Assert
    with pytest.raises(ValueError, match=r"comp\.p'\[1\] \('i1'\)"):
        _values(p)


def test_the_whole_vector_refusal_still_fires_and_shares_the_text():
    """
    Given a wholly derived parameter (one expression, no per-element specs),
    When a bound link targets it,
    Then it raises as it always did -- with the same advice as the per-element
      refusal, because the two share one message.

    This is the precedent the per-element refusal was matched to; it is here so
    the two cannot drift apart unnoticed.
    """
    # Arrange
    p = Parameter(
        label="comp.q",
        initval=np.array([0.5]),
        init_scale=np.array([0.1]),
        lower=np.array([0.0]),
        upper=np.array([1.0]),
        unit="",
        internal_unit="",
        shape=(1,),
        expression=lambda: pt.constant(DERIVED),
        element_links={"lower": _link(0.25, index=0)},
    )

    # Act / Assert
    with pytest.raises(ValueError) as excinfo:
        _values(p)

    msg = str(excinfo.value)
    assert "comp.q" in msg and "[0]" not in msg
    assert "NUMERIC" in msg and "'mu' link" in msg


def test_the_live_surface_refuses_a_bound_link_on_a_vcve_orbits_ecc(tmp_path):
    """
    Given a two-orbit system with one orbit in V_c/V_e mode and one not -- so
      `orbit.ecc` is derived element by element, which is the shipped
      configuration that made this reachable,
    When a params file puts a `lower:` LINK on the V_c/V_e orbit's ecc,
    Then the build raises, naming `orbit.ecc[0]` and the orbit's own name.

    The synthetic cases above pin build_pymc; this one pins the whole path the
    user actually writes -- extract_links -> _wire_user_links -> build_pymc --
    which is the only place the per-element role and the link meet for real.
    Before the fix this built and scored a model whose ecc was `+inf` (the
    parameter has no finite static upper), crashing much later in unrelated
    physics with "eccentricity must be in the range [0, 1)".
    """
    # Arrange
    rv = tmp_path / "rv.dat"
    rng = np.random.default_rng(7)
    t = np.linspace(2459600.0, 2459660.0, 40)
    np.savetxt(
        rv,
        np.column_stack(
            [t, rng.normal(0.0, 10.0, t.size), np.full(t.size, 10.0)]
        ),
    )
    config = {
        "star": [{"name": "A", "mist": False}],
        "planet": [{"name": "b"}, {"name": "c", "orbit_ndx": 1}],
        "orbit": [
            {"name": "b", "fitvcve": True},
            {"name": "c", "fitvcve": False},
        ],
        "rvinstrument": [{"name": "inst0", "file": str(rv)}],
    }
    params = {
        "star.0.radius": {"initval": 1.61, "sigma": 0.05},
        "star.0.mass": {"initval": 1.204, "sigma": 0.05},
        "star.0.teff": {"initval": 6207, "sigma": 100},
        "star.0.feh": {"initval": -0.116, "sigma": 0.08},
        "orbit.0.period": {"initval": 2.99},
        "orbit.0.tc": {"initval": 2459634.3},
        "orbit.0.cosi": {"initval": 0.05},
        "orbit.1.period": {"initval": 7.5},
        "orbit.1.tc": {"initval": 2459634.4},
        "orbit.1.cosi": {"initval": 0.03},
        "planet.0.radius": {"initval": 1.7},
        "planet.1.radius": {"initval": 1.1},
        # The one user line the defect needed.
        "orbit.0.ecc": {"lower": "0.2 + 0.1 * star.0.feh"},
    }
    system = System(config, user_params=params)
    system.prepare()

    # Act / Assert
    with pytest.raises(ValueError) as excinfo:
        system.build_model()

    msg = str(excinfo.value)
    assert "orbit.ecc'[0] ('b')" in msg
    assert "DERIVED" in msg
    # The element really is derived per element here -- not by a whole-vector
    # expression, which the older check would have caught instead.
    assert system.orbit.ecc.expression is None
    assert bool(system.orbit.ecc.element_expressions)
    assert np.atleast_1d(system.orbit.ecc.is_derived).tolist() == [True, True]


# ---------------------------------------------------------------------------
# 2. What must still WORK
# ---------------------------------------------------------------------------


def test_a_numeric_bound_on_a_derived_element_still_gets_its_soft_barrier():
    """
    Given a derived element with a NUMERIC `lower` above its derived value,
    When the model is built,
    Then the element keeps its derived value, a soft barrier is added, the
      logp is finite, and the barrier's GRADIENT reaches the parameter the
      element is derived from.

    This is the regression guard for the scope of the refusal above.  A
    numeric bound and a bound LINK share one params-file spelling and are
    different mechanisms; refusing the link must not touch the barrier, which
    is a working, deliberately-built feature -- and the gradient on the parent
    is the whole reason a bound on a derived quantity means anything.
    """

    # Arrange -- the derived element is a function of a real model parameter,
    # so there is a parent for the barrier's gradient to reach.
    def build(with_bound):
        with pm.Model() as model:
            parent = pm.Normal("parent", 0.0, 1.0)
            p = _mixed(
                lower=(0.0, 3.5 if with_bound else -np.inf),
                upper=(1.0, np.inf),
                bound_scale=np.array([np.nan, 1.0]),
                element_expressions=[
                    ElementExpression(
                        mask=[False, True],
                        expr=lambda: 3.0 + parent,
                    )
                ],
            )
            val = p.build_pymc()
        point = model.initial_point()
        fn = model.compile_fn(
            model.replace_rvs_by_values([val]),
            inputs=model.value_vars,
            point_fn=True,
            on_unused_input="ignore",
        )
        names = [v.name for v in model.value_vars]
        grads = model.compile_dlogp()(point)
        return (
            np.atleast_1d(fn(point)[0]),
            float(model.compile_logp()(point)),
            dict(zip(names, np.atleast_1d(grads))),
            model,
        )

    # Act
    values, logp, grads, model = build(with_bound=True)
    _, base_logp, base_grads, base_model = build(with_bound=False)

    # Assert -- the physics, not the bound, still supplies the value
    assert values[1] == pytest.approx(3.0, rel=1e-12)
    assert "low_bound.comp.p" in {p.name for p in model.potentials}
    assert np.isfinite(logp)
    # The barrier is ACTIVE (3.0 is below the bound of 3.5) and pushes the
    # PARENT up, hard: the transition width is 0.01 * bound_scale, so the
    # slope there is order 1e2 nats per unit.  Without the bound the parent
    # feels only its own prior, which at the start point is exactly zero.
    assert base_grads["parent"] == pytest.approx(0.0, abs=1e-12)
    assert grads["parent"] > 100.0
    assert logp < base_logp
    # ...and the control: no barrier is created at all without the bound, so
    # the comparison above is a real difference and not two copies of one
    # graph.
    assert "low_bound.comp.p" not in {p.name for p in base_model.potentials}


def test_a_mu_link_on_a_derived_element_still_applies():
    """
    Given a derived element with a `mu` LINK and a `sigma`,
    When the model is built,
    Then the element keeps its derived value and the soft Gaussian pull toward
      the linked center is added with exactly that center.

    `mu` is the legal link channel on a derived value -- the whole-vector path
    has always exempted it -- and the refusal above must not widen to it.  It
    is also the workaround the error message points at.
    """
    # Arrange
    p = _mixed(
        links={"mu": _link(6.0, index=1)},
        sigma=np.array([np.nan, 0.5]),
    )

    # Act
    values, model = _values(p)

    # Assert
    assert values[1] == pytest.approx(DERIVED, rel=1e-12)
    (potential,) = [
        q for q in model.potentials if q.name == "link_mu.comp.p.1"
    ]
    fn = model.compile_fn(
        model.replace_rvs_by_values([potential]),
        inputs=model.value_vars,
        point_fn=True,
        on_unused_input="ignore",
    )
    term = float(np.atleast_1d(fn(model.initial_point())[0])[0])
    assert term == pytest.approx(-0.5 * ((DERIVED - 6.0) / 0.5) ** 2, rel=1e-9)


def test_a_bound_link_on_the_sampled_element_of_a_mixed_vector_still_works():
    """
    Given a mixed vector, element 0 SAMPLED and element 1 DERIVED,
    When a `lower` LINK targets the SAMPLED element,
    Then it is honored -- element 0 is the image of its own coordinate inside
      the linked interval -- and element 1 keeps its derived value.

    The refusal is per ELEMENT, not per parameter: one element's role must not
    decide another's.  The unlinked value is asserted too, so the linked number
    is a real remap rather than a coincidence.
    """
    # Arrange
    linked = _mixed(links={"lower": _link(0.25, index=0)})
    unlinked = _mixed()

    # Act
    values, _ = _values(linked)
    base, _ = _values(unlinked)

    # Assert -- at the start the logit coordinate sits at the center of its
    # interval (initval 0.5 of [0, 1] -> q = 0.5), so the remapped value is
    # 0.25 + 0.75 * 0.5.
    assert base[0] == pytest.approx(0.5, rel=1e-12)
    assert values[0] == pytest.approx(0.625, rel=1e-12)
    assert values[1] == pytest.approx(DERIVED, rel=1e-12)


def test_a_hard_link_on_the_sampled_element_of_a_mixed_vector_still_works():
    """
    Given a mixed vector whose SAMPLED element is pinned by a hard link,
    When the model is built,
    Then that element tracks the link expression and the derived element keeps
      its expression -- the hard loop's refusal is per element too.
    """
    # Arrange
    p = _mixed(
        links={"hard": _link(0.8, index=0)},
        sigma=np.array([0.0, np.nan]),
    )

    # Act
    values, _ = _values(p)

    # Assert
    assert values[0] == pytest.approx(0.8, rel=1e-12)
    assert values[1] == pytest.approx(DERIVED, rel=1e-12)
