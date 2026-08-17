"""Per-element parameterization: one instance's coordinates are not another's.

``Parameter.build_pymc`` used to set ``is_derived`` for a WHOLE vector
(``np.full(n_elements, expr_raw is not None)``), so every modeling choice that
is really per instance had to be uniform across a component.  Four shipped
features paid for it: ``band.ld_law`` raised on a system mixing quadratic and
linear bands, ``planet.mass_parameterization`` raised on explicit disagreement
and silently fell back to all-linear on an implicit one, ``star.mist``'s
``mask`` was declared and never read (so a premature ``evolutionarymodel:``
block materialized three free likelihood-free dimensions), and
``orbit.fitvcve`` named the unconsumed ``mask`` field as its real blocker,
ahead of the missing physics.

The primitive is four per-element roles, interpreted in ``manifest.py`` and
consumed in ``build_pymc``: SAMPLED, DERIVED (an expression the model
consumes), REPORTED (an expression nothing consumes -- declared, built with its
first consumer) and INACTIVE (not a parameter of that instance at all).

What these tests pin, in order:
  1. the vocabulary (selector normalization, per-element expr_key, overlap and
     sizing errors);
  2. the build: mixed derived/sampled vectors, inactive pins, which elements
     get raw coordinates and potentials, and a finite gradient;
  3. the dependency slicing that keeps an unused instance's numbers out of the
     other parameterization's physics, its static alignment proof, and the
     start-point check that catches a non-elementwise expression;
  4. that the whole-vector paths are untouched (a bit-identical graph).
"""

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest
import yaml

from exozippy.components.component import Component
from exozippy.components.parameter import ElementExpression, Parameter
from exozippy.manifest import (
    ElementSelectorError,
    interpret_manifest_entry,
    normalize_selector,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _param(label="comp.p", n=2, **kwargs):
    """A two-element sampled Parameter with everything the build needs."""
    defaults = dict(
        initval=np.full(n, 0.5),
        init_scale=np.full(n, 0.1),
        lower=np.zeros(n),
        upper=np.ones(n),
        unit="",
        internal_unit="",
        shape=(n,),
        names=[f"i{i}" for i in range(n)],
    )
    defaults.update(kwargs)
    return Parameter(label=label, **defaults)


def _raw_names(model):
    return sorted(v.name for v in model.free_RVs)


# ---------------------------------------------------------------------------
# 1. The vocabulary
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "selector,expected",
    [
        (None, [True, True, True]),
        (True, [True, True, True]),
        (False, [False, False, False]),
        ([True, False, True], [True, False, True]),
        ([0, 2], [True, False, True]),
        (np.array([1]), [False, True, False]),
    ],
)
def test_every_selector_spelling_normalizes_to_one_mask(selector, expected):
    """
    Given the selector spellings components already write by hand,
    When normalize_selector reads them for a 3-element parameter,
    Then each becomes the same boolean mask.

    star passes a list of bools, orbit a numpy bool array and
    PriorContribution an index list; one normalizer serves all three so the
    roles cannot disagree with the priors about which element is which.
    """
    assert normalize_selector(selector, 3).tolist() == expected


def test_a_mask_sized_from_the_wrong_element_count_raises():
    """
    Given a boolean mask whose length is not the parameter's element count,
    When it is normalized,
    Then ElementSelectorError names both counts and says where the size
      should come from.

    This is review 1.1.1's hazard in a new place: a mask sized from the
    component's CONFIG LIST is silently short for any parameter whose vector is
    longer (lens has one config entry and per-source vectors), and a short mask
    would mark real elements inactive with no message.
    """
    with pytest.raises(ElementSelectorError) as exc:
        normalize_selector([True, False], 3, "lens.t_0")

    msg = str(exc.value)
    assert "lens.t_0" in msg
    assert "length 2" in msg and "3 element" in msg


def test_expr_key_may_select_a_different_expression_per_element():
    """
    Given a manifest entry whose expr_key is a dict of block -> selector,
    When its expression configs are read for a 2-element parameter,
    Then one selection per block comes back, each with its own mask, and the
      entry still reports itself as naming an expression.
    """
    entry = interpret_manifest_entry(
        {"expr_key": {"from_hk": [True, False], "from_vcve": [False, True]}}
    )

    sels = entry.expression_configs(
        {"from_hk": {"func_name": "a"}, "from_vcve": {"func_name": "b"}},
        n_elements=2,
        where="orbit.ecc",
    )

    assert entry.names_expression and entry.is_per_element
    assert [(s.key, s.mask.tolist()) for s in sels] == [
        ("from_hk", [True, False]),
        ("from_vcve", [False, True]),
    ]


def test_two_expressions_claiming_one_element_raises():
    """
    Given per-element selectors that overlap,
    When the expression configs are read,
    Then ElementSelectorError names the contested element.

    An element takes its value from exactly one place.  Picking one silently is
    how the three hand-written manifest readers used to disagree, which is the
    whole reason this module exists.
    """
    entry = interpret_manifest_entry(
        {"expr_key": {"a": [True, True], "b": [True, False]}}
    )

    with pytest.raises(ElementSelectorError, match=r"\[0\]"):
        entry.expression_configs({"a": {}, "b": {}}, 2, "comp.p")


def test_the_single_block_accessor_refuses_a_per_element_entry():
    """
    Given a per-element entry,
    When the old single-block accessor is called,
    Then it raises and points at expression_configs.

    Answering with one of the blocks would be a consumer silently modeling
    every instance with one instance's physics -- exactly what the uniform
    is_derived did.
    """
    entry = interpret_manifest_entry({"expr_key": {"a": [True, False]}})

    with pytest.raises(TypeError, match="expression_configs"):
        entry.expression_config({"a": {}}, where="comp.p")


def test_activity_mask_defaults_to_every_element_active():
    """
    Given entries with and without a `mask` option,
    When their activity masks are read,
    Then a missing mask means every element is active and is_per_element is
      False, so nothing that predates this vocabulary changes behavior.
    """
    plain = interpret_manifest_entry({"shape": (3,)})
    masked = interpret_manifest_entry({"mask": [True, False, True]})

    assert plain.activity_mask(3).tolist() == [True, True, True]
    assert plain.is_per_element is False
    assert masked.activity_mask(3).tolist() == [True, False, True]
    assert masked.is_per_element is True


# ---------------------------------------------------------------------------
# 2. The build
# ---------------------------------------------------------------------------


def test_a_mixed_vector_samples_one_element_and_derives_the_other():
    """
    Given a 2-element Parameter whose element 1 is supplied by an expression,
    When it is built,
    Then element 0 keeps a raw coordinate, element 1 takes the expression's
      value, and the per-element role masks say exactly that.

    This is the primitive: before it, one expression made the whole vector
    derived and the other instance lost its sampled coordinate.
    """
    other = pt.as_tensor_variable(np.array([7.0, 0.25]))
    p = _param(
        element_expressions=[
            ElementExpression(mask=[False, True], expr=lambda: other)
        ]
    )

    with pm.Model() as model:
        val = p.build_pymc()

    assert _raw_names(model) == ["comp.p_raw"]
    assert p.is_sampled.tolist() == [True, False]
    assert p.is_derived.tolist() == [False, True]
    assert p.element_is_sampled(0) and not p.element_is_sampled(1)
    assert p.element_is_derived(1) and not p.element_is_derived(0)
    # The raw vector has ONE entry (element 0), and element 1 reads the
    # expression rather than the transform.
    assert np.size(p.raw_initval) == 1
    assert float(np.atleast_1d(val.eval())[1]) == pytest.approx(0.25)


def test_the_sampled_half_of_a_mixed_vector_starts_at_its_initval():
    """
    Given a mixed vector,
    When it is built and evaluated at raw = 0,
    Then the sampled element's physical value is its initval.

    The whitening probe, every multi-seed start and the startup table all
    assume raw = 0 maps to the start value; a mixed vector must not be the
    exception.
    """
    p = _param(
        initval=np.array([0.3, 0.9]),
        element_expressions=[
            ElementExpression(
                mask=[False, True],
                expr=lambda: pt.as_tensor_variable(np.array([0.0, 0.42])),
            )
        ],
    )

    with pm.Model() as model:
        val = p.build_pymc()

    at_start = model.compile_fn(
        model.replace_rvs_by_values([val]), point_fn=True
    )
    start = float(np.atleast_1d(at_start(model.initial_point())[0])[0])

    assert start == pytest.approx(0.3, rel=1e-9)


def test_an_inactive_element_is_pinned_unreported_and_potential_free():
    """
    Given a Parameter whose element 1 is masked out as inactive,
    When it is built,
    Then element 1 is neither sampled nor derived, it is held at its value,
      element_is_active reports it False, and no potential mentions it.

    The EEP case: for a star with no evolutionary model the parameter does not
    exist, so it must not consume a dimension, must not add a prior term, and
    must not be reported -- "at best meaningless and at worst misleading".
    """
    p = _param(mask=[True, False])

    with pm.Model() as model:
        val = p.build_pymc()

    assert p.is_sampled.tolist() == [True, False]
    assert p.is_active.tolist() == [True, False]
    assert p.element_is_active(0) and not p.element_is_active(1)
    assert np.size(p.raw_initval) == 1  # only the active element is sampled
    assert float(np.atleast_1d(val.eval())[1]) == pytest.approx(0.5)
    # The logit correction potential covers the sampled element only; nothing
    # in the model is a barrier or prior on the inactive one.
    assert [v.name for v in model.potentials] == ["logit_uniform_prior.comp.p"]


def test_an_inactive_element_can_be_pinned_at_a_defined_value():
    """
    Given a component that supplies inactive_value,
    When the parameter is built,
    Then the inactive element sits at that value instead of its resolved
      initval.

    A linear-law band's u2 is exactly 0, not "whatever the quadratic default
    was": where the other parameterization DEFINES the value, the pin says so
    and cannot drift with an unrelated default.
    """
    p = _param(mask=[True, False], inactive_value=0.0)

    with pm.Model():
        val = p.build_pymc()

    assert float(np.atleast_1d(val.eval())[1]) == 0.0


def test_an_inactive_element_with_no_start_value_does_not_raise():
    """
    Given an inactive element that no source ever gave a value,
    When the parameter is built,
    Then it builds.

    The pin-must-say-what-it-pins-to error exists because a pinned value is a
    modeling statement nobody made -- but an inactive element is read by
    nothing and reported nowhere, so there is no statement to get wrong and no
    fix to advise.
    """
    p = _param(mask=[True, False], initval=np.array([0.5, np.nan]))

    with pm.Model():
        p.build_pymc()  # must not raise

    assert p.is_active.tolist() == [True, False]


def test_a_user_constraint_on_an_inactive_element_warns_that_it_is_dropped(
    caplog,
):
    """
    Given a params-file prior on an element that is not a parameter of its
      instance's parameterization,
    When the parameter is built,
    Then one warning names the element and the dropped fields.

    This is the ONE lossy case in a parameterization switch: a prior on an
    element that flipped to DERIVED still applies, and a start value still
    feeds the relaxation engine, but an element that is no longer a parameter
    has nothing to carry the constraint.  A warning, not an error, because the
    point of per-element roles is that one params file survives a toggle.
    """
    p = _param(
        mask=[True, False],
        sigma=np.array([np.nan, 0.2]),
        mu=np.array([np.nan, 0.4]),
        user_params={"comp.1.p": {"mu": 0.4, "sigma": 0.2}},
        source_file="my.params.yaml",
    )

    with pm.Model():
        with caplog.at_level("WARNING", logger="exozippy"):
            p.build_pymc()

    dropped = [
        r.getMessage() for r in caplog.records if "DROPPED" in r.getMessage()
    ]
    assert len(dropped) == 1
    assert "mu/sigma" in dropped[0] and "my.params.yaml" in dropped[0]


def test_a_default_prior_on_an_inactive_element_warns_about_nothing(caplog):
    """
    Given an inactive element whose sigma and bounds come only from
      defaults.yaml,
    When the parameter is built,
    Then nothing warns.

    Almost every parameter has bounds from its defaults.yaml, so keying the
    warning on the resolved values would fire on every inactive element and
    teach users to ignore warnings.  It keys on what the USER wrote.
    """
    p = _param(mask=[True, False], sigma=np.array([np.nan, 0.2]))

    with pm.Model():
        with caplog.at_level("WARNING", logger="exozippy"):
            p.build_pymc()

    assert not [r for r in caplog.records if "DROPPED" in r.getMessage()]


def test_masking_an_element_out_that_an_expression_supplies_raises():
    """
    Given a component that both masks an element out and derives it,
    When the parameter is built,
    Then it raises naming the element.

    The two roles contradict: an element either is not a parameter of its
    instance or it has a value.  Silently preferring one would make a
    component's own mode table unreadable.
    """
    p = _param(
        mask=[True, False],
        element_expressions=[
            ElementExpression(
                mask=[False, True],
                expr=lambda: pt.as_tensor_variable(np.array([0.0, 0.5])),
            )
        ],
    )

    with pm.Model():
        with pytest.raises(ValueError, match="inactive AND claimed"):
            p.build_pymc()


def test_a_mixed_vector_keeps_a_finite_logp_and_gradient():
    """
    Given a model whose only parameter is a mixed vector whose derived element
      is a nonlinear function of the sampled one,
    When the logp and its gradient are evaluated at the start,
    Then both are finite.

    The assembly uses set_subtensor rather than a where over the two value
    vectors precisely to keep this true: where's VJP multiplies the unselected
    branch by zero, and 0*NaN poisons the gradient of the whole vector on every
    backend.
    """
    with pm.Model() as model:
        driver = _param(label="comp.driver")
        driver.build_pymc()
        dep = driver.value

        p = _param(
            label="comp.mixed",
            element_expressions=[
                ElementExpression(
                    mask=[False, True], expr=lambda: pt.sqrt(dep) * 0.5
                )
            ],
        )
        p.build_pymc()

    point = model.initial_point()
    logp = model.compile_logp()(point)
    dlogp = model.compile_dlogp()(point)

    assert np.isfinite(logp)
    assert np.all(np.isfinite(dlogp))


def test_a_mixed_vector_samples_under_numpyro():
    """
    Given the same mixed model,
    When a short chain is sampled with nuts_sampler="numpyro",
    Then every draw is finite.

    The house rule: a finite C-backend gradient is NOT proof the JAX path is
    sound.  The where-trap has bitten this codebase repeatedly -- a discarded
    branch's NaN reaching the selected branch's gradient through JAX's
    jnp.where -- and the mixed assembly is exactly the kind of code that
    invites it, so the JAX path is exercised rather than argued about.
    """
    with pm.Model() as model:
        driver = _param(label="comp.driver")
        driver.build_pymc()
        dep = driver.value

        p = _param(
            label="comp.mixed",
            element_expressions=[
                ElementExpression(
                    mask=[False, True], expr=lambda: pt.sqrt(dep) * 0.5
                )
            ],
        )
        p.build_pymc()

    with model:
        idata = pm.sample(
            draws=25,
            tune=25,
            chains=1,
            cores=1,
            nuts_sampler="numpyro",
            progressbar=False,
            random_seed=0,
        )

    for var in idata.posterior.data_vars:
        assert np.all(np.isfinite(idata.posterior[var].values)), (
            f"non-finite draws for {var}"
        )


# ---------------------------------------------------------------------------
# 3. Dependency slicing: the unused instance's numbers stay out
# ---------------------------------------------------------------------------


class _SliceComp(Component):
    """A component with two instances that choose different physics."""

    aligned_context_deps = frozenset({"ctx_aligned"})

    @property
    def prefix(self):
        return "slicecomp"

    def register_parameters(self, system):
        self.manifest = {}

    def build_likelihood(self, model, system):
        pass


def _slice_comp(n=2):
    comp = _SliceComp.__new__(_SliceComp)
    comp.config = [{"name": f"i{i}"} for i in range(n)]
    comp.n_elements = n
    comp.names = [f"i{i}" for i in range(n)]
    comp.manifest = {}
    return comp


def test_slicing_keeps_the_unused_elements_out_of_the_expression():
    """
    Given an expression that supplies element 1 only, and a dependency whose
      element 0 holds a value the physics cannot evaluate (a negative under a
      square root),
    When the parameter is built and its gradient is taken,
    Then the value of element 1 is right and the gradient is finite.

    An unused element's value is a bookkeeping pin that the OTHER
    parameterization's physics makes no promise about, so evaluating there can
    legitimately be NaN -- and a NaN in a discarded slot still reaches the
    input's gradient as 0*NaN.  Slicing the dependency is what makes the
    mixed vector safe rather than merely usually-safe.
    """
    with pm.Model() as model:
        driver = _param(
            label="comp.driver",
            initval=np.array([-1.0, 0.25]),
            lower=np.array([-2.0, 0.0]),
            upper=np.array([2.0, 1.0]),
        )
        driver.build_pymc()
        dep = driver.value
        idx = np.array([1], dtype="int32")

        p = _param(
            label="comp.mixed",
            element_expressions=[
                ElementExpression(
                    mask=[False, True],
                    expr=lambda: pt.sqrt(dep[idx]),
                    sliced=True,
                )
            ],
        )
        val = p.build_pymc()

    point = model.initial_point()
    assert float(np.atleast_1d(val.eval(point))[1]) == pytest.approx(0.5)
    assert np.all(np.isfinite(model.compile_dlogp()(point)))


def test_a_dependency_that_cannot_be_proven_aligned_refuses_to_be_sliced():
    """
    Given an expression supplying a subset of one component's elements whose
      dependency is a BARE cross-component vector,
    When the parameter is wired,
    Then it raises, naming the dep and the two ways to fix it.

    A bare cross-component vector is indexed by the OTHER component's
    elements; slicing it by this component's mask pairs the wrong instances,
    and where the lengths happen to match nothing complains.  The same file
    already refuses the analogous silent fallback for a dep that names a map
    the component does not have.
    """
    comp = _slice_comp()
    entry = interpret_manifest_entry({"expr_key": {"only_one": [True, False]}})
    sel = entry.expression_configs(
        {"only_one": {"func_name": "calc_density", "deps": ["star.mass"]}},
        n_elements=2,
        where="slicecomp.p",
    )[0]

    star = _param(label="star.mass")
    with pm.Model():
        star.build_pymc()
        system = type("S", (), {"star": type("C", (), {"mass": star})()})()
        with pytest.raises(ValueError, match="cannot be PROVEN"):
            comp._element_expression(
                None, system, {}, entry, sel, "slicecomp.p"
            )


def test_a_declared_aligned_context_dep_may_be_sliced():
    """
    Given a context dep the component declares element-aligned,
    When an expression supplying a subset of the elements is wired,
    Then it slices without complaint.

    A context node is an arbitrary tensor the component built, so nothing
    outside the component can prove its alignment -- which makes the
    declaration a promise, and the only honest way to allow it.
    """
    comp = _slice_comp()
    entry = interpret_manifest_entry({"expr_key": {"ctx": [True, False]}})
    sel = entry.expression_configs(
        {"ctx": {"func_name": "calc_density", "deps": ["ctx_aligned"]}},
        n_elements=2,
        where="slicecomp.p",
    )[0]
    ctx = {"ctx_aligned": pt.as_tensor_variable(np.array([1.0, 2.0]))}

    built = comp._element_expression(
        None, None, ctx, entry, sel, "slicecomp.p"
    )

    assert built.sliced is True
    assert built.mask.tolist() == [True, False]


def test_a_non_elementwise_expression_is_caught_at_the_start_point():
    """
    Given an expression that sums over the element axis (so slicing its
      dependencies changes its answer, not just its shape),
    When the start-point verification runs,
    Then it raises naming the parameter and the physics function.

    The alignment proof is static and cannot see this: "elementwise" is a
    property of the FUNCTION.  So both graphs are kept and compared at the
    start point, where real values exist -- dummy inputs could agree by
    accident, and evaluating a random variable would draw from its prior
    instead of reading the start.
    """
    from exozippy.system import System

    system = System.__new__(System)
    system._element_slice_checks = []

    with pm.Model() as model:
        driver = _param(label="comp.driver", initval=np.array([0.2, 0.8]))
        driver.build_pymc()
        dep = driver.value
        idx = np.array([1], dtype="int32")

        system.register_element_slice_check(
            "comp.p",
            "calc_total",
            idx,
            lambda: pt.sum(dep[idx]) * pt.ones(1),
            lambda: pt.sum(dep) * pt.ones(2),
        )

    system.get_raw_start = lambda m: m.initial_point()
    with pytest.raises(ValueError, match="not elementwise"):
        system.verify_element_slices(model)


def test_the_start_point_check_accepts_an_elementwise_expression():
    """
    Given a genuinely elementwise expression,
    When the same verification runs,
    Then it passes and reports the number of checks it made.
    """
    from exozippy.system import System

    system = System.__new__(System)
    system._element_slice_checks = []

    with pm.Model() as model:
        driver = _param(label="comp.driver", initval=np.array([0.2, 0.8]))
        driver.build_pymc()
        dep = driver.value
        idx = np.array([1], dtype="int32")

        system.register_element_slice_check(
            "comp.p",
            "calc_double",
            idx,
            lambda: dep[idx] * 2.0,
            lambda: dep * 2.0,
        )

    system.get_raw_start = lambda m: m.initial_point()
    assert system.verify_element_slices(model) == 1


# ---------------------------------------------------------------------------
# 4. The whole-vector paths are untouched
# ---------------------------------------------------------------------------


def test_an_all_elements_expression_takes_the_whole_vector_path():
    """
    Given a single ElementExpression covering EVERY element,
    When the parameter is built,
    Then it is treated as the historical whole-vector derived case: no
      set_subtensor, no raw variable, every element derived.

    Bit-identical graphs for the existing components are the acceptance bar
    for this work, so the uniform case must not merely agree numerically -- it
    must take the same code path.
    """
    node = pt.as_tensor_variable(np.array([0.25, 0.75]))
    p = _param(
        element_expressions=[ElementExpression(mask=True, expr=lambda: node)]
    )

    with pm.Model() as model:
        val = p.build_pymc()

    assert p.is_derived.tolist() == [True, True]
    assert not np.any(p.is_sampled)
    assert _raw_names(model) == []
    # The value node IS the expression, not a transform vector patched by
    # set_subtensor: no subtensor-assignment op appears anywhere in its graph.
    ops = [
        type(a.owner.op).__name__
        for a in pytensor.graph.traversal.ancestors([val])
        if a.owner is not None
    ]
    assert not [name for name in ops if "SetSubtensor" in name], ops


def test_a_plain_vector_still_builds_exactly_one_raw_variable():
    """
    Given a Parameter using none of the per-element vocabulary,
    When it is built,
    Then every element is sampled and active, as before.

    The regression guard for the four role masks defaulting the wrong way.
    """
    p = _param()

    with pm.Model() as model:
        p.build_pymc()

    assert p.is_sampled.tolist() == [True, True]
    assert p.is_derived.tolist() == [False, False]
    assert p.is_active.tolist() == [True, True]
    assert _raw_names(model) == ["comp.p_raw"]


def test_expression_and_element_expressions_together_are_refused():
    """
    Given both a whole-vector expression and per-element expressions,
    When the parameter is built,
    Then it raises.

    Two channels for one element's value is exactly the ambiguity this
    vocabulary exists to remove.
    """
    node = pt.as_tensor_variable(np.array([0.25, 0.75]))
    p = _param(
        expression=lambda: node,
        element_expressions=[
            ElementExpression(mask=[True, False], expr=lambda: node)
        ],
    )

    with pm.Model():
        with pytest.raises(ValueError, match="both a whole-vector"):
            p.build_pymc()


# ---------------------------------------------------------------------------
# 5. Reporting: an inactive element appears nowhere
# ---------------------------------------------------------------------------


def test_the_prior_column_does_not_depend_on_how_the_vector_was_declared():
    """
    Given a mixed vector, and the two uniform parameters its elements are the
      equivalents of (one plain sampled, one wholly derived),
    When the Prior column is rendered,
    Then the mixed vector's sampled element reads like the sampled parameter
      and its derived element reads like the derived one.

    `_own_prior_str` asked `self.expression is not None`, a WHOLE-VECTOR
    question, for the two branches that decide "a derived parameter with no
    constraint of its own has no prior to display".  It now asks per element,
    so the column describes the parameterization each instance actually chose
    rather than the shape of the declaration.
    """
    fields = dict(
        lower=np.array([-np.inf, -np.inf]),
        upper=np.array([np.inf, np.inf]),
        sigma=np.array([0.25, 0.25]),
        mu=np.array([0.5, 0.5]),
    )
    node = pt.as_tensor_variable(np.array([0.0, 0.5]))
    mixed = _param(
        label="comp.mixed",
        element_expressions=[
            ElementExpression(mask=[False, True], expr=lambda: node)
        ],
        **fields,
    )
    sampled = _param(label="comp.sampled", **fields)
    derived = _param(label="comp.derived", expression=lambda: node, **fields)

    with pm.Model():
        mixed.build_pymc()
        sampled.build_pymc()
        derived.build_pymc()

    assert mixed.get_prior_str(0, latex=False) == sampled.get_prior_str(
        0, latex=False
    )
    assert mixed.get_prior_str(1, latex=False) == derived.get_prior_str(
        1, latex=False
    )


def test_a_pinned_vector_with_a_broadcast_initval_defines_a_macro_per_element():
    """
    Given a fully pinned vector whose initval is a broadcast SCALAR,
    When the LaTeX variable definitions are emitted,
    Then one macro is defined per element.

    Review 1.2.3: the fixed path sized its loop on len(atleast_1d(initval)),
    so a scalar initval emitted ONE unsuffixed macro while the table body
    cites a suffixed one per element -- an "Undefined control sequence" at
    compile time, after a fit that may have run for hours.
    """
    p = _param(initval=0.5, sigma=np.zeros(2))

    defs = p.to_latex_def()

    assert defs.count(r"\providecommand") == 2
    assert "zero" in defs and "one" in defs


def test_an_inactive_element_gets_no_table_or_csv_row(tmp_path):
    """
    Given a two-instance component whose second element is inactive,
    When the LaTeX table and the results CSV are built,
    Then only the active instance has rows, and no sub-head is emitted for the
      instance that has none.

    An inactive element is held at a bookkeeping value; a row for it would
    report a fitted quantity that was never fitted.
    """
    from exozippy.outputs.latex import build_csv_output, build_latex_output

    p = _param(label="star.eep", mask=[True, False])
    with pm.Model():
        p.build_pymc()

    comp = type("C", (), {"label": "Star"})()
    comp.eep = p
    system = type(
        "S",
        (),
        {"get_all_components": lambda self: [comp], "name": "elemtest"},
    )()

    csv_path = tmp_path / "out.csv"
    build_csv_output(system, str(csv_path))
    body = csv_path.read_text()

    table_path = tmp_path / "out_table.tex"
    build_latex_output(
        system,
        var_filename=str(tmp_path / "out_definitions.tex"),
        table_filename=str(table_path),
    )
    table = table_path.read_text()

    assert "star.i0.eep" in body or "star.eep" in body
    assert "i1" not in body
    assert "i1" not in table


def test_element_roles_are_stamped_only_for_non_uniform_vectors():
    """
    Given one mixed vector and one ordinary vector,
    When the trace metadata records element roles,
    Then only the mixed vector appears, with its per-element masks.

    mkparam reads this because it has no System, and a raw variable's length
    says how MANY elements are sampled, never which.  Uniform vectors are
    omitted: "every element sampled" is what a missing entry has always meant,
    which is also what makes older traces safe to read.
    """
    from exozippy.trace_meta import element_roles

    node = pt.as_tensor_variable(np.array([0.0, 0.5]))
    mixed = _param(
        label="planet.mass",
        element_expressions=[
            ElementExpression(mask=[False, True], expr=lambda: node)
        ],
    )
    plain = _param(label="planet.period")
    with pm.Model():
        mixed.build_pymc()
        plain.build_pymc()

    system = type(
        "S", (), {"get_all_parameters": lambda self: [mixed, plain]}
    )()

    roles = element_roles(system)

    assert set(roles) == {"planet.mass"}
    assert roles["planet.mass"]["sampled"] == [True, False]
    assert roles["planet.mass"]["derived"] == [False, True]


def test_mkparam_writes_no_start_value_for_a_derived_element(tmp_path):
    """
    Given a trace whose star.mass vector is sampled on element 0 and derived on
      element 1 (recorded in the trace's element-role metadata),
    When mkparam writes the next params file,
    Then only the sampled element gets an entry.

    A start value for a derived element is a redundant constraint on the next
    fit -- exactly what mkparam's "only physically sampled variables" filter
    exists to prevent, which it could not enforce per element without this
    metadata.
    """
    import json

    from exozippy.mkparam import write_param_file
    from exozippy.trace_meta import ROLES_ATTR

    az = pytest.importorskip("arviz")
    rng = np.random.default_rng(0)
    nchain, ndraw = 2, 40
    post = {
        "star.mass": 1.0 + 0.1 * rng.standard_normal((nchain, ndraw, 2)),
        "star.mass_raw": rng.standard_normal((nchain, ndraw, 1)),
    }
    lp = -0.5 * rng.standard_normal((nchain, ndraw)) ** 2
    idata = az.from_dict({"posterior": post, "sample_stats": {"lp": lp}})
    idata.attrs[ROLES_ATTR] = json.dumps(
        {
            "star.mass": {
                "sampled": [True, False],
                "derived": [False, True],
                "active": [True, True],
            }
        }
    )
    trace = tmp_path / "run_trace.nc"
    idata.to_netcdf(str(trace))

    out = tmp_path / "out.params.yaml"
    write_param_file(
        {"prefix": "run", "star": [{"name": "A"}, {"name": "B"}]},
        base_dir=tmp_path,
        trace_path=trace,
        output_path=out,
    )
    params = yaml.safe_load(out.read_text())

    assert "star.A.mass" in params
    assert "star.B.mass" not in params


def test_export_solution_reports_derived_and_active_per_element():
    """
    Given per-element derived and activity masks,
    When export_solution labels an index path and a broadcast path,
    Then each element gets its own answer and the broadcast path is True only
      when every element agrees.

    Consumers act on these labels -- solve_api._bounds_diagnostics SKIPS a
    parameter reported derived -- so a whole-vector answer for a mixed vector
    means a bounds check that silently does not run on the sampled elements.
    """
    from exozippy.config import _element_flag

    derived = {("planet", "mass"): np.array([False, True])}

    assert (
        _element_flag(
            derived, ("planet", "mass"), 0, fallback=True, absent=False
        )
        is False
    )
    assert (
        _element_flag(
            derived, ("planet", "mass"), 1, fallback=True, absent=False
        )
        is True
    )
    assert (
        _element_flag(
            derived, ("planet", "mass"), None, fallback=True, absent=False
        )
        is False
    )
    # A parameter the table does not mention takes the `absent` answer, and a
    # missing table falls back to the caller's own guess.
    assert (
        _element_flag(
            derived, ("star", "teff"), 0, fallback=True, absent=False
        )
        is False
    )
    assert (
        _element_flag(None, ("star", "teff"), 0, fallback=True, absent=False)
        is True
    )
    # The historical whole-parameter spelling (a set) still answers.
    assert (
        _element_flag(
            {("planet", "mass")},
            ("planet", "mass"),
            0,
            fallback=False,
            absent=False,
        )
        is True
    )
