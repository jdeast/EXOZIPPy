"""A build must own every node it scores (reviews 1.5.2, 3.14.12).

Anything a component caches that IS a pytensor node, or is compiled against
one, belongs to the model that built it.  ``Transit._dilution_node`` and
``Instrument``'s lazily compiled outlier-probability evaluators were kept on
the component, which persists on the System, so a later build could be
handed the earlier model's graph (1.5.2).

Clearing those caches was necessary and not sufficient.  A second
``build_model()`` on one System failed far earlier, in
``Component.add_parameter``: its already-built guard asked only "is there a
Parameter here", and every parameter still held the FIRST model's node, so
the second model was assembled out of the first model's random variables and
its logp compile raised "Random variables detected in the logp graph".  That
was pinned here as a STRICT xfail until 3.14.12 narrowed the guard to "built
for THIS model" and gave the four hand-written copies of the predicate the
same treatment (``Orbit._chord_context``'s planet geometry,
``Orbit.add_parameter``'s group masses, ``SED._ensure_star_nodes``).  The
xfail is gone; the tests below are its replacement, and the graph-walk one
covers the silent half the logp compile cannot see.

The GUI does not exercise any of this -- it builds a FRESH System per solve
(``gui/tune.py``) -- so the second-build path is for scripts and for
``exozippy-modes``-style re-reporting.
"""

import numpy as np
import pytest

from exozippy.system import System

_LC = dict(baseline=1.0, depth=0.01, P=3.2, tc=2459100.0, err=4.0e-4)


def _write_lc(path, n=180):
    t = np.linspace(_LC["tc"] - 0.2, _LC["tc"] + 0.2, n)
    in_transit = np.abs(t - _LC["tc"]) < 0.04
    flux = _LC["baseline"] - _LC["depth"] * in_transit
    np.savetxt(path, np.column_stack([t, flux, np.full_like(t, _LC["err"])]))


@pytest.fixture(scope="module")
def lc_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("rebuild") / "rebuild.TESS.dat"
    _write_lc(path)
    return str(path)


def _prepared(lc_path):
    """A freshly prepared single-transit System (stages 1-4 only)."""
    config = {
        "run": {"name": "rebuild"},
        "star": [{"name": "A", "mist": False}],
        "planet": [{"name": "b"}],
        "orbit": [{"name": "b", "primary": ["A"], "companion": ["b"]}],
        "band": [{"name": "TESS", "filter": "TESS"}],
        "transit": [{"name": "TESS", "file": lc_path, "band": "TESS"}],
    }
    user_params = {
        "star.A.mass": {"initval": 1.0, "sigma": 0.05},
        "star.A.radius": {"initval": 1.0, "sigma": 0.1},
        "star.A.teff": {"initval": 5800, "sigma": 100},
        "star.A.feh": {"initval": 0.0, "sigma": 0.1},
        "orbit.b.period": {"initval": _LC["P"]},
        "orbit.b.tc": {"initval": _LC["tc"]},
    }
    system = System(config, user_params=user_params)
    system.prepare()
    return system


def test_build_likelihood_drops_a_stale_dilution_node(lc_path):
    """
    Given a component carrying a dilution node from an earlier model,
    When build_likelihood runs,
    Then the stale node is gone before anything can reference it.

    A sentinel stands in for the earlier model's Deterministic: what
    matters is that the cache is cleared unconditionally at the top of the
    build, not what happened to be in it.
    """
    system = _prepared(lc_path)
    sentinel = object()
    system.transit._dilution_node = sentinel

    model = system.build_model()

    assert system.transit._dilution_node is not sentinel
    # This topology has one star, so no dilution is built at all.
    assert system.star.n_elements == 1
    assert system.transit._dilution_node is None
    assert model is not None


def test_a_second_build_on_one_system_scores_the_same(lc_path):
    """
    Given a prepared System already built once,
    When build_model() is called again on the same System,
    Then the second model compiles and scores the same start logp.

    This was a STRICT xfail until review 3.14.12: the already-built guard
    tested only "is there a Parameter here", so the second build was handed
    the first model's nodes and its logp compile raised "Random variables
    detected in the logp graph".  Bit-identical (``abs=0``) is the assertion
    that matters -- a rebuild that merely compiles but scores differently is
    the silent half of the same bug.
    """
    system = _prepared(lc_path)

    first = system.build_model()
    lp1 = float(first.compile_logp()(system.get_raw_start(first)))
    second = system.build_model()
    lp2 = float(second.compile_logp()(system.get_raw_start(second)))

    assert second is not first
    assert lp2 == pytest.approx(lp1, rel=0, abs=0)


def test_a_rebuilt_model_owns_every_node_it_scores(lc_path):
    """
    Given a System built twice,
    When the second model's graph is walked,
    Then no node in it descends from a random variable of the FIRST model.

    The logp compile above is the symptom; this is the property.  It catches
    the silent half -- a leaked node that happens to sit outside the logp
    graph (a Deterministic the tables read, a plot node) and so never raises.
    """
    try:
        from pytensor.graph.traversal import ancestors
    except ImportError:  # pragma: no cover - older pytensor
        from pytensor.graph.basic import ancestors

    system = _prepared(lc_path)
    first = system.build_model()
    first_rvs = set(first.free_RVs)

    second = system.build_model()
    outputs = (
        list(second.free_RVs)
        + list(second.deterministics)
        + list(second.potentials)
        + list(second.observed_RVs)
    )
    leaked = sorted({v.name for v in set(ancestors(outputs)) & first_rvs})

    assert leaked == []


def test_the_build_guard_still_builds_each_parameter_once(lc_path):
    """
    Given one build_model(),
    When a parameter is reached both by the build order and as another
    parameter's dependency,
    Then it is materialized exactly once.

    The guard 3.14.12 narrowed is what stops a recursive dependency from
    building a second copy of the same node.  Narrowing it to "built for
    THIS model" must not weaken that: two Deterministics with one label
    would make PyMC raise on the duplicate name.
    """
    system = _prepared(lc_path)
    model = system.build_model()

    names = [v.name for v in model.deterministics] + [
        v.name for v in model.free_RVs
    ]
    assert len(names) == len(set(names))


def test_a_parameter_with_no_build_stamp_counts_as_current():
    """
    Given a component-like object carrying a Parameter that never went
    through add_parameter (a test double, or one set by hand),
    When the build-time predicate is asked,
    Then it answers "current" -- absent provenance is not staleness.
    """
    import types

    from exozippy.components.component import Component
    from exozippy.components.parameter import Parameter

    double = types.SimpleNamespace(
        mass=Parameter(label="x.mass", initval=1.0, unit="", internal_unit="")
    )

    assert Component._parameter_is_current(double, "mass", object())
    assert not Component._parameter_is_current(double, "absent", object())
