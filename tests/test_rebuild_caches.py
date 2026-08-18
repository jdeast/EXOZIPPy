"""Per-build caches must not outlive their build (review 1.5.2).

Anything a component caches that IS a pytensor node, or is compiled against
one, belongs to the model that built it.  ``Transit._dilution_node`` and
``Instrument``'s lazily compiled outlier-probability evaluators were kept on
the component, which persists on the System, so a later build could be
handed the earlier model's graph.

Re-verified 2026-08-18: the item's premise -- "a second
``system.build_model()`` is documented-supported (GUI)" -- does NOT hold.
The GUI builds a FRESH System every time (``gui/tune.py``), and a second
``build_model()`` on one System fails well before the dilution cache
matters, because ``Component.add_parameter``'s already-built guard returns
the FIRST model's node for every parameter.  The caches are still wrong to
keep, and are cleared; the second-build path itself is a System-level
problem and is pinned below as an xfail so a fix cannot land silently.
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


@pytest.mark.xfail(
    strict=True,
    reason=(
        "A second build_model() on one System is not supported: "
        "Component.add_parameter's already-built guard hands the second "
        "model the first model's nodes, so its logp compile raises "
        "'Random variables detected in the logp graph'. Clearing the "
        "per-build caches (1.5.2) removes one blocker, not this one. "
        "Remove this xfail when the System-level rebuild is fixed."
    ),
)
def test_a_second_build_on_one_system_scores_the_same(lc_path):
    """
    Given a prepared System already built once,
    When build_model() is called again on the same System,
    Then the second model should compile and score the same start logp.
    """
    system = _prepared(lc_path)

    first = system.build_model()
    lp1 = float(first.compile_logp()(system.get_raw_start(first)))
    second = system.build_model()
    lp2 = float(second.compile_logp()(system.get_raw_start(second)))

    assert lp2 == pytest.approx(lp1, rel=0, abs=0)
