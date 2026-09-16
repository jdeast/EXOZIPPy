"""Tests for the log_s reparameterization (notes/multimode_implementation.txt P2).

s (projected binary separation) is now derived from a sampled log10(s), so the
close/wide degeneracy is an exact reflection log_s -> -log_s.  These tests
assert that log_s is the sampled coordinate, s = 10**log_s is derived, and that
user-supplied lens.s initval / bounds translate onto log_s.

s / log_s are PER COMPANION and still live on the `lens` component, but that
component is now one entry per lens BODY (8.6.17): element 0 is the masked
primary and companion j is ELEMENT j+1, so the binary's separation is
lens.1.log_s (equivalently lens.b.log_s, the companion body's name).
"""

import numpy as np
import pytensor
import pytest

from exozippy.system import System


def _binary_config():
    """Minimal single-source binary lens (star + planet companion)."""
    return {
        "star": [{"name": "Lens"}, {"name": "Source"}],
        "planet": [{"name": "b"}],
        "mulensevent": [{"finite_source": False}],
        # One entry per lens body: the primary first, then the companion.
        "lens": [{"body": "star.Lens"}, {"body": "planet.b"}],
        "source": [{"body": "star.Source"}],
    }


def _binary_params(s_entry):
    p = {
        # Trajectory parameters are per SOURCE body; t_E is event level.
        "source.Source.t_0": {"initval": 2460025.0, "init_scale": 0.1},
        "source.Source.u_0": {"initval": 0.1, "init_scale": 0.01},
        "mulensevent.t_E": {"initval": 30.0, "init_scale": 1.0},
        # ... the geometry is per companion, i.e. lens element 1.
        "lens.b.s": s_entry,
        "lens.b.alpha": {"initval": 60.0, "init_scale": 0.9},
        "lens.b.q": {"initval": 1e-3, "init_scale": 1e-4},
        "star.radius": {"sigma": 0.0},
        "star.teff": {"sigma": 0.0},
        "star.feh": {"sigma": 0.0},
    }
    for nm in ("Lens", "Source"):
        p[f"star.{nm}.ra"] = {"initval": 264.0, "sigma": 0}
        p[f"star.{nm}.dec"] = {"initval": -27.0, "sigma": 0}
    return p


def _build(s_entry):
    system = System(_binary_config(), user_params=_binary_params(s_entry))
    system.prepare()
    model = system.build_model()
    return system, model


# The default (s=0.98, init_scale=0.01) build is shared by the two tests that
# only inspect it, so the System is built + compiled once instead of twice.
_DEFAULT_S_ENTRY = {"initval": 0.98, "init_scale": 0.01}


@pytest.fixture(scope="module")
def default_s_system():
    return _build(_DEFAULT_S_ENTRY)


def test_log_s_is_sampled_and_s_is_derived(default_s_system):
    """Given a binary lens, when the model is built, then log_s is a sampled
    free RV, s is not sampled, and s == 10**log_s at the start point."""
    system, model = default_s_system

    free_names = [v.name for v in model.free_RVs]
    assert "lens.log_s_raw" in free_names
    assert "lens.s_raw" not in free_names

    # log_s is sampled (no expr_key) and s is a derived expression node
    # (expr_key in manifest).  Both are masked OFF on element 0, the
    # primary, which has no separation of its own to speak of.
    log_s_entry = system.lens.manifest["log_s"]
    assert "expr_key" not in log_s_entry
    np.testing.assert_array_equal(log_s_entry["mask"], [False, True])

    s_entry = system.lens.manifest["s"]
    assert set(s_entry["expr_key"]) == {"default"}
    np.testing.assert_array_equal(
        s_entry["expr_key"]["default"], [False, True]
    )
    np.testing.assert_array_equal(s_entry["mask"], [False, True])

    with model:
        f = pytensor.function(
            model.free_RVs,
            [system.lens.s.value, system.lens.log_s.value],
            on_unused_input="ignore",
        )
        ip = model.initial_point()
        zeros = [
            np.zeros_like(ip[v.name]).astype(float) for v in model.free_RVs
        ]
        s_val, log_s_val = [np.atleast_1d(x) for x in f(*zeros)]

    # Both elements: the primary's inactive pins (s = 1, log_s = 0) are
    # deliberately consistent under the same relation, so no diagnostic that
    # prints the primary row can show a contradictory pair.
    assert s_val.size == 2 and log_s_val.size == 2
    np.testing.assert_allclose(s_val, 10.0**log_s_val, rtol=1e-6)


def test_user_s_initval_seeds_log_s_start(default_s_system):
    """Given a params entry setting lens.s initval, when the relaxation
    engine runs, then log_s is seeded with log10(s).  The entry's obsolete
    init_scale is stripped (whitening scales are measured at startup), so
    log_s keeps a finite preliminary scale from defaults.

    Read at ELEMENT 1, the companion: element 0 is the masked primary, whose
    s is not a parameter at all (its initval is the engine's bookkeeping
    NaN; the build pins it at the manifest's inactive_value)."""
    s0 = _DEFAULT_S_ENTRY["initval"]
    system, _ = default_s_system

    np.testing.assert_allclose(
        np.atleast_1d(system.lens.s.initval)[1], s0, rtol=1e-9
    )
    np.testing.assert_allclose(
        np.atleast_1d(system.lens.log_s.initval)[1], np.log10(s0), atol=1e-6
    )
    log_s_scale = np.atleast_1d(system.lens.log_s.init_scale)
    assert np.all(np.isfinite(log_s_scale)) and np.all(log_s_scale > 0)


def test_user_s_bounds_translate_to_log_s():
    """Given a params entry setting lens.s lower/upper, when the Lens component
    registers, then the bounds are moved onto log_s as log10(bound) and removed
    from the s entry."""
    system, _ = _build(
        {"initval": 1.0, "init_scale": 0.01, "lower": 0.5, "upper": 5.0}
    )

    np.testing.assert_allclose(
        np.atleast_1d(system.lens.log_s.lower)[1], np.log10(0.5), atol=1e-6
    )
    np.testing.assert_allclose(
        np.atleast_1d(system.lens.log_s.upper)[1], np.log10(5.0), atol=1e-6
    )

    # The user wrote 'lens.b.s'; standardization folded it onto the
    # companion's ELEMENT index, which is 1 (element 0 is the primary).
    up_s = system.config_manager.user_params.get("lens.1.s")
    assert "lower" not in up_s
    assert "upper" not in up_s


def test_nonpositive_s_bound_raises():
    """Given a non-positive lens.s bound (log10 undefined), when the Lens
    component registers, then a clear ValueError is raised."""
    with pytest.raises(ValueError, match="s > 0"):
        _build({"initval": 1.0, "init_scale": 0.01, "lower": -0.5})
