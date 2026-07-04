"""Tests for code-review fixes in the MulensModel Op layer and Lens/Instrument
config validation."""
import logging
from unittest.mock import patch

import numpy as np
import pytest
import MulensModel as mm

from exozippy.components.mulensing.op import (
    _get_sat_coord,
    _build_pspl_model,
    _build_binary_model,
)
from exozippy.components.mulensing.lens import Lens
from exozippy.components.mulensing.mulensinstrument import MulensInstrument
from exozippy.run import KNOWN_SAMPLER_KEYS
from conftest import _DummyConfigManager, _DummyComponent, _DummySystem


COORDS = "270.0d -28.0d"


def test_sat_coord_cache_distinguishes_same_length_arrays():
    """
    Given two observer-position arrays with the same number of epochs but
      different positions (e.g. Earth and a satellite over one model grid),
    When both are passed through the same coordinate cache,
    Then each gets its own SkyCoord (the old length-keyed cache silently
      returned the first observer's coordinates for the second).
    """
    # Arrange
    cache = {}
    times_np = np.linspace(2450000.0, 2450004.0, 5)
    earth = np.zeros((5, 3))
    satellite = np.ones((5, 3))

    # Mock Earth ephemeris so the test doesn't require network / JPL file access.
    with patch("exozippy.components.mulensing.op._earth_xyz_at",
               return_value=np.zeros((5, 3))):
        # Act
        coord_earth = _get_sat_coord(earth, times_np, cache)
        coord_sat = _get_sat_coord(satellite, times_np, cache)

        # Assert
        assert not np.allclose(coord_earth.cartesian.xyz.value,
                               coord_sat.cartesian.xyz.value)
        assert _get_sat_coord(earth, times_np, cache) is coord_earth
        assert _get_sat_coord(satellite, times_np, cache) is coord_sat


def test_pspl_model_floors_nonpositive_rho():
    """
    Given a finite-source PSPL parameter vector whose rho is 0
      (e.g. a transient sampler excursion or a bad initval),
    When the MulensModel is constructed,
    Then rho is floored to a small positive value instead of building an
      unphysical model.
    """
    # Arrange: [t_0, u_0, t_E, pi_E_N, pi_E_E, rho]
    p = np.array([2450000.0, 0.1, 20.0, 0.0, 0.0, 0.0])

    # Act
    model = _build_pspl_model(p, COORDS, "auto_vbbl", use_rho=True)

    # Assert
    assert float(model.parameters.rho) > 0


def test_binary_method_selection_follows_finite_source_flag():
    """
    Given a binary-lens parameter vector,
    When the model is built with and without finite_source (use_rho),
    Then the finite-source method (VBM) is selected iff use_rho is True,
      independent of the runtime rho value.
    """
    # Arrange: [t_0, u_0, t_E, pi_E_N, pi_E_E, (rho), s, q, alpha]
    p_fs = np.array([2450000.0, 0.1, 20.0, 0.0, 0.0, 1e-3, 1.2, 0.01, 30.0])
    p_ps = np.array([2450000.0, 0.1, 20.0, 0.0, 0.0, 1.2, 0.01, 30.0])

    # Act / Assert
    with patch.object(mm.Model, "set_magnification_methods") as set_methods:
        _build_binary_model(p_fs, COORDS, "auto_vbbl", use_rho=True)
        assert set_methods.call_args[0][0][1] == 'VBM'

    with patch.object(mm.Model, "set_magnification_methods") as set_methods:
        _build_binary_model(p_ps, COORDS, "auto_vbbl", use_rho=False)
        assert set_methods.call_args[0][0][1] == 'VBBL'


def test_lens_rejects_missing_body_component():
    """
    Given a lens config referencing 'planet.0' while the system has no
      planet component,
    When register_parameters validates the body references,
    Then a clear ValueError is raised instead of an AttributeError deep in
      the model build.
    """
    # Arrange
    lens = Lens([{"lenses": ["star.0", "planet.0"], "sources": ["star.1"]}],
                _DummyConfigManager())
    system = _DummySystem()
    system.star = _DummyComponent(2)

    # Act / Assert
    with pytest.raises(ValueError, match="planet"):
        lens.register_parameters(system)


def test_lens_rejects_out_of_range_body_index():
    """
    Given a lens config whose source index exceeds the number of configured
      star instances,
    When register_parameters validates the body references,
    Then a ValueError naming the out-of-range reference is raised.
    """
    # Arrange
    lens = Lens([{"lenses": ["star.0"], "sources": ["star.5"]}],
                _DummyConfigManager())
    system = _DummySystem()
    system.star = _DummyComponent(2)

    # Act / Assert
    with pytest.raises(ValueError, match="out of range"):
        lens.register_parameters(system)


def test_lens_rejects_malformed_body_reference():
    """
    Given a body reference without an index ('planet' instead of 'planet.0'),
    When the Lens component parses its config,
    Then a ValueError explaining the expected format is raised.
    """
    with pytest.raises(ValueError, match="body reference"):
        Lens([{"lenses": ["planet"], "sources": ["star.1"]}],
             _DummyConfigManager())


def test_lens_rejects_multiple_events():
    """
    Given a lens config with two entries (two independent event geometries),
    When the Lens component is constructed,
    Then a ValueError states that only one event may be modeled at a time
      (instead of downstream code silently fitting all data with event 0).
    """
    with pytest.raises(ValueError, match="one lensing event"):
        Lens([{"lenses": ["star.0"], "sources": ["star.1"]},
              {"lenses": ["star.2"], "sources": ["star.3"]}],
             _DummyConfigManager())


def test_n_lens_bodies_are_accepted_and_sized_per_companion():
    """
    Given a lens config with three lens bodies (one primary + two companions),
    When the Lens registers its parameters,
    Then construction succeeds (no triple-lens rejection) and the companion
      geometry parameters s/xalpha/yalpha are sized per companion.
    """
    # Arrange
    lens = Lens([{"lenses": ["star.0", "planet.0", "planet.1"],
                  "sources": ["star.1"]}],
                _DummyConfigManager())
    system = _DummySystem()
    system.star = _DummyComponent(2)
    system.planet = _DummyComponent(2)

    # Act
    lens.register_parameters(system)

    # Assert
    assert lens.n_companions == 2
    assert lens.manifest["s"]["shape"] == (2,)
    assert lens.manifest["xalpha"]["shape"] == (2,)
    assert lens.manifest["yalpha"]["shape"] == (2,)


def test_triple_lens_magnification_fails_loudly():
    """
    Given a lens with three bodies (no magnification backend for N > 2 yet),
    When get_magnification_op is called,
    Then a NotImplementedError names the backend limitation instead of
      silently computing a binary-lens magnification.
    """
    lens = Lens([{"lenses": ["star.0", "planet.0", "planet.1"],
                  "sources": ["star.1"]}],
                _DummyConfigManager())
    with pytest.raises(NotImplementedError, match="backend"):
        lens.get_magnification_op(None, None, None, index=0)


def _make_inst_with_q_source_data(n=870, t0=2458554.89, u0=0.143, tE=18.17,
                                   f_baseline=0.62, A_peak=6.0, peak_width=5):
    """Return a MulensInstrument whose _estimate_flux_components can be called.

    The synthetic light curve has f_baseline everywhere except for `peak_width`
    consecutive points near t_0 which are set to f_baseline * A_peak, simulating
    a sharp caustic crossing.  Magnitudes are pre-converted from flux.
    """
    inst = MulensInstrument.__new__(MulensInstrument)
    inst.config_manager = _DummyConfigManager()
    inst.config_manager.user_params = {
        "lens.0.t_0": {"initval": t0},
        "lens.0.u_0": {"initval": u0},
        "lens.0.t_E": {"initval": tE},
        "lens.0.pi_E_N": {"initval": 0.0},
        "lens.0.pi_E_E": {"initval": 0.0},
    }

    t = np.linspace(t0 - 40, t0 + 40, n)
    flux = np.full(n, f_baseline)
    # sharp caustic peak: `peak_width` points near t_0 boosted to A_peak * f_baseline
    peak_mask = np.abs(t - t0) < (peak_width * (t[1] - t[0]))
    flux[peak_mask] = f_baseline * A_peak
    m = -2.5 * np.log10(np.maximum(flux, 1e-30))
    xyz = np.zeros((n, 3))
    return inst, t, m, xyz


def test_q_source_estimate_pspl_broad_peak():
    """
    Given a PSPL-like light curve with a broad, well-sampled peak,
    When _estimate_flux_components runs,
    Then q_source is close to 1 (no blending, source is fully dominant).
    """
    inst, t, m, xyz = _make_inst_with_q_source_data(A_peak=7.0, peak_width=60)
    ra, dec = 0.0, 0.0
    _f_total, q, _q_flux = inst._estimate_flux_components(t, m, xyz, ra, dec, inst_idx=0)
    assert 0.7 < q <= 1.0, f"Expected q_source near 1, got {q:.3f}"


def test_flux_total_estimate_sharp_caustic_crossing():
    """
    Given a binary-lens light curve with a sharp caustic crossing where the
    true peak spans only a handful of data points (peak_width=5 out of 870),
    When _estimate_flux_components runs,
    Then f_total is within a factor of 2 of the true baseline flux.

    Note: q_source is underestimated for sharp binary caustics because the
    PSPL model used in the NNLS sees high A values at non-caustic near-peak
    times with baseline flux, driving f_source down.  f_total remains
    well-constrained because the sum f_source + f_blend ≈ f_baseline.
    """
    f_baseline = 0.62
    inst, t, m, xyz = _make_inst_with_q_source_data(
        A_peak=6.0, peak_width=5, f_baseline=f_baseline
    )
    ra, dec = 0.0, 0.0
    f_total, _q, _q_flux = inst._estimate_flux_components(t, m, xyz, ra, dec, inst_idx=0)
    assert 0.5 * f_baseline < f_total < 2.0 * f_baseline, (
        f"f_total should be within 2x of the true baseline {f_baseline:.3f}; "
        f"got {f_total:.3f}."
    )


def test_log_f_total_bootstrap_yields_to_user_params():
    """
    Given a MulensInstrument with a data-estimated total flux,
    When register_parameters declares the manifest,
    Then log_f_total is pushed as a RANK_DERIVED_DATA hint (so a user value in
      params.yaml wins) and the manifest carries no direct initval override
      (which would bypass provenance ranking and clobber the user's restart
      point from a previous MAP).
    """
    from exozippy.config import RANK_DERIVED_DATA

    class _RecordingConfigManager(_DummyConfigManager):
        def __init__(self):
            self.hints = {}

        def add_hint(self, path, value, rank=RANK_DERIVED_DATA):
            self.hints[path] = (value, rank)

    # Arrange
    inst = MulensInstrument.__new__(MulensInstrument)
    inst.config = [{"file": "dummy.txt"}]
    inst.n_elements = 1
    inst.names = ["Roman"]
    inst.config_manager = _RecordingConfigManager()
    inst.fs_init = [0.6038]
    inst.q_source_init = [0.65]

    # Act
    inst.register_parameters(_DummySystem())

    # Assert
    assert inst.manifest["log_f_total"] is None, (
        "manifest must not set initval directly — it would override the user's "
        "params.yaml value regardless of provenance rank"
    )
    hint_val, hint_rank = inst.config_manager.hints["mulensinstrument.0.log_f_total"]
    assert hint_val == pytest.approx(np.log10(0.6038))
    assert hint_rank == RANK_DERIVED_DATA


# ---------------------------------------------------------------------------
# q derived from masses (regression for ghost-parameter bug)
# ---------------------------------------------------------------------------

def test_q_absent_from_pspl_manifest():
    """
    Given a PSPL lens config with one lens body,
    When register_parameters runs,
    Then 'q' is not in the manifest (no companion, no mass ratio).
    """
    lens = Lens([{"lenses": ["star.0"], "sources": ["star.1"]}],
                _DummyConfigManager())
    system = _DummySystem()
    system.star = _DummyComponent(2)
    lens.build_maps()
    lens.register_parameters(system)

    assert "q" not in lens.manifest


def test_q_is_derived_for_planet_companion():
    """
    Given a binary lens with a planet companion,
    When register_parameters runs,
    Then 'q' is in the manifest as a derived parameter (has expr_key) and
      its deps reference 'planet.mass' for the companion.
    """
    lens = Lens([{"lenses": ["star.0", "planet.0"], "sources": ["star.1"]}],
                _DummyConfigManager())
    system = _DummySystem()
    system.star = _DummyComponent(2)
    system.planet = _DummyComponent(1)
    lens.build_maps()
    lens.register_parameters(system)

    assert "q" in lens.manifest
    q_entry = lens.manifest["q"]
    assert isinstance(q_entry, dict)
    assert q_entry.get("expr_key") == "default"
    deps = q_entry.get("deps", [])
    assert any("planet.mass" in d for d in deps), (
        f"planet companion: expected 'planet.mass' dep, got {deps}"
    )
    assert any("star.mass" in d for d in deps), (
        f"planet companion: expected 'star.mass' dep for primary, got {deps}"
    )


def test_q_deps_use_star_mass_for_stellar_binary():
    """
    Given a binary lens with a stellar companion (two stars),
    When register_parameters runs,
    Then 'q' deps reference 'star.mass' for both primary and companion
      (not 'planet.mass').
    """
    lens = Lens([{"lenses": ["star.0", "star.1"], "sources": ["star.2"]}],
                _DummyConfigManager())
    system = _DummySystem()
    system.star = _DummyComponent(3)
    lens.build_maps()
    lens.register_parameters(system)

    deps = lens.manifest["q"]["deps"]
    assert all("star.mass" in d for d in deps), (
        f"stellar binary: all q deps should reference star.mass, got {deps}"
    )
    assert not any("planet" in d for d in deps), (
        f"stellar binary: no q dep should reference planet, got {deps}"
    )


def test_companion_mass_map_points_to_correct_index():
    """
    Given a binary lens where the companion is planet.0,
    When build_maps runs,
    Then primary_lens_map points to star index 0 and
      companion_mass_map points to planet index 0.
    """
    lens = Lens([{"lenses": ["star.0", "planet.0"], "sources": ["star.1"]}],
                _DummyConfigManager())
    lens.build_maps()

    np.testing.assert_array_equal(lens.primary_lens_map, [0])
    np.testing.assert_array_equal(lens.companion_mass_map, [0])


def test_companion_mass_map_stellar_binary_points_to_second_star():
    """
    Given a stellar binary (star.0 primary, star.1 companion),
    When build_maps runs,
    Then companion_mass_map points to star index 1.
    """
    lens = Lens([{"lenses": ["star.0", "star.1"], "sources": ["star.2"]}],
                _DummyConfigManager())
    lens.build_maps()

    np.testing.assert_array_equal(lens.primary_lens_map, [0])
    np.testing.assert_array_equal(lens.companion_mass_map, [1])


def test_calc_q_returns_mass_ratio():
    """
    Given companion mass 0.001 and lens mass 0.5 (solar masses),
    When calc_q is called,
    Then the result is 0.001 / 0.5 = 0.002.
    """
    import pytensor.tensor as pt
    import pytensor
    from exozippy.components.mulensing.physics import calc_q

    m_companion = pt.as_tensor_variable(np.array([0.001]))
    m_lens = pt.as_tensor_variable(np.array([0.5]))
    result = float(pytensor.function([], calc_q(m_companion, m_lens))()[0])
    assert result == pytest.approx(0.002, rel=1e-6)


# ---------------------------------------------------------------------------
# sampler_requirements() hook
# ---------------------------------------------------------------------------

def test_pspl_lens_has_no_sampler_requirements():
    """
    Given a PSPL lens (single lens body, no finite source, no use_op flag),
    When sampler_requirements is called,
    Then it returns an empty dict — PSPL uses a symbolic PyTensor formula
      that is NUTS-compatible and imposes no sampler constraints.
    """
    lens = Lens([{"lenses": ["star.0"], "sources": ["star.1"]}],
                _DummyConfigManager())
    assert lens.sampler_requirements() == {}


def test_binary_lens_requires_ptde_and_rejects_gradient_samplers():
    """
    Given a binary lens (two lens bodies — uses the MulensModel Op),
    When sampler_requirements is called,
    Then the returned dict marks 'nuts', 'numpyro', and 'blackjax' as
      incompatible and recommends 'ptde', because the Op is not
      differentiable and gradient-based samplers produce invalid results.
    """
    lens = Lens([{"lenses": ["star.0", "planet.0"], "sources": ["star.1"]}],
                _DummyConfigManager())

    reqs = lens.sampler_requirements()

    assert 'incompatible' in reqs
    assert {'nuts', 'numpyro', 'blackjax'} <= reqs['incompatible']
    assert reqs.get('recommended') == 'ptde'


def test_pspl_finite_source_requires_ptde():
    """
    Given a PSPL lens with finite_source: True (also uses the MulensModel Op),
    When sampler_requirements is called,
    Then gradient-based samplers are marked incompatible and 'ptde' is recommended.
    """
    lens = Lens([{"lenses": ["star.0"], "sources": ["star.1"],
                  "finite_source": True}],
                _DummyConfigManager())
    reqs = lens.sampler_requirements()
    assert 'nuts' in reqs.get('incompatible', set())
    assert reqs.get('recommended') == 'ptde'


# ---------------------------------------------------------------------------
# Unknown sampler key warning
# ---------------------------------------------------------------------------

def test_known_sampler_keys_excludes_legacy_step_method():
    """
    Given the set of recognized sampler config keys,
    When checked for the legacy 'step_method' key a student used previously,
    Then 'step_method' is absent (so the unknown-key warning fires) and
      'method' is present (the correct key for choosing the sampler).
    """
    assert "step_method" not in KNOWN_SAMPLER_KEYS
    assert "method" in KNOWN_SAMPLER_KEYS


def test_unknown_sampler_key_is_detected(caplog):
    """
    Given a sampler config dict containing the unrecognized key 'step_method',
    When the set difference against KNOWN_SAMPLER_KEYS is computed
      (replicating the logic in run.py),
    Then 'step_method' appears in the unknown-key list and 'draws' does not.
    """
    sampler_cfg = {"step_method": "PTDE", "draws": 1000, "method": "ptde"}

    unknown = sorted(set(sampler_cfg) - KNOWN_SAMPLER_KEYS)

    assert "step_method" in unknown
    assert "draws" not in unknown
    assert "method" not in unknown
