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
      geometry parameters s/cosalpha/sinalpha are sized per companion.
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
    assert lens.manifest["cosalpha"]["shape"] == (2,)
    assert lens.manifest["sinalpha"]["shape"] == (2,)


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
    """Return a MulensInstrument whose _estimate_q_source can be called.

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
    When _estimate_q_source runs,
    Then q_source is close to 1 (no blending, source is fully dominant).
    """
    inst, t, m, xyz = _make_inst_with_q_source_data(A_peak=7.0, peak_width=60)
    ra, dec = 0.0, 0.0
    q = inst._estimate_q_source(t, m, xyz, ra, dec)
    assert 0.7 < q <= 1.95, f"Expected q_source near 1, got {q:.3f}"


def test_q_source_estimate_sharp_caustic_crossing():
    """
    Given a binary-lens light curve with a sharp caustic crossing where the
    true peak spans only a handful of data points (peak_width=5 out of 870),
    When _estimate_q_source runs,
    Then q_source is NOT driven near 0 by a median that misses the peak.

    This is a regression test: the old median-of-top-10% code returned
    q_source ~ 0.115 for the DC2018 event 128; the fix (using np.max) correctly
    recovers a value near 1 when there is no blending.
    """
    inst, t, m, xyz = _make_inst_with_q_source_data(A_peak=6.0, peak_width=5)
    ra, dec = 0.0, 0.0
    q = inst._estimate_q_source(t, m, xyz, ra, dec)
    assert q > 0.5, (
        f"q_source should be > 0.5 for an unblended sharp caustic crossing; "
        f"got {q:.3f}.  The median-of-top-10% bug would return ~0.1."
    )


def test_band_ndx_warns_that_ld_is_not_wired(caplog):
    """
    Given an instrument config that sets band_ndx,
    When register_parameters runs,
    Then a warning states that limb darkening is not yet applied to the
      magnification (instead of silently dropping it).
    """
    # Arrange
    inst = MulensInstrument([{"file": "unused.txt", "band_ndx": 0}],
                            _DummyConfigManager())
    inst.fs_init = [1000.0]
    inst.q_source_init = [1.0]

    # Act
    with caplog.at_level(logging.WARNING):
        inst.register_parameters(_DummySystem())

    # Assert
    assert any("limb darkening" in record.message for record in caplog.records)


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
