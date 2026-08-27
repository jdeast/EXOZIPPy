"""Tests for code-review fixes in the MulensModel Op layer and Lens/Instrument
config validation."""

import logging
import warnings

import numpy as np
import pytensor.tensor as pt
import pytest

from conftest import _DummyComponent, _DummyConfigManager, _DummySystem
from exozippy.components.mulensing.lens import Lens
from exozippy.components.mulensing.mulensinstrument import MulensInstrument
from exozippy.components.mulensing.op import (
    BinaryLensMagOp,
    MulensMagOp,
    VBMDirectMagOp,
    _build_binary_model,
    _build_pspl_model,
    _dev_skycoord,
    _MagGradOp,
)
from exozippy.run import KNOWN_SAMPLER_KEYS

COORDS = "270.0d -28.0d"


def test_dev_skycoord_cache_distinguishes_same_length_arrays():
    """
    Given two observer-deviation arrays with the same number of epochs but
      different positions (e.g. ground and a satellite over one model grid),
    When both are passed through the same coordinate cache,
    Then each gets its own SkyCoord (the old length-keyed cache silently
      returned the first observer's coordinates for the second).
    """
    # Arrange
    cache = {}
    ground = np.zeros((5, 3))
    satellite = np.ones((5, 3))

    # Act
    coord_ground = _dev_skycoord(ground, cache)
    coord_sat = _dev_skycoord(satellite, cache)

    # Assert
    assert not np.allclose(
        coord_ground.cartesian.xyz.value, coord_sat.cartesian.xyz.value
    )
    assert _dev_skycoord(ground, cache) is coord_ground
    assert _dev_skycoord(satellite, cache) is coord_sat


def test_ephemeris_caches_key_on_bytes_not_on_a_hash():
    """
    Given the two observer-deviation caches in op.py,
    When their keys are inspected,
    Then the array's BYTES are in the key, not hash(bytes): a 64-bit siphash
      can collide, and the whole point of these caches is telling apart two
      same-shaped deviation arrays, so a collision would silently reuse the
      wrong observer's parallax deltas (review 2.6.4).
    """
    # Arrange
    cache = {}
    dev = np.arange(15, dtype=float).reshape(5, 3)
    op = VBMDirectMagOp(COORDS, n_companions=1)

    # Act
    _dev_skycoord(dev, cache)
    op._deltas(dev)

    # Assert
    for store in (cache, op._delta_cache):
        ((shape, blob),) = store.keys()
        assert shape == (5, 3)
        assert blob == dev.tobytes()


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


# Binary-lens parameter vectors: [t_0, u_0, t_E, pi_E_N, pi_E_E, (rho), s, q, alpha]
_P_BINARY_FS = np.array(
    [2450000.0, 0.1, 20.0, 0.0, 0.0, 1e-3, 1.2, 0.01, 30.0]
)
_P_BINARY_PS = np.array([2450000.0, 0.1, 20.0, 0.0, 0.0, 1.2, 0.01, 30.0])


def test_binary_method_selection_follows_finite_source_flag():
    """
    Given a binary-lens parameter vector,
    When the model is built with and without finite_source (use_rho),
    Then the finite-source method (VBM) is selected iff use_rho is True,
      independent of the runtime rho value, and the point-source case gets
      MulensModel's point-source method (asking VBBL for a rho-less model
      raises inside MulensModel).

    Note this asserts against the REAL model object, not a mocked
    set_magnification_methods -- the mock is what let the point-source
    "VBBL" selection ship broken.
    """
    # Act
    model_fs = _build_binary_model(
        _P_BINARY_FS, COORDS, "auto_vbbl", use_rho=True
    )
    model_ps = _build_binary_model(
        _P_BINARY_PS, COORDS, "auto_vbbl", use_rho=False
    )

    # Assert
    assert model_fs.methods[1] == "VBM"
    assert model_ps.methods[1] == "point_source"


def test_point_source_binary_model_yields_finite_magnifications():
    """
    Given a point-source binary lens (backend: mulensmodel, finite_source
      false),
    When magnifications are computed through the real MulensModel call path,
    Then they are finite and above 1 -- the old "VBBL" selection made
      MulensModel raise, which perform() turned into an all-NaN curve and
      hence -inf logp for every proposal.
    """
    # Arrange
    op = BinaryLensMagOp(COORDS, mag_method="auto_vbbl", use_rho=False)
    times = np.linspace(2449980.0, 2450020.0, 41)
    obs_pos = np.zeros((times.size, 3))
    outputs = [[None]]

    # Act
    op.perform(None, [_P_BINARY_PS, times, obs_pos], outputs)
    magnification = outputs[0][0]

    # Assert
    assert np.all(np.isfinite(magnification))
    assert np.all(magnification > 1.0)


def test_finite_source_binary_model_yields_finite_magnifications():
    """
    Given a finite-source binary lens (finite_source true, so VBM),
    When magnifications are computed through the real MulensModel call path,
    Then they are finite and above 1, and agree with the point-source result
      to rtol 1e-3 -- rho is small, so the finite-source signal is tiny but
      NOT zero.  (The original rtol 1e-4 was calibrated while the silent
      2-element magnification-methods list made "VBM" a no-op, so the
      "finite-source" curve WAS the point-source curve; with the dispatch
      fixed -- mulensing.md "Lens orbital motion" -- the two genuinely
      differ by ~3e-4 at the peak epochs, and that difference is the
      evidence the method now runs.)
    """
    # Arrange
    times = np.linspace(2449980.0, 2450020.0, 41)
    obs_pos = np.zeros((times.size, 3))
    op_fs = BinaryLensMagOp(COORDS, mag_method="auto_vbbl", use_rho=True)
    op_ps = BinaryLensMagOp(COORDS, mag_method="auto_vbbl", use_rho=False)
    out_fs, out_ps = [[None]], [[None]]

    # Act
    op_fs.perform(None, [_P_BINARY_FS, times, obs_pos], out_fs)
    op_ps.perform(None, [_P_BINARY_PS, times, obs_pos], out_ps)

    # Assert
    assert np.all(np.isfinite(out_fs[0][0]))
    assert np.all(out_fs[0][0] > 1.0)
    np.testing.assert_allclose(out_fs[0][0], out_ps[0][0], rtol=1e-3)
    # ... and the finite-source signal is nonzero: the methods actually ran.
    assert np.max(np.abs(out_fs[0][0] / out_ps[0][0] - 1.0)) > 1e-5


def test_mag_op_warns_once_when_falling_back_to_nan():
    """
    Given an Op configured with a magnification method MulensModel rejects,
    When perform() is called twice,
    Then both calls return NaN (so the sampler rejects the proposal) but a
      single RuntimeWarning naming the underlying error is emitted -- the
      silent all-NaN fallback is what hid the point-source binary bug.
    """
    # Arrange
    op = BinaryLensMagOp(COORDS, mag_method="VBBL", use_rho=False)
    times = np.linspace(2449980.0, 2450020.0, 11)
    obs_pos = np.zeros((times.size, 3))
    first, second = [[None]], [[None]]

    # Act
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        op.perform(None, [_P_BINARY_PS, times, obs_pos], first)
        op.perform(None, [_P_BINARY_PS, times, obs_pos], second)

    # Assert
    assert np.all(np.isnan(first[0][0]))
    assert np.all(np.isnan(second[0][0]))
    runtime = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert len(runtime) == 1
    assert "VBBL" in str(runtime[0].message)


class _RaisingVBM:
    """A stand-in for the VBMicrolensing object that fails the way a broken
    backend does: every magnification call raises.  Real causes are a
    SetLensGeometry rejection, a SWIG ValueError, or API drift in a new
    wheel."""

    a1 = 0.0
    Tol = 1e-3
    RelTol = 0.0

    def _boom(self, *args, **kwargs):
        raise ValueError("SetLensGeometry: bad lens configuration")

    BinaryMag0 = BinaryMag2 = MultiMag0 = MultiMag2 = _boom
    SetLensGeometry = _boom


def test_vbm_direct_op_warns_once_when_the_backend_raises():
    """
    Given a VBMDirectMagOp whose VBMicrolensing backend raises on every call
      (a misconfigured/broken backend, not a bad proposal),
    When perform() is called twice with a perfectly ordinary parameter vector,
    Then both calls return NaN (so the proposal is rejected) AND a single
      RuntimeWarning naming the underlying error is emitted -- silently
      returning all-NaN forever is how a default-backend binary fit runs to a
      garbage posterior with no message (review 2.6.1).
    """
    # Arrange
    op = VBMDirectMagOp(COORDS, n_companions=1, use_rho=False)
    op._vbm = _RaisingVBM()
    times = np.linspace(2449980.0, 2450020.0, 11)
    obs_pos = np.zeros((times.size, 3))
    first, second = [[None]], [[None]]

    # Act
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        op.perform(None, [_P_BINARY_PS, times, obs_pos], first)
        op.perform(None, [_P_BINARY_PS, times, obs_pos], second)

    # Assert
    assert np.all(np.isnan(first[0][0]))
    assert np.all(np.isnan(second[0][0]))
    runtime = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert len(runtime) == 1
    assert "SetLensGeometry" in str(runtime[0].message)
    assert "VBMDirectMagOp" in str(runtime[0].message)


def test_vbm_direct_op_nonfinite_warning_does_not_silence_the_backend_one():
    """
    Given a VBMDirectMagOp that first sees a non-finite parameter vector and
      then hits a raising backend,
    When perform() is called for each in turn,
    Then BOTH are reported: the two failure modes have different fixes (fix
      the model vs. fix the install), so one warn-once flag would let a burst
      of NaN proposals permanently hide a broken backend.
    """
    # Arrange
    op = VBMDirectMagOp(COORDS, n_companions=1, use_rho=False)
    times = np.linspace(2449980.0, 2450020.0, 11)
    obs_pos = np.zeros((times.size, 3))
    nonfinite = _P_BINARY_PS.copy()
    nonfinite[1] = np.nan
    out = [[None]]

    # Act
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        op.perform(None, [nonfinite, times, obs_pos], out)
        op._vbm = _RaisingVBM()
        op.perform(None, [_P_BINARY_PS, times, obs_pos], out)

    # Assert
    messages = [
        str(w.message)
        for w in caught
        if issubclass(w.category, RuntimeWarning)
    ]
    assert len(messages) == 2
    assert any("non-finite" in m for m in messages)
    assert any("SetLensGeometry" in m for m in messages)


# ---------------------------------------------------------------------------
# Gradient honesty on the Op path (reviews 1.6.3, 2.6.5)
# ---------------------------------------------------------------------------


def _mag_op_output(op, n_times=7):
    """Apply an Op to symbolic inputs and hand back (output, param vector)."""
    p = pt.dvector("p")
    times = pt.dvector("t")
    obs = pt.dmatrix("obs")
    return op(p, times, obs), p


@pytest.mark.parametrize(
    "op",
    [
        MulensMagOp(COORDS, mag_method="point_source"),
        BinaryLensMagOp(COORDS, mag_method="auto_vbbl", use_rho=False),
    ],
    ids=["pspl", "binary"],
)
def test_mulensmodel_op_refuses_to_be_differentiated(op):
    """
    Given a MulensModel-backed magnification Op,
    When a gradient is requested through it,
    Then it raises rather than quietly wiring a forward-difference _MagGradOp
      that burns N_params+1 full light curves per step for an O(eps) answer
      (review 2.6.5) -- the same refusal VBMDirectMagOp.pullback already makes.
    """
    # Arrange
    out, p = _mag_op_output(op)

    # Act / Assert
    with pytest.raises(NotImplementedError, match="no gradient"):
        pt.grad(pt.sum(out), p)


def test_vbm_direct_op_refuses_to_be_differentiated():
    """
    Given the default binary backend,
    When a gradient is requested through it,
    Then it raises -- pinned here beside the MulensModel Ops so the two
      backends cannot drift apart on this again.
    """
    # Arrange
    out, p = _mag_op_output(VBMDirectMagOp(COORDS, n_companions=1))

    # Act / Assert
    with pytest.raises(NotImplementedError, match="no gradient"):
        pt.grad(pt.sum(out), p)


def test_mag_grad_op_refuses_a_second_derivative():
    """
    Given _MagGradOp (the forward-difference gradient machinery, now reachable
      only if someone revives it),
    When a second derivative is requested,
    Then it RAISES rather than returning DisconnectedType: the output IS
      linear in the incoming cotangent, so "disconnected" would be a false
      claim handing back an absent gradient, while a Hessian built on a
      first-order finite difference is numerically meaningless anyway
      (review 1.6.3).
    """
    # Arrange
    grad_op = _MagGradOp(_build_pspl_model, COORDS, "point_source")
    p = pt.dvector("p")
    times = pt.dvector("t")
    obs = pt.dmatrix("obs")
    g = pt.dvector("g")
    out = grad_op(p, times, obs, g)

    # Act / Assert
    with pytest.raises(NotImplementedError, match="second derivative"):
        pt.grad(pt.sum(out), g)
    with pytest.raises(NotImplementedError, match="second derivative"):
        pt.grad(pt.sum(out), p)


def test_mag_grad_op_still_declares_the_cotangent_connected():
    """
    Given _MagGradOp,
    When its connection_pattern is read,
    Then the incoming cotangent is CONNECTED -- the output really is linear in
      it, and declaring it disconnected (the obvious way to "fix" the wrongly
      shaped VJP it used to return) would let pytensor skip pullback and hand
      back an absent gradient instead of the error above.
    """
    # Arrange
    grad_op = _MagGradOp(_build_pspl_model, COORDS, "point_source")

    # Act
    pattern = grad_op.connection_pattern(None)

    # Assert
    assert pattern[0] == [True]  # parameters
    assert pattern[3] == [True]  # incoming cotangent


def test_lens_rejects_missing_body_component():
    """
    Given a lens config referencing 'planet.0' while the system has no
      planet component,
    When register_parameters validates the body references,
    Then a clear ValueError is raised instead of an AttributeError deep in
      the model build.
    """
    # Arrange
    lens = Lens(
        [{"lenses": ["star.0", "planet.0"], "sources": ["star.1"]}],
        _DummyConfigManager(),
    )
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
    lens = Lens(
        [{"lenses": ["star.0"], "sources": ["star.5"]}], _DummyConfigManager()
    )
    system = _DummySystem()
    system.star = _DummyComponent(2)

    # Act / Assert
    with pytest.raises(ValueError, match="out of range"):
        lens.register_parameters(system)


def test_lens_rejects_a_non_star_primary_body():
    """
    Given a lens config whose PRIMARY (first) lens body is a planet,
    When register_parameters validates the body references,
    Then a ValueError explains that the primary must be a star and points at
      the workaround.

    This config used to build happily and be silently wrong: the lens maps
    carry only an index and every primary-side dependency is hard-coded to
    the star component (star.mass[lens_map], star.distance[lens_map],
    star.pm_*[lens_map]).  Measured on examples/ob08092, lenses:
    ["planet.0"] produced a theta_E bit-identical to lenses: ["star.0"],
    responding to that star's mass and completely insensitive to the
    planet's -- a fit that finishes and reports a lens mass which never
    entered the likelihood.
    """
    # Arrange
    lens = Lens(
        [{"lenses": ["planet.0"], "sources": ["star.0"]}],
        _DummyConfigManager(),
    )
    system = _DummySystem()
    system.star = _DummyComponent(1)
    system.planet = _DummyComponent(1)

    # Act / Assert
    with pytest.raises(ValueError, match="primary") as excinfo:
        lens.register_parameters(system)
    message = str(excinfo.value)
    assert "planet.0" in message, "the offending entry must be named"
    assert "must be a star" in message
    assert "logmass" in message, "the workaround must be spelled out"


def test_lens_accepts_a_planet_companion_behind_a_star_primary():
    """
    Given the standard binary-lens topology (star primary, planet companion),
    When register_parameters validates the body references,
    Then no error is raised -- the primary-type guard must not touch the
      companion slots, whose mass dependencies already carry the component
      type.
    """
    # Arrange
    lens = Lens(
        [{"lenses": ["star.0", "planet.0"], "sources": ["star.1"]}],
        _DummyConfigManager(),
    )
    system = _DummySystem()
    system.star = _DummyComponent(2)
    system.planet = _DummyComponent(1)

    # Act
    lens._validate_bodies(system)

    # Assert
    assert lens.lens_bodies[0] == [("star", 0), ("planet", 0)]
    assert lens.n_companions == 1


def test_lens_accepts_a_stellar_companion_and_a_plain_single_star_lens():
    """
    Given a stellar-binary lens and a plain single-star PSPL lens,
    When the body references are validated,
    Then neither raises (the guard is scoped to a non-star PRIMARY only).
    """
    # Arrange
    system = _DummySystem()
    system.star = _DummyComponent(3)

    binary = Lens(
        [{"lenses": ["star.0", "star.1"], "sources": ["star.2"]}],
        _DummyConfigManager(),
    )
    single = Lens([{"lens_ndx": 0, "source_ndx": 1}], _DummyConfigManager())

    # Act / Assert
    binary._validate_bodies(system)
    single._validate_bodies(system)
    assert single.lens_bodies[0] == [("star", 0)]


def test_lens_rejects_a_non_star_source_body():
    """
    Given a lens config whose source body is a planet,
    When register_parameters validates the body references,
    Then a ValueError explains that a source must be a star, names the star
      that would otherwise have been modeled, and gives the workaround.

    source_map is index-only exactly like lens_map and the whole
    source-side chain resolves through the star component
    (star.distance[source_map], star.pm_*[source_map],
    star.radius[source_map], get_magnification's star.ra/dec), so the
    failure mode is identical to the lens-primary one.
    """
    # Arrange
    lens = Lens(
        [{"lenses": ["star.0"], "sources": ["planet.0"]}],
        _DummyConfigManager(),
    )
    system = _DummySystem()
    system.star = _DummyComponent(1)
    system.planet = _DummyComponent(1)

    # Act / Assert
    with pytest.raises(ValueError, match="must be a") as excinfo:
        lens.register_parameters(system)
    message = str(excinfo.value)
    assert "planet.0" in message, "the offending entry must be named"
    assert "star.0" in message, "name the star that would be modeled instead"
    assert "logmass" in message, "the workaround must be spelled out"


def test_lens_rejects_a_non_star_body_in_a_second_source_slot():
    """
    Given a binary-source (2S) config whose SECOND source body is a planet,
    When the body references are validated,
    Then it is rejected too -- unlike the lens side, there is no companion
      position where a non-star body is meaningful: every source body is an
      independently monitored luminous star.
    """
    # Arrange
    lens = Lens(
        [{"lenses": ["star.0"], "sources": ["star.1", "planet.0"]}],
        _DummyConfigManager(),
    )
    system = _DummySystem()
    system.star = _DummyComponent(2)
    system.planet = _DummyComponent(1)

    # Act / Assert
    with pytest.raises(ValueError, match="source body 'planet.0'"):
        lens._validate_bodies(system)


def test_lens_accepts_multiple_star_sources():
    """
    Given a binary-source (2S) config whose sources are both stars,
    When the body references are validated,
    Then no error is raised and both source slots survive.
    """
    # Arrange
    lens = Lens(
        [{"lenses": ["star.0"], "sources": ["star.1", "star.2"]}],
        _DummyConfigManager(),
    )
    system = _DummySystem()
    system.star = _DummyComponent(3)

    # Act
    lens._validate_bodies(system)

    # Assert
    assert lens.source_bodies[0] == [("star", 1), ("star", 2)]
    assert lens.n_sources == 2


def test_omitted_sources_key_with_one_star_is_caught_not_silent():
    """
    Given a config that omits 'sources:' entirely while defining only ONE
      star (so the source_ndx default of 1 points at a star that does not
      exist),
    When the body references are validated,
    Then the existing out-of-range check catches it with a clear message.

    Recorded because the default is easy to trip: 'sources' defaults to
    [("star", source_ndx)] with source_ndx = 1, i.e. the SECOND star.
    """
    # Arrange
    lens = Lens([{}], _DummyConfigManager())
    system = _DummySystem()
    system.star = _DummyComponent(1)

    # Act / Assert
    with pytest.raises(ValueError, match="out of range") as excinfo:
        lens._validate_bodies(system)
    assert "star.1" in str(excinfo.value)


def test_lens_rejects_a_body_that_is_both_lens_and_source():
    """
    Given a config listing the same star as both the lens and the source,
    When the body references are validated,
    Then a ValueError explains that they must be distinct objects and why.

    pi_rel = 1000/d_L - 1000/d_S is identically 0 for one body, so theta_E
    collapses onto its floor and the likelihood is NaN from the first
    evaluation -- which otherwise surfaces as a baffling sampler-init
    failure far from the config line that caused it.
    """
    # Arrange
    lens = Lens(
        [{"lenses": ["star.0"], "sources": ["star.0"]}],
        _DummyConfigManager(),
    )
    system = _DummySystem()
    system.star = _DummyComponent(2)

    # Act / Assert
    with pytest.raises(ValueError, match="BOTH a lens body") as excinfo:
        lens._validate_bodies(system)
    message = str(excinfo.value)
    assert "star.0" in message
    assert "pi_rel" in message and "NaN" in message


def test_lens_rejects_self_lensing_via_the_legacy_ndx_keys():
    """
    Given the legacy spelling lens_ndx == source_ndx,
    When the body references are validated,
    Then the same error is raised -- the legacy keys normalize into
      lens_bodies/source_bodies in __init__, so one check covers both
      spellings.
    """
    # Arrange
    lens = Lens([{"lens_ndx": 0, "source_ndx": 0}], _DummyConfigManager())
    system = _DummySystem()
    system.star = _DummyComponent(2)

    # Act / Assert
    with pytest.raises(ValueError, match="BOTH a lens body"):
        lens._validate_bodies(system)


def test_lens_rejects_overlap_with_a_second_source_body():
    """
    Given a binary-source (2S) config where the lens star also appears in
      the SECOND source slot,
    When the body references are validated,
    Then the overlap is caught: the check compares the whole lists, not
      just the primary slots.
    """
    # Arrange
    lens = Lens(
        [{"lenses": ["star.0", "star.1"], "sources": ["star.2", "star.0"]}],
        _DummyConfigManager(),
    )
    system = _DummySystem()
    system.star = _DummyComponent(3)

    # Act / Assert
    with pytest.raises(ValueError, match="'star.0' is listed as BOTH"):
        lens._validate_bodies(system)


def test_distinct_lens_and_source_bodies_are_accepted():
    """
    Given ordinary configs (single-star PSPL by legacy default, an explicit
      binary lens, and a 2S source list) where no body is shared,
    When the body references are validated,
    Then nothing is raised.
    """
    # Arrange
    system = _DummySystem()
    system.star = _DummyComponent(4)
    system.planet = _DummyComponent(1)
    configs = [
        {},  # legacy defaults: lens_ndx 0, source_ndx 1
        {"lenses": ["star.0", "planet.0"], "sources": ["star.1"]},
        {"lenses": ["star.0"], "sources": ["star.1", "star.2"]},
    ]

    # Act / Assert
    for cfg in configs:
        Lens([cfg], _DummyConfigManager())._validate_bodies(system)


def test_lens_rejects_malformed_body_reference():
    """
    Given a body reference without an index ('planet' instead of 'planet.0'),
    When the Lens component parses its config,
    Then a ValueError explaining the expected format is raised.
    """
    with pytest.raises(ValueError, match="body reference"):
        Lens(
            [{"lenses": ["planet"], "sources": ["star.1"]}],
            _DummyConfigManager(),
        )


def test_lens_rejects_multiple_events():
    """
    Given a lens config with two entries (two independent event geometries),
    When the Lens component is constructed,
    Then a ValueError states that only one event may be modeled at a time
      (instead of downstream code silently fitting all data with event 0).
    """
    with pytest.raises(ValueError, match="one lensing event"):
        Lens(
            [
                {"lenses": ["star.0"], "sources": ["star.1"]},
                {"lenses": ["star.2"], "sources": ["star.3"]},
            ],
            _DummyConfigManager(),
        )


def test_n_lens_bodies_are_accepted_and_sized_per_companion():
    """
    Given a lens config with three lens bodies (one primary + two companions),
    When the Lens registers its parameters,
    Then construction succeeds (no triple-lens rejection) and the companion
      geometry parameters s/xalpha/yalpha are sized per companion.
    """
    # Arrange
    lens = Lens(
        [
            {
                "lenses": ["star.0", "planet.0", "planet.1"],
                "sources": ["star.1"],
            }
        ],
        _DummyConfigManager(),
    )
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


def test_triple_lens_mulensmodel_backend_fails_loudly():
    """
    Given a lens with three bodies and backend: mulensmodel (which caps the
      lens side at binary),
    When get_magnification_op is called,
    Then a NotImplementedError names the backend limitation instead of
      silently computing a binary-lens magnification.

    The default vbm_direct backend supports 3+ bodies via VBMicrolensing
    MultiMag2 (see test_vbm_direct_vs_mulensmodel.py).
    """
    lens = Lens(
        [
            {
                "lenses": ["star.0", "planet.0", "planet.1"],
                "sources": ["star.1"],
                "backend": "mulensmodel",
            }
        ],
        _DummyConfigManager(),
    )
    with pytest.raises(NotImplementedError, match="backend"):
        lens.get_magnification_op(None, None, None, index=0)


def test_lens_backend_defaults_to_vbm_direct_and_validates():
    """
    Given a lens block without a backend key,
    When the Lens is constructed,
    Then backend defaults to 'vbm_direct'; an unknown backend raises.
    """
    lens = Lens(
        [{"lenses": ["star.0", "planet.0"], "sources": ["star.1"]}],
        _DummyConfigManager(),
    )
    assert lens.backend == "vbm_direct"

    with pytest.raises(ValueError, match="backend"):
        Lens(
            [
                {
                    "lenses": ["star.0", "planet.0"],
                    "sources": ["star.1"],
                    "backend": "nope",
                }
            ],
            _DummyConfigManager(),
        )


def _make_inst_with_q_source_data(
    n=870,
    t0=2458554.89,
    u0=0.143,
    tE=18.17,
    f_baseline=0.62,
    A_peak=6.0,
    peak_width=5,
):
    """Return a MulensInstrument whose _estimate_flux_components can be called.

    The synthetic light curve has f_baseline everywhere except for `peak_width`
    consecutive points near t_0 which are set to f_baseline * A_peak, simulating
    a sharp caustic crossing.  The bootstrap consumes flux directly (the whole
    component fits in flux now), so the curve is handed over as-is.
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
    xyz = np.zeros((n, 3))
    return inst, t, flux, xyz


def test_q_source_estimate_pspl_broad_peak():
    """
    Given a PSPL-like light curve with a broad, well-sampled peak,
    When _estimate_flux_components runs,
    Then q_source is close to 1 (no blending, source is fully dominant).
    """
    inst, t, f, xyz = _make_inst_with_q_source_data(A_peak=7.0, peak_width=60)
    ra, dec = 0.0, 0.0
    _f_total, q, _q_flux = inst._estimate_flux_components(
        t, f, xyz, ra, dec, inst_idx=0
    )
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
    inst, t, f, xyz = _make_inst_with_q_source_data(
        A_peak=6.0, peak_width=5, f_baseline=f_baseline
    )
    ra, dec = 0.0, 0.0
    f_total, _q, _q_flux = inst._estimate_flux_components(
        t, f, xyz, ra, dec, inst_idx=0
    )
    assert 0.5 * f_baseline < f_total < 2.0 * f_baseline, (
        f"f_total should be within 2x of the true baseline {f_baseline:.3f}; "
        f"got {f_total:.3f}."
    )


def _make_2s_inst(f_blend=0.4, f_src=(0.6, 0.2), n=400):
    """A two-source instrument whose synthetic flux is exactly
    f_src[0]*A_0 + f_src[1]*A_1 + f_blend, so the bootstrap has a right
    answer to be measured against."""
    t0a, t0b, u0, tE = 2458554.89, 2458560.0, 0.3, 18.17
    inst = MulensInstrument.__new__(MulensInstrument)
    inst._n_sources = 2
    inst.config_manager = _DummyConfigManager()
    inst.config_manager.user_params = {
        "lens.0.t_0": {"initval": t0a},
        "lens.0.u_0": {"initval": u0},
        "lens.0.t_E": {"initval": tE},
        "lens.1.t_0": {"initval": t0b},
        "lens.1.u_0": {"initval": 0.15},
        "lens.1.t_E": {"initval": tE},
        "lens.0.pi_E_N": {"initval": 0.0},
        "lens.0.pi_E_E": {"initval": 0.0},
    }

    t = np.linspace(t0a - 40, t0a + 40, n)

    def pspl(t0_, u0_):
        tau = (t - t0_) / tE
        u = np.sqrt(u0_**2 + tau**2)
        return (u**2 + 2.0) / (u * np.sqrt(u**2 + 4.0))

    flux = f_src[0] * pspl(t0a, u0) + f_src[1] * pspl(t0b, 0.15) + f_blend
    return inst, t, flux, np.zeros((n, 3))


def test_multisource_bootstrap_honors_a_user_f_blend():
    """
    Given a two-source light curve and a params entry pinning f_blend,
    When the flux bootstrap runs,
    Then f_total - f_source equals that entry exactly: a user blend is a
      statement, not a starting guess.  The multi-source branch used to leave
      the constant column in the NNLS design and never read f_blend_user at
      all, so log_f_total and q_source were seeded from an estimate that
      contradicted the user (review 1.6.2).
    """
    # Arrange
    f_blend, f_src = 0.4, (0.6, 0.2)
    inst, t, flux, xyz = _make_2s_inst(f_blend=f_blend, f_src=f_src)
    # Deliberately NOT the truth: a wrong-but-pinned blend is what exposes
    # whether the estimate is being overridden or merely coincides.
    pinned = 0.25
    inst.config_manager.user_params["mulensinstrument.0.f_blend"] = {
        "initval": pinned
    }

    # Act
    f_total, q_source, _q_flux = inst._estimate_flux_components(
        t, flux, xyz, 0.0, 0.0, inst_idx=0
    )

    # Assert
    assert f_total * q_source == pytest.approx(f_total - pinned, rel=1e-9)
    assert f_total - f_total * q_source == pytest.approx(pinned, rel=1e-9)


def test_multisource_bootstrap_without_a_user_f_blend_is_unchanged():
    """
    Given the same two-source light curve and NO f_blend entry,
    When the flux bootstrap runs,
    Then it recovers the truth via the full NNLS -- the pre-existing path is
      untouched by the fix above.
    """
    # Arrange
    f_blend, f_src = 0.4, (0.6, 0.2)
    inst, t, flux, xyz = _make_2s_inst(f_blend=f_blend, f_src=f_src)

    # Act
    f_total, q_source, _q_flux = inst._estimate_flux_components(
        t, flux, xyz, 0.0, 0.0, inst_idx=0
    )

    # Assert
    assert f_total == pytest.approx(sum(f_src) + f_blend, rel=1e-6)
    assert f_total * q_source == pytest.approx(sum(f_src), rel=1e-6)


def test_nonpositive_user_f_total_falls_back_with_a_warning(caplog):
    """
    Given f_source and f_blend entries whose SUM is not positive,
    When the flux bootstrap runs,
    Then it falls back to the data's own baseline and warns naming both
      entries -- the both-user branch used to return the sum unchecked, so
      log_f_total took log10 of a non-positive number (surfacing much later as
      a missing-start error naming the wrong parameter) and
      _scale_flux_amplitudes derived negative upper bounds from it
      (review 2.6.3).
    """
    # Arrange -- a negative BLEND is legitimate (difference imaging); it is
    # the negative TOTAL that is not.
    inst, t, flux, xyz = _make_inst_with_q_source_data(f_baseline=0.62)
    inst.config_manager.user_params["mulensinstrument.0.f_source"] = {
        "initval": 0.5
    }
    inst.config_manager.user_params["mulensinstrument.0.f_blend"] = {
        "initval": -0.5
    }

    # Act
    with caplog.at_level("WARNING"):
        f_total, q_source, _q_flux = inst._estimate_flux_components(
            t, flux, xyz, 0.0, 0.0, inst_idx=0
        )

    # Assert
    assert f_total > 0.0
    assert 0.0 < q_source <= 1.0
    assert "f_source" in caplog.text and "f_blend" in caplog.text


def test_positive_user_f_total_is_still_taken_verbatim():
    """
    Given f_source and f_blend entries summing to a positive baseline,
    When the flux bootstrap runs,
    Then the user's numbers are used exactly -- the guard above must not
      disturb the ordinary both-user path.
    """
    # Arrange
    inst, t, flux, xyz = _make_inst_with_q_source_data()
    inst.config_manager.user_params["mulensinstrument.0.f_source"] = {
        "initval": 0.5
    }
    inst.config_manager.user_params["mulensinstrument.0.f_blend"] = {
        "initval": 0.1
    }

    # Act
    f_total, q_source, _q_flux = inst._estimate_flux_components(
        t, flux, xyz, 0.0, 0.0, inst_idx=0
    )

    # Assert
    assert f_total == pytest.approx(0.6)
    assert q_source == pytest.approx(0.5 / 0.6)


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
    # __init__ is bypassed above, so stand in for the state
    # register_parameters reads from it: the base's GP and robust-likelihood
    # configs (no file sets gp: or likelihood:, so both register nothing) and
    # the detrend column count (no extra data columns here).
    inst._load_gp_config()
    inst._load_likelihood_config()
    inst.total_detrend_cols = 0

    # Act
    inst.register_parameters(_DummySystem())

    # Assert
    assert inst.manifest["log_f_total"] is None, (
        "manifest must not set initval directly — it would override the user's "
        "params.yaml value regardless of provenance rank"
    )
    hint_val, hint_rank = inst.config_manager.hints[
        "mulensinstrument.0.log_f_total"
    ]
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
    lens = Lens(
        [{"lenses": ["star.0"], "sources": ["star.1"]}], _DummyConfigManager()
    )
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
    lens = Lens(
        [{"lenses": ["star.0", "planet.0"], "sources": ["star.1"]}],
        _DummyConfigManager(),
    )
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
    lens = Lens(
        [{"lenses": ["star.0", "star.1"], "sources": ["star.2"]}],
        _DummyConfigManager(),
    )
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
      companion0_mass_map points to planet index 0.
    """
    lens = Lens(
        [{"lenses": ["star.0", "planet.0"], "sources": ["star.1"]}],
        _DummyConfigManager(),
    )
    lens.build_maps()

    np.testing.assert_array_equal(lens.primary_lens_map, [0])
    np.testing.assert_array_equal(lens.companion0_mass_map, [0])


def test_companion_mass_map_stellar_binary_points_to_second_star():
    """
    Given a stellar binary (star.0 primary, star.1 companion),
    When build_maps runs,
    Then companion0_mass_map points to star index 1.
    """
    lens = Lens(
        [{"lenses": ["star.0", "star.1"], "sources": ["star.2"]}],
        _DummyConfigManager(),
    )
    lens.build_maps()

    np.testing.assert_array_equal(lens.primary_lens_map, [0])
    np.testing.assert_array_equal(lens.companion0_mass_map, [1])


def test_calc_q_returns_mass_ratio():
    """
    Given companion mass 0.001 and lens mass 0.5 (solar masses),
    When calc_q is called,
    Then the result is 0.001 / 0.5 = 0.002.
    """
    import pytensor
    import pytensor.tensor as pt

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
    lens = Lens(
        [{"lenses": ["star.0"], "sources": ["star.1"]}], _DummyConfigManager()
    )
    assert lens.sampler_requirements() == {}


def test_binary_lens_requires_ptde_and_rejects_gradient_samplers():
    """
    Given a binary lens (two lens bodies — uses the MulensModel Op),
    When sampler_requirements is called,
    Then the returned dict marks 'nuts', 'numpyro', and 'blackjax' as
      incompatible and recommends 'ptde_async', because the Op is not
      differentiable and gradient-based samplers produce invalid results.
    """
    lens = Lens(
        [{"lenses": ["star.0", "planet.0"], "sources": ["star.1"]}],
        _DummyConfigManager(),
    )

    reqs = lens.sampler_requirements()

    assert "incompatible" in reqs
    assert {"nuts", "numpyro", "blackjax"} <= reqs["incompatible"]
    assert reqs.get("recommended") == "ptde_async"


def test_pspl_finite_source_requires_ptde():
    """
    Given a PSPL lens with finite_source: True (also uses the MulensModel Op),
    When sampler_requirements is called,
    Then gradient-based samplers are marked incompatible and 'ptde_async'
      is recommended.
    """
    lens = Lens(
        [{"lenses": ["star.0"], "sources": ["star.1"], "finite_source": True}],
        _DummyConfigManager(),
    )
    reqs = lens.sampler_requirements()
    assert "nuts" in reqs.get("incompatible", set())
    assert reqs.get("recommended") == "ptde_async"


# ---------------------------------------------------------------------------
# t0_par re-resolution in load_data (2026-08-08 review item 1.11)
# ---------------------------------------------------------------------------


class _T0ParConfigManager(_DummyConfigManager):
    def __init__(self, user_params=None, seed_t0=None):
        self.user_params = user_params or {}
        self._seed_t0 = seed_t0

    def seed_start_value(self, path, seed=0):
        return self._seed_t0 if path == "lens.0.t_0" else None


def _t0_par_fixture(user_params=None, seed_t0=None, lens_config=None):
    inst = MulensInstrument.__new__(MulensInstrument)
    inst.config_manager = _T0ParConfigManager(user_params, seed_t0)
    system = _DummySystem()
    system.lens = _DummySystem()
    system.lens.config = [lens_config or {}]
    return inst, system


def test_t0_par_explicit_config_wins():
    """
    Given an explicit lens t0_par alongside a user t_0 and a seed,
    When the final t0_par is resolved in load_data,
    Then the explicit config value wins.
    """
    inst, system = _t0_par_fixture(
        user_params={"lens.0.t_0": {"initval": 2458800.0}},
        seed_t0=2458700.0,
        lens_config={"t0_par": 2458554.89},
    )
    times = np.linspace(2458500.0, 2458600.0, 11)
    assert inst._resolve_t0_par_final(system, times) == 2458554.89


def test_t0_par_user_t0_beats_seed():
    """
    Given both a user lens.0.t_0 initval and an MMEXOFAST seed,
    When the final t0_par is resolved,
    Then the user's value wins (seeds sit below RANK_USER).
    """
    inst, system = _t0_par_fixture(
        user_params={"lens.0.t_0": {"initval": 2458800.0}},
        seed_t0=2458700.0,
    )
    times = np.linspace(2458500.0, 2458600.0, 11)
    assert inst._resolve_t0_par_final(system, times) == 2458800.0


def test_t0_par_uses_mmexofast_seed():
    """
    Given no explicit t0_par and no user t_0 (the automated MMEXOFAST
      workflow deliberately omits the microlensing start values),
    When the final t0_par is resolved after the seeds arrived,
    Then the seed t_0 is used -- NOT the 2450000.0 construction-time default
      that parked the Skowron reference epoch ~8300 days before the data
      (2026-08-08 review item 1.11).
    """
    inst, system = _t0_par_fixture(seed_t0=2458554.89)
    times = np.linspace(2458500.0, 2458600.0, 11)
    assert inst._resolve_t0_par_final(system, times) == 2458554.89


def test_t0_par_falls_back_to_median_data_time():
    """
    Given no explicit t0_par, no user t_0, and no seeds,
    When the final t0_par is resolved,
    Then the median data time anchors the frame (keeps the linear Earth
      extrapolation within the season) instead of a fixed ancient epoch.
    """
    inst, system = _t0_par_fixture()
    times = np.linspace(2458500.0, 2458600.0, 11)
    assert inst._resolve_t0_par_final(system, times) == pytest.approx(
        2458550.0
    )


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


# ---------------------------------------------------------------------------
# _check_data_format must see MMEXOFAST seed start values
# ---------------------------------------------------------------------------


class _SeedOnlyConfigManager(_DummyConfigManager):
    """ConfigManager stub carrying a trajectory only in the seed hints, as in
    the `mmexofast: auto` workflow (user_params names no lens parameter)."""

    def __init__(self, seeds, user_params=None):
        self.user_params = user_params or {}
        self._seeds = seeds

    def seed_start_value(self, path, seed=0):
        return self._seeds.get(path)


def _flux_labelled_as_magnitudes(t0=2458554.89, u0=0.14, tE=18.0, n=600):
    """A light curve in normalized FLUX that a user forgot to declare, so it
    is read as 'magnitudes'.  The file's values get LARGER at peak, so
    load_data's mag->flux conversion turns the peak into the FAINTEST part of
    the curve -- exactly what the check exists to catch.

    Returns the post-conversion arrays load_data would hand _check_data_format
    (which now works entirely in flux, like the rest of the component)."""
    t = np.linspace(t0 - 60, t0 + 60, n)
    tau = (t - t0) / tE
    u = np.sqrt(tau**2 + u0**2)
    file_values = (u**2 + 2.0) / (u * np.sqrt(u**2 + 4.0))
    file_err = np.full(n, 0.001)
    # load_data's magnitude branch, applied to values that are really flux.
    flux = 10.0 ** (-0.4 * file_values)
    err = (np.log(10.0) / 2.5) * flux * file_err
    return t, flux, err, np.zeros((n, 3))


def _run_check(config_manager, caplog):
    inst = MulensInstrument.__new__(MulensInstrument)
    inst.config_manager = config_manager
    t, f, e, xyz = _flux_labelled_as_magnitudes()
    with caplog.at_level(logging.WARNING):
        inst._check_data_format(t, f, e, xyz, 0.0, 0.0, "OGLE-I")
    return caplog.text


def test_check_data_format_uses_mmexofast_seed_start_values(caplog):
    """
    Given flux data mislabelled as magnitudes, and a trajectory known ONLY
      from the MMEXOFAST seed hints (the automated workflow, where the user
      typed no start values at all),
    When _check_data_format runs,
    Then it warns that the data may be in flux units.

    The check used to read cm.user_params alone, so it returned at the very
    first `t0 is None` in precisely the workflow it was most needed in.
    """
    # Arrange
    cm = _SeedOnlyConfigManager(
        {"lens.0.t_0": 2458554.89, "lens.0.u_0": 0.14, "lens.0.t_E": 18.0}
    )

    # Act
    text = _run_check(cm, caplog)

    # Assert
    assert "may be in flux units" in text


def test_check_data_format_user_params_still_win(caplog):
    """
    Given the same mislabelled data with the trajectory in user_params and a
      deliberately wrong seed,
    When _check_data_format runs,
    Then it still warns (the user's values are used, the seed is only a
      fallback).
    """
    # Arrange
    cm = _SeedOnlyConfigManager(
        {"lens.0.t_0": 2400000.0, "lens.0.u_0": 5.0, "lens.0.t_E": 1.0},
        user_params={
            "lens.0.t_0": {"initval": 2458554.89},
            "lens.0.u_0": {"initval": 0.14},
            "lens.0.t_E": {"initval": 18.0},
        },
    )

    # Act
    text = _run_check(cm, caplog)

    # Assert
    assert "may be in flux units" in text


def test_check_data_format_silent_without_any_trajectory(caplog):
    """
    Given neither user params nor seed hints,
    When _check_data_format runs,
    Then it returns silently (no trajectory, nothing to compare).
    """
    # Arrange
    cm = _SeedOnlyConfigManager({})

    # Act
    text = _run_check(cm, caplog)

    # Assert
    assert text == ""


# ---------------------------------------------------------------------------
# Review 3.4: the Op's line of sight is frozen at build time -- keep the
# freeze, but say so when (and only when) ra/dec are actually sampled.
# ---------------------------------------------------------------------------
def _build_ob08092_with_op(sampled_coords):
    """Build the OB08092 example on the MulensModel Op path.

    ``sampled_coords`` gives the microlensing SOURCE star a positive sigma on
    ra/dec, standing in for the mulens + gaia/abs astrometry topology that is
    the only way those coordinates ever get sampled (a microlensing-only
    source star is auto-pinned by Star.register_parameters).
    """
    import os

    import yaml

    from exozippy.system import System

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    exdir = os.path.join(root, "examples", "ob08092")
    if not os.path.exists(os.path.join(exdir, "ob08092.yaml")):
        pytest.skip("ob08092 example not present")
    with open(os.path.join(exdir, "ob08092.yaml")) as fh:
        cfg = yaml.safe_load(fh)
    with open(os.path.join(exdir, "ob08092.params.yaml")) as fh:
        user_params = yaml.safe_load(fh)

    cfg["lens"][0]["use_op"] = True  # the Op path is what bakes the coords
    if sampled_coords:
        user_params["star.Source.ra"] = {
            "initval": 266.872583333,
            "sigma": 0.5,
        }
        user_params["star.Source.dec"] = {
            "initval": -34.7265555556,
            "sigma": 0.5,
        }

    cwd = os.getcwd()
    try:
        os.chdir(exdir)  # config data paths are relative
        system = System(cfg, user_params)
        system.prepare()
        system.build_model()
    finally:
        os.chdir(cwd)
    return system


def test_op_coords_warn_when_ra_dec_are_sampled(caplog):
    """
    Given: a microlensing topology whose SOURCE star has sampled ra/dec
    When: the model is built on the MulensModel Op path
    Then: both the lens and the galactic model warn that the line of sight is
          frozen, quoting the frozen coordinates

    The freeze itself is correct and deliberate (the parallax projection moves
    ~1e-5 per arcsec of coordinate error); what was missing was saying so.
    """
    # Arrange / Act
    with caplog.at_level(logging.WARNING):
        system = _build_ob08092_with_op(sampled_coords=True)
    text = caplog.text

    # Assert
    assert "[lens]" in text and "FROZEN at the start value" in text
    assert "[galacticmodel]" in text and "FROZEN at the start value" in text
    assert "266.872583" in text  # the START value, not a prior draw
    # and the configuration under test really does sample them
    source_ndx = int(system.lens.source_map[0])
    assert bool(np.atleast_1d(system.star.ra.is_sampled)[source_ndx])


def test_op_coords_do_not_warn_when_ra_dec_are_pinned(caplog):
    """
    Given: an ordinary microlensing fit, whose ra/dec are pinned
    When: the model is built on the MulensModel Op path
    Then: nothing is said -- a warning on every microlensing fit is a warning
          nobody reads
    """
    # Arrange / Act
    with caplog.at_level(logging.WARNING):
        system = _build_ob08092_with_op(sampled_coords=False)
    text = caplog.text

    # Assert
    source_ndx = int(system.lens.source_map[0])
    assert not np.atleast_1d(system.star.ra.is_sampled)[source_ndx]
    assert not np.atleast_1d(system.star.dec.is_sampled)[source_ndx]
    assert "FROZEN at the start value" not in text


def test_op_coords_are_the_start_value_not_a_prior_draw():
    """
    Given: a microlensing topology with SAMPLED source ra/dec
    When: the coordinates baked into the Op are read
    Then: they are the resolved START value

    The old code read ``float(star.ra.value[i].eval())``.  A sampled element's
    value node is a random variable, so ``.eval()`` draws from its prior: the
    coordinates were not frozen at the start value at all, they were frozen at
    an arbitrary draw (measured 0.36 deg away for this configuration).
    """
    # Arrange
    system = _build_ob08092_with_op(sampled_coords=True)
    source_ndx = int(system.lens.source_map[0])

    # Act
    ra_deg, dec_deg = system.lens._frozen_op_coords_deg(system, source_ndx)
    drawn_deg = float(system.star.ra.value[source_ndx].eval()) * 180.0 / np.pi

    # Assert
    assert ra_deg == pytest.approx(266.872583333, abs=1e-6)
    assert dec_deg == pytest.approx(-34.7265555556, abs=1e-6)
    # and the draw really is somewhere else, so this is not a vacuous test
    assert abs(drawn_deg - ra_deg) > 1e-3
