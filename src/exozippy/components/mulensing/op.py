import pytensor.tensor as pt
from pytensor.graph import Apply, Op
from pytensor.gradient import DisconnectedType
import numpy as np
import MulensModel as mm
from astropy.coordinates import SkyCoord, get_body_barycentric, solar_system_ephemeris
from astropy.time import Time
import astropy.units as u

solar_system_ephemeris.set('jpl')

# Cache Earth positions keyed on sorted unique times (immutable between MCMC steps).
_earth_pos_cache = {}

def _earth_xyz_at(times_np):
    """Return (N,3) AU array of Earth barycentric positions, cached by times."""
    key = hash(np.asarray(times_np).tobytes())
    if key not in _earth_pos_cache:
        _earth_pos_cache[key] = np.array([
            get_body_barycentric('earth', Time(t, format='jd', scale='tdb')).xyz.to('au').value
            for t in times_np
        ])
    return _earth_pos_cache[key]


def _get_sat_coord(obs_pos_abs_np, times_np, cache):
    """Build and cache a geocentric SkyCoord from absolute observer positions.

    MulensModel's _get_delta_satellite computes -dot(satellite_skycoord, north),
    so satellite_skycoord must be GEOCENTRIC (obs_abs - earth_actual), not absolute.
    """
    obs_pos_2d = np.atleast_2d(obs_pos_abs_np)
    key = (obs_pos_2d.shape, hash(obs_pos_2d.tobytes()))
    if key not in cache:
        earth_xyz = _earth_xyz_at(times_np)
        geocentric = obs_pos_2d - earth_xyz
        cache[key] = SkyCoord(
            x=geocentric[:, 0] * u.au,
            y=geocentric[:, 1] * u.au,
            z=geocentric[:, 2] * u.au,
            representation_type='cartesian'
        )
    return cache[key]


def _base_mm_params(p):
    """Sanitized parameters shared by all models: [t_0, u_0, t_E, pi_E_N, pi_E_E]."""
    return {
        't_0': float(p[0]),
        'u_0': float(np.sign(float(p[1])) * max(abs(float(p[1])), 1e-9)),
        't_E': float(max(float(p[2]), 1e-4)),
        'pi_E_N': float(p[3]),
        'pi_E_E': float(p[4]),
    }


def _safe_rho(value):
    """rho <= 0 is unphysical and breaks finite-source methods; floor it."""
    return float(max(float(value), 1e-9))


def _build_pspl_model(p, coords, mag_method, use_rho=False):
    """Construct a MulensModel for PSPL (+ optional finite source).

    Param vector: [t_0, u_0, t_E, pi_E_N, pi_E_E] + optional [rho]
    """
    mm_params = _base_mm_params(p)
    if use_rho:
        mm_params['rho'] = _safe_rho(p[5])

    model = mm.Model(parameters=mm_params, coords=coords)
    # We supply geocentric satellite_skycoord, so satellite=True covers all
    # parallax.  Annual Earth parallax (earth_orbital) needs t_0_par which is
    # not in the Op param vector and would fail; its contribution is also
    # already embedded in the geocentric conversion (satellite - earth_actual).
    model.parallax(earth_orbital=False, satellite=True, topocentric=False)

    if isinstance(mag_method, list):
        model.set_magnification_methods(mag_method)
    elif mag_method == "auto_vbbl":
        if use_rho:
            t_0, t_E = mm_params['t_0'], mm_params['t_E']
            window = 5.0 * mm_params['rho'] * t_E
            model.set_magnification_methods(
                [t_0 - window, 'finite_source_LD_Lee09', t_0 + window])
        else:
            model.set_magnification_methods([0.0, 'point_source'])
    else:
        model.set_magnification_methods([0.0, mag_method])
    return model


def _build_binary_model(p, coords, mag_method, use_rho=False):
    """Construct a MulensModel for a binary lens.

    Param vector: [t_0, u_0, t_E, pi_E_N, pi_E_E] + optional [rho] + [s, q, alpha_deg]
    """
    mm_params = _base_mm_params(p)
    idx = 5
    if use_rho:
        mm_params['rho'] = _safe_rho(p[idx])
        idx += 1
    mm_params['s'] = float(max(float(p[idx]), 1e-6))
    mm_params['q'] = float(np.clip(float(p[idx + 1]), 1e-9, 1.0))
    mm_params['alpha'] = float(p[idx + 2])

    model = mm.Model(parameters=mm_params, coords=coords)
    model.parallax(earth_orbital=False, satellite=True, topocentric=False)

    if isinstance(mag_method, list):
        model.set_magnification_methods(mag_method)
    elif mag_method == "auto_vbbl":
        # Keyed on the finite_source config flag, not the runtime rho value.
        method = 'VBM' if use_rho else 'VBBL'
        model.set_magnification_methods([0.0, method])
    else:
        model.set_magnification_methods([0.0, mag_method])
    return model


# ---------------------------------------------------------------------------
# Magnification Ops
# ---------------------------------------------------------------------------

class _MagOpBase(Op):
    """Shared machinery for MulensModel-backed magnification Ops.

    Subclasses set `_builder` to the model-construction function, which fixes
    the expected param-vector layout.
    """
    itypes = [pt.dvector, pt.dvector, pt.dmatrix]
    otypes = [pt.dvector]
    _builder = None

    def __init__(self, coords, mag_method, use_rho=False):
        self.coords = coords
        self.mag_method = mag_method
        self.use_rho = use_rho
        self._coord_cache = {}

    def infer_shape(self, fgraph, node, input_shapes):
        return [input_shapes[1]]

    def perform(self, node, inputs, outputs):
        p, times_np, obs_pos_np = inputs
        model = self._builder(p, self.coords, self.mag_method, self.use_rho)
        sat_coord = _get_sat_coord(obs_pos_np, times_np, self._coord_cache)
        outputs[0][0] = np.asarray(model.get_magnification(times_np,
                                                            satellite_skycoord=sat_coord))

    def pullback(self, inputs, outputs, cotangents):
        p, times, obs_pos = inputs
        g = cotangents[0]
        grad_op = _MagGradOp(type(self)._builder, self.coords,
                             self.mag_method, self.use_rho)
        return [
            grad_op(p, times, obs_pos, g),
            DisconnectedType()(),
            DisconnectedType()()
        ]

    # Backward compatibility with PyTensor < 3 which calls grad() instead of pullback()
    def grad(self, inputs, gradients):
        return self.pullback(inputs, [], gradients)

    def connection_pattern(self, node):
        return [[True], [False], [False]]


class MulensMagOp(_MagOpBase):
    """PyTensor Op wrapping MulensModel for PSPL (+ optional finite source).

    Param vector: [t_0, u_0, t_E, pi_E_N, pi_E_E] + optional [rho]
    """
    _builder = staticmethod(_build_pspl_model)

    def __init__(self, coords, mag_method="point_source", use_rho=False):
        super().__init__(coords, mag_method, use_rho)


class BinaryLensMagOp(_MagOpBase):
    """PyTensor Op wrapping MulensModel for binary lens (+ optional finite source).

    Param vector: [t_0, u_0, t_E, pi_E_N, pi_E_E] + optional [rho] + [s, q, alpha_deg]
    """
    _builder = staticmethod(_build_binary_model)

    def __init__(self, coords, mag_method="auto_vbbl", use_rho=False):
        super().__init__(coords, mag_method, use_rho)


class _MagGradOp(Op):
    """Numerical (forward-difference) gradient for a magnification Op."""
    itypes = [pt.dvector, pt.dvector, pt.dmatrix, pt.dvector]
    otypes = [pt.dvector]

    def __init__(self, builder, coords, mag_method, use_rho=False, eps=1e-6):
        self._builder = builder
        self.coords = coords
        self.mag_method = mag_method
        self.use_rho = use_rho
        self.eps = eps
        self._coord_cache = {}

    def infer_shape(self, fgraph, node, input_shapes):
        return [input_shapes[0]]

    def _calc(self, p, times_1d, sat_coord):
        model = self._builder(p, self.coords, self.mag_method, self.use_rho)
        return model.get_magnification(times_1d, satellite_skycoord=sat_coord)

    def perform(self, node, inputs, outputs):
        params, times_np, obs_pos_np, g = inputs
        out = np.zeros(params.shape, dtype=params.dtype)
        times_1d = np.atleast_1d(times_np)
        sat_coord = _get_sat_coord(obs_pos_np, times_1d, self._coord_cache)
        f_x = self._calc(params, times_1d, sat_coord)
        for i in range(len(params)):
            p_plus = params.copy()
            p_plus[i] += self.eps
            out[i] = np.sum(g * (self._calc(p_plus, times_1d, sat_coord) - f_x) / self.eps)
        outputs[0][0] = out

    def pullback(self, inputs, outputs, cotangents):
        return [DisconnectedType()(), DisconnectedType()(),
                DisconnectedType()(), cotangents[0]]

    # Backward compatibility with PyTensor < 3 which calls grad() instead of pullback()
    def grad(self, inputs, gradients):
        return self.pullback(inputs, [], gradients)

    def connection_pattern(self, node):
        return [[True], [False], [False], [True]]
