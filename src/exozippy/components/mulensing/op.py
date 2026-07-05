import pytensor.tensor as pt
from pytensor.graph import Apply, Op
from pytensor.gradient import DisconnectedType
import numpy as np
import MulensModel as mm
import VBMicrolensing
from astropy.coordinates import SkyCoord, get_body_barycentric, solar_system_ephemeris
from astropy.time import Time
import astropy.units as u

solar_system_ephemeris.set('jpl')

# Cache Earth positions keyed on sorted unique times (immutable between MCMC steps).
_earth_pos_cache = {}

def _earth_xyz_at(times_np):
    """Return (N,3) AU array of Earth barycentric positions, cached by times."""
    times_np = np.asarray(times_np)
    key = hash(times_np.tobytes())
    if key not in _earth_pos_cache:
        t = Time(times_np, format='jd', scale='tdb')
        _earth_pos_cache[key] = get_body_barycentric('earth', t).xyz.to('au').value.T
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

    Param vector: [t_0, u_0, t_E, pi_E_N, pi_E_E] + optional [rho] + optional [u1]
    Extra trailing elements (u1) are ignored by the builder; LD is applied in perform().
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
            # Yoo04 (interpolated B0/B1 tables, ~1e-4 accuracy, exact direct
            # fallback outside the table range) instead of Lee09 (direct 2-D
            # integration, the slowest point-lens method in MulensModel).
            model.set_magnification_methods(
                [t_0 - window, 'finite_source_LD_Yoo04', t_0 + window])
        else:
            model.set_magnification_methods([0.0, 'point_source'])
    else:
        model.set_magnification_methods([0.0, mag_method])
    return model


def _build_binary_model(p, coords, mag_method, use_rho=False):
    """Construct a MulensModel for a binary lens.

    Param vector: [t_0, u_0, t_E, pi_E_N, pi_E_E] + optional [rho] + [s, q, alpha_deg]
    Extra trailing elements (u1) are ignored by the builder; LD is applied in perform().
    """
    mm_params = _base_mm_params(p)
    idx = 5
    if use_rho:
        mm_params['rho'] = _safe_rho(p[idx])
        idx += 1
    mm_params['s'] = float(max(float(p[idx]), 1e-6))
    mm_params['q'] = float(np.clip(float(p[idx + 1]), 1e-9, 100.0))
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

    When `bandpass` is not None, u1 is expected as the last element of the param
    vector. It is applied via set_limb_coeff_u before magnification is computed.
    """
    itypes = [pt.dvector, pt.dvector, pt.dmatrix]
    otypes = [pt.dvector]
    _builder = None

    def __init__(self, coords, mag_method, use_rho=False, bandpass=None):
        self.coords = coords
        self.mag_method = mag_method
        self.use_rho = use_rho
        self.bandpass = bandpass  # None = no LD; str = apply u1 LD for this bandpass
        self._coord_cache = {}

    def infer_shape(self, fgraph, node, input_shapes):
        return [input_shapes[1]]

    def perform(self, node, inputs, outputs):
        p, times_np, obs_pos_np = inputs
        try:
            model = self._builder(p, self.coords, self.mag_method, self.use_rho)
            sat_coord = _get_sat_coord(obs_pos_np, times_np, self._coord_cache)
            with np.errstate(invalid='ignore', divide='ignore'):
                if self.bandpass is not None:
                    model.set_limb_coeff_u(self.bandpass, float(p[-1]))
                    A = model.get_magnification(times_np, satellite_skycoord=sat_coord,
                                                bandpass=self.bandpass)
                else:
                    A = model.get_magnification(times_np, satellite_skycoord=sat_coord)
        except (ValueError, RuntimeError):
            # Invalid parameter combination (e.g. NaN source position from extreme
            # parallax values during sampler exploration). Return NaN so the
            # likelihood evaluates to -inf and the sampler rejects the proposal.
            A = np.full(len(times_np), np.nan)
        outputs[0][0] = np.asarray(A)

    def pullback(self, inputs, outputs, cotangents):
        p, times, obs_pos = inputs
        g = cotangents[0]
        grad_op = _MagGradOp(type(self)._builder, self.coords,
                             self.mag_method, self.use_rho, self.bandpass)
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

    Param vector: [t_0, u_0, t_E, pi_E_N, pi_E_E] + optional [rho] + optional [u1]
    """
    _builder = staticmethod(_build_pspl_model)

    def __init__(self, coords, mag_method="point_source", use_rho=False, bandpass=None):
        super().__init__(coords, mag_method, use_rho, bandpass)


class BinaryLensMagOp(_MagOpBase):
    """PyTensor Op wrapping MulensModel for binary lens (+ optional finite source).

    Param vector: [t_0, u_0, t_E, pi_E_N, pi_E_E] + optional [rho] + [s, q, alpha_deg]
    + optional [u1]
    """
    _builder = staticmethod(_build_binary_model)

    def __init__(self, coords, mag_method="auto_vbbl", use_rho=False, bandpass=None):
        super().__init__(coords, mag_method, use_rho, bandpass)


class VBMDirectMagOp(Op):
    """Direct-VBMicrolensing magnification Op for binary and N-body lenses.

    Bypasses MulensModel in the sampler hot path: everything MulensModel
    re-derives per call (SkyCoord parsing, Model construction, per-point
    python dispatch) is either precomputed once here or replaced by a
    vectorized numpy trajectory plus one VBM C++ call per epoch.

    Parallax convention mirrors the MulensModel Ops exactly (validated by
    tests/test_vbm_direct_vs_mulensmodel.py): observer positions arrive as
    absolute barycentric AU, are converted to geocentric (obs - earth_actual),
    projected on sky-plane north/east with a minus sign
    (MulensModel Trajectory._get_delta_satellite), and applied as
    delta_tau = +dN*pi_E_N + dE*pi_E_E, delta_beta = -dN*pi_E_E + dE*pi_E_N
    (Trajectory._project_delta).

    Magnification method selection is not needed: VBM's BinaryMag2/MultiMag2
    perform the quadrupole safety test internally in C++ and short-circuit to
    point-source when safe, which benchmarks faster than any python-level
    bracketing (see hpc_optimization.txt).

    Param vector: [t_0, u_0, t_E, pi_E_N, pi_E_E] + optional [rho]
                  + per companion j: [s_j, q_j, alpha_j_deg]
                  + optional [u1]

    Companion geometry convention (reduces exactly to the MulensModel /
    VBMicrolensing binary convention for one companion): alpha_j is the
    counterclockwise angle from the trajectory frame to the primary->companion_j
    axis, s_j the projected separation; all lengths in Einstein radii of the
    TOTAL lens mass. Internally the source moves in the trajectory frame at
    (-tau, -u) and companion j sits at s_j*(cos alpha_j, -sin alpha_j) from
    the primary, with the origin shifted to the lens center of mass.
    """
    itypes = [pt.dvector, pt.dvector, pt.dmatrix]
    otypes = [pt.dvector]

    def __init__(self, coords, n_companions=1, use_rho=False, bandpass=None,
                 accuracy=1e-3, relative_accuracy=0.0):
        # coords: "<ra>d <dec>d" string — same format the MulensModel Ops take.
        ra_deg, dec_deg = [float(v.rstrip('d')) for v in str(coords).split()]
        ra = np.radians(ra_deg)
        dec = np.radians(dec_deg)
        # Sky-plane projections, mirroring MulensModel Coordinates:
        # east = normalize(z x direction), north = direction x east.
        direction = np.array([np.cos(dec) * np.cos(ra),
                              np.cos(dec) * np.sin(ra),
                              np.sin(dec)])
        east = np.cross([0.0, 0.0, 1.0], direction)
        east /= np.linalg.norm(east)
        self._east = east
        self._north = np.cross(direction, east)

        self.n_companions = int(n_companions)
        self.use_rho = use_rho
        self.bandpass = bandpass  # None = no LD; str = u1 is last param element
        self._accuracy = float(accuracy)
        self._relative_accuracy = float(relative_accuracy)
        # One VBM instance per Op; PTDE fork workers each inherit a private
        # copy-on-write copy, so per-instance scratch state is never shared.
        self._vbm = VBMicrolensing.VBMicrolensing()
        self._vbm.Tol = self._accuracy
        self._vbm.RelTol = self._relative_accuracy
        if self.n_companions >= 2:
            # Multipoly beats the Nopoly default for 3 lenses; Nopoly wins
            # for 4+ (VBM docs, Bozza+2025 A&A 694, 219).  Must precede
            # SetLensGeometry, so it is fixed here.
            if self.n_companions == 2:
                self._vbm.SetMethod(VBMicrolensing.VBMicrolensing.Method.Multipoly)
            else:
                self._vbm.SetMethod(VBMicrolensing.VBMicrolensing.Method.Nopoly)
        self._delta_cache = {}

    def infer_shape(self, fgraph, node, input_shapes):
        return [input_shapes[1]]

    def _deltas(self, times_np, obs_pos_np):
        """Cached parallax offsets (delta_N, delta_E) for a (times, obs) pair.

        Independent of the sampled parameters: depend only on epochs, event
        coordinates, and the observer ephemeris, so they are computed once and
        reused for every proposal.
        """
        obs = np.atleast_2d(obs_pos_np)
        key = (hash(times_np.tobytes()), obs.shape, hash(obs.tobytes()))
        if key not in self._delta_cache:
            geo = obs - _earth_xyz_at(times_np)
            self._delta_cache[key] = (-geo @ self._north, -geo @ self._east)
        return self._delta_cache[key]

    def _magnify(self, companions, x, y, rho, u1):
        """One VBM call per epoch on trajectory (x, y); binary or N-lens.

        Far-field guard: all caustics lie within ~R_inf of the center of
        mass, so a source center farther than R_inf + 2*rho is point-source
        safe to well below Tol and is dispatched to BinaryMag0/MultiMag0.
        This is not just an optimization — VBMicrolensing <= 5.5 hardcodes
        safedist = 10 for q >= 0.01, so BinaryMag2's internal point-source
        shortcut can NEVER fire for rho > sqrt(10)/2 ~ 1.6 and each call
        costs ~0.1 s even with the source thousands of Einstein radii away
        (measured on the DC2018_128 eval_timeout rejections: 870 epochs of
        A=1 took 56 s). Fixed in our local VBMicrolensing copy, but guarded
        here too so PyPI wheels behave and the N-lens path is covered.
        """
        vbm = self._vbm
        vbm.a1 = 0.0 if u1 is None else u1
        if self.n_companions == 1:
            s, q, _ = companions[0]
            r_inf = s + 1.0 / s + 2.0
            far = (x * x + y * y) > (r_inf + 2.0 * rho) ** 2
            mag2, mag0 = vbm.BinaryMag2, vbm.BinaryMag0
            return np.array([
                mag0(s, q, xi, yi) if isfar else mag2(s, q, xi, yi, rho)
                for xi, yi, isfar in zip(x.tolist(), y.tolist(), far.tolist())
            ])

        # Lens positions in the trajectory frame, origin at center of mass;
        # mass fractions sum to 1 so VBM's unit-mass Einstein radius equals
        # our total-mass Einstein radius.
        q_tot = sum(q for (_, q, _) in companions)
        m = np.empty(self.n_companions + 1)
        pos = np.zeros((self.n_companions + 1, 2))
        m[0] = 1.0 / (1.0 + q_tot)
        for j, (s, q, alpha_rad) in enumerate(companions):
            m[j + 1] = q * m[0]
            pos[j + 1] = (s * np.cos(alpha_rad), -s * np.sin(alpha_rad))
        pos -= m @ pos
        vbm.SetLensGeometry(np.column_stack([pos, m]).ravel().tolist())
        r_inf = max(s + 1.0 / s for (s, _, _) in companions) + 2.0
        far = (x * x + y * y) > (r_inf + 2.0 * rho) ** 2
        mag2, mag0 = vbm.MultiMag2, vbm.MultiMag0
        return np.array([
            mag0(xi, yi) if isfar else mag2(xi, yi, rho)
            for xi, yi, isfar in zip(x.tolist(), y.tolist(), far.tolist())
        ])

    def perform(self, node, inputs, outputs):
        p, times_np, obs_pos_np = inputs
        try:
            A = self._compute(p, times_np, obs_pos_np)
        except (ValueError, RuntimeError):
            A = np.full(len(times_np), np.nan)
        outputs[0][0] = np.asarray(A, dtype=np.float64)

    def _compute(self, p, times_np, obs_pos_np):
        base = _base_mm_params(p)
        idx = 5
        rho = 0.0
        if self.use_rho:
            rho = _safe_rho(p[idx])
            idx += 1
        companions = []
        for _ in range(self.n_companions):
            companions.append((
                float(max(float(p[idx]), 1e-6)),
                float(np.clip(float(p[idx + 1]), 1e-9, 100.0)),
                float(np.radians(float(p[idx + 2]))),
            ))
            idx += 3
        u1 = float(p[-1]) if self.bandpass is not None else None

        if not np.all(np.isfinite(p)):
            return np.full(len(times_np), np.nan)

        dN, dE = self._deltas(times_np, obs_pos_np)
        tau = ((times_np - base['t_0']) / base['t_E']
               + dN * base['pi_E_N'] + dE * base['pi_E_E'])
        u = base['u_0'] - dN * base['pi_E_E'] + dE * base['pi_E_N']

        if self.n_companions == 1:
            # Rotate into the lens-axis frame (MulensModel Trajectory._get_xy).
            alpha_rad = companions[0][2]
            ca, sa = np.cos(alpha_rad), np.sin(alpha_rad)
            x = -tau * ca + u * sa
            y = -tau * sa - u * ca
        else:
            # Trajectory frame: same configuration with the rotation applied to
            # the lens positions instead (global rotations leave A invariant).
            x = -tau
            y = -u

        with np.errstate(invalid='ignore', divide='ignore'):
            return self._magnify(companions, x, y, rho, u1)

    def pullback(self, inputs, outputs, cotangents):
        # Deliberately loud: this Op is only reachable from non-gradient
        # samplers (see Lens.sampler_requirements). A gradient-based sampler
        # would otherwise silently burn N_params+1 evals per step through
        # numerical differences.
        raise NotImplementedError(
            "VBMDirectMagOp has no gradient; use the PTDE (or another "
            "gradient-free) sampler for binary/finite-source microlensing."
        )

    # Backward compatibility with PyTensor < 3 which calls grad() instead of pullback()
    def grad(self, inputs, gradients):
        return self.pullback(inputs, [], gradients)

    def connection_pattern(self, node):
        return [[True], [False], [False]]


class _MagGradOp(Op):
    """Numerical (forward-difference) gradient for a magnification Op."""
    itypes = [pt.dvector, pt.dvector, pt.dmatrix, pt.dvector]
    otypes = [pt.dvector]

    def __init__(self, builder, coords, mag_method, use_rho=False, bandpass=None, eps=1e-6):
        self._builder = builder
        self.coords = coords
        self.mag_method = mag_method
        self.use_rho = use_rho
        self.bandpass = bandpass
        self.eps = eps
        self._coord_cache = {}

    def infer_shape(self, fgraph, node, input_shapes):
        return [input_shapes[0]]

    def _calc(self, p, times_1d, sat_coord):
        try:
            model = self._builder(p, self.coords, self.mag_method, self.use_rho)
            with np.errstate(invalid='ignore', divide='ignore'):
                if self.bandpass is not None:
                    model.set_limb_coeff_u(self.bandpass, float(p[-1]))
                    return model.get_magnification(times_1d, satellite_skycoord=sat_coord,
                                                   bandpass=self.bandpass)
                return model.get_magnification(times_1d, satellite_skycoord=sat_coord)
        except (ValueError, RuntimeError):
            return np.full(len(times_1d), np.nan)

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
