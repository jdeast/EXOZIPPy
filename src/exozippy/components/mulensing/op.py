import functools
import warnings

import astropy.units as u
import MulensModel as mm
import numpy as np
import pytensor.tensor as pt
import VBMicrolensing
from astropy.coordinates import SkyCoord
from pytensor.graph import Apply, Op

from exozippy.compat import patch_mulensmodel_method_order
from exozippy.skyframe import parallax_factors

from .physics import (
    _MM_NAN_ADVICE,
    RHO_FLOOR,
    S_FLOOR,
    T_E_FLOOR,
    clip_q_value,
    floor_u_0_value,
    require_mm_number,
)

# MulensModel dispatches magnification methods in PYTHONHASHSEED order, and
# the VBMicrolensing backends are not order-independent, so identical inputs
# give different answers in different processes.  This is the exozippy module
# that owns the MulensModel import, so patch it here, before any Op can build
# an mm.Model.  See exozippy/compat/mulensmodel_method_order.py.
patch_mulensmodel_method_order()


def _clear_mm_satellite_cache():
    """Drop MulensModel's class-level satellite-delta cache before each call.

    Trajectory._get_delta_satellite_results is keyed on (ra, dec, times)
    ONLY -- it ignores the satellite positions themselves -- so two
    evaluations sharing a time grid but carrying different observer
    deviations (e.g. ground and satellite model curves on one plot grid)
    would silently reuse the first observer's parallax deltas.  Clearing is
    cheap next to the mm.Model rebuild this path already pays per call; the
    speed-critical binary path (VBMDirectMagOp) has its own correctly-keyed
    cache and never enters MulensModel.
    """
    mm.Trajectory._get_delta_satellite_results.clear()


def _dev_skycoord(obs_pos_np, cache):
    """Build and cache a SkyCoord from Skowron+2011 geocentric deviations.

    ``obs_pos_np`` are the observer's deviations from the linear Earth
    trajectory anchored at t0_par (MulensInstrument._abs_to_delta) -- the
    same array the symbolic path consumes.  Fed to MulensModel as
    satellite_skycoord with parallax(satellite=True, earth_orbital=False):
    _get_delta_satellite computes -dot(satellite_skycoord, north/east),
    which on these deviations carries ALL parallax (annual + satellite),
    exactly matching Lens.get_magnification.
    """
    obs_pos_2d = np.atleast_2d(obs_pos_np)
    # Keyed on the BYTES, not on hash(bytes): a 64-bit siphash can collide,
    # and this cache's whole reason to exist is telling apart two deviation
    # arrays of the SAME shape (ground and satellite over one plot grid), so a
    # collision would hand the second observer the first one's parallax
    # deltas.  Astronomically unlikely, and free to rule out (review 2.6.4);
    # the dict hashes the bytes for us and then compares them on a hit.
    key = (obs_pos_2d.shape, obs_pos_2d.tobytes())
    if key not in cache:
        cache[key] = SkyCoord(
            x=obs_pos_2d[:, 0] * u.au,
            y=obs_pos_2d[:, 1] * u.au,
            z=obs_pos_2d[:, 2] * u.au,
            representation_type="cartesian",
        )
    return cache[key]


_BASE_LABELS = (
    "lens.t_0",
    "lens.u_0",
    "lens.t_E",
    "lens.pi_E_N",
    "lens.pi_E_E",
)


def _base_mm_params(p):
    """Range-limited parameters shared by all models:
    [t_0, u_0, t_E, pi_E_N, pi_E_E].

    ``require_mm_number`` raises on a NaN instead of handing it to
    MulensModel, which used to report whatever generic failure the NaN
    happened to cause several frames away.  The caller's
    ``except (ValueError, RuntimeError)`` turns it into the same all-NaN
    answer as before, now under a message that names the parameter.  Finite
    values, in range or out, are untouched.
    """
    t_0, u_0, t_E, pi_E_N, pi_E_E = (
        require_mm_number(p[i], _BASE_LABELS[i]) for i in range(5)
    )
    return {
        "t_0": t_0,
        # Same floor, same expression as the symbolic path
        # (Lens._get_safe_mm_params): both go through physics, so the two
        # backends cannot disagree about where the model is defined.  This
        # used to be a hard-coded 1e-9 against physics.U_0_FLOOR = 1e-6, so a
        # fit visiting 1e-9 <= |u_0| < 1e-6 got a different answer depending
        # on which backend it was on.
        "u_0": floor_u_0_value(u_0),
        "t_E": float(max(t_E, T_E_FLOOR)),
        "pi_E_N": pi_E_N,
        "pi_E_E": pi_E_E,
    }


def _safe_rho(value):
    """Floor the source radius at physics.RHO_FLOOR.

    One number, shared with the flux bootstrap's own copy of this clip, for
    the same reason U_0_FLOOR is: two literals drift.
    """
    return float(max(float(value), RHO_FLOOR))


def _build_pspl_model(p, coords, mag_method, use_rho=False):
    """Construct a MulensModel for PSPL (+ optional finite source).

    Param vector: [t_0, u_0, t_E, pi_E_N, pi_E_E] + optional [rho] + optional [u1]
    Extra trailing elements (u1) are ignored by the builder; LD is applied in perform().
    """
    mm_params = _base_mm_params(p)
    if use_rho:
        mm_params["rho"] = _safe_rho(p[5])

    model = mm.Model(parameters=mm_params, coords=coords)
    # The satellite channel is fed the Skowron+2011 geocentric deviations
    # (see _dev_skycoord), which already contain the annual Earth term, so
    # satellite=True covers ALL parallax.  earth_orbital stays False: it
    # would double-count the annual term (and needs t_0_par, which is not
    # in the Op param vector).
    model.parallax(earth_orbital=False, satellite=True, topocentric=False)

    if isinstance(mag_method, list):
        model.set_magnification_methods(mag_method)
    elif mag_method == "auto_vbbl":
        if use_rho:
            t_0, t_E = mm_params["t_0"], mm_params["t_E"]
            window = 5.0 * mm_params["rho"] * t_E
            # Yoo04 (interpolated B0/B1 tables, ~1e-4 accuracy, exact direct
            # fallback outside the table range) instead of Lee09 (direct 2-D
            # integration, the slowest point-lens method in MulensModel).
            model.set_magnification_methods(
                [t_0 - window, "finite_source_LD_Yoo04", t_0 + window]
            )
        else:
            model.set_magnification_methods([-np.inf, "point_source", np.inf])
    else:
        model.set_magnification_methods([-np.inf, mag_method, np.inf])
    return model


def _build_binary_model(
    p, coords, mag_method, use_rho=False, orbital_motion=False, t_0_kep=None
):
    """Construct a MulensModel for a binary lens.

    Param vector: [t_0, u_0, t_E, pi_E_N, pi_E_E] + optional [rho] + [s, q, alpha_deg]
    + optional [ds_dt, dalpha_dt] (with ``orbital_motion=True``; MulensModel's
    LINEAR lens-motion branch, both rates per year and dalpha_dt in deg/yr,
    anchored at the ``t_0_kep`` given here -- EXOZIPPy passes t0_par, C24/5d).
    Extra trailing elements (u1) are ignored by the builder; LD is applied in perform().

    Only the LINEAR branch is ever requested: MulensModel's keplerian lens
    motion contradicts its own linear mode by a sign (conventions.md
    section 6) and is not used as a reference for anything.
    """
    mm_params = _base_mm_params(p)
    idx = 5
    if use_rho:
        mm_params["rho"] = _safe_rho(p[idx])
        idx += 1
    mm_params["s"] = float(max(float(p[idx]), S_FLOOR))
    mm_params["q"] = clip_q_value(p[idx + 1], "lens.q")
    mm_params["alpha"] = float(p[idx + 2])
    if orbital_motion:
        mm_params["ds_dt"] = float(p[idx + 3])
        mm_params["dalpha_dt"] = float(p[idx + 4])
        mm_params["t_0_kep"] = float(t_0_kep)

    model = mm.Model(parameters=mm_params, coords=coords)
    # Same convention as _build_pspl_model: the satellite channel carries
    # all parallax via the Skowron geocentric deviations.
    model.parallax(earth_orbital=False, satellite=True, topocentric=False)

    if isinstance(mag_method, list):
        model.set_magnification_methods(mag_method)
    elif mag_method == "auto_vbbl":
        # Keyed on the finite_source config flag, not the runtime rho value.
        # Point source must NOT ask for VBBL/VBM: MulensModel's _check_methods
        # only allows "point_source"/"point_source_point_lens" when the
        # parameters carry no rho, and otherwise raises ValueError -- which
        # perform() would swallow into an all-NaN (i.e. -inf logp) curve for
        # every proposal.  "point_source" selects
        # BinaryLensPointSourceMagnification, the exact binary point-source
        # solver (it reproduces VBBL at rho -> 0 to machine precision).
        method = "VBM" if use_rho else "point_source"
        model.set_magnification_methods([-np.inf, method, np.inf])
    else:
        model.set_magnification_methods([-np.inf, mag_method, np.inf])
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
        self.bandpass = (
            bandpass  # None = no LD; str = apply u1 LD for this bandpass
        )
        self._coord_cache = {}
        self._warned = False

    def infer_shape(self, node, input_shapes):
        return [input_shapes[1]]

    def perform(self, node, inputs, outputs):
        p, times_np, obs_pos_np = inputs
        try:
            model = self._builder(
                p, self.coords, self.mag_method, self.use_rho
            )
            sat_coord = _dev_skycoord(obs_pos_np, self._coord_cache)
            _clear_mm_satellite_cache()
            with np.errstate(invalid="ignore", divide="ignore"):
                if self.bandpass is not None:
                    model.set_limb_coeff_u(self.bandpass, float(p[-1]))
                    A = model.get_magnification(
                        times_np,
                        satellite_skycoord=sat_coord,
                        bandpass=self.bandpass,
                    )
                else:
                    A = model.get_magnification(
                        times_np, satellite_skycoord=sat_coord
                    )
        except (ValueError, RuntimeError) as exc:
            # Invalid parameter combination (e.g. NaN source position from extreme
            # parallax values during sampler exploration). Return NaN so the
            # likelihood evaluates to -inf and the sampler rejects the proposal.
            #
            # Warn once per Op instance: a *misconfigured* backend raises here
            # on every single proposal, which is indistinguishable from a
            # rejected proposal unless the first one is reported (this is
            # exactly how the point-source-binary "VBBL" bug stayed hidden).
            if not self._warned:
                self._warned = True
                warnings.warn(
                    f"{type(self).__name__}: MulensModel raised "
                    f"{type(exc).__name__}: {exc} -- returning NaN "
                    "magnifications (logp = -inf) for this proposal. "
                    "If this repeats for every proposal the backend is "
                    "misconfigured, not merely exploring bad parameters.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            A = np.full(len(times_np), np.nan)
        outputs[0][0] = np.asarray(A)

    def pullback(self, inputs, outputs, cotangents):
        # Deliberately loud, and deliberately the SAME refusal
        # VBMDirectMagOp.pullback makes (review 2.6.5).  This Op used to hand
        # back a _MagGradOp instead, which silently wired N_params+1 full
        # MulensModel light curves per gradient evaluation -- a forward
        # difference, so a NUTS step paid that cost for a gradient carrying
        # O(eps) error, and nothing anywhere said so.  We do not support
        # gradient-based samplers through the Op path (Lens.sampler_requirements
        # already declares that), so an attempt to build one is a
        # configuration error, not something to serve slowly and inaccurately.
        #
        # This is the MulensModel A/B reference backend, so the stakes are
        # lower than VBMDirectMagOp's -- but the failure mode is worse
        # precisely because it "works": the fit runs, burns the CPU, and
        # returns a posterior nobody has reason to distrust.  The symbolic
        # PSPL path (Lens.get_magnification) is separate and STAYS
        # differentiable; that is what NUTS-compatible microlensing means
        # here.  _MagGradOp is kept as the recipe if this is ever revisited.
        raise NotImplementedError(
            f"{type(self).__name__} has no gradient; use the PTDE (or another "
            "gradient-free) sampler for binary/finite-source microlensing. "
            "Point-source PSPL takes the symbolic path, which is "
            "differentiable."
        )

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

    def __init__(
        self, coords, mag_method="point_source", use_rho=False, bandpass=None
    ):
        super().__init__(coords, mag_method, use_rho, bandpass)


class BinaryLensMagOp(_MagOpBase):
    """PyTensor Op wrapping MulensModel for binary lens (+ optional finite source).

    Param vector: [t_0, u_0, t_E, pi_E_N, pi_E_E] + optional [rho] + [s, q, alpha_deg]
    + optional [ds_dt, dalpha_dt] (``orbital_motion=True``: MulensModel's
    LINEAR lens-motion branch anchored at ``t_0_kep`` = EXOZIPPy's t0_par --
    this is the A/B reference the per-epoch vbm_direct path is pinned
    against) + optional [u1]
    """

    _builder = staticmethod(_build_binary_model)

    def __init__(
        self,
        coords,
        mag_method="auto_vbbl",
        use_rho=False,
        bandpass=None,
        orbital_motion=False,
        t_0_kep=None,
    ):
        super().__init__(coords, mag_method, use_rho, bandpass)
        self.orbital_motion = bool(orbital_motion)
        self.t_0_kep = t_0_kep
        if self.orbital_motion:
            if t_0_kep is None:
                raise ValueError(
                    "BinaryLensMagOp(orbital_motion=True) needs t_0_kep "
                    "(EXOZIPPy passes t0_par; C24/5d -- one anchor for the "
                    "orbital and parallax terms)."
                )
            # functools.partial of a module-level function stays picklable
            # (PTDE spawn / numba object-mode caching), unlike a lambda.
            self._builder = functools.partial(
                _build_binary_model,
                orbital_motion=True,
                t_0_kep=float(t_0_kep),
            )


class VBMDirectMagOp(Op):
    """Direct-VBMicrolensing magnification Op for binary and N-body lenses.

    Bypasses MulensModel in the sampler hot path: everything MulensModel
    re-derives per call (SkyCoord parsing, Model construction, per-point
    python dispatch) is either precomputed once here or replaced by a
    vectorized numpy trajectory plus one VBM C++ call per epoch.

    Parallax convention mirrors the MulensModel Ops exactly (validated by
    tests/test_vbm_direct_vs_mulensmodel.py): observer positions arrive as
    Skowron+2011 geocentric deviations in AU (MulensInstrument._abs_to_delta,
    the same array the symbolic path consumes), are projected on sky-plane
    north/east with a minus sign (MulensModel
    Trajectory._get_delta_satellite), and applied as
    delta_tau = +dN*pi_E_N + dE*pi_E_E, delta_beta = -dN*pi_E_E + dE*pi_E_N
    (Trajectory._project_delta).

    Magnification method selection is not needed: VBM's BinaryMag2/MultiMag2
    perform the quadrupole safety test internally in C++ and short-circuit to
    point-source when safe, which benchmarks faster than any python-level
    bracketing (see hpc_optimization.txt).

    Param vector: [t_0, u_0, t_E, pi_E_N, pi_E_E] + optional [rho]
                  + per companion j: [s_j, q_j, alpha_j_deg]
                  + optional [u1] + optional [u2]

    Companion geometry convention (reduces exactly to the MulensModel /
    VBMicrolensing binary convention for one companion): alpha_j is the
    counterclockwise angle from the trajectory frame to the primary->companion_j
    axis, s_j the projected separation; all lengths in Einstein radii of the
    TOTAL lens mass. Internally the source moves in the trajectory frame at
    (-tau, -u) and companion j sits at s_j*(cos alpha_j, -sin alpha_j) from
    the primary, with the origin shifted to the lens center of mass.

    ``n_companions = 0`` is the SINGLE-lens (ESPL) case, which exists so that
    a finite-source point lens can carry a quadratic limb-darkening law: it is
    the only backend here that can (MulensModel's ``set_limb_coeff_u`` and its
    Yoo04 B0/B1 formalism are linear-only).  It is NOT the default for a
    single lens -- see Lens.get_magnification_op, which keeps MulensModel
    there unless a second LD coefficient is actually in play, because the two
    disagree by up to ~5 mmag in the deep finite-source regime (Yoo04's table
    interpolation) and a silent backend flip would move existing answers.

    ``quadratic_ld`` selects VBM's LDquadratic profile and reads u2 from the
    tail of the param vector.  At u2 = 0 that profile reproduces LDlinear to
    2.2e-16 fractional (measured over a caustic crossing at rho = 0.015), so
    turning it on is a no-op for a linear band -- which is what makes the
    parameter safe to key on the band's declared law rather than on a config
    flag of its own.
    """

    itypes = [pt.dvector, pt.dvector, pt.dmatrix]
    otypes = [pt.dvector]

    def __init__(
        self,
        coords,
        n_companions=1,
        use_rho=False,
        bandpass=None,
        accuracy=1e-3,
        relative_accuracy=0.0,
        quadratic_ld=False,
        orbital_motion=False,
        source_motion=False,
    ):
        # coords: "<ra>d <dec>d" string — same format the MulensModel Ops take.
        ra_deg, dec_deg = [float(v.rstrip("d")) for v in str(coords).split()]
        self._ra = np.radians(ra_deg)
        self._dec = np.radians(dec_deg)
        # One sky basis for the whole codebase (exozippy.skyframe).  This
        # used to be built here as MulensModel Coordinates builds it --
        # east = normalize(z x direction), north = direction x east -- which
        # is the same basis to within 1 ulp (pinned in
        # tests/test_skyframe.py::test_cross_product_construction_agrees).
        # Sharing the definition is what makes "the Op path and the symbolic
        # path see one line of sight" true by construction rather than by two
        # copies happening to agree.

        self.n_companions = int(n_companions)
        self.use_rho = use_rho
        self.bandpass = (
            bandpass  # None = no LD; str = u1 (then u2) at the param tail
        )
        # Lens orbital motion (C24, review 8.6.8 5c): two extra dvector
        # inputs carry the PER-EPOCH companion geometry -- s_t [r_E] and
        # alpha_t [DEG] -- built in the graph by
        # Lens._companion_geometry_series.  The param vector keeps its
        # static s_0/alpha_0 entries (the t0_par values; the series must
        # equal them there), so the layout and _param_labels are unchanged.
        self.orbital_motion = bool(orbital_motion)
        # Source orbital motion -- xallarap (C25, review 8.6.9): two more
        # dvector inputs carry the PER-EPOCH trajectory shift (dtau_t,
        # du_t) built by Lens._source_offset_series, added to (tau, u)
        # after the parallax terms -- the source's own offset enters at
        # exactly the parallax slot.  Input order when both motions are on:
        # [p, times, obs, s_t, alpha_t, dtau_t, du_t].
        self.source_motion = bool(source_motion)
        if self.orbital_motion and self.n_companions != 1:
            raise ValueError(
                "VBMDirectMagOp(orbital_motion=True) supports exactly "
                "one companion (Lens raises earlier; mulensing.md "
                "'3+ lens bodies')."
            )
        if self.orbital_motion or self.source_motion:
            self.itypes = [pt.dvector, pt.dvector, pt.dmatrix]
            if self.orbital_motion:
                self.itypes = self.itypes + [pt.dvector, pt.dvector]
            if self.source_motion:
                self.itypes = self.itypes + [pt.dvector, pt.dvector]
        # u2 only means anything alongside a u1, so a quadratic law without a
        # bandpass is a caller bug, not a silently-uniform source.
        self.quadratic_ld = bool(quadratic_ld) and bandpass is not None
        self._accuracy = float(accuracy)
        self._relative_accuracy = float(relative_accuracy)
        # One VBM instance per Op; PTDE fork workers each inherit a private
        # copy-on-write copy, so per-instance scratch state is never shared.
        self._vbm = self._build_vbm()
        self._delta_cache = {}
        # Warn-once flags, mirroring _MagOpBase._warned.  TWO of them, because
        # the two failure modes have different fixes and must not silence each
        # other: `_warned` covers a non-finite parameter vector reaching
        # _compute (the model is exploring/misconfigured upstream), while
        # `_warned_backend` covers VBMicrolensing itself raising (a
        # SetLensGeometry rejection, a SWIG ValueError, API drift in a new
        # wheel).  One shared flag would let a burst of NaN proposals early on
        # permanently suppress the report that the backend is broken.
        self._warned = False
        self._warned_backend = False

    def _build_vbm(self):
        vbm = VBMicrolensing.VBMicrolensing()
        vbm.Tol = self._accuracy
        vbm.RelTol = self._relative_accuracy
        # Profile is instance state, set once; a1/a2 are per-call and set in
        # _magnify.  Keep it that way: SetLDprofile on every epoch would be
        # the same shape of waste _deltas exists to avoid.
        vbm.SetLDprofile(
            VBMicrolensing.VBMicrolensing.LDquadratic
            if self.quadratic_ld
            else VBMicrolensing.VBMicrolensing.LDlinear
        )
        if self.n_companions >= 2:
            # Multipoly beats the Nopoly default for 3 lenses; Nopoly wins
            # for 4+ (VBM docs, Bozza+2025 A&A 694, 219).  Must precede
            # SetLensGeometry, so it is fixed here.
            if self.n_companions == 2:
                vbm.SetMethod(VBMicrolensing.VBMicrolensing.Method.Multipoly)
            else:
                vbm.SetMethod(VBMicrolensing.VBMicrolensing.Method.Nopoly)
        return vbm

    def __getstate__(self):
        # The VBMicrolensing SWIG object isn't picklable (needed for numba's
        # object-mode dispatch, which pickles perform()'s closure to cache it
        # to disk, and for spawn-based multiprocessing). Drop it and the
        # epoch-keyed cache; both are cheap to rebuild in __setstate__.
        state = self.__dict__.copy()
        del state["_vbm"]
        state["_delta_cache"] = {}
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._vbm = self._build_vbm()

    def infer_shape(self, node, input_shapes):
        return [input_shapes[1]]

    def _deltas(self, obs_pos_np):
        """Cached parallax offsets (delta_N, delta_E) for a deviation array.

        ``obs_pos_np`` are Skowron+2011 geocentric deviations (AU).
        Independent of the sampled parameters: depend only on epochs, event
        coordinates, and the observer ephemeris, so they are computed once and
        reused for every proposal.
        """
        dev = np.atleast_2d(obs_pos_np)
        # Bytes, not hash(bytes) -- see _dev_skycoord (review 2.6.4).
        key = (dev.shape, dev.tobytes())
        if key not in self._delta_cache:
            # MulensModel's delta convention is the NEGATED observer offset,
            # i.e. exactly parallax_factors (see exozippy.skyframe: the same
            # sign relation astrometryinstrument's P_E/P_N carry).
            p_e, p_n = parallax_factors(dev, self._ra, self._dec)
            self._delta_cache[key] = (p_n, p_e)
        return self._delta_cache[key]

    def _magnify(self, companions, x, y, rho, u1, u2=None):
        """One VBM call per epoch on trajectory (x, y); single, binary or N-lens.

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
        if self.quadratic_ld:
            vbm.a2 = 0.0 if u2 is None else u2

        if self.n_companions == 0:
            # Single lens.  u is rotation-invariant, so the trajectory frame
            # needs no alpha and (x, y) may arrive unrotated.
            u = np.sqrt(x * x + y * y)
            if not self.use_rho:
                # Paczynski in closed form -- cheaper and more accurate than a
                # VBM call, and the only reachable point-source single-lens Op
                # case is a forced `use_op: true` (the symbolic path otherwise
                # owns it, and stays differentiable).
                u2sq = u * u
                return (u2sq + 2.0) / np.sqrt(u2sq * (u2sq + 4.0))
            # ESPLMag2 is table-backed and internally short-circuits to the
            # point source far from the lens, so this needs no far-field guard
            # of the kind the binary branch below does (VBM's hardcoded
            # safedist bug is in BinaryMag2, not here).
            return np.array(
                [vbm.ESPLMag2(float(ui), rho) for ui in u.tolist()]
            )

        if self.n_companions == 1:
            s, q, _ = companions[0]
            # One per-epoch layout whether s is the static scalar or the
            # orbital-motion series (review 8.6.8 5c: the fast path merges
            # into the general layout; broadcast_to is a view, so the
            # static case pays nothing and computes identically).
            s_arr = np.broadcast_to(np.asarray(s, dtype=float), np.shape(x))
            mag2, mag0 = vbm.BinaryMag2, vbm.BinaryMag0
            if not self.use_rho:
                # Point source (user's finite_source: False): Mag2's
                # finite-source integration returns NaN for an exactly-zero
                # source radius (confirmed against VBM directly), so always
                # use the point-source call rather than gating on distance.
                # Gated on the user's config flag, not the numeric value of
                # rho, since rho is otherwise a derived/sampled quantity.
                return np.array(
                    [
                        mag0(si, q, xi, yi)
                        for si, xi, yi in zip(
                            s_arr.tolist(), x.tolist(), y.tolist()
                        )
                    ]
                )
            r_inf = s_arr + 1.0 / s_arr + 2.0
            far = (x * x + y * y) > (r_inf + 2.0 * rho) ** 2
            return np.array(
                [
                    (
                        mag0(si, q, xi, yi)
                        if isfar
                        else mag2(si, q, xi, yi, rho)
                    )
                    for si, xi, yi, isfar in zip(
                        s_arr.tolist(),
                        x.tolist(),
                        y.tolist(),
                        far.tolist(),
                    )
                ]
            )

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
        mag2, mag0 = vbm.MultiMag2, vbm.MultiMag0
        if not self.use_rho:
            return np.array(
                [mag0(xi, yi) for xi, yi in zip(x.tolist(), y.tolist())]
            )
        r_inf = max(s + 1.0 / s for (s, _, _) in companions) + 2.0
        far = (x * x + y * y) > (r_inf + 2.0 * rho) ** 2
        return np.array(
            [
                mag0(xi, yi) if isfar else mag2(xi, yi, rho)
                for xi, yi, isfar in zip(x.tolist(), y.tolist(), far.tolist())
            ]
        )

    def perform(self, node, inputs, outputs):
        p, times_np, obs_pos_np = inputs[:3]
        k = 3
        series = None
        if self.orbital_motion:
            series = inputs[k : k + 2]
            k += 2
        source_series = inputs[k : k + 2] if self.source_motion else None
        try:
            A = self._compute(p, times_np, obs_pos_np, series, source_series)
        except (ValueError, RuntimeError) as exc:
            # Invalid parameter combination -> NaN magnifications -> logp =
            # -inf -> the proposal is rejected.  That is the intended handling
            # of a bad proposal, and it is ALSO what a broken backend looks
            # like, which is why this cannot be silent: a VBM-level error (a
            # SetLensGeometry rejection, a SWIG ValueError, API drift in a new
            # wheel) rejects every proposal forever, and the default-backend
            # binary fit then runs to a garbage posterior with no message
            # anywhere.  This is exactly how the point-source-binary "VBBL"
            # bug stayed hidden, and it is what _MagOpBase.perform already
            # guards against -- mirror it here rather than trusting that this
            # Op's inputs are pre-validated.
            if not self._warned_backend:
                self._warned_backend = True
                warnings.warn(
                    f"{type(self).__name__}: VBMicrolensing raised "
                    f"{type(exc).__name__}: {exc} -- returning NaN "
                    "magnifications (logp = -inf) for this proposal. "
                    "If this repeats for every proposal the backend is "
                    "misconfigured, not merely exploring bad parameters.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            A = np.full(len(times_np), np.nan)
        outputs[0][0] = np.asarray(A, dtype=np.float64)

    def _param_labels(self):
        """Names of the entries of this Op's param vector, in order -- so the
        non-finite guard below can say WHICH parameter failed rather than
        just that something did."""
        labels = list(_BASE_LABELS)
        if self.use_rho:
            labels.append("lens.rho")
        for j in range(self.n_companions):
            labels += [f"lens.s[{j}]", f"lens.q[{j}]", f"lens.alpha[{j}]"]
        if self.bandpass is not None:
            labels.append("band.u1")
            if self.quadratic_ld:
                labels.append("band.u2")
        return labels

    def _compute(
        self, p, times_np, obs_pos_np, series=None, source_series=None
    ):
        # Non-finite check FIRST: it is the explicit handler for a NaN
        # parameter vector (return NaN magnifications -> logp = -inf ->
        # proposal rejected), and running it before the unpacking below means
        # clip_q_value never has to be the one to notice.  It used to sit
        # after the unpacking, which only worked because np.clip passed NaN
        # through silently.
        #
        # Warn once, naming the entries, for the same reason _MagOpBase warns
        # once: a *misconfigured* model is non-finite on every proposal, which
        # is indistinguishable from ordinary rejection unless the first one is
        # reported.  This branch used to return NaN in complete silence.
        #
        # The per-epoch geometry/xallarap series get the same treatment:
        # they are functions of the same sampled parameters, so a NaN there
        # is a rejected proposal, not a crash.
        all_series = list(series or []) + list(source_series or [])
        if all_series and not all(
            np.all(np.isfinite(np.asarray(v, dtype=float))) for v in all_series
        ):
            if not self._warned:
                self._warned = True
                warnings.warn(
                    f"{type(self).__name__}: non-finite per-epoch orbital-"
                    "motion geometry (s_t/alpha_t) -- returning NaN "
                    "magnifications (logp = -inf) for this proposal.  "
                    f"{_MM_NAN_ADVICE}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            return np.full(len(times_np), np.nan)
        bad = ~np.isfinite(np.asarray(p, dtype=float))
        if np.any(bad):
            if not self._warned:
                self._warned = True
                labels = self._param_labels()
                named = ", ".join(
                    f"{labels[i] if i < len(labels) else f'p[{i}]'}"
                    f" = {float(p[i])!r}"
                    for i in np.flatnonzero(bad)
                )
                warnings.warn(
                    f"{type(self).__name__}: non-finite magnification "
                    f"parameter(s) {named} -- returning NaN magnifications "
                    "(logp = -inf) for this proposal.  If this repeats for "
                    "every proposal the model is misconfigured, not merely "
                    f"exploring bad parameters.  {_MM_NAN_ADVICE}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            return np.full(len(times_np), np.nan)

        base = _base_mm_params(p)
        idx = 5
        rho = 0.0
        if self.use_rho:
            rho = _safe_rho(p[idx])
            idx += 1
        companions = []
        for j in range(self.n_companions):
            companions.append(
                (
                    float(max(float(p[idx]), S_FLOOR)),
                    clip_q_value(p[idx + 1], f"lens.q[{j}]"),
                    float(np.radians(float(p[idx + 2]))),
                )
            )
            idx += 3
        # Index FORWARD from the companion block, not backward from the end.
        # With u2 optionally following u1, `p[-1]` no longer identifies u1, and
        # a negative index that quietly means a different parameter depending
        # on the band's LD law is exactly the kind of layout bug _param_labels
        # exists to make visible.
        u1 = u2 = None
        if self.bandpass is not None:
            u1 = float(p[idx])
            if self.quadratic_ld:
                u2 = float(p[idx + 1])

        if series is not None:
            # Per-epoch geometry supersedes the scalar s_0/alpha_0 entries
            # (which remain the t0_par anchors).  Same S_FLOOR as the scalar
            # path; alpha arrives in degrees, like every alpha here.
            s_t = np.maximum(np.asarray(series[0], dtype=float), S_FLOOR)
            alpha_t = np.radians(np.asarray(series[1], dtype=float))
            companions[0] = (s_t, companions[0][1], alpha_t)

        dN, dE = self._deltas(obs_pos_np)
        tau = (
            (times_np - base["t_0"]) / base["t_E"]
            + dN * base["pi_E_N"]
            + dE * base["pi_E_E"]
        )
        u = base["u_0"] - dN * base["pi_E_E"] + dE * base["pi_E_N"]

        if source_series is not None:
            # Xallarap: the source's own per-epoch trajectory shift, at
            # exactly the slot the parallax terms occupy (C25).
            tau = tau + np.asarray(source_series[0], dtype=float)
            u = u + np.asarray(source_series[1], dtype=float)

        if self.n_companions == 1:
            # Rotate into the lens-axis frame (MulensModel Trajectory._get_xy).
            alpha_rad = companions[0][2]
            ca, sa = np.cos(alpha_rad), np.sin(alpha_rad)
            x = -tau * ca + u * sa
            y = -tau * sa - u * ca
        else:
            # Trajectory frame: same configuration with the rotation applied to
            # the lens positions instead (global rotations leave A invariant).
            # For n_companions == 0 there is no lens axis at all and only
            # |(x, y)| is read, so the same two lines serve.
            x = -tau
            y = -u

        with np.errstate(invalid="ignore", divide="ignore"):
            return self._magnify(companions, x, y, rho, u1, u2)

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
        # The per-epoch geometry/xallarap inputs are functions of the
        # sampled parameters, so they genuinely feed the output -- honesty
        # here is what keeps pullback's refusal reachable (same reasoning
        # as _MagGradOp.connection_pattern).
        pattern = [[True], [False], [False]]
        if self.orbital_motion:
            pattern += [[True], [True]]
        if self.source_motion:
            pattern += [[True], [True]]
        return pattern


class _MagGradOp(Op):
    """Numerical (forward-difference) gradient for a magnification Op."""

    itypes = [pt.dvector, pt.dvector, pt.dmatrix, pt.dvector]
    otypes = [pt.dvector]

    def __init__(
        self,
        builder,
        coords,
        mag_method,
        use_rho=False,
        bandpass=None,
        eps=1e-6,
    ):
        self._builder = builder
        self.coords = coords
        self.mag_method = mag_method
        self.use_rho = use_rho
        self.bandpass = bandpass
        self.eps = eps
        self._coord_cache = {}

    def infer_shape(self, node, input_shapes):
        return [input_shapes[0]]

    def _calc(self, p, times_1d, sat_coord):
        try:
            model = self._builder(
                p, self.coords, self.mag_method, self.use_rho
            )
            with np.errstate(invalid="ignore", divide="ignore"):
                if self.bandpass is not None:
                    model.set_limb_coeff_u(self.bandpass, float(p[-1]))
                    return model.get_magnification(
                        times_1d,
                        satellite_skycoord=sat_coord,
                        bandpass=self.bandpass,
                    )
                return model.get_magnification(
                    times_1d, satellite_skycoord=sat_coord
                )
        except (ValueError, RuntimeError):
            return np.full(len(times_1d), np.nan)

    def perform(self, node, inputs, outputs):
        params, times_np, obs_pos_np, g = inputs
        out = np.zeros(params.shape, dtype=params.dtype)
        times_1d = np.atleast_1d(times_np)
        sat_coord = _dev_skycoord(obs_pos_np, self._coord_cache)
        # One clear covers the whole finite-difference loop: every _calc
        # below shares this sat_coord and time grid.
        _clear_mm_satellite_cache()
        f_x = self._calc(params, times_1d, sat_coord)
        for i in range(len(params)):
            p_plus = params.copy()
            p_plus[i] += self.eps
            out[i] = np.sum(
                g * (self._calc(p_plus, times_1d, sat_coord) - f_x) / self.eps
            )
        outputs[0][0] = out

    def pullback(self, inputs, outputs, cotangents):
        # This Op has no second derivative, and says so rather than inventing
        # one (review 1.6.3).  What it used to return was wrong twice over:
        # `cotangents[0]` for the incoming-cotangent input g is n_params long
        # while the true VJP there is TIMES-shaped (this Op computes
        # out[i] = sum_t g[t] * D[i,t], so d out / d g is D itself), and the
        # params input was handed a DisconnectedType while connection_pattern
        # declared it connected.
        #
        # RAISE, not DisconnectedType, and the distinction matters.
        # DisconnectedType asserts that the output does not depend on that
        # input, and for g that assertion is FALSE -- the output is exactly
        # linear in g.  Declaring it disconnected would return a silently
        # ABSENT gradient where a real one exists, which is the same class of
        # bug this comment is about, only quieter.
        #
        # Nothing is lost by refusing.  A second derivative built on top of a
        # FIRST-ORDER FINITE DIFFERENCE carries O(eps) error on a quantity
        # that is itself only O(eps) accurate, so the Hessian would be
        # numerically meaningless even if it were implemented correctly.
        # There is no future in which second-order support is added inside
        # this Op rather than by rewriting the magnification in pytensor or
        # porting a differentiable library.
        raise NotImplementedError(
            "_MagGradOp has no second derivative: its own output is a "
            "forward-difference approximation, so any Hessian built on it "
            "would be numerically meaningless.  Use a gradient-free sampler "
            "(PTDE) for binary/finite-source microlensing."
        )

    # Backward compatibility with PyTensor < 3 which calls grad() instead of pullback()
    def grad(self, inputs, gradients):
        return self.pullback(inputs, [], gradients)

    def connection_pattern(self, node):
        # Honest, and honest is what makes the raise above reachable.  Both
        # the parameter vector and the incoming cotangent g genuinely feed the
        # output, so both are connected; declaring g disconnected -- the
        # obvious way to "fix" the wrong VJP it used to return -- would let
        # pytensor skip pullback entirely and hand back an absent gradient
        # instead of the error.  times/obs_pos stay False, matching
        # _MagOpBase: they are data, never differentiated against.
        return [[True], [False], [False], [True]]
