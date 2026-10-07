import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

import pymc as pm
import pytensor
import pytensor.tensor as pt
from scipy.optimize import nnls

from exozippy.compat import patch_mulensmodel_method_order
from exozippy.components.instrument import Instrument
from exozippy.config import PRECEDENCE_DERIVED_DATA, user_entry
from exozippy.ephemeris import get_observer_position
from exozippy.outputs.prose import get_collector
from exozippy.skyframe import observer_sky_offset

from ..parameterization import pin_unselected
from ..sed.magsys import AB, parse_magsys
from . import peakfind
from .physics import (
    RHO_FLOOR,
    S_FLOOR,
    T_E_FLOOR,
    clip_q_value,
    floor_u_0_value,
)


def _raw_initval(user_params, key, default=None):
    """Read a ``user_params`` initval, collapsing a list (P4 multi-seed
    sampling) to its first (seed 0) entry.  Only meaningful before
    ConfigManager.finalize_user_params runs -- afterwards ``user_params``
    already holds the resolved seed-0 scalar.  The entry is a field dict
    (a bare params value was translated at construction, review 1.1.7)."""
    data = user_entry(user_params, key)
    if data is None:
        return default
    val = data.get("initval", default)
    if isinstance(val, (list, tuple)):
        val = val[0] if val else default
    return val


class _BootstrapGeometry:
    """The flux bootstrap's reader of the event geometry (see
    ``MulensInstrument._probe_bootstrap_geometry``).

    ``geometry(path, default=None)`` is the path's start in USER units when it
    is INFORMED (``ProbedStart.informed``: it traces back to a user entry or a
    component hint or seed), else `default`.  `probed` is the
    ``ConfigManager.probe_start`` answer; asking for a path it does not hold
    is a KeyError -- a bookkeeping bug in the caller, not "unknown".
    `has_companion` says whether ``lens.1`` exists (element 0 is the masked
    primary), i.e. whether the lens.1.* geometry was probed at all.
    """

    def __init__(self, probed, has_companion):
        self.probed = probed
        self.has_companion = has_companion

    def __call__(self, path, default=None):
        ps = self.probed[path]
        return ps.user_value if ps.informed else default


class MulensInstrument(Instrument):
    """Microlensing photometry, modeled and fit entirely in FLUX.

    The likelihood is Gaussian in flux, never in magnitudes.  Photon-counting
    noise is (approximately) Gaussian in flux; a magnitude is a logarithm of
    it, so a Gaussian in magnitudes is only a first-order approximation that
    degrades exactly where the data are faint -- and is undefined for the
    non-positive fluxes difference imaging routinely produces.  Since the
    model itself is linear in flux (F = f_s*A + f_b), flux is also the natural
    internal quantity: the old code computed F and then took -2.5*log10 of it
    only to hand the result to a Normal.

    ``data_format`` is therefore purely a statement about the FILE:

    - ``flux`` (difference imaging and simulated data): used as given.
      Negative and zero fluxes are first class -- nothing is clamped.
    - ``magnitude`` (the default; the usual survey format): converted at load
      to ``F = 10**(-0.4 m)``, exact for the value, with the error propagated
      to first order as ``sigma_F = ln(10)/2.5 * F_obs * sigma_m`` evaluated
      at the OBSERVED flux (so sigma stays a fixed constant, not a function of
      the model).  The resulting posterior differs from the old magnitude-space
      one only at O(sigma_m) -- ~1% for 0.01 mag photometry.
    - ``dia`` (native difference-imaging output, e.g. KMTNet pySIS): a
      DIFFERENCE flux, which is NOT the modeled observable.  See below.

    ``data_format: dia`` and why it needs five columns
    --------------------------------------------------
    Difference imaging reports ``dflux``, the flux relative to a reference
    image, with the source's own reference flux subtracted away.  The model
    is ``F = f_s*A + f_b`` in TOTAL flux, and a difference curve cannot be
    written that way: its baseline sits at zero, so ``f_total = f_s + f_b``
    is ~0 while ``f_s`` is large, which drives ``q_source = f_s/f_total``
    far outside its ``[0, 2]`` bound.  (That bound is correct -- it says the
    blend is at most as negative as the total flux -- and must not be
    widened to accommodate a difference curve.)

    The missing piece is the reference flux, and a native pySIS file
    carries it implicitly.  Its five columns are
    ``time dflux dflux_err mag mag_err``, where ``mag`` is the TOTAL
    brightness, so the two are related by

        mag = zp - 2.5*log10(ref - dflux)

    which determines ``ref`` and ``zp`` exactly (verified on MulensModel's
    native KB180003 set: ``ref = 1584.9`` at ``zp = 28.000`` for all three
    sites, residual rms 2.8e-5 mag).  The total flux is then
    ``F = ref - dflux``.  Note the subtraction: in the pySIS files seen
    here, ``dflux`` grows MORE NEGATIVE as the star brightens.  That is
    taken as the pipeline's convention on the strength of one independently
    sourced set (MulensModel's KB180003) -- corroboration, not a survey --
    so a file that disagrees is not impossible.  The solved reference is
    checked against the mag column either way, which is what actually
    guards this.

    A three-column difference file therefore cannot be read as ``dia``: the
    reference flux is simply absent, and no choice of it is derivable from
    the data.  That is an error rather than a guess, because every available
    fallback silently changes the science -- offsetting to zero blending,
    for instance, IMPOSES a blending fraction and corrupts any ``theta_E``
    or lens mass derived from the source flux.  Supply the native file, or
    give ``reference_flux``/``reference_mag`` explicitly.

    ``reference_flux`` (file units) or ``reference_mag`` (with ``zp``, or
    with the file's own zeropoint once recovered) override the solved value
    and let a three-column file be read.  The recovered ``zp`` is also what
    ties this light curve's flux system to a magnitude, which is what an
    ``sed:`` block needs.

    ``f_source``/``f_blend``/``log_f_total`` are unchanged: they live in the
    file's own flux system, which for a magnitude file is the system in which
    ``F = 10**(-0.4 m)`` (i.e. an instrumental zeropoint of 0), exactly as
    before.  ``err_scale`` is dimensionless and its meaning is unchanged.
    """

    # Multiplicative per-instrument error scale (not additive jitter).
    noise_model = "err_scale"
    prose_noun = "microlensing photometry"
    # Magnitude-space detrend coefficients MULTIPLY the flux model by
    # 10**(-0.4 X.c) (Instrument.DETREND_SPACE).
    DETREND_SPACE = "magnitude"

    # Deps of the derived `zeropoint` that are injected as context nodes by
    # add_parameter below rather than resolved as manifest parameters, so
    # graph.py leaves them out of the build-order graph.
    context_dep_names = frozenset({"m_source_pred"})

    # ...built with one entry per light curve, so Component._element_expression
    # may slice it to the elements that have an SED prediction (the rest of
    # the zeropoint vector is INACTIVE -- see register_parameters).
    aligned_context_deps = frozenset({"m_source_pred"})

    def __init__(self, config, config_manager):
        super().__init__(config, config_manager)
        self.label = "Microlensing Data"
        self.total_detrend_cols = 0
        # _finite_source_limb_darkening is called from both build_likelihood
        # and compile_plotters; the notice belongs to the topology, not to the
        # call, so it is emitted once.
        self._warned_multiband_ld = False
        # Each light curve's zeropoint magnitude system AS STATED, in the one
        # internal spelling (None = unstated = the band filter's native
        # system), translated at this boundary by the SAME parser the SED's
        # rows use: exact "Vega"/"AB", a case variant raises with a "did you
        # mean" (issue #313).  Resolved against the BC column record in
        # _resolve_zeropoint_systems once the SED grid exists.
        self._zp_magsys_stated = [
            parse_magsys(c.get("magsys"), f"mulensinstrument {name!r}")
            for name, c in zip(self.names, self.config)
        ]

    @property
    def prefix(self):
        return "mulensinstrument"

    @classmethod
    def config_schema(cls):
        return [
            {
                "key": "file",
                "kind": "datafile",
                "accepts": "*.dat",
                "required": True,
                "doc": (
                    "Whitespace-delimited microlensing light curve; columns "
                    "are time, flux-or-magnitude, error (see data_format), "
                    "then optional detrend columns. Each extra column gets "
                    "its own coefficient for this instrument, applied to the "
                    "model magnitude (i.e. multiplicatively in flux). Comment "
                    "lines start with '#'."
                ),
            },
            {
                "key": "data_format",
                "kind": "option",
                "accepts": ["magnitude", "flux", "dia"],
                "required": False,
                "doc": (
                    "Photometry format of the data FILE. Default 'magnitude'. "
                    "The fit is always done in flux; magnitude files are "
                    "converted at load (F = 10**(-0.4 m), sigma_F = "
                    "ln(10)/2.5 * F * sigma_m). With 'flux', non-positive "
                    "fluxes are kept as-is -- nothing is clamped. With "
                    "'dia' the file is native difference-imaging output "
                    "(e.g. KMTNet pySIS) and MUST carry five columns -- "
                    "time, dflux, dflux_err, mag, mag_err -- from which the "
                    "reference flux is solved and the total flux "
                    "reconstructed as ref - dflux. A three-column "
                    "difference file raises, because the reference flux is "
                    "absent and guessing it changes the science; supply "
                    "reference_flux or reference_mag to read one anyway."
                ),
            },
            {
                "key": "reference_flux",
                "kind": "option",
                "accepts": None,
                "required": False,
                "doc": (
                    "data_format: dia only. The difference-imaging reference "
                    "flux in the FILE's own flux units. Overrides the value "
                    "solved from the mag column, and lets a file without one "
                    "be read. Total flux is reconstructed as ref - dflux."
                ),
            },
            {
                "key": "reference_mag",
                "kind": "option",
                "accepts": None,
                "required": False,
                "doc": (
                    "data_format: dia only. The reference magnitude, as an "
                    "alternative spelling of reference_flux; converted with "
                    "this file's zeropoint (zp, default 28.0 -- the pySIS "
                    "convention) as ref = 10**(-0.4*(reference_mag - zp))."
                ),
            },
            {
                "key": "zp",
                "kind": "option",
                "accepts": None,
                "required": False,
                "doc": (
                    "data_format: dia only. Zeropoint used with "
                    "reference_mag, and the fallback when the mag column is "
                    "absent. Default 28.0, the pySIS convention."
                ),
            },
            {
                "key": "observer_location",
                "kind": "option",
                "accepts": None,
                "required": False,
                "doc": (
                    "Observer location for parallax: 'earth', an ephemeris "
                    "name for a space-based observatory, an astropy site "
                    "name (e.g. 'CTIO'), or a geodetic "
                    "'lon_deg,lat_deg[,height_m]' string (lon FIRST) for "
                    "terrestrial parallax. Default 'earth'."
                ),
            },
            {
                "key": "band",
                "kind": "ref",
                "accepts": ["band"],
                "required": False,
                "doc": "Name of the band: block associated with this light curve.",
            },
            {
                "key": "magsys",
                "kind": "option",
                "accepts": ["Vega", "AB"],
                "required": False,
                "doc": (
                    "Magnitude system of this light curve's zeropoint "
                    "(issue #313): exactly 'Vega' or 'AB', case-sensitive. "
                    "Omit it to mean the native system of the band's filter "
                    "(Roman WFI is AB, Cousins/Bessell/2MASS are Vega), the "
                    "same rule as an SED row; a filter with no recorded "
                    "native system must state it. The SED predicts Vega "
                    "magnitudes, so an AB zeropoint is compared through the "
                    "filter's m_AB - m_Vega."
                ),
            },
            {
                "key": "sed_constrains_blend",
                "kind": "option",
                "accepts": [True, False],
                "required": False,
                "doc": (
                    "When an SED is present, also tie f_blend to the "
                    "SED-predicted flux. Default false. A tie is a physics LINK, not a one-way assignment: information flows toward whichever side is less constrained elsewhere (components.md, 'Config flag vocabulary')."
                ),
            },
            {
                "key": "sed_blend_sigma",
                "kind": "option",
                "accepts": None,
                "required": False,
                "doc": (
                    "Gaussian width (mag) of the SED f_blend constraint when "
                    "sed_constrains_blend is set. Default 0.2."
                ),
            },
            {
                "key": "reference",
                "kind": "option",
                "accepts": [True, False],
                "required": False,
                "doc": (
                    "Flag exactly one instrument with 'reference: true' to peg "
                    "every plotted data set and model curve onto its f_source/"
                    "f_blend flux system. Defaults to the first instrument."
                ),
            },
            cls._mask_config_schema(),
            cls._columns_config_schema(
                ("time", "mag", "err"),
                note=(
                    "With data_format: flux the observable role is named "
                    "'flux' instead of 'mag'."
                ),
            ),
            *cls._time_config_schema(),
            cls._plot_style_config_schema(),
            cls._gp_config_schema(),
            cls._likelihood_config_schema(),
        ]

    def _reference_index(self):
        """Index of the instrument whose flux system anchors the plot.

        Honors a per-instrument 'reference: true' flag; falls back to the
        first instrument when none (or an out-of-range name) is set. Warns if
        more than one instrument is flagged and uses the first flagged one.
        """
        flagged = [
            i
            for i in range(self.n_elements)
            if self.config[i].get("reference", False)
        ]
        if not flagged:
            return 0
        if len(flagged) > 1:
            names = ", ".join(self.names[i] for i in flagged)
            logger.warning(
                f"Multiple mulensinstrument entries flagged reference ({names}); "
                f"using '{self.names[flagged[0]]}'."
            )
        return flagged[0]

    def load_data(self, system):
        """Stage 1: Load photometry and pre-calculate observer positions.

        Single-event assumption (enforced by MulensEvent.__init__): there is
        one mulensevent instance, so its t0_par and magnification dispatch
        are used throughout.
        """
        # A light curve is meaningless without the event it is a light
        # curve OF (review 2.6.7).  MulensEvent.__init__ already refuses an
        # event without 'lens:'/'source:', so the three blocks are checked
        # here together: every reader below may then dereference them
        # outright, and a config missing one gets a config error naming it
        # instead of an AttributeError deep in stage 1.
        missing = [
            k
            for k in ("mulensevent", "lens", "source")
            if k not in system.active_components
        ]
        if missing:
            raise ValueError(
                f"[{self.prefix}] microlensing photometry needs the "
                f"{' and '.join(repr(k) for k in missing)} block(s), and the "
                f"config declares none: add a single mulensevent block plus "
                f"lens: [{{body: star.Lens}}] and source: "
                f"[{{body: star.Source}}] (one entry per body)."
            )

        self.fs_init = []
        self.q_source_init = []
        # data_format: dia only -- the zeropoint recovered per light curve
        # alongside its reference flux.  It is what ties this curve's flux
        # system to a magnitude, which is what an sed: block needs.
        self._dia_zp = {}
        self.q_flux_init = []  # per-instrument f_s2/f_s1 (binary source)
        blocks = self._concat_blocks()

        # n_elements, not star_map: this is stage 1, and the source
        # component's maps are built in stage 2 (component order within a
        # stage is the user's config key order, so they may not exist yet).
        self._n_sources = int(system.source.n_elements)

        # Source RA/Dec (degrees from resolve → radians for projection math).
        # Stashed for MulensEvent._earth_vperp_en: the mu_helio -> mu_geo
        # conversion must project Earth's velocity with the same (ra, dec)
        # the Skowron deltas are projected with.
        ra_deg, dec_deg = self._resolve_source_radec_deg(system)
        ra_rad = ra_deg * np.pi / 180.0
        dec_rad = dec_deg * np.pi / 180.0
        self._source_radec_rad = (ra_rad, dec_rad)

        # Pass 1: read every file.  The raw times must exist before the
        # Skowron reference frame is anchored below (t0_par can fall back to
        # the median data time).
        per_file = []
        for i in range(self.n_elements):
            fmt = self.config[i].get("data_format", "magnitude")
            if fmt == "dia":
                # Native difference imaging: five columns, and the mag
                # column is not optional -- it is the only thing that
                # carries the reference flux.  Validated before the read so
                # the message names the real problem rather than surfacing
                # as a column-selection IndexError.
                self._require_dia_columns(i)
                roles = ("time", "dflux", "err", "mag", "mag_err")
            else:
                roles = ("time", "flux" if fmt == "flux" else "mag", "err")
            # Shared reader: columns:, mask:, time_* conversion, then sort
            # before the observer positions are computed from t, so the
            # ephemeris rows stay aligned with the photometry.
            df = self._read_data(i, roles=roles, detrend=True)
            t, f, e = (
                df.iloc[:, 0].values.astype(float),
                df.iloc[:, 1].values.astype(float),
                df.iloc[:, 2].values.astype(float),
            )

            if fmt == "dia":
                # dflux -> TOTAL flux.  F = ref - dflux (pySIS dflux grows
                # more negative as the star brightens).  The error is
                # unchanged: the reference is a constant offset, so
                # sigma_F = sigma_dflux exactly, with no propagation.
                ref, zp = self._dia_reference(
                    i,
                    dflux=f,
                    mag=df.iloc[:, 3].values.astype(float),
                )
                self._dia_zp[self.names[i]] = zp
                f = ref - f

            if fmt not in ("flux", "dia"):
                # Magnitudes -> flux.  Exact for the value; the error is the
                # first-order propagation evaluated at the OBSERVED flux, so
                # sigma stays a data constant (using the model flux instead
                # would make sigma a function of the parameters and bias the
                # fit toward faint models).
                f = 10.0 ** (-0.4 * f)
                e = (np.log(10.0) / 2.5) * f * np.maximum(e, 0.0)

            per_file.append((t, f, e, df))

        # THE BUILT-IN PEAK FINDER (8.4.9).  Runs HERE, after pass 1, so it
        # sees the data as the model will -- masked, detrended, and already
        # converted to flux.  It also has to land BEFORE
        # `_estimate_flux_components` in pass 2, which reads the seed t_0 /
        # u_0 / t_E to decompose each band's flux and falls back to the
        # median flux and the declared q_source start without them.
        self._peak_find_seeds(system, per_file)

        # The event geometry the bootstrap below reads, as the relaxation
        # engine would start it given everything known NOW (user entries, the
        # peak finder's seed 0, hints so far) -- one probe for
        # every file.  See _probe_bootstrap_geometry.
        geometry = self._probe_bootstrap_geometry(system)

        # Geocentric reference (Skowron+2011 convention): Earth's position and
        # velocity at t_0_par define the inertial frame.  All observer positions
        # are stored as deviations from this linear Earth trajectory so that
        # t_0/u_0 remain geocentric parameters.  Re-resolved here rather
        # than taken from MulensEvent.__init__: the peak finder's seeds
        # arrive in stage 1 (just above), after MulensEvent snapshotted
        # user_params, and a reference epoch far from the data
        # makes the linear Earth extrapolation diverge (O(100) AU after
        # ~20 yr), shearing tau/u by O(deviation x pi_E).
        self._t0_par = self._resolve_t0_par_final(
            system, np.concatenate([f[0] for f in per_file])
        )
        system.mulensevent.t0_par[0] = self._t0_par
        # The frame anchors are the EVENT's (MulensEvent.geocentric_frame):
        # an astrometric dataset of a lensed source builds deviations in the
        # same frame (conventions.md C30), so there is one owner.  Kept as
        # attributes here because MulensEvent._earth_vperp_en reads
        # _earth_vel_ref off this instrument for the mu_helio -> mu_geo
        # conversion.
        self._event = system.mulensevent
        _, self._earth_pos_ref, self._earth_vel_ref = (
            self._event.geocentric_frame()
        )

        # Median absolute position per instrument (used by MulensEvent to
        # detect satellite parallax when sizing the logmass scale)
        self.inst_ref_pos = []

        # Pass 2: observer positions, flux bootstraps, and sanity checks.
        for i, (t, f, e, df) in enumerate(per_file):
            obs_loc = self.config[i].get("observer_location", "earth")
            xyz_abs = self.get_observer_position(t, observer_location=obs_loc)
            self.inst_ref_pos.append(np.median(xyz_abs, axis=0))

            xyz_delta = self._abs_to_delta(t, xyz_abs)

            f_total, q_source, q_flux = self._estimate_flux_components(
                t, f, xyz_delta, ra_rad, dec_rad, i, geometry
            )
            self.fs_init.append(f_total)
            self.q_source_init.append(q_source)
            self.q_flux_init.append(q_flux)

            self._check_data_format(
                t,
                f,
                e,
                xyz_delta,
                ra_rad,
                dec_rad,
                self.config[i].get("file", f"instrument {i}"),
                geometry,
                data_format=self.config[i].get("data_format", "magnitude"),
            )

            # Optional detrending against extra data columns (columns 4+ of
            # the file), exactly as rvinstrument/transit do: one coefficient
            # per column per instrument, kept from mixing across instruments
            # by the block-diagonal design matrix the accumulator builds.  A
            # column is a magnitude-space trend (airmass, seeing, ...), i.e.
            # it enters the flux model MULTIPLICATIVELY as 10**(-0.4 * X.c) --
            # algebraically identical to the additive magnitude detrending
            # this component used before it moved to a flux likelihood, and
            # the right form for a throughput/extinction trend either way.
            #
            # observer_pos rides along as a per-epoch side array, so the
            # Skowron geocentric deviations (used by both magnification
            # paths) stay row-aligned with the photometry by construction.
            blocks.add(i, time=t, obs=f, err=e, df=df, observer_pos=xyz_delta)

        self.inst_ref_pos = np.array(
            self.inst_ref_pos
        )  # (n_inst, 3) absolute AU

        # Shared accumulator: concatenation (time/flux/err/observer_pos),
        # inst_map, the per-file row ranges, the block-diagonal detrend
        # matrix, and the optional GP / robust-likelihood hooks.  No
        # user_factor: the errors are already in the amplitude parameters'
        # unit (flux, in each file's own flux system).
        #
        # `flux` is the modeled observable, in the file's own flux system.
        # Magnitude files were converted above; flux files are untouched,
        # negatives and all.  There is deliberately no `self.mag`: nothing
        # downstream may reintroduce a magnitude-space likelihood.
        blocks.finalize("flux")

    def _resolve_t0_par_final(self, system, all_times):
        """Final t0_par: the reference epoch anchoring the Skowron+2011 frame.

        MulensEvent.__init__ resolves t0_par from its config and
        user_params only; seed hints (the peak finder's) arrive later
        (stage 1), so a params file that omits the microlensing start
        values used to fall through to the 2450000.0 default, parking the
        reference epoch decades before the data.

        Priority: explicit mulensevent ``t0_par`` > user ``source.0.t_0``
        initval > seed-0 t_0 > median data time.  Any of these
        keeps the linear Earth extrapolation within the season it is a good
        approximation for.
        """
        event_config = system.mulensevent.config[0]
        if "t0_par" in event_config:
            return float(event_config["t0_par"])
        cm = self.config_manager
        val = _raw_initval(cm.user_params, "source.0.t_0")
        if val is None:
            val = cm.seed_start_value("source.0.t_0")
        if val is not None:
            return float(val)
        t_med = float(np.median(all_times))
        logger.info(
            f"[{self.prefix}] No t0_par, lens t_0, or seed t_0 found; "
            f"anchoring the parallax reference epoch at the median data "
            f"time ({t_med:.2f})."
        )
        return t_med

    def _peak_find_seeds(self, system, per_file):
        """Seed t_0/u_0/t_E with the built-in PSPL fit when nothing else did.

        THE microlensing seeder (review 8.6.25, JDE 2026-09-30; the only
        one since 2026-10-01).  `peak_find` on the
        mulensevent block takes:

          - absent (default): run when another seeder has not already
            registered seed sets (one seeder per fit, review 2.1.25) and
            ``peakfind.plan_peak_find`` says a start is missing.  It HOLDS
            every informed one of t_0/u_0/t_E and fits and pushes only the
            rest -- so a t_E the user gives or the galactic kinematics
            derive is never overridden.  Without it those starts sat at
            ``defaults.yaml``, which for a real event is a start no sampler
            recovers from.
          - ``true``: run REGARDLESS, fitting all three (and replacing any
            seed sets another seeder registered -- none does today); a user
            entry still outranks every seed.
          - ``false``: never run.

        ``auto`` (the old spelling of the default) RAISES: the default is
        the absent key, and an n-way choice is spelled with booleans, not an
        enum.

        Failure is not fatal.  A seeder that raises should leave the fit in
        exactly the state it would have been in without this module, which
        is why the search is wrapped -- a start value moves no posterior,
        and killing a run over one is the wrong trade.
        """
        # load_data has already refused a config without the event block,
        # and MulensEvent.__init__ one with an empty block.
        spec = system.mulensevent.config[0].get("peak_find")
        if spec == "auto":
            raise ValueError(
                f"[{self.prefix}] peak_find: auto is no longer a spelling "
                f"(review 8.6.25): omit the key for the default (find what "
                f"the params file does not give), or use true (always) or "
                f"false (never)."
            )
        if spec not in (None, True, False):
            raise ValueError(
                f"[{self.prefix}] peak_find: {spec!r} is not a spelling: "
                f"omit the key for the default, or use true or false."
            )
        if spec is False:
            return
        forced = spec is True

        # One seeder per fit (review 2.1.25): add_seed_hints raises on a
        # second registration.  By default an existing seeder owns the fit
        # and the finder stays out; `forced` REPLACES its sets, and says so.
        # No other seeder registers sets today, so this is the
        # generic rule rather than a live case.
        replace = False
        if self.config_manager.seed_hint_sets:
            if not forced:
                logger.debug(
                    f"[{self.prefix}] peak finder: "
                    f"{self.config_manager.seed_hint_source!r} already "
                    f"seeded this fit; not running."
                )
                return
            replace = True
            logger.warning(
                f"[{self.prefix}] peak_find: true REPLACES the "
                f"{len(self.config_manager.seed_hint_sets)} seed set(s) "
                f"already loaded for this fit: its multi-seed solutions "
                f"(and any s/q/alpha) are discarded."
            )

        # What is missing, from the engine-implied starts: see
        # peakfind.plan_peak_find for why this is not user_hints_sufficient,
        # and why an informed u_0 / t_E (e.g. a t_E the galactic kinematics
        # derive) is HELD rather than refit.  `forced` fits all three.
        fixed = {}
        if not forced:
            fixed = peakfind.plan_peak_find(self.config_manager)
            if fixed is None:
                return

        curves = []
        for t, f, e, _df in per_file:
            ok = np.isfinite(t) & np.isfinite(f) & np.isfinite(e) & (e > 0)
            if ok.sum() >= 4:
                curves.append((t[ok], f[ok], 1.0 / e[ok] ** 2))
        if not curves:
            logger.warning(
                f"[{self.prefix}] peak finder: no usable epochs; "
                f"t_0/u_0/t_E keep their defaults."
            )
            return

        # Hand the search this component's OWN magnification so the seed is
        # built with the same u_0 floor the likelihood uses.  Parallax is
        # held at zero (delta_e = delta_n = 0), matching run_or_load's
        # no_parallax default: a PSPL seed cannot resolve the trajectory
        # asymmetry, so it should not claim to.
        zeros = {}

        def mag_fn(t, t_0, u_0, t_E):
            d = zeros.get(len(t))
            if d is None:
                d = np.zeros_like(t)
                zeros[len(t)] = d
            return self._pspl_magnification(t, d, d, t_0, u_0, t_E, 0.0, 0.0)

        try:
            seed = peakfind.find_pspl_seed(curves, mag_fn=mag_fn, fixed=fixed)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"[{self.prefix}] peak finder failed "
                f"({type(exc).__name__}: {exc}); t_0/u_0/t_E keep their "
                f"defaults."
            )
            return
        if seed is None:
            logger.warning(
                f"[{self.prefix}] peak finder found no PSPL solution; "
                f"t_0/u_0/t_E keep their defaults."
            )
            return
        peakfind.push_peak_find_hints(
            seed, self.config_manager, source=self.prefix, replace=replace
        )
        get_collector(system).add(
            "Starting values for the microlensing trajectory "
            "($t_0$, $u_0$, $t_{\\rm E}$) were derived from a point-lens "
            "point-source fit to the light curves."
            if not seed["fixed"]
            else "Starting values for the microlensing trajectory "
            "parameters not given by the user were derived from a "
            "point-lens point-source fit to the light curves.",
            section="microlensing",
            key=f"{self.prefix}.peakfind",
            rank=30,
        )

    def _resolve_source_radec_deg(self, system):
        """Source star's sky position in degrees.

        Falls back to the (primary) lens star's ra/dec when the user never
        explicitly set the source's own -- source and lens are angularly
        coincident by construction, so params.yaml only needs to state the
        target coordinates once, on the lens.
        """
        # bodies, not star_map: called from load_data (stage 1), before the
        # source component's build_maps has necessarily run.
        source_ndx = int(system.source.bodies[0][1])
        n_stars = system.star.n_elements
        ra_all = self.config_manager.resolve("star", "ra", shape=(n_stars,))[
            "initval"
        ]
        dec_all = self.config_manager.resolve("star", "dec", shape=(n_stars,))[
            "initval"
        ]

        star_names = getattr(system.star, "names", None)
        keys = [f"star.{source_ndx}.ra", "star.ra"]
        if star_names:
            keys.append(f"star.{star_names[source_ndx]}.ra")
        user_set_source = any(
            k in self.config_manager.user_params for k in keys
        )

        ndx = source_ndx
        if not user_set_source:
            # The PRIMARY lens body (entry 0), never "the first star among
            # the lens bodies": MulensEvent._validate_bodies (stage 3)
            # refuses a non-star primary, but this runs in stage 1, so the
            # same rule is enforced here rather than silently borrowing a
            # star COMPANION's coordinates.
            p_type, p_ndx = system.lens.bodies[0]
            if p_type != "star":
                raise ValueError(
                    f"[{self.prefix}] the primary lens body (lens entry 0) "
                    f"is '{p_type}.{p_ndx}', but it must be a star "
                    f"(see MulensEvent._validate_bodies)."
                )
            ndx = int(p_ndx)

        return float(ra_all[ndx]), float(dec_all[ndx])

    def _probe_bootstrap_geometry(self, system):
        """The flux bootstrap's event geometry, read with ``probe_start``.

        Returns a ``_BootstrapGeometry``: ``geometry(path, default=None)`` is
        the path's start in USER units (the MulensModel convention --
        ``alpha`` in degrees) when it is INFORMED, i.e. traces back to a user
        entry or a component hint or seed, directly or through a relation;
        otherwise `default`.

        WHY A PROBE (config.md, "Reading a start before stage 4"; review
        2.6.30(b)).  This runs at stage 1, and the geometry it needs is often
        not WRITTEN anywhere yet: examples/ob170114 derives t_E and pi_E from
        the lens/source masses, distances and proper motions, and ob09020's
        q comes from the body masses.  Reading raw ``user_params`` (plus the
        seed-0 hints) saw ``t_E = None`` there and bootstrapped the flux
        split on an invented 30 d with no parallax, while the engine's own
        start was t_E = 123.2 d, pi_E = (0.110, 0.178).

        Uninformed is not the same as "use the default": a defaults.yaml
        ``t_0`` or ``u_0`` describes no event, so a bootstrap built on it
        would be a confident wrong answer.  The pi_E default (0, 0) is what
        the bootstrap has always assumed without parallax information, so
        the pi_E reads keep it; every other quantity comes back None and the
        consumers say what is missing.

        One probe per ``load_data`` (the probe is a full engine solve; see
        ``ConfigManager.probe_start`` on why it is not cached).  A companion
        lens's geometry is probed only when the lens HAS a companion
        (``lens.1`` -- element 0 is the masked primary).
        """
        paths = [
            "mulensevent.0.t_E",
            "mulensevent.0.pi_E_N",
            "mulensevent.0.pi_E_E",
        ]
        for j in range(self._n_sources):
            paths += [f"source.{j}.t_0", f"source.{j}.u_0", f"source.{j}.rho"]
        if system.lens.n_companions >= 1:
            paths += ["lens.1.s", "lens.1.log_s", "lens.1.q", "lens.1.alpha"]
        return _BootstrapGeometry(
            self.config_manager.probe_start(paths),
            has_companion=system.lens.n_companions >= 1,
        )

    # ------------------------------------------------------------------
    # data_format: dia -- native difference imaging
    # ------------------------------------------------------------------
    def _require_dia_columns(self, i):
        """Raise unless file ``i`` carries the five native pySIS columns.

        This is the guard the whole format rests on.  A difference flux is
        not the modeled observable, and the ONLY thing that turns it into
        one is the reference flux, which a native file carries implicitly
        through its ``mag`` column.  A three-column difference file does not
        contain it and nothing in the data can recover it, so reading one
        would mean choosing a reference -- and every choice silently changes
        the science (offsetting to zero blending imposes a blending
        fraction, which corrupts any theta_E or lens mass derived from the
        source flux).  So this raises rather than guessing, and names the
        two ways out.
        """
        label = self.names[i]
        n_user = 0
        spec = (self.config[i] or {}).get("columns")
        if isinstance(spec, dict):
            n_user = len(spec)
        if n_user >= 4:
            # An explicit columns: map naming mag is enough; trust it.
            return
        if self._reference_flux_override(i) is not None:
            # An explicit reference makes the mag column unnecessary.
            return
        import pandas as pd

        head = pd.read_csv(
            self.files[i],
            sep=r"\s+",
            engine="c",
            header=None,
            comment="#",
            nrows=1,
        )
        if head.shape[1] < 5:
            raise ValueError(
                f"[{self.prefix}:{label}] data_format: dia needs the five "
                f"native pySIS columns (time, dflux, dflux_err, mag, "
                f"mag_err) but '{self.files[i]}' has {head.shape[1]}. A "
                f"difference flux is not the modeled observable: the total "
                f"flux is ref - dflux, and the reference flux is carried "
                f"ONLY by the mag column (mag = zp - 2.5*log10(ref - "
                f"dflux)). It cannot be recovered from a three-column file, "
                f"and guessing it changes the science -- offsetting to zero "
                f"blending, for instance, imposes a blending fraction and "
                f"corrupts any theta_E or lens mass taken from the source "
                f"flux. Supply the native five-column file, or set "
                f"reference_flux (file units) or reference_mag on this "
                f"instrument."
            )

    def _reference_flux_override(self, i):
        """``reference_flux``/``reference_mag`` for file ``i``, or None."""
        c = self.config[i] or {}
        if c.get("reference_flux") is not None:
            return float(c["reference_flux"])
        if c.get("reference_mag") is not None:
            zp = float(c.get("zp", 28.0))
            return 10.0 ** (-0.4 * (float(c["reference_mag"]) - zp))
        return None

    def _dia_reference(self, i, dflux, mag):
        """(reference flux, zeropoint) for a native difference-imaging file.

        ``mag`` is the total brightness and ``dflux`` the difference flux of
        the same epoch, so

            mag = zp - 2.5*log10(ref - dflux)

        holds exactly and determines both unknowns.  Solved by least squares
        in magnitudes over the epochs with finite, usable values.  On
        MulensModel's native KB180003 set this returns ref = 1584.9 at
        zp = 28.000 for all three sites with a residual rms of 2.8e-5 mag.

        An explicit ``reference_flux``/``reference_mag`` short-circuits the
        solve; the zeropoint then comes from ``zp:`` (default 28.0, the
        pySIS convention).
        """
        import numpy as np
        from scipy.optimize import curve_fit

        label = self.names[i]
        override = self._reference_flux_override(i)
        if override is not None:
            return override, float((self.config[i] or {}).get("zp", 28.0))

        ok = np.isfinite(dflux) & np.isfinite(mag)
        if ok.sum() < 3:
            raise ValueError(
                f"[{self.prefix}:{label}] data_format: dia could not solve "
                f"for the reference flux: only {int(ok.sum())} epochs have "
                f"both a finite dflux and a finite mag. Set reference_flux "
                f"or reference_mag explicitly."
            )
        x, y = dflux[ok], mag[ok]

        def model(d, zp, ref):
            return zp - 2.5 * np.log10(np.maximum(ref - d, 1e-12))

        # ref must exceed max(dflux) for every epoch to have positive total
        # flux; start just above it, with the pySIS zeropoint.
        p0 = [28.0, float(np.max(x)) + max(1.0, float(np.ptp(x)) * 0.01)]
        try:
            (zp, ref), _ = curve_fit(model, x, y, p0=p0, maxfev=20000)
        except Exception as exc:
            raise ValueError(
                f"[{self.prefix}:{label}] data_format: dia could not solve "
                f"mag = zp - 2.5*log10(ref - dflux) for the reference flux "
                f"({type(exc).__name__}: {exc}). Set reference_flux or "
                f"reference_mag explicitly."
            ) from exc

        resid = float(np.std(y - model(x, zp, ref)))
        if not np.isfinite(ref) or ref <= float(np.max(x)):
            raise ValueError(
                f"[{self.prefix}:{label}] data_format: dia solved an "
                f"unusable reference flux ({ref!r}): it must exceed the "
                f"largest dflux ({float(np.max(x)):.6g}) so that every "
                f"epoch's total flux ref - dflux is positive. Set "
                f"reference_flux or reference_mag explicitly."
            )
        if resid > 0.05:
            logger.warning(
                "[%s:%s] data_format: dia -- the mag and dflux columns do "
                "not follow mag = zp - 2.5*log10(ref - dflux) tightly "
                "(residual rms %.3g mag). The recovered reference flux "
                "%.6g at zp %.4f may be wrong; check the file's column "
                "order, or set reference_flux explicitly.",
                self.prefix,
                label,
                resid,
                ref,
                zp,
            )
        else:
            logger.info(
                "[%s:%s] data_format: dia -- reference flux %.6g at "
                "zp %.4f (residual rms %.3g mag); total flux = ref - dflux.",
                self.prefix,
                label,
                ref,
                zp,
                resid,
            )
        return float(ref), float(zp)

    def _check_data_format(
        self,
        t,
        f,
        e,
        xyz_delta,
        ra_rad,
        dec_rad,
        label,
        geometry,
        data_format="magnitude",
    ):
        """Warn if data appears fainter at peak than at baseline.

        By the time this runs ``f`` is always the modeled observable, flux
        (magnitude files have already been converted).  A valid microlensing
        event must show brightening (LARGER flux) near peak.  If the data
        instead grow fainter, either:
          - data_format is 'magnitude' but the data are really in flux units
            (values rise at peak, so 10**(-0.4 value) falls), or
          - data_format is 'flux' but the data are really in magnitudes
            (values rise at peak in the file, and are taken at face value).

        Returns silently when the dataset has fewer than 3 epochs near baseline
        (e.g., Spitzer peak-only data) -- no comparison is possible there.

        Also returns silently when t_0, u_0 or t_E has no informed start
        (`geometry`, from ``_probe_bootstrap_geometry``): the check needs the
        event's timescale to find its peak, and used to substitute an
        invented 30 d, describing a different event (review 2.6.30(b)).
        The seed-0 peak-finder hints ARE informed, so the
        automated workflow -- the one most likely to have mislabelled a flux
        file -- still gets the check.
        """
        t0 = geometry("source.0.t_0")
        u0 = geometry("source.0.u_0")
        tE = geometry("mulensevent.0.t_E")
        if t0 is None or u0 is None or tE is None:
            return

        pi_E_N = geometry("mulensevent.0.pi_E_N", 0.0)
        pi_E_E = geometry("mulensevent.0.pi_E_E", 0.0)

        delta_e, delta_n = observer_sky_offset(xyz_delta, ra_rad, dec_rad)
        A_traj = self._pspl_magnification(
            t, delta_e, delta_n, t0, u0, tE, pi_E_N, pi_E_E
        )

        baseline_mask = A_traj < 1.1
        peak_mask = A_traj > 1.5

        # Skip if no baseline coverage (e.g., Spitzer peak-only data)
        if np.sum(baseline_mask) < 3 or np.sum(peak_mask) < 3:
            return

        f_baseline = float(np.median(f[baseline_mask]))
        f_peak = float(np.median(f[peak_mask]))

        # In flux, brighter = larger value.  Peak must be brighter.
        if f_peak < f_baseline:
            typical_err = float(np.median(np.abs(e)))
            n_sigma = (f_baseline - f_peak) / max(typical_err, 1e-30)
            if n_sigma > 10.0:
                if data_format == "flux":
                    logger.warning(
                        f"[{label}] Data appear fainter at peak "
                        f"({f_peak:.3g}) than at baseline "
                        f"({f_baseline:.3g}) in flux -- {n_sigma:.0f} sigma "
                        f"offset.  Data may actually be in magnitudes; "
                        f"remove 'data_format: flux' from the YAML config "
                        f"block if so."
                    )
                else:
                    logger.warning(
                        f"[{label}] After the mag->flux conversion, data "
                        f"appear fainter at peak ({f_peak:.3g}) than at "
                        f"baseline ({f_baseline:.3g}) -- {n_sigma:.0f} sigma "
                        f"offset.  Data may be in flux units; add "
                        f"'data_format: flux' to the YAML config block for "
                        f"this instrument if so."
                    )

    @staticmethod
    def _pspl_magnification(t, delta_e, delta_n, t0, u0, tE, pi_E_N, pi_E_E):
        """Point-source Paczynski magnification along one source trajectory.

        u_0 goes through ``physics.floor_u_0_value`` -- the same floor both
        magnification backends apply -- so the bootstrap's design matrix
        cannot contain the ``A = inf`` column an exactly central seed produces
        (``u_traj = 0`` at ``t = t_0``, which NNLS has no answer for).  This
        is also the expression ``_check_flux_direction`` uses; it carried a
        verbatim second copy, unfloored, until the floors were unified.

        ``tE`` is REQUIRED and must be positive.  A missing t_E used to become
        an invented 30 d and a negative one was abs()'d, so the NNLS flux
        split described a different event (review 2.6.30(b)); callers now
        take their "no geometry" branch instead, and a non-positive t_E is a
        seed bug, named here.
        """
        if tE is None or not float(tE) > 0.0:
            raise ValueError(
                f"flux bootstrap: t_E = {tE!r} is not a positive timescale; "
                f"the caller must skip the PSPL columns when t_E is unknown."
            )
        tE_safe = max(float(tE), 1.0)
        tau = (t - float(t0)) / tE_safe
        tau_p = tau - delta_n * float(pi_E_N) - delta_e * float(pi_E_E)
        u_p = (
            floor_u_0_value(u0)
            + delta_n * float(pi_E_E)
            - delta_e * float(pi_E_N)
        )
        u_traj = np.sqrt(tau_p**2 + u_p**2)
        return (u_traj**2 + 2.0) / (u_traj * np.sqrt(u_traj**2 + 4.0))

    @staticmethod
    def _binary_magnification_columns(t, n_src, _get, label=""):
        """Per-source magnification columns using the full binary-lens model.

        The flux bootstrap needs magnification columns that actually
        distinguish the sources.  For binary-source events the PSPL wings are
        nearly collinear (the trajectories differ mostly through their caustic
        features), which makes the NNLS decomposition degenerate; the binary
        model at the seeded (s, q, alpha) breaks that degeneracy.

        Returns a list of n_src columns, or None when the binary geometry is
        not specified (single-lens event, or missing per-source params) or
        MulensModel fails — the caller then falls back to the PSPL columns.
        Parallax is intentionally ignored (flux scales only).

        `_get` is ``_probe_bootstrap_geometry``'s reader, so a q derived from
        the body masses, or an s/alpha the engine can solve from what is
        known at stage 1, is seen here (review 2.6.30(b)).  A companion whose
        s/q/alpha is still not derivable -- e.g. s and alpha that only a
        Kepler orbit component produces, which the stage-1 engine cannot yet
        solve -- is LOGGED by name before degrading to the PSPL columns; it
        used to degrade silently.
        """
        if not _get.has_companion:
            return None
        # The companion geometry is LENS ELEMENT 1 (element 0 is the masked
        # primary; a lens.0.* read here would silently see nothing and drop
        # every event to the degenerate PSPL columns).
        s_val = _get("lens.1.s")
        if s_val is None:
            # A seed carries log_s (the sampled coordinate), not s.
            log_s = _get("lens.1.log_s")
            if log_s is not None:
                s_val = 10.0 ** float(log_s)
        q_val = _get("lens.1.q")
        alpha = _get("lens.1.alpha")
        missing = [
            name
            for name, val in (
                ("s (or log_s)", s_val),
                ("q", q_val),
                ("alpha", alpha),
            )
            if val is None
        ]
        if missing:
            logger.info(
                f"[{label}] flux bootstrap: the companion lens's "
                f"{', '.join(missing)} has no start derivable at stage 1, so "
                f"the flux split uses point-lens columns (degenerate for "
                f"overlapping binary-source trajectories)."
            )
            return None

        # Idempotent, and self-guarding if MulensModel is missing; op.py has
        # normally applied it already.  Repeated here because this is the
        # OTHER place exozippy calls MulensModel, and the fluxes bootstrapped
        # below land in the model's start values -- an unpatched call here
        # makes the whole build PYTHONHASHSEED-dependent.  Outside the try:
        # a failure to patch must not be swallowed into the silent
        # fall-back-to-PSPL path below.
        patch_mulensmodel_method_order()

        try:
            import MulensModel as mm

            cols = []
            for j in range(n_src):
                t0 = _get(f"source.{j}.t_0")
                u0 = _get(f"source.{j}.u_0")
                # ONE event t_E: the per-source fallback dance dissolved
                # with the split (design 5.2).
                tE = _get("mulensevent.0.t_E")
                if t0 is None or u0 is None or tE is None:
                    return None
                params = {
                    "t_0": float(t0),
                    # physics.U_0_FLOOR, the one floor both magnification
                    # backends use.  This was a third hard-coded copy of the
                    # clip (and, like them, engaged at every u_0 except 0).
                    "u_0": floor_u_0_value(u0),
                    "t_E": max(float(tE), T_E_FLOOR),
                    "s": max(float(s_val), S_FLOOR),
                    "q": clip_q_value(q_val, "lens.1.q (flux bootstrap)"),
                    "alpha": float(alpha),
                }
                rho = _get(f"source.{j}.rho")
                if rho is not None:
                    params["rho"] = max(float(rho), RHO_FLOOR)
                model = mm.Model(params)
                if rho is not None:
                    window = 3.0 * params["t_E"]
                    model.set_magnification_methods(
                        [params["t_0"] - window, "VBM", params["t_0"] + window]
                    )
                cols.append(np.asarray(model.get_magnification(t)))
            return cols
        except Exception as e:
            logger.warning(
                f"Binary-lens flux bootstrap failed ({e}); "
                "falling back to PSPL columns."
            )
            return None

    @staticmethod
    def _baseline_flux_fallback(f):
        """A strictly positive flux scale for one file, however odd the data.

        The median flux is the honest baseline; difference-imaging data can
        sit at (or below) zero, so fall back to the median |flux| and finally
        to 1.0 rather than returning something non-positive -- ``log_f_total``
        and every flux-scaled bound downstream need a positive number.
        """
        f = np.asarray(f, dtype=float)
        med = float(np.median(f))
        if med > 0.0 and np.isfinite(med):
            return med
        mad = float(np.median(np.abs(f)))
        if mad > 0.0 and np.isfinite(mad):
            return mad
        return 1.0

    def _q_source_declared(self, inst_idx):
        """(start, lower, upper) of ``q_source`` for one light curve.

        Read through ``ConfigManager.resolve`` -- defaults.yaml under any
        user entry for this element -- so the bootstrap's clip is the
        parameter's real support and its fallback start is the declared one.
        Never hand-copy these numbers here: the old [0.05, 0.95] clip and
        0.95 fallback were exactly such copies, and the clip was tighter than
        the declared [0, 2] (review 2.6.23).
        """
        cfg = self.config_manager.resolve(
            self.prefix, "q_source", element=inst_idx, names=self.names
        )
        return tuple(
            float(np.asarray(cfg[field], dtype=float).reshape(-1)[0])
            for field in ("initval", "lower", "upper")
        )

    @staticmethod
    def _clip_q_source(q, lower, upper, what):
        """``q`` as a float, clipped to the declared [lower, upper] -- loudly.

        A start outside two finite bounds is fatal at build (parameter.py),
        so the hint has to land inside; but a clip that moves it means the
        start no longer reproduces the split that ``what`` describes, and
        the user is told.  Inside the bounds the value passes untouched.
        """
        q = float(q)
        if lower <= q <= upper:
            return q
        clipped = float(np.clip(q, lower, upper))
        logger.warning(
            f"{what} give a source fraction q_source = {q:.6g}, outside its "
            f"declared bounds [{lower:g}, {upper:g}]; starting q_source at "
            f"{clipped:g} instead.  If the split is real, the q_source "
            f"bounds are too tight for this light curve."
        )
        return clipped

    @staticmethod
    def _nnls_free_blend(A, F):
        """Least squares for F = A @ f_s + f_b with f_s >= 0 and f_b FREE.

        The blend is split into two non-negative parts, f_b = b_plus -
        b_minus, so one NNLS call leaves it unconstrained while the source
        columns keep their sign constraint (review 2.6.23: the old design
        put a plain ones column under NNLS, forcing f_b >= 0, while the flux
        likelihood and defaults.yaml treat a negative blend as first class).
        When the best blend is positive the -1 column never enters the
        active set, so the solve is the old one.  Returns (f_s, f_b).
        """
        ones = np.ones(len(F))
        X = np.column_stack([A, ones, -ones])
        sol, _ = nnls(X, F)
        return sol[:-2], float(sol[-2] - sol[-1])

    def _estimate_flux_components(
        self, t, f_obs, xyz_au, ra_rad, dec_rad, inst_idx, geometry
    ):
        """Estimate (f_total, q_source, q_flux) for one instrument.

        f_total  = total baseline flux (all sources + blend)
        q_source = (Σ_j f_s,j) / f_total
        q_flux   = f_s,2 / f_s,1 (binary source; 1.0 for single source)

        With N sources the decomposition solves the linear model
        F(t) = sum_j f_s,j * A_j(t) + f_b by least squares with every
        f_s,j >= 0 and f_b FREE (``_nnls_free_blend``), where A_j is the PSPL
        magnification along source j's trajectory (source.<j>.t_0/u_0, the
        shared mulensevent t_E).
        The binary-lens perturbation is irrelevant here — we only need flux
        scales, not a precise model.

        A NEGATIVE BLEND IS FIRST CLASS (review 2.6.23): difference imaging
        and over-subtracted crowded fields produce one, the flux likelihood
        models it, and defaults.yaml bounds f_blend to [-1000, 1000] and
        q_source to [0, 2] (q_source > 1 IS a negative blend).  So nothing
        here clamps f_b at zero, and q_source is clipped only to its DECLARED
        bounds (``_q_source_declared``, read through ``resolve`` so a user
        bound counts too), never to an invented [0.05, 0.95].

        If the user has specified f_source and/or f_blend in their params file,
        those values are respected (they are TOTALS over sources), for any
        number of sources:
          - both given  -> skip estimation entirely, derive q from the ratio,
            unclipped (review 2.6.10: a stated 0.99 used to become 0.95)
          - f_source only → fix it and solve for f_blend via median residuals
          - f_blend only  → fix it and solve for f_source via NNLS
          - neither       -> solve everything (sources >= 0, blend free)

        ``f_obs`` is the file's flux (the modeled observable), so the
        design matrix acts on it directly -- there is no magnitude round trip.

        Falls back to the data median and the DECLARED q_source start when
        t_0, u_0 or t_E has no informed start, and says which (review
        2.6.30(b): a missing t_E used to be an invented 30 d).  When the
        free-blend solve finds source light but a non-positive TOTAL (a
        light curve with no baseline coverage leaves the blend unconstrained),
        it WARNS naming the file and re-solves with the blend held >= 0.  When
        no positive source flux survives, it WARNS naming the file and takes
        the data median and the configured q_source start -- that usually
        means the seeded geometry does not describe this band, a signal the
        old code discarded by returning q_source 0.95 without a word.

        The event geometry comes from `geometry`
        (``_probe_bootstrap_geometry``), so a start the user's entries only
        IMPLY -- t_E from masses and distances -- is seen, as are the seed-0
        peak-finder hints.  The instrument's own flux entries
        (f_source, f_blend, q_flux) are still read as written.
        """
        cm = self.config_manager
        n_src = self._n_sources
        label = self.config[inst_idx].get("file", f"{self.prefix}.{inst_idx}")
        q_start, q_lower, q_upper = self._q_source_declared(inst_idx)

        def _get_flux(param):
            # user_params keys are normalized to index form by
            # standardize_param_names.  User entry first, then the seed-0
            # hint, in user units.
            key = f"mulensinstrument.{inst_idx}.{param}"
            val = _raw_initval(cm.user_params, key)
            if val is None:
                val = cm.seed_start_value(key)
            return float(val) if val is not None else None

        q_flux_user = _get_flux("q_flux")
        q_flux_fallback = q_flux_user if q_flux_user is not None else 1.0

        t0 = geometry("source.0.t_0")
        u0 = geometry("source.0.u_0")
        tE = geometry("mulensevent.0.t_E")
        pi_E_N = geometry("mulensevent.0.pi_E_N", 0.0)
        pi_E_E = geometry("mulensevent.0.pi_E_E", 0.0)

        f_source_user = _get_flux("f_source")
        f_blend_user = _get_flux("f_blend")

        if f_source_user is not None and f_blend_user is not None:
            f_total = f_source_user + f_blend_user
            if not (f_total > 0.0) or not np.isfinite(f_total):
                # The one branch that returned the user's numbers unchecked.
                # f_total is the light curve's BASELINE FLUX SCALE, and every
                # consumer needs it strictly positive: log_f_total takes its
                # log10 (NaN, which resurfaces much later as the stage-6
                # missing-start error naming log_f_total rather than the two
                # entries that caused it), and _scale_flux_amplitudes
                # multiplies it into per-light-curve upper bounds (negative
                # bounds).  A negative blend is legitimate on its own -- that
                # is difference imaging -- but a negative TOTAL is a statement
                # that the star is not there.  Fall back to the data, loudly,
                # naming both entries (review 2.6.3).
                logger.warning(
                    f"{self.prefix}.{inst_idx}: f_source = {f_source_user!r} "
                    f"and f_blend = {f_blend_user!r} sum to a baseline flux of "
                    f"{f_total!r}, which is not positive.  Using the data's own "
                    "baseline instead for the flux scale; fix the two entries "
                    "if you meant them (a negative BLEND is fine, a negative "
                    "total is not)."
                )
                return (
                    self._baseline_flux_fallback(f_obs),
                    q_start,
                    q_flux_fallback,
                )
            # A ratio of two user STATEMENTS is not clipped to a comfortable
            # range (review 2.6.10: a stated 0.99 became a 0.95 start, and no
            # engine relation back-solves instrument fluxes, so nothing put
            # it back).  Only the declared support binds -- a start outside
            # it is fatal at build -- and that clip says so.
            q_source = self._clip_q_source(
                f_source_user / f_total,
                q_lower,
                q_upper,
                f"{self.prefix}.{inst_idx}: f_source = {f_source_user!r} and "
                f"f_blend = {f_blend_user!r}",
            )
            return f_total, q_source, q_flux_fallback

        missing = [
            name
            for name, val in (("t_0", t0), ("u_0", u0), ("t_E", tE))
            if val is None
        ]
        if missing:
            logger.info(
                f"[{label}] flux bootstrap: {', '.join(missing)} has no start "
                f"derivable at stage 1 (no user entry, seed or hint implies "
                f"one), so the flux scale is the data's median and "
                f"q_source takes its declared start, {q_start:g}."
            )
            return (
                self._baseline_flux_fallback(f_obs),
                q_start,
                q_flux_fallback,
            )

        delta_e, delta_n = observer_sky_offset(xyz_au, ra_rad, dec_rad)

        # One magnification column per source trajectory.  Prefer the full
        # binary-lens model (breaks the NNLS degeneracy between overlapping
        # source trajectories); fall back to PSPL columns.  Missing per-source
        # params (j > 0) degrade gracefully to the single-source estimate.
        A_cols = self._binary_magnification_columns(t, n_src, geometry, label)
        if A_cols is None:
            A_cols = [
                self._pspl_magnification(
                    t, delta_e, delta_n, t0, u0, tE, pi_E_N, pi_E_E
                )
            ]
            for j in range(1, n_src):
                t0_j = geometry(f"source.{j}.t_0")
                u0_j = geometry(f"source.{j}.u_0")
                tE_j = tE  # ONE event t_E now (the per-source fallback dance dissolved)
                if t0_j is None or u0_j is None:
                    logger.warning(
                        f"source.{j}.t_0/u_0 missing — flux bootstrap treats source {j} "
                        f"as blended into source 0."
                    )
                    continue
                A_cols.append(
                    self._pspl_magnification(
                        t, delta_e, delta_n, t0_j, u0_j, tE_j, pi_E_N, pi_E_E
                    )
                )

        A_traj = A_cols[0]
        # The observable already IS the flux the linear model predicts.
        F_obs = np.asarray(f_obs, dtype=float)

        def _decompose(free_blend):
            """(f_source, f_blend, q_flux) from the data; every source flux
            >= 0 and, with ``free_blend``, the blend unconstrained -- else
            held >= 0 (the second pass below)."""
            q_flux_est = q_flux_fallback
            if len(A_cols) > 1:
                # Multi-source: F = Sum_j f_s,j * A_j + f_b, every f_s,j >= 0
                # (the sign constraint is what breaks the degeneracy between
                # overlapping source trajectories).
                A_mat = np.column_stack(A_cols)
                if f_blend_user is not None:
                    # A user-supplied blend is a STATEMENT, not a starting
                    # guess: subtract it and drop the constant column, exactly
                    # as the single-source branch below does.  This branch
                    # used to leave the ones column in and never look at
                    # f_blend_user (review 1.6.2 -- the elif that reads it is
                    # reachable only for a single column), so a 2S fit with a
                    # pinned or seeded f_blend got its log_f_total and
                    # q_source hints from an NNLS estimate that contradicted
                    # the entry the user had written.
                    f_srcs, _ = nnls(A_mat, F_obs - f_blend_user)
                    f_blend_est = f_blend_user
                elif free_blend:
                    f_srcs, f_blend_est = self._nnls_free_blend(A_mat, F_obs)
                else:
                    X = np.column_stack([A_mat, np.ones(len(t))])
                    sol, _ = nnls(X, F_obs)
                    f_srcs, f_blend_est = sol[:-1], sol[-1]
                f_source_est = float(np.sum(f_srcs))
                if q_flux_user is None and f_srcs[0] > 1e-30:
                    q_flux_est = float(
                        np.clip(f_srcs[1] / f_srcs[0], 1e-3, 1e3)
                    )
                if f_source_user is not None and f_source_est > 0.0:
                    # honor the user's total source flux; keep the solved
                    # ratio.  (Unreachable with f_blend_user set -- both-user
                    # returns at the top -- but written against A_mat rather
                    # than a slice of X so it cannot silently mean the wrong
                    # columns.)
                    f_blend_est = float(
                        np.median(
                            F_obs
                            - A_mat @ (f_srcs * f_source_user / f_source_est)
                        )
                    )
                    if not free_blend:
                        f_blend_est = max(f_blend_est, 0.0)
                    f_source_est = f_source_user
            elif f_source_user is not None:
                f_blend_est = float(np.median(F_obs - f_source_user * A_traj))
                if not free_blend:
                    f_blend_est = max(f_blend_est, 0.0)
                f_source_est = f_source_user
            elif f_blend_user is not None:
                (f_source_est,), _ = nnls(
                    A_traj.reshape(-1, 1), F_obs - f_blend_user
                )
                f_blend_est = f_blend_user
            elif free_blend:
                (f_source_est,), f_blend_est = self._nnls_free_blend(
                    A_traj.reshape(-1, 1), F_obs
                )
            else:
                X = np.column_stack([A_traj, np.ones(len(A_traj))])
                (f_source_est, f_blend_est), _ = nnls(X, F_obs)
            return float(f_source_est), float(f_blend_est), q_flux_est

        # PASS 1: the blend free -- a negative blend is first class (review
        # 2.6.23; the solve used to force f_blend >= 0).
        f_source_est, f_blend_est, q_flux_est = _decompose(free_blend=True)
        f_total = f_source_est + f_blend_est

        # PASS 2, only when pass 1 found source light but a non-positive
        # TOTAL, which cannot be a flux scale (log_f_total takes its log10).
        # That is what light curves with no baseline coverage do -- follow-up
        # data taken only over the peak (ob09020's and ob07224's, A >= 14
        # throughout): the constant column is then nearly degenerate with
        # the magnification and the free blend extrapolates to a large
        # negative number.  Re-solve with the blend held >= 0, loudly.  The
        # data's median would be the wrong substitute there: it is the
        # MAGNIFIED flux, an order of magnitude above the baseline.  A user
        # f_blend is a statement and is never re-solved.
        if f_source_est > 0.0 and not f_total > 0.0 and f_blend_user is None:
            logger.warning(
                f"[{label}] flux bootstrap: with the blend free, this light "
                f"curve gives f_source = {f_source_est:.4g} and f_blend = "
                f"{f_blend_est:.4g}, a non-positive baseline flux "
                f"({f_total:.4g}) -- the data do not constrain the blend "
                f"(no baseline coverage?).  Starting from the split with the "
                f"blend held >= 0 instead; give f_blend for this file if you "
                f"know it."
            )
            f_source_est, f_blend_est, q_flux_est = _decompose(
                free_blend=False
            )
            f_total = f_source_est + f_blend_est

        if not (f_source_est > 0.0 and f_total > 0.0):
            # The data (user input), not an internal invariant: warn and take
            # q_source's configured start, never a silent substitute (review
            # 2.6.23 -- this returned q_source 0.95 without a word).  No
            # positive source flux means the seeded magnification does not
            # describe this light curve.
            logger.warning(
                f"[{label}] flux bootstrap: fitting the seeded magnification "
                f"(t_0={t0!r}, u_0={u0!r}, t_E={tE!r}) to this light curve "
                f"gives f_source = {f_source_est:.4g}, f_blend = "
                f"{f_blend_est:.4g} (total {f_total:.4g}); a usable split "
                f"needs a positive source flux and a positive total.  The "
                f"seeded geometry probably does not describe this file.  "
                f"Starting from the data's baseline flux and q_source's "
                f"configured start ({q_start:g}: your entry, else "
                f"defaults.yaml) instead; check the seed or give "
                f"f_source/f_blend for this file."
            )
            return (
                self._baseline_flux_fallback(f_obs),
                q_start,
                q_flux_est,
            )

        q_source = self._clip_q_source(
            f_source_est / f_total,
            q_lower,
            q_upper,
            f"[{label}] flux bootstrap: f_source = {f_source_est:.4g} and "
            f"f_blend = {f_blend_est:.4g}",
        )
        logger.debug(
            f"NNLS flux decomp: f_source={f_source_est:.3e}, f_blend={f_blend_est:.3e}"
            f" → q_source={q_source:.4f}, q_flux={q_flux_est:.4f}"
        )
        return f_total, q_source, q_flux_est

    def _abs_to_delta(self, t, xyz_abs):
        """Convert absolute barycentric positions to Skowron+2011 geocentric deviations.

        Converts to the Skowron+2011 geocentric inertial frame whose origin
        moves with Earth's position and velocity at t_0_par.  Any observer's
        position in this frame is:

        delta(t) = xyz_obs(t) - [xyz_earth(t_0_par) + v_earth(t_0_par)*(t - t_0_par)]

        For Earth: small deviation from straight-line motion (annual parallax).
        For Spitzer: ≈ Spitzer − Earth vector at t_0_par (satellite parallax offset,
        ~1–2 AU).  Yee+2014 §3: "Spitzer's offset from the centre of Earth is
        treated just as any other observatory."

        Delegates to ``MulensEvent.skowron_deviations`` -- the frame has one
        owner since the astrometric centroid shift (C30) consumes it too.
        """
        return self._event.skowron_deviations(t, xyz_abs)

    def get_observer_position(self, time, observer_location="earth"):
        """
        High-precision observer position dispatcher.
        Delegates to the shared exozippy.ephemeris module (major bodies,
        topocentric ground sites, and spacecraft ephemeris files).
        """
        return get_observer_position(time, observer_location=observer_location)

    def register_parameters(self, system):
        """Stage 3: Declare the manifest with bootstrapped fluxes."""
        f_total_init = np.array(self.fs_init)
        q_source_init = np.array(self.q_source_init)

        # Inject hints for derived f_source / f_blend so the relaxation engine
        # can resolve initial values.  Also push the data-estimated q_source and
        # log_f_total as PRECEDENCE_DERIVED_DATA hints so they override the defaults.yaml
        # values while still yielding to any explicit user override in params.yaml
        # (PRECEDENCE_USER wins — essential when restarting a fit from a previous MAP).
        for i in range(self.n_elements):
            q = q_source_init[i]
            f_source_guess = f_total_init[i] * q
            f_blend_guess = f_total_init[i] * (1.0 - q)
            self.config_manager.add_hint(
                f"{self.prefix}.{i}.f_source", f_source_guess
            )
            self.config_manager.add_hint(
                f"{self.prefix}.{i}.f_blend", f_blend_guess
            )
            self.config_manager.add_hint(
                f"{self.prefix}.{i}.q_source", q, rank=PRECEDENCE_DERIVED_DATA
            )
            self.config_manager.add_hint(
                f"{self.prefix}.{i}.log_f_total",
                float(np.log10(f_total_init[i])),
                rank=PRECEDENCE_DERIVED_DATA,
            )

        self.manifest = {
            "log_f_total": None,
            "q_source": None,
            "f_source": "default",
            "f_blend": "default",
        }
        # Multiplicative per-instrument error scale (shared base helper).
        self._register_noise(self.manifest)
        self._register_gp(self.manifest)
        self._register_robust(self.manifest)
        self._scale_flux_amplitudes(self.manifest, f_total_init)

        if self.total_detrend_cols > 0:
            self.manifest["detrend_coeffs"] = {
                "shape": (self.total_detrend_cols,)
            }

        # Binary source: one flux ratio q_flux = f_s2/f_s1 per instrument
        # (sources have different colors, so the ratio is chromatic).
        # _n_sources is set at the top of load_data (stage 1), before this.
        n_sources = self._n_sources
        if n_sources > 1:
            if n_sources > 2:
                raise NotImplementedError(
                    f"{self._n_sources}-source flux modeling is not yet "
                    "implemented: the per-instrument flux ratio q_flux only "
                    "handles 2 sources. The per-source magnification path is "
                    "generic; generalize the flux parameterization to add more."
                )
            self.manifest["q_flux"] = None
            for i in range(self.n_elements):
                self.config_manager.add_hint(
                    f"{self.prefix}.{i}.q_flux",
                    float(self.q_flux_init[i]),
                    rank=PRECEDENCE_DERIVED_DATA,
                )

        # Map each instrument to a Band instance by name.
        band_names = [c.get("band", None) for c in self.config]
        if hasattr(system, "band"):
            name_to_idx = {name: i for i, name in enumerate(system.band.names)}
            self.band_map = np.array(
                [
                    (
                        name_to_idx[n]
                        if (n is not None and n in name_to_idx)
                        else -1
                    )
                    for n in band_names
                ],
                dtype=int,
            )
            missing = [
                n for n in band_names if n is not None and n not in name_to_idx
            ]
            for n in missing:
                logger.warning(
                    f"Instrument references unknown band '{n}'; LD will be skipped."
                )
        else:
            self.band_map = np.full(self.n_elements, -1, dtype=int)

        # SED-tied photometric zeropoint, one per light curve.  Declared only
        # where there is an SED to tie to; it is DERIVED (see the defaults.yaml
        # note and _build_sed_flux_constraint's docstring), so it costs the
        # sampler nothing -- but it is a real Parameter, which is what makes
        # its unit, its prior, its LaTeX row and its start value behave like
        # every other parameter's instead of being read out of the config
        # block by hand.
        #
        # force_node: a purely derived Parameter is not tracked by default
        # (Parameter.build_pymc only emits a Deterministic when something is
        # sampled), and this one is worth reporting -- it is the calibration
        # of the light curve, and it was a named Deterministic before it
        # became a Parameter.
        #
        # A light curve with no SED prediction (no band:, or a band filter
        # the grid lacks) has no zeropoint at all: its element is INACTIVE
        # (held at a bookkeeping 0.0, reported nowhere) rather than
        # reporting a number nobody computed.  Until review 2.2.21 it
        # reported the prior center, i.e. the defaults.yaml 0.0 for a config
        # that stated none.
        if hasattr(system, "sed"):
            filter_keys = self._sed_filter_keys(system)
            has_pred = np.array([fk is not None for fk in filter_keys])
            self._resolve_zeropoint_systems(system)
            self._zp_tied = self._check_zeropoint_entries(has_pred)
            qualifier = [m or "" for m in self.zp_magsys]
            if np.all(has_pred):
                self.manifest["zeropoint"] = {
                    "expr_key": "default",
                    "force_node": True,
                    "unit_qualifier": qualifier,
                }
            elif np.any(has_pred):
                self.manifest["zeropoint"] = {
                    "expr_key": {"default": has_pred.copy()},
                    "mask": has_pred.copy(),
                    "inactive_value": 0.0,
                    "force_node": True,
                    "unit_qualifier": qualifier,
                }

            # Neighbor third light, per light curve, sampled only where the
            # blend tie is on: with the tie off f_blend is already free and
            # this parameter is exactly degenerate with it (the light curve
            # measures only the sum, finite-source or not).  With the tie on
            # it converts f_blend = f_lens_pred -- an identity the DC2018
            # Roman-fidelity sims violate in 80% of events (unrelated
            # line-of-sight stars dominate the blend; the lens supplies a
            # median 15% of it) -- into the correct inequality
            # f_blend >= f_lens_pred, positivity doing the work.  upper and
            # the selected elements' initval scale with the bootstrapped
            # baseline flux, the same reasoning as _scale_flux_amplitudes:
            # the file's flux zeropoint is arbitrary.  Pinned elements keep
            # the defaults.yaml 0.0 (NaN initval = leave alone).
            nb_selected = [
                bool(c.get("sed_constrains_blend", False)) for c in self.config
            ]
            if any(nb_selected):
                entry = pin_unselected(self.n_elements, nb_selected)
                overrides = dict(entry.get("overrides") or {})
                scale = np.asarray(f_total_init, dtype=float)
                overrides["upper"] = (2.0 * scale).tolist()
                overrides["initval"] = [
                    (0.05 * float(sc) if sel else float("nan"))
                    for sc, sel in zip(scale, nb_selected)
                ]
                entry["overrides"] = overrides
                self.manifest["neighbor_flux"] = entry
                # Preliminary whitening scale on the light curve's own flux
                # scale (the defaults.yaml 0.1 is meaningless against an
                # arbitrary flux zeropoint, and a scale >> span trips the
                # on-the-bound nudge).  The startup probe measures the real
                # scale; this only seeds it.
                for i, sel in enumerate(nb_selected):
                    if sel:
                        self.config_manager.add_scale_hint(
                            f"{self.prefix}.{i}.neighbor_flux",
                            0.1 * float(scale[i]),
                        )

            self._seed_source_star_from_flux(system)

    # Main-sequence dwarf locus for source seeding, from the shared
    # Mamajek-table reader (components/star/mamajek.py).
    _MS_LOCUS_CACHE = None

    @classmethod
    def _ms_locus(cls):
        """(teff K, radius Rsun, mass Msun) rows, brightest first.

        Backed by the shared Mamajek-table reader (components/star/
        mamajek.py, the getstar.pro port), capped at 2.2 Msun: a bulge
        source brighter than the dwarf-locus top is far likelier a giant
        (or a wrong zeropoint) than an early-type dwarf, and the bright
        guard should say so rather than seed one.
        """
        if cls._MS_LOCUS_CACHE is not None:
            return cls._MS_LOCUS_CACHE
        from exozippy.components.star.mamajek import read_mamajek

        table = read_mamajek(minmass=0.078)
        rows = [
            (float(te), float(r), float(m))
            for te, r, m in zip(table["Teff"], table["R_Rsun"], table["Msun"])
            if np.isfinite(te) and np.isfinite(r) and m <= 2.2
        ]
        if len(rows) < 10:
            raise RuntimeError(
                f"dwarf locus parsed to only {len(rows)} usable rows; "
                f"the Mamajek table or its reader changed."
            )
        rows.sort(key=lambda r: -r[2])  # brightest (most massive) first
        cls._MS_LOCUS_CACHE = rows
        return rows

    def _user_or_default(self, paths, field, default):
        """First user_params value for any spelling in ``paths``."""
        up = self.config_manager.user_params
        for path in paths:
            entry = user_entry(up, path)
            if entry is not None and entry.get(field) is not None:
                return float(entry[field])
        return default

    def _seed_source_star_from_flux(self, system):
        """Stage 3: seed the SOURCE star's stellar start from its own flux.

        Without this, a microlensing-only source starts as an exact solar
        clone (the star defaults), even though its apparent magnitude is
        MEASURED: the bootstrapped baseline f_source through the light
        curve's photometric zeropoint.  On DC2018 event 128 the solar-clone
        start let the polish/sampler walk into a swapped configuration
        (M-dwarf source, G-star lens) that the flux data disfavor.

        Chain: m_source = zp_mu - 2.5*log10(f_source_init), moved onto the
        BC grid's Vega system (minus m_AB - m_Vega on an AB light curve,
        issue #313 -- the same offset the zeropoint tie applies); assume the
        source is a main-sequence dwarf at the event's source distance
        (user initval, an existing engine hint, or 8 kpc -- microlensing
        sources are bulge stars by construction of the event rate); scan
        the approximate dwarf locus through the SED's own BC grid to find
        the (teff, radius, mass) whose predicted apparent magnitude
        matches; push PRECEDENCE_DERIVED_DATA hints (they override defaults and
        yield to the user, like every data-derived start).

        Only a light curve whose zeropoint the USER tied (mu AND sigma,
        ``_check_zeropoint_entries``) seeds: the zeropoint mu is the user's
        calibration statement, and with none there is no measured source
        magnitude to seed from.  (Until review 2.2.21 defaults.yaml's 0.0
        +/- 0.2 stood in for one.)  A linked mu has no number at stage 3 and
        does not seed either.  When the statement is wrong, the measured
        magnitude is absurd and the guard below skips seeding with a
        warning -- which doubles as the alarm that the zeropoint prior does
        not describe the file.

        A source BRIGHTER than the locus top is likely a giant (common for
        bulge sources): no seed, warned, the defaults stand.  Multi-source
        events are skipped -- f_source is the SUM and splitting it needs
        q_flux, which has its own hint path.
        """
        if self._n_sources > 1:
            return
        sed = system.sed
        filter_keys = self._sed_filter_keys(system)
        src = int(system.source.star_map[0])
        src_name = system.star.names[src]

        d_pc = self._user_or_default(
            [f"star.{src}.distance", f"star.{src_name}.distance"],
            "initval",
            None,
        )
        if d_pc is None:
            d_pc = self.config_manager.hints.get(f"star.{src}.distance")
        if d_pc is None:
            d_pc = 8000.0
            logger.info(
                f"source-flux seeding: no distance start for star "
                f"'{src_name}'; assuming a bulge source at {d_pc:.0f} pc."
            )
        av = self._user_or_default(
            [f"star.{src}.av", f"star.{src_name}.av"], "initval", 0.0
        )

        masses = []
        for i, name in enumerate(self.names):
            fk = filter_keys[i]
            if fk is None or not self._zp_tied[i]:
                continue
            zp_mu = self._user_or_default(
                [
                    f"{self.prefix}.{name}.zeropoint",
                    f"{self.prefix}.{i}.zeropoint",
                ],
                "mu",
                None,
            )
            if zp_mu is None:
                continue  # a linked mu: no number before the model exists
            f_src = float(self.fs_init[i]) * float(self.q_source_init[i])
            if f_src <= 0:
                continue
            # On the BC grid's (Vega) system, which m_pred below is in.
            m_meas = (
                zp_mu - 2.5 * np.log10(f_src) - float(self.zp_ab_minus_vega[i])
            )

            # Predicted apparent mag of each locus row through the SED's
            # own BC grid (teff, logg, feh=0, av), at the assumed distance.
            # ONE vectorized evaluate for the whole locus: a per-row
            # .eval() meant 61 pytensor compiles per band per prepare, and
            # under six xdist workers those serialized on the shared
            # compiledir's FileLock until pytest-timeout killed the
            # module fixture (ezsuite 15363115's deterministic errors).
            col = sed.filter_column(fk)
            locus = np.asarray(self._ms_locus(), dtype=float)
            teff_v, radius_v, mass_v = locus[:, 0], locus[:, 1], locus[:, 2]
            logg_v = 4.438 + np.log10(mass_v) - 2.0 * np.log10(radius_v)
            coords = np.column_stack(
                [
                    teff_v,
                    logg_v,
                    np.zeros_like(teff_v),
                    np.full_like(teff_v, av),
                ]
            )
            bc_v = np.asarray(sed.bc_interpolator.evaluate(coords).eval())[
                :, col
            ]
            lbol_v = radius_v**2 * (teff_v / 5772.0) ** 4
            mbol_v = 4.74 - 2.5 * np.log10(lbol_v)
            m_pred = mbol_v - bc_v + 5.0 * np.log10(d_pc) - 5.0

            if m_meas < m_pred[0]:
                logger.warning(
                    f"source-flux seeding ({name}): the measured source "
                    f"magnitude {m_meas:.2f} is BRIGHTER than the whole "
                    f"dwarf locus ({m_pred[0]:.2f} at its top, "
                    f"{d_pc:.0f} pc): a giant source, or a zeropoint "
                    f"prior that does not describe this file. Not seeding "
                    f"from this light curve."
                )
                continue
            if m_meas > m_pred[-1] + 3.0:
                logger.warning(
                    f"source-flux seeding ({name}): measured source mag "
                    f"{m_meas:.2f} is far fainter than the locus bottom "
                    f"({m_pred[-1]:.2f}): a sub-stellar source makes no "
                    f"sense, so the zeropoint prior likely does not "
                    f"describe this file. Not seeding from this light "
                    f"curve."
                )
                continue
            if m_meas > m_pred[-1]:
                logger.warning(
                    f"source-flux seeding ({name}): measured source mag "
                    f"{m_meas:.2f} is fainter than the locus bottom "
                    f"({m_pred[-1]:.2f}); seeding at the faint end."
                )
                m_meas = m_pred[-1]
            # BC wiggles can make m_pred locally non-monotonic; interp
            # over the magnitude-sorted pairs.
            order = np.argsort(m_pred)
            locus_masses = np.array([r[2] for r in self._ms_locus()])
            masses.append(
                float(np.interp(m_meas, m_pred[order], locus_masses[order]))
            )

        if not masses:
            return
        mass = float(np.median(masses))
        loci = np.array(self._ms_locus())[::-1]  # ascending mass for interp
        teff = float(np.interp(mass, loci[:, 2], loci[:, 0]))
        radius = float(np.interp(mass, loci[:, 2], loci[:, 1]))
        logger.info(
            f"source-flux seeding: star '{src_name}' starts as a "
            f"{mass:.2f} Msun / {radius:.2f} Rsun / {teff:.0f} K dwarf "
            f"(from f_source through the zeropoint at {d_pc:.0f} pc), "
            f"replacing the solar-clone default."
        )
        for param, val in (
            ("logmass", float(np.log10(mass))),
            ("teff", teff),
            ("radius", radius),
            ("teffsed", teff),
            ("radiussed", radius),
        ):
            self.config_manager.add_hint(
                f"star.{src}.{param}", val, rank=PRECEDENCE_DERIVED_DATA
            )

    # Flux-space images of the magnitude caps these amplitudes used to carry:
    # a 5 mag GP amplitude is a factor 10**(0.4*5) = 100 in flux, and a 10 mag
    # outlier scale a factor 10**(0.4*10) = 1e4.  Applied per light curve
    # against its own bootstrapped baseline flux, because a microlensing file's
    # flux zeropoint is arbitrary (10**(-0.4 m) ~ 1e-8 for a magnitude file,
    # O(1) or O(1e4) counts for difference imaging) and no single number in
    # defaults.yaml can serve both.
    _FLUX_AMPLITUDE_CAPS = {
        "gp_rot_sigma": (1.0e2, 1.85e-2),
        "gp_sho_sigma": (1.0e2, 1.85e-2),
        "out_scale": (1.0e4, 2.0e-1),
    }

    def _scale_flux_amplitudes(self, manifest, f_total_init):
        """Put the optional noise amplitudes on each light curve's flux scale.

        ``gp_*_sigma`` and ``out_scale`` are additive amplitudes in the
        observable's own units, which for this component is now flux in the
        FILE's arbitrary flux system.  Their defaults.yaml ``upper``/``initval``
        are therefore only ceilings; the usable per-element values are derived
        here from the bootstrapped baseline flux and installed through the
        ``overrides`` channel, i.e. layered UNDER the user's params file so an
        explicit bound or start still wins.

        The multipliers are the flux-space images of the magnitude caps these
        parameters carried before the switch (see ``_FLUX_AMPLITUDE_CAPS``).
        The ``initval`` is only a fallback -- ``Instrument._prepare_gp`` and
        ``_prepare_robust`` push data-driven hints (the median error bar,
        for both) which outrank it -- but it matters when a file has
        degenerate errors and those hints are skipped.  Likewise the
        ``upper`` here is only the flux-scaled ceiling: for a hogg file with
        usable errors ``_register_robust`` has already attached the 10x-
        median-error cap as an OPTION (review 8.6.3), which replaces this
        override's min-clip on those elements; this one stands on the
        degenerate-error files.
        """
        scale = np.asarray(f_total_init, dtype=float)
        for param, (cap, start) in self._FLUX_AMPLITUDE_CAPS.items():
            if param not in manifest:
                continue
            entry = dict(manifest[param] or {})
            overrides = dict(entry.get("overrides") or {})
            overrides["upper"] = (cap * scale).tolist()
            overrides["initval"] = (start * scale).tolist()
            entry["overrides"] = overrides
            manifest[param] = entry
        return manifest

    def _finite_source_limb_darkening(self, system):
        """(u1, u2, bandpass) for the magnification, or (None, None, None).

        ONE resolver, called by both `build_likelihood` and
        `compile_plotters`.  It used to live inline in `build_likelihood`
        only, so the plotters passed neither argument, `get_magnification_op`
        computed `effective_bandpass = None`, and every plotted/GUI model
        curve was the UNIFORM-source magnification while the likelihood fitted
        the limb-darkened one -- a discrepancy of up to several percent, and
        largest exactly where these plots are read (caustic crossings, the
        finite-source peak).  A helper rather than a cached value because both
        call sites want the live `band.u1` node; the warn-once flag is what
        keeps the multi-band notice from being printed twice.

        LD applies only when the lens is finite-source AND a Band component is
        wired to at least one of this instrument's light curves; a uniform
        source has no limb to darken.  Multiple distinct bands across one
        instrument's finite-source light curves are not yet supported -- the
        first band is used, and said so.

        u2 IS THE SECOND (QUADRATIC) COEFFICIENT, and it is returned rather
        than dropped because dropping it was a real defect: the magnification
        used to be a function of u1 alone, so a band declaring the DEFAULT
        `ld_law: quadratic` (Band._parse_ld_laws) had the wrong source profile
        AND one combination of its sampled (q1, q2) constrained by nothing but
        its prior.  Whether a given backend can honour it is not decided here
        -- see MulensEvent._resolve_quadratic_ld, which is where the backend is
        known and where the fallback is announced.

        The guard is on the MANIFEST, not on the law: with `ld_law: linear` on
        every band the parameter does not exist at all (Band.LD_MODE_TABLE via
        parameterization.mode_manifest omits a parameter no instance uses), so
        `"u2" in band.manifest` is the only safe test -- the same one
        transit.py and rm.py use (components/sed/sed.md).
        """
        if not (
            system.mulensevent.finite_source
            and hasattr(system, "band")
            and np.any(self.band_map >= 0)
        ):
            return None, None, None

        unique = sorted({int(b) for b in self.band_map if b >= 0})
        if len(unique) > 1 and not self._warned_multiband_ld:
            self._warned_multiband_ld = True
            logger.warning(
                "Multiple bands for finite-source instruments; using first band's u1."
            )
        band_idx = unique[0]
        u2 = (
            system.band.u2.value[band_idx]
            if "u2" in system.band.manifest
            else None
        )
        return (
            system.band.u1.value[band_idx],
            u2,
            system.band.names[band_idx],
        )

    def _model_flux(self, system, t, obs_pos, inst, resolve_method=False):
        """The ONE detrend-free flux-model expression, in each row's own
        instrument flux system.

        Both ``build_likelihood`` (the data's times, observer positions and
        ``inst_map_tensor``) and ``compile_plotters`` (a symbolic grid, its
        observer positions and one scalar instrument index) call it, so the
        plotted curve for instrument ``i`` IS instrument ``i``'s likelihood
        expression run on other times -- its own f_source / f_blend and, for
        a binary source, its own ``q_flux`` (review 1.6.11: the plot used to
        evaluate every observer group at the reference instrument, so a 2S
        fit's curve carried the reference's colour while each instrument's
        data carried its own).  The detrend term is NOT here: the likelihood
        applies it through ``Instrument._detrended_model`` and the plots take
        it off the data (``detrend_corrected``), instrument.md.

        Magnification: both the symbolic and Op paths take Skowron+2011
        geocentric deviations (AU); ``get_magnification_op`` dispatches.
        u1/u2/bandpass come from the one LD resolver.  ``resolve_method`` is
        the likelihood's: it lets ``resolve_auto_vbbl`` fix the backend's
        method list before each source's Op is built (the plot grid reuses
        that decision; the bracket spans the whole time axis, review 2.6.9).

        Flux: F = sum_j f_s,j A_j + f_b, with f_s,1 = f_s/(1+q_F) and
        f_s,2 = f_s q_F/(1+q_F) (q_F per instrument -- sources differ in
        colour).  No clamp: the model flux may legitimately be <= 0
        (f_blend may be negative, difference-imaging data live around zero),
        and the likelihood is Gaussian in flux, so nothing takes a log.
        """
        u1, u2, bandpass = self._finite_source_limb_darkening(system)
        n_src = self._n_sources
        A_per_source = []
        for j in range(n_src):
            if resolve_method:
                system.mulensevent.resolve_auto_vbbl(index=j)
            A_per_source.append(
                system.mulensevent.get_magnification_op(
                    t,
                    obs_pos,
                    system,
                    index=j,
                    u1=u1,
                    u2=u2,
                    bandpass=bandpass,
                )
            )

        fs = self.f_source.value[inst]
        fb = self.f_blend.value[inst]
        if n_src == 1:
            return fs * A_per_source[0] + fb
        qf_safe = pt.maximum(self.q_flux.value[inst], 0.0)
        return (
            fs / (1.0 + qf_safe) * A_per_source[0]
            + fs * qf_safe / (1.0 + qf_safe) * A_per_source[1]
            + fb
        )

    def build_likelihood(self, model, system):

        # 1. Constants
        t = pm.Data("mu_time", self.time)
        obs_flux = pm.Data("mu_obs_flux", self.flux)
        obs_err = pm.Data("mu_obs_err", self.err)

        # 2-3. The one flux-model expression (see _model_flux), on the data's
        # own times, observer positions and per-row instruments.
        model_flux = self._model_flux(
            system,
            t,
            self.observer_pos,
            self.inst_map_tensor,
            resolve_method=True,
        )

        # Optional detrending against extra data columns, through the shared
        # base mechanism: the coefficients are magnitude-space (airmass,
        # seeing, ...), so DETREND_SPACE = "magnitude" multiplies the flux by
        # 10**(-0.4 X.c) -- the same model as the additive magnitude
        # detrending used before, and well defined for negative fluxes -- and
        # the plots divide the same factor out of the data
        # (Instrument.detrend_corrected).  Block-diagonal, so coefficients
        # never mix across instruments.
        model_flux = self._detrended_model(model_flux, "mu_detrend")

        # 4. Error scaling & Likelihood (shared base helper: err * err_scale).
        # The shared dispatcher is the plain Normal unless a light curve asked
        # for a GP, in which case that curve gets a celerite2 marginal
        # likelihood around this same magnification model.
        sigma = self.total_sigma(obs_err)

        # Modeling-draft prose for the magnification model and the flux-space
        # likelihood, declared next to the model they describe.
        event = system.mulensevent
        if event.uses_op(0):
            if event.backend == "mulensmodel":
                mag_cite = (
                    r"computed with MulensModel \citep{Poleski:2019}, which "
                    r"wraps VBBinaryLensing \citep{Bozza:2010, Bozza:2018}"
                )
                get_collector(system).add_software("MulensModel")
                get_collector(system).add_software("VBBinaryLensing")
            else:
                mag_cite = (
                    r"computed with VBMicrolensing "
                    r"\citep{Bozza:2010, Bozza:2018, Bozza:2025}"
                )
                get_collector(system).add_software("VBMicrolensing")
        else:
            mag_cite = r"the analytic point-lens form \citep{Paczynski:1986}"
        get_collector(system).add(
            f"The microlensing magnification was {mag_cite}.",
            section="microlensing",
            key=f"{self.prefix}.magnification",
            rank=10,
        )
        get_collector(system).add(
            r"The microlens parallax is parameterized in the geocentric "
            r"frame of \citet{Gould:2004}, with observer positions "
            r"expressed as geocentric deviations following "
            r"\citet{Skowron:2011}; all sign conventions match "
            r"MulensModel \citep{Poleski:2019}.",
            section="microlensing",
            key=f"{self.prefix}.parallax_convention",
            rank=15,
        )
        get_collector(system).add(
            "The microlensing likelihood is Gaussian in flux (never in "
            "magnitudes): the model is linear in the per-instrument source "
            "and blend fluxes, photon-counting noise is approximately "
            "Gaussian in flux, and non-positive difference-imaging fluxes "
            "are retained as-is.",
            section="microlensing",
            key=f"{self.prefix}.flux_likelihood",
            rank=20,
        )

        self.add_observation_likelihood(
            f"{self.prefix}.model",
            mu=model_flux,
            sigma=sigma,
            observed=obs_flux,
            system=system,
        )

        # 5. SED-based source flux constraint (issue #18)
        if hasattr(system, "sed"):
            self._build_sed_flux_constraint(model, system)

    def _sed_source_indices(self, system):
        """Star indices whose blended SED flux is the microlensing source."""
        return [int(i) for i in system.source.star_map]

    def _sed_filter_keys(self, system):
        """Per light curve, the BC-grid filter key, or None where absent.

        None means "no SED tie for this element": either the light curve
        references no band: block, or its band's filter is not in the SED's
        BC grid.  Both are ordinary configurations, not errors.

        Cached: two callers ask (the stage-5 zeropoint expression and the
        stage-6 blend tie) and the diagnostics below should be said once.
        """
        cached = getattr(self, "_sed_filter_key_cache", None)
        if cached is not None:
            return cached
        sed = system.sed
        keys = [None] * self.n_elements
        for i, name in enumerate(self.names):
            band_idx = int(self.band_map[i])
            if band_idx < 0:
                logger.info(
                    f"mulensinstrument {name}: no band reference; skipping "
                    f"SED flux constraint."
                )
                continue
            filter_key = system.band.filter_mist[band_idx]
            if not sed.has_filter(filter_key):
                logger.warning(
                    f"mulensinstrument {name}: band filter '{filter_key}' "
                    f"is not in the SED's BC grid; skipping SED flux "
                    f"constraint."
                )
                continue
            keys[i] = filter_key
        self._sed_filter_key_cache = keys
        return keys

    def _resolve_zeropoint_systems(self, system):
        """Stage 3: each light curve's zeropoint magnitude system (#313).

        The SED predicts magnitudes on its BC columns' system (Vega), while
        a zeropoint is a calibration statement in whatever system the
        photometry was calibrated in -- the DC2018 challenge's 22.0 is AB.
        Sets ``self.zp_magsys`` (the resolved internal spelling per light
        curve, None where nothing resolves it) and ``self.zp_ab_minus_vega``
        (m_AB - m_Vega of the band's BC column on an AB light curve, 0.0
        otherwise).  The offset is applied in exactly one place,
        ``_predicted_mag_on_zp_system``.

        Resolution is the SED's own (``SED.filter_magsys``, the 1.9.1
        helper): a stated system is used as stated, an unstated one is the
        band filter's NATIVE system from the BC column record, and a filter
        with no recorded native system raises naming the light curve.  A
        light curve with no SED prediction has no zeropoint, so its stated
        system (if any) is kept for the log and nothing is resolved.
        """
        sed = system.sed
        filter_keys = self._sed_filter_keys(system)
        self.zp_magsys = []
        self.zp_ab_minus_vega = np.zeros(self.n_elements)
        for i, name in enumerate(self.names):
            stated = self._zp_magsys_stated[i]
            fk = filter_keys[i]
            if fk is None:
                self.zp_magsys.append(stated)
                continue
            where = f"mulensinstrument {name!r} (band filter {fk!r})"
            mag_system, offset = sed.filter_magsys(fk, stated, where)
            self.zp_magsys.append(mag_system)
            if mag_system == AB:
                self.zp_ab_minus_vega[i] = offset
            how = (
                "stated"
                if stated is not None
                else "the filter's native system"
            )
            logger.info(
                f"mulensinstrument {name}: zeropoint magnitude system "
                f"{mag_system} ({how}; the SED predicts Vega, so "
                + (
                    f"m_AB - m_Vega = {offset:+.4f} is added to the "
                    f"prediction before it meets the zeropoint)."
                    if mag_system == AB
                    else "no conversion)."
                )
            )

    def _user_states(self, field):
        """Per light curve: did the USER state zeropoint ``field``?

        A number (``Component.user_wrote_field``, all three spellings) or a
        link expression -- ``extract_links`` deletes a linked field from
        ``user_params``, so a linked ``mu`` is visible only in
        ``config_manager.links``.  READ-ONLY on both.
        """
        stated = self.user_wrote_field("zeropoint", field)
        links = self.config_manager.links
        for i, name in enumerate(self.names):
            for key in (
                f"{self.prefix}.zeropoint",
                f"{self.prefix}.{i}.zeropoint",
                f"{self.prefix}.{name}.zeropoint",
            ):
                if field in links.get(key, {}):
                    stated[i] = True
        return stated

    def _check_zeropoint_entries(self, has_pred):
        """Stage 3: the zeropoint tie is the USER's statement, or nothing.

        Review 2.2.21 (JDE 2026-09-30): "the user should specify a zero
        point and how much they trust it, we shouldn't decide that for
        them".  defaults.yaml carries NO mu/sigma, so the SED tie on a light
        curve applies exactly when the user states BOTH:

        * ``initval`` RAISES: the zeropoint is DERIVED (m_SED +
          2.5 log10 f_source), so a start value on it is a number the
          model never uses -- the user means a prior and must say how much
          they trust it.  This is also the bare ``zeropoint: x`` spelling,
          which the params boundary translates to ``{initval: x}``.
        * ``mu`` without ``sigma`` WARNS: a center with no width applies no
          tie, so it does nothing (``sigma`` without ``mu`` is already
          refused by ``config.validate_sigma_has_center``).
        * ``sigma: 0`` still raises, in ``_zeropoint_context``.

        Returns the per-light-curve tie mask (prediction AND mu AND sigma),
        which ``_seed_source_star_from_flux`` reads: a zeropoint nobody
        stated says nothing about the source's magnitude.
        """
        wrote_init = self.user_wrote_field("zeropoint", "initval")
        if np.any(wrote_init):
            bad = [n for n, w in zip(self.names, wrote_init) if w]
            raise ValueError(
                f"mulensinstrument zeropoint for {bad} is given an initval "
                f"(or a bare value). The zeropoint is DERIVED -- "
                f"m_SED + 2.5*log10(f_source) -- so a start value on it does "
                f"nothing. To tie a light curve's source flux to the SED, "
                f"state the calibration AND how much you trust it: "
                f"`mulensinstrument.<name>.zeropoint: {{mu: <zp>, sigma: "
                f"<mag>}}` (and `magsys:` on the light curve if it is not "
                f"the band filter's native system)."
            )
        has_mu = self._user_states("mu")
        has_sigma = self._user_states("sigma")
        for i, name in enumerate(self.names):
            if has_mu[i] and not has_sigma[i]:
                logger.warning(
                    f"mulensinstrument.{name}.zeropoint has a mu but no "
                    f"sigma: a zeropoint center with no width applies no SED "
                    f"tie, so it DOES NOTHING. Give a sigma (how much you "
                    f"trust the calibration, in mag) to tie f_source to the "
                    f"SED-predicted source magnitude."
                )
        tied = np.asarray(has_pred, dtype=bool) & has_mu & has_sigma
        for i, name in enumerate(self.names):
            if has_pred[i] and not tied[i]:
                logger.info(
                    f"mulensinstrument {name}: no zeropoint prior stated "
                    f"(mu and sigma), so f_source is NOT tied to the SED; "
                    f"the zeropoint is reported as derived "
                    f"({self.zp_magsys[i]})."
                )
        return tied

    def _predicted_mag_on_zp_system(self, star_indices, i, system):
        """SED-predicted magnitude of ``star_indices`` on light curve ``i``'s
        ZEROPOINT system.

        The ONE place the zeropoint's magnitude system is applied (issue
        #313): ``predict_blend_appmag`` returns the BC column's system
        (Vega), and a zeropoint in AB needs the AB magnitude, m_Vega +
        (m_AB - m_Vega).  Both consumers -- the derived zeropoint
        (``_zeropoint_context``) and the blend tie, which compares a
        prediction against that zeropoint -- read it here, so the two
        cannot disagree.  An exact 0.0 on a Vega light curve leaves the
        graph's value bit-identical.
        """
        m = system.sed.predict_blend_appmag(
            star_indices, self._sed_filter_keys(system)[i], system
        )
        offset = float(self.zp_ab_minus_vega[i])
        return m + offset if offset else m

    def add_parameter(self, model, param_name, system, context_nodes=None):
        """Inject the SED context nodes the derived zeropoint needs.

        ``m_source_pred`` (the SED-predicted source magnitude in each light
        curve's own band, on its zeropoint's magnitude system) is a
        cross-component forward-model node, not a manifest parameter, so the
        generic dep parser cannot reach it -- the same situation
        ``Orbit.add_parameter`` handles for its group masses.
        """
        if param_name == "zeropoint" and not context_nodes:
            context_nodes = self._zeropoint_context(system)
        return super().add_parameter(model, param_name, system, context_nodes)

    def _zeropoint_context(self, system):
        """Context nodes for the derived ``zeropoint`` expression.

        ``m_source_pred`` is one entry per light curve (aligned, so the
        expression may be sliced to the elements with a prediction); an
        element with none is INACTIVE and its entry is a finite placeholder
        the sliced expression never reads.
        """
        source_indices = self._sed_source_indices(system)
        filter_keys = self._sed_filter_keys(system)

        zp_cfg = self.config_manager.resolve(
            self.prefix,
            "zeropoint",
            shape=(self.n_elements,),
            names=self.names,
        )
        zp_sigma = zp_cfg.get("sigma")
        zp_sigma = (
            np.full(self.n_elements, np.nan)
            if zp_sigma is None
            else np.atleast_1d(np.asarray(zp_sigma, dtype=float))
        )

        m_pred = []
        for i, name in enumerate(self.names):
            if filter_keys[i] is None:
                m_pred.append(pt.constant(0.0))
                continue
            if zp_sigma[i] == 0:
                # Kept as a hard error rather than delegated to
                # Parameter.build_pymc, whose `sigma: 0` on a DERIVED element
                # is only a warning ("no effect") -- true in general, but here
                # it means the user asked for something the model cannot
                # express, and says so specifically.
                raise ValueError(
                    f"mulensinstrument.{name}.zeropoint has sigma=0. An "
                    f"exact zeropoint would make f_source deterministic "
                    f"given the SED; give a small nonzero sigma instead "
                    f"(e.g. 0.01)."
                )
            m_pred.append(
                self._predicted_mag_on_zp_system(source_indices, i, system)
            )

        return {"m_source_pred": pt.stack(m_pred)}

    def _build_sed_flux_constraint(self, model, system):
        """
        Tie each instrument's calibrated baseline source flux to the
        SED-predicted source magnitude (issue #18).

        The light curve's fluxes live in the data file's own flux system --
        for a magnitude file that is the system in which F = 10**(-0.4 m),
        for a flux file it is whatever the file uses -- so
        -2.5*log10(f_source) is the instrumental source magnitude and the
        arbitrary zeropoint is exactly what zp absorbs. A per-lightcurve
        zeropoint links it to the calibrated SED prediction:

            m_SED = -2.5*log10(f_source) + zp

        zp is the DERIVED Parameter ``mulensinstrument.zeropoint``,
        zp_i = m_SED + 2.5*log10(f_s,i) (physics.calc_zeropoint), with m_SED
        on the light curve's zeropoint magnitude system (``magsys:``, issue
        #313), and the USER's Gaussian prior on it -- there is no default
        (review 2.2.21); without both mu and sigma there is no tie -- is
        applied by Parameter.build_pymc's derived-with-sigma branch at
        stage 6.  This
        is the analytic marginalization of a zp nuisance tied exactly
        through the equation above; it adds no sampled dimension and leaves
        the (log_f_total, q_source) parameterization untouched.  sigma=0 is
        disallowed (an exact zp would make f_source deterministic given the
        SED; use a small sigma for a well-known calibration instead).

        SAMPLED vs DERIVED: zp is not a free parameter and must not become
        one.  Given f_source and the SED there is nothing left for it to do
        -- the equation above determines it exactly, and no data constrains
        it separately -- so sampling it would add a dimension identified
        only by its own prior.  What making it a Parameter buys is the
        generic machinery around it (unit conversion, resolve()'s
        mu-as-start rule, links, the LaTeX row, bound_scale), not a degree
        of freedom.

        For binary sources the constraint is on the TOTAL source flux
        against the SED-predicted blend of all source stars; a per-source
        flux-ratio (q_flux) constraint is future work.

        What remains here at stage 7 is the opt-in blend tie:
        `sed_constrains_blend: true` additionally ties f_blend to the
        SED-predicted blend of the modeled non-source stars through the same
        zeropoint (Gaussian potential with `sed_blend_sigma`, default 0.2
        mag). f_blend also contains any unrelated field stars, so leave this
        off unless the blend is understood.
        """
        source_indices = self._sed_source_indices(system)
        n_stars = system.star.n_elements
        other_indices = [i for i in range(n_stars) if i not in source_indices]
        filter_keys = self._sed_filter_keys(system)
        self._add_zeropoint_prose(system)

        for i, name in enumerate(self.names):
            if filter_keys[i] is None:
                continue
            if not self.config[i].get("sed_constrains_blend", False):
                continue
            if not other_indices:
                logger.warning(
                    f"mulensinstrument {name}: sed_constrains_blend is "
                    f"set but every modeled star is a source; skipping."
                )
                continue
            blend_sigma = float(self.config[i].get("sed_blend_sigma", 0.2))
            m_blend_pred = self._predicted_mag_on_zp_system(
                other_indices, i, system
            )
            fb_i = pt.maximum(self.f_blend.value[i], 1e-30)
            # Predicted blend in the INSTRUMENT's flux system: the modeled
            # non-source stars plus the fitted neighbor third light.  At
            # neighbor_flux = 0 the residual is algebraically identical to
            # the old m_blend_pred - m_blend_inst magnitude difference (same
            # square), so a config without the neighbor term builds the same
            # potential.  With it, positivity makes the tie one-sided: a
            # blend BRIGHTER than the lens is absorbed by f_nb, a blend
            # FAINTER than the lens still costs -- "the blend must contain
            # at least the lens's light".
            f_lens_inst = 10 ** (
                -0.4 * (m_blend_pred - self.zeropoint.value[i])
            )
            f_pred = f_lens_inst + self.neighbor_flux.value[i]
            resid = 2.5 * pt.log10(pt.maximum(f_pred, 1e-30) / fb_i)
            pm.Potential(
                f"{self.prefix}.{name}.sed_blend_prior",
                -0.5 * (resid / blend_sigma) ** 2,
            )

    def _add_zeropoint_prose(self, system):
        """The SED-tie sentence, declared where the tie is built.

        Only for the light curves the USER tied (mu and sigma stated,
        review 2.2.21) -- an untied zeropoint is a reported number, not a
        modeling choice -- and the AB clause only when one of them is AB
        (issue #313), so a Vega-only fit's draft carries no conversion
        sentence it did not use.
        """
        from ...outputs.prose import join_names
        from ...outputs.texutils import latex_escape

        tied = [i for i in range(self.n_elements) if self._zp_tied[i]]
        if not tied:
            return
        names = join_names(latex_escape(self.names[i]) for i in tied)
        text = (
            "The baseline source flux $F_S$ of each light curve with a "
            "stated photometric zeropoint prior (" + names + ") is tied to "
            "the SED-predicted source magnitude in its band through that "
            "zeropoint $z$, $m_{\\rm SED} = z - 2.5\\log_{10} F_S$, with "
            "the Gaussian prior on $z$ listed in the parameter table."
        )
        ab = [i for i in tied if self.zp_magsys[i] == AB]
        if ab:
            ab_names = join_names(latex_escape(self.names[i]) for i in ab)
            text += (
                " The zeropoints of " + ab_names + " are on the AB system "
                r"\citep{Oke:1983}, so the Vega-referenced SED prediction is "
                "converted with a per-filter $m_{\\rm AB} - m_{\\rm Vega}$ "
                "computed from the same filter transmission curve, flux "
                "weighting and Vega zero point as that filter's bolometric "
                "corrections."
            )
        get_collector(system).add(
            text,
            section="microlensing",
            key=f"{self.prefix}.zeropoint_tie",
            rank=25,
        )

    def compile_plotters(self, model, system):
        """Compile fast PyTensor functions for the lightcurve."""
        t_input = pt.vector("mu_t_input")
        obs_pos_input = pt.dmatrix("obs_pos")
        inst_idx = pt.iscalar("mu_inst_idx")

        param_symbols = [p.value for p in system.plot_params]

        # The model in instrument inst_idx's own FLUX system -- the very
        # builder build_likelihood scores against the data (_model_flux), so
        # the (u1, u2, bandpass) resolution (review 1.6.1: the plot once drew
        # the UNIFORM-source magnification for a limb-darkened fit) and the
        # per-instrument q_flux (1.6.11) cannot split.  It stops here:
        # the conversion to the plotted delta-magnitude is done in numpy by
        # plot_data, through the very `_flux_to_mag` the data traces go
        # through, so the model curve and the points it is drawn over cannot
        # disagree about what a non-positive flux means.  This graph used to
        # end in `-2.5*log10(maximum(A_eff, 1e-30))`, i.e. it kept the ~75 mag
        # spike the data path had already replaced with a gap (review 1.6.4):
        # a posterior draw with f_blend < -f_source*A is a real possibility in
        # heavy-negative-blending difference imaging, and the honest picture of
        # it is a break in the curve, not a spike off the bottom of the axis.
        #
        # The GP conditional mean is additive in this same space, which is why
        # the "physical + GP" curve is built on it too (see plot_data).  The
        # detrend term is not: it is per observation, and plot_data divides
        # it out of the data instead (Instrument.detrend_corrected).
        model_flux = self._model_flux(system, t_input, obs_pos_input, inst_idx)

        # Retained symbolically so plot_data can walk the graph for
        # param_deps (the evaluator skips components whose specs declare no
        # dependency on a moved slider -- empty deps would freeze the GUI's
        # microlensing charts in live mode).
        self._model_flux_node = model_flux

        self._compiled_model_flux = pytensor.function(
            inputs=[t_input, obs_pos_input, inst_idx] + param_symbols,
            outputs=model_flux,
            on_unused_input="ignore",
        )

        # Full per-instrument f_source / f_blend vectors at a given point, used
        # by plot() to rescale every data set onto the reference instrument's
        # flux system (peg all data + model to data set 0).
        self._compiled_flux = pytensor.function(
            inputs=param_symbols,
            outputs=[self.f_source.value, self.f_blend.value],
            on_unused_input="ignore",
        )

        # Per-file GP conditional-mean evaluators (no-op without a gp: key).
        self._compile_gp_plotters(system)

    # ------------------------------------------------------------------
    # Shared data preparation. The matplotlib plot() path (via
    # plotrender.plot_via_specs) and the GUI both consume plot_data(), so
    # there is a single description of the lightcurve chart.
    # ------------------------------------------------------------------
    def _seed_param(self, base_param):
        """t_0/t_E seed from the solved config (for the model time grid).

        t_0 is the source component's; t_E the event's.  Index form only:
        both forms standardize now that the instances carry real names, so
        the pre-split name-form fallback (which existed for the borrowed-
        name filing bug) is gone.
        """
        cm = self.config_manager
        owner = "source" if base_param == "t_0" else "mulensevent"
        d = user_entry(cm.user_params, f"{owner}.0.{base_param}")
        if d is not None:
            return d.get("initval")
        return None

    def _model_time_grid(self):
        """(t_model, t0, tE): +/-5 tE around t_0 when known, else data span."""
        t0 = self._seed_param("t_0")
        tE = self._seed_param("t_E")
        if t0 is not None and tE is not None:
            t_model = np.linspace(t0 - 5.0 * tE, t0 + 5.0 * tE, 2000).astype(
                np.float64
            )
        else:
            t_model = np.linspace(
                self.time.min(), self.time.max(), 2000
            ).astype(np.float64)
        return t_model, t0, tE

    def _observer_groups(self):
        """Unique observer_location strings and their instrument mapping.

        Model lines are one per unique observer_location: multiple earth
        instruments share one model curve (parallax between terrestrial sites
        is negligible unless a specific site is given, in which case
        each site is a distinct string).
        """
        unique_observers = []
        obs_to_inst = {}
        for i in range(self.n_elements):
            obs_loc = self.config[i].get("observer_location", "earth")
            if obs_loc not in obs_to_inst:
                unique_observers.append(obs_loc)
                obs_to_inst[obs_loc] = i
        inst_obs_loc = {
            i: self.config[i].get("observer_location", "earth")
            for i in range(self.n_elements)
        }
        return unique_observers, obs_to_inst, inst_obs_loc

    def _plot_model_layout(self, t_model):
        """Which instruments' OWN model curves the chart draws.

        ``[(i, name, grid_mask), ...]``: instrument ``i``'s likelihood
        expression (``_model_flux`` at ``inst=i``, with its own observer
        positions) is evaluated on ``t_model[grid_mask]`` and mapped onto
        the reference flux system by the SAME affine map as ``i``'s data
        (``_flux_alignment``'s ``align``).  So every plotted curve is the
        model some instrument's data are scored against, drawn exactly
        where those data are drawn -- never one instrument's model under
        another's data (review 1.6.11).

        Single source: after alignment instrument ``i``'s model is
        ``f_s,ref * A(t; observer) + f_b,ref`` whatever ``i`` is, so ONE
        curve per observer location represents all of them (the reference
        instrument where it sits at that location, else the first one
        there), over the whole grid -- the curves every shipped single-
        source chart has always drawn, under the same names.

        Binary source: ``q_flux`` is per instrument BY DESIGN (the sources
        differ in colour), so after alignment each instrument's curve is
        ``f_s,ref * A_eff,i(t) + f_b,ref`` with its own blend of the two
        source magnifications.  The reference instrument draws the chart's
        ``model`` over the whole grid; every other instrument draws
        ``<name> model`` over its own data span only (as RV's per-instrument
        curves do), so a seven-instrument chart is not seven full curves.
        The layout depends only on the config and the data, never on the
        point, so a live GUI eval keeps the same trace names.
        """
        ref_idx = self._reference_index()
        unique_observers, obs_to_inst, inst_obs_loc = self._observer_groups()
        full = np.ones(t_model.shape, dtype=bool)
        if self._n_sources == 1:
            layout = []
            for obs_loc in unique_observers:
                i = (
                    ref_idx
                    if inst_obs_loc[ref_idx] == obs_loc
                    else obs_to_inst[obs_loc]
                )
                name = (
                    f"model ({obs_loc})"
                    if len(unique_observers) > 1
                    else "model"
                )
                layout.append((i, name, full))
            return layout
        layout = [(ref_idx, "model", full)]
        for i in range(self.n_elements):
            if i == ref_idx:
                continue
            t_i = self.time[self.inst_map == i]
            span = (t_model >= t_i.min()) & (t_model <= t_i.max())
            if np.any(span):
                layout.append((i, f"{self.names[i]} model", span))
        return layout

    @staticmethod
    def _flux_to_mag(f):
        """Magnitudes of a flux array; NaN where the flux is not positive.

        Only ever used for DISPLAY.  The likelihood never calls this: a
        magnitude is undefined for the non-positive fluxes difference imaging
        produces, and the old code's clamp turned those points into ~75 mag
        spikes that both entered the fit and wrecked the plot's y axis.  NaN
        is what both renderers already skip.
        """
        f = np.asarray(f, dtype=np.float64)
        out = np.full(f.shape, np.nan)
        pos = f > 0.0
        out[pos] = -2.5 * np.log10(f[pos])
        return out

    def _flux_alignment(self, param_values):
        """Reference flux system and the aligner onto it.

        Peg everything to the reference data set's flux system (the first
        instrument by default, or one flagged 'reference: true').  Each
        instrument fits its own f_source/f_blend, so a raw delta-mag per
        instrument puts data on N scales.  Instead we recover each point's
        magnification with that instrument's own (f_source_i, f_blend_i) and
        re-inject it into the reference system (f_source_ref, f_blend_ref):
          A_obs = (F_i - f_blend_i) / f_source_i
          F_aln = f_source_ref * A_obs + f_blend_ref
        so all data lands on the reference scale.  The model curves go
        through the SAME map: each is some instrument i's own model flux,
        aligned with i's own (f_source_i, f_blend_i) exactly as i's data are
        (``_plot_model_layout``).  Using the plotted point's fitted
        fluxes keeps the alignment tied to the model rather than a stage-1
        estimate.

        In flux this map is AFFINE -- F_aln = (fs_ref/fs_i)*(F_i - fb_i) +
        fb_ref -- so errors propagate by a single symmetric factor and the GP
        conditional mean, being additive in flux, may be added to the model in
        instrument i's own system before the map (where it is the GP the
        likelihood fitted) or scaled by fs_ref/fs_i after it, equivalently.
        The remaining nonlinearity is purely presentational: the plot is drawn
        in delta-magnitudes, which is the convention microlensing light curves
        are read in, so ``align`` converts at the very end and returns NaN for
        any point whose aligned flux is not positive.
        """
        ref_idx = self._reference_index()
        fs_vec, fb_vec = self._compiled_flux(*param_values)
        fs_vec = np.atleast_1d(np.asarray(fs_vec, dtype=np.float64))
        fb_vec = np.atleast_1d(np.asarray(fb_vec, dtype=np.float64))
        fs_ref = max(float(fs_vec[ref_idx]), 1e-30)
        fb_ref = float(fb_vec[ref_idx])
        baseline_ref = -2.5 * np.log10(max(fs_ref + fb_ref, 1e-30))

        def align_flux(flux_arr, i):
            """Map instrument-i fluxes onto the reference flux system."""
            fs_i = max(float(fs_vec[i]), 1e-30)
            fb_i = float(fb_vec[i])
            F = np.asarray(flux_arr, dtype=np.float64)
            A_obs = (F - fb_i) / fs_i
            return fs_ref * A_obs + fb_ref

        def align(flux_arr, i):
            """Instrument-i fluxes -> reference-system delta-magnitudes."""
            return self._flux_to_mag(align_flux(flux_arr, i)) - baseline_ref

        return {
            "ref_idx": ref_idx,
            "fs_vec": fs_vec,
            "fb_vec": fb_vec,
            "align": align,
            "align_flux": align_flux,
            "baseline_ref": baseline_ref,
        }

    def plot_data(self, system, point=None):
        """GUI/PDF charts: the aligned delta-mag lightcurve, plus a zoom
        copy (x_range +/-3 tE) when t_0/t_E seeds are known.

        The chart is drawn in magnitudes even though the fit is in flux -- that
        is the convention these curves are read in -- so any point whose (in
        general aligned) flux is not positive comes back as NaN and is simply
        not drawn.  With point=None each instrument's data are returned in its
        own system (no fitted fluxes exist to align them onto one scale).
        See Component.plot_data and chart.Chart.
        """
        from exozippy.chart import Chart, Trace

        comp_id = {"yaml_key": self.prefix, "instance": None}
        sysname = getattr(system, "name", "")
        title = f"Microlensing photometry: {sysname}"

        def _data_style(i):
            # The historical plot used small dots for the (typically dense)
            # photometry; keep that unless the user configured a marker.
            style = self._data_trace_style(i)
            style.setdefault("marker", ".")
            return style

        if point is None:
            traces = []
            for i in range(self.n_elements):
                mask = self.inst_map == i
                # The fit is in flux, but the chart stays in magnitudes (the
                # convention these curves are read in).  Points whose flux is
                # not positive have no magnitude and are dropped as NaN rather
                # than clamped to a ~75 mag spike.
                f_i = self.flux[mask]
                e_i = self.err[mask]
                mag_i = self._flux_to_mag(f_i)
                traces.append(
                    Trace(
                        name=self.names[i],
                        role="data",
                        kind="scatter",
                        x=self.time[mask],
                        y=mag_i,
                        yerr=np.vstack(
                            [
                                mag_i - self._flux_to_mag(f_i + e_i),
                                self._flux_to_mag(f_i - e_i) - mag_i,
                            ]
                        ),
                        style=_data_style(i),
                    )
                )
            return [
                Chart(
                    id=f"{self.prefix}.lightcurve",
                    component=comp_id,
                    title=title,
                    xlabel="Time [BJD]",
                    ylabel="mag",
                    traces=traces,
                    y_inverted=True,
                    meta={
                        "file_tag": "mulens",
                        "figsize": (12, 6),
                        # Same caption as the model-bearing spec below: the
                        # modes-CLI paper rebuild collects figures from the
                        # data-only specs.
                        "caption": (
                            "Microlensing light curve. All instruments are "
                            "aligned onto the reference instrument's flux "
                            "system and shown in magnitudes; non-positive "
                            "aligned fluxes are not drawn."
                        ),
                    },
                )
            ]

        t_model, t0, tE = self._model_time_grid()
        unique_observers, _, inst_obs_loc = self._observer_groups()
        # Skowron geocentric deviations for each unique observer over the
        # model grid -- the single obs_pos convention both the symbolic PSPL
        # path and the MulensModel/VBM Ops consume.
        obs_model_pos = {
            obs_loc: self._abs_to_delta(
                t_model,
                self.get_observer_position(t_model, observer_location=obs_loc),
            )
            for obs_loc in unique_observers
        }
        param_values = self._point_to_plot_params(point, system)
        aln = self._flux_alignment(param_values)
        align = aln["align"]

        node = getattr(self, "_model_flux_node", None)
        # The detrend correction and the GP conditional mean reach the chart
        # in NUMPY (detrend_corrected / gp_mean_on_grid), so the graph walk
        # cannot see their parameters; without these a GUI slider on either
        # would never refresh the chart.
        deps = self._model_trace_param_deps(node, system)
        deps = deps + [
            lbl
            for lbl in self.detrend_dep_labels() + self.gp_dep_labels()
            if lbl not in deps
        ]

        traces = []
        for i, name, grid_mask in self._plot_model_layout(t_model):
            obs_loc = inst_obs_loc[i]
            t_i = t_model[grid_mask]
            try:
                # Instrument i's own likelihood expression (its f_source,
                # f_blend and q_flux; its observer's magnification, so
                # parallax between sites is preserved), mapped onto the
                # reference flux system by the same `align` as i's data.  The
                # model comes back as a FLUX and is converted through the same
                # `_flux_to_mag` + `baseline_ref` the data traces use -- so a
                # non-positive model flux becomes a NaN gap, exactly as a
                # non-positive datum does, instead of the ~75 mag spike the
                # old in-graph 1e-30 clamp drew (review 1.6.4).
                y_model = align(
                    self._compiled_model_flux(
                        t_i,
                        obs_model_pos[obs_loc][grid_mask],
                        i,
                        *param_values,
                    ),
                    i,
                )
            except Exception as e:
                logger.warning(f"Model eval failed for '{name}': {e}")
                continue
            traces.append(
                Trace(
                    name=name,
                    role="model",
                    kind="line",
                    x=t_i,
                    y=y_model,
                    node=node,
                    style={"series_index": int(i)},
                )
            )

        # One "physical + GP" curve per light curve that requested a GP.
        # The GP is additive in that instrument's own FLUX (that is the space
        # celerite2 conditioned in), so it is added to the model flux there and
        # the sum is then mapped onto the reference flux system.
        #
        # A light curve with detrend columns is the exception: celerite2
        # conditioned on r = y - m*f, the plotted points are y/f = m + r/f,
        # and the exact companion m + gp/f needs each datum's own f, which a
        # smooth grid cannot carry.  So for such a file the curve is drawn AT
        # ITS DATA EPOCHS, through the base mechanism
        # (Instrument.detrend_grid_exact / detrend_corrected_signal) -- every
        # plotted quantity stays exactly what the likelihood scored.
        # _gp_pred_on_grid is set in Instrument.__init__ (review 2.6.7).
        gp_files = sorted(self._gp_pred_on_grid)
        gp_at_data = None
        if any(not self.detrend_grid_exact(i) for i in gp_files):
            gp_at_data = self.detrend_corrected_signal(
                self.gp_mean_at_data(system, point), point
            )
        for i in gp_files:
            try:
                if self.detrend_grid_exact(i):
                    obs_pretty = obs_model_pos[inst_obs_loc[i]]
                    t_gp = t_model
                    flux_i = self._compiled_model_flux(
                        t_model, obs_pretty, i, *param_values
                    )
                    gp_i = self.gp_mean_on_grid(system, point, i, t_model)
                else:
                    rows = self.rows(i)
                    t_gp = self.time[rows]
                    flux_i = self._compiled_model_flux(
                        t_gp, self.observer_pos[rows], i, *param_values
                    )
                    gp_i = gp_at_data[rows]
                y_gp = align(np.asarray(flux_i, dtype=float) + gp_i, i)
            except Exception as e:
                logger.warning(
                    f"GP model eval failed for '{self.names[i]}': {e}"
                )
                continue
            traces.append(
                Trace(
                    name=f"{self.names[i]} model+GP",
                    role="model",
                    kind="line",
                    x=t_gp,
                    y=y_gp,
                    style={"series_index": int(i), "lw": 1.0},
                )
            )

        # The fitted detrend trend is per observation, so it comes off the
        # DATA -- divided out of the fluxes and their errors, the base
        # inverse of the factor the likelihood multiplied the model by
        # (Instrument.detrend_corrected) -- in each instrument's own flux
        # system, before the alignment.  The raw data without detrend columns.
        flux_corrected, err_corrected = self.detrend_corrected(
            self.flux, self.err, point
        )
        for i in range(self.n_elements):
            mask = self.inst_map == i
            flux_i = flux_corrected[mask]
            err_i = err_corrected[mask]
            delta_mag = align(flux_i, i)
            # Brighter (flux + err) -> smaller aligned mag (lower error bar).
            # NaN wherever the aligned flux is not positive.
            lo = delta_mag - align(flux_i + err_i, i)
            hi = align(flux_i - err_i, i) - delta_mag
            traces.append(
                Trace(
                    name=self.names[i],
                    role="data",
                    kind="scatter",
                    x=self.time[mask],
                    y=delta_mag,
                    yerr=np.vstack([lo, hi]),
                    style=_data_style(i),
                )
            )

        meta = {
            "file_tag": "mulens",
            "figsize": (12, 6),
            # The data traces are re-aligned onto the reference flux system
            # with the point's fitted f_source/f_blend (values AND asymmetric
            # errors), so live evals must re-ship them along with the models.
            "dynamic_data": True,
            "caption": (
                "Microlensing light curve with the best-fit model (red). "
                "All instruments are aligned onto the reference "
                "instrument's flux system and shown in magnitudes; "
                "non-positive aligned fluxes are not drawn."
                + (
                    " The two sources' flux ratio is fitted per instrument, "
                    "so each instrument's own model is drawn over its data "
                    "in that instrument's colour."
                    if self._n_sources > 1
                    else ""
                )
                + self.detrend_caption()
            ),
        }
        specs = [
            Chart(
                id=f"{self.prefix}.lightcurve",
                component=comp_id,
                title=title,
                xlabel="Time [BJD]",
                ylabel="mag - mag$_0$",
                traces=traces,
                param_deps=deps,
                y_inverted=True,
                meta=meta,
            )
        ]
        if t0 is not None and tE is not None:
            specs.append(
                Chart(
                    id=f"{self.prefix}.lightcurve_zoom",
                    component=comp_id,
                    title=f"{title} (zoom)",
                    xlabel="Time [BJD]",
                    ylabel="mag - mag$_0$",
                    traces=traces,
                    param_deps=deps,
                    y_inverted=True,
                    x_range=[t0 - 3.0 * tE, t0 + 3.0 * tE],
                    meta=dict(
                        meta,
                        file_tag="mulens_zoom",
                        caption=(
                            "As the previous figure, zoomed to "
                            r"$t_0 \pm 3\,t_E$."
                        ),
                    ),
                )
            )
        return specs

    def plot(self, system, points, filename_prefix="debug"):
        """Render the lightcurve (+zoom) PDFs from plot_data specs.

        The specs are the single description of these plots -- the GUI draws
        the same ones via plotly (see plotrender.py's module docstring).
        """
        from exozippy.plotrender import plot_via_specs

        plot_via_specs(self, system, points, filename_prefix=filename_prefix)
