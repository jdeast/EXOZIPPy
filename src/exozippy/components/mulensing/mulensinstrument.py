import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

import astropy.units as u
import pymc as pm
import pytensor
import pytensor.tensor as pt
from scipy.optimize import nnls

from exozippy.compat import patch_mulensmodel_method_order
from exozippy.components.instrument import Instrument
from exozippy.config import RANK_DERIVED_DATA
from exozippy.ephemeris import get_observer_position
from exozippy.outputs.prose import get_collector

from . import mmexofast_support
from .physics import (
    RHO_FLOOR,
    S_FLOOR,
    T_E_FLOOR,
    clip_q_value,
    floor_u_0_value,
)


def _raw_initval(data, default=None):
    """Read a raw ``user_params`` initval, collapsing a list (P4 multi-seed
    sampling) to its first (seed 0) entry.  Only meaningful before
    ConfigManager.finalize_user_params runs -- afterwards ``user_params``
    already holds the resolved seed-0 scalar."""
    if data is None:
        return default
    val = data.get("initval", default) if isinstance(data, dict) else data
    if isinstance(val, (list, tuple)):
        val = val[0] if val else default
    return val


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

    ``f_source``/``f_blend``/``log_f_total`` are unchanged: they live in the
    file's own flux system, which for a magnitude file is the system in which
    ``F = 10**(-0.4 m)`` (i.e. an instrumental zeropoint of 0), exactly as
    before.  ``err_scale`` is dimensionless and its meaning is unchanged.
    """

    # Multiplicative per-instrument error scale (not additive jitter).
    noise_model = "err_scale"
    prose_noun = "microlensing photometry"

    # Deps of the derived `zeropoint` that are injected as context nodes by
    # add_parameter below rather than resolved as manifest parameters, so
    # graph.py leaves them out of the build-order graph.
    context_dep_names = frozenset(
        {"m_source_pred", "zp_center", "sed_constrained"}
    )

    def __init__(self, config, config_manager):
        super().__init__(config, config_manager)
        self.label = "Microlensing Data"
        self.total_detrend_cols = 0

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
                "accepts": ["magnitude", "flux"],
                "required": False,
                "doc": (
                    "Photometry format of the data FILE. Default 'magnitude'. "
                    "The fit is always done in flux; magnitude files are "
                    "converted at load (F = 10**(-0.4 m), sigma_F = "
                    "ln(10)/2.5 * F * sigma_m). With 'flux', non-positive "
                    "fluxes are kept as-is -- nothing is clamped."
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
                "key": "sed_constrain_blend",
                "kind": "option",
                "accepts": [True, False],
                "required": False,
                "doc": (
                    "When an SED is present, also tie f_blend to the "
                    "SED-predicted flux. Default false."
                ),
            },
            {
                "key": "sed_blend_sigma",
                "kind": "option",
                "accepts": None,
                "required": False,
                "doc": (
                    "Gaussian width (mag) of the SED f_blend constraint when "
                    "sed_constrain_blend is set. Default 0.2."
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
        """Stage 1a: Load photometry and pre-calculate observer positions.

        Single-event assumption (enforced by Lens.__init__): index 0 is the
        only event, so the event-0 source, t0_par, and magnification are used
        throughout.
        """
        self.fs_init = []
        self.q_source_init = []
        self.q_flux_init = []  # per-instrument f_s2/f_s1 (binary source)
        self._raw_time_list = []
        blocks = self._concat_blocks()

        self._n_sources = int(system.lens.n_sources)

        # MMEXOFAST integration (masks + error factors + auto seeds) -- must
        # run before the file loop so excluded points never enter the arrays.
        self._resolve_mmexofast(system)

        # Source RA/Dec (degrees from resolve → radians for projection math).
        # Stashed for Lens._earth_vperp_en: the mu_helio -> mu_geo conversion
        # must project Earth's velocity with the same (ra, dec) the Skowron
        # deltas are projected with.
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
            # Shared reader: columns:, mask:, time_* conversion, then sort
            # before the observer positions are computed from t, so the
            # ephemeris rows stay aligned with the photometry.
            df = self._read_data(
                i,
                roles=("time", "flux" if fmt == "flux" else "mag", "err"),
                detrend=True,
            )
            t, f, e = (
                df.iloc[:, 0].values.astype(float),
                df.iloc[:, 1].values.astype(float),
                df.iloc[:, 2].values.astype(float),
            )

            if fmt != "flux":
                # Magnitudes -> flux.  Exact for the value; the error is the
                # first-order propagation evaluated at the OBSERVED flux, so
                # sigma stays a data constant (using the model flux instead
                # would make sigma a function of the parameters and bias the
                # fit toward faint models).
                f = 10.0 ** (-0.4 * f)
                e = (np.log(10.0) / 2.5) * f * np.maximum(e, 0.0)

            per_file.append((t, f, e, df))

        # Geocentric reference (Skowron+2011 convention): Earth's position and
        # velocity at t_0_par define the inertial frame.  All observer positions
        # are stored as deviations from this linear Earth trajectory so that
        # t_0/u_0 remain geocentric parameters.  Re-resolved here rather than
        # taken from Lens.__init__: MMEXOFAST seeds arrive in stage 1a (via
        # _resolve_mmexofast above), after the Lens snapshotted user_params,
        # and a reference epoch far from the data makes the linear Earth
        # extrapolation diverge (O(100) AU after ~20 yr), shearing tau/u by
        # O(deviation x pi_E).
        self._t0_par = self._resolve_t0_par_final(
            system, np.concatenate([f[0] for f in per_file])
        )
        system.lens.t0_par[0] = self._t0_par
        self._earth_pos_ref = self.get_observer_position(
            np.array([self._t0_par]), "earth"
        )[0]  # (3,) AU
        _dt = 0.5  # days for finite-difference velocity
        _ep = self.get_observer_position(
            np.array([self._t0_par + _dt]), "earth"
        )[0]
        _em = self.get_observer_position(
            np.array([self._t0_par - _dt]), "earth"
        )[0]
        self._earth_vel_ref = (_ep - _em) / (2.0 * _dt)  # AU/day

        # Median absolute position per instrument (used by Lens to detect parallax)
        self.inst_ref_pos = []

        # Pass 2: observer positions, flux bootstraps, and sanity checks.
        for i, (t, f, e, df) in enumerate(per_file):
            obs_loc = self.config[i].get("observer_location", "earth")
            xyz_abs = self.get_observer_position(t, observer_location=obs_loc)
            self.inst_ref_pos.append(np.median(xyz_abs, axis=0))

            xyz_delta = self._abs_to_delta(t, xyz_abs)

            f_total, q_source, q_flux = self._estimate_flux_components(
                t, f, xyz_delta, ra_rad, dec_rad, i
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
                data_format=self.config[i].get("data_format", "magnitude"),
            )

            self._raw_time_list.append(t)

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

        Lens.__init__ resolves t0_par from the lens config and user_params
        only; MMEXOFAST seeds arrive later (stage 1a, add_seed_hints), so
        the automated workflow -- whose params file deliberately omits the
        microlensing start values -- used to fall through to the 2450000.0
        default, parking the reference epoch decades before the data.

        Priority: explicit lens ``t0_par`` > user ``lens.0.t_0`` initval >
        MMEXOFAST seed t_0 > median data time.  Any of these keeps the
        linear Earth extrapolation within the season it is a good
        approximation for.
        """
        lens_config = system.lens.config[0]
        if "t0_par" in lens_config:
            return float(lens_config["t0_par"])
        cm = self.config_manager
        val = _raw_initval(cm.user_params.get("lens.0.t_0"))
        if val is None:
            val = cm.seed_start_value("lens.0.t_0")
        if val is not None:
            return float(val)
        t_med = float(np.median(all_times))
        logger.info(
            f"[{self.prefix}] No t0_par, lens t_0, or MMEXOFAST seed found; "
            f"anchoring the parallax reference epoch at the median data "
            f"time ({t_med:.2f})."
        )
        return t_med

    def _reject_time_spec_with_mmexofast(self, spec):
        """Refuse to mix MMEXOFAST seeding with a per-file time system.

        MMEXOFAST reads the raw data files itself, so its t_0 seeds (and
        the JSON's excluded_points/errfacs) are expressed in the files' own
        raw time system.  With a time_offset or a time_scale/time_frame
        conversion active, the model's times differ from the raw ones and
        the seeds would start the fit in the wrong time system -- an error
        that converges to a wrong answer rather than crashing.  Refuse
        loudly instead.
        """
        if self.has_nontrivial_time_spec:
            raise ValueError(
                f"[{self.prefix}] time_offset/time_scale/time_frame cannot "
                f"be combined with MMEXOFAST seeding (mmexofast: {spec!r}): "
                f"MMEXOFAST reads the raw files, so its t_0 seeds would be "
                f"in the raw time system, not the converted one. Either "
                f"pre-convert the data files, or set mmexofast: false and "
                f"provide start values for the microlensing observables."
            )

    def _resolve_mmexofast(self, system):
        """Stage-1a half of the MMEXOFAST integration.

        Three modes, keyed off the lens block's ``mmexofast`` entry:

        - explicit file path: the JSON's bad-data mask (``excluded_points``)
          and error factors (``errfacs``) are applied to this component's
          files; the seed hints are pushed by Lens at stage 2 as before. An
          absent file warns and skips; an unparseable one raises (see
          ``mmexofast_support.load_json``) rather than dropping the mask and
          the error factors along with the seeds.
        - absent (default) or ``auto``: when the params file lacks start
          values for the microlensing parameters (or always, for ``auto``),
          MMEXOFAST is run on the raw light curves -- renormalize_errors on,
          output cached at ``<prefix>_mmexofast.json`` -- and its seeds,
          masks and error factors are all consumed here.
        - ``false``: fully opts out.

        This lives on the instrument rather than Lens because the mask must
        exist before the photometry is read (load_data), and only this
        component knows its files; Lens owns the stage-2 seed path for
        explicit files, and both share mmexofast_support for the translation.
        """
        lens = getattr(system, "lens", None)
        if lens is None:
            return
        spec = lens.config[0].get("mmexofast") if lens.config else None
        if spec is False:
            return
        is_binary = lens.n_companions >= 1
        want_rho = bool(any(lens.finite_source))

        if isinstance(spec, str) and spec != "auto":
            # Explicit JSON: masks + error factors, and the seed hints too.
            # Lens re-pushes the same seeds at stage 2 (harmless, identical
            # content); pushing them HERE as well makes them visible to this
            # component's flux bootstrap (_estimate_flux_components), which
            # runs later in this same load_data call -- stage 2 would be too
            # late and the per-band flux decomposition would silently fall
            # back to median-flux / q_source=0.95.
            self._reject_time_spec_with_mmexofast(spec)
            data = mmexofast_support.load_json(spec)
            if data is not None:
                mmexofast_support.push_seed_hints(
                    data,
                    self.config_manager,
                    want_rho=want_rho,
                    is_binary=is_binary,
                    source=spec,
                )
        else:
            if spec != "auto" and mmexofast_support.user_hints_sufficient(
                self.config_manager, is_binary, want_rho
            ):
                return
            self._reject_time_spec_with_mmexofast(spec)
            prefix = system.config.get("prefix", "fitresults/planet")
            json_path = f"{prefix}_mmexofast.json"
            options = dict(lens.config[0].get("mmexofast_options") or {})
            data = mmexofast_support.run_or_load(
                json_path,
                self.files,
                coords=self._mmexofast_coords(system),
                fit_type="binary_lens" if is_binary else "point_lens",
                options=options,
            )
            if data is not None:
                mmexofast_support.push_seed_hints(
                    data,
                    self.config_manager,
                    want_rho=want_rho,
                    is_binary=is_binary,
                    source=json_path,
                )
        if data is None:
            return
        mmexofast_support.apply_excluded_points(
            data,
            self.files,
            self.mask_specs,
            self.prefix,
            robust_kinds=self.likelihood_kinds,
        )
        mmexofast_support.push_errfac_hints(
            data, self.files, self.prefix, self.config_manager
        )
        # Start values move no posterior, but the error renormalization and
        # any bad-data mask do -- so the draft must say where they came from.
        get_collector(system).add(
            "Starting values, per-instrument error renormalization "
            "factors, and bad-data masks for the microlensing light "
            "curves were derived with MMEXOFAST (in preparation).",
            section="microlensing",
            key=f"{self.prefix}.mmexofast",
            rank=30,
        )

    def _resolve_source_radec_deg(self, system):
        """Source star's sky position in degrees.

        Falls back to the (primary) lens star's ra/dec when the user never
        explicitly set the source's own -- source and lens are angularly
        coincident by construction, so params.yaml only needs to state the
        target coordinates once, on the lens.
        """
        source_ndx = int(system.lens.source_map[0])
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
            primary_lens_idx = next(
                (
                    idx
                    for (ctype, idx) in system.lens.lens_bodies[0]
                    if ctype == "star"
                ),
                None,
            )
            if primary_lens_idx is not None:
                ndx = primary_lens_idx

        return float(ra_all[ndx]), float(dec_all[ndx])

    def _mmexofast_coords(self, system):
        """Source-star coordinates as an 'hh:mm:ss dd:mm:ss' string, or None.

        Same resolve pathway load_data itself uses for the projection math;
        harmless under no_parallax (the default for the automatic run) but
        required if the user opts parallax back in via mmexofast_options.
        """
        try:
            from astropy.coordinates import SkyCoord

            ra_deg, dec_deg = self._resolve_source_radec_deg(system)
            return SkyCoord(ra_deg, dec_deg, unit="deg").to_string(
                style="hmsdms"
            )
        except Exception as e:
            logger.warning(
                f"Could not resolve source coordinates for MMEXOFAST: {e}; "
                f"running without coords."
            )
            return None

    def _check_data_format(
        self,
        t,
        f,
        e,
        xyz_delta,
        ra_rad,
        dec_rad,
        label,
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
        """
        cm = self.config_manager

        def _get(key, default=None):
            # User params first, then the seed-0 MMEXOFAST hints in user
            # units -- the same fallback _estimate_flux_components uses.
            # Without it this check silently did nothing in the automated
            # (mmexofast: auto) workflow, which is exactly the workflow
            # where the user typed the fewest start values and is therefore
            # most likely to have mislabelled a flux file as magnitudes.
            val = _raw_initval(cm.user_params.get(key), None)
            if val is None:
                val = cm.seed_start_value(key)
            return default if val is None else val

        t0 = _get("lens.0.t_0")
        u0 = _get("lens.0.u_0")
        tE = _get("lens.0.t_E")
        if t0 is None or u0 is None:
            return

        pi_E_N = _get("lens.0.pi_E_N", 0.0)
        pi_E_E = _get("lens.0.pi_E_E", 0.0)

        x, y, z = xyz_delta[:, 0], xyz_delta[:, 1], xyz_delta[:, 2]
        delta_e = -x * np.sin(ra_rad) + y * np.cos(ra_rad)
        delta_n = (
            -x * np.cos(ra_rad) * np.sin(dec_rad)
            - y * np.sin(ra_rad) * np.sin(dec_rad)
            + z * np.cos(dec_rad)
        )
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
        """
        tE_safe = max(abs(float(tE)), 1.0) if tE is not None else 30.0
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
    def _binary_magnification_columns(t, n_src, _get):
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
        """
        s_val = _get("lens.0.s")
        if s_val is None:
            # MMEXOFAST seeds carry log_s (the sampled coordinate), not s.
            log_s = _get("lens.0.log_s")
            if log_s is not None:
                s_val = 10.0 ** float(log_s)
        q_val = _get("lens.0.q")
        alpha = _get("lens.0.alpha")
        if s_val is None or q_val is None or alpha is None:
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
                t0 = _get(f"lens.{j}.t_0")
                u0 = _get(f"lens.{j}.u_0")
                tE = _get(f"lens.{j}.t_E", _get("lens.0.t_E"))
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
                    "q": clip_q_value(q_val, "lens.0.q (flux bootstrap)"),
                    "alpha": float(alpha),
                }
                rho = _get(f"lens.{j}.rho")
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

    def _estimate_flux_components(
        self, t, f_obs, xyz_au, ra_rad, dec_rad, inst_idx
    ):
        """Estimate (f_total, q_source, q_flux) for one instrument.

        f_total  = total baseline flux (all sources + blend)
        q_source = (Σ_j f_s,j) / f_total
        q_flux   = f_s,2 / f_s,1 (binary source; 1.0 for single source)

        With N sources the decomposition solves the linear model
        F(t) = Σ_j f_s,j · A_j(t) + f_b via NNLS, where A_j is the PSPL
        magnification along source j's trajectory (lens.<j>.t_0/u_0/t_E).
        The binary-lens perturbation is irrelevant here — we only need flux
        scales, not a precise model.

        If the user has specified f_source and/or f_blend in their params file,
        those values are respected (they are TOTALS over sources):
          - both given  → skip estimation entirely, derive q from the ratio
          - f_source only → fix it and solve for f_blend via median residuals
          - f_blend only  → fix it and solve for f_source via NNLS
          - neither       → solve everything via NNLS

        ``f_obs`` is the file's flux (the modeled observable), so the NNLS
        design matrix acts on it directly -- there is no magnitude round trip.

        Falls back to the data median / q=0.95 when t_0 or u_0 are absent.
        """
        cm = self.config_manager
        n_src = getattr(self, "_n_sources", 1)

        def _get(key, default=None):
            # User params first (they outrank everything), then the seed-0
            # MMEXOFAST hints in user units. Without the seed fallback the
            # automated workflow -- whose params file deliberately omits the
            # microlensing start values -- never sees a geometry here and
            # every band degrades to the median-flux / q_source=0.95 guess,
            # which badly mis-normalizes multi-band fits.
            val = _raw_initval(cm.user_params.get(key), None)
            if val is None:
                val = cm.seed_start_value(key)
            return default if val is None else val

        def _get_flux(param):
            # user_params keys are normalized to index form by standardize_param_names
            val = _get(f"mulensinstrument.{inst_idx}.{param}")
            return float(val) if val is not None else None

        q_flux_user = _get_flux("q_flux")
        q_flux_fallback = q_flux_user if q_flux_user is not None else 1.0

        t0 = _get("lens.0.t_0")
        u0 = _get("lens.0.u_0")
        tE = _get("lens.0.t_E")
        pi_E_N = _get("lens.0.pi_E_N", 0.0)
        pi_E_E = _get("lens.0.pi_E_E", 0.0)

        f_source_user = _get_flux("f_source")
        f_blend_user = _get_flux("f_blend")

        if f_source_user is not None and f_blend_user is not None:
            f_total = f_source_user + f_blend_user
            q_source = float(
                np.clip(f_source_user / max(f_total, 1e-30), 0.05, 0.95)
            )
            return f_total, q_source, q_flux_fallback

        if t0 is None or u0 is None:
            return self._baseline_flux_fallback(f_obs), 0.95, q_flux_fallback

        x, y, z = xyz_au[:, 0], xyz_au[:, 1], xyz_au[:, 2]
        delta_e = -x * np.sin(ra_rad) + y * np.cos(ra_rad)
        delta_n = (
            -x * np.cos(ra_rad) * np.sin(dec_rad)
            - y * np.sin(ra_rad) * np.sin(dec_rad)
            + z * np.cos(dec_rad)
        )

        # One magnification column per source trajectory.  Prefer the full
        # binary-lens model (breaks the NNLS degeneracy between overlapping
        # source trajectories); fall back to PSPL columns.  Missing per-source
        # params (j > 0) degrade gracefully to the single-source estimate.
        A_cols = self._binary_magnification_columns(t, n_src, _get)
        if A_cols is None:
            A_cols = [
                self._pspl_magnification(
                    t, delta_e, delta_n, t0, u0, tE, pi_E_N, pi_E_E
                )
            ]
            for j in range(1, n_src):
                t0_j = _get(f"lens.{j}.t_0")
                u0_j = _get(f"lens.{j}.u_0")
                tE_j = _get(f"lens.{j}.t_E", tE)
                if t0_j is None or u0_j is None:
                    logger.warning(
                        f"lens.{j}.t_0/u_0 missing — flux bootstrap treats source {j} "
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

        q_flux_est = q_flux_fallback
        if len(A_cols) > 1:
            # Multi-source NNLS: F = Σ_j f_s,j · A_j + f_b
            X = np.column_stack(A_cols + [np.ones(len(t))])
            sol, _ = nnls(X, F_obs)
            f_srcs, f_blend_est = sol[:-1], sol[-1]
            f_source_est = float(np.sum(f_srcs))
            if q_flux_user is None and f_srcs[0] > 1e-30 and len(f_srcs) > 1:
                q_flux_est = float(np.clip(f_srcs[1] / f_srcs[0], 1e-3, 1e3))
            if f_source_user is not None and f_source_est > 1e-30:
                # honor the user's total source flux; keep the NNLS ratio
                f_blend_est = max(
                    float(
                        np.median(
                            F_obs
                            - X[:, :-1]
                            @ (f_srcs * f_source_user / f_source_est)
                        )
                    ),
                    0.0,
                )
                f_source_est = f_source_user
        elif f_source_user is not None:
            f_blend_est = max(
                float(np.median(F_obs - f_source_user * A_traj)), 0.0
            )
            f_source_est = f_source_user
        elif f_blend_user is not None:
            (f_source_est,), _ = nnls(
                A_traj.reshape(-1, 1), F_obs - f_blend_user
            )
            f_blend_est = f_blend_user
        else:
            X = np.column_stack([A_traj, np.ones(len(A_traj))])
            (f_source_est, f_blend_est), _ = nnls(X, F_obs)

        f_total = f_source_est + f_blend_est
        if f_total < 1e-30 or f_source_est < 1e-30:
            return (
                self._baseline_flux_fallback(f_obs),
                0.95,
                q_flux_est,
            )

        q_source = float(np.clip(f_source_est / f_total, 0.05, 0.95))
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
        """
        t_delta = (t - self._t0_par)[:, np.newaxis]  # (N, 1)
        return xyz_abs - (self._earth_pos_ref + self._earth_vel_ref * t_delta)

    def get_observer_position(self, time, observer_location="earth"):
        """
        High-precision observer position dispatcher.
        Delegates to the shared exozippy.ephemeris module (major bodies,
        topocentric ground sites, and spacecraft ephemeris files).
        """
        return get_observer_position(time, observer_location=observer_location)

    def register_parameters(self, system):
        """Stage 2: Declare the manifest with bootstrapped fluxes."""
        f_total_init = np.array(self.fs_init)
        q_source_init = np.array(self.q_source_init)

        # Inject hints for derived f_source / f_blend so the relaxation engine
        # can resolve initial values.  Also push the data-estimated q_source and
        # log_f_total as RANK_DERIVED_DATA hints so they override the defaults.yaml
        # values while still yielding to any explicit user override in params.yaml
        # (RANK_USER wins — essential when restarting a fit from a previous MAP).
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
                f"{self.prefix}.{i}.q_source", q, rank=RANK_DERIVED_DATA
            )
            self.config_manager.add_hint(
                f"{self.prefix}.{i}.log_f_total",
                float(np.log10(f_total_init[i])),
                rank=RANK_DERIVED_DATA,
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
        n_sources = getattr(self, "_n_sources", 1)
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
                    rank=RANK_DERIVED_DATA,
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
        if hasattr(system, "sed"):
            self.manifest["zeropoint"] = {
                "expr_key": "default",
                "force_node": True,
            }

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
        ``_prepare_robust`` push data-driven hints (median error bar, and 10x
        that) which outrank it -- but it matters when a file has degenerate
        errors and those hints are skipped.
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

    def build_likelihood(self, model, system):

        # 1. Constants
        t = pm.Data("mu_time", self.time)
        obs_flux = pm.Data("mu_obs_flux", self.flux)
        obs_err = pm.Data("mu_obs_err", self.err)

        # 2. Magnification — both symbolic and Op paths take Skowron+2011
        #    geocentric deviations (AU).  get_magnification_op dispatches:
        #    PSPL→symbolic (NUTS-friendly), binary/finite-source→MulensModel
        #    Op (use Metropolis).
        #
        #    When finite source is active, pass u1 and bandpass from the connected
        #    Band component.  Multiple distinct bands across instruments are not yet
        #    supported for finite-source LD; the first band found is used.
        u1 = None
        bandpass = None
        if (
            system.lens.finite_source[0]
            and hasattr(system, "band")
            and np.any(self.band_map >= 0)
        ):
            band_indices = [
                self.band_map[i]
                for i in range(self.n_elements)
                if self.band_map[i] >= 0
            ]
            unique = sorted(set(band_indices))
            if len(unique) > 1:
                logger.warning(
                    "Multiple bands for finite-source instruments; using first band's u1."
                )
            band_idx = unique[0]
            u1 = system.band.u1.value[band_idx]
            bandpass = system.band.names[band_idx]

        # One magnification curve per source trajectory (NSNL)
        n_src = self._n_sources
        A_per_source = []
        for j in range(n_src):
            system.lens.resolve_auto_vbbl(self.time, index=j)
            A_per_source.append(
                system.lens.get_magnification_op(
                    t,
                    self.observer_pos,
                    system,
                    index=j,
                    u1=u1,
                    bandpass=bandpass,
                )
            )

        # 3. Flux Model: F = Σ_j f_s,j·A_j + f_b, with f_s,1 = f_s/(1+q_F),
        #    f_s,2 = f_s·q_F/(1+q_F) (q_F per instrument — sources differ in color)
        fs = self.f_source.value[self.inst_map_tensor]
        fb = self.f_blend.value[self.inst_map_tensor]

        if n_src == 1:
            model_flux = fs * A_per_source[0] + fb
        else:
            qf = self.q_flux.value[self.inst_map_tensor]
            qf_safe = pt.maximum(qf, 0.0)
            model_flux = (
                fs / (1.0 + qf_safe) * A_per_source[0]
                + fs * qf_safe / (1.0 + qf_safe) * A_per_source[1]
                + fb
            )

        # No clamp: model_flux may legitimately be <= 0 (f_blend is allowed to
        # be negative, and difference-imaging data live around zero).  The
        # likelihood is Gaussian in flux, so nothing here takes a logarithm.

        # Optional detrending against extra data columns.  The coefficients are
        # magnitude-space (airmass, seeing, ...), so they enter multiplicatively
        # in flux -- algebraically the same model as the additive magnitude
        # detrending used before, and well defined for negative fluxes.
        # Block-diagonal, so coefficients never mix across instruments.
        if self.total_detrend_cols > 0:
            detrend = pm.Data("mu_detrend", self.detrend_matrix)
            model_flux = model_flux * pt.power(
                10.0, -0.4 * pt.dot(detrend, self.detrend_coeffs.value)
            )

        # 4. Error scaling & Likelihood (shared base helper: err * err_scale).
        # The shared dispatcher is the plain Normal unless a light curve asked
        # for a GP, in which case that curve gets a celerite2 marginal
        # likelihood around this same magnification model.
        sigma = self.total_sigma(obs_err)

        # Modeling-draft prose for the magnification model and the flux-space
        # likelihood, declared next to the model they describe.
        lens = system.lens
        if lens.uses_op(0):
            if lens.backend == "mulensmodel":
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
        return [int(i) for i in system.lens.source_map]

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

    def add_parameter(self, model, param_name, system, context_nodes=None):
        """Inject the SED context nodes the derived zeropoint needs.

        ``m_source_pred`` (the SED-predicted source magnitude in each light
        curve's own band) is a cross-component forward-model node, not a
        manifest parameter, so the generic dep parser cannot reach it -- the
        same situation ``Orbit.add_parameter`` handles for its group masses.
        ``zp_center``/``sed_constrained`` carry the resolved prior center and
        the per-element on/off mask (see physics.calc_zeropoint).
        """
        if param_name == "zeropoint" and not context_nodes:
            context_nodes = self._zeropoint_context(system)
        return super().add_parameter(model, param_name, system, context_nodes)

    def _zeropoint_context(self, system):
        """Context nodes for the derived ``zeropoint`` expression."""
        sed = system.sed
        source_indices = self._sed_source_indices(system)
        filter_keys = self._sed_filter_keys(system)

        zp_cfg = self.config_manager.resolve(
            self.prefix,
            "zeropoint",
            shape=(self.n_elements,),
            names=self.names,
        )
        zp_mu = np.atleast_1d(np.asarray(zp_cfg.get("mu"), dtype=float))
        zp_sigma = np.atleast_1d(np.asarray(zp_cfg.get("sigma"), dtype=float))

        m_pred = []
        mask = np.zeros(self.n_elements, dtype=float)
        for i, name in enumerate(self.names):
            if filter_keys[i] is None:
                # No SED prediction to tie to.  calc_zeropoint reports
                # zp_center for this element, so the entry here is only a
                # placeholder that must stay finite (it is still evaluated,
                # just not selected, and switch's gradient would carry a NaN
                # through the zero it multiplies by).
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
                sed.predict_blend_appmag(
                    source_indices, filter_keys[i], system
                )
            )
            mask[i] = 1.0

        return {
            "m_source_pred": pt.stack(m_pred),
            "zp_center": pt.as_tensor_variable(zp_mu),
            "sed_constrained": pt.as_tensor_variable(mask),
        }

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
        zp_i = m_SED + 2.5*log10(f_s,i) (physics.calc_zeropoint), and its
        Gaussian prior (defaults: 0 +/- 0.2 mag) is applied by
        Parameter.build_pymc's derived-with-sigma branch at stage 5.  This
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

        What remains here at stage 6 is the opt-in blend tie:
        `sed_constrain_blend: true` additionally ties f_blend to the
        SED-predicted blend of the modeled non-source stars through the same
        zeropoint (Gaussian potential with `sed_blend_sigma`, default 0.2
        mag). f_blend also contains any unrelated field stars, so leave this
        off unless the blend is understood.
        """
        sed = system.sed
        source_indices = self._sed_source_indices(system)
        n_stars = system.star.n_elements
        other_indices = [i for i in range(n_stars) if i not in source_indices]
        filter_keys = self._sed_filter_keys(system)

        for i, name in enumerate(self.names):
            if filter_keys[i] is None:
                continue
            if not self.config[i].get("sed_constrain_blend", False):
                continue
            if not other_indices:
                logger.warning(
                    f"mulensinstrument {name}: sed_constrain_blend is "
                    f"set but every modeled star is a source; skipping."
                )
                continue
            blend_sigma = float(self.config[i].get("sed_blend_sigma", 0.2))
            m_blend_pred = sed.predict_blend_appmag(
                other_indices, filter_keys[i], system
            )
            fb_i = pt.maximum(self.f_blend.value[i], 1e-30)
            m_blend_inst = -2.5 * pt.log10(fb_i) + self.zeropoint.value[i]
            pm.Potential(
                f"{self.prefix}.{name}.sed_blend_prior",
                -0.5 * ((m_blend_pred - m_blend_inst) / blend_sigma) ** 2,
            )

    def compile_plotters(self, model, system):
        """Compile fast PyTensor functions for the lightcurve."""
        t_input = pt.vector("mu_t_input")
        obs_pos_input = pt.dmatrix("obs_pos")
        inst_idx = pt.iscalar("mu_inst_idx")

        param_symbols = [p.value for p in system.plot_params]

        n_src = self._n_sources
        A_per_source = [
            system.lens.get_magnification_op(
                t_input, obs_pos_input, system, index=j
            )
            for j in range(n_src)
        ]

        fs_inst = self.f_source.value[inst_idx]
        fb_inst = self.f_blend.value[inst_idx]

        # Δmag = mag(t) − mag_baseline = −2.5·log10(A_eff).
        # Zero at baseline, negative when brighter, independent of f_total.
        if n_src == 1:
            model_flux = fs_inst * A_per_source[0] + fb_inst
        else:
            qf_inst = pt.maximum(self.q_flux.value[inst_idx], 0.0)
            model_flux = (
                fs_inst / (1.0 + qf_inst) * A_per_source[0]
                + fs_inst * qf_inst / (1.0 + qf_inst) * A_per_source[1]
                + fb_inst
            )
        f_total_inst = pt.maximum(fs_inst + fb_inst, 1e-30)
        A_eff = model_flux / f_total_inst
        model_delta_mag = -2.5 * pt.log10(pt.maximum(A_eff, 1e-30))

        # Retained symbolically so plot_data can walk the graph for
        # param_deps (the evaluator skips components whose specs declare no
        # dependency on a moved slider -- empty deps would freeze the GUI's
        # microlensing charts in live mode).
        self._delta_mag_node = model_delta_mag

        self._compiled_delta_mag = pytensor.function(
            inputs=[t_input, obs_pos_input, inst_idx] + param_symbols,
            outputs=model_delta_mag,
            on_unused_input="ignore",
        )

        # The same curve before the delta-mag normalization: the model flux in
        # instrument inst_idx's own flux system.  The GP conditional mean is
        # additive there, so this is what the "physical + GP" plot curve is
        # built on (see plot_data).
        self._compiled_model_flux = pytensor.function(
            inputs=[t_input, obs_pos_input, inst_idx] + param_symbols,
            outputs=model_flux,
            on_unused_input="ignore",
        )

        # Baseline flux at a given parameter point, used by plot() to normalize
        # the data onto the same Δmag scale as the model curves.
        self._compiled_f_total = pytensor.function(
            inputs=[inst_idx] + param_symbols,
            outputs=f_total_inst,
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

        Tries the numeric index form first (user-provided params), then the
        name form (derived params stored by finalize_user_params under the
        name key).
        """
        cm = self.config_manager
        lens_name = (cm.system_config.get("lens") or [{}])[0].get("name", "0")
        for key in (
            f"lens.0.{base_param}",
            f"lens.{lens_name}.{base_param}",
        ):
            d = cm.user_params.get(key)
            if d is not None:
                return d.get("initval") if isinstance(d, dict) else float(d)
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
        so all data lands on the reference scale, matching the model (also
        drawn in the reference system).  Using the plotted point's fitted
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
        """GUI/PDF plot specs: the aligned delta-mag lightcurve, plus a zoom
        copy (x_range +/-3 tE) when t_0/t_E seeds are known.

        The chart is drawn in magnitudes even though the fit is in flux -- that
        is the convention these curves are read in -- so any point whose (in
        general aligned) flux is not positive comes back as NaN and is simply
        not drawn.  With point=None each instrument's data are returned in its
        own system (no fitted fluxes exist to align them onto one scale).
        See Component.plot_data and plotspec.PlotSpec.
        """
        from exozippy.plotspec import PlotSpec, Trace

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
                PlotSpec(
                    id=f"{self.prefix}.lightcurve",
                    component=comp_id,
                    title=title,
                    xlabel="Time [BJD]",
                    ylabel="mag",
                    traces=traces,
                    meta={
                        "y_inverted": True,
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
        unique_observers, obs_to_inst, inst_obs_loc = self._observer_groups()
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
        align, ref_idx = aln["align"], aln["ref_idx"]
        fs_vec, fb_vec = aln["fs_vec"], aln["fb_vec"]

        node = getattr(self, "_delta_mag_node", None)
        deps = self._model_trace_param_deps(node, system)

        traces = []
        for obs_loc in unique_observers:
            i = obs_to_inst[obs_loc]
            try:
                # ref_idx: reference flux system, this observer's
                # magnification (parallax between sites is preserved).
                y_model = self._compiled_delta_mag(
                    t_model, obs_model_pos[obs_loc], ref_idx, *param_values
                )
            except Exception as e:
                logger.warning(
                    f"Model eval failed for observer '{obs_loc}': {e}"
                )
                continue
            traces.append(
                Trace(
                    name=(
                        f"model ({obs_loc})"
                        if len(unique_observers) > 1
                        else "model"
                    ),
                    role="model",
                    kind="line",
                    x=t_model,
                    y=y_model,
                    node=node,
                    style={"series_index": int(i)},
                )
            )

        # One "physical + GP" curve per light curve that requested a GP.
        # The GP is additive in that instrument's own FLUX (that is the space
        # celerite2 conditioned in), so it is added to the model flux there and
        # the sum is then mapped onto the reference flux system.
        for i in sorted(getattr(self, "_gp_pred_on_grid", {})):
            obs_pretty = obs_model_pos.get(inst_obs_loc[i])
            if obs_pretty is None:
                continue
            try:
                flux_i = self._compiled_model_flux(
                    t_model, obs_pretty, i, *param_values
                )
                gp_i = self.gp_mean_on_grid(system, point, i, t_model)
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
                    x=t_model,
                    y=y_gp,
                    style={"series_index": int(i), "lw": 1.0},
                )
            )

        for i in range(self.n_elements):
            mask = self.inst_map == i
            flux_i = self.flux[mask]
            err_i = self.err[mask]
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
            "y_inverted": True,
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
            ),
        }
        specs = [
            PlotSpec(
                id=f"{self.prefix}.lightcurve",
                component=comp_id,
                title=title,
                xlabel="Time [BJD]",
                ylabel="mag - mag$_0$",
                traces=traces,
                param_deps=deps,
                meta=meta,
            )
        ]
        if t0 is not None and tE is not None:
            specs.append(
                PlotSpec(
                    id=f"{self.prefix}.lightcurve_zoom",
                    component=comp_id,
                    title=f"{title} (zoom)",
                    xlabel="Time [BJD]",
                    ylabel="mag - mag$_0$",
                    traces=traces,
                    param_deps=deps,
                    meta=dict(
                        meta,
                        file_tag="mulens_zoom",
                        x_range=[t0 - 3.0 * tE, t0 + 3.0 * tE],
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
