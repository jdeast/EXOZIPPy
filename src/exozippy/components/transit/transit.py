import logging

import numpy as np
import pymc as pm
import pytensor

logger = logging.getLogger(__name__)
import pytensor.tensor as pt

from exozippy.components.instrument import Instrument
from exozippy.components.limbdark import quad_limb_darkened_flux
from exozippy.components.parameterization import (
    merge_overrides,
    pin_unselected,
)
from exozippy.outputs.prose import get_collector
from exozippy.outputs.texutils import latex_escape

from .. import ltt
from ..orbit import physics as orbit_physics
from ..orbit.bodies import planet_membership
from . import physics, ramp, spline


class Transit(Instrument):
    prose_noun = "transit photometry"
    # The fitted trend is ADDED to the model (Instrument.DETREND_SPACE).
    DETREND_SPACE = "additive"
    # Exposure smearing through the base's smear() (instrument.md,
    # "Exposure smearing").  The rule is EXOFASTv2's midpoint Riemann grid,
    # kept for parity and for a bit-identical start logp on every shipped
    # example when the machinery moved into the base; a Gauss rule
    # converges far faster on smooth stretches, and switching transit to it
    # is a separate, measured change.
    supports_exposure = True
    EXPOSURE_RULE = "midpoint"

    def __init__(self, config, config_manager):
        super().__init__(config, config_manager)
        self.label = "Transit Parameters"
        # Filter identity and limb darkening live on the Band component;
        # each instrument references a band block by name.
        self.band_names = [c.get("band") for c in self.config]
        deprecated_filters = [
            c.get("filter") for c in self.config if c.get("filter")
        ]
        if deprecated_filters:
            logger.warning(
                "transit 'filter:' is deprecated and ignored; reference a "
                "band: block instead (bands carry the filter identity and "
                "limb darkening)."
            )
        self.total_detrend_cols = 0
        # Light-travel-time (Roemer delay) correction, per file (see
        # components/ltt.py) -- on by default (Jason's decision: transit/rm/
        # astrometry on, rv/mulens off; matches EXOFASTv2). Per-file, not a
        # single component-wide flag, for consistency with every other
        # per-file key (gp:, likelihood:, ninterp:, rm:) -- build_likelihood's
        # group loop and compile_plotters both handle a group/instrument
        # mix of on/off files (see the mask logic there).
        self._light_travel_time_active = np.array(
            [bool(c.get("light_travel_time", True)) for c in self.config]
        )
        self._load_spline_config()
        self._load_dilution_config()
        self._load_ramp_config()

    # EXOFASTv2's sed_frac_uncertainty (exofast_multised.pro): the fractional
    # systematic error on each star's SED-predicted band flux, propagated to
    # the floor on the SED's dilution prediction.
    DEFAULT_SEDDEBLEND_FLOOR = 0.02

    def _load_dilution_config(self):
        """Read the per-file ``fitdilute:`` / ``seddeblend:`` /
        ``seddeblend_floor:`` keys (at construction).

        ``seddeblend`` is kept RAW here (``True``, ``False`` -- the
        default, as in EXOFASTv2 -- or a list of star names): the star names
        resolve only against the star and SED components, in
        ``register_parameters`` (``_resolve_dilution``).  Type errors raise
        now, since each one silently changes the model if guessed at.
        """
        self.fitdilute = []
        self.seddeblend_raw = []
        self.seddeblend_floor = []
        for i, c in enumerate(self.config):
            label = f"{self.prefix}[{self.names[i]}]"
            fit = c.get("fitdilute", False)
            if not isinstance(fit, (bool, np.bool_)):
                raise ValueError(
                    f"[{label}] fitdilute must be true or false; got {fit!r}."
                )
            deb = c.get("seddeblend", False)
            if not isinstance(deb, (bool, np.bool_)):
                if not isinstance(deb, (list, tuple)) or not all(
                    isinstance(x, str) for x in deb
                ):
                    raise ValueError(
                        f"[{label}] seddeblend must be true, false or a "
                        f"list of star names; got {deb!r}."
                    )
            floor = c.get("seddeblend_floor", self.DEFAULT_SEDDEBLEND_FLOOR)
            if (
                isinstance(floor, (bool, np.bool_))
                or not isinstance(floor, (int, float, np.integer, np.floating))
                or not np.isfinite(floor)
                or floor <= 0
            ):
                raise ValueError(
                    f"[{label}] seddeblend_floor must be a positive "
                    f"fractional flux error (e.g. 0.02); got {floor!r}."
                )
            if "seddeblend_floor" in c and deb is False:
                logger.warning(
                    f"[{label}] seddeblend_floor is set but seddeblend is "
                    f"false; there is no SED constraint for it to floor."
                )
            self.fitdilute.append(bool(fit))
            self.seddeblend_raw.append(
                bool(deb) if isinstance(deb, (bool, np.bool_)) else list(deb)
            )
            self.seddeblend_floor.append(float(floor))

    @staticmethod
    def sed_reads_dilution(configs, n_stars):
        """Whether any transit config asks the SED for a dilution.

        Read from RAW configs, for ``SED.photometric_readers`` at stage 3
        (the same ruling as ``Band.ld_consumers``): ``seddeblend: true`` or a
        star list does, with more than one star modeled.  Conservative: it
        answers for every star rather than the named ones.
        """
        if n_stars < 2:
            return False
        return any(c.get("seddeblend", False) is not False for c in configs)

    def _resolve_dilution(self, system):
        """Stage 3: which files fit ``dilute``, and which the SED constrains.

        Sets ``self.dilute_fit`` (bool per file) and ``self.deblend_stars``
        (per file: ``None``, or the sorted star indices whose summed flux
        the SED dilution prediction is taken over, the host included).

        ``seddeblend`` defaults to false, as in EXOFASTv2: deblending is
        opt-in, since a resolved neighbor or a light curve already deblended
        upstream (TESS SPOC) must not be diluted again.  ``true`` or a list
        with no SED, one star,
        or a band the SED has no filter for RAISES: the user asked for a
        constraint the fit cannot apply.  ``seddeblend`` implies
        ``fitdilute``.
        """
        band = system.band
        star = getattr(system, "star", None)
        star_names = list(star.names) if star is not None else []
        n_stars = len(star_names)
        sed = getattr(system, "sed", None)
        self.dilute_fit = []
        self.deblend_stars = []
        for i in range(self.n_elements):
            label = f"{self.prefix}[{self.names[i]}]"
            raw = self.seddeblend_raw[i]
            band_idx = int(self.band_map[i])
            filter_key = band.filter_mist[band_idx]
            have_filter = bool(
                sed is not None and filter_key and sed.has_filter(filter_key)
            )
            host = int(band.star_indices[band_idx])
            stars = None
            if raw is not False:
                why = None
                if sed is None:
                    why = "there is no sed: block to predict the fluxes"
                elif n_stars < 2:
                    why = "only one star is modeled"
                elif not have_filter:
                    why = (
                        f"band filter '{filter_key}' is not in the SED's "
                        f"BC grid"
                    )
                if why:
                    raise ValueError(
                        f"[{label}] seddeblend is set but {why}. Use "
                        f"fitdilute: true to fit the dilution without the "
                        f"SED."
                    )
                if raw is True:
                    stars = list(range(n_stars))
                else:
                    unknown = [n for n in raw if n not in star_names]
                    if unknown:
                        raise ValueError(
                            f"[{label}] seddeblend names unknown star(s) "
                            f"{unknown}; the stars are {star_names}."
                        )
                    stars = sorted({star_names.index(n) for n in raw} | {host})
            if stars is not None and stars == [host]:
                raise ValueError(
                    f"[{label}] seddeblend names only the host star "
                    f"'{star_names[host]}'; list the stars whose light is "
                    f"blended with it, or set seddeblend: false."
                )
            self.deblend_stars.append(stars)
            self.dilute_fit.append(bool(self.fitdilute[i] or stars))

    def _hint_dilute_from_sed(self, system):
        """Start each seddeblend file's ``dilute`` at the SED's prediction.

        The stars' SED inputs are read with ``probe_start`` (the stage 1-3
        reader, ``config.md``), the predicted A_D = 1 - F_host / sum F is
        evaluated numerically (``SED.predict_flux_fraction_at``) and pushed
        as a PRECEDENCE_DERIVED_DATA hint, so a user ``initval`` or link on
        the file's dilute still wins.  A star input the engine cannot reach
        leaves the defaults.yaml start of 0, with a warning.
        """
        sed = system.sed
        band = system.band
        inputs = sed.FLUX_FRACTION_INPUTS
        stars = sorted({j for st in self.deblend_stars if st for j in st})
        paths = [f"star.{j}.{name}" for j in stars for name in inputs]
        probed = self.config_manager.probe_start(paths)
        missing = [p for p in paths if probed[p].value is None]
        if missing:
            logger.warning(
                f"[{self.prefix}] cannot start dilute at the SED prediction: "
                f"no start for {missing}; it starts at 0 and the seddeblend "
                f"prior pulls it in."
            )
            return
        values = {
            name: {j: float(probed[f"star.{j}.{name}"].value) for j in stars}
            for name in inputs
        }
        for i, st in enumerate(self.deblend_stars):
            if not st:
                continue
            band_idx = int(self.band_map[i])
            frac = sed.predict_flux_fraction_at(
                int(band.star_indices[band_idx]),
                band.filter_mist[band_idx],
                values,
                st,
            )
            self.config_manager.add_hint(
                f"{self.prefix}.{i}.dilute", 1.0 - frac
            )

    def _load_spline_config(self):
        """Read the per-file ``fitspline:`` / ``splinespace:`` keys (at
        construction).

        EXOFASTv2's per-transit FITSPLINE / SPLINESPACE (mkss.pro):
        ``fitspline`` co-fits a fixed-knot cubic B-spline in time with the
        transit model, ``splinespace`` is its breakpoint spacing in days
        (default 0.75).  Populates ``self.fitspline[i]`` (bool) and
        ``self.splinespace[i]`` (float, days).  Absent everywhere, no file
        gains a spline column and the model is byte-for-byte what it was
        before this feature existed.

        Both keys are validated strictly: ``fitspline: 1`` (EXOFASTv2's
        0/1 spelling) and a non-positive or non-numeric ``splinespace``
        raise, because either silently changes the model if guessed at.
        A ``splinespace`` on a file without ``fitspline: true`` does nothing
        and warns.
        """
        self.fitspline = []
        self.splinespace = []
        for i, c in enumerate(self.config):
            label = f"{self.prefix}[{self.names[i]}]"
            fit = c.get("fitspline", False)
            if not isinstance(fit, (bool, np.bool_)):
                raise ValueError(
                    f"[{label}] fitspline must be true or false; got {fit!r}."
                )
            space = c.get("splinespace", spline.DEFAULT_SPLINESPACE)
            if (
                isinstance(space, (bool, np.bool_))
                or not isinstance(space, (int, float, np.integer, np.floating))
                or not np.isfinite(space)
                or space <= 0
            ):
                raise ValueError(
                    f"[{label}] splinespace must be a positive number of "
                    f"days (the spline breakpoint spacing); got {space!r}."
                )
            if "splinespace" in c and not fit:
                logger.warning(
                    f"[{label}] splinespace is set but fitspline is not "
                    f"true; no spline is fit, so splinespace is ignored."
                )
            self.fitspline.append(bool(fit))
            self.splinespace.append(float(space))

    def _load_ramp_config(self):
        """Read the per-file ``fitramp:`` key (at construction).

        EXOFASTv2's per-transit FITRAMP (exofast_chi2v2.pro): the file's
        model is multiplied by an exponential detector ramp with its own
        ``transit.<name>.ramp_amp`` / ``ramp_tau`` (see ``_lc_model``).
        Populates ``self.fitramp[i]`` (bool).  Absent everywhere, neither
        parameter exists and the model is byte-for-byte what it was before
        this feature existed.  Validated strictly, like ``fitspline``:
        EXOFASTv2's 0/1 spelling raises.
        """
        self.fitramp = []
        for i, c in enumerate(self.config):
            fit = c.get("fitramp", False)
            if not isinstance(fit, (bool, np.bool_)):
                raise ValueError(
                    f"[{self.prefix}[{self.names[i]}]] fitramp must be true "
                    f"or false; got {fit!r}."
                )
            self.fitramp.append(bool(fit))

    def _ltt_active(self, orbits):
        """Per-file light-travel-time flags, forced off when the orbit
        cannot supply the parameters the correction needs.

        `Orbit.register_parameters` declares a/m_primary/m_companion/m_total
        only when its bodies resolve, so a geometry-only orbit has none of
        them. Since `light_travel_time` defaults to ON, reading them
        unguarded would turn any such config -- which built fine before the
        correction existed -- into an AttributeError at build time.
        """
        if ltt.orbit_supports_ltt(orbits):
            return self._light_travel_time_active
        if self._light_travel_time_active.any():
            logger.warning(
                "transit: light-travel-time correction disabled -- the orbit "
                "does not define %s (its bodies did not resolve; see the "
                "orbit component's own warning). Set light_travel_time: "
                "false on the affected transit file(s) to silence this.",
                ", ".join(ltt.REQUIRED_ORBIT_PARAMS),
            )
        return np.zeros_like(self._light_travel_time_active)

    @property
    def prefix(self):
        return "transit"

    @classmethod
    def get_utilities(cls):
        from ...utilities import bls, getdata
        from ...utilities.registry import (
            UtilitySpec,
            argparse_subprocess_runner,
        )

        return [
            UtilitySpec(
                name="getdata",
                label="Download TESS/Kepler light curves",
                description=(
                    "Fetch TESS/Kepler light curves via lightkurve and write "
                    "them in EXOFASTv2/EXOZIPPy format."
                ),
                component_keys=["transit"],
                available=True,
                build_parser=getdata.build_parser,
                run=argparse_subprocess_runner("exozippy.utilities.getdata"),
            ),
            UtilitySpec(
                name="bls",
                label="BLS period search",
                description=(
                    "Box Least Squares transit search: report the period, "
                    "epoch, depth and duration of the strongest signal."
                ),
                component_keys=["transit"],
                available=True,
                build_parser=bls.build_parser,
                run=argparse_subprocess_runner("exozippy.utilities.bls"),
            ),
        ]

    @classmethod
    def config_schema(cls):
        return [
            {
                "key": "file",
                "kind": "datafile",
                "accepts": "*.dat",
                "required": True,
                "doc": (
                    "Whitespace-delimited light curve; columns are time, "
                    "flux, flux_err, then optional detrend columns. Comment "
                    "lines start with '#'."
                ),
            },
            {
                "key": "band",
                "kind": "ref",
                "accepts": ["band"],
                "required": True,
                "doc": (
                    "Name of the band: block carrying this light curve's "
                    "filter identity and limb darkening."
                ),
            },
            {
                "key": "filter",
                "kind": "option",
                "accepts": None,
                "required": False,
                "doc": (
                    "Deprecated and ignored; reference a band: block "
                    "instead (bands carry the filter identity)."
                ),
            },
            cls._mask_config_schema(),
            cls._columns_config_schema(("time", "flux", "err")),
            *cls._time_config_schema(),
            cls._plot_style_config_schema(),
            cls._label_config_schema(),
            cls._gp_config_schema(),
            cls._likelihood_config_schema(),
            cls._out_frac_prior_config_schema(),
            *cls._exposure_config_schema(),
            {
                "key": "fitdilute",
                "kind": "option",
                "accepts": [True, False],
                "required": False,
                "doc": (
                    "Fit this file's dilution transit.<name>.dilute, the "
                    "fraction of the aperture's flux from other sources "
                    "(EXOFASTv2's FITDILUTE). With no seddeblend it is "
                    "constrained only by the data and any params-file "
                    "prior or link. Default false (implied by seddeblend)."
                ),
            },
            {
                "key": "seddeblend",
                "kind": "option",
                "accepts": None,
                "required": False,
                "doc": (
                    "Stars blended in this file's aperture: true (all), "
                    "false (none), or a list of star names (the host is "
                    "always included). Fits transit.<name>.dilute with a "
                    "Gaussian prior centered on the SED-predicted flux "
                    "fraction of the other stars in the band, whose width "
                    "propagates seddeblend_floor (EXOFASTv2's SEDDEBLEND). "
                    "Default false."
                ),
            },
            {
                "key": "seddeblend_floor",
                "kind": "option",
                "accepts": None,
                "required": False,
                "doc": (
                    "Fractional systematic error on each star's "
                    "SED-predicted band flux, propagated to the width of "
                    "the seddeblend prior (EXOFASTv2's "
                    "SED_FRAC_UNCERTAINTY). Default 0.02."
                ),
            },
            {
                "key": "fitspline",
                "kind": "option",
                "accepts": [True, False],
                "required": False,
                "doc": (
                    "Co-fit a fixed-knot cubic B-spline in time with the "
                    "transit model (EXOFASTv2's FITSPLINE, Vanderburg's "
                    "keplerspline): its coefficients join this file's "
                    "detrend coefficients. Default false."
                ),
            },
            {
                "key": "splinespace",
                "kind": "option",
                "accepts": None,
                "required": False,
                "doc": (
                    "Spline breakpoint spacing in days (default 0.75, "
                    "EXOFASTv2's SPLINESPACE). Gaps wider than this start a "
                    "new spline segment. Used only with fitspline: true."
                ),
            },
            {
                "key": "fitramp",
                "kind": "option",
                "accepts": [True, False],
                "required": False,
                "doc": (
                    "Multiply this file's model by an exponential detector "
                    "ramp 1 - A*(1 - exp(-(t - t0)/tau)), t0 the file's "
                    "earliest time (EXOFASTv2's FITRAMP; JWST/Spitzer), "
                    "fitting transit.<name>.ramp_amp and ramp_tau. Default "
                    "false."
                ),
            },
        ]

    def load_data(self, system):
        """Stage 1: Load CSVs and generate data-driven bounds/inits."""
        self.baseline_init = [1.0] * self.n_elements
        self.jittervar_lower = [0.0] * self.n_elements
        # fitramp: per file, the ramp's zero point (the file's MINIMUM time,
        # so an unsorted file cannot shift it), its time span (for tau's
        # upper bound) and the data-derived (A, tau) start.  NaN / None on
        # files without a ramp.
        self.ramp_t0 = [np.nan] * self.n_elements
        self.ramp_span = [np.nan] * self.n_elements
        self.ramp_seed = [None] * self.n_elements

        # The exposure description (exptime_s / ninterp) is the base's,
        # parsed at construction (Instrument._load_exposure_config).
        blocks = self._concat_blocks()
        for i in range(self.n_elements):
            # Shared reader: columns:, mask:, time_* conversion, then one
            # sort per file before anything is derived from it, keeping the
            # flux, errors and detrend columns aligned by construction.
            df = self._read_data(
                i, roles=("time", "flux", "err"), detrend=True
            )
            self.baseline_init[i] = np.median(df.iloc[:, 1].values)
            self.jittervar_lower[i] = self._jitter_floor(df.iloc[:, 2].values)
            if self.fitramp[i]:
                self._seed_ramp_file(i, df)

            # fitspline: the spline columns join this file's detrend block;
            # without it the accumulator reads the block off df exactly as
            # before (the backward-compatible path, untouched).
            detrend = None
            if self.fitspline[i]:
                detrend = self._spline_columns(i, df)

            blocks.add(
                i,
                time=df.iloc[:, 0].values,
                obs=df.iloc[:, 1].values,
                err=df.iloc[:, 2].values,
                df=df,
                detrend=detrend,
            )

        # Shared accumulator: concatenation (time/flux/err), inst_map, the
        # per-file row ranges, the block-diagonal detrend matrix, and the
        # optional GP / robust-likelihood hooks.  No user_factor: the errors
        # are already in the amplitude parameters' unit (relative flux).
        blocks.finalize("flux")

        self._record_exposure_groups(self.time, self.inst_map)

        # Blind seeding: measure the period and conjunction epoch from the
        # photometry when nothing else supplies them.  Stage 1a, not stage 2
        # -- see components/globalsearch.py for why (Orbit builds tc's hard
        # window at stage 2 from whatever start it can see).
        self.bls_signal = None
        self._seed_from_bls(system)

    def _seed_ramp_file(self, i, df):
        """Stage 1, ``fitramp`` files: the ramp's zero point, span and start.

        ``t0`` is the file's minimum time and the span its range; a file
        whose times do not span a positive interval has no ramp to fit and
        RAISES.  The ``(baseline, A, tau)`` seed is ``ramp.seed_ramp``'s
        least-squares fit of the decaying form; under the ``-A`` form
        ``baseline`` is the flux at ``t0``, so it replaces the median as this
        file's ``baseline_init`` (the median sits near the LATE level,
        ``baseline * (1 - A)``).  Files without a ramp keep the median.
        """
        time = df.iloc[:, 0].values.astype(float)
        span = float(np.max(time) - np.min(time))
        if not np.isfinite(span) or span <= 0 or time.size < 3:
            raise ValueError(
                f"[{self.prefix}[{self.names[i]}]] fitramp needs at least 3 "
                f"points spanning a positive time interval; got "
                f"{time.size} point(s) over {span} d."
            )
        self.ramp_t0[i] = float(np.min(time))
        self.ramp_span[i] = span
        base, amp, tau = ramp.seed_ramp(
            time,
            df.iloc[:, 1].values,
            df.iloc[:, 2].values,
            ramp.TAU_FLOOR,
        )
        self.ramp_seed[i] = (amp, tau)
        self.baseline_init[i] = base

    def _spline_columns(self, i, df):
        """File ``i``'s detrend block with its spline columns appended.

        ``df`` is ``_read_data``'s frame: time, flux, err, then any detrend
        columns.  Returns ``(n_obs, n_user + n_spline - 1)``: the user's
        columns as read, then the cubic B-spline basis
        (``spline.spline_basis``) with its LAST column dropped.

        Why one column is dropped, and why it is not the baseline instead:
        the basis sums to 1 at every point, so the full basis plus the
        file's ``baseline`` is exactly singular.  ``_build_block_detrend``
        then mean-subtracts every column, which removes the constant
        direction from each one -- and moves the degeneracy INSIDE the
        spline block (the whitened columns satisfy sum_k sd_k * W_k = 0).
        Pinning the baseline would therefore leave the block singular;
        dropping one column does not, and span{1, W_1..W_{K-1}} is exactly
        span{B_1..B_K}: EXOFASTv2's spline space, with ``baseline`` as its
        constant, matching EXOFASTv2 still fitting F0 alongside the spline.

        The combined design [1 | user columns | spline] must then have full
        column rank, and that is CHECKED here rather than assumed: a user
        detrend column that is linear (or up to cubic) in time lies inside
        the spline's span and is exactly as degenerate.  Rank is invariant
        under the whitening once the constant column is included, so the
        raw columns are checked.
        """
        label = f"{self.prefix}[{self.names[i]}]"
        time = df.iloc[:, 0].values.astype(float)
        user = df.iloc[:, 3:].values.astype(float)
        basis = spline.spline_basis(time, self.splinespace[i], label=label)[
            :, :-1
        ]
        block = np.column_stack([user, basis])

        design = np.column_stack([np.ones(len(time)), block])
        rank = np.linalg.matrix_rank(design)
        if rank < design.shape[1]:
            raise ValueError(
                f"[{label}] fitspline: the baseline, the {user.shape[1]} "
                f"detrend column(s) and the {basis.shape[1]} spline "
                f"column(s) are degenerate (rank {rank} of "
                f"{design.shape[1]}).  A detrend column that is a "
                f"polynomial in time up to cubic -- e.g. the time column "
                f"itself -- is already inside the spline's span; remove it "
                f"(or list only the others with `columns: {{detrend: "
                f"[...]}}`), or there are too few points per spline segment "
                f"for splinespace={self.splinespace[i]!r} d."
            )
        return block

    def epochs_constraining(self, system, orbit):
        """Every light curve's epochs on every planet's orbit, with the
        transit-timing information each carries.

        Every transit file models every planet (the assumption
        ``Orbit._transit_only`` and ``orbit.occultation_datasets`` make), so
        each file times each planet's orbit.  The information is Carter et
        al. (2008)'s, spread over the points between first and fourth
        contact of the seeded ephemeris (``timing.transit_information``),
        with the transit's depth, duration and ingress from the stage-3
        seeds (``Orbit.seeded_transit_shape``) and the errors relative to
        the file's baseline flux -- rescaled to the light curve's own
        point-to-point scatter (``timing.error_scale``), because that is
        the noise the fit's jitter will find, and a file whose quoted
        errors are off would otherwise be misweighted by the square of the
        error (measured: `examples/kelt17`'s MVRC errors are 1.7x too
        small; `examples/hat3`'s TESS S16 errors 1.2x too large).
        Occultations are not counted: their
        timing information is smaller by the depth ratio squared.
        """
        from ..orbit import timing

        # No planet block: nothing transits, and the build refuses the
        # config later with its own message.
        planets = system.active_components.get("planet")
        if planets is None:
            return {}
        shapes = [
            orbit.seeded_transit_shape(system, j)
            for j in range(planets.n_elements)
        ]
        seed = orbit.timing_seed
        out = {}
        for i in range(self.n_elements):
            t = self.dataset_times(i)
            sigma = (
                self.dataset_errors(i)
                * timing.error_scale(
                    np.asarray(self.flux[self.rows(i)], dtype=float),
                    self.dataset_errors(i),
                )
                / self.baseline_init[i]
            )
            per_orbit = {}
            for o, depth, T, tau in shapes:
                w = timing.transit_information(
                    t,
                    sigma,
                    seed.tc[o],
                    seed.period[o],
                    depth,
                    T,
                    tau,
                    exptime_s=self.exptime_s[i],
                )
                per_orbit[o] = per_orbit.get(o, 0.0) + w
            for o, w in per_orbit.items():
                out.setdefault(int(o), []).append((t, w, "conjunction"))
        return out

    def _seed_from_bls(self, system):
        """Seed orbital period, conjunction epoch and radius ratio from BLS.

        Runs only when the relaxation engine cannot already DERIVE the
        period and conjunction time (``globalsearch.starts_satisfied``), and
        pushes a value only for the quantities that were missing -- so a
        params file that gives the period but not the epoch keeps its period
        and gains an epoch, with no precedence question to adjudicate.

        The radius ratio is opportunistic: a missing ``planet.p`` does not
        trigger a search (a fit can start at 1 Jupiter radius), but a search
        that ran for the period reports a depth, and ``sqrt(depth)`` is the
        radius ratio to the accuracy a start value needs.  The engine turns
        it into a ``planet.radius`` through ``Eq(p, radius / star_radius)``.
        """
        from .. import globalsearch

        mode = globalsearch.search_mode(system)
        if mode == "off":
            return
        orbit_ndx = globalsearch.sole_orbit_index(system, self.prefix)
        if orbit_ndx is None:
            return

        cm = self.config_manager
        planet_ndx = self._sole_planet_index(system, orbit_ndx)
        groups = {
            "period": (
                f"orbit.{orbit_ndx}.period",
                f"orbit.{orbit_ndx}.logP",
            ),
            "tc": (f"orbit.{orbit_ndx}.tc",),
        }
        if planet_ndx is not None:
            groups["p"] = (
                f"planet.{planet_ndx}.p",
                f"planet.{planet_ndx}.radius",
            )
        satisfied = globalsearch.starts_satisfied(cm, groups)
        required_missing = not (satisfied["period"] and satisfied["tc"])
        if mode != "force" and not required_missing:
            logger.debug(
                "[%s] BLS not needed: the orbital period and conjunction "
                "time are already derivable.",
                self.prefix,
            )
            return

        logger.info(
            "[%s] no start value for %s -- running a Box Least Squares "
            "search over %d photometric points.",
            self.prefix,
            ", ".join(k for k, v in satisfied.items() if not v) or "(forced)",
            self.time.size,
        )

        # Each file in its own flux system: divide by that file's median so
        # the concatenation is one relative-flux series and the box depth
        # means the same thing in every row.
        baseline = np.asarray(self.baseline_init, dtype=float)
        scale = np.where(
            np.isfinite(baseline) & (baseline != 0.0), baseline, 1.0
        )
        norm = scale[self.inst_map]
        signal = globalsearch.bls_search(
            self.time,
            self.flux / norm,
            self.err / norm,
            context=self.prefix,
        )
        self.bls_signal = signal
        if signal is None:
            return

        q = globalsearch.QUALITY_TRANSIT
        source = f"BLS on {self.n_elements} light curve(s)"
        applied = []
        if mode == "force" or not satisfied["period"]:
            applied.append(
                globalsearch.seed_start(
                    cm, f"orbit.{orbit_ndx}.period", signal.period, q, source
                )
            )
        if mode == "force" or not satisfied["tc"]:
            applied.append(
                globalsearch.seed_start(
                    cm, f"orbit.{orbit_ndx}.tc", signal.epoch, q, source
                )
            )
        if planet_ndx is not None and (
            mode == "force" or not satisfied.get("p", True)
        ):
            applied.append(
                globalsearch.seed_start(
                    cm,
                    f"planet.{planet_ndx}.p",
                    float(np.sqrt(signal.depth)),
                    q,
                    source,
                )
            )

        if not any(applied):
            # Nothing was actually taken (another search of equal or better
            # quality got there first).  Prose describes what the fit did.
            return

        get_collector(system).add(
            "Initial values for the orbital period and time of conjunction "
            "were measured from the photometry with a Box Least Squares "
            r"periodogram \citep{Kovacs:2002}, as implemented in "
            r"\texttt{astropy} \citep{Astropy:2013,Astropy:2018,Astropy:2022}."
            " Starting values do not enter the likelihood and cannot move "
            "the posterior.",
            section="data",
            key=f"{self.prefix}.global_search",
            rank=70,
        )

    def _sole_planet_index(self, system, orbit_ndx):
        """The one planet on this orbit, or None if there is not exactly one.

        Read off the raw config (``orbit.bodies.planet_membership``, the one
        reader of planet membership) rather than ``planet.orbit_map``: this
        runs at stage 1a, where the maps may not have been built yet.
        """
        planets = getattr(system, "planet", None)
        if planets is None:
            return None
        membership = planet_membership(self.config_manager.system_config)
        on_orbit = [j for j, (o, _) in enumerate(membership) if o == orbit_ndx]
        return on_orbit[0] if len(on_orbit) == 1 else None

    def register_parameters(self, system):
        """Stage 3: Embed data-driven hints into the PyMC manifest."""
        self._hint_baseline()
        self.manifest = {"baseline": None}
        self._register_noise(self.manifest, self.jittervar_lower)
        self._register_gp(self.manifest)
        self._register_robust(self.manifest)

        if self.total_detrend_cols > 0:
            self.manifest["detrend_coeffs"] = self._detrend_manifest_entry()

        # Limb darkening comes from the Band component; map each
        # instrument (and each observation) to its band. A transit model
        # cannot be computed without LD, so a missing/unknown band
        # reference is an error, not a warning.
        if not hasattr(system, "band"):
            raise ValueError(
                "Transit instruments require band: blocks (bands carry the "
                "filter identity and limb-darkening parameters)."
            )
        name_to_idx = {name: i for i, name in enumerate(system.band.names)}
        missing = [
            (self.names[i], n)
            for i, n in enumerate(self.band_names)
            if n not in name_to_idx
        ]
        if missing:
            raise ValueError(
                f"Transit instrument(s) reference unknown band(s): "
                f"{missing}. Available bands: {list(name_to_idx)}."
            )
        self.band_map = np.array(
            [name_to_idx[n] for n in self.band_names], dtype=int
        )
        self.obs_band_map = self.band_map[self.inst_map]

        # Dilution (EXOFASTv2's DILUTE): a parameter only where a file fits
        # it, through the OPT-IN pin so the others cost nothing and a
        # params-file prior on one still frees it.  Absent everywhere, the
        # parameter does not exist and the model is undiluted.
        self._resolve_dilution(system)
        if any(self.dilute_fit):
            self.manifest["dilute"] = pin_unselected(
                self.n_elements, self.dilute_fit
            )
            if any(self.deblend_stars):
                self._hint_dilute_from_sed(system)

        # Exponential ramp (EXOFASTv2's FITRAMP): parameters only where a
        # file fits one, through the same OPT-IN pin as dilute.  tau's
        # per-file upper bound, 10 x the file's span, goes through the
        # "overrides" channel (resolve takes the tighter of it and a user
        # bound); NaN leaves the pinned files' elements alone.  With both
        # bounds finite tau is on the logit transform -- no hard chi2 wall.
        if any(self.fitramp):
            self.manifest["ramp_amp"] = pin_unselected(
                self.n_elements, self.fitramp
            )
            self.manifest["ramp_tau"] = merge_overrides(
                pin_unselected(self.n_elements, self.fitramp),
                {
                    "upper": [
                        ramp.TAU_SPAN_FACTOR * s if fit else np.nan
                        for fit, s in zip(self.fitramp, self.ramp_span)
                    ]
                },
            )
            self._hint_ramp()

    def _hint_baseline(self):
        """Push each light curve's median flux as a PRECEDENCE_DERIVED_DATA hint.

        The median is measured in ``load_data`` (stage 1), so it is ready
        by the time this runs at stage 3 -- which is what lets it go through
        the provenance pipeline at all.

        It used to be a plain manifest option (``{"baseline": {"initval":
        ...}}``), and options are merged as ``{**cfg, **options}`` AFTER
        ``resolve()``: they beat the user's params file outright and never
        acquire a rank.  For a data-derived START value that is backwards --
        an explicit ``transit.<name>.baseline`` in a params file (a restart
        file, say) was silently discarded.  As a hint it sits at
        PRECEDENCE_DERIVED_DATA (60), the tier this channel exists for: above the
        defaults.yaml 1.0 (20) and below the user (100), exactly like
        ``rvinstrument``'s gamma (median RV) and ``mulensinstrument``'s
        f_source/log_f_total.

        A non-finite median (a file with no usable flux) is skipped rather
        than hinted: the defaults.yaml 1.0 is then what the engine resolves,
        which is also the no-data fallback ``load_data`` seeds
        ``baseline_init`` with, so the two agree by construction.
        """
        baseline_init = getattr(self, "baseline_init", None)
        if baseline_init is None:
            return  # register_parameters without load_data (bare harness)
        arr = np.atleast_1d(baseline_init)
        for i in range(self.n_elements):
            val = float(arr[i])
            if not np.isfinite(val):
                logger.warning(
                    f"transit {self.names[i]}: median flux is not finite; "
                    f"leaving baseline at its defaults.yaml start value."
                )
                continue
            self.config_manager.add_hint(f"{self.prefix}.{i}.baseline", val)

    def _hint_ramp(self):
        """Push each ramp file's data-derived ``(A, tau)`` start
        (``_seed_ramp_file``) as PRECEDENCE_DERIVED_DATA hints, so a user
        ``initval`` still wins.  Pinned (non-ramp) elements keep the
        defaults.yaml start, which is what their pin holds."""
        for i, seed in enumerate(self.ramp_seed):
            if seed is None:
                continue
            amp, tau = seed
            self.config_manager.add_hint(f"{self.prefix}.{i}.ramp_amp", amp)
            self.config_manager.add_hint(f"{self.prefix}.{i}.ramp_tau", tau)

    def _add_sed_deblend_prior(self, system):
        """Stage 7: the SED constraint on ``dilute`` (EXOFASTv2's SEDDEBLEND).

        For each file with ``deblend_stars``, the SED predicts the dilution
        from the stars' band fluxes, ``A_D = F_other / (F_host + F_other)``,
        and ``dilute`` gets a Gaussian centered on it.  Its width propagates
        a fractional error ``seddeblend_floor`` on each of F_host and F_other
        (exofast_multised.pro's deblend_uncertainty):

            sigma = sqrt(2) * floor * F_host * F_other / (F_host + F_other)^2
                  = sqrt(2) * floor * A_D * (1 - A_D)

        sigma tracks sampled stellar parameters, so the potential keeps its
        -log(sigma) term (the house convention, ``SED._fractional_floor_logp``).
        The prediction is kept as the Deterministic ``transit.dilute_sed``,
        one entry per deblended file, for diagnostics.
        """
        idx = [i for i, st in enumerate(self.deblend_stars) if st]
        if not idx:
            return
        sed = system.sed
        band = system.band
        preds, sigmas = [], []
        for i in idx:
            band_idx = int(self.band_map[i])
            filter_key = band.filter_mist[band_idx]
            host = int(band.star_indices[band_idx])
            frac_host = sed.predict_flux_fraction(
                host, filter_key, system, star_indices=self.deblend_stars[i]
            )
            a_d = 1.0 - frac_host
            preds.append(a_d)
            sigmas.append(
                np.sqrt(2.0) * self.seddeblend_floor[i] * a_d * frac_host
            )
        pred = pm.Deterministic(f"{self.prefix}.dilute_sed", pt.stack(preds))
        sigma = pt.stack(sigmas)
        dilute = self.dilute.value[np.asarray(idx, dtype=int)]
        pm.Potential(
            f"{self.prefix}.seddeblend",
            pt.sum(-0.5 * pt.sqr((dilute - pred) / sigma) - pt.log(sigma)),
        )
        self.dilute.add_prior_contribution(
            r"$\mathcal{N}(\mathrm{SED}, \sigma_{\rm sys})$",
            text="Normal(SED, sigma_sys)",
            elements=idx,
        )

    def _lc_model(self, system, t, blocks):
        """The transit model on times ``t`` -- THE expression, built once.

        ``build_likelihood`` calls this with the data's ``pm.Data`` time
        vector and the files' own row blocks; ``compile_plotters`` calls it
        with a symbolic grid laid out in the same per-file blocks.  So the
        plotted curve is literally the likelihood code run on other times:
        there is no second light-curve graph to keep in step, and every
        per-file choice (band, limb darkening, exptime/ninterp smearing,
        light-travel-time gating, SED dilution, baseline) is resolved from
        the block a row belongs to exactly as it is for the data.  The
        in-repo template is ``AstrometryInstrument._rel_model``.

        ``t``      -- ``(N,)`` time tensor.
        ``blocks`` -- ``[(inst_idx, rows), ...]``: the concrete row indices
                      of ``t`` that belong to each file.  Together they
                      cover ``t``; a file may appear in more than one block
                      and a file with no rows may be omitted.

        THE BASELINE IS MULTIPLICATIVE.  The astrophysical model is built
        as a unit-normalized relative flux ``rel`` (1 out of transit: the
        star's own flux, minus every planet's diluted transit dip, plus its
        diluted thermal/reflected light and beaming, times the ellipsoidal
        factor) and only THEN multiplied by the file's ``baseline``:
        ``lc_model = baseline * rel``.  ``baseline`` is a flux
        NORMALIZATION -- the planet blocks a FRACTION of the star's light,
        so the dip scales with whatever the light is normalized to -- which
        is EXOFASTv2's ``modelflux *= f0`` (exofast_chi2v2.pro) and its
        ``f0*((model-1)*(1-dilute)+1)`` (exofast_tran.pro).  Until the PR
        that introduced this paragraph the baseline was ADDED
        (``baseline - dip``), which underestimates the depth by a factor
        ``baseline`` and is grossly wrong for a light curve in raw counts.
        The full equation, and how detrending and a GP compose with it, is
        in ``components/instrument.md`` ("The transit light-curve model").

        Returns ``(lc_model, planet_terms, ramp)``:

        ``lc_model`` ``(N,)`` -- ``baseline * ramp * rel`` with ``rel`` as
            above (``ramp`` is 1 except on ``fitramp`` files' rows):
            transit/eclipse/thermal/reflection exposure-smeared on each
            row's own sub-exposure grid, beaming + ellipsoidal un-smeared
            at the flat per-observation time.  WITHOUT the detrend term,
            which is per observation and is added by ``build_likelihood``
            alone.
        ``planet_terms`` ``(N, N_planets)`` -- the change planet p's step
            makes to the running model, IN THE DATA'S FLUX UNITS
            (``baseline`` times the change to ``rel``), so
            ``ramp * (baseline[inst] + sum(planet_terms, axis=1))``
            reproduces ``lc_model`` for ANY planet count.  The terms carry
            NO ramp, so a phased model curve on a grid away from the data
            is ramp-free; the phased panels divide the ramp out of the
            DATA.  The ellipsoidal factor
            multiplies the running total (1 + the earlier planets), exactly
            as it does in the model itself; the old plot path folded only
            the baseline in per planet, which was exact for one planet and
            an approximation for more (reviews 1.5.6, 7.14.1).
        ``ramp`` ``(N,)`` -- the per-row ramp factor (``_ramp_profile``);
            ones when no file sets ``fitramp``, and then not multiplied in
            at all, so ``lc_model`` is bit-identical to the ramp-free build.

        The op sequence inside the planet loop is the one the likelihood
        always had -- including associations kept for a 1-ulp reason (see
        the ``r_norm`` comment).  The multiplicative baseline moved the
        start logp of every example whose baseline is not exactly 1;
        change the op sequence only with the start-logp table in hand.
        """
        orbits = system.orbit
        planets = system.planet
        band = system.band

        # The block layout -> per-row instrument index (numpy, known at
        # graph-build time -- the group loop's gates are Python if/else on
        # it) and its int32 tensor twin for indexing per-instrument
        # parameter vectors (the same construction as stage 5's
        # ``*_map_tensor``).
        inst_map = self._blocks_inst_map(blocks)
        inst_map_t = pt.as_tensor_variable(inst_map).astype("int32")
        obs_band_map = self.band_map[inst_map]
        obs_band_map_t = pt.as_tensor_variable(obs_band_map).astype("int32")

        # 1. The photometric baseline, a per-file flux NORMALIZATION.  It
        # multiplies the finished relative model at the very end (see the
        # docstring); everything in between is built on ``lc_model`` as a
        # unit-normalized relative flux that starts at exactly 1.
        baseline_obs = self.baseline.value[inst_map_t]  # (N,)
        lc_model = pt.ones_like(baseline_obs)

        # 1b. Per-planet orbital vectors, gathered onto the planet index.
        ecc_p = orbits.ecc.value[planets.orbit_map]  # (N_planets,)
        inc_p = orbits.inc.value[planets.orbit_map]
        period_p = orbits.period.value[planets.orbit_map]
        tc_p = orbits.tc.value[planets.orbit_map]
        ar_p = planets.ar.value

        # The per-planet transit/occultation geometry (impact
        # parameters and durations) used to be built here as bare
        # Deterministics.  It moved to the planet component in review 8.8.7,
        # where it is a set of ordinary derived Parameters -- table rows,
        # LaTeX macros, units, and a user-settable Gaussian, which is what
        # lets a published duration or eclipse time constrain e and omega.
        # It is geometry, not photometry: no light curve enters it, and an
        # RV-only fit has the same durations.  The phased-plot x-range
        # reads `planet.t14` for that reason.

        # 2. Orbital elements per planet. These don't depend on the
        # observation/sub-exposure grid, so they're computed once and
        # reused by every node group below (see 4.).
        tp = orbits.tp_target.value[planets.orbit_map][
            None, None, :
        ]  # (1, 1, N_planets)
        n = orbits.n.value[planets.orbit_map][None, None, :]
        ecc = ecc_p[None, None, :]
        cosw = orbits.cosw.value[planets.orbit_map][None, None, :]
        sinw = orbits.sinw.value[planets.orbit_map][None, None, :]
        inc = inc_p[None, None, :]
        a_rstar = ar_p[None, None, :]
        sin_i = pt.sin(inc)
        cos_i = pt.cos(inc)

        # 2b. Light-travel-time (Roemer delay) inputs -- per-file gating via
        # ltt_active (self._light_travel_time_active, forced off when the
        # orbit cannot supply these parameters), resolved per group below.
        # Three roles, three factors -- see ltt.py's `factor` docs. The
        # occultation seam takes the mass DIFFERENCE; light EMITTED by the
        # planet (reflection) takes the planet's own barycentric fraction;
        # light emitted by the STAR (beaming, ellipsoidal) takes the
        # star's. One corrected time array cannot serve all three, and
        # using the geometry's for everything (or leaving the stellar
        # terms uncorrected, as here until 2026-08-15) mixes time
        # references differing by ~a/c within a single phase curve.
        ltt_active = self._ltt_active(orbits)
        # Structural, not a runtime test: True only where every orbit
        # these planets sit on has its sqrt(e) pair PINNED at zero, in
        # which case the Kepler solve is a sine and a cosine
        # (orbit.physics.solve_kepler, review 6.8.2).
        circular_kepler = orbits._all_circular(planets.orbit_map)
        a_rel = ltt_factor = ltt_reflect_factor = ltt_star_factor = None
        if ltt_active.any():
            a_rel = orbits.a.value[planets.orbit_map][
                None, None, :
            ]  # (1, 1, N_planets), physical semi-major axis [R_sun]
            # Barycentric scaling for an OCCULTATION seam: the mass
            # DIFFERENCE, not the planet's own barycentric fraction. A
            # transit is not an emission event -- the planet blocks light
            # the STAR emitted -- so both bodies enter at their own
            # retarded times and the star's delay partially cancels the
            # planet's. See ltt.py's `factor` docs for the derivation, and
            # for why m_primary/m_total (used here until 2026-08-15, and by
            # EXOFASTv2's target2bjd.pro) agrees to O(q) for a planet but
            # predicts a spurious a/c offset for a comparable-mass pair
            # whose true offset is exactly zero.
            m_primary = orbits.m_primary.value[planets.orbit_map]
            m_companion = orbits.m_companion.value[planets.orbit_map]
            m_total = orbits.m_total.value[planets.orbit_map]
            ltt_factor = ((m_primary - m_companion) / m_total)[None, None, :]
            # Reflected light comes off the planet's disk, so it rides the
            # planet's own delay about the barycenter.
            ltt_reflect_factor = (m_primary / m_total)[None, None, :]
            # Doppler beaming and ellipsoidal variation are the STAR's own
            # flux (its radial motion and its tidal shape), so they ride
            # the star's delay -- a factor ~q, not ~1. ltt.py's z is the
            # companion-minus-primary separation TOWARD the observer, and
            # the star sits at -(M_c/M) of it: the factor is NEGATIVE (it
            # was +M_c/M, a sign error, until review 1.5.11; the RV path,
            # RVInstrument._emission_time, has the same -M_c/M). Flat
            # (N_planets,): these terms are evaluated per observation, not
            # on the sub-exposure grid.
            ltt_star_factor = -m_companion / m_total

        # 3. Limb Darkening Setup (per observation, mapped from each
        # instrument's Band). When every band uses the linear law, Band's
        # manifest has no u2; the quadratic term is then zero.

        # Secondary-eclipse thermal emission (fitthermal): gate the whole
        # branch on the resolved parameter state, not the fitthermal
        # flags -- the manifest "overrides" pin is layered UNDER user
        # params, so params.yaml can free or set thermal without
        # fitthermal. When every element is pinned at exactly 0 the
        # thermal math (a second quad_solution_vector per planet, as
        # expensive as the transit itself) is skipped entirely, matching
        # exofast_tran.pro's `if thermal ne 0d0` runtime gate.
        thermal_active = band.thermal_may_be_nonzero()
        # BEER (PR 1.b): reflection is smeared on the same sub-exposure
        # grid as thermal (see the group loop below) and shares the same
        # resolved-state gate. Ellipsoidal is smooth on-orbit and is not
        # smeared -- it stays a flat per-observation array, applied once
        # after the group loop.
        reflect_active = band.reflect_may_be_nonzero()
        # Manifest-gated (see Band/Planet.register_parameters): with no
        # fitellip / beam flag anywhere the parameters do not exist, and
        # the terms are skipped entirely.
        ellip_active = band.ellipsoidal_may_be_nonzero()
        ellip_mapped = (
            band.ellipsoidal.value[obs_band_map_t] if ellip_active else None
        )
        beam_active = "beam" in planets.manifest

        # 3b. Dilution (EXOFASTv2's exofast_tran.pro:157,
        # f0*((model-1)*(1-dilute)+1)): a fraction ``dilute`` of the
        # aperture's flux comes from other sources, so every deviation of
        # the relative model from 1 is scaled by 1 - dilute (and the
        # baseline f0 then multiplies the result, step 5 below).  The SED constrains dilute
        # where seddeblend asks it to (_add_sed_deblend_prior); it does not
        # set it.
        dil_inst = None
        if "dilute" in self.manifest:
            dil_inst = 1.0 - self.dilute.value
        # Flat per-observation dilution for the post-loop beam term below
        # (EXOFASTv2 parity -- see the beam comment past the group loop).
        dil_obs_flat = None
        if dil_inst is not None:
            dil_obs_flat = dil_inst[inst_map_t]  # (N,)

        # 4. Exoplanet-core Transit Model, evaluated on each exposure's
        # sub-exposure nodes by the base's smear (instrument.md, "Exposure
        # smearing"): once per distinct ninterp group, each group's node
        # axis exactly that group's own ninterp wide, so a short-cadence
        # (ninterp=1) observation is never evaluated at another
        # instrument's larger ninterp.  With ninterp==1 everywhere there is
        # exactly one group of width 1, identical to the original
        # (pre-oversampling) computation.  _planet_dips returns one
        # (n_g, k_g) perturbation per planet; smear averages each over the
        # nodes and returns them (N,) in row order.
        def _planet_dips(t_sub, rows):
            """Every planet's instantaneous perturbation on one node group:
            ``t_sub`` the group's ``(n_g, k_g)`` sub-exposure times, ``rows``
            its row indices (numpy).  A tuple of ``(n_g, k_g)`` tensors."""
            t_grid = t_sub[:, :, None]  # (n_g, k_g, 1)
            nets = []

            # Light-travel-time correction. The factor depends on WHICH
            # observable is being retarded, not on the timestamps, so one
            # corrected time array cannot serve the whole model (see
            # ltt.py's `factor` docs). This group needs two of them:
            #
            #   geometry (transit/eclipse shape, and via planetvisible the
            #     thermal gating) -- the occultation seam, ltt_factor;
            #   reflected light -- emitted by the PLANET, so its own
            #     barycentric fraction m_primary/m_total.
            #
            # Beaming and ellipsoidal are stellar and un-smeared; they are
            # corrected after this loop with the primary's factor.
            #
            # Per-file gate (this group's own rows may mix files that want
            # it on and off, since groups are formed by ninterp value, not
            # by file). Costs nothing extra when every row in the group is
            # off (no ltt.retarded_time call, no pt.where); costs one extra
            # Kepler solve per role in use, no pt.where, when every row is
            # on (the default); costs one extra Kepler solve PLUS one
            # pt.where only for a genuinely mixed group.
            lt_active_rows = ltt_active[
                inst_map[rows]
            ]  # (n_g,) bool, numpy -- known at graph-build time

            def _retard_grid(role_factor):
                """t_grid retarded with `role_factor`, honoring the
                per-row gate. Returns t_grid untouched when no row in this
                group wants the correction."""
                if not lt_active_rows.any():
                    return t_grid
                corrected, _ = ltt.retarded_time(
                    t_grid,
                    tp,
                    n,
                    ecc,
                    sinw,
                    cosw,
                    sin_i,
                    a_rel,
                    factor=role_factor,
                    z0=0.0,
                    circular=circular_kepler,
                )
                if lt_active_rows.all():
                    return corrected
                # Both branches are ordinary, everywhere-finite time
                # values (no singularity like solve_delay's az=0
                # branch), so this pt.where carries none of the
                # where-trap risk that formula guarded against --
                # verified directly (not just asserted) by
                # tests/test_transit_ltt.py's
                # test_mixed_group_ltt_gradient_is_finite.
                lt_mask = pt.constant(
                    lt_active_rows[:, None, None].astype("float64")
                )
                return pt.where(lt_mask > 0.5, corrected, t_grid)

            t_grid_final = _retard_grid(ltt_factor)
            # Broadcast to (n_g, k_g, N_planets) unconditionally (a no-op
            # when t_grid_final already has that shape) so the per-planet
            # slice below is safe regardless of N_planets: with LTT off (or
            # a face-value t_grid pass-through), the last dim is the
            # unbroadcast size 1 from t_grid, and t_grid_final[:, :, p]
            # would index-error for any p > 0 without this.
            time_g_corrected = t_grid_final + pt.zeros(
                (1, 1, planets.n_elements)
            )

            # Reflected light is emitted by the PLANET, so its phase runs
            # on the planet's own retarded time, not the occultation
            # seam's. Only built when reflection is actually on.
            time_g_reflect = None
            if reflect_active and ltt_reflect_factor is not None:
                time_g_reflect = _retard_grid(ltt_reflect_factor) + pt.zeros(
                    (1, 1, planets.n_elements)
                )

            # The shared Kepler-to-state kernel (skips the Newton iteration
            # outright when every orbit these planets sit on is pinned
            # circular, review 6.8.2).
            terms = orbit_physics.state_vector_terms(
                t_grid_final,
                tp,
                n,
                ecc,
                sinw=sinw,
                cosw=cosw,
                circular=orbits._all_circular(planets.orbit_map),
            )
            # NOT a_rstar * terms.r_over_a: this association is the
            # pre-refactor one, kept because the reassociated product
            # rounded differently (1 ulp on hat3's start logp) and the
            # 4.8.2 refactor is pinned bit-identical.
            r_norm = a_rstar * (1.0 - pt.sqr(ecc)) / (1.0 + ecc * terms.cosf)
            sin_wf = terms.sinwf
            cos_wf = terms.coswf

            # (n_g, k_g, N_planets)
            b = pt.sqrt(
                pt.sqr(r_norm * cos_wf) + pt.sqr(r_norm * sin_wf * cos_i)
            )
            Z = r_norm * sin_wf * sin_i

            u1_mapped = band.u1.value[obs_band_map[rows]]  # (n_g,)
            if "u2" in band.manifest:
                u2_mapped = band.u2.value[obs_band_map[rows]]  # (n_g,)
            else:
                u2_mapped = pt.zeros_like(u1_mapped)

            thermal_g = None
            if thermal_active:
                # (n_g,) ppm; 0 for any band pinned off (see
                # Band.register_parameters).
                thermal_g = band.thermal.value[obs_band_map[rows]]

            reflect_g = None
            if reflect_active:
                # (n_g,) ppm; 0 for any band pinned off (see
                # Band.register_parameters).
                reflect_g = band.reflect.value[obs_band_map[rows]]

            dil_obs = None
            if dil_inst is not None:
                dil_obs = dil_inst[inst_map[rows]]  # (n_g,)

            for p_idx in range(planets.n_elements):
                b_p = b[
                    :, :, p_idx
                ]  # (n_g, k_g) sky-plane separation in units of R_*
                Z_p = Z[
                    :, :, p_idx
                ]  # (n_g, k_g) line-of-sight coord (+ = planet in front of star)
                r_p = planets.p.value[p_idx]  # scalar R_p/R_*

                # Limb-darkened flux fraction: 1.0 off-disk, <1.0 during transit.
                # u1/u2_mapped are (n_g,); add the sub-exposure axis so they
                # broadcast against b_p.
                flux_frac = quad_limb_darkened_flux(
                    b_p, r_p, u1_mapped[:, None], u2_mapped[:, None]
                )  # (n_g, k_g)

                # Fraction of stellar flux blocked (0 off-disk, ~ r^2 at disk centre)
                blocked = 1.0 - flux_frac

                # Primary transit only; secondary eclipse (planet behind star) has Z < 0
                blocked = pt.where(Z_p > 0.0, blocked, 0.0)

                # Secondary eclipse / constant thermal emission (fitthermal)
                # and BEER (PR 1.b) reflection. Both come off the planet's
                # disk and share planetvisible; both live inside this group
                # loop so they get exposure-smeared on the same sub-exposure
                # grid as the transit (EXOFASTv2 averages the full model,
                # thermal included). Beam/ellipsoidal are smooth on-orbit
                # and are handled separately, un-smeared, after this loop.
                net = blocked
                if thermal_g is not None or reflect_g is not None:
                    visible = physics.calc_planet_visible(
                        b_p, Z_p, r_p
                    )  # (n_g, k_g)
                    if thermal_g is not None:
                        net = net - 1e-6 * thermal_g[:, None] * visible
                    if reflect_g is not None:
                        t_ref = (
                            time_g_reflect
                            if time_g_reflect is not None
                            else time_g_corrected
                        )
                        reflect_term_g = physics.calc_reflect_term(
                            t_ref[:, :, p_idx],
                            tc_p[p_idx],
                            period_p[p_idx],
                            reflect_g[:, None],
                            visible,
                        )
                        net = net - reflect_term_g

                if dil_obs is not None:
                    # One blended-aperture dilution for the whole
                    # perturbation from unity: EXOFASTv2's
                    # f0*(model*(1-dilute)+dilute) scales the transit dip
                    # and the planet's extra flux by the same factor.
                    net = net * dil_obs[:, None]

                nets.append(net)  # (n_g, k_g), averaged by smear
            return tuple(nets)

        planet_decrement = self.smear(_planet_dips, t, blocks)

        planet_terms = []
        for p_idx in range(planets.n_elements):
            # This planet's exposure-averaged decrement, (N,) in row order:
            # the transit, thermal, and (BEER, PR 1.b) reflection terms,
            # all computed and exposure-smeared by _planet_dips above.
            net_avg = planet_decrement[p_idx]
            lc_before = lc_model
            lc_model = lc_model - net_avg
            # This planet's own change to the running RELATIVE model,
            # accumulated alongside lc_model term by term (see the
            # docstring); scaled by the baseline when it is reported.
            step = -net_avg

            tc_this = tc_p[p_idx]  # scalar, this planet's time of conjunction
            period_this = period_p[p_idx]

            # Beaming and ellipsoidal are the STAR's own flux, so they are
            # evaluated at the star's retarded time -- a different time
            # base from the occultation geometry above (see ltt.py's
            # `factor` docs). Until 2026-08-15 they used the uncorrected
            # time while reflection used the geometry's, so a phase curve
            # mixed three time references differing by ~a/c. Un-smeared,
            # so this is a flat (N,) correction, and it costs one Kepler
            # solve per planet only when a stellar term is on.
            time_star = t
            need_stellar = beam_active or ellip_mapped is not None
            if (
                need_stellar
                and ltt_star_factor is not None
                and ltt_active.any()
            ):
                # The orbital elements above are shaped (1, 1, N_planets)
                # for the sub-exposure grid; this term is un-smeared and
                # per planet, so it needs the FLAT scalars -- indexing the
                # 3-D versions with [p_idx] would slice axis 0 (size 1).
                star_corrected, _ = ltt.retarded_time(
                    t,
                    tp[0, 0, p_idx],
                    n[0, 0, p_idx],
                    ecc[0, 0, p_idx],
                    sinw[0, 0, p_idx],
                    cosw[0, 0, p_idx],
                    sin_i[0, 0, p_idx],
                    orbits.a.value[planets.orbit_map][p_idx],
                    factor=ltt_star_factor[p_idx],
                    z0=0.0,
                    circular=circular_kepler,
                )
                if ltt_active.all():
                    time_star = star_corrected
                else:
                    star_mask = pt.constant(
                        ltt_active[inst_map].astype("float64")
                    )
                    time_star = pt.where(star_mask > 0.5, star_corrected, t)

            # Beaming is diluted the same way thermal/reflect are above --
            # EXOFASTv2 parity: exofast_chi2v2.pro:1517/1556 pass both beam
            # and dilute into exofast_tran, which adds beam at
            # exofast_tran.pro:146 and applies the dilution scaling at
            # exofast_tran.pro:157 (after beam, so beam is diluted too).
            # Ellipsoidal is a multiplicative factor on the running relative
            # lc_model (1 + this planet's transit/eclipse/thermal/reflect/
            # beam so far), not an additive flux term; its DEVIATION is
            # diluted (see its own comment below). Neither term is gated by planetvisible: both are stellar
            # effects (the star's own RV motion / tidal shape), present
            # regardless of the planet's occultation state. Both are smooth
            # on-orbit and are deliberately NOT exposure-smeared -- they are
            # evaluated at the flat per-observation time, unlike thermal/
            # reflect above.
            if beam_active:
                beam_p = planets.beam.value[p_idx]  # scalar, ppm
                beam_term = physics.calc_beam_term(
                    time_star, tc_this, period_this, beam_p
                )
                if dil_obs_flat is not None:
                    beam_term = beam_term * dil_obs_flat
                lc_model = lc_model + beam_term
                step = step + beam_term

            # Ellipsoidal is multiplicative (exofast_tran.pro), applied to
            # the running relative lc_model (1 + the planets so far,
            # this planet's transit/eclipse/thermal/reflect/beam included). With >1 planet sharing
            # a band, each planet's factor multiplies in turn -- order-
            # dependent for N>1, exact for the single-planet case this
            # targets.  Its DEVIATION is diluted like every other term:
            # exofast_tran.pro applies the dilution to (modelflux - 1)
            # after the ellipsoidal factor multiplies in, so ellipsoidal
            # IS diluted there (review of PR #53; the difference is the
            # second-order (1-d)^2 vs (1-d) cross term, ~ppm x depth).
            if ellip_mapped is not None:
                ellip_dev = (
                    physics.calc_ellipsoidal_factor(
                        time_star, tc_this, period_this, ellip_mapped
                    )
                    - 1.0
                )
                if dil_obs_flat is not None:
                    ellip_dev = ellip_dev * dil_obs_flat
                lc_model = lc_model * (1.0 + ellip_dev)
                # (before + step) * (1 + dev) - before == step + (before +
                # step) * dev: the factor acts on the whole running total.
                step = step + (lc_before + step) * ellip_dev
            # Reported in the data's flux units: baseline * (1 + sum of
            # the relative steps) == baseline + sum(baseline * step).
            planet_terms.append(baseline_obs * step)

        # 5. The normalization multiplies the WHOLE astrophysical model
        # (EXOFASTv2's ``modelflux *= f0``): flux = baseline * rel.  A
        # fitramp file's exponential ramp multiplies with it (6.), averaged
        # over each exposure on the planet terms' own nodes (EXOFASTv2
        # evaluates it unsmeared, at the timestamp).
        if "ramp_amp" in self.manifest:
            ramp_obs = self._ramp_profile(t, blocks)
            lc_model = (baseline_obs * ramp_obs) * lc_model
        else:
            ramp_obs = pt.ones_like(baseline_obs)
            lc_model = baseline_obs * lc_model

        return lc_model, pt.stack(planet_terms, axis=1), ramp_obs

    def _ramp_profile(self, t, blocks):
        """6. The per-row exponential ramp (EXOFASTv2's FITRAMP), ``(N,)``,
        exposure-averaged.

        For a ``fitramp`` file's rows,
        ``R = 1 - A * (1 - exp(-(t - t0) / tau)) = 1 + A * expm1(-dt / tau)``
        with ``t0`` the file's MINIMUM time: ``R(t0) = 1`` exactly and
        ``R -> 1 - A`` late.  EXOFASTv2's ``-A`` reparameterization (its
        comment: the leading term "looks a lot like F0"), so ``baseline`` is
        the flux at ``t0``.  ``expm1`` keeps ``R`` exact near ``t0``.
        TODO: if the baseline-A covariance is bad for a short tau, try the
        plain ``1 + A * exp(-dt / tau)`` form (baseline = the late level).

        ``R`` is averaged over each exposure on the SAME sub-exposure nodes
        and weights as the planet terms (the base's ``smear``), so a file
        with ``exptime_s``/``ninterp`` sees the exposure-mean ramp; the model
        is then ``baseline * <R> * <rel>`` (instrument.md, "fitramp").  With
        ``ninterp == 1`` the average is the identity and ``R`` is evaluated
        at the timestamp.

        Every other row is exactly 1.0 by construction (``set_subtensor``
        onto ones, never ``pt.where``).  ``dt`` is clamped at 0 so neither a
        plot grid that starts before the data (the phased window sits at the
        seeded epoch, possibly periods away) nor the first exposure's early
        sub-exposure nodes can overflow at a short tau; the clamp acts on a
        time, not a parameter, and at ``dt == 0`` the gradient in ``A`` and
        ``tau`` is exactly 0 (``expm1(0) == 0``, ``d/dtau = dt/tau**2 == 0``),
        so there is no ``0 * inf`` to trap.
        """
        inst_map = self._blocks_inst_map(blocks)
        fitramp = np.asarray(self.fitramp, dtype=bool)
        ramp_t0 = np.asarray(self.ramp_t0, dtype=float)
        amp = self.ramp_amp.value
        tau = self.ramp_tau.value

        def _ramp_nodes(t_sub, rows):
            """The instantaneous ramp on one node group, ``(n_g, k_g)``."""
            inst = inst_map[rows]
            local = np.flatnonzero(fitramp[inst])
            out = pt.ones_like(t_sub)
            if local.size == 0:
                return (out,)
            inst_r = inst[local]
            dt = pt.maximum(
                t_sub[local] - pt.constant(ramp_t0[inst_r])[:, None], 0.0
            )
            r = 1.0 + amp[inst_r][:, None] * pt.expm1(
                -dt / tau[inst_r][:, None]
            )
            return (pt.set_subtensor(out[local], r),)

        (ramp_obs,) = self.smear(_ramp_nodes, t, blocks)
        return ramp_obs

    def _data_blocks(self):
        """The data's own row blocks, ``[(i, rows_i), ...]`` in file order.

        ``Instrument.rows`` publishes each file's contiguous range;
        materialized as an index array rather than left as the slice for
        the reason ``rows`` documents (a ``pm.Data``'s length is symbolic,
        so a slice's shape is too, and the JAX backend cannot trace it).
        """
        blocks = []
        for i in range(self.n_elements):
            sl = self.rows(i)
            blocks.append((i, np.arange(sl.start, sl.stop)))
        return blocks

    def _exposure_check_terms(self, system):
        """``Instrument._exposure_check_terms``: the likelihood's
        detrend-free light curve at the data (``_lc_model``, the one
        builder) and its sigma, every row in flux."""
        lc_model, _, _ = self._lc_model(
            system, pt.as_tensor_variable(self.time), self._data_blocks()
        )
        sigma = self.total_sigma(pt.as_tensor_variable(self.err))
        return [(self.inst_map, lc_model, sigma)]

    def _refresh_exposure_state(self):
        """The recorded node groups follow the node counts."""
        self._record_exposure_groups(self.time, self.inst_map)

    def build_likelihood(self, model, system):
        self._add_sed_deblend_prior(system)

        time = pm.Data("transit_time", self.time)
        flux = pm.Data("transit_data", self.flux)
        err = pm.Data("transit_err", self.err)

        planets = system.planet
        band = system.band

        # The one light-curve expression (see _lc_model), on the data's own
        # times and row blocks.  Retained as plain attributes, not
        # Deterministics: at (N_obs,) either would add N_obs * draws *
        # chains floats to every trace.  compile_plotters compiles them as
        # the plotted model AT the observations (the phased panels' "other
        # planets" cleaning), and tests compile a one-off function from
        # _model_flux_node.
        lc_model, planet_terms, ramp_obs = self._lc_model(
            system, time, self._data_blocks()
        )
        self._lc_physical_node = lc_model
        self._lc_planet_terms_node = planet_terms
        self._lc_ramp_node = ramp_obs

        # Detrending: the shared base mechanism (additive here) -- the one
        # place the likelihood meets the trend, inverted onto the plotted
        # data by detrend_corrected (Instrument._detrended_model).
        lc_model = self._detrended_model(lc_model, "transit_detrend")

        # Full per-observation model prediction:
        # baseline * (relative astrophysical model) + detrend.  A GP, when
        # a file asks for one, is additive on top of this (it models the
        # residual flux y - mu; Instrument.add_observation_likelihood).
        self._model_flux_node = lc_model

        # 5. Likelihood (shared base helper: sqrt(err^2 + jitter_variance)).
        # add_observation_likelihood is the plain Normal unless a light curve
        # asked for a GP, in which case that curve gets a celerite2 marginal
        # likelihood around this same transit model.
        sigma = self.total_sigma(err)
        self.add_observation_likelihood(
            "transit_likelihood",
            mu=lc_model,
            sigma=sigma,
            observed=flux,
            system=system,
        )

        # Modeling-draft prose for the transit model itself (the shared
        # data/noise sentences came from the dispatcher above).  The same
        # resolved-state gates _lc_model builds the terms from.
        terms = []
        if band.thermal_may_be_nonzero():
            terms.append("constant thermal emission")
        if band.reflect_may_be_nonzero():
            terms.append("reflected light")
        if band.ellipsoidal_may_be_nonzero():
            terms.append("ellipsoidal variation")
        if "beam" in planets.manifest:
            terms.append(r"Doppler beaming \citep{Faigler:2011}")
        if terms:
            from exozippy.outputs.prose import join_names

            get_collector(system).add(
                "The transit model includes phase-curve terms for "
                + join_names(terms)
                + ", following EXOFASTv2's parameterization "
                + r"\citep{Eastman:2019}.",
                section="planetary",
                key=f"{self.prefix}.phase_curve",
                rank=22,
            )
        spline_files = [i for i in range(self.n_elements) if self.fitspline[i]]
        if spline_files:
            from exozippy.outputs.prose import join_names

            spaces = sorted({self.splinespace[i] for i in spline_files})
            spacing = (
                f"{spaces[0]:g}~d"
                if len(spaces) == 1
                else "per-file spacings of "
                + join_names([f"{v:g}~d" for v in spaces])
            )
            get_collector(system).add(
                "We detrended the "
                + join_names(
                    [latex_escape(self.names[i]) for i in spline_files]
                )
                + (
                    " light curve"
                    if len(spline_files) == 1
                    else " light curves"
                )
                + r" with a cubic B-spline in time \citep{Vanderburg:2014}, "
                + f"with breakpoints every {spacing}, "
                + "fit simultaneously with the transit model: the spline "
                + "coefficients were sampled jointly with the other "
                + "parameters.",
                section="planetary",
                key=f"{self.prefix}.fitspline",
                rank=23,
            )
        get_collector(system).add(
            r"We modeled each transit with the analytic quadratic "
            r"limb-darkening light curve of \citet{Agol:2020}, as "
            r"implemented in exoplanet-core \citep{ForemanMackey:2021}.",
            section="planetary",
            key=f"{self.prefix}.lc_model",
            rank=20,
        )
        get_collector(system).add_software("exoplanet-core")
        if getattr(system, "band", None) is not None and (
            system.band.ld_laws and system.band.ld_laws[0] == "quadratic"
        ):
            get_collector(system).add(
                r"Limb-darkening coefficients were sampled in the "
                r"$(q_1, q_2)$ parameterization of \citet{Kipping:2013}.",
                section="planetary",
                key=f"{self.prefix}.ld_param",
                rank=21,
            )

    def detrend_caption(self):
        """The base caption, plus the spline when a file sets fitspline.

        The spline columns are detrend columns, so ``detrend_at_data``
        already subtracts the fitted spline from the plotted points; the
        caption has to say so, since "linear trend against the detrend
        columns" alone would misdescribe a file with no columns of its own.
        Unchanged when no file sets fitspline.
        """
        base = super().detrend_caption()
        if not any(self.fitspline):
            return base
        return base + (
            " For a light curve with fitspline, that trend includes the "
            "co-fit cubic B-spline in time."
        )

    # Points per instrument on a plotted model grid.  One size for the
    # unphased span and the phased period window, so the two panels share
    # a single compiled layout (see compile_plotters).
    _PLOT_GRID_N = 2000

    def compile_plotters(self, model, system):
        """Compile the plotted model -- the likelihood's expression, twice.

        Two compiled functions, both from ``_lc_model`` (there is no
        separate plotting graph any more; reviews 1.5.6, 7.14.1):

        ``_lc_data_fn(*params)`` -> ``(lc_physical, planet_terms)`` AT the
            observations: literally the nodes ``build_likelihood`` scored
            (minus the detrend term, which comes off the plotted data
            instead -- see Instrument.detrend_at_data), compiled against
            the plot parameters.  The phased panels' "other planets"
            cleaning reads its ``planet_terms``.
        ``_lc_grid_fn(t, *params)`` -> ``(lc_full, lc_matrix)`` on a plot
            GRID laid out in the same per-file blocks as the data:
            ``_PLOT_GRID_N`` rows per instrument, in file order, so a row's
            band, smearing, LTT gate and dilution resolve exactly as a data
            row's would.  Every model curve -- unphased span, phased period
            window -- is this one function fed a different ``t``.

        Exposure smearing therefore comes from the builder's own ninterp/
        exptime path; the NumPy re-implementation that used to average an
        instantaneous compiled function over sub-exposure offsets is gone,
        and with it the drift it caused: it smeared beaming and ellipsoidal
        too, which the likelihood deliberately does not (review 1.5.6).
        """
        t_input = pt.vector("lc_t_input")
        param_symbols = [p.value for p in system.plot_params]
        planets = getattr(system, "planet", None)
        orbits = getattr(system, "orbit", None)

        if planets is not None and orbits is not None:
            n_grid = self._PLOT_GRID_N
            blocks = [
                (i, np.arange(i * n_grid, (i + 1) * n_grid))
                for i in range(self.n_elements)
            ]
            lc_full_node, lc_matrix, _ = self._lc_model(
                system, t_input, blocks
            )

            # Retain the symbolic nodes and their time input so plot_data
            # can derive param_deps (graph walk) and hand the GUI the
            # symbolic tensors behind the model traces.  Unused by plot().
            self._lc_t_input = t_input
            self._lc_matrix_node = lc_matrix
            self._lc_full_node = lc_full_node

            self._lc_grid_fn = pytensor.function(
                inputs=[t_input] + param_symbols,
                outputs=[lc_full_node, lc_matrix],
                on_unused_input="ignore",
            )
            phys = getattr(self, "_lc_physical_node", None)
            terms = getattr(self, "_lc_planet_terms_node", None)
            ramp_node = getattr(self, "_lc_ramp_node", None)
            self._lc_data_fn = (
                pytensor.function(
                    inputs=param_symbols,
                    outputs=[phys, terms, ramp_node],
                    on_unused_input="ignore",
                )
                if phys is not None
                and terms is not None
                and ramp_node is not None
                else None
            )

        # Per-file GP conditional-mean evaluators (no-op without a gp: key).
        self._compile_gp_plotters(system)

    def _eval_lc_grid(self, t_blocks, param_values):
        """Evaluate the plotted model on one grid per instrument.

        ``t_blocks[i]`` is instrument ``i``'s ``(_PLOT_GRID_N,)`` time
        grid.  Returns ``(full, matrix)`` with ``full[i]`` the model light
        curve (baseline included) and ``matrix[i]`` the ``(_PLOT_GRID_N,
        N_planets)`` per-planet terms on that grid -- the concatenated
        layout ``compile_plotters`` compiled, split back per file.
        """
        n_grid = self._PLOT_GRID_N
        t_all = np.concatenate(
            [np.asarray(t, dtype=np.float64) for t in t_blocks]
        )
        if t_all.shape[0] != n_grid * self.n_elements:
            raise ValueError(
                f"[{self.prefix}] plot grid layout needs {n_grid} points "
                f"per instrument, got {[len(t) for t in t_blocks]}."
            )
        full, matrix = self._lc_grid_fn(t_all, *param_values)
        full = np.asarray(full).reshape(self.n_elements, n_grid)
        matrix = np.asarray(matrix).reshape(self.n_elements, n_grid, -1)
        return full, matrix

    def _eval_lc_data(self, param_values):
        """The likelihood's own model at the observations: ``(lc_physical,
        planet_terms)`` as ``(N_obs,)`` and ``(N_obs, N_planets)``."""
        phys, terms, _ = self._eval_lc_data_and_ramp(param_values)
        return phys, terms

    def _eval_lc_data_and_ramp(self, param_values):
        """``_eval_lc_data`` plus the per-observation ramp factor ``(N_obs,)``
        (ones without ``fitramp``), from the SAME compiled call."""
        phys, terms, ramp_vals = self._lc_data_fn(*param_values)
        return np.asarray(phys), np.asarray(terms), np.asarray(ramp_vals)

    def _lc_at_times(self, param_values, i, t):
        """Instrument ``i``'s plotted model at arbitrary times ``t``.

        The compiled grid layout is fixed at ``_PLOT_GRID_N`` rows per
        instrument, so ``t`` is fed through it in chunks of that size --
        block ``i`` carrying the chunk (padded with its last time), every
        other block a dummy -- and the results stitched back.  A
        convenience for a caller that wants the likelihood's model for ONE
        file at times of its own choosing (tests, a GUI probe); the panels
        feed the layout directly.  Returns ``(full, matrix)``: the model
        light curve (baseline included, GP excluded) ``(len(t),)`` and the
        per-planet terms ``(len(t), N_planets)``, smeared over instrument
        ``i``'s own exposure exactly as its data rows are.
        """
        t = np.asarray(t, dtype=np.float64).ravel()
        n_grid = self._PLOT_GRID_N
        fulls, mats = [], []
        for start in range(0, max(t.size, 1), n_grid):
            chunk = t[start : start + n_grid]
            fill = chunk[-1] if chunk.size else float(self.time[0])
            block = np.full(n_grid, fill)
            block[: chunk.size] = chunk
            t_blocks = [np.full(n_grid, fill) for _ in range(self.n_elements)]
            t_blocks[i] = block
            full, matrix = self._eval_lc_grid(t_blocks, param_values)
            fulls.append(full[i][: chunk.size])
            mats.append(matrix[i][: chunk.size])
        return np.concatenate(fulls), np.concatenate(mats)

    # ------------------------------------------------------------------
    # Shared data preparation. The matplotlib plot() path and the GUI
    # plot_data() path both go through these helpers, so the two paths
    # always draw the exact same arrays (see chart.Chart).
    # ------------------------------------------------------------------
    def _unphased_lc_curves(self, system, point):
        """Full model light curve per instrument: baseline + transit + GP.

        Returns ``[(t_pretty_i, y_i), ...]`` in file order, from ONE
        evaluation of the compiled grid layout.  The unphased panel shows
        the model the likelihood actually fits, so any GP this light curve
        requested is included; the phased panels take it back out of the
        data instead (see _phased_lc_arrays).  The GP term is zero for a
        light curve without a gp: key.
        """
        param_values = self._point_to_plot_params(point, system)
        t_blocks = []
        for i in range(self.n_elements):
            t_data = self.time[self.rows(i)]
            t_blocks.append(
                np.linspace(
                    t_data.min(), t_data.max(), self._PLOT_GRID_N
                ).astype(np.float64)
            )
        full, _ = self._eval_lc_grid(t_blocks, param_values)
        curves = []
        for i, t_pretty in enumerate(t_blocks):
            y_gp = self.gp_mean_on_grid(system, point, i, t_pretty)
            curves.append((t_pretty, full[i] + y_gp))
        return curves

    def _eval_unphased_lc(self, system, point, i):
        """Instrument ``i``'s unphased curve, ``(t_pretty, y)``; see
        ``_unphased_lc_curves``, which plot_data calls once for all."""
        return self._unphased_lc_curves(system, point)[i]

    def _phased_lc_shared(self, system, point):
        """The parts of a phased panel that do NOT depend on which planet.

        ``_phased_lc_arrays`` is called once per (planet, instrument) and
        recomputed the same things every time: the marshalled parameter
        values, the per-observation GP and detrend corrections (both
        point-only), and the model at the observations -- which the
        likelihood's own compiled node now supplies for every instrument
        at once (review 6.5.1 hoisted the per-instrument copies; the
        builder made the pass a single call).

        Returned as one dict per (component, point).  The per-planet phased
        model grids fill in lazily as planets are reached
        (``_phased_lc_grid``), one layout evaluation per planet shared by
        every instrument's panel.
        """
        param_values = self._point_to_plot_params(point, system)
        _, data_terms, data_ramp = self._eval_lc_data_and_ramp(param_values)
        # The fitted trend is a per-observation term no pretty-grid curve
        # carries, so it comes off the data through the shared base inverse
        # of the likelihood's own combination (Instrument.detrend_corrected);
        # unchanged without detrend columns.
        flux_corrected, err_corrected = self.detrend_corrected(
            self.flux, self.err, point
        )
        return {
            "param_values": param_values,
            "flux_corrected": flux_corrected,
            "err_corrected": err_corrected,
            # Removed from the phased data along with the other planets':
            # the correlated component would smear the fold.  Zeros when no
            # file has a GP.
            # Mapped into the corrected-data space by the same base
            # mechanism (identity for this additive space).
            "extra_signals": self.detrend_corrected_signal(
                self.gp_mean_at_data(system, point), point
            ),
            # (N_obs, N_planets): each planet's term at every observation,
            # from the likelihood's own node.
            "data_terms": data_terms,
            # (N_obs,): the fitramp factor at every observation, from the
            # same node (exactly 1.0 on files without a ramp).
            "data_ramp": data_ramp,
            "grid": {},
        }

    def _phased_lc_grid(self, shared, p_idx, t_model):
        """Every instrument's per-planet terms on planet ``p_idx``'s phase
        grid, ``(n_inst, _PLOT_GRID_N, N_planets)``, cached per planet."""
        cache = shared["grid"]
        if p_idx not in cache:
            _, matrix = self._eval_lc_grid(
                [t_model] * self.n_elements, shared["param_values"]
            )
            cache[p_idx] = matrix
        return cache[p_idx]

    def _phased_lc_arrays(self, system, point, p_idx, i, shared=None):
        """
        One-period phase grid, isolated model term for planet p_idx on
        instrument i, and the baseline-subtracted, other-planet-cleaned
        flux at the observed times -- used by plot_data() (and via it
        plot()).  Both the grid curve and the cleaning come from
        ``_lc_model``, so the panel shows the exposure-smeared model the
        likelihood fit, with beaming/ellipsoidal un-smeared as there.

        Everything here is DIVIDED BY the file's baseline, so the panel is
        in relative flux (deviation from 1) whatever units the file is in:
        the baseline is a multiplicative normalization
        (``_lc_model``'s docstring), the model curve is then exactly the
        planet's step of the unit-normalized model, and files on one
        summary stack are comparable even when one is in raw counts.
        Dividing the cleaned data, its error bars and the model by the same
        constant leaves every (data - model) / err unchanged.

        ``shared`` is this point's ``_phased_lc_shared`` dict; omit it and
        one is built, which is what a standalone caller wants and what the
        per-planet loop must NOT do.
        """
        if shared is None:
            shared = self._phased_lc_shared(system, point)
        planets = system.planet
        # orbit.period / orbit.tc are per-ORBIT vectors, so read them at the
        # orbit this planet sits on, not at its planet index -- the two agree
        # only by config-order coincidence (review 1.5.7).  orbit_map is the
        # numpy map build_maps wrote (build_tensor_maps adds a separate
        # *_map_tensor), and _lc_model maps period/tc the same way.
        o_idx = int(planets.orbit_map[p_idx])
        P_ref = self._point_value(point, system.orbit.period, o_idx)
        tc_ref = self._point_value(point, system.orbit.tc, o_idx)

        t_model = np.linspace(
            tc_ref - 0.5 * P_ref, tc_ref + 0.5 * P_ref, self._PLOT_GRID_N
        ).astype(np.float64)
        phase_model = ((t_model - tc_ref) / P_ref + 0.5) % 1.0 - 0.5
        time_from_center_model = phase_model * P_ref
        sort_m = np.argsort(phase_model)

        y_planet = self._phased_lc_grid(shared, p_idx, t_model)[i][:, p_idx]

        rows = self.rows(i)
        data_terms = shared["data_terms"][rows]
        other_mask = np.ones(planets.n_elements, dtype=bool)
        other_mask[p_idx] = False
        other_terms = np.sum(data_terms[:, other_mask], axis=1)

        baseline = self._point_value(point, self.baseline, i)
        # The data are baseline * ramp * (1 + sum of steps), and the
        # planet terms carry no ramp, so the data side divides by
        # baseline * ramp while the model curve divides by baseline alone
        # (it is a grid at the seeded epoch, where no ramp applies).
        # Without fitramp the ramp is exactly 1.0, so every product below
        # is exact and this is bit-for-bit the ramp-free expression.
        ramp_rows = shared["data_ramp"][rows]
        norm = baseline * ramp_rows
        cleaned_flux = (
            shared["flux_corrected"][rows]
            - norm
            - other_terms * ramp_rows
            - shared["extra_signals"][rows]
        ) / norm
        data_phases = ((self.time[rows] - tc_ref) / P_ref + 0.5) % 1.0 - 0.5

        return {
            "P_ref": P_ref,
            "tc_ref": tc_ref,
            "x_model": time_from_center_model[sort_m],
            "y_model": y_planet[sort_m] / baseline,
            "x_data": data_phases * P_ref,
            "y_data": cleaned_flux,
            "yerr_data": shared["err_corrected"][rows] / norm,
        }

    def plot(self, system, points, filename_prefix="debug"):
        """Render the unphased + phased LC PDFs from plot_data specs.

        The specs are the single description of these plots -- the GUI draws
        the same ones via plotly (see plotrender.py's module docstring).
        """
        from exozippy.plotrender import plot_via_specs

        plot_via_specs(self, system, points, filename_prefix=filename_prefix)

    #: A band whose filter is this one is TESS: on the summary figure its
    #: files share one row per exposure time (cadence).  Filter names are
    #: case-sensitive, as everywhere in EXOZIPPy.
    SUMMARY_TESS_FILTER = "TESS"

    def summary_rows(self, system):
        """Each file's row on the summary figure's transit stack:
        ``{name: (row key, row label)}``, the files sharing a key drawn on
        one row.

        The TESS files (band filter ``SUMMARY_TESS_FILTER``) share a row per
        exposure time (``exptime_s:``, rounded to the second): the sectors of
        one cadence are one instrument setup with one model curve.  Every
        other file keeps its own row (key and label its name and
        ``display_label``): ground-based follow-up from different
        telescopes or nights stays apart even when filter and exposure time
        agree.  A TESS row is labelled with its members' common ``label:``
        when they share one that no other cadence uses, and otherwise
        ``"<label or TESS> <seconds> s"``, e.g. ``"TESS 120 s"`` -- the
        cadence only when every member states an ``exptime_s:`` this
        component accepted (``exptime_stated``).  Without one, or with one
        it rejected, the fit uses the inert 60 s default (no
        smearing), which is not the data's cadence, so such a row is just
        ``"TESS"``.  Two rows with one label are the renderer's to refuse
        (``summary_plot``), since only a figure needs them distinct.
        """
        band = system.band
        by_cadence = {}
        rows = {}
        for i, name in enumerate(self.names):
            b = band.names.index(self.band_names[i])
            if band.filter_names[b] != self.SUMMARY_TESS_FILTER:
                rows[name] = (name, self.display_label(i))
                continue
            seconds = int(round(float(self.exptime_s[i])))
            by_cadence.setdefault(seconds, []).append(i)

        common = {}
        for seconds, members in by_cadence.items():
            labels = {self.plot_label[i] for i in members}
            one = len(labels) == 1 and None not in labels
            common[seconds] = labels.pop() if one else None
        for seconds, members in by_cadence.items():
            base = common[seconds]
            stated = all(self.exptime_stated[i] for i in members)
            if base is not None and list(common.values()).count(base) == 1:
                label = base
            elif stated:
                label = f"{base or 'TESS'} {seconds} s"
            else:
                label = base or "TESS"
            for i in members:
                rows[self.names[i]] = (f"TESS cadence {seconds} s", label)
        return rows

    def plot_data(self, system, point=None):
        """
        GUI charts for the transit photometry: per instrument an
        unphased flux-vs-time chart, and (with a point) one phased chart
        per planet/instrument. point=None returns only the raw data
        traces. See Component.plot_data and chart.Chart.

        Each phased chart is a row of its planet's stack on the summary
        figure (``meta["summary"]``, layout ``stacked_fold``), grouped as
        ``summary_rows`` says.
        """
        from exozippy.chart import Chart, Trace

        specs = []
        full_deps = self._model_trace_param_deps(
            getattr(self, "_lc_full_node", None), system
        )
        matrix_deps = self._model_trace_param_deps(
            getattr(self, "_lc_matrix_node", None), system
        )

        # The baseline, the fitted detrend model and the GP conditional
        # mean enter the panels in numpy (_point_value / detrend_at_data /
        # gp_mean_at_data), not through the symbolic nodes, so the graph
        # walk cannot see them -- without these deps a baseline, detrend
        # coefficient or GP hyperparameter slider would never refresh these
        # charts in the GUI (review 1.12.9 for the GP labels).
        baseline_label = getattr(
            getattr(self, "baseline", None), "label", None
        )
        # The fitramp factor reaches the phased DATA the same way (it is
        # divided out there, and the planet terms carry none of it).
        ramp_labels = [
            getattr(self, name).label
            for name in ("ramp_amp", "ramp_tau")
            if name in self.manifest
        ]
        numpy_deps = (
            ([baseline_label] if baseline_label else [])
            + ramp_labels
            + self.detrend_dep_labels()
            + self.gp_dep_labels()
        )
        full_deps = full_deps + [
            lbl for lbl in numpy_deps if lbl not in full_deps
        ]
        matrix_deps = matrix_deps + [
            lbl for lbl in numpy_deps if lbl not in matrix_deps
        ]

        # ---- Unphased: flux vs time, per instrument -------------------
        # The fitted trend is per observation, so it comes off the DATA
        # rather than going onto the model curve, through the base inverse of
        # the likelihood's own combination (Instrument.detrend_corrected);
        # the raw data without detrend columns.
        flux_corrected, err_corrected = self.detrend_corrected(
            self.flux, self.err, point
        )
        # Every instrument's curve comes from ONE evaluation of the compiled
        # grid layout.  A failed model eval keeps the data-only panels
        # (matching the per-point tolerance of the old hand-drawn loop).
        curves = None
        if point is not None:
            try:
                curves = self._unphased_lc_curves(system, point)
            except Exception as e:  # noqa: BLE001 - bad point/draw
                logger.warning(f"LC model eval failed: {e}")
        for i in range(self.n_elements):
            mask = self.inst_map == i
            traces = []
            deps = []
            if curves is not None:
                t_pretty, y_full = curves[i]
                deps = full_deps
                traces.append(
                    Trace(
                        name="model",
                        role="model",
                        kind="line",
                        x=t_pretty,
                        y=y_full,
                        node=getattr(self, "_lc_full_node", None),
                    )
                )
            traces.append(
                Trace(
                    name=self.names[i],
                    role="data",
                    kind="scatter",
                    x=self.time[mask],
                    y=flux_corrected[mask],
                    yerr=err_corrected[mask],
                    # Black-dot default (the historical PDF look); a user
                    # plot: color/marker still wins via _data_trace_style.
                    style={
                        "color": "k",
                        "marker": ".",
                        **self._data_trace_style(i),
                    },
                )
            )
            specs.append(
                Chart(
                    id=f"{self.prefix}.unphased.{self.names[i]}",
                    component={
                        "yaml_key": self.prefix,
                        "instance": self.names[i],
                    },
                    title=f"Transit Photometry: {self.names[i]}",
                    xlabel="Time [BJD]",
                    ylabel="Relative Flux",
                    traces=traces,
                    param_deps=deps,
                    meta={
                        "phase_folded": False,
                        "instrument": self.names[i],
                        "file_tag": f"LC_unphased_{self.names[i]}",
                        "figsize": (12, 5),
                        # The unphased DATA are detrend-subtracted, so they
                        # move with the point whenever this instrument has
                        # detrend columns (and only then).
                        "dynamic_data": self.total_detrend_cols > 0,
                        "caption": (
                            "Transit photometry from "
                            + latex_escape(self.names[i])
                            + " with the best-fit model (red)."
                            + self.detrend_caption()
                        ),
                    },
                )
            )

        # ---- Phased: one chart per planet/instrument (needs a model) --
        if point is not None:
            planets = system.planet
            # Once per (component, point), not once per planet x instrument
            # (6.5.1); the per-instrument matrices fill in lazily.
            shared = self._phased_lc_shared(system, point)
            summary_rows = self.summary_rows(system)
            for p_idx in range(planets.n_elements):
                for i in range(self.n_elements):
                    # A failed prep skips this panel, exactly as the old
                    # hand-drawn loop skipped its figure.
                    try:
                        prep = self._phased_lc_arrays(
                            system, point, p_idx, i, shared=shared
                        )
                    except Exception as e:  # noqa: BLE001 - bad point/draw
                        logger.warning(f"LC phased model eval failed: {e}")
                        continue
                    mask = self.inst_map == i
                    traces = [
                        Trace(
                            name="model",
                            role="model",
                            kind="line",
                            x=prep["x_model"],
                            y=prep["y_model"],
                            node=getattr(self, "_lc_matrix_node", None),
                            style={"lw": 1.0},
                        ),
                        Trace(
                            name=self.names[i],
                            role="data",
                            kind="scatter",
                            x=prep["x_data"],
                            y=prep["y_data"],
                            yerr=prep["yerr_data"],
                            style={
                                "color": "k",
                                "marker": ".",
                                **self._data_trace_style(i),
                            },
                        ),
                    ]
                    pname = planets.names[p_idx]
                    meta = {
                        "phase_folded": True,
                        "planet": pname,
                        "instrument": self.names[i],
                        "period": prep["P_ref"],
                        "tc": prep["tc_ref"],
                        "file_tag": (f"LC_phased_{self.names[i]}_{pname}"),
                        "figsize": (10, 6),
                        "hline_y": 0.0,
                        "caption": (
                            "Phase-folded transit of planet "
                            + latex_escape(pname)
                            + " in "
                            + latex_escape(self.names[i])
                            + ", other planets removed and normalized "
                            + (
                                "by the baseline and the exponential ramp."
                                if self.fitramp[i]
                                else "by the baseline."
                            )
                            + self.detrend_caption()
                        ),
                        # The phased DATA re-folds with tc/P and its cleaning
                        # subtracts the baseline, other planets and any GP --
                        # all point-dependent, so live evals must re-ship it.
                        "dynamic_data": True,
                        "summary": {
                            "layout": "stacked_fold",
                            "stack": pname,
                            "instrument": self.names[i],
                            "label": self.display_label(i),
                            "color": self.plot_color[i],
                            "row_key": summary_rows[self.names[i]][0],
                            "row_label": summary_rows[self.names[i]][1],
                            # The fold is in days; the stack reads hours.
                            "x_scale": 24.0,
                            "xlabel": "Time from Mid-Transit [hr]",
                            "ylabel": "Normalized Flux + Constant",
                        },
                    }
                    # Zoom to +/- t14 around mid-transit when a transit
                    # duration is known for this planet.  `planet.t14`, not
                    # `transit.t14`: the durations are planet geometry now
                    # (review 8.8.7).  Read through _point_value like the
                    # period and tc above, NOT `point["planet.t14"][p_idx]`:
                    # this read sits outside the panel's try, so a scalar or
                    # short t14 in the point IndexError'ed on p_idx > 0 and
                    # killed plot_data for the whole component (review
                    # 2.5.4); the helper falls back to element 0 for a
                    # broadcast scalar and to the Parameter's own start when
                    # the point lacks it.  A non-finite duration (no start
                    # to fall back on) simply means no zoom.
                    x_range = None
                    t14_param = getattr(planets, "t14", None)
                    if t14_param is not None:
                        t14_ref = self._point_value(point, t14_param, p_idx)
                        if np.isfinite(t14_ref):
                            x_range = [-t14_ref, t14_ref]
                    specs.append(
                        Chart(
                            id=f"{self.prefix}.phased.{self.names[i]}.{pname}",
                            component={
                                "yaml_key": self.prefix,
                                "instance": self.names[i],
                            },
                            title=f"Phased LC: {pname} -- {self.names[i]}",
                            xlabel=f"Time from Mid-Transit [d] (P = {prep['P_ref']:.5f} d)",
                            ylabel="Flux / Baseline - 1",
                            traces=traces,
                            param_deps=matrix_deps,
                            x_range=x_range,
                            meta=meta,
                        )
                    )

        return specs
