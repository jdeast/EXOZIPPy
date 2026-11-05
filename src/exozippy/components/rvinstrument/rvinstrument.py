import logging

import numpy as np

logger = logging.getLogger(__name__)

import astropy.units as u
import pymc as pm
import pytensor
import pytensor.tensor as pt

from exozippy.components.instrument import Instrument
from exozippy.constants import C_MPS
from exozippy.outputs.prose import get_collector
from exozippy.outputs.texutils import latex_escape

from .redshift import DEFAULT_EXPOSURE_NODES

#: ``data_format:`` values.  ``rv`` (the default, and what an absent key
#: means) is today's file: barycentric-corrected velocities, modeled as
#: gamma + the reflex.  ``raw_redshift`` is the measured c z before any
#: barycentric correction, which the model corrects itself
#: (``rvinstrument.md``, "Raw redshifts"; ``redshift.py``).
DATA_FORMAT_RV = "rv"
RAW_REDSHIFT = "raw_redshift"
DATA_FORMATS = (DATA_FORMAT_RV, RAW_REDSHIFT)
#: Keys that exist only on a raw_redshift file.
RAW_ONLY_KEYS = ("observer_location", "epoch", "exposure_meter", "ninterp")
#: The optional column roles that resolve each exposure in time
#: (rvinstrument.md, "The exposure average"), all in SECONDS of the file's
#: own time scale: ``exptime`` the duration of a uniform-flux exposure (the
#: FITS EXPTIME), ``t_rms`` the flux-weighted RMS of the photon arrival
#: times about the flux-weighted midpoint, ``t_skew3`` (s^3) their third
#: central moment (needs ``t_rms``).  Read only when ``columns:`` names
#: them, in this order after time/rv/err.
EXPOSURE_ROLES = ("exptime", "t_rms", "t_skew3")
#: An exposure_meter: group's midpoint matches a data-file tag when the
#: two differ by less than this (days, in the file's own units after
#: time_offset): 0.0864 s.  Write the midpoint exactly as in the data file.
METER_MATCH_TOL_D = 1e-6


class RVInstrument(Instrument):
    prose_noun = "radial velocity"
    # The fitted trend is ADDED to the model (Instrument.DETREND_SPACE).
    DETREND_SPACE = "additive"
    # The raw-redshift exposure average reads its own ninterp: (Gauss nodes,
    # rvinstrument.md "The exposure average"), so the base's refusal of the
    # exposure keys on a non-smearing child leaves that one alone; a
    # corrected file's ninterp: is refused by RAW_ONLY_KEYS.
    OWN_EXPOSURE_KEYS = ("ninterp",)

    def __init__(self, config, config_manager):
        super().__init__(config, config_manager)
        self.label = "Instrument Parameters"
        self.units = [
            self._parse_rv_unit(i, c) for i, c in enumerate(self.config)
        ]
        # Which star the RVs are of; its Doppler signal is the sum over
        # every orbit that star is a body of (planetary reflex and stellar
        # companions alike).
        # Index or name (as the schema below advertises): resolved through
        # the one shared translator, which reads the star instance names off
        # the raw system config -- available here, where system.star is not.
        self.star_ndx = [
            self.resolve_star_ndx(
                c.get("star_ndx"),
                f"[{self.prefix}] {c.get('name', i)} star_ndx",
            )
            for i, c in enumerate(self.config)
        ]
        # Rossiter-McLaughlin: a file may set `rm: <orbit_name>` to add the
        # in-transit RM distortion of that orbit to this instrument's RV
        # model (off by default -> the RV likelihood is unchanged). Optional
        # `rm_band: <band_name>` selects the limb darkening; else defaults.
        self.rm_orbit = [c.get("rm") for c in self.config]
        self.rm_band = [c.get("rm_band") for c in self.config]
        # `rm_model: hirano2010 | hirano2011` (default hirano2011). hirano2010 is
        # the fast closed-form series; hirano2011 the disk integral.
        self.rm_model = [c.get("rm_model", "hirano2011") for c in self.config]
        _valid_rm = {"hirano2010", "hirano2011"}
        for m in self.rm_model:
            if m not in _valid_rm:
                raise ValueError(
                    f"[{self.prefix}] unknown rm_model {m!r}; expected one of "
                    f"{sorted(_valid_rm)}."
                )
        # Light-travel time, per file, on by default: the star's own
        # retardation across its orbits on every reflex column (_rv_model,
        # _emission_time; JDE 2026-11-04: "we should do it right and
        # include the ltt term"), and the RM occultation geometry's own
        # (components/rm.py, components/ltt.py) on a file with `rm:`.
        self.light_travel_time = [
            bool(c.get("light_travel_time", True)) for c in self.config
        ]
        self._ltt_unsupported_warned = False
        self.total_detrend_cols = 0
        # data_format: raw_redshift -- the barycentric correction inside the
        # model (rvinstrument.md, "Raw redshifts").  Parsed here so a
        # malformed spec fails at construction.
        self.data_formats = [
            self._parse_data_format(i, c) for i, c in enumerate(self.config)
        ]
        self.raw = [f == RAW_REDSHIFT for f in self.data_formats]
        self.observers = [
            self._parse_raw_keys(i, c) if self.raw[i] else None
            for i, c in enumerate(self.config)
        ]

    @property
    def prefix(self):
        return "rvinstrument"

    def _parse_data_format(self, i, c):
        """File ``i``'s ``data_format:`` (absent: ``rv``, today's file)."""
        fmt = c.get("data_format", DATA_FORMAT_RV)
        if fmt not in DATA_FORMATS:
            raise ValueError(
                f"[{self.prefix}.{self.names[i]}] data_format: {fmt!r} is "
                f"not one of {list(DATA_FORMATS)} ('rv': barycentric-"
                f"corrected velocities, the default; 'raw_redshift': the "
                f"measured c*z before any barycentric correction)."
            )
        if fmt != RAW_REDSHIFT:
            for key in RAW_ONLY_KEYS:
                if key in c:
                    raise ValueError(
                        f"[{self.prefix}.{self.names[i]}] {key}: belongs to "
                        f"data_format: raw_redshift; a corrected-velocity "
                        f"file does not read it."
                    )
        return fmt

    def _parse_raw_keys(self, i, c):
        """Validate a raw_redshift file's keys; returns its observer.

        The observatory is ``observer_location:`` (the vocabulary of
        ``ephemeris.get_observer_position``: an astropy site name, a
        ``'lon,lat,height'`` string, or a spacecraft ``.eph`` file) and
        nothing else -- ``time_location`` would be a second spelling of the
        same place, and the geocenter ('earth', 'moon') is no spectrograph's
        site (the Earth's rotation alone is 0.46 km/s).  ``time_frame`` must
        be SAID: its default, bjd, is what a corrected file carries and
        almost never what a raw spectrum's exposure midpoint is.
        """
        where = f"[{self.prefix}.{self.names[i]}] data_format: raw_redshift"
        loc = c.get("observer_location")
        if not isinstance(loc, str) or not loc:
            raise ValueError(
                f"{where} needs observer_location: -- the observatory, as "
                f"an astropy site name, a 'lon_deg,lat_deg,height_m' string "
                f"or a spacecraft .eph file (e.g. 'gaia'); got {loc!r}."
            )
        if loc in ("earth", "moon"):
            raise ValueError(
                f"{where}: observer_location: {loc!r} is a body's centre, "
                f"not an observatory; name the site (the Earth's rotation "
                f"is up to 0.46 km/s of the barycentric velocity)."
            )
        if "time_location" in c:
            raise ValueError(
                f"{where} reads the observatory from observer_location:; "
                f"time_location: would be a second spelling of the same "
                f"place -- remove it."
            )
        if "time_frame" not in c:
            raise ValueError(
                f"{where} needs time_frame: said explicitly -- jd for the "
                f"observatory's own clock (a flux-weighted exposure "
                f"midpoint, with time_scale: utc), or bjd for barycentric "
                f"tags.  The default (bjd) is a corrected file's frame."
            )
        ep = c.get("epoch")
        if ep is not None and (
            isinstance(ep, bool)
            or not isinstance(ep, (int, float, np.integer, np.floating))
            or not ep > 2_000_000.0
        ):
            raise ValueError(
                f"{where}: epoch: is the BJD_TDB reference epoch of "
                f"star.ra/dec/pm_ra/pm_dec (e.g. 2457389.0 for Gaia DR3's "
                f"J2016.0); got {ep!r}."
            )
        n = c.get("ninterp", None)
        if n is not None and (
            isinstance(n, bool)
            or not isinstance(n, (int, np.integer))
            or n < 1
        ):
            raise ValueError(
                f"{where}: ninterp: is the number of Gauss nodes per "
                f"exposure (a positive integer; default "
                f"{DEFAULT_EXPOSURE_NODES}); got {n!r}."
            )
        meter = c.get("exposure_meter")
        if meter is not None and (not isinstance(meter, str) or not meter):
            raise ValueError(
                f"{where}: exposure_meter: is the path of the exposure-meter "
                f"file (columns: t_mid, t, dt, counts); got {meter!r}."
            )
        # Resolve the location now, so a typo fails at construction.
        from exozippy.ephemeris import _resolve_observer

        _resolve_observer(loc)
        return loc

    def _parse_rv_unit(self, i, entry):
        """Resolve a file's ``unit:`` key to an astropy Unit.

        The YAML value is a plain string (``unit: km/s``), so it has to go
        through ``u.Unit`` before ``load_data`` can call ``.to()`` on it --
        exactly what ``astrometryinstrument`` does for its ``sep_unit``.
        Anything astropy accepts as a velocity works; the default is m/s.
        """
        raw = entry.get("unit", "m/s")
        name = entry.get("name", i)
        try:
            unit = u.Unit(raw)
            unit.to(u.m / u.s)
        except Exception as exc:
            raise ValueError(
                f"[{self.prefix}] {name}: unit: {raw!r} is not a velocity "
                f"astropy can parse (e.g. 'm/s', 'km/s', 'km s-1')."
            ) from exc
        return unit

    @classmethod
    def get_utilities(cls):
        from ...utilities import lomb_scargle
        from ...utilities.registry import (
            UtilitySpec,
            argparse_subprocess_runner,
        )

        return [
            UtilitySpec(
                name="lomb_scargle",
                label="Lomb-Scargle periodogram",
                description=(
                    "Lomb-Scargle radial-velocity periodogram: report the "
                    "period, epoch and semi-amplitude of the strongest "
                    "signal."
                ),
                component_keys=["rvinstrument"],
                available=True,
                build_parser=lomb_scargle.build_parser,
                run=argparse_subprocess_runner(
                    "exozippy.utilities.lomb_scargle"
                ),
            ),
        ]

    @classmethod
    def config_schema(cls):
        return [
            {
                "key": "file",
                "kind": "datafile",
                "accepts": "*.rv",
                "required": True,
                "doc": (
                    "Whitespace-delimited radial-velocity data; columns are "
                    "time, RV, RV_err, then optional detrend columns. Comment "
                    "lines start with '#'."
                ),
            },
            {
                "key": "star_ndx",
                "kind": "ref",
                "accepts": ["star"],
                "required": False,
                "doc": (
                    "Index or name of the observed star (default 0); the "
                    "'star.<name>' path spelling works too. The RV model "
                    "sums orbit.K over every orbit containing this star."
                ),
            },
            {
                "key": "unit",
                "kind": "option",
                "accepts": None,
                "required": False,
                "doc": (
                    "Astropy unit string for the RV/error columns. Default "
                    "'m/s'."
                ),
            },
            {
                "key": "data_format",
                "kind": "option",
                "accepts": list(DATA_FORMATS),
                "required": False,
                "doc": (
                    "'rv' (default): barycentric-corrected velocities, "
                    "modeled as gamma + the reflex. 'raw_redshift': the "
                    "measured c*z (in `unit:`, default m/s) BEFORE any "
                    "barycentric correction, which the model applies "
                    "itself from the fitted star.ra/dec/pm_ra/pm_dec/"
                    "parallax/rv and every orbit (Wright & Eastman 2014); "
                    "needs observer_location and an explicit time_frame, "
                    "and a prior on star.rv."
                ),
            },
            {
                "key": "observer_location",
                "kind": "option",
                "accepts": None,
                "required": False,
                "doc": (
                    "raw_redshift only: the observatory -- an astropy site "
                    "name, 'lon_deg,lat_deg,height_m', or a spacecraft "
                    ".eph file (e.g. 'gaia')."
                ),
            },
            {
                "key": "epoch",
                "kind": "option",
                "accepts": None,
                "required": False,
                "doc": (
                    "raw_redshift only: BJD_TDB reference epoch of "
                    "star.ra/dec/pm (e.g. 2457389.0 = J2016.0 for Gaia "
                    "DR3). Required unless the system has absolute "
                    "astrometry, whose epoch it must then equal."
                ),
            },
            {
                "key": "exposure_meter",
                "kind": "option",
                "accepts": None,
                "required": False,
                "doc": (
                    "raw_redshift only: path of an exposure-meter file, "
                    "one row per meter bin: t_mid (the exposure's midpoint "
                    "exactly as the data file writes it), t (bin center, "
                    "the data file's time system), dt (bin width, s), "
                    "counts. The model averages each spectrum's predicted "
                    "redshift over its photons (Tronsgaard et al. 2019); "
                    "overrides the exposure columns for that exposure."
                ),
            },
            {
                "key": "ninterp",
                "kind": "option",
                "accepts": None,
                "required": False,
                "doc": (
                    "raw_redshift only: Gauss nodes per exposure for the "
                    "photon-weighted average (default "
                    f"{DEFAULT_EXPOSURE_NODES}, converged to < 0.001 mm/s "
                    "for 1-hour exposures); read only with an exptime "
                    "column or an exposure_meter."
                ),
            },
            cls._mask_config_schema(),
            cls._columns_config_schema(
                ("time", "rv", "err"),
                note=(
                    "A raw_redshift file may also name exposure columns, "
                    "in seconds of its own time scale: 'exptime' (a "
                    "uniform-flux exposure's duration), 't_rms' (the "
                    "flux-weighted RMS of the photon arrival times about "
                    "the midpoint) and 't_skew3' (their third central "
                    "moment, s^3; needs t_rms); the predicted redshift is "
                    "averaged over the exposure they describe."
                ),
            ),
            *cls._time_config_schema(),
            cls._plot_style_config_schema(),
            cls._label_config_schema(),
            cls._gp_config_schema(),
            cls._likelihood_config_schema(),
            cls._out_frac_prior_config_schema(),
        ]

    def load_data(self, system):
        """Stage 1: Load CSVs and generate data-driven bounds/inits."""
        self.gamma_init = [0.0] * self.n_elements
        self.jittervar_lower = [0.0] * self.n_elements
        self._raw_files = [None] * self.n_elements
        seed_blocks = []

        blocks = self._concat_blocks()
        for i in range(self.n_elements):
            if self.raw[i]:
                # The raw redshifts: observer-frame times, the observer's
                # barycentric state, and a start-star correction for the
                # SEEDS only (the likelihood corrects in the graph).
                df, t_bary, v_seed = self._load_raw_file(system, i)
                factor = self.units[i].to(u.solRad / u.d)
                m_s_factor = self.units[i].to(u.m / u.s)
                self.gamma_init[i] = float(np.mean(v_seed))
                self.jittervar_lower[i] = self._jitter_floor(
                    df.iloc[:, 2].values, factor=m_s_factor
                )
                blocks.add(
                    i,
                    time=t_bary,
                    obs=df.iloc[:, 1].values * factor,
                    err=df.iloc[:, 2].values * factor,
                    df=df,
                )
                seed_blocks.append(
                    v_seed * float((u.m / u.s).to(u.solRad / u.d))
                )
                continue
            # Shared reader: columns:, mask:, time_* conversion, then one
            # sort per file before anything is derived from it, keeping the
            # RVs, errors and detrend columns aligned by construction.
            df = self._read_data(i, roles=("time", "rv", "err"), detrend=True)
            factor = self.units[i].to(u.solRad / u.d)

            m_s_factor = self.units[i].to(u.m / u.s)
            self.gamma_init[i] = np.mean(df.iloc[:, 1].values) * m_s_factor
            self.jittervar_lower[i] = self._jitter_floor(
                df.iloc[:, 2].values, factor=m_s_factor
            )

            blocks.add(
                i,
                time=df.iloc[:, 0].values,
                obs=df.iloc[:, 1].values * factor,
                err=df.iloc[:, 2].values * factor,
                df=df,
            )
            seed_blocks.append(df.iloc[:, 1].values * factor)

        # Shared accumulator: concatenation (time/rv/err), inst_map, the
        # per-file row ranges, the block-diagonal detrend matrix, and the
        # optional GP / robust-likelihood hooks.  self.err ends up in
        # solRad/d while the GP amplitude and out_scale are declared in m/s,
        # hence user_factor.
        blocks.finalize(
            "rv",
            user_factor=(u.solRad / u.d).to(u.m / u.s),
            hint_unit="m/s",
        )
        # The velocities every SEED reads (the K estimate, the Lomb-Scargle
        # search, the conjunction-epoch information): the data themselves
        # for a corrected file -- the very same array, so nothing moves --
        # and for a raw file its redshifts corrected with the START star
        # (_load_raw_file).  The likelihood never reads this.
        self.rv_seed = (
            np.concatenate(seed_blocks).astype(float)
            if any(self.raw)
            else self.rv
        )
        if any(self.raw):
            self._concat_raw_observers()

        self.k_init = self._estimate_k_init()

        # Blind seeding: measure the period and conjunction epoch from the
        # velocities when nothing else supplies them.  Stage 1a, not stage 2
        # -- see components/globalsearch.py for why (Orbit builds tc's hard
        # window at stage 2 from whatever start it can see).
        self.ls_signal = None
        self._seed_from_lombscargle(system)

    # ------------------------------------------------------------------
    # data_format: raw_redshift (rvinstrument.md, "Raw redshifts")
    # ------------------------------------------------------------------
    def _load_raw_file(self, system, i):
        """Read raw file ``i``: ``(df, t_bary, v_seed)``.

        ``df`` -- the shared reader's frame, its time column the OBSERVER's
        coordinate time JD_TDB at reception (``_to_observer_tdb``), any
        ``exptime`` column split off (it is not a detrend column);
        ``t_bary`` -- each spectrum's BJD_TDB at the START star (what the
        plots, the GP's ordering and every seed read as ``self.time``; the
        likelihood forms its own from the fitted star);
        ``v_seed`` -- the redshifts corrected with the start star, m/s:
        ``c[(1 + z_B)(1 + z_meas) - 1]`` (``redshift.barycentric_redshift``),
        with ``1/(1 + z_B)`` photon-averaged over the exposure where it is
        resolved, as the model does.

        The observer's barycentric state and the solar-system potential
        at each sub-exposure node are parameter-independent and computed
        here, once (``redshift.observer_terms``); the nodes and their
        photon weights come from ``_exposure_nodes``.
        """

        from .redshift import (
            barycentric_redshift,
            observer_terms,
            star_geometry,
        )

        loc = self.observers[i]
        spec = self.column_specs[i] or {}
        extra = tuple(r for r in EXPOSURE_ROLES if r in spec)
        if "t_skew3" in extra and "t_rms" not in extra:
            raise ValueError(
                f"[{self.prefix}.{self.names[i]}] columns: t_skew3 (the "
                f"third central moment of the photon arrival times) needs "
                f"t_rms (their RMS) as well."
            )
        roles = ("time", "rv", "err") + extra
        df, disk_rows = self._read_data(
            i, roles=roles, detrend=True, observer_time=loc, row_index=True
        )
        moments = {}
        for k, role in enumerate(extra):
            moments[role] = df.pop(3 + k).values.astype(float)
        df.columns = range(df.shape[1])
        t_tdb = df.iloc[:, 0].values.astype(float)
        node_row, t_node, w_node = self._exposure_nodes(
            i, t_tdb, moments, disk_rows
        )
        eph = self.time_specs[i]["ephemeris"]
        averaged = t_node.size > t_tdb.size
        obs = observer_terms(t_tdb, loc, eph)
        obs_node = observer_terms(t_node, loc, eph) if averaged else obs
        star = self._seed_star(i)
        ep = self.config[i].get("epoch")
        # The seed's epoch: the file's own, else its midpoint -- the
        # correction moves by ~1 cm/s per year of epoch error, which no
        # start value notices.  The model's epoch is settled at stage 3
        # (_reference_epoch), when the astrometry has loaded.
        epoch = float(ep) if ep is not None else 0.5 * (t_tdb[0] + t_tdb[-1])
        t_bary = star_geometry(t_tdb, obs["r_au"], *star, epoch)["t_bary"]
        if averaged:
            # 1 + z_meas = (1 + z_true) <1/(1 + z_B)>: the photon mean of
            # the instantaneous ratio, with z_true the star's constant one.
            inv = np.bincount(
                node_row,
                weights=w_node
                / (1.0 + barycentric_redshift(t_node, obs_node, star, epoch)),
                minlength=t_tdb.size,
            )
            z_b = 1.0 / inv - 1.0
        else:
            z_b = barycentric_redshift(t_tdb, obs, star, epoch)
        cz = df.iloc[:, 1].values * float(self.units[i].to(u.m / u.s))
        v_seed = C_MPS * ((1.0 + z_b) * (1.0 + cz / C_MPS) - 1.0)
        self._raw_files[i] = {
            "t_tdb": t_tdb,
            "node_row": node_row,
            "node_t_tdb": t_node,
            "node_w": w_node,
            "node_obs": obs_node,
        }
        return df, np.asarray(t_bary, dtype=float), v_seed

    def _exposure_nodes(self, i, t_tdb, moments, disk_rows):
        """Raw file ``i``'s sub-exposure quadrature: ``(row, t, w)``, one
        entry per node -- the file-local row (index into ``t_tdb``) it
        belongs to, its observer JD_TDB and its photon weight (each row's
        summing to 1).  rvinstrument.md, "The exposure average".

        Per exposure, the most complete description it has wins:

        1. its ``exposure_meter:`` bins -> the ``ninterp``-node Gauss rule
           of the meter's photon distribution, each bin a uniform box
           (``redshift.weighted_exposure_rule``; exact to degree
           2 ninterp - 1);
        2. its ``t_rms`` (and ``t_skew3``) columns -> the 2-node rule with
           exactly those moments about the tag (``redshift.moment_rule``;
           exact to degree 3, so its error is the photon distribution's
           unknown 4th moment: < 0.5 mm/s for a 1-hour exposure);
        3. its ``exptime`` column -> ``ninterp``-node Gauss-Legendre over
           ``[t - exptime/2, t + exptime/2]``, uniform flux;
        4. nothing -> one node at the tag (the model at the flux-weighted
           midpoint: up to 1.3 m/s (exposure / 1 h)^2 off, Tronsgaard et
           al. 2019).
        """
        from .redshift import (
            moment_rule,
            uniform_exposure_rule,
            weighted_exposure_rule,
        )

        c = self.config[i]
        where = f"[{self.prefix}.{self.names[i]}]"
        n = int(c.get("ninterp", DEFAULT_EXPOSURE_NODES))
        N = t_tdb.size
        for role, vals in moments.items():
            ok = np.isfinite(vals)
            if role != "t_skew3":
                ok &= vals > 0.0
            if not ok.all():
                raise ValueError(
                    f"{where} {role} column: must be "
                    + ("finite" if role == "t_skew3" else "positive (s)")
                    + f"; data rows {sorted(int(r) for r in disk_rows[~ok])} "
                    f"are not ({vals[~ok][:5].tolist()})."
                )
        meter = self._read_exposure_meter(i, disk_rows)
        if "ninterp" in c and not meter and "exptime" not in moments:
            logger.warning(
                "%s ninterp: %d is unread: it sets the nodes of an exptime "
                "column or an exposure_meter:, and the file has neither.",
                where,
                n,
            )
        x_u, w_u = uniform_exposure_rule(n)
        rows, ts, ws = [], [], []
        bare, worst = [], 0.0
        for j in range(N):
            if j in meter:
                t_k, w_k = weighted_exposure_rule(*meter[j], n)
                # The meter's photon-weighted mean time IS the flux-weighted
                # midpoint the file's time column claims to be.
                worst = max(worst, abs(float(np.sum(w_k * t_k)) - t_tdb[j]))
            elif "t_rms" in moments:
                skew = moments["t_skew3"][j] if "t_skew3" in moments else 0.0
                x, w_k = moment_rule(moments["t_rms"][j], skew)
                t_k = t_tdb[j] + x / 86400.0
            elif "exptime" in moments:
                t_k = t_tdb[j] + x_u * (moments["exptime"][j] / 86400.0)
                w_k = w_u
            else:
                t_k, w_k = np.array([t_tdb[j]]), np.array([1.0])
                if meter:
                    bare.append(int(disk_rows[j]))
            rows.append(np.full(t_k.size, j, dtype=int))
            ts.append(t_k)
            ws.append(w_k)
        if bare:
            logger.warning(
                "%s exposure_meter: no bins for data rows %s and no "
                "exposure columns: those spectra are modeled at their "
                "midpoints (up to ~1.3 m/s (exposure/1 h)^2 off; "
                "Tronsgaard et al. 2019).",
                where,
                bare,
            )
        if worst * 86400.0 > 1.0:
            logger.warning(
                "%s exposure_meter: a spectrum's photon-weighted mean time "
                "differs from its time-column tag by up to %.1f s; the model "
                "uses the meter (the tag only places the plotted point).",
                where,
                worst * 86400.0,
            )
        return np.concatenate(rows), np.concatenate(ts), np.concatenate(ws)

    def _read_exposure_meter(self, i, disk_rows):
        """Raw file ``i``'s ``exposure_meter:`` file as ``{row: (t, counts)}``
        keyed by the file-local (sorted, unmasked) row, each bin expanded
        into its 3-node Gauss-Legendre sub-points (a bin is a uniform box
        of photons: its moments through degree 5 are then exact), times
        converted exactly as the data's (``_to_observer_tdb``).  ``{}``
        without one.

        The file (long format; rvinstrument.md, docs/raw_redshift_data.md):
        whitespace columns ``t_mid t dt counts``, ``#`` comments.
        ``t_mid`` is the exposure's midpoint EXACTLY as the data file's
        time column writes it, which is how a bin finds its exposure
        (within ``METER_MATCH_TOL_D``; no match, or more than one, raises
        -- a bin of a masked exposure is ignored); ``t`` the bin center in
        the data file's own time system, ``dt`` its width (s), ``counts``
        the photons (or anything proportional) in it.
        """
        from .redshift import uniform_exposure_rule

        path = self.config[i].get("exposure_meter")
        if path is None:
            return {}
        where = f"[{self.prefix}.{self.names[i]}] exposure_meter: {path!r}"
        try:
            tab = np.loadtxt(path, comments="#", ndmin=2)
        except (OSError, ValueError) as e:
            raise ValueError(
                f"{where} could not be read as four numeric columns "
                f"(t_mid, t, dt, counts): {e}"
            ) from e
        if tab.shape[1] != 4 or tab.shape[0] == 0:
            raise ValueError(
                f"{where} must have four columns (t_mid, t, dt[s], counts) "
                f"and at least one row; got shape {tab.shape}."
            )
        t_mid, t, dt, counts = tab.T
        if not np.all(np.isfinite(dt) & (dt > 0.0)):
            raise ValueError(f"{where}: dt (bin width, s) must be > 0.")
        if not np.all(np.isfinite(counts) & (counts >= 0.0)):
            raise ValueError(f"{where}: counts must be finite and >= 0.")
        # The data file's tags as written, every on-disk row (masked ones
        # too, so a masked exposure's bins are recognized, not unmatched).
        spec = self.column_specs[i] or {}
        roles = ("time", "rv", "err") + tuple(
            r for r in EXPOSURE_ROLES if r in spec
        )
        tags = (
            self._select_columns(self._read_table(i), i, roles, detrend=True)
            .iloc[:, 0]
            .values.astype(float)
        )
        local = {int(r): j for j, r in enumerate(disk_rows)}
        out_t, out_c, out_row = [], [], []
        x3, w3 = uniform_exposure_rule(3)
        t_obs = self._to_observer_tdb(t, i, self.observers[i])
        for m in np.unique(t_mid):
            hit = np.nonzero(np.abs(tags - m) < METER_MATCH_TOL_D)[0]
            if hit.size != 1:
                raise ValueError(
                    f"{where}: t_mid {m!r} matches "
                    + ("no" if hit.size == 0 else f"{hit.size}")
                    + f" data-file time tags within {METER_MATCH_TOL_D} d; "
                    f"write each exposure's midpoint exactly as the data "
                    f"file does."
                )
            r = int(hit[0])
            if r not in local:
                continue  # a masked exposure
            sel = t_mid == m
            if not np.any(counts[sel] > 0.0):
                raise ValueError(
                    f"{where}: exposure t_mid {m!r} has no counts."
                )
            sub_t = (
                t_obs[sel][:, None] + x3[None, :] * dt[sel][:, None] / 86400.0
            )
            sub_c = counts[sel][:, None] * w3[None, :]
            out_row.append(local[r])
            out_t.append(sub_t.ravel())
            out_c.append(sub_c.ravel())
        return {r: (tt, cc) for r, tt, cc in zip(out_row, out_t, out_c)}

    def _seed_star(self, i):
        """The start star for file ``i``'s seeds, internal units:
        ``(ra, dec, pm_ra, pm_dec, parallax, rv)``.

        ra/dec must be INFORMED (``require_star_radec`` raises otherwise:
        a correction toward the defaults.yaml placeholder would be wrong by
        tens of km/s).  The rest are the user's starts where given; where
        not, the seed takes 0 (pm, rv) and 1 mas (parallax) -- terms that
        move this SEED by < 1 m/s (the systemic rv enters only through its
        product with the barycentric velocity, ~v_B rv / c), never the model.
        """
        from exozippy.components.star.position import (
            _user_start,
            require_star_radec,
        )

        cm = self.config_manager
        s = int(self.star_ndx[i])
        ra, dec = require_star_radec(
            cm, s, f"[{self.prefix}.{self.names[i]}] data_format: raw_redshift"
        )

        def start(name, absent):
            path = f"star.{s}.{name}"
            val = _user_start(cm, path)
            return absent if val is None else cm.to_internal(path, val)

        plx = start("parallax", None)
        if plx is None:
            dist = start("distance", None)
            plx = 1.0 if dist is None else 1000.0 / dist
        return (
            ra,
            dec,
            start("pm_ra", 0.0),
            start("pm_dec", 0.0),
            plx,
            start("rv", 0.0),
        )

    def _concat_raw_observers(self):
        """The raw files' sub-exposure nodes (``_exposure_nodes``),
        concatenated in row order, and the rows they sit on
        (``self._raw_rows``, a concrete index array: the likelihood writes
        the raw model into exactly those rows).

        ``self._raw_nodes``: ``t_tdb`` (M,), ``obs`` (``observer_terms`` at
        the nodes), ``row`` (M,) the position in ``_raw_rows`` each node
        averages into, ``w`` (M,) its photon weight, ``file`` (M,) its file
        index, ``blocks`` the per-file node blocks (``_rv_model``'s
        ``blocks``), and ``averaged`` -- False when every spectrum is one
        node (no exposure information anywhere), the model then the one
        at each midpoint.
        """
        idx = [i for i in range(self.n_elements) if self.raw[i]]
        self._raw_rows = np.concatenate(
            [np.arange(*self.row_ranges[i]) for i in idx]
        ).astype(int)
        f = [self._raw_files[i] for i in idx]
        row_off = np.cumsum([0] + [p["t_tdb"].size for p in f])
        node_off = np.cumsum([0] + [p["node_t_tdb"].size for p in f])
        obs = {
            key: np.concatenate([p["node_obs"][key] for p in f], axis=axis)
            for key, axis in (
                ("r_au", 0),
                ("beta", 0),
                ("gamma", 0),
                ("one_plus_zgrav", 0),
                ("shapiro_hat", 1),
                ("shapiro_coef", 1),
                ("shapiro_beta", 1),
            )
        }
        self._raw_nodes = {
            "t_tdb": np.concatenate([p["node_t_tdb"] for p in f]),
            "obs": obs,
            "row": np.concatenate(
                [p["node_row"] + row_off[k] for k, p in enumerate(f)]
            ).astype(int),
            "w": np.concatenate([p["node_w"] for p in f]),
            "file": np.concatenate(
                [np.full(p["node_t_tdb"].size, i) for i, p in zip(idx, f)]
            ).astype(int),
            "blocks": [
                (i, np.arange(node_off[k], node_off[k + 1]))
                for k, i in enumerate(idx)
            ],
        }
        self._raw_nodes["averaged"] = bool(
            self._raw_nodes["t_tdb"].size > self._raw_rows.size
        )

    def _reference_epoch(self, system):
        """The BJD_TDB epoch star.ra/dec/pm_ra/pm_dec refer to.

        ONE star, one set of astrometric parameters, one epoch: with
        absolute astrometry in the system it is the astrometry's
        (``AstrometryInstrument.epoch``), and a raw file's ``epoch:`` must
        equal it; without, every raw file must name it, and agree.
        """
        from ..astrometryinstrument.astrometryinstrument import ABSOLUTE_MODES

        stated = sorted(
            {
                float(self.config[i]["epoch"])
                for i in range(self.n_elements)
                if self.raw[i] and "epoch" in self.config[i]
            }
        )
        if len(stated) > 1:
            raise ValueError(
                f"[{self.prefix}] raw_redshift files name different "
                f"epoch: values {stated}; star.ra/dec/pm are one set of "
                f"parameters with one reference epoch."
            )
        ast = system.active_components.get("astrometryinstrument")
        if ast is not None and any(m in ABSOLUTE_MODES for m in ast.modes):
            if stated and stated[0] != float(ast.epoch):
                raise ValueError(
                    f"[{self.prefix}] raw_redshift epoch: {stated[0]} "
                    f"disagrees with the astrometry's reference epoch "
                    f"{float(ast.epoch)} for star.ra/dec/pm; drop epoch: "
                    f"from the rvinstrument (the astrometry's is used)."
                )
            return float(ast.epoch)
        if not stated:
            raise ValueError(
                f"[{self.prefix}] data_format: raw_redshift needs epoch: "
                f"(BJD_TDB), the reference epoch of star.ra/dec/pm_ra/"
                f"pm_dec -- e.g. 2457389.0 for Gaia DR3's J2016.0 -- since "
                f"no absolute astrometry in this system supplies one."
            )
        return stated[0]

    def _check_raw_rv_prior(self, system):
        """star.rv must carry a prior: the data constrain only each
        instrument's product (1 + gamma_i/c) = (1 + z0_i)(1 + rv/c)
        (rvinstrument.md, "What the data constrain"), so without one rv
        is flat over +/-c, and at a relativistic rv the terms that DO
        read it alone (gamma_*, the perspective stretch) are nonsense."""
        # A star in an orbit group shares its anchor's systemic rv
        # (Star.ORBIT_LINKED_PARAMS), so the prior lives on the anchor.
        s = int(self.star_ndx[0])
        s = int(system.star.orbit_anchors(system).get(s, s))
        r = self.config_manager.resolve("star", "rv", element=s)
        # A Gaussian (star/defaults.yaml gives rv no sigma, so any sigma is
        # the user's), or bounds that confine rv to |rv| < 0.01 c: either
        # states what rv can be.  The defaults' +/-3e8 m/s states nothing.
        sigma = r["sigma"]
        has_sigma = sigma is not None and bool(np.all(np.isfinite(sigma)))
        bounded = (
            r["lower"] is not None
            and r["upper"] is not None
            and float(np.max(np.abs([r["lower"], r["upper"]]))) < 0.01 * C_MPS
        )
        if not (has_sigma or bounded):
            raise ValueError(
                f"[{self.prefix}] data_format: raw_redshift needs a prior on "
                f"star.{s}.rv (the star's absolute radial velocity, m/s): "
                f"the redshifts constrain only (1 + gamma_i/c) = (1 + z0_i)"
                f"(1 + rv/c) per instrument.  A Gaia DR3 radial_velocity is "
                f"the natural one, e.g. star.{s}.rv: {{mu: <v>, sigma: "
                f"<err>}}; ~1 km/s is ample (an rv error dv moves the model "
                f"by ~dv K / c)."
            )

    def _reflex_sky(self, system, t):
        """The observed star's reflex velocity ACROSS the line of sight,
        ``(v_north, v_east)`` in m/s, and its displacement there,
        ``(p_north, p_east)`` in solRad, each orbit's at its own times
        ``t[orbit]`` (``_orbit_emission_times``' dict), from the full orbit
        state -- or all None where the system does not model the
        orbit's sky orientation (no ``bigomega``: no astrometry).  Only
        Keplerian orbits: a Taylor orbit's sky terms are angular and need
        a parallax, and are < 1e-3 of a resolved reflex.  The
        line-of-sight part is the one ``_rv_model`` builds."""
        from exozippy.constants import SOLRAD_PER_DAY_TO_MPS

        orbits = system.orbit
        if "bigomega" not in orbits.manifest:
            return None, None, None, None
        members = orbits.star_membership(self.star_ndx[0])
        if not members:
            return None, None, None, None
        v_n = v_e = p_n = p_e = 0.0
        for o, role in members:
            omap = np.array([o])
            primary = role == "primary"
            X, Y, _, VX, VY, _ = orbits.state_vectors(
                t[int(o)],
                omap,
                relative=not primary,
            )
            frac = (
                orbits.m_companion.value[o] / orbits.m_total.value[o]
                if primary
                else orbits.m_primary.value[o] / orbits.m_total.value[o]
            )
            v_n = v_n + VX[:, 0] * frac
            v_e = v_e + VY[:, 0] * frac
            p_n = p_n + X[:, 0] * frac
            p_e = p_e + Y[:, 0] * frac
        return (
            v_n * SOLRAD_PER_DAY_TO_MPS,
            v_e * SOLRAD_PER_DAY_TO_MPS,
            p_n,
            p_e,
        )

    def _reflex_transverse_speed2(self, system, t, terms):
        """The observed star's squared transverse reflex SPEED, m^2/s^2,
        where the sky orientation is NOT modeled (``_reflex_sky`` returned
        None: no ``bigomega``), each orbit's at its own times ``t[orbit]``
        -- or None without a Keplerian orbit.

        The time dilation ``gamma_*`` needs ``|v_*|^2``, and leaving the
        transverse reflex out of it (as before review 8.5.10) let the
        line-of-sight part alone vary: a spurious ``K^2/(4c) cos(2 phase)``
        on a circular orbit (0.75 m/s at K = 30 km/s, 8 m/s at 100 km/s),
        whose true ``|v|^2`` is constant.  A rotation about the line of sight leaves
        the transverse speed unchanged, so each orbit supplies it from its
        own elements and inclination (``Orbit.transverse_speed2``), scaled
        by the star's share (``M_c/M`` on the primary side, ``M_p/M`` on
        the companion side, as ``_reflex_sky``).  ``terms`` -- the
        per-orbit ``Orbit.orbit_terms`` ``_companion_potential`` already
        built, so the Kepler solve is not repeated.

        Several orbits: their transverse velocities add as vectors whose
        relative orientation is the unknown relative node, so the sum of
        their SQUARES is used -- the expectation over that node (the cross
        terms ``2 v_i,perp . v_j,perp`` average to zero over it), exact
        for one orbit.  A Taylor orbit adds nothing: it has no inclination
        and its transverse motion is unknown without astrometry
        (rvinstrument.md, "Without astrometry", has the sizes of both)."""
        from exozippy.constants import SOLRAD_PER_DAY_TO_MPS

        orbits = system.orbit
        members = orbits.star_membership(int(self.star_ndx[0]))
        if not members:
            return None
        v2 = 0.0
        for o, role in members:
            o = int(o)
            frac = (
                orbits.m_companion.value[o] / orbits.m_total.value[o]
                if role == "primary"
                else orbits.m_primary.value[o] / orbits.m_total.value[o]
            )
            v2 = v2 + pt.sqr(frac) * orbits.transverse_speed2(
                t[o], o, terms=terms[o]
            )
        return v2 * SOLRAD_PER_DAY_TO_MPS**2

    def _companion_potential(self, system, t, terms=None):
        """``sum_o G M_other,o / (r_o c^2)`` at the star, each orbit's at its
        own times ``t[o]`` (``_orbit_emission_times``' dict):
        the gravitational redshift of the star's light in its companions'
        potential, which varies with each orbit's separation ``r_o``
        (``M_other`` the companion group on the primary side, the primary
        group on the companion side).  For a star deep inside a hierarchy
        the separation to an outer group is taken as that orbit's relative
        separation (the inner offset is a fraction a_in/a_out of it).
        A Taylor orbit has no separation and adds nothing.  None without a
        Keplerian orbit.  ``terms`` -- a dict to fill with each orbit's
        ``Orbit.orbit_terms`` (the one Kepler solve the time dilation
        reuses, ``_reflex_transverse_speed2``)."""
        from exozippy.constants import C_LIGHT_RSUN_PER_DAY, G

        orbits = system.orbit
        phi = None
        for o, role in orbits.star_membership(int(self.star_ndx[0])):
            o = int(o)
            kt = orbits.orbit_terms(t[o], o)
            if terms is not None:
                terms[o] = kt
            r = orbits.a.value[o] * kt.r_over_a
            m_other = (
                orbits.m_companion.value[o]
                if role == "primary"
                else orbits.m_primary.value[o]
            )
            term = G * m_other / (r * C_LIGHT_RSUN_PER_DAY**2)
            phi = term if phi is None else phi + term
        return phi

    def _emission_time(
        self, system, t_bary, files, keplerian=None, taylor=None
    ):
        """The star's EMISSION time for light reaching the barycenter at
        ``t_bary``: retarded by the star's own reflex displacement along
        the line of sight, summed over the orbits ``keplerian``/``taylor``
        (default every orbit it is a body of),
        ``t_emit = t_bary - Z_star(t_emit)/c`` with Z away from the
        observer -- the same retardation ``components/ltt.py`` applies to
        transits and RM, through its one implementation
        (``line_of_sight_kinematics`` per Keplerian orbit, the summed
        kinematics through ``solve_delay``).

        ltt.py's z is TOWARD the observer and is the companion-minus-
        primary separation r sin(omega_* + f) sin i, so the star's own
        toward-observer displacement is ``-(M_c/M) z`` on an orbit's
        primary side and ``+(M_p/M) z`` on its companion side -- the
        barycentric shares the RV terms use, with the RV terms' signs.
        A Taylor orbit adds its own line-of-sight displacement (its
        acceleration and jerk terms; the constant and linear ones are the
        gamma and the system's motion, already in t_bary).  Per file,
        ``light_travel_time: false`` turns it off (default on, like every
        other retarded consumer); the orbits it retards then report their
        target-frame timing (Orbit._ltt_reporting_mask).

        ``files`` -- (N,) the file index of each entry of ``t_bary`` (its
        ``light_travel_time:`` decides whether that entry is retarded).
        ``keplerian`` / ``taylor`` -- the orbit indices whose displacement
        to sum (None: every one of the star's).  The raw-redshift mode's
        DIRECTION (the wobble in rho_hat) sums them all; each reflex
        column's TIMING sums only its own subsystem
        (``_orbit_emission_times``).

        Returns ``(t_emit, z)``: ``z`` the star's summed toward-observer
        displacement at ``t_bary`` (solRad), or None without any.
        """
        from .. import ltt

        flags = np.array([bool(self.light_travel_time[i]) for i in files])
        if not flags.any():
            return t_bary, None
        orbits = system.orbit
        s = int(self.star_ndx[0])
        z = vz = az = 0.0
        members = [
            (o, role)
            for o, role in orbits.star_membership(s)
            if keplerian is None or int(o) in keplerian
        ]
        if members and not self._keplerian_ltt_supported(orbits):
            # The same per-orbit decision rm.py and transit.py make
            # (ltt.orbit_supports_ltt): an orbit whose bodies did not
            # resolve has no physical a/masses, so no displacement to
            # retard by.  Warned once; the Keplerian terms are then
            # evaluated at the arrival time.
            members = []
        for o, role in members:
            o = int(o)
            if role == "primary":
                factor = -orbits.m_companion.value[o] / orbits.m_total.value[o]
            else:
                factor = orbits.m_primary.value[o] / orbits.m_total.value[o]
            zo, vzo, azo = ltt.line_of_sight_kinematics(
                t_bary,
                orbits.tp_target.value[o],
                orbits.n.value[o],
                orbits.ecc.value[o],
                orbits.sinw.value[o],
                orbits.cosw.value[o],
                orbits.sini.value[o],
                orbits.a.value[o],
                factor=factor,
                circular=orbits._all_circular(np.array([o])),
            )
            z, vz, az = z + zo, vz + vzo, az + azo
        for o, _role in orbits.taylor_membership(s):
            if taylor is not None and int(o) not in taylor:
                continue
            omap = np.array([int(o)])
            Z = orbits.state_vectors(t_bary, omap, line_of_sight_only=True)
            # state_vectors' Z is AWAY; ltt's z is toward the observer.
            z, vz = z - Z[2][:, 0], vz - Z[5][:, 0]
        if isinstance(z, float):
            return t_bary, None
        delay = ltt.solve_delay(z, vz, pt.zeros_like(z) + az)
        if not flags.all():
            delay = delay * flags.astype(float)
        return t_bary - delay, z

    def _keplerian_ltt_supported(self, orbits):
        """Whether the star's Keplerian orbits carry what the star's
        light-travel time needs (``ltt.orbit_supports_ltt`` plus ``sini``).
        Warns once per component when they do not and some file asked for
        the term (rvinstrument.md, "The star's own light-travel time")."""
        from .. import ltt

        if ltt.orbit_supports_ltt(orbits) and hasattr(orbits, "sini"):
            return True
        if not self._ltt_unsupported_warned:
            self._ltt_unsupported_warned = True
            logger.warning(
                "[%s] light-travel-time correction of the RV reflex "
                "disabled for the Keplerian orbits -- the orbit component "
                "does not define %s (its bodies did not resolve). Set "
                "light_travel_time: false on the RV files to silence this.",
                self.prefix,
                ", ".join(ltt.REQUIRED_ORBIT_PARAMS + ("sini",)),
            )
        return False

    def _block_files(self, blocks):
        """(N,) the file index of every row of a builder call's ``blocks``
        (which tile the time vector, instrument.md); None when no file in
        them has ``light_travel_time`` on."""
        if not any(
            self.light_travel_time[i] for i, rows in blocks if len(rows)
        ):
            return None
        n = sum(len(rows) for _, rows in blocks)
        files = np.full(n, -1, dtype=int)
        for i, rows in blocks:
            files[np.asarray(rows, dtype=int)] = i
        if (files < 0).any():
            # Every builder call lays its rows out as per-file blocks that
            # tile the time vector (instrument.md); a gap is a caller bug.
            raise ValueError(
                f"[{self.prefix}] _rv_model blocks do not tile the time "
                f"vector: rows {np.nonzero(files < 0)[0][:10].tolist()} "
                f"belong to no file."
            )
        return files

    def _orbit_emission_times(self, system, t, files):
        """``{orbit index: emission times}`` for every orbit of the star
        (Keplerian and Taylor), or None when ``files`` is None (no row
        retarded).

        Each orbit's reflex is retarded by the star's displacement in THAT
        orbit and in the orbits NESTED inside it (an orbit whose bodies are
        all bodies of this one: a Jacobi inner planet) -- the subsystem
        whose light travel a transit of that orbit sees.  NOT by the
        star's motion in an orbit outside it: that is the light-time effect
        of an outer companion on the inner orbit (2.5 s for a Jupiter at
        5 AU, HOURS for a stellar companion), which no other consumer
        models -- transits evaluate the inner orbit against its own
        barycenter (ltt.py, "MULTIPLE COMPANIONS", item 2) and every orbit's
        period is the observed one -- so retarding the RVs alone by it puts
        the RV and transit ephemerides of the inner orbit apart (on KELT-4's
        hierarchical triple, 8.3 h).  Sibling orbits (both around the star
        alone) are not nested in each other.  A Taylor orbit (an
        unresolved outer companion) is retarded by its own displacement
        only, and is nested in no Keplerian column.
        """
        if files is None:
            return None
        orbits = system.orbit
        s = int(self.star_ndx[0])
        kep = [int(o) for o, _ in orbits.star_membership(s)]
        out, cache = {}, {}
        for o in kep:
            own = set(orbits.bodies(o))
            subsystem = frozenset(
                [o]
                + [q for q in kep if q != o and set(orbits.bodies(q)) <= own]
            )
            if subsystem not in cache:
                cache[subsystem] = self._emission_time(
                    system, t, files, keplerian=subsystem, taylor=()
                )[0]
            out[o] = cache[subsystem]
        for o, _ in orbits.taylor_membership(s):
            out[int(o)] = self._emission_time(
                system, t, files, keplerian=(), taylor=(int(o),)
            )[0]
        return out

    def _raw_rv_model(self, system, time, blocks):
        """The likelihood's model when some file is raw: ``(model,
        matrix, kep, offset)``, internal units (rvinstrument.md).

        Corrected rows are ``gamma + reflex`` exactly as without raw files.
        Raw rows are ``c z_meas`` predicted by ``redshift.doppler_ratio``
        from the FITTED star (direction, Roemer delay to ``t_B``, space
        velocity) and the reflex of every orbit -- the same ``_rv_model``
        columns -- evaluated at each sub-exposure node's own retarded
        emission time, and averaged over the exposure with the nodes'
        photon weights (``_raw_nodes``; rvinstrument.md, "The exposure
        average"): the measured redshift is the photon-weighted mean of
        the instantaneous one, and the correction is not linear in time
        (Tronsgaard et al. 2019).  Each raw file's ``gamma`` is its
        instrument-frame systemic velocity, ``c[(1 + z0_i)(1 + rv/c) - 1]``.

        ``matrix`` and ``kep`` are the corrected-mode columns on the data's
        own time vector (``time``: for a raw row, the START star's BJD_TDB
        of its tag), so the plots draw a raw point as ``gamma + reflex(tag)
        + its likelihood residual``, on the plotted curve within the noise.
        """
        from exozippy.constants import RSUN_TO_AU, SOLRAD_PER_DAY_TO_MPS

        from .redshift import displace_star, doppler_ratio, star_geometry

        rows = self._raw_rows
        nodes = self._raw_nodes
        star = system.star
        s = int(self.star_ndx[0])
        rv = star.rv.value[s]
        geom = star_geometry(
            nodes["t_tdb"],
            nodes["obs"]["r_au"],
            star.ra.value[s],
            star.dec.value[s],
            star.pm_ra.value[s],
            star.pm_dec.value[s],
            star.parallax.value[s],
            rv,
            self.reference_epoch,
            xp=pt,
        )
        self._raw_t_bary_node = geom["t_bary"]
        # The DIRECTION carries the star's whole displacement (every orbit);
        # each orbit's reflex its own subsystem's retardation.
        _, z_toward = self._emission_time(
            system, geom["t_bary"], nodes["file"]
        )
        files = (
            nodes["file"]
            if any(self.light_travel_time[i] for i in set(nodes["file"]))
            else None
        )
        t_emit = self._orbit_emission_times(system, geom["t_bary"], files)
        v_los, _ = self._rv_model(
            system, geom["t_bary"], nodes["blocks"], offset=None, t_emit=t_emit
        )
        if t_emit is None:
            orbits = system.orbit
            t_emit = {
                int(o): geom["t_bary"]
                for o, _ in orbits.star_membership(s)
                + orbits.taylor_membership(s)
            }
        v_los = v_los * SOLRAD_PER_DAY_TO_MPS
        kep, matrix = self._rv_model(system, time, blocks, offset=None)
        offset = self.gamma.value[self.inst_map_tensor]
        v_n, v_e, p_n, p_e = self._reflex_sky(system, t_emit)
        # The star's own orbital displacement moves the direction rho_hat
        # (the astrometric wobble): along the line of sight always, across
        # it where the sky orientation is modeled.
        r0, e_hat, n_hat = geom["frame"]
        delta = []
        for k in range(3):
            d = 0.0
            if z_toward is not None:
                d = d - z_toward * r0[k]
            if p_n is not None:
                d = d + p_n * n_hat[k] + p_e * e_hat[k]
            delta.append(d * RSUN_TO_AU)
        if any(not isinstance(d, float) for d in delta):
            geom = displace_star(
                geom, nodes["obs"]["r_au"], tuple(delta), xp=pt
            )
        terms = {}
        phi = self._companion_potential(system, t_emit, terms=terms)
        # Without the sky orientation the time dilation still gets the
        # star's whole reflex SPEED (review 8.5.10,
        # _reflex_transverse_speed2), from the same Kepler solves.
        v_perp2 = (
            self._reflex_transverse_speed2(system, t_emit, terms)
            if v_n is None
            else None
        )
        ratio = doppler_ratio(
            geom,
            nodes["obs"],
            (v_los, v_n, v_e),
            xp=pt,
            phi_star=phi,
            v_perp2=v_perp2,
        )
        if nodes["averaged"]:
            # The photon-weighted mean over each exposure's nodes: a
            # segment sum by a CONSTANT index (duplicates accumulate).
            ratio = pt.inc_subtensor(
                pt.zeros(rows.size)[nodes["row"]], nodes["w"] * ratio
            )
        gamma_ms = (
            self.gamma.value[self.inst_map[rows]] * SOLRAD_PER_DAY_TO_MPS
        )
        one_plus_z0 = (1.0 + gamma_ms / C_MPS) / (1.0 + rv / C_MPS)
        cz = C_MPS * (one_plus_z0 * ratio - 1.0) / SOLRAD_PER_DAY_TO_MPS
        base = offset + kep
        model = pt.set_subtensor(base[rows], cz)
        return model, matrix, kep, offset

    def epochs_constraining(self, system, orbit):
        """Each RV file's epochs on every orbit its star is a body of, with
        the information each carries on the conjunction time.

        The same membership ``orbit.amplitude_constrained_orbits`` reads:
        the observed star's Doppler signal is the sum over every orbit
        containing it (``Orbit.star_membership``), primary or companion side.
        Per star, the RVs' own optimal epoch and their information on the
        conjunction there come from the full linearized Fisher matrix of a
        Keplerian at the seeded period, e, omega and conjunction
        (``timing.rv_epoch_information``: tc, P, K, the eccentricity where
        it is free, and the file offsets, marginalized to the ``(tc, P)``
        block), returned as ONE epoch per orbit carrying all of it -- which
        is all an information-weighted mean needs.
        The amplitude and the noise come from the data, through the seeds
        (``timing.rv_amplitudes_and_jitter``): per star, a linear fit of one
        offset per file plus one seeded Keplerian shape per orbit gives each
        orbit's ``K``, and each file's residual scatter beyond its quoted
        errors is the jitter the fit will find (measured on
        `examples/kelt4`'s RVs: HIRES 21 m/s and TRES 48 m/s here, against
        posterior medians of 25 and 59 m/s, where the quoted errors are 3.6
        and 22 m/s).  These Keplerian epochs time the orbital PHASE
        (``Orbit.TIMING_KINDS``; ``Orbit._sampling_epochs`` explains why
        that differs from timing the conjunction on an eccentric orbit).

        A file with ``rm:`` also times the transit by its Rossiter-McLaughlin
        anomaly, an eclipse of amplitude ~ ``p^2 vsini``: it adds that
        orbit's transit information (``timing.transit_information``) at
        that depth in velocity, with ``vsini`` from the seeded
        ``svcoslam``/``svsinlam``.
        """
        from ..orbit import timing

        seed = orbit.timing_seed
        circular = orbit._seeded_circular()
        out = {}
        for s in sorted(set(int(x) for x in self.star_ndx)):
            files = [i for i, x in enumerate(self.star_ndx) if int(x) == s]
            orbits = [int(o) for o, _role in orbit.star_membership(s)]
            if not orbits:
                continue
            ts = [self.dataset_times(i) for i in files]
            vs = [
                np.asarray(self.rv_seed[self.rows(i)], dtype=float)
                for i in files
            ]
            es = [self.dataset_errors(i) for i in files]
            shapes = [
                [
                    timing.rv_shape(
                        t,
                        seed.tc[o],
                        seed.period[o],
                        seed.ecc[o],
                        seed.omega[o],
                    )
                    for o in orbits
                ]
                for t in ts
            ]
            K, jitter = timing.rv_amplitudes_and_jitter(vs, es, shapes)
            sigmas = [
                np.sqrt(es[k] ** 2 + jitter[k] ** 2) for k in range(len(files))
            ]
            fisher = timing.rv_epoch_information(
                ts,
                sigmas,
                [
                    (
                        seed.tc[o],
                        seed.period[o],
                        seed.ecc[o],
                        seed.omega[o],
                        K[m],
                    )
                    for m, o in enumerate(orbits)
                ],
                [not circular[o] for o in orbits],
            )
            for m, o in enumerate(orbits):
                E, info = fisher[m]
                out.setdefault(o, []).append(
                    (
                        np.array([seed.tc[o] + E * seed.period[o]]),
                        np.array([info]),
                        "phase",
                    )
                )
        for i in range(self.n_elements):
            if self.rm_orbit[i]:
                t = self.dataset_times(i)
                o, w = self._rm_information(
                    system, orbit, i, t, self.dataset_errors(i)
                )
                out.setdefault(int(o), []).append((t, w, "conjunction"))
        return out

    def _rm_information(self, system, orbit, i, t, err):
        """``(orbit, per-point information)`` of file ``i``'s RM anomaly
        (see ``epochs_constraining``)."""
        from ..orbit import timing
        from ..rm import resolve_rm_indices

        o, j, _ = resolve_rm_indices(system, self.rm_orbit[i], self.rm_band[i])
        seed = orbit.timing_seed
        sv = [
            orbit._resolve_initval(name, (orbit.n_elements,))[o]
            for name in ("svcoslam", "svsinlam")
        ]
        vsini = (sv[0] ** 2 + sv[1] ** 2) * (u.m / u.s).to(u.solRad / u.d)
        _, depth, T, tau = orbit.seeded_transit_shape(system, int(j))
        w = timing.transit_information(
            t, err, seed.tc[o], seed.period[o], depth * vsini, T, tau
        )
        return o, w

    def _seed_from_lombscargle(self, system):
        """Seed orbital period and conjunction epoch from a Lomb-Scargle peak.

        Runs only when the relaxation engine cannot already DERIVE the period
        and conjunction time, and seeds only the quantities that were
        missing.  A transit search on the same orbit outranks this one
        (``globalsearch.QUALITY_TRANSIT``), so on a system with both, the
        photometric period and epoch stand and the RVs contribute the
        semi-amplitude.

        The semi-amplitude is not pushed as a hint: it REPLACES
        ``self.k_init``, which ``Planet._push_rv_mass_hints`` turns into the
        ``planet.K`` hint at stage 3.  One channel, one number --
        and the sinusoid fit is the better estimator of the two, since
        ``sqrt(2) * std`` counts the noise variance as signal.
        """
        from .. import globalsearch

        mode = globalsearch.search_mode(system)
        if mode == "off":
            return
        orbit_ndx = globalsearch.sole_orbit_index(system, self.prefix)
        if orbit_ndx is None:
            return

        cm = self.config_manager
        groups = {
            "period": (
                f"orbit.{orbit_ndx}.period",
                f"orbit.{orbit_ndx}.logP",
            ),
            "tc": (f"orbit.{orbit_ndx}.tc",),
        }
        satisfied = globalsearch.starts_satisfied(cm, groups)
        if mode != "force" and all(satisfied.values()):
            logger.debug(
                "[%s] Lomb-Scargle not needed: the orbital period and "
                "conjunction time are already derivable.",
                self.prefix,
            )
            return

        logger.info(
            "[%s] no start value for %s -- running a Lomb-Scargle search "
            "over %d velocities.",
            self.prefix,
            ", ".join(k for k, v in satisfied.items() if not v) or "(forced)",
            self.time.size,
        )

        # Work in m/s with each instrument's own offset removed: a
        # periodogram of the raw concatenation measures the offsets, not the
        # planet.
        to_ms = (u.solRad / u.d).to(u.m / u.s)
        gammas = np.asarray(self.gamma_init, dtype=float)
        residual = self.rv_seed * to_ms - gammas[self.inst_map]
        signal = globalsearch.lombscargle_search(
            self.time,
            residual,
            self.err * to_ms,
            inst_map=self.inst_map,
            context=self.prefix,
        )
        self.ls_signal = signal
        if signal is None:
            return

        q = globalsearch.QUALITY_RV
        source = f"Lomb-Scargle on {self.n_elements} RV data set(s)"
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
        if np.isfinite(signal.amplitude) and signal.amplitude > 0.0:
            logger.info(
                "[%s] RV semi-amplitude estimate moved from %.4g to %.4g m/s (the "
                "Lomb-Scargle sinusoid's semi-amplitude replaces "
                "sqrt(2) x scatter).",
                self.prefix,
                self.k_init,
                signal.amplitude,
            )
            self.k_init = float(signal.amplitude)

        if not any(applied):
            # Every orbital seed was declined -- a transit search got there
            # first (QUALITY_TRANSIT).  Saying the period came from the RVs
            # would be false, and the transit component has already said
            # where it really came from.
            return

        get_collector(system).add(
            "Initial values for the orbital period and time of conjunction "
            "were measured from the radial velocities with a Lomb-Scargle "
            r"periodogram \citep{Lomb:1976,Scargle:1982}, as implemented in "
            r"\texttt{astropy} \citep{VanderPlas:2018,Astropy:2022}. "
            "Starting values do not enter the likelihood and cannot move the "
            "posterior.",
            section="data",
            key=f"{self.prefix}.global_search",
            rank=70,
        )

    def _estimate_k_init(self):
        """Seed for the planetary RV semi-amplitude, in m/s.

        ``sqrt(2) * std`` is the semi-amplitude of a sinusoid, but only when
        the scatter it is measured on is the SIGNAL's.  Measured on the raw
        concatenation of every instrument it is dominated instead by the
        constant offsets BETWEEN instruments: one absolute-RV instrument
        sitting at a ~30 km/s systemic velocity next to a relative one seeds
        ``planet.K`` at ~20 km/s for an m/s-level planet.  So each file's own
        ``gamma_init`` (its mean, already computed above) is removed first.

        With a single instrument this is identical to the old expression --
        subtracting a constant does not change a standard deviation.

        Degenerate inputs (one point per file, a file whose RVs are all
        identical, zero-variance data) leave no scatter to measure at all and
        would seed K = 0, which the relaxation engine happily turns into a
        ~1e-20 Mjup planet mass.  There the median error bar -- the white
        noise level, i.e. the amplitude at the detection limit -- is the
        honest answer, and 1 m/s the last resort if even that vanishes.
        """
        to_ms = (u.solRad / u.d).to(u.m / u.s)
        gammas = np.asarray(self.gamma_init, dtype=float)
        residual = self.rv_seed * to_ms - gammas[self.inst_map]
        k = np.sqrt(2.0) * np.std(residual)
        if np.isfinite(k) and k > 0.0:
            return float(k)

        median_err = float(np.median(self.err)) * to_ms
        if np.isfinite(median_err) and median_err > 0.0:
            logger.warning(
                "[%s] the RVs carry no scatter about their per-instrument "
                "means; seeding K from the median error bar (%.3g m/s) "
                "instead.",
                self.prefix,
                median_err,
            )
            return median_err

        logger.warning(
            "[%s] the RVs carry neither scatter nor usable error bars; "
            "seeding K at 1 m/s.",
            self.prefix,
        )
        return 1.0

    def register_parameters(self, system):
        """Stage 3: Embed data-driven hints into the PyMC manifest."""
        gamma_arr = np.atleast_1d(self.gamma_init)
        for i in range(self.n_elements):
            val = (
                gamma_arr[i].item()
                if hasattr(gamma_arr[i], "item")
                else float(gamma_arr[i])
            )
            # gamma_init is the file mean in m/s (load_data): say so, or a
            # user `unit: km/s` on gamma reads it 1000x off (review 1.5.10).
            self.config_manager.add_hint(
                f"{self.prefix}.{i}.gamma", val, unit="m/s"
            )

        # gamma is FREE: it is the sampled per-instrument RV offset, and
        # rvinstrument/defaults.yaml has carried no expressions: block for it
        # since cc26d77.  It said "default" until 2026-08 -- harmless only
        # because there was no block to find, and only until someone added
        # one, at which point every RV fit would have quietly derived its
        # offset instead of sampling it.  manifest.expression_config now
        # raises on that mismatch, so this has to say what it means.
        self.manifest = {"gamma": None}
        if any(self.raw):
            # Settled here, not at stage 1: the astrometry's epoch exists
            # once every component has loaded.
            self.reference_epoch = self._reference_epoch(system)
            self._check_raw_rv_prior(system)
        self._register_noise(self.manifest, self.jittervar_lower)
        self._register_gp(self.manifest)
        self._register_robust(self.manifest)

        if self.total_detrend_cols > 0:
            self.manifest["detrend_coeffs"] = self._detrend_manifest_entry()

    def _orbit_rv_terms(self, system, star_idx):
        """
        Per-orbit RV semi-amplitudes for one star: (K_vec, orbit_map) over
        the orbits that star is a body of.  A primary-group member carries
        the primary reflex K directly; a companion-group member moves with
        the opposite phase (omega_* + 180 deg) and an amplitude scaled by
        the group mass ratio, expressed here as a negated, rescaled K with
        the same omega_* phase formula.
        """
        orbits = system.orbit
        members = orbits.star_membership(star_idx)
        if not members and not orbits.taylor_membership(star_idx):
            raise ValueError(
                f"[{self.prefix}] star {star_idx} is not a body of any "
                f"orbit; no RV model can be built. Add it to an orbit's "
                f"primary/companion group."
            )
        if not members:
            # Only Taylor orbits move this star: no Keplerian term at all.
            return None, np.zeros(0, dtype=int)
        if not hasattr(orbits, "K"):
            raise ValueError(
                f"[{self.prefix}] the orbit component has no K parameter "
                f"(its body groups did not resolve against the active "
                f"system); RVs require orbits with resolvable bodies."
            )
        k_nodes, omap = [], []
        for o, role in members:
            if role == "primary":
                k_nodes.append(orbits.K.value[o])
            else:
                k_nodes.append(
                    -orbits.K.value[o]
                    * orbits.m_primary.value[o]
                    / orbits.m_companion.value[o]
                )
            omap.append(o)
        return pt.stack(k_nodes), np.asarray(omap, dtype=int)

    def _rv_model(self, system, t, blocks, offset=None, t_emit=None):
        """The RV model on times ``t`` -- THE expression, built once.

        ``build_likelihood`` calls this with the data's ``pm.Data`` time
        vector, the files' own row blocks and the per-row ``gamma`` as
        ``offset``; ``compile_plotters`` calls it with a symbolic grid laid
        out in per-file blocks and no offset (the plotted curves are
        gamma-free, the data are drawn gamma-subtracted).  So the plotted
        curve is literally the likelihood code run on other times, and the
        Rossiter-McLaughlin anomaly lands only on the rows of a file that
        asked for it -- by INDEX, exactly as in the likelihood -- rather
        than on a whole matrix column for every instrument (review 1.5.5).
        The in-repo template is ``AstrometryInstrument._rel_model``.

        ``t``      -- ``(N,)`` time tensor.
        ``blocks`` -- ``[(inst_idx, rows), ...]``: the concrete row indices
                      of ``t`` that belong to each file (a file may appear
                      more than once; a file with no rows may be omitted).
        ``offset`` -- optional ``(N,)`` tensor added first (the likelihood's
                      ``gamma[inst_map]``), or None.
        ``t_emit`` -- ``{orbit: emission times}`` for ``t``, when the
                      caller has them already (the raw-redshift nodes);
                      None forms them here (``_orbit_emission_times``),
                      per file.

        THE STAR'S OWN LIGHT-TRAVEL TIME.  ``t`` are arrival times
        (BJD_TDB); every orbit column -- Keplerian and Taylor -- is
        evaluated at the star's EMISSION time, retarded by its own
        line-of-sight displacement in that orbit's own subsystem (~K^2/c
        in velocity; ``_orbit_emission_times``), on the rows of every file with
        ``light_travel_time`` on (the default).  The Rossiter-McLaughlin
        anomaly is NOT: it is an occultation seam with its own retardation
        (``rm.compute_rm_rv``), which reads the arrival times.  With the
        key off on every file the graph is the arrival-time one, node for
        node.

        Returns ``(rv_model, rv_matrix)``, both in the internal RV unit:

        ``rv_model`` ``(N,)`` -- ``offset`` + the sum over every orbit
            containing the observed star of ``K * phase`` + each RM file's
            anomaly on that file's own rows.  WITHOUT the detrend term,
            which is per observation and is added by ``build_likelihood``
            alone.
        ``rv_matrix`` ``(N, n_member_orbits)`` -- the per-orbit columns of
            the same model (``self._plot_orbit_map`` names the orbit of
            each column), an RM file's anomaly sitting in ITS orbit's
            column on ITS rows only, so ``sum(rv_matrix, axis=1)`` is
            ``rv_model - offset`` and the phased panels' "other orbits"
            cleaning subtracts the RM from a non-RM instrument's points
            never, and from the RM instrument's points exactly when the
            panel is for another orbit.

        The likelihood's numerics are pinned bit-identical on every shipped
        example, so the op sequence in here -- offset first, then the
        Keplerian sum, then the RM rows in file order -- is the op sequence
        the likelihood always had.
        """
        orbits = system.orbit
        K_vec, omap = self._orbit_rv_terms(system, self.star_ndx[0])
        if t_emit is None:
            t_emit = self._orbit_emission_times(
                system, t, self._block_files(blocks)
            )
        # Taylor orbits (type: constant_acceleration | constant_jerk) the observed star is the
        # PRIMARY of -- Orbit._taylor_consumers refused the companion side
        # at stage 3.  Each is one more column, after the Keplerian ones.
        taylor = [o for o, _ in orbits.taylor_membership(self.star_ndx[0])]
        self._plot_orbit_map = np.concatenate(
            [omap, np.asarray(taylor, dtype=int)]
        ).astype(int)

        # sum the contribution from every orbit containing the observed star
        columns = []
        if omap.size:
            columns.append(
                orbits.get_radial_velocity(
                    t
                    if t_emit is None
                    else pt.stack([t_emit[int(o)] for o in omap], axis=1),
                    K_vec,
                    omap,
                )  # (N, n_member)
            )
        if taylor:
            columns.append(
                pt.stack(
                    [
                        orbits.taylor_radial_velocity(
                            t if t_emit is None else t_emit[int(o)], o
                        )
                        for o in taylor
                    ],
                    axis=1,
                )
            )
        # One block (the common case, and the only one before Taylor orbits)
        # is used as it is, so a Keplerian-only graph is unchanged.
        kep = (
            columns[0]
            if len(columns) == 1
            else pt.concatenate(columns, axis=1)
        )
        rv_model = pt.sum(kep, axis=1)
        if offset is not None:
            rv_model = offset + rv_model
        rv_matrix = kep

        # Rossiter-McLaughlin in-transit distortion. No-op unless a file set
        # `rm: <orbit_name>` -> the RV model above is unchanged byte for
        # byte (mirrors the GP opt-in). compute_rm_rv returns m/s; convert to
        # the internal RV unit (solRad/d) and add only to that file's rows.
        #
        # INDEX, do not pt.switch. A switch over the branch VALUES would
        # evaluate the Hirano kernel at every instrument's timestamps and then
        # throw away the rows it does not apply to -- the JAX where-trap
        # (the CLAUDE.md invariant): a `where` whose unselected branch can be
        # invalid poisons the gradient of the selected one too. Slicing the
        # RM instrument's own rows makes the unselected rows unreachable by
        # construction instead of merely masked, and is cheaper by exactly
        # the fraction of the data that is not the RM file (the H2011 kernel
        # is a 201 x 64 quadrature PER ROW; on a 40-of-73-row example it was
        # 83% wasted work).
        if any(self.rm_orbit):
            from ..rm import compute_rm_rv, resolve_rm_indices

            rv_ms_per_internal = float((u.solRad / u.d).to(u.m / u.s))
            omap_list = list(omap)
            for inst_idx, rows in blocks:
                oname = self.rm_orbit[inst_idx]
                if not oname:
                    continue
                # A concrete index array, not a slice: `t` may be a pm.Data
                # whose length pytensor treats as symbolic, so a slice's
                # shape is symbolic too and the JAX backend cannot trace the
                # RM subtensor ("Shapes must be 1D sequences of concrete
                # values of integer type").  An advanced index of constants
                # has a statically known length and traces fine.
                rows = np.asarray(rows, dtype=int)
                if rows.size == 0:
                    continue
                oidx, pidx, bidx = resolve_rm_indices(
                    system, oname, self.rm_band[inst_idx]
                )
                if oidx not in omap_list:
                    # The anomaly is the transited star's own line
                    # distortion; an RM orbit this instrument's star is not
                    # a body of has no column to live in and no physics to
                    # justify.  The old plot path dropped it silently.
                    raise ValueError(
                        f"[{self.prefix}.{self.names[inst_idx]}] rm: "
                        f"{oname!r} names an orbit that does not contain "
                        f"the observed star (star_ndx={self.star_ndx[inst_idx]}"
                        f"); the RM anomaly can only be modeled on an orbit "
                        f"the star is a body of."
                    )
                rm_ms = compute_rm_rv(
                    system,
                    t[rows],
                    oidx,
                    pidx,
                    bidx,
                    model=self.rm_model[inst_idx],
                    light_travel_time_active=self.light_travel_time[inst_idx],
                )  # (len(rows),) m/s
                rm_internal = rm_ms / rv_ms_per_internal
                rv_model = pt.inc_subtensor(rv_model[rows], rm_internal)
                col = omap_list.index(oidx)
                rv_matrix = pt.inc_subtensor(rv_matrix[rows, col], rm_internal)

        return rv_model, rv_matrix

    def _data_blocks(self):
        """The data's own row blocks, ``[(i, rows_i), ...]`` in file order
        (``Instrument.rows``, materialized -- see ``_rv_model``)."""
        blocks = []
        for i in range(self.n_elements):
            sl = self.rows(i)
            blocks.append((i, np.arange(sl.start, sl.stop)))
        return blocks

    def build_likelihood(self, model, system):
        time = pm.Data("rv_time", self.time)
        rv = pm.Data("rv_data", self.rv)
        err = pm.Data("rv_err", self.err)

        if len(set(self.star_ndx)) > 1:
            raise NotImplementedError(
                f"[{self.prefix}] all RV instruments must observe the same "
                f"star for now (got star_ndx={self.star_ndx})."
            )

        # 1. The one RV expression (see _rv_model): the gamma constant
        # offset first, then the Keplerian sum, then each RM file's rows.
        # Retained as plain attributes (not Deterministics -- (N_obs,) per
        # draw per chain); compile_plotters compiles the matrix as the
        # plotted model AT the observations.
        if any(self.raw):
            # Some file is raw: its rows are the in-graph barycentric
            # correction of the fitted star (_raw_rv_model).  The gap
            # between that and the corrected-mode model (gamma + reflex)
            # is kept for the plots, which draw the data as reflex +
            # likelihood residual (_plotted_rv).
            rv_model, rv_matrix, kep, offset = self._raw_rv_model(
                system, time, self._data_blocks()
            )
            self._rv_bary_data_node = rv_model - (offset + kep)
        else:
            rv_model, rv_matrix = self._rv_model(
                system,
                time,
                self._data_blocks(),
                offset=self.gamma.value[self.inst_map_tensor],
            )
        self._rv_model_data_node = rv_model
        self._rv_matrix_data_node = rv_matrix

        # Detrending: the shared base mechanism (additive here) -- the one
        # place the likelihood meets the trend, inverted onto the plotted
        # data by detrend_corrected (Instrument._detrended_model).
        rv_model = self._detrended_model(rv_model, "rv_detrend")

        # 2. Define the Likelihood.  Total variance = data_error^2 + jitter^2
        # (shared base helper).  The shared dispatcher writes the plain Normal
        # unless a file asked for a GP, in which case that file's residuals get
        # a celerite2 marginal likelihood with this same mu and sigma.
        sigma = self.total_sigma(err)

        self.add_observation_likelihood(
            f"{self.prefix}.model",
            mu=rv_model,
            sigma=sigma,
            observed=rv,
            system=system,
        )

        # Modeling-draft prose for the RV model itself (the shared
        # data/noise sentences came from the dispatcher above).
        get_collector(system).add(
            "Radial velocities were modeled as a sum of Keplerian orbits "
            "(every orbit containing the observed star), plus a "
            "per-instrument velocity offset, using the Kepler solver in "
            r"exoplanet-core \citep{ForemanMackey:2021}.",
            section="orbits",
            key=f"{self.prefix}.rv_model",
            rank=20,
        )
        taylor_types = sorted(
            {
                system.orbit.types[o]
                for o, _ in system.orbit.taylor_membership(self.star_ndx[0])
            }
        )
        if taylor_types:
            # The orbit too long to resolve, as the Taylor expansion of the
            # star's reflex velocity about the orbit's reference epoch.
            quad = "constant_jerk" in taylor_types
            get_collector(system).add(
                "An additional companion on an orbit too long for the data "
                "to resolve was modeled as a "
                + ("quadratic" if quad else "linear")
                + r" trend in the stellar velocity, $\dot{\gamma}\,(t - "
                r"t_{\rm ref})"
                + (r" + \ddot{\gamma}\,(t - t_{\rm ref})^2/2" if quad else "")
                + r"$, about a reference epoch $t_{\rm ref}$.",
                section="orbits",
                key=f"{self.prefix}.rv_trend",
                rank=21,
            )
        get_collector(system).add_software("exoplanet-core")
        if any(self.raw):
            self._add_raw_prose(system)

    def _add_raw_prose(self, system):
        """The raw-redshift sentence (outputs.md: declared where the model
        is built)."""
        names = ", ".join(
            latex_escape(self.names[i])
            for i in range(self.n_elements)
            if self.raw[i]
        )
        transverse = (
            " and its full three-dimensional reflex velocity"
            if "bigomega" in system.orbit.manifest
            else " and its line-of-sight reflex velocity (with the full "
            "orbital speed in the time dilation)"
        )
        get_collector(system).add(
            "The " + names + " spectra were fit as raw measured redshifts: "
            "the barycentric correction was computed inside the model "
            r"following \citet{Wright:2014}, as a function of the fitted "
            "stellar position, proper motion, parallax and absolute radial "
            "velocity, with the star's space motion" + transverse + " "
            "entering the multiplicative relativistic Doppler relation, so "
            "the cross terms between the barycentric and stellar velocities "
            "and the perspective acceleration are modeled rather than "
            "approximated. Each instrument's zero point absorbs the "
            "template, gravitational-redshift and convective-blueshift "
            "offsets, and the absolute radial velocity is set by its prior.",
            section="orbits",
            key=f"{self.prefix}.raw_redshift",
            rank=22,
        )
        if self._raw_nodes["averaged"]:
            get_collector(system).add(
                "Because the barycentric correction is not linear in time, "
                "each predicted redshift is the photon-weighted mean of the "
                "instantaneous model over its exposure "
                r"\citep{Tronsgaard:2019}, from the exposure-meter record, "
                "the photon-time moments or the exposure time, whichever "
                "each spectrum provides.",
                section="orbits",
                key=f"{self.prefix}.exposure_average",
                rank=23,
            )

    # Points per block on a plotted model grid; one size for the unphased
    # span and the phased period window so both share one compiled layout.
    _PLOT_GRID_N = 2000
    # The unphased model curves (_unphased_grid): sampled only where there
    # are data -- the observations split into seasons at gaps longer than
    # _PLOT_GAP_DAYS, the curve broken (NaN) between them -- at least
    # _PLOT_SAMPLES_PER_ORBIT points per shortest Keplerian period, never
    # sparser than _PLOT_GRID_N spread over the seasons, and at most
    # _PLOT_GRID_MAX points in all (one compiled call per _PLOT_GRID_N).
    _PLOT_GAP_DAYS = 50.0
    _PLOT_SAMPLES_PER_ORBIT = 50
    _PLOT_GRID_MAX = 25 * _PLOT_GRID_N

    def _rm_signature(self, i):
        """What makes file ``i``'s RV model differ from another file's:
        its Rossiter-McLaughlin settings (None without ``rm:``)."""
        if not self.rm_orbit[i]:
            return None
        return (
            self.rm_orbit[i],
            self.rm_band[i],
            self.rm_model[i],
            bool(self.light_travel_time[i]),
        )

    def _plot_layout(self):
        """The per-file blocks the plotted model is compiled for.

        Block 0 is the REFERENCE instrument: the first file without
        ``rm:`` -- the plain Keplerian every non-RM file's likelihood
        scores -- or file 0 when every file has RM.  Its curve is the
        chart's "model" trace.  Then one block per instrument whose model
        differs from the reference's (a different RM signature) and per GP
        instrument: those get their own curves, mirroring the GP treatment
        (a per-instrument "model+GP" curve over that instrument's own
        span).  Returns ``(ref, extra)`` with ``extra`` the sorted list of
        instrument indices after block 0.
        """
        ref = next(
            (i for i in range(self.n_elements) if not self.rm_orbit[i]), 0
        )
        ref_sig = self._rm_signature(ref)
        extra = {
            i
            for i in range(self.n_elements)
            if self._rm_signature(i) != ref_sig
        }
        extra.update(int(i) for i in getattr(self, "_gp_pred_on_grid", {}))
        return ref, sorted(extra)

    def compile_plotters(self, model, system):
        """Compile the plotted model -- the likelihood's expression, twice.

        ``_rv_data_fn(*params)`` -> the per-orbit matrix AT the
            observations: literally the node ``build_likelihood`` built
            (``_rv_model`` on the data blocks), compiled against the plot
            parameters.  The phased panels' "other orbits" cleaning reads
            it, so a non-RM instrument's in-transit points no longer have a
            spurious RM bump subtracted (review 1.5.5).
        ``_rv_grid_fn(t, *params)`` -> ``(rv_full, rv_matrix)`` on a plot
            GRID laid out in ``_plot_layout``'s blocks, ``_PLOT_GRID_N``
            rows each.  Every model curve -- unphased span, phased period
            window, per-instrument RM and GP curves -- is this one function
            fed a different ``t``.
        """
        t_input = pt.vector("t_input")
        param_symbols = [p.value for p in system.plot_params]
        orbits = getattr(system, "orbit", None)

        # Per-file GP conditional-mean evaluators (no-op without a gp: key).
        # First, because the plot layout gives every GP instrument a block.
        self._compile_gp_plotters(system)

        if orbits is not None:
            n_grid = self._PLOT_GRID_N
            ref, extra = self._plot_layout()
            self._rv_layout = [ref] + extra
            blocks = [
                (inst_idx, np.arange(k * n_grid, (k + 1) * n_grid))
                for k, inst_idx in enumerate(self._rv_layout)
            ]
            rv_full_node, rv_matrix_node = self._rv_model(
                system, t_input, blocks
            )

            # Retain the symbolic nodes and their time input so plot_data
            # can (a) derive param_deps by walking the graph and (b) hand
            # the GUI the symbolic tensors behind the model traces.  Not
            # needed by the CLI plot() path.
            self._rv_t_input = t_input
            self._rv_matrix_node = rv_matrix_node
            self._rv_full_node = rv_full_node

            # Save them to SELF, not the system!
            self._rv_grid_fn = pytensor.function(
                inputs=[t_input] + param_symbols,
                outputs=[rv_full_node, rv_matrix_node],
                on_unused_input="ignore",
            )
            if any(self.raw):
                self._rv_bary_fn = pytensor.function(
                    inputs=param_symbols,
                    outputs=self._rv_bary_data_node,
                    on_unused_input="ignore",
                )
            data_node = getattr(self, "_rv_matrix_data_node", None)
            self._rv_data_fn = (
                pytensor.function(
                    inputs=param_symbols,
                    outputs=data_node,
                    on_unused_input="ignore",
                )
                if data_node is not None
                else None
            )

    # ------------------------------------------------------------------
    # Shared data preparation. Both the matplotlib plot() path and the
    # GUI plot_data() path go through these helpers, so the two paths
    # always draw the exact same arrays (see chart.Chart).
    # ------------------------------------------------------------------
    def _rv_factor(self):
        """Internal-units -> user-units (m/s) conversion for RV values.

        Uses the gamma Parameter's factor once the model is built; falls
        back to the raw solRad/d -> m/s conversion so plot_data works in
        data-only mode (point=None), before any Parameter exists.
        """
        gamma = getattr(self, "gamma", None)
        if gamma is not None and hasattr(gamma, "element_factor"):
            return gamma.element_factor(0)
        return (u.solRad / u.d).to(u.m / u.s)

    def _plotted_rv(self, system, point):
        """The observations the plots draw, internal units.

        A corrected file's are its data.  A raw file's c z_meas carries the
        barycentric velocity (tens of km/s), so at a point it is drawn as
        what the corrected-mode panels expect -- gamma + reflex + the
        likelihood's own residual: ``data - (model - (gamma + reflex))``
        (``_rv_bary_data_node``, compiled from the likelihood's graph).
        Without a point (the data-only preview) a raw file is drawn as
        measured."""
        if point is None or not any(self.raw):
            return self.rv
        bary = self._rv_bary_fn(*self._point_to_plot_params(point, system))
        return self.rv - np.asarray(bary, dtype=float)

    def _shortest_period(self, system, point):
        """The shortest period (days) of this star's Keplerian member
        orbits at ``point``, or None when it has only Taylor orbits."""
        periods = [
            float(self._point_value(point, system.orbit.period, int(o)))
            for o in self._plot_orbit_map
            if not system.orbit.is_taylor[int(o)]
        ]
        return min(periods) if periods else None

    def _unphased_grid(self, times, p_min):
        """A 64-bit model-curve grid over the seasons of ``times``.

        Returns ``(t, breaks)``: ``t`` ascending, one ``linspace`` per
        season (``times`` split at gaps over ``_PLOT_GAP_DAYS``), each
        spanning that season's first to last observation and reaching half
        of ``_PLOT_GAP_DAYS`` into each neighboring gap -- so a season of
        one observation still has a curve, and a renderer that pads its
        axes around a season finds the model there -- but never before the
        first or after the last observation, where the curve always ended;
        ``breaks`` the indices in ``t`` where a new season starts, for
        ``_with_gaps``.
        Each season gets its share of ``_PLOT_GRID_N`` by length, raised to
        ``_PLOT_SAMPLES_PER_ORBIT`` points per ``p_min`` (None: no raise),
        and the total is held to ``_PLOT_GRID_MAX``.  One grid over the
        whole span put ~5 points in each orbit of a 3-day planet observed
        for 1100 days, a curve that missed the data its O-C fits.
        """
        t_obs = np.sort(np.asarray(times, dtype=np.float64))
        cuts = np.flatnonzero(np.diff(t_obs) > self._PLOT_GAP_DAYS)
        starts = np.r_[t_obs[0], t_obs[cuts + 1]]
        ends = np.r_[t_obs[cuts], t_obs[-1]]
        # Gaps are longer than _PLOT_GAP_DAYS, so half of it on each side
        # of a gap never makes two seasons overlap.
        starts[1:] -= 0.5 * self._PLOT_GAP_DAYS
        ends[:-1] += 0.5 * self._PLOT_GAP_DAYS
        spans = ends - starts
        total = float(spans.sum())
        share = self._PLOT_GRID_N * (spans / total if total > 0 else 1.0)
        n = np.ceil(share)
        if p_min is not None:
            n = np.maximum(
                n, np.ceil(spans * self._PLOT_SAMPLES_PER_ORBIT / p_min) + 1
            )
        if n.sum() > self._PLOT_GRID_MAX:
            n = np.floor(n * self._PLOT_GRID_MAX / n.sum())
        n = np.maximum(n, 2).astype(int)
        t = np.concatenate(
            [np.linspace(a, b, k) for a, b, k in zip(starts, ends, n)]
        )
        return t, np.cumsum(n)[:-1]

    @staticmethod
    def _with_gaps(t, y, breaks):
        """``(t, y)`` with a NaN point between seasons (``breaks`` from
        ``_unphased_grid``), so a drawn curve does not join them."""
        t_mid = 0.5 * (t[breaks - 1] + t[breaks])
        return np.insert(t, breaks, t_mid), np.insert(y, breaks, np.nan)

    def _eval_rv_grid(self, t_blocks, param_values):
        """Evaluate the plotted model on one grid per layout block.

        ``t_blocks[k]`` is the ``(_PLOT_GRID_N,)`` grid for block ``k`` of
        ``self._rv_layout``.  Returns ``(full, matrix)`` split back per
        block: ``full[k]`` the gamma-free model on that grid, ``matrix[k]``
        its ``(_PLOT_GRID_N, n_member_orbits)`` per-orbit columns.
        Internal units.
        """
        n_grid = self._PLOT_GRID_N
        t_all = np.concatenate(
            [np.asarray(t, dtype=np.float64) for t in t_blocks]
        )
        if t_all.shape[0] != n_grid * len(self._rv_layout):
            raise ValueError(
                f"[{self.prefix}] plot grid layout needs {n_grid} points "
                f"per block, got {[len(t) for t in t_blocks]}."
            )
        full, matrix = self._rv_grid_fn(t_all, *param_values)
        full = np.asarray(full).reshape(len(self._rv_layout), n_grid)
        matrix = np.asarray(matrix).reshape(len(self._rv_layout), n_grid, -1)
        return full, matrix

    def _eval_rv_blocks(self, t_blocks, param_values):
        """``_eval_rv_grid`` for blocks of ANY length: the layout is fed in
        lockstep chunks of ``_PLOT_GRID_N`` (a block shorter than the
        chunk padded with its last time) and each block's results stitched
        back.  Returns ``(full, matrix)``, lists with one array per block,
        ``(len(t_blocks[k]),)`` and ``(len(t_blocks[k]), n_member_orbits)``.
        """
        n_grid = self._PLOT_GRID_N
        blocks = [np.asarray(t, dtype=np.float64) for t in t_blocks]
        n_chunks = max(-(-b.size // n_grid) for b in blocks)
        fulls = [[] for _ in blocks]
        mats = [[] for _ in blocks]
        for c in range(n_chunks):
            chunk_blocks = []
            for b in blocks:
                piece = b[c * n_grid : (c + 1) * n_grid]
                padded = np.full(n_grid, b[-1])
                padded[: piece.size] = piece
                chunk_blocks.append(padded)
            full, matrix = self._eval_rv_grid(chunk_blocks, param_values)
            for k, b in enumerate(blocks):
                m = min(n_grid, max(b.size - c * n_grid, 0))
                fulls[k].append(full[k][:m])
                mats[k].append(matrix[k][:m])
        return (
            [np.concatenate(f) for f in fulls],
            [np.concatenate(m) for m in mats],
        )

    def _rv_at_times(self, param_values, i, t):
        """Instrument ``i``'s plotted (gamma-free) model at arbitrary times.

        The compiled grid layout is fixed at ``_PLOT_GRID_N`` rows per
        block, so ``t`` is fed through it in chunks of that size --
        instrument ``i``'s block carrying the chunk (padded with its last
        time), every other block a dummy -- and the results stitched back.
        An instrument that has no block of its own shares the reference's
        model by construction (see ``_plot_layout``) and reads block 0.
        A convenience for a caller that wants the likelihood's model for
        ONE file at times of its own choosing (tests, a GUI probe); the
        panels feed the layout directly.  Returns ``(full, matrix)`` in
        internal units: ``(len(t),)`` and ``(len(t), n_member_orbits)``.
        """
        t = np.asarray(t, dtype=np.float64).ravel()
        n_grid = self._PLOT_GRID_N
        k = self._rv_layout.index(i) if i in self._rv_layout else 0
        fulls, mats = [], []
        for start in range(0, max(t.size, 1), n_grid):
            chunk = t[start : start + n_grid]
            fill = chunk[-1] if chunk.size else float(self.time[0])
            block = np.full(n_grid, fill)
            block[: chunk.size] = chunk
            t_blocks = [np.full(n_grid, fill) for _ in self._rv_layout]
            t_blocks[k] = block
            full, matrix = self._eval_rv_grid(t_blocks, param_values)
            fulls.append(full[k][: chunk.size])
            mats.append(matrix[k][: chunk.size])
        return np.concatenate(fulls), np.concatenate(mats)

    def _unphased_shared(self, system, point):
        """One evaluation of the plot layout for the unphased panel: block
        0 on the full data span, every other block on its own instrument's
        span.  ``_eval_unphased_model``, ``_eval_unphased_rm_models`` and
        ``_eval_unphased_gp_models`` all read from it."""
        param_values = self._point_to_plot_params(point, system)
        p_min = self._shortest_period(system, point)
        grids = [self._unphased_grid(self.time, p_min)] + [
            self._unphased_grid(self.time[self.rows(i)], p_min)
            for i in self._rv_layout[1:]
        ]
        t_blocks = [t for t, _ in grids]
        full, _ = self._eval_rv_blocks(t_blocks, param_values)
        return {
            "param_values": param_values,
            "t": t_blocks,
            "breaks": [b for _, b in grids],
            "full": full,
        }

    def _eval_unphased_model(self, system, point, shared=None):
        """Summed RV model on the pretty grid, returned in m/s.

        The reference instrument's physical (gamma-free) model over the
        whole data span (see ``_plot_layout``); the per-instrument RM and
        GP curves are added by the two helpers below.
        """
        if shared is None:
            shared = self._unphased_shared(system, point)
        return self._with_gaps(
            shared["t"][0],
            shared["full"][0] * self._rv_factor(),
            shared["breaks"][0],
        )

    def _eval_unphased_rm_models(self, system, point, shared=None):
        """Physical curves for the instruments whose model differs from the
        reference's (their own RM settings), each over that instrument's
        own span.  Returns a list of (instrument index, t, y in m/s); empty
        when every file shares the reference's model -- the single-RM-file
        case then draws exactly what it always did.
        """
        if shared is None:
            shared = self._unphased_shared(system, point)
        ref_sig = self._rm_signature(self._rv_layout[0])
        factor = self._rv_factor()
        out = []
        for k, i in enumerate(self._rv_layout):
            if k == 0 or self._rm_signature(i) == ref_sig:
                continue
            out.append(
                (
                    i,
                    *self._with_gaps(
                        shared["t"][k],
                        shared["full"][k] * factor,
                        shared["breaks"][k],
                    ),
                )
            )
        return out

    def _eval_unphased_gp_models(self, system, point, shared=None):
        """Full (physical + GP) unphased curves, one per GP instrument.

        The GP is a per-instrument noise model, so there is no single "full
        model" curve: each instrument that requested a GP gets its own,
        evaluated only over the span where that instrument actually has data
        (the conditional mean reverts to zero outside it, which would draw a
        misleading flat line across the whole plot).  Its physical part is
        that instrument's OWN block of the layout, so an RM file's GP curve
        carries its RM too.  Returns a list of (instrument index, t, y in
        m/s); empty without any GP.
        """
        if not self.has_gp_plotters():
            return []
        if shared is None:
            shared = self._unphased_shared(system, point)
        factor = self._rv_factor()
        out = []
        for k, i in enumerate(self._rv_layout):
            if k == 0 or i not in self._gp_pred_on_grid:
                continue
            t_i = shared["t"][k]
            y_gp = self.gp_mean_on_grid(system, point, i, t_i)
            out.append(
                (
                    i,
                    *self._with_gaps(
                        t_i,
                        (shared["full"][k] + y_gp) * factor,
                        shared["breaks"][k],
                    ),
                )
            )
        return out

    def _phased_shared(self, system, point):
        """The parts of a phased panel that do NOT depend on which orbit.

        ``_phased_arrays`` is called once per member orbit, and three of the
        arrays it built were the same every time: the marshalled parameter
        values, the RV matrix at the OBSERVED times (the model grid's matrix
        does vary -- its time grid is that orbit's own period window), and
        the per-observation GP + detrend corrections.  Recomputing them per
        orbit meant N_orbits evaluations of a compiled function over the full
        data set per posterior draw, and the spaghetti re-runs the whole
        thing per draw (review 6.5.1).  Hoisted to once per (instrument,
        point) and passed down.

        Kept as a separate method rather than a cache keyed on the point:
        ``point`` is a plain dict, so identity is the only key available and
        it is not a safe one.
        """
        param_values = self._point_to_plot_params(point, system)
        # The fitted detrend model comes off the DATA through the shared
        # base inverse of the likelihood's own combination
        # (Instrument.detrend_corrected); unchanged without detrend columns.
        rv_corrected, err_corrected = self.detrend_corrected(
            self._plotted_rv(system, point), self.err, point
        )
        return {
            "param_values": param_values,
            # The likelihood's own per-orbit matrix at the observations
            # (see compile_plotters): an RM file's anomaly in its orbit's
            # column on its rows only.
            "data_rv_matrix": np.asarray(self._rv_data_fn(*param_values)),
            "rv_corrected": rv_corrected,
            "err_corrected": err_corrected,
            # Phasing data that still contains the correlated (e.g. rotation)
            # signal just smears the panel, so the GP conditional mean comes
            # out of the data along with the other orbits' signal.  Zeros
            # when no file has a GP, so this is a no-op then.
            # Mapped into the corrected-data space by the same base
            # mechanism (identity for this additive space).
            "extra_signals": self.detrend_corrected_signal(
                self.gp_mean_at_data(system, point), point
            ),
        }

    def _phased_arrays(self, system, point, col, o_idx, shared=None):
        """
        Phase grid, isolated model curve, and the per-observation
        background (all other member orbits' signal) for one member
        orbit -- used by plot_data() (and via it plot()).

        ``y_model`` is the reference instrument's curve; ``extra_models``
        lists ``(instrument index, y)`` on the same phase grid for every
        instrument whose model differs on THIS orbit (its RM lives in this
        column, or the reference's does), so the RM anomaly is drawn on its
        own instrument's curve and nowhere else.

        ``shared`` is this (instrument, point)'s ``_phased_shared`` dict;
        omit it and one is built, which is what a standalone caller wants
        and what the per-orbit loop must NOT do.
        """
        if shared is None:
            shared = self._phased_shared(system, point)
        factor = self._rv_factor()
        P_ref = self._point_value(point, system.orbit.period, o_idx)
        tc_ref = self._point_value(point, system.orbit.tc, o_idx)

        t_model = np.linspace(
            tc_ref - 0.5 * P_ref, tc_ref + 0.5 * P_ref, self._PLOT_GRID_N
        ).astype(np.float64)
        phase_model = np.mod((t_model - tc_ref) / P_ref + 0.25, 1.0)
        sort_m = np.argsort(phase_model)

        _, matrix = self._eval_rv_grid(
            [t_model] * len(self._rv_layout), shared["param_values"]
        )
        y_orbit = matrix[0][:, col]

        oname = system.orbit.names[o_idx]
        ref = self._rv_layout[0]
        ref_sig = self._rm_signature(ref)
        extra_models = []
        for k, i in enumerate(self._rv_layout):
            if k == 0 or self._rm_signature(i) == ref_sig:
                continue
            if oname not in (self.rm_orbit[i], self.rm_orbit[ref]):
                continue  # identical to the reference on this orbit
            extra_models.append((i, matrix[k][:, col][sort_m] * factor))

        other_mask = np.ones(len(self._plot_orbit_map), dtype=bool)
        other_mask[col] = False
        other_signals = np.sum(shared["data_rv_matrix"][:, other_mask], axis=1)

        return {
            "P_ref": P_ref,
            "tc_ref": tc_ref,
            "factor": factor,
            "phase_model": phase_model[sort_m],
            "y_model": y_orbit[sort_m] * factor,
            "extra_models": extra_models,
            "other_signals": other_signals + shared["extra_signals"],
            # The detrend-corrected data and errors the cleaning starts from.
            "rv_corrected": shared["rv_corrected"],
            "err_corrected": shared["err_corrected"],
        }

    def plot(self, system, points, filename_prefix="debug"):
        """Render the unphased + per-orbit phased PDFs from plot_data specs.

        The specs are the single description of these plots -- the GUI draws
        the same ones via plotly (see plotrender.py's module docstring).
        """
        from exozippy.plotrender import plot_via_specs

        plot_via_specs(self, system, points, filename_prefix=filename_prefix)

    def plot_data(self, system, point=None):
        """
        GUI charts for the RV instrument: one unphased RV-vs-time
        chart plus one phased chart per member orbit. With point=None only
        the observed data traces are returned (raw preview, no model);
        with a point, model curves are added via the shared prep helpers.
        See Component.plot_data and chart.Chart.
        """
        from exozippy.chart import Chart, Trace

        factor = self._rv_factor()
        specs = []

        # ---- Unphased: RV vs time -------------------------------------
        traces = []
        model_deps = []
        if point is not None:
            # One evaluation of the compiled layout serves the reference
            # curve and every per-instrument curve below.
            shared = self._unphased_shared(system, point)
            t_pretty, y_model = self._eval_unphased_model(
                system, point, shared=shared
            )
            deps = self._model_trace_param_deps(
                getattr(self, "_rv_full_node", None), system
            )
            model_deps = deps
            traces.append(
                Trace(
                    name="model",
                    role="model",
                    kind="line",
                    x=t_pretty,
                    y=y_model,
                    node=getattr(self, "_rv_full_node", None),
                )
            )
            # One physical curve per instrument whose model differs from the
            # reference's -- its own Rossiter-McLaughlin settings -- over
            # that instrument's own span, so the RM anomaly is drawn on the
            # curve of the file whose likelihood scored it and on no other
            # (review 1.5.5).  Empty when every file shares one model.
            for i, t_rm, y_rm in self._eval_unphased_rm_models(
                system, point, shared=shared
            ):
                traces.append(
                    Trace(
                        name=f"{self.names[i]} model",
                        role="model",
                        kind="line",
                        x=t_rm,
                        y=y_rm,
                        node=getattr(self, "_rv_full_node", None),
                        style={"series_index": int(i), "lw": 1.0},
                    )
                )
            # One physical+GP curve per GP instrument (see
            # _eval_unphased_gp_models). No symbolic node: the GP conditional
            # mean is not part of the model graph, so the GUI re-renders
            # these by asking for a fresh point on a slider move -- which its
            # eval path does, PROVIDED the GP hyperparameters are declared in
            # param_deps (gp_dep_labels, below).  A previous version of this
            # comment said the GUI "cannot re-render these on a slider move";
            # it could all along, and the missing deps were the only blocker
            # (review 1.12.9).
            for i, t_gp, y_gp in self._eval_unphased_gp_models(
                system, point, shared=shared
            ):
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
        # The fitted trend is per observation, so it comes off the DATA
        # rather than going onto the model curve, through the base inverse of
        # the likelihood's own combination (Instrument.detrend_corrected);
        # the raw data without detrend columns.
        rv_corrected, err_corrected = self.detrend_corrected(
            self._plotted_rv(system, point), self.err, point
        )
        for i in range(self.n_elements):
            mask = self.inst_map == i
            # gamma offset only when a point supplies it; raw data otherwise
            g = (
                self._point_value(point, self.gamma, i)
                if point is not None
                else 0.0
            )
            traces.append(
                Trace(
                    name=self.names[i],
                    role="data",
                    kind="scatter",
                    x=self.time[mask],
                    y=(rv_corrected[mask] - g) * factor,
                    yerr=err_corrected[mask] * factor,
                    style=self._data_trace_style(i),
                )
            )
        # The data traces are gamma- and detrend-subtracted, so they move with
        # the point too (dynamic_data) and those sliders must reach this
        # component through param_deps -- both are applied in numpy, not
        # through the symbolic model node, so the graph walk alone would miss
        # them.  The GP hyperparameters are the same case (the model+GP
        # curves above and the phased cleaning both use the compiled
        # celerite2 conditional mean), see Instrument.gp_dep_labels.
        gamma_label = getattr(getattr(self, "gamma", None), "label", None)
        numpy_deps = (
            ([gamma_label] if gamma_label else [])
            + self.detrend_dep_labels()
            + self.gp_dep_labels()
        )
        if point is not None and any(self.raw):
            # The raw data move with the star too (_plotted_rv).
            numpy_deps = numpy_deps + self._model_trace_param_deps(
                self._rv_bary_data_node, system
            )
        if point is not None:
            model_deps = model_deps + [
                lbl for lbl in numpy_deps if lbl not in model_deps
            ]
        # Once per (instrument, point), not once per orbit (6.5.1): the
        # phased panels and the O-C of every panel read the same arrays.
        omap = getattr(self, "_plot_orbit_map", None)
        phased_shared = (
            self._phased_shared(system, point)
            if point is not None and omap is not None
            else None
        )
        unphased_meta = {
            "phase_folded": False,
            "summary": self._summary_meta("time_series"),
            "file_tag": "RV_unphased",
            "figsize": (12, 6),
            "dynamic_data": True,
            "caption": (
                "Radial velocities with the best-fit model "
                "(red); posterior draws are overplotted with "
                "low opacity." + self.detrend_caption()
            ),
        }
        if phased_shared is not None:
            unphased_meta["residuals"] = self._residuals_meta(
                point, phased_shared, self.time
            )
        specs.append(
            Chart(
                id=f"{self.prefix}.unphased",
                component={"yaml_key": self.prefix, "instance": None},
                title=f"Unphased RV Model: {getattr(system, 'name', '')}",
                xlabel="Time [BJD]",
                ylabel="Relative RV [m/s]",
                traces=traces,
                param_deps=model_deps,
                meta=unphased_meta,
            )
        )

        # ---- Phased: one chart per member orbit (needs a model) -------
        if phased_shared is not None:
            deps = self._model_trace_param_deps(
                getattr(self, "_rv_matrix_node", None), system
            )
            # The phased DATA moves with the point too: the fold uses tc/P,
            # and the cleaning subtracts gamma + the fitted detrend model +
            # the other orbits' signal (all applied in numpy) -- hence
            # dynamic_data below and the explicit deps the graph walk cannot
            # see.
            deps = deps + [lbl for lbl in numpy_deps if lbl not in deps]
            for col, o_idx in enumerate(omap):
                if system.orbit.is_taylor[o_idx]:
                    # No period to fold on: a Taylor orbit's trend is drawn
                    # in the unphased chart and removed from every phased
                    # one as an "other orbit" (its column is in the matrix).
                    continue
                prep = self._phased_arrays(
                    system, point, col, o_idx, shared=phased_shared
                )
                P_ref, tc_ref = prep["P_ref"], prep["tc_ref"]
                otraces = [
                    Trace(
                        name="model",
                        role="model",
                        kind="line",
                        x=prep["phase_model"],
                        y=prep["y_model"],
                        node=getattr(self, "_rv_matrix_node", None),
                    )
                ]
                # An instrument whose RM sits on this orbit gets its own
                # isolated curve, the anomaly included (see _phased_arrays).
                for i, y_extra in prep["extra_models"]:
                    otraces.append(
                        Trace(
                            name=f"{self.names[i]} model",
                            role="model",
                            kind="line",
                            x=prep["phase_model"],
                            y=y_extra,
                            node=getattr(self, "_rv_matrix_node", None),
                            style={"series_index": int(i), "lw": 1.0},
                        )
                    )
                for i in range(self.n_elements):
                    mask = self.inst_map == i
                    g = self._point_value(point, self.gamma, i)
                    cleaned = (
                        prep["rv_corrected"][mask]
                        - g
                        - prep["other_signals"][mask]
                    ) * factor
                    data_phases = np.mod(
                        (self.time[mask] - tc_ref) / P_ref + 0.25, 1.0
                    )
                    otraces.append(
                        Trace(
                            name=self.names[i],
                            role="data",
                            kind="scatter",
                            x=data_phases,
                            y=cleaned,
                            yerr=prep["err_corrected"][mask] * factor,
                            style=self._data_trace_style(i),
                        )
                    )
                oname = system.orbit.names[o_idx]
                specs.append(
                    Chart(
                        id=f"{self.prefix}.phased.{oname}",
                        component={"yaml_key": self.prefix, "instance": None},
                        title=(
                            f"Phased RV: {oname} "
                            f"({getattr(system, 'name', '')})"
                        ),
                        xlabel=f"Phase (P = {P_ref:.5f} d, Tc at 0.25)",
                        ylabel="Isolated RV [m/s]",
                        traces=otraces,
                        param_deps=deps,
                        meta={
                            "phase_folded": True,
                            "summary": self._summary_meta("phase_fold"),
                            "orbit": oname,
                            "period": P_ref,
                            "tc": tc_ref,
                            "file_tag": f"RV_phased_{oname}",
                            "figsize": (10, 6),
                            "caption": (
                                "Radial velocities phase-folded on "
                                "orbit "
                                + latex_escape(oname)
                                + ", with the other orbits' "
                                "contributions removed."
                                + self.detrend_caption()
                            ),
                            "hline_y": 0.0,
                            "dynamic_data": True,
                            "residuals": self._residuals_meta(
                                point,
                                phased_shared,
                                np.mod(
                                    (self.time - tc_ref) / P_ref + 0.25, 1.0
                                ),
                            ),
                        },
                    )
                )

        return specs

    #: The RV axes of the summary figure, all of them: one label (and,
    #: across the panels, one range), and the BJD zero points its time axis
    #: may subtract (TESS's BTJD, then the older 2450000 convention).
    SUMMARY_YLABEL = "RV [m/s]"
    SUMMARY_X_OFFSETS = (2457000, 2450000)

    def _summary_meta(self, layout):
        """A chart's ``meta["summary"]``: drawn in ``layout``, each
        instrument shown by its ``display_label`` and, by series index, its
        user ``plot: color:`` (None: the figure's palette)."""
        meta = {
            "layout": layout,
            "labels": {
                name: self.display_label(i)
                for i, name in enumerate(self.names)
            },
            "colors": {i: self.plot_color[i] for i in range(self.n_elements)},
            "ylabel": self.SUMMARY_YLABEL,
        }
        if layout == "time_series":
            meta["x_offsets"] = list(self.SUMMARY_X_OFFSETS)
        return meta

    def _residuals_meta(self, point, shared, x):
        """The O-C at the observations, as a chart's ``meta["residuals"]``:
        one residual Trace per instrument, in m/s like the chart's data.

        Data minus the likelihood's own model at the observed times: the
        detrend-corrected RVs less gamma, every member orbit's term (the
        ``_rv_data_fn`` matrix, i.e. the node ``build_likelihood`` scored,
        so an RM file's anomaly is in it on that file's rows only) and any
        GP conditional mean.  The numbers are the same on every panel and
        only ``x`` differs -- the observed time on the unphased chart, the
        orbit's phase on a phased one -- and on a phased panel they are
        exactly its data minus its plotted orbit curve, since both of those
        come out of the same matrix.  The error bars are the quoted ones, as
        on the data traces.

        ``shared`` is this point's ``_phased_shared`` dict; ``x`` is
        per-observation (``(N_obs,)``) and is split by instrument here.
        """
        from exozippy.chart import Trace

        factor = self._rv_factor()
        model = (
            np.sum(shared["data_rv_matrix"], axis=1) + shared["extra_signals"]
        )
        traces = []
        for i in range(self.n_elements):
            mask = self.inst_map == i
            g = self._point_value(point, self.gamma, i)
            traces.append(
                Trace(
                    name=self.names[i],
                    role="residual",
                    kind="scatter",
                    x=x[mask],
                    y=(shared["rv_corrected"][mask] - g - model[mask])
                    * factor,
                    yerr=shared["err_corrected"][mask] * factor,
                    style=self._data_trace_style(i),
                )
            )
        return traces
