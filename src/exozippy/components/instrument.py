"""
Shared base class for data (instrument) components.

rvinstrument, transit, mulensinstrument and astrometryinstrument are four
near-parallel data components that each re-implement the same scaffolding:
per-observation instrument maps, a per-instrument noise term, a jitter floor
derived from the smallest error bar, optional detrending against extra data
columns, and per-instrument plot styling.  ``Instrument`` extracts that
common machinery so it is written and tested once; the physics (RV curve vs
transit light curve vs magnification vs astrometric position) stays in each
child.

This class is deliberately NOT a discoverable component: it declares none of
``Component``'s abstract methods (``prefix``, ``register_parameters``,
``build_likelihood``), so it stays abstract and ``factory.discover_components``
skips it (see the ``inspect.isabstract`` guard there).

Two noise parameterizations are supported through the ``noise_model`` class
attribute so the base does not hardcode one:

  ``"jitter_variance"`` (default; rv/transit/astrometry)
      An additive per-instrument variance ``jitter_variance`` plus a reported
      ``jitter``; total sigma is ``sqrt(err**2 + jitter_variance[inst])``.
  ``"err_scale"`` (mulensinstrument)
      A multiplicative per-instrument error scale ``err_scale``; total sigma
      is ``err * err_scale[inst]``.

Bands are optional: this base never references a ``band:`` block, so children
that need one (transit, mulens, astrometry) resolve it themselves and children
that do not (rv) are unaffected.

Gaussian processes are optional too, and off unless asked for.  A data file
carrying a ``gp:`` key (``rotation``, ``sho``, or both -- see components/gp.py)
gets a celerite2 kernel instead of independent Gaussian errors; every other
file in the same component keeps the plain ``pm.Normal``.  Children opt in by
calling ``_prepare_gp`` at the end of ``load_data``, ``_register_gp`` in
``register_parameters``, and ``add_observation_likelihood`` in place of their
final ``pm.Normal``; a component that does none of these is unaffected, and a
component that does all three costs nothing extra when no file sets ``gp:``.

Robust observation likelihoods are optional in exactly the same shape.  A
data file carrying a ``likelihood:`` key (``hogg`` or ``studentt`` -- see
components/likelihood.py) swaps its plain ``pm.Normal`` for a marginalized
inlier/outlier mixture or a Student-t; every other file is untouched.  The
same three hooks carry it (``_prepare_robust``, ``_register_robust``, and the
shared ``add_observation_likelihood`` dispatcher), the parameters are
full-length pinned vectors like the GP hyperparameters, and a file cannot set
both ``gp:`` and ``likelihood:`` (celerite2's closed-form marginal is
Gaussian-only).  Off by default everywhere: with no ``likelihood:`` key the
model is byte-for-byte what it was before this feature existed.
"""

import logging

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from . import gp as gp_support
from . import likelihood as robust_support
from .component import Component

logger = logging.getLogger(__name__)

# Time-system vocabulary for the per-file time_scale/time_frame keys.
# Scales are astropy.time scale names ("ut" is accepted as an alias for
# ut1); frames name where the clock sits: jd = the observatory/geocenter,
# hjd = heliocenter, bjd = solar-system barycenter.  The astropy
# light_travel_time "kind" implementing each frame's correction is the
# mapped value (None = no light-travel correction).
_TIME_SCALES = ("utc", "tai", "tt", "tdb", "tcb", "tcg", "ut1")
_TIME_SCALE_ALIASES = {"ut": "ut1"}
_TIME_FRAMES = {"jd": None, "hjd": "heliocentric", "bjd": "barycentric"}
# Frame/scale conversion is only meaningful on absolute Julian Dates;
# anything below this is a truncated time (BJD-2450000, MJD, ...) that
# needs time_offset first.
_MIN_ABS_JD = 2_000_000.0


class Instrument(Component):
    # Noise parameterization: "jitter_variance" (additive) or "err_scale"
    # (multiplicative).  Subclasses override; the default matches the majority
    # (rv/transit/astrometry).
    noise_model = "jitter_variance"

    # Whether this child has wired up the GP hooks below.  Children that model
    # more than one observable per data file (astrometry: E/N or sep/PA, which
    # differ in units and so cannot share one amplitude) leave this False and
    # reject the ``gp:`` key outright rather than silently ignoring it.
    supports_gp = True

    # Same switch for the robust `likelihood:` hooks (hogg mixture needs a
    # single out_scale unit per file, so astrometry's two-observable files
    # opt out for the same reason they opt out of GPs).
    supports_robust_likelihood = True

    def __init__(self, component_config, config_manager):
        super().__init__(component_config, config_manager)
        # Every instrument reads its data from per-element files and tracks a
        # running observation count.
        self.files = [c.get("file") for c in self.config]
        self.n_total_obs = 0
        # Optional per-file exclusion mask (see _apply_mask). Parsed here so a
        # malformed spec fails at construction, not mid-load.
        self.mask_specs = [c.get("mask") for c in self.config]
        # Optional per-file time system (time_offset/time_scale/time_frame/
        # time_location/time_ephemeris) and column layout (columns:), both
        # applied by _read_data.  Parsed here so malformed specs fail at
        # construction too.
        self.time_specs = [
            self._parse_time_spec(c, i) for i, c in enumerate(self.config)
        ]
        self.column_specs = [
            self._parse_columns_spec(c, i) for i, c in enumerate(self.config)
        ]
        self._load_plot_styles()
        self._load_gp_config()
        self._load_likelihood_config()

    # ------------------------------------------------------------------
    # Per-instrument plot styling (config, not Parameters -- no sampling)
    # ------------------------------------------------------------------
    def _load_plot_styles(self):
        """Read an optional per-instrument ``plot: {color, marker}`` block.

        Populates ``self.plot_color[i]`` / ``self.plot_marker[i]`` (each None
        when unset).  These are user overrides for the categorical data-series
        style; the theme supplies the default by series index (see
        ``notes/gui_todo.txt`` precedence: user > theme).  A GUI can later
        expose a picker that writes this same key via the ruamel round-trip.
        """
        self.plot_color = []
        self.plot_marker = []
        for c in self.config:
            style = c.get("plot") or {}
            if not isinstance(style, dict):
                style = {}
            self.plot_color.append(style.get("color"))
            self.plot_marker.append(style.get("marker"))

    def _data_trace_style(self, i):
        """Style override dict for instrument ``i``'s data trace.

        Always carries the categorical ``series_index``; adds an explicit
        ``color`` / ``marker`` only when the user configured one (so the theme
        default by index still applies otherwise).  Suitable for
        ``plotspec.Trace.style``.
        """
        style = {"series_index": int(i)}
        if self.plot_color[i] is not None:
            style["color"] = self.plot_color[i]
        if self.plot_marker[i] is not None:
            style["marker"] = self.plot_marker[i]
        return style

    @staticmethod
    def _plot_style_config_schema():
        """The shared ``plot`` config-schema entry (per-instrument styling).

        Children append this to their ``config_schema()`` so introspection and
        a GUI discover the key generically.
        """
        return {
            "key": "plot",
            "kind": "option",
            "accepts": None,
            "required": False,
            "doc": (
                "Optional per-instrument plot styling: a mapping with "
                "'color' and/or 'marker' overriding the theme default for "
                "this data series (e.g. {color: '#1f77b4', marker: 's'})."
            ),
        }

    # ------------------------------------------------------------------
    # Shared data loading
    # ------------------------------------------------------------------
    @staticmethod
    def _sort_by_time(df, time_col=0):
        """Return ``df`` sorted ascending by its time column.

        Every instrument calls this on each data file right after reading it,
        BEFORE splitting out columns or deriving anything from the times, so
        one sort keeps the observable, the errors, the detrend columns and any
        per-epoch quantity computed downstream (observer positions, parallax
        factors) aligned by construction.

        Sorting per file rather than globally is deliberate: the concatenated
        arrays stay contiguous per instrument, which ``_build_block_detrend``
        relies on to place each instrument's columns on the block diagonal,
        and which keeps every row-aligned side array (mulens's observer
        positions) in step with its own file.  A global sort across files
        would interleave instruments and quietly break both.

        Stable sort, so files already in time order are untouched and repeated
        timestamps keep their relative order.
        """
        order = np.argsort(df.iloc[:, time_col].values, kind="stable")
        if np.all(order == np.arange(len(order))):
            return df  # already sorted: no copy
        return df.iloc[order].reset_index(drop=True)

    def _apply_mask(self, df, i):
        """Drop excluded rows of file ``i`` per its optional ``mask:`` config.

        Called on each data file right after reading and BEFORE
        ``_sort_by_time``, so every mask spec refers to the file's own row
        order as it is on disk (comment lines not counted).  Dropping whole
        rows here -- before any column is split out or anything is derived
        from the times -- keeps the observable, errors, detrend columns and
        every row-aligned side array (observer positions, parallax factors)
        consistent by construction, exactly like the sort.

        Accepted ``mask:`` forms on an instrument's config entry:

        - a path to a whitespace/newline-delimited file with ONE value per
          data row (0/1, true/false): nonzero/true means EXCLUDE that row.
          This is the shape a bad-data flag vector (e.g. from MMEXOFAST)
          saves to naturally.
        - a list of booleans, one per data row: True means EXCLUDE.
        - a list of integers: 0-based row indices to EXCLUDE.

        Absent or ``null`` keeps every point (the default: byte-for-byte the
        pre-feature behavior).
        """
        spec = self.mask_specs[i]
        if spec is None:
            return df
        label = f"{self.prefix}[{self.names[i]}]"
        n = len(df)

        if isinstance(spec, str):
            flags = np.loadtxt(spec, dtype=float, comments="#").ravel()
            if flags.size != n:
                raise ValueError(
                    f"[{label}] mask file '{spec}' has {flags.size} entries "
                    f"but the data file has {n} rows -- one flag per data "
                    f"row is required."
                )
            exclude = flags.astype(bool)
        elif isinstance(spec, (list, tuple, np.ndarray)):
            spec = list(spec)
            if all(isinstance(v, (bool, np.bool_)) for v in spec):
                if len(spec) != n:
                    raise ValueError(
                        f"[{label}] boolean mask has {len(spec)} entries but "
                        f"the data file has {n} rows."
                    )
                exclude = np.asarray(spec, dtype=bool)
            elif all(
                isinstance(v, (int, np.integer))
                and not isinstance(v, (bool, np.bool_))
                for v in spec
            ):
                idx = np.asarray(spec, dtype=int)
                if idx.size and (idx.min() < 0 or idx.max() >= n):
                    raise ValueError(
                        f"[{label}] mask indices must be 0-based row indices "
                        f"in [0, {n - 1}]; got min {idx.min()}, max "
                        f"{idx.max()}."
                    )
                exclude = np.zeros(n, dtype=bool)
                exclude[idx] = True
            else:
                raise ValueError(
                    f"[{label}] mask list must be all booleans (per-row "
                    f"flags) or all integers (row indices to exclude)."
                )
        else:
            raise ValueError(
                f"[{label}] mask must be a file path or a list; got "
                f"{type(spec).__name__}."
            )

        n_masked = int(exclude.sum())
        if n_masked == n:
            raise ValueError(
                f"[{label}] mask excludes every one of the {n} data points."
            )
        if n_masked:
            logger.info(f"[{label}] mask excluded {n_masked}/{n} data points.")
            df = df.loc[~exclude].reset_index(drop=True)
        return df

    @staticmethod
    def _mask_config_schema():
        """The shared ``mask`` config-schema entry (per-file point exclusion).

        Children append this to their ``config_schema()`` so introspection and
        a GUI discover the key generically.
        """
        return {
            "key": "mask",
            "kind": "option",
            "accepts": None,
            "required": False,
            "doc": (
                "Optional per-file point exclusion, applied to the data "
                "file's own row order before anything is derived from it: a "
                "path to a file with one 0/1 flag per data row (nonzero = "
                "exclude), a list of booleans (one per row, true = exclude), "
                "or a list of 0-based row indices to exclude."
            ),
        }

    # ------------------------------------------------------------------
    # Shared file reading: columns, mask, time system, sort
    # ------------------------------------------------------------------
    def _read_data(self, i, roles, detrend=False):
        """Read data file ``i`` into canonical column order, ready to use.

        One call replaces the read/mask/sort triplet every child used to
        write, and is the single place the user-facing file conveniences
        happen, in this order:

        1. ``columns:``   reorder/select columns into the canonical layout
                          (``_select_columns``);
        2. ``mask:``      drop excluded rows, in on-disk row order
                          (``_apply_mask``);
        3. ``time_*:``    add ``time_offset`` and convert the time column
                          to BJD_TDB (``_to_bjd_tdb``);
        4. sort ascending by (converted) time (``_sort_by_time``).

        ``roles`` names the canonical columns in order and must start with
        ``"time"``; with ``detrend=True`` any detrend columns follow them.
        The returned DataFrame is indexed positionally: column ``j`` is
        ``roles[j]``, columns ``len(roles):`` are the detrend columns.
        With none of the optional keys set this is byte-for-byte the old
        read (all columns, as-is on disk, sorted by column 0).
        """
        if roles[0] != "time":
            raise ValueError(
                f"[{self.prefix}] _read_data roles must start with 'time'; "
                f"got {list(roles)}."
            )
        df = pd.read_csv(
            self.files[i], sep=r"\s+", engine="c", header=None, comment="#"
        )
        df = self._select_columns(df, i, roles, detrend)
        df = self._apply_mask(df, i)
        t = self._to_bjd_tdb(df.iloc[:, 0].values.astype(float), i)
        df[df.columns[0]] = t
        return self._sort_by_time(df)

    # ------------------------------------------------------------------
    # Optional per-file column layout (columns:)
    # ------------------------------------------------------------------
    def _parse_columns_spec(self, c, i):
        """Validate one config entry's optional ``columns:`` key (stage 0).

        The value is a mapping from role name to a 0-based column index in
        the data file, plus an optional ``detrend`` role mapping to a LIST
        of 0-based column indices.  Role-name validity is checked later, in
        ``_select_columns``, because only the child knows its roles (and
        astrometry's depend on the file's mode); the structure is checked
        here so a malformed spec fails at construction.
        """
        spec = c.get("columns")
        if spec is None:
            return None
        label = f"{self.prefix}[{self.names[i]}]"
        if not isinstance(spec, dict):
            raise ValueError(
                f"[{label}] columns must be a mapping of role name to "
                f"0-based column index (plus optional 'detrend: [i, ...]'); "
                f"got {type(spec).__name__}."
            )

        def _col(v):
            if (
                isinstance(v, bool)
                or not isinstance(v, (int, np.integer))
                or v < 0
            ):
                raise ValueError(
                    f"[{label}] columns entries must be 0-based column "
                    f"indices (non-negative integers); got {v!r}."
                )
            return int(v)

        out = {}
        for k, v in spec.items():
            if k == "detrend":
                if not isinstance(v, (list, tuple)):
                    raise ValueError(
                        f"[{label}] columns.detrend must be a list of "
                        f"0-based column indices; got {v!r}."
                    )
                out[k] = [_col(x) for x in v]
            else:
                out[k] = _col(v)
        return out

    def _select_columns(self, df, i, roles, detrend):
        """Apply file ``i``'s ``columns:`` spec, returning the canonical layout.

        Without a spec the file is returned untouched (the on-disk order IS
        the canonical order, detrend columns are everything past the named
        roles).  With a spec, each role defaults to its canonical position
        and named roles override it; detrend columns are ONLY the ones the
        spec lists (an explicit layout leaves no 'rest of the columns' to
        guess at).
        """
        spec = self.column_specs[i]
        label = f"{self.prefix}[{self.names[i]}]"
        if spec is None:
            if df.shape[1] < len(roles):
                raise ValueError(
                    f"[{label}] data file has {df.shape[1]} columns but "
                    f"needs at least {len(roles)} ({', '.join(roles)})."
                )
            return df

        unknown = sorted(set(spec) - set(roles) - {"detrend"})
        if unknown:
            raise ValueError(
                f"[{label}] columns names unknown role(s) {unknown}; valid "
                f"roles are {list(roles)}"
                + (" plus 'detrend'." if detrend else ".")
            )
        if "detrend" in spec and not detrend:
            raise ValueError(
                f"[{label}] columns.detrend is not supported by this "
                f"component (it has no detrending)."
            )
        idx = [spec.get(role, j) for j, role in enumerate(roles)]
        det = spec.get("detrend", []) if detrend else []
        too_big = [c for c in idx + det if c >= df.shape[1]]
        if too_big:
            raise ValueError(
                f"[{label}] columns indices are 0-based and the data file "
                f"has {df.shape[1]} columns; got {sorted(set(too_big))}."
            )
        out = df.iloc[:, idx + det].copy()
        out.columns = range(out.shape[1])
        return out

    @staticmethod
    def _columns_config_schema(roles, detrend=True, note=""):
        """The shared ``columns`` config-schema entry for the given roles."""
        extra = (
            ", plus 'detrend' mapping to a list of column indices to "
            "detrend against (when columns: is given, detrend columns "
            "must be listed explicitly)"
            if detrend
            else ""
        )
        return {
            "key": "columns",
            "kind": "option",
            "accepts": None,
            "required": False,
            "doc": (
                f"Optional column layout: a mapping from role name to "
                f"0-based column index in the data file. Roles: "
                f"{', '.join(roles)}{extra}. Unnamed roles keep their "
                f"default position. Default: the documented column order."
                + (f" {note}" if note else "")
            ),
        }

    # ------------------------------------------------------------------
    # Optional per-file time system (time_offset / time_scale / time_frame)
    # ------------------------------------------------------------------
    def _parse_time_spec(self, c, i):
        """Validate one config entry's optional time-system keys (stage 0).

        Returns a dict with ``offset`` (days, added to the raw times
        first), ``scale`` (astropy time scale of the input), ``frame``
        (jd/hjd/bjd: where the input's clock sits), ``location`` (observer
        for the jd frame's light-travel and topocentric-scale terms),
        ``ephemeris`` (solar-system ephemeris for light_travel_time) and
        ``needs_conversion``.  The default -- offset 0, scale tdb, frame
        bjd -- is BJD_TDB in, BJD_TDB out, untouched.
        """
        label = f"{self.prefix}[{self.names[i]}]"

        offset = c.get("time_offset", 0.0)
        if isinstance(offset, bool) or not isinstance(
            offset, (int, float, np.integer, np.floating)
        ):
            raise ValueError(
                f"[{label}] time_offset must be a number (days, added to "
                f"every input time); got {offset!r}."
            )

        scale = c.get("time_scale", "tdb")
        if not isinstance(scale, str):
            raise ValueError(
                f"[{label}] time_scale must be one of {list(_TIME_SCALES)}; "
                f"got {scale!r}."
            )
        scale = _TIME_SCALE_ALIASES.get(scale.lower(), scale.lower())
        if scale not in _TIME_SCALES:
            raise ValueError(
                f"[{label}] time_scale must be one of {list(_TIME_SCALES)} "
                f"(or 'ut' for ut1); got {c.get('time_scale')!r}."
            )

        frame = c.get("time_frame", "bjd")
        if not isinstance(frame, str) or frame.lower() not in _TIME_FRAMES:
            raise ValueError(
                f"[{label}] time_frame must be one of "
                f"{list(_TIME_FRAMES)}; got {frame!r}."
            )
        frame = frame.lower()

        location = c.get("time_location")
        if location is not None:
            ok = isinstance(location, str) or (
                isinstance(location, (list, tuple))
                and len(location) in (2, 3)
                and all(
                    isinstance(v, (int, float, np.integer, np.floating))
                    and not isinstance(v, bool)
                    for v in location
                )
            )
            if not ok:
                raise ValueError(
                    f"[{label}] time_location must be an observatory name "
                    f"(astropy EarthLocation.of_site) or [lon_deg, lat_deg"
                    f"(, height_m)]; got {location!r}."
                )

        ephemeris = c.get("time_ephemeris", "builtin")
        if not isinstance(ephemeris, str):
            raise ValueError(
                f"[{label}] time_ephemeris must be an astropy solar-system "
                f"ephemeris name ('builtin', 'jpl', 'de440', ...); got "
                f"{ephemeris!r}."
            )

        return {
            "offset": float(offset),
            "scale": scale,
            "frame": frame,
            "location": location,
            "ephemeris": ephemeris,
            "needs_conversion": scale != "tdb" or frame != "bjd",
        }

    @property
    def has_nontrivial_time_spec(self):
        """True when any file sets a time offset or a time-system conversion."""
        return any(
            s["offset"] != 0.0 or s["needs_conversion"]
            for s in self.time_specs
        )

    def _to_bjd_tdb(self, t, i):
        """Convert file ``i``'s raw times to BJD_TDB, per its time spec.

        ``time_offset`` is added first (so truncated times like
        BJD-2450000 or MJD become absolute JDs); the scale/frame
        conversion then runs on absolute JDs only.  The algorithm is the
        standard one (Eastman, Siverd & Gaudi 2010):

        1. strip the input frame's light-travel correction to recover the
           observer's JD in the input scale -- ``t = t_obs + ltt(t_obs)``
           is inverted by fixed-point iteration, which converges below a
           nanosecond in 3 passes because d(ltt)/dt <= v_earth/c ~ 1e-4;
        2. convert the time scale to TDB (astropy/erfa: leap seconds for
           UTC/TAI, the erfa TDB-TT model, IERS tables for UT1);
        3. add back the barycentric light-travel time in TDB.

        Input already in the bjd frame skips 1 and 3: the barycentric
        correction appears identically on both sides and cancels exactly,
        so a scale-only conversion (BJD_UTC -> BJD_TDB) needs no
        coordinates.

        Accuracy notes (why the remaining terms are out of scope):
        the observer's position enters through ``time_location`` (omitting
        it costs up to 21 ms of geocenter-vs-observatory Romer delay);
        the builtin (erfa) ephemeris is good to a few microseconds of
        light travel (``time_ephemeris: de440`` reaches ns, needs
        jplephem); a single float64 JD quantizes at ~40 microseconds
        anyway, which is the real floor here; TT(BIPM) (~30 us), the
        Shapiro delay (~us; ~100 us within ~1 deg of the Sun), and
        proper-motion/parallax evolution of the source direction (~us/yr
        for mas/yr motions) are all below that floor's usefulness and are
        not modeled.
        """
        spec = self.time_specs[i]
        if spec["offset"] != 0.0:
            t = t + spec["offset"]
        if not spec["needs_conversion"]:
            return t

        label = f"{self.prefix}[{self.names[i]}]"
        if t.min() < _MIN_ABS_JD:
            raise ValueError(
                f"[{label}] time_scale/time_frame conversion needs absolute "
                f"Julian Dates, but the smallest time after time_offset is "
                f"{t.min():.3f}. Set time_offset to restore full JDs (e.g. "
                f"2450000 for BJD-2450000 data, 2400000.5 for MJD)."
            )

        # astropy.coordinates is deliberately imported lazily: it is slow to
        # import and only needed when a file actually opts into conversion.
        from astropy.time import Time

        location = self._time_location(i)

        if spec["frame"] == "bjd":
            # Scale-only conversion: the barycentric light-travel term is
            # identical on both sides and cancels exactly (the TDB-vs-UTC
            # evaluation epoch of the correction matters at the 0.1 us
            # level), so no coordinates are needed at all.
            out = Time(
                t, format="jd", scale=spec["scale"], location=location
            ).tdb.jd
        else:
            coord = self._time_coord(i, label)
            kind = _TIME_FRAMES[spec["frame"]]
            ephemeris = spec["ephemeris"]

            t_obs = t
            if kind is not None:
                for _ in range(3):
                    ltt = Time(
                        t_obs,
                        format="jd",
                        scale=spec["scale"],
                        location=location,
                    ).light_travel_time(coord, kind=kind, ephemeris=ephemeris)
                    t_obs = t - ltt.jd
            t_tdb = Time(
                t_obs, format="jd", scale=spec["scale"], location=location
            ).tdb
            out = (
                t_tdb.jd
                + t_tdb.light_travel_time(
                    coord, kind="barycentric", ephemeris=ephemeris
                ).jd
            )
        logger.info(
            "[%s] converted %d times from %s_%s to BJD_TDB "
            "(median shift %+.3f s).",
            label,
            t.size,
            spec["frame"].upper(),
            spec["scale"].upper(),
            float(np.median(out - t)) * 86400.0,
        )
        return out

    def _time_coord(self, i, label):
        """The target ICRS direction for file ``i``'s light-travel terms.

        Reuses the star component's ra/dec exactly as astrometry and mulens
        do (``star_ndx`` on the file's config entry picks the star, default
        0; every star in one system is the same direction at the accuracy
        that matters here).  Requiring ``user_modified`` is deliberate: the
        defaults.yaml ra/dec are placeholders, and a conversion run against
        them would corrupt every BJD by up to +/-8 minutes with no error.
        """
        star_ndx = int(self.config[i].get("star_ndx", 0))
        ra = self.config_manager.resolve("star", "ra", element=star_ndx)
        dec = self.config_manager.resolve("star", "dec", element=star_ndx)
        if not (ra["user_modified"] and dec["user_modified"]):
            raise ValueError(
                f"[{label}] time_scale/time_frame conversion needs the "
                f"target's coordinates: set star.{star_ndx}.ra and "
                f"star.{star_ndx}.dec (deg) in the params file."
            )
        import astropy.units as u
        from astropy.coordinates import SkyCoord

        # resolve() returns per-element arrays even for shape=(); take the
        # single element rather than float()-ing an ndarray (NumPy 2 error).
        return SkyCoord(
            ra=float(np.ravel(ra["initval"])[0]) * u.Unit(ra["unit"] or "deg"),
            dec=float(np.ravel(dec["initval"])[0])
            * u.Unit(dec["unit"] or "deg"),
        )

    def _time_location(self, i):
        """File ``i``'s observer EarthLocation (geocenter when unset)."""
        import astropy.units as u
        from astropy.coordinates import EarthLocation

        location = self.time_specs[i]["location"]
        if location is None:
            return EarthLocation.from_geocentric(0.0, 0.0, 0.0, unit=u.m)
        if isinstance(location, str):
            return EarthLocation.of_site(location)
        lon, lat = float(location[0]), float(location[1])
        height = float(location[2]) if len(location) == 3 else 0.0
        return EarthLocation.from_geodetic(lon, lat, height)

    @staticmethod
    def _time_config_schema():
        """The shared time-system config-schema entries; children append them."""
        return [
            {
                "key": "time_offset",
                "kind": "option",
                "accepts": None,
                "required": False,
                "doc": (
                    "Optional offset in days added to every input time "
                    "before anything else (e.g. 2450000 for BJD-2450000 "
                    "data, 2400000.5 for MJD). Default 0."
                ),
            },
            {
                "key": "time_scale",
                "kind": "option",
                "accepts": None,
                "required": False,
                "doc": (
                    "Time scale of the input times: utc, tai, tt, tdb, "
                    "tcb, tcg or ut1 ('ut' is accepted for ut1; ut1 may "
                    "trigger an IERS table download). Default tdb."
                ),
            },
            {
                "key": "time_frame",
                "kind": "option",
                "accepts": None,
                "required": False,
                "doc": (
                    "Reference frame of the input times: jd (observer/"
                    "geocenter), hjd (heliocentric) or bjd (barycentric). "
                    "Anything but the default bjd+tdb is converted to "
                    "BJD_TDB at load time, which requires absolute JDs "
                    "(see time_offset); jd/hjd also require user-set star "
                    "ra/dec. Default bjd."
                ),
            },
            {
                "key": "time_location",
                "kind": "option",
                "accepts": None,
                "required": False,
                "doc": (
                    "Observer location for the time conversion: an astropy "
                    "observatory name (EarthLocation.of_site) or [lon_deg, "
                    "lat_deg(, height_m)]. Default geocenter (up to 21 ms "
                    "of Romer delay is unmodeled without it)."
                ),
            },
            {
                "key": "time_ephemeris",
                "kind": "option",
                "accepts": None,
                "required": False,
                "doc": (
                    "Solar-system ephemeris for the light-travel terms: "
                    "'builtin' (erfa, ~us accuracy) or a JPL kernel like "
                    "'de440' (~ns, needs jplephem + download). Default "
                    "builtin."
                ),
            },
        ]

    # ------------------------------------------------------------------
    # Optional per-file Gaussian-process noise (celerite2)
    # ------------------------------------------------------------------
    def _load_gp_config(self):
        """Read the per-instrument ``gp:`` key (stage 0, in ``__init__``).

        Populates ``self.gp_terms[i]`` -- a (possibly empty) tuple of kernel
        names for element ``i`` -- and ``self.has_gp``.  Absent or ``none``
        means independent Gaussian errors, which is the default for every
        file and leaves the model byte-for-byte what it was before this
        feature existed.
        """
        self.gp_terms = []
        for i, c in enumerate(self.config):
            terms = gp_support.parse_gp_spec(
                c.get("gp"), context=f"{self.prefix}[{self.names[i]}]"
            )
            if terms and not self.supports_gp:
                raise NotImplementedError(
                    f"[{self.prefix}[{self.names[i]}]] Gaussian-process noise "
                    f"is not supported by this component: it models more than "
                    f"one observable per data file (with different units), so "
                    f"a single GP kernel is ambiguous. Remove the 'gp:' key."
                )
            self.gp_terms.append(terms)
        self.has_gp = any(self.gp_terms)

        # Element index -> indices into the concatenated observation arrays,
        # sorted by time.  Filled by _prepare_gp.
        self._gp_obs_index = {}
        self._gp_time = None
        # Linear values of the log-sampled hyperparameters, as Deterministics
        # (built once in add_observation_likelihood).
        self._gp_linear = {}
        # celerite2 GaussianProcess objects, keyed by element index, and the
        # observations each conditioned on -- both filled by
        # add_observation_likelihood and consumed by _compile_gp_plotters.
        self.gp_objects = {}
        self._gp_observed_node = {}
        self._gp_pred_at_data = {}
        self._gp_pred_on_grid = {}

    def _gp_elements(self, kind):
        """Element indices that requested GP term ``kind``."""
        return [i for i, terms in enumerate(self.gp_terms) if kind in terms]

    @staticmethod
    def _gp_config_schema():
        """The shared ``gp`` config-schema entry; children append it."""
        return gp_support.gp_config_schema_entry()

    @classmethod
    def shared_parameter_names(cls):
        """The GP and robust-likelihood parameters, declared at the root of
        components/defaults.yaml.

        Only the scale-carrying ones are redeclared per component (their unit
        is the data's), so introspection needs this list to report the rest.
        Components that opt out of a feature declare none of its names.
        """
        names = []
        if cls.supports_gp:
            names += [
                name
                for kind in gp_support.GP_TERMS
                for name in gp_support.GP_TERM_PARAMS[kind]
            ]
        if cls.supports_robust_likelihood:
            names += [
                name
                for kind in robust_support.LIKELIHOOD_KINDS
                for name in robust_support.LIKELIHOOD_PARAMS[kind]
            ]
        return names

    def _prepare_gp(self, time, err, inst_map, user_factor=1.0):
        """Stage 1a: index and seed the GP terms from the loaded data.

        Call at the end of ``load_data``, once the concatenated ``time``,
        error and ``inst_map`` arrays exist.  Two jobs:

        1. Record, per GP element, the time-sorted indices into the
           concatenated arrays.  celerite2's semiseparable solver requires
           ascending times and does not check, so every array handed to it
           (times, sigmas, model, data) goes through this one permutation.
        2. Push a data-driven amplitude hint: the median reported error bar,
           i.e. the file's white-noise level.  Anything derived from the
           *observations* instead (their scatter, or the point-to-point
           scatter) measures the signal the physical model is supposed to
           explain -- an eccentric RV orbit or a microlensing peak dominates
           both -- and seeding the GP there invites it to swallow the signal
           before the sampler has moved.  Starting at the white-noise level is
           conservative in the safe direction, and the logit transform lets
           the amplitude climb during tuning if the data want it.  A user who
           knows the activity amplitude should just set ``initval``.

        ``user_factor`` converts the error from the internal unit the caller
        holds it in to the user unit the amplitude parameter is declared in
        (hints are pushed in user units).
        """
        if not self.has_gp:
            return

        time = np.asarray(time, dtype=float)
        err = np.asarray(err, dtype=float)
        inst_map = np.asarray(inst_map, dtype=int)
        self._gp_time = time

        for i, terms in enumerate(self.gp_terms):
            if not terms:
                continue
            sel = np.flatnonzero(inst_map == i)
            if sel.size < 3:
                raise ValueError(
                    f"[{self.prefix}[{self.names[i]}]] a Gaussian process "
                    f"needs at least 3 observations; this file has "
                    f"{sel.size}."
                )
            self._gp_obs_index[i] = sel[np.argsort(time[sel], kind="stable")]

            amp = float(np.median(err[sel])) * user_factor
            if not np.isfinite(amp) or amp <= 0.0:
                # Degenerate (zero/absent) errors: fall back to the
                # defaults.yaml start rather than hinting a zero amplitude,
                # which would pin the logit transform against its lower bound.
                continue
            for kind in terms:
                path = (
                    f"{self.prefix}.{i}.{gp_support.GP_AMPLITUDE_PARAM[kind]}"
                )
                self.config_manager.add_hint(path, amp)
                self.config_manager.add_scale_hint(path, amp)

    def _register_gp(self, manifest):
        """Stage 2: add this component's GP hyperparameters to ``manifest``.

        Every GP parameter is a full-length (``n_elements``) vector so that a
        user path resolves the same way as any other instrument parameter --
        ``rvinstrument.HARPS.gp_rot_period`` means the HARPS element,
        whichever element that is.  Elements that did not opt into a term are
        pinned fixed (``sigma: 0``) through ``internal_overrides``, which sits
        below RANK_USER: they cost the sampler nothing, and a user who wants
        one back can still override it.

        Returns ``manifest`` for chaining, like ``_register_noise``.
        """
        if not self.has_gp:
            return manifest

        for kind in gp_support.GP_TERMS:
            on = set(self._gp_elements(kind))
            if not on:
                continue
            off = [i for i in range(self.n_elements) if i not in on]
            entry = {}
            if off:
                pin = np.full(self.n_elements, np.nan)
                pin[off] = 0.0
                entry["overrides"] = {"sigma": pin.tolist()}
            for param in gp_support.GP_TERM_PARAMS[kind]:
                manifest[param] = dict(entry)
        return manifest

    def _build_gp_deterministics(self):
        """Record the linear value of every log-sampled hyperparameter.

        ``gp_rot_log_q0`` and friends are sampled in log10 (see
        components/gp.py); this exposes ``gp_rot_q0`` etc. as Deterministics so
        posterior tables and plots report the quantity the kernel actually
        uses, and caches the tensors for ``_gp_kernel``.
        """
        self._gp_linear = {}
        for log_name, lin_name in gp_support.GP_LOG_PARAMS.items():
            param = getattr(self, log_name, None)
            if param is None or param.value is None:
                continue
            self._gp_linear[lin_name] = pm.Deterministic(
                f"{self.prefix}.{lin_name}", pt.power(10.0, param.value)
            )

    def _gp_kernel(self, i):
        """The summed celerite2 kernel for element ``i``."""
        params_by_kind = {}
        for kind in self.gp_terms[i]:
            if kind == "rotation":
                params_by_kind[kind] = {
                    "sigma": self.gp_rot_sigma.value[i],
                    "period": self.gp_rot_period.value[i],
                    "Q0": self._gp_linear["gp_rot_q0"][i],
                    "dQ": self._gp_linear["gp_rot_dq"][i],
                    "f": self.gp_rot_f.value[i],
                }
            else:  # "sho"
                params_by_kind[kind] = {
                    "sigma": self.gp_sho_sigma.value[i],
                    "rho": self.gp_sho_rho.value[i],
                    "Q": self._gp_linear["gp_sho_q"][i],
                }
        return gp_support.build_kernel(self.gp_terms[i], params_by_kind)

    def add_observation_likelihood(self, name, mu, sigma, observed):
        """Stage 6: the observational likelihood, with or without GPs and
        robust families.

        Drop-in replacement for the ``pm.Normal(name, mu, sigma, observed)``
        each child used to write.  With no ``gp:`` or ``likelihood:`` key
        anywhere it *is* that call, unchanged.  Otherwise the observations
        split by instrument: the plain files keep one shared Normal, each GP
        file gets a celerite2 marginal likelihood, and each robust file gets
        its mixture Potential or Student-t -- all around the same physical
        model ``mu`` and the same ``sigma`` (data error plus jitter/err_scale,
        which stays the inlier scale under the mixture).

        ``mu``, ``sigma`` and ``observed`` are (n_total_obs,) tensors in the
        concatenated order the child built; the per-file sort recorded by
        ``_prepare_gp`` is applied here.
        """
        if not (self.has_gp or self.has_robust_likelihood):
            return pm.Normal(name, mu=mu, sigma=sigma, observed=observed)

        if self.has_gp and self._gp_time is None:
            raise RuntimeError(
                f"[{self.prefix}] a 'gp:' key is configured but load_data "
                f"never called _prepare_gp; the GP times are unknown."
            )
        if self.has_robust_likelihood and not self._robust_obs_index:
            raise RuntimeError(
                f"[{self.prefix}] a 'likelihood:' key is configured but "
                f"load_data never called _prepare_robust; the per-file "
                f"observation indices are unknown."
            )

        special = np.zeros(self.n_total_obs, dtype=bool)
        for idx in self._gp_obs_index.values():
            special[idx] = True
        for idx in self._robust_obs_index.values():
            special[idx] = True

        plain = np.flatnonzero(~special)
        if plain.size:
            pm.Normal(
                name,
                mu=mu[plain],
                sigma=sigma[plain],
                observed=observed[plain],
            )

        self._add_robust_likelihoods(name, mu, sigma, observed)

        if not self.has_gp:
            return

        self._build_gp_deterministics()
        for i in sorted(self._gp_obs_index):
            idx = self._gp_obs_index[i]
            t = gp_support.check_sorted(
                self._gp_time[idx], context=f"{self.prefix}[{self.names[i]}]"
            )
            self.gp_objects[i] = gp_support.marginal_likelihood(
                f"{name}.gp.{self.names[i]}",
                self._gp_kernel(i),
                t=t,
                yerr=sigma[idx],
                mean=mu[idx],
                observed=observed[idx],
            )
            # Kept for the plotting path: predicting the GP component needs
            # the same observations the likelihood conditioned on, in the same
            # sorted order.
            self._gp_observed_node[i] = observed[idx]
            logger.info(
                "[%s] %s: %s GP on %d observations.",
                self.prefix,
                self.names[i],
                "+".join(self.gp_terms[i]),
                idx.size,
            )

    # ------------------------------------------------------------------
    # GP conditional prediction, for plotting
    # ------------------------------------------------------------------
    def _compile_gp_plotters(self, system):
        """Compile per-file evaluators of the GP conditional mean.

        Call from a child's ``compile_plotters``.  Builds two compiled
        functions per GP file, both of the *pure GP component*
        (``include_mean=False``, so celerite2 subtracts the physical model the
        likelihood conditioned on and returns only the correlated part):

          ``_gp_pred_at_data[i]``  the GP at that file's own observation
              times -- what plots subtract from the data.  Reuses the
              likelihood's factorization, so it costs about one extra logp
              evaluation.
          ``_gp_pred_on_grid[i]``  the GP on an arbitrary (sorted) time grid
              -- what plots add to a model curve.  Needs celerite2's
              general_matmul ops, O((N+M) J^2).

        Both take the same positional ``system.plot_params`` the other
        compiled plotters take, so a caller can evaluate them per posterior
        draw exactly like the physical model curves.
        """
        if not self.gp_objects:
            return
        import pytensor

        t_input = pt.vector("gp_t_input")
        param_symbols = [p.value for p in system.plot_params]

        self._gp_pred_at_data = {}
        self._gp_pred_on_grid = {}
        for i, gp in self.gp_objects.items():
            y = self._gp_observed_node[i]
            self._gp_pred_at_data[i] = pytensor.function(
                inputs=param_symbols,
                outputs=gp.predict(y, include_mean=False),
                on_unused_input="ignore",
            )
            self._gp_pred_on_grid[i] = pytensor.function(
                inputs=[t_input] + param_symbols,
                outputs=gp.predict(y, t=t_input, include_mean=False),
                on_unused_input="ignore",
            )

    def has_gp_plotters(self):
        """True when _compile_gp_plotters has produced evaluators."""
        return bool(getattr(self, "_gp_pred_at_data", None))

    def gp_mean_at_data(self, system, point):
        """GP component at every observation, in the concatenated data order.

        Returns an ``(n_total_obs,)`` array that is zero for observations from
        files without a GP, so a caller can subtract it unconditionally.  In
        the internal units of the observable (the same units ``mu`` was in).
        """
        out = np.zeros(self.n_total_obs, dtype=float)
        if not self.has_gp_plotters():
            return out
        values = self._point_to_plot_params(point, system)
        for i, fn in self._gp_pred_at_data.items():
            out[self._gp_obs_index[i]] = np.asarray(fn(*values), dtype=float)
        return out

    def gp_mean_on_grid(self, system, point, i, t_grid):
        """GP component for file ``i`` on ``t_grid`` (must be sorted ascending).

        Zeros when that file has no GP, so a caller can add it
        unconditionally.  Internal units of the observable.
        """
        t_grid = np.asarray(t_grid, dtype=float)
        if not self.has_gp_plotters() or i not in self._gp_pred_on_grid:
            return np.zeros_like(t_grid)
        values = self._point_to_plot_params(point, system)
        return np.asarray(
            self._gp_pred_on_grid[i](t_grid, *values), dtype=float
        )

    # ------------------------------------------------------------------
    # Optional per-file robust likelihood (hogg mixture / Student-t)
    # ------------------------------------------------------------------
    def _load_likelihood_config(self):
        """Read the per-instrument ``likelihood:`` key (stage 0, in __init__).

        Populates ``self.likelihood_kinds[i]`` -- ``""`` (plain Gaussian, the
        default for every file) or a key from
        ``likelihood.LIKELIHOOD_KINDS`` -- and ``self.has_robust_likelihood``.
        Runs after ``_load_gp_config`` so the gp/likelihood conflict is
        caught at construction.
        """
        self.likelihood_kinds = []
        for i, c in enumerate(self.config):
            label = f"{self.prefix}[{self.names[i]}]"
            kind = robust_support.parse_likelihood_spec(
                c.get("likelihood"), context=label
            )
            if kind and not self.supports_robust_likelihood:
                raise NotImplementedError(
                    f"[{label}] a robust likelihood is not supported by this "
                    f"component: it models more than one observable per data "
                    f"file (with different units), so a single outlier scale "
                    f"is ambiguous. Remove the 'likelihood:' key."
                )
            if kind and self.gp_terms[i]:
                raise ValueError(
                    f"[{label}] 'gp:' and 'likelihood:' cannot be combined "
                    f"on one file: celerite2's closed-form marginal "
                    f"likelihood is Gaussian-only, so a robust observation "
                    f"model cannot be marginalized through it. Choose one."
                )
            self.likelihood_kinds.append(kind)
        self.has_robust_likelihood = any(self.likelihood_kinds)

        # Element index -> indices into the concatenated observation arrays
        # (filled by _prepare_robust), the per-file symbolic outlier log-odds
        # (filled by _add_robust_likelihoods), their lazily compiled
        # evaluators, and the linear values of log-sampled parameters.
        self._robust_obs_index = {}
        self._hogg_logodds = {}
        self._hogg_prob_fns = None
        self._robust_linear = {}

    def _robust_elements(self, kind):
        """Element indices that requested likelihood family ``kind``."""
        return [i for i, k in enumerate(self.likelihood_kinds) if k == kind]

    @staticmethod
    def _likelihood_config_schema():
        """The shared ``likelihood`` config-schema entry; children append it."""
        return robust_support.likelihood_config_schema_entry()

    def _prepare_robust(self, err, inst_map, user_factor=1.0):
        """Stage 1a: index the robust files and seed the mixture scale.

        Call at the end of ``load_data``, right next to ``_prepare_gp``.
        Records each opted-in element's indices into the concatenated
        observation arrays (no sort needed -- both families are per-point
        independent), and pushes a data-driven hint for the hogg background
        scale: ``10 x median(err)``, i.e. well clear of the inlier scatter,
        so the two mixture components start separated and cannot swap roles
        during tuning.  Seeding from the observations' own scatter would
        instead measure the physical signal (the same reasoning as the GP
        amplitude seed).

        ``user_factor`` converts the error from the internal unit the caller
        holds it in to the user unit ``out_scale`` is declared in.
        """
        if not self.has_robust_likelihood:
            return

        err = np.asarray(err, dtype=float)
        inst_map = np.asarray(inst_map, dtype=int)

        for i, kind in enumerate(self.likelihood_kinds):
            if not kind:
                continue
            sel = np.flatnonzero(inst_map == i)
            if sel.size == 0:
                raise ValueError(
                    f"[{self.prefix}[{self.names[i]}]] a 'likelihood:' key "
                    f"is set but the file contributed no observations."
                )
            self._robust_obs_index[i] = sel

            scale_param = robust_support.LIKELIHOOD_SCALE_PARAM.get(kind)
            if scale_param is None:
                continue
            scale = 10.0 * float(np.median(err[sel])) * user_factor
            if not np.isfinite(scale) or scale <= 0.0:
                # Degenerate (zero/absent) errors: keep the defaults.yaml
                # start rather than pinning the logit against its bound.
                continue
            path = f"{self.prefix}.{i}.{scale_param}"
            self.config_manager.add_hint(path, scale)
            self.config_manager.add_scale_hint(path, scale)

    def _register_robust(self, manifest):
        """Stage 2: add this component's robust-likelihood parameters.

        Same shape as ``_register_gp``: every parameter is a full-length
        (``n_elements``) vector so user paths resolve by instrument name, and
        elements that did not opt into a family are pinned fixed
        (``sigma: 0``) through ``internal_overrides`` -- free to the sampler,
        still user-overridable.  Returns ``manifest`` for chaining.
        """
        if not self.has_robust_likelihood:
            return manifest

        for kind in robust_support.LIKELIHOOD_KINDS:
            on = set(self._robust_elements(kind))
            if not on:
                continue
            off = [i for i in range(self.n_elements) if i not in on]
            entry = {}
            if off:
                pin = np.full(self.n_elements, np.nan)
                pin[off] = 0.0
                entry["overrides"] = {"sigma": pin.tolist()}
            for param in robust_support.LIKELIHOOD_PARAMS[kind]:
                manifest[param] = dict(entry)
        return manifest

    def _build_robust_deterministics(self):
        """Record the linear value of every log-sampled robust parameter.

        ``t_log_nu`` is sampled in log10 (see components/likelihood.py); this
        exposes ``t_nu`` as a Deterministic so posterior tables report the
        degrees of freedom the likelihood actually uses.
        """
        self._robust_linear = {}
        for log_name, lin_name in robust_support.LIKELIHOOD_LOG_PARAMS.items():
            param = getattr(self, log_name, None)
            if param is None or param.value is None:
                continue
            self._robust_linear[lin_name] = pm.Deterministic(
                f"{self.prefix}.{lin_name}", pt.power(10.0, param.value)
            )

    def _add_robust_likelihoods(self, name, mu, sigma, observed):
        """Add each robust file's likelihood term (from the dispatcher)."""
        if not self.has_robust_likelihood:
            return

        self._build_robust_deterministics()
        for i in sorted(self._robust_obs_index):
            idx = self._robust_obs_index[i]
            kind = self.likelihood_kinds[i]
            if kind == "hogg":
                resid = observed[idx] - mu[idx]
                logp = robust_support.hogg_logp(
                    resid,
                    sigma[idx],
                    self.out_frac.value[i],
                    self.out_scale.value[i],
                )
                pm.Potential(f"{name}.hogg.{self.names[i]}", pt.sum(logp))
                self._hogg_logodds[i] = robust_support.hogg_outlier_logodds(
                    resid,
                    sigma[idx],
                    self.out_frac.value[i],
                    self.out_scale.value[i],
                )
            else:  # "studentt"
                pm.StudentT(
                    f"{name}.t.{self.names[i]}",
                    nu=self._robust_linear["t_nu"][i],
                    mu=mu[idx],
                    sigma=sigma[idx],
                    observed=observed[idx],
                )
            logger.info(
                "[%s] %s: %s likelihood on %d observations.",
                self.prefix,
                self.names[i],
                kind,
                idx.size,
            )

    def outlier_prob_at_data(self, system, point):
        """Posterior outlier probability of every observation, at ``point``.

        Returns an ``(n_total_obs,)`` array in the concatenated data order:
        ``sigmoid`` of the hogg mixture's per-point log-odds for observations
        from hogg files, and zero everywhere else, so a caller can threshold
        or color by it unconditionally.  This is the auditable replacement
        for a hard bad-data mask -- evaluate it per posterior draw (it is
        compiled lazily against ``system.plot_params``, like the GP
        evaluators) and report the average.
        """
        out = np.zeros(self.n_total_obs, dtype=float)
        if not self._hogg_logodds:
            return out
        if self._hogg_prob_fns is None:
            import pytensor

            param_symbols = [p.value for p in system.plot_params]
            self._hogg_prob_fns = {
                i: pytensor.function(
                    inputs=param_symbols,
                    outputs=pt.sigmoid(node),
                    on_unused_input="ignore",
                )
                for i, node in self._hogg_logodds.items()
            }
        values = self._point_to_plot_params(point, system)
        for i, fn in self._hogg_prob_fns.items():
            out[self._robust_obs_index[i]] = np.asarray(
                fn(*values), dtype=float
            )
        return out

    # NOTE: a GP imposes NO sampler constraint, so this class deliberately
    # does not override sampler_requirements().  celerite2 ships a JAX
    # implementation of every op it uses and registers it with PyTensor's JAX
    # linker (``@jax_funcify.register(_CeleriteOp)`` in celerite2/pymc/ops.py,
    # mapping factor/solve/matmul and their _rev gradients onto
    # celerite2.jax.ops primitives), so the JAX-funcifying samplers work.
    # Verified 2026-07-24 on celerite2 0.3.3 + pymc 6.1 + pytensor 3.1: the
    # JAX logp matches the C backend to 1e-14, its gradient is finite, and
    # nuts_sampler='numpyro' samples the full RV+GP model on 1, 2 and 4
    # chains.  ('blackjax' is separately broken by pymc passing a progress_bar
    # kwarg blackjax 1.6.2's kernel does not accept, but that fails
    # identically with no GP in the model, so it is not this component's
    # constraint to declare -- see exozippy/compat/blackjax_progressbar.py.)
    #
    # Before adding a sampler exclusion anywhere, SAMPLE with the sampler.
    # This class briefly carried one written by analogy with transit's, which
    # was wrong for celerite2; transit's own turned out to be fixable rather
    # than fundamental, and is gone too (exoplanet-core >= 0.4.0 makes the
    # limb-darkening op JAX-differentiable).

    # ------------------------------------------------------------------
    # Shared noise machinery
    # ------------------------------------------------------------------
    @staticmethod
    def _jitter_floor(err, factor=1.0):
        """Additive jitter-variance floor: ``-0.95 * min(err*factor)**2``.

        Keeps the total variance ``err**2 + jitter_variance`` strictly
        positive.  ``factor`` converts the raw error column to the internal
        unit first (rv scales to m/s; transit/astrometry use factor 1).
        """
        scaled_min = np.min(np.asarray(err, dtype=float)) * factor
        return -0.95 * scaled_min**2

    def _register_noise(self, manifest, jittervar_lower=None):
        """Add the shared per-instrument noise term to ``manifest``.

        ``"jitter_variance"``: additive variance (needs a per-instrument
        ``jittervar_lower`` floor) plus a reported ``jitter``.
        ``"err_scale"``: a single multiplicative scale.
        Returns ``manifest`` for chaining.
        """
        if self.noise_model == "err_scale":
            manifest["err_scale"] = None
        else:
            if jittervar_lower is None:
                raise ValueError(
                    f"[{self.prefix}] jitter_variance noise model requires a "
                    f"jittervar_lower floor."
                )
            manifest["jitter_variance"] = {"lower": jittervar_lower}
            manifest["jitter"] = "default"
        return manifest

    def total_sigma(self, err_tensor):
        """Per-observation sigma combining data error with the instrument's
        noise term, mapped over ``inst_map_tensor``.

        Requires the per-observation ``inst_map`` machinery (rv/transit/
        mulens); astrometry, whose data are stored per dataset, forms its
        sigma inline instead.
        """
        if self.noise_model == "err_scale":
            return err_tensor * self.err_scale.value[self.inst_map_tensor]
        return pt.sqrt(
            pt.sqr(err_tensor)
            + self.jitter_variance.value[self.inst_map_tensor]
        )

    # ------------------------------------------------------------------
    # Shared optional detrending against extra data columns
    # ------------------------------------------------------------------
    @staticmethod
    def _build_block_detrend(all_detrend, n_total_obs):
        """Block-diagonal detrend design matrix from per-instrument blocks.

        ``all_detrend`` is one ``(n_obs_i, n_cols_i)`` array per instrument
        (``n_cols_i == 0`` when that instrument has no detrend columns).
        Returns ``(matrix, n_detrend_per_inst, total_detrend_cols)`` where the
        matrix is ``(n_total_obs, total_detrend_cols)`` with each instrument's
        columns placed on the block diagonal so coefficients never mix across
        instruments.
        """
        n_per = [d.shape[1] for d in all_detrend]
        total = sum(n_per)
        matrix = np.zeros((n_total_obs, total))
        r, c = 0, 0
        for block in all_detrend:
            n_r, n_c = block.shape
            if n_c > 0:
                matrix[r : r + n_r, c : c + n_c] = block
            r, c = r + n_r, c + n_c
        return matrix, n_per, total
