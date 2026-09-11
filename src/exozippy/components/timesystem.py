"""Astronomical time systems for a data component.

The per-file ``time_offset`` / ``time_scale`` / ``time_frame`` /
``time_location`` / ``time_ephemeris`` keys, and the conversion of raw input
times to BJD_TDB.  Extracted VERBATIM from ``components/instrument.py`` in
2026-09; the bodies below are byte-identical to what lived there, so no
shipped fit moves.

WHY THIS IS ITS OWN MODULE
--------------------------

``Instrument`` is the shared scaffolding for a data component, and most of it
-- columns, masks, detrending, GP kernels, robust likelihoods, jitter, plot
styles -- is FIELD-NEUTRAL.  This part is not.  It assumes the independent
variable is a time, that the time is a Julian Date, that there is a solar
system to correct light-travel across, and that the observation has a place on
the Earth.  Every one of those is true of a telescope and none is true of a
blood draw.

The distinction matters because the defaults are SILENT: ``time_frame``
defaults to ``bjd`` and ``time_scale`` to ``tdb``, which together set
``needs_conversion`` False and pass the input through untouched.  A
non-astronomy data component could therefore inherit the whole of
``Instrument`` and appear to work perfectly, while its schema advertised a
barycentric correction on a plasma sample and its documentation promised an
observatory location for a clinical assay.  Working-but-wrong is the failure
this split exists to make impossible: a component that does not mix
:class:`TimeSystem` in has no time vocabulary at all, and cannot accidentally
claim one.

That was found by writing ``components/pharmacokinetics``, which inherits
``Component`` directly for exactly this reason -- see
``components/pharmacokinetics/pharmacokinetics.md``.

WHAT A HOST CLASS MUST PROVIDE
------------------------------

:class:`TimeSystem` is a plain mixin, not a ``Component`` subclass, so the
factory's sweep for ``Component`` subclasses cannot pick it up.  Use it as
``class Instrument(TimeSystem, Component)``.  It needs, from the host:

``prefix``, ``names``, ``config``   -- for error messages and per-file lookup
``config_manager``                  -- to resolve the target's ra/dec
``resolve_star_ndx``                -- ``Component``'s star reference resolver
``time_specs``                      -- built by :meth:`parse_time_specs`

and it provides ``_to_bjd_tdb`` for the host's own read path to call.
"""

import logging

import numpy as np

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


class TimeSystem:
    """Per-file astronomical time-system handling for a data component.

    See the module docstring for why this is separable from the rest of a
    data component, and for what a host class has to provide.
    """

    def parse_time_specs(self):
        """Build ``self.time_specs``, one parsed spec per config entry.

        Called from the host's ``__init__`` so a malformed spec fails at
        construction rather than part-way through a data load.  A host that
        wants the vocabulary but parses its config elsewhere may set
        ``time_specs`` itself; nothing here requires this entry point.
        """
        self.time_specs = [
            self._parse_time_spec(c, i) for i, c in enumerate(self.config)
        ]
        return self.time_specs

    def _parse_time_spec(self, c, i):
        """Validate one config entry's optional time-system keys (at construction).

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
        star_ndx = self.resolve_star_ndx(
            self.config[i].get("star_ndx"),
            f"[{self.prefix}] {self.names[i]} star_ndx",
        )
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
