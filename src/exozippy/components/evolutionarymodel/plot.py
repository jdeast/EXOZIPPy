# general imports
import logging

logger = logging.getLogger(__name__)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba

from exozippy.outputs.contour_plot import Contour

# exozippy local imports
from exozippy.outputs.plot_helper_functions import (
    _extend_window,
    _padded_range,
)
from exozippy.outputs.texutils import latex_escape
from exozippy.reporting import SIGMA_WIDTHS


class MISTPlot:
    # The EEP window the seed search and the Kiel diagram treat as "the part of a
    # track a reader of this chart came for": 202 is MIST's zero-age main sequence
    # and 630 is well up the red giant branch.  It is a PLOTTING/SEEDING window,
    # never a bound -- the fit is free to sit anywhere the grid covers, and where
    # it does the window is widened rather than hiding the star it describes.
    KIEL_EEP_WINDOW = (202.0, 630.0)
    # MIST's primary EEPs bounding the main sequence: the zero-age main
    # sequence (202) and the terminal-age main sequence (454, core hydrogen
    # exhaustion -- the turnoff).  The summary figure's Kiel panel frames
    # them (meta["summary"]["window"]) and marks reference ages along them.
    MAIN_SEQUENCE_EEPS = (202.0, 454.0)
    # Reference ages on the summary figure: this many, round, along the
    # main sequence up to the turnoff or the age of the universe, whichever
    # is younger.
    SUMMARY_N_AGES = 3
    AGE_UNIVERSE_GYR = 13.8

    # Nominal logg axis window, chosen to exclude the giant regime for the same
    # reason.  Ascending [lo, hi]; the Kiel convention's reversal is meta's
    # `y_inverted`, applied after the range (see plotrender._apply_axes).
    KIEL_LOGG_WINDOW = (3.0, 5.0)

    # How close to an edge counts as "near" it, as a fraction of the nominal
    # span, and equally how far past a violating value the widened edge is put.
    # 10% is 0.2 dex in logg and ~43 EEPs -- enough that a star sitting on an
    # edge is drawn with visible context on both sides rather than clipped.
    KIEL_WINDOW_MARGIN_FRAC = 0.1

    # Breathing room around the Teff axis, as a fraction of the span it has to
    # cover.  0.05 is matplotlib's own default axis margin, so a chart whose
    # x_range is computed here looks like one that autoscaled.
    KIEL_X_PAD_FRAC = 0.05

    # Column order of compile_plotters' Kiel node, mirroring mist_grid's
    # OUTPUT_INDEX idiom so plot_data slices it by name rather than by a magic
    # integer that silently shifts when a column is added.
    KIEL_COLUMNS = (
        "logmass",
        "initfeh",
        "logmass_initial",
        "teff_mist",
        "logg_mist",
        "teff_fit",
        "logg_fit",
        "sigma_teff_mist",
        "sigma_logg_mist",
        "eep",
    )
    KIEL_INDEX = {name: i for i, name in enumerate(KIEL_COLUMNS)}

    MIST_PLOT_COLORS = {
        "mist_track": "#005f73",
        "mist_point": "#005f73",
        "fit_point": "#a72564",
        "mist_contour": "#1f77b4",
        "fit_contour": "#a72564",
    }

    def __init__(self, system, points):
        # general information
        self.system = system
        self.points = points
        self.evolutionarymodel = self.system.active_components[
            "evolutionarymodel"
        ]
        # Parameter.posterior is None until System.distribute_posterior attaches
        # the chain (an xarray DataArray); no posterior means this is the start.
        self._posteriorBool = (
            getattr(self.system.star.teff, "posterior", None) is not None
        )

    # Private Chart.meta key: the on-chart Teff extent of ONE draw, which
    # `kiel_spec_groups` unions over draws for the spaghetti axis and then
    # strips, so it reaches neither renderer nor the draft.
    TEFF_EXTENT = "_teff_extent"

    @staticmethod
    def _summary_at(param, index):
        """One element's posterior summary, or None before a posterior exists.

        ``Parameter.summary`` is ONE summary object for a scalar parameter
        and a LIST of them for a vector one -- a system with more than one
        star.  Both plots here indexed it as an object, so a two-star system
        drew zero-length error bars on the Kiel diagram (the ``hasattr``
        guard turned the list into 0.0) and raised in the contours (review
        1.8.8).  This is the one reader.
        """
        summary = param.summary
        if summary is None:
            return None
        if isinstance(summary, (list, tuple)):
            # A summary list shorter than the star vector is a bookkeeping
            # bug upstream, not "no posterior yet" (review 1.8.8 (c)).
            if index >= len(summary):
                raise IndexError(
                    f"{param.label}: posterior summary has {len(summary)} "
                    f"element(s), asked for element {index}."
                )
            return summary[index]
        return summary

    def _fit_error_bars(self, param, star_index):
        """``(err_minus, err_plus)`` of the reported value for one star,
        ``(0.0, 0.0)`` before a posterior (the pre-flight plot)."""
        summary = self._summary_at(param, star_index)
        if summary is None:
            return 0.0, 0.0
        return float(summary.err_minus), float(summary.err_plus)

    def _reported_kiel(self):
        """The Kiel quantities at the REPORTED (posterior-median) parameters.

        The two marks on this chart -- the MIST prediction and the fitted
        point -- are single numbers a reader compares, so they are drawn once
        at the same medians the LaTeX/CSV tables quote, while the TRACKS stay
        per draw and carry the uncertainty.  ``System.distribute_posterior``
        has already attached the whole chain to each Parameter by the time
        run.py plots (``build_mode_reports`` runs first), in USER units and
        with the sample dimension last -- exactly what ``compute_summary``
        reads, so the median here is the number in the table.

        Returns ``None`` when no posterior is attached (the pre-flight plot,
        and the GUI's live-slider mode), leaving the caller to fall back to
        the point it was handed -- which in both of those cases is the only
        point there is.

        One caveat, deliberately left rather than papered over: the per-mode
        figures (run.py's ``_emit_per_mode_outputs``) hand ``plot()`` one
        mode's draws but no mode label, so these marks are the COMBINED
        posterior's medians there.  Fixing that means giving ``Component.plot``
        a mode argument, which is a change to every component's signature.
        """
        cached = self.evolutionarymodel._reported_kiel_cache
        if cached is not None:
            return cached

        point = {}
        for param in self.system.plot_params:
            post = getattr(param, "posterior", None)
            if post is None:
                continue
            # nanmedian over the sample axis (last) is exactly the median
            # compute_summary reports, without requiring it to have run --
            # param.summary is None until it does, and falling back to the
            # raw posterior there handed every draw in as the "median".
            arr = np.asarray(getattr(post, "values", post), dtype=float)
            point[param.label] = param.to_internal(np.nanmedian(arr, axis=-1))

        if not point:
            return None

        self.evolutionarymodel._reported_kiel_cache = np.atleast_2d(
            self.evolutionarymodel._compiled_kiel(
                *self.evolutionarymodel._point_to_plot_params(
                    point, self.system
                )
            )
        )
        return self.evolutionarymodel._reported_kiel_cache

    @staticmethod
    def reference_ages(age_lo, age_hi, n):
        """``n`` round ages (Gyr) spread through ``(age_lo, age_hi)``.

        The ages at 1/(n+1), 2/(n+1), ... of the span, each rounded to one
        significant figure -- two, then three, when rounding would merge two
        of them or push one out of the span.  Returns ``[]`` for an empty
        span.
        """
        if not age_hi > age_lo:
            return []
        fractions = np.arange(1, n + 1) / (n + 1)
        targets = age_lo + fractions * (age_hi - age_lo)
        for digits in (1, 2, 3):
            ages = [float(f"{t:.{digits}g}") for t in targets]
            if len(set(ages)) == n and all(age_lo < a < age_hi for a in ages):
                return ages
        return [float(t) for t in targets]

    def summary_track_marks(self, track_rows):
        """The Kiel chart's summary hints from the drawn track's rows:
        ``{"window": [teff_lo, teff_hi, logg_lo, logg_hi], "marks": [{"x",
        "y", "text"}, ...]}``.

        The window is the extent of the track's main sequence
        (``MAIN_SEQUENCE_EEPS``), and the marks are ``SUMMARY_N_AGES``
        round ages along it up to the turnoff or ``AGE_UNIVERSE_GYR``, each
        at the track's (Teff, logg) at that age.  A track with no
        main-sequence row (a star drawn off it) gets neither.
        """
        eep = np.asarray(track_rows["eep"], dtype=float)
        zams, tams = self.MAIN_SEQUENCE_EEPS
        rows = (eep >= zams) & (eep <= tams)
        if not rows.any():
            return {"window": None, "marks": []}
        teff = np.asarray(track_rows["teff"], dtype=float)[rows]
        logg = np.asarray(track_rows["logg"], dtype=float)[rows]
        age = np.asarray(track_rows["age"], dtype=float)[rows]
        order = np.argsort(age)
        teff, logg, age = teff[order], logg[order], age[order]
        last = min(age[-1], self.AGE_UNIVERSE_GYR)
        marks = [
            {
                "x": float(np.interp(a, age, teff)),
                "y": float(np.interp(a, age, logg)),
                "text": f"{a:g} Gyr",
            }
            for a in self.reference_ages(age[0], last, self.SUMMARY_N_AGES)
        ]
        window = [
            float(teff.min()),
            float(teff.max()),
            float(logg.min()),
            float(logg.max()),
        ]
        return {"window": window, "marks": marks}

    def _track_rows(self, i, logmass_initial, initfeh, eep_window):
        """Instance ``i``'s track over ``eep_window``: ``{"teff", "logg",
        "eep", "age"}`` arrays, row-aligned, age in Gyr.

        The real evolutionary track: the tabulated grid at the star's
        INITIAL mass (``mist_grid.interpolate_track(..., mass="initial")``),
        with each row's logg from that row's PRESENT-DAY mass, so the curve
        is the one star's path including its mass loss. (At a fixed
        present-day mass the re-axed grid is not one star's track.)

        Rows the grid flags as unreliable (``here_be_dragons``) are cut, so
        the drawn track stops where the models stop being trustworthy rather
        than running out into the hydrogen-exhausted tail, where feh_mist is
        30.0 and the tabulated ages are unresolved.  The fit is not forbidden
        from going there -- the dragon penalty is smooth, not a wall -- this
        is only about not drawing a curve nobody should read.

        ``eep_window`` restricts the drawn arc to the main sequence and red
        giant branch (KIEL_EEP_WINDOW), already widened by ``_extend_window``
        for a star that sits near or past an edge.  That is presentation, not
        physics, for the same reason: the window must never hide the star the
        chart is describing.

        The logg window is deliberately NOT applied here.  It is the y axis
        (``meta["y_range"]``), so a widened axis cannot open a gap where the
        curve had already been cut away.
        """
        from exozippy.constants import LOGG_CONST

        from .mist_grid import TRACK_INDEX, interpolate_track

        grid = self.evolutionarymodel._grids[i]
        track = interpolate_track(
            grid, logmass_initial, initfeh, mass="initial"
        )
        # EEP is the grid AXIS the track's rows are indexed by, not one of
        # the interpolated TRACK_COLUMNS, and interpolate_track leaves that
        # axis alone -- so grid["eep_pts"] is row-aligned with `track` by
        # construction.
        eep = np.asarray(grid["eep_pts"], dtype=float)
        teff = track[:, TRACK_INDEX["teff_mist"]]
        radius = track[:, TRACK_INDEX["radius_mist"]]
        logg = (
            LOGG_CONST
            + track[:, TRACK_INDEX["logmass_present"]]
            - 2.0 * np.log10(np.where(radius > 0, radius, np.nan))
        )

        keep = (
            (track[:, TRACK_INDEX["here_be_dragons"]] <= 0)
            & np.isfinite(teff)
            & (radius > 0)
            & (eep >= eep_window[0])
            & (eep <= eep_window[1])
        )

        return {
            "teff": teff[keep],
            "logg": logg[keep],
            "eep": eep[keep],
            "age": track[:, TRACK_INDEX["age_mist_gyr"]][keep],
        }

    def _track_curve(self, i, logmass_initial, initfeh, eep_window):
        """(teff, logg) along instance ``i``'s track, over ``eep_window``
        (``_track_rows`` without the EEP and age)."""
        rows = self._track_rows(i, logmass_initial, initfeh, eep_window)
        return rows["teff"], rows["logg"]

    def _plot_kiel_trace(self, star_idx, point=None):
        """Kiel diagram: the track, the MIST point, and the fitted point.

        We only want to plot the MIST and fit point once per star, not once
        per instance, and not once per trace.  So we use the star_indices to
        get the unique stars, and both marks are given ``role="data"``:
        ``plotrender.render_spec_groups`` takes only ``role="model"`` traces
        from the non-reference draws, so a data trace is drawn exactly once
        however many draws are overlaid.  Their VALUES come from
        ``_reported_kiel`` rather than from this draw, so "once" means once
        at the reported medians and not once at whichever sample happened to
        be first.
        """
        from exozippy.chart import Chart, Trace

        if point is None:
            return []

        kiel = np.atleast_2d(
            self.evolutionarymodel._compiled_kiel(
                *self.evolutionarymodel._point_to_plot_params(
                    point, self.system
                )
            )
        )
        marks = self._reported_kiel()
        if (
            marks is None or not self._posteriorBool
        ):  # marks is None when there is no posterior attached
            marks = kiel

        # One y axis, so the logg window is the union over stars.  The fit
        # point's systematic bars count: a mark whose error bar runs off the
        # chart reads as a mark nobody bothered to fit inside it.
        logg_margin = self.KIEL_WINDOW_MARGIN_FRAC * (
            self.KIEL_LOGG_WINDOW[1] - self.KIEL_LOGG_WINDOW[0]
        )
        logg_fit = np.array([marks[star_idx, self.KIEL_INDEX["logg_fit"]]])
        logg_mist = np.array([marks[star_idx, self.KIEL_INDEX["logg_mist"]]])
        sigma_logg_mist = np.array(
            [np.abs(marks[star_idx, self.KIEL_INDEX["sigma_logg_mist"]])]
        )
        logg_window = _extend_window(
            self.KIEL_LOGG_WINDOW,
            np.concatenate(
                [
                    logg_fit,
                    logg_mist - sigma_logg_mist,
                    logg_mist + sigma_logg_mist,
                ]
            ),
            logg_margin,
        )
        eep_margin = self.KIEL_WINDOW_MARGIN_FRAC * (
            self.KIEL_EEP_WINDOW[1] - self.KIEL_EEP_WINDOW[0]
        )

        # Teff values that will actually be ON the chart, gathered as the
        # traces are built -- see the x_range comment below.
        teff_fit = marks[star_idx, self.KIEL_INDEX["teff_fit"]]
        teff_mist = marks[star_idx, self.KIEL_INDEX["teff_mist"]]
        sigma_teff_mist = np.abs(
            marks[star_idx, self.KIEL_INDEX["sigma_teff_mist"]]
        )
        visible_teff = [
            teff_fit,
            teff_mist - sigma_teff_mist,
            teff_mist + sigma_teff_mist,
        ]

        # initialize the list of traces to be returned
        traces = []

        # `star_idx` indexes this component's INSTANCES (the rows of the
        # compiled Kiel node); the star it describes is star_indices[star_idx].
        # The trace names used star_idx directly, so instance 1 tied to star
        # "C" was labelled with star "B"'s name.
        star_index = self.evolutionarymodel.star_indices[star_idx]
        star_name = self.system.star.names[star_index]
        # grab the logmass and initfeh for the draw
        logmass_initial = float(
            kiel[star_idx, self.KIEL_INDEX["logmass_initial"]]
        )
        initfeh = float(kiel[star_idx, self.KIEL_INDEX["initfeh"]])

        eep = marks[star_idx, self.KIEL_INDEX["eep"]]
        eep_window = _extend_window(
            self.KIEL_EEP_WINDOW,
            [eep],
            eep_margin,
        )
        track_rows = self._track_rows(
            star_idx, logmass_initial, initfeh, eep_window
        )
        teff_track, logg_track = track_rows["teff"], track_rows["logg"]
        # Only the part of the arc inside the logg window: the rest is
        # drawn but clipped away, and letting it set the axis is what
        # made the panel mostly whitespace.
        on_chart = (logg_track >= logg_window[0]) & (
            logg_track <= logg_window[1]
        )
        visible_teff.append(teff_track[on_chart])
        traces.append(
            Trace(
                name=f"Star {star_name} MIST track",
                role="model",
                kind="line",
                x=teff_track,
                y=logg_track,
                style={
                    "color": self.MIST_PLOT_COLORS["mist_track"],
                    "lw": 1.0,
                    "legend": True,
                },
            )
        )
        # Both marks are role="data" so the renderers draw them ONCE,
        # from the reference group, on top of the track spaghetti
        # (zorder above _draw_model's 2).

        # The error bars ARE the systematic floor, i.e. how far the fit
        # is free to sit from the track before the penalty bites.  A
        # fitted quantity, not an observation; no other role draws
        # xerr/yerr.
        traces.append(
            Trace(
                name=f"Star {star_name} MIST model point",
                role="data",
                kind="scatter",
                x=np.array(teff_mist),
                y=np.array(logg_mist),
                xerr=np.array(sigma_teff_mist)
                if self._posteriorBool
                else None,
                yerr=np.array(sigma_logg_mist)
                if self._posteriorBool
                else None,
                node=self.evolutionarymodel._kiel_node,
                style={
                    "color": self.MIST_PLOT_COLORS["mist_point"],
                    "marker": "d",
                    "zorder": 3,
                },
            )
        )

        star = self.system.star
        # The fitted point's error bars are its posterior interval, as
        # ``(2, 1)`` arrays for errorbar's asymmetric form.
        sigma_teff_fit = [
            [v] for v in self._fit_error_bars(star.teff, star_index)
        ]
        sigma_logg_fit = [
            [v] for v in self._fit_error_bars(star.logg, star_index)
        ]

        traces.append(
            Trace(
                name=f"Star {star_name} fit value",
                role="data",
                kind="scatter",
                x=np.array(teff_fit),
                y=np.array(logg_fit),
                xerr=np.array(sigma_teff_fit) if self._posteriorBool else None,
                yerr=np.array(sigma_logg_fit) if self._posteriorBool else None,
                style={
                    "color": self.MIST_PLOT_COLORS["fit_point"],
                    "marker": "o",
                    "zorder": 4,
                },
            )
        )

        # The Teff axis follows the logg window and the EEP cut rather than
        # the raw extent of the drawn arcs.  A main-sequence star's track
        # runs on down the red giant branch to ~3000 K at logg < 3, which is
        # off the top of the y window and clipped away -- but matplotlib's
        # (and plotly's) autoscale still counts those points, which left
        # more than half of a HAT-P-3 panel empty.
        # This draw's on-chart extent, padded, is the Chart's own axis, so a
        # single Chart (the GUI's live point, a pre-flight plot) stands
        # alone; `kiel_spec_groups` reads the raw extent back out of meta
        # to widen a spaghetti figure's axis to the union over draws.
        visible_teff_flat = np.concatenate(
            [np.asarray(t, dtype=float).ravel() for t in visible_teff]
        )
        finite_teff = visible_teff_flat[np.isfinite(visible_teff_flat)]
        extent = (
            [float(finite_teff.min()), float(finite_teff.max())]
            if finite_teff.size
            else None
        )
        x_range = (
            _padded_range(extent, self.KIEL_X_PAD_FRAC)
            if extent is not None
            else None
        )

        meta = {
            # The system summary figure's Kiel panel: the window frames the
            # drawn track's main sequence, and round reference ages are
            # marked along it (summary_track_marks).  Its posterior contours
            # and the star's reported point come from
            # EvolutionaryModel.summary_posterior.
            "summary": {
                "layout": "track",
                **self.summary_track_marks(track_rows),
            },
            # Per star, like every other component's per-instance tag: with
            # one tag for all stars the last star's PDF overwrote the rest.
            "file_tag": f"kiel_{star_name}",
            "figsize": (6.5, 5.5),
            "caption": (
                f"Kiel diagram for star {latex_escape(star_name)}. The lines "
                "are the MIST evolutionary tracks interpolated at the "
                "initial mass and metallicity of each plotted "
                "posterior draw, the diamond is the MIST model prediction "
                "at the median equivalent evolutionary point, and "
                r"the circle is the median $T_{\rm eff}$ and "
                r"$\log{g}$ predicted by the full global model."
            ),
        }
        if extent is not None:
            meta[self.TEFF_EXTENT] = extent

        return [
            Chart(
                id=f"{self.evolutionarymodel.prefix}.kiel.star.{star_name}",
                component={
                    "yaml_key": self.evolutionarymodel.yaml_key,
                    "instance": None,
                },
                title="MIST evolutionary tracks",
                xlabel=r"$T_{\rm eff}$ (K)",
                ylabel=r"$\log{g}$ (cgs)",
                traces=traces,
                param_deps=self.evolutionarymodel._model_trace_param_deps(
                    self.evolutionarymodel._kiel_node, self.system
                ),
                # Ascending [lo, hi]; the *_inverted flags reverse them, so the
                # drawn axes run Teff decreasing rightward and logg 5.0
                # (dwarfs, bottom) to 3.0 (giants, top).  These six fields were
                # promoted out of `meta` (review 4.11.3): both renderers must
                # consult them to lay an axis out at all.
                x_range=list(x_range) if x_range is not None else None,
                y_range=list(logg_window),
                # Both axes reversed: the Kiel-diagram convention, with Teff
                # decreasing rightward and surface gravity increasing downward,
                # so dwarfs sit low and giants high exactly as in an
                # observational HR diagram.
                x_inverted=True,
                y_inverted=True,
                meta=meta,
            )
        ]

    def kiel_specs(self, point):
        """The Kiel Chart of every instance at one ``point``.

        The body of ``EvolutionaryModel.plot_data``: one Chart per instance
        (per star), each with its own Teff axis, since the EEP window is per
        star (evolutionarymodel.md) and one giant must not stretch a dwarf's
        panel.  ``[]`` without a point, as the base contract says for a
        component with no observations of its own.
        """
        if point is None:
            return []
        specs = []
        for star_idx in range(self.evolutionarymodel.n_elements):
            specs.extend(self._plot_kiel_trace(star_idx, point))
        return specs

    def kiel_spec_groups(self):
        """``plot_via_specs``'s group list, plus one shared Teff axis per star.

        One group per point (posterior draw), each holding every star's
        Chart, for ``plotrender.render_spec_groups``.  That renderer lays
        each figure out from the FIRST group's Chart, so a Teff range
        computed from one draw would clip the other draws' tracks:
        ``_unify_x_ranges`` widens every group's range for a Chart id to
        the union over draws first.  A draw whose Charts fail is skipped
        with a warning, except the first, which supplies the marks and the
        axes -- the same tolerance as ``plot_via_specs``.
        """
        points = self.points
        if isinstance(points, dict):
            points = [points]
        if not points:
            logger.warning("No points provided for plotting.")
            return []
        spec_groups = []
        for idx, point in enumerate(points):
            try:
                spec_groups.append(self.kiel_specs(point))
            except Exception as exc:  # noqa: BLE001 - skip a bad posterior draw
                if idx == 0:
                    raise
                logger.warning(
                    "plot_data failed for draw %d of %s: %s",
                    idx,
                    getattr(
                        self.evolutionarymodel,
                        "prefix",
                        self.evolutionarymodel,
                    ),
                    exc,
                )
        self._unify_x_ranges(spec_groups)
        return spec_groups

    def _unify_x_ranges(self, spec_groups):
        """Widen each Chart id's ``x_range`` to the padded union, over
        draws, of the on-chart Teff extents recorded in ``meta[TEFF_EXTENT]``.

        ``_plot_kiel_trace`` already pads its OWN draw's extent into
        ``x_range``, so a single Chart (the GUI's live point, a pre-flight
        plot) is complete by itself; this pass is what a spaghetti figure
        adds.  The private meta key is removed afterwards so it reaches
        neither the GUI nor the draft.
        """
        extents = {}
        for group in spec_groups:
            for spec in group:
                ext = (spec.meta or {}).get(self.TEFF_EXTENT)
                if ext is None:
                    continue
                lo, hi = extents.get(spec.id, (np.inf, -np.inf))
                extents[spec.id] = (min(lo, ext[0]), max(hi, ext[1]))
        for group in spec_groups:
            for spec in group:
                if spec.meta:
                    spec.meta.pop(self.TEFF_EXTENT, None)
                if spec.id in extents and spec.x_range is not None:
                    x_range = _padded_range(
                        list(extents[spec.id]), self.KIEL_X_PAD_FRAC
                    )
                    if x_range is not None:
                        spec.x_range = list(x_range)

    ################## contour plot #################
    ######## will only trigger after sampling #######

    def _compiled_values_at(self, points):
        """The compiled Kiel node, ``(n_instances, len(KIEL_COLUMNS))``, at
        each of ``points`` (dicts of INTERNAL values keyed by label, the
        shape ``plot_data`` takes)."""
        return [
            np.atleast_2d(
                self.evolutionarymodel._compiled_kiel(
                    *self.evolutionarymodel._point_to_plot_params(
                        p, self.system
                    )
                )
            )
            for p in points
        ]

    def _get_posterior_compiled_values(self, max_draws=None):
        """The compiled Kiel node at every distributed posterior draw -- or,
        with ``max_draws``, at that many evenly spaced ones (a contour does
        not need tens of thousands of draws, and each is a compiled call).

        This is the COMBINED posterior ``System.distribute_posterior``
        attached to the Parameters, which is what the system summary figure
        wants.  The contour PDF does not use it: that follows the draws the
        caller handed ``plot`` (``_points_compiled_values``), so a per-mode
        figure contours its own mode.
        """
        ndraws = self.system.plot_params[0].posterior.values.shape[-1]
        draws = range(ndraws)
        if max_draws is not None and ndraws > max_draws:
            draws = np.unique(
                np.linspace(0, ndraws - 1, int(max_draws)).astype(int)
            )
        post_points = []
        for i in draws:
            post_point = {}
            for param in self.system.plot_params:
                post = param.posterior
                if post is not None:
                    post_point[param.label] = param.to_internal(
                        post.values[:, i]
                    )
            post_points.append(post_point)
        return self._compiled_values_at(post_points)

    def _points_compiled_values(self):
        """The compiled Kiel node at each point the caller passed to
        ``plot`` -- the draws the contour PDF describes.

        These, not the distributed posterior, because run.py's per-mode
        loop hands ``plot`` one mode's draws: contouring the Parameters'
        posterior drew the COMBINED posterior on every per-mode figure
        (review 1.8.8 (a)).
        """
        points = self.points
        if isinstance(points, dict):
            points = [points]
        return self._compiled_values_at(points)

    # The contour PDF's enclosed probabilities, widest first: a fixed 1/2/3
    # sigma ladder, the exact Gaussian widths.  Deliberately NOT the
    # run-level `reporting: {credible_interval:}`, which sets the width the
    # TABLES quote; a contour plot draws a ladder of levels, like a corner
    # plot, and its levels say which in the legend-free shading (review
    # 2.11.23).
    CONTOUR_PROBS = tuple(SIGMA_WIDTHS[n] for n in (3, 2, 1))

    def plot_contours(self, values, star_idx, filename_prefix="debug"):
        """The posterior contours of instance ``star_idx``'s fitted and MIST
        (Teff, logg) over ``values`` (``_points_compiled_values``), written
        as ``{prefix}_evolutionarymodel_contours_star_<star>.pdf``.  Returns
        the paths written."""
        star = self.system.star
        star_index = self.evolutionarymodel.star_indices[star_idx]
        star_name = star.names[star_index]

        def column(name):
            return np.array(
                [v[star_idx, self.KIEL_INDEX[name]] for v in values]
            )

        # Each density's grid is sized by its own draws' spread, as the
        # summary figure's is.  Sizing MIST's by the systematic floor put a
        # posterior much narrower than the floor into one or two cells of a
        # +/-3-floor grid, so the 99.7% and 95% levels came out equal and
        # matplotlib refused the contour.  The floors and the fit's
        # interval still frame the axes (the pads below).
        def contour(x, y):
            return Contour(
                x=x,
                y=y,
                x_err=np.std(x),
                y_err=np.std(y),
                probs=self.CONTOUR_PROBS,
            )

        fitted = contour(column("teff_fit"), column("logg_fit"))
        mist = contour(column("teff_mist"), column("logg_mist"))
        # The larger side of the fitted point's posterior interval, per star
        # (see _summary_at for why this is not `.summary.err_minus`), and
        # the largest systematic floor over the draws.
        fit_pad = (
            max(self._fit_error_bars(star.teff, star_index)),
            max(self._fit_error_bars(star.logg, star_index)),
        )
        mist_pad = (
            np.max(np.abs(column("sigma_teff_mist"))),
            np.max(np.abs(column("sigma_logg_mist"))),
        )

        # Created before the try, so a KDE or save failure inside it is the
        # exception the caller sees -- with the figure made inside, a failure
        # before the assignment raised UnboundLocalError from `finally`
        # instead (review 1.8.8 (b)).
        fig, ax = plt.subplots(figsize=(7, 6))
        try:
            _draw_contours(
                ax,
                [
                    (
                        mist,
                        self.MIST_PLOT_COLORS["mist_contour"],
                        "MIST Model Parameters",
                        mist_pad,
                    ),
                    (
                        fitted,
                        self.MIST_PLOT_COLORS["fit_contour"],
                        "Model Parameters",
                        fit_pad,
                    ),
                ],
            )
            fig.tight_layout()
            tag = f"{self.evolutionarymodel.prefix}.contours.star.{star_name}"
            tag = tag.replace(".", "_")
            path = f"{filename_prefix}_{tag}.pdf"
            fig.savefig(path)
        finally:
            plt.close(fig)
        return [path]


# Shading of the contour PDF's filled levels, widest to narrowest.
_CONTOUR_FILL_ALPHAS = (0.25, 0.5, 0.75)


def _draw_contours(ax, entries):
    """Draw each ``(Contour, color, label, (x_pad, y_pad))`` on a Kiel
    axis: the draws as faint points, the filled and outlined density
    levels, axes framed at 3 x the largest pad past the draws and both
    reversed (the Kiel convention).

    Lives here, not in ``outputs/contour_plot``: the labels and the
    inverted axes are this diagram's, and ``Contour`` itself is shared with
    the summary figure (review 4.11.7 (b)).
    """
    xs, ys, x_errs, y_errs = [], [], [], []
    for g, color, label, (x_pad, y_pad) in entries:
        if len(g.levels) != len(_CONTOUR_FILL_ALPHAS):
            raise ValueError(
                f"_draw_contours: {label} has {len(g.levels)} levels, "
                f"{len(_CONTOUR_FILL_ALPHAS)} fill shades."
            )
        ax.scatter(
            g.x, g.y, s=35, color=to_rgba(color, 0.05), label=label, zorder=3
        )
        shades = [to_rgba(color, alpha=a) for a in _CONTOUR_FILL_ALPHAS]
        ax.contourf(
            g.X, g.Y, g.Z, levels=g.levels + [g.Z.max()], colors=shades
        )
        ax.contour(g.X, g.Y, g.Z, levels=g.levels, colors=shades, linewidths=2)
        xs.append(g.x)
        ys.append(g.y)
        x_errs.append(x_pad)
        y_errs.append(y_pad)

    x, y = np.concatenate(xs), np.concatenate(ys)
    ax.set_xlim(x.min() - 3 * max(x_errs), x.max() + 3 * max(x_errs))
    ax.set_ylim(y.min() - 3 * max(y_errs), y.max() + 3 * max(y_errs))
    ax.set_xlabel(r"$T_{\rm{eff}}$ (K)")
    ax.set_ylabel(r"$\log g$ (cgs)")
    leg = ax.legend(loc="upper left")
    for handle in leg.legend_handles:
        handle.set_alpha(0.8)
    ax.invert_xaxis()
    ax.invert_yaxis()
