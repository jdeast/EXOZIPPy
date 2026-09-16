# general imports
import logging
logger = logging.getLogger(__name__)

import numpy as np
import matplotlib.pyplot as plt

# exozippy local imports
from exozippy.outputs.plot_helper_functions import _extend_window, _padded_range
from exozippy.plotrender import _draw_data, _draw_model, _draw_residual, _apply_axes
from exozippy.outputs.contour_plot import Contour, plot_contours


class MISTPlot:

    # The EEP window the seed search and the Kiel diagram treat as "the part of a
    # track a reader of this chart came for": 202 is MIST's zero-age main sequence
    # and 630 is well up the red giant branch.  It is a PLOTTING/SEEDING window,
    # never a bound -- the fit is free to sit anywhere the grid covers, and where
    # it does the window is widened rather than hiding the star it describes.
    KIEL_EEP_WINDOW = (202.0, 630.0)

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
        self.evolutionarymodel = self.system.active_components["evolutionarymodel"]
        self._plot_x_range = {
            "start": None,
            "post": None,
            }
        # Parameter.posterior is None until System.distribute_posterior attaches
        # the chain (an xarray DataArray); no posterior means this is the start.
        self._posteriorBool = getattr(self.system.star.teff, "posterior", None) is not None


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
        cached = getattr(self.evolutionarymodel, "_reported_kiel_cache", None)
        if cached is not None:
            return cached
        if getattr(self.evolutionarymodel, "_compiled_kiel", None) is None:
            return None

        point = {}
        for param in getattr(self.system, "plot_params", []):
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
            self.evolutionarymodel._compiled_kiel(*self.evolutionarymodel._point_to_plot_params(point, self.system))
        )
        return self.evolutionarymodel._reported_kiel_cache


    def _track_curve(self, i, logmass, initfeh, eep_window):
        """(teff, logg) along instance ``i``'s track, over ``eep_window``.

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
        from .mist_grid import OUTPUT_INDEX, interpolate_track

        grid = self.evolutionarymodel._grids[i]
        track = interpolate_track(grid, logmass, initfeh)
        # EEP is the grid AXIS the track's rows are indexed by, not one of
        # the interpolated OUTPUT_COLUMNS, and interpolate_track leaves that
        # axis alone -- so grid["eep_pts"] is row-aligned with `track` by
        # construction.
        eep = np.asarray(grid["eep_pts"], dtype=float)
        teff = track[:, OUTPUT_INDEX["teff_mist"]]
        radius = track[:, OUTPUT_INDEX["radius_mist"]]
        logg = LOGG_CONST + logmass - 2.0 * np.log10(
            np.where(radius > 0, radius, np.nan)
        )

        keep = (
            (track[:, OUTPUT_INDEX["here_be_dragons"]] <= 0)
            & np.isfinite(teff)
            & (radius > 0)
            & (eep >= eep_window[0])
            & (eep <= eep_window[1])
        )

        return teff[keep], logg[keep]


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

        if point is None or getattr(self.evolutionarymodel, "_compiled_kiel", None) is None:
            return []

        kiel = np.atleast_2d(
            self.evolutionarymodel._compiled_kiel(*self.evolutionarymodel._point_to_plot_params(point, self.system))
        )
        marks = self._reported_kiel()
        if marks is None or not self._posteriorBool: # marks is None when there is no posterior attached
            marks = kiel

        # One y axis, so the logg window is the union over stars.  The fit
        # point's systematic bars count: a mark whose error bar runs off the
        # chart reads as a mark nobody bothered to fit inside it.
        logg_margin = self.KIEL_WINDOW_MARGIN_FRAC * (
            self.KIEL_LOGG_WINDOW[1] - self.KIEL_LOGG_WINDOW[0]
        )
        logg_fit = np.array([marks[star_idx, self.KIEL_INDEX["logg_fit"]]])
        logg_mist = np.array([marks[star_idx, self.KIEL_INDEX["logg_mist"]]])
        sigma_logg_mist = np.array([np.abs(marks[star_idx, self.KIEL_INDEX["sigma_logg_mist"]])])
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
        sigma_teff_mist = np.abs(marks[star_idx, self.KIEL_INDEX["sigma_teff_mist"]])
        visible_teff = [
            teff_fit,
            teff_mist - sigma_teff_mist,
            teff_mist + sigma_teff_mist,
        ]

        # initialize the list of traces to be returned
        traces = []

        star_name = self.system.star.names[star_idx]
        # grab the logmass and initfeh for the draw
        logmass = float(kiel[star_idx, self.KIEL_INDEX["logmass"]]) 
        initfeh = float(kiel[star_idx, self.KIEL_INDEX["initfeh"]])

        eep = marks[star_idx, self.KIEL_INDEX["eep"]]
        eep_window = _extend_window(
            self.KIEL_EEP_WINDOW,
            [eep],
            eep_margin,
        )
        teff_track, logg_track = self._track_curve(
            star_idx, logmass, initfeh, eep_window
        )
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
                style={"color": self.MIST_PLOT_COLORS["mist_track"], "lw": 1.0, "legend": True},
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
                xerr=np.array(sigma_teff_mist) if self._posteriorBool else None,
                yerr=np.array(sigma_logg_mist) if self._posteriorBool else None,
                node=self.evolutionarymodel._kiel_node,
                style={"color": self.MIST_PLOT_COLORS["mist_point"], "marker": "d", "zorder": 3},
            )
        )
        
        star = self.system.star
        # grab the error bars for the fitted point
        sigma_teff_fit = [
            [star.teff.summary.err_minus if hasattr(star.teff.summary, "err_minus") else 0.0],
            [star.teff.summary.err_plus if hasattr(star.teff.summary, "err_plus") else 0.0],
            ]

        sigma_logg_fit = [
            [star.logg.summary.err_minus if hasattr(star.logg.summary, "err_minus") else 0.0],
            [star.logg.summary.err_plus if hasattr(star.logg.summary, "err_plus") else 0.0],
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
                style={"color": self.MIST_PLOT_COLORS["fit_point"], "marker": "o", "zorder": 4},
            )
        )

        # The Teff axis follows the logg window and the EEP cut rather than
        # the raw extent of the drawn arcs.  A main-sequence star's track
        # runs on down the red giant branch to ~3000 K at logg < 3, which is
        # off the top of the y window and clipped away -- but matplotlib's
        # (and plotly's) autoscale still counts those points, which left
        # more than half of a HAT-P-3 panel empty.
        # only care about high and low values of teff that are finite and on the chart
        # want to keep track and update the x_range for the plot, so we can set it in the meta data
        max_x_range = self._plot_x_range.get("post" if self._posteriorBool else "start", None)
        visible_teff_flat = np.concatenate(
            [np.asarray(t, dtype=float).ravel() for t in visible_teff]
        )
        current_x_min = np.min(visible_teff_flat)
        current_x_max = np.max(visible_teff_flat)
        # update the max_x_range if the current values are outside of it
        if max_x_range is not None:
            if current_x_min < max_x_range[0]:
                max_x_range[0] = current_x_min
            if current_x_max > max_x_range[1]:
                max_x_range[1] = current_x_max
        else:
            # set the max_x_range to the first set of values if it is None
            max_x_range = [current_x_min, current_x_max]
        self._plot_x_range["post" if self._posteriorBool else "start"] = max_x_range

        x_range = _padded_range(max_x_range, self.KIEL_X_PAD_FRAC)

        meta = {
            "file_tag": "kiel",
            "figsize": (6.5, 5.5),
            "caption": (
                "Kiel diagram for the modeled star. The lines are "
                "the MIST evolutionary tracks interpolated at the "
                "initial mass and metallicity of each plotted "
                "posterior draw, the diamond is the MIST model prediction "
                "at the median equivalent evolutionary point, and "
                r"the circle is the median $T_{\rm eff}$ and "
                r"$\log{g}$ predicted by the full global model."
            ),
        }

        star_name = self.system.star.names[self.evolutionarymodel.star_indices[star_idx]]

        return [
            Chart(
                id=f"{self.evolutionarymodel.prefix}.kiel.star.{star_name}",
                component={"yaml_key": self.evolutionarymodel.yaml_key, "instance": None},
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

    def _get_kiel_trace_spec_groups(self, star_idx):

        if isinstance(self.points, dict):
            self.points = [self.points]
        if not self.points:
            logger.warning("No points provided for plotting.")
            return []
    
        spec_groups = []
        for idx, point in enumerate(self.points):
            try:
                spec_groups.append(self._plot_kiel_trace(star_idx, point))
            except Exception as exc:  # noqa: BLE001 - skip a bad posterior draw
                if idx == 0:
                    raise
                logger.warning(
                    "plot_data failed for draw %d of %s: %s",
                    idx,
                    getattr(self.evolutionarymodel, "prefix", self.evolutionarymodel),
                    exc,
                )
    
        # Before rendering, widen every draw's x_range to the union computed
        # across all of them, so the spaghetti shares one axis.  x_range is a
        # first-class Chart field now, not a meta key (review 4.11.3).
        # Same key _plot_kiel_trace wrote to -- "start" when no posterior exists.
        max_x_range = self._plot_x_range.get("post" if self._posteriorBool else "start")
        if max_x_range is not None:
            x_range = _padded_range(max_x_range, self.KIEL_X_PAD_FRAC)
            if x_range is not None:
                for spec_group in spec_groups:
                    for spec in spec_group:
                        if spec.x_range is not None:
                            spec.x_range = list(x_range)

        return spec_groups


    # almost identical to its sister function in plotrender
    # but added changes to how legend is generated
    def _kiel_render_spec_groups(self, spec_groups, filename_prefix="debug"):
        """Render one figure per spec, overlaying model traces from every group.

        Parameters
        ----------
        spec_groups : list[list[Chart]]
            One ``plot_data`` result per posterior point.  The FIRST group is the
            reference: it supplies the data traces, labels, and decorations
            (matching the historical convention that data offsets/cleaning use
            ``points[0]``).  Later groups contribute only their model traces.
            The reference group is NOT privileged in the model layer: with more
            than one group every model trace is drawn at the same spaghetti
            alpha, the reference's included.
        filename_prefix : str
            Output files are ``{filename_prefix}_{file_tag}.pdf``.
        """
        if not spec_groups:
            return []
        ref_specs = spec_groups[0]
        # Model spaghetti from the later groups, matched to the reference spec by
        # id (a draw whose plot_data failed simply contributes nothing).
        extra_models = {}
        for group in spec_groups[1:]:
            for spec in group:
                models = [t for t in spec.traces if t.role == "model"]
                if models:
                    extra_models.setdefault(spec.id, []).append(models)

        # One alpha for the whole figure, applied to the reference group's model
        # traces below as well as to extra_models -- see the module docstring.
        model_alpha = 0.8 if len(spec_groups) == 1 else 0.1
        written = []
        for spec in ref_specs:
            meta = spec.meta or {}
            fig, ax = plt.subplots(figsize=tuple(meta.get("figsize") or (10, 6)))
            try:
                for trace in spec.traces:
                    if trace.role == "model":
                        _draw_model(ax, trace, model_alpha)
                    elif trace.role == "residual":
                        _draw_residual(ax, trace)
                    else:
                        _draw_data(ax, trace)
                for models in extra_models.get(spec.id, []):
                    for trace in models:
                        _draw_model(ax, trace, model_alpha)

                _apply_axes(ax, spec)
                ax.set_xlabel(spec.xlabel)
                ax.set_ylabel(spec.ylabel)
                ax.set_title(spec.title)
                # De-duplicate by LABEL: a spaghetti figure overlays the same
                # named model trace once per draw (50 identical "A MIST track"
                # rows would otherwise swamp the panel), and a component may
                # legitimately name the same series in more than one trace.
                # First occurrence wins, so the legend keeps spec order.
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    unique = {}
                    for handle, label in zip(handles, labels):
                        unique.setdefault(label, handle)
                    leg = ax.legend(
                        list(unique.values()),
                        list(unique.keys()),
                        loc="best",
                        fontsize="small",
                    )
                    # get_legend_handles_labels returns the artists actually
                    # drawn on the axes, so set_alpha on those changes the
                    # plotted lines.  leg.legend_handles are the legend's own
                    # proxy copies: raising alpha there touches only the legend.
                    for handle, label in zip(leg.legend_handles, unique.keys()):
                        if "track" in label:
                            handle.set_alpha(0.8)

                fig.tight_layout()

                tag = meta.get("file_tag") or spec.id.replace(".", "_")
                path = f"{filename_prefix}_{tag}.pdf"
                fig.savefig(path)
                written.append(path)
            finally:
                plt.close(fig)
        return written


    def plot_kiel_diagram(self, star_idx, filename_prefix="debug"):

        spec_groups_one_star = self._get_kiel_trace_spec_groups(star_idx)
        self._kiel_render_spec_groups(spec_groups_one_star, filename_prefix)


    ################## contour plot #################
    ######## will only trigger after sampling #######

    def _get_posterior_compiled_values(self):

        ndraws = self.system.plot_params[0].posterior.values.shape[:][-1] # should take shape (nstars, ndraws)
        post_points = []
        for i in range(ndraws):
            post_point = {}
            for param in getattr(self.system, "plot_params", []):
                post = getattr(param, "posterior", None)
                values = getattr(post, "values", None)
                if values is not None:
                    post_point[param.label] = param.to_internal(values[:, i])

            post_points.append(post_point)

        values = [np.atleast_2d(
            self.evolutionarymodel._compiled_kiel(*self.evolutionarymodel._point_to_plot_params(p, self.system))
        ) for p in post_points]

        return values


    def plot_contours(self, values, star_idx, filename_prefix="debug"):

        star = self.system.star
        star_name = star.names[self.evolutionarymodel.star_indices[star_idx]]

        # grab the error bars for the fitted point
        # summary values should take the shape of (nstars,)
        # which simplifies to a shape of () when nstars=1
        # so we need make sure the resulting value grabbed is always at least 1-d
        sigma_teff_fit = np.max([
            np.atleast_1d(star.teff.summary.err_minus)[star_idx], 
            np.atleast_1d(star.teff.summary.err_plus)[star_idx]
            ])
        sigma_logg_fit = np.max([
            np.atleast_1d(star.logg.summary.err_minus)[star_idx], 
            np.atleast_1d(star.logg.summary.err_plus)[star_idx]
            ])

        # global fit values
        teff_fit = np.array([v[star_idx, self.KIEL_INDEX["teff_fit"]] for v in values]).T
        logg_fit = np.array([v[star_idx, self.KIEL_INDEX["logg_fit"]] for v in values]).T

        fitted_contours = Contour(x=teff_fit, y=logg_fit, 
                                x_err=sigma_teff_fit, y_err=sigma_logg_fit, 
                                color=self.MIST_PLOT_COLORS["fit_contour"], 
                                label="Model Parameters")


        # MIST model parameters
        sigma_teff_mist = np.max(np.abs([v[star_idx, self.KIEL_INDEX["sigma_teff_mist"]] for v in values]))
        sigma_logg_mist = np.max(np.abs([v[star_idx, self.KIEL_INDEX["sigma_logg_mist"]] for v in values]))

        # mist values
        teff_mist = np.array([v[star_idx, self.KIEL_INDEX["teff_mist"]] for v in values]).T
        logg_mist = np.array([v[star_idx, self.KIEL_INDEX["logg_mist"]] for v in values]).T

        mist_contours = Contour(x=teff_mist, y=logg_mist, 
                                x_err=sigma_teff_mist, y_err=sigma_logg_mist, 
                                color=self.MIST_PLOT_COLORS["mist_contour"], 
                                label="MIST Model Parameters")

        written = []

        try:
            fig, ax = plot_contours([mist_contours, fitted_contours])

            fig.tight_layout()
            id=f"{self.evolutionarymodel.prefix}.contours.star.{star_name}"
            tag = id.replace(".", "_")
            path = f"{filename_prefix}_{tag}.pdf"
            fig.savefig(path)
            written.append(path)

        finally:
            plt.close(fig)

        return written