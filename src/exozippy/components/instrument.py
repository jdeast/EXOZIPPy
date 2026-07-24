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
"""

import numpy as np
import pytensor.tensor as pt

from .component import Component


class Instrument(Component):
    # Noise parameterization: "jitter_variance" (additive) or "err_scale"
    # (multiplicative).  Subclasses override; the default matches the majority
    # (rv/transit/astrometry).
    noise_model = "jitter_variance"

    def __init__(self, component_config, config_manager):
        super().__init__(component_config, config_manager)
        # Every instrument reads its data from per-element files and tracks a
        # running observation count.
        self.files = [c.get("file") for c in self.config]
        self.n_total_obs = 0
        self._load_plot_styles()

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
        return -0.95 * scaled_min ** 2

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
                    f"jittervar_lower floor.")
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
        return pt.sqrt(pt.sqr(err_tensor)
                       + self.jitter_variance.value[self.inst_map_tensor])

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
                matrix[r:r + n_r, c:c + n_c] = block
            r, c = r + n_r, c + n_c
        return matrix, n_per, total
