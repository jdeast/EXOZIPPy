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
"""

import logging

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from . import gp as gp_support
from .component import Component

logger = logging.getLogger(__name__)


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

    def __init__(self, component_config, config_manager):
        super().__init__(component_config, config_manager)
        # Every instrument reads its data from per-element files and tracks a
        # running observation count.
        self.files = [c.get("file") for c in self.config]
        self.n_total_obs = 0
        self._load_plot_styles()
        self._load_gp_config()

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
        """The GP hyperparameters, declared at the root of components/defaults.yaml.

        Only the amplitudes are redeclared per component (their unit is the
        data's), so introspection needs this list to report the rest.
        Components that opt out of GP support declare none.
        """
        if not cls.supports_gp:
            return []
        return [
            name
            for kind in gp_support.GP_TERMS
            for name in gp_support.GP_TERM_PARAMS[kind]
        ]

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
        """Stage 6: the observational likelihood, with or without GPs.

        Drop-in replacement for the ``pm.Normal(name, mu, sigma, observed)``
        each child used to write.  With no ``gp:`` key anywhere it *is* that
        call, unchanged.  Otherwise the observations split by instrument: the
        non-GP files keep one shared Normal, and each GP file gets a celerite2
        marginal likelihood whose mean is the same physical model ``mu`` and
        whose diagonal is the same ``sigma`` (data error plus jitter), so the
        GP describes only what the noise model left over.

        ``mu``, ``sigma`` and ``observed`` are (n_total_obs,) tensors in the
        concatenated order the child built; the per-file sort recorded by
        ``_prepare_gp`` is applied here.
        """
        if not self.has_gp:
            return pm.Normal(name, mu=mu, sigma=sigma, observed=observed)

        if self._gp_time is None:
            raise RuntimeError(
                f"[{self.prefix}] a 'gp:' key is configured but load_data "
                f"never called _prepare_gp; the GP times are unknown."
            )

        self._build_gp_deterministics()

        gp_obs = np.zeros(self.n_total_obs, dtype=bool)
        for idx in self._gp_obs_index.values():
            gp_obs[idx] = True

        plain = np.flatnonzero(~gp_obs)
        if plain.size:
            pm.Normal(
                name,
                mu=mu[plain],
                sigma=sigma[plain],
                observed=observed[plain],
            )

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
