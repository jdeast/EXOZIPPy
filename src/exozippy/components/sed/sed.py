# generic imports
from pathlib import Path
import yaml
import ast
import importlib

# scientific imports
import numpy as np
import pandas as pd

# astronomy imports
import astropy.units as u

# pymc imports
import pymc as pm
import pytensor.tensor as pt
import pytensor

# local imports
from exozippy.components.parameter import Parameter
from exozippy.components.component import Component
from exozippy.constants import LOGG_CONST, ANG_TO_MICRON_CONST
# this import is required even though it's not used explicitly
# it registers all the mathematical relations
from . import physics
from .physics import *
from .bc_grid import (
    build_bc_grid,
    peek_grid_axes,
    slice_bc,
    RegularGridInterpolator,
    DEFAULT_BC_ROOT,
)
from ..celestial_body.physics import calc_logg
from ..star.physics import calc_luminosity

# debugging imports
import ipdb

# plotting imports
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple

try:
    current_dir = Path(__file__).parent
except NameError:
    current_dir = Path.cwd()

class SED(Component):
    """
    Spectral Energy Distribution likelihood driven by a precomputed
    grid of bolometric corrections (BCs).

    Architecture mirrors components/rv_instrument/rv_instrument.py:
      * __init__ pulls per-element metadata (file, model, staridx).
      * load_data parses the .sed file(s) and builds a pytensor BC
        interpolator per SED over (lgTeff, logg, feh, Av).
      * build_map pushes any int8 mapping tensors onto self.
      * build_dependent_parameters is a no-op in v1 — the SED
        component relies on the star.sedfile hint to have already
        produced teffsed/radiussed/av/distance/fbolsed etc.
      * build_likelihood computes predicted apparent magnitudes via
        m_pred = m_bol - BC, then adds a Normal likelihood against
        the observed .sed mags with the sampled errscale.

    Notes / v1 choices
    ------------------
    * One Component instance may hold multiple SEDs, each tied to a
      star via `staridx`. This matches rv_instrument's pattern of
      "one component that knows about N instruments".
    * Each SED may request its own filter set and its own `model`
      (e.g. "NextGen"); load_data builds an independent
      RegularGridInterpolator per SED.
    * logg for the BC interpolation is reconstructed from
      (star.mass, star.radiussed) inside build_likelihood, matching
      the original Grid.GRID_VARS_CALC['star.logg'] behavior rather
      than the star component's star.logg (which uses star.radius).
    * Rv is assumed fixed at the grid's value (3.1 for NextGen) in
      v1. Promote to a parameter if/when the science calls for it.
    """

    # factory.discover_components() maps a Component subclass to a
    # YAML key by lowercasing the class name. "SED" -> "sed" is what
    # we want here, so no yaml_key override needed.

    def __init__(self, config, config_manager):

        # 1. Initialize the base Component
        # sets self.config and self.config_manager
        # SED is a singleton component, but Component.__init__ expects a
        # list of dicts for its n_elements / names bookkeeping (enumerate +
        # .get("name", ...)). Wrap a bare dict into a 1-element list for
        # the base, then restore the dict shape the rest of SED expects
        # for its self.config.get(...) access pattern below.
        base_cfg = (
            [{"name": "sed", **config}] if isinstance(config, dict) else config
        )
        super().__init__(base_cfg, config_manager)
        self.config = config if isinstance(config, dict) else config[0]

        self.label = "SED Parameters"
        self.bc_root = Path(self.config.get("bc_root", DEFAULT_BC_ROOT))

        # for now lets assume only one SED file
        self.sedfile = self.config.get("file")
        # read in sed file to get model
        with open(self.sedfile, 'r') as f:
            self.SED_yaml = yaml.safe_load(f)
        self.sedmodel = self.SED_yaml.get('model', 'NextGen')
        
        self.teffsedfloor = self.config.get("teffsedfloor", 0.020)
        self.fbolsedfloor = self.config.get("fbolsedfloor", 0.024)

        # Data buckets — filled by load_data.
        self.user_filters = []
        self.mist_filters = []
        self.mags = np.array([])
        self.errs = np.array([])
        self.bc_grid_data = [None]
        self.bc_interpolator = [None]

        # Grid-axis caches, filled by _inject_grid_bounds below. These
        # are kept on the component so build_likelihood can consult
        # the (logg_min, logg_max) range when adding a soft potential
        # on the inline loggsed expression (loggsed isn't a named
        # star Parameter, so it can't be bounded via user_params the
        # way teffsed/feh/av are).
        self.grid_axes = [None]
        self._inject_grid_bounds()

    # ------------------------------------------------------------------
    # (pre-phase) Coordinate star parameter bounds with the BC grid.
    #
    # The BC interpolator can't extrapolate meaningfully beyond its
    # grid in ex: (teff, logg, feh, av). Letting the sampler wander
    # outside those ranges produces NaNs or garbage. Rather than push
    # that responsibility onto the user's YAML, we read the grid axes
    # here and inject them as overrides into config_manager.user_params
    # BEFORE any component's build_parameters() runs.
    #
    # Why this works timing-wise:
    #   System.__init__ instantiates every Component in one loop, then
    #   System.build_model() calls build_parameters() for every
    #   component in a second loop. So any mutation of
    #   config_manager.user_params from SED.__init__ is visible by the
    #   time star.build_parameters() calls config_manager.resolve().
    #
    # Why the bound-tightening semantics are safe:
    #   ConfigManager.resolve() applies `max` to competing lower bounds
    #   and `min` to competing upper bounds. So:
    #     * user silent  + grid bound  -> grid bound wins  ✓
    #     * user tighter + grid bound  -> user wins        ✓
    #     * user looser  + grid bound  -> grid wins        ✓ (safe,
    #       sampler is kept inside the physically valid region)
    #
    # Future refinement (see README): the cleaner long-term design is
    # a `contribute_param_overrides(config_manager)` hook on Component
    # called explicitly before build_parameters in System.build_model.
    # That requires touching component.py/system.py/config.py, which
    # is deliberately out of scope for this scaffold — the __init__
    # channel is functionally equivalent and zero-patch.
    # ------------------------------------------------------------------
    def _inject_grid_bounds(self):

        try:
            axes = peek_grid_axes(
                model=self.sedmodel, bc_root=self.bc_root
            )
        except FileNotFoundError as e:
            # If the grid isn't findable at construction time,
            # defer to load_data's own error handling rather than
            # crashing here — the user's config may be wrong but
            # that's a load-time failure, not an init failure.
            print(
                f"Warning: sed could not peek "
                f"grid axes for model={self.sedmodel} at "
                f"{self.bc_root}: {e}. Skipping bound injection."
            )

        self.grid_axes = axes

        teff_lo = float(axes["teff_pts"].min())
        teff_hi = float(axes["teff_pts"].max())
        feh_lo = float(axes["feh_pts"].min())
        feh_hi = float(axes["feh_pts"].max())
        av_lo = float(axes["av_pts"].min())
        av_hi = float(axes["av_pts"].max())

        overrides = {
            f"star.teffsed": {"lower": teff_lo, "upper": teff_hi},
            f"star.feh":     {"lower": feh_lo,  "upper": feh_hi},
            f"star.av":      {"lower": av_lo,   "upper": av_hi},
        }

        for key, bounds in overrides.items():
            existing = self.config_manager.user_params.get(key)
            if existing is None:
                self.config_manager.user_params[key] = dict(bounds)
            elif isinstance(existing, dict):
                # Don't clobber — only fill in bounds the user
                # hasn't already specified. ConfigManager.resolve
                # will tighten correctly when both are present,
                # but we still want the grid bound to appear on
                # the other side (lower *or* upper) when the user
                # only set one of them.
                for bk, bv in bounds.items():
                    existing.setdefault(bk, bv)
            else:
                # User supplied a scalar initval shorthand; promote
                # to dict and tack on the grid bounds.
                self.config_manager.user_params[key] = {
                    "initval": existing,
                    **bounds,
                }
    
    @property
    def prefix(self):
        return "sed"

    # ------------------------------------------------------------------
    # 1) build_parameters — just the SED-specific errscale.
    # ------------------------------------------------------------------
    def build_parameters(self, model):
        
        # in future could forsee doing per facility error scaling
        parameters = {
            "errscale": None,
        }
        self.build_pars_from_dict(
            parameters, shape=(1,)#, prefix=self._prefix
        )

    # ------------------------------------------------------------------
    # 2) load_data — parse .sed file and build one BC interpolator
    # ------------------------------------------------------------------
    def _process_SED_yaml(self):

        ALLOWED_PHOTTYPES = ["blend", "diff"]

        # read in nstars and set default to 1 star
        self.nstars = self.SED_yaml.get("nstars", 1)
        filter_dict = self.SED_yaml.get("filters")
        self.nfilters = len(filter_dict)

        # per filter information
        self.filters = [c.get("name") for c in filter_dict]
        self.magsys = [c.get("magsys", "Vega") for c in filter_dict]
        self.photType_unprocessed = [c.get("photType") for c in filter_dict]

        mag = np.array([c.get("mag") for c in filter_dict])
        err = np.array([c.get("err") for c in filter_dict])
        self.mag = np.broadcast_to(mag[np.newaxis], (self.nstars, *mag.shape))
        self.err = np.broadcast_to(err[np.newaxis], (self.nstars, *err.shape))

        if self.nstars > 1:
            # need to determine the photometry type for each listed mag
            photType = []

            for i, phot in enumerate(self.photType_unprocessed):
                try:
                    key = phot.keys()
                    if (len(list(key)) > 1):
                        raise ValueError(
                            f"Only list one type of photometry for ``{self.filters[i]}``\n"
                            f"You can select: {', '.join(ALLOWED_PHOTTYPES)}")
                    if (list(key)[0] not in ALLOWED_PHOTTYPES):
                        raise ValueError(
                            f"``{list(key)[0]}`` not an allowed photometry for ``{self.filters[i]}``\n"
                            f"You can select: {', '.join(ALLOWED_PHOTTYPES)}")
                    photType.append(phot)
                except:
                    photType.append({"blend": list(range(self.nstars))})

        else:
            photType = [None]*self.nfilters
    

    def load_data(self):
        if self.sedfile is None:
            raise ValueError(
                f"sed is missing the required 'file' key"
            )
        
        self._process_SED_yaml()

        grid = build_bc_grid(
            user_filter_names=self.filters,
            model=self.sedmodel,
            bc_root=self.bc_root,
        )

        self.bc_grid_data = grid
        self.mist_filters = grid["filter_order"]

    # ------------------------------------------------------------------
    # 3) build_map — nothing vectorized yet. We keep staridx as a
    #    python int and look up scalar star-params in the likelihood
    #    loop. If/when multi-star fits become a thing, convert to
    #    tensor_variable here and vectorize build_likelihood.
    # ------------------------------------------------------------------
    def build_map(self, system):
        pass

    # ------------------------------------------------------------------
    # 4) build_dependent_parameters — no-op in v1. We rely on the
    #    star component having already built teffsed, radiussed, av,
    #    distance, fbolsed, luminositysed because the user set
    #    star.sedfile (which is the path we've already read).
    # ------------------------------------------------------------------
    def build_dependent_parameters(self, model, system):
        # Guard rail: fail loudly if the required star params are missing.
        star = getattr(system, "star", None)
        if star is None:
            raise RuntimeError(
                "sed component requires a 'star' component in the system"
            )
        required = [
            "teffsed",
            "radiussed",
            "feh",
            "av",
            "distance",
            "fbolsed",
            "mass",
        ]
        missing = [r for r in required if not hasattr(star, r)]
        if missing:
            raise RuntimeError(
                f"sed component requires the following star parameters "
                f"to exist but they are missing: {missing}. "
                "Make sure star.sedfile is set in your system YAML so "
                "star.build_parameters() creates them."
            )
        
        # create grid based on star/user bounds
        bc_values = self.bc_grid_data['bc_values']
        star_teff = star.teffsed
        star_feh = star.feh
        star_av = star.av
        # slice_bc expects a grid-yaml-shaped dict (see NextGen.grid.yaml),
        # not the SED yaml. Reconstitute it from bc_grid_data, which is the
        # actual source of truth for the loaded axes.
        grid_dict = {
            "model": self.sedmodel,
            "grid": {
                "teff": self.bc_grid_data["teff_pts"],
                "logg": self.bc_grid_data["logg_pts"],
                "feh":  self.bc_grid_data["feh_pts"],
                "av":   self.bc_grid_data["av_pts"],
            },
        }
        bc_slice, axes = slice_bc(grid_dict, bc_values,
                                  teff=(star_teff.lower, star_teff.upper),
                                  feh=(star_feh.lower, star_feh.upper),
                                  av=(star_av.lower, star_av.upper))

        # slice_bc only populates `axes` for constrained params (teff/feh/av
        # here). logg is unconstrained, so its full grid survives in bc_slice
        # but needs to be supplied to RegularGridInterpolator from
        # bc_grid_data. Overlay the slices on top of the full axes.
        axes_full = {
            "teff": self.bc_grid_data["teff_pts"],
            "logg": self.bc_grid_data["logg_pts"],
            "feh":  self.bc_grid_data["feh_pts"],
            "av":   self.bc_grid_data["av_pts"],
        }
        axes_full.update(axes)

        self.bc_interpolator = RegularGridInterpolator(
            points=[
                axes_full['teff'],
                axes_full['logg'],
                axes_full['feh'],
                axes_full['av'],
            ],
            values=bc_slice,
        )

    # ------------------------------------------------------------------
    # 5) build_likelihood — per-SED Normal likelihood over observed
    #    magnitudes, with a predicted mag coming from (mbol - BC).
    # ------------------------------------------------------------------
    def build_likelihood(self, model, system):
        star = system.star

        teff = star.teff.value             # K
        teffsed = star.teffsed.value       # K
        radiussed = star.radiussed.value   # R_sun
        fbol = star.fbol.value             # erg/s/cm2
        fbolsed = star.fbolsed.value       # erg/s/cm2
        logmass = star.logmass.value             # M_sun
        feh = star.feh.value               # dex
        av = star.av.value                 # mag

        # Reconstruct loggsed from logmass + radiussed (NOT radius).
        loggsed = calc_logg(logmass, radiussed) 

        # RegularGridInterpolator.evaluate expects shape (ntest, ndim).
        coords = pt.stack([teffsed, loggsed, feh, av], axis=-1)  # (nstars, 4)
        bc_vec = self.bc_interpolator.evaluate(coords)   # (nfilters,)

        # Bolometric luminosity
        Lbol = calc_luminosity(radiussed, teffsed) 
        # Absolute bolometric magnitude from bolometric luminosity.
        # M_bol = -2.5 log10(L_bol / L_bol_0) with IAU 2015 zero
        # point (see physics.L_BOL_ZERO_LSUN).
        Mbol_abs = calc_absbolmag(Lbol) 

        # Absolute magnitude in each filter.
        M_abs = calc_absmag_from_bc(Mbol_abs, bc_vec)

        # Predicted apparent magnitude in each filter using distance modulus
        distance = star.distance.value
        m_app_predicted = calc_appmag(M_abs, distance)

        # Observed data
        mag_data = pm.Data(f"sed_mag_data", self.mag)
        err_data = pm.Data(f"sed_mag_err", self.err)

        sigma = self.err * self.errscale.value

        pm.Normal(
            f"{self.prefix}.model",
            mu=m_app_predicted,
            sigma=sigma,
            observed=self.mag,
        )

        # this links the two with a user settable error floor
        self.teffsed_floor_prior = pm.Potential("sed.teffsed_floor_prior",
                                                -0.5 * ((teff - teffsed) / (teff * self.teffsedfloor)) ** 2)
        self.fbolsed_floor_prior = pm.Potential("sed.fbolsed_floor_prior",
                                                -0.5 * ((fbol - fbolsed) / (fbol * self.fbolsedfloor)) ** 2)

    # ------------------------------------------------------------------
    # 6) compile_plotters — stash the compiled pytensor functions we
    #    need to evaluate the model at arbitrary MCMC draws. For SED
    #    the "time grid" concept doesn't apply (measurements are
    #    per-filter at a single epoch), so we just need one compiled
    #    function per SED that returns predicted mags given the full
    #    parameter bundle.
    # ------------------------------------------------------------------
    def compile_plotters(self, model, system):
        star = system.star

        param_symbols = [p.value for p in system.plot_params]
        self._compiled_mag_predictors = [None]

        teffsed = star.teffsed.value
        radiussed = star.radiussed.value
        logmass = star.logmass.value
        feh = star.feh.value
        av = star.av.value

        logg_vars = ['star.logmass', 'star.radiussed']
        logg_inputs = []
        for p in system.plot_params:
            if p.label in logg_vars:
                logg_inputs.append(p.value)
        
        self._compiled_logg_calc = [None]
        loggsed_node = calc_logg(logmass, radiussed)

        # RegularGridInterpolator.evaluate expects shape (ntest, ndim).
        coords = pt.stack([teffsed, loggsed_node, feh, av], axis=-1)  # (nstars, 4)
        bc_vec = self.bc_interpolator.evaluate(coords)   # (nfilters,)

        # Bolometric luminosity
        Lbol = calc_luminosity(radiussed, teffsed)
        # Absolute bolometric magnitude from bolometric luminosity.
        # M_bol = -2.5 log10(L_bol / L_bol_0) with IAU 2015 zero
        # point (see physics.L_BOL_ZERO_LSUN).
        Mbol_abs = calc_absbolmag(Lbol)

        # Absolute magnitude in each filter.
        M_abs = calc_absmag_from_bc(Mbol_abs, bc_vec)

        # Predicted apparent magnitude in each filter using distance modulus
        distance = star.distance.value
        m_app_predicted_node = calc_appmag(M_abs, distance)

        try:
            self._compiled_mag_predictors = pytensor.function(
                inputs=param_symbols,
                outputs=m_app_predicted_node,
                on_unused_input="ignore",
            )
            self._compiled_logg_calc = pytensor.function(
                inputs=logg_inputs,
                outputs=calc_logg(*logg_inputs),
                on_unused_input="ignore",
            )
        except Exception as e:
            # Don't let plotting infra kill model build; warn
            # and leave None in place so plot() skips gracefully.
            print(
                f"Warning: SED plotter compile "
                f"failed: {e}"
            )
            self._compiled_mag_predictors = None
            self._compiled_logg_calc = None

    # ------------------------------------------------------------------
    # 7) plot — observed mag vs predicted mag per filter, per SED.
    # ------------------------------------------------------------------
    def plot(self, system, points, filename_prefix="debug"):
        if isinstance(points, dict):
            points = [points]
        if not points:
            print("SED.plot: no points provided.")
            return
        
        # retrieve model plotting class
        plot_class_path = Path(current_dir / "models" / system.sed.sedmodel / "plot.py")
        parsed_ast = ast.parse(plot_class_path.read_text())
        plot_cls_str = [node.name for node in parsed_ast.body if isinstance(node, ast.ClassDef)][0]
        mod_name = f"exozippy.components.sed.models.{system.sed.sedmodel}.plot"
        module = importlib.import_module(mod_name)
        plot_cls = getattr(module, plot_cls_str)
        plot_obj = plot_cls(system, points)

        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1, figsize=(max(6, 0.6 * plot_obj.nfilters + 2), 6),
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
        )
        plt.subplots_adjust(wspace=0, hspace=0.05)

        mag_pred_med = np.median(plot_obj.mag_pred_draws, axis=0)
        for nstar in range(plot_obj.nstars):
            # Observed fluxes
            x = plot_obj.wave_filter*ANG_TO_MICRON_CONST
            xerr = plot_obj.wave_err*ANG_TO_MICRON_CONST
            
            y = np.log10(plot_obj.flux_obs[nstar]*plot_obj.wave_filter)
            y_limits_from_err = np.log10(plot_obj.f_limits_from_err[nstar]*plot_obj.wave_filter)

            # initialize yerr array
            yerr = np.zeros(np.shape(y_limits_from_err))
            yerr[0] = y - y_limits_from_err[0]
            yerr[1] = y_limits_from_err[1] - y
            
            ax_top.errorbar(
                x, y, xerr=xerr, yerr=yerr, fmt='',
                color=plot_obj.colors_obs[nstar], capsize=3, linestyle='None',
                zorder=3
            )

            ax_top.scatter(x, y, color=plot_obj.colors_obs[nstar], 
                        marker=plot_obj.markers[nstar], zorder=3)

            for d, draw in enumerate(plot_obj.draws):
                # model spectra
                alpha_spec = 0.7 if plot_obj.ndraws == 1 else 0.15
                x_spec = plot_obj.df_wave['wavelength_micron']
                y_spec = np.log10(plot_obj.flux_model_draws[d][nstar]*plot_obj.df_wave['wavelength_angstrom'])
                ax_top.plot(x_spec, 
                            y_spec, 
                            ls=plot_obj.linetypes[nstar], 
                            color=plot_obj.colors_spec[nstar], 
                            alpha=alpha_spec, 
                            label=f"Star {plot_obj.star_names[nstar]}")


            residual = plot_obj.mag_obs[nstar] - mag_pred_med[nstar]
            alpha_res = 1.0 if plot_obj.nstars == 1 else 0.5
            
            ax_bot.errorbar(
                x, residual, yerr=plot_obj.mag_obs_err[nstar], capsize=3,
                fmt=plot_obj.markers[nstar], color=plot_obj.colors_obs[nstar],
                alpha=alpha_res
            )

        ax_top.set_xscale("log")

        # set y limits dynamically
        plot_obj._get_ylim()
        ax_top.set_ylim(plot_obj.y_lower, plot_obj.y_upper)
        ax_top.set_ylabel(r"log $\lambda F_\lambda$ (erg s$^{-1}$ cm$^{-2}$)")

        # Two artists per star: translucent line + opaque marker
        legend_handles = []
        for nstar in range(plot_obj.nstars):
            line = Line2D([0], [0], color=plot_obj.colors_spec[nstar], 
                            linestyle=plot_obj.linetypes[nstar],
                            alpha=0.7, 
                            label=f"Star {plot_obj.star_names[nstar]}")
            marker = Line2D([0], [0], color='none', 
                            marker=plot_obj.markers[nstar],
                            markerfacecolor=plot_obj.colors_obs[nstar], 
                            markeredgecolor=plot_obj.colors_obs[nstar],
                            alpha=1.0)
            legend_handles.append((line, marker))

        labels = [f"Star {plot_obj.star_names[nstar]}" for nstar in range(plot_obj.nstars)]

        ax_top.legend(
            handles=legend_handles,
            labels=labels,
            handler_map={tuple: HandlerTuple(ndivide=None, pad=-5)},
            fontsize="small", handlelength=5.0
        )

        ax_bot.set_xscale("log")
        ax_bot.set_xlim(5e-2, 30)
        ax_bot.set_ylabel("Residuals (mag)")
        ax_bot.set_xlabel(r"$\lambda$ ($\mu$m)")
        # horizontal line for match between obs and pred
        ax_bot.axhline(0.0, color="#001219", lw=1, zorder=1, ls='dashed')

        plt.tight_layout()
        pdf_path = f"{filename_prefix}_SED.pdf"
        plt.savefig(pdf_path)
        plt.close(fig) 