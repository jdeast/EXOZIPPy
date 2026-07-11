# generic imports
import logging
import urllib.request
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
    resolve_filter_name,
    facility_from_svo_name,
    _load_alias_table,
    _collect_facility_files,
)
from ..star.physics import calc_logg, calc_luminosity

# plotting imports
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple

logger = logging.getLogger(__name__)

try:
    current_dir = Path(__file__).parent
except NameError:
    current_dir = Path.cwd()

class SED(Component):
    """
    Spectral Energy Distribution likelihood driven by a precomputed
    grid of bolometric corrections (BCs).

    Architecture:
      * __init__ pulls the SED config (file, model, error floors) and
        injects the BC grid's axis limits as star-parameter bounds.
      * load_data parses the .sed file, resolves each filter row's
        photType (which stars it blends or differences) into a
        +1/0/-1 blend matrix over the system's stars, and builds one
        pytensor BC interpolator over (teff, logg, feh, Av) whose
        trailing axis spans the .sed filters plus any Band-referenced
        filters (so other components can request flux predictions).
      * build_likelihood predicts per-star apparent magnitudes via
        m_pred = m_bol - BC + distance modulus, combines them per row
        (flux sums for blends, -2.5*log10(F_pos/F_neg) for
        differential mags, following EXOFASTv2's exofast_multised),
        and adds a Normal likelihood with the sampled errscale.
      * predict_star_appmag / predict_blend_appmag expose calibrated
        flux predictions to other components (mulensing f_source
        constraint, transit deblending, astrometry fluxfrac).

    Notes
    -----
    * logg for the BC interpolation is reconstructed from
      (star.mass, star.radiussed) — NOT the star component's
      star.logg (which uses star.radius).
    * Rv is assumed fixed at the grid's value (3.1 for NextGen).
      Promote to a parameter if/when the science calls for it.
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

        # Cached (nstars, n_all_filters) predicted-mag graph node, built
        # lazily by _predicted_appmag_node and shared by the likelihood,
        # the plot compiler, and the cross-component predict_* API.
        self._m_pred_matrix = None

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
            logger.warning(
                f"SED could not peek grid axes for model={self.sedmodel} "
                f"at {self.bc_root}: {e}. Skipping bound injection."
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
    # 1) register_parameters — declare the manifest for stage 2.
    # ------------------------------------------------------------------
    def register_parameters(self, system):
        # in future could foresee doing per facility error scaling
        self.manifest = {"errscale": None}

    # ------------------------------------------------------------------
    # 2) load_data — parse .sed file and build one BC interpolator
    # ------------------------------------------------------------------
    def _process_SED_yaml(self):
        # An empty (or absent) filter list is legal: the SED component can
        # still serve cross-component flux predictions (e.g. the mulensing
        # f_source constraint) through Band-referenced filters alone.
        filter_dict = self.SED_yaml.get("filters") or []
        self.nfilters = len(filter_dict)

        # per filter information
        self.filters = [c.get("name") for c in filter_dict]
        self.magsys = [c.get("magsys", "Vega") for c in filter_dict]
        self.photType_unprocessed = [c.get("photType") for c in filter_dict]

        # One observation per filter row (a row may be the blended or
        # differential magnitude of several stars; which stars is encoded
        # in photType and resolved into self.blend_matrix by load_data).
        self.mag = np.array([c.get("mag") for c in filter_dict], dtype=float)
        self.err = np.array([c.get("err") for c in filter_dict], dtype=float)

    @staticmethod
    def _resolve_star_ref(ref, star_names):
        """Translate a photType star reference (name or index) to an index."""
        n = len(star_names)
        if isinstance(ref, bool):
            raise ValueError(f"Invalid star reference {ref!r} in photType.")
        if isinstance(ref, (int, np.integer)):
            idx = int(ref)
        elif isinstance(ref, str):
            if ref in star_names:
                idx = star_names.index(ref)
            else:
                try:
                    idx = int(ref)
                except ValueError:
                    raise ValueError(
                        f"Unknown star reference '{ref}' in photType. "
                        f"Known stars: {star_names} (or indices 0..{n - 1}).")
        else:
            raise ValueError(f"Invalid star reference {ref!r} in photType.")
        if not 0 <= idx < n:
            raise ValueError(
                f"Star index {idx} in photType out of range; the system "
                f"defines {n} star(s): {star_names}.")
        return idx

    @staticmethod
    def _combo_label(pos, neg, star_names):
        """Human-readable label for a pos/neg combination, e.g. 'A+B' or 'A-(B+C)'."""
        plus = "+".join(star_names[i] for i in pos)
        if not neg:
            return plus
        if len(pos) > 1:
            plus = f"({plus})"
        minus = "+".join(star_names[i] for i in neg)
        if len(neg) > 1:
            minus = f"({minus})"
        return f"{plus}-{minus}"

    def _build_blend_matrix(self, star_names):
        """
        Resolve each filter row's photType into a +1/0/-1 coefficient matrix
        (EXOFASTv2 exofast_multised convention), shape (nfilters, nstars).

        photType schema per filter row:
            photType:
              pos: [Lens, Source]   # stars added in flux space
              neg: [0, Companion]   # optional; makes the row differential:
                                    #   mag = -2.5*log10(F_pos / F_neg)
        `blend: [...]` is accepted as an alias for `pos` (original schema).
        Star references may be names or integer indices. A missing photType
        means the blend of all stars (the historical default).
        """
        nstars = len(star_names)
        blend_matrix = np.zeros((self.nfilters, nstars), dtype=int)
        combo_labels = []

        for i, phot in enumerate(self.photType_unprocessed):
            if phot is None:
                pos, neg = list(range(nstars)), []
            elif isinstance(phot, dict):
                unknown = set(phot) - {"pos", "neg", "blend"}
                if unknown:
                    raise ValueError(
                        f"Unknown photType key(s) {sorted(unknown)} for "
                        f"``{self.filters[i]}``. Allowed: pos, neg, blend "
                        f"(alias for pos).")
                if "blend" in phot and "pos" in phot:
                    raise ValueError(
                        f"photType for ``{self.filters[i]}`` sets both "
                        f"'blend' and 'pos'; 'blend' is an alias for 'pos' — "
                        f"use one.")
                pos_refs = phot.get("pos", phot.get("blend"))
                if not pos_refs:
                    raise ValueError(
                        f"photType for ``{self.filters[i]}`` needs a "
                        f"non-empty 'pos' (or 'blend') star list.")
                pos = [self._resolve_star_ref(r, star_names) for r in np.atleast_1d(pos_refs)]
                neg = [self._resolve_star_ref(r, star_names) for r in np.atleast_1d(phot.get("neg") or [])]
            else:
                raise ValueError(
                    f"photType for ``{self.filters[i]}`` must be a mapping "
                    f"with 'pos'/'neg' (or 'blend') star lists; got {phot!r}.")

            overlap = set(pos) & set(neg)
            if overlap:
                raise ValueError(
                    f"photType for ``{self.filters[i]}`` lists star(s) "
                    f"{sorted(overlap)} in both pos and neg.")

            blend_matrix[i, pos] = 1
            blend_matrix[i, neg] = -1
            combo_labels.append(self._combo_label(pos, neg, star_names))

        self.blend_matrix = blend_matrix
        self.combo_labels = combo_labels


    _MODEL_DATA_URLS = {
        "NextGen": {
            "NextGen.spectra.csv":    "https://zenodo.org/records/20547997/files/NextGen.spectra.csv?download=1",
            "NextGen.wavelength.csv": "https://zenodo.org/records/20547997/files/NextGen.wavelength.csv?download=1",
        }
    }

    def _ensure_model_data(self):
        """Download large model data files from Zenodo if not present locally."""
        urls = self._MODEL_DATA_URLS.get(self.sedmodel, {})
        model_dir = current_dir / "models" / self.sedmodel
        for filename, url in urls.items():
            dest = model_dir / filename
            if not dest.exists():
                logger.info(f"Downloading {filename} from Zenodo...")
                urllib.request.urlretrieve(url, dest)
                logger.info(f"Saved {filename} to {dest}")

    def _collect_band_filters(self):
        """
        Gather filters referenced by Band blocks so cross-component flux
        predictions (mulensing f_source, transit deblending, astrometry
        fluxfrac) share this SED's BC grid. Band filters whose BC tables
        are not available are skipped with a warning (they can be
        generated with the BC table machinery) rather than failing the
        whole SED.

        Returns a list of filter names to append to the BC grid build,
        deduplicated against the .sed file's own filters.
        """
        band_cfgs = self.config_manager.system_config.get("band") or []
        alias_df = _load_alias_table()

        known_mist = {
            resolve_filter_name(n, alias_df, alias="MIST") for n in self.filters
        }
        extra = []
        for cfg in band_cfgs:
            name = cfg.get("filter")
            if not name:
                continue
            mist = resolve_filter_name(name, alias_df, alias="MIST")
            if mist in known_mist:
                continue
            svo = resolve_filter_name(name, alias_df, alias="SVO")
            facility = facility_from_svo_name(svo)
            try:
                _collect_facility_files(self.bc_root, self.sedmodel, facility)
            except (FileNotFoundError, NotImplementedError) as e:
                logger.warning(
                    f"SED: no BC tables for band filter '{name}' "
                    f"(facility '{facility}'): {e} Flux predictions in this "
                    f"band will be unavailable."
                )
                continue
            known_mist.add(mist)
            extra.append(name)
        return extra

    def load_data(self, system):
        if self.sedfile is None:
            raise ValueError(
                f"sed is missing the required 'file' key"
            )

        self._ensure_model_data()
        self._process_SED_yaml()

        # Star bookkeeping: the blend matrix is indexed over the system's
        # stars, referenced by name or index in each row's photType.
        star_names = list(system.star.names)
        self.nstars = system.star.n_elements
        yaml_nstars = self.SED_yaml.get("nstars")
        if yaml_nstars is not None and int(yaml_nstars) != self.nstars:
            logger.warning(
                f"SED file declares nstars={yaml_nstars} but the system "
                f"defines {self.nstars} star(s); using the system value."
            )
        self._build_blend_matrix(star_names)

        # Filters from the .sed file first (their columns feed the SED
        # likelihood), then any Band-referenced filters for cross-component
        # flux predictions.
        self.band_filters = self._collect_band_filters()
        self.all_filters = list(self.filters) + self.band_filters
        if not self.all_filters:
            raise ValueError(
                "SED has no usable filters: the .sed file lists none and no "
                "band block references a filter with BC tables."
            )

        grid = build_bc_grid(
            user_filter_names=self.all_filters,
            model=self.sedmodel,
            bc_root=self.bc_root,
        )
        self.bc_grid_data = grid
        self.mist_filters = grid["filter_order"]

        # Column lookup by user-facing name, MIST name, or SVO name so
        # other components can ask for a filter in whatever convention
        # their config uses.
        alias_df = _load_alias_table()
        self.filter_columns = {}
        for col, name in enumerate(self.all_filters):
            keys = {
                name,
                resolve_filter_name(name, alias_df, alias="MIST"),
                resolve_filter_name(name, alias_df, alias="SVO"),
            }
            for key in keys:
                self.filter_columns.setdefault(key, col)

        # Build the BC interpolator now, using config_manager.resolve() for
        # the star parameter bounds. _inject_grid_bounds() already wrote the
        # grid-axis limits into config_manager.user_params during __init__,
        # so resolve() returns the correct tightened bounds here.
        teff_cfg = self.config_manager.resolve('star', 'teffsed')
        feh_cfg  = self.config_manager.resolve('star', 'feh')
        av_cfg   = self.config_manager.resolve('star', 'av')

        grid_dict = {
            "model": self.sedmodel,
            "grid": {
                "teff": self.bc_grid_data["teff_pts"],
                "logg": self.bc_grid_data["logg_pts"],
                "feh":  self.bc_grid_data["feh_pts"],
                "av":   self.bc_grid_data["av_pts"],
            },
        }
        # Bounds may differ per star (user overrides); keep every grid
        # point any star can reach.
        bc_slice, axes = slice_bc(
            grid_dict, self.bc_grid_data['bc_values'],
            teff=(float(np.min(teff_cfg['lower'])), float(np.max(teff_cfg['upper']))),
            feh=(float(np.min(feh_cfg['lower'])),   float(np.max(feh_cfg['upper']))),
            av=(float(np.min(av_cfg['lower'])),      float(np.max(av_cfg['upper']))),
        )
        axes_full = {
            "teff": self.bc_grid_data["teff_pts"],
            "logg": self.bc_grid_data["logg_pts"],
            "feh":  self.bc_grid_data["feh_pts"],
            "av":   self.bc_grid_data["av_pts"],
        }
        axes_full.update(axes)
        self.bc_interpolator = RegularGridInterpolator(
            points=[axes_full['teff'], axes_full['logg'], axes_full['feh'], axes_full['av']],
            values=bc_slice,
        )

    # ------------------------------------------------------------------
    # 3) build_maps — the star axis is handled by blend_matrix; nothing
    #    to convert here.
    # ------------------------------------------------------------------
    def build_maps(self):
        pass

    # ------------------------------------------------------------------
    # Shared prediction nodes. Both the likelihood and the plotting
    # compiler (and other components, via the predict_* API below) need
    # the same (nstars, nfilters) apparent-magnitude graph; build it
    # once and cache the node.
    # ------------------------------------------------------------------
    def _predicted_appmag_node(self, system):
        """Per-star predicted apparent mags, shape (nstars, n_all_filters)."""
        if getattr(self, "_m_pred_matrix", None) is not None:
            return self._m_pred_matrix

        star = system.star
        teffsed = star.teffsed.value       # K,        (nstars,)
        radiussed = star.radiussed.value   # R_sun,    (nstars,)
        logmass = star.logmass.value       # dex(M_sun)
        feh = star.feh.value               # dex
        av = star.av.value                 # mag
        distance = star.distance.value     # pc

        # Reconstruct loggsed from logmass + radiussed (NOT radius).
        loggsed = calc_logg(logmass, radiussed)

        # RegularGridInterpolator.evaluate expects shape (ntest, ndim).
        coords = pt.stack([teffsed, loggsed, feh, av], axis=-1)  # (nstars, 4)
        bc = self.bc_interpolator.evaluate(coords)   # (nstars, n_all_filters)

        # Bolometric luminosity
        Lbol = calc_luminosity(radiussed, teffsed)
        # Absolute bolometric magnitude from bolometric luminosity.
        # M_bol = -2.5 log10(L_bol / L_bol_0) with IAU 2015 zero
        # point (see physics.L_BOL_ZERO_LSUN).
        Mbol_abs = calc_absbolmag(Lbol)              # (nstars,)

        # Absolute magnitude per star per filter.
        M_abs = calc_absmag_from_bc(Mbol_abs[:, None], bc)

        # Predicted apparent magnitude using the distance modulus.
        self._m_pred_matrix = calc_appmag(M_abs, distance[:, None])
        return self._m_pred_matrix

    def _combined_appmag_node(self, system):
        """
        Predicted magnitude for each .sed filter row, combining stars per
        the blend matrix: blended rows are flux sums, differential rows
        are -2.5*log10(F_pos/F_neg) (exofast_multised convention).
        Shape (nfilters,).
        """
        m_app = self._predicted_appmag_node(system)[:, :self.nfilters]
        F = 10 ** (-0.4 * m_app)                         # (nstars, nfilters)
        Cpos = np.clip(self.blend_matrix, 0, None)       # (nfilters, nstars)
        Cneg = np.clip(-self.blend_matrix, 0, None)
        F_pos = pt.sum(Cpos * F.T, axis=1)               # (nfilters,)
        F_neg = pt.sum(Cneg * F.T, axis=1)
        has_neg = (self.blend_matrix < 0).any(axis=1)    # static numpy mask
        F_neg_safe = pt.switch(pt.as_tensor(has_neg), F_neg, 1.0)
        return -2.5 * (pt.log10(F_pos) - pt.log10(F_neg_safe))

    # ------------------------------------------------------------------
    # Public flux-prediction API for other components (mulensinstrument
    # f_source constraint, transit deblending, astrometry fluxfrac).
    # Component-specific SED logic stays here; callers just hand over a
    # star index (or indices) and a filter in any supported naming
    # convention (user/VOID, MIST, or SVO).
    # ------------------------------------------------------------------
    def filter_column(self, filter_key):
        """Grid column for a filter name; raises KeyError with guidance."""
        if filter_key in self.filter_columns:
            return self.filter_columns[filter_key]
        alias_df = _load_alias_table()
        for alias in ("MIST", "SVO"):
            resolved = resolve_filter_name(filter_key, alias_df, alias=alias)
            if resolved in self.filter_columns:
                return self.filter_columns[resolved]
        raise KeyError(
            f"Filter '{filter_key}' is not in this SED's BC grid. "
            f"Available: {sorted(self.filter_columns)}. Band-referenced "
            f"filters are only included when their BC tables exist."
        )

    def has_filter(self, filter_key):
        try:
            self.filter_column(filter_key)
            return True
        except KeyError:
            return False

    def predict_star_appmag(self, star_idx, filter_key, system):
        """Predicted apparent magnitude of one star in one filter (scalar node)."""
        col = self.filter_column(filter_key)
        return self._predicted_appmag_node(system)[star_idx, col]

    def predict_blend_appmag(self, star_indices, filter_key, system):
        """Predicted apparent magnitude of the flux sum of several stars."""
        col = self.filter_column(filter_key)
        m = self._predicted_appmag_node(system)[:, col]   # (nstars,)
        idx = np.asarray(list(star_indices), dtype=int)
        F_sum = pt.sum(10 ** (-0.4 * m[idx]))
        return -2.5 * pt.log10(F_sum)

    def predict_flux_fraction(self, star_idx, filter_key, system):
        """
        Fraction of the total flux of all modeled stars contributed by
        one star in one filter. Used for transit deblending
        (depth dilution) and the astrometric photocenter fluxfrac.
        """
        col = self.filter_column(filter_key)
        m = self._predicted_appmag_node(system)[:, col]   # (nstars,)
        F = 10 ** (-0.4 * m)
        return F[star_idx] / pt.sum(F)

    # ------------------------------------------------------------------
    # 4) build_likelihood — Normal likelihood over the .sed filter rows
    #    (blended/differential magnitudes), plus the teffsed/fbolsed
    #    floor potentials tying the SED-side parameters to the primary
    #    stellar parameters.
    # ------------------------------------------------------------------
    def build_likelihood(self, model, system):
        star = system.star

        teff = star.teff.value             # K
        teffsed = star.teffsed.value       # K
        fbol = star.fbol.value             # erg/s/cm2
        fbolsed = star.fbolsed.value       # erg/s/cm2

        if self.nfilters > 0:
            m_pred = self._combined_appmag_node(system)   # (nfilters,)

            mag_data = pm.Data(f"sed_mag_data", self.mag)
            err_data = pm.Data(f"sed_mag_err", self.err)

            sigma = err_data * self.errscale.value

            pm.Normal(
                f"{self.prefix}.model",
                mu=m_pred,
                sigma=sigma,
                observed=mag_data,
            )

        # this links the two with a user settable error floor
        self.teffsed_floor_prior = pm.Potential("sed.teffsed_floor_prior",
                                                pt.sum(-0.5 * ((teff - teffsed) / (teff * self.teffsedfloor)) ** 2))
        self.fbolsed_floor_prior = pm.Potential("sed.fbolsed_floor_prior",
                                                pt.sum(-0.5 * ((fbol - fbolsed) / (fbol * self.fbolsedfloor)) ** 2))

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

        # All compiled functions take the FULL plot-parameter bundle
        # (vectors stay vectors) and return per-star / per-row arrays:
        #   _compiled_mag_predictors : (nstars, nfilters) per-star app mags
        #   _compiled_combined_mag   : (nfilters,) blended/diff row mags
        #   _compiled_logg_calc      : (nstars,) loggsed
        m_star_node = self._predicted_appmag_node(system)[:, :self.nfilters]
        loggsed_node = calc_logg(star.logmass.value, star.radiussed.value)

        try:
            self._compiled_mag_predictors = pytensor.function(
                inputs=param_symbols,
                outputs=m_star_node,
                on_unused_input="ignore",
            )
            self._compiled_logg_calc = pytensor.function(
                inputs=param_symbols,
                outputs=loggsed_node,
                on_unused_input="ignore",
            )
            if self.nfilters > 0:
                self._compiled_combined_mag = pytensor.function(
                    inputs=param_symbols,
                    outputs=self._combined_appmag_node(system),
                    on_unused_input="ignore",
                )
            else:
                self._compiled_combined_mag = None
        except Exception as e:
            # Don't let plotting infra kill model build; warn
            # and leave None in place so plot() skips gracefully.
            logger.warning(f"SED plotter compile failed: {e}")
            self._compiled_mag_predictors = None
            self._compiled_logg_calc = None
            self._compiled_combined_mag = None

    # ------------------------------------------------------------------
    # 7) plot — observed mag vs predicted mag per filter, per SED.
    # ------------------------------------------------------------------
    def plot(self, system, points, filename_prefix="debug"):
        if isinstance(points, dict):
            points = [points]
        if not points:
            logger.warning("SED.plot: no points provided.")
            return
        if self.nfilters == 0:
            logger.info("SED.plot: no catalog photometry rows; skipping SED plot.")
            return
        if getattr(self, "_compiled_mag_predictors", None) is None:
            logger.warning(
                "SED.plot: plotting functions failed to compile "
                "(see the earlier 'SED plotter compile failed' warning); "
                "skipping SED plot."
            )
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

        n_colors = len(plot_obj.colors_obs)
        n_markers = len(plot_obj.markers)
        n_lines = len(plot_obj.linetypes)

        # ---- model spectra: one line per star, plus the blended total ----
        alpha_spec = 0.7 if plot_obj.ndraws == 1 else 0.15
        x_spec = plot_obj.df_wave['wavelength_micron']
        wave_ang = plot_obj.df_wave['wavelength_angstrom']
        for d, draw in enumerate(plot_obj.draws):
            for nstar in range(plot_obj.nstars):
                y_spec = np.log10(plot_obj.flux_model_draws[d][nstar]*wave_ang)
                ax_top.plot(x_spec,
                            y_spec,
                            ls=plot_obj.linetypes[nstar % n_lines],
                            color=plot_obj.colors_spec[nstar % n_colors],
                            alpha=alpha_spec,
                            label=f"Star {plot_obj.star_names[nstar]}")
            if plot_obj.nstars > 1:
                # blended total (exofast_multised plots this in black)
                total = np.sum(plot_obj.flux_model_draws[d], axis=0)
                ax_top.plot(x_spec, np.log10(total*wave_ang),
                            color="#001219", lw=1.2, alpha=alpha_spec)

        # ---- observed photometry: one point per filter row, colored ----
        # ---- and markered by its star combination                    ----
        x = plot_obj.wave_filter*ANG_TO_MICRON_CONST
        xerr = plot_obj.wave_err*ANG_TO_MICRON_CONST

        y = np.log10(plot_obj.flux_obs*plot_obj.wave_filter)
        y_limits_from_err = np.log10(plot_obj.f_limits_from_err*plot_obj.wave_filter)

        # initialize yerr array
        yerr = np.zeros(np.shape(y_limits_from_err))
        yerr[0] = y - y_limits_from_err[0]
        yerr[1] = y_limits_from_err[1] - y

        combined_pred_med = np.median(plot_obj.combined_pred_draws, axis=0)
        alpha_res = 1.0 if plot_obj.nstars == 1 else 0.5

        for row in range(plot_obj.nfilters):
            ci = plot_obj.row_combo_idx[row]
            color = plot_obj.colors_obs[ci % n_colors]
            marker = plot_obj.markers[ci % n_markers]

            ax_top.errorbar(
                x[row:row+1], y[row:row+1],
                xerr=xerr[:, row:row+1], yerr=yerr[:, row:row+1], fmt='',
                color=color, capsize=3, linestyle='None',
                zorder=3
            )
            ax_top.scatter(x[row], y[row], color=color, marker=marker, zorder=3)

            # residual of this row against its combined (blend/diff) prediction
            residual = plot_obj.mag_obs[row] - combined_pred_med[row]
            ax_bot.errorbar(
                x[row], residual, yerr=plot_obj.mag_obs_err[row], capsize=3,
                fmt=marker, color=color,
                alpha=alpha_res
            )

        ax_top.set_xscale("log")

        # set y limits dynamically
        plot_obj._get_ylim()
        ax_top.set_ylim(plot_obj.y_lower, plot_obj.y_upper)
        ax_top.set_ylabel(r"log $\lambda F_\lambda$ (erg s$^{-1}$ cm$^{-2}$)")

        if plot_obj.nstars == 1:
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
        else:
            # per-star spectrum lines, the blended total, and one marker
            # entry per observed star combination ("A+B", "A-(B+C)", ...)
            legend_handles, labels = [], []
            for nstar in range(plot_obj.nstars):
                legend_handles.append(Line2D(
                    [0], [0], color=plot_obj.colors_spec[nstar % n_colors],
                    linestyle=plot_obj.linetypes[nstar % n_lines], alpha=0.7))
                labels.append(f"Star {plot_obj.star_names[nstar]}")
            legend_handles.append(Line2D([0], [0], color="#001219", lw=1.2))
            labels.append("Total")
            for ci, combo in enumerate(plot_obj.unique_combos):
                legend_handles.append(Line2D(
                    [0], [0], color='none',
                    marker=plot_obj.markers[ci % n_markers],
                    markerfacecolor=plot_obj.colors_obs[ci % n_colors],
                    markeredgecolor=plot_obj.colors_obs[ci % n_colors]))
                labels.append(combo)

            ax_top.legend(handles=legend_handles, labels=labels,
                          fontsize="small")

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