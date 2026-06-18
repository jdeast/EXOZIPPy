import logging
import numpy as np
import pymc as pm
import pytensor.tensor as pt

from exozippy.components.component import Component

logger = logging.getLogger(__name__)


class Band(Component):
    """Photometric band with limb-darkening coefficients.

    One Band instance per filter × source-star combination.  Reused across
    microlensing (finite-source magnification) and transits.

    Config keys (per instance):
      filter       : SVO filter name, e.g. "Cousins.I" or "Spitzer.IRAC.36"
      star_ndx     : index into the star component (the star providing Teff/logg/feh)
      ld_law       : "quadratic" (default) | "linear"
      claret_sigma : if > 0, penalise the sampled LD against the Claret grid prediction
                     with this Gaussian sigma per coefficient
    """

    yaml_key = "band"

    @property
    def prefix(self):
        return "band"

    def load_data(self, system):
        self.filter_names = [c.get("filter", "") for c in self.config]
        self.star_indices = [c.get("star_ndx", 0) for c in self.config]
        self.ld_laws = [c.get("ld_law", "quadratic") for c in self.config]
        self.claret_sigmas = [float(c.get("claret_sigma", 0.0)) for c in self.config]

        # Load Claret grids for instances that request them
        self._claret_grids = [None] * self.n_elements
        for i, (filt, sigma) in enumerate(zip(self.filter_names, self.claret_sigmas)):
            if sigma > 0 and filt:
                self._claret_grids[i] = self._load_claret_grid(filt)

    def build_maps(self):
        self.star_map = np.array(self.star_indices, dtype=int)

    def register_parameters(self, system):
        self.manifest = {}
        for i, law in enumerate(self.ld_laws):
            if law == "linear":
                self.manifest["gamma"] = None
            else:
                # quadratic (default)
                self.manifest["u1"] = None
                self.manifest["u2"] = None
                break  # manifest is shared; all instances use the same keys

    def build_likelihood(self, model, system):
        for i in range(self.n_elements):
            if self.claret_sigmas[i] <= 0 or self._claret_grids[i] is None:
                continue

            grid = self._claret_grids[i]
            star_i = self.star_map[i]
            teff = system.star.teff.value[star_i]
            logg = system.star.logg.value[star_i]
            feh = system.star.feh.value[star_i]
            sigma = self.claret_sigmas[i]

            # Claret prediction via pytensor-compatible interpolation
            u1_claret = self._claret_interp_pt(grid, teff, logg, feh, "u1")
            u2_claret = self._claret_interp_pt(grid, teff, logg, feh, "u2")

            u1_sampled = self.u1.value[i]
            u2_sampled = self.u2.value[i]

            pm.Potential(
                f"band.claret_prior.{i}",
                -0.5 * (((u1_sampled - u1_claret) / sigma) ** 2
                        + ((u2_sampled - u2_claret) / sigma) ** 2)
            )

    # ------------------------------------------------------------------
    # Claret grid helpers
    # ------------------------------------------------------------------

    def _load_claret_grid(self, filter_name):
        """Load a Claret (2000/2004) LD grid for this filter.

        Looks for a file in the package data directory named
        ``claret_{filter_name}.npz`` (with slashes/dots replaced by underscores).
        Returns None silently if the file is not found.
        """
        import os
        safe_name = filter_name.replace("/", "_").replace(".", "_")
        data_dir = os.path.join(os.path.dirname(__file__), "claret_grids")
        path = os.path.join(data_dir, f"claret_{safe_name}.npz")
        if not os.path.exists(path):
            logger.warning(f"Claret grid not found for filter '{filter_name}' at {path}; "
                           "Claret penalty disabled.")
            return None
        data = np.load(path)
        return {k: data[k] for k in data.files}

    def _claret_interp_pt(self, grid, teff, logg, feh, coeff):
        """PyTensor-compatible bilinear interpolation at a single (teff, logg, feh) point.

        Falls back to the nearest grid point when extrapolating.
        """
        from scipy.interpolate import RegularGridInterpolator
        axes = (grid['teff'], grid['logg'], grid['feh'])
        interp = RegularGridInterpolator(axes, grid[coeff], method='linear',
                                          bounds_error=False, fill_value=None)
        # Evaluate at the CURRENT numerical values (for the prior penalty, we use
        # the pytensor Op pattern: wrap numpy inside a blackbox Op).
        return _ClaretInterpOp(interp)(teff, logg, feh)

    def compile_plotters(self, model, system):
        pass

    def plot(self, system, points, filename_prefix="debug"):
        pass


# ---------------------------------------------------------------------------
# Minimal PyTensor Op for Claret interpolation
# ---------------------------------------------------------------------------

from pytensor.graph import Op, Apply
import pytensor.tensor as _pt


class _ClaretInterpOp(Op):
    """Wraps a scipy RegularGridInterpolator as a PyTensor scalar Op."""

    itypes = [_pt.dscalar, _pt.dscalar, _pt.dscalar]
    otypes = [_pt.dscalar]

    def __init__(self, interp):
        self._interp = interp

    def perform(self, node, inputs, outputs):
        teff, logg, feh = inputs
        outputs[0][0] = float(self._interp([[float(teff), float(logg), float(feh)]])[0])

    def pullback(self, inputs, outputs, cotangents):
        from pytensor.gradient import DisconnectedType
        return [DisconnectedType()(), DisconnectedType()(), DisconnectedType()()]

    # Backward compatibility with PyTensor < 3 which calls grad() instead of pullback()
    def grad(self, inputs, gradients):
        return self.pullback(inputs, [], gradients)
