import logging
import numpy as np
import pytensor.tensor as pt
import pymc as pm
from exozippy.components.component import Component
from . import physics

logger = logging.getLogger(__name__)


class Planet(Component):
    def __init__(self, config, config_manager):
        super().__init__(config, config_manager)
        self.label = "Planet Parameters"

    @property
    def prefix(self):
        return "planet"

    def build_maps(self):
        """Stage 1b: Define logical Numpy arrays. (Base class auto-converts to Tensors)."""
        self.star_map = np.array([p.get("star_ndx", 0) for p in self.config])
        self.orbit_map = np.array([p.get("orbit_ndx", 0) for p in self.config])

    def register_parameters(self, system):
        """Stage 2: Auto-estimates and Manifest declaration."""
        self.manifest = {
            "mass": None,
            "radius": None,
            "density": "default",
            "logg": "default"
        }

        # 2. Add Planet-specific observables
        self.manifest.update({
            "m_total": "default",
            "p": "default",
            "arsun": "default",
            "ar": "default",
            "b": "default",
            "K": "default",
            "max_ecc": "default",
        })

        # 3. Data-driven estimate: Initialize 'K' directly from the RV data variance
        rv_comps = [c for c in system.active_components.values() if hasattr(c, 'k_init')]
        if rv_comps:
            # Split the total observed RV variance equally among all planets
            k_ms_guess = rv_comps[0].k_init / np.sqrt(self.n_elements)

            for i in range(self.n_elements):
                # Pass K_init to the ConfigManager as a low-rank fact
                self.config_manager.add_hint(f"planet.{i}.K", k_ms_guess)

    def build_likelihood(self, model, system):
        # (Keep the soft boundary Potentials here, as they are system-level likelihoods, not parameters)
        steepness = 500.0
        pm.Potential(f"{self.prefix}.m_pos_constraint", pm.math.log(pt.sigmoid(self.m_total.value * steepness)))

        orbits = system.orbit
        diff = self.max_ecc.value - orbits.ecc.value[self.orbit_map]
        pm.Potential(f"{self.prefix}.e_collision_bound", pm.math.log(pt.sigmoid(diff * steepness)))


        if self.n_elements < 2: return

        logger.warning("Planet collision penalty is untested.")

        # 1. Sort planets by semi-major axis (using the PyMC variables)
        # Note: Since these are tensors, we usually assume the user
        # provided them in order, or we use their 'initval' to sort.
        sorted_planets = sorted(self.planets, key=lambda p: p.a.initval)

        for i in range(len(sorted_planets) - 1):
            inner = sorted_planets[i]
            outer = sorted_planets[i + 1]

            # Get the symbolic apastron (furthest point) of the inner planet
            # Q = a * (1 + e)
            inner_apastron = inner.orbit.a.value * (1.0 + inner.orbit.ecc.value)

            # Get the symbolic periastron (closest point) of the outer planet
            # q = a * (1 - e)
            outer_periastron = outer.orbit.a_val * (1.0 - outer.orbit.ecc.value)

            # Potential: If they cross, log-probability goes to -inf
            pm.Potential(
                f"crossing_penalty_{inner.name}_{outer.name}",
                pt.switch(outer_periastron > inner_apastron, 0.0, -np.inf)
            )

    def plot(self, system, points, filename_prefix="debug"):
        pass