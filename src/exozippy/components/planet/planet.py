import logging

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from exozippy.components.component import Component
from exozippy.potentials import soft_lower_bound, soft_upper_bound

from . import physics

logger = logging.getLogger(__name__)


class Planet(Component):
    def __init__(self, config, config_manager):
        super().__init__(config, config_manager)
        self.label = "Planet Parameters"
        # BEER (PR 1.b): Doppler beaming amplitude. Per-planet, not
        # per-band (EXOFASTv2 declares it ss.planet[i].beam, unlike
        # thermal/reflect/ellipsoidal which are ss.band[i].*).
        self.beam_free = [
            bool(c.get("beam_free", False)) for c in self.config
        ]
        self.beam_constrains_mass = [
            bool(c.get("beam_constrains_mass", False)) for c in self.config
        ]

    @property
    def prefix(self):
        return "planet"

    @classmethod
    def config_schema(cls):
        return [
            {
                "key": "beam_free",
                "kind": "option",
                "accepts": None,
                "required": False,
                "doc": (
                    "Fit a Doppler beaming amplitude (ppm) for this planet "
                    "directly from the photometry, independent of the RV "
                    "semi-amplitude K -- does not constrain planet mass. "
                    "Default False, which pins beam at 0. If "
                    "beam_constrains_mass is also set, beam_constrains_mass "
                    "wins and beam is derived from K instead of fit freely."
                ),
            },
            {
                "key": "beam_constrains_mass",
                "kind": "option",
                "accepts": None,
                "required": False,
                "doc": (
                    "Compute this planet's Doppler beaming amplitude from "
                    "the RV semi-amplitude K (Faigler & Mazeh 2011 eq. 1, "
                    "bolometric approximation) instead of fitting it "
                    "freely -- ties the photometric beaming signal to the "
                    "same mass/K driving the RV model. Default False."
                ),
            },
        ]

    def build_maps(self):
        """Stage 1b: Define logical Numpy arrays. (Base class auto-converts to Tensors)."""
        self.star_map = np.array([p.get("star_ndx", 0) for p in self.config])
        self.orbit_map = np.array([p.get("orbit_ndx", 0) for p in self.config])

    def register_parameters(self, system):
        """Stage 2: Auto-estimates and Manifest declaration."""
        has_orbit = "orbit" in system.active_components

        self.manifest = {
            "mass": None,
            "radius": None,
            "density": "default",
            "logg": "default",
            "m_total": "default",
        }

        if has_orbit:
            self.manifest.update(
                {
                    "p": "default",
                    "arsun": "default",
                    "ar": "default",
                    "b": "default",
                    "K": "default",
                    "max_ecc": "default",
                }
            )

        # BEER (PR 1.b): beam is either (a) derived from K for every planet
        # (beam_constrains_mass), (b) free-fit for every planet (beam_free),
        # or (c) pinned at 0 everywhere (neither set -- transit model
        # unchanged). The two flags are not mutually exclusive: per
        # EXOFASTv2's step2pars.pro (~line 256), beam is computed whenever
        # either is set, and beam_constrains_mass takes priority -- when
        # both are set, beam is still derived from K, not fit freely.
        # Unlike thermal/reflect/ellipsoidal's per-band opt-in, this is a
        # single mode for the whole component: Component.add_parameter
        # resolves one manifest entry as either a whole-component expression
        # ("default") or a whole-component free/fixed tensor (via per-element
        # sigma), never a mix of the two within one parameter -- so a
        # per-planet mix of derived/free/off beam isn't supported yet.
        any_beam_constrains_mass = any(self.beam_constrains_mass)
        any_beam_free = any(self.beam_free)
        if any_beam_constrains_mass and not has_orbit:
            raise ValueError(
                f"[{self.prefix}] beam_constrains_mass requires an orbit "
                f"component (beam is derived from K, which requires the "
                f"orbital elements)."
            )
        if any_beam_constrains_mass:
            self.manifest["beam"] = "default"
        elif any_beam_free:
            off = [
                i for i in range(self.n_elements) if not self.beam_free[i]
            ]
            entry = {}
            if off:
                pin = np.full(self.n_elements, np.nan)
                pin[off] = 0.0
                entry["overrides"] = {"sigma": pin.tolist()}
            self.manifest["beam"] = entry
        else:
            self.manifest["beam"] = {
                "overrides": {"sigma": [0.0] * self.n_elements}
            }

        # Data-driven estimate: Initialize 'K' directly from the RV data variance
        rv_comps = [
            c
            for c in system.active_components.values()
            if hasattr(c, "k_init")
        ]
        if rv_comps and has_orbit:
            k_ms_guess = rv_comps[0].k_init / np.sqrt(self.n_elements)
            for i in range(self.n_elements):
                self.config_manager.add_hint(f"planet.{i}.K", k_ms_guess)

    def build_likelihood(self, model, system):
        # Soft barriers via the shared clipped log-sigmoid helpers.
        # scale * softness = 0.0088 reproduces the historical steepness of
        # 500 nats/unit.  The raw pm.math.log(pt.sigmoid(z)) form used here
        # previously NaNs in the JAX gradient once z > ~709 (exp(z) overflows
        # inside an unselected jnp.where branch of pytensor's softplus): any
        # system with m_total > 1.42 Msun silently froze every numpyro chain
        # at its starting point.  See potentials.py.
        pm.Potential(
            f"{self.prefix}.m_pos_constraint",
            soft_lower_bound(self.m_total.value, 0.0, scale=0.88),
        )

        if "orbit" not in system.active_components:
            return

        orbits = system.orbit
        pm.Potential(
            f"{self.prefix}.e_collision_bound",
            soft_upper_bound(
                orbits.ecc.value[self.orbit_map],
                self.max_ecc.value,
                scale=0.88,
            ),
        )

        if self.n_elements < 2:
            return

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
            inner_apastron = inner.orbit.a.value * (
                1.0 + inner.orbit.ecc.value
            )

            # Get the symbolic periastron (closest point) of the outer planet
            # q = a * (1 - e)
            outer_periastron = outer.orbit.a_val * (
                1.0 - outer.orbit.ecc.value
            )

            # Potential: If they cross, log-probability goes to -inf
            pm.Potential(
                f"crossing_penalty_{inner.name}_{outer.name}",
                pt.switch(outer_periastron > inner_apastron, 0.0, -np.inf),
            )

    def plot(self, system, points, filename_prefix="debug"):
        pass
