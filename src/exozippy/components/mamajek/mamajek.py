import numpy as np
import pymc as pm
import pytensor.tensor as pt

from exozippy.components.component import Component
from exozippy.components.relations import (
    StellarRelation,
    as_float_vector,
    constrain_schema_entry,
    star_schema_entry,
)

from . import physics


class Mamajek(StellarRelation, Component):
    """Tie a dwarf's Teff (and optionally radius) to its MASS through the
    Pecaut & Mamajek mean sequence.

    One instance per constrained star::

        mamajek:
          - star: "Lens"
            constrain: [teff]        # the default; add radius to tie that too
            teff_floor: 0.04         # fractional sigma, default 4%

    Why it exists (notes 2026-09-25/27).  A star nobody photometers -- a
    microlensing lens -- has its mass from the event (theta_E, pi_rel) and,
    with ``mann`` on it, its radius and absolute Ks from that mass.  Nothing
    holds its Teff but the Ks tie, and Ks is a weak thermometer for a hot
    star (dM_Ks/dTeff falls from -0.95 to -0.22 mag/1000 K between 3300 and
    5500 K), so the lens's Teff posterior grows 3-4x wider between a 0.26
    and a 0.8 solMass lens.  That mass-dependent nuisance volume, once
    marginalized, tilted pi_rel 0.3-0.6 dex low and the lens mass 2-5x high
    on the DC2018 sweep while the profile maximum sat at the truth.  This
    relation says what Mann cannot: a dwarf of mass M has the Teff a dwarf
    of mass M has, to the sequence's own spread.

    Complementary to ``mann`` and consistent with it: evaluated at the
    table's own M_Ks, Mann reproduces the table's mass and radius to ~2%
    from 0.1 to 0.6 solMass, so a Mann mass/radius and a Mamajek Teff on one
    star do not fight.  Radius is constrainable too but off by default:
    below 0.7 solMass Mann's Ks-based radius is the more accurate one, and
    two radius priors on one star would double-count.

    Statistically like ``mann``: the prediction is a table lookup at the
    sampled log-mass, the sigma is a FRACTION of the prediction, so the
    ``-log(sigma)`` normalization is kept (``normalize=True``; see
    ``relations._add_penalty``).
    """

    constrainable = ("teff", "radius")

    @property
    def prefix(self):
        return "mamajek"

    @classmethod
    def config_schema(cls):
        return [
            star_schema_entry("Pecaut & Mamajek dwarf-sequence"),
            constrain_schema_entry("mass", options=list(cls.constrainable)),
            {
                "key": "teff_floor",
                "kind": "option",
                "accepts": None,
                "required": False,
                "doc": (
                    "Fractional Gaussian width of the Teff tie. Default "
                    f"{physics.TEFF_FLOOR} (the sequence's own spread at "
                    "fixed mass, ~100-150 K)."
                ),
            },
            {
                "key": "radius_floor",
                "kind": "option",
                "accepts": None,
                "required": False,
                "doc": (
                    "Fractional Gaussian width of the radius tie, when "
                    f"'radius' is constrained. Default {physics.RADIUS_FLOOR}."
                ),
            },
        ]

    def load_data(self, system):
        """Stage 1: resolve the target stars and parse the per-instance config."""
        self.star_indices = []
        self.constrain = []
        self.teff_floor = []
        self.radius_floor = []
        for c, nm in zip(self.config, self.names):
            self.star_indices.append(
                self._resolve_star(system, nm, c.get("star"))
            )
            # Teff only by default -- see the class docstring on radius.
            self.constrain.append(
                self._parse_constrain(nm, c.get("constrain", ["teff"]))
            )
            self.teff_floor.append(
                float(c.get("teff_floor", physics.TEFF_FLOOR))
            )
            self.radius_floor.append(
                float(c.get("radius_floor", physics.RADIUS_FLOOR))
            )
            if self.teff_floor[-1] <= 0 or self.radius_floor[-1] <= 0:
                raise ValueError(
                    f"mamajek '{nm}': 'teff_floor:'/'radius_floor:' must be > 0."
                )

    def register_parameters(self, system):
        """Stage 3: nothing to declare -- potentials only, like torres."""
        self.manifest = {}

    def _warn_outside_calibration(self, system):
        self._warn_outside_range(
            system,
            system.star.mass,
            physics.MSTAR_MIN,
            physics.MSTAR_MAX,
            message=(
                "star '{star}' starts at {value:.3f} solMass, outside the "
                f"dwarf sequence's tabulated range "
                f"[{physics.MSTAR_MIN}, {physics.MSTAR_MAX}] solMass; the "
                "prediction is clamped at the nearest edge there."
            ),
        )

    def build_likelihood(self, model, system):
        star = system.star
        smap = self.star_map_tensor
        self._warn_outside_calibration(system)

        logmass = star.logmass.value[smap]
        teff_pred = 10.0 ** physics.calc_mamajek_logteff(logmass)
        radius_pred = 10.0 ** physics.calc_mamajek_logradius(logmass)
        pm.Deterministic(f"{self.prefix}.teff_pred", teff_pred)
        pm.Deterministic(f"{self.prefix}.radius_pred", radius_pred)

        self._add_penalty(
            "teff",
            star.teff.value[smap],
            teff_pred,
            teff_pred * as_float_vector(self.teff_floor),
            normalize=True,
        )
        self._add_penalty(
            "radius",
            star.radius.value[smap],
            radius_pred,
            radius_pred * as_float_vector(self.radius_floor),
            normalize=True,
        )
        self._add_relation_prose(
            system,
            cite_by_quantity={
                "teff": r"\citet{Pecaut:2013}",
                "radius": r"\citet{Pecaut:2013}",
            },
            input_desc="its mass, along the main sequence",
        )
