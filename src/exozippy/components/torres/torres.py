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


class Torres(StellarRelation, Component):
    """Constrain a star's mass and/or radius with the Torres+2010 relations.

    One instance per constrained star::

        torres:
          - star: "A"
            constrain: [mass, radius]

    The relations predict log10(M) and log10(R) from Teff, logg and [Fe/H];
    each requested prediction becomes a Gaussian potential on the star's own
    node.  This mirrors EXOFASTv2's ss.torres[i] switch in
    exofast_chi2v2.pro, which always applies both.

    Unlike components/mann this needs no new parameters at all: every input
    (star.teff, star.logg, star.feh) is already on the star, and there is no
    Ks latent, so the manifest is empty and the component contributes only
    potentials.

    Two things differ statistically from mann, which is why the two are
    separate components sharing only the ``StellarRelation`` scaffolding
    rather than a common base class that owns the likelihood:

    * The relations predict logarithms, and their scatter is quoted in dex,
      not as a fraction.  The mass penalty therefore acts directly on
      star.logmass -- the parameter EXOZIPPy already samples -- with no
      exponentiation round trip.
    * That scatter is a constant, so (unlike mann's prediction-proportional
      sigma) the -log(sigma) normalization is constant and is dropped, which
      is exactly EXOFASTv2's chi2.  That is the ``normalize=False`` in
      build_likelihood below; mann passes ``normalize=True``.
    """

    yaml_key = "torres"

    @property
    def prefix(self):
        return "torres"

    @classmethod
    def config_schema(cls):
        return [
            star_schema_entry("Torres"),
            constrain_schema_entry("teff/logg/feh"),
        ]

    def load_data(self, system):
        """Stage 1a: resolve the target stars and parse the per-instance config."""
        self.star_indices = []
        self.constrain = []
        self.logm_floor = []
        self.logr_floor = []

        for c, nm in zip(self.config, self.names):
            self.star_indices.append(
                self._resolve_star(system, nm, c.get("star"))
            )
            self.constrain.append(
                self._parse_constrain(nm, c.get("constrain"))
            )

            # Scatter overrides, in dex (mann's floors are fractional -- hence
            # the different names).
            self.logm_floor.append(
                float(c.get("logm_floor", physics.LOGM_FLOOR))
            )
            self.logr_floor.append(
                float(c.get("logr_floor", physics.LOGR_FLOOR))
            )
            if self.logm_floor[-1] <= 0 or self.logr_floor[-1] <= 0:
                raise ValueError(
                    f"torres '{nm}': 'logm_floor:'/'logr_floor:' must be > 0 dex."
                )

    def register_parameters(self, system):
        """Stage 2: nothing to declare -- Torres adds only potentials.

        Its inputs (star.teff, star.logg, star.feh) are already on the star
        component, and the relation has no latent of its own.
        """
        self.manifest = {}

    def _warn_outside_calibration(self, system):
        """Warn when a star starts below the relations' calibrated mass range.

        EXOFASTv2 warns on every likelihood call; this is a startup warning
        only, and nothing here bounds the posterior (see
        ``StellarRelation._warn_outside_range``).
        """
        self._warn_outside_range(
            system,
            system.star.mass,
            physics.MSTAR_MIN,
            np.inf,
            message=(
                "star '{star}' starts at "
                "{value:.3f} solMass, below the Torres+2010 calibration "
                f"floor of {physics.MSTAR_MIN} solMass. If it stays there, "
                "prefer the Mann+ relations (components/mann), MIST or "
                "PARSEC."
            ),
        )

    def build_likelihood(self, model, system):
        star = system.star
        smap = self.star_map_tensor

        # Deferred to stage 6: the relaxation engine (stage 3) has run by now,
        # so these initvals are the ones the sampler will actually start from.
        self._warn_outside_calibration(system)

        # star.teff is K, logg cgs dex, feh dex -- all identity unit
        # conversions, so these are already in the relations' units.
        teff = star.teff.value[smap]
        logg = star.logg.value[smap]
        feh = star.feh.value[smap]

        logm_pred = physics.calc_torres_logmass(teff, logg, feh)
        logr_pred = physics.calc_torres_logradius(teff, logg, feh)

        # Report in linear units for humans; the penalties stay in log space.
        pm.Deterministic(f"{self.prefix}.mass_pred", 10.0**logm_pred)
        pm.Deterministic(f"{self.prefix}.radius_pred", 10.0**logr_pred)

        # star.logmass IS log10(M/Msun), so the mass penalty needs no
        # conversion. star.radius is linear, so take its log.  sigma is a
        # CONSTANT scatter in dex, so its -log(sigma) is an additive constant:
        # normalize=False drops it, which is exactly EXOFASTv2's
        # (alog10(mstar/mstar_prior)/umstar)^2 chi2 term.  (components/mann
        # passes normalize=True, and is right to: its sigma is a fraction of
        # a sampled prediction.)
        self._add_penalty(
            "mass",
            star.logmass.value[smap],
            logm_pred,
            as_float_vector(self.logm_floor),
            normalize=False,
        )
        self._add_penalty(
            "radius",
            pt.log10(star.radius.value[smap]),
            logr_pred,
            as_float_vector(self.logr_floor),
            normalize=False,
        )

        # Modeling-draft prose, next to the penalties it describes.
        self._add_relation_prose(
            system,
            cite_by_quantity={
                "mass": r"\citet{Torres:2010}",
                "radius": r"\citet{Torres:2010}",
            },
            input_desc="its effective temperature, surface gravity, and "
            "metallicity",
        )
