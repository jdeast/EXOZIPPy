import logging

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from exozippy.components.component import Component
from . import physics

logger = logging.getLogger(__name__)


class Torres(Component):
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

    Two things differ structurally from mann, which is why the two are
    separate components rather than sharing a base:

    * The relations predict logarithms, and their scatter is quoted in dex,
      not as a fraction.  The mass penalty therefore acts directly on
      star.logmass -- the parameter EXOZIPPy already samples -- with no
      exponentiation round trip.
    * That scatter is a constant, so (unlike mann's prediction-proportional
      sigma) the -log(sigma) normalization is constant and is dropped, which
      is exactly EXOFASTv2's chi2.
    """

    yaml_key = "torres"

    def __init__(self, component_config, config_manager):
        # Name each instance after the star it constrains, so the base
        # class's duplicate-name check also enforces one Torres per star.
        for c in component_config:
            if c.get("name") is None and c.get("star") is not None:
                c["name"] = str(c["star"]).split(".")[-1]
        super().__init__(component_config, config_manager)

    @property
    def prefix(self):
        return "torres"

    @classmethod
    def config_schema(cls):
        return [
            {
                "key": "star",
                "kind": "ref",
                "accepts": ["star"],
                "required": True,
                "doc": (
                    "Name (or index) of the star this Torres relation "
                    "constrains. One Torres instance per star."
                ),
            },
            {
                "key": "constrain",
                "kind": "option",
                "accepts": ["mass", "radius"],
                "required": False,
                "doc": (
                    "Which stellar quantities to constrain from "
                    "teff/logg/feh (a list; default both mass and radius)."
                ),
            },
        ]

    def load_data(self, system):
        """Stage 1a: resolve the target stars and parse the per-instance config."""
        star_names = list(system.star.names)
        name_to_idx = {n: i for i, n in enumerate(star_names)}

        self.star_indices = []
        self.constrain = []
        self.logm_floor = []
        self.logr_floor = []

        for c, nm in zip(self.config, self.names):
            raw_star = c.get("star")
            if raw_star is None:
                raise ValueError(
                    f"torres '{nm}': a 'star:' key is required naming the star "
                    f"to constrain. Available stars: {star_names}."
                )
            key = str(raw_star).split(".")[-1]
            if key in name_to_idx:
                self.star_indices.append(name_to_idx[key])
            elif key.isdigit() and int(key) < len(star_names):
                self.star_indices.append(int(key))
            else:
                raise ValueError(
                    f"torres '{nm}': unknown star '{raw_star}'. "
                    f"Available stars: {star_names}."
                )

            con = c.get("constrain", ["mass", "radius"])
            if isinstance(con, str):
                con = [con]
            con = set(con)
            bad = con - {"mass", "radius"}
            if bad:
                raise ValueError(
                    f"torres '{nm}': unknown 'constrain:' entries {sorted(bad)}; "
                    f"valid entries are 'mass' and 'radius'."
                )
            if not con:
                raise ValueError(
                    f"torres '{nm}': 'constrain:' is empty, so this block would "
                    f"do nothing. Remove it or list 'mass' and/or 'radius'."
                )
            self.constrain.append(con)

            # Scatter overrides, in dex (mann's floors are fractional -- hence
            # the different names).
            self.logm_floor.append(
                float(c.get("logm_floor", physics.LOGM_FLOOR)))
            self.logr_floor.append(
                float(c.get("logr_floor", physics.LOGR_FLOOR)))
            if self.logm_floor[-1] <= 0 or self.logr_floor[-1] <= 0:
                raise ValueError(
                    f"torres '{nm}': 'logm_floor:'/'logr_floor:' must be > 0 dex.")

    def build_maps(self):
        """Stage 1b: index array linking each instance to its star."""
        self.star_map = np.array(self.star_indices, dtype=int)

    def register_parameters(self, system):
        """Stage 2: nothing to declare -- Torres adds only potentials.

        Its inputs (star.teff, star.logg, star.feh) are already on the star
        component, and the relation has no latent of its own.
        """
        self.manifest = {}

    def _warn_outside_calibration(self, system):
        """Warn when a star starts below the relations' calibrated mass range.

        EXOFASTv2 warns on every likelihood call; this is a startup warning
        only, and nothing here bounds the posterior.
        """
        for i, nm in enumerate(self.names):
            si = self.star_indices[i]
            v = getattr(system.star.mass, "initval", None)
            if v is None:
                continue
            arr = np.atleast_1d(np.asarray(v, dtype=float))
            mass = float(arr[0] if arr.size == 1 else arr[si])
            if np.isnan(mass):
                continue
            if mass < physics.MSTAR_MIN:
                logger.warning(
                    f"torres '{nm}': star '{system.star.names[si]}' starts at "
                    f"{mass:.3f} solMass, below the Torres+2010 calibration "
                    f"floor of {physics.MSTAR_MIN} solMass. If it stays there, "
                    f"prefer the Mann+ relations (components/mann), MIST or "
                    f"PARSEC."
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
        pm.Deterministic(f"{self.prefix}.mass_pred", 10.0 ** logm_pred)
        pm.Deterministic(f"{self.prefix}.radius_pred", 10.0 ** logr_pred)

        # star.logmass IS log10(M/Msun), so the mass penalty needs no
        # conversion. star.radius is linear, so take its log.
        self._add_penalty(
            "mass", star.logmass.value[smap], logm_pred, self.logm_floor)
        self._add_penalty(
            "radius", pt.log10(star.radius.value[smap]), logr_pred,
            self.logr_floor)

    def _add_penalty(self, which, observed_log, predicted_log, floors):
        """Gaussian potential in log space tying a star parameter to Torres.

        Only instances that asked for this parameter in `constrain:`
        contribute. sigma is a constant in dex, so its -log(sigma)
        normalization is constant and dropped -- this is exactly EXOFASTv2's
        (alog10(mstar/mstar_prior)/umstar)^2 chi2 term.
        """
        mask = np.array([which in c for c in self.constrain], dtype=bool)
        if not mask.any():
            return
        sigma = pt.as_tensor_variable(np.asarray(floors, dtype=float))
        logp = -0.5 * pt.sqr((observed_log - predicted_log) / sigma)
        pm.Potential(
            f"{self.prefix}.{which}_prior",
            pt.sum(pt.where(pt.as_tensor_variable(mask), logp, 0.0)),
        )

    def compile_plotters(self, model, system):
        pass

    def plot(self, system, points, filename_prefix="debug"):
        pass
