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

from ..star.physics import calc_absmag
from . import physics

# The bandpass the Mann relations are calibrated on.  EXOFASTv2 hardcodes
# the same curve (sed/filtercurves/2MASS_2MASS.Ks.idl).
KS_FILTER = "2MASS/2MASS.Ks"


class Mann(StellarRelation, Component):
    """Constrain a star's mass and/or radius with the Mann+ relations.

    One instance per constrained star::

        mann:
          - star: "B"
            ks: synthetic          # or a number: the observed apparent Ks
            constrain: [mass, radius]

    The relations predict mass and radius from a star's *absolute* Ks (and
    optionally [Fe/H]); each requested prediction becomes a Gaussian
    potential on the corresponding star parameter, with the relation's
    published fractional scatter as sigma.  This mirrors EXOFASTv2's
    mannmass/mannrad (observed Ks) and mannsynmass/mannsynrad (synthetic Ks)
    switches in exofast_chi2v2.pro.

    Two Ks pathways, differing only in where the apparent Ks comes from:

    * ``ks: synthetic`` -- the SED component's model spectrum for this star,
      via ``sed.predict_star_appmag``.  This is the per-star, unblended
      prediction, so it works even when the observed Ks photometry is a
      blend of several modeled stars (as in examples/kelt4).
    * ``ks: <number>`` -- a directly observed apparent Ks, with ``ks_err``.

    Either way the value handed to the relations is
    ``appks = ks_source + ks_err * ks_offset`` with ``ks_offset ~ N(0, 1)``,
    which is EXOFASTv2's 0.02 mag systematic floor written non-centered.

    The wiring shared with ``components/torres`` (instance naming, star and
    ``constrain:`` resolution, the masked penalty, the calibration-range
    warning) comes from the ``StellarRelation`` mixin; the statistics -- the
    linear predictions, the fractional scatter, and the kept ``-log(sigma)``
    normalization that follows from it -- stay here.
    """

    yaml_key = "mann"

    @property
    def prefix(self):
        return "mann"

    @classmethod
    def config_schema(cls):
        return [
            star_schema_entry("Mann"),
            constrain_schema_entry("absolute Ks"),
            {
                "key": "ks",
                "kind": "option",
                "accepts": None,
                "required": False,
                "doc": (
                    "'synthetic' to take the star's Ks from the SED model, or "
                    "a number for a directly observed apparent Ks (with "
                    "'ks_err')."
                ),
            },
            {
                "key": "ks_err",
                "kind": "option",
                "accepts": None,
                "required": False,
                "doc": "Uncertainty on an observed apparent Ks. Default 0.02.",
            },
        ]

    def load_data(self, system):
        """Stage 1a: resolve the target stars and parse the per-instance config."""
        self.star_indices = []
        self.ks_synthetic = []
        self.ks_observed = []
        self.ks_err = []
        self.constrain = []
        self.use_feh = []
        self.mstar_floor = []
        self.rstar_floor = []

        for c, nm in zip(self.config, self.names):
            # --- target star ---
            self.star_indices.append(
                self._resolve_star(system, nm, c.get("star"))
            )

            # --- Ks pathway ---
            ks = c.get("ks", "synthetic")
            if isinstance(ks, str):
                if ks.lower() != "synthetic":
                    raise ValueError(
                        f"mann '{nm}': 'ks:' must be either the string "
                        f"'synthetic' or a number (the observed apparent Ks "
                        f"magnitude); got '{ks}'."
                    )
                self.ks_synthetic.append(True)
                self.ks_observed.append(np.nan)
                # EXOFASTv2's hardcoded systematic floor between the SED's
                # synthetic Ks and the Ks fed to the relation.
                self.ks_err.append(float(c.get("ks_err", 0.02)))
            else:
                self.ks_synthetic.append(False)
                self.ks_observed.append(float(ks))
                if c.get("ks_err") is None:
                    raise ValueError(
                        f"mann '{nm}': 'ks_err:' is required when 'ks:' is an "
                        f"observed magnitude ({ks})."
                    )
                self.ks_err.append(float(c["ks_err"]))
            if self.ks_err[-1] <= 0:
                raise ValueError(f"mann '{nm}': 'ks_err:' must be > 0.")

            # --- which stellar parameters to constrain ---
            self.constrain.append(
                self._parse_constrain(nm, c.get("constrain"))
            )

            # --- relation form and scatter ---
            uf = bool(c.get("feh", True))
            self.use_feh.append(uf)
            self.mstar_floor.append(
                float(c.get("mstar_floor", physics.MSTAR_FLOOR[uf]))
            )
            self.rstar_floor.append(
                float(c.get("rstar_floor", physics.RSTAR_FLOOR[uf]))
            )

    def register_parameters(self, system):
        """Stage 2: declare the Ks latent and validate the requested pathways."""
        self.manifest = {"ks_offset": None}

        if any(self.ks_synthetic):
            sed = getattr(system, "sed", None)
            if sed is None:
                names = [n for n, s in zip(self.names, self.ks_synthetic) if s]
                raise ValueError(
                    f"mann {names}: 'ks: synthetic' needs a sed: block to "
                    f"predict the synthetic Ks from. Either add one, or give "
                    f"an observed 'ks:'/'ks_err:' instead."
                )
            if not sed.has_filter(KS_FILTER):
                raise ValueError(
                    f"mann: 'ks: synthetic' needs '{KS_FILTER}' in the SED's "
                    f"BC grid, but it is not there. The grid is built from the "
                    f".sed file's filters plus any band-referenced filters, so "
                    f"add a '{KS_FILTER}' row to the .sed file (or a band: "
                    f"block referencing it)."
                )

    def _warn_outside_calibration(self, system):
        """Warn when a star starts outside the relations' calibration ranges.

        EXOFASTv2 checks these every likelihood call, rejecting outright on
        [Fe/H] in the observed-Ks pathway and warning otherwise. A hard
        rejection is a wall with no gradient, which NUTS cannot navigate, so
        these are startup warnings only -- nothing here bounds the posterior
        (see ``StellarRelation._warn_outside_range``).
        """
        self._warn_outside_range(
            system,
            system.star.mass,
            *physics.MSTAR_RANGE,
            message=(
                "star '{star}' starts at "
                "{value:.3f} solMass, outside the Mann+2019 calibration "
                f"range {physics.MSTAR_RANGE} solMass. If it stays there, "
                "prefer MIST/PARSEC/Torres or a direct mass prior."
            ),
        )
        self._warn_outside_range(
            system,
            system.star.feh,
            *physics.FEH_RANGE,
            message=(
                "star '{star}' starts at [Fe/H] = "
                "{value:.3f}, outside the Mann+ calibration range "
                f"{physics.FEH_RANGE} dex."
            ),
        )

    def _ks_source(self, system):
        """Per-instance apparent Ks the relations key on, as an (n_elements,) node."""
        src = []
        for i in range(self.n_elements):
            if self.ks_synthetic[i]:
                src.append(
                    system.sed.predict_star_appmag(
                        self.star_indices[i], KS_FILTER, system
                    )
                )
            else:
                # Explicit float64: pytensor autocasts a bare Python float to
                # the smallest dtype that holds it, so a round magnitude would
                # otherwise become a float32 constant.
                src.append(pt.constant(self.ks_observed[i], dtype="float64"))
        return pt.stack(src)

    def build_likelihood(self, model, system):
        star = system.star
        smap = self.star_map_tensor

        # Deferred to stage 6: the relaxation engine (stage 3) has run by now,
        # so these initvals are the ones the sampler will actually start from.
        self._warn_outside_calibration(system)

        # star.distance is pc, feh dex, mass solMass, radius solRad -- all
        # identity unit conversions, so these are already in the relations'
        # units.
        distance = star.distance.value[smap]
        feh = star.feh.value[smap]

        # 1. Apparent Ks, non-centered about its source (see defaults.yaml).
        ks_err = as_float_vector(self.ks_err)
        appks = self._ks_source(system) + ks_err * self.ks_offset.value
        pm.Deterministic(f"{self.prefix}.appks", appks)

        # 2. Absolute Ks -- the relations' actual input.
        absks = calc_absmag(appks, distance)
        pm.Deterministic(f"{self.prefix}.absks", absks)

        # 3. Relation predictions. The [Fe/H]-dependent form is a per-instance
        # choice, so evaluate one element at a time and stack.
        mass_pred, radius_pred = [], []
        for i in range(self.n_elements):
            f = feh[i] if self.use_feh[i] else None
            mass_pred.append(physics.calc_mann_mass(absks[i], f))
            radius_pred.append(physics.calc_mann_radius(absks[i], f))
        mass_pred = pt.stack(mass_pred)
        radius_pred = pt.stack(radius_pred)
        pm.Deterministic(f"{self.prefix}.mass_pred", mass_pred)
        pm.Deterministic(f"{self.prefix}.radius_pred", radius_pred)

        # 4. Gaussian potentials on the star's own mass/radius nodes. sigma is
        # the relation's *fractional* scatter times its own prediction,
        # matching EXOFASTv2's sigma_mstar = mstar*mstar_floor -- so sigma is a
        # function of the sampled Ks, not a constant, and normalize=True keeps
        # the -log(sigma) term it therefore owes. EXOFASTv2 accumulates chi2
        # only and drops it; the term is worth a few tenths of a nat here, but
        # omitting it would leave the posterior in Ks subtly improper.
        # (components/torres passes normalize=False, and is right to: its
        # scatter is a constant in dex.)
        self._add_penalty(
            "mass",
            star.mass.value[smap],
            mass_pred,
            mass_pred * as_float_vector(self.mstar_floor),
            normalize=True,
        )
        self._add_penalty(
            "radius",
            star.radius.value[smap],
            radius_pred,
            radius_pred * as_float_vector(self.rstar_floor),
            normalize=True,
        )
