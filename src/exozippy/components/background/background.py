"""The unresolved background of a microlensing sightline (review 8.10.3;
background.md).

The light inside a point-spread function that no star block models -- the
stars below a high-resolution catalog's detection limit, summed over the
PSF -- is ONE physical object per sightline, and this component is its one
spelling.  An instance owns

  * ``fluxratio``: its flux in each band of its ``bands:`` list, relative to
    the primary source's baseline flux in that band (one element per
    (instance, band), ``background.<name>.fluxratio_<k>``);
  * ``offset_e``/``offset_n``: its photocenter at the astrometric reference
    epoch relative to the primary source's position (mas);
  * ``pm_ra``/``pm_dec``: that photocenter's proper motion (mas/yr), with a
    field-population prior by default on a bulge sightline.

Modeled stars (the lens, companions, resolved neighbors given their own star
block) are NOT part of it.  A dataset names the instance whose light is in
its PSF (``astrometryinstrument``'s ``background:``) and reads it only
through ``flux_ratio`` and ``track``, the two seams a later component may
build on: the planned resolved-blends follow-up adds a prior on the
per-band fluxes and colors (a luminosity function integrated below the
completeness limit, with Poisson covariance) on the ``fluxratio`` elements,
and lets SED filter rows name the instance, both without changing a name or
a coordinate here.
"""

import logging

import numpy as np

from exozippy.components.component import Component
from exozippy.components.star.position import require_star_radec
from exozippy.constants import (
    BULGE_VELOCITY_SIGMA_2,
    K_VEL_CONVERSION,
    SUN_GC_DISTANCE,
)

logger = logging.getLogger(__name__)

RAD2MAS = 180.0 / np.pi * 3600.0 * 1000.0

# The field prior applies on a bulge sightline: within this box of
# Galactic (l, b) in degrees.  Outside it the pm is uniform on its bounds.
BULGE_BOX_DEG = 10.0


def bulge_field_pm_prior(ra_rad, dec_rad):
    """``(mu, sigma)`` mas/yr per axis (pm_ra*, pm_dec) of the field prior:
    the galactic model's mean proper motion of a bulge star at the Galactic
    center's distance on this sightline, and the bulge velocity dispersion
    there (Koshimoto & Bennett 2020's intermediate axis, 100 km/s) as an
    isotropic sigma.  The mean is ``galacticmodel.physics.
    expected_proper_motion``, the same one the microlensing pm seeds use."""
    from exozippy.components.galacticmodel.physics import (
        expected_proper_motion,
    )

    d_pc = SUN_GC_DISTANCE * 1000.0
    pm_ra, pm_dec, _rv = expected_proper_motion(ra_rad, dec_rad, d_pc, "bulge")
    sigma = BULGE_VELOCITY_SIGMA_2 / (K_VEL_CONVERSION * SUN_GC_DISTANCE)
    return (float(pm_ra), float(pm_dec)), float(sigma)


def on_bulge_sightline(ra_rad, dec_rad):
    """Is (ra, dec) inside the Galactic (l, b) box BULGE_BOX_DEG?"""
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    g = SkyCoord(ra=ra_rad * u.rad, dec=dec_rad * u.rad).galactic
    lon = (g.l.deg + 180.0) % 360.0 - 180.0
    return abs(lon) < BULGE_BOX_DEG and abs(g.b.deg) < BULGE_BOX_DEG


class Background(Component):
    prose_noun = "unresolved background"

    def __init__(self, config, config_manager):
        super().__init__(config, config_manager)
        self.label = "Unresolved Background"
        self.bands = []
        for i, c in enumerate(self.config):
            where = f"[{self.prefix}.{self.names[i]}]"
            bands = c.get("bands")
            if (
                not isinstance(bands, list)
                or not bands
                or not all(isinstance(b, str) for b in bands)
            ):
                raise ValueError(
                    f"{where} `bands:` must list the band names (as the "
                    f"datasets' `band:` keys spell them) this background's "
                    f"flux is sampled in; got {bands!r}."
                )
            if len(set(bands)) != len(bands):
                raise ValueError(
                    f"{where} `bands:` names a band twice: {bands}."
                )
            self.bands.append(list(bands))
        # Element index of (instance, band) in the fluxratio vector.
        self._flux_index = {}
        for i, bands in enumerate(self.bands):
            for b in bands:
                self._flux_index[(i, b)] = len(self._flux_index)

    @property
    def prefix(self):
        return "background"

    @classmethod
    def config_schema(cls):
        return [
            {
                "key": "name",
                "kind": "option",
                "accepts": None,
                "required": False,
                "doc": "Instance name (one per sightline, usually one).",
            },
            {
                "key": "bands",
                "kind": "option",
                "accepts": None,
                "required": True,
                "doc": (
                    "Band names (as the datasets' `band:` keys spell them) "
                    "in which this background's flux relative to the "
                    "source is sampled, background.<name>.fluxratio_<k>, k "
                    "the position in this list."
                ),
            },
        ]

    def register_parameters(self, system):
        event = getattr(system, "mulensevent", None)
        if event is None:
            raise ValueError(
                f"[{self.prefix}] the unresolved background is a "
                f"microlensing sightline's: its fluxes are relative to the "
                f"primary source and its photocenter is anchored on it, and "
                f"the system has no mulensevent: block."
            )
        s_type, s_idx = event.source_bodies[0]
        if s_type != "star":
            raise ValueError(
                f"[{self.prefix}] the primary microlensing source must be a "
                f"star to anchor the unresolved background; it is {s_type}."
            )
        self.source_star = int(s_idx)
        owners = [
            self.names[i] for i, bands in enumerate(self.bands) for _ in bands
        ]
        self.manifest = {
            "offset_e": None,
            "offset_n": None,
            "fluxratio": {
                "shape": (len(owners),),
                "element_owners": owners,
            },
        }
        ra, dec = require_star_radec(
            self.config_manager,
            self.source_star,
            f"[{self.prefix}] the sightline of the field proper-motion prior",
        )
        if on_bulge_sightline(ra, dec):
            (mu_ra, mu_dec), sigma = bulge_field_pm_prior(ra, dec)
            n = self.n_elements
            for key, mu in (("pm_ra", mu_ra), ("pm_dec", mu_dec)):
                self.manifest[key] = {
                    "overrides": {
                        "mu": [mu] * n,
                        "sigma": [sigma] * n,
                        "initval": [mu] * n,
                    }
                }
            logger.info(
                f"[{self.prefix}] field proper-motion prior on "
                f"{self.names}: (pm_ra*, pm_dec) ~ N(({mu_ra:.2f}, "
                f"{mu_dec:.2f}), {sigma:.2f}) mas/yr per axis -- the bulge "
                f"at {SUN_GC_DISTANCE} kpc on this sightline (a params "
                f"entry replaces it)."
            )
        else:
            self.manifest["pm_ra"] = None
            self.manifest["pm_dec"] = None
            logger.warning(
                f"[{self.prefix}] the sightline is outside the bulge box "
                f"(|l|, |b| < {BULGE_BOX_DEG} deg), so {self.names}'s proper "
                f"motion has no field prior: uniform on its bounds.  State "
                f"the field population's mean and dispersion as "
                f"background.<name>.pm_ra/pm_dec {{mu, sigma}}."
            )

    def build_likelihood(self, model, system):
        """No likelihood of its own: the datasets that name it read it."""
        return None

    # ------------------------------------------------------------------
    # The two seams
    # ------------------------------------------------------------------
    def instance_index(self, ref, where):
        """Instance index of a dataset's ``background:`` reference (name or
        index), raising with the known names."""
        if isinstance(ref, bool):
            raise ValueError(f"{where} invalid background reference {ref!r}.")
        if isinstance(ref, int) or str(ref).isdigit():
            k = int(ref)
            if k < self.n_elements:
                return k
        elif ref in self.names:
            return self.names.index(ref)
        raise ValueError(
            f"{where} unknown background {ref!r}; instances are {self.names}."
        )

    def flux_ratio(self, i, band, where):
        """Instance ``i``'s flux in ``band`` relative to the primary
        source's baseline flux there (tensor element).  A band the
        instance does not list RAISES: its flux there is not a parameter."""
        k = self._flux_index.get((i, band))
        if k is None:
            raise ValueError(
                f"{where} reads background '{self.names[i]}' in band "
                f"{band!r}, which its `bands:` ({self.bands[i]}) does not "
                f"list; add it there."
            )
        return self.fluxratio.value[k]

    def track(self, i, source_e, source_n, dt_yr):
        """(dE, dN) of instance ``i``'s photocenter, given the primary
        source's position at the reference epoch (``source_e/n``, in the
        caller's tangent-plane mas) and the time from that epoch in years:
        offset plus proper motion, no parallax of its own."""
        return (
            source_e + self.offset_e.value[i] + self.pm_ra.value[i] * dt_yr,
            source_n + self.offset_n.value[i] + self.pm_dec.value[i] * dt_yr,
        )
