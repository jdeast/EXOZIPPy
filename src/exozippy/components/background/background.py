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
from exozippy.components.orbit.bodies import component_instance_names
from exozippy.components.star.position import require_star_radec
from exozippy.constants import (
    BULGE_VELOCITY_SIGMA_2,
    K_VEL_CONVERSION,
    SUN_GC_DISTANCE,
)

from . import physics  # noqa: F401 -- registers calc_background_fluxratio

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
    # member_flux (one entry per fluxratio element) is injected by
    # add_parameter for the derived from_lightcurve expression.
    context_dep_names = frozenset({"member_flux"})
    aligned_context_deps = frozenset({"member_flux"})

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
        self.psf_stars = []
        for i, c in enumerate(self.config):
            where = f"[{self.prefix}.{self.names[i]}]"
            refs = c.get("psf_stars") or []
            if not isinstance(refs, list):
                raise ValueError(
                    f"{where} psf_stars must be a list of star names or "
                    f"indices; got {refs!r}."
                )
            stars = [
                self.resolve_star_ndx(r, f"{where} psf_stars {r!r}")
                for r in refs
            ]
            if len(set(stars)) != len(stars):
                raise ValueError(
                    f"{where} psf_stars names a star twice: {refs}."
                )
            self.psf_stars.append(stars)
        self._resolve_light_curve_ties()

    def _resolve_light_curve_ties(self):
        """Which fluxratio elements are DERIVED from a light curve.

        A band of an instance's ``bands:`` that a microlensing light curve
        covers (its ``band:``) is tied to it: there the background's flux is
        ``f_blend/f_source`` of that light curve minus the instance's
        ``psf_stars`` (background.md).  ``light_curves:`` names the ones to
        tie; without it every light curve in a listed band is a candidate
        and two in one band RAISE (KMT's three I-band sites each have their
        own blend -- which one the background is, is the user's statement).
        ``light_curves: []`` ties none.  Read from the raw system config, so
        the map exists by stage 2.  Sets ``self._tie[e]`` per fluxratio
        element: a light-curve index or None.
        """
        sys_cfg = getattr(self.config_manager, "system_config", None) or {}
        lc_cfgs = sys_cfg.get("mulensinstrument") or []
        if not isinstance(lc_cfgs, list):
            lc_cfgs = [lc_cfgs]
        lc_names = component_instance_names(sys_cfg, "mulensinstrument")
        lc_bands = [(c or {}).get("band") for c in lc_cfgs]
        self._tie = [None] * len(self._flux_index)
        for i, c in enumerate(self.config):
            where = f"[{self.prefix}.{self.names[i]}]"
            named = c.get("light_curves")
            if named is None:
                candidates = list(range(len(lc_names)))
            else:
                if not isinstance(named, list):
                    raise ValueError(
                        f"{where} light_curves must be a list of "
                        f"mulensinstrument names; got {named!r}."
                    )
                candidates = []
                for ref in named:
                    if ref not in lc_names:
                        raise ValueError(
                            f"{where} light_curves names {ref!r}; the light "
                            f"curves are {lc_names}."
                        )
                    k = lc_names.index(ref)
                    if lc_bands[k] not in self.bands[i]:
                        raise ValueError(
                            f"{where} light curve {ref!r} is in band "
                            f"{lc_bands[k]!r}, which `bands:` "
                            f"({self.bands[i]}) does not list."
                        )
                    candidates.append(k)
            for b in self.bands[i]:
                hits = [k for k in candidates if lc_bands[k] == b]
                if len(hits) > 1:
                    raise ValueError(
                        f"{where} light curves {[lc_names[k] for k in hits]} "
                        f"are all in band {b!r}, and each has its own blend: "
                        f"name the one this background is with "
                        f"`light_curves:` (the others then carry their own "
                        f"f_blend), or `light_curves: []` to tie none."
                    )
                for k in hits:
                    if (lc_cfgs[k] or {}).get("sed_constrains_blend", False):
                        # The same decomposition stated twice (background.md,
                        # "The light-curve tie": not with sed_constrains_blend).
                        raise ValueError(
                            f"{where} light curve {lc_names[k]!r} sets "
                            f"sed_constrains_blend: true, and this background "
                            f"is its blend's remainder in band {b!r}: both "
                            f"say f_blend = (modeled non-source stars) + "
                            f"(unmodeled light), the blend tie through its "
                            f"own neighbor_flux and a sed_blend_sigma "
                            f"Gaussian, the background exactly, with its fluxratio "
                            f">= 0 barrier already the 'blend holds at least "
                            f"the lens' bound.  Drop sed_constrains_blend "
                            f"there, or leave that light curve out of "
                            f"`light_curves:` (the background is then "
                            f"sampled in band {b!r})."
                        )
                if hits:
                    self._tie[self._flux_index[(i, b)]] = hits[0]
        self._lc_names = lc_names

    def build_maps(self):
        # Per fluxratio element, the light curve a derived element reads
        # (0, unread, on a sampled one).
        self.lc_map = np.array(
            [0 if k is None else int(k) for k in self._tie], dtype=int
        )

    def tied_band(self, i, band):
        """Is instance ``i``'s flux in ``band`` derived from a light curve?"""
        return self._tie[self._flux_index[(i, band)]] is not None

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
            {
                "key": "light_curves",
                "kind": "option",
                "accepts": None,
                "required": False,
                "doc": (
                    "mulensinstrument light curves whose blend this "
                    "background is: in each one's band the fluxratio is "
                    "DERIVED, f_blend/f_source minus the psf_stars' SED "
                    "fluxes (needs the SED to predict that band). Default: "
                    "every light curve in a listed band (two in one band "
                    "raise); [] ties none. Not with sed_constrains_blend "
                    "on the same light curve (raises)."
                ),
            },
            {
                "key": "psf_stars",
                "kind": "option",
                "accepts": None,
                "required": False,
                "doc": (
                    "Every modeled non-source star inside the tied light "
                    "curves' PSF -- the lens (always), a luminous lens "
                    "companion, an unlensed companion on a source's orbit, "
                    "a resolved neighbor -- whose SED fluxes the tie "
                    "subtracts; the same list an astrometry dataset reading "
                    "this background in a tied band names. A tied band "
                    "missing a lens body or source-orbit companion raises. "
                    "Default: none."
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
        fluxratio = {"shape": (len(owners),), "element_owners": owners}
        tied = np.array([k is not None for k in self._tie], dtype=bool)
        self._tie_filter = {}
        if tied.any():
            # One spelling, two roles (parameter.md): DERIVED where a light
            # curve covers the band, sampled elsewhere.  Its defaults.yaml
            # bounds become a soft barrier on the derived elements, which
            # is what keeps "the modeled stars exceed the light curve's
            # blend" a pull rather than a NaN (background.md).
            fluxratio["expr_key"] = {"from_lightcurve": tied.copy()}
            self._check_tie_members(system)
            star_names = list(system.star.names)
            for e, k in enumerate(self._tie):
                if k is not None:
                    members = [
                        star_names[s]
                        for s in self.psf_stars[self.names.index(owners[e])]
                    ]
                    logger.info(
                        f"[{self.prefix}] {self.prefix}.{owners[e]}."
                        f"fluxratio_{self._element_band_pos(e)} is DERIVED: "
                        f"f_blend/f_source of light curve "
                        f"'{self._lc_names[k]}' minus psf_stars "
                        f"{members} (their SED fluxes)."
                    )
        self.manifest = {
            "offset_e": None,
            "offset_n": None,
            "fluxratio": fluxratio,
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
                # The prior through "overrides" (under the params file);
                # the start at its center through add_hint, the one
                # channel for a component-computed start.
                self.manifest[key] = {
                    "overrides": {"mu": [mu] * n, "sigma": [sigma] * n}
                }
                for i in range(n):
                    self.config_manager.add_hint(
                        f"{self.prefix}.{i}.{key}", mu
                    )
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

    def _element_band_pos(self, e):
        """Position k of fluxratio element ``e`` in its instance's bands."""
        for (i, b), idx in self._flux_index.items():
            if idx == e:
                return self.bands[i].index(b)
        raise KeyError(e)

    def event_blend_stars(self, system):
        """The modeled stars the EVENT itself puts inside every light
        curve's PSF, so in every tied band's blend (background.md, "Which
        stars").  Every lens body that is a star (the primary, always, and a
        luminous companion: the lens-companion separation is ~theta_E) and
        every star on the far side of a source's own orbit (an unlensed
        xallarap companion: an orbit the light curve resolves in time is far
        below any PSF), less the sources (their light is lensed, the
        f_source half).  A planet is dark here (the SED predicts no planet
        flux).  A star merely bound to a lens or source body in a static or
        RV orbit group, or a resolved neighbor, is NOT on this list: whether
        its light is in a given PSF is the user's statement, `psf_stars:`.
        """
        from exozippy.components.orbit.bodies import parse_orbit_bodies

        event = system.mulensevent
        sources = {int(i) for t, i in event.source_bodies if t == "star"}
        stars = {int(i) for t, i in event.lens_bodies if t == "star"}
        xal = [
            (o, side)
            for o, side in zip(event.source_orbit_idx, event.source_orbit_side)
            if o is not None
        ]
        if xal:
            sys_cfg = self.config_manager.system_config
            prim, comp = parse_orbit_bodies(sys_cfg.get("orbit"), sys_cfg)
            for o, side in xal:
                far = comp[o] if side == 1 else prim[o]
                stars |= {int(i) for t, i in far if t == "star"}
        return sorted(stars - sources)

    def load_data(self, system):
        """No data of its own; stage 1 is where the tie's MEMBERSHIP is
        checked (config only, so before any other component's stage-3
        bookkeeping can trip over a star the tie would have made SEEN)."""
        if getattr(system, "mulensevent", None) is not None:
            self._check_tie_membership(system)

    def _check_tie_membership(self, system):
        """In a tied band the derived flux subtracts exactly the instance's
        psf_stars, so the identity

            F_bg/F_S = f_blend/f_source - sum_{j in psf_stars} F_j/F_S

        holds with the RIGHT stars only if psf_stars holds every star the
        event puts in the blend (``event_blend_stars``) and no source -- both
        RAISE here, naming the star."""
        star_names = list(system.star.names)
        sources = {int(i) for t, i in system.mulensevent.source_bodies}
        required = self.event_blend_stars(system)
        for (i, b), e in self._flux_index.items():
            if self._tie[e] is None:
                continue
            where = f"[{self.prefix}.{self.names[i]}]"
            members = self.psf_stars[i]
            lc = self._lc_names[self._tie[e]]
            named = [star_names[s] for s in members]
            lensed = [star_names[s] for s in members if s in sources]
            if lensed:
                raise ValueError(
                    f"{where} psf_stars names {lensed}, a microlensing "
                    f"SOURCE: its light is the light curve's f_source, not "
                    f"part of its blend."
                )
            missing = [star_names[s] for s in required if s not in members]
            if missing:
                raise ValueError(
                    f"{where} the background's flux in band {b!r} is light "
                    f"curve {lc!r}'s blend minus psf_stars {named}, and "
                    f"{missing} -- a lens body, or an unlensed companion on "
                    f"a source's orbit -- is necessarily in that blend: "
                    f"list it in psf_stars (psf_stars is every modeled "
                    f"non-source star in the PSF, the lens included, as on "
                    f"an astrometry dataset), or leave {lc!r} out of "
                    f"`light_curves:` so the band is sampled."
                )

    def _check_tie_members(self, system):
        """Stage 3: every member's flux in a tied band must be the SED's
        prediction -- a light curve has no psf_fluxratio of its own, and a
        sampled member ratio would be a second, competing statement of the
        same flux.  The lens is always a member (stage 1,
        ``_check_tie_membership``), so a tied band always needs the SED:
        there is no member-free tie to skip."""
        star_names = list(system.star.names)
        for (i, b), e in self._flux_index.items():
            if self._tie[e] is None:
                continue
            where = f"[{self.prefix}.{self.names[i]}]"
            lc = self._lc_names[self._tie[e]]
            named = [star_names[s] for s in self.psf_stars[i]]
            sed = getattr(system, "sed", None)
            band = getattr(system, "band", None)
            fix = (
                f"or leave band {b!r} untied (`light_curves:` without light "
                f"curve {lc!r}), which samples the background's flux there"
            )
            if sed is None or band is None or b not in list(band.names):
                raise ValueError(
                    f"{where} the background's flux in band {b!r} is derived "
                    f"from light curve {lc!r} minus psf_stars {named}, whose "
                    f"fluxes there need the SED and a band: block naming "
                    f"{b!r}; add them, {fix}."
                )
            fk = band.filter_mist[list(band.names).index(b)]
            if not sed.has_filter(fk):
                raise ValueError(
                    f"{where} band {b!r} (filter {fk!r}) is not in the SED's "
                    f"BC grid, so psf_stars {named} have no predicted flux "
                    f"there; {fix}."
                )
            self._tie_filter[e] = fk

    def sed_read_stars(self, system):
        """The stars whose SED flux the light-curve tie reads: the primary
        source and every tied instance's psf_stars (empty with no tie).
        ``SED.photometric_readers`` asks, so a lens only the tie sees is
        SEEN (its SED-side parameters sampled, not pinned)."""
        if not any(k is not None for k in self._tie):
            return []
        # MulensEvent._validate_bodies: every source body is a star.
        out = {int(system.mulensevent.source_bodies[0][1])}
        for (i, _b), e in self._flux_index.items():
            if self._tie[e] is not None:
                out |= set(self.psf_stars[i])
        return sorted(out)

    def add_parameter(self, model, param_name, system, context_nodes=None):
        """Inject ``member_flux`` -- per fluxratio element, the summed SED
        flux of the instance's psf_stars relative to the primary source in
        that element's band (0 where untied) -- for the
        derived ``from_lightcurve`` expression; an SED forward-model node,
        which no manifest dep can spell (the zeropoint idiom)."""
        if param_name == "fluxratio" and not context_nodes:
            context_nodes = {"member_flux": self._member_flux(system)}
        return super().add_parameter(model, param_name, system, context_nodes)

    def _member_flux(self, system):
        import pytensor.tensor as pt

        out = []
        for (i, b), e in sorted(self._flux_index.items(), key=lambda t: t[1]):
            fk = self._tie_filter.get(e)
            if fk is None:
                out.append(pt.constant(0.0, dtype="float64"))
                continue
            sed = system.sed
            needed = [self.source_star] + list(self.psf_stars[i])
            if not sed.can_predict(needed):
                raise ValueError(
                    f"[{self.prefix}.{self.names[i]}] the SED severed one of "
                    f"stars {needed} (no BC grid hosts its start; sed.md, "
                    f"the no-grid fallback), so the light-curve tie cannot "
                    f"subtract psf_stars in band {b!r}; leave that light "
                    f"curve out of `light_curves:` (the band is then "
                    f"sampled)."
                )
            m_s = sed.predict_star_appmag(self.source_star, fk, system)
            total = 0.0
            for s in self.psf_stars[i]:
                total = total + 10 ** (
                    -0.4 * (sed.predict_star_appmag(s, fk, system) - m_s)
                )
            out.append(pt.as_tensor_variable(total))
        return pt.stack(out)

    def build_likelihood(self, model, system):
        """No likelihood of its own: the datasets that name it read it, and
        the light-curve tie is the derived expression itself."""
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
