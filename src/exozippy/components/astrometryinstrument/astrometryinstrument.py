"""
Astrometry instrument component.

Supports three data modes (set per instrument via 'mode' in the config):

  gaia : 1-D epoch astrometry (Gaia-like along-scan abscissae).
         File columns: time[BJD_TDB]  w[mas]  err[mas]  scan_pa[deg]
         w is the along-scan coordinate of the source relative to the
         reference position (the star's ra/dec initval at the reference
         epoch), containing the full proper motion + parallax + orbital
         signal.  The scan position angle is measured East of North, so
         w = dE*sin(psi) + dN*cos(psi).

  abs  : ground/space-based 2-D absolute astrometry.
         File columns: time[BJD_TDB]  ra[deg]  dec[deg]  err_ra[mas]  err_dec[mas]
         err_ra is the true-arc (ra*cos(dec)) uncertainty.  Positions are
         converted at load time to (dE, dN) offsets in mas from the
         reference position (small-angle approximation).

  rel  : relative astrometry of a companion with respect to its host.
         File columns: time[BJD_TDB]  sep[mas]  err_sep[mas]  pa[deg]  err_pa[deg]
         The position angle is measured East of North, of the orbit's
         COMPANION group relative to its PRIMARY group.  Use 'orbit'
         (orbit instance name or index) to select which orbit the
         measurements trace; 'planet_ndx' remains as a legacy alias
         (resolved through that planet's orbit_ndx).  When a group holds
         several bodies, the modeled position is the group photocenter:
         the barycenter orbit of the referenced orbit, plus the
         photocenter wobble of any sub-orbit nested inside a group
         (flux-weighted through the SED when 'band' is given; a dark
         companion needs no SED; otherwise the photocenter is
         approximated by the barycenter with a warning).

Per-instrument config keys:
  file              : data file (whitespace-separated, '#' comments)
  mode              : gaia | abs | rel   (default gaia)
  observer_location : ephemeris for parallax factors (default 'gaia' for
                      gaia mode, 'earth' otherwise; see exozippy.ephemeris)
  star_ndx          : index of the host star (default 0); also sets the
                      parallax used to scale rel-mode separations
  orbit             : rel mode, orbit instance name (or index) the
                      measurements trace
  planet_ndx        : rel mode legacy alias for 'orbit' (default 0)
  epoch             : reference epoch [BJD_TDB] for ra/dec/pm (default:
                      mean time of all gaia/abs observations)
  sep_unit          : rel mode separation unit (default 'mas')
  band              : name of a band: block (filter identity for the
                      SED-derived fluxfrac below)
  companion_star_ndx: index of the star modeling the luminous companion.
                      When band, companion_star_ndx, and a sed: block are
                      all present, the photocenter flux fraction is
                      derived from the SED (beta = F_c/(F_c+F_host) in
                      the band) instead of being sampled; the sampled
                      fluxfrac element is fixed (unused).

Conventions follow EXOFASTv2: omega is the argument of periastron of the
primary's orbit (omega_*); bigomega is the position angle of the ascending
node (East of North), where the ascending node is the node at which the
body recedes from the observer.  See Orbit.get_sky_position.  Without
RVs, (bigomega, omega) <-> (bigomega+180, omega+180) is exactly
degenerate for astrometry of every kind (which node is ascending is
unknowable from sky-plane data); this is handled by restricting bigomega
to [0, 180] (Orbit._restrict_bigomega_halfplane), with a table note
documenting the artificial boundary.

Model limitations (v1): the gaia/abs photocenter wobble sums over the
orbits whose PRIMARY group contains the target star, with a single
per-instrument companion flux fraction (fluxfrac, default 0 = dark
companions); orbits holding the star in their companion group are not yet
modeled there (warned).  Relative positions are Keplerian two-body per
orbit plus one level of nested photocenter wobble; light travel time and
aberration are neglected.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

import astropy.units as u
import matplotlib.pyplot as plt

import pymc as pm
import pytensor
import pytensor.tensor as pt

from exozippy.components.component import Component
from exozippy.components.orbit.bodies import component_instance_names
from exozippy.ephemeris import get_observer_position
from . import physics

RAD2MAS = (1.0 * u.rad).to(u.mas).value          # 2.06264806e8
RSUN_AU = (1.0 * u.solRad).to(u.AU).value        # 4.6505e-3
DAYS_PER_YEAR = 365.25

VALID_MODES = ("gaia", "abs", "rel")


class AstrometryInstrument(Component):
    def __init__(self, config, config_manager):
        super().__init__(config, config_manager)
        self.label = "Astrometry Parameters"
        self.files = [c.get("file") for c in self.config]
        self.modes = [c.get("mode", "gaia") for c in self.config]
        for m in self.modes:
            if m not in VALID_MODES:
                raise ValueError(
                    f"astrometryinstrument mode '{m}' not recognized; "
                    f"must be one of {VALID_MODES}")
        self.observers = [
            c.get("observer_location", "gaia" if c.get("mode", "gaia") == "gaia" else "earth")
            for c in self.config
        ]

        # rel mode: which orbit the measurements trace (name or index in
        # 'orbit'; legacy 'planet_ndx' resolves through the planet's
        # orbit_ndx).
        sys_cfg = getattr(config_manager, "system_config", None) or {}
        orbit_names = component_instance_names(sys_cfg, "orbit")
        planet_cfgs = sys_cfg.get("planet") or []
        if not isinstance(planet_cfgs, list):
            planet_cfgs = [planet_cfgs]
        self.rel_orbit = []
        for i, c in enumerate(self.config):
            if c.get("mode", "gaia") != "rel":
                self.rel_orbit.append(None)
                continue
            ref = c.get("orbit")
            if ref is not None:
                if isinstance(ref, int) or str(ref).isdigit():
                    idx = int(ref)
                    if orbit_names and idx >= len(orbit_names):
                        raise ValueError(
                            f"[{self.prefix}.{self.names[i]}] orbit index "
                            f"{idx} out of range; orbits are {orbit_names}.")
                elif ref in orbit_names:
                    idx = orbit_names.index(ref)
                else:
                    raise ValueError(
                        f"[{self.prefix}.{self.names[i]}] unknown orbit "
                        f"'{ref}'; orbits are {orbit_names}.")
            else:
                j = int(c.get("planet_ndx", 0))
                p_cfg = planet_cfgs[j] if j < len(planet_cfgs) else {}
                # same default as Planet.build_maps' orbit_map
                idx = int((p_cfg or {}).get("orbit_ndx", 0))
            self.rel_orbit.append(idx)

    @property
    def prefix(self):
        return "astrometryinstrument"

    @classmethod
    def config_schema(cls):
        return [
            {
                "key": "file",
                "kind": "datafile",
                "accepts": "*.dat",
                "required": True,
                "doc": (
                    "Whitespace-delimited astrometry data. Column layout "
                    "depends on mode (gaia/abs absolute positions vs rel "
                    "relative separations)."
                ),
            },
            {
                "key": "mode",
                "kind": "option",
                "accepts": ["gaia", "abs", "rel"],
                "required": False,
                "doc": (
                    "Astrometry data mode: 'gaia'/'abs' absolute positions "
                    "or 'rel' relative separations. Default 'gaia'."
                ),
            },
            {
                "key": "observer_location",
                "kind": "option",
                "accepts": None,
                "required": False,
                "doc": (
                    "Observer location. Defaults to 'gaia' in gaia mode, "
                    "else 'earth'."
                ),
            },
            {
                "key": "star_ndx",
                "kind": "ref",
                "accepts": ["star"],
                "required": False,
                "doc": "Index or name of the target star. Default 0.",
            },
            {
                "key": "orbit",
                "kind": "ref",
                "accepts": ["orbit"],
                "required": False,
                "doc": (
                    "rel mode: name of the orbit modeled relative to the "
                    "primary group (the companion group is modeled relative "
                    "to the primary)."
                ),
            },
            {
                "key": "planet_ndx",
                "kind": "ref",
                "accepts": ["planet"],
                "required": False,
                "doc": (
                    "Legacy rel-mode orbit reference by planet index; "
                    "prefer the 'orbit' key."
                ),
            },
            {
                "key": "band",
                "kind": "ref",
                "accepts": ["band"],
                "required": False,
                "doc": (
                    "Band used to derive the SED-weighted photocenter flux "
                    "fraction (with companion_star_ndx)."
                ),
            },
            {
                "key": "companion_star_ndx",
                "kind": "ref",
                "accepts": ["star"],
                "required": False,
                "doc": (
                    "Index or name of the companion star for the SED-derived "
                    "photocenter flux fraction (with band)."
                ),
            },
            {
                "key": "sep_unit",
                "kind": "option",
                "accepts": None,
                "required": False,
                "doc": "Astropy unit string for rel-mode separations. Default 'mas'.",
            },
            {
                "key": "epoch",
                "kind": "option",
                "accepts": None,
                "required": False,
                "doc": "Reference epoch for the astrometric positions.",
            },
        ]

    # ------------------------------------------------------------------
    # Stage 1a
    # ------------------------------------------------------------------
    def load_data(self, system):
        """Load per-instrument astrometry and precompute parallax factors."""
        self.datasets = []
        self.jittervar_lower = [0.0] * self.n_elements
        self.n_total_obs = 0

        n_stars = system.star.n_elements if hasattr(system, "star") else 1
        ra_cfg = self.config_manager.resolve("star", "ra", shape=(n_stars,))["initval"]
        dec_cfg = self.config_manager.resolve("star", "dec", shape=(n_stars,))["initval"]
        ra_cfg = np.atleast_1d(ra_cfg)
        dec_cfg = np.atleast_1d(dec_cfg)

        for i, file in enumerate(self.files):
            mode = self.modes[i]
            df = pd.read_csv(file, sep=r"\s+", engine="c", header=None, comment="#")
            t = df.iloc[:, 0].values.astype(float)

            star_ndx = int(self.config[i].get("star_ndx", 0))
            ra_ref = float(ra_cfg[star_ndx]) * np.pi / 180.0     # rad
            dec_ref = float(dec_cfg[star_ndx]) * np.pi / 180.0   # rad

            d = {
                "name": self.names[i],
                "mode": mode,
                "time": t,
                "star_ndx": star_ndx,
                "ra_ref": ra_ref,
                "dec_ref": dec_ref,
            }

            if mode == "gaia":
                d["w"] = df.iloc[:, 1].values.astype(float)          # mas
                d["err"] = df.iloc[:, 2].values.astype(float)        # mas
                psi = df.iloc[:, 3].values.astype(float) * np.pi / 180.0
                d["sin_psi"] = np.sin(psi)
                d["cos_psi"] = np.cos(psi)
                min_err = np.min(d["err"])
            elif mode == "abs":
                ra_obs = df.iloc[:, 1].values.astype(float) * np.pi / 180.0
                dec_obs = df.iloc[:, 2].values.astype(float) * np.pi / 180.0
                # Small-angle offsets from the reference position, in mas
                d["dE_obs"] = (ra_obs - ra_ref) * np.cos(dec_ref) * RAD2MAS
                d["dN_obs"] = (dec_obs - dec_ref) * RAD2MAS
                d["err_E"] = df.iloc[:, 3].values.astype(float)      # mas
                d["err_N"] = df.iloc[:, 4].values.astype(float)      # mas
                min_err = min(np.min(d["err_E"]), np.min(d["err_N"]))
            else:  # rel
                factor = u.Unit(self.config[i].get("sep_unit", "mas")).to(u.mas)
                d["sep"] = df.iloc[:, 1].values.astype(float) * factor
                d["err_sep"] = df.iloc[:, 2].values.astype(float) * factor
                d["pa"] = df.iloc[:, 3].values.astype(float) * np.pi / 180.0
                d["err_pa"] = df.iloc[:, 4].values.astype(float) * np.pi / 180.0
                min_err = np.min(d["err_sep"])

            # Parallax factors (needed for gaia/abs only)
            if mode in ("gaia", "abs"):
                xyz = get_observer_position(t, observer_location=self.observers[i])
                X, Y, Z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
                # Apparent displacement of the source = parallax * (P_E, P_N)
                d["P_E"] = X * np.sin(ra_ref) - Y * np.cos(ra_ref)
                d["P_N"] = (X * np.cos(ra_ref) * np.sin(dec_ref)
                            + Y * np.sin(ra_ref) * np.sin(dec_ref)
                            - Z * np.cos(dec_ref))

            self.jittervar_lower[i] = -0.95 * min_err ** 2
            self.n_total_obs += len(t)
            self.datasets.append(d)

        # Reference epoch for ra/dec/pm: first explicit config wins, else the
        # mean time of all gaia/abs observations (arbitrary but harmless for
        # rel-only data, where it is unused).
        epochs = [c.get("epoch") for c in self.config if c.get("epoch") is not None]
        if epochs:
            self.epoch = float(epochs[0])
        else:
            t_all = [d["time"] for d in self.datasets if d["mode"] in ("gaia", "abs")]
            self.epoch = float(np.mean(np.concatenate(t_all))) if t_all else 0.0
        logger.info(f"[{self.prefix}] reference epoch for ra/dec/pm: {self.epoch:.4f}")

    # ------------------------------------------------------------------
    # Stage 1b
    # ------------------------------------------------------------------
    def build_maps(self):
        self.star_map = np.array([c.get("star_ndx", 0) for c in self.config])
        self.planet_map = np.array([c.get("planet_ndx", 0) for c in self.config])

    # ------------------------------------------------------------------
    # Stage 2
    # ------------------------------------------------------------------
    def register_parameters(self, system):
        self.manifest = {
            "jitter_variance": {"lower": self.jittervar_lower},
            "jitter": "default",
            "fluxfrac": None,
        }

        # SED-derived fluxfrac: instruments with band + companion_star_ndx
        # in a system with a sed: block get their photocenter flux
        # fraction from the SED (see _sed_beta_node). Fix the sampled
        # fluxfrac element (it is unused) unless the user already
        # configured it.
        self._sed_fluxfrac = [False] * self.n_elements
        if "sed" in (self.config_manager.system_config or {}):
            for i, c in enumerate(self.config):
                if c.get("band") is None or c.get("companion_star_ndx") is None:
                    continue
                self._sed_fluxfrac[i] = True
                key = f"{self.prefix}.{i}.fluxfrac"
                existing = self.config_manager.user_params.get(key)
                if existing is None:
                    self.config_manager.user_params[key] = {"sigma": 0}
                elif isinstance(existing, dict):
                    existing.setdefault("sigma", 0)

    # ------------------------------------------------------------------
    # Model pieces (PyTensor)
    # ------------------------------------------------------------------
    def _sed_beta_node(self, system, i):
        """
        Photocenter flux fraction for instrument i: the SED-predicted
        beta = F_companion / (F_companion + F_host) in the instrument's
        band when configured (see module docstring), else the sampled
        fluxfrac element. Falls back to the sampled parameter with a
        warning if the band's filter is missing from the SED's BC grid.
        """
        if not getattr(self, "_sed_fluxfrac", [False] * self.n_elements)[i]:
            return self.fluxfrac.value[i]

        sed = system.sed
        band_name = self.config[i].get("band")
        band_names = list(system.band.names) if hasattr(system, "band") else []
        if band_name not in band_names:
            raise ValueError(
                f"astrometryinstrument {self.names[i]} references unknown "
                f"band '{band_name}'. Available bands: {band_names}.")
        filter_key = system.band.filter_mist[band_names.index(band_name)]
        if not sed.has_filter(filter_key):
            logger.warning(
                f"astrometryinstrument {self.names[i]}: band filter "
                f"'{filter_key}' is not in the SED's BC grid; using the "
                f"sampled fluxfrac (fixed at its initval).")
            return self.fluxfrac.value[i]

        host = int(self.star_map[i])
        comp = int(self.config[i]["companion_star_ndx"])
        F_c = 10 ** (-0.4 * sed.predict_star_appmag(comp, filter_key, system))
        F_h = 10 ** (-0.4 * sed.predict_star_appmag(host, filter_key, system))
        return F_c / (F_c + F_h)

    def _photocenter_terms(self, system, star_idx, beta):
        """
        Photocenter wobble of one star for gaia/abs data: per-orbit
        semimajor axes in mas, a_phot = a_rel*(M_c/M_tot - beta)*plx, over
        the orbits whose primary group contains the star.  Returns
        (a_phot vector node, orbit index array), or (None, None) when no
        orbit moves this star.
        """
        orbits = getattr(system, "orbit", None)
        if orbits is None:
            return None, None
        members = orbits.star_membership(star_idx)
        if any(role == "companion" for _, role in members):
            names = [orbits.names[o] for o, role in members
                     if role == "companion"]
            logger.warning(
                f"[{self.prefix}] star {star_idx} is in the COMPANION "
                f"group of orbit(s) {names}; that photocenter contribution "
                f"is not modeled for gaia/abs data yet.")
        prim = [o for o, role in members if role == "primary"]
        if not prim:
            return None, None
        if not hasattr(orbits, "arsun"):
            raise ValueError(
                f"[{self.prefix}] astrometry needs the orbit scale "
                f"parameters (arsun/masses), but the orbit component's "
                f"body groups did not resolve against the active system.")
        omap = np.asarray(prim, dtype=int)
        mass_frac = orbits.m_companion.value[omap] / orbits.m_total.value[omap]
        plx = system.star.parallax.value[star_idx]
        return orbits.arsun.value[omap] * RSUN_AU * (mass_frac - beta) * plx, omap

    def _pair_beta(self, system, i, o2):
        """
        Flux fraction beta = F_companion/(F_companion + F_primary) among
        the bodies of orbit o2, in instrument i's band.  Planets are dark.
        Returns None when both sides are luminous but no SED/band is
        available to weigh them (caller approximates photocenter by
        barycenter).
        """
        orbits = system.orbit
        comp_stars = [idx for (t, idx) in orbits.companion_bodies[o2] if t == "star"]
        prim_stars = [idx for (t, idx) in orbits.primary_bodies[o2] if t == "star"]
        if not comp_stars and not prim_stars:
            return None
        if not comp_stars:
            return 0.0
        if not prim_stars:
            return 1.0

        band_name = self.config[i].get("band")
        sed = getattr(system, "sed", None)
        if band_name is not None and sed is not None:
            band_names = list(system.band.names) if hasattr(system, "band") else []
            if band_name not in band_names:
                raise ValueError(
                    f"astrometryinstrument {self.names[i]} references "
                    f"unknown band '{band_name}'. Available: {band_names}.")
            filter_key = system.band.filter_mist[band_names.index(band_name)]
            if sed.has_filter(filter_key):
                def group_flux(stars):
                    F = 0.0
                    for s_idx in stars:
                        F = F + 10 ** (-0.4 * sed.predict_star_appmag(
                            s_idx, filter_key, system))
                    return F
                F_c = group_flux(comp_stars)
                F_p = group_flux(prim_stars)
                return F_c / (F_c + F_p)
            logger.warning(
                f"astrometryinstrument {self.names[i]}: band filter "
                f"'{filter_key}' is not in the SED's BC grid; "
                f"approximating the photocenter of orbit "
                f"{orbits.names[o2]} by its barycenter.")
            return None
        logger.warning(
            f"astrometryinstrument {self.names[i]}: both sides of nested "
            f"orbit {orbits.names[o2]} are luminous but no band/sed is "
            f"configured; approximating its photocenter by its barycenter.")
        return None

    def _rel_model(self, system, i, t_node):
        """
        (dE, dN) of instrument i's rel-mode target: the companion group of
        the referenced orbit relative to its primary group, in mas.  Each
        group's position is its photocenter: the two-body barycenter arc,
        corrected by the photocenter wobble of any orbit fully nested in a
        group, r_phot - r_bary = (beta - M_c/M_tot) * r_rel.
        """
        orbits = system.orbit
        o = self.rel_orbit[i]
        if not hasattr(orbits, "arsun"):
            raise ValueError(
                f"[{self.prefix}.{self.names[i]}] relative astrometry "
                f"needs the orbit scale parameters (arsun/masses), but the "
                f"orbit component's body groups did not resolve against "
                f"the active system.")
        plx = system.star.parallax.value[self.star_map[i]]
        a_rel = orbits.arsun.value[o] * RSUN_AU * plx
        dE_rel, dN_rel = orbits.get_sky_position(
            t_node, pt.stack([a_rel]), np.array([o]), relative=True)
        dE, dN = dE_rel[:, 0], dN_rel[:, 0]

        comp_set = set(orbits.companion_bodies[o])
        prim_set = set(orbits.primary_bodies[o])
        for o2 in range(orbits.n_elements):
            if o2 == o:
                continue
            bodies2 = set(orbits.bodies(o2))
            for group, sgn in ((comp_set, 1.0), (prim_set, -1.0)):
                if len(group) < 2 or not bodies2 <= group:
                    continue
                beta = self._pair_beta(system, i, o2)
                if beta is None:
                    continue
                mfrac = (orbits.m_companion.value[o2]
                         / orbits.m_total.value[o2])
                amp = (orbits.arsun.value[o2] * RSUN_AU * plx
                       * (beta - mfrac) * sgn)
                dE2, dN2 = orbits.get_sky_position(
                    t_node, pt.stack([amp]), np.array([o2]), relative=True)
                dE = dE + dE2[:, 0]
                dN = dN + dN2[:, 0]
        return dE, dN

    def _absolute_model(self, system, d, t, beta):
        """(dE, dN) model in mas relative to the reference position."""
        star = system.star
        s = d["star_ndx"]
        dt_yr = (t - self.epoch) / DAYS_PER_YEAR

        dE = ((star.ra.value[s] - d["ra_ref"]) * np.cos(d["dec_ref"]) * RAD2MAS
              + star.pm_ra.value[s] * dt_yr
              + star.parallax.value[s] * d["P_E"])
        dN = ((star.dec.value[s] - d["dec_ref"]) * RAD2MAS
              + star.pm_dec.value[s] * dt_yr
              + star.parallax.value[s] * d["P_N"])

        a_phot, omap = self._photocenter_terms(system, s, beta)
        if a_phot is not None:
            dE_orb, dN_orb = system.orbit.get_sky_position(
                pt.as_tensor_variable(t), a_phot, omap)
            dE = dE + pt.sum(dE_orb, axis=1)
            dN = dN + pt.sum(dN_orb, axis=1)

        return dE, dN

    # ------------------------------------------------------------------
    # Stage 6
    # ------------------------------------------------------------------
    def build_likelihood(self, model, system):
        if not hasattr(system, "star"):
            raise ValueError(f"[{self.prefix}] astrometry requires a star component.")
        if any(m == "rel" for m in self.modes) and not hasattr(system, "orbit"):
            raise ValueError(f"[{self.prefix}] relative astrometry requires an orbit component.")

        for i, d in enumerate(self.datasets):
            name = d["name"]
            mode = d["mode"]
            t = d["time"]
            jv = self.jitter_variance.value[i]
            beta = self._sed_beta_node(system, i)
            if self._sed_fluxfrac[i]:
                pm.Deterministic(f"{self.prefix}.{name}.fluxfrac_sed", beta)

            if mode in ("gaia", "abs"):
                dE, dN = self._absolute_model(system, d, t, beta)

            if mode == "gaia":
                w_model = dE * d["sin_psi"] + dN * d["cos_psi"]
                sigma = pt.sqrt(pt.sqr(pm.Data(f"{self.prefix}.{name}_err", d["err"])) + jv)
                pm.Normal(
                    f"{self.prefix}.model_{name}",
                    mu=w_model,
                    sigma=sigma,
                    observed=pm.Data(f"{self.prefix}.{name}_w", d["w"]),
                )

            elif mode == "abs":
                sigma_E = pt.sqrt(pt.sqr(pm.Data(f"{self.prefix}.{name}_errE", d["err_E"])) + jv)
                sigma_N = pt.sqrt(pt.sqr(pm.Data(f"{self.prefix}.{name}_errN", d["err_N"])) + jv)
                pm.Normal(
                    f"{self.prefix}.model_{name}_E",
                    mu=dE, sigma=sigma_E,
                    observed=pm.Data(f"{self.prefix}.{name}_dE", d["dE_obs"]),
                )
                pm.Normal(
                    f"{self.prefix}.model_{name}_N",
                    mu=dN, sigma=sigma_N,
                    observed=pm.Data(f"{self.prefix}.{name}_dN", d["dN_obs"]),
                )

            else:  # rel
                dEj, dNj = self._rel_model(system, i, pt.as_tensor_variable(t))

                sep_model = pt.sqrt(pt.sqr(dEj) + pt.sqr(dNj))
                pa_model = pt.arctan2(dEj, dNj)

                sigma_sep = pt.sqrt(
                    pt.sqr(pm.Data(f"{self.prefix}.{name}_errsep", d["err_sep"])) + jv)
                pm.Normal(
                    f"{self.prefix}.model_{name}_sep",
                    mu=sep_model, sigma=sigma_sep,
                    observed=pm.Data(f"{self.prefix}.{name}_sep", d["sep"]),
                )

                # Wrapped position-angle residual: jitter maps to PA as
                # (jitter / sep)^2, using the observed separation for stability.
                delta = pa_model - d["pa"]
                wrapped = pt.arctan2(pt.sin(delta), pt.cos(delta))
                sigma_pa = pt.sqrt(d["err_pa"] ** 2 + jv / d["sep"] ** 2)
                pm.Normal(
                    f"{self.prefix}.model_{name}_pa",
                    mu=wrapped, sigma=sigma_pa,
                    observed=np.zeros(len(t)),
                )

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def compile_plotters(self, model, system):
        """Compile fast PyTensor functions for plotting."""
        if not hasattr(system, "orbit") or not hasattr(system.orbit, "arsun"):
            return

        param_symbols = [p.value for p in system.plot_params]
        t_input = pt.vector("t_input")

        # Photocenter orbit (summed over the star's member orbits), per
        # gaia/abs instrument (beta varies); None when nothing moves the star.
        self._compiled_photo = []
        for i in range(self.n_elements):
            if self.modes[i] == "rel":
                self._compiled_photo.append(None)
                continue
            beta = self._sed_beta_node(system, i)
            a_phot, omap = self._photocenter_terms(
                system, self.datasets[i]["star_ndx"], beta)
            if a_phot is None:
                self._compiled_photo.append(None)
                continue
            dE_orb, dN_orb = system.orbit.get_sky_position(
                t_input, a_phot, omap)
            self._compiled_photo.append(pytensor.function(
                inputs=[t_input] + param_symbols,
                outputs=[pt.sum(dE_orb, axis=1), pt.sum(dN_orb, axis=1)],
                on_unused_input="ignore",
            ))

        # Relative model per rel instrument (nested photocenter terms
        # included; matches the likelihood graph exactly)
        self._compiled_rel = []
        for i in range(self.n_elements):
            if self.modes[i] != "rel":
                self._compiled_rel.append(None)
                continue
            dE, dN = self._rel_model(system, i, t_input)
            self._compiled_rel.append(pytensor.function(
                inputs=[t_input] + param_symbols,
                outputs=[dE, dN],
                on_unused_input="ignore",
            ))

        # Orbital elements in internal units (all orbits), for the node/
        # direction annotations of the sky plot.
        orb = system.orbit
        self._compiled_elements = pytensor.function(
            inputs=param_symbols,
            outputs=[orb.tp.value, orb.n.value,
                     orb.ecc.value, orb.omega.value, orb.bigomega.value],
            on_unused_input="ignore",
        )

    def _point_values(self, system, point):
        vals = []
        for p in system.plot_params:
            val = np.asarray(point.get(p.label, p.initval), dtype=np.float64)
            if getattr(p.value, "ndim", 0) == 0:
                vals.append(float(np.squeeze(val)))
            else:
                vals.append(np.atleast_1d(val))
        return vals

    def _eval_photo(self, i, t, vals):
        """Photocenter orbit at times t for instrument i; zeros when no
        orbit moves the instrument's star."""
        fn = getattr(self, "_compiled_photo", None)
        fn = fn[i] if fn is not None else None
        if fn is None:
            z = np.zeros(len(t), dtype=float)
            return z, z.copy()
        return fn(np.asarray(t, dtype=np.float64), *vals)

    def _linear_terms(self, d, t, point, system):
        """Numpy pm+parallax+offset model (mas) at the reference point."""
        star = system.star
        s = d["star_ndx"]
        dt_yr = (t - self.epoch) / DAYS_PER_YEAR

        def get(label, default):
            return np.atleast_1d(point.get(label, default))[s]

        ra = get(star.ra.label, d["ra_ref"])
        dec = get(star.dec.label, d["dec_ref"])
        pm_ra = get(star.pm_ra.label, 0.0)
        pm_dec = get(star.pm_dec.label, 0.0)
        # parallax is a derived parameter and is usually absent from
        # posterior draws; falling back to 0 silently removed the parallax
        # wiggles from the plots.  Recover it from the sampled distance.
        plx = point.get(star.parallax.label)
        if plx is not None:
            plx = np.atleast_1d(plx)[s]
        else:
            dist = get(star.distance.label,
                       np.atleast_1d(star.distance.initval)[s])
            plx = 1000.0 / dist

        dE = (ra - d["ra_ref"]) * np.cos(d["dec_ref"]) * RAD2MAS + pm_ra * dt_yr + plx * d["P_E"]
        dN = (dec - d["dec_ref"]) * RAD2MAS + pm_dec * dt_yr + plx * d["P_N"]
        return dE, dN

    def plot(self, system, points, filename_prefix="debug"):
        if not hasattr(self, "_compiled_photo") and not hasattr(self, "_compiled_rel"):
            return
        if isinstance(points, dict):
            points = [points]
        if len(points) == 0:
            logger.warning("No points provided for plotting.")
            return
        ref_point = points[0]

        for i, d in enumerate(self.datasets):
            t = d["time"]
            t_pretty = np.linspace(t.min(), t.max(), 2000)
            plt.figure(figsize=(12, 6))

            if d["mode"] == "gaia":
                # Along-scan residuals about the linear (pm+plx) model
                dE_lin, dN_lin = self._linear_terms(d, t, ref_point, system)
                for idx, point in enumerate(points):
                    vals = self._point_values(system, point)
                    dE_orb, dN_orb = self._eval_photo(i, t, vals)
                    w_model = ((dE_lin + dE_orb) * d["sin_psi"]
                               + (dN_lin + dN_orb) * d["cos_psi"])
                    alpha = 0.8 if len(points) == 1 else 0.1
                    plt.plot(t, w_model, "r.", alpha=alpha, zorder=2)
                plt.errorbar(t, d["w"], yerr=d["err"], fmt="o", alpha=0.6, zorder=1)
                plt.xlabel("Time [BJD_TDB]")
                plt.ylabel("Along-scan position [mas]")
                plt.title(f"Epoch astrometry: {d['name']} ({system.name})")

            elif d["mode"] == "abs":
                dE_lin, dN_lin = self._linear_terms(d, t, ref_point, system)
                vals = self._point_values(system, ref_point)
                dE_orb, dN_orb = self._eval_photo(i, t, vals)
                plt.errorbar(d["dE_obs"] - dE_lin, d["dN_obs"] - dN_lin,
                             xerr=d["err_E"], yerr=d["err_N"], fmt="o",
                             alpha=0.6, zorder=1, label="data - (pm+plx)")
                t_lin_pretty = np.linspace(t.min(), t.max(), 2000)
                dE_p, dN_p = self._eval_photo(i, t_lin_pretty, vals)
                plt.plot(dE_p, dN_p, "r-", lw=1, zorder=2, label="photocenter orbit")
                plt.gca().invert_xaxis()  # East to the left
                plt.gca().set_aspect("equal", adjustable="datalim")
                plt.xlabel(r"$\Delta\alpha^*$ [mas]")
                plt.ylabel(r"$\Delta\delta$ [mas]")
                plt.title(f"Absolute astrometry: {d['name']} ({system.name})")
                plt.legend(loc="best", fontsize="small")

            else:  # rel
                vals = self._point_values(system, ref_point)
                dE_m, dN_m = self._compiled_rel[i](t_pretty.astype(np.float64), *vals)
                plt.errorbar(d["sep"] * np.sin(d["pa"]), d["sep"] * np.cos(d["pa"]),
                             fmt="o", alpha=0.6, zorder=1, label="data")
                plt.plot(dE_m, dN_m, "r-", lw=1, zorder=2, label="model")
                plt.plot(0, 0, "k*", markersize=12)
                plt.gca().invert_xaxis()
                plt.gca().set_aspect("equal", adjustable="datalim")
                plt.xlabel(r"$\Delta\alpha^*$ [mas]")
                plt.ylabel(r"$\Delta\delta$ [mas]")
                plt.title(f"Relative astrometry: {d['name']} ({system.name})")
                plt.legend(loc="best", fontsize="small")

            plt.tight_layout()
            plt.savefig(f"{filename_prefix}_astrometry_{d['name']}.pdf")
            plt.close()

            if d["mode"] in ("gaia", "abs"):
                self.plot_sky(system, ref_point, i, filename_prefix)

    def plot_sky(self, system, point, i, filename_prefix="debug"):
        """
        Two-panel sky plot for an absolute-astrometry instrument, in the
        style of El-Badry et al. (2023) Figure 3, but with East to the LEFT
        (standard on-sky orientation; their Delta-RA axis increases to the
        right).

        Left: the full path of the photocenter on the sky (proper motion +
        parallax + orbit) over the data span, with the model position at
        each observation epoch.
        Right: the photocenter orbit alone, with the barycenter, the line
        of nodes, the ascending node (where the photocenter recedes from
        the observer), and the direction of motion.
        """
        d = self.datasets[i]
        t = d["time"]
        vals = self._point_values(system, point)
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 6))

        # ---------------- left: full path over the data span ----------------
        t_dense = np.linspace(t.min(), t.max(), 4000)
        # parallax factors on the dense grid (numpy, same observer)
        xyz = get_observer_position(t_dense, observer_location=self.observers[i])
        P_E = xyz[:, 0] * np.sin(d["ra_ref"]) - xyz[:, 1] * np.cos(d["ra_ref"])
        P_N = (xyz[:, 0] * np.cos(d["ra_ref"]) * np.sin(d["dec_ref"])
               + xyz[:, 1] * np.sin(d["ra_ref"]) * np.sin(d["dec_ref"])
               - xyz[:, 2] * np.cos(d["dec_ref"]))
        d_dense = dict(d, P_E=P_E, P_N=P_N)
        dE_lin, dN_lin = self._linear_terms(d_dense, t_dense, point, system)
        dE_orb, dN_orb = self._eval_photo(i, t_dense, vals)
        axL.plot(dE_lin, dN_lin, ":", color="tab:blue", lw=0.8, zorder=1,
                 label="pm + parallax")
        axL.plot(dE_lin + dE_orb, dN_lin + dN_orb, "-", color="0.4", lw=0.8,
                 zorder=2, label="pm + parallax + orbit")

        dE_lin_o, dN_lin_o = self._linear_terms(d, t, point, system)
        dE_orb_o, dN_orb_o = self._eval_photo(i, t, vals)
        axL.plot(dE_lin_o + dE_orb_o, dN_lin_o + dN_orb_o, "r.", ms=5,
                 zorder=2, label="observation epochs")
        if d["mode"] == "abs":
            axL.errorbar(d["dE_obs"], d["dN_obs"], xerr=d["err_E"],
                         yerr=d["err_N"], fmt="o", ms=3, alpha=0.5, zorder=3,
                         label="data")
        axL.invert_xaxis()  # East to the left
        # No equal aspect here: the pm-dominated path is highly anisotropic
        # and equal axes squeeze the parallax/orbit loops into invisibility
        # (the orbit panel keeps equal aspect, where shape fidelity matters).
        axL.set_xlabel(r"$\Delta\alpha^*$ [mas]")
        axL.set_ylabel(r"$\Delta\delta$ [mas]")
        axL.set_title("Path on sky")
        axL.legend(loc="best", fontsize="small")

        # ---------------- right: orbit alone, with annotations --------------
        # elements in internal units (per planet)
        tp_arr, n_arr, ecc_arr, w_arr, bigom_arr = self._compiled_elements(*vals)
        P_orb = 2.0 * np.pi / np.atleast_1d(n_arr)
        t1 = t.max()
        t_orb = np.linspace(t1 - np.max(P_orb), t1, 2000)
        dE_o, dN_o = self._eval_photo(i, t_orb, vals)
        axR.plot(dE_o, dN_o, "k-", lw=1.2, zorder=2, label="photocenter orbit")
        dE_ep, dN_ep = self._eval_photo(i, t, vals)
        axR.plot(dE_ep, dN_ep, "r.", ms=5, zorder=3, label="observation epochs")
        axR.plot(0, 0, "k+", ms=12, mew=2, zorder=4)  # barycenter

        n_planets = len(np.atleast_1d(tp_arr))
        if n_planets == 1:
            tp = float(np.atleast_1d(tp_arr)[0])
            n_mm = float(np.atleast_1d(n_arr)[0])
            ecc = float(np.atleast_1d(ecc_arr)[0])
            w = float(np.atleast_1d(w_arr)[0])
            bigom = float(np.atleast_1d(bigom_arr)[0])
            P = 2.0 * np.pi / n_mm

            # Line of nodes: through the barycenter at PA = bigomega
            r_max = 1.1 * np.max(np.hypot(dE_o, dN_o))
            axR.plot([r_max * np.sin(bigom), -r_max * np.sin(bigom)],
                     [r_max * np.cos(bigom), -r_max * np.cos(bigom)],
                     "--", color="0.5", lw=1, zorder=1, label="line of nodes")

            # Ascending node: f = -omega_*; with our conventions the
            # photocenter crosses it moving AWAY from the observer (max RV)
            f_node = -w
            E_node = 2.0 * np.arctan2(
                np.sqrt(1 - ecc) * np.sin(f_node / 2.0),
                np.sqrt(1 + ecc) * np.cos(f_node / 2.0))
            M_node = E_node - ecc * np.sin(E_node)
            t_node = tp + M_node / n_mm
            # shift into the plotted window
            t_node += np.ceil((t_orb[0] - t_node) / P) * P
            (xn,), (yn,) = self._eval_photo(
                i, np.array([t_node], dtype=np.float64), vals)
            axR.plot(xn, yn, "o", color="tab:blue", ms=10, mfc="none", mew=2,
                     zorder=5, label="ascending node")

            # Direction of motion: a curved arrow tracing the orbit away
            # from the ascending node, drawn 25% outside the orbit itself.
            t_arc = np.linspace(t_node, t_node + 0.06 * P, 60)
            xa, ya = self._eval_photo(i, t_arc, vals)
            xa, ya = 1.25 * np.asarray(xa), 1.25 * np.asarray(ya)
            axR.plot(xa[:-1], ya[:-1], "-", color="tab:blue", lw=2, zorder=6)
            axR.annotate("", xy=(xa[-1], ya[-1]), xytext=(xa[-2], ya[-2]),
                         zorder=6,
                         arrowprops=dict(arrowstyle="-|>", color="tab:blue",
                                         lw=2, mutation_scale=22,
                                         shrinkA=0, shrinkB=0))

        axR.invert_xaxis()  # East to the left
        axR.set_aspect("equal", adjustable="datalim")
        axR.set_xlabel(r"$\Delta\alpha^*$ [mas]")
        axR.set_ylabel(r"$\Delta\delta$ [mas]")
        axR.set_title("Photocenter orbit")
        axR.legend(loc="best", fontsize="small")

        fig.suptitle(f"{d['name']} ({system.name})")
        fig.tight_layout()
        fig.savefig(f"{filename_prefix}_astrometry_{d['name']}_sky.pdf")
        plt.close(fig)
