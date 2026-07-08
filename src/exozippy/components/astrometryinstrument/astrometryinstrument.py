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
         The position angle is measured East of North.  Use 'planet_ndx'
         to select which companion the measurements refer to.

Per-instrument config keys:
  file              : data file (whitespace-separated, '#' comments)
  mode              : gaia | abs | rel   (default gaia)
  observer_location : ephemeris for parallax factors (default 'gaia' for
                      gaia mode, 'earth' otherwise; see exozippy.ephemeris)
  star_ndx          : index of the host star (default 0)
  planet_ndx        : rel mode only, index of the companion (default 0)
  epoch             : reference epoch [BJD_TDB] for ra/dec/pm (default:
                      mean time of all gaia/abs observations)
  sep_unit          : rel mode separation unit (default 'mas')

Conventions follow EXOFASTv2: omega is the argument of periastron of the
primary's orbit (omega_*); bigomega is the position angle of the ascending
node (East of North), where the ascending node is the node at which the
body recedes from the observer.  See Orbit.get_sky_position.

Model limitations (v1): the photocenter wobble sums over all planets with
a single per-instrument companion flux fraction (fluxfrac, default 0 =
dark companions); relative positions are Keplerian two-body (host wobble
from other companions is neglected); light travel time and aberration are
neglected.
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

    @property
    def prefix(self):
        return "astrometryinstrument"

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

    # ------------------------------------------------------------------
    # Model pieces (PyTensor)
    # ------------------------------------------------------------------
    def _photocenter_amplitude(self, system, beta):
        """Per-planet photocenter semimajor axis in mas: a_rel*(B - beta)*plx."""
        planets = system.planet
        star = system.star
        plx = star.parallax.value[planets.star_map_tensor]
        mass_frac = planets.mass.value / planets.m_total.value
        return planets.arsun.value * RSUN_AU * (mass_frac - beta) * plx

    def _relative_amplitude(self, system):
        """Per-planet relative semimajor axis in mas: a_rel * plx."""
        planets = system.planet
        star = system.star
        plx = star.parallax.value[planets.star_map_tensor]
        return planets.arsun.value * RSUN_AU * plx

    def _absolute_model(self, system, d, t, beta, has_orbit):
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

        if has_orbit:
            a_phot = self._photocenter_amplitude(system, beta)
            dE_orb, dN_orb = system.orbit.get_sky_position(
                pt.as_tensor_variable(t), a_phot, system.planet.orbit_map)
            dE = dE + pt.sum(dE_orb, axis=1)
            dN = dN + pt.sum(dN_orb, axis=1)

        return dE, dN

    # ------------------------------------------------------------------
    # Stage 6
    # ------------------------------------------------------------------
    def build_likelihood(self, model, system):
        has_orbit = hasattr(system, "orbit") and hasattr(system, "planet")
        needs_star = any(m in ("gaia", "abs") for m in self.modes)
        if needs_star and not hasattr(system, "star"):
            raise ValueError(f"[{self.prefix}] gaia/abs astrometry requires a star component.")

        for i, d in enumerate(self.datasets):
            name = d["name"]
            mode = d["mode"]
            t = d["time"]
            jv = self.jitter_variance.value[i]
            beta = self.fluxfrac.value[i]

            if mode in ("gaia", "abs"):
                dE, dN = self._absolute_model(system, d, t, beta, has_orbit)

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
                if not has_orbit:
                    raise ValueError(
                        f"[{self.prefix}.{name}] relative astrometry requires "
                        f"orbit and planet components.")
                j = int(self.planet_map[i])
                a_rel = self._relative_amplitude(system)
                dE_rel, dN_rel = system.orbit.get_sky_position(
                    pt.as_tensor_variable(t), a_rel, system.planet.orbit_map,
                    relative=True)
                dEj = dE_rel[:, j]
                dNj = dN_rel[:, j]

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
        has_orbit = hasattr(system, "orbit") and hasattr(system, "planet")
        if not has_orbit:
            return

        param_symbols = [p.value for p in system.plot_params]
        t_input = pt.vector("t_input")

        # Photocenter orbit (summed over planets), per instrument (beta varies)
        self._compiled_photo = []
        for i in range(self.n_elements):
            beta = self.fluxfrac.value[i]
            a_phot = self._photocenter_amplitude(system, beta)
            dE_orb, dN_orb = system.orbit.get_sky_position(
                t_input, a_phot, system.planet.orbit_map)
            self._compiled_photo.append(pytensor.function(
                inputs=[t_input] + param_symbols,
                outputs=[pt.sum(dE_orb, axis=1), pt.sum(dN_orb, axis=1)],
                on_unused_input="ignore",
            ))

        # Relative orbit matrix (N_times, N_planets)
        a_rel = self._relative_amplitude(system)
        dE_rel, dN_rel = system.orbit.get_sky_position(
            t_input, a_rel, system.planet.orbit_map, relative=True)
        self._compiled_rel = pytensor.function(
            inputs=[t_input] + param_symbols,
            outputs=[dE_rel, dN_rel],
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
        plx = get(star.parallax.label, 0.0)

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
                    dE_orb, dN_orb = self._compiled_photo[i](t.astype(np.float64), *vals)
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
                dE_orb, dN_orb = self._compiled_photo[i](t.astype(np.float64), *vals)
                plt.errorbar(d["dE_obs"] - dE_lin, d["dN_obs"] - dN_lin,
                             xerr=d["err_E"], yerr=d["err_N"], fmt="o",
                             alpha=0.6, zorder=1, label="data - (pm+plx)")
                t_lin_pretty = np.linspace(t.min(), t.max(), 2000)
                dE_p, dN_p = self._compiled_photo[i](t_lin_pretty.astype(np.float64), *vals)
                plt.plot(dE_p, dN_p, "r-", lw=1, zorder=2, label="photocenter orbit")
                plt.gca().invert_xaxis()  # East to the left
                plt.gca().set_aspect("equal", adjustable="datalim")
                plt.xlabel(r"$\Delta\alpha^*$ [mas]")
                plt.ylabel(r"$\Delta\delta$ [mas]")
                plt.title(f"Absolute astrometry: {d['name']} ({system.name})")
                plt.legend(loc="best", fontsize="small")

            else:  # rel
                j = int(self.planet_map[i])
                vals = self._point_values(system, ref_point)
                dE_m, dN_m = self._compiled_rel(t_pretty.astype(np.float64), *vals)
                plt.errorbar(d["sep"] * np.sin(d["pa"]), d["sep"] * np.cos(d["pa"]),
                             fmt="o", alpha=0.6, zorder=1, label="data")
                plt.plot(dE_m[:, j], dN_m[:, j], "r-", lw=1, zorder=2, label="model")
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
