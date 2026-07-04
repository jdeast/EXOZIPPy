import logging
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

import pymc as pm
import pytensor.tensor as pt
from scipy.interpolate import CubicSpline
from scipy.optimize import nnls

from astropy.coordinates import (
    get_body_barycentric,
    solar_system_ephemeris,
    EarthLocation,
    ICRS,
    SkyCoord
)
from astropy.time import Time

import astropy.units as u
from exozippy.components.component import Component
from exozippy.config import RANK_DERIVED_DATA
import pytensor.tensor as pt
import pytensor
import pymc as pm

class MulensInstrument(Component):
    def __init__(self, config, config_manager):
        super().__init__(config, config_manager)
        self.label = "Microlensing Data"
        self.files = [c.get("file") for c in self.config]

    @property
    def prefix(self):
        return "mulensinstrument"

    def load_data(self, system):
        """Stage 1a: Load photometry and pre-calculate observer positions.

        Single-event assumption (enforced by Lens.__init__): index 0 is the
        only event, so the event-0 source, t0_par, and magnification are used
        throughout.
        """
        all_times, all_mags, all_errs, inst_indices = [], [], [], []
        self.fs_init = []
        self.q_source_init = []
        self.q_flux_init = []      # per-instrument f_s2/f_s1 (binary source)
        self._raw_time_list = []
        all_obspos = []
        all_obspos_abs = []

        self._n_sources = int(system.lens.n_sources)

        # Source RA/Dec (degrees from resolve → radians for projection math)
        source_ndx = int(system.lens.source_map[0])
        n_stars = system.star.n_elements
        ra_deg  = self.config_manager.resolve("star", "ra",  shape=(n_stars,))['initval'][source_ndx]
        dec_deg = self.config_manager.resolve("star", "dec", shape=(n_stars,))['initval'][source_ndx]
        ra_rad  = float(ra_deg)  * np.pi / 180.0
        dec_rad = float(dec_deg) * np.pi / 180.0

        # Geocentric reference (Skowron+2011 convention): Earth's position and
        # velocity at t_0_par define the inertial frame.  All observer positions
        # are stored as deviations from this linear Earth trajectory so that
        # t_0/u_0 remain geocentric parameters.
        self._t0_par = float(system.lens.t0_par[0])
        self._earth_pos_ref = self.get_observer_position(
            np.array([self._t0_par]), 'earth')[0]                # (3,) AU
        _dt = 0.5  # days for finite-difference velocity
        _ep = self.get_observer_position(np.array([self._t0_par + _dt]), 'earth')[0]
        _em = self.get_observer_position(np.array([self._t0_par - _dt]), 'earth')[0]
        self._earth_vel_ref = (_ep - _em) / (2.0 * _dt)         # AU/day

        # Median absolute position per instrument (used by Lens to detect parallax)
        self.inst_ref_pos = []

        for i, file in enumerate(self.files):
            df = pd.read_csv(file, sep=r'\s+', engine='c', header=None, comment='#')
            t, m, e = df.iloc[:, 0].values, df.iloc[:, 1].values, df.iloc[:, 2].values

            if self.config[i].get("data_format", "magnitude") == "flux":
                # Convert normalized flux to instrumental magnitudes.
                # mag = -2.5*log10(flux);  err = (2.5/ln10) * flux_err/flux
                safe_f = np.maximum(m, 1e-30)
                e = (2.5 / np.log(10)) * np.maximum(e, 0.0) / safe_f
                m = -2.5 * np.log10(safe_f)

            obs_loc = self.config[i].get("observer_location", "earth")
            xyz_abs = self.get_observer_position(t, observer_location=obs_loc)
            self.inst_ref_pos.append(np.median(xyz_abs, axis=0))

            xyz_delta = self._abs_to_delta(t, xyz_abs)
            all_obspos.append(xyz_delta)
            all_obspos_abs.append(xyz_abs)

            f_total, q_source, q_flux = self._estimate_flux_components(
                t, m, xyz_delta, ra_rad, dec_rad, i
            )
            self.fs_init.append(f_total)
            self.q_source_init.append(q_source)
            self.q_flux_init.append(q_flux)

            self._check_data_format(
                t, m, e, xyz_delta, ra_rad, dec_rad,
                self.config[i].get("file", f"instrument {i}"),
                data_format=self.config[i].get("data_format", "magnitude"),
            )

            all_times.append(t)
            all_mags.append(m)
            all_errs.append(e)
            inst_indices.append(np.full(len(t), i))
            self._raw_time_list.append(t)

        self.inst_ref_pos = np.array(self.inst_ref_pos)   # (n_inst, 3) absolute AU
        self.time     = np.concatenate(all_times).astype(float)
        self.mag      = np.concatenate(all_mags).astype(float)
        self.err      = np.concatenate(all_errs).astype(float)
        self.inst_map = np.concatenate(inst_indices).astype(int)
        self.observer_pos     = np.vstack(all_obspos).astype(float)      # geocentric deviations
        self.observer_pos_abs = np.vstack(all_obspos_abs).astype(float)  # absolute barycentric (for get_magnification_op)

    def _check_data_format(self, t, m, e, xyz_delta, ra_rad, dec_rad,
                           label, data_format="magnitude"):
        """Warn if data appears fainter at peak than at baseline.

        By the time this runs m is always in magnitudes (flux data has already been
        converted).  A valid microlensing event must show brightening (smaller mag
        value) near peak.  If the data instead grows fainter, either:
          - data_format is 'magnitude' but the data are really in flux units, or
          - data_format is 'flux' but the data are really in magnitudes.

        Returns silently when the dataset has fewer than 3 epochs near baseline
        (e.g., Spitzer peak-only data) — no comparison is possible there.
        """
        cm = self.config_manager
        def _get(key, default=None):
            data = cm.user_params.get(key)
            if data is None:
                return default
            return data.get("initval", default) if isinstance(data, dict) else float(data)

        t0 = _get("lens.0.t_0")
        u0 = _get("lens.0.u_0")
        tE = _get("lens.0.t_E")
        if t0 is None or u0 is None:
            return

        pi_E_N = _get("lens.0.pi_E_N", 0.0)
        pi_E_E = _get("lens.0.pi_E_E", 0.0)
        tE_safe = max(abs(float(tE)), 1.0) if tE is not None else 30.0

        x, y, z = xyz_delta[:, 0], xyz_delta[:, 1], xyz_delta[:, 2]
        delta_e = -x * np.sin(ra_rad) + y * np.cos(ra_rad)
        delta_n = (-x * np.cos(ra_rad) * np.sin(dec_rad)
                   - y * np.sin(ra_rad) * np.sin(dec_rad)
                   + z * np.cos(dec_rad))
        tau   = (t - float(t0)) / tE_safe
        tau_p = tau - delta_n * float(pi_E_N) - delta_e * float(pi_E_E)
        u_p   = float(u0) + delta_n * float(pi_E_E) - delta_e * float(pi_E_N)
        u_traj = np.sqrt(tau_p ** 2 + u_p ** 2)
        A_traj = (u_traj ** 2 + 2.0) / (u_traj * np.sqrt(u_traj ** 2 + 4.0))

        baseline_mask = A_traj < 1.1
        peak_mask     = A_traj > 1.5

        # Skip if no baseline coverage (e.g., Spitzer peak-only data)
        if np.sum(baseline_mask) < 3 or np.sum(peak_mask) < 3:
            return

        m_baseline = float(np.median(m[baseline_mask]))
        m_peak     = float(np.median(m[peak_mask]))

        # In magnitudes, brighter = smaller value.  Peak must be brighter.
        if m_peak > m_baseline:
            typical_err = float(np.median(np.abs(e)))
            n_sigma = (m_peak - m_baseline) / max(typical_err, 0.001)
            if n_sigma > 10.0:
                if data_format == "flux":
                    logger.warning(
                        f"[{label}] After flux→mag conversion, data appears fainter at "
                        f"peak ({m_peak:.3g}) than at baseline ({m_baseline:.3g}) — "
                        f"{n_sigma:.0f}σ offset.  Data may actually be in magnitudes; "
                        f"remove 'data_format: flux' from the YAML config block if so."
                    )
                else:
                    logger.warning(
                        f"[{label}] Data appears fainter at peak ({m_peak:.3g}) than at "
                        f"baseline ({m_baseline:.3g}) — {n_sigma:.0f}σ offset.  "
                        f"Data may be in flux units; add 'data_format: flux' to the "
                        f"YAML config block for this instrument if so."
                    )

    @staticmethod
    def _pspl_magnification(t, delta_e, delta_n, t0, u0, tE, pi_E_N, pi_E_E):
        """Point-source Paczynski magnification along one source trajectory."""
        tE_safe = max(abs(float(tE)), 1.0) if tE is not None else 30.0
        tau   = (t - float(t0)) / tE_safe
        tau_p = tau - delta_n * float(pi_E_N) - delta_e * float(pi_E_E)
        u_p   = float(u0) + delta_n * float(pi_E_E) - delta_e * float(pi_E_N)
        u_traj = np.sqrt(tau_p ** 2 + u_p ** 2)
        return (u_traj ** 2 + 2.0) / (u_traj * np.sqrt(u_traj ** 2 + 4.0))

    @staticmethod
    def _binary_magnification_columns(t, n_src, _get):
        """Per-source magnification columns using the full binary-lens model.

        The flux bootstrap needs magnification columns that actually
        distinguish the sources.  For binary-source events the PSPL wings are
        nearly collinear (the trajectories differ mostly through their caustic
        features), which makes the NNLS decomposition degenerate; the binary
        model at the seeded (s, q, alpha) breaks that degeneracy.

        Returns a list of n_src columns, or None when the binary geometry is
        not specified (single-lens event, or missing per-source params) or
        MulensModel fails — the caller then falls back to the PSPL columns.
        Parallax is intentionally ignored (flux scales only).
        """
        s_val = _get("lens.0.s")
        q_val = _get("lens.0.q")
        alpha = _get("lens.0.alpha")
        if s_val is None or q_val is None or alpha is None:
            return None

        try:
            import MulensModel as mm
            cols = []
            for j in range(n_src):
                t0 = _get(f"lens.{j}.t_0")
                u0 = _get(f"lens.{j}.u_0")
                tE = _get(f"lens.{j}.t_E", _get("lens.0.t_E"))
                if t0 is None or u0 is None or tE is None:
                    return None
                params = {
                    "t_0": float(t0),
                    "u_0": float(np.sign(u0) * max(abs(u0), 1e-9)),
                    "t_E": max(float(tE), 1e-4),
                    "s": max(float(s_val), 1e-6),
                    "q": float(np.clip(q_val, 1e-9, 100.0)),
                    "alpha": float(alpha),
                }
                rho = _get(f"lens.{j}.rho")
                if rho is not None:
                    params["rho"] = max(float(rho), 1e-9)
                model = mm.Model(params)
                if rho is not None:
                    window = 3.0 * params["t_E"]
                    model.set_magnification_methods(
                        [params["t_0"] - window, "VBM", params["t_0"] + window])
                cols.append(np.asarray(model.get_magnification(t)))
            return cols
        except Exception as e:
            logger.warning(f"Binary-lens flux bootstrap failed ({e}); "
                           "falling back to PSPL columns.")
            return None

    def _estimate_flux_components(self, t, m, xyz_au, ra_rad, dec_rad, inst_idx):
        """Estimate (f_total, q_source, q_flux) for one instrument.

        f_total  = total baseline flux (all sources + blend)
        q_source = (Σ_j f_s,j) / f_total
        q_flux   = f_s,2 / f_s,1 (binary source; 1.0 for single source)

        With N sources the decomposition solves the linear model
        F(t) = Σ_j f_s,j · A_j(t) + f_b via NNLS, where A_j is the PSPL
        magnification along source j's trajectory (lens.<j>.t_0/u_0/t_E).
        The binary-lens perturbation is irrelevant here — we only need flux
        scales, not a precise model.

        If the user has specified f_source and/or f_blend in their params file,
        those values are respected (they are TOTALS over sources):
          - both given  → skip estimation entirely, derive q from the ratio
          - f_source only → fix it and solve for f_blend via median residuals
          - f_blend only  → fix it and solve for f_source via NNLS
          - neither       → solve everything via NNLS

        Falls back to the data median / q=0.95 when t_0 or u_0 are absent.
        """
        cm = self.config_manager
        n_src = getattr(self, "_n_sources", 1)

        def _get(key, default=None):
            data = cm.user_params.get(key)
            if data is None:
                return default
            return data.get("initval", default) if isinstance(data, dict) else float(data)

        def _get_flux(param):
            # user_params keys are normalized to index form by standardize_param_names
            val = _get(f"mulensinstrument.{inst_idx}.{param}")
            return float(val) if val is not None else None

        q_flux_user = _get_flux("q_flux")
        q_flux_fallback = q_flux_user if q_flux_user is not None else 1.0

        t0     = _get("lens.0.t_0")
        u0     = _get("lens.0.u_0")
        tE     = _get("lens.0.t_E")
        pi_E_N = _get("lens.0.pi_E_N", 0.0)
        pi_E_E = _get("lens.0.pi_E_E", 0.0)

        f_source_user = _get_flux("f_source")
        f_blend_user  = _get_flux("f_blend")

        if f_source_user is not None and f_blend_user is not None:
            f_total = f_source_user + f_blend_user
            q_source = float(np.clip(f_source_user / max(f_total, 1e-30), 0.05, 0.95))
            return f_total, q_source, q_flux_fallback

        if t0 is None or u0 is None:
            f_total = 10.0 ** (-0.4 * np.median(m))
            return f_total, 0.95, q_flux_fallback

        x, y, z = xyz_au[:, 0], xyz_au[:, 1], xyz_au[:, 2]
        delta_e = -x * np.sin(ra_rad) + y * np.cos(ra_rad)
        delta_n = (-x * np.cos(ra_rad) * np.sin(dec_rad)
                   - y * np.sin(ra_rad) * np.sin(dec_rad)
                   + z * np.cos(dec_rad))

        # One magnification column per source trajectory.  Prefer the full
        # binary-lens model (breaks the NNLS degeneracy between overlapping
        # source trajectories); fall back to PSPL columns.  Missing per-source
        # params (j > 0) degrade gracefully to the single-source estimate.
        A_cols = self._binary_magnification_columns(t, n_src, _get)
        if A_cols is None:
            A_cols = [self._pspl_magnification(t, delta_e, delta_n,
                                               t0, u0, tE, pi_E_N, pi_E_E)]
            for j in range(1, n_src):
                t0_j = _get(f"lens.{j}.t_0")
                u0_j = _get(f"lens.{j}.u_0")
                tE_j = _get(f"lens.{j}.t_E", tE)
                if t0_j is None or u0_j is None:
                    logger.warning(
                        f"lens.{j}.t_0/u_0 missing — flux bootstrap treats source {j} "
                        f"as blended into source 0."
                    )
                    continue
                A_cols.append(self._pspl_magnification(t, delta_e, delta_n,
                                                       t0_j, u0_j, tE_j,
                                                       pi_E_N, pi_E_E))

        A_traj = A_cols[0]
        F_obs = 10.0 ** (-0.4 * m)

        q_flux_est = q_flux_fallback
        if len(A_cols) > 1:
            # Multi-source NNLS: F = Σ_j f_s,j · A_j + f_b
            X = np.column_stack(A_cols + [np.ones(len(t))])
            sol, _ = nnls(X, F_obs)
            f_srcs, f_blend_est = sol[:-1], sol[-1]
            f_source_est = float(np.sum(f_srcs))
            if q_flux_user is None and f_srcs[0] > 1e-30 and len(f_srcs) > 1:
                q_flux_est = float(np.clip(f_srcs[1] / f_srcs[0], 1e-3, 1e3))
            if f_source_user is not None and f_source_est > 1e-30:
                # honor the user's total source flux; keep the NNLS ratio
                f_blend_est = max(float(np.median(
                    F_obs - X[:, :-1] @ (f_srcs * f_source_user / f_source_est))), 0.0)
                f_source_est = f_source_user
        elif f_source_user is not None:
            f_blend_est = max(float(np.median(F_obs - f_source_user * A_traj)), 0.0)
            f_source_est = f_source_user
        elif f_blend_user is not None:
            (f_source_est,), _ = nnls(A_traj.reshape(-1, 1), F_obs - f_blend_user)
            f_blend_est = f_blend_user
        else:
            X = np.column_stack([A_traj, np.ones(len(A_traj))])
            (f_source_est, f_blend_est), _ = nnls(X, F_obs)

        f_total = f_source_est + f_blend_est
        if f_total < 1e-30 or f_source_est < 1e-30:
            f_total = 10.0 ** (-0.4 * np.median(m))
            return f_total, 0.95, q_flux_est

        q_source = float(np.clip(f_source_est / f_total, 0.05, 0.95))
        logger.debug(
            f"NNLS flux decomp: f_source={f_source_est:.3e}, f_blend={f_blend_est:.3e}"
            f" → q_source={q_source:.4f}, q_flux={q_flux_est:.4f}"
        )
        return f_total, q_source, q_flux_est

    def _abs_to_delta(self, t, xyz_abs):
        """Convert absolute barycentric positions to Skowron+2011 geocentric deviations.

        Converts to the Skowron+2011 geocentric inertial frame whose origin
        moves with Earth's position and velocity at t_0_par.  Any observer's
        position in this frame is:

        delta(t) = xyz_obs(t) - [xyz_earth(t_0_par) + v_earth(t_0_par)*(t - t_0_par)]

        For Earth: small deviation from straight-line motion (annual parallax).
        For Spitzer: ≈ Spitzer − Earth vector at t_0_par (satellite parallax offset,
        ~1–2 AU).  Yee+2014 §3: "Spitzer's offset from the centre of Earth is
        treated just as any other observatory."
        """
        t_delta = (t - self._t0_par)[:, np.newaxis]  # (N, 1)
        return xyz_abs - (self._earth_pos_ref + self._earth_vel_ref * t_delta)

    def get_observer_position(self, time, observer_location='earth'):
        """
        High-precision observer position dispatcher.
        Supports:
          - Major bodies ('earth', 'mars')
          - Topocentric ground sites ('lat,lon,alt' or 'CTIO')
          - Satellite Ephemeris files (Interpolated)
        """
        solar_system_ephemeris.set('jpl')
        t_obj = Time(time, format='jd', scale='tdb')

        # 1. Handle Terrestrial / Topocentric (Lat/Lon)
        # Check if string looks like "lat, lon, alt" or a known site name
        try:
            if ',' in observer_location:
                # Parse "lat, lon, alt"
                loc = EarthLocation.from_geodetic(*[float(x) for x in observer_location.split(',')])
            else:
                # Check for site names like 'CTIO' or 'Siding Spring'
                loc = EarthLocation.of_site(observer_location)

            # Get topocentric position: Barycentric Earth + Geocentric Offset
            # This accounts for Earth's orbit and the observer's specific spot on the globe
            return loc.get_itrs(t_obj).transform_to(ICRS()).cartesian.xyz.to('au').value.T
        except:
            # Not a ground site, move to next check
            pass

        # 2. Handle Major Bodies
        if observer_location in ['earth', 'moon']:
            return get_body_barycentric(observer_location, t_obj).xyz.to('au').value.T

        # 3. Handle Satellite Ephemeris Files with Search Paths
        search_paths = [
            observer_location,  # absolute/relative path
            os.path.join(os.path.dirname(__file__), 'ephemerides', observer_location), # Package internal
            os.path.join(os.path.dirname(__file__), 'ephemerides', observer_location + '.eph')  # Package internal
        ]

        for path in search_paths:
            if os.path.exists(path):
                return self._interpolate_ephemeris(time, path)

        raise ValueError(f"observer location not recognized: {observer_location}; see $EXOZIPPY_PATH/src/exozippy/components/mulensing/ephemerides/get_ephemeris.py to generate an ephemeris")

    def _interpolate_ephemeris(self, time, ephemeris_file):
        """
        Interpolates a Barycentric ephemeris file.

        Parameters:
        -----------
        time : float or ndarray
            The time(s) at which to calculate coordinates (BJD_TDB).
        ephemeris_file : str
            Path to the file generated by get_ephemeris.py
            (Format: BJD_TDB, X, Y, Z)

        Returns:
        --------
        xyz_au : ndarray
            (N, 3) array of X, Y, Z coordinates in AU.
        """
        # Load data, skipping the header lines
        # data[:, 0] = BJD_TDB, [:, 1:4] = X, Y, Z
        data = np.loadtxt(ephemeris_file)

        t_grid = data[:, 0]
        xyz_grid = data[:, 1:4]

        # Check if we are extrapolating (which is dangerous)
        t_min, t_max = np.min(t_grid), np.max(t_grid)
        if np.any(time < t_min) or np.any(time > t_max):
            import warnings
            warnings.warn(f"Extrapolating outside ephemeris range! "
                          f"Grid: {t_min:.2f}-{t_max:.2f}, Requested: {np.min(time):.2f}")

        # Create the spline object
        # bc_type='not-a-knot' is standard for smooth orbital curves
        cs = CubicSpline(t_grid, xyz_grid, axis=0, bc_type='not-a-knot')

        return cs(time)

    def register_parameters(self, system):
        """Stage 2: Declare the manifest with bootstrapped fluxes."""
        f_total_init = np.array(self.fs_init)
        q_source_init = np.array(self.q_source_init)

        # Inject hints for derived f_source / f_blend so the relaxation engine
        # can resolve initial values.  Also push the data-estimated q_source and
        # log_f_total as RANK_DERIVED_DATA hints so they override the defaults.yaml
        # values while still yielding to any explicit user override in params.yaml
        # (RANK_USER wins — essential when restarting a fit from a previous MAP).
        for i in range(self.n_elements):
            q = q_source_init[i]
            f_source_guess = f_total_init[i] * q
            f_blend_guess = f_total_init[i] * (1.0 - q)
            self.config_manager.add_hint(f"{self.prefix}.{i}.f_source", f_source_guess)
            self.config_manager.add_hint(f"{self.prefix}.{i}.f_blend", f_blend_guess)
            self.config_manager.add_hint(
                f"{self.prefix}.{i}.q_source", q, rank=RANK_DERIVED_DATA
            )
            self.config_manager.add_hint(
                f"{self.prefix}.{i}.log_f_total",
                float(np.log10(f_total_init[i])),
                rank=RANK_DERIVED_DATA,
            )

        self.manifest = {
            "log_f_total": None,
            "q_source": None,
            "f_source": "default",
            "f_blend": "default",
            "err_scale": None,
        }

        # Binary source: one flux ratio q_flux = f_s2/f_s1 per instrument
        # (sources have different colors, so the ratio is chromatic).
        n_sources = getattr(self, "_n_sources", 1)
        if n_sources > 1:
            if n_sources > 2:
                raise NotImplementedError(
                    f"{self._n_sources}-source flux modeling is not yet "
                    "implemented: the per-instrument flux ratio q_flux only "
                    "handles 2 sources. The per-source magnification path is "
                    "generic; generalize the flux parameterization to add more."
                )
            self.manifest["q_flux"] = None
            for i in range(self.n_elements):
                self.config_manager.add_hint(
                    f"{self.prefix}.{i}.q_flux", float(self.q_flux_init[i]),
                    rank=RANK_DERIVED_DATA,
                )

        # Map each instrument to a Band instance by name.
        band_names = [c.get("band", None) for c in self.config]
        if hasattr(system, 'band'):
            name_to_idx = {name: i for i, name in enumerate(system.band.names)}
            self.band_map = np.array([
                name_to_idx[n] if (n is not None and n in name_to_idx) else -1
                for n in band_names
            ], dtype=int)
            missing = [n for n in band_names if n is not None and n not in name_to_idx]
            for n in missing:
                logger.warning(f"Instrument references unknown band '{n}'; LD will be skipped.")
        else:
            self.band_map = np.full(self.n_elements, -1, dtype=int)

    def build_likelihood(self, model, system):

        if hasattr(system, 'sed'):
            raise NotImplementedError(
                "SED-based f_source prior for mulensing instruments is not yet implemented."
            )

        # 1. Constants
        t = pm.Data("mu_time", self.time)
        obs_mag = pm.Data("mu_obs_mag", self.mag)
        obs_err = pm.Data("mu_obs_err", self.err)

        # 2. Magnification — both symbolic and Op paths take absolute barycentric AU.
        #    get_magnification_op dispatches: PSPL→symbolic (NUTS-friendly),
        #    binary/finite-source→MulensModel Op (use Metropolis).
        #
        #    When finite source is active, pass u1 and bandpass from the connected
        #    Band component.  Multiple distinct bands across instruments are not yet
        #    supported for finite-source LD; the first band found is used.
        u1 = None
        bandpass = None
        if (system.lens.finite_source[0]
                and hasattr(system, 'band')
                and np.any(self.band_map >= 0)):
            band_indices = [self.band_map[i] for i in range(self.n_elements)
                            if self.band_map[i] >= 0]
            unique = sorted(set(band_indices))
            if len(unique) > 1:
                logger.warning(
                    "Multiple bands for finite-source instruments; using first band's u1."
                )
            band_idx = unique[0]
            u1 = system.band.u1.value[band_idx]
            bandpass = system.band.names[band_idx]

        # One magnification curve per source trajectory (NSNL)
        n_src = self._n_sources
        A_per_source = []
        for j in range(n_src):
            system.lens.resolve_auto_vbbl(self.time, index=j)
            A_per_source.append(system.lens.get_magnification_op(
                t, self.observer_pos_abs, system, index=j, u1=u1, bandpass=bandpass
            ))

        # 3. Flux Model: F = Σ_j f_s,j·A_j + f_b, with f_s,1 = f_s/(1+q_F),
        #    f_s,2 = f_s·q_F/(1+q_F) (q_F per instrument — sources differ in color)
        fs = self.f_source.value[self.inst_map_tensor]
        fb = self.f_blend.value[self.inst_map_tensor]
        k_scale = self.err_scale.value[self.inst_map_tensor]

        if n_src == 1:
            model_flux = fs * A_per_source[0] + fb
        else:
            qf = self.q_flux.value[self.inst_map_tensor]
            qf_safe = pt.maximum(qf, 0.0)
            model_flux = (fs / (1.0 + qf_safe) * A_per_source[0]
                          + fs * qf_safe / (1.0 + qf_safe) * A_per_source[1]
                          + fb)

        # Guard against negative flux causing log10(NaN) crash during tuning
        safe_flux = pt.maximum(model_flux, 1e-12)
        model_mag = -2.5 * pt.log10(safe_flux)

        # 4. Error scaling & Likelihood
        sigma = obs_err * k_scale

        pm.Normal(
            f"{self.prefix}.model",
            mu=model_mag,
            sigma=sigma,
            observed=obs_mag
        )

    def compile_plotters(self, model, system):
        """Compile fast PyTensor functions for the lightcurve."""
        t_input = pt.vector("mu_t_input")
        obs_pos_input = pt.dmatrix("obs_pos")
        inst_idx = pt.iscalar("mu_inst_idx")

        param_symbols = [p.value for p in system.plot_params]

        n_src = self._n_sources
        A_per_source = [
            system.lens.get_magnification_op(t_input, obs_pos_input, system, index=j)
            for j in range(n_src)
        ]

        fs_inst = self.f_source.value[inst_idx]
        fb_inst = self.f_blend.value[inst_idx]

        # Δmag = mag(t) − mag_baseline = −2.5·log10(A_eff).
        # Zero at baseline, negative when brighter, independent of f_total.
        if n_src == 1:
            model_flux = fs_inst * A_per_source[0] + fb_inst
        else:
            qf_inst = pt.maximum(self.q_flux.value[inst_idx], 0.0)
            model_flux = (fs_inst / (1.0 + qf_inst) * A_per_source[0]
                          + fs_inst * qf_inst / (1.0 + qf_inst) * A_per_source[1]
                          + fb_inst)
        f_total_inst = pt.maximum(fs_inst + fb_inst, 1e-30)
        A_eff = model_flux / f_total_inst
        model_delta_mag = -2.5 * pt.log10(pt.maximum(A_eff, 1e-30))

        self._compiled_delta_mag = pytensor.function(
            inputs=[t_input, obs_pos_input, inst_idx] + param_symbols,
            outputs=model_delta_mag,
            on_unused_input='ignore'
        )

        # Baseline flux at a given parameter point, used by plot() to normalize
        # the data onto the same Δmag scale as the model curves.
        self._compiled_f_total = pytensor.function(
            inputs=[inst_idx] + param_symbols,
            outputs=f_total_inst,
            on_unused_input='ignore'
        )

    def plot(self, system, points, filename_prefix="debug"):
        if isinstance(points, dict): points = [points]
        if len(points) == 0: return

        # Model time grid: ±5 tE around t_0 when known, else full data span
        cm = self.config_manager
        _lens_name = (cm.system_config.get("lens") or [{}])[0].get("name", "0")

        def _get_param(base_param):
            # Try numeric index form first (user-provided params), then name form
            # (derived params stored by finalize_user_params under the name key).
            for key in (f"lens.0.{base_param}", f"lens.{_lens_name}.{base_param}"):
                d = cm.user_params.get(key)
                if d is not None:
                    return d.get("initval") if isinstance(d, dict) else float(d)
            return None

        t0 = _get_param("t_0")
        tE = _get_param("t_E")
        if t0 is not None and tE is not None:
            t_model = np.linspace(t0 - 5.0*tE, t0 + 5.0*tE, 2000).astype(np.float64)
        else:
            t_model = np.linspace(self.time.min(), self.time.max(), 2000).astype(np.float64)

        # Each instrument gets its own color.  Model lines are one per unique
        # observer_location: multiple earth instruments share one model curve
        # (parallax between terrestrial sites is negligible unless lat/lon is
        # explicitly specified, in which case each site is a distinct string).
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        inst_color = {i: colors[i % len(colors)] for i in range(self.n_elements)}

        # Map unique observer_location strings to the first instrument with that location.
        unique_observers = []
        obs_to_inst = {}
        for i in range(self.n_elements):
            obs_loc = self.config[i].get("observer_location", "earth")
            if obs_loc not in obs_to_inst:
                unique_observers.append(obs_loc)
                obs_to_inst[obs_loc] = i

        # Absolute barycentric positions for each unique observer over the model grid.
        # Both the symbolic PSPL path and the MulensModel Op expect absolute barycentric AU;
        # get_magnification converts internally to geocentric deviations as needed.
        obs_model_pos = {
            obs_loc: self.get_observer_position(t_model, observer_location=obs_loc)
            for obs_loc in unique_observers
        }

        def _point_values(point):
            return [
                float(np.squeeze(np.asarray(point.get(p.label, p.initval))))
                if getattr(p.value, "ndim", 0) == 0
                else np.atleast_1d(point.get(p.label, p.initval))
                for p in system.plot_params
            ]

        # Per-instrument baseline magnitude from the f_total of the plotted
        # point (first point when several are drawn), so the data land on the
        # same Δmag scale as the self-normalized model curves.  Normalizing by
        # the stage-1 fs_init estimate instead would shift the data by any
        # error in that estimate, faking a constant model-data offset.
        ref_values = _point_values(points[0])
        mag_baseline = np.array([
            -2.5 * np.log10(max(float(self._compiled_f_total(i, *ref_values)), 1e-30))
            for i in range(self.n_elements)
        ])

        def draw(ax):
            for i in range(self.n_elements):
                mask = (self.inst_map == i)
                delta_mag = self.mag[mask] - mag_baseline[i]
                ax.errorbar(
                    self.time[mask], delta_mag, yerr=self.err[mask],
                    fmt='.', color=inst_color[i], alpha=0.6, zorder=1, label=self.names[i]
                )
            for obs_loc in unique_observers:
                i = obs_to_inst[obs_loc]
                obs_pretty = obs_model_pos[obs_loc]
                for point in points:
                    param_values = _point_values(point)
                    try:
                        y_model = self._compiled_delta_mag(t_model, obs_pretty, i, *param_values)
                        alpha = 0.8 if len(points) == 1 else 0.1
                        ax.plot(t_model, y_model, '-', color=inst_color[i], lw=1.5, alpha=alpha, zorder=2)
                    except Exception as e:
                        logger.warning(f"Model eval failed for observer '{obs_loc}': {e}")
            ax.set_xlabel("Time [BJD]")
            ax.set_ylabel("mag − mag$_0$")
            ax.invert_yaxis()
            ax.legend()

        fig, ax = plt.subplots(figsize=(12, 6))
        draw(ax)
        fig.tight_layout()
        fig.savefig(f"{filename_prefix}_mulens.pdf")
        plt.close(fig)

        if t0 is not None and tE is not None:
            fig_z, ax_z = plt.subplots(figsize=(12, 6))
            draw(ax_z)
            ax_z.set_xlim(t0 - 3.0 * tE, t0 + 3.0 * tE)
            fig_z.tight_layout()
            fig_z.savefig(f"{filename_prefix}_mulens_zoom.pdf")
            plt.close(fig_z)
