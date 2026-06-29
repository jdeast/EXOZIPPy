import logging
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

import pymc as pm
import pytensor.tensor as pt
from scipy.interpolate import CubicSpline

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
        self._raw_time_list = []
        all_obspos = []
        all_obspos_abs = []

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

            self.fs_init.append(
                self._estimate_baseline_flux(t, m, xyz_delta, ra_rad, dec_rad)
            )
            self.q_source_init.append(
                self._estimate_q_source(t, m, xyz_delta, ra_rad, dec_rad)
            )

            self._check_data_format(
                t, m, e, xyz_delta, ra_rad, dec_rad,
                self.fs_init[-1], self.q_source_init[-1],
                self.config[i].get("file", f"instrument {i}"),
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
        self.observer_pos     = np.vstack(all_obspos).astype(float)      # geocentric deviations (for _estimate_baseline_flux)
        self.observer_pos_abs = np.vstack(all_obspos_abs).astype(float)  # absolute barycentric (for get_magnification_op)

    def _check_data_format(self, t, m, e, xyz_delta, ra_rad, dec_rad,
                           f_total_init, q_source_init, label):
        """Warn if peak residuals suggest data is in flux units, not magnitudes.

        The baseline is always self-consistent (f_total_init is derived from it),
        so we check the event peak: if |model_mag_peak - obs_peak| >> sigma,
        the model and data are in incompatible spaces (flux vs. magnitude).
        """
        if f_total_init <= 0:
            return

        # Re-derive trajectory to identify peak epochs (same as _estimate_baseline_flux)
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

        peak_mask = A_traj > 1.5
        if np.sum(peak_mask) < 3:
            return

        # Model magnitudes at peak: what we expect to see if data are in magnitudes
        q = float(np.clip(q_source_init, 0.05, 1.95))
        model_flux_peak = f_total_init * (q * A_traj[peak_mask] + (1.0 - q))
        model_mag_peak  = -2.5 * np.log10(np.maximum(model_flux_peak, 1e-30))

        residuals   = m[peak_mask] - model_mag_peak
        rms_peak    = float(np.sqrt(np.mean(residuals ** 2)))
        typical_err = float(np.median(e))

        if typical_err > 0 and rms_peak > 10.0 * typical_err:
            logger.warning(
                f"[{label}] Peak residuals ({rms_peak:.3g}) are "
                f"{rms_peak / typical_err:.0f}× the typical error ({typical_err:.3g}). "
                f"Data may be in flux units — add 'data_format: flux' to the YAML "
                f"config block for this instrument if so."
            )

    def _estimate_baseline_flux(self, t, m, xyz_au, ra_rad, dec_rad):
        """
        Estimate the unmagnified (baseline) total flux for one instrument.

        Strategy:
        - Compute the Paczynski impact-parameter trajectory u(t) for this
          observer using geocentric-deviation positions (Skowron+2011 convention)
          and the lens parameters already available in user_params.
        - If the data contains observations near baseline (A < 1.05),
          return the median flux of those points.
        - Otherwise (peak-only coverage, e.g. Spitzer), divide the
          estimated peak flux by the peak magnification A(u_min) to
          recover the baseline.

        Falls back to the data median if t_0 or u_0 are not yet in
        user_params (e.g. the user has not supplied them).
        """
        cm = self.config_manager

        def _get(key, default=None):
            data = cm.user_params.get(key)
            if data is None:
                return default
            return data.get("initval", default) if isinstance(data, dict) else float(data)

        t0    = _get("lens.0.t_0")
        u0    = _get("lens.0.u_0")
        tE    = _get("lens.0.t_E")
        pi_E_N = _get("lens.0.pi_E_N", 0.0)
        pi_E_E = _get("lens.0.pi_E_E", 0.0)

        if t0 is None or u0 is None:
            return 10.0 ** (-0.4 * np.median(m))

        # Project heliocentric observer positions onto the sky-plane
        # North/East axes (same convention as get_magnification).
        x, y, z = xyz_au[:, 0], xyz_au[:, 1], xyz_au[:, 2]
        delta_e = -x * np.sin(ra_rad) + y * np.cos(ra_rad)
        delta_n = (-x * np.cos(ra_rad) * np.sin(dec_rad)
                   - y * np.sin(ra_rad) * np.sin(dec_rad)
                   + z * np.cos(dec_rad))

        tE_safe = max(abs(float(tE)), 1.0) if tE is not None else 30.0
        tau   = (t - float(t0)) / tE_safe
        # MulensModel convention: minus on both N and E in tau, plus on N in u.
        tau_p = tau - delta_n * float(pi_E_N) - delta_e * float(pi_E_E)
        u_p   = float(u0) + delta_n * float(pi_E_E) - delta_e * float(pi_E_N)
        u_traj = np.sqrt(tau_p ** 2 + u_p ** 2)
        A_traj = (u_traj ** 2 + 2.0) / (u_traj * np.sqrt(u_traj ** 2 + 4.0))

        # Baseline-covered: at least 3 points with A < 1.05
        baseline_mask = A_traj < 1.05
        if np.sum(baseline_mask) >= 3:
            return 10.0 ** (-0.4 * np.median(m[baseline_mask]))

        # Peak-only: back-calculate from u_min over the observed window.
        # Using the 10th magnitude percentile (brightest ~10%) as F_peak_est
        # and assuming pure source (q_frac = 1) as the zeroth-order approximation.
        u_min  = max(float(np.min(u_traj)), 1e-6)
        A_peak = (u_min ** 2 + 2.0) / (u_min * np.sqrt(u_min ** 2 + 4.0))
        F_peak_est = 10.0 ** (-0.4 * np.percentile(m, 10))
        return F_peak_est / A_peak

    def _estimate_q_source(self, t, m, xyz_delta, ra_rad, dec_rad):
        """Estimate q_source = f_source / f_total from data and initial lens params.

        For baseline-covered instruments (>=3 points with A_traj < 1.05):
          q_source = (A_eff_obs - 1) / (A_peak_model - 1)
        where A_eff_obs is the ratio of observed peak flux to observed baseline flux
        and A_peak_model is A(u_min) from the initial PSPL trajectory.

        Returns a float (clamped to 0.05–1.95), or 0.95 if underdetermined.
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
        pi_E_N = _get("lens.0.pi_E_N", 0.0)
        pi_E_E = _get("lens.0.pi_E_E", 0.0)

        if t0 is None or u0 is None:
            return 0.95

        x, y, z = xyz_delta[:, 0], xyz_delta[:, 1], xyz_delta[:, 2]
        delta_e = -x * np.sin(ra_rad) + y * np.cos(ra_rad)
        delta_n = (-x * np.cos(ra_rad) * np.sin(dec_rad)
                   - y * np.sin(ra_rad) * np.sin(dec_rad)
                   + z * np.cos(dec_rad))

        tE_safe = max(abs(float(tE)), 1.0) if tE is not None else 30.0
        tau = (t - float(t0)) / tE_safe
        tau_p = tau - delta_n * float(pi_E_N) - delta_e * float(pi_E_E)
        u_p = float(u0) + delta_n * float(pi_E_E) - delta_e * float(pi_E_N)
        u_traj = np.sqrt(tau_p**2 + u_p**2)
        A_traj = (u_traj**2 + 2.0) / (u_traj * np.sqrt(u_traj**2 + 4.0))

        baseline_mask = A_traj < 1.05
        if np.sum(baseline_mask) < 3:
            return 0.95

        f_baseline = 10.0 ** (-0.4 * np.median(m[baseline_mask]))

        u_min = max(float(np.min(u_traj)), 1e-6)
        A_peak_model = (u_min**2 + 2.0) / (u_min * np.sqrt(u_min**2 + 4.0))
        if A_peak_model < 1.01:
            return 0.95

        # Observed peak: maximum flux among the top 10% PSPL-ordered points.
        # Median would underestimate the peak for sharp caustic crossings where
        # the true peak spans only a few data points within the top 10%.
        n_peak = max(1, len(m) // 10)
        peak_idx = np.argsort(u_traj)[:n_peak]
        f_peak_obs = np.max(10.0 ** (-0.4 * m[peak_idx]))

        A_eff_obs = f_peak_obs / f_baseline
        q_est = (A_eff_obs - 1.0) / (A_peak_model - 1.0)
        q_est = float(np.clip(q_est, 0.05, 1.95))
        logger.debug(
            f"q_source estimate: A_eff_obs={A_eff_obs:.4f}, A_peak_model={A_peak_model:.4f} → q={q_est:.4f}"
        )
        return q_est

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
        # can resolve initial values.  Also push the data-estimated q_source as a
        # RANK_DERIVED_DATA hint so it overrides the defaults.yaml 0.95 while still
        # yielding to any explicit user override in params.yaml (RANK_USER wins).
        for i in range(self.n_elements):
            q = q_source_init[i]
            f_source_guess = f_total_init[i] * q
            f_blend_guess = f_total_init[i] * (1.0 - q)
            self.config_manager.add_hint(f"{self.prefix}.{i}.f_source", f_source_guess)
            self.config_manager.add_hint(f"{self.prefix}.{i}.f_blend", f_blend_guess)
            self.config_manager.add_hint(
                f"{self.prefix}.{i}.q_source", q, rank=RANK_DERIVED_DATA
            )

        self.manifest = {
            "log_f_total": {"initval": np.log10(f_total_init)},
            "q_source": None,
            "f_source": "default",
            "f_blend": "default",
            "err_scale": None,
        }

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
            raise NotImplemented
            expected_fs = system.sed.Source.get_band_flux("OGLE_I") # this is the part we need from the SED
            sigma_fs = 0.05 * expected_fs
            pm.Potential(f"{self.prefix}_sedprior",
                         -0.5 * pt.sqr((self.f_source.value - expected_fs) / sigma_fs))
        else:
            if False:
                # Linear anchor: f_source ~ 0.2 * f_bol
                # this accounts for heavy extinction in the bulge, but is highly uncertain
                expected_fs = pt.log(0.2 * system.star.fbol.value[system.lens.source_map])
                sigma_fs = 1.0
                log_f_source = pt.log(self.f_source.value)
                pm.Potential(f"{self.prefix}.fbolprior",
                             -0.5 * pt.sqr((log_f_source - expected_fs) / sigma_fs))

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

        system.lens.resolve_auto_vbbl(self.time, index=0)
        A = system.lens.get_magnification_op(
            t, self.observer_pos_abs, system, index=0, u1=u1, bandpass=bandpass
        )

        # 3. Flux Model
        fs = self.f_source.value[self.inst_map_tensor]
        fb = self.f_blend.value[self.inst_map_tensor]
        k_scale = self.err_scale.value[self.inst_map_tensor]

        model_flux = fs * A + fb

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

        A = system.lens.get_magnification_op(t_input, obs_pos_input, system, index=0)

        fs_inst = self.f_source.value[inst_idx]
        fb_inst = self.f_blend.value[inst_idx]

        # Δmag = mag(t) − mag_baseline = −2.5·log10(A_eff).
        # Zero at baseline, negative when brighter, independent of f_total.
        model_flux = fs_inst * A + fb_inst
        f_total_inst = pt.maximum(fs_inst + fb_inst, 1e-30)
        A_eff = model_flux / f_total_inst
        model_delta_mag = -2.5 * pt.log10(pt.maximum(A_eff, 1e-30))

        self._compiled_delta_mag = pytensor.function(
            inputs=[t_input, obs_pos_input, inst_idx] + param_symbols,
            outputs=model_delta_mag,
            on_unused_input='ignore'
        )

    def plot(self, system, points, filename_prefix="debug"):
        if isinstance(points, dict): points = [points]
        if len(points) == 0: return

        # Model time grid: ±5 tE around t_0 when known, else full data span
        cm = self.config_manager
        def _get_param(key):
            d = cm.user_params.get(key)
            if d is None: return None
            return d.get("initval") if isinstance(d, dict) else float(d)

        t0 = _get_param("lens.0.t_0")
        tE = _get_param("lens.0.t_E")
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

        # Per-instrument baseline magnitude from the data-derived fs_init.
        # Δmag = mag(t) − mag_baseline is 0 at baseline for every instrument,
        # so all datasets land on the same scale with no model-flux dependency.
        mag_baseline = np.array([
            -2.5 * np.log10(max(f, 1e-30)) for f in self.fs_init
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
                    param_values = [
                        float(np.squeeze(np.asarray(point.get(p.label, p.initval))))
                        if getattr(p.value, "ndim", 0) == 0
                        else np.atleast_1d(point.get(p.label, p.initval))
                        for p in system.plot_params
                    ]
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
