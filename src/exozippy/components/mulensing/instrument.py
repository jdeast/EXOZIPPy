import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pytensor.tensor as pt
import pytensor
import pymc as pm

from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.coordinates import get_body_barycentric, solar_system_ephemeris
from astropy.time import Time

from exozippy.components.component import Component
from .utils import get_deltas


class MulensInstrument(Component):
    def __init__(self, config, config_manager):
        super().__init__(config, config_manager)
        self.label = "Microlensing Data"

        # Metadata
        self.files = [c.get("file") for c in self.config]
        self.coords = [c.get("coords") for c in self.config]
        self.t0_par = [c.get("t0_par", 2450000.0) for c in self.config]

    @property
    def prefix(self):
        return "mulensinst"

    def load_data(self, observer_location='earth'):
        """Loads photometry and pre-calculates parallax shifts."""
        all_times, all_mags, all_errs, inst_indices, all_obspos = [], [], [], [], []

        # to initialize f_source
        self.fs_init = []

        for i, file in enumerate(self.files):
            df = pd.read_csv(file, sep=r'\s+', engine='c', header=None, comment='#')
            t, m, e = df.iloc[:, 0].values, df.iloc[:, 1].values, df.iloc[:, 2].values

            # Pre-calculate Earth position shifts (so ERFA stays out of the MCMC)
            skycoord = SkyCoord(self.coords[i], unit=(u.hourangle, u.deg))

            med_mag = np.median(m)
            self.fs_init.append(10.0 ** (-0.4 * med_mag))

            xyz_au = self.get_observer_position(t, observer_location=observer_location)

            all_times.append(t)
            all_mags.append(m)
            all_errs.append(e)
            all_obspos.append(xyz_au)
            inst_indices.append(np.full(len(t), i))

        self.time = np.concatenate(all_times).astype(float)
        self.mag = np.concatenate(all_mags).astype(float)
        self.err = np.concatenate(all_errs).astype(float)
        self.observer_pos = np.vstack(all_obspos).astype(float)
        self.inst_map = np.concatenate(inst_indices).astype(int)

    def get_observer_position(self, time, observer_location='earth'):
        """
        Pre-calculates Earth's SSB position for all observation times.
        This is done ONCE before the MCMC starts.
        """
        # Use JPL ephemeris for micro-arcsecond reliability
        solar_system_ephemeris.set('jpl')

        # Convert BJD_TDB times to Astropy Time objects
        # Technically, I think we want JD_UTC here, but I don't think it matters, even at Roman precision
        t_obj = Time(time, format='jd', scale='tdb')

        # Get Earth's position in the Solar System Barycenter (SSB) frame
        # Result is an (N, 3) array in Kilometers
        return get_body_barycentric(observer_location, t_obj).xyz.to('au').value.T

    def build_parameters(self, model):
        parameters = {
            "err_scale": None
        }
        self.build_pars_from_dict(parameters, shape=(self.n_elements,))

    def build_map(self, system):
        # Convert the integer map into a PyTensor variable for indexing
        self.inst_map_tensor = pt.as_tensor_variable(self.inst_map).astype("int32")

    def build_dependent_parameters(self, model, system):
        # f_source and f_blend depend on the scale of the loaded data
        fs_inits = np.array(self.fs_init)

        parameters = {
            "f_source": {
                "initval": fs_inits,
                "init_scale": fs_inits * 0.1  # Set NUTS tuning scale proportional to flux
            },
            "f_blend": {
                "initval": np.zeros(self.n_elements),
                "init_scale": fs_inits * 0.1
            }
        }
        self.build_pars_from_dict(parameters, shape=(self.n_elements,))

    def build_likelihood(self, model, system):
        # 1. Constants
        t = pm.Data("mu_time", self.time)
        obs_pos = pm.Data("obs_pos", self.observer_pos)
        obs_mag = pm.Data("mu_obs_mag", self.mag)
        obs_err = pm.Data("mu_obs_err", self.err)

        # 2. Magnification from the Lens
        # (Assuming single lens at index 0 for PSPL)
        A = system.lens.get_magnification(t, obs_pos, system, index=0)
        #A = system.lens.get_magnification_op(t, obs_pos, system, index=0)

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

        A = system.lens.get_magnification(t_input, obs_pos_input, system, index=0)

        fs_inst = self.f_source.value[inst_idx]
        fb_inst = self.f_blend.value[inst_idx]

        model_flux = fs_inst * A + fb_inst
        safe_flux = pt.maximum(model_flux, 1e-12)
        model_mag = -2.5 * pt.log10(safe_flux)

        self._compiled_mag = pytensor.function(
            inputs=[t_input, obs_pos_input, inst_idx] + param_symbols,
            outputs=model_mag,
            on_unused_input='ignore'
        )

    def plot(self, system, points, filename_prefix="debug"):
        if isinstance(points, dict): points = [points]
        if len(points) == 0: return

        for i in range(self.n_elements):
            plt.figure(figsize=(10, 6))

            mask = (self.inst_map == i)
            t_data = self.time[mask]
            m_data = self.mag[mask]
            e_data = self.err[mask]
            obs_pos = self.observer_pos[mask,:]

            # Generate a dense, smooth time grid
            t_pretty = np.linspace(t_data.min(), t_data.max(), 2000).astype(np.float64)

            # Smoothly interpolate the parallax shifts (avoids slow Astropy calls)
            obsx_pretty = np.interp(t_pretty, t_data, obs_pos[:,0])
            obsy_pretty = np.interp(t_pretty, t_data, obs_pos[:,1])
            obsz_pretty = np.interp(t_pretty, t_data, obs_pos[:,2])
            obs_pretty = np.column_stack((obsx_pretty, obsy_pretty, obsz_pretty))

            # 1. Plot spaghetti models
            for point in points:
                param_values = [
                    float(np.squeeze(np.asarray(point.get(p.label, p.initval)))) if getattr(p.value, "ndim", 0) == 0
                    else np.atleast_1d(point.get(p.label, p.initval)) for p in system.plot_params]

                try:
                    y_model = self._compiled_mag(t_pretty, obs_pretty, i, *param_values)
                    alpha = 0.8 if len(points) == 1 else 0.1
                    plt.plot(t_pretty, y_model, 'r-', lw=1.5, alpha=alpha, zorder=2)
                except Exception as e:
                    print(f"Warning: Microlensing model eval failed: {e}")
                    continue

            # 2. Plot raw data
            plt.errorbar(t_data, m_data, yerr=e_data, fmt='k.', alpha=0.5, zorder=1, label=self.names[i])

            # Invert Y-axis since magnitudes go down as flux goes up!
            plt.gca().invert_yaxis()
            plt.xlabel("Time [BJD]")
            plt.ylabel("Magnitude")
            plt.title(f"Microlensing Lightcurve: {self.names[i]}")
            plt.legend()
            plt.tight_layout()

            plt.savefig(f"{filename_prefix}_mulens_{self.names[i]}.pdf")
            plt.close()