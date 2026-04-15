import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pytensor.tensor as pt
import pytensor
import pymc as pm

from astropy.coordinates import SkyCoord
import astropy.units as u

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

    def load_data(self):
        """Loads photometry and pre-calculates parallax shifts."""
        all_times, all_mags, all_errs, inst_indices = [], [], [], []
        all_dn, all_de = [], []

        # to initialize f_source
        self.fs_init = []

        for i, file in enumerate(self.files):
            df = pd.read_csv(file, sep=r'\s+', engine='c', header=None, comment='#')
            t, m, e = df.iloc[:, 0].values, df.iloc[:, 1].values, df.iloc[:, 2].values

            # Pre-calculate Earth position shifts (so ERFA stays out of the MCMC)
            skycoord = SkyCoord(self.coords[i], unit=(u.hourangle, u.deg))
            deltas = get_deltas(t, self.t0_par[i], skycoord)

            med_mag = np.median(m)
            self.fs_init.append(10.0 ** (-0.4 * med_mag))

            all_times.append(t)
            all_mags.append(m)
            all_errs.append(e)
            all_dn.append(deltas['N'])
            all_de.append(deltas['E'])
            inst_indices.append(np.full(len(t), i))

        self.time = np.concatenate(all_times).astype(float)
        self.mag = np.concatenate(all_mags).astype(float)
        self.err = np.concatenate(all_errs).astype(float)
        self.delta_n = np.concatenate(all_dn).astype(float)
        self.delta_e = np.concatenate(all_de).astype(float)
        self.inst_map = np.concatenate(inst_indices).astype(int)

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
        t = pm.ConstantData("mu_time", self.time)
        dn = pm.ConstantData("mu_delta_n", self.delta_n)
        de = pm.ConstantData("mu_delta_e", self.delta_e)
        obs_mag = pm.ConstantData("mu_obs_mag", self.mag)
        obs_err = pm.ConstantData("mu_obs_err", self.err)

        # 2. Magnification from the Lens
        # (Assuming single lens at index 0 for PSPL)
        A = system.lens.get_magnification(t, dn, de, index=0)

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
            "microlensing_obs",
            mu=model_mag,
            sigma=sigma,
            observed=obs_mag
        )

    def compile_plotters(self, model, system):
        """Compile fast PyTensor functions for the lightcurve."""
        t_input = pt.vector("mu_t_input")
        dn_input = pt.vector("mu_dn_input")
        de_input = pt.vector("mu_de_input")
        inst_idx = pt.iscalar("mu_inst_idx")

        param_symbols = [p.value for p in system.plot_params]

        A = system.lens.get_magnification(t_input, dn_input, de_input, index=0)

        fs_inst = self.f_source.value[inst_idx]
        fb_inst = self.f_blend.value[inst_idx]

        model_flux = fs_inst * A + fb_inst
        safe_flux = pt.maximum(model_flux, 1e-12)
        model_mag = -2.5 * pt.log10(safe_flux)

        self._compiled_mag = pytensor.function(
            inputs=[t_input, dn_input, de_input, inst_idx] + param_symbols,
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
            dn_data = self.delta_n[mask]
            de_data = self.delta_e[mask]

            # Generate a dense, smooth time grid
            t_pretty = np.linspace(t_data.min(), t_data.max(), 2000).astype(np.float64)

            # Smoothly interpolate the parallax shifts (avoids slow Astropy calls)
            dn_pretty = np.interp(t_pretty, t_data, dn_data)
            de_pretty = np.interp(t_pretty, t_data, de_data)

            # 1. Plot spaghetti models
            for point in points:
                param_values = [
                    float(np.squeeze(np.asarray(point.get(p.label, p.initval)))) if getattr(p.value, "ndim", 0) == 0
                    else np.atleast_1d(point.get(p.label, p.initval)) for p in system.plot_params]

                try:
                    y_model = self._compiled_mag(t_pretty, dn_pretty, de_pretty, i, *param_values)
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