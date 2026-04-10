import numpy as np
import pandas as pd
import os

import pytensor.tensor as pt
import pymc as pm

from astropy.coordinates import SkyCoord
import astropy.units as u

from exozippy.components.component import Component
from exozippy.components.parameter import Parameter
from exozippy.physics import PHYSICS_REGISTRY
from .utils import get_deltas

class MulensInstrument(Component):
    def __init__(self, config, config_manager):
        super().__init__(config, config_manager)
        self.label = "Microlensing Data"

        # Metadata
        self.files = [c.get("file") for c in self.config]
        self.coords = [c.get("coords") for c in self.config]  # "RA Dec" string
        self.t0_par = [c.get("t0_par", 2450000.0) for c in self.config]

    def load_data(self):
        """
        Loads photometry and pre-calculates parallax shifts (delta_N, delta_E).
        This keeps Astropy/ERFA calls outside the PyMC sampling loop.
        """

        all_times, all_mags, all_errs, inst_indices = [], [], [], []
        all_dn, all_de = [], []

        for i, file in enumerate(self.files):
            # Load raw data (assumes BJD, Mag, Err)
            df = pd.read_csv(file, sep=r'\s+', engine='c', header=None, comment='#')
            t, m, e = df.iloc[:, 0].values, df.iloc[:, 1].values, df.iloc[:, 2].values

            # Pre-calculate Earth position shifts for this dataset
            skycoord = SkyCoord(self.coords[i], unit=(u.hourangle, u.deg))
            deltas = get_deltas(t, self.t0_par[i], skycoord)

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

    def build_parameters(self):
        prefix = "mulens"
        # We define Source Flux and Blend Flux per instrument
        # In Microlensing, these are often called 'fs' and 'fb'
        parameters = {
            "f_source": {"initval": 1.0, "lower": 0.0},
            "f_blend": {"initval": 0.0, "lower": -0.1}  # Allow slight negative for sky sub
        }
        self.build_pars_from_dict(parameters, shape=(self.n_elements,), prefix=prefix)

    def build_likelihood(self, model, lenses):
        """
        lenses: The Lens component containing t0, u0, tE, and pi_E
        """
        # 1. Wrap static data as PyMC Constants
        t = pm.ConstantData("mu_time", self.time)
        dn = pm.ConstantData("mu_delta_n", self.delta_n)
        de = pm.ConstantData("mu_delta_e", self.delta_e)
        obs_mag = pm.ConstantData("mu_obs_mag", self.mag)
        obs_err = pm.ConstantData("mu_obs_err", self.err)

        # 2. Get Magnification from the Lens (Symbolic)
        # Assumes lens.get_magnification() implements Paczyński + Parallax
        A = lenses.get_magnification(t, dn, de)

        # 3. Construct Flux Model: F = fs * A + fb
        # Map parameters to instruments via inst_map
        fs = self.f_source.value[self.inst_map]
        fb = self.f_blend.value[self.inst_map]

        model_flux = fs * A + fb

        # 4. Convert model flux to magnitude (if data is in mags)
        # Mag = -2.5 * log10(Flux) + zero_point (Zero point is usually arbitrary/fixed)
        model_mag = -2.5 * pt.log10(model_flux)

        pm.Normal(
            "microlensing_obs",
            mu=model_mag,
            sigma=obs_err,
            observed=obs_mag
        )