# this is a pure AI-generated, untested class as a starting point/placeholder.

# generic imports
import numpy as np
import pandas as pd
from pathlib import Path

# plotting imports
import matplotlib.pyplot as plt

# astronomy imports
import astropy.units as u

# pymc imports
import pymc as pm
import pytensor.tensor as pt
from exoplanet_core.pymc import ops as ops

# local imports
from .parameter import Parameter
from .component import Component

# debugging imports
import ipdb

class Transit(Component):
    def __init__(self, config, user_params):
        self.config = config
        self.user_params = user_params
        self.ninstruments = len(self.config)
        self.names = [c.get("name") for c in self.config]
        self.files = [c.get("file") for c in self.config]

        # Photometry defaults
        self.units = [c.get("unit", u.dimensionless_unscaled) for c in self.config]
        self.filters = [c.get("filter", "Kepler") for c in self.config]

        self.baseline_init = [None] * self.ninstruments
        self.jittervar_lower = [None] * self.ninstruments
        self.load_all_data()

    def load_data(self):
        """
        Vectorized loader for photometry that handles concatenated data and
        padded block-diagonal detrending matrices, mirroring RVInstrument.
        """
        all_times, all_fluxes, all_errs, inst_indices = [], [], [], []
        all_detrend = []

        for i, file in enumerate(self.files):
            df = pd.read_csv(file, sep=r'\s+', engine='c', header=None, comment='#')

            n_obs = len(df)

            all_times.append(df.iloc[:, 0].values)
            all_fluxes.append(df.iloc[:, 1].values)
            all_errs.append(df.iloc[:, 2].values)
            inst_indices.append(np.full(n_obs, i))

            # Baseline is usually near 1.0 for normalized relative flux
            self.baseline_init[i] = np.median(df.iloc[:, 1].values)
            self.jittervar_lower[i] = -0.95 * (np.min(df.iloc[:, 2].values) ** 2)

            # Capture extra columns (detrending vectors)
            if df.shape[1] > 3:
                all_detrend.append(df.iloc[:, 3:].values.astype(float))
            else:
                all_detrend.append(np.empty((n_obs, 0)))

        # 1. Flatten standard vectors
        self.time = np.concatenate(all_times).astype(float)
        self.flux = np.concatenate(all_fluxes).astype(float)
        self.err = np.concatenate(all_errs).astype(float)
        self.inst_map = np.concatenate(inst_indices).astype(int)
        self.n_total_obs = len(self.time)

        # 2. Build the Padded Block Diagonal Matrix
        self.n_detrend_per_inst = [d.shape[1] for d in all_detrend]
        self.total_detrend_cols = sum(self.n_detrend_per_inst)

        self.detrend_matrix = np.zeros((self.n_total_obs, self.total_detrend_cols))

        current_row = 0
        current_col = 0
        for i, d_block in enumerate(all_detrend):
            n_r, n_c = d_block.shape
            if n_c > 0:
                self.detrend_matrix[current_row:current_row + n_r, current_col:current_col + n_c] = d_block

            current_row += n_r
            current_col += n_c

    def build_parameters(self, model):
        prefix = f"transit"

        # 1. Photometric Baseline (Analogous to Gamma)
        self.baseline = Parameter(f"{prefix}.baseline",
                                  lower=-1e6, upper=1e6,
                                  initval=self.baseline_init, init_scale=0.01,
                                  unit=u.dimensionless_unscaled, internal_unit=u.dimensionless_unscaled,
                                  latex=r"F_0", description="Relative flux baseline",
                                  user_params=self.user_params, names=self.names,
                                  shape=(self.ninstruments,))
        self.baseline.build_pymc()

        # 2. Jitter
        self.jitter_variance = Parameter(f"{prefix}.jitter_variance",
                                         lower=self.jittervar_lower, upper=1e5,
                                         initval=0.0, init_scale=1e-6,
                                         unit=u.dimensionless_unscaled, internal_unit=u.dimensionless_unscaled,
                                         latex=r"\sigma_J^2", description="Photometric Jitter variance",
                                         user_params=self.user_params, names=self.names,
                                         shape=(self.ninstruments,))
        self.jitter_variance.build_pymc()

        self.jitter = Parameter(f"{prefix}.jitter",
                                expression=lambda: pt.switch(pt.lt(self.jitter_variance.value, 0.0), 0.0,
                                                             pt.sqrt(self.jitter_variance.value)),
                                unit=u.dimensionless_unscaled, internal_unit=u.dimensionless_unscaled,
                                latex=r"\sigma_J",
                                description="Photometric Jitter",
                                user_params=self.user_params, names=self.names,
                                shape=(self.ninstruments,))
        self.jitter.build_pymc()

        # 3. Limb Darkening (Kipping 2013 parametrization for uniform sampling)
        self.q1 = Parameter(f"{prefix}.q1",
                            lower=0.0, upper=1.0,
                            initval=0.5, init_scale=0.1,
                            unit=u.dimensionless_unscaled, internal_unit=u.dimensionless_unscaled,
                            latex=r"q_1", description="Limb darkening q1",
                            user_params=self.user_params, names=self.names,
                            shape=(self.ninstruments,))
        self.q1.build_pymc()

        self.q2 = Parameter(f"{prefix}.q2",
                            lower=0.0, upper=1.0,
                            initval=0.5, init_scale=0.1,
                            unit=u.dimensionless_unscaled, internal_unit=u.dimensionless_unscaled,
                            latex=r"q_2", description="Limb darkening q2",
                            user_params=self.user_params, names=self.names,
                            shape=(self.ninstruments,))
        self.q2.build_pymc()

        # Transform q1, q2 -> u1, u2 for the exoplanet-core solver
        self.u1 = Parameter(f"{prefix}.u1",
                            expression=lambda: 2.0 * pt.sqrt(self.q1.value) * self.q2.value,
                            unit=u.dimensionless_unscaled, internal_unit=u.dimensionless_unscaled,
                            latex=r"u_1", description="Linear limb darkening",
                            user_params=self.user_params, names=self.names)
        self.u1.build_pymc()

        self.u2 = Parameter(f"{prefix}.u2",
                            expression=lambda: pt.sqrt(self.q1.value) * (1.0 - 2.0 * self.q2.value),
                            unit=u.dimensionless_unscaled, internal_unit=u.dimensionless_unscaled,
                            latex=r"u_2", description="Quadratic limb darkening",
                            user_params=self.user_params, names=self.names)
        self.u2.build_pymc()

        # 4. Detrending Coefficients
        if self.total_detrend_cols > 0:
            self.detrend_coeffs = Parameter(f"{prefix}.detrend_coeffs",
                                            initval=0.0, init_scale=10.0,
                                            lower=-1e6, upper=1e6,
                                            unit=u.dimensionless_unscaled, internal_unit=u.dimensionless_unscaled,
                                            user_params=self.user_params, shape=(self.total_detrend_cols,))
            self.detrend_coeffs.build_pymc()

    def build_likelihood(self, model, stars, orbits, planets, star_map, orbit_map):
        time = pm.ConstantData("transit_time", self.time)
        flux = pm.ConstantData("transit_data", self.flux)
        err = pm.ConstantData("transit_err", self.err)

        # 1. Start with the photometric baseline
        lc_model = self.baseline.value[self.inst_map_tensor]

        # 2. Orbital Geometry Broadcast (Solving Kepler's Equation for the lightcurve)
        t_grid = time[:, None]
        tp = orbits.tp.value[orbit_map][None, :]
        n = orbits.n.value[orbit_map][None, :]
        ecc = orbits.ecc.value[orbit_map][None, :]
        cosw = orbits.cosw.value[orbit_map][None, :]
        sinw = orbits.sinw.value[orbit_map][None, :]
        inc = orbits.inc.value[orbit_map][None, :]

        M = (t_grid - tp) * n
        sinf, cosf = ops.kepler(M, ecc + pt.zeros_like(M))

        # r_norm = r / R_*
        a_rstar = planets.ar.value[orbit_map][None, :]
        p_ratio = planets.p.value[orbit_map][None, :]
        r_norm = a_rstar * (1.0 - pt.sqr(ecc)) / (1.0 + ecc * cosf)

        # 3. Calculate Projected Separation (b) and Z-axis (Line of Sight)
        sin_wf = sinw * cosf + cosw * sinf
        cos_wf = cosw * cosf - sinw * sinf
        sin_i = pt.sin(inc)
        cos_i = pt.cos(inc)

        # b = sqrt(X^2 + Y^2)
        b = pt.sqrt(pt.sqr(r_norm * cos_wf) + pt.sqr(r_norm * sin_wf * cos_i))

        # Z > 0 means the planet is in front of the star (transit)
        Z = r_norm * sin_wf * sin_i

        # 4. Map Limb Darkening per instrument
        u1_mapped = self.u1.value[self.inst_map_tensor]
        u2_mapped = self.u2.value[self.inst_map_tensor]
        u_stack = pt.stack([u1_mapped, u2_mapped], axis=0)  # Shape (2, N_obs)

        # 5. Apply the exoplanet-core limb darkened transit model
        for p_idx in range(planets.nplanets):
            b_p = b[:, p_idx]
            p_ratio_p = p_ratio[:, p_idx]
            Z_p = Z[:, p_idx]

            # quad_limb_dark returns the flux decrement (<= 0)
            decrement = ops.quad_limb_dark(u_stack, b_p, p_ratio_p)

            # Mask out secondary eclipses / geometric overlaps behind the star
            decrement = pt.where(Z_p > 0, decrement, 0.0)
            lc_model += decrement

        # 6. Apply Detrending
        if self.total_detrend_cols > 0:
            detrend = pm.ConstantData("transit_detrend", self.detrend_matrix)
            lc_model += pt.dot(detrend, self.detrend_coeffs.value)

        # 7. Likelihood
        sigma = pt.sqrt(pt.sqr(err) + self.jitter_variance.value[self.inst_map_tensor])

        pm.Normal(
            "transit_likelihood",
            mu=lc_model,
            sigma=sigma,
            observed=flux
        )

    def _is_sampling_param(self, attr):
        from .parameter import Parameter
        return isinstance(attr, Parameter) and attr.expression is None

    def plot_model(self, stellar_system, planets, points, filename_prefix="debug", label="model"):
        self.plot_unphased(stellar_system, points, filename_prefix=filename_prefix)
        self.plot_phased(stellar_system, planets, points, filename_prefix=filename_prefix)

    def plot_unphased(self, system, points, filename_prefix="mwe"):
        # Matches the RV plotting architecture; left generic for user implementation
        # based on specific plotting needs for photometry.
        pass

    def plot_phased(self, system, planets, points, filename_prefix="mwe"):
        # Matches the RV plotting architecture; left generic for user implementation.
        pass