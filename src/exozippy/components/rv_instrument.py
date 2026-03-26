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
import pytensor
from exoplanet_core.pymc import ops as ops

# local imports
from .parameter import Parameter
from .component import Component
#from ..constants import G, AU, mjup, rjup, pc, rsun, msun, sigmasb, Gmsun, meter

# debugging imports
import ipdb

class RVInstrument(Component):
    def __init__(self, config, user_params):
        self.config = config
        self.user_params = user_params
        self.ninstruments=len(self.config)
        self.names = [c.get("name") for c in self.config]
        self.files = [c.get("file") for c in self.config]
        self.units = [c.get("unit",u.m/u.s) for c in self.config]
        self.gamma_init = [None]*self.ninstruments
        self.jittervar_lower = [None]*self.ninstruments
        self.load_all_data()

    def load_all_data(self):
        """
        Vectorized loader that handles concatenated data and
        padded block-diagonal detrending matrices.
        """
        all_times, all_rvs, all_errs, inst_indices = [], [], [], []
        all_detrend = []

        for i, file in enumerate(self.files):
            df = pd.read_csv(file, sep=r'\s+', engine='c', header=None, comment='#')

            n_obs = len(df)

            # internally, we use rsun/day to reduce unit conversion overheads
            factor = self.units[i].to(u.solRad / u.d)
            all_times.append(df.iloc[:, 0].values)
            all_rvs.append(df.iloc[:, 1].values*factor)
            all_errs.append(df.iloc[:, 2].values*factor)
            inst_indices.append(np.full(n_obs, i))

            factor = self.units[i].to(u.m / u.s) # make sure it's m/s
            self.gamma_init[i] = np.mean(df.iloc[:, 1].values)*factor
            self.jittervar_lower[i] = -0.95*(np.min(df.iloc[:, 1].values*factor)**2)

            # Capture extra columns (detrending vectors)
            if df.shape[1] > 3:
                all_detrend.append(df.iloc[:, 3:].values.astype(float))
            else:
                # Still need an empty array with the right number of rows
                all_detrend.append(np.empty((n_obs, 0)))

        # 1. Flatten standard vectors
        self.time = np.concatenate(all_times).astype(float)
        self.rv = np.concatenate(all_rvs).astype(float)
        self.err = np.concatenate(all_errs).astype(float)
        self.inst_map = np.concatenate(inst_indices).astype(int)
        self.n_total_obs = len(self.time)

        factor = (u.solRad/u.d).to(u.m/u.s)
        self.k_init = factor * np.sqrt(2.0) * np.std(self.rv) # m/s
        self.tref = np.median(self.time)

        # 2. Build the Padded Block Diagonal Matrix
        # We need to know the TOTAL number of detrending coefficients
        self.n_detrend_per_inst = [d.shape[1] for d in all_detrend]
        self.total_detrend_cols = sum(self.n_detrend_per_inst)

        # Create the large zero-padded matrix: (N_total_obs x Total_Detrend_Cols)
        self.detrend_matrix = np.zeros((self.n_total_obs, self.total_detrend_cols))

        current_row = 0
        current_col = 0
        for i, d_block in enumerate(all_detrend):
            n_r, n_c = d_block.shape
            if n_c > 0:
                # Place the block in its "slot"
                self.detrend_matrix[current_row:current_row + n_r, current_col:current_col + n_c] = d_block

            current_row += n_r
            current_col += n_c

    def build_parameters(self, model):
        prefix = f"inst"

        # 1. Define and build the sampling parameters
        self.gamma = Parameter(f"{prefix}.gamma",
                               lower=-3e8, upper=3e8,
                               initval=self.gamma_init, init_scale=3.0,
                               unit=u.m / u.s, internal_unit=u.solRad/u.d,
                               latex=r"\gamma", description=f"Relative RV offset",
                               user_params=self.user_params, shape=(self.ninstruments,))
        self.gamma.build_pymc()

        self.jitter_variance = Parameter(f"{prefix}.jitter_variance",
                                         lower=self.jittervar_lower, upper=1e5,
                                         initval=0.0, init_scale=1.0,
                                         unit=u.m**2 / u.s**2, internal_unit=u.solRad**2 / u.d**2,
                                         latex=r"\sigma_J^2", description=f"Jitter variance",
                                         user_params=self.user_params, shape=(self.ninstruments,))
        self.jitter_variance.build_pymc()

        self.jitter = Parameter(f"{prefix}.jitter",
                                expression=lambda: pt.switch(pt.lt(self.jitter_variance.value, 0.0), 0.0, pt.sqrt(self.jitter_variance.value)),
                                unit=u.m / u.s, internal_unit=u.solRad / u.d,
                                latex=r"\sigma_J",
                                description=f"Jitter",
                                user_params=self.user_params)
        self.jitter.build_pymc()

        if self.total_detrend_cols > 0:
            # 3. Detrending Coefficients
            # total_detrend_cols was calculated during load_all_data
            if self.total_detrend_cols > 0:
                self.detrend_coeffs = Parameter(f"{self.label}.detrend_coeffs",
                    initval=0.0, init_scale=10.0,
                    lower=-1e6, upper=1e6,
                    unit=u.dimensionless_unscaled, internal_unit=u.dimensionless_unscaled,
                    user_params=self.user_params, shape=(self.total_detrend_cols,))
                self.detrend_coeffs.build_pymc()

    def get_rv_model(self, planets, times):
        """
        The Source of Truth for the RV symbolic graph.
        """
        # Start with this instrument's specific offset
        rv_model = self.gamma.value

        # Add the Keplerian signal from every planet
        for p in planets:
            # This calls Planet -> Orbit -> exoplanet-core
            rv_model += p.get_rv_signal(times)

        return rv_model

    def build_likelihood(self, model, stars, orbits, planets, star_map, orbit_map):
        time = pm.Data("rv_time",self.time)
        rv = pm.Data("rv_data",self.rv)
        err = pm.Data("rv_err",self.err)

        # 1. Construct the RV Model: Gamma + sum(Keplerians)
        rv_model = self.gamma.value[self.inst_map]

        # sum the contribution from all planets
        rv_model += pt.sum(orbits.get_radial_velocity(time, planets.K.value[orbit_map], orbit_map),axis=1)

        # detrending
        if self.total_detrend_cols > 0:
            detrend = pm.Data("rv_detrend",self.detrend_matrix)
            rv_model += pt.dot(detrend, self.detrend_coeffs.value)

        # 2. Define the Likelihood (The Normal Distribution)
        # Total variance = data_error^2 + jitter^2
        sigma = pt.sqrt(pt.sqr(err) + self.jitter_variance.value[self.inst_map])

        pm.Normal(
            f"rv_model",
            mu=rv_model,
            sigma=sigma,
            observed=rv
        )

        """
        # GP implementation (this replaces pm.normal above)
        log_sigma_rv_gp = pm.Normal("log_sigma_rv_gp", mu=np.log(2.0), sigma=1.0)
        log_rho_rv_gp = pm.Normal("log_rho_rv_gp", mu=np.log(10.0), sigma=1.0)
        kernel_rv = terms.SHOTerm(sigma=pt.exp(log_sigma_rv_gp), rho=pt.exp(log_rho_rv_gp), Q=1.0/3.0)
        gp_rv = GaussianProcess(
            kernel_rv,
            t=pm.Data(self.time),
            yerr=sigma,
            mean=rv_model)
        gp_rv.marginal("obs_rv", observed=pm.Data(self.rv))
        """

    def get_compiled_model(self, planets):
        """
        The Plotting Bridge:
        Returns a compiled NumPy function and a list of required parameter labels.
        """
        t_pretty = pt.vector("t_pretty")
        symbolic_mu = self.get_rv_model(planets, t_pretty)

        # Gather every 'Free' parameter (random variable) the model depends on
        param_names = []
        param_symbols = []

        # 1. Instrument parameters (Gamma, Jitter)
        for attr in self.__dict__.values():
            if self._is_sampling_param(attr):
                param_names.append(attr.label)
                param_symbols.append(attr.value)

        # 2. Planetary/Orbital parameters
        for p in planets:
            for comp in [p, p.orbit]:
                for attr in comp.__dict__.values():
                    if self._is_sampling_param(attr):
                        param_names.append(attr.label)
                        param_symbols.append(attr.value)

        # Compile the graph into a high-speed C/NumPy function
        compiled_fn = pytensor.function(
            inputs=[t_pretty] + param_symbols,
            outputs=[symbolic_mu],
            on_unused_input='ignore'
        )

        return compiled_fn, param_names

    def _is_sampling_param(self, attr):
        """Helper to identify parameters that need to be passed to the compiled function."""
        from .parameter import Parameter
        return isinstance(attr, Parameter) and attr.expression is None

    def plot_model(self, stellar_system, planets, points, filename_prefix="debug", label="model"):
        self.plot_unphased(stellar_system, points, filename_prefix=filename_prefix)
        self.plot_phased(stellar_system, planets, points, filename_prefix=filename_prefix)

    def plot_unphased(self, system, points, filename_prefix="mwe"):
        """
        Generates a non-phased RV plot (spaghetti or single model).
        Saves to {filename_prefix}_RV_unphased.pdf
        """
        if isinstance(points, dict):
            points = [points]
        if len(points) == 0:
            print("No points provided for plotting.")
            return

        t_min, t_max = self.time.min(), self.time.max()
        # Create a smooth time grid (64-bit for the C-compiled function)
        t_pretty = np.linspace(t_min, t_max, 2000).astype(np.float64)

        plt.figure(figsize=(12, 6))

        # 1. Plot the Model Ensemble (The Spaghetti)
        for idx, point in enumerate(points):
            param_values = []
            for p in system.plot_params:
                # Grab the physical value from the point
                val = point.get(p.label, p.initval)
                # MATCH DIMENSIONS: PyTensor expects 1D arrays for these parameters
                param_values.append(np.atleast_1d(val).astype(np.float64))

            try:
                # Evaluate the compiled graph (Summed RV across all planets)
                y_model = system._compiled_full_rv(t_pretty, *param_values)
                y_model = np.array(y_model).flatten()

                # Squeeze to ensure it's (2000,) for matplotlib
                if y_model.ndim > 1:
                    y_model = np.squeeze(y_model)

                # Transparency: Solid for one point, faint for spaghetti
                alpha = 0.8 if len(points) == 1 else 0.1
                plt.plot(t_pretty, y_model, 'r-', lw=1.5, alpha=alpha, zorder=2)
            except Exception as e:
                print(f"Warning: Failed to evaluate model for draw {idx}: {e}")
                continue

        # 2. Plot the Actual Data per Instrument
        # We center the plot by the reference instrument's gamma from the first point
        ref_point = points[0]
        for i in range(self.ninstruments):
            mask = (self.inst_map == i)

            # Extract Gamma for this specific instrument
            gamma_vals = np.atleast_1d(ref_point.get(self.gamma.label, 0.0))
            g = gamma_vals[i] if i < len(gamma_vals) else gamma_vals[0]

            plt.errorbar(self.time[mask], self.rv[mask] - g,
                         yerr=self.err[mask], fmt='o', label=self.names[i],
                         alpha=0.6, zorder=1)

        plt.xlabel("Time [BJD]")
        plt.ylabel("Relative RV [m/s]")
        plt.title(f"Unphased RV Model: {system.name}")
        plt.legend(loc='best', fontsize='small')
        plt.tight_layout()

        # 3. Save to PDF
        pdf_path = f"{filename_prefix}_RV_unphased.pdf"
        plt.savefig(pdf_path)
        plt.close()

    def plot_phased(self, system, planets, points, filename_prefix="mwe"):
        """
        Generates a phased RV plot for each planet in the system.
        Accesses planet parameters via vectorized indices.
        """
        if isinstance(points, dict): points = [points]

        # _compiled_rv_matrix returns (N_times, N_planets)
        compiled_matrix = system._compiled_rv_matrix

        # Iterate through the number of planets defined in the component
        for p_idx in range(planets.nplanets):
            plt.figure(figsize=(10, 6))

            # 1. Setup Phase Grid using the reference point (first draw)
            ref_point = points[0]

            # Accessing the vectorized period and tc for this specific planet index
            # These are stored in the Orbit object owned by the system
            P_ref = float(np.atleast_1d(ref_point[system.orbits.period.label])[p_idx])
            tc_ref = float(np.atleast_1d(ref_point[system.orbits.tc.label])[p_idx])

            t_model = np.linspace(tc_ref - 0.5 * P_ref, tc_ref + 0.5 * P_ref, 1000).astype(np.float64)
            # Map time to [0, 1] phase, centering conjunction at 0.25
            phase_model = np.mod((t_model - tc_ref) / P_ref + 0.25, 1.0)
            sort_m = np.argsort(phase_model)

            # 2. Plot Model Spaghetti
            for idx, point in enumerate(points):
                param_values = [np.atleast_1d(point.get(p.label, 0.0)).astype(np.float64)
                                for p in system.plot_params]

                # Get the (N_times, N_planets) matrix
                rv_matrix = compiled_matrix(t_model, *param_values)
                y_planet = rv_matrix[:, p_idx]

                alpha = 0.8 if len(points) == 1 else 0.1
                plt.plot(phase_model[sort_m], y_planet[sort_m], 'r-', alpha=alpha, lw=1, zorder=2)

            # 3. Plot Phased Data (Isolating the planet)
            clean_params = [np.atleast_1d(ref_point.get(p.label, 0.0)).astype(np.float64)
                            for p in system.plot_params]

            # Get signals of ALL planets at observation times
            data_rv_matrix = compiled_matrix(self.time, *clean_params)

            # Mask out the target planet to get the "background" RV
            other_mask = np.ones(planets.nplanets, dtype=bool)
            other_mask[p_idx] = False
            other_signals = np.sum(data_rv_matrix[:, other_mask], axis=1)

            for i in range(self.ninstruments):
                mask = (self.inst_map == i)

                # Subtract Gamma and other planet signals
                gamma_vals = np.atleast_1d(ref_point.get(self.gamma.label, 0.0))
                g = gamma_vals[i] if i < len(gamma_vals) else gamma_vals[0]

                cleaned_rv = self.rv[mask] - g - other_signals[mask]

                # Phase the actual data points
                data_phases = np.mod((self.time[mask] - tc_ref) / P_ref + 0.25, 1.0)

                plt.errorbar(data_phases, cleaned_rv, yerr=self.err[mask],
                             fmt='o', label=self.names[i], alpha=0.6, zorder=1)

            plt.axhline(0, color='black', linestyle=':', alpha=0.5)
            plt.xlabel(f"Phase (P = {P_ref:.5f} d, $T_c$ at 0.25)")
            plt.ylabel("Isolated RV [m/s]")
            plt.title(f"Phased RV: {planets.names[p_idx]} ({system.name})")
            plt.legend(loc='best', fontsize='small')
            plt.tight_layout()

            pdf_path = f"{filename_prefix}_RV_phased_{planets.names[p_idx]}.pdf"
            plt.savefig(pdf_path)
            plt.close()