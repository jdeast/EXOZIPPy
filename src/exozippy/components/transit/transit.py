
import logging
import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
import pytensor.tensor as pt
from exoplanet_core.pymc import ops as ops
from exozippy.components.component import Component
from . import physics


class Transit(Component):
    def __init__(self, config, config_manager):
        super().__init__(config, config_manager)
        self.label = "Transit Parameters"
        self.files = [c.get("file") for c in self.config]
        # Filter identity and limb darkening live on the Band component;
        # each instrument references a band block by name.
        self.band_names = [c.get("band") for c in self.config]
        deprecated_filters = [c.get("filter") for c in self.config if c.get("filter")]
        if deprecated_filters:
            logger.warning(
                "transit 'filter:' is deprecated and ignored; reference a "
                "band: block instead (bands carry the filter identity and "
                "limb darkening)."
            )
        self.total_detrend_cols = 0
        self.n_total_obs = 0
        # SED depth-dilution node, built once by build_likelihood and
        # reused by compile_plotters.
        self._dilution_node = None

    @property
    def prefix(self):
        return "transit"

    def sampler_requirements(self):
        """Declare sampler constraints for limb-darkened transit models.

        The quadratic limb-darkening solution vector (exoplanet_core's
        ``quad_solution_vector`` Op) is only differentiable through
        PyTensor's own gradient machinery (used by the C/numba-backed
        'nuts' and 'nutpie' samplers). The installed exoplanet_core's
        jax_support wires the PyTensor Op straight to the raw,
        non-custom_jvp JAX FFI call, so any sampler that funcifies the
        whole logp graph to JAX ('numpyro', 'blackjax') fails at HMC
        init with "cannot be differentiated".
        """
        return {
            'incompatible': {'numpyro', 'blackjax'},
            'recommended': 'nuts',
            'reason': (
                "the transit component's limb-darkening op "
                "(exoplanet_core quad_solution_vector) is not "
                "differentiable through JAX with the installed "
                "exoplanet_core build — use a PyTensor-backed sampler"
            ),
        }

    def load_data(self, system):
        """Stage 1a: Load CSVs and generate data-driven bounds/inits."""
        all_times, all_fluxes, all_errs, inst_indices, all_detrend = [], [], [], [], []
        self.baseline_init = [1.0] * self.n_elements
        self.jittervar_lower = [0.0] * self.n_elements

        # Fixed instrument properties (not fitted parameters), read straight
        # off the config like self.files/self.band_names: exptime is the
        # exposure duration in minutes, ninterp the number of sub-samples
        # used to smear the model over that exposure (EXOFASTv2 parity).
        self.exptime_min = [float(c.get("exptime", 0.0)) for c in self.config]
        self.ninterp = [int(c.get("ninterp", 1)) for c in self.config]

        for i, file in enumerate(self.files):
            df = pd.read_csv(file, sep=r'\s+', engine='c', header=None, comment='#')
            n_obs = len(df)
            all_times.append(df.iloc[:, 0].values)
            all_fluxes.append(df.iloc[:, 1].values)
            all_errs.append(df.iloc[:, 2].values)
            inst_indices.append(np.full(n_obs, i))

            self.baseline_init[i] = np.median(df.iloc[:, 1].values)
            self.jittervar_lower[i] = -0.95 * (np.min(df.iloc[:, 2].values) ** 2)

            if df.shape[1] > 3:
                all_detrend.append(df.iloc[:, 3:].values.astype(float))
            else:
                all_detrend.append(np.empty((n_obs, 0)))

        self.time = np.concatenate(all_times).astype(float)
        self.flux = np.concatenate(all_fluxes).astype(float)
        self.err = np.concatenate(all_errs).astype(float)
        self.inst_map = np.concatenate(inst_indices).astype(int)
        self.n_total_obs = len(self.time)

        self._build_oversample_grid()

        # Block Diagonal Matrix
        self.n_detrend_per_inst = [d.shape[1] for d in all_detrend]
        self.total_detrend_cols = sum(self.n_detrend_per_inst)
        self.detrend_matrix = np.zeros((self.n_total_obs, self.total_detrend_cols))

        r, c = 0, 0
        for d_block in all_detrend:
            n_r, n_c = d_block.shape
            if n_c > 0: self.detrend_matrix[r:r + n_r, c:c + n_c] = d_block
            r, c = r + n_r, c + n_c

    def _build_oversample_grid(self):
        """
        Build the (N_obs, max_ninterp) sub-exposure time grid and matching
        per-observation averaging weights used to smear the model over each
        instrument's exposure time (EXOFASTv2 exofast_chi2v2.pro parity:
        a 2D time grid collapsed with total(modelflux, 2) / ninterp).

        exptime/ninterp are per-instrument, but self.time is concatenated
        across instruments, so instruments may disagree on both. Rather than
        assuming a single ninterp for the whole component, every instrument
        gets its own sub-sampling: observations are padded out to the
        largest ninterp among the active instruments, and the padding
        columns carry weight 0 (they duplicate that observation's own
        timestamp, so they are finite but inert) so the weighted sum still
        averages over only that observation's own instrument's ninterp.
        """
        exptime_days = np.asarray(self.exptime_min, dtype=float) / 1440.0
        ninterp_per_inst = np.asarray(self.ninterp, dtype=int)
        self.max_ninterp = int(ninterp_per_inst.max()) if len(ninterp_per_inst) else 1

        if self.max_ninterp <= 1:
            self.oversample_time = self.time[:, None]
            self.oversample_weights = np.ones((self.n_total_obs, 1))
            return

        ninterp_obs = ninterp_per_inst[self.inst_map]  # (N_obs,)
        exptime_obs = exptime_days[self.inst_map]  # (N_obs,)

        k = self.max_ninterp
        j = np.arange(k)
        valid = j[None, :] < ninterp_obs[:, None]  # (N_obs, k)

        # Evenly spaced sub-times from -exptime/2 to +exptime/2; a lone
        # sample (ninterp==1) sits at the timestamp itself (frac 0).
        denom = np.maximum(ninterp_obs[:, None] - 1, 1)
        frac = j[None, :] / denom - 0.5
        frac = np.where(ninterp_obs[:, None] == 1, 0.0, frac)
        frac = np.where(valid, frac, 0.0)  # padding collapses onto the timestamp

        self.oversample_time = self.time[:, None] + frac * exptime_obs[:, None]
        self.oversample_weights = np.where(valid, 1.0 / ninterp_obs[:, None], 0.0)

    def register_parameters(self, system):
        """Stage 2: Embed data-driven hints into the PyMC manifest."""
        self.manifest = {
            "baseline": {"initval": self.baseline_init},
            "jitter_variance": {"lower": self.jittervar_lower},
            "jitter": "default",
        }

        if self.total_detrend_cols > 0:
            self.manifest["detrend_coeffs"] = {"shape": (self.total_detrend_cols,)}

        # Limb darkening comes from the Band component; map each
        # instrument (and each observation) to its band. A transit model
        # cannot be computed without LD, so a missing/unknown band
        # reference is an error, not a warning.
        if not hasattr(system, "band"):
            raise ValueError(
                "Transit instruments require band: blocks (bands carry the "
                "filter identity and limb-darkening parameters)."
            )
        name_to_idx = {name: i for i, name in enumerate(system.band.names)}
        missing = [
            (self.names[i], n) for i, n in enumerate(self.band_names)
            if n not in name_to_idx
        ]
        if missing:
            raise ValueError(
                f"Transit instrument(s) reference unknown band(s): "
                f"{missing}. Available bands: {list(name_to_idx)}."
            )
        self.band_map = np.array(
            [name_to_idx[n] for n in self.band_names], dtype=int)
        self.obs_band_map = self.band_map[self.inst_map]

    def _build_dilution(self, system):
        """
        Per-instrument SED-predicted depth dilution factor
        F_host / sum_j F_j in the instrument's band, as a (n_elements,)
        tensor (Deterministic "transit.dilution" for diagnostics), or
        None if no instrument's band filter is in the SED's BC grid.
        Instruments whose band filter is unavailable get dilution 1.
        """
        if getattr(self, "_dilution_node", None) is not None:
            return self._dilution_node

        sed = system.sed
        band = system.band
        dils = []
        any_diluted = False
        for i in range(self.n_elements):
            band_idx = int(self.band_map[i])
            filter_key = band.filter_mist[band_idx]
            if filter_key and sed.has_filter(filter_key):
                host = int(band.star_indices[band_idx])
                dils.append(sed.predict_flux_fraction(host, filter_key, system))
                any_diluted = True
            else:
                logger.warning(
                    f"transit {self.names[i]}: band filter '{filter_key}' "
                    f"is not in the SED's BC grid; no depth deblending "
                    f"applied for this instrument."
                )
                dils.append(pt.constant(1.0))
        if not any_diluted:
            return None
        self._dilution_node = pm.Deterministic(
            f"{self.prefix}.dilution", pt.stack(dils))
        return self._dilution_node

    def build_likelihood(self, model, system):
        time = pm.Data("transit_time", self.time)
        flux = pm.Data("transit_data", self.flux)
        err = pm.Data("transit_err", self.err)

        orbits = system.orbit
        planets = system.planet

        # 1. Start with the photometric baseline
        lc_model = self.baseline.value[self.inst_map_tensor]

        # 1b. Per-planet transit/occultation geometry (impact parameter & durations),
        # exposed as Deterministics for diagnostics and plotting (e.g. phased-plot xlim).
        ecc_p = orbits.ecc.value[planets.orbit_map]  # (N_planets,)
        esinw_p = orbits.esinw.value[planets.orbit_map]
        inc_p = orbits.inc.value[planets.orbit_map]
        period_p = orbits.period.value[planets.orbit_map]
        ar_p = planets.ar.value
        p_p = planets.p.value

        # Numerical-stability floor for the geometry below. Keeps arcsin arguments
        # strictly inside (-1, 1) (where its derivative is finite) and denominators
        # away from 0, so a transient excursion during NUTS leapfrog steps (e.g.
        # inc away from 90 deg, or ecc/esinw near 1) can't produce a NaN/inf
        # gradient. Values at the actual posterior mode are far from these floors,
        # so the reported b/t14/tau are unaffected.
        _GEOM_EPS = 1e-6

        sini_p = pt.sin(inc_p)
        cosi_p = pt.cos(inc_p)
        ecc_factor = pt.sqrt(pt.clip(1.0 - pt.sqr(ecc_p), _GEOM_EPS, 1.0))

        denom_minus = pt.clip(1.0 - esinw_p, _GEOM_EPS, np.inf)
        denom_plus = pt.clip(1.0 + esinw_p, _GEOM_EPS, np.inf)
        sini_ar = pt.clip(pt.abs(sini_p * ar_p), _GEOM_EPS, np.inf)

        dur_b = ar_p * cosi_p * (1.0 - pt.sqr(ecc_p)) / denom_minus
        dur_bs = ar_p * cosi_p * (1.0 - pt.sqr(ecc_p)) / denom_plus

        def _arcsin_term(p_offset_sq, dur_bx):
            radicand = pt.clip(p_offset_sq - pt.sqr(dur_bx), 0.0, np.inf)
            arg = pt.clip(pt.sqrt(radicand) / sini_ar, -1.0 + _GEOM_EPS, 1.0 - _GEOM_EPS)
            return pt.arcsin(arg)

        dur_t14 = (period_p / np.pi) * _arcsin_term(pt.sqr(1.0 + p_p), dur_b) * ecc_factor / denom_minus
        dur_t14s = (period_p / np.pi) * _arcsin_term(pt.sqr(1.0 + p_p), dur_bs) * ecc_factor / denom_plus

        dur_tfwhm = (period_p / np.pi) * _arcsin_term(pt.sqr(1.0 - p_p), dur_b) * ecc_factor / denom_minus
        dur_tfwhms = (period_p / np.pi) * _arcsin_term(pt.sqr(1.0 - p_p), dur_bs) * ecc_factor / denom_plus

        dur_tau = (dur_t14 - dur_tfwhm) / 2.0
        dur_taus = (dur_t14s - dur_tfwhms) / 2.0

        pm.Deterministic(f"{self.prefix}.b", dur_b)
        pm.Deterministic(f"{self.prefix}.bs", dur_bs)
        pm.Deterministic(f"{self.prefix}.t14", dur_t14)
        pm.Deterministic(f"{self.prefix}.t14s", dur_t14s)
        pm.Deterministic(f"{self.prefix}.tfwhm", dur_tfwhm)
        pm.Deterministic(f"{self.prefix}.tfwhms", dur_tfwhms)
        pm.Deterministic(f"{self.prefix}.tau", dur_tau)
        pm.Deterministic(f"{self.prefix}.taus", dur_taus)

        # 2. Orbital Geometry Broadcast
        # t_grid carries a sub-exposure axis (N_obs, max_ninterp): each
        # observation's own instrument contributes ninterp evenly-spaced
        # sub-times spanning its own exptime (see _build_oversample_grid);
        # observations from a shorter-ninterp instrument are padded with
        # inert (weight-0) columns so mixed-ninterp instruments don't need
        # a shared/uniform ninterp. With ninterp==1 everywhere this is
        # exactly time[:, None] and reduces to the original computation.
        time_grid = pm.Data("transit_time_grid", self.oversample_time)  # (N_obs, K)
        oversample_weights = pm.Data(
            "transit_oversample_weights", self.oversample_weights)  # (N_obs, K)

        t_grid = time_grid[:, :, None]  # (N_obs, K, 1)
        tp = orbits.tp.value[planets.orbit_map][None, None, :]  # (1, 1, N_planets)
        n = orbits.n.value[planets.orbit_map][None, None, :]
        ecc = ecc_p[None, None, :]
        cosw = orbits.cosw.value[planets.orbit_map][None, None, :]
        sinw = orbits.sinw.value[planets.orbit_map][None, None, :]
        inc = inc_p[None, None, :]

        M = (t_grid - tp) * n
        sinf, cosf = ops.kepler(M, ecc + pt.zeros_like(M))

        a_rstar = planets.ar.value[None, None, :]
        p_ratio = planets.p.value[None, :]

        r_norm = a_rstar * (1.0 - pt.sqr(ecc)) / (1.0 + ecc * cosf)

        sin_wf = sinw * cosf + cosw * sinf
        cos_wf = cosw * cosf - sinw * sinf
        sin_i = pt.sin(inc)
        cos_i = pt.cos(inc)

        # (N_obs, K, N_planets)
        b = pt.sqrt(pt.sqr(r_norm * cos_wf) + pt.sqr(r_norm * sin_wf * cos_i))
        Z = r_norm * sin_wf * sin_i

        # 3. Limb Darkening Setup (per observation, mapped from each
        # instrument's Band). When every band uses the linear law, Band's
        # manifest has no u2; the quadratic term is then zero.
        band = system.band
        u1_mapped = band.u1.value[self.obs_band_map_tensor]  # (N_obs,)
        if "u2" in band.manifest:
            u2_mapped = band.u2.value[self.obs_band_map_tensor]  # (N_obs,)
        else:
            u2_mapped = pt.zeros_like(u1_mapped)

        # 3b. SED deblending (EXOFASTv2 parity): with more than one
        # modeled star, only the host contributes the transit, so the
        # observed depth is diluted by dil = F_host / sum_j F_j in the
        # instrument's band (host = the band's star_ndx).
        dil_obs = None
        if hasattr(system, "sed") and system.star.n_elements > 1:
            dil_inst = self._build_dilution(system)
            if dil_inst is not None:
                dil_obs = dil_inst[self.inst_map_tensor]  # (N_obs,)
        # exoplanet_core's quad_solution_vector returns s in starry's Green's basis,
        # not powers of mu. Converting the quadratic law (u1, u2) into that basis
        # requires the change-of-basis in Agol, Luger & Foreman-Mackey (2020),
        # matching exoplanet.light_curves.limb_dark.get_cl():
        #   c0 = 1 - u1 - 1.5*u2, c1 = u1 + 2*u2, c2 = -0.25*u2
        #   norm = dot(s_off, c) = pi*(c0 + c1/1.5), s_off = [pi, 2pi/3, 0]
        c0 = 1.0 - u1_mapped - 1.5 * u2_mapped
        c1 = u1_mapped + 2.0 * u2_mapped
        c2 = -0.25 * u2_mapped
        ld_norm = np.pi * (c0 + c1 / 1.5)

        # 4. Exoplanet-core Transit Model
        for p_idx in range(planets.n_elements):
            b_p = b[:, :, p_idx]  # (N_obs, K) sky-plane separation in units of R_*
            Z_p = Z[:, :, p_idx]  # (N_obs, K) line-of-sight coord (+ = planet in front of star)
            r_p = planets.p.value[p_idx]  # scalar R_p/R_*

            # quad_solution_vector(b, r) -> (N_obs, K, 3) solution vector s.
            # Broadcast scalar r_p to (N_obs, K) following the ops.kepler() pattern.
            sol = ops.quad_solution_vector(b_p, r_p + pt.zeros_like(b_p))

            # Limb-darkened flux fraction: 1.0 off-disk, <1.0 during transit.
            # Verified against brute-force disk integration: lc = dot(s, c) / dot(s_off, c)
            # c0/c1/c2/ld_norm are (N_obs,); broadcast against the sub-exposure axis.
            flux_frac = (
                sol[:, :, 0] * c0[:, None] +
                sol[:, :, 1] * c1[:, None] +
                sol[:, :, 2] * c2[:, None]
            ) / ld_norm[:, None]  # (N_obs, K)

            # Fraction of stellar flux blocked (0 off-disk, ≈ r² at disk centre)
            blocked = 1.0 - flux_frac

            # Primary transit only; secondary eclipse (planet behind star) has Z < 0
            blocked = pt.where(Z_p > 0.0, blocked, 0.0)
            if dil_obs is not None:
                blocked = blocked * dil_obs[:, None]

            # Collapse the sub-exposure axis: a weighted mean over each
            # observation's own ninterp sub-samples (weights sum to 1 per
            # row; padding columns carry weight 0). With ninterp==1 this is
            # a no-op identity (single column, weight 1).
            blocked_avg = pt.sum(blocked * oversample_weights, axis=1)  # (N_obs,)
            lc_model = lc_model - blocked_avg

        if self.total_detrend_cols > 0:
            detrend = pm.Data("transit_detrend", self.detrend_matrix)
            lc_model += pt.dot(detrend, self.detrend_coeffs.value)

        # Full per-observation model prediction (baseline + detrend +
        # exposure-averaged transit decrement). Kept as a plain attribute,
        # not a Deterministic: at (N_obs,) this would add N_obs * draws *
        # chains floats to every trace (tens of thousands x the size of the
        # other diagnostics here, which are all (N_planets,)). Tests compile
        # a one-off pytensor.function from this node directly instead.
        self._model_flux_node = lc_model

        # 5. Likelihood
        sigma = pt.sqrt(pt.sqr(err) + self.jitter_variance.value[self.inst_map_tensor])
        pm.Normal("transit_likelihood", mu=lc_model, sigma=sigma, observed=flux)

    def compile_plotters(self, model, system):
        """Compiles the fast PyTensor functions for generating plotting lightcurves."""
        t_input = pt.vector("lc_t_input")
        inst_idx = pt.iscalar("lc_inst_idx")

        param_symbols = [p.value for p in system.plot_params]
        planets = getattr(system, 'planet', None)
        orbits = getattr(system, 'orbit', None)

        if planets is not None and orbits is not None:
            t_grid = t_input[:, None]
            tp = orbits.tp.value[planets.orbit_map][None, :]
            n = orbits.n.value[planets.orbit_map][None, :]
            ecc = orbits.ecc.value[planets.orbit_map][None, :]
            cosw = orbits.cosw.value[planets.orbit_map][None, :]
            sinw = orbits.sinw.value[planets.orbit_map][None, :]
            inc = orbits.inc.value[planets.orbit_map][None, :]

            M = (t_grid - tp) * n
            sinf, cosf = ops.kepler(M, ecc + pt.zeros_like(M))

            a_rstar = planets.ar.value[None, :]
            p_ratio = planets.p.value[None, :]
            r_norm = a_rstar * (1.0 - pt.sqr(ecc)) / (1.0 + ecc * cosf)

            sin_wf = sinw * cosf + cosw * sinf
            cos_wf = cosw * cosf - sinw * sinf
            sin_i = pt.sin(inc)
            cos_i = pt.cos(inc)

            b = pt.sqrt(pt.sqr(r_norm * cos_wf) + pt.sqr(r_norm * sin_wf * cos_i))
            Z = r_norm * sin_wf * sin_i

            band = system.band
            band_idx = self.band_map_tensor[inst_idx]
            u1_inst = band.u1.value[band_idx]  # scalar for this instrument
            if "u2" in band.manifest:
                u2_inst = band.u2.value[band_idx]
            else:
                u2_inst = pt.zeros_like(u1_inst)
            # See build_likelihood for the Green's-basis change-of-basis derivation.
            c0_inst = 1.0 - u1_inst - 1.5 * u2_inst
            c1_inst = u1_inst + 2.0 * u2_inst
            c2_inst = -0.25 * u2_inst
            ld_norm_inst = np.pi * (c0_inst + c1_inst / 1.5)

            decrement_matrix_list = []
            for p_idx in range(planets.n_elements):
                b_p = b[:, p_idx]   # (N_times,)
                Z_p = Z[:, p_idx]
                r_p = planets.p.value[p_idx]

                sol = ops.quad_solution_vector(b_p, r_p + pt.zeros_like(b_p))
                flux_frac = (
                    sol[:, 0] * c0_inst +
                    sol[:, 1] * c1_inst +
                    sol[:, 2] * c2_inst
                ) / ld_norm_inst
                # Negative so that _compiled_full_lc output + baseline gives a transit dip
                blocked = pt.where(Z_p > 0.0, 1.0 - flux_frac, 0.0)
                # match the likelihood's SED depth dilution (built there first)
                dil_node = getattr(self, "_dilution_node", None)
                if dil_node is not None:
                    blocked = blocked * dil_node[inst_idx]
                decrement_matrix_list.append(-blocked)

            lc_matrix = pt.stack(decrement_matrix_list, axis=1)  # (N_times, N_planets)

            self._compiled_full_lc = pytensor.function(
                inputs=[t_input, inst_idx] + param_symbols,
                outputs=pt.sum(lc_matrix, axis=1),
                on_unused_input='ignore'
            )
            self._compiled_lc_matrix = pytensor.function(
                inputs=[t_input, inst_idx] + param_symbols,
                outputs=lc_matrix,
                on_unused_input='ignore'
            )

    def plot(self, system, points, filename_prefix="debug"):
        self.plot_unphased(system, points, filename_prefix=filename_prefix)
        self.plot_phased(system, points, filename_prefix=filename_prefix)

    def plot_unphased(self, system, points, filename_prefix="debug"):
        if isinstance(points, dict): points = [points]
        if len(points) == 0: return

        for i in range(self.n_elements):
            plt.figure(figsize=(12, 5))
            mask = (self.inst_map == i)
            t_data, f_data, e_data = self.time[mask], self.flux[mask], self.err[mask]

            t_pretty = np.linspace(t_data.min(), t_data.max(), 2000).astype(np.float64)

            # Plot spaghetti models
            for point in points:
                param_values = [
                    float(np.squeeze(np.asarray(point.get(p.label, p.initval)))) if getattr(p.value, "ndim", 0) == 0
                    else np.atleast_1d(point.get(p.label, p.initval)) for p in system.plot_params]

                try:
                    # _compiled_full_lc returns the flux decrement (<= 0)
                    y_decrement = self._compiled_full_lc(t_pretty, i, *param_values)

                    # Get the baseline for this instrument
                    base_vals = np.atleast_1d(point.get(self.baseline.label, 1.0))
                    baseline = float(base_vals[i])

                    alpha = 0.8 if len(points) == 1 else 0.1
                    plt.plot(t_pretty, baseline + y_decrement, 'r-', lw=1.5, alpha=alpha, zorder=2)
                except Exception as e:
                    logger.warning(f"LC model eval failed: {e}")
                    continue

            # Plot raw data
            plt.errorbar(t_data, f_data, yerr=e_data, fmt='k.', alpha=0.5, zorder=1, label=self.names[i])

            plt.xlabel("Time [BJD]")
            plt.ylabel("Relative Flux")
            plt.title(f"Transit Photometry: {self.names[i]}")
            plt.tight_layout()
            plt.savefig(f"{filename_prefix}_LC_unphased_{self.names[i]}.pdf")
            plt.close()

    def plot_phased(self, system, points, filename_prefix="debug"):
        planets = system.planet
        if isinstance(points, dict): points = [points]

        for p_idx in range(planets.n_elements):

            for i in range(self.n_elements):
                plt.figure(figsize=(10, 6))

                # 1. Spaghetti models
                for point in points:
                    ref_point = point
                    P_ref = float(np.atleast_1d(ref_point.get(system.orbit.period.label))[p_idx])
                    tc_ref = float(np.atleast_1d(ref_point.get(system.orbit.tc.label))[p_idx])

                    # One-period model grid; phase in [-0.5, 0.5] with transit centred at 0
                    t_model = np.linspace(tc_ref - 0.5 * P_ref, tc_ref + 0.5 * P_ref, 1000).astype(np.float64)
                    phase_model = ((t_model - tc_ref) / P_ref + 0.5) % 1.0 - 0.5
                    time_from_center_model = phase_model * P_ref
                    sort_m = np.argsort(phase_model)

                    param_values = [
                        float(np.squeeze(np.asarray(point.get(p.label, p.initval)))) if getattr(p.value, "ndim", 0) == 0
                        else np.atleast_1d(point.get(p.label, p.initval)) for p in system.plot_params
                    ]
                    try:
                        lc_matrix = self._compiled_lc_matrix(t_model, i, *param_values)
                        y_planet = lc_matrix[:, p_idx]
                        alpha = 0.8 if len(points) == 1 else 0.1
                        plt.plot(time_from_center_model[sort_m], y_planet[sort_m], 'r-', alpha=alpha, lw=1, zorder=2)
                    except Exception as e:
                        logger.warning(f"LC phased model eval failed: {e}")
                        continue

                # 2. Clean and phase the data
                mask = (self.inst_map == i)
                clean_params = [
                    float(np.squeeze(np.asarray(ref_point.get(p.label, p.initval)))) if getattr(p.value, "ndim", 0) == 0
                    else np.atleast_1d(ref_point.get(p.label, p.initval)) for p in system.plot_params
                ]
                try:
                    data_lc_matrix = self._compiled_lc_matrix(self.time[mask], i, *clean_params)
                except Exception as e:
                    logger.warning(f"LC phased data eval failed: {e}")
                    plt.close()
                    continue

                base_vals = np.atleast_1d(ref_point.get(self.baseline.label, 1.0))
                baseline = float(base_vals[i])

                # Subtract baseline and other planets' contributions to isolate p_idx
                other_mask = np.ones(planets.n_elements, dtype=bool)
                other_mask[p_idx] = False
                other_decrements = np.sum(data_lc_matrix[:, other_mask], axis=1)

                cleaned_flux = self.flux[mask] - baseline - other_decrements
                data_phases = ((self.time[mask] - tc_ref) / P_ref + 0.5) % 1.0 - 0.5
                time_from_center_data = data_phases * P_ref

                plt.errorbar(time_from_center_data, cleaned_flux, yerr=self.err[mask],
                             fmt='k.', alpha=0.5, zorder=1, label=self.names[i])

                t14_raw = ref_point.get(f"{self.prefix}.t14")
                if t14_raw is not None:
                    t14_ref = float(np.atleast_1d(t14_raw)[p_idx])
                    plt.xlim(-t14_ref, t14_ref)

                plt.axhline(0, color='gray', linestyle=':', alpha=0.5)
                plt.xlabel(f"Time from Mid-Transit [d] (P = {P_ref:.5f} d)")
                plt.ylabel("Flux − Baseline")
                plt.title(f"Phased LC: {planets.names[p_idx]} — {self.names[i]}")
                plt.legend(loc='best', fontsize='small')
                plt.tight_layout()
                plt.savefig(f"{filename_prefix}_LC_phased_{self.names[i]}_{planets.names[p_idx]}.pdf")
                plt.close()