import pytensor.tensor as pt
import pymc as pm
import numpy as np
from exozippy.components.component import Component
from .op import MulensMagOp

class Lens(Component):
    def __init__(self, config, config_manager):
        super().__init__(config, config_manager)
        self.label = "Lens Parameters"
        self.finite_source = [c.get("finite_source", False) for c in self.config]
        self.t0_par = [self._resolve_t0_par(i, c, config_manager) for i, c in enumerate(self.config)]
        self.mag_method = [c.get("mag_method", "auto_vbbl" if c.get("finite_source", False) else "point_source") for c
                           in self.config]

    @staticmethod
    def _resolve_t0_par(i, c, config_manager):
        """Return t_0_par for lens i: explicit yaml value > user params t_0 > fallback."""
        if 't0_par' in c:
            return float(c['t0_par'])
        entry = config_manager.user_params.get(f"lens.{i}.t_0")
        if isinstance(entry, dict):
            val = entry.get('initval')
        else:
            val = entry
        return float(val) if val is not None else 2450000.0

    @property
    def prefix(self):
        return "lens"

    def build_maps(self):
        """Stage 1b: Map the lens and source to their respective stars."""
        # The base Component will automatically convert these to PyTensor int32
        # variables named `self.lens_map_tensor` and `self.source_map_tensor`
        self.lens_map = np.array([c.get("lens_ndx", 0) for c in self.config])
        self.source_map = np.array([c.get("source_ndx", 1) for c in self.config])

    def register_parameters(self, system):
        """Stage 2: Declare the microlensing manifest."""
        self.manifest = {
            "t_0": None, "u_0": None,
            "pi_rel": "default", "theta_E": "default",
            "mu_ra_rel": "default", "mu_dec_rel": "default",
            "mu_rel_mag": "default", "t_E": "default",
            "pi_E_N": "default", "pi_E_E": "default",
        }

        # Inject microlensing initval and init_scale defaults for all events.
        # Initval hints (rank=30): override the generic 10 pc star default but
        # yield to data-derived (rank 60) or user-supplied values (rank 100).
        # Scale hints: replace the 0.1 pc stellar default which is ~5000x too
        # small for bulge distances; these yield to user-provided init_scale.
        for i in range(self.n_elements):
            l_idx = self.lens_map[i]
            s_idx = self.source_map[i]

            self.config_manager.add_hint(f"star.{l_idx}.distance", 4000.0, rank=30)
            self.config_manager.add_hint(f"star.{s_idx}.distance", 8000.0, rank=30)

            # Distance scale: lens uncertain over disk range, source over bulge depth
            self.config_manager.add_scale_hint(f"star.{l_idx}.distance", 5.0)
            self.config_manager.add_scale_hint(f"star.{s_idx}.distance", 5.0)

            # Proper motion scale: galactic velocity dispersion ~3–5 mas/yr
            # (default 0.05 is ~100x too small for bulge stars)
            self.config_manager.add_scale_hint(f"star.{l_idx}.pm_ra",  3.0)
            self.config_manager.add_scale_hint(f"star.{l_idx}.pm_dec", 3.0)
            self.config_manager.add_scale_hint(f"star.{s_idx}.pm_ra",  3.0)
            self.config_manager.add_scale_hint(f"star.{s_idx}.pm_dec", 3.0)

            # Logmass scale: Chabrier IMF sigma ~0.5 dex; data typically constrains
            # to 0.05–0.2 dex — 0.3 dex is a reasonable middle-ground default
            self.config_manager.add_scale_hint(f"star.{l_idx}.logmass", 0.001)
            self.config_manager.add_scale_hint(f"star.{s_idx}.logmass", 0.3)

        # Tighten lens logmass scale when parallax is constrained by multi-observer data.
        if hasattr(system, 'mulensinstrument') and hasattr(system.mulensinstrument, 'inst_ref_pos'):
            ref_pos = system.mulensinstrument.inst_ref_pos  # (n_inst, 3) absolute AU
            max_sep = 0.0
            for ii in range(len(ref_pos)):
                for jj in range(ii + 1, len(ref_pos)):
                    max_sep = max(max_sep, float(np.linalg.norm(ref_pos[ii] - ref_pos[jj])))
            # > 0.5 AU: satellite parallax → mass constrained to ~0.05 dex
            # > 1e-5 AU (~1500 km): terrestrial parallax in high-mag events → ~0.15 dex
            if max_sep > 0.5:
                lens_logmass_scale = 0.0005
            elif max_sep > 1e-5:
                lens_logmass_scale = 0.00075
            else:
                lens_logmass_scale = None
            if lens_logmass_scale is not None:
                for i in range(self.n_elements):
                    self.config_manager.add_scale_hint(
                        f"star.{self.lens_map[i]}.logmass", lens_logmass_scale)

        if any(self.finite_source):
            self.manifest["rho"] = "default"

    def build_likelihood(self, model, system):

        mu_rel = self.mu_rel_mag.value
        theta_E = self.theta_E.value

        # account for observation bias
        # bigger theta_E, higher mu_rel are more likely to be detected
        pm.Potential(f"{self.prefix}.event_rate_prior", pt.sum(pt.log(mu_rel) + pt.log(theta_E)))

        # this penalty ensures the source is behind the lens
        d_l = system.star.distance.value[self.lens_map]
        d_s = system.star.distance.value[self.source_map]
        pi_rel_penalty = -1e6 * pt.sigmoid(-(d_s - d_l - 10.0) * 1.0)
        pm.Potential(f"{self.prefix}.source_behind_lens", pt.sum(pi_rel_penalty))

        # this avoids the theta_E singularity
        mu_penalty = -1e6 * pt.sigmoid(-(mu_rel - 1e-6) * 1e7)
        pm.Potential(f"{self.prefix}.mu_rel_singularity", pt.sum(mu_penalty))

        # this avoids the pi_E singularity
        theta_E = self.theta_E.value
        theta_penalty = -1e6 * pt.sigmoid(-(theta_E - 1e-6) * 1e7)
        pm.Potential(f"{self.prefix}.theta_E_singularity", pt.sum(theta_penalty))

    def compile_plotters(self, model, system):
        pass
    def plot(self, system, points, filename_prefix="debug"):
        pass

    def _get_safe_mm_params(self, index=0):
        # 1. Capture the raw values
        tE_raw = self.t_E.value[index]
        u0_raw = self.u_0.value[index]
        theta_E_raw = self.theta_E.value[index]
        pi_N_raw = self.pi_E_N.value[index]
        pi_E_raw = self.pi_E_E.value[index]

        # 2. Scrub NaNs immediately
        # Replace NaNs with 0.0 or a safe neutral value.
        # This prevents NaN poisoning even in the 'inactive' branch of a switch.
        tE_scrubbed = pt.nan_to_num(tE_raw, nan=100.0)
        u0_scrubbed = pt.nan_to_num(u0_raw, nan=1.0)
        theta_E_scrubbed = pt.nan_to_num(theta_E_raw, nan=0.0)
        pi_N_scrubbed = pt.nan_to_num(pi_N_raw, nan=0.0)
        pi_E_scrubbed = pt.nan_to_num(pi_E_raw, nan=0.0)

        # 3. Apply Clamps to the scrubbed values
        tE_safe = pt.maximum(tE_scrubbed, 1e-4)
        u0_safe = pt.sign(u0_scrubbed) * pt.maximum(pt.abs(u0_scrubbed), 1e-6)

        # We use a threshold check to decide if we trust the parallax
        is_physical = pt.gt(theta_E_scrubbed, 1e-6)

        # 4. Final Logic
        return {
            't0': self.t_0.value[index],
            'u0': u0_safe,
            'tE': tE_safe,
            'pi_N': pt.switch(is_physical, pi_N_scrubbed, 0.0),
            'pi_E': pt.switch(is_physical, pi_E_scrubbed, 0.0)
        }

    def get_magnification_old(self, time, delta_n, delta_e, index=0):
        """Symbolic Paczynski magnification including parallax."""
        # Because we derived everything physically, we now have exactly the nodes
        # we need for the standard Paczynski math!
        t0 = self.t_0.value[index]
        u0 = self.u_0.value[index]
        tE = self.t_E.value[index]
        pi_N = self.pi_E_N.value[index]
        pi_E = self.pi_E_E.value[index]

        tau = (time - t0) / tE
        tau_p = tau + delta_n * pi_N - delta_e * pi_E
        u_p = u0 - delta_n * pi_E - delta_e * pi_N

        u2 = pt.sqr(tau_p) + pt.sqr(u_p)
        A = (u2 + 2.0) / pt.sqrt(u2 * (u2 + 4.0))

        return A

    def get_magnification(self, times, obs_pos, system, index=0):
        # 1. Get RA/Dec nodes (symbolic)
        source_ndx = self.source_map[index]
        ra = system.star.ra.value[source_ndx]
        dec = system.star.dec.value[source_ndx]

        # 2. Rotation Matrix to project Earth XYZ onto North/East/Parallel
        # Earth pos is (N, 3), we want to project onto the sky plane
        # X is towards RA=0, Y is towards RA=90, Z is North Pole

        # East Vector: (-sin(ra), cos(ra), 0)
        # North Vector: (-cos(ra)sin(dec), -sin(ra)sin(dec), cos(dec))

        x, y, z = obs_pos[:, 0], obs_pos[:, 1], obs_pos[:, 2]

        delta_e = -x * pt.sin(ra) + y * pt.cos(ra)
        delta_n = -x * pt.cos(ra) * pt.sin(dec) - y * pt.sin(ra) * pt.sin(dec) + z * pt.cos(dec)

        # 3. Paczynski math
        s = self._get_safe_mm_params(index)
        # Sky convention: E is to the left (looking up), so the East term picks up
        # a sign flip relative to the standard right-handed dot product.
        tau_p = (times - s['t0']) / s['tE'] + delta_n * s['pi_N'] - delta_e * s['pi_E']
        u_p = s['u0'] - delta_n * s['pi_E'] - delta_e * s['pi_N']

        u2 = pt.sqr(tau_p) + pt.sqr(u_p)
        return (u2 + 2.0) / pt.sqrt(u2 * (u2 + 4.0))

    def get_magnification_op(self, times, obs_pos, system, index=0):
        """
        Uses MulensModel via a PyTensor Op to calculate magnification.
        This handles complex orbital parallax more accurately than simple delta_n/e.
        """

        # Get RA/Dec from the source star (usually the coordinate anchor)
        source_ndx = int(self.source_map[index])
        ra_deg = float(system.star.ra.value[source_ndx].eval()) * (180.0 / np.pi)
        dec_deg = float(system.star.dec.value[source_ndx].eval()) * (180.0 / np.pi)

        # MulensModel can take a string: "RA Dec" in degrees
        coords = f"{ra_deg}d {dec_deg}d"

        # 1. Gather the derived parameters from the Switchboard/Parameters
        s = self._get_safe_mm_params(index)
        param_list = [s['t0'], s['u0'], s['tE'], s['pi_N'], s['pi_E']]

        # 2. Check for rho (finite source star radius)
        if self.finite_source[index]:
            param_list.append(self.rho.value[index])

        param_vector = pt.stack(param_list)

        # 1. Convert the inputs to PyTensor variables so they can be tracked in the graph
        times_tensor = pt.as_tensor_variable(times)
        obs_tensor = pt.as_tensor_variable(obs_pos)

        # 2. Initialize the Op with ONLY the static Python dictionary
        mag_op = MulensMagOp(coords=coords, mag_method=self.mag_method[index])

        # 3. Call the Op with all three dynamic tensors!
        return mag_op(param_vector, times_tensor, obs_tensor)
