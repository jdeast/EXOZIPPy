import pytensor.tensor as pt
import pymc as pm
import numpy as np
from exozippy.components.component import Component
from . import physics
from .op import MulensMagOp



class Lens(Component):
    def __init__(self, config, config_manager):
        super().__init__(config, config_manager)
        self.label = "Lens Parameters"

        self.finite_source = [c.get("finite_source", False) for c in self.config]
        self.t0_par = [c.get("t0_par", 2450000.0) for c in self.config]

        self.mag_method = []
        for c in self.config:
            default_mag = "auto_vbbl" if c.get("finite_source", False) else "point_source"
            self.mag_method.append(c.get("mag_method", default_mag))

    @property
    def prefix(self):
        return "lens"

    def build_parameters(self, model):
        # We only build the base integration constants here
        parameters = {
            "t_0": None,
            "u_0": None,
        }
        self.build_pars_from_dict(parameters, shape=(self.n_elements,))

    def load_data(self):
        pass

    def build_map(self, system):
        # Map which star is the lens and which is the source
        # Defaults: lens is star 0, source is star 1
        lens_indices = np.array([c.get("lens_ndx", 0) for c in self.config])
        self.lens_map = pt.as_tensor_variable(lens_indices).astype("int32")

        source_indices = np.array([c.get("source_ndx", 1) for c in self.config])
        self.source_map = pt.as_tensor_variable(source_indices).astype("int32")

    def build_dependent_parameters(self, model, system):
        stars = system.star

        # 1. Provide the physical inputs from the mapped stars
        context_nodes = {
            "dist_lens": stars.distance.value[self.lens_map],
            "dist_source": stars.distance.value[self.source_map],
            "mass_lens": stars.mass.value[self.lens_map],
            "pm_ra_lens": stars.pm_ra.value[self.lens_map],
            "pm_dec_lens": stars.pm_dec.value[self.lens_map],
            "pm_ra_source": stars.pm_ra.value[self.source_map],
            "pm_dec_source": stars.pm_dec.value[self.source_map],
            "distance_lens": system.star.distance.value[self.lens_map],
            "distance_source": system.star.distance.value[self.source_map]
        }

        # 2. Command the switchboard to build the derived microlensing observables
        parameters = {
            "pi_rel": "default",
            "theta_E": "default",
            "mu_ra_rel": "default",
            "mu_dec_rel": "default",
            "mu_rel_mag": "default",
            "t_E": "default",
            "pi_E_N": "default",
            "pi_E_E": "default",
        }

        if any(self.finite_source):
            context_nodes["radius"] = stars.radius.value[self.source_map]
            context_nodes["distance"] =stars.distance.value[self.source_map]
            parameters["rho"] = "default"

        self.build_pars_from_dict(parameters, shape=(self.n_elements,), context_nodes=context_nodes)

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
        tau_p = tau + delta_n * pi_N + delta_e * pi_E
        u_p = u0 + delta_n * pi_E - delta_e * pi_N

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
        tau_p = (times - s['t0']) / s['tE'] + delta_n * s['pi_N'] + delta_e * s['pi_E']
        u_p = s['u0'] + delta_n * s['pi_E'] - delta_e * s['pi_N']

        u2 = pt.sqr(tau_p) + pt.sqr(u_p)
        return (u2 + 2.0) / pt.sqrt(u2 * (u2 + 4.0))

    def get_magnification_op(self, times, obs_pos, system, index=0):
        """
        Uses MulensModel via a PyTensor Op to calculate magnification.
        This handles complex orbital parallax more accurately than simple delta_n/e.
        """

        # Get RA/Dec from the source star (usually the coordinate anchor)
        source_ndx = int(self.source_map[index].eval())
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

        # 4. Instantiate and call the Op
        # 'times' must be a numpy array of BJD_TDB times
        mag_op = MulensMagOp(times, coords, obs_pos)

        return mag_op(param_vector)
