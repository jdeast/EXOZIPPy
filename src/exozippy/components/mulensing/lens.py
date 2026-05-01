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
            "pi_E_E": "default"
        }

        self.build_pars_from_dict(parameters, shape=(self.n_elements,), context_nodes=context_nodes)

    def build_likelihood(self, model, system):
        d_l_raw = system.star.distance.value[self.lens_map]
        d_s_raw = system.star.distance.value[self.source_map]
        # this penalty ensures the source is behind the lens
        pos_penalty = -1e4 * (pt.sigmoid(-d_l_raw / 10.0) + pt.sigmoid(-d_s_raw / 10.0))
        order_diff = d_l_raw - d_s_raw
        order_penalty = -1e4 * pt.sigmoid(order_diff / 10.0)
        pm.Potential(f"{self.prefix}.source_behind_lens", pt.sum(pos_penalty + order_penalty))

    def compile_plotters(self, model, system):
        pass
    def plot(self, system, points, filename_prefix="debug"):
        pass

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
        t0, u0, tE = self.t_0.value[index], self.u_0.value[index], self.t_E.value[index]
        pi_N, pi_E = self.pi_E_N.value[index], self.pi_E_E.value[index]

        tau_p = (times - t0) / tE + delta_n * pi_N + delta_e * pi_E
        u_p = u0 + delta_n * pi_E - delta_e * pi_N

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
        t0 = self.t_0.value[index]
        u0 = self.u_0.value[index]
        tE = self.t_E.value[index]
        pi_N = self.pi_E_N.value[index]
        pi_E = self.pi_E_E.value[index]

        # 2. Check for rho (finite source star radius)
        # If it doesn't exist in the component, we default to 0.0 (point source)
        rho = getattr(self, 'rho', None)
        rho_val = rho.value[index] if rho is not None else pt.constant(0.0)

        # 3. Stack into the input vector for the Op
        param_vector = pt.stack([t0, u0, tE, pi_N, pi_E, rho_val])

        # 4. Instantiate and call the Op
        # 'times' must be a numpy array of BJD_TDB times
        mag_op = MulensMagOp(times, coords, obs_pos)

        return mag_op(param_vector)
