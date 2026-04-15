import pytensor.tensor as pt
import numpy as np
from exozippy.components.component import Component
from . import physics


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
        pass
    def compile_plotters(self, model, system):
        pass
    def plot(self, system, points, filename_prefix="debug"):
        pass

    def get_magnification(self, time, delta_n, delta_e, index=0):
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

    # ... keep the rest (load_data, build_likelihood, plot) as pass for now