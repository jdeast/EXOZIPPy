import pytensor.tensor as pt
from exozippy.components.component import Component
# this import is required even though it's not used explicitly
# it registers all the mathematical relations
from . import physics

class Lens(Component):
    def __init__(self, config, config_manager):
        super().__init__(config, config_manager)
        self.label = "Lens Parameters"

    def build_parameters(self):
        prefix = "lens"
        # Standard Point Lens parameters (PSPL)
        parameters = {
            "t0": None,  # Time of peak
            "u0": None,  # Impact parameter
            "tE": None,  # Einstein crossing time
            "pi_E_N": None,  # Parallax North
            "pi_E_E": None  # Parallax East
        }
        self.build_pars_from_dict(parameters, shape=(self.n_elements,), prefix=prefix)

    def get_magnification(self, time, delta_n, delta_e, index=0):
        """
        Symbolic Paczynski magnification including parallax.
        """
        # Pull the PyTensor variables for this lens
        # (Assuming single lens for now, but self.t0.value is a vector)
        t0 = self.t0.value[index]
        u0 = self.u0.value[index]
        tE = self.tE.value[index]
        pi_N = self.pi_E_N.value[index]
        pi_E = self.pi_E_E.value[index]

        # Trajectory in units of Einstein radii
        tau = (time - t0) / tE

        # Apply Parallax Shifts
        # Note: Math follows standard Gould (2000) or Skowron (2011) convention
        tau_p = tau + delta_n * pi_N + delta_e * pi_E
        u_p = u0 + delta_n * pi_E - delta_e * pi_N

        u2 = pt.sqr(tau_p) + pt.sqr(u_p)

        # Paczynski Magnification A(u)
        A = (u2 + 2.0) / pt.sqrt(u2 * (u2 + 4.0))
        return A