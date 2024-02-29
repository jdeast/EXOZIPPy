"""
Functions and classes for converting between microlensing and physical
parameters.
"""
import numpy as np
import MulensModel as mm


kappa = 8.14  # Should this be imported from mulens model for consistency?


class Phys2UlensConverter(object):

    def __init__(self, source=None, lens=None, coords=None, t_ref=None):
        self.source = source
        self.lens = lens

        self._coords = None
        self.coords = coords

        self.t_ref = t_ref

        # Probably all need to be replaced with private variables and
        # properties...
        self.theta_E = None
        self.pi_rel = None
        self.mu_rel = None

        self.t_E = None
        self.rho = None
        self.ulens_params = None

    def get_ulens_params(self):
        self.calc_t_E()
        self.calc_rho()
        self.ulens_params = {'t_E': self.t_E, 'rho': self.rho}

        return self.ulens_params

    def calc_pi_rel(self):
        self.pi_rel = self.lens.pi.value - self.source.pi.value

    def calc_theta_E(self):
        if self.pi_rel is None:
            self.calc_pi_rel()

        self.theta_E = np.sqrt(kappa * self.lens.mass * self.pi_rel)

    def calc_t_E(self):
        if self.theta_E is None:
            self.calc_theta_E()

        if self.mu_rel is None:
            self.calc_mu_rel()

        self.t_E = self.theta_E / self.mu_rel

    def calc_rho(self):
        if self.theta_E is None:
            self.calc_theta_E()

        self.rho = self.source.theta_star / self.theta_E

    def calc_mu_rel(self):
        pass

    @property
    def coords(self):
        """
        see :py:class:`~MulensModel.coordinates.Coordinates`
        """
        return self._coords

    @coords.setter
    def coords(self, new_value):
        self._coords = mm.Coordinates(new_value)
