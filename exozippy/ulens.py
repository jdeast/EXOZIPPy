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
        self._theta_E = None
        self._pi_rel = None
        self._mu_rel = None
        self._mu_rel_hel = None
        self._v_earth_perp = None

        self._t_E = None
        self._rho = None
        self.ulens_params = None

    def get_ulens_params(self):
        """
        :return: *dict* of relevant microlensing model parameters.
        """
        self.ulens_params = {'t_E': self.t_E, 'rho': self.rho}

        return self.ulens_params

    @property
    def coords(self):
        """
        see :py:class:`~MulensModel.coordinates.Coordinates`
        """
        return self._coords

    @coords.setter
    def coords(self, new_value):
        self._coords = mm.Coordinates(new_value)

    @property
    def pi_rel(self):
        """
        *float*

        _Magnitude_ of the lens-source relative parallax in _mas_.
        """
        if self._pi_rel is None:
            self._pi_rel = self.lens.pi - self.source.pi

        return self.pi_rel

    @property
    def theta_E(self):
        """
        *float*

        Angular Einstein radius in _mas_.
        """
        if self._theta_E is None:
            self._theta_E = np.sqrt(kappa * self.lens.mass * self.pi_rel)

        return self._theta_E

    @property
    def t_E(self):
        """
        *float*

        Einstein timescale in _days_.
        """
        if self._t_E is None:
            self._t_E = self.theta_E / self.mu_rel.value

        return self._t_E

    @property
    def rho(self):
        """
        *float*

        Source radius normalized to the Einstein ring radius.
        """
        if self._rho is None:
            self._rho = self.source.theta_star / self.theta_E

        return self._rho

    @property
    def mu_rel(self):
        """
        *vector*

        Lens-source relative proper motion in _mas/yr_ in the _geocentric_
        frame.
        """
        # mu_rel, hel = mu_rel + v_earth,perp * pi_rel / au
        if self._mu_rel is None:
            self._mu_rel = self.mu_rel_hel - self.v_earth_perp * self.pi_rel
            # Units are going to be a thing...

        return self._mu_rel

    @property
    def mu_rel_hel(self):
        """
        *vector*

        Lens-source relative proper motion in _mas/yr_ in the _helocentric_
        frame.
        """
        return self.lens.mu - self.source.mu