"""
Functions and classes for converting between microlensing and physical
parameters.
"""
import numpy as np
import MulensModel as mm


_kappa = 8.14  # Should this be imported from mulens model for consistency?
_masyr2kms = 4.740470463533349


class Star(object):
    """
    Class defining a star and its properties.

    Arguments:
        mass: *float*
            Mass of the star in _Solar Masses_.

        radius: *float*
            Radius of the star in _Solar Radii_.

        distance: *float*
            Distance to the star in _kpc_. Use either `distance` OR `pi`, not
            both.

        pi: *float*
            parallax to the star in _mas_. Use either `distance` OR `pi`, not
            both.

        mu: *list*, *np.array*, *astropy.Quantity*
            [N, E] heliocentric proper motion of the star in _mas/yr_.
            Use either `mu` OR `vel`, not both.

        vel: *list*, *np.array*, *astropy.Quantity*
            [N, E] heliocentric velocity of the star in _km/s_.
            Use either `mu` OR `vel`, not both.
    """

    def __init__(
            self,
            mass=None, radius=None,
            distance=None, pi=None, mu=None, vel=None):

        self.mass = mass
        self.radius = radius
        self._theta_star = None

        if (distance is not None) and (pi is not None):
            raise KeyError('Define either *distance* or *pi*, not both!')
        else:
            self._distance = distance
            self._pi = pi

        self._mu = None
        self._vel = None
        if (mu is not None) and (vel is not None):
            raise KeyError('Define either *mu* or *vel*, not both!')
        elif mu is not None:
            self.mu = mu
        elif vel is not None:
            self.vel = vel

    @property
    def distance(self):
        """
        *float*

        Distance to the star in _kpc_.
        """
        if self._distance is None:
            if self._pi is None:
                raise AttributeError(
                    'Cannot return distance. ' +
                    'Neither *distance* nor *pi* were defined.')
            else:
                self._distance = 1. / self._pi

        return self._distance

    @property
    def pi(self):
        """
        *float*

        Parallax to the star in _mas_.
        """
        if self._pi is None:
            if self._distance is None:
                raise AttributeError(
                    'Cannot return pi. ' +
                    'Neither *distance* nor *pi* were defined.')
            else:
                self._pi = 1. / self._distance

        return self._pi

    @property
    def mu(self):
        """
        *np.ndarray*

        [N, E] heliocentric proper motion of the star in _mas/yr_.
        """
        if self._mu is None:
            if self._vel is None:
                raise AttributeError(
                    'Cannot return mu. ' +
                    'Neither *mu* nor *vel* were defined.')
            else:
                self._mu = self._vel / self.distance / _masyr2kms

        return self._mu

    @mu.setter
    def mu(self, new_value):
        if isinstance(new_value, (tuple, list)):
            self._mu = np.array(new_value)
        elif isinstance(new_value, (np.ndarray)):
            self._mu = new_value
        else:
            raise ValueError(
                'Invalid type for mu. Must be *tuple*, *list*, ' +
                ' or *np.ndarray: ', type(new_value))

    @property
    def vel(self):
        """
        *np.ndarray*

        [N, E] heliocentric velocity of the star in _km/s_.
        """
        if self._vel is None:
            if self._mu is None:
                raise AttributeError(
                    'Cannot return vel. ' +
                    'Neither *mu* nor *vel* were defined.')
            else:
                self._vel = self._mu * self.distance * _masyr2kms

        return self._vel

    @vel.setter
    def vel(self, new_value):
        if isinstance(new_value, (tuple, list)):
            self._vel = np.array(new_value)
        elif isinstance(new_value, (np.ndarray)):
            self._vel = new_value
        else:
            raise TypeError(
                'Invalid type for vel. Must be *tuple*, *list*, ' +
                ' or *np.ndarray: ', type(new_value))

    @property
    def theta_star(self):
        """
        *float*

        Angular radius of the star in _mas_
        """
        if self._theta_star is None:
            # 215.032 RSun = 1 AU
            if self.radius is not None:
                self._theta_star = self.radius / 215.032 / self.distance

        return self._theta_star


class Phys2UlensConverter(object):
    """
    Class for converting from physical parameters to microlensing model
    parameters.

    Arguments :
        source: :py:class:`~Star`
            The source star.

        lens: :py:class:`~Star`
            The lens star.  # eventually, will need to upgrade to systems.

        coords: *str* or *astropy.coordinates.SkyCoords*
            The coordinates of the event.

        t_ref: *float*
            A reference time in BJD for converting from heliocentric to
            geocentric proper motions. Recommend using t_0.

    """

    def __init__(self, source=None, lens=None, coords=None, t_ref=None):
        self.source = source
        self.lens = lens

        self._coords = None
        self.coords = coords

        self.t_ref = t_ref

        self._theta_E = None
        self._pi_rel = None
        self._mu_rel = None
        self._mu_rel_vec = None
        self._mu_rel_hat = None
        self._mu_rel_hel = None
        self._v_earth_perp = None

        self._t_E = None
        self._rho = None
        self._pi_E_N = None
        self._pi_E_E = None
        self._pi_E = None
        self.ulens_params = None

    def get_ulens_params(self):
        """
        :return: *dict* of relevant microlensing model parameters.
        """
        self.ulens_params = {'t_E': self.t_E, 'rho': self.rho,
                             'pi_E_N': self.pi_E_N, 'pi_E_E': self.pi_E_E}

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

        return self._pi_rel

    @property
    def theta_E(self):
        """
        *float*

        Angular Einstein radius in _mas_.
        """
        if self._theta_E is None:
            self._theta_E = np.sqrt(_kappa * self.lens.mass * self.pi_rel)

        return self._theta_E

    @property
    def t_E(self):
        """
        *float*

        Einstein timescale in _days_.
        """
        if self._t_E is None:
            self._t_E = 365.25 * self.theta_E / self.mu_rel

        return self._t_E

    @property
    def rho(self):
        """
        *float*

        Source radius normalized to the Einstein ring radius.
        """
        if self._rho is None:
            if self.source.theta_star is not None:
                self._rho = self.source.theta_star / self.theta_E

        return self._rho

    @property
    def pi_E(self):
        """
        *float*

        The magnitude of the microlens parallax vector.
        """
        if self._pi_E is None:
            self._pi_E = self.pi_rel / self.theta_E

        return self._pi_E

    @property
    def pi_E_vec(self):
        """
        *list*

        The [N, E] components of the microlens parallax vector.
        """
        return [self.pi_E_N, self.pi_E_E]

    @property
    def pi_E_N(self):
        """
        *float*

        The North component of the microlens parallax vector.
        """
        if self._pi_E_N is None:
            self._pi_E_N = self.pi_E * self.mu_rel_hat[0]

        return self._pi_E_N

    @property
    def pi_E_E(self):
        """
        *float*

        The East component of the microlens parallax vector.
        """
        if self._pi_E_E is None:
            self._pi_E_E = self.pi_E * self.mu_rel_hat[1]

        return self._pi_E_E

    @property
    def mu_rel_vec(self):
        """
        *np.ndarray*

        Lens-source relative proper motion in _mas/yr_ in the _geocentric_
        frame.
        """
        # mu_rel, hel = mu_rel + v_earth,perp * pi_rel / au
        if self._mu_rel_vec is None:
            self._mu_rel_vec = (self.mu_rel_hel -
                                self.v_earth_perp * self.pi_rel / _masyr2kms)

        return self._mu_rel_vec

    @property
    def mu_rel(self):
        """
        *float*

        _magnitude_ of lens-source relative proper motion in _mas/yr_ in the
        _geocentric_frame.
        """
        if self._mu_rel is None:
            self._mu_rel = np.sqrt(
                self.mu_rel_vec[0] ** 2 + self.mu_rel_vec[1] ** 2)

        return self._mu_rel

    @property
    def mu_rel_hat(self):
        """
        *np.ndarray*

        _Direction_ of the lens-source relative proper motion in the
        _geocentric_ frame.
        """
        if self._mu_rel_hat is None:
            self._mu_rel_hat = self.mu_rel_vec / self.mu_rel

        return self._mu_rel_hat

    @property
    def mu_rel_hel(self):
        """
        *np.ndarray*

        Lens-source relative proper motion in _mas/yr_ in the _heliocentric_
        frame.
        """
        return self.lens.mu - self.source.mu

    @property
    def v_earth_perp(self):
        """
        *np.ndarray*

        The velocity of Earth projected on the sky at t_ref.
        """
        return np.array(self._coords.v_Earth_projected(self.t_ref))
