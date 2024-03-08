"""
Unit tests for parameter conversions
"""
import unittest
from numpy import testing
import MulensModel as mm

from exozippy.ulens import Star, Phys2UlensConverter


class TestPhys2ulens(unittest.TestCase):

    def setUp(self):
        self.coords = "18:00:00 -30:00:00"
        self.ulens_expected = {'t_0': 2461000., 'u_0': 0.1, 't_E': 1., 'rho': 0.1}

    def test_phys2ulens_mu(self):
        """
        test conversion of physical parameters to microlensing parameters.
        Also serves as a use case.
        """
        raise NotImplementedError('Need to update with real values')

        # Possible real events for examples:
        # GRAVITY
        # Aparna: 2 epochs of HST

        lens = Star(mass=0.5, distance=6.0, mu=[0, 1])
        source = Star(mass=1.0, distance=8.0, radius=1.0, mu=[1, 0])

        converter = Phys2UlensConverter(
            source=source, lens=lens, coords=self.coords,
            t_ref=self.ulens_expected['t_0'])
        ulens_params = converter.get_ulens_params()
        testing.assert_almost_equal(
            ulens_params['t_E'], self.ulens_expected['t_E'])
        testing.assert_almost_equal(
            ulens_params['rho'], self.ulens_expected['rho'])

    def test_phys2ulens_vel(self):
        """
        Same as test_phys2ulens_mu() but using stellar velocities instead of proper
        motions.
        """
        raise NotImplementedError('Still need to add real values')

        lens = Star(mass=0.5, distance=6.0, vel=[0, 200])
        source = Star(mass=1.0, distance=8.0, radius=1.0, vel=[200, 0])

        converter = Phys2UlensConverter(
            source=source, lens=lens, coords=self.coords,
            t_ref=self.ulens_expected['t_0'])
        ulens_params = converter.get_ulens_params()
        testing.assert_almost_equal(
            ulens_params['t_E'], self.ulens_expected['t_E'])
        testing.assert_almost_equal(
            ulens_params['rho'], self.ulens_expected['rho'])


class TestGRAVITYEvent(unittest.TestCase):
    """
    Test based on properties of Kojima-1:
    https://ui.adsabs.harvard.edu/abs/2020ApJ...897..180Z/abstract

    So far, this only tests the total lens system properties, not the
    planet properties, but those could be implemented
    """

    def setUp(self):
        self.coords = "05:07:42.72 24:47:56.4" # RA, Dec

        # From Table 4
        ## Inputs
        self.M_L = 0.495
        self.D_L = 0.429
        self.D_S = 692
        self.mu_L = [-28.89, 13.39] # N, E, helio
        self.mu_S = [-6.42, -0.80]
        ## Outputs
        self.mu_rel_hel = [-22.45, 14.18]
        self.mu_rel_geo = [-22.73, 9.83]

        # Table 1, w/Lens light
        ## Inputs
        self.t_0 = 2458058.76
        ## Outputs
        self.t_E = 27.89
        self.pi_E_vec = [-0.430, 0.186] # N, E
        self.pi_E = 0.469

        # From Section 4.2
        ## Outputs
        self.theta_E = 1.891
        self.v_earth_perp = [1.47, 23.29] # km/s

        self.lens = Star(mass=self.M_L, distance=self.D_L, mu=self.mu_L)
        self.source = Star(distance=self.D_S, mu=self.mu_S)
        self.converter = Phys2UlensConverter(
            source=self.source, lens=self.lens, coords=self.coords, t_ref=self.t_0)

    def test_basic_calculations(self):
        testing.assert_almost_equal(
            self.converter.theta_E, self.theta_E, decimal=3)
        testing.assert_almost_equal(self.converter.t_E, self.t_E, decimal=2)
        testing.assert_almost_equal(self.converter.pi_E, self.pi_E, decimal=3)
        testing.assert_almost_equal(
            self.converter.pi_rel, self.theta_E * self.pi_E, decimal=3)

    def test_mu_calculations(self):
        testing.assert_almost_equal(
            self.converter.mu_rel_vec, self.mu_rel_geo, decimal=2)
        testing.assert_almost_equal(
            self.converter.mu_rel_hel, self.mu_rel_hel, decimal=2)
        testing.assert_almost_equal(
            self.converter.v_earth_perp, self.v_earth_perp, decimal=2)

    def test_get_ulens_params(self):
        ulens_params = self.converter.get_ulens_params()
        testing.assert_almost_equal(ulens_params['t_E'], self.t_E, decimal=2)
        assert ulens_params['rho'] is None
        testing.assert_almost_equal(
            ulens_params['pi_E_N'], self.pi_E_vec[0], decimal=3)
        testing.assert_almost_equal(
            ulens_params['pi_E_E'], self.pi_E_vec[1], decimal=3)


def test_v_earth_perp():
    """
    Adapted from MulensModel.test_Coords.py.

    checks function that calculates Earth projected velocity (N, E)
    """
    # Yee et al. 2015, ob140939:
    coords = "17:47:12.25 -21:22:58.7"
    t_ref = 2456836.06
    conv = Phys2UlensConverter(coords=coords, t_ref=t_ref)
    v_perp = conv.v_earth_perp
    testing.assert_almost_equal([-0.5, 28.9], v_perp, decimal=1)

    # Batista et al. 2011, mb09387:
    coords = "17:53:50.79 -33:59:25"
    t_ref = 2455042.34
    conv = Phys2UlensConverter(coords=coords, t_ref=t_ref)
    v_perp = conv.v_earth_perp
    testing.assert_almost_equal([-3.60, 22.95], v_perp, decimal=2)
