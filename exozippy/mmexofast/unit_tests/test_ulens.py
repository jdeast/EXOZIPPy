"""
Unit tests for parameter conversions
"""
import unittest
import numpy as np
from numpy import testing
import astropy.units as u
from astropy.coordinates import SkyCoord

from exozippy.mmexofast.ulens import Star, Phys2UlensConverter


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
        self.D_S = 0.692
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
        self.pi_E_vec = [-0.430, 0.186]  # N, E
        self.pi_E = 0.469

        # From Section 4.2
        ## Outputs
        self.theta_E = 1.891
        self.v_earth_perp = [1.47, 23.29]  # km/s

        # From Section 4.3
        ## Outputs
        self.pi_L = 2.332

        # From RP - r_source and rho are based also on Fukui et al. 2019
        # https://ui.adsabs.harvard.edu/abs/2019AJ....158..206F/abstract
        ## Inputs
        self.r_source = 1.284
        ## Outputs
        self.v_L_hel = [-58.75, 27.23] # N, E
        self.v_S_hel = [-21.06, -2.624]
        self.rho = 4.567e-3

        self.lens = Star(mass=self.M_L, distance=self.D_L, mu=self.mu_L)
        self.source = Star(
            distance=self.D_S, radius=self.r_source, mu=self.mu_S)
        self.converter = Phys2UlensConverter(
            source=self.source, lens=self.lens, coords=self.coords, t_ref=self.t_0)

    def test_Star_calculations(self):
        testing.assert_allclose(self.lens.pi, self.pi_L, rtol=0.001)

    def test_basic_calculations(self):
        print(self.lens.pi, self.source.pi)
        testing.assert_allclose(
            self.converter.pi_rel, self.theta_E * self.pi_E, atol=0.001)
        testing.assert_allclose(
            self.converter.theta_E, self.theta_E, rtol=0.001)
        testing.assert_allclose(self.converter.pi_E, self.pi_E, rtol=0.001)
        testing.assert_allclose(self.converter.t_E, self.t_E, rtol=0.02)

    def test_mu_calculations(self):
        testing.assert_allclose(
            self.converter.mu_rel_hel, self.mu_rel_hel, rtol=0.001)
        testing.assert_allclose(
            self.converter.v_earth_perp, self.v_earth_perp, rtol=0.02)
        testing.assert_allclose(
            self.converter.mu_rel_vec, self.mu_rel_geo, rtol=0.02)
        mu_rel = np.sqrt(self.mu_rel_geo[0]**2 + self.mu_rel_geo[1]**2)
        testing.assert_allclose(
            self.converter.mu_rel, mu_rel, rtol=0.02)
        mu_rel_hat = self.mu_rel_geo / np.linalg.norm(self.mu_rel_geo)
        testing.assert_allclose(
            self.converter.mu_rel_hat, mu_rel_hat, rtol=0.02)

    def test_vel_calculations(self):
        testing.assert_allclose(
            self.lens.vel, self.v_L_hel, rtol=0.001)
        testing.assert_allclose(
            self.source.vel, self.v_S_hel, rtol=0.001)

    def test_get_ulens_params(self):
        ulens_params = self.converter.get_ulens_params()
        testing.assert_allclose(ulens_params['t_E'], self.t_E, rtol=0.02)
        testing.assert_allclose(ulens_params['rho'], self.rho, rtol=0.001)
        testing.assert_allclose(
            ulens_params['pi_E_N'], self.pi_E_vec[0], atol=0.002)
        testing.assert_allclose(
            ulens_params['pi_E_E'], self.pi_E_vec[1], atol=0.002)

    def test_ulens_properties(self):
        testing.assert_allclose(self.converter.rho, self.rho, rtol=0.001)
        testing.assert_allclose(
            self.converter.pi_E_vec[0], self.pi_E_vec[0], atol=0.002)
        testing.assert_allclose(
            self.converter.pi_E_vec[1], self.pi_E_vec[1], atol=0.002)
        testing.assert_allclose(
            self.converter.pi_E_N, self.pi_E_vec[0], atol=0.002)
        testing.assert_allclose(
            self.converter.pi_E_E, self.pi_E_vec[1], atol=0.002)

    def test_coords(self):
        astropy_coords = SkyCoord(self.coords, unit=[u.hourangle, u.deg])
        assert self.converter.coords.ra.deg == astropy_coords.ra.deg
        assert self.converter.coords.dec.deg == astropy_coords.dec.deg

    def test_coords_setter(self):
        test_conv = Phys2UlensConverter(
            source=self.source, lens=self.lens, coords=self.coords, t_ref=self.t_0)
        test_conv.coords = '18:00:00 -30:00:00'
        testing.assert_almost_equal(test_conv.coords.ra.deg, 270.)
        testing.assert_almost_equal(test_conv.coords.dec.deg, -30.)


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


class TestStar(unittest.TestCase):
    """
        Test based on properties of Kojima-1:
        https://ui.adsabs.harvard.edu/abs/2020ApJ...897..180Z/abstract
    """

    def setUp(self):
        self.coords = "05:07:42.72 24:47:56.4" # RA, Dec
        # From Table 4
        ## Inputs
        self.M_L = 0.495
        self.D_L = 0.429
        self.D_S = 0.692
        self.mu_L = [-28.89, 13.39]  # N, E, helio
        self.mu_S = [-6.42, -0.80]
        ## Outputs
        self.mu_rel_hel = [-22.45, 14.18]
        self.mu_rel_geo = [-22.73, 9.83]

        # Table 1, w/Lens light
        ## Inputs
        self.t_0 = 2458058.76
        ## Outputs
        self.t_E = 27.89
        self.pi_E_vec = [-0.430, 0.186]  # N, E
        self.pi_E = 0.469

        # From Section 4.2
        ## Outputs
        self.theta_E = 1.891
        self.v_earth_perp = [1.47, 23.29]  # km/s

        # From Section 4.3
        ## Outputs
        self.pi_L = 2.332

        # From RP - r_source and rho are based also on Fukui et al. 2019
        # https://ui.adsabs.harvard.edu/abs/2019AJ....158..206F/abstract
        ## Inputs
        self.r_source = 1.284
        ## Outputs
        self.v_L_hel = [-58.75, 27.23]  # N, E
        self.v_S_hel = [-21.06, -2.624]
        self.rho = 4.567e-3

        self.lens = Star(mass=self.M_L, distance=self.D_L, mu=self.mu_L)
        self.alt_lens = Star(mass=self.M_L, pi=self.pi_L, vel=self.v_L_hel)
        self.test_star = Star(mass=self.M_L, distance=self.D_L, mu=self.mu_L)
        self.source = Star(
            distance=self.D_S, radius=self.r_source, mu=self.mu_S)

    def test_distance(self):
        assert self.lens.distance == self.D_L
        testing.assert_allclose(self.alt_lens.distance, self.D_L, rtol=0.001)

    def test_pi(self):
        testing.assert_allclose(self.lens.pi, self.pi_L, rtol=0.001)
        assert self.alt_lens.pi == self.pi_L

    def test_mu(self):
        assert self.lens.mu[0] == self.mu_L[0]
        assert self.lens.mu[1] == self.mu_L[1]
        testing.assert_allclose(self.alt_lens.mu, self.mu_L, rtol=0.001)

    def test_mu_setter(self):
        self.test_star.mu = [-50, 50]
        assert self.test_star.mu[0] == -50
        assert self.test_star.mu[1] == 50
        self.test_star.mu = np.array([-5, 5])
        assert self.test_star.mu[0] == -5
        assert self.test_star.mu[1] == 5

    def test_vel(self):
        testing.assert_allclose(self.lens.vel, self.v_L_hel, rtol=0.001)

    def test_vel_setter(self):
        self.test_star.vel = [-99, 99]
        assert self.test_star.vel[0] == -99
        assert self.test_star.vel[1] == 99
        self.test_star.vel = np.array([-9, 9])
        assert self.test_star.vel[0] == -9
        assert self.test_star.vel[1] == 9

    def test_theta_star(self):
        theta_star = self.r_source * 0.00465047 / self.D_S  # R_Sun --> au, au / kpc = mas
        testing.assert_allclose(self.source.theta_star, theta_star, rtol=0.001)


class TestStarErrors(unittest.TestCase):

    def setUp(self):
        self.star = Star()

    def test_distance_pi_conflict(self):
        with self.assertRaises(KeyError):
            star = Star(distance=3., pi=1.)

    def test_mu_vel_conflict(self):
        with self.assertRaises(KeyError):
            star = Star(mu=3., vel=20.)

    def test_distance_error(self):
        with self.assertRaises(AttributeError):
            print(self.star.distance)

    def test_pi_error(self):
        with self.assertRaises(AttributeError):
            print(self.star.pi)

    def test_mu_error(self):
        with self.assertRaises(AttributeError):
            print(self.star.mu)

    def test_mu_setter_type_error(self):
        with self.assertRaises(TypeError):
            self.star.mu = 6.

    def test_vel_error(self):
        with self.assertRaises(AttributeError):
            print(self.star.vel)

    def test_vel_setter_type_error(self):
        with self.assertRaises(TypeError):
            self.star.vel = 6.
