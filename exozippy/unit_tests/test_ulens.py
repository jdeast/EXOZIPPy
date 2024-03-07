"""
Unit tests for parameter conversions
"""
import unittest
from numpy import testing

from ulens import Star, Phys2UlensConverter



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
    result = [v_perp.N, v_perp.E]
    testing.assert_almost_equal([-0.5, 28.9], result, decimal=1)

    # Batista et al. 2011, mb09387:
    coords = "17:53:50.79 -33:59:25"
    t_ref = 2455042.34
    conv = Phys2UlensConverter(coords=coords, t_ref=t_ref)
    v_perp = conv.v_earth_perp
    result = [v_perp.N, v_perp.E]
    testing.assert_almost_equal([-3.60, 22.95], result, decimal=2)
