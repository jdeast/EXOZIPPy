"""
Unit tests for parameter conversions
"""
import unittest
from numpy import testing

from ulens import Star, phys2ulens


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

        lens = Star(mass=0.5, distance=6.0, mu=(0, 1))
        source = Star(mass=1.0, distance=8.0, radius=1.0, mu=(1, 0))

        ulens_params = phys2ulens(
            source=source, lens=lens, coords=self.coords,
            tref=self.ulens_expected['t_0'])
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

        lens = Star(mass=0.5, distance=6.0, vel=(0, 200))
        source = Star(mass=1.0, distance=8.0, radius=1.0, vel=(200, 0))

        ulens_params = phys2ulens(
            source=source, lens=lens, coords=self.coords,
            tref=self.ulens_expected['t_0'])
        testing.assert_almost_equal(
            ulens_params['t_E'], self.ulens_expected['t_E'])
        testing.assert_almost_equal(
            ulens_params['rho'], self.ulens_expected['rho'])
