import unittest
import numpy as np
from exozippy.mmexofast import estimate_params


class TestGetWideParams(unittest.TestCase):
    """
    Parameters for KMT-2016-BLG-0625 Shin et al. 2024.
    """

    def setUp(self):
        self.t_0 = 7655.951
        self.u_0 = 0.073
        self.t_E = 11.5
        self.t_pl = 7662.95
        self.tau_pl = 0.609
        self.u_pl = 0.613
        self.s_close = 0.739
        self.s_wide = 1.352
        self.alpha = [0.12, 3.26]  # radians

        # dt and dmag estimated by-eye from figure in paper.
        self.dt = 0.25
        self.dmag = 0.3

        self.tol = 0.03  # 2% uncertainty based on variation in u0

        self.params = {
            't_0': self.t_0, 'u_0': self.u_0, 't_E': self.t_E, 't_pl': self.t_pl,
            'dt': self.dt, 'dmag': self.dmag
            }

        self.ulens_params = estimate_params.get_wide_params(self.params)

    def test_pspl(self):
        """
        t_0, u_0, t_E
        """
        for key in ['t_0', 'u_0', 't_E']:
            assert self.ulens_params.ulens[key] == self.__getattribute__(key)

    def test_s(self):
        np.testing.assert_allclose(self.ulens_params.ulens['s'], self.s_wide, rtol=self.tol)

    def test_alpha(self):
        alpha = np.min(np.abs(self.ulens_params.ulens['alpha'] - np.rad2deg(self.alpha)))
        np.testing.assert_allclose(self.ulens_params.ulens['alpha'], alpha, rtol=self.tol)

    def test_q(self):
        raise NotImplementedError()

    def test_rho(self):
        raise NotImplementedError()

    def test_mag_methods(self):
        expected_values = [self.t_pl - 5. * self.dt, 'VBBL', self.t_pl + 5. * self.dt]
        for actual, expected in zip(self.ulens_params.mag_method, expected_values):
            if isinstance(actual, str):
                assert actual == expected
            else:
                np.testing.assert_allclose(actual, expected, atol=0.001)


def test_get_close_params():
    raise NotImplementedError()


def test_model_pspl_at_pl():
    raise NotImplementedError()


class TestBinarySourceParams(unittest.TestCase):

    def test_set_source_flux_ratio(self):
        raise NotImplementedError()


def test_get_binary_source_params():
    raise NotImplementedError()
