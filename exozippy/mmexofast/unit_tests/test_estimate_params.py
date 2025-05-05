import unittest
import numpy.testing
import os.path
import MulensModel
import numpy as np
import matplotlib.pyplot as plt

from exozippy.mmexofast import estimate_params
from exozippy import MULENS_DATA_PATH
from exozippy.mmexofast import fitters


class TestGetWideParams_alpha(unittest.TestCase):
    """
    Test that the model calculated from a given t_pl produces an
    anomaly at t_pl.

            - 't_0' (*float*): Time of maximum magnification.
            - 'u_0' (*float*): Impact parameter.
            - 't_E' (*float*): Einstein crossing time.
            - 't_pl' (*float*): Time at which to compute the wide model parameters.
            - 'dt' (*float*): Duration of the anomaly
            - 'dmag' (*float*): Magnitude difference of the perturbation

    """

    def setUp(self):
        self.params = {'t_0': 0., 'u_0': 0.5, 't_E': 15., 't_pl': 0, 'dt': 0.7, 'dmag': 0.5}

    def estimate_params(self, params):
        return estimate_params.get_wide_params(params)

    def do_test(self, t_pl):
        params = {key: value for key, value in self.params.items()}
        params['t_pl'] = t_pl
        planet_params = self.estimate_params(params)
        if isinstance(planet_params, estimate_params.BinaryLensParams):
            planet_params = [planet_params]

        for params in planet_params:
            planet_model = MulensModel.Model(parameters=params.ulens)
            planet_model.set_magnification_methods([t_pl - 1., 'VBBL', t_pl + 1.])
            planet_model_mag = planet_model.get_magnification(t_pl)

            pspl_model = MulensModel.Model(
                parameters={'t_0': self.params['t_0'], 'u_0': self.params['u_0'], 't_E': self.params['t_E']})
            pspl_model_mag = pspl_model.get_magnification(t_pl)

            print(params.ulens)
            assert np.abs((planet_model_mag - pspl_model_mag) / pspl_model_mag) > 0.05

    def test_1(self):
        t_pl = self.params['t_0'] - 0.5 * self.params['t_E']
        self.do_test(t_pl)

    def test_2(self):
        t_pl = self.params['t_0'] - 1.5 * self.params['t_E']
        self.do_test(t_pl)

    def test_3(self):
        t_pl = self.params['t_0'] + 0.5 * self.params['t_E']
        self.do_test(t_pl)

    def test_4(self):
        t_pl = self.params['t_0'] + 1.5 * self.params['t_E']
        self.do_test(t_pl)


class TestGetCloseParams_alpha(TestGetWideParams_alpha):

    def estimate_params(self, params):
        return estimate_params.get_close_params(params)


class KB160625():
    """
    Parameters for KMT-2016-BLG-0625 Shin et al. 2024.
    """
    def __init__(self):
        # Section 3.4
        self.tau_pl = 0.609
        self.u_pl = 0.613
        self.t_pl = 7662.95
        self.t_E = 11.5
        self.s_close = 0.739
        self.s_wide = 1.352
        self.alpha = np.pi - np.array([0.12, 3.26])  # radians, alpha corrected to MMv3 system

        # Data from Table 5 (alpha corrected to MMv3 system):
        # s-
        self.close_params = {'t_0': 7655.951, 'u_0': 0.073, 't_E': 11.494,
                             's': 0.741, 'q': 2.357e-4, 'alpha': 180. - np.rad2deg(3.217),
                             'rho': 1.2256e-3}
        # s+
        self.wide_params = {'t_0': 7655.951, 'u_0': 0.075, 't_E':11.335,
                            's': 1.367, 'q': 0.727e-4, 'alpha': 180. - np.rad2deg(0.122), 'rho': 1.7656e-3}
        # 1L2S
        self.binary_source_params = {
            't_0_1': 7655.953, 'u_0_1': 0.078, 't_E': 10.946, 't_0_2': 7662.943, 'u_0_2': 3.751e-4, 'rho_2': 5.1309e-3, 'q_flux':  0.005
        }

        # dt and dmag estimated by-eye from figure in paper.
        self.dt = 0.25
        self.dmag = 0.3

        # Approximate values of t_0 and u_0
        self.t_0 = 7655.95
        self.u_0 = 0.075

        self.tol = 0.03  # 2% uncertainty based on variation in u0

        self.params = {
            't_0': self.t_0, 'u_0': self.u_0, 't_E': self.t_E, 't_pl': self.t_pl,
            'dt': self.dt, 'dmag': self.dmag
            }


class OB180383():
    """
    Parameters for OGLE-2018-BLG-0383 Wang et al. 2022.
    """
    def __init__(self):
        self.dmag = 0.07  # preamble to Section 3.2

        # Section 3.2.1 Heuristic analysis
        self.pspl_est = {'t_0': 8199.2, 'u_0': 0.071, 't_E': 11.3}  # Eq. 8
        self.t_0, self.u_0, self.t_E = self.pspl_est['t_0'], self.pspl_est['u_0'], self.pspl_est['t_E']
        self.tau_anom, self.u_pl, self.alpha_est = -2.04, 2.05, 1.98  # deg, Eq. 9, u_anom --> u_pl
        self.alpha = np.deg2rad(self.alpha_est)  # might need to check reflections/conversion to MMv3 system
        self.t_pl = self.pspl_est['t_0'] + self.tau_anom * self.pspl_est['t_E']  # derived
        self.s_wide, s_close = 2.46, 0.41  # Eq. 10: s_plus, s_minus
        self.dt = 2. * 0.55  # 2. * t_fwhm
        self.rho_est = 0.024  # Eq. 12
        self.delta_A = 0.61  # Eq. 13
        self.q_est = 1.8e-4  # Eq. 14

        self.params = {
            't_0': self.t_0, 'u_0': self.u_0, 't_E': self.t_E, 't_pl': self.t_pl,
            'dt': self.dt, 'dmag': self.dmag
            }
        self.tol = 0.03  # 2% uncertainty based on variation in u0 for 1S solutions. (incr to 7% for 2S soln)

        # Table 2: Fitted parameters
        pspl = {'t_0': 8199.244, 'u_0': 0.072, 't_E': 11.15}
        self.wide_params = {'t_0': 8199.239, 'u_0': 0.071, 't_E': 11.35, 'rho': 0.0238, 'alpha': 181.98, 's': 2.453, 'q': 2.14e-4}
        close_upper = {
            't_0': 8199.247, 'u_0': 0.071, 't_E': 11.34, 'rho': 0.0060, 'alpha': 355.86, 's': 0.405, 'q': 23.6e-4}
        close_lower = {
            't_0': 8199.247, 'u_0': 0.072, 't_E': 11.46, 'rho': 0.0056, 'alpha': 7.84, 's': 0.404, 'q': 21.5e-4}
        binary_source = {
            't_0_1': 8199.244, 't_0_2': 8176.022, 'u_0_1': 0.074, 'u_0_2': 0.0007, 't_E': 11.34,
            'rho_1': 0.058, 'rho_2': 0.0202, 'q_flux_I': 0.0057}


class TestGetWideParams(unittest.TestCase, KB160625):

    def setUp(self):
        KB160625.__init__(self)
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
        # self.alpha has 2 values. choose the one closest to the results.
        index = np.argmin(np.abs(self.ulens_params.ulens['alpha'] - np.rad2deg(self.alpha)))
        np.testing.assert_allclose(self.ulens_params.ulens['alpha'], np.rad2deg(self.alpha[index]), rtol=self.tol)

    def test_mag_methods(self):
        expected_values = [self.t_pl - 5. * self.dt / 2., 'VBBL', self.t_pl + 5. * self.dt / 2.]
        for actual, expected in zip(self.ulens_params.mag_methods[4:7], expected_values):
            if isinstance(actual, str):
                assert actual == expected
            else:
                np.testing.assert_allclose(actual, expected, atol=0.001)


class TestGetWideParams2(TestGetWideParams, OB180383):

    def setUp(self):
        OB180383.__init__(self)
        self.ulens_params = estimate_params.get_wide_params(self.params)

    def test_q_rho(self):
        # Gould & Gaucherel approximation
        # Ap = 2(q / ρ^2)
        np.testing.assert_allclose(self.ulens_params.ulens['rho'], self.rho_est)

        Ap_est = 2. * self.ulens_params.ulens['q'] / self.ulens_params.ulens['rho']**2
        np.testing.assert_allclose(Ap_est, self.delta_A, rtol=self.tol)

        np.testing.assert_allclose(self.ulens_params.ulens['q'], self.q_est)

class TestGetCloseParams(unittest.TestCase, KB160625):

    def setUp(self):
        KB160625.__init__(self)
        self.ulens_params = estimate_params.get_close_params(self.params, q=self)


class TestAnomalyParameterEstimator(unittest.TestCase):

    def setUp(self):
        datafile = os.path.join(MULENS_DATA_PATH, 'unit_test_data', 'planet4AF.dat')
        self.data = MulensModel.MulensData(
            file_name=datafile,
            phot_fmt='mag')
        self.true_params, self.input_fluxes = self._parse_header(datafile)
        self.pspl_params = {key: self.true_params[key] for key in ['t_0', 'u_0', 't_E']}
        self.af_results = {'t_0': 17.43489583333333, 't_eff': 0.421875, 'j': 2.0, 'chi2': 98.97724735834696,
                           'dchi2_zero': 218.83573427369782, 'dchi2_flat': 143.937564049782}

    def _parse_header(self, datafile):
        with open(datafile, 'r') as file_:
            lines = file_.readlines()

        elements = lines[0].split()
        ulens_params = {}
        for i, element in enumerate(elements):
            if element[-2:] == "':":
                key = element.strip('{')[1:-2]
                ulens_params[key] = float(elements[i + 1].strip(',').strip('}'))

        elements = lines[1].split()
        fluxes = {}
        for i, element in enumerate(elements):
            if element == '=':
                fluxes[elements[i - 1]] = float(elements[i + 1].strip(','))

        return ulens_params, fluxes

    def test_update_pspl_model(self):
        print('JCY: this test does not work. Maybe there is a change in origin. Maybe it is a bad test, regardless.')
        fitter = fitters.SFitFitter(datasets=[self.data], initial_model=self.pspl_params)
        fitter.run()
        test_pspl = {key: fitter.best[key] for key in self.pspl_params.keys()}
        estimator = estimate_params.AnomalyPropertyEstimator(
            datasets=self.data, pspl_params=test_pspl, af_results=self.af_results
        )
        estimator.update_pspl_model()

        #print(test_pspl)
        #print(self.pspl_params)
        #print(estimator.refined_pspl_params)
        #event = MulensModel.Event(datasets=self.data, model=MulensModel.Model(parameters=test_pspl))
        #event.plot()
        #
        #event_2 = MulensModel.Event(datasets=estimator.masked_datasets, model=MulensModel.Model(parameters=estimator.refined_pspl_params))
        #event_2.plot(show_bad=True)
        #plt.show()

        for key, value in estimator.refined_pspl_params.items():
            np.testing.assert_allclose(value, self.pspl_params[key], rtol=0.001)

    def test_get_anomaly_lc_params(self):
        estimator = estimate_params.AnomalyPropertyEstimator(
            datasets=self.data, pspl_params=self.pspl_params, af_results=self.af_results
        )
        results = estimator.get_anomaly_lc_parameters()
        expected = {'t_pl': 17.44, 'dt': 0.43, 'dmag': -0.12}
        print(results)
        for key, value in expected.items():
            np.testing.assert_allclose(results[key], value, rtol=0.2)


def test_model_pspl_at_pl():
    raise NotImplementedError()


def test_correct_alpha():
    expected = [20., 20., -20., -20.]
    input = [20., 380., -20, -740.]
    for value_exp, value_in in zip(expected, input):
        print('expected', value_exp, 'input', value_in)
        numpy.testing.assert_almost_equal(value_exp, estimate_params.correct_alpha(value_in))


class TestBinarySourceParams(unittest.TestCase):

    def test_set_source_flux_ratio(self):
        raise NotImplementedError()


def test_get_binary_source_params():
    raise NotImplementedError()
