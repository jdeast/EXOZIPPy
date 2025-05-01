import unittest
import MulensModel as mm
import os.path
import numpy as np
import matplotlib.pyplot as plt

from exozippy import MULENS_DATA_PATH
from exozippy.mmexofast import gridsearches


class TestEventFinderGridSearch_1(unittest.TestCase):

    def setUp(self):
        datafile = os.path.join(MULENS_DATA_PATH, 'unit_test_data', 'pspl4EF_1.dat')
        self.data = mm.MulensData(
            file_name= datafile,
            phot_fmt='mag')
        self.expected = self._parse_header(datafile)
        self.ef_grid = gridsearches.EventFinderGridSearch(datasets=self.data)

    def _parse_header(self, datafile):
        with open(datafile, 'r') as file_:
            lines = file_.readlines()

        elements = lines[0].split()
        expected = {}
        for i, element in enumerate(elements):
            if element == '=':
                expected[elements[i - 1]] = float(elements[i + 1].strip(','))

        return expected

    def test_grid_t_0(self):
        np.testing.assert_almost_equal(self.ef_grid.grid_params['t_0_min'], self.data.time[0] - (1./3.))
        np.testing.assert_almost_equal(self.ef_grid.grid_params['t_0_max'], self.data.time[-1] + (1. / 3.))

    def test_grid_t_eff(self):
        np.testing.assert_almost_equal(self.ef_grid.grid_params['t_eff_3'], 1.)
        np.testing.assert_almost_equal(self.ef_grid.grid_params['t_eff_max'], 99.)
        np.testing.assert_almost_equal(self.ef_grid.grid_params['d_t_eff'], 1. / 3.)

    def test_get_trimmed_datasets(self):
        params = {'t_0': 3., 't_eff': 5.}
        #print(self.ef_grid.z_t_eff)
        n_expected = params['t_eff'] * 2 * 5 / (self.data.time[1] - self.data.time[0])
        trimmed_data = self.ef_grid.get_trimmed_datasets(parameters=params)
        for dataset in trimmed_data:
            #print(n_expected, dataset.n_epochs, self.data.n_epochs)
            np.testing.assert_almost_equal(dataset.n_epochs, n_expected)

    def test_get_flat_chi2(self):
        raise NotImplementedError(
            'Technically this part of the code is covered. But it might be a good idea to have a special test for ' +
            'debugging.')

    def test_do_fits(self):
        raise NotImplementedError(
            'Technically this part of the code is covered. But it might be a good idea to have a special test for ' +
            'debugging.')

    def test_best(self):
        self.ef_grid.run()
        #for i in range(self.ef_grid.results.shape[1]):
        #    plt.figure()
        #    plt.title(i + 1)
        #    plt.scatter(self.ef_grid.grid_t_0, self.ef_grid.grid_t_eff, c=self.ef_grid.results[:, i])
        #    plt.colorbar()
        #
        #plt.show()

        index = self.ef_grid.grid_t_eff == self.ef_grid.best['t_eff']
        closest_t_0 = np.argmin(np.abs(self.ef_grid.grid_t_0[index] - self.expected['t_0']))
        best_expected = self.ef_grid.grid_t_0[index][closest_t_0]
        np.testing.assert_almost_equal(self.ef_grid.best['t_0'], best_expected)


class TestEventFinderGridSearch_2(TestEventFinderGridSearch_1):

    def setUp(self):
        datafile = os.path.join(MULENS_DATA_PATH, 'unit_test_data', 'pspl4EF_2.dat')
        self.data = mm.MulensData(
            file_name= datafile,
            phot_fmt='mag')
        self.expected = self._parse_header(datafile)
        self.ef_grid = gridsearches.EventFinderGridSearch(datasets=self.data)

    def test_bad_input_data(self):
        with self.assertRaises(ValueError):
            foo = gridsearches.EventFinderGridSearch(datasets=None)

        with self.assertRaises(TypeError):
            foo = gridsearches.EventFinderGridSearch(datasets=16.)

    def test_set_t_0_multiple_datasets(self):
        alt_times = [time for time in self.data.time]
        alt_times[0] -= 10.
        alt_times[-1] += 10.
        data_2 = mm.MulensData([alt_times, self.data.mag, self.data.err_mag], phot_fmt='mag')

        datasets = [self.data, data_2]
        test_grid = gridsearches.EventFinderGridSearch(datasets=datasets)

        np.testing.assert_almost_equal(test_grid.grid_params['t_0_min'], alt_times[0] - (1./3.))
        np.testing.assert_almost_equal(test_grid.grid_params['t_0_max'], alt_times[-1] + (1. / 3.))

    def test_set_t_0_manual(self):
        t_0_min = -5.
        t_0_max = 16.
        test_grid = gridsearches.EventFinderGridSearch(
            datasets=self.data, t_0_min=t_0_min, t_0_max=t_0_max)
        np.testing.assert_almost_equal(test_grid.grid_params['t_0_min'], t_0_min)
        np.testing.assert_almost_equal(test_grid.grid_params['t_0_max'], t_0_max)


class TestFlatSFitFunction(unittest.TestCase):

    def test_flatten_data(self):
        raise NotImplementedError()

    def test_calc_model(self):
        raise NotImplementedError()

    def test_calc_df(self):
        raise NotImplementedError()


class TestEFSFitFunction():

    def test_calc_model(self):
        raise NotImplementedError()

    def test_calc_residuals(self):
        raise NotImplementedError()

    def test_calc_df(self):
        raise NotImplementedError()

    def test_q(self):
        raise NotImplementedError()

    def test_magnification(self):
        raise NotImplementedError()

    def test_get_magnification(self):
        raise NotImplementedError()


class TestAnomalyFinderGridSearch(unittest.TestCase):

    def setUp(self):
        datafile = os.path.join(MULENS_DATA_PATH, 'unit_test_data', 'planet4AF.dat')
        self.data = mm.MulensData(
            file_name= datafile,
            phot_fmt='mag')
        self.expected = self._parse_header(datafile)
        print(self.expected)

        self.residuals = self._get_residuals()
        self.af_grid = gridsearches.AnomalyFinderGridSearch(residuals=self.residuals)

    def _parse_header(self, datafile):
        with open(datafile, 'r') as file_:
            lines = file_.readlines()

        elements = lines[0].split()
        expected = {}
        for i, element in enumerate(elements):
            if element[-2:] == "':":
                key = element.strip('{')[1:-2]
                expected[key] = float(elements[i + 1].strip(',').strip('}'))

        elements = lines[1].split()
        for i, element in enumerate(elements):
            if element == '=':
                expected[elements[i - 1]] = float(elements[i + 1].strip(','))

        return expected

    def _get_residuals(self):
        pspl_model = mm.Model(
            parameters={'t_0': self.expected['t_0'], 'u_0': self.expected['u_0'],
                        't_E': self.expected['t_E']})
        fit = mm.FitData(model=pspl_model, dataset=self.data)
        fit.fit_fluxes()
        res, err = fit.get_residuals(phot_fmt='flux')
        residuals = mm.MulensData([self.data.time, res, err], phot_fmt='flux')
        return residuals

    def test_run(self):
        self.af_grid.run()

        # t_eff is reasonable
        t_star = self.expected['t_E'] * self.expected['rho']
        assert self.af_grid.best['t_eff'] < (5. * t_star)

        # t_0 is as close to the expected value as possible given t_eff and the data spacing.
        index = self.af_grid.grid_t_eff == self.af_grid.xbest['t_eff']
        closest_t_0 = np.argmin(np.abs(self.af_grid.grid_t_0[index] - self.expected['t_pl']))
        best_expected = self.af_grid.grid_t_0[index][closest_t_0]
        np.testing.assert_allclose(
            self.af_grid.best['t_0'], best_expected,
            atol=(self.data.time[1] - self.data.time[0]))

    def test_get_zero_chi2(self):
        raise NotImplementedError()

    def test_check_successive(self):
        raise NotImplementedError()

    def test_do_fits(self):
        raise NotImplementedError()

    def test_get_anomalies(self):
        raise NotImplementedError()

    def test_filter_anomalies(self):
        raise NotImplementedError()

    def test_anomalies(self):
        raise NotImplementedError()

    def test_best(self):
        raise NotImplementedError()
    