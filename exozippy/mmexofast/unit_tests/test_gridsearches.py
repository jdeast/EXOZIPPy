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

    def get_flat_chi2(self):
        raise NotImplementedError()

    def test_do_fits(self):
        raise NotImplementedError()

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

    def test_run(self):
        raise NotImplementedError()

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
    