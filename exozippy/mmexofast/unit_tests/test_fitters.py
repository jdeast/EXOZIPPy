import unittest
import os.path
import MulensModel as mm
import numpy as np

from exozippy import MULENS_DATA_PATH
from exozippy.mmexofast import fitters


class TestSFitFitter_1(unittest.TestCase):

    def setUp(self):
        datafile = os.path.join(MULENS_DATA_PATH, 'unit_test_data', 'pspl4EF_1.dat')
        self.data = mm.MulensData(
            file_name= datafile,
            phot_fmt='mag')
        self.true_params, self.input_fluxes = self._parse_header(datafile)
        self.initial_guess = {'t_0': 4., 'u_0': 0.01, 't_E': 20.}

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

    def test_initial_model(self):
        sfit = fitters.SFitFitter(datasets=self.data, initial_model=self.initial_guess)
        for key, value in sfit.initial_model.items():
            assert self.initial_guess[key] == value

    def test_run(self):
        sfit = fitters.SFitFitter(datasets=[self.data], initial_model=self.initial_guess)
        sfit.run()
        for key, value in sfit.best.items():
            if key in self.true_params.keys():
                np.testing.assert_allclose(value, self.true_params[key], rtol=0.001)


class TestSFitFitter_2(TestSFitFitter_1):

    def setUp(self):
        datafile = os.path.join(MULENS_DATA_PATH, 'unit_test_data', 'pspl4EF_2.dat')
        self.data = mm.MulensData(
            file_name= datafile,
            phot_fmt='mag')
        self.true_params, self.input_fluxes = self._parse_header(datafile)
        self.initial_guess = {'t_0': 4., 'u_0': 0.7, 't_E': 20.}


class TestWidePlanetFitter(unittest.TestCase):

    def setUp(self):
        datafile = os.path.join(MULENS_DATA_PATH, 'unit_test_data', 'planet4AF.dat')
        self.data = mm.MulensData(
            file_name= datafile,
            phot_fmt='mag')
        self.true_params, self.input_fluxes = self._parse_header(datafile)
        #self.pspl = {key: self.true_params[key] for key in ['t_0', 'u_0', 't_E']}
        self.anomaly_lc_params = {'t_0': 5.012676614091774, 'u_0': 0.3001165942835024, 't_E': 16.049057513235873, 't_pl': 17.43489583333333, 'dt': 0.421875, 'dmag': -0.12254925495518165}
        self.fitter = fitters.WidePlanetFitter(datasets=[self.data], anomaly_lc_params=self.anomaly_lc_params)

    def _parse_header(self, datafile):
        return TestSFitFitter_1._parse_header(self, datafile=datafile)

    def test_run(self):
        self.fitter.run()
        for key, value in self.fitter.best.items():
            if key in self.true_params.keys():
                np.testing.assert_allclose(value, self.true_params[key], rtol=0.01)

