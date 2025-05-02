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
        init_params = {'t_0': 4., 'u_0': 0.01, 't_E': 20.}
        sfit = fitters.SFitFitter(datasets=self.data, initial_model=init_params)
        for key, value in sfit.initial_model.items():
            assert init_params[key] == value

    def test_run(self):
        sfit = fitters.SFitFitter(datasets=[self.data], initial_model={'t_0': 4., 'u_0': 0.01, 't_E': 20.})
        sfit.run()
        for key, value in sfit.best.items():
            if key in self.true_params.keys():
                np.testing.assert_allclose(value, self.true_params[key], rtol=0.01)
