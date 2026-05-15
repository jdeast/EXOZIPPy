import unittest

from exozippy.mmexofast import AnomalyClassifier


class TestAnomalyClassifier(unittest.TestCase):

    def setUp(self):
        self.classifier = AnomalyClassifier()

    def test_dip(self):
        dip_params = {'t_0': 2457942.6, 't_E': 47., 'u_0': 0.25, 'dmag': 0.25, 'dt': 1.0, 't_pl': 2457958.7}  # KB171194
        result = self.classifier.classify(dip_params)
        assert result == 'close'

    def test_bump(self):
        bump_params = {'t_0':   2453582.7281740606, 'u_0':   0.355227507989543,'t_E':   11.106795114521415, 'dmag': -0.1, 'dt': 0.5, 't_pl': 2453592.85}  # OB05390
        result = self.classifier.classify(bump_params)
        assert result == 'wide'

    def test_hm(self):
        hm_params = {'t_0': 2453480.68, 't_E': 73.9, 'u_0': 0.023, 'dmag': 0.3, 'dt': 1.5, 't_pl': 2453480.6}  # OB05071
        result = self.classifier.classify(hm_params)
        assert result == 'high_mag'
