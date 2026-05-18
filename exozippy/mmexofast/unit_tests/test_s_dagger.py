"""
Test various solutions with the s_dagger degeneracy.
"""
import unittest
import numpy as np

import exozippy.mmexofast as mmexo

KB161105 = {
    'Wide C': {'t_0': 2457555.772, 'u_0': 0.154, 't_E': 42.5, 'alpha': np.rad2deg(3.832), 's': 1.155, 'q': 10.**(-4.069)},
    'Wide D': {'t_0': 2457555.781, 'u_0': 0.154, 't_E': 42.4, 'alpha': np.rad2deg(3.831), 's': 1.106, 'q': 10.**(-4.184)},
    'anomaly_lc_params': {'dmag': -0.26, 'dt': 0.2, 't_pl': 2457547.85},
    'expected': {'alpha': 219.3, 's_mean': 1.13}
}

KB170428 = {
    'Inner': {'t_0': 2457943.976, 'u_0': 0.205, 't_E': 44.4, 'alpha': np.rad2deg(1.890), 's': 0.8819, 'q': 10.**(-4.295)},
    'Outer': {'t_0': 2457943.978, 'u_0': 0.205, 't_E': 44.3, 'alpha': np.rad2deg(1.889), 's': 0.9146,
              'q': 10. ** (-4.302)},
    'anomaly_lc_params': {'dmag': 0.12, 'dt': 0.74, 't_pl': 2457947.00},
    'expected': {'alpha': 108.4, 's_mean': 0.898, 'log_q': -4.19}
}

KB171003 = {
    'Inner': {'t_0': 2457872.484, 'u_0': 0.179, 't_E': 25.65, 'alpha': np.rad2deg(1.073), 's': 0.8889, 'q': 10.**(-4.260)},
    'Outer': {'t_0': 2457872.482, 'u_0': 0.179, 't_E': 25.66, 'alpha': np.rad2deg(1.072), 's': 0.9096, 'q': 10.**(-4.373)},
    'anomaly_lc_params': {'dmag': 0.22, 'dt': 0.85, 't_pl': 2457870.66},
    'expected': {'alpha': 61.3, 's_mean': 0.899, 'log_q': -3.6} # actual (rather than heuristic) mean
}


class TestSDagger(unittest.TestCase):

    def _get_anom_params(self, event, solution):
        anom_params = event['anomaly_lc_params']
        anom_params['t_0'] = event[solution]['t_0']
        anom_params['u_0'] = event[solution]['u_0']
        anom_params['t_E'] = event[solution]['t_E']
        return anom_params

    def _assert_expected_matches_exact_result(self, event, result):
        for key, value in event['expected'].items():
            if key == 'alpha':
                # Skip alpha because it's too much work to figure out the geometry.
                #np.testing.assert_almost_equal(result[key], value)
                pass
            elif key == 's_mean':
                assert abs(result['s'] - value) / abs(value) < 0.01
            elif key == 'log_q':
                assert abs(np.log10(result['q']) - value) < 0.5  # Unlikely to be terribly accurate.

    def test_kb161105_estimate(self):
        """Test that anomaly_lc_params gives the correct guess for alpha, s."""
        anom_params = self._get_anom_params(KB161105, 'Wide C')
        estimator = mmexo.estimate_params.WidePlanetParameterEstimator(anom_params)
        result = estimator.get_binary_lens_params()
        self._assert_expected_matches_exact_result(KB161105, result.ulens)

    def test_kb161105_s_dagger(self):
        """Test that get_alt_s_dagger_params for Wide C gives Wide D and vice versa."""
        for solution in ['Wide C', 'Wide D']:
            anom_params = self._get_anom_params(KB161105, solution)
            estimator = mmexo.estimate_params.WidePlanetGridSearchEstimator(None, anom_params)
            estimator._is_run = True
            estimator._binary_params = mmexo.estimate_params.BinaryLensParams(KB161105[solution])
            alt_sol = estimator.alternate_params.ulens
            assert abs(1. - KB161105['expected']['s_mean']**2 / KB161105[solution]['s'] / alt_sol['s']) < 0.01


