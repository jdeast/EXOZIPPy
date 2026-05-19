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
    'Outer': {'t_0': 2457943.978, 'u_0': 0.205, 't_E': 44.3, 'alpha': np.rad2deg(1.889), 's': 0.9146, 'q': 10.**(-4.302)},
    'anomaly_lc_params': {'dmag': 0.12, 'dt': 0.74, 't_pl': 2457947.00},
    'expected': {'alpha': 108.4, 's_mean': 0.898, 'log_q': -4.19}
}

KB171003 = {
    'Inner': {'t_0': 2457872.484, 'u_0': 0.179, 't_E': 25.65, 'alpha': np.rad2deg(1.073), 's': 0.8889, 'q': 10.**(-4.260)},
    'Outer': {'t_0': 2457872.482, 'u_0': 0.179, 't_E': 25.66, 'alpha': np.rad2deg(1.072), 's': 0.9096, 'q': 10.**(-4.373)},
    'anomaly_lc_params': {'dmag': 0.22, 'dt': 0.85, 't_pl': 2457870.66},
    'expected': {'alpha': 61.3, 's_mean': 0.899, 'log_q': -3.6}  # actual (rather than heuristic) mean
}


class SDaggerTestBase():
    """
    Base class for s_dagger degeneracy tests. Subclasses must define:
        event                    - the event dict
        solutions                - list of solution keys within event (e.g. ['Inner', 'Outer'])
        parameter_estimator_class
        grid_search_estimator_class
    """
    s_dagger_tol = 0.01  # override in subclasses as needed
    event = None
    solutions = None
    parameter_estimator_class = None
    grid_search_estimator_class = None

    def _get_anom_params(self, solution):
        anom_params = self.event['anomaly_lc_params'].copy()  # avoid mutating shared fixture
        anom_params['t_0'] = self.event[solution]['t_0']
        anom_params['u_0'] = self.event[solution]['u_0']
        anom_params['t_E'] = self.event[solution]['t_E']
        return anom_params

    def _assert_expected_matches_exact_result(self, result):
        for key, value in self.event['expected'].items():
            if key == 'alpha':
                # Skip alpha because it's too much work to figure out the geometry.
                pass
            elif key == 's_mean':
                #assert abs(result['s'] - value) / abs(value) < 0.01
                assert abs(1. - result['s'] / value) < self.s_dagger_tol
            elif key == 'log_q':
                assert abs(np.log10(result['q']) - value) < 0.5  # Unlikely to be terribly accurate.

    def test_estimate(self):
        """Test that anomaly_lc_params gives the correct guess for alpha, s."""
        anom_params = self._get_anom_params(self.solutions[0])
        estimator = self.parameter_estimator_class(anom_params)
        result = estimator.get_binary_lens_params()
        self._assert_expected_matches_exact_result(result.ulens)

    def _check_s_dagger(self, solution):
        """Helper: check alternate_params for one solution gives its counterpart."""
        anom_params = self._get_anom_params(solution)
        estimator = self.grid_search_estimator_class(None, anom_params)
        estimator._is_run = True
        estimator._binary_params = mmexo.estimate_params.BinaryLensParams(
            self.event[solution])
        alt_sol = estimator.alternate_params.ulens
        ratio = self.event['expected']['s_mean'] ** 2 / self.event[solution]['s'] / alt_sol['s']
        assert abs(1. - ratio) < self.s_dagger_tol, (
            f"solution='{solution}': expected ratio~1.0, got {ratio:.4f} "
            f"(s={self.event[solution]['s']:.4f}, alt_s={alt_sol['s']:.4f}, "
            f"s_mean={self.event['expected']['s_mean']:.4f}, "
            f"tol={self.s_dagger_tol})"
        )

    def test_s_dagger_1(self):
        self._check_s_dagger(self.solutions[0])

    def test_s_dagger_2(self):
        self._check_s_dagger(self.solutions[1])


class TestKB161105SDagger(SDaggerTestBase, unittest.TestCase):
    event = KB161105
    solutions = ['Wide C', 'Wide D']
    parameter_estimator_class = mmexo.estimate_params.WidePlanetParameterEstimator
    grid_search_estimator_class = mmexo.estimate_params.WidePlanetGridSearchEstimator


class TestKB170428SDagger(SDaggerTestBase, unittest.TestCase):
    event = KB170428
    solutions = ['Inner', 'Outer']
    parameter_estimator_class = mmexo.estimate_params.ClosePlanetParameterEstimator
    grid_search_estimator_class = mmexo.estimate_params.ClosePlanetGridSearchEstimator


class TestKB171003SDagger(SDaggerTestBase, unittest.TestCase):
    s_dagger_tol = 0.03  # Sloped sides, different q for each solution.
    event = KB171003
    solutions = ['Inner', 'Outer']
    parameter_estimator_class = mmexo.estimate_params.ClosePlanetParameterEstimator
    grid_search_estimator_class = mmexo.estimate_params.ClosePlanetGridSearchEstimator
