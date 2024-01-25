import unittest
import numpy as np
from ..exozippy_rv import exozippy_rv

class TestExofastRVFunction(unittest.TestCase):
    def test_circular_orbit(self):
        # Test for a circular orbit (eccentricity = 0)
        tp = 2450000.0
        period = 10.123456
        bjd = np.linspace(tp,tp+period,10)
        gamma = 20.0
        K = 5.0

        result = exozippy_rv(bjd, tp, period, gamma, K)
        expected_result = np.array([20.0000000000,16.7860619518,15.0759612351,15.6698729813,18.2898992834,21.7101007163,24.3301270186,24.9240387650,23.2139380484,20.0000000004])

        np.testing.assert_allclose(result, expected_result, atol=1e-8)

    def test_non_circular_orbit(self):
        # Test for a non-circular orbit (eccentricity > 0)
        tp = 2450000.0
        period = 10.123456
        bjd = np.linspace(tp,tp+period,10)
        gamma = 20.0
        K = 5.0
        ecc = 0.2
        omega = np.pi/3.0
        result = exozippy_rv(bjd, tp, period, gamma, K, e=ecc, omega=omega)
        expected_result = np.array([23.0000000000,18.1619749562,15.7195132598,15.7280199920,17.0473193210,19.0943685806,21.5932583602,24.1591838314,25.4965743008,23.0000000005])

        self.assertTrue(np.all(np.isfinite(result)))  # Check if all values are finite

if __name__ == '__main__':
    unittest.main()
