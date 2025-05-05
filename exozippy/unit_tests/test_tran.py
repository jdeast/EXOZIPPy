import unittest
import numpy as np
import sys, os
# sys.path.append('../../../sfit_minimizer/source')
from exozippy.exozippy_tran import exozippy_tran  # Adjust import if needed

class TestExozippyTranFunction(unittest.TestCase):
    def test_transit_model_vs_exofastv2(self):
        # Input time array (days) centered on mid-transit
        time = np.linspace(-0.05, 0.05, 50)

        # Parameters for the transit model
        inc = np.radians(90.0)      # inclination in radians
        ar = 15.0                   # semi-major axis / stellar radius
        tp = 0.0                    # time of periastron
        period = 3.0                # orbital period (days)
        e = 0.0                     # eccentricity
        omega = np.radians(90.0)   # argument of periastron
        p = 0.1                     # planet radius / stellar radius
        u1 = 0.3                    # limb darkening coefficient 1
        u2 = 0.2                    # limb darkening coefficient 2
        f0 = 1.0                    # baseline flux

        # Reference EXOFASTv2 flux values (from earlier)
        exofastv2_transit = np.array([
            1.0000000, 1.0000000, 1.0000000, 1.0000000, 1.0000000, 1.0000000, 1.0000000,
            1.0000000, 0.99892081, 0.99583177, 0.99241022, 0.99085690, 0.99027731, 0.98986287,
            0.98954655, 0.98929758, 0.98909840, 0.98893790, 0.98880854, 0.98870500, 0.98862345,
            0.98856108, 0.98851590, 0.98848651, 0.98847203, 0.98847203, 0.98848651, 0.98851590,
            0.98856109, 0.98862345, 0.98870500, 0.98880854, 0.98893790, 0.98909840, 0.98929758,
            0.98954656, 0.98986287, 0.99027732, 0.99085691, 0.99241028, 0.99583183, 0.99892086,
            1.0000000, 1.0000000, 1.0000000, 1.0000000, 1.0000000, 1.0000000, 1.0000000, 1.0000000
        ])

        # Run the model
        model_flux = exozippy_tran(time, inc, ar, tp, period, e, omega, p, u1, u2, f0)

        # Check shape
        self.assertEqual(model_flux.shape, exofastv2_transit.shape, "Output shape mismatch")

        # Assert that the model produces a dip
        self.assertLess(np.min(model_flux), f0 - 1e-6, "No transit dip detected")

        # Compare to EXOFASTv2 within a tolerance
        np.testing.assert_allclose(model_flux, exofastv2_transit, atol=1e-6, err_msg="Model does not match EXOFASTv2 output within tolerance")

if __name__ == '__main__':
    unittest.main()
