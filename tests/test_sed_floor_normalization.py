"""The teffsed/fbolsed floor potentials keep their -log(sigma) (review 1.9.3).

Both tie the star's own teff/fbol to the SED's with a sigma that is a
FRACTION of the star's value, i.e. a function of a sampled parameter --
exactly the case where the house convention keeps the normalization
(components/relations.py's ``_add_penalty``, and mann vs torres, whose
sigma is a constant in dex and so may drop it). It was dropped here,
leaving a systematic 1 nat/e-fold tilt toward larger teff and fbol.
"""

import numpy as np
import pytensor.tensor as pt

from exozippy.components.sed.sed import SED


def _eval(value, sed_value, floor):
    return float(
        SED._fractional_floor_logp(
            pt.as_tensor_variable(float(value)),
            pt.as_tensor_variable(float(sed_value)),
            floor,
        ).eval()
    )


def test_the_floor_logp_carries_the_minus_log_sigma_term():
    """
    Given a value, an SED value and a fractional floor,
    When the floor logp is evaluated,
    Then it is the Gaussian exponent MINUS log(value * floor).

    The 2pi is dropped, matching relations._add_penalty.
    """
    # ARRANGE
    value, sed_value, floor = 5000.0, 5100.0, 0.02
    sigma = value * floor

    # ACT
    got = _eval(value, sed_value, floor)

    # ASSERT
    expected = -0.5 * ((value - sed_value) / sigma) ** 2 - np.log(sigma)
    assert got == expected


def test_the_normalization_removes_the_one_nat_per_efold_tilt():
    """
    Given a perfectly matched pair (value == sed_value, so the chi2 term
      and its derivative both vanish),
    When the value is raised by one e-fold,
    Then the logp falls by exactly 1 nat.

    That is the whole content of the fix: without -log(sigma), widening
    sigma by raising the value bought free likelihood, tilting teff and
    fbol upward for no physical reason.
    """
    # ARRANGE
    floor = 0.024
    value = 1.0

    # ACT
    at_1 = _eval(value, value, floor)
    at_e = _eval(value * np.e, value * np.e, floor)

    # ASSERT
    np.testing.assert_allclose(at_1 - at_e, 1.0, rtol=1e-12)


def test_the_chi2_half_is_unchanged():
    """
    Given two points at the same value but different SED values,
    When their floor logps are differenced,
    Then the difference is exactly the chi2 difference -- the added term
      depends on the value only, so it cancels.
    """
    # ARRANGE
    value, floor = 5000.0, 0.02
    sigma = value * floor

    # ACT
    delta = _eval(value, 5100.0, floor) - _eval(value, 4900.0, floor)

    # ASSERT
    expected = -0.5 * (100.0 / sigma) ** 2 + 0.5 * (100.0 / sigma) ** 2
    np.testing.assert_allclose(delta, expected, atol=1e-12)
