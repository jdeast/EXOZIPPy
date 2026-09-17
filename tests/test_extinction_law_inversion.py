"""components/sed/extinction.py: running the shipped law backwards.

Review 2.9.16.  The sweep wrote a BAND extinction (A_W149 = 2.75) into
`star.av`, which is A_V, and the SED then resolved the Teff-extinction
degeneracy by inventing a 2634 K source at half the true angular size.  The
inversion these tests pin is the fix, and its convention is the part that
bites: a band's extinction must be evaluated at the band's EFFECTIVE
wavelength, not integrated across the passband.
"""

import numpy as np
import pytest

from exozippy.components.sed.extinction import (
    a_lambda_over_av,
    av_from_band_extinction,
    av_from_colour_excess,
)

# Roman effective wavelengths, from the pickled SVO filter profiles
# (WavelengthEff / 1e4): W149 = 1.30496 um, Z087 = 0.865097 um.
W149_UM, Z087_UM = 1.3050, 0.8651
# DC2018 event 194's red-clump extinctions, event_info.txt columns 6 and 8.
A_W149, S_W149 = 2.75, 0.33
A_Z087, S_Z087 = 5.29, 0.63


def test_the_law_is_normalized_at_v():
    """
    Given the shipped extinction law,
    When A(lambda)/A_V is evaluated at V itself,
    Then it is 1 by construction.
    """
    assert a_lambda_over_av(0.55) == pytest.approx(1.0, abs=1e-12)


def test_every_band_inverts_to_the_same_av_when_the_law_shape_is_right():
    """
    Given two band extinctions and a reddening measured on one line of sight,
    When each is inverted through the law independently,
    Then all three agree -- and that agreement IS the law-shape check.

    This is the one diagnostic available with no truth: the two bands
    OVER-determine A_V under a fixed law, so a disagreement is the curve's
    shape being wrong and its size is the error.  On DC2018 event 194 they
    agree to 0.2%, which is how we know the shipped law is right for this
    dataset -- and it is the check to repeat on any new field.
    """
    av_w, sig_w = av_from_band_extinction(A_W149, W149_UM, S_W149)
    av_z, _ = av_from_band_extinction(A_Z087, Z087_UM, S_Z087)
    av_e, _ = av_from_colour_excess(
        A_Z087 - A_W149, Z087_UM, W149_UM, np.hypot(S_W149, S_Z087)
    )
    assert av_w == pytest.approx(9.877, abs=0.01)
    for other in (av_z, av_e):
        assert other == pytest.approx(av_w, rel=0.01), (
            f"per-band inversions disagree: {av_w:.3f} vs {other:.3f} -- "
            f"that is the law's SHAPE, not a bookkeeping slip"
        )
    # the sigma is carried through the same divisor, not invented
    assert sig_w == pytest.approx(S_W149 / a_lambda_over_av(W149_UM), rel=1e-9)


def test_a_band_extinction_is_not_av_and_the_difference_is_the_whole_bug():
    """
    Given A_W149 = 2.75 on event 194,
    When it is used AS A_V instead of inverted,
    Then it is wrong by the ~3.6x that produced review 2.9.16.
    """
    av, _ = av_from_band_extinction(A_W149, W149_UM)
    assert av / A_W149 == pytest.approx(3.59, abs=0.05)


def test_the_colour_excess_inversion_refuses_a_swapped_wavelength_order():
    """
    Given the two wavelengths handed over in the wrong order,
    When the reddening is inverted,
    Then it raises rather than returning a negative A_V.

    A silently negative A_V would be pinned at the grid's lower bound and
    read as "no dust", which is the same class of quiet wrongness as the
    band-for-A_V substitution this module exists to prevent.
    """
    with pytest.raises(ValueError, match="SHORTER wavelength"):
        av_from_colour_excess(2.54, W149_UM, Z087_UM)
