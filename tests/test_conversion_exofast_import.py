"""EXOFASTv2 -> EXOZIPPy prior import: the unit scales in ``PRIOR_MAP``.

Review 7.16.2.  ``utilities/exofast2exozippy.py`` rescales two EXOFASTv2
prior names on the way over: ``omega`` (radians in EXOFASTv2) onto
``orbit.omega`` (user unit deg), and ``mpsun`` (solar masses) onto
``planet.mass`` (user unit jupiterMass).  The EXOZIPPy side of each is read
through astropy by ``config.unit_conversion``, so the import scale is only
right if it is the SAME ratio astropy uses -- the IAU 2015 Resolution B3
nominal GM_sun/GM_jup = 1.3271244e20 / 1.2668653e17 = 1047.5655 -- and a
round trip (EXOFASTv2 value -> imported user value -> internal -> back to
EXOFASTv2's unit) has to close.
"""

import math

import numpy as np
import pytest

from exozippy.components.parameter import Parameter
from exozippy.config import unit_conversion
from exozippy.utilities import exofast2exozippy as e2z

GM_SUN = 1.3271244e20  # IAU 2015 B3
GM_JUP = 1.2668653e17


def test_omega_import_scale_is_radians_to_degrees_and_round_trips():
    """
    Given EXOFASTv2's omega in radians,
    When it is imported onto orbit.omega (deg) and that is converted to
      the internal unit (rad) by the Parameter layer,
    Then the import scale is 180/pi exactly and the internal value is the
      EXOFASTv2 number again.
    """
    _, target, scale = e2z.PRIOR_MAP["omega"]
    assert target == "orbit.{n}.omega"
    assert scale == pytest.approx(180.0 / math.pi, rel=1e-15)

    omega_rad = 1.234
    p = Parameter(
        label="orbit.0.omega",
        unit="deg",
        internal_unit="rad",
        initval=omega_rad * scale,
    )
    assert float(np.ravel(p.initval)[0]) == pytest.approx(omega_rad, rel=1e-15)


@pytest.mark.xfail(
    strict=True,
    reason="review 1.9.13: MSUN_PER_MJUP is 1047.348644 (the IAU 2009 Sun/Jupiter-system "
    "mass ratio), while planet.mass's jupiterMass is astropy's IAU 2015 "
    "nominal 1047.5655; an mpsun prior imports 2.1e-4 low",
)
def test_mpsun_import_scale_is_the_ratio_astropy_reads_jupiter_mass_with():
    """
    Given EXOFASTv2's planet mass in solar masses (mpsun),
    When it is imported onto planet.mass (user unit jupiterMass),
    Then the scale is the IAU 2015 nominal GM_sun/GM_jup -- the ratio the
      Parameter layer then uses to take jupiterMass back to solMass -- so
      the imported prior lands on the EXOFASTv2 mass exactly (round trip).
    """
    _, target, scale = e2z.PRIOR_MAP["mpsun"]
    assert target == "planet.{n}.mass"
    assert scale == pytest.approx(GM_SUN / GM_JUP, rel=1e-9)
    assert scale == pytest.approx(
        unit_conversion("solMass", "jupiterMass", "t"), rel=1e-12
    )

    mpsun = 0.00123
    p = Parameter(
        label="planet.0.mass",
        unit="jupiterMass",
        internal_unit="solMass",
        initval=mpsun * scale,
    )
    assert float(np.ravel(p.initval)[0]) == pytest.approx(mpsun, rel=1e-12)
