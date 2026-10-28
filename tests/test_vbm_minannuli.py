"""VBMDirectMagOp forces a second limb-darkening annulus (vbm.minannuli = 2).

Pins the workaround for a VBMicrolensing false convergence described in
src/exozippy/components/mulensing/mulensing.md ("VBM's limb-darkened finite
source never stops after one annulus"): with VBM's default minannuli = 1,
MultiMag2 can accept its first annulus and return the uniform-source
magnification for a limb-darkened source.  Remove this file together with
the setting and the doc paragraph once VBM's own default guards it.
"""

import numpy as np
import VBMicrolensing

from exozippy.components.mulensing.op import VBMDirectMagOp

H = float.fromhex

# A three-lens geometry from a fit of OGLE-2013-BLG-0341 (Gould et al. 2014)
# near the central caustic; linear LD.  Exact inputs: the trigger is a
# coincidence (the uniform-disk and center point-source magnifications agree
# to 1.5e-5), so they are given bit for bit.
X = H("0x1.87c40a0f01044p-8")
Y = H("-0x1.2af987e7626f1p-9")
RHO = H("0x1.a088a123d196ep-9")
A1 = H("0x1.820c49ba5e354p-1")
GEOMETRY = [
    H(s)
    for s in [
        "0x1.0c7b4bec5a1b5p-3",
        "0x1.917a3d9fffc25p-6",
        "0x1.2a9d1593c9bbap-2",
        "-0x1.a0b517d718751p-1",
        "0x1.c9f1ab9f63cdap-4",
        "0x1.b9ca9b3aa3a86p-15",
        "-0x1.b9a19be509294p-5",
        "-0x1.4ad7b3c39ba4ap-7",
        "0x1.6aaa8e0bae379p-1",
    ]
]

A_LIMB_DARKENED = 174.6366  # minannuli = 2 at Tol 1e-3 (174.6367 at Tol 1e-5)
A_UNIFORM = 170.1815  # what VBM's default returns here: the uniform source


def _companions():
    """(s, q, alpha) per companion that reproduce GEOMETRY exactly under the
    Op's own layout (companion j at s (cos alpha, -sin alpha) from the
    primary, shifted to the center of mass)."""
    g = np.array(GEOMETRY).reshape(3, 3)
    pos, m = g[:, :2], g[:, 2]
    out = []
    for j in (1, 2):
        dx, dy = pos[j] - pos[0]
        out.append((np.hypot(dx, dy), m[j] / m[0], np.arctan2(-dy, dx)))
    return out


def test_limb_darkened_multimag_does_not_stop_after_one_annulus():
    """
    Given the reproducer geometry, where VBM's first-annulus error estimate
    vanishes by coincidence,
    When the Op computes the limb-darkened triple-lens magnification,
    Then it returns the limb-darkened value, not the uniform-source one.
    """
    # Arrange
    op = VBMDirectMagOp(
        "270.0d -30.0d", n_companions=2, use_rho=True, bandpass="I"
    )
    # Precondition: the bug this guards is still present in the installed
    # VBM.  If this fails, upstream's default now guards it: remove the
    # workaround (op._build_vbm), its mulensing.md paragraph and this file.
    raw = VBMicrolensing.VBMicrolensing()
    raw.SetLDprofile(VBMicrolensing.VBMicrolensing.LDlinear)
    raw.SetMethod(VBMicrolensing.VBMicrolensing.Method.Multipoly)
    raw.SetLensGeometry(GEOMETRY)
    raw.a1 = A1
    assert abs(raw.MultiMag2(X, Y, RHO) - A_UNIFORM) < 1e-3, (
        "VBM's default no longer falls back to the uniform source here; "
        "the minannuli workaround can be removed"
    )

    # Act
    A = op._magnify(_companions(), np.array([X]), np.array([Y]), RHO, A1)

    # Assert
    assert op._vbm.minannuli == 2
    np.testing.assert_allclose(A, [A_LIMB_DARKENED], atol=1e-4)
    # A uniform-source Op is exact in one annulus and keeps VBM's default
    # (a second annulus would nearly double its cost).
    uniform = VBMDirectMagOp("270.0d -30.0d", n_companions=2, use_rho=True)
    assert uniform._vbm.minannuli == 1
