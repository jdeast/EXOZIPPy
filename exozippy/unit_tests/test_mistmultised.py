import unittest
import pathlib
import numpy as np

import exozippy
from exozippy.sed.utils import mistmultised


class TestMistMultiSED(unittest.TestCase):
    """Validate that mistmultised reproduces EXOFASTv2 reference numbers."""

    # ---- reference values from EXOFASTv2 ---------------------------------
    _BLENDMAG_REF = np.array(
        [4.6577497, 4.9720854, 4.1773909, 3.6583169, 3.3270956,
         3.2899675, 3.2783915, 3.2992370, 3.2675890]
    )
    _MODELFLUX_REF = np.array(
        [0.013705665, 0.010260436, 0.021332591, 0.034409093, 0.046683319,
         0.048307324, 0.048825126, 0.047896652, 0.049313337]
    )
    _MAGRESID_REF = np.array(
        [6.6217503, 6.7965146, 6.4947091, 6.2776831, 6.2149044,
         6.1580325, 6.0986085, 6.1517630, 6.1214110]
    )

    def test_hat3_example(self):
        # ---------------- input setup -------------------------------------
        teff   = np.array([5777.0])
        logg   = np.array([4.4])
        feh    = np.array([0.0])
        av     = np.array([0.0])
        dist   = np.array([10.0])
        lstar  = np.array([1.0])
        errsc  = np.array([1.0])

        sedfile = (
            pathlib.Path(exozippy.MODULE_PATH) /
            "EXOZIPPy" / "data" / "exofastv2" / "examples" / "hat3" /
            "HAT-3.sed"
        )

        # ---------------- call mistmultised ------------------------------
        sedchi2, blendmag, modelflux, magresiduals = mistmultised(
            teff, logg, feh, av, dist, lstar, errsc,
            sedfile=sedfile,
            redo=False, psname=None, debug=False,
            atmospheres=None, wavelength=None, logname=None,
            xyrange=None, blend0=None
        )

        # ---------------- assertions -------------------------------------
        # np.testing.assert_allclose(sedchi2, 0.0, atol=1e-6,
        #                            err_msg="sedchi2 differs from reference")
        np.testing.assert_allclose(blendmag,
                                   self._BLENDMAG_REF,
                                   atol=1e-6,
                                   err_msg="blendmag differs from reference")
        # mistmultised returns flux with shape (nbands, nstars); transpose
        np.testing.assert_allclose(modelflux.T[0],
                                   self._MODELFLUX_REF,
                                   atol=1e-6,
                                   err_msg="modelflux differs from reference")
        np.testing.assert_allclose(magresiduals,
                                   self._MAGRESID_REF,
                                   atol=1e-6,
                                   err_msg="magresiduals differ from reference")


if __name__ == "__main__":
    unittest.main()
