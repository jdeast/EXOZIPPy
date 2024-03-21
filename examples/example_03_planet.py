"""
Example 03: Show how MM-EXOFAST should work on OB05390, a classic planetary
microlensing event.

Using YAML as input.

 https://ui.adsabs.harvard.edu/abs/2006Natur.439..437B/abstract
"""
import os.path
import numpy as np
import astropy.units as u

import MulensModel as mm
import mmexofast as mmexo

from print_ex_metrics import print_metrics

# REWRITE INPUTS USING YAML BECAUSE IT NEEDS TO CAPTURE BAND INFORMATION!
# Data files
dir_ = os.path.join(mm.MODULE_PATH, "examples", "OB05390_lightcurves")
file_1 = os.path.join(dir_, "OGLE_I.dat mag")
file_2 = os.path.join(dir_, "MOA_r.dat mag")
file_3 = os.path.join(dir_, "Canopus_I.dat mag")
file_4 = os.path.join(dir_, "Danish_I.dat mag")
file_5 = os.path.join(dir_, "FTN_R.dat mag")
file_6 = os.path.join(dir_, "Perth_I.dat mag")
files = [file_1, file_2, file_3, file_4, file_5, file_6]

coords = "17:54:19.2 -30:22:38"

priors = {'theta_star': ['gauss', 5.25, 0.73], 'D_S': ['gauss', 8.001, 1.905],
          'gamma': {'I': 0.538, 'R': 0.626}}
# Paper defines: DS = 1.05 +- 0.25 RGC, RGC = 7.62+- 0.32 kpc. Not sure how the
# uncertainty in RGC should propagate into the uncertainty in DS, so I'm
# ignoring it.

expected = {
    'mulens': {'s': [1.610, 0.008], 'q': [0.000076, 0.000007],
               'u_0': [0.359, 0.005], 't_E': [11.03, 0.11],
               't_0': [2453582.731, 0.005], 't_star': [0.282, 0.010],
               'alpha': [np.rad2deg(2.756), np.rad2deg(0.003)]},
    'physical': {'R_S': [9.6, 1.3]}, 'Teff_S': 5200,
        'M_2': [5.5, -2.7, 5.5, u.M_e], 'a': [2.6, -0.6, 1.5],
        'M_1': [0.22, -0.11, 0.21], 'D_L': [6.6, 1.0]}, 'P': [9, -3, 9]}}
# Should the source and lens parameters be reported separately?

results = mmexo.mmexofast.fit(
    files=files, coords=coords, priors=priors, fit_type='binary lens',
    print_results=True)
print_metrics(results.final_parameters, expected)
