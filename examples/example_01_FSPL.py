"""
Example 01: Show how MM-EXOFAST should work on a point lens microlensing event
(OB08092).

Uses a data file as input.

"""
import os.path
import MulensModel as mm
import mmexofast as mmexo

from print_ex_metrics import print_metrics

# Read in the data file
SAMPLE_FILE_01 = os.path.join(
    mm.DATA_PATH, "photometry_files", "OB08092",
    "phot_ob08092_O4.dat")

coords = "17:47:29.42, -34:43:35.6"

expected = {'mulens': {'t_0': 5379.571, 'u_0': 0.523, 't_E': 17.94},
            'physical': {'D_L': 8.1, 'D_S': 9.7, 'M_L': 0.15,
                     'mu_S_geo': [1.17, 0.25], 'mu_rel': 3.2,
                     'theta_E': 0.33}}
# The physical parameters are NOT self-consistent. RP needs to fix this.
# Also, we probably want a class for holding the parameters.

results = mmexo.mmexofast.fit(
    files=SAMPLE_FILE_01, coords=coords, fit_type='point lens',
    print_results=True, verbose=True)
print_metrics(results.final_parameters, expected)
