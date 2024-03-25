"""
Example 01: Show how MM-EXOFAST should work on a point lens microlensing event
(OB08092).

Uses a data file as input.

https://ui.adsabs.harvard.edu/abs/2014ApJ...795...42P/abstract
"""
import os.path
import MulensModel as mm
import mmexofast as mmexo

from print_ex_metrics import print_metrics

# Read in the data file
SAMPLE_FILE_01 = os.path.join(
    mmexo.MULENS_DATA_PATH, "OB08092",
    "n20100309.I.OGLE4.ob08092.dat")

coords = "17:47:29.42, -34:43:35.6"

expected = {'mulens': {'t_0': 2455379.571, 'u_0': 0.523, 't_E': 17.94},
            'physical': {'D_L': 8.1, 'D_S': 9.7, 'M_L': 0.15,
                    'mu_rel': 3.2,
                     'theta_E': 0.33}}
# The physical parameters are NOT self-consistent. RP needs to fix this.
# Also, we probably want a class for holding the parameters.

priors = {'mu_S_geo': {'N': 1.17, 'E': 0.25}}

results = mmexo.mmexofast.fit(
    files=SAMPLE_FILE_01, coords=coords, priors=priors, fit_type='point lens',
    print_results=True, verbose=True)
print_metrics(results.final_parameters, expected)
