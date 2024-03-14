"""
Example 01: Show how MM-EXOFAST should work on an FSPL microlensing event.

DRAFT!!! As can be seen below, some architecture things need more thought.
"""
import os.path
import MulensModel as mm
import mmexofast as mmexo
import exozippy

# Read in the data file
SAMPLE_FILE_01 = os.path.join(
    mm.DATA_PATH, "photometry_files", "OB08092",
    "phot_ob08092_O4.dat")

expected = {'ulens': {'t_0': 5379.571, 'u_0': 0.523, 't_E': 17.94},
            'phys': {'D_L': 8.1, 'D_S': 9.7, 'M_L': 0.15,
                     'mu_S_geo': [1.17, 0.25], 'mu_rel': 3.2, 'theta_E': 0.33}}
# The physical parameters are NOT self-consistent. RP needs to fix this.
# Also, we probably want a class for holding the parameters.


def print_metrics(results, expected):
    for key in expected.keys():
        print('Param type:', key)
        print('key, Dsig^2, Dfrac^2')
        for param_key in expected[key].keys():
            exp_value = expected[key][param_key]
            obs_value = results[key][param_key]
            obs_sigma = results['{0}_sigma'.format(key)][param_key]
            print(param_key,
                  ((obs_value - exp_value) / obs_sigma)**2,
                  ((obs_value - exp_value) / exp_value)**2)


# Long form:
fitter = mmexo.MMEXOFASTFitter(SAMPLE_FILE_01)
fitter.get_initial_ulens_params()
print(fitter.initial_ulens_params)
fitter.get_initial_phys_params()
print(fitter.initial_phys_params)
fitter.mcmc_fit()
print_metrics(fitter.results, expected)

# Short form:
fitter_2 = mmexo.MMEXOFASTFitter(SAMPLE_FILE_01)
fitter_2.fit()
print_metrics(fitter_2.results, expected)
