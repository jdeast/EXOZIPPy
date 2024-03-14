"""
Example 01: Show how MM-EXOFAST should work on an FSPL microlensing event.
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
init_ulens_params = mmexo.get_initial_ulens_params(SAMPLE_FILE_01)
init_phys_params = mmexo.get_phys_params(init_ulens_params)
results = mmexo.mcmc_fit(SAMPLE_FILE_01, init_params=init_phys_params)
print_metrics(results, expected)

# Short form:
results_short = mmexo.fit(SAMPLE_FILE_01)
print_metrics(results_short, expected)
