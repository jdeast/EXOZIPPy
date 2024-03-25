"""
Example 02: Show how MM-EXOFAST should work on OB140939, a point lens
microlensing event with satellite parallax.

Uses MulensModel.MulensData objects as input.

https://ui.adsabs.harvard.edu/abs/2015ApJ...802...76Y/abstract
"""

import os.path
import MulensModel as mm
import mmexofast as mmexo

from print_ex_metrics import print_metrics


# Read the data (note that we do not rescale errorbars here):
# File Formatting convention: nYYYYMMDD.filtername.telescope.whateveryouwant.
# YYYYMMDD = UT date of first datapoint
dir_ = os.path.join(mm.DATA_PATH, "photometry_files", "OB140939")
file_ground = os.path.join(dir_, "ob140939_OGLE.dat")
file_spitzer = os.path.join(dir_, "ob140939_Spitzer.dat")
data_ground = mm.MulensData(
    file_name=file_ground, plot_properties={'label': 'OGLE'})

# Here is the main difference - we provide the ephemeris for Spitzer:
file_spitzer_eph = os.path.join(
    mm.DATA_PATH, 'ephemeris_files', 'Spitzer_ephemeris_01.dat')
data_spitzer = mm.MulensData(
    file_name=file_spitzer, ephemerides_file=file_spitzer_eph,
    plot_properties={'label': 'Spitzer'})

# For parallax calculations we need event coordinates:
coords = "17:47:12.25 -21:22:58.7"

# 4 degenerate solutions
expected_mp = {'mulens': {'t_0': [2456836.22, 0.11], 'u_0': [0.840, 0.002], 't_E': [24.29, 0.16],
                          'pi_E_N': [-0.214, 0.044], 'pi_E_E': [0.217, 0.006]},
               'physical': {'v_tilde_hel':
                            {'N': [-164.9, 4.8], 'E': [195.5, 34.2]}}
               }  # X2 = 273.6
expected_mm = {'mulens': {'t_0': [2456836.20, 0.11], 'u_0': [-0.840, 0.002], 't_E': [24.27, 0.16],
                          'pi_E_N': [0.192, 0.043], 'pi_E_E': [0.222, 0.008]},
               'physical': {'v_tilde_hel':
                            {'N': [158.3, 4.7], 'E': [212.4, 36.3]}}
               }  # X2 = 274.1
expected_pp = {'mulens': {'t_0': [2456836.07, 0.10], 'u_0': [0.840, 0.002], 't_E': [23.92, 0.15],
                          'pi_E_N': [-1.292, 0.029], 'pi_E_E': [-0.052, 0.018]},
               'physical': {'v_tilde_hel':
                            {'N': [-56.4, 1.3], 'E': [26.7, 0.7]}}
               }  # X2 = 281.8
expected_pm = {'mulens': {'t_0': [2456835.95, 0.11], 'u_0': [-0.840, 0.002], 't_E': [23.93, 0.15],
                          'pi_E_N': [1.321, 0.029], 'pi_E_E': [0.024, 0.033]},
               'physical': {'v_tilde_hel':
                            {'N': [54.3, 1.3], 'E': [29.9, 0.8]}}
               }  # X2 = 290.2

expected_phys = {'physical': {'pi_rel': [0.20, 0.04], 'D_L': [3.1, 0.4],
                              'M_L': [0.23, 0.07]}}
# 1. These physical parameters are only valid for the u0_-,+- solutions.
# 2. Propagation of errors in the paper for the final, physical parameters is
# complex, so our results may not match perfectly.

v_earth_perp = [-0.5, 28.9]  # [N, E] km/s
# This value could be tested.

priors = {'mu_s_hel': {'N': ['gauss', -0.64, 0.45],
                       'E': ['gauss', -5.31, 0.45]}}

results = mmexo.mmexofast.fit(
    data_sets=[data_ground, data_spitzer], fit_type='point lens',
    print_results=True, verbose=True, priors=priors)

# This solution matching will not necessarily work if the solutions aren't
# close enough.
for solution in results.solutions:
    if solution.mulens.u_0 < 0:
        if solution.mulens.pi_E_N > 1:
            print_metrics(solution, expected_pm)
        else:
            print_metrics(solution, expected_mm)

    else:
        if solution.mulens.pi_E_N < -1:
            print_metrics(solution, expected_pp)
        else:
            print_metrics(solution, expected_mp)

print_metrics(results.final_parameters, expected_phys)
