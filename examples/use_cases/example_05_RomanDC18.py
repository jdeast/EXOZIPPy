"""
Analyze a planet light curve from the 2018 Data Challenge. Minimal user effort.
"""
import os.path
import glob
import numpy as np
import exozippy

from DC18_classes import dir_, TestDataSet


base_dir = os.path.join(
            exozippy.MODULE_PATH, 'EXOZIPPy', 'DC18Test', 'temp_output')


def fit_lc(lc_num, verbose=False):
    data = TestDataSet(lc_num)

    fitter = exozippy.mmexofast.mmexofast.fit(
        files=[data.file_w149, data.file_z087], coords=data.coords, fit_type='binary lens',
        verbose=verbose,
        # print_results=True, emcee=False, emcee_settings = {'n_walkers': 20, 'n_burn': 50, 'n_steps': 100},
        stop_before='emcee',
        output_config=exozippy.mmexofast.OutputConfig(
            base_dir=base_dir, file_head='WFIRST.{0:03}'.format(lc_num), save_log=True,
            save_latex_tables=True, save_restart_files=True)
    )

    return fitter.all_fit_results


def evaluate_results(lc_num):
    """
    Calculate metrics between input and output values
    Assume pymc output.
    """
    pass


files = glob.glob(os.path.join(dir_, 'n2018*.W149.*.txt'))
lc_nums = []
for file_ in files:
    elements = file_.split('.')
    lc_nums.append(int(elements[-2]))

# lc_nums = [4]  # favorite test case 004
# lcs for wide planets:
wide_planets = [8, 53, 107, 131, 152, 194, 208, 214, 217, 226]

lc_nums = wide_planets
for lc_num in np.sort(lc_nums):
    print('\n...Fitting light curve {0}...'.format(lc_num))
    try:
        results = fit_lc(lc_num, verbose=True)
        print(results)
        evaluate_results(lc_num)
    except NotImplementedError:
        pass
    except Exception as e:
        print('Run {0} ABORTED. {1}: {2}'.format(lc_num, type(e).__name__, e))
