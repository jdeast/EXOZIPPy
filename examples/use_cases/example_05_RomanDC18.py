"""
Analyze a planet light curve from the 2018 Data Challenge. Minimal user effort.
"""
import os.path
import glob
import numpy as np
import exozippy

from DC18_classes import dir_, TestDataSet


def fit_lc(lc_num, verbose=False):
    data = TestDataSet(lc_num)

    results = exozippy.mmexofast.fit(
        files=[data.file_w149, data.file_z087], coords=data.coords, fit_type='binary lens',
        print_results=True, verbose=verbose, emcee=False, #emcee_settings = {'n_walkers': 20, 'n_burn': 50, 'n_steps': 100},
        log_file=os.path.join(
            exozippy.MODULE_PATH, 'EXOZIPPy', 'DC18Test', 'temp_output', 'WFIRST.{0:03}.log'.format(lc_num))
    )

    return results


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

for lc_num in np.sort(lc_nums):
    print('\n...Fitting light curve {0}...'.format(lc_num))
    try:
        results = fit_lc(lc_num, verbose=True)
        evaluate_results(lc_num)
    except NotImplementedError:
        pass
    except Exception as e:
        print('Run {0} ABORTED. {1}: {2}'.format(lc_num, type(e).__name__, e))
