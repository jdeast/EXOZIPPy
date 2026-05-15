"""
Analyze a planet light curve from the 2018 Data Challenge. Minimal user effort.
"""
import os.path
import glob
import numpy as np
import exozippy
import traceback

from DC18_classes import dir_, TestDataSet


base_dir = os.path.join(
            exozippy.MODULE_PATH, 'EXOZIPPy', 'DC18Test', 'temp_output')


def fit_lc(lc_num, verbose=False):
    data = TestDataSet(lc_num)

    file_prefix = 'WFIRST.{0:03}'.format(lc_num)
    fitter = exozippy.mmexofast.mmexofast.fit(
        files=[data.file_w149, data.file_z087], coords=data.coords, fit_type='binary lens',
        verbose=verbose, renormalize_errors=False,
        log_file=os.path.join(base_dir, file_prefix + '.log'),
        restart_file=os.path.join(base_dir, file_prefix + 'pkl'),
        stop_after='fit_binary_lens:est_binary_params',
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
big_wide_planets = [4, 62]

lc_nums = wide_planets[1:2]
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
        traceback.print_exc()
