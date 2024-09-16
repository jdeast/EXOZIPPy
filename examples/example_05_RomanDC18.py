"""
Analyze a planet light curve from the 2018 Data Challenge. Minimal user effort.
"""
import os.path
import exozippy
import numpy as np

dir_ = os.path.join(exozippy.MULENS_DATA_PATH, "2018DataChallenge")

event_info = np.genfromtxt(
    os.path.join(dir_, 'event_info.txt'), dtype=None, encoding='utf-8',
    names=['file', 'num', 'ra', 'dec'], usecols=range(4))


class TestDataSet():

    def __init__(self, lc_num):
        self.file_w149 = os.path.join(
            dir_, 'n20180816.W149.WFIRST18.{0:03}.txt'.format(lc_num))
        self.file_z087 = os.path.join(
            dir_, 'n20180816.Z087.WFIRST18.{0:03}.txt'.format(lc_num))

        index = np.where(event_info['num'] == lc_num)
        self.coords = '{0} {1}'.format(
            event_info['ra'][index][0], event_info['dec'][index][0])


def fit_lc(lc_num, verbose=False):
    data = TestDataSet(lc_num)

    results = exozippy.mmexofast.fit(
        files=[data.file_w149, data.file_z087], coords=data.coords, fit_type='binary lens',
        print_results=True, verbose=verbose,
        output_file=os.path.join(
            dir_, 'temp_output', 'WFIRST.{0:03f}.csv'.format(lc_num))
    )

    return results


def evaluate_results(lc_num):
    """Calculate metrics between input and output values"""
    pass


if __name__ == '__main__':
    for lc_num in [1, 4, 8]:
        print('\n...Fitting light curve {0}...'.format(lc_num))
        try:
            results = fit_lc(lc_num, verbose=True)
            evaluate_results(lc_num)
        except NotImplementedError:
            pass
        except Exception as e:
            print('Run {0} ABORTED. {1}: {2}'.format(lc_num, type(e).__name__, e))
