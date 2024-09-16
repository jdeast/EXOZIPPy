"""
Analyze a planet light curve from the 2018 Data Challenge. Minimal user effort.
"""
import os.path
import exozippy

from DC18_classes import dir_, TestDataSet


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
