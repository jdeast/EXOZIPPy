"""
Use mmexofast.estimate_params.WidePlanetGridSearchEstimator to find an initial solution for lc 107 from the 2018 Data
Challenge.
"""
import matplotlib.pyplot as plt
import pandas as pd

import MulensModel as mm
from examples.use_cases.DC18_classes import TestDataSet
from exozippy.mmexofast.estimate_params import WidePlanetGridSearchEstimator


def make_event(binary_params):
    model = mm.Model(binary_params.ulens)
    model.set_magnification_methods(binary_params.mag_methods)
    event = mm.Event(datasets=datasets, model=model)
    return event


def plot_event(binary_params, title=None):
    event = make_event(binary_params)
    event.plot(t_range=[2458350., 2458354.], trajectory=False, title=title)
    plt.xlim(8350, 8354.)


lc_num = 107
anomaly_params = {
    't_0': 2458345.037965581, 't_E': 5.877334611399052, 'u_0': 1.4810921099077317,
    'dmag': -0.11758628263201487, 'dt': 0.06315800035372376, 't_pl': 2458352.3170020003
}

data = TestDataSet(lc_num)
datasets = [mm.MulensData(file_name=f, phot_fmt='flux')
            for f in [data.file_w149, data.file_z087]]

estimator = WidePlanetGridSearchEstimator(datasets=datasets, params=anomaly_params)
estimator.run()
print('Best-fit:\n', estimator.best_params)

with pd.option_context('display.width', None, 'display.max_rows', None, 'display.max_columns', None):
    print('\nw\\in 3 sigma:\n', estimator.get_results_within_n_sigma(n_sigma=3))

estimator.plot_sigma_maps()
plot_event(estimator.binary_params)
plt.show()
