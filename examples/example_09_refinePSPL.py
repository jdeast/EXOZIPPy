"""
Refine the PSPL fit after masking the anomaly
"""
from data_for_test_examples import datasets
import exozippy as mmexo
import MulensModel
import matplotlib.pyplot as plt

pspl_params = {
    't_0': 2460023.2579352586, 't_E': 7.39674573, 'u_0': 1.4295525500937951}
af_grid_params = {'t_0': 2460017.625, 't_eff': 0.75, 'j': 2.0,
                  'chi2': 395.2493798573346, 'dchi2_zero': 5563.0445050189755,
                  'dchi2_flat': 4186.45789320002}

results = mmexo.mmexofast.refine_pspl_params(
    datasets, pspl_params, af_grid_params)
print(results)

masked_datasets = mmexo.mmexofast.get_datasets_with_anomaly_masked(
    datasets, af_grid_params)
event = MulensModel.Event(
    datasets=masked_datasets, model=MulensModel.Model(results))
event.plot(show_bad=True)
plt.show()
