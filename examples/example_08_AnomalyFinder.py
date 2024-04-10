"""
Example to show that the AnomalyFinder works as expected.
"""
import numpy as np
import matplotlib.pyplot as plt

import MulensModel
import exozippy as mmexo
from data_for_test_examples import datasets

pspl_params = {
    't_0': 2460023.2579352586, 't_E': 7.39674573, 'u_0': 1.4295525500937951}

residuals = mmexo.mmexofast.get_residuals(datasets, pspl_params)
af = mmexo.gridsearches.AnomalyFinderGridSearch(residuals=residuals)
af.run(verbose=True)

# Print best-fit parameters
print(af.best)

# Grid search chi2 plot with data
sorted = np.argsort(af.results[:])[::-1]
plt.scatter(
    af.grid_t_0[sorted], af.grid_t_eff[sorted],
    c=af.results[sorted],
    edgecolors='black', cmap='tab20b')
plt.colorbar(label='chi2 - chi2_flat')
plt.scatter(
    af.best['t_0'], af.best['t_eff'], color='black', marker='x', zorder=5)
plt.scatter(
    af.anomalies[:, 0], af.anomalies[:, 1],
    color='red', facecolor='none', marker='s', zorder=5)
plt.minorticks_on()
plt.xlabel('t_0')
plt.ylabel('t_eff')
plt.yscale('log')
plt.tight_layout()

best_model = mmexo.gridsearches.EFSFitFunction(datasets, af.best)
best_model.update_all()
theta_new = best_model.theta + best_model.get_step()
best_model.update_all(theta=theta_new)

plt.figure()
event = MulensModel.Event(
    datasets=datasets, model=MulensModel.Model(pspl_params))
event.plot_data(phot_fmt='flux')
event.plot_model(phot_fmt='flux')
plt.axvline(af.best['t_0'], color='black')
plt.axvline(af.best['t_0'] - af.best['t_eff'], color='black', linestyle='--')
plt.axvline(af.best['t_0'] + af.best['t_eff'], color='black', linestyle='--')
plt.plot(
    best_model.data[best_model.data_indices[0]:best_model.data_indices[1], 0],
    best_model.ymod[best_model.data_indices[0]:best_model.data_indices[1]],
    color='black', zorder=5)
plt.xlabel('HJD')
plt.ylabel('W149 flux')
plt.show()



