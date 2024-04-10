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

t_start = 2459980.
t_stop = 2460060.

residuals = mmexo.mmexofast.get_residuals(datasets, pspl_params)
af = mmexo.gridsearches.AnomalyFinderGridSearch(
    residuals=residuals, t_0_min=t_start, t_0_max=t_stop)
af.run(verbose=True)

# Print best-fit parameters
print(af.best)
print(af.results.shape)

# Grid search chi2 plot with data

labels = ['1', '2', 'flat', 'zero']
for j in range(4):
    sorted = np.argsort(af.results[:, j])
    plt.figure()
    plt.scatter(
        af.grid_t_0[sorted], af.grid_t_eff[sorted],
        c=af.results[sorted, j],
        edgecolors='black', cmap='tab20b')
    plt.title('chi2_{0}'.format(labels[j]))
    plt.colorbar(label='chi2_{0}'.format(labels[j]))
    plt.minorticks_on()
    plt.xlabel('t_0')
    plt.ylabel('t_eff')
    plt.yscale('log')
    plt.tight_layout()

plt.figure(figsize=(8, 4))
for j in [1, 2]:
    plt.subplot(1, 2, j)
    plt.title('j={0}'.format(j))
    sorted = np.argsort(af.results[:, 3] - af.results[:, j-1])[::-1]

    plt.scatter(
        af.grid_t_0[sorted], af.grid_t_eff[sorted],
        c=af.results[sorted, 3] - af.results[sorted, j-1],
        edgecolors='black', cmap='tab20b')
    plt.colorbar(label='chi2 - chi2_zero')
    plt.scatter(
        af.best['t_0'], af.best['t_eff'], color='black', marker='x', zorder=10)
    index = af.anomalies[:, 2] == j
    #plt.scatter(
    #    af.anomalies[index, 0], af.anomalies[index, 1], color='red',
    #    facecolor='none', marker='s', zorder=5)
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
plt.gca().invert_yaxis()

times = np.linspace(t_start, t_stop, 1000)
ref_fluxes = event.get_ref_fluxes()
print(ref_fluxes)
model_fluxes = (ref_fluxes[0][0] * event.model.get_magnification(times) +
                ref_fluxes[1])
print(model_fluxes.shape)
plt.plot(times, model_fluxes, color='black', zorder=5)

plt.axvline(af.best['t_0'], color='black')
plt.axvline(af.best['t_0'] - af.best['t_eff'], color='black', linestyle='--')
plt.axvline(af.best['t_0'] + af.best['t_eff'], color='black', linestyle='--')
#plt.plot(
#    best_model.data[best_model.data_indices[0]:best_model.data_indices[1], 0],
#    best_model.ymod[best_model.data_indices[0]:best_model.data_indices[1]],
#    color='black', zorder=5)
plt.xlabel('HJD')
plt.ylabel('W149 flux')
plt.show()



