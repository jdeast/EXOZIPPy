"""
Example to check that the EventFinderGridSearch, EFSFit, and
EFMagnificationCurve classes work as expected.
"""
import numpy as np
import matplotlib.pyplot as plt

import MulensModel
import exozippy as mmexo
from data_for_test_examples import datasets, data_w149


# Do the EF search
ef = mmexo.gridsearches.EventFinderGridSearch(
    datasets=datasets, t_0_min=2459980., t_0_max=2460060.)
ef.run(verbose=True)

# Print best-fit parameters
print(ef.best)

# Grid search chi2 plot with data (x2 for j=0, 1)
plt.figure(figsize=(8, 4))
for j in [1, 2]:
    plt.subplot(1, 2, j)
    plt.title('j={0}'.format(j))
    sorted = np.argsort(ef.results[:, j-1])[::-1]
    plt.scatter(
        ef.grid_t_0[sorted], ef.grid_t_eff[sorted],
        c=ef.results[sorted, j-1],
        edgecolors='black', cmap='tab20b')
    plt.colorbar(label='chi2 - chi2_flat')
    plt.scatter(
        ef.best['t_0'], ef.best['t_eff'], color='black', marker='x', zorder=5)
    plt.minorticks_on()
    plt.xlabel('t_0')
    plt.ylabel('t_eff')
    plt.yscale('log')
    plt.tight_layout()

# Plot best-fit and best-fit for other j.
best_model = mmexo.gridsearches.EFSFitFunction(
    datasets, ef.best)
best_model.update_all()
theta_new = best_model.theta + best_model.get_step()
best_model.update_all(theta=theta_new)

plt.figure()
data_w149.plot(phot_fmt='flux')
plt.axvline(ef.best['t_0'], color='black')
plt.axvline(ef.best['t_0'] - ef.best['t_eff'], color='black', linestyle='--')
plt.axvline(ef.best['t_0'] + ef.best['t_eff'], color='black', linestyle='--')
plt.plot(
    best_model.data[best_model.data_indices[0]:best_model.data_indices[1], 0],
    best_model.ymod[best_model.data_indices[0]:best_model.data_indices[1]],
    color='black', zorder=5)
plt.xlabel('HJD')
plt.ylabel('W149 flux')
plt.show()
