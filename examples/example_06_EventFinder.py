"""
Example to check that the EventFinderGridSearch, EFSFit, and
EFMagnificationCurve classes work as expected.
"""
import os.path
import numpy as np
import matplotlib.pyplot as plt

import MulensModel
import exozippy as mmexo

dir_ = os.path.join(mmexo.MULENS_DATA_PATH, "2018DataChallenge")

# Test data
lc_num = 4
file_w149 = os.path.join(
    dir_, 'n20180816.W149.WFIRST.{0:03}.txt'.format(lc_num))
file_z087 = os.path.join(
    dir_, 'n20180816.Z087.WFIRST.{0:03}.txt'.format(lc_num))
data_w149 = MulensModel.MulensData(file_name=file_w149, phot_fmt='flux')
data_z087 = MulensModel.MulensData(file_name=file_z087, phot_fmt='flux')
datasets = [data_w149, data_z087]

# data_w149.plot()
# data_z087.plot(color='red')
#
# # Test EFSFit
# # {'t_0': 2460025, 't_eff': 15, 'j': 1}
# ef_sfit = mmexo.gridsearches.EFSFitFunction(
#     datasets=datasets, parameters={'t_0': 2460017.5, 't_eff': 0.3, 'j': 1})
# ef_sfit.update_all()
# theta_new = ef_sfit.theta + ef_sfit.get_step()
# ef_sfit.update_all(theta=theta_new)
# print('chi2', ef_sfit.chi2, ef_sfit.parameters)
# plt.scatter(
#     ef_sfit.data[:, 0], ef_sfit.ymod, color='black', facecolor='none',
#     marker='s', s=3, zorder=3)
# plt.show()

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
#sorted = np.argsort(best_model.data[:, 0])
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
