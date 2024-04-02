"""
Example to check that the EventFinderGridSearch, EFModel, and
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
# Do the EF search
ef = mmexo.gridsearches.EventFinderGridSearch(
    datasets=datasets, t_0_min=2459980., t_0_max=2460060.)
ef.run(verbose=True)

# Print best-fit parameters
print(ef.best)

# Grid search chi2 plot with data (x2 for j=0, 1)
plt.figure()
for j in [1, 2]:
    plt.subplot(1, 2, j)
    plt.title('j={0}'.format(j))
    sorted = np.argsort(ef.results[:, j-1])[::-1]
    plt.scatter(
        ef.grid_t_0[sorted], ef.grid_t_eff[sorted],
        c=ef.results[sorted, j-1]-ef.best['chi2'],
        edgecolors='black', vmin=0, vmax=100, cmap='tab20b')
    plt.colorbar(label='Delta chi2')
    plt.scatter(
        ef.best['t_0'], ef.best['t_eff'], color='black', marker='x', zorder=5)
    plt.minorticks_on()
    plt.xlabel('t_0')
    plt.ylabel('t_eff')

# Plot best-fit and best-fit for other j.
best_model = mmexo.gridsearches.EFSFitFunction(data_w149, ef.best)
best_model.update_all()

plt.figure()
data_w149.plot(phot_fmt='flux')
sorted = np.argsort(best_model.data[:, 0])
plt.axvline(ef.best['t_0'], color='black')
plt.axvline(ef.best['t_0'] - ef.best['t_eff'], color='black', linestyle='--')
plt.axvline(ef.best['t_0'] + ef.best['t_eff'], color='black', linestyle='--')
plt.plot(
    best_model.data[sorted, 0], best_model.ymod[sorted], color='black')

plt.show()
