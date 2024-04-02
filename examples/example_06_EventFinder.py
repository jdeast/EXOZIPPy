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

# Do the EF search
ef = mmexo.gridsearches.EventFinderGridSearch(datasets=[data_w149, data_z087])
ef.run(verbose=True)

# Print best-fit parameters
print(ef.best)

# Grid search chi2 plot with data (x2 for j=0, 1)
plt.figure()
for j in [1, 2]:
    plt.subplot(2, 1, 1)
    plt.title('j={0}'.format(j))
    plt.scatter(
        ef.grid_t_0, ef.grid_t_eff, c=ef.results[:, j-1], edgecolors='black')
    plt.minorticks_on()
    plt.xlabel('t_0')
    plt.ylabel('t_eff')

# Plot best-fit and best-fit for other j.
best_model = mmexo.gridsearches.EFModel(ef.best)
fit = MulensModel.FitData(dataset=data_w149, model=best_model)
fit.fit_fluxes()
plt.figure()
data_w149.plot(phot_fmt='flux')
best_model.plot_lc(t_range=[np.min(data_w149.time), np.max(data_w149.time)],
    phot_fmt='mag', source_flux=fit.source_flux, blend_flux=fit.blend_flux,
    color='black', lw=3)

plt.show()
