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

#residuals = mmexo.mmexofast.get_residuals(datasets, pspl_params)
#print('max flux', [(np.max(residual.flux)) for residual in residuals])


def plot_EFSFitFunction(datasets, params, verbose=False):
    model = mmexo.gridsearches.EFSFitFunction(
        datasets, params)
    model.update_all()
    theta_new = model.theta + model.get_step()
    model.update_all(theta=theta_new)
    if verbose:
        print('chi2', model.chi2)

    plt.errorbar(
        datasets[0].time, datasets[0].flux, yerr=datasets[0].err_flux,
        fmt='o')

    plt.axhline(0, color='black', linestyle='--')
    plt.plot(
        model.data[model.data_indices[0]:model.data_indices[1], 0],
        model.ymod[model.data_indices[0]:model.data_indices[1]],
        color='black', zorder=5)
    plt.xlabel('HJD')
    plt.ylabel('W149 flux')
    plt.minorticks_on()
    plt.tight_layout()


# Test Grid Search
fitter = mmexo.mmexofast.MMEXOFASTFitter(datasets=datasets)
fitter.pspl_params = pspl_params
fitter.set_residuals()
af = mmexo.gridsearches.AnomalyFinderGridSearch(
    residuals=fitter.residuals, t_0_min=t_start, t_0_max=t_stop)

# Test Single Fit
plt.figure()
plt.title('Test Single element')
for test_params in [
    {'t_0': 2460017.56, 't_eff': 1., 'j': 1},
    {'t_0': 2460028.2788192877, 't_eff': 9.988721231519582, 'j': 1}]:
    print(test_params)
    trimmed_residuals = af.get_trimmed_datasets(test_params)
    print('chi2_zero', np.sum(np.hstack(
        [(dataset.flux /dataset.err_flux)**2 for dataset in trimmed_residuals]))
          )
    plot_EFSFitFunction(trimmed_residuals, test_params, verbose=True)

#plt.show()

# Run Grid Search
af.run(verbose=True)
# Print best-fit parameters
print('Best:')
print(af.best)
print('# of anomalies', len(af.anomalies))

# Grid search chi2 plot with data
chi2_range = 1000
labels = ['1', '2', 'flat', 'zero']
for j in range(4):
    sorted = np.argsort(af.results[:, j])[::-1]
    plt.figure()
    plt.scatter(
        af.grid_t_0[sorted], af.grid_t_eff[sorted],
        c=af.results[sorted, j],
        edgecolors='black', cmap='tab20b')
    plt.colorbar(label='chi2_{0}'.format(labels[j]))
    plt.scatter(
        af.best['t_0'], af.best['t_eff'], color='black', marker='x',
        zorder=10, s=100)
    plt.title('chi2_{0}'.format(labels[j]))
    plt.minorticks_on()
    plt.xlabel('t_0')
    plt.ylabel('t_eff')
    plt.yscale('log')
    plt.tight_layout()

plt.figure(figsize=(8, 4))
for j in [1, 2]:
    plt.subplot(1, 2, j)
    plt.title('j={0}'.format(j))
    dchi2_zero = af.results[:, 3] - af.results[:, j-1]
    sorted = np.argsort(dchi2_zero)
    plt.scatter(
        af.grid_t_0[sorted], af.grid_t_eff[sorted],
        c=dchi2_zero[sorted],
        edgecolors='black', cmap='tab20b', vmin=0)
    plt.colorbar(label='chi2_zero - chi2 ')
    plt.scatter(
        af.best['t_0'], af.best['t_eff'], color='black', marker='x', zorder=10)
    index = (af.anomalies[:, 2] == j)
    plt.minorticks_on()
    plt.xlabel('t_0')
    plt.ylabel('t_eff')
    plt.yscale('log')
    plt.tight_layout()


plt.figure()
plt.title('Event')
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
plt.xlabel('HJD')
plt.ylabel('W149 flux')
plt.minorticks_on()
plt.tight_layout()

plt.figure()
plt.title('Residuals')
plot_EFSFitFunction(fitter.residuals, af.best)

plt.show()



