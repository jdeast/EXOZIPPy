"""
Truth values:
u0        alpha       t0         tE        rE       thetaE    piE     rhos
-1.52343 38.6665 1790.31690896 6.6877 2.15354 0.35275 0.171343 0.00120717
   MP        a  inc    phase    q          s       period
0.00273006 49.2163 83.7988 270.63 0.0109202 2.48124 686.808
"""
import MulensModel
from data_for_test_examples import datasets
import exozippy
import matplotlib.pyplot as plt

pspl_params = {'t_0': 2460024.676357266, 't_E': 6.96083126,
               'u_0': 1.4639013963602456}
af_grid_params = {'t_0': 2460017.625, 't_eff': 0.75, 'j': 2.0,
                  'chi2': 395.2493798573346, 'dchi2_zero': 5563.0445050189755,
                  'dchi2_flat': 4186.45789320002}

estimator = exozippy.estimate_params.AnomalyPropertyEstimator(
    datasets=datasets, pspl_params=pspl_params, af_results=af_grid_params)
anomaly_lc_params = estimator.get_anomaly_lc_parameters()
print(anomaly_lc_params)
binary_params = exozippy.estimate_params.BinaryLensParams(ulens=None)
binary_params.set_mag_method(anomaly_lc_params)
print(binary_params.mag_methods)

solutions = exozippy.estimate_params.get_possible_bump_anomaly_solutions(anomaly_lc_params)
print(solutions)

model = MulensModel.Model(parameters=list(solutions.values())[0])
model.set_magnification_methods(binary_params.mag_methods)

event = MulensModel.Event(datasets=datasets, model=model)
print('chi2', event.get_chi2())

event.plot_data(markerfacecolor='none')
ref_fluxes = event.get_ref_fluxes()

t_range = [anomaly_lc_params['t_pl'] - 10. * anomaly_lc_params['dt'],
           anomaly_lc_params['t_pl'] + 10. * anomaly_lc_params['dt']]
for key, solution in solutions.items():
    new_model = MulensModel.Model(parameters=solution)
    new_model.set_magnification_methods(binary_params.mag_methods)
    new_model.plot_lc(
        t_range=t_range, phot_fmt='mag', source_flux=ref_fluxes[0],
        blend_flux=ref_fluxes[1], label=key)


plt.xlim(t_range)
plt.gca().minorticks_on()
plt.show()