"""
Example to show that the basic sfit fit works as expected.

Truth values:
u0        alpha       t0         tE        rE       thetaE    piE     rhos
-1.52343 38.6665 1790.31690896 6.6877 2.15354 0.35275 0.171343 0.00120717
"""
import matplotlib.pyplot as plt

import MulensModel
import exozippy as mmexo
from data_for_test_examples import datasets

results_EF = {'t_0': 2460023.2844586717, 't_eff': 9.988721231519582,
              'j': 2, 'chi2': -42226.356330652685}

fitter = mmexo.mmexofast.MMEXOFASTFitter(
    datasets=datasets, renormalize_errors=False, verbose=True)
fitter.best_ef_grid_point = results_EF
fitter.fit_point_lens()
results = fitter.all_fit_results.get(fitter._label_to_model_key('static PSPL'))

# JCY: In the new architecture, the verbose option will output the initial parameters, but they
# aren't stored, so can't be plotted.
#
#print('initial parameters', initial_pspl_params)
#init_event = MulensModel.Event(
#    datasets=datasets, model=MulensModel.Model(initial_pspl_params))
#print('initial chi2', init_event.get_chi2())
#
#
#fitter.pspl_params = initial_pspl_params
#results = fitter.do_sfit(fitter.datasets, verbose=True)
#print(results)
#
#init_event.plot(title='Initial Parameters')

event = MulensModel.Event(datasets=datasets, model=MulensModel.Model(results.params))
print(event)
print('final chi2', event.get_chi2())
event.plot(title='Final Parameters')
plt.show()
