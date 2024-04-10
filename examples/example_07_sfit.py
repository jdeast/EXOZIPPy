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

results_EF = {'t_0': 2460023.2844586717, 't_eff': 9.988721231519582, 'j': 2,
              'chi2': -45944.85009340058}

initial_pspl_params = mmexo.mmexofast.get_initial_pspl_params(
    datasets, results_EF, verbose=True)
print('initial parameters', initial_pspl_params)

init_event = MulensModel.Event(
    datasets=datasets, model=MulensModel.Model(initial_pspl_params))
print('initial chi2', init_event.get_chi2())

results = mmexo.mmexofast.do_sfit(datasets, initial_pspl_params, verbose=True)
print(results)

init_event.plot(title='Initial Parameters')

event = MulensModel.Event(datasets=datasets, model=MulensModel.Model(results))
print(event)
print('final chi2', event.get_chi2())
event.plot(title='Final Parameters')
plt.show()
