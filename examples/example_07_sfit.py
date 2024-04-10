"""
Example to show that the basic sfit fit works as expected.
"""
import os.path
import matplotlib.pyplot as plt

import MulensModel
import exozippy as mmexo

dir_ = os.path.join(mmexo.MULENS_DATA_PATH, "2018DataChallenge")

results_EF = {'t_0': 2460023.2844586717, 't_eff': 9.988721231519582, 'j': 2,
              'chi2': -45944.85009340058}

# Test data
lc_num = 4
file_w149 = os.path.join(
    dir_, 'n20180816.W149.WFIRST.{0:03}.txt'.format(lc_num))
file_z087 = os.path.join(
    dir_, 'n20180816.Z087.WFIRST.{0:03}.txt'.format(lc_num))
data_w149 = MulensModel.MulensData(file_name=file_w149, phot_fmt='flux')
data_z087 = MulensModel.MulensData(file_name=file_z087, phot_fmt='flux')
datasets = [data_w149, data_z087]

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
