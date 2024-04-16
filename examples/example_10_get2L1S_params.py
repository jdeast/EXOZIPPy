"""
Truth values:
u0        alpha       t0         tE        rE       thetaE    piE     rhos
-1.52343 38.6665 1790.31690896 6.6877 2.15354 0.35275 0.171343 0.00120717
   MP        a  inc    phase    q          s       period
0.00273006 49.2163 83.7988 270.63 0.0109202 2.48124 686.808
"""

from data_for_test_examples import datasets
import exozippy as mmexo
import MulensModel
import matplotlib.pyplot as plt

pspl_params = {'t_0': 2460024.676357266, 't_E': 6.96083126,
               'u_0': 1.4639013963602456}
af_grid_params = {'t_0': 2460017.625, 't_eff': 0.75, 'j': 2.0,
                  'chi2': 395.2493798573346, 'dchi2_zero': 5563.0445050189755,
                  'dchi2_flat': 4186.45789320002}

binary_params = mmexo.mmexofast.get_initial_2L1S_params(
    datasets, pspl_params, af_grid_params)
print(binary_params.ulens)
print(binary_params.mag_method)

binary_model = MulensModel.Model(binary_params.ulens)
binary_model.set_magnification_methods(binary_params.mag_method)

event = MulensModel.Event(datasets=datasets, model=binary_model)
event.plot()
plt.show()
