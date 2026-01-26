"""
Analyze the ground-based data for microlensing event OB140939
"""
import exozippy
import os.path
import matplotlib.pyplot as plt


data_file = os.path.join(
    exozippy.MULENS_DATA_PATH, 'OB140939', 'n20100310.I.OGLE.OB140939.txt')

fitter = exozippy.mmexofast.fit(
    files=[data_file], fit_type='point lens', renormalize_errors=False, coords='17:47:12.25 -21:22:58.7',
    verbose=True, log_file='test_output/test_ob0939_raw.log', latex_file='test_output/test_ob09393_raw.tex')
#fitter = exozippy.mmexofast.fit(
#    files=[data_file], fit_type='point lens', coords='17:47:12.25 -21:22:58.7',
#    verbose=True, log_file='test_output/test_ob0939.log', latex_file='test_output/test_ob09393.tex')
print(fitter.initialize_exozippy())

"""
Desired outputs:

Microlensing:
- Static PSPL model parameters & uncertainties, chi2
- Parallax PSPL model parameters & uncertainties, chi2
- Parallax contour plot

EXOZIPPy:
- Best parallax PSPL model parameters & uncertainties, chi2
- Posteriors for physical parameters

Note: all chi2s need to be on the same system.
"""

#fitter.print_static_pspl_results()
#fitter.print_parallax_pspl_results()
#fitter.fit_parallax_grid(plot=True)
#fitter.print_final_model()
#fitter.plot_posteriors()

plt.show()
