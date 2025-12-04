"""
Analyze the ground-based data for microlensing event OB161045
"""
import exozippy
import os.path
import matplotlib.pyplot as plt
import glob


data_files = glob.glob(
    os.path.join(
        exozippy.MULENS_DATA_PATH, 'OB161045', 'n20*OB161045.txt'))

fitter = exozippy.mmexofast.MMEXOFASTFitter(
    files=data_files, fit_type='point lens', coords='17:36:51.18 -34:32:39.77',
    finite_source=True, limb_darkening_coeffs_gamma={'I': 0.5103, 'R': 0.6583},
    mag_methods=[2457558.5, 'finite_source_LD_Yoo04', 2457559.7],
    verbose=True)

for dataset in fitter.datasets:
    print(dataset)

fitter.fit()
print(fitter.initialize_exozippy())

#fitter.output_latex_table()  # make a LateX table file summarizing the results of all fits (PSPL, FSPL, FSPL+Par, EXOZIPPy)

