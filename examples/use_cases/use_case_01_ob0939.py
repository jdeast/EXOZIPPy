"""
Analyze the ground-based data for microlensing event OB140939
"""
import exozippy
import os.path

data_file = os.path.join(
    exozippy.MULENS_DATA_PATH, 'OB140939', 'n20100310.I.OGLE.OB140939.txt')

fitter = exozippy.mmexofast.MMEXOFASTFitter(
    files=[data_file], fit_type='point lens', coords='17:47:12.25 -21:22:58.7',
    verbose=True)
fitter.fit()
