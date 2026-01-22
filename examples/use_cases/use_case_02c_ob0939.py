"""
Use case to show stopping and restarting a fit.
"""
import exozippy
import os.path

ground_data_files = [os.path.join(
        exozippy.MULENS_DATA_PATH, 'OB140939', 'n20100310.I.OGLE.OB140939.txt')]
space_data_files = [os.path.join(
        exozippy.MULENS_DATA_PATH, 'OB140939', 'n20100310.I.OGLE.OB140939.txt')]
coords='17:47:12.25 -21:22:58.7'

raw_fitter = exozippy.mmexofast.MMEXOFASTFitter(
    files=ground_data_files, coords=coords, fit_type='point lens', renormalize_errors=False,
    output_file='test_output/OB0939_raw_results.txt')
raw_fitter.fit()

cont_fitter = exozippy.mmexofast.MMEXOFASTFitter(
    files=ground_data_files, coords=coords, fit_type='point lens',
    prev_results='test_output/OB0939_raw_results.txt')
cont_fitter.fit()

sponly_fitter = exozippy.mmexofast.MMEXOFASTFitter(
    files=space_data_files, coords=coords, fit_type='point lens',
    parameters_to_fit=['pi_E_E', 'pi_E_N'],  # Should this specify fixed parameters instead?
    piE_grid=True,
    prev_results='test_output/OB09393_raw_results.txt'
)
sponly_fitter.fit()
