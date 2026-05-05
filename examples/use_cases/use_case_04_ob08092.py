"""
Use case to show stopping and restarting a fit.
"""
import exozippy
import os.path
from pathlib import Path

from exozippy.mmexofast import OutputConfig

ground_data_files = [os.path.join(
        exozippy.MULENS_DATA_PATH, 'OB08092', 'n20100309.I.OGLE.OB08092.txt')]
coords='17:47:29.42 -34:43:35.6'

base_dir = Path('test_output')

print('=== Fit raw data ===')
raw_fitter = exozippy.mmexofast.MMEXOFASTFitter(
    files=ground_data_files, coords=coords, fit_type='point lens', renormalize_errors=True,
    verbose=True,
    parallax_grid=True,
    output_config=OutputConfig(
        base_dir=base_dir, file_head='ob08092', save_log=True, save_plots=True,
        save_latex_tables=True, save_restart_files=True))
raw_fitter.fit()
print('initialize_exozippy:\n', raw_fitter.initialize_exozippy())
