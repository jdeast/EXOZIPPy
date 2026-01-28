"""
Use case to show stopping and restarting a fit.
"""
import exozippy
import os.path
from pathlib import Path

from exozippy.mmexofast import OutputConfig

ground_data_files = [os.path.join(
        exozippy.MULENS_DATA_PATH, 'OB140939', 'n20100310.I.OGLE.OB140939.txt')]
space_data_files = [os.path.join(
        exozippy.MULENS_DATA_PATH, 'OB140939', 'n20100310.I.OGLE.OB140939.txt')]
coords='17:47:12.25 -21:22:58.7'

base_dir = Path('test_output')

# print('=== Fit raw data ===')
# raw_fitter = exozippy.mmexofast.MMEXOFASTFitter(
#     files=ground_data_files, coords=coords, fit_type='point lens', renormalize_errors=False,
#     verbose=True,
#     output_config=OutputConfig(
#         base_dir=base_dir, file_head='ob0939_uc02c_raw', save_log=True, save_plots=True,
#         save_latex_tables=True, save_restart_files=True))
# raw_fitter.fit()
# ------
# Expected workflow: fit_point_lens (incl. 2 parallax fits)
# output:
# A log file: ob0939_uc02c_raw.log [DONE]
#    with results at each stage of the fitting
#
# diagnostic plots: ob0939_uc02c_ef_grid.png [DONE]
#
# latex table: ob0939_uc02c_raw_results.tex [DONE]
#
# restart_files: ob0939_uc02c_raw_restart.pkl [DONE]
#    containing everything needed to initialize the next step (below)
# ------

print('=== Restart from pickle and Fit w/Error Renorm ===')
cont_fitter = exozippy.mmexofast.MMEXOFASTFitter(
    restart_file='test_output/ob0939_uc02c_raw_restart.pkl',  # change "gr" back to "raw" later.
    renormalize_errors=True, parallax_grid=True, verbose=True,
    output_config=OutputConfig(
        base_dir=base_dir, file_head='ob0939_uc02c_gr', save_log=True, save_plots=True,
        save_latex_tables=True, save_restart_files=True, save_grid_results=True))
cont_fitter.fit()
# ------
# Expected workflow: Renormalize errors, refit all models, run parallax grids
#
# Need to implement:
# 1. [DONE] Error Renormalization and outlier removal
# 2. [DONE] Refitting
# 3. Parallax grids
#
# output:
# A log file: ob0939_uc02c_gr.log
#    with results at each stage of the fitting
#    [TODO] Include fitted fluxes and full event/model info after major fits.
#
# diagnostic plots: ob0939_uc02c_gr_par_u0[p/m].png
#
# piE grids: ob0939_uc02c_gr_par_grid_u0[p/m].txt
#
# latex table: ob0939_uc02c_gr_results.tex
#
# restart_files: ob0939_uc02c_gr_restart.pkl
#    containing everything needed to initialize the next step
# ------

# complete_fitter = exozippy.mmexofast.fit(
#     files=ground_data_files + space_data_files, coords=coords, fit_type='point lens',
#     parallax_grid=True,
#     restart_file='test_output/ob0939_uc02c_gr_restart.txt',
#     output_config=OutputConfig(
#         base_dir=base_dir, file_head='ob0939_uc02c_full', save_log=True, save_plots=True,
#         save_latex_tables=True, save_restart_files=True, save_grid_results=True)
#     )
#
# fixed_fb_fitter = exozippy.mmexofast.fit(
#     files=ground_data_files + space_data_files, coords=coords, fit_type='point lens',
#     parallax_grid=True,
#     fix_blend_flux={ground_data_files[0]: True},
#     restart_file='test_output/ob0939_uc02c_full_restart.txt',
#     output_config=OutputConfig(
#         base_dir=base_dir, file_head='ob0939_uc02c_fb0', save_log=True, save_plots=True,
#         save_latex_tables=True, save_restart_files=True, save_grid_results=True)
#     )
# 
# sponly_fitter = exozippy.mmexofast.MMEXOFASTFitter(
#     files=space_data_files, coords=coords, fit_type='space-only piE_grid',
#     restart_file='test_output/ob0939_uc02c_full_restart.txt',
#     output_config=OutputConfig(
#         base_dir=base_dir, file_head='ob0939_uc02c_sponly', save_log=True, save_plots=True,
#         save_latex_tables=True, save_restart_files=True, save_grid_results=True))
# sponly_fitter.fit()
# 
