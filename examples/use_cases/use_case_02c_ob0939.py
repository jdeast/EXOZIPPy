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
        exozippy.MULENS_DATA_PATH, 'OB140939', 'n20140605.L.Spitzer.OB140939.txt')]
coords='17:47:12.25 -21:22:58.7'

base_dir = Path('test_output')

print('=== Fit raw data ===')
raw_fitter = exozippy.mmexofast.MMEXOFASTFitter(
    files=ground_data_files, coords=coords, fit_type='point lens', renormalize_errors=False,
    verbose=True,
    output_config=OutputConfig(
        base_dir=base_dir, file_head='ob0939_uc02c_raw', save_log=True, save_plots=True,
        save_latex_tables=True, save_restart_files=True))
raw_fitter.fit()
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
    restart_file='test_output/ob0939_uc02c_raw_restart.pkl',
    renormalize_errors=True, verbose=True,
    #parallax_grid=True,
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
# 3. [DONE] Parallax grids
#
# output:
# A log file: ob0939_uc02c_gr.log [DONE]
#    with results at each stage of the fitting
#    [DONE] Include fitted fluxes and full event/model info after major fits.
#
# diagnostic plots: ob0939_uc02c_gr_piE_grid.png [DONE]
#
# piE grids: ob0939_uc02c_gr_par_grid_u0_[PLUS/MINUS].txt [DONE]
#
# latex table: ob0939_uc02c_gr_results.tex [DONE]
#
# restart_files: ob0939_uc02c_gr_restart.pkl [DONE]
#    containing everything needed to initialize the next step
# ------

print('=== Restart from pickle and ADD Spitzer Data ===')
complete_fitter = exozippy.mmexofast.fit(
    files=ground_data_files + space_data_files,
    parallax_grid=True,
    renormalize_errors=True,
    verbose=True,
    restart_file='test_output/ob0939_uc02c_gr_restart.pkl',
    output_config=OutputConfig(
        base_dir=base_dir, file_head='ob0939_uc02c_complete', save_log=True, save_plots=True,
        save_latex_tables=True, save_restart_files=True, save_grid_results=True)
    )
complete_fitter.fit()
# ------
# Need to implement:
# 1. Adding datafiles with a pickle
# 2. Spitzer in observatories, incl. ephemerides file.
# ------
"""
died here:

Optimizing from u_0+ grid point: pi_E_E=-0.690, pi_E_N=1.200
Optimized: chi2=228028.08, {'t_0': 2456836.163014465, 't_E': 209.43827447995199, 'u_0': 0.055728482821769385, 'pi_E_E': -2.0399809873892885, 'pi_E_N': 4.279236693634176, 'chi2': 228028.07703645638}
Traceback (most recent call last):
  File "/Users/jyee/PycharmProjects/EXOZIPPy/examples/use_cases/use_case_02c_ob0939.py", line 73, in <module>
    complete_fitter = exozippy.mmexofast.fit(
  File "/Users/jyee/PycharmProjects/EXOZIPPy/exozippy/mmexofast/mmexofast.py", line 50, in fit
    fitter.fit()
  File "/Users/jyee/PycharmProjects/EXOZIPPy/exozippy/mmexofast/mmexofast.py", line 986, in fit
    self.fit_point_lens()
  File "/Users/jyee/PycharmProjects/EXOZIPPy/exozippy/mmexofast/mmexofast.py", line 1209, in fit_point_lens
    comprehensive_parallax_fitting(initial_grids)
  File "/Users/jyee/PycharmProjects/EXOZIPPy/exozippy/mmexofast/mmexofast.py", line 1175, in comprehensive_parallax_fitting
    self._extract_and_optimize_parallax_solutions(
  File "/Users/jyee/PycharmProjects/EXOZIPPy/exozippy/mmexofast/mmexofast.py", line 2201, in _extract_and_optimize_parallax_solutions
    secondary_sign = self._get_space_u0_sign(fit_result, secondary_ephem)
  File "/Users/jyee/PycharmProjects/EXOZIPPy/exozippy/mmexofast/mmexofast.py", line 2033, in _get_space_u0_sign
    result = minimize_scalar(
  File "/opt/miniconda3/envs/MMpyTorch/lib/python3.10/site-packages/scipy/optimize/_minimize.py", line 945, in minimize_scalar
    res = _minimize_scalar_bounded(fun, bounds, args, **options)
  File "/opt/miniconda3/envs/MMpyTorch/lib/python3.10/site-packages/scipy/optimize/_optimize.py", line 2426, in _minimize_scalar_bounded
    fu = func(x, *args)
  File "/Users/jyee/PycharmProjects/EXOZIPPy/exozippy/mmexofast/mmexofast.py", line 2026, in u_squared
    trajectory = model.get_trajectory([time])
  File "/Users/jyee/PycharmProjects/MulensModel/source/MulensModel/model.py", line 724, in get_trajectory
    satellite_skycoord = self.get_satellite_coords(times)
  File "/Users/jyee/PycharmProjects/MulensModel/source/MulensModel/model.py", line 1253, in get_satellite_coords
    return satellite_skycoords.get_satellite_coords(times)
  File "/Users/jyee/PycharmProjects/MulensModel/source/MulensModel/satelliteskycoord.py", line 55, in get_satellite_coords
    self._check_times(times)
  File "/Users/jyee/PycharmProjects/MulensModel/source/MulensModel/satelliteskycoord.py", line 104, in _check_times
    raise ValueError(msg.format(*args))
ValueError: Satellite ephemeris doesn't cover requested epochs.
Ephemerides file: 2456658.500777592 2458849.5008007395
Requested dates: 2456615.053144975 2456615.053144975
"""

#print('=== Run the full end-to-end Ground+Spitzer workflow ===')
## Run the full ground+space workflow without stopping.
#full_fitter = exozippy.mmexofast.fit(
#    files=ground_data_files + space_data_files,
#    coords=coords, fit_type='point lens',
#    parallax_grid=True, renormalize_errors=True,
#    verbose=True,
#    output_config=OutputConfig(
#        base_dir=base_dir, file_head='ob0939_uc02c_full', save_log=True, save_plots=True,
#        save_latex_tables=True, save_restart_files=True, save_grid_results=True)
#    )
#full_fitter.fit()

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
