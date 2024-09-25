"""
Show how different different keywords can be used to control fitting.
"""
import glob, os.path
import exozippy

files = glob.glob(os.path.join(exozippy.MULENS_DATA_PATH, 'OB05390', '*.dat'))

# Fit the anomaly starting from the assumption that it is a wide-orbit (s>1) planetary caustic anomaly.
results_1 = exozippy.mmexofast.fit(
        files=[files], fit_type='binary lens', assumption='wide_planet',
        print_results=True)

# Fit the anomaly starting from the assumption that it is a planetary caustic anomaly.
# Report results for both close and wide solutions, even if the alternate solution isn't competitive.
results_2 = exozippy.mmexofast.fit(
        files=[files], fit_type='binary lens', assumption='planetary_caustic',
        check_degenerate=True, print_results=True)

# Perform a blind search for the best 2L1S model.
results_3 = exozippy.mmexofast.fit(
        files=[files], fit_type='binary lens', assumption=None,
        print_results=True)

# Find the best binary source model.
results_bs = exozippy.mmexofast.fit(
        files=[files], fit_type='binary source',
        print_results=True)

# Find the best model and check for standard degeneracies (including binary source).
results_full = exozippy.mmexofast.fit(
        files=[files], check_degenerate=True, print_results=True)

# Find the best model
results_full = exozippy.mmexofast.fit(files=[files], print_results=True)
