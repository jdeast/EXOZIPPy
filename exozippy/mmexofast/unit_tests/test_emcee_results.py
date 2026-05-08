import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd

import MulensModel
from exozippy.mmexofast import EmceeFitResults, FitRecord


# ===========================================================================
# Module-level constants
# ===========================================================================

_DEFAULT_PARAMETERS_TO_FIT = [
    't_0', 'u_0', 't_E', 'log_rho', 'log_s', 'log_q', 'alpha'
]
_TRUE_VALUES_MAP = {
    't_0': 2459000.0,
    'u_0': 0.1,
    't_E': 20.0,
    'log_rho': -2.0,
    'log_s': 0.1,
    'log_q': -3.0,
    'alpha': 270.0,
}
_STEP_SIZES_MAP = {
    't_0': 0.1,
    'u_0': 0.001,
    't_E': 0.5,
    'log_rho': 0.01,
    'log_s': 0.01,
    'log_q': 0.01,
    'alpha': 1.0,
}

_N_WALKERS = 20
_N_STEPS = 100
_N_BURN = 20

_N_GOOD_DATASET_1 = 100
_N_GOOD_DATASET_2 = 80
_N_DATA = _N_GOOD_DATASET_1 + _N_GOOD_DATASET_2  # 180

# Fluxes returned by mock_event.source_fluxes() and mock_event.blend_fluxes().
# Index 0 = OGLE (dataset 1), index 1 = MOA (dataset 2).
_SOURCE_FLUXES = [1000.0, 500.0]
_BLEND_FLUXES = [100.0, 50.0]

# Magnitudes derived from mag = 22 - 2.5 * log10(flux)
_MOCK_MAG_I_SOURCE_OGLE = 22 - 2.5 * np.log10(_SOURCE_FLUXES[0])   # 14.5
_MOCK_MAG_I_BLEND_OGLE  = 22 - 2.5 * np.log10(_BLEND_FLUXES[0])    # 17.0
_MOCK_MAG_R_SOURCE_MOA  = 22 - 2.5 * np.log10(_SOURCE_FLUXES[1])   # ~15.253
_MOCK_MAG_R_BLEND_MOA   = 22 - 2.5 * np.log10(_BLEND_FLUXES[1])    # ~17.753

_EXPECTED_FLUX_PARAM_NAMES = ['I_S_OGLE', 'I_B_OGLE', 'R_S_MOA', 'R_B_MOA']
_EXPECTED_FLUX_PARAM_VALUES = {
    'I_S_OGLE': _MOCK_MAG_I_SOURCE_OGLE,
    'I_B_OGLE': _MOCK_MAG_I_BLEND_OGLE,
    'R_S_MOA':  _MOCK_MAG_R_SOURCE_MOA,
    'R_B_MOA':  _MOCK_MAG_R_BLEND_MOA,
}

_FLUX_TO_MAG = {
    _SOURCE_FLUXES[0]: _MOCK_MAG_I_SOURCE_OGLE,
    _BLEND_FLUXES[0]:  _MOCK_MAG_I_BLEND_OGLE,
    _SOURCE_FLUXES[1]: _MOCK_MAG_R_SOURCE_MOA,
    _BLEND_FLUXES[1]:  _MOCK_MAG_R_BLEND_MOA,
}

_MAG_FROM_FLUX_PATCH_PATH = 'MulensModel.utils.Utils.get_mag_and_err_from_flux'


# ===========================================================================
# Module-level helpers
# ===========================================================================

def _get_parameter_name(param):
    """
    Strip log_ prefix from a parameter name.

    Mirrors the behavior of WidePlanetFitter.get_parameter_name so that
    the mock and tests stay in sync with the fitter without importing it.
    """
    if param.startswith('log_'):
        return param[4:]
    return param


def _mock_get_mag_and_err_from_flux(flux, err_flux):
    """
    Stand-in for MulensModel.utils.Utils.get_mag_and_err_from_flux using
    mag = 22 - 2.5 * log10(flux). Returns err_mag = 0.0 for all inputs.
    """
    mag = 22 - 2.5 * np.log10(flux)
    return mag, 0.0


def make_mock_emcee_fitter(
        parameters_to_fit=None,
        fixed_params_dict=None,
        seed=42):
    """
    Build a mock WidePlanetFitter after a completed emcee run.

    Generates a synthetic chain and log-probability array, computes the
    expected post-burn-in percentiles and the max-likelihood sample, and
    returns these alongside the mock fitter for use in test assertions.

    Parameters
    ----------
    parameters_to_fit : list of str, optional
        Parameters sampled by emcee. Defaults to the full 7-parameter
        WidePlanetFitter set.
    fixed_params_dict : dict, optional
        Linear-space parameter values to include in ``best`` as fixed
        (i.e. present in ``best`` but absent from ``parameters_to_fit``).
        Example: ``{'s': 1.1, 'q': 0.001}`` to simulate fixed log_s and
        log_q.
    seed : int
        NumPy random seed for reproducibility.

    Returns
    -------
    mock_fitter : MagicMock
        Mock with attributes ``parameters_to_fit``, ``best``,
        ``best_theta``, ``sampler.chain``, ``sampler.lnprobability``,
        ``emcee_settings``, ``datasets``, ``_event``, and
        ``get_parameter_name``.
    parameters_to_fit : list of str
    best_dict : dict
        Max-likelihood parameters in linear space including ``chi2``.
    best_theta : np.ndarray, shape (n_params,)
        Max-likelihood parameter vector in emcee (log/native) space.
    expected : dict
        Keys ``p16``, ``p50``, ``p84``, ``sigma_minus``, ``sigma_plus``,
        all of shape ``(n_params,)``. Derived from the post-burn-in chain.

    Notes
    -----
    sigma_minus = p50 - p16  (stored as a positive number; the minus sign
    is a display concern only).
    sigma_plus  = p84 - p50  (stored as a positive number).
    """
    np.random.seed(seed)

    if parameters_to_fit is None:
        parameters_to_fit = list(_DEFAULT_PARAMETERS_TO_FIT)

    n_params   = len(parameters_to_fit)
    true_values = np.array([_TRUE_VALUES_MAP[p] for p in parameters_to_fit])
    step_sizes  = np.array([_STEP_SIZES_MAP[p]  for p in parameters_to_fit])

    # chain: (n_walkers, n_steps, n_params)
    chain = (true_values
             + np.random.randn(_N_WALKERS, _N_STEPS, n_params) * step_sizes)
    # lnprobability: (n_walkers, n_steps)

    # Inside make_mock_emcee_fitter()  (changed sections only)

    # lnprobability: (n_walkers, n_steps)
    # ln_prob = -0.5 * chi2, where chi2 = N_data at the best fit plus
    # a penalty term for parameter deviations from the true values.
    lnprobability = -0.5 * (
            _N_DATA
            + np.sum(((chain - true_values) / step_sizes) ** 2, axis=2)
    )

    samples = chain[:, _N_BURN:, :].reshape((-1, n_params))
    prob = lnprobability[:, _N_BURN:].reshape(-1)
    best_theta = samples[np.argmax(prob)]

    # chi2 at the best sample: N_data plus the residual deviation term.
    # Will be slightly above N_data unless best_theta == true_values exactly.
    chi2_best = _N_DATA + np.sum(((best_theta - true_values) / step_sizes) ** 2)

    best_dict = {}
    for i, param in enumerate(parameters_to_fit):
        linear_key = _get_parameter_name(param)
        best_dict[linear_key] = (
            10. ** best_theta[i] if param.startswith('log_') else best_theta[i]
        )
    if fixed_params_dict is not None:
        best_dict.update(fixed_params_dict)
    best_dict['chi2'] = chi2_best

    # Expected percentiles over post-burn-in samples
    p = np.percentile(samples, [16, 50, 84], axis=0)
    expected = {
        'p16':         p[0],
        'p50':         p[1],
        'p84':         p[2],
        'sigma_minus': p[1] - p[0],
        'sigma_plus':  p[2] - p[1],
    }

    # --- mock objects -------------------------------------------------------

    mock_sampler = MagicMock()
    mock_sampler.chain         = chain
    mock_sampler.lnprobability = lnprobability

    mock_dataset_1 = MagicMock()
    mock_dataset_1.plot_properties = {'label': 'n20100309.I.OGLE.OB08092.txt'}
    mock_dataset_1.good = np.ones(_N_GOOD_DATASET_1, dtype=bool)

    mock_dataset_2 = MagicMock()
    mock_dataset_2.plot_properties = {'label': 'n20100309.R.MOA.OB08092.txt'}
    mock_dataset_2.good = np.ones(_N_GOOD_DATASET_2, dtype=bool)

    mock_event = MagicMock()
    mock_event.source_fluxes.return_value = list(_SOURCE_FLUXES)
    mock_event.blend_fluxes.return_value  = list(_BLEND_FLUXES)

    mock_fitter = MagicMock()
    mock_fitter.parameters_to_fit  = parameters_to_fit
    mock_fitter.best               = best_dict
    mock_fitter.best_theta         = best_theta
    mock_fitter.sampler            = mock_sampler
    mock_fitter.emcee_settings     = {'n_burn': _N_BURN, 'n_dim': n_params}
    mock_fitter.get_parameter_name = MagicMock(side_effect=_get_parameter_name)
    mock_fitter.datasets           = [mock_dataset_1, mock_dataset_2]
    mock_fitter._event             = mock_event

    return mock_fitter, parameters_to_fit, best_dict, best_theta, expected


# ===========================================================================
# TestWidePlanetFitterBestTheta
# ===========================================================================

class TestWidePlanetFitterBestTheta(unittest.TestCase):
    """
    Documents and verifies the contract for ``best_theta`` on
    ``WidePlanetFitter`` after ``run()``.

    ``EmceeFitResults`` relies on this contract, so these tests should pass
    before ``EmceeFitResults`` is written. They also serve as regression
    tests once ``best_theta = samples[best_index, :]`` is added to ``run()``.

    These are just tests to verify the mock_fitter is internally consistent.
    """

    def setUp(self):
        (self.mock_fitter,
         self.parameters_to_fit,
         self.best_dict,
         self.best_theta,
         _) = make_mock_emcee_fitter()

    def test_best_theta_is_numpy_array(self):
        """best_theta must be a numpy ndarray."""
        self.assertIsInstance(self.mock_fitter.best_theta, np.ndarray)

    def test_best_theta_has_correct_shape(self):
        """best_theta must have shape (n_params,)."""
        self.assertEqual(
            self.mock_fitter.best_theta.shape,
            (len(self.parameters_to_fit),)
        )

    def test_best_theta_non_log_params_match_best(self):
        """For non-log params, best_theta[i] must equal best[param]."""
        for i, param in enumerate(self.parameters_to_fit):
            if not param.startswith('log_'):
                np.testing.assert_almost_equal(
                    self.mock_fitter.best_theta[i],
                    self.best_dict[param],
                    err_msg=f"Mismatch for non-log param '{param}'"
                )

    def test_best_theta_log_params_round_trip_to_best(self):
        """For log_ params, 10**best_theta[i] must equal best[linear_param]."""
        for i, param in enumerate(self.parameters_to_fit):
            if param.startswith('log_'):
                linear_key = _get_parameter_name(param)
                np.testing.assert_almost_equal(
                    10. ** self.mock_fitter.best_theta[i],
                    self.best_dict[linear_key],
                    err_msg=f"Round-trip mismatch: '{param}' -> '{linear_key}'"
                )


# ===========================================================================
# TestEmceeFitResults  (all parameters free)
# ===========================================================================

class TestEmceeFitResults(unittest.TestCase):
    """
    Unit tests for ``EmceeFitResults`` when all parameters are free.
    """

    def setUp(self):
        (self.mock_fitter,
         self.parameters_to_fit,
         self.best_dict,
         self.best_theta,
         self.expected) = make_mock_emcee_fitter()
        self.results = EmceeFitResults(self.mock_fitter)

    def _get_df(self):
        """Return format_results_as_df() with get_mag_and_err_from_flux patched."""
        with patch(_MAG_FROM_FLUX_PATCH_PATH,
                   side_effect=_mock_get_mag_and_err_from_flux):
            return self.results.format_results_as_df()

    # -----------------------------------------------------------------------
    # get_params_from_results
    # -----------------------------------------------------------------------

    def test_get_params_excludes_chi2(self):
        """get_params_from_results() must not include 'chi2'."""
        self.assertNotIn('chi2', self.results.get_params_from_results())

    def test_get_params_contains_all_fitted_params_in_linear_space(self):
        """get_params_from_results() must include every fitted parameter
        using linear-space keys (e.g. 'rho' not 'log_rho')."""
        params = self.results.get_params_from_results()
        for param in self.parameters_to_fit:
            self.assertIn(
                _get_parameter_name(param), params,
                msg=f"Linear key for '{param}' missing from get_params_from_results()"
            )

    def test_get_params_values_equal_best(self):
        """get_params_from_results() values must equal best (excluding chi2)."""
        params = self.results.get_params_from_results()
        expected = {k: v for k, v in self.best_dict.items() if k != 'chi2'}
        self.assertEqual(set(params.keys()), set(expected.keys()))
        for key in expected:
            np.testing.assert_almost_equal(
                params[key], expected[key],
                err_msg=f"Value mismatch for param '{key}'"
            )

    # -----------------------------------------------------------------------
    # get_sigmas_from_results
    # -----------------------------------------------------------------------

    def test_get_sigmas_contains_all_fitted_params(self):
        """get_sigmas_from_results() must return a key for every fitted parameter."""
        sigmas = self.results.get_sigmas_from_results()
        for param in self.parameters_to_fit:
            self.assertIn(param, sigmas,
                          msg=f"'{param}' missing from get_sigmas_from_results()")

    def test_get_sigmas_values_are_mean_of_sigma_minus_and_sigma_plus(self):
        """get_sigmas_from_results() must return (sigma_minus + sigma_plus) / 2."""
        sigmas = self.results.get_sigmas_from_results()
        for i, param in enumerate(self.parameters_to_fit):
            expected_sigma = (
                (self.expected['sigma_minus'][i] + self.expected['sigma_plus'][i]) / 2
            )
            np.testing.assert_almost_equal(
                sigmas[param], expected_sigma,
                err_msg=f"Sigma mismatch for '{param}'"
            )

    # -----------------------------------------------------------------------
    # format_results_as_df — structure
    # -----------------------------------------------------------------------

    def test_format_returns_dataframe(self):
        """format_results_as_df() must return a pandas DataFrame."""
        self.assertIsInstance(self._get_df(), pd.DataFrame)

    def test_format_has_required_columns(self):
        """DataFrame must have columns: parameter_names, values,
        sigma_minus, sigma_plus."""
        df = self._get_df()
        for col in ['parameter_names', 'values', 'sigma_minus', 'sigma_plus']:
            self.assertIn(col, df.columns, msg=f"Missing column '{col}'")

    def test_format_has_correct_row_count(self):
        """DataFrame must have n_fitted + 1 (chi2) + 1 (N_data) +
        2 * n_datasets (flux) rows."""
        df = self._get_df()
        expected_rows = (
            len(self.parameters_to_fit)     # 7 fitted
            + 1                             # chi2
            + 1                             # N_data
            + 2 * len(self.mock_fitter.datasets)  # source + blend per dataset
        )
        self.assertEqual(len(df), expected_rows)

    # -----------------------------------------------------------------------
    # format_results_as_df — fitted parameter rows
    # -----------------------------------------------------------------------

    def test_format_fitted_params_all_present(self):
        """All fitted parameter names must appear in the dataframe."""
        df = self._get_df()
        for param in self.parameters_to_fit:
            self.assertIn(param, df['parameter_names'].values,
                          msg=f"Fitted param '{param}' missing from dataframe")

    def test_format_fitted_param_values_are_p50(self):
        """values for fitted params must be the 50th percentile."""
        df = self._get_df()
        for i, param in enumerate(self.parameters_to_fit):
            row = df[df['parameter_names'] == param].iloc[0]
            np.testing.assert_almost_equal(
                row['values'], self.expected['p50'][i],
                err_msg=f"values mismatch for fitted param '{param}'"
            )

    def test_format_fitted_param_sigma_minus_is_p50_minus_p16(self):
        """sigma_minus for fitted params must be p50 - p16 (positive)."""
        df = self._get_df()
        for i, param in enumerate(self.parameters_to_fit):
            row = df[df['parameter_names'] == param].iloc[0]
            np.testing.assert_almost_equal(
                row['sigma_minus'], self.expected['sigma_minus'][i],
                err_msg=f"sigma_minus mismatch for fitted param '{param}'"
            )

    def test_format_fitted_param_sigma_plus_is_p84_minus_p50(self):
        """sigma_plus for fitted params must be p84 - p50 (positive)."""
        df = self._get_df()
        for i, param in enumerate(self.parameters_to_fit):
            row = df[df['parameter_names'] == param].iloc[0]
            np.testing.assert_almost_equal(
                row['sigma_plus'], self.expected['sigma_plus'][i],
                err_msg=f"sigma_plus mismatch for fitted param '{param}'"
            )

    def test_format_fitted_param_sigma_minus_is_positive(self):
        """sigma_minus for fitted params must be strictly positive."""
        df = self._get_df()
        for param in self.parameters_to_fit:
            row = df[df['parameter_names'] == param].iloc[0]
            self.assertGreater(row['sigma_minus'], 0,
                               msg=f"sigma_minus not positive for '{param}'")

    def test_format_fitted_param_sigma_plus_is_positive(self):
        """sigma_plus for fitted params must be strictly positive."""
        df = self._get_df()
        for param in self.parameters_to_fit:
            row = df[df['parameter_names'] == param].iloc[0]
            self.assertGreater(row['sigma_plus'], 0,
                               msg=f"sigma_plus not positive for '{param}'")

    # -----------------------------------------------------------------------
    # format_results_as_df — fixed parameter rows (chi2 and N_data)
    # -----------------------------------------------------------------------

    def test_format_chi2_present(self):
        """'chi2' must appear in the fixed parameters section."""
        self.assertIn('chi2', self._get_df()['parameter_names'].values)

    def test_format_chi2_value(self):
        """chi2 value must equal the best-fit chi2 derived from the chain."""
        df = self._get_df()
        row = df[df['parameter_names'] == 'chi2'].iloc[0]
        np.testing.assert_almost_equal(row['values'], self.best_dict['chi2'])

    def test_format_chi2_sigma_minus_is_nan(self):
        """chi2 must have NaN for sigma_minus."""
        row = self._get_df()[self._get_df()['parameter_names'] == 'chi2'].iloc[0]
        self.assertTrue(np.isnan(row['sigma_minus']))

    def test_format_chi2_sigma_plus_is_nan(self):
        """chi2 must have NaN for sigma_plus."""
        row = self._get_df()[self._get_df()['parameter_names'] == 'chi2'].iloc[0]
        self.assertTrue(np.isnan(row['sigma_plus']))

    def test_format_n_data_present(self):
        """'N_data' must appear in the fixed parameters section."""
        self.assertIn('N_data', self._get_df()['parameter_names'].values)

    def test_format_n_data_value(self):
        """N_data must equal the total number of good data points."""
        df = self._get_df()
        row = df[df['parameter_names'] == 'N_data'].iloc[0]
        self.assertEqual(row['values'], _N_GOOD_DATASET_1 + _N_GOOD_DATASET_2)

    def test_format_n_data_sigma_minus_is_nan(self):
        """N_data must have NaN for sigma_minus."""
        row = self._get_df()[self._get_df()['parameter_names'] == 'N_data'].iloc[0]
        self.assertTrue(np.isnan(row['sigma_minus']))

    def test_format_n_data_sigma_plus_is_nan(self):
        """N_data must have NaN for sigma_plus."""
        row = self._get_df()[self._get_df()['parameter_names'] == 'N_data'].iloc[0]
        self.assertTrue(np.isnan(row['sigma_plus']))

    # -----------------------------------------------------------------------
    # format_results_as_df — flux parameter rows
    # -----------------------------------------------------------------------

    def test_format_flux_param_names_all_present(self):
        """All flux parameter names must follow the band_S/B_obs convention."""
        df = self._get_df()
        for name in _EXPECTED_FLUX_PARAM_NAMES:
            self.assertIn(name, df['parameter_names'].values,
                          msg=f"Flux param '{name}' missing from dataframe")

    def test_format_flux_param_values_are_mock_magnitudes(self):
        """Flux parameter values must be the magnitudes from the mock event."""
        df = self._get_df()
        for name, expected_mag in _EXPECTED_FLUX_PARAM_VALUES.items():
            row = df[df['parameter_names'] == name].iloc[0]
            np.testing.assert_almost_equal(
                row['values'], expected_mag,
                err_msg=f"Magnitude mismatch for flux param '{name}'"
            )

    def test_format_flux_param_sigma_minus_is_nan(self):
        """Flux parameter rows must have NaN for sigma_minus."""
        df = self._get_df()
        for name in _EXPECTED_FLUX_PARAM_NAMES:
            row = df[df['parameter_names'] == name].iloc[0]
            self.assertTrue(np.isnan(row['sigma_minus']),
                            msg=f"sigma_minus not NaN for flux param '{name}'")

    def test_format_flux_param_sigma_plus_is_nan(self):
        """Flux parameter rows must have NaN for sigma_plus."""
        df = self._get_df()
        for name in _EXPECTED_FLUX_PARAM_NAMES:
            row = df[df['parameter_names'] == name].iloc[0]
            self.assertTrue(np.isnan(row['sigma_plus']),
                            msg=f"sigma_plus not NaN for flux param '{name}'")

    # -----------------------------------------------------------------------
    # format_results_as_df — row ordering
    # -----------------------------------------------------------------------

    def test_format_fitted_params_before_fixed_params(self):
        """All fitted rows must precede all fixed rows (chi2, N_data)."""
        names = list(self._get_df()['parameter_names'].values)
        for fitted in self.parameters_to_fit:
            for fixed in ['chi2', 'N_data']:
                self.assertLess(
                    names.index(fitted), names.index(fixed),
                    msg=f"'{fitted}' must come before '{fixed}'"
                )

    def test_format_fixed_params_before_flux_params(self):
        """All fixed rows (chi2, N_data) must precede all flux rows."""
        names = list(self._get_df()['parameter_names'].values)
        for fixed in ['chi2', 'N_data']:
            for flux in _EXPECTED_FLUX_PARAM_NAMES:
                self.assertLess(
                    names.index(fixed), names.index(flux),
                    msg=f"'{fixed}' must come before '{flux}'"
                )


# ===========================================================================
# TestEmceeFitResultsWithFixedParams  (log_s and log_q fixed)
# ===========================================================================

class TestEmceeFitResultsWithFixedParams(unittest.TestCase):
    """
    Unit tests for ``EmceeFitResults`` when log_s and log_q are fixed.

    Verifies that fixed parameters appear correctly in the dataframe and
    that log-space names do not leak into the output.
    """

    _PARAMETERS_TO_FIT = ['t_0', 'u_0', 't_E', 'log_rho', 'alpha']
    _FIXED_S = 1.1
    _FIXED_Q = 0.001

    def setUp(self):
        (self.mock_fitter,
         self.parameters_to_fit,
         self.best_dict,
         self.best_theta,
         self.expected) = make_mock_emcee_fitter(
            parameters_to_fit=self._PARAMETERS_TO_FIT,
            fixed_params_dict={'s': self._FIXED_S, 'q': self._FIXED_Q},
        )
        self.results = EmceeFitResults(self.mock_fitter)

    def _get_df(self):
        with patch(_MAG_FROM_FLUX_PATCH_PATH,
                   side_effect=_mock_get_mag_and_err_from_flux):
            return self.results.format_results_as_df()

    # -----------------------------------------------------------------------
    # Structure
    # -----------------------------------------------------------------------

    def test_format_has_correct_row_count(self):
        """DataFrame must have 5 fitted + 3 fixed (s, q, chi2) +
        1 N_data + 4 flux rows."""
        df = self._get_df()
        n_fitted  = len(self._PARAMETERS_TO_FIT)   # 5
        n_fixed   = 3                               # s, q, chi2
        n_n_data  = 1
        n_flux    = 2 * len(self.mock_fitter.datasets)  # 4
        self.assertEqual(len(df), n_fitted + n_fixed + n_n_data + n_flux)

    # -----------------------------------------------------------------------
    # Fitted parameter rows
    # -----------------------------------------------------------------------

    def test_format_fitted_params_all_present(self):
        """All fitted parameter names must appear in the dataframe."""
        df = self._get_df()
        for param in self._PARAMETERS_TO_FIT:
            self.assertIn(param, df['parameter_names'].values,
                          msg=f"Fitted param '{param}' missing from dataframe")

    def test_format_fitted_param_values_are_p50(self):
        """values for fitted parameters must be the 50th percentile."""
        df = self._get_df()
        for i, param in enumerate(self._PARAMETERS_TO_FIT):
            row = df[df['parameter_names'] == param].iloc[0]
            np.testing.assert_almost_equal(
                row['values'], self.expected['p50'][i],
                err_msg=f"values mismatch for '{param}'"
            )

    def test_format_fitted_param_sigma_minus(self):
        """sigma_minus for fitted params must be p50 - p16."""
        df = self._get_df()
        for i, param in enumerate(self._PARAMETERS_TO_FIT):
            row = df[df['parameter_names'] == param].iloc[0]
            np.testing.assert_almost_equal(
                row['sigma_minus'], self.expected['sigma_minus'][i],
                err_msg=f"sigma_minus mismatch for '{param}'"
            )

    def test_format_fitted_param_sigma_plus(self):
        """sigma_plus for fitted params must be p84 - p50."""
        df = self._get_df()
        for i, param in enumerate(self._PARAMETERS_TO_FIT):
            row = df[df['parameter_names'] == param].iloc[0]
            np.testing.assert_almost_equal(
                row['sigma_plus'], self.expected['sigma_plus'][i],
                err_msg=f"sigma_plus mismatch for '{param}'"
            )

    # -----------------------------------------------------------------------
    # Fixed parameter rows: s and q
    # -----------------------------------------------------------------------

    def test_format_fixed_s_present(self):
        """Fixed parameter 's' must appear in the dataframe."""
        self.assertIn('s', self._get_df()['parameter_names'].values)

    def test_format_fixed_q_present(self):
        """Fixed parameter 'q' must appear in the dataframe."""
        self.assertIn('q', self._get_df()['parameter_names'].values)

    def test_format_fixed_s_value(self):
        """Fixed parameter 's' must have the correct value."""
        row = self._get_df()[self._get_df()['parameter_names'] == 's'].iloc[0]
        np.testing.assert_almost_equal(row['values'], self._FIXED_S)

    def test_format_fixed_q_value(self):
        """Fixed parameter 'q' must have the correct value."""
        row = self._get_df()[self._get_df()['parameter_names'] == 'q'].iloc[0]
        np.testing.assert_almost_equal(row['values'], self._FIXED_Q)

    def test_format_fixed_s_sigma_minus_is_nan(self):
        """Fixed parameter 's' must have NaN for sigma_minus."""
        row = self._get_df()[self._get_df()['parameter_names'] == 's'].iloc[0]
        self.assertTrue(np.isnan(row['sigma_minus']))

    def test_format_fixed_s_sigma_plus_is_nan(self):
        """Fixed parameter 's' must have NaN for sigma_plus."""
        row = self._get_df()[self._get_df()['parameter_names'] == 's'].iloc[0]
        self.assertTrue(np.isnan(row['sigma_plus']))

    def test_format_fixed_q_sigma_minus_is_nan(self):
        """Fixed parameter 'q' must have NaN for sigma_minus."""
        row = self._get_df()[self._get_df()['parameter_names'] == 'q'].iloc[0]
        self.assertTrue(np.isnan(row['sigma_minus']))

    def test_format_fixed_q_sigma_plus_is_nan(self):
        """Fixed parameter 'q' must have NaN for sigma_plus."""
        row = self._get_df()[self._get_df()['parameter_names'] == 'q'].iloc[0]
        self.assertTrue(np.isnan(row['sigma_plus']))

    def test_format_log_s_absent(self):
        """'log_s' must not appear in the dataframe when s is fixed."""
        self.assertNotIn('log_s', self._get_df()['parameter_names'].values)

    def test_format_log_q_absent(self):
        """'log_q' must not appear in the dataframe when q is fixed."""
        self.assertNotIn('log_q', self._get_df()['parameter_names'].values)

    # -----------------------------------------------------------------------
    # Row ordering
    # -----------------------------------------------------------------------

    def test_format_fitted_params_before_fixed_s_and_q(self):
        """All fitted rows must precede the fixed s and q rows."""
        names = list(self._get_df()['parameter_names'].values)
        for fitted in self._PARAMETERS_TO_FIT:
            for fixed in ['s', 'q']:
                self.assertLess(
                    names.index(fitted), names.index(fixed),
                    msg=f"'{fitted}' must come before '{fixed}'"
                )

    def test_format_fixed_s_and_q_before_flux_params(self):
        """Fixed s and q rows must precede all flux rows."""
        names = list(self._get_df()['parameter_names'].values)
        for fixed in ['s', 'q']:
            for flux in _EXPECTED_FLUX_PARAM_NAMES:
                self.assertLess(
                    names.index(fixed), names.index(flux),
                    msg=f"'{fixed}' must come before '{flux}'"
                )


# ===========================================================================
# TestFitRecordWithEmcee
# ===========================================================================

class TestFitRecordWithEmcee(unittest.TestCase):
    """
    Integration tests for ``FitRecord`` constructed from an
    ``EmceeFitResults`` object.
    """

    def setUp(self):
        (self.mock_fitter,
         self.parameters_to_fit,
         self.best_dict,
         self.best_theta,
         self.expected) = make_mock_emcee_fitter()
        self.emcee_results   = EmceeFitResults(self.mock_fitter)
        self.mock_model_key  = MagicMock()

    def _make_record(self):
        return FitRecord.from_full_result(
            model_key=self.mock_model_key,
            full_result=self.emcee_results,
        )

    def _get_df(self, record):
        with patch(_MAG_FROM_FLUX_PATCH_PATH,
                   side_effect=_mock_get_mag_and_err_from_flux):
            return record.to_dataframe()

    # -----------------------------------------------------------------------
    # FitRecord construction
    # -----------------------------------------------------------------------

    def test_from_full_result_returns_fit_record(self):
        """from_full_result() must accept an EmceeFitResults and return a FitRecord."""
        self.assertIsInstance(self._make_record(), FitRecord)

    def test_from_full_result_is_complete(self):
        """FitRecord created via from_full_result() must be marked complete."""
        self.assertTrue(self._make_record().is_complete)

    def test_from_full_result_params_equal_best(self):
        """FitRecord.params must equal best (excluding chi2)."""
        record   = self._make_record()
        expected = {k: v for k, v in self.best_dict.items() if k != 'chi2'}
        self.assertEqual(set(record.params.keys()), set(expected.keys()))
        for key in expected:
            np.testing.assert_almost_equal(
                record.params[key], expected[key],
                err_msg=f"params mismatch for '{key}'"
            )

    # -----------------------------------------------------------------------
    # to_dataframe
    # -----------------------------------------------------------------------

    def test_to_dataframe_returns_dataframe(self):
        """to_dataframe() must return a pandas DataFrame."""
        self.assertIsInstance(self._get_df(self._make_record()), pd.DataFrame)

    def test_to_dataframe_has_sigma_minus_column(self):
        """to_dataframe() must expose a sigma_minus column for emcee results."""
        self.assertIn('sigma_minus', self._get_df(self._make_record()).columns)

    def test_to_dataframe_has_sigma_plus_column(self):
        """to_dataframe() must expose a sigma_plus column for emcee results."""
        self.assertIn('sigma_plus', self._get_df(self._make_record()).columns)

    def test_to_dataframe_fitted_param_values_are_p50(self):
        """to_dataframe() values for fitted params must be the 50th percentile."""
        df = self._get_df(self._make_record())
        for i, param in enumerate(self.parameters_to_fit):
            row = df[df['parameter_names'] == param].iloc[0]
            np.testing.assert_almost_equal(
                row['values'], self.expected['p50'][i],
                err_msg=f"values mismatch for '{param}'"
            )

    def test_to_dataframe_fitted_param_sigma_minus(self):
        """to_dataframe() sigma_minus for fitted params must be p50 - p16."""
        df = self._get_df(self._make_record())
        for i, param in enumerate(self.parameters_to_fit):
            row = df[df['parameter_names'] == param].iloc[0]
            np.testing.assert_almost_equal(
                row['sigma_minus'], self.expected['sigma_minus'][i],
                err_msg=f"sigma_minus mismatch for '{param}'"
            )

    def test_to_dataframe_fitted_param_sigma_plus(self):
        """to_dataframe() sigma_plus for fitted params must be p84 - p50."""
        df = self._get_df(self._make_record())
        for i, param in enumerate(self.parameters_to_fit):
            row = df[df['parameter_names'] == param].iloc[0]
            np.testing.assert_almost_equal(
                row['sigma_plus'], self.expected['sigma_plus'][i],
                err_msg=f"sigma_plus mismatch for '{param}'"
            )


if __name__ == '__main__':
    unittest.main()
