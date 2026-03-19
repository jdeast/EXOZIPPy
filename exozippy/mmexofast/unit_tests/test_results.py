"""Unit tests for results module."""

import unittest
import pandas as pd
from unittest.mock import Mock

import MulensModel
from sfit_minimizer.sfit_classes import SFitResults
from sfit_minimizer.mm_funcs import PointLensSFitFunction

from exozippy.mmexofast import results, fit_types


from exozippy.mmexofast import results, fit_types


class TestMMEXOFASTFitResults(unittest.TestCase):
    """Test MMEXOFASTFitResults class."""

    def setUp(self):
        """Create mock fitter for testing."""
        # Define parameter values once
        t_0 = 2456789.0
        u_0 = 0.5
        t_E = 20.0
        pi_E_N = 0.0
        pi_E_E = 0.0

        # Flux parameters
        f1_S = 1.5  # log(1.5) ≈ 0.4 (within [-1, 1])
        f1_B = 0.3
        f2_S = 2.0  # log(2.0) ≈ 0.7 (within [-1, 1])
        f2_B = -0.5

        # Uncertainties
        t_0_sigma = 0.1
        u_0_sigma = 0.05
        t_E_sigma = 0.5
        f1_S_sigma = 0.1
        f1_B_sigma = 0.05
        f2_S_sigma = 0.15
        f2_B_sigma = 0.08

        parameters_to_fit = ['t_0', 'u_0', 't_E']

        # Create MulensData objects
        dataset1 = MulensModel.MulensData(
            data_list=[[t_0, t_0 + 1.0], [110.0, 100.0], [1.0, 1.0]],
            phot_fmt='flux'
        )
        dataset1.plot_properties['label'] = 'n20200101.I.test.dataset_1.txt'

        dataset2 = MulensModel.MulensData(
            data_list=[[t_0 + 0.01, t_0 + 1.01], [210.0, 200.0], [2.0, 2.0]],
            phot_fmt='flux'
        )
        dataset2.plot_properties['label'] = 'n20200101.I.test.dataset_2.txt'

        mock_event = MulensModel.Event(
            datasets=[dataset1, dataset2],
            model=MulensModel.Model(
                {'t_0': t_0, 'u_0': u_0, 't_E': t_E, 'pi_E_N': pi_E_N, 'pi_E_E': pi_E_E},
                coords='18:00:00 -30:00:00')
        )
        mock_event.fit_fluxes()
        chi2 = mock_event.get_chi2()

        mock_func = PointLensSFitFunction(mock_event, parameters_to_fit, estimate_fluxes=True)
        mock_func.update_all([t_0, u_0, t_E, f1_S, f1_B, f2_S, f2_B])

        # Create SFitResults object
        fit_results = SFitResults(mock_func)
        fit_results.x = [t_0, u_0, t_E, f1_S, f1_B, f2_S, f2_B]
        fit_results.fun = chi2
        fit_results.sigmas = [t_0_sigma, u_0_sigma, t_E_sigma, f1_S_sigma, f1_B_sigma, f2_S_sigma, f2_B_sigma]
        fit_results.success = True
        fit_results.nit = 10

        self.mock_fitter = Mock()
        self.mock_fitter.datasets = [dataset1, dataset2]
        # best has all model params (fitted + fixed) but NO flux params
        self.mock_fitter.best = {'t_0': t_0, 'u_0': u_0, 't_E': t_E,
                                 'pi_E_N': pi_E_N, 'pi_E_E': pi_E_E,
                                 'chi2': chi2}
        self.mock_fitter.results = fit_results
        self.mock_fitter.parameters_to_fit = parameters_to_fit
        self.mock_fitter.initial_model_params = {'t_0': t_0, 'u_0': u_0, 't_E': t_E,
                                                 'pi_E_N': pi_E_N, 'pi_E_E': pi_E_E}

    def test_init(self):
        """Test MMEXOFASTFitResults initialization."""
        result = results.MMEXOFASTFitResults(self.mock_fitter)
        self.assertEqual(result.fitter, self.mock_fitter)

    def test_datasets_property(self):
        """Test datasets property returns fitter's datasets."""
        result = results.MMEXOFASTFitResults(self.mock_fitter)
        self.assertEqual(result.datasets, self.mock_fitter.datasets)

    def test_best_property(self):
        """Test best property returns fitter's best results."""
        result = results.MMEXOFASTFitResults(self.mock_fitter)
        self.assertEqual(result.best, self.mock_fitter.best)

    def test_best_includes_fixed_params(self):
        """Test that best includes both fitted and fixed parameters."""
        result = results.MMEXOFASTFitResults(self.mock_fitter)
        best = result.best

        # Should include fitted parameters
        for param in self.mock_fitter.parameters_to_fit:
            self.assertIn(param, best)

        # Should include fixed parameters
        self.assertIn('pi_E_N', best)
        self.assertIn('pi_E_E', best)

        # Should NOT include flux parameters
        self.assertNotIn('f1_S', best)

    def test_results_property(self):
        """Test results property returns fitter's results."""
        result = results.MMEXOFASTFitResults(self.mock_fitter)
        self.assertEqual(result.results, self.mock_fitter.results)

    def test_parameters_to_fit_property(self):
        """Test parameters_to_fit property."""
        result = results.MMEXOFASTFitResults(self.mock_fitter)
        self.assertEqual(result.parameters_to_fit, self.mock_fitter.parameters_to_fit)

    def test_all_model_parameters_property(self):
        """Test all_model_parameters property."""
        result = results.MMEXOFASTFitResults(self.mock_fitter)
        all_params = result.all_model_parameters

        # Should include all parameters from initial_model_params
        for param in self.mock_fitter.initial_model_params:
            self.assertIn(param, all_params)

    def test_chi2_property(self):
        """Test chi2 property extracts from best dict."""
        result = results.MMEXOFASTFitResults(self.mock_fitter)
        self.assertEqual(result.chi2, self.mock_fitter.best['chi2'])

    def test_get_params_from_results(self):
        """Test get_params_from_results extracts parameters correctly."""
        result = results.MMEXOFASTFitResults(self.mock_fitter)
        params = result.get_params_from_results()

        # Should return dict
        self.assertIsInstance(params, dict)

        # Should NOT include chi2
        self.assertNotIn('chi2', params)

        # Should match best (all model params, no flux)
        expected_len = len(self.mock_fitter.best) - 1  # Exclude chi2
        self.assertEqual(len(params), expected_len)

        # Explicit check for t_0
        self.assertIn('t_0', params)
        self.assertEqual(params['t_0'], self.mock_fitter.best['t_0'])

        # Loop over all parameters in best (except chi2)
        for param in self.mock_fitter.best:
            if param != 'chi2':
                with self.subTest(param=param):
                    self.assertIn(param, params)
                    self.assertEqual(params[param], self.mock_fitter.best[param])

    def test_get_sigmas_from_results(self):
        """Test get_sigmas_from_results extracts uncertainties correctly."""
        result = results.MMEXOFASTFitResults(self.mock_fitter)
        sigmas = result.get_sigmas_from_results()

        # Should return dict
        self.assertIsInstance(sigmas, dict)

        # Check length matches parameters_to_fit only (sigmas only for fitted params)
        self.assertEqual(len(sigmas), len(self.mock_fitter.parameters_to_fit))

        # Explicit check for t_0
        self.assertIn('t_0', sigmas)
        self.assertAlmostEqual(sigmas['t_0'], 0.1)

        # Loop over fitted parameters
        for i, param in enumerate(self.mock_fitter.parameters_to_fit):
            with self.subTest(param=param):
                self.assertIn(param, sigmas)
                self.assertEqual(sigmas[param], self.mock_fitter.results.sigmas[i])

    def test_format_results_as_df(self):
        """Test format_results_as_df creates proper DataFrame."""
        result = results.MMEXOFASTFitResults(self.mock_fitter)
        df = result.format_results_as_df()

        # Should return DataFrame
        self.assertIsInstance(df, pd.DataFrame)

        # Should have rows for:
        # - All model parameters (5: t_0, u_0, t_E, pi_E_N, pi_E_E)
        # - 4 flux parameters (f1_S, f1_B, f2_S, f2_B)
        # - 2 metadata (chi2, N_data)
        expected_length = len(self.mock_fitter.best) - 1 + 2 * len(self.mock_fitter.datasets) + 2  # 5 + 4 + 2 = 11
        self.assertEqual(len(df), expected_length)

        # Should have required columns
        self.assertIn('parameter_names', df.columns)
        self.assertIn('values', df.columns)
        self.assertIn('sigmas', df.columns)

        # Check that all model parameters are in the DataFrame
        param_names = df['parameter_names'].tolist()
        for param in self.mock_fitter.best:
            if param != 'chi2':
                with self.subTest(param=param):
                    self.assertIn(param, param_names)

        # Check chi2 is included
        self.assertIn('chi2', param_names)

        # Check specific value
        t_0_row = df[df['parameter_names'] == 't_0']
        self.assertEqual(len(t_0_row), 1)
        self.assertAlmostEqual(t_0_row['values'].iloc[0], self.mock_fitter.best['t_0'])
        self.assertAlmostEqual(t_0_row['sigmas'].iloc[0], 0.1)


if __name__ == '__main__':
    unittest.main()
