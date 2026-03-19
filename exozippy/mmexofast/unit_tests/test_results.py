"""Unit tests for results module."""

import unittest
import pandas as pd
from unittest.mock import Mock

import MulensModel
from sfit_minimizer.sfit_classes import SFitResults
from sfit_minimizer.mm_funcs import PointLensSFitFunction

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
        self.params = {'t_0': t_0, 'u_0': u_0, 't_E': t_E, 'pi_E_N': pi_E_N, 'pi_E_E': pi_E_E}

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
        self.sigmas = {'t_0': t_0_sigma, 'u_0': u_0_sigma, 't_E': t_E_sigma, 'pi_E_E': 'nan', 'pi_E_N': 'nan'}

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
            datasets=[dataset1, dataset2], model=MulensModel.Model(self.params,
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

    def _assert_dataframe_format(self, df, expected_param_count):
        """Helper method to validate DataFrame structure and content."""
        # Should return DataFrame
        self.assertIsInstance(df, pd.DataFrame)

        # Should have required columns
        self.assertIn('parameter_names', df.columns)
        self.assertIn('values', df.columns)
        self.assertIn('sigmas', df.columns)

        # Should have correct number of rows
        self.assertEqual(len(df), expected_param_count)

    def test_format_results_as_df(self):
        """Test format_results_as_df creates proper DataFrame."""
        result = results.MMEXOFASTFitResults(self.mock_fitter)
        df = result.format_results_as_df()

        # Should have rows for:
        # - All model parameters (5: t_0, u_0, t_E, pi_E_N, pi_E_E)
        # - 4 flux parameters (f1_S, f1_B, f2_S, f2_B)
        # - 2 metadata (chi2, N_data)
        expected_length = len(self.mock_fitter.best) - 1 + 2 * len(self.mock_fitter.datasets) + 2  # 5 + 4 + 2 = 11
        self._assert_dataframe_format(df, expected_length)

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


class TestFitRecord(TestMMEXOFASTFitResults):
    """Test FitRecord class."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()

        # Create MMEXOFASTFitResults from mock_fitter
        self.full_result = results.MMEXOFASTFitResults(self.mock_fitter)

        # Create model key
        self.model_key = fit_types.FitKey(
            lens_type=fit_types.LensType.POINT,
            source_type=fit_types.SourceType.POINT,
            parallax_branch=fit_types.ParallaxBranch.NONE,
            lens_orb_motion=fit_types.LensOrbMotion.NONE,
        )

        self.test_renorm_factors = {'factor1': 1.5, 'factor2': 0.9}

    def test_from_full_result_populates_all_fields(self):
        """Test that from_full_result() populates all fields correctly."""
        # Call from_full_result
        record = results.FitRecord.from_full_result(
            model_key=self.model_key,
            full_result=self.full_result,
            renorm_factors=self.test_renorm_factors,
            fixed=False,
        )

        # Verify all fields are set correctly
        self.assertEqual(record.model_key, self.model_key)
        self.assertEqual(record.params, self.full_result.get_params_from_results())
        self.assertEqual(record.sigmas, self.full_result.get_sigmas_from_results())
        self.assertEqual(record.renorm_factors, self.test_renorm_factors)
        self.assertEqual(record.full_result, self.full_result)
        self.assertFalse(record.fixed)
        self.assertTrue(record.is_complete)
        self.assertEqual(record.chi2(), self.full_result.chi2)

    def test_renorm_factors_stores_values(self):
        """Test that renorm_factors dict is stored and accessible."""
        record = results.FitRecord.from_full_result(
            model_key=self.model_key,
            full_result=self.full_result,
            renorm_factors=self.test_renorm_factors,
            fixed=False,
        )

        self.assertEqual(record.renorm_factors, self.test_renorm_factors)

    def test_renorm_factors_none(self):
        """Test that renorm_factors can be None."""
        record = results.FitRecord.from_full_result(
            model_key=self.model_key,
            full_result=self.full_result,
            renorm_factors=None,
            fixed=False,
        )

        self.assertIsNone(record.renorm_factors)

    def test_renorm_factors_empty_dict(self):
        """Test that renorm_factors can be an empty dict."""
        record = results.FitRecord.from_full_result(
            model_key=self.model_key,
            full_result=self.full_result,
            renorm_factors={},
            fixed=False,
        )

        self.assertEqual(record.renorm_factors, {})

    def test_is_complete_true_from_full_result(self):
        """Test that is_complete is True when created via from_full_result()."""
        record = results.FitRecord.from_full_result(
            model_key=self.model_key,
            full_result=self.full_result,
            renorm_factors=self.test_renorm_factors,
            fixed=False,
        )

        self.assertTrue(record.is_complete)

    def test_is_complete_false_user_supplied(self):
        """Test that is_complete is False when user supplies params/sigmas only."""
        record = results.FitRecord(
            model_key=self.model_key,
            params=self.params,
            sigmas=self.sigmas,
            renorm_factors=self.test_renorm_factors,
            full_result=None,
            fixed=False,
            is_complete=False,
        )

        self.assertFalse(record.is_complete)

    def test_to_dataframe_with_full_result(self):
        """Test to_dataframe() when full_result is available."""
        record = results.FitRecord.from_full_result(
            model_key=self.model_key,
            full_result=self.full_result,
            renorm_factors=self.test_renorm_factors,
            fixed=False,
        )

        df = record.to_dataframe()

        # Verify it matches what full_result.format_results_as_df() returns
        expected_df = self.full_result.format_results_as_df()
        pd.testing.assert_frame_equal(df, expected_df)

    def test_to_dataframe_without_full_result(self):
        """Test to_dataframe() when user supplies params/sigmas only."""
        record = results.FitRecord(
            model_key=self.model_key,
            params=self.params,
            sigmas=self.sigmas,
            renorm_factors=self.test_renorm_factors,
            full_result=None,
            fixed=False,
            is_complete=False,
        )

        df = record.to_dataframe()

        # Validate DataFrame format
        self._assert_dataframe_format(df, len(self.params))

        # Verify parameter names match
        self.assertEqual(list(df['parameter_names']), list(self.params.keys()))

        # Verify values match
        self.assertEqual(list(df['values']), list(self.params.values()))

        # Verify sigmas match
        self.assertEqual(list(df['sigmas']), list(self.sigmas.values()))

    def test_from_full_result_sigmas_exception(self):
        """Test that sigmas is None when get_sigmas_from_results() raises exception."""
        # Mock full_result to raise exception on get_sigmas_from_results()
        mock_full_result = Mock(spec=results.MMEXOFASTFitResults)
        mock_full_result.get_params_from_results.return_value = self.params
        mock_full_result.get_sigmas_from_results.side_effect = Exception("Sigmas not available")
        mock_full_result.chi2 = 100.0

        record = results.FitRecord.from_full_result(
            model_key=self.model_key,
            full_result=mock_full_result,
            fixed=False,
        )

        self.assertIsNone(record.sigmas)

    def test_to_dataframe_without_sigmas(self):
        """Test to_dataframe() when params defined but sigmas is None."""
        record = results.FitRecord(
            model_key=self.model_key,
            params=self.params,
            sigmas=None,
            renorm_factors=self.test_renorm_factors,
            full_result=None,
            fixed=False,
            is_complete=False,
        )

        df = record.to_dataframe()

        # Validate DataFrame format
        self._assert_dataframe_format(df, len(self.params))

        # Verify parameter names match
        self.assertEqual(list(df['parameter_names']), list(self.params.keys()))

        # Verify values match
        self.assertEqual(list(df['values']), list(self.params.values()))

        # Verify all sigmas are None
        self.assertTrue(df['sigmas'].isna().all())

    def test_chi2_returns_none_without_full_result(self):
        """Test that chi2() returns None when full_result is None."""
        record = results.FitRecord(
            model_key=self.model_key,
            params=self.params,
            sigmas=self.sigmas,
            renorm_factors=self.test_renorm_factors,
            full_result=None,
            fixed=False,
            is_complete=False,
        )

        self.assertIsNone(record.chi2())


class TestAllFitResults(TestFitRecord):
    """Test AllFitResults class."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()

        # Create the 3 model keys
        self.model_key_pspl_static = fit_types.FitKey(
            lens_type=fit_types.LensType.POINT,
            source_type=fit_types.SourceType.POINT,
            parallax_branch=fit_types.ParallaxBranch.NONE,
            lens_orb_motion=fit_types.LensOrbMotion.NONE,
        )

        self.model_key_pspl_parallax = fit_types.FitKey(
            lens_type=fit_types.LensType.POINT,
            source_type=fit_types.SourceType.POINT,
            parallax_branch=fit_types.ParallaxBranch.U0_MINUS,
            lens_orb_motion=fit_types.LensOrbMotion.NONE,
        )

        self.model_key_binary = fit_types.FitKey(
            lens_type=fit_types.LensType.BINARY,
            source_type=fit_types.SourceType.POINT,
            parallax_branch=fit_types.ParallaxBranch.NONE,
            lens_orb_motion=fit_types.LensOrbMotion.NONE,
        )

        # Create AllFitResults with 3 FitRecords
        self.all_results = results.AllFitResults()

        self.record_pspl_static = results.FitRecord.from_full_result(
            model_key=self.model_key_pspl_static,
            full_result=self.full_result,
            fixed=False,
        )
        self.all_results[self.model_key_pspl_static] = self.record_pspl_static

        self.record_pspl_parallax = results.FitRecord.from_full_result(
            model_key=self.model_key_pspl_parallax,
            full_result=self.full_result,
            fixed=False,
        )
        self.all_results[self.model_key_pspl_parallax] = self.record_pspl_parallax

        self.record_binary = results.FitRecord.from_full_result(
            model_key=self.model_key_binary,
            full_result=self.full_result,
            fixed=False,
        )
        self.all_results[self.model_key_binary] = self.record_binary

    def test_getitem_by_fitkey(self):
        """Test __getitem__ retrieves record by FitKey."""
        retrieved = self.all_results[self.model_key_pspl_static]
        self.assertEqual(retrieved, self.record_pspl_static)

    def test_getitem_by_label(self):
        """Test __getitem__ retrieves record by string label."""
        label = fit_types.model_key_to_label(self.model_key_pspl_static)
        retrieved = self.all_results[label]
        self.assertEqual(retrieved, self.record_pspl_static)

    def test_getitem_keyerror(self):
        """Test __getitem__ raises KeyError for non-existent record."""
        fake_key = fit_types.FitKey(
            lens_type=fit_types.LensType.POINT,
            source_type=fit_types.SourceType.FINITE,
            parallax_branch=fit_types.ParallaxBranch.NONE,
            lens_orb_motion=fit_types.LensOrbMotion.NONE,
        )

        with self.assertRaises(KeyError):
            _ = self.all_results[fake_key]

    def test_setitem_new_record(self):
        """Test __setitem__ adds a new record."""
        new_key = fit_types.FitKey(
            lens_type=fit_types.LensType.POINT,
            source_type=fit_types.SourceType.FINITE,
            parallax_branch=fit_types.ParallaxBranch.NONE,
            lens_orb_motion=fit_types.LensOrbMotion.NONE,
        )

        new_record = results.FitRecord.from_full_result(
            model_key=new_key,
            full_result=self.full_result,
            fixed=False,
        )

        self.all_results[new_key] = new_record

        retrieved = self.all_results[new_key]
        self.assertEqual(retrieved, new_record)
        self.assertEqual(len(self.all_results), 4)  # Was 3, now 4

    def test_setitem_overwrite_record(self):
        """Test __setitem__ overwrites an existing record."""
        new_record = results.FitRecord.from_full_result(
            model_key=self.model_key_pspl_static,
            full_result=self.full_result,
            fixed=True,  # Different from original
        )

        original_length = len(self.all_results)
        self.all_results[self.model_key_pspl_static] = new_record

        retrieved = self.all_results[self.model_key_pspl_static]
        self.assertEqual(retrieved, new_record)
        self.assertEqual(len(self.all_results), original_length)  # Length unchanged
        self.assertTrue(retrieved.fixed)

    def test_delitem_existing_record(self):
        """Test __delitem__ removes an existing record."""
        original_length = len(self.all_results)

        del self.all_results[self.model_key_pspl_static]

        self.assertEqual(len(self.all_results), original_length - 1)
        with self.assertRaises(KeyError):
            _ = self.all_results[self.model_key_pspl_static]

    def test_delitem_by_label(self):
        """Test __delitem__ removes record by string label."""
        label = fit_types.model_key_to_label(self.model_key_pspl_parallax)
        original_length = len(self.all_results)

        del self.all_results[label]

        self.assertEqual(len(self.all_results), original_length - 1)
        with self.assertRaises(KeyError):
            _ = self.all_results[self.model_key_pspl_parallax]

    def test_delitem_keyerror(self):
        """Test __delitem__ raises KeyError for non-existent record."""
        fake_key = fit_types.FitKey(
            lens_type=fit_types.LensType.POINT,
            source_type=fit_types.SourceType.FINITE,
            parallax_branch=fit_types.ParallaxBranch.NONE,
            lens_orb_motion=fit_types.LensOrbMotion.NONE,
        )

        with self.assertRaises(KeyError):
            del self.all_results[fake_key]

    def test_iter_len_contains(self):
        """Test __iter__, __len__, and __contains__ work together."""
        # Test __len__
        self.assertEqual(len(self.all_results), 3)

        # Test __iter__ and __contains__
        keys_from_iter = list(self.all_results)
        self.assertEqual(len(keys_from_iter), 3)

        # Verify all expected keys are in the iterator
        self.assertIn(self.model_key_pspl_static, keys_from_iter)
        self.assertIn(self.model_key_pspl_parallax, keys_from_iter)
        self.assertIn(self.model_key_binary, keys_from_iter)

        # Test __contains__ for existing keys
        self.assertIn(self.model_key_pspl_static, self.all_results)
        self.assertIn(self.model_key_pspl_parallax, self.all_results)
        self.assertIn(self.model_key_binary, self.all_results)

        # Test __contains__ for non-existent key
        fake_key = fit_types.FitKey(
            lens_type=fit_types.LensType.POINT,
            source_type=fit_types.SourceType.FINITE,
            parallax_branch=fit_types.ParallaxBranch.NONE,
            lens_orb_motion=fit_types.LensOrbMotion.NONE,
        )
        self.assertNotIn(fake_key, self.all_results)

    def test_normalize_key_fitkey_unchanged(self):
        """Test that FitKey objects are returned unchanged by _normalize_key()."""
        all_results = results.AllFitResults()

        # Access _normalize_key through the object
        normalized = all_results._normalize_key(self.model_key_pspl_static)

        self.assertEqual(normalized, self.model_key_pspl_static)

    def test_normalize_key_label_to_fitkey(self):
        """Test that string labels are converted to FitKey by _normalize_key()."""
        all_results = results.AllFitResults()

        label = fit_types.model_key_to_label(self.model_key_pspl_static)
        normalized = all_results._normalize_key(label)

        self.assertEqual(normalized, self.model_key_pspl_static)

    def test_key_normalization_consistency(self):
        """Test that label and FitKey access retrieve the same record."""
        label = fit_types.model_key_to_label(self.model_key_pspl_static)

        # Access by label
        retrieved_by_label = self.all_results[label]

        # Access by FitKey
        retrieved_by_key = self.all_results[self.model_key_pspl_static]

        self.assertEqual(retrieved_by_label, retrieved_by_key)
        self.assertEqual(retrieved_by_label, self.record_pspl_static)

    def test_get_by_fitkey(self):
        """Test get() retrieves record by FitKey."""
        retrieved = self.all_results.get(self.model_key_pspl_static)
        self.assertEqual(retrieved, self.record_pspl_static)

    def test_get_by_label(self):
        """Test get() retrieves record by string label."""
        label = fit_types.model_key_to_label(self.model_key_pspl_static)
        retrieved = self.all_results.get(label)
        self.assertEqual(retrieved, self.record_pspl_static)

    def test_get_nonexistent_returns_none(self):
        """Test get() returns None for non-existent record."""
        fake_key = fit_types.FitKey(
            lens_type=fit_types.LensType.POINT,
            source_type=fit_types.SourceType.FINITE,
            parallax_branch=fit_types.ParallaxBranch.NONE,
            lens_orb_motion=fit_types.LensOrbMotion.NONE,
        )

        retrieved = self.all_results.get(fake_key)
        self.assertIsNone(retrieved)

    def test_set_new_record(self):
        """Test set() adds a new record."""
        new_key = fit_types.FitKey(
            lens_type=fit_types.LensType.POINT,
            source_type=fit_types.SourceType.FINITE,
            parallax_branch=fit_types.ParallaxBranch.NONE,
            lens_orb_motion=fit_types.LensOrbMotion.NONE,
        )

        new_record = results.FitRecord.from_full_result(
            model_key=new_key,
            full_result=self.full_result,
            fixed=False,
        )

        original_length = len(self.all_results)
        self.all_results.set(new_record)

        self.assertEqual(len(self.all_results), original_length + 1)
        self.assertEqual(self.all_results.get(new_key), new_record)

    def test_set_overwrite_record(self):
        """Test set() overwrites an existing record."""
        new_record = results.FitRecord.from_full_result(
            model_key=self.model_key_pspl_static,
            full_result=self.full_result,
            fixed=True,
        )

        original_length = len(self.all_results)
        self.all_results.set(new_record)

        self.assertEqual(len(self.all_results), original_length)
        self.assertEqual(self.all_results.get(self.model_key_pspl_static), new_record)
        self.assertTrue(self.all_results.get(self.model_key_pspl_static).fixed)

    def test_has_by_fitkey(self):
        """Test has() returns True for existing record by FitKey."""
        self.assertTrue(self.all_results.has(self.model_key_pspl_static))

    def test_has_by_label(self):
        """Test has() returns True for existing record by label."""
        label = fit_types.model_key_to_label(self.model_key_pspl_static)
        self.assertTrue(self.all_results.has(label))

    def test_has_nonexistent_returns_false(self):
        """Test has() returns False for non-existent record."""
        fake_key = fit_types.FitKey(
            lens_type=fit_types.LensType.POINT,
            source_type=fit_types.SourceType.FINITE,
            parallax_branch=fit_types.ParallaxBranch.NONE,
            lens_orb_motion=fit_types.LensOrbMotion.NONE,
        )

        self.assertFalse(self.all_results.has(fake_key))

    def test_get_returns_none_not_keyerror(self):
        """Test get() returns None instead of raising KeyError for non-existent."""
        fake_key = fit_types.FitKey(
            lens_type=fit_types.LensType.POINT,
            source_type=fit_types.SourceType.FINITE,
            parallax_branch=fit_types.ParallaxBranch.NONE,
            lens_orb_motion=fit_types.LensOrbMotion.NONE,
        )

        result = self.all_results.get(fake_key)
        self.assertIsNone(result)

    def test_set_uses_record_model_key(self):
        """Test set() automatically uses record.model_key for storage."""
        new_key = fit_types.FitKey(
            lens_type=fit_types.LensType.POINT,
            source_type=fit_types.SourceType.FINITE,
            parallax_branch=fit_types.ParallaxBranch.NONE,
            lens_orb_motion=fit_types.LensOrbMotion.NONE,
        )

        new_record = results.FitRecord.from_full_result(
            model_key=new_key,
            full_result=self.full_result,
            fixed=False,
        )

        # set() should use new_record.model_key automatically
        self.all_results.set(new_record)

        # Should be retrievable by the record's model_key
        self.assertEqual(self.all_results.get(new_key), new_record)

    def test_keys_labels_false(self):
        """Test keys() returns FitKey objects when labels=False."""
        keys = self.all_results.keys(labels=False)

        self.assertEqual(len(keys), 3)
        self.assertIn(self.model_key_pspl_static, keys)
        self.assertIn(self.model_key_pspl_parallax, keys)
        self.assertIn(self.model_key_binary, keys)

    def test_keys_labels_true(self):
        """Test keys() returns string labels when labels=True."""
        keys = self.all_results.keys(labels=True)

        self.assertEqual(len(keys), 3)

        label_pspl_static = fit_types.model_key_to_label(self.model_key_pspl_static)
        label_pspl_parallax = fit_types.model_key_to_label(self.model_key_pspl_parallax)
        label_binary = fit_types.model_key_to_label(self.model_key_binary)

        self.assertIn(label_pspl_static, keys)
        self.assertIn(label_pspl_parallax, keys)
        self.assertIn(label_binary, keys)

        # Verify they are strings
        for key in keys:
            self.assertIsInstance(key, str)

    def test_items_labels_false(self):
        """Test items() returns (FitKey, record) tuples when labels=False."""
        items = self.all_results.items(labels=False)

        self.assertEqual(len(items), 3)

        items_dict = dict(items)
        self.assertEqual(items_dict[self.model_key_pspl_static], self.record_pspl_static)
        self.assertEqual(items_dict[self.model_key_pspl_parallax], self.record_pspl_parallax)
        self.assertEqual(items_dict[self.model_key_binary], self.record_binary)

    def test_items_labels_true(self):
        """Test items() returns (label, record) tuples when labels=True."""
        items = self.all_results.items(labels=True)

        self.assertEqual(len(items), 3)

        label_pspl_static = fit_types.model_key_to_label(self.model_key_pspl_static)
        label_pspl_parallax = fit_types.model_key_to_label(self.model_key_pspl_parallax)
        label_binary = fit_types.model_key_to_label(self.model_key_binary)

        items_dict = dict(items)
        self.assertEqual(items_dict[label_pspl_static], self.record_pspl_static)
        self.assertEqual(items_dict[label_pspl_parallax], self.record_pspl_parallax)
        self.assertEqual(items_dict[label_binary], self.record_binary)

        # Verify all keys are strings
        for label in items_dict.keys():
            self.assertIsInstance(label, str)

    def test_iter_point_lens_records_excludes_binary(self):
        """Test A: iter_point_lens_records() does not yield binary lens records."""
        yielded_keys = [key for key, _ in self.all_results.iter_point_lens_records()]
        self.assertNotIn(self.model_key_binary, yielded_keys)

    def test_iter_point_lens_records_yields_all_point_lens(self):
        """Test B: iter_point_lens_records() yields all point lens records."""
        yielded_keys = [key for key, _ in self.all_results.iter_point_lens_records()]
        self.assertIn(self.model_key_pspl_static, yielded_keys)
        self.assertIn(self.model_key_pspl_parallax, yielded_keys)
        self.assertEqual(len(yielded_keys), 2)

    def test_iter_point_lens_records_yields_key_record_tuples(self):
        """Test C: iter_point_lens_records() yields (key, record) tuples."""
        for item in self.all_results.iter_point_lens_records():
            self.assertIsInstance(item, tuple)
            key, record = item
            self.assertIsInstance(key, fit_types.FitKey)
            self.assertIsInstance(record, results.FitRecord)


if __name__ == '__main__':
    unittest.main()
