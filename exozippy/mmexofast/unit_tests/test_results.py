"""Unit tests for results module."""

import unittest
import pandas as pd
from unittest.mock import Mock
import pickle
import os.path

import exozippy
from exozippy.mmexofast import results, fit_types
from exozippy.mmexofast.unit_tests.test_utils import (
    create_mock_fitter,
    create_mock_params_and_sigmas,
)


class TestMMEXOFASTFitResults(unittest.TestCase):
    """Test MMEXOFASTFitResults class."""

    def setUp(self):
        """Create mock fitter for testing."""
        self.params, self.sigmas = create_mock_params_and_sigmas()
        self.mock_fitter = create_mock_fitter()

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

        self.assertIsInstance(params, dict)
        self.assertNotIn('chi2', params)

        expected_len = len(self.mock_fitter.best) - 1  # Exclude chi2
        self.assertEqual(len(params), expected_len)

        self.assertIn('t_0', params)
        self.assertEqual(params['t_0'], self.mock_fitter.best['t_0'])

        for param in self.mock_fitter.best:
            if param != 'chi2':
                with self.subTest(param=param):
                    self.assertIn(param, params)
                    self.assertEqual(params[param], self.mock_fitter.best[param])

    def test_get_sigmas_from_results(self):
        """Test get_sigmas_from_results extracts uncertainties correctly."""
        result = results.MMEXOFASTFitResults(self.mock_fitter)
        sigmas = result.get_sigmas_from_results()

        self.assertIsInstance(sigmas, dict)
        self.assertEqual(len(sigmas), len(self.mock_fitter.parameters_to_fit))

        self.assertIn('t_0', sigmas)
        self.assertAlmostEqual(sigmas['t_0'], 0.1)

        for i, param in enumerate(self.mock_fitter.parameters_to_fit):
            with self.subTest(param=param):
                self.assertIn(param, sigmas)
                self.assertEqual(sigmas[param], self.mock_fitter.results.sigmas[i])

    def _assert_dataframe_format(self, df, expected_param_count):
        """Helper method to validate DataFrame structure and content."""
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn('parameter_names', df.columns)
        self.assertIn('values', df.columns)
        self.assertIn('sigmas', df.columns)
        self.assertEqual(len(df), expected_param_count)

    def test_format_results_as_df(self):
        """Test format_results_as_df creates proper DataFrame."""
        result = results.MMEXOFASTFitResults(self.mock_fitter)
        df = result.format_results_as_df()

        expected_length = (
            len(self.mock_fitter.best) - 1          # model params (no chi2)
            + 2 * len(self.mock_fitter.datasets)     # flux params
            + 2                                       # chi2 + N_data
        )
        self._assert_dataframe_format(df, expected_length)

        param_names = df['parameter_names'].tolist()
        for param in self.mock_fitter.best:
            if param != 'chi2':
                with self.subTest(param=param):
                    self.assertIn(param, param_names)

        self.assertIn('chi2', param_names)

        t_0_row = df[df['parameter_names'] == 't_0']
        self.assertEqual(len(t_0_row), 1)
        self.assertAlmostEqual(t_0_row['values'].iloc[0], self.mock_fitter.best['t_0'])
        self.assertAlmostEqual(t_0_row['sigmas'].iloc[0], 0.1)


class TestFitRecord(TestMMEXOFASTFitResults):
    """Test FitRecord class."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()

        self.full_result = results.MMEXOFASTFitResults(self.mock_fitter)

        self.model_key = fit_types.FitKey(
            lens_type=fit_types.LensType.POINT,
            source_type=fit_types.SourceType.POINT,
            parallax_branch=fit_types.ParallaxBranch.NONE,
            lens_orb_motion=fit_types.LensOrbMotion.NONE,
        )

        self.test_renorm_factors = {'factor1': 1.5, 'factor2': 0.9}

    def test_from_full_result_populates_all_fields(self):
        """Test that from_full_result() populates all fields correctly."""
        record = results.FitRecord.from_full_result(
            model_key=self.model_key,
            full_result=self.full_result,
            renorm_factors=self.test_renorm_factors,
            fixed=False,
        )

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
        self._assert_dataframe_format(df, len(self.params))
        self.assertEqual(list(df['parameter_names']), list(self.params.keys()))
        self.assertEqual(list(df['values']), list(self.params.values()))
        self.assertEqual(list(df['sigmas']), list(self.sigmas.values()))

    def test_from_full_result_sigmas_exception(self):
        """Test that sigmas is None when get_sigmas_from_results() raises exception."""
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
        self._assert_dataframe_format(df, len(self.params))
        self.assertEqual(list(df['parameter_names']), list(self.params.keys()))
        self.assertEqual(list(df['values']), list(self.params.values()))
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
        self.assertEqual(self.all_results[new_key], new_record)
        self.assertEqual(len(self.all_results), 4)

    def test_setitem_overwrite_record(self):
        """Test __setitem__ overwrites an existing record."""
        new_record = results.FitRecord.from_full_result(
            model_key=self.model_key_pspl_static,
            full_result=self.full_result,
            fixed=True,
        )

        original_length = len(self.all_results)
        self.all_results[self.model_key_pspl_static] = new_record

        retrieved = self.all_results[self.model_key_pspl_static]
        self.assertEqual(retrieved, new_record)
        self.assertEqual(len(self.all_results), original_length)
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
        self.assertEqual(len(self.all_results), 3)

        keys_from_iter = list(self.all_results)
        self.assertEqual(len(keys_from_iter), 3)

        self.assertIn(self.model_key_pspl_static, keys_from_iter)
        self.assertIn(self.model_key_pspl_parallax, keys_from_iter)
        self.assertIn(self.model_key_binary, keys_from_iter)

        self.assertIn(self.model_key_pspl_static, self.all_results)
        self.assertIn(self.model_key_pspl_parallax, self.all_results)
        self.assertIn(self.model_key_binary, self.all_results)

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
        retrieved_by_label = self.all_results[label]
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
        self.assertIsNone(self.all_results.get(fake_key))

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
        self.assertIsNone(self.all_results.get(fake_key))

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

        self.all_results.set(new_record)
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



ANOMALY_LC_PARAMS = {
    't_0':   2459846.8256940236,
    't_E':   6.092881664273319,
    'u_0':   0.8565730569521676,
    'dmag': -0.14566435684576362,
    'dt':    0.17894899984821677,
    't_pl':  2459840.6340004997,
}


class TestIntermediateResults(unittest.TestCase):

    def test_all_fields_default_to_none(self):
        """
        All fields of IntermediateResults default to None on construction.
        """
        ir = results.IntermediateResults()
        self.assertIsNone(ir.best_ef_grid_point)
        self.assertIsNone(ir.best_af_grid_point)
        self.assertIsNone(ir.est_pl_params)
        self.assertIsNone(ir.est_binary_params)
        self.assertIsNone(ir.anomaly_type)
        self.assertIsNone(ir.anomaly_lc_params)

    def test_best_ef_grid_point_can_be_set(self):
        """
        best_ef_grid_point can be set and retrieved.
        """
        ir = results.IntermediateResults()
        value = {
            't_0':   2456836.080383359,
            't_eff': 23.67696884508345,
            'j':     2,
            'chi2':  -137842.8089725696,
        }
        ir.best_ef_grid_point = value
        self.assertEqual(ir.best_ef_grid_point, value)

    def test_best_af_grid_point_can_be_set(self):
        """
        best_af_grid_point can be set and retrieved.
        """
        ir = results.IntermediateResults()
        value = {'t_0': 2456836.0, 'chi2': -1000.0}
        ir.best_af_grid_point = value
        self.assertEqual(ir.best_af_grid_point, value)

    def test_est_pl_params_can_be_set(self):
        """
        est_pl_params can be set and retrieved.
        """
        ir = results.IntermediateResults()
        value = {'t_0': 2456836., 'u_0': 1.012, 't_E': 21.48}
        ir.est_pl_params = value
        self.assertEqual(ir.est_pl_params, value)

    def test_est_binary_params_can_be_set(self):
        """
        est_binary_params can be set and retrieved.
        """
        ir = results.IntermediateResults()
        value = {
            't_0':   2453582.7281740606,
            'u_0':   0.355227507989543,
            't_E':   11.106795114521415,
            'rho':   0.024632765186197645,
            'q':     7.524529162733864e-05,
            's':     1.6044784697939465,
            'alpha': 157.9506556145345,
        }
        ir.est_binary_params = value
        self.assertEqual(ir.est_binary_params, value)

    def test_valid_anomaly_types_are_expected_values(self):
        """
        VALID_ANOMALY_TYPES contains exactly the expected set of values.
        If this fails, check whether test_anomaly_type_accepts_valid_values
        and test_anomaly_type_rejects_invalid_value need updating.
        """
        self.assertEqual(
            results.IntermediateResults.VALID_ANOMALY_TYPES,
            {'close', 'wide', 'high_mag'})

    def test_anomaly_type_accepts_valid_values(self):
        """
        anomaly_type accepts every value listed in
        IntermediateResults.VALID_ANOMALY_TYPES and round-trips correctly.
        """
        ir = results.IntermediateResults()
        for value in results.IntermediateResults.VALID_ANOMALY_TYPES:
            with self.subTest(value=value):
                ir.anomaly_type = value
                self.assertEqual(ir.anomaly_type, value)

    def test_anomaly_type_rejects_invalid_value(self):
        """
        Assigning an unrecognised string to anomaly_type raises ValueError.
        """
        ir = results.IntermediateResults()
        with self.assertRaises(ValueError):
            ir.anomaly_type = 'planetary'

    def test_anomaly_lc_params_can_be_set(self):
        """
        anomaly_lc_params can be set and retrieved.
        """
        ir = results.IntermediateResults()
        ir.anomaly_lc_params = ANOMALY_LC_PARAMS
        self.assertEqual(ir.anomaly_lc_params, ANOMALY_LC_PARAMS)

    def test_fields_are_independent(self):
        """
        Setting one field does not affect others.
        """
        ir = results.IntermediateResults()
        ir.best_ef_grid_point = {'t_0': 2456836.0}
        self.assertIsNone(ir.best_af_grid_point)
        self.assertIsNone(ir.est_pl_params)
        self.assertIsNone(ir.est_binary_params)
        self.assertIsNone(ir.anomaly_type)
        self.assertIsNone(ir.anomaly_lc_params)

    def test_serialization_roundtrip(self):
        """
        IntermediateResults survives a pickle roundtrip with all fields intact.
        """
        ir = results.IntermediateResults()
        ir.best_ef_grid_point = {'t_0': 2456836.0, 't_eff': 23.67}
        ir.est_pl_params = {'t_0': 2456836., 'u_0': 1.012, 't_E': 21.48}
        ir.anomaly_type = 'close'
        ir.anomaly_lc_params = ANOMALY_LC_PARAMS

        restored = pickle.loads(pickle.dumps(ir))
        self.assertEqual(restored.best_ef_grid_point, ir.best_ef_grid_point)
        self.assertEqual(restored.best_af_grid_point, ir.best_af_grid_point)
        self.assertEqual(restored.est_pl_params, ir.est_pl_params)
        self.assertEqual(restored.est_binary_params, ir.est_binary_params)
        self.assertEqual(restored.anomaly_type, ir.anomaly_type)
        self.assertEqual(restored.anomaly_lc_params, ir.anomaly_lc_params)

    def test_fitter_has_intermediate_results_attribute(self):
        """
        MMEXOFASTFitter has an intermediate_results attribute of type
        IntermediateResults, initialized with all fields None.
        """
        fitter = exozippy.mmexofast.MMEXOFASTFitter(
            files=[os.path.join(
                exozippy.MULENS_DATA_PATH, 'OB140939',
                'n20100310.I.OGLE.OB140939.txt')],
            coords='17:47:12.25 -21:22:58.7',
            fit_type='point lens',
            renormalize_errors=False)

        self.assertIsInstance(fitter.intermediate_results, results.IntermediateResults)
        self.assertIsNone(fitter.intermediate_results.best_ef_grid_point)
        self.assertIsNone(fitter.intermediate_results.best_af_grid_point)
        self.assertIsNone(fitter.intermediate_results.est_pl_params)
        self.assertIsNone(fitter.intermediate_results.est_binary_params)
        self.assertIsNone(fitter.intermediate_results.anomaly_type)
        self.assertIsNone(fitter.intermediate_results.anomaly_lc_params)

    def test_intermediate_results_restored_from_state(self):
        """
        intermediate_results is correctly restored by _restore_state().
        """
        fitter = exozippy.mmexofast.MMEXOFASTFitter(
            files=[os.path.join(
                exozippy.MULENS_DATA_PATH, 'OB140939',
                'n20100310.I.OGLE.OB140939.txt')],
            coords='17:47:12.25 -21:22:58.7',
            fit_type='point lens',
            renormalize_errors=False)

        ir = results.IntermediateResults()
        ir.best_ef_grid_point = {'t_0': 2456836.0}
        ir.est_pl_params = {'t_0': 2456836., 'u_0': 1.012, 't_E': 21.48}
        ir.anomaly_type = 'wide'
        ir.anomaly_lc_params = ANOMALY_LC_PARAMS

        fitter._restore_state({'intermediate_results': ir})

        self.assertEqual(
            fitter.intermediate_results.best_ef_grid_point,
            {'t_0': 2456836.0})
        self.assertEqual(
            fitter.intermediate_results.est_pl_params,
            {'t_0': 2456836., 'u_0': 1.012, 't_E': 21.48})
        self.assertEqual(
            fitter.intermediate_results.anomaly_type, 'wide')
        self.assertEqual(
            fitter.intermediate_results.anomaly_lc_params, ANOMALY_LC_PARAMS)
        self.assertIsNone(fitter.intermediate_results.best_af_grid_point)
        self.assertIsNone(fitter.intermediate_results.est_binary_params)


if __name__ == '__main__':
    unittest.main()