import unittest


class TestMMEXOFASTFitter(unittest.TestCase):

    def test_fit(self):
        self.skipTest('Not Implemented')

    def test_do_ef_grid_search(self):
        self.skipTest('Not Implemented')

    def test_get_initial_pspl_params(self):
        self.skipTest('Not Implemented')

    def test_do_sfit(self):
        self.skipTest('Not Implemented')

    def test_do_mmexofast_fit(self):
        self.skipTest('Not Implemented')

    def test_set_datasets_with_anomaly_masked(self):
        self.skipTest('Not Implemented')

    def test_get_residuals_mask(self):
        self.skipTest('Not Implemented')

    def test_refine_pspl_params(self):
        self.skipTest('Not Implemented')

    def test_set_residuals(self):
        self.skipTest('Not Implemented')

    def test_do_af_grid_search(self):
        self.skipTest('Not Implemented')

    def test_get_dmag(self):
        self.skipTest('Not Implemented')

    def test_get_initial_2L1S_params(self):
        self.skipTest('Not Implemented')

    def test_residuals(self):
        self.skipTest('Not Implemented')

    def test_residuals_setter(self):
        self.skipTest('Not Implemented')

    def test_masked_datasets(self):
        self.skipTest('Not Implemented')

    def test_masked_datasets_setter(self):
        self.skipTest('Not Implemented')

    def test_best_ef_grid_point(self):
        self.skipTest('Not Implemented')

    def test_best_ef_grid_point_setter(self):
        self.skipTest('Not Implemented')

    def test_pspl_params(self):
        self.skipTest('Not Implemented')

    def test_pspl_params_setter(self):
        self.skipTest('Not Implemented')

    def test_best_af_grid_point(self):
        self.skipTest('Not Implemented')

    def test_best_af_grid_point_setter(self):
        self.skipTest('Not Implemented')

    def test_binary_params(self):
        self.skipTest('Not Implemented')

    def test_binary_params_setter(self):
        self.skipTest('Not Implemented')

    def test_results(self):
        self.skipTest('Not Implemented')

    def test_results_setter(self):
        self.skipTest('Not Implemented')


class TestSatelliteData(unittest.TestCase):

    def setUp(self):
        self.ground_data = None
        self.spitzer_data = None
        self.kepler_data = None
        self.skipTest('Need to setup test datasets for this test (among other things).')

    def do_test_file_list(self, files):
        self.skipTest('Not Implemented')
        fitter = MMEXOFASTFitter(files=files)
        assert fitter.n_loc == len(files)

    def _get_datasets(self, file_list):
        # Do something to read in the datasets
        return datasets

    def do_test_datasets(self, file_list):
        self.skipTest('Not Implemented')
        datasets = self._get_datasets(file_list)
        fitter = MMEXOFASTFitter(datasets=datasets)
        assert fitter.n_loc == len(datasets)

    def test_gr_only(self):
        self.do_test_file_list(self.ground_data)
        self.do_test_datasets(self.ground_data)

    def test_spz_only(self):
        self.do_test_file_list(self.spitzer_data)
        self.do_test_datasets(self.spitzer_data)

    def test_gr_spz(self):
        files = [self.ground_data, self.spitzer_data]
        self.do_test_file_list(files)
        self.do_test_datasets(files)

    def test_kep_spz(self):
        files = [self.kepler_data, self.spitzer_data]
        self.do_test_file_list(files)
        self.do_test_datasets(files)

    def test_all(self):
        files = [self.ground_data, self.kepler_data, self.spitzer_data]
        self.do_test_file_list(files)
        self.do_test_datasets(files)
