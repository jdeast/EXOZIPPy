import unittest


class TestEventFinderGridSearch(unittest.TestCase):

    def test_run(self):
        raise NotImplementedError()

    def test_get_trimmed_datasets(self):
        raise NotImplementedError()

    def get_flat_chi2(self):
        raise NotImplementedError()

    def test_do_fits(self):
        raise NotImplementedError()

    def test_grid_t_0(self):
        raise NotImplementedError()

    def test_grid_t_eff(self):
        raise NotImplementedError()

    def test_best(self):
        raise NotImplementedError()


class TestFlatSFitFunction(unittest.TestCase):

    def test_flatten_data(self):
        raise NotImplementedError()

    def test_calc_model(self):
        raise NotImplementedError()

    def test_calc_df(self):
        raise NotImplementedError()


class TestEFSFitFunction():

    def test_calc_model(self):
        raise NotImplementedError()

    def test_calc_residuals(self):
        raise NotImplementedError()

    def test_calc_df(self):
        raise NotImplementedError()

    def test_q(self):
        raise NotImplementedError()

    def test_magnification(self):
        raise NotImplementedError()

    def test_get_magnification(self):
        raise NotImplementedError()


class TestAnomalyFinderGridSearch(unittest.TestCase):

    def test_run(self):
        raise NotImplementedError()

    def test_get_zero_chi2(self):
        raise NotImplementedError()

    def test_check_successive(self):
        raise NotImplementedError()

    def test_do_fits(self):
        raise NotImplementedError()

    def test_get_anomalies(self):
        raise NotImplementedError()

    def test_filter_anomalies(self):
        raise NotImplementedError()

    def test_anomalies(self):
        raise NotImplementedError()

    def test_best(self):
        raise NotImplementedError()
    