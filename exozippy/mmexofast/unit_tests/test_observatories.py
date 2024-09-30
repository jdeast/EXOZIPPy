import os.path
import unittest

#from exozippy import MULENS_DATA_PATH
import exozippy
from exozippy.mmexofast import observatories


def test_get_kwargs():
    filename = os.path.join(
            exozippy.MULENS_DATA_PATH, '2018DataChallenge',
            'n20180816.Z087.WFIRST18.004.txt')
    expected = {
        'phot_fmt': 'mag', 'usecols': [0, 1, 2],
        'ephemerides_file': os.path.join(
            exozippy.MULENS_DATA_PATH, '2018DataChallenge',
            'wfirst_ephemeris_W149.txt'),
        'bandpass': 'Z087',
        'plot_properties': {'zorder': 5, 'label': 'WFIRST18-Z087', 'color': 'blue',
                'marker': 's'}}

    results = observatories.get_kwargs(filename)
    print('results', results)

    for key, value in expected.items():
        print(key, value)
        if key != 'plot_properties':
            assert results[key] == value
        else:
            for plot_key, plot_value in value.items():
                print(plot_key, plot_value)
                assert results[key][plot_key] == plot_value


class TestGetTelescopeBandFromilename(unittest.TestCase):

    def test_get_telescope_band_from_filename(self):
        telescope, band = observatories.get_telescope_band_from_filename(
            os.path.join(
                exozippy.MULENS_DATA_PATH, '2018DataChallenge',
                'n20180816.W149.WFIRST18.004.txt')
        )
        assert telescope == 'WFIRST18'
        assert band == 'W149'

    def test_Error(self):
        with self.assertRaises(ValueError):
            telescope, band = observatories.get_telescope_band_from_filename(
                os.path.join(
                    exozippy.MULENS_DATA_PATH, '2018DataChallenge',
                    'ulwdc1_001_W149.txt')
            )


def test_get_observatory_kwargs():
    telescope = 'WFIRST18'
    expected = {
        'phot_fmt': 'mag', 'usecols': [0, 1, 2],
        'ephemerides_file': os.path.join(
         exozippy.MULENS_DATA_PATH, '2018DataChallenge',
            'wfirst_ephemeris_W149.txt')}

    results = observatories.get_observatory_kwargs(telescope)
    for key, value in expected.items():
        print(key, value)
        assert value == results[key]


def test_get_plot_properties_Z087():
    telescope = 'WFIRST18'
    band = 'Z087'

    expected = {'zorder': 5, 'label': 'WFIRST18-Z087', 'color': 'blue',
                'marker': 's'}

    results = observatories.get_plot_properties(telescope, band)
    for key, value in expected.items():
        print(key, value)
        assert value == results[key]


def test_get_plot_properties_W149():
    telescope = 'WFIRST18'
    band = 'W149'

    expected = {'label': 'WFIRST18-W149', 'color': 'magenta',
                'marker': 'o'}

    results = observatories.get_plot_properties(telescope, band)
    for key, value in expected.items():
        print(key, value)
        assert value == results[key]
