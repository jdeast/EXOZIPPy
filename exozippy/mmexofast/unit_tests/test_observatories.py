import os.path
import unittest

#from exozippy import MULENS_DATA_PATH
import exozippy
from exozippy.mmexofast import observatories


class TestGetTelescopeBandFromFilename(unittest.TestCase):
    """Test filename parsing for telescope and band extraction."""

    def test_standard_format_wfirst(self):
        """Test standard format with WFIRST telescope."""
        telescope, band = observatories.get_telescope_band_from_filename(
            os.path.join(
                exozippy.MULENS_DATA_PATH, '2018DataChallenge',
                'n20180816.W149.WFIRST18.004.txt')
        )
        self.assertEqual(telescope, 'WFIRST18')
        self.assertEqual(band, 'W149')

    def test_standard_format_ogle(self):
        """Test standard format with OGLE telescope."""
        telescope, band = observatories.get_telescope_band_from_filename(
            'n20100310.I.OGLE.OB140939.txt'
        )
        self.assertEqual(telescope, 'OGLE')
        self.assertEqual(band, 'I')

    def test_different_bands(self):
        """Test various band identifiers."""
        test_cases = [
            ('n20050724.r.MOA.OB05390.txt', 'MOA', 'r'),
            ('n20050725.R.FTN.OB05390.txt', 'FTN', 'R'),
            ('n20140605.L.Spitzer.OB140939.txt', 'Spitzer', 'L'),
        ]

        for filename, expected_tel, expected_band in test_cases:
            with self.subTest(filename=filename):
                telescope, band = observatories.get_telescope_band_from_filename(filename)
                self.assertEqual(telescope, expected_tel)
                self.assertEqual(band, expected_band)

    def test_different_telescopes(self):
        """Test various telescope identifiers."""
        test_cases = [
            ('n20100101.I.OGLE.test.txt', 'OGLE', 'I'),
            ('n20100101.V.MOA.test.txt', 'MOA', 'V'),
            ('n20100101.I.Danish.test.txt', 'Danish', 'I'),
            ('n20100101.I.Perth.test.txt', 'Perth', 'I'),
        ]

        for filename, expected_tel, expected_band in test_cases:
            with self.subTest(filename=filename):
                telescope, band = observatories.get_telescope_band_from_filename(filename)
                self.assertEqual(telescope, expected_tel)
                self.assertEqual(band, expected_band)

    def test_malformed_filename_no_dots(self):
        """Test error on filename without proper format."""
        with self.assertRaises(ValueError):
            observatories.get_telescope_band_from_filename('ulwdc1_001_W149.txt')

    def test_malformed_filename_too_few_parts(self):
        """Test error on filename with too few dot-separated parts."""
        with self.assertRaises(ValueError):
            observatories.get_telescope_band_from_filename('n20100101.I.txt')


class TestObservatory(unittest.TestCase):
    """Test Observatory class."""

    def test_init_minimal(self):
        """Test Observatory creation with minimal parameters."""
        obs = observatories.Observatory(name='TestObs')
        self.assertEqual(obs.name, 'TestObs')
        self.assertIsNone(obs.ephemerides_file)
        self.assertEqual(obs.phot_fmt, 'flux')  # Default

    def test_init_full(self):
        """Test Observatory creation with all parameters."""
        obs = observatories.Observatory(
            name='Spitzer',
            ephemerides_file='spitzer.txt',
            phot_fmt='mag',
            bands={'L': {'marker': 's', 'color': 'red'}, 'M': {'marker': 'o', 'color': 'blue'}},
            usecols=[0, 1, 2]
        )
        self.assertEqual(obs.name, 'Spitzer')
        self.assertEqual(obs.ephemerides_file, 'spitzer.txt')
        self.assertEqual(obs.phot_fmt, 'mag')
        self.assertIsInstance(obs.bands, dict)
        self.assertIn('L', obs.bands)
        self.assertIsInstance(obs.usecols, list)
        self.assertEqual(obs.usecols, [0, 1, 2])

    def test_get_kwargs_wfirst(self):
        """Test get_kwargs method for WFIRST18."""
        # Get WFIRST from registry
        wfirst = observatories.OBSERVATORIES['WFIRST18']
        kwargs = wfirst.get_kwargs()

        self.assertEqual(kwargs['phot_fmt'], 'flux')
        self.assertEqual(kwargs['usecols'], [0, 1, 2])
        self.assertEqual(kwargs['ephemerides_file'],
                         os.path.join(exozippy.MULENS_DATA_PATH, '2018DataChallenge',
                                      'wfirst_ephemeris_W149.txt'))

    def test_get_kwargs_basic(self):
        """Test get_kwargs method with basic observatory."""
        obs = observatories.Observatory(name='TestObs', phot_fmt='mag')
        kwargs = obs.get_kwargs()

        self.assertIn('phot_fmt', kwargs)
        self.assertEqual(kwargs['phot_fmt'], 'mag')

    def test_get_plot_properties_ogle(self):
        """Test get_plot_properties for OGLE I-band."""
        ogle = observatories.OBSERVATORIES['OGLE']
        props = ogle.get_plot_properties('I')

        self.assertEqual(props['label'], 'OGLE-I')
        self.assertEqual(props['color'], 'black')
        self.assertEqual(props['marker'], 'o')

    def test_get_plot_properties_wfirst_z087(self):
        """Test get_plot_properties for WFIRST Z087 band."""
        wfirst = observatories.OBSERVATORIES['WFIRST18']
        props = wfirst.get_plot_properties('Z087')

        self.assertEqual(props['label'], 'WFIRST18-Z087')
        self.assertEqual(props['color'], 'blue')
        self.assertEqual(props['marker'], 's')
        self.assertEqual(props['zorder'], 5)

    def test_get_plot_properties_wfirst_w149(self):
        """Test get_plot_properties for WFIRST W149 band."""
        wfirst = observatories.OBSERVATORIES['WFIRST18']
        props = wfirst.get_plot_properties('W149')

        self.assertEqual(props['label'], 'WFIRST18-W149')
        self.assertEqual(props['color'], 'magenta')
        self.assertEqual(props['marker'], 'o')


class TestGetKwargs(unittest.TestCase):
    """Test get_kwargs function."""

    def test_known_observatory_wfirst(self):
        """Test kwargs for known observatory (WFIRST18)."""
        filename = os.path.join(
            exozippy.MULENS_DATA_PATH, '2018DataChallenge',
            'n20180816.Z087.WFIRST18.004.txt')

        results = observatories.get_kwargs(filename)

        # Check all non-plot_properties fields
        self.assertEqual(results['phot_fmt'], 'flux')
        self.assertEqual(results['usecols'], [0, 1, 2])
        self.assertEqual(results['ephemerides_file'],
                         os.path.join(exozippy.MULENS_DATA_PATH, '2018DataChallenge',
                                      'wfirst_ephemeris_W149.txt'))
        self.assertEqual(results['bandpass'], 'Z087')

        # Check plot_properties (but label is now filename basename, not TELESCOPE-BAND)
        self.assertIn('plot_properties', results)
        self.assertEqual(results['plot_properties']['zorder'], 5)
        self.assertEqual(results['plot_properties']['color'], 'blue')
        self.assertEqual(results['plot_properties']['marker'], 's')
        self.assertEqual(results['plot_properties']['label'], 'n20180816.Z087.WFIRST18.004.txt')

    def test_known_observatory_ogle(self):
        """Test kwargs for known observatory (OGLE)."""
        filename = 'n20100310.I.OGLE.OB140939.txt'
        kwargs = observatories.get_kwargs(filename)

        # Check basic kwargs
        self.assertIn('phot_fmt', kwargs)
        self.assertIn('bandpass', kwargs)
        self.assertEqual(kwargs['bandpass'], 'I')

        # Check plot_properties explicitly
        self.assertIn('plot_properties', kwargs)
        self.assertEqual(kwargs['plot_properties']['label'], 'n20100310.I.OGLE.OB140939.txt')
        self.assertEqual(kwargs['plot_properties']['color'], 'black')
        self.assertEqual(kwargs['plot_properties']['marker'], 'o')

    def test_known_observatory_spitzer(self):
        """Test kwargs for space observatory (Spitzer)."""
        filename = 'n20140605.L.Spitzer.OB140939.txt'
        kwargs = observatories.get_kwargs(filename)

        # Check basic kwargs
        self.assertIn('ephemerides_file', kwargs)
        self.assertIsNotNone(kwargs['ephemerides_file'])
        self.assertEqual(kwargs['bandpass'], 'L')

        # Check plot_properties explicitly
        self.assertIn('plot_properties', kwargs)
        self.assertEqual(kwargs['plot_properties']['label'], 'n20140605.L.Spitzer.OB140939.txt')
        self.assertEqual(kwargs['plot_properties']['color'], 'red')
        self.assertEqual(kwargs['plot_properties']['marker'], 'o')

    def test_unknown_observatory_defaults(self):
        """Test default kwargs for unknown observatory."""
        filename = 'n20100101.I.UnknownTel.test.txt'
        kwargs = observatories.get_kwargs(filename)

        self.assertEqual(kwargs['phot_fmt'], 'flux')
        self.assertEqual(kwargs['bandpass'], 'I')
        self.assertEqual(kwargs['plot_properties']['label'], 'n20100101.I.UnknownTel.test.txt')
        self.assertEqual(kwargs['plot_properties']['marker'], 'o')

    def test_label_is_basename_with_path(self):
        """Test that label is basename even when full path provided."""
        filename = '/full/path/to/directory/n20100310.I.OGLE.event.txt'
        kwargs = observatories.get_kwargs(filename)

        self.assertEqual(kwargs['plot_properties']['label'], 'n20100310.I.OGLE.event.txt')


class TestObservatoriesRegistry(unittest.TestCase):
    """Test OBSERVATORIES registry and related functions."""

    def test_registry_not_empty(self):
        """Test that OBSERVATORIES registry exists and has entries."""
        self.assertIsInstance(observatories.OBSERVATORIES, dict)
        self.assertGreater(len(observatories.OBSERVATORIES), 0)

    def test_all_observatories_have_required_methods(self):
        """Test that all registered observatories are Observatory instances."""
        for name, obs in observatories.OBSERVATORIES.items():
            with self.subTest(observatory=name):
                self.assertIsInstance(obs, observatories.Observatory)
                self.assertTrue(callable(obs.get_kwargs))
                self.assertTrue(callable(obs.get_plot_properties))

    def test_ephemerides_files_exist(self):
        """Test that ephemerides files exist for space observatories."""
        for name, obs in observatories.OBSERVATORIES.items():
            if obs.ephemerides_file is not None:
                with self.subTest(observatory=name):
                    # Check if it's a full path or relative
                    if os.path.isabs(obs.ephemerides_file):
                        self.assertTrue(os.path.exists(obs.ephemerides_file),
                                        f"Ephemerides file not found: {obs.ephemerides_file}")

    def test_ephemerides_to_observatory_mapping(self):
        """Test EPHEMERIDES_TO_OBSERVATORY mapping consistency."""
        if hasattr(observatories, 'EPHEMERIDES_TO_OBSERVATORY'):
            self.assertIsInstance(observatories.EPHEMERIDES_TO_OBSERVATORY, dict)

            # Check reverse mapping consistency
            for ephem, obs_name in observatories.EPHEMERIDES_TO_OBSERVATORY.items():
                with self.subTest(ephemeris=ephem):
                    self.assertIn(obs_name, observatories.OBSERVATORIES)
                    # Check that observatory actually has this ephemerides file
                    self.assertEqual(observatories.OBSERVATORIES[obs_name].ephemerides_file, ephem)


class TestObservatoryManagement(unittest.TestCase):
    """Test observatory registration and retrieval functions."""

    def test_register_observatory(self):
        """Test registering a new observatory."""
        if not hasattr(observatories, 'register_observatory'):
            self.skipTest("register_observatory not implemented")

        # Create test observatory
        test_obs = observatories.Observatory(name='TestObs', phot_fmt='flux')
        observatories.register_observatory('TestObs', test_obs)

        # Check it was added
        self.assertIn('TestObs', observatories.OBSERVATORIES)
        self.assertEqual(observatories.OBSERVATORIES['TestObs'], test_obs)

        # Cleanup
        del observatories.OBSERVATORIES['TestObs']

    def test_list_observatories(self):
        """Test listing all registered observatories."""
        if not hasattr(observatories, 'list_observatories'):
            self.skipTest("list_observatories not implemented")

        obs_list = observatories.list_observatories()
        self.assertIsInstance(obs_list, list)
        self.assertGreater(len(obs_list), 0)
        # Should include known observatories
        self.assertIn('OGLE', obs_list)
        self.assertIn('Spitzer', obs_list)
        self.assertIn('WFIRST18', obs_list)

    def test_get_observatory_exists(self):
        """Test retrieving an existing observatory."""
        if not hasattr(observatories, 'get_observatory'):
            self.skipTest("get_observatory not implemented")

        obs = observatories.get_observatory('OGLE')
        self.assertIsInstance(obs, observatories.Observatory)
        self.assertEqual(obs.name, 'OGLE')

    def test_get_observatory_not_exists(self):
        """Test retrieving non-existent observatory returns None."""
        if not hasattr(observatories, 'get_observatory'):
            self.skipTest("get_observatory not implemented")

        result = observatories.get_observatory('NonExistentObservatory')
        self.assertIsNone(result)


class TestValidateFilename(unittest.TestCase):
    """Test filename validation."""

    def test_valid_filename_standard(self):
        """Test validation of correct filename format."""
        if not hasattr(observatories, 'validate_filename'):
            self.skipTest("validate_filename not implemented")

        valid_filenames = [
            'n20100310.I.OGLE.event.txt',
            'n20140605.L.Spitzer.OB140939.txt',
            'n20180816.W149.WFIRST18.004.txt',
        ]

        for filename in valid_filenames:
            with self.subTest(filename=filename):
                self.assertTrue(observatories.validate_filename(filename))

    def test_invalid_filename_formats(self):
        """Test validation rejects incorrect formats."""
        if not hasattr(observatories, 'validate_filename'):
            self.skipTest("validate_filename not implemented")

        invalid_filenames = [
            'bad_format.txt',
            'n20100310.txt',  # Too few parts
            'ulwdc1_001_W149.txt',  # Wrong format
            'n20100310.I.txt',  # Missing telescope
        ]

        for filename in invalid_filenames:
            with self.subTest(filename=filename):
                self.assertFalse(observatories.validate_filename(filename))


class TestLoadObservatoriesFromConfig(unittest.TestCase):
    """Test loading observatory configuration."""

    def test_load_from_config(self):
        """Test loading observatories from config file/dict."""
        if not hasattr(observatories, 'load_observatories_from_config'):
            self.skipTest("load_observatories_from_config not implemented")

        # TODO: Implement based on actual config format
        # This will depend on how config is structured (file, dict, etc.)
        pass


def test_get_observatory_kwargs():
    telescope = 'WFIRST18'
    expected = {
        'phot_fmt': 'flux', 'usecols': [0, 1, 2],
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
