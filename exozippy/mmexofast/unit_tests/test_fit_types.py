"""Unit tests for fit_types module."""

import unittest

from exozippy.mmexofast import fit_types


class TestEnums(unittest.TestCase):
    """Test enum definitions."""

    def test_lens_type_values(self):
        """Test LensType enum has expected values."""
        expected_values = ['POINT', 'BINARY']

        for value in expected_values:
            with self.subTest(value=value):
                self.assertTrue(hasattr(fit_types.LensType, value))

    def test_source_type_values(self):
        """Test SourceType enum has expected values."""
        expected_values = ['POINT', 'FINITE']

        for value in expected_values:
            with self.subTest(value=value):
                self.assertTrue(hasattr(fit_types.SourceType, value))

    def test_parallax_branch_values(self):
        """Test ParallaxBranch enum has expected values."""
        expected_values = ['NONE', 'U0_PLUS', 'U0_MINUS', 'U0_PP', 'U0_PM', 'U0_MP', 'U0_MM']

        for value in expected_values:
            with self.subTest(value=value):
                self.assertTrue(hasattr(fit_types.ParallaxBranch, value))

    def test_lens_orb_motion_values(self):
        """Test LensOrbMotion enum has expected values."""
        expected_values = ['NONE', 'KEPLER', 'ORB_2D']  # Adjust based on actual values

        for value in expected_values:
            with self.subTest(value=value):
                self.assertTrue(hasattr(fit_types.LensOrbMotion, value))


class TestTagConstants(unittest.TestCase):
    """Test TAG constant dictionaries."""

    def test_lens_tags_exist(self):
        """Test LENS_TAGS dictionary exists and maps correctly."""
        self.assertIsInstance(fit_types.LENS_TAGS, dict)
        self.assertGreater(len(fit_types.LENS_TAGS), 0)

        # Check it maps to LensType enum values
        for tag, lens_type in fit_types.LENS_TAGS.items():
            self.assertIsInstance(tag, str)
            self.assertIsInstance(lens_type, fit_types.LensType)

    def test_source_tags_exist(self):
        """Test SOURCE_TAGS dictionary exists and maps correctly."""
        self.assertIsInstance(fit_types.SOURCE_TAGS, dict)
        self.assertGreater(len(fit_types.SOURCE_TAGS), 0)

        for tag, source_type in fit_types.SOURCE_TAGS.items():
            self.assertIsInstance(tag, str)
            self.assertIsInstance(source_type, fit_types.SourceType)

    def test_parallax_branch_tags_exist(self):
        """Test PARALLAX_BRANCH_TAGS dictionary exists and maps correctly."""
        self.assertIsInstance(fit_types.PARALLAX_BRANCH_TAGS, dict)
        self.assertGreater(len(fit_types.PARALLAX_BRANCH_TAGS), 0)

        for tag, branch in fit_types.PARALLAX_BRANCH_TAGS.items():
            self.assertIsInstance(tag, str)
            self.assertIsInstance(branch, fit_types.ParallaxBranch)

    def test_lens_motion_tags_exist(self):
        """Test LENS_MOTION_TAGS dictionary exists and maps correctly."""
        self.assertIsInstance(fit_types.LENS_MOTION_TAGS, dict)
        self.assertGreater(len(fit_types.LENS_MOTION_TAGS), 0)

        for tag, motion in fit_types.LENS_MOTION_TAGS.items():
            self.assertIsInstance(tag, str)
            self.assertIsInstance(motion, fit_types.LensOrbMotion)


class TestFitKey(unittest.TestCase):
    """Test FitKey dataclass."""

    def test_creation_minimal(self):
        """Test FitKey creation with required fields."""
        key = fit_types.FitKey(
            lens_type=fit_types.LensType.POINT,
            source_type=fit_types.SourceType.POINT,
            parallax_branch=fit_types.ParallaxBranch.NONE,
            lens_orb_motion=fit_types.LensOrbMotion.NONE
        )

        self.assertEqual(key.lens_type, fit_types.LensType.POINT)
        self.assertEqual(key.source_type, fit_types.SourceType.POINT)
        self.assertEqual(key.parallax_branch, fit_types.ParallaxBranch.NONE)
        self.assertEqual(key.lens_orb_motion, fit_types.LensOrbMotion.NONE)
        self.assertIsNone(key.locations_used)

    def test_creation_with_locations(self):
        """Test FitKey creation with locations_used."""
        key = fit_types.FitKey(
            lens_type=fit_types.LensType.POINT,
            source_type=fit_types.SourceType.POINT,
            parallax_branch=fit_types.ParallaxBranch.U0_PLUS,
            lens_orb_motion=fit_types.LensOrbMotion.NONE,
            locations_used='ground+Spitzer'
        )

        self.assertEqual(key.locations_used, 'ground+Spitzer')

    def test_equality_same_values(self):
        """Test FitKey equality for identical keys."""
        key1 = fit_types.FitKey(
            lens_type=fit_types.LensType.POINT,
            source_type=fit_types.SourceType.POINT,
            parallax_branch=fit_types.ParallaxBranch.NONE,
            lens_orb_motion=fit_types.LensOrbMotion.NONE
        )
        key2 = fit_types.FitKey(
            lens_type=fit_types.LensType.POINT,
            source_type=fit_types.SourceType.POINT,
            parallax_branch=fit_types.ParallaxBranch.NONE,
            lens_orb_motion=fit_types.LensOrbMotion.NONE
        )

        self.assertEqual(key1, key2)

    def test_equality_different_values(self):
        """Test FitKey inequality for different keys."""
        key1 = fit_types.FitKey(
            lens_type=fit_types.LensType.POINT,
            source_type=fit_types.SourceType.POINT,
            parallax_branch=fit_types.ParallaxBranch.NONE,
            lens_orb_motion=fit_types.LensOrbMotion.NONE
        )
        key2 = fit_types.FitKey(
            lens_type=fit_types.LensType.POINT,
            source_type=fit_types.SourceType.FINITE,
            parallax_branch=fit_types.ParallaxBranch.NONE,
            lens_orb_motion=fit_types.LensOrbMotion.NONE
        )

        self.assertNotEqual(key1, key2)

    def test_hashability(self):
        """Test FitKey can be used as dict key."""
        key1 = fit_types.FitKey(
            lens_type=fit_types.LensType.POINT,
            source_type=fit_types.SourceType.POINT,
            parallax_branch=fit_types.ParallaxBranch.U0_PLUS,
            lens_orb_motion=fit_types.LensOrbMotion.NONE
        )
        key2 = fit_types.FitKey(
            lens_type=fit_types.LensType.POINT,
            source_type=fit_types.SourceType.POINT,
            parallax_branch=fit_types.ParallaxBranch.U0_PLUS,
            lens_orb_motion=fit_types.LensOrbMotion.NONE
        )

        # Should be able to use as dict keys
        test_dict = {key1: 'value1'}
        self.assertEqual(test_dict[key2], 'value1')  # key2 should match key1

    def test_string_representation(self):
        """Test FitKey string representation for various combinations."""
        test_cases = [
            # (key_params, expected_substrings)
            (
                {
                    'lens_type': fit_types.LensType.POINT,
                    'source_type': fit_types.SourceType.POINT,
                    'parallax_branch': fit_types.ParallaxBranch.NONE,
                    'lens_orb_motion': fit_types.LensOrbMotion.NONE
                },
                ['POINT', 'NONE']
            ),
            (
                {
                    'lens_type': fit_types.LensType.POINT,
                    'source_type': fit_types.SourceType.FINITE,
                    'parallax_branch': fit_types.ParallaxBranch.U0_PLUS,
                    'lens_orb_motion': fit_types.LensOrbMotion.NONE
                },
                ['POINT', 'FINITE', 'U0_PLUS']
            ),
            (
                {
                    'lens_type': fit_types.LensType.BINARY,
                    'source_type': fit_types.SourceType.POINT,
                    'parallax_branch': fit_types.ParallaxBranch.U0_MM,
                    'lens_orb_motion': fit_types.LensOrbMotion.KEPLER
                },
                ['BINARY', 'U0_MM', 'KEPLER']
            ),
            (
                {
                    'lens_type': fit_types.LensType.POINT,
                    'source_type': fit_types.SourceType.POINT,
                    'parallax_branch': fit_types.ParallaxBranch.NONE,
                    'lens_orb_motion': fit_types.LensOrbMotion.NONE,
                    'locations_used': 'ground+Spitzer'
                },
                ['POINT', 'NONE', 'ground+Spitzer']
            ),
        ]

        for params, expected_substrings in test_cases:
            with self.subTest(params=params):
                key = fit_types.FitKey(**params)
                key_str = str(key)

                self.assertIsInstance(key_str, str)
                for substring in expected_substrings:
                    self.assertIn(substring, key_str)


class TestLabelConversions(unittest.TestCase):
    """Test label <-> model key conversion functions."""

    def test_label_to_key_pspl_static(self):
        """Test converting 'PSPL static' label."""
        key = fit_types.label_to_model_key('PSPL static')

        self.assertEqual(key.lens_type, fit_types.LensType.POINT)
        self.assertEqual(key.source_type, fit_types.SourceType.POINT)
        self.assertEqual(key.parallax_branch, fit_types.ParallaxBranch.NONE)
        self.assertEqual(key.lens_orb_motion, fit_types.LensOrbMotion.NONE)
        self.assertIsNone(key.locations_used)

    def test_label_to_key_fspl_static(self):
        """Test converting 'FSPL static' label."""
        key = fit_types.label_to_model_key('FSPL static')

        self.assertEqual(key.lens_type, fit_types.LensType.POINT)
        self.assertEqual(key.source_type, fit_types.SourceType.FINITE)
        self.assertEqual(key.parallax_branch, fit_types.ParallaxBranch.NONE)
        self.assertEqual(key.lens_orb_motion, fit_types.LensOrbMotion.NONE)
        self.assertIsNone(key.locations_used)

    def test_label_to_key_parallax_u0_plus(self):
        """Test converting parallax label 'PSPL par u0+'."""
        key = fit_types.label_to_model_key('PSPL par u0+')

        self.assertEqual(key.lens_type, fit_types.LensType.POINT)
        self.assertEqual(key.source_type, fit_types.SourceType.POINT)
        self.assertEqual(key.parallax_branch, fit_types.ParallaxBranch.U0_PLUS)
        self.assertEqual(key.lens_orb_motion, fit_types.LensOrbMotion.NONE)
        self.assertIsNone(key.locations_used)

    def test_label_to_key_parallax_u0_minus(self):
        """Test converting parallax label 'PSPL par u0-'."""
        key = fit_types.label_to_model_key('PSPL par u0-')


        self.assertEqual(key.lens_type, fit_types.LensType.POINT)
        self.assertEqual(key.source_type, fit_types.SourceType.POINT)
        self.assertEqual(key.parallax_branch, fit_types.ParallaxBranch.U0_MINUS)
        self.assertEqual(key.lens_orb_motion, fit_types.LensOrbMotion.NONE)
        self.assertIsNone(key.locations_used)

    def test_label_to_key_parallax_multi_loc(self):
        """Test converting multi-location parallax labels."""
        test_cases = [
            ('PSPL par u0++', fit_types.ParallaxBranch.U0_PP),
            ('PSPL par u0--', fit_types.ParallaxBranch.U0_MM),
            ('PSPL par u0+-', fit_types.ParallaxBranch.U0_PM),
            ('PSPL par u0-+', fit_types.ParallaxBranch.U0_MP),
        ]

        for label, expected_branch in test_cases:
            with self.subTest(label=label):
                key = fit_types.label_to_model_key(label)
                self.assertEqual(key.parallax_branch, expected_branch)

    def test_label_to_key_with_locations(self):
        """Test converting label with locations specified."""
        key = fit_types.label_to_model_key('PSPL static (ground+Spitzer)')

        self.assertEqual(key.parallax_branch, fit_types.ParallaxBranch.NONE)
        self.assertEqual(key.locations_used, 'ground+Spitzer')

    def test_label_to_key_invalid(self):
        """Test that invalid labels raise appropriate error."""
        with self.assertRaises((ValueError, KeyError)):
            fit_types.label_to_model_key('InvalidLabel')

    def test_key_to_label_pspl_static(self):
        """Test converting static PSPL key to label."""
        key = fit_types.FitKey(
            lens_type=fit_types.LensType.POINT,
            source_type=fit_types.SourceType.POINT,
            parallax_branch=fit_types.ParallaxBranch.NONE,
            lens_orb_motion=fit_types.LensOrbMotion.NONE
        )

        label = fit_types.model_key_to_label(key)
        self.assertEqual(label, 'PSPL static')

    def test_key_to_label_with_locations(self):
        """Test converting key with locations to label."""
        key = fit_types.FitKey(
            lens_type=fit_types.LensType.POINT,
            source_type=fit_types.SourceType.POINT,
            parallax_branch=fit_types.ParallaxBranch.NONE,
            lens_orb_motion=fit_types.LensOrbMotion.NONE,
            locations_used='ground+Spitzer'
        )

        label = fit_types.model_key_to_label(key)
        self.assertIn('ground+Spitzer', label)
        self.assertIn('PSPL', label)

    def test_round_trip_conversion(self):
        """Test that label -> key -> label is consistent."""
        original_labels = [
            'PSPL static',
            'FSPL static',
            'PSPL par u0+',
            'PSPL par u0-',
            'PSPL par u0++',
            'PSPL static (ground)',
            'PSPL par u0+ (ground+Spitzer)',
            '2L1S 2Dorb',  # 2DORB case
            '2L1S kep par u0+',  # Kepler + parallax case (adjust format as needed)
        ]

        for label in original_labels:
            with self.subTest(label=label):
                key = fit_types.label_to_model_key(label)
                back_to_label = fit_types.model_key_to_label(key)
                # Convert back to key to ensure semantic equivalence
                key2 = fit_types.label_to_model_key(back_to_label)
                self.assertEqual(key, key2)

    def test_all_enum_combinations(self):
        """Test key to label conversion for various combinations."""
        test_cases = [
            # (lens, source, parallax, motion, expected_label_parts)
            (fit_types.LensType.POINT, fit_types.SourceType.POINT,
             fit_types.ParallaxBranch.NONE, fit_types.LensOrbMotion.NONE,
             ['PSPL', 'static']),

            (fit_types.LensType.POINT, fit_types.SourceType.FINITE,
             fit_types.ParallaxBranch.NONE, fit_types.LensOrbMotion.NONE,
             ['FSPL', 'static']),

            (fit_types.LensType.POINT, fit_types.SourceType.POINT,
             fit_types.ParallaxBranch.U0_PLUS, fit_types.LensOrbMotion.NONE,
             ['PSPL', 'par', 'u0+']),

            (fit_types.LensType.POINT, fit_types.SourceType.FINITE,
             fit_types.ParallaxBranch.U0_MINUS, fit_types.LensOrbMotion.NONE,
             ['FSPL', 'par', 'u0-']),

            (fit_types.LensType.BINARY, fit_types.SourceType.POINT,
             fit_types.ParallaxBranch.NONE, fit_types.LensOrbMotion.KEPLER,
             ['2L1S', 'kep']),

            (fit_types.LensType.BINARY, fit_types.SourceType.POINT,
             fit_types.ParallaxBranch.NONE, fit_types.LensOrbMotion.ORB_2D,
             ['2L1S', '2Dorb']),
        ]

        for lens, source, parallax, motion, expected_parts in test_cases:
            with self.subTest(lens=lens, source=source, parallax=parallax, motion=motion):
                key = fit_types.FitKey(
                    lens_type=lens,
                    source_type=source,
                    parallax_branch=parallax,
                    lens_orb_motion=motion
                )

                label = fit_types.model_key_to_label(key)

                # Check that all expected parts are in the label
                for part in expected_parts:
                    self.assertIn(part, label)

    def test_locations_none_vs_absent(self):
        """Test that None locations_used handled correctly."""
        key_with_none = fit_types.FitKey(
            lens_type=fit_types.LensType.POINT,
            source_type=fit_types.SourceType.POINT,
            parallax_branch=fit_types.ParallaxBranch.NONE,
            lens_orb_motion=fit_types.LensOrbMotion.NONE,
            locations_used=None
        )

        label = fit_types.model_key_to_label(key_with_none)
        self.assertIsInstance(label, str)
        # Should not have location info in label
        self.assertNotIn('(', label)


class TestInvalidLabels(unittest.TestCase):
    """Test error handling for invalid labels."""

    def test_empty_label(self):
        """Test that empty label raises error."""
        with self.assertRaises((ValueError, KeyError)):
            fit_types.label_to_model_key('')

    def test_label_with_nonexistent_keys(self):
        """Test that labels with invalid keys raise error."""
        invalid_labels = [
            'InvalidLens static',
            'PSPL invalidkey',
            'PSPL par invalidbranch',
        ]

        for label in invalid_labels:
            with self.subTest(label=label):
                with self.assertRaises((ValueError, KeyError)):
                    fit_types.label_to_model_key(label)

    def test_point_lens_with_orbital_motion(self):
        """Test that point lens with orbital motion raises error."""
        invalid_combinations = [
            'PSPL kep',
            'PSPL 2Dorb',
            'FSPL kep',
        ]

        for label in invalid_combinations:
            with self.subTest(label=label):
                with self.assertRaises((ValueError, KeyError)):
                    fit_types.label_to_model_key(label)


if __name__ == '__main__':
    unittest.main()
