"""Unit tests for dataset initialization and merging."""

import unittest
import tempfile
import os
import pickle
from pathlib import Path

import MulensModel

from exozippy import MULENS_DATA_PATH
from exozippy.mmexofast import MMEXOFASTFitter


class TestDatasetInitialization(unittest.TestCase):
    """Test dataset initialization from files, datasets, and pickles."""

    @classmethod
    def setUpClass(cls):
        """Set up test data paths."""
        cls.ob140939_dir = Path(MULENS_DATA_PATH) / 'OB140939'
        cls.ob05390_dir = Path(MULENS_DATA_PATH) / 'OB05390'

        # Read coordinates
        with open(cls.ob140939_dir / 'coords.txt') as f:
            cls.ob140939_coords = f.read().strip()
        with open(cls.ob05390_dir / 'coords.txt') as f:
            cls.ob05390_coords = f.read().strip()

        # Define file paths
        cls.ob140939_ogle = str(cls.ob140939_dir / 'n20100310.I.OGLE.OB140939.txt')
        cls.ob140939_spitzer = str(cls.ob140939_dir / 'n20140605.L.Spitzer.OB140939.txt')

        cls.ob05390_ogle = str(cls.ob05390_dir / 'n20010804.I.OGLE.OB05390.txt')
        cls.ob05390_moa = str(cls.ob05390_dir / 'n20050724.r.MOA.OB05390.txt')

    def test_i_datasets_only(self):
        """Test initialization with datasets parameter only."""
        # Create datasets manually
        dataset_ogle = MulensModel.MulensData(file_name=self.ob140939_ogle)

        fitter = MMEXOFASTFitter(
            datasets=[dataset_ogle],
            coords=self.ob140939_coords,
            fit_type='point lens',
            verbose=False
        )

        self.assertEqual(len(fitter.datasets), 1)
        self.assertIn(fitter.datasets[0], fitter.dataset_to_filename)
        self.assertEqual(len(fitter.renorm_factors), 0)

    def test_ii_files_only(self):
        """Test initialization with files parameter only."""
        fitter = MMEXOFASTFitter(
            files=[self.ob140939_ogle],
            coords=self.ob140939_coords,
            fit_type='point lens',
            verbose=False
        )

        self.assertEqual(len(fitter.datasets), 1)
        self.assertIn(fitter.datasets[0], fitter.dataset_to_filename)
        self.assertEqual(
            fitter.dataset_to_filename[fitter.datasets[0]],
            self.ob140939_ogle
        )
        self.assertEqual(len(fitter.renorm_factors), 0)

    def test_iii_pickle_no_renorm(self):
        """Test initialization from pickle without error renormalization."""
        # Create initial fitter and save
        with tempfile.TemporaryDirectory() as tmpdir:
            pickle_path = os.path.join(tmpdir, 'test.pkl')

            fitter1 = MMEXOFASTFitter(
                files=[self.ob140939_ogle],
                coords=self.ob140939_coords,
                fit_type='point lens',
                verbose=False
            )
            fitter1._save_restart_state(pickle_path)

            # Load from pickle
            fitter2 = MMEXOFASTFitter(
                restart_file=pickle_path,
                coords=self.ob140939_coords,
                fit_type='point lens',
                verbose=False
            )

            self.assertEqual(len(fitter2.datasets), 1)
            self.assertIn(fitter2.datasets[0], fitter2.dataset_to_filename)
            self.assertEqual(
                fitter2.dataset_to_filename[fitter2.datasets[0]],
                self.ob140939_ogle
            )
            self.assertEqual(len(fitter2.renorm_factors), 0)

    def test_iii_pickle_with_renorm(self):
        """Test initialization from pickle with error renormalization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pickle_path = os.path.join(tmpdir, 'test_renorm.pkl')

            # Create initial fitter with renormalization
            fitter1 = MMEXOFASTFitter(
                files=[self.ob140939_ogle],
                coords=self.ob140939_coords,
                fit_type='point lens',
                verbose=False
            )

            # Manually set renorm factors to simulate renormalization
            fitter1.renorm_factors[self.ob140939_ogle] = 1.5
            fitter1._save_restart_state(pickle_path)

            # Load from pickle
            fitter2 = MMEXOFASTFitter(
                restart_file=pickle_path,
                coords=self.ob140939_coords,
                fit_type='point lens',
                verbose=False
            )

            self.assertEqual(len(fitter2.datasets), 1)
            self.assertEqual(len(fitter2.renorm_factors), 1)
            self.assertEqual(fitter2.renorm_factors[self.ob140939_ogle], 1.5)

    def test_iv_pickle_plus_new_same_location_no_renorm(self):
        """Test pickle + new dataset from same location, no initial renorm."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pickle_path = os.path.join(tmpdir, 'test.pkl')

            # Create initial fitter with OGLE only
            fitter1 = MMEXOFASTFitter(
                files=[self.ob05390_ogle],
                coords=self.ob05390_coords,
                fit_type='point lens',
                verbose=False
            )
            fitter1._save_restart_state(pickle_path)

            # Load from pickle and add MOA (same location)
            dataset_moa = MulensModel.MulensData(file_name=self.ob05390_moa)

            fitter2 = MMEXOFASTFitter(
                restart_file=pickle_path,
                datasets=[dataset_moa],
                coords=self.ob05390_coords,
                fit_type='point lens',
                verbose=False
            )

            # Should have 2 datasets: restored OGLE + new MOA
            self.assertEqual(len(fitter2.datasets), 2)

            # Check that filenames are tracked
            filenames = list(fitter2.dataset_to_filename.values())
            self.assertIn(self.ob05390_ogle, filenames)
            self.assertIn(self.ob05390_moa, filenames)

            self.assertEqual(len(fitter2.renorm_factors), 0)

    def test_iv_pickle_plus_new_same_location_with_renorm(self):
        """Test pickle + new dataset from same location, with initial renorm."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pickle_path = os.path.join(tmpdir, 'test_renorm.pkl')

            # Create initial fitter with OGLE and renormalization
            fitter1 = MMEXOFASTFitter(
                files=[self.ob05390_ogle],
                coords=self.ob05390_coords,
                fit_type='point lens',
                verbose=False
            )
            fitter1.renorm_factors[self.ob05390_ogle] = 1.3
            fitter1._save_restart_state(pickle_path)

            # Load from pickle and add MOA
            dataset_moa = MulensModel.MulensData(file_name=self.ob05390_moa)

            fitter2 = MMEXOFASTFitter(
                restart_file=pickle_path,
                datasets=[dataset_moa],
                coords=self.ob05390_coords,
                fit_type='point lens',
                verbose=False
            )

            self.assertEqual(len(fitter2.datasets), 2)

            # Check renorm factors - should have OGLE but not MOA yet
            self.assertEqual(len(fitter2.renorm_factors), 1)
            self.assertEqual(fitter2.renorm_factors[self.ob05390_ogle], 1.3)
            self.assertNotIn(self.ob05390_moa, fitter2.renorm_factors)

    def test_v_pickle_plus_new_different_location_no_renorm(self):
        """Test pickle + new dataset from different location, no initial renorm."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pickle_path = os.path.join(tmpdir, 'test.pkl')

            # Create initial fitter with OB140939 OGLE
            fitter1 = MMEXOFASTFitter(
                files=[self.ob140939_ogle],
                coords=self.ob140939_coords,
                fit_type='point lens',
                verbose=False
            )
            fitter1._save_restart_state(pickle_path)

            # Load and add Spitzer (different location)
            dataset_spitzer = MulensModel.MulensData(file_name=self.ob140939_spitzer)

            fitter2 = MMEXOFASTFitter(
                restart_file=pickle_path,
                datasets=[dataset_spitzer],
                coords=self.ob140939_coords,
                fit_type='point lens',
                verbose=False
            )

            # Should have 2 datasets
            self.assertEqual(len(fitter2.datasets), 2)

            filenames = list(fitter2.dataset_to_filename.values())
            self.assertIn(self.ob140939_ogle, filenames)
            self.assertIn(self.ob140939_spitzer, filenames)

            self.assertEqual(len(fitter2.renorm_factors), 0)

    def test_v_pickle_plus_new_different_location_with_renorm(self):
        """Test pickle + new dataset from different location, with initial renorm."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pickle_path = os.path.join(tmpdir, 'test_renorm.pkl')

            # Create initial fitter with OB140939 OGLE and renormalization
            fitter1 = MMEXOFASTFitter(
                files=[self.ob140939_ogle],
                coords=self.ob140939_coords,
                fit_type='point lens',
                verbose=False
            )
            fitter1.renorm_factors[self.ob140939_ogle] = 1.2
            fitter1._save_restart_state(pickle_path)

            # Load and add Spitzer
            dataset_spitzer = MulensModel.MulensData(file_name=self.ob140939_spitzer)

            fitter2 = MMEXOFASTFitter(
                restart_file=pickle_path,
                datasets=[dataset_spitzer],
                coords=self.ob140939_coords,
                fit_type='point lens',
                verbose=False
            )

            self.assertEqual(len(fitter2.datasets), 2)

            # Check renorm factors - should have OGLE but not Spitzer yet
            self.assertEqual(len(fitter2.renorm_factors), 1)
            self.assertEqual(fitter2.renorm_factors[self.ob140939_ogle], 1.2)
            self.assertNotIn(self.ob140939_spitzer, fitter2.renorm_factors)


if __name__ == '__main__':
    unittest.main()
