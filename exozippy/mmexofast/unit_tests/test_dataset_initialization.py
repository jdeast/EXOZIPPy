"""Unit tests for dataset initialization and merging."""

import unittest
import tempfile
import os
import pickle
from pathlib import Path

import MulensModel

from exozippy import MULENS_DATA_PATH
from exozippy.mmexofast import MMEXOFASTFitter, OutputConfig


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

    def test_1_datasets_only_single_ground(self):
        """Test initialization with datasets parameter only (single ground-based dataset)."""
        # Create dataset manually
        dataset_ogle = MulensModel.MulensData(file_name=self.ob140939_ogle, phot_fmt='flux')

        fitter = MMEXOFASTFitter(
            datasets=[dataset_ogle],
            coords=self.ob140939_coords,
            fit_type='point lens',
            verbose=False
        )

        # Check datasets
        self.assertEqual(len(fitter.datasets), 1)
        self.assertIn(fitter.datasets[0], fitter.dataset_to_filename)

        # Check renormalization tracking
        self.assertEqual(len(fitter.renorm_factors), 0)

        # Check number of locations
        self.assertEqual(fitter.n_loc, 1)

    def test_2_files_only_single_ground(self):
        """Test initialization with files parameter only (single ground-based dataset)."""
        fitter = MMEXOFASTFitter(
            files=[self.ob140939_ogle],
            coords=self.ob140939_coords,
            fit_type='point lens',
            verbose=False
        )

        # Check datasets
        self.assertEqual(len(fitter.datasets), 1)
        self.assertIn(fitter.datasets[0], fitter.dataset_to_filename)
        self.assertEqual(
            fitter.dataset_to_filename[fitter.datasets[0]],
            self.ob140939_ogle
        )

        # Check renormalization tracking
        self.assertEqual(len(fitter.renorm_factors), 0)

        # Check number of locations
        self.assertEqual(fitter.n_loc, 1)

    def test_3_pickle_only_no_renorm_single_ground(self):
        """Test initialization from pickle without error renormalization (single ground-based dataset)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_head = 'test_ob140939'

            # Create initial fitter and save
            fitter1 = MMEXOFASTFitter(
                files=[self.ob140939_ogle],
                coords=self.ob140939_coords,
                fit_type='point lens',
                output_config=OutputConfig(base_dir=tmpdir, file_head=file_head, save_restart_files=True),
                verbose=False
            )
            fitter1._save_restart_state()
            # Debug: check what files exist

            # Load from pickle
            restart_file = os.path.join(tmpdir, f'{file_head}_restart.pkl')
            fitter2 = MMEXOFASTFitter(
                restart_file=restart_file,
                coords=self.ob140939_coords,
                fit_type='point lens',
                verbose=False
            )

            # Check datasets
            self.assertEqual(len(fitter2.datasets), 1)
            self.assertIn(fitter2.datasets[0], fitter2.dataset_to_filename)
            self.assertEqual(
                fitter2.dataset_to_filename[fitter2.datasets[0]],
                self.ob140939_ogle
            )

            # Check renormalization tracking
            self.assertEqual(len(fitter2.renorm_factors), 0)

            # Check number of locations
            self.assertEqual(fitter2.n_loc, 1)

    def test_4_pickle_only_with_renorm_single_ground(self):
        """Test initialization from pickle with error renormalization (single ground-based dataset)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_head = 'test_ob140939_renorm'

            # Create initial fitter with renormalization
            fitter1 = MMEXOFASTFitter(
                files=[self.ob140939_ogle],
                coords=self.ob140939_coords,
                fit_type='point lens',
                output_config=OutputConfig(
                    base_dir=tmpdir,
                    file_head=file_head,
                    save_restart_files=True
                ),
                verbose=False
            )

            # Manually set renorm factors to simulate renormalization
            fitter1.renorm_factors[self.ob140939_ogle] = 1.5
            fitter1._save_restart_state()

            # Load from pickle
            restart_file = os.path.join(tmpdir, f'{file_head}_restart.pkl')
            fitter2 = MMEXOFASTFitter(
                restart_file=restart_file,
                coords=self.ob140939_coords,
                fit_type='point lens',
                verbose=False
            )

            # Check datasets
            self.assertEqual(len(fitter2.datasets), 1)

            # Check renormalization tracking
            self.assertEqual(len(fitter2.renorm_factors), 1)
            self.assertEqual(fitter2.renorm_factors[self.ob140939_ogle], 1.5)

            # Check number of locations
            self.assertEqual(fitter2.n_loc, 1)

    def test_5a_pickle_plus_new_only_same_location_no_renorm(self):
        """Test pickle + only new dataset from same location, no initial renorm (load only new dataset)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_head = 'test_ob05390'

            # Create initial fitter with OGLE only
            fitter1 = MMEXOFASTFitter(
                files=[self.ob05390_ogle],
                coords=self.ob05390_coords,
                fit_type='point lens',
                output_config=OutputConfig(
                    base_dir=tmpdir,
                    file_head=file_head,
                    save_restart_files=True
                ),
                verbose=False
            )
            fitter1._save_restart_state()

            # Load from pickle with only MOA (new dataset)
            dataset_moa = MulensModel.MulensData(file_name=self.ob05390_moa, phot_fmt='flux')
            restart_file = os.path.join(tmpdir, f'{file_head}_restart.pkl')

            fitter2 = MMEXOFASTFitter(
                restart_file=restart_file,
                datasets=[dataset_moa],
                coords=self.ob05390_coords,
                fit_type='point lens',
                verbose=False
            )

            # Should have 1 dataset: only MOA
            self.assertEqual(len(fitter2.datasets), 1)

            # Check that only MOA filename is tracked
            filenames = list(fitter2.dataset_to_filename.values())
            self.assertNotIn(self.ob05390_ogle, filenames)
            self.assertIn(self.ob05390_moa, filenames)

            # Check renormalization tracking
            self.assertEqual(len(fitter2.renorm_factors), 0)

            # Check number of locations (ground-based)
            self.assertEqual(fitter2.n_loc, 1)

            # all_fit_results should be loaded from pickle
            self.assertIsNotNone(fitter2.all_fit_results)

    def test_5b_pickle_plus_old_and_new_same_location_no_renorm(self):
        """Test pickle + old and new datasets from same location, no initial renorm (merge datasets)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_head = 'test_ob05390'

            # Create initial fitter with OGLE only
            fitter1 = MMEXOFASTFitter(
                files=[self.ob05390_ogle],
                coords=self.ob05390_coords,
                fit_type='point lens',
                output_config=OutputConfig(
                    base_dir=tmpdir,
                    file_head=file_head,
                    save_restart_files=True
                ),
                verbose=False
            )
            fitter1._save_restart_state()

            # Load from pickle with both OGLE and MOA
            restart_file = os.path.join(tmpdir, f'{file_head}_restart.pkl')

            fitter2 = MMEXOFASTFitter(
                restart_file=restart_file,
                files=[self.ob05390_ogle, self.ob05390_moa],
                coords=self.ob05390_coords,
                fit_type='point lens',
                verbose=False
            )

            # Should have 2 datasets: OGLE from pickle + new MOA
            self.assertEqual(len(fitter2.datasets), 2)

            # Check that both filenames are tracked
            filenames = list(fitter2.dataset_to_filename.values())
            self.assertIn(self.ob05390_ogle, filenames)
            self.assertIn(self.ob05390_moa, filenames)

            # Check renormalization tracking
            self.assertEqual(len(fitter2.renorm_factors), 0)

            # Check number of locations (both ground-based)
            self.assertEqual(fitter2.n_loc, 1)

    def test_6a_pickle_plus_new_only_same_location_with_renorm(self):
        """Test pickle + only new dataset from same location, with initial renorm (load only new dataset)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_head = 'test_ob05390_renorm'

            # Create initial fitter with OGLE and renormalization
            fitter1 = MMEXOFASTFitter(
                files=[self.ob05390_ogle],
                coords=self.ob05390_coords,
                fit_type='point lens',
                output_config=OutputConfig(
                    base_dir=tmpdir,
                    file_head=file_head,
                    save_restart_files=True
                ),
                verbose=False
            )
            fitter1.renorm_factors[self.ob05390_ogle] = 1.3
            fitter1._save_restart_state()

            # Load from pickle with only MOA (new dataset)
            restart_file = os.path.join(tmpdir, f'{file_head}_restart.pkl')

            fitter2 = MMEXOFASTFitter(
                restart_file=restart_file,
                files=[self.ob05390_moa],
                coords=self.ob05390_coords,
                fit_type='point lens',
                verbose=False
            )

            # Should have 1 dataset: only MOA
            self.assertEqual(len(fitter2.datasets), 1)

            # Check that only MOA filename is tracked
            filenames = list(fitter2.dataset_to_filename.values())
            self.assertNotIn(self.ob05390_ogle, filenames)
            self.assertIn(self.ob05390_moa, filenames)

            # Check renorm factors - should have OGLE from pickle but not MOA yet
            self.assertEqual(len(fitter2.renorm_factors), 1)
            self.assertEqual(fitter2.renorm_factors[self.ob05390_ogle], 1.3)
            self.assertNotIn(self.ob05390_moa, fitter2.renorm_factors)

            # Check number of locations (ground-based)
            self.assertEqual(fitter2.n_loc, 1)

    def test_6b_pickle_plus_old_and_new_same_location_with_renorm(self):
        """Test pickle + old and new datasets from same location, with initial renorm (merge datasets)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_head = 'test_ob05390_renorm'

            # Create initial fitter with OGLE and renormalization
            fitter1 = MMEXOFASTFitter(
                files=[self.ob05390_ogle],
                coords=self.ob05390_coords,
                fit_type='point lens',
                output_config=OutputConfig(
                    base_dir=tmpdir,
                    file_head=file_head,
                    save_restart_files=True
                ),
                verbose=False
            )
            fitter1.renorm_factors[self.ob05390_ogle] = 1.3
            fitter1._save_restart_state()

            # Load from pickle with both OGLE and MOA
            restart_file = os.path.join(tmpdir, f'{file_head}_restart.pkl')

            fitter2 = MMEXOFASTFitter(
                restart_file=restart_file,
                files=[self.ob05390_ogle, self.ob05390_moa],
                coords=self.ob05390_coords,
                fit_type='point lens',
                verbose=False
            )

            # Should have 2 datasets: OGLE from pickle + new MOA
            self.assertEqual(len(fitter2.datasets), 2)

            # Check that both filenames are tracked
            filenames = list(fitter2.dataset_to_filename.values())
            self.assertIn(self.ob05390_ogle, filenames)
            self.assertIn(self.ob05390_moa, filenames)

            # Check renorm factors - should have OGLE but not MOA yet
            self.assertEqual(len(fitter2.renorm_factors), 1)
            self.assertEqual(fitter2.renorm_factors[self.ob05390_ogle], 1.3)
            self.assertNotIn(self.ob05390_moa, fitter2.renorm_factors)

            # Check number of locations (both ground-based)
            self.assertEqual(fitter2.n_loc, 1)

    def test_7a_pickle_plus_new_only_different_location_no_renorm(self):
        """Test pickle + only new dataset from different location, no initial renorm (load only new dataset)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_head = 'test_ob140939'

            # Create initial fitter with OGLE (ground)
            fitter1 = MMEXOFASTFitter(
                files=[self.ob140939_ogle],
                coords=self.ob140939_coords,
                fit_type='point lens',
                output_config=OutputConfig(
                    base_dir=tmpdir,
                    file_head=file_head,
                    save_restart_files=True
                ),
                verbose=False
            )
            fitter1._save_restart_state()

            # Load from pickle with only Spitzer (satellite - different location)
            restart_file = os.path.join(tmpdir, f'{file_head}_restart.pkl')

            fitter2 = MMEXOFASTFitter(
                restart_file=restart_file,
                files=[self.ob140939_spitzer],
                coords=self.ob140939_coords,
                fit_type='point lens',
                verbose=False
            )

            # Should have 1 dataset: only Spitzer
            self.assertEqual(len(fitter2.datasets), 1)

            # Check that only Spitzer filename is tracked
            filenames = list(fitter2.dataset_to_filename.values())
            self.assertNotIn(self.ob140939_ogle, filenames)
            self.assertIn(self.ob140939_spitzer, filenames)

            # Check renormalization tracking
            self.assertEqual(len(fitter2.renorm_factors), 0)

            # Check number of locations (only Spitzer loaded, but should still be 1?)
            self.assertEqual(fitter2.n_loc, 1)

    def test_7b_pickle_plus_old_and_new_different_location_no_renorm(self):
        """Test pickle + old and new datasets from different locations, no initial renorm (merge datasets)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_head = 'test_ob140939'

            # Create initial fitter with OGLE (ground)
            fitter1 = MMEXOFASTFitter(
                files=[self.ob140939_ogle],
                coords=self.ob140939_coords,
                fit_type='point lens',
                output_config=OutputConfig(
                    base_dir=tmpdir,
                    file_head=file_head,
                    save_restart_files=True
                ),
                verbose=False
            )
            fitter1._save_restart_state()

            # Load from pickle with both OGLE and Spitzer
            restart_file = os.path.join(tmpdir, f'{file_head}_restart.pkl')

            fitter2 = MMEXOFASTFitter(
                restart_file=restart_file,
                files=[self.ob140939_ogle, self.ob140939_spitzer],
                coords=self.ob140939_coords,
                fit_type='point lens',
                verbose=False
            )

            # Should have 2 datasets: OGLE from pickle + new Spitzer
            self.assertEqual(len(fitter2.datasets), 2)

            # Check that both filenames are tracked
            filenames = list(fitter2.dataset_to_filename.values())
            self.assertIn(self.ob140939_ogle, filenames)
            self.assertIn(self.ob140939_spitzer, filenames)

            # Check renormalization tracking
            self.assertEqual(len(fitter2.renorm_factors), 0)

            # Check number of locations (ground + satellite)
            self.assertEqual(fitter2.n_loc, 2)

    def test_8a_pickle_plus_new_only_different_location_with_renorm(self):
        """Test pickle + only new dataset from different location, with initial renorm (load only new dataset)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_head = 'test_ob140939_renorm'

            # Create initial fitter with OGLE (ground) and renormalization
            fitter1 = MMEXOFASTFitter(
                files=[self.ob140939_ogle],
                coords=self.ob140939_coords,
                fit_type='point lens',
                output_config=OutputConfig(
                    base_dir=tmpdir,
                    file_head=file_head,
                    save_restart_files=True
                ),
                verbose=False
            )
            fitter1.renorm_factors[self.ob140939_ogle] = 1.2
            fitter1._save_restart_state()

            # Load from pickle with only Spitzer (satellite - different location)
            restart_file = os.path.join(tmpdir, f'{file_head}_restart.pkl')

            fitter2 = MMEXOFASTFitter(
                restart_file=restart_file,
                files=[self.ob140939_spitzer],
                coords=self.ob140939_coords,
                fit_type='point lens',
                verbose=False
            )

            # Should have 1 dataset: only Spitzer
            self.assertEqual(len(fitter2.datasets), 1)

            # Check that only Spitzer filename is tracked
            filenames = list(fitter2.dataset_to_filename.values())
            self.assertNotIn(self.ob140939_ogle, filenames)
            self.assertIn(self.ob140939_spitzer, filenames)

            # Check renorm factors - should have OGLE from pickle but not Spitzer yet
            self.assertEqual(len(fitter2.renorm_factors), 1)
            self.assertEqual(fitter2.renorm_factors[self.ob140939_ogle], 1.2)
            self.assertNotIn(self.ob140939_spitzer, fitter2.renorm_factors)

            # Check number of locations
            self.assertEqual(fitter2.n_loc, 1)

    def test_8b_pickle_plus_old_and_new_different_location_with_renorm(self):
        """Test pickle + old and new datasets from different locations, with initial renorm (merge datasets)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_head = 'test_ob140939_renorm'

            # Create initial fitter with OGLE (ground) and renormalization
            fitter1 = MMEXOFASTFitter(
                files=[self.ob140939_ogle],
                coords=self.ob140939_coords,
                fit_type='point lens',
                output_config=OutputConfig(
                    base_dir=tmpdir,
                    file_head=file_head,
                    save_restart_files=True
                ),
                verbose=False
            )
            fitter1.renorm_factors[self.ob140939_ogle] = 1.2
            fitter1._save_restart_state()

            # Load from pickle with both OGLE and Spitzer
            restart_file = os.path.join(tmpdir, f'{file_head}_restart.pkl')

            fitter2 = MMEXOFASTFitter(
                restart_file=restart_file,
                files=[self.ob140939_ogle, self.ob140939_spitzer],
                coords=self.ob140939_coords,
                fit_type='point lens',
                verbose=False
            )

            # Should have 2 datasets: OGLE from pickle + new Spitzer
            self.assertEqual(len(fitter2.datasets), 2)

            # Check that both filenames are tracked
            filenames = list(fitter2.dataset_to_filename.values())
            self.assertIn(self.ob140939_ogle, filenames)
            self.assertIn(self.ob140939_spitzer, filenames)

            # Check renorm factors - should have OGLE but not Spitzer yet
            self.assertEqual(len(fitter2.renorm_factors), 1)
            self.assertEqual(fitter2.renorm_factors[self.ob140939_ogle], 1.2)
            self.assertNotIn(self.ob140939_spitzer, fitter2.renorm_factors)

            # Check number of locations (ground + satellite)
            self.assertEqual(fitter2.n_loc, 2)


if __name__ == '__main__':
    unittest.main()
