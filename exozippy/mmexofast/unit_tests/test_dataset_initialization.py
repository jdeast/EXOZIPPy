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

    def test_1_datasets_only_single_ground(self):
        """Test initialization with datasets parameter only (single ground-based dataset)."""
        # Create dataset manually
        dataset_ogle = MulensModel.MulensData(file_name=self.ob140939_ogle)

        fitter = MMEXOFASTFitter(
            datasets=[dataset_ogle],
            coords=self.ob140939_coords,
            fit_type='point lens',
            verbose=False
        )

        # Check datasets
        self.assertEqual(len(fitter.datasets), 1)
        print('accessing dataset_to_filename', fitter.dataset_to_filename)
        self.assertIn(fitter.datasets[0], fitter.dataset_to_filename)

        # Check renormalization tracking
        self.assertEqual(len(fitter.renorm_factors), 0)

        # Check number of locations
        self.assertEqual(fitter.n_loc, 1)


if __name__ == '__main__':
    unittest.main()
