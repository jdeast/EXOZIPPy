"""Unit tests for output module."""

import logging
import tempfile
import unittest
from pathlib import Path

from exozippy.mmexofast import OutputConfig, OutputManager
from exozippy.mmexofast.unit_tests.test_utils import (
    create_mock_grid_search_result,
    create_mock_figure,
)


class TestOutputConfig(unittest.TestCase):
    """Test OutputConfig dataclass."""

    def test_default_values(self):
        """Test that OutputConfig has correct default values."""
        config = OutputConfig()
        self.assertEqual(config.base_dir, Path('.'))
        self.assertEqual(config.file_head, 'mmexo')
        self.assertTrue(config.save_log)
        self.assertFalse(config.save_plots)
        self.assertFalse(config.save_latex_tables)
        self.assertFalse(config.save_restart_files)
        self.assertFalse(config.save_grid_results)

    def test_path_conversion_from_string(self):
        """Test that string base_dir is converted to Path in __post_init__."""
        config = OutputConfig(base_dir='some/path')
        self.assertIsInstance(config.base_dir, Path)
        self.assertEqual(config.base_dir, Path('some/path'))

    def test_all_fields_accessible(self):
        """Test that all fields are accessible after creation with custom values."""
        config = OutputConfig(
            base_dir=Path('/tmp'),
            file_head='test',
            save_log=False,
            save_plots=True,
            save_latex_tables=True,
            save_restart_files=True,
            save_grid_results=True,
        )
        self.assertEqual(config.base_dir, Path('/tmp'))
        self.assertEqual(config.file_head, 'test')
        self.assertFalse(config.save_log)
        self.assertTrue(config.save_plots)
        self.assertTrue(config.save_latex_tables)
        self.assertTrue(config.save_restart_files)
        self.assertTrue(config.save_grid_results)


class TestOutputManager(unittest.TestCase):
    """Test OutputManager class."""

    def setUp(self):
        """Create temporary directory for file operations."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_dir = Path(self.temp_dir.name)

    def tearDown(self):
        """Clean up logging handlers and temporary directory."""
        logger = logging.getLogger("mmexofast")
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
        self.temp_dir.cleanup()

    def _make_manager(self, save_log=False, **kwargs):
        """Create OutputManager with temp dir. Defaults save_log=False."""
        config = OutputConfig(base_dir=self.base_dir, save_log=save_log, **kwargs)
        return OutputManager(config)

    # ------------------------------------------------------------------
    # Path handling
    # ------------------------------------------------------------------

    def test_creates_directory_if_needed(self):
        """Test that OutputManager creates base_dir if it doesn't exist."""
        nested_dir = self.base_dir / 'sub' / 'nested'
        config = OutputConfig(base_dir=nested_dir, save_log=False)
        OutputManager(config)
        self.assertTrue(nested_dir.exists())

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def test_log_creates_file_when_save_log_true(self):
        """Test that a log file is created when save_log=True."""
        self._make_manager(file_head='test', save_log=True)
        log_path = self.base_dir / 'test.log'
        self.assertTrue(log_path.exists())

    def test_log_writes_message_to_file(self):
        """Test that log() writes the message to the log file."""
        manager = self._make_manager(file_head='test', save_log=True)
        manager.log("hello from test")

        for handler in logging.getLogger("mmexofast").handlers:
            handler.flush()

        content = (self.base_dir / 'test.log').read_text()
        self.assertIn("hello from test", content)

    def test_log_does_not_create_file_when_save_log_false(self):
        """Test that no log file is created when save_log=False."""
        self._make_manager(file_head='test', save_log=False)
        self.assertFalse((self.base_dir / 'test.log').exists())

    # ------------------------------------------------------------------
    # save_plot
    # ------------------------------------------------------------------

    def test_save_plot_calls_savefig_when_enabled(self):
        """Test that save_plot() calls fig.savefig() when save_plots=True."""
        manager = self._make_manager(file_head='test', save_plots=True)
        fig = create_mock_figure()
        manager.save_plot('myplot', fig)
        fig.savefig.assert_called_once()

    def test_save_plot_correct_path(self):
        """Test that save_plot() passes correct path and dpi to fig.savefig()."""
        manager = self._make_manager(file_head='test', save_plots=True)
        fig = create_mock_figure()
        manager.save_plot('myplot', fig)
        expected_path = self.base_dir / 'test_myplot.png'
        fig.savefig.assert_called_once_with(expected_path, dpi=300)

    def test_save_plot_calls_clf(self):
        """Test that save_plot() calls fig.clf() after saving."""
        manager = self._make_manager(file_head='test', save_plots=True)
        fig = create_mock_figure()
        manager.save_plot('myplot', fig)
        fig.clf.assert_called_once()

    def test_save_plot_does_nothing_when_disabled(self):
        """Test that save_plot() does not call savefig or clf when save_plots=False."""
        manager = self._make_manager(file_head='test', save_plots=False)
        fig = create_mock_figure()
        manager.save_plot('myplot', fig)
        fig.savefig.assert_not_called()
        fig.clf.assert_not_called()

    # ------------------------------------------------------------------
    # save_latex_table
    # ------------------------------------------------------------------

    def test_save_latex_table_writes_file_when_enabled(self):
        """Test that save_latex_table() writes .tex file when save_latex_tables=True."""
        manager = self._make_manager(file_head='test', save_latex_tables=True)
        manager.save_latex_table('mytable', r'\begin{table}...\end{table}')
        self.assertTrue((self.base_dir / 'test_mytable.tex').exists())

    def test_save_latex_table_correct_content(self):
        """Test that save_latex_table() writes the correct content to the file."""
        manager = self._make_manager(file_head='test', save_latex_tables=True)
        table_str = r'\begin{table}...\end{table}'
        manager.save_latex_table('mytable', table_str)
        content = (self.base_dir / 'test_mytable.tex').read_text()
        self.assertEqual(content, table_str)

    def test_save_latex_table_does_nothing_when_disabled(self):
        """Test that save_latex_table() writes no file when save_latex_tables=False."""
        manager = self._make_manager(file_head='test', save_latex_tables=False)
        manager.save_latex_table('mytable', r'\begin{table}...\end{table}')
        self.assertFalse((self.base_dir / 'test_mytable.tex').exists())

    # ------------------------------------------------------------------
    # save_restart_state
    # ------------------------------------------------------------------

    def test_save_restart_state_writes_file_when_enabled(self):
        """Test that save_restart_state() writes file when save_restart_files=True."""
        manager = self._make_manager(file_head='test', save_restart_files=True)
        manager.save_restart_state(b"restart bytes")
        self.assertTrue((self.base_dir / 'test_restart.pkl').exists())

    def test_save_restart_state_correct_naming(self):
        """Test that restart file uses {file_head}_restart.pkl naming."""
        manager = self._make_manager(file_head='myrun', save_restart_files=True)
        manager.save_restart_state(b"restart bytes")
        self.assertTrue((self.base_dir / 'myrun_restart.pkl').exists())

    def test_save_restart_state_correct_content(self):
        """Test that save_restart_state() writes the correct bytes to the file."""
        manager = self._make_manager(file_head='test', save_restart_files=True)
        state_bytes = b"restart bytes"
        manager.save_restart_state(state_bytes)
        content = (self.base_dir / 'test_restart.pkl').read_bytes()
        self.assertEqual(content, state_bytes)

    def test_save_restart_state_does_nothing_when_disabled(self):
        """Test that save_restart_state() writes no file when save_restart_files=False."""
        manager = self._make_manager(file_head='test', save_restart_files=False)
        manager.save_restart_state(b"restart bytes")
        self.assertFalse((self.base_dir / 'test_restart.pkl').exists())

    # ------------------------------------------------------------------
    # handle_grid_search_result
    # ------------------------------------------------------------------

    def test_handle_grid_search_result_writes_file_when_enabled(self):
        """Test that handle_grid_search_result() writes .npz file when save_grid_results=True."""
        manager = self._make_manager(file_head='test', save_grid_results=True)
        grid_result = create_mock_grid_search_result(name='EF')
        manager.handle_grid_search_result(grid_result)
        self.assertTrue((self.base_dir / 'test_EF_grid.npz').exists())

    def test_handle_grid_search_result_does_nothing_when_disabled(self):
        """Test that handle_grid_search_result() writes no file when save_grid_results=False."""
        manager = self._make_manager(file_head='test', save_grid_results=False)
        grid_result = create_mock_grid_search_result(name='EF')
        manager.handle_grid_search_result(grid_result)
        self.assertFalse((self.base_dir / 'test_EF_grid.npz').exists())


if __name__ == '__main__':
    unittest.main()
