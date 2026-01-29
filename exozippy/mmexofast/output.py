"""
output.py

Output configuration and manager for MMEXOFASTFitter.

Responsibilities:
- Centralize all decisions about *where* to write and *what* to write.
- Provide simple hooks that the fitter can call (log messages, save grids, tables, restart files, plots).
- Avoid forcing the core fitting code to know about paths / formats.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .results import GridSearchResult   # defined in results.py


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class OutputConfig:
    """
    User-configurable options controlling all file-based output.

    Parameters
    ----------
    base_dir : Path
        Directory where output files will be written.
    file_head : str
        Common prefix for all output filenames.

    Flags
    -----
    save_log : bool
        If True, write a log file (in addition to any console prints).
    save_plots : bool
        If True, enable saving diagnostic plots via OutputManager.save_plot().
    save_latex_tables : bool
        If True, enable saving LaTeX tables via OutputManager.save_latex_table().
    save_restart_files : bool
        If True, enable saving restart state via OutputManager.save_restart_state().
    save_grid_results : bool
        If True, enable saving grid search results (EF/AF/PAR/etc.).
    """
    base_dir: Path = Path('.')
    file_head: str = 'mmexo'

    save_log: bool = True
    save_plots: bool = False
    save_latex_tables: bool = False
    save_restart_files: bool = False
    save_grid_results: bool = False


# ============================================================================
# Output manager
# ============================================================================


class OutputManager:
    """
    Central output manager.

    The fitter can call these methods at well-defined points:
    - log()
    - handle_grid_search_result()
    - save_latex_table()
    - save_restart_state()
    - save_plot()

    The behavior of each is controlled by OutputConfig flags.
    """

    def __init__(self, config: OutputConfig, verbose: bool = False):
        self.config = config
        self.verbose = verbose
        self.config.base_dir.mkdir(parents=True, exist_ok=True)
        self._setup_logging_if_needed()

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _setup_logging_if_needed(self) -> None:
        import logging

        self.logger = logging.getLogger("mmexofast")
        self.logger.setLevel(logging.INFO)

        # Clear any existing handlers (avoid duplicates)
        self.logger.handlers.clear()

        # Console handler (if verbose)
        if self.verbose:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(
                logging.Formatter("%(message)s")  # Simple format for console
            )
            self.logger.addHandler(console_handler)

        # File handler (if save_log)
        if self.config.save_log:
            logfile = self.config.base_dir / f"{self.config.file_head}.log"
            file_handler = logging.FileHandler(logfile)
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            )
            self.logger.addHandler(file_handler)

    def log(self, msg: str) -> None:
        """Log message to console (if verbose) and/or file (if save_log)."""
        self.logger.info(msg)

    # ------------------------------------------------------------------
    # Grid search results
    # ------------------------------------------------------------------

    def handle_grid_search_result(self, result: GridSearchResult) -> None:
        """
        Optionally save a GridSearchResult to disk.

        Writes a compressed NumPy .npz file:
            <file_head>_<name>_grid.npz
        """
        if not self.config.save_grid_results:
            return

        path = self.config.base_dir / f"{self.config.file_head}_{result.name}_grid.npz"
        np.savez_compressed(
            path,
            param_names=result.param_names,
            grid_points=result.grid_points,
            chi2=result.chi2,
            metadata=result.metadata,
            best_index=result.best_index,
        )

    # ------------------------------------------------------------------
    # Tables
    # ------------------------------------------------------------------

    def save_latex_table(self, name: str, table_str: str) -> None:
        """
        Save a LaTeX table to:
            <file_head>_<name>.tex
        """
        if not self.config.save_latex_tables:
            return

        path = self.config.base_dir / f"{self.config.file_head}_{name}.tex"
        path.write_text(table_str)

    # ------------------------------------------------------------------
    # Restart state
    # ------------------------------------------------------------------

    def save_restart_state(self, state_bytes: bytes) -> None:
        """
        Save a restart state (e.g., pickled dict) to:
            <file_head>_restart.pkl
        """
        if not self.config.save_restart_files:
            return

        path = self.config.base_dir / f"{self.config.file_head}_restart.pkl"
        path.write_bytes(state_bytes)

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def save_plot(self, name: str, fig) -> None:
        """
        Save a Matplotlib figure to:
            <file_head>_<name>.png
        """
        if not self.config.save_plots:
            return

        path = self.config.base_dir / f"{self.config.file_head}_{name}.png"
        fig.savefig(path, dpi=300)
        fig.clf()