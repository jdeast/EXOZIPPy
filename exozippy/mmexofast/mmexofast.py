# mm_exofast_fitter.py
"""
High-level class and convenience wrapper for fitting microlensing events
with MM-EXOFASTv2.
"""
from __future__ import annotations

import inspect
import logging
import json
import os.path
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import MulensModel
import numpy as np
import pandas as pd
from scipy.special import erfcinv

import exozippy.mmexofast as mmexo
from .results import AllFitResults, FitRecord, IntermediateResults
from .workflow_step import WorkflowStep

logger = logging.getLogger(__name__)


# ===========================================================================
# Module-level constants for table formatting
# ===========================================================================

_PARAMETER_DECIMAL_PLACES = {
    'chi2':                              2,
    'N_data':                            0,
    't_0':                               6,
    'u_0':                               6,
    't_E':                               2,
    'rho':                               6,
    'log_rho':                           3,
    't_star':                            6,
    'pi_E_N':                            4,
    'pi_E_E':                            4,
    't_0_par':                           6,
    's':                                 6,
    'log_s':                             3,
    'q':                                 6,
    'log_q':                             3,
    'alpha':                             2,
    'convergence_K':                     6,
    'shear_G':                           6,
    'ds_dt':                             3,
    'dalpha_dt':                         3,
    's_z':                               6,
    'ds_z_dt':                           3,
    't_0_kep':                           6,
    'x_caustic_in':                      6,
    'x_caustic_out':                     6,
    't_caustic_in':                      6,
    't_caustic_out':                     6,
    'xi_period':                         3,
    'xi_semimajor_axis':                 6,
    'xi_inclination':                    2,
    'xi_Omega_node':                     2,
    'xi_argument_of_latitude_reference': 2,
    'xi_eccentricity':                   4,
    'xi_omega_periapsis':                2,
    'q_source':                          6,
    't_0_xi':                            6,
}

_FLUX_PARAM_DECIMAL_PLACES = 3


def _get_decimal_places(param_name: str) -> Optional[int]:
    """
    Return the number of decimal places for formatting a parameter value.

    Returns None if the parameter is not in the known list.

    Handles binary source parameters (e.g. ``'t_0_1'``, ``'t_0_2'``) by
    stripping the trailing source index and looking up the base name.

    Handles flux parameters (e.g. ``'I_S_OGLE'``, ``'R_B_MOA'``) by
    detecting the ``'_S_'`` or ``'_B_'`` pattern.

    Parameters
    ----------
    param_name : str
        Parameter name to look up.

    Returns
    -------
    int or None
        Number of decimal places, or None if not in the known list.
    """
    if param_name in _PARAMETER_DECIMAL_PLACES:
        return _PARAMETER_DECIMAL_PLACES[param_name]

    # Binary source parameters: e.g. 't_0_1' -> 't_0'
    parts = param_name.rsplit('_', 1)
    if len(parts) == 2 and parts[1].isdigit():
        base = parts[0]
        if base in _PARAMETER_DECIMAL_PLACES:
            return _PARAMETER_DECIMAL_PLACES[base]

    # Flux parameters: e.g. 'I_S_OGLE'
    parts = param_name.split('_')
    if len(parts) >= 3 and parts[1] in ('S', 'B'):
        return _FLUX_PARAM_DECIMAL_PLACES

    return None


def _format_results_column(df: pd.DataFrame, pm_symbol: str) -> pd.DataFrame:
    """
    Format values and sigma columns in a single-model results DataFrame.

    Applies parameter-specific decimal places to ``'values'``,
    ``'sigmas'``, ``'sigma_minus'``, and ``'sigma_plus'`` columns.  NaN
    sigmas become empty strings.  Sigma values are prefixed with the
    appropriate symbol:

    - ``sigmas``:      ``"{pm_symbol} {value}"``
    - ``sigma_minus``: ``"- {value}"``
    - ``sigma_plus``:  ``"+ {value}"``

    String values (e.g. ``'neg flux'``) are passed through unchanged.
    Parameters not in the known list are formatted with ``str()``.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns ``'parameter_names'``, ``'values'``, and
        optionally ``'sigmas'``, ``'sigma_minus'``, ``'sigma_plus'``.
    pm_symbol : str
        Symbol to prefix symmetric sigma values: ``'+/-'`` for ASCII,
        ``r'$\\pm$'`` for LaTeX.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with formatted string values.
    """
    df = df.copy()

    def fmt(param_name, value, prefix=''):
        if pd.isna(value):
            return ''
        if isinstance(value, str):
            return f'{prefix}{value}' if prefix else value
        decimal_places = _get_decimal_places(param_name)
        if decimal_places is None:
            return f'{prefix}{value}' if prefix else str(value)
        formatted = f'{value:.{decimal_places}f}'
        return f'{prefix}{formatted}' if prefix else formatted

    df['values'] = [
        fmt(param, val)
        for param, val in zip(df['parameter_names'], df['values'])
    ]
    if 'sigmas' in df.columns:
        df['sigmas'] = [
            fmt(param, val, prefix=f'{pm_symbol} ')
            for param, val in zip(df['parameter_names'], df['sigmas'])
        ]
    if 'sigma_minus' in df.columns:
        df['sigma_minus'] = [
            fmt(param, val, prefix='- ')
            for param, val in zip(df['parameter_names'], df['sigma_minus'])
        ]
    if 'sigma_plus' in df.columns:
        df['sigma_plus'] = [
            fmt(param, val, prefix='+ ')
            for param, val in zip(df['parameter_names'], df['sigma_plus'])
        ]

    return df


# ===========================================================================
# OutputConfig
# ===========================================================================

@dataclass
class OutputConfig:
    """
    Lightweight configuration for file output.

    Parameters
    ----------
    output_dir : Path
        Directory to write output files.  Created if it does not exist.
    file_prefix : str
        Prefix added to all output file names.
    save_plots : bool
        Whether to save figures to disk.
    save_grid_results : bool
        Whether to save raw grid search results to text files.
    save_table : bool
        Whether to save fit results tables to disk.
    table_formats : str or list of str
        Table format(s) to save.  Each entry must be ``'ascii'`` or
        ``'latex'``.  A bare string is accepted and wrapped in a list.
        Defaults to ``['latex']``.
    save_exozippy_init : bool
        Whether to save the EXOZIPPy initialization dict to a JSON file.

    """
    output_dir: Path = field(default_factory=Path)
    file_prefix: str = ''
    save_plots: bool = True
    save_grid_results: bool = False
    save_table: bool = False
    table_formats: list = field(default_factory=lambda: ['latex'])
    save_exozippy_init: bool = False

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if isinstance(self.table_formats, str):
            self.table_formats = [self.table_formats]

    def plot_path(self, name: str, ext: str = 'pdf') -> Path:
        """
        Return the full path for a named plot file.

        Parameters
        ----------
        name : str
            Base name of the plot.
        ext : str
            File extension without leading dot.

        Returns
        -------
        Path
        """
        prefix = f'{self.file_prefix}_' if self.file_prefix else ''
        return self.output_dir / f'{prefix}{name}.{ext}'

    def grid_path(self, name: str) -> Path:
        """
        Return the full path for a named grid result file.

        Parameters
        ----------
        name : str
            Base name of the grid file.

        Returns
        -------
        Path
        """
        prefix = f'{self.file_prefix}_' if self.file_prefix else ''
        return self.output_dir / f'{prefix}{name}.txt'

    def table_path(self, fmt: str) -> Path:
        """
        Return the full path for a results table file.

        Parameters
        ----------
        fmt : str
            Table format: ``'ascii'`` or ``'latex'``.

        Returns
        -------
        Path
        """
        ext = 'tex' if fmt == 'latex' else 'txt'
        prefix = f'{self.file_prefix}_' if self.file_prefix else ''
        return self.output_dir / f'{prefix}results.{ext}'

    def exozippy_init_path(self) -> Path:
        """
        Return the full path for the EXOZIPPy initialization JSON file.

        Returns
        -------
        Path
        """
        prefix = f'{self.file_prefix}_' if self.file_prefix else ''
        return self.output_dir / f'{prefix}exozippy_init.json'

# ===========================================================================
# Module-level convenience wrapper
# ===========================================================================

def fit(
    datasets=None,
    files=None,
    fit_type: str = 'point_lens',
    **kwargs,
) -> 'MMEXOFASTFitter':
    """
    Convenience wrapper: construct an ``MMEXOFASTFitter`` and run it.

    Parameters
    ----------
    datasets : list of MulensModel.MulensData, optional
        Pre-loaded dataset objects.
    files : str or list of str, optional
        Data file paths to load.
    fit_type : str
        ``'point_lens'`` or ``'binary_lens'``.
    **kwargs
        Passed directly to ``MMEXOFASTFitter.__init__``.

    Returns
    -------
    MMEXOFASTFitter
        The fitter after ``fit()`` has been called.
    """
    with MMEXOFASTFitter(
        datasets=datasets,
        files=files,
        fit_type=fit_type,
        **kwargs,
    ) as fitter:
        return fitter.fit()


# ===========================================================================
# MMEXOFASTFitter
# ===========================================================================

class MMEXOFASTFitter:
    """
    Orchestrates the full MM-EXOFASTv2 microlensing fitting workflow.

    Parameters
    ----------
    datasets : list of MulensModel.MulensData, optional
        Pre-loaded dataset objects, each with a unique label in
        ``plot_properties['label']``.  Mutually exclusive with *files*.
    files : str or list of str, optional
        Data file paths to load.  Mutually exclusive with *datasets*.
    coords : str or MulensModel.Coordinates, optional
        Sky coordinates of the event.
    fit_type : str
        ``'point_lens'`` or ``'binary_lens'``.
    finite_source : bool
        If True, include FSPL fitting steps after PSPL.
    mag_methods : list, optional
        Magnification methods in MulensModel convention.
    limb_darkening_coeffs_u : dict, optional
        Linear limb-darkening coefficients keyed by bandpass.
    limb_darkening_coeffs_gamma : dict, optional
        Gamma limb-darkening coefficients keyed by bandpass.
    fix_blend_flux : dict, optional
        Mapping of dataset label to blend flux fixing flag.
    fix_source_flux : dict, optional
        Mapping of dataset label to source flux fixing flag.
    renormalize_errors : bool
        Whether to renormalize dataset errors during the workflow.
    parallax_grid : bool
        Whether to run a parallax grid search.
    primary_location : str, optional
        Location name to treat as primary (e.g. ``'ground'``,
        ``'Spitzer'``).
    primary_dataset : str, optional
        Label of dataset used to identify the primary location.
    emcee_settings : dict, optional
        Settings passed to the EMCEE-based binary fitter.
    dry_run : bool
        If True, build the workflow but do not execute any steps.
    stop_before : str, optional
        Name of the step before which execution halts.
    stop_after : str, optional
        Name of the step after which execution halts.
    restart_file : path-like, optional
        Path to a restart pickle file.  If the file exists it is loaded
        to restore previous state.  After every completed step the
        current state is written back to this same path, so the file
        always reflects the latest checkpoint.
    restart_from : str, optional
        Step name from which to re-run; all completed steps recorded
        after this point are discarded.
    initial_results : dict, optional
        User-supplied fit results to seed the workflow.  See
        ``_load_initial_results`` for the expected key/value format.
        Mutually exclusive with ``restart_from``.  Only two entry points
        are supported: ``fit_type='point_lens'`` with a PSPL result
        (starts at ``fit_pspl``), and ``fit_type='binary_lens'`` with any
        PSPL result (starts at ``select_best_point_lens_model``,
        skipping all point-lens stages).
    output_config : OutputConfig, optional
        Controls file output (plots, grid files).  If None, no files are
        written.
    verbose : bool
        If True, configure the module logger to emit DEBUG-level messages
        to stdout.
    log_file : path-like, optional
        Path to a file for DEBUG-level log output.  The file is created
        (or appended to) at construction time.  Call ``close()`` to
        release the file handle when the fitter is no longer needed.


    Notes
    -----
    **Workflow entry points via** ``initial_results``

    When the user supplies pre-computed fit results, the workflow skips
    the steps needed to produce those results.  Only the following
    combinations are supported:

    .. list-table::
       :header-rows: 1
       :widths: 20 25 55

       * - ``fit_type``
         - Result supplied
         - First step executed
       * - ``'point_lens'``
         - Static PSPL (``lens_type=POINT``, ``parallax_branch=NONE``)
         - ``fit_pspl`` — the supplied params are used as the fitting
           seed; ``est_pl_params`` is skipped
       * - ``'binary_lens'``
         - Any PSPL (static or parallax)
         - ``select_best_point_lens_model`` — all point-lens stages are
           skipped; the supplied model is used immediately as the
           reference for the anomaly search

    ``initial_results`` and ``restart_from`` are mutually exclusive;
    providing both raises ``ValueError`` at construction time.

    **Restart behavior**

    When ``restart_file`` is provided, all saved state (fit results,
    completed steps, renormalization factors, datasets) is restored
    before any new steps run.  The ``restart_from`` parameter discards
    all recorded progress at and after the named step, forcing those
    steps to re-run.

    **Stop points**

    ``stop_before`` and ``stop_after`` accept either a stage name
    (e.g. ``'fit_static_point_lens'``) or a ``stage:step`` string
    (e.g. ``'fit_static_point_lens:fit_pspl'``).  When a stage name is
    used, ``stop_before`` halts before the first step of that stage and
    ``stop_after`` halts after the last step of that stage.
    """

    CONFIG_KEYS = [
        'fit_type',
        'coords',
        'finite_source',
        'mag_methods',
        'limb_darkening_coeffs_u',
        'limb_darkening_coeffs_gamma',
        'fix_blend_flux',
        'fix_source_flux',
        'renormalize_errors',
        'parallax_grid',
        'primary_location',
        'primary_dataset',
        'emcee_settings',
        'stop_before',
        'stop_after',
    ]

    PARALLAX_GRID_PARAMS_COARSE = {
        'pi_E_E': [-1.0, 1.0, 0.15],
        'pi_E_N': [-1.5, 1.5, 0.30],
    }

    PARALLAX_GRID_PARAMS_FINE = {
        'pi_E_E': [-0.7, 0.7, 0.025],
        'pi_E_N': [-1.0, 1.0, 0.050],
    }

    RENORM_THRESHOLD = 0.02

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def __init__(
        self,
        datasets=None,
        files=None,
        coords=None,
        fit_type: str = 'point_lens',
        finite_source: bool = False,
        mag_methods=None,
        limb_darkening_coeffs_u=None,
        limb_darkening_coeffs_gamma=None,
        fix_blend_flux=None,
        fix_source_flux=None,
        renormalize_errors: bool = True,
        parallax_grid: bool = False,
        primary_location=None,
        primary_dataset=None,
        emcee_settings=None,
        dry_run: bool = False,
        stop_before: Optional[str] = None,
        stop_after: Optional[str] = None,
        restart_file=None,
        restart_from=None,
        initial_results=None,
        output_config=None,
        verbose: bool = False,
        log_file=None,
    ) -> None:
        # Mutually exclusive input validation
        if files is not None and datasets is not None:
            raise ValueError(
                "Specify 'files' or 'datasets', not both."
            )
        if initial_results is not None and restart_from is not None:
            raise ValueError(
                "Specify 'initial_results' or 'restart_from', not both."
            )

        # Track handlers added by this instance for cleanup via close()
        self._log_handlers: list[logging.Handler] = []

        # verbose / log_file → configure module logger
        if verbose or log_file is not None:
            _mod_logger = logging.getLogger(__name__)
            _mod_logger.setLevel(logging.DEBUG)
            if verbose:
                _handler = logging.StreamHandler()
                _mod_logger.addHandler(_handler)
                self._log_handlers.append(_handler)
            if log_file is not None:
                _handler = logging.FileHandler(log_file)
                _mod_logger.addHandler(_handler)
                self._log_handlers.append(_handler)

        # Output config
        self._output_config: Optional[OutputConfig] = output_config

        # Config from restart file merged with current call
        saved_config, saved_state = self._load_restart_data(restart_file)
        self._restart_path = Path(restart_file) if restart_file is not None else None
        config = self._merge_config(saved_config, locals())
        self._set_config_attributes(config)

        # Execution-time controls (not persisted in CONFIG_KEYS)
        self.dry_run = dry_run

        # WorkflowStep tracking
        self.completed_steps: list[WorkflowStep] = []
        self.planned_steps: list[WorkflowStep] = []

        # Restore computed state
        self._restore_state(saved_state)

        # Truncate completed_steps if restart_from is specified
        if restart_from is not None:
            idx = next(
                (i for i, s in enumerate(self.completed_steps)
                 if self._step_matches_stop_value(restart_from, s)),
                None,
            )
            if idx is not None:
                self.completed_steps = self.completed_steps[:idx]

        # Dataset construction
        if files is not None:
            self.datasets = self._create_mulensdata_objects(
                files, saved_datasets=saved_state.get('datasets')
            )
        elif datasets is not None:
            self.datasets = datasets
            self._validate_dataset_labels()
        elif saved_state.get('datasets'):
            self.datasets = saved_state['datasets']
        else:
            raise ValueError(
                "Provide at least one of: 'files', 'datasets', or "
                "'restart_file'."
            )

        self._check_dataset_labels_unique()

        # Flux-fixing maps (depend on self.fix_blend_flux /
        # self.fix_source_flux set by _set_config_attributes above)
        self.fix_blend_flux_map = self._map_label_dict_to_datasets(
            self.fix_blend_flux
        )
        self.fix_source_flux_map = self._map_label_dict_to_datasets(
            self.fix_source_flux
        )

        # Load initial results and infer entry point
        self._initial_entry_point: Optional[str] = None
        if initial_results is not None:
            self._load_initial_results(initial_results)
            if not self.completed_steps:
                self._initial_entry_point = (
                    self._infer_entry_point_from_initial_results()
                )

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _load_restart_data(self, restart_file) -> tuple[dict, dict]:
        """
        Load a saved restart file.

        Parameters
        ----------
        restart_file : path-like or None
            Path to the restart pickle file.

        Returns
        -------
        saved_config : dict
            Configuration dict stored at save time.
        saved_state : dict
            Runtime state dict stored at save time.
        """
        if restart_file is None:
            return {}, {}
        if not Path(restart_file).exists():
            logger.info(
                'No restart file found at %s; starting fresh.', restart_file
            )
            return {}, {}

        logger.info('Loading restart data from: %s', restart_file)
        with open(restart_file, 'rb') as f:
            data = pickle.load(f)

        config = data.get('config', {})
        state = data.get('state', {})
        logger.info(
            '  Loaded %d fit result(s).',
            len(state.get('all_fit_results', AllFitResults())),
        )
        return config, state

    def _merge_config(self, saved_config: dict, locals_: dict) -> dict:
        """
        Merge a saved config with the current call's local config dict.

        Current call values win on conflict.

        Parameters
        ----------
        saved_config : dict
            Configuration loaded from the restart file.
        locals_ : dict
            Configuration derived from the current ``__init__`` arguments.

        Returns
        -------
        dict
            Merged configuration dict.
        """
        merged = {}
        for key in self.CONFIG_KEYS:
            if key in locals_ and locals_[key] is not None:
                merged[key] = locals_[key]
            elif key in saved_config:
                merged[key] = saved_config[key]
            else:
                merged[key] = None
        return merged

    def _set_config_attributes(self, config: dict) -> None:
        """
        Apply a config dict as attributes on self.

        Parameters
        ----------
        config : dict
            Key-value pairs to set as instance attributes.
        """
        for key in self.CONFIG_KEYS:
            setattr(self, key, config[key])

    def _restore_state(self, saved_state: dict) -> None:
        """
        Restore runtime state from a serialized dict.

        Parameters
        ----------
        saved_state : dict
            State dict previously produced by ``_get_state()``.

        Notes
        -----
        ``completed_steps`` may be stored as ``(name, stage)`` tuples in
        the restart file because ``WorkflowStep.func`` callables cannot be
        pickled.  Each tuple is reconstructed as a stub ``WorkflowStep``
        with a no-op func sufficient for tracking purposes.
        """
        self.all_fit_results = saved_state.get(
            'all_fit_results', AllFitResults()
        )

        raw_completed = saved_state.get('completed_steps', [])
        self.completed_steps = [
            step if isinstance(step, WorkflowStep)
            else WorkflowStep(
                name=step[0],
                func=lambda: None,
                stage=step[1],
                description='(restored from restart file)',
            )
            for step in raw_completed
        ]

        self.intermediate_results = saved_state.get(
            'intermediate_results', IntermediateResults()
        )
        self.renorm_factors: dict = saved_state.get('renorm_factors', {})

    def _get_state(self) -> dict:
        """
        Return a serializable dict of current runtime state.

        Returns
        -------
        dict
            Contains ``all_fit_results``, ``completed_steps`` (as
            ``(name, stage)`` tuples since callables cannot be pickled),
            ``intermediate_results``, ``renorm_factors``, and ``datasets``.
        """
        return {
            'all_fit_results': self.all_fit_results,
            'completed_steps': [
                (s.name, s.stage) for s in self.completed_steps
            ],
            'intermediate_results': self.intermediate_results,
            'renorm_factors': self.renorm_factors,
            'datasets': self.datasets,
        }

    def _load_initial_results(self, initial_results) -> None:
        """
        Load user-supplied fit results into ``self.all_fit_results``.

        Parameters
        ----------
        initial_results : dict
            Mapping of model label strings to payload dicts.  Each
            payload must contain ``'params'`` and may contain
            ``'sigmas'``, ``'renorm_factors'``, and ``'fixed'``.
        """
        for label, payload in initial_results.items():
            key = mmexo.fit_types.label_to_model_key(label)
            record = mmexo.FitRecord(
                model_key=key,
                params=payload['params'],
                sigmas=payload.get('sigmas'),
                renorm_factors=payload.get('renorm_factors'),
                full_result=None,
                fixed=payload.get('fixed', False),
                is_complete=False,
            )
            self.all_fit_results.set(record)

    # ------------------------------------------------------------------
    # Workflow execution
    # ------------------------------------------------------------------

    def fit(self) -> AllFitResults:
        """
        Main entry point; build and execute the workflow steps.

        Returns
        -------
        AllFitResults
            All fit records accumulated during the workflow.

        Raises
        ------
        ValueError
            If ``fit_type`` is not set or is unrecognized.
        """
        if self.fit_type is None:
            raise ValueError(
                "fit_type must be set before calling fit(): "
                "'point_lens' or 'binary_lens'."
            )

        self.planned_steps = self._build_remaining_steps()
        logger.info(
            '\nPlanned workflow: \n%s\n', '\n'.join(
                ['{0}: {1}'.format(step.stage, step.name) for step in self.planned_steps]))
        i = 0
        while i < len(self.planned_steps):
            step = self.planned_steps[i]

            #debugging:
            #logger.info(f'\nDEBUG running step: {step.stage}:{step.name}')
            #logger.info(f'DEBUG remaining steps: %s', [f'{s.stage}:{s.name}' for s in self.planned_steps])

            if (self.stop_before is not None
                    and self._matches_stop_point(
                        self.stop_before, step, mode='before'
                    )):
                logger.info("Stopping before step '%s'.", step.name)
                break

            if self.dry_run:
                logger.info('[dry_run] Would execute: %s', step.name)
            else:
                step.run()
                self.completed_steps.append(step)

                # Insert any dynamically generated follow-up steps
                if isinstance(step.result, list) and step.result:
                    #logger.info('DEBUG: steps to insert: %s', [f'{s.stage}:{s.name}' for s in step.result])

                    for j, dynamic_step in enumerate(step.result):
                        self.planned_steps.insert(i + 1 + j, dynamic_step)

                self._save_restart_state()
                logger.info("\n")

            # Lookahead uses the queue *after* any dynamic insertions
            remaining = self.planned_steps[i + 1:]
            if (self.stop_after is not None
                    and self._matches_stop_point(
                        self.stop_after, step, mode='after',
                        remaining_steps=remaining
                    )):
                logger.info("Stopping after step '%s'.", step.name)
                break

            i += 1

        if self._output_config is not None:
            if self._output_config.save_table:
                for fmt in self._output_config.table_formats:
                    table_str = self.make_ulens_table(table_type=fmt)
                    path = self._output_config.table_path(fmt)
                    path.write_text(table_str)
                    logger.info('Saved %s results table to %s.', fmt, path)

            if self._output_config.save_exozippy_init:
                path = self._output_config.exozippy_init_path()
                with open(path, 'w') as f:
                    json.dump(self.initialize_exozippy(), f)
                logger.info('Saved EXOZIPPy init data to %s.', path)

            if self._output_config.save_plots:
                self._plot_best_fit_event()

        return self.all_fit_results

    def _build_remaining_steps(self) -> list[WorkflowStep]:
        """
        Build the planned step queue, skipping already completed steps.

        Steps are identified by ``(name, stage)`` pairs to allow the same
        step name to appear in multiple stages (e.g. ``renormalize_datasets``
        appears in both ``'renormalize'`` and ``'check_binary_renorm'``).

        Returns
        -------
        list of WorkflowStep
            Ordered steps remaining to be executed.

        Raises
        ------
        ValueError
            If ``fit_type`` is unrecognized.
        """
        completed_ids = {
            (step.name, step.stage) for step in self.completed_steps
        }

        if self.fit_type == 'point_lens':
            all_steps = self._build_point_lens_steps()
        elif self.fit_type == 'binary_lens':
            all_steps = self._build_binary_lens_steps()
        else:
            raise ValueError(
                f"Unknown fit_type {self.fit_type!r}. "
                "Expected 'point_lens' or 'binary_lens'."
            )

        if self._initial_entry_point is not None:
            entry_idx = next(
                (i for i, s in enumerate(all_steps)
                 if s.name == self._initial_entry_point),
                0,
            )
            all_steps = all_steps[entry_idx:]

        return [
            s for s in all_steps
            if (s.name, s.stage) not in completed_ids
        ]

    def _build_common_point_lens_steps(self) -> list[WorkflowStep]:
        """
        Build the point-lens steps shared by both the point-lens and
        binary-lens workflows.

        Includes event search, static fitting, parallax fitting, and
        (when enabled) renormalization.  Does not include the parallax
        grid search.

        Returns
        -------
        list of WorkflowStep
        """
        steps: list[WorkflowStep] = []
        steps.extend(self._build_event_search_steps())
        steps.extend(self._build_static_fit_steps())
        steps.extend(self._build_parallax_steps())
        if self.renormalize_errors:
            steps.extend(self._build_renormalize_steps())
        return steps

    def _build_point_lens_steps(self) -> list[WorkflowStep]:
        """
        Build the step list for a point-lens workflow.

        Returns
        -------
        list of WorkflowStep
        """
        steps = self._build_common_point_lens_steps()
        if self.parallax_grid:
            steps.extend(self._build_parallax_grid_steps())

        return steps

    def _build_binary_lens_steps(self) -> list[WorkflowStep]:
        """
        Build the step list for a binary-lens workflow.

        Returns
        -------
        list of WorkflowStep
        """

        if self._initial_entry_point != 'search_for_anomaly':
            steps = self._build_common_point_lens_steps()
        else:
            steps: list[WorkflowStep] = []

        steps.extend(self._build_anomaly_search_steps())
        steps.extend(self._build_binary_fit_steps())
        if self.renormalize_errors:
            steps.extend(self._build_check_binary_renorm_steps())
        if self.parallax_grid:
            steps.extend(self._build_parallax_grid_steps())

        return steps

    def _build_event_search_steps(self) -> list[WorkflowStep]:
        return [
            WorkflowStep(
                name='run_ef_grid',
                func=self.run_ef_grid,
                stage='event_search',
                description='Run EventFinder grid search',
            ),
        ]

    def _build_static_fit_steps(self) -> list[WorkflowStep]:
        """
        Build steps covering the static (no-parallax) fit.

        If a PSPL record already exists in ``all_fit_results`` (e.g. from
        user-supplied ``initial_results``), its params are passed to
        ``fit_pspl`` as the initial seed.

        Includes an FSPL step when ``self.finite_source`` is True.

        Returns
        -------
        list of WorkflowStep
        """
        if self._initial_entry_point == 'fit_pspl':
            static_pspl_key = mmexo.FitKey(
                lens_type=mmexo.LensType.POINT,
                source_type=mmexo.SourceType.POINT,
                parallax_branch=mmexo.ParallaxBranch.NONE,
                lens_orb_motion=mmexo.LensOrbMotion.NONE,
            )
            existing = self.all_fit_results.get(static_pspl_key)
            pspl_seed = existing.params
            steps = []
        else:
            steps = [
                WorkflowStep(
                    name='est_pl_params',
                    func=self.est_pl_params,
                    stage='fit_static_point_lens',
                    description='Estimate point-lens parameters from EF grid result',
                )]
            pspl_seed = None

        steps.append(
            WorkflowStep(
                name='fit_pspl',
                func=lambda p=pspl_seed: self.fit_pspl(initial_params=p),
                stage='fit_static_point_lens',
                description='Fit static PSPL model',
            ),
        )

        if self.finite_source:
            steps.append(
                WorkflowStep(
                    name='fit_fspl',
                    func=self.fit_fspl,
                    stage='fit_static_point_lens',
                    description='Fit static FSPL model',
                )
            )
        return steps

    def _build_parallax_steps(self) -> list[WorkflowStep]:
        """
        Build one WorkflowStep per parallax branch.

        Returns
        -------
        list of WorkflowStep
        """
        steps = []
        for key in self._iter_parallax_point_lens_keys():
            branch = key.parallax_branch
            name = f'fit_parallax_{branch.value.lower()}'
            steps.append(
                WorkflowStep(
                    name=name,
                    func=lambda b=branch: self.fit_parallax(branch=b),
                    stage='fit_point_lens_parallax',
                    description=f'Fit parallax model for branch {branch.value}',
                )
            )
        return steps

    def _build_renormalize_steps(
            self,
            stage: str = 'renormalize',
    ) -> list[WorkflowStep]:
        """
        Build steps covering error renormalization.

        Parameters
        ----------
        stage : str
            Stage name to assign to the returned steps.  Defaults to
            ``'renormalize'``.  Pass ``'check_binary_renorm'`` when these
            steps are inserted dynamically after the binary fit.

        Returns
        -------
        list of WorkflowStep
        """
        return [
            WorkflowStep(
                name='renormalize_datasets',
                func=self.renormalize_datasets,
                stage=stage,
                description=(
                    'Remove outliers and compute per-dataset error '
                    'rescaling factors'
                ),
            ),
            WorkflowStep(
                name='refit_all',
                func=self.refit_all,
                stage=stage,
                description='Refit all stored fits with updated error normalization',
            ),
        ]

    def _build_anomaly_search_steps(self) -> list[WorkflowStep]:
        """
        Build steps covering the AnomalyFinder grid search.

        Returns
        -------
        list of WorkflowStep
        """
        return [
            WorkflowStep(
                name='select_best_point_lens_model',
                func=self.select_best_point_lens_model,
                stage='search_for_anomaly',
                description='Select the best point-lens model for anomaly search',
            ),
            WorkflowStep(
                name='compute_residuals',
                func=self.compute_residuals,
                stage='search_for_anomaly',
                description='Compute residuals from best point-lens model',
            ),
            WorkflowStep(
                name='run_af_grid',
                func=self.run_af_grid,
                stage='search_for_anomaly',
                description='Run AnomalyFinder grid search',
            ),
            WorkflowStep(
                name='get_anomaly_lc_params',
                func=self.get_anomaly_lc_params,
                stage='search_for_anomaly',
                description='Measure observable anomaly properties',
            ),
            WorkflowStep(
                name='classify_anomaly',
                func=self.classify_anomaly,
                stage='search_for_anomaly',
                description='Classify the anomaly type',
            ),
        ]

    def _build_binary_fit_steps(self) -> list[WorkflowStep]:
        return [
            WorkflowStep(
                name='est_binary_params',
                func=self.est_binary_params,
                stage='fit_binary_lens',
                description='Estimate binary-lens parameters from AF grid result',
            ),
            WorkflowStep(
                name='fit_binary_models',
                func=self.fit_binary_models,
                stage='fit_binary_lens',
                description=(
                    'Fit binary-lens models; may return dynamic follow-up steps'
                ),
            ),
        ]

    def _build_check_binary_renorm_steps(self) -> list[WorkflowStep]:
        return [
            WorkflowStep(
                name='check_needs_renorm',
                func=self.check_needs_renorm,
                stage='check_binary_renorm',
                description='Check whether binary fits require renormalization',
            ),
        ]

    def _build_parallax_grid_steps(self) -> list[WorkflowStep]:
        """
        Build one WorkflowStep per parallax branch for the grid search.

        Always yields two steps (U0_PLUS and U0_MINUS), regardless of the
        number of observing locations.

        Returns
        -------
        list of WorkflowStep
        """
        steps = [
            WorkflowStep(
                name=f'run_parallax_grids',
                func=self.run_parallax_grids,
                stage='parallax_grids',
                description=f'Run both parallax grid searches',
            )
        ]
        return steps

    def _step_matches_stop_value(
            self,
            stop_value: str,
            step: WorkflowStep,
    ) -> bool:
        """
        Return True if *step* matches a stop value string.

        Handles both ``'stage'`` and ``'stage:step'`` syntax without any
        mode-specific lookahead logic.

        Parameters
        ----------
        stop_value : str
            A stage name or ``stage:step`` string.
        step : WorkflowStep
            The step to test.

        Returns
        -------
        bool
        """
        if ':' in stop_value:
            stage_name, step_name = stop_value.split(':', 1)
            return step.stage == stage_name and step.name == step_name
        return step.stage == stop_value

    def _matches_stop_point(
            self,
            stop_value: str,
            step: WorkflowStep,
            mode: str,
            remaining_steps: Optional[list[WorkflowStep]] = None,
    ) -> bool:
        if not self._step_matches_stop_value(stop_value, step):
            return False

        if ':' in stop_value:
            return True  # stage:step — exact match, no lookahead needed

        # Stage-only: mode-specific logic
        if mode == 'before':
            return not any(
                s.stage == step.stage for s in self.completed_steps
            )
        # mode == 'after'
        if remaining_steps is None:
            return True
        return not any(s.stage == step.stage for s in remaining_steps)

    # ------------------------------------------------------------------
    # Step action methods
    # ------------------------------------------------------------------

    def run_ef_grid(self) -> None:
        """
        Run the EventFinder grid.

        Stores the best grid point in
        ``self.intermediate_results.best_ef_grid_point``.
        """
        ef_grid = mmexo.EventFinderGridSearch(datasets=self.datasets)
        ef_grid.run()

        if (self._output_config is not None
                and self._output_config.save_plots):
            fig = ef_grid.plot()
            fig.savefig(self._output_config.plot_path('ef_grid'))
            plt.close(fig)
            logger.info(
                'Saved EF grid plot to %s.',
                self._output_config.plot_path('ef_grid'),
            )

        logger.info('Best EF grid point: %s', ef_grid.best)
        self.intermediate_results.best_ef_grid_point = ef_grid.best

    def est_pl_params(self) -> None:
        """
        Estimate point-lens parameters from the EventFinder grid result.

        Stores estimates in ``self.intermediate_results.est_pl_params``.
        """
        est = mmexo.estimate_params.get_PSPL_params(
            self.intermediate_results.best_ef_grid_point,
            self.datasets,
        )
        logger.info('Estimated point-lens params: %s', est)
        self.intermediate_results.est_pl_params = est

    def fit_pspl(self, initial_params=None) -> None:
        """
        Fit a static PSPL model.

        Parameters
        ----------
        initial_params : dict, optional
            Starting parameter values.  If None, uses
            ``self.intermediate_results.est_pl_params``.

        Notes
        -----
        Stores the resulting ``FitRecord`` in ``self.all_fit_results``.
        """
        if initial_params is None:
            initial_params = self.intermediate_results.est_pl_params

        fitter = mmexo.fitters.SFitFitter(
            initial_model_params=initial_params,
            datasets=self.datasets,
            **self._get_fitter_kwargs(
                source_type=mmexo.SourceType.POINT
            ),
        )
        fitter.run()
        logger.info('Static PSPL: %s', fitter.best)
        logger.info('    sigmas:  %s', list(fitter.results.sigmas))

        key = mmexo.FitKey(
            lens_type=mmexo.LensType.POINT,
            source_type=mmexo.SourceType.POINT,
            parallax_branch=mmexo.ParallaxBranch.NONE,
            lens_orb_motion=mmexo.LensOrbMotion.NONE,
        )
        self.all_fit_results.set(
            mmexo.FitRecord.from_full_result(
                model_key=key,
                full_result=mmexo.MMEXOFASTFitResults(fitter),
                renorm_factors=self.renorm_factors,
                fixed=False,
            )
        )

    def fit_fspl(self, initial_params=None) -> None:
        """
        Fit a static FSPL model.

        Parameters
        ----------
        initial_params : dict, optional
            Starting parameter values.  If None, seeds from the PSPL
            result in ``self.all_fit_results`` with
            ``rho = 1.5 * u_0``.

        Notes
        -----
        Stores the resulting ``FitRecord`` in ``self.all_fit_results``.
        Requires a static PSPL result to already exist.

        Raises
        ------
        RuntimeError
            If no static PSPL fit exists to seed from.
        """
        if initial_params is None:
            pspl_key = mmexo.FitKey(
                lens_type=mmexo.LensType.POINT,
                source_type=mmexo.SourceType.POINT,
                parallax_branch=mmexo.ParallaxBranch.NONE,
                lens_orb_motion=mmexo.LensOrbMotion.NONE,
            )
            pspl_record = self.all_fit_results.get(pspl_key)
            if pspl_record is None:
                raise RuntimeError(
                    'A static PSPL fit must exist before fitting FSPL.'
                )
            initial_params = dict(pspl_record.params)
            initial_params['rho'] = 1.5 * initial_params['u_0']

        fitter = mmexo.fitters.SFitFitter(
            initial_model_params=initial_params,
            datasets=self.datasets,
            **self._get_fitter_kwargs(),
        )
        fitter.run()
        logger.info('Static FSPL: %s', fitter.best)
        logger.info('    sigmas:  %s', list(fitter.results.sigmas))

        key = mmexo.FitKey(
            lens_type=mmexo.LensType.POINT,
            source_type=mmexo.SourceType.FINITE,
            parallax_branch=mmexo.ParallaxBranch.NONE,
            lens_orb_motion=mmexo.LensOrbMotion.NONE,
        )
        self.all_fit_results.set(
            mmexo.FitRecord.from_full_result(
                model_key=key,
                full_result=mmexo.MMEXOFASTFitResults(fitter),
                renorm_factors=self.renorm_factors,
                fixed=False,
            )
        )

    def fit_parallax(self, branch=None) -> None:
        """
        Fit a parallax model for the given branch.

        Parameters
        ----------
        branch : mmexo.ParallaxBranch, optional
            Which parallax branch to fit.  If None, fits all branches
            appropriate for the current data via
            ``_iter_parallax_point_lens_keys()``.

        Notes
        -----
        Stores each resulting ``FitRecord`` in ``self.all_fit_results``.
        """
        if branch is not None:
            source_type = (
                mmexo.SourceType.FINITE
                if self.finite_source
                else mmexo.SourceType.POINT
            )
            keys = [
                mmexo.FitKey(
                    lens_type=mmexo.LensType.POINT,
                    source_type=source_type,
                    parallax_branch=branch,
                    lens_orb_motion=mmexo.LensOrbMotion.NONE,
                )
            ]
        else:
            keys = list(self._iter_parallax_point_lens_keys())

        for key in keys:
            seed = self._get_parallax_seed_params(key)
            result = self._do_parallax_fit(
                seed, source_type=key.source_type
            )
            if result is not None:
                self.all_fit_results.set(
                    mmexo.FitRecord.from_full_result(
                        model_key=key,
                        full_result=result,
                        renorm_factors=self.renorm_factors,
                        fixed=False,
                    )
                )
                logger.info(
                    'Parallax %s: chi2=%.2f',
                    key.parallax_branch.value,
                    result.chi2,
                )

    def renormalize_datasets(self) -> None:
        """
        Run outlier rejection and compute per-dataset error rescaling
        factors.

        Notes
        -----
        Datasets already present in ``self.renorm_factors`` are skipped.
        Combines logic from the old ``_remove_outliers_and_calc_errfacs``
        and ``_apply_error_renormalization``.  Rebuilds
        ``fix_blend_flux_map`` and ``fix_source_flux_map`` after
        replacing dataset objects.
        """
        reference_fit = self.select_best_point_lens_model()
        reference_model = reference_fit.full_result.fitter.get_model()
        logger.info(
            'Renormalizing using: %s',
            mmexo.fit_types.model_key_to_label(reference_fit.model_key),
        )

        event = MulensModel.Event(
            datasets=self.datasets,
            model=reference_model,
            coords=self.coords,
        )
        event.fit_fluxes()

        sig = inspect.signature(MulensModel.MulensData.__init__)
        new_datasets: list = []

        for i, dataset in enumerate(self.datasets):
            label = dataset.plot_properties['label']

            if label in self.renorm_factors:
                new_datasets.append(dataset)
                continue

            n_params = len(reference_model.parameters.as_dict())
            bad_index: Any = -1

            # Iterative outlier removal
            while bad_index is not None:
                event.fit_fluxes()
                n_good = int(np.sum(dataset.good))
                dof = n_good - n_params
                if dof <= 0:
                    break
                max_sig = max(
                    np.sqrt(2.0) * erfcinv(1.0 / dof), 3.0
                )
                chi2 = event.get_chi2_for_dataset(i)
                errfac = np.sqrt(chi2 / dof)
                res, err = event.fits[i].get_residuals(
                    phot_fmt='flux', bad=True
                )
                sigma = np.abs(res / (err * errfac))
                if np.any(sigma[dataset.good] > max_sig):
                    i_worst = np.argmax(sigma[dataset.good])
                    bad_idx = np.argwhere(
                        sigma == sigma[dataset.good][i_worst]
                    )[0]
                    new_bad = dataset.bad.copy()
                    new_bad[bad_idx] = True
                    dataset.bad = new_bad
                    bad_index = bad_idx
                else:
                    bad_index = None

            # Final error factor
            event.fit_fluxes()
            final_chi2 = event.get_chi2_for_dataset(i)
            final_dof = int(np.sum(dataset.good)) - n_params
            errfac = (
                np.sqrt(final_chi2 / final_dof)
                if final_dof > 0
                else 1.0
            )
            logger.info('  %s: errfac=%.3f', label, errfac)

            # Recreate dataset with scaled errors
            kwargs = {
                k: getattr(dataset, k)
                for k in sig.parameters
                if k not in (
                    'self', 'data_list', 'good',
                    'phot_fmt', 'file_name',
                )
                and hasattr(dataset, k)
            }
            new_datasets.append(
                MulensModel.MulensData(
                    data_list=[
                        dataset.time,
                        dataset.flux,
                        errfac * dataset.err_flux,
                    ],
                    phot_fmt='flux',
                    **kwargs,
                )
            )
            self.renorm_factors[label] = errfac

        self.datasets = new_datasets

        # Rebuild flux-fixing maps with updated dataset objects
        self.fix_blend_flux_map = self._map_label_dict_to_datasets(
            self.fix_blend_flux
        )
        self.fix_source_flux_map = self._map_label_dict_to_datasets(
            self.fix_source_flux
        )

    def refit_all(self) -> None:
        """
        Refit all stored fits using updated error normalization.
        """
        logger.info('Refitting all stored models...')
        for key, fit_record in self.all_fit_results.items():
            fitter = fit_record.full_result.fitter
            fitter.datasets = self.datasets
            fitter.initial_model_params = fit_record.params
            fitter.run()
            self.all_fit_results.set(
                mmexo.FitRecord.from_full_result(
                    model_key=key,
                    full_result=mmexo.MMEXOFASTFitResults(fitter),
                    renorm_factors=self.renorm_factors,
                    fixed=False,
                )
            )
            logger.info(
                '%s: %s',
                mmexo.fit_types.model_key_to_label(key),
                fitter.best,
            )
            logger.info('    sigmas:  %s', list(fitter.results.sigmas))

    def select_best_point_lens_model(self) -> FitRecord:
        """
        Return the best point-lens ``FitRecord`` from
        ``all_fit_results``.

        Prefers the best parallax model when its chi-squared improvement
        over the best static model exceeds 50; otherwise returns the best
        static model.

        If no complete point-lens fits exist but exactly one point-lens
        record is present (e.g. a user-supplied initial guess), that
        record is returned directly regardless of chi-squared.

        Returns
        -------
        FitRecord

        Raises
        ------
        RuntimeError
            If no point-lens fits are available.
        """
        DELTA_CHI2 = 50.0

        point_lens_fits = [
            rec for key, rec in self.all_fit_results.items()
            if key.lens_type == mmexo.LensType.POINT
        ]

        if not point_lens_fits:
            raise RuntimeError('No point-lens fits found in all_fit_results.')

        if self._initial_entry_point == 'search_for_anomaly':
            return point_lens_fits[0]

        complete_fits = [rec for rec in point_lens_fits if rec.chi2() is not None]

        # If no complete fits exist, return the sole available record
        # (e.g. a user-supplied initial guess without chi2)
        if not complete_fits:
            raise RuntimeError(
                'No complete point-lens fits found and more than one '
                'incomplete record exists — cannot select best model.'
            )

        static_fits = [
            rec for key, rec in self.all_fit_results.items()
            if key.lens_type == mmexo.LensType.POINT
               and key.parallax_branch == mmexo.ParallaxBranch.NONE
               and rec.chi2() is not None
        ]
        parallax_fits = [
            rec for key, rec in self.all_fit_results.items()
            if key.lens_type == mmexo.LensType.POINT
               and key.parallax_branch != mmexo.ParallaxBranch.NONE
               and rec.chi2() is not None
        ]

        best_static = (
            min(static_fits, key=lambda r: r.chi2()) if static_fits else None
        )
        best_par = (
            min(parallax_fits, key=lambda r: r.chi2()) if parallax_fits else None
        )

        if best_par is None:
            return best_static
        if best_static is None:
            return best_par
        if best_static.chi2() - best_par.chi2() > DELTA_CHI2:
            return best_par
        return best_static

    def compute_residuals(self) -> None:
        """
        Compute residuals from the best point-lens model.

        Notes
        -----
        Residuals are stored in ``self._residuals`` as a list of
        ``MulensData`` objects in flux format, for use by
        ``run_af_grid()``.
        """
        best = self.select_best_point_lens_model()
        event = MulensModel.Event(
            datasets=self.datasets,
            model=MulensModel.Model(best.params),
            coords=self.coords
        )
        event.fit_fluxes()

        self._residuals: list = []
        for i, dataset in enumerate(self.datasets):
            res, err = event.fits[i].get_residuals(phot_fmt='flux')
            self._residuals.append(
                MulensModel.MulensData(
                    [dataset.time, res, err],
                    phot_fmt='flux',
                    bandpass=dataset.bandpass,
                    ephemerides_file=dataset.ephemerides_file,
                )
            )

    def run_af_grid(self) -> None:
        """
        Run the AnomalyFinder grid.

        Stores the best grid point in
        ``self.intermediate_results.best_af_grid_point``.
        """
        af_grid = mmexo.AnomalyFinderGridSearch(
            residuals=self._residuals
        )
        af_grid.run()
        logger.info('Best AF grid point: %s', af_grid.best)
        self.intermediate_results.best_af_grid_point = af_grid.best

    def get_anomaly_lc_params(self):
        """
        Estimate anomaly properties from the AnomalyFinder grid
        result.

        Stores estimates in
        ``self.intermediate_results.anomaly_lc_params``.
        """
        best_pspl = self.select_best_point_lens_model()
        estimator = mmexo.estimate_params.AnomalyPropertyEstimator(
            datasets=self.datasets,
            pspl_params=best_pspl.params,
            af_results=self.intermediate_results.best_af_grid_point,
        )
        params = estimator.get_anomaly_lc_parameters()
        logger.info('Estimated anomaly params: %s', params)
        self.intermediate_results.anomaly_lc_params = params

    def classify_anomaly(self) -> None:
        """
        Use estimated anomaly properties from the AnomalyFinder grid
        result to classify the anomaly.

        Stores estimates in
        ``self.intermediate_results.anomaly_type``.
        """
        classifier = mmexo.AnomalyClassifier()
        self.intermediate_results.anomaly_type = classifier.classify(self.intermediate_results.anomaly_lc_params)
        logger.info('Anomaly classified as anomaly_type = %s', self.intermediate_results.anomaly_type)

    def est_binary_params(self) -> None:
        """
        Estimate binary-lens parameters from the AnomalyFinder grid
        result.

        Stores estimates in
        ``self.intermediate_results.est_binary_params``.
        """

        if self.intermediate_results.anomaly_type == 'wide':
            est_params = {}

            estimator = mmexo.estimate_params.WidePlanetGridSearchEstimator(
                datasets=self.datasets,
                params=self.intermediate_results.anomaly_lc_params,
            )
            estimator.run()
            params = estimator.binary_params
            logger.info('Estimated binary params: %s', params.ulens)
            logger.info('mag_methods: %s', params.mag_methods)
            est_params['wide'] = params

        elif self.intermediate_results.anomaly_type == 'close':
            est_params = {}

            estimator = mmexo.estimate_params.ClosePlanetGridSearchEstimator(
                datasets=self.datasets,
                params=self.intermediate_results.anomaly_lc_params,
            )
            estimator.run()
            params = estimator.binary_params
            logger.info('Estimated binary params: %s', params.ulens)
            logger.info('mag_methods: %s', params.mag_methods)
            est_params['close'] = params
        else:
            est_params = None
            logger.info('Binary params estimate not implemented for ', self.intermediate_results.anomaly_type)

        self.intermediate_results.est_binary_params = est_params
        if (self._output_config is not None) and self._output_config.save_plots:
            self._plot_initial_2L1S_guess()

    def fit_binary_models(self) -> Optional[list[WorkflowStep]]:
        """
        Fit binary lens models.

        Returns
        -------
        list of WorkflowStep or None
            Dynamically generated follow-up steps, or None if no
            additional steps are required.
        """
        best_pspl = self.select_best_point_lens_model()
        sigmas = dict(best_pspl.sigmas)

        t_E = self.intermediate_results.est_binary_params['t_E']
        u_0 = self.intermediate_results.est_binary_params['u_0']
        max_sigma_u_0 = 0.1
        max_sigma_t_E = max_sigma_u_0 * t_E / u_0
        sigmas['u_0'] = min(sigmas['u_0'], max_sigma_u_0)
        sigmas['t_E'] = min(sigmas['t_E'], max_sigma_t_E)

        wide_planet_fitter = mmexo.fitters.WidePlanetFitter(
            datasets=self.datasets,
            anomaly_lc_params=self.intermediate_results.est_binary_params,
            sigmas=sigmas,
            emcee_settings=self.emcee_settings,
        )
        logger.info(
            'Initial 2L1S Wide Model: %s',
            wide_planet_fitter.initial_model,
        )
        wide_planet_fitter.run()

        key = mmexo.FitKey(
            lens_type=mmexo.LensType.BINARY,
            source_type=(
                mmexo.SourceType.FINITE
                if self.finite_source
                else mmexo.SourceType.POINT
            ),
            parallax_branch=mmexo.ParallaxBranch.NONE,
            lens_orb_motion=mmexo.LensOrbMotion.NONE,
        )
        self.all_fit_results.set(
            mmexo.FitRecord.from_full_result(
                model_key=key,
                full_result=mmexo.EmceeFitResults(wide_planet_fitter),
                renorm_factors=self.renorm_factors,
                fixed=False,
            )
        )
        return None

    def check_needs_renorm(self) -> Optional[list[WorkflowStep]]:
        """
        Check whether renormalization is needed after binary fits.

        Returns
        -------
        list of WorkflowStep or None
            Dynamically generated renormalization steps with stage
            ``'check_binary_renorm'``, or None if renormalization is
            not required.
        """
        if self._needs_renormalization():
            logger.info(
                'Renormalization required after binary fit; inserting steps.'
            )
            return self._build_renormalize_steps(stage='check_binary_renorm')

        return None

    def run_parallax_grids(self, branch=None) -> None:
        """
        Run a parallax grid search for the given branch.

        Parameters
        ----------
        branch : mmexo.ParallaxBranch, optional
            Which parallax branch to search.  If None, searches both
            U0_PLUS and U0_MINUS.
        """
        branches = (
            [mmexo.ParallaxBranch.U0_PLUS, mmexo.ParallaxBranch.U0_MINUS]
            if branch is None
            else [branch]
        )

        reference_fit = self.select_best_point_lens_model()
        static_params = (
            reference_fit.full_result.fitter.get_model()
            .parameters.parameters
        )
        source_type = reference_fit.model_key.source_type

        grids: dict = {}
        for par_branch in branches:
            logger.info(
                'Running parallax grid for %s.', par_branch.value
            )
            grid = mmexo.ParallaxGridSearch(
                static_params,
                datasets=self.datasets,
                grid_params=self.PARALLAX_GRID_PARAMS_COARSE,
                fitter_kwargs=self._get_fitter_kwargs(
                    source_type=source_type
                ),
                skip_optimization=False,
                verbose=False,
            )
            grid.run(refine=True)
            grids[par_branch] = grid

            if (self._output_config is not None
                    and self._output_config.save_grid_results):
                path = self._output_config.grid_path(
                    f'piE_grid_{par_branch.value.lower()}'
                )
                grid.save_grid_points(path)
                logger.info('Saved grid results to %s.', path)

        if (self._output_config is not None
                and self._output_config.save_plots):
            self._plot_piE_grid_search(grids)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _do_parallax_fit(
        self,
        params,
        source_type=None,
    ) -> mmexo.MMEXOFASTFitResults:
        """
        Shared implementation invoked by ``fit_parallax()`` and
        ``run_parallax_grids()``.

        Parameters
        ----------
        params : dict
            Starting parameter dict for the parallax model.
        source_type : mmexo.SourceType, optional
            Source type for the fit.  If None, inferred from *params*:
            FINITE when ``'rho'`` or ``'t_star'`` are present, otherwise
            POINT.

        Returns
        -------
        mmexo.MMEXOFASTFitResults
            Fit results from the optimizer.
        """
        if source_type is None:
            source_type = (
                mmexo.SourceType.FINITE
                if ('rho' in params or 't_star' in params)
                else mmexo.SourceType.POINT
            )

        fitter = mmexo.fitters.SFitFitter(
            initial_model_params=params,
            datasets=self.datasets,
            **self._get_fitter_kwargs(source_type=source_type),
        )
        try:
            fitter.run()
        except Exception as e:
            logger.info('Parallax fit failed:\n{0}: {1}'.format(type(e).__name__, e))
            return None

        logger.info('Parallax fit: %s', fitter.best)
        logger.info('      sigmas: %s', list(fitter.results.sigmas))
        return mmexo.MMEXOFASTFitResults(fitter)

    def _get_parallax_seed_params(self, key: mmexo.FitKey) -> dict:
        """
        Return starting parameters for a parallax fit key.

        Priority
        --------
        1. Existing result for the same key.
        2. Another parallax branch result, sign-transformed via
           BRANCH_SIGNS.
        3. Static PSPL/FSPL with ``pi_E_N = pi_E_E = 0``.

        Parameters
        ----------
        key : mmexo.FitKey
            The parallax ``FitKey`` being started.

        Returns
        -------
        dict
            Initial parameter dict.

        Raises
        ------
        RuntimeError
            If no static point-lens result is available as a fallback.
        """
        BRANCH_SIGNS = {
            mmexo.ParallaxBranch.U0_PLUS:  (+1, +1),
            mmexo.ParallaxBranch.U0_MINUS: (-1, -1),
            mmexo.ParallaxBranch.U0_PP:    (+1, +1),
            mmexo.ParallaxBranch.U0_MM:    (-1, -1),
            mmexo.ParallaxBranch.U0_PM:    (+1, -1),
            mmexo.ParallaxBranch.U0_MP:    (-1, +1),
        }

        # 1. Exact match
        existing = self.all_fit_results.get(key)
        if existing is not None:
            return dict(existing.params)

        # 2. Sign-transform from another parallax branch
        # "s" = sign, "tgt" = target
        su0_tgt, spi_tgt = BRANCH_SIGNS[key.parallax_branch]
        for other_key, other_rec in self.all_fit_results.items():
            if (other_key.lens_type == key.lens_type
                    and other_key.source_type == key.source_type
                    and other_key.parallax_branch in BRANCH_SIGNS
                    and other_key.parallax_branch != key.parallax_branch
                    and ((other_rec.sigmas is None) or
                         (np.abs(other_rec.sigmas['pi_E_E'] / other_rec.params['pi_E_E']) < 0.5)
                    )  # don't bother if there parallax isn't well constrained.
            ):
                su0_src, spi_src = BRANCH_SIGNS[other_key.parallax_branch]
                base = dict(other_rec.params)
                if 'u_0' in base:
                    base['u_0'] *= su0_tgt / su0_src
                if 'pi_E_N' in base:
                    base['pi_E_N'] *= spi_tgt / spi_src
                logger.debug(
                    'Seeding %s from %s (sign-transformed).',
                    key.parallax_branch.value,
                    other_key.parallax_branch.value,
                )
                return base

        # 3. Static point-lens fallback
        static_key = mmexo.FitKey(
            lens_type=mmexo.LensType.POINT,
            source_type=key.source_type,
            parallax_branch=mmexo.ParallaxBranch.NONE,
            lens_orb_motion=mmexo.LensOrbMotion.NONE,
        )
        static_rec = self.all_fit_results.get(static_key)
        if static_rec is None:
            raise RuntimeError(
                'A static point-lens fit must exist before fitting '
                'parallax branches.'
            )
        base = dict(static_rec.params)
        base['pi_E_N'] = 0.0
        base['pi_E_E'] = 0.0
        logger.debug(
            'Seeding %s from static model with pi_E = 0.',
            key.parallax_branch.value,
        )
        return base

    def _needs_renormalization(self) -> bool:
        """
        Return True if any dataset's chi-squared per degree of freedom
        deviates from 1 by more than ``RENORM_THRESHOLD``.

        Uses the best available fit across all models as the reference.

        Returns
        -------
        bool
        """
        if self.RENORM_THRESHOLD is None:
            return False

        all_fits = [
            rec for rec in self.all_fit_results.values()
            if rec.chi2() is not None
        ]
        if not all_fits:
            return False

        reference_fit = min(all_fits, key=lambda r: r.chi2())
        event = reference_fit.full_result.fitter.get_event()
        event.fit_fluxes()

        for i in range(len(event.datasets)):
            chi2 = event.get_chi2_for_dataset(i)
            n_good = np.sum(event.datasets[i].good)
            if n_good == 0:
                continue
            if np.abs(chi2 / n_good - 1.0) > self.RENORM_THRESHOLD:
                return True

        return False

    def _infer_entry_point_from_initial_results(self) -> Optional[str]:
        """
        Determine the workflow entry point from user-supplied
        ``initial_results``.

        Rules
        -----
        - ``fit_type='point_lens'`` with PSPL params supplied →
          ``'fit_pspl'`` (``est_pl_params`` is skipped; supplied params
          used as seed).
        - ``fit_type='binary_lens'`` with any PSPL params supplied →
          ``search_for_anomaly`` (all point-lens stages
          skipped; the supplied model is used directly).

        Returns
        -------
        str or None
            Step name to start from, or None if no shortcut applies.
        """
        has_pspl = any(
            key.lens_type == mmexo.LensType.POINT
            and key.parallax_branch == mmexo.ParallaxBranch.NONE
            for key in self.all_fit_results.keys()
        )

        if not has_pspl:
            return None

        if self.fit_type == 'point_lens':
            return 'fit_pspl'
        if self.fit_type == 'binary_lens':
            return 'search_for_anomaly'

        return None

    def _save_restart_state(self) -> None:
        """
        Serialize current state to a restart pickle file.

        Notes
        -----
        Only writes to disk when ``self._restart_path`` is set, which
        happens automatically when ``restart_file`` is passed to
        ``__init__``.
        """
        if not getattr(self, '_restart_path', None):
            return

        restart_data = {
            'config': self._get_config(),
            'state':  self._get_state(),
        }

        # debugging:
        #step = self.completed_steps[-1]
        #logger.info(f'DEBUG save_restart_state called after: {step.stage}:{step.name}')

        with open(self._restart_path, 'wb') as f:
            pickle.dump(restart_data, f)

        logger.debug('Restart state saved to %s.', self._restart_path)

    def _get_config(self) -> dict:
        """
        Return the current configuration as a plain dict.

        Returns
        -------
        dict
        """
        return {
            key: getattr(self, key, None) for key in self.CONFIG_KEYS
        }

    def _iter_parallax_point_lens_keys(self) -> Iterable[mmexo.FitKey]:
        """
        Yield FitKeys for all parallax branches appropriate for the
        current data.

        Uses U0_PLUS and U0_MINUS for single-location data and
        U0_PP, U0_MM, U0_PM, U0_MP for multi-location data.

        Yields
        ------
        mmexo.FitKey
            Parallax model keys.
        """
        n_loc = len(
            {getattr(ds, 'ephemerides_file', None)
             for ds in self.datasets}
        )
        if n_loc <= 1:
            branches = [
                mmexo.ParallaxBranch.U0_PLUS,
                mmexo.ParallaxBranch.U0_MINUS,
            ]
        else:
            branches = [
                mmexo.ParallaxBranch.U0_PP,
                mmexo.ParallaxBranch.U0_MM,
                mmexo.ParallaxBranch.U0_PM,
                mmexo.ParallaxBranch.U0_MP,
            ]

        source_type = (
            mmexo.SourceType.FINITE
            if self.finite_source
            else mmexo.SourceType.POINT
        )
        for branch in branches:
            yield mmexo.FitKey(
                lens_type=mmexo.LensType.POINT,
                source_type=source_type,
                parallax_branch=branch,
                lens_orb_motion=mmexo.LensOrbMotion.NONE,
            )

    def _plot_piE_grid_search(self, grids: dict) -> None:
        """
        Create a two-panel piE grid search figure and save to disk.

        Parameters
        ----------
        grids : dict
            Mapping of ``mmexo.ParallaxBranch`` to
            ``ParallaxGridSearch`` objects.
        """
        #logger.info('Plotting piE grids: %s.', grids.keys())
        all_chi2 = [
            r['chi2_grid']
            for grid in grids.values()
            for r in grid.results_history
        ]
        min_chi2 = np.nanmin(all_chi2)

        fig = plt.figure(figsize=(8, 6))
        gs = gridspec.GridSpec(
            1, 3,
            figure=fig,
            width_ratios=[1, 1, 0.05],
            wspace=0.3,
        )
        axes = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1])]
        cax = fig.add_subplot(gs[2])

        branches = [
            mmexo.ParallaxBranch.U0_PLUS,
            mmexo.ParallaxBranch.U0_MINUS,
        ]
        scatter = None
        for i, (ax, par_branch) in enumerate(zip(axes, branches)):
            if par_branch not in grids:
                continue
            scatter = grids[par_branch].plot_grid_points(
                ax=ax, min_chi2=min_chi2
            )
            ax.set_xlabel(r'$\pi_{\rm E,E}$')
            ax.set_ylabel(r'$\pi_{\rm E,N}$')
            ax.set_title(par_branch.value)
            ax.invert_xaxis()
            ax.set_aspect('equal')
            ax.minorticks_on()
            if i == 1:
                ax.set_ylabel('')
                ax.tick_params(labelleft=False)

        if scatter is not None:
            fig.colorbar(
                scatter,
                cax=cax,
                label=(
                    r'$\sigma$ (min $\chi^2$ = '
                    + f'{min_chi2:.2f})'
                ),
            )

        path = self._output_config.plot_path('piE_grid')
        fig.savefig(path)
        plt.close(fig)
        logger.info('Saved piE grid plot to %s.', path)

    def _get_event_t_range(self, event, n_tE=5):
        params = event.model.parameters.parameters
        start = params['t_0'] - n_tE * params['t_E']
        stop = params['t_0'] + n_tE * params['t_E']
        return [start, stop]

    def _get_planet_t_range(self, event, n_tE=5):
        model = event.model
        if model.methods is not None and (model.n_lenses > 1):
            if model.methods is dict:
                raise NotImplementedError('Plotting for Binary Source models not implemented, yet.')
                #probably want to loop over the sources and find min/max values of hexadecapole
            else:
                hex_indices = [i for i in range(1, len(model.methods), 2) if model.methods[i] == "hexadecapole"]
                first_idx = hex_indices[0] - 1 if hex_indices else 0
                last_idx = min(hex_indices[-1] + 1, len(model.methods) - 1) if hex_indices else len(model.methods) - 1

                return [model.methods[first_idx], model.methods[last_idx]]

        elif self.intermediate_results.best_af_grid_point is not None:
            n_teff = 3
            start = self.intermediate_results.best_af_grid_point['t_0'] - n_teff * self.intermediate_results.best_af_grid_point['t_eff']
            stop = self.intermediate_results.best_af_grid_point['t_0'] + n_teff * self.intermediate_results.best_af_grid_point['t_eff']
            return [start, stop]
        else:
            return self._get_event_t_range(event, n_tE=n_tE)

    def _plot_planet_window(self):
        if self.intermediate_results.best_af_grid_point is not None:
            plt.axvline(self.intermediate_results.best_af_grid_point['t_0'] - 2450000., color='black', linestyle=':')
            plt.axvline(
                self.intermediate_results.best_af_grid_point['t_0'] - self.intermediate_results.best_af_grid_point['t_eff'] - 2450000.,
                color='black', linestyle='--')
            plt.axvline(
                self.intermediate_results.best_af_grid_point['t_0'] + self.intermediate_results.best_af_grid_point['t_eff'] - 2450000.,
                color='black', linestyle='--')

    def _plot_event(self, event, n_tE=5, suptitle=None):
        if suptitle is None:
            suptitle = '{0}'.format(event.model.parameters)

        plt.figure(figsize=(10, 6))
        plt.suptitle(suptitle)
        plt.subplot(1, 2, 1)
        event.plot_data(show_bad=True, subtract_2450000=True)
        t_range = self._get_event_t_range(event, n_tE=n_tE)
        event.plot_model(
            t_range=t_range,
            subtract_2450000=True, color='black', zorder=10)
        if event.model.n_lenses > 1:
            self._plot_planet_window()

        plt.xlim(np.array(t_range) - 2450000.)
        plt.minorticks_on()

        plt.subplot(1, 2, 2)
        event.plot_data(show_bad=True, subtract_2450000=True)
        if event.model.n_lenses > 1:
            planet_t_range = self._get_planet_t_range(event)
        else:
            planet_t_range = self._get_event_t_range(event, n_tE=0.5)

        event.plot_model(t_range=planet_t_range, color='black', subtract_2450000=True, zorder=10)
        self._plot_planet_window()
        plt.xlim(np.array(planet_t_range) - 2450000.)
        plt.minorticks_on()

        plt.tight_layout()

    def _plot_initial_2L1S_guess(self):
        #print(self.intermediate_results.est_binary_params)
        for key, params in self.intermediate_results.est_binary_params.items():
            model = MulensModel.Model(parameters=params.ulens)
            model.set_magnification_methods(params.mag_methods)
            self._plot_event(
                MulensModel.Event(model=model, datasets=self.datasets),
                suptitle=f'{key}:\n{model.parameters}')
            path = self._output_config.plot_path(f'af_{key}')
            plt.savefig(path)

    def _plot_best_fit_event(self):
        # Get the best fit
        complete_fits = [
            rec for key, rec in self.all_fit_results.items()
            if rec.full_result is not None
        ]
        best_fit = (
            min(complete_fits, key=lambda r: r.chi2()) if complete_fits else None
        )
        if best_fit.model_key.lens_type == mmexo.LensType.POINT:
            best_fit = self.select_best_point_lens_model()

        event = best_fit.full_result.fitter.get_event()

        # plot the light curve
        #event.plot(trajectory=False)
        self._plot_event(event)
        path = self._output_config.plot_path('lc')
        plt.savefig(path)
        logger.info('Saved light curve plot to %s.', path)

        # if it's a binary, also plot the trajectory with caustics
        if event.model.n_lenses > 1:
            event.plot_trajectory()
            path = self._output_config.plot_path('traj')
            plt.savefig(path)
            logger.info('Saved trajectory plot to %s.', path)

    # ------------------------------------------------------------------
    # Dataset helpers
    # ------------------------------------------------------------------

    def _create_mulensdata_objects(
        self,
        files,
        saved_datasets=None,
    ) -> list:
        """
        Create MulensData objects from file paths, reusing saved
        datasets when labels match.

        Parameters
        ----------
        files : str or list of str
            File path(s) to load.
        saved_datasets : list or None
            Previously saved datasets; any whose label matches a file
            basename are reused rather than re-loaded.

        Returns
        -------
        list of MulensModel.MulensData

        Raises
        ------
        FileNotFoundError
            If a requested file does not exist on disk.
        """
        if isinstance(files, str):
            files = [files]

        saved_by_label: dict = {}
        if saved_datasets:
            for ds in saved_datasets:
                label = ds.plot_properties.get('label')
                if label:
                    saved_by_label[label] = ds

        datasets = []
        for filename in files:
            label = os.path.basename(filename)
            if label in saved_by_label:
                datasets.append(saved_by_label[label])
            else:
                if not os.path.exists(filename):
                    raise FileNotFoundError(
                        f'Data file does not exist: {filename}'
                    )
                kwargs = mmexo.observatories.get_kwargs(filename)
                datasets.append(
                    MulensModel.MulensData(
                        file_name=filename, **kwargs
                    )
                )

        return datasets

    def _validate_dataset_labels(self) -> None:
        """
        Validate that all user-provided datasets have labels set.

        For datasets with a ``file_name`` but no label, sets the label
        to the file basename.

        Raises
        ------
        ValueError
            If any dataset has neither ``file_name`` nor a label.
        """
        for i, dataset in enumerate(self.datasets):
            label = dataset.plot_properties.get('label')
            if not label:
                if getattr(dataset, 'file_name', None):
                    dataset.plot_properties['label'] = (
                        os.path.basename(dataset.file_name)
                    )
                else:
                    raise ValueError(
                        f'Dataset at index {i} has no label in '
                        "plot_properties['label'] and was not loaded "
                        'from a file.  Set '
                        "plot_properties['label'] to a unique string "
                        'before passing to MMEXOFASTFitter.'
                    )

    def _map_label_dict_to_datasets(self, label_dict) -> dict:
        """
        Convert a ``{label: value}`` dict to a
        ``{MulensData: value}`` dict.

        Parameters
        ----------
        label_dict : dict or None
            Keys are dataset label strings; values are booleans or
            floats.  If None, all datasets default to False.

        Returns
        -------
        dict
            Keys are MulensData objects; values from *label_dict*.
        """
        if label_dict is None:
            return {ds: False for ds in self.datasets}

        result = {}
        for ds in self.datasets:
            label = ds.plot_properties.get('label')
            result[ds] = (
                label_dict.get(label, False) if label else False
            )
        return result

    def _get_fitter_kwargs(self, source_type=None) -> dict:
        """
        Bundle fitter options for passing to ``SFitFitter``.

        Parameters
        ----------
        source_type : mmexo.SourceType, optional
            When ``SourceType.POINT``, ``mag_methods`` is suppressed
            (not needed for point-source magnification).

        Returns
        -------
        dict
            Keyword arguments ready to unpack into a fitter constructor.
        """
        kwargs = {
            'coords':                      self.coords,
            'mag_methods':                 self.mag_methods,
            'limb_darkening_coeffs_u':     self.limb_darkening_coeffs_u,
            'limb_darkening_coeffs_gamma': (
                self.limb_darkening_coeffs_gamma
            ),
            'fix_source_flux':             self.fix_source_flux_map,
            'fix_blend_flux':              self.fix_blend_flux_map,
        }
        if source_type == mmexo.SourceType.POINT:
            kwargs['mag_methods'] = None
        return kwargs

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _check_dataset_labels_unique(self) -> None:
        """
        Raise if any two datasets share the same label.

        Raises
        ------
        ValueError
            If duplicate dataset labels are found, or if any label is
            None.
        """
        labels = [
            ds.plot_properties.get('label') for ds in self.datasets
        ]

        if None in labels:
            raise ValueError(
                'Some datasets do not have labels set in '
                "plot_properties['label'].  All datasets must have "
                'unique labels.'
            )

        duplicates = [
            label
            for label in set(labels)
            if labels.count(label) > 1
        ]
        if duplicates:
            raise ValueError(
                f'Duplicate dataset labels found: {duplicates}.  '
                'All datasets must have unique labels in '
                "plot_properties['label']."
            )

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def make_ulens_table(
        self,
        table_type: Optional[str] = 'ascii',
        models=None,
    ) -> str:
        """
        Return a formatted table summarizing microlensing fit results.

        Parameters
        ----------
        table_type : str or None
            ``'ascii'`` (default) or ``'latex'``.
        models : list or None
            - None: include all models in ``self.all_fit_results``.
            - list of str: model label strings.
            - list of ``mmexo.FitKey``: explicit selection.

        Returns
        -------
        str
            Table in the requested format.

        Raises
        ------
        ValueError
            If a requested model label or key is not found.
        NotImplementedError
            If *table_type* is not ``'ascii'`` or ``'latex'``.
        """
        def _order_df(df: pd.DataFrame) -> pd.DataFrame:
            """
            Order parameters in a human-friendly way (ulens params
            first, then fluxes).

            Parameters
            ----------
            df : pd.DataFrame
                DataFrame to order.

            Returns
            -------
            pd.DataFrame
                Ordered DataFrame.
            """
            def _get_ordered_ulens_keys(n_sources: int = 1) -> list[str]:
                """
                Return the default ordering of microlensing parameters.

                Parameters
                ----------
                n_sources : int
                    Number of sources.

                Returns
                -------
                list of str
                """
                basic_keys = [
                    't_0', 'u_0', 't_E', 'rho', 'log_rho', 't_star',
                ]
                additional_keys = [
                    'pi_E_N', 'pi_E_E', 't_0_par',
                    's', 'log_s', 'q', 'log_q', 'alpha',
                    'convergence_K', 'shear_G',
                    'ds_dt', 'dalpha_dt', 's_z', 'ds_z_dt', 't_0_kep',
                    'x_caustic_in', 'x_caustic_out',
                    't_caustic_in', 't_caustic_out',
                    'xi_period', 'xi_semimajor_axis',
                    'xi_inclination', 'xi_Omega_node',
                    'xi_argument_of_latitude_reference',
                    'xi_eccentricity', 'xi_omega_periapsis',
                    'q_source', 't_0_xi',
                ]
                if n_sources > 1:
                    ordered: list[str] = []
                    for param_head in basic_keys:
                        if param_head == 't_E':
                            ordered.append(param_head)
                        else:
                            for idx in range(n_sources):
                                ordered.append(
                                    f'{param_head}_{idx + 1}'
                                )
                else:
                    ordered = list(basic_keys)
                ordered.extend(additional_keys)
                return ['chi2', 'N_data'] + ordered

            def _get_ordered_flux_keys() -> list[str]:
                """
                Return ordered flux parameter names for all datasets.

                Returns
                -------
                list of str
                """
                flux_keys: list[str] = []
                for i, dataset in enumerate(self.datasets):
                    if 'label' in dataset.plot_properties:
                        obs, band = (
                            mmexo.observatories
                            .get_telescope_band_from_filename(
                                dataset.plot_properties['label']
                            )
                        )
                    else:
                        obs, band = i, None
                    flux_keys.append(f'{band}_S_{obs}')
                    flux_keys.append(f'{band}_B_{obs}')
                return flux_keys

            desired_order = (
                _get_ordered_ulens_keys() + _get_ordered_flux_keys()
            )
            order_map = {
                name: idx for idx, name in enumerate(desired_order)
            }
            df['sort_key'] = df['parameter_names'].map(order_map)
            df['orig_pos'] = range(len(df))
            df['sort_key'] = df['sort_key'].fillna(len(desired_order))
            df = (
                df.sort_values(['sort_key', 'orig_pos'])
                .reset_index()
                .drop(columns=['index', 'sort_key', 'orig_pos'])
            )
            return df

        if table_type is None:
            table_type = 'ascii'

        pm_symbol = r'$\pm$' if table_type == 'latex' else '+/-'

        # Resolve models to (label, FitRecord) pairs
        pairs: list[tuple[str, mmexo.FitRecord]] = []
        if models is None:
            for key, record in self.all_fit_results.items():
                label = mmexo.fit_types.model_key_to_label(key)
                pairs.append((label, record))
        else:
            for m in models:
                key = (
                    m if isinstance(m, mmexo.FitKey)
                    else mmexo.fit_types.label_to_model_key(m)
                )
                record = self.all_fit_results.get(key)
                if record is None:
                    raise ValueError(
                        f'No FitRecord found for model {m!r}.'
                    )
                pairs.append(
                    (mmexo.fit_types.model_key_to_label(key), record)
                )

        results_table: Optional[pd.DataFrame] = None
        for label, record in pairs:
            new_col = record.to_dataframe()
            new_col = _format_results_column(new_col, pm_symbol)
            new_col = new_col.rename(
                columns={
                    'values':      label,
                    'sigmas':      f'sig [{label}]',
                    'sigma_minus': f'sig- [{label}]',
                    'sigma_plus':  f'sig+ [{label}]',
                }
            )
            results_table = (
                new_col
                if results_table is None
                else results_table.merge(
                    new_col, on='parameter_names', how='outer'
                )
            )

        if results_table is None:
            return ''

        results_table = _order_df(results_table)

        if table_type == 'latex':
            def _fmt_latex(name: str) -> str:
                if name == 'chi2':
                    return r'$\chi^2$'
                parts = name.split('_')
                if len(parts) == 1:
                    return f'${name}$'
                first = parts[0]
                rest = ', '.join(parts[1:])
                return f'${first}' + '_{' + rest + '}$'

            results_table['parameter_names'] = (
                results_table['parameter_names'].apply(_fmt_latex)
            )
            return results_table.to_latex(index=False)

        if table_type == 'ascii':
            with pd.option_context(
                'display.max_rows',    None,
                'display.max_columns', None,
                'display.width',       None,
            ):
                return results_table.to_string(index=False)

        raise NotImplementedError(
            f"table_type {table_type!r} is not implemented."
        )

    def initialize_exozippy(self) -> dict:
        """
        Return best-fit microlensing parameters for initializing
        EXOZIPPy fitting.

        Returns
        -------
        dict
            With keys:

            ``'fits'``
                List of ``{'parameters': dict, 'sigmas': dict}``.
            ``'errfacs'``
                Per-dataset error renormalization factors
                (``self.renorm_factors``).
            ``'mag_methods'``
                Magnification methods in MulensModel convention.

        Raises
        ------
        NotImplementedError
            If ``fit_type`` is not ``'point_lens'``.
        """
        if self.fit_type != 'point_lens':
            raise NotImplementedError(
                'initialize_exozippy is currently only implemented for '
                'point_lens fits.'
            )

        fits = []
        for key in self._iter_parallax_point_lens_keys():
            record = self.all_fit_results.get(key)
            if record is not None:
                fits.append(
                    {
                        'parameters': record.params,
                        'sigmas':     record.sigmas,
                    }
                )

        return {
            'fits':        fits,
            'errfacs':     self.renorm_factors,
            'mag_methods': self.mag_methods,
        }

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f'MMEXOFASTFitter('
            f'fit_type={self.fit_type!r}, '
            f'completed_steps={len(self.completed_steps)}, '
            f'planned_steps={len(self.planned_steps)}, '
            f'n_fits={len(self.all_fit_results)})'
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False  # Don't suppress exceptions

    def close(self) -> None:
        """
        Remove and close any logging handlers added by this fitter.

        Call this when the fitter is no longer needed to prevent handler
        accumulation when multiple fitter instances are created in one
        process.
        """
        _mod_logger = logging.getLogger(__name__)
        for handler in self._log_handlers:
            handler.close()
            _mod_logger.removeHandler(handler)
        self._log_handlers.clear()

