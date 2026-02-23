"""
High-level functions for fitting microlensing events.
"""
"""
mmexofast_fitter_arch.py

Architectural sketch for MMEXOFASTFitter with:
- Structured model keys
- Centralized fit result registry (mmexo.AllFitResults)
- mmexo.FitRecord for partial/user-supplied vs full results
"""

from typing import Dict, Any, Optional, Iterable
import pickle
import inspect

import pandas as pd
from scipy.special import erfcinv
from scipy.optimize import minimize_scalar
import numpy as np
import os.path

import MulensModel

import exozippy.mmexofast as mmexo


# ============================================================================
# MMEXOFASTFitter
# ============================================================================
def fit(files=None, fit_type=None, **kwargs):
    """
    Fit a microlensing light curve using MMEXOFAST.

    Parameters
    ----------
    files : str or list, optional
        Data file(s) to fit
    fit_type : str, optional
        Type of fit ('point lens', 'binary lens')
    **kwargs : dict
        Additional arguments passed to MMEXOFASTFitter

    Returns
    -------
    MMEXOFASTFitter
        Fitted fitter object
    """
    fitter = MMEXOFASTFitter(files=files, fit_type=fit_type, **kwargs)
    fitter.fit()
    return fitter


class MMEXOFASTFitter:
    """
    Orchestrates workflows (PSPL, parallax, binary, etc.) and uses:
    - mmexo.ModelKey to identify models
    - mmexo.AllFitResults to store/reuse results
    """

    CONFIG_KEYS = [
        'files', 'fit_type', 'finite_source', 'coords', 'mag_methods',
        'limb_darkening_coeffs_u', 'limb_darkening_coeffs_gamma',
        'renormalize_errors', 'parallax_grid', 'verbose', 'fix_blend_flux',
        'fix_source_flux', 'primary_location', 'primary_dataset'
    ]

    # Parallax grid search parameters
    PARALLAX_GRID_PARAMS_COARSE = {
        'pi_E_E_min': -1.0,
        'pi_E_E_max': 1.0,
        'pi_E_E_step': 0.15,
        'pi_E_N_min': -1.5,
        'pi_E_N_max': 1.5,
        'pi_E_N_step': 0.3
    }

    PARALLAX_GRID_PARAMS_FINE = {
        'pi_E_E_min': -0.7,
        'pi_E_E_max': 0.7,
        'pi_E_E_step': 0.025,
        'pi_E_N_min': -1.0,
        'pi_E_N_max': 1.0,
        'pi_E_N_step': 0.05
    }

    def __init__(
            self,
            files=None,
            datasets=None,
            fit_type=None,
            finite_source=False,
            coords=None,
            mag_methods=None,
            limb_darkening_coeffs_u=None,
            limb_darkening_coeffs_gamma=None,
            fix_blend_flux=None,
            fix_source_flux=None,
            renormalize_errors=False,
            parallax_grid=False,
            verbose=False,
            initial_results=None,
            output_config=None,
            restart_file=None,
    ):
        # Validate mutually exclusive parameters
        if files is not None and datasets is not None:
            raise ValueError("Cannot specify both 'files' and 'datasets'. Provide only one.")

        # Ensure output_config exists
        if output_config is None:
            output_config = mmexo.OutputConfig()  # Uses default values

        # Output
        self.verbose = verbose
        self.output = mmexo.OutputManager(output_config, verbose=self.verbose) if output_config is not None else None

        # Load restart data
        saved_config, saved_state = self._load_restart_data(restart_file)
        config = self._merge_config(saved_config, locals())
        self._set_config_attributes(config)

        # Restore state
        self._restore_state(saved_state)

        # Create or use datasets
        if files:
            self.datasets = self._create_mulensdata_objects(
                files, saved_datasets=saved_state.get('datasets')
            )
        elif datasets:
            self.datasets = datasets
            self._validate_dataset_labels()  # NEW: Validate user-provided datasets
            if saved_state.get('datasets'):
                self._merge_with_saved_datasets(saved_state['datasets'])
        elif saved_state.get('datasets'):
            # Using only restart file datasets
            self.datasets = saved_state['datasets']
        else:
            raise ValueError("Must provide files, datasets, or restart_file")

        # Verify dataset labels are unique
        self._check_dataset_labels_unique()

        # Recalculate n_loc based on current datasets
        old_n_loc = saved_state.get('n_loc')
        self.n_loc = self._count_loc()
        self._location_groups = None

        # If datasets were updated, old fit results need to be refit
        self._datasets_changed = False
        if (files or datasets) and saved_state.get('all_fit_results'):
            self._datasets_changed = True

            # If n_loc changed, also remove parallax fits (wrong branches)
            #if old_n_loc is not None and old_n_loc != self.n_loc:
            #    self._remove_parallax_fits()

        # Map flux fixing options using label mapping
        self.fix_blend_flux_map = self._map_label_dict_to_datasets(self.fix_blend_flux)
        self.fix_source_flux_map = self._map_label_dict_to_datasets(self.fix_source_flux)

        self.residuals = None

        if self.parallax_grid:
            if not (self.output.config.save_plots or self.output.config.save_grid_results):
                raise ValueError(
                    "parallax_grid is enabled but neither save_plots nor save_grid_results "
                    "is set in output config. At least one must be enabled to use parallax_grid."
                )

        # Load initial results if provided
        if initial_results is not None:
            self._load_initial_results(initial_results)

        self._log_workflow_state()

    # ---------------------------------------------------------------------
    # restart helpers:
    # ---------------------------------------------------------------------
    def _get_config(self) -> dict:
        """
        Automatically extract config from attributes.

        Returns
        -------
        dict
            Configuration dictionary with all CONFIG_KEYS
        """
        return {key: getattr(self, key, None) for key in self.CONFIG_KEYS}

    def _merge_config(self, saved_config, provided_params):
        """
        Merge saved config with provided params (provided wins).

        Parameters
        ----------
        saved_config : dict
            Configuration from restart file
        provided_params : dict
            Parameters provided to __init__

        Returns
        -------
        dict
            Merged configuration
        """
        merged = {}
        for key in self.CONFIG_KEYS:
            if key in provided_params and provided_params[key] is not None:
                merged[key] = provided_params[key]
            elif key in saved_config:
                merged[key] = saved_config[key]
            else:
                merged[key] = None
        return merged

    def _set_config_attributes(self, config):
        """
        Set all config attributes from config dict.

        Parameters
        ----------
        config : dict
            Configuration dictionary
        """
        for key in self.CONFIG_KEYS:
            setattr(self, key, config[key])

    def _get_fitter_kwargs(self) -> dict:
        """
        Bundle fitter options for passing to SFitFitter.

        Returns
        -------
        dict
            Fitter configuration options
        """
        return {
            'coords': self.coords,
            'mag_methods': self.mag_methods,
            'limb_darkening_coeffs_u': self.limb_darkening_coeffs_u,
            'limb_darkening_coeffs_gamma': self.limb_darkening_coeffs_gamma,
            'fix_source_flux': self.fix_source_flux_map,
            'fix_blend_flux': self.fix_blend_flux_map
        }

    def _get_state(self) -> dict:
        """
        Get all computed state (fit results).

        Returns
        -------
        dict
            State dictionary for pickling
        """
        return {
            'all_fit_results': self.all_fit_results,
            'best_ef_grid_point': self.best_ef_grid_point,
            'best_af_grid_point': self.best_af_grid_point,
            'anomaly_lc_params': self.anomaly_lc_params,
            'n_loc': self.n_loc,
            'datasets': self.datasets,
            'renorm_factors': self.renorm_factors,
        }

    def _load_restart_data(self, restart_file):
        """
        Load config and state from restart file.

        Parameters
        ----------
        restart_file : str or None
            Path to restart pickle file

        Returns
        -------
        tuple
            (config_dict, state_dict)
        """
        if restart_file is None:
            return {}, {}

        self._log(f"Loading restart data from: {restart_file}")

        with open(restart_file, 'rb') as f:
            data = pickle.load(f)

        config = data.get('config', {})
        state = data.get('state', {})

        # Log what was loaded
        n_datasets = len(state.get('datasets', []))
        n_fits = len(state.get('all_fit_results', mmexo.AllFitResults()))
        n_renorm = len(state.get('renorm_factors', {}))

        self._log(f"  Loaded {n_datasets} datasets")
        self._log(f"  Loaded {n_fits} fit results")
        self._log(f"  Loaded {n_renorm} renormalization factors")

        if n_renorm > 0:
            self._log(f"  Renormalized datasets: {list(state.get('renorm_factors', {}).keys())}")

        return config, state

    def _restore_state(self, saved_state):
        """
        Restore computed state from saved data.

        Parameters
        ----------
        saved_state : dict
            State dictionary from restart file
        """
        self.all_fit_results = saved_state.get('all_fit_results', mmexo.AllFitResults())
        self.best_ef_grid_point = saved_state.get('best_ef_grid_point')
        self.best_af_grid_point = saved_state.get('best_af_grid_point')
        self.anomaly_lc_params = saved_state.get('anomaly_lc_params')
        self.renorm_factors = saved_state.get('renorm_factors', {})

    def _load_initial_results(self, initial_results: Dict[str, Dict[str, Any]]) -> None:
        """
        Load user-supplied initial results into mmexo.AllFitResults.

        Expected format for each entry:
        {
            "params": {...},              # required
            "sigmas": {...},              # optional
            "renorm_factors": {...},      # optional
            "fixed": bool,                # optional
        }

        Parameters
        ----------
        initial_results : dict
            Dictionary mapping model labels to result dictionaries
        """
        for label, payload in initial_results.items():
            key = mmexo.fit_types.label_to_model_key(label)
            record = mmexo.FitRecord(
                model_key=key,
                params=payload["params"],
                sigmas=payload.get("sigmas"),
                renorm_factors=payload.get("renorm_factors"),
                full_result=None,
                fixed=payload.get("fixed", False),
                is_complete=False,
            )
            self.all_fit_results.set(record)

   # def _remove_parallax_fits(self):
   #     """
   #     Remove parallax fits from all_fit_results.
   #
   #     Called when n_loc changes, since parallax branches depend on n_loc.
   #     Keeps static point lens models (PSPL/FSPL) which are n_loc-independent.
   #     """
   #     keys_to_remove = []
   #
   #     for key in self.all_fit_results:
   #         # Remove if parallax branch is not NONE
   #         if key.parallax_branch != mmexo.ParallaxBranch.NONE:
   #             keys_to_remove.append(key)
   #
   #     for key in keys_to_remove:
   #         del self.all_fit_results[key]

    def _infer_workflow_state(self):
        """
        Infer the current workflow state from available data.

        Returns
        -------
        dict
            Dictionary containing workflow state information:
            - n_loc: Number of observing locations
            - primary_location: Inferred primary location name
            - renorm_by_location: Dict mapping location to renormalized/not_renormalized dataset labels
            - has_static_fits: Whether static point lens fits exist
            - has_parallax_fits: Whether parallax fits exist
            - locations_in_static_fits: Which locations were used in static fits
            - locations_in_parallax_fits: List of locations used in parallax fits
            - complete_fit_labels: List of completed fit model labels
            - incomplete_fit_labels: List of incomplete fit model labels
        """
        workflow_state = {}

        # Basic info
        workflow_state['n_loc'] = self.n_loc

        # Infer primary location (priority order)
        primary_location = None

        # 1. Check if already set from previous multi-location run
        if hasattr(self, '_primary_location') and self._primary_location is not None:
            primary_location = self._primary_location

        # 2. Check config parameter
        elif self.primary_location is not None:
            primary_location = self.primary_location

        # 3. Check primary_dataset config parameter
        elif self.primary_dataset is not None:
            # Find location for this dataset
            for loc, datasets in self.location_groups.items():
                labels = [ds.plot_properties['label'] for ds in datasets]
                if self.primary_dataset in labels:
                    primary_location = loc
                    break

        # 4. Infer from renormalized datasets
        elif len(self.renorm_factors) > 0:
            # Find which location has renormalized datasets
            for loc, datasets in self.location_groups.items():
                labels = [ds.plot_properties['label'] for ds in datasets]
                if any(label in self.renorm_factors for label in labels):
                    primary_location = loc
                    break

        # 5. Infer from static fits
        elif self.all_fit_results.select_best_static_pspl() is not None:
            best_static = self._select_preferred_static_point_lens_model()
            static_key = None
            for key, record in self.all_fit_results.items():
                if record == best_static:
                    static_key = key
                    break

            if static_key and static_key.locations_used:
                # Parse locations_used (could be 'ground', 'All', 'ground+Spitzer', etc.)
                if static_key.locations_used != 'All':
                    # Take first location mentioned
                    primary_location = static_key.locations_used.split('+')[0]

        # 6. Fall back to longest coverage
        if primary_location is None and self.n_loc > 1:
            primary_datasets = self._select_primary_location_by_coverage()
            for loc, datasets in self.location_groups.items():
                if set(datasets) == set(primary_datasets):
                    primary_location = loc
                    break

        workflow_state['primary_location'] = primary_location

        # Build renorm_by_location
        renorm_by_location = {}
        for loc, datasets in self.location_groups.items():
            labels = [ds.plot_properties['label'] for ds in datasets]
            renormalized = [label for label in labels if label in self.renorm_factors]
            not_renormalized = [label for label in labels if label not in self.renorm_factors]

            renorm_by_location[loc] = {
                'renormalized': renormalized,
                'not_renormalized': not_renormalized
            }

        workflow_state['renorm_by_location'] = renorm_by_location

        # Check for fits
        has_static = False
        has_parallax = False
        locations_in_static = set()
        locations_in_parallax = set()

        for key, record in self.all_fit_results.items():
            if key.parallax_branch == mmexo.ParallaxBranch.NONE:
                has_static = True
                if key.locations_used:
                    if key.locations_used == 'All':
                        locations_in_static.add('All')
                    else:
                        # Parse 'ground+Spitzer' format
                        for loc in key.locations_used.split('+'):
                            locations_in_static.add(loc)
            else:
                has_parallax = True
                if key.locations_used:
                    if key.locations_used == 'All':
                        locations_in_parallax.add('All')
                    else:
                        for loc in key.locations_used.split('+'):
                            locations_in_parallax.add(loc)

        workflow_state['has_static_fits'] = has_static
        workflow_state['has_parallax_fits'] = has_parallax

        # Convert to single string or None for static fits
        if len(locations_in_static) == 0:
            workflow_state['locations_in_static_fits'] = None
        elif 'All' in locations_in_static:
            workflow_state['locations_in_static_fits'] = 'All'
        else:
            workflow_state['locations_in_static_fits'] = '+'.join(sorted(locations_in_static))

        # Keep as list for parallax fits
        if 'All' in locations_in_parallax:
            workflow_state['locations_in_parallax_fits'] = ['All']
        else:
            workflow_state['locations_in_parallax_fits'] = sorted(list(locations_in_parallax))

        # Get complete and incomplete fit labels
        complete_fit_labels = []
        incomplete_fit_labels = []

        for key, record in self.all_fit_results.items():
            label = mmexo.fit_types.model_key_to_label(key)
            if record.is_complete:
                complete_fit_labels.append(label)
            else:
                incomplete_fit_labels.append(label)

        workflow_state['complete_fit_labels'] = complete_fit_labels
        workflow_state['incomplete_fit_labels'] = incomplete_fit_labels

        return workflow_state

    def _log_workflow_state(self, workflow_state=None):
        """
        Log the current workflow state for debugging.

        Parameters
        ----------
        workflow_state : dict or None, optional
            Workflow state dict from _infer_workflow_state().
            If None, calls _infer_workflow_state() to get current state.
        """
        if workflow_state is None:
            workflow_state = self._infer_workflow_state()

        self._log("\n" + "=" * 60)
        self._log("WORKFLOW STATE")
        self._log("=" * 60)

        self._log(f"Number of locations: {workflow_state['n_loc']}")
        self._log(f"Primary location: {workflow_state['primary_location']}")

        self._log("\nDatasets by location:")
        for loc, info in workflow_state['renorm_by_location'].items():
            n_renorm = len(info['renormalized'])
            n_not_renorm = len(info['not_renormalized'])
            self._log(f"  {loc}: {n_renorm} renormalized, {n_not_renorm} not renormalized")
            if n_renorm > 0:
                for label in info['renormalized']:
                    self._log(f"    ✓ {label}")
            if n_not_renorm > 0:
                for label in info['not_renormalized']:
                    self._log(f"    ✗ {label}")

        self._log("\nFit status:")
        self._log(f"  Has static fits: {workflow_state['has_static_fits']}")
        if workflow_state['has_static_fits']:
            self._log(f"    Locations: {workflow_state['locations_in_static_fits']}")

        self._log(f"  Has parallax fits: {workflow_state['has_parallax_fits']}")
        if workflow_state['has_parallax_fits']:
            self._log(f"    Locations: {', '.join(workflow_state['locations_in_parallax_fits'])}")

        n_complete = len(workflow_state['complete_fit_labels'])
        n_incomplete = len(workflow_state['incomplete_fit_labels'])
        self._log(f"\n  Complete fits: {n_complete}")
        if n_complete > 0:
            for label in workflow_state['complete_fit_labels']:
                self._log(f"    {label}")

        if n_incomplete > 0:
            self._log(f"  Incomplete fits: {n_incomplete}")
            for label in workflow_state['incomplete_fit_labels']:
                self._log(f"    {label}")

        self._log("=" * 60 + "\n")

    # ---------------------------------------------------------------------
    # Working with datasets:
    # ---------------------------------------------------------------------
    def _create_mulensdata_objects(self, files, saved_datasets=None):
        """
        Create MulensData objects, reusing saved datasets when labels match.

        Parameters
        ----------
        files : str or list
            File paths to load
        saved_datasets : list or None
            Previously saved datasets to reuse if labels match

        Returns
        -------
        list
            List of MulensData objects
        """
        if isinstance(files, str):
            files = [files]

        # Build mapping of saved datasets by label
        saved_by_label = {}
        if saved_datasets:
            for dataset in saved_datasets:
                label = dataset.plot_properties.get('label')
                if label:
                    saved_by_label[label] = dataset

        datasets = []

        for filename in files:
            # Extract label from filename (basename)
            label = os.path.basename(filename)

            # Check if we have a saved version with this label
            if label in saved_by_label:
                data = saved_by_label[label]
            else:
                # Load fresh from file
                if not os.path.exists(filename):
                    raise FileNotFoundError(f"Data file {filename} does not exist")

                kwargs = mmexo.observatories.get_kwargs(filename)
                data = MulensModel.MulensData(file_name=filename, **kwargs)
                # Label is already set by get_kwargs()

            datasets.append(data)

        return datasets

    def _validate_dataset_labels(self):
        """
        Validate that all user-provided datasets have labels set.

        For datasets with file_name but no label, sets label to basename.
        For datasets without file_name or label, raises an error.

        Raises
        ------
        ValueError
            If any dataset lacks both file_name and label
        """
        for i, dataset in enumerate(self.datasets):
            label = dataset.plot_properties.get('label')

            if not label:
                # Try to get from file_name
                if hasattr(dataset, 'file_name') and dataset.file_name:
                    label = os.path.basename(dataset.file_name)
                    dataset.plot_properties['label'] = label
                else:
                    raise ValueError(
                        f"Dataset at index {i} does not have a label set in "
                        f"plot_properties['label'] and was not created from a file. "
                        f"Please set dataset.plot_properties['label'] to a unique "
                        f"identifier before passing to MMEXOFASTFitter."
                    )

    def _map_label_dict_to_datasets(self, label_dict) -> dict:
        """
        Map a dict[label: value] to dict[dataset: value].

        Parameters
        ----------
        label_dict : dict or None
            Keys are dataset labels, values are bool or other values

        Returns
        -------
        dict
            Keys are MulensData objects, values from label_dict
        """
        if label_dict is None:
            # Default: False for all datasets
            return {dataset: False for dataset in self.datasets}

        result = {}
        for dataset in self.datasets:
            # Get label from dataset
            label = dataset.plot_properties.get('label')

            if label and label in label_dict:
                result[dataset] = label_dict[label]
            else:
                # Default if not specified
                result[dataset] = False

        return result

    def _check_dataset_labels_unique(self):
        """
        Verify that all dataset labels are unique.

        Raises
        ------
        ValueError
            If duplicate labels are found
        """
        labels = [ds.plot_properties.get('label') for ds in self.datasets]

        # Check for None labels
        if None in labels:
            raise ValueError(
                "Some datasets do not have labels set in plot_properties['label']. "
                "All datasets must have unique labels."
            )

        # Check for duplicates
        duplicates = [label for label in set(labels) if labels.count(label) > 1]
        if duplicates:
            raise ValueError(
                f"Duplicate dataset labels found: {duplicates}. "
                "All datasets must have unique labels in plot_properties['label']."
            )

    def _merge_with_saved_datasets(self, saved_datasets):
        """
        Replace current datasets with saved versions if labels match.

        This ensures renormalized datasets from restart_file are used instead
        of freshly loaded versions.

        Parameters
        ----------
        saved_datasets : list
            List of MulensData objects from restart file
        """
        # Build mapping: label -> saved dataset
        saved_by_label = {}
        for dataset in saved_datasets:
            label = dataset.plot_properties.get('label')
            if label:
                saved_by_label[label] = dataset

        # Replace matching datasets
        n_replaced = 0
        for i, dataset in enumerate(self.datasets):
            label = dataset.plot_properties.get('label')
            if label and label in saved_by_label:
                self.datasets[i] = saved_by_label[label]
                n_replaced += 1
                self._log(f"  Replaced dataset '{label}' with saved version")

        if n_replaced == 0:
            self._log("  No datasets replaced (no label matches)")
        else:
            self._log(f"  Replaced {n_replaced} dataset(s) with saved versions")

    # Location grouping
    def _count_loc(self):
        """
        Determine how many locations an event was observed from.

        Returns
        -------
        int
            Number of distinct observing locations
        """
        if len(self.datasets) == 1:
            return 1

        else:
            locs = []
            for dataset in self.datasets:
                if dataset.ephemerides_file not in locs:
                    locs.append(dataset.ephemerides_file)

            return len(locs)

    @property
    def location_groups(self):
        """
        Dictionary mapping location names to lists of datasets.

        Cached after first access. Keys are location names like 'ground',
        'Spitzer', or ephemerides file paths for unregistered observatories.

        Returns
        -------
        dict
            Keys are location names (str), values are lists of MulensData objects
        """
        if not hasattr(self, '_location_groups') or self._location_groups is None:
            self._location_groups = self._group_datasets_by_location()
        return self._location_groups

    def _group_datasets_by_location(self):
        """
        Group datasets by observing location.

        Returns
        -------
        dict
            Keys are location names ('ground', observatory names, or ephemerides paths).
            Values are lists of MulensData objects from that location.
        """
        groups = {}

        for dataset in self.datasets:
            ephem_file = getattr(dataset, 'ephemerides_file', None)

            if ephem_file is None:
                # Ground-based observation
                location = 'ground'
            elif ephem_file in mmexo.observatories.EPHEMERIDES_TO_OBSERVATORY:
                # Registered space observatory
                location = mmexo.observatories.EPHEMERIDES_TO_OBSERVATORY[ephem_file]
            else:
                # Unknown/custom ephemerides file
                location = ephem_file

            if location not in groups:
                groups[location] = []
            groups[location].append(dataset)

        return groups

    def _count_locations_used(self, locations_used):
        """
        Count number of locations represented in locations_used string.

        Parameters
        ----------
        locations_used : str or None
            Location identifier from FitKey

        Returns
        -------
        int
            Number of locations (higher is more complete)
        """
        if locations_used == 'All':
            return self.n_loc  # All available locations
        elif locations_used is None:
            return 0  # Lowest priority (single location, n_loc=1)
        else:
            # Count locations in string like 'ground+Spitzer'
            return locations_used.count('+') + 1

    def _get_location_group_by_name(self, location_name):
        """
        Get datasets for a specific location by name.

        Parameters
        ----------
        location_name : str
            Location name ('ground', observatory name, or ephemerides path)

        Returns
        -------
        list
            Datasets from that location

        Raises
        ------
        ValueError
            If location name not found
        """
        groups = self._group_datasets_by_location()
        if location_name not in groups:
            available = list(groups.keys())
            raise ValueError(
                f"Location '{location_name}' not found. Available locations: {available}"
            )
        return groups[location_name]

    def _get_location_group_for_dataset(self, label):
        """
        Get the location group containing a specific dataset.

        Parameters
        ----------
        label : str
            Label of the dataset

        Returns
        -------
        list
            All datasets from the same location

        Raises
        ------
        ValueError
            If label not found
        """
        # Find the dataset with this label
        target_dataset = None
        for dataset in self.datasets:
            if dataset.plot_properties.get('label') == label:
                target_dataset = dataset
                break

        if target_dataset is None:
            raise ValueError(f"Dataset with label '{label}' not found")

        # Find which group it belongs to
        groups = self._group_datasets_by_location()
        for location, datasets in groups.items():
            if target_dataset in datasets:
                return datasets

        # Should never reach here
        raise ValueError(f"Dataset not found in any location group")

    def _get_location_time_coverage(self, location):
        """
        Calculate time coverage for a location.

        Parameters
        ----------
        location : str
            Location name

        Returns
        -------
        float
            Time span (max - min) for all datasets from this location
        """
        datasets = self.location_groups[location]
        all_times = np.concatenate([ds.time for ds in datasets])
        coverage = np.max(all_times) - np.min(all_times)
        return coverage

    def _select_primary_location_by_coverage(self):
        """
        Automatically select primary location based on time coverage.

        Returns
        -------
        list
            Datasets from the location with longest time coverage
        """
        groups = self._group_datasets_by_location()

        best_location = None
        max_coverage = 0.0

        for location in groups.keys():
            coverage = self._get_location_time_coverage(location)

            if coverage > max_coverage:
                max_coverage = coverage
                best_location = location

        return groups[best_location]
    def _get_location_for_datasets(self, datasets):
        """
        Determine which location(s) a set of datasets belongs to.

        Parameters
        ----------
        datasets : list
            List of MulensData objects

        Returns
        -------
        str or None
            Location name ('ground', 'Spitzer', etc.), 'All' if multiple
            locations when n_loc > 1, or None if single location (n_loc == 1)
        """
        dataset_set = set(datasets)

        # Check if using all datasets
        if dataset_set == set(self.datasets):
            return 'All' if self.n_loc > 1 else None

        # Find which location(s) match
        matching_locations = []
        for loc, loc_datasets in self.location_groups.items():
            if dataset_set == set(loc_datasets):
                matching_locations.append(loc)

        if len(matching_locations) == 1:
            return matching_locations[0]
        elif len(matching_locations) > 1:
            # Datasets span multiple locations
            return 'All' if self.n_loc > 1 else None
        else:
            # Partial subset - best guess
            return 'All' if self.n_loc > 1 else None

    # ---------------------------------------------------------------------
    # Public orchestration methods:
    # ---------------------------------------------------------------------
    def fit(self):
        """
        Perform the fit according to settings.
        """
        if self.fit_type is None:
            raise ValueError(
                'You must set the fit_type when initializing the ' +
                'MMEXOFASTFitter(): fit_type=("point lens", "binary lens")')

        if self.fit_type == 'point lens':
            # Use unified workflow for all cases (single or multi-location)
            self.fit_point_lens()

        elif self.fit_type == 'binary lens':
            # Binary lens workflow
            self.fit_binary_lens()

        self._output_latex_table()

    def fit_point_lens(self):
        """
        Run the unified point-lens workflow for single or multi-location data.

        Workflow:
        - Fit primary location static models (if needed)
        - Process new locations with incremental parallax fitting + renormalization
        - Renormalize new datasets added to existing locations
        - Comprehensive parallax fitting (all locations)
        - Optional detailed parallax grid search
        """
        # Infer current state
        state = self._infer_workflow_state()
        self._log_workflow_state(state)

        primary_loc = state['primary_location']
        self._primary_location = primary_loc  # Store for use by other methods

        # ----------------------------------------------------------------
        # Fit primary location if needed
        # ----------------------------------------------------------------
        primary_needs_fitting = (
                not state['has_static_fits'] or
                (self.renormalize_errors and
                 len(state['renorm_by_location'][primary_loc]['not_renormalized']) > 0) or
                (state['locations_in_static_fits'] and
                 primary_loc not in state['locations_in_static_fits'])
        )

        if primary_needs_fitting:
            self._log(f"\n{'=' * 60}")
            self._log("FITTING PRIMARY LOCATION")
            self._log(f"Primary location: {primary_loc}")
            self._log(f"{'=' * 60}")

            primary_datasets = self.location_groups[primary_loc]

            # Fit static models
            self._ensure_static_point_lens(primary_datasets)
            self._ensure_static_finite_point_lens(primary_datasets)

            # Renormalize if requested
            if self.renormalize_errors:
                ref_model = self._select_preferred_static_point_lens_model().full_result.fitter.get_model()
                self._renormalize_location(ref_model, primary_datasets, primary_datasets)

            self._save_restart_state()

        # ----------------------------------------------------------------
        # Identify new vs. existing locations with new datasets
        # ----------------------------------------------------------------
        new_locations = []
        new_datasets_to_existing_locs = []

        for loc, info in state['renorm_by_location'].items():
            if len(info['not_renormalized']) > 0:
                if len(info['renormalized']) == 0:
                    # No datasets from this location renormalized yet = NEW location
                    new_locations.append(loc)
                else:
                    # Some datasets already renormalized = new datasets to EXISTING location
                    new_datasets_to_existing_locs.append(loc)

        # Sort new locations by time coverage (process longer coverage first)
        new_locations.sort(
            key=lambda loc: self._get_location_time_coverage(loc),
            reverse=True
        )

        # ----------------------------------------------------------------
        # Process new locations (if any)
        # ----------------------------------------------------------------
        if len(new_locations) > 0:
            self._log(f"\n{'=' * 60}")
            self._log(f"PROCESSING NEW LOCATIONS")
            self._log(f"New locations: {new_locations}")
            self._log(f"{'=' * 60}")

            # Get currently processed datasets (all renormalized datasets)
            current_datasets = []
            for loc, info in state['renorm_by_location'].items():
                for label in info['renormalized']:
                    # Find dataset with this label
                    for ds in self.datasets:
                        if ds.plot_properties['label'] == label:
                            current_datasets.append(ds)
                            break

            # Include all new location datasets for the grid search
            all_new_datasets = []
            for new_loc in new_locations:
                all_new_datasets.extend(self.location_groups[new_loc])

            grid_datasets = current_datasets + all_new_datasets

            # Run initial grid with ALL datasets (current + new)
            self._log("\nRunning initial parallax grid with all datasets (fast chi2 survey)")
            initial_grids = self._run_piE_grid_search(
                datasets=grid_datasets,
                grid_params=self.PARALLAX_GRID_PARAMS_FINE,
                skip_optimization=True,
                save_results=False
            )

            # Optimize best solution
            self._log("\nOptimizing best parallax solution")
            best_solution = self._get_best_from_grids(initial_grids)
            reference_fit = self._optimize_parallax_solution(best_solution[1], grid_datasets)
            reference_model = reference_fit.fitter.get_model()

            # Renormalize ALL new locations using this reference
            if self.renormalize_errors:
                self._log(f"\nRenormalizing {len(new_locations)} new location(s)")

                for new_loc in new_locations:
                    new_loc_datasets = self.location_groups[new_loc]
                    # Add new location datasets to current for fitting context
                    fit_datasets = current_datasets + new_loc_datasets

                    self._renormalize_location(reference_model, new_loc_datasets, fit_datasets)

                    # Update current_datasets to include newly renormalized
                    current_datasets = fit_datasets

            self._save_restart_state()

        # ----------------------------------------------------------------
        # Renormalize new datasets to existing locations (if any)
        # ----------------------------------------------------------------
        elif len(new_datasets_to_existing_locs) > 0:
            self._log(f"\n{'=' * 60}")
            self._log("RENORMALIZING NEW DATASETS TO EXISTING LOCATIONS")
            self._log(f"Locations: {new_datasets_to_existing_locs}")
            self._log(f"{'=' * 60}")

            if self.renormalize_errors:
                # Use preferred point lens fit as reference
                reference_fit = self._select_preferred_point_lens()
                reference_model = reference_fit.full_result.fitter.get_model()

                # Renormalize new datasets
                error_factors = self._remove_outliers_and_calc_errfacs(
                    reference_model,
                    fit_datasets=self.datasets
                )
                self._apply_error_renormalization(error_factors)

                self._save_restart_state()

        # ----------------------------------------------------------------
        # Comprehensive parallax fitting
        # ----------------------------------------------------------------
        self._log(f"\n{'=' * 60}")
        self._log("COMPREHENSIVE PARALLAX FITTING")
        self._log(f"{'=' * 60}")

        all_datasets = self.datasets

        if state['has_parallax_fits']:
            # Check if existing fits match current n_loc
            existing_branches = set(key.parallax_branch for key, _ in self.all_fit_results.items()
                                    if key.parallax_branch != mmexo.ParallaxBranch.NONE)

            if self.n_loc == 1:
                expected_branches = {mmexo.ParallaxBranch.U0_PLUS, mmexo.ParallaxBranch.U0_MINUS}
            else:
                expected_branches = {mmexo.ParallaxBranch.U0_PP, mmexo.ParallaxBranch.U0_PM,
                                     mmexo.ParallaxBranch.U0_MP, mmexo.ParallaxBranch.U0_MM}

            if existing_branches & expected_branches:
                # Have at least some appropriate fits for current n_loc - re-optimize them
                self._log(f"\nRe-optimizing existing parallax fits (n_loc={self.n_loc})")
                self._reoptimize_existing_parallax_fits(datasets=all_datasets)
            else:
                # Fits are for wrong n_loc - create new ones
                self._log(f"\nExisting parallax fits are for different n_loc, creating new fits")
                self._log(f"  Existing branches: {[b.value for b in existing_branches]}")
                self._log(f"  Expected branches: {[b.value for b in expected_branches]}")

                # Initial grid
                initial_grids = self._run_piE_grid_search(
                    datasets=all_datasets,
                    grid_params=self.PARALLAX_GRID_PARAMS_FINE,
                    skip_optimization=True,
                    save_results=False
                )

                # Optimize and store solutions
                self._extract_and_optimize_parallax_solutions(
                    initial_grids[mmexo.ParallaxBranch.U0_PLUS],
                    initial_grids[mmexo.ParallaxBranch.U0_MINUS],
                    datasets=all_datasets
                )
        else:
            # Create parallax fits from scratch
            self._log("\nCreating parallax fits")

            # Initial grid
            initial_grids = self._run_piE_grid_search(
                datasets=all_datasets,
                grid_params=self.PARALLAX_GRID_PARAMS_FINE,
                skip_optimization=True,
                save_results=False
            )

            # Optimize and store solutions
            self._extract_and_optimize_parallax_solutions(
                initial_grids[mmexo.ParallaxBranch.U0_PLUS],
                initial_grids[mmexo.ParallaxBranch.U0_MINUS],
                datasets=all_datasets
            )

        self._save_restart_state()

        # ----------------------------------------------------------------
        # Optional detailed parallax grid
        # ----------------------------------------------------------------
        if self.parallax_grid:
            self._log(f"\n{'=' * 60}")
            self._log("DETAILED PARALLAX GRID SEARCH")
            self._log(f"{'=' * 60}")

            final_grids = self._run_piE_grid_search(
                datasets=all_datasets,
                grid_params=self.PARALLAX_GRID_PARAMS_FINE,
                skip_optimization=False,
                save_results=True
            )

            # Extract and optimize all solutions from final grid
            self._extract_and_optimize_parallax_solutions(
                final_grids[mmexo.ParallaxBranch.U0_PLUS],
                final_grids[mmexo.ParallaxBranch.U0_MINUS],
                datasets=all_datasets
            )

            self._save_restart_state()

    def fit_binary_lens(self) -> None:
        """
        Run binary-lens workflow, building on point-lens pieces.

        - static PSPL
        - Anomaly Finder search
        """
        # TODO: rework this workflow similar to how fit_point_lens was reworked.
        if self._datasets_changed:
            self._refit_models()

        # Reuse the shared pieces you actually need:
        self._ensure_static_point_lens()
        self._ensure_static_finite_point_lens()
        self._save_restart_state()

        # Now do binary-specific stuff:
        self._run_af_grid_search()
        self._fit_binary_models()
        self._save_restart_state()

    # ---------------------------------------------------------------------
    # Core helper: run_fit_if_needed
    # ---------------------------------------------------------------------

    def run_fit_if_needed(
            self,
            key: mmexo.FitKey,
            fit_func,
            datasets=None,
    ) -> mmexo.FitRecord:
        """
        Ensure there is an up-to-date mmexo.FitRecord for `key`.

        Parameters
        ----------
        key : mmexo.FitKey
            Which fit to run.
        fit_func : callable
            Function that runs the fit:
            `fit_func(initial_params: Optional[dict], datasets: Optional[list]) -> mmexo.MMEXOFASTFitResults`.
        datasets : list or None, optional
            Datasets to use for fitting. If None, uses self.datasets.

        Returns
        -------
        mmexo.FitRecord
            The current record for this fit (existing or newly fitted).
        """
        if datasets is None:
            datasets = self.datasets

        record = self.all_fit_results.get(key)

        # If we have a fixed or complete result, reuse it
        if record is not None and (record.fixed or record.is_complete):
            return record

        # Use existing params as a starting point, if present
        initial_params = record.params if record is not None else None

        # Run the actual fit
        full_result = fit_func(initial_params=initial_params, datasets=datasets)

        # Derive renorm factors from current state, if any
        renorm_factors = self.renorm_factors

        new_record = mmexo.FitRecord.from_full_result(
            model_key=key,
            full_result=full_result,
            renorm_factors=renorm_factors,
            fixed=False,
        )
        self.all_fit_results.set(new_record)

        if self.verbose:
            self._save_restart_state()

        return new_record

    # ------------------------------------------------------------------
    # Shared point-lens steps
    # ------------------------------------------------------------------

    def _ensure_static_point_lens(self, datasets=None) -> None:
        """
        Make sure static PSPL exists in all_fit_results.

        Parameters
        ----------
        datasets : list or None, optional
            Datasets to use. If None, uses self.datasets.
        """
        if datasets is None:
            datasets = self.datasets

        # Determine locations_used for the key
        locations_used = self._get_location_for_datasets(datasets)

        static_pspl_key = mmexo.FitKey(
            lens_type=mmexo.LensType.POINT,
            source_type=mmexo.SourceType.POINT,
            parallax_branch=mmexo.ParallaxBranch.NONE,
            lens_orb_motion=mmexo.LensOrbMotion.NONE,
            locations_used=locations_used,
        )

        def fit_static_pspl(initial_params=None, datasets=None):
            return self._fit_initial_pspl_model(initial_params=initial_params, datasets=datasets)

        self.run_fit_if_needed(static_pspl_key, fit_static_pspl, datasets=datasets)

    def _ensure_static_finite_point_lens(self, datasets=None) -> None:
        """
        Make sure static FSPL exists, if finite_source is enabled.

        Parameters
        ----------
        datasets : list or None, optional
            Datasets to use. If None, uses self.datasets.
        """
        if not self.finite_source:
            return

        if datasets is None:
            datasets = self.datasets

        # Determine locations_used for the key
        locations_used = self._get_location_for_datasets(datasets)

        static_fspl_key = mmexo.FitKey(
            lens_type=mmexo.LensType.POINT,
            source_type=mmexo.SourceType.FINITE,
            parallax_branch=mmexo.ParallaxBranch.NONE,
            lens_orb_motion=mmexo.LensOrbMotion.NONE,
            locations_used=locations_used,
        )

        def fit_static_fspl(initial_params=None, datasets=None):
            return self._fit_static_fspl_model(initial_params=initial_params, datasets=datasets)

        self.run_fit_if_needed(static_fspl_key, fit_static_fspl, datasets=datasets)

    def _ensure_point_lens_parallax_models(self, datasets=None) -> None:
        """
        Make sure all configured point-lens parallax branches are fitted.

        Parameters
        ----------
        datasets : list or None, optional
            Datasets to use. If None, uses self.datasets.
        """
        if datasets is None:
            datasets = self.datasets

        # Determine locations_used for the key
        locations_used = self._get_location_for_datasets(datasets)

        for par_key_base in self._iter_parallax_point_lens_keys():
            # Add locations_used to the key
            par_key = mmexo.FitKey(
                lens_type=par_key_base.lens_type,
                source_type=par_key_base.source_type,
                parallax_branch=par_key_base.parallax_branch,
                lens_orb_motion=par_key_base.lens_orb_motion,
                locations_used=locations_used,
            )

            def make_fit_func(k: mmexo.FitKey):
                def fit_func(initial_params=None, datasets=None):
                    return self._fit_pl_parallax_model(k, initial_params=initial_params, datasets=datasets)

                return fit_func

            self.run_fit_if_needed(par_key, make_fit_func(par_key), datasets=datasets)

    def _renormalize_location(self, reference_model, datasets_to_renorm, fit_datasets):
        """
        Renormalize errors for specific datasets using a reference model.

        Parameters
        ----------
        reference_model : MulensModel.Model
            Reference model for error renormalization
        datasets_to_renorm : list
            Datasets to renormalize (typically from one location)
        fit_datasets : list
            All datasets to include in the event for proper flux fitting
            (provides context for renormalization)
        """
        self._log(f"Renormalizing {len(datasets_to_renorm)} dataset(s)")

        # Calculate error factors for datasets_to_renorm only
        error_factors = self._remove_outliers_and_calc_errfacs(
            reference_model,
            fit_datasets=fit_datasets
        )

        # Apply renormalization (only to datasets_to_renorm)
        self._apply_error_renormalization(error_factors, datasets=datasets_to_renorm)

    def _reoptimize_existing_parallax_fits(self, datasets):
        """
        Re-optimize existing parallax fits with updated datasets.

        Takes existing parallax fit parameters and re-optimizes them
        with the current dataset collection.

        Parameters
        ----------
        datasets : list
            Datasets to use for re-optimization
        """
        # Get all parallax fits
        parallax_fits = []
        for key, record in self.all_fit_results.items():
            if key.parallax_branch != mmexo.ParallaxBranch.NONE:
                parallax_fits.append((key, record))

        if len(parallax_fits) == 0:
            self._log("Warning: No existing parallax fits to re-optimize")
            return

        self._log(f"Re-optimizing {len(parallax_fits)} parallax fit(s)")

        for key, record in parallax_fits:
            label = mmexo.fit_types.model_key_to_label(key)
            self._log(f"  Re-optimizing {label}")

            # Use existing params as starting point
            initial_params = record.params

            # Re-optimize
            fitter = mmexo.fitters.SFitFitter(
                initial_model_params=initial_params,
                datasets=datasets,
                **self._get_fitter_kwargs()
            )
            fitter.run()

            # Update record
            full_result = mmexo.MMEXOFASTFitResults(fitter)
            new_record = mmexo.FitRecord.from_full_result(
                model_key=key,
                full_result=full_result,
                renorm_factors=self.renorm_factors,
                fixed=False,
            )
            self.all_fit_results.set(new_record)

            self._log(f"    chi2 = {fitter.chi2:.2f}")

    # ---------------------------------------------------------------------
    # Point-lens helpers:
    # ---------------------------------------------------------------------

    def _fit_initial_pspl_model(
            self,
            initial_params: Optional[Dict[str, float]] = None,
            datasets=None,
    ) -> mmexo.MMEXOFASTFitResults:
        """
        Estimate or accept starting point for PSPL, then run SFitFitter.

        EF grid is only used if `initial_params` is None and
        best_ef_grid_point is not yet available.

        Parameters
        ----------
        initial_params : dict or None, optional
            Starting parameters for fit
        datasets : list or None, optional
            Datasets to use. If None, uses self.datasets.

        Returns
        -------
        mmexo.MMEXOFASTFitResults
            Fit results
        """
        if datasets is None:
            datasets = self.datasets

        if initial_params is None:
            if self.best_ef_grid_point is None:
                self.best_ef_grid_point = self.do_ef_grid_search()
                self._log(f"Best EF grid point {self.best_ef_grid_point}")

            pspl_est_params = mmexo.estimate_params.get_PSPL_params(
                self.best_ef_grid_point,
                datasets,
            )
            self._log(f"Initial PSPL Estimate {pspl_est_params}")
        else:
            pspl_est_params = initial_params
            self._log(f"Using initial PSPL params (user/previous): {pspl_est_params}")

        fitter = mmexo.fitters.SFitFitter(
            initial_model_params=pspl_est_params, datasets=datasets, **self._get_fitter_kwargs())
        fitter.run()
        self._log(f'Initial SFit {fitter.best}')
        self._log_file_only(fitter.get_diagnostic_str())

        return mmexo.MMEXOFASTFitResults(fitter)

    def _fit_static_fspl_model(
            self,
            initial_params: Optional[Dict[str, float]] = None,
            datasets=None,
    ) -> mmexo.MMEXOFASTFitResults:
        """
        Fit a finite-source point-lens (FSPL) model.

        Parameters
        ----------
        initial_params : dict or None, optional
            Starting parameters for fit
        datasets : list or None, optional
            Datasets to use. If None, uses self.datasets.

        Returns
        -------
        mmexo.MMEXOFASTFitResults
            Fit results
        """
        if datasets is None:
            datasets = self.datasets

        if initial_params is None:
            # Seed from static PSPL record if available
            static_pspl_key = mmexo.FitKey(
                lens_type=mmexo.LensType.POINT,
                source_type=mmexo.SourceType.POINT,
                parallax_branch=mmexo.ParallaxBranch.NONE,
                lens_orb_motion=mmexo.LensOrbMotion.NONE,
            )
            pspl_record = self.all_fit_results.get(static_pspl_key)
            if pspl_record is None:
                raise RuntimeError(
                    "Static PSPL must be fitted (or provided) before FSPL."
                )
            fspl_est_params = dict(pspl_record.params)
            fspl_est_params['rho'] = 1.5 * fspl_est_params['u_0']

        else:
            fspl_est_params = initial_params

        fitter = mmexo.fitters.SFitFitter(
            initial_model_params=fspl_est_params, datasets=datasets, **self._get_fitter_kwargs())
        fitter.run()
        self._log(f'FSPL: {fitter.best}')
        self._log_file_only(fitter.get_diagnostic_str())

        return mmexo.MMEXOFASTFitResults(fitter)

    # --- parallax branch sign definitions and helpers ----------------

    BRANCH_SIGNS = {
        mmexo.ParallaxBranch.U0_PLUS: (+1, +1),
        mmexo.ParallaxBranch.U0_MINUS: (-1, -1),
        mmexo.ParallaxBranch.U0_PP: (+1, +1),
        mmexo.ParallaxBranch.U0_MM: (-1, -1),
        mmexo.ParallaxBranch.U0_PM: (+1, -1),
        mmexo.ParallaxBranch.U0_MP: (-1, +1),
    }

    def _apply_branch_signs(
            self,
            params: Dict[str, float],
            src_branch: mmexo.ParallaxBranch,
            target_branch: mmexo.ParallaxBranch,
    ) -> None:
        """
        In-place: adjust params from src_branch convention to target_branch.

        Flips signs of u_0 and/or pi_E_N as needed.

        Parameters
        ----------
        params : dict
            Parameter dictionary to modify in place
        src_branch : mmexo.ParallaxBranch
            Source parallax branch
        target_branch : mmexo.ParallaxBranch
            Target parallax branch
        """
        su0_src, spi_src = self.BRANCH_SIGNS[src_branch]
        su0_tgt, spi_tgt = self.BRANCH_SIGNS[target_branch]

        u0_factor = su0_tgt / su0_src
        piN_factor = spi_tgt / spi_src

        if "u_0" in params:
            params["u_0"] *= u0_factor
        if "pi_E_N" in params:
            params["pi_E_N"] *= piN_factor

    def _iter_parallax_point_lens_keys(self) -> Iterable[mmexo.FitKey]:
        """
        Yield mmexo.ModelKeys for all point-lens parallax models consistent with n_loc.

        Yields
        ------
        mmexo.FitKey
            Parallax model keys
        """
        if self.n_loc == 1:
            branches = [mmexo.ParallaxBranch.U0_PLUS, mmexo.ParallaxBranch.U0_MINUS]
        else:
            branches = [
                mmexo.ParallaxBranch.U0_PP,
                mmexo.ParallaxBranch.U0_MM,
                mmexo.ParallaxBranch.U0_PM,
                mmexo.ParallaxBranch.U0_MP,
            ]

        for branch in branches:
            yield mmexo.FitKey(
                lens_type=mmexo.LensType.POINT,
                source_type=(
                    mmexo.SourceType.FINITE if self.finite_source else mmexo.SourceType.POINT
                ),
                parallax_branch=branch,
                lens_orb_motion=mmexo.LensOrbMotion.NONE,
            )

    def _get_parallax_initial_params(
            self,
            key: mmexo.FitKey,
            initial_params: Optional[Dict[str, float]],
    ) -> Dict[str, float]:
        """
        Decide how to initialize parameters for a parallax point-lens fit.

        Priority:
        1. Use provided initial_params if not None.
        2. Seed from an existing parallax branch result, transformed via sign flips.
        3. Fallback to static point-lens params (PSPL/FSPL) for this source_type.

        Parameters
        ----------
        key : mmexo.FitKey
            Fit key for the parallax model
        initial_params : dict or None
            Provided initial parameters

        Returns
        -------
        dict
            Initial parameters for the fit
        """
        # 1. Caller-supplied initial params
        if initial_params is not None:
            self._log(
                f"Using provided initial params for parallax branch "
                f"{key.parallax_branch}: {initial_params}"
            )
            return dict(initial_params)

        # 2. Seed from any existing parallax branch
        for other_branch in self.BRANCH_SIGNS.keys():
            if other_branch == key.parallax_branch:
                continue

            other_key = mmexo.FitKey(
                lens_type=key.lens_type,
                source_type=key.source_type,
                parallax_branch=other_branch,
                lens_orb_motion=key.lens_orb_motion,
            )
            other_record = self.all_fit_results.get(other_key)
            if other_record is None:
                continue

            base = dict(other_record.params)
            self._apply_branch_signs(
                base,
                src_branch=other_branch,
                target_branch=key.parallax_branch,
            )
            self._log(
                f"Seeding parallax branch {key.parallax_branch.value} from "
                f"existing branch {other_branch.value} with transformed params: {base}"
            )
            return base

        # 3. Fallback: static point-lens
        static_key = mmexo.FitKey(
            lens_type=mmexo.LensType.POINT,
            source_type=key.source_type,
            parallax_branch=mmexo.ParallaxBranch.NONE,
            lens_orb_motion=mmexo.LensOrbMotion.NONE,
        )
        static_record = self.all_fit_results.get(static_key)
        if static_record is None:
            raise RuntimeError(
                "Static point-lens model must be available before parallax fits."
            )

        base = dict(static_record.params)
        base['pi_E_N'] = 0.
        base['pi_E_E'] = 0.
        self._log(
            f"Seeding parallax branch {key.parallax_branch.value} from "
            f"static model (source_type={key.source_type.value}): {base}"
        )
        return base

    def _fit_pl_parallax_model(
            self,
            key: mmexo.FitKey,
            initial_params: Optional[Dict[str, float]] = None,
            datasets=None,
    ) -> mmexo.MMEXOFASTFitResults:
        """
        Fit a point-lens parallax model for the given parallax branch.

        Parameters
        ----------
        key : mmexo.FitKey
            Fit key identifying the parallax model
        initial_params : dict or None, optional
            Starting parameters for fit
        datasets : list or None, optional
            Datasets to use. If None, uses self.datasets.

        Returns
        -------
        mmexo.MMEXOFASTFitResults
            Fit results
        """
        if datasets is None:
            datasets = self.datasets

        par_est_params = self._get_parallax_initial_params(key, initial_params)

        fitter = mmexo.fitters.SFitFitter(
            initial_model_params=par_est_params, datasets=datasets, **self._get_fitter_kwargs())
        fitter.run()
        self._log(f'{mmexo.fit_types.model_key_to_label(key)}: {fitter.best}')
        self._log_file_only(fitter.get_diagnostic_str())

        return mmexo.MMEXOFASTFitResults(fitter)

    def _run_piE_grid_search(self, datasets=None, grid_params=None, skip_optimization=False, save_results=True):
        """
        Run parallax grid search over pi_E_E and pi_E_N for U0_PLUS and U0_MINUS branches.

        For each branch, performs a grid search and optionally saves results and/or plots.
        Results are saved to files named: {file_head}_piE_grid_{branch}.txt
        Plot is saved as: {file_head}_piE_grid.png

        Parameters
        ----------
        datasets : list or None, optional
            Datasets to use. If None, uses self.datasets.
        grid_params : dict or None, optional
            Grid parameters (min, max, step for pi_E_E and pi_E_N).
            If None, uses self.PARALLAX_GRID_PARAMS_FINE.
        skip_optimization : bool, optional
            If True, calculate chi2 without optimization (faster, for coarse grids).
            Default is False.
        save_results : bool, optional
            If True, save grid results to file (when save_grid_results is configured).
            Default is True.

        Returns
        -------
        dict
            Dictionary mapping ParallaxBranch to ParallaxGridSearch objects
        """
        if datasets is None:
            datasets = self.datasets

        if grid_params is None:
            grid_params = self.PARALLAX_GRID_PARAMS_FINE

        # Get reference model
        reference_fit = self._select_preferred_static_point_lens_model()
        reference_model = reference_fit.full_result.fitter.get_model()
        static_params = reference_model.parameters.parameters

        # Iterate over parallax branches
        branches = [mmexo.ParallaxBranch.U0_PLUS, mmexo.ParallaxBranch.U0_MINUS]

        grids = {}  # Store grid objects for plotting

        for branch in branches:
            self._log(f"Running piE grid search for {branch.value}")

            # Create and run grid search
            grid = mmexo.ParallaxGridSearch(
                datasets=datasets,
                static_params=static_params,
                grid_params=grid_params,
                fitter_kwargs=self._get_fitter_kwargs(),
                skip_optimization=skip_optimization,
                verbose=self.verbose,
            )
            grid.run()

            grids[branch] = grid

            # Save results if configured AND requested
            if self.output.config.save_grid_results and save_results:
                filename = f"{self.output.config.file_head}_piE_grid_{branch.value.lower()}.txt"
                filepath = self.output.config.base_dir / filename
                grid.save_results(filepath, parallax_branch=branch.value)
                self._log(f"Saved grid results to {filepath}")

        # Create plot if configured
        if self.output.config.save_plots and save_results:
            self._plot_piE_grid_search(grids)

        return grids

    def _plot_piE_grid_search(self, grids):
        """
        Create 2-panel plot of piE grid search results.

        Parameters
        ----------
        grids : dict
            Dictionary mapping ParallaxBranch to ParallaxGridSearch objects
        """
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        # Find global minimum chi2 for consistent coloring
        all_chi2 = []
        for grid in grids.values():
            all_chi2.extend([r['chi2'] for r in grid.results])

        min_chi2 = min(all_chi2)

        # Create figure with gridspec layout: 2 plots + colorbar
        fig = plt.figure(figsize=(8, 6))
        gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 0.05], wspace=0.3)

        axes = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1])]
        cax = fig.add_subplot(gs[2])

        branches = [mmexo.ParallaxBranch.U0_PLUS, mmexo.ParallaxBranch.U0_MINUS]

        for i, (ax, branch) in enumerate(zip(axes, branches)):
            grid = grids[branch]

            # Plot grid points
            scatter = grid.plot_grid_points(ax=ax, min_chi2=min_chi2)

            # Formatting
            ax.set_xlabel(r'$\pi_{\rm E,E}$')
            ax.set_ylabel(r'$\pi_{\rm E,N}$')
            ax.set_title(branch.value)
            ax.invert_xaxis()
            ax.set_aspect('equal')
            ax.minorticks_on()

            # Turn off y-axis label and tick labels on right plot
            if i == 1:
                ax.set_ylabel('')
                ax.tick_params(labelleft=False)

        # Add colorbar
        fig.colorbar(scatter, cax=cax, label=r'$\sigma$ (min $\chi^2$ = ' + f'{min_chi2:.2f})')

        # Save plot
        self.output.save_plot('piE_grid', fig)

    def _get_best_from_grids(self, grids):
        """
        Get the best solution across multiple grid searches.

        Parameters
        ----------
        grids : dict
            Dictionary mapping ParallaxBranch to ParallaxGridSearch objects

        Returns
        -------
        tuple
            (chi2, params) of the best solution across all grids
        """
        best_overall = None
        best_chi2 = float('inf')

        for branch, grid in grids.items():
            minima = grid.find_local_minima()
            if len(minima) > 0:
                # Get best from this grid
                chi2, params = minima[0]  # Already sorted by chi2
                if chi2 < best_chi2:
                    best_chi2 = chi2
                    best_overall = (chi2, params)

        if best_overall is None:
            raise ValueError("No valid minima found in any grid")

        return best_overall

    # space parallax
    def _get_space_u0_sign(self, fit_result, space_ephemerides_file):
        """
        Determine the sign of u0 for the space observatory trajectory.

        Parameters
        ----------
        fit_result : FitRecord
            Fit result containing the model
        space_ephemerides_file : str
            Path to ephemerides file for the space observatory

        Returns
        -------
        str
            'P' for positive u0, 'M' for negative u0
        """
        params = fit_result.get_params_from_results()
        
        # Get model from fit_result
        model = MulensModel.Model(
            parameters=params,
            coords=self.coords,
            ephemerides_file=space_ephemerides_file
        )

        # Define function to minimize: u(t) = sqrt(x^2 + y^2)
        def u_squared(time):
            trajectory = model.get_trajectory([time])
            return trajectory.x[0] ** 2 + trajectory.y[0] ** 2

        # Find time of minimum u
        t_0 = params['t_0']
        t_E = params['t_E']

        result = minimize_scalar(
            u_squared,
            bounds=(t_0 - 2 * t_E, t_0 + 2 * t_E),
            method='bounded'
        )

        # Get y-coordinate at minimum
        t_min = result.x
        trajectory = model.get_trajectory([t_min])
        y_min = trajectory.y[0]

        # Return sign
        if y_min >= 0:
            return 'P'
        else:
            return 'M'

    def _fit_primary_location(self, primary_location=None, primary_dataset=None):
        """
        Fit primary location with static point lens models.

        Selects the primary location (automatically or by specification),
        fits static PSPL/FSPL, and renormalizes errors if configured.

        Parameters
        ----------
        primary_location : str or None, optional
            Location name to use as primary ('ground', 'Spitzer', etc.).
            If None, automatically selects location with longest time coverage.
        primary_dataset : str or None, optional
            Label of dataset to use for identifying primary location.
            Takes precedence over primary_location.

        Returns
        -------
        list
            Primary location datasets that were fit
        """
        # Select primary location datasets
        if primary_dataset is not None:
            primary_datasets = self._get_location_group_for_dataset(primary_dataset)
            # Find location name
            for loc, datasets in self.location_groups.items():
                if set(datasets) == set(primary_datasets):
                    self._primary_location = loc
                    break

            self._log(f"Using primary dataset: {primary_dataset} (location: {self._primary_location})")
        elif primary_location is not None:
            primary_datasets = self._get_location_group_by_name(primary_location)
            self._primary_location = primary_location
            self._log(f"Using primary location: {primary_location}")

        else:
            primary_datasets = self._select_primary_location_by_coverage()
            # Determine location name
            self._primary_location = None
            for loc, datasets in self.location_groups.items():
                if set(datasets) == set(primary_datasets):
                    self._primary_location = loc
                    break

            self._log(f"Auto-selected primary location: {self._primary_location}")

        # Fit static models with primary location only (skip parallax)
        self._log(f"Fitting static models with {len(primary_datasets)} primary location datasets")
        self.fit_point_lens(datasets=primary_datasets, skip_parallax=True)

        return primary_datasets

    def _add_location_and_grid_search(self, datasets, grid_params, skip_optimization=False, save_results=True):
        """
        Run parallax grid search with specified datasets and grid parameters.

        Parameters
        ----------
        datasets : list
            Datasets to use for grid search
        grid_params : dict
            Grid parameters (pi_E_E_min, pi_E_E_max, pi_E_E_step, etc.)
        skip_optimization : bool, optional
            If True, skip parameter optimization (faster). Default is False.
        save_results : bool, optional
            If True, save grid results to file. Default is True.

        Returns
        -------
        dict
            Dictionary mapping ParallaxBranch to ParallaxGridSearch objects
            Keys: mmexo.ParallaxBranch.U0_PLUS, mmexo.ParallaxBranch.U0_MINUS
        """
        locations_used = self._get_location_for_datasets(datasets)
        self._log(f"Running parallax grid search with locations: {locations_used}")

        # Run grid search and return the grids
        grids = self._run_piE_grid_search(
            datasets=datasets,
            grid_params=grid_params,
            skip_optimization=skip_optimization,
            save_results=save_results
        )

        return grids

    def _extract_and_optimize_parallax_solutions(self, u0_plus_grid, u0_minus_grid, datasets):
        """
        Extract minima from parallax grids and optimize each solution.

        For n_loc=2, extracts minima from both grids, optimizes each, determines
        u0 signs for both locations, and stores as PP/PM/MP/MM solutions.

        For n_loc>2, stores solutions as U0_PLUS or U0_MINUS based on primary
        location u0 sign only.

        Parameters
        ----------
        u0_plus_grid : ParallaxGridSearch
            Grid search results for U0_PLUS branch
        u0_minus_grid : ParallaxGridSearch
            Grid search results for U0_MINUS branch
        datasets : list
            Datasets to use for optimization (typically all available)
        """
        locations_used = self._get_location_for_datasets(datasets)

        # For n_loc=2, identify secondary location and get its ephemerides
        secondary_ephem = None
        if self.n_loc == 2:
            for loc, loc_datasets in self.location_groups.items():
                if loc != self._primary_location:
                    # Found secondary location
                    if len(loc_datasets) > 0:
                        # Get ephemerides file (could be None for ground)
                        secondary_ephem = getattr(loc_datasets[0], 'ephemerides_file', None)
                    break

            if secondary_ephem is None and self._primary_location != 'ground':
                # Secondary must be ground (no ephemerides)
                pass  # secondary_ephem stays None, which is correct for ground

        # Process both grids in a loop
        grid_configs = [
            (u0_plus_grid, mmexo.ParallaxBranch.U0_PLUS, 'U0_PLUS'),
            (u0_minus_grid, mmexo.ParallaxBranch.U0_MINUS, 'U0_MINUS')
        ]

        for grid, base_branch, grid_name in grid_configs:
            # Extract minima from this grid
            minima = grid.find_local_minima()
            self._log(f"Found {len(minima)} minima in {grid_name} grid")

            # Optimize each minimum
            for i, (chi2, params) in enumerate(minima):
                self._log(f"Optimizing {grid_name} minimum {i + 1}/{len(minima)}")

                # Optimize with full datasets
                fit_result = self._optimize_parallax_solution(params, datasets)

                # Determine branch based on n_loc
                if self.n_loc == 2:
                    # Check secondary location u0 sign
                    secondary_sign = self._get_space_u0_sign(fit_result, secondary_ephem)

                    # Map to PP/PM/MP/MM based on primary (base_branch) and secondary signs
                    if base_branch == mmexo.ParallaxBranch.U0_PLUS:
                        branch = mmexo.ParallaxBranch.U0_PP if secondary_sign == 'P' else mmexo.ParallaxBranch.U0_PM
                    else:  # U0_MINUS
                        branch = mmexo.ParallaxBranch.U0_MP if secondary_sign == 'P' else mmexo.ParallaxBranch.U0_MM
                else:
                    # n_loc > 2: just use base branch (U0_PLUS or U0_MINUS)
                    branch = base_branch

                # Create FitKey
                fit_key = mmexo.FitKey(
                    lens_type=mmexo.LensType.POINT,
                    source_type=mmexo.SourceType.FINITE if self.finite_source else mmexo.SourceType.POINT,
                    parallax_branch=branch,
                    lens_orb_motion=mmexo.LensOrbMotion.NONE,
                    locations_used=locations_used,
                )

                # Check if this key already exists (edge case warning)
                if fit_key in self.all_fit_results:
                    self._log(f"WARNING: FitKey {branch.value} already exists in all_fit_results. "
                                    f"This may indicate multiple minima with the same u0 sign combination. "
                                    f"Overwriting previous result.")

                # Create and store FitRecord
                record = mmexo.FitRecord.from_full_result(
                    model_key=fit_key,
                    full_result=fit_result,
                    renorm_factors=self.renorm_factors,
                    fixed=False,
                )
                self.all_fit_results.set(record)
                self._log(f"Stored {branch.value} solution (chi2={fit_result.chi2:.2f})")

    def _optimize_parallax_solution(self, initial_params, datasets):
        """
        Optimize a parallax solution starting from grid parameters.

        Parameters
        ----------
        initial_params : dict
            Starting parameters from grid search minimum
        datasets : list
            Datasets to use for optimization

        Returns
        -------
        MMEXOFASTFitResults
            Optimized fit results
        """
        self._log(f"Optimizing from grid point: pi_E_E={initial_params.get('pi_E_E', 'N/A'):.3f}, "
                        f"pi_E_N={initial_params.get('pi_E_N', 'N/A'):.3f}")

        # Run SFitFitter with all parameters free
        fitter = mmexo.fitters.SFitFitter(
            initial_model_params=initial_params,
            datasets=datasets,
            **self._get_fitter_kwargs()
        )
        fitter.run()

        self._log(f"Optimized: chi2={fitter.best.get('chi2'):.2f}, {fitter.best}")

        return mmexo.MMEXOFASTFitResults(fitter)

    # other parallax helpers
    def _select_preferred_static_point_lens_model(self, chi2_threshold=20):
        """
        Select the preferred static point lens model from self.all_fit_results.

        Prefers models with more complete location coverage. Among models with
        the same location coverage, prefers FSPL over PSPL if chi2 improvement
        exceeds chi2_threshold.

        Parameters
        ----------
        chi2_threshold : float, optional
            Minimum chi2 improvement required to prefer FSPL over PSPL.
            Default is 20.

        Returns
        -------
        FitRecord
            The preferred static point lens fit result

        Raises
        ------
        ValueError
            If no static point lens models exist
        """
        # Find all PSPL and FSPL fits regardless of locations_used
        pspl_fits = []
        fspl_fits = []

        for key, fit in self.all_fit_results.items():
            if (key.lens_type == mmexo.LensType.POINT and
                    key.parallax_branch == mmexo.ParallaxBranch.NONE and
                    key.lens_orb_motion == mmexo.LensOrbMotion.NONE):

                if key.source_type == mmexo.SourceType.POINT:
                    pspl_fits.append((key, fit))
                elif key.source_type == mmexo.SourceType.FINITE:
                    fspl_fits.append((key, fit))

        # Check if at least one exists
        if len(pspl_fits) == 0 and len(fspl_fits) == 0:
            raise ValueError("No static point lens models found in all_fit_results")

        # Select most complete version of each type
        def get_most_complete(fits_list):
            if len(fits_list) == 0:
                return None
            # Sort by location completeness (descending), then by chi2 (ascending)
            return max(fits_list, key=lambda x: (
                self._count_locations_used(x[0].locations_used),
                -x[1].chi2()  # Negative for ascending chi2
            ))[1]

        pspl_fit = get_most_complete(pspl_fits)
        fspl_fit = get_most_complete(fspl_fits)

        # If only one type exists, return it
        if pspl_fit is None:
            return fspl_fit
        if fspl_fit is None:
            return pspl_fit

        # Both exist - compare chi2
        pspl_chi2 = pspl_fit.chi2()
        fspl_chi2 = fspl_fit.chi2()

        # Return FSPL only if significantly better
        if fspl_chi2 < pspl_chi2 - chi2_threshold:
            return fspl_fit
        else:
            return pspl_fit

    def _select_preferred_point_lens(
            self,
            delta_chi2_threshold: float = 50.0,
    ) -> Optional[mmexo.FitRecord]:
        """
        Choose the preferred PSPL model for the binary workflow.

        Policy:
        - If any parallax models exist, pick the best parallax model.
        - Compare its chi^2 to the best static PL chi^2.
        - If (chi2_static - chi2_parallax) > delta_chi2_threshold,
            → use parallax model.
          Else
            → use static PL model.
        - If no parallax models exist, fall back to static PL.
        - If neither exists (or no chi^2), return None.

        Parameters
        ----------
        delta_chi2_threshold : float, optional
            Minimum chi2 improvement for parallax to be preferred. Default is 50.

        Returns
        -------
        FitRecord or None
            Preferred point lens model
        """
        best_static = self._select_preferred_static_point_lens_model()
        best_par = self.all_fit_results.select_best_parallax_pspl()

        chi2_static = best_static.chi2() if best_static is not None else None
        chi2_par = best_par.chi2() if best_par is not None else None

        # Case 1: no parallax available or no parallax chi^2
        if chi2_par is None:
            return best_static

        # Case 2: no static available or no static chi^2 → default to parallax
        if chi2_static is None:
            return best_par

        # Case 3: both exist; apply threshold rule
        improvement = chi2_static - chi2_par
        if improvement > delta_chi2_threshold:
            # parallax is significantly better
            return best_par
        else:
            # static is as good or better (within threshold)
            return best_static

    # ---------------------------------------------------------------------
    # Binary-lens helpers:
    # ---------------------------------------------------------------------
    def _run_af_grid_search(self):
        """
        Run Anomaly Finder grid search if not already done.
        """
        if self.best_af_grid_point is None:
            self.best_af_grid_point = self.do_af_grid_search()
            self._log(f'Best AF grid {self.best_af_grid_point}')

        if self.anomaly_lc_params is None:
            self.anomaly_lc_params = self.get_anomaly_lc_params()
            self._log(f'Anomaly Params {self.anomaly_lc_params}')

    def _fit_binary_models(self):
        """
        Fit binary lens models (currently only wide planet in GG97 limit).

        Raises
        ------
        NotImplementedError
            Binary fitting only partially implemented
        """

        def fit_wide_planet():
            wide_planet_fitter = mmexo.fitters.WidePlanetFitter(
                datasets=self.datasets, anomaly_lc_params=self.anomaly_lc_params,
            )
            wide_planet_fitter.estimate_initial_parameters()
            self._log(
                f'Initial 2L1S Wide Model {wide_planet_fitter.initial_model}' +
                f'\nmag methods {wide_planet_fitter.mag_methods}')

            wide_planet_fitter.run()
            self._log_file_only(wide_planet_fitter.get_diagnostic_str())
            return wide_planet_fitter.best

        fit_wide_planet()
        raise NotImplementedError('fitting binary models only partially implemented')

    # ---------------------------------------------------------------------
    # Data helpers:
    # ---------------------------------------------------------------------
    def set_residuals(self, pspl_params):
        """
        Calculate and store residuals from a PSPL model.

        Parameters
        ----------
        pspl_params : dict
            PSPL model parameters
        """
        event = MulensModel.Event(
            datasets=self.datasets, model=MulensModel.Model(pspl_params))
        event.fit_fluxes()
        residuals = []
        for i, dataset in enumerate(self.datasets):
            res, err = event.fits[i].get_residuals(phot_fmt='flux')
            residuals.append(
                MulensModel.MulensData(
                    [dataset.time, res, err], phot_fmt='flux',
                    bandpass=dataset.bandpass,
                    ephemerides_file=dataset.ephemerides_file))

        self.residuals = residuals

# ---------------------------------------------------------------------
    # Renormalization helpers:
    # ---------------------------------------------------------------------
    def renormalize_errors_and_refit(
            self,
            reference_model,
            datasets=None,
    ):
        """
        Renormalize photometric errors and refit all models.

        Parameters
        ----------
        reference_model : MulensModel.Model
            The model to use as reference for error renormalization.
            Can be obtained from a FitRecord via
            FitRecord.full_result.get_model()
        datasets : list or None, optional
            List of datasets to process. If None, use all datasets.

        Returns
        -------
        list
            Updated list of dataset objects after renormalization.
            If datasets=None was passed, returns self.datasets.
        """
        if datasets is None:
            datasets = self.datasets
            return_all = True
        else:
            return_all = False
            # Track labels of input datasets
            input_labels = [ds.plot_properties['label'] for ds in datasets]

        # Renormalize errors using the reference model
        error_factors = self._remove_outliers_and_calc_errfacs(
            reference_model,
            fit_datasets=datasets
        )
        self._apply_error_renormalization(error_factors)

        # Refit all models with renormalized errors
        self._refit_models()

        # Return updated dataset objects
        if return_all:
            return self.datasets
        else:
            # Map labels back to new dataset objects
            label_to_dataset = {ds.plot_properties['label']: ds
                               for ds in self.datasets}
            return [label_to_dataset[label] for label in input_labels]

    def _remove_outliers_and_calc_errfacs(self, reference_model, fit_datasets=None):
        """
        Remove outliers and calculate error renormalization factors.

        Parameters
        ----------
        reference_model : mmexo.Model
            Model to use for outlier detection
        fit_datasets : list or None, optional
            Datasets to fit. If None, uses self.datasets.

        Returns
        -------
        dict
            Dictionary mapping label to error renormalization factor
            for each dataset that was processed.
        """
        if fit_datasets is None:
            fit_datasets = self.datasets

        # Determine which datasets need processing
        datasets_to_process = [
            dataset for dataset in fit_datasets
            if dataset.plot_properties['label'] not in self.renorm_factors
        ]

        if not datasets_to_process:
            self._log("All datasets already have renormalization factors applied.")
            return {}

        # Create event with ALL datasets for proper flux fitting
        event = MulensModel.Event(
            datasets=fit_datasets, model=reference_model, coords=self.coords)
        event.fit_fluxes()

        self._log("Starting outlier removal...")

        error_factors_dict = {}

        # Process only the specified datasets
        for dataset in datasets_to_process:
            # Find index in fit_datasets
            if dataset not in fit_datasets:
                raise ValueError(f"Dataset {dataset} not found in fit_datasets")

            i = fit_datasets.index(dataset)
            dataset_name = dataset.plot_properties.get('label', f'Dataset {i}')
            self._log(f"\nProcessing {dataset_name}:")

            bad_index = -1
            n_good = np.sum(dataset.good)
            n_params = len(reference_model.parameters.as_dict())
            found_bad = []

            # Iteratively remove outliers
            while (bad_index is not None) and (n_good > 0):
                event.fit_fluxes()

                n_good = np.sum(dataset.good)
                dof = n_good - n_params

                if dof <= 0:
                    self._log(f"  Warning: dof={dof}, stopping outlier removal")
                    break

                # Calculate significance threshold
                max_sig = np.max([np.sqrt(2) * erfcinv(1. / dof), 3])

                # Get chi2 and error factor
                chi2 = event.get_chi2_for_dataset(i)
                errfac = np.sqrt(chi2 / dof)

                self._log_file_only(f"  errfac={errfac:.3f}, n_good={n_good}, dof={dof}")

                # Get residuals and calculate sigma
                (res, err) = event.fits[i].get_residuals(phot_fmt='flux', bad=True)
                sigma = np.abs(res / (err * errfac))

                # Find outliers
                n_still_bad = np.sum(sigma[dataset.good] > max_sig)

                if n_still_bad > 0:
                    # Find and mark the worst point
                    i_worst = np.argmax(sigma[dataset.good])
                    bad_index = np.argwhere(sigma == sigma[dataset.good][i_worst])[0]

                    new_bad = dataset.bad.copy()
                    new_bad[bad_index] = True
                    dataset.bad = new_bad

                    found_bad.append(bad_index[0])
                    self._log_file_only(
                        f"  Marked point {bad_index[0]} as bad: "
                        f"n_bad={np.sum(dataset.bad)}, n_good={np.sum(dataset.good)}"
                    )
                else:
                    bad_index = None

            # Calculate final error factor
            event.fit_fluxes()
            final_chi2 = event.get_chi2_for_dataset(i)
            final_dof = np.sum(dataset.good) - n_params

            if final_dof > 0:
                final_errfac = np.sqrt(final_chi2 / final_dof)
            else:
                final_errfac = 1.0

            error_factors_dict[dataset.plot_properties['label']] = final_errfac

            # Summary
            if len(found_bad) > 0:
                self._log(f"  Removed {len(found_bad)} outliers, errfac={final_errfac:.3f}")
            else:
                self._log(f"  No outliers removed, errfac={final_errfac:.3f}")

        return error_factors_dict

    def _apply_error_renormalization(self, error_factors, datasets=None):
        """
        Recreate datasets with renormalized errors.

        Parameters
        ----------
        error_factors : dict
            Dictionary mapping label to error renormalization factor
        datasets : list or None, optional
            Datasets to renormalize. If None, renormalizes all datasets
            that have labels in error_factors.
        """
        if datasets is None:
            # Apply to all datasets that have factors
            datasets = [ds for ds in self.datasets
                        if ds.plot_properties['label'] in error_factors]

        if not datasets:
            self._log("No datasets to renormalize.")
            return

        self._log("\nApplying error renormalization...")

        # Get the signature of MulensData.__init__
        sig = inspect.signature(MulensModel.MulensData.__init__)

        new_datasets = []
        for dataset in datasets:
            # Get error factor for this dataset
            label = dataset.plot_properties['label']
            errfac = error_factors.get(label)
            if errfac is None:
                self._log(f"Warning: No error factor for {label}, skipping")
                continue

            # Build kwargs dict from original object's attributes
            kwargs = {}
            for param_name in sig.parameters:
                if param_name in ['self', 'data_list', 'good', 'phot_fmt', 'file_name']:
                    continue

                if hasattr(dataset, param_name):
                    kwargs[param_name] = getattr(dataset, param_name)

            # Create new dataset with scaled errors
            new_dataset = MulensModel.MulensData(
                data_list=[dataset.time, dataset.flux, errfac * dataset.err_flux],
                phot_fmt='flux',
                **kwargs
            )

            self._log(new_dataset)
            new_datasets.append(new_dataset)

        # Build mapping old -> new and replace in self.datasets
        old_to_new = dict(zip(datasets, new_datasets))
        self.datasets = [old_to_new.get(ds, ds) for ds in self.datasets]

        # Update flux fixing maps with new dataset objects
        self.fix_blend_flux_map = self._map_label_dict_to_datasets(self.fix_blend_flux)
        self.fix_source_flux_map = self._map_label_dict_to_datasets(self.fix_source_flux)

        # Store applied factors in state
        self.renorm_factors.update(error_factors)

        self._log("Datasets recreated with renormalized errors")

    def _refit_models(self):
        """
        Refit all models using current datasets and previous fit results as
        initial parameters.

        Updates all_fit_results in place with new fit results.
        """
        self._log("\nUpdating fits...")
        for key, fit_record in self.all_fit_results.items():
            # Get the fitter object
            fitter = fit_record.full_result.fitter

            # Update with current (potentially renormalized) datasets
            fitter.datasets = self.datasets

            # Use previous fit as starting point
            fitter.initial_model_params = fit_record.params

            # Refit
            fitter.run()

            full_result = mmexo.MMEXOFASTFitResults(fitter)
            new_record = mmexo.FitRecord.from_full_result(
                model_key=key,
                full_result=full_result,
                fixed=False,
            )
            self.all_fit_results.set(new_record)

            self._log(f'{mmexo.fit_types.model_key_to_label(key)}: {fitter.best}')
            self._log_file_only(fitter.get_diagnostic_str())

# ---------------------------------------------------------------------
    # External search helpers:
    # ---------------------------------------------------------------------
    def do_ef_grid_search(self):
        """
        Run a EventFinderGridSearch.

        Returns
        -------
        dict
            Best EventFinder grid point parameters
        """
        ef_grid = mmexo.EventFinderGridSearch(datasets=self.datasets)
        ef_grid.run()

        if self.output is not None and self.output.config.save_plots:
            fig = ef_grid.plot()
            self.output.save_plot('ef_grid', fig)

        return ef_grid.best

    def do_af_grid_search(self):
        """
        Run an AnomalyFinderGridSearch.

        Returns
        -------
        dict
            Best AnomalyFinder grid point parameters
        """
        self.set_residuals(self.all_fit_results.select_best_static_pspl().params)
        af_grid = mmexo.AnomalyFinderGridSearch(residuals=self.residuals)
        # May need to update value of teff_min
        af_grid.run()
        return af_grid.best

    def get_anomaly_lc_params(self):
        """
        Estimate anomaly light curve parameters.

        Returns
        -------
        dict
            Anomaly light curve parameters
        """
        estimator = mmexo.estimate_params.AnomalyPropertyEstimator(
            datasets=self.datasets, pspl_params=self.all_fit_results.select_best_static_pspl().params,
            af_results=self.best_af_grid_point)
        return estimator.get_anomaly_lc_parameters()

    # ---------------------------------------------------------------------
    # Output
    # ---------------------------------------------------------------------
    def _log(self, msg: str) -> None:
        """
        Log message to console/file based on verbose/save_log settings.

        Parameters
        ----------
        msg : str
            Message to log
        """
        if self.output is not None:
            self.output.log(msg)
        elif self.verbose:
            # Fallback: print to console if no output manager but verbose=True
            print(msg)

    def _log_file_only(self, msg: str) -> None:
        """
        Log message to file only (never console).

        Parameters
        ----------
        msg : str
            Message to log to file
        """
        if self.output is not None and self.output.logger is not None:
            self.output.logger.info(msg)

    def _output_latex_table(self, name: str = 'results', models=None) -> None:
        """
        Output LaTeX table of results.

        Parameters
        ----------
        name : str, optional
            Table name. Default is 'results'.
        models : list or None, optional
            Models to include in table. If None, includes all.
        """
        if self.output is not None:
            table_str = self.make_ulens_table(table_type='latex', models=models)
            self.output.save_latex_table(name, table_str)

    def _save_restart_state(self) -> None:
        """Save current state for restarting fits."""
        if self.output is None:
            return

        restart_data = {
            'config': self._get_config(),
            'state': self._get_state(),
        }

        state_bytes = pickle.dumps(restart_data)
        self.output.save_restart_state(state_bytes)

    def make_ulens_table(self, table_type: Optional[str], models=None) -> str:
        """
        Return a string consisting of a formatted table summarizing the results
        of the microlensing fits.

        Parameters
        ----------
        table_type : str or None
            'ascii' (default) or 'latex'.
        models : list, optional
            - None: include all models in self.all_fit_results
            - list of labels (str): e.g., ['PSPL static', 'PSPL par u0+']
            - list of mmexo.ModelKey: explicit selection.

        Returns
        -------
        str
            Table in the requested format.
        """

        def order_df(df: pd.DataFrame) -> pd.DataFrame:
            """
            Order parameters in a human-friendly way (ulens params first,
            then fluxes).

            Parameters
            ----------
            df : pd.DataFrame
                DataFrame to order

            Returns
            -------
            pd.DataFrame
                Ordered DataFrame
            """

            def get_ordered_ulens_keys_for_repr(n_sources: int = 1):
                """
                Define the default order of microlensing parameters.

                Parameters
                ----------
                n_sources : int, optional
                    Number of sources. Default is 1.

                Returns
                -------
                list
                    Ordered list of parameter names
                """
                basic_keys = ["t_0", "u_0", "t_E", "rho", "t_star"]
                additional_keys = [
                    "pi_E_N", "pi_E_E", "t_0_par", "s", "q", "alpha",
                    "convergence_K", "shear_G", "ds_dt", "dalpha_dt", "s_z",
                    "ds_z_dt", "t_0_kep",
                    "x_caustic_in", "x_caustic_out", "t_caustic_in", "t_caustic_out",
                    "xi_period", "xi_semimajor_axis", "xi_inclination",
                    "xi_Omega_node", "xi_argument_of_latitude_reference",
                    "xi_eccentricity", "xi_omega_periapsis", "q_source", "t_0_xi",
                ]

                ordered_keys: list[str] = []
                if n_sources > 1:
                    for param_head in basic_keys:
                        if param_head == "t_E":
                            ordered_keys.append(param_head)
                        else:
                            for i in range(n_sources):
                                ordered_keys.append(f"{param_head}_{i + 1}")
                else:
                    ordered_keys = list(basic_keys)

                ordered_keys.extend(additional_keys)

                # New for MMEXOFAST:
                ordered_keys = ["chi2", "N_data"] + ordered_keys

                return ordered_keys

            def get_ordered_flux_keys_for_repr() -> list[str]:
                """
                Get ordered list of flux parameter names.

                Returns
                -------
                list
                    Ordered list of flux parameter names
                """
                flux_keys: list[str] = []
                for i, dataset in enumerate(self.datasets):
                    if "label" in dataset.plot_properties.keys():
                        obs = dataset.plot_properties["label"].split("-")[0]
                    else:
                        obs = i

                    if dataset.bandpass is not None:
                        band = dataset.bandpass
                    else:
                        band = "mag"

                    flux_keys.append(f"{band}_S_{obs}")
                    flux_keys.append(f"{band}_B_{obs}")

                return flux_keys

            def get_ordered_keys_for_repr() -> list[str]:
                """
                Get complete ordered list of parameter names.

                Returns
                -------
                list
                    Ordered list of all parameter names
                """
                ulens_keys = get_ordered_ulens_keys_for_repr()
                flux_keys = get_ordered_flux_keys_for_repr()
                return ulens_keys + flux_keys

            desired_order = get_ordered_keys_for_repr()
            order_map = {name: i for i, name in enumerate(desired_order)}

            df["sort_key"] = df["parameter_names"].map(order_map)
            df["orig_pos"] = range(len(df))

            # Anything not in desired_order goes to the end
            max_key = len(desired_order)
            df["sort_key"] = df["sort_key"].fillna(max_key)
            df = (
                df.sort_values(["sort_key", "orig_pos"])
                    .reset_index()
                    .drop(columns=["index", "sort_key", "orig_pos"])
            )
            return df

        if table_type is None:
            table_type = "ascii"

        # Normalize `models` to a list of (label, mmexo.FitRecord)
        model_label_record_pairs: list[tuple[str, mmexo.FitRecord]] = []

        if models is None:
            # All models currently in mmexo.AllFitResults
            for key, record in self.all_fit_results.items():
                label = mmexo.fit_types.model_key_to_label(key)
                model_label_record_pairs.append((label, record))
        else:
            for m in models:
                if isinstance(m, mmexo.FitKey):
                    key = m
                else:
                    # assume string label
                    key = mmexo.fit_types.label_to_model_key(m)
                record = self.all_fit_results.get(key)
                if record is None:
                    raise ValueError(f"No mmexo.FitRecord found for model {m!r}")
                label = mmexo.fit_types.model_key_to_label(key)
                model_label_record_pairs.append((label, record))

        results_table: Optional[pd.DataFrame] = None

        for label, record in model_label_record_pairs:
            new_column = record.to_dataframe()
            new_column = new_column.rename(
                columns={
                    "values": label,
                    "sigmas": f"sig [{label}]",
                }
            )

            if results_table is None:
                results_table = new_column
            else:
                results_table = results_table.merge(
                    new_column,
                    on="parameter_names",
                    how="outer",
                )

        if results_table is None:
            # No models; return empty table
            return ""

        results_table = order_df(results_table)

        if table_type == "latex":
            def fmt(name: str) -> str:
                """
                Format parameter name for LaTeX.

                Parameters
                ----------
                name : str
                    Parameter name

                Returns
                -------
                str
                    LaTeX formatted name
                """
                if name == "chi2":
                    return r"$\chi^2$"

                parts = name.split("_")
                if len(parts) == 1:
                    return f"${name}$"
                first = parts[0]
                rest = ", ".join(parts[1:])
                return f"${first}" + "_{" + rest + "}$"

            results_table["parameter_names"] = results_table["parameter_names"].apply(
                fmt
            )
            return results_table.to_latex(index=False)

        elif table_type == "ascii":
            with pd.option_context(
                    "display.max_rows", None,
                    "display.max_columns", None,
                    "display.width", None,
                    "display.float_format", "{:f}".format,
            ):
                return results_table.to_string(index=False)

        else:
            raise NotImplementedError(table_type + " not implemented.")

    # ---------------------------------------------------------------------
    # EXOZIPPy Helpers
    # ---------------------------------------------------------------------
    def initialize_exozippy(self):
        """
        Get the best-fit microlensing parameters for initializing exozippy fitting.

        Returns
        -------
        dict
            Dictionary with keys:
                'fits': list of dict
                    [{'parameters': {dict of ulens parameters},
                      'sigmas': {dict of uncertainties in ulens parameters}} ...]
                'errfacs': list of error renormalization factors for each dataset.
                    DEFAULT: None
                'mag_methods': list of magnification methods following the MulensModel
                    convention. DEFAULT: None
        """
        initializations = {'fits': [], 'errfacs': None, 'mag_methods': None}

        if self.fit_type == 'point lens':
            fits = []
            for par_key in self._iter_parallax_point_lens_keys():
                fits.append({'parameters': self.all_fit_results.get(par_key).params,
                             'sigmas': self.all_fit_results.get(par_key).sigmas})

            initializations['fits'] = fits
        else:
            raise NotImplementedError('initialize_exozippy only implemented for point lens fits')

        return initializations
