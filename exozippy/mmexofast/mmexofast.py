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

import pandas as pd
import os.path

import MulensModel

import exozippy.mmexofast as mmexo


# ============================================================================
# MMEXOFASTFitter
# ============================================================================
def fit(files=None, fit_type=None, **kwargs):
    """
    # Fit a microlensing light curve using MMEXOFAST
    #
    # :param files:
    # :param coords:
    # :param priors:
    # :param fit_type:
    # :param print_results:
    # :param verbose:
    # :param output_file:
    # :return:
    #
    # ***
    # Q1: Should this also include an `input_file` option?
    # Q2: What about `initial_param` = *dict* of microlensing parameters option?
    # Open issue: as written, only supports single solutions. 1/26/26 JCY: might be fixed?
    # ***
    #
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
        'fix_source_flux'
    ]

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
            fix_sourc_flux=None,
            renormalize_errors=False,
            parallax_grid=False,
            verbose=False,
            initial_results=None,
            output_config=None,
            restart_file=None,
    ):
        # Load restart data
        saved_config, saved_state = self._load_restart_data(restart_file)

        # Merge provided params with saved config
        config = self._merge_config(saved_config, locals())

        # Set all config attributes
        self._set_config_attributes(config)

        # Setup datasets - PRIORITY ORDER:
        # 1. Explicitly provided datasets (highest priority)
        # 2. Saved datasets from restart (preserves bad flags!)
        # 3. Create from files
        if datasets is not None:
            self.datasets = datasets
            self.dataset_to_filename = {}
        elif 'datasets' in saved_state:
            # Restore from pickle (preserves bad flags, error renorm, etc.)
            self.datasets = saved_state['datasets']
            self.dataset_to_filename = saved_state.get('dataset_to_filename', {})
        elif self.files is not None:
            self.datasets, self.dataset_to_filename = self._create_mulensdata_objects(self.files)
        else:
            raise ValueError("Must provide files, datasets, or restart_file")

        # Map flux fixing options using filename mapping
        self.fix_blend_flux_map = self._map_filename_dict_to_datasets(self.fix_blend_flux)
        self.fix_source_flux_map = self._map_filename_dict_to_datasets(self.fix_source_flux)

        self.n_loc = self._count_loc()
        self.residuals = None

        # Output
        self.output = mmexo.OutputManager(output_config, verbose=self.verbose) if output_config is not None else None

        # Restore state
        self._restore_state(saved_state)

        # Load initial results if provided
        if initial_results is not None:
            self._load_initial_results(initial_results)

    # ---------------------------------------------------------------------
    # restart helpers:
    # ---------------------------------------------------------------------
    def _get_config(self) -> dict:
        """Automatically extract config from attributes."""
        return {key: getattr(self, key, None) for key in self.CONFIG_KEYS}

    def _merge_config(self, saved_config, provided_params):
        """Merge saved config with provided params (provided wins)."""
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
        """Set all config attributes from config dict."""
        for key in self.CONFIG_KEYS:
            setattr(self, key, config[key])

    def _get_fitter_kwargs(self) -> dict:
        """Bundle fitter options for passing to SFitFitter."""
        return {
            'coords': self.coords,
            'mag_methods': self.mag_methods,
            'limb_darkening_coeffs_u': self.limb_darkening_coeffs_u,
            'limb_darkening_coeffs_gamma': self.limb_darkening_coeffs_gamma,
            'fix_source_flux': self.fix_source_flux_map,
            'fix_blend_flux': self.fix_blend_flux_map
        }

    def _get_state(self) -> dict:
        """Get all computed state (fit results)."""
        return {
            'all_fit_results': self.all_fit_results,
            'best_ef_grid_point': self.best_ef_grid_point,
            'best_af_grid_point': self.best_af_grid_point,
            'anomaly_lc_params': self.anomaly_lc_params,
            'n_loc': self.n_loc,
            'datasets': self.datasets,
        }

    def _load_restart_data(self, restart_file):
        """Load config and state from restart file."""
        if restart_file is None:
            return {}, {}

        with open(restart_file, 'rb') as f:
            data = pickle.load(f)
        return data.get('config', {}), data.get('state', {})

    def _restore_state(self, saved_state):
        """Restore computed state from saved data."""
        self.all_fit_results = saved_state.get('all_fit_results', mmexo.AllFitResults())
        self.best_ef_grid_point = saved_state.get('best_ef_grid_point')
        self.best_af_grid_point = saved_state.get('best_af_grid_point')
        self.anomaly_lc_params = saved_state.get('anomaly_lc_params')

    # ---------------------------------------------------------------------
    # Loading initial (user-supplied) information
    # ---------------------------------------------------------------------
    def _create_mulensdata_objects(self, files):
        """
        Create MulensData objects and map them to filenames.

        Returns
        -------
        datasets : list
            List of MulensData objects
        dataset_to_filename : dict
            Maps each dataset to its source filename
        """
        if isinstance(files, str):
            files = [files]

        datasets = []
        dataset_to_filename = {}

        for filename in files:
            if not os.path.exists(filename):
                raise FileNotFoundError(f"Data file {filename} does not exist")

            kwargs = mmexo.observatories.get_kwargs(filename)
            data = MulensModel.MulensData(file_name=filename, **kwargs)
            datasets.append(data)
            dataset_to_filename[data] = filename  # ← Store mapping

        return datasets, dataset_to_filename

    def _map_filename_dict_to_datasets(self, filename_dict) -> dict:
        """
        Map a dict[filename: value] to dict[dataset: value].

        Parameters
        ----------
        filename_dict : dict or None
            Keys are filenames, values are bool

        Returns
        -------
        dict
            Keys are MulensData objects, values from filename_dict
        """
        if filename_dict is None:
            # Default: False for all datasets
            return {dataset: False for dataset in self.datasets}

        result = {}
        for dataset in self.datasets:
            # Use our stored mapping
            filename = self.dataset_to_filename.get(dataset)

            if filename and filename in filename_dict:
                result[dataset] = filename_dict[filename]
            else:
                # Default if not specified
                result[dataset] = False

        return result

    def _count_loc(self):
        """
        # Determine how many locations (e.g. Earth vs. Earth + Space) an event was observed from.
        # :return:
        """

        if len(self.datasets) == 1:
            return 1

        else:
            locs = []
            for dataset in self.datasets:
                if dataset.ephemerides_file not in locs:
                    locs.append(dataset.ephemerides_file)

            return len(locs)

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
        """
        for label, payload in initial_results.items():
            key = mmexo.model_types.label_to_model_key(label)
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

    # ---------------------------------------------------------------------
    # Public orchestration methods:
    # ---------------------------------------------------------------------
    """
    #
    #    fit
    #    fit_point_lens
    #    fit_binary_lens
    # 
    """

    def fit(self):
        """
        Perform the fit according to the settings established when the MMEXOFASTFitter object was created.

        :return: None
        """
        if self.fit_type is None:
            # Maybe "None" means initial mulens parameters were passed,
            # so we can go straight to a mmexofast_fit?
            raise ValueError(
                'You must set the fit_type when initializing the ' +
                'MMEXOFASTFitter(): fit_type=("point lens", "binary lens")')

        if self.fit_type == 'point lens':
            self.fit_point_lens()

        elif self.fit_type == 'binary lens':
            self.fit_binary_lens()

        self._output_latex_table()

    def fit_point_lens(self) -> None:
        """
        Run the full point-lens workflow.

        - static PSPL
        - optional static FSPL
        - parallax branches (PSPL or FSPL depending on finite_source)
        - optional renormalization + refits
        """
        self._ensure_static_point_lens()  # shared
        self._ensure_static_finite_point_lens()  # shared (if finite_source)
        self._ensure_point_lens_parallax_models()  # shared (if you want)
        if not self.verbose():
            self._save_restart_state()

        if self.renormalize_errors:
            self.renormalize_errors_and_refit()

    def fit_binary_lens(self) -> None:
        """
        Run binary-lens workflow, building on point-lens pieces.

        - static PSPL
        - Anomaly Finder search
        # TODO: re-fit as needed depending on AF results


        """
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
            key: mmexo.ModelKey,
            fit_func,
    ) -> mmexo.FitRecord:
        """
        Ensure there is an up-to-date mmexo.FitRecord for `key`.

        Parameters
        ----------
        key : mmexo.ModelKey
            Which model to fit.
        fit_func : callable
            Function that runs the fit:
            `fit_func(initial_params: Optional[dict]) -> mmexo.MMEXOFASTFitResults`.

        Returns
        -------
        mmexo.FitRecord
            The current record for this model (existing or newly fitted).
        """
        record = self.all_fit_results.get(key)

        # If we have a fixed or complete result, reuse it
        if record is not None and (record.fixed or record.is_complete):
            return record

        # Use existing params as a starting point, if present
        initial_params = record.params if record is not None else None

        # Run the actual fit
        full_result = fit_func(initial_params=initial_params)

        # Derive renorm factors from current state, if any
        renorm_factors = self._current_renorm_factors()

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

    # "ensure" means "Make sure that this thing exists and is up to date; if it already does, don’t redo the work.”

    def _ensure_static_point_lens(self) -> None:
        """Make sure static PSPL exists in all_fit_results."""
        static_pspl_key = mmexo.ModelKey(
            lens_type=mmexo.LensType.POINT,
            source_type=mmexo.SourceType.POINT,
            parallax_branch=mmexo.ParallaxBranch.NONE,
            lens_orb_motion=mmexo.LensOrbMotion.NONE,
        )

        def fit_static_pspl(initial_params=None):
            return self._fit_initial_pspl_model(initial_params=initial_params)

        self.run_fit_if_needed(static_pspl_key, fit_static_pspl)

    def _ensure_static_finite_point_lens(self) -> None:
        """Make sure static FSPL exists, if finite_source is enabled."""
        if not self.finite_source:
            return

        static_fspl_key = mmexo.ModelKey(
            lens_type=mmexo.LensType.POINT,
            source_type=mmexo.SourceType.FINITE,
            parallax_branch=mmexo.ParallaxBranch.NONE,
            lens_orb_motion=mmexo.LensOrbMotion.NONE,
        )

        def fit_static_fspl(initial_params=None):
            return self._fit_static_fspl_model(initial_params=initial_params)

        self.run_fit_if_needed(static_fspl_key, fit_static_fspl)

    def _ensure_point_lens_parallax_models(self) -> None:
        """Make sure all configured point-lens parallax branches are fitted."""
        for par_key in self._iter_parallax_point_lens_keys():
            def make_fit_func(k: mmexo.ModelKey):
                def fit_func(initial_params=None):
                    return self._fit_pl_parallax_model(k, initial_params=initial_params)

                return fit_func

            self.run_fit_if_needed(par_key, make_fit_func(par_key))

    # ---------------------------------------------------------------------
    # Point-lens helpers:
    # ---------------------------------------------------------------------
    """
    #
    #    _fit_initial_pspl_model
    #    _fit_static_fspl_model
    #
    #    _iter_parallax_point_lens_keys
    #    BRANCH_SIGNS
    #    _apply_branch_signs
    #    _get_parallax_initial_params
    #    _fit_pl_parallax_model
    #
    """
    # static point lenses
    def _fit_initial_pspl_model(
        self,
        initial_params: Optional[Dict[str, float]] = None,
    ) -> mmexo.MMEXOFASTFitResults:
        """
        Estimate or accept starting point for PSPL, then run SFitFitter.
        Returns mmexo.MMEXOFASTFitResults.

        EF grid is only used if `initial_params` is None and
        best_ef_grid_point is not yet available.
        """
        if initial_params is None:
            if self.best_ef_grid_point is None:
                self.best_ef_grid_point = self.do_ef_grid_search()
                self._log(f"Best EF grid point {self.best_ef_grid_point}")

            pspl_est_params = mmexo.estimate_params.get_PSPL_params(
                self.best_ef_grid_point,
                self.datasets,
            )
            self._log(f"Initial PSPL Estimate {pspl_est_params}")
        else:
            pspl_est_params = initial_params
            self._log(f"Using initial PSPL params (user/previous): {pspl_est_params}")

        fitter = mmexo.fitters.SFitFitter(
            initial_model_params=pspl_est_params, datasets=self.datasets, **self._get_fitter_kwargs())
        fitter.run()
        self._log(f'Initial SFit {fitter.best}')

        return mmexo.MMEXOFASTFitResults(fitter)

    def _fit_static_fspl_model(
        self,
        initial_params: Optional[Dict[str, float]] = None,
    ) -> mmexo.MMEXOFASTFitResults:
        """
        Fit a finite-source point-lens (FSPL) model.

        TODO: implement actual FSPL parameter estimation and fitting logic.
        """
        if initial_params is None:
            # Example: seed from static PSPL record if available.
            static_pspl_key = mmexo.ModelKey(
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
            initial_model_params=fspl_est_params, datasets=self.datasets, **self._get_fitter_kwargs())
        fitter.run()
        self._log(f'FSPL: {fitter.best}')

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
        """
        su0_src, spi_src = self.BRANCH_SIGNS[src_branch]
        su0_tgt, spi_tgt = self.BRANCH_SIGNS[target_branch]

        u0_factor = su0_tgt / su0_src
        piN_factor = spi_tgt / spi_src

        if "u_0" in params:
            params["u_0"] *= u0_factor
        if "pi_E_N" in params:
            params["pi_E_N"] *= piN_factor

    def _iter_parallax_point_lens_keys(self) -> Iterable[mmexo.ModelKey]:
        """
        Yield mmexo.ModelKeys for all point-lens parallax models consistent with n_loc.
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
            yield mmexo.ModelKey(
                lens_type=mmexo.LensType.POINT,
                source_type=(
                    mmexo.SourceType.FINITE if self.finite_source else mmexo.SourceType.POINT
                ),
                parallax_branch=branch,
                lens_orb_motion=mmexo.LensOrbMotion.NONE,
            )

    def _get_parallax_initial_params(
            self,
            key: mmexo.ModelKey,
            initial_params: Optional[Dict[str, float]],
    ) -> Dict[str, float]:
        """
        Decide how to initialize parameters for a parallax point-lens fit.

        Priority:
        1. Use provided initial_params if not None.
        2. Seed from an existing parallax branch result, transformed via sign flips.
        3. Fallback to static point-lens params (PSPL/FSPL) for this source_type.
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

            other_key = mmexo.ModelKey(
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
        static_key = mmexo.ModelKey(
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

    # --- UPDATED: parallax fitter uses the helpers ------------------------

    def _fit_pl_parallax_model(
            self,
            key: mmexo.ModelKey,
            initial_params: Optional[Dict[str, float]] = None,
    ) -> mmexo.MMEXOFASTFitResults:
        """
        Fit a point-lens parallax model for the given parallax branch.
        """
        par_est_params = self._get_parallax_initial_params(key, initial_params)

        fitter = mmexo.fitters.SFitFitter(
            initial_model_params=par_est_params, datasets=self.datasets, **self._get_fitter_kwargs())
        fitter.run()
        self._log(f'{mmexo.model_types.model_key_to_label(key)}: {fitter.best}')

        return mmexo.MMEXOFASTFitResults(fitter)

    # ---------------------------------------------------------------------
    # Binary-lens helpers:
    # ---------------------------------------------------------------------
    def _run_af_grid_search(self):
        if self.best_af_grid_point is None:
            self.best_af_grid_point = self.do_af_grid_search()
            self._log(f'Best AF grid {self.best_af_grid_point}')

        if self.anomaly_lc_params is None:
            self.anomaly_lc_params = self.get_anomaly_lc_params()
            self._log(f'Anomaly Params {self.anomaly_lc_params}')

    def _fit_binary_models(self):

        def fit_wide_planet():
            # So far, this only fits wide planet models in the GG97 limit.
            # print(self.anomaly_lc_params)
            wide_planet_fitter = mmexo.fitters.WidePlanetFitter(
                datasets=self.datasets, anomaly_lc_params=self.anomaly_lc_params,
                #emcee_settings=self.emcee_settings, pool=self.pool)
                )
            wide_planet_fitter.estimate_initial_parameters()
            self._log(
                f'Initial 2L1S Wide Model {wide_planet_fitter.initial_model}' +
                f'\nmag methods {wide_planet_fitter.mag_methods}')

            wide_planet_fitter.run()
            return wide_planet_fitter.best

        fit_wide_planet()
        raise NotImplementedError('fitting binary models only partially implemented')
        #if self.emcee:
        #    self.results = self.fit_anomaly()
        #    if self.verbose:
        #        print('Results', self.results)

    # ---------------------------------------------------------------------
    # Data helpers:
    # ---------------------------------------------------------------------
    def set_residuals(self, pspl_params):
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
    """
    #    _current_renorm_factors
    #    renormalize_errors_and_refit
    """

    def _current_renorm_factors(self) -> Optional[Dict[str, Any]]:
        """
        Return the current renormalization factors, if any.

        TODO: Implement this based on how you store per-dataset renorm info.
        """
        return None

    def renormalize_errors_and_refit(self) -> None:
        """
        Renormalize photometric errors based on selected best model, then
        optionally refit some or all models.

        TODO: implement:
        - choose reference model (e.g., best χ² among static PSPL/FSPL/parallax)
        - compute new renorm factors per dataset
        - update mmexo.FitRecord.renorm_factors for impacted models
        - optionally clear is_complete and call run_fit_if_needed again.
        """
        raise NotImplementedError

    # ---------------------------------------------------------------------
    # External search helpers:
    # ---------------------------------------------------------------------
    """
    #    do_ef_grid_search
    #    do_af_grid_search
    """
    def do_ef_grid_search(self):
        """
        Run a :py:class:`mmexofast.gridsearches.EventFinderGridSearch`
        :return: *dict* of best EventFinder grid point.
        """
        ef_grid = mmexo.EventFinderGridSearch(datasets=self.datasets)
        ef_grid.run()

        if self.output is not None and self.output.config.save_plots:
            fig = ef_grid.plot()
            self.output.save_plot('ef_grid', fig)

        return ef_grid.best

    def do_af_grid_search(self):
        self.set_residuals(self.all_fit_results.select_best_static_pspl().params)
        af_grid = mmexo.AnomalyFinderGridSearch(residuals=self.residuals)
        # May need to update value of teff_min
        af_grid.run()
        return af_grid.best

    def get_anomaly_lc_params(self):
        estimator = mmexo.estimate_params.AnomalyPropertyEstimator(
            datasets=self.datasets, pspl_params=self.all_fit_results.select_best_static_pspl().params,
            af_results=self.best_af_grid_point)
        return estimator.get_anomaly_lc_parameters()

    # ---------------------------------------------------------------------
    # Output
    # ---------------------------------------------------------------------
    def _log(self, msg: str) -> None:
        """Log message to console/file based on verbose/save_log settings."""
        if self.output is not None:
            self.output.log(msg)
        elif self.verbose:
            # Fallback: print to console if no output manager but verbose=True
            print(msg)

    def _log_file_only(self, msg: str) -> None:
        """Log message to file only (never console)."""
        if self.output is not None and self.output.logger is not None:
            self.output.logger.info(msg)

    def _output_latex_table(self,  name: str = 'results', models=None) -> None:
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
        ----------class mmexo.AllFitResults:
    def __init__(self):
        self._records: Dict[ModelKey, mmexo.FitRecord] = {}

    # --- internal helper ---
    def _normalize_key(self, key_or_label: str | ModelKey) -> ModelKey:
        if isinstance(key_or_label, ModelKey):
            return key_or_label
        return label_to_model_key(key_or_label)

    def get(self, key_or_label: str | ModelKey) -> Optional[mmexo.FitRecord]:
        key = self._normalize_key(key_or_label)
        return self._records.get(key)

    def set(self, record: mmexo.FitRecord) -> None:
        self._records[record.model_key] = record

    def has(self, key_or_label: str | ModelKey) -> bool:
        key = self._normalize_key(key_or_label)
        return key in self._records

    def keys(self, labels: bool = False):
        if labels:
            return [model_key_to_label(k) for k in self._records.keys()]
        return list(self._records.keys())

    def items(self, labels: bool = False):
        if labels:
            return [(model_key_to_label(k), r) for k, r in self._records.items()]
        return list(self._records.items())
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
            """

            def get_ordered_ulens_keys_for_repr(n_sources: int = 1):
                """
                Define the default order of microlensing parameters.
                """
                print(
                    "make_ulens_table.order_df(): This code was lifted verbatim "
                    "from MM. It would be better to refactor it in MM and just "
                    "use the function. Maybe as a Utils."
                )

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
                label = mmexo.model_types.model_key_to_label(key)
                model_label_record_pairs.append((label, record))
        else:
            for m in models:
                if isinstance(m, mmexo.ModelKey):
                    key = m
                else:
                    # assume string label
                    key = mmexo.model_types.label_to_model_key(m)
                record = self.all_fit_results.get(key)
                if record is None:
                    raise ValueError(f"No mmexo.FitRecord found for model {m!r}")
                label = mmexo.model_types.model_key_to_label(key)
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
           #Get the best-fit microlensing parameters for initializing exozippy fitting.
           #
           #:return: *dict*
           #    items:
           #        'fits': *list* of *dict*
           #            [{'parameters': {*dict* of ulens parameters}, 'sigmas': {*dict* of uncertainties in
           #            ulensparameters}} ...]
           #        'errfacs': *list* of error renormalization factors for each dataset. DEFAULT: None
           #        'mag_methods': *list* of magnification methods following the MulensModel convention. DEFAULT: None
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
