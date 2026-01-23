"""
High-level functions for fitting microlensing events.
"""
"""
mmexofast_fitter_arch.py

Architectural sketch for MMEXOFASTFitter with:
- Structured model keys
- Centralized fit result registry (AllFitResults)
- FitRecord for partial/user-supplied vs full results
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, Iterable, Tuple

import numpy as np
import pandas as pd
import os.path

import MulensModel

import exozippy.mmexofast as mmexo
from exozippy.mmexofast import observatories # Seems untidy


# ============================================================================
# Enums and ModelKey
# ============================================================================


class LensType(Enum):
    POINT = "point"
    BINARY = "binary"   # for 2L1S, etc.


class SourceType(Enum):
    POINT = "point"
    FINITE = "finite"


class ParallaxBranch(Enum):
    NONE = "none"
    U0_PLUS = "u0+"
    U0_MINUS = "u0-"
    U0_PP = "u0++"
    U0_MM = "u0--"
    U0_PM = "u0+-"
    U0_MP = "u0-+"


class LensOrbMotion(Enum):
    NONE = "none"
    ORB_2D = "2Dorb"  # 2L1S 2-parameter orbital motion
    KEP = "kep"       # 2L1S Full Keplerian orbital motion
    # extendable


@dataclass(frozen=True)
class ModelKey:
    lens_type: LensType
    source_type: SourceType
    parallax_branch: ParallaxBranch
    lens_orb_motion: LensOrbMotion

    def __post_init__(self):
        # Enforce: point lens cannot have orbital motion.
        if (
            self.lens_type == LensType.POINT
            and self.lens_orb_motion is not LensOrbMotion.NONE
        ):
            raise ValueError("Point lenses must have lens_orb_motion == NONE")


# ============================================================================
# MMEXOFASTFitResults wrapper
# ============================================================================


class MMEXOFASTFitResults:
    """
    Wrapper containing results of a single fit, with convenience methods.
    This assumes `fitter` exposes `.best`, `.results`, `.parameters_to_fit`,
    `.datasets`, etc.
    """

    def __init__(self, fitter):
        self.fitter = fitter

    def get_params_from_results(self) -> Dict[str, float]:
        """
        Return a dict with just the best-fit microlensing parameters and values,
        i.e., something appropriate for using as input to `MulensModel.Model()`.
        """
        params = {key: value for key, value in self.best.items()}
        params.pop("chi2", None)
        return params

    def get_sigmas_from_results(self) -> Dict[str, float]:
        """
        Return a dict mapping parameter name -> 1-sigma uncertainty.
        """
        sigmas = {}
        for param, sigma in zip(self.parameters_to_fit, self.results.sigmas):
            sigmas[param] = sigma
        return sigmas

    def format_results_as_df(self) -> pd.DataFrame:
        """
        Build a summary DataFrame (fitted params, fixed params, flux params).
        """
        def get_df_fitted_parameters():
            parameters = list(self.parameters_to_fit)
            values = list(self.results.x[0:len(parameters)])
            sigmas = list(self.results.sigmas[0:len(parameters)])

            df = pd.DataFrame({
                "parameter_names": parameters,
                "values": values,
                "sigmas": sigmas,
            })
            return df

        def get_df_fixed_parameters():
            fixed_parameters = [
                p for p in self.all_model_parameters
                if p not in self.parameters_to_fit
            ]
            values = [self.best[param] for param in fixed_parameters]
            fixed_parameters.append("N_data")
            values.append(np.sum([np.sum(dataset.good) for dataset in self.datasets]))
            # TODO: optionally add chi2/N_data per dataset if desired.

            df = pd.DataFrame({
                "parameter_names": fixed_parameters,
                "values": values,
                "sigmas": [None] * len(fixed_parameters),
            })
            return df

        def get_df_flux_parameters():
            # TODO: decide if you want fluxes/magnitudes for all datasets or subset.
            parameters: list[str] = []
            values: list[float] = []
            sigmas: list[float] = []

            for i, dataset in enumerate(self.datasets):
                if "label" in dataset.plot_properties.keys():
                    obs = dataset.plot_properties["label"].split("-")[0]
                else:
                    obs = i

                if dataset.bandpass is not None:
                    band = dataset.bandpass
                else:
                    band = "mag"

                parameters.append(f"{band}_S_{obs}")
                parameters.append(f"{band}_B_{obs}")

                obs_index = len(self.parameters_to_fit) + 2 * i
                for index in range(2):
                    flux = self.results.x[obs_index + index]
                    if flux > 0:
                        err_flux = self.results.sigmas[obs_index + index]
                        mag, err_mag = MulensModel.utils.Utils.get_mag_and_err_from_flux(
                            flux, err_flux
                        )
                    else:
                        mag = "neg flux"
                        err_mag = np.nan

                    values.append(mag)
                    sigmas.append(err_mag)

            df = pd.DataFrame({
                "parameter_names": parameters,
                "values": values,
                "sigmas": sigmas,
            })
            return df

        df_fit = get_df_fitted_parameters()
        df_fixed = get_df_fixed_parameters()
        df_ulens = pd.concat((df_fit, df_fixed))
        df_flux = get_df_flux_parameters()
        df = pd.concat((df_ulens, df_flux), ignore_index=True)
        return df

    # Convenience properties
    @property
    def datasets(self):
        return self.fitter.datasets

    @property
    def best(self):
        return self.fitter.best

    @property
    def results(self):
        return self.fitter.results

    @property
    def parameters_to_fit(self):
        return self.fitter.parameters_to_fit

    @property
    def all_model_parameters(self):
        return self.fitter.best.keys()


# ============================================================================
# FitRecord and AllFitResults
# ============================================================================


@dataclass
class FitRecord:
    model_key: ModelKey

    # Core downstream data
    params: Dict[str, float]
    sigmas: Optional[Dict[str, float]] = None

    # Dataset / systematics state
    renorm_factors: Optional[Dict[str, Any]] = None

    # Rich fit output
    full_result: Optional[MMEXOFASTFitResults] = None

    # Control flags
    fixed: bool = False
    is_complete: bool = False

    @classmethod
    def from_full_result(
        cls,
        model_key: ModelKey,
        full_result: MMEXOFASTFitResults,
        renorm_factors: Optional[Dict[str, Any]] = None,
        fixed: bool = False,
    ) -> "FitRecord":
        params = full_result.get_params_from_results()
        try:
            sigmas = full_result.get_sigmas_from_results()
        except Exception:
            sigmas = None

        return cls(
            model_key=model_key,
            params=params,
            sigmas=sigmas,
            renorm_factors=renorm_factors,
            full_result=full_result,
            fixed=fixed,
            is_complete=True,
        )

    def to_dataframe(self) -> pd.DataFrame:
        """
        Return a DataFrame with columns:
        - 'parameter_names'
        - 'values'
        - 'sigmas'

        If full_result is available, delegate to it. Otherwise build a minimal
        DataFrame from params/sigmas (no fluxes, N_data, etc.).
        """
        if self.full_result is not None:
            return self.full_result.format_results_as_df()
        return self._minimal_dataframe()

    def _minimal_dataframe(self) -> pd.DataFrame:
        """
        Minimal fallback when we only have params/sigmas and no full_result.
        """
        param_names = list(self.params.keys())
        values = [self.params[name] for name in param_names]
        if self.sigmas is not None:
            sigmas = [self.sigmas.get(name, None) for name in param_names]
        else:
            sigmas = [None] * len(param_names)

        return pd.DataFrame(
            {
                "parameter_names": param_names,
                "values": values,
                "sigmas": sigmas,
            }
        )


class AllFitResults:
    """
    Central registry for all fit results, keyed by ModelKey.
    """

    def __init__(self):
        self._records: Dict[ModelKey, FitRecord] = {}

    def get(self, key: ModelKey) -> Optional[FitRecord]:
        return self._records.get(key)

    def set(self, record: FitRecord) -> None:
        self._records[record.model_key] = record

    def has(self, key: ModelKey) -> bool:
        return key in self._records

    def items(self) -> Iterable[Tuple[ModelKey, FitRecord]]:
        return self._records.items()


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
    # Open issue: as written, only supports single solutions.
    # ***
    #
    """
    fitter = MMEXOFASTFitter(files=files, fit_type=fit_type, **kwargs)
    fitter.fit()
    return fitter.results


class MMEXOFASTFitter:
    """
    Orchestrates workflows (PSPL, parallax, binary, etc.) and uses:
    - ModelKey to identify models
    - AllFitResults to store/reuse results
    """

    def __init__(
        self,
        files: list = None,
        datasets: list = None,
        fit_type: str = None,
        finite_source: bool = False,
        limb_darkening_coeffs_gamma: dict = None,
        limb_darkening_coeffs_u: dict = None,
        mag_methods: list = None,
        coords: str = None,
        renormalize_errors: bool = False,
        verbose: bool = False,
        initial_results: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        if datasets is not None:
            self.datasets = datasets
        else:
            self.datasets = self._create_mulensdata_objects(files)

        self.fit_type = fit_type
        self.finite_source = finite_source
        self.n_loc = self._count_loc()
        self.renormalize_errors = renormalize_errors

        self.fitter_kwargs = {
            'coords': coords, 'mag_methods': mag_methods,
            'limb_darkening_coeffs_u': limb_darkening_coeffs_u,
            'limb_darkening_coeffs_gamma': limb_darkening_coeffs_gamma}

        self.verbose = verbose

        self.best_ef_grid_point = None  # set by do_ef_grid_search()
        self.best_af_grid_point = None  # set by do_af_grid_search()

        self.all_fit_results = AllFitResults()

        if initial_results is not None:
            self._load_initial_results(initial_results)

    # ---------------------------------------------------------------------
    # Loading initial (user-supplied) information
    # ---------------------------------------------------------------------
    def _create_mulensdata_objects(self, files):
        if isinstance(files, (str)):
            files = [files]

        datasets = []
        for filename in files:
            if not os.path.exists(filename):
                raise FileNotFoundError(
                    "Data file {0} does not exist".format(filename))

            kwargs = observatories.get_kwargs(filename)
            data = MulensModel.MulensData(file_name=filename, **kwargs)
            datasets.append(data)

        return datasets

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

    def _label_to_model_key(self, label: str) -> ModelKey:
        """
        Map user-facing string labels (e.g. 'static PSPL', 'PL par u0+')
        to structured ModelKey.

        TODO: Implement mapping rules that correspond to your current labels.
        """
        if label == "static PSPL":
            return ModelKey(
                lens_type=LensType.POINT,
                source_type=SourceType.POINT,
                parallax_branch=ParallaxBranch.NONE,
                lens_orb_motion=LensOrbMotion.NONE,
            )
        if label == "static FSPL":
            return ModelKey(
                lens_type=LensType.POINT,
                source_type=SourceType.FINITE,
                parallax_branch=ParallaxBranch.NONE,
                lens_orb_motion=LensOrbMotion.NONE,
            )

        if label == "PL par u0+":
            return ModelKey(
                lens_type=LensType.POINT,
                source_type=(
                    SourceType.FINITE if self.finite_source else SourceType.POINT
                ),
                parallax_branch=ParallaxBranch.U0_PLUS,
                lens_orb_motion=LensOrbMotion.NONE,
            )

        if label == "PL par u0-":
            return ModelKey(
                lens_type=LensType.POINT,
                source_type=(
                    SourceType.FINITE if self.finite_source else SourceType.POINT
                ),
                parallax_branch=ParallaxBranch.U0_MINUS,
                lens_orb_motion=LensOrbMotion.NONE,
            )

        # TODO: add mappings for other parallax branches and binary models.
        raise ValueError(f"Unknown model label: {label}")

    def _model_key_to_label(self, key: ModelKey) -> str:
        """
        Convert a ModelKey back to a user-facing label like 'static PSPL',
        'PL par u0+', etc.

        TODO: make this consistent with _label_to_model_key and your existing
        naming scheme.
        """
        if key.lens_type == LensType.POINT and key.parallax_branch == ParallaxBranch.NONE:
            if key.source_type == SourceType.POINT:
                return "static PSPL"
            elif key.source_type == SourceType.FINITE:
                return "static FSPL"

        if key.lens_type == LensType.POINT and key.parallax_branch in (
                ParallaxBranch.U0_PLUS,
                ParallaxBranch.U0_MINUS,
                ParallaxBranch.U0_PP,
                ParallaxBranch.U0_MM,
                ParallaxBranch.U0_PM,
                ParallaxBranch.U0_MP,
        ):
            return f"PL par {key.parallax_branch.value}"

        # TODO: extend for binary models, motion models, etc.
        return f"{key.lens_type.value} / {key.source_type.value} / " \
               f"{key.parallax_branch.value} / {key.lens_orb_motion.value}"

    def _load_initial_results(self, initial_results: Dict[str, Dict[str, Any]]) -> None:
        """
        Load user-supplied initial results into AllFitResults.

        Expected format for each entry:
        {
            "params": {...},              # required
            "sigmas": {...},              # optional
            "renorm_factors": {...},      # optional
            "fixed": bool,                # optional
        }
        """
        for label, payload in initial_results.items():
            key = self._label_to_model_key(label)
            record = FitRecord(
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
    # Core helper: run_fit_if_needed
    # ---------------------------------------------------------------------

    def _current_renorm_factors(self) -> Optional[Dict[str, Any]]:
        """
        Return the current renormalization factors, if any.

        TODO: Implement this based on how you store per-dataset renorm info.
        """
        return None

    def run_fit_if_needed(
        self,
        key: ModelKey,
        fit_func,
    ) -> FitRecord:
        """
        Ensure there is an up-to-date FitRecord for `key`.

        Parameters
        ----------
        key : ModelKey
            Which model to fit.
        fit_func : callable
            Function that runs the fit:
            `fit_func(initial_params: Optional[dict]) -> MMEXOFASTFitResults`.

        Returns
        -------
        FitRecord
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

        new_record = FitRecord.from_full_result(
            model_key=key,
            full_result=full_result,
            renorm_factors=renorm_factors,
            fixed=False,
        )
        self.all_fit_results.set(new_record)
        return new_record

    # ---------------------------------------------------------------------
    # Point-lens fitting workflow
    # ---------------------------------------------------------------------

    def _iter_parallax_point_lens_keys(self) -> Iterable[ModelKey]:
        """
        Yield ModelKeys for all point-lens parallax models consistent with n_loc.
        """
        if self.n_loc == 1:
            branches = [ParallaxBranch.U0_PLUS, ParallaxBranch.U0_MINUS]
        else:
            branches = [
                ParallaxBranch.U0_PP,
                ParallaxBranch.U0_MM,
                ParallaxBranch.U0_PM,
                ParallaxBranch.U0_MP,
            ]

        for branch in branches:
            yield ModelKey(
                lens_type=LensType.POINT,
                source_type=(
                    SourceType.FINITE if self.finite_source else SourceType.POINT
                ),
                parallax_branch=branch,
                lens_orb_motion=LensOrbMotion.NONE,
            )

    def _fit_initial_pspl_model(
        self,
        initial_params: Optional[Dict[str, float]] = None,
    ) -> MMEXOFASTFitResults:
        """
        Estimate or accept starting point for PSPL, then run SFitFitter.
        Returns MMEXOFASTFitResults.

        EF grid is only used if `initial_params` is None and
        best_ef_grid_point is not yet available.
        """
        if initial_params is None:
            if self.best_ef_grid_point is None:
                self.best_ef_grid_point = self.do_ef_grid_search()
                if self.verbose:
                    print("Best EF grid point", self.best_ef_grid_point)

            pspl_est_params = mmexo.estimate_params.get_PSPL_params(
                self.best_ef_grid_point,
                self.datasets,
            )
            if self.verbose:
                print("Initial PSPL Estimate", pspl_est_params)
        else:
            pspl_est_params = initial_params
            if self.verbose:
                print("Using initial PSPL params (user/previous):", pspl_est_params)

        fitter = mmexo.fitters.SFitFitter(
            initial_model_params=pspl_est_params, datasets=self.datasets, **self.fitter_kwargs)
        fitter.run()
        if self.verbose:
            print('Initial SFit', fitter.best)

        return MMEXOFASTFitResults(fitter)

    def _fit_static_fspl_model(
        self,
        initial_params: Optional[Dict[str, float]] = None,
    ) -> MMEXOFASTFitResults:
        """
        Fit a finite-source point-lens (FSPL) model.

        TODO: implement actual FSPL parameter estimation and fitting logic.
        """
        if initial_params is None:
            # Example: seed from static PSPL record if available.
            static_pspl_key = ModelKey(
                lens_type=LensType.POINT,
                source_type=SourceType.POINT,
                parallax_branch=ParallaxBranch.NONE,
                lens_orb_motion=LensOrbMotion.NONE,
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
            initial_model_params=fspl_est_params, datasets=self.datasets, **self.fitter_kwargs)
        fitter.run()
        if self.verbose:
            print(f'FSPL: {fitter.best}')

        return MMEXOFASTFitResults(fitter)

    # --- parallax branch sign definitions and helpers ----------------

    BRANCH_SIGNS = {
        ParallaxBranch.U0_PLUS: (+1, +1),
        ParallaxBranch.U0_MINUS: (-1, -1),
        ParallaxBranch.U0_PP: (+1, +1),
        ParallaxBranch.U0_MM: (-1, -1),
        ParallaxBranch.U0_PM: (+1, -1),
        ParallaxBranch.U0_MP: (-1, +1),
    }

    def _apply_branch_signs(
            self,
            params: Dict[str, float],
            src_branch: ParallaxBranch,
            target_branch: ParallaxBranch,
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

    def _get_parallax_initial_params(
            self,
            key: ModelKey,
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
            if self.verbose:
                print(
                    f"Using provided initial params for parallax branch "
                    f"{key.parallax_branch}: {initial_params}"
                )
            return dict(initial_params)

        # 2. Seed from any existing parallax branch
        for other_branch in self.BRANCH_SIGNS.keys():
            if other_branch == key.parallax_branch:
                continue

            other_key = ModelKey(
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
            if self.verbose:
                print(
                    f"Seeding parallax branch {key.parallax_branch} from "
                    f"existing branch {other_branch} with transformed params: {base}"
                )
            return base

        # 3. Fallback: static point-lens
        static_key = ModelKey(
            lens_type=LensType.POINT,
            source_type=key.source_type,
            parallax_branch=ParallaxBranch.NONE,
            lens_orb_motion=LensOrbMotion.NONE,
        )
        static_record = self.all_fit_results.get(static_key)
        if static_record is None:
            raise RuntimeError(
                "Static point-lens model must be available before parallax fits."
            )

        base = dict(static_record.params)
        base['pi_E_N'] = 0.
        base['pi_E_E'] = 0.
        if self.verbose:
            print(
                f"Seeding parallax branch {key.parallax_branch} from "
                f"static model (source_type={key.source_type}): {base}"
            )
        return base

    # --- UPDATED: parallax fitter uses the helpers ------------------------

    def _fit_pl_parallax_model(
            self,
            key: ModelKey,
            initial_params: Optional[Dict[str, float]] = None,
    ) -> MMEXOFASTFitResults:
        """
        Fit a point-lens parallax model for the given parallax branch.
        """
        par_est_params = self._get_parallax_initial_params(key, initial_params)

        fitter = mmexo.fitters.SFitFitter(
            initial_model_params=par_est_params, datasets=self.datasets, **self.fitter_kwargs)
        fitter.run()
        if self.verbose:
            print(f'{key.parallax_branch}: {fitter.best}')

        return MMEXOFASTFitResults(fitter)

    def fit_point_lens(self) -> None:
        """
        Run the point-lens sequence:

        - static PSPL
        - optional static FSPL
        - parallax branches (PSPL or FSPL depending on finite_source)
        - optional renormalization + refits
        """
        # 1. static PSPL
        static_pspl_key = ModelKey(
            lens_type=LensType.POINT,
            source_type=SourceType.POINT,
            parallax_branch=ParallaxBranch.NONE,
            lens_orb_motion=LensOrbMotion.NONE,
        )

        def fit_static_pspl(initial_params=None):
            return self._fit_initial_pspl_model(initial_params=initial_params)

        self.run_fit_if_needed(static_pspl_key, fit_static_pspl)

        # 2. static FSPL (if requested)
        if self.finite_source:
            static_fspl_key = ModelKey(
                lens_type=LensType.POINT,
                source_type=SourceType.FINITE,
                parallax_branch=ParallaxBranch.NONE,
                lens_orb_motion=LensOrbMotion.NONE,
            )

            def fit_static_fspl(initial_params=None):
                return self._fit_static_fspl_model(initial_params=initial_params)

            self.run_fit_if_needed(static_fspl_key, fit_static_fspl)

        # 3. parallax point-lens models
        for par_key in self._iter_parallax_point_lens_keys():

            def make_fit_func(k: ModelKey):

                def fit_func(initial_params=None):
                    return self._fit_pl_parallax_model(
                        k, initial_params=initial_params
                    )

                return fit_func

            self.run_fit_if_needed(par_key, make_fit_func(par_key))

        # 4. optional renormalization + refits
        if self.renormalize_errors:
            self.renormalize_errors_and_refit()

    # ---------------------------------------------------------------------
    # TODO: other workflows (binary lens, anomaly search, etc.)
    # ---------------------------------------------------------------------

    # ---------------------------------------------------------------------
    # General workflow
    # ---------------------------------------------------------------------
    def fit(self):
        """
        # Perform the fit according to the settings established when the MMEXOFASTFitter object was created.
        #
        # :return: None
        """
        if self.fit_type is None:
            raise ValueError(
                'You must set the fit_type when initializing the ' +
                'MMEXOFASTFitter(): fit_type=("point lens", "binary lens")')

        self.fit_point_lens()

        if self.fit_type == 'binary lens':
            raise NotImplementedError('Need to update binary lens fitting to match the new architecture.')
            #self.best_af_grid_point = self.do_af_grid_search()
            #if self.verbose:
            #    print('Best AF grid', self.best_af_grid_point)
            #
            #self.anomaly_lc_params = self.get_anomaly_lc_params()
            #if self.verbose:
            #    print('Anomaly Params', self.anomaly_lc_params)
            #
            #if self.emcee:
            #    self.results = self.fit_anomaly()
            #    if self.verbose:
            #        print('Results', self.results)

    def do_ef_grid_search(self):
        """
        Run a :py:class:`mmexofast.gridsearches.EventFinderGridSearch`
        :return: *dict* of best EventFinder grid point.
        """
        ef_grid = mmexo.EventFinderGridSearch(datasets=self.datasets)
        ef_grid.run()
        return ef_grid.best

    def renormalize_errors_and_refit(self) -> None:
        """
        Renormalize photometric errors based on selected best model, then
        optionally refit some or all models.

        TODO: implement:
        - choose reference model (e.g., best χ² among static PSPL/FSPL/parallax)
        - compute new renorm factors per dataset
        - update FitRecord.renorm_factors for impacted models
        - optionally clear is_complete and call run_fit_if_needed again.
        """
        raise NotImplementedError

    # ---------------------------------------------------------------------
    # Output
    # ---------------------------------------------------------------------
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
            - list of labels (str): e.g., ['static PSPL', 'PL par u0+']
            - list of ModelKey: explicit selection.

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

        # Normalize `models` to a list of (label, FitRecord)
        model_label_record_pairs: list[tuple[str, FitRecord]] = []

        if models is None:
            # All models currently in AllFitResults
            for key, record in self.all_fit_results.items():
                label = self._model_key_to_label(key)
                model_label_record_pairs.append((label, record))
        else:
            for m in models:
                if isinstance(m, ModelKey):
                    key = m
                else:
                    # assume string label
                    key = self._label_to_model_key(m)
                record = self.all_fit_results.get(key)
                if record is None:
                    raise ValueError(f"No FitRecord found for model {m!r}")
                label = self._model_key_to_label(key)
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


'''
# OLD STUFF


import os.path
import warnings

import pandas as pd
import numpy as np
import copy
from collections import OrderedDict
import emcee

import MulensModel
import astropy.units

import sfit_minimizer as sfit
from exozippy import exozippy_getmcmcscale
import exozippy.mmexofast as mmexo
from exozippy.mmexofast import observatories

# Stuff ChatGPT thought was a good idea
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, Any, Optional


# ---- Data classes for defining model types ---- #
class LensType(Enum):
    POINT = "point"
    BINARY = "binary"   # 2L1S, etc.


class SourceType(Enum):
    POINT = "point"
    FINITE = "finite"


class ParallaxBranch(Enum):
    NONE = "none"
    U0_PLUS = "u0+"
    U0_MINUS = "u0-"
    U0_PP = "u0++"
    U0_MM = "u0--"
    U0_PM = "u0+-"
    U0_MP = "u0-+"


class LensOrbMotion(Enum):
    NONE = "none"
    ORB_2D_PAR = "2Dorb"  # 2L1S 2 parameter orbital motion
    KEP_PAR = "kep"       # 2L1S Kepler
    # extendable


@dataclass(frozen=True)
class ModelKey:
    lens_type: LensType
    source_type: SourceType
    parallax_branch: ParallaxBranch
    lens_orb_motion: LensOrbMotion

    def __post_init__(self):
        if self.lens_type == LensType.POINT and self.lens_orb_motion is not LensOrbMotion.NONE:
            raise ValueError("Point lenses must have lens_orb_motion == NONE")


# ---- Classes for Storing Fit Results ---- #

class FitResult():
"""
# Class containing the results of a fit and nice methods for accessing the results.
"""

    def __init__(self, fitter):
        self.fitter = fitter

    def get_params_from_results(self):
"""
# Take the results of a fit and return a dictionary with just the best-fit microlensing parameters and values,
# i.e., something appropriate for using as input to `MulensModel.Model()`.
# 
# :param :py:class:`MMEXOFASTFit` object.
# 
# :return: *dict* of microlensing parameters and values
"""
        params = {key: value for key, value in self.best.items()}
        params.pop('chi2')
        return params

    def get_sigmas_from_results(self):
"""
#  Take the results of a fit and return a dictionary with the uncertainties for each microlensing parameter.
# 
# :param :return: :py:class:`MMEXOFASTFit` object.
# 
# :return: *dict* of uncertainties in microlensing parameters and values
"""
        sigmas = {}
        for param, sigma in zip(self.parameters_to_fit, self.results.sigmas):
            sigmas[param] = sigma

        return sigmas

    def format_results_as_df(self):

        def get_df_fitted_parameters():
            parameters = [x for x in self.parameters_to_fit]
            values = [x for x in self.results.x[0:len(parameters)]]
            sigmas = [x for x in self.results.sigmas[0:len(parameters)]]

            df = pd.DataFrame({
                'parameter_names': parameters,
                'values': values,
                'sigmas': sigmas
            })

            return df

        def get_df_fixed_parameters():
            fixed_parameters = [p for p in self.all_model_parameters if p not in self.parameters_to_fit]
            values = [self.best[param] for param in fixed_parameters]
            fixed_parameters.append('N_data')
            values.append(np.sum([np.sum(dataset.good) for dataset in self.datasets]))
            print('QUESTION: Do we also want chi2s and N_data for individual datasets? Is that too much info?')

            df = pd.DataFrame({
                'parameter_names': fixed_parameters,
                'values': values,
                'sigmas': [None] * len(fixed_parameters)
            })
            return df

        def get_df_flux_parameters():
            print('QUESTION: Do we actually want magnitudes for all datasets or just the reference dataset?')
            parameters = []
            values = []
            sigmas = []

            for i, dataset in enumerate(self.datasets):
                if 'label' in dataset.plot_properties.keys():
                    obs = dataset.plot_properties['label'].split('-')[0]
                else:
                    obs = i

                if dataset.bandpass is not None:
                    band = dataset.bandpass
                else:
                    band = 'mag'

                parameters.append('{0}_S_{1}'.format(band, obs))
                parameters.append('{0}_B_{1}'.format(band, obs))

                obs_index = len(self.parameters_to_fit) + 2 * i
                for index in range(2):
                    flux = self.results.x[obs_index + index]
                    if flux > 0:
                        err_flux = self.results.sigmas[obs_index + index]
                        mag, err_mag = MulensModel.utils.Utils.get_mag_and_err_from_flux(flux, err_flux)
                    else:
                        mag = 'neg flux'
                        err_mag = np.nan

                    values.append(mag)
                    sigmas.append(err_mag)

            df = pd.DataFrame({
                'parameter_names': parameters,
                'values': values,
                'sigmas': sigmas
            })

            return df

        df_fit = get_df_fitted_parameters()
        df_fixed = get_df_fixed_parameters()
        df_ulens = pd.concat((df_fit, df_fixed))
        df_flux = get_df_flux_parameters()
        df = pd.concat((df_ulens, df_flux), ignore_index=True)
        #print('As df w/flux\n', df)
        return df

    @property
    def datasets(self):
        return self.fitter.datasets

    @property
    def best(self):
        return self.fitter.best

    @property
    def results(self):
        return self.fitter.results

    @property
    def parameters_to_fit(self):
        return self.fitter.parameters_to_fit

    @property
    def all_model_parameters(self):
        return self.fitter.best.keys()


@dataclass
class FitRecord:
    model_key: ModelKey

    # Core data needed downstream
    params: Dict[str, float]
    sigmas: Optional[Dict[str, float]] = None

    # Dataset / systematics
    renorm_factors: Optional[Dict[str, Any]] = None

    # Rich result object from a real run
    full_result: Optional[FitResult] = None

    # Control flags
    fixed: bool = False          # if True, don’t refit this model
    is_complete: bool = False    # True if full_result is present and trusted

    @classmethod
    def from_full_result(
        cls,
        model_key: ModelKey,
        full_result: FitResult,
        renorm_factors: Optional[Dict[str, Any]] = None,
        fixed: bool = False,
    ) -> "FitRecord":
        params = full_result.get_params_from_results()
        try:
            sigmas = full_result.get_sigmas_from_results()
        except Exception:
            sigmas = None

        return cls(
            model_key=model_key,
            params=params,
            sigmas=sigmas,
            renorm_factors=renorm_factors,
            full_result=full_result,
            fixed=fixed,
            is_complete=True,
        )


class AllFitResults:
    def __init__(self):
        self._records: dict[ModelKey, FitRecord] = {}

    def get(self, key: ModelKey) -> Optional[FitRecord]:
        return self._records.get(key)

    def set(self, record: FitRecord) -> None:
        self._records[record.model_key] = record

    def has(self, key: ModelKey) -> bool:
        return key in self._records

    def items(self):
        return self._records.items()


# ---- MMEXOFAST Fitters ---- #


class MMEXOFASTFitter():

    def __init__(
            self, files=None, fit_type=None, renormalize_errors=True,
            finite_source=False, limb_darkening_coeffs_gamma=None,
            limb_darkening_coeffs_u=None, mag_methods=None,
            datasets=None, coords=None,
            initial_results: Optional[dict] = None,
            priors=None, print_results=False, verbose=False,
            output_file=None, latex_file=None, log_file=None, emcee=True, emcee_settings=None, pool=None):
        # JCY: Can we reduce the number of separate inputs by grouping them as dicts? (actually probably want @dataclass classes)
        # fit_type = {'n_lenses': int, 'n_sources': int, 'finite_source': True/False, 'renormalize_errors': True, 'piE_grid': True}
        # fitter_kwargs = {[see below]}
        # outputs = {'log_file': str, 'latex_table': str, 'piE_grid_file': str}
        # emcee = {'emcee_settings': {}, 'pool':}
        #
        # Also, isn't emcee --> fitter_kwargs and fitter_kwargs --> model_kwargs?

        # Output
        self.verbose = verbose
        self.log_file = log_file
        self.latex_file = latex_file

        # setup datasets.
        if datasets is not None:
            self.datasets = datasets
        else:
            self.datasets = self._create_mulensdata_objects(files)

        self.n_loc = self._count_loc()

        self.fit_type = fit_type
        self.finite_source = finite_source

        self.renormalize_errors = renormalize_errors

        self.fitter_kwargs = {
            'coords': coords, 'mag_methods': mag_methods,
            'limb_darkening_coeffs_u': limb_darkening_coeffs_u,
            'limb_darkening_coeffs_gamma': limb_darkening_coeffs_gamma}
        # JCY: Can this just be passed as a dict to begin with, rather than separate options?
        #print(self.fitter_kwargs)

        self.emcee = emcee
        self.emcee_settings = emcee_settings
        self.pool = pool

        # initialize additional data versions
        self._residuals = None
        self._masked_datasets = None

        # initialize params
        # Should the grid_points be kept together? Are they part of results?
        # Should results --> fitted_models? or something?
        # Note: best_af_grid_point doesn't need to exist if we're only doing 1L1S fitting.
        self._best_ef_grid_point = None
        self._best_af_grid_point = None

        # Do we need to keep this around?
        self._anomaly_lc_params = None

        # Fit results
        # These should be deprecated
        self._pspl_static_results = None
        self._fspl_static_results = None
        self._pl_parallax_results = None

        self._binary_params = None

        self.result_store = AllFitResults()
        # existing stuff: datasets, best_ef_grid_point, etc.

        if initial_results is not None:
            self._load_initial_results(initial_results)

    def _create_mulensdata_objects(self, files):
        if isinstance(files, (str)):
            files = [files]

        datasets = []
        for filename in files:
            if not os.path.exists(filename):
                raise FileNotFoundError(
                    "Data file {0} does not exist".format(filename))

            kwargs = observatories.get_kwargs(filename)
            data = MulensModel.MulensData(file_name=filename, **kwargs)
            datasets.append(data)

        return datasets

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

    def _load_initial_results(self, initial_results: dict) -> None:
        for label, payload in initial_results.items():
            key = self._label_to_model_key(label)  # mapping label -> ModelKey

            record = FitRecord(
                model_key=key,
                params=payload["params"],
                sigmas=payload.get("sigmas"),
                renorm_factors=payload.get("renorm_factors"),
                full_result=None,
                fixed=payload.get("fixed", False),
                is_complete=False,
            )
            self.result_store.set(record)

    def _setup_results(self, prev_results):
        # Types of fits: (might want to make these properties of the class)

        # Types of parallax solutions
        single_loc_par_key_suffixes = ['u0+', 'u0-']
        two_loc_par_key_suffixes = ['u0++', 'u0--', 'u0+-', 'u0-+']

        # Point Lenses
        point_lens_keys = ['static PSPL']
        if self.finite_source:
            point_lens_keys.append('FSPL')

        if self.n_loc == 1:
            par_suffixes = single_loc_par_key_suffixes
        else:
            par_suffixes = two_loc_par_key_suffixes

        for suffix in par_suffixes:
            point_lens_keys.append('PL par' + suffix)

        results = {key: None for key in point_lens_keys}

        # Binary Lenses
        if self.fit_type == 'binary':
            binary_lens_keys = ['static 2L1S', ]
            motion_types = ['par', '2Dorb+par']

            if self.kepler:
                motion_types.append('kep+par')

            for motion in motion_types:
                for suffix in par_suffixes:
                    binary_lens_keys.append('2L1S {0} {1}'.format(motion, suffix))

            for key in binary_lens_keys:
                results = {key: None for key in binary_lens_keys}

        if prev_results is not None:
            for key, value in prev_results.items():
                results[key] = value

    def run_fit_if_needed(
            self,
            key: ModelKey,
            fit_func,
    ) -> FitRecord:
        """
        # Ensure there is an up-to-date FitRecord for `key`.
        # 
        # Parameters
        # ----------
        # key : ModelKey
        #     Which model to fit.
        # fit_func : callable
        #     Function that actually runs the fit:
        #     `fit_func(initial_params: Optional[dict]) -> MMEXOFASTFitResults`.
        # 
        # Returns
        # -------
        # FitRecord
        #     The current record for this model (existing or newly fitted).
        """
        record = self.result_store.get(key)

        # If we have a fixed or complete result, reuse it.
        if record is not None and (record.fixed or record.is_complete):
            return record

        # Use existing params as a starting point if present.
        initial_params = record.params if record is not None else None

        # Run the actual fit.
        full_result = fit_func(initial_params=initial_params)

        # Optional: derive current renorm factors from fitter state.
        renorm_factors = self._current_renorm_factors()

        # Wrap into a FitRecord (auto-populates params/sigmas via from_full_result).
        new_record = FitRecord.from_full_result(
            model_key=key,
            full_result=full_result,
            renorm_factors=renorm_factors,
            fixed=False,
        )
        self.result_store.set(new_record)
        return new_record

    def fit(self):
        """
        # Perform the fit according to the settings established when the MMEXOFASTFitter object was created.
        # 
        # :return: None
        """
        if self.fit_type is None:
            # Maybe "None" means initial mulens parameters were passed,
            # so we can go straight to a mmexofast_fit?
            raise ValueError(
                'You must set the fit_type when initializing the ' +
                'MMEXOFASTFitter(): fit_type=("point lens", "binary lens")')

        # ADD a condition to check whether point lens fits already exist...
        self.fit_point_lens()

        if self.fit_type == 'binary lens':
            self.best_af_grid_point = self.do_af_grid_search()
            if self.verbose:
                print('Best AF grid', self.best_af_grid_point)

            self.anomaly_lc_params = self.get_anomaly_lc_params()
            if self.verbose:
                print('Anomaly Params', self.anomaly_lc_params)

            if self.emcee:
                self.results = self.fit_anomaly()
                if self.verbose:
                    print('Results', self.results)

    def fit_point_lens(self):
        ### Next steps:
        # 1. Think about how to implement log file.
        # 2. Implement renormalize errors.

        results = {}

        if self.best_ef_grid_point is None:
            self.best_ef_grid_point = self.do_ef_grid_search()
            if self.verbose:
                print('Best EF grid point ', self.best_ef_grid_point)

        static_pspl_key = ModelKey(
            lens_type=LensType.POINT,
            source_type=SourceType.POINT,
            parallax_branch=ParallaxBranch.NONE,
            lens_orb_motion=LensOrbMotion.NONE,
        )
        self.run_fit_if_needed(static_pspl_key, fit_static_pspl)

        if 'static PSPL' in self.results.keys():
            print('is this really how you want this to work????') ###
            results['static PSPL'] = self.results['static PSPL']
            if self.verbose:
                print('static PSPL exists', self.results['static PSPL'].best)
        else:
            results['static PSPL'] = self.fit_initial_pspl_model()
            if self.verbose:
                print('Initial SFit', self.results['static PSPL'].best)

        if self.finite_source:

            if self.verbose:
                print('SFit FSPL', self.results['static FSPL'].best)

        pl_results = self.fit_pl_parallax_models()
        for key, value in pl_results.items():
            results[key] = value

        self.results = results

        if self.renormalize_errors:
            self.renormalize_errors_and_refit()


    def renormalize_errors_and_refit(self):
        """
        #Given the existing fits, take the best one and renormalize the errorbars of each dataset relative to that fit.
        #Then, re-optimize all the fits with the new errorbars.
        #:return:
        """
        pass

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
            for fit in self.pl_parallax_results:
                fits.append({'parameters': fit.get_params_from_results(),
                             'sigmas': fit.get_sigmas_from_results()})

            initializations['fits'] = fits
        else:
            raise NotImplementedError('initialize_exozippy only implemented for point lens fits')

        return initializations

    def make_ulens_table(self, table_type, models=None):
        """
        # Return a string consisting of a formatted table summarizing the results of the microlensing fits.
        # 
        # :param table_type:
        # models = *list*, Optional
        #     default is to make a table for all models
        # 
        # :return: *str*
        """
        def order_df(df):
            #raise NotImplementedError("THIS IS ALL VERY MESSY. I CAN'T FIGURE OUT HOW TO STRUCTURE THIS TO AVOID DUPLICATNG CODE.")

            def get_ordered_ulens_keys_for_repr(n_sources=1):
                """
                # define the default order of parameters
                """
                print(
                    'make_ulens_table.order_df(): This code was lifted verbatim from MM. It would be better to refactor it in MM and just use the function. Maybe as a Utils.')

                basic_keys = ['t_0', 'u_0', 't_E', 'rho', 't_star']
                additional_keys = [
                    'pi_E_N', 'pi_E_E', 't_0_par', 's', 'q', 'alpha',
                    'convergence_K', 'shear_G', 'ds_dt', 'dalpha_dt', 's_z',
                    'ds_z_dt', 't_0_kep',
                    'x_caustic_in', 'x_caustic_out', 't_caustic_in', 't_caustic_out',
                    'xi_period', 'xi_semimajor_axis', 'xi_inclination',
                    'xi_Omega_node', 'xi_argument_of_latitude_reference',
                    'xi_eccentricity', 'xi_omega_periapsis', 'q_source', 't_0_xi'
                ]

                ordered_keys = []
                if n_sources > 1:
                    for param_head in basic_keys:
                        if param_head == 't_E':
                            ordered_keys.append(param_head)
                        else:
                            for i in range(n_sources):
                                ordered_keys.append('{0}_{1}'.format(param_head, i + 1))

                else:
                    ordered_keys = basic_keys

                for key in additional_keys:
                    ordered_keys.append(key)

                # New for MMEXOFAST:
                ordered_keys = ['chi2', 'N_data'] + ordered_keys

                return ordered_keys

            def get_ordered_flux_keys_for_repr():
                flux_keys = []
                for i, dataset in enumerate(self.datasets):
                    if 'label' in dataset.plot_properties.keys():
                        obs = dataset.plot_properties['label'].split('-')[0]
                    else:
                        obs = i

                    if dataset.bandpass is not None:
                        band = dataset.bandpass
                    else:
                        band = 'mag'

                    flux_keys.append('{0}_S_{1}'.format(band, obs))
                    flux_keys.append('{0}_B_{1}'.format(band, obs))

                return flux_keys

            def get_ordered_keys_for_repr():
                ulens_keys = get_ordered_ulens_keys_for_repr()
                flux_keys = get_ordered_flux_keys_for_repr()
                return ulens_keys + flux_keys

            desired_order = get_ordered_keys_for_repr()

            order_map = {name: i for i, name in enumerate(desired_order)}

            df["sort_key"] = df["parameter_names"].map(order_map)
            df["orig_pos"] = range(len(df))

            # Anything not in desired_order gets a large sort key → goes to the end
            max_key = len(desired_order)
            df["sort_key"] = df["sort_key"].fillna(max_key)
            df = df.sort_values(["sort_key", "orig_pos"]).reset_index().drop(columns=["index", "sort_key", "orig_pos"])
            return df

        if table_type is None:
            table_type = 'ascii'

        if models is None:
            models = self.results.keys()

        results_table = None
        for name in models:
            new_column = self.results[name].format_results_as_df()
            new_column = new_column.rename(columns={'values': name, 'sigmas': 'sig [{0}]'.format(name)})

            if results_table is None:
                results_table = new_column
            else:
                results_table = results_table.merge(new_column, on="parameter_names", how="outer")

        results_table = order_df(results_table)
        if table_type == 'latex':
            def fmt(name):
                if name == 'chi2':
                    return '$\chi^2$'

                parts = name.split("_")
                if len(parts) == 1:
                    return f"${name}$"
                first = parts[0]
                rest = ", ".join(parts[1:])
                return f"${first}" + "_{" + rest + "}$"

            results_table["parameter_names"] = results_table["parameter_names"].apply(fmt)

            return results_table.to_latex(index=False)
        elif table_type == 'ascii':
            with pd.option_context("display.max_rows", None,
                                   "display.max_columns", None,
                                   "display.width", None, "display.float_format", "{:f}".format):
                return results_table.to_string(index=False)
        else:
            raise NotImplementedError(table_type + ' not implemented.')

    def do_ef_grid_search(self):
        """
        # Run a :py:class:`mmexofast.gridsearches.EventFinderGridSearch`
        # :return: *dict* of best EventFinder grid point.
        """
        ef_grid = mmexo.EventFinderGridSearch(datasets=self.datasets)
        ef_grid.run()
        return ef_grid.best

    def fit_initial_pspl_model(self, verbose=False):
        """
        # Estimate a starting point for the PSPL fitting from the EventFinder search (:py:attr:`best_ef_grid_point`)
        # and then optimize the parameters using :py:class:`mmexofast.fitters.SFitFitter`.
        # 
        # :param verbose: *bool* optional
        # :return: :py:class:`MMEXOFASTFit` object.
        """
        pspl_est_params = mmexo.estimate_params.get_PSPL_params(self.best_ef_grid_point, self.datasets)
        if self.verbose:
            print('Initial PSPL Estimate', pspl_est_params)

        fitter = mmexo.fitters.SFitFitter(initial_model_params=pspl_est_params, datasets=self.datasets)
        fitter.run()

        return FitResult(fitter)

    def fit_static_fspl_model(self):
        """
        # Use the results from the static PSPL fit (:py:attr:`pspl_static_results`) to initialize and optimize an FSPL
        # fit.
        # 
        # :return: :py:class:`MMEXOFASTFit` object.
        """
        init_params = self.results['static PSPL'].get_params_from_results()
        init_params['rho'] = 1.5 * init_params['u_0']
        fitter = mmexo.fitters.SFitFitter(
            initial_model_params=init_params, datasets=self.datasets, **self.fitter_kwargs)
        fitter.run()
        return FitResult(fitter)

    def fit_pl_parallax_models(self):
        """
        # Use the results from the static fit (either PSPL or FSPL according to the value of `finite_source`) to
        # initialize u0+ and u0- parallax fits.
        # 
        # :return: *list* of 2 :py:class:`MMEXOFASTFit` objects.
        """
        if self.finite_source:
            init_params = self.results['static FSPL'].get_params_from_results()
        else:
            init_params = self.results['static PSPL'].get_params_from_results()

        init_params['pi_E_N'] = 0.
        init_params['pi_E_E'] = 0

        results = {}
        for sign in [1, -1]:
            init_params['u_0'] *= sign
            if sign >= 0:
                key = 'PL parallax (+u_0)'
            else:
                key = 'PL parallax (-u_0)'

            fitter = mmexo.fitters.SFitFitter(
                initial_model_params=init_params, datasets=self.datasets, **self.fitter_kwargs)
            fitter.run()
            results[key] = FitResult(fitter)

        return results

    def fit_parallax_grid(self, grid=None, plot=False):
        if grid is None:
            grid = {'pi_E_E': (-1, 1, 0.05), 'pi_E_N': (-2., 2., 0.1)}

        raise NotImplementedError('Need to refactor for new architecture and u0+/-')
        init_params = self.get_params_from_results(self.pl_parallax_results)
        parameters_to_fit = list(init_params.keys())
        parameters_to_fit.remove('pi_E_E')
        parameters_to_fit.remove('pi_E_N')
        print(parameters_to_fit)

        pi_E_E = np.arange(grid['pi_E_E'][0], grid['pi_E_E'][1] + grid['pi_E_E'][2], grid['pi_E_E'][2])
        pi_E_N = np.arange(grid['pi_E_N'][0], grid['pi_E_N'][1] + grid['pi_E_N'][2], grid['pi_E_N'][2])
        chi2 = np.zeros((len(pi_E_E), len(pi_E_N)))
        for i, east in enumerate(pi_E_E):
            init_params['pi_E_E'] = east
            for j, north in enumerate(pi_E_N):
                init_params['pi_E_N'] = north
                fitter = mmexo.fitters.SFitFitter(
                    initial_model_params=init_params, parameters_to_fit=parameters_to_fit, datasets=self.datasets, **self.fitter_kwargs)
                fitter.run()
                print(fitter.best)
                chi2[i, j] = fitter.best['chi2']

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

    def do_af_grid_search(self):
        self.set_residuals(self.initial_pspl_params)
        af_grid = mmexo.AnomalyFinderGridSearch(residuals=self.residuals)
        # May need to update value of teff_min
        af_grid.run()
        return af_grid.best

    def get_anomaly_lc_params(self):
        estimator = mmexo.estimate_params.AnomalyPropertyEstimator(
            datasets=self.datasets, pspl_params=self.initial_pspl_params, af_results=self.best_af_grid_point)
        return estimator.get_anomaly_lc_parameters()

    def fit_anomaly(self):
        # So far, this only fits wide planet models in the GG97 limit.
        #print(self.anomaly_lc_params)
        wide_planet_fitter = mmexo.fitters.WidePlanetFitter(
            datasets=self.datasets, anomaly_lc_params=self.anomaly_lc_params,
            emcee_settings=self.emcee_settings, pool=self.pool)
        if self.verbose:
            wide_planet_fitter.estimate_initial_parameters()
            print('Initial 2L1S Wide Model', wide_planet_fitter.initial_model)
            print('mag methods', wide_planet_fitter.mag_methods)

        wide_planet_fitter.run()
        return wide_planet_fitter.best

    @property
    def residuals(self):
        return self._residuals

    @residuals.setter
    def residuals(self, value):
        self._residuals = value

    @property
    def masked_datasets(self):
        return self._masked_datasets

    @masked_datasets.setter
    def masked_datasets(self, value):
        self._masked_datasets = value

    @property
    def best_ef_grid_point(self):
        return self._best_ef_grid_point

    @best_ef_grid_point.setter
    def best_ef_grid_point(self, value):
        self._best_ef_grid_point = value

    @property
    def pspl_static_results(self):
        """
        # Results from fitting a static PSPL Model
        # :return: :py:class:`MMEXOFASTFit` object.
        """
        return self._pspl_static_results

    @pspl_static_results.setter
    def pspl_static_results(self, value):
        self._pspl_static_results = value

    @property
    def fspl_static_results(self):
        """
        # Results from fitting a static FSPL Model
        # :return: :py:class:`MMEXOFASTFit` object.
        """
        return self._fspl_static_results

    @fspl_static_results.setter
    def fspl_static_results(self, value):
        self._fspl_static_results = value

    @property
    def pl_parallax_results(self):
        """
        # Results from fitting a static FSPL Model
        # :return: *list* of :py:class:`MMEXOFASTFit` objects.
        """
        return self._pl_parallax_results

    @pl_parallax_results.setter
    def pl_parallax_results(self, value):
        self._pl_parallax_results = value


    @property
    def best_af_grid_point(self):
        return self._best_af_grid_point

    @best_af_grid_point.setter
    def best_af_grid_point(self, value):
        self._best_af_grid_point = value

    @property
    def anomaly_lc_params(self):
        return self._anomaly_lc_params

    @anomaly_lc_params.setter
    def anomaly_lc_params(self, value):
        self._anomaly_lc_params = value

    @property
    def binary_params(self):
        return self._binary_params

    @binary_params.setter
    def binary_params(self, value):
        self._binary_params = value

    @property
    def results(self):
        return self._results

    @results.setter
    def results(self, value):
        self._results = value
        
'''
