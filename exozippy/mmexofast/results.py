from typing import Dict, Any, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np

import MulensModel
import exozippy.mmexofast as mmexo


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
    model_key: mmexo.ModelKey

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
        model_key: mmexo.ModelKey,
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

    def __repr__(self) -> str:
        has_full = self.full_result is not None
        has_sigmas = self.sigmas is not None
        has_renorm = self.renorm_factors is not None
        n_params = len(self.params) if self.params is not None else 0

        # Display the actual fit set (params + sigmas) in a compact way
        # Truncate long dicts for readability if desired.
        def _short_dict(d: Optional[Dict[str, float]], max_items: int = 5) -> str:
            if not d:
                return "{}"
            items = list(d.items())
            if len(items) > max_items:
                head = ", ".join(f"{k}={v}" for k, v in items[:max_items])
                return "{" + head + ", ...}"
            return "{" + ", ".join(f"{k}={v}" for k, v in items) + "}"

        params_repr = _short_dict(self.params)
        sigmas_repr = _short_dict(self.sigmas)

        return (
            f"<FitRecord("
            f"lens={self.model_key.lens_type.value}, "
            f"source={self.model_key.source_type.value}, "
            f"parallax={self.model_key.parallax_branch.value}, "
            f"motion={self.model_key.lens_orb_motion.value}; "
            f"params={params_repr}, sigmas={sigmas_repr}; "
            f"full={has_full}, fixed={self.fixed}, complete={self.is_complete}, "
            f"renorm={has_renorm}, n_params={n_params}"
            f")>"
        )

    def chi2(self) -> Optional[float]:
        """
        Return chi^2 if available (from full_result.best['chi2']),
        otherwise None.
        """
        if self.full_result is None:
            return None
        best = self.full_result.best
        # your existing MMEXOFASTFitResults.best is a dict
        return best.get("chi2")


@dataclass
class GridSearchResult:
    """
    Results of a grid search, intended to be optionally persisted to disk.

    Attributes
    ----------
    name : str
        Short name of the grid search, e.g. 'EF', 'AF', 'PAR'.
    param_names : tuple[str, ...]
        Names of the grid parameters, e.g. ('s', 'q'), ('pi_E_N', 'pi_E_E').
    grid_points : np.ndarray
        Array of shape (N_points, n_params) containing grid coordinates.
    chi2 : np.ndarray
        Array of shape (N_points,) with chi^2 (or other scalar merit) values.
    metadata : dict
        Arbitrary extra info (datasets used, dates, config settings, etc.).
    best_index : int
        Index into grid_points / chi2 of the best point.
    """
    name: str
    param_names: tuple[str, ...]
    grid_points: np.ndarray
    chi2: np.ndarray
    metadata: Dict[str, Any]
    best_index: int


class AllFitResults:
    """
    Central registry for all fit results, keyed by mmexo.ModelKey.
    """
    def __init__(self):
        self._records: Dict[mmexo.ModelKey, FitRecord] = {}

    # --- internal helper ---
    def _normalize_key(self, key_or_label: str | mmexo.ModelKey) -> mmexo.ModelKey:
        if isinstance(key_or_label, mmexo.ModelKey):
            return key_or_label
        return mmexo.model_types.label_to_model_key(key_or_label)

    def get(self, key_or_label: str | mmexo.ModelKey) -> Optional[FitRecord]:
        key = self._normalize_key(key_or_label)
        return self._records.get(key)

    def set(self, record: FitRecord) -> None:
        self._records[record.model_key] = record

    def has(self, key_or_label: str | mmexo.ModelKey) -> bool:
        key = self._normalize_key(key_or_label)
        return key in self._records

    def keys(self, labels: bool = False):
        if labels:
            return [mmexo.model_types.model_key_to_label(k) for k in self._records.keys()]
        return list(self._records.keys())

    def items(self, labels: bool = False):
        if labels:
            return [(mmexo.model_types.model_key_to_label(k), r) for k, r in self._records.items()]
        return list(self._records.items())

    def __repr__(self) -> str:
        if not self._records:
            return "<AllFitResults: (empty)>"

        lines = ["<AllFitResults:"]
        for key, record in self._records.items():
            label = mmexo.model_types.model_key_to_label(key)
            lines.append(f"  {label!r}: {record}")
        lines.append(">")
        return "\n".join(lines)

    def iter_point_lens_records(self):
        """Yield (key, record) pairs for all point-lens models (PSPL/FSPL)."""
        for key, record in self._records.items():
            if key.lens_type == mmexo.LensType.POINT:
                yield key, record

    def select_best_static_pspl(self) -> Optional[FitRecord]:
        """
        Among all static PSPL models (point lens, point source, no parallax, no motion),
        return the one with lowest chi^2. Returns None if not found or no chi^2.
        """
        best_record = None
        best_chi2 = None

        for key, record in self._records.items():
            if not (
                    key.lens_type == mmexo.LensType.POINT
                    and key.source_type == mmexo.SourceType.POINT
                    and key.parallax_branch == mmexo.ParallaxBranch.NONE
                    and key.lens_orb_motion == mmexo.LensOrbMotion.NONE
            ):
                continue

            chi2 = record.chi2()
            if chi2 is None:
                continue

            if best_chi2 is None or chi2 < best_chi2:
                best_chi2 = chi2
                best_record = record

        return best_record

    def select_best_parallax_pspl(self) -> Optional[FitRecord]:
        """
        Among all PSPL parallax models (point lens, point source,
        parallax_branch != NONE, no orbital motion), return the one with
        lowest chi^2. Returns None if not found or no chi^2.
        """
        best_record = None
        best_chi2 = None

        for key, record in self._records.items():
            if not (
                    key.lens_type == mmexo.LensType.POINT
                    and key.source_type == mmexo.SourceType.POINT
                    and key.parallax_branch != mmexo.ParallaxBranch.NONE
                    and key.lens_orb_motion == mmexo.LensOrbMotion.NONE
            ):
                continue

            chi2 = record.chi2()
            if chi2 is None:
                continue

            if best_chi2 is None or chi2 < best_chi2:
                best_chi2 = chi2
                best_record = record

        return best_record
