from typing import Dict, Any, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np
from collections.abc import MutableMapping
from abc import ABC, abstractmethod

import MulensModel
import exozippy.mmexofast as mmexo


# ============================================================================
# FitResults wrappers
# ============================================================================
class BaseFitResults(ABC):
    """
    Abstract base class for fit results wrappers.

    Defines the interface that all fit results classes must implement so that
    ``FitRecord`` can consume them interchangeably, regardless of the
    underlying fitter (e.g. SFit, emcee).

    Concrete subclasses must implement:
        - ``get_params_from_results()``
        - ``get_sigmas_from_results()``
        - ``format_results_as_df()``

    Parameters
    ----------
    fitter : object
        The fitter object whose results are being wrapped. Must expose
        ``best``, ``parameters_to_fit``, and ``datasets`` attributes.

    Attributes
    ----------
    fitter : object
        The wrapped fitter object.
    """

    def __init__(self, fitter):
        self.fitter = fitter

    # -----------------------------------------------------------------------
    # Abstract interface — must be implemented by subclasses
    # -----------------------------------------------------------------------

    @abstractmethod
    def get_params_from_results(self) -> dict:
        """
        Return the best-fit model parameters as a dict.

        Returns a dictionary mapping linear-space parameter names to their
        best-fit values, suitable for use as input to
        ``MulensModel.Model()``. Must not include ``'chi2'``.

        Returns
        -------
        dict
            Parameter name -> best-fit value.
        """

    @abstractmethod
    def get_sigmas_from_results(self) -> dict:
        """
        Return 1-sigma uncertainties as a dict.

        Returns a dictionary mapping parameter names to their 1-sigma
        uncertainties. For asymmetric uncertainties (e.g. from emcee),
        returns the mean of the upper and lower uncertainties.

        Returns
        -------
        dict
            Parameter name -> 1-sigma uncertainty.
        """

    @abstractmethod
    def format_results_as_df(self) -> pd.DataFrame:
        """
        Return fit results as a pandas DataFrame.

        The DataFrame must contain at minimum the columns
        ``'parameter_names'`` and ``'values'``. Sigma columns vary by
        subclass: ``MMEXOFASTFitResults`` produces ``'sigmas'``;
        ``EmceeFitResults`` produces ``'sigma_minus'`` and
        ``'sigma_plus'``.

        Returns
        -------
        pd.DataFrame
        """

    # -----------------------------------------------------------------------
    # Concrete shared properties
    # -----------------------------------------------------------------------

    @property
    def datasets(self):
        """list : Datasets from the fitter."""
        return self.fitter.datasets

    @property
    def best(self):
        """dict : Best-fit parameters including chi2."""
        return self.fitter.best

    @property
    def parameters_to_fit(self):
        """list of str : Names of the parameters sampled by the fitter."""
        return self.fitter.parameters_to_fit

    @property
    def all_model_parameters(self):
        """dict_keys : All parameter names in best, including fixed ones."""
        return self.fitter.best.keys()

    @property
    def chi2(self):
        """float or None : Best-fit chi2, or None if not available."""
        return self.fitter.best.get('chi2')


class MMEXOFASTFitResults(BaseFitResults):
    """
    Wrapper for results from an SFit minimizer run.

    Exposes the ``BaseFitResults`` interface so that ``FitRecord`` can
    consume SFit results identically to emcee results.

    Assumes ``fitter`` exposes ``.best``, ``.results``,
    ``.parameters_to_fit``, and ``.datasets``.

    Parameters
    ----------
    fitter : object
        The fitter object after the fit has completed. Must expose
        ``best``, ``results``, ``parameters_to_fit``, and ``datasets``.

    Notes
    -----
    Unlike ``EmceeFitResults``, this class produces a single ``'sigmas'``
    column in ``format_results_as_df()`` since SFit returns symmetric
    uncertainties.
    """

    def __init__(self, fitter):
        super().__init__(fitter)


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
                #if "label" in dataset.plot_properties.keys():
                #    obs = dataset.plot_properties["label"].split("-")[0]
                #else:
                #    obs = i
                #
                #if dataset.bandpass is not None:
                #    band = dataset.bandpass
                #else:
                #    band = "mag"
                obs, band = mmexo.observatories.get_telescope_band_from_filename(dataset.plot_properties['label'])

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

    @property
    def results(self):
        """object : Full SFit results object from the fitter."""
        return self.fitter.results


class EmceeFitResults(BaseFitResults):
    """
    Wrapper for results from a WidePlanetFitter emcee MCMC run.

    Computes post-burn-in percentiles from the sampler chain and exposes
    them via the ``BaseFitResults`` interface so that ``FitRecord`` can
    consume emcee results identically to SFit results.

    Parameters
    ----------
    fitter : WidePlanetFitter
        The fitter object after ``run()`` has completed. Must expose
        ``sampler.chain``, ``sampler.lnprobability``, ``emcee_settings``,
        ``best``, ``best_theta``, ``parameters_to_fit``, ``datasets``,
        ``_event``, ``initialize_event()``, and ``get_parameter_name()``.

    Attributes
    ----------
    fitter : WidePlanetFitter
        The wrapped fitter object.

    Notes
    -----
    ``sigma_minus = p50 - p16`` and ``sigma_plus = p84 - p50`` are both
    stored as positive numbers. The minus sign is a display concern only.
    """

    def __init__(self, fitter):
        super().__init__(fitter)
        self._percentiles = None

    # -----------------------------------------------------------------------
    # Percentiles (lazily computed and cached)
    # -----------------------------------------------------------------------

    @property
    def percentiles(self):
        """
        np.ndarray, shape (3, n_params) : 16th, 50th, and 84th percentiles
        of the post-burn-in chain, one column per parameter in
        ``parameters_to_fit``. Computed once and cached.
        """
        if self._percentiles is None:
            n_burn = self.fitter.emcee_settings['n_burn']
            n_dim  = self.fitter.emcee_settings['n_dim']
            samples = self.fitter.sampler.chain[:, n_burn:, :].reshape(
                (-1, n_dim))
            self._percentiles = np.percentile(samples, [16, 50, 84], axis=0)
        return self._percentiles

    # -----------------------------------------------------------------------
    # BaseFitResults interface
    # -----------------------------------------------------------------------

    def get_params_from_results(self) -> dict:
        """
        Return the max-likelihood parameters in linear space.

        Returns ``self.best`` excluding ``'chi2'``. Keys are linear-space
        parameter names (e.g. ``'rho'``, not ``'log_rho'``), suitable for
        use as input to ``MulensModel.Model()``.

        Returns
        -------
        dict
            Linear-space parameter name -> max-likelihood value.
        """
        return {k: v for k, v in self.best.items() if k != 'chi2'}

    def get_sigmas_from_results(self) -> dict:
        """
        Return mean 1-sigma uncertainties for each fitted parameter.

        Computes ``(sigma_minus + sigma_plus) / 2`` per parameter, where
        ``sigma_minus = p50 - p16`` and ``sigma_plus = p84 - p50``
        (both positive).

        Returns
        -------
        dict
            Parameter name (as in ``parameters_to_fit``) -> mean 1-sigma
            uncertainty.
        """
        p = self.percentiles
        sigmas = {}
        for i, param in enumerate(self.parameters_to_fit):
            sigma_minus = p[1, i] - p[0, i]
            sigma_plus  = p[2, i] - p[1, i]
            sigmas[param] = (sigma_minus + sigma_plus) / 2
        return sigmas

    def format_results_as_df(self) -> pd.DataFrame:
        """
        Build a summary DataFrame with fitted, fixed, and flux parameters.

        Sections (in order):

        1. **Fitted parameters**: 50th percentile values with asymmetric
           ``sigma_minus`` (p50 - p16, positive) and ``sigma_plus``
           (p84 - p50, positive).
        2. **Fixed parameters and chi2**: values from ``best``, NaN sigmas.
        3. **N_data**: total number of good data points, NaN sigmas.
        4. **Flux parameters**: source and blend magnitudes at the
           max-likelihood parameters, NaN sigmas.

        Returns
        -------
        pd.DataFrame
            Columns: ``'parameter_names'``, ``'values'``,
            ``'sigma_minus'``, ``'sigma_plus'``.
        """
        df_fitted = self._get_df_fitted_parameters()
        df_fixed  = self._get_df_fixed_parameters()
        df_flux   = self._get_df_flux_parameters()
        return pd.concat([df_fitted, df_fixed, df_flux], ignore_index=True)

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _get_df_fitted_parameters(self) -> pd.DataFrame:
        """
        Build the fitted parameters section of the DataFrame.

        Uses the 50th percentile as values. ``sigma_minus = p50 - p16``
        and ``sigma_plus = p84 - p50``, both stored as positive numbers.
        """
        p = self.percentiles
        return pd.DataFrame({
            'parameter_names': list(self.parameters_to_fit),
            'values':          list(p[1]),
            'sigma_minus':     list(p[1] - p[0]),
            'sigma_plus':      list(p[2] - p[1]),
        })

    def _get_df_fixed_parameters(self) -> pd.DataFrame:
        """
        Build the fixed parameters and N_data section of the DataFrame.

        Fixed parameters are those present in ``best`` but absent from the
        linear-mapped ``parameters_to_fit``. ``chi2`` is included here.
        ``N_data`` (total good data points across all datasets) is appended
        last. All sigma columns are NaN.
        """
        linear_params_to_fit = {
            self.fitter.get_parameter_name(p)
            for p in self.parameters_to_fit
        }
        fixed_parameters = [
            p for p in self.all_model_parameters
            if p not in linear_params_to_fit
        ]
        values = [self.best[p] for p in fixed_parameters]

        fixed_parameters.append('N_data')
        values.append(
            int(np.sum([np.sum(dataset.good) for dataset in self.datasets]))
        )

        n = len(fixed_parameters)
        return pd.DataFrame({
            'parameter_names': fixed_parameters,
            'values':          values,
            'sigma_minus':     [np.nan] * n,
            'sigma_plus':      [np.nan] * n,
        })

    def _get_df_flux_parameters(self) -> pd.DataFrame:
        """
        Build the flux parameters section of the DataFrame.

        Sets the fitter event to ``best_theta`` before computing fluxes to
        ensure magnitudes correspond to the max-likelihood parameters.
        Initializes the event first if it has not been set.

        Source and blend fluxes are converted to magnitudes via
        ``MulensModel.utils.Utils.get_mag_and_err_from_flux``. Negative
        fluxes are reported as ``'neg flux'``. All sigma columns are NaN
        since flux uncertainties are not available from the emcee chain.
        """
        if self.fitter._event is None:
            self.fitter.initialize_event()
        self.fitter.event = self.fitter.best_theta

        parameters = []
        values     = []

        source_fluxes = self.fitter._event.source_fluxes
        blend_fluxes  = self.fitter._event.blend_fluxes

        for i, dataset in enumerate(self.datasets):
            obs, band = mmexo.observatories.get_telescope_band_from_filename(
                dataset.plot_properties['label']
            )
            if len(source_fluxes[i]) == 1:
                parameters.append(f'{band}_S_{obs}')
            else:
                for j in range(len(source_fluxes[i])):
                    parameters.append(f'{band}_S{j}_{obs}')

            parameters.append(f'{band}_B_{obs}')

            for flux in list(source_fluxes[i]) + [blend_fluxes[i]]:
                flux_scalar = float(np.squeeze(flux))
                if flux_scalar > 0:
                    mag, _ = MulensModel.utils.Utils.get_mag_and_err_from_flux(
                        flux_scalar, 0.0
                    )
                else:
                    mag = 'neg flux'

                values.append(mag)

        n = len(parameters)
        print(parameters)
        print(values)
        print(source_fluxes, blend_fluxes)
        return pd.DataFrame({
            'parameter_names': parameters,
            'values':          values,
            'sigma_minus':     [np.nan] * n,
            'sigma_plus':      [np.nan] * n,
        })


# ============================================================================
# FitRecord and AllFitResults
# ============================================================================
@dataclass
class FitRecord:
    """
    Container for a fit result from MMEXOFAST.

    Stores model parameters, uncertainties, and associated fit metadata for a
    single model configuration (lens type, source type, parallax branch, etc.).
    Optionally retains the full fit result object for downstream analysis.

    Attributes
    ----------
    model_key : mmexo.FitKey
        Key identifying the model configuration (lens type, source type, etc.).
    params : dict
        Dictionary mapping parameter names to fitted values.
    sigmas : dict, optional
        Dictionary mapping parameter names to 1-sigma uncertainties.
        None if uncertainties were not computed.
    renorm_factors : dict, optional
        Dictionary of renormalization/systematics factors applied.
        None if no renormalization was needed.
    full_result : object, optional
        Complete fit results object from MMEXOFAST.
        None if only summary data is retained.
    fixed : bool
        Whether the fit was performed with fixed parameters.
    is_complete : bool
        Whether the fit completed successfully.

    """
    model_key: mmexo.FitKey
    params: dict
    sigmas: dict = None
    renorm_factors: dict = None
    full_result: object = None
    fixed: bool = False
    is_complete: bool = False

    @classmethod
    def from_full_result(cls, model_key, full_result, renorm_factors=None, fixed=False):
        """
        Construct a FitRecord from a full MMEXOFASTFitResults object.

        Parameters
        ----------
        model_key : mmexo.FitKey
            Key identifying the model configuration.
        full_result : MMEXOFASTFitResults
            Complete fit results from MMEXOFAST.
        renorm_factors : dict, optional
            Dictionary of renormalization factors. Default is None.
        fixed : bool, optional
            Whether the fit used fixed parameters. Default is False.

        Returns
        -------
        FitRecord
            New FitRecord instance populated from the full result.

        """
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

    def to_dataframe(self):
        """
        Export fit results as a pandas DataFrame.

        If ``full_result`` is available, delegates to its
        ``format_results_as_df()`` method. Otherwise returns a minimal
        DataFrame constructed from ``params`` and ``sigmas``.

        The column structure depends on the type of ``full_result``:

        - ``MMEXOFASTFitResults``: columns are ``'parameter_names'``,
          ``'values'``, ``'sigmas'``. Uncertainties are symmetric.
        - ``EmceeFitResults``: columns are ``'parameter_names'``,
          ``'values'``, ``'sigma_minus'``, ``'sigma_plus'``. Uncertainties
          are asymmetric. ``sigma_minus = p50 - p16`` and
          ``sigma_plus = p84 - p50``, both stored as positive numbers.
          Fixed parameters, ``N_data``, and flux parameters have
          ``NaN`` for both sigma columns.
        - Minimal fallback (no ``full_result``): columns are
          ``'parameter_names'``, ``'values'``, ``'sigmas'``.

        Returns
        -------
        pd.DataFrame
            DataFrame with ``'parameter_names'`` and ``'values'`` columns
            at minimum. Sigma column(s) vary by ``full_result`` type as
            described above.

        See Also
        --------
        MMEXOFASTFitResults.format_results_as_df
        EmceeFitResults.format_results_as_df
        """
        if self.full_result is not None:
            return self.full_result.format_results_as_df()
        return self._minimal_dataframe()

    def _minimal_dataframe(self):
        """
        Construct a minimal DataFrame from params and sigmas only.

        Used when full_result is unavailable. Returns basic parameter values
        and uncertainties without additional fit metadata (e.g., fluxes, N_data).

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: 'parameter_names', 'values', 'sigmas'.

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

    def __repr__(self):
        """
        Return a compact string representation of the FitRecord.

        Displays model configuration, parameter count, and fit status.
        Parameter dictionaries are truncated for readability if they exceed
        5 items.

        Returns
        -------
        str
            String representation of the FitRecord.

        """
        has_full = self.full_result is not None
        has_sigmas = self.sigmas is not None
        has_renorm = self.renorm_factors is not None
        n_params = len(self.params) if self.params is not None else 0

        def _short_dict(d, max_items=5):
            """Truncate dictionary representation for display."""
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

    def chi2(self):
        """
        Extract chi-squared value from the fit result.

        Returns the best-fit chi-squared statistic if full_result is available,
        otherwise returns None.

        Returns        -------
        float or None
            Chi-squared value, or None if full_result is unavailable.

        """
        if self.full_result is None:
            return None

        return self.full_result.chi2

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


class AllFitResults(MutableMapping):
    """
    Central registry for all fit results, keyed by mmexo.FitKey.
    """
    def __init__(self):
        self._records: Dict[mmexo.FitKey, FitRecord] = {}

    # --- Required MutableMapping methods ---
    def __getitem__(self, key_or_label: str | mmexo.FitKey) -> FitRecord:
        key = self._normalize_key(key_or_label)
        return self._records[key]

    def __setitem__(self, key_or_label: str | mmexo.FitKey, record: FitRecord) -> None:
        key = self._normalize_key(key_or_label)
        self._records[key] = record

    def __delitem__(self, key_or_label: str | mmexo.FitKey) -> None:
        key = self._normalize_key(key_or_label)
        del self._records[key]

    def __iter__(self):
        return iter(self._records)

    def __len__(self):
        return len(self._records)

    # --- internal helper ---
    def _normalize_key(self, key_or_label: str | mmexo.FitKey) -> mmexo.FitKey:
        if isinstance(key_or_label, mmexo.FitKey):
            return key_or_label
        return mmexo.fit_types.label_to_model_key(key_or_label)

    # --- custom convenience methods ---
    def get(self, key_or_label: str | mmexo.FitKey) -> Optional[FitRecord]:
        key = self._normalize_key(key_or_label)
        return self._records.get(key)

    def set(self, record: FitRecord) -> None:
        self._records[record.model_key] = record

    def has(self, key_or_label: str | mmexo.FitKey) -> bool:
        key = self._normalize_key(key_or_label)
        return key in self._records

    def keys(self, labels: bool = False):
        if labels:
            return [mmexo.fit_types.model_key_to_label(k) for k in self._records.keys()]
        return list(self._records.keys())

    def items(self, labels: bool = False):
        if labels:
            return [(mmexo.fit_types.model_key_to_label(k), r) for k, r in self._records.items()]
        return list(self._records.items())

    def __repr__(self) -> str:
        if not self._records:
            return "<AllFitResults: (empty)>"

        lines = ["<AllFitResults:"]
        for key, record in self._records.items():
            label = mmexo.fit_types.model_key_to_label(key)
            lines.append(f"  {label!r}: {record}")
        lines.append(">")
        return "\n".join(lines)

    def iter_point_lens_records(self):
        """Yield (key, record) pairs for all point-lens models (PSPL/FSPL)."""
        for key, record in self._records.items():
            if key.lens_type == mmexo.LensType.POINT:
                yield key, record

    #def select_best_static_pspl(self) -> Optional[FitRecord]:
    #    """
    #    Among all static PSPL models (point lens, point source, no parallax, no motion),
    #    return the one with lowest chi^2. Returns None if not found or no chi^2.
    #    """
    #    best_record = None
    #    best_chi2 = None
    #
    #    for key, record in self._records.items():
    #        if not (
    #                key.lens_type == mmexo.LensType.POINT
    #                and key.source_type == mmexo.SourceType.POINT
    #                and key.parallax_branch == mmexo.ParallaxBranch.NONE
    #                and key.lens_orb_motion == mmexo.LensOrbMotion.NONE
    #        ):
    #            continue
    #
    #        chi2 = record.chi2()
    #        if chi2 is None:
    #            continue
    #
    #        if best_chi2 is None or chi2 < best_chi2:
    #            best_chi2 = chi2
    #            best_record = record
    #
    #    return best_record

    #def select_best_parallax_pspl(self) -> Optional[FitRecord]:
    #    """
    #    Among all PSPL parallax models (point lens, point source,
    #    parallax_branch != NONE, no orbital motion), return the one with
    #    lowest chi^2. Returns None if not found or no chi^2.
    #    """
    #    best_record = None
    #    best_chi2 = None
    #
    #    for key, record in self._records.items():
    #        if not (
    #                key.lens_type == mmexo.LensType.POINT
    #                and key.source_type == mmexo.SourceType.POINT
    #                and key.parallax_branch != mmexo.ParallaxBranch.NONE
    #                and key.lens_orb_motion == mmexo.LensOrbMotion.NONE
    #        ):
    #            continue
    #
    #        chi2 = record.chi2()
    #        if chi2 is None:
    #            continue
    #
    #        if best_chi2 is None or chi2 < best_chi2:
    #            best_chi2 = chi2
    #            best_record = record
    #
    #    return best_record
