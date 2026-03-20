"""Shared test utilities for MMEXOFAST unit tests."""

import numpy as np
from unittest.mock import Mock

import MulensModel
from sfit_minimizer.sfit_classes import SFitResults
from sfit_minimizer.mm_funcs import PointLensSFitFunction

from exozippy.mmexofast import results, fit_types


# ============================================================================
# Module-level defaults
# ============================================================================

DEFAULT_PARAMS = {
    't_0': 2456789.0,
    'u_0': 0.5,
    't_E': 20.0,
    'pi_E_N': 0.0,
    'pi_E_E': 0.0,
}

DEFAULT_SIGMAS = {
    't_0': 0.1,
    'u_0': 0.05,
    't_E': 0.5,
    'pi_E_E': 'nan',
    'pi_E_N': 'nan',
}

DEFAULT_FLUX_PARAMS = {
    'f1_S': 1.5,
    'f1_B': 0.3,
    'f2_S': 2.0,
    'f2_B': -0.5,
}

DEFAULT_FLUX_SIGMAS = {
    'f1_S': 0.1,
    'f1_B': 0.05,
    'f2_S': 0.15,
    'f2_B': 0.08,
}


# ============================================================================
# Factory functions
# ============================================================================

def create_mock_params_and_sigmas():
    """Create standard mock params and sigmas for testing.

    Returns
    -------
    tuple
        (params, sigmas) where params is a dict of model parameters
        and sigmas is a dict of parameter uncertainties.
    """
    return DEFAULT_PARAMS.copy(), DEFAULT_SIGMAS.copy()


def create_mock_fitter():
    """Create a mock fitter object for testing.

    Returns
    -------
    Mock
        Mock fitter with datasets, best, results, parameters_to_fit,
        and initial_model_params attributes.
    """
    t_0 = DEFAULT_PARAMS['t_0']
    u_0 = DEFAULT_PARAMS['u_0']
    t_E = DEFAULT_PARAMS['t_E']
    pi_E_N = DEFAULT_PARAMS['pi_E_N']
    pi_E_E = DEFAULT_PARAMS['pi_E_E']

    f1_S = DEFAULT_FLUX_PARAMS['f1_S']
    f1_B = DEFAULT_FLUX_PARAMS['f1_B']
    f2_S = DEFAULT_FLUX_PARAMS['f2_S']
    f2_B = DEFAULT_FLUX_PARAMS['f2_B']

    t_0_sigma = DEFAULT_SIGMAS['t_0']
    u_0_sigma = DEFAULT_SIGMAS['u_0']
    t_E_sigma = DEFAULT_SIGMAS['t_E']
    f1_S_sigma = DEFAULT_FLUX_SIGMAS['f1_S']
    f1_B_sigma = DEFAULT_FLUX_SIGMAS['f1_B']
    f2_S_sigma = DEFAULT_FLUX_SIGMAS['f2_S']
    f2_B_sigma = DEFAULT_FLUX_SIGMAS['f2_B']

    parameters_to_fit = ['t_0', 'u_0', 't_E']

    # Create MulensData objects
    dataset1 = MulensModel.MulensData(
        data_list=[[t_0, t_0 + 1.0], [110.0, 100.0], [1.0, 1.0]],
        phot_fmt='flux'
    )
    dataset1.plot_properties['label'] = 'n20200101.I.test.dataset_1.txt'

    dataset2 = MulensModel.MulensData(
        data_list=[[t_0 + 0.01, t_0 + 1.01], [210.0, 200.0], [2.0, 2.0]],
        phot_fmt='flux'
    )
    dataset2.plot_properties['label'] = 'n20200101.I.test.dataset_2.txt'

    mock_event = MulensModel.Event(
        datasets=[dataset1, dataset2],
        model=MulensModel.Model(
            DEFAULT_PARAMS.copy(),
            coords='18:00:00 -30:00:00'
        )
    )
    mock_event.fit_fluxes()
    chi2 = mock_event.get_chi2()

    mock_func = PointLensSFitFunction(mock_event, parameters_to_fit, estimate_fluxes=True)
    mock_func.update_all([t_0, u_0, t_E, f1_S, f1_B, f2_S, f2_B])

    fit_results = SFitResults(mock_func)
    fit_results.x = [t_0, u_0, t_E, f1_S, f1_B, f2_S, f2_B]
    fit_results.fun = chi2
    fit_results.sigmas = [
        t_0_sigma, u_0_sigma, t_E_sigma,
        f1_S_sigma, f1_B_sigma, f2_S_sigma, f2_B_sigma,
    ]
    fit_results.success = True
    fit_results.nit = 10

    mock_fitter = Mock()
    mock_fitter.datasets = [dataset1, dataset2]
    mock_fitter.best = {**DEFAULT_PARAMS, 'chi2': chi2}
    mock_fitter.results = fit_results
    mock_fitter.parameters_to_fit = parameters_to_fit
    mock_fitter.initial_model_params = DEFAULT_PARAMS.copy()

    return mock_fitter


def create_mock_fit_record(model_key=None, fixed=False, renorm_factors=None):
    """Create a FitRecord using a mock fitter for testing.

    Parameters
    ----------
    model_key : fit_types.FitKey, optional
        Model key for the fit record. Defaults to PSPL static.
    fixed : bool, optional
        Whether the record is fixed. Default is False.
    renorm_factors : dict, optional
        Renormalization factors. Default is None.

    Returns
    -------
    results.FitRecord
        FitRecord created from a mock fitter via from_full_result().
    """
    if model_key is None:
        model_key = fit_types.FitKey(
            lens_type=fit_types.LensType.POINT,
            source_type=fit_types.SourceType.POINT,
            parallax_branch=fit_types.ParallaxBranch.NONE,
            lens_orb_motion=fit_types.LensOrbMotion.NONE,
        )

    full_result = results.MMEXOFASTFitResults(create_mock_fitter())

    return results.FitRecord.from_full_result(
        model_key=model_key,
        full_result=full_result,
        renorm_factors=renorm_factors,
        fixed=fixed,
    )


def create_mock_grid_search_result(name='EF', n_points=10):
    """Create a mock GridSearchResult for testing.

    Parameters
    ----------
    name : str, optional
        Name of the grid search. Default is 'EF'.
    n_points : int, optional
        Number of grid points. Default is 10.

    Returns
    -------
    results.GridSearchResult
        GridSearchResult with synthetic data.
    """
    param_names = ('s', 'q')
    grid_points = np.random.rand(n_points, len(param_names))
    chi2 = np.random.rand(n_points) * 100.0
    best_index = int(np.argmin(chi2))

    return results.GridSearchResult(
        name=name,
        param_names=param_names,
        grid_points=grid_points,
        chi2=chi2,
        metadata={'test': True},
        best_index=best_index,
    )


def create_mock_figure():
    """Create a mock matplotlib figure for testing.

    Returns
    -------
    Mock
        Mock figure with savefig() and clf() methods.
    """
    fig = Mock()
    fig.savefig = Mock()
    fig.clf = Mock()
    return fig
