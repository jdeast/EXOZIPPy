import numpy as np
from abc import ABC, abstractmethod
import datetime
import matplotlib.pyplot as plt

import MulensModel
import sfit_minimizer
import exozippy.mmexofast as mmexo

# ---------------------------------------------------------------------
# EventFinder & AnomalyFinder Grid searches:
# ---------------------------------------------------------------------

class EventFinderGridSearch():
    """
    Based on Kim et al. 2018, AJ, 155, 76
    """

    def __init__(self,
                 datasets=None, t_eff_3=1, d_t_eff=1/3., t_eff_max=99.,
                 d_t_0=1/3, t_0_min=None, t_0_max=None, z_t_eff=5,
                 n_min=50):
        if datasets is None:
            raise ValueError('You must define the datasets!')
        elif isinstance(datasets, (MulensModel.MulensData)):
            datasets = [datasets]
        elif not isinstance(datasets, (list)):
            raise TypeError(
                'datasets must be *list* or *MulensData*! Not',
                type(datasets))

        self.datasets = datasets
        self.grid_params = {'t_eff_3': t_eff_3, 'd_t_eff': d_t_eff,
                            't_eff_max': t_eff_max, 'd_t_0': d_t_0}
        if t_0_min is None:
            self.grid_params['t_0_min'] = self._get_t_0_min()
        else:
            self.grid_params['t_0_min'] = t_0_min

        if t_0_max is None:
            self.grid_params['t_0_max'] = self._get_t_0_max()
        else:
            self.grid_params['t_0_max'] = t_0_max

        self._grid_t_eff = None
        self._grid_t_0 = None
        self.results = None
        self._best = None

        # parameters for trimming data
        self.z_t_eff = z_t_eff
        self.n_min = n_min

    # ---------------------------------------------------------------------
    # Setup functions:
    # ---------------------------------------------------------------------

    def _get_t_0_min(self):
        t_0_min = None
        for dataset in self.datasets:
            t_min = np.nanmin(dataset.time)
            if t_0_min is None:
                t_0_min = t_min
            else:
                if t_min < t_0_min:
                    t_0_min = t_min

        return t_0_min - self.grid_params['d_t_0']

    def _get_t_0_max(self):
        t_0_max = None
        for dataset in self.datasets:
            t_max = np.nanmax(dataset.time)
            if t_0_max is None:
                t_0_max = t_max
            else:
                if t_max > t_0_max:
                    t_0_max = t_max

        return t_0_max + self.grid_params['d_t_0']

    def _setup_grid(self):
        t_eff_factor = (1 + self.grid_params['d_t_eff'])
        t_eff = self.grid_params['t_eff_3'] / t_eff_factor**2
        self._grid_t_eff = []
        self._grid_t_0 = []
        while t_eff < self.grid_params['t_eff_max']:
            t_0 = self.grid_params['t_0_min']
            while t_0 < self.grid_params['t_0_max']:
                self._grid_t_0.append(t_0)
                self._grid_t_eff.append(t_eff)
                t_0 += self.grid_params['d_t_0'] * t_eff

            t_eff *= t_eff_factor

    # ---------------------------------------------------------------------
    # Core functions:
    # ---------------------------------------------------------------------

    def run(self, verbose=False):
        results = []
        for (t_0, t_eff) in zip(self.grid_t_0, self.grid_t_eff):
            chi2s = self.do_fits({'t_0': t_0, 't_eff': t_eff}, verbose=verbose)
            dchi2s = [chi2s['1'] - chi2s['flat'], chi2s['2'] - chi2s['flat']]
            if verbose:
                print('t_0, t_eff, chi2s:', t_0, t_eff, chi2s)

            results.append(dchi2s)

        self.results = np.array(results)

    def plot(self, fig=None):
        """
        Plot chi2 improvement for j=1 and j=2 grid searches.

        Parameters
        ----------
        fig : matplotlib.figure.Figure, optional
            Figure to plot on. If None, creates new figure.

        Returns
        -------
        fig : matplotlib.figure.Figure
        """

        if self.results is None:
            raise ValueError("Must run grid search before plotting. Call .run() first.")

        if fig is None:
            fig = plt.figure(figsize=(8, 4))

        if self.grid_params['t_0_min'] > 2460000.:
            delta = 2460000.
        elif self.grid_params['t_0_min'] > 2450000.:
            delta = 2450000.
        else:
            delta = 0.

        for j in [1, 2]:
            plt.subplot(1, 2, j)
            plt.title(f'j={j}')
            sorted_idx = np.argsort(self.results[:, j - 1])[::-1]  # smallest chi2 on top
            plt.scatter(
                self.grid_t_0[sorted_idx] - delta, self.grid_t_eff[sorted_idx],
                c=self.results[sorted_idx, j - 1],
                edgecolors='black', cmap='Set1')
            plt.colorbar(label='chi2 - chi2_flat')

            if self.best is not None:
                plt.scatter(
                    self.best['t_0'] - delta, self.best['t_eff'],
                    color='black', marker='x', s=100, zorder=5)

            plt.minorticks_on()
            plt.xlabel(f't_0 - {delta}')
            plt.ylabel('t_eff')
            plt.yscale('log')

        plt.tight_layout()
        return fig

    # ---------------------------------------------------------------------
    # Helper Functions:
    # ---------------------------------------------------------------------

    def get_trimmed_datasets(self, parameters, verbose=False):
        trimmed_datasets = []
        for dataset in self.datasets:
            # Restrict fitting to points t_0 +- z * t_eff with the
            # requirement N > 50.
            index = ((dataset.time >
                      (parameters['t_0'] - self.z_t_eff * parameters['t_eff'])) &
                     (dataset.time <
                      (parameters['t_0'] + self.z_t_eff * parameters['t_eff'])) &
                     dataset.good)
            # Minimum requirement for including a dataset
            if np.sum(index) >= self.n_min:
                trimmed_dataset = MulensModel.MulensData(
                    [dataset.time[index], dataset.flux[index],
                     dataset.err_flux[index]], phot_fmt='flux')
                trimmed_datasets.append(trimmed_dataset)

        if verbose:
            print('trimmed datasets (N, epochs_i):', len(trimmed_datasets),
                  [dataset.n_epochs for dataset in trimmed_datasets])

        return trimmed_datasets

    def get_flat_chi2(self, trimmed_datasets):
        flat_sfit = FlatSFitFunction(trimmed_datasets)
        flat_sfit.update_all(theta=flat_sfit.theta + flat_sfit.get_step())
        flat_chi2 = flat_sfit.chi2
        return flat_chi2

    def do_fits(self, parameters, verbose=False):
        chi2s = {'1': np.nan, '2': np.nan, 'flat': np.nan}

        trimmed_datasets = self.get_trimmed_datasets(
            parameters, verbose=verbose)

        # Only fit the window if there's enough data to do so.
        if len(trimmed_datasets) >= 1:
            chi2s['flat'] = self.get_flat_chi2(trimmed_datasets)

            for j in [1, 2]:
                parameters['j'] = j
                ef_sfit = EFSFitFunction(trimmed_datasets, parameters)
                ef_sfit.update_all(theta=ef_sfit.theta + ef_sfit.get_step())
                chi2s['{0}'.format(j)] = ef_sfit.chi2

        return chi2s

    # ---------------------------------------------------------------------
    # Properties:
    # ---------------------------------------------------------------------
    @property
    def grid_t_0(self):
        if self._grid_t_0 is None:
            self._setup_grid()

        return np.array(self._grid_t_0)

    @property
    def grid_t_eff(self):
        if self._grid_t_eff is None:
            self._setup_grid()

        return np.array(self._grid_t_eff)

    @property
    def best(self):
        if (self._best is None) & (self.results is not None):
            try:
                index_1 = np.nanargmin(self.results[:, 0])
                index_2 = np.nanargmin(self.results[:, 1])
                if self.results[index_1, 0] < self.results[index_2, 1]:
                    j = 1
                    index = index_1
                else:
                    j = 2
                    index = index_2

                self._best = {'t_0': self.grid_t_0[index],
                              't_eff': self.grid_t_eff[index],
                              'j': j,
                              'chi2': self.results[index, j-1]}
            except ValueError as e:
                print(np.array(self.results).shape, np.sum(np.isnan(self.results)), np.sum(np.isfinite(self.results)))
                print(self.grid_params)
                raise ValueError(e)

        return self._best


class FlatSFitFunction(sfit_minimizer.SFitFunction):

    def __init__(self, datasets):
        if isinstance(datasets, (MulensModel.MulensData)):
            self.datasets = [datasets]
        else:
            self.datasets = datasets

        self.n_params = len(self.datasets)
        self._theta = None
        self.data_len = None
        self.flatten_data() # destroys self.theta
        self.data_indices = self._set_data_indices()
        self.theta = np.zeros(self.n_params)

    def _set_data_indices(self):
        data_indices = np.cumsum(self.data_len)
        data_indices = np.hstack((0, data_indices))
        return data_indices

    def flatten_data(self):
        """ Concatenate good points for all datasets into a single array with
                columns: Date, flux, err.
                """
        self.data_len = []
        flattened_data = []
        for i, dataset in enumerate(self.datasets):
            data = [dataset.time[dataset.good], dataset.flux[dataset.good],
                    dataset.err_flux[dataset.good]]
            self.data_len.append(np.sum(dataset.good))
            if i == 0:
                flattened_data = np.array(data)
            else:
                flattened_data = np.hstack((flattened_data, data))

        flattened_data = flattened_data.transpose()
        sfit_minimizer.SFitFunction.__init__(self, data=flattened_data)

    def calc_model(self):
        model = None
        for i, dataset in enumerate(self.datasets):
            fs = self.theta[i]
            model_fluxes = fs * np.ones(self.data_len[i])
            if i == 0:
                model = np.array(model_fluxes[dataset.good])
            else:
                model = np.hstack(
                    (model, model_fluxes[dataset.good]))

        self.ymod = model

    def calc_df(self):
        """
        Calculate the derivatives of the fitting function and store as
        self.df.

        """
        dfunc = np.zeros((self.n_params, self.data_indices[-1]))
        for i, dataset in enumerate(self.datasets):
            ind_start = self.data_indices[i]
            ind_stop = self.data_indices[i+1]
            dfunc_df_blend = np.ones((1, np.sum(dataset.good)))
            dfunc[i, ind_start:ind_stop] = dfunc_df_blend

        self.df = dfunc


class EFSFitFunction(FlatSFitFunction):

    def __init__(self, datasets, parameters):
        self.parameters = parameters
        self._q = None
        self._magnification = None

        FlatSFitFunction.__init__(self, datasets=datasets)
        self.n_params = 2 * len(datasets)
        self.theta = np.array([np.array([1, 0])
                               for i in range(len(self.datasets))]).flatten()

    def calc_model(self):
        model = None
        for i, dataset in enumerate(self.datasets):
            fs = self.theta[2 * i]
            fb = self.theta[2 * i + 1]
            mag = self.magnification[
                  self.data_indices[i]:self.data_indices[i + 1]]
            model_fluxes = fs * mag + fb
            if i == 0:
                model = np.array(model_fluxes[dataset.good])
            else:
                model = np.hstack(
                    (model, model_fluxes[dataset.good]))

        self.ymod = model

    def calc_residuals(self):
        """Calculate expected values of the residuals"""
        sfit_minimizer.SFitFunction.calc_residuals(self)

    def calc_df(self):
        """
        Calculate the derivatives of the fitting function and store as
        self.df.

        """
        dfunc = np.zeros((self.n_params, self.data_indices[-1]))
        for i, dataset in enumerate(self.datasets):
            ind_start = self.data_indices[i]
            ind_stop = self.data_indices[i+1]
            dfunc_df_source = np.array(
                [self.magnification[ind_start:ind_stop][dataset.good]])
            dfunc[2 * i, ind_start:ind_stop] = dfunc_df_source
            dfunc_df_blend = np.ones((1, np.sum(dataset.good)))
            dfunc[2 * i + 1, ind_start:ind_stop] = dfunc_df_blend

        self.df = dfunc

    def _get_q(self, time):
        q_ = 1. + ((time - self.parameters['t_0']) /
                       self.parameters['t_eff']) ** 2
        return q_

    @property
    def q(self):
        if self._q is None:
            self._q = self._get_q(self.data[:, 0])

        return self._q

    def _get_magnification(self, q):
        if self.parameters['j'] == 1:
            magnification = 1. / np.sqrt(q)
        elif self.parameters['j'] == 2:
            magnification = 1. / np.sqrt(1. - (q / 2 + 1) ** (-2))
        else:
            raise ValueError('Invalid value for j.', self.parameters)

        return magnification

    @property
    def magnification(self):
        """
        The magnification

        Parameters : None

        Returns :
            magnification: *float* or *np.ndarray*
                The magnification for each data point.
        """
        self._magnification = self._get_magnification(self.q)

        return self._magnification

    def get_magnification(self, time):
        """
        The magnification

        Parameters :
            time : *float*, np.array
                times at which to calculation the magnification.

        Returns :
            magnification: *float* or *np.ndarray*
                The magnification for given time(s).
        """
        q = self._get_q(time)
        magnification = self._get_magnification(q)
        return magnification


class AnomalyFinderGridSearch(EventFinderGridSearch):
    """
    https://ui.adsabs.harvard.edu/abs/2021AJ....162..163Z/abstract
    """

    def __init__(self,
        residuals=None, t_eff_3=0.75, d_t_eff=1/3., t_eff_max=10.,
        d_t_0=1/6., z_t_eff=3, n_min=2, **kwargs):
        EventFinderGridSearch.__init__(
            self,
            datasets=residuals, t_eff_3=t_eff_3, d_t_eff=d_t_eff,
            t_eff_max=t_eff_max, d_t_0=d_t_0, z_t_eff=z_t_eff, n_min=n_min,
            **kwargs)
        #print('max flux AF input',
        #      [np.max(residuals.flux) for residuals in self.datasets])
        self._anomalies = None

    def run(self, verbose=False):
        results = []
        for (t_0, t_eff) in zip(self.grid_t_0, self.grid_t_eff):
            chi2s = self.do_fits({'t_0': t_0, 't_eff': t_eff}, verbose=verbose)
            if verbose:
                print(t_0, t_eff, chi2s)

            results.append([chi2s[key] for key in ['1', '2', 'flat', 'zero']])

        self.results = np.array(results)

    def get_zero_chi2(self, trimmed_datasets):
        chi2 = 0.
        for dataset in trimmed_datasets:
            chi2 += np.sum( (dataset.flux / dataset.err_flux)**2 )

        return chi2

    def check_successive(self, trimmed_datasets):
        times = None
        residuals = None
        for dataset in trimmed_datasets:
            if times is None:
                times = dataset.time
                residuals = dataset.flux / dataset.err_flux
            else:
                times = np.hstack((times, dataset.time))
                residuals = np.hstack((residuals,
                                       dataset.flux / dataset.err_flux))

        ind_sort = np.argsort(times)
        times = times[ind_sort]
        residuals = residuals[ind_sort]

        n_success = 0
        for i in range(times.shape[0]):
            if residuals[i] > 2:
                n_success += 1
            else:
                n_success = 0

            if n_success > 3:
                return True

        return False

    def do_fits(self, parameters, verbose=False):
        chi2s = {'1': np.nan, '2': np.nan, 'flat': np.nan, 'zero': np.nan}

        trimmed_datasets = self.get_trimmed_datasets(
            parameters, verbose=verbose)

        do_fit = False
        # Only fit the window if there's enough data to do so.
        if len(trimmed_datasets) >= 1:
            #print('max flux AF trimmed input',
            #      [np.max(residuals.flux) for residuals in trimmed_datasets])

            # Check for a minimum of 5 datapoints
            n_tot = np.sum(np.hstack(
                [dataset.good for dataset in trimmed_datasets]))
            #print('n_tot', n_tot)
            successive = self.check_successive(trimmed_datasets)
            if (n_tot > 5) and (successive):
                do_fit = True

        if do_fit:
            chi2s['zero'] = self.get_zero_chi2(trimmed_datasets)
            chi2s['flat'] = self.get_flat_chi2(trimmed_datasets)

            for j in [1, 2]:
                parameters['j'] = j
                ef_sfit = EFSFitFunction(trimmed_datasets, parameters)
                ef_sfit.update_all(theta=ef_sfit.theta + ef_sfit.get_step())
                chi2s['{0}'.format(j)] = ef_sfit.chi2

        return chi2s

    def get_anomalies(self):

        anomalies = None

        for j in [1, 2]:
            dchi2_zero = self.results[:, 3] - self.results[:, j-1]
            dchi2_flat = self.results[:, 2] - self.results[:, j-1]
            values = np.vstack(
                (self.grid_t_0, self.grid_t_eff,
                 j * np.ones(self.results.shape[0], dtype=int), self.results[:, j - 1],
                 dchi2_flat, dchi2_zero))

            if anomalies is None:
                anomalies = values
            else:
                anomalies = np.hstack((anomalies, values))

        return anomalies.transpose()

    def filter_anomalies(self, tol_zero=120., tol_flat=35, tol_zero_alt=75.):
        # Loop over j
        #    index_zero = dchi2_zero > tol_zero
        #    index_flat = (dchi2_zero > tol_zero_alt) & (dchi2_flat > tol_flat)
        #    index = index_zero | index_flat
        #
        #    values = np.vstack(
        #        (self.grid_t_0[index], self.grid_t_eff[index],
        #        j * np.ones(np.sum(index)).astype(int), self.results[index, j-1],
        #        dchi2_flat[index], dchi2_zero[index]))
        #
        raise NotImplementedError()

    @property
    def anomalies(self):
        if (self._anomalies is None) and (self.results is not None):
            self._anomalies = self.get_anomalies()

        return self._anomalies

    @property
    def best(self):
        if (self.results is not None) and (self._best is None):
            index = np.nanargmax(self.anomalies[:, 5])
            self._best = {'t_0': self.anomalies[index, 0],
                          't_eff': self.anomalies[index, 1],
                          'j': self.anomalies[index, 2],
                          'chi2': self.anomalies[index, 3],
                          'dchi2_zero':
                              (self.anomalies[index, 5] -
                              self.anomalies[index, 3]),
                          'dchi2_flat':
                              (self.anomalies[index, 4] -
                               self.anomalies[index, 3])
                          }

        return self._best


# ---------------------------------------------------------------------
# General, Rectangular Grid Searches
# ---------------------------------------------------------------------

class BaseRectGridSearch(ABC):
    """
    Abstract base class for rectangular grid searches over independent parameters.

    A rectangular grid search tests all combinations of parameter values where
    each parameter is varied independently (unlike EventFinderGridSearch where
    t_0 spacing depends on t_eff).

    Child classes must implement:
        - _setup_grid(): Define the parameter grid
        - _fit_grid_point(): Run fit for one grid point
    """

    def __init__(self, datasets, verbose=False):
        """
        Parameters:
            datasets: MulensData object(s) to fit
            verbose: *bool*
                If True, print progress information
        """
        if datasets is None:
            raise ValueError('You must define the datasets!')
        elif isinstance(datasets, MulensModel.MulensData):
            self.datasets = [datasets]
        elif not isinstance(datasets, list):
            raise TypeError(
                'datasets must be *list* or *MulensData*! Not',
                type(datasets))

        self.datasets = datasets
        self.verbose = verbose
        self.results = None
        self._best = None
        self._grid = None

    @abstractmethod
    def _setup_grid(self):
        """
        Define the parameter grid. Must set self._grid to a list of dicts,
        where each dict contains the parameters for one grid point.

        Example:
            self._grid = [
                {'pi_E_E': 0.1, 'pi_E_N': 0.2},
                {'pi_E_E': 0.1, 'pi_E_N': 0.3},
                ...
            ]
        """
        pass

    @abstractmethod
    def _fit_grid_point(self, grid_params):
        """
        Run fit for one grid point.

        Parameters:
            grid_params: *dict*
                Parameter values for this grid point

        Returns:
            result: *dict*
                Must contain at least 'chi2'. Can include additional keys
                like 'params', 'success', etc.
        """
        pass

    def run(self):
        """
        Execute the grid search by iterating over all grid points.
        """
        if self._grid is None:
            self._setup_grid()

        results = []
        n_total = len(self._grid)

        for i, grid_params in enumerate(self._grid):
            if self.verbose:
                print(f"Grid point {i + 1}/{n_total}: {grid_params}")

            result = self._fit_grid_point(grid_params)

            # Merge grid parameters with fit results
            full_result = {**grid_params, **result}
            results.append(full_result)

            if self.verbose:
                print(f"  chi2 = {result.get('chi2', 'N/A')}")

        self.results = results

    @property
    def grid(self):
        """The parameter grid as a list of dicts"""
        if self._grid is None:
            self._setup_grid()
        return self._grid

    @property
    def best(self):
        """
        Find the grid point with minimum chi2.

        Returns:
            best: *dict*
                The result dict for the best-fitting grid point
        """
        if self._best is None and self.results is not None:
            # Filter out any results without valid chi2
            valid_results = [r for r in self.results if
                             'chi2' in r and np.isfinite(r['chi2'])]

            if len(valid_results) == 0:
                raise ValueError("No valid results found (all chi2 are NaN or missing)")

            # Find minimum chi2
            self._best = min(valid_results, key=lambda x: x['chi2'])

        return self._best

    # Utility methods for grid generation

    def _linear_grid_1d(self, param_min, param_max, step):
        """
        Generate 1D linear grid.

        Parameters:
            param_min: *float*
                Minimum parameter value
            param_max: *float*
                Maximum parameter value
            step: *float*
                Step size

        Returns:
            grid: *np.ndarray*
                1D array of parameter values
        """
        n_steps = int(np.round((param_max - param_min) / step)) + 1

        # Use linspace for exact spacing without accumulation errors
        grid = np.linspace(param_min, param_max, n_steps)
        # Round to eliminate floating-point errors
        # Determine decimal places based on step size
        decimal_places = max(0, -int(np.floor(np.log10(abs(step)))) + 1)
        grid = np.round(grid, decimals=decimal_places)
        return grid


    def _log_grid_1d(self, param_min, param_max, n_steps):
        """
        Generate 1D logarithmic grid.

        Parameters:
            param_min: *float*
                Minimum parameter value (must be > 0)
            param_max: *float*
                Maximum parameter value
            n_steps: *int*
                Number of grid points

        Returns:
            grid: *np.ndarray*
                1D array of parameter values
        """
        if param_min <= 0 or param_max <= 0:
            raise ValueError("Log grid requires positive values")
        return np.logspace(np.log10(param_min), np.log10(param_max), n_steps)

    def _make_rect_grid(self, param_arrays, param_names):
        """
        Create rectangular grid from 1D parameter arrays.

        Parameters:
            param_arrays: *list of np.ndarray*
                1D arrays for each parameter
            param_names: *list of str*
                Names for each parameter

        Returns:
            grid: *list of dict*
                All combinations of parameters
        """
        # Use meshgrid to get all combinations
        meshes = np.meshgrid(*param_arrays, indexing='ij')

        # Flatten and create list of dicts
        grid = []
        for i in range(meshes[0].size):
            point = {}
            for name, mesh in zip(param_names, meshes):
                point[name] = mesh.flat[i]
            grid.append(point)

        return grid


class ParallaxGridSearch(BaseRectGridSearch):
    """
    Grid search over parallax parameters pi_E_E and pi_E_N.

    For each grid point, runs mmexo.fitters.SFitFitter with fixed parallax
    parameters and returns chi2 + fit results.
    """

    def __init__(self, datasets, static_params, grid_params=None,
                 pi_E_E_min=-0.7, pi_E_E_max=0.7, pi_E_E_step=0.05,
                 pi_E_N_min=-1.0, pi_E_N_max=1.0, pi_E_N_step=0.1,
                 fitter_kwargs=None, verbose=False):
        """
        Parameters:
            datasets: MulensData object(s) to fit
            static_params: *dict*
                Dictionary of fixed model parameters (e.g., 'XSPL static' params)
                Should NOT include pi_E_E or pi_E_N
            grid_params: *dict* or None
                If provided, specifies all grid parameters and takes precedence
                over individual parameters. Expected keys:
                    'pi_E_E_min', 'pi_E_E_max', 'pi_E_E_step',
                    'pi_E_N_min', 'pi_E_N_max', 'pi_E_N_step'
            pi_E_E_min, pi_E_E_max, pi_E_E_step: *float*
                Grid definition for pi_E_E (East component).
                Ignored if grid_params is provided.
            pi_E_N_min, pi_E_N_max, pi_E_N_step: *float*
                Grid definition for pi_E_N (North component).
                Ignored if grid_params is provided.
            fitter_kwargs: *dict* or None
                Additional keyword arguments to pass to SFitFitter
            verbose: *bool*
                If True, print progress information
        """
        super().__init__(datasets=datasets, verbose=verbose)

        self.static_params = static_params.copy()
        self.current_params = static_params.copy()  # Working copy

        # If grid_params dict is provided, use it; otherwise build from individual params
        if grid_params is not None:
            self.grid_params = grid_params.copy()
            # Validate required keys
            required_keys = ['pi_E_E_min', 'pi_E_E_max', 'pi_E_E_step',
                             'pi_E_N_min', 'pi_E_N_max', 'pi_E_N_step']
            missing_keys = [k for k in required_keys if k not in self.grid_params]
            if missing_keys:
                raise ValueError(f"grid_params missing required keys: {missing_keys}")
        else:
            self.grid_params = {
                'pi_E_E_min': pi_E_E_min,
                'pi_E_E_max': pi_E_E_max,
                'pi_E_E_step': pi_E_E_step,
                'pi_E_N_min': pi_E_N_min,
                'pi_E_N_max': pi_E_N_max,
                'pi_E_N_step': pi_E_N_step,
            }

        self.fitter_kwargs = fitter_kwargs if fitter_kwargs is not None else {}

    def _setup_grid(self):
        """Set up rectangular grid over pi_E_E and pi_E_N"""
        pi_E_E_array = self._linear_grid_1d(
            self.grid_params['pi_E_E_min'],
            self.grid_params['pi_E_E_max'],
            self.grid_params['pi_E_E_step']
        )

        pi_E_N_array = self._linear_grid_1d(
            self.grid_params['pi_E_N_min'],
            self.grid_params['pi_E_N_max'],
            self.grid_params['pi_E_N_step']
        )

        self._grid = self._make_rect_grid(
            [pi_E_E_array, pi_E_N_array],
            ['pi_E_E', 'pi_E_N']
        )

    def run(self):
        """
        Execute the grid search by iterating over all grid points in spiral order.

        Starts from the grid point closest to (0, 0) and spirals outward,
        using the best-fit parameters from the previous point as the starting
        point for each new fit.
        """
        if self._grid is None:
            self._setup_grid()

        # Reorder grid in spiral pattern starting from (0, 0)
        spiral_grid = self._spiral_order_grid()

        results = []
        n_total = len(spiral_grid)

        for i, grid_params in enumerate(spiral_grid):
            if self.verbose:
                print(f"Grid point {i + 1}/{n_total}: {grid_params}")

            result = self._fit_grid_point(grid_params)

            # Merge grid parameters with fit results
            full_result = {**grid_params, **result}
            results.append(full_result)

            if self.verbose:
                print(f"  chi2 = {result.get('chi2', 'N/A')}")

        self.results = results

    def _spiral_order_grid(self):
        """
        Reorder grid points in counter-clockwise rectangular spiral from (0, 0).

        Returns
        -------
        list of dict
            Grid points ordered in counter-clockwise spiral starting from
            the point closest to (0, 0)
        """
        # Organize grid into 2D array
        pi_E_E_vals = sorted(set(p['pi_E_E'] for p in self._grid))
        pi_E_N_vals = sorted(set(p['pi_E_N'] for p in self._grid))

        # Create mapping from (i, j) to grid point
        grid_2d = {}
        for point in self._grid:
            i = pi_E_E_vals.index(point['pi_E_E'])
            j = pi_E_N_vals.index(point['pi_E_N'])
            grid_2d[(i, j)] = point

        # Find starting point closest to (0, 0)
        min_dist = float('inf')
        start_i, start_j = 0, 0
        for (i, j), point in grid_2d.items():
            dist = np.sqrt(point['pi_E_E'] ** 2 + point['pi_E_N'] ** 2)
            if dist < min_dist:
                min_dist = dist
                start_i, start_j = i, j

        ni = len(pi_E_E_vals)
        nj = len(pi_E_N_vals)

        spiral_order = [(start_i, start_j)]
        visited = {(start_i, start_j)}

        # Spiral outward layer by layer (counter-clockwise)
        layer = 1
        while len(visited) < len(grid_2d):
            # Define bounds for this layer
            i_min = max(0, start_i - layer)
            i_max = min(ni - 1, start_i + layer)
            j_min = max(0, start_j - layer)
            j_max = min(nj - 1, start_j + layer)

            # Bottom-left corner, then up left edge
            for j in range(j_min, j_max + 1):
                i, j_idx = i_min, j
                if (i, j_idx) not in visited and (i, j_idx) in grid_2d:
                    spiral_order.append((i, j_idx))
                    visited.add((i, j_idx))

            # Top edge (left to right), skip first point (already added)
            for i in range(i_min + 1, i_max + 1):
                j_idx = j_max
                if (i, j_idx) not in visited and (i, j_idx) in grid_2d:
                    spiral_order.append((i, j_idx))
                    visited.add((i, j_idx))

            # Right edge (top to bottom), skip first point
            for j in range(j_max - 1, j_min - 1, -1):
                i = i_max
                if (i, j) not in visited and (i, j) in grid_2d:
                    spiral_order.append((i, j))
                    visited.add((i, j))

            # Bottom edge (right to left), skip corners
            for i in range(i_max - 1, i_min, -1):
                j_idx = j_min
                if (i, j_idx) not in visited and (i, j_idx) in grid_2d:
                    spiral_order.append((i, j_idx))
                    visited.add((i, j_idx))

            layer += 1

        return [grid_2d[idx] for idx in spiral_order]

    def _fit_grid_point(self, grid_params):
        """
        Run SFitFitter for one grid point.

        Uses self.current_params as the starting point, which gets updated
        with the best-fit from each successful fit.

        Parameters
        ----------
        grid_params : dict
            Contains 'pi_E_E' and 'pi_E_N' for this grid point

        Returns
        -------
        dict
            Contains 'chi2', 'params' (best-fit parameters),
            'success', and any other fitter output
        """
        parameters_to_fit = list(self.static_params.keys())
        # Combine current params with this grid point's parallax values
        model_params = self.current_params.copy()
        model_params['pi_E_E'] = grid_params['pi_E_E']
        model_params['pi_E_N'] = grid_params['pi_E_N']

        # Run fitter
        try:
            fitter = mmexo.fitters.SFitFitter(
                initial_model_params=model_params,
                datasets=self.datasets,
                parameters_to_fit=parameters_to_fit,
                **self.fitter_kwargs
            )
            fitter.run()

            params = {key: value for key, value in fitter.best.items()}
            params.pop("chi2", None)
            result = {
                'chi2': fitter.best['chi2'],
                'params': params,
                'success': fitter.results.success,
            }

            # Update current_params with best fit for next iteration
            if result['success'] and result['params'] is not None:
                self.current_params.update(result['params'])

        except Exception as e:
            if self.verbose:
                print(f"  Fit failed: {e}")

            result = {
                'chi2': np.nan,
                'params': None,
                'success': False,
                'error': str(e)
            }

        return result

    def save_results(self, filepath, parallax_branch=None):
        """
        Save grid search results to ASCII file.

        Parameters
        ----------
        filepath : str or Path
            Full path where results should be saved
        parallax_branch : str or None, optional
            Optional parallax branch identifier for metadata

        Raises
        ------
        ValueError
            If no results exist (grid search hasn't been run yet)
        """
        if self.results is None:
            raise ValueError("No results to save. Run grid search first.")

        # Create a temporary fitter to get diagnostic string for metadata
        temp_fitter = mmexo.fitters.SFitFitter(
            initial_model_params=self.static_params,
            datasets=self.datasets,
            **self.fitter_kwargs
        )

        with open(filepath, 'w') as f:
            # Write metadata header
            f.write("# ParallaxGridSearch Results\n")
            f.write(f"# Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            if parallax_branch is not None:
                f.write(f"# Parallax Branch: {parallax_branch}\n")
            f.write(f"# Grid: pi_E_E=[{self.grid_params['pi_E_E_min']}, "
                    f"{self.grid_params['pi_E_E_max']}] step={self.grid_params['pi_E_E_step']}, "
                    f"pi_E_N=[{self.grid_params['pi_E_N_min']}, "
                    f"{self.grid_params['pi_E_N_max']}] step={self.grid_params['pi_E_N_step']}\n")
            f.write("#\n")
            f.write("# Reference Model:\n")

            # Write diagnostic string with # prefix on each line
            diagnostic_str = temp_fitter.get_diagnostic_str()
            for line in diagnostic_str.split('\n'):
                f.write(f"# {line}\n")

            f.write("#\n")

            # Determine column names from first result
            if len(self.results) > 0:
                # Start with grid params, then chi2, success, then fit params
                first_result = self.results[0]

                # Grid parameters
                grid_param_names = ['pi_E_E', 'pi_E_N']

                # Standard columns
                standard_cols = ['chi2', 'success']

                # Best-fit parameter names (if they exist)
                fit_param_names = []
                if first_result.get('params') is not None:
                    fit_param_names = list(first_result['params'].keys())

                all_column_names = grid_param_names + standard_cols + fit_param_names

                # Write column header
                f.write('# ' + '  '.join(all_column_names) + '\n')

                # Write data rows
                for result in self.results:
                    row = []

                    # Grid params
                    for name in grid_param_names:
                        row.append(str(result[name]))

                    # Standard columns
                    row.append(str(result['chi2']))
                    row.append(str(result['success']))

                    # Fit params
                    if result.get('params') is not None:
                        for name in fit_param_names:
                            row.append(str(result['params'][name]))
                    else:
                        # Fill with NaN if fit failed
                        row.extend(['nan'] * len(fit_param_names))

                    f.write('  '.join(row) + '\n')

    def plot_grid_points(self, ax=None, cmap='Set1', min_chi2=None):
        """
        Plot grid search results as colored points.

        Parameters
        ----------
        ax : matplotlib.axes.Axes or None, optional
            Axes to plot on. If None, uses current axes.
        cmap : str, optional
            Colormap name for chi2 values. Default is 'viridis_r'.
        min_chi2 : float or None, optional
            Minimum chi2 value for calculating sigma. If None, uses the
            minimum chi2 from this grid's results.

        Returns
        -------
        matplotlib.collections.PathCollection
            The scatter plot object (can be used for colorbar)

        Raises
        ------
        ValueError
            If no results exist (grid search hasn't been run yet)
        """
        if self.results is None:
            raise ValueError("No results to plot. Run grid search first.")

        if ax is None:
            ax = plt.gca()

        # Extract data for plotting
        pi_E_E = [r['pi_E_E'] for r in self.results]
        pi_E_N = [r['pi_E_N'] for r in self.results]
        chi2 = [r['chi2'] for r in self.results]

        # Calculate sigma (delta chi2 from minimum)
        if min_chi2 is None:
            min_chi2 = min(chi2)
        sigma = [np.sqrt(c - min_chi2) for c in chi2]

        # Create scatter plot
        scatter = ax.scatter(pi_E_E, pi_E_N, c=sigma, cmap=cmap, vmin=0, vmax=9, s=50)

        return scatter


class BinaryGridSearch(BaseRectGridSearch):
    """
    Grid search over binary lens parameters s (separation), q (mass ratio),
    and alpha (source trajectory angle).

    For each grid point, runs mmexo.fitters.SFitFitter with fixed binary
    parameters and returns chi2 + fit results.
    """

    def __init__(self, datasets, static_params, grid_params=None,
                 s_min=0.5, s_max=2.0, s_n=10, s_log=True,
                 q_min=0.001, q_max=1.0, q_n=10, q_log=True,
                 alpha_min=0.0, alpha_max=360.0, alpha_step=30.0,
                 fitter_kwargs=None, verbose=False):
        """
        Parameters:
            datasets: MulensData object(s) to fit
            static_params: *dict*
                Dictionary of fixed model parameters
                Should NOT include s, q, or alpha
            grid_params: *dict* or None
                If provided, specifies all grid parameters and takes precedence
                over individual parameters. Expected keys:
                    's_min', 's_max', 's_n', 's_log',
                    'q_min', 'q_max', 'q_n', 'q_log',
                    'alpha_min', 'alpha_max', 'alpha_step'
            s_min, s_max: *float*
                Min/max values for separation. Ignored if grid_params provided.
            s_n: *int*
                Number of grid points for s (used if s_log=True).
                Ignored if grid_params provided.
            s_log: *bool*
                If True, use logarithmic spacing for s.
                Ignored if grid_params provided.
            q_min, q_max: *float*
                Min/max values for mass ratio. Ignored if grid_params provided.
            q_n: *int*
                Number of grid points for q (used if q_log=True).
                Ignored if grid_params provided.
            q_log: *bool*
                If True, use logarithmic spacing for q.
                Ignored if grid_params provided.
            alpha_min, alpha_max: *float*
                Min/max values for source trajectory angle (degrees).
                Ignored if grid_params provided.
            alpha_step: *float*
                Step size for alpha (degrees). Alpha always uses linear spacing.
                Ignored if grid_params provided.
            fitter_kwargs: *dict* or None
                Additional keyword arguments to pass to SFitFitter
            verbose: *bool*
                If True, print progress information
        """
        super().__init__(datasets=datasets, verbose=verbose)

        self.static_params = static_params.copy()

        # If grid_params dict is provided, use it; otherwise build from individual params
        if grid_params is not None:
            self.grid_params = grid_params.copy()
            # Validate required keys
            required_keys = ['s_min', 's_max', 's_n', 's_log',
                             'q_min', 'q_max', 'q_n', 'q_log',
                             'alpha_min', 'alpha_max', 'alpha_step']
            missing_keys = [k for k in required_keys if k not in self.grid_params]
            if missing_keys:
                raise ValueError(f"grid_params missing required keys: {missing_keys}")
        else:
            self.grid_params = {
                's_min': s_min,
                's_max': s_max,
                's_n': s_n,
                's_log': s_log,
                'q_min': q_min,
                'q_max': q_max,
                'q_n': q_n,
                'q_log': q_log,
                'alpha_min': alpha_min,
                'alpha_max': alpha_max,
                'alpha_step': alpha_step,
            }

        self.fitter_kwargs = fitter_kwargs if fitter_kwargs is not None else {}

    def _setup_grid(self):
        """Set up rectangular grid over s, q, and alpha"""
        # Create s grid (log or linear)
        if self.grid_params['s_log']:
            s_array = self._log_grid_1d(
                self.grid_params['s_min'],
                self.grid_params['s_max'],
                self.grid_params['s_n']
            )
        else:
            # For linear spacing, would need s_step parameter
            raise NotImplementedError(
                "Linear spacing for s not implemented. Use s_log=True.")

        # Create q grid (log or linear)
        if self.grid_params['q_log']:
            q_array = self._log_grid_1d(
                self.grid_params['q_min'],
                self.grid_params['q_max'],
                self.grid_params['q_n']
            )
        else:
            raise NotImplementedError(
                "Linear spacing for q not implemented. Use q_log=True.")

        # Create alpha grid (always linear)
        alpha_array = self._linear_grid_1d(
            self.grid_params['alpha_min'],
            self.grid_params['alpha_max'],
            self.grid_params['alpha_step']
        )

        self._grid = self._make_rect_grid(
            [s_array, q_array, alpha_array],
            ['s', 'q', 'alpha']
        )

        if self.verbose:
            print(f"Grid setup: {len(s_array)} s × {len(q_array)} q × "
                  f"{len(alpha_array)} alpha = {len(self._grid)} total points")

    def _fit_grid_point(self, grid_params):
        """
        Run SFitFitter for one grid point.

        Parameters:
            grid_params: *dict*
                Contains 's', 'q', and 'alpha' for this grid point

        Returns:
            result: *dict*
                Contains 'chi2', 'params' (best-fit parameters),
                'success', and any other fitter output
        """

        # Combine static params with this grid point's binary values
        model_params = self.static_params.copy()
        model_params['s'] = grid_params['s']
        model_params['q'] = grid_params['q']
        model_params['alpha'] = grid_params['alpha']

        # Run fitter
        try:
            fitter = mmexo.fitters.SFitFitter(
                initial_model_params=model_params,
                datasets=self.datasets,
                **self.fitter_kwargs
            )
            fitter.run()

            result = {
                'chi2': fitter.chi2,
                'params': fitter.best_fit_params,
                'success': fitter.converged if hasattr(fitter, 'success') else True,
            }

        except Exception as e:
            if self.verbose:
                print(f"  Fit failed: {e}")
            result = {
                'chi2': np.nan,
                'params': None,
                'success': False,
                'error': str(e)
            }

        return result
