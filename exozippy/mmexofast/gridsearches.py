import numpy as np
from abc import ABC, abstractmethod
import datetime
import matplotlib.pyplot as plt
from itertools import product
from scipy.ndimage import minimum_filter, label

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
    """Abstract base class for rectangular grid searches.

    Parameters
    ----------
    grid_params : dict or None
        Grid specification: {'param': [min, max, step]}.
        Prefix 'log_' on param name for log10-spaced parameters,
        where min, max, and step are all in log10 space.
    datasets : list or None
        Data to fit. Required by child classes.
    evaluation_order : str
        'standard' or 'outward'. Default 'standard'.
    start_point : dict or None
        Starting point for 'outward' evaluation. Dict of param values.
    use_nearest_neighbor_init : bool
        Initialize each fit from nearest successful fit. Default True.
    max_refinements : int
        Maximum refinement iterations per minimum. Default 5.
    max_expansions : int
        Maximum edge expansion iterations. Default 5.
    verbose : bool
        Print progress. Default False.
    """

    def __init__(self, grid_params=None, datasets=None,
                 evaluation_order='standard', start_point=None,
                 use_nearest_neighbor_init=True,
                 max_refinements=5, max_expansions=5,
                 verbose=False):
        self.grid_params = grid_params
        self.datasets = datasets
        self.evaluation_order = evaluation_order
        self.start_point = start_point
        self.use_nearest_neighbor_init = use_nearest_neighbor_init
        self.max_refinements = max_refinements
        self.max_expansions = max_expansions
        self.verbose = verbose

        self.results_history = None
        self.minima = None
        self._point_cache = {}

    @abstractmethod
    def _fit_grid_point(self, grid_params):
        """Fit model at one grid point.

        Parameters
        ----------
        grid_params : dict
            Parameter values for this grid point.

        Returns
        -------
        dict
            Must contain:
            - 'chi2' : float
            - 'success' : bool
            - 'params' : dict of fitted parameter values
            May contain additional keys.
        """
        pass

    # ----------------------------------------------------------------
    # Grid construction
    # ----------------------------------------------------------------

    def _is_log_param(self, name):
        """Check if parameter is log-spaced.

        Parameters
        ----------
        name : str

        Returns
        -------
        bool
        """
        return name.startswith('log_')

    def _build_param_array(self, name, spec):
        """Build 1D array of parameter values for one parameter.

        For log params, spec values are in log10 space;
        returned array contains actual (linear) values.

        Parameters
        ----------
        name : str
        spec : list of float
            [min, max, step]

        Returns
        -------
        np.ndarray
        """
        min_val, max_val, step = spec
        n = int(np.round((max_val - min_val) / step)) + 1
        values = np.linspace(min_val, max_val, n)
        if self._is_log_param(name):
            return 10.0 ** values
        return values

    def _build_all_param_arrays(self):
        """Build 1D arrays for all parameters.

        Returns
        -------
        dict
            {param_name: np.ndarray of actual parameter values}
        """
        return {name: self._build_param_array(name, spec)
                for name, spec in self.grid_params.items()}

    def _build_grid_metadata(self):
        """Build grid metadata dict.

        Returns
        -------
        dict
            Keys: 'param_names', 'param_arrays', 'grid_shape'
        """
        param_names = list(self.grid_params.keys())
        param_arrays = self._build_all_param_arrays()
        grid_shape = tuple(len(param_arrays[name]) for name in param_names)
        return {
            'param_names': param_names,
            'param_arrays': param_arrays,
            'grid_shape': grid_shape,
            'steps': {name: spec[2] for name, spec in self.grid_params.items()}
        }

    def _build_empty_arrays(self, grid_shape):
        """Build empty chi2 and result arrays.

        Parameters
        ----------
        grid_shape : tuple of int

        Returns
        -------
        chi2_grid : np.ndarray
            Filled with NaN, shape grid_shape.
        result_grid : np.ndarray
            Filled with None, shape grid_shape, dtype object.
        """
        chi2_grid = np.full(grid_shape, np.nan)
        result_grid = np.empty(grid_shape, dtype=object)
        return chi2_grid, result_grid

    # ----------------------------------------------------------------
    # Grid indexing
    # ----------------------------------------------------------------

    def _value_to_index(self, value, param_array):
        """Find nearest grid index for a parameter value.

        Parameters
        ----------
        value : float
            Actual parameter value.
        param_array : np.ndarray
            1D array of actual parameter values.

        Returns
        -------
        int
        """
        return int(np.argmin(np.abs(param_array - value)))

    def _point_to_indices(self, point, metadata):
        """Convert parameter dict to grid index tuple.

        Parameters
        ----------
        point : dict
            Parameter name -> actual value.
        metadata : dict
            Grid metadata from _build_grid_metadata().

        Returns
        -------
        tuple of int
        """
        return tuple(
            self._value_to_index(point[name], metadata['param_arrays'][name])
            for name in metadata['param_names']
        )

    def _indices_to_point(self, indices, metadata):
        """Convert grid index tuple to parameter dict.

        Parameters
        ----------
        indices : tuple of int
        metadata : dict

        Returns
        -------
        dict
            Parameter name -> actual value.
        """
        return {
            name: metadata['param_arrays'][name][idx]
            for name, idx in zip(metadata['param_names'], indices)
        }

    def _shift_indices(self, indices, offsets):
        """Shift grid indices by offsets after array expansion.

        Parameters
        ----------
        indices : tuple of int
        offsets : tuple of int

        Returns
        -------
        tuple of int
        """
        return tuple(indices[i] + offsets[i] for i in range(len(indices)))

    # ----------------------------------------------------------------
    # Evaluation order
    # ----------------------------------------------------------------

    def _standard_order(self, grid_shape):
        """Generate grid indices in row-major order.

        Parameters
        ----------
        grid_shape : tuple of int

        Returns
        -------
        list of tuple of int
        """
        return list(np.ndindex(*grid_shape))

    def _outward_order(self, grid_shape, start_indices):
        """Generate grid indices via BFS expansion from start point.

        Parameters
        ----------
        grid_shape : tuple of int
        start_indices : tuple of int

        Returns
        -------
        list of tuple of int
        """
        from collections import deque
        from itertools import product

        visited = {start_indices}
        order = []
        queue = deque([start_indices])

        while queue:
            idx = queue.popleft()
            order.append(idx)

            for delta in product([-1, 0, 1], repeat=len(grid_shape)):
                if all(d == 0 for d in delta):
                    continue
                neighbor = tuple(idx[i] + delta[i] for i in range(len(grid_shape)))
                if (neighbor not in visited and
                        all(0 <= neighbor[i] < grid_shape[i]
                            for i in range(len(grid_shape)))):
                    visited.add(neighbor)
                    queue.append(neighbor)

        return order

    def _get_evaluation_order(self, metadata, evaluation_order, start_point):
        """Get ordered list of grid indices to evaluate.

        Parameters
        ----------
        metadata : dict
        evaluation_order : str
            'standard' or 'outward'
        start_point : dict or None

        Returns
        -------
        list of tuple of int
        """
        grid_shape = metadata['grid_shape']

        if evaluation_order == 'standard':
            return self._standard_order(grid_shape)

        if start_point is not None:
            start_indices = self._point_to_indices(start_point, metadata)
        else:
            start_indices = tuple(s // 2 for s in grid_shape)

        return self._outward_order(grid_shape, start_indices)

    # ----------------------------------------------------------------
    # Nearest neighbor initialization
    # ----------------------------------------------------------------

    def _is_adjacent(self, indices1, indices2):
        """Check if two grid points are adjacent (Chebyshev distance <= 1).

        Parameters
        ----------
        indices1 : tuple of int
        indices2 : tuple of int

        Returns
        -------
        bool
        """
        return all(abs(indices1[i] - indices2[i]) <= 1
                   for i in range(len(indices1)))

    def _build_grid_metadata(self):
        """Build grid metadata dict.

        Returns
        -------
        dict
            Keys: 'param_names', 'param_arrays', 'grid_shape', 'steps'
        """
        param_names = list(self.grid_params.keys())
        param_arrays = self._build_all_param_arrays()
        grid_shape = tuple(len(param_arrays[name]) for name in param_names)
        return {
            'param_names': param_names,
            'param_arrays': param_arrays,
            'grid_shape': grid_shape,
            'steps': {name: self.grid_params[name][2] for name in param_names}
        }

    def _find_nearest_successful(self, params, param_names):
        """Find nearest successfully evaluated point in cache.

        Parameters
        ----------
        params : dict
            Parameter values of the query point.
        param_names : list of str

        Returns
        -------
        dict or None
            Result dict of nearest successful point, or None if not found.
        """
        best_dist = np.inf
        best_result = None

        for cached_key, result in self._point_cache.items():
            if not result.get('success', False):
                continue
            dist = sum(
                (params[name] - cached_key[i]) ** 2
                for i, name in enumerate(param_names)
            ) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_result = result

        return best_result

    def _get_init_params(self, point, indices, last_indices,
                         last_result, metadata):
        """Get initialization parameters for a grid point.

        Uses last point if adjacent and successful, otherwise
        searches cache for nearest successful point.

        Parameters
        ----------
        point : dict
            Parameter values for this grid point.
        indices : tuple of int
        last_indices : tuple of int or None
        last_result : dict or None
        metadata : dict

        Returns
        -------
        dict or None
            Fitted params to initialize from, or None.
        """
        if (last_indices is not None and
                last_result is not None and
                last_result.get('success', False) and
                self._is_adjacent(indices, last_indices)):
            return last_result.get('params')

        nearest = self._find_nearest_successful(
            point, metadata['param_names'])
        if nearest is not None:
            return nearest.get('params')

        return None

    # ----------------------------------------------------------------
    # Core evaluation loop
    # ----------------------------------------------------------------

    def _run_grid(self, metadata, chi2_grid, result_grid,
                  evaluation_order, start_point, use_nn_init):
        """Evaluate all grid points and populate chi2 and result arrays.

        Skips already-evaluated points. Caches results by parameter
        values so sub-grids can reuse coarse grid evaluations.

        Parameters
        ----------
        metadata : dict
        chi2_grid : np.ndarray
            Modified in place.
        result_grid : np.ndarray
            Modified in place.
        evaluation_order : str
        start_point : dict or None
        use_nn_init : bool
        """
        ordered_indices = self._get_evaluation_order(
            metadata, evaluation_order, start_point)
        n_total = len(ordered_indices)

        last_indices = None
        last_result = None

        for i, indices in enumerate(ordered_indices):
            if result_grid[indices] is not None:
                continue

            point = self._indices_to_point(indices, metadata)

            if use_nn_init:
                init_params = self._get_init_params(
                    point, indices, last_indices, last_result, metadata)
                if init_params is not None:
                    point['_init_params'] = init_params

            if self.verbose:
                print(f"Grid point {i + 1}/{n_total}: {point}")

            result = self._fit_grid_point(point)

            chi2_grid[indices] = result.get('chi2', np.nan)
            result_grid[indices] = result

            key = self._make_cache_key(point, metadata['param_names'])
            self._point_cache[key] = result

            last_indices = indices
            last_result = result

            if self.verbose:
                print(f"  chi2 = {result.get('chi2', 'N/A')}")

    # ----------------------------------------------------------------
    # Run
    # ----------------------------------------------------------------

    def _resolve_run_params(self, evaluation_order, start_point,
                            use_nearest_neighbor_init,
                            max_refinements, max_expansions):
        """Resolve run() parameters against instance defaults.

        Parameters
        ----------
        evaluation_order : str or None
        start_point : dict or None
        use_nearest_neighbor_init : bool or None
        max_refinements : int or None
        max_expansions : int or None

        Returns
        -------
        tuple
            Resolved (order, start, nn_init, max_ref, max_exp)
        """
        order = evaluation_order or self.evaluation_order
        start = start_point or self.start_point
        nn_init = (use_nearest_neighbor_init
                   if use_nearest_neighbor_init is not None
                   else self.use_nearest_neighbor_init)
        max_ref = max_refinements or self.max_refinements
        max_exp = max_expansions or self.max_expansions
        return order, start, nn_init, max_ref, max_exp

    def _run_coarse_grid(self, order, start, nn_init):
        """Build and evaluate coarse grid, store in results_history[0].

        Parameters
        ----------
        order : str
        start : dict or None
        nn_init : bool
        """
        metadata = self._build_grid_metadata()
        chi2_grid, result_grid = self._build_empty_arrays(
            metadata['grid_shape'])
        self._run_grid(metadata, chi2_grid, result_grid,
                       order, start, nn_init)
        self.results_history = [{
            'chi2_grid': chi2_grid,
            'result_grid': result_grid,
            'metadata': metadata
        }]

    def run(self, refine=False, point_density_in_minimum=None,
            evaluation_order=None, start_point=None,
            use_nearest_neighbor_init=None,
            max_refinements=None, max_expansions=None):
        """Execute grid search.

        Parameters
        ----------
        refine : bool
            If True, refine around local minima after coarse grid.
        point_density_in_minimum : int or None
            Number of points within dchi2=1 of each minimum.
            Required if refine=True.
        evaluation_order : str or None
            'standard' or 'outward'. Overrides instance default.
        start_point : dict or None
            Starting point for 'outward' order. Overrides instance default.
        use_nearest_neighbor_init : bool or None
            Overrides instance default.
        max_refinements : int or None
            Overrides instance default.
        max_expansions : int or None
            Overrides instance default.
        """
        if self.grid_params is None:
            raise ValueError("grid_params must be set before running.")

        if refine and point_density_in_minimum is None:
            raise ValueError(
                "point_density_in_minimum required when refine=True.")

        order, start, nn_init, max_ref, max_exp = self._resolve_run_params(
            evaluation_order, start_point, use_nearest_neighbor_init,
            max_refinements, max_expansions)

        self._point_cache = {}
        self._run_coarse_grid(order, start, nn_init)

        if refine:
            self._run_refinement(point_density_in_minimum, max_ref, max_exp,
                                 order, nn_init)

    # ----------------------------------------------------------------
    # Minima finding
    # ----------------------------------------------------------------

    def _make_inverted_grid(self, chi2_grid):
        """Invert chi2 grid for peak_local_max, replacing NaN with -inf.

        Parameters
        ----------
        chi2_grid : np.ndarray

        Returns
        -------
        np.ndarray
        """
        safe = np.where(np.isnan(chi2_grid), np.inf, chi2_grid)
        return -safe

    def _make_footprint(self, n_dims):
        """Build n-dimensional 8-connectivity footprint.

        Parameters
        ----------
        n_dims : int

        Returns
        -------
        np.ndarray
        """
        return np.ones(tuple([3] * n_dims))

    def _peak_indices(self, chi2_grid):
        """Find local minima indices.

        Uses minimum_filter for local minima detection with proper edge
        handling, then removes dominated minima via flood-fill.

        A candidate minimum is dominated if an already-accepted minimum
        is reachable via a monotonically non-increasing path.

        Parameters
        ----------
        chi2_grid : np.ndarray

        Returns
        -------
        np.ndarray
            Shape (n_minima, n_dims) with grid indices.
        """
        chi2_safe = np.where(np.isnan(chi2_grid), np.inf, chi2_grid)
        structure = self._make_footprint(chi2_grid.ndim)

        local_min = minimum_filter(chi2_safe, size=3,
                                   mode='constant', cval=np.inf)
        is_local_min = (chi2_grid == local_min) & np.isfinite(chi2_grid)

        candidates = np.argwhere(is_local_min)
        if len(candidates) == 0:
            return candidates

        candidates = sorted(candidates,
                            key=lambda idx: chi2_grid[tuple(idx)])

        accepted = []
        for candidate in candidates:
            candidate_tuple = tuple(candidate)
            threshold = chi2_grid[candidate_tuple]
            mask = chi2_safe <= threshold
            labeled, _ = label(mask, structure=structure)
            candidate_label = labeled[candidate_tuple]

            dominated = any(labeled[tuple(acc)] == candidate_label
                            for acc in accepted)
            if not dominated:
                accepted.append(candidate)

        return (np.array(accepted) if accepted
                else np.empty((0, chi2_grid.ndim), dtype=int))

    def _build_minimum_entry(self, indices, chi2_grid, result_grid, level):
        """Build minimum dict for one grid point.

        Parameters
        ----------
        indices : tuple of int
        chi2_grid : np.ndarray
        result_grid : np.ndarray
        level : int

        Returns
        -------
        dict or None
            None if point has no valid result.
        """
        chi2 = chi2_grid[indices]
        result = result_grid[indices]
        if result is None or not np.isfinite(chi2):
            return None
        return {
            'indices': indices,
            'chi2': chi2,
            'params': result.get('params', {}),
            'level': level,
            'parent': None,
            'children': [],
            'refinement_history': []
        }

    def _find_minima_in_grid(self, chi2_grid, result_grid, level=0):
        """Find all local minima in a chi2 grid.

        Parameters
        ----------
        chi2_grid : np.ndarray
        result_grid : np.ndarray
        level : int

        Returns
        -------
        list of dict
        """
        peak_idx_array = self._peak_indices(chi2_grid)
        minima = []
        for idx in peak_idx_array:
            entry = self._build_minimum_entry(
                tuple(idx), chi2_grid, result_grid, level)
            if entry is not None:
                minima.append(entry)
        return minima

    def _is_edge_minimum(self, indices, grid_shape):
        """Check if a minimum lies on the grid edge.

        Parameters
        ----------
        indices : tuple of int
        grid_shape : tuple of int

        Returns
        -------
        bool
        """
        return any(indices[i] == 0 or indices[i] == grid_shape[i] - 1
                   for i in range(len(indices)))

    # ----------------------------------------------------------------
    # Public minima access
    # ----------------------------------------------------------------

    def _format_minima_output(self, minima):
        """Format minima as sorted list of (chi2, params, level) tuples.

        Parameters
        ----------
        minima : list of dict

        Returns
        -------
        list of tuple
            (chi2, params, level) sorted by chi2 ascending.
        """
        result = [(m['chi2'], m['params'], m['level']) for m in minima]
        result.sort(key=lambda x: x[0])
        return result

    def find_local_minima(self):
        """Find all local minima in the grid search results.

        If refinement was run, returns stored minima across all levels.
        Otherwise computes minima from the coarse grid.

        Returns
        -------
        list of tuple
            (chi2, params, level) sorted by chi2 ascending.
        """
        if self.results_history is None:
            raise ValueError(
                "No results available. Run grid search first.")

        if self.minima is not None:
            return self._format_minima_output(self.minima)

        level = self.results_history[0]
        raw_minima = self._find_minima_in_grid(
            level['chi2_grid'],
            level['result_grid'],
            level=0)

        return self._format_minima_output(raw_minima)

    # ----------------------------------------------------------------
    # Edge expansion helpers
    # ----------------------------------------------------------------

    def _get_edge_dims(self, indices, grid_shape):
        """Find dimensions where a point lies on the grid edge.

        Parameters
        ----------
        indices : tuple of int
        grid_shape : tuple of int

        Returns
        -------
        list of tuple
            (dim, direction) where direction is -1 (start) or +1 (end).
        """
        edge_dims = []
        for i in range(len(indices)):
            if indices[i] == 0:
                edge_dims.append((i, -1))
            elif indices[i] == grid_shape[i] - 1:
                edge_dims.append((i, +1))
        return edge_dims

    def _extend_param_array(self, name, param_array, direction,
                            n_expand, step=None):
        """Extend a 1D parameter array by n_expand steps in one direction.

        For log params, extension is in log10 space.

        Parameters
        ----------
        name : str
        param_array : np.ndarray
            Actual (linear) parameter values.
        direction : int
            -1 to prepend, +1 to append.
        n_expand : int
        step : float or None
            Step size in spec space. If None, uses self.grid_params.

        Returns
        -------
        np.ndarray
        """
        if step is None:
            step = self.grid_params[name][2]

        if self._is_log_param(name):
            if direction == -1:
                new_log = (np.log10(param_array[0])
                           - np.arange(n_expand, 0, -1) * step)
                return np.concatenate([10.0 ** new_log, param_array])
            new_log = (np.log10(param_array[-1])
                       + np.arange(1, n_expand + 1) * step)
            return np.concatenate([param_array, 10.0 ** new_log])

        if direction == -1:
            new_vals = param_array[0] - np.arange(n_expand, 0, -1) * step
            return np.concatenate([new_vals, param_array])
        new_vals = param_array[-1] + np.arange(1, n_expand + 1) * step
        return np.concatenate([param_array, new_vals])

    def _expand_array_along_dim(self, arr, dim, direction, n_expand):
        """Expand a numpy array along one dimension with NaN/None fill.

        Parameters
        ----------
        arr : np.ndarray
        dim : int
        direction : int
            -1 to prepend, +1 to append.
        n_expand : int

        Returns
        -------
        np.ndarray
        """
        shape = list(arr.shape)
        shape[dim] = n_expand
        if arr.dtype == object:
            new_slice = np.full(shape, None, dtype=object)
        else:
            new_slice = np.full(shape, np.nan)
        if direction == -1:
            return np.concatenate([new_slice, arr], axis=dim)
        return np.concatenate([arr, new_slice], axis=dim)

    def _apply_expansion(self, chi2_grid, result_grid, metadata,
                          edge_dims, n_expand):
        """Expand arrays simultaneously in all edge dimensions.

        Parameters
        ----------
        chi2_grid : np.ndarray
        result_grid : np.ndarray
        metadata : dict
        edge_dims : list of tuple
        n_expand : int

        Returns
        -------
        chi2_grid : np.ndarray
        result_grid : np.ndarray
        metadata : dict
        offsets : tuple of int
        """
        new_chi2 = chi2_grid
        new_result = result_grid
        new_param_arrays = dict(metadata['param_arrays'])
        offsets = [0] * len(metadata['param_names'])

        for dim, direction in edge_dims:
            name = metadata['param_names'][dim]
            new_chi2 = self._expand_array_along_dim(
                new_chi2, dim, direction, n_expand)
            new_result = self._expand_array_along_dim(
                new_result, dim, direction, n_expand)
            new_param_arrays[name] = self._extend_param_array(
                name, new_param_arrays[name], direction, n_expand,
                step=metadata['steps'][name])
            if direction == -1:
                offsets[dim] = n_expand

        new_metadata = {
            'param_names': metadata['param_names'],
            'param_arrays': new_param_arrays,
            'grid_shape': new_chi2.shape,
            'steps': metadata['steps']
        }
        return new_chi2, new_result, new_metadata, tuple(offsets)

    def _get_strip_ranges(self, minimum_indices, old_shape, new_shape,
                          edge_dims, strip_width):
        """Compute index ranges for the strip to evaluate after expansion.

        In edge dimensions: all new indices.
        In non-edge dimensions: within strip_width of minimum.

        Parameters
        ----------
        minimum_indices : tuple of int
            Minimum indices in the expanded grid.
        old_shape : tuple of int
        new_shape : tuple of int
        edge_dims : list of tuple
        strip_width : int

        Returns
        -------
        list of range
        """
        edge_dim_map = {dim: (direction, old_shape[dim])
                        for dim, direction in edge_dims}
        ranges = []
        for d in range(len(minimum_indices)):
            if d in edge_dim_map:
                direction, old_size = edge_dim_map[d]
                if direction == -1:
                    ranges.append(range(new_shape[d] - old_size))
                else:
                    ranges.append(range(old_size, new_shape[d]))
            else:
                lo = max(0, minimum_indices[d] - strip_width)
                hi = min(new_shape[d] - 1, minimum_indices[d] + strip_width)
                ranges.append(range(lo, hi + 1))
        return ranges

    def _evaluate_strip_points(self, chi2_grid, result_grid, metadata,
                               strip_ranges, nn_init):
        """Evaluate all unevaluated points in the strip.

        Parameters
        ----------
        chi2_grid : np.ndarray
            Modified in place.
        result_grid : np.ndarray
            Modified in place.
        metadata : dict
        strip_ranges : list of range
        nn_init : bool
        """
        from itertools import product as iproduct
        for indices in iproduct(*strip_ranges):
            if result_grid[indices] is not None:
                continue
            point = self._indices_to_point(indices, metadata)
            if nn_init:
                init_params = self._get_init_params(
                    point, indices, None, None, metadata)
                if init_params is not None:
                    point['_init_params'] = init_params
            result = self._fit_grid_point(point)
            chi2_grid[indices] = result.get('chi2', np.nan)
            result_grid[indices] = result
            key = self._make_cache_key(point, metadata['param_names'])
            self._point_cache[key] = result

    def _find_minimum_in_strip(self, chi2_grid, strip_ranges):
        """Find index of minimum chi2 point within strip ranges.

        Parameters
        ----------
        chi2_grid : np.ndarray
        strip_ranges : list of range

        Returns
        -------
        tuple of int or None
        """
        from itertools import product as iproduct
        best_chi2 = np.inf
        best_indices = None
        for indices in iproduct(*strip_ranges):
            chi2 = chi2_grid[indices]
            if np.isfinite(chi2) and chi2 < best_chi2:
                best_chi2 = chi2
                best_indices = indices
        return best_indices

    def _expand_once(self, indices, chi2_grid, result_grid, metadata,
                     n_expand, strip_width, nn_init):
        """Perform one expansion step around an edge minimum.

        Parameters
        ----------
        indices : tuple of int
        chi2_grid : np.ndarray
        result_grid : np.ndarray
        metadata : dict
        n_expand : int
        strip_width : int
        nn_init : bool

        Returns
        -------
        indices, chi2_grid, result_grid, metadata
        """
        edge_dims = self._get_edge_dims(indices, chi2_grid.shape)
        old_shape = chi2_grid.shape
        chi2_grid, result_grid, metadata, offsets = self._apply_expansion(
            chi2_grid, result_grid, metadata, edge_dims, n_expand)
        indices = self._shift_indices(indices, offsets)
        strip_ranges = self._get_strip_ranges(
            indices, old_shape, chi2_grid.shape, edge_dims, strip_width)
        self._evaluate_strip_points(
            chi2_grid, result_grid, metadata, strip_ranges, nn_init)
        strip_min = self._find_minimum_in_strip(chi2_grid, strip_ranges)
        if strip_min is not None and chi2_grid[strip_min] < chi2_grid[indices]:
            indices = strip_min
        return indices, chi2_grid, result_grid, metadata

    # ----------------------------------------------------------------
    # Edge expansion entry point
    # ----------------------------------------------------------------

    def _expand_edge_minimum(self, minimum, level_data, n_expand,
                              strip_width, nn_init, max_expansions):
        """Expand grid around an edge minimum until interior or limit reached.

        Parameters
        ----------
        minimum : dict
        level_data : dict
        n_expand : int
        strip_width : int
        nn_init : bool
        max_expansions : int

        Returns
        -------
        minimum : dict
            Updated indices and chi2.
        level_data : dict
            Updated arrays and metadata.
        """
        indices = minimum['indices']
        chi2_grid = level_data['chi2_grid']
        result_grid = level_data['result_grid']
        metadata = level_data['metadata']

        for _ in range(max_expansions):
            if not self._is_edge_minimum(indices, chi2_grid.shape):
                break
            indices, chi2_grid, result_grid, metadata = self._expand_once(
                indices, chi2_grid, result_grid, metadata,
                n_expand, strip_width, nn_init)

        minimum['indices'] = indices
        minimum['chi2'] = chi2_grid[indices]
        level_data = {
            'chi2_grid': chi2_grid,
            'result_grid': result_grid,
            'metadata': metadata
        }
        return minimum, level_data

# ----------------------------------------------------------------
    # Refinement: grid spec computation
    # ----------------------------------------------------------------

    def _get_spec_space_value(self, name, actual_value):
        """Convert actual parameter value to spec space.

        Parameters
        ----------
        name : str
        actual_value : float

        Returns
        -------
        float
            log10(actual_value) for log params, actual_value otherwise.
        """
        if self._is_log_param(name):
            return np.log10(actual_value)
        return actual_value

    def _compute_refined_spec_1d(self, name, center_val, current_step, n):
        """Compute refined grid spec for one parameter around a minimum.

        Parameters
        ----------
        name : str
        center_val : float
            Actual parameter value at minimum.
        current_step : float
            Step size in spec space.
        n : int
            point_density_in_minimum

        Returns
        -------
        list
            [min_spec, max_spec, new_step] in spec space.
        """
        half_n = n // 2
        new_step = current_step / (half_n + 2)
        n_steps = half_n + 1
        extent = n_steps * new_step
        center_spec = self._get_spec_space_value(name, center_val)
        return [center_spec - extent, center_spec + extent, new_step]

    def _compute_refined_spec(self, minimum, metadata, n):
        """Compute refined grid_params spec around a minimum.

        Parameters
        ----------
        minimum : dict
        metadata : dict
            Must contain 'steps' key.
        n : int

        Returns
        -------
        dict
            Refined grid_params in same format as self.grid_params.
        """
        return {
            name: self._compute_refined_spec_1d(
                name, minimum['params'][name], metadata['steps'][name], n)
            for name in metadata['param_names']
        }

    # ----------------------------------------------------------------
    # Cache management
    # ----------------------------------------------------------------

    def _make_cache_key(self, params, param_names):
        """Build cache key from parameter values.

        Parameters
        ----------
        params : dict
        param_names : list of str

        Returns
        -------
        tuple of float
        """
        return tuple(round(params[name], 10) for name in param_names)

    def _fill_from_cache(self, chi2_grid, result_grid, metadata):
        """Fill sub-grid from point cache where available.

        Parameters
        ----------
        chi2_grid : np.ndarray
            Modified in place.
        result_grid : np.ndarray
            Modified in place.
        metadata : dict
        """
        for indices in np.ndindex(*metadata['grid_shape']):
            point = self._indices_to_point(indices, metadata)
            key = self._make_cache_key(point, metadata['param_names'])
            if key in self._point_cache:
                result = self._point_cache[key]
                chi2_grid[indices] = result.get('chi2', np.nan)
                result_grid[indices] = result

    # ----------------------------------------------------------------
    # Refinement: sub-grid evaluation
    # ----------------------------------------------------------------

    def _build_subgrid_metadata(self, refined_spec):
        """Build metadata for a refined sub-grid.

        Parameters
        ----------
        refined_spec : dict

        Returns
        -------
        dict
            Keys: 'param_names', 'param_arrays', 'grid_shape', 'steps'
        """
        param_names = list(refined_spec.keys())
        param_arrays = {
            name: self._build_param_array(name, spec)
            for name, spec in refined_spec.items()
        }
        grid_shape = tuple(len(param_arrays[name]) for name in param_names)
        return {
            'param_names': param_names,
            'param_arrays': param_arrays,
            'grid_shape': grid_shape,
            'steps': {name: refined_spec[name][2] for name in param_names}
        }

    def _build_subgrid_data(self, refined_spec, nn_init):
        """Build and evaluate a refined sub-grid.

        Parameters
        ----------
        refined_spec : dict
        nn_init : bool

        Returns
        -------
        dict
            Keys: 'chi2_grid', 'result_grid', 'metadata'
        """
        metadata = self._build_subgrid_metadata(refined_spec)
        chi2_grid, result_grid = self._build_empty_arrays(metadata['grid_shape'])
        self._fill_from_cache(chi2_grid, result_grid, metadata)
        self._run_grid(metadata, chi2_grid, result_grid,
                       'standard', None, nn_init)
        return {
            'chi2_grid': chi2_grid,
            'result_grid': result_grid,
            'metadata': metadata
        }

    # ----------------------------------------------------------------
    # Refinement: convergence
    # ----------------------------------------------------------------

    def _count_below_threshold_1d(self, chi2_slice, center_pos,
                                   threshold, direction):
        """Count consecutive points at or below threshold in one direction.

        Parameters
        ----------
        chi2_slice : np.ndarray
        center_pos : int
        threshold : float
        direction : int
            +1 or -1

        Returns
        -------
        int
        """
        count = 0
        pos = center_pos + direction
        while 0 <= pos < len(chi2_slice):
            if np.isfinite(chi2_slice[pos]) and chi2_slice[pos] <= threshold:
                count += 1
                pos += direction
            else:
                break
        return count

    def _is_converged_along_dim(self, chi2_grid, min_idx, dim, n, threshold):
        """Check convergence along one parameter axis.

        Parameters
        ----------
        chi2_grid : np.ndarray
        min_idx : tuple of int
        dim : int
        n : int
        threshold : float

        Returns
        -------
        bool
        """
        slice_idx = list(min_idx)
        slice_idx[dim] = slice(None)
        chi2_slice = chi2_grid[tuple(slice_idx)]
        half_n = n // 2
        count_pos = self._count_below_threshold_1d(
            chi2_slice, min_idx[dim], threshold, +1)
        count_neg = self._count_below_threshold_1d(
            chi2_slice, min_idx[dim], threshold, -1)
        return count_pos >= half_n and count_neg >= half_n

    def _is_converged(self, min_idx, chi2_grid, n):
        """Check if minimum has enough points within dchi2=1 in all directions.

        Parameters
        ----------
        min_idx : tuple of int
        chi2_grid : np.ndarray
        n : int

        Returns
        -------
        bool
        """
        threshold = chi2_grid[min_idx] + 1.0
        return all(
            self._is_converged_along_dim(chi2_grid, min_idx, dim, n, threshold)
            for dim in range(len(min_idx))
        )

    # ----------------------------------------------------------------
    # Refinement: minimum management
    # ----------------------------------------------------------------

    def _create_child_minimum(self, sub_min, parent, level):
        """Create a child minimum from a sub-grid result.

        Parameters
        ----------
        sub_min : dict
        parent : dict
        level : int

        Returns
        -------
        dict
        """
        return {
            'indices': sub_min['indices'],
            'chi2': sub_min['chi2'],
            'params': sub_min['params'],
            'level': level,
            'parent': parent,
            'children': [],
            'refinement_history': []
        }

    def _update_minimum_from_subgrid(self, minimum, sub_min, level):
        """Update minimum location from sub-grid result.

        Parameters
        ----------
        minimum : dict
            Modified in place.
        sub_min : dict
        level : int
        """
        minimum['indices'] = sub_min['indices']
        minimum['chi2'] = sub_min['chi2']
        minimum['params'] = sub_min['params']
        minimum['level'] = level

    # ----------------------------------------------------------------
    # Refinement: core
    # ----------------------------------------------------------------

    def _refine_once(self, minimum, n, level, nn_init, max_exp):
        """Perform one refinement iteration around a minimum.

        Parameters
        ----------
        minimum : dict
        n : int
        level : int
        nn_init : bool
        max_exp : int

        Returns
        -------
        sub_minima : list of dict
        subgrid_data : dict
        converged : bool
        """
        current_metadata = (minimum['refinement_history'][-1]['metadata']
                            if minimum['refinement_history']
                            else self.results_history[0]['metadata'])

        refined_spec = self._compute_refined_spec(minimum, current_metadata, n)
        subgrid_data = self._build_subgrid_data(refined_spec, nn_init)
        minimum['refinement_history'].append(subgrid_data)

        sub_minima = self._find_minima_in_grid(
            subgrid_data['chi2_grid'],
            subgrid_data['result_grid'],
            level=level)

        if len(sub_minima) == 0:
            return [], subgrid_data, True

        strip_width = max(3, n)
        for i, sub_min in enumerate(sub_minima):
            if self._is_edge_minimum(sub_min['indices'],
                                     subgrid_data['chi2_grid'].shape):
                sub_min, subgrid_data = self._expand_edge_minimum(
                    sub_min, subgrid_data, strip_width, strip_width,
                    nn_init, max_exp)
                sub_minima[i] = sub_min
                minimum['refinement_history'][-1] = subgrid_data

        if len(sub_minima) == 1:
            converged = self._is_converged(
                sub_minima[0]['indices'],
                subgrid_data['chi2_grid'], n)
            return sub_minima, subgrid_data, converged

        return sub_minima, subgrid_data, False

    def _refine_minimum(self, minimum, n, max_ref, nn_init, max_exp):
        """Refine around one minimum until convergence or limit reached.

        Parameters
        ----------
        minimum : dict
        n : int
        max_ref : int
        nn_init : bool
        max_exp : int

        Returns
        -------
        list of dict
            Final minima (may include children from splits).
        """
        for level in range(1, max_ref + 1):
            sub_minima, subgrid_data, converged = self._refine_once(
                minimum, n, level, nn_init, max_exp)

            if len(sub_minima) == 0:
                if self.verbose:
                    print(f"  Minimum disappeared at refinement level {level}")
                return [minimum]

            if len(sub_minima) > 1:
                if self.verbose:
                    print(f"  Split into {len(sub_minima)} at level {level}")
                children = []
                for sub_min in sub_minima:
                    child = self._create_child_minimum(sub_min, minimum, level)
                    minimum['children'].append(child)
                    children.append(child)
                remaining = max_ref - level
                result = []
                for child in children:
                    result.extend(
                        self._refine_minimum(child, n, remaining, nn_init, max_exp))
                return result

            self._update_minimum_from_subgrid(minimum, sub_minima[0], level)

            if converged:
                if self.verbose:
                    print(f"  Converged at refinement level {level}")
                return [minimum]

        if self.verbose:
            print("  Max refinements reached")
        return [minimum]

    # ----------------------------------------------------------------
    # Refinement: entry point
    # ----------------------------------------------------------------

    def _run_refinement(self, n, max_ref, max_exp, order, nn_init):
        """Find coarse minima, expand edges, refine each minimum.

        Parameters
        ----------
        n : int
            point_density_in_minimum
        max_ref : int
        max_exp : int
        order : str
        nn_init : bool
        """
        level_data = self.results_history[0]
        coarse_minima = self._find_minima_in_grid(
            level_data['chi2_grid'],
            level_data['result_grid'],
            level=0)

        if len(coarse_minima) == 0:
            self.minima = []
            return

        strip_width = max(3, n)
        for i, minimum in enumerate(coarse_minima):
            if self._is_edge_minimum(minimum['indices'],
                                     level_data['chi2_grid'].shape):
                minimum, level_data = self._expand_edge_minimum(
                    minimum, level_data, strip_width, strip_width,
                    nn_init, max_exp)
                coarse_minima[i] = minimum

        self.results_history[0] = level_data

        self.minima = []
        for minimum in coarse_minima:
            refined = self._refine_minimum(minimum, n, max_ref, nn_init, max_exp)
            self.minima.extend(refined)

# ----------------------------------------------------------------
    # Uncertainties
    # ----------------------------------------------------------------

    def _get_minimum_grid(self, minimum):
        """Get best available chi2 grid for a minimum.

        Uses last refinement level if available, else coarse grid.

        Parameters
        ----------
        minimum : dict

        Returns
        -------
        dict
            Keys: 'chi2_grid', 'result_grid', 'metadata'
        """
        if minimum['refinement_history']:
            return minimum['refinement_history'][-1]
        return self.results_history[0]

    def _chi2_slice_1d(self, chi2_grid, min_idx, dim):
        """Extract 1D chi2 slice along one dimension through minimum.

        Parameters
        ----------
        chi2_grid : np.ndarray
        min_idx : tuple of int
        dim : int

        Returns
        -------
        np.ndarray
        """
        idx = list(min_idx)
        idx[dim] = slice(None)
        return chi2_grid[tuple(idx)]

    def _find_bounds_1d(self, chi2_slice, center_pos, threshold):
        """Find range of indices where chi2 <= threshold.

        Parameters
        ----------
        chi2_slice : np.ndarray
        center_pos : int
        threshold : float

        Returns
        -------
        tuple of int
            (lo, hi) inclusive indices
        """
        lo = center_pos
        while (lo > 0 and np.isfinite(chi2_slice[lo - 1])
               and chi2_slice[lo - 1] <= threshold):
            lo -= 1
        hi = center_pos
        while (hi < len(chi2_slice) - 1 and np.isfinite(chi2_slice[hi + 1])
               and chi2_slice[hi + 1] <= threshold):
            hi += 1
        return lo, hi

    def _indices_to_param_range(self, lo, hi, param_array):
        """Convert index bounds to parameter value range.

        Parameters
        ----------
        lo : int
        hi : int
        param_array : np.ndarray

        Returns
        -------
        tuple of float
            (lo_val, hi_val)
        """
        return param_array[lo], param_array[hi]

    def _get_uncertainty_1d(self, minimum, grid_data, dim):
        """Get uncertainty for one parameter dimension.

        Parameters
        ----------
        minimum : dict
        grid_data : dict
        dim : int

        Returns
        -------
        tuple of float
            (lo_val, hi_val) actual parameter values
        """
        chi2_grid = grid_data['chi2_grid']
        metadata = grid_data['metadata']
        name = metadata['param_names'][dim]
        param_array = metadata['param_arrays'][name]
        chi2_slice = self._chi2_slice_1d(chi2_grid, minimum['indices'], dim)
        threshold = minimum['chi2'] + 1.0
        lo, hi = self._find_bounds_1d(
            chi2_slice, minimum['indices'][dim], threshold)
        return self._indices_to_param_range(lo, hi, param_array)

    def get_uncertainties(self, minimum):
        """Get 1-sigma parameter uncertainties for a minimum.

        Uncertainties are the range of parameter values where
        chi2 <= chi2_min + 1.0 along each axis independently.

        Parameters
        ----------
        minimum : dict
            Minimum object from self.minima.

        Returns
        -------
        dict
            {param_name: (lo_val, hi_val)} for each parameter.
        """
        grid_data = self._get_minimum_grid(minimum)
        metadata = grid_data['metadata']
        return {
            name: self._get_uncertainty_1d(minimum, grid_data, dim)
            for dim, name in enumerate(metadata['param_names'])
        }

    # ----------------------------------------------------------------
    # Plotting helpers
    # ----------------------------------------------------------------

    def _marginalize_2d(self, chi2_grid, dim1, dim2):
        """Marginalize chi2 grid to 2D by taking minimum along other axes.

        Parameters
        ----------
        chi2_grid : np.ndarray
        dim1 : int
        dim2 : int

        Returns
        -------
        np.ndarray, shape (n_dim1, n_dim2)
        """
        dims_to_reduce = tuple(d for d in range(chi2_grid.ndim)
                               if d != dim1 and d != dim2)
        result = (np.nanmin(chi2_grid, axis=dims_to_reduce)
                  if dims_to_reduce else chi2_grid.copy())
        if dim1 > dim2:
            result = result.T
        return result

    def _marginalize_1d(self, chi2_grid, dim):
        """Marginalize chi2 grid to 1D by taking minimum along other axes.

        Parameters
        ----------
        chi2_grid : np.ndarray
        dim : int

        Returns
        -------
        np.ndarray
        """
        dims_to_reduce = tuple(d for d in range(chi2_grid.ndim) if d != dim)
        return (np.nanmin(chi2_grid, axis=dims_to_reduce)
                if dims_to_reduce else chi2_grid.copy().ravel())

    def _plot_heatmap_2d(self, ax, chi2_2d, param_arrays, names):
        """Plot 2D chi2 heatmap.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
        chi2_2d : np.ndarray
        param_arrays : tuple of np.ndarray
        names : tuple of str

        Returns
        -------
        matplotlib.image.AxesImage
        """
        arr1, arr2 = param_arrays
        finite = chi2_2d[np.isfinite(chi2_2d)]
        vmin = finite.min() if len(finite) else 0
        vmax = np.percentile(finite, 95) if len(finite) else 1
        im = ax.imshow(chi2_2d.T, origin='lower', aspect='auto',
                       extent=[arr1[0], arr1[-1], arr2[0], arr2[-1]],
                       vmin=vmin, vmax=vmax, cmap='viridis')
        ax.set_xlabel(names[0])
        ax.set_ylabel(names[1])
        return im

    def _mark_minima_2d(self, ax, minima_list, name1, name2):
        """Mark local minima on a 2D plot.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
        minima_list : list of tuple
        name1 : str
        name2 : str
        """
        for chi2, params, level in minima_list:
            ax.plot(params[name1], params[name2], 'kx',
                    markersize=10, markeredgewidth=2)

    def _plot_1d_projection(self, ax, chi2_1d, param_array, name, minima_list):
        """Plot 1D marginalized chi2 projection.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
        chi2_1d : np.ndarray
        param_array : np.ndarray
        name : str
        minima_list : list of tuple
        """
        ax.plot(param_array, chi2_1d)
        chi2_min = np.nanmin(chi2_1d)
        ax.axhline(chi2_min + 1.0, color='r', linestyle='--',
                   label=r'$\Delta\chi^2=1$')
        for chi2, params, level in minima_list:
            ax.axvline(params[name], color='k', linestyle=':')
        ax.set_xlabel(name)
        ax.set_ylabel(r'$\chi^2$')
        ax.legend(fontsize=8)

    def _plot_2d_projections(self, axes, chi2_grid, metadata, minima_list):
        """Plot all 2D chi2 projections.

        Parameters
        ----------
        axes : list of matplotlib.axes.Axes
        chi2_grid : np.ndarray
        metadata : dict
        minima_list : list of tuple
        """
        import matplotlib.pyplot as plt
        from itertools import combinations
        param_names = metadata['param_names']
        param_arrays = metadata['param_arrays']
        for ax, (dim1, dim2) in zip(
                axes, combinations(range(len(param_names)), 2)):
            chi2_2d = self._marginalize_2d(chi2_grid, dim1, dim2)
            arrays = (param_arrays[param_names[dim1]],
                      param_arrays[param_names[dim2]])
            im = self._plot_heatmap_2d(
                ax, chi2_2d, arrays,
                (param_names[dim1], param_names[dim2]))
            self._mark_minima_2d(
                ax, minima_list, param_names[dim1], param_names[dim2])
            plt.colorbar(im, ax=ax)

    def _plot_1d_projections(self, axes, chi2_grid, metadata, minima_list):
        """Plot all 1D chi2 projections.

        Parameters
        ----------
        axes : list of matplotlib.axes.Axes
        chi2_grid : np.ndarray
        metadata : dict
        minima_list : list of tuple
        """
        param_names = metadata['param_names']
        param_arrays = metadata['param_arrays']
        for ax, (dim, name) in zip(axes, enumerate(param_names)):
            chi2_1d = self._marginalize_1d(chi2_grid, dim)
            self._plot_1d_projection(
                ax, chi2_1d, param_arrays[name], name, minima_list)

    def _create_figure(self, n_2d, n_1d):
        """Create figure and axes layout.

        Parameters
        ----------
        n_2d : int
        n_1d : int

        Returns
        -------
        fig : matplotlib.figure.Figure
        axes_2d : list of matplotlib.axes.Axes
        axes_1d : list of matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt
        n_cols = max(n_2d, n_1d, 1)
        n_rows = 1 if n_2d == 0 else 2
        fig, axes = plt.subplots(n_rows, n_cols,
                                  figsize=(4 * n_cols, 4 * n_rows),
                                  squeeze=False)
        for ax in axes.ravel():
            ax.set_visible(False)
        axes_2d = list(axes[0, :n_2d]) if n_2d > 0 else []
        axes_1d = list(axes[-1, :n_1d])
        for ax in axes_2d + axes_1d:
            ax.set_visible(True)
        return fig, axes_2d, axes_1d

    # ----------------------------------------------------------------
    # Plotting entry point
    # ----------------------------------------------------------------

    def plot(self):
        """Plot chi2 grid search results.

        Shows 2D chi2 projections for all parameter pairs,
        marginalized over remaining parameters. Shows 1D marginalized
        projections along each axis. Marks all local minima.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt
        if self.results_history is None:
            raise ValueError("No results. Run grid search first.")
        level_data = self.results_history[0]
        chi2_grid = level_data['chi2_grid']
        metadata = level_data['metadata']
        n = len(metadata['param_names'])
        minima_list = self.find_local_minima()
        n_2d = n * (n - 1) // 2
        fig, axes_2d, axes_1d = self._create_figure(n_2d, n)
        if n_2d > 0:
            self._plot_2d_projections(
                axes_2d, chi2_grid, metadata, minima_list)
        self._plot_1d_projections(
            axes_1d, chi2_grid, metadata, minima_list)
        plt.tight_layout()
        return fig

    # ----------------------------------------------------------------
    # Convenience properties
    # ----------------------------------------------------------------

    @property
    def best(self):
        """Best minimum dict from self.minima, or best coarse grid point.

        Returns
        -------
        dict or None
        """
        if self.minima:
            return min(self.minima, key=lambda m: m['chi2'])
        if self.results_history is None:
            return None
        chi2_grid = self.results_history[0]['chi2_grid']
        result_grid = self.results_history[0]['result_grid']
        idx = np.unravel_index(np.nanargmin(chi2_grid), chi2_grid.shape)
        return result_grid[idx]


class ParallaxGridSearch(BaseRectGridSearch):
    """
    Grid search over parallax parameters pi_E_E and pi_E_N.

    For each grid point, runs mmexo.fitters.SFitFitter with fixed parallax
    parameters and returns chi2 + fit results.
    """

    def __init__(self, datasets, static_params, grid_params=None,
                 pi_E_E_min=-0.7, pi_E_E_max=0.7, pi_E_E_step=0.05,
                 pi_E_N_min=-1.0, pi_E_N_max=1.0, pi_E_N_step=0.1,
                 fitter_kwargs=None, skip_optimization=False, verbose=False):
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
            skip_optimization : bool, optional
                If True, calculate chi2 without optimization (much faster).
                If False, optimize parameters at each grid point. Default is False.
            verbose: *bool*
                If True, print progress information
        """
        super().__init__(datasets=datasets, skip_optimization=skip_optimization, verbose=verbose)

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
        Evaluate or optimize one grid point.

        If skip_optimization=True, calculates chi2 directly with fixed parameters.
        Otherwise, uses self.current_params as the starting point and optimizes.

        Parameters
        ----------
        grid_params : dict
            Contains 'pi_E_E' and 'pi_E_N' for this grid point

        Returns
        -------
        dict
            Contains 'chi2', 'params' (best-fit or input parameters),
            'success', and any other output
        """
        # Combine current params with this grid point's parallax values
        model_params = self.current_params.copy()
        model_params['pi_E_E'] = grid_params['pi_E_E']
        model_params['pi_E_N'] = grid_params['pi_E_N']

        if self.skip_optimization:
            # Just calculate chi2 without optimization
            try:
                # Create fitter but don't run optimization
                fitter = mmexo.fitters.SFitFitter(
                    initial_model_params=model_params,
                    datasets=self.datasets,
                    parameters_to_fit=[],  # Empty list - no optimization
                    **self.fitter_kwargs
                )
                event = fitter.get_event()
                chi2 = event.get_chi2()

                result = {
                    'chi2': chi2,
                    'params': model_params,
                    'success': True,
                }
            except Exception as e:
                if self.verbose:
                    print(f"  Chi2 calculation failed: {e}")

                result = {
                    'chi2': np.nan,
                    'params': None,
                    'success': False,
                    'error': str(e)
                }
        else:
            # Optimize parameters
            parameters_to_fit = list(self.static_params.keys())

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

    def run_with_refinement(self, chi2_threshold=200, min_step_size=0.005, radius_steps=2):
        """
        Execute grid search with iterative refinement around minima.

        Prioritizes global minimum - refines it completely before processing other minima.

        Workflow:
        1. Run initial grid
        2. Find global minimum
        3. If has incomplete neighbors: extend edges at current level
        4. Else if can refine further: refine to next level
        5. Else: global min complete, process other minima within chi2_threshold
        6. Repeat until all minima complete or step size < min_step_size

        Parameters
        ----------
        chi2_threshold : float, optional
            After global min complete, refine other minima within this chi2. Default is 200.
        min_step_size : float, optional
            Stop refining when step size reaches this value. Default is 0.005.
        radius_steps : int, optional
            Number of steps (at current resolution) to extend around each minimum.
            Default is 2.
        """
        # Run initial grid
        self.run()

        # Store min_step_size for use in find_local_minima()
        self.min_step_size = min_step_size

        # Add refinement_level to initial results
        for result in self.results:
            result['refinement_level'] = 0

        # Get parameter names and initial step sizes
        param_names = []
        original_steps = {}
        for key in self.grid_params:
            if key.endswith('_step'):
                param_name = key[:-5]
                param_names.append(param_name)
                original_steps[param_name] = self.grid_params[key]

        # Track evaluated points to avoid duplicates
        evaluated_points = set()
        for result in self.results:
            point = tuple(round(result[param], 10) for param in param_names)
            evaluated_points.add(point)

        # Tolerance for neighbor matching
        tolerance = min_step_size / 10

        # Iterative refinement
        iteration = 0
        global_min_complete = False

        while True:
            iteration += 1

            if self.verbose:
                print(f"\nRefinement iteration {iteration}")

            # Filter successful results
            valid_results = [r for r in self.results if r.get('success', False)]

            # Find local minima
            local_minima = self.find_local_minima()

            if len(local_minima) == 0:
                if self.verbose:
                    print("No local minima found, stopping refinement")
                break

            min_chi2 = local_minima[0][0]

            # Determine which minima to process
            if not global_min_complete:
                # Focus on global minimum only
                minima_to_process = [local_minima[0]]
                if self.verbose:
                    print(f"Processing global minimum: chi2={min_chi2:.2f}")
            else:
                # Global min done, process others within threshold
                other_minima = [(chi2, params) for chi2, params in local_minima[1:]
                                if np.sqrt(chi2 - min_chi2) < np.sqrt(chi2_threshold)]

                if len(other_minima) == 0:
                    if self.verbose:
                        print("No other minima within threshold, refinement complete")
                    break

                minima_to_process = other_minima
                if self.verbose:
                    print(f"Processing {len(other_minima)} other minima within threshold")

            # Check which minima need edge extension vs refinement
            needs_edge_extension = []
            can_refine = []
            already_at_finest = []

            for chi2, params in minima_to_process:
                # Get this point's refinement level and step size
                refinement_level = None
                for r in valid_results:
                    if all(abs(r[param] - params[param]) < tolerance for param in param_names):
                        refinement_level = r.get('refinement_level', 0)
                        break

                if refinement_level is None:
                    continue  # Shouldn't happen

                current_point_steps = {param: original_steps[param] / (2 ** refinement_level)
                                       for param in param_names}

                # Check if has incomplete neighbors at current level
                if self._has_incomplete_neighbors(params, current_point_steps,
                                                  evaluated_points, tolerance):
                    needs_edge_extension.append((chi2, params, refinement_level))
                elif min(current_point_steps.values()) / 2 > min_step_size:
                    # Can refine to next level
                    can_refine.append((chi2, params, refinement_level))
                else:
                    # Already at finest resolution
                    already_at_finest.append((chi2, params))

            if self.verbose:
                print(f"  {len(needs_edge_extension)} need edge extension")
                print(f"  {len(can_refine)} can be refined")
                print(f"  {len(already_at_finest)} at finest resolution")

            # Extend edges first (at current level)
            if needs_edge_extension:
                if self.verbose:
                    print(f"Extending edges at current level")

                new_grid_points = []
                for chi2, params, ref_level in needs_edge_extension:
                    step_sizes_at_level = {param: original_steps[param] / (2 ** ref_level)
                                           for param in param_names}

                    # Find missing neighbors
                    for deltas in product([-1, 0, 1], repeat=len(param_names)):
                        if all(d == 0 for d in deltas):
                            continue

                        expected_neighbor = tuple(
                            round(params[param_names[i]] + deltas[i] * step_sizes_at_level[param_names[i]], 10)
                            for i in range(len(param_names))
                        )

                        # Check if this neighbor exists (within tolerance)
                        neighbor_exists = False
                        for eval_point in evaluated_points:
                            distance = sum((eval_point[i] - expected_neighbor[i]) ** 2
                                           for i in range(len(param_names))) ** 0.5
                            if distance < tolerance:
                                neighbor_exists = True
                                break

                        if not neighbor_exists:
                            # Add this missing neighbor
                            point_dict = {param_names[i]: expected_neighbor[i]
                                          for i in range(len(param_names))}
                            new_grid_points.append((point_dict, ref_level))
                            evaluated_points.add(expected_neighbor)

                if len(new_grid_points) == 0:
                    if self.verbose:
                        print("No new edge points to add")
                    continue

                if self.verbose:
                    print(f"Evaluating {len(new_grid_points)} edge points")

                # Evaluate new edge points
                for i, (grid_params, ref_level) in enumerate(new_grid_points):
                    if self.verbose and (i + 1) % 10 == 0:
                        print(f"  Edge point {i + 1}/{len(new_grid_points)}")

                    result = self._fit_grid_point(grid_params)
                    full_result = {**grid_params, **result}
                    full_result['refinement_level'] = ref_level

                    self.results.append(full_result)
                    self._grid.append(grid_params)

                # Reset cached best and continue
                self._best = None
                continue

            # No edges to extend - refine to next level
            if len(can_refine) == 0:
                if not global_min_complete:
                    # Global minimum is now complete
                    if self.verbose:
                        print("Global minimum refinement complete")
                    global_min_complete = True
                    continue  # Process other minima
                else:
                    # All minima complete
                    if self.verbose:
                        print("All minima complete, stopping refinement")
                    break

            if self.verbose:
                print(f"Refining {len(can_refine)} minima to next level")

            # Generate refined grid points around each minimum
            new_grid_points = []
            for chi2, params, ref_level in can_refine:
                step_sizes_at_level = {param: original_steps[param] / (2 ** ref_level)
                                       for param in param_names}

                # Create sub-grid around this minimum
                param_ranges = []
                for param in param_names:
                    center = params[param]
                    step = step_sizes_at_level[param] / 2  # Half the step for next level
                    radius = radius_steps * step

                    # Generate points in this dimension
                    param_min_local = center - radius
                    param_max_local = center + radius
                    n_points = 2 * radius_steps + 1
                    param_grid = np.linspace(param_min_local, param_max_local, n_points)
                    param_grid = np.round(param_grid, decimals=10)
                    param_ranges.append(param_grid)

                # Create rectangular sub-grid
                sub_grid = self._make_rect_grid(param_ranges, param_names)

                # Filter out already-evaluated points
                for point_dict in sub_grid:
                    point = tuple(round(point_dict[param], 10) for param in param_names)

                    # Check with tolerance
                    point_exists = False
                    for eval_point in evaluated_points:
                        distance = sum((eval_point[i] - point[i]) ** 2
                                       for i in range(len(param_names))) ** 0.5
                        if distance < tolerance:
                            point_exists = True
                            break

                    if not point_exists:
                        new_grid_points.append((point_dict, ref_level + 1))
                        evaluated_points.add(point)

            if len(new_grid_points) == 0:
                if self.verbose:
                    print("No new points to evaluate")
                if not global_min_complete:
                    global_min_complete = True
                    continue
                else:
                    break

            if self.verbose:
                print(f"Evaluating {len(new_grid_points)} refined points")

            # Evaluate new points
            for i, (grid_params, ref_level) in enumerate(new_grid_points):
                if self.verbose and (i + 1) % 10 == 0:
                    print(f"  Refined point {i + 1}/{len(new_grid_points)}")

                result = self._fit_grid_point(grid_params)
                full_result = {**grid_params, **result}
                full_result['refinement_level'] = ref_level

                # Add to results and grid
                self.results.append(full_result)
                self._grid.append(grid_params)

            # Reset cached best
            self._best = None

        if self.verbose:
            print(f"\nRefinement complete after {iteration} iterations")
            print(f"Total grid points evaluated: {len(self.results)}")

    def _has_incomplete_neighbors(self, params, step_sizes_at_level, evaluated_points, tolerance):
        """
        Check if a point is missing any neighbors at a specific refinement level.

        Parameters
        ----------
        params : dict
            Parameter values for the point to check
        step_sizes_at_level : dict
            Step size for each parameter at this refinement level
        evaluated_points : set
            Set of already-evaluated points as tuples
        tolerance : float
            Distance tolerance for considering a point as a neighbor

        Returns
        -------
        bool
            True if any neighbors are missing (point is at edge), False if all neighbors exist
        """

        param_names = list(step_sizes_at_level.keys())

        # Check all neighbors
        for deltas in product([-1, 0, 1], repeat=len(param_names)):
            if all(d == 0 for d in deltas):
                continue  # Skip center point

            # Calculate expected neighbor position
            expected_neighbor = tuple(
                round(params[param_names[i]] + deltas[i] * step_sizes_at_level[param_names[i]], 10)
                for i in range(len(param_names))
            )

            # Check if neighbor exists within tolerance
            neighbor_exists = False
            for eval_point in evaluated_points:
                distance = sum((eval_point[i] - expected_neighbor[i]) ** 2
                               for i in range(len(param_names))) ** 0.5
                if distance < tolerance:
                    neighbor_exists = True
                    break

            if not neighbor_exists:
                return True  # Missing at least one neighbor

        return False  # All neighbors exist

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
                f.write('  '.join(all_column_names) + '\n')

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
