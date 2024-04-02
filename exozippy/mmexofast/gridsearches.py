import numpy as np
import MulensModel
import sfit_minimizer


# The way this is implemented uses a lot of architecture from MulensModel.
# This may make it inefficient.

class EventFinderGridSearch():
    """
    Based on Kim et al. 2018, AJ, 155, 76
    """

    def __init__(self,
                 datasets=None, t_eff_3=1, d_t_eff=1/3., t_eff_max=99,
                 d_t_0=1/3, t_0_min=None, t_0_max=None):
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

    def run(self, verbose=False):
        results = []
        for (t_0, t_eff) in zip(self.grid_t_0, self.grid_t_eff):
            chi2s = self.do_fits({'t_0': t_0, 't_eff': t_eff}, verbose=verbose)
            if verbose:
                print(t_0, t_eff, chi2s)

            results.append(chi2s)

        self.results = np.array(results)

    def do_fits(self, parameters, verbose=False):
        z_t_eff = 5
        n_min = 50

        chi2s = np.array([np.nan, np.nan])

        trimmed_datasets = []
        for dataset in self.datasets:
            # Restrict fitting to points t_0 +- z * t_eff with the
            # requirement N > 50.
            index = ((dataset.time >
                     (parameters['t_0'] - z_t_eff * parameters['t_eff'])) &
                     (dataset.time <
                      (parameters['t_0'] + z_t_eff * parameters['t_eff'])))
            # Minimum requirement for including a dataset
            if np.sum(index) >= n_min:
                trimmed_dataset = MulensModel.MulensData(
                    [dataset.time[index], dataset.flux[index],
                     dataset.err_flux[index]])
                trimmed_datasets.append(trimmed_dataset)

        if verbose:
            print('trimmed datasets', len(trimmed_datasets),
                  [dataset.n_epochs for dataset in trimmed_datasets])

        #print('trimmed datasets', trimmed_datasets)
        #print('dtype', type(trimmed_datasets))
        # Only fit the window if there's enough data to do so.
        if len(trimmed_datasets) >= 1:
            for j in [1, 2]:
                parameters['j'] = j
                #print('len datasets, len trimmed', len(self.datasets), len(trimmed_datasets))
                ef_sfit = EFSFitFunction(trimmed_datasets, parameters)
                #print('init theta, chi2', ef_sfit.theta, ef_sfit.get_chi2())
                ef_sfit.update_all(theta=ef_sfit.theta + ef_sfit.get_step())
                #print('final theta, chi2', ef_sfit.theta, ef_sfit.chi2)
                chi2 = ef_sfit.chi2

                chi2s[j-1] = chi2

        return chi2s

    @property
    def grid_t_0(self):
        if self._grid_t_0 is None:
            self._setup_grid()

        return self._grid_t_0

    @property
    def grid_t_eff(self):
        if self._grid_t_eff is None:
            self._setup_grid()

        return self._grid_t_eff

    @property
    def best(self):
        if (self._best is None) & (self.results is not None):
            index_1 = np.nanargmin(self.results[:, 0])
            index_2 = np.nanargmin(self.results[:, 1])
            #print('index_1, 2', index_1, index_2)
            #print(self.results.shape)
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

        return self._best


class EFSFitFunction(sfit_minimizer.SFitFunction):

    def __init__(self, datasets, parameters):
        self.datasets = datasets
        self.parameters = parameters
        self.parameters_to_fit = []
        self.n_params = 2 * len(self.datasets)
        #(len(self.datasets))

        self._theta = None
        self._q = None
        self._magnification = None

        self.data_len = None
        self.flatten_data() # destroys self.theta
        self.data_indices = self._set_data_indices()
        self.theta = np.array([np.array([1, 0])
                               for i in range(len(self.datasets))]).flatten()
        #print('after setup', self.theta)
        #print('data_indices', self.data_indices)
        #print('n_params', self.n_params)

    def _set_data_indices(self):
        #print('data_len', self.data_len)
        data_indices = np.cumsum(self.data_len)
        #print('cumsum', np.cumsum(self.data_len))
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

    def update_all(self, theta=None, verbose=False):
        sfit_minimizer.sfit_classes.SFitFunction.update_all(self,
            theta=theta, verbose=verbose)

    def calc_model(self):
        #print('model theta', self.theta)
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

    @property
    def q(self):
        if self._q is None:
            self._q = 1 + ((self.data[:, 0] - self.parameters['t_0']) /
                           self.parameters['t_eff'])**2

        return self._q

    @property
    def magnification(self):
        """
        The magnification

        Parameters : None

        Returns :
            magnification: *float* or *np.ndarray*
                The magnification for each point
                specified by `u` in :py:attr:`~trajectory`.
        """
        if self.parameters['j'] == 1:
            self._magnification = 1. / np.sqrt(self.q)
        elif self.parameters['j'] == 2:
            self._magnification = 1. / np.sqrt(1. - (self.q / 2 + 1) ** (-2))
        else:
            raise ValueError('Invalid value for j.', self.parameters)

        return self._magnification
