import numpy as np
import MulensModel
import sfit_minimizer


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
            dchi2s = [chi2s['1'] - chi2s['flat'], chi2s['2'] - chi2s['flat']]
            if verbose:
                print('t_0, t_eff, chi2s:', t_0, t_eff, chi2s)

            results.append(dchi2s)

        self.results = np.array(results)

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
