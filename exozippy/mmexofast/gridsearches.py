import numpy as np
import MulensModel


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

    def run(self):
        results = []
        for (t_0, t_eff) in zip(self.grid_t_0, self.grid_t_eff):
            chi2s = self.do_fits({'t_0': t_0, 't_eff': t_eff})
            results.append(chi2s)

        self.results = np.array(results)

    def do_fit(self, parameters):
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

        # Only fit the window if there's enough data to do so.
        if len(trimmed_datasets) >= 1:
            for j in [1, 2]:
                parameters['j'] = j
                model = EFModel(parameters)
                chi2 = 0
                for dataset in trimmed_datasets:
                    fit = MulensModel.FitData(dataset=dataset, model=model)
                    fit.fit_fluxes()
                    chi2 += fit.chi2

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
            # Need to check indexing
            index_1 = np.nanargmin(self.results[0, :])
            index_2 = np.nanargmin(self.results[1, :])
            if self.results[0, index_1[0]] < self.results[1, index_2[0]]:
                j = 1
                index = index_1
            else:
                j = 2
                index = index_2

            self._best = {'t_0': self.grid_t_0[index],
                          't_eff': self.grid_t_eff[index],
                          'j': j,
                          'chi2': self.results[j-1, index]}

        return self._best


class EFModel(MulensModel.Model):

    def __init__(self, parameters=None):
        self.parameters = parameters

    def get_magnification(self, time):
        time = np.atleast_1d(time)
        magnification_curve = EFMagnificationCurve(
            time, parameters=self.parameters)
        return magnification_curve.get_magnification()


class EFMagnificationCurve(
    MulensModel.PointSourcePointLensMagnification):

    def __init__(self, time, parameters=None):
        self.time = time
        self.parameters = parameters
        self._q = None
        self._magnification = None

    @property
    def q(self):
        if self._q is None:
            self._q = 1 + ((self.time - self.parameters['t_0']) /
                           self.parameters['t_eff'])**2

        return self._q

    def get_magnification(self):
        """
        Calculate the magnification

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

