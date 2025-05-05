import MulensModel
import sfit_minimizer as sfit

import exozippy.mmexofast as mmexo


class MulensFitter():
    """
    Parent class of the various microlensing model fitters

    datasets = *list* of data
    initial_model = *dict*
        initial parameters of the model.
    parameters_to_fit = *list* of parameters to be fitted. If None, fits all model parameters.
    mag_methods = *list*
        see MulensModel.model.mag_methods.
    verbose = *bool*
        default is False.
    """

    def __init__(self, datasets=None, initial_model=None, parameters_to_fit=None, mag_methods=None, verbose=False):
        self.datasets = datasets
        self.initial_model = initial_model
        if parameters_to_fit is None:
            self.parameters_to_fit = list(self.initial_model.keys())
        else:
            self.parameters_to_fit = parameters_to_fit

        self.mag_methods = mag_methods
        self.verbose = verbose

        self._best = None

    def run(self):
        pass

    @property
    def best(self):
        """
        *dict* containing the best-fit model parameters and chi2.
        """
        return self._best

    @best.setter
    def best(self, params_dict):
        self._best = params_dict

    @property
    def initial_model(self):
        return self._initial_model

    @initial_model.setter
    def initial_model(self, params_dict):
        if (params_dict is not None) and (not isinstance(params_dict, dict)):
            raise ValueError('initial_model must be set with either *None* or *dict*.')

        self._initial_model = params_dict


class SFitFitter(MulensFitter):
    """
    Fit a point lens model to the data using the SFit method.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self):
        event = MulensModel.Event(
            datasets=self.datasets, model=MulensModel.Model(self.initial_model))
        event.fit_fluxes()

        my_func = sfit.mm_funcs.PointLensSFitFunction(
            event, self.parameters_to_fit)

        initial_guess = [self.initial_model[key] for key in self.parameters_to_fit]
        for i in range(len(self.datasets)):
            initial_guess.append(event.fits[i].source_flux)
            initial_guess.append(event.fits[i].blend_flux)

        result = sfit.minimize(
            my_func, x0=initial_guess, tol=1e-5,
            options={'step': 'adaptive'}, verbose=self.verbose)

        if self.verbose:
            print(result)

        if not result.success:
            result = sfit.minimize(
                my_func, x0=initial_guess, tol=1e-5, max_iter=10000,
                options={'step': 0.001}, verbose=self.verbose)
            if self.verbose:
                print(result)

        best = my_func.event.model.parameters.parameters
        best['chi2'] = my_func.event.get_chi2()
        self.best = best


class AnomalyFitter(MulensFitter):

    def __init__(self, datasets=None, pspl_model_params=None, af_results=None, **kwargs):
        super().__init__(datasets=datasets, **kwargs)
        self.af_results = af_results
        self.pspl_model_params = None

    def estimate_initial_parameters(self):
        pass


class WidePlanetFitter(AnomalyFitter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def estimate_initial_parameters(self):
        dmag = self.get_dmag()
        if isinstance(self.pspl_params['t_E'], (astropy.units.Quantity)):
            t_E = self.pspl_params['t_E'].value
        elif isinstance(self.pspl_params['t_E'], (float)):
            t_E = self.pspl_params['t_E']
        else:
            raise TypeError(
                'Invalid type for t_E:', self.pspl_params['t_E'],
                type(self.pspl_params['t_E']))

        params = {
            't_0': self.pspl_params['t_0'], 'u_0': self.pspl_params['u_0'],
            't_E': t_E, 't_pl': self.best_af_grid_point['t_0'],
            'dt': 2. * self.best_af_grid_point['t_eff'], 'dmag': dmag}
        self.initial_model = estimate_params.get_wide_params(params)

    def run(self):
        if self.initial_model is None:
            self.estimate_initial_parameters()
