import MulensModel
import numpy as np
import sfit_minimizer as sfit
import emcee

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

    @property
    def parameters_to_fit(self):
        if self._parameters_to_fit is None:
            self.parameters_to_fit = list(self.initial_model.keys())

        return self._parameters_to_fit

    @parameters_to_fit.setter
    def parameters_to_fit(self, params_dict):
        if (params_dict is not None) and (not isinstance(params_dict, (list, tuple))):
            raise ValueError('initial_model must be set with either *None* or *list* or *tuple*.')

        self._parameters_to_fit = params_dict


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

    def __init__(self, datasets=None, anomaly_lc_params=None, **kwargs):
        self.anomaly_lc_params = anomaly_lc_params
        self.datasets = datasets
        self.mag_methods = None
        self._initial_model = None
        self._best = None

    def estimate_initial_parameters(self):
        pass

    def get_parameter_name(self, parameter):
        if 'log_' in parameter:
            key = parameter[4:]
        else:
            key = parameter

        return key

    def set_event_parameters(self, theta, event, parameters_to_fit):
        raise NotImplementedError('You need to implement set_event_parameters() for this class.')
        return event

    def ln_like(self, theta, event, parameters_to_fit, temperature):
        """ likelihood function """

        event = self.set_event_parameters(theta, event, parameters_to_fit)
        try:
            chi2 = event.get_chi2()
            if temperature is not None:
                chi2 /= temperature ** 2
        except:
            return -np.inf

        #print(chi2, theta)
        return -0.5 * chi2

    def ln_prior(self, theta, parameters_to_fit):
        """priors - we only reject obviously wrong models"""
        for key, value in zip(parameters_to_fit, theta):
            if ((key == 't_E') or (key == 'rho') or (key == 'q') or (key == 's')) and (value <= 0.):
                return np.inf
            elif (key == 'q') and (value >= 1.):
                return np.inf  # Should be fixed in MMv3.

        return 0.0

    def ln_prob(self, theta, event, parameters_to_fit, temperature):
        """ combines likelihood and priors"""
        ln_prior_ = self.ln_prior(theta, parameters_to_fit)
        if not np.isfinite(ln_prior_):
            return -np.inf
        ln_like_ = self.ln_like(theta, event, parameters_to_fit, temperature)

        # In the cases that source fluxes are negative we want to return
        # these as if they were not in priors.
        if np.isnan(ln_like_):
            return -np.inf

        return ln_prior_ + ln_like_


default_emcee_settings = {'n_walkers': 40, 'n_burn': 500, 'n_steps': 1000,
                              'temperature': 1.}


class WidePlanetFitter(AnomalyFitter):

    def __init__(self, emcee_settings=None, **kwargs):
        super().__init__(**kwargs)
        self.parameters_to_fit = ['t_0', 'u_0', 't_E', 'log_rho', 'log_s', 'log_q', 'd_xsi']
        self.sigmas = [0.1, 0.01, 0.01, 0.01, 0.001, 0.5, 0.0005]
        if emcee_settings is None:
            emcee_settings = default_emcee_settings

        for key in default_emcee_settings.keys():
            if key not in list(emcee_settings.keys()):
                emcee_settings[key] = default_emcee_settings[key]

        if 'ndim' not in list(emcee_settings.keys()):
            emcee_settings['n_dim'] = len(self.parameters_to_fit)

        self.emcee_settings = emcee_settings
        self._initial_guess = None
        #print('datasets', self.datasets)

    def estimate_initial_parameters(self):
        binary_params = mmexo.estimate_params.get_wide_params(self.anomaly_lc_params)
        self.mag_methods = binary_params.mag_methods
        self.initial_model = binary_params.ulens

    def get_alpha_from_d_xsi(self, parameters, d_xsi):
        s = parameters.s
        q = parameters.q
        u_0 = parameters.u_0
        xloc_prim = s - 1 / s
        a_prim = s * q / (1 + q)
        xloc = xloc_prim - a_prim
        sin_alpha = u_0 / (xloc + d_xsi)
        alpha = -np.rad2deg(np.arcsin(sin_alpha))

        return alpha

    def set_event_parameters(self, theta, event, parameters_to_fit):
        # print('before set:\n', event.model)
        d_xsi = None
        for parameter, value in zip(parameters_to_fit, theta):
            key = self.get_parameter_name(parameter)
            if key != parameter:
                value = 10. ** value

            # print(key, parameter, value)
            if key == 'd_xsi':
                d_xsi = value
            else:
                event.model.parameters.__setattr__(key, value)

        if d_xsi is not None:
            event.model.parameters.alpha = self.get_alpha_from_d_xsi(event.model.parameters, d_xsi)

        # print('after set\n', event.model)
        return event

    def make_starting_vector(self):
        #t_pl = self.anomaly_lc_params['t_pl']
        #t_range = [t_pl - self.initial_model['t_E'] / 4, t_pl + self.initial_model['t_E'] / 4]
        starting_vector = []
        for i in np.arange(self.emcee_settings['n_walkers']):
            test_vector = self.initial_guess + np.random.randn(self.emcee_settings['n_dim']) * self.sigmas
            starting_vector.append(test_vector)
            #random_params_dict = {}
            #for parameter, value in zip(self.parameters_to_fit, test_vector):
            #    key = self.get_parameter_name(parameter)
            #    if key == parameter:
            #        random_params_dict[key] = value
            #    else:
            #        random_params_dict[key] = 10.**value
            #
            #for parameter in ['t_0', 'u_0', 't_E', 'rho', 's', 'q', 'alpha']:
            #    if not (parameter in list(random_params_dict.keys())):
            #        random_params_dict[parameter] = self.initial_model[parameter]
            #
            #random_model_params = MulensModel.ModelParameters(random_params_dict)
            ## update to go through the caustic:
            #random_anomaly_lc_params = {key: value for key, value in self.anomaly_lc_params.items()}
            #for key in ['t_0', 'u_0', 't_E']:
            #    random_anomaly_lc_params[key] = random_params_dict[key]
            #
            #random_causticx_params = mmexo.estimate_params.get_wide_params(random_anomaly_lc_params)
            #random_model_params.s = random_causticx_params['s']
            #random_model_params.alpha = random_causticx_params['alpha']
            #
            #starting_vector.append(
            #    self.make_emcee_vector_from_ModelParameters(random_model_params))
            #
        return starting_vector

    def make_emcee_vector_from_ModelParameters(self, parameters):
        initial_guess = []
        for parameter in self.parameters_to_fit:
            key = self.get_parameter_name(parameter)
            if key == 'd_xsi':
                u_0 = parameters.u_0
                alpha = parameters.alpha
                s = parameters.s
                q = parameters.q

                xloc_prim = s - 1 / s
                a_prim = s * q / (1 + q)
                xloc = xloc_prim - a_prim
                sin_alpha = np.sin(np.deg2rad(alpha))
                d_xsi = -u_0 / sin_alpha - xloc
                value = d_xsi
                #print(u_0, xloc, sin_alpha, alpha)
                #print('initial d_xsi', d_xsi)
            else:
                value = parameters.__getattribute__(key)

            if key != parameter:
                value = np.log10(value)

            initial_guess.append(value)

        return initial_guess

    def run(self, verbose=False):
        if self.initial_model is None:
            self.estimate_initial_parameters()

        #print(self.initial_model)
        model = MulensModel.Model(parameters=self.initial_model)
        model.default_magnification_method = 'point_source_point_lens'
        model.set_magnification_methods(self.mag_methods)
        event = MulensModel.Event(datasets=self.datasets, model=model)

        starting_vector = self.make_starting_vector()

        sampler = emcee.EnsembleSampler(
            self.emcee_settings['n_walkers'], self.emcee_settings['n_dim'], self.ln_prob,
            args=(event, self.parameters_to_fit, self.emcee_settings['temperature']))
        sampler.run_mcmc(starting_vector, self.emcee_settings['n_steps'])

        # Remove burn-in samples and reshape:
        samples = sampler.chain[:, self.emcee_settings['n_burn']:, :].reshape((-1, self.emcee_settings['n_dim']))

        # Results:
        results = np.percentile(samples, [16, 50, 84], axis=0)
        if verbose:
            print("Fitted parameters:")
            for i in range(self.emcee_settings['n_dim']):
                r = results[1, i]
                print("${:.5f}^{{+{:.5f}}}_{{-{:.5f}}}$ &".format(
                    r, results[2, i] - r, r - results[0, i]))

        prob = sampler.lnprobability[:, self.emcee_settings['n_burn']:].reshape((-1))
        best_index = np.argmax(prob)
        # self.best_chi2 = prob[best_index] / -0.5
        event = self.set_event_parameters(samples[best_index, :], event, self.parameters_to_fit)

        self.best = event.model.parameters.parameters
        self.best['chi2'] = event.get_chi2()

    @property
    def initial_guess(self):
        if self._initial_guess is None:
            self._initial_guess = self.make_emcee_vector_from_ModelParameters(
                MulensModel.ModelParameters(self.initial_model))

        return self._initial_guess
