import MulensModel
import numpy as np
import sfit_minimizer as sfit
import emcee
from multiprocessing import Pool, cpu_count
import os

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

    def __init__(self, datasets=None, initial_model=None, parameters_to_fit=None, sigmas=None, mag_methods=None, verbose=False, pool=None):
        self._initial_model = None
        self._best = None

        self.datasets = datasets
        self.initial_model = initial_model
        self.parameters_to_fit = parameters_to_fit
        self.sigmas = sigmas
        self.mag_methods = mag_methods
        self.verbose = verbose
        self.pool = pool


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
            self._parameters_to_fit = list(self.initial_model.keys())

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
    default_emcee_settings = {
        'n_walkers': 40, 'n_burn': 500, 'n_steps': 1000,
        'temperature': 1., 'max_steps': 10000, 'progress': False}

    def __init__(self, datasets=None, anomaly_lc_params=None, **kwargs):
        super().__init__(**kwargs)
        self.anomaly_lc_params = anomaly_lc_params
        self.datasets = datasets
        #self.mag_methods = None

    def estimate_initial_parameters(self):
        pass

    def get_parameter_name(self, parameter):
        if 'log_' in parameter:
            key = parameter[4:]
        else:
            key = parameter

        return key

    def set_event_parameters(self, theta, event):
        raise NotImplementedError('You need to implement set_event_parameters() for this class.')
        # return mm.Event()

    def make_starting_vector(self):
        raise NotImplementedError('You need to implement make_starting_vector() for this class.')
        # return []

    def ln_like(self, theta):
        """ likelihood function """

        self.event = theta
        try:
            chi2 = self.event.get_chi2()
            if 'temperature' in self.emcee_settings.keys(): #['temperature'] is not None:
                chi2 /= self.emcee_settings['temperature'] ** 2
        except:
            return -np.inf

        #print(chi2, theta)
        return -0.5 * chi2

    def ln_prior(self, theta):
        """priors - we only reject obviously wrong models"""
        for key, value in zip(self.parameters_to_fit, theta):
            if ((key == 't_E') or (key == 'rho') or (key == 'q') or (key == 's')) and (value <= 0.):
                return np.inf
            elif (key == 'q') and (value >= 1.):
                return np.inf  # Should be fixed in MMv3.

        return 0.0

    def ln_prob(self, theta):
        """ combines likelihood and priors"""
        ln_prior_ = self.ln_prior(theta)
        if not np.isfinite(ln_prior_):
            return -np.inf
        ln_like_ = self.ln_like(theta)

        # In the cases that source fluxes are negative we want to return
        # these as if they were not in priors.
        if np.isnan(ln_like_):
            return -np.inf

        return ln_prior_ + ln_like_


class WidePlanetFitter(AnomalyFitter):

    def __init__(self, emcee_settings=None, **kwargs):
        super().__init__(**kwargs)
        if not ('parameters_to_fit' in kwargs.keys()):
            self.parameters_to_fit = ['t_0', 'u_0', 't_E', 'log_rho', 'log_s', 'log_q', 'd_xsi']
            self.sigmas = [0.1, 0.01, 0.01, 0.01, 0.001, 0.5, 0.0005]
        elif self.sigmas is None:
            raise AttributeError('If parameters_to_fit is set, sigmas must also be set.')

        if emcee_settings is None:
            emcee_settings = AnomalyFitter.default_emcee_settings

        for key in AnomalyFitter.default_emcee_settings.keys():
            if key not in list(emcee_settings.keys()):
                emcee_settings[key] = AnomalyFitter.default_emcee_settings[key]

        if 'ndim' not in list(emcee_settings.keys()):
            emcee_settings['n_dim'] = len(self.parameters_to_fit)

        self.emcee_settings = emcee_settings

        self._best = None
        self._event = None
        self._initial_guess = None

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

    def initialize_event(self):
        if self.initial_model is None:
            self.estimate_initial_parameters()

        model = MulensModel.Model(parameters=self.initial_model)
        model.default_magnification_method = 'point_source_point_lens'
        if self.mag_methods is None:
            raise AttributeError(
                'self.mag_methods is not set! Either pass it as an kwarg to __init__ or calculate it using estimate_initial_parameters(). ')
        else:
            model.set_magnification_methods(self.mag_methods)

        self._event = MulensModel.Event(datasets=self.datasets, model=model)

    def make_starting_vector(self):
        starting_vector = []
        for i in np.arange(self.emcee_settings['n_walkers']):
            test_vector = self.initial_guess + np.random.randn(self.emcee_settings['n_dim']) * self.sigmas
            starting_vector.append(test_vector)

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
            else:
                value = parameters.__getattribute__(key)

            if key != parameter:
                value = np.log10(value)

            initial_guess.append(value)

        return initial_guess

    def run(self, verbose=False):
        self.initialize_event()
        starting_vector = self.make_starting_vector()

        if self.pool:
            ncpu = cpu_count()
            print("{0} CPUs".format(ncpu))
            os.environ["OMP_NUM_THREADS"] = "1"
            pool = Pool()
            sampler = emcee.EnsembleSampler(
                self.emcee_settings['n_walkers'], self.emcee_settings['n_dim'], self.ln_prob,
                pool=pool)
        else:
            sampler = emcee.EnsembleSampler(
                self.emcee_settings['n_walkers'], self.emcee_settings['n_dim'], self.ln_prob)

        sampler.run_mcmc(
            starting_vector, self.emcee_settings['n_steps'],
            progress=self.emcee_settings['progress'])

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
        self.event = samples[best_index, :]

        self.best = self.event.model.parameters.parameters
        self.best['chi2'] = self.event.get_chi2()

    @property
    def initial_guess(self):
        if self._initial_guess is None:
            self._initial_guess = self.make_emcee_vector_from_ModelParameters(
                MulensModel.ModelParameters(self.initial_model))

        return self._initial_guess

    @property
    def event(self):
        return self._event

    @event.setter
    def event(self, theta):
        if self._event is None:
            raise AttributeError('Event has not been created. Run initialize_event() first!')

        d_xsi = None
        for parameter, value in zip(self.parameters_to_fit, theta):
            key = self.get_parameter_name(parameter)
            if key != parameter:
                value = 10. ** value

            if key == 'd_xsi':
                d_xsi = value
            else:
                self._event.model.parameters.__setattr__(key, value)

        if d_xsi is not None:
            self._event.model.parameters.alpha = self.get_alpha_from_d_xsi(
                self._event.model.parameters, d_xsi)
