#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Created by Luca Campiani in January 2024
# Updated by Jennifer Yee, May 2025

import MulensModel
import MulensModel as mm
import matplotlib.pyplot as plt
import numpy as np
#import warnings
import copy

import exozippy.mmexofast as mmexo

# In[ ]:


def get_PSPL_params(ef_grid_point, datasets, verbose=False):
    t_0 = ef_grid_point['t_0']
    u_0s = [0.01, 0.1, 0.3, 1.0, 1.5]
    t_Es = [1., 3., 10., 20., 40., 100.]
    best_chi2 = np.inf
    best_params = None
    for u_0 in u_0s:
        for t_E in t_Es:
            params = {'t_0': t_0, 't_E': t_E, 'u_0': u_0}
            event = MulensModel.Event(
                datasets=datasets, model=MulensModel.Model(params))
            if event.get_chi2() < best_chi2:
                best_params = params
                best_chi2 = event.chi2

            #print(u_0, t_E, event.chi2)

    return best_params


class BinaryLensParams():
    """
    A class for managing parameters related to binary lens models.

    Attributes:
        ulens: *object*
            Object representing the lens model.
        
        mag_method: *object*
            Object representing the magnification method.            
    """
    def __init__(self, ulens):
        self.ulens = ulens
        self.mag_methods = None
        
    def set_mag_method(self, params):
        """
        Sets the magnification calculation method based on input parameters.

        Arguments :
            params: *dictionary*
                Initilal parameters.
            
                - 't_0' (*float*): Time of maximum magnification.
                - 'u_0' (*float*): Impact parameter.
                - 't_E' (*float*): Einstein crossing time.
                - 't_pl' (*float*): Time at which to compute the wide model parameters.
                - 'dt' (*float*): Duration of the anomaly
                - 'dmag' (*float*): Magnitude difference of the perturbation

        Returns :
           None
        """
        #t1 = params['t_pl'] - (5 * params['dt'])
        #t2 = params['t_pl'] + (5 * params['dt'])
        #self.mag_method = [t1, 'VBBL', t2]
        #print(params)

        t_E = params['t_E']
        t_0 = params['t_0']
        t_pl = params['t_pl']
        t_star = params['dt'] / 2.
        self.mag_methods = [
            np.min((t_0 - t_E, t_pl - t_E / 2., t_pl - 20. * t_star)),
            'point_source',
            t_pl - 10. * t_star,
            'hexadecapole',
            t_pl - 5. * t_star,
            'VBBL',
            t_pl + 5. * t_star,
            'hexadecapole',
            t_pl + 10. * t_star,
            'point_source',
            np.max((t_0 + t_E, t_pl + t_E / 2., t_pl + 20. * t_star))]


def get_wide_params(params, limit='GG97'):
    """
    Transform initial parameters into wide model parameters.

    Arguments :
        params: *dictionary*
            Initial parameters.
            
            - 't_0' (*float*): Time of maximum magnification.
            - 'u_0' (*float*): Impact parameter.
            - 't_E' (*float*): Einstein crossing time.
            - 't_pl' (*float*): Time at which to compute the wide model parameters.
            - 'dt' (*float*): Duration of the anomaly
            - 'dmag' (*float*): Magnitude difference of the perturbation

        limit: *str*
            Method to use for estimating *rho* and *q*.

    Returns :
        wide_params : *BinaryLensParams*
             Wide model parameters for the binary lens.
    """
    estimator = WidePlanetParameterEstimator(params, limit=limit)
    
    return estimator.binary_params


def get_possible_bump_anomaly_solutions(params):
    solutions = {}

    # large rho limit
    estimator = WidePlanetParameterEstimator(params, limit='GG97')
    solutions['GG97'] = estimator.calc_binary_ulens_params()

    return solutions


class ParameterEstimator():

    def __init__(self, params, limit=None):
        self.params = params
        self.limit = limit

        self._tau_pl, self._u_pl = None, None
        self._s, self._alpha = None, None
        self._q = None
        self._rho = None
        self._binary_params = None

    def get_binary_lens_params(self):
        pass

    def get_rho(self):
        if self.limit == 'dwarf':
            return 0.001
        elif self.limit == 'giant':
            return 0.05
        elif self.limit == 'point':
            return None
        else:
            raise ValueError('Your limit for calculating rho is not implemented: ', self.limit)

    @property
    def binary_params(self):
        if self._binary_params is None:
            self._binary_params = self.get_binary_lens_params()

        return self._binary_params

    @property
    def t_0(self):
        return self.params['t_0']

    @property
    def u_0(self):
        return self.params['u_0']

    @property
    def t_E(self):
        return self.params['t_E']

    @property
    def tau_pl(self):
        if self._tau_pl is None:
            self._tau_pl = (self.params['t_pl'] - self.params['t_0']) / self.params['t_E']

        return self._tau_pl

    @property
    def u_pl(self):
        if self._u_pl is None:
            self._u_pl = np.sqrt(self.params['u_0'] ** 2 + self.tau_pl ** 2)

        return self._u_pl

    def _correct_alpha(self, alpha):
        while alpha > 360.:
            alpha -= 360.

        while alpha < -360:
            alpha += 360.

        return alpha

    @property
    def alpha(self):
        if self._alpha is None:
            alpha = np.pi - np.arctan2(self.params['u_0'], self.tau_pl)
            alpha = np.rad2deg(alpha)
            self._alpha = self._correct_alpha(alpha)

        return self._alpha

    @property
    def rho(self):
        if self._rho is None:
            self._rho = self.get_rho()

        return self._rho

    @rho.setter
    def rho(self, value):
        self._rho = value


class WidePlanetParameterEstimator(ParameterEstimator):

    def __init__(self, params, limit='GG97'):
        super().__init__(params, limit=limit)
        self._delta_A = None
        self._a_pspl = None

    def get_rho(self):
        if self.limit == 'GG97':
            rho = self.params['dt'] / self.params['t_E'] / 4.
        else:
            rho = super().get_rho()

        return rho

    def calc_binary_ulens_params(self):
        new_params = {'t_0': self.t_0, 'u_0':self.u_0, 't_E': self.t_E, 's': self.s, 'alpha': self.alpha}
        rho = self.rho
        if rho is not None:
            new_params['rho'] = rho

        new_params['q'] = self.q

        return new_params

    def get_binary_lens_params(self):
        binary_ulens_params = self.calc_binary_ulens_params()
        out = BinaryLensParams(binary_ulens_params)
        out.set_mag_method(self.params)
        return out

    @property
    def s(self):
        if self._s is None:
            u = self.u_pl
            self._s = 0.5 * (np.sqrt(u ** 2 + 4) + u)
        return self._s

    @property
    def q(self):
        if self._q is None:
            self._q = 0.5 * np.abs(self.delta_A) * (self.rho ** 2)

        return self._q

    @property
    def a_pspl(self):
        if self._a_pspl is None:
            self._a_pspl = (self.u_pl**2 + 2.) / np.sqrt(self.u_pl**2 * (self.u_pl**2 + 4.))

        return self._a_pspl

    @property
    def delta_A(self):
        """
        Might want to add an option to calculate delta_A using PSPL fitted fs and fb.
        Current calculation assumes fb=0. This could be a problem if fb is large, e.g. OB180383.
        :return:
        """
        if self._delta_A is None:
            self._delta_A = self.a_pspl * (10.**(self.params['dmag'] / -2.5) - 1.)

        return self._delta_A

# In[ ]:


def get_close_params(params, q=None, rho=None):
    """
    Transform initial parameters into two close model parameters for a binary lens. One for upper and one for lower caustics. 

    Arguments:
        params: *dictionary*
            Initial parameters.

            - 't_0' (*float*): Time of maximum magnification.
            - 'u_0' (*float*): Impact parameter.
            - 't_E' (*float*): Einstein crossing time.
            - 't_pl' (*float*): Time at which to compute the close model parameters.
            - 'dt' (*float*), optional: Duration of the anomaly
            - 'q' (*float*): trial value of q for calculating the caustic,
                default is 0.004
            - 'rho' (*float*): value of rho for the model. If 'dt' is specified,
                'rho' is calculated from 'dt'. If neither are specified,
                default is 0.001.

    Returns:
        lens1, lens2 : *tuple of BinaryLensParams*
            Two instances of BinaryLensParams representing close model parameters.
    """
    estimator_upper = CloseUpperPlanetParameterEstimator(params=params, q=q)
    estimator_lower = CloseLowerPlanetParameterEstimator(params=params, q=q)

    return estimator_upper.binary_params, estimator_lower.binary_params

    #if q is None:
    #    q = 0.0040
    #
    #tau = (params['t_pl'] - params['t_0']) / params['t_E']
    #u = np.sqrt(params['u_0']**2 + tau**2)
    #
    #s = 0.5 * (np.sqrt(u**2 + 4) - u)
    #
    #eta_not = (q**0.5 / s) * (1 / (np.sqrt(1 + s**2)) + np.sqrt(1 - s**2))
    #mu = np.arctan2(eta_not, (s - 1 / s) / (1 + q)) # correction for primary --> COM
    #phi = np.arctan2(tau, params['u_0'])
    #
    #alpha1 = np.pi / 2 - mu - phi
    #alpha2 = alpha1 + 2 * mu
    #
    #alpha1_deg = correct_alpha(-np.rad2deg(alpha1) + 180.)
    #alpha2_deg = correct_alpha(-np.rad2deg(alpha2) + 180.)
    #
    #if 'dt' in params.keys():
    #    rho = params['dt'] / params['t_E'] / 2.
    #elif 'rho' not in params.keys():
    #    rho = 0.001
    #
    #new_params1 = {'t_0': params['t_0'],
    #            'u_0': params['u_0'],
    #            't_E': params['t_E'],
    #            's': s,
    #            'q': q,
    #            'rho': rho,
    #            'alpha': alpha1_deg}
    #
    #new_params2 = {'t_0': params['t_0'],
    #            'u_0': params['u_0'],
    #            't_E': params['t_E'],
    #            's': s,
    #            'q': q,
    #            'rho': rho,
    #            'alpha': alpha2_deg}
    #
    #out1 = BinaryLensParams(new_params1)
    #out2 = BinaryLensParams(new_params2)
    #out1.set_mag_method(params)
    #out2.set_mag_method(params)
    #
    #return out1, out2


class CloseUpperPlanetParameterEstimator(WidePlanetParameterEstimator):

    def __init__(self, params, limit='GG97', q=None):
        super().__init__(params, limit=limit)
        if q is None:
            q = 0.004

        self._q = q
        self._eta_not, self._mu, self._phi = None, None, None
        #self._alpha_upper, self._alpha_lower = None, None

    def setup_close_ulens_params(self):
        new_params = {'t_0': self.t_0,
                      'u_0': self.u_0,
                      't_E': self.t_E,
                      's': self.s,
                      'q': self.q}

        if self.rho is not None:
            new_params['rho'] = self.rho

        return new_params

    def calc_binary_params(self):
        new_params = self.setup_close_ulens_params()
        new_params['alpha'] = self.alpha

        return new_params

    def get_binary_lens_params(self):
        binary_ulens_params = self.calc_binary_ulens_params()
        binary_params = BinaryLensParams(binary_ulens_params)
        binary_params.set_mag_method(self.params)

        return binary_params

    @property
    def binary_params(self):
        if self._binary_params is None:
            self._binary_params = self.get_binary_lens_params()

        return self._binary_params

    @property
    def s(self):
        if self._s is None:
            u = self.u_pl
            self._s = 0.5 * (np.sqrt(u**2 + 4) - u)

        return self._s

    @property
    def q(self):
        return self._q

    @property
    def eta_not(self):
        if self._eta_not is None:
            self._eta_not = (self.q**0.5 / self.s) * (1 / (np.sqrt(1 + self.s**2)) + np.sqrt(1 - self.s**2))

        return self._eta_not

    @property
    def mu(self):
        if self._mu is None:
            self._mu = np.arctan2(self.eta_not, (self.s - 1 / self.s) / (1 + self.q))
            # correction for primary --> COM

        return self._mu

    @property
    def phi(self):
        if self._phi is None:
            self._phi = np.arctan2(self.u_0, self.tau_pl)

        return self._phi

    @property
    def alpha(self):
        if self._alpha is None:
            alpha = 180. - np.rad2deg(self.phi - self.mu)
            self._alpha = self._correct_alpha(alpha)

        return self._alpha


class CloseLowerPlanetParameterEstimator(CloseUpperPlanetParameterEstimator):

    @property
    def alpha(self):
        if self._alpha is None:
            alpha = 180. - np.rad2deg(self.phi + self.mu)
            self._alpha = self._correct_alpha(alpha)

        return self._alpha


def model_pspl_mag_at_pl(params):
    """
    Gets the magnification at second lense time assuming point lense model.

    Arguments :
        params: *dictionary*
            Initilal parameters.
            
            - 't_0' (*float*): The time of maximum magnification.
            - 'u_0' (*float*): The impact parameter.
            - 't_E' (*float*): The Einstein crossing time.
            - 't_pl' (*float*): The time at which to compute the magnification.
            
    Returns :
        mag :*float*
             Magnification at the specified time 't_pl' based on the point lens model.
            
    """
    model1 = mm.Model({'t_0': params['t_0'], 
                       'u_0': params['u_0'], 
                       't_E': params['t_E']})
    return model1.get_magnification(params['t_pl'])


# In[ ]:


class BinarySourceParams():
    """
    A class for managing parameters related to binary source models. Derived from equation 2.5 from Gaudi 1998.
   
   Attributes:
        ulens: *object*
            Object representing the underlying lens model.
            
        source_flux_ratio: *object*
            Object representing the source flux ratio.  
            
        set_source_flux_ratio(params):
            Sets the source flux ratio based on input parameters.
  
    """
    def __init__(self, ulens):
        self.ulens = ulens
        self.source_flux_ratio = None
        
    def set_source_flux_ratio(self, params):
        """
         Sets the source flux ratio based on input parameters.

        Arguments :
            params: *dictionary*
                Initilal parameters.
            
                - 't_0' (*float*): Time of maximum magnification.
                - 'u_0' (*float*): Impact parameter.
                - 't_E' (*float*): Einstein crossing time.
                - 't_pl' (*float*): Time at which to compute the wide model parameters.
                - 'dt' (*float*): Duration of the anomaly
                - 'dmag' (*float*): Magnitude difference of the perturbation

        Returns :
           None
        """
        A1 = model_pspl_mag_at_pl(params)
        u_0_2 = params["dt"] / (12**0.5 * params["t_E"])
        e = params["dmag"] * u_0_2 * A1
        self.source_flux_ratio = e


# In[ ]:


def get_binary_source_params(params):
    """
    Transform initial parameters into binary source model parameters.

    Arguments:
        params: *dictionary*
            Initial parameters.

            - 't_0' (*float*): Time of maximum magnification for the first lens.
            - 'u_0' (*float*): Impact parameter for the first lens.
            - 't_pl' (*float*): Time at which to compute the source flux ratio.
            - 'dt' (*float*): Duration of the anomaly
            - 't_E' (*float*): Einstein crossing time.
            - 'dmag' (*float*): Magnitude difference of the perturbation

    Returns:
        source_params : *BinarySourceParams*
            Binary source model parameters.
    """
    u_0_2 = params["dt"] / (12**0.5 * params["t_E"])
    new_params= {'t_0_1': params['t_0'],
              'u_0_1': params['u_0'],
              't_0_2': params['t_pl'],
              'u_0_2': u_0_2,
              't_E': params['t_E']}
    out = BinarySourceParams(new_params)
    out.set_source_flux_ratio(params)
    return out


class AnomalyPropertyEstimator():
    # The old version revised the PSPL parameters after masking the anomaly.
    # Could consider whether it would be a good idea to reimplement that.

    def __init__(self, datasets=None, pspl_params=None, af_results=None, mask_type='t_eff', n_mask=3):
        if isinstance(datasets, MulensModel.MulensData):
            datasets = [datasets]

        self.datasets = datasets
        self.pspl_params = pspl_params
        self.af_results = af_results
        self.n_mask = n_mask

        self.anom_t_range_af = self.af_results['t_0'] + self.n_mask * np.array(
            [-1, 1]) * self.af_results['t_eff']

        self._peak_index = None
        self._peak_dflux = None
        self._t_start = None
        self._t_stop = None

        self._pspl_event = None
        self._source_flux = None
        self._blend_flux = None

        self._anom_type = None
        self._anom_index = None
        self._sorted_index = None
        self._times = None
        self._scaled_fluxes = None
        self._scaled_residuals = None
        self._chi2s = None
        self._expected_model_fluxes = None

    def get_pspl_event(self):
        event = mm.Event(datasets=self.datasets,
                         model=mm.Model(self.pspl_params))
        event.fit_fluxes()
        return event

    def get_anom_type(self):
        n_pts = np.sum(self.anom_index)
        sigmas = np.sign(self.residuals) * np.sqrt(self.chi2s)
        med, std = np.nanmedian(sigmas), np.nanstd(sigmas)
        #print('sigma dist', med, std, np.percentile(sigmas, q=[0, 1, 2, 98, 99, 100]))
        #plt.figure()
        #plt.hist(sigmas, bins=int(n_pts/40))
        #plt.axvline(med, color='black')
        #plt.axvline(med - std, color='black')
        #plt.axvline(med + std, color='black')
        #plt.gca().minorticks_on()
        #plt.xlabel('sigmas')

        if n_pts > 10:
            max_res = np.percentile(sigmas, q=98)
            min_res = np.percentile(sigmas, q=2)
        else:
            min_res, max_res = -np.inf, np.inf

        #print('res', n_pts, min_res, max_res)
        if (min_res < 0) and (np.abs(min_res) > max_res):
            return 'negative'

        top_index = (sigmas > 0) & (sigmas < max_res)
        bot_index = (sigmas < 0) & (sigmas > min_res)
        #print('n', np.sum(bot_index), np.sum(top_index))
        if np.sum(top_index) == 0:
            return 'negative'
        elif np.sum(bot_index) == 0:
            return 'positive'
        else:
            top_chi2 = np.sum(self.chi2s[top_index])
            bot_chi2 = np.sum(self.chi2s[bot_index])
            #print('chi2', bot_chi2, top_chi2)
            if top_chi2 > bot_chi2:
                return 'positive'
            else:
                return 'negative'

    def set_anom_prop(self):
        self._peak_dflux, self._peak_index, self._t_start, self._t_stop = self.find_extremum(
            method='rolling')

    def get_anom_prop(self):
        if (self.peak_dflux is None) or (self.t_start is None) or (self.t_stop is None):
            self.set_anom_prop()

        return self.peak_dflux, self.peak_index, self.t_start, self.t_stop, self.peak_width

    def _find_extremum_with_simple_line(self):
        peak_index = np.nanargmax(self.chi2s)
        peak_dflux = self.residuals[peak_index]
        t_start, t_stop = None, None
        for i in [1, -1]:
            slope = (self.sorted_times[peak_index] - self.sorted_times[i]) / (self.peak_dflux - self.residuals[i])
            intercept = self.sorted_times[peak_index] - slope * peak_dflux
            t = slope * peak_dflux / 2. + intercept
            if i == 1:
                t_start = t
            else:
                t_stop = t

        return peak_dflux, peak_index, t_start, t_stop

    def _get_window_size(self):
        n_pts = np.sum(self.anom_index)

        if n_pts < 10:
            window_size = 1
        elif n_pts < 50:
            window_size = int(np.floor(n_pts / 10))
        elif n_pts < 100:
            window_size = int(np.floor(n_pts / 20))
        elif n_pts < 500:
            window_size = int(np.floor(n_pts / 50))
        else:
            window_size = int(np.floor(n_pts / 100))

        #window_size = int(np.floor(n_pts / 10))
        #print('points', n_pts, 'window', window_size)
        return window_size

    def _find_extremum_with_rolling_mean(self):
        window_size = self._get_window_size()
        kernel = np.ones(window_size) / window_size
        #print('points:', np.sum(t_index), 'window:', window_size,
        #      'half window:', int(window_size / 2))

        if (window_size > 0) and (window_size < np.sum(self.anom_index)):
            chi2_rolling_mean = np.convolve(self.chi2s, kernel, mode='same')
            peak_index = np.argmax(chi2_rolling_mean)

            res_rolling_mean = np.convolve(self.residuals, kernel, mode='same')
            #print('rolling mean:', len(res_rolling_mean))

            peak_dflux = res_rolling_mean[peak_index]
            # start_dflux = res_rolling_mean[half_window]
            # end_dflux = res_rolling_mean[-half_window]

            if peak_dflux > 0:
                half_anomaly = res_rolling_mean > (peak_dflux / 2.)
            else:
                half_anomaly = res_rolling_mean < (peak_dflux / 2.)
                # raise NotImplementedError('negative perturbations not implemented')

            t_start = np.min(self.sorted_times[half_anomaly])
            t_stop = np.max(self.sorted_times[half_anomaly])

            return peak_dflux, peak_index, t_start, t_stop
        else:
            return self._find_extremum_with_simple_line()

    def find_extremum(self, method=None):
        if method == 'rolling':
            return self._find_extremum_with_rolling_mean()

    def get_anomaly_lc_parameters(self):
        self.set_anom_prop()
        params = {key: value for key, value in self.pspl_params.items()}
        params['dmag'] = self.dmag
        params['dt'] = self.t_stop - self.t_start
        params['t_pl'] = np.mean((self.t_start, self.t_stop))

        return params

    def _plot_peak_lines(self):
        plt.axvline(self.peak_time, color='darkgray', zorder=10, linestyle=':')
        plt.axvline(self.t_start, color='darkgray')
        plt.axvline(self.t_stop, color='darkgray')
        #plt.axvline(self.peak_time - self.peak_width / 2., color='darkgray')
        #plt.axvline(self.peak_time + self.peak_width / 2., color='darkgray')

    def _plot_af_lines(self):
        plt.axvline(self.af_results['t_0'] +
                    self.af_results['t_eff'], color='black')
        plt.axvline(self.af_results['t_0'] -
                    self.af_results['t_eff'], color='black')

    def _setup_anom_xaxis(self):
        plt.xlim(self.af_results['t_0'] + 5. * np.array([-1, 1]) *
                 self.af_results['t_eff'])
        plt.xlabel('time')

    def plot_residuals(self):
        plt.figure()
        plt.title(self.anom_type)
        plt.axhline(0, color='black')
        plt.scatter(self.sorted_times, self.residuals)
        self._plot_peak_lines()
        self._plot_af_lines()
        self._setup_anom_xaxis()
        plt.ylabel('res')

    def plot_anomaly(self):
        plt.figure()
        plt.title(self.anom_type)
        self.pspl_event.plot_data()
        self.pspl_event.plot_model(color='black', zorder=5)
        peak_anom_mag = mm.Utils.get_mag_from_flux(self.expected_model_fluxes[self.peak_index] + self.peak_dflux)
        plt.scatter(self.peak_time, peak_anom_mag, marker='d', color='darkgray', zorder=10)

        self._plot_peak_lines()
        self._plot_af_lines()
        self._setup_anom_xaxis()

        plt.ylabel('mag')

    @property
    def anom_type(self):
        if self._anom_type is None:
            self._anom_type = self.get_anom_type()

        return self._anom_type

    @property
    def peak_dflux(self):
        return self._peak_dflux

    @property
    def peak_index(self):
        return self._peak_index

    @property
    def peak_time(self):
        return self.sorted_times[self.peak_index]

    @property
    def t_start(self):
        return self._t_start

    @property
    def t_stop(self):
        return self._t_stop

    @property
    def dmag(self):
        expected_mag =  mm.Utils.get_mag_from_flux(
            self.expected_model_fluxes[self.peak_index])
        peak_anom_mag = mm.Utils.get_mag_from_flux(
            self.expected_model_fluxes[self.peak_index] + self.peak_dflux)

        return peak_anom_mag - expected_mag

    @property
    def peak_width(self):
        return self.t_stop - self.t_start

    @property
    def anom_index(self):
        if self._anom_index is None:
            self._anom_index = (self.times > self.anom_t_range_af[0]) & (self.times < self.anom_t_range_af[1])

        return self._anom_index

    @property
    def sorted_index(self):
        if self._sorted_index is None:
            self._sorted_index = np.argsort(self.times[self.anom_index])

        return self._sorted_index

    @property
    def times(self):
        if self._times is None:
            self._times = np.hstack([dataset.time for dataset in self.pspl_event.datasets])

        return self._times

    @property
    def sorted_times(self):
        return self.times[self.anom_index][self.sorted_index]

    @property
    def pspl_event(self):
        if self._pspl_event is None:
            self._pspl_event = self.get_pspl_event()

        return self._pspl_event

    @property
    def source_flux(self):
        if self._source_flux is None:
            self._source_flux, foo = self.pspl_event.get_ref_fluxes()

        return self._source_flux

    @property
    def blend_flux(self):
        if self._blend_flux is None:
            foo, self._blend_flux = self.pspl_event.get_ref_fluxes()

        return self._blend_flux

    @property
    def scaled_fluxes(self):
        if self._scaled_fluxes is None:
            self._scaled_fluxes = np.hstack(
                [np.array(flux) for (flux, err) in self.pspl_event.get_scaled_fluxes()])[self.anom_index][self.sorted_index]

        return self._scaled_fluxes

    @property
    def residuals(self):
        if self._scaled_residuals is None:
            self._scaled_residuals = self.scaled_fluxes - self.expected_model_fluxes

        return self._scaled_residuals

    @property
    def chi2s(self):
        if self._chi2s is None:
            self._chi2s = np.hstack(self.pspl_event.get_chi2_per_point())[self.anom_index][self.sorted_index]

        return self._chi2s

    @property
    def expected_model_fluxes(self):
        if self._expected_model_fluxes is None:
            self._expected_model_fluxes = self.source_flux * self.pspl_event.model.get_magnification(
                self.sorted_times) + self.blend_flux

        return self._expected_model_fluxes

#        self.mask_type = mask_type
#
#        self._refined_pspl_params = None
#        self._masked_datasets = None
#
#    def set_datasets_with_anomaly_masked(self, n_mask=3, tol=0.3):
#        """
#        Mask points associated with the anomaly.
#
#        :param mask_type: *str*
#            `t_eff' or `residuals'. If `t_eff' mask based on t_pl +- n_mask * t_eff. If `residuals', mask based on
#            deviation from existing point lens fit.
#
#        :param n_mask: *int*
#            Number of +- `t_eff' to mask. Only used with mask_type = `t_eff'.
#
#        :param tol: *float*
#            Maximum allowed deviation from point-lens in sigma. Only used with mask_type = `residuals'.
#
#        creates self.masked_datasets = *list* of MulensModel.MulensData objects with bad points masked.
#
#        """
#        masked_datasets = []
#        for dataset in self.datasets:
#            masked_datasets.append(copy.copy(dataset))
#
#        for dataset in masked_datasets:
#            if self.mask_type == 't_eff':
#                index = ((dataset.time >
#                         self.af_results['t_0'] -
#                         n_mask * self.af_results['t_eff']) &
#                         (dataset.time <
#                          self.af_results['t_0'] +
#                          n_mask * self.af_results['t_eff']))
#            elif self.mask_type == 'residuals':
#                index = self.get_residuals_mask(dataset, tol=tol)
#                print(np.sum(index))
#            else:
#                raise ValueError("mask_type must be one of ['t_eff', 'residuals']. Your value ", self.mask_type)
#
#            dataset.bad = index
#
#        self._masked_datasets = masked_datasets
#
#    def get_residuals_mask(self, dataset, tol=None, max_diff=1):
#        fit = MulensModel.FitData(dataset=dataset, model=MulensModel.Model(self.pspl_params))
#        fit.fit_fluxes()
#        ind_pl = np.argmin(np.abs(dataset.time - self.af_results['t_0']))
#
#        res, err = fit.get_residuals(phot_fmt='mag')
#        out_tol = np.argwhere(((np.abs(res) / err) > tol) & fit.dataset.good).flatten()
#        print(out_tol)
#        diff = np.ediff1d(out_tol)
#
#        start = np.argmin(np.abs(out_tol - ind_pl))
#        first, last = 0, len(out_tol) - 1
#        for i in range(start, 0, -1):
#            if diff[i] <= max_diff:
#                first = i
#            else:
#                break
#
#        for i in range(start, len(out_tol)):
#            if diff[i] <= max_diff:
#                last = i
#            else:
#                break
#
#        print(ind_pl, res[ind_pl])
#        print(ind_pl in out_tol)
#        print(first, last, len(out_tol))
#        print(out_tol[first], out_tol[last], out_tol[last] - out_tol[first])
#        mask = np.zeros(len(dataset.time), dtype=bool)
#        mask[out_tol[first]:out_tol[last]+1] = True
#
#        return mask
#
#    def get_dmag(self):
#        """
#        Find the magnitude difference at t_pl (af_results['t_0'])
#        :return: dmag: *float*
#        """
#        event = mm.Event(datasets=self.masked_datasets, model=mm.Model(self.refined_pspl_params))
#        dmag = []
#        event.fit_fluxes()
#        for fit in event.fits:
#            residuals, errors = fit.get_residuals(bad=True, phot_fmt='flux')
#            sigma = residuals[fit.dataset.bad]**2 / errors[fit.dataset.bad]**2
#            index = np.argmax(sigma)
#            mag_residuals, mag_errs = fit.get_residuals(bad=True, phot_fmt='mag')
#            dmag.append(mag_residuals[fit.dataset.bad][index])
#
#        max_ind = np.argmax(np.abs(dmag))
#        return dmag[max_ind]
#
#    def update_pspl_model(self):
#        fitter = mmexo.fitters.SFitFitter(datasets=self.datasets, initial_model=self.pspl_params)
#        fitter.run()
#        new_params = {key: fitter.best[key] for key in self.pspl_params.keys()}
#        self._refined_pspl_params = new_params
#
#
#    @property
#    def refined_pspl_params(self):
#        if self._refined_pspl_params is None:
#            self.update_pspl_model()
#
#        return self._refined_pspl_params
#
#    @property
#    def masked_datasets(self):
#        if self._masked_datasets is None:
#            self.set_datasets_with_anomaly_masked()
#
#        return self._masked_datasets
