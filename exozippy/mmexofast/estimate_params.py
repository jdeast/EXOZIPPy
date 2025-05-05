#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Created by Luca Campiani in January 2024
import MulensModel
import MulensModel as mm
#import matplotlib.pyplot as plt
import numpy as np
#import warnings
import copy

import exozippy.mmexofast as mmexo

# In[ ]:


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
        print(params)

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


def correct_alpha(alpha):
    while alpha > 360.:
        alpha -= 360.

    while alpha < -360:
        alpha += 360.

    return alpha


def get_wide_params(params):
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

    Returns :
        wide_params : *BinaryLensParams*
             Wide model parameters for the binary lens.
    """
    # JCY: Should these calculations be broken out into individual parameters?
    # e.g., so they can be tested individually?
    # This would mean function --> class.
    tau = (params['t_pl'] - params['t_0']) / params['t_E']
    u = np.sqrt(params['u_0']**2 + tau**2)
    s = 0.5 * (np.sqrt(u**2 + 4) + u)
    #alpha = np.arctan2(-params['u_0'], tau)
    alpha = np.pi - np.arctan2(params['u_0'], tau)
    rho = params['dt'] / params['t_E'] / 2.
    q = 0.5 * params['dmag'] * (rho**2)

    alpha_deg = correct_alpha(np.rad2deg(alpha))
    new_params = {'t_0': params['t_0'],
               'u_0': params['u_0'],
               't_E': params['t_E'],
               's': s,
               'q': q,
               'rho': rho,
               'alpha': alpha_deg}

    out = BinaryLensParams(new_params)
    out.set_mag_method(params)
    
    return out


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
    if q is None:
        q = 0.0040

    tau = (params['t_pl'] - params['t_0']) / params['t_E']
    u = np.sqrt(params['u_0']**2 + tau**2)

    s = 0.5 * (np.sqrt(u**2 + 4) - u)

    eta_not = (q**0.5 / s) * (1 / (np.sqrt(1 + s**2)) + np.sqrt(1 - s**2))
    mu = np.arctan2(eta_not, (s - 1 / s) / (1 + q)) # correction for primary --> COM
    phi = np.arctan2(tau, params['u_0'])
    alpha1 = np.pi / 2 - mu - phi
    alpha2 = alpha1 + 2 * mu

    alpha1_deg = correct_alpha(-np.rad2deg(alpha1) + 180.)
    alpha2_deg = correct_alpha(-np.rad2deg(alpha2) + 180.)

    if 'dt' in params.keys():
        rho = params['dt'] / params['t_E'] / 2.
    elif 'rho' not in params.keys():
        rho = 0.001
    
    new_params1 = {'t_0': params['t_0'],
                'u_0': params['u_0'],
                't_E': params['t_E'],
                's': s,
                'q': q,
                'rho': rho,
                'alpha': alpha1_deg}
    
    new_params2 = {'t_0': params['t_0'],
                'u_0': params['u_0'],
                't_E': params['t_E'],
                's': s,
                'q': q,
                'rho': rho,
                'alpha': alpha2_deg}
    
    out1 = BinaryLensParams(new_params1)
    out2 = BinaryLensParams(new_params2)
    out1.set_mag_method(params)
    out2.set_mag_method(params)
    
    return out1, out2


# In[ ]:


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

    def __init__(self, datasets=None, pspl_params=None, af_results=None, mask_type='t_eff'):
        if isinstance(datasets, MulensModel.MulensData):
            datasets = [datasets]

        self.datasets = datasets
        self.pspl_params = pspl_params
        self.af_results = af_results
        self.mask_type = mask_type

        self._refined_pspl_params = None
        self._masked_datasets = None

    def set_datasets_with_anomaly_masked(self, n_mask=3, tol=0.3):
        """
        Mask points associated with the anomaly.

        :param mask_type: *str*
            `t_eff' or `residuals'. If `t_eff' mask based on t_pl +- n_mask * t_eff. If `residuals', mask based on
            deviation from existing point lens fit.

        :param n_mask: *int*
            Number of +- `t_eff' to mask. Only used with mask_type = `t_eff'.

        :param tol: *float*
            Maximum allowed deviation from point-lens in sigma. Only used with mask_type = `residuals'.

        creates self.masked_datasets = *list* of MulensModel.MulensData objects with bad points masked.

        """
        masked_datasets = []
        for dataset in self.datasets:
            masked_datasets.append(copy.copy(dataset))

        for dataset in masked_datasets:
            if self.mask_type == 't_eff':
                index = ((dataset.time >
                         self.af_results['t_0'] -
                         n_mask * self.af_results['t_eff']) &
                         (dataset.time <
                          self.af_results['t_0'] +
                          n_mask * self.af_results['t_eff']))
            elif self.mask_type == 'residuals':
                index = self.get_residuals_mask(dataset, tol=tol)
                print(np.sum(index))
            else:
                raise ValueError("mask_type must be one of ['t_eff', 'residuals']. Your value ", self.mask_type)

            dataset.bad = index

        self._masked_datasets = masked_datasets

    def get_residuals_mask(self, dataset, tol=None, max_diff=1):
        fit = MulensModel.FitData(dataset=dataset, model=MulensModel.Model(self.pspl_params))
        fit.fit_fluxes()
        ind_pl = np.argmin(np.abs(dataset.time - self.af_results['t_0']))

        res, err = fit.get_residuals(phot_fmt='mag')
        out_tol = np.argwhere(((np.abs(res) / err) > tol) & fit.dataset.good).flatten()
        print(out_tol)
        diff = np.ediff1d(out_tol)

        start = np.argmin(np.abs(out_tol - ind_pl))
        first, last = 0, len(out_tol) - 1
        for i in range(start, 0, -1):
            if diff[i] <= max_diff:
                first = i
            else:
                break

        for i in range(start, len(out_tol)):
            if diff[i] <= max_diff:
                last = i
            else:
                break

        print(ind_pl, res[ind_pl])
        print(ind_pl in out_tol)
        print(first, last, len(out_tol))
        print(out_tol[first], out_tol[last], out_tol[last] - out_tol[first])
        mask = np.zeros(len(dataset.time), dtype=bool)
        mask[out_tol[first]:out_tol[last]+1] = True

        return mask

    def get_dmag(self):
        """
        Find the magnitude difference at t_pl (af_results['t_0'])
        :return: dmag: *float*
        """
        event = mm.Event(datasets=self.masked_datasets, model=mm.Model(self.refined_pspl_params))
        dmag = []
        event.fit_fluxes()
        for fit in event.fits:
            residuals, errors = fit.get_residuals(bad=True, format='flux')
            sigma = residuals[fit.dataset.bad]**2 / errors[fit.dataset.bad]**2
            index = np.argmax(sigma)
            mag_residuals, mag_errs = fit.get_residuals(bad=True, format='mag')
            dmag.append(mag_residuals[fit.dataset.bad][index])

        max_ind = np.argmax(np.abs(dmag))
        return dmag[max_ind]

    def update_pspl_model(self):
        fitter = mmexo.fitters.SFitFitter(datasets=self.datasets, initial_model=self.pspl_params)
        fitter.run()
        new_params = {key: fitter.best[key] for key in self.pspl_params.keys()}
        self._refined_pspl_params = new_params

    def get_anomaly_lc_parameters(self):
        params = {key: value for key, value in self.refined_pspl_params.items()}
        params['t_pl'] = self.af_results['t_0']
        params['dt'] = 2. * self.af_results['t_eff']
        params['dmag'] = self.get_dmag()

        return params

    @property
    def refined_pspl_params(self):
        if self._refined_pspl_params is None:
            self.update_pspl_model()

        return self._refined_pspl_params

    @property
    def masked_datasets(self):
        if self._masked_datasets is None:
            self.set_datasets_with_anomaly_masked()

        return self._masked_datasets
