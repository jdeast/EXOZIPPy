#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Created by LucaCampiani Campiani in January 2024

import MulensModel as mm
import matplotlib.pyplot as plt
import numpy as np


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
        self.mag_method = None
        
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
        t1 = params['t_pl'] - (5 * params['dt'])
        t2 = params['t_pl'] + (5 * params['dt'])
        self.mag_method = [t1, 'VBBL', t2]


# In[ ]:


def get_wide_params(params):
    """
    Transform initial parameters into wide model parameters.

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
        wide_params : *BinaryLensParams*
             Wide model parameters for the binary lens.
    """
    tau = (params['t_pl'] - params['t_0']) / params['t_E']
    u = np.sqrt(params['u_0']**2 + tau**2)
    s = 0.5 * (np.sqrt(u**2 + 4) + u)
    alpha = np.arctan2(-params['u_0'], tau)
    rho = params['dt'] / params['t_E'] / 2.
    q = 0.5 * params['dmag'] * (rho**2)
   
    new_params = {'t_0': params['t_0'],
               'u_0': params['u_0'],
               't_E': params['t_E'],
               's': s,
               'q': q,
               'rho': rho,
               'alpha': np.rad2deg(alpha)}
    
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

    if 'dt' in params.keys():
        rho = params['dt'] / params['t_E'] / 2.
    elif 'rho' is None:
        rho = 0.001
    
    new_params1 = {'t_0': params['t_0'],
                'u_0': params['u_0'],
                't_E': params['t_E'],
                's': s,
                'q': q,
                'rho': rho,
                'alpha': -np.rad2deg(alpha1)}
    
    new_params2 = {'t_0': params['t_0'],
                'u_0': params['u_0'],
                't_E': params['t_E'],
                's': s,
                'q': q,
                'rho': rho,
                'alpha': -np.rad2deg(alpha2)}
    
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

