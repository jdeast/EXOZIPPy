import numpy as np


def coords_rel2co_magnif(params):
    co_magnif_to_co_mass(params)


def get_co_mass_co_manif_offset(params):
    if params['s'] > 1.:
        delta_x = ((params['q'] / (1. + params['q'])) *
                   ((1. / params['s']) - params['s']))
        delta_u0 = delta_x * np.sin(np.deg2rad(params['alpha']))
        delta_t0 = delta_x * params['t_E'] * np.cos(
            np.deg2rad(params['alpha']))
    else:
        delta_t0 = 0.
        delta_u0 = 0.

    return delta_t0, delta_u0


def co_mass_to_co_magnif(params):
    """Transform from center of mass to center of magnification coordinates"""
    delta_t0, delta_u0 = get_co_mass_co_manif_offset(params)
    return {'t_0': params['t_0'] - delta_t0,
            'u_0': params['u_0'] - delta_u0}


def co_magnif_to_co_mass(params):
    """Transform from center of magnification to center of mass coordinates"""
    delta_t0, delta_u0 = get_co_mass_co_manif_offset(params)

    return {'t_0': params['t_0'] - delta_t0,
            'u_0': params['u_0'] + delta_u0}


def primary_to_co_magnif(params):
    """Transform from primary location to center of magnification coordinates"""
    delta_x = params['q'] * ((1. / params['s']) + params['s'])
    delta_u0 = delta_x * np.sin(np.deg2rad(params['alpha']))
    delta_t0 = delta_x * params['t_E'] * np.cos(
        np.deg2rad(-params['alpha']))

    return {'t_0': params['t_0'] - delta_t0,
            'u_0': params['u_0'] + delta_u0}


def primary_to_co_mass(params):
    """Transform from primary location to center of mass coordinates"""
    new_coords = primary_to_co_magnif(params)
    new_params = {'t_0': new_coords['t_0'], 'u_0': new_coords['u_0'], 't_E': params['t_E'],
                  's': params['s'], 'q': params['q'], 'alpha': params['alpha']}

    return co_magnif_to_co_mass(new_params)
