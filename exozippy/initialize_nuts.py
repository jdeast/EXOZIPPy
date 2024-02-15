from exozippy_getmcmcscale import exozippy_getmcmcscale

def initialize_nuts(model, fit_params, chi2func, n=36):
    with model:
        map = pm.find_MAP(return_raw=True)
    map_vars = map[0]
    map_dict = dict()
    for p in fit_params:
        map_dict[p] = map[0][p]
    scale, bestpars = exozippy_getmcmcscale(map_dict, chi2func)

    init_vals = add_jitter(n, bestpars, scale, fit_params)

    for key, val in bestpars.items():
        bestpars[key] = np.array(val)

    apoint = DictToArrayBijection.map(bestpars)

    n=len(apoint.data)

    scaling = fmt_scale(model, scale)

    potential = quadpotential.QuadPotentialDiagAdapt(n, apoint.data, scaling, 10)
    with model:
        nuts_sampler = pm.NUTS(target_accept=0.99, potential=potential, max_treedepth=12)

    return init_vals, nuts_sampler