
def print_metrics(results, expected):
    for key in expected.keys():
        print('Param type:', key)
        print('key, exp_value, exp_sigma, obs_value, obs_sigma, Dsig^2, Dfrac^2, sig_frac^2')
        for param_key in expected[key].keys():
            obs_value = results[key][param_key]
            obs_sigma = results['{0}_sigma'.format(key)][param_key]

            exp = expected[key][param_key]
            if isinstance(exp, (float)):
                exp_value = exp
                exp_sigma = None
                sig_frac2 = None
            else:
                exp_value = exp[0]
                exp_sigma = exp[1]
                sig_frac2 = ((exp_sigma - obs_sigma) / exp_sigma)**2

            dsig2 = ((obs_value - exp_value) / obs_sigma)**2
            dfrac2 = ((obs_value - exp_value) / exp_value)**2

            print(param_key, exp_value, exp_sigma, obs_value, obs_sigma,
                  dsig2, dfrac2, sig_frac2)