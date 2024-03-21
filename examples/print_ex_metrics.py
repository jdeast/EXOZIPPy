
def print_metrics(results, expected):
    for key in expected.keys():
        print('Param type:', key)
        print('key, Dsig^2, Dfrac^2')
        for param_key in expected[key].keys():
            exp_value = expected[key][param_key]
            obs_value = results[key][param_key]
            obs_sigma = results['{0}_sigma'.format(key)][param_key]
            print(param_key,
                  ((obs_value - exp_value) / obs_sigma)**2,
                  ((obs_value - exp_value) / exp_value)**2)