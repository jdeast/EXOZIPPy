"""
High-level functions for fitting microlensing events.
"""
import os.path
import warnings

import numpy as np
import copy
from collections import OrderedDict
import emcee

import MulensModel
import astropy.units

import sfit_minimizer as sfit
from exozippy import exozippy_getmcmcscale
import exozippy.mmexofast as mmexo
from exozippy.mmexofast import observatories


def fit(files=None, fit_type=None, **kwargs):
    """
    Fit a microlensing light curve using MMEXOFAST

    :param files:
    :param coords:
    :param priors:
    :param fit_type:
    :param print_results:
    :param verbose:
    :param output_file:
    :return:

    ***
    Q1: Should this also include an `input_file` option?
    Q2: What about `initial_param` = *dict* of microlensing parameters option?
    Open issue: as written, only supports single solutions.
    ***

    """
    fitter = MMEXOFASTFitter(files=files, fit_type=fit_type, **kwargs)
    fitter.fit()
    return fitter.results


class MMEXOFASTFitter():

    def __init__(self, files=None, fit_type=None, datasets=None, coords=None,
                 priors=None, print_results=False, verbose=False,
                 output_file=None, log_file=None, emcee_settings=None):
        self.verbose = verbose
        if log_file is not None:
            self.log_file = log_file

        # setup datasets.
        if datasets is not None:
            self.datasets = datasets
        else:
            self.datasets = self._create_mulensdata_objects(files)

        self.fit_type = fit_type
        self.emcee_settings = emcee_settings

        # initialize additional data versions
        self._residuals = None
        self._masked_datasets = None

        # initialize params
        self._best_ef_grid_point = None
        self._initial_pspl_params = None
        self._initial_pspl_results = None
        self._pspl_params = None
        self._best_af_grid_point = None
        self._anomaly_lc_params = None
        self._binary_params = None

        self._results = None

    def _create_mulensdata_objects(self, files):
        if isinstance(files, (str)):
            files = [files]

        datasets = []
        for filename in files:
            if not os.path.exists(filename):
                raise FileNotFoundError(
                    "Data file {0} does not exist".format(filename))

            kwargs = observatories.get_kwargs(filename)
            data = MulensModel.MulensData(file_name=filename, **kwargs)
            datasets.append(data)

        return datasets

    def fit(self):
        if self.fit_type is None:
            # Maybe "None" means initial mulens parameters were passed,
            # so we can go straight to a mmexofast_fit?
            raise ValueError(
                'You must set the fit_type when initializing the ' +
                'MMEXOFASTFitter(): fit_type=("point lens", "binary lens")')

        self.best_ef_grid_point = self.do_ef_grid_search()
        if self.verbose:
            print('Best EF grid point ', self.best_ef_grid_point)

        self.initial_pspl_results = self.get_initial_pspl()
        if self.verbose:
            print('Initial SFit', self.initial_pspl_results)

        self.best_af_grid_point = self.do_af_grid_search()
        if self.verbose:
            print('Best AF grid', self.best_af_grid_point)

        self.anomaly_lc_params = self.get_anomaly_lc_params()
        if self.verbose:
            print('Anomaly Params', self.anomaly_lc_params)

        self.results = self.fit_anomaly()
        if self.verbose:
            print('Results', self.results)

        #self.get_best_point_lens_model()
        #self.do_af_grid_search()
        #self.results = self.find_best_binary_model()

    #def get_best_point_lens_model(self):
    #    if self.log_file is not None:
    #        log = open(self.log_file, 'a')
    #
    #    # Find initial Point Lens model
    #    self.best_ef_grid_point = self.do_ef_grid_search()
    #    if self.verbose:
    #        print('Best EF grid point', self.best_ef_grid_point)
    #    if self.log_file is not None:
    #        log.write('Best EF grid point {0}\n'.format(self.best_ef_grid_point))
    #
    #    self.pspl_params = self.get_initial_pspl_params(
    #        verbose=self.verbose)
    #    if self.verbose:
    #        print('Initial PSPL', self.pspl_params)
    #    if self.log_file is not None:
    #        log.write('Initial PSPL {0}\n'.format(self.pspl_params))
    #
    #    self.pspl_params = self.do_sfit(self.datasets)
    #    if self.verbose:
    #        print('SFIT params:', self.pspl_params)
    #    if self.log_file is not None:
    #        log.write('SFIT params {0}\n'.format(self.pspl_params))
    #        log.close()
    #
    ##    if self.fit_type == 'point lens':
    ##        # Do the full MMEXOFAST fit to get physical parameters
    ##        self.results = self.do_mmexofast_fit()
    ##        return self.results
    ##    elif self.fit_type == 'binary lens':
    ##        # Find the initial planet parameters
    ##        self.best_af_grid_point = self.do_af_grid_search()
    #
    #def find_best_binary_model(self):
    #    if self.log_file is not None:
    #        log = open(self.log_file, 'a')
    #
    #    self.pspl_params = self.refine_pspl_params()
    #    if self.verbose:
    #        print('Revised SFIT', self.pspl_params)
    #
    #    if self.log_file is not None:
    #        log.write('Revised SFIT {0}\n'.format(self.pspl_params))
    #
    #    self.binary_params = self.get_initial_2L1S_params()
    #    if self.verbose:
    #        print(
    #            'Initial 2L1S params', self.binary_params.ulens)
    #        print('mag_methods', self.binary_params.mag_methods)
    #    if self.log_file is not None:
    #        log.write('Initial 2L1S params {0}\n'.format(self.binary_params.ulens))
    #        log.write('mag_methods {0}\n'.format(self.binary_params.mag_methods))
    #        log.flush()
    #
    #    # Do the full MMEXOFAST fit to get physical parameters
    #    #self.results = self.do_mmexofast_fit()
    #    self.results = None
    #    if self.verbose:
    #        print('Final params', self.results)
    #
    #    return self.results
    #
    def do_ef_grid_search(self):
        # Should probably scrape t_0_1 from the filenames
        # JCY: what did this comment mean???
        ef_grid = mmexo.EventFinderGridSearch(datasets=self.datasets)
        ef_grid.run()
        return ef_grid.best

    def get_initial_pspl(self, verbose=False):
        pspl_est_params = mmexo.estimate_params.get_PSPL_params(self.best_ef_grid_point, self.datasets)
        if self.verbose:
            print('Initial PSPL Estimate', pspl_est_params)

        fitter = mmexo.fitters.SFitFitter(initial_model=pspl_est_params, datasets=self.datasets)
        fitter.run()
        return fitter.best
        #pspl_params = fitter.best.copy()
        #del pspl_params['chi2']
        #return pspl_params

    #    t_0 = self.best_ef_grid_point['t_0']
    #    if self.best_ef_grid_point['j'] == 1:
    #        u_0 = 0.01
    #    elif self.best_ef_grid_point['j'] == 2:
    #        u_0s = [0.01, 0.1, 0.3, 1.0, 1.5]
    #        chi2s = []
    #        for u_0 in u_0s:
    #            t_E = self.best_ef_grid_point['t_eff']
    #            params = {'t_0': t_0, 't_E': t_E, 'u_0': u_0}
    #            event = MulensModel.Event(
    #                datasets=self.datasets, model=MulensModel.Model(params))
    #            chi2s.append(event.get_chi2())
    #
    #        index = np.nanargmin(chi2s)
    #        u_0 = u_0s[index]
    #        if verbose:
    #            print('u0s', u_0s)
    #            print('chi2s', chi2s)
    #            print('selected', index, u_0)
    #
    #    else:
    #        raise ValueError(
    #            'j may only be 1 or 2. Your input: ', self.best_ef_grid_point)
    #
    #    t_E = self.best_ef_grid_point['t_eff']
    #
    #    return {'t_0': t_0, 't_E': t_E, 'u_0': u_0}
    #
    #def do_sfit(self, datasets, verbose=False):
    #    param_sets = [['t_0', 't_E'], ['t_0', 'u_0', 't_E']]
    #
    #    params = self.pspl_params
    #    for i in range(len(param_sets)):
    #        parameters_to_fit = param_sets[i]
    #        event = MulensModel.Event(
    #            datasets=datasets, model=MulensModel.Model(params))
    #        event.fit_fluxes()
    #
    #        my_func = sfit.mm_funcs.PointLensSFitFunction(
    #            event, parameters_to_fit)
    #
    #        initial_guess = []
    #        for key in parameters_to_fit:
    #            if isinstance(params[key], (astropy.units.Quantity)):
    #                initial_guess.append(params[key].value)
    #            else:
    #                initial_guess.append(params[key])
    #
    #        for i in range(len(datasets)):
    #            initial_guess.append(event.fits[i].source_flux)
    #            initial_guess.append(event.fits[i].blend_flux)
    #
    #        result = sfit.minimize(
    #            my_func, x0=initial_guess, tol=1e-5,
    #            options={'step': 'adaptive'}, verbose=verbose)
    #
    #        if verbose:
    #            print(result)
    #
    #        params = my_func.event.model.parameters.parameters
    #
    #    return params
    #
    #def _setup_model(self, model_type):
    #    if model_type == 'pspl':
    #        model = MulensModel.Model(self.pspl_params)
    #    elif model_type == 'binary':
    #        model = MulensModel.Model(self.binary_params.ulens)
    #        model.set_default_magnification_method = 'point_source_point_lens'
    #        #mag_methods = [model.parameters.t_0 - 3. * model.parameters.t_E, 'point_source'] +  self.binary_params.mag_method + ['point_source', model.parameters.t_0 + 3. * model.parameters.t_E]
    #        model.set_magnification_methods(self.binary_params.mag_methods)
    #    else:
    #        raise KeyError('model_type must be ["pspl", "binary"]. Your value: ', model_type)
    #
    #    return model
    #
    #def _setup_event(self, model):
    #    # Create Event object
    #    if self.masked_datasets is None:
    #        datasets = self.datasets
    #    else:
    #        datasets = self.masked_datasets
    #
    #    event = MulensModel.Event(model=model, datasets=datasets)
    #    for dataset in datasets:
    #        if dataset.bandpass is not None:
    #            event.model.set_limb_coeff_gamma(dataset.bandpass, 0.)
    #            warnings.warn('Assuming ld coeff = 0 for bandpass: ' +
    #                          dataset.bandpass)
    #
    #    return event
    #
    #def do_mmexofast_fit(self):
    #    # Set initial parameters
    #    if self._binary_params is not None:
    #        params_dict = OrderedDict(self.binary_params.ulens)
    #        model = self._setup_model('binary')
    #    else:
    #        params_dict = OrderedDict(self.pspl_params)
    #        model = self._setup_model('pspl')
    #
    #    event = self._setup_event(model)
    #
    #    def chi2func(bestpars):
    #        for (parameter, value) in bestpars.items():
    #            setattr(event.model.parameters, parameter, value)
    #
    #        return event.get_chi2()
    #
    #    def get_seedscale(params_dict):
    #        seedscale = []
    #        for key, value in params_dict.items():
    #            if key == 't_0':
    #                seedscale.append(1.0)
    #            elif key == 'q':
    #                seedscale.append(0.001 * value)
    #            elif key == 's':
    #                seedscale.append(0.01 * value)
    #            elif key == 'alpha':
    #                seedscale.append(0.1)
    #            else:
    #                seedscale.append(0.02 * value)
    #
    #        return seedscale
    #
    #    seedscale = get_seedscale(params_dict)
    #
    #    mcmcscale_avg, bestpars = exozippy_getmcmcscale.exozippy_getmcmcscale(
    #        params_dict, chi2func, tofit=None,
    #        seedscale=seedscale, bestchi2=None,
    #        angular=None, debug=False,
    #        skipiter=False, logname=None)
    #
    #    print(mcmcscale_avg, bestpars)
    #
    #    # Could be replaced with EXOZippy later.
    #    results = self.do_emcee_fit(mcmcscale_avg, bestpars)
    #    print(results)
    #
    #    return results
    #
    #def do_emcee_fit(self, mcmcscale_avg, bestpars):
    #    # Define likelihood functions
    #
    #    def ln_like(theta, event, parameters_to_fit):
    #        """ likelihood function """
    #        for (parameter, value) in zip(parameters_to_fit, theta):
    #            if (parameter in ['t_E', 'rho', 's', 'q']) and (value < 0.):
    #                return -np.inf
    #
    #            setattr(event.model.parameters, parameter, value)
    #
    #        chi2 = event.get_chi2()
    #        if chi2 < ln_like.best[0]:
    #            ln_like.best = [chi2, theta]
    #        return -0.5 * chi2
    #
    #    ln_like.best = [np.inf]
    #
    #    model = self._setup_model('binary')
    #    model.parameters = bestpars
    #
    #    my_event = self._setup_event(model)
    #    parameters_to_fit = list(bestpars.keys())
    #
    #    # Initializations for EMCEE
    #    n_dim = len(bestpars.keys())
    #    n_walkers = 40
    #    n_steps = 500
    #    #n_burn = 150
    #    # Including the set of n_walkers starting points:
    #    start_1 = list(bestpars.values())
    #    start = [start_1 + np.random.randn(n_dim) * np.array(list(mcmcscale_avg.values()))
    #             for i in range(n_walkers)]
    #
    #    # Run emcee (this can take some time):
    #    sampler = emcee.EnsembleSampler(
    #        n_walkers, n_dim, ln_like, args=(my_event, parameters_to_fit))
    #    sampler.run_mcmc(start, n_steps)
    #
    #    # Remove burn-in samples and reshape:
    #    #samples = sampler.chain[:, n_burn:, :].reshape((-1, n_dim))
    #    results = {key: b for key, b in zip(bestpars.keys(), ln_like.best[1])}
    #    return results
    #
    #def set_datasets_with_anomaly_masked(self, mask_type='t_eff', n_mask=3, tol=0.3):
    #    """
    #    Mask points associated with the anomaly.
    #
    #    :param mask_type: *str*
    #        `t_eff' or `residuals'. If `t_eff' mask based on t_pl +- n_mask * t_eff. If `residuals', mask based on
    #        deviation from existing point lens fit.
    #
    #    :param n_mask: *int*
    #        Number of +- `t_eff' to mask. Only used with mask_type = `t_eff'.
    #
    #    :param tol: *float*
    #        Maximum allowed deviation from point-lens in sigma. Only used with mask_type = `residuals'.
    #
    #    creates self.masked_datasets = *list* of MulensModel.MulensData objects with bad points masked.
    #
    #    """
    #    masked_datasets = []
    #    for dataset in self.datasets:
    #        masked_datasets.append(copy.copy(dataset))
    #
    #    for dataset in masked_datasets:
    #        if mask_type == 't_eff':
    #            index = ((dataset.time >
    #                     self.best_af_grid_point['t_0'] -
    #                     n_mask * self.best_af_grid_point['t_eff']) &
    #                     (dataset.time <
    #                      self.best_af_grid_point['t_0'] +
    #                      n_mask * self.best_af_grid_point['t_eff']))
    #        elif mask_type == 'residuals':
    #            index = self.get_residuals_mask(dataset, tol=tol)
    #            print(np.sum(index))
    #        else:
    #            raise ValueError("mask_type must be one of ['t_eff', 'residuals']. Your value ", mask_type)
    #
    #        dataset.bad = index
    #
    #    self.masked_datasets = masked_datasets
    #
    #def get_residuals_mask(self, dataset, tol=None, max_diff=1):
    #    fit = MulensModel.FitData(dataset=dataset, model=MulensModel.Model(self.pspl_params))
    #    fit.fit_fluxes()
    #    ind_pl = np.argmin(np.abs(dataset.time - self.best_af_grid_point['t_0']))
    #
    #    res, err = fit.get_residuals(phot_fmt='mag')
    #    out_tol = np.argwhere(((np.abs(res) / err) > tol) & fit.dataset.good).flatten()
    #    print(out_tol)
    #    diff = np.ediff1d(out_tol)
    #
    #    start = np.argmin(np.abs(out_tol - ind_pl))
    #    first, last = 0, len(out_tol) - 1
    #    for i in range(start, 0, -1):
    #        if diff[i] <= max_diff:
    #            first = i
    #        else:
    #            break
    #
    #    for i in range(start, len(out_tol)):
    #        if diff[i] <= max_diff:
    #            last = i
    #        else:
    #            break
    #
    #    print(ind_pl, res[ind_pl])
    #    print(ind_pl in out_tol)
    #    print(first, last, len(out_tol))
    #    print(out_tol[first], out_tol[last], out_tol[last] - out_tol[first])
    #    mask = np.zeros(len(dataset.time), dtype=bool)
    #    mask[out_tol[first]:out_tol[last]+1] = True
    #
    #    return mask
    #
    #def refine_pspl_params(self):
    #    self.set_datasets_with_anomaly_masked()
    #    results = self.do_sfit(self.masked_datasets)
    #    return results
    #
    def set_residuals(self, pspl_params):
        #print(pspl_params)
        event = MulensModel.Event(
            datasets=self.datasets, model=MulensModel.Model(pspl_params))
        event.fit_fluxes()
        residuals = []
        for i, dataset in enumerate(self.datasets):
            res, err = event.fits[i].get_residuals(phot_fmt='flux')
            residuals.append(
                MulensModel.MulensData(
                    [dataset.time, res, err], phot_fmt='flux',
                    bandpass=dataset.bandpass,
                    ephemerides_file=dataset.ephemerides_file))

        self.residuals = residuals

    def do_af_grid_search(self):
        #if self.log_file is not None:
        #    log = open(self.log_file, 'a')
        self.set_residuals(self.initial_pspl_params)
        af_grid = mmexo.AnomalyFinderGridSearch(residuals=self.residuals)
        # May need to update value of teff_min
        af_grid.run()
        return af_grid.best

        #if self.verbose:
        #    print('Best AF grid', self.best_af_grid_point)
        #
        #if self.log_file is not None:
        #    log.write('Best AF grid {0}\n'.format(self.best_af_grid_point))
        #    log.close()

    def get_anomaly_lc_params(self):
        estimator = mmexo.estimate_params.AnomalyPropertyEstimator(
            datasets=self.datasets, pspl_params=self.initial_pspl_params, af_results=self.best_af_grid_point)
        return estimator.get_anomaly_lc_parameters()

    def fit_anomaly(self):
        # So far, this only fits wide planet models in the GG97 limit.
        #print(self.anomaly_lc_params)
        wide_planet_fitter = mmexo.fitters.WidePlanetFitter(
            datasets=self.datasets, anomaly_lc_params=self.anomaly_lc_params,
            emcee_settings=self.emcee_settings)
        if self.verbose:
            wide_planet_fitter.estimate_initial_parameters()
            print('Initial 2L1S Wide Model', wide_planet_fitter.initial_model)
            print('mag methods', wide_planet_fitter.mag_methods)

        wide_planet_fitter.run()
        return wide_planet_fitter.best

    #
    #def get_dmag(self):
    #    pspl_event = MulensModel.Event(
    #        datasets=self.masked_datasets,
    #        model=MulensModel.Model(self.pspl_params))
    #    source_flux, blend_flux = pspl_event.get_ref_fluxes()
    #    flux_pspl = source_flux[0] * pspl_event.model.get_magnification(
    #        self.best_af_grid_point['t_0']) + blend_flux
    #
    #    # Can this be cleaned up?
    #    af_grid_search = mmexo.gridsearches.AnomalyFinderGridSearch(
    #        self.residuals)
    #    trimmed_datasets = af_grid_search.get_trimmed_datasets(
    #        self.best_af_grid_point)
    #    ef_sfit = mmexo.gridsearches.EFSFitFunction(
    #        trimmed_datasets, self.best_af_grid_point)
    #    ef_sfit.update_all()
    #    new_theta = ef_sfit.theta + ef_sfit.get_step()
    #    ef_sfit.update_all(new_theta)
    #
    #    i = pspl_event.data_ref
    #    fs = ef_sfit.theta[2 * i]
    #    fb = ef_sfit.theta[2 * i + 1]
    #    mag = ef_sfit.get_magnification(self.best_af_grid_point['t_0'])
    #    d_flux_anom = fs * mag * fb
    #
    #    dmag = -2.5 * np.log10((flux_pspl + d_flux_anom) / flux_pspl)
    #    return dmag[0]
    #
    #def get_initial_2L1S_params(self):
    #    dmag = self.get_dmag()
    #    if isinstance(self.pspl_params['t_E'], (astropy.units.Quantity)):
    #        t_E = self.pspl_params['t_E'].value
    #    elif isinstance(self.pspl_params['t_E'], (float)):
    #        t_E = self.pspl_params['t_E']
    #    else:
    #        raise TypeError(
    #            'Invalid type for t_E:', self.pspl_params['t_E'],
    #            type(self.pspl_params['t_E']))
    #
    #    params = {
    #        't_0': self.pspl_params['t_0'], 'u_0': self.pspl_params['u_0'],
    #        't_E': t_E, 't_pl': self.best_af_grid_point['t_0'],
    #        'dt': 2. * self.best_af_grid_point['t_eff'], 'dmag': dmag}
    #    binary_params = mmexo.estimate_params.get_wide_params(params)
    #    return binary_params

    @property
    def residuals(self):
        return self._residuals

    @residuals.setter
    def residuals(self, value):
        self._residuals = value

    @property
    def masked_datasets(self):
        return self._masked_datasets

    @masked_datasets.setter
    def masked_datasets(self, value):
        self._masked_datasets = value

    @property
    def best_ef_grid_point(self):
        return self._best_ef_grid_point

    @best_ef_grid_point.setter
    def best_ef_grid_point(self, value):
        self._best_ef_grid_point = value

    @property
    def initial_pspl_results(self):
        return self._initial_pspl_results

    @initial_pspl_results.setter
    def initial_pspl_results(self, value):
        self._initial_pspl_results = value

    @property
    def initial_pspl_params(self):
        if self._initial_pspl_params is None:
            if self.initial_pspl_results is not None:
                self._initial_pspl_params = self.initial_pspl_results.copy()
                del self._initial_pspl_params['chi2']
            else:
                raise AttributeError('No Initial PSPL results.')

        return self._initial_pspl_params

    @property
    def pspl_params(self):
        return self._pspl_params

    @pspl_params.setter
    def pspl_params(self, value):
        self._pspl_params = value

    @property
    def best_af_grid_point(self):
        return self._best_af_grid_point

    @best_af_grid_point.setter
    def best_af_grid_point(self, value):
        self._best_af_grid_point = value

    @property
    def anomaly_lc_params(self):
        return self._anomaly_lc_params

    @anomaly_lc_params.setter
    def anomaly_lc_params(self, value):
        self._anomaly_lc_params = value

    @property
    def binary_params(self):
        return self._binary_params

    @binary_params.setter
    def binary_params(self, value):
        self._binary_params = value

    @property
    def results(self):
        return self._results

    @results.setter
    def results(self, value):
        self._results = value

