"""
High-level functions for fitting microlensing events.
"""
import os.path
import warnings

import pandas as pd
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


class MMEXOFASTFitResults():
    """
    Class containing the results of a fit and nice methods for accessing the results.
    """

    def __init__(self, fitter):
        self.fitter = fitter

    def get_params_from_results(self):
        """
        Take the results of a fit and return a dictionary with just the best-fit microlensing parameters and values,
        i.e., something appropriate for using as input to `MulensModel.Model()`.

        :param :py:class:`MMEXOFASTFit` object.

        :return: *dict* of microlensing parameters and values
        """
        params = {key: value for key, value in self.best.items()}
        params.pop('chi2')
        return params

    def get_sigmas_from_results(self):
        """
         Take the results of a fit and return a dictionary with the uncertainties for each microlensing parameter.

        :param :return: :py:class:`MMEXOFASTFit` object.

        :return: *dict* of uncertainties in microlensing parameters and values
        """
        sigmas = {}
        for param, sigma in zip(self.parameters_to_fit, self.results.sigmas):
            sigmas[param] = sigma

        return sigmas

    def format_results_as_df(self):
        parameters = [x for x in self.parameters_to_fit]
        values = [x for x in self.results.x]
        sigmas = [x for x in self.results.sigmas]

        for i, dataset in enumerate(self.datasets):
            if 'label' in dataset.plot_properties.keys():
                obs = dataset.plot_properties['label'].split('-')[0]
            else:
                obs = i

            if dataset.bandpass is not None:
                band = dataset.bandpass
            else:
                band = 'mag'

            parameters.append('{0}_S_{1}'.format(band, obs))
            parameters.append('{0}_B_{1}'.format(band, obs))

            obs_index = len(self.parameters_to_fit) + 2 * i
            for index in range(2):
                flux = values[obs_index + index]
                if flux > 0:
                    err_flux = sigmas[obs_index + index]
                    mag, err_mag = MulensModel.utils.Utils.get_mag_and_err_from_flux(flux, err_flux)
                else:
                    mag = 'neg flux'
                    err_mag = np.nan

                values[obs_index + index] = mag
                sigmas[obs_index + index] = err_mag

        df = pd.DataFrame({
            'parameter_names': parameters,
            'values': values,
            'sigmas': sigmas
        })
        df = pd.concat((
            pd.DataFrame({'parameter_names': ['chi2', 'N_data'],
                          'values': [
                              self.best['chi2'],
                              np.sum([np.sum(dataset.good) for dataset in self.datasets])],
                          'sigmas': [None, None]}), df))

        return df

    @property
    def datasets(self):
        return self.fitter.datasets

    @property
    def best(self):
        return self.fitter.best

    @property
    def results(self):
        return self.fitter.results

    @property
    def parameters_to_fit(self):
        return self.fitter.parameters_to_fit


class MMEXOFASTFitter():

    def __init__(
            self, files=None, fit_type=None, renormalize_errors=True,
            finite_source=False, limb_darkening_coeffs_gamma=None,
            limb_darkening_coeffs_u=None, mag_methods=None,
            datasets=None, coords=None, prev_results=None,
            priors=None, print_results=False, verbose=False,
            output_file=None, latex_file=None, log_file=None, emcee=True, emcee_settings=None, pool=None):

        # Output
        self.verbose = verbose
        self.log_file = log_file
        self.latex_file = latex_file

        # setup datasets.
        if datasets is not None:
            self.datasets = datasets
        else:
            self.datasets = self._create_mulensdata_objects(files)

        self.fit_type = fit_type
        self.renormalize_errors = renormalize_errors
        self.finite_source = finite_source
        self.fitter_kwargs = {
            'coords': coords, 'mag_methods': mag_methods,
            'limb_darkening_coeffs_u': limb_darkening_coeffs_u,
            'limb_darkening_coeffs_gamma': limb_darkening_coeffs_gamma}
        #print(self.fitter_kwargs)

        self.emcee = emcee
        self.emcee_settings = emcee_settings
        self.pool = pool

        # initialize additional data versions
        self._residuals = None
        self._masked_datasets = None

        # initialize params
        self._best_ef_grid_point = None

        # Fit results
        self._pspl_static_results = None
        self._fspl_static_results = None
        self._pl_parallax_results = None

        self._best_af_grid_point = None
        self._anomaly_lc_params = None
        self._binary_params = None

        if prev_results is not None:
            print(
                'prev_results NEEDS WORK! should create a MMEXOFASTFitResults object so the datasets are not needed '+
                'when creating prev_results items')

        self._results = prev_results

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
        """
        Perform the fit according to the settings established when the MMEXOFASTFitter object was created.

        :return: None
        """
        if self.fit_type is None:
            # Maybe "None" means initial mulens parameters were passed,
            # so we can go straight to a mmexofast_fit?
            raise ValueError(
                'You must set the fit_type when initializing the ' +
                'MMEXOFASTFitter(): fit_type=("point lens", "binary lens")')

        # ADD a condition to check whether point lens fits already exist...
        self.fit_point_lens()

        if self.fit_type == 'binary lens':
            self.best_af_grid_point = self.do_af_grid_search()
            if self.verbose:
                print('Best AF grid', self.best_af_grid_point)

            self.anomaly_lc_params = self.get_anomaly_lc_params()
            if self.verbose:
                print('Anomaly Params', self.anomaly_lc_params)

            if self.emcee:
                self.results = self.fit_anomaly()
                if self.verbose:
                    print('Results', self.results)

    def fit_point_lens(self):
        ### Next steps:
        # 1. Think about how to implement log file.
        # 2. Implement renormalize errors.

        results = {}

        if self.best_ef_grid_point is None:
            self.best_ef_grid_point = self.do_ef_grid_search()
            if self.verbose:
                print('Best EF grid point ', self.best_ef_grid_point)

        if 'static PSPL' in self.results.keys():
            print('is this really how you want this to work????') ###
            results['static PSPL'] = self.results['static PSPL']
            if self.verbose:
                print('static PSPL exists', self.results['static PSPL'].best)
        else:
            results['static PSPL'] = self.fit_initial_pspl_model()
            if self.verbose:
                print('Initial SFit', self.results['static PSPL'].best)

        if self.finite_source:
            results['static FSPL'] = self.fit_static_fspl_model()
            if self.verbose:
                print('SFit FSPL', self.results['static FSPL'].best)

        pl_results = self.fit_pl_parallax_models()
        for key, value in pl_results.items():
            results[key] = value

        self.results = results

        if self.renormalize_errors:
            self.renormalize_errors_and_refit()



    def renormalize_errors_and_refit(self):
        """
        Given the existing fits, take the best one and renormalize the errorbars of each dataset relative to that fit.
        Then, re-optimize all the fits with the new errorbars.
        :return:
        """
        pass

    def initialize_exozippy(self):
        """
        Get the best-fit microlensing parameters for initializing exozippy fitting.

        :return: *dict*
            items:
                'fits': *list* of *dict*
                    [{'parameters': {*dict* of ulens parameters}, 'sigmas': {*dict* of uncertainties in
                    ulensparameters}} ...]
                'errfacs': *list* of error renormalization factors for each dataset. DEFAULT: None
                'mag_methods': *list* of magnification methods following the MulensModel convention. DEFAULT: None
        """
        initializations = {'fits': [], 'errfacs': None, 'mag_methods': None}

        if self.fit_type == 'point lens':
            fits = []
            for fit in self.pl_parallax_results:
                fits.append({'parameters': fit.get_params_from_results(),
                             'sigmas': fit.get_sigmas_from_results()})

            initializations['fits'] = fits
        else:
            raise NotImplementedError('initialize_exozippy only implemented for point lens fits')

        return initializations

    def make_ulens_table(self, table_type, models=None):
        """
        Return a string consisting of a formatted table summarizing the results of the microlensing fits.

        :param table_type:
        models = *list*, Optional
            default is to make a table for all models

        :return: *str*
        """
        print('Need to add parameter heirarchy to table construction.')

        if table_type is None:
            table_type = 'ascii'

        if models is None:
            models = self.results.keys()

        results_table = None
        for name in models:
            new_column = self.results[name].format_results_as_df()
            new_column = new_column.rename(columns={'values': name, 'sigmas': 'sig [{0}]'.format(name)})

            if results_table is None:
                results_table = new_column
            else:
                results_table = results_table.merge(new_column, on="parameter_names", how="outer", sort=False)

        if table_type == 'latex':
            def fmt(name):
                if name == 'chi2':
                    return '$\chi^2$'

                parts = name.split("_")
                if len(parts) == 1:
                    return f"${name}$"
                first = parts[0]
                rest = ", ".join(parts[1:])
                return f"${first}" + "_{" + rest + "}$"

            results_table["parameter_names"] = results_table["parameter_names"].apply(fmt)

            return results_table.to_latex(index=False)
        elif table_type == 'ascii':
            with pd.option_context("display.max_rows", None,
                                   "display.max_columns", None,
                                   "display.width", None, "display.float_format", "{:f}".format):
                return results_table.to_string(index=False)
        else:
            raise NotImplementedError(table_type + ' not implemented.')

    def do_ef_grid_search(self):
        """
        Run a :py:class:`mmexofast.gridsearches.EventFinderGridSearch`
        :return: *dict* of best EventFinder grid point.
        """
        ef_grid = mmexo.EventFinderGridSearch(datasets=self.datasets)
        ef_grid.run()
        return ef_grid.best

    def fit_initial_pspl_model(self, verbose=False):
        """
        Estimate a starting point for the PSPL fitting from the EventFinder search (:py:attr:`best_ef_grid_point`)
        and then optimize the parameters using :py:class:`mmexofast.fitters.SFitFitter`.

        :param verbose: *bool* optional
        :return: :py:class:`MMEXOFASTFit` object.
        """
        pspl_est_params = mmexo.estimate_params.get_PSPL_params(self.best_ef_grid_point, self.datasets)
        if self.verbose:
            print('Initial PSPL Estimate', pspl_est_params)

        fitter = mmexo.fitters.SFitFitter(initial_model_params=pspl_est_params, datasets=self.datasets)
        fitter.run()

        return MMEXOFASTFitResults(fitter)

    def fit_static_fspl_model(self):
        """
        Use the results from the static PSPL fit (:py:attr:`pspl_static_results`) to initialize and optimize an FSPL
        fit.

        :return: :py:class:`MMEXOFASTFit` object.
        """
        init_params = self.results['static PSPL'].get_params_from_results()
        init_params['rho'] = 1.5 * init_params['u_0']
        fitter = mmexo.fitters.SFitFitter(
            initial_model_params=init_params, datasets=self.datasets, **self.fitter_kwargs)
        fitter.run()
        return MMEXOFASTFitResults(fitter)

    def fit_pl_parallax_models(self):
        """
        Use the results from the static fit (either PSPL or FSPL according to the value of `finite_source`) to
        initialize u0+ and u0- parallax fits.

        :return: *list* of 2 :py:class:`MMEXOFASTFit` objects.
        """
        if self.finite_source:
            init_params = self.results['static FSPL'].get_params_from_results()
        else:
            init_params = self.results['static PSPL'].get_params_from_results()

        init_params['pi_E_N'] = 0.
        init_params['pi_E_E'] = 0

        results = {}
        for sign in [1, -1]:
            init_params['u_0'] *= sign
            if sign >= 0:
                key = 'PL parallax (+u_0)'
            else:
                key = 'PL parallax (-u_0)'

            fitter = mmexo.fitters.SFitFitter(
                initial_model_params=init_params, datasets=self.datasets, **self.fitter_kwargs)
            fitter.run()
            results[key] = MMEXOFASTFitResults(fitter)

        return results

    def fit_parallax_grid(self, grid=None, plot=False):
        if grid is None:
            grid = {'pi_E_E': (-1, 1, 0.05), 'pi_E_N': (-2., 2., 0.1)}

        init_params = self.get_params_from_results(self.pl_parallax_results)
        parameters_to_fit = list(init_params.keys())
        parameters_to_fit.remove('pi_E_E')
        parameters_to_fit.remove('pi_E_N')
        print(parameters_to_fit)

        pi_E_E = np.arange(grid['pi_E_E'][0], grid['pi_E_E'][1] + grid['pi_E_E'][2], grid['pi_E_E'][2])
        pi_E_N = np.arange(grid['pi_E_N'][0], grid['pi_E_N'][1] + grid['pi_E_N'][2], grid['pi_E_N'][2])
        chi2 = np.zeros((len(pi_E_E), len(pi_E_N)))
        for i, east in enumerate(pi_E_E):
            init_params['pi_E_E'] = east
            for j, north in enumerate(pi_E_N):
                init_params['pi_E_N'] = north
                fitter = mmexo.fitters.SFitFitter(
                    initial_model_params=init_params, parameters_to_fit=parameters_to_fit, datasets=self.datasets, **self.fitter_kwargs)
                fitter.run()
                print(fitter.best)
                chi2[i, j] = fitter.best['chi2']

    def set_residuals(self, pspl_params):
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
        self.set_residuals(self.initial_pspl_params)
        af_grid = mmexo.AnomalyFinderGridSearch(residuals=self.residuals)
        # May need to update value of teff_min
        af_grid.run()
        return af_grid.best

    def get_anomaly_lc_params(self):
        estimator = mmexo.estimate_params.AnomalyPropertyEstimator(
            datasets=self.datasets, pspl_params=self.initial_pspl_params, af_results=self.best_af_grid_point)
        return estimator.get_anomaly_lc_parameters()

    def fit_anomaly(self):
        # So far, this only fits wide planet models in the GG97 limit.
        #print(self.anomaly_lc_params)
        wide_planet_fitter = mmexo.fitters.WidePlanetFitter(
            datasets=self.datasets, anomaly_lc_params=self.anomaly_lc_params,
            emcee_settings=self.emcee_settings, pool=self.pool)
        if self.verbose:
            wide_planet_fitter.estimate_initial_parameters()
            print('Initial 2L1S Wide Model', wide_planet_fitter.initial_model)
            print('mag methods', wide_planet_fitter.mag_methods)

        wide_planet_fitter.run()
        return wide_planet_fitter.best

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
    def pspl_static_results(self):
        """
        Results from fitting a static PSPL Model
        :return: :py:class:`MMEXOFASTFit` object.
        """
        return self._pspl_static_results

    @pspl_static_results.setter
    def pspl_static_results(self, value):
        self._pspl_static_results = value

    @property
    def fspl_static_results(self):
        """
        Results from fitting a static FSPL Model
        :return: :py:class:`MMEXOFASTFit` object.
        """
        return self._fspl_static_results

    @fspl_static_results.setter
    def fspl_static_results(self, value):
        self._fspl_static_results = value

    @property
    def pl_parallax_results(self):
        """
        Results from fitting a static FSPL Model
        :return: *list* of :py:class:`MMEXOFASTFit` objects.
        """
        return self._pl_parallax_results

    @pl_parallax_results.setter
    def pl_parallax_results(self, value):
        self._pl_parallax_results = value


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
