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
        self.fitter = self.fitter

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
            datasets=None, coords=None,
            priors=None, print_results=False, verbose=False,
            output_file=None, log_file=None, emcee=True, emcee_settings=None, pool=None):
        self.verbose = verbose
        if log_file is not None:
            self.log_file = log_file

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
        print(self.fitter_kwargs)

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

        self.best_ef_grid_point = self.do_ef_grid_search()
        if self.verbose:
            print('Best EF grid point ', self.best_ef_grid_point)

        self.pspl_static_results = self.get_initial_pspl()
        if self.verbose:
            print('Initial SFit', self.pspl_static_results.best)

        if self.fit_type == 'point lens':
            if self.finite_source:
                self.fspl_static_results = self.fit_fspl()
                if self.verbose:
                    print('SFit FSPL', self.fspl_static_results.best)

            self.pl_parallax_results = self.fit_pl_parallax()
            if self.verbose:
                for i in range(2):
                    print('SFit w/par', i+1, self.pl_parallax_results[i].best)

            if self.renormalize_errors:
                self.renormalize_errors_and_refit()

        elif self.fit_type == 'binary lens':
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

    def get_results_df(self, model):
        if self.fit_type == 'point lens':
            if (model.lower() == 'static') & self.finite_source:
                return self.fspl_static_results.format_results_as_df().rename(columns=lambda c: c + "_static" if c != "parameter_names" else c)
            elif (model.lower() == 'static') or (model.lower() == 'point lens static'):
                return self.pspl_static_results.format_results_as_df().rename(columns=lambda c: c + "_pl_static" if c != "parameter_names" else c)
            elif model.lower() == 'parallax':
                df = None
                for results in self.pl_parallax_results:
                    formatted_df = results.format_results_as_df()
                    if df is None:
                        df = formatted_df
                    else:
                        df["_order"] = np.arange(len(df))
                        df = df.merge(
                            formatted_df, on="parameter_names", how="outer", suffixes=("_par_p", "_par_m"), sort=False)
                        df = df.sort_values("_order", na_position="last").drop(columns='_order')

                return df
            else:
                raise ValueError(
                    'argument for point lens fits can be "static", "point lens static" or "parallax". Your value: ',
                    model)

        else:
            raise NotImplementedError('Only point lenses have been implemented.')

    def make_ulens_table(self, table_type):
        """
        Return a string consisting of a formatted table summarizing the results of the microlensing fits.

        :param type:
        :return: *str*
        """
        if table_type is None:
            table_type = 'ascii'

        static = self.get_results_df('static')
        parallax = self.get_results_df('parallax')

        if self.finite_source:
            point_lens = self.get_results_df('point lens static')
            results = point_lens.merge(static, on="parameter_names", how="outer", sort=False)
        else:
            results = static

        parallax["_order"] = np.arange(len(parallax))
        results = results.merge(parallax, on="parameter_names", how="outer", sort=False)
        results = results.sort_values("_order", na_position="last").drop(columns='_order')

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

            results["parameter_names"] = results["parameter_names"].apply(fmt)
            results.columns = [fmt(c) for c in results.columns]

            return results.to_latex(index=False)
        elif table_type == 'ascii':
            with pd.option_context("display.max_rows", None,
                                   "display.max_columns", None,
                                   "display.width", None, "display.float_format", "{:f}".format):
                return results.to_string(index=False)
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

    def get_initial_pspl(self, verbose=False):
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

    def fit_fspl(self):
        """
        Use the results from the static PSPL fit (:py:attr:`pspl_static_results`) to initialize and optimize an FSPL
        fit.

        :return: :py:class:`MMEXOFASTFit` object.
        """
        init_params = self.pspl_static_results.get_params_from_results()
        init_params['rho'] = 1.5 * init_params['u_0']
        fitter = mmexo.fitters.SFitFitter(
            initial_model_params=init_params, datasets=self.datasets, **self.fitter_kwargs)
        #print('mmexo237', fitter.mag_methods)
        fitter.run()
        return MMEXOFASTFitResults(fitter)

    def fit_pl_parallax(self):
        """
        Use the results from the static fit (either PSPL or FSPL according to the value of `finite_source`) to
        initialize u0+ and u0- parallax fits.

        :return: *list* of 2 :py:class:`MMEXOFASTFit` objects.
        """
        if self.finite_source:
            init_params = self.fspl_static_results.get_params_from_results()
        else:
            init_params = self.pspl_static_results.get_params_from_results()

        init_params['pi_E_N'] = 0.
        init_params['pi_E_E'] = 0

        results = []
        for sign in [1, -1]:
            init_params['u_0'] *= sign
            fitter = mmexo.fitters.SFitFitter(
                initial_model_params=init_params, datasets=self.datasets, **self.fitter_kwargs)
            fitter.run()
            results.append(MMEXOFASTFitResults(fitter))

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
