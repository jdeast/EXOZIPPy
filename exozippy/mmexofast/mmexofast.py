"""
High-level functions for fitting microlensing events.
"""
import os.path
import numpy as np
import copy

import MulensModel
import MulensModel as mm
import astropy.units
import sfit_minimizer as sfit
import exozippy as mmexo


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
                 output_file=None):
        # setup datasets.
        if datasets is not None:
            self.datasets = datasets
        else:
            self.datasets = self._create_mulensdata_objects(files)

        self.fit_type = fit_type

        # initialize additional data versions
        self._residuals = None
        self._masked_data = None

        # initialize params
        self._best_ef_grid_point = None
        self._pspl_params = None
        self._best_af_grid_point = None
        self._binary_params = None

        self._results = None

    def _create_mulensdata_objects(self, files):
        if isinstance(files, (str)):
            files = [files]

        raise NotImplementedError('Need to write code to ingest data.')

    def fit(self):
        if self.fit_type is None:
            # Maybe "None" means initial mulens parameters were passed,
            # so we can go straight to a mmexofast_fit?
            raise ValueError('You must set the fit_type.')

        # Find initial Point Lens model
        self.best_ef_grid_point = self.do_ef_grid_search()
        self.pspl_params = self.get_initial_pspl_params()
        self.pspl_params = self.do_sfit(self.datasets)
        if self.fit_type == 'point lens':
            # Do the full MMEXOFAST fit to get physical parameters
            self.results = self.do_mmexofast_fit()
            return self.results
        elif self.fit_type == 'binary_lens':
            # Find the initial planet parameters
            self.best_af_grid_params = self.do_af_grid_search()
            self.pspl_params = self.refine_pspl_params()
            self.binary_params = self.get_initial_2L1S_params()

            # Do the full MMEXOFAST fit to get physical parameters
            self.results = self.do_mmexofast_fit()
            return self.results
        else:
            raise ValueError(
                'fit_type not recognized. Your value', self.fit_type)

    def do_ef_grid_search(self):
        # Should probably scrape t_0_1 from the filenames
        ef_grid = mmexo.EventFinderGridSearch(datasets=self.datasets)
        ef_grid.run()
        return ef_grid.best

    def get_initial_pspl_params(self, verbose=False):
        t_0 = self.best_ef_grid_point['t_0']
        if self.best_ef_grid_point['j'] == 1:
            u_0 = 0.01
        elif self.best_ef_grid_point['j'] == 2:
            u_0s = [0.1, 0.3, 0.5, 1.0, 1.3, 2.0]
            chi2s = []
            for u_0 in u_0s:
                t_E = self.best_ef_grid_point['t_eff'] / u_0
                params = {'t_0': t_0, 't_E': t_E, 'u_0': u_0}
                event = MulensModel.Event(
                    datasets=self.datasets, model=MulensModel.Model(params))
                chi2s.append(event.get_chi2())

            index = np.nanargmin(chi2s)
            u_0 = u_0s[index]
            if verbose:
                print('u0s', u_0s)
                print('chi2s', chi2s)
                print('selected', index, u_0)

        else:
            raise ValueError(
                'j may only be 1 or 2. Your input: ', self.best_ef_grid_point)

        t_E = self.best_ef_grid_point['t_eff'] / u_0

        return {'t_0': t_0, 't_E': t_E, 'u_0': u_0}

    def do_sfit(self, datasets, verbose=False):
        param_sets = [['t_0', 't_E'], ['t_0', 'u_0', 't_E']]

        params = self.pspl_params
        for i in range(len(param_sets)):
            parameters_to_fit = param_sets[i]
            event = MulensModel.Event(
                datasets=datasets, model=MulensModel.Model(params))
            event.fit_fluxes()

            my_func = sfit.mm_funcs.PointLensSFitFunction(
                event, parameters_to_fit)

            initial_guess = []
            for key in parameters_to_fit:
                if isinstance(params[key], (astropy.units.Quantity)):
                    initial_guess.append(params[key].value)
                else:
                    initial_guess.append(params[key])

            for i in range(len(datasets)):
                initial_guess.append(event.fits[i].source_flux)
                initial_guess.append(event.fits[i].blend_flux)

            result = sfit.minimize(
                my_func, x0=initial_guess, tol=1e-5,
                options={'step': 'adaptive'}, verbose=verbose)

            if verbose:
                print(result)

            params = my_func.event.model.parameters.parameters

        return params

    def do_mmexofast_fit(self):
        raise NotImplementedError(
            'do_mmexofast_fit needs to be implemented')

    def set_datasets_with_anomaly_masked(self, n_mask=3):
        masked_datasets = []
        for dataset in self.datasets:
            masked_datasets.append(copy.copy(dataset))

        for dataset in masked_datasets:
            index = ((dataset.time >
                     self.best_af_grid_point['t_0'] -
                     n_mask * self.best_af_grid_point['t_eff']) &
                     (dataset.time <
                      self.best_af_grid_point['t_0'] +
                      n_mask * self.best_af_grid_point['t_eff']))
            dataset.bad = index

        self.masked_datasets = masked_datasets

    def refine_pspl_params(self):
        self.set_datasets_with_anomaly_masked()
        results = self.do_sfit(self.masked_datasets)
        return results

    def set_residuals(self):
        event = mm.Event(
            datasets=self.datasets, model=mm.Model(self.pspl_params))
        event.fit_fluxes()
        residuals = []
        for i, dataset in enumerate(self.datasets):
            res, err = event.fits[i].set_residuals(phot_fmt='flux')
            residuals.append(
                mm.MulensData(
                    [dataset.time, res, err], phot_fmt='flux',
                    bandpass=dataset.bandpass,
                    ephemerides_file=dataset.ephemerides_file))

        self.residuals = residuals

    def do_af_grid_search(self):
        self.set_residuals()
        af_grid = mmexo.AnomalyFinderGridSearch(residuals=self.residuals)
        # May need to update value of teff_min
        af_grid.run()
        return af_grid.best

    def get_dmag(self):
        pspl_event = MulensModel.Event(
            datasets=self.masked_datasets,
            model=MulensModel.Model(self.pspl_params))
        source_flux, blend_flux = pspl_event.get_ref_fluxes()
        flux_pspl = source_flux[0] * pspl_event.model.get_magnification(
            self.af_grid_params['t_0']) + blend_flux

        # Can this be cleaned up?
        af_grid_search = mmexo.gridsearches.AnomalyFinderGridSearch(
            self.residuals)
        trimmed_datasets = af_grid_search.get_trimmed_datasets(
            self.af_grid_params)
        ef_sfit = mmexo.gridsearches.EFSFitFunction(
            trimmed_datasets, self.af_grid_params)
        ef_sfit.update_all()
        new_theta = ef_sfit.theta + ef_sfit.get_step()
        ef_sfit.update_all(new_theta)

        i = pspl_event.data_ref
        fs = ef_sfit.theta[2 * i]
        fb = ef_sfit.theta[2 * i + 1]
        mag = ef_sfit.get_magnification(self.af_grid_params['t_0'])
        d_flux_anom = fs * mag * fb

        dmag = -2.5 * np.log10((flux_pspl + d_flux_anom) / flux_pspl)
        return dmag[0]

    def get_initial_2L1S_params(self):
        dmag = self.get_dmag()
        params = {
            't_0': self.pspl_params['t_0'], 'u_0': self.pspl_params['u_0'],
            't_E': self.pspl_params['t_E'], 't_pl': self.best_af_grid_point['t_0'],
            'dt': 2. * self.best_af_grid_point['t_eff'], 'dmag': dmag}
        binary_params = mmexo.estimate_params.get_wide_params(params)
        return binary_params

    @property
    def residuals(self):
        return self._residuals

    @residuals.setter
    def residuals(self, value):
        self._residuals = value

    @property
    def masked_data(self):
        return self._masked_data

    @masked_data.setter
    def masked_data(self, value):
        self._masked_data = value

    @property
    def best_ef_grid_point(self):
        return self._best_ef_grid_point

    @best_ef_grid_point.setter
    def best_ef_grid_point(self, value):
        self._best_ef_grid_point = value

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
    def binary_params(self):
        return self._binary_params

    @property
    def results(self):
        return self._results

    @results.setter
    def results(self, value):
        self._results = value


#### CODE BELOW HERE IS PROBABLY DEFUNCT.
class MMEXOFASTSingleLensFitter():

    def __init__(self, datafiles=None, data=None):
        if datafiles is not None:
            if isinstance(datafiles, (str)):
                datafiles = [datafiles]

            self.data = self._read_data(datafiles)

        if data is not None:
            self.data = data

        self.initial_mulens_parameters = None
        self.initial_physical_parameters = None
        self.results = None

    def _read_data(self, datafiles):
        """
        Read in the datafiles and make them into a *list* of
        mm.MulensData objects
        """
        data = []
        for datafile in datafiles:
            dataset = mm.MulensData(
                file_name=datafile,
                plot_properties={'label': os.path.basename(datafile)})
            data.append(dataset)

        return data

    def get_initial_mulens_parameters(self):
        """
        Figure out initial values for the microlensing parameters
        :return:
        """
        ef = mmexo.gridsearches.EventFinder(self.data) # architecture issue
        ef.fit_grid()

        init_sfit_params = mmexo.convert_ef_params(ef.best_grid_point)
        event = mm.Event(datasets=self.data, model=mm.Model(init_sfit_params))

        # Need to break this out into a separate function
        parameters_to_fit = ['t_0', 'u_0', 't_E']
        initial_guess = []
        for key in parameters_to_fit:
            if key == 't_E':
                initial_guess.append(event.model.parameters.parameters[key].value)
            else:
                initial_guess.append(event.model.parameters.parameters[key])

        for i in range(len(self.data)):
            initial_guess.append(1.0)
            initial_guess.append(0.0)

        my_func = sfit.mm_funcs.PointLensSFitFunction(event, parameters_to_fit)

        # Do the fit
        result = sfit.minimize(
            my_func, x0=initial_guess, tol=1e-5,
            options={'step': 'adaptive'}, verbose=True)

        # Probably want to convert to dict:
        self.initial_mulens_parameters = result.x

        # Need to repeat with FSEffects?

    def get_initial_physical_parameters(self):
        pass

    def mcmc_fit(self):
        pass

    def fit(self):
        self.get_initial_mulens_parameters()
        self.get_initial_physical_parameters()
        self.mcmc_fit()
        return self.results
