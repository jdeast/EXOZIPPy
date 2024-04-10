"""
High-level functions for fitting microlensing events.
"""
import os.path
import numpy as np

import MulensModel
import MulensModel as mm
import astropy.units
import sfit_minimizer as sfit
import exozippy as mmexo


def fit(files=None, coords=None, priors=None, fit_type=None,
        print_results=False, verbose=False, output_file=None):
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
    if isinstance(files, (str)):
        files = [files]

    if files is not None:
        datasets = create_mulensdata_objects(files)

    if fit_type is None:
        # Maybe "None" means initial mulens parameters were passed, so we can
        # go straight to a mmexofast_fit?
        raise ValueError('You must set the fit_type.')

    # Find initial Point Lens model
    best_ef_grid_params = do_ef_grid_search(datasets)
    initial_pspl_params = get_initial_pspl_params(
        datasets, best_ef_grid_params)
    best_pspl_params = do_sfit(datasets, initial_pspl_params)
    if fit_type == 'point lens':
        # Do the full MMEXOFAST fit to get physical parameters
        point_lens_results = do_mmexofast_fit(datasets, best_pspl_params)
        return point_lens_results
    elif fit_type == 'binary_lens':
        # Find the initial planet parameters
        initial_af_grid_params = do_af_grid_search(datasets, best_pspl_params)
        initial_2L1S_params = get_initial_2L1S_params(initial_af_grid_params)

        # Do the full MMEXOFAST fit to get physical parameters
        binary_lens_results = do_mmexofast_fit(datasets, initial_2L1S_params)
        return binary_lens_results
    else:
        raise ValueError('fit_type not recognized. Your value', fit_type)


def do_ef_grid_search(datasets):
    # Should probably scrape t_0_1 from the filenames
    ef_grid = mmexo.EventFinderGridSearch(datasets=datasets)
    ef_grid.run()
    return ef_grid.best


def get_initial_pspl_params(datasets, best_ef_params, verbose=False):
    t_0 = best_ef_params['t_0']
    if best_ef_params['j'] == 1:
        u_0 = 0.01
    elif best_ef_params['j'] == 2:
        u_0s = [0.1, 0.3, 0.5, 1.0, 1.3, 2.0]
        chi2s = []
        for u_0 in u_0s:
            t_E = best_ef_params['t_eff'] / u_0
            params = {'t_0': t_0, 't_E': t_E, 'u_0': u_0}
            event = MulensModel.Event(
                datasets=datasets, model=MulensModel.Model(params))
            chi2s.append(event.get_chi2())

        index = np.nanargmin(chi2s)
        u_0 = u_0s[index]
        if verbose:
            print('u0s', u_0s)
            print('chi2s', chi2s)
            print('selected', index, u_0)

    else:
        raise ValueError(
            'j may only be 1 or 2. Your input: ', best_ef_params)

    t_E = best_ef_params['t_eff'] / u_0

    return {'t_0': t_0, 't_E': t_E, 'u_0': u_0}


def do_sfit(datasets, initial_params, verbose=False):
    param_sets = [['t_0', 't_E'], ['t_0', 'u_0', 't_E']]

    params = initial_params
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


def do_af_grid_search(datasets, best_pspl_params):
    event = mm.Event(datasets=datasets, model=mm.Model(best_pspl_params))
    event.fit_fluxes()
    residuals = []
    for i, dataset in enumerate(datasets):
        res, err = event.fit[i].get_residuals(phot_fmt='flux')
        residuals.append(
            mm.MulensData(
                [dataset.time, res, err], phot_fmt='flux',
                bandpass=dataset.bandpass,
                ephemerides_file=dataset.ephemerides_file))

    af_grid = mmexo.AnomalyFinderGridSearch(datasets=residuals, teff_min=0.3)
    # May need to update value of teff_min
    # Is the AnomalyFinderGridSearch really the same as EventFinderGridSearch?
    # Or are they just based on the same principles? (but differ in the details)
    af_grid.run()
    return af_grid.best


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
