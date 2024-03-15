"""
High-level functions for fitting microlensing events.
"""
import os.path

import MulensModel as mm
import sfit_minimizer as sfit


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
        ef = mmexo.EventFinder(self.data) # architecture issue
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
