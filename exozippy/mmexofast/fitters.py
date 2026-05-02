import MulensModel
import numpy as np
import pandas as pd
import sfit_minimizer as sfit
import emcee
from multiprocessing import Pool, cpu_count
import os

import exozippy.mmexofast as mmexo


class MulensFitter():
    """
    Parent class for microlensing model fitters.

    Parameters
    ----------
    datasets : list
        List of MulensModel.MulensData objects.
    initial_model_params : dict
        Initial parameters of the model.
    parameters_to_fit : list, optional
        Parameters to be fitted. If None, all keys in initial_model_params are fitted.
    sigmas : list, optional
        Step sizes for each parameter in parameters_to_fit.
    mag_methods : list, optional
        Magnification methods specification; see MulensModel.Model.set_magnification_methods.
    coords : str or astropy.SkyCoord, optional
        Sky coordinates of the event.
    limb_darkening_coeffs_gamma : dict, optional
        Limb darkening gamma coefficients keyed by photometric band.
    limb_darkening_coeffs_u : dict, optional
        Limb darkening u coefficients keyed by photometric band.
    fix_source_flux : dict, optional
        Source flux(es) to hold fixed. The dataset is the key and the value to
        be fixed is the value. See MulensModel.Event and MulensModel.FitData.
    fix_blend_flux : dict, optional
        Blend flux to hold fixed. The dataset is the key and the value to
        be fixed is the value. See MulensModel.Event and MulensModel.FitData.
    verbose : bool, optional
        If True, print progress information. Default is False.
    pool : multiprocessing.Pool, optional
        Pool for parallel computation.
    """

    def __init__(
            self, datasets=None, initial_model_params=None, parameters_to_fit=None, sigmas=None,
            mag_methods=None, coords=None, limb_darkening_coeffs_gamma=None,
            limb_darkening_coeffs_u=None,
            fix_source_flux=None, fix_blend_flux=None,
            verbose=False, pool=None):
        self._initial_model = None
        self._best = None
        self._results = None

        self.datasets = datasets

        self.initial_model_params = initial_model_params
        self.parameters_to_fit = parameters_to_fit
        self.sigmas = sigmas

        self.mag_methods = mag_methods
        self.limb_darkening_coeffs_gamma = limb_darkening_coeffs_gamma
        self.limb_darkening_coeffs_u = limb_darkening_coeffs_u
        self.fix_source_flux = fix_source_flux
        self.fix_blend_flux = fix_blend_flux

        self.coords = coords

        self.verbose = verbose
        self.pool = pool

    def run(self):
        """
        Run the fitter. Implemented by subclasses.

        Notes
        -----
        This method is not formally declared as abstract but is intended to be
        overridden by subclasses. Consider using ``abc.abstractmethod`` to enforce
        this, consistent with ``set_event_parameters()`` and
        ``make_starting_vector()`` in ``AnomalyFitter``.
        """
        pass

    def get_model(self):
        """
        Create a MulensModel.Model with best-fit parameters, or initial parameters
        if no fit has been run yet.

        If set, magnification methods and limb darkening coefficients (u or gamma)
        are applied to the model before returning.

        Returns
        -------
        MulensModel.Model
            Model with best-fit or initial parameters, with magnification methods
            and limb darkening coefficients applied if provided.
        """
        # Use best-fit if available, else initial
        if self.best is not None:
            params = dict(self.best)
            params.pop('chi2', None)  # Remove chi2 key
        else:
            params = self.initial_model_params

        model = MulensModel.Model(params)

        # Apply magnification methods
        if self.mag_methods is not None:
            model.set_magnification_methods(self.mag_methods)

        # Apply limb darkening coefficients
        if self.limb_darkening_coeffs_u is not None:
            for band, value in self.limb_darkening_coeffs_u.items():
                model.set_limb_coeff_u(band, value)

        if self.limb_darkening_coeffs_gamma is not None:
            for band, value in self.limb_darkening_coeffs_gamma.items():
                model.set_limb_coeff_gamma(band, value)

        return model

    def get_event(self):
        """
        Create a MulensModel.Event using the current datasets, model, coordinates,
        and any fixed source or blend fluxes.

        Returns
        -------
        MulensModel.Event
            Event constructed from the current fitter state.

        See Also
        --------
        get_model : Creates the MulensModel.Model used by this event.
        """
        event = MulensModel.Event(
            datasets=self.datasets, model=self.get_model(), coords=self.coords,
            fix_source_flux=self.fix_source_flux, fix_blend_flux=self.fix_blend_flux)

        return event

    def get_diagnostic_str(self):
        """
        Build a diagnostic string summarising the current event fit.

        Calls ``fit_fluxes()`` on the event before building the string. The
        returned string includes the model parameters, and for each dataset: the
        label, number of good data points, chi2, source flux(es), and blend flux.

        Returns
        -------
        str
            Formatted string containing event and dataset fit information.

        Notes
        -----
        Despite the name, this method does not print anything. Use
        ``print(get_diagnostic_str())`` to print the output.
        """
        event = self.get_event()
        event.fit_fluxes()
        msg = f'\n---- Event Info ----\nModel:\n{event.model}\n\nDatasets:'
        msg += '\n{0:20} {1:>4} {2:>12} {3} {4}'.format('Label', 'N_good', 'chi2', 'f_source', 'f_blend')
        for i, dataset in enumerate(event.datasets):
            msg += ('\n{0:20} {1:4} {2:12.2f} {3} {4}'.format(
                dataset.plot_properties['label'], np.sum(dataset.good),
                event.get_chi2_for_dataset(i),
                event.fits[i].source_fluxes,
                event.fits[i].blend_flux))

        msg += '\n--------------------\n'
        return msg

    @property
    def best(self):
        """
        Best-fit model parameters and chi2.

        Returns
        -------
        dict or None
            Dictionary of best-fit model parameter names and values, with an
            additional ``'chi2'`` key. Returns None if no fit has been run yet.
        """
        return self._best

    @best.setter
    def best(self, params_dict):
        self._best = params_dict

    @property
    def results(self):
        """
        Full results object from the fitter.

        Returns
        -------
        dict or None
            Full results from the fitting routine. The structure depends on the
            subclass: for example, ``SFitFitter`` stores the result object returned
            by ``sfit.minimize()``. Returns None if no fit has been run yet.

        Notes
        -----
        For the best-fit model parameters specifically, use ``best`` instead.
        """
        return self._results

    @results.setter
    def results(self, value):
        self._results = value

    @property
    def initial_model_params(self):
        """
        Initial model parameters used as the starting point for the fit.

        Parameters
        ----------
        params_dict : dict or None
            Dictionary of model parameter names and values. Must be a dict or
            None; raises ValueError otherwise.

        Returns
        -------
        dict or None
            Dictionary of model parameter names and values.

        Raises
        ------
        ValueError
            If set with a value that is neither None nor a dict.
        """
        return self._initial_model_params

    @initial_model_params.setter
    def initial_model_params(self, params_dict):
        if (params_dict is not None) and (not isinstance(params_dict, dict)):
            raise ValueError('initial_model must be set with either *None* or *dict*.')

        self._initial_model_params = params_dict

    @property
    def parameters_to_fit(self):
        """
        List of model parameters to be fitted.

        If not explicitly set, defaults to all keys in ``initial_model_params``.

        Parameters
        ----------
        params_dict : list, tuple, or None
            Names of parameters to fit. Must be a list, tuple, or None; raises
            ValueError otherwise. If None, all keys in ``initial_model_params``
            will be fitted.

        Returns
        -------
        list
            Names of parameters to be fitted.

        Raises
        ------
        ValueError
            If set with a value that is neither None, a list, nor a tuple.
        """
        if self._parameters_to_fit is None:
            self._parameters_to_fit = list(self.initial_model_params.keys())

        return self._parameters_to_fit

    @parameters_to_fit.setter
    def parameters_to_fit(self, params_dict):
        if (params_dict is not None) and (not isinstance(params_dict, (list, tuple))):
            raise ValueError('parameters_to_fit must be set with either *None* or *list* or *tuple*.')

        self._parameters_to_fit = params_dict


class SFitFitter(MulensFitter):
    """
    Fit a point lens model to the data using the SFit method.

    Wraps ``sfit.minimize()`` with a ``PointLensSFitFunction``. First attempts
    the fit with an adaptive step size; if unsuccessful, retries with a fixed
    step size of 0.001 and a maximum of 10000 iterations.

    All parameters are inherited from ``MulensFitter``.

    See Also
    --------
    MulensFitter : Parent class defining all constructor parameters.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self):
        """
        Fit the point lens model using the SFit method.

        Constructs an initial guess vector from ``initial_model_params`` and the
        source and blend fluxes from an initial ``fit_fluxes()`` call. First
        attempts minimization with an adaptive step size; if unsuccessful, retries
        with a fixed step size of 0.001 and a maximum of 10000 iterations.

        On completion, sets ``results`` to the full ``sfit.minimize()`` result
        object and ``best`` to the best-fit model parameters plus ``'chi2'``.

        Notes
        -----
        ``event.fits[i].source_flux`` should be ``event.fits[i].source_fluxes``
        for MMv3, where source fluxes are returned as an array. The way the value
        is subsequently appended to ``initial_guess`` may also need revisiting.
        See also ``get_diagnostic_str()``, which correctly uses ``source_fluxes``.
        """
        event = self.get_event()
        event.fit_fluxes()

        my_func = sfit.mm_funcs.PointLensSFitFunction(
            event, self.parameters_to_fit)

        initial_guess = [self.initial_model_params[key] for key in self.parameters_to_fit]
        for i in range(len(self.datasets)):
            initial_guess.append(event.fits[i].source_flux)
            initial_guess.append(event.fits[i].blend_flux)

        result = sfit.minimize(
            my_func, x0=initial_guess, tol=1e-5,
            options={'step': 'adaptive'}, verbose=self.verbose)

        if self.verbose:
            print(result)

        if not result.success:
            result = sfit.minimize(
                my_func, x0=initial_guess, tol=1e-5, max_iter=10000,
                options={'step': 0.001}, verbose=self.verbose)
            if self.verbose:
                print(result)


        self.results = result
        best = my_func.event.model.parameters.parameters
        best['chi2'] = my_func.event.get_chi2()
        self.best = best


class AnomalyFitter(MulensFitter):
    """
    Base class for fitting microlensing anomalies using emcee.

    Extends ``MulensFitter`` with emcee-based MCMC sampling and likelihood,
    prior, and probability functions. Intended to be subclassed for specific
    anomaly types (e.g. ``WidePlanetFitter``).

    Parameters
    ----------
    anomaly_lc_params : dict, optional
        Parameters describing the anomaly light curve. Used to estimate
        initial model parameters in ``estimate_initial_parameters()``.
        See ``mmexo.estimate_params.AnomalyPropertyEstimator.get_anomaly_lc_parameters()``
        for the expected structure.
    datasets : list, optional
        List of MulensModel.MulensData objects. Inherited from ``MulensFitter``
        but also accepted here directly.

    Class Attributes
    ----------------
    default_emcee_settings : dict
        Default emcee settings: ``{'n_walkers': 40, 'n_burn': 500, 'n_steps': 1000}``.

    Notes
    -----
    ``set_event_parameters()`` and ``make_starting_vector()`` must be
    implemented by subclasses. ``estimate_initial_parameters()`` is also
    intended to be overridden.

    ``datasets`` appears in both ``MulensFitter.__init__()`` and
    ``AnomalyFitter.__init__()``, which may cause confusion. This could
    be simplified by relying solely on the parent class argument.
    """

    default_emcee_settings = {'n_walkers': 40, 'n_burn': 500, 'n_steps': 1000}

    def __init__(self, datasets=None, anomaly_lc_params=None, **kwargs):
        super().__init__(**kwargs)
        self.anomaly_lc_params = anomaly_lc_params
        self.datasets = datasets

    def estimate_initial_parameters(self):
        """
        Estimate initial model parameters from the anomaly light curve parameters.

        Intended to be overridden by subclasses. See ``WidePlanetFitter.estimate_initial_parameters()``
        for an example implementation.

        Notes
        -----
        This method is not formally declared as abstract but is intended to be
        overridden by subclasses. Consider using ``abc.abstractmethod`` to enforce
        this, consistent with ``set_event_parameters()`` and
        ``make_starting_vector()``.
        """
        pass

    def get_parameter_name(self, parameter):
        """
        Strip the ``log_`` prefix from a parameter name if present.

        Parameters
        ----------
        parameter : str
            Parameter name, e.g. ``'log_rho'`` or ``'t_E'``.

        Returns
        -------
        str
            Parameter name with ``log_`` prefix removed if present, e.g.
            ``'rho'`` or ``'t_E'``.
        """
        if 'log_' in parameter:
            key = parameter[4:]
        else:
            key = parameter

        return key

    def set_event_parameters(self, theta, event):
        """
        Set the event model parameters from a parameter vector.

        Must be implemented by subclasses.

        Parameters
        ----------
        theta : array-like
            Parameter vector corresponding to ``parameters_to_fit``.
        event : MulensModel.Event
            Event object whose model parameters will be updated.

        Returns
        -------
        MulensModel.Event
            Event object with updated model parameters.

        Raises
        ------
        NotImplementedError
            If called on ``AnomalyFitter`` directly rather than a subclass.
        """
        raise NotImplementedError('You need to implement set_event_parameters() for this class.')
        # return mm.Event()

    def make_starting_vector(self):
        """
        Construct the starting vector for the emcee sampler.

        Must be implemented by subclasses.

        Returns
        -------
        list
            Starting vector for the emcee sampler, typically a list of
            ``n_walkers`` parameter vectors.

        Raises
        ------
        NotImplementedError
            If called on ``AnomalyFitter`` directly rather than a subclass.
        """
        raise NotImplementedError('You need to implement make_starting_vector() for this class.')

    def ln_like(self, theta):
        """
        Log-likelihood function for the emcee sampler.

        Computes the log-likelihood as ``-0.5 * chi2``. If ``'temperature'`` is
        present in ``emcee_settings``, chi2 is divided by the temperature squared
        before computing the log-likelihood, implementing simulated annealing.

        Parameters
        ----------
        theta : array-like
            Parameter vector corresponding to ``parameters_to_fit``.

        Returns
        -------
        float
            Log-likelihood value, or ``-np.inf`` if the model could not be
            evaluated.

        Notes
        -----
        The bare ``except`` clause silently catches all exceptions, which may
        make debugging difficult. Consider catching specific exceptions instead.
        """

        self.event = theta
        try:
            chi2 = self.event.get_chi2()
            if 'temperature' in self.emcee_settings.keys(): #['temperature'] is not None:
                chi2 /= self.emcee_settings['temperature'] ** 2
        except:
            return -np.inf

        #print(chi2, theta)
        return -0.5 * chi2

    def ln_prior(self, theta):
        """
        Log-prior function for the emcee sampler.

        Implements flat priors with hard boundaries. Rejects models where
        ``t_E``, ``rho``, ``q``, or ``s`` are non-positive.

        Parameters
        ----------
        theta : array-like
            Parameter vector corresponding to ``parameters_to_fit``.

        Returns
        -------
        float
            0.0 if the parameters are within the prior bounds, ``np.inf``
            otherwise.

        Notes
        -----
        Returns ``np.inf`` (rather than the more conventional ``-np.inf``)
        for rejected models. This is handled correctly by ``ln_prob()``, which
        checks ``np.isfinite(ln_prior_)``.
        """
        for key, value in zip(self.parameters_to_fit, theta):
            if ((key == 't_E') or (key == 'rho') or (key == 'q') or (key == 's')) and (value <= 0.):
                return np.inf

        return 0.0

    def ln_prob(self, theta):
        """
        Log-probability function for the emcee sampler.

        Combines the log-prior and log-likelihood. Returns ``-np.inf`` if the
        prior rejects the model, if the likelihood cannot be evaluated, or if
        the likelihood is NaN (e.g. due to negative source fluxes).

        Parameters
        ----------
        theta : array-like
            Parameter vector corresponding to ``parameters_to_fit``.

        Returns
        -------
        float
            Sum of log-prior and log-likelihood, or ``-np.inf`` if the model
            is rejected.

        See Also
        --------
        ln_prior : Log-prior function.
        ln_like : Log-likelihood function.
        """
        ln_prior_ = self.ln_prior(theta)
        if not np.isfinite(ln_prior_):
            return -np.inf
        ln_like_ = self.ln_like(theta)

        # In the cases that source fluxes are negative we want to return
        # these as if they were not in priors.
        if np.isnan(ln_like_):
            return -np.inf

        return ln_prior_ + ln_like_


class WidePlanetFitter(AnomalyFitter):
    """
    Fit a wide binary lens (planet) model to the data using emcee.

    Extends ``AnomalyFitter`` for the specific case of a wide planet geometry.
    By default, fits ``['t_0', 'u_0', 't_E', 'log_rho', 'log_s', 'log_q', 'd_xsi']``
    with corresponding default sigmas ``[0.1, 0.01, 0.01, 0.01, 0.001, 0.5, 0.0005]``.
    ``log_rho``, ``log_s``, and ``log_q`` are fitted in log10 space. By default,
    ``d_xsi`` is used in place of ``alpha`` as the parameterization of the source
    trajectory angle, as it tends to be a more stable and less correlated fitting
    parameter. ``alpha`` is derived from ``d_xsi`` at each step via
    ``get_alpha_from_d_xsi()``.

    Parameters
    ----------
    emcee_settings : dict, optional
        Settings for the emcee sampler. Any missing keys are filled in from
        ``AnomalyFitter.default_emcee_settings``. Valid keys are
        ``'n_walkers'``, ``'n_burn'``, ``'n_steps'``, and optionally
        ``'temperature'``.

    perturbation_scale : float, optional
        Scale factor for perturbing the seeded grid values in
        _get_seeded_grid_values(). Perturbations are drawn from a Gaussian
        with standard deviation perturbation_scale * width, where width is
        the range of surviving results within 3 sigma of the minimum chi2.
        Defaults to 0.05.

    seed_strategy : str, optional
        Strategy for seeding subsequent estimators from grid search results.
        Options are:

            - ``'chain'``: each estimator seeds the next
              (1->2->3->...->15). Default.
            - ``'first'``: all remaining estimators are seeded from
              estimator 1, allowing parallelization.

        Defaults to ``'chain'``.

    Notes
    -----
    If ``parameters_to_fit`` is provided, ``sigmas`` must also be provided,
    otherwise an ``AttributeError`` is raised.

    All other parameters are inherited from ``AnomalyFitter`` and
    ``MulensFitter``.

    See Also
    --------
    AnomalyFitter : Parent class.
    get_alpha_from_d_xsi : Converts ``d_xsi`` to ``alpha``.

    WidePlanetGridSearchEstimator : Parameter estimator used to seed the
        emcee starting vector.
    _get_seeded_grid_values : Computes perturbed grid values from surviving
        results.
    _get_grid_params_from_estimator : Extracts seeded grid parameters from
        a completed estimator.
    """

    def __init__(self, emcee_settings=None, perturbation_scale=0.05,
                 seed_strategy='chain', **kwargs):
        super().__init__(**kwargs)
        if not ('parameters_to_fit' in kwargs.keys()):
            self.parameters_to_fit = ['t_0', 'u_0', 't_E', 'log_rho', 'log_s', 'log_q', 'd_xsi']
            self.sigmas = [0.1, 0.01, 0.01, 0.01, 0.001, 0.5, 0.0005]
        elif self.sigmas is None:
            raise AttributeError('If parameters_to_fit is set, sigmas must also be set.')

        if emcee_settings is None:
            emcee_settings = AnomalyFitter.default_emcee_settings

        for key in AnomalyFitter.default_emcee_settings.keys():
            if key not in list(emcee_settings.keys()):
                emcee_settings[key] = AnomalyFitter.default_emcee_settings[key]

        if 'n_dim' not in list(emcee_settings.keys()):
            emcee_settings['n_dim'] = len(self.parameters_to_fit)

        self.emcee_settings = emcee_settings
        self.perturbation_scale = perturbation_scale
        self.seed_strategy = seed_strategy

        self._best = None
        self._event = None
        self._estimators = None
        self._pooled_results = None
        self._starting_vector = None

    def _get_seeded_grid_values(self, surviving, param):
        """
        Get 3 new grid values for a parameter from surviving results.

        Takes the min, mid, and max of the surviving results for the given
        parameter and perturbs each by perturbation_scale * width.

        If all surviving values are identical, uses the spacing to the next
        unique value to set the width. If there is only one unique value,
        falls back to perturbation_scale * abs(value).

        Arguments:
            surviving: *pandas.DataFrame*
                Rows within n_sigma of the minimum chi2.

            param: *str*
                Column name in surviving to use.

        Returns:
            *list* of 3 floats
        """
        lo = surviving[param].min()
        hi = surviving[param].max()
        mid = (lo + hi) / 2.
        width = hi - lo

        if width == 0:
            sorted_vals = sorted(surviving[param].unique())
            if len(sorted_vals) > 1:
                width = np.abs(sorted_vals[1] - sorted_vals[0])
            else:
                width = np.abs(lo * self.perturbation_scale)

        return [
            lo + np.random.randn() * self.perturbation_scale * width,
            mid + np.random.randn() * self.perturbation_scale * width,
            hi + np.random.randn() * self.perturbation_scale * width
        ]

    def _get_grid_params_from_estimator(self, estimator):
        """
        Extract seeded grid parameters from a completed estimator's results.

        Uses get_results_within_n_sigma(3) to find surviving results, then
        calls _get_seeded_grid_values() for each of log_q, log_rho, alpha, s.

        log_q and log_rho are computed from the actual q and rho values in
        surviving results rather than the rounded subplot-mapping values stored
        in all_results.

        Arguments:
            estimator: *WidePlanetGridSearchEstimator*
                A completed estimator to seed from.

        Returns:
            *dict* with keys 'log_q_values', 'log_rho_values', 'alpha_grid',
            's_grid', each containing a list of 3 floats.
        """
        surviving = estimator.get_results_within_n_sigma(3).copy()
        surviving['log_q_actual'] = np.log10(surviving['q'])
        surviving['log_rho_actual'] = np.log10(surviving['rho'])

        return {
            'log_q_values': self._get_seeded_grid_values(surviving, 'log_q_actual'),
            'log_rho_values': self._get_seeded_grid_values(surviving, 'log_rho_actual'),
            'alpha_grid': self._get_seeded_grid_values(surviving, 'alpha'),
            's_grid': self._get_seeded_grid_values(surviving, 's')
        }

    def _build_estimators(self):
        """
        Build and run 15 WidePlanetGridSearchEstimators with perturbed PSPL
        parameters.

        Constructs a 5x3 grid of estimators:
            - 5 systematic values of t_E: t_E + [-2, -1, 0, 1, 2] * sigma_tE
            - 3 random perturbations of t_0 and u_0 per t_E value

        The first estimator uses the default broad grid. Subsequent estimators
        are seeded using _get_grid_params_from_estimator() according to
        seed_strategy:
            - 'chain': each estimator seeds the next (1->2->3->...->15)
            - 'first': all remaining estimators seeded from estimator 1
                       (parallelizable)

        Sigma values for t_0, u_0, t_E are taken from self.sigmas, indexed by
        self.parameters_to_fit.

        Results are stored in self._estimators.
        """
        t_0 = self.anomaly_lc_params['t_0']
        u_0 = self.anomaly_lc_params['u_0']
        t_E = self.anomaly_lc_params['t_E']

        param_index = {p: i for i, p in enumerate(self.parameters_to_fit)}
        sigma_t0 = self.sigmas[param_index['t_0']]
        sigma_u0 = self.sigmas[param_index['u_0']]
        sigma_tE = self.sigmas[param_index['t_E']]

        t_E_values = t_E + np.array([-2, -1, 0, 1, 2]) * sigma_tE

        estimators = []
        seed_estimator = None

        for t_E_val in t_E_values:
            for _ in range(3):
                params = self.anomaly_lc_params.copy()
                params['t_0'] = t_0 + np.random.randn() * sigma_t0
                params['u_0'] = u_0 + np.random.randn() * sigma_u0
                params['t_E'] = t_E_val

                if seed_estimator is None:
                    # First estimator: use default broad grid
                    estimator = mmexo.estimate_params.WidePlanetGridSearchEstimator(
                        self.datasets, params)
                else:
                    grid_params = self._get_grid_params_from_estimator(seed_estimator)
                    estimator = mmexo.estimate_params.WidePlanetGridSearchEstimator(
                        self.datasets, params,
                        log_q_values=grid_params['log_q_values'],
                        log_rho_values=grid_params['log_rho_values'],
                        alpha_grid=grid_params['alpha_grid'],
                        s_grid=grid_params['s_grid'])

                estimator.run()
                estimators.append(estimator)

                if self.seed_strategy == 'chain':
                    seed_estimator = estimator
                elif self.seed_strategy == 'first' and seed_estimator is None:
                    seed_estimator = estimator

        self._estimators = estimators

    def _pool_results(self):
        """
        Pool all_results from all estimators into a single DataFrame.

        Adds t_0, u_0, t_E, and estimator_index columns to each estimator's
        all_results before concatenating. Recomputes sigma relative to the
        global chi2 minimum across all estimators.

        Sets self.initial_model and self.mag_methods from the estimator
        containing the global best chi2.
        """
        if self._estimators is None:
            self._build_estimators()

        dfs = []
        for i, estimator in enumerate(self._estimators):
            df = estimator.all_results.copy()
            df['t_0'] = estimator.params['t_0']
            df['u_0'] = estimator.params['u_0']
            df['t_E'] = estimator.params['t_E']
            df['estimator_index'] = i
            dfs.append(df)

        pooled = pd.concat(dfs, ignore_index=True)

        # Recompute sigma relative to global minimum
        min_chi2 = pooled['chi2'].min()
        pooled['sigma'] = np.sqrt(pooled['chi2'] - min_chi2)
        self._pooled_results = pooled

        # Set initial_model and mag_methods from global best
        best_row = pooled.loc[pooled['chi2'].idxmin()]
        best_estimator = self._estimators[int(best_row['estimator_index'])]

        self.initial_model = {
            't_0': best_row['t_0'],
            'u_0': best_row['u_0'],
            't_E': best_row['t_E'],
            's': best_row['s'],
            'q': best_row['q'],
            'rho': best_row['rho'],
            'alpha': best_row['alpha']
        }
        self.mag_methods = best_estimator.binary_params.mag_methods

    def make_starting_vector(self):
        """
        Construct the starting vector for the emcee sampler from pooled
        grid search results.

        Pools all_results across all 15 estimators, sorts by chi2, and
        takes the top n_walkers rows as starting points. Each row is
        converted to an emcee parameter vector using
        make_emcee_vector_from_ModelParameters().

        Also sets self.initial_model and self.mag_methods as side effects
        via _pool_results().

        Returns
        -------
        list
            List of n_walkers parameter vectors, each of length n_dim.

        Raises
        ------
        ValueError
            If the number of pooled results is less than n_walkers.
        """
        if self._pooled_results is None:
            self._pool_results()

        n_walkers = self.emcee_settings['n_walkers']
        df = self._pooled_results.sort_values(by='chi2')

        if len(df) < n_walkers:
            raise ValueError(
                f'Not enough pooled results ({len(df)}) to fill '
                f'n_walkers ({n_walkers}). Consider increasing the grid '
                f'resolution or number of estimators.')

        top_rows = df.head(n_walkers)

        starting_vector = []
        for _, row in top_rows.iterrows():
            params = {
                't_0': row['t_0'],
                'u_0': row['u_0'],
                't_E': row['t_E'],
                's': row['s'],
                'q': row['q'],
                'rho': row['rho'],
                'alpha': row['alpha']
            }
            vector = self.make_emcee_vector_from_ModelParameters(
                MulensModel.ModelParameters(params))
            starting_vector.append(vector)

        return starting_vector

    def initialize_event(self):
        """
        Initialize the MulensModel.Event object for the wide planet model.

        Requires make_starting_vector() to have been called first, which
        sets self.initial_model and self.mag_methods as side effects.

        Raises
        ------
        AttributeError
            If self.initial_model or self.mag_methods is not set.
        """
        if self.initial_model is None:
            raise AttributeError(
                'initial_model is not set. Call make_starting_vector() first.')

        model = MulensModel.Model(parameters=self.initial_model)
        model.default_magnification_method = 'point_source_point_lens'

        if self.mag_methods is None:
            raise AttributeError(
                'self.mag_methods is not set. Call make_starting_vector() first.')
        else:
            model.set_magnification_methods(self.mag_methods)

        self._event = MulensModel.Event(datasets=self.datasets, model=model)

    def run(self, verbose=False):
        """
        Fit the wide planet model using emcee MCMC sampling.

        Builds and runs 15 grid search estimators, pools results to construct
        the starting vector, initializes the event, and runs the emcee sampler.

        Parameters
        ----------
        verbose : bool, optional
            If True, prints the fitted parameters with 16th, 50th, and 84th
            percentile uncertainties. Default is False.
        """
        starting_vector = self.starting_vector # sets initial_model and mag_methods
        self.initialize_event()

        if self.pool:
            ncpu = cpu_count()
            print("{0} CPUs".format(ncpu))
            os.environ["OMP_NUM_THREADS"] = "1"
            pool = Pool()
            sampler = emcee.EnsembleSampler(
                self.emcee_settings['n_walkers'], self.emcee_settings['n_dim'], self.ln_prob,
                pool=pool)
        else:
            sampler = emcee.EnsembleSampler(
                self.emcee_settings['n_walkers'], self.emcee_settings['n_dim'], self.ln_prob)

        sampler.run_mcmc(starting_vector, self.emcee_settings['n_steps'])

        samples = sampler.chain[:, self.emcee_settings['n_burn']:, :].reshape(
            (-1, self.emcee_settings['n_dim']))

        results = np.percentile(samples, [16, 50, 84], axis=0)
        if verbose:
            print("Fitted parameters:")
            for i in range(self.emcee_settings['n_dim']):
                r = results[1, i]
                print("${:.5f}^{{+{:.5f}}}_{{-{:.5f}}}$ &".format(
                    r, results[2, i] - r, r - results[0, i]))

        prob = sampler.lnprobability[:, self.emcee_settings['n_burn']:].reshape((-1))
        best_index = np.argmax(prob)
        self.event = samples[best_index, :]

        self.best = self.event.model.parameters.parameters
        self.best['chi2'] = self.event.get_chi2()

    def get_alpha_from_d_xsi(self, parameters, d_xsi):
        """
        Convert the ``d_xsi`` parameter to the source trajectory angle ``alpha``.

       ``d_xsi`` is the default parameterization of the source trajectory angle
        in ``WidePlanetFitter``, used in place of ``alpha`` because it tends to be
        a more stable and less correlated fitting parameter.This
        conversion is applied at every step of the emcee sampler via the
        ``event`` setter.

        ``d_xsi`` is the offset of the source trajectory from the center of the
        planetary caustic.

        Parameters
        ----------
        parameters : MulensModel.ModelParameters
            Current model parameters, used to retrieve ``s``, ``q``, and ``u_0``.
        d_xsi : float
            Offset of the source trajectory from the center of the planetary
            caustic.

        Returns
        -------
        float
            Source trajectory angle ``alpha`` in degrees.

        Notes
        -----
        The exact direction of the offset (vertical or perpendicular to the
        source trajectory) is to be confirmed.

        See Also
        --------
        make_emcee_vector_from_ModelParameters : Performs the inverse conversion
            from ``alpha`` to ``d_xsi``.
        event.setter : Calls this method at each emcee step.
        """
        s = parameters.s
        q = parameters.q
        u_0 = parameters.u_0
        xloc_prim = s - 1 / s
        a_prim = s * q / (1 + q)
        xloc = xloc_prim - a_prim
        sin_alpha = u_0 / (xloc + d_xsi)
        alpha = -np.rad2deg(np.arcsin(sin_alpha))

        return alpha

    def make_emcee_vector_from_ModelParameters(self, parameters):
        """
        Convert a MulensModel.ModelParameters object to an emcee parameter vector.

        Constructs a parameter vector corresponding to ``parameters_to_fit``,
        converting parameters with a ``log_`` prefix to log10 space, and
        converting ``alpha`` to ``d_xsi`` if present.

        Parameters
        ----------
        parameters : MulensModel.ModelParameters
            Model parameters to convert.

        Returns
        -------
        list
            Parameter vector corresponding to ``parameters_to_fit``, suitable
            for use as the initial guess for the emcee sampler.

        See Also
        --------
        get_alpha_from_d_xsi : Converts ``d_xsi`` to ``alpha``.
        event.setter : Performs the inverse conversion from emcee vector to
            model parameters.
        """
        initial_guess = []
        for parameter in self.parameters_to_fit:
            key = self.get_parameter_name(parameter)
            if key == 'd_xsi':
                u_0 = parameters.u_0
                alpha = parameters.alpha
                s = parameters.s
                q = parameters.q

                xloc_prim = s - 1 / s
                a_prim = s * q / (1 + q)
                xloc = xloc_prim - a_prim
                sin_alpha = np.sin(np.deg2rad(alpha))
                d_xsi = -u_0 / sin_alpha - xloc
                value = d_xsi
            else:
                value = parameters.__getattribute__(key)

            if key != parameter:
                value = np.log10(value)

            initial_guess.append(value)

        return initial_guess

    @property
    def event(self):
        """
        The MulensModel.Event object for the current fit.

        Returns
        -------
        MulensModel.Event or None
            The current event object, or None if ``initialize_event()`` has
            not yet been called.

        See Also
        --------
        initialize_event : Creates the event object.
        """
        return self._event

    @event.setter
    def event(self, theta):
        """
        Update the event model parameters from an emcee parameter vector.

        Converts logarithmic parameters from log10 space back to linear space,
        and converts ``d_xsi`` to ``alpha`` via ``get_alpha_from_d_xsi()`` if
        present in ``parameters_to_fit``.

        Parameters
        ----------
        theta : array-like
            Parameter vector corresponding to ``parameters_to_fit``.

        Raises
        ------
        AttributeError
            If the event has not been initialized via ``initialize_event()``.

        See Also
        --------
        get_alpha_from_d_xsi : Converts ``d_xsi`` to ``alpha``.
        make_emcee_vector_from_ModelParameters : Performs the inverse conversion.
        """
        if self._event is None:
            raise AttributeError('Event has not been created. Run initialize_event() first!')

        d_xsi = None
        for parameter, value in zip(self.parameters_to_fit, theta):
            key = self.get_parameter_name(parameter)
            if key != parameter:
                value = 10. ** value

            if key == 'd_xsi':
                d_xsi = value
            else:
                self._event.model.parameters.__setattr__(key, value)

        if d_xsi is not None:
            self._event.model.parameters.alpha = self.get_alpha_from_d_xsi(
                self._event.model.parameters, d_xsi)

    @property
    def starting_vector(self):
        if self._starting_vector is None:
            self._starting_vector = self.make_starting_vector()
        return self._starting_vector

