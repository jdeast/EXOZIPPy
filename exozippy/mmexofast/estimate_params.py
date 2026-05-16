#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Created by Luca Campiani in January 2024
# Updated by Jennifer Yee, May 2025
from itertools import product
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import warnings
#import copy

import MulensModel
import MulensModel as mm
import exozippy.mmexofast as mmexo

# In[ ]:


def get_PSPL_params(ef_grid_point, datasets, verbose=False):
    t_0 = ef_grid_point['t_0']
    u_0s = [0.01, 0.1, 0.3, 1.0, 1.5]
    t_Es = [1., 3., 10., 20., 40., 100.]
    best_chi2 = np.inf
    best_params = None
    for u_0 in u_0s:
        for t_E in t_Es:
            params = {'t_0': t_0, 't_E': t_E, 'u_0': u_0}
            event = MulensModel.Event(
                datasets=datasets, model=MulensModel.Model(params))
            if event.get_chi2() < best_chi2:
                best_params = params
                best_chi2 = event.chi2

            #print(u_0, t_E, event.chi2)

    return best_params


class BinaryLensParams():
    """
    A class for managing parameters related to binary lens models.

    Attributes:
        ulens: *object*
            Object representing the lens model.
        
        mag_method: *object*
            Object representing the magnification method.            
    """
    def __init__(self, ulens):
        self.ulens = ulens
        self.mag_methods = None
        
    def set_mag_method(self, params):
        """
        Sets the magnification calculation method based on input parameters.

        Arguments :
            params: *dictionary*
                Initilal parameters.
            
                - 't_0' (*float*): Time of maximum magnification.
                - 'u_0' (*float*): Impact parameter.
                - 't_E' (*float*): Einstein crossing time.
                - 't_pl' (*float*): Time at which to compute the wide model parameters.
                - 'dt' (*float*): Duration of the anomaly
                - 'dmag' (*float*): Magnitude difference of the perturbation

        Returns :
           None
        """
        #t1 = params['t_pl'] - (5 * params['dt'])
        #t2 = params['t_pl'] + (5 * params['dt'])
        #self.mag_method = [t1, 'VBBL', t2]
        #print(params)

        t_E = params['t_E']
        t_0 = params['t_0']
        t_pl = params['t_pl']
        t_star = params['dt'] / 2.
        self.mag_methods = [
            np.min((t_0 - t_E, t_pl - t_E / 2., t_pl - 20. * t_star)),
            'point_source',
            t_pl - 10. * t_star,
            'hexadecapole',
            t_pl - 5. * t_star,
            'VBBL',
            t_pl + 5. * t_star,
            'hexadecapole',
            t_pl + 10. * t_star,
            'point_source',
            np.max((t_0 + t_E, t_pl + t_E / 2., t_pl + 20. * t_star))]


def get_wide_params(params, limit='GG97'):
    """
    Transform initial parameters into wide model parameters.

    Arguments :
        params: *dictionary*
            Initial parameters.
            
            - 't_0' (*float*): Time of maximum magnification.
            - 'u_0' (*float*): Impact parameter.
            - 't_E' (*float*): Einstein crossing time.
            - 't_pl' (*float*): Time at which to compute the wide model parameters.
            - 'dt' (*float*): Duration of the anomaly
            - 'dmag' (*float*): Magnitude difference of the perturbation

        limit: *str*
            Method to use for estimating *rho* and *q*.

    Returns :
        wide_params : *BinaryLensParams*
             Wide model parameters for the binary lens.
    """
    estimator = WidePlanetParameterEstimator(params, limit=limit)
    
    return estimator.binary_params


def get_possible_bump_anomaly_solutions(params):
    solutions = {}

    # large rho limit
    estimator = WidePlanetParameterEstimator(params, limit='GG97')
    solutions['GG97'] = estimator.calc_binary_ulens_params()

    return solutions


class ParameterEstimator():

    def __init__(self, params, limit=None):
        self.params = params
        self.limit = limit

        self._tau_pl, self._u_pl = None, None
        self._s, self._alpha = None, None
        self._q = None
        self._rho = None
        self._binary_params = None

    def get_binary_lens_params(self):
        pass

    def get_rho(self):
        if self.limit == 'dwarf':
            return 0.001
        elif self.limit == 'giant':
            return 0.05
        elif self.limit == 'point':
            return None
        else:
            raise ValueError('Your limit for calculating rho is not implemented: ', self.limit)

    @property
    def binary_params(self):
        if self._binary_params is None:
            self._binary_params = self.get_binary_lens_params()

        return self._binary_params

    @property
    def t_0(self):
        return self.params['t_0']

    @property
    def u_0(self):
        return self.params['u_0']

    @property
    def t_E(self):
        return self.params['t_E']

    @property
    def tau_pl(self):
        if self._tau_pl is None:
            self._tau_pl = (self.params['t_pl'] - self.params['t_0']) / self.params['t_E']

        return self._tau_pl

    @property
    def u_pl(self):
        if self._u_pl is None:
            self._u_pl = np.sqrt(self.params['u_0'] ** 2 + self.tau_pl ** 2)

        return self._u_pl

    def _correct_alpha(self, alpha):
        while alpha > 360.:
            alpha -= 360.

        while alpha < -360:
            alpha += 360.

        return alpha

    @property
    def alpha(self):
        if self._alpha is None:
            alpha = np.pi - np.arctan2(self.params['u_0'], self.tau_pl)
            alpha = np.rad2deg(alpha)
            self._alpha = self._correct_alpha(alpha)

        return self._alpha

    @property
    def rho(self):
        if self._rho is None:
            self._rho = self.get_rho()

        return self._rho

    @rho.setter
    def rho(self, value):
        self._rho = value


class WidePlanetParameterEstimator(ParameterEstimator):

    def __init__(self, params, limit='GG97'):
        super().__init__(params, limit=limit)
        self._delta_A = None
        self._a_pspl = None

    def get_rho(self):
        if self.limit == 'GG97':
            rho = self.params['dt'] / self.params['t_E'] / 4.
        else:
            rho = super().get_rho()

        return rho

    def calc_binary_ulens_params(self):
        new_params = {'t_0': self.t_0, 'u_0':self.u_0, 't_E': self.t_E, 's': self.s, 'alpha': self.alpha}
        rho = self.rho
        if rho is not None:
            new_params['rho'] = rho

        new_params['q'] = self.q

        return new_params

    def get_binary_lens_params(self):
        binary_ulens_params = self.calc_binary_ulens_params()
        out = BinaryLensParams(binary_ulens_params)
        out.set_mag_method(self.params)
        return out

    @property
    def s(self):
        if self._s is None:
            u = self.u_pl
            self._s = 0.5 * (np.sqrt(u ** 2 + 4) + u)
        return self._s

    @property
    def q(self):
        if self._q is None:
            self._q = 0.5 * np.abs(self.delta_A) * (self.rho ** 2)

        return self._q

    @property
    def a_pspl(self):
        if self._a_pspl is None:
            self._a_pspl = (self.u_pl**2 + 2.) / np.sqrt(self.u_pl**2 * (self.u_pl**2 + 4.))

        return self._a_pspl

    @property
    def delta_A(self):
        """
        Might want to add an option to calculate delta_A using PSPL fitted fs and fb.
        Current calculation assumes fb=0. This could be a problem if fb is large, e.g. OB180383.
        :return:
        """
        if self._delta_A is None:
            self._delta_A = self.a_pspl * (10.**(self.params['dmag'] / -2.5) - 1.)

        return self._delta_A


class WidePlanetGridSearchEstimator(WidePlanetParameterEstimator):
    """
    Estimates wide planet binary lens parameters by performing a chi2 grid
    search centered on the analytic parameter estimates from
    WidePlanetParameterEstimator.

    The grid spans alpha, s, log_q, and log_rho. The best-fit parameters
    are identified by minimizing chi2 over the grid.

    Attributes:
        datasets: *list* of *MulensModel.MulensData*
            Photometric datasets to evaluate chi2 against.

        params: *dict*
            Anomaly parameters. See WidePlanetParameterEstimator for details.

        d_alpha: *float*, optional
            Step size for alpha grid. Defaults to 0.1.

        n_alpha: *int*, optional
            Number of alpha grid points. Defaults to 6.

        d_s: *float*, optional
            Step size for s grid. Defaults to 0.01 * s.

        n_s: *int*, optional
            Number of s grid points. Defaults to 4.

        log_q_values: *array-like*, optional
            Grid values for log10(q). Defaults to np.arange(-6, -1).

        log_rho_values: *array-like*, optional
            Grid values for log10(rho). Defaults to np.arange(-4, -1).

        alpha_grid: *array-like*, optional
            Explicit grid values for alpha. If provided, overrides d_alpha
            and n_alpha. Defaults to None.

        s_grid: *array-like*, optional
            Explicit grid values for s. If provided, overrides d_s and n_s.
            Defaults to None.

        refine: *bool*, optional
            If True, runs Nelder-Mead refinement after the grid search.
            Defaults to True.

        nelder_mead_options: *dict*, optional
            Options passed to scipy.optimize.minimize with method='Nelder-Mead'.
            Supported keys: 'maxfev' (default 500), 'xatol' (default 1e-3),
            'fatol' (default 0.1). Any key not specified falls back to the
            default. Note: 'initial_simplex' is computed internally from the
            grid step sizes and should not be passed here.

    Note: In future it might be a good idea to refactor best_params (and
    related methods) to use dynamic lists of grid parameters rather than
    hardcoding ['alpha', 's', 'q', 'rho'].
    """

    def __init__(self, datasets, params,
                 d_alpha=None, n_alpha=None,
                 d_s=None, n_s=None,
                 log_q_values=None, log_rho_values=None,
                 alpha_grid=None, s_grid=None,
                 refine=True,
                 nelder_mead_options=None):
        super().__init__(params)
        self.datasets = datasets
        self.d_alpha = d_alpha
        self.n_alpha = n_alpha
        self.d_s = d_s
        self.n_s = n_s
        self.log_q_values = log_q_values
        self.log_rho_values = log_rho_values
        self._alpha_grid = alpha_grid
        self._s_grid = s_grid
        self.refine = refine
        self.nelder_mead_options = nelder_mead_options
        self._results = None
        self._refinement_results = None
        self._refinement_result = None
        self._all_results = None
        self._is_run = False

    @property
    def _base_binary_params(self):
        """
        Internal access to binary_params without run check. Used by all
        internal methods to avoid triggering the RuntimeError guard on
        binary_params before run() has been called.
        """
        if self._binary_params is None:
            self._binary_params = self.get_binary_lens_params()
        return self._binary_params

    @property
    def _nelder_mead_options(self):
        defaults = {'maxfev': 500, 'xatol': 1e-3, 'fatol': 0.1}
        if self.nelder_mead_options is not None:
            defaults.update(self.nelder_mead_options)
        return defaults

    @property
    def binary_params(self):
        """
        Returns binary_params populated with best-fit parameters from the
        grid search and refinement. Call run() first.
        """
        if not self._is_run:
            raise RuntimeError(
                "binary_params is not available until run() has been called.")
        return self._binary_params

    @property
    def alternate_params(self):
        base_params = self.get_binary_lens_params()
        s_new = base_params.ulens['s']**2 / self.best_params['s']
        alt_params = BinaryLensParams(base_params.ulens)
        alt_params.mag_methods = base_params.mag_methods
        alt_params.ulens['s'] = s_new
        return alt_params

    @property
    def best_params(self):
        """
        Returns the best-fit parameter dictionary from the grid search and
        refinement. Call run() first.
        """
        if not self._is_run:
            raise RuntimeError(
                "best_params is not available until run() has been called.")
        return self._binary_params.ulens

    def run(self):
        """
        Runs the full pipeline: grid search and (if refine=True) iterative
        refinement. Populates binary_params and best_params.

        Returns:
            binary_params: *BinaryLensParams*
                Binary lens parameters populated with best-fit values.
        """
        _ = self.all_results  # triggers grid search + refinement
        self._is_run = True
        return self._binary_params

    @property
    def alpha_values(self):
        if self._alpha_grid is not None:
            return self._alpha_grid
        d_alpha = self.d_alpha if self.d_alpha is not None else 0.1
        n_alpha = self.n_alpha if self.n_alpha is not None else 6
        alpha_offset = np.arange(n_alpha) - (n_alpha - 1) / 2
        return self.alpha + alpha_offset * d_alpha

    @property
    def s_values(self):
        if self._s_grid is not None:
            return self._s_grid
        d_s = self.d_s if self.d_s is not None else 0.01 * self.s
        n_s = self.n_s if self.n_s is not None else 4
        s_offset = np.arange(n_s) - (n_s - 1) / 2
        return self.s + s_offset * d_s

    @property
    def log_q_grid(self):
        return self.log_q_values if self.log_q_values is not None else np.arange(-6, -1)

    @property
    def log_rho_grid(self):
        return self.log_rho_values if self.log_rho_values is not None else np.arange(-4, -1)

    def _make_event(self, grid_params):
        model = mm.Model(grid_params)
        model.set_magnification_methods(self._base_binary_params.mag_methods)
        model.default_magnification_method = 'point_source_point_lens'
        event = mm.Event(datasets=self.datasets, model=model)
        return event

    def _grid_iterator(self):
        return product(
            self.alpha_values, self.s_values,
            self.log_q_grid, self.log_rho_grid)

    def _run_grid_search(self):
        results = []
        grid_params = self._base_binary_params.ulens.copy()

        event = self._make_event(grid_params)

        for alpha, s, log_q, log_rho in self._grid_iterator():
            event.model.parameters.alpha = alpha
            event.model.parameters.s = s
            event.model.parameters.q = 10. ** log_q
            event.model.parameters.rho = 10. ** log_rho

            results.append({
                'chi2': event.get_chi2(),
                'alpha': alpha,
                's': s,
                'q': event.model.parameters.q,
                'rho': event.model.parameters.rho
            })

        df = pd.DataFrame(results)
        best_row = df.loc[df['chi2'].idxmin()]
        self._base_binary_params.ulens.update(best_row[['alpha', 's', 'q', 'rho']].to_dict())
        return df

    def _run_refinement(self):
        best = self._base_binary_params.ulens.copy()
        x0 = np.array([
            best['alpha'],
            best['s'],
            np.log10(best['q']),
            np.log10(best['rho'])
        ])

        # Build initial simplex scaled to the grid step sizes used in the
        # grid search. This is important: Nelder-Mead's default simplex
        # perturbs each coordinate by 5% of x0, which is arbitrary and can
        # be badly scaled here (e.g. log_q near 0 gets almost no perturbation).
        d_alpha = self.d_alpha if self.d_alpha is not None else 0.1
        d_s = self.d_s if self.d_s is not None else 0.01 * self.s
        simplex_deltas = np.array([d_alpha, d_s, 0.5, 0.5])
        n = len(x0)
        initial_simplex = np.vstack(
            [x0] + [x0 + simplex_deltas[i] * np.eye(n)[i] for i in range(n)])

        # Single Event created once; parameters updated in-place each call
        event = self._make_event(best)

        trajectory = []

        def chi2_fn(x):
            alpha, s, log_q, log_rho = x
            event.model.parameters.alpha = alpha
            event.model.parameters.s = s
            event.model.parameters.q = 10. ** log_q
            event.model.parameters.rho = 10. ** log_rho
            chi2 = event.get_chi2()
            trajectory.append({
                'chi2': chi2,
                'alpha': alpha,
                's': s,
                'q': 10. ** log_q,
                'rho': 10. ** log_rho,
            })
            return chi2

        result = minimize(
            chi2_fn, x0, method='Nelder-Mead',
            options={**self._nelder_mead_options, 'initial_simplex': initial_simplex})

        if not result.success:
            warnings.warn(
                f"Nelder-Mead refinement did not converge: {result.message}. "
                f"Best chi2={result.fun:.4f} after {result.nfev} evaluations.")

        self._refinement_result = result

        df = pd.DataFrame(trajectory)
        # Guard against Nelder-Mead wandering to a worse basin than the grid:
        # take the global best across both grid and refinement trajectory.
        # Use result.x directly — scipy guarantees this is the best point found
        best_grid_chi2 = self.results['chi2'].min()
        if result.fun < best_grid_chi2:
            alpha, s, log_q, log_rho = result.x
            self._base_binary_params.ulens.update({
                'alpha': alpha,
                's': s,
                'q': 10. ** log_q,
                'rho': 10. ** log_rho
            })
        # else: grid best is already set by _run_grid_search — leave it

        return df

    @property
    def results(self):
        if self._results is None:
            df = self._run_grid_search()
            self._results = self._postprocess_grid_results(df)
        return self._results

    @property
    def refinement_result(self):
        """Raw scipy OptimizeResult from Nelder-Mead. Check result.success and
        result.nfev for convergence diagnostics."""
        _ = self.refinement_results  # ensure refinement has run
        return self._refinement_result

    @property
    def refinement_results(self):
        """DataFrame of all points evaluated during Nelder-Mead refinement."""
        if self._refinement_results is None:
            _ = self.results  # ensure grid search has run first
            self._refinement_results = self._run_refinement()
        return self._refinement_results

    @property
    def all_results(self):
        if self._all_results is None:
            df_grid = self.results.copy()
            df_grid['source'] = 'grid'
            df_grid['iteration'] = 0

            if self.refine:
                df_refine = self.refinement_results.copy()
                df_refine['source'] = 'refinement'
                df_refine['log_q'] = np.round(np.log10(df_refine['q'])).astype(int)
                df_refine['log_rho'] = np.round(np.log10(df_refine['rho'])).astype(int)
                combined = pd.concat([df_grid, df_refine], ignore_index=True)
            else:
                combined = df_grid

            # Recompute sigma relative to global minimum
            min_chi2 = combined['chi2'].min()
            combined['sigma'] = np.sqrt(combined['chi2'] - min_chi2)
            self._all_results = combined

        return self._all_results

    def _postprocess_grid_results(self, df):
        df = df.copy()
        df['log_q'] = np.round(np.log10(df['q'])).astype(int)
        df['log_rho'] = np.round(np.log10(df['rho'])).astype(int)
        df['sigma'] = np.sqrt(df['chi2'] - df['chi2'].min())
        return df

    def get_results_within_n_sigma(self, n_sigma=3):
        """
        Return all results (grid and refinement) within n_sigma of the
        minimum chi2.

        Arguments:
            n_sigma: *float*, optional
                Maximum sigma threshold. Defaults to 3.

        Returns:
            *pandas.DataFrame*
                Subset of all_results with sigma <= n_sigma.
        """
        df = self.all_results
        return df[df['sigma'] <= n_sigma]

    @staticmethod
    def _get_sigma_marker(sigma):
        if sigma < 1:
            return '*', 200
        elif sigma < 2:
            return 'D', 100
        elif sigma < 3:
            return 'o', 60
        else:
            return '^', 30

    def plot_sigma_maps(self):
        df_all = self.all_results
        df_grid = df_all[df_all['source'] == 'grid']

        unique_log_q = sorted(df_grid['log_q'].unique())
        unique_log_rho = sorted(df_grid['log_rho'].unique())
        n_rho = len(unique_log_rho)

        if self.refine:
            df_refine = df_all[df_all['source'] == 'refinement']

        for log_q in unique_log_q:
            fig = plt.figure(figsize=(10, 4 * n_rho))
            gs = GridSpec(n_rho, 1, figure=fig, hspace=0.3)

            for idx, log_rho in enumerate(unique_log_rho):
                ax = fig.add_subplot(gs[idx, 0])

                # Grid imshow
                mask = (df_grid['log_q'] == log_q) & (df_grid['log_rho'] == log_rho)
                subset = df_grid[mask]
                grid = subset.pivot(index='s', columns='alpha', values='sigma')
                im = ax.imshow(grid, cmap='Set1', vmin=0, vmax=100, aspect='auto',
                               origin='lower',
                               extent=[subset['alpha'].min(), subset['alpha'].max(),
                                       subset['s'].min(), subset['s'].max()])

                # Refinement scatter overlay
                if self.refine:
                    refine_mask = (
                        (df_refine['log_q'] == log_q) &
                        (df_refine['log_rho'] == log_rho))
                    refine_subset = df_refine[refine_mask]

                    for sigma_low, sigma_high in [(0, 1), (1, 2), (2, 3), (3, np.inf)]:
                        pts = refine_subset[
                            (refine_subset['sigma'] >= sigma_low) &
                            (refine_subset['sigma'] < sigma_high)]
                        if not pts.empty:
                            marker, size = self._get_sigma_marker(sigma_low)
                            ax.scatter(pts['alpha'], pts['s'],
                                       marker=marker, s=size,
                                       c=pts['sigma'], cmap='Set1', vmin=0, vmax=100,
                                       edgecolors='black', linewidths=0.5, zorder=5)

                ax.set_xlabel('alpha', fontsize=10)
                ax.set_ylabel('s', fontsize=10)
                ax.set_title(f'log_q={log_q}, log_rho={log_rho}', fontsize=11)

                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label('sigma', fontsize=10)

            fig.suptitle(f'log_q = {log_q}', fontsize=13, fontweight='bold')
            plt.tight_layout()


class WidePlanetEnsembleInitializer():
    """
    Builds an ensemble of starting points for emcee by running multiple
    WidePlanetGridSearchEstimators with perturbed PSPL parameters.

    The first estimator uses a broad default grid. Its best log_q and
    log_rho are used to seed a narrower grid for all subsequent estimators.

    Attributes:
        datasets: *list* of *MulensModel.MulensData*
            Photometric datasets.

        anomaly_params: *dict*
            Anomaly light curve parameters.

        sigmas: *dict*
            Step sizes for PSPL parameter perturbations. Expected keys:
            't_0', 'u_0', 't_E'.

        n_estimators: *int*, optional
            Number of estimators to run. Should equal n_walkers.
            Defaults to 40.

        pspl_chi2: *float*, optional
            Chi2 of the no-planet PSPL model. Used only for diagnostics
            (delta_chi2, summary counts). Defaults to None.

    # TODO: Hypothesis that this is very slow because the event/Estimator class is getting  created anew every time.
    """

    def __init__(self, datasets, anomaly_params, sigmas,
                 n_estimators=40, pspl_chi2=None):
        self.datasets = datasets
        self.anomaly_params = anomaly_params
        self.sigmas = sigmas
        self.n_estimators = n_estimators
        self.pspl_chi2 = pspl_chi2

        self._results = None
        self._mag_methods = None
        self._initial_model = None
        self._seed_log_q = None
        self._seed_log_rho = None

    @property
    def sigma_t0(self):
        return self.sigmas.get('t_0', 0.)

    @property
    def sigma_u0(self):
        return self.sigmas.get('u_0', 0.)

    @property
    def sigma_tE(self):
        return self.sigmas.get('t_E', 0.)

    def _perturb_params(self):
        """
        Generate one set of perturbed PSPL parameters.

        Override to implement different perturbation strategies.

        Returns
        -------
        dict
            Perturbed anomaly_params.
        """
        params = self.anomaly_params.copy()
        params['t_0'] = self.anomaly_params['t_0'] + np.random.randn() * self.sigma_t0
        params['u_0'] = self.anomaly_params['u_0'] + np.random.randn() * self.sigma_u0
        params['t_E'] = self.anomaly_params['t_E'] + np.random.randn() * self.sigma_tE
        return params

    def _get_seeded_grid_values(self, best_log_val):
        """
        Generate a 3-point grid from the seed estimator's best log value.

        Perturbs best_log_val by 5% and returns
        [rand_best - 0.5, rand_best, rand_best + 0.5].

        Arguments:
            best_log_val: *float*
                Best log10 value from the seed estimator.

        Returns:
            *list* of 3 floats
        """
        rand_best = best_log_val + np.random.randn() * 0.05 * np.abs(best_log_val)
        return [rand_best - 0.5, rand_best, rand_best + 0.5]

    def _run_single_estimator(self, params, log_q_values=None, log_rho_values=None):
        """
        Run a single WidePlanetGridSearchEstimator for the given params.

        Override to use different estimator settings.

        Arguments:
            params: *dict*
                Anomaly parameters for this estimator.

            log_q_values: *list*, optional
                If provided, passed as the log_q grid. If None, the
                estimator uses its default broad grid.

            log_rho_values: *list*, optional
                If provided, passed as the log_rho grid. If None, the
                estimator uses its default broad grid.

        Returns:
            best: *dict*
                Best-fit binary lens parameters.

            mag_methods: *list*
                Magnification methods from this estimator.
        """
        estimator = WidePlanetGridSearchEstimator(
            datasets=self.datasets, params=params, refine=True,
            log_q_values=log_q_values, log_rho_values=log_rho_values)
        estimator.run()
        return estimator.binary_params.ulens.copy(), estimator.binary_params.mag_methods

    def _evaluate_chi2(self, best, mag_methods):
        """
        Compute chi2 for a set of binary lens parameters.
        """
        model = mm.Model(best)
        model.set_magnification_methods(mag_methods)
        model.default_magnification_method = 'point_source_point_lens'
        event = mm.Event(datasets=self.datasets, model=model)
        return event.get_chi2()

    def _run_all_estimators(self):
        """
        Run all n_estimators and collect results into a DataFrame.

        The first estimator uses the default broad grid. Its best log_q
        and log_rho seed all subsequent estimators via
        _get_seeded_grid_values().
        """
        rows = []

        for i in range(self.n_estimators):
            params = self._perturb_params()

            if self._seed_log_q is None:
                best, mag_methods = self._run_single_estimator(params)
                self._seed_log_q = np.log10(best['q'])
                self._seed_log_rho = np.log10(best['rho'])
            else:
                log_q_values = self._get_seeded_grid_values(self._seed_log_q)
                log_rho_values = self._get_seeded_grid_values(self._seed_log_rho)
                best, mag_methods = self._run_single_estimator(
                    params,
                    log_q_values=log_q_values,
                    log_rho_values=log_rho_values)

            if self._mag_methods is None:
                self._mag_methods = mag_methods

            chi2 = self._evaluate_chi2(best, mag_methods)

            row = {
                'chi2': chi2,
                't_0': best['t_0'],
                'u_0': best['u_0'],
                't_E': best['t_E'],
                's': best['s'],
                'q': best['q'],
                'rho': best['rho'],
                'alpha': best['alpha']
            }
            if self.pspl_chi2 is not None:
                row['delta_chi2'] = self.pspl_chi2 - chi2

            rows.append(row)

            log_str = (f'Estimator {i:3d}: chi2={chi2:.2f}  '
                       f't_E={best["t_E"]:.3f}  '
                       f'log_q={np.log10(best["q"]):.2f}  '
                       f'log_rho={np.log10(best["rho"]):.2f}  '
                       f'{"[seed]" if i == 0 else "[seeded]"}')
            if self.pspl_chi2 is not None:
                log_str += f'  delta_chi2={self.pspl_chi2 - chi2:.2f}'
            print(log_str)

        return pd.DataFrame(rows)

    @property
    def results(self):
        if self._results is None:
            self._results = self._run_all_estimators()
        return self._results

    @property
    def mag_methods(self):
        _ = self.results  # ensure estimators have run
        return self._mag_methods

    @property
    def initial_model(self):
        """
        Best-fit binary lens parameters across all estimators (lowest chi2).
        """
        if self._initial_model is None:
            df = self.results
            best_row = df.loc[df['chi2'].idxmin()]
            self._initial_model = {
                k: best_row[k]
                for k in ['t_0', 'u_0', 't_E', 's', 'q', 'rho', 'alpha']}
        return self._initial_model

    def summary(self):
        """
        Print a summary of all estimator results sorted by chi2.
        """
        df = self.results
        if 'delta_chi2' in df.columns:
            n_better = np.sum(df['delta_chi2'] > 0)
            print(f'\n{n_better} / {self.n_estimators} estimators better than PSPL')

        cols = ['chi2', 't_E', 'u_0', 's', 'q', 'rho', 'alpha']
        if 'delta_chi2' in df.columns:
            cols.insert(1, 'delta_chi2')
        print('\nSorted by chi2:')
        print(df.sort_values('chi2')[cols].to_string())

    def plot_chi2_distribution(self):
        """
        Plot histogram of chi2 values across all estimators.
        """
        df = self.results
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(df['chi2'], bins=20)
        if self.pspl_chi2 is not None:
            ax.axvline(self.pspl_chi2, color='red', linestyle='--',
                       label=f'PSPL chi2={self.pspl_chi2:.0f}')
            ax.legend()
        ax.set_xlabel('chi2')
        ax.set_ylabel('N')
        ax.set_title(f'{self.n_estimators} estimators: chi2 distribution')

    def plot_models(self):
        """
        Plot all estimator models in the anomaly region and VBBL zoom,
        colour-coded by chi2 (red=worst, green=best).
        """
        df = self.results
        t_range_anomaly = [self.mag_methods[2], self.mag_methods[8]]
        t_range_vbbl = [self.mag_methods[4], self.mag_methods[6]]

        ref_model = mm.Model(self.initial_model)
        ref_model.set_magnification_methods(self.mag_methods)
        ref_model.default_magnification_method = 'point_source_point_lens'
        ref_event = mm.Event(datasets=self.datasets, model=ref_model)
        source_flux, blend_flux = ref_event.get_ref_fluxes()

        sorted_idx = df['chi2'].argsort().values[::-1]  # worst first
        cmap = plt.cm.get_cmap('RdYlGn', self.n_estimators)

        for fig_title, t_range in [('Anomaly region', t_range_anomaly),
                                    ('VBBL zoom', t_range_vbbl)]:
            fig, ax = plt.subplots(figsize=(10, 5))
            plt.sca(ax)
            ref_event.plot_data()

            for rank, idx in enumerate(sorted_idx):
                row = df.iloc[idx]
                params = {k: row[k] for k in
                          ['t_0', 'u_0', 't_E', 's', 'q', 'rho', 'alpha']}
                model = mm.Model(params)
                model.set_magnification_methods(self.mag_methods)
                model.default_magnification_method = 'point_source_point_lens'
                model.plot_lc(source_flux=source_flux, blend_flux=blend_flux,
                              color=cmap(rank), alpha=0.6, t_range=t_range)

            ref_event.plot_model(label='Best (grid)', color='black',
                                 zorder=10, t_range=t_range, linewidth=2)
            ax.set_xlim(t_range)
            ax.set_xlabel('Time (HJD)')
            ax.set_ylabel('Magnitude')
            ax.set_title(f'{fig_title} — red=worst, green=best chi2')
            ax.minorticks_on()


def get_close_params(params, q=None, rho=None):
    """
    Transform initial parameters into two close model parameters for a binary lens. One for upper and one for lower caustics. 

    Arguments:
        params: *dictionary*
            Initial parameters.

            - 't_0' (*float*): Time of maximum magnification.
            - 'u_0' (*float*): Impact parameter.
            - 't_E' (*float*): Einstein crossing time.
            - 't_pl' (*float*): Time at which to compute the close model parameters.
            - 'dt' (*float*), optional: Duration of the anomaly
            - 'q' (*float*): trial value of q for calculating the caustic,
                default is 0.004
            - 'rho' (*float*): value of rho for the model. If 'dt' is specified,
                'rho' is calculated from 'dt'. If neither are specified,
                default is 0.001.

    Returns:
        lens1, lens2 : *tuple of BinaryLensParams*
            Two instances of BinaryLensParams representing close model parameters.
    """
    estimator_upper = CloseUpperPlanetParameterEstimator(params=params, q=q)
    estimator_lower = CloseLowerPlanetParameterEstimator(params=params, q=q)

    return estimator_upper.binary_params, estimator_lower.binary_params


class CloseUpperPlanetParameterEstimator(WidePlanetParameterEstimator):

    def __init__(self, params, limit='GG97', q=None):
        super().__init__(params, limit=limit)
        if q is None:
            q = 0.004

        self._q = q
        self._eta_not, self._mu, self._phi = None, None, None
        #self._alpha_upper, self._alpha_lower = None, None

    def setup_close_ulens_params(self):
        new_params = {'t_0': self.t_0,
                      'u_0': self.u_0,
                      't_E': self.t_E,
                      's': self.s,
                      'q': self.q}

        if self.rho is not None:
            new_params['rho'] = self.rho

        return new_params

    def calc_binary_params(self):
        new_params = self.setup_close_ulens_params()
        new_params['alpha'] = self.alpha

        return new_params

    def get_binary_lens_params(self):
        binary_ulens_params = self.calc_binary_ulens_params()
        binary_params = BinaryLensParams(binary_ulens_params)
        binary_params.set_mag_method(self.params)

        return binary_params

    @property
    def binary_params(self):
        if self._binary_params is None:
            self._binary_params = self.get_binary_lens_params()

        return self._binary_params

    @property
    def s(self):
        if self._s is None:
            u = self.u_pl
            self._s = 0.5 * (np.sqrt(u**2 + 4) - u)

        return self._s

    @property
    def q(self):
        return self._q

    @property
    def eta_not(self):
        if self._eta_not is None:
            self._eta_not = (self.q**0.5 / self.s) * (1 / (np.sqrt(1 + self.s**2)) + np.sqrt(1 - self.s**2))

        return self._eta_not

    @property
    def mu(self):
        if self._mu is None:
            self._mu = np.arctan2(self.eta_not, (self.s - 1 / self.s) / (1 + self.q))
            # correction for primary --> COM

        return self._mu

    @property
    def phi(self):
        if self._phi is None:
            self._phi = np.arctan2(self.u_0, self.tau_pl)

        return self._phi

    @property
    def alpha(self):
        if self._alpha is None:
            alpha = 180. - np.rad2deg(self.phi - self.mu)
            self._alpha = self._correct_alpha(alpha)

        return self._alpha


class CloseLowerPlanetParameterEstimator(CloseUpperPlanetParameterEstimator):

    @property
    def alpha(self):
        if self._alpha is None:
            alpha = 180. - np.rad2deg(self.phi + self.mu)
            self._alpha = self._correct_alpha(alpha)

        return self._alpha


def model_pspl_mag_at_pl(params):
    """
    Gets the magnification at second lense time assuming point lense model.

    Arguments :
        params: *dictionary*
            Initilal parameters.
            
            - 't_0' (*float*): The time of maximum magnification.
            - 'u_0' (*float*): The impact parameter.
            - 't_E' (*float*): The Einstein crossing time.
            - 't_pl' (*float*): The time at which to compute the magnification.
            
    Returns :
        mag :*float*
             Magnification at the specified time 't_pl' based on the point lens model.
            
    """
    model1 = mm.Model({'t_0': params['t_0'], 
                       'u_0': params['u_0'], 
                       't_E': params['t_E']})
    return model1.get_magnification(params['t_pl'])


class BinarySourceParams():
    """
    A class for managing parameters related to binary source models. Derived from equation 2.5 from Gaudi 1998.
   
   Attributes:
        ulens: *object*
            Object representing the underlying lens model.
            
        source_flux_ratio: *object*
            Object representing the source flux ratio.  
            
        set_source_flux_ratio(params):
            Sets the source flux ratio based on input parameters.
  
    """
    def __init__(self, ulens):
        self.ulens = ulens
        self.source_flux_ratio = None
        
    def set_source_flux_ratio(self, params):
        """
         Sets the source flux ratio based on input parameters.

        Arguments :
            params: *dictionary*
                Initilal parameters.
            
                - 't_0' (*float*): Time of maximum magnification.
                - 'u_0' (*float*): Impact parameter.
                - 't_E' (*float*): Einstein crossing time.
                - 't_pl' (*float*): Time at which to compute the wide model parameters.
                - 'dt' (*float*): Duration of the anomaly
                - 'dmag' (*float*): Magnitude difference of the perturbation

        Returns :
           None
        """
        A1 = model_pspl_mag_at_pl(params)
        u_0_2 = params["dt"] / (12**0.5 * params["t_E"])
        e = params["dmag"] * u_0_2 * A1
        self.source_flux_ratio = e


def get_binary_source_params(params):
    """
    Transform initial parameters into binary source model parameters.

    Arguments:
        params: *dictionary*
            Initial parameters.

            - 't_0' (*float*): Time of maximum magnification for the first lens.
            - 'u_0' (*float*): Impact parameter for the first lens.
            - 't_pl' (*float*): Time at which to compute the source flux ratio.
            - 'dt' (*float*): Duration of the anomaly
            - 't_E' (*float*): Einstein crossing time.
            - 'dmag' (*float*): Magnitude difference of the perturbation

    Returns:
        source_params : *BinarySourceParams*
            Binary source model parameters.
    """
    u_0_2 = params["dt"] / (12**0.5 * params["t_E"])
    new_params= {'t_0_1': params['t_0'],
              'u_0_1': params['u_0'],
              't_0_2': params['t_pl'],
              'u_0_2': u_0_2,
              't_E': params['t_E']}
    out = BinarySourceParams(new_params)
    out.set_source_flux_ratio(params)
    return out


class AnomalyPropertyEstimator():
    # The old version revised the PSPL parameters after masking the anomaly.
    # Could consider whether it would be a good idea to reimplement that.

    def __init__(self, datasets=None, pspl_params=None, af_results=None, mask_type='t_eff', n_mask=3):
        if isinstance(datasets, MulensModel.MulensData):
            datasets = [datasets]

        self.datasets = datasets
        self.pspl_params = pspl_params
        self.af_results = af_results
        self.n_mask = n_mask

        self.anom_t_range_af = self.af_results['t_0'] + self.n_mask * np.array(
            [-1, 1]) * self.af_results['t_eff']

        self._peak_index = None
        self._peak_dflux = None
        self._t_start = None
        self._t_stop = None

        self._pspl_event = None
        self._source_flux = None
        self._blend_flux = None

        self._anom_type = None
        self._anom_index = None
        self._sorted_index = None
        self._times = None
        self._scaled_fluxes = None
        self._scaled_residuals = None
        self._chi2s = None
        self._expected_model_fluxes = None

    def get_pspl_event(self):
        event = mm.Event(datasets=self.datasets,
                         model=mm.Model(self.pspl_params))
        event.fit_fluxes()
        return event

    def get_anom_type(self):
        n_pts = np.sum(self.anom_index)
        sigmas = np.sign(self.residuals) * np.sqrt(self.chi2s)
        #med, std = np.nanmedian(sigmas), np.nanstd(sigmas)
        #print('sigma dist', med, std, np.percentile(sigmas, q=[0, 1, 2, 98, 99, 100]))
        #plt.figure()
        #plt.hist(sigmas, bins=int(n_pts/40))
        #plt.axvline(med, color='black')
        #plt.axvline(med - std, color='black')
        #plt.axvline(med + std, color='black')
        #plt.gca().minorticks_on()
        #plt.xlabel('sigmas')

        if n_pts > 10:
            max_res = np.percentile(sigmas, q=98)
            min_res = np.percentile(sigmas, q=2)
        else:
            min_res, max_res = -np.inf, np.inf

        #print('res', n_pts, min_res, max_res)
        if (min_res < 0) and (np.abs(min_res) > max_res):
            return 'negative'

        top_index = (sigmas > 0) & (sigmas < max_res)
        bot_index = (sigmas < 0) & (sigmas > min_res)
        #print('n', np.sum(bot_index), np.sum(top_index))
        if np.sum(top_index) == 0:
            return 'negative'
        elif np.sum(bot_index) == 0:
            return 'positive'
        else:
            top_chi2 = np.sum(self.chi2s[top_index])
            bot_chi2 = np.sum(self.chi2s[bot_index])
            #print('chi2', bot_chi2, top_chi2)
            if top_chi2 > bot_chi2:
                return 'positive'
            else:
                return 'negative'

    def set_anom_prop(self):
        self._peak_dflux, self._peak_index, self._t_start, self._t_stop = self.find_extremum(
            method='rolling')

    def get_anom_prop(self):
        if (self.peak_dflux is None) or (self.t_start is None) or (self.t_stop is None):
            self.set_anom_prop()

        return self.peak_dflux, self.peak_index, self.t_start, self.t_stop, self.peak_width

    def _find_extremum_with_simple_line(self):
        peak_index = np.nanargmax(self.chi2s)
        peak_dflux = self.residuals[peak_index]
        t_start, t_stop = None, None
        for i in [1, -1]:
            slope = (self.sorted_times[peak_index] - self.sorted_times[i]) / (self.peak_dflux - self.residuals[i])
            intercept = self.sorted_times[peak_index] - slope * peak_dflux
            t = slope * peak_dflux / 2. + intercept
            if i == 1:
                t_start = t
            else:
                t_stop = t

        return peak_dflux, peak_index, t_start, t_stop

    def _get_window_size(self):
        n_pts = np.sum(self.anom_index)

        if n_pts < 10:
            window_size = 1
        elif n_pts < 50:
            window_size = int(np.floor(n_pts / 10))
        elif n_pts < 100:
            window_size = int(np.floor(n_pts / 20))
        elif n_pts < 500:
            window_size = int(np.floor(n_pts / 50))
        else:
            window_size = int(np.floor(n_pts / 100))

        #window_size = int(np.floor(n_pts / 10))
        #print('points', n_pts, 'window', window_size)
        return window_size

    def _find_extremum_with_rolling_mean(self):
        window_size = self._get_window_size()
        kernel = np.ones(window_size) / window_size
        #print('points:', np.sum(t_index), 'window:', window_size,
        #      'half window:', int(window_size / 2))

        if (window_size > 0) and (window_size < np.sum(self.anom_index)):
            chi2_rolling_mean = np.convolve(self.chi2s, kernel, mode='same')
            peak_index = np.argmax(chi2_rolling_mean)

            res_rolling_mean = np.convolve(self.residuals, kernel, mode='same')
            #print('rolling mean:', len(res_rolling_mean))

            peak_dflux = res_rolling_mean[peak_index]
            # start_dflux = res_rolling_mean[half_window]
            # end_dflux = res_rolling_mean[-half_window]

            if peak_dflux > 0:
                half_anomaly = res_rolling_mean > (peak_dflux / 2.)
            else:
                half_anomaly = res_rolling_mean < (peak_dflux / 2.)
                # raise NotImplementedError('negative perturbations not implemented')

            t_start = np.min(self.sorted_times[half_anomaly])
            t_stop = np.max(self.sorted_times[half_anomaly])

            return peak_dflux, peak_index, t_start, t_stop
        else:
            return self._find_extremum_with_simple_line()

    def find_extremum(self, method=None):
        if method == 'rolling':
            return self._find_extremum_with_rolling_mean()

    def get_anomaly_lc_parameters(self):
        self.set_anom_prop()
        params = {key: value for key, value in self.pspl_params.items()}
        params['dmag'] = self.dmag
        params['dt'] = self.t_stop - self.t_start
        params['t_pl'] = np.mean((self.t_start, self.t_stop))

        return params

    def _plot_peak_lines(self):
        plt.axvline(self.peak_time, color='darkgray', zorder=10, linestyle=':')
        plt.axvline(self.t_start, color='darkgray')
        plt.axvline(self.t_stop, color='darkgray')
        #plt.axvline(self.peak_time - self.peak_width / 2., color='darkgray')
        #plt.axvline(self.peak_time + self.peak_width / 2., color='darkgray')

    def _plot_af_lines(self):
        plt.axvline(self.af_results['t_0'] +
                    self.af_results['t_eff'], color='black')
        plt.axvline(self.af_results['t_0'] -
                    self.af_results['t_eff'], color='black')

    def _setup_anom_xaxis(self):
        plt.xlim(self.af_results['t_0'] + 5. * np.array([-1, 1]) *
                 self.af_results['t_eff'])
        plt.xlabel('time')

    def plot_residuals(self):
        plt.figure()
        plt.title(self.anom_type)
        plt.axhline(0, color='black')
        plt.scatter(self.sorted_times, self.residuals)
        self._plot_peak_lines()
        self._plot_af_lines()
        self._setup_anom_xaxis()
        plt.ylabel('res')

    def plot_anomaly(self):
        plt.figure()
        plt.title(self.anom_type)
        self.pspl_event.plot_data()
        self.pspl_event.plot_model(color='black', zorder=5)
        peak_anom_mag = mm.Utils.get_mag_from_flux(self.expected_model_fluxes[self.peak_index] + self.peak_dflux)
        plt.scatter(self.peak_time, peak_anom_mag, marker='d', color='darkgray', zorder=10)

        self._plot_peak_lines()
        self._plot_af_lines()
        self._setup_anom_xaxis()

        plt.ylabel('mag')

    @property
    def anom_type(self):
        if self._anom_type is None:
            self._anom_type = self.get_anom_type()

        return self._anom_type

    @property
    def peak_dflux(self):
        return self._peak_dflux

    @property
    def peak_index(self):
        return self._peak_index

    @property
    def peak_time(self):
        return self.sorted_times[self.peak_index]

    @property
    def t_start(self):
        return self._t_start

    @property
    def t_stop(self):
        return self._t_stop

    @property
    def dmag(self):
        expected_mag =  mm.Utils.get_mag_from_flux(
            self.expected_model_fluxes[self.peak_index])
        peak_anom_mag = mm.Utils.get_mag_from_flux(
            self.expected_model_fluxes[self.peak_index] + self.peak_dflux)

        return peak_anom_mag - expected_mag

    @property
    def peak_width(self):
        return self.t_stop - self.t_start

    @property
    def anom_index(self):
        if self._anom_index is None:
            self._anom_index = (self.times > self.anom_t_range_af[0]) & (self.times < self.anom_t_range_af[1])

        return self._anom_index

    @property
    def sorted_index(self):
        if self._sorted_index is None:
            self._sorted_index = np.argsort(self.times[self.anom_index])

        return self._sorted_index

    @property
    def times(self):
        if self._times is None:
            self._times = np.hstack([dataset.time for dataset in self.pspl_event.datasets])

        return self._times

    @property
    def sorted_times(self):
        return self.times[self.anom_index][self.sorted_index]

    @property
    def pspl_event(self):
        if self._pspl_event is None:
            self._pspl_event = self.get_pspl_event()

        return self._pspl_event

    @property
    def source_flux(self):
        if self._source_flux is None:
            self._source_flux, foo = self.pspl_event.get_ref_fluxes()

        return self._source_flux

    @property
    def blend_flux(self):
        if self._blend_flux is None:
            foo, self._blend_flux = self.pspl_event.get_ref_fluxes()

        return self._blend_flux

    @property
    def scaled_fluxes(self):
        if self._scaled_fluxes is None:
            self._scaled_fluxes = np.hstack(
                [np.array(flux) for (flux, err) in self.pspl_event.get_scaled_fluxes()])[self.anom_index][self.sorted_index]

        return self._scaled_fluxes

    @property
    def residuals(self):
        if self._scaled_residuals is None:
            self._scaled_residuals = self.scaled_fluxes - self.expected_model_fluxes

        return self._scaled_residuals

    @property
    def chi2s(self):
        if self._chi2s is None:
            self._chi2s = np.hstack(self.pspl_event.get_chi2_per_point())[self.anom_index][self.sorted_index]

        return self._chi2s

    @property
    def expected_model_fluxes(self):
        if self._expected_model_fluxes is None:
            self._expected_model_fluxes = self.source_flux * self.pspl_event.model.get_magnification(
                self.sorted_times) + self.blend_flux

        return self._expected_model_fluxes
