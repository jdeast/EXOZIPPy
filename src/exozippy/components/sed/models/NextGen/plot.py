import warnings

import numpy as np
from scipy.interpolate import interpn

from ...plot import Plot


class NextGenPlot(Plot):

    ALPHA_GRID_PTS = np.array([0, 0.2, -0.2, 0.4, 0.6])
    axis_alias = {
            'teff': 'star.teffsed',
            'feh': 'star.feh',
        }

    def __init__(self, system, draws):

        super().__init__(system, draws)
        self._calc_compiled_func()
        self._interp_spectra()

        # calculate model flux at earth
        self._normalize_optical_depth()
        self._calc_model_flux()

        # calculate observed flux from observed mags
        self._calc_obs_flux_from_obs_mag()


    def _calc_compiled_func(self):
        """
        Calculates the two compiled functions for use in plotting:
        Logg and predicted magnitudes

        Created Class Attributes
        -------
            self.logg_vals_draws :  np.ndarray, shape (ndraws, nstars)
            self.mag_pred_draws  :  np.ndarray, shape (ndraws, nstars, nfilters)
        """
        # grab compiled functions
        mag_compiled = getattr(self.system.sed, "_compiled_mag_predictors", [None])
        logg_compiled = getattr(self.system.sed, "_compiled_logg_calc", [None])

        # collect parameters used as inputs to compiled functions
        logg_vals_draws = np.zeros((self.ndraws, self.nstars))
        mag_pred_draws = np.zeros((self.ndraws, self.nstars, self.nfilters))
        for d, draw in enumerate(self.draws):
            logg_params = []
            mag_params = []
            for p in self.system.plot_params:
                val = np.asarray(
                draw.get(p.label, p.initval), dtype=np.float64
                    )
                mag_params = np.append(mag_params, np.atleast_1d(val), axis=0)
                if p.label in ['star.logmass', 'star.radiussed']:
                    logg_params = np.append(logg_params, np.atleast_1d(val), axis=0)
            # reshape arrays into shapes compatible with function inputs
            logg_params = np.reshape(logg_params, (self.nstars, 2, 1)) 
            mag_params = np.reshape(mag_params, (len(self.system.plot_params), self.nstars, 1)) 
            
            # calculate outputs for compiled functions
            logg_vals = np.array([])
            mag_vals_pred = np.array([])
            for nstar in range(self.nstars):
                if self.nstars == 1:
                    l_params = logg_params[nstar]
                else:
                    l_params = logg_params[:, nstar]
                m_params = mag_params[:, nstar]
                logg_vals = np.append(logg_vals, np.atleast_1d(logg_compiled(*l_params)))
                mag_vals_pred = np.append(mag_vals_pred, np.atleast_1d(mag_compiled(*m_params)))

            mag_vals_pred = np.reshape(mag_vals_pred, (self.nstars, self.system.sed.nfilters))

            logg_vals_draws[d, :] = logg_vals
            mag_pred_draws[d, :] = mag_vals_pred

        self.logg_vals_draws = logg_vals_draws
        self.mag_pred_draws = mag_pred_draws


    def _interp_spectra(self):
        """
        Linearly interpolates model spectra in n-dimensions 
        for star(s) parameters reported in draw

        Created Class Attribute
        -------
            self.model_spectrum_flux_draws  :  np.ndarray, shape (ndraws, nstars, len(self.df_wave))
                                         model flux is unextincted and represents flux at stellar surface
        """        
        model_spectrum_flux_draws = np.zeros((self.ndraws, self.nstars, len(self.df_wave)))

        for d, draw in enumerate(self.draws):
            pt_dict = {}
            for col in self.df_spec.columns:
                if col in self.grid_axes:
                    if col == 'logg':
                        pt_dict[col] = self.logg_vals_draws[d]
                    else:
                        pt_dict[col] = draw[self.axis_alias[col]]
            
            nearPts = self._findNearestGridPoints(pt_dict, self.df_spec)
            
            teff_near, logg_near, feh_near = [np.array(pts) for pts in nearPts]  # unpack the three parameter axes
            flux_near = np.zeros(shape=(self.nstars, len(teff_near), len(logg_near), len(feh_near), len(self.df_wave)))

            for nstar in range(self.nstars):
                for i_t, teff_pt in enumerate(teff_near[:, nstar]):
                    for i_l, logg_pt in enumerate(logg_near[:, nstar]):
                        for i_f, feh_pt in enumerate(feh_near[:, nstar]):
                            matched = False
                            for alpha in self.ALPHA_GRID_PTS:
                                flux_vals = self.df_spec.loc[
                                    (self.df_spec['teff'] == teff_pt) &
                                    (self.df_spec['logg'] == logg_pt) &
                                    (self.df_spec['feh'] == feh_pt) &
                                    (self.df_spec['alpha'] == alpha)
                                ]['flux'].values
                                if len(flux_vals) > 0:
                                    flux_near[nstar, i_t, i_l, i_f, :] = flux_vals[0]
                                    matched = True
                                    break

                            if not matched:
                                warnings.warn(
                                    f"No spectrum found for star {nstar} at "
                                    f"teff={teff_pt}, logg={logg_pt}, feh={feh_pt}"
                                )

            interp_flux = np.zeros((self.nstars, len(self.df_wave)))
            for nstar in range(self.nstars):
                points = (teff_near[:, nstar], logg_near[:, nstar], feh_near[:, nstar])
                eval_point = np.array([pt_dict[ax][nstar] for ax in pt_dict])
                interp_flux[nstar, :] = interpn(points, flux_near[nstar], eval_point)

            model_spectrum_flux_draws[d, :] = interp_flux

        self.model_spectrum_flux_draws = model_spectrum_flux_draws