# general imports
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Literal
import json
import yaml

# scientific imports
import numpy as np
import pandas as pd
from scipy import interpolate

# astronomy imports
import astropy.units as u
import astropy.constants as const

# internal imports
#from exozippy.constants import *
from ...constants import *
from .filters import filter as VOID
from .bc_grid import _load_alias_table, resolve_filter_name
#import exozippy.components.sed.filters.filter as VOID
#from exozippy.components.sed.bc_grid import _load_alias_table, resolve_filter_name

# plotting imports
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple


try:
    current_dir = Path(__file__).parent
except NameError:
    current_dir = Path.cwd()


class Plot:

    colors_spec = ["#0a9396", "#ca6702", "#ee9b00"]
    colors_obs = ["#005f73", "#bb3e03", "#ae7409"]
    linetypes = ["solid", "dashed", "dashdot", "dotted"]
    markers = ["o", "D", "v"] # circle, diamond, triangle

    # read in extinction values
    extinction_dir = current_dir / "models" / "extinction_law.ascii"
    extinction_df = pd.read_csv(extinction_dir, names=['wavelength', 'extinction'], 
                                delimiter=' ', index_col=False, skipinitialspace=True)
    
    # read in filter magnitude systems
    filtersys_dir = current_dir / "filters" / "filter_magsys.txt"
    filtersys_df = pd.read_csv(filtersys_dir, sep='\t', comment='#', skipinitialspace=True)

    filter_alias_df = _load_alias_table()


    def _findNearestGridPoints(self, pt_dict: Dict[str, np.ndarray], 
                               df_spec: pd.DataFrame) -> np.ndarray:
        """
        Finds nearest grid points as defined in df_spec to the point values in pt_dict

        Inputs
        -------
            pt_dict   :  dict, keys: [AXES_THE_MODEL_WILL_BE_INTERPOLATED_ON]
                                pt_dict[key] = value, where value is that specific model point
                                Example: {'teff': np.array([5128.1, 9032]),
                                        'logg': np.array([4.546, 3.67]),
                                        'feh': np.array([0.316, -0.13])}
            df_spec   :  pd.DataFrame, columns: ['filename', AXES_THE_MODEL_DEFINED_ON, 'flux']
                                Example: ['filename', 'teff', 'logg', 'feh', 'alpha', 'flux']
                                flux will be in erg/s/cm^2/Angstrom

        Output
        -------
            nearestGridPts  :  np.ndarray, shape (len(pt_dict), 2, len(pt_dict[KEY]))
                                Example:
                                [[array([5100, 9000]), array([5200, 9200])],
                                [array([4.5, 3.5]), array([5., 4.])],
                                [array([ 0.3, -0.5]), array([0.5, 0. ])]]                  
        """
        nearestGridPts = []
        for axis in pt_dict:
            pt = pt_dict[axis]
            # remove any singelton dimensions/then make it at least 1d
            pt = pt.squeeze()
            pt = np.atleast_1d(pt)
            grid_pts = np.sort(df_spec[axis].unique())

            # check for edges of grid
            index = np.searchsorted(grid_pts, pt)
            index = np.clip(index, 1, len(grid_pts)-1)
            lower = grid_pts[index - 1]
            upper = grid_pts[index]

            nearestGridPts.append([lower, upper])
        
        return nearestGridPts
    

    def __init__(self, system, draws):

        # general information
        self.system = system
        self.draws = draws
        self.ndraws = len(draws)
        self.nstars = self.system.sed.nstars
        self.star_names = self.system.star.names
        self.filters = self.system.sed.filters
        self.nfilters = len(self.filters)
        # one observation per filter row (blended or differential mags;
        # star membership per row lives in the blend matrix)
        self.mag_obs = self.system.sed.mag                 # (nfilters,)
        self.mag_obs_err = self.system.sed.err             # (nfilters,)
        self.blend_matrix = self.system.sed.blend_matrix   # (nfilters, nstars)
        self.combo_labels = self.system.sed.combo_labels

        # rows sharing a star combination share a color/marker
        self.unique_combos = list(dict.fromkeys(self.combo_labels))
        self.row_combo_idx = np.array(
            [self.unique_combos.index(l) for l in self.combo_labels], dtype=int
        )

        # load data
        self._load_spectra_data()
        self._load_model_grid_yaml()
        self._load_filter_data()

        ### calculations on spectra performed in models/MODEL/plot.py ###


    def _load_spectra_data(self):
        """
        Loads spectra (R=150) and wavelength grid for specified model; 
        Defaults to NextGen for plotting

        Created Class Attributes
        -------
            self.sedmodel  :  str
            self.df_spec   :  pd.DataFrame, columns: ['filename', AXES_THE_MODEL_DEFINED_ON, 'flux']
                              Example: ['filename', 'teff', 'logg', 'feh', 'alpha', 'flux']
                              flux will be in erg/s/cm^2/Angstrom
            self.df_wave   :  pd.DataFrame, columns: ['wavelength_micron', 'wavelength_angstrom', 'dlambda_micron']               
        """

        self.sedmodel = self.system.sed.sedmodel

        try:
            df_spec = pd.read_csv(current_dir / "models" / f"{self.sedmodel}" / f"{self.sedmodel}.spectra.csv")
            df_wave = pd.read_csv(current_dir / "models" / f"{self.sedmodel}" / f"{self.sedmodel}.wavelength.csv")
        except:
            print(f"No spectra found for ``{self.sedmodel}`` model.\n"
                "Defaulting to using ``NextGen`` spectra for plotting.")
            df_spec = pd.read_csv(current_dir / "models" / "NextGen" / "NextGen.spectra.csv")
            df_wave = pd.read_csv(current_dir / "models" / "NextGen" / "NextGen.wavelength.csv")

        df_spec['flux'] = df_spec['flux'].apply(json.loads).apply(np.array)

        self.df_spec = df_spec
        self.df_wave = df_wave


    def _load_model_grid_yaml(self):
        """
        Load model grid axes/points from yaml file

        Created Class Attributes
        -------
            self.model_grid_yaml    :  dict, keys: ['model', 'grid': [AXES_THE_MODEL_WILL_BE_INTERPOLATED_ON]]
                                        'grid' axes will have the associated grid points for each axis

            self.grid_axes          :  Set(List)
                                        'grid' axes that the model will be interpolated on
        """
        with open(current_dir / "models" / f"{self.sedmodel}" / f"{self.sedmodel}.grid.yaml", 'r') as f:
            self.model_grid_yaml = yaml.safe_load(f)

        # when dealing with the raw model spectra 
        # we don't need to deal with the extinction Av
        self.grid_axes = set(list(self.model_grid_yaml.get("grid").keys())) - set(["av"])
    

    def _load_filter_data(self):
        """
        calculate the optical depth at each wavelength normalized to V-band extinction

        Created Class Attribute
        -------
            self.filter_params   :  dict, keys: [FILTER_NAME: ['zp', 'wave_eff', 'wave_min', 'wave_max']]
                                    'zp': Vega filter zero point;
                                          uses specified value if given, calculated otherwise;
                                          given in F_lambda units: [erg/s/cm^2/Angstrom]
                                    'wave_eff': effective wavelength of filter [Angstrom]
                                    'wave_min': minimum wavelength of filter [Angstrom]
                                    'wave_max': maximum wavelength of filter [Angstrom]
                                    len(dict) = nfilters

        """
        filter_dict = {}
        for filter in self.filters:
            SVO_name = resolve_filter_name(filter, self.filter_alias_df, "SVO")
            filter_dict[filter] = VOID.Filter(SVO_name)

        filter_params = {}

        for filter in filter_dict:
            filter_params[filter] = {}
            obj = filter_dict[filter]
            if obj.Zp_Spec_Fl_Vega:
                filter_params[filter]['zp'] = obj.Zp_Spec_Fl_Vega
            else:
                filter_params[filter]['zp'] = obj.Zp_Calc_Fl_Vega

            filter_params[filter]['wave_eff'] = obj.WavelengthEff # angstroms
            filter_params[filter]['wave_min'] = obj.WavelengthMin # angstroms
            filter_params[filter]['wave_max'] = obj.WavelengthMax # angstroms
        
        self.filter_params = filter_params
    

    ######## model specific functions defined in models/MODEL/plot.py ########

    def _calc_compiled_func(self):
        """
        Calculates the compiled functions for use in plotting
        """
        pass

    def _interp_spectra(self):
        """
        Linearly interpolates model spectra in n-dimensions 
        for star(s) parameters reported in draw

        Created Class Attribute
        -------
            self.model_spectrum_flux_draws  :  np.ndarray, shape (ndraws, nstars, len(self.df_wave))
                                         model flux is unextincted and represents flux at stellar surface
        """
        pass

    ######## functions used in calculating flux ########

    def _normalize_optical_depth(self):
        """
        calculate the optical depth at each wavelength normalized to V-band extinction

        Created Class Attribute
        -------
            self.optical_depth_draws   :  np.ndarray, shape (ndraws, nstars, len(self.df_wave))
        """
        # interpolate extinction function onto same wavelength scale, but in microns
        V_BAND_MICRON = 0.55
        extinction_func = interpolate.interp1d(self.extinction_df['wavelength'], self.extinction_df['extinction'], fill_value='extrapolate')
        V_band_extinction = extinction_func(V_BAND_MICRON)
        extinction_modeled = extinction_func(self.df_wave['wavelength_micron'])
        extinction_modeled = np.broadcast_to(extinction_modeled[np.newaxis], (self.nstars, *extinction_modeled.shape))

        optical_depth_draws = np.zeros((self.ndraws, self.nstars, len(self.df_wave)))
        for d, draw in enumerate(self.draws):
            av = draw.get("star.av")
            optical_depth = (extinction_modeled/V_band_extinction) * av[:, np.newaxis]/1.086 
            optical_depth_draws[d, :] = optical_depth

        self.optical_depth_draws = optical_depth_draws


    def _calc_model_flux(self):
        """
        calculate the model flux as measured at earth using inverse square law, stellar radius, distance, optical depth

        Created Class Attribute
        -------
            self.flux_model_draws   :  np.ndarray, shape (ndraws, nstars, len(self.df_wave))

        """
        flux_model_draws = np.zeros((self.ndraws, self.nstars, len(self.df_wave)))
        for d, draw in enumerate(self.draws):
            radiussed = draw.get("star.radiussed")
            distance = draw.get("star.distance")
            f_model = self.model_spectrum_flux_draws[d, :] * np.exp(-self.optical_depth_draws[d, :]) * (radiussed[:, np.newaxis] / (PC_TO_RSUN_CONST * distance[:, np.newaxis]))**2
            flux_model_draws[d, :] = f_model

        self.flux_model_draws = flux_model_draws


    def _calc_obs_flux_from_obs_mag(self):
        """
        calculate observed flux from observed magnitudes; all wavelengths in angstroms

        Blended rows convert directly through the filter's Vega zeropoint.
        Differential rows (mag = -2.5*log10(F_pos/F_neg)) have no absolute
        flux of their own, so they are anchored to the model: the plotted
        flux is F_neg(model, median draw) * 10**(-0.4*mag_obs), i.e. the
        positive stars' flux implied by the observed magnitude difference.

        Created Class Attributes
        -------
            self.flux_obs           :  np.ndarray, shape (nfilters,)
            self.f_limits_from_err  :  np.ndarray, shape (2, nfilters)
            self.wave_filter        :  np.ndarray, shape (nfilters,)
            self.wave_err           :  np.ndarray, shape (2, nfilters)

        """
        f_obs = np.zeros(self.nfilters)
        # yerr == error from mags == error in flux
        # shape (2, nfilters) --> 2 is from lower/upper bounds for error bars
        f_limits_from_err = np.zeros((2, self.nfilters))

        wave_filter = np.zeros(self.nfilters)
        # xerr == error from filter bandwidth == error in wavelength
        # shape (2, nfilters) --> 2 is from lower/upper bounds for error bars
        wave_err = np.zeros((2, self.nfilters))

        # per-star predicted mags at the median draw, for anchoring
        # differential rows (available: model plot classes call
        # _calc_compiled_func before this method)
        mag_pred_med = np.median(self.mag_pred_draws, axis=0)  # (nstars, nfilters)

        # iterate over filter ROWS (self.filters may repeat a filter, e.g.
        # J of A+B and J of A-B; filter_params is keyed by unique name)
        for i, filter in enumerate(self.filters):

            # flux calculations
            zp = self.filter_params[filter]['zp']

            neg_stars = np.where(self.blend_matrix[i] < 0)[0]
            if neg_stars.size > 0:
                # differential row: anchor on the model's negative-side flux
                F_neg_model = np.sum(zp * 10 ** (-0.4 * mag_pred_med[neg_stars, i]))
                scale = F_neg_model
            else:
                scale = zp

            f =  10 ** (-0.4 * self.mag_obs[i])
            f_lower_from_err = 10 ** (-0.4 * (self.mag_obs[i] + self.mag_obs_err[i]))
            f_upper_from_err = 10 ** (-0.4 * (self.mag_obs[i] - self.mag_obs_err[i]))

            f_obs[i] = scale*f
            f_limits_from_err[0, i] = scale*f_lower_from_err
            f_limits_from_err[1, i] = scale*f_upper_from_err

            # wavelength calculations
            wave_eff = self.filter_params[filter]['wave_eff']
            wave_err_lower = wave_eff - self.filter_params[filter]['wave_min']
            wave_err_upper = self.filter_params[filter]['wave_max'] - wave_eff

            wave_filter[i] = wave_eff
            wave_err[0, i] = wave_err_lower
            wave_err[1, i] = wave_err_upper

        self.flux_obs = f_obs
        self.f_limits_from_err = f_limits_from_err
        self.wave_filter = wave_filter
        self.wave_err = wave_err


    ######## functions used for plotting ########

    def _get_ylim(self):
        """
        calculate plotting y-axis limits using maximum brightness measured as reference point

        Created Class Attribute
        -------
            y_lower  :  float
            y_upper  :  float
        """
        y = np.log10(self.flux_obs * self.wave_filter)

        self.y_lower = round(np.nanmin(y))-2.5
        self.y_upper = round(np.nanmax(y))+0.5