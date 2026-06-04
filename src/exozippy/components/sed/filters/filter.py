# general imports
import requests
import os
import pathlib
from typing import Tuple

# scientific imports
import pandas as pd
import numpy as np
from scipy import interpolate
from astropy import units as u
from astropy.units import Quantity

# pickling and querying
import pickle
from astroquery.query import BaseQuery
from astropy.io.votable import parse


def construct_wave_grid(R: int | float, wave_min: float, wave_max: float, 
                        input_unit: Quantity["length"] = u.micron, 
                        output_unit: Quantity["length"] = u.Angstrom,
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Constructs grid of wavelengths for a specified wavelength range at a specific spectral resolution

    Parameters
    ----------
    R : int | float
        spectral resolution of wavelength grid
    wave_min : float
        minimum wavelength
    wave_max : float
        maximum wavelength
    input_unit : astropy.units.Quantity["length"], optional
        wavelength unit that matches the inputs ``wave_min`` and ``wave_max``. By default, set to ``u.micron``
    output_unit : astropy.units.Quantity["length"], optional
        wavelength unit that matches the second output array ``wave_array_output``. By default, set to ``u.Angstrom``

    Returns
    -------
        wave_array         : np.ndarray, shape 
            wavelength array in units of ``input_unit``
        wave_array_output  : np.ndarray, shape
            wavelength array in units of ``output_unit``
        dwave_array        : np.ndarray, shape
            wavelength spacing (dlambda) array in units of ``input_unit``
    """

    init_dlambda = wave_min / R         # initial delta lambda
    wave_range = wave_max - wave_min    # full wavelength range
    max_array_size = int(np.ceil(wave_range / init_dlambda))    # max array length
    
    # initialize arrays
    wave_array = np.zeros(max_array_size)
    dwave_array = np.zeros(max_array_size)

    i = 0
    wave_array[i] = wave_min
    dwave_array[i] = wave_min / R
    next_wave = wave_min

    while next_wave < wave_max:
        
        # iterate forward wavelength
        next_wave += dwave_array[i]

        wave_array[i+1] = next_wave
        dwave_array[i+1] = next_wave / R

        # iterate forward index
        i+=1

    # unit conversion
    CONVERT = input_unit.to(output_unit)
    wave_array_output = CONVERT * wave_array

    return np.trim_zeros(wave_array, 'b'), np.trim_zeros(wave_array_output, 'b'), np.trim_zeros(dwave_array, 'b')


class Filter(BaseQuery):

    SVO_BASE_URL = 'https://svo2.cab.inta-csic.es/theory/fps/'
    DEFAULT_FILTER_DIR = pathlib.Path(__file__).parent

    VOTABLE_FIELD_NAMES = ['FilterProfileService', 'Facility', 'Instrument', 'ProfileReference', 'CalibrationReference',
                           'PhotSystem', 'MagSys', 'WavelengthEff', 'WavelengthRef', 'WavelengthMin', 'WavelengthMax',
                           'FWHM', 'WidthEff', 'Fsun']
    
    # define common wavelength grid for all spectra
    # everything will use the same grid of wavelengths
    RESOLUTION = 20000     # Wavelength resolution  
    _LAMBDA_MIN = 0.03     # microns
    _LAMBDA_MAX = 30       # microns
    _LAMBDA_GRID_COMMON_MICRONS, WAVELENGTH_PTS, _DLAMBDA_GRID_COMMON_MICRONS = construct_wave_grid(RESOLUTION, _LAMBDA_MIN, _LAMBDA_MAX)
    

    def __init__(self, filterID, **kwargs):

        """
        Parameters
        ----------
        filterID : string
            Used to create a HTTP query string i.e. send to SVO FPS to get data.
            String will take form of 'FACILTIY/INSTRUMENT.FILTER
            All filter options are available at: https://svo2.cab.inta-csic.es/theory/fps/index.php?mode=browse
            Examples: '2MASS/2MASS.Ks' or 'Keck/NIRC2.Brgamma'
        """

        self.filterID = filterID
        self.facility = filterID.split('/')[0]
        self.filterName = filterID.split('/')[-1]
        self.filterDirectory = None
        self._session = requests.Session()
        self._check_if_filter_saved(**kwargs)


    def __getstate__(self):
        state = self.__dict__.copy()
        return(state)
    

    def __setstate__(self, state):
        for field in state:
            setattr(self, field, state[field])


    def __str__(self):
        return(self.filterName + ' filter data available in ' + self.filterDirectory)


    def _check_if_filter_saved(self, filterDir=DEFAULT_FILTER_DIR, overwrite=False):

        self.filterDirectory = filterDir / self.facility
        filename_filter = self.filterName + '.filter'
       
        os.makedirs(self.filterDirectory, exist_ok=True)

        # check if we already have created the filter file
        # if it doesn't exist, let's create it
        if (not os.path.exists(self.filterDirectory / filename_filter)) or (overwrite):
            self._download_filter()
            self._set_attrs()
            self._create_filter_file()
            
            return
        
        else:
            self._read_filter_file()

            return


    def _download_filter(self):

        """Get and save all filter data in response a query sent to SVO FPS.
    
        Parameters
        ----------
        filterDir : String
            Directory where filter VOTables will be saved/checked for

        Returns
        -------
        Dictionary of filter properties, including full filter transmission profile
        """

        filename_VOTable = self.filterName + '.xml'

        # set URL
        url = self.SVO_BASE_URL + 'fps.php?ID=' + self.filterID
        #response = self._request('GET', url, save=True, savedir=filterDir)
        response = self._download_file(url, local_filepath=self.filterDirectory / filename_VOTable, verbose=False)

        return
        

    def _set_zeropoint_values(self):

        ### access zeropoint information ###
        CALIBRATION_TABLE_DROP_ROWS = [0,3,4,5]
        CALIBRATION_TABLE_DROP_COLUMNS = [0]
        CALIBRATION_TABLE_INDEX = {1: 'ZeroPoint_Fl', 2: 'ZeroPoint_Fv'}
        CALIBRATION_TABLE_HEADER = {1: 'Specified', 2: 'Calculated', 3: 'Unit'}
        MISSING_VALUE = '--'

        url = self.SVO_BASE_URL + 'index.php?id=' + self.filterID

        # read tables from html url provided
        extracted_tables = pd.read_html(url)
        df_calibration_vega = extracted_tables[-3]
        df_calibration_ab = extracted_tables[-2] 
        df_calibration_st = extracted_tables[-1] 

        # make the tables nice and easily parsable
        df_vega = df_calibration_vega.drop(CALIBRATION_TABLE_DROP_ROWS).drop(columns=CALIBRATION_TABLE_DROP_COLUMNS).rename(index=CALIBRATION_TABLE_INDEX).rename(columns=CALIBRATION_TABLE_HEADER)
        df_ab = df_calibration_ab.drop(CALIBRATION_TABLE_DROP_ROWS).drop(columns=CALIBRATION_TABLE_DROP_COLUMNS).rename(index=CALIBRATION_TABLE_INDEX).rename(columns=CALIBRATION_TABLE_HEADER)
        df_st = df_calibration_st.drop(CALIBRATION_TABLE_DROP_ROWS).drop(columns=CALIBRATION_TABLE_DROP_COLUMNS).rename(index=CALIBRATION_TABLE_INDEX).rename(columns=CALIBRATION_TABLE_HEADER)

        # initialize dictionary and lists
        df_dict = {'Vega': {'df':df_vega, 'suffix':'_Vega'},
                     'AB': {'df':df_ab,   'suffix':'_AB'},
                     'ST': {'df':df_st,   'suffix':'_ST'}}
        rows = ['ZeroPoint_Fl', 'ZeroPoint_Fv']
        columns = ['Specified', 'Calculated']
        attr_names = ['Zp_Spec_Fl', 'Zp_Calc_Fl', 'Zp_Spec_Fv', 'Zp_Calc_Fv']

        # loops through lists and set zeropoint attributes
        for df_sys in df_dict:
            count = 0
            for row in rows:
                for column in columns:
                    field = attr_names[count]+df_dict[df_sys]['suffix']
                    if MISSING_VALUE not in df_dict[df_sys]['df'].loc[row][column]:
                        value = float(df_dict[df_sys]['df'].loc[row][column])
                        setattr(self, field, value)
                    else:
                        setattr(self, field, None)
                    count += 1
        
        return


    def _process_raw_filter(self, raw):

        # pad the filter profile to force it to go zero on either side
        # first calculate lower and upper wavelength spacing for the padding
        wave_filter_profile = raw[0]
        wave_dlambda = np.diff(wave_filter_profile)
        wave_lower_dlambda = wave_dlambda[0]
        wave_upper_dlambda = wave_dlambda[-1]

        N_pad = 10
        wave_pad_lower = np.linspace(min(wave_filter_profile)-N_pad*wave_lower_dlambda, min(wave_filter_profile), N_pad, endpoint=False)
        wave_pad_upper = np.linspace(max(wave_filter_profile)+wave_upper_dlambda, max(wave_filter_profile)+(N_pad+1)*wave_upper_dlambda, N_pad)
        wave_filter_profile_padded = np.concatenate((wave_pad_lower, wave_filter_profile, wave_pad_upper), axis=None)

        # pad filter transmission curve with zeros
        trans_filter_profile = raw[1]
        trans_filter_profile_padded = np.pad(trans_filter_profile, (N_pad, N_pad), 'constant')

        filter_func = interpolate.interp1d(wave_filter_profile_padded, trans_filter_profile_padded, fill_value='extrapolate') # in angstroms
        filter_interpolated = filter_func(self.WAVELENGTH_PTS)
        filter_interpolated[filter_interpolated < 0] = 0

        return(np.array([wave_filter_profile_padded, trans_filter_profile_padded]), 
               np.array([self.WAVELENGTH_PTS, filter_interpolated]))

        
    def _set_attrs(self):

        filename_VOTable = self.filterName + '.xml'
        votable = parse(self.filterDirectory / filename_VOTable)

        for field in self.VOTABLE_FIELD_NAMES:
            try:
                value = votable.get_field_by_id_or_name(field).value
                setattr(self, field, value)
            except:
                setattr(self, field, None)

        self._set_zeropoint_values()

        self.TransmissionUnit = votable.get_field_by_id_or_name('Transmission').unit
        self.RawFilterCurve = np.array([list(i) for i in votable.get_first_table().array.data]).T
        self.PaddedFilterCurve, self.ProcessedFilterCurve = self._process_raw_filter(self.RawFilterCurve)

        return


    def _create_filter_file(self):

        filename_filter = self.filterName + '.filter'
        state = self.__getstate__()
        with open(self.filterDirectory / filename_filter, 'wb') as file:
            pickle.dump(state, file, protocol=pickle.HIGHEST_PROTOCOL)

        return
    
    
    def _read_filter_file(self):

        filename_filter = self.filterName + '.filter'
        with open(self.filterDirectory / filename_filter, 'rb') as file:
            state = pickle.load(file)

        self.__setstate__(state)

        return
