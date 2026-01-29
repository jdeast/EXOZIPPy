import os.path
import exozippy


# ============================================================================
# Public API
# ============================================================================

def get_kwargs(filename):
    """
    Parse the filename to create a dict of kwargs for MulensModel.MulensData.

    Parameters
    ----------
    filename : str
        Format: nYYYYMMDD.BAND.TELESCOPE.whateveryouwant

    Returns
    -------
    dict
        Kwargs for MulensData constructor
    """
    telescope, band = get_telescope_band_from_filename(filename)

    if telescope in OBSERVATORIES:
        obs = OBSERVATORIES[telescope]
        kwargs = obs.get_kwargs()
        kwargs['bandpass'] = band
        kwargs['plot_properties'] = obs.get_plot_properties(band)
    else:
        # Default for unknown telescopes
        kwargs = {
            'phot_fmt': 'flux',
            'bandpass': band,
            'plot_properties': {'label': f'{telescope}-{band}', 'marker': 'o'}
        }

    return kwargs


def get_telescope_band_from_filename(filename):
    """
    Get the telescope name and band from the filename.

    Parameters
    ----------
    filename : str
        Format: nYYYYMMDD.BAND.TELESCOPE.whateveryouwant

    Returns
    -------
    tuple
        (telescope, band)
    """
    basename = os.path.basename(filename).split('.')
    if len(basename) < 3:
        raise ValueError(
            f"Filename ({filename}) must have the format " +
            "nYYYYMMDD.BAND.TELESCOPE.whateveryouwant")

    band = basename[1]
    telescope = basename[2]
    return telescope, band

# ============================================================================
# Observatory Class and Registry
# ============================================================================

class Observatory:
    """
    Container for observatory-specific MulensData configuration.

    Parameters
    ----------
    name : str
        Observatory name
    phot_fmt : str, optional
        Photometry format ('flux' or 'mag')
    usecols : list, optional
        Columns to read from data file
    ephemerides_file : str or None, optional
        Path to ephemerides file for space-based observatories
    bands : dict, optional
        Dict mapping band names to plot properties dicts
    """

    def __init__(self, name, phot_fmt='flux', usecols=None,
                 ephemerides_file=None, bands=None):
        self.name = name
        self.phot_fmt = phot_fmt
        self.usecols = usecols if usecols is not None else [0, 1, 2]
        self.ephemerides_file = ephemerides_file
        self.bands = bands if bands is not None else {}

    def get_kwargs(self):
        """Get kwargs dict for MulensData creation."""
        kwargs = {'phot_fmt': self.phot_fmt}
        if self.usecols is not None:
            kwargs['usecols'] = self.usecols
        if self.ephemerides_file is not None:
            kwargs['ephemerides_file'] = self.ephemerides_file
        return kwargs

    def get_plot_properties(self, band):
        """Get plot properties for a specific band."""
        default = {
            'label': f'{self.name}-{band}',
            'marker': 'o'
        }
        if band in self.bands:
            return {**default, **self.bands[band]}
        return default


# Observatory registry (public)
OBSERVATORIES = {}


def register_observatory(observatory):
    """Register an observatory instance."""
    OBSERVATORIES[observatory.name] = observatory


# Create reverse mapping from ephemerides_file to observatory name
EPHEMERIDES_TO_OBSERVATORY = {
    obs.ephemerides_file: name
    for name, obs in OBSERVATORIES.items()
    if obs.ephemerides_file is not None
}

# ============================================================================
# Utility Functions
# ============================================================================

def list_observatories():
    """
    List all registered observatories.

    Returns
    -------
    list
        List of observatory names
    """
    return list(OBSERVATORIES.keys())


def get_observatory(name):
    """
    Get an observatory by name.

    Parameters
    ----------
    name : str
        Observatory name

    Returns
    -------
    Observatory or None
        Observatory instance if registered, None otherwise
    """
    return OBSERVATORIES.get(name)


def validate_filename(filename):
    """
    Check if filename follows the expected format.

    Parameters
    ----------
    filename : str
        Filename to validate

    Returns
    -------
    bool
        True if valid, False otherwise
    """
    try:
        get_telescope_band_from_filename(filename)
        return True
    except ValueError:
        return False


def load_observatories_from_config(config_file):
    """
    Load and register observatories from a configuration file.

    Parameters
    ----------
    config_file : str
        Path to YAML or JSON config file

    Notes
    -----
    Expected format (YAML):
        observatories:
          - name: MyTelescope
            phot_fmt: mag
            bands:
              V:
                color: blue
                marker: s

    Expected format (JSON):
        {
          "observatories": [
            {
              "name": "MyTelescope",
              "phot_fmt": "mag",
              "bands": {
                "V": {"color": "blue", "marker": "s"}
              }
            }
          ]
        }
    """
    import json

    # Try JSON first
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
    except json.JSONDecodeError:
        # Try YAML
        try:
            import yaml
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
        except ImportError:
            raise ImportError("YAML support requires PyYAML: pip install pyyaml")

    # Register observatories from config
    for obs_config in config.get('observatories', []):
        obs = Observatory(
            name=obs_config['name'],
            phot_fmt=obs_config.get('phot_fmt', 'flux'),
            usecols=obs_config.get('usecols'),
            ephemerides_file=obs_config.get('ephemerides_file'),
            bands=obs_config.get('bands', {})
        )
        register_observatory(obs)

# ============================================================================
# Built-in Observatories
# ============================================================================

# Fake

register_observatory(Observatory(
    name='WFIRST18',
    phot_fmt='flux',
    usecols=[0, 1, 2],
    ephemerides_file=os.path.join(
        exozippy.MULENS_DATA_PATH, '2018DataChallenge',
        'wfirst_ephemeris_W149.txt'),
    bands={
        'W149': {'color': 'darkorange', 'marker': 'o'},
        'Z087': {'color': 'darkcyan', 'marker': 's', 'zorder': 5}
    }
))

# Space-based
register_observatory(Observatory(
    name='Spitzer',
    phot_fmt='flux',
    usecols=[0, 1, 2],
    ephemerides_file=os.path.join(
        exozippy.MULENS_DATA_PATH, 'spitzer_ephemerides_2014_to_2019.txt'),
    bands={
        'L': {'color': 'red', 'marker': 'o'}
    }
))

# Ground-based


