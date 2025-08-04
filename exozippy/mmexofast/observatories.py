import os.path

import exozippy


def get_kwargs(filename):
    """
    Parse the filename to create a *dict* of kwargs for creating a *MulensModel.MulensData* object.

    :param filename: *str*
        format = nYYYYMMDD.BAND.TELESCOPE.whateveryouwant

    :return: *dict*

    """
    telescope, band = get_telescope_band_from_filename(filename)
    kwargs = get_observatory_kwargs(telescope)
    kwargs['bandpass'] = band
    kwargs['plot_properties'] = get_plot_properties(telescope, band)
    return kwargs


def get_telescope_band_from_filename(filename):
    """
    get the telescope name and band from the filename

    :param filename: *str*
        format = nYYYYMMDD.BAND.TELESCOPE.whateveryouwant

    :return: *tuple* = *str*, *str*
        where the tuple is TELESCOPE, BAND

    """
    basename = os.path.basename(filename).split('.')
    if len(basename) < 3:
        raise ValueError(
            "Filename ({0}) must have the format ".format(filename) +
            "nYYYYMMDD.BAND.TELESCOPE.whateveryouwant")

    band = basename[1]
    telescope = basename[2]
    return telescope, band


def get_observatory_kwargs(telescope):
    """
    Use the TELESCOPE name to create a *dict* of kwargs for creating a *MulensModel.MulensData* object.

    :param telescope: *str*

    :return: *dict*
            If the TELESCOPE is not implemented, returns an empty *dict*

    """
    kwargs = {}
    if telescope == 'WFIRST18':
        kwargs['phot_fmt'] = 'flux'  # Note: ORIGINAL files are in mag, but they have been CONVERTED TO FLUX.
        kwargs['usecols'] = [0, 1, 2]
        kwargs['ephemerides_file'] = os.path.join(
            exozippy.MULENS_DATA_PATH, '2018DataChallenge',
             'wfirst_ephemeris_W149.txt')

    return kwargs


def get_plot_properties(telescope, band):
    """
    Use the TELESCOPE name and BAND to create a *dict* of plot_properties for a *MulensModel.MulensData* object.

    :param telescope: *str*
    :param band: *str*

    :return: *dict*

    """

    plot_kwargs = {}
    plot_kwargs['label'] = '{0}-{1}'.format(telescope, band)

    if telescope == 'WFIRST18':
        if band == 'W149':
            plot_kwargs['color'] = 'darkorange'
            plot_kwargs['marker'] = 'o'
        elif band == 'Z087':
            plot_kwargs['color'] = 'darkcyan'
            plot_kwargs['marker'] = 's'
            plot_kwargs['zorder'] = 5

    return plot_kwargs

