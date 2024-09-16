import os.path

import exozippy


def get_kwargs(filename):
    telescope, band = get_telescope_band_from_filename(filename)
    kwargs = get_observatory_kwargs(telescope)
    kwargs['bandpass'] = band
    kwargs['plot_properties'] = get_plot_properties(telescope, band)
    return kwargs


def get_telescope_band_from_filename(filename):
        basename = os.path.basename(filename).split('.')
        if len(basename) < 3:
            raise ValueError(
                "Filename ({0}) must have the format ".format(filename) +
                "nYYYYMMDD.FILTER.TELESCOPE.whateveryouwant")

        band = basename[1]
        telescope = basename[2]
        return telescope, band


def get_observatory_kwargs(telescope):
    kwargs = {}
    if telescope == 'WFIRST18':
        kwargs['phot_fmt'] = 'flux'
        kwargs['usecols'] = [0, 1, 2]
        kwargs['ephemerides_file'] = os.path.join(
            exozippy.MULENS_DATA_PATH, '2018DataChallenge',
             'wfirst_ephemeris_W149.txt')

    return kwargs


def get_plot_properties(telescope, band):
    plot_kwargs = {}
    plot_kwargs['label'] = '{0}-{1}'.format(telescope, band)

    if telescope == 'WFIRST18':
        if band == 'W149':
            plot_kwargs['color'] = 'magenta'
            plot_kwargs['marker'] = 'o'
        elif band == 'Z087':
            plot_kwargs['color'] = 'blue'
            plot_kwargs['marker'] = 's'
            plot_kwargs['zorder'] = 5

    return plot_kwargs

