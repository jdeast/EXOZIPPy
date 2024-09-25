import os.path
import numpy as np
import astropy.units as u
import sys
import yaml

import MulensModel as mm
import mmexofast as mmexo

from print_ex_metrics import print_metrics


def divide_settings(settings):
    """
    divide settings into expected, files, and all others
    """
    out = {**settings}
    expected = out.pop('expected')
    files_settings = out.pop('files')

    return (out, expected, files_settings)


def process_files(files, root=None, subdir=None):
    """
    make a list with list of files based on root and subdir provided.
    """
    if root is None:
        root_path = ""
    elif root == "MODULE_PATH":
        root_path = mm.MODULE_PATH
    else:
        raise ValueError("root not recognized: " + str(root))

    if subdir is not None:
        root_path = os.path.join(root_path, subdir)

    if not isinstance(files, list):
        raise TypeError('wrong value of files kwarg: ' + str(type(files)))

    out = []
    for file_ in files:
        out.append(os.path.join(root_path, file_))

    return out


def parse_units(physical):
    """
    Replace strings with corresponding astropy.units instances.
    The input is a dict with values being lists and we only check
    the last elements of these lists.
    """
    conversion = dict()

    for text in ['u.earthMass', 'u.M_earth', 'u.Mearth']:
        conversion[text] = u.earthMass
    for text in ['u.jupiterMass', 'u.M_jup', 'u.Mjup', 'u.M_jupiter', 'u.Mjupiter']:
        conversion[text] = u.jupiterMass
    for text in ['u.solMass', 'u.M_sun', 'u.Msun']:
        conversion[text] = u.solMass

    for (_, value) in physical.items():
        if not isinstance(value, list):
            continue
        if value[-1] in conversion:
            value[-1] = conversion[value[-1]]


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise ValueError('single yaml file with settings is needed')

    with open(sys.argv[1]) as in_file:
        settings = yaml.safe_load(in_file)

    (kwargs, expected, file_settings) = divide_settings(settings)

    files = process_files(**file_settings)
    kwargs['files'] = files

    parse_units(expected['physical'])

    print(kwargs)
    print()
    print(expected)

    results = mmexo.mmexofast.fit(**kwargs)

    print_metrics(results.final_parameters, expected)
