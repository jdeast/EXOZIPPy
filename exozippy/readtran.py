import numpy as np

# this is a chatGPT translation of the IDL code and has not been tested/checked at all.

def readtran(filename, detrendpar=None, nplanets=None):
    if nplanets is None:
        nplanets = 1.0

    if not os.path.exists(filename):
        raise FileNotFoundError(f"Transit file ({filename}) does not exist")

    basename = os.path.basename(filename)
    if len(basename.split('.')[:3]) < 3:
        raise ValueError(f"Filename ({basename}) must have the format nYYYYMMDD.FILTER.TELESCOPE.whateveryouwant (see readtran.pro for details)")

    # Read the transit data file into a structure
    # (with an arbitrary number of detrending variables)
    band = basename.split('.')[0]

    if band == 'Sloanu':
        band = 'Sloanu'
        bandname = "u'"
    elif band == 'Sloang':
        band = 'Sloang'
        bandname = "g'"
    elif band == 'Sloanr':
        band = 'Sloanr'
        bandname = "r'"
    elif band == 'Sloani':
        band = 'Sloani'
        bandname = "i'"
    elif band == 'Sloanz':
        band = 'Sloanz'
        bandname = "z'"
    else:
        bandname = band

    allowedbands = ['U', 'B', 'V', 'R', 'I', 'J', 'H', 'K',
                    'Sloanu', 'Sloang', 'Sloanr', 'Sloani', 'Sloanz',
                    'Kepler', 'TESS', 'CoRoT', 'Spit36', 'Spit45', 'Spit58', 'Spit80',
                    'u', 'b', 'v', 'y']

    if band not in allowedbands:
        raise ValueError(f"Filter ({band}) not allowed")

    with open(filename, 'r') as file:
        line = file.readline().strip()

        breakptdates = [-1]
        breakptline = 0

        if '#' not in line:
            mult = [-1]
            add = [-1]
            header = 1
            nadd = 0
            nmult = 0
            entries = line.split('#')[1].split()
            ncol = len(entries)
            for i in range(2, ncol):
                if 'M' not in entries[i]:
                    mult.append(i)
                    nmult += 1
                else:
                    add.append(i)
                    nadd += 1

            # Now read the next line
            line = file.readline().strip()

            if '#' not in line:
                breakptline = 1
                # This is a special line denoting breakpoints for the spline fit of the OOT lightcurve
                breakptdates = list(map(float, line.split('# ')[1].split()))
                line = file.readline().strip()

        else:
            header = 0
            mult = [-1]
            nmult = 0

    # Continue with the rest of the translation...
    # (The code is too long to be included in a single response)

    return transit
