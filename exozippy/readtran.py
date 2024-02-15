import numpy as np
import glob
import ipdb
import os
from parameter import Parameter
import pymc as pm
import pytensor.tensor as pt

# this is a chatGPT translation of the IDL code and has not been tested/checked at all.

def readtran(filename, ndx=0, 
             tiv=False, ttv=False, tdeltav=False, 
             claret=False,
             user_params=None):

    if not os.path.exists(filename):
        raise FileNotFoundError(f"Transit file ({filename}) does not exist")

    basename = os.path.basename(filename)
    if len(basename.split('.')[:3]) < 3:
        raise ValueError(f"Filename ({basename}) must have the format nYYYYMMDD.FILTER.TELESCOPE.whateveryouwant (see readtran.pro for details)")

    # Read the transit data file into a structure
    # (with an arbitrary number of detrending variables)
    band = basename.split('.')[1]

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

    if band not in allowedbands and claret:
        raise ValueError(f"Filter ({band}) not allowed")

    mult = []
    add = []
    with open(filename, 'r') as file:
        line = file.readline().strip()

        breakptdates = []
        breakptline = 0

        if '#' in line:
            header = True
            entries = line.split('#')[1].strip().split()
            ncol = len(entries)
            for i in range(3, ncol):
                if entries[i][0] == "M": mult.append(i)
                else: add.append(i)

            # Now read the next line
            line = file.readline().strip()

            if '#' in line:
                breakptline = True
                # This is a special line denoting breakpoints for the spline fit of the OOT lightcurve
                breakptdates = list(map(float, line.split('# ')[1].split()))
                line = file.readline().strip()
            else: breakptline = False

        else:
            header = False
            breakptline=False

    entries = np.array(line.split())
    ncol = len(entries)

    if ncol < 3:
        raise ValueError(f"Transit file ({filename}) must contain at least 3 white-space delimited columns (BJD_TDB flux err). Comments are not allowed. The first line is {line}")

    # if no header, assume all detrending additive
    if not header:
        add = list(3 + np.arange(ncol-3))

    nadd = len(add)
    nmult = len(mult)

    if nadd + nmult + 3 != ncol:
        raise ValueError(f"Mismatch between header line and data lines for {filename}. The header MUST have one white-space delimited entry per column.")
        
    nskip = header+breakptline
    data = np.loadtxt(filename,comments='#',skiprows=nskip)

    night = 'n' + basename[1:4] + '-' + basename[5:6] + '-' + basename[7:8]
    label = basename.split('.')[2] + ' UT ' + night + ' ( ' + bandname + ')' 
    transit = {
        "band":band,
        "label": label,
        "rootlabel": "Transit Parameters:",
        "residuals" : None,
        "model" : None,
        "prettytime" : None,
        "prettymodel" : None,
    }
    transit["bjd"] = data[:,0]
    transit["flux"] = data[:,1]
    transit["err"] = data[:,2]
    transit["da"] = data[:,add]
    transit["dm"] = data[:,mult]

    transit["f0"] = Parameter('f0_'+str(ndx), lower=0.0, initval=1.0,upper=2.0,
                              latex='F_0',description='Baseline flux',latex_unit='', user_params=user_params)
    transit["jittervar"] = Parameter('jittervar_'+str(ndx), initval=0.0, upper=9e16,lower=-9e16,
                                     latex='\sigma_j^2',description='jitter variance',latex_unit='', user_params=user_params)
    transit["u1"] = Parameter('u1_'+str(ndx), lower=0.0, upper=2.0, initval=0.2,
                              latex='\mu_1',description='Linear limb darkening parameter',latex_unit='', user_params=user_params)
    transit["u2"] = Parameter('u2_'+str(ndx), lower=-1.0, upper=1.0, initval=0.2,
                              latex='\mu_2',description='Quadratic limb darkening parameter',latex_unit='', user_params=user_params)

    # physical limb darkening bounds (Kipping, 2013)
    # http://adsabs.harvard.edu/abs/2013MNRAS.435.2152K, eq 8
#    u1bound = pm.Potential("u1bound_" + str(ndx), pt.switch(transit["u1"].value < 0.0, -np.inf,0.0))
#    u1u2bound = pm.Potential("u1u2bound_" + str(ndx), pt.switch(transit["u1"].value+transit["u2"].value > 1.0, -np.inf,0.0))
#    u12u2bound = pm.Potential("u12u2bound_" + str(ndx), pt.switch(transit["u1"].value+2.0*transit["u2"].value < 0.0, -np.inf,0.0))

    if ttv:
        transit["ttv"] = Parameter('ttv_'+str(ndx), initval=0.0,unit=u.days,
                                   latex='TTV',description='Transit Timing Variation',latex_unit='days', user_params=user_params)
    if tdeltav:
            transit["tdeltav"] = Parameter('tdeltav_'+str(ndx), initval=0.0,
                                       latex='T$\delta$V',description='Transit depth Variation',latex_unit='', user_params=user_params)
    if tiv:
        transit["tiv"] = Parameter('tiv_'+str(ndx), initval=0.0,unit=u.deg,
                                   latex='TiV',description='Transit inclination Variation',latex_unit='deg', user_params=user_params)

    for i in range(transit["da"].shape[1]):
        transit["C_"+str(i)] = Parameter('C_'+str(ndx) + '_' + str(i), initval=0.0, lower=-1e9,upper=1e9,
                                         latex='C_{'+str(ndx)+','+str(i)+'}',
                                         description='Additive detrending coeff',latex_unit='', user_params=user_params)
        
    for i in range(transit["dm"].shape[1]):
        transit["M_"+str(i)] = Parameter('M_'+str(ndx) + '_' + str(i), initval=0.0, lower=-1e9,upper=1e9,
                                         latex='M_{'+str(ndx)+','+str(i)+'}',
                                         description='Multiplicative detrending coeff',latex_unit='', user_params=user_params)

    breakpts = []
    for i, date in enumerate(breakptdates):
        ndx = np.argmin(abs(transit.bjd-date))
        breakpts.append(ndx)
    transit["breakpts"] = breakpts

    # TODO: apply theoretical limb darkening priors based on Teff, logg, [Fe/H], band
    # probably doesn't belong here...
    
    return transit

if __name__ == "__main__":
    files = glob.glob('/home/jeastman/python/EXOZIPPy/examples/gj1132/*.dat')
    for file in files:
        transit = readtran(file)
        ipdb.set_trace()
        
    ipdb.set_trace()

