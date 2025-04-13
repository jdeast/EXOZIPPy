# Full rewrite of mkticsed.pro from EXOFASTv2
# This version requires astroquery and astropy

import os
import numpy as np
from astroquery.vizier import Vizier
from astroquery.irsa_dust import IrsaDust
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.io import ascii

Vizier.ROW_LIMIT = -1

# unfinished
def get_tic_catalog(ticid=None, ra=None, dec=None):
    """Query TICv8.2 by TIC ID or RA/Dec"""
    catalog = 'IV/39/tic82'
    if ticid is not None:
        result = Vizier(columns=['*']).query_constraints(catalog=catalog, TIC=str(ticid))
    elif ra is not None and dec is not None:
        coord = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg))
        result = Vizier(columns=['*']).query_region(coord, radius=2*u.arcsec, catalog=catalog)
    else:
        raise ValueError("Must provide TIC ID or RA/Dec")

    if len(result) == 0:
        raise ValueError("No match found in TICv8.2")
    print(f"Found {len(result)} matches in TICv8.2")
    return result[0][0]  # return best match


def get_av_prior(ra, dec):
    coord = SkyCoord(ra, dec, unit='deg')
    result = IrsaDust.get_extinction_table(coord, show_progress=False)
    for col in result.colnames:
        if 'SandF' in col:
            ebv = result[col][0]  # use any column containing SandF
            break
    else:
        raise KeyError("No valid Schlafly & Finkbeiner extinction column found.")

    maxav = ebv * 3.1 * 1.5 * 0.87
    return float(maxav)


def write_prior_file(tic, filename='tic.priors'):
    with open(filename, 'w') as f:
        f.write("#### TICv8.2 ####\n")

        if np.isfinite(tic['Mass']) and np.isfinite(tic['Rad']) and np.isfinite(tic['Teff']):
            f.write(f"mstar {tic['Mass']:.2f}\n")
            f.write(f"rstar {tic['Rad']:.2f}\n")
            f.write(f"teff {int(tic['Teff'])}\n")

        feh = tic.get('[M/H] ')
        ufeh = tic.get('e__m_H_')
        if feh is not None and ufeh is not None and np.isfinite(feh) and np.isfinite(ufeh):
            ufeh = max(ufeh, 0.08)
            f.write(f"feh {feh:.5f} {ufeh:.5f}\n")
        else:
            print(f"[WARNING] No [Fe/H] information for TIC {tic['TIC']}")

        try:
            av = tic['e_B_V'] * 3.1
            uav = max(tic['s_E_B_V'] * 3.1, 0.02)
            if np.isfinite(av) and np.isfinite(uav):
                f.write(f"av {av:.5f} {uav:.5f}\n")
                f.write("##############\n")
            else:
                raise ValueError
        except:
            maxav = get_av_prior(tic['RAJ2000'], tic['DEJ2000'])
            f.write("##############\n")
            f.write(f"av 0 -1 0 {maxav:.5f}\n")

        if np.isfinite(tic['Plx']) and tic['Plx'] > 0 and np.isfinite(tic['e_Plx']):
            corrected = tic['Plx'] + 0.03
            sigma = np.sqrt(1.08**2 * tic['e_Plx']**2 + 0.01**2)
            f.write(f"# parallax corrected\n")
            f.write(f"parallax {corrected:.5f} {sigma:.5f}\n")


def write_sed_file(tic, filename='tic.sed'):
    with open(filename, 'w') as f:
        f.write("# bandname magnitude used_errors catalog_errors\n")
        fmt = "{:<13} {:9.6f} {:6.3f} {:6.3f}\n"

        # Gaia
        if np.isfinite(tic['Gmag']) and np.isfinite(tic['e_Gmag']):
            f.write(fmt.format('Gaia', tic['Gmag'], max(0.02, tic['e_Gmag']), tic['e_Gmag']))
        if np.isfinite(tic['BPmag']) and np.isfinite(tic['e_BPmag']):
            f.write(fmt.format('GaiaBP', tic['BPmag'], max(0.02, tic['e_BPmag']), tic['e_BPmag']))
        if np.isfinite(tic['RPmag']) and np.isfinite(tic['e_RPmag']):
            f.write(fmt.format('GaiaRP', tic['RPmag'], max(0.02, tic['e_RPmag']), tic['e_RPmag']))

        # 2MASS
        if np.isfinite(tic['Jmag']) and np.isfinite(tic['e_Jmag']):
            f.write(fmt.format('J2M', tic['Jmag'], max(0.02, tic['e_Jmag']), tic['e_Jmag']))
        if np.isfinite(tic['Hmag']) and np.isfinite(tic['e_Hmag']):
            f.write(fmt.format('H2M', tic['Hmag'], max(0.02, tic['e_Hmag']), tic['e_Hmag']))
        if np.isfinite(tic['Kmag']) and np.isfinite(tic['e_Kmag']):
            f.write(fmt.format('K2M', tic['Kmag'], max(0.02, tic['e_Kmag']), tic['e_Kmag']))
            f.write("# Apparent 2MASS K magnitude for the Mann relation\n")
            f.write(f"appks {tic['Kmag']:.6f} {max(0.02, tic['e_Kmag']):.3f}\n")

        # WISE
        for band, abbr, err in [("W1mag", "WISE1", "e_W1mag"),
                                ("W2mag", "WISE2", "e_W2mag"),
                                ("W3mag", "WISE3", "e_W3mag"),
                                ("W4mag", "WISE4", "e_W4mag")]:
            if np.isfinite(tic.get(band, np.nan)) and np.isfinite(tic.get(err, np.nan)):
                default_err = 0.03 if band != "W4mag" else 0.10
                f.write(fmt.format(abbr, tic[band], max(default_err, tic[err]), tic[err]))


def mkticsed(ticid=None, ra=None, dec=None, priorfile='tic.priors', sedfile='tic.sed'):
    tic = get_tic_catalog(ticid=ticid, ra=ra, dec=dec)
    write_prior_file(tic, priorfile)
    write_sed_file(tic, sedfile)
    print(f"Wrote {priorfile} and {sedfile} for TIC {tic['TIC']}")
