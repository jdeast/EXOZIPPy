#!/usr/bin/env python3
"""
mkticsed.py

Create an EXOZIPPy params YAML and SED YAML file for a TESS target.
Analogous to EXOFASTv2's mkticsed.pro -- queries TICv8.2 and associated catalogs.

Band names use the SVO Filter Profile Service standard (FACILITY/INSTRUMENT.FILTER).
The SED file is an EXOZIPPy YAML (not the EXOFASTv2 text table format).

Gaia DR3 photometry is used with Gaia DR2 filter curves (GAIA/GAIA2r.*) because
the NextGen BC grid ships only DR2 curves; the two are nearly identical.

This is the importable home of the former scripts/mkticsed.py. The CLI is
defined by build_parser() and driven by main(argv=None); scripts/mkticsed.py
is now a thin wrapper that calls main().

Usage:
    poetry run python scripts/mkticsed.py <TICID> [options]

Examples:
    poetry run python scripts/mkticsed.py 402026209 --star-name WASP-4
    poetry run python scripts/mkticsed.py TIC402026209 --outpath examples/wasp4
"""

import argparse
import math
import sys
import warnings
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier
from astroquery.ipac.irsa.irsa_dust import IrsaDust

try:
    from zero_point import zpt as _gaia_zpt
    HAS_GAIADR3_ZPT = True
except ImportError:
    HAS_GAIADR3_ZPT = False


# --- helpers ------------------------------------------------------------------

def strom_conv(V, sigV, by, sigby, m1, sigm1, c1, sigc1):
    """Convert Stromgren catalog indices to individual uvby magnitudes."""
    u_mag = V + 3 * by + 2 * m1 + c1
    v_mag = V + 2 * by + m1
    b_mag = V + by
    y_mag = V
    sig_u = math.sqrt(sigV**2 + (3 * sigby)**2 + (2 * sigm1)**2 + sigc1**2)
    sig_v = math.sqrt(sigV**2 + (2 * sigby)**2 + sigm1**2)
    sig_b = math.sqrt(sigV**2 + sigby**2)
    sig_y = sigV
    return u_mag, sig_u, v_mag, sig_v, b_mag, sig_b, y_mag, sig_y


def _get(table, col, row=0):
    """Return float from Vizier table row; NaN on any failure or mask."""
    if col not in table.colnames:
        return float('nan')
    try:
        val = table[col][row]
        if hasattr(val, 'mask') and val.mask:
            return float('nan')
        f = float(val)
        return f if np.isfinite(f) else float('nan')
    except (TypeError, ValueError):
        return float('nan')


def _gets(table, col, row=0):
    """Return stripped string from Vizier table row; '' on failure."""
    if col not in table.colnames:
        return ''
    try:
        val = table[col][row]
        if hasattr(val, 'mask') and val.mask:
            return ''
        return str(val).strip()
    except Exception:
        return ''


def _sep(ra1, dec1, ra2, dec2):
    """Angular separation in arcseconds; inf if any coordinate is NaN."""
    if not all(np.isfinite(v) for v in (ra1, dec1, ra2, dec2)):
        return float('inf')
    c1 = SkyCoord(ra=ra1 * u.deg, dec=dec1 * u.deg)
    c2 = SkyCoord(ra=ra2 * u.deg, dec=dec2 * u.deg)
    return c1.separation(c2).arcsec


def _nearest(table, ra, dec, racol='RAJ2000', deccol='DEJ2000'):
    """Return (index, sep_arcsec) of nearest row in table to (ra, dec)."""
    seps = [_sep(ra, dec, _get(table, racol, i), _get(table, deccol, i))
            for i in range(len(table))]
    idx = int(np.argmin(seps))
    return idx, seps[idx]


def query_region(catalog, ra, dec, radius_arcmin):
    """Cone-search Vizier; return first Table or None."""
    v = Vizier(columns=['**'], row_limit=-1)
    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame='icrs')
    try:
        res = v.query_region(coord, radius=radius_arcmin * u.arcmin, catalog=catalog)
        return res[0] if len(res) > 0 else None
    except Exception as e:
        warnings.warn(f"Vizier {catalog} query failed: {e}")
        return None


def query_id(catalog, target_id):
    """Name-based Vizier query; return first Table or None."""
    v = Vizier(columns=['**'], row_limit=-1)
    try:
        res = v.query_object(target_id, catalog=catalog)
        return res[0] if len(res) > 0 else None
    except Exception as e:
        warnings.warn(f"Vizier {catalog} query for '{target_id}' failed: {e}")
        return None


def schlegel_av(ra, dec):
    """Max Av upper limit from Schlegel+1998 dust map via IRSA (3.1 * E(B-V))."""
    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame='icrs')
    try:
        t = IrsaDust.get_query_table(coord, section='ebv')
        for col in ('ext SFD mean', 'EBV_SFD', 'ext SFD', 'E(B-V)'):
            if col in t.colnames:
                return float(t[col][0]) * 3.1
        # last resort: first finite positive column value
        for col in t.colnames:
            try:
                val = float(t[col][0])
                if 0 < val < 50:
                    return val * 3.1
            except Exception:
                pass
    except Exception as e:
        warnings.warn(f"Dust map query failed: {e}")
    return None


def _sed_entry(svo_name, mag, used_err, enabled=True, magsys="Vega"):
    """Return a SED YAML filter entry dict (or a commented-out version)."""
    return {
        "_enabled": enabled,
        "name": svo_name,
        "mag": round(float(mag), 6),
        "err": round(float(used_err), 6),
        "magsys": magsys,
    }


def _write_sed_yaml(path, sed_entries, model="NextGen", nstars=1, notes=None):
    """
    Write an EXOZIPPy SED YAML.

    Entries with _enabled=True go into the filters list.
    Entries with _enabled=False are written as YAML comments so the user
    can manually enable them later.
    """
    with open(path, "w") as f:
        if notes:
            for note in notes:
                f.write(f"# {note}\n")
            f.write("\n")
        f.write(f"model: {model}\n")
        f.write(f"nstars: {nstars}\n")
        f.write("filters:\n")
        for e in sed_entries:
            enabled = e.get("_enabled", True)
            name = e["name"]
            mag = e["mag"]
            err = e["err"]
            msys = e.get("magsys", "Vega")
            if enabled:
                f.write(f'    - name: "{name}"\n')
                f.write(f'      mag: {mag}\n')
                f.write(f'      err: {err}\n')
                if msys != "Vega":
                    f.write(f'      magsys: {msys}\n')
                f.write("\n")
            else:
                f.write(f'    # - name: "{name}"\n')
                f.write(f'    #   mag: {mag}\n')
                f.write(f'    #   err: {err}\n')
                if msys != "Vega":
                    f.write(f'    #   magsys: {msys}\n')
                f.write("\n")


# --- main function ------------------------------------------------------------

def mkticsed(ticid, star_name="Host", outpath=".", priorfile=None, sedfile=None,
             galex=False, tycho=False, stromgren=False, ucac=False, merm=False,
             kepler=False, dist=120.0, exofast=False):
    """
    Query TICv8.2 and photometric catalogs to create:
      - <ticid>.params.yaml  -- EXOZIPPy stellar priors
      - <ticid>.sed          -- photometric SED data

    Parameters
    ----------
    ticid : str or int
        TIC ID (numeric portion; 'TIC' prefix accepted).
    star_name : str
        Instance name for the star in params.yaml (e.g. 'Host').
    outpath : str
        Output directory.
    dist : float
        Cone-search radius in arcseconds (default 120).
    galex, tycho, stromgren, ucac, merm, kepler : bool
        Uncomment these photometry bands in the SED file.
    """
    outpath = Path(outpath)
    ticid = str(ticid).strip()
    if ticid.upper().startswith('TIC'):
        ticid = ticid[3:].strip()

    if priorfile is None:
        priorfile = outpath / f"{ticid}.params.yaml"
    else:
        priorfile = Path(priorfile)
    if sedfile is None:
        sedfile = outpath / f"{ticid}.sed"
    else:
        sedfile = Path(sedfile)

    sed_entries = []   # list of _sed_entry dicts for the SED YAML
    sed_notes = []     # comment lines for the SED YAML header
    yaml_data = {}     # full YAML key (star.Name.param) -> {field: value}
    notes = []         # comment lines for the params YAML header

    def key(param):
        return f"star.{star_name}.{param}"

    # --- 1. TICv8.2 -----------------------------------------------------------
    print(f"Querying TICv8.2 for TIC {ticid} ...", flush=True)
    qtic = query_id('IV/39/tic82', f"TIC {ticid}")
    if qtic is None or len(qtic) == 0:
        sys.exit(f"ERROR: TIC {ticid} not found in TICv8.2")

    # Find the matching row
    row = 0
    tic_col = 'TIC' if 'TIC' in qtic.colnames else None
    if tic_col:
        for i, val in enumerate(qtic[tic_col]):
            if str(val).strip() == ticid:
                row = i
                break

    disp = _gets(qtic, 'Disp', row)
    if disp in ('SPLIT', 'DUPLICATE'):
        notes.append(f"WARNING: TICv8.2 disposition is {disp}")
        dup = _gets(qtic, 'm_TIC', row)
        if dup and dup != '-1' and tic_col:
            notes.append(f"WARNING: redirecting to duplicate TIC {dup}")
            for i, val in enumerate(qtic[tic_col]):
                if str(val).strip() == dup:
                    row = i
                    break

    # Check for Washington Double Star catalog
    tic_ra = _get(qtic, 'RAJ2000', row)
    tic_dec = _get(qtic, 'DEJ2000', row)

    qwds = query_region('B/wds/wds', tic_ra, tic_dec, dist / 60.)
    if qwds is not None and len(qwds) > 0:
        sep2 = _gets(qwds, 'sep2', 0)
        sed_notes.append(f"WARNING: star in Washington Double Star catalog (sep {sep2}\")")
        sed_notes.append("WARNING: unresolved photometry will bias the SED fit")

    mass = _get(qtic, 'Mass', row)
    rad = _get(qtic, 'Rad', row)
    teff = _get(qtic, 'Teff', row)
    feh_tic = _get(qtic, '[M/H]', row)
    efeh = _get(qtic, 'e_[M/H]', row)
    ebv = _get(qtic, 'E_B-V', row)
    sebv = _get(qtic, 's_E_B-V', row)
    gaia_id = _gets(qtic, 'GAIA', row)
    mass2id = _gets(qtic, '_2MASS', row)
    wise_id = _gets(qtic, 'WISEA', row)
    tyc_id = _gets(qtic, 'TYC', row)

    # --- 2. Stellar params from TIC -------------------------------------------
    if np.isfinite(mass) and np.isfinite(rad) and np.isfinite(teff):
        yaml_data[key('logmass')] = {'initval': round(math.log10(mass), 5)}
        yaml_data[key('radius')] = {'initval': round(float(rad), 4)}
        yaml_data[key('teff')] = {'initval': round(float(teff), 1)}
    else:
        notes.append("WARNING: TIC mass/radius/teff incomplete -- using defaults")

    if np.isfinite(feh_tic):
        ufeh = max(0.08, float(efeh) if np.isfinite(efeh) else 0.08)
        yaml_data[key('feh')] = {
            'initval': round(float(feh_tic), 5),
            'mu': round(float(feh_tic), 5),
            'sigma': round(ufeh, 5),
        }

    # --- 3. Gaia DR3 parallax + photometry ------------------------------------
    print("Querying Gaia DR2 ...", flush=True)
    dr2_fallback_plx = float('nan')
    dr2_fallback_uplx = float('nan')

    qgaia2 = query_region('I/345/gaia2', tic_ra, tic_dec, dist / 60.)
    if qgaia2 is not None and len(qgaia2) > 0 and gaia_id:
        dr2_row = -1
        if 'Source' in qgaia2.colnames:
            for i, src in enumerate(qgaia2['Source']):
                if str(src).strip() == gaia_id:
                    dr2_row = i
                    break
        if dr2_row >= 0:
            dr2_plx = _get(qgaia2, 'Plx', dr2_row)
            dr2_eplx = _get(qgaia2, 'e_Plx', dr2_row)
            dr2_gmag = _get(qgaia2, 'Gmag', dr2_row)
            if np.isfinite(dr2_plx) and np.isfinite(dr2_eplx) and dr2_plx > 0:
                k = 1.08
                sigma_s = 0.021 if np.isfinite(dr2_gmag) and dr2_gmag <= 13 else 0.043
                c_plx = dr2_plx + 0.030    # Lindegren+2018 offset
                if c_plx > 0:
                    dr2_fallback_plx = c_plx
                    dr2_fallback_uplx = math.sqrt((k * dr2_eplx)**2 + sigma_s**2)

    print("Querying Gaia DR3 ...", flush=True)
    qgaia3 = query_region('I/355/gaiadr3', tic_ra, tic_dec, dist / 60.)
    target_ra = tic_ra
    target_dec = tic_dec
    target_pmra = float('nan')
    target_pmdec = float('nan')
    target_plx = float('nan')
    gaia_dr3_done = False

    if qgaia3 is not None and len(qgaia3) > 0:
        # Match by Gaia DR2 source ID (same ID is used in TICv8.2), fall back to position
        g3row = -1
        if gaia_id and 'Source' in qgaia3.colnames:
            for i, src in enumerate(qgaia3['Source']):
                if str(src).strip() == gaia_id:
                    g3row = i
                    break
        if g3row == -1:
            idx, sep = _nearest(qgaia3, tic_ra, tic_dec, 'RA_ICRS', 'DE_ICRS')
            if sep < 1.0:
                g3row = idx
                notes.append(f"TICv8.2 Gaia ID didn't match DR3 Source; using star at {sep:.2f}\"")

        if g3row >= 0:
            g3_plx = _get(qgaia3, 'Plx', g3row)
            g3_eplx = _get(qgaia3, 'e_Plx', g3row)
            g3_gmag = _get(qgaia3, 'Gmag', g3row)
            g3_egmag = _get(qgaia3, 'e_Gmag', g3row)
            g3_bpmag = _get(qgaia3, 'BPmag', g3row)
            g3_ebpm = _get(qgaia3, 'e_BPmag', g3row)
            g3_rpmag = _get(qgaia3, 'RPmag', g3row)
            g3_erpm = _get(qgaia3, 'e_RPmag', g3row)
            g3_ruwe = _get(qgaia3, 'RUWE', g3row)
            g3_nueff = _get(qgaia3, 'nueff', g3row)
            g3_pscol = _get(qgaia3, 'pscol', g3row)
            g3_elat = _get(qgaia3, 'ELAT', g3row)
            g3_solv = _get(qgaia3, 'Solved', g3row)

            if np.isfinite(g3_ruwe):
                sed_notes.append(f"Gaia DR3 RUWE = {g3_ruwe:.4f}")
                sed_notes.append("RUWE > 1.4 is a strong indicator of stellar multiplicity")

            if np.isfinite(g3_plx) and np.isfinite(g3_eplx) and g3_plx > 0:
                uplx = math.sqrt(g3_eplx**2 + 0.01**2)  # 0.01 mas systematic floor
                zp = 0.0
                zp_msg = "raw (gaiadr3-zeropoint not installed)"
                if HAS_GAIADR3_ZPT:
                    in5 = (np.isfinite(g3_solv) and int(g3_solv) == 31
                           and np.isfinite(g3_nueff) and 1.1 <= g3_nueff <= 1.9)
                    in6 = (np.isfinite(g3_solv) and int(g3_solv) == 95
                           and np.isfinite(g3_pscol) and 1.24 <= g3_pscol <= 1.72)
                    if (in5 or in6) and np.isfinite(g3_gmag) and 6 <= g3_gmag <= 21:
                        try:
                            zp = _gaia_zpt.get_zpt(g3_gmag, g3_nueff, g3_pscol,
                                                   g3_elat, int(g3_solv))
                            zp_msg = f"corrected by -{zp:.5f} mas (Lindegren+2021)"
                        except Exception:
                            zp_msg = "correction failed; using raw"
                    else:
                        zp_msg = "out of Lindegren+2021 range; using raw"

                corrected_plx = g3_plx - zp
                notes.append(
                    f"Gaia DR3 parallax {g3_plx:.5f} mas, {zp_msg}; "
                    f"uncertainty {g3_eplx:.5f} + 0.01 mas systematic = {uplx:.5f}")

                if corrected_plx > 0:
                    d_pc = 1000.0 / corrected_plx
                    d_sig = 1000.0 * uplx / corrected_plx**2
                    target_plx = corrected_plx
                    yaml_data[key('distance')] = {
                        'initval': round(d_pc, 3),
                        'mu': round(d_pc, 3),
                        'sigma': round(d_sig, 3),
                    }
                    gaia_dr3_done = True

                target_ra = _get(qgaia3, 'RA_ICRS', g3row)
                target_dec = _get(qgaia3, 'DE_ICRS', g3row)
                target_pmra = _get(qgaia3, 'pmRA', g3row)
                target_pmdec = _get(qgaia3, 'pmDE', g3row)

            # Gaia DR3 photometry with DR2 filter curves (nearest available BC grid)
            sed_notes.append("Gaia DR3 photometry used with GAIA/GAIA2r filter curves (DR2); "
                             "differences are <1 mmag for typical stars")
            if g3_gmag > -9 and np.isfinite(g3_egmag) and g3_egmag < 1:
                sed_entries.append(_sed_entry('GAIA/GAIA2r.G', g3_gmag, max(0.02, g3_egmag)))
            if g3_bpmag > -9 and np.isfinite(g3_ebpm) and g3_ebpm < 1:
                sed_entries.append(_sed_entry('GAIA/GAIA2r.Gbp', g3_bpmag, max(0.02, g3_ebpm)))
            if g3_rpmag > -9 and np.isfinite(g3_erpm) and g3_erpm < 1:
                sed_entries.append(_sed_entry('GAIA/GAIA2r.Grp', g3_rpmag, max(0.02, g3_erpm)))

    if not gaia_dr3_done and np.isfinite(dr2_fallback_plx):
        notes.append("DR3 parallax unavailable; using Gaia DR2 with Lindegren+2018 correction")
        d_pc = 1000.0 / dr2_fallback_plx
        d_sig = 1000.0 * dr2_fallback_uplx / dr2_fallback_plx**2
        yaml_data[key('distance')] = {
            'initval': round(d_pc, 3),
            'mu': round(d_pc, 3),
            'sigma': round(d_sig, 3),
        }

    # --- 4. 2MASS photometry --------------------------------------------------
    print("Querying 2MASS ...", flush=True)
    q2m = query_region('II/246/out', tic_ra, tic_dec, dist / 60.)
    if q2m is not None and len(q2m) > 0:
        m_row = -1
        if mass2id and '_2MASS' in q2m.colnames:
            for i, tid in enumerate(q2m['_2MASS']):
                if str(tid).strip() == mass2id:
                    m_row = i
                    break
        if m_row == -1:
            idx, sep = _nearest(q2m, tic_ra, tic_dec)
            if sep < 2.0:
                m_row = idx
        if m_row >= 0:
            jmag = _get(q2m, 'Jmag', m_row)
            ejmag = _get(q2m, 'e_Jmag', m_row)
            hmag = _get(q2m, 'Hmag', m_row)
            ehmag = _get(q2m, 'e_Hmag', m_row)
            kmag = _get(q2m, 'Kmag', m_row)
            ekmag = _get(q2m, 'e_Kmag', m_row)
            if np.isfinite(jmag) and np.isfinite(ejmag) and ejmag < 1:
                sed_entries.append(_sed_entry('2MASS/2MASS.J', jmag, max(0.02, ejmag)))
            if np.isfinite(hmag) and np.isfinite(ehmag) and ehmag < 1:
                sed_entries.append(_sed_entry('2MASS/2MASS.H', hmag, max(0.02, ehmag)))
            if np.isfinite(kmag) and np.isfinite(ekmag) and ekmag < 1:
                sed_entries.append(_sed_entry('2MASS/2MASS.Ks', kmag, max(0.02, ekmag)))
                yaml_data[key('appks')] = {
                    'initval': round(kmag, 6),
                    'mu': round(kmag, 6),
                    'sigma': round(max(0.02, ekmag), 6),
                }

    # --- 5. WISE photometry ---------------------------------------------------
    print("Querying AllWISE ...", flush=True)
    qw = query_region('II/328/allwise', tic_ra, tic_dec, dist / 60.)
    if qw is not None and len(qw) > 0:
        w_row = -1
        if wise_id and 'AllWISE' in qw.colnames:
            for i, wid in enumerate(qw['AllWISE']):
                if str(wid).strip() == wise_id:
                    w_row = i
                    break
        if w_row == -1:
            idx, sep = _nearest(qw, tic_ra, tic_dec)
            if sep < 15.0:
                w_row = idx
        if w_row >= 0:
            w1 = _get(qw, 'W1mag', w_row)
            ew1 = _get(qw, 'e_W1mag', w_row)
            w2 = _get(qw, 'W2mag', w_row)
            ew2 = _get(qw, 'e_W2mag', w_row)
            w3 = _get(qw, 'W3mag', w_row)
            ew3 = _get(qw, 'e_W3mag', w_row)
            w4 = _get(qw, 'W4mag', w_row)
            ew4 = _get(qw, 'e_W4mag', w_row)
            if np.isfinite(w1) and np.isfinite(ew1) and ew1 < 1:
                sed_entries.append(_sed_entry('WISE/WISE.W1', w1, max(0.03, ew1)))
            if np.isfinite(w2) and np.isfinite(ew2) and ew2 < 1:
                sed_entries.append(_sed_entry('WISE/WISE.W2', w2, max(0.03, ew2)))
            if np.isfinite(w3) and np.isfinite(ew3) and ew3 < 1:
                sed_entries.append(_sed_entry('WISE/WISE.W3', w3, max(0.03, ew3)))
            if np.isfinite(w4) and np.isfinite(ew4) and ew4 < 1:
                sed_entries.append(_sed_entry('WISE/WISE.W4', w4, max(0.10, ew4)))

    # --- 6. Extinction upper limit from Schlegel+1998 dust map ----------------
    print("Querying Schlegel dust map ...", flush=True)
    max_av = schlegel_av(tic_ra, tic_dec)
    if max_av is not None and max_av > 0:
        yaml_data[key('av')] = {'upper': round(max_av, 4)}
        notes.append(f"Av < {max_av:.4f} mag  (3.1 x E(B-V) Schlegel+1998 upper limit)")
    elif np.isfinite(ebv) and np.isfinite(sebv):
        av_val = ebv * 3.1
        uav = max(0.02, sebv) * 3.1
        yaml_data[key('av')] = {
            'initval': round(av_val, 5),
            'mu': round(av_val, 5),
            'sigma': round(uav, 5),
        }
    else:
        notes.append("WARNING: could not determine extinction -- using default bounds")

    # --- 7. [Fe/H] from Paunzen+2015 Stromgren if TIC has none ----------------
    if not np.isfinite(feh_tic) and key('feh') not in yaml_data:
        print("Querying Paunzen+2015 Stromgren for [Fe/H] ...", flush=True)
        qpz = query_region('J/A+A/580/A23/catalog', tic_ra, tic_dec, dist / 60.)
        if qpz is not None and len(qpz) > 0:
            p_row = -1
            if tyc_id and 'TYC1' in qpz.colnames:
                for i in range(len(qpz)):
                    try:
                        tyc_str = (f"{int(_get(qpz,'TYC1',i)):04d}-"
                                   f"{int(_get(qpz,'TYC2',i)):05d}-"
                                   f"{int(_get(qpz,'TYC3',i)):01d}")
                        if tyc_str == tyc_id:
                            p_row = i
                            break
                    except (ValueError, OverflowError):
                        continue
            if p_row == -1:
                idx, sep = _nearest(qpz, tic_ra, tic_dec)
                if sep < dist:
                    p_row = idx

            if p_row >= 0:
                by = _get(qpz, 'b-y', p_row)
                sby = _get(qpz, 'e_b-y', p_row)
                m1 = _get(qpz, 'm1', p_row)
                sm1 = _get(qpz, 'e_m1', p_row)
                c1 = _get(qpz, 'c1', p_row)
                sc1 = _get(qpz, 'e_c1', p_row)

                if np.isfinite(by) and np.isfinite(m1) and np.isfinite(c1) and m1 > 0:
                    # Cassegrande+2011, eq. 2 (FGK solar neighbourhood)
                    if 0.23 < by < 0.63 and 0.05 < m1 <= 0.68 and 0.13 < c1 <= 0.60:
                        feh_val = (3.927 * math.log10(m1) - 14.459 * m1**3
                                   - 5.394 * by * math.log10(m1) + 36.069 * by * m1**3
                                   + 3.537 * c1 * math.log10(m1) - 3.500 * m1**3 * c1
                                   + 11.034 * by - 22.780 * by**2
                                   + 10.684 * c1 - 6.759 * c1**2 - 1.548)
                        yaml_data[key('feh')] = {
                            'initval': round(feh_val, 5),
                            'mu': round(feh_val, 5),
                            'sigma': 0.10,
                        }
                        notes.append("[Fe/H] from Paunzen+2015 Stromgren via Cassegrande+2011 eq. 2")
                    # Cassegrande+2011, eq. 3 (cooler range)
                    elif 0.43 < by < 0.63 and 0.07 < m1 <= 0.68 and 0.16 < c1 <= 0.49:
                        feh_val = (-0.116 * c1 - 1.624 * c1**2 + 8.955 * c1 * by
                                   + 42.008 * by - 99.596 * by**2 + 64.245 * by**3
                                   + 8.928 * c1 * m1 + 17.275 * m1 - 48.106 * m1**2
                                   + 45.802 * m1**3 - 8.467)
                        yaml_data[key('feh')] = {
                            'initval': round(feh_val, 5),
                            'mu': round(feh_val, 5),
                            'sigma': 0.12,
                        }
                        notes.append("[Fe/H] from Paunzen+2015 Stromgren via Cassegrande+2011 eq. 3")

    # Last-resort: wide Gaussian [Fe/H] prior
    if key('feh') not in yaml_data:
        yaml_data[key('feh')] = {'initval': 0.0, 'mu': 0.0, 'sigma': 1.0}
        notes.append("[Fe/H]: no value found; using wide prior N(0, 1)")

    # --- 8. Optional: Tycho-2 BT/VT (disabled by default) ---------------------
    qtyc2 = query_region('I/259/TYC2', tic_ra, tic_dec, dist / 60.)
    if qtyc2 is not None and len(qtyc2) > 0:
        t_row = 0
        if len(qtyc2) > 1:
            t_row, _ = _nearest(qtyc2, tic_ra, tic_dec, 'RAmdeg', 'DEmdeg')
        bt = _get(qtyc2, 'BTmag', t_row)
        ebt = _get(qtyc2, 'e_BTmag', t_row)
        vt = _get(qtyc2, 'VTmag', t_row)
        evt = _get(qtyc2, 'e_VTmag', t_row)
        if np.isfinite(bt) and np.isfinite(ebt):
            sed_entries.append(_sed_entry('TYCHO/TYCHO.B', bt, max(0.02, ebt), enabled=tycho))
        if np.isfinite(vt) and np.isfinite(evt):
            sed_entries.append(_sed_entry('TYCHO/TYCHO.V', vt, max(0.02, evt), enabled=tycho))
    else:
        qtyc2 = None

    # --- 9. Optional: UCAC4 / APASS DR6 (disabled by default) -----------------
    qucac = query_region('UCAC4', tic_ra, tic_dec, dist / 60.)
    if qucac is not None and len(qucac) > 0:
        u_row = 0
        if len(qucac) > 1:
            u_row, _ = _nearest(qucac, tic_ra, tic_dec)
        bt_ref = _get(qtyc2, 'BTmag', 0) if qtyc2 is not None else float('nan')
        vt_ref = _get(qtyc2, 'VTmag', 0) if qtyc2 is not None else float('nan')
        B_mag = _get(qucac, 'Bmag', u_row)
        eB = _get(qucac, 'e_Bmag', u_row)
        V_mag = _get(qucac, 'Vmag', u_row)
        eV = _get(qucac, 'e_Vmag', u_row)
        g_mag = _get(qucac, 'gmag', u_row)
        eg = _get(qucac, 'e_gmag', u_row)
        r_mag = _get(qucac, 'rmag', u_row)
        er = _get(qucac, 'e_rmag', u_row)
        i_mag = _get(qucac, 'imag', u_row)
        ei = _get(qucac, 'e_imag', u_row)
        # UCAC4 stores APASS errors as 0.01 mag integers; avoid duplicating Tycho
        if np.isfinite(B_mag) and np.isfinite(eB) and eB != 99 and abs(B_mag - bt_ref) > 0.01:
            sed_entries.append(_sed_entry('Generic/Bessell.B', B_mag, max(0.02, eB * 0.01), enabled=ucac))
        if np.isfinite(V_mag) and np.isfinite(eV) and eV != 99 and abs(V_mag - vt_ref) > 0.01:
            sed_entries.append(_sed_entry('Generic/Bessell.V', V_mag, max(0.02, eV * 0.01), enabled=ucac))
        if np.isfinite(g_mag) and np.isfinite(eg):
            sed_entries.append(_sed_entry('SLOAN/SDSS.g', g_mag, max(0.02, eg * 0.01), enabled=ucac))
        if np.isfinite(r_mag) and np.isfinite(er):
            sed_entries.append(_sed_entry('SLOAN/SDSS.r', r_mag, max(0.02, er * 0.01), enabled=ucac))
        if np.isfinite(i_mag) and np.isfinite(ei):
            sed_entries.append(_sed_entry('SLOAN/SDSS.i', i_mag, max(0.02, ei * 0.01), enabled=ucac))

    # --- 10. Optional: Paunzen+2015 Stromgren photometry for SED --------------
    qpz_full = query_region('J/A+A/580/A23/catalog', tic_ra, tic_dec, dist / 60.)
    if qpz_full is not None and len(qpz_full) > 0:
        pz_row = 0
        if len(qpz_full) > 1:
            pz_row, _ = _nearest(qpz_full, tic_ra, tic_dec)
        vmag = _get(qpz_full, 'Vmag', pz_row)
        evmag = _get(qpz_full, 'e_Vmag', pz_row)
        by = _get(qpz_full, 'b-y', pz_row)
        eby = _get(qpz_full, 'e_b-y', pz_row)
        m1 = _get(qpz_full, 'm1', pz_row)
        em1 = _get(qpz_full, 'e_m1', pz_row)
        c1 = _get(qpz_full, 'c1', pz_row)
        ec1 = _get(qpz_full, 'e_c1', pz_row)
        if np.isfinite(vmag) and np.isfinite(by) and np.isfinite(m1) and np.isfinite(c1):
            u_m, su, v_m, sv, b_m, sb, y_m, sy = strom_conv(
                vmag, max(0.01, evmag if np.isfinite(evmag) else 0.01),
                by, max(0.02, eby if np.isfinite(eby) else 0.02),
                m1, max(0.02, em1 if np.isfinite(em1) else 0.02),
                c1, max(0.02, ec1 if np.isfinite(ec1) else 0.02))
            if np.isfinite(u_m):
                sed_entries.append(_sed_entry('Generic/Stromgren.u', u_m, max(0.02, su), enabled=stromgren))
            if np.isfinite(v_m):
                sed_entries.append(_sed_entry('Generic/Stromgren.v', v_m, max(0.02, sv), enabled=stromgren))
            if np.isfinite(b_m):
                sed_entries.append(_sed_entry('Generic/Stromgren.b', b_m, max(0.02, sb), enabled=stromgren))
            if np.isfinite(y_m):
                sed_entries.append(_sed_entry('Generic/Stromgren.y', y_m, max(0.02, sy), enabled=stromgren))

    # --- 11. Optional: Mermilliod+1994 UBV (disabled by default) --------------
    qmerm = query_region('II/168/ubvmeans', tic_ra, tic_dec, dist / 60.)
    if qmerm is not None and len(qmerm) > 0:
        me_row = 0
        if len(qmerm) > 1:
            me_row, _ = _nearest(qmerm, tic_ra, tic_dec)
        V_m = _get(qmerm, 'Vmag', me_row)
        eV_m = _get(qmerm, 'e_Vmag', me_row)
        BV_m = _get(qmerm, 'B-V', me_row)
        eBV = _get(qmerm, 'e_B-V', me_row)
        UB_m = _get(qmerm, 'U-B', me_row)
        eUB = _get(qmerm, 'e_U-B', me_row)
        if np.isfinite(V_m) and np.isfinite(eV_m) and eV_m < 1:
            B_v = BV_m + V_m
            eB_v = math.sqrt(eBV**2 + eV_m**2) if np.isfinite(eBV) else float('nan')
            U_v = UB_m + B_v
            eU_v = math.sqrt(eUB**2 + eB_v**2) if (np.isfinite(eUB) and np.isfinite(eB_v)) else float('nan')
            if np.isfinite(U_v) and np.isfinite(eU_v) and eU_v < 1:
                sed_entries.append(_sed_entry('Generic/Bessell.U', U_v, max(0.02, eU_v), enabled=merm))
            if np.isfinite(B_v) and np.isfinite(eB_v) and eB_v < 1:
                sed_entries.append(_sed_entry('Generic/Bessell.B', B_v, max(0.02, eB_v), enabled=merm))
            sed_entries.append(_sed_entry('Generic/Bessell.V', V_m, max(0.02, eV_m), enabled=merm))

    # --- 12. Optional: GALEX DR5 (disabled by default; UV models unreliable) --
    qgalex = query_region('II/312/ais', tic_ra, tic_dec, dist / 60.)
    if qgalex is not None and len(qgalex) > 0 and 'FUV' in qgalex.colnames:
        ga_row = 0
        if len(qgalex) > 1:
            ga_row, _ = _nearest(qgalex, tic_ra, tic_dec)
        fuv = _get(qgalex, 'FUV', ga_row)
        efuv = _get(qgalex, 'e_FUV', ga_row)
        nuv = _get(qgalex, 'NUV', ga_row)
        enuv = _get(qgalex, 'e_NUV', ga_row)
        sed_notes.append("GALEX: atmospheric models are unreliable in the UV")
        if np.isfinite(fuv) and np.isfinite(efuv):
            sed_entries.append(_sed_entry('GALEX/GALEX.FUV', fuv, max(0.1, efuv), enabled=galex))
        if np.isfinite(nuv) and np.isfinite(enuv):
            sed_entries.append(_sed_entry('GALEX/GALEX.NUV', nuv, max(0.1, enuv), enabled=galex))

    # --- 13. Write output files -----------------------------------------------
    outpath.mkdir(parents=True, exist_ok=True)

    _write_sed_yaml(sedfile,
                    sed_entries,
                    notes=[f"TIC {ticid}"] + sed_notes)

    if exofast:
        exofast_sedfile = sedfile.with_suffix('.sed.txt')
        with open(exofast_sedfile, 'w') as f:
            f.write("# bandname magnitude used_errors catalog_errors\n")
            f.write(f"# TIC {ticid}\n")
            for e in sed_entries:
                name = e["name"]
                mag = e["mag"]
                err = e["err"]
                en = e.get("_enabled", True)
                pfx = "" if en else "# "
                f.write(f"{pfx}{name:<30s} {mag:9.6f} {err:.6f} {err:.6f}\n")
        print(f"Written: {exofast_sedfile}")

    with open(priorfile, 'w') as f:
        f.write(f"# EXOZIPPy params for TIC {ticid}  (star instance: {star_name})\n")
        f.write("# Generated by mkticsed.py from TICv8.2\n")
        if notes:
            f.write("#\n")
            for note in notes:
                f.write(f"# {note}\n")
        f.write("\n")
        for ykey, fields in yaml_data.items():
            f.write(f"{ykey}:\n")
            for field, val in fields.items():
                f.write(f"    {field}: {val}\n")
            f.write("\n")

    print(f"Written: {sedfile}")
    print(f"Written: {priorfile}")


def build_parser():
    """Return the argparse parser for the mkticsed utility."""
    p = argparse.ArgumentParser(
        prog='mkticsed.py',
        description="Create EXOZIPPy params YAML and SED file from TICv8.2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('ticid', help='TIC ID (numeric, or TIC####)')
    p.add_argument('--star-name', default='Host', help='Star instance name in params YAML')
    p.add_argument('--outpath', default='.', help='Output directory')
    p.add_argument('--priorfile', default=None, help='Override params YAML path')
    p.add_argument('--sedfile', default=None, help='Override SED file path')
    p.add_argument('--dist', default=120.0, type=float,
                   help='Cone-search radius in arcseconds')
    p.add_argument('--galex', action='store_true', help='Uncomment GALEX photometry')
    p.add_argument('--tycho', action='store_true', help='Uncomment Tycho-2 BT/VT')
    p.add_argument('--stromgren', action='store_true', help='Uncomment Stromgren photometry')
    p.add_argument('--ucac', action='store_true', help='Uncomment UCAC4/APASS photometry')
    p.add_argument('--merm', action='store_true', help='Uncomment Mermilliod UBV')
    p.add_argument('--kepler', action='store_true', help='(reserved, no-op)')
    p.add_argument('--exofast', action='store_true',
                   help='Also write an EXOFASTv2-format text SED file (<ticid>.sed.txt)')
    return p


def main(argv=None):
    """CLI entry point. Parses argv (or sys.argv) and runs mkticsed."""
    args = build_parser().parse_args(argv)
    mkticsed(
        ticid=args.ticid,
        star_name=args.star_name,
        outpath=args.outpath,
        priorfile=args.priorfile,
        sedfile=args.sedfile,
        galex=args.galex,
        tycho=args.tycho,
        stromgren=args.stromgren,
        ucac=args.ucac,
        merm=args.merm,
        kepler=args.kepler,
        dist=args.dist,
        exofast=args.exofast,
    )


if __name__ == '__main__':
    main()
