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
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astroquery.ipac.irsa.irsa_dust import IrsaDust
from astroquery.vizier import Vizier

try:
    from zero_point import zpt as _gaia_zpt

    # get_zpt() raises "The table of coefficients have not been
    # initialized!!" until this runs.  It never ran, so the Lindegren+2021
    # parallax zero point silently evaluated to 0.0 on every target, for
    # every run, and the uncorrected parallax became the star's distance
    # PRIOR (mu/sigma) in the params file the user then fits with.
    _gaia_zpt.load_tables()
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
    sig_u = math.sqrt(sigV**2 + (3 * sigby) ** 2 + (2 * sigm1) ** 2 + sigc1**2)
    sig_v = math.sqrt(sigV**2 + (2 * sigby) ** 2 + sigm1**2)
    sig_b = math.sqrt(sigV**2 + sigby**2)
    sig_y = sigV
    return u_mag, sig_u, v_mag, sig_v, b_mag, sig_b, y_mag, sig_y


def _get(table, col, row=0):
    """Return float from Vizier table row; NaN on any failure or mask."""
    if col not in table.colnames:
        return float("nan")
    try:
        val = table[col][row]
        if hasattr(val, "mask") and val.mask:
            return float("nan")
        f = float(val)
        return f if np.isfinite(f) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _gets(table, col, row=0):
    """Return stripped string from Vizier table row; '' on failure."""
    if col not in table.colnames:
        return ""
    try:
        val = table[col][row]
        if hasattr(val, "mask") and val.mask:
            return ""
        return str(val).strip()
    except Exception:
        return ""


def _is_distinct(mag, ref, tol=0.01):
    """True if `mag` is a different measurement from the reference `ref`.

    Used to keep a catalog's photometry from entering the SED twice when two
    catalogs report the same measurement. A non-finite `ref` means the
    comparison star was never found, so there is nothing to duplicate and the
    magnitude is kept -- the NaN must not silently vote "drop it", which is
    what a bare ``abs(mag - ref) > tol`` does.
    """
    if not np.isfinite(ref):
        return True
    return abs(mag - ref) > tol


def _sep(ra1, dec1, ra2, dec2):
    """Angular separation in arcseconds; inf if any coordinate is NaN."""
    if not all(np.isfinite(v) for v in (ra1, dec1, ra2, dec2)):
        return float("inf")
    c1 = SkyCoord(ra=ra1 * u.deg, dec=dec1 * u.deg)
    c2 = SkyCoord(ra=ra2 * u.deg, dec=dec2 * u.deg)
    return c1.separation(c2).arcsec


def _nearest(table, ra, dec, racol="RAJ2000", deccol="DEJ2000"):
    """Return (index, sep_arcsec) of nearest row in table to (ra, dec)."""
    seps = [
        _sep(ra, dec, _get(table, racol, i), _get(table, deccol, i))
        for i in range(len(table))
    ]
    idx = int(np.argmin(seps))
    return idx, seps[idx]


def query_region(catalog, ra, dec, radius_arcmin):
    """Cone-search Vizier; return first Table or None."""
    v = Vizier(columns=["**"], row_limit=-1)
    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    try:
        res = v.query_region(
            coord, radius=radius_arcmin * u.arcmin, catalog=catalog
        )
        return res[0] if len(res) > 0 else None
    except Exception as e:
        warnings.warn(f"Vizier {catalog} query failed: {e}")
        return None


def query_id(catalog, target_id):
    """Name-based Vizier query; return first Table or None."""
    v = Vizier(columns=["**"], row_limit=-1)
    try:
        res = v.query_object(target_id, catalog=catalog)
        return res[0] if len(res) > 0 else None
    except Exception as e:
        warnings.warn(f"Vizier {catalog} query for '{target_id}' failed: {e}")
        return None


def schlegel_av(ra, dec):
    """Max Av upper limit from Schlegel+1998 dust map via IRSA (3.1 * E(B-V))."""
    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    try:
        t = IrsaDust.get_query_table(coord, section="ebv")
        for col in ("ext SFD mean", "EBV_SFD", "ext SFD", "E(B-V)"):
            if col in t.colnames:
                return float(t[col][0]) * 3.1
        # No "last resort: first column that parses in (0, 50)".  That
        # returned an arbitrary column -- a coordinate, or SandF rather than
        # SFD -- times 3.1, and the caller writes the result as a HARD upper
        # bound on star.av (the logit transform truncates the posterior
        # there) under a note claiming Schlegel+1998 provenance it does not
        # have.  Returning None is handled by the caller.
        warnings.warn(
            f"IRSA dust table has none of the expected E(B-V) columns "
            f"(got {list(t.colnames)}); no Av upper limit will be written."
        )
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


def _write_parallax_prior(yaml_data, notes, key, plx, uplx):
    """Record the astrometric measurement as a prior in PARALLAX space.

    The measurement is Gaussian in parallax, so that is where the prior
    belongs.  Propagating it into a distance prior
    (``d = 1000/plx``, ``sigma_d = 1000*sigma_plx/plx**2``) is a
    first-order expansion of a nonlinear map: it is symmetric in distance
    while the true distance posterior implied by a Gaussian parallax is
    skewed with a long far-side tail, so it biases whenever
    ``plx/sigma_plx`` is not large (Lutz & Kelker 1973; Bailer-Jones 2015).
    EXOFASTv2's ``mkticsed.pro`` writes ``parallax`` too; the distance
    conversion was introduced by the Python port.

    ``star.parallax`` is a derived parameter (``1000/distance``) and takes
    an ordinary ``mu``/``sigma`` Gaussian prior through the standard
    machinery -- ``Parameter.build_pymc`` adds ``gaussian_prior.<label>``
    for derived elements with a sigma.  The relaxation engine then
    inverts ``Eq(parallax, 1000/distance)`` to seed ``distance``, so no
    distance entry is written here.

    A NEGATIVE parallax is written live rather than commented out.  It is
    a perfectly good measurement, and in parallax space it is well
    defined: ``distance`` is the sampled coordinate and is bounded
    positive, so ``1000/distance`` is always positive and a Gaussian
    centered below zero is a finite one-sided penalty favoring large
    distances -- nothing ever inverts a negative parallax at runtime.
    (``mkticsed.pro`` comments the line out with "Negative parallax is not
    allowed"; that reflects EXOFASTv2 sampling distance directly.)  The
    relaxation engine declines to seed a distance from it, so ``distance``
    keeps its defaults.yaml start -- which can be a poor one, hence the
    suggested seed in the notes.
    """
    yaml_data[key("parallax")] = {
        "mu": round(float(plx), 5),
        "sigma": round(float(uplx), 5),
    }
    if plx > 0:
        notes.append(
            f"parallax prior N({plx:.5f}, {uplx:.5f}) mas -> "
            f"distance ~ {1000.0 / plx:.3f} pc.  The prior is applied in "
            f"parallax space (star.parallax = 1000/distance): a distance "
            f"prior from first-order propagation biases below "
            f"plx/sigma ~ 10"
        )
        return
    notes.append(
        f"WARNING: the corrected parallax is NEGATIVE ({plx:.5f} +/- "
        f"{uplx:.5f} mas).  It is still applied, as a prior on "
        f"star.parallax: distance is the sampled coordinate and is bounded "
        f"positive, so the prior is a finite one-sided penalty favoring "
        f"large distances."
    )
    hi = plx + 3.0 * uplx
    if hi > 0:
        notes.append(
            f"WARNING: no distance start can be derived from a negative "
            f"parallax, so star.distance keeps its 10 pc default -- a very "
            f"poor start here.  To start at the 3-sigma minimum distance "
            f"instead, add:  {key('distance')}: {{initval: "
            f"{1000.0 / hi:.3f}}}"
        )
    else:
        notes.append(
            "WARNING: the 3-sigma upper limit on the parallax is still "
            "negative; the astrometry constrains the distance only very "
            "weakly.  Set a star.distance initval by hand."
        )


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
                f.write(f"      mag: {mag}\n")
                f.write(f"      err: {err}\n")
                if msys != "Vega":
                    f.write(f"      magsys: {msys}\n")
                f.write("\n")
            else:
                f.write(f'    # - name: "{name}"\n')
                f.write(f"    #   mag: {mag}\n")
                f.write(f"    #   err: {err}\n")
                if msys != "Vega":
                    f.write(f"    #   magsys: {msys}\n")
                f.write("\n")


# --- the catalog table --------------------------------------------------------
#
# Every photometric catalog is queried, cross-matched, floored and appended by
# the same three functions below (_query_catalog, _add_band, _add_catalog_bands).
# Everything that differs between catalogs lives in CATALOGS, so the table is
# the documentation: read a row to know that catalog's match radius, its error
# floors, which sentinel values it uses for "not measured", and which CLI flag
# gates its rows.
#
# Two catalogs contribute photometry that is not a column -- Stromgren uvby
# (converted from the b-y/m1/c1 indices) and Mermilliod UBV (converted from the
# B-V/U-B colors) -- so they carry no `bands` and call _add_band directly with
# the magnitudes they compute. They are still in the table for their query and
# cross-match settings.

_INF = float("inf")
_NAN = float("nan")


@dataclass(frozen=True)
class Band:
    """One photometric row a catalog can contribute to the SED.

    ``floor`` is the error floor in magnitudes: the SED gets
    ``max(floor, err)``, never the catalog's raw error alone.  ``dedup`` names
    a reference magnitude (see ``_add_catalog_bands``) that this measurement
    must differ from, so the same photometry cannot enter the SED twice from
    two catalogs.
    """

    svo: str  # SVO Filter Profile Service name written to the .sed
    mag_col: str  # Vizier column holding the magnitude
    err_col: str  # Vizier column holding its uncertainty
    floor: float  # error floor, mag
    dedup: str = ""  # reference magnitude this must be distinct from


@dataclass(frozen=True)
class Catalog:
    """How to query one catalog, find the target in it, and floor its errors.

    ``max_sep`` is the positional-fallback cutoff in arcsec, used only when the
    cross-match ID misses: ``inf`` accepts the nearest row whatever the
    separation (catalogs with no ID to match on), and ``None`` declines a
    positional fallback entirely (Gaia DR2, where an ID mismatch means the TIC
    is pointing at a different star).
    """

    vizier: str  # Vizier catalog identifier
    label: str  # what the progress line calls it
    bands: tuple = ()  # rows this catalog contributes, in SED order
    flag: str = ""  # mkticsed kwarg gating these rows ("" = always live)
    id_col: str = ""  # cross-match column, matched against a TIC ID
    ra_col: str = "RAJ2000"  # coordinate columns for the positional fallback
    dec_col: str = "DEJ2000"
    max_sep: float = _INF  # positional-fallback cutoff, arcsec (see above)
    max_err: float = _INF  # reject a band whose error is >= this, mag
    err_scale: float = 1.0  # catalog error units -> magnitudes
    no_data_err: float = _NAN  # raw error value meaning "not measured"
    min_mag: float = -_INF  # magnitudes <= this are "not measured" sentinels
    require_col: str = ""  # skip the catalog outright without this column


CATALOGS = {
    # Gaia DR3 photometry is written against the DR2 filter curves: the
    # NextGen BC grid ships only those, and they agree to <1 mmag.
    "gaia3": Catalog(
        vizier="I/355/gaiadr3",
        label="Gaia DR3",
        id_col="Source",  # TICv8.2 carries the DR2 source ID, reused in DR3
        ra_col="RA_ICRS",
        dec_col="DE_ICRS",
        max_sep=1.0,
        max_err=1.0,
        min_mag=-9.0,  # Gaia writes -99 for an absent band
        bands=(
            Band("GAIA/GAIA2r.G", "Gmag", "e_Gmag", 0.02),
            Band("GAIA/GAIA2r.Gbp", "BPmag", "e_BPmag", 0.02),
            Band("GAIA/GAIA2r.Grp", "RPmag", "e_RPmag", 0.02),
        ),
    ),
    # Parallax only -- the DR3 photometry above supersedes DR2's.
    "gaia2": Catalog(
        vizier="I/345/gaia2",
        label="Gaia DR2",
        id_col="Source",
        max_sep=None,
    ),
    "2mass": Catalog(
        vizier="II/246/out",
        label="2MASS",
        id_col="_2MASS",
        max_sep=2.0,
        max_err=1.0,
        bands=(
            Band("2MASS/2MASS.J", "Jmag", "e_Jmag", 0.02),
            Band("2MASS/2MASS.H", "Hmag", "e_Hmag", 0.02),
            Band("2MASS/2MASS.Ks", "Kmag", "e_Kmag", 0.02),
        ),
    ),
    "wise": Catalog(
        vizier="II/328/allwise",
        label="AllWISE",
        id_col="AllWISE",
        max_sep=15.0,  # WISE's beam is wide; so is its acceptable match
        max_err=1.0,
        bands=(
            Band("WISE/WISE.W1", "W1mag", "e_W1mag", 0.03),
            Band("WISE/WISE.W2", "W2mag", "e_W2mag", 0.03),
            Band("WISE/WISE.W3", "W3mag", "e_W3mag", 0.03),
            # W4 is the least trustworthy band, hence its own floor.
            Band("WISE/WISE.W4", "W4mag", "e_W4mag", 0.10),
        ),
    ),
    "tycho": Catalog(
        vizier="I/259/TYC2",
        label="Tycho-2",
        flag="tycho",
        ra_col="RAmdeg",
        dec_col="DEmdeg",
        bands=(
            Band("TYCHO/TYCHO.B", "BTmag", "e_BTmag", 0.02),
            Band("TYCHO/TYCHO.V", "VTmag", "e_VTmag", 0.02),
        ),
    ),
    # UCAC4 republishes APASS, storing its errors as hundredths of a magnitude
    # with 99 as the "no data" sentinel -- which must be rejected on every
    # band, or it survives as a fabricated max(0.02, 99 * 0.01) = 0.99 mag
    # error on a magnitude that was never measured.  B and V are deduplicated
    # against the Tycho photometry the SED may already carry.
    "ucac": Catalog(
        vizier="UCAC4",
        label="UCAC4/APASS DR6",
        flag="ucac",
        err_scale=0.01,
        no_data_err=99.0,
        bands=(
            Band("Generic/Bessell.B", "Bmag", "e_Bmag", 0.02, dedup="BT"),
            Band("Generic/Bessell.V", "Vmag", "e_Vmag", 0.02, dedup="VT"),
            Band("SLOAN/SDSS.g", "gmag", "e_gmag", 0.02),
            Band("SLOAN/SDSS.r", "rmag", "e_rmag", 0.02),
            Band("SLOAN/SDSS.i", "imag", "e_imag", 0.02),
        ),
    ),
    # uvby, converted from the catalog's b-y/m1/c1 indices by strom_conv.
    "stromgren": Catalog(
        vizier="J/A+A/580/A23/catalog",
        label="Stromgren photometry (Paunzen+2015)",
        flag="stromgren",
    ),
    # UBV, reconstructed from V and the B-V/U-B colors.
    "mermilliod": Catalog(
        vizier="II/168/ubvmeans",
        label="Mermilliod+1994 UBV",
        flag="merm",
    ),
    "galex": Catalog(
        vizier="II/312/ais",
        label="GALEX DR5",
        flag="galex",
        require_col="FUV",
        bands=(
            Band("GALEX/GALEX.FUV", "FUV", "e_FUV", 0.10),
            Band("GALEX/GALEX.NUV", "NUV", "e_NUV", 0.10),
        ),
    ),
}

# Floors used by the two converted-photometry catalogs, which have no Band
# rows of their own.  0.02 mag, the same floor every optical catalog above
# carries.
_STROMGREN_FLOOR = 0.02
_MERMILLIOD_FLOOR = 0.02


def _match_row(table, ra, dec, cat, id_value=""):
    """Locate the target's row in a cone-search result.

    The catalog's own cross-match ID wins; failing that the nearest row is
    taken if it lies within ``cat.max_sep`` arcsec.  Returns
    ``(index, separation_arcsec)`` -- index -1 when nothing matched, and a NaN
    separation when the ID matched (no positional comparison was made).
    """
    if cat.id_col and id_value and cat.id_col in table.colnames:
        for i, val in enumerate(table[cat.id_col]):
            if str(val).strip() == id_value:
                return i, _NAN
    if cat.max_sep is None:
        return -1, _INF
    idx, sep = _nearest(table, ra, dec, cat.ra_col, cat.dec_col)
    if not math.isinf(cat.max_sep) and not sep < cat.max_sep:
        return -1, sep
    return idx, sep


def _query_catalog(cat, ra, dec, radius_arcsec, id_value=""):
    """Cone-search one catalog and find the target in it.

    Returns ``(table, row, sep)``; ``row`` is -1 when the catalog has nothing
    usable for this star (no result, no rows, a missing ``require_col``, or no
    acceptable match).
    """
    print(f"Querying {cat.label} ...", flush=True)
    table = query_region(cat.vizier, ra, dec, radius_arcsec / 60.0)
    if table is None or len(table) == 0:
        return None, -1, _INF
    if cat.require_col and cat.require_col not in table.colnames:
        return table, -1, _INF
    row, sep = _match_row(table, ra, dec, cat, id_value)
    return table, row, sep


def _add_band(
    entries,
    svo,
    mag,
    err,
    floor,
    enabled=True,
    max_err=_INF,
    min_mag=-_INF,
):
    """Append one SED row, applying its rejection gates and error floor.

    Returns the ``(mag, used_err)`` written, or None when the measurement was
    rejected: a non-finite or sentinel magnitude (``mag <= min_mag``), or a
    non-finite or implausibly large error (``err >= max_err``).  A finite
    uncertainty is a precondition for inclusion everywhere -- substituting a
    plausible-looking one for an absent one over-weights the row in the SED
    likelihood and biases Teff, radius and Av.
    """
    if not (np.isfinite(mag) and mag > min_mag):
        return None
    if not (np.isfinite(err) and err < max_err):
        return None
    used_err = max(floor, err)
    entries.append(_sed_entry(svo, mag, used_err, enabled=enabled))
    return mag, used_err


def _add_catalog_bands(entries, cat, table, row, enabled=True, refs=None):
    """Append every band ``cat`` contributes for its matched row.

    ``refs`` maps a Band's ``dedup`` key to a reference magnitude that the
    measurement must differ from (see ``_is_distinct``).  Returns
    ``{svo_name: (mag, used_err)}`` for the rows actually written, so a caller
    can reuse a magnitude it also needs as a prior.
    """
    written = {}
    refs = refs or {}
    for band in cat.bands:
        raw_err = _get(table, band.err_col, row)
        if raw_err == cat.no_data_err:
            continue
        mag = _get(table, band.mag_col, row)
        if band.dedup and not _is_distinct(mag, refs.get(band.dedup, _NAN)):
            continue
        got = _add_band(
            entries,
            band.svo,
            mag,
            raw_err * cat.err_scale,
            band.floor,
            enabled=enabled,
            max_err=cat.max_err,
            min_mag=cat.min_mag,
        )
        if got is not None:
            written[band.svo] = got
    return written


# --- main function ------------------------------------------------------------


def mkticsed(
    ticid,
    star_name="Host",
    outpath=".",
    priorfile=None,
    sedfile=None,
    galex=False,
    tycho=False,
    stromgren=False,
    ucac=False,
    merm=False,
    kepler=False,
    dist=120.0,
    exofast=False,
):
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
    if ticid.upper().startswith("TIC"):
        ticid = ticid[3:].strip()

    if priorfile is None:
        priorfile = outpath / f"{ticid}.params.yaml"
    else:
        priorfile = Path(priorfile)
    if sedfile is None:
        sedfile = outpath / f"{ticid}.sed"
    else:
        sedfile = Path(sedfile)

    sed_entries = []  # list of _sed_entry dicts for the SED YAML
    sed_notes = []  # comment lines for the SED YAML header
    yaml_data = {}  # full YAML key (star.Name.param) -> {field: value}
    notes = []  # comment lines for the params YAML header

    def key(param):
        return f"star.{star_name}.{param}"

    # Which CLI flag un-comments each catalog's rows (Catalog.flag); a catalog
    # with no flag is always live.
    enabled_by_flag = {
        "galex": galex,
        "tycho": tycho,
        "stromgren": stromgren,
        "ucac": ucac,
        "merm": merm,
    }

    def is_live(cat):
        """True when this catalog's rows go into the SED uncommented."""
        return enabled_by_flag.get(cat.flag, True)

    # --- 1. TICv8.2 -----------------------------------------------------------
    print(f"Querying TICv8.2 for TIC {ticid} ...", flush=True)
    qtic = query_id("IV/39/tic82", f"TIC {ticid}")
    if qtic is None or len(qtic) == 0:
        sys.exit(f"ERROR: TIC {ticid} not found in TICv8.2")

    # Find the matching row
    row = 0
    tic_col = "TIC" if "TIC" in qtic.colnames else None
    if tic_col:
        for i, val in enumerate(qtic[tic_col]):
            if str(val).strip() == ticid:
                row = i
                break

    disp = _gets(qtic, "Disp", row)
    if disp in ("SPLIT", "DUPLICATE"):
        notes.append(f"WARNING: TICv8.2 disposition is {disp}")
        dup = _gets(qtic, "m_TIC", row)
        if dup and dup != "-1" and tic_col:
            notes.append(f"WARNING: redirecting to duplicate TIC {dup}")
            for i, val in enumerate(qtic[tic_col]):
                if str(val).strip() == dup:
                    row = i
                    break

    # Check for Washington Double Star catalog
    tic_ra = _get(qtic, "RAJ2000", row)
    tic_dec = _get(qtic, "DEJ2000", row)

    qwds = query_region("B/wds/wds", tic_ra, tic_dec, dist / 60.0)
    if qwds is not None and len(qwds) > 0:
        sep2 = _gets(qwds, "sep2", 0)
        sed_notes.append(
            f'WARNING: star in Washington Double Star catalog (sep {sep2}")'
        )
        sed_notes.append(
            "WARNING: unresolved photometry will bias the SED fit"
        )

    mass = _get(qtic, "Mass", row)
    rad = _get(qtic, "Rad", row)
    teff = _get(qtic, "Teff", row)
    feh_tic = _get(qtic, "[M/H]", row)
    efeh = _get(qtic, "e_[M/H]", row)
    ebv = _get(qtic, "E_B-V", row)
    sebv = _get(qtic, "s_E_B-V", row)
    gaia_id = _gets(qtic, "GAIA", row)
    mass2id = _gets(qtic, "_2MASS", row)
    wise_id = _gets(qtic, "WISEA", row)
    tyc_id = _gets(qtic, "TYC", row)

    # --- 2. Stellar params from TIC -------------------------------------------
    if np.isfinite(mass) and np.isfinite(rad) and np.isfinite(teff):
        yaml_data[key("logmass")] = {"initval": round(math.log10(mass), 5)}
        yaml_data[key("radius")] = {"initval": round(float(rad), 4)}
        yaml_data[key("teff")] = {"initval": round(float(teff), 1)}
    else:
        notes.append(
            "WARNING: TIC mass/radius/teff incomplete -- using defaults"
        )

    if np.isfinite(feh_tic):
        ufeh = max(0.08, float(efeh) if np.isfinite(efeh) else 0.08)
        yaml_data[key("feh")] = {
            "initval": round(float(feh_tic), 5),
            "mu": round(float(feh_tic), 5),
            "sigma": round(ufeh, 5),
        }

    # --- 3. Gaia DR3 parallax + photometry ------------------------------------
    dr2_fallback_plx = float("nan")
    dr2_fallback_uplx = float("nan")

    qgaia2, dr2_row, _ = _query_catalog(
        CATALOGS["gaia2"], tic_ra, tic_dec, dist, gaia_id
    )
    if dr2_row >= 0:
        dr2_plx = _get(qgaia2, "Plx", dr2_row)
        dr2_eplx = _get(qgaia2, "e_Plx", dr2_row)
        dr2_gmag = _get(qgaia2, "Gmag", dr2_row)
        if np.isfinite(dr2_plx) and np.isfinite(dr2_eplx) and dr2_plx > 0:
            k = 1.08
            sigma_s = (
                0.021 if np.isfinite(dr2_gmag) and dr2_gmag <= 13 else 0.043
            )
            c_plx = dr2_plx + 0.030  # Lindegren+2018 offset
            if c_plx > 0:
                dr2_fallback_plx = c_plx
                dr2_fallback_uplx = math.sqrt((k * dr2_eplx) ** 2 + sigma_s**2)

    # Matched by Gaia DR2 source ID (TICv8.2 carries it and DR3 reuses it),
    # falling back to the nearest star within 1".
    qgaia3, g3row, g3sep = _query_catalog(
        CATALOGS["gaia3"], tic_ra, tic_dec, dist, gaia_id
    )
    target_ra = tic_ra
    target_dec = tic_dec
    target_pmra = float("nan")
    target_pmdec = float("nan")
    gaia_dr3_done = False

    if g3row >= 0:
        if np.isfinite(g3sep):
            notes.append(
                f"TICv8.2 Gaia ID didn't match DR3 Source; "
                f'using star at {g3sep:.2f}"'
            )
        g3_plx = _get(qgaia3, "Plx", g3row)
        g3_eplx = _get(qgaia3, "e_Plx", g3row)
        g3_gmag = _get(qgaia3, "Gmag", g3row)
        g3_ruwe = _get(qgaia3, "RUWE", g3row)
        g3_nueff = _get(qgaia3, "nueff", g3row)
        g3_pscol = _get(qgaia3, "pscol", g3row)
        g3_elat = _get(qgaia3, "ELAT", g3row)
        g3_solv = _get(qgaia3, "Solved", g3row)

        if np.isfinite(g3_ruwe):
            sed_notes.append(f"Gaia DR3 RUWE = {g3_ruwe:.4f}")
            sed_notes.append(
                "RUWE > 1.4 is a strong indicator of stellar multiplicity"
            )

        # No positivity gate: the prior is written in PARALLAX space, so
        # a negative measured parallax is representable (see below).
        if np.isfinite(g3_plx) and np.isfinite(g3_eplx):
            uplx = math.sqrt(g3_eplx**2 + 0.01**2)  # 0.01 mas systematic floor
            zp = 0.0
            zp_msg = "raw (gaiadr3-zeropoint not installed)"
            if HAS_GAIADR3_ZPT:
                in5 = (
                    np.isfinite(g3_solv)
                    and int(g3_solv) == 31
                    and np.isfinite(g3_nueff)
                    and 1.1 <= g3_nueff <= 1.9
                )
                in6 = (
                    np.isfinite(g3_solv)
                    and int(g3_solv) == 95
                    and np.isfinite(g3_pscol)
                    and 1.24 <= g3_pscol <= 1.72
                )
                if (
                    (in5 or in6)
                    and np.isfinite(g3_gmag)
                    and 6 <= g3_gmag <= 21
                ):
                    try:
                        # numpy scalars, not Python floats: under NEP 50
                        # (numpy >= 2, our floor) zpt's internal
                        # np.can_cast(inp, float) raises TypeError on a
                        # Python float.  That was the second reason the
                        # correction never once applied.
                        zp = float(
                            _gaia_zpt.get_zpt(
                                np.float64(g3_gmag),
                                np.float64(g3_nueff),
                                np.float64(g3_pscol),
                                np.float64(g3_elat),
                                np.int64(int(g3_solv)),
                            )
                        )
                        zp_msg = (
                            f"corrected by {-zp:+.5f} mas (Lindegren+2021)"
                        )
                    except Exception as exc:
                        # Do NOT fall through with zp = 0.  The result
                        # here is not a missing number, it is a distance
                        # PRIOR biased by the whole zero point (~0.02-0.05
                        # mas, i.e. several sigma at 1 kpc) with nothing
                        # but a note string to say so.  The out-of-range
                        # case is handled by the else branch below and is
                        # a legitimate "no published correction".
                        raise RuntimeError(
                            f"Gaia DR3 parallax zero-point correction "
                            f"failed for this target "
                            f"({type(exc).__name__}: {exc}).  Refusing to "
                            f"write a distance prior from the uncorrected "
                            f"parallax."
                        ) from exc
                else:
                    zp_msg = "out of Lindegren+2021 range; using raw"

            corrected_plx = g3_plx - zp
            notes.append(
                f"Gaia DR3 parallax {g3_plx:.5f} mas, {zp_msg}; "
                f"uncertainty {g3_eplx:.5f} + 0.01 mas systematic = {uplx:.5f}"
            )

            _write_parallax_prior(yaml_data, notes, key, corrected_plx, uplx)
            gaia_dr3_done = True

            target_ra = _get(qgaia3, "RA_ICRS", g3row)
            target_dec = _get(qgaia3, "DE_ICRS", g3row)
            target_pmra = _get(qgaia3, "pmRA", g3row)
            target_pmdec = _get(qgaia3, "pmDE", g3row)

        # Gaia DR3 photometry with DR2 filter curves (nearest available BC grid)
        sed_notes.append(
            "Gaia DR3 photometry used with GAIA/GAIA2r filter curves (DR2); "
            "differences are <1 mmag for typical stars"
        )
        _add_catalog_bands(sed_entries, CATALOGS["gaia3"], qgaia3, g3row)

    if not gaia_dr3_done and np.isfinite(dr2_fallback_plx):
        notes.append(
            "DR3 parallax unavailable; using Gaia DR2 with Lindegren+2018 correction"
        )
        _write_parallax_prior(
            yaml_data, notes, key, dr2_fallback_plx, dr2_fallback_uplx
        )

    # --- 4. 2MASS photometry --------------------------------------------------
    q2m, m_row, _ = _query_catalog(
        CATALOGS["2mass"], tic_ra, tic_dec, dist, mass2id
    )
    if m_row >= 0:
        written = _add_catalog_bands(
            sed_entries, CATALOGS["2mass"], q2m, m_row
        )
        # Ks doubles as the input to the Mann+2015/2019 relations, so it is
        # written as a prior too -- with the same floored error the SED got.
        if "2MASS/2MASS.Ks" in written:
            kmag, ekmag = written["2MASS/2MASS.Ks"]
            yaml_data[key("appks")] = {
                "initval": round(kmag, 6),
                "mu": round(kmag, 6),
                "sigma": round(ekmag, 6),
            }

    # --- 5. WISE photometry ---------------------------------------------------
    qw, w_row, _ = _query_catalog(
        CATALOGS["wise"], tic_ra, tic_dec, dist, wise_id
    )
    if w_row >= 0:
        _add_catalog_bands(sed_entries, CATALOGS["wise"], qw, w_row)

    # --- 6. Extinction upper limit from Schlegel+1998 dust map ----------------
    print("Querying Schlegel dust map ...", flush=True)
    max_av = schlegel_av(tic_ra, tic_dec)
    if max_av is not None and max_av > 0:
        yaml_data[key("av")] = {"upper": round(max_av, 4)}
        notes.append(
            f"Av < {max_av:.4f} mag  (3.1 x E(B-V) Schlegel+1998 upper limit)"
        )
    elif np.isfinite(ebv) and np.isfinite(sebv):
        av_val = ebv * 3.1
        uav = max(0.02, sebv) * 3.1
        yaml_data[key("av")] = {
            "initval": round(av_val, 5),
            "mu": round(av_val, 5),
            "sigma": round(uav, 5),
        }
    else:
        notes.append(
            "WARNING: could not determine extinction -- using default bounds"
        )

    # --- 7. [Fe/H] from Paunzen+2015 Stromgren if TIC has none ----------------
    if not np.isfinite(feh_tic) and key("feh") not in yaml_data:
        print("Querying Paunzen+2015 Stromgren for [Fe/H] ...", flush=True)
        qpz = query_region(
            CATALOGS["stromgren"].vizier, tic_ra, tic_dec, dist / 60.0
        )
        if qpz is not None and len(qpz) > 0:
            p_row = -1
            if tyc_id and "TYC1" in qpz.colnames:
                for i in range(len(qpz)):
                    try:
                        tyc_str = (
                            f"{int(_get(qpz, 'TYC1', i)):04d}-"
                            f"{int(_get(qpz, 'TYC2', i)):05d}-"
                            f"{int(_get(qpz, 'TYC3', i)):01d}"
                        )
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
                by = _get(qpz, "b-y", p_row)
                sby = _get(qpz, "e_b-y", p_row)
                m1 = _get(qpz, "m1", p_row)
                sm1 = _get(qpz, "e_m1", p_row)
                c1 = _get(qpz, "c1", p_row)
                sc1 = _get(qpz, "e_c1", p_row)

                if (
                    np.isfinite(by)
                    and np.isfinite(m1)
                    and np.isfinite(c1)
                    and m1 > 0
                ):
                    # Casagrande+2011, eq. 2 (FGK solar neighbourhood)
                    if (
                        0.23 < by < 0.63
                        and 0.05 < m1 <= 0.68
                        and 0.13 < c1 <= 0.60
                    ):
                        feh_val = (
                            3.927 * math.log10(m1)
                            - 14.459 * m1**3
                            - 5.394 * by * math.log10(m1)
                            + 36.069 * by * m1**3
                            + 3.537 * c1 * math.log10(m1)
                            - 3.500 * m1**3 * c1
                            + 11.034 * by
                            - 22.780 * by**2
                            + 10.684 * c1
                            - 6.759 * c1**2
                            - 1.548
                        )
                        yaml_data[key("feh")] = {
                            "initval": round(feh_val, 5),
                            "mu": round(feh_val, 5),
                            "sigma": 0.10,
                        }
                        notes.append(
                            "[Fe/H] from Paunzen+2015 Stromgren via Casagrande+2011 eq. 2"
                        )
                    # Casagrande+2011, eq. 3 (cooler range)
                    elif (
                        0.43 < by < 0.63
                        and 0.07 < m1 <= 0.68
                        and 0.16 < c1 <= 0.49
                    ):
                        feh_val = (
                            -0.116 * c1
                            - 1.624 * c1**2
                            + 8.955 * c1 * by
                            + 42.008 * by
                            - 99.596 * by**2
                            + 64.245 * by**3
                            + 8.928 * c1 * m1
                            + 17.275 * m1
                            - 48.106 * m1**2
                            + 45.802 * m1**3
                            - 8.467
                        )
                        yaml_data[key("feh")] = {
                            "initval": round(feh_val, 5),
                            "mu": round(feh_val, 5),
                            "sigma": 0.12,
                        }
                        notes.append(
                            "[Fe/H] from Paunzen+2015 Stromgren via Casagrande+2011 eq. 3"
                        )

    # Last-resort: wide Gaussian [Fe/H] prior
    if key("feh") not in yaml_data:
        yaml_data[key("feh")] = {"initval": 0.0, "mu": 0.0, "sigma": 1.0}
        notes.append("[Fe/H]: no value found; using wide prior N(0, 1)")

    # --- 8. Optional: Tycho-2 BT/VT (disabled by default) ---------------------
    # BT/VT of the *matched* Tycho star, carried into section 9 as the APASS
    # dedup reference. Reading row 0 instead would compare the target's APASS
    # photometry against an unrelated star whenever the cone holds more than
    # one Tycho source. The references are the matched star's catalog values
    # whether or not the rows made it into the SED.
    dedup_refs = {"BT": float("nan"), "VT": float("nan")}
    qtyc2, t_row, _ = _query_catalog(CATALOGS["tycho"], tic_ra, tic_dec, dist)
    if t_row >= 0:
        dedup_refs["BT"] = _get(qtyc2, "BTmag", t_row)
        dedup_refs["VT"] = _get(qtyc2, "VTmag", t_row)
        _add_catalog_bands(
            sed_entries,
            CATALOGS["tycho"],
            qtyc2,
            t_row,
            enabled=is_live(CATALOGS["tycho"]),
        )

    # --- 9. Optional: UCAC4 / APASS DR6 (disabled by default) -----------------
    qucac, u_row, _ = _query_catalog(CATALOGS["ucac"], tic_ra, tic_dec, dist)
    if u_row >= 0:
        _add_catalog_bands(
            sed_entries,
            CATALOGS["ucac"],
            qucac,
            u_row,
            enabled=is_live(CATALOGS["ucac"]),
            refs=dedup_refs,
        )

    # --- 10. Optional: Paunzen+2015 Stromgren photometry for SED --------------
    # A second query of the section-7 catalog: that one prefers the Tycho ID
    # cross-match (it is after a [Fe/H] for this star), while the photometry
    # here takes the nearest source in the cone.
    qpz_full, pz_row, _ = _query_catalog(
        CATALOGS["stromgren"], tic_ra, tic_dec, dist
    )
    if pz_row >= 0:
        vmag = _get(qpz_full, "Vmag", pz_row)
        evmag = _get(qpz_full, "e_Vmag", pz_row)
        by = _get(qpz_full, "b-y", pz_row)
        eby = _get(qpz_full, "e_b-y", pz_row)
        m1 = _get(qpz_full, "m1", pz_row)
        em1 = _get(qpz_full, "e_m1", pz_row)
        c1 = _get(qpz_full, "c1", pz_row)
        ec1 = _get(qpz_full, "e_c1", pz_row)
        if (
            np.isfinite(vmag)
            and np.isfinite(by)
            and np.isfinite(m1)
            and np.isfinite(c1)
            # A finite uncertainty is a PRECONDITION for inclusion, exactly
            # as it is for every other photometry block in this file.  The
            # old `evmag if np.isfinite(evmag) else 0.01` turned an ABSENT
            # uncertainty into an implausibly tight one (0.01 mag is near
            # the best achievable in Stromgren V), which then over-weighted
            # this photometry in the SED likelihood and biased Teff, radius
            # and Av.
            and np.isfinite(evmag)
            and np.isfinite(eby)
            and np.isfinite(em1)
            and np.isfinite(ec1)
        ):
            u_m, su, v_m, sv, b_m, sb, y_m, sy = strom_conv(
                vmag,
                max(0.01, evmag),
                by,
                max(0.02, eby),
                m1,
                max(0.02, em1),
                c1,
                max(0.02, ec1),
            )
            for svo, mag, err in (
                ("Generic/Stromgren.u", u_m, su),
                ("Generic/Stromgren.v", v_m, sv),
                ("Generic/Stromgren.b", b_m, sb),
                ("Generic/Stromgren.y", y_m, sy),
            ):
                _add_band(
                    sed_entries,
                    svo,
                    mag,
                    err,
                    _STROMGREN_FLOOR,
                    enabled=is_live(CATALOGS["stromgren"]),
                )

    # --- 11. Optional: Mermilliod+1994 UBV (disabled by default) --------------
    qmerm, me_row, _ = _query_catalog(
        CATALOGS["mermilliod"], tic_ra, tic_dec, dist
    )
    if me_row >= 0:
        V_m = _get(qmerm, "Vmag", me_row)
        eV_m = _get(qmerm, "e_Vmag", me_row)
        BV_m = _get(qmerm, "B-V", me_row)
        eBV = _get(qmerm, "e_B-V", me_row)
        UB_m = _get(qmerm, "U-B", me_row)
        eUB = _get(qmerm, "e_U-B", me_row)
        if np.isfinite(V_m) and np.isfinite(eV_m) and eV_m < 1:
            B_v = BV_m + V_m
            eB_v = (
                math.sqrt(eBV**2 + eV_m**2)
                if np.isfinite(eBV)
                else float("nan")
            )
            U_v = UB_m + B_v
            eU_v = (
                math.sqrt(eUB**2 + eB_v**2)
                if (np.isfinite(eUB) and np.isfinite(eB_v))
                else float("nan")
            )
            for svo, mag, err in (
                ("Generic/Bessell.U", U_v, eU_v),
                ("Generic/Bessell.B", B_v, eB_v),
                ("Generic/Bessell.V", V_m, eV_m),
            ):
                _add_band(
                    sed_entries,
                    svo,
                    mag,
                    err,
                    _MERMILLIOD_FLOOR,
                    enabled=is_live(CATALOGS["mermilliod"]),
                    max_err=1.0,
                )

    # --- 12. Optional: GALEX DR5 (disabled by default; UV models unreliable) --
    qgalex, ga_row, _ = _query_catalog(
        CATALOGS["galex"], tic_ra, tic_dec, dist
    )
    if ga_row >= 0:
        sed_notes.append("GALEX: atmospheric models are unreliable in the UV")
        _add_catalog_bands(
            sed_entries,
            CATALOGS["galex"],
            qgalex,
            ga_row,
            enabled=is_live(CATALOGS["galex"]),
        )

    # --- 13. Write output files -----------------------------------------------
    outpath.mkdir(parents=True, exist_ok=True)

    _write_sed_yaml(sedfile, sed_entries, notes=[f"TIC {ticid}"] + sed_notes)

    if exofast:
        exofast_sedfile = sedfile.with_suffix(".sed.txt")
        with open(exofast_sedfile, "w") as f:
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

    with open(priorfile, "w") as f:
        f.write(
            f"# EXOZIPPy params for TIC {ticid}  (star instance: {star_name})\n"
        )
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
        prog="mkticsed.py",
        description="Create EXOZIPPy params YAML and SED file from TICv8.2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("ticid", help="TIC ID (numeric, or TIC####)")
    p.add_argument(
        "--star-name", default="Host", help="Star instance name in params YAML"
    )
    p.add_argument("--outpath", default=".", help="Output directory")
    p.add_argument(
        "--priorfile", default=None, help="Override params YAML path"
    )
    p.add_argument("--sedfile", default=None, help="Override SED file path")
    p.add_argument(
        "--dist",
        default=120.0,
        type=float,
        help="Cone-search radius in arcseconds",
    )
    p.add_argument(
        "--galex", action="store_true", help="Uncomment GALEX photometry"
    )
    p.add_argument(
        "--tycho", action="store_true", help="Uncomment Tycho-2 BT/VT"
    )
    p.add_argument(
        "--stromgren",
        action="store_true",
        help="Uncomment Stromgren photometry",
    )
    p.add_argument(
        "--ucac", action="store_true", help="Uncomment UCAC4/APASS photometry"
    )
    p.add_argument(
        "--merm", action="store_true", help="Uncomment Mermilliod UBV"
    )
    p.add_argument("--kepler", action="store_true", help="(reserved, no-op)")
    p.add_argument(
        "--exofast",
        action="store_true",
        help="Also write an EXOFASTv2-format text SED file (<ticid>.sed.txt)",
    )
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


if __name__ == "__main__":
    main()
