"""Epoch astrometry downloads for exozippy-getdata: Hipparcos and Gaia DR4.

The two products a joint Hipparcos + Gaia fit of one star reads
(components/astrometryinstrument/frametie.py):

* **Hipparcos Intermediate Astrometric Data** (``--hipparcos-iad``): the
  1997-consortium IAD (ESA 1997, Vol. 1 Sect. 2.8; van Leeuwen & Evans 1998)
  from the ESA Hipparcos tools service, one HTTP request per star.  Each
  record holds the partial derivatives of the abscissa with respect to the
  five astrometric parameters and the residual from the catalog's
  five-parameter reference solution (header IH3-IH7, at J1991.25):

    IA3 = da/dalpha* (= sin psi), IA4 = da/ddelta (= cos psi),
    IA5 = da/dparallax (the along-scan parallax factor),
    IA6 = da/dmu_alpha* (= dt sin psi), IA7 = da/dmu_delta (= dt cos psi),
    IA8 = residual [mas], IA9 = its error [mas], IA10 = FAST/NDAC correlation.

  The along-scan coordinate relative to the J1991.25 catalog position is
  rebuilt by adding the reference solution's parallax and proper motion back
  (``w = IA8 + plx IA5 + (pm_ra* dt) sin psi + (pm_dec dt) cos psi``) with
  the epoch ``J1991.25 + dt``, dt from whichever of IA6/IA3 and IA7/IA4 is
  better conditioned; the result is the ``mode: gaia`` file examples/HIP1349
  fits (``time[BJD_TDB] w[mas] err[mas] scan_PA[deg]``) with
  ``reference_point:`` the header position.  The reconstruction is checked:
  the two epoch estimates must agree, and the model's own Earth-ephemeris
  along-scan parallax factors must reproduce IA5 (examples/HIP1349: < 1e-3).

  Why the 1997 IAD and not van Leeuwen's 2007 re-reduction: the 1997 IAD is
  one public HTTP request per star, while the 2007 IAD ships as one bulk
  archive (the Java tool's resrec files) whose residuals are relative to the
  2007 solution -- the same reconstruction applies, but the download is a
  separate piece of work.  Brandt (2018, 2021) find a 60/40 mixture of the
  two reductions beats either; that mixture is a catalog-level construction
  with no epoch-level counterpart, so a fit uses one reduction.
  The FAST and NDAC abscissae of one orbit are correlated (IA10, typically
  +0.5) because they reduce the same photons; by default both are written
  and a fit treats them as independent (as examples/HIP1349 does), which
  over-weights the orbits observed twice.  ``--hip-merge-consortia`` instead
  writes ONE abscissa per orbit, the minimum-variance combination of the
  pair under its stated correlation (a record without a partner, or without
  a correlation, is kept as is).

  Supported solution types (IH8): 5 (five-parameter) and O (orbital; the
  residuals then still hold the orbit -- HIP1349).  The 7-/9-parameter
  (acceleration), C (component), V and X solutions carry terms this
  reconstruction does not add back, and raise.

* **Gaia DR4 epoch astrometry** (``--gaia-epoch-astrometry``): the
  ``EPOCH_ASTROMETRY`` DataLink product of the Gaia archive, written as the
  archive delivers it (a VOTable in the RAW data structure: one row per
  field-of-view transit, the per-CCD quantities as arrays), which is exactly
  what ``data_format: gaia_epoch`` reads.  DR4 is public on 2026-12-02; until
  then the archive refuses the retrieval type, and this says so and points
  at ``--gaia-prerelease``: the June 2026 pre-release of twelve sources
  (Gaia Collaboration, DPAC), whose zip holds the same RAW VOTable for all
  twelve; the target's rows are extracted by source_id (binary2 encoding,
  so the values round-trip exactly).

The Gaia source_id is ``--source-id``, else read off a ``Gaia DR3 <id>`` /
``Gaia DR4 <id>`` name, else the archive's Hipparcos-2 cross-match
(``gaiadr3.hipparcos2_best_neighbour``) for a ``HIP <n>`` name, else
a cone match of the Hipparcos catalog position propagated to J2016.0 (one
G < 13 source within 2 arcsec; the cross-match table misses some
high-proper-motion Hipparcos stars -- HIP 1349 and HIP 64426 among them),
else SIMBAD; a name none of them resolves raises, naming ``--source-id``.

Network access goes through ``http_get`` only, so tests replace it.
"""

import io
import logging
import os
import re
import urllib.parse
import urllib.request
import zipfile

import numpy as np

logger = logging.getLogger(__name__)

HIPPARCOS_IAD_URL = (
    "https://hipparcos-tools.cosmos.esa.int/cgi-bin/"
    "HIPcatalogueSearch.pl?hipiId={hip}"
)
GAIA_TAP_URL = "https://gea.esac.esa.int/tap-server/tap/sync"
GAIA_DATALINK_URL = "https://gea.esac.esa.int/data-server/data"
GAIA_DR4_RELEASE = "Gaia DR4"
GAIA_DR4_PUBLIC_DATE = "2026-12-02"
DR4_PRERELEASE_URL = (
    "https://anonftp.cosmos.esa.int/pub/GAIA_PUBLIC_DATA/Gaia_DR4/"
    "dr4-prerelease/gaia-dr4-prerelease-epoch-astrometry_2026-06-26.zip"
)
DR4_PRERELEASE_MEMBER = "GAIA_DR4_PRERELEASE_EPOCH_ASTROMETRY_RAW.xml"

# J1991.25 (TT) as a JD: the Hipparcos catalog epoch.
HIPPARCOS_EPOCH_JD = 2448349.0625
DAYS_PER_JULIAN_YEAR = 365.25
# IA6/IA3 and IA7/IA4 must give the same epoch to this many years (four
# printed decimals on partials of order one: ~1e-3 yr is rounding).
EPOCH_CONSISTENCY_YR = 0.02
# Our along-scan parallax factors vs the consortium's IA5: warned beyond
# 1e-3 (HIP1349 agrees to that), raised beyond 0.02 (a convention error).
PFAL_WARN, PFAL_RAISE = 1e-3, 0.02
SUPPORTED_SOLUTION_TYPES = ("5", "O")


def http_get(url, data=None, timeout=120):
    """GET ``url`` (POST when ``data``, a dict, is given) and return the
    body as bytes.  The one network call of this module."""
    body = None if data is None else urllib.parse.urlencode(data).encode()
    req = urllib.request.Request(
        url, data=body, headers={"User-Agent": "exozippy-getdata"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


# ----------------------------------------------------------------------
# Names
# ----------------------------------------------------------------------
def hip_number(name):
    """The HIP number of a ``HIP <n>`` name, else None."""
    m = re.fullmatch(r"\s*HIP\s*(\d+)\s*", str(name), flags=re.IGNORECASE)
    return int(m.group(1)) if m else None


def gaia_number(name):
    """The source_id of a ``Gaia DR2/DR3/DR4 <id>`` name, else None."""
    m = re.fullmatch(r"\s*Gaia\s*(?:E?DR[234])?\s*(\d{6,20})\s*", str(name))
    return int(m.group(1)) if m else None


def _simbad_ids(name):
    """Every SIMBAD identifier of ``name`` (astroquery, imported here)."""
    from astroquery.simbad import Simbad

    table = Simbad.query_objectids(name)
    if table is None or len(table) == 0:
        return []
    col = [c for c in table.colnames if c.lower() == "id"][0]
    return [str(x) for x in table[col]]


def resolve_hip(name):
    """The HIP number of ``name``: itself if it is ``HIP <n>``, else SIMBAD's
    HIP identifier.  Raises naming the remedy when neither gives one."""
    hip = hip_number(name)
    if hip is not None:
        return hip
    try:
        ids = _simbad_ids(name)
    except Exception as e:  # network / service failure: say what to do
        raise RuntimeError(
            f"could not resolve {name!r} to a HIP number: SIMBAD failed "
            f"({e!r}).  Give the star as 'HIP <n>'."
        ) from e
    for i in ids:
        hip = hip_number(i)
        if hip is not None:
            return hip
    raise ValueError(
        f"{name!r} has no HIP identifier in SIMBAD (identifiers: {ids}); "
        f"it is not a Hipparcos star, or give it as 'HIP <n>'."
    )


def gaia_tap(adql, timeout=120):
    """An ADQL query against the Gaia archive's synchronous TAP service,
    returned as an astropy Table (CSV transport)."""
    from astropy.table import Table

    body = http_get(
        GAIA_TAP_URL,
        data={
            "REQUEST": "doQuery",
            "LANG": "ADQL",
            "FORMAT": "csv",
            "QUERY": adql,
        },
        timeout=timeout,
    )
    return Table.read(io.BytesIO(body), format="ascii.csv")


# A Hipparcos star propagated to J2016.0 with its catalog proper motion
# lands within ~0.1 arcsec of its Gaia DR3 position unless it accelerates
# hard; the cone is generous, and more than one bright source in it raises.
HIP_CONE_ARCSEC = 2.0
HIP_CONE_GMAG = 13.0
J2016_JD = 2457389.0


def _cone_match_hipparcos(hip):
    """``(source_id, note)``: the one Gaia DR3 source with G < 13 within
    2 arcsec of HIP ``hip``'s catalog position propagated to J2016.0 (the
    IAD header's position and proper motion), else ``(None, why)``."""
    header, _ = parse_hipparcos_iad(fetch_hipparcos_iad(hip))
    ra, dec, pmra, pmdec = (
        float(header[k]) for k in ("IH3", "IH4", "IH6", "IH7")
    )
    dt = (J2016_JD - HIPPARCOS_EPOCH_JD) / DAYS_PER_JULIAN_YEAR
    dec16 = dec + pmdec * dt / 3.6e6
    ra16 = ra + pmra * dt / 3.6e6 / np.cos(np.radians(dec))
    t = gaia_tap(
        "SELECT source_id, phot_g_mean_mag, DISTANCE(POINT(ra, dec), "
        f"POINT({ra16:.8f}, {dec16:.8f})) * 3600 AS sep FROM "
        "gaiadr3.gaia_source WHERE 1 = CONTAINS(POINT(ra, dec), "
        f"CIRCLE({ra16:.8f}, {dec16:.8f}, {HIP_CONE_ARCSEC / 3600.0:.8f})) "
        f"AND phot_g_mean_mag < {HIP_CONE_GMAG}"
    )
    if len(t) == 1:
        logger.info(
            f"HIP {hip}: Gaia DR3 {int(t['source_id'][0])} by cone match "
            f"({float(t['sep'][0]):.3f} arcsec from the propagated "
            f"Hipparcos position, G = {float(t['phot_g_mean_mag'][0]):.2f})."
        )
        return int(t["source_id"][0]), ""
    return None, (
        f"a {HIP_CONE_ARCSEC}-arcsec cone (G < {HIP_CONE_GMAG}) around the "
        f"propagated Hipparcos position ({len(t)} sources: "
        f"{[int(x) for x in t['source_id']] if len(t) else []})"
    )


def resolve_source_id(name, source_id=None):
    """The Gaia source_id for ``name`` (see the module docstring for the
    order).  Raises naming ``--source-id`` when nothing resolves it."""
    if source_id is not None:
        return int(source_id)
    sid = gaia_number(name)
    if sid is not None:
        return sid
    tried = []
    hip = hip_number(name)
    if hip is not None:
        t = gaia_tap(
            "SELECT source_id FROM gaiadr3.hipparcos2_best_neighbour "
            f"WHERE original_ext_source_id = {hip}"
        )
        if len(t) == 1:
            return int(t["source_id"][0])
        tried.append(f"gaiadr3.hipparcos2_best_neighbour ({len(t)} rows)")
        sid, note = _cone_match_hipparcos(hip)
        if sid is not None:
            return sid
        tried.append(note)
    try:
        for i in _simbad_ids(name):
            m = re.fullmatch(r"Gaia DR3 (\d+)", i)
            if m:
                return int(m.group(1))
        tried.append("SIMBAD (no Gaia DR3 identifier)")
    except Exception as e:  # network / service failure: say what to do
        tried.append(f"SIMBAD ({e!r})")
    raise ValueError(
        f"could not resolve {name!r} to a Gaia source_id; tried "
        f"{'; '.join(tried)}.  Give it with --source-id."
    )


# ----------------------------------------------------------------------
# Hipparcos IAD
# ----------------------------------------------------------------------
def fetch_hipparcos_iad(hip, timeout=60):
    """The ESA Hipparcos tools page holding HIP ``hip``'s 1997 IAD (text)."""
    body = http_get(HIPPARCOS_IAD_URL.format(hip=int(hip)), timeout=timeout)
    return body.decode(errors="replace")


_IAD_RECORD = re.compile(
    r"^\s*(\d+)\|([FN])\|"
    r"([-\d. ]+)\|([-\d. ]+)\|([-\d. ]+)\|([-\d. ]+)\|"
    r"([-\d. ]+)\|([-\d. ]+)\|([-\d. ]+)\|([-\d. ]*)"
)
_IAD_HEADER = re.compile(r"^\s*IH(\d+)\s*:\s*(\S+)")


def parse_hipparcos_iad(body):
    """``(header, records)`` from a Hipparcos tools IAD page.

    ``header`` maps ``IH<k>`` to its value (string); ``records`` is a dict
    of arrays: ``orbit`` (int), ``consortium`` ("F"/"N"), ``IA3``-``IA9``
    and ``IA10`` (NaN where blank).  Raises when the page holds no IAD or
    its record count disagrees with IH9.
    """
    header, rows = {}, []
    for line in body.splitlines():
        m = _IAD_RECORD.match(line)
        if m:
            g = m.groups()
            corr = g[9].strip()
            rows.append(
                (
                    int(g[0]),
                    g[1],
                    *[float(v) for v in g[2:9]],
                    float(corr) if corr else np.nan,
                )
            )
            continue
        h = _IAD_HEADER.match(line)
        if h and not rows:
            header[f"IH{h.group(1)}"] = h.group(2)
    if not rows or "IH1" not in header:
        raise ValueError(
            "the page holds no Hipparcos Intermediate Astrometric Data "
            "(no IH header or no abscissa records); is the HIP number right?"
        )
    n_expect = int(header.get("IH9", len(rows)))
    if n_expect != len(rows):
        raise ValueError(
            f"HIP {header['IH1']}: IH9 says {n_expect} abscissa records, the "
            f"page holds {len(rows)}."
        )
    cols = list(zip(*rows))
    rec = {
        "orbit": np.array(cols[0], dtype=int),
        "consortium": np.array(cols[1]),
    }
    for k, name in enumerate(
        ("IA3", "IA4", "IA5", "IA6", "IA7", "IA8", "IA9", "IA10")
    ):
        rec[name] = np.array(cols[2 + k], dtype=float)
    return header, rec


def merge_consortia(rec):
    """One abscissa per orbit: the FAST and NDAC pair of an orbit combined
    with the minimum-variance weights under their correlation IA10.

    For errors s_F, s_N and correlation r the weights are
    (s_N^2 - r s_F s_N, s_F^2 - r s_F s_N)/(s_F^2 + s_N^2 - 2 r s_F s_N) and
    the combined variance s_F^2 s_N^2 (1 - r^2)/(s_F^2 + s_N^2 - 2 r s_F
    s_N).  The partials (IA3-IA7) are combined with the same weights (the
    two consortia's differ in the third decimal).  A record without a
    partner or a correlation is kept as it is.
    """
    out = {k: [] for k in rec}
    orbits = rec["orbit"]
    for o in dict.fromkeys(orbits.tolist()):  # file order
        idx = np.flatnonzero(orbits == o)
        pair = len(idx) == 2 and set(rec["consortium"][idx]) == {"F", "N"}
        r = rec["IA10"][idx[0]] if pair else np.nan
        if not pair or not np.isfinite(r):
            for i in idx:
                for k in rec:
                    out[k].append(rec[k][i])
            continue
        i, j = idx
        sf, sn = rec["IA9"][i], rec["IA9"][j]
        den = sf**2 + sn**2 - 2 * r * sf * sn
        wf = (sn**2 - r * sf * sn) / den
        wn = (sf**2 - r * sf * sn) / den
        for k in ("IA3", "IA4", "IA5", "IA6", "IA7", "IA8"):
            out[k].append(wf * rec[k][i] + wn * rec[k][j])
        out["IA9"].append(np.sqrt(sf**2 * sn**2 * (1 - r**2) / den))
        out["IA10"].append(np.nan)
        out["orbit"].append(o)
        out["consortium"].append("M")
    return {k: np.asarray(v) for k, v in out.items()}


def reconstruct_hipparcos_abscissae(header, rec):
    """The ``mode: gaia`` columns from parsed IAD: a dict with ``time``
    (JD, J1991.25 + dt), ``w``/``err`` [mas], ``psi`` [deg], ``pfal`` (IA5,
    for the parallax-factor check), and ``ra0``/``dec0`` [deg], the
    reference point w is measured from.  Raises on an unsupported solution
    type or epochs that IA6/IA3 and IA7/IA4 disagree on."""
    hip = header["IH1"]
    soln = header.get("IH8", "?")
    if soln not in SUPPORTED_SOLUTION_TYPES:
        raise ValueError(
            f"HIP {hip}: Hipparcos solution type {soln!r} (IH8).  Only "
            f"{SUPPORTED_SOLUTION_TYPES} are reconstructed: a 7-/9-parameter, "
            f"C, V or X solution carries terms (acceleration, components, "
            f"photocenter variability) whose partials the IAD records do "
            f"not hold, so w cannot be rebuilt by adding the five-parameter "
            f"model back."
        )
    ra0, dec0 = float(header["IH3"]), float(header["IH4"])
    plx, pmra, pmdec = (float(header[k]) for k in ("IH5", "IH6", "IH7"))
    s, c = rec["IA3"], rec["IA4"]
    with np.errstate(divide="ignore", invalid="ignore"):
        dt_a = np.where(np.abs(s) > 1e-3, rec["IA6"] / s, np.nan)
        dt_d = np.where(np.abs(c) > 1e-3, rec["IA7"] / c, np.nan)
    dt = np.where(np.abs(s) > np.abs(c), dt_a, dt_d)
    both = np.isfinite(dt_a) & np.isfinite(dt_d)
    if both.any():
        worst = float(np.max(np.abs(dt_a[both] - dt_d[both])))
        if worst > EPOCH_CONSISTENCY_YR:
            raise ValueError(
                f"HIP {hip}: the epochs from IA6/IA3 and IA7/IA4 disagree "
                f"by up to {worst:.4f} yr (> {EPOCH_CONSISTENCY_YR}); the "
                f"records were not parsed as the IAD lays them out."
            )
    # The proper-motion terms use the SAME dt the epoch is built from, so the
    # model's pm * (t - J1991.25) reproduces them exactly; IA6/IA7 printed
    # to four decimals would add a rounding the epoch does not share.
    w = rec["IA8"] + plx * rec["IA5"] + (pmra * dt) * s + (pmdec * dt) * c
    return {
        "time": HIPPARCOS_EPOCH_JD + dt * DAYS_PER_JULIAN_YEAR,
        "w": w,
        "err": rec["IA9"],
        "psi": np.degrees(np.arctan2(s, c)),
        "pfal": rec["IA5"],
        "ra0": ra0,
        "dec0": dec0,
        "hip": int(hip),
        "solution": soln,
    }


def check_hipparcos_parallax_factors(ab):
    """Max |ours - IA5| of the along-scan parallax factor, ours from the
    Earth ephemeris the model uses (Hipparcos was within ~3e-4 AU of the
    geocenter).  Warns beyond PFAL_WARN, raises beyond PFAL_RAISE."""
    from exozippy.ephemeris import get_observer_position
    from exozippy.skyframe import parallax_factors

    ra, dec = np.radians(ab["ra0"]), np.radians(ab["dec0"])
    P_E, P_N = parallax_factors(
        get_observer_position(ab["time"], "earth"), ra, dec
    )
    psi = np.radians(ab["psi"])
    diff = float(
        np.max(np.abs(P_E * np.sin(psi) + P_N * np.cos(psi) - ab["pfal"]))
    )
    if diff > PFAL_RAISE:
        raise ValueError(
            f"HIP {ab['hip']}: along-scan parallax factors disagree with the "
            f"IAD's IA5 by {diff:.4f} (> {PFAL_RAISE}): a convention or "
            f"epoch error in the reconstruction."
        )
    if diff > PFAL_WARN:
        logger.warning(
            f"HIP {ab['hip']}: along-scan parallax factors differ from IA5 "
            f"by up to {diff:.4f} (> {PFAL_WARN})."
        )
    return diff


def write_hipparcos_astrom(path, ab, note=""):
    """The ``mode: gaia`` file: ``BJD_TDB w[mas] err[mas] scan_PA[deg]``,
    in the IAD's record order (the reader sorts by time), with the reference
    point in the header comment (for the reader of the file; the config
    states it with reference_point:)."""
    order = np.arange(len(ab["time"]))
    np.savetxt(
        path,
        np.column_stack(
            [
                ab["time"][order],
                ab["w"][order],
                ab["err"][order],
                ab["psi"][order],
            ]
        ),
        header=(
            f"Hipparcos epoch astrometry of HIP {ab['hip']}, reconstructed "
            f"from the 1997 IAD (exozippy-getdata --hipparcos-iad){note}\n"
            f"w measured from the J1991.25 catalog position "
            f"reference_point: [{ab['ra0']:.8f}, {ab['dec0']:.8f}]\n"
            "BJD_TDB  w[mas]  err[mas]  scan_PA[deg]"
        ),
        fmt="%.6f %.4f %.3f %.6f",
    )


def get_hipparcos_iad(
    name, path=".", merge=False, cache=None, overwrite=False
):
    """Download (or read ``cache``), reconstruct, check and write one star's
    Hipparcos epoch astrometry.  Returns ``(astrom_path, abscissae)``."""
    hip = resolve_hip(name)
    tag = f"HIP{hip}"
    cache = cache or os.path.join(path, f"{tag}.iad.txt")
    if os.path.exists(cache) and not overwrite:
        body = open(cache).read()
    else:
        body = fetch_hipparcos_iad(hip)
        with open(cache, "w") as f:
            f.write(body)
    header, rec = parse_hipparcos_iad(body)
    if merge:
        rec = merge_consortia(rec)
    ab = reconstruct_hipparcos_abscissae(header, rec)
    diff = check_hipparcos_parallax_factors(ab)
    out = os.path.join(path, f"{tag}.Hipparcos.astrom")
    write_hipparcos_astrom(
        out, ab, note=" -- FAST/NDAC merged" if merge else ""
    )
    print(
        f"Wrote {out}: {len(ab['time'])} abscissae of HIP {hip} (solution "
        f"type {ab['solution']}; parallax factors match IA5 to {diff:.1e}).\n"
        f"astrometryinstrument entry:\n"
        f"  - name: Hipparcos\n"
        f"    file: {os.path.basename(out)}\n"
        f"    mode: gaia\n"
        f"    observer_location: earth\n"
        f"    reference_point: [{ab['ra0']:.8f}, {ab['dec0']:.8f}]\n"
        f"    fithipframe: true   # when fit together with Gaia epoch astrometry"
    )
    return out, ab


# ----------------------------------------------------------------------
# Gaia DR4 epoch astrometry
# ----------------------------------------------------------------------
class GaiaReleaseNotAvailable(RuntimeError):
    """The archive does not (yet) serve the requested product."""


def fetch_gaia_epoch_astrometry(
    source_id, release=GAIA_DR4_RELEASE, timeout=300
):
    """The RAW epoch-astrometry VOTable of ``source_id`` from the Gaia
    archive's DataLink service, as bytes.  Raises GaiaReleaseNotAvailable
    when the archive does not serve EPOCH_ASTROMETRY for ``release``."""
    import urllib.error

    params = {
        "ID": str(int(source_id)),
        "RELEASE": release,
        "RETRIEVAL_TYPE": "EPOCH_ASTROMETRY",
        "DATA_STRUCTURE": "RAW",
        "FORMAT": "votable",
        "USE_ZIP_ALWAYS": "true",
        "VALID_DATA": "false",
    }
    try:
        body = http_get(GAIA_DATALINK_URL, data=params, timeout=timeout)
    except urllib.error.HTTPError as e:
        text = " ".join(
            re.sub(r"<[^>]+>", " ", e.read().decode(errors="replace")).split()
        )
        if any(
            k in text
            for k in (
                "Unknown retrieval type",
                "Unknown release",
                "Invalid release",
            )
        ):
            said = (
                text.split("Message:")[-1].split("Cannot process")[0].strip()
            )
            raise GaiaReleaseNotAvailable(
                f"the Gaia archive does not serve EPOCH_ASTROMETRY for "
                f"{release!r} (it answered: {said[:200]!r}).  "
                f"{GAIA_DR4_RELEASE} epoch astrometry is public from "
                f"{GAIA_DR4_PUBLIC_DATE}; until then only the twelve sources "
                f"of the June 2026 pre-release exist: use --gaia-prerelease."
            ) from e
        raise
    with zipfile.ZipFile(io.BytesIO(body)) as z:
        xml = [n for n in z.namelist() if n.endswith(".xml")]
        if len(xml) != 1:
            raise ValueError(
                f"the DataLink bundle for {source_id} holds {z.namelist()}; "
                f"expected one VOTable."
            )
        return z.read(xml[0])


def _prerelease_votable(src=None, timeout=300):
    """The pre-release RAW VOTable (bytes) from ``src`` -- the zip, the
    extracted xml, or None to download the zip."""
    if src is None:
        blob = http_get(DR4_PRERELEASE_URL, timeout=timeout)
    elif str(src).lower().endswith(".xml"):
        return open(src, "rb").read()
    else:
        blob = open(src, "rb").read()
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        if DR4_PRERELEASE_MEMBER not in z.namelist():
            raise ValueError(
                f"the pre-release zip holds {z.namelist()}, not "
                f"{DR4_PRERELEASE_MEMBER}."
            )
        return z.read(DR4_PRERELEASE_MEMBER)


def select_source_votable(xml_bytes, source_id, out_path):
    """Write the rows of ``source_id`` from a RAW epoch-astrometry VOTable
    to ``out_path``, keeping its TIMESYS, fields and units (binary2 cells,
    so every value round-trips exactly).  Raises when the source is absent."""
    from astropy.io.votable import parse

    vot = parse(io.BytesIO(xml_bytes))
    tab = vot.get_first_table()
    ids = np.asarray(tab.array["source_id"], dtype=np.int64)
    keep = ids == int(source_id)
    if not keep.any():
        raise ValueError(
            f"source_id {source_id} is not in the epoch-astrometry table; it "
            f"holds {sorted(set(ids.tolist()))}."
        )
    tab.array = tab.array[keep]
    vot.to_xml(str(out_path), tabledata_format="binary2")
    return int(keep.sum())


def get_gaia_epoch_astrometry(
    name, path=".", source_id=None, prerelease=False, prerelease_file=None
):
    """Fetch and write one source's DR4 epoch astrometry for ``data_format:
    gaia_epoch``.  Returns the written path."""
    sid = resolve_source_id(name, source_id)
    out = os.path.join(
        path,
        f"Gaia{sid}."
        + ("DR4prerelease" if prerelease else "DR4")
        + ".epoch_astrometry.xml",
    )
    if prerelease:
        xml = _prerelease_votable(prerelease_file)
    else:
        xml = fetch_gaia_epoch_astrometry(sid)
    n = select_source_votable(xml, sid, out)
    print(
        f"Wrote {out}: {n} field-of-view transits of Gaia source {sid}.\n"
        f"astrometryinstrument entry:\n"
        f"  - name: GaiaDR4\n"
        f"    file: {os.path.basename(out)}\n"
        f"    mode: gaia\n"
        f"    data_format: gaia_epoch\n"
        f"    source_id: {sid}"
    )
    return out
