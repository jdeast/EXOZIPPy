#!/usr/bin/env python3
"""
mkdust.py

Query the extinction maps at a sky position and write the dust-profile
prior a fit consumes, as a commented params-file fragment the user edits.
The dust analogue of mkticsed: every map that answers is written, and all
but ONE are commented out, so switching maps is uncommenting a line.

What it feeds
-------------
The SED's sight-line dust profile (components/sed/sed.md, "The sight-line
dust profile") samples ONE total, ``sed.dust_anchor``: the extinction at the
profile's anchor, in A_V unless the system config's ``sed:`` block names a
``dust_anchor_band``.  A 2D red-clump map states the extinction TO THE
CLUMP, at the bar, which is where the anchor sits when the fit has a red
clump or a galactic model.  So a map value is a ``{mu, sigma, citation}``
prior on ``sed.dust_anchor`` -- no new config channel.

The default (the live line)
---------------------------
Following Nataf (2026 Sagan workshop): Surot et al. (2020)'s VVV E(J-Ks)
map, else Zelakiewicz et al. (2025)'s UKIRT A_Ks map, converted to A_V with
the Nataf et al. (2013) bulge-mean ratios (A_I = 1.217 E(V-I),
E(J-Ks)/E(V-I) = 0.3433).  Why converted by hand rather than written in Ks
with ``dust_anchor_band``: the fit's own law (CCM89 + O'Donnell, tabulated
at R_V = 3.1 or fitted) has A_V/A_Ks ~ 8.4 at R_V = 3.1, where the bulge
measures 13-15, so a Ks anchor would put ~40% too little dust in the optical
bands most microlensing data are in.  The Ks-native lines are written too
(commented); they become the better choice once the NIR slope of the law is
fitted.  The numbers behind each conversion and width are in
docs/dust_maps.md.

Maps that cannot be queried programmatically are listed in the output as
skipped, with the reason.  None is ever approximated.

Usage:
    exozippy-mkdust <ra> <dec> [--name NAME] [--outpath DIR]

    ra/dec in decimal degrees, or sexagesimal (ra in hours) with colons:
    exozippy-mkdust 17:53:48.01 -29:15:47.48 --name KB242087
"""

from __future__ import annotations

import argparse
import logging
import math
import re
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.coordinates import Angle, SkyCoord

from ..components.sed.bc_grid import DEFAULT_MODEL_ROOT
from . import zenodo, zenodo_assets

logger = logging.getLogger(__name__)

# --- the conversions (docs/dust_maps.md) --------------------------------------

# Nataf et al. (2013) bulge means: A_I/E(V-I) (eq. 29 at the mean R_JKVI) and
# R_JKVI = E(J-Ks)/E(V-I) (Sec. 8.1).
NATAF13_AI_EVI = 1.217
NATAF13_RJKVI = 0.3433
# A_V / E(J-Ks) = (A_I + E(V-I)) / E(J-Ks) = (1.217 + 1) / 0.3433 = 6.458.
AV_PER_EJK = (NATAF13_AI_EVI + 1.0) / NATAF13_RJKVI
# Sight-line scatter of that ratio: the 1-sigma fractional spread of
# (A_I + E(V-I)) / (R_JKVI E(V-I)) over the reliable Nataf+2013 Table 3
# sight lines with E(V-I) > 0.5 (mean 6.465, sd/mean 0.070).
AV_PER_EJK_FRAC_SIGMA = 0.07
# Surot et al. (2020) eq. 1, their measured VVV bulge reddening vector:
# A_Ks = 0.422167 E(J-Ks).  Used (1) to write Surot's map in Ks and (2) to
# take Zelakiewicz's A_Ks to E(J-Ks) before the Nataf+2013 ratio.
SUROT20_AKS_PER_EJK = 0.422167
# Fractional uncertainty of that A_Ks -> E(J-Ks) step for a map measured in
# Ks: published bulge A_Ks/E(J-Ks) run 0.42-0.49 (Surot+2020, Alonso-Garcia
# +2017 0.428, Nagatomo+2019 0.423, Wang & Chen 2019 0.49).
AKS_TO_EJK_FRAC_SIGMA = 0.10
# Schlafly & Finkbeiner (2011) recalibrated E(B-V), to A_V at R_V = 3.1.
SF11_RV = 3.1
# Nataf et al. (2013) eq. 20: field-scale extinction granularity, ~9% of the
# total.  Used as the width of a Nataf+2013 Table 3 sight line that lists no
# differential reddening (S_E(V-I) = 0).
NATAF13_GRANULARITY = 0.09

# The anchor's band for each map's NATIVE-band line (sed.dust_anchor_band).
VVV_KS = "Paranal/VISTA.Ks"
# UKIRT WFCAM K has no BC column; 2MASS Ks is the nearest band that does.
UKIRT_KS = "2MASS/2MASS.Ks"
COUSINS_I = "Generic/Cousins.I"

# --- where the downloaded map files live --------------------------------------

DUSTMAP_DIR = DEFAULT_MODEL_ROOT / "dustmaps"
ZELAKIEWICZ_RECORD = "zelakiewicz2025_ukirt_extinction"
ZELAKIEWICZ_FILE = "ukirt_map.csv"
# Nataf et al. (2013) Table 3, from the OGLE archive (not on Zenodo).  Size
# and md5 pin the 2013-05-31 version (the ~19,000-point grid).
NATAF13_TABLE3 = {
    "Table3.dat": {
        "url": "https://www.astrouw.edu.pl/ogle/ogle3/ext/blg/Table3.dat",
        "size": 3227029,
        "md5": "9fbbcb21fc019f947fc0335949f32c5f",
    }
}

SUROT20_VIZIER = "J/A+A/644/A140"
# The Surot nodes reach < 10" resolution near the plane; a 1' cone always
# holds one inside the map.
SUROT20_RADIUS_ARCMIN = 1.0
# Zelakiewicz pixels are 2' x 2': a target inside the map is within the
# half-diagonal (1.41') of one pixel center.
ZELAKIEWICZ_MATCH_ARCMIN = 1.5
# The OGLE calculator's own match radius for Table 3.
NATAF13_MATCH_ARCSEC = 600.0


class MapFormatError(RuntimeError):
    """A map service answered, but not in the format this tool reads."""


class MapFetchError(RuntimeError):
    """A map file could not be downloaded (zenodo.fetch_assets gave up)."""


def _fetch(assets: dict) -> None:
    """``zenodo.fetch_assets`` into DUSTMAP_DIR, its give-up as MapFetchError."""
    try:
        zenodo.fetch_assets(assets, DUSTMAP_DIR)
    except RuntimeError as e:
        raise MapFetchError(str(e)) from e


@dataclass
class Prior:
    """One candidate ``sed.dust_anchor`` prior."""

    mu: float
    sigma: float
    citation: tuple[str, ...]
    band: str | None = None  # None: A_V (no dust_anchor_band)
    note: str = ""


@dataclass
class MapReading:
    """What one map said at the target (or why it said nothing)."""

    key: str
    title: str
    status: str  # "ok", "outside", "failed", "skipped"
    detail: str = ""
    av: Prior | None = None
    native: list[Prior] = field(default_factory=list)


# --- coordinates --------------------------------------------------------------


def parse_coord(ra: str, dec: str) -> SkyCoord:
    """ICRS position from decimal degrees, or sexagesimal with ra in hours."""
    if ":" in ra or ":" in dec:
        return SkyCoord(ra, dec, unit=(u.hourangle, u.deg), frame="icrs")
    return SkyCoord(float(ra), float(dec), unit=u.deg, frame="icrs")


def _lb(coord: SkyCoord) -> tuple[float, float]:
    """Galactic (l, b) in degrees, l wrapped to (-180, 180]."""
    g = coord.galactic
    return float(g.l.wrap_at(180 * u.deg).deg), float(g.b.deg)


def _sep_arcmin(l1, b1, l2, b2):
    """Angular separation in arcmin (haversine), vectorized over (l2, b2)."""
    l1, b1, l2, b2 = map(np.radians, (l1, b1, l2, b2))
    h = (
        np.sin((b2 - b1) / 2) ** 2
        + np.cos(b1) * np.cos(b2) * np.sin((l2 - l1) / 2) ** 2
    )
    return np.degrees(2 * np.arcsin(np.sqrt(h))) * 60.0


# --- the maps -----------------------------------------------------------------


def _av_from_ejk(ejk, e_ejk, citation, note):
    """A_V from E(J-Ks) through the Nataf+2013 bulge-mean ratios."""
    av = AV_PER_EJK * ejk
    sigma = av * math.hypot(e_ejk / ejk, AV_PER_EJK_FRAC_SIGMA)
    return Prior(av, sigma, citation, note=note)


def _vizier_cone(coord, catalog, radius_arcmin):
    """Rows of `catalog` within `radius_arcmin`, nearest first (``_r``)."""
    from astroquery.vizier import Vizier

    tables = Vizier(columns=["**", "_r"], row_limit=5000).query_region(
        coord, radius=radius_arcmin * u.arcmin, catalog=catalog
    )
    if len(tables) == 0:
        return None
    return tables[0]


def read_surot2020(coord: SkyCoord) -> MapReading:
    """Surot et al. (2020) VVV E(J-Ks), nearest node (VizieR cone)."""
    m = MapReading("surot2020", "Surot et al. (2020), VVV E(J-Ks)", "ok")
    table = _vizier_cone(coord, SUROT20_VIZIER, SUROT20_RADIUS_ARCMIN)
    if table is None or len(table) == 0:
        m.status = "outside"
        m.detail = (
            f"no map node within {SUROT20_RADIUS_ARCMIN}' (outside the VVV "
            f"bulge area, |l| < 10, -10 < b < 5)"
        )
        return m
    for col in ("E(J-Ks)", "e_E(J-Ks)", "_r"):
        if col not in table.colnames:
            raise MapFormatError(
                f"VizieR {SUROT20_VIZIER} has no column '{col}' (columns: "
                f"{', '.join(table.colnames)}); mkdust must be updated to "
                f"the catalog's current format."
            )
    row = table[int(np.argmin(np.asarray(table["_r"], dtype=float)))]
    ejk = float(row["E(J-Ks)"])
    e_ejk = float(row["e_E(J-Ks)"])
    r_arcsec = float(row["_r"]) * 60.0  # VizieR _r is in arcmin
    m.detail = (
        f"E(J-Ks) = {ejk:.3f} +/- {e_ejk:.3f} (error of the node mean) at "
        f'the nearest node, {r_arcsec:.1f}" away'
    )
    m.av = _av_from_ejk(
        ejk,
        e_ejk,
        ("Surot:2020", "Nataf:2013"),
        "A_V = 6.458 E(J-Ks) (Nataf+2013 means)",
    )
    aks = SUROT20_AKS_PER_EJK * ejk
    m.native.append(
        Prior(
            aks,
            SUROT20_AKS_PER_EJK * e_ejk,
            ("Surot:2020",),
            band=VVV_KS,
            note="A_Ks = 0.422 E(J-Ks), Surot+2020 eq. 1",
        )
    )
    return m


def zelakiewicz_path() -> Path:
    """The Zelakiewicz+2025 map, fetched (md5-checked) on first use."""
    fetch_all()
    return DUSTMAP_DIR / ZELAKIEWICZ_FILE


def fetch_all() -> list[Path]:
    """Fetch the Zenodo-hosted map into DUSTMAP_DIR (the record's owner hook).

    Zenodo's ``fetch_every_record`` skips it (an on-demand record): only this
    tool reads it.
    """
    _fetch(zenodo_assets.assets(ZELAKIEWICZ_RECORD))
    return [DUSTMAP_DIR / ZELAKIEWICZ_FILE]


def read_zelakiewicz2025(coord: SkyCoord, path: Path) -> MapReading:
    """Zelakiewicz et al. (2025) UKIRT A_Ks, the 2' x 2' pixel holding it."""
    m = MapReading(
        "zelakiewicz2025", "Zelakiewicz et al. (2025), UKIRT A_Ks", "ok"
    )
    data = np.genfromtxt(path, delimiter=",", names=True)
    expected = ("l", "b", "aks", "aks_err_up", "aks_err_low", "Flag")
    if data.dtype.names is None or not set(expected) <= set(data.dtype.names):
        raise MapFormatError(
            f"{path} lacks the columns {expected} (has {data.dtype.names})."
        )
    l0, b0 = _lb(coord)
    sep = _sep_arcmin(l0, b0, data["l"], data["b"])
    i = int(np.argmin(sep))
    if sep[i] > ZELAKIEWICZ_MATCH_ARCMIN:
        m.status = "outside"
        m.detail = (
            "outside the map (-2.15 < l < 2.71, -2.69 < b < 2.03, and the "
            "UKIRT 2017 fields within it)"
        )
        return m
    row = data[i]
    aks = float(row["aks"])
    e_aks = 0.5 * (float(row["aks_err_up"]) + float(row["aks_err_low"]))
    flagged = int(row["Flag"]) == 1
    m.detail = (
        f"A_Ks = {aks:.3f} +{row['aks_err_up']:.3f}/-{row['aks_err_low']:.3f} "
        f'in the pixel centered {sep[i] * 60:.0f}" away'
    )
    if flagged:
        m.detail += (
            "; FLAGGED by the authors as a LOWER LIMIT (too few clump stars "
            "detected in H): do not use it as a prior"
        )
    ejk = aks / SUROT20_AKS_PER_EJK
    av = AV_PER_EJK * ejk
    sigma = av * math.sqrt(
        (e_aks / aks) ** 2
        + AKS_TO_EJK_FRAC_SIGMA**2
        + AV_PER_EJK_FRAC_SIGMA**2
    )
    m.av = Prior(
        av,
        sigma,
        ("Zelakiewicz:2025", "Surot:2020", "Nataf:2013"),
        note="A_V = 15.30 A_Ks (Surot+2020 eq. 1, then Nataf+2013 means)",
    )
    m.native.append(
        Prior(
            aks,
            e_aks,
            ("Zelakiewicz:2025",),
            band=UKIRT_KS,
            note="UKIRT K written as 2MASS Ks (no UKIRT K BC column)",
        )
    )
    if flagged:
        m.status = "lower_limit"
    return m


def nataf2013_path() -> Path:
    """Nataf+2013 Table 3, fetched (size- and md5-checked) on first use."""
    _fetch(NATAF13_TABLE3)
    return DUSTMAP_DIR / "Table3.dat"


def read_nataf2013(coord: SkyCoord, path: Path) -> MapReading:
    """Nataf et al. (2013) OGLE-III A_I, E(V-I): nearest reliable point."""
    m = MapReading(
        "nataf2013", "Nataf et al. (2013), OGLE-III A_I and E(V-I)", "ok"
    )
    data = np.loadtxt(path, comments="#")
    if data.ndim != 2 or data.shape[1] != 11:
        raise MapFormatError(
            f"{path} is not the 11-column Nataf+2013 Table 3 "
            f"(shape {data.shape})."
        )
    data = data[data[:, 10] == 0]  # QualityFlag 0: a reliable sight line
    grid = SkyCoord(data[:, 2] * 15.0, data[:, 3], unit=u.deg)
    sep = coord.separation(grid).arcsec
    i = int(np.argmin(sep))
    if sep[i] > NATAF13_MATCH_ARCSEC:
        m.status = "outside"
        m.detail = (
            f"no reliable OGLE-III sight line within "
            f'{NATAF13_MATCH_ARCSEC:.0f}"'
        )
        return m
    _, _, _, _, ai, evi, s_evi, _, _, rjkvi, _ = data[i]
    av = ai + evi
    m.detail = (
        f"A_I = {ai:.3f}, E(V-I) = {evi:.3f}, differential reddening "
        f"S_E(V-I) = {s_evi:.3f}, R_JKVI = {rjkvi:.3f}, at the grid point "
        f'{sep[i]:.0f}" away'
    )
    if s_evi > 0:
        sigma_av = (ai / evi + 1.0) * s_evi
        sigma_ai = (ai / evi) * s_evi
        note = "sigma from S_E(V-I)"
    else:
        sigma_av = NATAF13_GRANULARITY * av
        sigma_ai = NATAF13_GRANULARITY * ai
        note = "no S_E(V-I) listed: sigma = 9% (Nataf+2013 eq. 20)"
    m.av = Prior(
        av, sigma_av, ("Nataf:2013",), note=f"A_V = A_I + E(V-I); {note}"
    )
    m.native.append(
        Prior(ai, sigma_ai, ("Nataf:2013",), band=COUSINS_I, note=note)
    )
    return m


def read_sfd(coord: SkyCoord) -> MapReading:
    """Schlafly & Finkbeiner (2011) recalibrated SFD E(B-V), via IRSA."""
    from astroquery.ipac.irsa.irsa_dust import IrsaDust

    m = MapReading(
        "sfd", "Schlafly & Finkbeiner (2011) / SFD (1998), E(B-V)", "ok"
    )
    t = IrsaDust.get_query_table(coord, section="ebv")
    for col in ("ext SandF mean", "ext SandF std"):
        if col not in t.colnames:
            raise MapFormatError(
                f"IRSA dust table has no '{col}' (columns: "
                f"{', '.join(t.colnames)})."
            )
    ebv = float(t["ext SandF mean"][0])
    e_ebv = float(t["ext SandF std"][0])
    m.detail = (
        f"E(B-V)_SF11 = {ebv:.3f} +/- {e_ebv:.3f} (5' mean/std). The WHOLE "
        f"column, not the extinction to the bar"
    )
    m.av = Prior(
        SF11_RV * ebv,
        SF11_RV * e_ebv,
        ("Schlafly:2011", "Schlegel:1998"),
        note="A_V = 3.1 E(B-V)",
    )
    return m


# Maps that exist but this tool cannot read, and why.  Written to the output.
SKIPPED = (
    (
        "Zucker et al. (2025), 3D (DECaPS)",
        "a multi-GB 3D cube read through the `dustmaps` package, not a web "
        "query; it also saturates at A_V ~ 12 and does not reach the bulge "
        "itself",
    ),
    (
        "Green et al. (2019), Bayestar19 3D",
        "read through the `dustmaps` package (multi-GB download); dec > -30 "
        "only",
    ),
    (
        "Gonzalez et al. (2012), BEAM E(J-Ks)",
        "an interactive web calculator only (no catalog service); Surot+2020 "
        "supersedes it at higher resolution over the same VVV area",
    ),
    (
        "VVV-ExMaps (BaSTI web service of Surot+2020)",
        "needs a user ID from the BaSTI team; the same map is read from "
        "VizieR above",
    ),
    (
        "Marshall et al. (2006), 3D Besancon-model map",
        "not read by this tool (15' resolution, model-dependent)",
    ),
)

# The order the default (live) line is chosen in: the first map that gives a
# usable value.  Nataf's 2026 recommendation (Surot, Zelakiewicz), then the
# OGLE-III optical map; SFD only off the bulge, where it is valid.
DEFAULT_ORDER = ("surot2020", "zelakiewicz2025", "nataf2013", "sfd")
SFD_MIN_ABS_B = 5.0

_NETWORK_ERRORS = (OSError, MapFormatError, MapFetchError)


def query_maps(coord: SkyCoord) -> list[MapReading]:
    """Read every map at `coord`; a map that fails says so, and why."""
    readers = (
        ("surot2020", "Surot et al. (2020)", lambda: read_surot2020(coord)),
        (
            "zelakiewicz2025",
            "Zelakiewicz et al. (2025)",
            lambda: read_zelakiewicz2025(coord, zelakiewicz_path()),
        ),
        (
            "nataf2013",
            "Nataf et al. (2013)",
            lambda: read_nataf2013(coord, nataf2013_path()),
        ),
        ("sfd", "Schlafly & Finkbeiner (2011)", lambda: read_sfd(coord)),
    )
    out = []
    for key, title, read in readers:
        try:
            out.append(read())
        except _NETWORK_ERRORS as e:
            logger.warning(
                f"[mkdust] {title}: the query failed ({type(e).__name__}: "
                f"{e}); it is listed as failed in the output.\n"
                f"Rerun when the service is back to include it."
            )
            out.append(
                MapReading(key, title, "failed", f"{type(e).__name__}: {e}")
            )
    return out


def choose_default(readings: list[MapReading], b_deg: float):
    """The reading whose A_V line is live, or None."""
    by_key = {m.key: m for m in readings}
    for key in DEFAULT_ORDER:
        m = by_key.get(key)
        if m is None or m.status != "ok" or m.av is None:
            continue
        if key == "sfd" and abs(b_deg) < SFD_MIN_ABS_B:
            continue
        return m
    return None


# --- the output ---------------------------------------------------------------


def _fmt(p: Prior) -> str:
    cites = ", ".join(p.citation)
    return (
        f"sed.dust_anchor: {{mu: {p.mu:.3f}, sigma: {p.sigma:.3f}, "
        f"citation: [{cites}]}}"
    )


def render(coord: SkyCoord, readings: list[MapReading], name: str) -> str:
    """The params-file fragment, as text."""
    l0, b0 = _lb(coord)
    default = choose_default(readings, b0)
    ra_s = coord.ra.to_string(u.hourangle, sep=":", precision=2, pad=True)
    dec_s = coord.dec.to_string(
        u.deg, sep=":", precision=2, alwayssign=True, pad=True
    )
    lines = [
        f"# Dust-map priors for {name}: RA, Dec = {ra_s} {dec_s} "
        f"({coord.ra.deg:.5f}, {coord.dec.deg:.5f}); "
        f"l, b = {l0:.4f}, {b0:.4f}",
        f"# Written by exozippy-mkdust on {date.today().isoformat()}.",
        "#",
        "# Paste into the params file.  Each line is a prior on",
        "# sed.dust_anchor, the total of the SED's sight-line dust profile",
        "# (components/sed/sed.md, 'The sight-line dust profile'), which needs",
        "# the profile on: two or more SED objects at different distances, or",
        "# `sed: {distance_constrains_av: true}` for one.",
        "#",
        "# A red-clump map states the extinction TO THE CLUMP (at the bar).",
        "# The anchor sits there when the fit has a red clump or a galactic",
        "# model; with neither it is the whole column, a few percent more.",
        "# With calibrated red-clump rows in the fit, use NONE of the bulge",
        "# maps: they are built from the same clump stars (counted twice).",
        "#",
        "# Uncomment exactly ONE sed.dust_anchor line.  A line marked",
        "# [band X] is in that band, not A_V, and ALSO needs",
        "#     sed: {dust_anchor_band: X}",
        "# in the system config, or it is read as A_V.",
        "",
    ]
    if default is None:
        lines += [
            "# No map gave a usable default here; every line below is",
            "# commented out.  State an A_V prior by hand if you have one.",
            "",
        ]
    else:
        passed = [
            m.title
            for m in readings
            if DEFAULT_ORDER.index(m.key) < DEFAULT_ORDER.index(default.key)
        ]
        lines.append(
            f"# DEFAULT: {default.title}, as A_V.  {default.av.note}."
        )
        if passed:
            lines.append(
                f"# (Preferred but unusable here, see below: "
                f"{'; '.join(passed)}.)"
            )
        lines += [_fmt(default.av), ""]
    for m in readings:
        lines.append(f"# --- {m.title}: {m.status.upper()}")
        if m.detail:
            lines.append(f"#     {m.detail}")
        if m.av is not None and m is not default:
            lines.append(f"#     A_V: {m.av.note}")
            lines.append(f"# {_fmt(m.av)}")
        for p in m.native:
            lines.append(f"#     [band {p.band}]: {p.note}")
            lines.append(f"# {_fmt(p)}")
        lines.append("")
    lines.append("# --- Not queried (cannot be read programmatically here):")
    for title, why in SKIPPED:
        lines.append(f"#     {title}: {why}.")
    return "\n".join(lines) + "\n"


def mkdust(
    ra: str,
    dec: str,
    name: str = "dust",
    outpath: str = ".",
    readings: list[MapReading] | None = None,
) -> Path:
    """Query every map at (ra, dec) and write ``<name>.dust.params.yaml``.

    ``readings`` replaces the queries (tests, or a caller that has them).
    """
    coord = parse_coord(ra, dec)
    if readings is None:
        readings = query_maps(coord)
    text = render(coord, readings, name)
    out = Path(outpath) / f"{name}.dust.params.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    logger.info(f"[mkdust] Wrote {out}")
    return out


def build_parser():
    """Return the argparse parser for the mkdust utility."""
    p = argparse.ArgumentParser(
        prog="exozippy-mkdust",
        description=(
            "Query the extinction maps at a position and write the "
            "sed.dust_anchor prior (all maps, all but one commented out)"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "ra", help="RA: decimal degrees, or hh:mm:ss.ss (hours) with colons"
    )
    p.add_argument(
        "dec", help="Dec: decimal degrees, or +dd:mm:ss.s with colons"
    )
    p.add_argument(
        "--name",
        default="dust",
        help="Base name of the output file <name>.dust.params.yaml",
    )
    p.add_argument("--outpath", default=".", help="Output directory")
    return p


def _protect_negative_sexagesimal(argv):
    """Let a negative sexagesimal Dec through argparse.

    argparse takes "-29:15:47.48" for an option (it accepts only plain
    negative NUMBERS as positionals), so such a token is rewritten to the
    same angle in decimal degrees before parsing.
    """
    out = []
    for tok in argv:
        if re.fullmatch(r"-\d+:\d+(:\d+(\.\d*)?)?", tok):
            tok = f"{Angle(tok, unit=u.deg).deg:.8f}"
        out.append(tok)
    return out


def main(argv=None):
    """CLI entry point. Parses argv (or sys.argv) and runs mkdust."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    argv = sys.argv[1:] if argv is None else list(argv)
    args = build_parser().parse_args(_protect_negative_sexagesimal(argv))
    out = mkdust(args.ra, args.dec, name=args.name, outpath=args.outpath)
    print(out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
