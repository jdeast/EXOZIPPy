"""
Build the Claret limb-darkening tables EXOZIPPy publishes on Zenodo, and
check them cell by cell against EXOFASTv2's copies.

The rulings the tables serve are in components/sed/sed.md ("The Claret
limb-darkening tie"). In short:

1. Download the published coefficients from VizieR into a fresh directory
   and pin every source file by size and md5:
     - Claret & Bloemen (2011), J/A+A/529/A75: ``tableab.dat`` (quadratic)
       and ``tableu.dat`` (linear), 23 bands;
     - Claret (2017), J/A+A/600/A30: ``table25.dat`` (TESS quadratic).
   The subset kept is EXOFASTv2's: ATLAS models, least-squares method (L),
   microturbulence 2 km/s.

2. Place each band's rows on EXOFASTv2's (Teff, logg, [Fe/H]) axes (the
   ones hard-coded in quadld.pro). A row off those axes, or two rows on one
   node, raises.

3. Cross-check every cell against EXOFASTv2's ``$EXOFAST_PATH/quadld/
   ATLAS.L.2.<band>[.linear].sav``. A cell valid in both copies must be
   EXACTLY equal: both come from the same 4-decimal publication, so any
   difference is a regridding disagreement to be investigated and ruled on
   here, never absorbed by a tolerance. Cells valid in only one copy are
   counted and reported, not treated as errors. EXOFASTv2's converter
   (claret.pro, which does not ship) interpolated some nodes VizieR does not
   tabulate; those carry more than 4 decimals, which the report counts as
   evidence of where they came from. None of them lies inside the box.
   EXOFASTv2's copies were built from the original April 2011 release, and
   three of them still carry that release's errors, which the 17 Jun 2011
   correction (the VizieR version read here) fixed: its Sloan z' quadratic
   table has a and b swapped, and its Stromgren b and v quadratic tables are
   copies of Johnson B and V (jdeast/EXOFASTv2#10). They are declared in
   EXOFAST_DEFECTS.

4. Trim every table to the box that is complete in all of them (see
   BOX), cutting along [Fe/H] the way the MIST grid is trimmed rather than
   with the BC grid's most-incomplete-first rule, which would cut Teff and
   lose the K dwarfs. A missing cell inside the box raises.

5. Write the flat Zenodo bundle: one long-format parquet file per (band,
   law) -- ``claret_ATLAS_L_vt2.<band>.<quad|linear>.parquet`` with columns
   teff, logg, feh and a, b (quadratic) or u (linear), and the provenance in
   ``df.attrs["meta"]`` (the BC tables' convention, read back with
   ``bc_grid.read_bc_meta``) -- plus a README.md stating the sources, the
   selection, the box and the cross-check result. ``scripts/
   zenodo_publish.py`` writes MANIFEST.txt; see docs/zenodo.md.

Claret's metallicity column is log[M/H]; EXOFASTv2 (and EXOZIPPy) index the
tables by the star's [Fe/H], so the axis is named ``feh`` here.

Where things go. The tables are written to ``models/Claret/build/``
(DEFAULT_BUILD_DIR), which is the Zenodo bundle -- deliberately NOT
``models/Claret/ATLAS_L_vt2/`` (DEFAULT_TABLE_DIR), where a fit fetches the
PUBLISHED tables and checks each against its pinned md5: a rebuild sitting
there would fail that check. To fit against a rebuild before publishing it,
pass its directory to ``claret_tables.load_table(..., table_dir=...)``. The
raw VizieR downloads go to ``models/Claret/sources/`` (DEFAULT_SOURCE_DIR).
All three are git-ignored. Re-running replaces this generator's own files in
the output directory and refuses if anything else is there.

Usage (from a git checkout, so the git revision can be recorded)::

    python -m exozippy.models.Claret.generate_claret_tables --exofast ~/EXOFASTv2

    # re-run without the network on the files downloaded before
    python -m exozippy.models.Claret.generate_claret_tables --exofast ~/EXOFASTv2 \\
        --source-dir src/exozippy/models/Claret/sources
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import requests
from scipy.io import readsav

from exozippy.models.Claret.claret_tables import (
    BOX,
    DATASET,
    DEFAULT_BUILD_DIR,
    DEFAULT_SOURCE_DIR,
    LAW_COLUMNS,
    TABLE_BANDS,
    table_filename,
)

# -------------------------------------------------------------------
# Sources
# -------------------------------------------------------------------

VIZIER_FTP = "https://cdsarc.cds.unistra.fr/ftp"

# Seconds of silence before the socket errors (not a budget for the whole
# transfer), as in models/MIST/download_MIST_EEPs.py.
_HTTP_TIMEOUT = 60


@dataclass(frozen=True)
class Catalog:
    key: str
    vizier_id: str
    bibcode: str
    reference: str
    files: Tuple[str, ...]


CATALOGS = {
    "cb11": Catalog(
        key="cb11",
        vizier_id="J/A+A/529/A75",
        bibcode="2011A&A...529A..75C",
        reference="Claret & Bloemen (2011), A&A 529, A75",
        files=("ReadMe", "tableab.dat.gz", "tableu.dat.gz"),
    ),
    "c17": Catalog(
        key="c17",
        vizier_id="J/A+A/600/A30",
        bibcode="2017A&A...600A..30C",
        reference="Claret (2017), A&A 600, A30",
        files=("ReadMe", "table25.dat.gz"),
    ),
}

# Fixed-width layouts, transcribed from each catalog's ReadMe
# ("Byte-by-byte Description"). The ReadMe's 1-based inclusive byte range
# a-b is the 0-based half-open colspec (a-1, b).
CB11_QUADRATIC = {
    "logg": (0, 5),
    "teff": (6, 12),
    "feh": (13, 17),
    "xi": (18, 22),
    "a": (23, 30),
    "b": (31, 38),
    "filt": (39, 41),
    "met": (42, 43),
    "mod": (44, 51),
}
CB11_LINEAR = {
    "logg": (0, 5),
    "teff": (6, 12),
    "feh": (13, 17),
    "xi": (18, 22),
    "u": (23, 30),
    "filt": (31, 33),
    "met": (34, 35),
    "mod": (37, 44),
}
# table25.dat has no method column: the least-squares (LSM) and flux-
# conservation (PCM) fits are separate columns, and only the ATLAS model
# ("A") is in this table.
C17_TABLE25 = {
    "logg": (0, 5),
    "teff": (6, 12),
    "feh": (13, 17),
    "xi": (19, 22),
    "aLSM": (23, 31),
    "bLSM": (33, 41),
    "aPCM": (43, 51),
    "bPCM": (53, 61),
    "chi2": (65, 71),
    "mod": (72, 74),
}

# The subset EXOFASTv2 uses (quadld.pro's defaults, never overridden).
MODEL = "ATLAS"
METHOD = "L"
XI = 2.0
# DATASET, the table and source directories, the file names, the per-law
# coefficient columns and BOX are claret_tables.py's, shared with the
# reader so the two cannot drift.

# EXOFASTv2's band name (also the `Claret` column of filters/
# filternames.txt) -> Claret & Bloemen's filter code, per law. The two
# files disagree on one spelling: SDSS u'g'r'i'z' are coded with '*'
# standing in for the prime in tableab.dat (quadratic) and with the prime
# itself in tableu.dat (linear). Stromgren uvby are the lower-case single
# letters, distinct from Johnson-Cousins only by case.
_CB11_SDSS = {"quadratic": "*", "linear": "'"}


def _cb11_codes(law: str) -> Dict[str, str]:
    prime = _CB11_SDSS[law]
    return {
        "U": "U",
        "B": "B",
        "V": "V",
        "R": "R",
        "I": "I",
        "J": "J",
        "H": "H",
        "K": "K",
        "Sloanu": "u" + prime,
        "Sloang": "g" + prime,
        "Sloanr": "r" + prime,
        "Sloani": "i" + prime,
        "Sloanz": "z" + prime,
        "Kepler": "Kp",
        "CoRoT": "C",
        "Spit36": "S1",
        "Spit45": "S2",
        "Spit58": "S3",
        "Spit80": "S4",
        "Stromu": "u",
        "Stromb": "b",
        "Stromv": "v",
        "Stromy": "y",
    }


CB11_BANDS = {law: _cb11_codes(law) for law in _CB11_SDSS}
# TESS has a quadratic table only (EXOFASTv2 ships no TESS .linear.sav).
C17_BANDS = {"TESS": "TESS"}

# EXOFASTv2's save-file variable names per law.
EXOFAST_VARIABLES = {
    "quadratic": {"a": "quadlda", "b": "quadldb"},
    "linear": {"u": "linld"},
}
EXOFAST_SUFFIX = {"quadratic": "", "linear": ".linear"}


@dataclass(frozen=True)
class ExofastDefect:
    """A known difference between one of EXOFASTv2's save files and VizieR:
    an error of the April 2011 release EXOFASTv2 was built from, which the
    corrected release fixed (jdeast/EXOFASTv2#10).

    The comparison stays EXACT in the corrected form, so any further change
    in either copy still raises:

    * ``holds_band``: the file is a copy of ANOTHER band's table, and is
      compared against that band's VizieR rows instead;
    * ``swap``: maps each of our coefficients to the EXOFASTv2 coefficient
      holding the same number.
    """

    note: str
    holds_band: str | None = None
    swap: Dict[str, str] | None = None


# Every difference below was found by this cross-check (all 24 quadratic and
# 23 linear EXOFASTv2 tables were compared; the other 44 agree exactly). Each
# is an error of the original April 2011 release EXOFASTv2's tables were
# built from, fixed in the 17 Jun 2011 correction this reads
# (jdeast/EXOFASTv2#10). The tables published here use VizieR's rows.
EXOFAST_DEFECTS = {
    # ATLAS.L.2.Sloanz.sav stores Claret's b as quadlda (its u1) and
    # Claret's a as quadldb (its u2), on all 7695 tabulated nodes. VizieR's
    # assignment is the physical one: at Teff 5750 K, logg 4.5, [Fe/H] 0,
    # Claret's z' (a, b) = (0.2717, 0.2522) sits on the smooth
    # r' -> i' -> z' -> J sequence of a (0.3941, 0.3111, 0.2717, 0.1520),
    # which the swapped pair breaks.
    ("Sloanz", "quadratic"): ExofastDefect(
        note="a and b are swapped (its u1 is Claret's b)",
        swap={"a": "b", "b": "a"},
    ),
    # ATLAS.L.2.Stromb.sav and ATLAS.L.2.Stromv.sav are exact copies of the
    # Johnson-Cousins B and V tables (all 7695 nodes, both coefficients),
    # not Stromgren b and v. Its Stromgren u and y tables, and all four
    # Stromgren linear tables, match VizieR.
    ("Stromb", "quadratic"): ExofastDefect(
        note="it is a copy of the Johnson B table", holds_band="B"
    ),
    ("Stromv", "quadratic"): ExofastDefect(
        note="it is a copy of the Johnson V table", holds_band="V"
    ),
}

# EXOFASTv2's axes, as quadld.pro builds them, rounded to the tabulated
# values: its arithmetic gives -0.3 + 3*0.1 = 5.6e-17 for the solar node,
# and a published table must hold 0.0 there.
EXOFAST_FEH = np.round(
    np.concatenate(
        [
            -5.0 + np.arange(10) * 0.5,
            -0.3 + np.arange(7) * 0.1,
            0.5 + np.arange(2) * 0.5,
        ]
    ),
    2,
)
EXOFAST_LOGG = np.arange(11) * 0.5
EXOFAST_TEFF = np.concatenate(
    [
        3.5e3 + np.arange(39) * 2.5e2,
        1.4e4 + np.arange(24) * 1e3,
        [3.75e4],
        3.8e4 + np.arange(5) * 1e3,
        [4.25e4],
        4.3e4 + np.arange(5) * 1e3,
        [4.75e4],
        4.8e4 + np.arange(3) * 1e3,
    ]
)


# Axis values are compared after rounding to these decimals, so a node is
# identified the same way however its float was produced.
_KEY_DECIMALS = {"teff": 0, "logg": 2, "feh": 2}


# -------------------------------------------------------------------
# Download and provenance
# -------------------------------------------------------------------


def _local_name(cat: Catalog, fname: str) -> str:
    """Both catalogs ship a ReadMe, so local copies carry the key."""
    return f"{cat.key}.{fname}"


def download_sources(dest: Path) -> Dict[str, Path]:
    """Fetch every source file into `dest`, which must be new or empty.

    A fresh directory per run: a downloaded file is untrusted input, and a
    directory holding files from some earlier run could silently mix two
    versions of a catalog.
    """
    dest = Path(dest).expanduser()
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError(
            f"download directory {dest} is not empty; give a new one, or pass "
            f"it as --source-dir to reuse the files already there."
        )
    dest.mkdir(parents=True, exist_ok=True)
    paths = {}
    for cat in CATALOGS.values():
        for fname in cat.files:
            url = f"{VIZIER_FTP}/{cat.vizier_id}/{fname}"
            out = dest / _local_name(cat, fname)
            print(f"Downloading {url}")
            with requests.get(url, stream=True, timeout=_HTTP_TIMEOUT) as resp:
                resp.raise_for_status()
                with open(out, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        fh.write(chunk)
            if out.stat().st_size == 0:
                raise RuntimeError(f"{url} downloaded as an empty file.")
            paths[_local_name(cat, fname)] = out
    return paths


def existing_sources(src: Path) -> Dict[str, Path]:
    """The source files of an earlier download, all of which must exist."""
    src = Path(src).expanduser()
    paths = {}
    for cat in CATALOGS.values():
        for fname in cat.files:
            p = src / _local_name(cat, fname)
            if not p.is_file():
                raise FileNotFoundError(
                    f"{p} is missing: --source-dir must hold every file "
                    f"download_sources writes (the {cat.vizier_id} "
                    f"{fname!r} is absent)."
                )
            paths[_local_name(cat, fname)] = p
    return paths


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def source_record(cat: Catalog, fname: str, path: Path) -> Dict:
    return {
        "catalog": cat.vizier_id,
        "bibcode": cat.bibcode,
        "reference": cat.reference,
        "file": fname,
        "url": f"{VIZIER_FTP}/{cat.vizier_id}/{fname}",
        "size": path.stat().st_size,
        "md5": _md5(path),
    }


def git_revision() -> str:
    """`git describe` of the checkout this script runs from.

    Required, not best-effort: a published record must say which generator
    produced it. A dirty tree is recorded as such ("-dirty").
    """
    try:
        out = subprocess.run(
            ["git", "describe", "--always", "--dirty"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "could not read the git revision of the generator; run it from "
            f"an EXOZIPPy git checkout ({exc})."
        ) from exc
    return out.stdout.strip()


# -------------------------------------------------------------------
# Parsing
# -------------------------------------------------------------------


def _read_fixed_width(
    path: Path, layout: Dict[str, Tuple[int, int]]
) -> pd.DataFrame:
    """One VizieR fixed-width table. Text columns stay text (a filter coded
    "NA" or "C" must not become a missing value)."""
    text_cols = {"filt", "met", "mod"}
    df = pd.read_fwf(
        path,
        colspecs=list(layout.values()),
        names=list(layout.keys()),
        dtype={c: str for c in layout if c in text_cols},
        keep_default_na=False,
        # Every data file VizieR serves here is gzipped; stated rather than
        # inferred from the name.
        compression="gzip",
    )
    for col in layout:
        if col in text_cols:
            df[col] = df[col].str.strip()
        else:
            df[col] = pd.to_numeric(df[col], errors="raise").astype(float)
    if df.empty:
        raise ValueError(f"{path}: no rows parsed.")
    return df


def _check_codes(
    df: pd.DataFrame, col: str, allowed: Sequence[str], path: Path
):
    found = set(df[col].unique())
    unknown = found - set(allowed)
    if unknown:
        raise ValueError(
            f"{path}: unexpected {col!r} values {sorted(unknown)} (expected a "
            f"subset of {sorted(allowed)}); the file does not match the "
            f"layout transcribed from its ReadMe."
        )


def read_cb11(path: Path, law: str) -> pd.DataFrame:
    """Claret & Bloemen (2011) rows of the EXOFASTv2 subset, per law."""
    layout = CB11_QUADRATIC if law == "quadratic" else CB11_LINEAR
    df = _read_fixed_width(path, layout)
    _check_codes(df, "met", ("L", "F"), path)
    _check_codes(df, "mod", ("ATLAS", "PHOENIX"), path)
    _check_codes(df, "filt", tuple(CB11_BANDS[law].values()), path)
    sel = df[(df["mod"] == MODEL) & (df["met"] == METHOD) & (df["xi"] == XI)]
    if sel.empty:
        raise ValueError(f"{path}: no {MODEL}/{METHOD}/xi={XI} rows.")
    return sel[["teff", "logg", "feh", "filt", *LAW_COLUMNS[law]]]


def read_c17_table25(path: Path) -> pd.DataFrame:
    """Claret (2017) TESS quadratic rows of the EXOFASTv2 subset.

    Method L is the LSM column pair; the PCM pair is the flux-conservation
    fit and is not used.
    """
    df = _read_fixed_width(path, C17_TABLE25)
    _check_codes(df, "mod", ("A",), path)
    sel = df[df["xi"] == XI].rename(columns={"aLSM": "a", "bLSM": "b"})
    if sel.empty:
        raise ValueError(f"{path}: no xi={XI} rows.")
    sel = sel.assign(filt="TESS")
    return sel[["teff", "logg", "feh", "filt", "a", "b"]]


# -------------------------------------------------------------------
# Cubes on EXOFASTv2's axes
# -------------------------------------------------------------------

AXES = {"feh": EXOFAST_FEH, "teff": EXOFAST_TEFF, "logg": EXOFAST_LOGG}


def _axis_index(name: str) -> Dict[float, int]:
    dec = _KEY_DECIMALS[name]
    return {round(float(v), dec): i for i, v in enumerate(AXES[name])}


def build_cubes(
    rows: pd.DataFrame, columns: Sequence[str], label: str
) -> Dict[str, np.ndarray]:
    """One (feh, teff, logg) cube per coefficient, NaN where Claret has no
    value. The axis order is EXOFASTv2's save files' (IDL's (logg, teff,
    feh), reversed by readsav)."""
    shape = (len(EXOFAST_FEH), len(EXOFAST_TEFF), len(EXOFAST_LOGG))
    cubes = {c: np.full(shape, np.nan) for c in columns}
    index = {ax: _axis_index(ax) for ax in AXES}
    seen = np.zeros(shape, dtype=bool)
    for r in rows.itertuples(index=False):
        key = []
        for ax in ("feh", "teff", "logg"):
            v = round(float(getattr(r, ax)), _KEY_DECIMALS[ax])
            if v not in index[ax]:
                raise ValueError(
                    f"{label}: node ({ax}={v}) is not on EXOFASTv2's {ax} "
                    f"axis; the grid assumption behind these tables is wrong."
                )
            key.append(index[ax][v])
        key = tuple(key)
        if seen[key]:
            raise ValueError(
                f"{label}: two rows on node feh={r.feh}, teff={r.teff}, "
                f"logg={r.logg}; refusing to choose between them."
            )
        seen[key] = True
        for c in columns:
            cubes[c][key] = getattr(r, c)
    return cubes


def _box_masks():
    return {
        ax: (AXES[ax] >= BOX[ax][0] - 1e-9) & (AXES[ax] <= BOX[ax][1] + 1e-9)
        for ax in AXES
    }


def trim_to_box(cubes: Dict[str, np.ndarray], label: str) -> pd.DataFrame:
    """Long-format table of the box, which must be complete."""
    m = _box_masks()
    sub = {
        c: cube[np.ix_(m["feh"], m["teff"], m["logg"])]
        for c, cube in cubes.items()
    }
    for c, arr in sub.items():
        if not np.isfinite(arr).all():
            fe, te, lg = (
                AXES["feh"][m["feh"]],
                AXES["teff"][m["teff"]],
                AXES["logg"][m["logg"]],
            )
            holes = [
                f"(teff={te[j]:.0f}, logg={lg[k]:.1f}, feh={fe[i]:+.1f})"
                for i, j, k in np.argwhere(~np.isfinite(arr))[:5]
            ]
            raise ValueError(
                f"{label}: coefficient {c!r} has {int((~np.isfinite(arr)).sum())} "
                f"missing cell(s) inside the box {BOX}, e.g. {', '.join(holes)}."
            )
    F, T, L = np.meshgrid(
        AXES["feh"][m["feh"]],
        AXES["teff"][m["teff"]],
        AXES["logg"][m["logg"]],
        indexing="ij",
    )
    df = pd.DataFrame({"teff": T.ravel(), "logg": L.ravel(), "feh": F.ravel()})
    for c, arr in sub.items():
        df[c] = arr.ravel()
    return df.sort_values(["teff", "logg", "feh"], ignore_index=True)


# -------------------------------------------------------------------
# Cross-check against EXOFASTv2
# -------------------------------------------------------------------


def load_exofast_cubes(
    exofast_root: Path, band: str, law: str
) -> Dict[str, np.ndarray]:
    path = (
        Path(exofast_root).expanduser()
        / "quadld"
        / f"{MODEL}.{METHOD}.{int(XI)}.{band}{EXOFAST_SUFFIX[law]}.sav"
    )
    if not path.is_file():
        raise FileNotFoundError(f"EXOFASTv2 table {path} not found.")
    data = readsav(str(path))
    # Compared at the node-key rounding, not bit for bit: the save files hold
    # the tabulated values (-0.2) where quadld.pro's arithmetic gives
    # -0.3 + 0.1 = -0.19999999999999998.
    for ax, key in (("feh", "fehs"), ("teff", "teffs"), ("logg", "loggs")):
        dec = _KEY_DECIMALS[ax]
        theirs = np.round(np.asarray(data[key], float), dec)
        if not np.array_equal(theirs, np.round(AXES[ax], dec)):
            raise ValueError(
                f"{path}: its {key} axis differs from quadld.pro's; the cubes "
                f"cannot be compared cell by cell."
            )
    return {
        c: np.asarray(data[v], float)
        for c, v in EXOFAST_VARIABLES[law].items()
    }


def _more_than_4_decimals(values: np.ndarray) -> np.ndarray:
    scaled = values * 1e4
    return np.abs(scaled - np.round(scaled)) > 1e-6


def cross_check(
    band: str,
    law: str,
    cubes: Dict[Tuple[str, str], Dict[str, np.ndarray]],
    theirs: Dict[str, np.ndarray],
) -> Dict:
    """Exact agreement wherever both copies hold a value; counts otherwise.

    ``cubes`` holds every band's VizieR cubes, because a declared
    ``holds_band`` defect compares EXOFASTv2's file against ANOTHER band.
    """
    label = f"{band} ({law})"
    defect = EXOFAST_DEFECTS.get((band, law))
    ref_band = (
        band
        if defect is None or defect.holds_band is None
        else defect.holds_band
    )
    swap = None if defect is None else defect.swap
    ours = cubes[(ref_band, law)]
    m = _box_masks()
    in_box = (
        m["feh"][:, None, None]
        & m["teff"][None, :, None]
        & m["logg"][None, None, :]
    )
    stats = {}
    for c in ours:
        theirs_c = c if swap is None else swap[c]
        a, b = ours[c], theirs[theirs_c]
        both = np.isfinite(a) & np.isfinite(b)
        differ = both & (a != b)
        if differ.any():
            cells = [
                f"(teff={EXOFAST_TEFF[j]:.0f}, logg={EXOFAST_LOGG[k]:.1f}, "
                f"feh={EXOFAST_FEH[i]:+.1f}): VizieR {a[i, j, k]!r} vs "
                f"EXOFASTv2 {b[i, j, k]!r}"
                for i, j, k in np.argwhere(differ)[:5]
            ]
            raise ValueError(
                f"{label}: VizieR {ref_band} {c!r} vs EXOFASTv2 {band} "
                f"{theirs_c!r}: {int(differ.sum())} cell(s) disagree, e.g. "
                f"{'; '.join(cells)}. Investigate and rule on it in this "
                f"generator (EXOFAST_DEFECTS is where a ruled defect goes); "
                f"do not loosen the comparison."
            )
        exofast_only = np.isfinite(b) & ~np.isfinite(a)
        stats[c] = {
            # Every compared cell is equal; anything else raised above.
            "pairing": f"VizieR {ref_band} {c} = EXOFASTv2 {band} {theirs_c}",
            "compared": int(both.sum()),
            "vizier_only": int((np.isfinite(a) & ~np.isfinite(b)).sum()),
            "exofast_only": int(exofast_only.sum()),
            "exofast_only_interpolated": int(
                _more_than_4_decimals(b[exofast_only]).sum()
            ),
            "exofast_only_in_box": int((exofast_only & in_box).sum()),
        }
    return stats


# -------------------------------------------------------------------
# Output
# -------------------------------------------------------------------


def write_table(df: pd.DataFrame, meta: Dict, path: Path) -> None:
    out = df.copy()
    out.attrs["meta"] = meta
    out.to_parquet(path, index=False)


def write_readme(
    bundle: Path, records: List[Dict], sources: List[Dict], revision: str
):
    lines = [
        "# Claret limb-darkening tables for EXOZIPPy",
        "",
        "Quadratic and linear limb-darkening coefficients from Claret's published",
        "tables, regridded and trimmed for EXOZIPPy's Claret limb-darkening prior.",
        "The coefficients are Claret's; cite the source papers when you use them.",
        "",
        "## Sources",
        "",
    ]
    for cat in CATALOGS.values():
        lines.append(
            f"- {cat.reference}, VizieR {cat.vizier_id} ({cat.bibcode})"
        )
    lines += [
        "",
        "Source files, as downloaded:",
        "",
        "| file | size | md5 |",
        "|---|---|---|",
    ]
    for s in sources:
        lines.append(
            f"| {s['catalog']}/{s['file']} | {s['size']} | {s['md5']} |"
        )
    lines += [
        "",
        "## Selection and grid",
        "",
        f"- Model {MODEL}, method {METHOD} (least squares), microturbulence {XI:g} km/s:",
        "  the subset EXOFASTv2 uses. For TESS (Claret 2017, table25) method L is",
        "  the LSM column pair.",
        "- Nodes are placed on EXOFASTv2's (Teff, logg, [Fe/H]) axes (quadld.pro).",
        "  Claret's metallicity is log[M/H]; the column is named `feh` because",
        "  EXOFASTv2 and EXOZIPPy index the tables by the star's [Fe/H].",
        "- Every table is trimmed to the box complete in all of them:",
        f"  Teff {BOX['teff'][0]:.0f}-{BOX['teff'][1]:.0f} K, logg {BOX['logg'][0]:.1f}-"
        f"{BOX['logg'][1]:.1f}, [Fe/H] {BOX['feh'][0]:+.1f} to {BOX['feh'][1]:+.1f}.",
        "  No cell is filled or interpolated.",
        "- One file per band and law,",
        f"  `{DATASET}.<band>.<quad|linear>.parquet`, in long format:",
        "  columns teff, logg, feh, and a, b (quadratic,",
        "  I(mu)/I(1) = 1 - a(1-mu) - b(1-mu)^2) or u (linear).",
        "  Provenance is in the pandas attrs (`df.attrs['meta']`).",
        "",
        "## Cross-check against EXOFASTv2",
        "",
        "Every cell valid in both this build (full grid, before trimming) and",
        "EXOFASTv2's `quadld/ATLAS.L.2.<band>[.linear].sav` was compared and is",
        "exactly equal. `EXOFASTv2 only` counts nodes EXOFASTv2's converter added",
        "that VizieR does not tabulate at this selection; `interpolated` is how many",
        "of those carry more than 4 decimals (Claret publishes 4). None lies inside",
        "the box.",
        "",
    ]
    if EXOFAST_DEFECTS:
        lines += [
            "EXOFASTv2's copies were built from the original April 2011 release, and",
            "these still carry that release's errors, which the corrected release fixed",
            "(jdeast/EXOFASTv2#10). The comparison is still exact, in the pairing shown",
            "in the table's `pairing` column, and these tables use VizieR's rows:",
            "",
        ]
        for (band, law), defect in EXOFAST_DEFECTS.items():
            lines.append(
                f"- `{MODEL}.{METHOD}.{int(XI)}.{band}{EXOFAST_SUFFIX[law]}.sav` "
                f"({law}): {defect.note}."
            )
        lines.append("")
    lines += [
        "| band | law | coefficient | pairing | compared (all equal) | VizieR only | EXOFASTv2 only | interpolated | in box |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for rec in records:
        for c, s in rec["exofastv2_check"].items():
            lines.append(
                f"| {rec['band']} | {rec['law']} | {c} | {s['pairing']} | {s['compared']} | "
                f"{s['vizier_only']} | {s['exofast_only']} | "
                f"{s['exofast_only_interpolated']} | {s['exofast_only_in_box']} |"
            )
    lines += [
        "",
        f"Generated by `exozippy.models.Claret.generate_claret_tables` at git revision {revision}.",
        "",
    ]
    (bundle / "README.md").write_text("\n".join(lines))


# -------------------------------------------------------------------
# Driver
# -------------------------------------------------------------------


def _is_own_output(path: Path) -> bool:
    """A file this generator writes into the table directory."""
    return path.name == "README.md" or (
        path.name.startswith(f"{DATASET}.") and path.suffix == ".parquet"
    )


def prepare_table_dir(out: Path) -> Path:
    """Create `out`, or clear this generator's previous output from it.

    The directory doubles as the Zenodo bundle, which must hold exactly the
    record's files, so anything this generator did not write (a stray file,
    a fetch's staging copy) RAISES rather than being published or deleted.
    Previous tables are removed, not overwritten in place, so a band dropped
    from the build cannot linger.
    """
    out = Path(out).expanduser()
    if out.exists():
        others = [p.name for p in out.iterdir() if not _is_own_output(p)]
        if others:
            raise FileExistsError(
                f"{out} holds files this generator did not write: "
                f"{sorted(others)}. Move them aside; the directory is the "
                f"Zenodo bundle and must hold exactly the record's files."
            )
        for p in out.iterdir():
            p.unlink()
    out.mkdir(parents=True, exist_ok=True)
    return out


def build_bundle(
    exofast_root: Path, out: Path, sources: Dict[str, Path]
) -> List[Dict]:
    # Everything that can fail before writing is done first, so a failed
    # run leaves the previous tables in place.
    revision = git_revision()

    source_info = {
        name: source_record(
            CATALOGS[name.split(".", 1)[0]], name.split(".", 1)[1], p
        )
        for name, p in sources.items()
    }

    tables = {
        ("quadratic", "cb11"): read_cb11(
            sources["cb11.tableab.dat.gz"], "quadratic"
        ),
        ("linear", "cb11"): read_cb11(sources["cb11.tableu.dat.gz"], "linear"),
        ("quadratic", "c17"): read_c17_table25(sources["c17.table25.dat.gz"]),
    }
    source_file = {
        ("quadratic", "cb11"): "cb11.tableab.dat.gz",
        ("linear", "cb11"): "cb11.tableu.dat.gz",
        ("quadratic", "c17"): "c17.table25.dat.gz",
    }
    jobs = [
        (band, code, "quadratic", "cb11")
        for band, code in CB11_BANDS["quadratic"].items()
    ]
    jobs += [
        (band, code, "quadratic", "c17") for band, code in C17_BANDS.items()
    ]
    jobs += [
        (band, code, "linear", "cb11")
        for band, code in CB11_BANDS["linear"].items()
    ]
    # The readers trust TABLE_BANDS to say which tables exist; hold the
    # build to exactly that set.
    for law, bands in TABLE_BANDS.items():
        built = sorted(b for b, _c, lw, _cat in jobs if lw == law)
        if built != sorted(bands):
            raise ValueError(
                f"the {law} tables built ({built}) differ from "
                f"claret_tables.TABLE_BANDS ({sorted(bands)}); update one."
            )

    # Every cube first: a declared defect may compare EXOFASTv2's file
    # against another band's VizieR rows.
    cubes = {}
    for band, code, law, cat in jobs:
        label = f"{band} ({law})"
        rows = tables[(law, cat)]
        rows = rows[rows["filt"] == code]
        if rows.empty:
            raise ValueError(
                f"{label}: no rows for Claret filter code {code!r}."
            )
        cubes[(band, law)] = build_cubes(rows, LAW_COLUMNS[law], label)

    checked = []
    for band, code, law, cat in jobs:
        label = f"{band} ({law})"
        check = cross_check(
            band, law, cubes, load_exofast_cubes(exofast_root, band, law)
        )
        checked.append(
            (
                band,
                code,
                law,
                cat,
                check,
                trim_to_box(cubes[(band, law)], label),
            )
        )

    out = prepare_table_dir(out)
    records = []
    for band, code, law, cat, check, df in checked:
        meta = {
            "dataset": DATASET,
            "band": band,
            "law": law,
            "claret_filter": code,
            "model": MODEL,
            "method": METHOD,
            "xi_km_s": XI,
            "columns": list(df.columns),
            "box": {ax: list(BOX[ax]) for ax in ("teff", "logg", "feh")},
            "feh_note": "Claret's log[M/H], used as [Fe/H] as in EXOFASTv2",
            "source": source_info[source_file[(law, cat)]],
            "exofastv2_check": check,
            "generator": "exozippy.models.Claret.generate_claret_tables",
            "git_revision": revision,
        }
        path = out / table_filename(band, law)
        write_table(df, meta, path)
        records.append(
            {
                "band": band,
                "law": law,
                "file": path.name,
                "exofastv2_check": check,
            }
        )
        n_only = sum(s["exofast_only"] for s in check.values())
        print(
            f"{path.name}: {len(df)} nodes; EXOFASTv2 agrees on "
            f"{sum(s['compared'] for s in check.values())} cell(s), "
            f"{n_only} EXOFASTv2-only"
        )

    write_readme(out, records, list(source_info.values()), revision)
    print(f"Wrote {len(records)} tables and README.md to {out}")
    return records


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--exofast",
        required=True,
        type=Path,
        help="EXOFASTv2 checkout (its quadld/ directory is the cross-check).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_BUILD_DIR,
        help="Output directory, which is also the Zenodo bundle "
        f"(default: {DEFAULT_BUILD_DIR}).",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--download-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="New or empty directory to download the VizieR files into "
        f"(default: {DEFAULT_SOURCE_DIR}).",
    )
    group.add_argument(
        "--source-dir",
        type=Path,
        help="Directory of an earlier download to reuse instead of fetching.",
    )
    args = parser.parse_args(argv)

    if args.source_dir is not None:
        sources = existing_sources(args.source_dir)
    else:
        sources = download_sources(args.download_dir)
    build_bundle(args.exofast, args.out, sources)
    return 0


if __name__ == "__main__":
    sys.exit(main())
