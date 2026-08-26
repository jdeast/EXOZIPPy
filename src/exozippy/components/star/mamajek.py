"""Mamajek's mean dwarf sequence as a general stellar-property lookup.

Python port of EXOFASTv2's getstar.pro / read_mamajek.pro (Eastman, CfA,
public release 2024/08), consuming the same shipped table
(EEM_dwarf_UBVIJHK_colors_Teff.txt, v2022.04.16 -- cite Pecaut & Mamajek
2013, ApJS 208, 9).  Given ANY tabulated quantity and its value, get_star
interpolates EVERY column of the table at that point, so one measured
number (a Teff, an absolute magnitude, a color) yields a full
main-sequence stellar description -- mass, radius, logL, BCv, colors.
Assumes a dwarf.

Faithful-port notes:
- rows are split on whitespace AND colons (some table rows carry stray
  colons); '...' placeholders (any run of >= 3 dots) are NaN;
  '-' -> '_' in column names (B-V -> B_V).
- minmass drops rows whose Msun is missing or below the floor BEFORE
  interpolation; getstar.pro documents "will not extrapolate, but will
  use the minimum value in the table that is above this" -- implemented
  here by clamping the fractional index into the filtered table's range
  (the IDL original's interpolate() clamps the same way).  minmass=0.1
  is the EXOFASTv2 recommendation when seeding MIST models.
- 'SpT' cannot be interpolated: exact (case-preserving) match only,
  mirroring the IDL special case.
"""

from pathlib import Path

import numpy as np

_TABLE = Path(__file__).parent / "EEM_dwarf_UBVIJHK_colors_Teff.txt"
_CACHE = None


def read_mamajek(minmass=None):
    """The full table as {column: float array} plus 'SpT' (str list).

    Rows with a missing or sub-``minmass`` Msun are dropped when
    ``minmass`` is given.  The unfiltered parse is cached.
    """
    global _CACHE
    if _CACHE is None:
        cols = None
        spt = []
        rows = []
        for line in _TABLE.read_text(encoding="utf-8").splitlines():
            tokens = line.replace(":", " ").split()
            if not tokens:
                continue
            if tokens[0] == "#SpT" and "Teff" in tokens:
                if cols is not None:
                    break  # the footer repeats the header: end of data
                # drop the leading and trailing SpT bookend columns
                cols = [tok.replace("-", "_") for tok in tokens[1:-1]]
                continue
            if line.startswith("#") or cols is None:
                continue
            vals = tokens[1:-1]
            if len(vals) != len(cols):
                continue
            spt.append(tokens[0])
            rows.append([np.nan if "..." in v else float(v) for v in vals])
        data = np.asarray(rows, dtype=float)
        _CACHE = {"SpT": spt}
        for j, name in enumerate(cols):
            _CACHE[name] = data[:, j]
    out = dict(_CACHE)
    if minmass is not None:
        msun = out["Msun"]
        keep = np.isfinite(msun) & (msun >= float(minmass))
        for k, v in out.items():
            out[k] = (
                [s for s, g in zip(v, keep) if g] if k == "SpT" else v[keep]
            )
    return out


def get_star(tagname, value, minmass=None):
    """Every table column interpolated at ``tagname == value``.

    Returns {column: float} (plus 'SpT': the nearest row's type).  The
    tagname is case-insensitive and must be a table column; 'SpT' is
    special-cased to an exact match.  No extrapolation: the fractional
    index is clamped to the (minmass-filtered) table.
    """
    table = read_mamajek(minmass=minmass)
    names = {k.upper(): k for k in table}
    key = names.get(str(tagname).upper())
    if key is None:
        raise ValueError(
            f"'{tagname}' not supported, use one of "
            f"{', '.join(k for k in table)}"
        )

    if key == "SpT":
        try:
            i = table["SpT"].index(value)
        except ValueError:
            raise ValueError(
                f"SpT value ({value}) not found and cannot be interpolated"
            ) from None
        return {
            k: (v[i] if k == "SpT" else float(v[i])) for k, v in table.items()
        }

    col = table[key]
    ok = np.isfinite(col)
    idx = np.arange(col.size, dtype=float)[ok]
    coln = col[ok]
    # np.interp needs ascending x; the table's columns are monotone in
    # either direction (Teff descends, magnitudes ascend).
    order = np.argsort(coln)
    x = float(np.interp(float(value), coln[order], idx[order]))
    x = min(max(x, 0.0), col.size - 1.0)

    lo, hi = int(np.floor(x)), int(np.ceil(x))
    f = x - lo
    out = {}
    for k, v in table.items():
        if k == "SpT":
            out[k] = v[lo if f < 0.5 else hi]
            continue
        out[k] = float((1.0 - f) * v[lo] + f * v[hi])
    return out
