"""Magnitude systems of SED photometry rows (review 1.9.1).

Every bolometric-correction column is referenced to ONE magnitude system,
the table's ``mag_system`` (Vega for every NextGen column), while a catalog
publishes its magnitudes in its own: SDSS, Pan-STARRS, GALEX and Euclid in
AB, 2MASS, Gaia, WISE and Tycho in Vega.  An AB magnitude fitted against a
Vega-referenced BC is wrong by m_AB - m_Vega in its band (SDSS u +0.90, PS1 z
+0.51, GALEX FUV +2.09 mag) against errors of a few hundredths, so the SED
converts every row onto the column's system before anything reads it.

Three things live here, and nowhere else:

* the ONE internal spelling of a magnitude system (``VEGA``, ``AB``) and the
  boundary translation from a .sed file's spelling (``parse_magsys``,
  case-SENSITIVE: a case variant raises with a "did you mean" hint, as does
  any unknown spelling, naming the row);
* the per-column record the BC generator writes beside its tables
  (``{model}.magsys.csv``: the column's native system and its m_AB - m_Vega
  offset, computed from the same filter profile, flux weighting and Vega
  zeropoint as the column's BCs -- ``models/NextGen/generate_NextGen_BC_
  Tables.py:magsys_table``), and its reader;
* the drift check that ties a record to the column it describes: the
  record's SVO id, Vega zeropoint and flux weighting must equal the
  column's own ``filter_meta``, so an offset can never outlive the column it
  was computed for.

A row with no stated system is in its filter's NATIVE system (JDE ruling
2026-10-01), read from the same record; a filter with no recorded native
system and no stated one raises naming the filter.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

# The one internal spelling of each supported magnitude system.  ST is not
# supported: no shipped catalog publishes it and no column records an
# m_ST - m_Vega offset, so it raises at the boundary like any other unknown
# spelling rather than being silently fitted as something else.
VEGA = "Vega"
AB = "AB"
MAG_SYSTEMS = (VEGA, AB)

# Case variants, used ONLY to suggest the right spelling in the error: the
# boundary is case-sensitive (JDE 2026-10-01: "g is different than G").
_CASE_HINTS = {name.lower(): name for name in MAG_SYSTEMS}

MAGSYS_COLUMNS = (
    "column",
    "svo_id",
    "native_system",
    "zeropoint_Fl_Vega",
    "flux_weighting",
    "ab_minus_vega",
)


def parse_magsys(value, where):
    """A .sed row's ``magsys`` in the internal spelling, or None if unstated.

    ``where`` names the row (file, index, filter) for the error message.
    CASE-SENSITIVE (JDE 2026-10-01): only the exact spellings ``"Vega"``
    and ``"AB"`` are accepted.  A case variant raises with a "did you mean"
    hint -- suggesting is fine, accepting is not, because user-facing
    names are case-sensitive throughout (SDSS g is not Gaia G).
    """
    if value is None:
        return None
    if value in MAG_SYSTEMS:
        return value
    hint = _CASE_HINTS.get(str(value).strip().lower())
    did_you_mean = f" Did you mean {hint!r}?" if hint else ""
    raise ValueError(
        f"{where}: unknown magnitude system magsys: {value!r}.{did_you_mean} "
        f"Supported, case-sensitive: {', '.join(repr(m) for m in MAG_SYSTEMS)}; "
        f"omit the key to mean the filter's native system."
    )


def magsys_table_path(model_root, model):
    """Where the BC generator writes the per-column magnitude-system record,
    beside ``{model}.grid.yaml``."""
    return Path(model_root) / model / "BCs" / f"{model}.magsys.csv"


def read_magsys_table(model_root, model):
    """The per-column magnitude-system record, indexed by BC column name.

    Raises naming the file when it is absent: every SED row needs it (its
    native system, and its offset when AB), so there is nothing to fall
    back to.
    """
    path = magsys_table_path(model_root, model)
    if not path.is_file():
        raise FileNotFoundError(
            f"No magnitude-system record for the {model} BC tables at "
            f"{path}. It is written by the table generator "
            f"(models/NextGen/generate_NextGen_BC_Tables.py, "
            f"write_magsys_table) beside {model}.grid.yaml; a model_root "
            f"copied from the package must carry it too."
        )
    stat = path.stat()
    return _read_magsys_table_cached(path, stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=4)
def _read_magsys_table_cached(path, _mtime_ns, _size):
    df = pd.read_csv(
        path,
        comment="#",
        dtype={"native_system": "string", "flux_weighting": "string"},
        keep_default_na=False,
    )
    if tuple(df.columns) != MAGSYS_COLUMNS:
        raise ValueError(
            f"{path} has columns {list(df.columns)}; expected "
            f"{list(MAGSYS_COLUMNS)}."
        )
    if df["column"].duplicated().any():
        dup = sorted(df.loc[df["column"].duplicated(), "column"])
        raise ValueError(f"{path} lists BC column(s) {dup} more than once.")
    bad = sorted(set(df["native_system"]) - set(MAG_SYSTEMS) - {""})
    if bad:
        raise ValueError(
            f"{path}: native_system value(s) {bad} are not one of "
            f"{MAG_SYSTEMS} (or empty for a filter with no native system)."
        )
    return df.set_index("column")


def native_system(filter_name, model_root, model):
    """The native magnitude system of a filter, from the column record.

    The ONE source of a filter's native system, read both by the SED (what
    an unstated row means) and by mkticsed (the system it writes on every
    row), so the two cannot disagree.  Raises naming the filter when the
    record has no row for its column, or records no native system for it.
    """
    from .bc_grid import _load_alias_table, resolve_filter_name

    col = resolve_filter_name(filter_name, _load_alias_table(), alias="MIST")
    record = read_magsys_table(model_root, model)
    path = magsys_table_path(model_root, model)
    if col not in record.index:
        raise ValueError(
            f"Filter {filter_name!r} (BC column {col}) has no magnitude-"
            f"system record in {path}."
        )
    system = record.loc[col, "native_system"]
    if not system:
        raise ValueError(
            f"Filter {filter_name!r} (BC column {col}) has no native "
            f"magnitude system recorded in {path} (none exists, or it is "
            f"unresolved); state magsys explicitly."
        )
    return system


def check_record_matches_column(record, column, filter_meta, path):
    """Raise unless ``record`` was computed for the column as it now stands.

    The offset is a property of (filter profile, flux weighting, Vega
    zeropoint); the column's ``filter_meta`` records all three for the BCs
    actually in the table.  A regenerated column with a new zeropoint or
    weighting, and a record left behind, would otherwise convert AB rows
    with an offset for a different column.
    """
    problems = []
    if record["svo_id"] != filter_meta["svo_id"]:
        problems.append(
            f"svo_id {record['svo_id']!r} vs {filter_meta['svo_id']!r}"
        )
    if record["flux_weighting"] != filter_meta["flux_weighting"]:
        problems.append(
            f"flux_weighting {record['flux_weighting']!r} vs "
            f"{filter_meta['flux_weighting']!r}"
        )
    if not np.isclose(
        float(record["zeropoint_Fl_Vega"]),
        float(filter_meta["zeropoint_Fl_Vega"]),
        rtol=1e-9,
        atol=0.0,
    ):
        problems.append(
            f"zeropoint_Fl_Vega {record['zeropoint_Fl_Vega']!r} vs "
            f"{filter_meta['zeropoint_Fl_Vega']!r}"
        )
    if problems:
        raise ValueError(
            f"The magnitude-system record for BC column {column!r} in {path} "
            f"does not describe the column in its table "
            f"({'; '.join(problems)}). Re-run write_magsys_table in "
            f"models/NextGen/generate_NextGen_BC_Tables.py so the AB offset "
            f"matches the column's BCs."
        )
