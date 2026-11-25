#!/usr/bin/env python
"""Renormalize published NextGen BC tables from sigma Teff^4 to each
spectrum's own F_bol (review 8.9.13) -- the one-time conversion that made
the own-F_bol version of the Zenodo record ``nextgen_bc_tables``.

The generator divided every band flux by sigma Teff^4
(``bolometric_correction._calc_BC``, before the fix).  That factor is one
number per spectrum, independent of filter and A_V, so the corrected BC
at every row is

    BC_new = BC_old + 2.5 log10(sigma Teff^4 / F_bol)

with F_bol the spectrum's own bolometric flux, read from the per-node
table ``NextGen.fbol.parquet`` (``generate_NextGen_BC_Tables.
write_fbol_table``), which is published with the tables.  The fixed
generator reproduces this to float rounding (tests/test_nextgen_fbol.py;
verified on the real spectra to <1e-6 mag, PR body),
so a column regenerated from the spectra and a column converted here are
the same column.

Refuses a table whose columns already carry the own-F_bol normalization
(no double application), and checks that every padded node's source in
the table's ``meta["padding"]`` is the source the F_bol table used.

    python scripts/renormalize_nextgen_bc_tables.py \\
        --fbol NextGen.fbol.parquet --src <published tables dir> \\
        --out <new version dir>
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from exozippy.components.sed.bc_grid import (
    _KEY_DECIMALS,
    bc_filter_columns,
    read_bc_table,
    write_bc_table,
)
from exozippy.models.NextGen.bolometric_correction import NORMALIZATION


def _key(t, g, z):
    return tuple(round(float(x), _KEY_DECIMALS) for x in (t, g, z))


def delta_by_node(fbol: pd.DataFrame) -> dict:
    """{(teff, logg, feh): 2.5 log10(sigma Teff^4 / F_bol)}."""
    return {
        _key(t, g, z): -2.5 * np.log10(r)
        for t, g, z, r in zip(
            fbol.teff, fbol.logg, fbol.feh, fbol.fbol_over_sigmaT4
        )
    }


def _check_padding(meta: dict, fbol: pd.DataFrame, name: str) -> None:
    source = {
        _key(t, g, z): s
        for t, g, z, s in zip(fbol.teff, fbol.logg, fbol.feh, fbol.source)
    }
    for p in meta.get("padding", []):
        k = _key(p["teff"], p["logg"], p["feh"])
        if "from_logg" in p:
            want = f"from_logg {float(p['from_logg']):g}"
        else:
            t0, t1 = p["from_teff"]
            want = f"from_teff {float(t0):g} {float(t1):g}"
        if source.get(k) != want:
            raise ValueError(
                f"{name}: padded node {k} came from {want!r}, but the F_bol "
                f"table's source there is {source.get(k)!r}."
            )


def renormalize_table(src: Path, out: Path, fbol: pd.DataFrame) -> dict:
    df = read_bc_table(src)
    meta = dict(df.attrs.get("meta", {}))
    fmeta = dict(meta.pop("filters", {}))
    meta.pop("version", None)
    cols = bc_filter_columns(df)
    done = [c for c in cols if fmeta.get(c, {}).get("normalization")]
    if done or meta.get("normalization"):
        raise ValueError(
            f"{src.name}: already normalized ({done or 'table meta'}); "
            f"refusing to apply the correction twice."
        )
    _check_padding(meta, fbol, src.name)
    deltas = delta_by_node(fbol)
    keys = [_key(t, g, z) for t, g, z in zip(df.teff, df.logg, df.feh)]
    missing = sorted({k for k in keys if k not in deltas})
    if missing:
        raise KeyError(f"{src.name}: no F_bol for nodes {missing[:5]}")
    d = np.array([deltas[k] for k in keys])
    out_df = df.copy()
    for c in cols:
        out_df[c] = df[c].to_numpy(float) + d
        fmeta[c] = {**fmeta[c], "normalization": NORMALIZATION}
    meta["normalization"] = NORMALIZATION
    meta["renormalized"] = (
        "BC + 2.5 log10(sigma Teff^4 / F_bol) per (teff, logg, feh) node, "
        "F_bol from NextGen.fbol.parquet (scripts/renormalize_nextgen_bc_"
        "tables.py; review 8.9.13). The four interior nodes SVO serves no "
        "spectrum for take the F_bol of the Teff-interpolated spectrum."
    )
    out.unlink(missing_ok=True)
    write_bc_table(out_df, out, fmeta, table_meta=meta)
    return {"rows": len(df), "delta_min": d.min(), "delta_max": d.max()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fbol", required=True, type=Path)
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    fbol = pd.read_parquet(a.fbol)
    a.out.mkdir(parents=True, exist_ok=True)
    for src in sorted(a.src.glob("*.bc.parquet")):
        info = renormalize_table(src, a.out / src.name, fbol)
        print(f"{src.name}: {info}")
    if a.fbol.resolve() != (a.out / a.fbol.name).resolve():
        shutil.copy2(a.fbol, a.out / a.fbol.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
