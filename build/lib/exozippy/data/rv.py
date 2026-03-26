from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class RVInstrumentData:
    name: str
    t: np.ndarray
    y: np.ndarray
    yerr: np.ndarray


def _col(df: pd.DataFrame, candidates: List[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"RV CSV missing required columns; tried: {candidates}. Found: {list(df.columns)}")


def load_rv_csv(path):
    """
    Loads RV data with a flexible header/no-header approach.
    Defaults to: Col 0 = time, Col 1 = rv, Col 2 = error, Col 3+ = detrending.
    """
    # 1. Peek at the file to determine if a header exists
    with open(path, 'r') as f:
        first_line = f.readline().strip()

    # Heuristic: if first line doesn't start with a number/sign, it's likely a header
    has_header = not first_line[0].isdigit() and first_line[0] not in ['-', '+', '.']

    # 2. Read the file (sep=None handles commas, tabs, or spaces automatically)
    df = pd.read_csv(path, sep=None, engine='python', header=0 if has_header else None)

    # 3. Column Mapping
    if has_header:
        # Search for column names using the helper logic
        time_col = _col(df, ["time", "t", "bjd", "BJD", "bjd_tdb"])
        rv_col = _col(df, ["rv", "RV", "mnvel", "vrad", "vel"])
        error_col = _col(df, ["err", "rv_err", "rverr", "sig", "sigma"])

        time = df[time_col].values
        rv = df[rv_col].values
        error = df[error_col].values

        # Capture any extra columns for detrending
        used_cols = [time_col, rv_col, error_col]
        detrend = df.drop(columns=used_cols).values if len(df.columns) > 3 else None
    else:
        # Positional fallback: 0=time, 1=rv, 2=error
        time = df.iloc[:, 0].values
        rv = df.iloc[:, 1].values
        error = df.iloc[:, 2].values

        # Columns 3 and beyond are detrending vectors
        detrend = df.iloc[:, 3:].values if df.shape[1] > 3 else None

    # Ensure data is returned as float64 arrays
    return time.astype(float), rv.astype(float), error.astype(float), detrend


def load_rv_data(fit_config: Mapping, base_dir: Path) -> Dict[str, RVInstrumentData]:
    rv_cfg = (fit_config.get("data", {}) or {}).get("rv", {}) or {}
    insts = rv_cfg.get("instruments", []) or []
    out: Dict[str, RVInstrumentData] = {}
    for inst in insts:
        name = inst["name"]
        f = Path(inst["file"])
        if not f.is_absolute():
            f = base_dir / f
        t, y, e = load_rv_csv(f)
        out[name] = RVInstrumentData(name=name, t=t, y=y, yerr=e)
    return out
