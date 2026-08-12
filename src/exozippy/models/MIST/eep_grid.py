"""Reader for the processed MISTv2.5 EEP track grid, with a lazy fetch.

The grid is the output of the authoring workflow in this directory's
README (download_MIST_EEPs -> generate_MIST_EEP_Tables -> the missing-point
interpolation steps). Running that workflow needs ~186 GB of raw MIST
tarballs and hours of processing, so the finished product is published on
Zenodo and fetched on first use instead.

This module is the ONE place the shipped grid is read. Anything that wants
EEP tracks calls ``load_eep_grid()`` (or ``ensure_eep_grid()`` for just the
path) and gets the download for free, on first use -- not at import, and not
at ``System.prepare()`` unless a component reads the grid there. Nothing is
downloaded by importing this module.

The parquet is git-ignored, cached in place at
``models/MIST/MISTv{version}/EEPs/``, and pinned by size and md5 from the
Zenodo record's API, so a truncated or re-uploaded file is caught rather
than silently used. See :mod:`exozippy.utilities.zenodo` for the mechanics.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

from ...utilities.zenodo import fetch_assets

logger = logging.getLogger(__name__)

try:
    current_dir = Path(__file__).parent
except NameError:  # pragma: no cover - interactive use only
    current_dir = Path.cwd()

# Cache directory. Matches the .gitignore entry for the parquet, and the
# EEPs/ tree the grid.yaml for this version already lives in.
EEP_GRID_DIR = current_dir / "MISTv2.5" / "EEPs"

# Published grids, keyed by the same {afe}_{vvcrit} stem
# generate_MIST_EEP_Tables._generate_alpha_vvcrit_filename_parts builds, so
# a grid produced locally and one fetched from Zenodo are interchangeable.
#
# size and md5 come from the Zenodo record's own API
# (https://zenodo.org/api/records/21893308). They pin the content, so a
# re-uploaded or truncated file is caught rather than silently used.
_EEP_GRID_ASSETS = {
    "afe_p0_vvcrit0.0.grid.parquet": {
        "url": (
            "https://zenodo.org/records/21893308/files/"
            "afe_p0_vvcrit0.0.grid.parquet"
        ),
        "size": 127992558,
        "md5": "30545e0087ddd7dd79e87f294f4d8d58",
    },
}


def eep_grid_filename(alpha: float = 0.0, vvcrit: float = 0.0) -> str:
    """Grid filename for an (alpha, vvcrit) pair, MISTv2.5 spelling.

    Kept byte-identical to
    generate_MIST_EEP_Tables._generate_alpha_vvcrit_filename_parts so this
    module and the authoring script never disagree about a name. Duplicated
    rather than imported because that script pulls in tqdm and the whole
    track-processing stack, which a fit that only reads the finished grid has
    no reason to load.
    """
    sign = "m" if alpha < 0 else "p"
    return f"afe_{sign}{abs(alpha) * 10:.0f}_vvcrit{vvcrit:0.1f}.grid.parquet"


def ensure_eep_grid(alpha: float = 0.0, vvcrit: float = 0.0) -> Path:
    """Return the local path to the EEP grid parquet, downloading if needed.

    The fetch is skipped entirely when the file is already cached at the
    pinned size. A failure -- no network, a checksum mismatch, a partial
    download -- raises with the file named; there is no fallback to a missing
    or half-written grid.

    Raises
    ------
    KeyError
        No grid is published for this (alpha, vvcrit). Generate it yourself
        with the workflow in this directory's README.
    RuntimeError
        The download failed or did not verify; see utilities.zenodo.
    """
    filename = eep_grid_filename(alpha, vvcrit)
    meta = _EEP_GRID_ASSETS.get(filename)
    if meta is None:
        available = ", ".join(sorted(_EEP_GRID_ASSETS)) or "(none)"
        raise KeyError(
            f"No published MIST EEP grid for alpha={alpha}, "
            f"vvcrit={vvcrit} (would be '{filename}'). Published grids: "
            f"{available}. Generate others locally with the workflow in "
            f"{current_dir / 'README.md'}."
        )

    fetch_assets({filename: meta}, EEP_GRID_DIR)
    return EEP_GRID_DIR / filename


def load_eep_grid(
    alpha: float = 0.0,
    vvcrit: float = 0.0,
    columns: Sequence[str] | None = None,
):
    """Read the EEP track grid into a DataFrame, fetching it on first use.

    `columns` is forwarded to pandas; the full grid is ~128 MB on disk and
    considerably more in memory, so callers that need three columns should
    ask for three columns.
    """
    import pandas as pd

    path = ensure_eep_grid(alpha=alpha, vvcrit=vvcrit)
    return pd.read_parquet(path, columns=columns, engine="pyarrow")
