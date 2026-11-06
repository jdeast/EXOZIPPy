"""
Fetch raw BT-NextGen (AGSS2009) spectra from SVO's theory server.

Step 0 of models/NextGen/README.md used to be manual ("download the
spectra from SVO").  This module does it: it reads SVO's SSAP index of
the model (one VOTable row per spectrum, carrying teff, logg, [M/H],
alpha and the file id), maps the (teff, logg, feh) nodes of
NextGen.grid.yaml onto file ids -- trying the alphas in
nextgen_spectra.ALPHA_GRID_PTS order, exactly as find_spectrum_file
reads them back -- and downloads each spectrum as the ASCII file
process_spectrum parses, under the name get_NextGen2009_filename gives
it, so the rest of the pipeline sees what it always saw.

The index is also the pipeline's AUTHORITY on which nodes exist: the
raw grid is not a rectangle (logg 5.5 stops at 5000 K, logg 6.0 at
4000 K), and a node the index does not list is one no download can
fill.  generate_NextGen_BC_Tables.process_missing_spectra skips such
nodes on request and pad_missing_nodes fills their BC rows from the
nearest lower-gravity plane.

    python -m exozippy.models.NextGen.fetch_raw_spectra --logg 5.5
    python -m exozippy.models.NextGen.fetch_raw_spectra  # the whole grid
"""

from __future__ import annotations

import argparse
import logging
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from exozippy.models.NextGen.nextgen_spectra import (
    ALPHA_GRID_PTS,
    SPECTRA_RAW_PATH_DEFAULT,
    get_NextGen2009_filename,
)

logger = logging.getLogger(__name__)

SVO_MODEL = "bt-nextgen-agss2009"
SSAP_URL = (
    f"https://svo2.cab.inta-csic.es/theory/newov2/ssap.php?model={SVO_MODEL}"
)
# The ASCII the whole pipeline reads (two columns, '#' comments).
ASCII_SUFFIX = "&format=ascii"

# A complete BT-NextGen ASCII spectrum runs to ~1e7 Angstrom in ~380k
# rows (12.5 MB); anything much smaller is a truncated download.
_MIN_BYTES = 1_000_000
_INDEX_CACHE = "svo_index.votable.xml"

Node = Tuple[float, float, float]  # (teff, logg, feh)


def _round(x) -> float:
    return float(round(float(x), 6))


def svo_index(
    raw_path: Path | str = SPECTRA_RAW_PATH_DEFAULT,
    refresh: bool = False,
    timeout: float = 120.0,
) -> pd.DataFrame:
    """SVO's index of the model: one row per spectrum, columns teff, logg,
    feh, alpha, fid.

    Cached as ``svo_index.votable.xml`` under ``raw_path`` (5 MB), so a
    rerun needs no network; ``refresh`` re-downloads it.
    """
    raw_path = Path(raw_path)
    cache = raw_path / _INDEX_CACHE
    if refresh or not cache.is_file():
        raw_path.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(SSAP_URL, timeout=timeout) as resp:
            cache.write_bytes(resp.read())
    return parse_ssap_index(
        cache.read_text(encoding="utf-8", errors="replace")
    )


def parse_ssap_index(text: str) -> pd.DataFrame:
    """The (teff, logg, feh, alpha, fid) rows of an SSAP VOTable.

    SVO's row layout is positional: TD 0-3 are teff, logg, [M/H] and
    alpha (alpha blank for the plain grid), and the access URL carries
    ``fid=<id>``.  A row without a fid is not a spectrum and is skipped.
    """
    recs = []
    for row in re.findall(r"<TR[^>]*>(.*?)</TR>", text, flags=re.S | re.I):
        m = re.search(r"fid=(\d+)", row)
        tds = re.findall(r"<TD[^>]*>(.*?)</TD>", row, flags=re.S | re.I)
        if not m or len(tds) < 4:
            continue
        try:
            teff, logg, feh = (float(tds[k]) for k in range(3))
        except ValueError:
            continue
        alpha = float(tds[3]) if tds[3].strip() else 0.0
        recs.append((teff, logg, feh, alpha, int(m.group(1))))
    if not recs:
        raise ValueError("no spectrum rows found in the SVO SSAP index")
    return pd.DataFrame(recs, columns=["teff", "logg", "feh", "alpha", "fid"])


def available_nodes(index: pd.DataFrame) -> Set[Node]:
    """Every (teff, logg, feh) that has a spectrum at SOME alpha."""
    return {
        (_round(t), _round(g), _round(z))
        for t, g, z in zip(index.teff, index.logg, index.feh)
    }


def fid_for_node(index: pd.DataFrame, teff, logg, feh) -> Tuple[float, int]:
    """``(alpha, fid)`` of a node's spectrum, first alpha in
    ALPHA_GRID_PTS order that the index lists; KeyError when none."""
    sub = index[
        (np.isclose(index.teff, teff))
        & (np.isclose(index.logg, logg))
        & (np.isclose(index.feh, feh))
    ]
    for alpha in ALPHA_GRID_PTS:
        hit = sub[np.isclose(sub.alpha, alpha)]
        if len(hit):
            return float(alpha), int(hit.fid.iloc[0])
    raise KeyError(
        f"SVO lists no BT-NextGen spectrum for teff={teff}, logg={logg}, "
        f"feh={feh} at any alpha in {list(ALPHA_GRID_PTS)}"
    )


def spectrum_url(fid: int) -> str:
    return f"{SSAP_URL}&fid={int(fid)}{ASCII_SUFFIX}"


def _complete(path: Path) -> bool:
    """A downloaded ASCII spectrum that parses to its last row."""
    if not path.is_file() or path.stat().st_size < _MIN_BYTES:
        return False
    with path.open("rb") as fh:
        fh.seek(-200, 2)
        tail = fh.read().decode("ascii", errors="replace").strip().splitlines()
    if not tail:
        return False
    parts = tail[-1].split()
    try:
        return len(parts) == 2 and float(parts[0]) > 1e6
    except ValueError:
        return False


def fetch_spectrum(
    fid: int,
    dest: Path,
    timeout: float = 300.0,
    retries: int = 4,
) -> Path:
    """Download one spectrum to ``dest`` (atomically, via ``dest.part``),
    skipping a complete file already there."""
    dest = Path(dest)
    if _complete(dest):
        return dest
    part = dest.with_suffix(dest.suffix + ".part")
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(
                spectrum_url(fid), timeout=timeout
            ) as r:
                part.write_bytes(r.read())
            if not _complete(part):
                raise OSError(f"truncated download ({part.stat().st_size} B)")
            part.replace(dest)
            return dest
        except (OSError, ValueError) as exc:  # network, HTTP, truncation
            last = exc
            time.sleep(min(60.0, 5.0 * 2**attempt))
    raise RuntimeError(f"could not fetch fid={fid} -> {dest}: {last}")


def fetch_nodes(
    nodes: Iterable[Node],
    raw_path: Path | str = SPECTRA_RAW_PATH_DEFAULT,
    index: Optional[pd.DataFrame] = None,
    n_workers: int = 4,
) -> Dict[str, list]:
    """Download the spectra of ``nodes`` ((teff, logg, feh) triples) that are
    not already on disk.  Returns {"fetched": [...], "present": [...],
    "unavailable": [...]} of nodes; unavailable ones are not an error --
    they are what pad_missing_nodes exists for."""
    raw_path = Path(raw_path)
    index = svo_index(raw_path) if index is None else index
    out = {"fetched": [], "present": [], "unavailable": []}
    jobs = []
    for teff, logg, feh in nodes:
        try:
            alpha, fid = fid_for_node(index, teff, logg, feh)
        except KeyError:
            out["unavailable"].append((teff, logg, feh))
            continue
        dest = raw_path / get_NextGen2009_filename(
            {"teff": teff, "logg": logg, "feh": feh, "alpha": alpha}
        )
        if _complete(dest):
            out["present"].append((teff, logg, feh))
            continue
        jobs.append(((teff, logg, feh), fid, dest))
    if jobs:
        raw_path.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(max_workers=max(1, n_workers)) as pool:
            for node, _path in zip(
                [j[0] for j in jobs],
                pool.map(lambda j: fetch_spectrum(j[1], j[2]), jobs),
            ):
                out["fetched"].append(node)
    logger.info(
        "BT-NextGen raw spectra: %d fetched, %d already present, %d not on "
        "SVO at any alpha",
        len(out["fetched"]),
        len(out["present"]),
        len(out["unavailable"]),
    )
    return out


def grid_nodes_from_yaml(
    grid: dict,
    logg: Optional[Sequence[float]] = None,
    teff_max: Optional[float] = None,
) -> list:
    """The (teff, logg, feh) triples of a grid dict, optionally restricted
    to some logg planes and to teff <= teff_max."""
    loggs = [float(g) for g in grid["logg"]]
    if logg is not None:
        wanted = {float(g) for g in logg}
        loggs = [g for g in loggs if g in wanted]
    return [
        (float(t), g, float(z))
        for z in grid["feh"]
        for g in loggs
        for t in grid["teff"]
        if teff_max is None or float(t) <= teff_max
    ]


def main(argv=None):
    from exozippy.models.NextGen.generate_NextGen_BC_Tables import (
        read_grid_yaml,
    )

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--raw-path", default=str(SPECTRA_RAW_PATH_DEFAULT))
    p.add_argument(
        "--logg",
        type=float,
        nargs="*",
        default=None,
        help="only these logg planes of the grid (default: all)",
    )
    p.add_argument("--teff-max", type=float, default=None)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--refresh-index", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    raw_path = Path(args.raw_path)
    index = svo_index(raw_path, refresh=args.refresh_index)
    nodes = grid_nodes_from_yaml(read_grid_yaml(), args.logg, args.teff_max)
    out = fetch_nodes(nodes, raw_path, index=index, n_workers=args.workers)
    print(
        f"fetched {len(out['fetched'])}, present {len(out['present'])}, "
        f"unavailable {len(out['unavailable'])} of {len(nodes)} nodes"
    )


if __name__ == "__main__":
    main()
