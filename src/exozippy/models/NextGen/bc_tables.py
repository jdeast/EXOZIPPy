"""The published NextGen bolometric-correction tables, with a lazy fetch.

The per-facility ``{FACILITY}.bc.parquet`` tables that
``components/sed/bc_grid.py`` reads are the output of
``generate_NextGen_BC_Tables.py`` run on the full-resolution BT-NextGen
spectra (~250 GB, so the generation happens centrally, never on a user's
machine). They used to be tracked in git and shipped in the wheel; at ~96 MB
for fourteen facilities they pushed the wheel past PyPI's 100 MB per-file
default (PR #349, JDE ruling 2026-10-01), so they are published on Zenodo
instead and fetched on first use -- exactly like the NextGen spectra
(``components/sed/make_bc.py``) and the MIST EEP grid (``models/MIST/
eep_grid.py``), through the same ``utilities/zenodo.fetch_assets`` core and
the same machine-level cache behind it.

This module is the ONE place the published set is pinned. Nothing here runs
at import; ``bc_grid.find_bc_table`` / ``bc_grid.peek_grid_axes`` call
``ensure_tables`` for the facilities a fit actually asks for, so a 2MASS +
Gaia fit fetches two tables, not fourteen. ``fetch_all`` (CLI:
``exozippy-fetch-bc-tables``) pre-fetches the whole set, for a machine that
will run offline.

PUBLISHING A NEW VERSION of the table set (new filters, a longer Av axis,
the HPC full-resolution rebuild of review 2.9.13) is a matter of this file
alone: upload the new tables as a new VERSION of the same Zenodo concept
record, then set ``ZENODO_RECORD`` to the new version's record id and
replace ``_BC_TABLE_FILES`` with the new sizes and md5s (the md5s come from
``https://zenodo.org/api/records/<id>``; the record's ``files[*].checksum``
is ``md5:<hex>``). Every cache is keyed by md5, so the old version's cached
copies are simply never matched again.

Integrity, and why it is STRICTER than the other two assets
-------------------------------------------------------------
``fetch_assets`` re-downloads a destination file whose size is wrong, on
the reasoning that the only way it got that way is a truncated download.
That is not true here: ``generate_NextGen_BC_Tables.py`` (and the manual
``make_bc.py``) MERGE new cells into these very files, so a table that
differs from the pinned one may be hours of someone's generation work,
not corruption. So a present table is checked against the pinned size AND
md5 (once per process) and a mismatch RAISES, naming the file and both
ways out, instead of being silently overwritten -- or silently used, which
would score a fit on a table nobody can reproduce from the record. The
generators pass ``allow_local_changes=True``: they own the edit.

Offline behaviour is explicit: a table that is not on disk and cannot be
fetched raises ``RuntimeError`` naming the file, the record and how to
pre-fetch it. There is no fallback to any other copy. (A RuntimeError, and
deliberately not a FileNotFoundError: ``SED._inject_grid_bounds`` and
``SED._collect_band_filters`` treat FileNotFoundError as "this model has no
tables" and carry on, which would turn a failed download into a fit that
quietly lost its grid bounds.)
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import urllib.error
from pathlib import Path
from typing import Iterable, List, Sequence

from ...utilities import zenodo

logger = logging.getLogger(__name__)

try:
    current_dir = Path(__file__).parent
except NameError:  # pragma: no cover - interactive use only
    current_dir = Path.cwd()

# Destination directory: the BCs/ directory bc_grid.bc_table_path points at
# for the default model root, next to NextGen.grid.yaml and plot.py (which
# stay tracked). The tables themselves are git-ignored there.
BC_TABLE_DIR = current_dir / "BCs"

# Zenodo record of the PUBLISHED VERSION these pins describe, and the
# concept record that groups every version (the DOI to cite for "the NextGen
# BC tables" without naming a version: 10.5281/zenodo.23074950). Version 1,
# published by JDE 2026-10-01 (DOI 10.5281/zenodo.23074951, CC-BY-4.0).
# Browse every EXOZIPPy data record at https://zenodo.org/communities/exozippy
# -- for humans only; the code pins record ids and md5s, never the community.
ZENODO_RECORD = 23074951
ZENODO_CONCEPT_RECORD = 23074950

# size and md5 of every published table, from the record's own API
# (https://zenodo.org/api/records/23074951, files[*].size / checksum). They
# pin the CONTENT: the tables generated at commit d9929630 ("Added BCs
# calculated for more filters", PR #349).
_BC_TABLE_FILES = {
    "2MASS.bc.parquet": {
        "size": 4253006,
        "md5": "41ff8f2f6e9a1b88f3f085f128dfcc2a",
    },
    "Euclid.bc.parquet": {
        "size": 4252914,
        "md5": "44925baf29ad0d1d4d3cedd4d45abc07",
    },
    "GAIA.bc.parquet": {
        "size": 8464605,
        "md5": "073dcf7acbd6195478954cc13a9fd742",
    },
    "GALEX.bc.parquet": {
        "size": 2849599,
        "md5": "e5f774a917144a0c942ea65e2a613b11",
    },
    "Gemini.bc.parquet": {
        "size": 2849819,
        "md5": "73e689d4edafe76052f29f16e5ee9d52",
    },
    "Generic.bc.parquet": {
        "size": 25307518,
        "md5": "4e067a46d646edd2991141f5a3cdd728",
    },
    "Keck.bc.parquet": {
        "size": 7060295,
        "md5": "7b0be59aef0304dcc68b03a7831b7b85",
    },
    "Kepler.bc.parquet": {
        "size": 1445932,
        "md5": "9b21575a77dd737b6f0fa4d43c00b52d",
    },
    "PAN-STARRS.bc.parquet": {
        "size": 5656266,
        "md5": "8f5d5a4c24ad21a93028b52e4414e1ab",
    },
    "Roman.bc.parquet": {
        "size": 11270909,
        "md5": "5e431ef09b0a65a4471f6598f7f7cdae",
    },
    "SLOAN.bc.parquet": {
        "size": 7059882,
        "md5": "95c997b768f763faf51ed41305ff2dcb",
    },
    "TESS.bc.parquet": {
        "size": 1445767,
        "md5": "af9cabbe4018da49e57ccf88c48072ca",
    },
    "TYCHO.bc.parquet": {
        "size": 8464106,
        "md5": "6cbe366d4c185da85d9ab399d609b517",
    },
    "WISE.bc.parquet": {
        "size": 5656474,
        "md5": "7738d148d3dd9e0a37c84720d3737614",
    },
}

FETCH_COMMAND = "exozippy-fetch-bc-tables"

# (path, size, mtime_ns, inode) of tables already md5-verified this process.
# A full md5 is ~0.1 s per 25 MB; a fit reads a handful of tables, often
# more than once, so it is paid once per file per process.
_verified: set[tuple[str, int, int, int]] = set()


def published_tables() -> List[str]:
    """Filenames of every published table, sorted."""
    return sorted(_BC_TABLE_FILES)


def record_url() -> str:
    """Human-facing URL of the pinned record."""
    return f"https://zenodo.org/records/{ZENODO_RECORD}"


def _assets(filenames: Iterable[str]) -> dict:
    """{filename: {"url", "size", "md5"}} for fetch_assets."""
    out = {}
    for name in filenames:
        meta = dict(_BC_TABLE_FILES[name])
        meta["url"] = f"{record_url()}/files/{name}"
        out[name] = meta
    return out


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _how_to_prefetch() -> str:
    return (
        f"Pre-fetch the tables on a machine with network access with "
        f"`{FETCH_COMMAND}` (or `python -m exozippy.models.NextGen."
        f"bc_tables`); the machine cache it fills "
        f"({zenodo.shared_cache_root() or 'switched off by EXOZIPPY_CACHE_DIR'}"
        f"/downloads, relocatable with EXOZIPPY_CACHE_DIR) then serves every "
        f"checkout on this machine without the network."
    )


def _check_present(path: Path, name: str) -> None:
    """Raise unless the table at `path` is byte-identical to the pin."""
    meta = _BC_TABLE_FILES[name]
    st = path.stat()
    key = (str(path), st.st_size, st.st_mtime_ns, st.st_ino)
    if key in _verified:
        return
    if st.st_size != meta["size"] or _md5(path) != meta["md5"]:
        raise RuntimeError(
            f"{path} is not the published {name} (pinned size "
            f"{meta['size']}, md5 {meta['md5']}; record {record_url()}). "
            f"It was most likely regenerated or extended locally "
            f"(generate_NextGen_BC_Tables.py and make_bc.py merge into it). "
            f"It is NOT overwritten, because that may be hours of "
            f"generation work. Either publish it as a new version of the "
            f"Zenodo record and update the pins in "
            f"models/NextGen/bc_tables.py, or move it aside and re-run to "
            f"fetch the published table. To fit against a modified table "
            f"deliberately, copy the NextGen/ tree elsewhere and point the "
            f"SED's `model_root:` at the copy (an explicit root is used as "
            f"given and never fetched into)."
        )
    _verified.add(key)


def ensure_tables(
    filenames: Sequence[str] | None = None,
    dest_dir: Path | str | None = None,
    allow_local_changes: bool = False,
) -> List[Path]:
    """Make sure the named published tables are on disk; return their paths.

    Parameters
    ----------
    filenames
        Table filenames (``"2MASS.bc.parquet"``); None means every published
        table. A name that is not published raises KeyError -- the caller
        (bc_grid) decides which facilities are published before asking.
    dest_dir
        Defaults to ``BC_TABLE_DIR``.
    allow_local_changes
        Accept a present table that differs from the pin (the generators,
        which merge into these files). Absent tables are still fetched.

    Raises
    ------
    RuntimeError
        A present table differs from the pin (see the module docstring), or
        an absent one could not be fetched (no network, Zenodo down). The
        message names the file(s), the record and how to pre-fetch.
    """
    dest_dir = BC_TABLE_DIR if dest_dir is None else Path(dest_dir)
    names = published_tables() if filenames is None else list(filenames)
    unknown = [n for n in names if n not in _BC_TABLE_FILES]
    if unknown:
        raise KeyError(
            f"{unknown} are not published NextGen BC tables; published: "
            f"{published_tables()}"
        )

    missing = []
    for name in names:
        path = dest_dir / name
        if path.is_file():
            if not allow_local_changes:
                _check_present(path, name)
        else:
            missing.append(name)

    if missing:
        assets = _assets(missing)
        try:
            zenodo.fetch_assets(assets, dest_dir)
        except (RuntimeError, urllib.error.URLError, OSError) as e:
            raise RuntimeError(
                f"Could not fetch the NextGen BC table(s) {missing} into "
                f"{dest_dir} from Zenodo record {record_url()}: {e}. "
                f"{_how_to_prefetch()}"
            ) from e
        for name in missing:
            # fetch_assets verified size+md5 before the file became visible.
            path = dest_dir / name
            st = path.stat()
            _verified.add((str(path), st.st_size, st.st_mtime_ns, st.st_ino))

    return [dest_dir / n for n in names]


def fetch_all(dest_dir: Path | str | None = None) -> List[Path]:
    """Fetch every published table (pre-fetch for offline use)."""
    return ensure_tables(None, dest_dir=dest_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=FETCH_COMMAND,
        description=(
            "Download (or link from the machine cache) every published "
            "NextGen bolometric-correction table, so SED fits can run "
            "offline afterwards."
        ),
    )
    parser.add_argument(
        "--dest",
        default=None,
        help=f"destination directory (default: {BC_TABLE_DIR})",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = fetch_all(args.dest)
    total = sum(p.stat().st_size for p in paths)
    print(
        f"{len(paths)} NextGen BC tables ({total / 1e6:.1f} MB) present in "
        f"{paths[0].parent}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
