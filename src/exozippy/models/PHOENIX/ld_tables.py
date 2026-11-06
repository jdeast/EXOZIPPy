"""
Fetch the published PHOENIX limb-darkening tables from Zenodo on first use.

The record (``utilities/zenodo_assets.py``, ``phoenix_ld_tables``) holds
one file per facility and law, flattened as ``{FACILITY}.{law}.ld.parquet``
(Zenodo records have no directories); on disk they live where
``generate_PHOENIX_LD_Tables.ld_table_path`` puts them,
``LDs/{law}/{FACILITY}.ld.parquet``.  The rules are the NextGen BC
tables' (``models/NextGen/bc_tables.py``): fetched into the DEFAULT model
root only, md5-checked once per process, a local edit never overwritten
(a previous published version is re-fetched), offline failures named with
the pre-fetch command.

    exozippy-fetch-ld-tables        # every table, for offline use
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import urllib.error
from pathlib import Path
from typing import Dict, List, Sequence

from ...utilities import zenodo, zenodo_assets
from .generate_PHOENIX_LD_Tables import LAWS, ld_table_path

logger = logging.getLogger(__name__)

try:
    current_dir = Path(__file__).parent
except NameError:  # pragma: no cover - interactive use only
    current_dir = Path.cwd()

# The model root the tables are fetched into: the package tree, so
# ld_table_path(DEFAULT_MODEL_ROOT, law, facility) finds them.
MODEL_ROOT = current_dir.parent

_RECORD = zenodo_assets.record("phoenix_ld_tables")
ZENODO_RECORD = _RECORD.record_id
ZENODO_CONCEPT_RECORD = _RECORD.concept_record_id
_TABLE_FILES: Dict[str, Dict] = {
    name: {"size": pin.size, "md5": pin.md5}
    for name, pin in _RECORD.files.items()
    if name.endswith(".ld.parquet")
}
_SUPERSEDED = {
    name: set(md5s)
    for name, md5s in _RECORD.superseded.items()
    if name in _TABLE_FILES
}

FETCH_COMMAND = "exozippy-fetch-ld-tables"

_verified: set[tuple[str, int, int, int]] = set()


def record_name(facility: str, law: str) -> str:
    """The record's flattened filename of one table."""
    return f"{facility}.{law}.ld.parquet"


def split_record_name(name: str) -> tuple[str, str]:
    """``(facility, law)`` of a record filename; raises on any other."""
    stem = name[: -len(".ld.parquet")] if name.endswith(".ld.parquet") else ""
    fac, _, law = stem.rpartition(".")
    if not fac or law not in LAWS:
        raise ValueError(
            f"{name!r} is not a published PHOENIX limb-darkening table name "
            f"({{FACILITY}}.{{law}}.ld.parquet, law in {LAWS})"
        )
    return fac, law


def table_path(name: str, model_root: Path | str | None = None) -> Path:
    fac, law = split_record_name(name)
    return ld_table_path(
        MODEL_ROOT if model_root is None else model_root, law, fac
    )


def published_tables() -> List[str]:
    return sorted(_TABLE_FILES)


def is_published(facility: str, law: str) -> bool:
    return record_name(facility, law) in _TABLE_FILES


def record_url() -> str:
    return f"https://zenodo.org/records/{ZENODO_RECORD}"


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _check_present(path: Path, name: str) -> bool:
    """True when `path` is the pinned table; False for a previous published
    version (re-fetched over); raises on a local edit."""
    meta = _TABLE_FILES[name]
    st = path.stat()
    key = (str(path), st.st_size, st.st_mtime_ns, st.st_ino)
    if key in _verified:
        return True
    md5 = None
    if st.st_size != meta["size"] or (md5 := _md5(path)) != meta["md5"]:
        if (md5 or _md5(path)) in _SUPERSEDED.get(name, ()):
            logger.info(
                "%s is a previous published version of %s; fetching the "
                "pinned one (record %s) over it.",
                path,
                name,
                record_url(),
            )
            return False
        raise RuntimeError(
            f"{path} is not the published {name} (pinned size "
            f"{meta['size']}, md5 {meta['md5']}; record {record_url()}). "
            f"It was most likely regenerated locally "
            f"(generate_PHOENIX_LD_Tables.py merges into it). It is NOT "
            f"overwritten. Either publish it as a new version of the Zenodo "
            f"record and update the pins in utilities/zenodo_assets.py, or "
            f"move it aside and re-run to fetch the published table."
        )
    _verified.add(key)
    return True


def ensure_tables(
    filenames: Sequence[str] | None = None,
    model_root: Path | str | None = None,
    allow_local_changes: bool = False,
) -> List[Path]:
    """Make sure the named published tables (record filenames) are on disk
    under ``model_root`` (default: the package tree); return their paths."""
    root = MODEL_ROOT if model_root is None else Path(model_root)
    names = published_tables() if filenames is None else list(filenames)
    unknown = [n for n in names if n not in _TABLE_FILES]
    if unknown:
        raise KeyError(
            f"{unknown} are not published PHOENIX limb-darkening tables; "
            f"published: {published_tables()}"
        )
    missing = []
    for name in names:
        path = table_path(name, root)
        if path.is_file():
            if not allow_local_changes and not _check_present(path, name):
                path.unlink()
                missing.append(name)
        else:
            missing.append(name)
    # fetch_assets fetches into ONE directory per call; the tables live in
    # one directory per law, so fetch per law.
    for law in LAWS:
        todo = [n for n in missing if split_record_name(n)[1] == law]
        if not todo:
            continue
        dest_dir = table_path(todo[0], root).parent
        assets = {
            n: {**_TABLE_FILES[n], "url": f"{record_url()}/files/{n}"}
            for n in todo
        }
        try:
            zenodo.fetch_assets(assets, dest_dir)
        except (RuntimeError, urllib.error.URLError, OSError) as e:
            raise RuntimeError(
                f"Could not fetch the PHOENIX limb-darkening table(s) {todo} "
                f"into {dest_dir} from Zenodo record {record_url()}: {e}. "
                f"Pre-fetch them on a machine with network access with "
                f"`{FETCH_COMMAND}`; the machine cache it fills then serves "
                f"every checkout on this machine."
            ) from e
        for n in todo:
            # the record's flat name -> the tree's per-law name
            fetched = dest_dir / n
            final = table_path(n, root)
            if fetched != final:
                fetched.replace(final)
            st = final.stat()
            _verified.add((str(final), st.st_size, st.st_mtime_ns, st.st_ino))
    return [table_path(n, root) for n in names]


def fetch_all() -> List[Path]:
    """Fetch every published table: the owner hook for ``phoenix_ld_tables``
    (``zenodo_assets.OWNERS``).  The record's MANIFEST.txt is not fetched."""
    return ensure_tables(None)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog=FETCH_COMMAND,
        description=(
            "Download (or link from the machine cache) every published "
            "PHOENIX limb-darkening table, so fits can run offline afterwards."
        ),
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=None,
        help=f"Model root to fetch into (default: {MODEL_ROOT}).",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    paths = ensure_tables(None, model_root=args.model_root)
    print(
        f"{len(paths)} table(s) in place under {args.model_root or MODEL_ROOT}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
