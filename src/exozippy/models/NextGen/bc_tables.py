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

This module owns the fetch of the published set; the pins themselves (record
id, concept record, every table's size and md5) live in
``utilities/zenodo_assets.py`` as record ``nextgen_bc_tables``, with every
other Zenodo pin (review 4.9.2). Nothing here downloads at import;
``bc_grid.find_bc_table`` / ``bc_grid.peek_grid_axes`` call ``ensure_tables``
for the facilities a fit actually asks for, so a 2MASS + Gaia fit fetches
two tables, not fourteen. ``fetch_all`` (CLI:
``exozippy-fetch-bc-tables``) pre-fetches the whole set, for a machine that
will run offline.

PUBLISHING A NEW VERSION of the table set (new filters, a longer Av axis,
the HPC full-resolution rebuild of review 2.9.13) touches no code here:
upload the new tables as a new VERSION of the same Zenodo concept record,
then update the ``nextgen_bc_tables`` entry in ``utilities/zenodo_assets.py``
and run its network test (that module's docstring has the steps). Every
cache is keyed by md5, so the old version's cached copies are simply never
matched again.

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

from ...utilities import zenodo, zenodo_assets

logger = logging.getLogger(__name__)

try:
    current_dir = Path(__file__).parent
except NameError:  # pragma: no cover - interactive use only
    current_dir = Path.cwd()

# Destination directory: the BCs/ directory bc_grid.bc_table_path points at
# for the default model root, next to NextGen.grid.yaml and plot.py (which
# stay tracked). The tables themselves are git-ignored there.
BC_TABLE_DIR = current_dir / "BCs"

# The published record (utilities/zenodo_assets.py, "nextgen_bc_tables").
# Module-level copies so the fetch below reads -- and tests substitute -- one
# name each: the VERSION the pins describe, the concept record grouping every
# version, and each table's size and md5.
_RECORD = zenodo_assets.record("nextgen_bc_tables")
ZENODO_RECORD = _RECORD.record_id
ZENODO_CONCEPT_RECORD = _RECORD.concept_record_id
# The per-node bolometric flux of every spectrum the tables were computed
# from (review 8.9.13; generate_NextGen_BC_Tables.write_fbol_table): the
# tables' normalization, published with them, and read -- never recomputed
# -- by the SED's fitted-R_V mode (spectral_bc.py) and the SED figure.
FBOL_TABLE = "NextGen.fbol.parquet"

# Only the tables (and the F_bol table): the record also carries the publish
# script's MANIFEST.txt, which no reader opens (the Claret record's rule).
_BC_TABLE_FILES = {
    name: {"size": pin.size, "md5": pin.md5}
    for name, pin in _RECORD.files.items()
    if name.endswith(".bc.parquet") or name == FBOL_TABLE
}
# md5s of each table in EARLIER published versions (zenodo_assets
# ZenodoRecord.superseded): a checkout that fetched the previous version
# holds one of these, and gets the pinned version fetched over it.
_SUPERSEDED = {
    name: set(md5s)
    for name, md5s in _RECORD.superseded.items()
    if name in _BC_TABLE_FILES
}

FETCH_COMMAND = "exozippy-fetch-bc-tables"

# (path, size, mtime_ns, inode) of tables already md5-verified this process.
# A full md5 is ~0.1 s per 25 MB; a fit reads a handful of tables, often
# more than once, so it is paid once per file per process.
_verified: set[tuple[str, int, int, int]] = set()


def published_tables() -> List[str]:
    """Filenames of every published BC table, sorted (not the F_bol
    table, which is no facility's)."""
    return sorted(n for n in _BC_TABLE_FILES if n.endswith(".bc.parquet"))


def ensure_fbol_table(model_root: Path | str | None = None) -> Path:
    """The per-node F_bol table (``FBOL_TABLE``), fetched on first use.

    Under the default model root it is fetched like a table and checked
    against its pin; an explicit ``model_root`` is the caller's own tree,
    used as given (``bc_grid.ensure_bc_tables``' rule), so a missing file
    there raises naming the path.  A pinned record version that does not
    carry the table predates the own-F_bol normalization (review 8.9.13)
    and raises: its BC tables are normalized by sigma Teff^4, and nothing
    may silently mix the two normalizations.
    """
    from ...components.sed.bc_grid import DEFAULT_MODEL_ROOT

    if model_root is not None and Path(model_root) != Path(DEFAULT_MODEL_ROOT):
        path = Path(model_root) / "NextGen" / "BCs" / FBOL_TABLE
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} is missing. An explicit model_root is used as "
                f"given; copy the published {FBOL_TABLE} (record "
                f"{record_url()}) there, or drop model_root to fetch it."
            )
        return path
    if FBOL_TABLE not in _BC_TABLE_FILES:
        raise RuntimeError(
            f"The pinned NextGen BC record ({record_url()}) has no "
            f"{FBOL_TABLE}: it predates the own-F_bol normalization "
            f"(review 8.9.13), so its tables are normalized by sigma "
            f"Teff^4. Pin the version that carries {FBOL_TABLE} in "
            f"utilities/zenodo_assets.py ('nextgen_bc_tables')."
        )
    return ensure_tables([FBOL_TABLE])[0]


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


def _check_present(path: Path, name: str) -> bool:
    """True when the table at `path` is byte-identical to the pin; False
    when it is a PREVIOUS published version of it (the caller re-fetches
    over it); raises on anything else -- a local edit is never overwritten.
    """
    meta = _BC_TABLE_FILES[name]
    st = path.stat()
    key = (str(path), st.st_size, st.st_mtime_ns, st.st_ino)
    if key in _verified:
        return True
    md5 = None
    if st.st_size != meta["size"] or (md5 := _md5(path)) != meta["md5"]:
        if (md5 or _md5(path)) in _SUPERSEDED.get(name, ()):
            logger.debug(
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
            f"It was most likely regenerated or extended locally "
            f"(generate_NextGen_BC_Tables.py and make_bc.py merge into it). "
            f"It is NOT overwritten, because that may be hours of "
            f"generation work. Either publish it as a new version of the "
            f"Zenodo record and update the pins in "
            f"utilities/zenodo_assets.py, or move it aside and re-run to "
            f"fetch the published table. To fit against a modified table "
            f"deliberately, copy the NextGen/ tree elsewhere and point the "
            f"SED's `model_root:` at the copy (an explicit root is used as "
            f"given and never fetched into)."
        )
    _verified.add(key)
    return True


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
            if not allow_local_changes and not _check_present(path, name):
                # A superseded version: a hard link into the machine cache
                # (the old md5's entry), so unlinking here leaves the cache
                # alone and the fetch below links or downloads the new one.
                path.unlink()
                missing.append(name)
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


def read_fbol_grid(path: Path | str, axes: dict):
    """log10 F_bol (erg/s/cm^2) on the (teff, logg, feh) box `axes`, an
    (n_teff, n_logg, n_feh) array, from the per-node table at `path`.
    Every node of the box must be in the table; a missing one raises
    naming it."""
    import numpy as np
    import pandas as pd

    from ...components.sed.bc_grid import _KEY_DECIMALS

    df = pd.read_parquet(path, columns=["teff", "logg", "feh", "fbol"])
    lookup = {
        (
            round(float(t), _KEY_DECIMALS),
            round(float(g), _KEY_DECIMALS),
            round(float(z), _KEY_DECIMALS),
        ): float(f)
        for t, g, z, f in zip(df.teff, df.logg, df.feh, df.fbol)
    }
    teffs, loggs, fehs = (
        [round(float(x), _KEY_DECIMALS) for x in axes[k]]
        for k in ("teff", "logg", "feh")
    )
    out = np.empty((len(teffs), len(loggs), len(fehs)), dtype=float)
    for i, t in enumerate(teffs):
        for j, g in enumerate(loggs):
            for k, z in enumerate(fehs):
                f = lookup.get((t, g, z))
                if f is None or not f > 0:
                    raise KeyError(
                        f"{path} has no positive F_bol for teff={t}, "
                        f"logg={g}, feh={z}, a node of the grid being "
                        f"interpolated."
                    )
                out[i, j, k] = np.log10(f)
    return out


def fetch_all(dest_dir: Path | str | None = None) -> List[Path]:
    """Fetch every published table, and the F_bol table when the pinned
    version carries it (pre-fetch for offline use)."""
    extra = [FBOL_TABLE] if FBOL_TABLE in _BC_TABLE_FILES else []
    return ensure_tables(published_tables() + extra, dest_dir=dest_dir)


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
