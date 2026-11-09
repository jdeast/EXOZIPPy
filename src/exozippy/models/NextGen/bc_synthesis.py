"""On-the-fly synthesis of a NextGen BC column the published tables lack.

The published tables (``bc_tables.py``) carry the filters someone asked for.
A fit whose ``band:`` filter is not among them, and whose band's SED flux
some component READS (``Component.sed_flux_bands``: a microlensing zeropoint
tie, a transit dilution, an astrometric flux fraction), used to be refused.
Since the processed full-resolution spectra are published too (step 1 of
``README.md``: the R = 20000 BT-NextGen spectra, flux as float32, Zenodo
record ``nextgen_spectra_r20000`` in ``utilities/zenodo_assets.py``), the
column can instead be computed by the SAME step-2 code that made the
published ones (``generate_NextGen_BC_Tables.generate_bc_tables``), for just
that filter, and kept for this user.  BCs recomputed from those spectra match
the published tables to 8e-10 mag, so a synthesized column is as good as a
published one -- unlike the R = 150 synthesis that was removed in 2026-09
(~2 percent wrong; ``components/sed/sed.md``).

What happens, once per (filter, user):

1. ``synthesize_column`` takes a file lock for the (facility, filter) pair,
   so two fits never compute the same column at once: the second WAITS, then
   finds the column current and reads it.
2. It fetches the processed spectra on demand (3.3 GB; the record is listed
   in ``zenodo_assets.ON_DEMAND``, so no prewarm or CI cache ever fetches it)
   into ``<machine cache>/nextgen_spectra_r20000`` -- through
   ``utilities/zenodo.fetch_assets`` and the md5-keyed machine cache, so a
   second checkout costs a hard link.
3. It runs step 2 for that one filter into a private scratch model root
   (the grid yaml copied in), including step 2's padding of the grid nodes
   no spectrum exists for -- which nodes those are is read from the bundle
   itself (a complete step-1 product: a node it lacks is one SVO has no
   spectrum for), so no SVO index and no other network is needed.
4. It merges the column -- with the published pipeline's own GENERATOR /
   SPECTRA_TAG provenance and its padding record -- into the per-user cache
   table ``<machine cache>/bc_synthesized/NextGen/BCs/{FACILITY}.bc.parquet``
   (``bc_grid.synthesized_model_root``), and its native magnitude system and
   m_AB - m_Vega offset, computed exactly as ``write_magsys_table`` computes
   a published column's, into that root's ``NextGen.magsys.csv``.  Each merge
   holds a lock of its own, because two filters of one facility share a
   table and every facility shares the magsys record.

The package's published tables are never written: they are pinned by md5
(``bc_tables.py``), and a fit that edited them would then refuse them.
``bc_grid.locate_bc_column`` reads a cached column exactly as a published
one, and the published table wins if it ever carries the same column.  A
cached column counts only while CURRENT (``cached_column_is_current``: this
pipeline's provenance, every grid-yaml node covered); a stale one is
recomputed in full.

Nothing here downloads at import.  ``fetch_all`` pre-fetches the spectra by
hand, for a machine that will synthesize while offline.
"""

from __future__ import annotations

import contextlib
import logging
import shutil
import tempfile
import time
from pathlib import Path
from typing import Iterator, List, Sequence

import pandas as pd
import yaml

from ...utilities import zenodo, zenodo_assets

try:
    import fcntl
except ImportError:  # pragma: no cover
    # native Windows (docs/windows-native.md)
    fcntl = None

logger = logging.getLogger(__name__)

MODEL = "NextGen"

# The record (utilities/zenodo_assets.py) and where its files are put, under
# the machine cache root (zenodo.shared_cache_root).
RECORD = "nextgen_spectra_r20000"
SPECTRA_SUBDIR = "nextgen_spectra_r20000"

# How often a waiting fit re-tries a held lock, and how often it says so.
_LOCK_POLL = 2.0
_LOCK_REPORT = 300.0


def _cache_root_or_raise(what: str) -> Path:
    root = zenodo.shared_cache_root()
    if root is None:
        raise RuntimeError(
            f"{what} needs the per-user EXOZIPPy cache, and it is switched "
            f"off (EXOZIPPY_CACHE_DIR is set to an 'off' value).  Point "
            f"EXOZIPPY_CACHE_DIR at a writable directory, or use a filter "
            f"the published NextGen BC tables cover."
        )
    return root


def spectra_dir() -> Path:
    """Where the processed full-resolution spectra are put."""
    return _cache_root_or_raise("Fetching the R = 20000 NextGen spectra") / (
        SPECTRA_SUBDIR
    )


def _spectra_files() -> List[str]:
    """Every file of the record a synthesis reads (not MANIFEST.txt)."""
    return sorted(
        f for f in zenodo_assets.record(RECORD).files if f != "MANIFEST.txt"
    )


def fetch_all(dest_dir: Path | str | None = None) -> List[Path]:
    """Fetch every processed spectrum (3.3 GB) -- by hand, for a machine
    that will synthesize offline.  Never called by the prewarm
    (zenodo_assets.ON_DEMAND)."""
    dest = Path(dest_dir) if dest_dir is not None else spectra_dir()
    names = _spectra_files()
    zenodo.fetch_assets(zenodo_assets.assets(RECORD, names), dest)
    return [dest / n for n in names]


def fetch_spectra(fehs: Sequence[float]) -> Path:
    """Fetch the processed spectra of these [Fe/H] values; returns their
    directory.  A [Fe/H] the record does not carry raises KeyError naming
    the file (zenodo_assets.file_pin)."""
    from .generate_NextGen_BC_Tables import _processed_filename

    dest = spectra_dir()
    names = [_processed_filename(float(f)) for f in fehs]
    zenodo.fetch_assets(zenodo_assets.assets(RECORD, names), dest)
    return dest


def _download_bytes(fehs: Sequence[float]) -> int:
    from .generate_NextGen_BC_Tables import _processed_filename

    rec = zenodo_assets.record(RECORD)
    return sum(rec.file(_processed_filename(float(f))).size for f in fehs)


def target_grid() -> dict:
    """The (teff, logg, feh, av) axes every column is built on: the
    package's NextGen.grid.yaml, the published tables' own grid."""
    from ...components.sed.bc_grid import DEFAULT_MODEL_ROOT
    from .generate_NextGen_BC_Tables import read_grid_yaml

    return read_grid_yaml(DEFAULT_MODEL_ROOT)


def cached_column_is_current(path: Path | str, column: str) -> bool:
    """Is `column` of the cached table `path` usable as it stands?

    Only if the step-2 generator wrote it from the published spectra (its
    filter_meta carries GENERATOR and SPECTRA_TAG, exactly the test step 2
    applies before reusing a published column) and it has a value at every
    node of the grid yaml.  Otherwise it is recomputed, not read: a grid
    extended since the column was cached, or a newer pipeline, must not
    leave a fit on an old column.
    """
    from ...components.sed.bc_grid import (
        bc_nodes_to_compute,
        bc_table_filter_columns,
    )
    from .generate_NextGen_BC_Tables import _reusable

    path = Path(path)
    if not path.is_file() or column not in bc_table_filter_columns(path):
        return False
    todo = bc_nodes_to_compute(path, target_grid(), [column], _reusable)
    return len(todo) == 0


def unavailable_nodes(processed_path: Path, grid: dict) -> List[tuple]:
    """The (teff, logg, feh) nodes of `grid` the processed bundle lacks.

    The published bundle is a complete step-1 product (step 1 resamples
    every node SVO lists and skips only the ones it does not), so a node it
    lacks is a node with no spectrum, which step 2 pads.  Only the teff and
    logg columns are read.
    """
    from .generate_NextGen_BC_Tables import _node_key, _processed_filename

    out = []
    for feh in grid["feh"]:
        df = pd.read_parquet(
            Path(processed_path) / _processed_filename(float(feh)),
            columns=["teff", "logg"],
        )
        have = {_node_key(t, g) for t, g in zip(df["teff"], df["logg"])}
        for logg in grid["logg"]:
            for teff in grid["teff"]:
                if _node_key(teff, logg) not in have:
                    out.append((float(teff), float(logg), float(feh)))
    return out


@contextlib.contextmanager
def _exclusive(lock_path: Path, what: str) -> Iterator[None]:
    """Hold an exclusive flock on `lock_path` for the block.

    Blocking with no timeout: the holder is computing what this process is
    about to read, and the kernel drops the lock if the holder dies, so a
    waiter can neither deadlock on a crash nor give up and duplicate the
    work.  It says so at INFO every few minutes while it waits.
    """
    if fcntl is None:  # pragma: no cover - native Windows only
        logger.warning(
            f"No file locking on this platform; {what} is not serialized "
            f"against another fit."
        )
        yield
        return
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = open(lock_path, "a+")
    try:
        start = time.monotonic()
        reported = None
        while True:
            try:
                fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                now = time.monotonic()
                if reported is None or now - reported >= _LOCK_REPORT:
                    logger.info(
                        f"Waiting for another fit to finish {what} "
                        f"(lock {lock_path}; {now - start:.0f} s so far)."
                    )
                    reported = now
                time.sleep(_LOCK_POLL)
        yield
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(fd.fileno(), fcntl.LOCK_UN)
        fd.close()


def _generate(filter_sets, processed_path, model_root, unavailable, natives):
    """Step 2 for these filters into `model_root` (the seam tests replace
    when they need no real computation)."""
    from .generate_NextGen_BC_Tables import generate_bc_tables

    return generate_bc_tables(
        filter_sets=filter_sets,
        processed_path=processed_path,
        model_root=model_root,
        save=True,
        unavailable=unavailable,
        native_systems=natives,
    )


def _merge_padding(old: list, new: list) -> list:
    """Union of two tables' padding records, one per node, with the union
    of the columns each node was padded for."""
    key = lambda p: tuple(
        round(float(p[k]), 6) for k in ("teff", "logg", "feh")
    )
    out = {key(p): dict(p) for p in old}
    for p in new:
        k = key(p)
        if k in out:
            cols = list(
                dict.fromkeys(
                    list(out[k].get("columns", []))
                    + list(p.get("columns", []))
                )
            )
            out[k] = {**out[k], **p, "columns": cols}
        else:
            out[k] = dict(p)
    return list(out.values())


def _merge_into_cache(
    work_root: Path, synth_root: Path, fac, col, svo, native
):
    """Move the freshly computed column from the scratch root into the
    per-user cache table and record, each under its own lock."""
    from ...components.sed.bc_grid import (
        BC_PARAM_COLS,
        bc_table_path,
        read_bc_meta,
        read_bc_table,
        write_bc_table,
    )
    from .generate_NextGen_BC_Tables import write_magsys_table

    src = bc_table_path(work_root, MODEL, fac)
    df = read_bc_table(src)
    meta = df.attrs.get("meta", {})
    filter_meta = meta.get("filters", {})
    if col not in df.columns or col not in filter_meta:
        raise RuntimeError(
            f"Step 2 wrote no column {col} for {svo} into {src}; nothing to "
            f"cache."
        )
    final = bc_table_path(synth_root, MODEL, fac)
    tables = final.parent
    with _exclusive(tables / f".{fac}.bc.parquet.lock", f"the {fac} table"):
        old = read_bc_meta(final).get("padding", []) if final.is_file() else []
        padding = _merge_padding(old, meta.get("padding", []))
        table_meta = {
            k: v
            for k, v in meta.items()
            if k not in ("filters", "padding", "version")
        }
        if padding:
            table_meta["padding"] = padding
        write_bc_table(
            df[BC_PARAM_COLS + [col]],
            final,
            {col: filter_meta[col]},
            table_meta=table_meta,
            allow_new_nodes=True,
            replace_columns=[col],
        )
    with _exclusive(tables / ".magsys.lock", "the magnitude-system record"):
        write_magsys_table({fac: [svo]}, synth_root, {svo: native})
    return final


def synthesize_column(svo_id: str, native_system: str, where: str) -> Path:
    """Make sure the per-user cache holds a CURRENT BC column for `svo_id`;
    returns the cache table that holds it.

    `native_system` ("Vega" or "AB") is the filter's native magnitude system
    for its record (the SED requires the band to state it unless the filter
    is in NATIVE_SYSTEM); `where` names the requester for the log.  Waits on
    the (facility, filter) lock if another fit is computing the same
    column, and computes nothing if that fit left it current.
    """
    from ...components.sed.bc_grid import (
        DEFAULT_MODEL_ROOT,
        _load_alias_table,
        bc_table_path,
        facility_from_svo_name,
        resolve_filter_name,
        synthesized_model_root,
    )

    if native_system not in ("Vega", "AB"):
        raise ValueError(
            f"{where}: the native magnitude system of a synthesized column "
            f"must be 'Vega' or 'AB'; got {native_system!r}."
        )
    synth_root = synthesized_model_root(DEFAULT_MODEL_ROOT, MODEL)
    if synth_root is None:
        _cache_root_or_raise(f"{where}: synthesizing a BC column for {svo_id}")
    alias_df = _load_alias_table()
    col = resolve_filter_name(svo_id, alias_df, alias="MIST")
    fac = facility_from_svo_name(svo_id)
    final = bc_table_path(synth_root, MODEL, fac)
    tables = final.parent
    tables.mkdir(parents=True, exist_ok=True)

    with _exclusive(
        tables / f".{fac}.{col}.synth.lock", f"the BC column {col} ({svo_id})"
    ):
        if cached_column_is_current(final, col):
            logger.debug(
                f"{where}: BC column {col} ({svo_id}) was synthesized by "
                f"another fit while this one waited; reading it from {final}."
            )
            return final
        grid = target_grid()
        fehs = sorted(float(f) for f in grid["feh"])
        dest = spectra_dir()
        logger.info(
            f"{where}: synthesizing the NextGen BC column {col} ({svo_id}) "
            f"from the R = 20000 spectra ({zenodo_assets.record(RECORD).doi}); "
            f"any of the {len(fehs)} spectra files not yet in {dest} are "
            f"downloaded first ({_download_bytes(fehs) / 1e9:.1f} GB in all)."
        )
        t0 = time.monotonic()
        processed = fetch_spectra(fehs)
        work = Path(tempfile.mkdtemp(prefix=f".{fac}.{col}.", dir=tables))
        try:
            # The scratch root's grid IS the target grid, so step 2 plans
            # every node of it and nothing else.
            grid_yaml = work / MODEL / "BCs" / f"{MODEL}.grid.yaml"
            grid_yaml.parent.mkdir(parents=True)
            grid_yaml.write_text(
                yaml.safe_dump(
                    {
                        "grid": {
                            k: [float(x) for x in v] for k, v in grid.items()
                        }
                    }
                )
            )
            _generate(
                {fac: [svo_id]},
                processed,
                work,
                unavailable_nodes(processed, grid),
                {svo_id: native_system},
            )
            _merge_into_cache(
                work, synth_root, fac, col, svo_id, native_system
            )
        finally:
            shutil.rmtree(work, ignore_errors=True)
        if not cached_column_is_current(final, col):
            raise RuntimeError(
                f"{where}: synthesizing BC column {col} ({svo_id}) left it "
                f"incomplete in {final}; it does not cover the grid yaml."
            )
    # WARNING, not INFO: the fit now reads a bolometric correction that is
    # not in the published, centrally checked tables -- worth seeing in the
    # log of every run that made one, even though it is computed exactly as
    # the published columns are.  Later fits that only READ it say so at
    # INFO (sed.py) and in the paper draft.
    logger.warning(
        f"{where}: the published NextGen BC tables have no column for "
        f"{svo_id}; synthesized {col} from the full-resolution spectra "
        f"(native system {native_system}) in {time.monotonic() - t0:.0f} s "
        f"and cached it at {final} for later fits."
    )
    return final
