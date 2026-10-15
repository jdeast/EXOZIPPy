"""Fill and trim the machine-level Zenodo download cache for CI.

CI keeps ONE actions/cache entry: the machine cache directory that
``exozippy.utilities.zenodo`` fills (``$EXOZIPPY_CACHE_DIR/downloads``),
keyed on the hash of ``utilities/zenodo_assets.py``. Its files are named
``<md5>-<filename>``, so restoring it can never overwrite a tracked file, and
a new pin is a new name by construction. See the cache steps in
``.github/workflows/tests.yml`` and docs/testing.md ("Zenodo data in CI").

    python scripts/zenodo_ci_cache.py fetch   # every record, via its owner
    python scripts/zenodo_ci_cache.py prune   # drop what no pin names

``fetch`` is ``zenodo.fetch_every_record()``: the same call the test suite's
prewarm makes, run as its own step so its success decides whether the entry
is complete enough to save. It exits non-zero if any record failed.

``prune`` deletes every file in the cache directory that is not the entry
of a currently pinned file -- the superseded versions a restore-keys
fallback carried in, and the ``.lock`` / ``.part`` / ``.adopt`` debris --
so the entry saved under a new key does not grow by one stale file per pin
change. It touches nothing outside that one directory, and refuses to run
when the cache is switched off.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _cache_downloads() -> Path:
    from exozippy.utilities import zenodo

    root = zenodo.shared_cache_root()
    if root is None:
        raise SystemExit(
            "The machine cache is switched off (EXOZIPPY_CACHE_DIR); there "
            "is nothing to prune."
        )
    return root / zenodo._CACHE_SUBDIR


def current_entry_names() -> set[str]:
    """``<md5>-<filename>`` for every file of every registered record."""
    from exozippy.utilities import zenodo_assets

    return {
        f"{pin.md5}-{Path(filename).name}"
        for rec in zenodo_assets.RECORDS.values()
        for filename, pin in rec.files.items()
    }


def prune(downloads: Path) -> list[Path]:
    """Delete every non-current file in `downloads`; return what went."""
    if not downloads.is_dir():
        return []
    keep = current_entry_names()
    removed = []
    for path in sorted(downloads.iterdir()):
        if path.is_file() and path.name not in keep:
            path.unlink()
            removed.append(path)
    return removed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("action", choices=("fetch", "prune"))
    args = parser.parse_args(argv)

    if args.action == "fetch":
        # The fetch core's INFO lines, and only those (not every library's),
        # so the CI log says per file whether it was linked from the
        # restored cache or downloaded from Zenodo.
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("zenodo: %(message)s"))
        core = logging.getLogger("exozippy.utilities.zenodo")
        core.addHandler(handler)
        core.setLevel(logging.INFO)
        core.propagate = False

        from exozippy.utilities import zenodo

        zenodo.fetch_every_record()
        print("every registered Zenodo record is in place")
        return 0

    downloads = _cache_downloads()
    for path in prune(downloads):
        print(f"pruned {path.name}")
    kept = (
        sorted(p.name for p in downloads.iterdir())
        if downloads.is_dir()
        else []
    )
    total = sum((downloads / n).stat().st_size for n in kept)
    print(f"{downloads}: {len(kept)} entries, {total / 1e6:.1f} MB")
    for name in kept:
        print(f"  {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
