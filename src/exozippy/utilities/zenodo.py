"""Fetch-verify-cache core for the large data files we host on Zenodo.

Some of the data EXOZIPPy needs is far too large to ship in the package --
the NextGen model spectra (~250 MB) that synthesize bolometric corrections
for filters with no precomputed BC table, and the MIST EEP track grid
(~128 MB) the evolutionary model interpolates. Both are git-ignored, fetched
on first use, and cached in place.

This module owns the mechanics; the callers own their asset tables. It lives
under ``utilities/`` rather than inside a component because its two callers
(``components/sed/make_bc.py`` and ``models/MIST/eep_grid.py``) sit in
different trees and a cross-component import would be the wrong dependency.
Note it is deliberately NOT a registry utility: it has no ``build_parser`` /
``main`` pair and is never surfaced by ``Component.get_utilities()``.

Integrity is enforced, because the failure mode without it is silent and
lasting: urlretrieve happily writes a truncated body to the destination, and
every later run then reads that half-file. It surfaced once as
``pandas.errors.ParserError: EOF inside string starting at row 11248``, in a
test that has nothing to do with downloading. So:

* a cached file is size-checked on every call (cheap, and truncation -- the
  observed failure -- always changes the size);
* a download lands on a .part file, is checked for size AND md5, and only
  then atomically renamed into place, so an interrupted fetch can never be
  mistaken for a cached one;
* a corrupt cached file is re-fetched rather than raising, since the recovery
  is unambiguous.
"""

from __future__ import annotations

import hashlib
import logging
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Mapping

logger = logging.getLogger(__name__)

# Zenodo is a free academic host and 5xx-es under load. Four attempts with
# doubling backoff spans ~35s, which covers the transient gateway errors seen
# in CI without making a genuinely dead URL take minutes to report.
_DOWNLOAD_ATTEMPTS = 4
_RETRY_BACKOFF = 5  # seconds; doubles each attempt


def _md5(path: Path, chunk: int = 1 << 20) -> str:
    """Streaming md5 of a file (the spectra grid is ~250 MB)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def fetch_assets(
    assets: Mapping[str, Mapping[str, object]],
    dest_dir: Path | str,
    on_fetch: Callable[[str], None] | None = None,
) -> None:
    """Ensure every asset in `assets` is present and intact in `dest_dir`.

    Parameters
    ----------
    assets
        ``{filename: {"url": str, "size": int, "md5": str}}``. The size and
        md5 must come from the Zenodo record's own API
        (``https://zenodo.org/api/records/<id>``); they pin the content, so a
        re-uploaded or truncated file is caught rather than silently used.
    dest_dir
        Cache directory. Created if a fetch is actually needed; a call that
        finds everything cached touches nothing.
    on_fetch
        Called with the filename immediately before each real download, and
        never when the cache is warm. Callers use it for one-time warnings
        that only make sense when the asset is genuinely being fetched (see
        make_bc's downsampling warning). Keep it cheap and idempotent.

    Raises
    ------
    RuntimeError
        The attempt budget was exhausted (transport failures, truncation, or
        an md5 mismatch on every try). The message names the file and quotes
        the last error.
    urllib.error.HTTPError
        Re-raised immediately for any 4xx: a 404 means the URL or record is
        wrong, and retrying it just delays a real failure.
    """
    dest_dir = Path(dest_dir)
    for filename, meta in assets.items():
        dest = dest_dir / filename
        if dest.exists():
            actual = dest.stat().st_size
            if actual == meta["size"]:
                continue
            logger.warning(
                "Cached %s is %d bytes, expected %d -- it is truncated or "
                "stale. Re-downloading.",
                dest,
                actual,
                meta["size"],
            )
            dest.unlink()

        if on_fetch is not None:
            on_fetch(filename)

        logger.info(f"Downloading {filename} from Zenodo...")
        dest_dir.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".part")

        # Retried with backoff. Zenodo is a free academic host serving a 250 MB
        # file, and it returns 5xx under load -- observed as
        # `HTTPError: HTTP Error 504: Gateway Time-out` failing CI on pull
        # requests that had touched nothing related. A transient gateway error
        # should cost a few seconds, not a whole run.
        #
        # Only transport and integrity errors are retried. A 404 means the URL
        # or record is wrong, and retrying that just delays a real failure by
        # _RETRY_BACKOFF seconds, so it is re-raised at once.
        last_error = None
        for attempt in range(1, _DOWNLOAD_ATTEMPTS + 1):
            try:
                urllib.request.urlretrieve(meta["url"], part)
                size = part.stat().st_size
                if size != meta["size"]:
                    raise RuntimeError(
                        f"{filename} downloaded {size} bytes, expected "
                        f"{meta['size']}. The download was truncated; retry."
                    )
                digest = _md5(part)
                if digest != meta["md5"]:
                    raise RuntimeError(
                        f"{filename} has md5 {digest}, expected "
                        f"{meta['md5']}. The file on Zenodo may have been "
                        f"replaced, or the download was corrupted."
                    )
                part.replace(dest)  # atomic within the same directory
                last_error = None
                break
            except urllib.error.HTTPError as e:
                if e.code < 500:
                    raise
                last_error = e
            except (urllib.error.URLError, TimeoutError, RuntimeError) as e:
                last_error = e
            finally:
                part.unlink(missing_ok=True)

            if attempt < _DOWNLOAD_ATTEMPTS:
                delay = _RETRY_BACKOFF * 2 ** (attempt - 1)
                logger.warning(
                    "Download of %s failed (attempt %d/%d): %s. "
                    "Retrying in %ds.",
                    filename,
                    attempt,
                    _DOWNLOAD_ATTEMPTS,
                    last_error,
                    delay,
                )
                time.sleep(delay)

        if last_error is not None:
            raise RuntimeError(
                f"Could not download {filename} from Zenodo after "
                f"{_DOWNLOAD_ATTEMPTS} attempts: {last_error}"
            ) from last_error

        logger.info(f"Saved {filename} to {dest}")
