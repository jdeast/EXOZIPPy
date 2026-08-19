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

Machine-level shared cache
--------------------------
The destination directory is derived from the *source tree*
(``DEFAULT_MODEL_ROOT = source_code_dir / "models"``, ``EEP_GRID_DIR``), so
every git worktree and every checkout is a separate destination. During
development that means re-fetching a quarter-gigabyte from a free academic
host once per worktree, which is both slow and rude.

So a machine-level cache sits *behind* this function, keyed by the asset's
pinned md5, under ``$XDG_CACHE_HOME/exozippy/downloads`` (default
``~/.cache/exozippy/downloads``). It does not change where callers read
from: the destination is still populated, it is just populated by a hard
link from the cache instead of by a download. Properties:

* **Verified on adoption.** A cache entry is only published after the full
  size+md5 check, by atomic rename, and its md5 is re-checked in full every
  time it is linked into a new destination. A corrupt shared entry is
  discarded and re-fetched rather than propagated to every worktree at once
  -- a shared cache concentrates that risk, so it gets the stronger check.
* **Hard link, copy only if forced.** ``os.link`` first, so N worktrees cost
  one copy on disk and one file in the quota; any ``OSError`` (a
  cross-device ``EXDEV`` being the expected one) falls back to
  ``shutil.copyfile``. Entries are published mode 0444 so an in-place write
  through one of the links cannot corrupt the shared copy.
* **Concurrency-safe.** Fetches of the same asset serialize on an
  ``flock``-ed lock file next to the entry (auto-released if the holder
  dies), so parallel agents download it once; and even with no working lock
  the publish is an atomic rename of a fully verified file, so the worst
  case is duplicate work, never a torn entry.
* **Never fatal.** An unset home, an unwritable or full cache directory, a
  filesystem with no ``flock`` -- each degrades to exactly the old
  behaviour (download straight to the destination) with one warning.
* **Adopts what is already there.** A destination that already holds the
  asset at the pinned size and md5 is hard-linked *into* the cache, so an
  existing checkout seeds the machine cache for every later worktree with
  no network access at all.

Set ``EXOZIPPY_CACHE_DIR`` to relocate the cache (small home quota, shared
scratch), or to an empty string / ``none`` / ``off`` / ``0`` to switch it
off. Entries are keyed by content hash, so a re-uploaded asset simply gets a
new entry; the stale one is inert and safe to delete by hand.
"""

from __future__ import annotations

import contextlib
import errno
import hashlib
import logging
import os
import shutil
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Iterator, Mapping

try:  # POSIX only; the cache is correct without it, just less efficient.
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Zenodo is a free academic host and 5xx-es under load. Four attempts with
# doubling backoff spans ~35s, which covers the transient gateway errors seen
# in CI without making a genuinely dead URL take minutes to report.
_DOWNLOAD_ATTEMPTS = 4
_RETRY_BACKOFF = 5  # seconds; doubles each attempt

# Shared-cache configuration.
_CACHE_ENV = "EXOZIPPY_CACHE_DIR"
_CACHE_APP_DIR = "exozippy"
_CACHE_SUBDIR = "downloads"
_CACHE_OFF_VALUES = frozenset({"", "0", "off", "none", "false"})

# How long to wait for another process to finish fetching the same asset
# before giving up on the lock and fetching it ourselves. Generous, because
# what we are waiting on is a 250 MB download on someone else's link; and
# proceeding without the lock is safe (see _publish), only wasteful.
_LOCK_TIMEOUT = 900.0  # seconds
_LOCK_POLL = 0.5  # seconds

# Set once, per process, the first time the cache proves unusable. Every
# later call then goes straight down the old download-to-destination path
# instead of re-discovering the same broken directory (and re-warning).
_cache_disabled_reason: str | None = None

# (destination, md5) pairs whose adoption into the cache has already been
# attempted this process. Adoption costs a full md5 of a file we are
# otherwise happy to trust on its size alone, so it happens at most once.
_adoption_attempted: set[tuple[str, str]] = set()


def _md5(path: Path, chunk: int = 1 << 20) -> str:
    """Streaming md5 of a file (the spectra grid is ~250 MB)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


# --- the machine-level shared cache ---------------------------------------


def shared_cache_root() -> Path | None:
    """Where the machine-level cache lives, or None if it is switched off.

    ``EXOZIPPY_CACHE_DIR`` wins; otherwise the XDG convention
    (``$XDG_CACHE_HOME/exozippy``, falling back to ``~/.cache/exozippy``).
    Nothing is created here -- this is a pure path computation, so callers
    and tests can ask the question without side effects.

    XDG is a POSIX convention with no Windows equivalent: there is no
    ``XDG_CACHE_HOME`` there and the fallback would put a Unix-shaped
    ``~/.cache`` in a Windows home directory.  Native Windows wants
    ``%LOCALAPPDATA%``, i.e. a ``sys.platform`` branch rather than another
    fallback.  Not done, because native Windows does not work for larger
    reasons (the samplers need ``fork``); logged with the rest of them in
    ``docs/windows-native.md``, and ``EXOZIPPY_CACHE_DIR`` overrides all of
    this in the meantime.
    """
    override = os.environ.get(_CACHE_ENV)
    if override is not None:
        if override.strip().lower() in _CACHE_OFF_VALUES:
            return None
        return Path(override.strip()).expanduser()

    xdg = os.environ.get("XDG_CACHE_HOME", "").strip()
    try:
        base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    except RuntimeError:  # pragma: no cover - no home directory at all
        return None
    return base / _CACHE_APP_DIR


def _disable_cache(reason: str) -> None:
    """Latch the cache off for this process, warning once."""
    global _cache_disabled_reason
    if _cache_disabled_reason is None:
        _cache_disabled_reason = reason
        logger.warning(
            "Shared download cache unavailable (%s). Falling back to "
            "downloading into the destination directory; set %s to a "
            "writable path to re-enable it.",
            reason,
            _CACHE_ENV,
        )


def _cache_dir() -> Path | None:
    """The cache directory, created and writable, or None if unusable."""
    if _cache_disabled_reason is not None:
        return None
    root = shared_cache_root()
    if root is None:
        return None
    path = root / _CACHE_SUBDIR
    try:
        path.mkdir(parents=True, exist_ok=True)
        if not os.access(path, os.W_OK | os.X_OK):
            raise PermissionError(f"{path} is not writable")
    except OSError as e:
        _disable_cache(str(e))
        return None
    return path


def _entry_path(
    cache: Path, filename: str, meta: Mapping[str, object]
) -> Path:
    """Cache path for an asset: content-addressed, but still readable.

    Keyed by the pinned md5 so two assets can never collide and a
    re-uploaded file gets a fresh entry; suffixed with the filename so the
    directory can be understood by a human with ``ls``.
    """
    return cache / f"{meta['md5']}-{Path(filename).name}"


@contextlib.contextmanager
def _entry_lock(entry: Path) -> Iterator[None]:
    """Serialize fetches of one asset across processes, best effort.

    An exclusive ``flock`` on ``<entry>.lock``, polled so a stuck holder
    cannot wedge a fit forever. The kernel drops the lock when the holder
    exits, crashed or not, so there is no stale-lock recovery to get wrong.
    Losing the lock (no fcntl, a filesystem that ignores it, the timeout) is
    not an error: the publish path is safe without it, and the only cost is
    a duplicated download.
    """
    if fcntl is None:
        yield
        return

    lock_path = entry.with_name(entry.name + ".lock")
    fd = None
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
        deadline = time.monotonic() + _LOCK_TIMEOUT
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as e:
                if e.errno not in (
                    errno.EACCES,
                    errno.EAGAIN,
                    errno.EWOULDBLOCK,
                ):
                    # Not "somebody else holds it" but "this filesystem does
                    # not do locks" (ENOLCK, EOPNOTSUPP, ...). Polling that
                    # for 15 minutes would be a hang, so proceed unlocked.
                    logger.debug("flock unsupported on %s (%s)", lock_path, e)
                    break
                if time.monotonic() >= deadline:
                    logger.warning(
                        "Waited %.0fs for another process to fetch %s; "
                        "fetching it ourselves.",
                        _LOCK_TIMEOUT,
                        entry.name,
                    )
                    break
                time.sleep(_LOCK_POLL)
    except OSError as e:
        logger.debug("No lock on %s (%s); proceeding unlocked.", entry, e)

    try:
        yield
    finally:
        if fd is not None:
            with contextlib.suppress(OSError):
                fcntl.flock(fd, fcntl.LOCK_UN)
            with contextlib.suppress(OSError):
                os.close(fd)


def _entry_is_intact(entry: Path, meta: Mapping[str, object]) -> bool:
    """Full size+md5 verification of a shared-cache entry.

    Deliberately stricter than the destination's size-only check: one
    corrupt entry here would be linked into every worktree on the machine,
    and the md5 is read once per destination, not once per run.
    """
    try:
        if entry.stat().st_size != meta["size"]:
            logger.warning(
                "Shared cache entry %s is the wrong size; discarding it.",
                entry,
            )
            return False
        if _md5(entry) != meta["md5"]:
            logger.warning(
                "Shared cache entry %s fails its md5 check; discarding it.",
                entry,
            )
            return False
    except OSError as e:
        logger.warning("Cannot read shared cache entry %s: %s", entry, e)
        return False
    return True


def _discard_entry(entry: Path) -> None:
    """Remove a cache entry that failed verification, best effort."""
    with contextlib.suppress(OSError):
        entry.unlink(missing_ok=True)


def _publish(source: Path, entry: Path) -> bool:
    """Move a verified file into the cache as `entry`.

    `source` must already have passed size+md5 and must live in the cache
    directory, so the rename is atomic: a concurrent reader sees either no
    entry or a complete, verified one, never a partial write. Two processes
    publishing the same asset simply overwrite identical bytes.
    """
    try:
        with contextlib.suppress(OSError):
            os.chmod(source, 0o444)  # a shared copy is nobody's scratch file
        os.replace(source, entry)
        return True
    except OSError as e:
        logger.warning(
            "Could not populate the shared cache at %s: %s", entry, e
        )
        return False


def _link_into(entry: Path, dest: Path, meta: Mapping[str, object]) -> bool:
    """Materialize `entry` at `dest`; hard link if possible, else copy.

    Staged under a temporary name in the destination directory and renamed,
    for the same reason the download is: a half-copied file must never be
    reachable at the destination path. A hard link is the same inode as the
    already-verified entry and so is intact by construction; a copy is fresh
    bytes and gets the same md5 check a fresh download would have got.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=dest.parent, prefix=dest.name + ".", suffix=".link"
    )
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        tmp.unlink()  # os.link refuses an existing target
        try:
            os.link(entry, tmp)
            how = "hard link"
        except OSError as e:
            # EXDEV (cache and destination on different filesystems) is the
            # expected case; EPERM/EMLINK/EOPNOTSUPP happen on some network
            # and container filesystems. All of them mean the same thing
            # here: we have to spend the disk.
            logger.debug(
                "Hard link %s -> %s failed (%s); copying.", entry, tmp, e
            )
            shutil.copyfile(entry, tmp)
            how = f"copy ({e.strerror or e})"
            if _md5(tmp) != meta["md5"]:
                raise OSError(f"the copy of {entry} at {tmp} is corrupt")
        if tmp.stat().st_size != meta["size"]:
            raise OSError(f"{tmp} is short after {how}")
        os.replace(tmp, dest)
        if how == "hard link":
            logger.info(
                "Linked %s from the shared cache (hard link).", dest.name
            )
        else:
            # Worth saying: st_dev is not a reliable test here (an automounted
            # NFS home can report the same st_dev as a local disk while still
            # refusing the link), so the only honest detection is to try it --
            # and the user is the one who can fix it, by co-locating the cache.
            logger.info(
                "Copied %s from the shared cache (%s). Set %s to a directory "
                "on the same filesystem as %s to hard-link instead, so every "
                "checkout shares one copy on disk.",
                dest.name,
                how,
                _CACHE_ENV,
                dest.parent,
            )
        return True
    except OSError as e:
        logger.warning(
            "Could not materialize %s from the shared cache: %s", dest, e
        )
        return False
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)


def _adopt_destination(
    dest: Path, filename: str, meta: Mapping[str, object]
) -> None:
    """Seed the cache from a destination that already holds the asset.

    An existing checkout typically has these files already. Verifying one
    locally (an md5, once per process) is far cheaper than re-downloading a
    quarter-gigabyte, so the first run in such a tree populates the machine
    cache for every later worktree with no network access at all.
    """
    cache = _cache_dir()
    if cache is None:
        return
    entry = _entry_path(cache, filename, meta)
    key = (str(dest), str(meta["md5"]))
    if key in _adoption_attempted or entry.exists():
        return
    _adoption_attempted.add(key)

    if _md5(dest) != meta["md5"]:
        logger.warning(
            "%s is the pinned size but not the pinned md5. Leaving it alone "
            "(it is what this tree has always used) but NOT adopting it into "
            "the shared cache. Delete it to force a clean re-download.",
            dest,
        )
        return

    with _entry_lock(entry):
        if entry.exists():
            return
        fd, tmp_name = tempfile.mkstemp(
            dir=cache, prefix=Path(filename).name + ".", suffix=".adopt"
        )
        os.close(fd)
        tmp = Path(tmp_name)
        try:
            tmp.unlink()
            try:
                os.link(dest, tmp)
            except OSError:
                # Same rule as _link_into: a copy is new bytes, so it is
                # re-verified before it can become the entry every worktree
                # on this machine will link to.
                shutil.copyfile(dest, tmp)
                if _md5(tmp) != meta["md5"]:
                    raise OSError(f"the copy of {dest} at {tmp} is corrupt")
            if _publish(tmp, entry):
                logger.info(
                    "Adopted the existing %s into the shared cache at %s.",
                    dest.name,
                    entry,
                )
        except OSError as e:
            logger.warning(
                "Could not adopt %s into the shared cache: %s", dest, e
            )
        finally:
            with contextlib.suppress(OSError):
                tmp.unlink(missing_ok=True)


# --- downloading -----------------------------------------------------------


def _download_verified(
    meta: Mapping[str, object], filename: str, part: Path
) -> None:
    """Download `meta["url"]` onto `part`, verified, with retries.

    On return `part` exists and has the pinned size and md5. On failure it
    does not exist at all.

    Retried with backoff. Zenodo is a free academic host serving a 250 MB
    file, and it returns 5xx under load -- observed as
    `HTTPError: HTTP Error 504: Gateway Time-out` failing CI on pull
    requests that had touched nothing related. A transient gateway error
    should cost a few seconds, not a whole run.

    Only transport and integrity errors are retried. A 404 means the URL or
    record is wrong, and retrying that just delays a real failure by
    _RETRY_BACKOFF seconds, so it is re-raised at once.
    """
    last_error: BaseException | None = None
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
            return
        except urllib.error.HTTPError as e:
            if e.code < 500:
                part.unlink(missing_ok=True)
                raise
            last_error = e
        except (urllib.error.URLError, TimeoutError, RuntimeError) as e:
            last_error = e
        part.unlink(missing_ok=True)

        if attempt < _DOWNLOAD_ATTEMPTS:
            delay = _RETRY_BACKOFF * 2 ** (attempt - 1)
            logger.warning(
                "Download of %s failed (attempt %d/%d): %s. Retrying in %ds.",
                filename,
                attempt,
                _DOWNLOAD_ATTEMPTS,
                last_error,
                delay,
            )
            time.sleep(delay)

    raise RuntimeError(
        f"Could not download {filename} from Zenodo after "
        f"{_DOWNLOAD_ATTEMPTS} attempts: {last_error}"
    ) from last_error


def _fetch_via_cache(
    filename: str,
    meta: Mapping[str, object],
    dest: Path,
    on_fetch: Callable[[str], None] | None,
) -> bool:
    """Populate `dest` through the machine-level cache.

    Returns False if the cache could not serve or be populated, in which
    case the caller falls back to downloading straight to the destination
    (i.e. to the behaviour that predates the cache).
    """
    cache = _cache_dir()
    if cache is None:
        return False
    entry = _entry_path(cache, filename, meta)

    # Unlocked fast path: a warm cache costs one stat plus one md5.
    if entry.exists():
        if _entry_is_intact(entry, meta) and _link_into(entry, dest, meta):
            return True
        _discard_entry(entry)

    with _entry_lock(entry):
        # Re-check: whoever held the lock may have just fetched it for us.
        if entry.exists():
            if _entry_is_intact(entry, meta) and _link_into(entry, dest, meta):
                return True
            _discard_entry(entry)

        if on_fetch is not None:
            on_fetch(filename)
        logger.info(f"Downloading {filename} from Zenodo...")

        fd, tmp_name = tempfile.mkstemp(
            dir=cache, prefix=Path(filename).name + ".", suffix=".part"
        )
        os.close(fd)
        part = Path(tmp_name)
        part.unlink()  # urlretrieve creates it; keep the name reserved only
        try:
            _download_verified(meta, filename, part)
            if not _publish(part, entry):
                # The download is good but the cache would not take it; use
                # it rather than throwing away a 250 MB transfer.
                return _link_into(part, dest, meta) or _move_into(part, dest)
        finally:
            with contextlib.suppress(OSError):
                part.unlink(missing_ok=True)

        return _link_into(entry, dest, meta)


def _move_into(source: Path, dest: Path) -> bool:
    """Last-ditch: move a verified download to the destination."""
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(dest))
        return True
    except OSError as e:  # pragma: no cover - both link and move failing
        logger.warning("Could not place %s: %s", dest, e)
        return False


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
        Destination directory -- unchanged by the shared cache, which sits
        behind this function and only changes where the bytes come from.
        Created if a fetch is actually needed; a call that finds everything
        cached touches nothing.
    on_fetch
        Called with the filename immediately before each real download, and
        never when either cache is warm. Callers use it for one-time
        warnings that only make sense when the asset is genuinely being
        fetched (see make_bc's downsampling warning). Keep it cheap and
        idempotent.

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
                _adopt_destination(dest, filename, meta)
                continue
            logger.warning(
                "Cached %s is %d bytes, expected %d -- it is truncated or "
                "stale. Re-downloading.",
                dest,
                actual,
                meta["size"],
            )
            dest.unlink()

        if _fetch_via_cache(filename, meta, dest, on_fetch):
            logger.info(f"Saved {filename} to {dest}")
            continue

        # No usable shared cache: download straight into the destination,
        # exactly as this function did before the cache existed.
        if on_fetch is not None:
            on_fetch(filename)

        logger.info(f"Downloading {filename} from Zenodo...")
        dest_dir.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".part")
        try:
            _download_verified(meta, filename, part)
            part.replace(dest)  # atomic within the same directory
        finally:
            part.unlink(missing_ok=True)

        logger.info(f"Saved {filename} to {dest}")
