import logging
import os
import sys

_OWNER_ATTR = "_exozippy_role"
"""Marks the handlers this module owns, so it never touches anyone else's.

`setup_logging` re-points its own console/file handlers on every call, and
clearing `log.handlers` wholesale would silently drop handlers something else
attached -- pytest's caplog, the GUI's, a user's.  Ownership is stamped on the
handler instance rather than inferred from its type: a foreign `FileHandler`
on the exozippy logger is somebody else's file, not ours to close.
"""


class _ColorFormatter(logging.Formatter):
    _COLORS = {
        logging.WARNING: "\033[93m",
        logging.ERROR: "\033[91m",
        logging.CRITICAL: "\033[91m",
    }
    _RESET = "\033[0m"

    def format(self, record):
        color = self._COLORS.get(record.levelno, "")
        msg = super().format(record)
        return f"{color}{msg}{self._RESET}" if color else msg


def _owned_handler(log, role):
    """The handler this module installed for `role`, or None.

    Any duplicate (which should not happen, but a caller could have copied a
    handler onto the logger) is removed and closed so the role stays single.
    """
    ours = [h for h in log.handlers if getattr(h, _OWNER_ATTR, None) == role]
    for extra in ours[1:]:
        log.removeHandler(extra)
        extra.close()
    return ours[0] if ours else None


def setup_logging(prefix, level="INFO"):
    """Configure the exozippy logger: console at `level`, file at DEBUG always.

    Safe to call more than once in one process -- which is what a script
    driving several fits through `run.run_fit(config, user_params=<dict>)`,
    the documented in-memory entry point, does.  A call naming a NEW prefix
    re-points the file handler at `<prefix>.log` (closing the old file) and
    applies the new console level; a call naming the SAME prefix is a no-op
    for the file, so a log is never truncated out from under a run in
    progress and handlers never accumulate.
    """
    log = logging.getLogger("exozippy")
    log.setLevel(logging.DEBUG)

    console_level = getattr(logging, level.upper(), logging.INFO)
    ch = _owned_handler(log, "console")
    if ch is None:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(_ColorFormatter("%(message)s"))
        setattr(ch, _OWNER_ATTR, "console")
        log.addHandler(ch)
    # Applied on every call, including a repeat of the same prefix: the level
    # is what the caller just asked for, not what the first caller asked for.
    ch.setLevel(console_level)

    path = str(prefix) + ".log"
    fh = _owned_handler(log, "file")
    if fh is not None:
        if os.path.abspath(fh.baseFilename) == os.path.abspath(path):
            return  # same file: keep it open rather than truncating mid-run
        log.removeHandler(fh)
        fh.close()  # release the fd; many fits in one process must not leak

    # mode="w" for a new prefix, matching a fresh single-fit run: each fit
    # owns its own <prefix>.log and starts it empty.
    fh = logging.FileHandler(path, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s")
    )
    setattr(fh, _OWNER_ATTR, "file")
    log.addHandler(fh)
