"""The exozippy logger: console and file handlers, the warning recap, and the
coarse progress cadence.  The policy every message follows -- which level,
what format, how often -- is docs/logging.md; read it before adding or
changing a log message.
"""

import logging
import os
import re
import sys
import time

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

    # The recap handler collects every WARNING record for the setup and
    # end-of-run recaps (docs/logging.md).  One per process; each fit
    # (each call naming a new prefix) starts a fresh collection below.
    recap = _owned_handler(log, "recap")
    if recap is None:
        recap = WarningRecap()
        setattr(recap, _OWNER_ATTR, "recap")
        log.addHandler(recap)

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

    # A new fit: its recap starts empty, like its log file, and every
    # once-per-run message may be said again.
    recap.clear()
    _ONCE.clear()

    # mode="w" for a new prefix, matching a fresh single-fit run: each fit
    # owns its own <prefix>.log and starts it empty.  UTF-8 explicitly: the
    # locale's encoding wrote a non-ASCII character as '?' (the
    # `????` banner of 2026-11), and docs/logging.md asks for ASCII anyway.
    fh = logging.FileHandler(path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s")
    )
    setattr(fh, _OWNER_ATTR, "file")
    log.addHandler(fh)


def fmt_duration(seconds):
    """Compact human duration for a log line: '45 s', '38.2 min', '3.1 h'.

    Here rather than beside either caller because both the wrap-up stage
    announcer (run.py) and the polish heartbeat (samplers/ptde.py) print
    one, and two spellings of "how long has this been running" in the same
    fit log is exactly the drift this module exists to prevent.
    """
    s = float(seconds)
    if s < 90.0:
        return f"{s:.0f} s"
    if s < 5400.0:
        return f"{s / 60.0:.1f} min"
    return f"{s / 3600.0:.1f} h"


# ---------------------------------------------------------------------------
# The warning recap (docs/logging.md, "Warnings must not be buried")
# ---------------------------------------------------------------------------

RECAP_ATTR = "exozippy_recap"
"""``extra={RECAP_ATTR: True}`` marks a recap's own record, so the recap
handler does not collect the recap into the next recap."""

_SEPARATOR = re.compile(r"^[\s\-=_*#~.:|+]*$")
HEADLINE_MAX = 200


def warning_headline(message):
    """The first line of a warning that says something, trimmed for a recap.

    A banner line made only of punctuation (``-----``, ``=====``) is skipped,
    so a boxed warning recaps as its first sentence rather than its frame.
    """
    for line in str(message).splitlines():
        line = line.strip()
        if line and not _SEPARATOR.match(line):
            if len(line) > HEADLINE_MAX:
                line = line[: HEADLINE_MAX - 3].rstrip() + "..."
            return line
    return str(message).strip()[:HEADLINE_MAX]


class WarningRecap(logging.Handler):
    """Collects every WARNING-or-worse record so it can be repeated as a recap.

    Installed by `setup_logging` on the ``exozippy`` logger.  Records carry
    their headline (`warning_headline`), and identical headlines are counted
    rather than repeated: ``[star] X ...`` fired for three stars recaps as one
    line with ``(x3)``.  `mark()` returns a position, and `lines(since=...)`
    recaps only what came after it -- which is how the pre-sampling recap and
    the end-of-run recap share one collection.
    """

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.records = []

    def emit(self, record):
        if getattr(record, RECAP_ATTR, False):
            return
        self.records.append(
            (record.levelname, warning_headline(record.getMessage()))
        )

    def clear(self):
        self.records = []

    def mark(self):
        return len(self.records)

    def lines(self, since=0):
        counts = {}
        for level, headline in self.records[since:]:
            key = (level, headline)
            counts[key] = counts.get(key, 0) + 1
        out = []
        for (level, headline), n in counts.items():
            tag = "" if level == "WARNING" else f"{level}: "
            mult = f" (x{n})" if n > 1 else ""
            out.append(f"  - {tag}{headline}{mult}")
        return out, len(self.records) - since


def warning_recap():
    """The installed `WarningRecap`, or None when `setup_logging` never ran
    (a library caller driving `System` directly has no recap to show)."""
    return _owned_handler(logging.getLogger("exozippy"), "recap")


def log_warning_recap(stage, since=0, logfile=None):
    """Repeat the WARNING headlines collected since `since`, one line each.

    Emitted as ONE warning record, so it is colored like the warnings it
    repeats and sits right where the user is looking (before sampling, at the
    end of the run).  Silent when there is nothing to repeat.  Returns the
    number of warnings recapped.
    """
    recap = warning_recap()
    if recap is None:
        return 0
    lines, n = recap.lines(since=since)
    if not n:
        return 0
    where = (
        f"; full text above and in {logfile}"
        if logfile
        else "; full text above"
    )
    logging.getLogger("exozippy").warning(
        f"Recap: {n} warning(s) {stage}{where}:\n" + "\n".join(lines),
        extra={RECAP_ATTR: True},
    )
    return n


# ---------------------------------------------------------------------------
# Progress cadence (docs/logging.md, INFO: "progress at a coarse cadence")
# ---------------------------------------------------------------------------

INFO_PROGRESS_S = 300.0
"""At most one INFO progress line per this many seconds of wall clock per
long stage.  The finer heartbeat lines in between go to DEBUG -- the file log
keeps every one of them, which is where "computing or hung?" is answered."""


class ProgressCadence:
    """Decides the level of each heartbeat line of one long-running stage.

    The stage's own heartbeat keeps its own (fine) interval; this promotes
    one line per `INFO_PROGRESS_S` to INFO and leaves the rest at DEBUG.  The
    clock starts at construction, which is the stage's start line.
    """

    def __init__(self, info_interval_s=None):
        self.info_interval_s = (
            INFO_PROGRESS_S
            if info_interval_s is None
            else float(info_interval_s)
        )
        self._last_info = time.monotonic()

    def level(self, force=False):
        now = time.monotonic()
        if force or now - self._last_info >= self.info_interval_s:
            self._last_info = now
            return logging.INFO
        return logging.DEBUG


# ---------------------------------------------------------------------------
# Once per run (docs/logging.md, "logged ONCE")
# ---------------------------------------------------------------------------

_ONCE = set()


def first_time(key):
    """True the first time ``key`` is seen in this fit, False after.

    For a WARNING whose cause is reached several times per run with
    different numbers -- once per light curve, once per solve pass -- so
    `system._RepeatedRecords` (which drops only IDENTICAL repeats) cannot
    collapse it: log the per-call detail at DEBUG every time and the
    WARNING headline only when ``first_time(key)``.  `setup_logging`
    resets the registry when a new fit starts.
    """
    if key in _ONCE:
        return False
    _ONCE.add(key)
    return True


def reset_first_time():
    """Forget every `first_time` key: a new model is being built.

    Called by `System.__init__`, so a library caller (the GUI, a script
    building several Systems in one process) hears each once-per-run
    warning once per System, not once per process.
    """
    _ONCE.clear()
