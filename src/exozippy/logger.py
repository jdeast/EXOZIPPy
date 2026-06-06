import logging
import sys


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


def setup_logging(prefix, level="INFO"):
    """Configure the exozippy logger: console at `level`, file at DEBUG always."""
    log = logging.getLogger("exozippy")
    if log.handlers:
        return  # already configured (e.g. called twice in tests)
    log.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(getattr(logging, level.upper(), logging.INFO))
    ch.setFormatter(_ColorFormatter("%(message)s"))
    log.addHandler(ch)

    fh = logging.FileHandler(str(prefix) + ".log", mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s"))
    log.addHandler(fh)
