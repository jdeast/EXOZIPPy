"""System.prepare's second stage-3/4 pass does not repeat the first's log lines.

The filter (``system._RepeatedRecords``) is checked here on its own; the
end-to-end case, an orbit group's link line printed once through a forced
second pass, is ``tests/test_orbit_star_links.py``.
"""

import logging

from exozippy.system import _RepeatedRecords


def test_a_repeat_is_dropped_and_a_new_line_is_kept(caplog):
    """
    Given a line logged while the filter records, and the filter then
    switched to dropping,
    When the same line and a different one are logged again,
    Then only the different one is emitted -- and once the filter is
    removed, the repeated line prints again.
    """
    log = logging.getLogger("exozippy.components.band.band")
    f = _RepeatedRecords()
    with caplog.at_level(logging.INFO, logger="exozippy"):
        f.install()
        try:
            log.warning("skipped the tie on band V")
            f.dropping = True
            log.warning("skipped the tie on band V")
            log.warning("skipped the tie on band I")
            log.info("skipped the tie on band V")  # another level: new
        finally:
            f.remove()
        log.warning("skipped the tie on band V")
    got = [(r.levelno, r.getMessage()) for r in caplog.records]
    assert got == [
        (logging.WARNING, "skipped the tie on band V"),
        (logging.WARNING, "skipped the tie on band I"),
        (logging.INFO, "skipped the tie on band V"),
        (logging.WARNING, "skipped the tie on band V"),
    ]
