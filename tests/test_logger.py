"""Tests for exozippy.logger.setup_logging.

The reason this file exists: `run.run_fit(config, user_params=<dict>)` is the
documented in-memory API for driving many fits from one process, and each fit
calls `setup_logging` with its own prefix. Before 2026-08 an early return on
`if log.handlers` made every fit after the first log into the FIRST fit's file
-- the second file was never even created -- and froze the console level at the
first call's setting.
"""

import logging

import pytest

from exozippy.logger import setup_logging


@pytest.fixture
def clean_exozippy_logger():
    """Restore the module-level `exozippy` logger after each test.

    setup_logging mutates a process-global logger, so a test that leaves its
    handlers attached would leak an open file handle into every later test.
    """
    log = logging.getLogger("exozippy")
    saved_handlers = list(log.handlers)
    saved_level = log.level
    log.handlers = []
    yield log
    for handler in log.handlers:
        handler.close()
    log.handlers = saved_handlers
    log.setLevel(saved_level)


def _read(path):
    with open(path) as fh:
        return fh.read()


def test_second_prefix_repoints_the_file_handler(
    clean_exozippy_logger, tmp_path
):
    """
    Given two fits run back to back in one process, under different prefixes,
    When each calls setup_logging with its own prefix and logs one message,
    Then each prefix's log file exists and holds only its own message.
    """
    # Arrange
    log = clean_exozippy_logger
    prefix_a = tmp_path / "fit_a"
    prefix_b = tmp_path / "fit_b"

    # Act
    setup_logging(prefix_a)
    log.info("message from fit A")
    setup_logging(prefix_b)
    log.info("message from fit B")

    # Assert
    text_a = _read(str(prefix_a) + ".log")
    assert "message from fit A" in text_a
    assert "message from fit B" not in text_a

    text_b = _read(str(prefix_b) + ".log")
    assert "message from fit B" in text_b
    assert "message from fit A" not in text_b


def test_repointing_closes_the_old_file(clean_exozippy_logger, tmp_path):
    """
    Given a logger already writing to the first fit's file,
    When setup_logging is called with a new prefix,
    Then the old file handler is detached and closed, so many fits in one
    process cannot leak file descriptors.
    """
    # Arrange
    log = clean_exozippy_logger
    setup_logging(tmp_path / "first")
    old = [h for h in log.handlers if isinstance(h, logging.FileHandler)][0]

    # Act
    setup_logging(tmp_path / "second")

    # Assert
    assert old not in log.handlers
    assert old.stream is None  # logging.FileHandler.close() clears the stream


def test_same_prefix_does_not_duplicate_handlers_or_messages(
    clean_exozippy_logger, tmp_path
):
    """
    Given setup_logging has already configured the logger for a prefix,
    When it is called a second time with that same prefix,
    Then no second handler is added, nothing already written is truncated,
    and a message is recorded exactly once.
    """
    # Arrange
    log = clean_exozippy_logger
    prefix = tmp_path / "same"
    setup_logging(prefix)
    log.info("written before the second call")
    handlers_after_first = list(log.handlers)

    # Act
    setup_logging(prefix)
    log.info("written after the second call")

    # Assert
    assert log.handlers == handlers_after_first
    text = _read(str(prefix) + ".log")
    assert text.count("written before the second call") == 1
    assert text.count("written after the second call") == 1


def test_console_level_follows_the_latest_call(
    clean_exozippy_logger, tmp_path
):
    """
    Given a logger configured with one console level,
    When setup_logging is called again with a different level,
    Then the console handler carries the level from the most recent call.
    """
    # Arrange
    log = clean_exozippy_logger
    setup_logging(tmp_path / "quiet", level="WARNING")

    # Act
    setup_logging(tmp_path / "loud", level="DEBUG")

    # Assert
    console = [
        h
        for h in log.handlers
        if isinstance(h, logging.StreamHandler)
        and not isinstance(h, logging.FileHandler)
    ]
    assert len(console) == 1
    assert console[0].level == logging.DEBUG


def test_foreign_handlers_are_left_alone(clean_exozippy_logger, tmp_path):
    """
    Given something else has attached its own handler to the exozippy logger
    (pytest's caplog, the GUI, a user's script),
    When setup_logging is called repeatedly with different prefixes,
    Then that handler is neither removed nor closed and keeps receiving records.
    """
    # Arrange
    log = clean_exozippy_logger
    foreign_path = tmp_path / "foreign.log"
    foreign = logging.FileHandler(str(foreign_path), mode="w")
    foreign.setLevel(logging.DEBUG)
    log.addHandler(foreign)

    # Act
    setup_logging(tmp_path / "one")
    log.info("first fit")
    setup_logging(tmp_path / "two")
    log.info("second fit")

    # Assert
    assert foreign in log.handlers
    assert foreign.stream is not None
    foreign.flush()
    text = _read(str(foreign_path))
    assert "first fit" in text
    assert "second fit" in text
