"""The logging policy (docs/logging.md), pinned.

Three things the policy promises and a later edit could quietly undo:

* the console narrative of a setup stays SHORT -- a small shipped example's
  prepare + build emits a handful of INFO records, and every WARNING it emits
  names a remedy;
* warnings are not buried -- `logger.WarningRecap` collects them, counts
  repeats, and a recap is not collected into the next recap;
* user input is never silently ignored -- an unknown key in a component's
  config block raises, naming the key and suggesting the right spelling.
"""

import glob
import logging
from pathlib import Path

import pytest
import yaml

from exozippy.components.factory import discover_components
from exozippy.logger import (
    RECAP_ATTR,
    first_time,
    log_warning_recap,
    setup_logging,
    warning_headline,
)

REPO = Path(__file__).parent.parent
EXAMPLES = REPO / "examples"

# Words a remedy sentence starts with or contains.  Deliberately broad: the
# test guards against a bare FYI at WARNING ("X happened."), not against any
# particular phrasing of the fix.
REMEDY_WORDS = (
    "set ",
    "remove",
    "add ",
    "give",
    "check",
    "use ",
    "pin ",
    "raise ",
    "refresh",
    "state ",
    "write ",
    "edit ",
    "polish",
    "to keep",
    "to tie",
    "to break",
    "rerun",
    "re-run",
    "move ",
    "install",
    "supply",
    "correct",
)


def _has_remedy(message):
    low = message.lower()
    return any(w in low for w in REMEDY_WORDS)


@pytest.fixture
def exozippy_logger():
    """setup_logging mutates the process-global exozippy logger; put it back."""
    log = logging.getLogger("exozippy")
    saved = (list(log.handlers), log.level)
    yield log
    for h in list(log.handlers):
        if h not in saved[0]:
            log.removeHandler(h)
            h.close()
    log.setLevel(saved[1])


# ---------------------------------------------------------------------------
# The recap
# ---------------------------------------------------------------------------


def test_the_recap_counts_repeats_and_is_not_recollected(
    tmp_path, exozippy_logger, caplog
):
    """
    Given a fit's log with three warnings, two of them identical,
    When the recap is emitted twice,
    Then the first recap is ONE warning record listing two headlines, the
      repeated one counted (x2), and the second recap lists exactly the same
      thing -- the first recap was not collected into it.
    """
    # Arrange
    setup_logging(tmp_path / "fit")
    log = logging.getLogger("exozippy.test")
    log.warning("[star A] something the user must fix.\nSet x: 1.")
    log.warning("[star A] something the user must fix.\nSet x: 1.")
    log.warning("-----\n[band I] another problem.  Remove it.")

    # Act
    with caplog.at_level(logging.WARNING, logger="exozippy"):
        n1 = log_warning_recap("during setup")
        n2 = log_warning_recap("during this run")

    # Assert
    recaps = [r for r in caplog.records if getattr(r, RECAP_ATTR, False)]
    assert (n1, n2) == (3, 3)
    assert len(recaps) == 2
    text = recaps[0].getMessage()
    assert text.splitlines()[0].startswith("Recap: 3 warning(s) during setup")
    assert "[star A] something the user must fix. (x2)" in text
    assert "[band I] another problem.  Remove it." in text  # banner skipped
    assert recaps[1].getMessage().splitlines()[1:] == text.splitlines()[1:]


def test_a_new_fit_starts_an_empty_recap(tmp_path, exozippy_logger, caplog):
    """
    Given a warning logged during one fit,
    When setup_logging starts a second fit (a new prefix) with no warnings,
    Then the second fit's recap is silent.
    """
    setup_logging(tmp_path / "first")
    logging.getLogger("exozippy.test").warning("old warning.  Fix it.")
    setup_logging(tmp_path / "second")
    with caplog.at_level(logging.WARNING, logger="exozippy"):
        assert log_warning_recap("during setup") == 0
    assert not [r for r in caplog.records if getattr(r, RECAP_ATTR, False)]


def test_warning_headline_skips_a_banner_and_trims():
    assert warning_headline("=====\n\nreal headline\nmore") == "real headline"
    long = "x" * 500
    assert len(warning_headline(long)) <= 200


def test_first_time_is_once_per_fit(tmp_path, exozippy_logger):
    setup_logging(tmp_path / "a")
    assert first_time("k") is True
    assert first_time("k") is False
    setup_logging(tmp_path / "b")
    assert first_time("k") is True


# ---------------------------------------------------------------------------
# Unknown config keys raise
# ---------------------------------------------------------------------------


def _cls(yaml_key):
    return discover_components()[yaml_key]


def test_a_zeropoint_in_the_config_block_raises_pointing_at_the_params_file():
    """
    Given `zero_point: {mu: 0, sigma: 0.05}` on a mulensinstrument entry --
      the 2026-11 case where it was silently ignored for a whole fit,
    When the config keys are validated,
    Then it raises naming the key and the instance, and says the zeropoint
      is a parameter set in the params file.
    """
    block = [
        {
            "name": "MOA",
            "file": "moa.dat",
            "band": "R",
            "zero_point": {"mu": 0, "sigma": 0.05},
        }
    ]
    with pytest.raises(ValueError) as err:
        _cls("mulensinstrument").validate_config_keys(
            "mulensinstrument", block
        )
    msg = str(err.value)
    assert "[mulensinstrument MOA]" in msg
    assert "'zero_point'" in msg
    assert "mulensinstrument.MOA.zeropoint: {mu: ..., sigma: ...}" in msg
    assert "params file" in msg


def test_a_near_miss_key_raises_with_a_suggestion_and_no_case_folding():
    block = [{"name": "I", "Filter": "Generic/Cousins.I"}]
    with pytest.raises(ValueError) as err:
        _cls("band").validate_config_keys("band", block)
    msg = str(err.value)
    assert "'Filter'" in msg
    assert "did you mean 'filter'?" in msg


def test_known_keys_pass():
    block = [{"name": "I", "filter": "Generic/Cousins.I"}]
    _cls("band").validate_config_keys("band", block)


def test_every_shipped_config_uses_only_keys_its_components_read():
    """
    Given every YAML under examples/ and tests/,
    When each component block is validated,
    Then none raises -- a shipped example with a key nobody reads is the
      same silent-ignore bug in the documentation users copy from.
    """
    registry = discover_components()
    failures = []
    files = glob.glob(str(EXAMPLES / "**" / "*.yaml"), recursive=True)
    files += glob.glob(str(REPO / "tests" / "**" / "*.yaml"), recursive=True)
    for f in files:
        try:
            doc = yaml.safe_load(open(f))
        except yaml.YAMLError:
            continue
        if not isinstance(doc, dict):
            continue
        for key, block in doc.items():
            if key not in registry:
                continue
            try:
                registry[key].validate_config_keys(key, block)
            except ValueError as exc:
                failures.append(f"{f}: {exc}")
    assert not failures, "\n".join(failures)


# ---------------------------------------------------------------------------
# A setup's narrative is short, and its warnings actionable
# ---------------------------------------------------------------------------

# The INFO budget for kelt4 RV-only's System construction + prepare +
# build_model: the one-line model summary, "Start values solved." and a few
# component lines.  The startup table and the run-level lines are run.py's
# and are not built here.
MAX_SETUP_INFO = 8


def test_a_small_example_setup_is_short_and_its_warnings_actionable(
    monkeypatch, caplog
):
    from exozippy import System

    monkeypatch.chdir(EXAMPLES / "kelt4")
    with open(EXAMPLES / "kelt4" / "kelt4_rvonly.yaml") as f:
        config = yaml.safe_load(f)
    with caplog.at_level(logging.DEBUG, logger="exozippy"):
        system = System(config)
        system.prepare()
        system.build_model()
    infos = [r.getMessage() for r in caplog.records if r.levelname == "INFO"]
    warns = [
        r.getMessage() for r in caplog.records if r.levelname == "WARNING"
    ]
    assert any(m.startswith("Model: ") for m in infos)
    assert len(infos) <= MAX_SETUP_INFO, "\n".join(infos)
    no_remedy = [m for m in warns if not _has_remedy(m)]
    assert not no_remedy, "\n".join(no_remedy)
    assert all(m.isascii() for m in infos + warns)
