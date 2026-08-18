"""`maxtime:` must never be silently ignored (review 2.3.2).

A key that is silently ignored is worse than one that is refused: the whole
point of `maxtime` is that a scheduler-bound job stops itself before the
queue kills it, so a user who sets it and gets nothing has no partial trace
AND no idea why.  `demc` already warned (PyMC's population path discards
per-draw callbacks); numpyro, blackjax and nutpie were the remaining silent
ones -- external NUTS samplers run the chain outside Python's per-draw loop
and invoke no callback at all.
"""

import inspect
import logging

import pymc as pm
import pytest

from exozippy import run


@pytest.mark.parametrize("method", ["numpyro", "blackjax", "nutpie"])
def test_an_unsupported_sampler_says_maxtime_is_ignored(method, caplog):
    """
    Given maxtime set and a sampler that cannot honor it,
    When the check runs,
    Then it warns, naming the method, the value, and a sampler that can.
    """
    with caplog.at_level(logging.WARNING):
        warned = run.warn_maxtime_unsupported(method, 3600.0)

    assert warned is True
    assert method in caplog.text
    assert "IGNORED" in caplog.text
    assert "3600" in caplog.text
    assert "ptde_async" in caplog.text  # points at one that works


@pytest.mark.parametrize("method", ["nuts", "ptde", "ptde_async", "demcz"])
def test_a_supporting_sampler_is_not_warned_about(method, caplog):
    """
    Given maxtime set and a sampler that honors it,
    When the check runs,
    Then nothing is said -- the warning must not cry wolf about a limit that
      really will be applied.
    """
    with caplog.at_level(logging.WARNING):
        warned = run.warn_maxtime_unsupported(method, 3600.0)

    assert warned is False
    assert caplog.text == ""


def test_no_warning_when_maxtime_is_unset(caplog):
    """
    Given no maxtime,
    When the check runs against an unsupporting sampler,
    Then nothing is said -- there is no key being ignored.
    """
    with caplog.at_level(logging.WARNING):
        assert run.warn_maxtime_unsupported("numpyro", None) is False

    assert caplog.text == ""


def test_the_check_runs_after_the_missing_backend_fallback():
    """
    Given a config asking for numpyro on a box without it,
    When _run_fit falls back to PyMC NUTS (where maxtime IS honored),
    Then the maxtime check runs AFTER that reassignment, so the user is not
      warned about a limit that will in fact be applied.

    Pinned on the source order because reproducing it needs a real fit with
    a real missing backend.
    """
    source = inspect.getsource(run._run_fit)
    fallback = source.index('method = "nuts"')
    check = source.index("warn_maxtime_unsupported(method, maxtime)")

    assert fallback < check


def test_pymc_still_refuses_a_callback_for_an_external_nuts_sampler():
    """
    Given PyMC's own sampling entry point,
    When an external nuts_sampler is combined with a callback,
    Then PyMC rejects it -- which is WHY maxtime cannot be honored there.

    Pins the upstream fact the warning rests on: if a future PyMC starts
    forwarding callbacks to external samplers, this fails and the warning
    should become a real wall-clock cap instead.
    """
    source = inspect.getsource(pm.sample)

    assert "`callback` is not supported with `nuts_sampler=" in source
