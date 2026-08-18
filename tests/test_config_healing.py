# tests/test_config_healing.py
import numpy as np
import pytest

from exozippy.config import ConfigManager


def test_config_derives_te_from_physical_input():
    """
    Given: User only provides Mass, Distance, and Proper Motion.
    When: ConfigManager is provided topology and finalized.
    Then: It should derive t_E and inject it into user_params.
    """
    # 1. Define the system topology so it knows what "Lens" and "Source" mean
    system_config = {
        "star": [{"name": "Lens"}, {"name": "Source"}],
        "lens": [{"name": "Lens", "lens_ndx": 0, "source_ndx": 1}],
    }

    user_params = {
        "star.Lens.mass": {"initval": 0.5},
        "star.Lens.distance": {"initval": 4000.0},
        "star.Source.distance": {"initval": 8000.0},
        "lens.Lens.u_0": {"initval": 0.5},
        "lens.Lens.t_0": {"initval": 2460000.0},
        "star.Lens.pm_ra": {"initval": 5.0},  # mas/yr
        "star.Lens.pm_dec": {"initval": 0.0},
        "star.Source.pm_ra": {"initval": 0.0},
        "star.Source.pm_dec": {"initval": 0.0},
    }

    cm = ConfigManager(user_params, system_config=system_config)
    cm.finalize_user_params()

    # Check if t_E was derived and injected.  The engine's solution is filed
    # under the canonical INDEX form (lens.0.t_E) -- the only spelling
    # ConfigManager.resolve reads for every element of every component.  See
    # the inject-back comment in finalize_user_params and tests/test_nsnl.py.
    assert "lens.0.t_E" in cm.user_params
    derived_te = cm.user_params["lens.0.t_E"]["initval"]

    # Manual check:
    # pi_rel = 1000/4000 - 1000/8000 = 0.125
    # theta_E = sqrt(8.144 * 0.5 * 0.125) = 0.7134
    # t_E = (0.7134 / 5.0) * 365.25 = 52.12
    assert np.isclose(derived_te, 52.12, atol=0.1)


def test_symbolic_time_limit_is_inert_where_sigalrm_is_absent(monkeypatch):
    """
    Given a platform whose signal module has no SIGALRM (Windows),
    When the symbolic solver's timeout guard is armed and disarmed,
    Then it degrades to a no-op instead of raising AttributeError.

    Regression: config.py called signal.signal(signal.SIGALRM, ...) and
    signal.alarm() unguarded, so every Windows run died with
    `AttributeError: module 'signal' has no attribute 'SIGALRM'` before the
    solver could do any work. Caught by adding Windows to the CI matrix.
    """
    # ARRANGE
    import exozippy.config as cfg

    monkeypatch.setattr(cfg, "_HAS_SIGALRM", False)
    calls = []
    monkeypatch.setattr(
        cfg.signal, "alarm", lambda *a: calls.append(a), raising=False
    )

    def boom(*args, **kwargs):  # must never be reached
        raise AssertionError("signal.signal called without SIGALRM support")

    monkeypatch.setattr(cfg.signal, "signal", boom)

    # ACT
    old = cfg._arm_alarm(2, lambda *a: None)
    cfg._disarm_alarm(old)
    with cfg._sympy_time_limit(2):
        result = 1 + 1

    # ASSERT
    assert old is None
    assert calls == [], "signal.alarm must not be called without SIGALRM"
    assert result == 2, "the guarded block must still execute"


def test_symbolic_time_limit_still_arms_where_sigalrm_exists():
    """
    Given a POSIX platform,
    When _sympy_time_limit guards a block that overruns its limit,
    Then SymbolicTimeout is still raised -- the Windows guard must not have
    disabled the timeout everywhere.
    """
    # ARRANGE
    import time

    import exozippy.config as cfg

    if not cfg._HAS_SIGALRM:
        pytest.skip("platform has no SIGALRM")

    # ACT / ASSERT
    with pytest.raises(cfg.SymbolicTimeout):
        with cfg._sympy_time_limit(1):
            time.sleep(5)


def test_a_symbolic_solve_restores_the_previous_sigalrm_handler():
    """
    Given a process-wide SIGALRM handler installed by unrelated code,
    When a relaxation solve runs (which arms its own 2-second symbolic
      timeout),
    Then the previous handler is back in place afterwards.

    _execute_solve used to arm SIGALRM by hand and drop the handler
    _arm_alarm returned, so from the first symbolic solve onwards the process
    handler WAS its local TimeoutError-raiser -- and any later code arming
    SIGALRM got "Symbolic solver timed out!" raised at it out of a frame that
    has nothing to do with sympy.  It now uses the _sympy_time_limit context
    manager, which restores in its finally.
    """
    # ARRANGE
    import signal

    if not hasattr(signal, "SIGALRM"):  # pragma: no cover - POSIX only
        pytest.skip("no SIGALRM on this platform")

    def sentinel(signum, frame):  # pragma: no cover - never fires
        pass

    config = {"star": [{"name": "A"}]}
    user_params = {
        "star.A.mass": {"initval": 1.0},
        "star.A.radius": {"initval": 1.0},
    }
    previous = signal.signal(signal.SIGALRM, sentinel)

    # ACT
    try:
        cm = ConfigManager(user_params, system_config=config)
        cm.finalize_user_params()
        after = signal.getsignal(signal.SIGALRM)
    finally:
        signal.signal(signal.SIGALRM, previous)

    # ASSERT
    assert after is sentinel
