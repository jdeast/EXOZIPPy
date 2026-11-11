"""A pool worker dies with its parent (samplers/_common._worker_init).

The workers ignore SIGINT and SIGTERM by design, so a batch scheduler's
process-group signal cannot break the parent mid-step, and the parent stops
them itself (_shutdown_pool).  When the parent is killed instead, a worker
waiting on its queue exits, but one still inside a long logp call has nobody
left to stop it and ignores SIGTERM: in a sweep of his examples on one box,
two seed polishes stopped by a 1200 s limit left such a worker behind
(examples ogle0383 and ob09020).
"""

import os
import subprocess
import sys
import textwrap
import time

import pytest

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="PR_SET_PDEATHSIG is Linux"
)

_CHILD = textwrap.dedent(
    """
    import multiprocessing as mp, sys, time
    from exozippy.samplers import _common

    pool = mp.get_context("fork").Pool(1, initializer=_common._worker_init)
    pool.apply_async(time.sleep, (120,))  # a worker busy in a long call
    time.sleep(1.0)
    print(pool._pool[0].pid, flush=True)
    time.sleep(120)
    """
)


def _alive(pid):
    try:
        with open(f"/proc/{pid}/status") as fh:
            state = next(line for line in fh if line.startswith("State:"))
    except FileNotFoundError:
        return False
    return " Z " not in state and "zombie" not in state


def test_a_busy_worker_dies_when_its_parent_is_killed():
    """
    Given a process holding a pool of _worker_init workers, one of them busy
      in a two-minute call,
    When that process is killed outright (SIGKILL: no handler runs),
    Then the busy worker is gone within five seconds.
    """
    # ARRANGE
    child = subprocess.Popen(
        [sys.executable, "-c", _CHILD], stdout=subprocess.PIPE, text=True
    )
    worker = int(child.stdout.readline())
    assert _alive(worker)  # not vacuous: the worker is running
    try:
        # ACT
        child.kill()
        child.wait(timeout=10)
        deadline = time.time() + 5.0
        while _alive(worker) and time.time() < deadline:
            time.sleep(0.1)

        # ASSERT
        assert not _alive(worker)
    finally:
        if _alive(worker):
            os.kill(worker, 9)
