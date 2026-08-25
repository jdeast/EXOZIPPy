"""Tests for the CI worker-count rule (scripts/pytest_workers.py).

The rule replaced a hardcoded `-n 2` in the CI workflow, and it exists
because that 2 went stale unnoticed: it was right when a hosted Linux runner
had two cores and wrong for years afterwards. So what is worth pinning here
is not the arithmetic but the two properties that keep it from going stale in
the other direction -- it never exceeds what the memory allows, and it never
returns zero.
"""

import importlib.util
import sys
from pathlib import Path

# By path, for the same reason tests/test_pytensor_cache_budget.py loads its
# subject by path: scripts/ is not a package, and putting it on sys.path would
# make getdata.py / mkparam.py importable as top-level modules.
_NAME = "_pytest_workers"
if _NAME in sys.modules:
    workers = sys.modules[_NAME]
else:
    _SPEC = importlib.util.spec_from_file_location(
        _NAME,
        Path(__file__).resolve().parents[1] / "scripts" / "pytest_workers.py",
    )
    workers = importlib.util.module_from_spec(_SPEC)
    sys.modules[_NAME] = workers
    _SPEC.loader.exec_module(workers)


def test_memory_binds_before_cores_do():
    """Given more cores than the memory can feed workers for,
    When the worker count is chosen,
    Then memory decides it.

    This is the case the number exists for. Overshooting it does not fail an
    assertion, it produces "worker 'gwN' crashed" on whichever heavy test drew
    the short straw -- a memory ceiling that presents as a wandering flaky
    failure rather than as an out-of-memory error."""
    # Arrange -- a 3-core / 7 GB runner, i.e. the macOS shape.
    # Act
    chosen = workers.choose_workers(cpus=3, mem_gb=7.0, gb_per_worker=3)

    # Assert -- 7 // 3 = 2, not the 3 the cores would allow.
    assert chosen == 2


def test_cores_bind_when_memory_is_plentiful():
    """Given more memory than the cores can use,
    When the worker count is chosen,
    Then it never exceeds the core count."""
    # Arrange / Act -- a 4-core / 16 GB runner, i.e. the Linux shape.
    chosen = workers.choose_workers(cpus=4, mem_gb=16.0, gb_per_worker=3)

    # Assert -- 16 // 3 = 5, capped at the 4 cores.
    assert chosen == 4


def test_unknown_memory_falls_back_to_the_core_count():
    """Given a platform whose memory cannot be read,
    When the worker count is chosen,
    Then the core count decides it.

    Falling back rather than guessing a size: memory_gb() returns None instead
    of a fabricated figure, and the resulting behaviour is the CPU-only rule
    we had before the memory term existed."""
    # Arrange / Act
    chosen = workers.choose_workers(cpus=8, mem_gb=None, gb_per_worker=3)

    # Assert
    assert chosen == 8


def test_a_tiny_machine_still_gets_one_worker():
    """Given less memory than one worker's share,
    When the worker count is chosen,
    Then it is 1, never 0.

    `pytest -n 0` is not "one worker", it is xdist disabled -- a different
    execution mode with different fixture sharing. Returning 0 here would
    silently change what CI runs."""
    # Arrange / Act
    chosen = workers.choose_workers(cpus=2, mem_gb=1.0, gb_per_worker=3)

    # Assert
    assert chosen == 1


def test_this_machine_gets_a_usable_count():
    """Given the machine actually running the suite,
    When the probes are read for real,
    Then they yield at least one worker.

    An end-to-end check on the two probes, because both are platform
    conditionals: a sysconf key that is absent, or an os attribute that moved,
    would otherwise only show up as a CI job silently running single-file."""
    # Arrange / Act
    chosen = workers.choose_workers(workers.cpu_count(), workers.memory_gb())

    # Assert
    assert chosen >= 1
    assert workers.cpu_count() >= 1
