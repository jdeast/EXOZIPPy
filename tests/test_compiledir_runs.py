"""Tests for the per-RUN compiledir layout (review 2.13.5).

Two suites started on one machine against the suite's shared compiledir base
used to share ``base/gwN`` -- the xdist worker id was the only suffix -- and
each controller's startup prune deleted the other's in-flight compiles. The
fix gives every run a private ``base/runs/<token>/`` tree, seeded from
``base/shared`` by hard link and promoted back at exit, with ``base/.lock``
guarding every touch of the shared tree. These tests pin the properties that
fix rests on: a live run's tree is never deleted (by a prune, a sweep or a
reap), a dead one is, promotion moves only finished entries, and a failure
carrying the race's signature is named in the terminal summary.
"""

import importlib.util
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_NAME = "_pytensor_cache_budget"
if _NAME in sys.modules:
    budget = sys.modules[_NAME]
else:
    _SPEC = importlib.util.spec_from_file_location(
        _NAME,
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "pytensor_cache_budget.py",
    )
    budget = importlib.util.module_from_spec(_SPEC)
    sys.modules[_NAME] = budget
    _SPEC.loader.exec_module(budget)

if budget.fcntl is None:  # pragma: no cover - Windows
    pytest.skip(
        "the per-run layout needs fcntl.flock", allow_module_level=True
    )

PLATFORM = "compiledir_fake-3.12-64"
REPO = Path(__file__).resolve().parents[1]


def _entry(compiledir, name, key=b"(fake pickle).", module=True):
    """One cache entry shaped like PyTensor's; key=None means none yet."""
    entry = compiledir / name
    entry.mkdir(parents=True)
    (entry / "mod.cpp").write_text("// source")
    if module:
        (entry / "mod.so").write_bytes(b"\x7fELF fake")
    if key is not None:
        (entry / "key.pkl").write_bytes(key)
    return entry


def _names(compiledir):
    if not compiledir.is_dir():
        return set()
    return {p.name for p in compiledir.iterdir() if p.is_dir()}


# ---------------------------------------------------------------------------
# The reproduction, at unit level
# ---------------------------------------------------------------------------


def test_the_legacy_layout_deletes_another_suites_in_flight_compile(tmp_path):
    """Given suite A compiling into base/gw0 (an entry with no key.pkl yet),
    When suite B starts on the same base and prunes it, as master did,
    Then A's in-flight entry is deleted -- the 2.13.5 race.

    Kept as a test of the OLD layout on purpose: it is the deterministic
    form of the race, and it shows the mechanism is the prune's "no key.pkl
    means broken" rule meeting an entry another process is still writing.
    """
    # Arrange
    gw0 = tmp_path / "gw0" / PLATFORM
    for i in range(3):
        _entry(gw0, f"tmpdone{i}")
    in_flight = _entry(gw0, "tmpbuilding", key=None)

    # Act -- suite B's startup prune, budget below the count so it scans.
    budget.enforce_budget_tree(
        tmp_path, tmp_path / PLATFORM, 2, sweep_platforms=True
    )

    # Assert
    assert not in_flight.exists()


def test_a_prune_of_the_shared_tree_never_reaches_a_live_runs_tree(tmp_path):
    """Given a live run holding an in-flight entry and seeded entries,
    When another run prunes the shared tree to ZERO and sweeps platforms,
    Then the live run's tree is untouched, links and all.
    """
    # Arrange
    shared = budget.shared_compiledir(tmp_path, PLATFORM)
    for i in range(4):
        _entry(shared, f"tmpshared{i}")
    run = budget.create_run_dir(tmp_path)
    try:
        budget.seed_run(run, shared, PLATFORM, n_workers=1)
        tree = run.path / "gw0" / PLATFORM
        in_flight = _entry(tree, "tmpbuilding", key=None)
        before = _names(tree)

        # Act
        budget.enforce_shared_budget(tmp_path, PLATFORM, 0)

        # Assert
        assert _names(shared) == set()
        assert _names(tree) == before
        assert in_flight.is_dir()
        assert (tree / "tmpshared0" / "mod.so").read_bytes() == b"\x7fELF fake"
    finally:
        run.close()


# ---------------------------------------------------------------------------
# Run directories and liveness
# ---------------------------------------------------------------------------


def test_a_new_run_dir_is_live_until_its_owner_closes_it(tmp_path):
    # Arrange
    run = budget.create_run_dir(tmp_path)

    # Act / Assert -- flock is per open file description, so a second open
    # in this same process sees the owner's lock exactly as another process
    # would.
    assert run.path.parent == tmp_path / "runs"
    pid, host = budget.parse_run_token(run.path.name)
    assert pid == os.getpid()
    assert budget.run_state(run.path) == "live"
    run.close()
    assert budget.run_state(run.path) == "dead"


def test_a_dead_run_is_reaped_and_a_live_one_is_kept(tmp_path):
    # Arrange
    live = budget.create_run_dir(tmp_path)
    dead = budget.create_run_dir(tmp_path)
    _entry(dead.path / "gw0" / PLATFORM, "tmpleftover")
    dead.close()  # what the kernel does when a run is killed outright
    me = budget.create_run_dir(tmp_path)
    try:
        # Act
        stats = budget.reap_dead_runs(tmp_path, keep=me.path)

        # Assert
        assert stats.reaped == [dead.path.name]
        assert stats.live == [live.path.name]
        assert not dead.path.exists()
        assert live.path.is_dir() and me.path.is_dir()
    finally:
        live.close()
        me.close()


def test_a_run_dir_from_another_host_is_never_reaped(tmp_path):
    """On a base shared over NFS a remote run's liveness cannot be checked
    from here, so its directory is left alone however it looks."""
    # Arrange -- no alive.lock holder at all, i.e. it would look dead.
    remote = budget.create_run_dir(
        tmp_path, token="abcd1234-1-elsewhere.example"
    )
    remote.close()

    # Act
    stats = budget.reap_dead_runs(tmp_path, host="here.example")

    # Assert
    assert stats.remote == [remote.path.name]
    assert remote.path.is_dir()


def test_a_half_created_run_dir_is_judged_by_its_pid(tmp_path):
    """A run dir is built under a dot name and renamed once its lock is
    held, so a dot name is a run that is starting -- or died starting."""
    # Arrange
    host = "here.example"
    runs = tmp_path / "runs"
    starting = runs / f".aaaa0000-{os.getpid()}-{host}"
    dead_pid = _a_dead_pid()
    crashed = runs / f".bbbb0000-{dead_pid}-{host}"
    starting.mkdir(parents=True)
    crashed.mkdir()

    # Act
    stats = budget.reap_dead_runs(tmp_path, host=host)

    # Assert
    assert stats.live == [starting.name]
    assert stats.reaped == [crashed.name]


def test_a_foreign_directory_under_runs_raises_naming_it(tmp_path):
    (tmp_path / "runs" / "not-a-run").mkdir(parents=True)
    with pytest.raises(ValueError, match="not-a-run"):
        budget.reap_dead_runs(tmp_path)


def _a_dead_pid():
    done = subprocess.run(
        [sys.executable, "-c", "import os; print(os.getpid())"],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(done.stdout)


# ---------------------------------------------------------------------------
# The base lock
# ---------------------------------------------------------------------------


def test_shared_holders_coexist_and_exclude_an_exclusive_one(tmp_path):
    # Arrange
    first, second, writer = (budget.BaseLock(tmp_path) for _ in range(3))

    # Act / Assert
    assert first.acquire(exclusive=False, timeout=0)
    assert second.acquire(exclusive=False, timeout=0)
    assert not writer.acquire(exclusive=True, timeout=0.3, poll=0.05)
    first.release()
    second.release()
    assert writer.acquire(exclusive=True, timeout=0)
    writer.release()


def test_an_exclusive_holder_makes_the_others_skip(tmp_path):
    """Contention SKIPS the step (returns False); it never raises and never
    proceeds unlocked."""
    holder, other = budget.BaseLock(tmp_path), budget.BaseLock(tmp_path)
    assert holder.acquire(exclusive=True, timeout=0)
    try:
        assert not other.acquire(exclusive=False, timeout=0.2, poll=0.05)
        assert not other.acquire(exclusive=True, timeout=0.2, poll=0.05)
        assert other.fd == -1
    finally:
        holder.release()


# ---------------------------------------------------------------------------
# Seeding a run
# ---------------------------------------------------------------------------


def test_each_worker_tree_of_a_run_is_seeded_from_the_shared_tree(tmp_path):
    # Arrange
    shared = budget.shared_compiledir(tmp_path, PLATFORM)
    _entry(shared, "tmpa")
    _entry(shared, "tmpb")
    run = budget.create_run_dir(tmp_path)
    try:
        # Act
        stats = budget.seed_run(run, shared, PLATFORM, n_workers=2)

        # Assert
        assert stats.seeded_dirs == ["gw0", "gw1"]
        for gw in ("gw0", "gw1"):
            tree = run.path / gw / PLATFORM
            assert _names(tree) == {"tmpa", "tmpb"}
            so, key = tree / "tmpa" / "mod.so", tree / "tmpa" / "key.pkl"
            assert (
                os.stat(so).st_ino
                == os.stat(shared / "tmpa" / "mod.so").st_ino
            )
            assert (
                os.stat(key).st_ino
                != os.stat(shared / "tmpa" / "key.pkl").st_ino
            )
    finally:
        run.close()


def test_a_serial_run_seeds_its_own_tree(tmp_path):
    shared = budget.shared_compiledir(tmp_path, PLATFORM)
    _entry(shared, "tmpa")
    run = budget.create_run_dir(tmp_path)
    try:
        stats = budget.seed_run(run, shared, PLATFORM, n_workers=0)
        assert stats.seeded_dirs == ["run"]
        assert _names(run.path / PLATFORM) == {"tmpa"}
    finally:
        run.close()


def test_the_seed_source_is_shared_once_warm_and_legacy_until_then(tmp_path):
    """The pre-2.13.5 trees seed the first run (and the first CI run
    restoring an old-layout cache); after that the shared tree wins even
    though a frozen legacy tree may hold a few more entries."""
    # Arrange
    legacy = tmp_path / "gw0" / PLATFORM
    for i in range(10):
        _entry(legacy, f"tmpold{i}")

    # Act / Assert -- shared empty: the legacy tree.
    assert budget.choose_run_seed_source(tmp_path, PLATFORM) == legacy

    shared = budget.shared_compiledir(tmp_path, PLATFORM)
    for i in range(8):
        _entry(shared, f"tmpnew{i}")
    assert budget.choose_run_seed_source(tmp_path, PLATFORM) == shared


def test_nothing_warm_anywhere_gives_no_seed_source(tmp_path):
    assert budget.choose_run_seed_source(tmp_path, PLATFORM) is None


# ---------------------------------------------------------------------------
# Promotion
# ---------------------------------------------------------------------------


def test_promotion_moves_only_new_finished_entries(tmp_path):
    """Given a run with a seeded entry, a new finished one, one still being
    built and one whose key.pkl a killed worker truncated,
    When it promotes,
    Then only the new finished entry reaches the shared tree, whole.
    """
    # Arrange
    shared = budget.shared_compiledir(tmp_path, PLATFORM)
    _entry(shared, "tmpseeded")
    run = budget.create_run_dir(tmp_path)
    try:
        budget.seed_run(run, shared, PLATFORM, n_workers=2)
        gw0, gw1 = (run.path / gw / PLATFORM for gw in ("gw0", "gw1"))
        _entry(gw0, "tmpnew")
        _entry(gw1, "tmpbuilding", key=None)
        _entry(gw1, "tmptruncated", key=b"\x80\x04\x95partial")
        _entry(gw1, "tmpnoso", module=False)

        # Act
        stats = budget.promote_run(run, shared, PLATFORM)

        # Assert
        assert _names(shared) == {"tmpseeded", "tmpnew"}
        assert sorted(os.listdir(shared / "tmpnew")) == [
            "key.pkl",
            "mod.cpp",
            "mod.so",
        ]
        assert not (gw0 / "tmpnew").exists()  # moved, not copied
        assert stats.promoted == 1
        assert stats.already_shared == 2  # tmpseeded, in gw0 and in gw1
        assert stats.incomplete == 3
    finally:
        run.close()


def test_a_finished_run_dir_is_removed_and_only_after_unlocking(tmp_path):
    run = budget.create_run_dir(tmp_path)
    with pytest.raises(RuntimeError, match="liveness lock"):
        budget.remove_run_dir(run)
    run.close()
    budget.remove_run_dir(run)
    assert not run.path.exists()


# ---------------------------------------------------------------------------
# The whole lifecycle, through the root conftest
# ---------------------------------------------------------------------------


def _run_pytest(tmp_path, base, *args):
    """A real pytest session with the ROOT conftest loaded as a plugin.

    ``-p conftest`` from the repo root imports the root conftest.py itself
    (tests/conftest.py is a different module path), so this exercises the
    exact import-time and exit-time hooks a suite run does, against a
    private base.
    """
    test = tmp_path / "test_compiles.py"
    test.write_text(
        textwrap.dedent(
            """
            import os
            from pathlib import Path

            import numpy as np
            import pytensor
            import pytensor.tensor as pt

            def test_compiles():
                # Each process compiles in ITS OWN tree of THIS run: a worker
                # in <run>/gwN/compiledir_*, a -n0 run in <run>/compiledir_*.
                # (A worker once lost a flag-precedence contest to the
                # controller's run directory and all workers shared one
                # unseeded tree -- green, and recompiling everything.)
                here = Path(pytensor.config.compiledir)
                worker = os.environ.get("PYTEST_XDIST_WORKER")
                run = here.parent.parent if worker else here.parent
                assert run.parent.name == "runs", here
                if worker:
                    assert here.parent.name == worker, here
                x = pt.dvector("x")
                f = pytensor.function(
                    [x], pt.exp(x) * 2.0 + pt.sqrt(x + 7.0), mode="FAST_RUN"
                )
                assert f(np.ones(3))[0] > 0
            """
        )
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTEST_")}
    env.pop("_EXOZIPPY_TEST_RUN_DIR", None)
    env["EXOZIPPY_TEST_COMPILEDIR"] = str(base)
    # cvm, not numba: see test_pytensor_really_gets_cache_hits_from_a_seeded_
    # compiledir for why the C cache is the one to exercise.
    env["PYTENSOR_FLAGS"] = "linker=cvm"
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "conftest",
            "-p",
            "no:cacheprovider",
            "-q",
            "--rootdir",
            str(tmp_path),
            str(test),
            *args,
        ],
        cwd=str(REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    return done


def test_a_run_promotes_what_it_compiled_and_leaves_no_run_dir(tmp_path):
    """Given an empty base,
    When an xdist run compiles a graph, and then a serial run compiles it
    again,
    Then the first run's entries end up in base/shared, neither run leaves
    a directory under base/runs, and the second run is a pure cache hit.
    """
    pytest.importorskip("xdist")
    pytensor = pytest.importorskip("pytensor")
    if not pytensor.config.cxx:
        pytest.skip("no C compiler configured; there is no C cache")
    base = tmp_path / "base"

    # Act
    _run_pytest(tmp_path, base, "-n", "2")
    (shared,) = (base / "shared").iterdir()
    after_first = _names(shared)
    _run_pytest(tmp_path, base, "-n", "0")

    # Assert
    assert after_first, "the first run promoted nothing"
    assert list((base / "runs").iterdir()) == []
    assert _names(shared) == after_first, "the seeded serial run recompiled"


# ---------------------------------------------------------------------------
# Naming the race in the terminal summary
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        "/bin/ld: cannot open output file /x/gw2/compiledir_Linux-64/"
        "tmpab12cd/m.so: No such file or directory",
        "ImportError: dlimport failed for /x/gw0/compiledir_Linux-64/tmpzz/m.so",
        "ModuleNotFoundError: No module named 'tmpq1w2e3' "
        "(/x/compiledir_Linux-64/tmpq1w2e3)",
    ],
    ids=["link", "dlimport", "tmp-module"],
)
def test_each_recorded_race_signature_is_named_once(tmp_path, message):
    """Given a failing test whose error carries a compiledir_*/tmp path,
    When the session ends,
    Then one summary line names review 2.13.5 and the triage rule, and the
    failure is still a failure.
    """
    # Arrange -- a session that loads only the hook, not the root conftest.
    (tmp_path / "conftest.py").write_text(
        textwrap.dedent(
            f"""
            import importlib.util, sys
            spec = importlib.util.spec_from_file_location(
                "_budget_under_test", {str(budget.__file__)!r})
            module = importlib.util.module_from_spec(spec)
            sys.modules["_budget_under_test"] = module
            spec.loader.exec_module(module)

            def pytest_terminal_summary(terminalreporter):
                module.report_compiledir_race(terminalreporter)
            """
        )
    )
    (tmp_path / "test_victim.py").write_text(
        f"def test_victim():\n    raise RuntimeError({message!r})\n\n"
        "def test_real():\n    assert 1.0 == 2.0\n"
    )

    # Act
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "-p",
            "no:xdist",
            "-q",
            "--rootdir",
            str(tmp_path),
            str(tmp_path),
        ],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        timeout=300,
    )

    # Assert
    assert done.returncode == 1
    assert "2 failed" in done.stdout
    assert done.stdout.count("review 2.13.5") == 1
    assert "1 failure(s)/error(s) name a compiledir_*/tmp path" in done.stdout
    assert "an assertion on a number still has to be explained" in done.stdout


def test_an_ordinary_failure_is_not_blamed_on_the_race():
    class Report:
        nodeid = "t::x"
        longreprtext = "AssertionError: planet mass 2.870 outside its bound"

    assert budget.compiledir_race_hits([Report()]) == []
