"""Tests for the compiledir entry-count budget (scripts/pytensor_cache_budget.py).

The thing under test decides what gets DELETED off a developer's disk, so the
cases that matter are the ones where it must not act: the compile lock
directory, an entry that is merely old but still within budget, and anything
at all under --dry-run.
"""

import importlib.util
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

# Loaded by path for the same reason conftest.py loads it by path: scripts/ is
# not a package, and putting it on sys.path would make getdata.py / mkparam.py
# importable as top-level modules. The root conftest has almost always loaded
# it already, so this normally just hands back the module it registered.
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
    # Before exec_module: @dataclass looks the module up in sys.modules while
    # processing the class body.
    sys.modules[_NAME] = budget
    _SPEC.loader.exec_module(budget)


def _make_entry(compiledir, name, age_seconds=0.0, key=True, module=True):
    """Create one cache entry shaped like PyTensor's, aged by age_seconds."""
    entry = compiledir / name
    entry.mkdir(parents=True)
    if module:
        (entry / "mod.so").write_bytes(b"\x7fELF fake")
    if key:
        key_pkl = entry / "key.pkl"
        key_pkl.write_bytes(b"fake pickle")
        stamp = 1_700_000_000.0 - age_seconds
        os.utime(key_pkl, (stamp, stamp))
    return entry


@pytest.fixture
def compiledir(tmp_path):
    d = tmp_path / "compiledir_fake-3.12-64"
    d.mkdir()
    return d


def test_eviction_keeps_the_most_recently_used_entries(compiledir):
    """Given more entries than the budget allows,
    When the compiledir is pruned,
    Then the newest survive and the oldest are deleted."""
    # Arrange -- five entries, entry0 newest through entry4 oldest.
    for i in range(5):
        _make_entry(compiledir, f"entry{i}", age_seconds=i * 3600)

    # Act
    stats = budget.prune_compiledir(compiledir, max_entries=2)

    # Assert
    assert stats.kept == 2
    assert stats.removed_lru == 3
    assert sorted(p.name for p in compiledir.iterdir()) == ["entry0", "entry1"]


def test_broken_entries_are_removed_without_using_up_the_budget(compiledir):
    """Given entries PyTensor's own refresh() would discard as broken,
    When the compiledir is pruned,
    Then they are deleted and the healthy entries all stay inside budget."""
    # Arrange -- a key with no compiled module, and a module with no key.
    # refresh() deletes the first and cannot index the second, so both are
    # pure cost: they are walked on every startup and never hit.
    _make_entry(compiledir, "good0", age_seconds=0)
    _make_entry(compiledir, "good1", age_seconds=10)
    _make_entry(compiledir, "keyless", key=False)
    _make_entry(compiledir, "moduleless", module=False)

    # Act
    stats = budget.prune_compiledir(compiledir, max_entries=2)

    # Assert -- both good entries survive; the budget was not spent on junk.
    assert stats.removed_broken == 2
    assert stats.removed_lru == 0
    assert sorted(p.name for p in compiledir.iterdir()) == ["good0", "good1"]


def test_the_compile_lock_directory_is_never_touched(compiledir):
    """Given a compiledir holding PyTensor's lock_dir,
    When everything is pruned to a budget of zero,
    Then lock_dir survives."""
    # Arrange -- deleting this out from under a concurrent process would break
    # PyTensor's compilation mutual exclusion, so refresh() skips it by name
    # and so must we.
    (compiledir / "lock_dir").mkdir()
    _make_entry(compiledir, "entry0")

    # Act
    stats = budget.prune_compiledir(compiledir, max_entries=0)

    # Assert
    assert (compiledir / "lock_dir").is_dir()
    assert not (compiledir / "entry0").exists()
    assert stats.scanned == 1


def test_dry_run_reports_without_deleting(compiledir):
    """Given a compiledir over budget,
    When it is pruned with dry_run,
    Then the counts are reported and every entry is still on disk."""
    # Arrange
    for i in range(4):
        _make_entry(compiledir, f"entry{i}", age_seconds=i)

    # Act
    stats = budget.prune_compiledir(compiledir, max_entries=1, dry_run=True)

    # Assert
    assert stats.removed_lru == 3
    assert len(list(compiledir.iterdir())) == 4


def test_a_compiledir_within_budget_is_left_alone(compiledir):
    """Given fewer entries than the budget,
    When the compiledir is pruned,
    Then nothing is removed -- age alone is not a reason to evict."""
    # Arrange -- deliberately ancient. This is the case `pytensor-cache
    # cleanup` gets backwards: it evicts on age and so does nothing useful on
    # a daily-run cache, while this budget evicts on pressure and so leaves a
    # small cache entirely warm.
    _make_entry(compiledir, "ancient", age_seconds=365 * 24 * 3600)

    # Act
    stats = budget.prune_compiledir(compiledir, max_entries=10)

    # Assert
    assert stats.removed == 0
    assert (compiledir / "ancient").is_dir()


def test_under_budget_skips_the_per_entry_scan_entirely(compiledir):
    """Given a compiledir within budget that also holds a broken entry,
    When it is pruned,
    Then nothing is touched -- not even the broken entry."""
    # Arrange -- this pins the cheap pre-check, which is load-bearing rather
    # than an optimization: the per-entry pass opens a directory and stats a
    # key.pkl for every entry, 72 s over 4034 of them on a cold page cache,
    # and it would otherwise run on every `pytest -n0 -x` single-test debug
    # invocation. One listdir of the parent bounds the count from above, so
    # being under budget is provable without the pass.
    _make_entry(compiledir, "good")
    _make_entry(compiledir, "broken", module=False)

    # Act
    stats = budget.prune_compiledir(compiledir, max_entries=10)

    # Assert -- the broken entry surviving IS the documented tradeoff:
    # refresh() removes those itself, and they are rare.
    assert stats.removed == 0
    assert (compiledir / "broken").is_dir()


def test_stranded_platform_trees_are_swept_but_the_live_one_is_kept(tmp_path):
    """Given sibling compiledirs left by another kernel or Python version,
    When the base compiledir is swept,
    Then only the tree this interpreter will actually read survives."""
    # Arrange -- PyTensor names the compiledir after platform/processor/python
    # version, so an OS upgrade or a CI image bump strands the whole previous
    # tree: nothing reads it and nothing deletes it.
    live = tmp_path / "compiledir_Linux-6.1-x86_64-3.12.13-64"
    stale = tmp_path / "compiledir_Linux-4.18-x86_64-3.12.13-64"
    for d in (live, stale):
        d.mkdir()
    unrelated = tmp_path / "numba"
    unrelated.mkdir()

    # Act
    removed = budget.sweep_other_platform_dirs(tmp_path, live)

    # Assert
    assert removed == [stale.name]
    assert live.is_dir()
    # Only compiledir_* trees are in scope; PyTensor keeps other caches here.
    assert unrelated.is_dir()


def test_enforce_budget_combines_the_prune_and_the_sweep(tmp_path):
    """Given a base compiledir with a stale sibling and an over-budget live tree,
    When enforce_budget runs with sweeping enabled,
    Then both the sibling and the excess entries are gone."""
    # Arrange
    live = tmp_path / "compiledir_live"
    live.mkdir()
    (tmp_path / "compiledir_stale").mkdir()
    for i in range(3):
        _make_entry(live, f"entry{i}", age_seconds=i)

    # Act
    stats = budget.enforce_budget(
        tmp_path, live, max_entries=1, sweep_platforms=True
    )

    # Assert
    assert stats.kept == 1
    assert stats.removed_platform_dirs == ["compiledir_stale"]
    assert not (tmp_path / "compiledir_stale").exists()
    assert [p.name for p in live.iterdir()] == ["entry0"]


def _worker_tree(base, worker, platform, entries, extra_platform=None):
    """One gwN/compiledir_<platform>/ tree with `entries` entries in it."""
    compiledir = base / worker / platform
    compiledir.mkdir(parents=True)
    for i in range(entries):
        _make_entry(compiledir, f"entry{i}", age_seconds=i)
    if extra_platform is not None:
        (base / worker / extra_platform).mkdir()
    return compiledir


def test_worker_compiledirs_finds_each_workers_own_platform_tree(tmp_path):
    """Given the per-worker layout the root conftest creates,
    When the worker compiledirs are enumerated,
    Then each gwN contributes its tree named for the LIVE platform."""
    # Arrange -- every worker on one machine resolves the same platform
    # directory name, because the conftest changes only the base.
    platform = "compiledir_Linux-6.1-x86_64-3.12.13-64"
    controller = tmp_path / platform
    controller.mkdir()
    for worker in ("gw0", "gw1", "gw2"):
        (tmp_path / worker / platform).mkdir(parents=True)
    # Not a worker directory, and PyTensor keeps other caches beside these.
    (tmp_path / "numba").mkdir()

    # Act
    pairs = budget.worker_compiledirs(tmp_path, controller)

    # Assert
    assert [base.name for base, _ in pairs] == ["gw0", "gw1", "gw2"]
    assert [d.name for _, d in pairs] == [platform] * 3


def test_the_budget_reaches_the_per_worker_compiledirs(tmp_path):
    """Given per-worker compiledirs over budget and an empty controller one,
    When the whole tree is bounded,
    Then every worker's directory is trimmed, not just what pytensor resolves.

    This is the regression. The prune used to run on
    pytensor.config.compiledir alone, which in the controller process is the
    top-level tree -- the one directory no worker ever writes to. It was
    within budget by virtue of being nearly empty, the pass reported success
    having removed nothing, and the directories holding the entire cache were
    never looked at. In CI that made every saved cache artifact bigger than
    the last until the repository's 10 GB budget went into eviction."""
    # Arrange
    platform = "compiledir_Linux-6.1-x86_64-3.12.13-64"
    controller = tmp_path / platform
    controller.mkdir()
    workers = [
        _worker_tree(tmp_path, name, platform, entries=5)
        for name in ("gw0", "gw1")
    ]

    # Act
    results = budget.enforce_budget_tree(tmp_path, controller, max_entries=2)

    # Assert -- the controller's tree first, then one per worker.
    assert [d for d, _ in results] == [controller, *workers]
    for worker_dir in workers:
        assert sorted(p.name for p in worker_dir.iterdir()) == [
            "entry0",
            "entry1",
        ]


def test_the_budget_is_per_compiledir_not_a_total_across_the_tree(tmp_path):
    """Given two workers each holding exactly the budget,
    When the tree is bounded,
    Then nothing is evicted.

    The denominator is deliberate: each worker process builds its own
    ModuleCache and walks only its own directory, so one directory's entry
    count is what the startup walk is linear in. A total across the tree
    would evict entries a worker still needs on the very run that made them."""
    # Arrange
    platform = "compiledir_fake-3.12-64"
    controller = tmp_path / platform
    controller.mkdir()
    for name in ("gw0", "gw1"):
        _worker_tree(tmp_path, name, platform, entries=3)

    # Act
    results = budget.enforce_budget_tree(tmp_path, controller, max_entries=3)

    # Assert
    assert all(stats.removed == 0 for _, stats in results)
    for name in ("gw0", "gw1"):
        assert len(list((tmp_path / name / platform).iterdir())) == 3


def test_a_worker_tree_stranded_by_a_platform_bump_is_swept_too(tmp_path):
    """Given a worker directory carrying a compiledir from an older kernel,
    When the tree is swept,
    Then that stranded sibling goes, the same as at the top level.

    Nothing reads a stranded tree and nothing else would ever delete it, so
    inside a worker directory it rides along in the CI cache forever -- the
    footgun the restore step's comment describes, one level down."""
    # Arrange
    platform = "compiledir_Linux-6.1-x86_64-3.12.13-64"
    stale = "compiledir_Linux-4.18-x86_64-3.12.13-64"
    controller = tmp_path / platform
    controller.mkdir()
    _worker_tree(tmp_path, "gw0", platform, entries=1, extra_platform=stale)

    # Act
    results = budget.enforce_budget_tree(
        tmp_path, controller, max_entries=10, sweep_platforms=True
    )

    # Assert
    assert not (tmp_path / "gw0" / stale).exists()
    assert (tmp_path / "gw0" / platform).is_dir()
    swept = [n for _, st in results for n in st.removed_platform_dirs]
    assert swept == [stale]


def test_dry_run_reaches_the_worker_dirs_without_deleting_from_them(tmp_path):
    """Given per-worker compiledirs over budget,
    When the tree is bounded with dry_run,
    Then the excess is reported and every entry survives."""
    # Arrange
    platform = "compiledir_fake-3.12-64"
    controller = tmp_path / platform
    controller.mkdir()
    worker = _worker_tree(tmp_path, "gw0", platform, entries=4)

    # Act
    results = budget.enforce_budget_tree(
        tmp_path, controller, max_entries=1, dry_run=True
    )

    # Assert
    assert dict(results)[worker].removed_lru == 3
    assert len(list(worker.iterdir())) == 4


def test_the_tree_summary_reports_the_total_that_grew_unnoticed(tmp_path):
    """Given a bounded tree,
    When it is summarized for the pytest header,
    Then the line carries the number of compiledirs and the total held.

    The total is the point. Seven per-directory lines saying "within budget"
    is noise nobody reads; one number that would have visibly climbed is the
    thing whose absence let this go unnoticed for weeks."""
    # Arrange
    platform = "compiledir_fake-3.12-64"
    controller = tmp_path / platform
    controller.mkdir()
    for name in ("gw0", "gw1"):
        _worker_tree(tmp_path, name, platform, entries=2)

    # Act
    results = budget.enforce_budget_tree(tmp_path, controller, max_entries=5)
    line = budget.summarize_tree(results, 5)

    # Assert
    assert "3 pruned" in line
    assert "at most 4 entries held in total" in line


# ---------------------------------------------------------------------------
# Seeding a cold worker compiledir from a warm one
# ---------------------------------------------------------------------------


def test_seeding_hardlinks_the_so_and_copies_the_key(tmp_path):
    """Given a warm compiledir and an empty target,
    When the target is seeded,
    Then the module is hard-linked and key.pkl is a private copy.

    The split is the whole design. key.pkl is the only file PyTensor rewrites
    in place -- it appends to it when a second key maps to one compiled module
    -- so sharing that inode across workers would let one worker's write land
    in every other worker's cache. Everything else is write-once, and it is
    98.7% of the bytes (.so 85.3%, .cpp 13.4%, measured over 148 entries)."""
    # Arrange
    warm = tmp_path / "warm"
    warm.mkdir()
    _make_entry(warm, "entry0")
    (warm / "entry0" / "mod.cpp").write_text("int main(){}")
    target = tmp_path / "gw1" / "compiledir_fake"

    # Act
    entries, linked, copied, fallbacks = budget.seed_compiledir(warm, target)

    # Assert
    assert (entries, fallbacks) == (1, 0)
    assert linked == 2  # mod.so and mod.cpp
    assert copied == 1  # key.pkl
    so_src = (warm / "entry0" / "mod.so").stat()
    so_dst = (target / "entry0" / "mod.so").stat()
    assert so_dst.st_ino == so_src.st_ino, "the .so must be a hard link"
    key_src = (warm / "entry0" / "key.pkl").stat()
    key_dst = (target / "entry0" / "key.pkl").stat()
    assert key_dst.st_ino != key_src.st_ino, "key.pkl must be a private copy"


def test_a_seeded_key_pkl_write_does_not_reach_the_source(tmp_path):
    """Given a seeded compiledir,
    When the seeded copy's key.pkl is rewritten,
    Then the warm source's key.pkl is untouched.

    The consequence of the inode split above, stated as behaviour rather than
    as st_ino: this is the corruption the design exists to prevent."""
    # Arrange
    warm = tmp_path / "warm"
    warm.mkdir()
    _make_entry(warm, "entry0")
    target = tmp_path / "gw1" / "compiledir_fake"
    budget.seed_compiledir(warm, target)

    # Act -- PyTensor rewrites the whole file, it does not append in place.
    (target / "entry0" / "key.pkl").write_bytes(b"a second key was added")

    # Assert
    assert (warm / "entry0" / "key.pkl").read_bytes() == b"fake pickle"


def test_seeding_skips_a_worker_that_is_already_warm(tmp_path):
    """Given one cold worker dir and one already holding the entries,
    When the tree is seeded,
    Then only the cold one is touched.

    Idempotence is what lets this run on every single pytest invocation. In
    steady state it costs one listdir per worker and copies nothing."""
    # Arrange
    platform = "compiledir_fake-3.12-64"
    controller = tmp_path / platform
    controller.mkdir()
    for i in range(4):
        _make_entry(controller, f"entry{i}")
    warm_worker = tmp_path / "gw0" / platform
    warm_worker.mkdir(parents=True)
    for i in range(4):
        _make_entry(warm_worker, f"entry{i}")

    # Act -- two workers: gw0 is warm, gw1 does not exist yet.
    stats = budget.seed_worker_compiledirs(tmp_path, controller, n_workers=2)

    # Assert
    assert stats.seeded_dirs == ["gw1"]
    assert stats.skipped_dirs == ["gw0"]
    assert sorted(p.name for p in (tmp_path / "gw1" / platform).iterdir()) == [
        f"entry{i}" for i in range(4)
    ]


def test_seeding_is_a_no_op_when_nothing_is_warm_yet(tmp_path):
    """Given a tree with no populated compiledir anywhere,
    When seeding runs,
    Then it does nothing and says so.

    The genuinely-first-ever run. There is nothing to copy from, so every
    worker compiles from scratch, which is correct and unavoidable."""
    # Arrange
    controller = tmp_path / "compiledir_fake"
    controller.mkdir()

    # Act
    stats = budget.seed_worker_compiledirs(tmp_path, controller, n_workers=4)

    # Assert
    assert stats.seeded_dirs == []
    assert stats.summary() == ""


def test_a_compiledir_holding_only_init_py_is_not_a_seed_source(tmp_path):
    """Given compiledirs that contain no cache entries, only an __init__.py,
    When seeding runs,
    Then nothing is seeded and nothing is claimed.

    PyTensor puts an ``__init__.py`` in every compiledir, so counting raw
    listdir names made a brand-new tree look warm enough to seed FROM. The
    observed symptom was a first run printing "seeded 2 worker compiledir(s)
    from a warm one (gw0, gw1): 0 entries, 0 hard-linked, 0 copied" -- a
    header line asserting work that did not happen."""
    # Arrange
    platform = "compiledir_fake-3.12-64"
    controller = tmp_path / platform
    controller.mkdir()
    (controller / "__init__.py").touch()
    for name in ("gw0", "gw1"):
        worker = tmp_path / name / platform
        worker.mkdir(parents=True)
        (worker / "__init__.py").touch()

    # Act
    stats = budget.seed_worker_compiledirs(tmp_path, controller, n_workers=2)

    # Assert
    assert stats.seeded_dirs == []
    assert stats.summary() == ""


def test_a_seed_that_copies_nothing_is_not_reported_as_a_seed(tmp_path):
    """Given a source whose entry count is non-zero but yields no files,
    When seeding runs,
    Then the target is reported as skipped rather than seeded.

    Belt to the previous test's braces: the guard is on what was actually
    copied, not only on how the source looked before starting."""
    # Arrange -- an entry directory that exists but is empty, so it is counted
    # by _entry_names and yet contributes no files to copy.
    platform = "compiledir_fake-3.12-64"
    controller = tmp_path / platform
    controller.mkdir()
    (controller / "hollow_entry").mkdir()

    # Act
    stats = budget.seed_worker_compiledirs(tmp_path, controller, n_workers=1)

    # Assert -- one entry walked, but zero files, so no claim of a seed.
    assert stats.seeded_dirs == []
    assert stats.entries == 0


def test_the_seed_source_is_the_warmest_tree_in_the_tree(tmp_path):
    """Given several compiledirs holding different numbers of entries,
    When the seed source is chosen,
    Then it is the one with the most entries."""
    # Arrange
    platform = "compiledir_fake"
    controller = tmp_path / platform
    controller.mkdir()
    _make_entry(controller, "entry0")
    fat = tmp_path / "gw2" / platform
    fat.mkdir(parents=True)
    for i in range(5):
        _make_entry(fat, f"entry{i}")

    # Act
    source = budget.choose_seed_source(tmp_path, controller)

    # Assert
    assert source == fat


def test_a_cross_device_fallback_is_counted_and_reported(
    tmp_path, monkeypatch
):
    """Given a filesystem where os.link fails,
    When a compiledir is seeded,
    Then the copies still happen and the fallback is reported loudly.

    Silence here would be expensive rather than wrong: the fallback turns a
    5-second metadata operation into a multi-minute byte copy. It is exactly
    how the first measurement of this code was taken by mistake -- the target
    was on a different filesystem from the source, every os.link raised EXDEV,
    and the run reported 0 hard-linked / 1988 copied without complaint."""
    # Arrange
    warm = tmp_path / "warm"
    warm.mkdir()
    _make_entry(warm, "entry0")

    def no_links(*args, **kwargs):
        raise OSError(18, "Invalid cross-device link")

    monkeypatch.setattr(budget.os, "link", no_links)

    # Act
    entries, linked, copied, fallbacks = budget.seed_compiledir(
        warm, tmp_path / "gw1" / "compiledir_fake"
    )
    stats = budget.SeedStats(
        seeded_dirs=["gw1"],
        entries=entries,
        linked=linked,
        copied=copied,
        link_fallbacks=fallbacks,
    )

    # Assert
    assert linked == 0
    assert fallbacks == 1  # the .so; key.pkl was always a copy
    assert (
        tmp_path / "gw1" / "compiledir_fake" / "entry0" / "mod.so"
    ).is_file()
    assert "fell back to copies" in stats.summary()


def test_pytensor_really_gets_cache_hits_from_a_seeded_compiledir(tmp_path):
    """Given a compiledir seeded from a warm one,
    When PyTensor compiles the same graph against it,
    Then no new cache entry is created and the result is unchanged.

    The other seeding tests prove the FILE MECHANICS -- hard links, a private
    key.pkl. They prove nothing about whether PyTensor recognizes the result
    as its own cache, which is the only claim that matters. A miss would have
    to write a new entry, so "no new entries" is the assertion.

    Subprocesses because ``base_compiledir`` is declared ``mutable=False``: it
    can only be set through PYTENSOR_FLAGS before pytensor is imported, so one
    process cannot exercise two compiledirs.

    ``linker=cvm`` is not incidental. PyTensor 3 made the "auto" linker numba,
    which caches into ``base/numba/`` and leaves the C compiledir EMPTY --
    which is exactly how the first version of this check passed while proving
    nothing: nothing cached, nothing recompiled, "hit". exozippy/__init__.py
    flips auto to cvm, so cvm is what the suite's cache actually holds and
    what this has to exercise.
    """
    # Arrange
    pytensor = pytest.importorskip("pytensor")
    if not pytensor.config.cxx:
        pytest.skip("no C compiler configured; there is no C cache to seed")

    build = textwrap.dedent(
        """
        import numpy as np
        import pytensor
        import pytensor.tensor as pt

        x = pt.dvector("x")
        y = pt.dvector("y")
        out = pt.log(pt.exp(x) + pt.exp(y)) * pt.sqrt(pt.abs(x - y) + 1.0)
        f = pytensor.function([x, y], out, mode="FAST_RUN")
        r = f(np.array([1.0, 2.0]), np.array([0.5, 0.25]))
        print("RESULT", " ".join(f"{v:.9f}" for v in r))
        print("COMPILEDIR", pytensor.config.compiledir)
        """
    )

    def compile_in(base):
        env = dict(os.environ)
        env["PYTENSOR_FLAGS"] = f"base_compiledir={base},linker=cvm"
        env.pop("EXOZIPPY_TEST_COMPILEDIR", None)
        done = subprocess.run(
            [sys.executable, "-c", build],
            capture_output=True,
            text=True,
            env=env,
            timeout=900,
        )
        assert done.returncode == 0, done.stderr[-2000:]
        fields = dict(
            line.split(" ", 1)
            for line in done.stdout.splitlines()
            if line.startswith(("RESULT ", "COMPILEDIR "))
        )
        return fields["RESULT"].strip(), Path(fields["COMPILEDIR"].strip())

    def entry_names(compiledir):
        if not compiledir.is_dir():
            return set()
        return {
            name
            for name in os.listdir(compiledir)
            if name != "lock_dir" and (compiledir / name).is_dir()
        }

    src_base, dst_base = tmp_path / "src", tmp_path / "dst"
    src_base.mkdir()
    dst_base.mkdir()

    # Act -- compile cold, seed from it, compile the same graph against the seed.
    cold_result, src_dir = compile_in(src_base)
    assert entry_names(src_dir), (
        "the cold compile cached nothing, so this test proves nothing"
    )

    _, _, _, fallbacks = budget.seed_compiledir(
        src_dir, dst_base / src_dir.name
    )
    assert fallbacks == 0, (
        "os.link fell back to copying, so tmp_path spans filesystems and this "
        "run measured a copy rather than the mechanism under test"
    )
    seeded = entry_names(dst_base / src_dir.name)
    warm_result, dst_dir = compile_in(dst_base)

    # Assert
    assert entry_names(dst_dir) - seeded == set(), (
        "the seeded compiledir was a MISS: PyTensor recompiled instead of "
        "reusing the entries hard-linked into place"
    )
    assert warm_result == cold_result
