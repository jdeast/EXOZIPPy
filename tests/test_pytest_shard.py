"""Tests for the CI shard split (scripts/pytest_shard.py).

The failure mode this guards is specific and quiet: a split that drops a file
still reports green on every shard, so the coverage is gone and nothing says
so. Hence the properties pinned here are exhaustiveness and disjointness
against the REAL tests/ directory, not just against a fixture.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]

# By path, for the same reason the other scripts/ tests do it: scripts/ is not
# a package, and putting it on sys.path would make getdata.py / mkparam.py
# importable as top-level modules.
_NAME = "_pytest_shard"
if _NAME in sys.modules:
    shard_mod = sys.modules[_NAME]
else:
    _SPEC = importlib.util.spec_from_file_location(
        _NAME, _REPO_ROOT / "scripts" / "pytest_shard.py"
    )
    shard_mod = importlib.util.module_from_spec(_SPEC)
    sys.modules[_NAME] = shard_mod
    _SPEC.loader.exec_module(shard_mod)


@pytest.mark.parametrize("total", [1, 2, 3, 4])
def test_the_real_test_tree_partitions_exactly(total):
    """Given this repository's own tests/ directory,
    When it is split into shards,
    Then every file lands in exactly one shard.

    Against the real directory rather than a fixture, because the thing that
    breaks this is a real file appearing somewhere the globs do not look --
    a subpackage, or a `*_test.py`. A fixture would keep passing."""
    # Arrange
    files = shard_mod.discover_test_files(_REPO_ROOT / "tests")
    assert files, "no test files discovered at all"

    # Act
    shards = [shard_mod.shard(files, i, total) for i in range(1, total + 1)]

    # Assert -- exhaustive and disjoint.
    union = [f for s in shards for f in s]
    assert sorted(union) == files
    assert len(union) == len(set(union)), "a file landed in two shards"


def test_this_very_file_is_in_some_shard():
    """Given the shard split,
    When it runs over tests/,
    Then this file is in it.

    A canary for the glob: if someone narrows the patterns or makes the walk
    non-recursive, the file asserting it is covered stops being covered, and
    the assertion is what notices."""
    # Arrange / Act
    files = shard_mod.discover_test_files(_REPO_ROOT / "tests")

    # Assert
    assert Path(__file__).resolve().as_posix() in files


def test_both_of_pytests_default_patterns_are_collected(tmp_path):
    """Given a tests dir holding test_*.py and *_test.py and a non-test file,
    When the files are discovered,
    Then both test patterns are found and the other file is not.

    `*_test.py` matters even though the repo has none today: pytest's default
    `python_files` collects it, so a `test_*.py`-only glob would hand pytest a
    file list that silently omits it."""
    # Arrange
    (tmp_path / "test_alpha.py").touch()
    (tmp_path / "beta_test.py").touch()
    (tmp_path / "conftest.py").touch()
    (tmp_path / "helper.py").touch()

    # Act
    found = {Path(f).name for f in shard_mod.discover_test_files(tmp_path)}

    # Assert
    assert found == {"test_alpha.py", "beta_test.py"}


def test_a_test_file_in_a_subdirectory_is_not_lost(tmp_path):
    """Given a test file nested in a subpackage,
    When the files are discovered,
    Then it is included.

    tests/ is flat today. If it stops being flat, a non-recursive walk would
    drop the whole subtree out of every shard without failing anything."""
    # Arrange
    nested = tmp_path / "sub" / "deeper"
    nested.mkdir(parents=True)
    (nested / "test_nested.py").touch()
    (tmp_path / "test_top.py").touch()

    # Act
    found = {Path(f).name for f in shard_mod.discover_test_files(tmp_path)}

    # Assert
    assert found == {"test_top.py", "test_nested.py"}


def test_the_split_is_balanced_on_a_long_tail(tmp_path):
    """Given many files of widely varying cost,
    When they are round-robined,
    Then the shards come out close to even.

    This is the property that justifies round-robin over a duration-aware
    pack. Measured on the real suite: 2 shards land within 1.04x of ideal.
    Here the same shape is reproduced synthetically -- one 5% file and a long
    tail -- so the reasoning is pinned rather than just asserted in a comment."""
    # Arrange -- costs shaped like the measured suite: no file dominant.
    costs = {f"test_{i:03d}.py": 1.0 / (i + 1) for i in range(200)}
    files = sorted(costs)

    # Act
    shards = [shard_mod.shard(files, i, 2) for i in range(1, 3)]
    loads = [sum(costs[Path(f).name] for f in s) for s in shards]

    # Assert
    ideal = sum(costs.values()) / 2
    assert max(loads) / ideal < 1.15


@pytest.mark.parametrize("index,total", [(0, 2), (3, 2), (-1, 4)])
def test_an_out_of_range_shard_index_is_rejected(index, total):
    """Given a shard index outside 1..total,
    When a shard is requested,
    Then it raises rather than silently returning something.

    Shard indices come from a CI matrix, where an off-by-one is easy and would
    otherwise mean a shard that runs nothing (green, and no coverage) or two
    shards running the same files."""
    # Arrange / Act / Assert
    with pytest.raises(ValueError):
        shard_mod.shard(["test_a.py", "test_b.py"], index, total)


# ---------------------------------------------------------------------------
# The duration-aware split
# ---------------------------------------------------------------------------


def _durations_file(tmp_path, durations):
    (tmp_path / "durations.json").write_text(
        json.dumps({"durations": durations})
    )


def test_the_shipped_durations_file_covers_almost_every_test_file():
    """Given the durations file checked in beside the tests,
    When it is compared against the files actually present,
    Then it accounts for the large majority of them.

    Not an equality assertion, deliberately: new test files land constantly
    and must not turn the suite red for it. But a file that has drifted so far
    that most weights are guesses has stopped doing its job, and the split
    silently degrades toward round-robin -- which at 4 shards is a third worse.
    Regenerate with scripts/gen_durations.py."""
    # Arrange
    tests_dir = _REPO_ROOT / "tests"
    files = shard_mod.discover_test_files(tests_dir)
    durations = shard_mod.load_durations(tests_dir)
    assert durations, "tests/durations.json is missing or unreadable"

    # Act
    unknown = [f for f in files if Path(f).name not in durations]

    # Assert
    share = len(unknown) / len(files)
    assert share < 0.35, (
        f"{len(unknown)} of {len(files)} test files ({share:.0%}) have no "
        f"recorded duration; regenerate tests/durations.json"
    )


def test_duration_aware_packing_beats_round_robin_on_the_real_suite():
    """Given the real per-file durations,
    When the suite is split four ways both ways,
    Then the duration-aware split is measurably better balanced.

    This is the entire justification for carrying a durations file at all. If
    it ever stops being true the file is dead weight and should go."""
    # Arrange
    tests_dir = _REPO_ROOT / "tests"
    files = shard_mod.discover_test_files(tests_dir)
    durations = shard_mod.load_durations(tests_dir)
    weights = shard_mod.weigh(files, durations)

    def worst(groups):
        loads = [sum(weights[f] for f in g) for g in groups]
        return max(loads) / (sum(loads) / len(loads))

    # Act
    packed = worst(shard_mod.pack(files, 4, weights))
    robin = worst([files[i::4] for i in range(4)])

    # Assert
    assert packed < 1.10, f"duration-aware split is {packed:.2f}x of ideal"
    assert packed < robin, (
        f"duration-aware ({packed:.2f}x) is no better than round-robin "
        f"({robin:.2f}x); the durations file is not earning its keep"
    )


def test_packing_partitions_exactly_and_is_deterministic(tmp_path):
    """Given weighted files,
    When they are packed,
    Then every file appears exactly once and repeated runs agree.

    Determinism is not cosmetic here: each shard computes the partition
    independently in its own job, so two of them disagreeing would double-run
    some files and skip others."""
    # Arrange
    files = [f"tests/test_{i:03d}.py" for i in range(37)]
    weights = {f: (i % 7) + 1.0 for i, f in enumerate(files)}

    # Act
    first = shard_mod.pack(files, 4, weights)
    second = shard_mod.pack(files, 4, weights)

    # Assert
    assert first == second
    union = [f for g in first for f in g]
    assert sorted(union) == sorted(files)
    assert len(union) == len(set(union))


def test_an_unrecorded_file_is_charged_the_median_not_zero(tmp_path):
    """Given a file absent from the durations record,
    When the files are weighed,
    Then it costs the median of the known files.

    Zero is the trap: every newly added file would look free, so the packer
    would pile all of them into one shard -- and new files are exactly the ones
    nobody has measured yet."""
    # Arrange
    durations = {"test_a.py": 1.0, "test_b.py": 3.0, "test_c.py": 100.0}
    files = [
        "tests/test_a.py",
        "tests/test_b.py",
        "tests/test_c.py",
        "tests/test_new.py",
    ]

    # Act
    weights = shard_mod.weigh(files, durations)

    # Assert -- median of (1, 3, 100) is 3, not 0 and not the mean of 34.7.
    assert weights["tests/test_new.py"] == 3.0


def test_a_corrupt_durations_file_degrades_to_round_robin(tmp_path):
    """Given a durations file that is not usable JSON,
    When it is loaded,
    Then the result is empty rather than an exception.

    The file is a weighting HINT. A truncated or hand-mangled one must cost
    balance, never correctness -- the caller then falls back to round-robin,
    which still partitions every file exactly once."""
    # Arrange
    (tmp_path / "durations.json").write_text("{ this is not json")

    # Act
    durations = shard_mod.load_durations(tmp_path)

    # Assert
    assert durations == {}


def test_a_missing_durations_file_is_not_an_error(tmp_path):
    """Given no durations file at all,
    When it is loaded,
    Then the result is empty.

    The state of a fresh checkout before the file existed, and of any tree
    where someone deleted it."""
    # Arrange / Act
    durations = shard_mod.load_durations(tmp_path)

    # Assert
    assert durations == {}


def test_packing_puts_the_heaviest_files_in_different_shards(tmp_path):
    """Given a few dominant files among many cheap ones,
    When they are packed into as many shards as there are heavy files,
    Then no shard gets two of them.

    The property that makes the split worth computing: the slow files are what
    set a shard's wall clock, so concentrating two of them in one shard wastes
    the whole point."""
    # Arrange
    heavy = [f"tests/test_heavy_{i}.py" for i in range(4)]
    light = [f"tests/test_light_{i:02d}.py" for i in range(40)]
    files = sorted(heavy + light)
    weights = {f: (100.0 if f in heavy else 1.0) for f in files}

    # Act
    groups = shard_mod.pack(files, 4, weights)

    # Assert
    for group in groups:
        assert sum(1 for f in group if f in heavy) == 1


def test_the_generator_and_the_loader_agree_on_the_file_format(tmp_path):
    """Given a pytest --durations transcript,
    When the generator writes a durations file and the loader reads it back,
    Then the weights survive the round trip.

    These are two separate scripts with no shared code, joined only by a JSON
    shape. If they drift, `load_durations` returns {} and the split silently
    degrades to round-robin -- no error, just a third of the balance gone at
    4 shards. This is the contract, pinned."""
    # Arrange -- the exact shape pytest emits for --durations.
    transcript = tmp_path / "durations.txt"
    transcript.write_text(
        "============ slowest durations ============\n"
        "12.34s call     tests/test_alpha.py::test_one\n"
        "1.20s setup    tests/test_alpha.py::test_one\n"
        "5.00s call     tests/test_beta.py::test_two[case-1]\n"
        "0.10s teardown tests/test_beta.py::test_two[case-1]\n"
        "not a duration line at all\n"
    )
    out = tmp_path / "durations.json"

    # Act -- the generator is a script, so drive it the way a developer would.
    done = subprocess.run(
        [
            sys.executable,
            str(_REPO_ROOT / "scripts" / "gen_durations.py"),
            str(transcript),
            "--output",
            str(out),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr
    loaded = shard_mod.load_durations(tmp_path)

    # Assert -- summed across call/setup/teardown, keyed by BASENAME.
    assert loaded == {"test_alpha.py": 13.54, "test_beta.py": 5.10}


def test_several_transcripts_merge_into_one_whole_suite(tmp_path):
    """Given one transcript per CI shard, each covering different files,
    When they are merged,
    Then the result is the union with each file's cost intact.

    This is the shape a CI run produces: every job runs ONE shard and uploads
    its own transcript, so a whole suite only exists as the union of all four.
    If merging silently dropped or double-counted a file the weights would be
    wrong in a way nothing else would notice -- the split would still
    partition correctly, just badly."""
    # Arrange -- two shards, disjoint files, as CI would emit them.
    shard1 = tmp_path / "s1.txt"
    shard1.write_text(
        "10.00s call     tests/test_alpha.py::test_one\n"
        "2.00s setup    tests/test_alpha.py::test_one\n"
    )
    shard2 = tmp_path / "s2.txt"
    shard2.write_text("5.00s call     tests/test_beta.py::test_two\n")
    out = tmp_path / "durations.json"

    # Act
    done = subprocess.run(
        [
            sys.executable,
            str(_REPO_ROOT / "scripts" / "gen_durations.py"),
            str(shard1),
            str(shard2),
            "--output",
            str(out),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr
    loaded = shard_mod.load_durations(tmp_path)

    # Assert
    assert loaded == {"test_alpha.py": 12.0, "test_beta.py": 5.0}


def test_the_recorded_source_says_where_the_weights_came_from(tmp_path):
    """Given a --source note,
    When the durations file is written,
    Then it is recorded in the file.

    Load-bearing metadata rather than decoration: weights measured on a
    workstation and weights measured on a CI runner produce measurably
    different balance, so "which machine was this?" has to survive in the
    file. It is the question anyone debugging a lopsided shard asks first."""
    # Arrange
    transcript = tmp_path / "t.txt"
    transcript.write_text("1.00s call     tests/test_a.py::test_x\n")
    out = tmp_path / "durations.json"
    note = "CI run 12345, ubuntu-latest 3.12, 4 shards at -n4"

    # Act
    done = subprocess.run(
        [
            sys.executable,
            str(_REPO_ROOT / "scripts" / "gen_durations.py"),
            str(transcript),
            "--output",
            str(out),
            "--source",
            note,
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr

    # Assert
    assert json.loads(out.read_text())["_generated_from"] == note


# ---------------------------------------------------------------------------
# The loadfile floor, the split file, and the drift-prevention loop
# ---------------------------------------------------------------------------

_GEN_NAME = "_gen_durations"
if _GEN_NAME in sys.modules:
    gen_mod = sys.modules[_GEN_NAME]
else:
    _GEN_SPEC = importlib.util.spec_from_file_location(
        _GEN_NAME, _REPO_ROOT / "scripts" / "gen_durations.py"
    )
    gen_mod = importlib.util.module_from_spec(_GEN_SPEC)
    sys.modules[_GEN_NAME] = gen_mod
    _GEN_SPEC.loader.exec_module(gen_mod)


@pytest.mark.parametrize("total", [3, 4])
def test_duration_aware_packing_partitions_the_real_tree_exactly(total):
    """Given this repository's tests/ and its shipped durations file,
    When the files are PACKED (the CI path, not round-robin) 3 and 4 ways,
    Then every file lands in exactly one shard.

    3 and 4 because those are the two shard counts CI runs today (macOS and
    ubuntu). The round-robin partition is pinned above; this is the one the
    matrix actually uses, against the real tree, so a real file the packer
    somehow mishandles is caught here and not by a green shard."""
    # Arrange
    tests_dir = _REPO_ROOT / "tests"
    files = shard_mod.discover_test_files(tests_dir)
    durations = shard_mod.load_durations(tests_dir)
    assert durations, "tests/durations.json is missing or unreadable"

    # Act
    groups = shard_mod.pack(files, total, shard_mod.weigh(files, durations))

    # Assert
    union = [f for g in groups for f in g]
    assert sorted(union) == files
    assert len(union) == len(set(union)), "a file landed in two shards"
    assert all(groups), "an empty shard would be a job that runs nothing"


def test_the_mulens_acceptance_gate_is_two_weighted_files():
    """Given the test tree and the durations file,
    When the mulens acceptance files are looked up,
    Then both halves are discovered, both carry a recorded weight, and the
    single file they replaced is gone.

    The split exists because --dist loadfile pins a file to one worker and
    the single 873 s file set the shard floor (2026-09-14). Folding it back
    together would fail nothing else; the weights matter because a half that
    is absent from the map is charged the median, which is precisely how the
    original was mis-packed."""
    # Arrange
    tests_dir = _REPO_ROOT / "tests"
    names = {Path(f).name for f in shard_mod.discover_test_files(tests_dir)}
    durations = shard_mod.load_durations(tests_dir)

    # Assert
    for half in ("test_mulens_acceptance_a.py", "test_mulens_acceptance_b.py"):
        assert half in names, f"{half} is not collected"
        assert half in durations, f"{half} has no recorded weight"
        assert durations[half] > 0
    assert "test_mulens_acceptance.py" not in names
    assert "test_mulens_acceptance.py" not in durations


def test_the_generator_keeps_tracked_examples_and_drops_only_named_ones():
    """Given transcript rows parametrized over an example directory,
    When they are parsed with and without that example named local-only,
    Then the default keeps every row and the exclusion drops exactly them.

    The regression: a hardcoded ("ob09020",) outlived the example being
    untracked, and the generator dropped 316 s that CI was paying on every
    run. The default must therefore be "exclude nothing", with the exclusion
    an explicit, derived input -- and matched inside the parametrize brackets
    only, so a test whose NAME mentions an example is not swept up."""
    # Arrange -- the shapes CI actually emits for ob09020.
    transcript = (
        "141.15s call     tests/test_mulens_acceptance_a.py::"
        "test_the_model_still_matches_its_recorded_decomposition[ob09020]\n"
        "35.00s call     tests/test_examples_prepare.py::"
        "test_shipped_example_prepares[ob09020/ob09020.yaml]\n"
        "2.00s call     tests/test_examples_prepare.py::"
        "test_shipped_example_prepares[kelt4/kelt4.yaml]\n"
        "1.00s call     tests/test_ob09020_helpers.py::test_name_mentions_it\n"
    )

    # Act
    kept, skipped = gen_mod.parse(transcript)
    dropped, excluded = gen_mod.parse(transcript, local_only=("ob09020",))

    # Assert -- nothing excluded by default...
    assert skipped == {}
    assert kept["tests/test_mulens_acceptance_a.py"] == 141.15
    assert kept["tests/test_examples_prepare.py"] == 37.0
    # ...and the named exclusion takes exactly the parametrized rows.
    assert "tests/test_mulens_acceptance_a.py" not in dropped
    assert dropped["tests/test_examples_prepare.py"] == 2.0
    assert excluded == {
        "tests/test_mulens_acceptance_a.py": 141.15,
        "tests/test_examples_prepare.py": 35.0,
    }
    # A test whose function name carries the tag is not a parametrized case
    # of that example and stays counted.
    assert dropped["tests/test_ob09020_helpers.py"] == 1.0


def test_local_only_examples_are_the_untracked_example_dirs(tmp_path):
    """Given a repository with one tracked and one untracked example,
    When the local-only set is derived,
    Then it names the untracked one only -- and nothing when there is no
    examples/ directory at all.

    Derived from git rather than written down, because the written-down
    version went stale the moment the example was committed."""
    # Arrange -- a throwaway repository, not this one: the real tree's
    # untracked examples are whatever the developer happens to have.
    repo = tmp_path / "repo"
    (repo / "examples" / "tracked").mkdir(parents=True)
    (repo / "examples" / "tracked" / "t.yaml").write_text("a: 1\n")
    (repo / "examples" / "untracked").mkdir()
    (repo / "examples" / "untracked" / "u.yaml").write_text("b: 2\n")
    (repo / "examples" / "loose_file.txt").write_text("not a dir\n")
    env = {
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
        "HOME": str(tmp_path),
    }
    for cmd in (
        ["git", "init", "-q"],
        ["git", "add", "examples/tracked"],
        ["git", "commit", "-q", "-m", "tracked example"],
    ):
        subprocess.run(cmd, cwd=repo, check=True, env=env, timeout=60)

    # Act / Assert
    assert gen_mod.local_only_examples(repo) == ("untracked",)
    assert gen_mod.local_only_examples(tmp_path / "nowhere") == ()


def test_measured_age_is_counted_from_the_measurement_date():
    """Given a durations payload with a _measured_on stamp,
    When its age is asked for,
    Then it is the days since that stamp -- and None when there is none.

    The stamp is the transcript's date, so this is the age of the
    MEASUREMENT; the point of reporting it is judging staleness."""
    import datetime

    today = datetime.date(2026, 9, 14)
    assert (
        shard_mod.measured_age_days({"_measured_on": "2026-08-25"}, today)
        == 20
    )
    assert shard_mod.measured_age_days({}, today) is None
    assert shard_mod.measured_age_days({"_measured_on": "soon"}, today) is None


def test_verify_writes_the_job_summary_and_annotates_from_shard_one(
    tmp_path, monkeypatch, capsys
):
    """Given a tests dir with a file the durations map does not know,
    When --verify runs with GITHUB_STEP_SUMMARY set,
    Then the summary file names the absent file and the age, the ::warning::
    annotation is emitted from shard 1 and NOT from shard 2, and the file
    list is still printed.

    This is the loud half of drift prevention. A warning line in a log
    nobody opens is how 20 days of staleness went unnoticed; the summary
    page and the annotation are where people actually look. From shard 1
    only so a run carries one annotation per os+python leg, not one per
    job."""
    # Arrange
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    for name in ("test_a.py", "test_b.py", "test_new.py"):
        (tests_dir / name).touch()
    (tests_dir / "durations.json").write_text(
        json.dumps(
            {
                "_measured_on": "2026-01-01",
                "_generated_from": "CI run 1",
                "durations": {"test_a.py": 10.0, "test_b.py": 4.0},
            }
        )
    )
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))

    # Act -- shard 1
    rc = shard_mod.main(
        [
            "--shard",
            "1",
            "--of",
            "2",
            "--verify",
            "--tests-dir",
            str(tests_dir),
        ]
    )
    out1 = capsys.readouterr()

    # Assert
    assert rc == 0
    assert "::warning title=Stale shard durations::" in out1.out
    assert "1 of 3 test files" in out1.err
    text = summary.read_text()
    assert "test_new.py" in text, text
    assert "absent" in text
    assert "measured on 2026-01-01" in text
    assert "partition cleanly into 2 shards" in out1.err
    # The file list itself is the LAST stdout line, untouched by the report.
    assert out1.out.strip().splitlines()[-1].endswith("test_a.py")

    # Act -- shard 2: same warning on stderr, no annotation.
    summary.unlink()
    rc = shard_mod.main(
        [
            "--shard",
            "2",
            "--of",
            "2",
            "--verify",
            "--tests-dir",
            str(tests_dir),
        ]
    )
    out2 = capsys.readouterr()
    assert rc == 0
    assert "::warning" not in out2.out
    assert "WARNING" in out2.err
    assert summary.exists(), "the summary is written from every shard"


def test_without_the_summary_variable_nothing_is_written(
    tmp_path, monkeypatch
):
    """Given no GITHUB_STEP_SUMMARY in the environment,
    When the summary writer is called,
    Then it reports that it wrote nothing and touches no file.

    Off CI the variable is unset; a developer's shell must not grow a
    stray file."""
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    monkeypatch.chdir(tmp_path)
    assert shard_mod.write_step_summary("t", ["x"]) is False
    assert list(tmp_path.iterdir()) == []


def test_balance_json_prices_the_stale_packing_by_the_new_weights(
    tmp_path, capsys
):
    """Given an old and a new durations file over the same four test files,
    When --balance-json --compare-to is asked for,
    Then the stale packing is the OLD file's partition costed at the NEW
    weights, and the current packing is the new file's own.

    This is the number the refresh workflow decides on. Priced any other way
    -- old packing at old weights -- a stale file always looks balanced,
    because it balanced the numbers it believed."""
    # Arrange -- equal old weights put {a, c} and {b, d} together; the new
    # measurement says a and c are the heavy ones, so that packing is bad.
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    for name in "abcd":
        (tests_dir / f"test_{name}.py").touch()
    old = tmp_path / "old.json"
    old.write_text(
        json.dumps({"durations": {f"test_{n}.py": 1.0 for n in "abcd"}})
    )
    (tests_dir / "durations.json").write_text(
        json.dumps(
            {
                "_measured_on": "2026-09-14",
                "durations": {
                    "test_a.py": 10.0,
                    "test_b.py": 1.0,
                    "test_c.py": 10.0,
                    "test_d.py": 1.0,
                },
            }
        )
    )

    # Act
    rc = shard_mod.main(
        [
            "--of",
            "2",
            "--balance-json",
            "--compare-to",
            str(old),
            "--tests-dir",
            str(tests_dir),
        ]
    )
    report = json.loads(capsys.readouterr().out)

    # Assert
    assert rc == 0
    assert report["worst_shard_seconds"] == 11.0
    assert report["ideal_shard_seconds"] == 11.0
    assert report["stale_packing_worst_shard_seconds"] == 20.0
    assert report["unknown"] == 0 and report["stale_unknown"] == 0
    assert report["slowest_file"] == "test_a.py"
    assert report["measured_on"] == "2026-09-14"
    assert "groups" not in report, "the file lists are not part of the report"


def test_the_refresh_workflow_exists_and_drives_the_scripts():
    """Given the drift-prevention workflow,
    When it is read,
    Then it is scheduled, dispatchable, and calls both scripts by name with
    the flags this module tests.

    A light contract test so that renaming a flag here cannot silently
    orphan the workflow that depends on it -- the workflow is YAML, so no
    import would notice."""
    text = (
        _REPO_ROOT / ".github" / "workflows" / "refresh-durations.yml"
    ).read_text()
    assert "schedule:" in text and "workflow_dispatch:" in text
    assert "scripts/gen_durations.py" in text
    assert "scripts/pytest_shard.py" in text
    assert "--balance-json" in text and "--compare-to" in text
    assert "durations-ubuntu-latest-3.12-*" in text
    # The matrix step passes the per-OS shard count, not a literal 4.
    tests_yml = (
        _REPO_ROOT / ".github" / "workflows" / "tests.yml"
    ).read_text()
    assert "--of ${{ matrix.shards }} --verify" in tests_yml
