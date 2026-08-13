"""
Tests for the shared Zenodo fetch-verify-cache core
(utilities/zenodo.py) and the two wrappers that own asset tables
(components/sed/make_bc.ensure_model_data, models/MIST/eep_grid).

Nothing here touches the network: urlretrieve is always monkeypatched.

The machine-level cache is switched off for the whole suite by conftest's
autouse _no_shared_download_cache fixture, so every test that is not about
the cache sees exactly the pre-cache behaviour. The cache tests below opt
back in with the `cache_root` fixture.
"""

import errno
import hashlib
import multiprocessing
import os
import time
import urllib.error
from pathlib import Path

import pytest

from conftest import requires_fork
from exozippy.utilities import zenodo


def _fake_assets(payload=b"hello", name="f.csv"):
    """A one-file asset table whose size/md5 match `payload`."""
    return {
        name: {
            "url": "https://example.invalid/f.csv",
            "size": len(payload),
            "md5": hashlib.md5(payload).hexdigest(),
        }
    }


@pytest.fixture
def cache_root(tmp_path, monkeypatch):
    """Point the machine-level cache at a private directory and enable it."""
    root = tmp_path / "xdg-cache" / "exozippy"
    monkeypatch.setenv("EXOZIPPY_CACHE_DIR", str(root))
    return root


def _entry_of(cache_root, assets, name="f.csv"):
    """Where the cache keeps `name`: content-addressed by its pinned md5."""
    return Path(cache_root) / "downloads" / f"{assets[name]['md5']}-{name}"


def _refuse_to_download(url, dest):
    raise AssertionError(f"network access attempted for {url}")


# --- retry / failure behavior --------------------------------------------


def test_download_retries_a_transient_gateway_error(tmp_path, monkeypatch):
    """
    Given Zenodo returns 504 once and then succeeds,
    When fetch_assets runs,
    Then it retries and the file lands, rather than failing the caller.

    Regression: two consecutive Dependabot PRs went red purely because a
    250 MB Zenodo fetch returned `HTTPError: HTTP Error 504: Gateway
    Time-out` mid-CI. A transient gateway error must cost seconds, not a run.
    """
    # ARRANGE
    payload = b"hello"
    monkeypatch.setattr(zenodo.time, "sleep", lambda *a: None)

    calls = []

    def flaky(url, dest):
        calls.append(url)
        if len(calls) == 1:
            raise urllib.error.HTTPError(
                url, 504, "Gateway Time-out", {}, None
            )
        Path(dest).write_bytes(payload)

    monkeypatch.setattr(zenodo.urllib.request, "urlretrieve", flaky)

    # ACT
    zenodo.fetch_assets(_fake_assets(payload), tmp_path)

    # ASSERT
    assert len(calls) == 2, "should have retried exactly once"
    assert (tmp_path / "f.csv").read_bytes() == payload


def test_download_does_not_retry_a_404(tmp_path, monkeypatch):
    """
    Given the URL is simply wrong (404),
    When fetch_assets runs,
    Then it raises immediately instead of burning the backoff schedule.
    """
    # ARRANGE
    monkeypatch.setattr(zenodo.time, "sleep", lambda *a: None)

    calls = []

    def missing(url, dest):
        calls.append(url)
        raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)

    monkeypatch.setattr(zenodo.urllib.request, "urlretrieve", missing)

    # ACT / ASSERT
    with pytest.raises(urllib.error.HTTPError):
        zenodo.fetch_assets(_fake_assets(), tmp_path)
    assert len(calls) == 1, "a 404 is not transient; do not retry it"


def test_download_gives_up_after_the_attempt_budget(tmp_path, monkeypatch):
    """
    Given every attempt fails,
    When the budget is exhausted,
    Then a RuntimeError names the file and no .part is left behind.
    """
    # ARRANGE
    monkeypatch.setattr(zenodo.time, "sleep", lambda *a: None)

    calls = []

    def always_504(url, dest):
        calls.append(url)
        raise urllib.error.HTTPError(url, 504, "Gateway Time-out", {}, None)

    monkeypatch.setattr(zenodo.urllib.request, "urlretrieve", always_504)

    # ACT / ASSERT
    with pytest.raises(RuntimeError, match="f.csv"):
        zenodo.fetch_assets(_fake_assets(), tmp_path)
    assert len(calls) == zenodo._DOWNLOAD_ATTEMPTS
    assert not list(tmp_path.glob("*.part")), "left a .part behind"


# --- integrity: .part staging, size and md5 -------------------------------


def test_download_stages_on_a_part_file_and_renames_only_when_intact(
    tmp_path, monkeypatch
):
    """
    Given a download in progress,
    When fetch_assets writes it,
    Then the bytes land on <name>.part and the destination does not exist
    until size and md5 have both been verified.

    This is the whole point of the staging dance: urlretrieve writing
    straight to the destination is how a truncated body became a
    permanently cached half-file, surfacing later as
    `pandas.errors.ParserError: EOF inside string starting at row 11248`
    in a test that had nothing to do with downloading.
    """
    # ARRANGE
    payload = b"hello"
    seen = {}

    def observant(url, dest):
        dest = Path(dest)
        seen["dest_arg"] = dest.name
        dest.write_bytes(payload)
        # Mid-download: the real destination must not exist yet.
        seen["final_exists_during"] = (tmp_path / "f.csv").exists()

    monkeypatch.setattr(zenodo.urllib.request, "urlretrieve", observant)

    # ACT
    zenodo.fetch_assets(_fake_assets(payload), tmp_path)

    # ASSERT
    assert seen["dest_arg"] == "f.csv.part"
    assert seen["final_exists_during"] is False
    assert (tmp_path / "f.csv").read_bytes() == payload
    assert not list(tmp_path.glob("*.part"))


def test_a_truncated_download_never_reaches_the_destination(
    tmp_path, monkeypatch
):
    """
    Given every fetch returns a short body,
    When fetch_assets runs,
    Then it raises and the destination is never created.
    """
    # ARRANGE
    monkeypatch.setattr(zenodo.time, "sleep", lambda *a: None)

    def truncated(url, dest):
        Path(dest).write_bytes(b"hel")  # payload is b"hello"

    monkeypatch.setattr(zenodo.urllib.request, "urlretrieve", truncated)

    # ACT / ASSERT
    with pytest.raises(RuntimeError, match="truncated"):
        zenodo.fetch_assets(_fake_assets(b"hello"), tmp_path)
    assert not (tmp_path / "f.csv").exists()


def test_a_wrong_md5_never_reaches_the_destination(tmp_path, monkeypatch):
    """
    Given a fetch whose body has the right length but the wrong content,
    When fetch_assets runs,
    Then the md5 check rejects it and the destination is never created.
    """
    # ARRANGE
    monkeypatch.setattr(zenodo.time, "sleep", lambda *a: None)

    def wrong_bytes(url, dest):
        Path(dest).write_bytes(b"world")  # same length as b"hello"

    monkeypatch.setattr(zenodo.urllib.request, "urlretrieve", wrong_bytes)

    # ACT / ASSERT
    with pytest.raises(RuntimeError, match="md5"):
        zenodo.fetch_assets(_fake_assets(b"hello"), tmp_path)
    assert not (tmp_path / "f.csv").exists()


# --- caching --------------------------------------------------------------


def test_a_cached_file_of_the_right_size_is_not_refetched(
    tmp_path, monkeypatch
):
    """
    Given the asset is already cached at the pinned size,
    When fetch_assets runs,
    Then nothing is downloaded and no on_fetch callback fires.
    """
    # ARRANGE
    payload = b"hello"
    (tmp_path / "f.csv").write_bytes(payload)

    calls = []
    monkeypatch.setattr(
        zenodo.urllib.request,
        "urlretrieve",
        lambda url, dest: calls.append(url),
    )
    fetched = []

    # ACT
    zenodo.fetch_assets(
        _fake_assets(payload), tmp_path, on_fetch=fetched.append
    )

    # ASSERT
    assert calls == []
    assert fetched == []


def test_a_truncated_cached_file_is_refetched_not_raised(
    tmp_path, monkeypatch
):
    """
    Given a cached file whose size does not match the pinned size,
    When fetch_assets runs,
    Then it re-downloads over it rather than raising -- the recovery from a
    corrupt cache entry is unambiguous, so making the user delete it by hand
    would be gratuitous.
    """
    # ARRANGE
    payload = b"hello"
    (tmp_path / "f.csv").write_bytes(b"hel")  # truncated leftover

    def good(url, dest):
        Path(dest).write_bytes(payload)

    monkeypatch.setattr(zenodo.urllib.request, "urlretrieve", good)

    # ACT
    zenodo.fetch_assets(_fake_assets(payload), tmp_path)

    # ASSERT
    assert (tmp_path / "f.csv").read_bytes() == payload


# --- the machine-level shared cache ---------------------------------------


def test_cache_location_follows_xdg_and_the_env_override(monkeypatch):
    """
    Given the usual environments,
    When the cache root is resolved,
    Then it is $XDG_CACHE_HOME/exozippy, ~/.cache/exozippy without XDG, the
    override when EXOZIPPY_CACHE_DIR is set, and None when it is switched
    off -- never a temp filesystem someone's /tmp cannot hold 250 MB in.
    """
    # ARRANGE / ACT / ASSERT
    monkeypatch.delenv("EXOZIPPY_CACHE_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", "/xdg")
    assert zenodo.shared_cache_root() == Path("/xdg/exozippy")

    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    assert zenodo.shared_cache_root() == Path.home() / ".cache" / "exozippy"

    monkeypatch.setenv("EXOZIPPY_CACHE_DIR", "/scratch/mine")
    assert zenodo.shared_cache_root() == Path("/scratch/mine")

    for off in ("", "none", "0", "OFF"):
        monkeypatch.setenv("EXOZIPPY_CACHE_DIR", off)
        assert zenodo.shared_cache_root() is None


def test_a_second_destination_is_served_from_the_cache_with_no_network(
    tmp_path, cache_root, monkeypatch
):
    """
    Given one destination has already been populated on this machine,
    When a second destination (a second worktree) asks for the same asset,
    Then it is served from the machine-level cache without any download.

    This is the whole point: a dozen worktrees used to mean a dozen 250 MB
    fetches from a free academic host.
    """
    # ARRANGE
    payload = b"hello"
    assets = _fake_assets(payload)
    downloads = []

    def counting(url, dest):
        downloads.append(url)
        Path(dest).write_bytes(payload)

    monkeypatch.setattr(zenodo.urllib.request, "urlretrieve", counting)
    first = tmp_path / "worktree_a" / "BCs"
    zenodo.fetch_assets(assets, first)
    assert len(downloads) == 1

    # ACT -- a fetcher that would blow up if it were called at all
    monkeypatch.setattr(
        zenodo.urllib.request, "urlretrieve", _refuse_to_download
    )
    second = tmp_path / "worktree_b" / "BCs"
    fetched = []
    zenodo.fetch_assets(assets, second, on_fetch=fetched.append)

    # ASSERT
    assert (second / "f.csv").read_bytes() == payload
    assert len(downloads) == 1, "the second worktree re-downloaded"
    assert fetched == [], "a shared-cache hit is a cache hit, not a fetch"
    assert _entry_of(cache_root, assets).exists()


def test_the_cache_hard_links_so_n_worktrees_cost_one_copy(
    tmp_path, cache_root, monkeypatch
):
    """
    Given a cache on the same filesystem as the destinations,
    When two destinations are populated,
    Then all three paths are the same inode -- one copy on disk, one file
    against the user's quota, however many worktrees there are.
    """
    # ARRANGE
    payload = b"hello"
    assets = _fake_assets(payload)
    monkeypatch.setattr(
        zenodo.urllib.request,
        "urlretrieve",
        lambda url, dest: Path(dest).write_bytes(payload),
    )

    # ACT
    zenodo.fetch_assets(assets, tmp_path / "a")
    zenodo.fetch_assets(assets, tmp_path / "b")

    # ASSERT
    entry = _entry_of(cache_root, assets)
    a, b = tmp_path / "a" / "f.csv", tmp_path / "b" / "f.csv"
    assert a.stat().st_ino == entry.stat().st_ino
    assert b.stat().st_ino == entry.stat().st_ino
    assert entry.stat().st_nlink == 3
    assert b.read_bytes() == payload


def test_it_copies_when_the_filesystems_do_not_allow_a_hard_link(
    tmp_path, cache_root, monkeypatch
):
    """
    Given the cache and the destination are on different filesystems
    (os.link raising EXDEV -- the case detected by simply trying it),
    When the asset is materialized,
    Then it is copied instead, and the copy is a distinct, intact file.
    """
    # ARRANGE
    payload = b"hello"
    assets = _fake_assets(payload)
    monkeypatch.setattr(
        zenodo.urllib.request,
        "urlretrieve",
        lambda url, dest: Path(dest).write_bytes(payload),
    )

    def cross_device(src, dst):
        raise OSError(errno.EXDEV, "Invalid cross-device link")

    monkeypatch.setattr(zenodo.os, "link", cross_device)

    # ACT
    zenodo.fetch_assets(assets, tmp_path / "a")

    # ASSERT
    entry = _entry_of(cache_root, assets)
    dest = tmp_path / "a" / "f.csv"
    assert dest.read_bytes() == payload
    assert dest.stat().st_ino != entry.stat().st_ino, "expected a real copy"
    assert entry.read_bytes() == payload


def test_a_corrupt_cache_entry_is_refetched_not_propagated(
    tmp_path, cache_root, monkeypatch
):
    """
    Given a shared-cache entry with the right size but the wrong bytes,
    When a destination asks for the asset,
    Then the md5 check rejects the entry, the asset is re-downloaded, and
    both the destination and the repaired entry hold the real payload.

    A shared cache concentrates the truncated-download failure -- one bad
    entry would otherwise poison every worktree on the machine at once --
    so it is verified in full, not on size alone, every time it is adopted.
    """
    # ARRANGE
    payload = b"hello"
    assets = _fake_assets(payload)
    entry = _entry_of(cache_root, assets)
    entry.parent.mkdir(parents=True)
    entry.write_bytes(b"world")  # same length, wrong content

    downloads = []

    def counting(url, dest):
        downloads.append(url)
        Path(dest).write_bytes(payload)

    monkeypatch.setattr(zenodo.urllib.request, "urlretrieve", counting)

    # ACT
    zenodo.fetch_assets(assets, tmp_path / "a")

    # ASSERT
    assert (tmp_path / "a" / "f.csv").read_bytes() == payload
    assert len(downloads) == 1, "a corrupt entry must be re-fetched"
    assert entry.read_bytes() == payload, "the entry must be repaired"


def test_a_truncated_cache_entry_is_refetched(
    tmp_path, cache_root, monkeypatch
):
    """
    Given a shared-cache entry left short by a crashed writer,
    When a destination asks for the asset,
    Then the size check rejects it and it is re-downloaded.
    """
    # ARRANGE
    payload = b"hello"
    assets = _fake_assets(payload)
    entry = _entry_of(cache_root, assets)
    entry.parent.mkdir(parents=True)
    entry.write_bytes(b"hel")

    monkeypatch.setattr(
        zenodo.urllib.request,
        "urlretrieve",
        lambda url, dest: Path(dest).write_bytes(payload),
    )

    # ACT
    zenodo.fetch_assets(assets, tmp_path / "a")

    # ASSERT
    assert (tmp_path / "a" / "f.csv").read_bytes() == payload
    assert entry.read_bytes() == payload


def test_the_destination_never_sees_a_partial_file_via_the_cache(
    tmp_path, cache_root, monkeypatch
):
    """
    Given a download in progress through the shared cache,
    When fetch_assets writes it,
    Then the bytes land on a .part file inside the cache directory and
    neither the cache entry nor the destination exists until size and md5
    have been verified.

    Same staging guarantee as the no-cache path, one directory further
    back: the rename into the cache is what makes a concurrent reader see
    all-or-nothing.
    """
    # ARRANGE
    payload = b"hello"
    assets = _fake_assets(payload)
    entry = _entry_of(cache_root, assets)
    seen = {}

    def observant(url, dest):
        dest = Path(dest)
        seen["staged_in_cache"] = dest.parent == entry.parent
        seen["suffix"] = dest.suffix
        dest.write_bytes(payload)
        seen["entry_exists_during"] = entry.exists()
        seen["dest_exists_during"] = (tmp_path / "a" / "f.csv").exists()

    monkeypatch.setattr(zenodo.urllib.request, "urlretrieve", observant)

    # ACT
    zenodo.fetch_assets(assets, tmp_path / "a")

    # ASSERT
    assert seen["staged_in_cache"] is True
    assert seen["suffix"] == ".part"
    assert seen["entry_exists_during"] is False
    assert seen["dest_exists_during"] is False
    assert not list(entry.parent.glob("*.part"))


def test_an_existing_destination_is_adopted_into_the_cache(
    tmp_path, cache_root, monkeypatch
):
    """
    Given a checkout that already holds the asset (every worktree created
    before this cache existed does),
    When fetch_assets is called there and then in a fresh worktree,
    Then the existing file is hard-linked into the cache and serves the new
    worktree -- the ~250 MB already on disk is adopted, not re-downloaded.
    """
    # ARRANGE
    payload = b"hello"
    assets = _fake_assets(payload)
    old = tmp_path / "old_worktree"
    old.mkdir()
    (old / "f.csv").write_bytes(payload)
    monkeypatch.setattr(
        zenodo.urllib.request, "urlretrieve", _refuse_to_download
    )

    # ACT
    zenodo.fetch_assets(assets, old)  # warm destination: adopt, do not fetch
    zenodo.fetch_assets(assets, tmp_path / "new_worktree")

    # ASSERT
    entry = _entry_of(cache_root, assets)
    assert entry.read_bytes() == payload
    assert (tmp_path / "new_worktree" / "f.csv").read_bytes() == payload
    assert entry.stat().st_ino == (old / "f.csv").stat().st_ino


def test_a_destination_with_the_wrong_md5_is_not_adopted(
    tmp_path, cache_root, monkeypatch, caplog
):
    """
    Given a destination file of the right size but the wrong content,
    When it is considered for adoption,
    Then it is left alone (that is the pre-cache behaviour) but is NOT
    published to the cache, so one bad local file cannot poison the
    machine.
    """
    # ARRANGE
    payload = b"hello"
    assets = _fake_assets(payload)
    old = tmp_path / "old"
    old.mkdir()
    (old / "f.csv").write_bytes(b"world")
    monkeypatch.setattr(
        zenodo.urllib.request, "urlretrieve", _refuse_to_download
    )

    # ACT
    with caplog.at_level("WARNING"):
        zenodo.fetch_assets(assets, old)

    # ASSERT
    assert not _entry_of(cache_root, assets).exists()
    assert any("shared cache" in r.getMessage() for r in caplog.records)


def test_an_unwritable_cache_degrades_to_a_plain_download(
    tmp_path, monkeypatch, caplog
):
    """
    Given a cache directory that cannot be created (read-only parent),
    When fetch_assets runs,
    Then it warns once and downloads straight to the destination, exactly
    as it did before the cache existed. A cache is an optimization; it must
    never fail a fit.
    """
    # ARRANGE
    if os.geteuid() == 0:
        pytest.skip("root ignores directory permissions")
    locked = tmp_path / "locked"
    locked.mkdir(mode=0o500)
    monkeypatch.setenv("EXOZIPPY_CACHE_DIR", str(locked / "exozippy"))

    payload = b"hello"
    assets = _fake_assets(payload)
    fetched = []

    def observant(url, dest):
        assert Path(dest).name == "f.csv.part", "expected the legacy staging"
        Path(dest).write_bytes(payload)

    monkeypatch.setattr(zenodo.urllib.request, "urlretrieve", observant)

    # ACT
    with caplog.at_level("WARNING"):
        zenodo.fetch_assets(assets, tmp_path / "a", on_fetch=fetched.append)

    # ASSERT
    assert (tmp_path / "a" / "f.csv").read_bytes() == payload
    assert fetched == ["f.csv"]
    assert any(
        "cache unavailable" in r.getMessage() for r in caplog.records
    ), "an unusable cache must say so once"


@requires_fork
def test_concurrent_fetches_download_once_and_never_tear_the_entry(
    tmp_path, cache_root, monkeypatch
):
    """
    Given four processes (parallel agents, or four worktrees at once) racing
    for the same asset,
    When they all call fetch_assets,
    Then the flock around the cache entry means exactly one download, every
    destination is intact, and no .part or .link debris is left behind.
    """
    # ARRANGE
    payload = b"payload-bytes" * 64
    assets = _fake_assets(payload)
    counter = tmp_path / "downloads.log"
    counter.write_bytes(b"")

    def slow_download(url, dest):
        with open(counter, "ab") as f:  # O_APPEND: atomic for one small write
            f.write(b"x")
        time.sleep(0.5)  # long enough for the others to pile up on the lock
        Path(dest).write_bytes(payload)

    monkeypatch.setattr(zenodo.urllib.request, "urlretrieve", slow_download)

    dests = [tmp_path / f"worktree{i}" for i in range(4)]

    def child(dest_dir):
        zenodo.fetch_assets(assets, dest_dir)

    ctx = multiprocessing.get_context("fork")

    # ACT
    procs = [ctx.Process(target=child, args=(d,)) for d in dests]
    for p in procs:
        p.start()
    for p in procs:
        p.join(120)

    # ASSERT
    assert [p.exitcode for p in procs] == [0, 0, 0, 0]
    for d in dests:
        assert (d / "f.csv").read_bytes() == payload
    entry = _entry_of(cache_root, assets)
    assert entry.read_bytes() == payload
    assert counter.stat().st_size == 1, (
        f"{counter.stat().st_size} processes downloaded the same asset; "
        "the cache lock did not serialize them"
    )
    assert not list(entry.parent.glob("*.part"))


# --- the two wrappers -----------------------------------------------------


def test_make_bc_warns_about_downsampling_only_when_it_really_fetches(
    tmp_path, monkeypatch, caplog
):
    """
    Given the NextGen spectra are missing and then cached,
    When ensure_model_data runs twice,
    Then the downsampling warning is emitted on the fetch and not on the
    cache hit, and the file lands under {model}/BCs/.
    """
    # ARRANGE
    from exozippy.components.sed import make_bc

    payload = b"hello"
    monkeypatch.setattr(make_bc, "_MODEL_DATA", {"M": _fake_assets(payload)})
    monkeypatch.setattr(make_bc, "_warned_models", set())
    monkeypatch.setattr(
        zenodo.urllib.request,
        "urlretrieve",
        lambda url, dest: Path(dest).write_bytes(payload),
    )

    # ACT
    with caplog.at_level("WARNING"):
        make_bc.ensure_model_data("M", tmp_path)
    first = [r for r in caplog.records if "DOWNSAMPLED" in r.getMessage()]
    caplog.clear()
    with caplog.at_level("WARNING"):
        make_bc.ensure_model_data("M", tmp_path)
    second = [r for r in caplog.records if "DOWNSAMPLED" in r.getMessage()]

    # ASSERT
    assert len(first) == 1
    assert second == []
    assert (tmp_path / "M" / "BCs" / "f.csv").read_bytes() == payload


def test_mist_grid_fetch_does_not_emit_the_spectra_warning(
    tmp_path, monkeypatch, caplog
):
    """
    Given the MIST EEP grid is missing,
    When ensure_eep_grid runs,
    Then it downloads through the same core but says nothing about
    downsampled spectra -- that warning is about the NextGen spectra
    specifically, not about Zenodo fetches in general.
    """
    # ARRANGE
    from exozippy.models.MIST import eep_grid

    payload = b"parquet-bytes"
    assets = _fake_assets(payload, name="afe_p0_vvcrit0.0.grid.parquet")
    monkeypatch.setattr(eep_grid, "_EEP_GRID_ASSETS", assets)
    monkeypatch.setattr(eep_grid, "EEP_GRID_DIR", tmp_path)
    monkeypatch.setattr(
        zenodo.urllib.request,
        "urlretrieve",
        lambda url, dest: Path(dest).write_bytes(payload),
    )

    # ACT
    with caplog.at_level("WARNING"):
        path = eep_grid.ensure_eep_grid()

    # ASSERT
    assert path == tmp_path / "afe_p0_vvcrit0.0.grid.parquet"
    assert path.read_bytes() == payload
    assert not [r for r in caplog.records if "DOWNSAMPLED" in r.getMessage()]


def test_the_mist_grid_shares_the_machine_cache_too(
    tmp_path, cache_root, monkeypatch
):
    """
    Given the EEP grid has been fetched once on this machine,
    When a second checkout's EEP_GRID_DIR asks for it,
    Then it comes from the shared cache with no download -- the second
    caller of fetch_assets gets the cache for free, with no code of its own.
    """
    # ARRANGE
    from exozippy.models.MIST import eep_grid

    payload = b"parquet-bytes"
    name = "afe_p0_vvcrit0.0.grid.parquet"
    assets = _fake_assets(payload, name=name)
    monkeypatch.setattr(eep_grid, "_EEP_GRID_ASSETS", assets)
    monkeypatch.setattr(eep_grid, "EEP_GRID_DIR", tmp_path / "checkout_a")
    monkeypatch.setattr(
        zenodo.urllib.request,
        "urlretrieve",
        lambda url, dest: Path(dest).write_bytes(payload),
    )
    eep_grid.ensure_eep_grid()

    # ACT
    monkeypatch.setattr(eep_grid, "EEP_GRID_DIR", tmp_path / "checkout_b")
    monkeypatch.setattr(
        zenodo.urllib.request, "urlretrieve", _refuse_to_download
    )
    path = eep_grid.ensure_eep_grid()

    # ASSERT
    assert path == tmp_path / "checkout_b" / name
    assert path.read_bytes() == payload
    assert _entry_of(cache_root, assets, name=name).exists()


def test_mist_grid_is_not_refetched_when_already_cached(tmp_path, monkeypatch):
    """
    Given the EEP grid parquet is already cached at the pinned size,
    When ensure_eep_grid runs,
    Then it returns the cached path without downloading.
    """
    # ARRANGE
    from exozippy.models.MIST import eep_grid

    payload = b"parquet-bytes"
    name = "afe_p0_vvcrit0.0.grid.parquet"
    (tmp_path / name).write_bytes(payload)
    monkeypatch.setattr(
        eep_grid, "_EEP_GRID_ASSETS", _fake_assets(payload, name=name)
    )
    monkeypatch.setattr(eep_grid, "EEP_GRID_DIR", tmp_path)

    calls = []
    monkeypatch.setattr(
        zenodo.urllib.request,
        "urlretrieve",
        lambda url, dest: calls.append(url),
    )

    # ACT
    path = eep_grid.ensure_eep_grid()

    # ASSERT
    assert calls == []
    assert path == tmp_path / name


def test_an_unknown_alpha_vvcrit_combination_raises(monkeypatch):
    """
    Given a (alpha, vvcrit) pair with no published grid,
    When ensure_eep_grid is asked for it,
    Then it raises a KeyError naming what is available, rather than
    attempting a download of a URL that does not exist.
    """
    # ARRANGE
    from exozippy.models.MIST import eep_grid

    # ACT / ASSERT
    with pytest.raises(KeyError, match="afe_p0_vvcrit0.0"):
        eep_grid.ensure_eep_grid(alpha=0.4, vvcrit=0.4)
