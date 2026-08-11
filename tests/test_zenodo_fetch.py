"""
Tests for the shared Zenodo fetch-verify-cache core
(utilities/zenodo.py) and the two wrappers that own asset tables
(components/sed/make_bc.ensure_model_data for now; the MIST EEP
grid is the second).

Nothing here touches the network: urlretrieve is always monkeypatched.
"""

import hashlib
import urllib.error
from pathlib import Path

import pytest

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


# --- the NextGen wrapper -----------------------------------------------------


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
    monkeypatch.setattr(
        make_bc, "_MODEL_DATA", {"M": _fake_assets(payload)}
    )
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
