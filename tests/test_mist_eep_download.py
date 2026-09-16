"""The MIST EEP tarball downloader (review 1.9.7).

An offline authoring script: it fetches ~186 GB of raw evolutionary tracks,
one tarball per grid point, and is meant to be resumable. It was not --
every tarball landed in ./temp_files instead of the grid point's own
directory, so the caller's "already have it?" check could never become
true and a re-run re-downloaded everything.

Nothing here touches the network: requests.get is monkeypatched with a
tarball built in the test.
"""

import io
import tarfile

import pytest

from exozippy.models.MIST import download_MIST_EEPs as dl


def _tarball(top_level=None, names=("0100M.track.eep", "0200M.track.eep")):
    """A .tar.gz in memory, optionally wrapped in one top-level directory."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name in names:
            payload = b"# eep track\n"
            info = tarfile.TarInfo(
                f"{top_level}/{name}" if top_level else name
            )
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    return buf.getvalue()


class _FakeResponse:
    def __init__(self, body, chunks=1):
        self._body = body
        self._chunks = chunks
        self.raised = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size=1):
        step = max(1, len(self._body) // self._chunks)
        for start in range(0, len(self._body), step):
            yield self._body[start : start + step]


def _patch_get(monkeypatch, body, timeout_seen=None, fail_after=None):
    def fake_get(url, stream=False, timeout=None):
        if timeout_seen is not None:
            timeout_seen.append(timeout)
        if fail_after is not None:
            raise ConnectionError("peer went away")
        return _FakeResponse(body)

    monkeypatch.setattr(dl.requests, "get", fake_get)


def test_the_tarball_is_extracted_into_the_requested_folder(
    tmp_path, monkeypatch
):
    """
    Given a grid point's destination folder,
    When download_tarfiles is called with it,
    Then the tracks land there -- not in a ./temp_files beside the cwd.

    Regression: __main__ called download_tarfiles(url) with no dest_folder,
    so every one of the ~160 tarballs extracted into the same ./temp_files,
    the caller's folder_path.is_dir() skip-check never became true, and a
    re-run of the 186 GB workflow re-downloaded all of it.
    """
    # ARRANGE
    dest = tmp_path / "feh_p000_afe_p0_vvcrit0.0"
    _patch_get(monkeypatch, _tarball())

    # ACT
    dl.download_tarfiles("https://example.invalid/x.txz", dest_folder=dest)

    # ASSERT
    assert (dest / "0100M.track.eep").exists()
    assert not (tmp_path / "temp_files").exists()
    assert dest.is_dir(), "the caller's skip-check keys on exactly this"


def test_an_archive_wrapper_directory_is_flattened_away(tmp_path, monkeypatch):
    """
    Given a tarball with its own top-level directory (how MIST ships them),
    When it is extracted,
    Then the tracks sit directly in the destination folder, which is the
      layout every reader of these directories assumes.
    """
    # ARRANGE
    dest = tmp_path / "feh_p000_afe_p0_vvcrit0.0"
    _patch_get(monkeypatch, _tarball(top_level="MIST_v2.5_feh_p000_EEPS"))

    # ACT
    dl.download_tarfiles("https://example.invalid/x.txz", dest_folder=dest)

    # ASSERT
    assert (dest / "0100M.track.eep").exists()


def test_an_interrupted_download_leaves_no_destination_behind(
    tmp_path, monkeypatch
):
    """
    Given a transfer that fails part way,
    When download_tarfiles raises,
    Then the destination folder does not exist and no partial archive is
      left where a later run could extract it.

    The skip-check is "does the folder exist", so a half-populated folder
    would be silently accepted as a finished grid point forever after.
    """
    # ARRANGE
    dest = tmp_path / "feh_p000_afe_p0_vvcrit0.0"
    _patch_get(monkeypatch, _tarball(), fail_after=0)

    # ACT / ASSERT
    with pytest.raises(ConnectionError):
        dl.download_tarfiles("https://example.invalid/x.txz", dest_folder=dest)

    assert not dest.exists()
    assert list(tmp_path.iterdir()) == []


def test_an_already_downloaded_folder_is_not_refetched(tmp_path, monkeypatch):
    """
    Given a destination folder that already exists,
    When download_tarfiles is called for it,
    Then nothing is fetched.
    """
    # ARRANGE
    dest = tmp_path / "feh_p000_afe_p0_vvcrit0.0"
    dest.mkdir()

    def refuse(*a, **k):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(dl.requests, "get", refuse)

    # ACT
    dl.download_tarfiles("https://example.invalid/x.txz", dest_folder=dest)

    # ASSERT -- no exception; the folder is untouched
    assert list(dest.iterdir()) == []


def test_the_fetch_carries_a_timeout(tmp_path, monkeypatch):
    """
    Given the downloader,
    When it issues its request,
    Then a finite timeout is passed, so a stalled peer cannot hang the
      whole multi-hour workflow.
    """
    # ARRANGE
    seen = []
    _patch_get(monkeypatch, _tarball(), timeout_seen=seen)

    # ACT
    dl.download_tarfiles(
        "https://example.invalid/x.txz", dest_folder=tmp_path / "d"
    )

    # ASSERT
    assert seen == [dl._HTTP_TIMEOUT]
    assert 0 < dl._HTTP_TIMEOUT < float("inf")
