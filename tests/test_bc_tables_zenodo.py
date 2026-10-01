"""The NextGen BC tables are fetched from Zenodo, pinned, and never re-shipped.

models/NextGen/bc_tables.py owns the fetch;
components/sed/bc_grid.ensure_bc_tables is the hook every table reader
calls.  Nothing here touches the network: zenodo._urlretrieve is replaced
by a fake, and the manifest by a one-file fake -- except for the one
real download of the smallest table from the pinned record, which skips
only if Zenodo is unreachable.  The pins themselves live in
utilities/zenodo_assets.py; tests/test_zenodo_assets.py checks them
against the live record's file list.
"""

import hashlib
import urllib.error

import pytest

from exozippy.components.sed import bc_grid
from exozippy.models.NextGen import bc_tables
from exozippy.utilities import zenodo

_PAYLOAD = b"not really a parquet file, but bytes with a pinned md5\n" * 7
_NAME = "Fake.bc.parquet"


@pytest.fixture
def fake_manifest(monkeypatch, tmp_path):
    """One fake published table, a pinned record id, a tmp destination."""
    files = {
        _NAME: {
            "size": len(_PAYLOAD),
            "md5": hashlib.md5(_PAYLOAD).hexdigest(),
        }
    }
    monkeypatch.setattr(bc_tables, "_BC_TABLE_FILES", files)
    monkeypatch.setattr(bc_tables, "ZENODO_RECORD", 123)
    monkeypatch.setattr(bc_tables, "BC_TABLE_DIR", tmp_path / "BCs")
    monkeypatch.setattr(bc_tables, "_verified", set())
    monkeypatch.setattr(zenodo.time, "sleep", lambda *a: None)
    return tmp_path / "BCs"


def _serving(calls, payload=_PAYLOAD):
    def fake(url, dest):
        calls.append(url)
        dest.write_bytes(payload)

    return fake


def test_the_bc_table_record_is_pinned():
    """
    Given the pinned manifest,
    When its Zenodo record ids are checked,
    Then both the version record and its concept record are filled in, and
    a table's URL is that record's file URL.
    """
    assert isinstance(bc_tables.ZENODO_RECORD, int)
    assert isinstance(bc_tables.ZENODO_CONCEPT_RECORD, int)
    url = bc_tables._assets(["TESS.bc.parquet"])["TESS.bc.parquet"]["url"]
    assert url == (
        f"https://zenodo.org/records/{bc_tables.ZENODO_RECORD}/files/"
        f"TESS.bc.parquet"
    )


def _zenodo_record_or_skip():
    """The pinned record's API JSON, or skip if Zenodo cannot be reached.

    Only a TRANSPORT failure skips (no network, DNS, a 5xx): an HTTP 4xx
    means the pinned record id is wrong, which is exactly what this is for,
    so it fails.
    """
    import json
    import urllib.request

    url = f"https://zenodo.org/api/records/{bc_tables.ZENODO_RECORD}"
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as e:
        if e.code < 500:
            raise
        pytest.skip(f"Zenodo unavailable ({e}); cannot check the record")
    except (urllib.error.URLError, TimeoutError) as e:
        pytest.skip(f"no network ({e}); cannot reach the pinned record")


@pytest.mark.network
def test_a_table_really_downloads_from_the_pinned_record(
    monkeypatch, tmp_path
):
    """
    Given the real pinned record and an empty destination and machine cache,
    When the smallest published table is ensured,
    Then it is downloaded through utilities/zenodo, lands with the pinned
    md5, and is published into the machine cache -- the path every fresh
    install takes.  (~1.4 MB; skipped only if Zenodo cannot be reached.)
    """
    _zenodo_record_or_skip()
    name = min(
        bc_tables._BC_TABLE_FILES,
        key=lambda n: bc_tables._BC_TABLE_FILES[n]["size"],
    )
    cache = tmp_path / "cache"
    monkeypatch.setenv("EXOZIPPY_CACHE_DIR", str(cache))
    monkeypatch.setattr(bc_tables, "_verified", set())

    (path,) = bc_tables.ensure_tables([name], dest_dir=tmp_path / "BCs")

    md5 = bc_tables._BC_TABLE_FILES[name]["md5"]
    assert hashlib.md5(path.read_bytes()).hexdigest() == md5
    assert (cache / "downloads" / f"{md5}-{name}").is_file()


def test_every_pin_is_a_bc_table_with_a_size_and_an_md5():
    """
    Given the pinned manifest,
    When each entry is inspected,
    Then it names a .bc.parquet table and pins a positive size and a
    32-hex-digit md5 -- what fetch_assets checks a download against.
    """
    assert bc_tables.published_tables(), "no tables pinned"
    for name, meta in bc_tables._BC_TABLE_FILES.items():
        assert name.endswith(".bc.parquet"), name
        assert isinstance(meta["size"], int) and meta["size"] > 0, name
        assert len(meta["md5"]) == 32, name
        int(meta["md5"], 16)


def test_an_absent_table_is_fetched_from_the_pinned_record(
    fake_manifest, monkeypatch
):
    """
    Given a published table that is not on disk,
    When ensure_tables asks for it,
    Then it is downloaded from the pinned record's file URL and verified.
    """
    calls = []
    monkeypatch.setattr(zenodo, "_urlretrieve", _serving(calls))

    paths = bc_tables.ensure_tables([_NAME])

    assert paths == [fake_manifest / _NAME]
    assert paths[0].read_bytes() == _PAYLOAD
    assert calls == [f"https://zenodo.org/records/123/files/{_NAME}"]


def test_an_unreachable_record_raises_naming_the_file_and_the_fix(
    fake_manifest, monkeypatch
):
    """
    Given no network,
    When an absent table is asked for,
    Then RuntimeError (not FileNotFoundError, which the SED reads as "no
    tables" and carries on) names the file, the record and the pre-fetch
    command -- and nothing is left at the destination.
    """

    def offline(url, dest):
        raise urllib.error.URLError("Temporary failure in name resolution")

    monkeypatch.setattr(zenodo, "_urlretrieve", offline)

    with pytest.raises(RuntimeError) as err:
        bc_tables.ensure_tables([_NAME])

    assert not isinstance(err.value, FileNotFoundError)
    msg = str(err.value)
    assert _NAME in msg
    assert "https://zenodo.org/records/123" in msg
    assert bc_tables.FETCH_COMMAND in msg
    assert not (fake_manifest / _NAME).exists()


def test_a_locally_changed_table_raises_and_is_not_overwritten(
    fake_manifest, monkeypatch
):
    """
    Given a table on disk that differs from the pin (a generator merged into
    it),
    When a fit asks for it,
    Then it raises naming the file, and the file is untouched -- not
    re-downloaded over (fetch_assets' rule for a truncated download), and
    not silently used.  The generators' allow_local_changes accepts it.
    """
    calls = []
    monkeypatch.setattr(zenodo, "_urlretrieve", _serving(calls))
    fake_manifest.mkdir(parents=True)
    local = fake_manifest / _NAME
    edited = _PAYLOAD.replace(b"bytes", b"BYTES")  # same size, other md5
    local.write_bytes(edited)

    with pytest.raises(RuntimeError, match="not the published"):
        bc_tables.ensure_tables([_NAME])
    assert local.read_bytes() == edited

    assert bc_tables.ensure_tables([_NAME], allow_local_changes=True) == [
        local
    ]
    assert local.read_bytes() == edited
    assert calls == []


def test_a_cached_table_is_linked_without_a_download(
    fake_manifest, monkeypatch, tmp_path
):
    """
    Given the machine cache holding the exact file (<md5>-<name>, as
    utilities/zenodo keys it) and an empty destination,
    When the table is asked for,
    Then it is linked into place with no download -- how every checkout on
    a machine after the first gets its tables.
    """
    calls = []
    monkeypatch.setattr(zenodo, "_urlretrieve", _serving(calls))
    cache = tmp_path / "cache"
    monkeypatch.setenv("EXOZIPPY_CACHE_DIR", str(cache))
    (cache / "downloads").mkdir(parents=True)
    md5 = hashlib.md5(_PAYLOAD).hexdigest()
    (cache / "downloads" / f"{md5}-{_NAME}").write_bytes(_PAYLOAD)

    (path,) = bc_tables.ensure_tables([_NAME])

    assert path.read_bytes() == _PAYLOAD
    assert calls == []


@pytest.fixture
def ensure_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(
        bc_tables,
        "ensure_tables",
        lambda names, allow_local_changes=False: calls.append(list(names)),
    )
    return calls


def test_find_bc_table_fetches_only_the_facility_it_reads(ensure_calls):
    """
    Given the default model root,
    When find_bc_table locates the 2MASS table, and then a facility with no
    published table,
    Then only 2MASS.bc.parquet is ensured, and the unpublished facility
    still raises the reader's own "not calculated" error.
    """
    root = bc_grid.DEFAULT_MODEL_ROOT
    bc_grid.find_bc_table(root, "NextGen", "2MASS")
    with pytest.raises(NotImplementedError):
        bc_grid.find_bc_table(root, "NextGen", "NoSuchFacility")

    assert ensure_calls == [["2MASS.bc.parquet"]]


def test_an_explicit_model_root_is_never_fetched_into(ensure_calls, tmp_path):
    """
    Given a `model_root:` that is not the package's,
    When its tables are looked up,
    Then nothing is fetched: an explicit root is used as given (the rule
    mist_grid applies to the MIST grids).
    """
    (tmp_path / "NextGen" / "BCs").mkdir(parents=True)
    with pytest.raises(NotImplementedError):
        bc_grid.find_bc_table(tmp_path, "NextGen", "2MASS")
    assert ensure_calls == []


def test_peek_grid_axes_with_no_filters_fetches_every_table(
    ensure_calls, monkeypatch
):
    """
    Given the default root,
    When peek_grid_axes is asked about every filter (it globs the tables),
    Then every published table is ensured first.
    """
    monkeypatch.setattr(bc_grid, "read_bc_table", _stop)
    with pytest.raises(_Stop):
        bc_grid.peek_grid_axes("NextGen", bc_grid.DEFAULT_MODEL_ROOT)
    assert ensure_calls == [bc_tables.published_tables()]


class _Stop(Exception):
    pass


def _stop(*args, **kwargs):
    raise _Stop
