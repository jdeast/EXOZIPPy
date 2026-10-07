"""Where Filter reads and writes its cached profiles (reviews 2.9.3, 1.9.6).

Nothing here touches the network: every test either reads a profile that is
already on disk or monkeypatches the download.
"""

import os
import pickle

import pytest

from exozippy.filters import filter as filter_module
from exozippy.filters.filter import (
    DEFAULT_FILTER_DIR,
    Filter,
    _PortableUnpickler,
    _writable_filter_root,
    filter_cache_root,
)

_SHIPPED_ID = "2MASS/2MASS.J"


def _refuse_to_download(self):
    raise AssertionError("network access attempted")


def test_a_shipped_profile_is_read_without_touching_the_package_directory(
    tmp_path, monkeypatch
):
    """
    Given a filter whose profile ships inside the package,
    When Filter is constructed for it,
    Then the profile is read from the package directory and nothing is
      created anywhere.

    Regression (2.9.3): the unconditional os.makedirs ran even on the pure
    READ path, so merely naming a filter created an (often empty) facility
    directory inside the install -- a PermissionError on a read-only
    site-packages, source-tree litter in a dev checkout.
    """
    # ARRANGE
    monkeypatch.setenv("EXOZIPPY_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(Filter, "_download_filter", _refuse_to_download)

    # ACT
    filt = Filter(_SHIPPED_ID)

    # ASSERT
    assert filt.filterDirectory == DEFAULT_FILTER_DIR / "2MASS"
    assert not (tmp_path / "cache").exists()


def test_a_downloaded_profile_is_written_to_the_cache_not_the_package(
    tmp_path, monkeypatch
):
    """
    Given a filter with no shipped profile and a writable cache,
    When Filter is constructed for it,
    Then the pickle lands under the cache root and the package directory
      gains nothing.
    """
    # ARRANGE
    cache = tmp_path / "cache"
    monkeypatch.setenv("EXOZIPPY_CACHE_DIR", str(cache))

    def fake_download(self):
        self._downloaded = True

    def fake_set_attrs(self):
        self.WavelengthEff = 12345.0

    monkeypatch.setattr(Filter, "_download_filter", fake_download)
    monkeypatch.setattr(Filter, "_set_attrs", fake_set_attrs)

    # ACT
    filt = Filter("Fake/Fake.Band")

    # ASSERT
    written = cache / "filters" / "Fake" / "Fake.Band.filter"
    assert written.exists()
    assert filt.filterDirectory == cache / "filters" / "Fake"
    assert not (DEFAULT_FILTER_DIR / "Fake").exists()

    # and it round-trips: a second construction reads it back, no download
    monkeypatch.setattr(Filter, "_download_filter", _refuse_to_download)
    again = Filter("Fake/Fake.Band")
    assert again.WavelengthEff == 12345.0
    assert pickle.loads(written.read_bytes()) is not None


def test_cache_root_follows_the_shared_cache_switch(tmp_path, monkeypatch):
    """
    Given EXOZIPPY_CACHE_DIR set, and then switched off,
    When filter_cache_root is asked,
    Then it tracks the same switch the Zenodo download cache uses.
    """
    # ARRANGE / ACT / ASSERT
    monkeypatch.setenv("EXOZIPPY_CACHE_DIR", str(tmp_path / "c"))
    assert filter_cache_root() == tmp_path / "c" / "filters"

    monkeypatch.setenv("EXOZIPPY_CACHE_DIR", "")
    assert filter_cache_root() is None


def test_with_the_cache_off_writes_fall_back_to_the_package_directory(
    monkeypatch,
):
    """
    Given the machine-level cache switched off (as the test suite runs),
    When a write root is chosen,
    Then it is the package directory -- exactly the pre-2026-08 behaviour.

    "Cache off" means "do not put things in my home", not "write into a
    read-only install"; the package directory is writable in the dev
    checkout this switch is used in.
    """
    # ARRANGE
    monkeypatch.setenv("EXOZIPPY_CACHE_DIR", "")

    # ACT
    root = _writable_filter_root()

    # ASSERT
    assert root == DEFAULT_FILTER_DIR


def test_an_unwritable_cache_and_package_fall_back_to_a_temp_dir(
    tmp_path, monkeypatch
):
    """
    Given neither the cache nor the package directory is writable
      (a read-only site-packages install with no cache),
    When a write root is chosen,
    Then a temporary directory is used and the run continues.

    Regression (2.9.3): this combination raised PermissionError out of
    os.makedirs on first use of any filter without a shipped pickle.
    """
    # ARRANGE
    monkeypatch.setenv("EXOZIPPY_CACHE_DIR", "")
    real_access = os.access
    monkeypatch.setattr(
        filter_module.os,
        "access",
        lambda path, mode: (
            False
            if str(path) == str(DEFAULT_FILTER_DIR)
            else real_access(path, mode)
        ),
    )

    # ACT
    root = _writable_filter_root()

    # ASSERT
    assert root != DEFAULT_FILTER_DIR
    assert root.name == "exozippy-filters"


def test_str_does_not_raise():
    """
    Given a Filter whose filterDirectory is a pathlib.Path (always),
    When it is printed,
    Then the directory appears in the message.

    Regression (1.9.6): __str__ concatenated a str with a Path, so the one
    debugging affordance the class offers raised TypeError when used.
    """
    # ARRANGE
    filt = object.__new__(Filter)
    filt.filterName = "2MASS.J"
    filt.filterDirectory = DEFAULT_FILTER_DIR / "2MASS"

    # ACT
    text = str(filt)

    # ASSERT
    assert "2MASS.J" in text
    assert str(DEFAULT_FILTER_DIR / "2MASS") in text


_GAIA_DR3_IDS = ("GAIA/GAIA3.G", "GAIA/GAIA3.Gbp", "GAIA/GAIA3.Grp")


@pytest.mark.parametrize("filter_id", _GAIA_DR3_IDS)
def test_the_gaia_dr3_profiles_ship_and_load_without_the_network(
    filter_id, tmp_path, monkeypatch
):
    """
    Given the Gaia DR3 curves mkticsed writes its Gaia rows against,
    When Filter is constructed for one with downloads refused,
    Then it is read from the package directory, nothing is created in the
      cache or the package, and it carries a transmission curve.
    """
    # ARRANGE
    monkeypatch.setenv("EXOZIPPY_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(Filter, "_download_filter", _refuse_to_download)
    before = sorted((DEFAULT_FILTER_DIR / "GAIA").iterdir())

    # ACT
    filt = Filter(filter_id)

    # ASSERT
    assert filt.filterDirectory == DEFAULT_FILTER_DIR / "GAIA"
    assert filt.filterID == filter_id
    wave, trans = filt.ProcessedFilterCurve
    assert len(wave) == len(trans) > 0
    assert not (tmp_path / "cache").exists()
    assert sorted((DEFAULT_FILTER_DIR / "GAIA").iterdir()) == before


@pytest.mark.parametrize("filter_id", _GAIA_DR3_IDS)
def test_the_shipped_gaia_dr3_pickles_carry_no_machine_baggage(filter_id):
    """
    Given a shipped Gaia DR3 .filter pickle,
    When it is unpickled raw,
    Then it holds no requests.Session and no filterDirectory of the machine
      that wrote it, so the file is the same wherever it is installed.
    """
    # ARRANGE
    path = DEFAULT_FILTER_DIR / "GAIA" / (filter_id.split("/")[1] + ".filter")

    # ACT
    with open(path, "rb") as fh:
        state = _PortableUnpickler(fh).load()

    # ASSERT
    assert "_session" not in state
    assert state["filterDirectory"] is None
    assert b"requests" not in path.read_bytes()


def test_a_written_filter_file_drops_the_session_and_directory():
    """
    Given a loaded Filter,
    When the state a .filter file stores is taken,
    Then the session and the writer's directory are gone from it, while
      __getstate__ (which copy/deepcopy also use) still keeps both.
    """
    # ARRANGE
    filt = Filter(_SHIPPED_ID)

    # ACT
    file_state = filt._file_state()
    full_state = filt.__getstate__()

    # ASSERT
    assert "_session" not in file_state
    assert file_state["filterDirectory"] is None
    assert "_session" in full_state
    assert full_state["filterDirectory"] == str(DEFAULT_FILTER_DIR / "2MASS")
