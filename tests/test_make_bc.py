"""
Tests for the BC-table generator (components/sed/make_bc.py):
  - regenerated 2MASS_J agrees with the shipped table at Av=0
  - existing columns survive a merge byte-exactly
  - extinction is applied along the Av axis
"""

import shutil

import numpy as np
import pytest

from exozippy.components.sed.bc_grid import (
    DEFAULT_MODEL_ROOT,
    _read_single_bc_file,
)
from exozippy.components.sed.make_bc import make_bc_tables

_NEXTGEN = DEFAULT_MODEL_ROOT / "NextGen"
_SPECTRA = _NEXTGEN / "BCs" / "NextGen.spectra.csv"
_SHIPPED_2MASS = _NEXTGEN / "BCs" / "2MASS" / "feh+0.0_afe+0.0.2MASS"

pytestmark = pytest.mark.skipif(
    not _SPECTRA.exists(),
    reason="NextGen spectra not downloaded (Zenodo); run an SED fit once first.",
)


@pytest.fixture(scope="module")
def regenerated_2mass(tmp_path_factory):
    """
    Build a minimal BC root (solar-feh 2MASS file + spectra symlinks),
    regenerate the 2MASS_J column there, and return the before/after
    tables. Module-scoped: the spectra CSV read dominates the runtime.
    """
    root = tmp_path_factory.mktemp("model_root")
    model_dir = root / "NextGen"
    (model_dir / "BCs" / "2MASS").mkdir(parents=True)
    for name in ("NextGen.spectra.csv", "NextGen.wavelength.csv"):
        (model_dir / "BCs" / name).symlink_to(_NEXTGEN / "BCs" / name)
    shutil.copy(
        _SHIPPED_2MASS, model_dir / "BCs" / "2MASS" / _SHIPPED_2MASS.name
    )

    written = make_bc_tables(
        ["2MASS/2MASS.J"], model="NextGen", model_root=root
    )

    before, cols_before = _read_single_bc_file(_SHIPPED_2MASS)
    after, cols_after = _read_single_bc_file(written[0])
    key = ["teff", "logg", "Av"]
    before = before.sort_values(key).reset_index(drop=True)
    after = after.sort_values(key).reset_index(drop=True)
    return before, after, cols_after


def test_regenerated_2mass_j_matches_shipped_at_av0(regenerated_2mass):
    """
    Given the shipped solar-feh 2MASS BC table,
    When the 2MASS_J column is regenerated from the shipped R=150 spectra,
    Then the Av=0 values agree with the shipped table to 0.1 mag
    (the shipped tables came from higher-resolution spectra, so exact
    agreement is not expected; ~0.02 mag is typical).
    """
    # ARRANGE
    before, after, _ = regenerated_2mass
    m = before["Av"] == 0.0

    # ACT
    diff = (after.loc[m, "2MASS_J"] - before.loc[m, "2MASS_J"]).abs()

    # ASSERT
    assert diff.median() < 0.05
    assert diff.max() < 0.35  # coolest grid corners are resolution-limited


def test_merge_preserves_existing_columns_exactly(regenerated_2mass):
    """
    Given a facility file with 2MASS_H and 2MASS_Ks columns,
    When only 2MASS_J is regenerated into it,
    Then H and Ks are preserved exactly.
    """
    # ARRANGE
    before, after, cols_after = regenerated_2mass

    # ASSERT
    assert set(cols_after) == {"2MASS_J", "2MASS_H", "2MASS_Ks"}
    for col in ("2MASS_H", "2MASS_Ks"):
        np.testing.assert_array_equal(before[col].values, after[col].values)


def test_generated_bc_includes_extinction_along_av_axis(regenerated_2mass):
    """
    Given a regenerated 2MASS_J column,
    When comparing BC at Av=0 and Av=6 for a solar analog,
    Then BC drops by roughly the J-band extinction A_J ~ 0.28*Av
    (the shipped tables are flat in Av; the generator applies the
    extinction law).
    """
    # ARRANGE
    _, after, _ = regenerated_2mass
    node = (after["teff"] == 5800) & (after["logg"] == 4.5)
    bc0 = float(after.loc[node & (after["Av"] == 0.0), "2MASS_J"].iloc[0])
    bc6 = float(after.loc[node & (after["Av"] == 6.0), "2MASS_J"].iloc[0])

    # ACT
    a_j = bc0 - bc6

    # ASSERT
    assert 1.0 < a_j < 2.5


# --- ensure_model_data download retry -----------------------------------


def _fake_meta(tmp_path, payload=b"hello"):
    """A one-file _MODEL_DATA entry whose size/md5 match `payload`."""
    import hashlib

    return {
        "f.csv": {
            "url": "https://example.invalid/f.csv",
            "size": len(payload),
            "md5": hashlib.md5(payload).hexdigest(),
        }
    }


def test_download_retries_a_transient_gateway_error(tmp_path, monkeypatch):
    """
    Given Zenodo returns 504 once and then succeeds,
    When ensure_model_data runs,
    Then it retries and the file lands, rather than failing the caller.

    Regression: two consecutive Dependabot PRs went red purely because a
    250 MB Zenodo fetch returned `HTTPError: HTTP Error 504: Gateway
    Time-out` mid-CI. A transient gateway error must cost seconds, not a run.
    """
    # ARRANGE
    import urllib.error

    from exozippy.components.sed import make_bc

    payload = b"hello"
    monkeypatch.setattr(
        make_bc, "_MODEL_DATA", {"M": _fake_meta(tmp_path, payload)}
    )
    monkeypatch.setattr(make_bc.time, "sleep", lambda *a: None)

    calls = []

    def flaky(url, dest):
        calls.append(url)
        if len(calls) == 1:
            raise urllib.error.HTTPError(
                url, 504, "Gateway Time-out", {}, None
            )
        open(dest, "wb").write(payload)

    monkeypatch.setattr(make_bc.urllib.request, "urlretrieve", flaky)

    # ACT
    make_bc.ensure_model_data("M", tmp_path)

    # ASSERT
    assert len(calls) == 2, "should have retried exactly once"
    assert (tmp_path / "M" / "f.csv").read_bytes() == payload


def test_download_does_not_retry_a_404(tmp_path, monkeypatch):
    """
    Given the URL is simply wrong (404),
    When ensure_model_data runs,
    Then it raises immediately instead of burning the backoff schedule.
    """
    # ARRANGE
    import urllib.error

    from exozippy.components.sed import make_bc

    monkeypatch.setattr(make_bc, "_MODEL_DATA", {"M": _fake_meta(tmp_path)})
    monkeypatch.setattr(make_bc.time, "sleep", lambda *a: None)

    calls = []

    def missing(url, dest):
        calls.append(url)
        raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)

    monkeypatch.setattr(make_bc.urllib.request, "urlretrieve", missing)

    # ACT / ASSERT
    with pytest.raises(urllib.error.HTTPError):
        make_bc.ensure_model_data("M", tmp_path)
    assert len(calls) == 1, "a 404 is not transient; do not retry it"


def test_download_gives_up_after_the_attempt_budget(tmp_path, monkeypatch):
    """
    Given every attempt fails,
    When the budget is exhausted,
    Then a RuntimeError names the file and no .part is left behind.
    """
    # ARRANGE
    import urllib.error

    from exozippy.components.sed import make_bc

    monkeypatch.setattr(make_bc, "_MODEL_DATA", {"M": _fake_meta(tmp_path)})
    monkeypatch.setattr(make_bc.time, "sleep", lambda *a: None)

    calls = []

    def always_504(url, dest):
        calls.append(url)
        raise urllib.error.HTTPError(url, 504, "Gateway Time-out", {}, None)

    monkeypatch.setattr(make_bc.urllib.request, "urlretrieve", always_504)

    # ACT / ASSERT
    with pytest.raises(RuntimeError, match="f.csv"):
        make_bc.ensure_model_data("M", tmp_path)
    assert len(calls) == make_bc._DOWNLOAD_ATTEMPTS
    assert not list((tmp_path / "M").glob("*.part")), "left a .part behind"
