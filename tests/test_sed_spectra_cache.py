"""The SED figure's spectra table is read once per file version (6.9.1).

NextGen's spectra table is ~250 MB and every row's ``flux`` cell is a JSON
array that must be parsed into a numpy array. It was re-read and re-parsed
on EVERY Plot construction -- once per posterior-plot pass, and in the
GUI's live mode once per slider move.

The tiny table built here is not about the size; it is about how many
times the file is opened.
"""

import json

import pandas as pd

from exozippy.components.sed import plot as plot_module
from exozippy.components.sed.plot import _read_spectra_csv


def _write_spectra(path, teff):
    pd.DataFrame(
        {
            "filename": ["a"],
            "teff": [teff],
            "logg": [4.5],
            "feh": [0.0],
            "alpha": [0.0],
            "flux": [json.dumps([1.0, 2.0])],
        }
    ).to_csv(path, index=False)


def test_the_same_table_is_read_once(tmp_path, monkeypatch):
    """
    Given a spectra table on disk,
    When it is read twice,
    Then the file is opened once and the same frame comes back.
    """
    # ARRANGE
    path = tmp_path / "M.spectra.csv"
    _write_spectra(path, 3000)
    reads = []
    real_read_csv = pd.read_csv

    def counting(*args, **kwargs):
        reads.append(args[0])
        return real_read_csv(*args, **kwargs)

    monkeypatch.setattr(plot_module.pd, "read_csv", counting)

    # ACT
    first = _read_spectra_csv(path)
    second = _read_spectra_csv(path)

    # ASSERT
    assert len(reads) == 1
    assert first is second


def test_the_flux_column_is_parsed_into_arrays(tmp_path):
    """
    Given a spectra table whose flux cells are JSON arrays,
    When it is read,
    Then the column holds numpy arrays, as before the cache.
    """
    # ARRANGE
    path = tmp_path / "M2.spectra.csv"
    _write_spectra(path, 3100)

    # ACT
    df = _read_spectra_csv(path)

    # ASSERT
    assert list(df["flux"].iloc[0]) == [1.0, 2.0]


def test_a_regenerated_table_is_re_read(tmp_path):
    """
    Given a spectra table regenerated in place (make_bc rewrites these),
    When it is read again,
    Then the new contents come back rather than the cached ones.
    """
    # ARRANGE
    path = tmp_path / "M3.spectra.csv"
    _write_spectra(path, 3000)
    before = _read_spectra_csv(path)

    # ACT
    _write_spectra(path, 9999)
    after = _read_spectra_csv(path)

    # ASSERT
    assert before["teff"].iloc[0] == 3000
    assert after["teff"].iloc[0] == 9999
