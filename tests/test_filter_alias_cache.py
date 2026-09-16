"""The filter alias table is read once per file version (review 6.9.2).

``_load_alias_table`` has six call sites, several inside per-filter loops,
and re-read + re-stripped the same small table on every one. The cache has
to key on the file's CONTENT VERSION, not just its path, or a rewritten
table (a test's, or a user's) is silently served from the cache.
"""

import pandas as pd

from exozippy.components.sed import bc_grid
from exozippy.components.sed.bc_grid import _load_alias_table

_HEADER = "Keivan\tMIST\tClaret\tSVO\tVOID\n"


def _write_table(root, svo):
    root.mkdir(parents=True, exist_ok=True)
    (root / "filternames.txt").write_text(
        _HEADER + f"J\t2MASS_J\tJ\t{svo}\t2MASS.J\n"
    )


def test_the_same_table_is_read_once(tmp_path, monkeypatch):
    """
    Given an alias table on disk,
    When it is loaded twice,
    Then the second call does not re-read the file.
    """
    # ARRANGE
    _write_table(tmp_path, "2MASS/2MASS.J")
    reads = []
    real_read_csv = pd.read_csv

    def counting(*args, **kwargs):
        reads.append(args[0])
        return real_read_csv(*args, **kwargs)

    monkeypatch.setattr(bc_grid.pd, "read_csv", counting)

    # ACT
    first = _load_alias_table(tmp_path)
    second = _load_alias_table(tmp_path)

    # ASSERT
    assert len(reads) == 1
    assert first is second


def test_a_rewritten_table_is_re_read(tmp_path):
    """
    Given an alias table that is rewritten in place,
    When it is loaded again,
    Then the new contents come back.

    Keying the cache on the path alone is how a cache turns a correct test
    (or a user editing filternames.txt between runs) into a stale one.
    """
    # ARRANGE
    _write_table(tmp_path, "2MASS/2MASS.J")
    before = _load_alias_table(tmp_path)

    # ACT -- same path, different content (and a different size, which the
    # key carries alongside mtime for filesystems with coarse timestamps)
    _write_table(tmp_path, "2MASS/2MASS.Jsomethingelse")
    after = _load_alias_table(tmp_path)

    # ASSERT
    assert before["SVO"].iloc[0] == "2MASS/2MASS.J"
    assert after["SVO"].iloc[0] == "2MASS/2MASS.Jsomethingelse"


def test_a_missing_table_is_still_none(tmp_path):
    """
    Given a directory with no filternames.txt,
    When the alias table is loaded,
    Then None comes back, as before -- the caller then treats user labels
      as canonical names.
    """
    # ARRANGE / ACT / ASSERT
    assert _load_alias_table(tmp_path / "nothing-here") is None


def test_the_shipped_table_still_resolves():
    """
    Given the shipped alias table,
    When it is loaded through the cache,
    Then it is the same table the resolver has always seen.
    """
    # ARRANGE / ACT
    df = _load_alias_table()

    # ASSERT
    assert df is not None
    assert (df["MIST"] == "2MASS_J").any()
