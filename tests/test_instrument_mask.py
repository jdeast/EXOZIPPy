"""Tests for the generic Instrument per-file mask feature.

The ``mask:`` key on any instrument config entry excludes data points at
load time. It lives on the shared Instrument base, so RVInstrument stands
in for all four data components here.
"""

import numpy as np
import pytest

from exozippy.components.rvinstrument.rvinstrument import RVInstrument
from exozippy.config import ConfigManager


def _write_rv_file(path, times, values=None, errs=None):
    times = np.asarray(times, dtype=float)
    values = (
        np.asarray(values, dtype=float)
        if values is not None
        else np.arange(len(times), dtype=float)
    )
    errs = (
        np.asarray(errs, dtype=float)
        if errs is not None
        else np.ones(len(times))
    )
    with open(path, "w") as f:
        f.write("# time rv err\n")
        for t, v, e in zip(times, values, errs):
            f.write(f"{t:.6f} {v:.6f} {e:.6f}\n")


def _load(tmp_path, mask, times=(1.0, 2.0, 3.0, 4.0, 5.0)):
    data_file = tmp_path / "inst.rv"
    _write_rv_file(data_file, times)
    config = [{"name": "Inst", "file": str(data_file), "mask": mask}]
    inst = RVInstrument(config, ConfigManager({}))
    inst.load_data(system=None)
    return inst


def test_no_mask_keeps_every_point(tmp_path):
    """
    Given an instrument entry without a mask key,
    When the data file is loaded,
    Then every data point survives (the pre-feature behavior).
    """
    inst = _load(tmp_path, mask=None)
    assert inst.n_total_obs == 5


def test_index_list_mask_excludes_rows(tmp_path):
    """
    Given a mask that is a list of 0-based row indices,
    When the data file is loaded,
    Then exactly those rows are excluded from the concatenated arrays.
    """
    inst = _load(tmp_path, mask=[0, 3])
    assert inst.n_total_obs == 3
    np.testing.assert_allclose(inst.time, [2.0, 3.0, 5.0])


def test_boolean_list_mask_excludes_flagged_rows(tmp_path):
    """
    Given a mask that is a per-row boolean list (True = exclude),
    When the data file is loaded,
    Then the flagged rows are excluded.
    """
    inst = _load(tmp_path, mask=[False, True, False, False, True])
    assert inst.n_total_obs == 3
    np.testing.assert_allclose(inst.time, [1.0, 3.0, 4.0])


def test_mask_file_excludes_flagged_rows(tmp_path):
    """
    Given a mask that is a path to a file of 0/1 flags (1 = exclude),
    When the data file is loaded,
    Then the flagged rows are excluded.
    """
    mask_file = tmp_path / "inst.mask"
    mask_file.write_text("# flag per row\n0\n1\n0\n1\n0\n")
    inst = _load(tmp_path, mask=str(mask_file))
    assert inst.n_total_obs == 3
    np.testing.assert_allclose(inst.time, [1.0, 3.0, 5.0])


def test_mask_indices_refer_to_raw_file_row_order(tmp_path):
    """
    Given an unsorted data file and an index mask naming its first raw row,
    When the data file is loaded (mask, then sort),
    Then the excluded row is the file's first ON-DISK row, not the earliest
    time, and the survivors come out time-sorted.
    """
    # Raw row 0 has the LATEST time; masking [0] must remove t=9.
    inst = _load(tmp_path, mask=[0], times=(9.0, 1.0, 5.0, 3.0, 7.0))
    assert inst.n_total_obs == 4
    np.testing.assert_allclose(inst.time, [1.0, 3.0, 5.0, 7.0])


def test_mask_file_word_literals_raise_a_clear_error(tmp_path):
    """
    Given a mask FILE written with true/false word literals rather than 1/0,
    When the data file is loaded,
    Then a ValueError naming the instrument, the mask file and the two
    working spellings is raised -- not numpy's bare "could not convert
    string 'true' to float64", which names none of them.

    The flag file is deliberately numeric-only (``np.loadtxt(dtype=float)``);
    the boolean spelling lives on the inline YAML list form, exercised by
    test_boolean_list_mask_excludes_flagged_rows.
    """
    mask_file = tmp_path / "inst.mask"
    mask_file.write_text("false\ntrue\nfalse\ntrue\nfalse\n")
    with pytest.raises(ValueError) as excinfo:
        _load(tmp_path, mask=str(mask_file))
    msg = str(excinfo.value)
    assert "Inst" in msg
    assert str(mask_file) in msg
    assert "numeric flag" in msg
    assert "1/0" in msg
    assert "booleans" in msg


def test_mask_file_length_mismatch_raises(tmp_path):
    """
    Given a mask file whose flag count differs from the data row count,
    When the data file is loaded,
    Then a ValueError naming both counts is raised.
    """
    mask_file = tmp_path / "inst.mask"
    mask_file.write_text("0\n1\n0\n")
    with pytest.raises(ValueError, match="3 entries.*5 rows"):
        _load(tmp_path, mask=str(mask_file))


def test_mask_index_out_of_range_raises(tmp_path):
    """
    Given an index mask referencing a row past the end of the file,
    When the data file is loaded,
    Then a ValueError describing the legal index range is raised.
    """
    with pytest.raises(ValueError, match="0-based row indices"):
        _load(tmp_path, mask=[2, 17])


def test_mask_excluding_all_points_raises(tmp_path):
    """
    Given a mask that excludes every data point in the file,
    When the data file is loaded,
    Then a ValueError is raised instead of building an empty instrument.
    """
    with pytest.raises(ValueError, match="every one"):
        _load(tmp_path, mask=[0, 1, 2, 3, 4])


def test_mixed_type_mask_list_raises(tmp_path):
    """
    Given a mask list mixing booleans and integers,
    When the data file is loaded,
    Then a ValueError explaining the accepted list forms is raised.
    """
    with pytest.raises(ValueError, match="all booleans.*or all integers"):
        _load(tmp_path, mask=[True, 2, 3])


# --------------------------------------------------------------------------
# 2.5.2: an all-0/1 integer list, one per row, is ambiguous
# --------------------------------------------------------------------------
def test_all_zero_one_integer_mask_raises_as_ambiguous(tmp_path):
    """
    Given an inline mask of 5 integers, all 0 or 1, on a 5-row file,
    When the data file is loaded,
    Then a ValueError names the ambiguity and both explicit spellings.

    The two readings are OPPOSITES: as row INDICES (what the integer branch
    does) [1, 0, 1, 0, 1] excludes rows 0 and 1; as per-row FLAGS -- what
    the same numbers mean in the mask-FILE form this list mirrors -- it
    excludes rows 0, 2 and 4.  Picking either silently is the bug.
    """
    with pytest.raises(ValueError) as exc:
        _load(tmp_path, mask=[1, 0, 1, 0, 1])

    msg = str(exc.value)
    assert "ambiguous" in msg
    # both readings are spelled out, so the user can see which they meant
    assert "[0, 1]" in msg  # as row indices
    assert "[0, 2, 4]" in msg  # as per-row flags
    assert "booleans" in msg


def test_ambiguous_mask_is_accepted_as_booleans(tmp_path):
    """
    Given the same flags written as booleans,
    When the data file is loaded,
    Then they are per-row flags and the named rows are excluded -- the
    spelling the error tells the user to use.
    """
    inst = _load(tmp_path, mask=[True, False, True, False, True])

    assert inst.n_total_obs == 2
    np.testing.assert_allclose(inst.time, [2.0, 4.0])


def test_short_zero_one_integer_mask_is_still_row_indices(tmp_path):
    """
    Given an integer mask of 0s and 1s that is SHORTER than the file,
    When the data file is loaded,
    Then it is read as row indices with no complaint.

    The guard is deliberately narrow: only a list whose length equals the
    row count could be a flag vector, so `mask: [0, 1]` on a 5-row file has
    exactly one sensible reading and keeps it.
    """
    inst = _load(tmp_path, mask=[0, 1])

    assert inst.n_total_obs == 3
    np.testing.assert_allclose(inst.time, [3.0, 4.0, 5.0])
