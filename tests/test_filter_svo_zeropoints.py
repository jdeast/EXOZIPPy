"""Scraping SVO's calibration tables for filter zeropoints (review 2.9.1).

These values are read off an HTML page by table and cell POSITION, and then
pickled into the filter cache forever and fed into generated BC tables. The
scrape had no HTTP status check and no schema check at all, so an SVO error
page or a layout change produced a wrong number that nothing would ever
notice. Everything checkable is now checked, and a failure raises instead of
being cached.

Nothing here touches the network: the tables are built in the test.

The rowspan cases at the bottom are the 2026-09-14 regression: SVO spans one
"Zero Point" label down BOTH zeropoint rows, so the second row's label cell
arrives as NaN and the schema check rejected the page outright -- which
blocked BC generation for every facility whose tables were not already
cached. A missing label is not a wrong one, and the unit cell is what
actually identifies the row.
"""

import numpy as np
import pandas as pd
import pytest
import requests

from exozippy.filters.filter import Filter

_URL = "https://svo2.cab.inta-csic.es/theory/fps/index.php?id=X/Y.Z"


def _calibration_table(
    fl="3.129e-9",
    fv="3562.5",
    fl_unit="erg/cm2/s/A",
    fv_unit="Jy",
    fl_label="ZeroPoint_Fl",
    fv_label="ZeroPoint_Fv",
):
    """One SVO-shaped calibration table (header row, then the two rows)."""
    return pd.DataFrame(
        [
            ["", "Specified", "Calculated", "Unit"],
            [fl_label, fl, "3.130e-9", fl_unit],
            [fv_label, fv, "3562.9", fv_unit],
            ["Mag", "0.0", "0.0", "mag"],
        ]
    )


def _filter_stub():
    obj = object.__new__(Filter)
    obj.filterID = "X/Y.Z"
    obj.facility = "X"
    obj.filterName = "Y.Z"
    return obj


def test_a_well_formed_table_sets_all_four_zeropoint_attributes():
    """
    Given an SVO calibration table in the documented layout,
    When it is parsed,
    Then the specified/calculated F_lambda and F_nu zeropoints come back
      under their per-system attribute names.
    """
    # ARRANGE
    df = _calibration_table()

    # ACT
    got = Filter._parse_calibration_table(df, "Vega", _URL)

    # ASSERT
    assert got == {
        "Zp_Spec_Fl_Vega": 3.129e-9,
        "Zp_Calc_Fl_Vega": 3.130e-9,
        "Zp_Spec_Fv_Vega": 3562.5,
        "Zp_Calc_Fv_Vega": 3562.9,
    }


def test_the_missing_value_marker_becomes_none():
    """
    Given a calibration cell SVO has no value for ("--"),
    When it is parsed,
    Then the attribute is None rather than an error.
    """
    # ARRANGE
    df = _calibration_table(fl="--")

    # ACT
    got = Filter._parse_calibration_table(df, "AB", _URL)

    # ASSERT
    assert got["Zp_Spec_Fl_AB"] is None
    assert got["Zp_Calc_Fl_AB"] == 3.130e-9


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"fl_label": "Flux"}, "not a zeropoint"),
        ({"fv_label": "Wavelength"}, "not a zeropoint"),
        ({"fl_unit": "Jy"}, "does not look like a erg unit"),
        ({"fv_unit": "erg/cm2/s/A"}, "does not look like a jy unit"),
        ({"fl": "-1.0"}, "finite and positive"),
        ({"fv": "0"}, "finite and positive"),
        ({"fv": "nan"}, "finite and positive"),
        ({"fl": "not a number"}, "not a\nnumber|not a number"),
    ],
)
def test_a_table_that_does_not_check_out_raises(kwargs, match):
    """
    Given a calibration table whose rows have shifted, whose units name a
      different quantity, or whose value is not a usable flux,
    When it is parsed,
    Then it raises instead of returning a number.

    Raising is the point: this runs inside _set_attrs, BEFORE
    _create_filter_file, so a refusal here is a refusal to cache. The old
    code read the cells positionally with no check and pickled whatever
    came back.
    """
    # ARRANGE
    df = _calibration_table(**kwargs)

    # ACT / ASSERT
    with pytest.raises(ValueError, match=match):
        Filter._parse_calibration_table(df, "ST", _URL)


def test_a_table_that_is_too_small_raises():
    """
    Given a page whose "calibration table" has no zeropoint rows at all,
    When it is parsed,
    Then the shape is reported rather than an IndexError from deep inside
      pandas.
    """
    # ARRANGE
    df = pd.DataFrame([["only", "one", "row"]])

    # ACT / ASSERT
    with pytest.raises(ValueError, match="expected at least 3 rows"):
        Filter._parse_calibration_table(df, "Vega", _URL)


def test_an_http_error_raises_before_anything_is_parsed(monkeypatch):
    """
    Given SVO returns an error page (404, 503, a proxy notice),
    When the calibration tables are fetched,
    Then the status is raised on.

    Regression: pd.read_html(url) fetched the page itself and never saw the
    status, so an error page was just another document with tables in it --
    parsed positionally, and cached.
    """
    # ARRANGE
    filt = _filter_stub()

    class _Resp:
        status_code = 503
        text = (
            "<html><table><tr><td>Service Unavailable</td></tr></table></html>"
        )

        def raise_for_status(self):
            raise requests.HTTPError("503 Server Error")

    class _Session:
        def get(self, url, timeout=None):
            return _Resp()

    filt._session = _Session()

    # ACT / ASSERT
    with pytest.raises(requests.HTTPError):
        filt._fetch_calibration_tables(_URL)


def test_a_page_with_too_few_tables_raises(monkeypatch):
    """
    Given a page carrying fewer than the three calibration tables,
    When the zeropoints are read,
    Then it raises and says so, rather than indexing off the front of the
      list and parsing whatever is there.
    """
    # ARRANGE
    filt = _filter_stub()
    monkeypatch.setattr(
        Filter,
        "_fetch_calibration_tables",
        lambda self, url: [_calibration_table()],
    )

    # ACT / ASSERT
    with pytest.raises(ValueError, match="only 1 HTML table"):
        filt._set_zeropoint_values()


def test_the_three_systems_are_read_in_order(monkeypatch):
    """
    Given the three calibration tables SVO ends its page with,
    When the zeropoints are read,
    Then Vega, AB and ST each land on their own attributes.
    """
    # ARRANGE
    filt = _filter_stub()
    tables = [
        pd.DataFrame([["something", "else"]]),
        _calibration_table(fl="1e-9"),
        _calibration_table(fl="2e-9"),
        _calibration_table(fl="3e-9"),
    ]
    monkeypatch.setattr(
        Filter, "_fetch_calibration_tables", lambda self, url: tables
    )

    # ACT
    filt._set_zeropoint_values()

    # ASSERT
    assert filt.Zp_Spec_Fl_Vega == 1e-9
    assert filt.Zp_Spec_Fl_AB == 2e-9
    assert filt.Zp_Spec_Fl_ST == 3e-9
    assert np.isfinite(filt.Zp_Spec_Fv_Vega)


def _svo_rowspan_table(fv_unit="(Jy)", fl_label="Zero Point (?) :"):
    """The layout SVO actually served on 2026-09-14, reproduced exactly.

    Captured from Roman/WFI.F146, and identical in shape on
    Roman/WFI.F087 and 2MASS/2MASS.Ks. The F_nu row's label is NaN because
    the "Zero Point" cell above it carries a rowspan of 2, and the
    "-------" in the Specified column is SVO's missing-value marker.
    """
    return pd.DataFrame(
        [
            ["Property", "Specified", "Calculated", "Unit"],
            [fl_label, "-------", "2.02559e-10", "(erg/cm2/s/A)"],
            [np.nan, "-------", "1396.74", fv_unit],
            ["ZP Type (?) :", "Pogson", "Pogson", "Pogson"],
            [
                "PhotCal ID (?) :",
                "Roman/WFI.F146/Vega",
                "Roman/WFI.F146/Vega",
                "Roman/WFI.F146/Vega",
            ],
            [np.nan, np.nan, np.nan, np.nan],
        ]
    )


def test_the_live_svo_rowspan_layout_parses():
    """
    Given the calibration table SVO serves today, whose F_nu row inherits
      its label from a rowspan and so reads as NaN,
    When it is parsed,
    Then both zeropoints come back, and the absent "Specified" values are
      None rather than an error.

    Regression: this raised "row 2 is labelled 'nan', not a zeropoint" and
    took the whole BC auto-generation path down with it, so a machine with
    no cached Roman tables could not build a Roman SED at all.
    """
    # ARRANGE
    df = _svo_rowspan_table()

    # ACT
    got = Filter._parse_calibration_table(df, "Vega", _URL)

    # ASSERT
    assert got == {
        "Zp_Spec_Fl_Vega": None,
        "Zp_Calc_Fl_Vega": 2.02559e-10,
        "Zp_Spec_Fv_Vega": None,
        "Zp_Calc_Fv_Vega": 1396.74,
    }


def test_a_rowspan_continuation_still_requires_a_confirming_unit():
    """
    Given a row whose label is missing AND whose unit does not name the
      expected quantity,
    When it is parsed,
    Then it raises.

    This is why accepting the NaN label is not a guess: with no label, the
    unit cell is the only thing left that identifies the row, so it becomes
    mandatory rather than optional. The accepted row is checked MORE
    strictly than a labelled one, not less.
    """
    # ARRANGE
    df = _svo_rowspan_table(fv_unit="(erg/cm2/s/A)")

    # ACT / ASSERT
    with pytest.raises(ValueError, match="not a zeropoint"):
        Filter._parse_calibration_table(df, "Vega", _URL)


def test_a_missing_label_on_the_first_zeropoint_row_still_raises():
    """
    Given a table whose FIRST zeropoint row has no label either,
    When it is parsed,
    Then it raises, because there is no labelled row above it for the NaN
      to be a continuation OF.

    Without this, "the label may be missing" would degrade into reading
    rows 1 and 2 positionally again, which is the whole of 2.9.1.
    """
    # ARRANGE
    df = _svo_rowspan_table(fl_label=np.nan)

    # ACT / ASSERT
    with pytest.raises(ValueError, match="not a zeropoint"):
        Filter._parse_calibration_table(df, "Vega", _URL)
