"""Tests for the getdata download utility.

No network access: lightkurve's search is monkeypatched with a fake search
result whose "downloads" are synthetic light curves built in memory, and the
TIC contamination lookup is monkeypatched with a constant.
"""

import lightkurve as lk
import numpy as np
import pytest
from astropy.table import Row, Table

from exozippy.utilities import getdata

TRUE_DEPTH = 0.01  # undiluted (deblended) transit depth
BJD_OFFSET = 2457000.0


def make_lightcurve(depth=TRUE_DEPTH, meta=None, n=400):
    """Return a synthetic PDCSAP-like TESS light curve with one box transit."""
    time = np.linspace(1500.0, 1510.0, n)
    flux = np.full(n, 1000.0)
    flux[(time > 1504.9) & (time < 1505.1)] *= 1.0 - depth
    err = np.full(n, 1.0)
    return lk.LightCurve(time=time, flux=flux, flux_err=err, meta=meta or {})


class FakeSearchResult:
    """The slice of lightkurve's SearchResult API that getdata.run() uses.

    Backed by an astropy Table so that indexing (including the tuple that
    np.where returns) behaves exactly as it does in lightkurve.
    """

    def __init__(self, table, lightcurves):
        self.table = table
        self.lightcurves = lightcurves

    def __len__(self):
        return len(self.table)

    def __getitem__(self, key):
        selection = self.table[key]
        if isinstance(selection, Row):
            selection = Table(selection)
        return FakeSearchResult(selection, self.lightcurves)

    def __iter__(self):
        for i in range(len(self.table)):
            yield self[i]

    @property
    def target_name(self):
        return self.table["target_name"].data

    @property
    def mission(self):
        return self.table["mission"].data

    @property
    def author(self):
        return self.table["author"].data

    @property
    def exptime(self):
        return self.table["exptime"].quantity

    def download(self, *args, **kwargs):
        return self.lightcurves[int(self.table["lcindex"][0])]


def make_search_result(target_names, lightcurves):
    """One SPOC 120 s TESS sector per entry of target_names."""
    n = len(target_names)
    table = Table(
        {
            "target_name": np.array([str(t) for t in target_names]),
            "mission": np.array(
                ["TESS Sector %02d" % (i + 1) for i in range(n)]
            ),
            "author": np.array(["SPOC"] * n),
            "exptime": np.full(n, 120.0),
            "lcindex": np.arange(n),
        }
    )
    table["exptime"].unit = "s"
    return FakeSearchResult(table, lightcurves)


@pytest.fixture
def patched(monkeypatch):
    """Install fakes for the two network calls; return a config dict."""
    state = {
        "target_names": ["375506058"],
        "lightcurves": [make_lightcurve()],
        "contratio": 0.0,
    }

    def fake_search(target_id, **kwargs):
        return make_search_result(state["target_names"], state["lightcurves"])

    monkeypatch.setattr(lk, "search_lightcurve", fake_search)
    monkeypatch.setattr(
        getdata,
        "tic_contamination_ratio",
        lambda target_id, verbose=False: state["contratio"],
    )
    return state


def written_depth(path, pattern):
    """Return 1 - min(flux) from the single file matching pattern."""
    import glob

    files = sorted(glob.glob(str(path / pattern)))
    assert len(files) == 1, "expected one file matching %s, got %r" % (
        pattern,
        files,
    )
    data = np.loadtxt(files[0])
    return 1.0 - data[:, 1].min()


# ---------------------------------------------------------------- 2.5.4


def test_undeblend_dilutes_the_depth_by_crowdsap(patched, tmp_path):
    """
    Given a light curve whose header says CROWDSAP = 0.8,
    When getdata runs with -u,
    Then the written transit depth is 0.8 x the undiluted depth.
    """
    # Arrange
    crowdsap = 0.8
    patched["lightcurves"] = [make_lightcurve(meta={"CROWDSAP": crowdsap})]
    args = getdata.build_parser().parse_args(
        ["TIC375506058", "-u", "-n", "-1", "-p", str(tmp_path)]
    )

    # Act
    getdata.run(args)

    # Assert
    depth = written_depth(tmp_path, "*.undeblended.dat")
    assert depth == pytest.approx(crowdsap * TRUE_DEPTH, rel=1e-9)


def test_without_undeblend_the_depth_is_unchanged(patched, tmp_path):
    """
    Given the same CROWDSAP = 0.8 light curve,
    When getdata runs without -u,
    Then the written depth is the pipeline's undiluted depth.
    """
    # Arrange
    patched["lightcurves"] = [make_lightcurve(meta={"CROWDSAP": 0.8})]
    args = getdata.build_parser().parse_args(
        ["TIC375506058", "-n", "-1", "-p", str(tmp_path)]
    )

    # Act
    getdata.run(args)

    # Assert
    depth = written_depth(tmp_path, "*.SPOC.dat")
    assert depth == pytest.approx(TRUE_DEPTH, rel=1e-9)


def test_undeblend_falls_back_to_the_tic_contamination_ratio(
    patched, tmp_path
):
    """
    Given a light curve with no CROWDSAP but a TIC Rcont of 0.25,
    When getdata runs with -u,
    Then the depth is diluted by 1 / (1 + Rcont).
    """
    # Arrange
    patched["lightcurves"] = [make_lightcurve()]
    patched["contratio"] = 0.25

    args = getdata.build_parser().parse_args(
        ["TIC375506058", "-u", "-n", "-1", "-p", str(tmp_path)]
    )

    # Act
    getdata.run(args)

    # Assert
    depth = written_depth(tmp_path, "*.undeblended.dat")
    assert depth == pytest.approx(TRUE_DEPTH / 1.25, rel=1e-9)


def test_undeblend_without_any_crowding_information_warns_and_proceeds(
    patched, tmp_path, capsys
):
    """
    Given no CROWDSAP and no TIC contamination,
    When getdata runs with -u,
    Then it warns and writes the light curve unchanged.
    """
    # Arrange
    args = getdata.build_parser().parse_args(
        ["TIC375506058", "-u", "-n", "-1", "-p", str(tmp_path)]
    )

    # Act
    getdata.run(args)

    # Assert
    assert "WARNING: no CROWDSAP" in capsys.readouterr().out
    depth = written_depth(tmp_path, "*.undeblended.dat")
    assert depth == pytest.approx(TRUE_DEPTH, rel=1e-9)


def test_reblend_uses_the_additive_form(patched):
    """
    Given a light curve and a crowding fraction,
    When reblend_lightcurve is applied,
    Then flux and error follow f -> C f + (1 - C) once normalized.
    """
    # Arrange
    crowdsap = 0.6
    lc = make_lightcurve()
    reference = lc.normalize()

    # Act
    blended = getdata.reblend_lightcurve(lc, crowdsap).normalize()

    # Assert
    expected_flux = crowdsap * reference.flux.value + (1.0 - crowdsap)
    assert np.allclose(blended.flux.value, expected_flux, rtol=1e-12)
    assert np.allclose(
        blended.flux_err.value,
        crowdsap * reference.flux_err.value,
        rtol=1e-12,
    )


def test_reblend_rejects_an_out_of_range_crowding_fraction():
    """
    Given a crowding fraction outside (0, 1],
    When reblend_lightcurve is called,
    Then it raises rather than applying a wrong correction.
    """
    # Arrange
    lc = make_lightcurve()

    # Act / Assert
    with pytest.raises(ValueError, match="crowdsap must be in"):
        getdata.reblend_lightcurve(lc, 1.5)


def test_crowding_fraction_prefers_the_header_over_the_tic(patched):
    """
    Given both a CROWDSAP header and a TIC contamination ratio,
    When crowding_fraction is asked,
    Then it returns the header value.
    """
    # Arrange
    lc = make_lightcurve(meta={"CROWDSAP": 0.75, "FLFRCSAP": 0.62})

    # Act
    crowdsap, source = getdata.crowding_fraction(lc, contratio=0.25)

    # Assert
    assert crowdsap == pytest.approx(0.75)
    assert source == "CROWDSAP"


def test_crowding_fraction_ignores_an_unusable_header_value(patched, capsys):
    """
    Given a nonsensical CROWDSAP,
    When crowding_fraction is asked,
    Then it warns and falls back on the TIC contamination ratio.
    """
    # Arrange
    lc = make_lightcurve(meta={"CROWDSAP": -1.0})

    # Act
    crowdsap, source = getdata.crowding_fraction(lc, contratio=0.25)

    # Assert
    assert "WARNING: ignoring unusable CROWDSAP" in capsys.readouterr().out
    assert crowdsap == pytest.approx(1.0 / 1.25)
    assert source == "Rcont"


# ---------------------------------------------------------------- 2.5.7


def test_zero_match_tic_id_raises(patched, tmp_path):
    """
    Given a TIC ID that matches none of the returned target names,
    When getdata runs,
    Then it raises naming the targets that were found, and writes nothing.
    """
    # Arrange
    patched["target_names"] = ["375506058", "99999999"]
    patched["lightcurves"] = [make_lightcurve(), make_lightcurve()]
    args = getdata.build_parser().parse_args(
        ["TIC12345678", "-n", "-1", "-p", str(tmp_path)]
    )

    # Act / Assert
    with pytest.raises(ValueError, match="No light curves match"):
        getdata.run(args)
    assert list(tmp_path.iterdir()) == []


def test_matching_tic_id_still_selects_its_light_curves(patched, tmp_path):
    """
    Given a TIC ID that does match one of several target names,
    When getdata runs,
    Then only that target's light curve is written.
    """
    # Arrange
    patched["target_names"] = ["375506058", "99999999"]
    patched["lightcurves"] = [make_lightcurve(), make_lightcurve(depth=0.05)]
    args = getdata.build_parser().parse_args(
        ["TIC375506058", "-n", "-1", "-p", str(tmp_path)]
    )

    # Act
    getdata.run(args)

    # Assert
    depth = written_depth(tmp_path, "*.SPOC.dat")
    assert depth == pytest.approx(TRUE_DEPTH, rel=1e-9)


# ---------------------------------------------------------------- 4.12
#
# The two priority tables and the author -> mission table, pinned. The
# expected orders are spelled out here rather than read back from getdata:
# a test that consults the table it is testing pins nothing.

EXPECTED_EXPTIME_ORDER = [120, 200, 20, 300, 600, 1800]
EXPECTED_AUTHOR_ORDER = [
    "TESS",
    "SPOC",
    "TESS-SPOC",
    "QLP",
    "Kepler",
    "K2SFF",
    "EVEREST",
    "K2",
    "CDIPS",
    "TASOC",
]


def test_the_exptime_chain_prefers_short_cadence_in_order():
    """
    Given every recognized exposure time on offer,
    When the priority chain picks one,
    Then it walks 120, 200, 20, 300, 600, 1800 s -- short cadence first, and
      each successive suffix of the table selects its own head.
    """
    # Arrange / Act / Assert
    for k, want in enumerate(EXPECTED_EXPTIME_ORDER):
        offered = np.array(EXPECTED_EXPTIME_ORDER[k:], dtype=float)
        picked = getdata._highest_priority(offered, getdata.EXPTIME_PRIORITY)
        assert offered[picked].tolist() == [float(want)]


def test_the_author_chain_prefers_pipelines_in_order():
    """
    Given every recognized author on offer,
    When the priority chain picks one,
    Then it walks the documented pipeline order, and each successive suffix
      of the table selects its own head.
    """
    # Arrange / Act / Assert
    for k, want in enumerate(EXPECTED_AUTHOR_ORDER):
        offered = np.array(EXPECTED_AUTHOR_ORDER[k:])
        picked = getdata._highest_priority(offered, getdata.AUTHOR_PRIORITY)
        assert offered[picked].tolist() == [want]


def test_an_unranked_value_selects_nothing():
    """
    Given only values the priority table does not rank,
    When the priority chain is asked,
    Then it returns nothing, so the caller warns and skips the sector rather
      than silently downloading an arbitrary product.
    """
    # Arrange
    offered = np.array([900.0, 1200.0])

    # Act
    picked = getdata._highest_priority(offered, getdata.EXPTIME_PRIORITY)

    # Assert
    assert len(picked) == 0


def test_a_tie_at_the_winning_rank_returns_every_holder():
    """
    Given two products sharing the best-ranked author,
    When the priority chain is asked,
    Then both indices come back -- the caller cannot break the tie and must
      say so rather than pick one.
    """
    # Arrange
    offered = np.array(["SPOC", "QLP", "SPOC"])

    # Act
    picked = getdata._highest_priority(offered, getdata.AUTHOR_PRIORITY)

    # Assert
    assert picked.tolist() == [0, 2]


@pytest.mark.parametrize(
    "author,mission_string,sector,filter_,telescope,offset,undeblend",
    [
        (
            "Kepler",
            "Kepler Quarter 07",
            "Q07",
            "Kepler",
            "Kepler",
            2454833.0,
            False,
        ),
        ("K2", "K2 Campaign 13", "C13", "Kepler", "Kepler", 2454833.0, False),
        (
            "EVEREST",
            "K2 Campaign 04",
            "C04",
            "Kepler",
            "Kepler",
            2454833.0,
            False,
        ),
        (
            "K2SFF",
            "K2 Campaign 04",
            "C04",
            "Kepler",
            "Kepler",
            2454833.0,
            False,
        ),
        (
            "K2VARCAT",
            "K2 Campaign 04",
            "C04",
            "Kepler",
            "Kepler",
            2454833.0,
            False,
        ),
        ("SPOC", "TESS Sector 5", "S05", "TESS", "TESS", 2457000.0, True),
        (
            "TESS-SPOC",
            "TESS Sector 42",
            "S42",
            "TESS",
            "TESS",
            2457000.0,
            True,
        ),
        ("QLP", "TESS Sector 7", "S07", "TESS", "TESS", 2457000.0, True),
    ],
)
def test_each_author_maps_to_its_missions_epoch_band_and_sector_label(
    author, mission_string, sector, filter_, telescope, offset, undeblend
):
    """
    Given a product's author and lightkurve mission string,
    When the mission table is consulted,
    Then it supplies that mission's BJD offset, bandpass, telescope, sector
      label and whether -u (undeblending) is supported there.
    """
    # Arrange
    mission = getdata.MISSIONS[getdata.AUTHOR_MISSION[author]]

    # Act
    label = mission.sector(mission_string)

    # Assert
    assert label == sector
    assert mission.filter == filter_
    assert mission.telescope == telescope
    assert mission.bjd_offset == offset
    assert mission.undeblend is undeblend


def test_only_tess_products_support_undeblending():
    """
    Given the mission table,
    When asked which missions support -u,
    Then only TESS does: CROWDSAP, the quantity the correction inverts, is a
      SPOC/TESS header keyword.
    """
    # Arrange / Act
    supported = {n for n, m in getdata.MISSIONS.items() if m.undeblend}

    # Assert
    assert supported == {"TESS"}


def test_cdips_and_tasoc_are_recognized_but_unreadable():
    """
    Given the two authors whose products put flux and errors elsewhere,
    When the author tables are consulted,
    Then they are listed as unsupported and have no mission, so run() skips
      them with a message instead of reading the wrong column.
    """
    # Arrange / Act / Assert
    assert set(getdata.UNSUPPORTED_AUTHORS) == {"CDIPS", "TASOC"}
    for author in getdata.UNSUPPORTED_AUTHORS:
        assert author not in getdata.AUTHOR_MISSION


def test_the_author_named_tess_is_ranked_but_has_no_reader():
    """
    Given the author priority table,
    When it is compared against the authors run() can actually read,
    Then "TESS" is the one entry with no mission and no unsupported-listing:
      it sorts first, yet a sector offering only "TESS" products is skipped
      with the unrecognized-author warning. Pre-existing behavior, pinned
      here so a future mapping for it is a deliberate change.
    """
    # Arrange
    readable = set(getdata.AUTHOR_MISSION) | set(getdata.UNSUPPORTED_AUTHORS)

    # Act
    unreadable = set(getdata.AUTHOR_PRIORITY) - readable

    # Assert
    assert unreadable == {"TESS"}
