"""Band-referenced filters reach the BC grid (review 2.9.6).

A filter named on a `band:` block is what the cross-component SED flux
hooks key on -- the mulensing zeropoint tie, the transit dilution, the
astrometry fluxfrac. `_collect_band_filters` used to DROP such a filter
whenever its facility had no BC tables yet, warning that the tables "can
be generated with the BC table machinery" -- which is exactly what
`build_bc_grid` did, unprompted, for a filter listed in the .sed file and
for a missing column inside a facility that does exist. The band filter
was the one case that silently lost its constraint instead.

Since 2026-09-28 `build_bc_grid` no longer generates anything: it REFUSES
a filter no table covers, naming the request path.  A band filter whose SED
flux a component READS (`Component.sed_flux_bands`) is still passed through
-- synthesized first where that applies (tests/test_bc_synthesis.py), and
refused loudly otherwise -- rather than dropped quietly, which is what these
tests check.  One nothing reads, with no column, is left out
(tests/test_bc_synthesis.py).
"""

import logging

from exozippy.components.sed.sed import SED


class _StubConfigManager:
    def __init__(self, bands):
        self.system_config = {"band": bands}


# Every band is READ by some component: the case where dropping the band
# would lose a constraint.
_READ = {str(i): ["mulensinstrument"] for i in range(4)}


def _sed_stub(bands, tmp_path):
    """An SED with only what _collect_band_filters reads populated."""
    sed = object.__new__(SED)
    sed.config_manager = _StubConfigManager(bands)
    sed.filters = []
    sed.sedmodel = "NextGen"
    # An empty model root: every facility is "missing", which is the
    # situation under test -- and an explicit model_root, so no column is
    # synthesized: the read filter goes on to the grid build's refusal.
    sed.model_root = tmp_path
    return sed


def test_a_band_filter_with_no_tables_is_passed_on_to_the_grid_build(tmp_path):
    """
    Given a band naming a real SVO filter whose facility has no BC tables,
      and whose SED flux a component reads,
    When the band filters are collected,
    Then it is returned, so build_bc_grid sees it (and refuses it loudly).

    Regression: it was dropped, and with it the SED flux constraint the
    band exists to carry.
    """
    # ARRANGE
    sed = _sed_stub([{"filter": "2MASS/2MASS.J"}], tmp_path)

    # ACT
    extra = sed._collect_band_filters(_READ)

    # ASSERT
    assert extra == ["2MASS/2MASS.J"]


def test_a_filter_resolved_through_the_alias_table_counts_too(tmp_path):
    """
    Given a band naming a filter by its short label,
    When the band filters are collected,
    Then it is passed on -- the alias table gives it an SVO identity, so
      the generator has a bandpass to work from.
    """
    # ARRANGE
    sed = _sed_stub([{"filter": "2MASS.J"}], tmp_path)

    # ACT
    extra = sed._collect_band_filters(_READ)

    # ASSERT
    assert extra == ["2MASS.J"]


def test_a_label_with_no_svo_identity_is_still_skipped(tmp_path, caplog):
    """
    Given a band whose filter label resolves to no SVO id at all
      (examples/gj1214's "MIRILRS"),
    When the band filters are collected,
    Then it is skipped, and the warning says there is no bandpass to
      generate FROM rather than pointing at machinery that cannot help.
    """
    # ARRANGE
    sed = _sed_stub([{"filter": "MIRILRS"}], tmp_path)

    # ACT
    with caplog.at_level(logging.WARNING):
        extra = sed._collect_band_filters(_READ)

    # ASSERT
    assert extra == []
    assert "no SVO filter id" in caplog.text


def test_a_filter_already_in_the_sed_file_is_not_added_twice(tmp_path):
    """
    Given a band naming a filter the .sed file already lists,
    When the band filters are collected,
    Then nothing extra is returned.
    """
    # ARRANGE
    sed = _sed_stub([{"filter": "2MASS.J"}], tmp_path)
    sed.filters = ["2MASS/2MASS.J"]

    # ACT
    extra = sed._collect_band_filters(_READ)

    # ASSERT
    assert extra == []


def test_a_band_with_no_filter_key_is_ignored(tmp_path):
    """
    Given a band block that names no filter,
    When the band filters are collected,
    Then it contributes nothing and nothing raises.
    """
    # ARRANGE
    sed = _sed_stub([{"name": "unnamed"}], tmp_path)

    # ACT
    extra = sed._collect_band_filters(_READ)

    # ASSERT
    assert extra == []
