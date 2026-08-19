"""`_provenance_label` maps a numeric rank to a coarse source (review 3.14.4).

The label reaches the user in three places -- run.py's startup table, the
`export_solution` payload and the GUI -- so a wrong one is a wrong statement
about where a number came from, which is exactly what `initval_source` exists
to prevent.

It used to hardcode `rank == RANK_DERIVED_DATA or rank == 30`, so any NEW
intermediate data rank silently reported "solved".  The replacement is a SET
(`DATA_RANKS`) rather than a numeric band, because the two data ranks do not
bracket a contiguous range: RANK_DERIVED_MIXED (40) sits between them and
RANK_DERIVED_USER (80) above them, and both are SOLVER ranks.
"""

import pytest

from exozippy.config import (
    DATA_RANKS,
    RANK_DEFAULT,
    RANK_DERIVED_DATA,
    RANK_DERIVED_MIXED,
    RANK_DERIVED_USER,
    RANK_MULENS_SOURCE_DISTANCE,
    RANK_USER,
    ConfigManager,
)


@pytest.fixture
def label():
    cm = ConfigManager({}, system_config={})
    return cm._provenance_label


@pytest.mark.parametrize(
    "rank, expected",
    [
        (None, "default"),
        (RANK_DEFAULT, "default"),
        (RANK_DEFAULT - 1, "default"),
        (RANK_MULENS_SOURCE_DISTANCE, "data"),
        (RANK_DERIVED_DATA, "data"),
        (RANK_DERIVED_MIXED, "solved"),
        (RANK_DERIVED_USER, "solved"),
        (RANK_USER, "user"),
        (RANK_USER + 1, "user"),
    ],
)
def test_every_declared_rank_maps_to_its_source(label, rank, expected):
    """
    Given each rank the provenance ledger declares,
    When it is labelled,
    Then the coarse source is the one that rank means.
    """
    assert label(rank) == expected


def test_every_data_rank_is_labelled_data(label):
    """
    Given the declared set of data ranks,
    When each is labelled,
    Then all of them say "data" -- so adding a rank to DATA_RANKS is the
      whole of what a new data channel has to do.
    """
    assert {label(rank) for rank in DATA_RANKS} == {"data"}


def test_a_solver_rank_between_the_data_ranks_is_not_called_data(label):
    """
    Given RANK_DERIVED_MIXED, which sits numerically BETWEEN the two data
      ranks,
    When it is labelled,
    Then it says "solved".

    This is why the fix is a set and not the obvious `>= 30 and < RANK_USER`
    band: that band would report every engine-derived value as data-derived,
    the mirror image of the bug it replaced.
    """
    assert RANK_MULENS_SOURCE_DISTANCE < RANK_DERIVED_MIXED < RANK_DERIVED_DATA
    assert label(RANK_DERIVED_MIXED) == "solved"


def test_the_data_ranks_are_named_constants_not_literals():
    """
    Given the data-rank set,
    When its members are compared with the named rank constants,
    Then it is exactly those -- no bare literal survives in the predicate.
    """
    assert DATA_RANKS == frozenset(
        {RANK_DERIVED_DATA, RANK_MULENS_SOURCE_DISTANCE}
    )
    assert RANK_MULENS_SOURCE_DISTANCE == 30  # the microlensing distance hint
