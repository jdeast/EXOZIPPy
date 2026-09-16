"""`_provenance_label` maps a numeric rank to a coarse source (review 3.14.4).

The label reaches the user in three places -- run.py's startup table, the
`export_solution` payload and the GUI -- so a wrong one is a wrong statement
about where a number came from, which is exactly what `initval_source` exists
to prevent.

It used to hardcode `rank == PRECEDENCE_DERIVED_DATA or rank == 30`, so any NEW
intermediate data rank silently reported "solved".  The replacement is a SET
(`DATA_PRECEDENCES`) rather than a numeric band, because the two data ranks do not
bracket a contiguous range: PRECEDENCE_DERIVED_MIXED (40) sits between them and
PRECEDENCE_DERIVED_USER (80) above them, and both are SOLVER ranks.
"""

import pytest

from exozippy.config import (
    DATA_PRECEDENCES,
    PRECEDENCE_DEFAULT,
    PRECEDENCE_DERIVED_DATA,
    PRECEDENCE_DERIVED_MIXED,
    PRECEDENCE_DERIVED_USER,
    PRECEDENCE_MULENS_SOURCE_DISTANCE,
    PRECEDENCE_USER,
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
        (PRECEDENCE_DEFAULT, "default"),
        (PRECEDENCE_DEFAULT - 1, "default"),
        (PRECEDENCE_MULENS_SOURCE_DISTANCE, "data"),
        (PRECEDENCE_DERIVED_DATA, "data"),
        (PRECEDENCE_DERIVED_MIXED, "solved"),
        (PRECEDENCE_DERIVED_USER, "solved"),
        (PRECEDENCE_USER, "user"),
        (PRECEDENCE_USER + 1, "user"),
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
    Then all of them say "data" -- so adding a rank to DATA_PRECEDENCES is the
      whole of what a new data channel has to do.
    """
    assert {label(rank) for rank in DATA_PRECEDENCES} == {"data"}


def test_a_solver_rank_between_the_data_ranks_is_not_called_data(label):
    """
    Given PRECEDENCE_DERIVED_MIXED, which sits numerically BETWEEN the two data
      ranks,
    When it is labelled,
    Then it says "solved".

    This is why the fix is a set and not the obvious `>= 30 and < PRECEDENCE_USER`
    band: that band would report every engine-derived value as data-derived,
    the mirror image of the bug it replaced.
    """
    assert (
        PRECEDENCE_MULENS_SOURCE_DISTANCE
        < PRECEDENCE_DERIVED_MIXED
        < PRECEDENCE_DERIVED_DATA
    )
    assert label(PRECEDENCE_DERIVED_MIXED) == "solved"


def test_the_data_ranks_are_named_constants_not_literals():
    """
    Given the data-rank set,
    When its members are compared with the named rank constants,
    Then it is exactly those -- no bare literal survives in the predicate.
    """
    assert DATA_PRECEDENCES == frozenset(
        {PRECEDENCE_DERIVED_DATA, PRECEDENCE_MULENS_SOURCE_DISTANCE}
    )
    assert (
        PRECEDENCE_MULENS_SOURCE_DISTANCE == 30
    )  # the microlensing distance hint
