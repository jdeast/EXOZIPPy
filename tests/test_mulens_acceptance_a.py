"""The `mulensevent` acceptance gate, file A of two (review 8.6.17).

Replays GROUP_A of the recorded fixtures -- ob09020, the KMT-2019-BLG-1806
family and the DC2018_128 family -- against their recorded per-term logp
decompositions, and carries the DC2018_128 potential inventory.  Its twin,
test_mulens_acceptance_b.py, replays GROUP_B; the tolerances, the helpers and
the whole account of WHY the gate exists are in mulens_acceptance_replay.py.

WHY TWO FILES.  `--dist loadfile` pins a file to one xdist worker, so a CI
shard can never finish faster than its slowest file's SERIAL time.  As one
file this gate cost 873 worker-seconds on CI (2026-09-14), the heaviest file
in the suite and the floor under the shard that carried it.  Two files of
~435 s each dissolve the floor.  A NEW FIXTURE goes into whichever of
GROUP_A / GROUP_B is lighter in tests/durations.json;
`test_the_two_files_partition_the_fixture_set` below is what fails if it goes
in neither or both.
"""

import os

import pytest
from mulens_acceptance import term_names
from mulens_acceptance_replay import (
    FIXTURES,
    GROUP_A,
    GROUP_B,
    HERE,
    build,
    check_recorded_decomposition,
    fixture_files,
    fixture_names,
    load_fixture,
    replay_cases,
)


def test_the_fixture_set_is_not_empty():
    """
    Given the recorded stage-0 fixtures,
    When they are collected,
    Then there is at least one.

    Guards the whole gate against becoming a silent no-op: the replay tests
    in both files are parameterized over the fixture set, so an empty
    directory would make them all vanish while the suite stayed green --
    which is exactly how a green suite comes to mean nothing.
    """
    assert fixture_files(), f"no fixtures in {FIXTURES}"


def test_the_two_files_partition_the_fixture_set():
    """
    Given the fixtures on disk and the two group lists,
    When they are compared,
    Then every fixture is in exactly one group, and both test files exist.

    THE SPLIT'S OWN GUARD.  A fixture in neither group would be recorded,
    committed and never replayed -- the same silent no-op as an empty
    directory, one fixture at a time.  A fixture in both would be replayed
    twice, which is a cost with no coverage.  And the point of two files is
    two FILES: if someone folds B back into A the loadfile floor returns
    without anything else failing, so the sibling's existence is asserted
    too.
    """
    # Arrange
    on_disk = set(fixture_names())
    a, b = set(GROUP_A), set(GROUP_B)

    # Assert -- disjoint, exhaustive, and nothing listed that is not there.
    assert not (a & b), f"fixtures in both groups: {sorted(a & b)}"
    assert on_disk == a | b, (
        f"fixtures in neither group (add them to the lighter one in "
        f"tests/durations.json): {sorted(on_disk - (a | b))}; "
        f"listed but not on disk: {sorted((a | b) - on_disk)}"
    )
    for sibling in (
        "test_mulens_acceptance_a.py",
        "test_mulens_acceptance_b.py",
    ):
        assert os.path.exists(os.path.join(HERE, sibling)), (
            f"{sibling} is missing: the gate is split in two so that "
            f"--dist loadfile has no 870 s file to pin to one worker"
        )


@pytest.mark.slow
def test_the_event_potentials_are_single_counted():
    """
    Given the DC2018_128 model (Op path, one companion, one source),
    When its logp terms are enumerated,
    Then each event potential exists exactly once, under the mulensevent
    prefix, and no term name is duplicated.

    The design's section-4 potential inventory, at its stage-2 vehicle
    (design section 7).  What a single-source event CAN pin here: the event
    potentials moved to their owning component without leaving a lens.*
    twin behind (a duplicate would double-count the event-rate weight, the
    defect class section 4 exists for), and `term_names` is collision-free
    so the decomposition's attribution is trustworthy.  What it CANNOT pin:
    that the event-rate term is single-counted ACROSS SOURCES -- with one
    source the sum and the scalar coincide; ob161003 (2S2L) carries that
    check at stage 3 (section 4's analytic reconciliation, file B).
    """
    # ARRANGE
    _, model = build(load_fixture("DC2018_128"))

    # ACT
    names = term_names(model)
    counts = {n: names.count(n) for n in names}

    # ASSERT -- no duplicates anywhere (a duplicate name is silently summed
    # by the decomposition, so it would LOOK reconciled while double
    # counting).
    dupes = {n: c for n, c in counts.items() if c > 1}
    assert not dupes, f"duplicated logp terms: {dupes}"

    for pot in (
        "POT:mulensevent.event_rate_prior",
        "POT:mulensevent.source_behind_lens",
        "POT:mulensevent.mu_rel_singularity",
        "POT:mulensevent.theta_E_singularity",
    ):
        assert counts.get(pot) == 1, f"{pot} missing or duplicated"

    # And the pre-split homes are GONE -- the move must not have left a
    # lens.* twin adding the same weight twice.
    stale = [
        n
        for n in names
        if n.startswith("POT:lens.")
        and n.split(".", 1)[1]
        in (
            "event_rate_prior",
            "source_behind_lens",
            "mu_rel_singularity",
            "theta_E_singularity",
            "fitpirel_jacobian",
            "fitu0te_jacobian",
        )
    ]
    assert not stale, f"pre-split event potentials survive on lens: {stale}"


@pytest.mark.slow
@pytest.mark.parametrize("name", replay_cases(GROUP_A))
def test_the_model_still_matches_its_recorded_decomposition(name):
    """
    Given a stage-0 fixture recorded before the split,
    When the example is rebuilt and decomposed,
    Then every term matches.

    This is the refactor's regression gate. It is EXPECTED to fail during the
    split -- that is its purpose. A failure must be read term by term and
    every moved term explained; a term that vanished alongside one that
    appeared with the same value is a rename, which the diff reports
    separately so it is not mistaken for a match.

    Evaluated at the fixture's STORED start, not at this machine's own solved
    one, so a cross-machine failure means the likelihood function differs
    rather than the solver -- see TERM_RTOL in mulens_acceptance_replay.py.
    """
    check_recorded_decomposition(name)
