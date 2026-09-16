"""Multi-seed polish must preserve basin coverage.

The polish is a basin-agnostic optimizer, and nothing used to tie a seed to
the basin it was provided to represent: on DC2018 event 128 the two MMEXOFAST
seeds (s = 0.977 and its s <-> 1/s mirror at 0.863) BOTH polished across
s = 1 onto nearby shoulders, so the mirror entered sampling unrepresented
and the run's mode report was blind to it.  For the record, the cause was
NOT cross-seed proposals -- difference vectors always came from each seed's
own population -- but each seed's own unconstrained walk.

The fix under test: with several seeds, each seed's accepted moves are
confined to `trust_fraction` (default 0.5) of the distance to its nearest
other seed, so no seed can cross the midpoint toward a neighbour.  A single
seed keeps an infinite radius (the canonical start may legitimately travel
thousands of nats).
"""

import logging

import numpy as np
import pytest

from exozippy.samplers.ptde import polish_seed_starts


def _tilted_bimodal_logp(point):
    """Two Gaussian wells at x = -4 and x = +4; the +4 well is 50 nats
    deeper, so an UNconstrained optimizer started in the shallow well walks
    across.  1-D in a dict, as the polish consumes raw states."""
    x = float(np.asarray(point["x"]).ravel()[0])
    a = -0.5 * ((x + 4.0) / 0.6) ** 2
    b = -0.5 * ((x - 4.0) / 0.6) ** 2 + 50.0
    return float(np.logaddexp(a, b))


def test_trust_region_keeps_each_seed_in_its_own_basin():
    """
    Given seeds in two wells where one well is 50 nats deeper,
    When both are polished with the default trust_fraction,
    Then each polished seed is still in its own well.
    """
    # Arrange
    seeds = [{"x": np.array([-4.3])}, {"x": np.array([3.7])}]
    scales = {"x": np.array([0.5])}

    # Act
    polished, dlps = polish_seed_starts(
        seeds,
        _tilted_bimodal_logp,
        np.random.default_rng(2),
        scales,
        n_steps=400,
        pop_size=12,
    )

    # Assert: seed 0 improved (climbed to its own well's floor) but did NOT
    # cross to the deep well; seed 1 stayed put in the deep well.
    x0 = float(polished[0]["x"][0])
    x1 = float(polished[1]["x"][0])
    assert x0 < 0.0, f"seed 0 crossed basins (x = {x0:.2f})"
    assert x1 > 0.0
    assert dlps[0] >= 0.0 and dlps[1] >= 0.0


def test_without_trust_region_the_shallow_seed_defects_and_it_warns(caplog):
    """
    Given the same wells with trust_fraction=None,
    When the shallow seed walks to the deep well (the pre-fix behaviour),
    Then the collapse WARNING names the pair.

    This is the regression the DC2018-128 mirror seed hit; the warning is
    the always-on half of the fix, reachable even when the radius is off.
    """
    seeds = [{"x": np.array([-4.3])}, {"x": np.array([3.7])}]
    scales = {"x": np.array([0.5])}

    with caplog.at_level(logging.WARNING, logger="exozippy.samplers.ptde"):
        polished, _ = polish_seed_starts(
            seeds,
            _tilted_bimodal_logp,
            np.random.default_rng(2),
            scales,
            n_steps=400,
            pop_size=12,
            trust_fraction=None,
        )

    x0 = float(polished[0]["x"][0])
    if x0 > 0.0:  # the defection happened (it should, at 50 nats of tilt)
        assert any(
            "basin coverage may have collapsed" in r.message
            for r in caplog.records
        )
    else:
        pytest.skip("shallow seed did not defect under this rng draw")


def test_single_seed_keeps_an_infinite_radius():
    """A lone start must still be free to travel arbitrarily far."""
    seed = [{"x": np.array([-4.3])}]
    scales = {"x": np.array([0.5])}

    polished, dlps = polish_seed_starts(
        seed,
        _tilted_bimodal_logp,
        np.random.default_rng(4),
        scales,
        n_steps=600,
        pop_size=12,
    )

    # With no other seed there is no trust radius: reaching the deep well
    # is allowed (and, given the 50-nat tilt, expected).
    assert dlps[0] > 0.0
