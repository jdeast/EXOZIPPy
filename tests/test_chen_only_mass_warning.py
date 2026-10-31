"""A Chen-only planet mass in the degenerate Jovian regime is WARNED about.

Review 2.8.18: in a transit-only fit the planet mass is constrained only by
the Chen & Kipping (2017) relation, which from ~1 RJ up is nearly flat in
radius with kinks at its regime breaks, so NUTS does not mix the mass (SBC:
218 of 300 transit replicates failed convergence on log_q, 296 of 300
converged with it pinned).  JDE 2026-10-30: do not smooth the relation (it
would no longer be Chen & Kipping) and do not change the coordinate; warn at
build time and recommend a pin or a mass prior.
"""

import logging

import numpy as np
import pytest

from exozippy.components.planet import physics
from exozippy.constants import RJUP_TO_REARTH
from exozippy.system import System

TC = 2459634.3
P = 2.99


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    d = tmp_path_factory.mktemp("chen_only")
    rng = np.random.default_rng(11)
    t = np.linspace(TC - 0.2, TC + 0.2, 150)
    lc = d / "lc.dat"
    np.savetxt(
        lc,
        np.column_stack(
            [t, 1.0 + rng.normal(0.0, 1e-3, t.size), np.full(t.size, 1e-3)]
        ),
    )
    t_rv = np.linspace(TC - 30.0, TC + 30.0, 30)
    rv = -100.0 * np.sin(2 * np.pi * (t_rv - TC) / P) + rng.normal(
        0, 5.0, t_rv.size
    )
    rvf = d / "rv.dat"
    np.savetxt(rvf, np.column_stack([t_rv, rv, np.full(t_rv.size, 5.0)]))
    return str(lc), str(rvf)


def _system(data, radius_rj=1.0, rv=False, extra=None):
    lc, rvf = data
    config = {
        "star": [{"name": "A", "mist": False}],
        "planet": [{"name": "b"}],
        "orbit": [{"name": "b", "primary": ["A"], "companion": ["b"]}],
        "band": [{"name": "TESS", "filter": "TESS"}],
        "transit": [{"name": "inst0", "file": lc, "band": "TESS"}],
    }
    if rv:
        config["rvinstrument"] = [{"name": "inst", "file": rvf}]
    params = {
        "star.0.radius": {"initval": 1.61, "sigma": 0.05},
        "star.0.mass": {"initval": 1.204, "sigma": 0.05},
        "star.0.teff": {"initval": 6207, "sigma": 100},
        "star.0.feh": {"initval": -0.116, "sigma": 0.08},
        "orbit.0.period": {"initval": P},
        "orbit.0.tc": {"initval": TC},
        "orbit.0.cosi": {"initval": 0.05},
        "planet.0.radius": {"initval": radius_rj},
    }
    params.update(extra or {})
    system = System(config, user_params=params)
    system.prepare()
    return system


def test_the_threshold_is_the_jovian_segment_less_three_sigma():
    """The edge is R(T3) (1.003 RJ) less three Jovian fractional scatters."""
    assert physics.CHEN_DEGENERATE_RPEARTH / RJUP_TO_REARTH == pytest.approx(
        0.781, abs=1e-3
    )


def test_a_transit_only_jupiter_warns_at_build(data, caplog):
    """
    Given a transit-only fit of a 1 RJ planet with no mass constraint,
    When the model is built,
    Then a WARNING names the planet, says the mass is unlikely to mix, and
      recommends a pin or a prior.
    """
    system = _system(data)
    with caplog.at_level(logging.WARNING):
        system.build_model()
    hits = [
        r.getMessage()
        for r in caplog.records
        if "Chen & Kipping" in r.getMessage() and r.levelno == logging.WARNING
    ]
    assert len(hits) == 1
    msg = hits[0]
    assert "[planet.b]" in msg
    assert "unlikely to mix" in msg
    assert "sigma: 0" in msg and "mu/sigma" in msg
    # The pin it recommends is on the SAMPLED coordinate (log_q here): a pin
    # on the derived planet.mass would raise.
    assert "planet.b.log_q:" in msg


def test_a_small_transit_only_planet_is_not_warned(data):
    """
    Given a transit-only fit of a 0.2 RJ planet (Neptunian: R(M) is
      monotonic there, one radius names one mass),
    Then nothing is flagged.
    """
    system = _system(data, radius_rj=0.2)
    assert system.planet.chen_only_mass_planets(system) == {}


def test_rv_reading_the_mass_silences_it(data):
    """
    Given the same 1 RJ planet with RVs that measure its mass,
    Then Chen does not derive the mass, and nothing is flagged.
    """
    system = _system(data, rv=True)
    assert system.planet.chen_only_mass_planets(system) == {}


@pytest.mark.parametrize(
    "extra",
    [
        # The default transit-only coordinate is log_q, so planet.mass is
        # derived and a sigma: 0 on it raises (2.1.26(e)): pin log_q.
        {"planet.b.log_q": {"initval": -3.0, "sigma": 0}},
        {"planet.b.mass": {"initval": 1.0, "mu": 1.0, "sigma": 0.3}},
    ],
    ids=["pinned", "prior"],
)
def test_a_user_pin_or_prior_silences_it(data, extra):
    """
    Given the transit-only 1 RJ planet with the user's own pin or prior on
      its mass (the warning's own advice),
    Then nothing is flagged.
    """
    system = _system(data, extra=extra)
    assert system.planet.chen_only_mass_planets(system) == {}


def test_the_unflagged_twin_of_the_warned_planet_is_flagged(data):
    """The positive control for the silent cases: same builder, flagged."""
    system = _system(data)
    flagged = system.planet.chen_only_mass_planets(system)
    assert list(flagged) == [0]
    assert flagged[0] == pytest.approx(RJUP_TO_REARTH, rel=1e-6)
