"""Blind global period searches (components/globalsearch.py).

BLS for transit photometry, Lomb-Scargle for radial velocities, and the
ranked-hint channel that turns a detection into a start value so a
params.yaml becomes optional for a single-planet blind fit.

The strongest tests here are inject-and-recover: data are generated from a
known period, epoch and amplitude, and the search has to get them back well
enough to seed a fit.  The end-to-end ones then build the real model and
compare the seeded start's logp against the unseeded one -- which is what
pins the CONJUNCTION SIGN, since a half-period phase error would make the
seeded start worse rather than better.
"""

import logging

import numpy as np
import pytest

from exozippy.components import globalsearch
from exozippy.config import RANK_DERIVED_DATA
from exozippy.system import System

# --- synthetic data -----------------------------------------------------------

TRANSIT_TRUTH = dict(
    period=3.14159, epoch=2459000.3, depth=0.01, duration=0.09
)
RV_TRUTH = dict(period=12.3456, epoch=2459005.7, amplitude=25.0)


def make_transit(n=1500, span=27.0, sigma=8e-4, seed=7, baseline=1.0):
    """A boxcar transit light curve at TRANSIT_TRUTH, in relative flux."""
    rng = np.random.default_rng(seed)
    t = 2459000.0 + np.sort(rng.uniform(0.0, span, n))
    p, t0 = TRANSIT_TRUTH["period"], TRANSIT_TRUTH["epoch"]
    phase = (t - t0 + 0.5 * p) % p - 0.5 * p
    flux = np.ones_like(t)
    flux[np.abs(phase) < 0.5 * TRANSIT_TRUTH["duration"]] -= TRANSIT_TRUTH[
        "depth"
    ]
    flux = baseline * (flux + rng.normal(0.0, sigma, n))
    return t, flux, np.full_like(t, baseline * sigma)


def make_rvs(n=120, span=300.0, sigma=3.0, seed=11, gamma=0.0):
    """A circular-orbit RV curve at RV_TRUTH, in m/s.

    The sign convention is EXOZIPPy's own: for a circular orbit the star's
    velocity is ``-K sin(2 pi (t - T_C) / P)`` about the systemic value, so
    it is falling through gamma at conjunction (the star recedes before and
    approaches after).  ``test_conjunction_convention_matches_the_orbit_model``
    checks that against the component's own Kepler chain rather than trusting
    this generator.
    """
    rng = np.random.default_rng(seed)
    t = 2459000.0 + np.sort(rng.uniform(0.0, span, n))
    p, tc, k = RV_TRUTH["period"], RV_TRUTH["epoch"], RV_TRUTH["amplitude"]
    rv = (
        gamma
        - k * np.sin(2.0 * np.pi * (t - tc) / p)
        + rng.normal(0.0, sigma, n)
    )
    return t, rv, np.full_like(t, sigma)


def nearest_cycle(epoch, period, times):
    """The injected epoch moved to the cycle the search reports it in."""
    return epoch + period * np.round((np.median(times) - epoch) / period)


def write_columns(path, *columns):
    np.savetxt(str(path), np.column_stack(columns))
    return str(path)


# --- unit: the searches -------------------------------------------------------


def test_bls_recovers_an_injected_transit():
    """Given a boxcar transit of known period, epoch and depth,
    when BLS searches the light curve,
    then all three come back close enough to start a fit."""
    # Arrange
    t, flux, err = make_transit()

    # Act
    signal = globalsearch.bls_search(t, flux, err)

    # Assert
    assert signal is not None
    assert signal.period == pytest.approx(TRANSIT_TRUTH["period"], rel=1e-3)
    assert signal.epoch == pytest.approx(
        nearest_cycle(TRANSIT_TRUTH["epoch"], signal.period, t), abs=0.01
    )
    assert signal.depth == pytest.approx(TRANSIT_TRUTH["depth"], rel=0.15)
    assert signal.significance >= globalsearch.MIN_BLS_SDE
    assert signal.detail["depth_snr"] >= globalsearch.MIN_BLS_DEPTH_SNR


def test_lombscargle_recovers_an_injected_orbit():
    """Given a circular RV signal of known period, epoch and semi-amplitude,
    when Lomb-Scargle searches the velocities,
    then all three come back close enough to start a fit."""
    # Arrange
    t, rv, err = make_rvs()

    # Act
    signal = globalsearch.lombscargle_search(t, rv, err)

    # Assert
    assert signal is not None
    assert signal.period == pytest.approx(RV_TRUTH["period"], rel=1e-3)
    assert signal.epoch == pytest.approx(
        nearest_cycle(RV_TRUTH["epoch"], signal.period, t),
        abs=0.02 * signal.period,
    )
    assert signal.amplitude == pytest.approx(RV_TRUTH["amplitude"], rel=0.05)
    assert signal.detail["fap"] < globalsearch.MAX_LS_FAP


def test_lombscargle_removes_per_instrument_offsets():
    """Given two RV sets of the same orbit at very different systemic offsets,
    when the offsets are removed and the concatenation is searched,
    then the period and semi-amplitude survive the concatenation."""
    # Arrange
    t1, rv1, e1 = make_rvs(n=80, seed=1, gamma=-30000.0)
    t2, rv2, e2 = make_rvs(n=80, seed=2, gamma=+12.0)
    t = np.concatenate([t1, t2])
    rv = np.concatenate([rv1 - rv1.mean(), rv2 - rv2.mean()])
    err = np.concatenate([e1, e2])
    inst = np.concatenate([np.zeros(t1.size, int), np.ones(t2.size, int)])
    order = np.argsort(t)

    # Act
    signal = globalsearch.lombscargle_search(
        t[order], rv[order], err[order], inst_map=inst[order]
    )

    # Assert
    assert signal is not None
    assert signal.period == pytest.approx(RV_TRUTH["period"], rel=1e-3)
    assert signal.amplitude == pytest.approx(RV_TRUTH["amplitude"], rel=0.05)


def test_conjunction_convention_matches_the_orbit_model():
    """Given an RV curve built from the orbit component's own Kepler chain,
    when Lomb-Scargle reports its time of conjunction,
    then the epoch agrees with the T_C that curve was built from.

    This is the check that a sign slip in ``_conjunction_from_sinusoid``
    cannot pass: a half-period error is the natural failure mode and it is
    an order of magnitude larger than the tolerance here.  Several omega are
    used because a circular orbit's conjunction must not depend on it.
    """
    # Arrange
    import pytensor.tensor as pt
    from exoplanet_core.pymc import ops

    from exozippy.components.orbit import physics as orbit_physics

    period, tc, k = 9.0, 2459010.25, 30.0
    n = 2.0 * np.pi / period
    t = 2459000.0 + np.linspace(0.0, 120.0, 400)
    rng = np.random.default_rng(5)

    for omega in (0.0, 0.5 * np.pi, np.pi, 1.7 * np.pi):
        tp = orbit_physics.calc_tp_from_ecc(
            pt.as_tensor_variable(np.float64(0.0)),
            pt.as_tensor_variable(np.float64(omega)),
            pt.as_tensor_variable(np.float64(tc)),
            pt.as_tensor_variable(np.float64(n)),
        )
        mean_anomaly = (pt.as_tensor_variable(t) - tp) * n
        sinf, cosf = ops.kepler(mean_anomaly, pt.zeros_like(mean_anomaly))
        # The RV formula of Orbit.get_radial_velocity, at ecc = 0.
        rv = (k * (np.cos(omega) * cosf - np.sin(omega) * sinf)).eval()
        rv = rv + rng.normal(0.0, 1.0, t.size)

        # Act
        signal = globalsearch.lombscargle_search(
            t, rv, np.full_like(t, 1.0), minimum_period=2.0
        )

        # Assert
        assert signal is not None, f"no detection at omega = {omega}"
        assert signal.epoch == pytest.approx(
            nearest_cycle(tc, signal.period, t), abs=0.02 * period
        ), f"conjunction epoch is wrong at omega = {omega}"


def test_searches_reject_pure_noise(caplog):
    """Given data with no signal in them at all,
    when either search runs,
    then it returns None and says what it looked at."""
    # Arrange
    rng = np.random.default_rng(3)
    t, _, err = make_transit()
    trv, _, erv = make_rvs()

    # Act
    with caplog.at_level(logging.WARNING):
        bls = globalsearch.bls_search(
            t, 1.0 + rng.normal(0.0, 8e-4, t.size), err
        )
        ls = globalsearch.lombscargle_search(
            trv, rng.normal(0.0, 3.0, trv.size), erv
        )

    # Assert
    assert bls is None and ls is None
    assert "no convincing transit" in caplog.text
    assert "no convincing periodicity" in caplog.text


def test_short_series_are_refused_rather_than_searched(caplog):
    """Given far too few points for the algorithm,
    when the search runs,
    then it declines instead of reporting a peak."""
    # Arrange
    t = np.linspace(2459000.0, 2459010.0, 8)

    # Act
    with caplog.at_level(logging.WARNING):
        bls = globalsearch.bls_search(t, np.ones_like(t), None)
        ls = globalsearch.lombscargle_search(t[:5], np.zeros(5), None)

    # Assert
    assert bls is None and ls is None
    assert "fewer than" in caplog.text


def test_fold_epoch_moves_whole_periods_only():
    """Given an epoch far from the data,
    when it is folded toward a reference time,
    then it moves by an integer number of periods."""
    # Arrange / Act
    folded = globalsearch.fold_epoch(100.0, 3.0, 1000.0)

    # Assert
    assert (folded - 100.0) % 3.0 == pytest.approx(0.0, abs=1e-9)
    assert abs(folded - 1000.0) <= 1.5


# --- configs used by the end-to-end tests -------------------------------------


def transit_config(tmp_path, **orbit_keys):
    write_columns(tmp_path / "blind.dat", *make_transit())
    return {
        "prefix": str(tmp_path / "out"),
        "star": [{"name": "A", "mist": False}],
        "planet": [{"name": "b"}],
        "orbit": [
            dict(name="b", primary=["A"], companion=["b"], **orbit_keys)
        ],
        "band": [{"name": "TESS", "filter": "TESS"}],
        "transit": [
            {"name": "LC", "file": str(tmp_path / "blind.dat"), "band": "TESS"}
        ],
    }


def rv_config(tmp_path, **orbit_keys):
    write_columns(tmp_path / "blind.rv", *make_rvs())
    return {
        "prefix": str(tmp_path / "out"),
        "star": [{"name": "A", "mist": False}],
        "planet": [{"name": "b"}],
        "orbit": [
            dict(name="b", primary=["A"], companion=["b"], **orbit_keys)
        ],
        "rvinstrument": [
            {"name": "HIRES", "file": str(tmp_path / "blind.rv")}
        ],
    }


def prepared(config, user_params=None):
    system = System(config, user_params=user_params or {})
    system.prepare()
    return system


def initval(system, comp, param):
    return float(
        np.atleast_1d(
            system.config_manager.resolve(comp, param, shape=(1,))["initval"]
        )[0]
    )


# --- end to end: a fit with no params file ------------------------------------


def test_blind_transit_fit_seeds_period_epoch_and_radius_ratio(tmp_path):
    """Given a transit config and an EMPTY params file,
    when the system is prepared,
    then the period, conjunction time and radius ratio come from the data."""
    # Arrange
    config = transit_config(tmp_path)
    t = make_transit()[0]

    # Act
    system = prepared(config)

    # Assert
    period = initval(system, "orbit", "period")
    assert period == pytest.approx(TRANSIT_TRUTH["period"], rel=1e-3)
    assert initval(system, "orbit", "logP") == pytest.approx(
        np.log10(period), rel=1e-9
    )
    assert initval(system, "orbit", "tc") == pytest.approx(
        nearest_cycle(TRANSIT_TRUTH["epoch"], period, t), abs=0.02
    )
    assert initval(system, "planet", "p") == pytest.approx(
        np.sqrt(TRANSIT_TRUTH["depth"]), rel=0.1
    )
    # The provenance ledger has to say these came from the data, not from
    # defaults.yaml -- that is what add_hint buys over the override channel.
    cm = system.config_manager
    assert cm.initval_source("orbit", "period", 0) == "data"
    assert cm.initval_source("orbit", "tc", 0) == "data"


def test_the_seeded_tc_window_is_the_seeded_period(tmp_path):
    """Given a blindly seeded transit fit,
    when the orbit declares tc's hard window at stage 2,
    then it is the SEARCHED epoch +/- half the SEARCHED period.

    The stage-1a placement exists for exactly this: a seed pushed at stage 2
    would race the orbit component, and a window built around the
    defaults.yaml 2460000 would make the searched epoch unreachable (and
    fatal, per Parameter.build_pymc)."""
    # Arrange
    config = transit_config(tmp_path)

    # Act
    system = prepared(config)

    # Assert
    tc = initval(system, "orbit", "tc")
    period = initval(system, "orbit", "period")
    window = system.orbit.manifest["tc"]
    assert float(np.atleast_1d(window["lower"])[0]) == pytest.approx(
        tc - 0.5 * period
    )
    assert float(np.atleast_1d(window["upper"])[0]) == pytest.approx(
        tc + 0.5 * period
    )


def test_blind_rv_fit_seeds_period_epoch_and_semi_amplitude(tmp_path):
    """Given an RV config and an EMPTY params file,
    when the system is prepared,
    then the period and conjunction time come from the data, and the
    Lomb-Scargle sinusoid's amplitude replaces the scatter-based K seed."""
    # Arrange
    config = rv_config(tmp_path)
    t = make_rvs()[0]

    # Act
    system = prepared(config)

    # Assert
    period = initval(system, "orbit", "period")
    assert period == pytest.approx(RV_TRUTH["period"], rel=1e-3)
    assert initval(system, "orbit", "tc") == pytest.approx(
        nearest_cycle(RV_TRUTH["epoch"], period, t), abs=0.02 * period
    )
    assert system.rvinstrument.k_init == pytest.approx(
        RV_TRUTH["amplitude"], rel=0.05
    )
    assert initval(system, "planet", "K") == pytest.approx(
        RV_TRUTH["amplitude"], rel=0.05
    )


def test_the_seeded_start_beats_the_unseeded_one(tmp_path):
    """Given the same RV data with and without the global search,
    when both models are built,
    then the seeded start's log-probability is enormously better.

    This is the end-to-end pin on the conjunction convention: a half-period
    phase error would leave the seeded start WORSE than the default one, so
    the sign cannot be wrong and pass."""
    # Arrange
    seeded = prepared(rv_config(tmp_path))
    blind = prepared(rv_config(tmp_path, global_search=False))

    # Act
    model_seeded = seeded.build_model()
    lp_seeded = float(
        model_seeded.compile_logp()(model_seeded.initial_point())
    )
    model_blind = blind.build_model()
    lp_blind = float(model_blind.compile_logp()(model_blind.initial_point()))

    # Assert
    assert np.isfinite(lp_seeded)
    assert lp_seeded > lp_blind + 100.0


def test_the_seed_is_pushed_through_the_hint_channel_alone(tmp_path):
    """Given a blindly seeded transit fit,
    when the searched period and epoch are pushed,
    then they reach the config manager through add_hint ONLY -- nothing is
    written a second time through the override channel.

    Until review 3.14.3 this module wrote every seed twice, because
    ConfigManager.resolve() did not layer self.hints and the stage-2 reader
    that builds tc's window could not otherwise see it. resolve() layers
    hints now (tests/test_hint_resolution.py), so the duplicate is gone. It
    must not come back: an override carries no rank, so a second write is
    invisible to the provenance ledger and can only ever drift from the hint
    it shadows."""
    # Arrange
    config = transit_config(tmp_path)

    # Act
    system = prepared(config)
    cm = system.config_manager

    # Assert -- the ranked channel carries them ...
    assert "orbit.0.tc" in cm.hints
    assert cm.hint_ranks["orbit.0.tc"] == RANK_DERIVED_DATA
    assert "orbit.0.logP" in cm.hints or "orbit.0.period" in cm.hints
    # ... and the unranked one carries nothing for this orbit.
    assert not [k for k in cm.param_overrides if k.startswith("orbit.")]


# --- end to end: precedence ---------------------------------------------------


def test_a_user_period_is_never_overridden(tmp_path):
    """Given a params file that names the period but not the epoch,
    when the search runs,
    then the user's period stands and only the epoch is seeded."""
    # Arrange
    config = transit_config(tmp_path)
    user = {"orbit.b.period": {"initval": 5.0}}

    # Act
    system = prepared(config, user_params=user)

    # Assert
    assert initval(system, "orbit", "period") == pytest.approx(5.0)
    assert initval(system, "orbit", "tc") != pytest.approx(2460000.0)
    assert system.config_manager.initval_source("orbit", "tc", 0) == "data"


@pytest.mark.parametrize("transit_first", [True, False])
def test_the_transit_period_beats_the_rv_period(tmp_path, transit_first):
    """Given a system with both photometry and velocities of one orbit,
    when both searches run,
    then the orbit is seeded from BLS whatever order the config lists them in.

    Order independence is the point: add_hint is last-writer-wins and
    System.prepare walks the components in config key order."""
    # Arrange
    config = transit_config(tmp_path)
    write_columns(tmp_path / "blind.rv", *make_rvs())
    rv_block = [{"name": "HIRES", "file": str(tmp_path / "blind.rv")}]
    if transit_first:
        config["rvinstrument"] = rv_block
    else:
        reordered = {k: v for k, v in config.items() if k != "transit"}
        reordered["rvinstrument"] = rv_block
        reordered["transit"] = config["transit"]
        config = reordered

    # Act
    system = prepared(config)

    # Assert -- the transit period, not the (very different) RV one.
    assert initval(system, "orbit", "period") == pytest.approx(
        TRANSIT_TRUTH["period"], rel=1e-3
    )
    seeds = system.config_manager._global_search_seeds
    assert "BLS" in seeds["orbit.0.period"][2]
    assert "BLS" in seeds["orbit.0.tc"][2]


# --- end to end: when the search must not run ---------------------------------


def test_global_search_false_opts_out(tmp_path):
    """Given global_search: false on the orbit,
    when the system is prepared,
    then no search runs and the defaults.yaml starts stand."""
    # Arrange
    config = transit_config(tmp_path, global_search=False)

    # Act
    system = prepared(config)

    # Assert
    assert system.transit.bls_signal is None
    assert initval(system, "orbit", "logP") == pytest.approx(1.0)
    assert initval(system, "orbit", "tc") == pytest.approx(2460000.0)


def test_a_multi_orbit_system_is_left_alone(tmp_path, caplog):
    """Given more than one orbit,
    when the search would run,
    then it declines: a periodogram peak names no orbit."""
    # Arrange
    config = transit_config(tmp_path)
    config["planet"].append({"name": "c", "orbit_ndx": 1})
    config["orbit"].append({"name": "c", "primary": ["A"], "companion": ["c"]})

    # Act
    with caplog.at_level(logging.WARNING):
        system = prepared(config)

    # Assert
    assert system.transit.bls_signal is None
    assert "carries no statement about WHICH orbit" in caplog.text
    assert initval(system, "orbit", "logP") == pytest.approx(1.0)


def test_a_supplied_period_and_epoch_skip_the_search(tmp_path):
    """Given a params file that supplies both required starts,
    when the system is prepared,
    then no search runs at all (the literal-name short circuit)."""
    # Arrange
    config = transit_config(tmp_path)
    user = {
        "orbit.b.period": {"initval": 5.0},
        "orbit.b.tc": {"initval": 2459001.0},
    }

    # Act
    system = prepared(config, user_params=user)

    # Assert
    assert system.transit.bls_signal is None
    assert initval(system, "orbit", "period") == pytest.approx(5.0)
    assert initval(system, "orbit", "tc") == pytest.approx(2459001.0)


# --- the astrometry placeholder stays a placeholder ---------------------------


def test_astrometry_declares_no_global_search():
    """Given the astrometry component,
    when its utilities are enumerated,
    then the period search is declared and explicitly unavailable."""
    # Arrange
    from exozippy.components.astrometryinstrument.astrometryinstrument import (
        AstrometryInstrument,
    )

    # Act
    specs = {s.name: s for s in AstrometryInstrument.get_utilities()}

    # Assert
    assert specs["astrometry_period_search"].available is False
    assert (
        "different algorithms" in specs["astrometry_period_search"].description
    )
