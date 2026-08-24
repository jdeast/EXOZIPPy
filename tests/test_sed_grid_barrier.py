"""The BC grid's logg axis is a soft bound on ``star.loggsed``.

Review item 11.6.  ``bc_grid.RegularGridInterpolator`` is built with
``fill_value=None``, so it computes ``out_of_bounds`` and then DISCARDS it:
an off-grid point is linearly extrapolated off the edge cell, not NaN and
not penalized.  Three of the grid's four axes (teff, feh, av) are sampled
star parameters whose grid extent is injected as a hard bound, so they
cannot leave the grid at all.  The fourth, logg, is derived from logmass
and radiussed -- and bounding those two SEPARATELY does not bound their
COMBINATION, which is what logg is -- so until 2026-08 a fit could wander
off the grid and be scored by an extrapolation with no restoring force.

``star.loggsed`` is now a derived Parameter carrying the grid's logg extent
as ``lower``/``upper``, which ``Parameter.build_pymc`` turns into the house
soft barrier (``potentials.soft_lower_bound`` / ``soft_upper_bound``, with
the steepness measured at startup by ``whitening.measure_barrier_scales``).
The tests below pin the three things that matter: it is ~0 on grid, it
grows off grid, and the gradient stays finite in both regimes (a NaN
gradient would make the barrier WORSE than the extrapolation it replaces).
"""

import logging
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytensor.tensor as pt
import pytest
import yaml

from exozippy.components.parameter import Parameter
from exozippy.components.sed.bc_grid import DEFAULT_MODEL_ROOT, peek_grid_axes
from exozippy.system import System

_HAT3_DIR = Path(__file__).parent.parent / "examples" / "hat3"

_MINIMAL_SED_YAML = """
model: NextGen
nstars: 1
filters:
  - name: "2MASS/2MASS.J"
    mag: 10.0
    err: 0.02
"""


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def hat3_star_only():
    """Given the hat3 star-only example (one star, a NextGen SED and no
    photometric time series), when it is prepared and built, provide
    (system, model, raw start point).  It is the cheapest shipped config
    that exercises the BC interpolator end to end."""
    if not _HAT3_DIR.is_dir():
        pytest.skip("examples/hat3 not present")

    cwd = os.getcwd()
    os.chdir(_HAT3_DIR)
    try:
        config = yaml.safe_load(Path("hat3_staronly.yaml").read_text())
        user_params = yaml.safe_load(
            Path(config["parameter_file"]).read_text()
        )
        for key in ("run", "prefix", "parameter_file", "sampler"):
            config.pop(key, None)
        system = System(config, user_params=user_params)
        system.prepare()
        model = system.build_model()
    finally:
        os.chdir(cwd)
    return system, model, system.get_raw_start(model)


def _barrier_value(model, point):
    """The summed loggsed barrier potentials, evaluated at ``point``.

    Reading the two named potentials rather than differencing the whole
    logp is what makes "negligible on grid" a statement about the barrier
    and not about float64 cancellation in an 1e5-nat total.
    """
    nodes = [
        p
        for p in model.potentials
        if p.name in ("low_bound.star.loggsed", "up_bound.star.loggsed")
    ]
    assert nodes, "no loggsed barrier potential was added"
    fn = model.compile_fn(
        model.replace_rvs_by_values(
            [pt.add(*nodes) if len(nodes) > 1 else nodes[0]]
        ),
        inputs=model.value_vars,
        on_unused_input="ignore",
    )
    return float(np.asarray(fn(point)[0]))


def _loggsed_value(system, model, point):
    fn = model.compile_fn(
        model.replace_rvs_by_values([pt.flatten(system.star.loggsed.value)]),
        inputs=model.value_vars,
        on_unused_input="ignore",
    )
    return np.asarray(fn(point)[0], dtype=float)


def _shifted(point, key, offset, index=0):
    out = {k: np.array(v, dtype=float).copy() for k, v in point.items()}
    out[key].flat[index] += offset
    return out


def _radiussed_key(point):
    for key in point:
        if "radiussed" in key:
            return key
    raise AssertionError(f"no radiussed raw variable in {sorted(point)}")


def _minimal_sed(tmp_path):
    """A SED component around a one-filter .sed file, with its own
    ConfigManager, so the override channel can be inspected."""
    from exozippy.components.sed.sed import SED
    from exozippy.config import ConfigManager

    sed_file = tmp_path / "test_star.sed"
    sed_file.write_text(_MINIMAL_SED_YAML)
    cm = ConfigManager({})
    config = {"file": str(sed_file), "model_root": str(DEFAULT_MODEL_ROOT)}
    return SED(config, cm), cm


def _fake_system(loggsed, logmass_init, radiussed_init, star_names=("A",)):
    """A stand-in for System carrying only what _declare_grid_support reads."""
    star = SimpleNamespace(
        names=list(star_names),
        loggsed=loggsed,
        logmass=SimpleNamespace(initval=np.atleast_1d(logmass_init)),
        radiussed=SimpleNamespace(initval=np.atleast_1d(radiussed_init)),
    )
    return SimpleNamespace(star=star, prose=None)


def _loggsed_parameter(bound_scale=None):
    return Parameter(
        label="star.loggsed",
        unit="dex(cm/s2)",
        internal_unit="dex(cm/s2)",
        lower=[0.0],
        upper=[5.0],
        bound_scale=bound_scale,
    )


def _expected_penalty(val, scale, edge=5.0):
    """The log-sigmoid barrier's penalty at ``val``, computed independently.

    ``potentials.soft_upper_bound`` with softness 0.01: the penalty is
    log(sigmoid((edge - val) * 4.4 / (scale * 0.01))).
    """
    arg = (edge - val) * 4.4 / (scale * 0.01)
    return -np.logaddexp(0.0, -arg)


# ---------------------------------------------------------------------------
# Section 1 -- the bound reaches star.loggsed at all
# ---------------------------------------------------------------------------


def test_grid_logg_extent_is_registered_on_the_override_channel(tmp_path):
    """
    Given a SED component built around the NextGen BC tree,
    When __init__ runs,
    Then 'star.loggsed' carries the grid's logg axis as lower/upper on the
    component override channel, alongside the three sampled axes.

    The override channel (not user_params) because these are a VALIDITY
    LIMIT of this component's grid; layered under the user, so an explicit
    params entry still wins.
    """
    # ARRANGE / ACT
    sed, cm = _minimal_sed(tmp_path)
    axes = peek_grid_axes(model="NextGen", model_root=DEFAULT_MODEL_ROOT)

    # ASSERT
    assert "star.loggsed" in cm.param_overrides
    entry = cm.param_overrides["star.loggsed"]
    assert entry["lower"] == pytest.approx(float(axes["logg_pts"].min()))
    assert entry["upper"] == pytest.approx(float(axes["logg_pts"].max()))
    assert "lower" not in (cm.user_params.get("star.loggsed") or {})


def test_the_interpolator_is_handed_the_bounded_node(hat3_star_only):
    """
    Given a built system with an SED,
    When the predicted-apparent-magnitude graph is inspected,
    Then star.loggsed's own value node is among its ancestors.

    The barrier restrains star.loggsed; if the interpolator were handed a
    second, independently recomputed logg expression (which is what
    _predicted_appmag_node did until 2026-08) the barrier would constrain
    a quantity the likelihood does not use.
    """
    # ARRANGE
    system, model, _ = hat3_star_only
    from pytensor.graph.traversal import ancestors

    # ACT
    node = system.sed._predicted_appmag_node(system)
    anc = set(ancestors([node]))

    # ASSERT
    assert system.star.loggsed.value in anc


def test_both_barrier_potentials_exist(hat3_star_only):
    """
    Given a built system with an SED,
    When the model's potentials are listed,
    Then both the lower and upper loggsed barriers are present.
    """
    # ARRANGE
    _, model, _ = hat3_star_only

    # ACT
    names = {p.name for p in model.potentials}

    # ASSERT
    assert "low_bound.star.loggsed" in names
    assert "up_bound.star.loggsed" in names


# ---------------------------------------------------------------------------
# Section 2 -- shape of the penalty
# ---------------------------------------------------------------------------


def test_barrier_is_negligible_on_grid(hat3_star_only):
    """
    Given the hat3 star-only example, whose start (logg ~ 4.54) is well
    inside the NextGen logg axis [0, 5],
    When the loggsed barrier potentials are evaluated at that start,
    Then their sum is zero to numerical precision.

    This is the on-grid no-op guarantee: adding the barrier must not move a
    fit that never leaves the grid.
    """
    # ARRANGE
    system, model, point = hat3_star_only
    logg = _loggsed_value(system, model, point)

    # ACT
    penalty = _barrier_value(model, point)

    # ASSERT
    assert 0.0 < logg.min() and logg.max() < 5.0
    assert abs(penalty) < 1e-12


def test_barrier_grows_as_the_fit_leaves_the_grid(hat3_star_only):
    """
    Given the hat3 star-only example,
    When radiussed is stepped far enough in raw space to drive loggsed past
    the grid's logg ceiling,
    Then the barrier penalty is ~0 while on grid and grows monotonically
    once off it.
    """
    # ARRANGE
    system, model, point = hat3_star_only
    key = _radiussed_key(point)
    # Smaller radius -> larger logg. Offsets are raw (whitened) units.
    offsets = [0.0, -20.0, -40.0, -60.0, -80.0]

    # ACT
    loggs, penalties = [], []
    for off in offsets:
        moved = _shifted(point, key, off)
        loggs.append(float(_loggsed_value(system, model, moved).max()))
        penalties.append(_barrier_value(model, moved))

    # ASSERT -- the sweep really does cross the ceiling
    assert loggs[0] < 5.0 < loggs[-1]
    assert sorted(loggs) == loggs
    on_grid = [p for lg, p in zip(loggs, penalties) if lg < 5.0]
    off_grid = [p for lg, p in zip(loggs, penalties) if lg > 5.0]
    assert off_grid, "the sweep never left the grid"
    assert all(abs(p) < 1e-6 for p in on_grid)
    # Strictly decreasing (more negative) once outside.
    assert all(b < a for a, b in zip(off_grid, off_grid[1:]))
    assert off_grid[-1] < -1.0


def test_gradient_is_finite_on_and_off_grid(hat3_star_only):
    """
    Given the hat3 star-only example,
    When dlogp is evaluated at the on-grid start and at points well past
    the grid's logg ceiling,
    Then every component of the gradient is finite.

    A -inf wall or a NaN gradient off the grid would be WORSE than the
    silent extrapolation it replaces: NUTS has nothing to follow back.
    """
    # ARRANGE
    system, model, point = hat3_star_only
    key = _radiussed_key(point)
    dlogp = model.compile_dlogp()

    # ACT / ASSERT
    for off in (0.0, -40.0, -80.0, -200.0):
        moved = _shifted(point, key, off)
        grad = dlogp(moved)
        flat = np.concatenate(
            [np.atleast_1d(np.asarray(g, dtype=float)).ravel() for g in grad]
            if isinstance(grad, (list, tuple))
            else [np.atleast_1d(np.asarray(grad, dtype=float)).ravel()]
        )
        assert np.all(np.isfinite(flat)), f"non-finite gradient at raw {off}"


def test_off_grid_logp_is_finite(hat3_star_only):
    """
    Given the hat3 star-only example,
    When logp is evaluated well past the grid's logg ceiling,
    Then it is finite (the barrier is a penalty, not a wall).
    """
    # ARRANGE
    system, model, point = hat3_star_only
    key = _radiussed_key(point)
    logp = model.compile_logp()

    # ACT
    values = [
        float(logp(_shifted(point, key, off))) for off in (-80.0, -200.0)
    ]

    # ASSERT
    assert all(np.isfinite(v) for v in values)


# ---------------------------------------------------------------------------
# Section 3 -- reporting
# ---------------------------------------------------------------------------


def test_prior_contribution_is_declared_and_reported(hat3_star_only):
    """
    Given a built system with an SED,
    When star.loggsed's Prior column is rendered,
    Then it names the BC grid and its support, and never says "Uniform".

    A derived parameter with two finite bounds reports "U(lo, hi)" from its
    own fields -- exactly the prior a barrier is NOT -- so the declaration
    is what keeps the table honest.  See "Reporting component-added priors"
    in src/exozippy/components/parameter.md.
    """
    # ARRANGE
    system, _, _ = hat3_star_only
    loggsed = system.star.loggsed

    # ACT
    text = loggsed.get_prior_str(0, latex=False)
    latex = loggsed.get_prior_str(0, latex=True)
    cell, notes = loggsed.prior_cell_and_notes(0)

    # ASSERT
    assert loggsed.prior_contributions, "no contribution was declared"
    for rendered in (text, latex):
        assert "NextGen" in rendered
        assert "bound" in rendered
        assert "Uniform" not in rendered and "U(" not in rendered
        assert "[0, 5]" in rendered.replace("$", "")
    assert cell == ""
    assert len(notes) == 1
    # A barrier is not a normalized density over the interval, and the note
    # must not claim it is.
    assert "normalized" not in notes[0]
    assert "logg support" in notes[0]


def test_declaration_is_idempotent(tmp_path):
    """
    Given a SED component and a loggsed Parameter,
    When _declare_grid_support runs twice (a second build_model() on one
    System, as the GUI does),
    Then only one prior contribution is recorded.
    """
    # ARRANGE
    sed, _ = _minimal_sed(tmp_path)
    loggsed = _loggsed_parameter()
    system = _fake_system(loggsed, 0.0, 1.0)

    # ACT
    sed._declare_grid_support(system)
    sed._declare_grid_support(system)

    # ASSERT
    assert len(loggsed.prior_contributions) == 1


# ---------------------------------------------------------------------------
# Section 4 -- the off-grid START warning
# ---------------------------------------------------------------------------


def test_off_grid_start_warns_and_names_the_star(tmp_path, caplog):
    """
    Given a star whose start (logmass, radiussed) put loggsed above the
    NextGen grid's logg ceiling,
    When _declare_grid_support runs,
    Then a warning names the star, the value and the grid range.

    A warning and not a raise: the interpolator extrapolates rather than
    failing there, so such fits have been running, and the barrier pulls
    the chain back by itself.  examples/gj1214 is the shipped case (its
    M dwarf starts at loggsed = 5.02).
    """
    # ARRANGE -- 0.178 solMass, 0.215 solRad is GJ 1214: logg ~ 5.03
    sed, _ = _minimal_sed(tmp_path)
    loggsed = _loggsed_parameter()
    system = _fake_system(loggsed, np.log10(0.178), 0.215, star_names=("A",))

    # ACT
    with caplog.at_level(logging.WARNING):
        sed._declare_grid_support(system)

    # ASSERT
    messages = [r.getMessage() for r in caplog.records]
    hits = [m for m in messages if "loggsed" in m and "OUTSIDE" in m]
    assert len(hits) == 1, messages
    assert "star 'A'" in hits[0]
    assert "5.0" in hits[0]
    assert "bound_scale" in hits[0]


def test_on_grid_start_is_silent(tmp_path, caplog):
    """
    Given a solar-analogue start (logg ~ 4.44), comfortably inside the grid,
    When _declare_grid_support runs,
    Then nothing is warned.
    """
    # ARRANGE
    sed, _ = _minimal_sed(tmp_path)
    loggsed = _loggsed_parameter()
    system = _fake_system(loggsed, 0.0, 1.0)

    # ACT
    with caplog.at_level(logging.WARNING):
        sed._declare_grid_support(system)

    # ASSERT
    assert not [r for r in caplog.records if "OUTSIDE" in r.getMessage()]


# ---------------------------------------------------------------------------
# Section 5 -- deliberately softening the barrier (examples/gj1214)
# ---------------------------------------------------------------------------


def test_softening_the_barrier_does_not_silence_the_warning(tmp_path, caplog):
    """
    Given an off-grid star whose loggsed barrier the user has widened with
    bound_scale,
    When _declare_grid_support runs,
    Then the warning still fires, and its text describes the WIDENED
    barrier rather than the measured one.

    The notice keys on the VALUE being off the grid, never on the bound
    being active -- it is the "as long as we know the caveats" half of
    shipping an extrapolated fit, and softening the barrier is exactly the
    moment it must not go quiet.  A message still claiming the start is
    "effectively clamped to the edge" would be the stale advice that
    teaches people to stop reading warnings.
    """
    # ARRANGE
    sed, _ = _minimal_sed(tmp_path)
    loggsed = _loggsed_parameter(bound_scale=25.0)
    system = _fake_system(loggsed, np.log10(0.178), 0.215)

    # ACT
    with caplog.at_level(logging.WARNING):
        sed._declare_grid_support(system)

    # ASSERT
    hits = [
        r.getMessage()
        for r in caplog.records
        if "loggsed" in r.getMessage() and "OUTSIDE" in r.getMessage()
    ]
    assert len(hits) == 1, [r.getMessage() for r in caplog.records]
    assert "WIDENED" in hits[0]
    assert "bound_scale = 25" in hits[0]
    assert "0.25 dex" in hits[0]  # transition width = 0.01 * bound_scale
    assert "17.6 nats per dex" in hits[0]
    assert "clamped to the edge" not in hits[0]


@pytest.mark.parametrize(
    "logg,scale",
    [(5.0223, 25.0), (5.25, 25.0), (5.5, 25.0), (5.0223, 50.0)],
)
def test_reported_penalty_matches_the_barrier_formula(logg, scale):
    """
    Given an off-grid loggsed and a bound_scale,
    When the warning's penalty figure is built,
    Then it equals log(sigmoid(...)) computed independently.

    The number is the whole point of the message -- it is what tells a user
    whether their softening admits the star cheaply or has effectively
    switched the barrier off -- so it must not be a hand-maintained
    approximation of the potential actually added.
    """
    # ARRANGE
    from exozippy.components.sed.sed import SED

    expected = -_expected_penalty(logg, scale)

    # ACT
    message = SED._barrier_advice("A", logg, 0.0, 5.0, scale)

    # ASSERT
    assert f"{expected:.2f} nats" in message


def test_gj1214_example_ships_the_softened_barrier():
    """
    Given the shipped gj1214 example, whose M dwarf starts at loggsed =
    5.022 -- past NextGen's 5.0 logg ceiling,
    Then its params file widens the loggsed barrier and says why.

    Not a style check: with no M-dwarf-capable BC grid shipping, the
    measured barrier would clamp this fit to the grid edge at a cost of
    ~434 nats, and the repo owner's ruling is that a modestly extrapolated
    bolometric correction beats refusing to fit -- provided the caveat is
    written where the person running the example will see it.
    """
    # ARRANGE
    params_file = (
        Path(__file__).parent.parent
        / "examples"
        / "gj1214"
        / "gj1214.params.yaml"
    )
    if not params_file.is_file():
        pytest.skip("examples/gj1214 not present")
    text = params_file.read_text()

    # ACT
    params = yaml.safe_load(text)

    # ASSERT -- the softening itself
    entry = params.get("star.A.loggsed")
    assert entry is not None, "gj1214 no longer softens the loggsed barrier"
    assert entry.get("bound_scale") == 25.0
    # ...and the caveat, in the file, in the words that matter
    assert "EXTRAPOLATED" in text
    assert "NextGen" in text
