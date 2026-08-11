"""Pre-whitening seed polish (polish.py) and its start re-anchoring.

The polish promotes a solution-estimate start to its basin's optimum BEFORE
the whitening probe measures scales around it (a start far off its optimum
makes the probe gradient-dominated).  These tests pin: the seed-provenance
gate, the L-BFGS engine, the gradient-free DE fallback dispatch, the
adoption of polished starts (raw_initval / initval / seed_resolved), and
set_whitening keeping a nonzero raw start pinned to the same physical point
through a rescale.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest
from pytensor.graph.op import Op

from exozippy.components.parameter import Parameter
from exozippy.polish import (
    DEFAULT_POLISH_STEPS,
    polish_raw_starts,
    resolve_polish_steps,
)
from exozippy.system import System

# ---------------------------------------------------------------------------
# resolve_polish_steps: the seed-provenance gate
# ---------------------------------------------------------------------------


def test_gate_auto_polishes_single_start_and_hint_sets_only():
    """
    Given the default 'auto' setting,
    When the seeds are a single canonical start or MMEXOFAST hint sets,
    Then polish runs; a multi-seed set WITHOUT hints (posterior-draw
      restart) is never polished -- polishing K draws per basin would
      collapse the restart's overdispersion.
    """
    assert (
        resolve_polish_steps("auto", n_seeds=1, has_seed_hints=False)
        == DEFAULT_POLISH_STEPS
    )
    assert (
        resolve_polish_steps("auto", n_seeds=3, has_seed_hints=True)
        == DEFAULT_POLISH_STEPS
    )
    assert resolve_polish_steps("auto", n_seeds=3, has_seed_hints=False) == 0


def test_gate_overrides():
    """
    Given explicit on/off/int settings,
    When resolve_polish_steps maps them,
    Then they override the provenance logic entirely.
    """
    assert (
        resolve_polish_steps("on", n_seeds=5, has_seed_hints=False)
        == DEFAULT_POLISH_STEPS
    )
    assert resolve_polish_steps(False, n_seeds=1, has_seed_hints=True) == 0
    assert resolve_polish_steps("off", n_seeds=1, has_seed_hints=True) == 0
    assert resolve_polish_steps(42, n_seeds=1, has_seed_hints=False) == 42


# ---------------------------------------------------------------------------
# engines
# ---------------------------------------------------------------------------


def test_lbfgs_polish_climbs_to_the_mode():
    """
    Given a differentiable model started 40 sigma below its mode,
    When polish_raw_starts runs,
    Then the L-BFGS engine is chosen and the polished start lands at the
      mode (lp gain ~0.5*40^2 = 800 nats).
    """
    # ARRANGE
    with pm.Model() as model:
        x = pm.Flat("x")
        pm.Potential("like", -0.5 * ((x - 40.0) / 1.0) ** 2)
    start = {"x": np.array(0.0)}

    # ACT
    polished, dlps, method = polish_raw_starts(model, [start])

    # ASSERT
    assert method == "lbfgs"
    assert polished[0]["x"] == pytest.approx(40.0, abs=0.1)
    assert dlps[0] == pytest.approx(800.0, rel=0.01)


class _NoGradSquare(Op):
    """-(x - 3)^2 / 2 with NO gradient implementation."""

    itypes = [pt.dscalar]
    otypes = [pt.dscalar]

    def perform(self, node, inputs, outputs):
        (x,) = inputs
        outputs[0][0] = np.asarray(-0.5 * (x - 3.0) ** 2)


def test_gradient_free_model_falls_back_to_de():
    """
    Given a model whose logp contains an Op with no analytic gradient (the
      binary-lens magnification situation),
    When polish_raw_starts runs,
    Then the DE engine is dispatched and still improves the start.
    """
    # ARRANGE
    with pm.Model() as model:
        x = pm.Flat("x")
        pm.Potential("like", _NoGradSquare()(x))
    start = {"x": np.array(0.0)}

    # ACT
    polished, dlps, method = polish_raw_starts(
        model, [start], n_steps=200, rng=np.random.default_rng(3)
    )

    # ASSERT
    assert method == "de"
    assert dlps[0] > 0
    assert abs(polished[0]["x"] - 3.0) < 1.0


def test_polish_never_returns_a_worse_seed():
    """
    Given a start already exactly at its mode,
    When polish_raw_starts runs,
    Then the returned start is not worse than the input (dlp >= 0) and
      stays at the mode.
    """
    with pm.Model() as model:
        x = pm.Flat("x")
        pm.Potential("like", -0.5 * x**2)
    start = {"x": np.array(0.0)}

    polished, dlps, method = polish_raw_starts(model, [start])

    assert dlps[0] >= 0.0
    assert polished[0]["x"] == pytest.approx(0.0, abs=1e-3)


# ---------------------------------------------------------------------------
# adoption + rescale invariance
# ---------------------------------------------------------------------------


def _toy_param_model():
    """One bounded parameter whose likelihood mode (7.5) is far from its
    initval (2.0) relative to the 0.01-wide likelihood."""
    p = Parameter(label="toy.x", initval=2.0, lower=0.0, upper=10.0)
    with pm.Model() as model:
        xv = p.build_pymc()
        pm.Potential("like", -0.5 * ((xv - 7.5) / 0.01) ** 2)
    return model, p


class _StubSystem:
    """Duck-typed stand-in for System: parameter lookup + seed storage."""

    def __init__(self, params, seed_resolved=None):
        self._params = params

        class _CM:
            pass

        self.config_manager = _CM()
        self.config_manager.seed_resolved = seed_resolved

    def get_all_parameters(self):
        return self._params


def test_apply_polished_starts_reanchors_seed0():
    """
    Given a polished raw start for seed 0,
    When System.apply_polished_starts adopts it,
    Then raw_initval carries the polished raw point and initval the
      polished physical value, so get_raw_start/get_mcmc_init and the
      startup table all report the polished start.
    """
    # ARRANGE
    model, p = _toy_param_model()
    polished, _, method = polish_raw_starts(
        model, [{"toy.x_raw": np.zeros(1)}]
    )
    assert method == "lbfgs"
    stub = _StubSystem([p])

    # ACT
    System.apply_polished_starts(stub, polished, [0])

    # ASSERT
    assert p.initval == pytest.approx(7.5, abs=0.01)
    np.testing.assert_allclose(
        p.raw_initval, polished[0]["toy.x_raw"].reshape(-1)
    )
    # and the raw start round-trips to the same physical point
    assert p.phys_from_raw(np.asarray(p.raw_initval))[0] == pytest.approx(
        7.5, abs=0.01
    )


def test_apply_polished_starts_writes_extra_seeds_to_seed_resolved():
    """
    Given polished raw starts for seeds 0 and 2,
    When System.apply_polished_starts adopts them,
    Then seed 2's polished PHYSICAL value lands in
      config_manager.seed_resolved[2] under the indexed path, so
      get_raw_starts re-derives it in whatever raw coordinates are current.
    """
    # ARRANGE
    model, p = _toy_param_model()
    seed_resolved = [{}, {"toy.0.x": 1.0}, {"toy.0.x": 3.0}]
    stub = _StubSystem([p], seed_resolved=seed_resolved)
    raw_seed2 = p.raw_from_initval(np.array([6.0]))
    polished = [
        {"toy.x_raw": np.zeros(1)},
        {"toy.x_raw": np.asarray(raw_seed2, dtype=float)},
    ]

    # ACT
    System.apply_polished_starts(stub, polished, [0, 2])

    # ASSERT
    assert seed_resolved[2]["toy.0.x"] == pytest.approx(6.0, abs=1e-6)
    assert seed_resolved[1]["toy.0.x"] == 1.0  # untouched


def test_set_whitening_keeps_polished_start_at_same_physical_point():
    """
    Given a nonzero raw_initval (a polished start off the raw=0 anchor),
    When set_whitening rescales by a large multiplier,
    Then raw_initval is re-expressed so it decodes to the SAME physical
      value -- the invariance that lets the whitening probe measure around
      the polished start and rescale in place.
    """
    # ARRANGE
    model, p = _toy_param_model()
    raw_pol = np.asarray(p.raw_from_initval(np.array([7.5])), dtype=float)
    p.raw_initval = raw_pol.copy()
    phys_before = p.phys_from_raw(raw_pol)[0]

    # ACT
    p.set_whitening(np.array([0.003]))

    # ASSERT
    raw_after = np.asarray(p.raw_initval, dtype=float)
    assert not np.allclose(raw_after, raw_pol)  # coordinates changed...
    assert p.phys_from_raw(raw_after)[0] == pytest.approx(
        phys_before, rel=1e-9
    )  # ...the physical start did not


def test_polished_start_survives_measure_and_whiten():
    """
    Given a model polished to its mode and a stub system exposing
      get_raw_start from the parameter's raw_initval,
    When measure_and_whiten runs (probe + rescale + barrier pass),
    Then the canonical start still decodes to the polished physical point
      and the re-probed scale at it is ~1 raw unit.
    """
    from exozippy.whitening import measure_and_whiten, probe_scales

    # ARRANGE
    model, p = _toy_param_model()
    polished, _, _ = polish_raw_starts(model, [{"toy.x_raw": np.zeros(1)}])
    stub = _StubSystem([p])
    System.apply_polished_starts(stub, polished, [0])

    def get_raw_start(mdl):
        return {"toy.x_raw": np.asarray(p.raw_initval, dtype=float).copy()}

    stub.get_raw_start = get_raw_start

    # ACT
    measure_and_whiten(stub, model)

    # ASSERT
    raw_now = np.asarray(p.raw_initval, dtype=float)
    assert p.phys_from_raw(raw_now)[0] == pytest.approx(7.5, abs=0.01)
    _, scales = probe_scales(get_raw_start(model), model.compile_logp())
    assert scales["toy.x_raw"][0] == pytest.approx(1.0, rel=0.15)


def test_polish_reaches_the_peak_when_lp_is_large_in_magnitude():
    """
    Given a curved valley whose logp carries a large constant offset
      (|lp| ~ 2000, like a real fit's likelihood normalization),
    When polish_raw_starts runs,
    Then the polished point reaches the true optimum to within a nat.

    Regression: scipy's ftol is RELATIVE to |f|, so a bare 1e-3 stopped
    the polish whenever an iteration gained < 1e-3*|lp| (~2 nats at
    lp ~ -2000) -- on ob140939 that stranded seeds ~15 nats below their
    basin peaks while hot-chain candidates reached the true optima.
    The tolerance is now an absolute 0.01 nats per iteration.
    """
    # ARRANGE: Rosenbrock-flavored curved valley, optimum at (1, 1) with
    # lp_max = -2000 exactly.
    with pm.Model() as model:
        x = pm.Flat("x")
        y = pm.Flat("y")
        pm.Potential(
            "like",
            -2000.0 - 0.5 * ((y - x**2) ** 2 / 0.01 + (x - 1.0) ** 2),
        )
    start = {"x": np.array(-1.0), "y": np.array(1.0)}

    # ACT
    polished, dlps, method = polish_raw_starts(model, [start], n_steps=500)

    # ASSERT
    assert method == "lbfgs"
    lp_fn = model.compile_logp()
    assert float(lp_fn(polished[0])) == pytest.approx(-2000.0, abs=1.0)


# ---------------------------------------------------------------------------
# review 2.9.1: `seed_polish: 1` is one step, and stopping is by TOLERANCE
# with the step count as a safety cap.
# ---------------------------------------------------------------------------


def test_integer_one_is_one_step_not_the_default():
    """
    Given `seed_polish: 1` -- the integer 1, not the boolean True,
    When resolve_polish_steps maps it,
    Then the cap is 1 step.

    Regression (notes/code_review_20260808.txt 2.9.1): the old
    `spec in (True, "on")` test matched the integer 1, because 1 == True in
    Python, so asking for a single step silently got DEFAULT_POLISH_STEPS
    (150).  Every small integer 2..N was honored, which is what made the
    one-value hole invisible.
    """
    assert resolve_polish_steps(1, n_seeds=1, has_seed_hints=False) == 1
    assert resolve_polish_steps(2, n_seeds=1, has_seed_hints=False) == 2
    # True must still mean "the default cap", not "1 step".
    assert (
        resolve_polish_steps(True, n_seeds=1, has_seed_hints=False)
        == DEFAULT_POLISH_STEPS
    )


def test_integer_zero_is_off_and_stays_off():
    """
    Given `seed_polish: 0`,
    When resolve_polish_steps maps it,
    Then it is 0 -- the symmetric `0 == False` collision was harmless
      (0 steps IS off) and must stay harmless after the bool fix.
    """
    assert resolve_polish_steps(0, n_seeds=1, has_seed_hints=False) == 0
    assert resolve_polish_steps(False, n_seeds=1, has_seed_hints=False) == 0
    assert resolve_polish_steps(None, n_seeds=1, has_seed_hints=False) == 0


def test_de_polish_step_count_is_a_cap_honored_exactly_at_one():
    """
    Given the gradient-free DE engine with n_steps=1,
    When polish_seed_starts runs,
    Then exactly ONE sweep of pop_size proposals happens (plus the
      pop_size population-seeding evaluations) -- `seed_polish: 1` really
      is one step of work, which is what 2.9.1 was about.
    """
    # ARRANGE
    from exozippy.samplers.ptde import polish_seed_starts

    calls = []

    def logp(p):
        calls.append(1)
        return float(-0.5 * np.sum((p["x"] - 3.0) ** 2))

    seed = {"x": np.array([0.0])}
    scales = {"x": np.ones(1)}

    # ACT
    polish_seed_starts(
        [seed],
        logp,
        np.random.default_rng(0),
        scales,
        n_steps=1,
        pop_size=8,
    )

    # ASSERT: 8 seeding evaluations + 8 proposals in the single sweep
    assert len(calls) == 16


def test_de_polish_default_is_the_step_cap_not_an_improvement_window():
    """
    Given a start already sitting at its basin optimum,
    When the gradient-free DE polish runs with the DEFAULT settings,
    Then all n_steps sweeps run: this engine's default stopping criterion
      is the cap, deliberately.

    Why (measured on examples/DC2018_128, tabulated on
    ptde.POLISH_TOL_NATS): the best-lp history of a T=1 Metropolis
    population is a STAIRCASE of exactly-flat plateaus, so a best-lp
    improvement window fires on a plateau and misses the next jump -- 38 to
    137 nats short there, by the SAME amount at tol = 0.05, 0.5 and 2.0,
    which proves no threshold separates the two cases.  A start left tens
    of nats below its basin optimum is exactly what poisons the whitening
    probe, i.e. the thing the polish exists to prevent.
    """
    # ARRANGE
    from exozippy.samplers.ptde import polish_seed_starts

    calls = []

    def logp(p):
        calls.append(1)
        return float(-0.5 * np.sum((p["x"] - 3.0) ** 2))

    pop = 8

    # ACT
    polish_seed_starts(
        [{"x": np.array([3.0])}],
        logp,
        np.random.default_rng(0),
        {"x": np.ones(1)},
        n_steps=25,
        pop_size=pop,
    )

    # ASSERT
    assert len(calls) == pop + 25 * pop


def test_de_polish_improvement_window_is_available_as_an_opt_in():
    """
    Given a caller who opts in with tol/tol_window on a smooth surface,
    When the DE polish runs from a start already at its optimum,
    Then it stops one window past the window length instead of burning the
      whole cap -- the machinery works; only the DEFAULT is off.
    """
    # ARRANGE
    from exozippy.samplers.ptde import (
        POLISH_TOL_NATS,
        POLISH_TOL_WINDOW,
        polish_seed_starts,
    )

    calls = []

    def logp(p):
        calls.append(1)
        return float(-0.5 * np.sum((p["x"] - 3.0) ** 2))

    pop = 8

    # ACT
    _polished, dlps = polish_seed_starts(
        [{"x": np.array([3.0])}],
        logp,
        np.random.default_rng(0),
        {"x": np.ones(1)},
        n_steps=150,
        pop_size=pop,
        tol=POLISH_TOL_NATS,
    )

    # ASSERT
    sweeps = (len(calls) - pop) / pop
    assert sweeps == POLISH_TOL_WINDOW + 1
    assert dlps[0] >= 0.0


def test_an_improvement_window_would_quit_on_a_staircase_plateau():
    """
    Given a best-lp history shaped like the DE polish's real one -- flat
      plateaus punctuated by jumps (examples/DC2018_128),
    When an improvement window of any tolerance is applied to it,
    Then it stops on the first plateau and misses every later jump.

    This is the measurement that keeps the DE tolerance OFF by default,
    kept as an executable statement so nobody turns it on by analogy with
    the L-BFGS path's gradient tolerance.  Note the verdict does not move
    with the tolerance: the plateaus are EXACTLY flat.
    """
    # ARRANGE: a plateau, a jump, a longer plateau, a bigger jump
    history = [100.0] * 15 + [140.0] * 40 + [180.0] * 30 + [220.0] * 15

    def first_stop(window, tol):
        for t in range(window, len(history)):
            if history[t] - history[t - window] < tol:
                return t + 1
        return len(history)

    # ACT + ASSERT
    for window in (10, 20, 30):
        stops = {tol: first_stop(window, tol) for tol in (0.05, 0.5, 2.0)}
        assert len(set(stops.values())) == 1, stops
        stop = stops[0.05]
        assert history[stop - 1] < history[-1]


def test_lbfgs_polish_stops_on_the_gradient_not_the_cap():
    """
    Given a smooth quadratic basin and the default 150-iteration cap,
    When the L-BFGS engine polishes,
    Then it converges on the gradient tolerance in a handful of iterations
      -- the cap is a safety net, never the stopping criterion -- and
      capping at 1 iteration measurably under-polishes the same problem.
    """
    # ARRANGE: a curved valley, so one iteration is demonstrably not enough
    with pm.Model() as model:
        x = pm.Flat("x")
        y = pm.Flat("y")
        pm.Potential("like", -0.5 * ((y - x**2) ** 2 / 0.01 + (x - 1.0) ** 2))
    start = {"x": np.array(-1.0), "y": np.array(1.0)}
    lp_fn = model.compile_logp()

    # ACT
    _cap1, dlp_1, _m1 = polish_raw_starts(model, [start], n_steps=1)
    full, dlp_full, _m2 = polish_raw_starts(model, [start], n_steps=150)

    # ASSERT
    assert dlp_1[0] < dlp_full[0]
    assert float(lp_fn(full[0])) == pytest.approx(0.0, abs=1.0)
