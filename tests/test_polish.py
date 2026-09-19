"""Pre-whitening seed polish (polish.py) and its start re-anchoring.

The polish promotes a solution-estimate start to its basin's optimum BEFORE
the whitening probe measures scales around it (a start far off its optimum
makes the probe gradient-dominated).  These tests pin: the seed-provenance
gate, the L-BFGS engine, the gradient-free DE fallback dispatch, the
adoption of polished starts (raw_initval / initval / seed_resolved), and
set_whitening keeping a nonzero raw start pinned to the same physical point
through a rescale.
"""

import logging
from pathlib import Path

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

    # ACT.  cores=1 is deliberate: `cores=None` means AUTO (the sampler's
    # own grant), so leaving it off would fork a pool of most of the box
    # for a one-parameter toy -- inside a suite that is already running
    # xdist workers.  Every test here that takes the DE branch says serial
    # explicitly for that reason.
    polished, dlps, method = polish_raw_starts(
        model, [start], n_steps=200, rng=np.random.default_rng(3), cores=1
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
    (then 150, now 400).  Every small integer 2..N was honored, which is
    what made the one-value hole invisible.
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
    Given a smooth quadratic basin and a 150-iteration cap (the shipped
      DEFAULT_POLISH_STEPS is larger; the point is that neither is reached),
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


# ---------------------------------------------------------------------------
# Gamma adaptation and cross-seed pooling in the DE polish
# ---------------------------------------------------------------------------


def _polish_gamma_transition(caplog, **kwargs):
    """Run the DE polish and read (gamma_initial, gamma_final) off its log.

    Asserting on gamma directly, rather than on how good the answer was,
    is deliberate: on any surface simple enough for a unit test both
    settings reach the same optimum, so an outcome comparison measures
    nothing (the first version of this test compared 0.4995 against
    0.4999 and failed on noise).  The behaviour under test is that gamma
    MOVES, and in which direction.
    """
    import re

    from exozippy.samplers.ptde import polish_seed_starts

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="exozippy.samplers.ptde"):
        polish_seed_starts(**kwargs)
    m = re.search(r"gamma ([0-9.eE+-]+)->([0-9.eE+-]+)", caplog.text)
    assert m, f"no gamma transition logged; got: {caplog.text!r}"
    return float(m.group(1)), float(m.group(2))


# A unit-variance well with the population born TIGHT inside it (jitter
# 1e-3), so the only thing setting acceptance is gamma.  Getting this
# arrangement right matters: an earlier version of these tests jittered the
# population by a full unit into the tails of a narrow well, where the
# members are so far out that almost any inward step is a huge lp gain --
# acceptance is then HIGH and growing gamma is the correct response, which
# is the opposite of the condition being tested.
def _tight_well(x_scale=1.0):
    def logp(p):
        return float(-0.5 * np.sum((p["x"] / x_scale) ** 2))

    return logp


def test_de_polish_shrinks_gamma_when_the_step_is_too_big(caplog):
    """
    Given a population sitting inside its well but a gamma so large that
      proposals land far outside it,
    When the DE polish runs with adaptation on (the default),
    Then gamma is driven DOWN, rather than the engine sitting at ~0
      acceptance for the whole run.

    This is the defect the adaptation exists for.  ptde_async starts from
    the same 2.38/sqrt(2D) rule of thumb and tunes it; on DC2018 event 128
    (D = 27) it settles at ~0.055 against the formula's 0.3239.  Run fixed
    at 0.3239 the polish sustained 0.003-0.004 acceptance with 84-88% of
    sweeps accepting nothing at all, so most of the population never moved
    off its birth position and best-minus-median stayed ~800-1000 nats
    where a converged population shows ~D/2.
    """
    # ARRANGE: steps of ~1e4 * 1e-3 = 10 sigma, essentially all rejected
    # ACT
    g0, g1 = _polish_gamma_transition(
        caplog,
        raw_starts=[{"x": np.array([0.0])}],
        logp_fn=_tight_well(),
        rng=np.random.default_rng(0),
        scales={"x": np.full(1, 1e-3)},
        n_steps=40,
        pop_size=8,
        gamma=1e4,
        adapt_gamma=True,
    )

    # ASSERT
    assert g1 < g0 / 5.0, f"gamma did not shrink: {g0} -> {g1}"


def test_de_polish_grows_gamma_when_the_step_is_too_small(caplog):
    """
    Given a gamma so small that every proposal is a no-op and therefore
      accepted,
    When the DE polish adapts,
    Then gamma is driven UP -- the rule is a controller, not a one-way
      shrink, so both directions are pinned.
    """
    # ARRANGE: steps of ~1e-6 * 1e-3, lp change negligible -> accept ~1
    # ACT
    g0, g1 = _polish_gamma_transition(
        caplog,
        raw_starts=[{"x": np.array([0.0])}],
        logp_fn=_tight_well(),
        rng=np.random.default_rng(0),
        scales={"x": np.full(1, 1e-3)},
        n_steps=40,
        pop_size=8,
        gamma=1e-6,
        adapt_gamma=True,
    )

    # ASSERT
    assert g1 > g0 * 5.0, f"gamma did not grow: {g0} -> {g1}"


def test_de_polish_leaves_gamma_alone_when_adaptation_is_off(caplog):
    """Opting out must really opt out -- the pre-2026-08 behaviour."""
    g0, g1 = _polish_gamma_transition(
        caplog,
        raw_starts=[{"x": np.array([0.0])}],
        logp_fn=_tight_well(),
        rng=np.random.default_rng(0),
        scales={"x": np.full(1, 1e-3)},
        n_steps=40,
        pop_size=8,
        gamma=1e4,
        adapt_gamma=False,
    )
    assert g0 == g1


def test_de_polish_gamma_shrinks_when_nothing_is_accepted():
    """
    Given a surface so sharp that a whole window accepts zero proposals,
    When the DE polish adapts gamma,
    Then gamma still shrinks rather than stalling -- the (ar/target)**0.5
      rule has no signal at ar = 0, so that case is handled explicitly.
    """
    # ARRANGE
    from exozippy.samplers.ptde import polish_seed_starts

    seen = []

    def logp(p):
        # -inf everywhere except exactly the seed: nothing can be accepted
        return 0.0 if np.all(p["x"] == 0.0) else -np.inf

    # ACT
    polished, dlps = polish_seed_starts(
        [{"x": np.array([0.0])}],
        logp,
        np.random.default_rng(0),
        {"x": np.ones(1)},
        n_steps=20,
        pop_size=8,
        adapt_gamma=True,
    )

    # ASSERT: it survives, and returns the seed rather than a -inf point
    np.testing.assert_allclose(polished[0]["x"], [0.0])
    assert dlps[0] == 0.0


def test_de_polish_pools_proposals_across_all_seeds():
    """
    Given several seeds and a pool,
    When the DE polish runs,
    Then every sweep hands the pool ONE batch containing all seeds'
      proposals, so workers are shared dynamically rather than the seeds
      being partitioned between them.

    Two seeds on 64 cores must not become 32 + 32 with one half idling once
    its seed finishes; the batch is n_seeds * pop_size items and workers
    take the next available one.
    """
    # ARRANGE
    from exozippy.samplers.ptde import polish_seed_starts

    batch_sizes = []

    class _RecordingPool:
        def map(self, fn, items):
            batch_sizes.append(len(items))
            return [fn(x) for x in items]

    def logp(p):
        return float(-0.5 * np.sum((p["x"] - 3.0) ** 2))

    n_seeds, pop = 3, 8
    seeds = [{"x": np.array([float(i)])} for i in range(n_seeds)]

    # ACT
    polish_seed_starts(
        seeds,
        logp,
        np.random.default_rng(0),
        {"x": np.ones(1)},
        n_steps=5,
        pop_size=pop,
        pool=_RecordingPool(),
    )

    # ASSERT: one seeding batch of all populations, then one batch per
    # sweep, each carrying every seed's proposals
    assert batch_sizes[0] == n_seeds * pop
    assert batch_sizes[1:] == [n_seeds * pop] * 5


def test_de_polish_serial_and_pooled_agree_on_the_same_stream():
    """
    Given the same seeds, rng stream and settings,
    When the polish runs serially and through a pool,
    Then both return the same polished point: the pool changes WHERE the
      logp calls happen, not the chain.
    """
    # ARRANGE
    from exozippy.samplers.ptde import polish_seed_starts

    class _SerialPool:
        def map(self, fn, items):
            return [fn(x) for x in items]

    def logp(p):
        return float(-0.5 * np.sum((p["x"] - 3.0) ** 2))

    kw = dict(n_steps=10, pop_size=8, adapt_gamma=True)
    seeds = [{"x": np.array([0.0])}, {"x": np.array([6.0])}]

    # ACT
    a, dlp_a = polish_seed_starts(
        [dict(s) for s in seeds],
        logp,
        np.random.default_rng(7),
        {"x": np.ones(1)},
        **kw,
    )
    b, dlp_b = polish_seed_starts(
        [dict(s) for s in seeds],
        logp,
        np.random.default_rng(7),
        {"x": np.ones(1)},
        pool=_SerialPool(),
        **kw,
    )

    # ASSERT
    for pa, pb in zip(a, b):
        np.testing.assert_allclose(pa["x"], pb["x"])
    np.testing.assert_allclose(dlp_a, dlp_b)


# ---------------------------------------------------------------------------
# The gamma rule itself, now shared by the polish and both samplers (4.4.1)
# ---------------------------------------------------------------------------


def test_next_gamma_is_the_damped_clipped_controller_all_three_callers_want():
    """
    Given an acceptance measurement and a target,
    When the shared step-size rule is applied,
    Then it is gamma * sqrt(ar/target), clipped to a factor 10 either way,
      and an acceptance of exactly zero shrinks by that same factor.

    One rule, three callers (ptde, ptde_async, polish_seed_starts): the
    polish engine IS the sampler's T=1 move, so a second tuning story would
    be one more thing to keep in sync.  The zero-acceptance branch is the
    one difference between the callers -- the samplers guard on ar > 0 and
    never reach it; the polish relies on it, because a T=1 optimizer
    routinely accepts nothing for a whole window.
    """
    from exozippy.samplers._common import GAMMA_CLIP_FACTOR, next_gamma

    # sqrt-damped, both directions
    assert next_gamma(1.0, 0.8, 0.2) == pytest.approx(2.0)
    assert next_gamma(1.0, 0.05, 0.2) == pytest.approx(0.5)
    assert next_gamma(3.0, 0.2, 0.2) == pytest.approx(3.0)

    # clipped either way
    assert next_gamma(1.0, 1.0, 1e-6) == pytest.approx(GAMMA_CLIP_FACTOR)
    assert next_gamma(1.0, 1e-9, 0.2) == pytest.approx(1.0 / GAMMA_CLIP_FACTOR)

    # zero acceptance shrinks by the clip factor rather than stalling, and
    # is bit-for-bit the `gamma * 0.1` the polish used to write inline
    assert next_gamma(0.3, 0.0, 0.2) == 0.3 * 0.1


# ---------------------------------------------------------------------------
# The core grant: `cores=None` means AUTO, not serial (6.11.3), and the
# expensive branch says which one it got (2.3.5 b/c)
# ---------------------------------------------------------------------------


def test_no_grant_means_the_machine_not_one_core():
    """
    Given a caller that names no core grant,
    When the DE polish resolves its worker count,
    Then it takes the SAME grant a sampler takes when nothing names one --
      not 1.

    This equivalence is the fix for 6.11.3.  `cores=None` meant serial here
    while it meant "take the machine" in _common.create_pool two modules
    away, so the hot-mode polish -- which passed nothing -- ran on 1 core of
    36 for 38 minutes right after a sampling phase that had been using 27.
    Nothing about the polish wants fewer cores than the sampling either side
    of it.
    """
    from exozippy.polish import _resolve_polish_cores
    from exozippy.samplers._common import default_cores

    assert _resolve_polish_cores(None, n_seeds=1) == default_cores()
    assert default_cores() >= 1
    # and serial is still reachable -- by ASKING for it, which is a
    # statement rather than an omission
    assert _resolve_polish_cores(1, n_seeds=1) == 1


def test_default_cores_leaves_one_core_for_the_machine():
    """
    Given the shared default-grant rule,
    When it resolves on a multi-core box,
    Then it is CORE_FRACTION of the physical cores AND at most n_phys - 1.

    The `n_phys - 1` arm is the half that kept getting dropped: one of the
    three hand-written copies (nested.py) had lost it, so an unconfigured
    nested run took every core the OS and the user's shell were meant to
    keep one of.
    """
    import multiprocessing as mp

    from exozippy.constants import CORE_FRACTION
    from exozippy.samplers._common import default_cores

    phys = mp.cpu_count()
    got = default_cores()
    assert got == max(1, min(int(phys * CORE_FRACTION), phys - 1))
    if phys > 1:
        assert got <= phys - 1


def _create_pool_cores(cores, monkeypatch):
    """``create_pool``'s resolved worker count, without forking a pool.

    The fork itself is not what is under test and would cost `default_cores()`
    processes per parametrized case, so the context's Pool is stubbed out.
    """
    import multiprocessing as mp

    from exozippy.samplers import _common

    class _FakeCtx:
        def Pool(self, n, initializer=None):
            return f"pool[{n}]"

    monkeypatch.setattr(mp, "get_context", lambda _kind: _FakeCtx())
    # total_proposals large enough that the batch-size cap never binds: the
    # grant is what is being measured, not `min(grant, work_available)`.
    _pool, actual = _common.create_pool(
        cores, 10**6, "test", logging.getLogger("test")
    )
    return actual


def _resolvers():
    """The three functions that turn a `cores` value into a worker count."""
    from exozippy.polish import _resolve_polish_cores
    from exozippy.samplers.nested import _resolve_nested_cores

    return {
        "create_pool": lambda c, mp_: _create_pool_cores(c, mp_),
        "polish": lambda c, mp_: _resolve_polish_cores(c, n_seeds=1000),
        "nested": lambda c, mp_: _resolve_nested_cores(c),
    }


@pytest.mark.parametrize("resolver", sorted(_resolvers()))
@pytest.mark.parametrize("value", [0, -4, None])
def test_zero_and_negative_cores_are_the_automatic_grant(
    resolver, value, monkeypatch
):
    """
    Given `cores` written as 0, as a negative number, or omitted,
    When each of the three resolvers turns it into a worker count,
    Then all three return the SAME automatic grant.

    Review 2.4.8: `0` is not `None`, so the `cores=None` rule (6.11.3) left
    it to each resolver's accident.  create_pool took
    `min(0, total_proposals)` and ran SERIAL, _resolve_polish_cores swept it
    into its `n <= 1` SERIAL arm, and nested.py read it as AUTO because
    `cores or default_cores()` treats 0 as falsy -- one written number, two
    behaviors inside a single run.  A negative value was worse: it reached
    nested.py's pool size unclamped.  JDE's ruling is that `<= 0` is the
    automatic grant everywhere.
    """
    from exozippy.samplers._common import default_cores

    # ACT
    got = _resolvers()[resolver](value, monkeypatch)

    # ASSERT
    assert got == default_cores()


@pytest.mark.parametrize("resolver", sorted(_resolvers()))
def test_one_core_is_still_serial_in_every_resolver(resolver, monkeypatch):
    """
    Given `cores: 1`,
    When each of the three resolvers turns it into a worker count,
    Then all three return 1.

    The other half of 2.4.8: folding `<= 0` into AUTO must not take serial
    away.  `cores=1` is the statement a caller makes when they mean one core,
    and it is the only way to ask for it.
    """
    # ACT
    got = _resolvers()[resolver](1, monkeypatch)

    # ASSERT
    assert got == 1


def test_zero_cores_warns_and_resolves_to_the_auto_sentinel(caplog):
    """
    Given a config that writes `sampler: cores: 0`,
    When run.py parses it,
    Then it returns the None AUTO sentinel and WARNS, naming serial as the
      thing `cores: 1` asks for.

    Rope, not gates: 0 is a plausible spelling of "let the machine decide",
    so the run continues with that reading rather than raising -- but the
    user is told which reading they got, because the other reading (serial)
    is the one two of the three resolvers used to take.  Normalizing here, at
    the parse boundary, is what keeps the three from disagreeing at all.
    """
    from exozippy.run import resolve_cores_setting

    # ACT
    with caplog.at_level(logging.WARNING, logger="exozippy.run"):
        got = resolve_cores_setting(0)

    # ASSERT
    assert got is None
    assert "automatic grant" in caplog.text
    assert "cores: 1" in caplog.text
    assert "serial" in caplog.text
    # and a real count is still passed through untouched
    assert resolve_cores_setting(3) == 3
    assert resolve_cores_setting(None) is None


def test_serial_de_polish_announces_the_cost(caplog):
    """
    Given a caller that explicitly asks for serial,
    When polish_raw_starts dispatches the DE engine,
    Then the log says it is running SERIAL, names the cores value it was
      handed, and says this branch is the expensive one.

    The gradient-fallback line reads as a note about capability; what it
    means for the user is one core for the whole stage, and on ob09020 that
    was 38+ minutes with nothing to read but /proc/<pid>/stat.
    """
    # ARRANGE
    with pm.Model() as model:
        x = pm.Flat("x")
        pm.Potential("like", _NoGradSquare()(x))

    # ACT
    with caplog.at_level(logging.INFO, logger="exozippy.polish"):
        _polished, _dlps, method = polish_raw_starts(
            model, [{"x": np.array(0.0)}], n_steps=3, cores=1
        )

    # ASSERT
    assert method == "de"
    assert "SERIAL" in caplog.text
    assert "cores=1" in caplog.text
    assert "expensive" in caplog.text


def test_pooled_de_polish_announces_its_worker_count(caplog):
    """
    Given a core grant of more than one,
    When the DE engine builds its pool,
    Then the log names the worker count instead of the serial line, so the
      two cases stay distinguishable in a fit log after the fact.
    """
    # ARRANGE
    with pm.Model() as model:
        x = pm.Flat("x")
        pm.Potential("like", _NoGradSquare()(x))

    # ACT
    with caplog.at_level(logging.INFO, logger="exozippy.polish"):
        polish_raw_starts(model, [{"x": np.array(0.0)}], n_steps=2, cores=2)

    # ASSERT
    assert "worker process(es)" in caplog.text
    assert "SERIAL" not in caplog.text


def test_de_polish_heartbeats_on_wall_clock_not_sweep_count(caplog):
    """
    Given a polish long enough to cross the heartbeat interval,
    When the sweep loop runs,
    Then it logs progress lines carrying the sweep count, the elapsed time
      and an upper-bound ETA -- and at the DEFAULT interval a short polish
      logs none of them.

    Wall clock rather than "every N sweeps": one sweep of a binary-lens
    model can take milliseconds or minutes, so a sweep count that is chatty
    on one model is silent for 40 minutes on another (review 2.3.5b).
    """
    # ARRANGE
    from exozippy.samplers.ptde import polish_seed_starts

    def logp(p):
        return float(-0.5 * np.sum((p["x"] - 3.0) ** 2))

    seeds = [{"x": np.array([0.0])}]
    scales = {"x": np.ones(1)}

    # ACT: an interval short enough that every sweep crosses it
    with caplog.at_level(logging.INFO, logger="exozippy.samplers.ptde"):
        polish_seed_starts(
            seeds,
            logp,
            np.random.default_rng(0),
            scales,
            n_steps=4,
            pop_size=8,
            progress_interval_s=1e-9,
        )
    beats = [
        r
        for r in caplog.records
        if "PTDE seed polish: sweep" in r.getMessage()
    ]

    # ASSERT
    assert len(beats) == 4
    assert "/4" in beats[0].getMessage()
    assert "elapsed=" in beats[0].getMessage()
    assert "eta<=" in beats[0].getMessage()

    # ACT again, at the shipped default: a sub-second polish says nothing
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="exozippy.samplers.ptde"):
        polish_seed_starts(
            seeds,
            logp,
            np.random.default_rng(0),
            scales,
            n_steps=4,
            pop_size=8,
        )

    # ASSERT
    assert "PTDE seed polish: sweep" not in caplog.text


# ---------------------------------------------------------------------------
# The heartbeat has to fire from INSIDE a sweep (3.4.4)
# ---------------------------------------------------------------------------


class _FakeAsyncResult:
    """One `pool.apply_async` handle that takes real wall-clock time.

    The clock starts when the collector FIRST waits on this result, not at
    submission, so results stay spread out over wall clock even if the test
    process is descheduled for a while before collection begins -- which on
    a loaded box is otherwise how every result in a batch becomes ready at
    once and the heartbeat has nothing to beat between.
    """

    def __init__(self, pool, fn, arg, wedged):
        self.pool, self.fn, self.arg = pool, fn, arg
        self.wedged = wedged
        self.ready_at = None

    def get(self, timeout=None):
        import multiprocessing as mp
        import time

        if self.wedged:
            # The near-caustic VBM evaluation: never comes back, whatever
            # anyone waits.
            if timeout is not None:
                time.sleep(timeout)
            raise mp.TimeoutError
        if self.ready_at is None:
            self.ready_at = time.monotonic() + self.pool.per_item
        remaining = self.ready_at - time.monotonic()
        if remaining > 0:
            if timeout is not None and timeout < remaining:
                time.sleep(timeout)
                raise mp.TimeoutError
            time.sleep(remaining)
        return self.fn(self.arg)


class _FakeAsyncPool:
    """A one-worker multiprocessing.Pool, in-process and on a real clock.

    Items queue behind each other (`per_item` seconds apiece) so a batch's
    results arrive spread out over wall clock, which is the only thing that
    makes a mid-batch heartbeat observable.  `wedge` names submission
    indices whose result never arrives.

    `map` deliberately REFUSES a batch containing a wedged item instead of
    emulating the block: a real pool.map would hang the test process
    forever, which is the defect, and a hung test is indistinguishable from
    an infrastructure problem.
    """

    def __init__(self, per_item=0.0, wedge=()):
        self.per_item = per_item
        self.wedge = set(wedge)
        self.n_submitted = 0
        self.n_apply_async = 0
        self.evaluated = []

    def map(self, fn, items):
        n0 = self.n_submitted
        self.n_submitted += len(items)
        if self.wedge & set(range(n0, self.n_submitted)):
            raise AssertionError(
                "pool.map was handed a batch containing a proposal that "
                "never returns; a real pool would block here forever"
            )
        self.evaluated.extend(items)
        return [fn(x) for x in items]

    def apply_async(self, fn, args):
        self.n_apply_async += 1
        idx = self.n_submitted
        self.n_submitted += 1
        wedged = idx in self.wedge
        if not wedged:
            self.evaluated.append(args[0])
        return _FakeAsyncResult(self, fn, args[0], wedged)


def _quadratic_logp(p):
    return float(-0.5 * np.sum((p["x"] - 3.0) ** 2))


def test_heartbeat_fires_inside_a_sweep_not_only_between_them(caplog):
    """
    Given a pool whose results arrive spread out over wall clock,
    When one sweep takes several heartbeat intervals,
    Then that ONE sweep produces several progress lines, each naming the
      sweep as in progress and how many of its proposals have come back.

    One sweep is one batch of n_seeds * pop_size evaluations.  While that
    batch was a single blocking pool.map the heartbeat could only fire
    after it returned, so the interval the user configured bought nothing
    on the one engine it was written for -- a sweep of a binary-lens model
    can take minutes, and a wedged evaluation takes forever (3.4.4).
    """
    # ARRANGE
    from exozippy.samplers.ptde import polish_seed_starts

    pool = _FakeAsyncPool(per_item=0.05)

    # ACT
    with caplog.at_level(logging.INFO, logger="exozippy.samplers.ptde"):
        polish_seed_starts(
            [{"x": np.array([0.0])}],
            _quadratic_logp,
            np.random.default_rng(0),
            {"x": np.ones(1)},
            n_steps=2,
            pop_size=6,
            pool=pool,
            progress_interval_s=0.06,
            asynchronous=False,
        )

    msgs = [r.getMessage() for r in caplog.records]
    mid = [m for m in msgs if "IN PROGRESS" in m]

    # ASSERT: several beats inside sweep 1 alone, each carrying the count
    in_sweep_1 = [m for m in mid if "sweep 1/2 IN PROGRESS" in m]
    assert len(in_sweep_1) >= 2, msgs
    assert all("proposals back)" in m for m in in_sweep_1)
    # sweep 2 gets its own, so the label tracks the sweep rather than being
    # a fixed string
    assert any("sweep 2/2 IN PROGRESS" in m for m in mid), msgs
    # the opening batch, which scores every population, is covered too
    assert any("scoring the initial population" in m for m in msgs), msgs


def test_a_wedged_proposal_does_not_silence_the_heartbeat(caplog):
    """
    Given one proposal whose logp call never returns and an eval_timeout,
    When the polish evaluates the sweep containing it,
    Then the heartbeat keeps beating while that proposal is outstanding,
      the proposal is abandoned and scored -inf (so it can never become the
      polished point), the pool is recycled, and the sweep completes.

    This is the case the heartbeat exists for and the one it used to miss
    entirely: a hung evaluation blocked the whole pool.map, so the log went
    quiet forever precisely when a watcher needed to tell computing from
    hung.
    """
    # ARRANGE
    from exozippy.samplers.ptde import polish_seed_starts

    # submissions 0-3 are the opening population batch; 4-7 are sweep 1
    pool = _FakeAsyncPool(per_item=0.01, wedge=(5,))
    recycled = []

    def _recycler(dead):
        recycled.append(dead)
        return dead  # the same fake stands in for the fresh pool

    # ACT
    with caplog.at_level(logging.INFO, logger="exozippy.samplers.ptde"):
        polished, dlps = polish_seed_starts(
            [{"x": np.array([0.0])}],
            _quadratic_logp,
            np.random.default_rng(0),
            {"x": np.ones(1)},
            n_steps=1,
            pop_size=4,
            pool=pool,
            progress_interval_s=0.05,
            asynchronous=False,
            eval_timeout=0.2,
            pool_recycler=_recycler,
        )

    msgs = [r.getMessage() for r in caplog.records]

    # ASSERT
    assert any("IN PROGRESS" in m for m in msgs), msgs
    assert any("exceeded" in m and "eval_timeout" in m for m in msgs), msgs
    assert recycled == [pool]
    # the sweep finished, and the abandoned proposal -- which was never
    # evaluated at all -- cannot be what the polish returned
    assert np.isfinite(dlps[0])
    assert any(np.allclose(polished[0]["x"], e["x"]) for e in pool.evaluated)


def test_the_interleaved_path_returns_the_same_stream_as_map():
    """
    Given the same seeds, rng stream and settings,
    When the polish runs serially, through a map-only pool and through a
      pool that supports apply_async,
    Then all of them return the same polished point.

    Collecting a batch one result at a time is a change to WHEN the parent
    regains control, not to the chain: submission order is preserved, the
    accept/reject loop still runs afterwards in per-seed order, and no
    random number is drawn differently.  The last arm also pins the one
    configuration that keeps the single blocking pool.map even on a pool
    that could interleave -- no heartbeat and no timeout means nothing to
    interleave for.

    All of this is the SYNCHRONOUS engine, selected explicitly: on a pool
    with apply_async the default is now the asynchronous engine, whose
    trajectory depends on arrival order by design (see the async tests
    below).
    """
    # ARRANGE
    from exozippy.samplers.ptde import polish_seed_starts

    class _SerialPool:
        def map(self, fn, items):
            return [fn(x) for x in items]

    kw = dict(n_steps=10, pop_size=8, adapt_gamma=True)
    seeds = [{"x": np.array([0.0])}, {"x": np.array([6.0])}]

    def _run(**extra):
        return polish_seed_starts(
            [dict(s) for s in seeds],
            _quadratic_logp,
            np.random.default_rng(7),
            {"x": np.ones(1)},
            **kw,
            **extra,
        )

    # ACT
    a, dlp_a = _run()
    b, dlp_b = _run(pool=_SerialPool())
    async_pool = _FakeAsyncPool()
    c, dlp_c = _run(pool=async_pool, asynchronous=False)
    quiet_pool = _FakeAsyncPool()
    d, dlp_d = _run(
        pool=quiet_pool, progress_interval_s=None, asynchronous=False
    )

    # ASSERT
    assert async_pool.n_apply_async > 0
    assert quiet_pool.n_apply_async == 0
    for other, dlp_other in ((b, dlp_b), (c, dlp_c), (d, dlp_d)):
        for pa, pb in zip(a, other):
            np.testing.assert_allclose(pa["x"], pb["x"])
        np.testing.assert_allclose(dlp_a, dlp_other)


# ---------------------------------------------------------------------------
# review 7.13.8: the stop must not amplify arithmetic noise.  A 1-ulp
# perturbation of (lp, grad) may not move the polished start.
# ---------------------------------------------------------------------------
#
# THE MECHANISM THIS GUARDS.  scipy's gtol test is `max|proj g| <= gtol` on
# the CURRENT iterate, so it fires on the FIRST evaluation that dips under
# the threshold.  On a ridge (kelt4 RV-only: a tc/logP degeneracy with
# Hessian condition number 5.4e6) a loose gtol therefore stops mid-climb, on
# a shoulder where the endpoint depends on every bit of the path -- and the
# path differs whenever the arithmetic does, which across CI runners it
# always does (a different OpenBLAS kernel in scipy's own L-BFGS-B
# bookkeeping perturbs one iterate by 1 ulp at evaluation 4).  At the old
# gtol = 0.01 that moved the polished cosi by 8.5% (16-seed full width) and
# the polished lp by 0.24 nats, and 3 of 16 arithmetics hit the 150 cap;
# tests/test_integration_kelt4.py went red three times on it.
#
# THE HARNESS.  "The same function computed by a different but equally
# correct arithmetic" is modelled as the compiled objective times
# (1 + s * 2**-52), s in {-1, 0, +1} a hash of (x, component, seed) -- one
# ulp of relative error on every output, chosen independently per
# evaluation.  Seed 0 is unperturbed.  Under the shipped constants the four
# seeds here must land on the same optimum: this is what makes the flake
# structurally impossible to reintroduce, because a first-dip stop fails it
# by two orders of magnitude while a converged stop passes with 5x to spare.
# The full 16-seed sweep and the numbers behind the bounds are in the header
# of tests/test_integration_kelt4.py.

_KELT4_DIR = Path(__file__).parent.parent / "examples" / "kelt4"

# Measured 2026-09-14, 16 seeds, gtol 1e-4 / cap 400: cosi full width
# 8.6e-4 relative (4.3e-4 absolute), polished lp full width 9.1e-7 nats,
# 240-294 iterations.  Bounds are ~5x those widths.  The absolute cosi
# bound also has a first-principles ceiling: cosi's Schur-complement
# curvature is 0.0044 nats/raw^2, so the |grad| < 1e-4 stopping set spans
# +/-0.023 raw = +/-5e-4 in cosi, and 2e-3 absolute sits above even that.
_ULP_COSI_ATOL = 2.0e-3  # absolute, in cosi
_ULP_LP_ATOL = 5.0e-6  # nats
_ULP_SEEDS = (0, 1, -1, 3)


@pytest.fixture(scope="module")
def kelt4_rvonly_polish_inputs():
    """The kelt4 RV-only model, its raw start and polish.py's own compiled
    lp+grad function -- exactly what polish_raw_starts hands _lbfgs_polish_one."""
    import os

    import yaml

    from exozippy.polish import _compile_logp_grad

    if not _KELT4_DIR.is_dir():
        pytest.skip("kelt4 example not present")
    cwd = os.getcwd()
    os.chdir(_KELT4_DIR)
    try:
        with open("kelt4_rvonly.yaml") as f:
            config = yaml.safe_load(f)
        system = System(config)
        system.prepare()
        model = system.build_model()
        raw_start = system.get_raw_start(model)
        fn = _compile_logp_grad(model)
    finally:
        os.chdir(cwd)
    assert fn is not None, "kelt4 RV-only must be on the L-BFGS path"
    cosi = next(
        p for p in system.get_all_parameters() if p.label == "orbit.cosi"
    )
    return model, raw_start, fn, cosi


def _ulp_perturbed(fn, keys, seed):
    """`fn` with every output multiplied by (1 + s * 2**-52), s in {-1, 0, 1}
    a deterministic hash of (x, output component, seed); seed 0 unperturbed."""
    import hashlib

    eps = 2.0**-52

    def perturbed(point):
        vals = fn(point)
        if seed == 0:
            return vals
        x = np.concatenate(
            [np.asarray(point[k], float).reshape(-1) for k in keys]
        )
        flat = [np.asarray(v, float) for v in vals]
        m = sum(v.size for v in flat)
        h = hashlib.blake2b(
            x.tobytes() + int(seed).to_bytes(4, "little", signed=True),
            digest_size=32,
        ).digest()
        r = np.frombuffer(
            hashlib.blake2b(h, digest_size=4 * m).digest(), dtype=np.uint32
        )
        s = (r % 3).astype(float) - 1.0
        out, ofs = [], 0
        for v in flat:
            n = v.size
            out.append(
                (v.reshape(-1) * (1.0 + s[ofs : ofs + n] * eps)).reshape(
                    v.shape
                )
            )
            ofs += n
        return out

    return perturbed


def test_ulp_perturbation_does_not_move_the_polished_start(
    kelt4_rvonly_polish_inputs,
):
    """
    Given the kelt4 RV-only model and its raw start, with the compiled
      (lp, grad) multiplied by (1 + s*2**-52) for four seeds -- one ulp of
      arithmetic difference per evaluation, the size of a cross-runner
      libm/BLAS-kernel difference,
    When each is polished by _lbfgs_polish_one under the SHIPPED constants,
    Then every seed converges (none hits DEFAULT_POLISH_STEPS) and the four
      polished starts agree in cosi to ~5x the measured 16-seed width and
      in lp to ~5x its width -- i.e. the stop is a converged optimum, not
      a first dip on the ridge (review 7.13.8).
    """
    from exozippy.polish import DEFAULT_POLISH_STEPS, _lbfgs_polish_one

    model, raw_start, fn, cosi_par = kelt4_rvonly_polish_inputs
    keys = list(raw_start.keys())
    shapes = [np.shape(raw_start[k]) for k in keys]
    sizes = [int(np.asarray(raw_start[k]).size) for k in keys]
    assert "orbit.cosi_raw" in keys

    # Act
    cosis, lps, iters, capped = [], [], [], []
    for seed in _ULP_SEEDS:
        best, lp0, lp_best, _n_evals, n_iter, hit_cap = _lbfgs_polish_one(
            raw_start,
            _ulp_perturbed(fn, keys, seed),
            keys,
            shapes,
            sizes,
            maxiter=DEFAULT_POLISH_STEPS,
        )
        raw = float(np.asarray(best["orbit.cosi_raw"]).reshape(-1)[0])
        cosis.append(float(cosi_par.element_phys_from_raw(0, raw)))
        lps.append(lp_best)
        iters.append(n_iter)
        capped.append(hit_cap)
    assert lps[0] > lp0 + 100.0, "the polish did not climb; harness is broken"

    # Assert: converged, not capped
    assert not any(capped), (
        f"seeds {[s for s, c in zip(_ULP_SEEDS, capped) if c]} hit the "
        f"{DEFAULT_POLISH_STEPS}-iteration cap ({iters} iterations). Under the "
        f"shipped gtol kelt4 needs 240-294; a cap-stop is an unconverged "
        f"start whose value depends on the arithmetic that produced it."
    )
    # Assert: one optimum, whatever the arithmetic
    cosi_width = max(cosis) - min(cosis)
    assert cosi_width <= _ULP_COSI_ATOL, (
        f"one ulp of arithmetic moved the polished cosi by {cosi_width:.3g} "
        f"({cosis}), more than {_ULP_COSI_ATOL}: the polish is stopping on a "
        f"first dip of |grad| below _LBFGS_GTOL mid-climb rather than at the "
        f"basin optimum (review 7.13.8; at gtol 0.01 this width was 4.3e-2). "
        f"Do not widen this bound -- tighten the stop."
    )
    lp_width = max(lps) - min(lps)
    assert lp_width <= _ULP_LP_ATOL, (
        f"one ulp of arithmetic moved the polished lp by {lp_width:.3g} nats "
        f"({lps}); at a converged optimum it moves at second order (~1e-6 "
        f"here; at the old gtol 0.01 stop it was 0.24)."
    )


# The asynchronous engine (review 2.4.14): ptde_async's procedure for the polish
# ---------------------------------------------------------------------------


class _CallbackPool:
    """A multiprocessing.Pool stand-in on threads, with the two apply_async
    shapes the engine uses: callbacks (the async loop) and a handle with
    ``get(timeout)`` (the opening batch through ``_map_logp_timeout``).

    ``delay(idx)`` is the wall-clock cost of submission ``idx``; ``hang``
    names submissions that never return until ``release()`` is called.
    ``completed`` records submission indices in the order their results
    were delivered, which is what makes non-blocking behaviour observable.
    """

    def __init__(self, workers=2, delay=None, hang=()):
        import concurrent.futures
        import threading

        self._ex = concurrent.futures.ThreadPoolExecutor(workers)
        self._delay = delay or (lambda idx: 0.0)
        self._hang = set(hang)
        self._release = threading.Event()
        self._lock = threading.Lock()
        self.n_apply_async = 0
        self.completed = []

    def release(self):
        self._release.set()
        self._ex.shutdown(wait=False)

    def apply_async(self, fn, args, callback=None, error_callback=None):
        import multiprocessing as mp
        import time

        idx = self.n_apply_async
        self.n_apply_async += 1

        def _run():
            if idx in self._hang:
                self._release.wait()
                return None
            time.sleep(self._delay(idx))
            try:
                r = fn(args[0])
            except Exception as exc:  # pragma: no cover - defensive
                if error_callback is not None:
                    error_callback(exc)
                raise
            with self._lock:
                self.completed.append(idx)
            if callback is not None:
                callback(r)
            return r

        fut = self._ex.submit(_run)

        class _Res:
            def get(self, timeout=None):
                import concurrent.futures

                try:
                    return fut.result(timeout=timeout)
                except concurrent.futures.TimeoutError:
                    raise mp.TimeoutError

        return _Res()


def test_async_de_polish_climbs_to_the_mode_and_reports_its_population(caplog):
    """
    Given a pool whose apply_async delivers results through callbacks,
    When the DE polish runs with the default asynchronous engine,
    Then it climbs the quadratic to its mode, spends exactly the sweep
      budget as completed proposals, and the wrap-up says how many members
      never moved -- the diagnostic 2.4.14 needed and could not get.
    """
    # ARRANGE
    from exozippy.samplers.ptde import polish_seed_starts

    pool = _CallbackPool(workers=3)
    seeds = [{"x": np.array([0.0])}]
    n_steps, pop = 40, 8

    # ACT
    with caplog.at_level(logging.INFO, logger="exozippy.samplers.ptde"):
        polished, dlps = polish_seed_starts(
            seeds,
            _quadratic_logp,
            np.random.default_rng(0),
            {"x": np.ones(1)},
            n_steps=n_steps,
            pop_size=pop,
            pool=pool,
        )
    pool.release()

    # ASSERT
    assert abs(float(polished[0]["x"][0]) - 3.0) < 0.5
    assert dlps[0] > 0
    # opening batch (pop) + n_steps sweeps of pop proposals
    assert pool.n_apply_async == pop * (n_steps + 1)
    assert f"{n_steps} steps x {pop} pop" in caplog.text
    assert "members never moved" in caplog.text
    assert "(asynchronous engine)" in caplog.text


def test_async_de_polish_does_not_wait_on_a_slow_proposal():
    """
    Given one proposal in the first sweep that takes far longer than the rest,
    When the asynchronous engine runs on two workers,
    Then many later-submitted proposals complete BEFORE it does: the slow
      evaluation costs one worker and nothing else waits on it.

    The synchronous batch engine could not do this -- a sweep is one
    barrier, so at most the rest of that batch (pop - 1 items) can finish
    ahead of its slowest member.
    """
    # ARRANGE
    from exozippy.samplers.ptde import polish_seed_starts

    pop, n_steps = 8, 12
    slow = pop + 3  # a first-sweep proposal; 0..pop-1 is the opening batch
    pool = _CallbackPool(
        workers=2, delay=lambda idx: 0.4 if idx == slow else 0.0
    )

    # ACT
    polish_seed_starts(
        [{"x": np.array([0.0])}],
        _quadratic_logp,
        np.random.default_rng(1),
        {"x": np.ones(1)},
        n_steps=n_steps,
        pop_size=pop,
        pool=pool,
    )
    pool.release()

    # ASSERT: the slow item was delivered long after items submitted after it
    position = pool.completed.index(slow)
    assert position >= 3 * pop, (position, pool.completed[: position + 1])


def test_async_de_polish_spends_exactly_the_budget_per_seed(caplog):
    """
    Given two seeds and the asynchronous engine,
    When the polish runs to its cap,
    Then each seed reports exactly n_steps steps and the pool saw exactly
      n_seeds * pop * (n_steps + 1) submissions: the budget is counted in
      completed proposals and does not leak across seeds or sweeps.
    """
    # ARRANGE
    from exozippy.samplers.ptde import polish_seed_starts

    pool = _CallbackPool(workers=4)
    seeds = [{"x": np.array([0.0])}, {"x": np.array([6.0])}]
    n_steps, pop = 7, 8

    # ACT
    with caplog.at_level(logging.INFO, logger="exozippy.samplers.ptde"):
        polish_seed_starts(
            seeds,
            _quadratic_logp,
            np.random.default_rng(3),
            {"x": np.ones(1)},
            n_steps=n_steps,
            pop_size=pop,
            pool=pool,
        )
    pool.release()

    # ASSERT
    assert pool.n_apply_async == len(seeds) * pop * (n_steps + 1)
    assert caplog.text.count(f"{n_steps} steps x {pop} pop") == len(seeds)


def test_async_de_polish_heartbeats_with_the_sweep_count(caplog):
    """
    Given a heartbeat interval every result crosses,
    When the asynchronous engine runs,
    Then progress lines carry the sweep count against the cap, elapsed and
      an upper-bound ETA, in the same shape the synchronous engine logs.
    """
    # ARRANGE
    from exozippy.samplers.ptde import polish_seed_starts

    pool = _CallbackPool(workers=2)

    # ACT
    with caplog.at_level(logging.INFO, logger="exozippy.samplers.ptde"):
        polish_seed_starts(
            [{"x": np.array([0.0])}],
            _quadratic_logp,
            np.random.default_rng(0),
            {"x": np.ones(1)},
            n_steps=6,
            pop_size=8,
            pool=pool,
            progress_interval_s=1e-9,
        )
    pool.release()
    beats = [
        r.getMessage()
        for r in caplog.records
        if "PTDE seed polish: sweep" in r.getMessage()
    ]

    # ASSERT
    assert beats
    assert any("/6" in b for b in beats)
    assert all("elapsed=" in b and "eta<=" in b for b in beats)


def test_async_de_polish_eval_timeout_writes_off_and_recycles(caplog):
    """
    Given a proposal whose evaluation never returns and an eval_timeout,
    When the asynchronous engine runs with a pool_recycler,
    Then the stale submission is scored -inf and logged, the recycler is
      called, the written-off legitimate work is resubmitted, and the polish
      still spends its full budget and returns a finite point.
    """
    # ARRANGE
    from exozippy.samplers.ptde import polish_seed_starts

    pop, n_steps = 8, 10
    hung = _CallbackPool(workers=2, hang={pop + 1})
    fresh = []

    def _recycle(dead):
        assert dead is hung or dead in fresh
        new = _CallbackPool(workers=2)
        fresh.append(new)
        return new

    # ACT
    with caplog.at_level(logging.INFO, logger="exozippy.samplers.ptde"):
        polished, dlps = polish_seed_starts(
            [{"x": np.array([0.0])}],
            _quadratic_logp,
            np.random.default_rng(5),
            {"x": np.ones(1)},
            n_steps=n_steps,
            pop_size=pop,
            pool=hung,
            eval_timeout=0.2,
            pool_recycler=_recycle,
        )
    hung.release()
    for p in fresh:
        p.release()

    # ASSERT
    assert fresh, "the recycler was never called"
    assert "exceeded eval_timeout" in caplog.text
    assert "recycling the worker pool" in caplog.text
    assert np.all(np.isfinite(polished[0]["x"]))
    assert f"{n_steps} steps x {pop} pop" in caplog.text
    assert dlps[0] > 0


def test_polish_raw_starts_forwards_the_engine_choice(monkeypatch):
    """
    Given polish_raw_starts on a gradient-free model,
    When it is called with asynchronous=False,
    Then the DE engine receives that flag: the pipeline can ask for the
      bit-reproducible synchronous engine without reaching into ptde.
    """
    # ARRANGE
    import exozippy.polish as polish_mod
    from exozippy.samplers import ptde as ptde_mod

    seen = {}

    def _spy(raw_starts, logp_fn, rng, scales, **kw):
        seen.update(kw)
        return list(raw_starts), [0.0 for _ in raw_starts]

    monkeypatch.setattr(ptde_mod, "polish_seed_starts", _spy)
    monkeypatch.setattr(polish_mod, "_compile_logp_grad", lambda model: None)

    class _Model:
        def compile_logp(self):
            return _quadratic_logp

    # ACT
    polish_mod.polish_raw_starts(
        _Model(),
        [{"x": np.array([0.0])}],
        n_steps=3,
        cores=1,
        asynchronous=False,
    )

    # ASSERT
    assert seen["asynchronous"] is False
