"""The startup whitening probe on a worker pool (review 2.6.41) and the seed
polish under eval_timeout (review 2.6.35).

OGLE-2013-BLG-0341 (~225 s per three-lens logp, 112 cores) spent 12.85 h in
the serial, untimed, silent probe and ~10.75 h draining one straggling
polish call past the polish's own budget.  What is pinned here:

- the pooled probe returns BIT-IDENTICAL scales to the serial one (a toy
  model and a shipped example), so moving it onto the pool changes nothing
  but the wall clock;
- a probe evaluation that outlives eval_timeout abandons its element, which
  keeps its preliminary scale with a WARNING naming it -- the probe's
  existing failed-probe contract -- instead of hanging the run;
- the probe says how far along it is;
- a DE polish whose objective hangs past eval_timeout returns within a
  bound, its seed stopped on the polish's timeout, and the drain after the
  last seed stops is not silent.
"""

import copy
import logging
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest
import yaml
from pytensor.graph.op import Op

import exozippy.whitening as wmod
from exozippy.components.parameter import Parameter
from exozippy.whitening import (
    apply_measured_whitening,
    measure_and_whiten,
    probe_scales,
)

EXAMPLES = Path(__file__).parent.parent / "examples"

needs_cores = pytest.mark.skipif(
    mp.cpu_count() < 3, reason="the pooled probe needs >= 2 worker cores"
)


def _aniso_logp(widths, center):
    widths = np.asarray(widths, dtype=float)
    center = np.asarray(center, dtype=float)

    def logp(point):
        x = np.asarray(point["a"], dtype=float)
        lp = -0.5 * np.sum(((x - center) / widths) ** 2)
        # A hard wall on one element and a flat one, so every branch of the
        # probe (curvature, wall-bounded, flat) is exercised.
        if point["b"][0] > 3.0:
            return -np.inf
        return float(lp - 0.5 * (point["b"][0] / 0.7) ** 2)

    return logp


def _start():
    return {"a": np.array([0.3, -2.0, 5.0, 0.0]), "b": np.array([0.1, 0.0])}


# ---------------------------------------------------------------------------
# Bit-identity: the pool changes the wall clock and nothing else.
# ---------------------------------------------------------------------------


@needs_cores
def test_pooled_probe_is_bit_identical_to_the_serial_probe():
    """
    Given an anisotropic logp with a curved, a wall-bounded and a flat
      direction,
    When probe_scales runs serially and then on a pool (eval_timeout set,
      which always takes the pool),
    Then every scale, the start lp and every diagnostic list are EXACTLY
      equal -- the same searches, the same steps, the same numbers.
    """
    # ARRANGE
    logp = _aniso_logp([0.01, 3.0, 1e-4, 50.0], [0.2, -1.0, 5.0, 0.0])
    diag_s, diag_p = {}, {}

    # ACT
    lp_s, sc_s = probe_scales(
        _start(), logp, flat_scale=np.nan, diagnostics=diag_s
    )
    lp_p, sc_p = probe_scales(
        _start(),
        logp,
        flat_scale=np.nan,
        diagnostics=diag_p,
        cores=3,
        eval_timeout=60.0,
    )

    # ASSERT
    assert lp_s == lp_p
    for k in sc_s:
        np.testing.assert_array_equal(sc_s[k], sc_p[k])
    assert diag_s == diag_p
    assert diag_p["timed_out"] == []


def _build_example(monkeypatch, example, config_name):
    from exozippy.system import System

    d = EXAMPLES / example
    config = yaml.safe_load((d / config_name).read_text())
    params = yaml.safe_load((d / config["parameter_file"]).read_text()) or {}
    config["parameter_file"] = None
    monkeypatch.chdir(d)
    system = System(copy.deepcopy(config), copy.deepcopy(params))
    system.prepare()
    model = system.build_model()
    return system, model


def _whitening_state(system):
    out = {}
    for p in system.get_all_parameters():
        st = p.export_whitening()
        if st:
            out[p.label] = {k: np.asarray(v) for k, v in st.items()}
    return out


@needs_cores
def test_shipped_example_whitening_is_bit_identical_on_the_pool(monkeypatch):
    """
    Given examples/kelt4's RV-only fit, built twice from the same config,
    When one build is whitened by the serial probe and the other by the
      pooled one (cores=3, eval_timeout set),
    Then the persisted whitening state (scales, anchors, barrier scales),
      the start lp, the multipliers and the start the sampler receives are
      bit-identical.
    """
    # ARRANGE
    sys_s, model_s = _build_example(monkeypatch, "kelt4", "kelt4_rvonly.yaml")
    sys_p, model_p = _build_example(monkeypatch, "kelt4", "kelt4_rvonly.yaml")

    # ACT
    rep_s = measure_and_whiten(sys_s, model_s, sys_s.get_raw_start(model_s))
    rep_p = measure_and_whiten(
        sys_p,
        model_p,
        sys_p.get_raw_start(model_p),
        cores=3,
        eval_timeout=600.0,
    )

    # ASSERT
    assert rep_s["map_lp"] == rep_p["map_lp"]
    assert rep_s["multipliers"].keys() == rep_p["multipliers"].keys()
    for k in rep_s["multipliers"]:
        np.testing.assert_array_equal(
            rep_s["multipliers"][k], rep_p["multipliers"][k]
        )
    st_s, st_p = _whitening_state(sys_s), _whitening_state(sys_p)
    assert st_s.keys() == st_p.keys() and st_s
    for label in st_s:
        assert st_s[label].keys() == st_p[label].keys()
        for k in st_s[label]:
            np.testing.assert_array_equal(st_s[label][k], st_p[label][k])
    start_s, start_p = (
        sys_s.get_raw_start(model_s),
        sys_p.get_raw_start(model_p),
    )
    for k in start_s:
        np.testing.assert_array_equal(start_s[k], start_p[k])
    assert float(model_s.compile_logp()(start_s)) == float(
        model_p.compile_logp()(start_p)
    )


# ---------------------------------------------------------------------------
# eval_timeout: a stuck evaluation abandons its element, never the run.
# ---------------------------------------------------------------------------


def _hangs_on_b(base):
    def logp(point):
        if point["b"][1] != 0.0:
            time.sleep(60.0)  # a "hang": far past eval_timeout
        return base(point)

    return logp


@needs_cores
def test_a_probe_call_past_eval_timeout_abandons_only_its_element(caplog):
    """
    Given a logp that hangs (60 s) whenever element b[1] is perturbed,
    When probe_scales runs on a pool with eval_timeout=0.5 s,
    Then it returns within a bound, b[1] is reported timed out (and NOT
      flat) with a WARNING naming it, it gets flat_scale, and every other
      element's scale equals the serial probe's exactly.
    """
    # ARRANGE
    base = _aniso_logp([0.01, 3.0, 1e-4, 50.0], [0.2, -1.0, 5.0, 0.0])
    diag = {}
    _, ref = probe_scales(_start(), base, flat_scale=np.nan)

    # ACT
    t0 = time.monotonic()
    with caplog.at_level(logging.WARNING, logger="exozippy.whitening"):
        _, sc = probe_scales(
            _start(),
            _hangs_on_b(base),
            flat_scale=np.nan,
            diagnostics=diag,
            cores=3,
            eval_timeout=0.5,
        )
    wall = time.monotonic() - t0

    # ASSERT
    assert wall < 20.0, f"the probe took {wall:.1f} s against a 0.5 s timeout"
    assert diag["timed_out"] == ["b[1]"]
    assert "b[1]" not in diag["flat"]
    assert np.isnan(sc["b"][1])
    assert sc["b"][0] == ref["b"][0]
    np.testing.assert_array_equal(sc["a"], ref["a"])
    assert "b[1]" in caplog.text
    assert "exceeded eval_timeout" in caplog.text


class _HangOnY:
    """A logp wrapper that hangs whenever toy.y_raw leaves the start."""

    def __init__(self, logp, y0):
        self.logp, self.y0 = logp, y0

    def __call__(self, point):
        if float(np.asarray(point["toy.y_raw"]).ravel()[0]) != self.y0:
            time.sleep(60.0)
        return self.logp(point)


class _StubSystem:
    def __init__(self, params):
        self._params = params

    def get_all_parameters(self):
        return self._params

    def get_raw_start(self, model):
        return model.initial_point()


def _toy():
    px = Parameter(
        label="toy.x", initval=2.0, lower=0.0, upper=10.0, init_scale=1.0
    )
    # Preliminary scale 1e-12 of a 10-wide span: a probe that ran would
    # escalate it by orders of magnitude, so "kept its preliminary scale"
    # is unmistakable.
    py = Parameter(
        label="toy.y", initval=-1.0, lower=-5.0, upper=5.0, init_scale=1e-12
    )
    with pm.Model() as model:
        xv = px.build_pymc()
        yv = py.build_pymc()
        pm.Potential(
            "like",
            -0.5 * ((xv - 2.5) / 0.01) ** 2 - 0.5 * ((yv + 1.2) / 2.0) ** 2,
        )
    return model, px, py


@needs_cores
def test_a_timed_out_element_keeps_its_preliminary_scale(caplog):
    """
    Given a toy model whose logp hangs whenever toy.y is perturbed, and
      whose toy.y preliminary scale is 12 orders too tight (a probe that ran
      would escalate it),
    When apply_measured_whitening runs on a pool with eval_timeout=0.5 s,
    Then it returns within a bound; toy.y keeps its preliminary scale
      exactly, is not escalated and is not called flat; a WARNING names it;
      and toy.x is whitened exactly as the serial probe whitens it.
    """
    # ARRANGE
    model, px, py = _toy()
    start = model.initial_point()
    y0 = float(np.asarray(start["toy.y_raw"]).ravel()[0])
    logp = _HangOnY(model.compile_logp(), y0)
    y_before = py._whiten_state["sv_scale_logits"].get_value().copy()

    ref_model, rx, _ry = _toy()
    apply_measured_whitening(
        _StubSystem([rx, _ry]), ref_model, ref_model.initial_point()
    )
    caplog.clear()  # the reference run escalates toy.y; that is not under test

    # ACT
    t0 = time.monotonic()
    with caplog.at_level(logging.WARNING, logger="exozippy.whitening"):
        report = apply_measured_whitening(
            _StubSystem([px, py]),
            model,
            start,
            logp_fn=logp,
            cores=3,
            eval_timeout=0.5,
        )
    wall = time.monotonic() - t0

    # ASSERT
    assert wall < 30.0, f"took {wall:.1f} s against a 0.5 s timeout"
    np.testing.assert_array_equal(
        py._whiten_state["sv_scale_logits"].get_value(), y_before
    )
    assert report["probe_diagnostics"]["timed_out"] == ["toy.y_raw[0]"]
    assert report["probe_diagnostics"]["flat"] == []
    np.testing.assert_array_equal(report["multipliers"]["toy.y_raw"], [1.0])
    assert "escalation" not in caplog.text
    assert "toy.y_raw[0]" in caplog.text
    assert "preliminary" in caplog.text
    np.testing.assert_array_equal(
        px._whiten_state["sv_scale_logits"].get_value(),
        rx._whiten_state["sv_scale_logits"].get_value(),
    )


# ---------------------------------------------------------------------------
# Progress: elements done / total, evaluations, ETA.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cores", [1, pytest.param(3, marks=needs_cores)], ids=["serial", "pool"]
)
def test_the_probe_reports_its_progress(cores, monkeypatch, caplog):
    """
    Given a progress interval every element crosses,
    When probe_scales runs (serially, and on a pool),
    Then it logs "k/N element(s) done" lines with the evaluation count and
      an ETA, and closes on N/N "done".
    """
    # ARRANGE
    monkeypatch.setattr(wmod, "PROBE_PROGRESS_S", 0.0)
    logp = _aniso_logp([0.01, 3.0, 1e-4, 50.0], [0.2, -1.0, 5.0, 0.0])
    kw = {} if cores == 1 else {"cores": cores, "eval_timeout": 60.0}

    # ACT
    with caplog.at_level(logging.INFO, logger="exozippy.whitening"):
        probe_scales(_start(), logp, **kw)
    lines = [
        r.getMessage()
        for r in caplog.records
        if r.getMessage().startswith("Whitening probe: ")
        and "element(s) done" in r.getMessage()
    ]

    # ASSERT
    assert lines
    assert all("logp evaluation(s)" in m and "elapsed" in m for m in lines)
    assert any("ETA ~" in m for m in lines)
    assert lines[-1].startswith("Whitening probe: 6/6 element(s) done")
    assert lines[-1].rstrip(".").endswith("done")


def test_a_fast_probe_logs_no_progress(caplog):
    """
    Given the default interval and a microsecond logp,
    When probe_scales runs,
    Then no progress line is logged -- a fast model's log stays quiet.
    """
    logp = _aniso_logp([0.01, 3.0, 1e-4, 50.0], [0.2, -1.0, 5.0, 0.0])
    with caplog.at_level(logging.INFO, logger="exozippy.whitening"):
        probe_scales(_start(), logp)
    assert "element(s) done" not in caplog.text


# ---------------------------------------------------------------------------
# The seed polish under eval_timeout (review 2.6.35).
# ---------------------------------------------------------------------------


class _NoGradSquare(Op):
    """-x^2/2 with no gradient: the DE engine's model."""

    itypes = [pt.dscalar]
    otypes = [pt.dscalar]

    def perform(self, node, inputs, outputs):
        (x,) = inputs
        outputs[0][0] = np.asarray(-0.5 * float(x) ** 2)


# Shared with forked workers (created before any pool forks, inherited by
# every recycled one): counts evaluations across processes, so exactly ONE
# call -- the K-th -- hangs.
_CALLS = None


class _HangOnce:
    def __init__(self, logp, k):
        self.logp, self.k = logp, k

    def __call__(self, point):
        with _CALLS.get_lock():
            _CALLS.value += 1
            n = _CALLS.value
        if n == self.k:
            time.sleep(120.0)
        time.sleep(0.002)
        return self.logp(point)


@needs_cores
def test_a_polish_call_past_eval_timeout_is_written_off_and_the_seed_stops(
    monkeypatch, caplog
):
    """
    Given the DE polish on 2 workers, a polish budget that runs out at the
      end of the first sweep (set from the monitor, not a clock), and an
      objective whose 10th call -- the second of the first sweep, after the
      8-member opening batch -- sleeps 120 s (the OGLE-2013-BLG-0341
      straggler),
    When polish_raw_starts runs with eval_timeout=3 s,
    Then the other 7 slots complete sweep 1, the driver stops the seed "on
      timeout" with the hung call still in flight -- so the drain is
      guaranteed, not a race -- the drain logs "draining N in-flight", the
      hung call is written off with an ERROR, and it all returns within a
      bound (not after 120 s).  The polish really ran: the dlp is finite.
      (On macOS / Python 3.14 the pool used to start by spawn / forkserver,
      its workers never saw the logp, every call scored -inf and the seed
      came back "lp -inf -> nan" -- the dlp check pins that.)
    """
    # ARRANGE
    global _CALLS
    import exozippy.polish as polish_mod
    from exozippy.polish import STOP_TIMEOUT, polish_raw_starts

    _CALLS = mp.get_context("fork").Value("i", 0)

    class _Chatty(polish_mod.PolishMonitor):
        def __init__(self, *a, **kw):
            kw["progress_interval_s"] = 1e-9  # every beat writes a line
            super().__init__(*a, **kw)

        def step(self, s, best_lp):
            # The budget runs out exactly at the end of sweep 1: no clock
            # race decides whether the hung call is still in flight.
            self.deadline = time.monotonic() - 1.0
            return super().step(s, best_lp)

    monkeypatch.setattr(polish_mod, "PolishMonitor", _Chatty)
    with pm.Model() as model:
        x = pm.Flat("x")
        pm.Potential("like", _NoGradSquare()(x))
    logp = _HangOnce(model.compile_logp(), k=10)
    stops = []

    # ACT
    t0 = time.monotonic()
    with caplog.at_level(logging.INFO):
        polished, dlps, method = polish_raw_starts(
            model,
            [{"x": np.array(4.0)}],
            n_steps=10**6,
            logp_fn=logp,
            cores=2,
            eval_timeout=3.0,
            deadline=time.monotonic() + 3600.0,
            budget_s=1.0,
            stops_out=stops,
        )
    wall = time.monotonic() - t0

    # ASSERT
    assert method == "de"
    assert wall < 60.0, f"the polish took {wall:.1f} s"
    assert stops[0][0] == STOP_TIMEOUT
    assert "exceeded eval_timeout=3s" in caplog.text
    assert "draining" in caplog.text and "in-flight" in caplog.text
    assert np.isfinite(polished[0]["x"]).all()
    assert np.isfinite(dlps[0]) and dlps[0] >= 0.0
    assert "raised" not in caplog.text  # workers saw the compiled logp


def test_run_forwards_eval_timeout_to_every_untimed_stage():
    """
    Given run.py and polish.py's source,
    When the calls into the stages 2.6.35 / 2.6.41 found untimed are read,
    Then every one of them passes `eval_timeout=` -- the pre-sampling polish
      (polish_rounds), the startup probe (prepare_whitening), the hot-mode
      polish at wrap-up (run_hot_mode_discovery), and inside polish_rounds
      each round's polish and its between-round probe.  Dropping one puts an
      untimed, minutes-per-call stage back between the user and the sampler.
    """
    import ast
    import inspect

    import exozippy.polish as polish_mod
    import exozippy.run as run_mod

    want = {
        run_mod: {
            "polish_rounds",
            "prepare_whitening",
            "run_hot_mode_discovery",
        },
        polish_mod: {"polish_raw_starts", "measure_and_whiten"},
    }
    missing, seen = [], set()
    for mod, names in want.items():
        tree = ast.parse(Path(inspect.getfile(mod)).read_text())
        scope = tree
        if mod is polish_mod:
            scope = next(
                n
                for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "polish_rounds"
            )
        for node in ast.walk(scope):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "id", getattr(node.func, "attr", None))
            if name not in names:
                continue
            seen.add(name)
            if "eval_timeout" not in {kw.arg for kw in node.keywords}:
                missing.append(f"{mod.__name__}:{node.lineno} {name}")
    assert seen == set().union(*want.values()), seen
    assert not missing, missing
