"""The seed ledger's curvature probes on a worker pool (review 2.6.43).

build_seed_ledger re-probes every polished seed: n_seeds x n_elements x
O(15) logp evaluations.  It ran them serially in the main process, untimed
and silent, so OGLE-2013-BLG-0341 (~225 s per three-lens logp, 62 raw
elements, 112 cores) sat 19.5 h in it with one core busy -- an estimated
116-175 h against a 72 h wall.  It now shares the whitening probe's
machinery (whitening.probe_points).  What is pinned here:

- pooled and serial ledgers are BIT-IDENTICAL (a toy and shipped examples);
- a probe call that outlives eval_timeout costs only its element, which
  records the ledger's failed-probe width (1 raw unit) with a WARNING
  naming the seed and element -- never a hang;
- the ledger says how far along it is;
- run.py and the hot-chain discovery hand it the sampler's grant and
  eval_timeout.
"""

import ast
import copy
import inspect
import logging
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pymc as pm
import pytest
import yaml

import exozippy.whitening as wmod
from exozippy.components.parameter import Parameter
from exozippy.outputs.ledger import (
    build_seed_ledger,
    ledger_from_json,
    ledger_to_json,
)

EXAMPLES = Path(__file__).parent.parent / "examples"

needs_cores = pytest.mark.skipif(
    mp.cpu_count() < 3, reason="the pooled probe needs >= 2 worker cores"
)


class _StubSystem:
    def __init__(self, params):
        self._params = params

    def get_all_parameters(self):
        return self._params


def _toy():
    px = Parameter(
        label="toy.x", initval=2.0, lower=0.0, upper=10.0, init_scale=1.0
    )
    py = Parameter(
        label="toy.y", initval=-1.0, lower=-5.0, upper=5.0, init_scale=1.0
    )
    with pm.Model() as model:
        xv = px.build_pymc()
        yv = py.build_pymc()
        pm.Potential(
            "like",
            -0.5 * ((xv - 2.5) / 0.3) ** 2 - 0.5 * ((yv + 1.2) / 2.0) ** 2,
        )
    return model, _StubSystem([px, py])


def _two_seeds(model):
    s0 = model.initial_point()
    # A non-dyadic offset: no bracket/bisect step from seed 0 lands on seed 1.
    s1 = {k: np.asarray(v, dtype=float) + 0.2371 for k, v in s0.items()}
    return [s0, s1]


def _assert_ledgers_identical(a, b):
    # The JSON text is the persisted form (trace_meta.SEED_LEDGER_ATTR); equal
    # text is equal records, every float to the bit.
    assert ledger_to_json(a) == ledger_to_json(b)
    assert len(a) == len(b) and a
    for ra, rb in zip(a, b):
        assert ra.lp_max == rb.lp_max
        assert ra.laplace_logw == rb.laplace_logw
        for k in ra.raw_scales:
            np.testing.assert_array_equal(ra.raw_scales[k], rb.raw_scales[k])


@needs_cores
def test_pooled_ledger_is_bit_identical_to_the_serial_ledger():
    """
    Given a toy model and two seeds,
    When the ledger is built serially and on a pool (eval_timeout set,
      which always takes the pool),
    Then the two ledgers are bit-identical.
    """
    # ARRANGE
    model, system = _toy()
    seeds = _two_seeds(model)

    # ACT
    serial = build_seed_ledger(system, model, seeds, [0, 1])
    pooled = build_seed_ledger(
        system, model, seeds, [0, 1], cores=3, eval_timeout=60.0
    )

    # ASSERT
    _assert_ledgers_identical(serial, pooled)
    assert all(np.all(r.raw_scales["toy.x_raw"] != 1.0) for r in serial)


class _HangOnSeed1Y:
    """Hangs whenever toy.y_raw is perturbed around seed 1 (toy.x_raw at
    seed 1's value, toy.y_raw not)."""

    def __init__(self, logp, seed1):
        self.logp = logp
        self.x1 = float(np.asarray(seed1["toy.x_raw"]).ravel()[0])
        self.y1 = float(np.asarray(seed1["toy.y_raw"]).ravel()[0])

    def __call__(self, point):
        x = float(np.asarray(point["toy.x_raw"]).ravel()[0])
        y = float(np.asarray(point["toy.y_raw"]).ravel()[0])
        if x == self.x1 and y != self.y1:
            time.sleep(60.0)  # a "hang": far past eval_timeout
        return self.logp(point)


@needs_cores
def test_a_ledger_probe_past_eval_timeout_is_a_named_fallback(caplog):
    """
    Given a logp that hangs (60 s) whenever seed 1's toy.y is perturbed,
    When the ledger is built on a pool with eval_timeout=0.5 s,
    Then it returns within a bound; seed 1's toy.y width is the ledger's
      failed-probe width (1 raw unit); a WARNING names "seed 1
      toy.y_raw[0]"; every other width equals the serial ledger's exactly.
    """
    # ARRANGE
    model, system = _toy()
    seeds = _two_seeds(model)
    ref = build_seed_ledger(system, model, seeds, [0, 1])
    logp = _HangOnSeed1Y(model.compile_logp(), seeds[1])

    # ACT
    t0 = time.monotonic()
    with caplog.at_level(logging.WARNING, logger="exozippy.whitening"):
        led = build_seed_ledger(
            system,
            model,
            seeds,
            [0, 1],
            logp_fn=logp,
            cores=3,
            eval_timeout=0.5,
        )
    wall = time.monotonic() - t0

    # ASSERT
    assert wall < 30.0, f"took {wall:.1f} s against a 0.5 s timeout"
    np.testing.assert_array_equal(led[1].raw_scales["toy.y_raw"], [1.0])
    assert ref[1].raw_scales["toy.y_raw"][0] != 1.0  # the fallback fired
    np.testing.assert_array_equal(
        led[1].raw_scales["toy.x_raw"], ref[1].raw_scales["toy.x_raw"]
    )
    for k in ref[0].raw_scales:
        np.testing.assert_array_equal(
            led[0].raw_scales[k], ref[0].raw_scales[k]
        )
    assert led[1].lp_max == ref[1].lp_max
    assert "seed 1 toy.y_raw[0]" in caplog.text
    assert "exceeded eval_timeout" in caplog.text
    assert "Seed ledger" in caplog.text


@pytest.mark.parametrize(
    "cores", [1, pytest.param(3, marks=needs_cores)], ids=["serial", "pool"]
)
def test_the_ledger_reports_its_progress(cores, monkeypatch, caplog):
    """
    Given a progress interval every element crosses,
    When the ledger is built (serially, and on a pool),
    Then it logs "Seed ledger: k/N element(s) done" lines and closes on N/N.
    """
    # ARRANGE
    monkeypatch.setattr(wmod, "PROBE_PROGRESS_S", 0.0)
    model, system = _toy()
    kw = {} if cores == 1 else {"cores": cores, "eval_timeout": 60.0}

    # ACT
    with caplog.at_level(logging.INFO, logger="exozippy.whitening"):
        build_seed_ledger(system, model, _two_seeds(model), [0, 1], **kw)

    # ASSERT
    lines = [
        r.getMessage()
        for r in caplog.records
        if r.getMessage().startswith("Seed ledger: ")
        and "element(s) done" in r.getMessage()
    ]
    assert lines, caplog.text
    assert lines[-1].startswith("Seed ledger: 4/4 element(s) done")
    assert "done." in lines[-1]


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


@needs_cores
@pytest.mark.slow
@pytest.mark.parametrize(
    "example,config",
    [("kelt4", "kelt4_rvonly.yaml"), ("ob140939", "ob140939.yaml")],
)
def test_shipped_example_ledger_is_bit_identical_on_the_pool(
    monkeypatch, example, config
):
    """
    Given a shipped example's build start and a displaced copy as two seeds,
    When the ledger is built serially and on a pool (cores=3, eval_timeout),
    Then the ledgers -- and their persisted JSON -- are bit-identical.
    """
    # ARRANGE
    system, model = _build_example(monkeypatch, example, config)
    s0 = system.get_raw_start(model)
    s1 = {k: np.asarray(v, dtype=float) + 0.1 for k, v in s0.items()}

    # ACT
    serial = build_seed_ledger(system, model, [s0, s1], [0, 1])
    pooled = build_seed_ledger(
        system, model, [s0, s1], [0, 1], cores=3, eval_timeout=600.0
    )

    # ASSERT
    _assert_ledgers_identical(serial, pooled)
    _assert_ledgers_identical(ledger_from_json(ledger_to_json(serial)), pooled)


def test_every_ledger_call_passes_the_grant_and_eval_timeout():
    """
    Given run.py's and outputs/ledger.py's source,
    When every call to build_seed_ledger is read,
    Then each passes `cores=` and `eval_timeout=` -- the pre-sampling ledger
      and the hot-chain discovery's.  Dropping either puts the serial,
      untimed ledger back.
    """
    import exozippy.outputs.ledger as ledger_mod
    import exozippy.run as run_mod

    calls, missing = [], []
    for mod in (run_mod, ledger_mod):
        tree = ast.parse(Path(inspect.getfile(mod)).read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "id", getattr(node.func, "attr", None))
            if name != "build_seed_ledger":
                continue
            calls.append(f"{mod.__name__}:{node.lineno}")
            kws = {kw.arg for kw in node.keywords}
            if not {"cores", "eval_timeout"} <= kws:
                missing.append(f"{mod.__name__}:{node.lineno}")
    assert len(calls) == 2, calls
    assert not missing, missing
