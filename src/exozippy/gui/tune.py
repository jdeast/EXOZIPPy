"""Solve + live-evaluator session for the GUI Tune tab (prompt G10).

The Tune tab implements a hybrid interaction model:

  * The user presses "Solve".  A worker runs G3 ``solve()`` (the relaxation
    engine, stages 1-3) then G5 ``compile_evaluator()`` (build the PyMC model
    and compile the plot predictors).  This is a seconds-scale step.
  * The app then enters LIVE mode: dragging a parameter slider posts a value,
    the evaluator inverts it to a raw point and re-renders every affected
    model curve in milliseconds.
  * A structural change (a bound / prior / fixed flag, or any G8 command that
    touches component wiring) changes the ``structural_hash`` and forces
    another Solve; the sliders go stale until then.

Because the pytensor compile + eval is CPU-heavy and must not stall the
FastAPI event loop, the evaluator lives in a DEDICATED WORKER PROCESS -- one
per open project.  :class:`EvaluatorWorker` owns that subprocess and speaks a
tiny request/response protocol over two multiprocessing queues.
:class:`TuneSession` drives it from the server side, tracking the
solving -> compiling -> live phase for the status endpoint.

This module is component-AGNOSTIC and imports nothing from FastAPI, so
``import exozippy`` (and the plain CLI) never touch it.  The heavy imports
(``System``, ``compile_evaluator``) happen only inside the worker child.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import threading
from typing import Callable, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Worker child: holds the System / model / Evaluator in memory
# ---------------------------------------------------------------------------

def _round_list(arr):
    """Convert a numpy array to a JSON-safe list (non-finite -> None)."""
    from exozippy.plotspec import _array_to_list

    return _array_to_list(arr)


def _do_solve(state, msg, resp_q):
    """Build System + model + Evaluator; return the panel + plot payload.

    Emits a ``{"progress": "compiling"}`` message once the relaxation engine
    (the "solving" half) has finished and the model build begins, so the
    parent can advance its phase indicator.
    """
    from exozippy.evaluator import compile_evaluator, structural_hash
    from exozippy.system import System

    config = msg["config"]
    params = msg["params"]
    workdir = msg["workdir"]

    prev_cwd = os.getcwd()
    if workdir:
        os.chdir(workdir)
    try:
        system = System(config, user_params=params)
        system.prepare()
        export = system.config_manager.export_solution()
        # Relaxation done; the seconds-scale compile begins now.
        resp_q.put({"progress": "compiling"})
        model = system.build_model()
        base_raw = system.get_raw_start(model)
        ev = compile_evaluator(system, model, base_raw)
    finally:
        os.chdir(prev_cwd)

    state["ev"] = ev
    state["raw"] = base_raw

    plots = [spec.to_json() for spec in ev.specs]
    return {
        "parameters": export["parameters"],
        "seeds": export.get("seeds"),
        "plots": plots,
        "structural_hash": structural_hash(config, params),
    }


def _do_eval(state, msg):
    """Apply one slider value to the retained raw point and re-render curves.

    Returns ``{"needs_resolve": True, ...}`` when the element has no static
    inverse (linked/dynamic bounds, or a fixed/derived parameter) so the GUI
    falls back to a re-solve.
    """
    from exozippy.evaluator import NeedsResolve

    ev = state.get("ev")
    if ev is None:
        raise RuntimeError("no evaluator; Solve first")

    raw = state["raw"]
    try:
        new_raw = ev.set_value(msg["path"], float(msg["value"]), raw)
    except NeedsResolve as exc:
        return {"needs_resolve": True, "reason": str(exc)}
    except ValueError as exc:
        return {"out_of_bounds": True, "reason": str(exc)}

    state["raw"] = new_raw
    out = ev.eval_plots(new_raw)
    plots = {
        pid: {name: _round_list(y) for name, y in traces.items()}
        for pid, traces in out.items()
    }
    return {"plots": plots}


def _worker_main(req_q, resp_q):
    """Entry point of the evaluator subprocess: a serve loop over the queues."""
    state: dict = {}
    while True:
        msg = req_q.get()
        op = msg.get("op")
        if op == "shutdown":
            break
        try:
            if op == "solve":
                result = _do_solve(state, msg, resp_q)
                resp_q.put({"ok": True, **result})
            elif op == "eval":
                result = _do_eval(state, msg)
                resp_q.put({"ok": True, **result})
            else:  # pragma: no cover - defensive
                resp_q.put({"ok": False, "error": f"unknown op '{op}'"})
        except Exception as exc:  # noqa: BLE001 - report, keep the loop alive
            logger.exception("tune worker error")
            resp_q.put({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


# ---------------------------------------------------------------------------
# Parent-side handle to the worker process
# ---------------------------------------------------------------------------

class EvaluatorWorker:
    """Owns the evaluator subprocess and its request/response queues.

    A spawn-context process is used deliberately: forking after pytensor /
    BLAS have initialised in the server process is unsafe, and spawn re-imports
    cleanly at the cost of a slower (seconds) startup that is dwarfed by the
    solve itself.
    """

    def __init__(self):
        self._ctx = multiprocessing.get_context("spawn")
        self._req_q = self._ctx.Queue()
        self._resp_q = self._ctx.Queue()
        self._proc = None

    def start(self):
        if self._proc is not None and self._proc.is_alive():
            return
        self._proc = self._ctx.Process(
            target=_worker_main, args=(self._req_q, self._resp_q), daemon=True
        )
        self._proc.start()

    def solve(self, config, params, workdir, on_progress=None):
        """Run a full solve; block for the result, forwarding progress states."""
        self._req_q.put(
            {"op": "solve", "config": config, "params": params,
             "workdir": workdir}
        )
        return self._await(on_progress)

    def set_and_eval(self, path, value):
        """Move one parameter and return the updated model traces."""
        self._req_q.put({"op": "eval", "path": path, "value": value})
        return self._await(None)

    def _await(self, on_progress):
        while True:
            msg = self._resp_q.get()
            if "progress" in msg:
                if on_progress:
                    on_progress(msg["progress"])
                continue
            if not msg.get("ok"):
                raise RuntimeError(msg.get("error", "worker error"))
            return msg

    def close(self):
        if self._proc is None:
            return
        try:
            self._req_q.put({"op": "shutdown"})
            self._proc.join(timeout=2.0)
        except Exception:  # pragma: no cover - best effort
            pass
        if self._proc.is_alive():
            self._proc.terminate()
        self._proc = None


# ---------------------------------------------------------------------------
# Server-side session (one per open project)
# ---------------------------------------------------------------------------

class TuneSession:
    """Tracks the Tune tab's solve state and brokers eval calls.

    ``worker_factory`` is injectable so tests can stub the (heavy) evaluator
    without a real pytensor compile; the default is looked up at solve time so
    monkeypatching ``exozippy.gui.tune.EvaluatorWorker`` still takes effect.
    """

    def __init__(self, worker_factory: Optional[Callable[[], object]] = None):
        self._worker_factory = worker_factory
        self._worker = None
        self._lock = threading.Lock()
        self.phase = "idle"          # idle|solving|compiling|live|error
        self.error: Optional[str] = None
        self.structural_hash: Optional[str] = None
        self.result: Optional[dict] = None   # {parameters, seeds, plots}

    def solve(self, config, params, workdir):
        """Blocking solve (call from a background thread). Sets phase/result."""
        factory = self._worker_factory or EvaluatorWorker
        self.phase = "solving"
        self.error = None
        try:
            if self._worker is not None:
                try:
                    self._worker.close()
                except Exception:  # pragma: no cover
                    pass
            worker = factory()
            worker.start()
            self._worker = worker

            def _progress(state):
                self.phase = state

            res = worker.solve(config, params, workdir, on_progress=_progress)
            self.result = {
                "parameters": res["parameters"],
                "seeds": res.get("seeds"),
                "plots": res["plots"],
            }
            self.structural_hash = res["structural_hash"]
            self.phase = "live"
        except Exception as exc:  # noqa: BLE001 - surfaced to the status endpoint
            logger.exception("tune solve failed")
            self.error = f"{type(exc).__name__}: {exc}"
            self.phase = "error"
        return self.phase

    def eval(self, path, value):
        """Move one parameter (LIVE mode only) and return updated traces."""
        if self._worker is None or self.phase != "live":
            raise RuntimeError("no live evaluator; press Solve first")
        with self._lock:
            return self._worker.set_and_eval(path, value)

    def status(self):
        return {
            "phase": self.phase,
            "error": self.error,
            "structural_hash": self.structural_hash,
            "has_result": self.result is not None,
        }

    def close(self):
        if self._worker is not None:
            try:
                self._worker.close()
            except Exception:  # pragma: no cover
                pass
            self._worker = None
