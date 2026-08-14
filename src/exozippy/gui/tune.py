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
tiny request/response protocol over two multiprocessing queues: every request
carries a uuid the worker echoes on every message it sends back, and each
caller waits only on its own id, so a slider eval and a Solve can be in flight
together without stealing each other's replies.  A worker that goes silent past
a generous deadline is terminated and respawned rather than wedging the tab.
:class:`TuneSession` drives it from the server side, tracking the
solving -> compiling -> live phase for the status endpoint.

This module is component-AGNOSTIC and imports nothing from FastAPI, so
``import exozippy`` (and the plain CLI) never touch it.  The heavy imports
(``System``, ``compile_evaluator``) happen only inside the worker child.
"""

from __future__ import annotations

import gc
import logging
import multiprocessing
import os
import queue
import signal
import sys
import threading
import time
import uuid
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# How long the parent waits on the response queue before re-checking that the
# worker process is still alive (see EvaluatorWorker._reader_loop / _take).
_AWAIT_POLL_S = 0.25

# SILENCE deadlines: how long the parent will wait with NO message at all from
# the worker before declaring it wedged, terminating it and respawning (see
# EvaluatorWorker._await).  Each progress message restarts the clock, so these
# bound one silent stretch, not the whole job.
#
# They are deliberately generous, because a false positive is far worse than a
# late detection: terminating a healthy worker throws away a compile that was
# about to finish and looks, from the UI, exactly like the bug it is meant to
# fix.  A legitimate first Solve is silent for as long as pytensor takes to
# build and compile the model on a cold cache -- tens of seconds routinely, and
# minutes for a big SED/transit topology -- so SOLVE gets 15 minutes, about an
# order of magnitude above the worst legitimate silence.  An eval is
# milliseconds by design, but the first one after a Solve can lazily compile
# plotters (the GP conditional means, the outlier probabilities), so it gets 2
# minutes rather than something that looks tight next to a slider drag.
#
# Read at call time (never captured at import) so a slow machine -- or a test
# -- can monkeypatch them.
SOLVE_TIMEOUT_S = 900.0
EVAL_TIMEOUT_S = 120.0


class WorkerTimeout(RuntimeError):
    """The worker went silent past its deadline and was terminated.

    A subclass of ``RuntimeError`` so every existing handler (``TuneSession``'s
    solve wrapper, ``app.py``'s ``/api/tune/eval``) already surfaces it.
    """


class _DeadlineExpired(Exception):
    """Internal: no message for this request within its silence deadline."""


# ---------------------------------------------------------------------------
# Worker child: holds the System / model / Evaluator in memory
# ---------------------------------------------------------------------------


def _round_list(arr):
    """Convert a numpy array to a JSON-safe list (non-finite -> None)."""
    from exozippy.plotspec import _array_to_list

    return _array_to_list(arr)


def _data_only_plots(system):
    """Data-only PlotSpec JSON from every data-bearing component.

    ``plot_data(point=None)`` is valid right after ``prepare()``, before any
    model exists -- so the GUI can draw the observations while the
    seconds-scale compile runs, with the model traces patched in at "live".
    """
    plots = []
    for comp in system.active_components.values():
        try:
            comp_specs = comp.plot_data(system, None)
        except Exception as exc:  # noqa: BLE001 - a component may lack data
            logger.warning(
                "tune: data-only plot_data failed for %s: %s",
                getattr(comp, "prefix", comp),
                exc,
            )
            continue
        plots.extend(spec.to_json() for spec in comp_specs)
    return plots


def _do_solve(state, msg, resp_q):
    """Build System + model + Evaluator; return the panel + plot payload.

    Emits a ``{"progress": "compiling", "data_plots": [...]}`` message once
    the relaxation engine (the "solving" half) has finished and the model
    build begins, so the parent can advance its phase indicator and render
    the data-only plots while the compile runs.  Like every response, it
    echoes the request's ``id`` so the parent can route it back to the caller
    that is waiting for it (see EvaluatorWorker).
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
        # Release the previous solve's System/model/evaluator BEFORE building the
        # new one. This worker process is reused across solves (to keep the heavy
        # imports and compile cache warm), so without this the old objects -- a
        # full System with SED spectral grids plus compiled pytensor functions --
        # stay alive through the entire next build, roughly doubling peak memory
        # and accumulating across re-Solves until the machine runs out of RAM and
        # thrashes. Dropping the references and collecting caps us at one System.
        state.pop("ev", None)
        state.pop("raw", None)
        gc.collect()

        system = System(config, user_params=params)
        system.prepare()
        export = system.config_manager.export_solution(
            derived_params=system.derived_params()
        )
        # Relaxation done; the seconds-scale compile begins now. Ship the
        # data-only plots along so the GUI has something to draw meanwhile.
        resp_q.put(
            {
                "id": msg.get("id"),
                "progress": "compiling",
                "data_plots": _data_only_plots(system),
            }
        )
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
        label = ev.label_for_path(msg["path"])
        new_raw = ev.set_value(msg["path"], float(msg["value"]), raw)
    except NeedsResolve as exc:
        return {"needs_resolve": True, "reason": str(exc)}
    except ValueError as exc:
        return {"out_of_bounds": True, "reason": str(exc)}

    state["raw"] = new_raw
    out = ev.eval_plots(new_raw, changed_label=label)
    plots = {}
    for pid, traces in out.items():
        packed = {}
        for name, xy in traces.items():
            entry = {"x": _round_list(xy["x"]), "y": _round_list(xy["y"])}
            # dynamic_data specs re-ship their data traces, whose errors can
            # move too (mulens re-aligns every data set onto the reference
            # instrument's flux system and then plots delta-magnitudes).
            if xy.get("yerr") is not None:
                entry["yerr"] = _round_list(xy["yerr"])
            packed[name] = entry
        plots[pid] = packed
    return {"plots": plots}


def _install_parent_death_signal():
    """On Linux, ask the kernel to SIGKILL this child when its parent dies.

    The worker holds ~800 MB (a full System + compiled pytensor + SED grids).
    ``daemon=True`` reaps it on a clean parent exit, but NOT when the server is
    hard-killed (SIGKILL / crash): the child is reparented to init and keeps its
    memory forever. Enough of those orphans accumulate to swap-thrash the whole
    machine. PR_SET_PDEATHSIG (prctl option 1) makes the kernel send us SIGKILL
    the moment the parent dies, however it dies. Best-effort and Linux-only;
    also handle the race where the parent already died before we armed it.
    """
    if not sys.platform.startswith("linux"):
        return
    try:
        import ctypes

        PR_SET_PDEATHSIG = 1
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL)
        # Race: if the parent exited between spawn and prctl, we were already
        # reparented to init (pid 1) and the death signal will never fire.
        if os.getppid() == 1:
            os._exit(0)
    except Exception:  # noqa: BLE001 - best effort; never block the worker
        logger.debug("PR_SET_PDEATHSIG unavailable", exc_info=True)


def _worker_main(req_q, resp_q):
    """Entry point of the evaluator subprocess: a serve loop over the queues.

    EVERY message this loop puts on the response queue echoes the request's
    ``id``; the parent routes on it, so a response with no id (or a wrong one)
    is dropped and its caller waits out its deadline.
    """
    _install_parent_death_signal()
    state: dict = {}
    while True:
        msg = req_q.get()
        op = msg.get("op")
        rid = msg.get("id")
        if op == "shutdown":
            break
        try:
            if op == "solve":
                result = _do_solve(state, msg, resp_q)
                resp_q.put({"ok": True, "id": rid, **result})
            elif op == "eval":
                result = _do_eval(state, msg)
                resp_q.put({"ok": True, "id": rid, **result})
            else:  # pragma: no cover - defensive
                resp_q.put(
                    {"ok": False, "id": rid, "error": f"unknown op '{op}'"}
                )
        except Exception as exc:  # noqa: BLE001 - report, keep the loop alive
            logger.exception("tune worker error")
            resp_q.put(
                {
                    "ok": False,
                    "id": rid,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )


# ---------------------------------------------------------------------------
# Parent-side handle to the worker process
# ---------------------------------------------------------------------------


class EvaluatorWorker:
    """Owns the evaluator subprocess and its request/response queues.

    A spawn-context process is used deliberately: forking after pytensor /
    BLAS have initialised in the server process is unsafe, and spawn re-imports
    cleanly at the cost of a slower (seconds) startup that is dwarfed by the
    solve itself.

    **The protocol is request-id addressed.**  Every request carries a fresh
    uuid; the child echoes it on every message it sends back (progress
    included); the parent runs ONE reader thread that drains the shared
    response queue and files each message under the id it echoes, and each
    caller waits only on its own id.  The queues are shared but the mailboxes
    are not, which is what makes it safe for a slider eval and a Solve to be in
    flight at the same time -- the case the server really produces, since a
    Solve runs on the tune pool while ``/api/tune/eval`` runs on FastAPI's
    threadpool.  Before ids, whichever thread called ``get()`` first took
    whatever message arrived: an eval could return the Solve's payload, and the
    ``_await(None)`` in the eval path silently ATE the Solve's
    ``{"progress": "compiling", "data_plots": ...}`` message (review 1.5).
    Both are structurally impossible now -- an eval never sees a message
    addressed to the Solve, so there is nothing for it to steal or drop.

    A message whose id nobody is waiting on (a superseded request, or an answer
    that arrived after its caller gave up) is DROPPED at the reader, which is
    the only place that decision is made.

    Waiting is bounded twice over: a worker that DIES raises immediately, and a
    worker that hangs while still alive -- a pytensor compile deadlock, native
    code, an OOM-frozen child -- is terminated and respawned when its silence
    deadline expires (review 1.4).
    """

    def __init__(self):
        self._ctx = multiprocessing.get_context("spawn")
        self._req_q = self._ctx.Queue()
        self._resp_q = self._ctx.Queue()
        self._proc = None
        # Response demultiplexer.  _cv guards all four fields below; its lock
        # is an RLock so a waiter can stop the reader without releasing it.
        self._cv = threading.Condition(threading.RLock())
        self._inbox: dict = {}  # request id -> [messages], oldest first
        self._pending: dict = {}  # request id -> generation it was sent in
        self._generation = 0  # bumped by every restart / close
        self._reader = None
        self._reader_stop = None

    def start(self):
        if self._proc is not None and self._proc.is_alive():
            return
        self._proc = self._ctx.Process(
            target=_worker_main, args=(self._req_q, self._resp_q), daemon=True
        )
        self._proc.start()

    def is_alive(self):
        return self._proc is not None and self._proc.is_alive()

    def solve(self, config, params, workdir, on_progress=None):
        """Run a full solve; block for the result, forwarding progress states."""
        rid = self._begin_request()
        self._req_q.put(
            {
                "op": "solve",
                "id": rid,
                "config": config,
                "params": params,
                "workdir": workdir,
            }
        )
        return self._await(rid, on_progress, SOLVE_TIMEOUT_S)

    def set_and_eval(self, path, value):
        """Move one parameter and return the updated model traces."""
        rid = self._begin_request()
        self._req_q.put(
            {"op": "eval", "id": rid, "path": path, "value": value}
        )
        return self._await(rid, None, EVAL_TIMEOUT_S)

    # -- request bookkeeping ------------------------------------------------

    def _begin_request(self):
        """Mint a request id and register it BEFORE the request is sent.

        Registering first is what makes the reader's drop rule safe: an id it
        does not know is one nobody will ever wait for, never one whose caller
        has not got around to registering yet.
        """
        rid = uuid.uuid4().hex
        with self._cv:
            self._pending[rid] = self._generation
        return rid

    def _end_request(self, rid):
        with self._cv:
            self._pending.pop(rid, None)
            self._inbox.pop(rid, None)

    # -- response demultiplexer --------------------------------------------

    def _ensure_reader(self):
        """Start the reader thread for the CURRENT response queue, if needed."""
        with self._cv:
            if self._reader is not None and self._reader.is_alive():
                return
            stop = threading.Event()
            self._reader_stop = stop
            self._reader = threading.Thread(
                target=self._reader_loop,
                args=(self._resp_q, stop),
                daemon=True,
                name="exozippy-tune-reader",
            )
            self._reader.start()

    def _stop_reader(self):
        with self._cv:
            stop, self._reader_stop, self._reader = (
                self._reader_stop,
                None,
                None,
            )
        if stop is not None:
            stop.set()

    def _reader_loop(self, resp_q, stop):
        """Drain ``resp_q`` and file every message under the id it echoes.

        Bound to the queue it was started with, so a restart's fresh queue gets
        a fresh reader and this one retires with the process it was reading.
        """
        while not stop.is_set():
            try:
                msg = resp_q.get(timeout=_AWAIT_POLL_S)
            except queue.Empty:
                continue
            except (OSError, ValueError, EOFError):  # queue closed under us
                break
            rid = msg.get("id") if isinstance(msg, dict) else None
            with self._cv:
                if rid in self._pending:
                    self._inbox.setdefault(rid, []).append(msg)
                    self._cv.notify_all()
                else:
                    logger.debug(
                        "tune: dropping worker message for unclaimed "
                        "request id %r",
                        rid,
                    )

    def _take(self, rid, deadline):
        """Next message addressed to ``rid``, or raise.

        Raises ``_DeadlineExpired`` on silence, ``RuntimeError`` if the worker
        died or was restarted out from under this request.
        """
        self._ensure_reader()
        with self._cv:
            while True:
                msgs = self._inbox.get(rid)
                if msgs:
                    return msgs.pop(0)
                if (
                    self._pending.get(rid, self._generation)
                    != self._generation
                ):
                    raise RuntimeError(
                        "evaluator worker was restarted; this request is void"
                    )
                if not self.is_alive():
                    # Grace: the child may have answered as it exited, with the
                    # payload still in flight through the pipe and the reader
                    # yet to file it.
                    self._cv.wait(_AWAIT_POLL_S)
                    msgs = self._inbox.get(rid)
                    if msgs:
                        return msgs.pop(0)
                    self._stop_reader()
                    raise RuntimeError("evaluator worker exited")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise _DeadlineExpired
                self._cv.wait(min(_AWAIT_POLL_S, remaining))

    def _await(self, rid, on_progress, timeout):
        """Block for ``rid``'s answer, forwarding its progress messages.

        ``timeout`` is a SILENCE budget, not a total: every message addressed
        to this request restarts it, so a job that keeps reporting is never
        killed for being long, only for going quiet.
        """
        deadline = time.monotonic() + timeout
        try:
            while True:
                try:
                    msg = self._take(rid, deadline)
                except _DeadlineExpired:
                    self._hard_restart()
                    raise WorkerTimeout(
                        f"evaluator worker stopped responding "
                        f"(no message for {timeout:.0f}s); it was terminated "
                        f"and respawned -- press Solve again"
                    ) from None
                deadline = time.monotonic() + timeout
                if "progress" in msg:
                    if on_progress:
                        # The full message: phase string plus any payload
                        # riding along (data_plots). TuneSession._progress also
                        # accepts a bare phase string for simpler worker stubs.
                        on_progress(msg)
                    continue
                if not msg.get("ok"):
                    raise RuntimeError(msg.get("error", "worker error"))
                return msg
        finally:
            self._end_request(rid)

    # -- lifecycle ----------------------------------------------------------

    def _hard_restart(self):
        """Kill a wedged worker and spawn a clean one on fresh queues.

        The queues are replaced rather than reused: a process killed mid-write
        can leave a partial pickle in the pipe, and every later ``get()`` would
        inherit that. Bumping the generation releases any OTHER request waiting
        on the dead process immediately, instead of leaving it to burn its own
        deadline and terminate the fresh worker in turn.
        """
        proc, self._proc = self._proc, None
        self._stop_reader()
        if proc is not None:
            try:
                proc.terminate()
                proc.join(timeout=2.0)
                if proc.is_alive():  # pragma: no cover - SIGTERM ignored
                    proc.kill()
                    proc.join(timeout=2.0)
            except Exception:  # noqa: BLE001 - best effort; we respawn anyway
                logger.debug("tune: terminate failed", exc_info=True)
        for q in (self._req_q, self._resp_q):
            try:
                q.cancel_join_thread()
                q.close()
            except Exception:  # noqa: BLE001 - best effort
                logger.debug("tune: queue close failed", exc_info=True)
        self._req_q = self._ctx.Queue()
        self._resp_q = self._ctx.Queue()
        with self._cv:
            self._generation += 1
            self._inbox.clear()
            self._cv.notify_all()
        logger.warning("tune: evaluator worker was wedged; respawning")
        self.start()

    def close(self):
        with self._cv:
            self._generation += 1  # release anyone still waiting
            self._inbox.clear()
            self._cv.notify_all()
        self._stop_reader()
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
        self.phase = "idle"  # idle|solving|compiling|live|error
        self.error: Optional[str] = None
        self.structural_hash: Optional[str] = None
        self.result: Optional[dict] = None  # {parameters, seeds, plots}
        # Data-only PlotSpec JSON, available from the "compiling" phase on so
        # the GUI can draw the observations before the evaluator is live.
        self.data_plots: Optional[list] = None

    def _ensure_worker(self):
        """Return a live worker subprocess, (re)spawning only if needed.

        Reusing a warm worker across solves is the main speed lever: a spawn
        re-imports pytensor/pymc/exozippy (~10s) and cold-starts the pytensor
        compile cache, so paying that once instead of on every re-Solve makes
        repeated tuning dramatically faster. ``_do_solve`` rebuilds the System /
        model / evaluator from scratch each call, so a reused worker holds no
        stale state, and the child's serve loop survives a solve error, so an
        errored worker is still safe to reuse.
        """
        factory = self._worker_factory or EvaluatorWorker
        worker = self._worker
        if worker is None or not getattr(worker, "is_alive", lambda: True)():
            if worker is not None:
                try:
                    worker.close()
                except Exception:  # pragma: no cover
                    pass
            worker = factory()
            worker.start()
            self._worker = worker
        return worker

    def solve(self, config, params, workdir):
        """Blocking solve (call from a background thread). Sets phase/result."""
        self.phase = "solving"
        self.error = None
        self.data_plots = None
        try:
            # Brief lock only around (re)spawn, not the seconds-long solve, so
            # concurrent callers can't double-spawn but eval is never blocked.
            with self._lock:
                worker = self._ensure_worker()

            def _progress(update):
                # The real worker forwards the full progress message (phase +
                # optional data_plots); test stubs may send a bare string.
                if isinstance(update, dict):
                    self.phase = update.get("progress", self.phase)
                    if update.get("data_plots") is not None:
                        self.data_plots = update["data_plots"]
                else:
                    self.phase = update

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
        """Move one parameter (LIVE mode only) and return updated traces.

        The phase gate is read WITHOUT the lock and the call is made outside
        it, deliberately.  Taking the lock across either would serialize the
        slider against a seconds-long Solve (and, with a deadline in play, let
        one wedged eval hold the lock long enough to block the next Solve's
        respawn).  Neither needs the lock any more: the worker addresses every
        response by request id, so an eval that slips through the gate
        microseconds before a re-Solve starts can only ever receive its OWN
        answer -- or a clean "worker was restarted" error.  That is the half of
        review 1.5 the gate could not close.
        """
        worker = self._worker
        if worker is None or self.phase != "live":
            raise RuntimeError("no live evaluator; press Solve first")
        try:
            return worker.set_and_eval(path, value)
        except WorkerTimeout as exc:
            # The worker was terminated and respawned, so it no longer holds
            # the compiled evaluator: nothing is live until the next Solve, and
            # the UI has to be told that rather than left on stale sliders.
            self.error = str(exc)
            self.phase = "error"
            raise

    def status(self):
        return {
            "phase": self.phase,
            "error": self.error,
            "structural_hash": self.structural_hash,
            "has_result": self.result is not None,
            "has_data_plots": self.data_plots is not None,
        }

    def close(self):
        if self._worker is not None:
            try:
                self._worker.close()
            except Exception:  # pragma: no cover
                pass
            self._worker = None
