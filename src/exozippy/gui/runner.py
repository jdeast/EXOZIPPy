"""Launch, monitor, and stop an EXOZIPPy fit as a subprocess.

Pure standard library plus exozippy (no fastapi/uvicorn): a GUI backend, a
run queue, or a plain script can drive fits with this without pulling any
optional extra.

Each fit runs in a *fresh* interpreter (`python -m exozippy.cli <config>`).
That is deliberate: PyTensor / fork-pool / signal-handler state does not bleed
between runs, and a hung or crashing fit can never take down the controller.
The child is launched with EXOZIPPY_GUI_SNAPSHOT=1 so it emits the
<prefix>_gui_status.json + <prefix>_gui_snapshot/ artifacts that
``status()`` reads back.

Typical use:

    handle = start_run("kelt4.yaml", cwd="examples/kelt4")
    while handle.status()["phase"] not in TERMINAL_PHASES:
        time.sleep(1)
        ...  # read handle.status(), plot the snapshot npz
    handle.stop()                 # graceful: one SIGINT, wrap up + save trace
    handle.stop(force=True)       # escalate: 2nd SIGINT, then SIGKILL
"""

import json
import logging
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

import yaml

from exozippy.gui import TERMINAL_PHASES

logger = logging.getLogger(__name__)

DEFAULT_PREFIX = "fitresults/planet"
_STATUS_SUFFIX = "_gui_status.json"
_SNAPSHOT_SUFFIX = "_gui_snapshot"
_CONSOLE_SUFFIX = "_gui_console.log"

RUN_ID_ENV = "EXOZIPPY_GUI_RUN_ID"
"""Env var carrying this launch's run id into the child (see start_run)."""

MAX_CONSOLE_TAIL = 4000
"""Bytes of the child's console kept as the `error` of a crashed run."""


def _new_run_id():
    """A launch identifier unique across runs, machines and pid reuse."""
    return f"{os.getpid()}-{time.time_ns()}-{uuid.uuid4().hex[:8]}"


def _console_tail(path, limit=MAX_CONSOLE_TAIL):
    """Last `limit` bytes of the child's captured console, or None.

    Used only to explain a run that died: a crash BEFORE run_fit installs its
    reporter (a bad config, an import error, a bad CLI argument) -- and any
    crash the interpreter cannot catch at all (SIGKILL, OOM, segfault) --
    leaves nothing in the status file, and this is the only surviving trace.
    """
    if not path:
        return None
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as fh:
            if size > limit:
                fh.seek(size - limit)
            data = fh.read()
    except OSError:
        return None
    text = data.decode("utf-8", "replace").strip()
    return text or None


def _parse_prefix(config_path):
    """Read the ``prefix:`` key from a run config (default fitresults/planet)."""
    try:
        with open(config_path, "r") as fh:
            cfg = yaml.safe_load(fh) or {}
    except (OSError, yaml.YAMLError):
        return DEFAULT_PREFIX
    prefix = cfg.get("prefix", DEFAULT_PREFIX)
    return str(prefix) if prefix else DEFAULT_PREFIX


def _pid_is_running(pid):
    """Liveness probe for a bare pid (used by list_runs, which has no Popen)."""
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists but owned by another user
    except (TypeError, ValueError):
        return False
    return True


class RunHandle:
    """A launched fit: its subprocess, output prefix, and working directory."""

    def __init__(self, proc, prefix, cwd, config_path, run_id=None):
        self.proc = proc
        self.pid = proc.pid
        self.prefix = prefix
        self.cwd = cwd
        self.config_path = config_path
        self.run_id = run_id

    @property
    def status_path(self):
        # os.path.join ignores cwd when prefix is absolute.
        return os.path.join(self.cwd, self.prefix + _STATUS_SUFFIX)

    @property
    def snapshot_dir(self):
        return os.path.join(self.cwd, self.prefix + _SNAPSHOT_SUFFIX)

    @property
    def console_path(self):
        """Where start_run captured this child's stdout+stderr."""
        return os.path.join(self.cwd, self.prefix + _CONSOLE_SUFFIX)

    def is_alive(self):
        return self.proc.poll() is None

    def _read_status_doc(self):
        """(doc, stale) for this run's status file.

        `doc` is None unless the file exists, parses, AND belongs to THIS
        launch; `stale` says a document was there but described a different
        run. Runs at a given prefix overwrite each other's status file, so
        without the run-id check a fit that crashed before writing anything
        would report the PREVIOUS run's terminal phase -- the user reads
        "done" about a run that died. The check is skipped only when this
        handle has no run id (a hand-built handle in a test); a document
        written before run ids existed never matches one.
        """
        try:
            with open(self.status_path, "r") as fh:
                doc = json.load(fh)
        except (OSError, json.JSONDecodeError):
            return None, False
        if not isinstance(doc, dict):
            return None, False
        if self.run_id is not None and doc.get("run_id") != self.run_id:
            return None, True
        return doc, False

    def status(self):
        """Parsed status.json augmented with a liveness check.

        Four states are distinguishable, and a crashed run is never confused
        with a finished one:

        * alive, with a status document  -> the fit's own phase
        * alive, no document for this run -> "starting"
        * dead, terminal document        -> "done"/"stopped"/"error" as written
        * dead, anything else            -> "error", with the child's console
          tail (or "died") as `error` and the process return code

        The last case covers both a crash before ``run_fit`` could record its
        traceback and a stale document left by an EARLIER run at this prefix;
        `stale_status` flags the latter so the caller can say "unknown"
        rather than repeat someone else's answer.
        """
        alive = self.is_alive()
        doc, stale = self._read_status_doc()

        if doc is None:
            # No status for THIS run. Alive -> still starting; dead -> it died
            # before writing anything of its own.
            result = {
                "phase": "starting" if alive else "error",
                "state": {},
                "pid": self.pid,
                "run_id": self.run_id,
                "alive": alive,
            }
            if stale:
                result["stale_status"] = True
            if not alive:
                result["error"] = self._death_reason(stale)
                result["returncode"] = self.proc.poll()
            return result

        phase = doc.get("phase")
        if not alive and phase not in TERMINAL_PHASES:
            doc["phase"] = "error"
            doc["error"] = self._death_reason(False)
            doc["returncode"] = self.proc.poll()
        doc["alive"] = alive
        return doc

    def _death_reason(self, stale):
        """Why this run is gone: the exit status, plus the console tail.

        Both halves are labeled: the console tail of a fit killed mid-sampling
        is ordinary output, not a traceback, and presenting it bare would read
        as an error message it is not.
        """
        head = (
            f"the fit process exited (code {self.proc.poll()}) without "
            "recording a terminal status"
        )
        if stale:
            head += "; the status file at this prefix is an earlier run's"
        tail = _console_tail(self.console_path)
        return f"{head}\nlast console output:\n{tail}" if tail else head

    def stop(self, force=False, graceful_timeout=30.0, kill_timeout=10.0):
        """Request a graceful stop (SIGINT); optionally escalate.

        force=False: send a single SIGINT and return immediately. The PTDE
        sampler finishes the current step, saves the partial trace, and the
        run wraps up to a terminal "stopped"/"done" phase. The caller polls
        status()/wait() for completion.

        force=True: send SIGINT, wait up to graceful_timeout; if still alive
        send a second SIGINT (the sampler's two-signal contract -> immediate
        KeyboardInterrupt), wait up to kill_timeout; if STILL alive, SIGKILL.
        Returns the process return code (None only if it somehow survives).
        """
        if self.proc.poll() is not None:
            return self.proc.returncode

        self._signal(signal.SIGINT)
        if not force:
            return None

        try:
            self.proc.wait(timeout=graceful_timeout)
            return self.proc.returncode
        except subprocess.TimeoutExpired:
            pass

        # Second SIGINT: the sampler's handler raises KeyboardInterrupt at once.
        self._signal(signal.SIGINT)
        try:
            self.proc.wait(timeout=kill_timeout)
            return self.proc.returncode
        except subprocess.TimeoutExpired:
            pass

        self.proc.kill()
        try:
            self.proc.wait(timeout=kill_timeout)
        except subprocess.TimeoutExpired:
            pass
        return self.proc.poll()

    def wait(self, timeout=None):
        """Block until the fit exits; returns the process return code."""
        return self.proc.wait(timeout=timeout)

    def _signal(self, sig):
        # Windows cannot deliver SIGINT to another process: send_signal raises
        # `ValueError: Unsupported signal: 2`. The only cross-process interrupts
        # it accepts are CTRL_C_EVENT and CTRL_BREAK_EVENT, and CTRL_C_EVENT
        # cannot be aimed -- it goes to every process sharing the console,
        # including the GUI itself. CTRL_BREAK_EVENT can be aimed, which is why
        # start_run gives the child its own process group.
        #
        # Python delivers CTRL_BREAK_EVENT to the child as SIGBREAK rather than
        # SIGINT, so the samplers register _stop_handler for SIGBREAK too. Both
        # halves are required; either alone leaves stop silently doing nothing.
        if sys.platform == "win32" and sig == signal.SIGINT:
            sig = signal.CTRL_BREAK_EVENT
        try:
            self.proc.send_signal(sig)
        except ProcessLookupError:
            pass


def start_run(config_path, cwd=None):
    """Launch `exozippy <config_path>` as a subprocess and return a RunHandle.

    config_path : path to the run config yaml (relative to `cwd` if not
        absolute).
    cwd : working directory for the child (defaults to the current dir). The
        child resolves data files and its output prefix relative to this.
    """
    cwd = str(cwd) if cwd is not None else os.getcwd()
    config_path = str(config_path)

    resolved_config = (
        config_path
        if os.path.isabs(config_path)
        else os.path.join(cwd, config_path)
    )
    prefix = _parse_prefix(resolved_config)

    # Every run at a given prefix writes the same status file, so the child
    # stamps this id into it and RunHandle.status() refuses a document
    # carrying any other -- otherwise a fit that dies before writing anything
    # reports the PREVIOUS run's "done".
    run_id = _new_run_id()

    env = dict(os.environ)
    env["EXOZIPPY_GUI_SNAPSHOT"] = "1"
    env[RUN_ID_ENV] = run_id

    # A fresh interpreter via -m avoids any dependence on the console-script
    # entry point being on PATH and gives the child a clean PyTensor/pymc state.
    # On Windows the child needs its own process group, or CTRL_BREAK_EVENT
    # cannot be aimed at it (see RunHandle._signal). Harmless elsewhere, but
    # the flag only exists on Windows, so it is added conditionally.
    popen_kwargs = {}
    if sys.platform == "win32":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

    # Capture the child's console to a file. A crash before run_fit installs
    # its GuiReporter (an unreadable config, an import error, a bad argument)
    # writes its traceback to stderr and NOTHING to the status file, and with
    # the streams inherited that traceback lands wherever the GUI server was
    # started -- invisible to the user, who only sees the run vanish. A file
    # (not a PIPE: nobody is draining it, and a full pipe would wedge the fit)
    # keeps it where status() can report it. Best-effort, exactly like the
    # snapshot artifacts: if it cannot be opened the run still starts, with
    # the streams inherited as before.
    console = None
    try:
        console_path = os.path.join(cwd, prefix + _CONSOLE_SUFFIX)
        os.makedirs(os.path.dirname(console_path) or ".", exist_ok=True)
        console = open(console_path, "wb")
    except OSError:
        logger.exception(
            "Could not open the run console log; the fit's output stays on "
            "the parent's streams (non-fatal)."
        )

    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "exozippy.cli", config_path],
            cwd=cwd,
            env=env,
            stdout=console if console is not None else None,
            stderr=subprocess.STDOUT if console is not None else None,
            **popen_kwargs,
        )
    finally:
        # The child holds its own duplicate of the descriptor; keeping the
        # parent's copy open would leak one per run.
        if console is not None:
            console.close()

    return RunHandle(proc, prefix, cwd, config_path, run_id=run_id)


def list_runs(directory):
    """Summarize every fit under `directory` that emitted a GUI status file.

    Walks the tree for ``*_gui_status.json`` and returns one dict per run:
    prefix, status_path, phase, pid, alive (from the recorded pid), timestamps,
    and the summary state. Newest (by updated_at) first. A results browser and
    a run queue both reuse this.
    """
    directory = str(directory)
    runs = []
    for root, _dirs, files in os.walk(directory):
        for name in files:
            if not name.endswith(_STATUS_SUFFIX):
                continue
            path = os.path.join(root, name)
            try:
                with open(path, "r") as fh:
                    doc = json.load(fh)
            except (OSError, json.JSONDecodeError):
                continue
            pid = doc.get("pid")
            alive = _pid_is_running(pid)
            phase = doc.get("phase")
            if not alive and phase not in TERMINAL_PHASES:
                phase = "error"
            runs.append(
                {
                    "prefix": path[: -len(_STATUS_SUFFIX)],
                    "status_path": path,
                    "phase": phase,
                    "recorded_phase": doc.get("phase"),
                    "pid": pid,
                    "alive": alive,
                    "run_id": doc.get("run_id"),
                    "state": doc.get("state", {}),
                    "started_at": doc.get("started_at"),
                    "updated_at": doc.get("updated_at"),
                }
            )
    runs.sort(key=lambda r: r.get("updated_at") or 0, reverse=True)
    return runs
