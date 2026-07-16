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
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml

from exozippy.gui import TERMINAL_PHASES

DEFAULT_PREFIX = "fitresults/planet"
_STATUS_SUFFIX = "_gui_status.json"
_SNAPSHOT_SUFFIX = "_gui_snapshot"


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
        return True   # exists but owned by another user
    except (TypeError, ValueError):
        return False
    return True


class RunHandle:
    """A launched fit: its subprocess, output prefix, and working directory."""

    def __init__(self, proc, prefix, cwd, config_path):
        self.proc = proc
        self.pid = proc.pid
        self.prefix = prefix
        self.cwd = cwd
        self.config_path = config_path

    @property
    def status_path(self):
        # os.path.join ignores cwd when prefix is absolute.
        return os.path.join(self.cwd, self.prefix + _STATUS_SUFFIX)

    @property
    def snapshot_dir(self):
        return os.path.join(self.cwd, self.prefix + _SNAPSHOT_SUFFIX)

    def is_alive(self):
        return self.proc.poll() is None

    def status(self):
        """Parsed status.json augmented with a liveness check.

        If the process is dead but the file never reached a terminal phase
        (crash before the run_fit wrapper could write one, or no file at all),
        the reported phase is forced to "error" with reason "died" so a
        monitor never waits forever on a stale non-terminal phase.
        """
        alive = self.is_alive()
        doc = None
        try:
            with open(self.status_path, "r") as fh:
                doc = json.load(fh)
        except (OSError, json.JSONDecodeError):
            doc = None

        if doc is None:
            # No readable status yet. Alive -> still starting; dead -> it died
            # before writing anything.
            phase = "starting" if alive else "error"
            result = {"phase": phase, "state": {}, "pid": self.pid}
            if not alive:
                result["error"] = "died"
                result["returncode"] = self.proc.poll()
            result["alive"] = alive
            return result

        phase = doc.get("phase")
        if not alive and phase not in TERMINAL_PHASES:
            doc["phase"] = "error"
            doc["error"] = "died"
            doc["returncode"] = self.proc.poll()
        doc["alive"] = alive
        return doc

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

    resolved_config = (config_path if os.path.isabs(config_path)
                       else os.path.join(cwd, config_path))
    prefix = _parse_prefix(resolved_config)

    env = dict(os.environ)
    env["EXOZIPPY_GUI_SNAPSHOT"] = "1"

    # A fresh interpreter via -m avoids any dependence on the console-script
    # entry point being on PATH and gives the child a clean PyTensor/pymc state.
    proc = subprocess.Popen(
        [sys.executable, "-m", "exozippy.cli", config_path],
        cwd=cwd,
        env=env,
    )
    return RunHandle(proc, prefix, cwd, config_path)


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
            runs.append({
                "prefix": path[: -len(_STATUS_SUFFIX)],
                "status_path": path,
                "phase": phase,
                "recorded_phase": doc.get("phase"),
                "pid": pid,
                "alive": alive,
                "state": doc.get("state", {}),
                "started_at": doc.get("started_at"),
                "updated_at": doc.get("updated_at"),
            })
    runs.sort(key=lambda r: (r.get("updated_at") or 0), reverse=True)
    return runs
