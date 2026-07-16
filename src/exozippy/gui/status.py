"""Atomic GUI status + snapshot writer for a running fit.

A GUI (or any external monitor) needs to watch a long fit without touching the
sampler's memory. This module writes two things into the run's output
directory, both atomically (write a temp file, then os.replace, so a reader
never sees a half-written file):

  <prefix>_gui_status.json
      A single small JSON document: the current phase
      ("preparing" | "compiling" | "tuning" | "sampling" | "writing" | "done"
      | "stopped" | "error"), the latest progress state dict, the pid, and
      started/updated timestamps.

  <prefix>_gui_snapshot/partial.npz (+ partial.json)
      A downsampled copy of the T=1 raw draws collected so far (thinned to
      <= MAX_SNAPSHOT_DRAWS per chain) plus the running log-posterior, so a
      GUI can render a live trace/corner without the sampler ever handing it
      an in-memory idata. Cheap: a slice + one savez per convergence check.

run.py owns a GuiReporter for the whole fit and calls .phase(...) at each
major transition; the PTDE samplers call the same reporter's
.progress_callback(state) at each geometric convergence check. Everything is a
no-op when the reporter is disabled (the default), so a normal non-GUI run
writes zero extra files.
"""

import json
import logging
import os
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# A GUI status file at rest on any of these phases means the run is over.
TERMINAL_PHASES = frozenset({"done", "stopped", "error"})

# Summary keys guaranteed in the state dict handed to progress_callback. Extra
# keys (stored_raw / stored_lp / raw_var_names) may also be present to feed the
# snapshot writer; they are stripped out of the JSON status document.
_SUMMARY_KEYS = ("n_draws", "n_chains", "max_rhat", "min_ess",
                 "elapsed_s", "stop_reason")

MAX_SNAPSHOT_DRAWS = 200
"""Per-chain draw cap for the downsampled snapshot npz."""

SNAPSHOT_BUDGET_S = 5.0
"""If a single snapshot write exceeds this, warn once and skip future ones."""


def gui_enabled(config):
    """True when this run should emit GUI status/snapshot files.

    Enabled by a truthy config["gui"]["snapshot"] OR the environment variable
    EXOZIPPY_GUI_SNAPSHOT=1 (how the subprocess runner opts a child fit in).
    """
    cfg = config.get("gui") if isinstance(config, dict) else None
    if isinstance(cfg, dict) and cfg.get("snapshot"):
        return True
    return os.environ.get("EXOZIPPY_GUI_SNAPSHOT", "") == "1"


def _json_default(obj):
    """Coerce numpy scalars/arrays so json.dumps never chokes on a state dict."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _atomic_write_bytes(path, data):
    """Write bytes to `path` atomically (temp file in the same dir + os.replace)."""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    tmp = os.path.join(directory, f".{os.path.basename(path)}.tmp.{os.getpid()}")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _atomic_write_text(path, text):
    _atomic_write_bytes(path, text.encode("utf-8"))


class GuiReporter:
    """Owns the status.json + snapshot files for one fit.

    Every public method is a no-op when ``enabled`` is False, and every write
    is wrapped so a filesystem hiccup logs a warning but never derails the
    fit -- a monitoring artifact must not be able to crash the science run.
    """

    def __init__(self, prefix, enabled=True):
        self.prefix = str(prefix)
        self.enabled = bool(enabled)
        self.status_path = self.prefix + "_gui_status.json"
        self.snapshot_dir = self.prefix + "_gui_snapshot"
        self._t_start = time.time()
        self._last_state = {}
        self._snapshot_over_budget = False

    @classmethod
    def from_config(cls, config):
        """Build a reporter from a run config, honoring the gui flag / env var."""
        prefix = Path(config.get("prefix", "fitresults/planet"))
        return cls(prefix, enabled=gui_enabled(config))

    # -- status.json ------------------------------------------------------

    def _write_status(self, phase, state):
        doc = {
            "phase": phase,
            "pid": os.getpid(),
            "state": {k: state.get(k) for k in _SUMMARY_KEYS} if state else {},
            "started_at": self._t_start,
            "updated_at": time.time(),
        }
        _atomic_write_text(
            self.status_path,
            json.dumps(doc, default=_json_default, indent=2))

    def phase(self, phase, state=None):
        """Write status.json at a phase transition.

        `state` defaults to the last progress state seen, so a phase change
        (e.g. sampling -> writing) keeps the most recent draw/convergence
        numbers visible instead of blanking them.
        """
        if not self.enabled:
            return
        if state is None:
            state = self._last_state
        else:
            self._last_state = {k: state.get(k) for k in _SUMMARY_KEYS}
        try:
            self._write_status(phase, state)
        except Exception:
            logger.warning("GuiReporter: failed to write status file %s",
                           self.status_path, exc_info=True)

    def terminal(self, phase):
        """Force a terminal phase (done/stopped/error), reusing the last state.

        Called on every exit path from run_fit so a reader never finds the
        file stuck on a non-terminal phase after the process is gone.
        """
        if phase not in TERMINAL_PHASES:
            raise ValueError(f"terminal phase must be one of {sorted(TERMINAL_PHASES)}")
        self.phase(phase, self._last_state)

    # -- progress callback (passed to the samplers) -----------------------

    def progress_callback(self, state):
        """Sampler hook: refresh status.json + write a partial-draw snapshot.

        Invoked at each PTDE convergence check (checks grow geometrically, so
        this stays bounded). `state` carries the documented summary keys plus,
        best-effort, stored_raw / stored_lp / raw_var_names for the snapshot.
        """
        if not self.enabled:
            return
        self.phase("sampling", state)
        try:
            self._write_snapshot(state)
        except Exception:
            logger.warning("GuiReporter: failed to write snapshot",
                           exc_info=True)

    def _write_snapshot(self, state):
        raw = state.get("stored_raw")
        names = state.get("raw_var_names")
        lp = state.get("stored_lp")
        n_draws = state.get("n_draws")
        if raw is None or not names or not n_draws:
            return
        if self._snapshot_over_budget:
            return

        t0 = time.time()
        n_draws = int(n_draws)
        thin = max(1, -(-n_draws // MAX_SNAPSHOT_DRAWS))  # ceil division
        sl = slice(0, n_draws, thin)

        arrays = {}
        for name in names:
            arr = raw.get(name)
            if arr is None:
                continue
            arrays[name] = np.asarray(arr[:, sl])
        if lp is not None:
            arrays["_lp"] = np.asarray(lp[:, sl])
        if not arrays:
            return

        n_kept = arrays[next(iter(arrays))].shape[1]
        os.makedirs(self.snapshot_dir, exist_ok=True)
        npz_path = os.path.join(self.snapshot_dir, "partial.npz")
        tmp_npz = npz_path + f".tmp.{os.getpid()}"
        with open(tmp_npz, "wb") as fh:
            np.savez_compressed(fh, **arrays)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_npz, npz_path)

        meta = {
            "var_names": [n for n in names if n in arrays],
            "n_draws": n_draws,
            "n_kept": int(n_kept),
            "thin": int(thin),
            "n_chains": int(state.get("n_chains") or 0),
            "max_rhat": _to_float(state.get("max_rhat")),
            "min_ess": _to_float(state.get("min_ess")),
            "updated_at": time.time(),
        }
        _atomic_write_text(
            os.path.join(self.snapshot_dir, "partial.json"),
            json.dumps(meta, default=_json_default, indent=2))

        cost = time.time() - t0
        if cost > SNAPSHOT_BUDGET_S:
            self._snapshot_over_budget = True
            logger.warning(
                "GuiReporter: snapshot write took %.1fs (> %.0fs budget); "
                "skipping further snapshots for the rest of this run.",
                cost, SNAPSHOT_BUDGET_S)


def _to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None
