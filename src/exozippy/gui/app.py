"""FastAPI application shell for the EXOZIPPy GUI.

The GUI is a pure, OPTIONAL wrapper around the backend. This module owns the
HTTP/WebSocket surface and the ``exozippy-gui`` entry point; it never encodes
component-specific knowledge -- everything it serves comes from the
component-declared contracts in :mod:`exozippy.introspect` and
:mod:`exozippy.utilities.registry`.

FastAPI, uvicorn, and pywebview are optional (the ``gui`` extra). Importing
this module without them installed raises a clear message only when you
actually try to build the app or run ``main()``; ``import exozippy`` and the
plain CLI never touch this file.

Endpoints (Phase 2, G7):
    GET  /api/health          -- liveness probe
    GET  /api/schema          -- full component + global schema (introspect.py)
    GET  /api/utilities       -- component-declared utility argument schemas
    POST /api/project/open    -- validate a dir, list its yaml + data files
    WS   /api/logs?file=...   -- tail a log file, following rotation/truncation
    GET  /                    -- the prebuilt React bundle (static/)

Data file manager (G9):
    GET  /api/files?dir=...       -- browse the project tree for data files
    GET  /api/browse?dir=...      -- unconfined browse for the project picker
    POST /api/files/eligible      -- schema-driven association menu for a file
    GET  /api/files/associations  -- current file -> instance associations

Run controls (G11):
    POST /api/run             -- launch a fit as a subprocess (one per project)
    GET  /api/run/status      -- poll the active run's phase + progress state
    POST /api/run/stop        -- graceful SIGINT stop (force=true escalates)
    GET  /api/run/plots       -- list start/progress plot images on disk
    GET  /api/run/image?path= -- serve a plot image from the run's output dir
    POST /api/utilities/run   -- run a component utility headless (G2 registry)

Tune tab (G10):
    POST /api/tune/solve      -- solve + compile the evaluator in a worker proc
    GET  /api/tune/status     -- poll the solve phase (solving/compiling/live)
    GET  /api/theme           -- a copy of the core chart style table
    GET  /api/tune/result     -- solved parameters + base Charts
    GET  /api/tune/plots/data -- data-only Charts (available from the
                                 "compiling" phase, before the solve is live)
    POST /api/tune/eval       -- move one parameter, get updated model curves
    GET  /api/tune/hash       -- structural hash of the open doc (staleness)
"""

import argparse
import asyncio
import glob
import json
import os
import shutil
import socket
import sys
import threading
import uuid
import webbrowser
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit

import yaml

# The built frontend bundle lives here (committed to the wheel). It may be
# absent in a source checkout that has not run `npm run build` yet; we degrade
# to a small placeholder page rather than failing to import.
STATIC_DIR = Path(__file__).parent / "static"

# YAML files that look like parameter-override files rather than system configs.
_PARAMS_SUFFIXES = (".params.yaml", ".params.yml")

# Data-file extensions worth surfacing in the project listing. Kept generic --
# the datafile schema (G1) is the real authority; this is only a listing
# convenience.
_DATA_EXTS = (
    ".sed",
    ".rv",
    ".tran",
    ".dat",
    ".txt",
    ".csv",
    ".eph",
    ".priors",
    ".fits",
    ".json",
)


def _require_fastapi():
    """Import FastAPI lazily, with an actionable error if the extra is missing."""
    try:
        import fastapi  # noqa: F401
        import uvicorn  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised via message only
        raise RuntimeError(
            "The EXOZIPPy GUI requires the optional 'gui' dependencies. "
            "Install them with:  pip install exozippy[gui]  "
            "(developers: poetry install -E gui)."
        ) from exc


# --- project directory listing ------------------------------------------------


_CONFIG_TOP_KEYS = None


def _config_top_keys():
    """Top-level YAML keys that mark a file as a system config.

    A system config's blocks are component yaml_keys (star, orbit, sed, ...)
    plus the global bookkeeping keys run.py understands (run, prefix, sampler,
    ...). We ask the introspection layer for the real, component-agnostic set so
    no component name is hardcoded; discovery imports the component stack, so it
    is cached and done at most once per process. The literal fallbacks cover the
    case where introspection is unavailable in a lightweight context.

    The cache is written only on SUCCESS, deliberately: an ``@lru_cache`` here
    froze the four-key literal fallback for the whole process lifetime after
    one transient import failure, and every config in the project then
    classified as "other" with no way to recover short of a restart.
    """
    global _CONFIG_TOP_KEYS
    if _CONFIG_TOP_KEYS is not None:
        return _CONFIG_TOP_KEYS
    keys = {"run", "prefix", "logger_level", "sampler", "parameter_file"}
    try:
        from ..introspect import _global_schema, list_components

        keys.update(list_components().keys())
        keys.update(_global_schema().keys())
    except Exception:  # pragma: no cover - defensive; fall back to literals
        return keys  # degraded answer: usable now, never remembered
    _CONFIG_TOP_KEYS = keys
    return keys


# Above this size a .yaml file is not a config or a params file by any
# plausible reading, and parsing it would stall the project listing (which
# classifies every YAML in the directory). Fall back to the naming convention.
_YAML_PARSE_MAX_BYTES = 4 * 1024 * 1024

# (path, mtime_ns, size) -> classification. Project open re-reads every YAML in
# the directory on every switch; the identity triple invalidates itself the
# moment a file is edited, so a memo is safe and the repeat opens are free.
_CLASSIFY_CACHE = {}
_CLASSIFY_CACHE_MAX = 512

# Ceiling on retained background-validation jobs (see doc_validate).
_MAX_VALIDATE_JOBS = 32


def _classify_yaml_uncached(path, size):
    """Classify a YAML file by its content: 'params', 'config', or 'other'.

    Filename conventions alone are unreliable (kelt4.params.3.yaml is a params
    file that does not end in .params.yaml), so content is the primary signal
    and the .params.yaml suffix is honored as a secondary one:

      * every top-level key is a dotted parameter path (comp.instance.param)
        -> 'params'  (content-decisive; catches oddly-named params files)
      * filename ends with .params.yaml/.yml  -> 'params'  (the user's naming
        convention, honored for params whose values we don't introspect)
      * at least one top-level key is a known component/global block  -> 'config'
      * anything else (a component's own input file like kelt4.sed.yaml, or
        arbitrary YAML)  -> 'other'

    A file with NO content to read falls back to the suffix convention:
    unreadable/invalid YAML, one past the parse size cap, and -- the case that
    used to be wrong -- an empty or comment-only file, which parses cleanly to
    ``None`` and so had no keys to match. That landed a freshly created
    template in "other", where the config picker cannot see it at all, which is
    exactly when a user needs to select it.
    """
    name = path.name.lower()
    by_suffix = "params" if name.endswith(_PARAMS_SUFFIXES) else "config"
    if size is not None and size > _YAML_PARSE_MAX_BYTES:
        return by_suffix
    try:
        with open(path, "r") as fh:
            data = yaml.safe_load(fh)
    except Exception:
        return by_suffix
    if data is None:
        return by_suffix

    keys = (
        [k for k in data if isinstance(k, str)]
        if isinstance(data, dict)
        else []
    )
    if keys and all("." in k for k in keys):
        return "params"
    if name.endswith(_PARAMS_SUFFIXES):
        return "params"
    if _config_top_keys().intersection(keys):
        return "config"
    return "other"


def _classify_yaml(path):
    """``_classify_yaml_uncached`` memoized on (path, mtime, size)."""
    try:
        st = path.stat()
        key = (str(path), st.st_mtime_ns, st.st_size)
        size = st.st_size
    except OSError:
        key, size = None, None
    if key is not None:
        cached = _CLASSIFY_CACHE.get(key)
        if cached is not None:
            return cached
    kind = _classify_yaml_uncached(path, size)
    if key is not None:
        if len(_CLASSIFY_CACHE) >= _CLASSIFY_CACHE_MAX:
            _CLASSIFY_CACHE.clear()
        _CLASSIFY_CACHE[key] = kind
    return kind


def open_project(path):
    """Validate a directory and describe the files a project GUI cares about.

    Returns a JSON-serializable dict:
        {dir, configs: [...], params: [...], data_files: [...], other: [...]}
    Each entry is {name, path, size, kind}. Raises ValueError if the path is
    not an existing directory. Component-agnostic: it does not parse the yaml,
    it only classifies by extension/name so the frontend can offer choices.
    """
    root = Path(path).expanduser()
    if not root.exists():
        raise ValueError(f"No such path: {root}")
    if not root.is_dir():
        raise ValueError(f"Not a directory: {root}")

    configs, params, data_files, other = [], [], [], []
    for child in sorted(root.iterdir()):
        if child.name.startswith("."):
            continue
        if not child.is_file():
            continue
        try:
            size = child.stat().st_size
        except OSError:
            size = None
        suffix = child.suffix.lower()
        entry = {
            "name": child.name,
            "path": str(child.resolve()),
            "size": size,
        }
        if suffix in (".yaml", ".yml"):
            kind = _classify_yaml(child)
            entry["kind"] = kind
            if kind == "params":
                params.append(entry)
            elif kind == "config":
                configs.append(entry)
            else:
                other.append(entry)
        elif suffix in _DATA_EXTS:
            entry["kind"] = "data"
            data_files.append(entry)
        else:
            entry["kind"] = "other"
            other.append(entry)

    return {
        "dir": str(root.resolve()),
        "configs": configs,
        "params": params,
        "data_files": data_files,
        "other": other,
    }


# --- log tailing --------------------------------------------------------------


def _read_last_lines(path, n):
    """Return the last ``n`` lines of a text file (best-effort, tolerant)."""
    try:
        with open(path, "r", errors="replace") as fh:
            return fh.readlines()[-n:]
    except OSError:
        return []


def _origin_is_local(origin):
    """Whether a WebSocket handshake ``Origin`` may open a log tail.

    WebSocket handshakes are NOT subject to CORS, so without this any web page
    the user happens to visit could open ``ws://127.0.0.1:<port>/api/logs`` and
    read back whatever it streams. A missing Origin is allowed: browsers always
    send one on a WS handshake, so an absent header means a non-browser client
    (curl, the test client, a script), which was never the threat.
    """
    if not origin:
        return True
    try:
        host = urlsplit(origin).hostname
    except ValueError:
        return False
    return host in ("127.0.0.1", "localhost", "::1")


async def _tail_log(websocket, file_path, from_lines=200, poll_s=0.5):
    """Stream a growing log file over a WebSocket, following rotation.

    Sends the last ``from_lines`` lines on connect, then polls for appended
    content. If the file shrinks (truncation) or its inode changes (rotation),
    reopens and streams the new file from its start. Runs until the client
    disconnects.

    **The client read is the sleep.** The poll interval is spent waiting on
    ``websocket.receive()`` rather than on ``asyncio.sleep``, so a disconnect
    ends the loop on the turn it arrives. Two failure modes hang on that
    detail:

    * Without reading at all, a disconnect is only discovered on the next
      SEND, so a tail on a QUIET file -- a finished run's log, a file that
      never appears -- looped stat+sleep forever holding an open file handle.
      ``LogTerminal`` opens a fresh socket per file switch, so those
      accumulate one immortal poller per switch for the life of the server.
    * Watching for the disconnect in a SECOND task works too, but then the
      handler needs several event-loop turns to wind down (cancel the sibling,
      gather it, return). That is enough to lose a race against a test
      harness's portal teardown -- observed only on the slower macOS runner.
      One task, and the loop returns directly.

    The receive future is created once and re-awaited across iterations rather
    than being cancelled and remade each poll: cancelling a half-completed
    receive is where a frame -- including the disconnect itself -- could go
    missing.
    """
    path = Path(file_path)

    # Seed with the tail so the user sees recent history immediately. Whether
    # the file existed AT THAT MOMENT decides where the first open starts.
    seeded = path.exists()
    for line in _read_last_lines(path, from_lines):
        await websocket.send_text(line.rstrip("\n"))

    fh = None
    inode = None
    skip_seeded_tail = seeded
    recv = asyncio.ensure_future(websocket.receive())
    try:
        while True:
            try:
                st = path.stat()
                # (Re)open on first sight, rotation (new inode), or truncation.
                reopen = (
                    fh is None or inode != st.st_ino or fh.tell() > st.st_size
                )
                if reopen:
                    if fh is not None:
                        fh.close()
                    fh = open(path, "r", errors="replace")
                    inode = st.st_ino
                    if skip_seeded_tail:
                        # The only open whose content we have already sent.
                        fh.seek(0, os.SEEK_END)
                        skip_seeded_tail = False
                    # Every LATER (re)open is a rotation or a truncation: the
                    # file under us is a different or restarted one, so its
                    # content is new and streams from the start -- which is
                    # what "follows rotation" has always claimed and what
                    # an unconditional seek-to-end quietly did not do. Same
                    # for a file that did not exist at connect time: its first
                    # lines are content, not history we already sent.
                line = fh.readline()
                if line:
                    await websocket.send_text(line.rstrip("\n"))
                    continue
            except FileNotFoundError:
                # File not created yet (or mid-rotation); wait and retry.
                fh = None
                inode = None
            # No new data: spend the poll interval waiting on the client.
            done, _pending = await asyncio.wait({recv}, timeout=poll_s)
            if not done:
                continue
            try:
                message = recv.result()
            except Exception:  # noqa: BLE001 - a dead socket ends the tail
                return
            if message.get("type") == "websocket.disconnect":
                return
            # Client chatter (this endpoint has no commands): keep listening.
            recv = asyncio.ensure_future(websocket.receive())
    finally:
        recv.cancel()
        if fh is not None:
            fh.close()


# --- run controls (G11) -------------------------------------------------------

# Image extensions worth surfacing as inline thumbnails in the plot galleries.
# Anything else a component writes (e.g. a multi-page .pdf) is still on disk but
# is not offered to the browser as an <img>.
_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".svg", ".gif")


def _prefix_path(handle):
    """Absolute output prefix of a run (handle.prefix resolved against its cwd)."""
    return os.path.join(handle.cwd, handle.prefix)


def _results_dir(handle):
    """Directory the run writes its outputs into (the dir of its prefix)."""
    return os.path.dirname(_prefix_path(handle)) or handle.cwd


def _log_path(handle):
    """The <prefix>.log file the fit's logger writes (see exozippy/logger.py)."""
    return _prefix_path(handle) + ".log"


def _console_path(handle):
    """The captured stdout+stderr of the fit subprocess, when there is one.

    This is where a crash that never reached the fit's own logger (an
    unreadable config, an import error) leaves its traceback. Optional: a
    hand-built handle (tests, a run adopted from elsewhere) has none.
    """
    return getattr(handle, "console_path", None)


def _read_snapshot_meta(handle):
    """Latest partial.json snapshot metadata for the run, or None if absent."""
    path = os.path.join(handle.snapshot_dir, "partial.json")
    try:
        with open(path, "r") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def run_status_payload(handle):
    """Assemble the JSON status document the Run tab polls.

    Augments RunHandle.status() with the paths a browser needs -- the log file
    to auto-attach the terminal to, the results directory to link, and the
    latest downsampled-snapshot metadata (n_draws/max_rhat/min_ess/updated_at)
    for the progress strip and rhat sparkline.

    `terminal` says the run is over (done/stopped/error, from RunHandle's own
    liveness-checked phase) so the frontend can stop offering Stop and show
    `error` -- a crashed run and a finished one must not look alike.
    """
    from ..gui import TERMINAL_PHASES

    status = handle.status()
    phase = status.get("phase")
    return {
        "active": True,
        "phase": phase,
        "terminal": phase in TERMINAL_PHASES,
        "state": status.get("state", {}),
        "alive": status.get("alive"),
        "pid": status.get("pid"),
        "run_id": status.get("run_id"),
        "stale_status": bool(status.get("stale_status")),
        "returncode": status.get("returncode"),
        "error": status.get("error"),
        "prefix": handle.prefix,
        "config_path": handle.config_path,
        "cwd": handle.cwd,
        "log_path": _log_path(handle),
        "console_path": _console_path(handle),
        "results_dir": _results_dir(handle),
        "snapshot": _read_snapshot_meta(handle),
    }


def _list_prefix_images(handle, pattern):
    """Sorted absolute image paths matching <prefix><pattern> (e.g. '_start*')."""
    out = []
    for path in sorted(glob.glob(_prefix_path(handle) + pattern)):
        if os.path.splitext(path)[1].lower() in _IMAGE_EXTS and os.path.isfile(
            path
        ):
            out.append(path)
    return out


def _params_file_of(config_path, cwd):
    """Absolute path of the params file a run of ``config_path`` will read.

    Read from the config on disk rather than taken from the caller: the fit
    subprocess resolves ``config['parameter_file']`` relative to its own cwd
    (System.__init__), so this is by construction the file the fit uses, and it
    cannot drift from it the way a separately-passed path can. Returns None
    when the config names no parameter_file or cannot be read -- the snapshot
    is best-effort and never blocks a run.
    """
    path = (
        config_path
        if os.path.isabs(config_path)
        else os.path.join(cwd, config_path)
    )
    try:
        with open(path, "r") as fh:
            config = yaml.safe_load(fh) or {}
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(config, dict):
        return None
    params = config.get("parameter_file")
    if not params:
        return None
    params = str(params)
    return params if os.path.isabs(params) else os.path.join(cwd, params)


def _snapshot_run_inputs(handle, params_path=None):
    """Copy the exact config/params used into the output dir for reproducibility.

    Writes '<stem>.used<ext>' beside the run's outputs so a finished fit always
    carries a frozen copy of what produced it, even if the source yaml is later
    edited. Copying onto the source path is skipped. Best-effort: an I/O error
    never blocks the run. Returns the list of copies made.

    ``params_path`` is an explicit override; with none given the config's own
    ``parameter_file`` is followed. That default is the point: no caller ever
    passed the argument, so the params half of the snapshot never happened, and
    the params file is precisely where the start values, priors and fixed flags
    that a rerun must reproduce are written.
    """
    results_dir = _results_dir(handle)
    try:
        os.makedirs(results_dir, exist_ok=True)
    except OSError:
        return []

    src_config = handle.config_path
    if not os.path.isabs(src_config):
        src_config = os.path.join(handle.cwd, src_config)

    if params_path is None:
        params_path = _params_file_of(src_config, handle.cwd)

    copied = []
    for src in (src_config, params_path):
        if not src or not os.path.isfile(src):
            continue
        stem, ext = os.path.splitext(os.path.basename(src))
        dst = os.path.join(results_dir, f"{stem}.used{ext}")
        if os.path.abspath(src) == os.path.abspath(dst):
            continue
        try:
            shutil.copy2(src, dst)
            copied.append(dst)
        except OSError:
            pass
    return copied


# --- app factory --------------------------------------------------------------


def create_app(project_dir=None, initial_config=None):
    """Build and return the FastAPI application.

    Requires the 'gui' extra. ``project_dir`` seeds the initial project the
    frontend opens; ``initial_config`` optionally names a specific config file
    within it to pre-select in the Config tab. Both are served to the client
    via GET /api/config.
    """
    _require_fastapi()

    from concurrent.futures import ThreadPoolExecutor

    from fastapi import FastAPI, WebSocket
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel

    from .document import ProjectDocument, command_from_json

    app = FastAPI(
        title="EXOZIPPy", docs_url="/api/docs", openapi_url="/api/openapi.json"
    )

    initial_project = str(Path(project_dir).resolve()) if project_dir else None
    initial_config = (
        str(Path(initial_config).resolve()) if initial_config else None
    )

    # The single document the GUI is editing. Held in server state so undo/redo
    # stacks survive across requests. A worker pool runs the (seconds-long)
    # relaxation-engine validation off the event loop.
    #
    # ``project_dir`` is the CURRENTLY open project, not the launch one:
    # /api/files' sandbox root and the log tail's confinement both follow it,
    # and a root frozen at ``initial_project`` meant the file browser could not
    # navigate above the config dir of any project opened later.
    state = {"doc": None, "project_dir": initial_project}
    validate_jobs = {}
    validate_pool = ThreadPoolExecutor(max_workers=1)

    # One active run per project (a queue lands in G14). The handle lives on the
    # app instance so each create_app() -- including per-test apps -- is isolated.
    run_state = {"handle": None}

    # One Tune-tab solve/evaluator session per project (G10). Held on the app
    # instance so the dedicated worker process survives across requests.
    tune_state = {"session": None}
    tune_pool = ThreadPoolExecutor(max_workers=1)

    class OpenProjectRequest(BaseModel):
        path: str

    class OpenDocRequest(BaseModel):
        config_path: str
        params_path: str | None = None

    class CommandRequest(BaseModel):
        op: str
        args: dict = {}

    def _require_doc():
        doc = state["doc"]
        if doc is None:
            raise ValueError("no document is open; POST /api/doc/open first")
        return doc

    class RunRequest(BaseModel):
        config: str
        params: Optional[str] = None
        project_dir: Optional[str] = None

    class StopRequest(BaseModel):
        force: bool = False

    class UtilityRunRequest(BaseModel):
        name: str
        args: dict = {}
        cwd: Optional[str] = None

    class EligibleRequest(BaseModel):
        filename: str

    class TuneSolveRequest(BaseModel):
        # All optional: when omitted, the currently-open document supplies the
        # config, params, and working directory.
        config: Optional[dict] = None
        params: Optional[dict] = None
        workdir: Optional[str] = None

    class TuneEvalRequest(BaseModel):
        path: str
        value: float

    @app.get("/api/health")
    def health():
        return {"status": "ok", "service": "exozippy-gui"}

    @app.get("/api/config")
    def gui_config():
        """Client bootstrap: which project (and config, if any) to open on load."""
        return {
            "initial_project": initial_project,
            "initial_config": initial_config,
        }

    @app.get("/api/theme")
    def theme():
        """A COPY of the core chart style table -- never the source of it.

        `plot_theme` lives in the core package precisely so the CLI can draw
        PDFs with no server running (review 4.11.4: an endpoint-as-source was
        rejected because it would make the CLI depend on the GUI). This hands
        the same table to the browser so the frontend does not keep a second,
        independently-maintained copy of the palette and role colors -- which
        is exactly how the residual color came to differ between the PDF and
        the GUI.
        """
        from ..plot_theme import as_json

        return JSONResponse(as_json())

    @app.get("/api/schema")
    def schema():
        # Imported lazily: introspection pulls the full component stack, which
        # is heavy; keep it off the import path of a bare `exozippy-gui --help`.
        from ..introspect import full_schema

        return JSONResponse(full_schema())

    @app.get("/api/utilities")
    def utilities():
        from ..utilities.registry import all_utilities

        specs = all_utilities()
        return JSONResponse(
            {name: spec.to_schema() for name, spec in specs.items()}
        )

    def _reset_tune_session():
        """Drop the Tune session (solved values + live evaluator worker).

        Closing runs on a DETACHED thread, never inline: ``close()`` joins the
        worker subprocess (up to ~2 s), and a solve may still be in flight
        against it, so the request that triggered the reset must not wait. The
        abandoned session object keeps the old worker alive only until that
        thread finishes with it.
        """
        session = tune_state.get("session")
        tune_state["session"] = None
        if session is not None:
            threading.Thread(target=session.close, daemon=True).start()

    def _file_roots():
        """Directories whose files this server may read back to the client.

        The open project, the open document's directory (a config opened from
        elsewhere), and the active run's working directory -- the three trees
        every file the GUI legitimately shows comes from. Used by the log-tail
        socket; ``/api/files`` uses the project root alone, since its job is to
        browse the project.
        """
        roots = []
        if state.get("project_dir"):
            roots.append(state["project_dir"])
        doc = state.get("doc")
        if doc is not None and doc.config_path is not None:
            roots.append(str(doc.config_path.parent))
        handle = run_state.get("handle")
        if handle is not None:
            roots.append(handle.cwd)
        return roots

    @app.post("/api/project/open")
    def project_open(req: OpenProjectRequest):
        try:
            listing = open_project(req.path)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

        # Opening a project must not leave the PREVIOUS project's state behind.
        # Every piece of per-project state below outlived the switch, so
        # project B showed A's solved parameters and A's plots under B's name
        # until a re-Solve -- and an edit made against those stale values was
        # committed into B's params file (review 2.11.1).
        _reset_tune_session()
        state["project_dir"] = listing["dir"]

        from . import datafiles

        doc = state["doc"]
        if doc is not None and not datafiles.is_within(
            doc.config_path, listing["dir"]
        ):
            # A document from another project can no longer be what the GUI is
            # editing: /api/doc, every doc command, and the Solve fallback all
            # read this one slot, so keeping it would mean B's screen editing
            # A's files. Unsaved edits are flushed to the autosave sidecar
            # first, so re-opening A offers them back via `recovery`.
            if doc.dirty:
                try:
                    doc.autosave()
                except OSError:  # pragma: no cover - best effort
                    pass
            state["doc"] = None

        return JSONResponse(listing)

    # --- config document editing (G8) ----------------------------------------

    @app.post("/api/doc/open")
    def doc_open(req: OpenDocRequest):
        # Re-opening the file that is already open must NOT clobber unsaved
        # edits: several tabs call open on mount, so a naive reload-from-disk
        # here silently reverted every edit (and its undo stack) on any tab
        # switch. A dirty same-path doc is returned as-is; a clean one is
        # reloaded so external file edits are picked up.
        current = state["doc"]
        if (
            current is not None
            and current.dirty
            and current.config_path is not None
            and Path(req.config_path).resolve()
            == current.config_path.resolve()
        ):
            payload = current.to_json()
            payload["recovery"] = current.autosave_recovery()
            return JSONResponse(payload)
        try:
            doc = ProjectDocument.open(
                req.config_path, params_path=req.params_path
            )
        except (OSError, ValueError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        state["doc"] = doc
        payload = doc.to_json()
        payload["recovery"] = doc.autosave_recovery()
        return JSONResponse(payload)

    @app.get("/api/doc")
    def doc_get():
        try:
            return JSONResponse(_require_doc().to_json())
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)

    @app.post("/api/doc/command")
    def doc_command(req: CommandRequest):
        try:
            doc = _require_doc()
            command = command_from_json({"op": req.op, "args": req.args})
            doc.execute(command)
        except (ValueError, KeyError) as exc:
            # These carry hand-written, user-facing prose; pass it through.
            return JSONResponse({"error": str(exc)}, status_code=400)
        # IndexError and TypeError are BAD INPUT here, not server faults: a
        # path segment indexing a list the client's view is one delete stale
        # about, or one traversing a scalar, reaches _set_nested and raises
        # them. They 500'd, so the user saw a generic failure for an edit the
        # command's snapshot restore had already cleanly rolled back. Their
        # messages ("list index out of range") need the type for context.
        except (IndexError, TypeError) as exc:
            return JSONResponse(
                {
                    "error": (
                        f"{req.op}: {type(exc).__name__}: {exc} -- the edit "
                        f"was rejected and the document is unchanged."
                    )
                },
                status_code=400,
            )
        return JSONResponse(doc.to_json())

    @app.post("/api/doc/undo")
    def doc_undo():
        try:
            doc = _require_doc()
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)
        doc.undo()
        return JSONResponse(doc.to_json())

    @app.post("/api/doc/redo")
    def doc_redo():
        try:
            doc = _require_doc()
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)
        doc.redo()
        return JSONResponse(doc.to_json())

    @app.post("/api/doc/save")
    def doc_save():
        try:
            doc = _require_doc()
            doc.save()
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        return JSONResponse(doc.to_json())

    def _run_validation(job_id, config, user_params, workdir):
        from ..solve_api import validate

        try:
            diagnostics = validate(
                config, user_params=user_params, workdir=workdir
            )
            validate_jobs[job_id] = {
                "status": "done",
                "diagnostics": diagnostics,
            }
        except Exception as exc:  # pragma: no cover - defensive
            validate_jobs[job_id] = {
                "status": "error",
                "diagnostics": [
                    {
                        "severity": "error",
                        "message": f"{type(exc).__name__}: {exc}",
                        "param_paths": [],
                    }
                ],
            }

    @app.post("/api/doc/validate")
    def doc_validate():
        """Kick off a background validation and return a job id to poll.

        Validation runs the relaxation engine (seconds), so it must not block
        the event loop; it runs in a worker thread. Poll GET
        /api/doc/validate/{job_id}.
        """
        try:
            doc = _require_doc()
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)
        from .document import _jsonable

        config = _jsonable(doc.config)
        user_params = _jsonable(doc.params)
        workdir = str(doc.config_path.parent) if doc.config_path else None
        job_id = uuid.uuid4().hex
        # Pop-on-terminal-read (below) retires every job the frontend follows
        # to completion; this bounds the ABANDONED ones -- a validation whose
        # tab unmounted, or whose poll loop hit its own deadline first -- which
        # nothing else would ever remove. Oldest first: dicts keep insertion
        # order, and the oldest running job is the one least likely to be read.
        while len(validate_jobs) >= _MAX_VALIDATE_JOBS:
            validate_jobs.pop(next(iter(validate_jobs)), None)
        validate_jobs[job_id] = {"status": "running", "diagnostics": []}
        validate_pool.submit(
            _run_validation, job_id, config, user_params, workdir
        )
        return JSONResponse({"job_id": job_id, "status": "running"})

    @app.get("/api/doc/validate/{job_id}")
    def doc_validate_poll(job_id: str):
        job = validate_jobs.get(job_id)
        if job is None:
            return JSONResponse({"error": "no such job"}, status_code=404)
        # A finished job is read exactly once -- ConfigTab polls until it stops
        # saying "running" and then drops the id. Retiring it here is what
        # keeps the dict from growing one entry per edit for the life of the
        # server (the tab starts a fresh validation 1.2 s after every keystroke
        # burst), and every entry holds a full diagnostics list.
        if job.get("status") != "running":
            validate_jobs.pop(job_id, None)
        return JSONResponse({"job_id": job_id, **job})

    # --- data file manager (G9) ---------------------------------------------
    #
    # A schema-driven file browser + association menu. Nothing here names a
    # component: eligibility and associations flow entirely from the datafile
    # globs declared in the config schema.

    @app.get("/api/files")
    def files_list(dir: Optional[str] = None):
        """List a directory for the file browser, sandboxed to the project.

        The root follows the CURRENTLY open project (``state["project_dir"]``),
        not the one the server was launched with -- and ``list_directory`` now
        REFUSES a directory outside it rather than merely withholding the
        parent link.
        """
        from . import datafiles

        doc = state["doc"]
        doc_dir = (
            str(doc.config_path.parent)
            if doc is not None and doc.config_path
            else None
        )
        # With no project opened (a bare create_app(), i.e. only the tests),
        # the open document's own directory is the tightest honest root.
        root = state.get("project_dir") or doc_dir
        if dir is None:
            dir = doc_dir or root or os.getcwd()
        try:
            return JSONResponse(datafiles.list_directory(dir, root=root))
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

    @app.get("/api/browse")
    def browse_dirs(dir: Optional[str] = None):
        """List a directory for the sidebar project picker (no root confine).

        Unlike /api/files (which is sandboxed to the open project so the file
        browser cannot wander off), this powers the "Browse..." dialog whose
        whole job is to reach a *different* project, so it navigates up freely.
        It starts at the current project's parent, or the user's home.
        """
        from . import datafiles

        if dir is None:
            project = state.get("project_dir")
            if project:
                dir = str(Path(project).expanduser().resolve().parent)
            else:
                dir = os.path.expanduser("~")
        try:
            return JSONResponse(datafiles.list_directory(dir, root=None))
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

    def _schema():
        """The component schema, built at most once per process.

        ``full_schema()`` imports and walks the entire component stack, and
        these two endpoints called it fresh on every request. ``document``
        already memoizes exactly this; share that one cache rather than adding
        a second.
        """
        from .document import _default_schema

        return _default_schema()

    @app.post("/api/files/eligible")
    def files_eligible(req: EligibleRequest):
        """Schema-driven association menu: instance/key pairs for a filename."""
        from . import datafiles

        config = state["doc"].config if state["doc"] is not None else {}
        eligible = datafiles.eligible_associations(
            req.filename, config, _schema()
        )
        return JSONResponse({"eligible": eligible})

    @app.get("/api/files/associations")
    def files_associations():
        """Current file -> instance associations (chips), from the open doc."""
        from . import datafiles

        if state["doc"] is None:
            return JSONResponse({"associations": {}})
        assoc = datafiles.current_associations(state["doc"].config, _schema())
        return JSONResponse({"associations": assoc})

    # --- run controls (G11) -------------------------------------------------
    #
    # These endpoints are plain `def` (not `async def`), so FastAPI runs each in
    # a worker thread: a blocking start_run / stop / run_utility never stalls the
    # event loop that is also serving the log-tail WebSocket.

    @app.post("/api/run")
    def run_start(req: RunRequest):
        from ..gui import runner

        handle = run_state.get("handle")
        if handle is not None and handle.is_alive():
            return JSONResponse(
                {"error": "A run is already active for this project."},
                status_code=409,
            )

        cwd = req.project_dir or initial_project or os.getcwd()
        try:
            new_handle = runner.start_run(req.config, cwd=cwd)
        except Exception as exc:  # start_run failures surface as a 400
            return JSONResponse({"error": str(exc)}, status_code=400)

        # None (the usual case) -> the snapshot follows the config's own
        # parameter_file, which is what the fit subprocess reads.
        params_path = req.params
        if params_path and not os.path.isabs(params_path):
            params_path = os.path.join(cwd, params_path)
        _snapshot_run_inputs(new_handle, params_path or None)

        run_state["handle"] = new_handle
        return JSONResponse(run_status_payload(new_handle))

    @app.get("/api/run/status")
    def run_status():
        handle = run_state.get("handle")
        if handle is None:
            return {"active": False, "phase": "idle"}
        return JSONResponse(run_status_payload(handle))

    @app.post("/api/run/stop")
    def run_stop(req: StopRequest):
        handle = run_state.get("handle")
        if handle is None:
            return JSONResponse({"error": "No active run."}, status_code=400)
        handle.stop(force=req.force)
        return JSONResponse(run_status_payload(handle))

    @app.get("/api/run/plots")
    def run_plots():
        handle = run_state.get("handle")
        if handle is None:
            return {"start": [], "progress": []}
        return {
            "start": _list_prefix_images(handle, "_start*"),
            "progress": _list_prefix_images(handle, "_mcmc*"),
        }

    @app.get("/api/run/image")
    def run_image(path: str):
        # Serve a plot image, but only from inside the run's own tree -- never an
        # arbitrary path the query string asks for.
        from . import datafiles

        handle = run_state.get("handle")
        if handle is None:
            return JSONResponse({"error": "No active run."}, status_code=400)
        resolved = os.path.realpath(path)
        if not datafiles.is_within(resolved, handle.cwd) or not os.path.isfile(
            resolved
        ):
            return JSONResponse({"error": "forbidden"}, status_code=403)
        return FileResponse(resolved)

    @app.post("/api/utilities/run")
    def utilities_run(req: UtilityRunRequest):
        from ..utilities.registry import run_utility

        cwd = req.cwd or initial_project or os.getcwd()
        try:
            result = run_utility(req.name, req.args, cwd)
        except (KeyError, ValueError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

        # Persist the captured output so the terminal panel can tail it (the
        # utility already ran to completion; this is a static file to attach to).
        if "output" in result:
            log_path = os.path.join(cwd, f".exozippy_util_{req.name}.log")
            try:
                with open(log_path, "w") as fh:
                    fh.write(result.get("output") or "")
                result["log_path"] = log_path
            except OSError:
                pass
        return JSONResponse(result)

    # --- Tune tab: solve + live evaluator (G10) -----------------------------
    #
    # The heavy solve + pytensor compile runs in a dedicated worker PROCESS
    # (see gui/tune.py); these endpoints only broker it. Solve is kicked off on
    # a worker thread so the request returns immediately and the frontend polls
    # /api/tune/status through the solving -> compiling -> live phases.

    def _tune_session():
        from .tune import TuneSession

        session = tune_state.get("session")
        if session is None:
            session = TuneSession()
            tune_state["session"] = session
        return session

    def _tune_solve_inputs(req):
        """Resolve (config, params, workdir) from the request or the open doc."""
        config = req.config
        params = req.params
        workdir = req.workdir
        if config is None:
            from .document import _jsonable

            doc = _require_doc()
            config = _jsonable(doc.config)
            if params is None:
                params = _jsonable(doc.params)
            if workdir is None and doc.config_path is not None:
                workdir = str(doc.config_path.parent)
        return config, params or {}, workdir

    @app.post("/api/tune/solve")
    def tune_solve(req: TuneSolveRequest):
        try:
            config, params, workdir = _tune_solve_inputs(req)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        session = _tune_session()
        session.phase = "solving"
        session.error = None
        tune_pool.submit(session.solve, config, params, workdir)
        return JSONResponse(session.status())

    @app.get("/api/tune/status")
    def tune_status():
        session = tune_state.get("session")
        if session is None:
            return {
                "phase": "idle",
                "error": None,
                "structural_hash": None,
                "has_result": False,
                "has_data_plots": False,
            }
        return JSONResponse(session.status())

    @app.get("/api/tune/plots/data")
    def tune_data_plots():
        """Data-only plots for the in-flight solve (drawable pre-live)."""
        session = tune_state.get("session")
        plots = session.data_plots if session is not None else None
        return JSONResponse({"plots": plots or []})

    @app.get("/api/tune/result")
    def tune_result():
        session = tune_state.get("session")
        if session is None or session.result is None:
            return JSONResponse(
                {"error": "no solve result yet"}, status_code=409
            )
        return JSONResponse(session.result)

    @app.post("/api/tune/eval")
    def tune_eval(req: TuneEvalRequest):
        session = tune_state.get("session")
        if session is None:
            return JSONResponse(
                {"error": "no session; Solve first"}, status_code=409
            )
        try:
            return JSONResponse(session.eval(req.path, req.value))
        except RuntimeError as exc:
            return JSONResponse({"error": str(exc)}, status_code=409)

    @app.get("/api/tune/hash")
    def tune_hash():
        """Structural hash of the open document -- compare to the live hash to
        detect staleness after a bound/prior/fixed edit."""
        from ..evaluator import structural_hash
        from .document import _jsonable

        try:
            doc = _require_doc()
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)
        h = structural_hash(_jsonable(doc.config), _jsonable(doc.params))
        session = tune_state.get("session")
        live = session.structural_hash if session else None
        return {
            "structural_hash": h,
            "live_hash": live,
            "stale": live is not None and h != live,
        }

    @app.websocket("/api/logs")
    async def logs(websocket: WebSocket):
        # Refuse a cross-origin handshake BEFORE accepting it. WebSocket
        # handshakes bypass CORS entirely, so without this check any page the
        # user visits could open this socket against the loopback server and
        # read whatever it streams (the OS-random port is the only other
        # mitigation, and --port removes even that).
        if not _origin_is_local(websocket.headers.get("origin")):
            await websocket.close(code=1008)
            return
        await websocket.accept()
        file_path = websocket.query_params.get("file")
        if not file_path:
            await websocket.send_text("[error] no ?file= given")
            await websocket.close()
            return
        # gui.md invariant 5: file-serving endpoints stay confined to their
        # intended tree. The path arrives verbatim in the query string, so
        # without this the socket streamed ANY readable file on the machine.
        from . import datafiles

        roots = _file_roots()
        if not any(datafiles.is_within(file_path, root) for root in roots):
            await websocket.send_text(
                "[error] refusing to tail a file outside the open project "
                "or the active run's directory"
            )
            await websocket.close(code=1008)
            return
        await _tail_log(websocket, file_path)

    # Serve the prebuilt bundle at / with an SPA fallback. If the bundle has
    # not been built yet, serve a placeholder so the API is still usable.
    index_html = STATIC_DIR / "index.html"
    if index_html.exists():
        app.mount(
            "/",
            StaticFiles(directory=str(STATIC_DIR), html=True),
            name="static",
        )
    else:

        @app.get("/", response_class=HTMLResponse)
        def placeholder():
            return _PLACEHOLDER_HTML

    return app


_PLACEHOLDER_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>EXOZIPPy</title></head>
<body style="font-family: system-ui, sans-serif; background:#1a1d23; color:#e6e6e6;
 padding:3rem; max-width:44rem; margin:auto;">
<h1>EXOZIPPy GUI</h1>
<p>The frontend bundle has not been built yet.</p>
<pre style="background:#111; padding:1rem; border-radius:6px;">cd gui/frontend
npm install
npm run build</pre>
<p>The JSON API is live: try
<a style="color:#6cb6ff;" href="/api/schema">/api/schema</a> or
<a style="color:#6cb6ff;" href="/api/health">/api/health</a>.</p>
</body></html>
"""


# --- server + native window ---------------------------------------------------


def _find_free_port(host="127.0.0.1"):
    """Bind to port 0 to let the OS pick a free port, then release it."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def _serve(app, host, port):
    """Run uvicorn to completion (blocking). Used as a thread target."""
    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="warning")


def _wait_until_up(host, port, timeout=15.0):
    """Block until the server accepts a TCP connection, or time out."""
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.1)
    return False


def resolve_project_arg(project_arg, cwd=None):
    """Resolve the ``exozippy-gui [project]`` positional arg to (dir, config).

    ``project_arg`` may be None (default: ``cwd``, or ``os.getcwd()`` if that
    is also None), a project directory, or a specific config file (relative
    or absolute) -- e.g. ``kelt4.yaml``. A file resolves to its parent dir plus
    that file as the config to pre-select; a directory resolves to itself with
    no config pre-selected. Raises ValueError if a given path does not exist.
    """
    if not project_arg:
        return (cwd or os.getcwd()), None

    target = Path(project_arg).expanduser()
    if not target.exists():
        raise ValueError(f"no such path: {target}")
    target = target.resolve()
    if target.is_file():
        return str(target.parent), str(target)
    return str(target), None


def main(argv=None):
    """Entry point for the ``exozippy-gui`` console script."""
    parser = argparse.ArgumentParser(
        prog="exozippy-gui",
        description="Launch the EXOZIPPy graphical interface (local only).",
    )
    parser.add_argument(
        "project",
        nargs="?",
        default=None,
        help=(
            "Project directory or config file to open on launch, relative or "
            "absolute (e.g. 'kelt4.yaml'). Default: the current directory."
        ),
    )
    parser.add_argument(
        "--browser",
        action="store_true",
        help="Open a browser tab instead of a native pywebview window.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind (default 127.0.0.1; the GUI is local-only).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to bind (default: an OS-assigned free port).",
    )
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Serve the API/UI but do not open any window (for testing).",
    )
    args = parser.parse_args(argv)

    try:
        _require_fastapi()
    except RuntimeError as exc:
        parser.exit(status=1, message=f"{exc}\n")

    try:
        project_dir, initial_config = resolve_project_arg(args.project)
    except ValueError as exc:
        parser.exit(status=1, message=f"error: {exc}\n")

    app = create_app(project_dir=project_dir, initial_config=initial_config)
    host = args.host
    port = args.port or _find_free_port(host)
    url = f"http://{host}:{port}/"

    server_thread = threading.Thread(
        target=_serve,
        args=(app, host, port),
        daemon=True,
    )
    server_thread.start()

    if not _wait_until_up(host, port):
        print(f"error: server did not come up on {url}", file=sys.stderr)
        return 1

    print(f"EXOZIPPy GUI serving at {url}")

    if args.no_window:
        # Block on the server thread; Ctrl-C exits.
        try:
            server_thread.join()
        except KeyboardInterrupt:
            pass
        return 0

    if not args.browser:
        try:
            if sys.platform.startswith("linux"):
                # Force Mesa software rendering so QtWebEngine does not probe the
                # GPU DRI driver. Over X-forwarding / headless displays that probe
                # fails and spams "libGL error: ... failed to load driver: nouveau"
                # before falling back to software anyway; setdefault lets a user
                # who has working GL override it. WebGL is off regardless (plots
                # use SVG), so software GL costs nothing here.
                os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")

            import webview  # pywebview

            # On Linux force the Qt backend (the extra we ship). Left to probe,
            # pywebview tries GTK first and prints a noisy "No module named 'gi'"
            # ImportError before silently falling back to Qt; naming the backend
            # skips the GTK attempt entirely. macOS/Windows keep their native
            # default (Cocoa / EdgeChromium).
            gui_backend = "qt" if sys.platform.startswith("linux") else None
            webview.create_window("EXOZIPPy", url, width=1400, height=900)
            webview.start(gui=gui_backend)
            return 0
        except Exception as exc:  # pragma: no cover - env-dependent
            print(
                f"pywebview unavailable ({exc}); falling back to browser.",
                file=sys.stderr,
            )

    webbrowser.open(url)
    try:
        server_thread.join()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
