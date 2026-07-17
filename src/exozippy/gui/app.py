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
    GET  /api/tune/result     -- solved parameters + base PlotSpecs
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

# The built frontend bundle lives here (committed to the wheel). It may be
# absent in a source checkout that has not run `npm run build` yet; we degrade
# to a small placeholder page rather than failing to import.
STATIC_DIR = Path(__file__).parent / "static"

# YAML files that look like parameter-override files rather than system configs.
_PARAMS_SUFFIXES = (".params.yaml", ".params.yml")

# Data-file extensions worth surfacing in the project listing (preview/assoc.
# in later prompts). Kept generic -- the datafile schema (G1) is the real
# authority; this is only a listing convenience.
_DATA_EXTS = (
    ".sed", ".rv", ".tran", ".dat", ".txt", ".csv", ".eph",
    ".priors", ".fits", ".json",
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

def _classify_yaml(path):
    """Return 'params' for a parameter-override file, else 'config'."""
    name = path.name.lower()
    return "params" if name.endswith(_PARAMS_SUFFIXES) else "config"


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
            (params if kind == "params" else configs).append(entry)
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


async def _tail_log(websocket, file_path, from_lines=200, poll_s=0.5):
    """Stream a growing log file over a WebSocket, following rotation.

    Sends the last ``from_lines`` lines on connect, then polls for appended
    content. If the file shrinks (truncation) or its inode changes (rotation),
    reopens from the start. Runs until the client disconnects.
    """
    from fastapi import WebSocketDisconnect

    path = Path(file_path)

    # Seed with the tail so the user sees recent history immediately.
    for line in _read_last_lines(path, from_lines):
        await websocket.send_text(line.rstrip("\n"))

    fh = None
    inode = None
    try:
        while True:
            try:
                st = path.stat()
                # (Re)open on first sight, rotation (new inode), or truncation.
                reopen = (
                    fh is None
                    or inode != st.st_ino
                    or fh.tell() > st.st_size
                )
                if reopen:
                    if fh is not None:
                        fh.close()
                    fh = open(path, "r", errors="replace")
                    inode = st.st_ino
                    # On a fresh open after seeding, jump to the end so we do
                    # not resend the tail we already sent.
                    fh.seek(0, os.SEEK_END)
                line = fh.readline()
                if line:
                    await websocket.send_text(line.rstrip("\n"))
                    continue
            except FileNotFoundError:
                # File not created yet (or mid-rotation); wait and retry.
                fh = None
                inode = None
            # No new data -- yield, and let a client disconnect surface.
            await asyncio.sleep(poll_s)
    except WebSocketDisconnect:
        pass
    finally:
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
    """
    status = handle.status()
    return {
        "active": True,
        "phase": status.get("phase"),
        "state": status.get("state", {}),
        "alive": status.get("alive"),
        "pid": status.get("pid"),
        "returncode": status.get("returncode"),
        "error": status.get("error"),
        "prefix": handle.prefix,
        "config_path": handle.config_path,
        "cwd": handle.cwd,
        "log_path": _log_path(handle),
        "results_dir": _results_dir(handle),
        "snapshot": _read_snapshot_meta(handle),
    }


def _list_prefix_images(handle, pattern):
    """Sorted absolute image paths matching <prefix><pattern> (e.g. '_start*')."""
    out = []
    for path in sorted(glob.glob(_prefix_path(handle) + pattern)):
        if os.path.splitext(path)[1].lower() in _IMAGE_EXTS and os.path.isfile(path):
            out.append(path)
    return out


def _snapshot_run_inputs(handle, params_path=None):
    """Copy the exact config/params used into the output dir for reproducibility.

    Writes '<stem>.used<ext>' beside the run's outputs so a finished fit always
    carries a frozen copy of what produced it, even if the source yaml is later
    edited. Copying onto the source path is skipped. Best-effort: an I/O error
    never blocks the run. Returns the list of copies made.
    """
    results_dir = _results_dir(handle)
    try:
        os.makedirs(results_dir, exist_ok=True)
    except OSError:
        return []

    src_config = handle.config_path
    if not os.path.isabs(src_config):
        src_config = os.path.join(handle.cwd, src_config)

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

def create_app(project_dir=None):
    """Build and return the FastAPI application.

    Requires the 'gui' extra. ``project_dir`` seeds the initial project the
    frontend opens; it is served to the client via GET /api/config.
    """
    _require_fastapi()

    from concurrent.futures import ThreadPoolExecutor

    from fastapi import FastAPI, WebSocket
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel

    from .document import ProjectDocument, command_from_json

    app = FastAPI(title="EXOZIPPy", docs_url="/api/docs", openapi_url="/api/openapi.json")

    initial_project = str(Path(project_dir).resolve()) if project_dir else None

    # The single document the GUI is editing. Held in server state so undo/redo
    # stacks survive across requests. A worker pool runs the (seconds-long)
    # relaxation-engine validation off the event loop.
    state = {"doc": None}
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
        """Client bootstrap: which project (if any) to open on load."""
        return {"initial_project": initial_project}

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
        return JSONResponse({name: spec.to_schema() for name, spec in specs.items()})

    @app.post("/api/project/open")
    def project_open(req: OpenProjectRequest):
        try:
            return JSONResponse(open_project(req.path))
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

    # --- config document editing (G8) ----------------------------------------

    @app.post("/api/doc/open")
    def doc_open(req: OpenDocRequest):
        try:
            doc = ProjectDocument.open(req.config_path, params_path=req.params_path)
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
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        except KeyError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
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

    @app.post("/api/doc/autosave")
    def doc_autosave():
        try:
            doc = _require_doc()
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)
        written = doc.autosave()
        return JSONResponse({"written": [str(p) for p in written]})

    def _run_validation(job_id, config, user_params, workdir):
        from ..solve_api import validate

        try:
            diagnostics = validate(config, user_params=user_params, workdir=workdir)
            validate_jobs[job_id] = {"status": "done", "diagnostics": diagnostics}
        except Exception as exc:  # pragma: no cover - defensive
            validate_jobs[job_id] = {
                "status": "error",
                "diagnostics": [
                    {"severity": "error",
                     "message": f"{type(exc).__name__}: {exc}",
                     "param_paths": []}
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
        return JSONResponse({"job_id": job_id, **job})

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
                status_code=409)

        cwd = req.project_dir or initial_project or os.getcwd()
        try:
            new_handle = runner.start_run(req.config, cwd=cwd)
        except Exception as exc:  # start_run failures surface as a 400
            return JSONResponse({"error": str(exc)}, status_code=400)

        params_path = req.params
        if params_path and not os.path.isabs(params_path):
            params_path = os.path.join(cwd, params_path)
        _snapshot_run_inputs(new_handle, params_path)

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
        handle = run_state.get("handle")
        if handle is None:
            return JSONResponse({"error": "No active run."}, status_code=400)
        resolved = os.path.realpath(path)
        root = os.path.realpath(handle.cwd)
        try:
            inside = os.path.commonpath([resolved, root]) == root
        except ValueError:
            inside = False
        if not inside or not os.path.isfile(resolved):
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
            return {"phase": "idle", "error": None,
                    "structural_hash": None, "has_result": False}
        return JSONResponse(session.status())

    @app.get("/api/tune/result")
    def tune_result():
        session = tune_state.get("session")
        if session is None or session.result is None:
            return JSONResponse({"error": "no solve result yet"}, status_code=409)
        return JSONResponse(session.result)

    @app.post("/api/tune/eval")
    def tune_eval(req: TuneEvalRequest):
        session = tune_state.get("session")
        if session is None:
            return JSONResponse({"error": "no session; Solve first"},
                                status_code=409)
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
        return {"structural_hash": h, "live_hash": live,
                "stale": live is not None and h != live}

    @app.websocket("/api/logs")
    async def logs(websocket: WebSocket):
        await websocket.accept()
        file_path = websocket.query_params.get("file")
        if not file_path:
            await websocket.send_text("[error] no ?file= given")
            await websocket.close()
            return
        await _tail_log(websocket, file_path)

    # Serve the prebuilt bundle at / with an SPA fallback. If the bundle has
    # not been built yet, serve a placeholder so the API is still usable.
    index_html = STATIC_DIR / "index.html"
    if index_html.exists():
        app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
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


def main(argv=None):
    """Entry point for the ``exozippy-gui`` console script."""
    parser = argparse.ArgumentParser(
        prog="exozippy-gui",
        description="Launch the EXOZIPPy graphical interface (local only).",
    )
    parser.add_argument(
        "project", nargs="?", default=None,
        help="Project directory to open on launch (default: none).",
    )
    parser.add_argument(
        "--browser", action="store_true",
        help="Open a browser tab instead of a native pywebview window.",
    )
    parser.add_argument(
        "--host", default="127.0.0.1",
        help="Host to bind (default 127.0.0.1; the GUI is local-only).",
    )
    parser.add_argument(
        "--port", type=int, default=None,
        help="Port to bind (default: an OS-assigned free port).",
    )
    parser.add_argument(
        "--no-window", action="store_true",
        help="Serve the API/UI but do not open any window (for testing).",
    )
    args = parser.parse_args(argv)

    try:
        _require_fastapi()
    except RuntimeError as exc:
        parser.exit(status=1, message=f"{exc}\n")

    app = create_app(project_dir=args.project)
    host = args.host
    port = args.port or _find_free_port(host)
    url = f"http://{host}:{port}/"

    server_thread = threading.Thread(
        target=_serve, args=(app, host, port), daemon=True,
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
            import webview  # pywebview

            webview.create_window("EXOZIPPy", url, width=1400, height=900)
            webview.start()
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
