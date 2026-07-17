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
"""

import argparse
import asyncio
import os
import socket
import sys
import threading
import uuid
import webbrowser
from pathlib import Path

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


# --- app factory --------------------------------------------------------------

def create_app(project_dir=None):
    """Build and return the FastAPI application.

    Requires the 'gui' extra. ``project_dir`` seeds the initial project the
    frontend opens; it is served to the client via GET /api/config.
    """
    _require_fastapi()

    from concurrent.futures import ThreadPoolExecutor

    from fastapi import FastAPI, WebSocket
    from fastapi.responses import HTMLResponse, JSONResponse
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
