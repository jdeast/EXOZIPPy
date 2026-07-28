"""Subprocess entry point that builds a data-only PlotSpec preview (G9).

Run as ``python -m exozippy.gui.preview_worker``. It reads a JSON request on
stdin and writes a JSON response on stdout::

    request : {"config": {...}, "params": {...}, "workdir": "/abs/dir",
               "comp_type": "rvinstrument"}
    response: {"specs": [ <PlotSpec.to_json()>, ... ]}
           or {"error": "readable message"}

This runs in a SEPARATE process on purpose: a lightweight ``prepare()`` still
imports the full component stack and parses the data file, and a pathological
file could hang or crash. The parent kills this process on timeout, so the GUI
server itself can never be taken down by a bad preview. Errors from
``load_data`` (bad format) are caught and returned as a message -- surfacing
them in the pane is the whole point of the preview feature.
"""

import json
import os
import sys
import traceback


def _build_preview(request):
    """Prepare a System from the request and return data-only PlotSpec JSON."""
    config = request.get("config") or {}
    params = request.get("params") or {}
    workdir = request.get("workdir")
    comp_type = request.get("comp_type")

    if not comp_type:
        return {"error": "no component type given for preview"}

    if workdir:
        os.chdir(workdir)

    # Imported here (not at module top) so the import cost lands inside the
    # timed subprocess, never on the server's import path.
    from exozippy.system import System

    system = System(config, user_params=params)
    system.prepare()

    comp = system.active_components.get(comp_type)
    if comp is None:
        return {
            "error": (
                f"component '{comp_type}' is not present in this configuration"
            )
        }

    specs = comp.plot_data(system, point=None)
    return {"specs": [s.to_json() for s in specs]}


def main():
    try:
        request = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError) as exc:
        json.dump({"error": f"bad preview request: {exc}"}, sys.stdout)
        return 0

    try:
        result = _build_preview(request)
    except Exception as exc:  # noqa: BLE001 - any load_data/prepare error is the feature
        result = {
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }

    json.dump(result, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
