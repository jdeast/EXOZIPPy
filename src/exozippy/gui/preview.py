"""Run a data-file preview in a worker subprocess with a hard timeout (G9).

:func:`run_preview` drives :mod:`exozippy.gui.preview_worker` in a separate
process so a pathological data file can never hang the GUI server: the child
gets a wall-clock timeout and is killed if it overruns. The result is always a
JSON-able dict -- either ``{"specs": [...]}`` or ``{"error": "..."}`` -- so the
frontend renders a chart or a readable message with no special-casing.
"""

import json
import subprocess
import sys

# Default wall-clock budget for a preview build. prepare() on a real RV/transit
# file is well under a second; a runaway parse or an accidental heavy config is
# what this guards against.
DEFAULT_TIMEOUT_S = 60.0


def run_preview(config, params, workdir, comp_type, timeout=DEFAULT_TIMEOUT_S):
    """Build data-only PlotSpecs for one component in a worker subprocess.

    Parameters
    ----------
    config, params : dict
        The current document's system config and parameter-override trees.
    workdir : str
        Directory the worker chdirs into so relative data-file paths resolve.
    comp_type : str
        The component yaml_key to preview (e.g. ``"rvinstrument"``).
    timeout : float
        Seconds before the worker is killed and a timeout error returned.

    Returns
    -------
    dict
        ``{"specs": [...]}`` on success, or ``{"error": "..."}`` on any
        failure (bad file, missing component, crash, or timeout).
    """
    request = json.dumps({
        "config": config,
        "params": params,
        "workdir": workdir,
        "comp_type": comp_type,
    })

    try:
        proc = subprocess.run(
            [sys.executable, "-m", "exozippy.gui.preview_worker"],
            input=request,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {
            "error": (
                f"Preview timed out after {timeout:.0f}s -- the data file may "
                f"be malformed or far larger than expected."
            )
        }

    out = (proc.stdout or "").strip()
    if not out:
        stderr = (proc.stderr or "").strip()
        detail = f": {stderr[-500:]}" if stderr else ""
        return {"error": f"Preview worker produced no output{detail}"}

    try:
        return json.loads(out)
    except json.JSONDecodeError:
        # The worker writes JSON to stdout; anything else is a hard crash whose
        # diagnostics land on stderr.
        stderr = (proc.stderr or "").strip()
        detail = f": {stderr[-500:]}" if stderr else ""
        return {"error": f"Preview worker crashed{detail}"}
