"""EXOZIPPy GUI support package.

Import-safe with no heavy dependencies: the two modules here (status, runner)
use only the standard library plus numpy/yaml (already core exozippy deps).
Nothing in this package imports fastapi, uvicorn, or any other optional GUI
extra, so `import exozippy.gui` always works in a bare install.

- status.py  -- GuiReporter: atomic status.json + partial-draw snapshot writer
                used by run.py / the PTDE samplers during a fit.
- runner.py  -- start_run / RunHandle / list_runs: launch a fit as a fresh
                subprocess, read its progress, and stop it gracefully.
"""

# Kept intentionally light: no eager submodule imports so this package stays
# cheap to import (runner and status pull numpy/yaml only when actually used).

TERMINAL_PHASES = frozenset({"done", "stopped", "error"})
"""Phases a GUI status file can end in; a live run never rests on these."""

__all__ = ["TERMINAL_PHASES"]
