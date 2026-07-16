"""In-memory solve / validate API for the EXOZIPPy relaxation engine.

This module is the headless entry point a GUI (or any non-CLI caller) uses to
answer two questions about a configuration WITHOUT building the PyMC model:

  1. solve(config, user_params, workdir) -> SolveResult
     "Solve this config and tell me every parameter's value, unit, bounds, and
     WHERE the value came from."  Runs only lifecycle stages 1-3
     (System.prepare(): data I/O, registration, symbolic relaxation) and reads
     back the in-memory solution via ConfigManager.export_solution().

  2. validate(config, user_params, workdir) -> list[dict]
     "Validate this config and give me structured contradiction diagnostics."
     Runs the same stages, catches any exception the engine raises, converts
     the engine's structured contradiction list, and adds a bounds check
     (a user initval that falls outside its resolved [lower, upper]).

Both functions accept user_params either as an already-loaded dict or as None
(in which case System reads the config's parameter_file, resolved relative to
workdir).  Data file paths inside a config are relative to workdir, so both
functions chdir into it for the duration of the solve and restore the previous
directory afterwards.

Determinism caveat: the relaxation engine has a known cross-build
nondeterminism -- two identical prepares can pick different derived bounds for
the same parameter (the lp for one raw point can differ across builds).  A
SolveResult therefore reports ONE valid solution, not a canonical one; do not
assume two solve() calls on the same input return byte-identical bounds for
solved quantities.  This module does not attempt to fix that; it only exposes
whatever the engine produced.

solve() builds a fresh System (and therefore a fresh ConfigManager) on every
call, so it is safe to call repeatedly in one process; it does not rely on any
mutable module-level state of its own.
"""

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

from exozippy.system import System

logger = logging.getLogger(__name__)


@dataclass
class SolveResult:
    """JSON-serializable result of solve().

    Fields:
      parameters: {user_path: {value, unit, internal_unit, lower, upper,
        init_scale, sigma, mu, fixed, derived, provenance}}.  provenance is
        {rank, label, relation}; label is one of "user" | "data" | "solved" |
        "default".  Numeric fields are in each parameter's user unit.
      seeds: list of {user_path: value} start points, present (non-None) only
        when multi-seed sampling produced more than one seed; otherwise None.
      warnings: log warnings emitted during prepare (strings).
      diagnostics: structured contradiction diagnostics (see validate()).
      elapsed_s: wall-clock seconds spent in prepare()+export.
    """

    parameters: dict = field(default_factory=dict)
    seeds: Optional[list] = None
    warnings: list = field(default_factory=list)
    diagnostics: list = field(default_factory=list)
    elapsed_s: float = 0.0

    def as_dict(self):
        """Return a plain dict suitable for json.dumps()."""
        return {
            "parameters": self.parameters,
            "seeds": self.seeds,
            "warnings": self.warnings,
            "diagnostics": self.diagnostics,
            "elapsed_s": self.elapsed_s,
        }


class _WarningCollector(logging.Handler):
    """Collect WARNING+ records emitted on the exozippy logger during prepare."""

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.messages = []

    def emit(self, record):
        try:
            self.messages.append(record.getMessage())
        except Exception:
            pass


def _prepare_system(config, user_params, workdir):
    """Build a fresh System and run stages 1-3 from within workdir.

    Returns (system, warnings).  Never builds the PyMC model.  Restores the
    previous working directory even on error.  Any exception raised by prepare
    propagates to the caller.
    """
    collector = _WarningCollector()
    pkg_logger = logging.getLogger("exozippy")
    pkg_logger.addHandler(collector)

    prev_cwd = os.getcwd()
    try:
        if workdir:
            os.chdir(workdir)
        # System accepts user_params as a dict directly; when None it reads the
        # config's parameter_file relative to the (now chdir'd) working dir.
        system = System(config, user_params=user_params)
        system.prepare()  # stages 1-3 only -- never build_model()
    finally:
        os.chdir(prev_cwd)
        pkg_logger.removeHandler(collector)

    return system, collector.messages


def _bounds_diagnostics(parameters):
    """Flag parameters whose value falls outside their resolved [lower, upper].

    Operates on the exported parameter dict (user units).  Fixed and derived
    parameters are skipped -- derived values are already bounds-filtered by the
    solver, and a fixed parameter is not sampled.
    """
    diags = []
    for path, info in parameters.items():
        if info.get("fixed") or info.get("derived"):
            continue
        value = info.get("value")
        lower = info.get("lower")
        upper = info.get("upper")
        if value is None:
            continue
        if lower is not None and value < lower:
            diags.append({
                "severity": "error",
                "message": (
                    f"initval {value:.6g} for '{path}' is below its lower "
                    f"bound {lower:.6g}; no in-bounds start exists."),
                "param_paths": [path],
            })
        elif upper is not None and value > upper:
            diags.append({
                "severity": "error",
                "message": (
                    f"initval {value:.6g} for '{path}' is above its upper "
                    f"bound {upper:.6g}; no in-bounds start exists."),
                "param_paths": [path],
            })
    return diags


def solve(config, user_params=None, workdir=None):
    """Solve a configuration and report every parameter's resolved state.

    Args:
      config: the parsed system-config dict (as loaded from *.yaml).
      user_params: the parsed params-override dict, or None to load the
        config's parameter_file (relative to workdir).
      workdir: directory the config's data-file paths are relative to; solve
        runs from here.  None means the current directory.

    Returns a SolveResult.  Runs only System.prepare() (stages 1-3); it never
    builds the PyMC model.  Safe to call repeatedly in one process.
    """
    start = time.time()
    system, warnings = _prepare_system(config, user_params, workdir)
    export = system.config_manager.export_solution()
    elapsed = time.time() - start

    parameters = export.get("parameters", {})
    diagnostics = list(system.config_manager.diagnostics)
    diagnostics.extend(_bounds_diagnostics(parameters))

    return SolveResult(
        parameters=parameters,
        seeds=export.get("seeds"),
        warnings=list(warnings),
        diagnostics=diagnostics,
        elapsed_s=elapsed,
    )


def validate(config, user_params=None, workdir=None):
    """Validate a configuration and return structured diagnostics.

    Returns a list of {severity, message, param_paths} dicts.  Never raises for
    a modeling-level contradiction: an exception raised during prepare is
    caught and converted into a single "error" diagnostic; the engine's
    structured contradiction list and a bounds check contribute the rest.  An
    empty list means no contradictions were found.
    """
    try:
        system, _ = _prepare_system(config, user_params, workdir)
    except Exception as e:
        return [{
            "severity": "error",
            "message": f"{type(e).__name__}: {e}",
            "param_paths": [],
        }]

    export = system.config_manager.export_solution()
    diagnostics = list(system.config_manager.diagnostics)
    diagnostics.extend(_bounds_diagnostics(export.get("parameters", {})))
    return diagnostics
