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
workdir).  Data file paths inside a config are relative to workdir, and both
functions RESOLVE them against it (_resolve_config_paths) rather than chdir'ing
into it: the working directory is process-global and these endpoints run on
FastAPI's threadpool, so two concurrent calls with different workdirs used to
be able to read each other's data files.  Nothing in stages 1-3 depends on the
process cwd any more.

Determinism: solve() IS reproducible, and this caveat used to say the
opposite.  The relaxation engine had a cross-build nondeterminism -- unsorted
walks of sympy `free_symbols` sets and an unsorted `rglob` of the component
directories, so ordering followed PYTHONHASHSEED and the filesystem -- and it
is fixed: every such walk is sorted, and a relation with several roots inside
the bounds breaks ties by value rather than by sp.solve's arrival order.
Two solve() calls on the same input return byte-identical values, bounds and
scales; verified across PYTHONHASHSEED 0/1/7 on ob08092, ob140939, kelt4
(rv+transit+sed) and DC2018_128, and tests/test_hashseed_determinism.py pins
the one remaining leak of this kind (MulensModel's set-ordered magnification
methods, patched in exozippy.compat).

What survives is weaker and is not about determinism: a SolveResult is ONE
valid solution, not a canonical one.  A system of relations can admit several,
and the engine picks by a fixed rule (lowest-provenance symbol, then roots
scored against the other relations) -- reproducible, but not unique.  Note
this never applied to bounds in the first place: the engine only READS
lower/upper, to reject roots outside the support.  It never writes them, so
exported bounds come from defaults.yaml, the component override channel and
the user's params file exactly as they would with no solve at all.

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


#: Config keys whose value is a filesystem path, and so the complete set
#: _resolve_config_paths rewrites against workdir.  Derived from the shipped
#: configs and from what stages 1-3 actually open: `file`/`files` (every data
#: component and the SED), `path` (the SED's file glob), `mask` (a flag file,
#: when it is a string rather than a list), `mmexofast` (an explicit seed JSON,
#: when it is a path rather than one of the `auto`/false keywords),
#: `parameter_file` (System reads it), and `prefix` (mulensinstrument builds
#: the MMEXOFAST cache path from it, and both reads and writes there).
_PATH_KEYS = (
    "file",
    "files",
    "path",
    "mask",
    "mmexofast",
    "parameter_file",
)

_GLOB_CHARS = "*?["


def _joined_if_real(value, workdir):
    """``workdir/value``, but only when that names something that exists.

    A string under a path key is not always a path: `mmexofast: auto` and a
    `mask:` given as a list of row indices share the key with real paths.
    Probing keeps those untouched, and keeps a genuinely missing file
    reporting the spelling the user wrote rather than an absolute path they
    never typed.  A glob is probed by its directory, since the pattern itself
    never "exists".
    """
    if not isinstance(value, str) or not value or os.path.isabs(value):
        return value
    candidate = os.path.join(workdir, value)
    probe = candidate
    if any(c in value for c in _GLOB_CHARS):
        probe = os.path.dirname(candidate) or workdir
    return candidate if os.path.exists(probe) else value


def _resolve_config_paths(config, workdir):
    """A copy of ``config`` with its relative paths resolved against workdir.

    THE ALTERNATIVE IS os.chdir, AND IT IS NOT THREAD SAFE.  This module's
    solve()/validate() are called from FastAPI, which runs sync endpoints on a
    threadpool, and the working directory is PROCESS-global: two concurrent
    validates with different workdirs would read each other's data files --
    silently, since a same-named file in the other project is a perfectly good
    read.  (The GUI's tune worker also chdirs, and that one is fine: it is a
    separate process.)  Resolving the paths up front means nothing in stages
    1-3 depends on the process cwd, so the calls no longer interact at all.

    Only the keys in _PATH_KEYS are touched, and only when the joined path
    really exists -- see _joined_if_real.  A future component inventing a new
    path key fails LOUDLY here (file not found, naming the path) instead of
    reading whatever the server's cwd happens to hold; add the key.

    `prefix` is handled separately: it is an OUTPUT path, so it need not exist
    yet, but mulensinstrument reads and writes the MMEXOFAST cache next to it
    during stage 1a and that must land in the project directory exactly as it
    did under chdir.
    """
    import copy

    resolved = copy.deepcopy(config)

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key in _PATH_KEYS:
                    if isinstance(value, (list, tuple)):
                        node[key] = [
                            _joined_if_real(v, workdir) for v in value
                        ]
                    else:
                        node[key] = _joined_if_real(value, workdir)
                else:
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(resolved)

    prefix = resolved.get("prefix")
    if isinstance(prefix, str) and prefix and not os.path.isabs(prefix):
        resolved["prefix"] = os.path.join(workdir, prefix)

    return resolved


def _prepare_system(config, user_params, workdir):
    """Build a fresh System and run stages 1-3 with paths rooted at workdir.

    Returns (system, warnings).  Never builds the PyMC model.  Does NOT change
    the process working directory (see _resolve_config_paths for why).  Any
    exception raised by prepare propagates to the caller.
    """
    collector = _WarningCollector()
    pkg_logger = logging.getLogger("exozippy")
    pkg_logger.addHandler(collector)

    if workdir:
        config = _resolve_config_paths(config, workdir)
    try:
        # System accepts user_params as a dict directly; when None it reads the
        # config's parameter_file, which _resolve_config_paths has already
        # rooted at workdir.
        system = System(config, user_params=user_params)
        system.prepare()  # stages 1-3 only -- never build_model()
    finally:
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
            diags.append(
                {
                    "severity": "error",
                    "message": (
                        f"initval {value:.6g} for '{path}' is below its lower "
                        f"bound {lower:.6g}; no in-bounds start exists."
                    ),
                    "param_paths": [path],
                }
            )
        elif upper is not None and value > upper:
            diags.append(
                {
                    "severity": "error",
                    "message": (
                        f"initval {value:.6g} for '{path}' is above its upper "
                        f"bound {upper:.6g}; no in-bounds start exists."
                    ),
                    "param_paths": [path],
                }
            )
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
    export = system.config_manager.export_solution(
        derived_params=system.derived_elements(),
        active_elements=system.active_elements(),
        manifest_overrides=system.manifest_overrides(),
    )
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
        return [
            {
                "severity": "error",
                "message": f"{type(e).__name__}: {e}",
                "param_paths": [],
            }
        ]

    export = system.config_manager.export_solution(
        derived_params=system.derived_elements(),
        active_elements=system.active_elements(),
        manifest_overrides=system.manifest_overrides(),
    )
    diagnostics = list(system.config_manager.diagnostics)
    diagnostics.extend(_bounds_diagnostics(export.get("parameters", {})))
    return diagnostics
