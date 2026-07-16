"""
Compiled forward evaluator for the EXOZIPPy GUI (prompt G5).

The GUI has a hybrid interaction model.  The user presses "Solve" (runs
the relaxation engine + builds the model + compiles predictors -- a
seconds-scale step), then drags parameter sliders and expects the model
curves on each chart to re-render in milliseconds.  A STRUCTURAL change
(a component added or removed, a bound or a fixed/free flag changed)
invalidates the compiled graph and forces another "Solve".

This module supplies the millisecond half of that loop:

  * ``compile_evaluator(system, model) -> Evaluator`` captures the base
    PlotSpecs (G4) and their retained symbolic nodes.
  * ``Evaluator.eval_plots(raw_point)`` returns, per plot, the updated
    model-trace y-arrays, using pytensor functions compiled directly
    from the retained nodes -- taking the model's raw free variables as
    inputs, compiled lazily per plot and cached, with compile times
    logged.
  * ``Evaluator.set_value(path, value_user, raw_point)`` maps a slider
    value (user units, physical parameter) back to a raw point by
    inverting the Parameter's logit/linear transform.  Parameters with
    tensor-valued (linked/dynamic) bounds cannot be inverted statically
    and raise :class:`NeedsResolve`, which the GUI catches to trigger a
    re-solve.
  * ``structural_hash(config, user_params)`` produces a stable hash over
    the component set, per-parameter bounds/fixed-ness/expression wiring
    and data file list -- insensitive to dict key order and to pure
    initval changes.  The GUI compares hashes to decide stale vs live.

Only model-role traces are evaluated; data traces never change with a
slider.  See exozippy.plotspec for the PlotSpec/Trace contract and
Component.plot_data for how the nodes are produced and retained.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytensor
from pytensor.compile import Function
from pytensor.compile.sharedvalue import SharedVariable
from pytensor.graph.basic import Constant

try:  # moved between pytensor releases
    from pytensor.graph.traversal import ancestors
except ImportError:  # pragma: no cover - version fallback
    from pytensor.graph.basic import ancestors

from exozippy.components.parameter import SeedBoundViolation

logger = logging.getLogger(__name__)


class NeedsResolve(Exception):
    """Raised by :meth:`Evaluator.set_value` when a parameter element cannot
    be moved by inverting a static transform -- because its bounds are
    tensor-valued (a user link / dynamic bound) or the element is not a
    sampled free variable (fixed, derived, or hard-linked).

    The GUI catches this to fall back to a full re-solve (the compiled
    functions may no longer describe the model once such an element moves).
    """


# ---------------------------------------------------------------------------
# Structural hash
# ---------------------------------------------------------------------------

# Config keys that carry no structural meaning for the compiled graph.
_NON_STRUCTURAL_CONFIG_KEYS = {"run"}


def _canon(value: Any) -> Any:
    """Canonicalize a scalar for hashing (floats -> repr-stable form)."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    return value


def _gather_files(obj: Any) -> List[str]:
    """Recursively collect every ``file``/``files`` value in a config tree."""
    found: List[str] = []

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k in ("file", "files"):
                    if isinstance(v, str):
                        found.append(v)
                    elif isinstance(v, (list, tuple)):
                        found.extend(str(x) for x in v)
                    else:
                        found.append(str(v))
                else:
                    walk(v)
        elif isinstance(o, (list, tuple)):
            for item in o:
                walk(item)

    walk(obj)
    return found


def _component_skeleton(config: dict) -> dict:
    """Structural view of the component set: top-level keys and, for
    list-valued component blocks, the sorted instance names.  Numeric or
    initval-like leaves are deliberately excluded."""
    skel: Dict[str, Any] = {}
    for key, block in config.items():
        if key in _NON_STRUCTURAL_CONFIG_KEYS:
            continue
        if isinstance(block, list):
            names = sorted(
                str(item.get("name", i))
                for i, item in enumerate(block)
                if isinstance(item, dict)
            )
            skel[key] = names if names else len(block)
        elif isinstance(block, dict):
            # single-instance component (e.g. sed): presence is structural
            skel[key] = True
        else:
            skel[key] = True
    return skel


def _param_structure(user_params: Optional[dict]) -> dict:
    """Per-parameter structural fields from user_params: bounds, fixed-ness,
    and expression wiring (string links).  Pure numeric initval/mu changes
    are intentionally omitted so the hash ignores them."""
    struct: Dict[str, Any] = {}
    for path, spec in (user_params or {}).items():
        if not isinstance(spec, dict):
            continue
        entry: Dict[str, Any] = {}
        # Bounds are structural (they set up the logit transform).
        if "lower" in spec and not isinstance(spec["lower"], str):
            entry["lower"] = _canon(spec["lower"])
        if "upper" in spec and not isinstance(spec["upper"], str):
            entry["upper"] = _canon(spec["upper"])
        # Fixed-ness: sigma == 0 fixes the parameter (only the flag matters,
        # not the magnitude of a positive sigma).
        if "sigma" in spec and not isinstance(spec["sigma"], str):
            try:
                entry["fixed"] = float(spec["sigma"]) == 0.0
            except (TypeError, ValueError):
                pass
        # Expression wiring: any string-valued field is a link (structural).
        for field in ("initval", "mu", "lower", "upper", "sigma", "init_scale"):
            if isinstance(spec.get(field), str):
                entry[f"{field}_link"] = spec[field]
        if entry:
            struct[str(path)] = entry
    return struct


def structural_hash(config: dict, user_params: Optional[dict] = None) -> str:
    """Stable structural fingerprint of a system configuration.

    The hash covers the component set, per-parameter bounds/fixed-ness and
    expression wiring, and the data file list.  It is insensitive to dict
    key order (canonicalized via ``sort_keys``) and to pure initval changes
    (numeric initval/mu values are not included).  Two configs with the same
    hash may share the same compiled evaluator; a changed hash means the GUI
    must re-solve.
    """
    payload = {
        "components": _component_skeleton(config or {}),
        "files": sorted(_gather_files(config or {})),
        "params": _param_structure(user_params),
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class _ModelTrace:
    """Bookkeeping for one model-role Trace: its owning plot, the symbolic
    node, the extra (non-free-variable) inputs that must be frozen, and the
    affine map from node output to the plotted (display-unit) y-array."""

    __slots__ = ("plot_id", "name", "node", "givens", "affine", "raw_ok",
                 "base_x", "base_y")

    def __init__(self, plot_id, name, node, base_x, base_y):
        self.plot_id = plot_id
        self.name = name
        self.node = node
        self.base_x = base_x
        self.base_y = base_y
        self.givens: List[Tuple[Any, np.ndarray]] = []
        self.affine: Optional[Tuple[float, float]] = None
        self.raw_ok = False   # True once affine/raw output validated at base


class Evaluator:
    """Millisecond forward evaluator for GUI slider updates.

    Construct via :func:`compile_evaluator`.  Holds the base PlotSpecs and,
    per plot, a lazily-compiled pytensor function mapping the model's raw
    free variables to that plot's model-trace node outputs.  Extra inputs
    (time grids, instrument indices) are frozen to the values used to build
    the base traces; display-unit conversion is recovered as a per-trace
    affine fit so ``eval_plots`` returns arrays directly comparable to the
    base PlotSpec y-values.
    """

    def __init__(self, system, model, specs, base_raw_point, model_traces,
                 free_rvs):
        self.system = system
        self.model = model
        self.specs = specs
        self.base_raw_point = base_raw_point
        self._free_rvs = free_rvs
        self._free_names = [v.name for v in free_rvs]
        # model traces grouped by plot id, in plot/trace order
        self._traces_by_plot: Dict[str, List[_ModelTrace]] = {}
        for mt in model_traces:
            self._traces_by_plot.setdefault(mt.plot_id, []).append(mt)
        # lazily-compiled per-plot functions: plot_id -> (Function, [traces])
        self._fn_cache: Dict[str, Tuple[Function, List[_ModelTrace]]] = {}
        # parameter lookup for set_value
        self._param_lookup = system.get_parameter_lookup()

    # -- compilation ------------------------------------------------------

    def _inputs_for(self, raw_point) -> List[np.ndarray]:
        """Positional input values in model.free_RVs order."""
        return [np.asarray(raw_point[name]) for name in self._free_names]

    def _compile_plot(self, plot_id) -> Tuple[Function, List[_ModelTrace]]:
        """Compile (once) the pytensor function for a plot's model traces."""
        if plot_id in self._fn_cache:
            return self._fn_cache[plot_id]

        traces = self._traces_by_plot.get(plot_id, [])
        outputs = [mt.node for mt in traces]
        givens: List[Tuple[Any, np.ndarray]] = []
        seen = set()
        for mt in traces:
            for var, val in mt.givens:
                if id(var) not in seen:
                    seen.add(id(var))
                    givens.append((var, val))

        t0 = time.perf_counter()
        fn = pytensor.function(
            inputs=list(self._free_rvs),
            outputs=outputs,
            givens=givens,
            on_unused_input="ignore",
        )
        dt = time.perf_counter() - t0
        logger.info("Evaluator: compiled plot '%s' (%d model trace(s)) in "
                    "%.3f s", plot_id, len(traces), dt)

        self._fn_cache[plot_id] = (fn, traces)
        return fn, traces

    def _apply_display(self, mt: _ModelTrace, node_out: np.ndarray) -> np.ndarray:
        """Map a node output to display units for the given trace."""
        arr = np.asarray(node_out, dtype=float)
        if mt.affine is not None:
            a, b = mt.affine
            return a * arr + b
        return arr

    # -- public API -------------------------------------------------------

    def eval_plots(self, raw_point) -> Dict[str, Dict[str, np.ndarray]]:
        """Evaluate every plot's model traces at ``raw_point``.

        Returns ``{plot_id: {trace_name: y_array}}`` with the updated
        model-trace y-arrays (data traces are omitted -- they never change).
        Functions are compiled on first touch and cached, so repeated calls
        are millisecond-scale.
        """
        inputs = self._inputs_for(raw_point)
        out: Dict[str, Dict[str, np.ndarray]] = {}
        for plot_id in self._traces_by_plot:
            fn, traces = self._compile_plot(plot_id)
            results = fn(*inputs)
            if not isinstance(results, (list, tuple)):
                results = [results]
            plot_out: Dict[str, np.ndarray] = {}
            for mt, node_out in zip(traces, results):
                plot_out[mt.name] = self._apply_display(mt, node_out)
            out[plot_id] = plot_out
        return out

    def _resolve_param(self, path: str):
        """Return (Parameter, element_index) for a user-facing path.

        Accepts ``comp.param``, ``comp.<index>.param`` and
        ``comp.<name>.param`` forms (the standard three).  Raises KeyError
        if the parameter is unknown.
        """
        parts = path.split(".")
        if len(parts) == 2:
            label, inst = path, None
        elif len(parts) == 3:
            label = f"{parts[0]}.{parts[2]}"
            inst = parts[1]
        else:
            raise KeyError(f"Unrecognized parameter path '{path}'")

        par = self._param_lookup.get(label)
        if par is None:
            raise KeyError(f"Parameter '{path}' (label '{label}') not found")

        if inst is None:
            elem = 0
        elif inst.isdigit():
            elem = int(inst)
        elif par.names and inst in list(par.names):
            elem = list(par.names).index(inst)
        else:
            raise KeyError(
                f"Instance '{inst}' not found for parameter '{label}' "
                f"(names={par.names})")
        return par, elem

    def set_value(self, param_path: str, value_in_user_units: float,
                  raw_point) -> Dict[str, np.ndarray]:
        """Return a NEW raw point with ``param_path`` set to a user-unit value.

        Converts user units to internal units and inverts the parameter's
        logit/linear transform for the target element, leaving all other
        elements (and parameters) untouched.

        Raises
        ------
        NeedsResolve
            If the element has tensor-valued (linked/dynamic) bounds, or is
            not a sampled free variable (fixed, derived or hard-linked) -- in
            either case there is no static inverse and the GUI must re-solve.
        ValueError
            If the requested value falls outside the element's hard bounds
            (a clipped start would sit in no posterior basin).
        """
        par, elem = self._resolve_param(param_path)

        links = par.element_links or {}
        if elem in links.get("lower", {}) or elem in links.get("upper", {}):
            raise NeedsResolve(
                f"{param_path}: element {elem} has tensor-valued (linked) "
                f"bounds; cannot invert statically -- re-solve required.")
        if elem in links.get("hard", {}):
            raise NeedsResolve(
                f"{param_path}: element {elem} is hard-linked (deterministic); "
                f"it is not a free slider -- re-solve required.")

        tf = getattr(par, "_raw_transform", None)
        if tf is None or elem not in list(tf["sampled_idx"]):
            raise NeedsResolve(
                f"{param_path}: element {elem} is not a sampled free variable "
                f"(fixed or derived); cannot set via slider -- re-solve required.")

        # user -> internal units
        factors = np.atleast_1d(np.asarray(par._get_conversion_factors(),
                                           dtype=float))
        factor = factors[elem] if elem < factors.size else factors[0]
        val_internal = float(value_in_user_units) / factor

        raw_key = f"{par.label}_raw"
        if raw_key not in raw_point:
            raise NeedsResolve(
                f"{param_path}: no raw variable '{raw_key}' in the point "
                f"(parameter not sampled in this model) -- re-solve required.")

        cur_raw = np.asarray(raw_point[raw_key], dtype=float)
        phys = par.phys_from_raw(cur_raw)      # full element vector (internal)
        phys = np.asarray(phys, dtype=float).copy()
        phys[elem] = val_internal
        try:
            new_raw = par.raw_from_initval(phys)
        except SeedBoundViolation as e:
            raise ValueError(
                f"{param_path}: value {value_in_user_units} is outside the "
                f"parameter's hard bounds ({e}).") from e

        new_point = {k: np.array(v, copy=True) for k, v in raw_point.items()}
        new_point[raw_key] = np.asarray(new_raw, dtype=float).reshape(
            np.shape(raw_point[raw_key]))
        return new_point

    # convenience: expose the module-level hash on the instance too
    @staticmethod
    def structural_hash(config: dict, user_params: Optional[dict] = None) -> str:
        return structural_hash(config, user_params)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def _is_extra_input(var, free_set) -> bool:
    """A graph root that must be supplied a value: not a free variable, not a
    constant, not a shared variable (RNGs etc.), not owned by another node."""
    if var.owner is not None:
        return False
    if isinstance(var, (Constant, SharedVariable)):
        return False
    if var in free_set:
        return False
    return True


def _node_extra_inputs(node, free_set) -> list:
    """The extra input variables of a node (in graph order, de-duplicated)."""
    out, seen = [], set()
    for anc in ancestors([node]):
        if _is_extra_input(anc, free_set) and id(anc) not in seen:
            seen.add(id(anc))
            out.append(anc)
    return out


class _CallRecorder:
    """Wraps a compiled pytensor Function to record, per call, the concrete
    value passed for each of its input variables.  Used to capture the
    grid/index arguments the base plot_data() call fed the retained nodes,
    without reimplementing any component-specific grid logic."""

    def __init__(self):
        # list of {input_variable: value} observations, one per recorded call
        self.observations: List[Dict[Any, Any]] = []
        self._patched: List[Tuple[Any, str, Function]] = []

    def wrap(self, owner, attr, fn: Function):
        recorder = self
        maker_vars = [inp.variable for inp in fn.maker.inputs]

        def wrapper(*args, **kwargs):
            if not kwargs:
                obs = {}
                for var, val in zip(maker_vars, args):
                    obs[var] = np.asarray(val) if np.ndim(val) else val
                recorder.observations.append(obs)
            return fn(*args, **kwargs)

        setattr(owner, attr, wrapper)
        self._patched.append((owner, attr, fn))

    def restore(self):
        for owner, attr, fn in self._patched:
            setattr(owner, attr, fn)
        self._patched = []


def _match_observation(extra_inputs, base_x, base_y, observations):
    """Choose the givens for a trace's extra inputs from recorded calls.

    Prefers a call whose vector input exactly equals the trace's x-array
    (the unphased case, where the plotted x IS the time grid), then a call
    whose vector input length matches the trace, then the first candidate.
    Returns a list of (variable, value) pairs, or None if no candidate
    supplies every extra input.
    """
    if not extra_inputs:
        return []
    want = set(id(v) for v in extra_inputs)
    candidates = [obs for obs in observations
                  if want <= set(id(k) for k in obs)]
    if not candidates:
        return None

    x = np.asarray(base_x, dtype=float) if base_x is not None else None
    y_len = len(np.atleast_1d(base_y)) if base_y is not None else None

    def score(obs):
        s = 0
        for v in extra_inputs:
            val = obs[v]
            arr = np.atleast_1d(np.asarray(val))
            if x is not None and arr.shape == x.shape and np.allclose(arr, x):
                s += 100                      # exact grid match (unphased)
            elif y_len is not None and arr.ndim == 1 and arr.size == y_len:
                s += 10                       # length match
        return s

    best = max(candidates, key=score)
    # Cast each frozen value to its input variable's dtype (e.g. an int32
    # instrument index recorded from a Python int / int64 array).
    out = []
    for v in extra_inputs:
        val = np.asarray(best[v])
        dtype = getattr(v, "dtype", None)
        if dtype is not None:
            val = val.astype(dtype)
        out.append((v, val))
    return out


def compile_evaluator(system, model, base_raw_point=None) -> Evaluator:
    """Build an :class:`Evaluator` for a prepared+built system.

    Captures the base PlotSpecs (via each component's ``plot_data`` at the
    base point) and, for every model-role trace carrying a symbolic node,
    records the concrete extra-input (grid/index) values it used and an
    affine map from node output to the plotted display units.  The pytensor
    functions themselves are compiled lazily, per plot, on first eval.

    Parameters
    ----------
    system, model
        A system whose ``build_model()`` has run (so ``compile_plotters``
        retained the symbolic nodes and ``system.plot_params`` exists).
    base_raw_point
        Optional raw-space start; defaults to ``system.get_raw_start(model)``.
    """
    if base_raw_point is None:
        base_raw_point = system.get_raw_start(model)

    free_rvs = list(model.free_RVs)
    free_set = set(free_rvs)

    # Internal (physical) point that plot_data expects.
    base_internal = system.get_internal_point(model, base_raw_point)

    # Record the grid/index arguments the base plot_data() feeds the nodes.
    recorder = _CallRecorder()
    for comp in system.active_components.values():
        for attr, val in list(vars(comp).items()):
            if isinstance(val, Function):
                recorder.wrap(comp, attr, val)

    specs = []
    try:
        for comp in system.active_components.values():
            try:
                comp_specs = comp.plot_data(system, base_internal)
            except Exception as e:  # noqa: BLE001 - a component may lack data
                logger.warning("Evaluator: plot_data failed for %s: %s",
                               getattr(comp, "prefix", comp), e)
                continue
            specs.extend(comp_specs)
    finally:
        recorder.restore()

    # Build one _ModelTrace per model-role trace that carries a node.
    model_traces: List[_ModelTrace] = []
    for spec in specs:
        for tr in spec.traces:
            if tr.role != "model" or tr.node is None:
                continue
            mt = _ModelTrace(spec.id, tr.name, tr.node, tr.x, tr.y)
            extra = _node_extra_inputs(tr.node, free_set)
            givens = _match_observation(extra, tr.x, tr.y,
                                        recorder.observations)
            if givens is None:
                logger.warning(
                    "Evaluator: could not resolve extra inputs for trace "
                    "'%s' of plot '%s'; skipping.", tr.name, spec.id)
                continue
            mt.givens = givens
            model_traces.append(mt)

    ev = Evaluator(system, model, specs, base_raw_point, model_traces, free_rvs)

    # Calibrate each trace's display transform against the base plotted y.
    _calibrate(ev, base_raw_point)
    return ev


def _calibrate(ev: Evaluator, base_raw_point) -> None:
    """Fit the per-trace affine map (node output -> display y) at the base
    point.  A true affine relation (RV: pure scale; transit: baseline
    offset) is recovered exactly; when the node output shape does not match
    the plotted y (e.g. SED spectra, phased matrices) the node output is
    returned as-is (``affine`` stays None)."""
    inputs = ev._inputs_for(base_raw_point)
    for plot_id in list(ev._traces_by_plot):
        fn, traces = ev._compile_plot(plot_id)
        results = fn(*inputs)
        if not isinstance(results, (list, tuple)):
            results = [results]
        for mt, node_out in zip(traces, results):
            node_arr = np.asarray(node_out, dtype=float).ravel()
            y = np.asarray(mt.base_y, dtype=float).ravel()
            if (node_arr.shape == y.shape and node_arr.size >= 2
                    and np.all(np.isfinite(node_arr)) and np.all(np.isfinite(y))
                    and np.nanstd(node_arr) > 0):
                a, b = np.polyfit(node_arr, y, 1)
                recon = a * node_arr + b
                if np.max(np.abs(recon - y)) <= 1e-6 * (np.max(np.abs(y)) + 1.0):
                    mt.affine = (float(a), float(b))
                    mt.raw_ok = True
                    continue
            # Fall back to raw node output (documented deviation for SED /
            # phased traces where display y is not affine in the node).
            mt.affine = None
            mt.raw_ok = False
