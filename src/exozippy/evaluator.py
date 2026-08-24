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
    PlotSpecs (G4) and remembers which component built each one.
  * ``Evaluator.eval_plots(raw_point)`` returns, per plot, the updated
    model-trace arrays by calling each owning component's own
    ``plot_data(system, point)`` again at the new point -- the SAME code
    that built the base specs and that the CLI's matplotlib ``plot()``
    reuses (see "Plotting: one description, two renderers" in
    ``src/exozippy/outputs/outputs.md``) -- rather than
    a second, parallel implementation. There is exactly one place that
    knows how to draw a phased light curve or an SED spectrum; a slider
    move can never drift from what a re-Solve (or the saved PDF) would
    show. The only optimization is a single cached raw-point ->
    internal-point pytensor function, built once, which replaces
    ``System.get_internal_point``'s per-call recompile; from there,
    plot_data's own array prep is plain NumPy over already-compiled
    pytensor outputs (compile_plotters), so it is cheap to call again.
    An optional ``changed_label`` skips components whose base specs
    declare no dependency on the moved parameter.
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

Model-role traces are always evaluated.  Data traces normally are not --
raw observations do not move with a slider -- EXCEPT on specs whose meta
declares ``dynamic_data``, where the plotted data ARE point-dependent
(phase folds, RV gamma subtraction, mulens flux alignment) and are
re-shipped too.  See exozippy.plotspec for the PlotSpec/Trace contract and
Component.plot_data for how a component draws itself at an arbitrary point.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pytensor

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
# "modeling" is output-only (the generated paper-draft scaffold): adding
# the block or flipping its `compile` key must not stale a finished trace.
_NON_STRUCTURAL_CONFIG_KEYS = {"run", "modeling"}


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


#: Per-instance config keys that do NOT affect the model, and so are the only
#: things dropped from the structural hash.
#:
#: This is a DENYLIST, and the direction matters.  Nearly every key inside a
#: component instance selects part of the model -- `light_travel_time`, `gp`,
#: `likelihood`, `mask`, `ld_law`, `data_format`, `rm`, `exptime`, `ninterp`,
#: `band`, `star_ndx`, the SED's `mag`/`err` photometry, all of it -- so an
#: allowlist is a list of everything, maintained by hand, and one forgotten
#: entry is a SILENT failure: the trace reloads under a model it was never
#: sampled from (via `recompute_trace: false`, `exozippy-modes` or
#: `mkparam.write_param_file`).  `light_travel_time` was exactly that, and it
#: defaulted to ON.  A denylist fails the other way: forget an entry here and
#: a cosmetic edit merely forces an honest re-run.
#:
#: Adopting it invalidated every trace on disk once, deliberately -- the
#: payload shape changed for every config.  That is a one-time cost paid for
#: never silently reusing foreign draws again.
#:
#: `file`/`files` are dropped only because `structural_payload` already hashes
#: them under its own "files" key; `path` is NOT (nothing else captures it).
_NON_STRUCTURAL_INSTANCE_KEYS = frozenset(
    {
        "plot",  # per-instrument plot styling (color/marker)
        "label",  # display-only
        "file",  # already hashed under payload["files"]
        "files",
    }
)


def _canon_leaf(value: Any) -> Any:
    """Canonicalize any config leaf -- scalar, list or nested dict -- into
    something json.dumps(sort_keys=True) renders stably."""
    if isinstance(value, dict):
        return {str(k): _canon_leaf(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_canon_leaf(v) for v in value]
    return _canon(value)


def _instance_skeleton(item: dict, index: int) -> Dict[str, Any]:
    """Structural view of one instance dict: its name plus every key that is
    not explicitly cosmetic (see _NON_STRUCTURAL_INSTANCE_KEYS)."""
    return {
        "name": str(item.get("name", index)),
        "cfg": {
            str(k): _canon_leaf(v)
            for k, v in sorted(item.items())
            if k not in _NON_STRUCTURAL_INSTANCE_KEYS and k != "name"
        },
    }


def _component_skeleton(config: dict) -> dict:
    """Structural view of the component set: top-level keys and, for
    list-valued component blocks, the sorted instances with their
    model-affecting config.  Only explicitly cosmetic keys are dropped."""
    skel: Dict[str, Any] = {}
    for key, block in config.items():
        if key in _NON_STRUCTURAL_CONFIG_KEYS:
            continue
        if isinstance(block, list):
            entries = sorted(
                (
                    _instance_skeleton(item, i)
                    for i, item in enumerate(block)
                    if isinstance(item, dict)
                ),
                key=lambda e: e["name"],
            )
            skel[key] = entries if entries else len(block)
        elif isinstance(block, dict):
            # single-instance component (e.g. sed): presence is structural,
            # and so is everything non-cosmetic it sets.
            skel[key] = _instance_skeleton(block, 0)["cfg"] or True
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
        # init_scale is not listed: it is no longer linkable (user entries
        # are warn-ignored; whitening scales are measured from the data).
        for field in (
            "initval",
            "mu",
            "lower",
            "upper",
            "sigma",
        ):
            if isinstance(spec.get(field), str):
                entry[f"{field}_link"] = spec[field]
        if entry:
            struct[str(path)] = entry
    return struct


def structural_payload(
    config: dict, user_params: Optional[dict] = None
) -> Dict[str, Any]:
    """The structural view :func:`structural_hash` hashes, as a plain dict.

    Split out so a consumer that detects a hash mismatch can say WHAT
    changed instead of only THAT something did (see
    ``exozippy.trace_meta``).  The hash is sha256 of this dict's
    sort_keys JSON, so the two can never describe different things.
    """
    return {
        "components": _component_skeleton(config or {}),
        "files": sorted(_gather_files(config or {})),
        "params": _param_structure(user_params),
    }


def structural_hash(config: dict, user_params: Optional[dict] = None) -> str:
    """Stable structural fingerprint of a system configuration.

    The hash covers the component set, per-parameter bounds/fixed-ness and
    expression wiring, and the data file list.  It is insensitive to dict
    key order (canonicalized via ``sort_keys``) and to pure initval changes
    (numeric initval/mu values are not included).  Two configs with the same
    hash may share the same compiled evaluator; a changed hash means the GUI
    must re-solve.
    """
    blob = json.dumps(
        structural_payload(config, user_params), sort_keys=True, default=str
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class Evaluator:
    """Millisecond-scale forward evaluator for GUI slider updates.

    Construct via :func:`compile_evaluator`.  Holds the base PlotSpecs, which
    component built each one, and a single cached pytensor function mapping
    the model's raw free variables to the full internal point (free RVs +
    deterministics).  A slider move calls :meth:`set_value` to invert a
    user-unit value into a new raw point, then :meth:`eval_plots` to re-render
    the model curves at that point -- by calling each affected component's own
    ``plot_data(system, point)`` again, not a separate compiled graph.
    """

    def __init__(
        self, system, model, specs, spec_owner, base_raw_point, free_rvs
    ):
        self.system = system
        self.model = model
        self.specs = specs
        self.base_raw_point = base_raw_point
        self._free_rvs = free_rvs
        self._free_names = [v.name for v in free_rvs]
        self._param_lookup = system.get_parameter_lookup()

        # One-time compile of a fast raw -> internal-point function, reused on
        # every slider move so live eval never pays System.get_internal_point's
        # per-call pytensor.function() compile cost.
        det_outputs = list(model.deterministics)
        internal_outputs = list(free_rvs) + det_outputs
        self._internal_names = [v.name for v in internal_outputs]
        self._internal_fn = pytensor.function(
            inputs=list(free_rvs),
            outputs=internal_outputs,
            on_unused_input="ignore",
        )

        # Which component built each base PlotSpec, and the sampled-parameter
        # labels its model traces depend on (PlotSpec.param_deps) -- so a
        # slider move only calls plot_data() on components it can affect.
        self._spec_owner = spec_owner
        self._comps_by_id: Dict[int, Any] = {}
        self._comp_deps: Dict[int, set] = {}
        for spec in specs:
            comp = spec_owner.get(spec.id)
            if comp is None:
                continue
            cid = id(comp)
            self._comps_by_id[cid] = comp
            self._comp_deps.setdefault(cid, set()).update(spec.param_deps)

    # -- public API -------------------------------------------------------

    def _inputs_for(self, raw_point) -> List[np.ndarray]:
        """Positional input values in model.free_RVs order."""
        return [np.asarray(raw_point[name]) for name in self._free_names]

    def internal_point(self, raw_point) -> Dict[str, np.ndarray]:
        """Fast raw -> internal-point map (free RVs + deterministics).

        Equivalent to ``system.get_internal_point(model, raw_point)`` but
        reuses a function compiled once at construction, instead of
        recompiling on every call -- the difference between millisecond and
        multi-second per slider move.
        """
        values = self._internal_fn(*self._inputs_for(raw_point))
        return dict(zip(self._internal_names, values))

    def eval_plots(
        self, raw_point, changed_label: Optional[str] = None
    ) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
        """Re-render every plot's model traces at ``raw_point``.

        Returns ``{plot_id: {trace_name: {"x": array, "y": array}}}``.  Data
        traces are omitted -- raw observations do not move with a slider --
        unless the spec's meta declares ``dynamic_data``, in which case its
        data traces (plus ``yerr`` where present) are re-shipped because the
        plotted data themselves depend on the point (see the loop below).
        Each component's own ``plot_data(system, point)`` is called fresh, so the
        result is always an exact recompute -- not an approximation -- even
        for phase-folded curves (re-sorted/re-selected from a multi-column
        node) and SED spectra (NumPy spectral-library interpolation), both of
        which defeated the previous affine-calibrated pytensor fast path.

        ``changed_label`` (a ``system.plot_params`` label, e.g.
        ``"star.radiussed"``), if given, skips components whose base specs
        declare no dependency on it -- the moved parameter cannot affect
        their traces, so there's nothing to recompute. Omit it to force a
        full recompute of every plot.
        """
        internal_point = self.internal_point(raw_point)
        out: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
        for cid, comp in self._comps_by_id.items():
            deps = self._comp_deps.get(cid, set())
            if changed_label is not None and changed_label not in deps:
                continue
            try:
                comp_specs = comp.plot_data(self.system, internal_point)
            except Exception as exc:  # noqa: BLE001 - keep sliders usable
                logger.warning(
                    "Evaluator: plot_data failed for %s during live eval: %s",
                    getattr(comp, "prefix", comp),
                    exc,
                )
                continue
            for spec in comp_specs:
                # Model traces always move with the point. Data traces
                # normally never do (raw observations), EXCEPT on specs whose
                # meta declares dynamic_data: phase-folded panels re-fold the
                # observations with tc/P, RV panels subtract the per-
                # instrument gamma, and mulens panels re-align every data set
                # onto the reference flux system -- all point-dependent.
                # Those specs' data traces (x, y, and yerr, which the mulens
                # alignment also changes) are re-shipped so the GUI can patch
                # them; otherwise a tc slider moves the model but leaves the
                # fold visibly frozen.
                dynamic = bool((spec.meta or {}).get("dynamic_data"))
                traces = {}
                for tr in spec.traces:
                    if tr.role == "model":
                        traces[tr.name] = {"x": tr.x, "y": tr.y}
                    elif dynamic and tr.role == "data":
                        entry = {"x": tr.x, "y": tr.y}
                        if tr.yerr is not None:
                            entry["yerr"] = tr.yerr
                        traces[tr.name] = entry
                if traces:
                    out[spec.id] = traces
        return out

    def label_for_path(self, path: str) -> str:
        """The ``system.plot_params`` label (e.g. ``"star.radiussed"``) a GUI
        path (``comp.instance.param``) resolves to -- what ``eval_plots``'s
        ``changed_label`` filter expects."""
        par, _ = self._resolve_param(path)
        return par.label

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
                f"(names={par.names})"
            )
        return par, elem

    def set_value(
        self, param_path: str, value_in_user_units: float, raw_point
    ) -> Dict[str, np.ndarray]:
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
                f"bounds; cannot invert statically -- re-solve required."
            )
        if elem in links.get("hard", {}):
            raise NeedsResolve(
                f"{param_path}: element {elem} is hard-linked (deterministic); "
                f"it is not a free slider -- re-solve required."
            )

        tf = getattr(par, "_raw_transform", None)
        if tf is None or elem not in list(tf["sampled_idx"]):
            raise NeedsResolve(
                f"{param_path}: element {elem} is not a sampled free variable "
                f"(fixed or derived); cannot set via slider -- re-solve required."
            )

        # user -> internal units
        val_internal = float(
            par.to_internal(float(value_in_user_units), index=elem)
        )

        raw_key = f"{par.label}_raw"
        if raw_key not in raw_point:
            raise NeedsResolve(
                f"{param_path}: no raw variable '{raw_key}' in the point "
                f"(parameter not sampled in this model) -- re-solve required."
            )

        cur_raw = np.asarray(raw_point[raw_key], dtype=float)
        phys = par.phys_from_raw(cur_raw)  # full element vector (internal)
        phys = np.asarray(phys, dtype=float).copy()
        phys[elem] = val_internal
        try:
            new_raw = par.raw_from_initval(phys)
        except SeedBoundViolation as e:
            raise ValueError(
                f"{param_path}: value {value_in_user_units} is outside the "
                f"parameter's hard bounds ({e})."
            ) from e

        new_point = {k: np.array(v, copy=True) for k, v in raw_point.items()}
        new_point[raw_key] = np.asarray(new_raw, dtype=float).reshape(
            np.shape(raw_point[raw_key])
        )
        return new_point

    # convenience: expose the module-level hash on the instance too
    @staticmethod
    def structural_hash(
        config: dict, user_params: Optional[dict] = None
    ) -> str:
        return structural_hash(config, user_params)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def compile_evaluator(system, model, base_raw_point=None) -> Evaluator:
    """Build an :class:`Evaluator` for a prepared+built system.

    Captures the base PlotSpecs (via each component's ``plot_data`` at the
    base point) and remembers which component built each one, so a later
    slider move can call the right component's ``plot_data`` again.

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

    # Internal (physical) point that plot_data expects. This one-time call
    # may recompile (System.get_internal_point does not cache); the Evaluator
    # itself compiles and caches its own fast version for every later eval.
    base_internal = system.get_internal_point(model, base_raw_point)

    specs = []
    spec_owner: Dict[str, Any] = {}
    for comp in system.active_components.values():
        try:
            comp_specs = comp.plot_data(system, base_internal)
        except Exception as e:  # noqa: BLE001 - a component may lack data
            logger.warning(
                "Evaluator: plot_data failed for %s: %s",
                getattr(comp, "prefix", comp),
                e,
            )
            continue
        for spec in comp_specs:
            spec_owner[spec.id] = comp
        specs.extend(comp_specs)

    return Evaluator(
        system, model, specs, spec_owner, base_raw_point, free_rvs
    )
