"""The relaxation engine: which value absorbs a discrepancy (review 3.14.22).

ONE RULE, in three steps (design "KS", JDE ruling 2026-10-14).  The
reasoning, the measurements and what it replaced are in
``src/exozippy/config.md``, "Precedence, and which value absorbs a
discrepancy".

1. WHICH TIER MOVES is decided by provenance, strictly in order.  Only SEEDS
   carry a precedence -- a user value, a component hint, a defaults.yaml
   value on a SAMPLED element.  A symbol nobody seeded (theta_E, mu_rel, a
   derived mass) carries none, so it can never win or lose a tie.  Tiers are
   fixed from the top of the ladder down.  A seed whose symbol the higher
   tiers already determine is ABSORBED (re-derived from them); a tier's own
   seeds move only when, with every lower tier still free, the relations
   cannot otherwise be satisfied.
2. USER VALUES ON DERIVED ELEMENTS are kept in order: shallowest derivation
   first, ties broken by params-file order.  Each is honored exactly if the
   relations can still be satisfied with it and every value kept before it,
   otherwise it is DROPPED and reported by name.  Relations the RIGID user
   values (values on sampled elements) already break among themselves are
   reported as contradictions and take no part in the solve.
3. HOW MUCH MOVES, inside the one tier that has to absorb, is the
   minimum-norm displacement in init_scale units.  A scale only ever splits
   a correction between seeds of the SAME tier.

It is implemented as forward PROPAGATION from a growing set of fixed values:
a relation with exactly one undetermined symbol is solved for it (the old
engine's "Condition A", with its custom solvers, cached sp.solve, nsolve
fallback, bounds test and multi-root scoring).  A relation whose symbols are
all determined and that determined none of them is a CHECK.  A violated
check at a tier is settled by a Gauss-Newton minimum-norm step over that
tier's new seeds, the Jacobian taken by finite differences through the
recorded propagation PLAN.  There is no global numerical solve.

Nothing here knows about any component.  ConfigManager builds the inputs
(anchors, scales, roles, the ladder) and reads the outputs.
"""

import dataclasses
import logging
import math
import time

import numpy as np
import sympy as sp

logger = logging.getLogger("exozippy.config")

# How far a value must move, IN UNITS OF ITS init_scale, to count as a
# change -- and so how far a relation may be off before it is VIOLATED
# (review 1.1.9, JDE 2026-10-06; why 1e-5 is in config.md, "When a
# re-solved value counts as a change").  config.py re-exports it.
CHANGE_TOL_IN_SCALES = 1e-5

# A settle drives every check to this fraction of the change tolerance, so
# the satisfaction test that follows (which recomputes the residual's scale
# at the settled point) cannot flicker.
_SETTLE_TARGET = 0.1 * CHANGE_TOL_IN_SCALES
_SETTLE_MAX_ITER = 60
# Rounds of the loop that lets a standalone (one-way) solver's answer
# replace a low-precedence start of its target; see Engine.solve.
_SOLVER_ROUNDS = 10

# The order user values on DERIVED elements are tried in (step 2): "depth"
# -- shallowest derivation from the sampled leaves first, then params-file
# order -- is JDE's ruling (R-a, 2026-10-14).  "file" (params-file order
# alone) and "reverse_depth" exist only so the ordering sweep can measure
# the alternatives; nothing in the package sets them.
KEEP_ORDER = "depth"


def scale_error(path, why):
    """The one error for a symbol the engine needs a scale for and has none."""
    comp = path.split(".")[0]
    param = path.split(".")[-1]
    return ValueError(
        f"The relaxation engine needs an init_scale for '{path}' {why}, and "
        f"there is none: no init_scale in {comp}/defaults.yaml, no component "
        f"add_scale_hint, no user sigma, and no symbolic path that "
        f"propagates one from the sampled parameters.  Declare an init_scale "
        f"for '{param}' in {comp}/defaults.yaml (review 3.14.22: no constant "
        f"backstop is ever substituted)."
    )


def wrap(diff, period):
    """`diff` folded into (-period/2, period/2] (no-op without a period)."""
    if period:
        return (diff + 0.5 * period) % period - 0.5 * period
    return diff


def value_moved(new_val, old_val, scale, path, period=None):
    """Has `path` moved from `old_val` to `new_val` by a meaningful amount?

    Measured in the symbol's init_scale (`scale`, INTERNAL units):
    ``|new - old| >= CHANGE_TOL_IN_SCALES * scale``.  A symbol with no
    init_scale RAISES (review 3.14.22): the relative fallback this kept for
    scale-less symbols is gone, because every engine symbol now has a
    declared or a propagated scale, and one that has neither is a
    defaults.yaml gap to fill, not a case to guess at.  `period` (internal
    units) wraps the difference for an angle.
    """
    if scale is None or not np.isfinite(scale) or scale <= 0:
        raise scale_error(path, "to judge whether its value changed")
    return abs(wrap(new_val - old_val, period)) >= CHANGE_TOL_IN_SCALES * scale


# ---------------------------------------------------------------------------
# Inputs and the propagation plan
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class Anchor:
    """One statement about a symbol's start, before any relation is used.

    ``kind``:
      * "rigid"     -- a user value on a SAMPLED element: produced exactly;
      * "candidate" -- a user value on a DERIVED (or inactive, or
                       non-parameter) element: kept in order, or dropped;
      * "seed"      -- a component hint on any element, or a defaults.yaml
                       value on a sampled element: held unless absorbed;
      * "fill"      -- a defaults.yaml value on a derived/inactive element:
                       never a seed (it cannot reach the model); only gives
                       a value to a symbol nothing else determines.
    ``tier`` DECIDES; ``rank`` is what the ledger REPORTS.  They differ only
    where a component declared the parameter likelihood-backed, which moves
    the decision without making a defaults.yaml value report as data.
    ``sampled`` marks a leaf for the propagated-scale and depth pass.
    """

    value: float
    kind: str
    tier: int
    rank: int
    sampled: bool
    source: str = ""


@dataclasses.dataclass
class Step:
    """One determination in a propagation plan, re-evaluable from values."""

    kind: str  # rel | custom | standalone | link | nsolve
    target: str
    inputs: tuple
    label: str
    fn: object  # callable(values) -> float
    root: tuple = None  # (relation position, root index) for kind "rel"
    # Several roots fit equally well (a sign from a squared relation with
    # nothing yet to choose by): the closure takes such a step only when
    # nothing unambiguous is left to do (see closure).
    ambiguous: bool = False


@dataclasses.dataclass
class Check:
    """A fully determined relation (or link) that determined nothing."""

    kind: str  # rel | link
    label: str
    symbols: tuple
    residual: object  # callable(values) -> float, wrapped if periodic
    partials: dict  # {symbol: callable(values) -> d residual / d symbol}
    key: object  # relation position, or ("link", target)


@dataclasses.dataclass
class Plan:
    values: dict
    steps: list
    checks: list
    determined_by: dict  # path -> Step
    scales: dict = dataclasses.field(default_factory=dict)  # plan-propagated
    # _plan_scale's cycle bookkeeping: symbols being computed, and whether
    # the computation under way met one of them
    scale_busy: set = dataclasses.field(default_factory=set)
    scale_cut: bool = False


def _as_real(x):
    """A real float from a lambdified result, or nan."""
    try:
        z = complex(x)
    except (TypeError, ValueError):
        return math.nan
    if math.isnan(z.real) or not math.isfinite(z.imag):
        return math.nan
    if math.isinf(z.real):
        return z.real  # a real infinity (x / 0) is not "undefined"
    if abs(z.imag) > 1e-10 * max(abs(z.real), 1e-300):
        return math.nan
    return float(z.real)


def _call(fn, args):
    """Evaluate a lambdified function at real args; retry in complex
    arithmetic when the real evaluation leaves the real line (the real
    cube root reached through a complex intermediate)."""
    with np.errstate(all="ignore"):
        try:
            # numpy scalars, not Python floats: x / 0 is then a real inf
            # (under the errstate above), not a ZeroDivisionError
            val = _as_real(fn(*[np.float64(a) for a in args]))
        except (TypeError, ValueError, ZeroDivisionError, OverflowError):
            val = math.nan
        if not math.isnan(val):  # finite, or a real +-inf (x / 0)
            return val
        try:
            return _as_real(fn(*[complex(a) for a in args]))
        except (TypeError, ValueError, ZeroDivisionError, OverflowError):
            return math.nan


class _Recorder(dict):
    """The values dict, recording which keys a SOLVER reads.

    A registered solver is a Python function of the whole values dict, so
    its inputs are not visible to the engine the way a relation's symbols
    are.  Reading them off one call makes a solver step a first-class plan
    step: its scale propagates (the orbit mass sum's), and a report can name
    what is upstream of it."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.read = set()

    def __getitem__(self, key):
        self.read.add(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        self.read.add(key)
        return super().get(key, default)

    def __contains__(self, key):
        self.read.add(key)
        return super().__contains__(key)


def _atan2(y, x):
    """sympy's atan2, not numpy's: UNDEFINED at the origin.  numpy answers
    0.0 there, which would hand a circular orbit (secosw = sesinw = 0) an
    omega of 0 instead of leaving it to the circular-orbit solver's pi/2
    convention -- the one the runtime uses."""
    if y == 0 and x == 0:
        return math.nan
    return np.arctan2(y, x)


class _Lambda:
    """A sympy expression compiled once, evaluated from a values dict."""

    __slots__ = ("names", "fn")

    def __init__(self, expr, symbols):
        self.names = tuple(str(s) for s in symbols)
        self.fn = sp.lambdify(
            list(symbols), expr, modules=[{"atan2": _atan2}, "numpy"]
        )

    def __call__(self, values):
        return _call(self.fn, [values[n] for n in self.names])


class _FDPartial:
    """d g / d `name` by a central difference of the compiled residual `g`.

    For a relation whose partial sympy cannot print: one built on an
    implemented function (a table lookup such as the Claret limb-darkening
    tie), whose derivative is an unevaluated ``Derivative`` that lambdify
    refuses.  The step is a millionth of the value (or of 1)."""

    __slots__ = ("names", "g", "name")

    def __init__(self, g, name):
        self.names = g.names
        self.g = g
        self.name = name

    def __call__(self, values):
        x = float(values[self.name])
        h = 1e-6 * max(1.0, abs(x))
        up, dn = dict(values), dict(values)
        up[self.name] = x + h
        dn[self.name] = x - h
        return (self.g(up) - self.g(dn)) / (2.0 * h)


def _compiled(cm, key, build):
    """A pure function of (expression, target) cached on the ConfigManager.

    Like ``_symbolic_solve_cache`` it sees no values, so it is safe to keep
    across seeds and probes (and is deliberately not rolled back by one)."""
    cache = cm.__dict__.setdefault("_ks_compiled_cache", {})
    if key not in cache:
        cache[key] = build()
    return cache[key]


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------
class Engine:
    """One relaxation solve: construct, :meth:`solve`, read the results."""

    def __init__(
        self,
        cm,
        anchors,
        declared_scales,
        file_order,
        directed_links,
        periodic,
        *,
        ladder,
        keep_order_mode="depth",
        constants=None,
    ):
        self.cm = cm
        self.anchors = anchors
        self.declared_scales = declared_scales
        self.file_order = file_order
        self.periodic = periodic
        self.constants = dict(constants or {})
        self.ladder = dict(ladder)
        self.keep_order_mode = keep_order_mode

        # Relations whose every symbol is an engine symbol or a supplied
        # constant (the old gatekeeper), constants substituted in.
        self.relations = []  # (eq, sorted symbol names)
        known = set(cm.master_symbol_map) | set(self.constants)
        csubs = {sp.Symbol(p): v for p, v in self.constants.items()}
        for eq in cm.all_relations:
            if not isinstance(eq, sp.Eq):
                continue
            names = {str(s) for s in eq.free_symbols}
            if not names or not names <= known:
                continue
            if names & set(self.constants):
                eq = eq.xreplace(csubs)
                names = {str(s) for s in eq.free_symbols}
                if not names:
                    continue
            self.relations.append((eq, tuple(sorted(names))))
        self.by_symbol = {}
        for pos, (_eq, names) in enumerate(self.relations):
            for n in names:
                self.by_symbol.setdefault(n, []).append(pos)

        # Directed user links (initval/mu): target := expr(deps), USER.
        self.links = []
        for target, fld, plink, expr in directed_links:
            a = anchors.get(target)
            if (
                fld == "mu"
                and a is not None
                and a.kind in ("rigid", "candidate")
            ):
                # A mu link only seeds the start; a numeric user initval on
                # the target outranks it.
                continue
            syms = sorted(expr.free_symbols, key=str)
            self.links.append(
                (
                    target,
                    plink.expr_str,
                    _Lambda(expr, syms),
                    expr,
                    tuple(str(s) for s in syms),
                )
            )
        self.link_targets = {t for t, *_ in self.links}

        # Standalone (one-way) solver instances: (lookup key, path, index),
        # one per INSTANCE the config declares.  A path can be an engine
        # symbol with no instance behind it -- planet relations map
        # orbit.<i>.ecc/sini/period for a planet that names no orbit block
        # (a microlensing planet) -- and a solver has nothing to read there.
        self.standalone = []
        for key in sorted(cm.standalone_solvers):
            comp, param = key.split(".")[0], key.split(".")[-1]
            block = (cm.system_config or {}).get(comp)
            if block is None:
                continue
            n_inst = len(block) if isinstance(block, list) else 1
            for path in cm.master_symbol_map:
                parts = path.split(".")
                if (
                    len(parts) == 3
                    and parts[0] == comp
                    and parts[2] == param
                    and parts[1].isdigit()
                    and int(parts[1]) < n_inst
                ):
                    self.standalone.append((key, path, int(parts[1])))
        self.standalone.sort(key=lambda t: (t[0], t[2]))

        self._bounds_cache = {}
        self.leaf_scales = {}
        self.depth = {}
        self.diagnostics = []
        self.keep_report = {}

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _bounds(self, path):
        """(lower, upper) of `path` in INTERNAL units, from resolve()."""
        if path in self._bounds_cache:
            return self._bounds_cache[path]
        parts = path.split(".")
        el = int(parts[1]) if len(parts) == 3 and parts[1].isdigit() else None
        cfg = self.cm.resolve(parts[0], parts[-1], shape=(), element=el)
        factor = self.cm.get_conversion_factor(
            parts[0], parts[-1], full_path=path
        )

        def first(key, default):
            v = cfg.get(key)
            if v is None:
                return default
            v = float(np.atleast_1d(np.asarray(v, dtype=float))[0])
            return v * factor if np.isfinite(v) else default

        lo, hi = first("lower", -math.inf), first("upper", math.inf)
        if lo > hi:
            lo, hi = hi, lo
        self._bounds_cache[path] = (lo, hi)
        return lo, hi

    def scale(self, path, plan=None):
        """The split and change-test scale of `path`.

        DECLARED (defaults.yaml, a component scale hint, a user sigma), else
        PROPAGATED from the sampled leaves through the relations at their
        anchors (the leaf pass), else -- when a `plan` is given -- propagated
        the same way along the plan that determined it, at the plan's
        values.  The last covers a symbol the leaves cannot reach at their
        own anchors: a stage-1 probe has every proper motion at its default,
        so mu_rel = 0 there and t_E has no finite value to propagate.  Where
        the plan determined `path` from inputs that have no scale (user pins
        on derived symbols), the scale a relation to scaled symbols implies
        (_implied_scale).  None when there is no source at all."""
        s = self.declared_scales.get(path)
        if s is not None:
            return s
        s = self.leaf_scales.get(path)
        if s is not None or plan is None:
            return s
        return self._plan_scale(plan, path)

    def _plan_scale(self, plan, path):
        """Secant propagation (see _leaf_pass) along `plan`, memoized on it."""
        memo = plan.scales
        if path in memo:
            return memo[path]
        # A symbol already being computed (a cycle through the implied
        # scales) answers None for now, and nothing that depended on that
        # provisional None is memoized -- it would poison the symbol for the
        # rest of the plan's life, whichever end of the cycle came first.
        if path in plan.scale_busy:
            plan.scale_cut = True
            return None
        plan.scale_busy.add(path)
        outer_cut, plan.scale_cut = plan.scale_cut, False
        try:
            s = self._forward_plan_scale(plan, path)
            if s is None:
                s = self._implied_scale(plan, path)
        finally:
            plan.scale_busy.discard(path)
        cut = plan.scale_cut
        if s is not None or not cut:
            memo[path] = s
        plan.scale_cut = cut or outer_cut
        return s

    def _implied_scale(self, plan, path):
        """The scale a relation IMPLIES for `path` from its other symbols'
        scales, |dg/dx|^-1 sqrt(sum_j (dg/dy_j s_j)^2), the tightest over
        the relations whose every other symbol has one.  For a symbol the
        plan determined from inputs with no scale of their own: a user's
        pinned t_E and pi_E, when the sampled coordinate the leaf pass
        would propagate them from (mu_rel under fitmurel, log_pi_rel under
        fitpirel) has no engine start because its start arrives at build.
        mu_rel_geo = mu_rel - pi_rel * v_perp then takes its scale from
        mu_rel and pi_rel, as the forward pass would have given it."""
        vals = plan.values
        best = None
        for pos in self.by_symbol.get(path, ()):
            names = self.relations[pos][1]
            if not all(n in vals for n in names):
                continue
            others = [n for n in names if n != path]
            if not others:
                continue
            sc = [self.scale(m, plan) for m in others]
            if any(s is None or not s > 0 for s in sc):
                continue
            _res, parts, _fl, _fr = self._residual(pos)[:4]
            try:
                dp = float(parts[path](vals))
                num = math.sqrt(
                    sum(
                        (float(parts[m](vals)) * s) ** 2
                        for m, s in zip(others, sc)
                    )
                )
            except (ZeroDivisionError, OverflowError, ValueError):
                continue
            if not (math.isfinite(dp) and math.isfinite(num)) or dp == 0.0:
                continue
            s = num / abs(dp)
            if s > 0.0 and (best is None or s < best):
                best = s
        return best

    def _forward_plan_scale(self, plan, path):
        st = plan.determined_by.get(path)
        if st is None or not st.inputs:
            return None
        vals = plan.values
        base = vals[path]
        var = 0.0
        for inp in st.inputs:
            s = self.scale(inp, plan)
            if s is None or not s > 0:
                return None
            delta = math.nan
            for sign in (1.0, -1.0):
                trial = dict(vals)
                trial[inp] = vals[inp] + sign * s
                delta = st.fn(trial) - base
                if math.isfinite(delta):
                    break
            if not math.isfinite(delta):
                return None
            var += wrap(delta, self.periodic.get(path)) ** 2
        return math.sqrt(var) if var > 0.0 else None

    def require_scale(self, path, why, plan=None):
        s = self.scale(path, plan)
        if s is None or not np.isfinite(s) or s <= 0:
            raise scale_error(path, why)
        return s

    def _label(self, pos):
        eq = self.relations[pos][0]
        return f"{eq.lhs} = {eq.rhs}"

    def _residual(self, pos):
        eq, names = self.relations[pos]

        def build():
            syms = [sp.Symbol(n) for n in names]
            g = eq.lhs - eq.rhs
            g_fn = _Lambda(g, syms)
            parts = {}
            for n, s in zip(names, syms):
                d = sp.diff(g, s)
                parts[n] = (
                    _FDPartial(g_fn, n)
                    if d.has(sp.Derivative)
                    else _Lambda(d, syms)
                )
            return (
                g_fn,
                parts,
                _Lambda(eq.lhs, syms),
                _Lambda(eq.rhs, syms),
            )

        res, parts, fl, fr = _compiled(self.cm, ("resid", eq), build)
        period = None
        for side in (eq.lhs, eq.rhs):
            if isinstance(side, sp.Symbol) and str(side) in self.periodic:
                period = self.periodic[str(side)]
                break
        return res, parts, fl, fr, period

    def _residual_scale(self, check, values, plan=None):
        """sqrt(sum_j (d r / d x_j * s_j)^2) over the check's symbols."""
        tot = 0.0
        seen = False
        for n in check.symbols:
            s = self.scale(n, plan)
            if s is None:
                continue
            d = check.partials[n](values)
            if not math.isfinite(d):
                continue
            seen = True
            tot += (d * s) ** 2
        if not seen:
            raise ValueError(
                f"The relaxation engine cannot judge '{check.label}': none of "
                f"its symbols {list(check.symbols)} has an init_scale, "
                f"declared or propagated.  Declare an init_scale on one of "
                f"them in its defaults.yaml (review 3.14.22)."
            )
        return math.sqrt(tot)

    def violation(self, check, values, plan=None):
        """|residual| in units of the residual's own scale.

        A relation that is UNDEFINED at these values -- omega =
        atan2(sesinw, secosw) on a circular orbit -- states nothing there,
        so it is not violated (0.0).  Propagation never produces a value
        outside a relation's domain (an out-of-domain root is rejected when
        it is solved for), so an undefined check is a degenerate point of
        the relation, not a broken start.  An INFINITE residual is not
        undefined: it is a relation that cannot hold (a finite t_E against
        mu_rel = 0), and is violated."""
        r = check.residual(values)
        if math.isnan(r):
            return 0.0
        if math.isinf(r):
            # not undefined but INFINITELY far: t_E = theta_E / mu_rel at
            # mu_rel = 0, against a finite pinned t_E
            return math.inf
        if r == 0.0:
            return 0.0
        sr = self._residual_scale(check, values, plan)
        return abs(r) / sr if sr > 0.0 else math.inf

    def violated(self, plan, excluded=()):
        return [
            c
            for c in plan.checks
            if c.key not in excluded
            and self.violation(c, plan.values, plan) > CHANGE_TOL_IN_SCALES
        ]

    # ------------------------------------------------------------------
    # Condition A: one relation, one undetermined symbol
    # ------------------------------------------------------------------
    def _solutions(self, pos, target):
        eq = self.relations[pos][0]
        cm = self.cm
        target_sym = sp.Symbol(target)
        if target in cm.symbolic_blacklist:
            return []
        cache_key = (eq, target_sym)
        if cache_key not in cm._symbolic_solve_cache:
            from .config import SymbolicTimeout, _sympy_time_limit

            try:
                with _sympy_time_limit():
                    sols = sp.solve(
                        eq, target_sym, dict=False, simplify=False, check=False
                    )
            except SymbolicTimeout:
                logger.debug(
                    f"sp.solve timed out for {target} -- blacklisting."
                )
                cm.symbolic_blacklist.add(target)
                return []
            except (NotImplementedError, ValueError, TypeError) as e:
                # sympy's own "cannot invert this" signals; the relation is
                # simply not usable in that direction.
                logger.debug(f"sp.solve cannot invert for {target}: {e}")
                sols = []
            cm._symbolic_solve_cache[cache_key] = list(sols)
        sols = cm._symbolic_solve_cache[cache_key]

        def build():
            return [
                _Lambda(sol, sorted(sol.free_symbols, key=str)) for sol in sols
            ]

        return _compiled(cm, ("sols", eq, target), build)

    def _score_root(self, target, val, values, skip_pos):
        """How well a candidate root fits the rest of the relations -- the
        multi-root choice (review 2.1.15).

        The squared relative residual of every OTHER evaluable relation that
        contains `target`, plus one LOOKAHEAD: a root that leaves a
        neighboring relation with no real solution for its one remaining
        unknown is outside that relation's domain (e = -1 from the V_c/V_e
        quadratic leaves sqrt(e) cos(omega) with none), and scores a full
        unit worse than any root that does not."""
        temp = dict(values)
        temp[target] = val
        score = 0.0
        for pos in self.by_symbol.get(target, ()):
            if pos == skip_pos:
                continue
            names = self.relations[pos][1]
            missing = [n for n in names if n not in temp]
            if len(missing) == 1:
                sols = self._solutions(pos, missing[0])
                if sols and not any(
                    all(n in temp for n in lam.names)
                    and math.isfinite(lam(temp))
                    for lam in sols
                ):
                    score += 1.0
                continue
            if missing:
                continue
            res, _parts, fl, fr, period = self._residual(pos)
            r = res(temp)
            if not math.isfinite(r):
                continue
            ref = max(abs(fl(temp)), abs(fr(temp)), 1e-9)
            score += (wrap(r, period) / ref) ** 2
        return score

    def _solve_for(self, pos, target, values):
        """A Step that determines `target` from relation `pos`, or None."""
        cm = self.cm
        eq, names = self.relations[pos]
        parts = target.split(".")
        lookup = f"{parts[0]}.{parts[-1]}"
        idx = int(parts[1]) if len(parts) >= 3 and parts[1].isdigit() else 0
        inputs = tuple(n for n in names if n != target)
        label = self._label(pos)

        # 1. A registered custom solver replaces the symbolic inversion of
        #    ANY relation for its target (planet.mass from K, planet.log_q).
        #    KeyError is the documented "cannot solve yet" signal.
        if lookup in cm.custom_solvers:
            fn = cm.custom_solvers[lookup]
            sysc = cm.system_config

            def custom_eval(vals, fn=fn, idx=idx, sysc=sysc):
                try:
                    return float(fn(vals, sysc, idx))
                except KeyError:
                    return math.nan

            rec = _Recorder(values)
            if math.isfinite(custom_eval(rec)):
                read = tuple(sorted(k for k in rec.read if k != target))
                return Step("custom", target, read, label, custom_eval)

        lo, hi = self._bounds(target)
        sols = self._solutions(pos, target)
        if not sols:
            # 2. No closed form: nsolve, from the target's own anchor.
            a = self.anchors.get(target)
            state = {"guess": a.value if a is not None else 1.0}
            target_sym = sp.Symbol(target)

            def nsolve_eval(vals, eq=eq, inputs=inputs, state=state):
                from .config import _sympy_time_limit

                try:
                    with _sympy_time_limit():
                        sub = {sp.Symbol(n): vals[n] for n in inputs}
                        expr = (eq.lhs - eq.rhs).subs(sub).evalf()
                        v = float(sp.nsolve(expr, target_sym, state["guess"]))
                except Exception:  # noqa: BLE001
                    # nsolve's failure modes are many (ValueError,
                    # ZeroDivisionError, mpmath's convergence error, the
                    # alarm) and every one means the same thing here: no
                    # root from this guess, i.e. this relation does not
                    # determine the target.  It is an answer, not a fault.
                    return math.nan
                state["guess"] = v
                return v

            v = nsolve_eval(values)
            if math.isfinite(v) and lo <= v <= hi:
                return Step("nsolve", target, inputs, label, nsolve_eval)
            return None

        # 3. Every real, finite, in-bounds root; several -> scored.
        valid = []
        for k, lam in enumerate(sols):
            if not all(n in values for n in lam.names):
                continue
            v = lam(values)
            if math.isfinite(v) and lo <= v <= hi:
                valid.append((v, k))
        if not valid:
            return None
        ambiguous = False
        if len(valid) == 1:
            k = valid[0][1]
        else:
            scored = sorted(
                (self._score_root(target, v, values, pos), v, k_)
                for v, k_ in valid
            )
            k = scored[0][2]
            if all(t[1] == scored[0][1] for t in scored):
                # A DOUBLE root (|mu_rel| = +-sqrt(0) at two equal default
                # proper motions): equal here, but the branches part as soon
                # as the inputs move, and only one stays in bounds.  Choose
                # the one that does, one scale away.
                nudged = dict(values)
                for n in names:
                    if n != target and n in nudged:
                        sc = self.scale(n)
                        nudged[n] = nudged[n] + (sc if sc else 1.0)
                ok = [t for t in scored if lo <= sols[t[2]](nudged) <= hi]
                if ok:
                    k = ok[0][2]
            # a tie between DIFFERENT values: nothing yet says which
            ambiguous = scored[1][0] - scored[0][0] <= 1e-12 and not (
                scored[1][1] == scored[0][1]
            )
            a = self.anchors.get(target)
            if ambiguous and a is not None:
                # What was STATED about the target breaks the tie: the root
                # nearest its own start -- a dropped user value keeps its
                # sign (pi_E_E = +0.127 over its -0.127 twin), a defaults.yaml
                # value its branch.  It decides nothing else.
                tied = [t for t in scored if t[0] - scored[0][0] <= 1e-12]
                k = min(tied, key=lambda t: (abs(t[1] - a.value), t[1]))[2]
                ambiguous = False
        lam = sols[k]
        return Step(
            "rel",
            target,
            tuple(lam.names),
            label,
            lam,
            root=(pos, k),
            ambiguous=ambiguous,
        )

    # ------------------------------------------------------------------
    # propagation
    # ------------------------------------------------------------------
    def closure(self, fixed, excluded=frozenset()):
        """Propagate from `fixed` until nothing more can be determined."""
        cm = self.cm
        values = dict(fixed)
        steps = []
        determined_by = {}
        used = set()
        failed = set()
        progress = True
        while progress:
            progress = False
            deferred = None
            for target, expr_str, lam, _expr, deps in self.links:
                if target in values or not all(d in values for d in deps):
                    continue
                v = lam(values)
                if not math.isfinite(v):
                    continue
                st = Step("link", target, deps, f"user link: {expr_str}", lam)
                values[target] = v
                steps.append(st)
                determined_by[target] = st
                progress = True
            for key, path, idx in self.standalone:
                if path in values:
                    continue
                fn = cm.custom_solvers[key]

                def sa_eval(vals, fn=fn, idx=idx, sysc=cm.system_config):
                    try:
                        return float(fn(vals, sysc, idx))
                    except KeyError:
                        return math.nan

                rec = _Recorder(values)
                v = sa_eval(rec)
                if not math.isfinite(v):
                    continue
                st = Step(
                    "standalone",
                    path,
                    tuple(sorted(k for k in rec.read if k != path)),
                    f"{key} (standalone solver)",
                    sa_eval,
                )
                values[path] = v
                steps.append(st)
                determined_by[path] = st
                progress = True
            for pos, (_eq, names) in enumerate(self.relations):
                if pos in used or pos in excluded:
                    continue
                unknown = [n for n in names if n not in values]
                if len(unknown) != 1:
                    continue
                t = unknown[0]
                if (pos, t) in failed:
                    continue
                st = self._solve_for(pos, t, values)
                v = st.fn(values) if st is not None else math.nan
                if not math.isfinite(v):
                    failed.add((pos, t))
                    continue
                if st.ambiguous:
                    # A root chosen by order alone, where another relation
                    # may yet decide it (|mu_rel| gives mu_ra up to a sign;
                    # a pinned pi_E_E gives the sign).  Wait for the pass to
                    # run out of unambiguous work first.
                    if deferred is None:
                        deferred = (pos, t, st, v)
                    continue
                values[t] = v
                steps.append(st)
                determined_by[t] = st
                if st.kind != "custom":
                    # A custom solver states ITS OWN relation (the K ->
                    # mass inversion), not this one, so this one still has
                    # to hold: it stays a check.
                    used.add(pos)
                progress = True
            if not progress and deferred is not None:
                pos, t, st, v = deferred
                values[t] = v
                steps.append(st)
                determined_by[t] = st
                used.add(pos)
                progress = True

        checks = []
        for pos, (_eq, names) in enumerate(self.relations):
            if pos in used or pos in excluded:
                continue
            if not all(n in values for n in names):
                continue
            res, parts, _fl, _fr, period = self._residual(pos)
            checks.append(
                Check(
                    "rel",
                    self._label(pos),
                    names,
                    lambda vals, res=res, period=period: wrap(
                        res(vals), period
                    ),
                    parts,
                    pos,
                )
            )
        for target, expr_str, _lam, expr, deps in self.links:
            key = ("link", target)
            st = determined_by.get(target)
            if st is not None and st.kind == "link":
                continue
            if key in excluded or target not in values:
                continue
            if not all(d in values for d in deps):
                continue

            def build(target=target, expr=expr, deps=deps):
                syms = [sp.Symbol(target)] + [sp.Symbol(d) for d in deps]
                g = sp.Symbol(target) - expr
                return (
                    _Lambda(g, syms),
                    {str(s): _Lambda(sp.diff(g, s), syms) for s in syms},
                    tuple(str(s) for s in syms),
                )

            gl, parts, syms = _compiled(cm, ("link", target, expr), build)
            checks.append(
                Check(
                    "link",
                    f"{target} = {expr_str} (user link)",
                    syms,
                    gl,
                    parts,
                    key,
                )
            )
        return Plan(values, steps, checks, determined_by)

    @staticmethod
    def evaluate(plan, fixed):
        """Re-run `plan`'s steps from `fixed` (its structure held)."""
        values = dict(fixed)
        for st in plan.steps:
            values[st.target] = st.fn(values)
        return values

    # ------------------------------------------------------------------
    # depth and propagated scales, from the sampled leaves alone
    # ------------------------------------------------------------------
    def _leaf_pass(self, leaves):
        """Propagate from the sampled seeds alone, each at its anchor, to get
        every derived symbol's DEPTH (the keep-in-order key: 1 + the deepest
        input of the step that determines it) and its PROPAGATED init_scale
        (the split scale of a seed that declares none, and the scale the
        change test uses for a derived symbol).

        The scale is carried through each step as the root-sum-square of the
        step's response to moving each input by ITS scale -- the Jacobian
        propagation the engine always did, taken as a secant one scale wide,
        which is the same to first order and stays defined at a kink (|mu_rel|
        at mu_rel = 0, where the derivative is 0/0) and through a solver,
        whose inputs are read off the call.  It propagates only when every
        input the step reads has a scale: an absent scale is "unknown", never
        1e-9 (review 2.1.16)."""
        plan = self.closure({p: self.anchors[p].value for p in leaves})
        depth = {p: 0 for p in leaves}
        scales = {}
        vals = plan.values
        for st in plan.steps:
            ins = [depth[i] for i in st.inputs if i in depth]
            depth[st.target] = 1 + max(ins, default=0)
            base = vals[st.target]
            var, ok = 0.0, bool(st.inputs)
            for inp in st.inputs:
                s = self.declared_scales.get(inp, scales.get(inp))
                if s is None or not s > 0:
                    ok = False
                    break
                delta = math.nan
                for sign in (1.0, -1.0):
                    trial = dict(vals)
                    trial[inp] = vals[inp] + sign * s
                    delta = st.fn(trial) - base
                    if math.isfinite(delta):
                        break
                if not math.isfinite(delta):
                    ok = False
                    break
                var += wrap(delta, self.periodic.get(st.target)) ** 2
            if ok and var > 0.0:
                scales[st.target] = math.sqrt(var)
        self.leaf_scales = scales
        self.depth = depth

    # ------------------------------------------------------------------
    # the settle: minimum-norm in init_scale units, inside one tier
    # ------------------------------------------------------------------
    def _settle(self, fixed, movable, excluded, tier_name):
        """Move `movable` (seeds of ONE tier, all in `fixed`) the
        minimum-norm amount, in init_scale units, that satisfies every
        check; everything else in `fixed` is held.  Returns
        (fixed, plan, still-violated checks).

        Solved by Gauss-Newton minimum-norm steps through the propagation
        plan, finished by SLSQP on the same problem where they stall.  Both
        are tried first from the anchors and then, if that does not settle,
        from a DISPLACED start: a relation can be undefined at the anchors,
        or have no usable gradient there (two proper motions at the same
        default make mu_rel = 0, where |mu_rel| is a kink and t_E =
        theta_E / mu_rel has no value).  The displaced start is about one
        scale from each anchor, in a fixed direction whose components DIFFER
        (a common shift would leave a difference such as mu_rel where it
        was).  The minimum norm is always measured from the anchors; only
        where the problem is symmetric (a circle of equally short
        corrections) does the start choose among the answers."""
        plan = self.closure(fixed, excluded)
        bad = self.violated(plan, excluded)
        if not bad or not movable:
            return fixed, plan, bad

        a = np.array([self.anchors[m].value for m in movable])
        lo = np.array([self._bounds(m)[0] for m in movable])
        hi = np.array([self._bounds(m)[1] for m in movable])
        sm = np.array(
            [self.scale(m, plan) or 0.0 for m in movable], dtype=float
        )
        all_checks = [c for c in plan.checks if c.key not in excluded]

        def make_resid(checks):
            def resid(xv):
                f2 = dict(fixed)
                f2.update(zip(movable, xv))
                vals = self.evaluate(plan, f2)
                return np.array([c.residual(vals) for c in checks]), vals

            return resid

        def rscale(checks, vv):
            out = np.array([self._residual_scale(c, vv, plan) for c in checks])
            return np.where(out > 0.0, out, 1.0)

        def worst(checks, rr, vv):
            if len(rr) == 0:
                return 0.0
            if not np.all(np.isfinite(rr)):
                return np.inf
            return float(np.max(np.abs(rr / rscale(checks, vv))))

        def attempt(x0):
            # A relation undefined at this start (omega = atan2(0, 0) on a
            # circular orbit the user pinned) states nothing to settle.
            r0, v0 = make_resid(all_checks)(x0)
            checks = [c for c, ok in zip(all_checks, ~np.isnan(r0)) if ok]
            resid = make_resid(checks)
            x = x0.copy()
            r, vals = resid(x)
            if not np.all(np.isfinite(r)):
                return None
            # FD step: a millionth of a scale, but never below the
            # round-off of the value (tc carries a 0.001 d scale on 2.46e6 d)
            h = np.array(
                [
                    max(1e-6 * sj, 1e-8 * abs(xi), 1e-12)
                    for sj, xi in zip(sm, x)
                ]
            )

            def jac(xv, rv):
                J = np.zeros((len(checks), len(movable)))
                for j in range(len(movable)):
                    x2 = xv.copy()
                    x2[j] += h[j]
                    r2, _ = resid(x2)
                    J[:, j] = (r2 - rv) / h[j]
                return J

            J = jac(x, r)
            reach = np.any(np.abs(J) > 0.0, axis=0) & np.all(
                np.isfinite(J), axis=0
            )
            s = np.ones(len(movable))
            for j in np.flatnonzero(reach):
                s[j] = self.require_scale(
                    movable[j],
                    f"to split a correction among the {tier_name} seeds",
                    plan,
                )
            # a coordinate no check sees keeps its anchor
            if np.any(~reach & (x != a)):
                x[~reach] = a[~reach]
                r, vals = resid(x)
                J = jac(x, r)
            x_start = x.copy()

            def gauss_newton(x, r, vals, J):
                free = reach.copy()
                for _it in range(_SETTLE_MAX_ITER):
                    # each residual in units of its OWN scale at the current
                    # point: a scale taken where d t_E / d mu_rel is enormous
                    # would call any point converged
                    sr = rscale(checks, vals)
                    rn = r / sr
                    if not np.all(np.isfinite(rn)):
                        break
                    if np.max(np.abs(rn), initial=0.0) <= _SETTLE_TARGET:
                        break
                    cols = np.flatnonzero(free)
                    if cols.size == 0:
                        break
                    Ju = (J[:, cols] / sr[:, None]) * s[cols][None, :]
                    Ju = np.where(np.isfinite(Ju), Ju, 0.0)
                    u = (x[cols] - a[cols]) / s[cols]
                    # rcond well above the finite-difference noise (~1e-7 of
                    # a column): a direction the relations barely see must
                    # not be given the giant step a near-zero singular value
                    # asks for -- it is the noise of the difference quotient.
                    u_new = np.linalg.pinv(Ju, rcond=1e-6) @ (Ju @ u - rn)
                    accepted = False
                    for frac in (
                        1.0,
                        0.5,
                        0.25,
                        0.125,
                        1 / 16,
                        1 / 32,
                        1 / 64,
                    ):
                        xt = x.copy()
                        xt[cols] = a[cols] + s[cols] * (u + frac * (u_new - u))
                        hit = (xt < lo) | (xt > hi)
                        xt = np.clip(xt, lo, hi)
                        rt, vt = resid(xt)
                        if np.all(np.isfinite(rt)) and np.sum(
                            (rt / sr) ** 2
                        ) < np.sum(rn**2):
                            x, r, vals = xt, rt, vt
                            free &= ~hit
                            accepted = True
                            break
                    if not accepted:
                        break
                    J = jac(x, r)
                return x, r, vals, J

            x, r, vals, J = gauss_newton(x, r, vals, J)
            if worst(checks, r, vals) > _SETTLE_TARGET:
                x, r = self._settle_slsqp(
                    x,
                    r,
                    x_start,
                    a,
                    s,
                    reach,
                    lo,
                    hi,
                    rscale(checks, vals),
                    resid,
                )
                r, vals = resid(x)
                return x, worst(checks, r, vals)
            # Gauss-Newton reached the constraint surface, but nothing holds
            # it to the NEAREST point of it: a direction no check sees (the
            # common motion of a lens and a source proper motion, which
            # every relation reads only as a difference) rides along on the
            # difference-quotient noise of a stiff check, and no later step
            # brings it back.  Measured on KMT-2019-BLG-1806 with t_E and
            # pi_E pinned: both proper motions at +25,027 mas/yr for a 0.6
            # mas/yr relative motion.  So the minimum norm is imposed
            # explicitly: project the correction onto the row space of the
            # (scaled) Jacobian at the converged point -- dropping exactly
            # the part no check sees -- and re-converge from there; keep the
            # result only if it satisfies every check and is nearer.
            cols = np.flatnonzero(reach)
            for _round in range(3):
                if cols.size == 0:
                    break
                sr = rscale(checks, vals)
                Ju = (J[:, cols] / sr[:, None]) * s[cols][None, :]
                Ju = np.where(np.isfinite(Ju), Ju, 0.0)
                u = (x[cols] - a[cols]) / s[cols]
                # rcond well above the difference-quotient noise, which is
                # exactly what this projection exists to drop (measured: the
                # common-motion singular value at 1.3e-6 of the largest)
                u_p = np.linalg.pinv(Ju, rcond=1e-4) @ (Ju @ u)
                if not float(u_p @ u_p) < float(u @ u) * (1.0 - 1e-9):
                    break
                xt = x.copy()
                xt[cols] = np.clip(a[cols] + s[cols] * u_p, lo[cols], hi[cols])
                rt, vt = resid(xt)
                if not np.all(np.isfinite(rt)):
                    break
                xt, rt, vt, Jt = gauss_newton(xt, rt, vt, jac(xt, rt))
                ut = (xt[cols] - a[cols]) / s[cols]
                if worst(checks, rt, vt) > _SETTLE_TARGET or not float(
                    ut @ ut
                ) < float(u @ u):
                    break
                x, r, vals, J = xt, rt, vt, Jt
            return x, worst(checks, r, vals)

        best = attempt(np.array([fixed[m] for m in movable], dtype=float))
        if best is None or best[1] > _SETTLE_TARGET:
            tilt = 1.0 + np.arange(len(movable)) / max(len(movable), 1)
            other = attempt(np.clip(a + sm * tilt, lo, hi))
            if other is not None and (best is None or other[1] < best[1]):
                best = other
        if best is None:
            return fixed, plan, bad
        x = best[0]
        logger.debug(
            "settle at %s: moved %s; worst check %.3g scales",
            tier_name,
            [m for m, a_m, x_m in zip(movable, a, x) if a_m != x_m],
            best[1],
        )
        new_fixed = dict(fixed)
        new_fixed.update(zip(movable, x))
        plan = self.closure(new_fixed, excluded)
        for m, a_m, x_m in zip(movable, a, x):
            if a_m != x_m and value_moved(
                x_m, a_m, self.scale(m), m, self.periodic.get(m)
            ):
                self.moved[m] = (float(a_m), float(x_m), tier_name)
        return new_fixed, plan, self.violated(plan, excluded)

    @staticmethod
    def _settle_slsqp(x_gn, r_gn, x0, a, s, reach, lo, hi, sr, resid):
        """The same minimum-norm problem, by SLSQP, where the Gauss-Newton
        steps did not converge (a strongly nonlinear check -- t_E against
        1/|mu_rel| from a near-zero start -- can send them round a spiral).
        min ||(x - a)/s||^2 subject to every check, inside the bounds; the
        better of the two answers is kept."""
        from scipy.optimize import minimize

        cols = np.flatnonzero(reach)
        if cols.size == 0:
            return x_gn, r_gn

        def to_x(uc):
            xx = a.copy()
            xx[cols] = a[cols] + s[cols] * uc
            return xx

        def cons(uc):
            rr, _ = resid(to_x(uc))
            rn = rr / sr
            return np.where(np.isfinite(rn), rn, 1e6)

        def cons_jac(uc):
            base = cons(uc)
            J = np.zeros((base.size, cols.size))
            for j in range(cols.size):
                du = max(1e-6, 1e-8 * abs(uc[j]))
                u2 = uc.copy()
                u2[j] += du
                J[:, j] = (cons(u2) - base) / du
            return J

        bnds = [((lo[c] - a[c]) / s[c], (hi[c] - a[c]) / s[c]) for c in cols]
        u0 = (x0[cols] - a[cols]) / s[cols]
        try:
            res = minimize(
                lambda uc: 0.5 * float(uc @ uc),
                u0,
                jac=lambda uc: uc,
                method="SLSQP",
                bounds=bnds,
                constraints=[{"type": "eq", "fun": cons, "jac": cons_jac}],
                options={"maxiter": 500, "ftol": 1e-14},
            )
        except (ValueError, np.linalg.LinAlgError):
            return x_gn, r_gn
        x_sq = to_x(res.x)
        r_sq, _ = resid(x_sq)

        def worst(rr):
            rn = rr / sr
            return np.max(np.abs(rn)) if np.all(np.isfinite(rn)) else np.inf

        if worst(r_sq) < worst(r_gn):
            return x_sq, r_sq
        return x_gn, r_gn

    # ------------------------------------------------------------------
    def _upstream(self, plan, symbols, kinds, fixed):
        """The fixed values of the given anchor kinds upstream of `symbols`."""
        out, seen, stack = set(), set(), list(symbols)
        while stack:
            s = stack.pop()
            if s in seen:
                continue
            seen.add(s)
            st = plan.determined_by.get(s)
            if st is not None:
                stack.extend(st.inputs)
                continue
            a = self.anchors.get(s)
            if a is not None and a.kind in kinds and s in fixed:
                out.add(s)
        return sorted(out)

    def _candidate_order(self, cands):
        big = 10**9
        d = self.depth
        f = self.file_order
        mode = self.keep_order_mode
        if mode == "depth":
            return sorted(
                cands, key=lambda p: (d.get(p, big), f.get(p, big), p)
            )
        if mode == "file":
            return sorted(cands, key=lambda p: (f.get(p, big), p))
        if mode == "reverse_depth":
            return sorted(
                cands, key=lambda p: (-d.get(p, -1), f.get(p, big), p)
            )
        raise ValueError(f"unknown keep-in-order mode {mode!r}")

    def _tier_name(self, tier):
        names = [k for k, v in self.ladder.items() if v == tier]
        return ("/".join(names) if names else "precedence") + f" ({tier})"

    # ------------------------------------------------------------------
    # the solve
    # ------------------------------------------------------------------
    def solve(self):
        """Run the rule.  Returns {path: internal value}.

        One outer loop wraps the tiers, for the STANDALONE solvers (a
        one-way target := f(inputs) that no relation can express: the orbit
        mass sum, tc from a time of periastron, the transit chord from cos
        i, the eclipse-timing eccentricity).  A solver's answer is a value
        at PRECEDENCE_DERIVED_MIXED: it fills its target whenever the target
        is undetermined, and where the target ended the solve held at that
        level or below (a defaults.yaml backstop, a value derived from
        defaults) it is pinned there and the tiers are solved again.  It
        never moves its own inputs, and never overrides a data or user
        value."""
        t0 = time.time()
        A = self.anchors
        rigid = sorted(p for p, a in A.items() if a.kind == "rigid")
        cands = [p for p, a in A.items() if a.kind == "candidate"]
        seeds = sorted(
            p
            for p, a in A.items()
            if a.kind == "seed" and p not in self.link_targets
        )
        fills = sorted(
            p
            for p, a in A.items()
            if a.kind == "fill" and p not in self.link_targets
        )
        leaves = [p for p in rigid + seeds if A[p].sampled]
        self._leaf_pass(leaves)

        solver_seeds = {}
        for _round in range(_SOLVER_ROUNDS):
            fixed, plan, excluded = self._tiered(
                rigid, cands, seeds, fills, solver_seeds
            )
            ranks = self.report_ranks(plan, fixed, solver_seeds)
            changed = False
            for key, path, idx in self.standalone:
                if path not in plan.values:
                    continue
                if ranks.get(path, 0) > self.ladder["DERIVED_MIXED"]:
                    continue
                st = plan.determined_by.get(path)
                if st is not None and st.kind == "standalone":
                    continue  # this solver produced it: nothing to compare
                try:
                    v = float(
                        self.cm.custom_solvers[key](
                            plan.values, self.cm.system_config, idx
                        )
                    )
                except KeyError:
                    continue
                if not math.isfinite(v):
                    continue
                if not value_moved(
                    v,
                    plan.values[path],
                    self.require_scale(
                        path, "to compare a solver's answer", plan
                    ),
                    path,
                    self.periodic.get(path),
                ):
                    continue
                solver_seeds[path] = v
                changed = True
            if not changed:
                break
        else:
            raise RuntimeError(
                "The relaxation engine's standalone solvers did not settle in "
                f"{_SOLVER_ROUNDS} rounds ({sorted(solver_seeds)}): each round "
                "pins a solver's answer on its target and solves again, so an "
                "answer that keeps moving means the solver reads its own "
                "target through one of its inputs."
            )
        self.plan = plan
        self.fixed = fixed
        self.solver_seeds = solver_seeds
        self.excluded = excluded
        self.ranks = ranks
        self.keep_report["solver_seeds"] = sorted(solver_seeds)
        self.keep_report["wall_s"] = round(time.time() - t0, 3)
        return plan.values

    def _tiered(self, rigid, cands, seeds, fills, solver_seeds):
        A = self.anchors
        rep = {
            "contradictions": [],
            "keep_order": [],
            "kept": [],
            "dropped": [],
            "absorbed": [],
            "moved": [],
            "unsettled": [],
        }
        self.keep_report = rep
        self.diagnostics = []
        self.moved = {}
        excluded = set()
        fixed = {p: A[p].value for p in rigid}

        # 1. Rigid user values: report what they break among themselves,
        #    and take those relations out of the solve.
        plan = self.closure(fixed, excluded)
        for c in self.violated(plan):
            excluded.add(c.key)
            ups = self._upstream(plan, c.symbols, ("rigid",), fixed)
            rep["contradictions"].append([c.label, ups])
            self.diagnostics.append(
                (
                    "error",
                    f"Over-constrained relation '{c.label}' is violated: the "
                    f"values you set ({', '.join(ups) or 'its user links'}) "
                    f"fix every quantity in it, so no start can satisfy it.  "
                    f"It is left out of the start solve; drop one of them.",
                    sorted(set(ups) | set(c.symbols)),
                    "contradiction",
                )
            )

        # 2. User values on derived elements, kept in order.
        order = self._candidate_order(cands)
        rep["keep_order"] = [[p, self.depth.get(p)] for p in order]
        # The test is always the same one: pin the value with everything
        # kept so far and ask whether every relation can still hold.  Where
        # the kept values already DETERMINE it, the relation that did so
        # becomes a check, judged in the scales of its other symbols -- so a
        # value with no scale of its own (a pi_E pinned where no proper
        # motion is known yet) is still judged, and judged in the units of
        # what it is compared against.
        for p in order:
            a = A[p]
            trial = dict(fixed)
            trial[p] = a.value
            tplan = self.closure(trial, excluded)
            bad = self.violated(tplan, excluded)
            if bad:
                ups = self._upstream(
                    tplan,
                    sorted({s for c in bad for s in c.symbols}),
                    ("rigid", "candidate"),
                    trial,
                )
                self._drop(p, a, [u for u in ups if u != p], bad[0].label, rep)
                continue
            fixed = trial
            rep["kept"].append(p)

        # 3. The tiers, from the top of the ladder down.
        sol_tier = self.ladder["DERIVED_MIXED"]
        tiers = {}
        for p in seeds:
            tiers.setdefault(A[p].tier, []).append(p)
        for p in solver_seeds:
            tiers.setdefault(sol_tier, [])
        for tier in sorted(tiers, reverse=True):
            plan = self.closure(fixed, excluded)
            new, solved = [], []
            if tier == sol_tier:
                for p in sorted(solver_seeds):
                    if p not in plan.values:
                        solved.append(p)
            for p in tiers[tier]:
                if p in plan.values or p in solved:
                    rep["absorbed"].append(
                        [p, A[p].value, plan.values.get(p), tier]
                    )
                    continue
                new.append(p)
            if not new and not solved:
                continue
            for p in new:
                fixed[p] = A[p].value
            for p in solved:
                fixed[p] = solver_seeds[p]
            name = self._tier_name(tier)
            fixed, plan, bad = self._settle(fixed, new, excluded, name)
            for c in bad:
                excluded.add(c.key)
                rep["unsettled"].append([c.label, name])
                self.diagnostics.append(
                    (
                        "warning",
                        f"The {name} starts could not satisfy '{c.label}' "
                        f"inside their bounds; the start leaves it violated "
                        f"and the sampler has to reconcile it.",
                        list(c.symbols),
                        "unsettled",
                    )
                )
        rep["moved"] = [[m, *v] for m, v in sorted(self.moved.items())]

        # 4. Fills: a value for whatever nothing else determines.  Only
        #    null-space values (nothing that is held depends on one).
        plan = self.closure(fixed, excluded)
        for p in fills:
            if p in plan.values:
                continue
            trial = dict(fixed)
            trial[p] = A[p].value
            tplan = self.closure(trial, excluded)
            if self.violated(tplan, excluded):
                continue
            fixed, plan = trial, tplan
        return fixed, plan, excluded

    def _drop(self, p, a, ups, label, rep):
        rep["dropped"].append([p, a.value, ups, label])
        self.diagnostics.append(
            (
                "warning",
                f"Your start for {p} cannot hold together with the values "
                f"kept before it ({', '.join(ups) or 'the sampled starts'}): "
                f"'{label}' would be violated.  It was DROPPED, and the "
                f"model starts at the value those imply.",
                [p] + list(ups),
                "dropped",
            )
        )

    # ------------------------------------------------------------------
    def report_ranks(self, plan, fixed, solver_seeds):
        """The rank each value REPORTS (ledger, initval_source, the labels).

        It decides nothing.  A held seed reports its own rank; a kept user
        value USER; a seed a settle moved, or a solver's pinned answer, the
        solved level (DERIVED_MIXED); a propagated value the weakest rank
        among its inputs, capped at DERIVED_USER -- min(input ranks), which
        is what the ledger has always reported."""
        L = self.ladder
        out = {}
        for p in fixed:
            if p in solver_seeds and p not in self.anchors:
                out[p] = L["DERIVED_MIXED"]
            elif p in solver_seeds and fixed[p] == solver_seeds[p]:
                out[p] = L["DERIVED_MIXED"]
            elif p in self.moved:
                out[p] = L["DERIVED_MIXED"]
            else:
                out[p] = self.anchors[p].rank
        for st in plan.steps:
            if st.kind == "standalone":
                out[st.target] = L["DERIVED_MIXED"]
            elif st.kind == "link":
                out[st.target] = L["USER"]
            else:
                ins = [out[i] for i in st.inputs if i in out]
                out[st.target] = (
                    min(L["DERIVED_USER"], min(ins)) if ins else L["DEFAULT"]
                )
        return out
