"""The binary mode finder: a blind (s, q) grid with per-node re-minimization.

WHY THIS EXISTS.  `peakfind.py` seeds t_0/u_0/t_E and deliberately says
nothing about the companion ("s, alpha and q stay generic: the sampler finds
the companion", mulensing.md).  That holds where the sampler can actually
cross between topologies.  It cannot here: `lens.log_s` starts at its
defaults.yaml `initval: 0.0`, which is s = 1.0 EXACTLY -- the s <-> 1/s
symmetry point -- so topology is set at initialization by an initialization
containing no topology information, and the close/wide barrier is tall enough
that no tempering ladder in use crosses it (DC2018 event 128: 42,563 nats
between the two solutions, against the 213 nats the hottest rung of a
T_max = 200 ladder sees).  A fit that starts at s = 1 reports whichever side
it fell down, with no indication that the other side exists.

So the topology has to be found BEFORE the fit, by enumeration rather than by
sampling.  The basin is far too small to stumble into: within dchi2 <= 100 of
the global minimum the truth basin occupies 2, 16 and 55 nodes of 2911 on
DC2018 events 163, 069 and 194 -- 0.07% to 1.9% of the (s, q) plane (notes
2.4.14).  Enumerating it costs a fraction of one production fit, because each
node is a 4-parameter least-squares problem rather than a 34-parameter
posterior.

WHAT IT DOES.  Grid (log s, log q).  At each node re-minimize (t_0, u_0, t_E,
alpha) from `n_alpha_starts` alpha starts, with rho handled per `rho_mode`.
The nuisances MUST be re-minimized: freezing them at the point-lens fit costs
thousands of chi2 on a sharply peaked light curve -- event 163's point-lens
fit is off by 2% in u_0 and 1.7% in t_E, and that alone accounts for ~11,565
chi2 over its 38,568 points, which buries the binary signal entirely.  The
first attempt at this gridded (s, q, alpha) with the nuisances frozen and
ranked the truth 35th to 144th among local minima.

EVERY local minimum is kept, not just the best.  The two clean
central-caustic DC2018 events (069, 258) put their best solution on the
s <-> 1/s MIRROR of the truth, to better than 1.5%: a search reporting one
best fit hands back the wrong topology and no warning.  `GridResult.
local_minima` returns K distinct basins, and `seed_fragment` writes them as
the multi-seed start of the real fit.

BLIND.  Nothing here reads an answer key.  `pspl` is the peak finder's own
output (`peakfind.find_pspl_seed`) and rho is optimized inside a prior range
that is a statement about Roman-era sources, not about any one event.

WHAT IT DOES NOT MODEL.  Parallax, lens orbital motion and xallarap are all
absent: every node is a static, geocentric binary.  That is a deliberate
restriction of the SEARCH, not of the fit that consumes its seeds -- the
question a node answers is "does a companion at this (s, q) explain the
anomaly", and the higher-order effects move chi2 by far less than the
topology does.  Chi2 values from this module are therefore NOT comparable to
a fit's, only to each other.
"""

from __future__ import annotations

import dataclasses
import logging
import multiprocessing as mp
import time
import traceback

import numpy as np

from .peakfind import _solve_fluxes

logger = logging.getLogger(__name__)

#: Keep epochs within this many t_E of t_0 (`GridSpec.trim_t_E`).
#:
#: A survey light curve spans YEARS and the event spans weeks, so most of the
#: file constrains the baseline and nothing else -- and every node pays for
#: every epoch, through a VBBL magnification that is the whole cost of the
#: grid.  Measured on RMDC26 event 096 (t_E = 8.42 d, 49,096 epochs over a
#: 1717 d baseline): +/-3 t_E keeps 3,678 epochs (13.3x fewer), +/-5 t_E
#: 5,618 (8.7x), +/-10 t_E 8,185 (6.0x) -- and +/-20 t_E keeps the SAME 8,185,
#: because 10 t_E already reaches the ends of the observing season and the
#: other 40,911 epochs are other seasons.
#:
#: 10 is the default because the trim must not be allowed to decide the
#: answer.  The wings are what separate a binary from a point lens with a
#: different t_E, and a tight window around t_0 would let a wrong-t_E solution
#: hide outside it.  At 10 t_E the magnification is 1 + 2/u^4 <= 1.00002 of
#: the baseline, so the window is "the whole season" rather than a choice
#: about the event.
DEFAULT_TRIM_T_E = 10.0

#: A curve with fewer epochs than this is DROPPED by `trim_curves`
#: rather than kept: `_solve_fluxes` profiles (f_source, f_blend) out
#: analytically, so a 1-epoch curve fits itself exactly and adds 0 to
#: every node's chi2 -- invisible rather than absent.
MIN_CURVE_EPOCHS = 2

#: Blind PRIOR range for rho.  Two orders of magnitude above anything real
#: (the DC2018 static truths run 5e-4..1.7e-3); VBBL's finite-source contour
#: integration gets rapidly more expensive as rho grows.
RHO_LO = 1e-5
RHO_HI = 1e-1

#: Suppression radius between accepted basins, in DEX rather than in NODES,
#: and PER AXIS because log s and log q are not gridded at the same spacing.
#:
#: A node count does not transfer between grid resolutions: 3 nodes means
#: 0.03 dex on a 0.01 dex grid and 0.21 dex on a 0.07 dex one, and the
#: latter swallows genuinely distinct basins.  Measured over the 30-event
#: DC2018 sweep, a fixed 3 nodes finds the truth among 5 seeds on 25/30
#: where the scaled radius finds it on 27/30, losing events 069 and 103
#: outright.
#:
#: Only the LOG S radius is measured; 0.03 dex is what the original 71-node
#: grid's default of 3 nodes meant.  The log q radius is the SAME node count
#: carried over at that grid's q spacing (2 nodes x 0.1 dex), so the package
#: reproduces the sweep -- it has NOT been swept independently, and basin
#: structure in q finer than 0.2 dex would need it reduced.
MIN_SEP_DEX = 0.03
MIN_SEP_DEX_Q = 0.2

#: A radius of one node suppresses nothing and spends seeds on adjacent
#: nodes of a single basin, so both axes floor here.
MIN_SEP_NODES_FLOOR = 2


def _sep_nodes(radius_dex, spacing_dex):
    """A suppression radius in dex -> a whole number of nodes, floored."""
    if not np.isfinite(spacing_dex) or spacing_dex <= 0:
        return MIN_SEP_NODES_FLOOR
    return max(MIN_SEP_NODES_FLOOR, int(round(radius_dex / spacing_dex)))


@dataclasses.dataclass(frozen=True)
class GridSpec:
    """Everything a node needs that is not the data.  Frozen, so it is safe
    to hand to a worker pool -- the DC2018 original carried these in
    module-level mutable latches set from `main()`, which works only under
    `fork` and made a start-vector/unpack mismatch invisible (every node
    returned inf and the run "succeeded" with no finished nodes).
    """

    # Grid EXTENT.  log_s spans the FULL support of `lens.log_s`
    # (defaults.yaml: [-1, 1], s in [0.1, 10]), symmetric about 0 so the
    # close/wide reflection maps the domain onto itself and a basin and its
    # mirror are both on the grid.  log_q covers the planetary regime; it
    # deliberately stops short of `planet.log_q`'s [-9, 1] support, which
    # includes stellar mass ratios no survey-detectable anomaly needs.
    log_s_lo: float = -1.0
    log_s_hi: float = 1.0
    n_s: int = 141
    log_q_lo: float = -5.5
    log_q_hi: float = -1.5
    n_q: int = 41

    # alpha is periodic and the chi2 surface in it is multimodal, so the
    # per-node minimization restarts from this many evenly spaced alphas.
    n_alpha_starts: int = 4

    #: Keep |t - t_0| <= trim_t_E * t_E.  None disables the trim.
    trim_t_E: float | None = DEFAULT_TRIM_T_E

    # How rho is handled at each node.  "simplex" puts log10 rho in the NM
    # vector and costs a measured 3.4x (70 nodes/h against 237), most of it
    # the simplex wandering into large rho.  "separate" is the standard
    # microlensing treatment and the default: rho sets the SMOOTHING of a
    # caustic crossing, not its location or width, so it is largely
    # independent of s and q and does not belong in a simplex shared with
    # parameters it barely couples to.  The 4-D fit runs with rho held, a
    # 1-D Brent search moves rho at that optimum, then the 4-D fit is re-run
    # once from there -- ~15 extra evaluations against the simplex's ~600.
    rho_mode: str = "separate"
    rho_lo: float = RHO_LO
    rho_hi: float = RHO_HI
    #: Fix rho here instead of optimizing it (a diagnostic; not blind if the
    #: value came from an answer key).
    rho_fixed: float | None = None

    # Half-width of the VBBL finite-source window around t_0, in units of
    # t_E, floored at `vbbl_half_window_min` days.  SCALED, because the
    # window has to cover the caustic approach and that lasts a fixed
    # fraction of t_E, not a fixed number of days: the DC2018 original used
    # a flat 2 days, which is 0.24 t_E on a 8 d event and 0.02 t_E on a 100 d
    # one.  Pass `vbbl_half_window_t_E=None` with a `vbbl_half_window_min`
    # to get the old fixed window back.
    vbbl_half_window_t_E: float | None = 0.25
    vbbl_half_window_min: float = 2.0

    optimizer: str = "nm"
    maxiter: int = 600

    def __post_init__(self):
        if self.rho_mode not in ("separate", "simplex"):
            raise ValueError(
                f"rho_mode must be 'separate' or 'simplex', got "
                f"{self.rho_mode!r}"
            )
        if self.optimizer not in ("nm", "powell", "de"):
            raise ValueError(
                f"optimizer must be 'nm', 'powell' or 'de', got "
                f"{self.optimizer!r}"
            )
        if self.n_s < 1 or self.n_q < 1:
            raise ValueError(f"empty grid: n_s={self.n_s} n_q={self.n_q}")

    @property
    def log_s(self):
        return np.linspace(self.log_s_lo, self.log_s_hi, self.n_s)

    @property
    def log_q(self):
        return np.linspace(self.log_q_lo, self.log_q_hi, self.n_q)

    @property
    def d_log_s(self):
        return float(self.log_s[1] - self.log_s[0]) if self.n_s > 1 else np.inf

    def vbbl_window(self, t_0, t_E):
        """(lo, hi) epochs bracketing the finite-source method."""
        half = self.vbbl_half_window_min
        if self.vbbl_half_window_t_E is not None:
            half = max(half, self.vbbl_half_window_t_E * float(t_E))
        return t_0 - half, t_0 + half


def trim_curves(curves, t_0, t_E, n_t_E=DEFAULT_TRIM_T_E):
    """Drop epochs further than ``n_t_E`` einstein times from ``t_0``.

    Returns ``(curves, n_kept, n_total)``.  ``n_t_E`` of None is a no-op.

    A curve left with fewer than 2 epochs is dropped entirely rather than
    kept as an unconstrained (f_source, f_blend) pair: `_solve_fluxes`
    profiles both out analytically, so a 1-epoch curve fits itself exactly
    and contributes 0 to every node's chi2 -- it would be invisible instead
    of absent.  See `MIN_CURVE_EPOCHS`.
    """
    n_total = sum(int(np.size(c[0])) for c in curves)
    if n_t_E is None:
        return list(curves), n_total, n_total
    half = float(n_t_E) * float(t_E)
    out = []
    n_kept = 0
    for t, flux, ivar in curves:
        t = np.asarray(t)
        keep = np.abs(t - float(t_0)) <= half
        if int(keep.sum()) < MIN_CURVE_EPOCHS:
            continue
        out.append((t[keep], np.asarray(flux)[keep], np.asarray(ivar)[keep]))
        n_kept += int(keep.sum())
    if not out:
        raise ValueError(
            f"the +/-{n_t_E} t_E window around t_0 = {t_0} (t_E = {t_E}) "
            f"leaves no curve with {MIN_CURVE_EPOCHS} epochs; widen "
            f"trim_t_E or check the peak-finder seed"
        )
    return out, n_kept, n_total


# ---------------------------------------------------------------------------
# The node
# ---------------------------------------------------------------------------

#: Per-worker state, set once by `_init_worker`.  The light curves travel
#: ONCE per worker rather than once per task: a 2911-node grid on three
#: 8,000-epoch bands would otherwise pickle ~570 MB of identical arrays.
_W = {}

#: One traceback per WORKER PROCESS, not per node.  A node is allowed to
#: fail -- VBBL throws on pathological geometry and one inf among thousands
#: is normal, which is why `_node` catches everything -- but a catch that
#: wide hides a programming error in every node equally well, and presents
#: only as "no finished nodes" after the whole walltime is gone.
_REPORTED_FAIL = []


def _init_worker(curves, pspl, spec):
    from exozippy.compat import patch_mulensmodel_method_order

    # MulensModel dispatches magnification methods in PYTHONHASHSEED order
    # and the VBMicrolensing backends are not order-independent, so the same
    # node evaluated in two processes can differ.  `op.py` applies this for
    # the fit; a search that runs without importing `op` has to apply it
    # itself or it is not reproducible run to run.  The effect is ~3 ulp --
    # nothing against the chi2 differences that separate basins -- but a
    # grid whose numbers move between runs cannot be differenced.
    patch_mulensmodel_method_order()
    _W["curves"] = curves
    _W["pspl"] = pspl
    _W["spec"] = spec


def _node_chi2_factory(spec, curves, pspl):
    """Build the node objective.  Separated from `_node` so a test can call
    it on one node without a pool.
    """
    import MulensModel as mm

    t_0_0 = float(pspl["t_0"])
    t_E_0 = float(pspl["t_E"])
    fs_lo, fs_hi = spec.vbbl_window(t_0_0, t_E_0)
    simplex_rho = spec.rho_fixed is None and spec.rho_mode == "simplex"
    # A one-element list, not a float: the separate-rho Brent search writes
    # through it between calls, and a closure cannot rebind a float.
    rho_box = [
        float(spec.rho_fixed)
        if spec.rho_fixed is not None
        else float(np.sqrt(spec.rho_lo * spec.rho_hi))
    ]

    def nll(p, s, q):
        # The vector carries log10 rho ONLY in simplex mode.  Gating this on
        # "is rho free" instead hands `nll` a 5-vector to unpack into 4
        # names, and the ValueError is eaten by `_node`'s blanket catch --
        # every node returns inf and the run finishes in seconds with
        # nothing done (DC2018 job 50664420 lost 8 h of queue time to
        # exactly this).
        if simplex_rho:
            t0, u0, tE, a, lrho = p
            rho_use = 10.0**lrho
            if not (spec.rho_lo < rho_use < spec.rho_hi):
                return 1e30
        else:
            t0, u0, tE, a = p
            rho_use = rho_box[0]
        if not (tE > 0) or not np.isfinite(t0):
            return 1e30
        par = {
            "t_0": t0,
            "u_0": u0,
            "t_E": tE,
            "s": s,
            "q": q,
            "alpha": float(a) % 360.0,
            "rho": rho_use,
        }
        try:
            model = mm.Model(par)
            model.set_magnification_methods([fs_lo, "VBBL", fs_hi])
            total = 0.0
            for t, flux, ivar in curves:
                # Each band gets its OWN f_source and f_blend -- a different
                # source colour and a different blend -- while SHARING the
                # geometry.  That sharing is what makes a three-band node
                # worth more than three one-band nodes.
                chi2, _, _ = _solve_fluxes(
                    model.get_magnification(t), flux, ivar
                )
                if not np.isfinite(chi2):
                    return 1e30
                total += chi2
            return total
        except Exception:
            return 1e30

    return nll, rho_box, simplex_rho


def _minimize_node(spec, curves, pspl, log_s, log_q):
    """Minimize (t_0, u_0, t_E, alpha) -- and rho -- at one (s, q) node.

    Returns ``(chi2, alpha, t_0, u_0, t_E, rho)``, all nan/inf on failure.
    """
    from scipy.optimize import minimize

    nll, rho_box, simplex_rho = _node_chi2_factory(spec, curves, pspl)
    s, q = 10.0**log_s, 10.0**log_q
    t0_0, u0_0, tE_0 = (
        float(pspl["t_0"]),
        float(pspl["u_0"]),
        float(pspl["t_E"]),
    )

    def f(p):
        return nll(p, s, q)

    if spec.optimizer == "de":
        from scipy.optimize import differential_evolution

        bounds = [
            (t0_0 - 5.0, t0_0 + 5.0),
            (-3.0, 3.0),
            (max(0.1, 0.2 * tE_0), 5.0 * tE_0),
            (0.0, 360.0),
        ] + (
            [(np.log10(spec.rho_lo), np.log10(spec.rho_hi))]
            if simplex_rho
            else []
        )
        results = [
            differential_evolution(
                f, bounds, maxiter=150, tol=1e-6, polish=True, seed=0
            )
        ]
    else:
        method = "Nelder-Mead" if spec.optimizer == "nm" else "Powell"
        options = (
            {"maxiter": spec.maxiter, "xatol": 1e-6, "fatol": 1e-2}
            if spec.optimizer == "nm"
            else {"maxiter": spec.maxiter}
        )
        results = [
            minimize(
                f,
                (
                    [t0_0, u0_0, tE_0, a0, -3.0]
                    if simplex_rho
                    else [t0_0, u0_0, tE_0, a0]
                ),
                method=method,
                options=options,
            )
            for a0 in np.linspace(
                0.0, 360.0, spec.n_alpha_starts, endpoint=False
            )
        ]

    if spec.rho_fixed is None and spec.rho_mode == "separate" and results:
        from scipy.optimize import minimize_scalar

        r0 = min(results, key=lambda r: r.fun)

        def nll_rho(lr):
            rho_box[0] = 10.0**lr
            return f(list(r0.x[:4]))

        try:
            rr = minimize_scalar(
                nll_rho,
                bounds=(np.log10(spec.rho_lo), np.log10(spec.rho_hi)),
                method="bounded",
                options={"xatol": 1e-3, "maxiter": 40},
            )
            rho_box[0] = 10.0**rr.x
            results = results + [
                minimize(
                    f,
                    list(r0.x[:4]),
                    method="Nelder-Mead",
                    options={
                        "maxiter": spec.maxiter,
                        "xatol": 1e-6,
                        "fatol": 1e-2,
                    },
                )
            ]
        except Exception:
            pass

    best = np.inf
    out = (np.inf, np.nan, np.nan, np.nan, np.nan, np.nan)
    for r in results:
        if r.fun < best:
            best = float(r.fun)
            # KEEP the whole fitted vector, not just alpha.  A node is only
            # useful as a SEED if it carries t_0/u_0/t_E too; without them a
            # seeder has to re-minimize every node it wants, which is the
            # same work twice.
            out = (
                best,
                float(r.x[3] % 360.0),
                float(r.x[0]),
                float(r.x[1]),
                float(r.x[2]),
                10.0 ** float(r.x[4]) if simplex_rho else float(rho_box[0]),
            )
    if not np.isfinite(best):
        return (np.inf, np.nan, np.nan, np.nan, np.nan, np.nan)
    return out


def _node(task):
    i, j, log_s, log_q = task
    try:
        return (i, j) + _minimize_node(
            _W["spec"], _W["curves"], _W["pspl"], log_s, log_q
        )
    except BaseException:
        if not _REPORTED_FAIL:
            _REPORTED_FAIL.append(1)
            traceback.print_exc()
        return (i, j, np.inf, np.nan, np.nan, np.nan, np.nan, np.nan)


# ---------------------------------------------------------------------------
# The grid
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class GridResult:
    """A finished (or partial) grid.  `chi2`, `alpha`, `t_0`, `u_0`, `t_E`
    and `rho` are all (n_s, n_q); unfinished nodes are inf/nan.
    """

    log_s: np.ndarray
    log_q: np.ndarray
    chi2: np.ndarray
    alpha: np.ndarray
    t_0: np.ndarray
    u_0: np.ndarray
    t_E: np.ndarray
    rho: np.ndarray
    n_done: int
    n_tasks: int
    complete: bool
    n_epochs: int = 0
    n_epochs_total: int = 0

    @property
    def d_log_s(self):
        return (
            float(self.log_s[1] - self.log_s[0])
            if self.log_s.size > 1
            else np.inf
        )

    @property
    def d_log_q(self):
        return (
            float(self.log_q[1] - self.log_q[0])
            if self.log_q.size > 1
            else np.inf
        )

    def default_min_sep(self):
        """Basin suppression radius as ``(n_s_nodes, n_q_nodes)``.

        PER AXIS, because the two axes are not gridded at the same spacing
        -- the package default is 0.0143 dex per node in log s and 0.1 in
        log q, so one node count cannot mean the same distance in both.
        Applying the log s count to log q, which the DC2018 original did,
        makes the q radius an accident of the s resolution.
        """
        return (
            _sep_nodes(MIN_SEP_DEX, self.d_log_s),
            _sep_nodes(MIN_SEP_DEX_Q, self.d_log_q),
        )

    def local_minima(self, k=1, min_sep=None):
        """The ``k`` best nodes, no two within ``min_sep`` nodes.

        DISTINCT matters: adjacent nodes of one basin are not k modes, so a
        candidate within `min_sep` of an already-accepted one is suppressed
        and k seeds are k basins.

        Returns a list of dicts carrying log_s, log_q, s, q, alpha, t_0,
        u_0, t_E, rho, chi2 and dchi2 (against the global minimum), best
        first.
        """
        if min_sep is None:
            sep_s, sep_q = self.default_min_sep()
        elif np.isscalar(min_sep):
            sep_s = sep_q = int(min_sep)
        else:
            sep_s, sep_q = (int(v) for v in min_sep)
        picks = []
        for idx in np.argsort(self.chi2, axis=None):
            if not np.isfinite(self.chi2.flat[idx]):
                break
            i, j = np.unravel_index(idx, self.chi2.shape)
            # AND, not OR: two nodes are the same basin only when they are
            # close in BOTH coordinates.  The same s at a mass ratio 0.2 dex
            # away is a different solution, not a duplicate of this one.
            if any(
                abs(i - a) < sep_s and abs(j - b) < sep_q for a, b in picks
            ):
                continue
            picks.append((i, j))
            if len(picks) == k:
                break
        if not picks:
            raise ValueError("no finite nodes in the grid")
        best = float(self.chi2[picks[0]])
        out = []
        for i, j in picks:
            out.append(
                {
                    "log_s": float(self.log_s[i]),
                    "log_q": float(self.log_q[j]),
                    "s": float(10.0 ** self.log_s[i]),
                    "q": float(10.0 ** self.log_q[j]),
                    "alpha": float(self.alpha[i, j]),
                    "t_0": float(self.t_0[i, j]),
                    "u_0": float(self.u_0[i, j]),
                    "t_E": float(self.t_E[i, j]),
                    "rho": float(self.rho[i, j]),
                    "chi2": float(self.chi2[i, j]),
                    "dchi2": float(self.chi2[i, j]) - best,
                }
            )
        return out

    def mirror_of(self, seed, tol_dex=0.05):
        """The best node at `seed`'s s <-> 1/s mirror, or None.

        The close/wide degeneracy is the reason this module exists, so "did
        the grid see BOTH sides, and by how much does it prefer one" is a
        first-class question rather than something a caller re-derives from
        log_s arrays.

        Scans the mirror COLUMN directly rather than enumerating every basin
        and looking for one near -log_s: with K = n_nodes the suppression
        loop is O(K^2), which on the default 141 x 41 grid is 33 million
        comparisons for an answer that is one argmin over a few columns.

        Returns None when the mirror lies off the grid or no node within
        `tol_dex` of it ever finished -- which is the honest answer, and not
        the same as "the grid prefers this side".

        Also None when the seed sits so close to s = 1 that the mirror
        window overlaps the seed's OWN basin (the log s suppression radius
        `local_minima` uses).  Otherwise the window holds the seed's column
        and, q being unconstrained, the seed itself comes back as "mirror,
        dchi2 = +0.0" (RMDC26 event 096, s = 0.9676).  Merely dropping the
        seed's rows is not enough: the window then returns the far slope of
        the same basin at a small dchi2, the same false degeneracy.  Near
        s = 1 the caustic is resonant and there is no close/wide pair.
        """
        target = -float(seed["log_s"])
        own = self.default_min_sep()[0] * self.d_log_s
        if abs(target - float(seed["log_s"])) <= own + tol_dex:
            return None
        near = np.flatnonzero(np.abs(self.log_s - target) <= tol_dex)
        if near.size == 0:
            return None
        block = self.chi2[near, :]
        if not np.isfinite(block).any():
            return None
        flat = np.nanargmin(np.where(np.isfinite(block), block, np.inf))
        bi, bj = np.unravel_index(flat, block.shape)
        i = int(near[bi])
        j = int(bj)
        best = float(np.nanmin(self.chi2[np.isfinite(self.chi2)]))
        return {
            "log_s": float(self.log_s[i]),
            "log_q": float(self.log_q[j]),
            "s": float(10.0 ** self.log_s[i]),
            "q": float(10.0 ** self.log_q[j]),
            "alpha": float(self.alpha[i, j]),
            "t_0": float(self.t_0[i, j]),
            "u_0": float(self.u_0[i, j]),
            "t_E": float(self.t_E[i, j]),
            "rho": float(self.rho[i, j]),
            "chi2": float(self.chi2[i, j]),
            "dchi2": float(self.chi2[i, j]) - best,
        }

    def to_npz(self, path):
        np.savez_compressed(
            path,
            log_s=self.log_s,
            log_q=self.log_q,
            chi2=self.chi2,
            alpha=self.alpha,
            t_0=self.t_0,
            u_0=self.u_0,
            t_E=self.t_E,
            rho=self.rho,
            n_done=np.array([self.n_done]),
            n_tasks=np.array([self.n_tasks]),
            complete=np.array([self.complete]),
            n_epochs=np.array([self.n_epochs]),
            n_epochs_total=np.array([self.n_epochs_total]),
        )

    @classmethod
    def from_npz(cls, path):
        z = np.load(path)
        return cls(
            log_s=z["log_s"],
            log_q=z["log_q"],
            chi2=z["chi2"],
            alpha=z["alpha"],
            t_0=z["t_0"],
            u_0=z["u_0"],
            t_E=z["t_E"],
            rho=z["rho"],
            n_done=int(z["n_done"][0]),
            n_tasks=int(z["n_tasks"][0]),
            complete=bool(z["complete"][0]),
            n_epochs=int(z["n_epochs"][0]),
            n_epochs_total=int(z["n_epochs_total"][0]),
        )


def grid_search(
    curves,
    pspl,
    spec=None,
    *,
    ncpu=None,
    maxtime=None,
    progress_every=500,
    pool=None,
):
    """Run the (s, q) grid.  Returns a `GridResult`.

    ``curves`` is `peakfind`'s shape -- a list of ``(time, flux,
    inverse_variance)``, one per band, all in FLUX units.  ``pspl`` is the
    peak finder's dict (t_0, u_0, t_E); `find_pspl_seed` produces exactly
    it, and nothing here reads an answer key.

    ``maxtime`` (seconds) stops accepting results and returns what finished.
    Size it UNDER the scheduler's walltime: the only write is at the end, so
    a run that hits the wall loses everything -- a DC2018 run did 3000 of
    11644 nodes in 24 h and kept none of it.  Unfinished nodes stay inf/nan
    and `local_minima` skips them, so a partial grid is still a usable grid.
    """
    spec = spec or GridSpec()
    curves = [
        (np.asarray(t), np.asarray(f), np.asarray(w)) for t, f, w in curves
    ]
    n_total = sum(int(c[0].size) for c in curves)
    if spec.trim_t_E is not None:
        curves, n_epochs, n_total = trim_curves(
            curves, pspl["t_0"], pspl["t_E"], spec.trim_t_E
        )
        logger.info(
            "grid: +/-%g t_E around t_0 = %.4f keeps %d of %d epochs "
            "(%.1fx fewer); every node pays for every epoch.",
            spec.trim_t_E,
            float(pspl["t_0"]),
            n_epochs,
            n_total,
            n_total / max(n_epochs, 1),
        )
    else:
        n_epochs = n_total

    log_s, log_q = spec.log_s, spec.log_q
    tasks = [
        (i, j, ls, lq)
        for i, ls in enumerate(log_s)
        for j, lq in enumerate(log_q)
    ]
    # SHUFFLED, deterministically, and that is load-bearing for `maxtime`.
    # In row-major order a grid stopped at its walltime has evaluated a
    # contiguous SLAB of low log_s and nothing above it -- so a run that
    # does 60% of the nodes reports a confident global minimum having never
    # looked at the wide half, which is precisely the s <-> 1/s mirror this
    # module exists to expose.  Shuffled, a partial grid is an unbiased
    # subsample of the whole box: coarser everywhere rather than complete in
    # one corner.  Fixed seed so a partial grid is still reproducible.
    np.random.default_rng(0).shuffle(tasks)
    shape = (spec.n_s, spec.n_q)
    chi2 = np.full(shape, np.inf)
    arrs = {
        k: np.full(shape, np.nan)
        for k in ("alpha", "t_0", "u_0", "t_E", "rho")
    }

    if ncpu is None:
        ncpu = default_grid_cores()
    logger.info(
        "grid: %d x %d = %d nodes on %d core(s), %d alpha start(s) each, "
        "rho %s",
        spec.n_s,
        spec.n_q,
        len(tasks),
        ncpu,
        spec.n_alpha_starts,
        "fixed at %g" % spec.rho_fixed
        if spec.rho_fixed is not None
        else spec.rho_mode,
    )

    done = 0
    stopped_early = False
    t_start = time.time()
    own_pool = pool is None
    if own_pool:
        # The platform-default start method is fine HERE, and only because
        # the workers get their state through initializer/initargs rather
        # than from a module global: spawn (macOS) and forkserver (Linux,
        # Python 3.14) deliver initargs, but inherit no runtime global
        # (review 2.6.42; tests/test_pool_start_method.py).
        pool = mp.Pool(
            ncpu, initializer=_init_worker, initargs=(curves, pspl, spec)
        )
    try:
        for i, j, c, a, t0, u0, tE, rho in pool.imap_unordered(
            _node, tasks, chunksize=4
        ):
            chi2[i, j] = c
            arrs["alpha"][i, j] = a
            arrs["t_0"][i, j] = t0
            arrs["u_0"][i, j] = u0
            arrs["t_E"][i, j] = tE
            arrs["rho"][i, j] = rho
            done += 1
            if progress_every and done % progress_every == 0:
                logger.info("grid: %d/%d nodes", done, len(tasks))
            if maxtime and (time.time() - t_start) > maxtime:
                stopped_early = True
                logger.warning(
                    "grid: maxtime %.0f s reached at %d/%d nodes -- "
                    "returning the partial grid.",
                    maxtime,
                    done,
                    len(tasks),
                )
                pool.terminate()
                break
    finally:
        if own_pool:
            # terminate() already ran on the maxtime path; close() after it
            # is a no-op, but join() is not -- without it the partial grid
            # returns while workers are still being reaped.
            if not stopped_early:
                pool.close()
            pool.join()

    return GridResult(
        log_s=log_s,
        log_q=log_q,
        chi2=chi2,
        alpha=arrs["alpha"],
        t_0=arrs["t_0"],
        u_0=arrs["u_0"],
        t_E=arrs["t_E"],
        rho=arrs["rho"],
        n_done=done,
        n_tasks=len(tasks),
        complete=not stopped_early,
        n_epochs=n_epochs,
        n_epochs_total=n_total,
    )


def default_grid_cores():
    """Cores for the pool, respecting a scheduler's CPU grant.

    `len(os.sched_getaffinity(0))` and NOT `mp.cpu_count()`: inside a 4-CPU
    Slurm allocation on a 112-core node, measured, `cpu_count()` returns 112
    and the affinity mask returns 4.  A pool sized from `cpu_count` oversubs
    the grant by 28x and the scheduler simply time-slices it.
    """
    import os

    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):  # pragma: no cover - non-Linux
        return max(1, mp.cpu_count())


# ---------------------------------------------------------------------------
# Grid -> the fit's start population
# ---------------------------------------------------------------------------

#: Which of a seed's numbers reach a params file, and under which element of
#: the path.  The caller supplies the element NAMES (`source.0` vs
#: `source.Source`, `lens.1` vs `lens.Companion`); only the leaf is fixed.
#:
#: WHAT IS DELIBERATELY ABSENT, and why -- both are DERIVED in a model with a
#: galactic prior, so an `initval` on them is a silent no-op:
#:
#:   t_E   = theta_E / mu_rel_geo, from masses, distances and proper motions.
#:   rho   = theta_star / theta_E, from the source radius and distance.
#:
#: This is not theoretical.  RMDC26 event 096's params file seeds both, and
#: the run's own auditor reports `mulensevent.0.t_E: you set 8.42027, the
#: model starts at 9.7949 (+16.33%) -- initval has no effect on a derived
#: parameter` and `source.0.rho: you set 0.00662299, the model starts at
#: 0.000976985 (-85.25%)`.  To convey a t_E, seed the LEAVES it derives from
#: (the lens and source proper motions, with the masses and distances) --
#: mulensing.md, "Seeding: the built-in peak finder", review 8.6.15.
SEED_LEAVES = ("t_0", "u_0", "log_s", "log_q", "alpha")

#: Leaves the grid measures but cannot convey, with the reason, so a writer
#: can say so in the file rather than dropping them silently.
SEED_DERIVED = {
    "t_E": "theta_E / mu_rel_geo (masses, distances, proper motions)",
    "rho": "theta_star / theta_E (source radius and distance)",
}


def seed_fragment(
    seeds,
    names,
    *,
    overdisperse=True,
    seed_polish=True,
    header=(),
):
    """Write K grid basins as a params-file fragment.

    ``names`` maps each leaf of `SEED_LEAVES` to the full parameter path in
    the config being seeded, e.g. ``{"t_0": "source.0.t_0", "log_s":
    "lens.Companion.log_s", ...}``.  `mkparam`'s multi-seed path consumes
    the result: a scalar `initval` for K = 1, a length-K list for K > 1.

    ``overdisperse`` DEFAULTS TO TRUE, and that is the opposite of what it
    looks like it should be.  The key answers exactly one question
    (samplers.md, review 8.3.3): are these K points K MODES, each of which
    still wants scattering, or are they K draws from one posterior that are
    ALREADY dispersed?  Grid basins are the first.  Declaring `false` means
    "use these seeds exactly as they are": every chain starts at its
    round-robin seed with NO jitter, so the start population's affine hull
    is the hull of the K unique seeds.  With 5 basins and a 34-parameter
    model that is 3 directions against the n_params + 2 = 36 needed to span
    the space at all, and the only escape is `DE_JITTER = 1e-4`, a random
    walk that needs ~1e8 accepted steps to cover one whitened sigma --
    `_common.warn_if_seed_population_degenerate` says exactly this at
    sampler start.  With `true`, K chains start exactly on the K basins (up
    to `max_exact = n_chains // 2`) and the rest start jittered around their
    round-robin basin, so the modes are kept AND the population spans.

    ``seed_polish`` is NOT written into this fragment -- it is a
    SAMPLER-config key, read from the config YAML's `sampler:` block
    (`run.py`: `sampler_cfg.get("seed_polish", "auto")`), while
    `overdisperse` is a params-file key (`System.overdisperse`).  Writing it
    here would not reach the polish at all, and worse: `finalize_user_params`
    registers every unmapped params-file key as a LEAF SYMBOL in the
    relaxation engine, so it would become an orphaned ledger row for a
    parameter that does not exist (CLAUDE.md, the `user_params` invariant).
    True (the default) therefore emits a COMMENT saying to set it, and
    `sampler_overrides()` returns the same thing for a generator that writes
    the config YAML.

    It matters because `auto` resolves to ZERO steps for any multi-seed set
    without component hints (`polish.resolve_polish_steps`), which would
    leave each seed sitting exactly on its grid node.  A node is not a local
    optimum in (s, q): it is the best point on a LATTICE, up to half a
    node-spacing from the basin's actual minimum.  Whether the polish
    converges from that offset is the one untested step in the chain (notes
    2.4.14), so it is worth being explicit rather than silently off.
    """
    seeds = list(seeds)
    if not seeds:
        raise ValueError("no seeds")
    missing = [k for k in SEED_LEAVES if k not in names]
    if missing:
        raise KeyError(
            f"`names` is missing a path for {missing}; it must map every "
            f"leaf of SEED_LEAVES {list(SEED_LEAVES)} to this config's "
            f"spelling (the element name differs between configs)"
        )
    lines = list(header)
    lines += [
        "#",
        "# %d basin(s) from an (s, q) grid search "
        "(components/mulensing/gridsearch.py)." % len(seeds),
        "#",
    ]
    for n, s in enumerate(seeds):
        lines.append(
            "#   %d: s=%.4f q=%.3g alpha=%6.1f  t_0=%.4f u_0=%+.4f"
            "  chi2=%.1f (+%.1f)"
            % (
                n,
                s["s"],
                s["q"],
                s["alpha"],
                s["t_0"],
                s["u_0"],
                s["chi2"],
                s.get("dchi2", 0.0),
            )
        )
    lines += [
        "#",
        "# The grid also measured t_E and rho at each basin, and they are",
        "# NOT written here: both are DERIVED in a model with a galactic",
        "# prior, so an initval on them is a silent no-op (the run's",
        "# auditor reports it as 'initval has no effect on a derived",
        "# parameter').  To convey a t_E, seed the leaves it derives from.",
    ]
    for k, why in SEED_DERIVED.items():
        # Skip a value the grid did not record.  A grid written before it
        # kept rho per node has none, and "nan" in a comment reads as a
        # measurement that came out nan rather than one never made.
        vals = ", ".join(
            "%.6g" % s[k] for s in seeds if k in s and np.isfinite(s[k])
        )
        if vals:
            lines.append("#   %s = %s -- measured: %s" % (k, why, vals))
    lines += [
        "",
        "# K basins are K MODES, which still want scattering: see",
        "# samplers.md (8.3.3) and seed_fragment's docstring.  `false` here",
        "# would start every chain exactly on one of %d points with no"
        % len(seeds),
        "# jitter, spanning %d directions of the parameter space."
        % max(len(seeds) - 2, 0),
        "overdisperse: %s" % ("true" if overdisperse else "false"),
    ]
    if seed_polish:
        lines += [
            "",
            "# A grid node is the best point on a LATTICE, not a local",
            "# optimum, and `seed_polish: auto` is 0 steps for a multi-seed",
            "# set -- so each seed would stay up to half a node-spacing from",
            "# its basin's minimum.  seed_polish is a SAMPLER key and does",
            "# not belong in this file; put it in the config YAML:",
            "#",
            "#     sampler:",
            "#       seed_polish: %s"
            % ("true" if seed_polish is True else seed_polish),
        ]
    lines.append("")
    for k in SEED_LEAVES:
        vals = [float(s[k]) for s in seeds]
        v = vals[0] if len(vals) == 1 else vals
        lines.append("%s:" % names[k])
        lines.append("  initval: %r" % (v,))
    return "\n".join(lines) + "\n"


def sampler_overrides(seed_polish=True):
    """The `sampler:` block entries a grid-seeded fit needs.

    `seed_fragment` can only write the PARAMS file, and `seed_polish` is not
    a params-file key -- see its docstring.  A generator that writes the
    config YAML as well (RMDC26's `make_config.py`, DC2018's) merges this
    into the `sampler:` block instead of relying on the comment.
    """
    return {"seed_polish": bool(seed_polish)}


# ---------------------------------------------------------------------------
# Reading an event the way the FIT reads it
# ---------------------------------------------------------------------------


def curves_from_system(system):
    """`(time, flux, inverse_variance)` per band off a prepared `System`.

    The search has to see the data the fit sees, or it optimizes a different
    chi2 from the one it is seeding.  Going through `System.prepare()` is
    what guarantees that: the magnitude -> flux conversion, the `mask:`
    entries, the detrending and the per-file row ranges are all applied by
    `MulensInstrument.load_data`, once, and this only slices the result.
    Re-reading the files here instead would be a second loader to keep in
    step with the first.

    `prepare()` alone is enough; `build_model()` is not needed and is the
    expensive half.
    """
    inst = getattr(system, "mulensinstrument", None)
    if inst is None:
        raise ValueError(
            "the system has no mulensinstrument component, so there is no "
            "microlensing light curve to grid"
        )
    curves = []
    for i in range(len(inst.names)):
        rows = inst.rows(i)
        t = np.asarray(inst.time[rows], dtype=float)
        f = np.asarray(inst.flux[rows], dtype=float)
        e = np.asarray(inst.err[rows], dtype=float)
        ok = np.isfinite(t) & np.isfinite(f) & np.isfinite(e) & (e > 0)
        if ok.sum() >= MIN_CURVE_EPOCHS:
            curves.append((t[ok], f[ok], 1.0 / e[ok] ** 2))
    if not curves:
        raise ValueError("no usable epochs in any band")
    return curves


def pspl_start(curves, system=None, use_config=False):
    """The point-lens (t_0, u_0, t_E) the grid re-minimizes from.

    BLIND BY DEFAULT: `peakfind.find_pspl_seed` on these curves, with no
    reference to a params file or an answer key.  That is the right default
    because it makes the grid self-contained -- the same call on the same
    data gives the same start whatever a config happens to say -- and
    because a params file's own seed may not be what the model starts at.

    ``use_config=True`` reads the config's resolved start instead, for a
    grid that must begin exactly where its fit does.  Note the trap it walks
    into: `mulensevent.0.t_E` is DERIVED from masses, distances and proper
    motions wherever a galactic prior is in play, so what comes back is the
    engine's back-solve and not the number in the file.  On RMDC26 event 096
    the file says 8.42027 and the model starts at 9.7949, a 16% difference,
    and the run's own auditor reports it.  Both are usable starts here --
    every node re-minimizes t_E anyway -- but they are not the same number
    and the caller should know which one it asked for.
    """
    if use_config:
        if system is None:
            raise ValueError("use_config=True needs a prepared system")
        probed = system.config_manager.probe_start(
            ["source.0.t_0", "source.0.u_0", "mulensevent.0.t_E"]
        )
        out = {}
        for leaf, path in (
            ("t_0", "source.0.t_0"),
            ("u_0", "source.0.u_0"),
            ("t_E", "mulensevent.0.t_E"),
        ):
            val = getattr(probed.get(path), "user_value", None)
            if val is None or not np.isfinite(val):
                raise ValueError(
                    f"the config has no start for {path}; the grid needs a "
                    f"point-lens t_0/u_0/t_E to re-minimize from"
                )
            out[leaf] = float(val)
        return out

    from . import peakfind

    seed = peakfind.find_pspl_seed(curves)
    if seed is None:
        raise ValueError(
            "the peak finder found nothing fittable in these curves; the "
            "grid needs a point-lens t_0/u_0/t_E to re-minimize from"
        )
    return {k: float(seed[k]) for k in ("t_0", "u_0", "t_E")}
