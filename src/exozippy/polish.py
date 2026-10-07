"""Pre-whitening seed polish: promote each solution-estimate start to its
basin's optimum BEFORE anything downstream consumes the start.

Why before whitening: the startup probe (whitening.probe_scales) measures
logp contours around the start.  From a start far below its basin's optimum
the contours are gradient-dominated and the measured scales come out
arbitrarily tight -- on examples/ob140939 a start ~5900 nats low (2026-08,
raw error bars underestimated; err_scale starting at 1.0) measured scales
~1000x too small, NUTS's mass matrix re-widened the raw posterior to sd ~5e3
units, and 86% of draws diverged against parameter.py's
_RAW_CANCELLATION_CLIP wall at |raw| = 1e4.  (The same example's four
literature seeds start 565-3493 nats below their polished values today,
2026-09-14.)  Polishing first fixes that at the source; the probe then
measures curvature at a genuine optimum, the barrier steepness measurement
uses honest unit steps, PTDE scatters chains around a real basin center, and
NUTS starts inside the typical set.  This generalizes the PTDE-only seed
polish (samplers/ptde.py polish_seed_starts, PR #56), which ran inside
_make_starts -- after the whitening probe had already measured its scales
around the unpolished start, and never on the NUTS path at all.

ONE DRIVER, TWO ENGINES (JDE 2026-10-07: "Can we have the same code path
only differing in the optimizer call to prevent this kind of drift?").  The
two engines used to be two code paths, and they drifted: review 2.4.14 gave
the DE polish rounds, a rate stop, a raised cap and a progress line, while
L-BFGS kept "one round, as before" with a hard 400-iteration cap -- and on
examples/ob140939 every seed hit that cap 2-7 nats below its basin optimum
(the independent run to convergence, 13000-29000 iterations from the same
start, is the reference), so the whitening was measured at an unconverged
point.  Now everything that is not the optimizer lives here, once:

- the ROUNDS (polish_rounds): polish -> re-center + re-whiten -> polish,
  until a round gains less than polish_tol_nats(D) on every seed, at most
  POLISH_MAX_ROUNDS;
- the IN-ROUND stopping rule (PolishMonitor): a per-round cap, a RATE stop
  (a seed whose best lp gained less than polish_tol_nats(D) over the last
  POLISH_TOL_WINDOW steps), and a WALL-CLOCK budget for the whole polish
  (``sampler: polish_timeout:``, DEFAULT_POLISH_TIMEOUT_S) that STOPS it;
- the progress line and the per-seed stop reason, in the per-round line and
  in the wrap-up: "converged: <why>", "stopped on rate", "stopped on cap",
  "stopped on timeout" (and "stopped by the optimizer: <message>" for an
  L-BFGS-B exit that is none of these, e.g. a failed line search).

An engine supplies only the optimizer: it is handed the seeds and the
monitor, advances each seed's optimizer, and reports every step's best lp to
``monitor.step(s, lp)``, stopping the seed the moment that returns True.  An
engine-side convergence statement (L-BFGS-B's projected-gradient test) is
reported with ``monitor.finish(s, "converged", why)``.  _Engine names the
rest of what differs -- the unit a step is counted in and the default
per-round cap -- and nothing else.

- L-BFGS-B on the compiled logp + gradient in raw space (raw space is
  unconstrained -- hard bounds live inside the logit transform -- and smooth:
  bounds are soft barriers, not -inf walls).  A step is one ITERATION; the
  monitor sees it through scipy's per-iteration callback and ends the run by
  raising StopIteration, so ONE scipy run per seed per round keeps its
  curvature memory (no chunked restarts).  Its own convergence test is the
  projected-gradient norm (_LBFGS_GTOL), reported as "converged: |grad| <
  1e-4 nats/unit".  No per-round iteration cap by default: the gradient
  test, the rate stop and the timeout bound it.
- The PR #56 T=1 DE-MC polish (samplers.ptde.polish_seed_starts; gradient-
  free, the sampler's own move) when the gradient graph cannot be built or
  is non-finite at the start -- e.g. the binary-lens magnification Op has no
  analytic gradient.  A step is one SWEEP; the per-round cap is
  DEFAULT_DE_POLISH_SWEEPS.  It has no convergence test of its own: its only
  observable, the best-lp history, is a STAIRCASE of exactly-flat plateaus,
  so no improvement threshold separates "converged" from "has not jumped
  yet" (ptde's note above POLISH_TOL_WINDOW has the measurements) -- which is
  why a rate stop ends a ROUND, never the polish.

Why the rounds help L-BFGS too.  Round 1 runs in PRELIMINARY whitening units
(the probe has not measured anything yet), and on ob140939 the preliminary
coordinates leave a flat, badly scaled direction that L-BFGS crawls along:
the reference runs gain the last 2-7 nats over 13000-29000 iterations.
Re-whitening at round 1's point measures that direction, and round 2 climbs
it in measured units.  The per-iteration gains of round 1 on that example
are why the window is 400 steps for L-BFGS as for DE: windows of 10-100
iterations fire during the slow first bend of the climb (10-13 nats short
with ~3000 still to come on seeds 2 and 3), 200-400 fire on the crawl.

Gains are compared in ABSOLUTE nats against polish_tol_nats(D) everywhere;
see _LBFGS_FTOL for why never relative to |lp|.

The caps always remain (on the number of rounds, and on the wall clock), so
nothing can polish forever.

Seed-provenance gate (resolve_polish_steps): 'auto' polishes SOLUTION
ESTIMATES -- the single canonical start (user/literature initvals, the
relaxation engine's solution) and component seed sets -- but never a
multi-seed set WITHOUT seed hints, which is a posterior-draw restart
(mkparam stratified draws): those are already at equilibrium, and polishing
K draws per basin would collapse them onto K copies of the basin optimum,
destroying the restart's overdispersion.  Nor, since review 2.4.14 (d), a
params file that DECLARES its seeds to be posterior draws
(`overdisperse: false`, which mkparam writes for K > 1): the declaration is
the writer's statement, and seed hints pushed by a component on top of such
a file used to switch the polish back on and collapse the draws.
"""

import logging
import math
import multiprocessing as mp
import time
from collections import deque

import numpy as np

from .logger import fmt_duration

logger = logging.getLogger(__name__)

# Per-ROUND sweep cap on the gradient-free DE polish (review 2.4.14, JDE
# ruling 2026-09-14: "raise the cap").  Order 10^3, not 10^4, because that is
# where a fresh population does its work: on examples/DC2018_128's two seeds
# run to 15000 sweeps, the first 1000 brought 80% / 74% of the total gain and
# the next 14000 the rest, at 14x the cost, from a population that had
# frozen (80-90% of its sweeps accepting nothing).  More sweeps than that go
# to a NEW round -- re-whitened, re-populated -- instead (polish_rounds).
DEFAULT_DE_POLISH_SWEEPS = 1000

# The L-BFGS engine has NO default per-round iteration cap (None).  It had
# one, DEFAULT_POLISH_STEPS = 400, and on examples/ob140939 it was the stop
# that fired on every seed, 2.4-6.7 nats short of the basin optimum (module
# docstring).  What bounds an L-BFGS round now is what bounds a DE round --
# the rate stop and the wall-clock budget -- plus its own gradient test.
# The cap's other documented job, bounding hierarchical-MAP drift toward a
# degenerate corner, is the rate stop's: a climb that still gains
# polish_tol_nats(D) per POLISH_TOL_WINDOW iterations is moving the start by
# more than the typical-set depth, and one that does not is stopped.
DEFAULT_LBFGS_POLISH_ITERATIONS = None

# Cap on the number of polish rounds.  With the DE per-round cap this bounds
# the DE polish at 8000 sweeps; the accepted price (JDE 2026-09-14: CPU-days,
# on EXOFASTv2 precedent) is the wall clock, not the bound.
POLISH_MAX_ROUNDS = 8

# RATE-STOP WINDOW, in the engine's steps (DE sweeps, L-BFGS iterations): a
# seed leaves its round when its best lp gained less than
# polish_tol_nats(D) over the last POLISH_TOL_WINDOW steps.  400 for both
# engines, each measured:
#
# - DE: the window has to outlast every flat run of a LIVE population (the
#   longest in a fresh population's first 1000 sweeps on DC2018_128 was
#   168 / 228), and it fires on the frozen population, where a restart pays
#   -- the full table is in the note above ptde.POLISH_TOL_WINDOW.
# - L-BFGS: on examples/ob140939 (D = 17, tol 8.5 nats, preliminary
#   whitening) the climb's FIRST iterations can gain less than 8.5 nats per
#   10-100 iterations before the line search finds the valley, so windows of
#   10 / 25 / 50 / 100 fire 13-110 iterations in, 4.8-13 nats short with up
#   to ~3400 still to climb; 200 and 400 fire on the crawl along the flat
#   direction (210-472 iterations), and 400 is the one that is never early.
#   A window can only fire after POLISH_TOL_WINDOW + 1 steps, so a seed whose
#   gradient test converges sooner (examples/kelt4: 240-294 iterations) never
#   meets it.
POLISH_TOL_WINDOW = 400

# Wall-clock budget for the WHOLE polish (every round, every seed), in
# seconds; `sampler: polish_timeout:` overrides it and `null` removes it.  It
# STOPS the polish -- each running seed ends its round "stopped on timeout"
# with its best point so far, and no further round starts -- rather than
# raising an alarm.  24 h, because the slowest polish measured end to end,
# DC2018-226's gradient-free six rounds, took 4563 s on 4 workers, and the
# ruling that made rounds automatic accepted CPU-days: a budget that cut a
# measured run short would change the DE defaults' meaning.  On the L-BFGS
# engine it is a backstop -- ob140939's whole polish is seconds.  Checked at
# step boundaries (one L-BFGS iteration, one DE sweep), never mid-evaluation.
DEFAULT_POLISH_TIMEOUT_S = 24.0 * 3600.0

# Wall-clock heartbeat for the polish, in seconds -- both engines, one line
# format (PolishMonitor.heartbeat).  WALL CLOCK rather than "every N steps"
# because the quantity a watcher needs is "is this process alive", and a DE
# sweep on a binary-lens model can take anywhere from milliseconds to minutes
# -- a step count that is chatty on one model is silent for 40 minutes on
# another.  30 s is short enough that a human (or an agent) never has to
# sample /proc/<pid>/stat to tell computing from hung, and long enough that a
# fast model adds a handful of lines to the whole run.  A short polish emits
# NOTHING: the first heartbeat is one interval in, so the test suite and
# every sub-30 s polish stay silent (review 2.3.5, which cost a wrong
# diagnosis in the session that found it).
POLISH_PROGRESS_S = 30.0


class _EngineDefaultCap:
    """Sentinel step cap: "the dispatched engine's own default".

    The two engines have different per-round caps (none for L-BFGS,
    DEFAULT_LBFGS_POLISH_ITERATIONS; DEFAULT_DE_POLISH_SWEEPS sweeps for DE),
    because their steps are different units, and which one runs is only
    known once the gradient graph has been tried -- so resolve_polish_steps
    cannot return a number for 'auto'/'on'.  Truthy, so `if polish_steps:`
    still reads "polish".
    """

    def __repr__(self):
        return "ENGINE_DEFAULT"


ENGINE_DEFAULT = _EngineDefaultCap()


def polish_tol_nats(n_params):
    """The polish's improvement threshold, in nats, for a D-dim model.

    max(1, D/2): a T=1 posterior's typical set lies ~D/2 nats below its
    maximum (the mean of chi2_D / 2), so a polish gain smaller than that
    moves the start by less than the depth the sampler will spread its chains
    over anyway -- it is invisible to everything downstream.  Used twice, for
    the same reason, on both engines: as the in-round rate threshold (a seed
    that gains less than this over POLISH_TOL_WINDOW steps stops) and as the
    round
    threshold (a round that gains less than this ends the loop).  The floor
    keeps a 1-parameter model from stopping on a half-nat jitter.  ABSOLUTE
    nats; see _LBFGS_FTOL for why never relative to |lp|.  Measured on the
    DC2018_128 histories the window's stop does not depend on the threshold
    (tol 1, 13.5 = D/2 and 50 nats stop on the same sweep): the staircase
    makes the window LENGTH the operative choice.
    """
    return max(1.0, 0.5 * float(n_params))


# L-BFGS's OWN stop: the GRADIENT, never per-iteration improvement (the
# driver's windowed rate stop is a different test -- 400 iterations, not
# one; POLISH_TOL_WINDOW has why that is not the trap described here). scipy's `ftol` fires on the FIRST
# iteration whose gain is small -- and it is RELATIVE to |f|, so the old
# 1e-3 quit whenever an iteration gained < 1e-3*|lp| (~2 nats at
# lp ~ 1900), stranding ob140939's seeds ~15 nats below their basin
# peaks when first measured (2026-08); re-measured 2026-09-14 on the
# current model (lp ~ 10600, so the threshold is ~10 nats/iteration) it
# quits after 6-11 iterations, 10-119 nats below where the gradient stop
# + cap land the same seeds.  Even an absolute per-iteration threshold
# dies on a slow first bend of a curved valley (measured: iteration 1
# gains 0.004 nats, the remaining 2.0 arrive over the next 33). ftol is
# therefore disabled.
#
# gtol is nats per raw unit -- one preliminary whitening scale.  It was
# 1e-2 ("deep inside the flat top of the basin"), and that reasoning is
# right about a well-conditioned basin and WRONG about a ridge, which is
# what a real start climbs: scipy's gtol test is `max|proj g| <= gtol`
# on the CURRENT iterate, so it fires on the FIRST evaluation that dips
# under the threshold, whether or not the neighbours do.  Measured on
# examples/kelt4 RV-only (review 7.13.8, 2026-09-14): of 177 evaluations
# exactly one had |grad|_inf < 0.01 -- the last; the one before it read
# 0.017 and four before that 0.19 -- and it fired 0.507 nats BELOW the
# basin optimum (81.440 against 81.947) on a ridge whose Hessian has
# condition number 5.4e6 (the tc/logP degeneracy; orbit.0.tc sits ~1200
# orbits from the RV data), at iteration 148 of a 150 cap.  A first-dip
# stop on such a ridge is an amplifier: a synthetic ONE-ULP perturbation
# of (lp, grad) -- the objective times (1 + s*2^-52), i.e. the same
# function computed by a different but equally correct arithmetic --
# moved the polished cosi across 0.487..0.530 (8.5% relative width over
# 16 seeds), the planet mass by 2.1%, the polished lp by 0.24 nats, and
# 3 of the 16 arithmetics hit the 150 cap.  That is the whole mechanism
# behind the cross-CI-runner scatter of tests/test_integration_kelt4.py:
# a different OpenBLAS kernel inside scipy's own L-BFGS-B bookkeeping
# (not the objective, which has no BLAS op) perturbs one iterate by 1
# ulp at evaluation 4 and the stop lands elsewhere on the shoulder.
#
# At 1e-4 the same 16 perturbed arithmetics span 8.6e-4 in cosi, 3.0e-4
# in the mass and 1e-6 nats in lp, all sitting at the basin optimum,
# for ~120 more iterations (240-294 against 134-150) and +0.05 s.  (Then
# under a 400-iteration cap, since removed: the driver's rate window cannot
# fire before iteration 401, so this band is still a gradient stop.)
# tests/test_polish.py pins the perturbation spread under the shipped
# constants so a first-dip stop cannot come back unnoticed.
_LBFGS_FTOL = 1e-12  # effectively off; gtol + the driver terminate
# scipy's own iteration / evaluation limits, set out of reach: the driver
# (PolishMonitor) owns the cap, through the per-iteration callback.
_LBFGS_SCIPY_LIMIT = 10**9
_LBFGS_GTOL = 1e-4
# Non-finite logp guard: L-BFGS line searches handle inf poorly, so a
# non-finite evaluation returns this plateau plus |x - x0|^2, whose gradient
# 2*(x - x0) points back at THE POLISH START x0 -- not at the last finite
# iterate, which the objective never records.  x0 is the fixed seed the
# polish was handed (and logp/grad there are checked finite before the
# L-BFGS path is taken at all), so the pull is always toward a point the
# logp is known to be defined at; the quadratic is what makes the plateau
# navigable, since a flat 1e15 gives the line search no direction.
_NONFINITE_PENALTY = 1e15

# Sentinel: "caller said nothing", so the driver's defaults apply.  `None`
# cannot serve -- tol=None is the meaningful "disable the rate stop, run to
# the cap" request.
_UNSET = object()


# The stop kinds PolishMonitor records, and their log spellings
# (PolishMonitor.describe).  "converged" and "optimizer" are the ENGINE's
# statements (L-BFGS-B's own termination); the rest are the driver's.
STOP_CONVERGED = "converged"
STOP_RATE = "rate"
STOP_CAP = "cap"
STOP_TIMEOUT = "timeout"
STOP_OPTIMIZER = "optimizer"
_ENGINE_STOPS = (STOP_CONVERGED, STOP_OPTIMIZER)


class PolishMonitor:
    """The polish's stopping rule and progress line: ONE, for both engines.

    An engine hands every seed's progress here and stops a seed the moment
    ``step`` returns True; it never decides by itself to stop on a cap, a
    rate or a clock.  That is the whole point of the class (JDE 2026-10-07):
    when the stopping rule lived inside each engine the two drifted -- the
    DE polish got rounds, a rate stop and a progress line, L-BFGS kept a
    400-iteration cap that stranded ob140939 short of its optimum.

    The protocol, per seed ``s`` of one round:

    - ``begin(s, lp0)`` once, with the seed's own lp; True means "do not
      start" (the polish's wall-clock budget is already spent, or the cap
      is zero).
    - ``step(s, best_lp)`` after every engine step (one L-BFGS iteration,
      one DE sweep) with the best lp the seed has reached; True means stop
      now, and the reason is recorded.  The tests, in this order: the RATE
      (gain < ``tol`` over the last ``window`` steps; ``tol=None`` or
      ``window=0`` switches it off), the wall-clock DEADLINE, the CAP
      (``cap=None``: none).
    - ``observe(s, lp)`` (optional) with any better lp seen between steps,
      so a progress line is current mid-step.
    - ``finish(s, kind, detail)`` when the ENGINE ends a seed on its own
      criterion (L-BFGS-B's projected-gradient test: ``"converged"``); a
      no-op on a seed the driver already stopped.
    - ``heartbeat(...)`` as often as the engine likes; it is rate-limited
      to ``progress_interval_s`` of wall clock and emits one line.

    ``unit`` is the plural name of an engine step ("sweeps", "iterations");
    ``label`` prefixes every progress line; ``log`` is the logger they go
    to (each engine keeps its historical channel).  ``deadline`` is an
    absolute ``time.monotonic()`` instant shared by every round of one
    polish, and ``budget_s`` the budget it came from (for the log only).
    """

    def __init__(
        self,
        n_seeds,
        *,
        unit,
        cap,
        tol,
        window,
        label,
        deadline=None,
        budget_s=None,
        log=None,
        progress_interval_s=POLISH_PROGRESS_S,
        opening_text="scoring the starting points",
    ):
        if not str(unit).endswith("s"):
            raise ValueError(
                f"PolishMonitor: unit={unit!r} must be a plural step name "
                f"such as 'sweeps' or 'iterations'"
            )
        self.n_seeds = int(n_seeds)
        self.unit = str(unit)
        self.cap = None if cap is None else int(cap)
        self.tol = None if tol is None else float(tol)
        self.window = int(window) if window else 0
        self.label = label
        self.deadline = deadline
        self.budget_s = budget_s
        self.log = log if log is not None else logger
        self.progress_interval_s = progress_interval_s
        self.opening_text = opening_text
        self.lp0 = [None] * self.n_seeds
        self.best = [-math.inf] * self.n_seeds
        self.steps = [0] * self.n_seeds
        self.stop_kind = [None] * self.n_seeds
        self.stop_detail = [""] * self.n_seeds
        self._hist = [
            deque(maxlen=self.window + 1) for _ in range(self.n_seeds)
        ]
        self.t_start = time.monotonic()
        self._t_last_log = self.t_start
        self._beat_t = self.t_start
        self._beat_done = 0.0

    # --- the stopping rule ----------------------------------------------

    def _stop(self, s, kind, detail=""):
        self.stop_kind[s] = kind
        self.stop_detail[s] = detail
        return True

    def _past_deadline(self):
        return self.deadline is not None and time.monotonic() >= self.deadline

    def begin(self, s, lp0):
        """Record seed ``s``'s starting lp; True means do not start it."""
        lp0 = float(lp0)
        self.lp0[s] = lp0
        self.best[s] = lp0
        self._hist[s].clear()
        if self._past_deadline():
            return self._stop(s, STOP_TIMEOUT)
        if self.cap is not None and self.cap <= 0:
            return self._stop(s, STOP_CAP)
        return False

    def observe(self, s, lp):
        """A better lp seen between steps (progress line only)."""
        if lp > self.best[s]:
            self.best[s] = float(lp)

    def step(self, s, best_lp):
        """One engine step done for seed ``s``; True means stop it now."""
        if self.stop_kind[s] is not None:
            return True
        self.steps[s] += 1
        self.observe(s, best_lp)
        if self.tol is not None and self.window:
            h = self._hist[s]
            h.append(self.best[s])
            if len(h) > self.window and h[-1] - h[0] < self.tol:
                return self._stop(s, STOP_RATE)
        if self._past_deadline():
            return self._stop(s, STOP_TIMEOUT)
        if self.cap is not None and self.steps[s] >= self.cap:
            return self._stop(s, STOP_CAP)
        return False

    def finish(self, s, kind, detail=""):
        """The ENGINE ended seed ``s`` on its own criterion."""
        if kind not in _ENGINE_STOPS:
            raise ValueError(
                f"PolishMonitor.finish: {kind!r} is not an engine stop "
                f"({_ENGINE_STOPS}); the driver's own stops are step()'s"
            )
        if self.stop_kind[s] is None:
            self._stop(s, kind, detail)

    def done(self, s):
        return self.stop_kind[s] is not None

    def n_live(self):
        return sum(1 for k in self.stop_kind if k is None)

    def timed_out(self):
        return any(k == STOP_TIMEOUT for k in self.stop_kind)

    def describe(self, s):
        """Seed ``s``'s stop reason, in the words the log uses."""
        kind = self.stop_kind[s]
        if kind == STOP_CONVERGED:
            return f"converged: {self.stop_detail[s]}"
        if kind == STOP_RATE:
            return (
                f"stopped on rate: gained < {self.tol:.3g} nats over the "
                f"last {self.window} {self.unit}"
            )
        if kind == STOP_CAP:
            return f"stopped on cap: ran all {self.cap} {self.unit}"
        if kind == STOP_TIMEOUT:
            budget = (
                f"{self.budget_s:.0f} s " if self.budget_s is not None else ""
            )
            return (
                f"stopped on timeout: the polish's {budget}wall-clock budget "
                f"ran out"
            )
        if kind == STOP_OPTIMIZER:
            return f"stopped by the optimizer: {self.stop_detail[s]}"
        raise RuntimeError(
            f"PolishMonitor.describe: seed {s} has no recorded stop -- the "
            f"engine returned it without step() or finish() ending it"
        )

    # --- the progress line ----------------------------------------------

    def heartbeat(self, n_done=None, partial=None, opening=None, detail=None):
        """One progress line, RATE-LIMITED to ``progress_interval_s``.

        Idempotent by construction: every caller shares the one clock, so
        calling it once per evaluation produces no more lines than calling
        it once per step -- which is what makes it safe to hand to a pool's
        poll hook.  ``n_done`` is the step count the engine is at (default:
        the fewest steps among live seeds), ``partial=(k, m)`` marks a step
        still in flight with k of its m evaluations back, ``opening=(k, m)``
        the opening batch before any step exists, and ``detail`` a callable
        returning engine-specific text, called only when a line is written.
        """
        if not self.progress_interval_s:
            return
        now = time.monotonic()
        if now - self._t_last_log < float(self.progress_interval_s):
            return
        self._t_last_log = now
        elapsed = fmt_duration(now - self.t_start)
        if opening is not None:
            # No step exists yet, so there is no rate to extrapolate from.
            k, m = opening
            self.log.info(
                f"{self.label}: {self.opening_text} ({k}/{m} evaluations "
                f"back)  elapsed={elapsed}"
            )
            return
        live = [s for s in range(self.n_seeds) if self.stop_kind[s] is None]
        if n_done is None:
            n_done = min((self.steps[s] for s in live), default=0)
        one = self.unit[:-1]
        cap = "" if self.cap is None else f"/{self.cap}"
        if partial is None:
            done = float(n_done)
            where = f"{one} {n_done}{cap}"
        else:
            k, m = partial
            done = n_done + (k / m if m else 0.0)
            where = (
                f"{one} {n_done + 1}{cap} IN PROGRESS ({k}/{m} proposals back)"
            )
        # The ETA comes from the CURRENT rate -- steps since the previous line
        # over the time since it -- not the history average: on DC2018-226
        # the DE sweep rate fell 66/min -> 2.6/min as the population migrated
        # into the expensive region, and the average understated the
        # remaining time 3x (review 2.4.14).  It is an UPPER bound and
        # labelled as one: it extrapolates to the cap, and the rate stop or
        # the engine's own test can end a seed sooner.  No cap, no ETA.
        d_done = done - self._beat_done
        d_t = now - self._beat_t
        if d_done > 0 and d_t > 0:
            rate = d_done / d_t
            rate_txt = f"{60.0 * rate:.3g} {self.unit}/min"
            eta = (
                fmt_duration((self.cap - done) / rate)
                if self.cap is not None
                else None
            )
        else:
            rate_txt = f"no {one} completed since the last line"
            eta = "?" if self.cap is not None else None
        self._beat_t, self._beat_done = now, done
        pace = (
            f"eta<={eta} ({rate_txt})" if eta is not None else f"({rate_txt})"
        )
        gains = ", ".join(
            "-"
            if self.lp0[s] is None
            else f"{self.best[s] - self.lp0[s]:+.1f}"
            for s in range(self.n_seeds)
        )
        extra = detail() if detail is not None else ""
        self.log.info(
            f"{self.label}: {where}  elapsed={elapsed}  {pace}  "
            f"dlp=[{gains}]{extra}  ({len(live)} seed(s) still running)"
        )


def resolve_polish_timeout(spec):
    """Map the sampler-config `polish_timeout` value to seconds or None.

    A positive number is the wall-clock budget, in seconds, for the whole
    polish (every round, every seed); ``None`` -- YAML ``null`` -- removes
    it.  The caller supplies DEFAULT_POLISH_TIMEOUT_S when the key is absent.
    Anything else raises: a budget of 0, a negative one or a bool is a
    typo, not a request, and silently running without a bound (or stopping
    at once) would be the wrong reading of either.
    """
    if spec is None:
        return None
    if isinstance(spec, bool) or not isinstance(spec, (int, float)):
        raise ValueError(
            f"sampler: polish_timeout: {spec!r} is not a number of seconds; "
            f"give a positive number, or null for no wall-clock budget"
        )
    if not math.isfinite(spec) or spec <= 0:
        raise ValueError(
            f"sampler: polish_timeout: {spec!r} must be a positive number "
            f"of seconds (null removes the budget)"
        )
    return float(spec)


def resolve_polish_steps(spec, n_seeds, has_seed_hints, *, overdisperse):
    """Map the sampler-config `seed_polish` value to a step CAP.

    'auto' (default): ENGINE_DEFAULT (the dispatched engine's own cap) when
    the starts are solution estimates -- a single canonical start
    (n_seeds == 1) or component-pushed seed hints (the peak finder) -- and 0
    for a multi-seed set without hints (posterior-draw restarts; see module
    docstring), and 0 whenever the params file declares
    `overdisperse: false` (``overdisperse``, System.overdisperse): the
    writer's statement that its seeds ARE posterior draws outranks seed
    hints a component pushed on top of them (review 2.4.14 (d)).
    True/'on' and False/None/'off' force it; an int gives the cap directly
    (`seed_polish: N` = "at most N steps" per ROUND -- L-BFGS iterations or
    DE sweeps -- not "exactly N"; the rate stop, the timeout and L-BFGS's
    gradient test can each end a round first; see "ONE DRIVER, TWO ENGINES"
    in the module docstring).  Forcing a polish onto a declared
    posterior-draw set is honored but warned about.

    ``overdisperse`` is REQUIRED: it was the missing input (2.4.14 (d)), and
    a default would let the next call site forget it the same way.

    The bool test comes FIRST and by isinstance.  `spec in (True, "on")`
    matched the integer 1 (1 == True in Python), so `seed_polish: 1` asked
    for one step and got 150 (review 2.9.1, #104).  The
    symmetric `0 == False` match was harmless -- 0 steps IS off -- and stays
    harmless here: 0 now falls through to the int path and returns 0.
    """
    if not isinstance(overdisperse, bool):
        raise TypeError(
            f"resolve_polish_steps: overdisperse must be the params file's "
            f"boolean declaration (System.overdisperse), got {overdisperse!r}"
        )
    if isinstance(spec, bool):
        cap = ENGINE_DEFAULT if spec else 0
    elif spec is None:
        cap = 0
    elif isinstance(spec, str) and spec.lower() == "auto":
        if not overdisperse:
            return 0
        return ENGINE_DEFAULT if (n_seeds == 1 or has_seed_hints) else 0
    elif isinstance(spec, str) and spec.lower() == "on":
        cap = ENGINE_DEFAULT
    elif isinstance(spec, str) and spec.lower() == "off":
        cap = 0
    else:
        cap = max(0, int(spec))
    if cap and not overdisperse and n_seeds > 1:
        logger.warning(
            f"seed_polish: {spec!r} forces the polish onto {n_seeds} seeds "
            f"the params file declares to be posterior draws "
            f"(`overdisperse: false`).  Each will be driven to its basin "
            f"optimum, which collapses the draws' spread; set "
            f"`seed_polish: auto` to keep them as written."
        )
    return cap


def _compile_logp_grad(model):
    """Compiled point-function returning [logp, grad_1, ..., grad_n] over
    model.value_vars, or None when the gradient graph cannot be built (an
    Op without an analytic gradient, e.g. binary-lens magnification)."""
    import pytensor

    value_vars = list(model.value_vars)
    try:
        lp_node = model.logp()
        grads = pytensor.grad(lp_node, wrt=value_vars)
        return model.compile_fn(
            [lp_node] + list(grads),
            inputs=value_vars,
            on_unused_input="ignore",
        )
    except Exception as e:
        logger.info(
            f"Seed polish: gradient graph unavailable ({type(e).__name__}: "
            f"{e}); falling back to the gradient-free DE polish."
        )
        return None


def _lbfgs_stop(res):
    """(kind, detail) for an L-BFGS-B run the driver did not stop."""
    msg = str(getattr(res, "message", "")).strip()
    upper = msg.upper()
    if "PGTOL" in upper or "PROJECTED GRADIENT" in upper:
        return STOP_CONVERGED, f"|grad| < {_LBFGS_GTOL} nats/unit"
    if "REL_REDUCTION" in upper or "RELATIVE REDUCTION" in upper:
        return STOP_CONVERGED, "logp stopped changing at machine precision"
    return STOP_OPTIMIZER, msg or f"status {getattr(res, 'status', '?')}"


def _lbfgs_polish_one(center, fn_lp_grad, keys, shapes, sizes, monitor, s):
    """L-BFGS-B ascent of logp from one raw start dict -- seed ``s`` of
    ``monitor``'s round.

    The ENGINE half only: one scipy run, whose per-iteration callback hands
    the iterate's lp to ``monitor.step`` and ends the run (StopIteration)
    when the driver says so -- cap, rate or timeout.  Curvature memory is
    therefore kept for the whole round; nothing is restarted in chunks.
    scipy's own limits are out of reach (_LBFGS_SCIPY_LIMIT); its gradient
    test (_LBFGS_GTOL) is reported to the monitor as "converged".

    Returns (polished_dict, lp0, lp_best, n_evals)."""
    from scipy.optimize import minimize

    def flatten(d):
        return np.concatenate(
            [np.asarray(d[k], dtype=float).reshape(-1) for k in keys]
        )

    def unflatten(x):
        out, ofs = {}, 0
        for k, shp, n in zip(keys, shapes, sizes):
            out[k] = x[ofs : ofs + n].reshape(shp)
            ofs += n
        return out

    x0 = flatten(center)
    n_evals = [0]

    def objective(x):
        n_evals[0] += 1
        vals = fn_lp_grad(unflatten(x))
        lp = float(vals[0])
        if not np.isfinite(lp):
            return (
                _NONFINITE_PENALTY + float(np.sum((x - x0) ** 2)),
                2.0 * (x - x0),
            )
        g = np.concatenate(
            [np.asarray(v, dtype=float).reshape(-1) for v in vals[1:]]
        )
        if not np.all(np.isfinite(g)):
            g = np.where(np.isfinite(g), g, 0.0)
        return -lp, -g

    def unchanged():
        return {
            k: np.array(v, dtype=float, copy=True) for k, v in center.items()
        }

    lp0 = -objective(x0)[0]
    if monitor.begin(s, lp0):
        return unchanged(), lp0, lp0, n_evals[0]

    def callback(intermediate_result):
        stop = monitor.step(s, -float(intermediate_result.fun))
        monitor.heartbeat(
            n_done=monitor.steps[s], detail=lambda: f"  (seed {s})"
        )
        if stop:
            raise StopIteration

    res = minimize(
        objective,
        x0,
        jac=True,
        method="L-BFGS-B",
        callback=callback,
        options={
            "maxiter": _LBFGS_SCIPY_LIMIT,
            "maxfun": _LBFGS_SCIPY_LIMIT,
            "ftol": _LBFGS_FTOL,
            "gtol": _LBFGS_GTOL,
        },
    )
    if not monitor.done(s):
        monitor.finish(s, *_lbfgs_stop(res))
    # res.x is the best iterate L-BFGS-B saw; never worse than the start
    # except pathological line-search exits -- guard anyway.
    lp_best = -float(res.fun)
    if np.isfinite(lp_best) and lp_best >= lp0:
        return unflatten(res.x), lp0, lp_best, n_evals[0]
    return unchanged(), lp0, lp0, n_evals[0]


class _Engine:
    """Everything that differs between the two polish engines -- and nothing
    else.

    ``name`` ("lbfgs" / "de", what polish_raw_starts returns), ``label``
    (the per-seed line's "Seed polish (<label>)"), ``unit`` (what one
    monitor step is), ``default_cap`` (the per-round cap ENGINE_DEFAULT
    resolves to), the progress line's prefix, logger and opening text, and
    ``run(raw_starts, monitor) -> (polished, lp_bests, notes)``: advance
    every seed under the monitor and return the best points, their lps and
    a per-seed note for the per-seed line.  The rounds, the stopping rule,
    the progress line and the reasons are the driver's (PolishMonitor,
    polish_raw_starts, polish_rounds), the same for both.
    """

    def __init__(
        self,
        name,
        label,
        unit,
        default_cap,
        run,
        progress_label,
        log,
        opening_text="scoring the starting points",
    ):
        self.name = name
        self.label = label
        self.unit = unit
        self.default_cap = default_cap
        self.run = run
        self.progress_label = progress_label
        self.log = log
        self.opening_text = opening_text


def _lbfgs_engine(fn_lp_grad, keys, shapes, sizes):
    def run(raw_starts, monitor):
        polished, lp_bests, notes = [], [], []
        for s, center in enumerate(raw_starts):
            best, _lp0, lp_best, n_evals = _lbfgs_polish_one(
                center, fn_lp_grad, keys, shapes, sizes, monitor, s
            )
            polished.append(best)
            lp_bests.append(lp_best)
            notes.append(f" / {n_evals} evaluations")
        return polished, lp_bests, notes

    return _Engine(
        "lbfgs",
        "L-BFGS",
        "iterations",
        DEFAULT_LBFGS_POLISH_ITERATIONS,
        run,
        progress_label="Seed polish (L-BFGS)",
        log=logger,
    )


def _de_engine(
    model,
    keys,
    shapes,
    n_seeds,
    *,
    logp_fn,
    rng,
    cores,
    eval_timeout,
    asynchronous,
    adapt_gamma,
):
    """The PR #56 T=1 DE-MC polish (samplers.ptde.polish_seed_starts),
    jittering at one raw unit, on a worker pool when there are cores."""
    from .samplers import _common
    from .samplers import ptde as ptde_mod

    if logp_fn is None:
        logp_fn = model.compile_logp()
    if rng is None:
        rng = np.random.default_rng(0)
    scales = {k: np.ones(shp, dtype=float) for k, shp in zip(keys, shapes)}
    de_kwargs = {}
    if adapt_gamma is not _UNSET:
        de_kwargs["adapt_gamma"] = adapt_gamma

    def run(raw_starts, monitor):
        # The DE engine is a population method with nothing shared between
        # members within a sweep, so it parallelizes exactly as the sampler
        # does -- and until this was wired up it ran SERIAL while the job
        # held every core the sampler was about to use.  Same worker
        # contract as ptde_async: install the compiled logp in _common
        # BEFORE forking so children inherit it copy-on-write, then hand the
        # pool _common._eval_logp (module-level, so picklable by reference).
        n_proc = _resolve_polish_cores(cores, n_seeds)
        cap = "no" if monitor.cap is None else f"at most {monitor.cap}"
        pool = None
        if n_proc > 1:
            _common.set_worker_globals(logp_fn)
            pool = mp.Pool(processes=n_proc)
            logger.info(
                f"Seed polish: DE engine on {n_proc} worker process(es), "
                f"proposals pooled across all {n_seeds} seed(s), "
                f"{cap} sweeps per round."
            )
        else:
            # The serial case is ANNOUNCED, not silent.  "gradient graph
            # unavailable" reads as a note about capability; what it
            # actually means for the user is the expensive branch, and on
            # one core it is the whole wall clock of this stage (review
            # 6.11.3, examples/ob09020: 1 core of 36 for 38 minutes).  Since
            # cores=None means AUTO, reaching here at all means somebody
            # asked for serial -- or the machine has one core.
            logger.info(
                f"Seed polish: DE engine running SERIAL on one core "
                f"(cores={cores!r}), {n_seeds} seed(s), {cap} sweeps per "
                f"round; this gradient-free branch is far more expensive "
                f"than L-BFGS."
            )
        _common.warn_serial_eval_timeout(
            eval_timeout, pool, n_proc, "Seed polish", logger
        )

        def _recycle(dead):
            """Swap in a fresh pool after a logp call wedged a worker.

            THIS FUNCTION IS WHY THE POOL STAYS OURS.  polish_seed_starts
            cannot own the teardown -- it is handed `pool` and does not know
            how many workers to fork -- so it calls back here and we rebind
            the name the `finally` below tears down.  Without the rebind
            that `finally` would close the corpse and leak the live pool's
            workers for the rest of the process.
            """
            nonlocal pool
            pool = _common.recycle_pool(dead, n_proc)
            return pool

        try:
            polished, dlps = ptde_mod.polish_seed_starts(
                raw_starts,
                _common._eval_logp if pool is not None else logp_fn,
                rng,
                scales,
                monitor=monitor,
                pool=pool,
                eval_timeout=eval_timeout,
                pool_recycler=_recycle if pool is not None else None,
                asynchronous=asynchronous,
                **de_kwargs,
            )
        finally:
            if pool is not None:
                # terminate(), never close() + join(): a worker wedged in a
                # pathological logp never finishes its task, so close()
                # leaves it running and join() waits for it forever -- and a
                # recycled pool has SIGTERM-ignoring workers on top of that
                # (review 2.4.1).
                _common._shutdown_pool(pool)
        lp_bests = [monitor.lp0[s] + float(d) for s, d in enumerate(dlps)]
        return polished, lp_bests, [""] * len(polished)

    return _Engine(
        "de",
        "DE",
        "sweeps",
        DEFAULT_DE_POLISH_SWEEPS,
        run,
        progress_label="PTDE seed polish",
        log=ptde_mod.logger,
        opening_text="scoring the initial population",
    )


def polish_raw_starts(
    model,
    raw_starts,
    n_steps=ENGINE_DEFAULT,
    seed_indices=None,
    logp_fn=None,
    rng=None,
    tol=_UNSET,
    tol_window=_UNSET,
    cores=None,
    adapt_gamma=_UNSET,
    eval_timeout=None,
    asynchronous=True,
    engine=None,
    deadline=None,
    budget_s=None,
    stops_out=None,
):
    """ONE polish round: each raw start toward its own basin's optimum.

    Dispatch (the only engine-specific decision here): L-BFGS-B on
    logp+grad when the model's gradient graph builds and is finite at seed
    0; otherwise the PR #56 T=1 DE-MC polish (samplers/ptde.polish_seed_
    starts) with unit jitter scales (one raw unit = one preliminary
    whitening scale; DE's population self-adapts from there).  Everything
    else is the shared driver: one PolishMonitor for the round, one
    per-seed line with the stop reason, whichever engine ran.

    ``n_steps`` is the per-round CAP in the engine's steps (ENGINE_DEFAULT:
    the engine's own -- none for L-BFGS, DEFAULT_DE_POLISH_SWEEPS for DE).
    The RATE stop is ON by default on both engines: ``polish_tol_nats(D)``
    nats over the last POLISH_TOL_WINDOW steps; ``tol`` / ``tol_window``
    override it (``tol=None`` switches it off).  ``deadline`` is the
    absolute ``time.monotonic()`` instant at which the polish's wall-clock
    budget runs out (None: no budget), shared by every round of one polish
    (polish_rounds computes it from ``polish_timeout``); ``budget_s`` is
    that budget, for the log.  polish_rounds is the pipeline's loop; this is
    one round of it, and what outputs/ledger.py's hot-mode polish calls.

    ``engine`` (None = dispatch on the gradient, as above) may be ``"de"``
    or ``"lbfgs"`` to name it -- what polish_rounds passes for its later
    rounds, where round 1 has already decided.

    ``cores`` is the DE engine's worker grant; None means AUTO (the same
    rule a sampler uses when nothing names one), and ``cores=1`` is how a
    caller asks for serial.  The L-BFGS path ignores it -- it forks nothing.

    ``eval_timeout`` (seconds, default None = wait forever) is the DE
    engine's per-logp-call wall-clock budget, the same contract the PTDE
    samplers' ``sampler: eval_timeout:`` key carries: a call that exceeds it
    is abandoned and scored -inf, and the pool it wedged is recycled before
    the next batch.  It needs a pool (``cores > 1``), and the L-BFGS path
    ignores it -- scipy calls the gradient function in-process, where there
    is nothing to time out against.  It is a different thing from
    ``deadline``, which bounds the whole polish.  **run.py does not
    currently pass one**; see run.md for why that is a config-vocabulary
    decision rather than an oversight.

    ``asynchronous`` (default True) selects the DE engine's ptde_async-style
    loop on a real pool -- one proposal in flight per population member,
    results consumed in arrival order, so one slow VBM evaluation costs one
    worker and not the batch (review 2.4.14).  ``False`` restores the
    synchronous sweep-batch engine, which is bit-reproducible for a given
    rng and the only one the serial path runs.  See
    ``ptde.polish_seed_starts``.

    ``stops_out``, when a list, is extended with one ``(kind, reason)`` per
    seed (PolishMonitor.stop_kind and .describe) -- how polish_rounds learns
    a round ended on the timeout and how its wrap-up reports the reasons.

    Returns (polished_starts, dlps, method) with method in {"lbfgs", "de"}.
    A seed is never made worse: an engine result below the seed's own lp is
    discarded in favor of the seed.
    """
    if isinstance(raw_starts, dict):
        raw_starts = [raw_starts]
    if seed_indices is None:
        seed_indices = list(range(len(raw_starts)))
    keys = list(raw_starts[0].keys())
    shapes = [np.shape(raw_starts[0][k]) for k in keys]
    sizes = [int(np.asarray(raw_starts[0][k]).size) for k in keys]

    if engine not in (None, "de", "lbfgs"):
        raise ValueError(
            f"polish_raw_starts: engine={engine!r}; expected None (dispatch "
            f"on the gradient), 'lbfgs' or 'de'"
        )

    fn_lp_grad = None if engine == "de" else _compile_logp_grad(model)
    if fn_lp_grad is not None:
        vals = fn_lp_grad(raw_starts[0])
        grad_finite = np.all(
            [np.all(np.isfinite(np.asarray(v))) for v in vals]
        )
        if not grad_finite:
            logger.info(
                "Seed polish: gradient non-finite at the start; falling "
                "back to the gradient-free DE polish."
            )
            fn_lp_grad = None
    if engine == "lbfgs" and fn_lp_grad is None:
        raise RuntimeError(
            "polish_raw_starts: engine='lbfgs' was requested (a later round "
            "of an L-BFGS polish) but the gradient is unavailable or "
            "non-finite at this round's seed 0; the rounds cannot switch "
            "engines midway"
        )

    if fn_lp_grad is not None:
        eng = _lbfgs_engine(fn_lp_grad, keys, shapes, sizes)
    else:
        eng = _de_engine(
            model,
            keys,
            shapes,
            len(raw_starts),
            logp_fn=logp_fn,
            rng=rng,
            cores=cores,
            eval_timeout=eval_timeout,
            asynchronous=asynchronous,
            adapt_gamma=adapt_gamma,
        )

    # --- the shared driver, from here on ---
    cap = eng.default_cap if n_steps is ENGINE_DEFAULT else int(n_steps)
    n_params = int(sum(sizes))
    monitor = PolishMonitor(
        len(raw_starts),
        unit=eng.unit,
        cap=cap,
        tol=polish_tol_nats(n_params) if tol is _UNSET else tol,
        window=POLISH_TOL_WINDOW if tol_window is _UNSET else tol_window,
        label=eng.progress_label,
        deadline=deadline,
        budget_s=budget_s,
        log=eng.log,
        opening_text=eng.opening_text,
    )
    polished, lp_bests, notes = eng.run(raw_starts, monitor)
    dlps = []
    for s, (lp_best, note) in enumerate(zip(lp_bests, notes)):
        lp0 = monitor.lp0[s]
        dlps.append(lp_best - lp0)
        logger.info(
            f"Seed polish ({eng.label}): seed {seed_indices[s]} lp "
            f"{lp0:.1f} -> {lp_best:.1f} (dlp=+{lp_best - lp0:.1f}, "
            f"{monitor.steps[s]} {eng.unit}{note}, {monitor.describe(s)})"
        )
    if stops_out is not None:
        stops_out.extend(
            (monitor.stop_kind[s], monitor.describe(s))
            for s in range(len(polished))
        )
    return polished, dlps, eng.name


def _seed_physical(lookup, raw):
    """{raw key: full physical element vector} for one raw start dict, under
    the CURRENT whitening (Parameter.phys_from_raw)."""
    from .system import _parameter_for_raw_key

    return {
        key: np.asarray(
            _parameter_for_raw_key(key, lookup).phys_from_raw(vec),
            dtype=float,
        )
        for key, vec in raw.items()
    }


def _seed_raw(lookup, phys, template):
    """Inverse of _seed_physical under the CURRENT whitening: the raw start
    dict (shaped like ``template``) of a physical point."""
    from .system import _parameter_for_raw_key

    return {
        key: np.asarray(
            _parameter_for_raw_key(key, lookup).raw_from_initval(phys[key]),
            dtype=float,
        ).reshape(np.shape(template[key]))
        for key in template
    }


def _raw_distance(a, b):
    return float(
        np.sqrt(
            sum(np.sum((np.asarray(a[k]) - np.asarray(b[k])) ** 2) for k in a)
        )
    )


def _warn_if_seeds_converged(system, model, lookup, origins_phys, r):
    """Across-round basin-coverage check for a multi-seed set.

    Each round's trust region (ptde.polish_seed_starts) is built from that
    round's own starts, so it guarantees per round that no seed crosses the
    midpoint toward a neighbour -- but across rounds the midpoints move,
    and the re-whitening rescales every axis by a different factor, so a
    region carried over from round 1 is not a region in round 2's
    coordinates at all.  Measured: carrying round 1's region forward
    (centred on the ORIGINAL seeds, half the ORIGINAL separation, re-expressed
    in the re-whitened coordinates) put DC2018_128's polished seeds OUTSIDE
    their own regions, and round 2 refused 6030 / 6795 Metropolis-accepted
    proposals -- all of them.  So the protection stays per round, and this
    check makes the cumulative drift VISIBLE instead: it compares the seeds'
    separation now with their original separation, both measured in the
    current coordinates, and warns past the same 1/4 the per-round check
    uses.
    """
    raws, idx = system.get_raw_starts(model)
    orig = [_seed_raw(lookup, ph, raws[0]) for ph in origins_phys]
    for a in range(len(raws)):
        for b in range(a + 1, len(raws)):
            d0 = _raw_distance(orig[a], orig[b])
            d1 = _raw_distance(raws[a], raws[b])
            if d0 > 0 and d1 < 0.25 * d0:
                logger.warning(
                    f"Seed polish: after round {r}, seeds {idx[a]} and "
                    f"{idx[b]} are {d1:.1f} scale units apart against "
                    f"{d0:.1f} originally (both measured in the current "
                    f"coordinates) -- the rounds have walked them toward "
                    f"one basin; the sampler may start these chains in what "
                    f"is effectively one basin."
                )


def polish_rounds(
    system,
    model,
    n_steps=ENGINE_DEFAULT,
    *,
    cores,
    rewhiten,
    max_rounds=POLISH_MAX_ROUNDS,
    tol=None,
    rng=None,
    timeout_s=DEFAULT_POLISH_TIMEOUT_S,
):
    """The pipeline's seed polish: polish, re-whiten, polish again, until a
    round stops paying (review 2.4.14, JDE ruling 2026-09-14 part 3) -- on
    BOTH engines, through this one loop (JDE 2026-10-07).

    Round 1 is one polish_raw_starts over the starts get_raw_starts builds,
    adopted with apply_polished_starts.  A round ends for reasons that are
    not all statements about the surface -- a DE population freezes (the
    fixed 2.38/sqrt(2D) step accepts ~0.3% at T=1, and 80-90% of sweeps
    accept nothing after ~1000 on DC2018_128); an L-BFGS run crawls along a
    direction the preliminary whitening scaled badly (ob140939: 2-7 nats
    still to gain after 400 iterations, gained over another 13000-29000) --
    and, in round 1, every step is taken in PRELIMINARY whitening units,
    because the probe has not measured anything yet.  JDE: "The probe
    measures scales AT the start and DE proposals are drawn in those units,
    so a bad start hobbles the optimizer meant to escape it" -- and an
    L-BFGS quasi-Newton model built in badly scaled units is hobbled the
    same way.  Each later round therefore:

      1. re-centers the whitening anchor on the polished start
         (recenter_whitening_anchor) and re-whitens there
         (whitening.measure_and_whiten -- the same probe run.py runs next,
         in the same order; skipped when ``rewhiten`` is False, i.e.
         `measure_scales: false`, which asks for no probe);
      2. re-derives every seed in the new raw coordinates (get_raw_starts);
      3. polishes EVERY seed again with the same engine (a FRESH DE
         population jittered at one, now measured, unit; a fresh L-BFGS
         run).  Every seed, not only the ones still climbing: the DE
         multi-seed trust region is built from the seeds in the call, and a
         lone seed would get an infinite one;
      4. adopts the result, as round 1 did.

    The anchor is NOT re-centered after the last round: run.py does that
    next, unconditionally, on every path, and owns it.

    Stopping: when no seed's round gained ``tol`` nats (default
    polish_tol_nats(D)), after ``max_rounds`` rounds, or when the wall-clock
    budget ``timeout_s`` (seconds for the WHOLE polish, every round; None:
    no budget) runs out -- that stops the running round's seeds where they
    are ("stopped on timeout") and starts no further round.  When
    re-whitening is on, round 2 always runs: round 1's gain in preliminary
    units is not evidence about the measured surface (on the 30-event
    static DC2018 sweep the 400-sweep polish bought EXACTLY 0.0 nats on
    seed 0 in 26/30 events).  Gains are comparable across rounds because
    each round's is measured from that round's own start, AFTER the
    re-whitening: the whitening is a reparameterization (whitening.md), and
    the only thing the probe moves in logp is the soft-bound barrier
    steepness, which is therefore never counted as a polish gain.

    The final whitening is NOT measured here: run.py's prepare_whitening
    probes once more at the final point and persists it, exactly as on
    every other path.

    ``n_steps`` is the per-ROUND cap (resolve_polish_steps' value); ``cores``
    the DE engine's worker grant.  Returns a summary dict: ``method``,
    ``rounds`` (one {seed index: gain} dict per round), ``reasons`` (the
    last round's {seed index: stop reason}), ``stop`` ("converged", "round
    cap" or "timeout") and ``wall_s``.
    """
    from .whitening import measure_and_whiten

    t0 = time.monotonic()
    deadline = None if timeout_s is None else t0 + float(timeout_s)
    raw_starts, seed_indices = system.get_raw_starts(model)
    logp_fn = model.compile_logp()
    if rng is None:
        rng = np.random.default_rng(0)
    n_params = int(sum(np.asarray(v).size for v in raw_starts[0].values()))
    if tol is None:
        tol = polish_tol_nats(n_params)
    lookup = {p.label: p for p in system.get_all_parameters()}
    # The ORIGINAL seeds, physically, for the across-round coverage check.
    origins_phys = (
        [_seed_physical(lookup, r) for r in raw_starts]
        if len(raw_starts) > 1
        else None
    )

    def _one_round(starts, engine):
        stops = []
        polished, dlps, method = polish_raw_starts(
            model,
            starts,
            n_steps=n_steps,
            seed_indices=seed_indices,
            logp_fn=logp_fn,
            rng=rng,
            cores=cores,
            engine=engine,
            deadline=deadline,
            budget_s=timeout_s,
            stops_out=stops,
        )
        if len(stops) != len(seed_indices):
            raise RuntimeError(
                f"polish_rounds: the round reported {len(stops)} stop "
                f"reason(s) for {len(seed_indices)} seed(s); "
                f"polish_raw_starts fills stops_out with one per seed"
            )
        system.apply_polished_starts(polished, seed_indices)
        rounds.append(dict(zip(seed_indices, (float(d) for d in dlps))))
        kinds.clear()
        kinds.update((k, kind) for k, (kind, _txt) in zip(seed_indices, stops))
        reasons.clear()
        reasons.update(
            (k, txt) for k, (_kind, txt) in zip(seed_indices, stops)
        )
        return method

    rounds, reasons, kinds = [], {}, {}
    method = _one_round(raw_starts, None)

    def _timed_out():
        return deadline is not None and time.monotonic() >= deadline

    def _wants_another():
        if rewhiten and len(rounds) == 1:
            return True
        return any(d >= tol for d in rounds[-1].values())

    def _round_hit_the_clock():
        return any(k == STOP_TIMEOUT for k in kinds.values())

    while (
        _wants_another()
        and len(rounds) < int(max_rounds)
        and not _round_hit_the_clock()
        and not _timed_out()
    ):
        r = len(rounds) + 1
        forced = rewhiten and r == 2
        climbing = [k for k, d in rounds[-1].items() if d >= tol]
        why = (
            "round 2 always runs when re-whitening: round 1 stepped in "
            "preliminary units"
            if forced
            else f"seed(s) {climbing} gained >= {tol:.3g} nats last round"
        )
        how = (
            "re-whitening at the polished point"
            if rewhiten
            else "restarting at the polished point (measure_scales is off, "
            "so no re-whitening)"
        )
        logger.info(f"Seed polish round {r}/{int(max_rounds)}: {how}; {why}.")
        # Re-center first, exactly as run.py orders it before its own probe:
        # the probe measures around the anchor, which must be the start.
        system.recenter_whitening_anchor(model)
        if rewhiten:
            measure_and_whiten(
                system, model, system.get_raw_start(model), logp_fn
            )
        raw_all, idx_all = system.get_raw_starts(model)
        if list(idx_all) != list(seed_indices):
            raise RuntimeError(
                f"polish_rounds: round {r} re-derived seed indices "
                f"{list(idx_all)} where round 1 had {list(seed_indices)}; "
                f"apply_polished_starts writes every seed back, so the set "
                f"cannot change between rounds"
            )
        _one_round(raw_all, method)
        if origins_phys is not None:
            _warn_if_seeds_converged(system, model, lookup, origins_phys, r)

    # A round the clock cut short is a timeout whatever it gained: its gain
    # is not a statement that the next round would have gained less.
    if _round_hit_the_clock() or (_wants_another() and _timed_out()):
        stop = "timeout"
    elif not _wants_another():
        stop = "converged"
    else:
        stop = "round cap"
    wall_s = time.monotonic() - t0
    summary = {
        "method": method,
        "rounds": rounds,
        "reasons": dict(reasons),
        "stop": stop,
        "wall_s": wall_s,
    }
    per_seed = "; ".join(
        f"seed {k}: +{sum(rd[k] for rd in rounds):.1f} ("
        + " + ".join(f"{rd[k]:.1f}" for rd in rounds)
        + f"), last round {reasons[k]}"
        for k in seed_indices
    )
    if stop == "converged":
        verdict = (
            f"Stopped: the last round gained < {tol:.3g} nats on every seed."
        )
    elif stop == "timeout":
        verdict = (
            f"Stopped on timeout: the polish's {float(timeout_s):.0f} s "
            f"wall-clock budget (sampler: polish_timeout) ran out before "
            f"the rounds converged -- the start may still be below its "
            f"basin optimum."
        )
    else:
        verdict = (
            f"Stopped at the {int(max_rounds)}-round cap with a seed still "
            f"gaining >= {tol:.3g} nats per round -- the start may still be "
            f"below its basin optimum."
        )
    engine_label = "L-BFGS" if method == "lbfgs" else "DE"
    logger.info(
        f"Seed polish: {len(rounds)} {engine_label} round(s) in {wall_s:.0f} "
        f"s; gain per seed (per round): {per_seed}.  {verdict}"
    )
    return summary


def _resolve_polish_cores(cores, n_seeds):
    """Worker count for the DE polish.

    ``cores=None`` means AUTO -- the same grant a sampler takes when nothing
    names one (_common.default_cores) -- and NOT serial.  It used to mean
    serial, which is how review 6.11.3 happened: outputs/ledger.py's
    hot-mode polish passed nothing, so the one branch every gradient-free
    microlensing fit takes ran on 1 core of 36 for 38 minutes while the rest
    of the machine the sampler had just filled sat idle.  Nothing about this
    stage wants fewer cores than the sampling either side of it, so the
    default now says so, and a caller that wants serial passes ``cores=1``.

    Capped at the batch size (n_seeds * pop_size) upstream by there simply
    being nothing more to hand out; here we only guard against asking for
    more processes than the machine has, and against the degenerate 1.
    """
    from .samplers._common import default_cores

    if cores is None:
        return default_cores()
    try:
        n = int(cores)
    except (TypeError, ValueError):
        # Unreadable value: say so and take the auto grant.  Silently
        # dropping to one core is what made the original bug invisible.
        #
        # The message says the same two things run.resolve_cores_setting's
        # refusal says -- an absent/None cores IS the automatic grant, and
        # cores=1 is how to ask for serial -- because the two used to
        # disagree about a value neither could use (review 5.3.3e).  Only
        # the OUTCOME differs, and positionally: run.py can still refuse the
        # run, while this can be reached from a wrap-up stage that must not
        # kill a finished fit.
        logger.warning(
            f"Seed polish: cores={cores!r} is not a number of cores; using "
            f"the automatic grant instead (which is also what an absent "
            f"cores takes; cores=1 is serial)."
        )
        return default_cores()
    # `cores <= 0` is the automatic grant, the same as None (review 2.4.8).
    # It used to be swept into the `n <= 1` serial arm, while the same 0 was
    # serial in create_pool and AUTO in nested.py.  run.py warns about it at
    # the parse boundary, where the user's own spelling is still in hand; by
    # the time it reaches this stage the only thing left to do is agree with
    # the other two resolvers.  ONE is still serial -- that is the statement
    # a caller makes when they mean it.
    if n <= 0:
        return default_cores()
    if n == 1:
        return 1
    return max(1, min(n, mp.cpu_count()))
