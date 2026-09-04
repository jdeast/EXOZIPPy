"""Blind global period searches, and the channel that turns one into a start.

Two searches live here -- Box Least Squares for transit photometry
(:func:`bls_search`) and Lomb-Scargle for radial velocities
(:func:`lombscargle_search`) -- plus the small amount of shared plumbing that
turns a detection into a start value the relaxation engine will honor.  The
point of the feature is that a `params.yaml` becomes OPTIONAL for the simple
blind case: point EXOZIPPy at a light curve (or a set of RVs) of a single
unknown planet and the period and conjunction epoch are measured from the
data rather than typed in.

Both implementations are ``astropy.timeseries``' (``BoxLeastSquares``,
``LombScargle``).  astropy is already a hard dependency and these are the
reference implementations of both algorithms; nothing here reimplements the
periodogram, only the grid limits, the detection thresholds and the
translation into EXOZIPPy parameters.

The pattern copied end to end is MMEXOFAST's (``mulensing/
mmexofast_support.py``), which solves the same problem for microlensing:

- **Ask whether seeding is needed by DERIVABILITY, not by presence.**
  ``ConfigManager.probe_derivable`` runs the relaxation engine on a snapshot
  and rolls every mutation back; a rank above ``PRECEDENCE_DEFAULT`` means someone
  really said so.  A literal-key scan is wrong for the same reason it was
  wrong there: ``orbit.period`` is a DERIVED parameter (``10**logP``), so a
  restart file written by ``mkparam`` never names it, and a scan would re-run
  the search on every second-iteration fit.  A literal scan is still used as
  a SHORT-CIRCUIT (naming a value outright makes it PRECEDENCE_USER, which is
  derivable by definition), so the ordinary hand-written params file pays
  nothing for the probe.
- **A search that finds nothing convincing seeds nothing.**  Both functions
  return ``None`` below their thresholds and say in a warning what they
  looked at and by how much it missed.  A wrong confident seed is worse than
  no seed: ``orbit.tc``'s hard window is ``tc +/- P/2``, so a bad epoch does
  not merely start the chain in the wrong place, it makes the right place
  unreachable.

Stage placement: **both the measurement and the push happen in stage 1a**
(``load_data``), not the stage-1a/stage-2 split ``Transit._hint_baseline``
uses.  That is not a style choice.  ``Orbit.register_parameters`` (stage 2)
builds ``tc``'s HARD bounds as ``tc_init +/- P/2`` from the start values it
can see, so a seed pushed from another component's stage 2 would race
against orbit's -- ``System.prepare`` walks ``active_components`` in config
key order -- and if orbit went first the searched epoch would land outside a
window built around the defaults.yaml 2460000, which ``Parameter.build_pymc``
correctly treats as fatal.  Every component's stage 1a precedes every
component's stage 2, so stage 1a is the only placement that is right for
both orderings.  MMEXOFAST pushes at stage 1a for the same class of reason
(its own flux bootstrap, later in the same ``load_data``).

The seed goes through ONE channel (:func:`seed_start`):
``ConfigManager.add_hint`` at ``PRECEDENCE_DERIVED_DATA``.  That is the ranked
start value -- what the relaxation engine solves from, what the provenance
ledger, ``initval_source`` ("data") and ``export_solution`` report, and what
every user entry outranks -- and since review 3.14.3 ``ConfigManager.resolve()``
layers ``self.hints`` too (under the user's params, over defaults), so the
stage-2 readers that ask ``resolve()`` for a start value (``Orbit``'s ``tc``
window, ``Orbit._seeded_period``) see it directly.  Until then they could
not, and this module wrote every seed a SECOND time through
``add_override(path, initval=...)`` purely to be visible to them.  That
duplicate is gone: one number, one channel.  Do not reintroduce it -- an
override carries no rank, so a second write is invisible to the ledger and
can only ever drift from the hint it shadows.

Deliberately NOT seeded, and each for a reason worth not rediscovering:

- **a/R\\* from the transit duration.**  ``planet.ar`` carries ``rank: 5`` in
  its defaults.yaml precisely so the relaxation engine's Condition B always
  rewrites IT rather than the period, mass or radius it is derived from, so a
  hint there is discarded by design.  Forcing it through would invert exactly
  the direction the ranking system exists to protect (a duration-derived
  a/R\\* silently moving a stellar mass).  The duration is reported in the
  log and by the CLI utility instead.
- **K from the Lomb-Scargle sinusoid.**  ``planet.K`` already gets a
  data-driven hint, from ``RVInstrument._estimate_k_init``, and for a single
  circular signal ``sqrt(2) * std`` IS the semi-amplitude exactly.  Rather
  than add a second competing hint on the same parameter (planet's is pushed
  at stage 2 and would overwrite a stage-1a one regardless), the LS fit's
  amplitude REPLACES ``self.k_init`` at the source, so it flows through the
  existing, tested channel.  The gain is confined to the low-SNR case, where
  ``sqrt(2) * std`` counts the noise variance as signal.
- **astrometry.**  See ``AstrometryInstrument.get_utilities``.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..config import PRECEDENCE_DERIVED_DATA

logger = logging.getLogger(__name__)


# --- detection thresholds -----------------------------------------------------
#
# Deliberately conservative.  The failure this feature must not have is a
# confident wrong seed, and the cost of a miss is only that the user writes
# the number down themselves, which is the status quo.

# Box Least Squares.  SDE (signal detection efficiency) is the classic BLS
# statistic (Kovacs+2002): (peak power - mean power) / std(power) over the
# period grid.  A strong peak inflates the std, which lowers the SDE, so this
# errs conservative by construction.  7 is the customary floor.
MIN_BLS_SDE = 7.0
# astropy reports depth / depth_err at the peak; a box deeper than 7.5 sigma
# is the second, independent leg of the test (SDE alone can be large for a
# shallow but very regular systematic).
MIN_BLS_DEPTH_SNR = 7.5
# Below this a box search has nothing to work with.
MIN_BLS_POINTS = 50

# Lomb-Scargle.  The Baluev (2008) analytic false-alarm probability, as
# implemented by astropy; 1e-3 is the conventional RV detection threshold.
MAX_LS_FAP = 1e-3
MIN_LS_POINTS = 10

# Which search's answer wins when two components seed the same orbit.  A
# transit epoch and period are better than an RV one by orders of magnitude
# (a box edge is minutes wide; an RV conjunction is a phase fit), so transit
# outranks RV -- as a QUALITY here rather than as a provenance rank, because
# both really are PRECEDENCE_DERIVED_DATA and config._provenance_label reports any
# other value as "solved".  add_hint is last-writer-wins and the components
# run in config key order, so the tie has to be broken explicitly.
QUALITY_TRANSIT = 20
QUALITY_RV = 10


@dataclass
class PeriodicSignal:
    """One accepted detection, in the data's own time system and units.

    ``epoch`` is a time of conjunction (mid-transit for BLS; the descending
    zero crossing of the RV curve for Lomb-Scargle, which is the same
    quantity -- see :func:`_conjunction_from_sinusoid`), folded to the cycle
    nearest the middle of the data.
    """

    kind: str
    period: float
    epoch: float
    significance: float
    n_points: int
    depth: Optional[float] = None
    duration: Optional[float] = None
    amplitude: Optional[float] = None
    detail: dict = field(default_factory=dict)

    def summary(self):
        parts = [f"P = {self.period:.6g} d", f"T_C = {self.epoch:.6f}"]
        if self.depth is not None:
            parts.append(f"depth = {self.depth:.4g}")
        if self.duration is not None:
            parts.append(f"duration = {self.duration * 24.0:.3g} h")
        if self.amplitude is not None:
            parts.append(f"K = {self.amplitude:.4g}")
        parts.append(
            f"{self.detail.get('stat', 'significance')} = "
            f"{self.significance:.3g}"
        )
        return ", ".join(parts)


def fold_epoch(epoch, period, t_ref):
    """Move ``epoch`` by whole periods to the cycle nearest ``t_ref``.

    An epoch and an epoch plus a period are the same solution, but they are
    not the same START: the further the seed sits from the data, the more a
    period error leverages into a phase error, and ``orbit.tc``'s hard window
    is only one period wide.  Both searches therefore report the conjunction
    nearest the middle of the observations.
    """
    if not np.isfinite(period) or period <= 0:
        return epoch
    return float(epoch + period * np.round((t_ref - epoch) / period))


# --- Box Least Squares --------------------------------------------------------


def bls_search(
    time,
    flux,
    err=None,
    minimum_period=None,
    maximum_period=None,
    durations=None,
    minimum_n_transit=2,
    frequency_factor=1.0,
    min_sde=MIN_BLS_SDE,
    min_depth_snr=MIN_BLS_DEPTH_SNR,
    context="transit",
):
    """Box Least Squares transit search (Kovacs, Zucker & Mazeh 2002).

    ``flux`` is relative flux (out-of-transit ~ 1); the caller normalizes.
    Returns a :class:`PeriodicSignal` (``depth`` fractional, ``duration`` in
    days) or ``None`` when nothing clears both thresholds.

    ``minimum_n_transit`` defaults to 2 rather than astropy's 3: a single
    TESS sector holding two transits of a ~12 d planet is exactly the blind
    case this exists for, and the SDE and depth-SNR thresholds -- not the
    grid -- are what reject a spurious pair.
    """
    from astropy.timeseries import BoxLeastSquares

    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    good = np.isfinite(time) & np.isfinite(flux)
    if err is not None:
        err = np.asarray(err, dtype=float)
        good &= np.isfinite(err) & (err > 0)
    time, flux = time[good], flux[good]
    err = err[good] if err is not None else None

    if time.size < MIN_BLS_POINTS:
        logger.warning(
            "[%s] BLS skipped: %d usable photometric points, fewer than the "
            "%d a box search needs.",
            context,
            time.size,
            MIN_BLS_POINTS,
        )
        return None

    baseline = float(time.max() - time.min())
    if not np.isfinite(baseline) or baseline <= 0:
        logger.warning(
            "[%s] BLS skipped: the times span no baseline.", context
        )
        return None

    if durations is None:
        # 1 h to 12 h, the range essentially every transiting planet's T14
        # falls in.  A duration grid that misses the true one costs depth
        # and SDE, not correctness.
        durations = np.array([1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]) / 24.0
    durations = np.asarray(durations, dtype=float)
    durations = durations[durations < 0.5 * baseline]
    if durations.size == 0:
        logger.warning(
            "[%s] BLS skipped: the %.4g d baseline is shorter than the "
            "shortest trial transit duration.",
            context,
            baseline,
        )
        return None

    model = BoxLeastSquares(time, flux, dy=err)
    try:
        periods = model.autoperiod(
            durations,
            minimum_period=minimum_period,
            maximum_period=maximum_period,
            minimum_n_transit=minimum_n_transit,
            frequency_factor=frequency_factor,
        )
    except ValueError as exc:
        logger.warning(
            "[%s] BLS skipped: astropy could not build a period grid over a "
            "%.4g d baseline (%s).",
            context,
            baseline,
            exc,
        )
        return None
    if np.size(periods) < 3:
        logger.warning(
            "[%s] BLS skipped: the period grid holds only %d trial period(s) "
            "over a %.4g d baseline.",
            context,
            int(np.size(periods)),
            baseline,
        )
        return None

    result = model.power(periods, durations)
    power = np.asarray(result.power, dtype=float)
    finite = np.isfinite(power)
    if not finite.any():
        logger.warning(
            "[%s] BLS skipped: the periodogram is all NaN.", context
        )
        return None

    i = int(np.nanargmax(power))
    spread = float(np.std(power[finite]))
    sde = (
        (power[i] - float(np.mean(power[finite]))) / spread
        if spread > 0
        else np.inf
    )

    depth = float(np.asarray(result.depth)[i])
    depth_err = float(np.asarray(result.depth_err)[i])
    depth_snr = depth / depth_err if depth_err > 0 else np.inf
    period = float(np.asarray(result.period)[i])
    duration = float(np.asarray(result.duration)[i])
    epoch = fold_epoch(
        float(np.asarray(result.transit_time)[i]),
        period,
        float(np.median(time)),
    )

    if depth <= 0 or sde < min_sde or depth_snr < min_depth_snr:
        logger.warning(
            "[%s] BLS found no convincing transit: best period %.6g d has "
            "SDE = %.3g (need %.3g) and depth = %.4g +/- %.4g "
            "(SNR %.3g, need %.3g), over %d points spanning %.4g d and "
            "%d trial periods. No start values were seeded from the "
            "photometry.",
            context,
            period,
            sde,
            min_sde,
            depth,
            depth_err,
            depth_snr,
            min_depth_snr,
            time.size,
            baseline,
            int(np.size(periods)),
        )
        return None

    signal = PeriodicSignal(
        kind="bls",
        period=period,
        epoch=epoch,
        significance=float(sde),
        n_points=int(time.size),
        depth=depth,
        duration=duration,
        detail={
            "stat": "SDE",
            "depth_snr": float(depth_snr),
            "depth_err": depth_err,
            "baseline": baseline,
            "n_periods": int(np.size(periods)),
        },
    )
    logger.info("[%s] BLS detection: %s", context, signal.summary())
    return signal


# --- Lomb-Scargle -------------------------------------------------------------


def _conjunction_from_sinusoid(a_cos, b_sin, period):
    """Time of conjunction of a circular orbit from its RV Fourier pair.

    EXOZIPPy's own convention, read straight off ``Orbit.get_radial_velocity``
    and ``physics.calc_tp``: with ``e = 0`` the true anomaly at conjunction is
    ``f = pi/2 - omega``, so ``tp = tc`` shifted by exactly that, and

        RV(t) = K (cos w cos f - sin w sin f) = -K sin(2 pi (t - tc) / P),

    independent of omega -- the star recedes before conjunction and
    approaches after it, which is what makes this epoch the same T_C a
    transit measures.  Matching that to ``a cos(2 pi t/P) + b sin(2 pi t/P)``
    gives ``a = K sin(phi)``, ``b = -K cos(phi)`` with ``phi = 2 pi tc / P``,
    hence the arctan2 below.  A sign error here would be a half-period phase
    error, i.e. a confidently wrong seed, so this is pinned by test.
    """
    phi = np.arctan2(a_cos, -b_sin)
    return float(period * phi / (2.0 * np.pi))


def _sinusoid_fit(time, y, err, period, inst_map=None):
    """Weighted least squares of one sinusoid plus per-instrument offsets.

    Returns ``(amplitude, epoch)`` with the epoch as an absolute time (not
    yet folded).  The offsets are fitted rather than assumed because the
    per-instrument means the caller removed are only the optimum when the
    signal averages to zero over each instrument's own sampling.
    """
    omega = 2.0 * np.pi / period
    cols = [np.cos(omega * time), np.sin(omega * time)]
    if inst_map is None:
        cols.append(np.ones_like(time))
    else:
        for k in np.unique(inst_map):
            cols.append((inst_map == k).astype(float))
    design = np.column_stack(cols)
    w = 1.0 / np.asarray(err, dtype=float) if err is not None else None
    if w is not None:
        design = design * w[:, None]
        target = y * w
    else:
        target = y
    theta, *_ = np.linalg.lstsq(design, target, rcond=None)
    a_cos, b_sin = float(theta[0]), float(theta[1])
    amplitude = float(np.hypot(a_cos, b_sin))
    epoch = _conjunction_from_sinusoid(a_cos, b_sin, period)
    return amplitude, epoch


def lombscargle_search(
    time,
    y,
    err=None,
    inst_map=None,
    minimum_period=0.5,
    maximum_period=None,
    samples_per_peak=10,
    max_fap=MAX_LS_FAP,
    context="rvinstrument",
):
    """Lomb-Scargle periodogram search (Lomb 1976; Scargle 1982).

    ``y`` is the velocity with each instrument's own offset already removed
    (feeding raw multi-instrument RVs to a periodogram buries the signal
    under the offsets).  Returns a :class:`PeriodicSignal` whose
    ``amplitude`` is the semi-amplitude of the best-fit circular orbit and
    whose ``epoch`` is its time of conjunction, or ``None``.

    The detection test is astropy's Baluev analytic false-alarm probability
    at the peak, over the frequency grid actually searched.
    """
    from astropy.timeseries import LombScargle

    time = np.asarray(time, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(time) & np.isfinite(y)
    if err is not None:
        err = np.asarray(err, dtype=float)
        good &= np.isfinite(err) & (err > 0)
    time, y = time[good], y[good]
    err = err[good] if err is not None else None
    inst_map = np.asarray(inst_map)[good] if inst_map is not None else None

    if time.size < MIN_LS_POINTS:
        logger.warning(
            "[%s] Lomb-Scargle skipped: %d usable velocities, fewer than the "
            "%d a periodogram needs.",
            context,
            time.size,
            MIN_LS_POINTS,
        )
        return None

    baseline = float(time.max() - time.min())
    if maximum_period is None:
        maximum_period = baseline
    if not np.isfinite(baseline) or maximum_period <= minimum_period:
        logger.warning(
            "[%s] Lomb-Scargle skipped: the %.4g d baseline admits no period "
            "range above the %.4g d floor.",
            context,
            baseline,
            minimum_period,
        )
        return None

    min_freq, max_freq = 1.0 / maximum_period, 1.0 / minimum_period
    ls = LombScargle(time, y, dy=err)
    frequency, power = ls.autopower(
        minimum_frequency=min_freq,
        maximum_frequency=max_freq,
        samples_per_peak=samples_per_peak,
    )
    if power.size == 0 or not np.isfinite(power).any():
        logger.warning(
            "[%s] Lomb-Scargle skipped: the periodogram is empty or all NaN.",
            context,
        )
        return None

    i = int(np.nanargmax(power))
    peak_power = float(power[i])
    peak_frequency = float(frequency[i])

    # Refine the peak on a dense local grid.  autopower's spacing is
    # 1/(samples_per_peak * baseline), so the grid alone limits the period
    # to a relative accuracy of P/(samples_per_peak * baseline) -- 4e-3 for
    # a 12 d period over 300 d, which accumulates to a tenth of a cycle of
    # phase error across the data and is a poor place to start a fit.  Two
    # grid steps either side, 200 samples, costs one extra periodogram of
    # 201 frequencies.
    if frequency.size > 2:
        step = float(frequency[1] - frequency[0])
        fine = np.linspace(
            max(peak_frequency - 2.0 * step, min_freq),
            min(peak_frequency + 2.0 * step, max_freq),
            201,
        )
        fine_power = ls.power(fine)
        j = int(np.nanargmax(fine_power))
        if np.isfinite(fine_power[j]) and fine_power[j] >= peak_power:
            peak_frequency, peak_power = float(fine[j]), float(fine_power[j])

    period = float(1.0 / peak_frequency)
    try:
        fap = float(
            ls.false_alarm_probability(
                # Baluev's expression evaluates (1 - z)**(N/2), so a peak
                # power that rounds to (or a hair above) 1 -- a noiseless
                # simulated curve does exactly that -- gives NaN or a
                # negative base.  A perfect fit's false-alarm probability is
                # zero, not unknown, so clip rather than reject.
                min(peak_power, 1.0 - 1e-12),
                method="baluev",
                minimum_frequency=min_freq,
                maximum_frequency=max_freq,
            )
        )
    except Exception as exc:  # astropy raises for some normalizations
        logger.warning(
            "[%s] Lomb-Scargle skipped: astropy could not evaluate a "
            "false-alarm probability (%s), and a periodogram peak with no "
            "significance is not a detection.",
            context,
            exc,
        )
        return None

    if not np.isfinite(fap) or fap > max_fap:
        logger.warning(
            "[%s] Lomb-Scargle found no convincing periodicity: the best "
            "period %.6g d has power %.4g, false-alarm probability %.3g "
            "(need < %.3g), over %d velocities spanning %.4g d. No start "
            "values were seeded from the RVs.",
            context,
            period,
            peak_power,
            fap,
            max_fap,
            time.size,
            baseline,
        )
        return None

    amplitude, epoch = _sinusoid_fit(time, y, err, period, inst_map)
    epoch = fold_epoch(epoch, period, float(np.median(time)))

    signal = PeriodicSignal(
        kind="lombscargle",
        period=period,
        epoch=epoch,
        # -log10(FAP): a "sigma-like" number that grows with confidence, so
        # the same field means the same direction as the BLS SDE.
        significance=float(-np.log10(max(fap, 1e-300))),
        n_points=int(time.size),
        amplitude=amplitude,
        detail={
            "stat": "-log10(FAP)",
            "fap": fap,
            "power": peak_power,
            "baseline": baseline,
        },
    )
    logger.info("[%s] Lomb-Scargle detection: %s", context, signal.summary())
    return signal


# --- turning a detection into a start value -----------------------------------


def _seed_registry(config_manager):
    """Per-ConfigManager record of which search already seeded which path.

    An attribute rather than a ConfigManager field so this feature needs no
    edit to ``config.py``; ``System`` sets ``config_manager.param_file`` the
    same way.  Holds ``{canonical path: (quality, value, source)}``.
    """
    reg = getattr(config_manager, "_global_search_seeds", None)
    if reg is None:
        reg = {}
        config_manager._global_search_seeds = reg
    return reg


def seed_start(config_manager, path, value, quality, source):
    """Push one searched start value, in the parameter's INTERNAL unit.

    Returns True when the value was applied.  A path already seeded by a
    better-quality search is left alone (see ``QUALITY_TRANSIT`` /
    ``QUALITY_RV``); the user always wins regardless, because the hint
    channel this writes is layered under ``user_params`` everywhere it is
    read -- by rank in the relaxation engine, by position in ``resolve()``.
    """
    key = config_manager.canonical_key(path)
    registry = _seed_registry(config_manager)
    previous = registry.get(key)
    if previous is not None and previous[0] >= quality:
        logger.debug(
            "Global search: %s already seeded at %.10g by %s; the %s value "
            "%.10g is not applied.",
            key,
            previous[1],
            previous[2],
            source,
            value,
        )
        return False

    parts = key.split(".")
    c_type, p_name = parts[0], parts[-1]
    # add_hint takes the parameter's USER unit and the searched quantity is
    # in the internal unit, so it needs the factor.  This used to divide out
    # a SECOND, different factor for an add_override duplicate -- whose
    # values are in the DEFAULTS-yaml unit, and which was therefore only
    # correct while the defaults unit happened to be the internal one, with
    # a warning for the case it was not.  resolve() layers hints now (review
    # 3.14.3), so the duplicate and its unit caveat are both gone.
    #
    # Ask about `key`, NOT `path`: `key` is the canonical (index-form)
    # spelling computed above, and after standardize_param_names that is the
    # only spelling `user_params` can hold.  Asking under the original path
    # found no `unit:` override for a name-form seed and silently used the
    # DEFAULTS unit -- review 2.14.6's defect, at a fifth site, and here the
    # canonical spelling was already sitting two lines up.
    user_factor = (
        config_manager.get_conversion_factor(c_type, p_name, full_path=key)
        or 1.0
    )
    config_manager.add_hint(
        key, value / user_factor, rank=PRECEDENCE_DERIVED_DATA
    )
    registry[key] = (quality, value, source)
    logger.info(
        "Global search: seeded %s = %.10g (internal units) from %s.",
        key,
        value,
        source,
    )
    return True


def _user_named(config_manager, path):
    """True when the params file literally gives this element a start.

    Both the index form the params are standardized to and the 2-part
    broadcast form, which ``standardize_param_names`` leaves alone.
    """
    parts = path.split(".")
    keys = [path]
    if len(parts) == 3:
        keys.append(f"{parts[0]}.{parts[2]}")
    for key in keys:
        entry = config_manager.user_params.get(key)
        if isinstance(entry, dict) and (
            entry.get("initval") is not None or entry.get("mu") is not None
        ):
            return True
    return False


def starts_satisfied(config_manager, groups):
    """Which of these start values already exist, by derivability.

    ``groups`` maps a label to a tuple of alternative paths -- any one of
    which satisfies the group, because ``period`` and ``logP`` are one fact
    in two coordinates.  Returns ``{label: bool}``.

    A group every alternative of which is literally named short-circuits the
    probe (naming a value makes it PRECEDENCE_USER, which is derivable by
    definition).  Otherwise the whole question goes to
    ``ConfigManager.probe_derivable``, which is the only thing that can see
    that a period is implied by ``a`` and the member masses, or that a
    restart file's sampled coordinates pin a derived one.

    **The answer is cached per group, and that is load-bearing, not an
    optimization.**  ``probe_derivable`` runs the relaxation engine, which
    layers ``config_manager.hints`` -- so once ANY search has seeded a
    period, every later caller is told the period is already derivable.  On
    a system with both photometry and RVs that made the answer depend on
    config key order: whichever component ran first seeded the orbit and the
    other skipped its search entirely, so an RV-first config would keep an
    RV period over a photometric one.  The question being asked is "did the
    INPUTS supply this?", which has exactly one answer per run; caching it
    is what makes it that.
    """
    cache = getattr(config_manager, "_global_search_satisfied", None)
    if cache is None:
        cache = {}
        config_manager._global_search_satisfied = cache

    satisfied = {}
    unknown = {}
    for label, paths in groups.items():
        key = tuple(paths)
        if key in cache:
            satisfied[label] = cache[key]
        elif any(_user_named(config_manager, p) for p in paths):
            satisfied[label] = cache[key] = True
        else:
            unknown[label] = paths

    if not unknown:
        return satisfied

    every_path = [p for paths in unknown.values() for p in paths]
    derivable = config_manager.probe_derivable(every_path)
    for label, paths in unknown.items():
        satisfied[label] = cache[tuple(paths)] = any(
            p in derivable for p in paths
        )
    return satisfied


def search_mode(system):
    """The ``global_search:`` switch, read off the orbit block.

    ``False`` opts out entirely; ``True`` forces the search even when the
    starts are already derivable; absent (the default) runs it only when
    something is missing.  It lives on the orbit block for the same reason
    ``mmexofast:`` lives on the lens block: that is the thing being seeded,
    and the instrument that runs the search only borrows the switch.
    Returns one of ``"off"``, ``"force"``, ``"auto"``.
    """
    orbit = getattr(system, "orbit", None)
    if orbit is None:
        return "off"
    values = {
        entry.get("global_search")
        for entry in orbit.config
        if isinstance(entry, dict)
    }
    values.discard(None)
    if not values:
        return "auto"
    if len(values) > 1:
        logger.warning(
            "orbit blocks disagree about global_search (%s); the search is a "
            "system-wide startup decision, so the most restrictive setting "
            "wins.",
            sorted(str(v) for v in values),
        )
    if any(
        v is False or str(v).lower() in ("false", "off", "none")
        for v in values
    ):
        return "off"
    if any(
        v is True or str(v).lower() in ("true", "force", "always")
        for v in values
    ):
        return "force"
    return "auto"


def sole_orbit_index(system, context):
    """The index of the one orbit a blind search may seed, or ``None``.

    Deliberately restrictive.  A periodogram returns peaks, not
    attributions: assigning the strongest peak to ``orbit[0]`` and the next
    to ``orbit[1]`` is a guess, and harmonics and aliases make it a guess
    that fails in exactly the way that hurts most -- one planet's period
    seeded twice, confidently, into two different orbits.  A multi-planet
    blind fit needs a params file; a single-orbit one is the case this
    feature exists for and the case where the attribution is not a guess.
    """
    orbit = getattr(system, "orbit", None)
    if orbit is None:
        return None
    if orbit.n_elements != 1:
        logger.warning(
            "[%s] the global period search is skipped: this system has %d "
            "orbits, and a periodogram peak carries no statement about WHICH "
            "orbit it belongs to. Supply start values for the orbital "
            "periods and conjunction times in the params file (the "
            "standalone BLS/Lomb-Scargle utilities will report the peaks).",
            context,
            orbit.n_elements,
        )
        return None
    return 0
