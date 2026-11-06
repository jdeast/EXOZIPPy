"""
Optional robust observation likelihoods for instrument components.

This module holds everything about robust likelihoods that is *not*
PyMC-model plumbing: the per-file ``likelihood:`` config vocabulary, the
parameter tables, and the log-probability builders.  ``Instrument``
(components/instrument.py) owns the lifecycle wiring that turns these tables
into manifest entries and per-file likelihood terms, exactly as it does for
the ``gp:`` key (components/gp.py); keeping the two apart means the
likelihood catalogue can grow without touching the component base class.

Two families are supported, chosen per data file with the ``likelihood:``
key (absent, ``none`` or ``gaussian`` keeps the plain ``pm.Normal``,
byte-for-byte the pre-feature behavior):

  ``hogg`` (Hogg, Bovy & Lang 2010, arXiv:1008.4686 section 3)
      A two-component mixture: each point is drawn from the inlier Normal
      ``N(mu, sigma)`` with probability ``1 - out_frac``, or from a wider
      background Normal ``N(mu, sqrt(sigma^2 + out_scale^2))`` with
      probability ``out_frac``.  The mixture is marginalized analytically
      (no per-point labels are sampled), so the logp is smooth and
      NUTS-friendly.  Inliers keep their full weight -- unlike the
      Student-t, clean data are not downweighted at all -- and the
      per-point posterior outlier probability is available afterwards
      (``Instrument.outlier_prob_at_data``), which makes the mixture an
      auditable replacement for a hard bad-data mask.  Parameters:
      ``out_frac`` (the outlier fraction) and ``out_scale`` (the extra
      scatter of the background component, in the data's own units).
  ``studentt``
      A Student-t observation likelihood with ``nu`` degrees of freedom --
      exactly the marginal of a hierarchical model in which every point's
      variance carries an independent scaled-inverse-chi2 inflation, so
      "Student-t" and "hierarchical error model" are the same posterior with
      the per-point latents integrated out analytically.  Heavy tails
      downweight outliers smoothly and *universally*: genuine model misfit
      is partially absorbed rather than flagged, which is robustness and a
      diagnostic-masking risk in one.  The right tool when the noise itself
      is fat-tailed (e.g. unmodeled stellar variability) rather than a
      distinct junk population.  Parameter: ``t_log_nu`` (log10 of nu).

Parameterization
----------------
``out_frac`` is sampled linearly on [0, 0.5] -- above one half the "outlier"
population would be the majority and the two components swap roles.
``out_scale`` is sampled linearly in the data's own units (m/s for RVs,
relative flux for a transit curve, flux in the file's own arbitrary system
for a microlensing curve -- each instrument overrides the unit in its
defaults.yaml exactly like the GP amplitudes, and MulensInstrument
additionally rescales the bounds per light curve).  Instrument seeds it at
``SCALE_START_FACTOR x median(err)`` and CAPS it at ``SCALE_CAP_FACTOR x
median(err)`` per file (review 8.6.3): left uncapped, the background
component is free to grow until it absorbs genuine model misfit -- on DC2018
event 128 it fitted 300-1000x the median error, forgave 777 nats of
caustic-crossing residuals and inverted the light curve's own preference
for its finite-source size.  With the cap a defecting point buys at most
~ln(SCALE_CAP_FACTOR) nats of width.  The cap is a modelling default the
params file can tighten or loosen, and a posterior piled against it (or
against out_frac's 0.5) is reported at wrap-up as an alarm
(``LIKELIHOOD_CAP_ALARM``): the noise model wants more freedom than the cap
allows, and the residuals decide whether that is junk or signal.
``out_frac``'s prior is flat on its bounds unless the file sets
``out_frac_prior: beta`` (``Beta(1, 100)``) or ``{a: ..., b: ...}``: a
normalized Beta restricted to those bounds (``beta_logp``), which charges
the mixture for every point it claims.  After the fit, the audit helpers
at the bottom of this module (``enrichment_ratio``, ``robust_flags``) feed
``outputs/outliers.py``.
``nu`` is sampled as ``t_log_nu`` (base 10) because it spans decades and only
its order of magnitude matters (nu ~ 2 is very heavy-tailed, nu >~ 50 is
Gaussian for practical purposes); the linear ``t_nu`` is recorded as a
pm.Deterministic, following the ``gp_*_log_q*`` convention.

Numerical note: the mixture logp is assembled with ``pt.logaddexp``, never a
``where`` over branch logps -- a ``where`` with a non-finite untaken branch
poisons the JAX gradient even though the value is correct (the where-trap
documented in potentials.py), and one job of this module is to stay safe
under ``nuts_sampler="numpyro"``.
"""

import numpy as np
import pytensor.tensor as pt

# Canonical family keys, and every spelling accepted in the config key.
LIKELIHOOD_KINDS = ("hogg", "studentt")

_KIND_ALIASES = {
    "hogg": "hogg",
    "mixture": "hogg",
    "hoggmixture": "hogg",
    "hogg_mixture": "hogg",
    "hogg-mixture": "hogg",
    "studentt": "studentt",
    "student_t": "studentt",
    "student-t": "studentt",
    "student": "studentt",
    "t": "studentt",
}

# Spellings that mean "plain Gaussian" (the default; absent means the same).
_KIND_OFF = {"none", "off", "false", "no", "", "gaussian", "normal"}

# The parameters each family declares, in registration order.  Bare names;
# the owning component supplies the prefix, so a user writes e.g.
# ``rvinstrument.HARPS.out_frac``.
LIKELIHOOD_PARAMS = {
    "hogg": ("out_frac", "out_scale"),
    "studentt": ("t_log_nu",),
}

# The parameter of each family whose scale is set by the data rather than by
# a generic default; Instrument pushes a hint for it (see _prepare_robust).
LIKELIHOOD_SCALE_PARAM = {
    "hogg": "out_scale",
}

# That parameter's data-derived START and CAP, as multiples of the file's
# median error bar (in the parameter's own user unit).  The start used to be
# 10x, "so the two mixture components start separated and cannot swap
# roles"; the cap now sits there instead and the start moved down to the
# median error itself.  The ruling (8.6.3) accepted the trade: the damage on
# DC2018 event 128 rode on the WIDTH the background component was allowed,
# not on the count of points it claimed, and out_frac's 0.5 ceiling already
# keeps the two components from swapping roles.
SCALE_START_FACTOR = 1.0
SCALE_CAP_FACTOR = 10.0

# Per family, the parameters whose UPPER bound is a modelling cap rather than
# a validity limit: out_scale's SCALE_CAP_FACTOR x median error, out_frac's
# 0.5 role-swap ceiling.  Instrument._register_robust flags the opted-in
# elements ``cap_alarm`` in the manifest, and the wrap-up
# (diagnostics.cap_alarm_findings) warns when a posterior piles against one.
LIKELIHOOD_CAP_ALARM = {
    "hogg": ("out_frac", "out_scale"),
}

# Parameters sampled as log10 of the quantity the likelihood actually wants,
# mapped to the name the linear value is reported under.
LIKELIHOOD_LOG_PARAMS = {
    "t_log_nu": "t_nu",
}

_LOG_2PI = float(np.log(2.0 * np.pi))


def parse_likelihood_spec(value, context=""):
    """Normalize one element's ``likelihood:`` config value to a family key.

    Returns ``""`` for the default Gaussian (absent, ``none``, ``gaussian``,
    ``false``), or one of ``LIKELIHOOD_KINDS``.  Unlike ``gp:``, a file gets
    exactly ONE likelihood family -- they are alternatives, not addends -- so
    a list is rejected.

    ``context`` is a human-readable location (e.g. "rvinstrument[HARPS]")
    used only in error messages.
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        if not value:
            return ""
        raise ValueError(
            f"[{context}] likelihood: true is ambiguous -- name the family "
            f"explicitly, e.g. likelihood: hogg or likelihood: studentt."
        )
    if isinstance(value, (list, tuple)):
        raise ValueError(
            f"[{context}] likelihood takes a single family, not a list -- "
            f"the families are alternatives to the Gaussian, not addends. "
            f"Got {value!r}."
        )
    key = str(value).strip().lower()
    if key in _KIND_OFF:
        return ""
    if key not in _KIND_ALIASES:
        raise ValueError(
            f"[{context}] unknown likelihood '{value}'. Supported: "
            f"{', '.join(LIKELIHOOD_KINDS)} (or 'gaussian'/'none' for the "
            f"default)."
        )
    return _KIND_ALIASES[key]


def likelihood_config_schema_entry():
    """The shared ``likelihood`` config-schema entry, for config_schema().

    Mirrors gp_config_schema_entry() so introspection and the GUI discover
    the key generically, without naming any component.
    """
    return {
        "key": "likelihood",
        "kind": "option",
        "accepts": list(LIKELIHOOD_KINDS) + ["gaussian", "none"],
        "required": False,
        "doc": (
            "Optional robust observation likelihood for this data file. "
            "'hogg' fits a marginalized inlier/outlier Normal mixture "
            "(Hogg, Bovy & Lang 2010): inliers keep full weight, discrete "
            "junk lands in a wide background component, and per-point "
            "outlier probabilities are available after the fit. 'studentt' "
            "fits heavy tails (a per-point hierarchical error model, "
            "marginalized). The default (absent, 'gaussian' or 'none') "
            "keeps independent Gaussian errors as before. The parameters "
            "(out_frac, out_scale; t_log_nu) become ordinary parameters "
            "that can be given priors, fixed, or linked in the params "
            "file. Mutually exclusive with 'gp:' on the same file."
        ),
    }


def out_frac_prior_config_schema_entry():
    """The shared ``out_frac_prior`` config-schema entry, for config_schema()."""
    return {
        "key": OUT_FRAC_PRIOR_KEY,
        "kind": "option",
        "accepts": ["beta", "uniform"],
        "required": False,
        "doc": (
            "Prior on this file's hogg outlier fraction (needs 'likelihood: "
            "hogg'). Absent or 'uniform' keeps the flat prior on "
            "[0, 0.5]. 'beta' adopts a normalized Beta(1, 100) (median "
            "f = 0.0069), which removes the flat prior's inflation of f on "
            "clean fits; a mapping {a: ..., b: ...} sets the shapes. To FIX the "
            "fraction instead, pin it in the params file: "
            "'<prefix>.<name>.out_frac: {initval: 0.01, sigma: 0}'."
        ),
    }


def hogg_branch_logps(resid, sigma, out_frac, out_scale):
    """The two weighted branch log-densities of the Hogg mixture.

    ``resid`` and ``sigma`` are (n,) tensors (data minus model, and the
    inlier sigma including the instrument's jitter/err_scale term);
    ``out_frac`` and ``out_scale`` are scalars for one file.  Returns
    ``(inlier, outlier)``, each an (n,) tensor holding ``log(weight) +
    log N(resid | 0, that branch's sigma)`` *without* the shared
    ``-0.5*log(2 pi)``, which cancels in the log-odds and is added back once
    in ``hogg_logp``.

    This is the single definition of the mixture's two components.  Both
    consumers -- the likelihood (``hogg_logp``) and the per-point posterior
    outlier log-odds (``hogg_outlier_logodds``) -- are built from it, so the
    probability the fit uses and the probability the audit reports cannot
    drift apart.  Each combines the branches its own way, which is the
    reason this returns them separately rather than returning either
    answer: the mixture must be assembled with ``pt.logaddexp`` and never a
    ``where`` over branch logps (see the numerical note above), while the
    log-odds is a plain difference.

    The ``maximum`` guards keep the logs finite if a user pins out_frac to
    an endpoint (0 or 0.5 exactly); sampled values live strictly inside the
    open interval via the logit transform and never hit them.
    """
    core = -0.5 * pt.sqr(resid / sigma) - pt.log(sigma)
    wide_sigma = pt.sqrt(pt.sqr(sigma) + pt.sqr(out_scale))
    wide = -0.5 * pt.sqr(resid / wide_sigma) - pt.log(wide_sigma)
    log_in = pt.log(pt.maximum(1.0 - out_frac, 1e-300))
    log_out = pt.log(pt.maximum(out_frac, 1e-300))
    return log_in + core, log_out + wide


def hogg_logp(resid, sigma, out_frac, out_scale):
    """Per-point log-probability of the marginalized Hogg mixture.

    Arguments as in ``hogg_branch_logps``.  Returns the (n,) per-point logp
    INCLUDING the -log(sqrt(2 pi)) normalization: unlike a fixed-sigma
    Gaussian, the mixture weights and widths are sampled, so every term that
    depends on them must be kept (the same reasoning as the mann component's
    -log(sigma) note).
    """
    inlier, outlier = hogg_branch_logps(resid, sigma, out_frac, out_scale)
    return pt.logaddexp(inlier, outlier) - 0.5 * _LOG_2PI


def hogg_outlier_logodds(resid, sigma, out_frac, out_scale):
    """Per-point posterior log-odds that a point is an outlier.

    ``sigmoid`` of this is the posterior probability the point came from the
    background component, given the mixture parameters -- the auditable
    replacement for a hard bad-data mask.  Same arguments as ``hogg_logp``,
    and the same two branch densities: the ``-0.5*log(2 pi)`` that
    ``hogg_logp`` adds is common to both branches and cancels here.
    """
    inlier, outlier = hogg_branch_logps(resid, sigma, out_frac, out_scale)
    return outlier - inlier


# ---------------------------------------------------------------------------
# An optional Beta prior on the hogg outlier fraction
# ---------------------------------------------------------------------------
# The per-file config key that turns it on.  Absent (or any spelling in
# _OUT_FRAC_PRIOR_OFF) keeps the flat prior on out_frac's [lower, upper],
# byte-for-byte the pre-feature model.  Why it exists (measured on DC2018
# traces, 2026-10): with the flat prior, a no-planet "plateau" fit pins
# out_scale at its cap and hands 2-6 planetary-anomaly points membership ~1;
# each then costs ~11 nats instead of ~600, which subsidizes the simpler,
# wrong model by up to ~1200 nats.  The prior does NOT cure that: the subsidy
# rides on out_scale's WIDTH (review 8.6.3, see _prepare_robust), not on the
# count, and a handful of claimed points barely moves any sane f prior.  The
# protection against anomaly absorption is the post-fit audit's ANOMALY flag
# plus per-mode fits.  What the prior does fix is the prior-driven
# inflation: with a flat prior, f on clean fits is prior-dominated at ~0.17.
OUT_FRAC_PRIOR_KEY = "out_frac_prior"
_OUT_FRAC_PRIOR_OFF = {"none", "off", "false", "no", "", "uniform", "flat"}

# ``out_frac_prior: beta`` with no shape given: Beta(1, 100), a FIXED
# modelling choice, independent of the file's size (PI ruling, 2026-10).
# a = 1 keeps the density finite and maximal at f = 0 (no 1/f spike); the
# median is 1 - 0.5**(1/100) = 0.0069, which removes the ~0.17 prior-driven
# inflation on clean fits.  It was briefly Beta(1, N/10) -- "about ten bad
# points per file whatever N" -- and that is wrong for real variability: on
# DC2026 five starspot events have 1,500-4,700 discrepant points out of
# 46k, which N/10 excludes by 66-215 orders of magnitude, pushing the misfit
# into err_scale instead.  ``{a: ..., b: ...}`` is the per-file override.
BETA_DEFAULT_A = 1.0
BETA_DEFAULT_B = 100.0


def parse_out_frac_prior(value, context=""):
    """Normalize one file's ``out_frac_prior:`` value to ``(a, b)`` or None.

    ``None`` (absent, ``uniform``, ``flat``, ``none``, ``false``) is the
    default flat prior.  ``beta`` (or ``true``) is ``Beta(BETA_DEFAULT_A,
    BETA_DEFAULT_B)`` = Beta(1, 100), whatever the file's size.  A mapping
    ``{a: ..., b: ...}`` sets either shape explicitly; an omitted one takes
    the same default.  Both shapes must be finite and positive.
    """
    if value is None or value is False:
        return None
    if isinstance(value, str) and value.strip().lower() in _OUT_FRAC_PRIOR_OFF:
        return None
    default_b = BETA_DEFAULT_B
    if value is True or (
        isinstance(value, str) and value.strip().lower() == "beta"
    ):
        a, b = BETA_DEFAULT_A, default_b
    elif isinstance(value, dict):
        unknown = set(value) - {"a", "b"}
        if unknown:
            raise ValueError(
                f"[{context}] {OUT_FRAC_PRIOR_KEY} takes the Beta shapes "
                f"'a' and 'b' only; got {sorted(unknown)}."
            )
        a = float(value.get("a", BETA_DEFAULT_A))
        b = float(value.get("b", default_b))
    else:
        raise ValueError(
            f"[{context}] {OUT_FRAC_PRIOR_KEY} must be 'beta', "
            f"'uniform', or a mapping {{a: ..., b: ...}}; got {value!r}."
        )
    if not (np.isfinite(a) and np.isfinite(b) and a > 0 and b > 0):
        raise ValueError(
            f"[{context}] {OUT_FRAC_PRIOR_KEY}: the Beta shapes must be "
            f"finite and positive; got a={a!r}, b={b!r}."
        )
    return a, b


def beta_log_norm(a, b, lower, upper):
    """``log`` of the Beta(a, b) normalization over ``[lower, upper]``.

    ``log B(a, b) + log(I_upper(a, b) - I_lower(a, b))``: the Beta density
    restricted to out_frac's support (default [0, 0.5]) and renormalized
    there, so the prior is a PROPER density on exactly the interval the
    parameter can take.  A plain number: the shapes and bounds are fixed at
    build time, so the constant cannot reweight anything, but it keeps the
    reported log-prior honest.  The mass difference uses the upper tail
    (``betaincc``) when both bounds sit above the median, for the same
    cancellation reason ``parameter._log_normal_mass`` gives.
    """
    from scipy import special

    lower = float(np.clip(lower, 0.0, 1.0))
    upper = float(np.clip(upper, 0.0, 1.0))
    if not upper > lower:
        raise ValueError(f"Beta prior support [{lower}, {upper}] is empty.")
    lo_cdf = special.betainc(a, b, lower)
    if lo_cdf > 0.5:
        mass = special.betaincc(a, b, lower) - special.betaincc(a, b, upper)
    else:
        mass = special.betainc(a, b, upper) - lo_cdf
    if not mass > 0.0:
        raise ValueError(
            f"Beta({a}, {b}) has no representable mass on [{lower}, {upper}]."
        )
    return float(special.betaln(a, b) + np.log(mass))


def beta_logp(frac, a, b, lower=0.0, upper=0.5):
    """Normalized log-density of Beta(a, b) restricted to [lower, upper].

    ``frac`` is a scalar tensor (one file's out_frac).  The ``maximum``
    floors are strictly positive (CLAUDE.md: a floor at ZERO rebuilds the
    0 * inf the where-trap rule forbids) and only matter at an endpoint a
    user pinned to; sampled values live strictly inside via the logit
    transform.  A shape of exactly 1 contributes no term at all, so the
    default a = 1 has no log(f) singularity to guard.
    """
    out = -beta_log_norm(a, b, lower, upper)
    if a != 1.0:
        out = out + (a - 1.0) * pt.log(pt.maximum(frac, 1e-300))
    if b != 1.0:
        out = out + (b - 1.0) * pt.log(pt.maximum(1.0 - frac, 1e-300))
    return out


# ---------------------------------------------------------------------------
# The post-fit robust-likelihood audit (outputs/outliers.py renders it)
# ---------------------------------------------------------------------------
# A point "belongs to the outlier component" when its posterior membership
# exceeds this.
MEMBERSHIP_THRESHOLD = 0.5
# Flag a file when the high-membership points are concentrated inside the
# signal's event window by more than ENRICH_FLAG over the baseline rate, with
# at least ENRICH_MIN_POINTS of them (fewer is noise), ...
ENRICH_FLAG = 3.0
ENRICH_MIN_POINTS = 3
# ... (the ANOMALY flag), or when MORE than VARIABILITY_FRAC of the file's
# points are outlier-component members and they are NOT concentrated at the
# event (enrich < VARIABILITY_MAX_ENRICH): diffuse variability over the
# baseline (the VARIABILITY flag).  DC2026 validation: 093 (starspots, 4155
# points, enrich 0.1) is VARIABILITY; 094 (344 points, enrich 18.6) is the
# missed ANOMALY; 072 (7 points) is clean.
VARIABILITY_FRAC = 0.01
VARIABILITY_MAX_ENRICH = 2.0
# ... or when the mixture forgives more than this many nats relative to a
# plain Gaussian with the same sigma (Gaussian logL - hogg logL < -this).
FORGIVEN_NATS_FLAG = 10.0
# out_scale above this fraction of its data-driven cap is a "look harder"
# trigger: the background width is what lets the mixture absorb a signal.
CAP_FLAG_FRAC = 0.95

# The flag kinds, as labelled in <prefix>_outliers.txt.
FLAG_ANOMALY = "ANOMALY"
FLAG_VARIABILITY = "VARIABILITY"
FLAG_FORGIVEN = "FORGIVEN"
FLAG_CAP = "AT_CAP"


def gaussian_logl(resid, sigma):
    """Per-point Gaussian log-likelihood (the mixture's f -> 0 limit)."""
    return -0.5 * pt.sqr(resid / sigma) - pt.log(sigma) - 0.5 * _LOG_2PI


def in_windows(times, windows):
    """Boolean mask of ``times`` inside any ``(lo, hi)`` of ``windows``."""
    times = np.asarray(times, dtype=float)
    mask = np.zeros(times.shape, dtype=bool)
    for lo, hi in windows or ():
        mask |= (times >= float(lo)) & (times <= float(hi))
    return mask


def enrichment_ratio(
    membership, times, windows, threshold=MEMBERSHIP_THRESHOLD
):
    """How concentrated the outlier-component points are inside ``windows``.

    ``enrich = P(in window | membership > threshold) / P(in window)``.  A
    planetary/binary anomaly the model is missing sits AT the event, so its
    points are enriched (DC2026 validation on 40 fits: a real missed anomaly
    18.6); bad data and stellar variability are spread over the baseline,
    so enrich ~ 1 (every starspot/noise case <= 0.9).  Returns
    ``(enrich, n_high)``; enrich is NaN when there are no windows, no point
    in them, or no high-membership point.
    """
    membership = np.asarray(membership, dtype=float)
    high = membership > threshold
    n_high = int(high.sum())
    inside = in_windows(times, windows)
    base = float(inside.mean()) if inside.size else 0.0
    if not windows or base == 0.0 or n_high == 0:
        return float("nan"), n_high
    return float(inside[high].mean()) / base, n_high


def robust_flags(delta_logl, enrich, n_high, n=None, scale_frac=None):
    """The reasons a file's mixture fit deserves a look ([] when none).

    Returns ``(kind, text)`` pairs, ``kind`` one of ``FLAG_ANOMALY`` (the
    outlier points sit at the event: likely a missed signal),
    ``FLAG_VARIABILITY`` (many outlier points spread over the baseline:
    real noise the mixture is right to take), ``FLAG_FORGIVEN`` (the
    mixture forgives > FORGIVEN_NATS_FLAG nats) and ``FLAG_CAP`` (out_scale
    at > CAP_FLAG_FRAC of its cap; ``scale_frac`` is out_scale / cap, None
    when unknown).  ``n`` is the file's number of points.
    """
    reasons = []
    if (
        np.isfinite(enrich)
        and enrich > ENRICH_FLAG
        and n_high >= ENRICH_MIN_POINTS
    ):
        reasons.append(
            (
                FLAG_ANOMALY,
                f"{n_high} outlier-component points are {enrich:.1f}x "
                f"concentrated in the event window (> {ENRICH_FLAG:g}): "
                f"likely a real signal the model is missing",
            )
        )
    if (
        n
        and np.isfinite(enrich)
        and n_high > VARIABILITY_FRAC * n
        and enrich < VARIABILITY_MAX_ENRICH
    ):
        reasons.append(
            (
                FLAG_VARIABILITY,
                f"{n_high} of {n} points ({100.0 * n_high / n:.1f}%) are "
                f"outlier-component members spread over the baseline "
                f"(enrich {enrich:.2f} < {VARIABILITY_MAX_ENRICH:g}): "
                f"diffuse variability or bad data, not the event",
            )
        )
    if np.isfinite(delta_logl) and delta_logl < -FORGIVEN_NATS_FLAG:
        reasons.append(
            (
                FLAG_FORGIVEN,
                f"the mixture forgives {-delta_logl:.1f} nats relative to "
                f"a Gaussian (> {FORGIVEN_NATS_FLAG:g})",
            )
        )
    if (
        scale_frac is not None
        and np.isfinite(scale_frac)
        and scale_frac > CAP_FLAG_FRAC
    ):
        reasons.append(
            (
                FLAG_CAP,
                f"out_scale is at {100.0 * scale_frac:.0f}% of its cap: the "
                f"background width is what lets the mixture absorb a "
                f"signal, look harder",
            )
        )
    return reasons
