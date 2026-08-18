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
additionally rescales the bounds per light curve), and Instrument pushes a
data-driven hint of ``10 x median(err)``
so the background component starts well separated from the inlier scatter.
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
