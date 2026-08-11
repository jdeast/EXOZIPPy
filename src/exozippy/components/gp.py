"""
Optional celerite2 Gaussian-process noise for instrument components.

This module holds everything about GPs that is *not* PyMC-model plumbing: the
``gp:`` config vocabulary, the per-term parameter tables, and the celerite2
kernel constructors.  ``Instrument`` (components/instrument.py) owns the
lifecycle wiring that turns these tables into manifest entries and a marginal
likelihood; keeping the two apart means the kernel catalogue can grow without
touching the component base class.

Two celerite2 terms are supported, chosen per data file with the ``gp:`` key:

  ``rotation`` (celerite2 ``RotationTerm``)
      A mixture of two SHO terms at P and P/2 -- the standard kernel for
      stellar rotation / spot modulation.  Parameters: amplitude ``sigma``,
      rotation ``period``, quality factor ``Q0`` of the secondary mode, the
      difference ``dQ`` between the primary and secondary quality factors,
      and the fractional amplitude ``f`` of the secondary mode.
  ``sho`` (celerite2 ``SHOTerm``)
      SHO is **simple harmonic oscillator**, not shot noise: the kernel of a
      stochastically-driven, damped simple harmonic oscillator, with PSD

          S(w) = sqrt(2/pi) * S0 * w0^4
                 / ((w^2 - w0^2)^2 + w0^2 * w^2 / Q^2)

      This is granulation, and the usual catch-all correlated-noise kernel.
      Parameters: amplitude ``sigma``, undamped period ``rho`` (= 2*pi/w0),
      and quality factor ``Q``.  Q > 1/2 is underdamped -- a real, ringing
      oscillation; Q < 1/2 is overdamped, decaying without ringing; Q = 1/2
      is the critically damped Matern-3/2-like limit.  The default start is
      Q = 1/3 (overdamped), the standard granulation choice.

Parameterization
----------------
The amplitudes and periods are sampled directly, in the data's own units, so a
user writing a prior writes it in units they can reason about
(``gp_rot_period: {mu: 22.56, sigma: 0.29}`` is 22.56 days).  The quality
factors are sampled as base-10 logarithms (``gp_rot_log_q0`` and friends)
because they span decades and a uniform prior on the linear value is a very
informative prior on the log -- the same reasoning behind sampling ``log_s``
in components/mulensing.  The linear values are recorded as pm.Deterministics
so posterior tables report ``Q0``, not ``log Q0``.

Numerical note: ``SHOTerm`` switches between an overdamped and an underdamped
coefficient formula at ``Q = 0.5``.  Both branches are evaluated and guarded
with ``maximum(..., eps)``, so neither poisons the gradient (unlike the JAX
where-trap in notes on potentials.py), but the logp does have a kink there.
A chain that needs to sit near Q = 0.5 will sample it less efficiently; fix Q
(``sigma: 0`` on ``gp_sho_log_q``) if that matters.
"""

import numpy as np

# Canonical term keys, and every spelling accepted in the ``gp:`` config key.
GP_TERMS = ("rotation", "sho")

_TERM_ALIASES = {
    "rotation": "rotation",
    "rotationterm": "rotation",
    "rotation_term": "rotation",
    "rot": "rotation",
    "sho": "sho",
    "shoterm": "sho",
    "sho_term": "sho",
}

# Spellings that mean "no GP on this file". ``gp:`` absent means the same.
_TERM_OFF = {"none", "off", "false", "no", ""}

# The parameters each term declares, in the order they are registered.  These
# are bare parameter names; the owning component supplies the prefix, so a
# user writes e.g. ``rvinstrument.HARPS.gp_rot_period``.
GP_TERM_PARAMS = {
    "rotation": (
        "gp_rot_sigma",
        "gp_rot_period",
        "gp_rot_log_q0",
        "gp_rot_log_dq",
        "gp_rot_f",
    ),
    "sho": ("gp_sho_sigma", "gp_sho_rho", "gp_sho_log_q"),
}

# The amplitude parameter of each term.  This is the one parameter whose scale
# is set by the data rather than by a generic default, so Instrument pushes a
# hint for it (see Instrument._prepare_gp).
GP_AMPLITUDE_PARAM = {
    "rotation": "gp_rot_sigma",
    "sho": "gp_sho_sigma",
}

# Parameters sampled as log10 of the quantity celerite2 actually wants, mapped
# to the name the linear value is reported under.
GP_LOG_PARAMS = {
    "gp_rot_log_q0": "gp_rot_q0",
    "gp_rot_log_dq": "gp_rot_dq",
    "gp_sho_log_q": "gp_sho_q",
}


def parse_gp_spec(value, context=""):
    """Normalize one element's ``gp:`` config value to a tuple of term keys.

    Accepts ``None``/absent (the default -- no GP), a single string, or a list
    of strings; every spelling in ``_TERM_ALIASES`` is allowed, case- and
    whitespace-insensitive.  Returns a tuple in canonical ``GP_TERMS`` order
    with duplicates collapsed, so ``["sho", "rotation", "sho"]`` and
    ``["rotation", "sho"]`` build the same kernel.

    ``context`` is a human-readable location (e.g. "rvinstrument[HARPS]")
    used only in error messages.
    """
    if value is None:
        return ()

    items = value if isinstance(value, (list, tuple)) else [value]

    found = set()
    for item in items:
        if item is None:
            continue
        if isinstance(item, bool):
            # `gp: false` reads as "off"; `gp: true` is too ambiguous to guess.
            if not item:
                continue
            raise ValueError(
                f"[{context}] gp: true is ambiguous -- name the term(s) "
                f"explicitly, e.g. gp: rotation or gp: [rotation, sho]."
            )
        key = str(item).strip().lower()
        if key in _TERM_OFF:
            continue
        if key not in _TERM_ALIASES:
            raise ValueError(
                f"[{context}] unknown GP term '{item}'. Supported terms: "
                f"{', '.join(GP_TERMS)} (or 'none' to disable)."
            )
        found.add(_TERM_ALIASES[key])

    return tuple(t for t in GP_TERMS if t in found)


def gp_config_schema_entry():
    """The shared ``gp`` config-schema entry, for Instrument.config_schema().

    Mirrors Instrument._plot_style_config_schema() so introspection and the
    GUI discover the key generically, without naming any component.
    """
    return {
        "key": "gp",
        "kind": "option",
        "accepts": list(GP_TERMS) + ["none"],
        "required": False,
        "doc": (
            "Optional celerite2 Gaussian-process noise for this data file. "
            "'rotation' adds a RotationTerm (stellar rotation / spot "
            "modulation), 'sho' adds an SHOTerm (granulation and generic "
            "correlated noise); a list adds both, and the default (absent, "
            "or 'none') fits with independent Gaussian errors as before. "
            "The GP hyperparameters become ordinary parameters "
            "(gp_rot_sigma, gp_rot_period, ... ) that can be given priors, "
            "fixed, or linked across files in the params file."
        ),
    }


def _celerite_terms():
    """Import celerite2's PyMC term catalogue, with an actionable error.

    Imported lazily so that neither ``import exozippy`` nor a fit without any
    ``gp:`` key pays for celerite2 (its ops register pytensor Ops at import).
    """
    try:
        from celerite2.pymc import terms
    except ImportError as exc:  # pragma: no cover - install-time failure only
        raise ImportError(
            "Gaussian-process noise (the 'gp:' key on a data file) requires "
            "celerite2, which is not installed. Install it with "
            "'poetry install' (it is a declared dependency) or "
            "'pip install celerite2'."
        ) from exc
    return terms


def build_term(kind, params):
    """Build one celerite2 kernel from a dict of PyTensor scalars.

    ``params`` is keyed by the celerite2 argument names (``sigma``,
    ``period``, ``Q0``, ``dQ``, ``f`` for rotation; ``sigma``, ``rho``, ``Q``
    for sho), i.e. the *linear* values -- the caller has already exponentiated
    anything sampled in log space.
    """
    terms = _celerite_terms()
    if kind == "rotation":
        return terms.RotationTerm(
            sigma=params["sigma"],
            period=params["period"],
            Q0=params["Q0"],
            dQ=params["dQ"],
            f=params["f"],
        )
    if kind == "sho":
        return terms.SHOTerm(
            sigma=params["sigma"], rho=params["rho"], Q=params["Q"]
        )
    raise ValueError(f"Unknown GP term '{kind}'; expected one of {GP_TERMS}.")


def build_kernel(kinds, params_by_kind):
    """Sum the kernels for one data file's terms into a single celerite2 term.

    celerite2 terms add, so a file with ``gp: [rotation, sho]`` gets a kernel
    that is the sum of the two -- rotation plus granulation, each with its own
    independent hyperparameters.
    """
    kernel = None
    for kind in kinds:
        term = build_term(kind, params_by_kind[kind])
        kernel = term if kernel is None else kernel + term
    if kernel is None:
        raise ValueError("build_kernel called with no GP terms.")
    return kernel


def marginal_likelihood(name, kernel, t, yerr, mean, observed):
    """Add a celerite2 GP marginal likelihood to the enclosing pm.Model.

    ``t`` must be sorted ascending -- celerite2's semiseparable solver assumes
    it, and silently returns nonsense otherwise, so Instrument sorts each
    file's observations once at load time and indexes everything else through
    the same permutation.
    """
    from celerite2.pymc import GaussianProcess

    gp = GaussianProcess(kernel, t=t, yerr=yerr, mean=mean)
    gp.marginal(name, observed=observed)
    return gp


def check_sorted(t, context=""):
    """Raise if a GP time array is not sorted ascending (a solver precondition).

    Instrument sorts before it gets here, so this is an internal invariant
    check, not user input validation.
    """
    t = np.asarray(t, dtype=float)
    if t.size and np.any(np.diff(t) < 0):
        raise ValueError(
            f"[{context}] GP times must be sorted ascending; got an unsorted "
            f"array. This is an internal error -- Instrument._prepare_gp is "
            f"responsible for the sort."
        )
    return t
