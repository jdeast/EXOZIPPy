"""Shared scaffolding for the empirical stellar-relation components.

``components/mann`` and ``components/torres`` both constrain a star's mass
and/or radius from other stellar properties, and both are wired the same way:
one instance per constrained star, named after that star, resolving a ``star:``
key and a ``constrain:`` list, warning (never rejecting) when the star starts
outside the relation's calibration range, and adding one masked Gaussian
``pm.Potential`` per constrained quantity.  That is roughly sixty lines of
identical bookkeeping, and it lives here so there is one copy of it.

**This module is deliberately scaffolding only.**  The two relations differ
*statistically*, not cosmetically, and that is why they are separate
components rather than a base class with a ``_predict()`` hook:

* Torres predicts ``log10(M)`` and ``log10(R)`` with scatter quoted in **dex**,
  so its mass penalty acts directly on ``star.logmass`` with no exponentiation
  round trip, and its ``-log(sigma)`` normalization is a constant that is
  dropped (exactly EXOFASTv2's chi2).
* Mann predicts **linear** mass and radius with **fractional** scatter, so its
  sigma is a function of the sampled prediction, its ``-log(sigma)``
  normalization is *not* constant and is kept, and it carries a latent
  non-centered ``appks``.

Nothing in this module knows about either.  ``StellarRelation`` owns no
lifecycle stage that produces a number: it does not define ``load_data``,
``register_parameters`` or ``build_likelihood``, and it holds no relation
coefficients, no sigma convention and no default for one.  Each relation
assembles its own ``observed``/``predicted``/``sigma`` and states its own
normalization choice at the call site (see ``_add_penalty`` below).

It is a plain mixin, not a ``Component`` subclass, so the factory's recursive
sweep for ``Component`` subclasses cannot pick it up.  Use it as
``class Mann(StellarRelation, Component)``.
"""

import logging

import numpy as np
import pymc as pm
import pytensor.tensor as pt

logger = logging.getLogger(__name__)

# The stellar quantities a relation may be asked to constrain.
CONSTRAINABLE = ("mass", "radius")


def star_schema_entry(relation_label):
    """The ``star:`` entry of a relation component's ``config_schema``."""
    return {
        "key": "star",
        "kind": "ref",
        "accepts": ["star"],
        "required": True,
        "doc": (
            f"Name (or index) of the star this {relation_label} relation "
            f"constrains. One {relation_label} instance per star."
        ),
    }


def constrain_schema_entry(inputs_doc):
    """The ``constrain:`` entry of a relation component's ``config_schema``.

    ``inputs_doc`` names what the relation predicts from, e.g.
    ``"absolute Ks"`` or ``"teff/logg/feh"``.
    """
    return {
        "key": "constrain",
        "kind": "option",
        "accepts": list(CONSTRAINABLE),
        "required": False,
        "doc": (
            f"Which stellar quantities to constrain from {inputs_doc} "
            f"(a list; default both mass and radius)."
        ),
    }


def as_float_vector(values):
    """A per-instance list of floats as an explicit float64 tensor constant.

    Explicit float64 because pytensor autocasts a bare Python float to the
    smallest dtype that represents it, and the model is float64 throughout.
    """
    return pt.as_tensor_variable(np.asarray(values, dtype=float))


def star_initval(param, idx):
    """One element of a built Parameter's initval, in internal units.

    Returns ``None`` when there is no initval or the element is NaN, so a
    caller can skip the check rather than compare against a NaN.
    """
    v = getattr(param, "initval", None)
    if v is None:
        return None
    arr = np.atleast_1d(np.asarray(v, dtype=float))
    val = arr[0] if arr.size == 1 else arr[idx]
    return None if np.isnan(val) else float(val)


class StellarRelation:
    """Mixin holding the wiring an empirical stellar relation needs.

    See the module docstring for what is deliberately *not* here.  A user of
    this mixin is expected to set ``self.star_indices`` (a list of star
    indices, one per instance) and ``self.constrain`` (a list of sets drawn
    from :data:`CONSTRAINABLE`) during ``load_data``, which is what
    :meth:`_resolve_star` and :meth:`_parse_constrain` are for.
    """

    def __init__(self, component_config, config_manager):
        # Name each instance after the star it constrains, so the base class's
        # duplicate-name check also enforces one instance per star.
        for c in component_config:
            if c.get("name") is None and c.get("star") is not None:
                c["name"] = str(c["star"]).split(".")[-1]
        super().__init__(component_config, config_manager)

    # ------------------------------------------------------------------
    # Stage 1a helpers: config parsing
    # ------------------------------------------------------------------

    def _resolve_star(self, system, nm, raw_star):
        """Index of the star instance ``nm`` constrains.

        Accepts a bare name (``"B"``), a path (``"star.B"``) and an index
        (``1`` or ``"star.1"``); raises a ValueError naming the available
        stars otherwise.
        """
        star_names = list(system.star.names)
        if raw_star is None:
            raise ValueError(
                f"{self.prefix} '{nm}': a 'star:' key is required naming the "
                f"star to constrain. Available stars: {star_names}."
            )
        key = str(raw_star).split(".")[-1]
        if key in star_names:
            return star_names.index(key)
        if key.isdigit() and int(key) < len(star_names):
            return int(key)
        raise ValueError(
            f"{self.prefix} '{nm}': unknown star '{raw_star}'. "
            f"Available stars: {star_names}."
        )

    def _parse_constrain(self, nm, raw):
        """The set of quantities instance ``nm`` asked to constrain.

        ``None`` (the key absent) means both.  A bare string is accepted as a
        one-element list.
        """
        con = list(CONSTRAINABLE) if raw is None else raw
        if isinstance(con, str):
            con = [con]
        con = set(con)
        bad = con - set(CONSTRAINABLE)
        if bad:
            raise ValueError(
                f"{self.prefix} '{nm}': unknown 'constrain:' entries "
                f"{sorted(bad)}; valid entries are 'mass' and 'radius'."
            )
        if not con:
            raise ValueError(
                f"{self.prefix} '{nm}': 'constrain:' is empty, so this block "
                f"would do nothing. Remove it or list 'mass' and/or 'radius'."
            )
        return con

    # ------------------------------------------------------------------
    # Stage 1b
    # ------------------------------------------------------------------

    def build_maps(self):
        """Stage 1b: index array linking each instance to its star."""
        self.star_map = np.array(self.star_indices, dtype=int)

    # ------------------------------------------------------------------
    # Stage 6 helpers
    # ------------------------------------------------------------------

    def _warn_outside_range(self, system, param, low, high, message):
        """Warn per instance whose star STARTS outside ``[low, high]``.

        EXOFASTv2 re-checks its calibration ranges on every likelihood call
        and hard-rejects out-of-range states.  A ``-inf`` wall has no gradient
        for NUTS to follow, so every range check in these components is a
        startup warning only -- nothing here bounds the posterior.  Called
        from ``build_likelihood`` (stage 6) rather than earlier so the
        initvals read are the relaxed ones the sampler will actually start
        from.

        ``message`` is a format string taking ``{star}`` (the star's name) and
        ``{value}``; it is prefixed with ``"<prefix> '<instance>': "``.  The
        wording is the caller's because the advice is relation-specific (which
        other relation to prefer, and whether the range is two-sided).
        """
        for i, nm in enumerate(self.names):
            si = self.star_indices[i]
            value = star_initval(param, si)
            if value is None or low <= value <= high:
                continue
            logger.warning(
                f"{self.prefix} '{nm}': "
                + message.format(star=system.star.names[si], value=value)
            )

    def _add_penalty(self, which, observed, predicted, sigma, *, normalize):
        """Masked Gaussian potential tying a star quantity to a prediction.

        Only instances that named ``which`` in ``constrain:`` contribute; the
        rest are zeroed elementwise (rather than compacted) so the vector stays
        aligned with ``star_map`` and the instance list.

        ``normalize`` has no default, deliberately: it is the one statistical
        choice the two relation components disagree about, and it has to be
        legible at the call site rather than inherited from here.

        * ``normalize=True`` keeps the ``-log|sigma|`` term.  That is right
          when sigma is a *fraction of the prediction* (mann), because sigma is
          then a function of sampled parameters -- dropping it would leave the
          posterior in those parameters subtly improper.
        * ``normalize=False`` drops it.  That is right when sigma is a
          constant (torres's scatter in dex), where the term is an additive
          constant and dropping it reproduces EXOFASTv2's chi2 exactly.

        Passing a constant sigma with ``normalize=True`` is harmless but
        shifts every logp by a constant; passing a prediction-dependent sigma
        with ``normalize=False`` is a real bug, so neither is defaulted.
        """
        mask = np.array([which in c for c in self.constrain], dtype=bool)
        if not mask.any():
            return
        logp = -0.5 * pt.sqr((observed - predicted) / sigma)
        if normalize:
            logp = logp - pt.log(pt.abs(sigma))
        pm.Potential(
            f"{self.prefix}.{which}_prior",
            pt.sum(pt.where(pt.as_tensor_variable(mask), logp, 0.0)),
        )

    # ------------------------------------------------------------------
    # Plotting: these components have no data of their own to draw.
    # ------------------------------------------------------------------

    def compile_plotters(self, model, system):
        pass

    def plot(self, system, points, filename_prefix="debug"):
        pass
