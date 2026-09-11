"""Per-individual pharmacokinetic parameters.

READ README.md IN THIS DIRECTORY FIRST.
"""

import logging

import numpy as np
from astropy import units as u

from ..component import Component

logger = logging.getLogger(__name__)

# Dose may be given as an absolute amount or per unit body weight; both are
# ordinary in the field and the distinction is carried by the unit rather than
# by a flag, so `dose_unit: mg/kg` is self-documenting where `per_kg: true`
# would not be.
_MASS = u.mg
_MASS_PER_WEIGHT = u.mg / u.kg
_WEIGHT = u.kg


class Subject(Component):
    """One individual's pharmacokinetic parameters.

    **This component was written by an astrophysicist and an LLM. No biologist,
    pharmacologist, clinician, or pharmacometrician has reviewed it.** It
    exists to demonstrate and enforce the component-agnostic architecture and
    to be a starting point for non-astronomy development. Reproducing a
    published fit validates that the code computes the model it claims to; it
    does not validate that the model or its priors suit anyone's data. See
    ``README.md`` in this directory before relying on any of it.

    One instance per individual::

        subject:
          - {name: "1", weight: 79.6, dose: 4.02, dose_unit: "mg/kg"}
          - {name: "2", weight: 72.4, dose: 4.40, dose_unit: "mg/kg"}

    Sampled coordinates are ``log_cl``, ``log_v`` and ``log_ka``; ``cl``,
    ``v``, ``ka`` and ``ke = cl/v`` are derived from them, and ``t_half``,
    ``tmax``, ``cmax`` and ``auc`` are derived for reporting. The forward
    model itself, and the removable ``ka == ke`` singularity it has to
    survive, are in ``physics.py``.

    Every subject is independent here. Between-subject variability -- the
    hierarchy that makes this "population" PK rather than a stack of separate
    fits -- is P4, and arrives as a separate component supplying a prior over
    these instances, the way ``galacticmodel`` supplies one over ``star``.

    THE FLIP-FLOP DEGENERACY. The likelihood has two exactly equal modes:
    swapping ``ka`` and ``ke`` and rescaling ``V -> V*ke/ka`` leaves every
    predicted concentration bit-identical (pinned in
    ``tests/test_pharmacokinetics_physics.py``). ``CL`` is invariant across
    the swap and ``V`` is not, so clearance-derived quantities (AUC,
    steady-state dosing) are the same in both modes while volume-derived ones
    (a loading dose) differ by ``ke/ka``. Nothing here breaks the symmetry:
    the two solutions are a real property of oral-only data, resolved in
    practice by an IV reference arm or by outside knowledge, and truncating
    one away would be a hard bound on a posterior that hugs it -- the failure
    ``_restrict_bigomega_halfplane``'s removal documents. ``expects_suppressed_modes``
    is therefore True, which is what turns on hot-chain retention generically
    without anyone editing the sampler.
    """

    # Declared on Component precisely so a component with degenerate solutions
    # can opt in without the sampler layer learning any component's name.
    expects_suppressed_modes = True

    # This component set's own prose topic.  Its "what we fitted"
    # sentence had to go under `data` until the topic band became
    # extensible -- see outputs/prose.py.
    prose_topic = "pharmacokinetics"

    label = "Subject"

    @property
    def prefix(self):
        return "subject"

    @classmethod
    def config_schema(cls):
        return [
            {
                "key": "weight",
                "kind": "option",
                "accepts": None,
                "required": False,
                "doc": (
                    "Body weight in kg. Required only when 'dose_unit' is "
                    "per unit weight (e.g. mg/kg), which is how most trial "
                    "data record a dose."
                ),
            },
            {
                "key": "dose",
                "kind": "option",
                "accepts": None,
                "required": True,
                "doc": "Administered dose, in the units of 'dose_unit'.",
            },
            {
                "key": "dose_unit",
                "kind": "option",
                "accepts": None,
                "required": False,
                "doc": (
                    "Unit of 'dose'. Either an amount ('mg', 'g') or an "
                    "amount per body weight ('mg/kg'); the latter is "
                    "multiplied by 'weight'. Default 'mg'."
                ),
            },
        ]

    # ------------------------------------------------------------------
    # Stage 1
    # ------------------------------------------------------------------

    def load_data(self, system):
        """Resolve each subject's absolute dose in mg.

        The dose is CONFIG, not a data file, but it is resolved here rather
        than in ``__init__`` because it has to be a per-element ``initval``
        pushed at stage 3, and stage 1 is where a component is allowed to
        compute the numbers it will later declare.
        """
        self.weight_kg = []
        self.dose_mg = []

        for cfg, name in zip(self.config, self.names):
            where = f"{self.prefix} '{name}'"
            self.weight_kg.append(self._parse_weight(cfg, where))
            self.dose_mg.append(
                self._parse_dose(cfg, where, self.weight_kg[-1])
            )

        self.weight_kg = np.asarray(self.weight_kg, dtype=float)
        self.dose_mg = np.asarray(self.dose_mg, dtype=float)

    @staticmethod
    def _parse_weight(cfg, where):
        """Body weight in kg, or NaN when none was given.

        NaN rather than a default: there is no safe stand-in for a body
        weight, and the only thing that reads it (a per-weight dose) raises
        its own error naming the missing key.
        """
        raw = cfg.get("weight")
        if raw is None:
            return float("nan")
        try:
            value = float(raw)
        except (TypeError, ValueError):
            raise ValueError(
                f"[{where}] 'weight:' must be a number in kg; got {raw!r}."
            ) from None
        if not np.isfinite(value) or value <= 0:
            raise ValueError(
                f"[{where}] 'weight:' must be a positive number in kg; "
                f"got {raw!r}."
            )
        return value

    @staticmethod
    def _parse_dose(cfg, where, weight_kg):
        """Absolute dose in mg.

        Accepts an amount ('mg', 'g') or an amount per body weight ('mg/kg').
        The unit carries the distinction, so the two spellings cannot be
        confused the way a boolean flag beside a bare number could be -- and a
        per-weight dose with no weight raises here rather than silently
        becoming a mg dose 70x too small.
        """
        raw = cfg.get("dose")
        if raw is None:
            raise ValueError(
                f"[{where}] a 'dose:' is required (with 'dose_unit:' if it "
                f"is not in mg)."
            )
        try:
            value = float(raw)
        except (TypeError, ValueError):
            raise ValueError(
                f"[{where}] 'dose:' must be a number; got {raw!r}."
            ) from None
        if not np.isfinite(value) or value <= 0:
            raise ValueError(
                f"[{where}] 'dose:' must be positive; got {raw!r}."
            )

        unit_str = str(cfg.get("dose_unit", "mg"))
        try:
            unit = u.Unit(unit_str)
        except Exception:
            raise ValueError(
                f"[{where}] 'dose_unit:' {unit_str!r} is not a unit astropy "
                f"understands. Use an amount ('mg', 'g') or an amount per "
                f"body weight ('mg/kg')."
            ) from None

        if unit.is_equivalent(_MASS):
            return float((value * unit).to(_MASS).value)

        if unit.is_equivalent(_MASS_PER_WEIGHT):
            if not np.isfinite(weight_kg):
                raise ValueError(
                    f"[{where}] 'dose_unit: {unit_str}' is a dose per body "
                    f"weight, so a 'weight:' in kg is required to turn it "
                    f"into an absolute dose."
                )
            per_kg = float((value * unit).to(_MASS_PER_WEIGHT).value)
            return per_kg * weight_kg

        raise ValueError(
            f"[{where}] 'dose_unit: {unit_str}' is neither an amount nor an "
            f"amount per body weight. Use e.g. 'mg' or 'mg/kg'."
        )

    # ------------------------------------------------------------------
    # Stage 3
    # ------------------------------------------------------------------

    def register_parameters(self, system):
        """Declare the manifest.

        ``dose`` is pinned per element through the ``"overrides"`` channel
        rather than ``add_hint``: it is not a ranked START value for something
        the relaxation engine might solve differently, it is the datum, and a
        pin must say what it pins to (parameter.md). ``"overrides"`` also
        layers UNDER the params file, so a user who really wants to probe a
        different dose can still say so.
        """
        self.manifest = {
            "log_cl": None,
            "log_v": None,
            "log_ka": None,
            "cl": "default",
            "v": "default",
            "ka": "default",
            "ke": "default",
            "t_half": "default",
            "tmax": "default",
            "cmax": "default",
            "auc": "default",
            "dose": {
                "overrides": {
                    "initval": list(self.dose_mg),
                    "sigma": 0.0,
                }
            },
        }

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def build_likelihood(self, model, system):
        """No likelihood of its own: a subject has parameters, not data.

        The data belong to ``assay``, which reads these parameters through its
        own subject map -- the same split as ``star`` and an instrument.
        """
        self._add_prose(system)

    def _add_prose(self, system):
        """Declare the modeling-draft sentence, at the site that owns it."""
        from ...outputs.prose import get_collector

        prose = get_collector(system)
        n = self.n_elements
        prose.add(
            f"We modelled the plasma concentrations of {n} "
            + ("subject" if n == 1 else "subjects")
            + " with a one-compartment model with first-order absorption and "
            "first-order elimination, parameterized by apparent clearance "
            "$CL/F$, apparent volume of distribution $V/F$, and absorption "
            "rate constant $k_a$, each sampled in $\\log_{10}$. "
            "Bioavailability $F$ is not identifiable from oral dosing alone "
            "and was fixed at unity, so clearance and volume are apparent "
            "values.",
            # This component's OWN topic, declared as `prose_topic` above.
            # It went under "data" until the prose topic band was made
            # extensible: the vocabulary was a closed astronomy list, so a
            # sentence from another field had nowhere of its own to stand and
            # was ordered among the data-inventory sentences rather than
            # after them.
            section=self.prose_topic,
            key=f"{self.prefix}.model",
        )

    def compile_plotters(self, model, system):
        pass

    def plot(self, system, points, filename_prefix="debug"):
        pass
