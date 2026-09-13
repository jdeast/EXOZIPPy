"""Per-individual pharmacokinetic parameters.

READ README.md IN THIS DIRECTORY FIRST.
"""

import logging

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from astropy import units as u

from ...potentials import soft_lower_bound
from ..component import Component, in_topology
from ..parameterization import merge_options, mode_manifest

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

    THE COORDINATE CHOICE, per subject::

        subject:
          - {name: "1", weight: 79.6, dose: 4.02, fitkev: true}

    The same model can be written in several coordinate bases and the field
    uses more than one: ``cl_v`` samples (CL, V) and derives ``ke = CL/V``
    (NONMEM's TRANS2, the default here and there); ``ke_v`` samples (ke, V)
    and derives ``CL = ke*V`` (NONMEM's TRANS1); ``cl_ke`` samples both rates
    and derives ``V = CL/ke`` (R's ``SSfol``).

    The LIKELIHOOD is identical in all three, to machine precision. The PRIOR
    is identical only where their supports overlap, and saying "nothing
    becomes more or less constrained" was too strong: each sampled coordinate
    is uniform over its own bounds, and while the three bases are related by
    unit-determinant maps -- all linear in log space, since
    ``log_ke = log_cl - log_v`` -- so that flat stays flat and no Jacobian
    term is owed, a box in one basis is a parallelogram in another. Each
    admits corners the others exclude. (``orbit``'s ``fitvcve``/``fitchord``
    are the non-linear case, where the component owes a Jacobian and supplies
    one; see ``orbit.md``.)

    Spelled as one BOOLEAN PER BASIS -- ``fitclv`` (the default),
    ``fitkev``, ``fitclke`` -- at most one true, in the house style of
    ``fitvcve`` and ``fitchord``. An enum would make the illegal "two bases
    at once" combination unrepresentable rather than merely rejected, and
    that argument lost to a stronger one: a user who meets ``fitvcve`` in an
    orbit block and an enum here has to learn two spellings for one idea.
    ``_parse_basis`` rejects more than one by name.

    The choice is per instance, so a system may mix them -- and the point of
    ``COORD_MODE_TABLE`` is that the roles fall out rather than being
    hand-masked. Under ``ke_v``, ``cl`` is consumed by nothing at all (the
    likelihood needs only ``ka``, ``ke`` and ``V``) while remaining the one
    quantity every reader of a PK table wants, which is precisely the
    ``reported`` element role.

    WITH A ``population:`` BLOCK the sampled log-coordinates stop being free:
    each becomes ``mu + beta*log10(WT/WT_ref) + omega*eta``, with the typical
    value, the covariate exponent and the between-subject SD owned by
    ``population`` and the standardized deviation ``eta`` owned here, where it
    inherits the subject's own name. Without one, every subject is
    independent -- which is a stack of separate fits sharing an error model
    rather than a population analysis. That gating is the
    ``evolutionarymodel`` pattern: what is in the topology decides what this
    component declares.

    Note what the flip does to this component's bounds, because it is easy to
    read as a double count and is not one: ``log_cl``'s ``lower``/``upper``
    were the hard logit support while it was sampled and become a soft
    barrier once it is derived (``parameter.md``). They make the same
    statement either way -- an individual's clearance stays in the stated
    range -- and the statement was always there.

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

    # The SOFT ordering bound `assume_fast_absorption:` applies, in dex of
    # log10(ka/ke).  `scale` is the natural unit of that quantity -- one dex
    # is a factor of ten in the rate ratio -- and `softness` sets the
    # transition width as a fraction of it, so the penalty runs from ~0 to
    # ~-4.4 nats over 0.1 dex (a 26% rate ratio) and then grows at about 44
    # nats per dex.  Firm enough to keep a chain out of the mirrored mode,
    # gentle enough that a chain started inside it is pushed rather than
    # stopped -- which is the whole difference from the hard truncation
    # `_restrict_bigomega_halfplane`'s removal documents.
    ABSORPTION_ORDER_SCALE = 1.0
    ABSORPTION_ORDER_SOFTNESS = 0.1

    # Attached to the rows the flip-flop degeneracy MOVES, so a reader of a
    # multimodal table knows which numbers to distrust.  CL, and everything
    # derived from it, is invariant across the swap; V is not.
    FLIP_FLOP_NOTE = (
        "not invariant under the flip-flop degeneracy: the mirrored solution "
        "(ka and ke exchanged) fits identically with this quantity scaled by "
        "ke/ka. CL, AUC and half-life are unchanged across the swap."
    )

    # This component set's own prose topic.  Its "what we fitted"
    # sentence had to go under `data` until the topic band became
    # extensible -- see outputs/prose.py.
    prose_topic = "pharmacokinetics"

    label = "Subject"

    # The coordinate bases, as a mode table (see
    # components/parameterization.py).  Mode keys name the SAMPLED PAIR
    # rather than NONMEM's numbers, because "cl_v" says what it selects and
    # "TRANS2" has to be looked up -- the field's names are in the docstring,
    # the schema doc and the log line.  They are also the user-facing config
    # values, so there is no second vocabulary to translate.
    #
    # Read it as a mirror: each mode samples one log-rate, derives the other
    # rate through V, and REPORTS the coordinates it did not sample.  What
    # makes it a mirror rather than two independent cases is that the derived
    # side of one mode is the reported side of the other, and no parameter is
    # ever derived from a quantity the other mode reports -- which is what
    # keeps the per-parameter build order acyclic.  `cl` is derived under
    # "cl" (ke consumes it) and reported under "ke" (nothing does), so the
    # would-be cycle cl -> ke -> cl in a MIXED system never forms: a reported
    # selection contributes no edge to graph.py (parameter.md).
    #
    # Both log-rates report the one they do not sample, rather than leaving it
    # inactive, because both have a computable inverse -- parameter.md's rule
    # for which role a masked-out coordinate takes.  That is also what keeps a
    # user's `subject.S1.log_ke: {lower: -3}` meaningful after a flip.
    # Every row is COMPLETE -- it names log_v/log_ka/v/ka too, which no basis
    # moves -- so that "which coordinates does this basis sample?" can be read
    # off the table rather than restated in a second list beside it.
    # `population` asks exactly that question, and a table carrying only the
    # coordinates that DIFFER answers it wrongly by omission.  A parameter
    # every mode samples expands to `None` and one every mode derives by the
    # same block to that block's name, so carrying them costs nothing:
    # the expansion is identical to the manifest written by hand.
    COORD_MODE_TABLE = {
        # NONMEM TRANS2, and the default here for the same reason it is
        # theirs: clearance is the quantity dosing decisions are made on.
        "cl_v": {
            "log_cl": None,
            "log_v": None,
            "log_ka": None,
            "cl": "default",
            "v": "default",
            "ka": "default",
            "ke": "default",
            "log_ke": {"output_expr_key": "from_cl"},
        },
        # NONMEM TRANS1.
        "ke_v": {
            "log_ke": None,
            "log_v": None,
            "log_ka": None,
            "ke": "from_log",
            "v": "default",
            "ka": "default",
            "cl": {"output_expr_key": "from_ke"},
            "log_cl": {"output_expr_key": "from_ke"},
        },
        # R's `SSfol`, and so the canonical `nlme` fit of the Theophylline
        # data: sample both RATES and derive the volume.  NONMEM has no TRANS
        # number for it.  It is here because between-subject variability is
        # defined IN a basis -- a diagonal set of omegas in one basis is not
        # diagonal in another -- so reproducing a published set of random
        # effects means fitting in the basis they were estimated in, and that
        # published fit's are on (lKe, lKa, lCl).
        "cl_ke": {
            "log_cl": None,
            "log_ke": None,
            "log_ka": None,
            "cl": "default",
            "ke": "from_log",
            "ka": "default",
            "v": "from_rates",
            "log_v": {"output_expr_key": "from_rates"},
        },
    }

    # Bases that cannot appear in one system, and why.  `cl_v` derives ke from
    # (cl, v) while `cl_ke` derives v from (cl, ke), so a system holding both
    # asks graph.py for an order in which v precedes ke AND ke precedes v.
    # The value graph is still acyclic per element -- it is the per-parameter
    # sort that cannot be done -- but there is no way to express that here, so
    # this raises with the reason instead of letting a cycle error surface
    # from the sort.  The other pairs are fine: `ke_v` reports cl rather than
    # deriving it, and a reported selection contributes no edge.
    INCOMPATIBLE_BASES = frozenset({("cl_ke", "cl_v")})

    # The bare quantity names a basis SAMPLES, read off the table above.
    # `population` needs these to know which mu/omega/eta to declare.
    QUANTITIES = ("cl", "v", "ka", "ke")

    @classmethod
    def sampled_quantities(cls, basis):
        """Which of ``QUANTITIES`` the given basis samples, in table order."""
        table = cls.COORD_MODE_TABLE[basis]
        return tuple(
            q for q in cls.QUANTITIES if table.get(f"log_{q}", "") is None
        )

    # The default, named once: `config_schema`'s doc, `_parse_basis` and the
    # log line all have to agree about it.
    DEFAULT_COORDS = "cl_v"

    # ONE BOOLEAN PER BASIS, mutually exclusive, none true = the default.
    # Not an enum, and the ruling is about consistency rather than about this
    # component: a user who meets `fitvcve` and `fitchord` in an orbit block
    # and then an enum in a subject block has to learn
    # two spellings for one idea.  The cost is real and is paid here -- three
    # alternatives cannot be encoded in booleans without an illegal
    # combination, so `_parse_basis` has to reject "more than one true"
    # explicitly, which an enum would have made unrepresentable.  See
    # notes/code_review_20260824.txt item 4.2.7 for the same ruling applied to
    # planet.mass_parameterization.
    BASIS_FLAGS = {
        "fitclv": "cl_v",
        "fitkev": "ke_v",
        "fitclke": "cl_ke",
    }

    # What each basis samples, in words, for the one line the run prints.  A
    # coordinate choice moves no posterior and produces a table with the same
    # rows either way -- which is the whole point of the `reported` role and
    # also what would make a stray basis flag invisible.
    BASIS_DESCRIPTIONS = {
        "cl_v": "(CL, V) -- NONMEM TRANS2",
        "ke_v": "(ke, V) -- NONMEM TRANS1",
        "cl_ke": "(CL, ke), with V derived -- R's SSfol",
    }

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
                "key": "fitclv",
                "kind": "option",
                "accepts": None,
                "required": False,
                "doc": (
                    "Sample this subject in the (CL, V) basis -- clearance "
                    "and volume, NONMEM's TRANS2 -- and derive ke = CL/V. "
                    "This is the default, so the flag exists to say so "
                    "explicitly. At most one of fitclv/fitkev/fitclke may be "
                    "true."
                ),
            },
            {
                "key": "fitkev",
                "kind": "option",
                "accepts": None,
                "required": False,
                "doc": (
                    "Sample this subject in the (ke, V) basis -- elimination "
                    "rate and volume, NONMEM's TRANS1 -- and derive "
                    "CL = ke*V. A coordinate choice: the same model, so "
                    "nothing becomes more or less constrained, and CL is "
                    "still computed and reported."
                ),
            },
            {
                "key": "fitclke",
                "kind": "option",
                "accepts": None,
                "required": False,
                "doc": (
                    "Sample this subject in the (CL, ke) basis -- both rates, "
                    "with the volume derived as V = CL/ke. This is R's "
                    "'SSfol' parameterization, and so the basis the canonical "
                    "nlme fit of the Theophylline data estimates its random "
                    "effects in. It matters once a 'population:' block "
                    "exists, because between-subject variability is defined "
                    "IN a basis."
                ),
            },
            {
                "key": "assume_fast_absorption",
                "kind": "option",
                "accepts": None,
                "required": False,
                "doc": (
                    "Add a SOFT ordering bound ka > ke for this subject, "
                    "which is outside knowledge that absorption is faster "
                    "than elimination. It selects one of the two exactly "
                    "equal solutions the flip-flop degeneracy produces. Off "
                    "by default: both solutions are a real property of "
                    "oral-only data, and they are resolved in practice by an "
                    "IV reference arm, not by a modelling choice. Soft, not "
                    "a truncation -- a chain that starts in the mirrored mode "
                    "is pushed out of it rather than walled in."
                ),
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
        self.coord_modes = []
        self.fast_absorption = []

        for cfg, name in zip(self.config, self.names):
            where = f"{self.prefix} '{name}'"
            self.weight_kg.append(self._parse_weight(cfg, where))
            self.dose_mg.append(
                self._parse_dose(cfg, where, self.weight_kg[-1])
            )
            self.coord_modes.append(self._parse_basis(cfg, where))
            self.fast_absorption.append(
                self._parse_flag(cfg, "assume_fast_absorption", where)
            )

        self.weight_kg = np.asarray(self.weight_kg, dtype=float)
        self.dose_mg = np.asarray(self.dose_mg, dtype=float)

        # Captured here because build_maps (stage 2) takes no `system`.
        # `in_topology` answers from `active_components`, which System fills
        # in its own __init__, so this does not depend on whether the
        # population's own stage-1 pass has run yet.
        self._has_population = in_topology(system, "population") is not None

    # ------------------------------------------------------------------
    # Stage 2
    # ------------------------------------------------------------------

    def build_maps(self):
        """``population_map``: which population each subject belongs to.

        All zeros, because exactly one population is supported -- but it is a
        real map rather than an implied broadcast, and that is the point. A
        bare ``population.mu_log_cl`` dep would resolve to the whole vector of
        the OTHER component's elements and line up with this one only by
        coincidence; naming the map is what makes the pairing provable (and
        is what a second population would extend).
        """
        if self._has_population:
            self.population_map = np.zeros(self.n_elements, dtype=int)

    @staticmethod
    def _parse_flag(cfg, key, where):
        """A per-subject boolean, with no truthiness.

        `assume_fast_absorption: 1` is not obviously an error to a reader and
        would be accepted by `bool()`; refusing it costs nothing and keeps the
        one legal spelling the only spelling.
        """
        raw = cfg.get(key, False)
        if not isinstance(raw, bool):
            raise ValueError(
                f"[{where}] '{key}:' must be true or false; got {raw!r}."
            )
        return raw

    @classmethod
    def _parse_basis(cls, cfg, where):
        """This subject's mode key for ``COORD_MODE_TABLE``.

        At most one basis flag may be true; none means the default. The
        "more than one" case is the price of spelling an n-way choice in
        booleans, so it is rejected by name rather than left to whichever
        flag happens to be checked first.
        """
        chosen = []
        for flag, mode in cls.BASIS_FLAGS.items():
            raw = cfg.get(flag, False)
            if not isinstance(raw, bool):
                extra = ""
                if raw in (1, 2, "1", "2"):
                    extra = (
                        " That looks like a NONMEM TRANS number: TRANS1 is "
                        "'fitkev: true' and TRANS2 is the default."
                    )
                raise ValueError(
                    f"[{where}] '{flag}:' must be true or false; got "
                    f"{raw!r}.{extra}"
                )
            if raw:
                chosen.append(flag)
        if len(chosen) > 1:
            raise ValueError(
                f"[{where}] {', '.join(chosen)} are all true, and they name "
                f"different coordinate bases -- a subject is sampled in one. "
                f"Set at most one (leave them all out for the default, "
                f"'{cls.DEFAULT_COORDS}')."
            )
        return cls.BASIS_FLAGS[chosen[0]] if chosen else cls.DEFAULT_COORDS

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
        # The coordinate choice, per subject (see COORD_MODE_TABLE).  An
        # all-`cl_v` system -- the default, and every example that does not ask
        # otherwise -- expands to exactly the entries this used to write by
        # hand, plus the `log_ke` it now REPORTS.
        coords = mode_manifest(
            self.coord_modes,
            self.COORD_MODE_TABLE,
            n_elements=self.n_elements,
            where=f"{self.prefix}.basis",
        )
        self._reject_incompatible_bases()
        self._log_parameterization_choices()

        # The hierarchy, if there is one: `population` turns the sampled
        # log-coordinates into expressions and adds this component's own
        # per-subject latents.  `coords` is edited IN PLACE, which is the
        # evolutionarymodel pattern in star.py -- a component's manifest
        # depends on what else is in the topology.
        hierarchy = self._apply_population(system, coords)

        # Insertion order is load-bearing: graph.py registers its build-order
        # nodes in manifest order, so this is the order the PyMC nodes -- and
        # so the terms of the summed logp -- are created in.  The historical
        # keys keep their historical positions and `log_ke` is inserted beside
        # the other log coordinates, where a reader looks for it.
        self.manifest = {
            "log_cl": coords["log_cl"],
            "log_v": coords["log_v"],
            "log_ka": coords["log_ka"],
            "log_ke": coords["log_ke"],
            "cl": coords["cl"],
            "v": coords["v"],
            "ka": coords["ka"],
            "ke": coords["ke"],
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
        # After the fixed keys, so an independent-subjects system's manifest
        # -- and so its build order, and so its table -- is unchanged by the
        # existence of this block.
        self.manifest.update(hierarchy)
        self._note_flip_flop_rows()

    def _note_flip_flop_rows(self):
        """Mark the rows the flip-flop degeneracy moves.

        Only when at least one subject is left degenerate: with
        ``assume_fast_absorption`` on everywhere the mirrored solution is
        pushed away, so the note would be describing a mode this fit does not
        report. A note that is sometimes wrong is worse than none, because the
        reader cannot tell which time it is.
        """
        if all(self.fast_absorption):
            return
        for name in ("v", "log_v"):
            entry = self.manifest.get(name)
            if entry is None and name not in self.manifest:
                continue
            self.manifest[name] = merge_options(
                entry, table_note=self.FLIP_FLOP_NOTE
            )

    def _apply_population(self, system, coords):
        """Wire this component into a ``population``, if one is present.

        Edits ``coords`` in place: each coordinate the population speaks for
        stops being sampled and becomes an expression of that population's
        parameters. Returns the entries this component gains as a result --
        one standardized deviation per varying coordinate, and the body
        weight the covariate model reads.

        Returns ``{}`` with no population, which is what makes the hierarchy
        genuinely optional rather than a default with a switch.
        """
        population = in_topology(system, "population")
        if population is None:
            return {}

        varying = set(population.varying)
        gained = {}
        for quantity in population.sampled_quantities:
            key = f"log_{quantity}"
            block = (
                "from_population"
                if quantity in varying
                else "from_population_typical"
            )
            # merge_options rather than plain assignment: the entry may
            # already carry a mask or an output_expr_key from the coordinate
            # choice, and both spellings that "obviously" work here drop one
            # of them (components/parameterization.py).
            coords[key] = merge_options(coords[key], expr_key=block)
            if quantity in varying:
                gained[f"eta_{quantity}"] = None

        # The covariate datum. Required for every subject once a population
        # exists, and not before: without one, `weight:` is config that
        # load_data uses to turn a per-kg dose into milligrams, and a subject
        # dosed in absolute mg legitimately has none.
        missing = [
            name
            for name, weight in zip(self.names, self.weight_kg)
            if not np.isfinite(weight)
        ]
        if missing:
            raise ValueError(
                f"[{self.prefix}] a 'population:' block scales clearance and "
                f"volume with body weight, so every subject needs a "
                f"'weight:' in kg. Missing for: {', '.join(missing)}."
            )
        gained["weight"] = {
            "overrides": {
                "initval": list(self.weight_kg),
                "sigma": 0.0,
            }
        }
        return gained

    def _reject_incompatible_bases(self):
        """Refuse a pair of bases whose build order cannot be sorted."""
        present = set(self.coord_modes)
        for pair in self.INCOMPATIBLE_BASES:
            if present.issuperset(pair):
                a, b = sorted(pair)
                named = {
                    mode: [
                        n
                        for n, m in zip(self.names, self.coord_modes)
                        if m == mode
                    ]
                    for mode in (a, b)
                }
                flag = {m: f for f, m in self.BASIS_FLAGS.items()}
                raise ValueError(
                    f"[{self.prefix}] subjects {named[a]} are in the '{a}' "
                    f"basis ({flag[a]}) and {named[b]} in '{b}' "
                    f"({flag[b]}), and those two cannot be mixed in one "
                    f"system: '{a}' derives one of (v, ke) from the other and "
                    f"'{b}' derives it the other way, so the build order "
                    f"would need each to come first. Use one of them for "
                    f"every subject, or pair either with 'fitkev', which "
                    f"reports the coordinate it does not sample rather than "
                    f"deriving it."
                )

    def _log_parameterization_choices(self):
        """Say which coordinates each subject samples, once, at stage 3.

        A coordinate choice moves no posterior, so the only way anybody
        notices a stray basis flag is if the run says so -- and the
        modes produce tables with the same rows, which is the whole point of
        the `reported` role and also what makes the choice invisible
        otherwise.
        """
        if set(self.coord_modes) == {self.DEFAULT_COORDS}:
            return
        for mode in dict.fromkeys(self.coord_modes):
            named = [
                name
                for name, m in zip(self.names, self.coord_modes)
                if m == mode
            ]
            logger.info(
                "[%s] sampling %s (%s) for %d subject(s): %s.",
                self.prefix,
                self.BASIS_DESCRIPTIONS[mode],
                mode,
                len(named),
                ", ".join(named),
            )

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def build_likelihood(self, model, system):
        """No likelihood of its own: a subject has parameters, not data.

        The data belong to ``assay``, which reads these parameters through its
        own subject map -- the same split as ``star`` and an instrument. The
        one term added here is not a likelihood either: it is the opt-in soft
        ordering bound that picks one of the two mirrored solutions.
        """
        self._add_absorption_order_bound(model)
        self._add_prose(system)

    def _add_absorption_order_bound(self, model):
        """A soft ``ka > ke`` for the subjects that asked for one.

        SOFT, and that is the ruling rather than an implementation detail. The
        two flip-flop solutions are exactly equal in likelihood and both are a
        real property of oral-only data; truncating one away would be a hard
        bound on a posterior that hugs it, which is the failure
        ``_restrict_bigomega_halfplane``'s removal documents. This is outside
        knowledge -- that absorption is faster than elimination -- entered as
        a penalty with a gradient pointing back, so a chain that starts in the
        mirrored mode is pushed out of it rather than walled in.

        Reads ``ka`` and ``ke``, never ``log_ka``/``log_ke``: under ``cl_v``
        the log-rate this would want is a REPORTED element, whose value is a
        placeholder until ``finalize_deferred`` patches it AFTER stage 7. The
        two derived rates are real in every basis.
        """
        selected = np.flatnonzero(self.fast_absorption)
        if not selected.size:
            return

        index = pt.as_tensor_variable(selected.astype("int64"))
        # log10(ka/ke) > 0, indexed rather than masked: pt.where over a
        # log-density is the where-trap, and selecting the elements up front
        # is both safer and cheaper than evaluating a penalty for subjects
        # that did not ask for one.
        ratio = pt.log10(self.ka.value[index] / self.ke.value[index])
        pm.Potential(
            f"{self.prefix}.absorption_order",
            pt.sum(
                soft_lower_bound(
                    ratio,
                    0.0,
                    self.ABSORPTION_ORDER_SCALE,
                    softness=self.ABSORPTION_ORDER_SOFTNESS,
                )
            ),
        )
        logger.info(
            "[%s] assuming ka > ke (a soft bound, not a truncation) for %s; "
            "the mirrored flip-flop solution is penalized, not excluded.",
            self.prefix,
            ", ".join(str(self.names[i]) for i in selected),
        )

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

        # The degeneracy, described where it is created.  It is a property of
        # the model rather than of this dataset, so it is stated whenever it
        # is left in -- a reader of a bimodal posterior needs to know that the
        # two modes are exactly equal by construction and not a feature of
        # the data.
        assumed = [
            name for name, on in zip(self.names, self.fast_absorption) if on
        ]
        if len(assumed) < n:
            prose.add(
                "Oral dosing alone does not distinguish absorption from "
                "elimination: exchanging $k_a$ and $k_e$ and scaling $V/F$ by "
                "$k_e/k_a$ reproduces every predicted concentration exactly, "
                "so the likelihood has two equal modes. Clearance, and "
                "therefore the area under the curve and the terminal "
                "half-life, is the same in both; the volume of distribution "
                "is not.",
                section=self.prose_topic,
                key=f"{self.prefix}.flipflop",
            )
        if assumed:
            prose.add(
                "For "
                + (
                    "every subject"
                    if len(assumed) == n
                    else ", ".join(assumed)
                )
                + " we broke that symmetry with a soft ordering constraint "
                "$k_a > k_e$, penalizing rather than excluding the mirrored "
                "solution.",
                section=self.prose_topic,
                key=f"{self.prefix}.absorption_order",
            )

    def compile_plotters(self, model, system):
        pass

    def plot(self, system, points, filename_prefix="debug"):
        pass
