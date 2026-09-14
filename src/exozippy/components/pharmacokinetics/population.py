"""Between-subject variability: the population level of a popPK model.

READ README.md IN THIS DIRECTORY FIRST.
"""

import logging

from ..component import Component
from .subject import Subject

logger = logging.getLogger(__name__)


class Population(Component):
    """The population distribution the subjects are drawn from.

    **This component was written by an astrophysicist and an LLM. No
    biologist, pharmacologist, clinician, or pharmacometrician has reviewed
    it.** It exists to demonstrate and enforce the component-agnostic
    architecture and to be a starting point for non-astronomy development.
    Reproducing a published fit validates that the code computes the model it
    claims to; it does not validate that the model or its priors suit anyone's
    data. See ``README.md`` in this directory before relying on any of it.

    One instance, and it is what turns twelve independent fits into a
    population analysis::

        population:
          - name: "adults"
            variability: [cl, v, ka]

    It supplies a prior over ANOTHER component's instances and owns no data of
    its own, which is ``galacticmodel``'s shape: ``subject``'s log-coordinates
    stop being free parameters and become

        log_q_i = mu_log_q + beta_q * log10(WT_i / WT_ref) + omega_q * eta_q_i

    with ``eta_q_i ~ N(0, 1)`` declared on ``subject`` (so it carries that
    subject's name) and everything else here.

    NON-CENTERED, which is structural and not a matter of taste. The centered
    form makes each subject's coordinate live on a scale of ``omega``, so a
    small ``omega`` closes a funnel that NUTS cannot climb -- and a small
    ``omega`` is exactly what these data produce: the canonical ``nlme`` fit
    of the Theophylline set drives the between-subject SD of one coordinate to
    zero. Sampling the standardized deviation instead keeps every coordinate
    O(1) whatever ``omega`` does.

    THE COVARIATE MODEL IS ALWAYS ON AND HAS NO FLAG. Allometric scaling of
    clearance as ``WT^0.75`` and volume as ``WT^1`` is in nearly every
    population PK model, and its exponents are ordinary PINNED parameters
    here -- so turning it off is ``population.beta_cl: {initval: 0.0}`` in a
    params file and estimating it is ``{sigma: 0.2}``, both without a flag
    that would have to be reconciled with them. A ``covariate: none`` switch
    plus a user-freed exponent is a fourth state that means nothing.

    ONE PARAMETERIZATION PER POPULATION. Every subject must be in the same
    coordinate basis when a population is present, and this raises otherwise.
    That is not an unimplemented case: between-subject variability is defined
    IN a basis, a diagonal ``omega`` in one basis is not diagonal in another,
    and a population whose members disagree about the basis does not name a
    distribution.
    """

    label = "Population"

    # Shares `subject`'s prose topic: they are two halves of one model
    # description, and a reader wants them in one paragraph.
    prose_topic = "pharmacokinetics"

    # Which quantities a `variability:` entry may name.  Bare, so a user
    # writes `cl` and not `log_cl` -- the log is the component's sampling
    # choice, not something a user should have to restate.  Taken from
    # `Subject` rather than restated, so the two cannot disagree about what
    # exists.
    QUANTITIES = Subject.QUANTITIES

    @property
    def prefix(self):
        return "population"

    @classmethod
    def config_schema(cls):
        return [
            {
                "key": "variability",
                "kind": "option",
                "accepts": None,
                "required": False,
                "doc": (
                    "Which of the subject coordinates vary between subjects, "
                    "e.g. [cl, v, ka]. Each named quantity gets a "
                    "between-subject SD (omega) and one standardized "
                    "deviation (eta) per subject. Must be a subset of the "
                    "coordinates the subjects sample. Defaults to all of "
                    "them, which is the usual starting model -- a coordinate "
                    "with no real variability shows up as its omega "
                    "collapsing toward zero rather than needing to be "
                    "removed by hand."
                ),
            },
        ]

    # ------------------------------------------------------------------
    # Stage 1
    # ------------------------------------------------------------------

    def load_data(self, system):
        """Resolve the basis, the varying quantities, and the subject count.

        Reads the subject CONFIG rather than the built ``subject`` component:
        both components' ``load_data`` run in the same stage and nothing
        promises an order, so anything one needs from the other has to come
        from the config it was built from.
        """
        if self.n_elements != 1:
            raise ValueError(
                f"[{self.prefix}] exactly one 'population:' block is "
                f"supported, and {self.n_elements} were given. Several "
                f"populations over disjoint subjects (treatment arms, say) "
                f"is a real model and is not implemented: it needs a "
                f"subject -> population map, and every subject would have to "
                f"say which population it belongs to."
            )

        subjects = self._subject_config(system)
        if not subjects:
            raise ValueError(
                f"[{self.prefix}] a 'population:' block needs 'subject:' "
                f"blocks to be a population OF. It supplies the distribution "
                f"their parameters are drawn from and has no data of its own."
            )
        self.n_subjects = len(subjects)
        self.basis = self._resolve_basis(subjects)
        self.varying = self._parse_variability(self.config[0])

    def _subject_config(self, system):
        """The raw ``subject:`` blocks, from wherever this System keeps them."""
        for holder in (
            getattr(system, "config", None),
            getattr(
                getattr(system, "config_manager", None), "system_config", None
            ),
        ):
            if isinstance(holder, dict) and holder.get("subject"):
                return list(holder["subject"])
        return []

    def _resolve_basis(self, subjects):
        """The single coordinate basis every subject must share.

        `Subject._parse_basis` is a classmethod precisely so this can ask
        the same question the same way: one parser, so a new basis cannot be
        legal on a subject and unknown here.
        """
        modes = {}
        for i, cfg in enumerate(subjects):
            name = cfg.get("name", i)
            mode = Subject._parse_basis(cfg, f"subject '{name}'")
            modes.setdefault(mode, []).append(str(name))
        if len(modes) > 1:
            listing = "; ".join(
                f"{mode}: {', '.join(names)}" for mode, names in modes.items()
            )
            raise ValueError(
                f"[{self.prefix}] the subjects are in more than one "
                f"coordinate basis ({listing}), and a population cannot span "
                f"them. Between-subject variability is defined IN a basis -- "
                f"a diagonal set of omegas in one basis is not diagonal in "
                f"another -- so give every subject the same basis flag "
                f"(fitclv / fitkev / fitclke), or drop the 'population:' "
                f"block and fit them independently."
            )
        return next(iter(modes))

    @property
    def sampled_quantities(self):
        """The bare quantity names the shared basis samples, e.g. (cl, v, ka).

        Read off ``Subject``'s own mode table rather than restated: the table
        is what decides which coordinates exist, and a second list here would
        be a copy to keep in step. It is why every row of that table is
        complete -- a table carrying only the coordinates a basis CHANGES
        would answer this by omission, and did, in the first draft: it
        reported one varying coordinate instead of three.
        """
        return Subject.sampled_quantities(self.basis)

    def _parse_variability(self, cfg):
        """Which quantities carry an eta.  Defaults to all of the basis."""
        sampled = self.sampled_quantities
        raw = cfg.get("variability")
        if raw is None:
            return sampled
        if isinstance(raw, str):
            raw = [raw]
        try:
            asked = [str(q).lower() for q in raw]
        except TypeError:
            raise ValueError(
                f"[{self.prefix}] 'variability:' must be a list of quantity "
                f"names, e.g. [cl, v, ka]; got {raw!r}."
            ) from None

        unknown = [q for q in asked if q not in self.QUANTITIES]
        if unknown:
            raise ValueError(
                f"[{self.prefix}] 'variability:' names unknown quantities "
                f"{unknown}. Known quantities are {list(self.QUANTITIES)} "
                f"(write 'cl', not 'log_cl' -- the log is how this component "
                f"samples it, not what it is)."
            )
        wrong_basis = [q for q in asked if q not in sampled]
        if wrong_basis:
            raise ValueError(
                f"[{self.prefix}] 'variability:' names {wrong_basis}, which "
                f"the subjects' '{self.basis}' basis does not sample (it "
                f"samples {list(sampled)}). Between-subject variability has "
                f"to be stated in the basis being fitted: put the variability "
                f"on a sampled coordinate, or change the subjects' "
                f"basis flag to the one you mean."
            )
        # Preserve the basis order rather than the user's, so the manifest --
        # and so the build order, and so the table -- does not depend on the
        # order somebody happened to type.
        return tuple(q for q in sampled if q in asked)

    # ------------------------------------------------------------------
    # Stage 3
    # ------------------------------------------------------------------

    def register_parameters(self, system):
        """Declare the typical values, the spreads, and the covariate model.

        Only the quantities the shared basis actually samples get a mu and a
        beta, and only the quantities in ``variability:`` get an omega and a
        CV -- a parameter no instance uses is not declared at all, the rule
        ``mode_manifest`` follows for the same reason (components.md).
        """
        manifest = {}
        for q in self.sampled_quantities:
            manifest[f"mu_log_{q}"] = None
            # Pinned at its theoretical exponent by `sigma: 0` in
            # defaults.yaml, and freed by a sigma in a params file.
            manifest[f"beta_{q}"] = None
        manifest["wt_ref"] = None
        for q in self.varying:
            manifest[f"omega_{q}"] = None
            # The field's own unit for a spread, as a derived parameter so it
            # arrives with a credible interval instead of as a point estimate
            # in an extra column (see pharmacokinetics.md).
            manifest[f"cv_{q}"] = "default"
        self.manifest = manifest

        fixed = [q for q in self.sampled_quantities if q not in self.varying]
        logger.info(
            "[%s] %d subjects in the '%s' basis; between-subject variability "
            "on %s%s.",
            self.prefix,
            self.n_subjects,
            self.basis,
            ", ".join(self.varying) or "nothing",
            f" (none on {', '.join(fixed)})" if fixed else "",
        )

    # ------------------------------------------------------------------
    # Stage 7
    # ------------------------------------------------------------------

    def build_likelihood(self, model, system):
        """No likelihood: a population has parameters, not data.

        Its whole effect is that ``subject``'s coordinates are expressions of
        these parameters rather than free -- the same way ``galacticmodel``
        acts on ``star``. The potentials that come with that (the N(0,1) on
        each eta) belong to the parameters that carry them.
        """
        self._add_prose(system)

    def _add_prose(self, system):
        from ...outputs.prose import get_collector

        quantities = {
            "cl": "apparent clearance",
            "v": "apparent volume of distribution",
            "ka": "absorption rate constant",
            "ke": "elimination rate constant",
        }
        varying = [quantities[q] for q in self.varying]
        if len(varying) > 1:
            varying_text = ", ".join(varying[:-1]) + " and " + varying[-1]
        else:
            varying_text = varying[0] if varying else "no parameter"

        prose = get_collector(system)
        prose.add(
            f"The {self.n_subjects} subjects were modelled as draws from a "
            "population: each individual's "
            + varying_text
            + " was written as a typical value times a log-normal "
            "between-subject deviation, sampled in the non-centred form, "
            "with clearance and volume scaled allometrically to body weight "
            "about a reference of $WT_{ref}$. Between-subject variability is "
            "reported as a coefficient of variation, "
            "$CV = 100\\sqrt{\\exp{(\\omega\\ln{10})^2} - 1}\\%$, with "
            "$\\omega$ the standard deviation of the base-10 logarithm.",
            section=self.prose_topic,
            key=f"{self.prefix}.model",
        )

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def compile_plotters(self, model, system):
        pass

    def plot(self, system, points, filename_prefix="debug"):
        pass
