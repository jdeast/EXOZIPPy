"""Population pharmacokinetics.

READ README.md IN THIS DIRECTORY FIRST.  This component set was written by an
astrophysicist and an LLM with no domain reviewer; it exists to demonstrate and
enforce the component-agnostic architecture and to be a starting point for
non-astronomy development, and its modelling choices are unreviewed.

Three components, mirroring the object / population-prior / instrument split
the tree already has:

``subject``     one instance per individual -- the PK parameters.
                Analogue: star.
``population``  one instance -- the distribution the subjects are drawn from,
                and the covariate model.  Owns no data.  Analogue:
                galacticmodel, which likewise exists only to put a prior on
                another component's instances.
``assay``       one instance per data file -- the observations, the residual
                error model, and the likelihood.  Analogue: rvinstrument.

``population`` is optional: with no such block every subject is independent,
which is twelve separate fits sharing an error model rather than a population
analysis.
"""

from .assay import Assay
from .population import Population
from .subject import Subject

__all__ = ["Assay", "Population", "Subject"]
