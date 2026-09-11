"""Population pharmacokinetics.

READ README.md IN THIS DIRECTORY FIRST.  This component set was written by an
astrophysicist and an LLM with no domain reviewer; it exists to demonstrate and
enforce the component-agnostic architecture and to be a starting point for
non-astronomy development, and its modelling choices are unreviewed.

Two components, mirroring the object/instrument split the tree already has:

``subject``  one instance per individual -- the PK parameters.  Analogue: star.
``assay``    one instance per data file -- the observations, the residual error
             model, and the likelihood.  Analogue: rvinstrument.

Between-subject variability (a ``population`` component supplying a prior over
``subject`` instances, the way ``galacticmodel`` does over ``star``) is P4 and
does not exist yet, so every subject here is independent.
"""

from .assay import Assay
from .subject import Subject

__all__ = ["Assay", "Subject"]
