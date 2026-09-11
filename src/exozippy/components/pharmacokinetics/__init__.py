"""Population pharmacokinetics.

READ README.md IN THIS DIRECTORY FIRST.  This component set was written by an
astronomer and an LLM with no domain reviewer; it exists to demonstrate and
enforce the component-agnostic architecture and to be a starting point for
non-astronomy development, and its modelling choices are unreviewed.

Nothing is exported yet: P1 (the ``subject`` and ``assay`` components) is in
progress, and ``physics.py`` is the only module here.  The factory discovers
``Component`` subclasses by scanning for them, so this file exporting nothing
is what keeps the package inert until there is something to find.
"""
