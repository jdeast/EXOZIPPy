"""Custom astropy units that a ``unit:`` string anywhere in EXOZIPPy may name.

astropy parses unit strings against a registry of ENABLED units, and it
ships no ``century`` -- so ``planet.omegagr`` could not be declared in the
deg/century that EXOFASTv2 and the precession literature use (43 arcsec per
century for Mercury) without either a bare ``unit: ""`` or reporting per
year and asking the reader to divide by 100.  ``def_unit`` plus
``add_enabled_units`` makes the name parseable everywhere astropy parses:
``config.parse_unit``, ``Parameter.__post_init__``, the ``UnitTranslator``
and a user's own ``unit:`` override in a params file.

This module is imported by the package ``__init__`` so the registration
runs before any submodule can parse a unit string -- Python imports the
parent package before any submodule, so nothing has to remember to import
it.  Adding a unit here is the whole job: no other module needs to change,
and a user can write it in a params file the moment it exists.

The Julian century follows IAU convention: 100 Julian years of 365.25 days
(36525 d), which is also EXOFASTv2's ``36525*86400`` seconds.
"""

import astropy.units as u

CENTURY = u.def_unit(
    "century",
    100.0 * u.yr,
    format={"latex": r"\mathrm{century}"},
    doc="Julian century, 100 Julian years (36525 d)",
)

u.add_enabled_units([CENTURY])

__all__ = ["CENTURY"]
