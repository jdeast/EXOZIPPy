import sympy as sp

from ...constants import FBOL_CONST, LOGG_CONST, LUM_CONST

# Piggyback on the star section so these relations are instantiated
# once per star instance (star.0, star.1, ...).
comp_key = "star"

teff = sp.Symbol("teff")
teffsed = sp.Symbol("teffsed")
radius = sp.Symbol("radius")
radiussed = sp.Symbol("radiussed")
logmass = sp.Symbol("logmass")
luminosity = sp.Symbol("luminosity")
distance = sp.Symbol("distance")
loggsed = sp.Symbol("loggsed")
luminositysed = sp.Symbol("luminositysed")
fbolsed = sp.Symbol("fbolsed")
fbol = sp.Symbol("fbol")
av = sp.Symbol("av")

# Every star parameter the SED declares (Star.register_parameters' `sed`
# branch, plus fbol, which the SED's fbolsed floor reads) is in the
# relaxation engine, so a start-value solver can read any of them with
# probe_start and a user may seed any of them.  The derived ones mirror
# their defaults.yaml expressions exactly.
RELATIONS = [
    # SED-specific stellar params default to the main stellar params.
    sp.Eq(teffsed, teff),
    sp.Eq(radiussed, radius),
    # The BC grid's surface gravity, from radiussed (NOT radius).
    sp.Eq(loggsed, LOGG_CONST + logmass - 2.0 * sp.log(radiussed, 10)),
    sp.Eq(luminositysed, LUM_CONST * radiussed**2 * teffsed**4),
    sp.Eq(fbolsed, FBOL_CONST * luminositysed / distance**2),
    sp.Eq(fbol, FBOL_CONST * luminosity / distance**2),
]


def relations_apply(system_config):
    """Only where there is a `sed:` block (review 1.9.4).

    ``comp_key`` is "star" above so these instantiate per star, but what
    they describe only EXISTS alongside an SED: Star.register_parameters
    declares teffsed/radiussed inside its ``if in_system("sed")`` branch.
    Registered unconditionally, every non-SED config carried two phantom
    leaf symbols per star -- each with its own default-armor value, its
    own provenance-ledger row and its own inject-back initval -- for
    parameter instances that do not exist.
    """
    return bool((system_config or {}).get("sed"))


def get_symbol_map(config, system_config):
    # av is in no relation: a leaf, mapped so the engine owns its start
    # like every other SED input.
    return {
        name: name
        for name in (
            "teff",
            "teffsed",
            "radius",
            "radiussed",
            "logmass",
            "luminosity",
            "distance",
            "loggsed",
            "luminositysed",
            "fbolsed",
            "fbol",
            "av",
        )
    }
