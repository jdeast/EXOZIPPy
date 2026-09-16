import sympy as sp

# Piggyback on the star section so these relations are instantiated
# once per star instance (star.0, star.1, ...).
comp_key = "star"

teff = sp.Symbol("teff")
teffsed = sp.Symbol("teffsed")
radius = sp.Symbol("radius")
radiussed = sp.Symbol("radiussed")

# SED-specific stellar params default to the main stellar params.
RELATIONS = [
    sp.Eq(teffsed, teff),
    sp.Eq(radiussed, radius),
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
    return {
        "teff": "teff",
        "teffsed": "teffsed",
        "radius": "radius",
        "radiussed": "radiussed",
    }
