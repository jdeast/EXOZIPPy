import sympy as sp

# Piggyback on the star section so these relations are instantiated
# once per star instance (star.0, star.1, …).
comp_key = "star"

teff      = sp.Symbol("teff")
teffsed   = sp.Symbol("teffsed")
radius    = sp.Symbol("radius")
radiussed = sp.Symbol("radiussed")

# SED-specific stellar params default to the main stellar params.
RELATIONS = [
    sp.Eq(teffsed, teff),
    sp.Eq(radiussed, radius),
]

def get_symbol_map(config):
    return {
        "teff":      "teff",
        "teffsed":   "teffsed",
        "radius":    "radius",
        "radiussed": "radiussed",
    }
