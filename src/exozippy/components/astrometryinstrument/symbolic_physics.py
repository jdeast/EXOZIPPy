import sympy as sp

from ..instrument import JITTER_RELATIONS, JITTER_SYMBOL_MAP

# ---------------------------------------------------------
# 1. Define Symbols
# ---------------------------------------------------------

# All parameters are strictly real.
# Positivity bounds are enforced downstream by defaults.yaml
# NOTE: symbol names must match the get_symbol_map keys exactly; the
# ConfigManager substitutes relation symbols by sym.name.  The jitter pair is
# therefore not declared here: it comes from components/instrument.py, the
# parent that owns the additive noise model.
fluxfrac = sp.symbols("fluxfrac", real=True)

comp_key = "astrometryinstrument"

# ---------------------------------------------------------
# 2. Symbol Map
# ---------------------------------------------------------
# Maps SymPy symbols back to the local parameter keys inside the
# Astrometry Instrument component.


def get_symbol_map(config, system_config):
    return {
        "fluxfrac": "fluxfrac",
        **JITTER_SYMBOL_MAP,
    }


# ---------------------------------------------------------
# 3. Physics Relations
# ---------------------------------------------------------
# Units: jitter in mas, jitter_variance in mas^2.

# Reparameterization bridge: user may provide 'jitter', sampler steps in
# 'jitter_variance'.  Signed square, defined once on the shared Instrument
# parent; see the note there.
RELATIONS = list(JITTER_RELATIONS)
