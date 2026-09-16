import sympy as sp

from ..instrument import JITTER_RELATIONS, JITTER_SYMBOL_MAP

# ---------------------------------------------------------
# 1. Define Symbols
# ---------------------------------------------------------

# All parameters are strictly real.
# Positivity bounds (e.g., jitter > 0) are enforced downstream by defaults.yaml
# NOTE: symbol names must match the get_symbol_map keys exactly; the
# ConfigManager substitutes relation symbols by sym.name, so a mismatched
# name (e.g. 'jittervar') leaves the symbol unbound in the relations.  The
# jitter pair is therefore not declared here: it comes from
# components/instrument.py, the parent that owns the additive noise model.
gamma = sp.symbols("gamma", real=True)

# ---------------------------------------------------------
# 2. Symbol Map
# ---------------------------------------------------------
# Maps SymPy symbols back to the local parameter keys inside the RV Instrument component.


def get_symbol_map(config, system_config):
    return {
        "gamma": "gamma",
        **JITTER_SYMBOL_MAP,
    }


# ---------------------------------------------------------
# 3. Physics Relations
# ---------------------------------------------------------
# Units:
# gamma and jitter are typically in m/s (or whatever your global RV unit is).

# Reparameterization bridge: the user may provide 'jitter' while the sampler
# steps in 'jitter_variance'.  It is the SIGNED square, defined once on the
# shared Instrument parent (see the note there for why registration stays
# per-child, and why jitter**2 would be wrong).
RELATIONS = list(JITTER_RELATIONS)
