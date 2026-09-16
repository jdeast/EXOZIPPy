import sympy as sp

from ..instrument import JITTER_RELATIONS, JITTER_SYMBOL_MAP

# ---------------------------------------------------------
# 1. Define Symbols
# ---------------------------------------------------------
# NOTE: symbol names must match the get_symbol_map keys exactly; the
# ConfigManager substitutes relation symbols by sym.name.  The jitter pair is
# not declared here at all -- it comes from components/instrument.py, the
# parent that owns the additive noise model, so the name can no longer drift
# away from the map key (it did: this file declared sp.symbols("jittervar")
# against a "jitter_variance" key, and its bridge relation was inert).
baseline = sp.symbols("baseline", real=True)


def get_symbol_map(config, system_config):
    return {
        "baseline": "baseline",
        **JITTER_SYMBOL_MAP,
    }


# ---------------------------------------------------------
# 2. Physics Relations
# ---------------------------------------------------------
# Reparameterization bridge: the user may provide 'jitter' (in relative flux)
# while the sampler steps in 'jitter_variance'.  Defined once on the shared
# Instrument parent; see the note there for why registration stays per-child.
RELATIONS = list(JITTER_RELATIONS)
