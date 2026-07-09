import sympy as sp

# ---------------------------------------------------------
# 1. Define Symbols
# ---------------------------------------------------------

# All parameters are strictly real.
# Positivity bounds (e.g., jitter > 0) are enforced downstream by defaults.yaml
# NOTE: symbol names must match the get_symbol_map keys exactly; the
# ConfigManager substitutes relation symbols by sym.name, so a mismatched
# name (e.g. 'jittervar') leaves the symbol unbound in the relations.
gamma = sp.symbols('gamma', real=True)
jitter = sp.symbols('jitter', real=True)
jitter_variance = sp.symbols('jitter_variance', real=True)

# ---------------------------------------------------------
# 2. Symbol Map
# ---------------------------------------------------------
# Maps SymPy symbols back to the local parameter keys inside the RV Instrument component.

def get_symbol_map(config):
    return {
        "gamma": "gamma",
        "jitter": "jitter",
        "jitter_variance": "jitter_variance"
    }

# ---------------------------------------------------------
# 3. Physics Relations
# ---------------------------------------------------------
# Units:
# gamma and jitter are typically in m/s (or whatever your global RV unit is).

RELATIONS = [
    # Reparameterization Bridge (Base-10)
    # Allows the user to provide 'jitter' but the sampler to step in 'logjitter'
    sp.Eq(jitter_variance, jitter**2)
]


def get_solver_paths():
    """
    Returns the equations defining the state of an RV Instrument.
    """
    return RELATIONS