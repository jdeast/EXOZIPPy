import sympy as sp

# ---------------------------------------------------------
# 1. Define Symbols
# ---------------------------------------------------------

# All parameters are strictly real.
# Positivity bounds (e.g., jitter > 0) are enforced downstream by defaults.yaml
# NOTE: symbol names must match the get_symbol_map keys exactly; the
# ConfigManager substitutes relation symbols by sym.name, so a mismatched
# name (e.g. 'jittervar') leaves the symbol unbound in the relations.
gamma = sp.symbols("gamma", real=True)
jitter = sp.symbols("jitter", real=True)
jitter_variance = sp.symbols("jitter_variance", real=True)

# ---------------------------------------------------------
# 2. Symbol Map
# ---------------------------------------------------------
# Maps SymPy symbols back to the local parameter keys inside the RV Instrument component.


def get_symbol_map(config):
    return {
        "gamma": "gamma",
        "jitter": "jitter",
        "jitter_variance": "jitter_variance",
    }


# ---------------------------------------------------------
# 3. Physics Relations
# ---------------------------------------------------------
# Units:
# gamma and jitter are typically in m/s (or whatever your global RV unit is).

RELATIONS = [
    # Reparameterization bridge: the user may provide 'jitter' while the
    # sampler steps in 'jitter_variance'.  It is the SIGNED square, because
    # the runtime relation is the signed square root
    # (components/instrument.py:calc_jitter) and jitter_variance is
    # deliberately allowed to go negative down to Instrument._jitter_floor.
    # Written as jitter**2 it would fold a negative 'jitter' seed onto a
    # POSITIVE variance -- a silent sign flip on the one direction of this
    # relation that matters (the user seeds jitter, the engine derives the
    # sampled variance).
    sp.Eq(jitter_variance, jitter * sp.Abs(jitter))
]


def get_solver_paths():
    """
    Returns the equations defining the state of an RV Instrument.
    """
    return RELATIONS
