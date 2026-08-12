import sympy as sp

# ---------------------------------------------------------
# 1. Define Symbols
# ---------------------------------------------------------

# All parameters are strictly real.
# Positivity bounds are enforced downstream by defaults.yaml
# NOTE: symbol names must match the get_symbol_map keys exactly; the
# ConfigManager substitutes relation symbols by sym.name.
jitter = sp.symbols("jitter", real=True)
jitter_variance = sp.symbols("jitter_variance", real=True)
fluxfrac = sp.symbols("fluxfrac", real=True)

comp_key = "astrometryinstrument"

# ---------------------------------------------------------
# 2. Symbol Map
# ---------------------------------------------------------
# Maps SymPy symbols back to the local parameter keys inside the
# Astrometry Instrument component.


def get_symbol_map(config):
    return {
        "jitter": "jitter",
        "jitter_variance": "jitter_variance",
        "fluxfrac": "fluxfrac",
    }


# ---------------------------------------------------------
# 3. Physics Relations
# ---------------------------------------------------------
# Units: jitter in mas, jitter_variance in mas^2.

RELATIONS = [
    # Reparameterization bridge: user may provide 'jitter', sampler steps
    # in 'jitter_variance'.  Signed square, matching the signed square root
    # the model reports (components/instrument.py:calc_jitter) over the
    # negative variances Instrument._jitter_floor deliberately allows;
    # jitter**2 would fold a negative seed onto a positive variance.
    sp.Eq(jitter_variance, jitter * sp.Abs(jitter))
]


def get_solver_paths():
    """
    Returns the equations defining the state of an Astrometry Instrument.
    """
    return RELATIONS
