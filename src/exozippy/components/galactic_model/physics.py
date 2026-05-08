import sympy as sp

# ---------------------------------------------------------
# 1. Define Symbols
# ---------------------------------------------------------

# All parameters are strictly real.
# Positivity bounds (period > 0, ecc >= 0) are enforced downstream by defaults.yaml

# Standard Keplerian Elements
period, a, k = sp.symbols('period a k', real=True)
ecc, w, inc, Omega = sp.symbols('ecc w inc Omega', real=True)

# Timing Parameters (Time of conjunction/transit, Time of periastron)
t_c, t_peri = sp.symbols('t_c t_peri', real=True)

# Log parameters (Base-10)
logperiod, loga, logk = sp.symbols('logperiod loga logk', real=True)

# Parameterized Vectors (To avoid e=0 / w degeneracy in HMC)
secw, sesw = sp.symbols('secw sesw', real=True)  # sqrt(e)*cos(w), sqrt(e)*sin(w)
ecosw, esinw = sp.symbols('ecosw esinw', real=True)  # e*cos(w), e*sin(w)

# ---------------------------------------------------------
# 2. Symbol Map
# ---------------------------------------------------------
# Maps SymPy symbols back to the local parameter keys inside the Orbit component.

SYMBOL_MAP = {
    # Scaling parameters
    "period": "period",
    "logperiod": "logperiod",
    "a": "a",
    "loga": "loga",
    "k": "k",
    "logk": "logk",

    # Shape & Orientation
    "ecc": "ecc",
    "w": "w",
    "inc": "inc",
    "Omega": "Omega",

    # Timing
    "t_c": "t_c",
    "t_peri": "t_peri",

    # HMC Sampler Vectors
    "secw": "secw",
    "sesw": "sesw",
    "ecosw": "ecosw",
    "esinw": "esinw"
}

# ---------------------------------------------------------
# 3. Physics Relations
# ---------------------------------------------------------

RELATIONS = [
    # Reparameterization Bridges (Base-10)
    sp.Eq(period, 10 ** logperiod),
    sp.Eq(a, 10 ** loga),
    sp.Eq(k, 10 ** logk),

    # Sqrt(e) Vector Bridges (Mathematically guarantees e >= 0 since squares are positive)
    # secw = sqrt(e) * cos(w)
    # sesw = sqrt(e) * sin(w)
    sp.Eq(ecc, secw ** 2 + sesw ** 2),

    # Linear 'e' Vector Bridges
    # ecosw = e * cos(w)
    # esinw = e * sin(w)
    # We use explicit sp.sqrt here so SymPy doesn't evaluate a negative eccentricity root
    sp.Eq(ecc, sp.sqrt(ecosw ** 2 + esinw ** 2))
]


def get_solver_paths():
    """
    Returns the equations defining the state and parameterizations of an Orbit.
    """
    return RELATIONS