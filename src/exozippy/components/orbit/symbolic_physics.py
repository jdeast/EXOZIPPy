import sympy as sp
from ...constants import KEPLER_CONST

# Define Symbols
period, a, k = sp.symbols('period a k', real=True)
ecc, w, inc, Omega = sp.symbols('ecc w inc Omega', real=True)
t_c, t_peri = sp.symbols('t_c t_peri', real=True)
logperiod, loga, logk = sp.symbols('logperiod loga logk', real=True)
secw, sesw = sp.symbols('secw sesw', real=True)
ecosw, esinw = sp.symbols('ecosw esinw', real=True)
m_total = sp.symbols('m_total', real=True) # Needs to be mapped to the planet's m_total

# ... [Keep your SYMBOL_MAP here] ...

RELATIONS = [
    # Reparameterization Bridges (Base-10)
    sp.Eq(period, 10 ** logperiod),
    sp.Eq(a, 10 ** loga),
    sp.Eq(k, 10 ** logk),

    # Kepler's Third Law
    sp.Eq(a**3, (KEPLER_CONST**3) * m_total * (period**2)),

    # The Sqrt(e) Vector Bridges (For HMC Sampling)
    sp.Eq(secw, sp.sqrt(ecc) * sp.cos(w)),
    sp.Eq(sesw, sp.sqrt(ecc) * sp.sin(w)),
    sp.Eq(ecc, secw ** 2 + sesw ** 2), # Redundant for SymPy, but safe to keep

    # The Linear 'e' Vector Bridges
    sp.Eq(ecosw, ecc * sp.cos(w)),
    sp.Eq(esinw, ecc * sp.sin(w)),
]