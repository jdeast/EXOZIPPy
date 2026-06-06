import sympy as sp
from ...constants import KEPLER_CONST

# Define Symbols
period, a, K, sini, cosi = sp.symbols('period a K sini cosi', real=True)
ecc, omega, inc, Omega = sp.symbols('ecc omega inc Omega', real=True)
tc, t_p = sp.symbols('tc t_p', real=True)
logP = sp.symbols('logP', real=True)
secosw, sesinw = sp.symbols('secosw sesinw', real=True)
ecosw, esinw = sp.symbols('ecosw esinw', real=True)
m_total = sp.symbols('m_total', real=True) # Needs to be mapped to the planet's m_total
M_c = sp.symbols('M_c', real=True)
t_s, f_s, E_s, M_s, n = sp.symbols('t_s f_s E_s M_s n', real=True)

comp_key = "orbit"

def get_symbol_map(config):
    return {
        "logP": "logP",
        "period": "period",
        "secosw": "secosw",
        "sesinw": "sesinw",
        "ecc": "ecc",
        "cosi": "cosi",
        "inc": "inc",
        "sini": "sini",
        "tc": "tc"
    }

RELATIONS = [
    # Reparameterization Bridges (Base-10)
    sp.Eq(period, 10 ** logP),

    # Kepler's Third Law
    sp.Eq(a**3, (KEPLER_CONST**3) * m_total * (period**2)),

    # The Sqrt(e) Vector Bridges (For HMC Sampling)
    sp.Eq(secosw, sp.sqrt(ecc) * sp.cos(omega)),
    sp.Eq(sesinw, sp.sqrt(ecc) * sp.sin(omega)),
    sp.Eq(ecc, secosw ** 2 + sesinw ** 2), # Redundant for SymPy, but safe to keep
    sp.Eq(omega, sp.atan2(sesinw, secosw)),

    # The Linear 'e' Vector Bridges
    sp.Eq(ecosw, ecc * sp.cos(omega)),
    sp.Eq(esinw, ecc * sp.sin(omega)),

    sp.Eq(inc, sp.acos(cosi)),
    sp.Eq(sini, sp.sin(inc)),

    # the solver hangs on transcendental equations
    #sp.Eq(t_p, tc - (M_c / (2*sp.pi/period))),
    #sp.Eq(M_c, sp.symbols('E_c') - ecc * sp.sin(sp.symbols('E_c'))),
    #sp.Eq(n, 2 * sp.pi / period),
    #sp.Eq(f_s, 3*sp.pi/2 - w),
    #sp.Eq(E_s, 2 * sp.atan(sp.sqrt((1-ecc)/(1+ecc)) * sp.tan(f_s/2))),
    #sp.Eq(M_s, E_s - ecc * sp.sin(E_s)),
    #sp.Eq(t_s, t_p + (M_s / n))
]