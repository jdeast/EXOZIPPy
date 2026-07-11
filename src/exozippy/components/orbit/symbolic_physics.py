import sympy as sp
from ...constants import KEPLER_CONST
from .bodies import parse_orbit_bodies

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
bigomega, xbigomega, ybigomega = sp.symbols('bigomega xbigomega ybigomega', real=True)

comp_key = "orbit"

def get_symbol_map(config):
    return {
        "logP": "logP",
        "period": "period",
        "secosw": "secosw",
        "sesinw": "sesinw",
        "ecc": "ecc",
        # omega must be mapped: without it the relations instantiate with a
        # bare shared 'omega' symbol, user omega initvals never bind, and
        # secosw is solved from ecc = secosw^2 + sesinw^2 with an
        # unresolvable sign ambiguity (wrong-branch omega ~ 180 deg).
        "omega": "omega",
        "ecosw": "ecosw",
        "esinw": "esinw",
        "cosi": "cosi",
        "inc": "inc",
        "sini": "sini",
        "tc": "tc",
        "bigomega": "bigomega",
        "xbigomega": "xbigomega",
        "ybigomega": "ybigomega",
        # Kepler's-third-law symbols.  These MUST be mapped: an unmapped
        # symbol instantiates as one bare symbol shared by every orbit,
        # letting the relaxation engine equate different orbits' physics
        # (see the omega symbol-map bug).  m_total initvals come from the
        # custom solver below (sum of the member bodies' masses), which
        # lets a user seed a wide orbit with arsun instead of logP.
        "a": "arsun",
        "m_total": "m_total",
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

    # Ascending-node direction vector (same sampler geometry as the
    # microlensing trajectory angle alpha): the engine uses these only
    # forward (bigomega -> xbigomega, ybigomega) to seed the unit-circle
    # direction from a user-supplied bigomega; the sampled xbigomega and
    # ybigomega have N(0,1) priors, giving a uniform marginal on bigomega.
    sp.Eq(xbigomega, sp.cos(bigomega)),
    sp.Eq(ybigomega, sp.sin(bigomega)),

    # (m_total has no closed-form relation here -- it is the sum of the
    # orbit's member-body masses, whose count varies per orbit; the custom
    # solver registered below computes it during relaxation.)

    # the solver hangs on transcendental equations
    #sp.Eq(t_p, tc - (M_c / (2*sp.pi/period))),
    #sp.Eq(M_c, sp.symbols('E_c') - ecc * sp.sin(sp.symbols('E_c'))),
    #sp.Eq(n, 2 * sp.pi / period),
    #sp.Eq(f_s, 3*sp.pi/2 - w),
    #sp.Eq(E_s, 2 * sp.atan(sp.sqrt((1-ecc)/(1+ecc)) * sp.tan(f_s/2))),
    #sp.Eq(M_s, E_s - ecc * sp.sin(E_s)),
    #sp.Eq(t_s, t_p + (M_s / n))
]

def register_solvers(config_manager):
    """orbit.m_total initval = sum of the member bodies' mass initvals.

    Star and planet masses are both solMass internally, so the sum needs no
    unit handling.  Raising KeyError when a mass is not yet resolved lets
    the relaxation engine retry on a later iteration (same protocol as the
    planet.mass custom solver).
    """
    def m_total_solver(resolved, system_config, index):
        orbit_cfgs = (system_config or {}).get("orbit") or []
        if not isinstance(orbit_cfgs, list):
            orbit_cfgs = [orbit_cfgs]
        primary, companion = parse_orbit_bodies(orbit_cfgs, system_config)
        total = 0.0
        for ctype, idx in primary[index] + companion[index]:
            val = resolved.get(f"{ctype}.{idx}.mass")
            if val is None:
                raise KeyError(f"Missing {ctype}.{idx}.mass for "
                               f"orbit.{index}.m_total")
            total += float(val)
        return total

    config_manager.register_custom_solver("orbit.m_total", m_total_solver,
                                          standalone=True)
