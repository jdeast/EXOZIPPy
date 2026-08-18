import numpy as np
import sympy as sp

from ...constants import KEPLER_CONST
from .bodies import parse_orbit_bodies

# Define Symbols.  Only symbols that appear in RELATIONS below are declared:
# ConfigManager binds a relation's free_symbols to the paths get_symbol_map
# names, by symbol NAME, so a declared-but-unused symbol is inert.  (The map
# itself is wider than this list on purpose -- every mapped path becomes a
# leaf symbol in the relaxation engine whether or not a relation mentions it,
# which is what carries `tc`.)
period, a, sini, cosi = sp.symbols("period a sini cosi", real=True)
ecc, omega, inc = sp.symbols("ecc omega inc", real=True)
logP = sp.symbols("logP", real=True)
secosw, sesinw = sp.symbols("secosw sesinw", real=True)
ecosw, esinw = sp.symbols("ecosw esinw", real=True)
m_total = sp.symbols(
    "m_total", real=True
)  # Needs to be mapped to the planet's m_total
bigomega, xbigomega, ybigomega = sp.symbols(
    "bigomega xbigomega ybigomega", real=True
)
vcve, xomega, yomega = sp.symbols("vcve xomega yomega", real=True)

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
        # Mapped for the same reason every symbol here is: an unmapped symbol
        # is ONE bare symbol shared by every orbit, so a two-orbit fit would
        # equate their eccentricity parameterizations.
        # Mapped here so the path exists for THIS component; the relation
        # that bridges it to cos i lives in planet/symbolic_physics.py,
        # because only the planet side can name both (the chord is defined by
        # a radius ratio, and get_symbol_map sees one component's config).
        "chord": "chord",
        "vcve": "vcve",
        "xomega": "xomega",
        "yomega": "yomega",
        # Kepler's-third-law symbols.  These MUST be mapped: an unmapped
        # symbol instantiates as one bare symbol shared by every orbit,
        # letting the relaxation engine equate different orbits' physics
        # (see the omega symbol-map bug).  m_total initvals come from the
        # custom solver below (sum of the member bodies' masses), which
        # lets a user seed a wide orbit with a instead of logP.
        "a": "a",
        "m_total": "m_total",
    }


RELATIONS = [
    # Reparameterization Bridges (Base-10)
    sp.Eq(period, 10**logP),
    # Kepler's Third Law
    sp.Eq(a**3, (KEPLER_CONST**3) * m_total * (period**2)),
    # The Sqrt(e) Vector Bridges (For HMC Sampling)
    sp.Eq(secosw, sp.sqrt(ecc) * sp.cos(omega)),
    sp.Eq(sesinw, sp.sqrt(ecc) * sp.sin(omega)),
    sp.Eq(ecc, secosw**2 + sesinw**2),  # Redundant for SymPy, but safe to keep
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
    # V_c/V_e (Eastman 2024 eq 4) and its omega direction vector.  The bridge
    # is what lets ONE params file drive either parameterization: a user's
    # ecc/omega seeds give a V_c/V_e start (and the direction vector), and a
    # V_c/V_e seed back-solves an eccentricity.  Written in the forward
    # direction because that one is single-valued; the inverse has two roots
    # (physics.calc_ecc_from_vcve and its _lo twin), and sympy is free to pick
    # either when it solves this for ecc -- which is exactly as good as the
    # engine needs, since a start value only has to be a valid solution.
    sp.Eq(vcve * (1 + ecc * sp.sin(omega)), sp.sqrt(1 - ecc**2)),
    sp.Eq(xomega, sp.cos(omega)),
    sp.Eq(yomega, sp.sin(omega)),
    # (m_total has no closed-form relation here -- it is the sum of the
    # orbit's member-body masses, whose count varies per orbit; the custom
    # solver registered below computes it during relaxation.)
    #
    # There is deliberately no Kepler-equation chain here (t_p <-> tc, and the
    # secondary-eclipse time t_s): sympy hangs on the transcendental
    # M = E - e*sin(E).  tc's start comes from the data instead
    # (Orbit._seeded_period sets its window), and t_p is a runtime
    # Deterministic.
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
                raise KeyError(
                    f"Missing {ctype}.{idx}.mass for orbit.{index}.m_total"
                )
            total += float(val)
        return total

    config_manager.register_custom_solver(
        "orbit.m_total", m_total_solver, standalone=True
    )

    def omega_solver(resolved, system_config, index):
        """orbit.omega for a CIRCULAR orbit, where the relation cannot say.

        `Eq(omega, atan2(sesinw, secosw))` is undefined when the sqrt(e) pair
        is exactly zero, which is how a circular fit is written (`sigma: 0` on
        both -- examples/kelt17), so the engine left omega unresolved forever.
        The RUNTIME has always had a convention for that case:
        physics.calc_omega returns pi/2 there, "so the RV phase stays perfectly
        aligned".  This states the same convention where the engine can use it,
        and it exists because a start value the model disagrees with is worse
        than no start value: every relation downstream of omega -- esinw,
        ecosw, V_c/V_e, the transit chord -- was blocked on it.

        Only the exactly-zero case.  Anywhere else the relation is well posed
        and must be left alone, so this raises KeyError (the "cannot solve yet"
        signal) rather than competing with it.
        """
        secosw = resolved.get(f"orbit.{index}.secosw")
        sesinw = resolved.get(f"orbit.{index}.sesinw")
        if secosw is None or sesinw is None:
            raise KeyError(f"Missing the sqrt(e) pair for orbit.{index}.omega")
        if float(secosw) != 0.0 or float(sesinw) != 0.0:
            raise KeyError(
                f"orbit.{index} is not circular; atan2 defines omega there"
            )
        return float(np.pi / 2.0)

    config_manager.register_custom_solver(
        "orbit.omega", omega_solver, standalone=True
    )

    def chord_solver(resolved, system_config, index):
        """orbit.chord initval from a seeded cos i (Eastman 2024's inversion).

        chord^2 = (1 + R_P/R_*)^2 - b^2 with b = a/R* cos i (1 - e^2)/(1 + esinw)
        -- Winn 2010 eq 7, the form calc_b uses.  Every input is read from the
        engine: `planet.ar` and `esinw` resolve through the relations in
        planet/ and orbit/symbolic_physics.py, so this recomputes no physics
        and cannot drift from them.

        A SOLVER rather than the relation it obviously wants to be, and the
        reason is measured, not assumed: as a relation it costs 7.6 s of sympy
        per System (prepare on examples/kelt17 goes 2.0 s -> 9.6 s), because
        the equation is quadratic in the chord and transcendental in omega and
        the engine attempts every unresolved symbol in it.  planet.log_q has a
        solver for exactly this reason.  What made the relation ATTRACTIVE --
        that a/R* is a first-class engine quantity now -- is what makes this
        function small.

        Standalone (it is never the last unknown of an equation, having no
        equation) and RANK_DERIVED_MIXED, so a user's own `orbit.<n>.chord`
        wins and a user's `orbit.<n>.cosi` -- what a params file written for the
        conventional parameterization carries -- reaches the chord fit as the
        start it implies.  The reverse is deliberately absent: where cos i is
        sampled the chord is a REPORTED output, and where the chord is sampled
        a user's seed lands on it directly.
        """
        orbit_cfgs = (system_config or {}).get("orbit") or []
        if not isinstance(orbit_cfgs, list):
            orbit_cfgs = [orbit_cfgs]
        primary, companion = parse_orbit_bodies(orbit_cfgs, system_config)
        planets = [
            idx for (ctype, idx) in companion[index] if ctype == "planet"
        ]
        if len(planets) != 1:
            # No radius ratio, or no single one: the chord is INACTIVE on this
            # orbit (Orbit.INC_MODE_TABLE), so there is nothing to seed.
            raise KeyError(f"orbit.{index} has no single transiting planet")
        j = planets[0]

        def need(path):
            val = resolved.get(path)
            if val is None:
                raise KeyError(f"Missing {path} for orbit.{index}.chord")
            return float(val)

        p_ratio = need(f"planet.{j}.p")
        ar = need(f"planet.{j}.ar")
        cosi_val = need(f"orbit.{index}.cosi")
        ecc_val = need(f"orbit.{index}.ecc")
        esinw = need(f"orbit.{index}.esinw")

        b = ar * cosi_val * (1.0 - ecc_val**2) / (1.0 + esinw)
        radicand = (1.0 + p_ratio) ** 2 - b**2
        if radicand <= 0.0:
            # cos i puts the companion off the stellar disc: there is no chord.
            # Refusing (rather than returning 0, a grazing transit) leaves the
            # defaults.yaml start in place and says why in the log.
            raise KeyError(
                f"orbit.{index}: b = {b:.3g} exceeds 1 + R_P/R_* = "
                f"{1.0 + p_ratio:.3g}, so this geometry has no transit chord"
            )
        return float(np.sqrt(radicand))

    config_manager.register_custom_solver(
        "orbit.chord", chord_solver, standalone=True
    )
