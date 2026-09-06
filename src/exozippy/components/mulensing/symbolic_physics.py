import sympy as sp

from ...constants import DAYS_PER_YEAR, KAPPA, RSUN_TO_AU

# 1. Define all possible symbols
# These MUST match the strings produced by ConfigManager.finalize_user_params
t_0, u_0, t_E = sp.symbols("t_0 u_0 t_E")
theta_E, mu_rel_mag = sp.symbols("theta_E mu_rel_mag")
pi_rel = sp.symbols("pi_rel")
# lens_mass_total drives theta_E/t_E/rho/pi_E (community convention: binary-lens
# parameters are referenced to the TOTAL lens mass).  For single lenses it maps
# directly to the primary star's mass; for binaries it maps to
# mulensevent.0.mlens_total and the mass-sum relation below ties it to the
# per-body masses.
lens_mass_total, primary_lens_mass = sp.symbols(
    "lens_mass_total primary_lens_mass"
)
lens_distance, source_distance = sp.symbols("lens_distance source_distance")
mu_ra_rel, mu_dec_rel = sp.symbols("mu_ra_rel mu_dec_rel")
lens_pm_ra, source_pm_ra = sp.symbols("lens_pm_ra source_pm_ra")
lens_pm_dec, source_pm_dec = sp.symbols("lens_pm_dec source_pm_dec")
pi_E_N, pi_E_E = sp.symbols("pi_E_N pi_E_E")
rho, source_radius = sp.symbols("rho source_radius")
log_rho = sp.symbols("log_rho", real=True)
log_pi_rel = sp.symbols("log_pi_rel", real=True)
u0te = sp.symbols("u0te", real=True)
log_theta_E = sp.symbols("log_theta_E", real=True)
q_lens, companion_mass = sp.symbols("q_lens companion_mass")
alpha, xalpha, yalpha = sp.symbols("alpha xalpha yalpha")
# Projected separation: log_s is sampled, s is derived (s = 10**log_s).  The
# relation lets the relaxation engine translate a user-supplied lens.s initval
# (and its init_scale, via the Jacobian) into a log_s start, exactly as
# mass/logmass does in the star component.
s, log_s = sp.symbols("s log_s", real=True)

comp_key = "mulensevent"


def _resolve_ref(ref, system_config, where):
    """'star.Lens' / 'planet.1' -> (comp_type, index), resolved against the
    raw config.  Local twin of bodies.resolve_body_ref: this module is
    loaded standalone by ConfigManager's rglob walk (spec_from_file_location,
    no package context), so it cannot use a relative import."""
    parts = str(ref).split(".")
    if len(parts) != 2:
        raise ValueError(
            f"{where}: invalid body reference '{ref}': expected "
            f"'<component>.<name-or-index>'."
        )
    comp_type, inst = parts
    entries = (system_config or {}).get(comp_type) or []
    if inst.isdigit():
        return comp_type, int(inst)
    names = [e.get("name") if isinstance(e, dict) else None for e in entries]
    if inst in names:
        return comp_type, names.index(inst)
    raise ValueError(
        f"{where}: body reference '{ref}' names no '{comp_type}' instance."
    )


def get_symbol_map(event_cfg, system_config):
    """Symbol maps for the mulensevent/lens/source split (8.6.17 stage 1).

    ``event_cfg`` is the mulensevent block's (single) entry;
    ``system_config`` is the WHOLE parsed config, which this builder needs
    because the lens and source BODY LISTS live on their own components'
    blocks -- the change get_symbol_map's two-argument contract (stage 1a)
    exists for.

    Returns a LIST of symbol maps, one per SOURCE INSTANCE: each source has
    its own trajectory offsets (t_0, u_0, rho and their coordinates) at
    source.<j>.<param>, while the EVENT CHAIN (t_E, theta_E, pi_rel, pi_E,
    mu_rel) maps to mulensevent.0.<param> IDENTICALLY in every per-source
    map -- so the shared relations dedup to ONE instance (ConfigManager
    collapses identical relation instances), which is ruling R2's collapse
    happening in the engine for free.  Companion symbols map to LENS
    ELEMENT 1 (the first companion; element 0 is the masked primary), and
    only when a companion exists -- the pre-split map registered the
    alpha/xalpha/yalpha paths for point lenses too, phantom leaf symbols
    for parameters no manifest declared.
    """
    lens_block = (system_config or {}).get("lens") or []
    source_block = (system_config or {}).get("source") or []
    if not lens_block or not source_block:
        # An event block without its body lists cannot map anything; the
        # component constructors raise the user-facing error.
        return []

    lens_bodies = [
        _resolve_ref(e.get("body"), system_config, f"lens.{i}")
        for i, e in enumerate(lens_block)
        if isinstance(e, dict) and e.get("body") is not None
    ]
    source_bodies = [
        _resolve_ref(e.get("body"), system_config, f"source.{i}")
        for i, e in enumerate(source_block)
        if isinstance(e, dict) and e.get("body") is not None
    ]
    if not lens_bodies or not source_bodies:
        return []

    _, l_idx = lens_bodies[0]
    is_binary_lens = len(lens_bodies) > 1

    companion_mass_path = None
    if len(lens_bodies) == 2:
        c_comp, c_idx = lens_bodies[1]
        companion_mass_path = f"{c_comp}.{c_idx}.mass"
    # 3+ bodies: the binary mass-sum/q relations cannot represent the
    # extra companions, so companion_mass stays unregistered (both
    # relations go inert) and MulensEvent.register_parameters seeds
    # mulensevent.mlens_total from the per-body mass initvals instead.

    # theta_E/t_E/rho/pi_E are referenced to the TOTAL lens mass: the
    # primary star's mass for a single lens, the derived
    # mulensevent.0.mlens_total for a binary (tied to the per-body masses
    # by the mass-sum relation).
    if is_binary_lens:
        lens_mass_total_path = "mulensevent.0.mlens_total"
    else:
        lens_mass_total_path = f"star.{l_idx}.mass"

    # In keplerian orbital-motion mode NONE of the geometry symbols
    # (alpha's arctan2 pair, s <-> log_s) map: the geometry is derived
    # from the referenced orbit and no sampled coordinate exists for the
    # engine to seed (conventions.md C24).  The key lives on the COMPANION
    # entries now.
    keplerian = any(
        isinstance(e, dict) and e.get("orbital_motion") == "keplerian"
        for e in lens_block[1:]
    )

    maps = []
    for j, (s_comp, s_idx) in enumerate(source_bodies):
        src_cfg = source_block[j] if isinstance(source_block[j], dict) else {}
        result = {
            # Per-source trajectory offsets: element j of the source
            # component's vectors.
            "t_0": f"source.{j}.t_0",
            "u_0": f"source.{j}.u_0",
            "rho": f"source.{j}.rho",
            # The event chain, mapped ONCE (identical strings in every
            # per-source map; relations sharing only these dedup).
            "t_E": "mulensevent.0.t_E",
            "theta_E": "mulensevent.0.theta_E",
            "pi_rel": "mulensevent.0.pi_rel",
            "pi_E_N": "mulensevent.0.pi_E_N",
            "pi_E_E": "mulensevent.0.pi_E_E",
            "mu_rel_mag": "mulensevent.0.mu_rel_mag",
            "mu_ra_rel": "mulensevent.0.mu_ra_rel",
            "mu_dec_rel": "mulensevent.0.mu_dec_rel",
            "lens_mass_total": lens_mass_total_path,
            "lens_distance": f"star.{l_idx}.distance",
            "lens_pm_ra": f"star.{l_idx}.pm_ra",
            "lens_pm_dec": f"star.{l_idx}.pm_dec",
            "lens_ra": f"star.{l_idx}.ra",
            "lens_dec": f"star.{l_idx}.dec",
            "source_mass": f"{s_comp}.{s_idx}.mass",
            "source_radius": f"{s_comp}.{s_idx}.radius",
            "source_distance": f"{s_comp}.{s_idx}.distance",
            "source_pm_ra": f"{s_comp}.{s_idx}.pm_ra",
            "source_pm_dec": f"{s_comp}.{s_idx}.pm_dec",
            "source_ra": f"{s_comp}.{s_idx}.ra",
            "source_dec": f"{s_comp}.{s_idx}.dec",
        }

        if companion_mass_path:
            result["companion_mass"] = companion_mass_path
            result["primary_lens_mass"] = f"star.{l_idx}.mass"

        # Companion geometry: LENS ELEMENT 1 (companion slot 0; element 0
        # is the masked primary -- no primary-element path may ever appear
        # here, or a seed would target a warn-dropped bookkeeping pin).
        # For 3+ body lenses only element 1 is covered by the relations;
        # MulensEvent.register_parameters seeds the remaining companions
        # from user body-mass/s hints.
        if is_binary_lens and not keplerian:
            result["q_lens"] = "lens.1.q"
            result["alpha"] = "lens.1.alpha"
            result["xalpha"] = "lens.1.xalpha"
            result["yalpha"] = "lens.1.yalpha"
            result["s"] = "lens.1.s"
            result["log_s"] = "lens.1.log_s"
        elif is_binary_lens:
            result["q_lens"] = "lens.1.q"

        # log_rho exists only where THIS source severs the stellar tie
        # (star_constrains_rho: false, a per-instance flag now); the
        # relation stays inert otherwise, exactly like s/log_s for PSPL.
        if not src_cfg.get("star_constrains_rho", True):
            result["log_rho"] = f"source.{j}.log_rho"

        # u0te exists only where THIS source sets fitu0te (swap 4).
        if src_cfg.get("fitu0te"):
            result["u0te"] = f"source.{j}.u0te"

        # log_pi_rel exists only under fitpirel (swap 2), an event flag.
        if event_cfg.get("fitpirel"):
            result["log_pi_rel"] = "mulensevent.0.log_pi_rel"

        # log_theta_E exists only under fitthetae (swap 3), an event flag.
        if event_cfg.get("fitthetae"):
            result["log_theta_E"] = "mulensevent.0.log_theta_E"

        maps.append(result)

    return maps


RELATIONS = [
    # Einstein Radius (total lens mass)
    sp.Eq(theta_E**2, KAPPA * lens_mass_total * pi_rel),
    # Relative Parallax (dist in pc -> pi in mas)
    sp.Eq(pi_rel, (1000 / lens_distance) - (1000 / source_distance)),
    # Einstein Time (mu in mas/yr -> t_E in days).  SEEDING APPROXIMATION:
    # the runtime graph derives t_E from mu_rel_GEO (= mu_rel_helio -
    # pi_rel * v_earth_perp(t0_par)/AU, Gould 2004), but t0_par and Earth's
    # velocity are resolved at stage 1 -- after these relations are
    # constructed -- so the engine seeds through the heliocentric value.
    # Starts land a few percent off for large-pi_rel events; the samplers
    # absorb that.  (MMEXOFAST t_E seeds are geocentric, so the back-solved
    # pms are helio-approximate too.)
    sp.Eq(t_E, theta_E / (mu_rel_mag / DAYS_PER_YEAR)),
    # Relative Motion Magnitude
    sp.Eq(mu_rel_mag**2, mu_ra_rel**2 + mu_dec_rel**2),
    # Proper Motion Vector Components
    sp.Eq(mu_ra_rel, lens_pm_ra - source_pm_ra),
    sp.Eq(mu_dec_rel, lens_pm_dec - source_pm_dec),
    # Parallax Vector Components
    sp.Eq(pi_E_N, (pi_rel / theta_E) * (mu_dec_rel / mu_rel_mag)),
    sp.Eq(pi_E_E, (pi_rel / theta_E) * (mu_ra_rel / mu_rel_mag)),
    # Derived shortcut: pi_rel = kappa * mass * |pi_E|^2
    # (obtained by eliminating theta_E from the Einstein-radius and pi_E-magnitude
    # equations: |pi_E|^2 = (pi_rel/theta_E)^2 and theta_E^2 = kappa*mass*pi_rel).
    # This gives the solver a direct rank-100 path when mass and pi_E are both
    # user-supplied, bypassing the distance hint and avoiding sign ambiguity in
    # the quadratic for mu_ra_rel / mu_dec_rel.
    sp.Eq(pi_rel, KAPPA * lens_mass_total * (pi_E_N**2 + pi_E_E**2)),
    # Finite Source (R_sun to AU, then to mas)
    sp.Eq(
        rho,
        ((source_radius * RSUN_TO_AU / source_distance) * 1000.0) / theta_E,
    ),
    # Projected separation reparameterization (base-10, mirrors mass/logmass).
    # Only active for binary lenses (s/log_s mapped in get_symbol_map there);
    # inert for PSPL.  Lets user lens.s initvals back-solve to a log_s start.
    sp.Eq(s, 10**log_s),
    # rho reparameterization, active only when the lens severs the
    # stellar tie (`star_constrains_rho: false`)
    # (log_rho mapped in get_symbol_map there; inert otherwise).  Lets a
    # user or MMEXOFAST rho seed back-solve to a log_rho start.
    sp.Eq(rho, 10**log_rho),
    # pi_rel reparameterization, active only for `fitpirel: true` lenses
    # (swap 2); inert otherwise.  Lets the distance-derived pi_rel seed
    # back-solve to a log_pi_rel start.
    sp.Eq(pi_rel, 10**log_pi_rel),
    # signed effective-timescale reparameterization, active only for
    # `fitu0te: true` (swap 4); inert otherwise.
    sp.Eq(u0te, u_0 * t_E),
    # theta_E reparameterization, active only for `fitthetae: true` (swap
    # 3); inert otherwise.
    sp.Eq(theta_E, 10**log_theta_E),
    # Binary lens mass ratio: q = M_companion / M_primary
    # companion_mass/primary_lens_mass are only in the symbol map for binary
    # events, so these relations are automatically inert for PSPL (relaxation
    # engine skips equations with unregistered symbols).  Propagates:
    # user-supplied q → companion mass initval, or known masses → q.
    sp.Eq(q_lens * primary_lens_mass, companion_mass),
    # Total lens mass = sum of body masses (binary only; inert for PSPL where
    # lens_mass_total maps directly onto the primary star's mass).
    sp.Eq(lens_mass_total, primary_lens_mass + companion_mass),
    # Source trajectory angle: alpha (radians, internal) → xalpha, yalpha.
    # xalpha = r·cos(alpha), yalpha = r·sin(alpha), where r is a free positive
    # scale sampled from the N(0,1) prior — only the direction arctan2(y,x) matters.
    # Wide bounds (±100) and N(0,1) priors give a uniform marginal prior on alpha;
    # bounding to [-1,1] would break isotropy and bias angles near ±45°.
    # The relaxation engine uses these only forward (alpha → xalpha, yalpha):
    # given alpha, set xalpha=cos(alpha), yalpha=sin(alpha) as unit-circle seeds.
    # mkparam converts the sampled xalpha/yalpha back to alpha via arctan2.
    # alpha itself is not in the symbol map for PSPL events (no xalpha/yalpha
    # registered), so both relations are automatically inert for point-source fits.
    sp.Eq(xalpha, sp.cos(alpha)),
    sp.Eq(yalpha, sp.sin(alpha)),
]
