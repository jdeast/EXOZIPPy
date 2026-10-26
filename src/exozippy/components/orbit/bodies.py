"""
Body-group parsing for hierarchical orbits.

An orbit is a two-body Keplerian arc between a "primary" group and a
"companion" group of physical bodies (stars and/or planets).  Groups are
declared in the orbit's YAML block:

    orbit:
      - name: "b"
        primary: ["A"]          # instance names, or "star.A"/"planet.0" paths
        companion: ["b"]
      - name: "BC"
        primary: ["C"]
        companion: ["B"]
      - name: "A-BC"
        primary: ["A", "b"]     # a group is treated as a point mass at its
        companion: ["B", "C"]   # barycenter (hierarchical approximation)

MEMBERSHIP HAS ONE SPELLING: these groups.  A planet's orbit and its host
star are DERIVED from them (`planet_membership`); the planet keys that used
to say the same thing a second time (`orbit_ndx:`, `star_ndx:`) are removed
and raise with the migration (`REMOVED_PLANET_KEYS`).  A planet with no
dynamics -- a static microlensing lens companion -- is paired by a
`type: static` orbit, which names its groups like any other and declares no
parameters.  Two unambiguous pairings may be left implicit (see
`parse_orbit_bodies` and `planet_membership`); anything else raises and asks
for the orbit block.

This module is intentionally lightweight (no pymc/pytensor imports) so the
symbolic_physics module can share the same parser for the relaxation-engine
custom solver.
"""

# Orbit `type:` values whose orbit is a low-order Taylor expansion about a
# reference epoch rather than a Keplerian (see orbit.md "Taylor orbits").
TAYLOR_TYPES = ("linear", "quadratic")

# The orbit `type:` of a PAIRING WITH NO DYNAMICS (orbit.md "Planet membership has one spelling"):
# it names a primary and a companion group like any orbit block -- which is
# how a planet that moves on nothing (a static lens companion) gets its host
# -- declares no parameters, and nothing reads geometry from it.
STATIC_TYPE = "static"


def _removed_membership_key(key):
    def message(value):
        return (
            f"'{key}:' on a planet was removed: a planet's orbit and its "
            f"host star are "
            f"declared ONCE, by name, in an orbit block's primary:/companion: "
            f"groups, e.g.\n"
            f"  orbit:\n"
            f"    - name: b\n"
            f"      primary: [A]\n"
            f"      companion: [b]\n"
            f"A planet with no orbital dynamics (a static microlensing lens "
            f"companion) is paired by a 'type: static' orbit block.  Delete "
            f"'{key}: {value}'."
        )

    return message


# Planet keys that were a SECOND spelling of the orbit groups, refused with
# their migration (the mulensing bodies.REMOVED_KEYS pattern).  Owned here, not
# by Planet, because every reader of planet membership -- the planet, the
# orbit, the planet symbol map, transit, astrometry -- goes through
# `planet_membership`, which may run before the Planet component exists.
REMOVED_PLANET_KEYS = {
    key: _removed_membership_key(key) for key in ("orbit_ndx", "star_ndx")
}


def orbit_types(system_config):
    """Each orbit block's `type:` (default "keplerian"), in block order.

    Read off the raw config so a component built BEFORE the orbit (the lens
    resolves its companion's `orbit:` at construction) asks the same
    question the Orbit itself answers.  Validation of the value is the
    Orbit's (`Orbit._parse_types`); this only reads it.
    """
    section = (system_config or {}).get("orbit") or []
    if not isinstance(section, list):
        section = [section]
    return [
        (c or {}).get("type", "keplerian")
        if isinstance(c, dict)
        else "keplerian"
        for c in section
    ]


def component_instance_names(system_config, comp_key):
    """Instance names of a component section, mirroring Component.names."""
    section = system_config.get(comp_key) or []
    if not isinstance(section, list):
        section = [section]
    return [
        c.get("name", str(i)) if isinstance(c, dict) else str(i)
        for i, c in enumerate(section)
    ]


def parse_body_ref(ref, star_names, planet_names, context=""):
    """
    Resolve one body reference to (comp_type, index).

    Accepted forms: a bare instance name ("A", "b"; must be unambiguous
    across stars and planets), or an explicit path ("star.A", "star.0",
    "planet.b", "planet.0").
    """
    s = str(ref)
    if "." in s:
        comp, tail = s.split(".", 1)
        if comp not in ("star", "planet"):
            raise ValueError(
                f"{context}invalid body reference '{ref}': component must be "
                f"'star' or 'planet'."
            )
        names = star_names if comp == "star" else planet_names
        if tail.isdigit():
            idx = int(tail)
            if idx >= len(names):
                raise ValueError(
                    f"{context}body reference '{ref}' is out of range "
                    f"({comp} has {len(names)} instance(s))."
                )
        elif tail in names:
            idx = names.index(tail)
        else:
            raise ValueError(
                f"{context}body reference '{ref}' not found; {comp} "
                f"instances are {names}."
            )
        return (comp, idx)

    hits = []
    if s in star_names:
        hits.append(("star", star_names.index(s)))
    if s in planet_names:
        hits.append(("planet", planet_names.index(s)))
    if not hits:
        raise ValueError(
            f"{context}body '{ref}' does not match any star or planet "
            f"instance name (stars: {star_names}, planets: {planet_names})."
        )
    if len(hits) > 1:
        raise ValueError(
            f"{context}body '{ref}' is ambiguous (both a star and a planet "
            f"use that name); qualify it as 'star.{ref}' or 'planet.{ref}'."
        )
    return hits[0]


def _as_list(val):
    return list(val) if isinstance(val, (list, tuple)) else [val]


def parse_orbit_bodies(orbit_cfgs, system_config):
    """
    Parse primary/companion body groups for every orbit config entry.

    Returns (primary_bodies, companion_bodies): two lists (one entry per
    orbit) of lists of (comp_type, index) tuples.

    A Keplerian orbit block may omit BOTH groups only when the pairing is
    unambiguous: the system declares exactly one orbit, one star and one
    planet, and the orbit then pairs them (the QoL default every
    one-planet transit/RV config relied on).  Anything else raises and asks
    for the groups -- it used to fall back to "the planets whose orbit_ndx
    points here", a second spelling of the same membership, and, failing
    that, to `planet.<i>` around `star.0` whether or not they existed.

    A system config that declares NO star and NO planet at all (a bare
    geometry harness: an orbit with nothing to orbit) keeps the historical
    placeholder pairing `planet.<i>` around `star.0`; `Orbit._validate_bodies`
    then finds those bodies absent and disables the orbit's masses.
    """
    system_config = system_config or {}
    star_names = component_instance_names(system_config, "star")
    planet_names = component_instance_names(system_config, "planet")
    n_orbits = len(orbit_cfgs or [])

    primary_bodies, companion_bodies = [], []
    for i, cfg in enumerate(orbit_cfgs):
        cfg = cfg or {}
        name = cfg.get("name", str(i))
        ctx = f"orbit '{name}': "
        if "bodies" in cfg:
            raise ValueError(
                f"{ctx}'bodies' is not a supported key; declare the two "
                f"sides explicitly with 'primary:' and 'companion:'."
            )

        prim_cfg = cfg.get("primary")
        comp_cfg = cfg.get("companion")
        otype = cfg.get("type", "keplerian")
        taylor = otype in TAYLOR_TYPES
        if otype == STATIC_TYPE and (prim_cfg is None or comp_cfg is None):
            # A static orbit exists only to say who is paired with whom, so
            # an implicit pairing would leave it saying nothing.
            raise ValueError(
                f"{ctx}a type: static orbit is a pairing and must name both "
                f"its 'primary:' and its 'companion:' group."
            )
        if taylor:
            # A Taylor orbit is a companion too long-period to resolve, and
            # usually one nothing else in the model names: its companion
            # group may be EMPTY (an unseen body), so the implicit planet
            # pairing below -- which would invent `planet.<i>` -- must not
            # run.  The primary group is the one that accelerates, and has
            # to be said.
            if prim_cfg is None:
                raise ValueError(
                    f"{ctx}a type: {cfg.get('type')} orbit needs an explicit "
                    f"'primary:' group (the bodies whose motion the Taylor "
                    f"terms describe); its 'companion:' may be omitted for "
                    f"an unseen companion."
                )
            prim_b = [
                parse_body_ref(r, star_names, planet_names, ctx)
                for r in _as_list(prim_cfg)
            ]
            comp_b = [
                parse_body_ref(r, star_names, planet_names, ctx)
                for r in _as_list(comp_cfg if comp_cfg is not None else [])
            ]
        elif prim_cfg is None and comp_cfg is None:
            if not star_names and not planet_names:
                # A bare geometry harness (no bodies declared anywhere):
                # the historical placeholder, whose absent bodies disable
                # the orbit's masses in Orbit._validate_bodies.
                comp_b = [("planet", i)]
                prim_b = [("star", 0)]
            elif (
                n_orbits == 1
                and len(star_names) == 1
                and len(planet_names) == 1
            ):
                # The one unambiguous implicit pairing.
                comp_b = [("planet", 0)]
                prim_b = [("star", 0)]
            else:
                raise ValueError(
                    f"{ctx}name the bodies this orbit pairs with 'primary:' "
                    f"and 'companion:' (e.g. primary: [{star_names[0] if star_names else 'A'}], "
                    f"companion: [{planet_names[0] if planet_names else 'b'}]).  "
                    f"An orbit block may leave both out only when the system "
                    f"has exactly one orbit, one star and one planet; this "
                    f"one has {n_orbits} orbit(s), {len(star_names)} star(s) "
                    f"and {len(planet_names)} planet(s)."
                )
        elif prim_cfg is None or comp_cfg is None:
            raise ValueError(
                f"{ctx}give both 'primary:' and 'companion:' (or neither, "
                f"for the implicit planet-orbit pairing)."
            )
        else:
            prim_b = [
                parse_body_ref(r, star_names, planet_names, ctx)
                for r in _as_list(prim_cfg)
            ]
            comp_b = [
                parse_body_ref(r, star_names, planet_names, ctx)
                for r in _as_list(comp_cfg)
            ]

        if not prim_b or (not comp_b and not taylor):
            raise ValueError(
                f"{ctx}primary and companion groups must each "
                f"contain at least one body."
            )
        overlap = set(prim_b) & set(comp_b)
        if overlap:
            raise ValueError(
                f"{ctx}body/bodies {sorted(overlap)} appear in "
                f"both the primary and companion groups."
            )
        if len(set(prim_b)) != len(prim_b) or len(set(comp_b)) != len(comp_b):
            raise ValueError(f"{ctx}duplicate body in a group.")

        primary_bodies.append(prim_b)
        companion_bodies.append(comp_b)

    return primary_bodies, companion_bodies


def _section(system_config, comp_key):
    section = (system_config or {}).get(comp_key) or []
    return section if isinstance(section, list) else [section]


def _lens_bodies(system_config, star_names, planet_names):
    """The (comp_type, index) bodies of the raw `lens:` block, in order.

    Read here rather than through `mulensing.bodies.body_entries` because
    this module must stay importable without the microlensing package (whose
    `__init__` imports this one).  An entry without `body:` is skipped: the
    lens component's own constructor refuses it with the migration message,
    and nothing here can say it better.
    """
    out = []
    for i, entry in enumerate(_section(system_config, "lens")):
        if not isinstance(entry, dict) or entry.get("body") is None:
            continue
        out.append(
            parse_body_ref(
                entry["body"], star_names, planet_names, f"lens.{i}: "
            )
        )
    return out


def planet_membership(system_config):
    """Per planet, in config order: ``(orbit index or None, host star index)``.

    THE one derivation of a planet's orbit and host (P3, one spelling): both
    come from the orbit blocks' primary:/companion: groups and from nothing
    else.  A planet's ORBIT is the orbit whose companion group names it; when
    several do, the innermost -- the one with the smallest companion group
    -- is its own (an outer orbit may treat `[A, b]` as one point mass), and
    two orbits tied for innermost RAISE, naming the planet and the orbits.
    Its HOST is the one star in that orbit's primary group; a primary group
    with no star or with several (a circumbinary planet) raises, because the
    planet's q = m_p/m_host and R_P/R_* need one host.

    A planet that no orbit names is paired IMPLICITLY, with no orbit at all
    (orbit index None: no dynamics, like an explicit `type: static` orbit),
    only when that is unambiguous: its scope holds exactly one star and no
    other planet.  The scope of a microlensing LENS body is the lens's own
    bodies (the source stars cannot host a lens companion); any other
    planet's is the whole system.  Every other unnamed planet raises and
    asks for the orbit block.  A leftover `orbit_ndx:` / `star_ndx:` key
    raises with its migration (`REMOVED_PLANET_KEYS`).

    Reads the raw system config, so any component may ask at any stage --
    the planet symbol map does at ConfigManager construction.
    """
    system_config = system_config or {}
    planet_cfgs = _section(system_config, "planet")
    planet_names = component_instance_names(system_config, "planet")
    star_names = component_instance_names(system_config, "star")
    for j, pc in enumerate(planet_cfgs):
        for key, message in REMOVED_PLANET_KEYS.items():
            if isinstance(pc, dict) and key in pc:
                raise ValueError(
                    f"[planet.{planet_names[j]}] {message(pc[key])}"
                )

    orbit_cfgs = _section(system_config, "orbit")
    orbit_names = component_instance_names(system_config, "orbit")
    if orbit_cfgs:
        prim, comp = parse_orbit_bodies(orbit_cfgs, system_config)
    else:
        prim, comp = [], []
    lens = _lens_bodies(system_config, star_names, planet_names)

    out = []
    for j, pname in enumerate(planet_names):
        key = ("planet", j)
        named = [o for o in range(len(comp)) if key in comp[o]]
        if named:
            innermost = min(len(comp[o]) for o in named)
            own = [o for o in named if len(comp[o]) == innermost]
            if len(own) > 1:
                raise ValueError(
                    f"[planet.{pname}] orbits "
                    f"{[orbit_names[o] for o in own]} all name this planet "
                    f"in a companion group of the same size, so none of "
                    f"them is its own orbit.  A planet moves on ONE orbit; "
                    f"an outer orbit may hold it only inside a larger group "
                    f"(e.g. companion: [A, {pname}])."
                )
            o = own[0]
            hosts = [idx for t, idx in prim[o] if t == "star"]
            if len(hosts) != 1:
                raise ValueError(
                    f"[planet.{pname}] its host is the star in the primary "
                    f"group of its orbit '{orbit_names[o]}', which holds "
                    f"{[star_names[h] for h in hosts] or 'no star'}.  A "
                    f"planet needs exactly one host star (its q = "
                    f"m_p/m_host and R_P/R_* are relative to it); a "
                    f"circumbinary host is not supported."
                )
            out.append((o, hosts[0]))
            continue

        if key in lens:
            scope, where = lens, "the lens"
        else:
            scope = [("star", i) for i in range(len(star_names))] + [
                ("planet", k) for k in range(len(planet_names))
            ]
            where = "the system"
        stars = [i for t, i in scope if t == "star"]
        planets = [k for t, k in scope if t == "planet"]
        if len(stars) == 1 and planets == [j]:
            out.append((None, stars[0]))
            continue
        raise ValueError(
            f"[planet.{pname}] no orbit block names this planet in a "
            f"'companion:' group, and its host is ambiguous: {where} holds "
            f"star(s) {[star_names[i] for i in stars]} and planet(s) "
            f"{[planet_names[k] for k in planets]}.  Add an orbit block "
            f"pairing it with its host -- 'type: static' for a planet with "
            f"no orbital dynamics (e.g. a microlensing lens companion that "
            f"does not move), e.g.\n"
            f"  orbit:\n"
            f"    - name: {pname}\n"
            f"      type: static\n"
            f"      primary: [{star_names[stars[0]] if stars else 'A'}]\n"
            f"      companion: [{pname}]"
        )
    return out
