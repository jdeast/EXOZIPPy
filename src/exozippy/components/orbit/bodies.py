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

When neither key is given, the legacy implicit topology is used: the
companion group is the planet(s) whose ``orbit_ndx`` points at this orbit,
and the primary group is those planets' host star(s).

This module is intentionally lightweight (no pymc/pytensor imports) so the
symbolic_physics module can share the same parser for the relaxation-engine
custom solver.
"""


def component_instance_names(system_config, comp_key):
    """Instance names of a component section, mirroring Component.names."""
    section = system_config.get(comp_key) or []
    if not isinstance(section, list):
        section = [section]
    return [c.get("name", str(i)) if isinstance(c, dict) else str(i)
            for i, c in enumerate(section)]


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
                f"'star' or 'planet'.")
        names = star_names if comp == "star" else planet_names
        if tail.isdigit():
            idx = int(tail)
            if idx >= len(names):
                raise ValueError(
                    f"{context}body reference '{ref}' is out of range "
                    f"({comp} has {len(names)} instance(s)).")
        elif tail in names:
            idx = names.index(tail)
        else:
            raise ValueError(
                f"{context}body reference '{ref}' not found; {comp} "
                f"instances are {names}.")
        return (comp, idx)

    hits = []
    if s in star_names:
        hits.append(("star", star_names.index(s)))
    if s in planet_names:
        hits.append(("planet", planet_names.index(s)))
    if not hits:
        raise ValueError(
            f"{context}body '{ref}' does not match any star or planet "
            f"instance name (stars: {star_names}, planets: {planet_names}).")
    if len(hits) > 1:
        raise ValueError(
            f"{context}body '{ref}' is ambiguous (both a star and a planet "
            f"use that name); qualify it as 'star.{ref}' or 'planet.{ref}'.")
    return hits[0]


def _as_list(val):
    return list(val) if isinstance(val, (list, tuple)) else [val]


def parse_orbit_bodies(orbit_cfgs, system_config):
    """
    Parse primary/companion body groups for every orbit config entry.

    Returns (primary_bodies, companion_bodies): two lists (one entry per
    orbit) of lists of (comp_type, index) tuples.
    """
    system_config = system_config or {}
    star_names = component_instance_names(system_config, "star")
    planet_names = component_instance_names(system_config, "planet")
    planet_cfgs = system_config.get("planet") or []
    if not isinstance(planet_cfgs, list):
        planet_cfgs = [planet_cfgs]

    primary_bodies, companion_bodies = [], []
    for i, cfg in enumerate(orbit_cfgs):
        cfg = cfg or {}
        name = cfg.get("name", str(i))
        ctx = f"orbit '{name}': "
        if "bodies" in cfg:
            raise ValueError(
                f"{ctx}'bodies' is not a supported key; declare the two "
                f"sides explicitly with 'primary:' and 'companion:'.")

        prim_cfg = cfg.get("primary")
        comp_cfg = cfg.get("companion")
        if prim_cfg is None and comp_cfg is None:
            # Legacy implicit topology: companion = planets pointing here,
            # primary = their host stars.
            comp_b = [("planet", j) for j, p in enumerate(planet_cfgs)
                      if int((p or {}).get("orbit_ndx", 0)) == i]
            prim_b = []
            for _, j in comp_b:
                host = ("star", int((planet_cfgs[j] or {}).get("star_ndx", 0)))
                if host not in prim_b:
                    prim_b.append(host)
            if not comp_b:
                # No planet references this orbit (standalone/test usage or
                # a sparse config): fall back to the historical one-to-one
                # pairing.  Validated against the live system at stage 2.
                comp_b = [("planet", i)]
                prim_b = [("star", 0)]
        elif prim_cfg is None or comp_cfg is None:
            raise ValueError(
                f"{ctx}give both 'primary:' and 'companion:' (or neither, "
                f"for the implicit planet-orbit pairing).")
        else:
            prim_b = [parse_body_ref(r, star_names, planet_names, ctx)
                      for r in _as_list(prim_cfg)]
            comp_b = [parse_body_ref(r, star_names, planet_names, ctx)
                      for r in _as_list(comp_cfg)]

        if not prim_b or not comp_b:
            raise ValueError(f"{ctx}primary and companion groups must each "
                             f"contain at least one body.")
        overlap = set(prim_b) & set(comp_b)
        if overlap:
            raise ValueError(f"{ctx}body/bodies {sorted(overlap)} appear in "
                             f"both the primary and companion groups.")
        if len(set(prim_b)) != len(prim_b) or len(set(comp_b)) != len(comp_b):
            raise ValueError(f"{ctx}duplicate body in a group.")

        primary_bodies.append(prim_b)
        companion_bodies.append(comp_b)

    return primary_bodies, companion_bodies
