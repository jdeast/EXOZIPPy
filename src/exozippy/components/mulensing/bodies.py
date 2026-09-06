"""Body references for the microlensing components (8.6.17 stage 1).

The `lens:` and `source:` components list one PHYSICAL BODY per entry:

    lens:
      - body: star.Lens        # entry 0 is ALWAYS the primary
      - body: planet.b         # companions may be planets or stars
    source:
      - body: star.SourceA

`body:` is a path-valued, type-agnostic reference `<component>.<name-or-
index>` (JDE ruling 2026-09, closing design risk 7): a lens body may be a
star or a planet, and the component does not grow a key per type.  The
helpers here resolve that spelling against the raw system config, so every
consumer -- the three component constructors and the symbol-map builder --
shares one parser and one error message.
"""


def resolve_body_ref(ref, system_config, where):
    """Resolve a `body:` reference to ``(comp_type, index)``.

    Accepts ``star.Lens`` (name form) and ``star.0`` (index form).  The
    component must exist in the config and the instance must be resolvable;
    errors name the available instances so a typo is a one-line fix.
    """
    parts = str(ref).split(".")
    if len(parts) != 2 or not parts[0]:
        raise ValueError(
            f"{where}: invalid body reference '{ref}': expected "
            f"'<component>.<name-or-index>', e.g. 'star.Lens', 'star.0' "
            f"or 'planet.b'."
        )
    comp_type, inst = parts
    entries = (system_config or {}).get(comp_type)
    if not isinstance(entries, list):
        raise ValueError(
            f"{where}: body reference '{ref}' names component "
            f"'{comp_type}', but no '{comp_type}' block exists in the "
            f"system config."
        )
    if inst.isdigit():
        ndx = int(inst)
        if ndx >= len(entries):
            raise ValueError(
                f"{where}: body reference '{ref}' is out of range: only "
                f"{len(entries)} '{comp_type}' instance(s) are configured."
            )
        return (comp_type, ndx)
    names = [e.get("name") if isinstance(e, dict) else None for e in entries]
    if inst in names:
        return (comp_type, names.index(inst))
    declared = [n for n in names if n is not None]
    raise ValueError(
        f"{where}: body reference '{ref}' names no '{comp_type}' "
        f"instance.  Declared names: {declared or '(none)'}; an index "
        f"('{comp_type}.<i>') also works."
    )


def body_entries(block, comp_key, system_config):
    """``[(comp_type, index), ...]`` for one component's config block.

    Every entry must carry exactly the ``body:`` key.  An entry without one
    is diagnosed by name: the pre-0.1.0 spellings (``lens_ndx``,
    ``source_ndx``, ``lenses:``, ``sources:`` and event-level options on the
    lens block) are a hard break (ruling R3), and the message says what the
    new shape is rather than leaving a KeyError.
    """
    out = []
    for i, entry in enumerate(block or []):
        where = f"{comp_key}.{i}"
        if not isinstance(entry, dict) or "body" not in entry:
            legacy = sorted(
                set(entry or {})
                & {"lens_ndx", "source_ndx", "lenses", "sources"}
            )
            hint = (
                f"  (Found pre-v0.1.0 key(s) {legacy}: the lens block was "
                f"split into `mulensevent:` [the event options], `lens:` "
                f"[one entry per lens body] and `source:` [one entry per "
                f"source body].)"
                if legacy
                else ""
            )
            raise ValueError(
                f"{where}: every '{comp_key}:' entry is one physical body "
                f"and must carry a `body:` reference, e.g.\n"
                f"  {comp_key}:\n"
                f"    - body: star.Lens\n"
                f"Event-level options (finite_source, t0_par, backend, "
                f"mag_method, use_op, mmexofast, fit* flags, "
                f"source_orbital_motion) live on the `mulensevent:` block."
                f"{hint}"
            )
        out.append(resolve_body_ref(entry["body"], system_config, where))
    return out


def derive_body_names(block):
    """Set each entry's ``name:`` from its ``body:`` ref (in place).

    The Mann/Torres idiom: the instance is named after the body it wraps
    (the trailing segment of the ref), so a user's ``source.SourceA.t_0``
    or ``lens.LensB.log_s`` folds to index form at ConfigManager
    construction.  Called from ``normalize_config_block``, which System runs
    BEFORE the ConfigManager exists -- the timing is the point (see
    relations.StellarRelation.normalize_config_block).
    """
    for c in block or []:
        if (
            isinstance(c, dict)
            and c.get("name") is None
            and c.get("body") is not None
        ):
            c["name"] = str(c["body"]).split(".")[-1]
    return block
