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

This module also owns the KNOWN-KEY sets for the three blocks and refuses
an unknown or misplaced key, naming the block it belongs on now.  Ruling R3
(no backward compatibility) forbids ACCEPTING an old spelling, not
diagnosing it: the migration mistake this catches is keeping the old
event-level options (finite_source, t0_par, fit* flags...) on a `lens:`
entry that now carries `body:` -- pre-fix that built silently with
finite_source=False and every flag ignored.  The sets are pinned against
each component's ``config_schema()`` by
tests/test_mulens_body_config.py, so a new schema key cannot silently
diverge from the gate here.
"""

# ---------------------------------------------------------------------------
# The known-key vocabulary, one frozenset per block.  `name` is universal
# (Component.__init__ reads it, and derive_body_names writes it).
# ---------------------------------------------------------------------------

EVENT_KEYS = frozenset(
    {
        "name",
        "finite_source",
        "t0_par",
        "backend",
        "mag_method",
        "use_op",
        "mmexofast",
        "mmexofast_options",
        "fitmurel",
        "fitpirel",
        "fitthetae",
        "source_orbital_motion",
        "source_orbit",
    }
)
LENS_ENTRY_KEYS = frozenset({"body", "name", "orbital_motion", "orbit"})
SOURCE_ENTRY_KEYS = frozenset(
    {"body", "name", "fitu0te", "star_constrains_rho"}
)

# Where a recognizably misplaced key lives now.  Keys in none of these maps
# get the generic refusal (typo, or a parameter that belongs in the params
# file).
_KEY_HOME = {}
for _k in EVENT_KEYS - {"name"}:
    _KEY_HOME[_k] = "the 'mulensevent:' block"
for _k in ("fitu0te", "star_constrains_rho"):
    _KEY_HOME[_k] = "the 'source:' entry of the source body it applies to"
for _k in ("orbital_motion", "orbit"):
    _KEY_HOME[_k] = (
        "the 'lens:' entry of the companion whose geometry it moves"
    )
for _k in ("lens_ndx", "source_ndx", "lenses", "sources"):
    _KEY_HOME[_k] = (
        "nowhere -- it is a pre-v0.1.0 spelling: the lens block was split "
        "into `mulensevent:` (event options), `lens:` (one entry per lens "
        "body) and `source:` (one entry per source body)"
    )


def reject_unknown_keys(entry, allowed, where):
    """Refuse any key outside ``allowed``, saying where each one lives now.

    A hard break must fail loudly and instructively (R3): a misplaced key
    silently ignored is a config the user believes is in effect and is not
    -- finite_source left on a lens entry built with finite_source=False and
    no diagnostic until this gate existed.
    """
    unknown = sorted(set(entry or {}) - set(allowed))
    if not unknown:
        return
    lines = []
    for k in unknown:
        home = _KEY_HOME.get(k)
        if home is not None:
            lines.append(f"  - '{k}' belongs on {home}")
        else:
            lines.append(
                f"  - '{k}' is not a config key of this block (a parameter "
                f"start value or prior belongs in the params file, e.g. "
                f"'{where.split('.')[0]}.<instance>.{k}:')"
            )
    raise ValueError(
        f"{where}: unknown config key(s) {unknown}.\n"
        + "\n".join(lines)
        + f"\n  Keys this block accepts: {sorted(allowed)}."
    )


def validate_event_config(block):
    """Enforce the known-key set on every ``mulensevent:`` entry."""
    for i, entry in enumerate(block or []):
        if isinstance(entry, dict):
            reject_unknown_keys(entry, EVENT_KEYS, f"mulensevent.{i}")
    return block


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

    Every entry must carry the ``body:`` key and NOTHING outside the
    block's known-key set.  An entry without ``body:`` is diagnosed by
    name: the pre-0.1.0 spellings (``lens_ndx``, ``source_ndx``,
    ``lenses:``, ``sources:`` and event-level options on the lens block)
    are a hard break (ruling R3), and the message says what the new shape
    is rather than leaving a KeyError.  An entry WITH ``body:`` but
    carrying extra keys is the other half of the same migration mistake
    (keep the old lens block, add ``body:``) and is refused just as
    loudly -- see ``reject_unknown_keys``.
    """
    allowed = LENS_ENTRY_KEYS if comp_key == "lens" else SOURCE_ENTRY_KEYS
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
        reject_unknown_keys(entry, allowed, where)
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
