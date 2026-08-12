PHYSICS_REGISTRY = {}


def register_physics(func):
    """Decorator to register physics functions as they are imported.

    The registry is a flat namespace keyed by function name -- `func_name:` in
    a component's defaults.yaml is looked up here with no component scoping --
    so two components registering the same name silently shadow each other,
    last import wins. That is not hypothetical: star and planet both defined
    `calc_logg` with different signatures (logmass vs linear mass), and
    planet's won. star.logg was therefore computed as
    `LOGG_CONST + log10(logmass) - 2*log10(radius)`: wrong for every star, and
    NaN for any star below 1 solMass. Nothing consumed star.logg, so it went
    unnoticed until components/torres needed it.

    Registering a duplicate name is now an error rather than a silent
    shadowing. If two components genuinely need the same physics, give it one
    owner and import it (see components/planet/physics.py's calc_density).
    """
    name = func.__name__
    existing = PHYSICS_REGISTRY.get(name)
    if existing is not None and existing is not func:
        raise ValueError(
            f"Duplicate physics function '{name}': already registered by "
            f"{existing.__module__}, now also by {func.__module__}. "
            f"PHYSICS_REGISTRY is a flat namespace shared by every component's "
            f"defaults.yaml 'func_name:', so the second registration would "
            f"silently shadow the first. Rename one to say what it takes (e.g. "
            f"calc_logg_from_logmass vs calc_logg_from_mass), or give the "
            f"shared implementation a single owner and import it."
        )
    PHYSICS_REGISTRY[name] = func
    return func
