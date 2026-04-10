PHYSICS_REGISTRY = {}

def register_physics(func):
    """Decorator to register physics functions as they are imported."""
    PHYSICS_REGISTRY[func.__name__] = func
    return func