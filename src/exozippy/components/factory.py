# src/exozippy/components/factory.py
import importlib
import inspect
import logging
from pathlib import Path

from .component import Component

logger = logging.getLogger(__name__)
# import ipdb

# Modules that failed to import during the last discover_components() sweep,
# keyed by the module's file stem -- which is also, for every component in the
# tree, its YAML key (lens.py -> `lens:`, sed.py -> `sed:`).  A tolerated
# ImportError is fine for a component nobody asked for, but if the user's
# config names that key the model would silently be built WITHOUT it: the
# key falls through System.__init__'s "does not match any registered
# component and will be ignored" warning, which reads like a typo and says
# nothing about the import.  Blocking VBMicrolensing (imported at module
# scope in mulensing/op.py, and not a declared dependency) turned a
# microlensing config into a star-only fit that ran to completion.
_IMPORT_FAILURES = {}


def import_failures():
    """Module stem -> (module_path, exception) for the last sweep."""
    return dict(_IMPORT_FAILURES)


def discover_components():
    """
    Scans the package recursively for Component subclasses and maps them
    to their lowercase class names. Also triggers local physics registration.
    """
    registry = {}
    _IMPORT_FAILURES.clear()

    # 1. Start at the components directory
    components_dir = Path(__file__).parent

    # 2. Use rglob to search recursively through all subfolders
    for file in components_dir.rglob("*.py"):
        # Skip infrastructure files
        if file.name in [
            "__init__.py",
            "component.py",
            "factory.py",
            "parameter.py",
        ]:
            continue

        # Construct the dynamic module path
        # e.g., star/star.py -> exozippy.components.star.star
        rel_path = file.relative_to(components_dir)
        module_path = "exozippy.components." + ".".join(
            rel_path.with_suffix("").parts
        )

        try:
            # 3. Import the component module
            module = importlib.import_module(module_path)

            # 5. Register any Component subclasses found
            for name, obj in inspect.getmembers(module, inspect.isclass):
                # Ensure it's a Component subclass, but NOT the base Component
                # itself and NOT an abstract intermediate base (e.g. the shared
                # Instrument base, which leaves Component's abstract methods
                # unimplemented so it is never instantiated as a component).
                if (
                    issubclass(obj, Component)
                    and obj is not Component
                    and not inspect.isabstract(obj)
                ):
                    key = getattr(obj, "yaml_key", name.lower())
                    registry[key] = obj

        except ImportError as e:
            # a developer might push an unused, broken component. that
            # shouldn't break the code -- but it MUST break the code if a
            # config actually asks for it.  System.__init__ consults
            # import_failures() before dismissing an unmatched YAML key.
            _IMPORT_FAILURES[rel_path.with_suffix("").parts[-1]] = (
                module_path,
                e,
            )
            logger.warning(
                f"Failed to load component module {module_path}: {e}"
            )

    return registry
