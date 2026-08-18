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

    # 2. Use rglob to search recursively through all subfolders.  Sorted, for
    # the reason ConfigManager's two walks are (see config.py): rglob yields
    # filesystem directory order, which is stable on one machine and differs
    # between machines (ext4's hashed btree vs xfs/NFS), so it survives
    # PYTHONHASHSEED randomization and looks reproducible until two boxes are
    # compared.  Lower stakes here than there -- `registry` is keyed by
    # yaml_key and System instantiates in the user's config key order -- so
    # what this pins is module IMPORT order: PHYSICS_REGISTRY insertion order
    # (and which of two colliding names its duplicate error blames),
    # _IMPORT_FAILURES order, utilities/registry.all_utilities() order, and
    # the last-wins tie-break if two Component subclasses ever declared the
    # same yaml_key.  One word of insurance.
    for file in sorted(components_dir.rglob("*.py")):
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

        except Exception as e:
            # a developer might push an unused, broken component. that
            # shouldn't break the code -- but it MUST break the code if a
            # config actually asks for it.  System.__init__ consults
            # import_failures() before dismissing an unmatched YAML key.
            #
            # EVERY exception, not only ImportError (review 2.2.3): a
            # SyntaxError, a NameError or a failed module-scope table build in
            # one UNUSED component aborted discovery for every fit and every
            # GUI open, which is exactly what the sentence above promises will
            # not happen.  The narrow catch kept that promise for a missing
            # dependency and broke it for a typo.  Widening it costs nothing,
            # because a config that names the broken component still fails
            # loudly through import_failures().
            _IMPORT_FAILURES[rel_path.with_suffix("").parts[-1]] = (
                module_path,
                e,
            )
            logger.warning(
                f"Failed to load component module {module_path}: {e}"
            )

    return registry
