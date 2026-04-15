# src/exozippy/components/factory.py
import importlib
import inspect
from pathlib import Path
from .component import Component
import ipdb

def discover_components():
    """
    Scans the package recursively for Component subclasses and maps them
    to their lowercase class names. Also triggers local physics registration.
    """
    registry = {}

    # 1. Start at the components directory
    components_dir = Path(__file__).parent

    # 2. Use rglob to search recursively through all subfolders
    for file in components_dir.rglob("*.py"):
        # Skip infrastructure files
        if file.name in ["__init__.py", "component.py", "factory.py", "parameter.py"]:
            continue

        # Construct the dynamic module path
        # e.g., star/star.py -> exozippy.components.star.star
        rel_path = file.relative_to(components_dir)
        module_path = "exozippy.components." + ".".join(rel_path.with_suffix("").parts)

        try:
            # 3. Import the component module
            module = importlib.import_module(module_path)

            # 5. Register any Component subclasses found
            for name, obj in inspect.getmembers(module, inspect.isclass):
                # Ensure it's a Component subclass, but NOT the base Component itself
                if issubclass(obj, Component) and obj is not Component:
                    key = getattr(obj, "yaml_key", name.lower())
                    registry[key] = obj

        except ImportError as e:
            # a developer might push an unused, broken component. that shouldn't break the code
            print(f"Warning: Failed to load component module {module_path}: {e}")

    return registry