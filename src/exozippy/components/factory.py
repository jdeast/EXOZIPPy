# src/exozippy/components/factory.py
import importlib
import inspect
from pathlib import Path
from .component import Component


def discover_components():
    """
    Scans the package for Component subclasses and maps them
    to their lowercase class names (or a mapping attribute).
    """
    registry = {}
    # Folders to scan
    search_paths = [
        Path(__file__).parent,  # src/exozippy/components/
        Path(__file__).parent.parent / "mulensing"  # src/exozippy/mulensing/
    ]

    for path in search_paths:
        for file in path.glob("*.py"):
            if file.name == "__init__.py":
                continue

            # Dynamic import
            module_path = f"exozippy.{path.name}.{file.stem}"
            module = importlib.import_module(module_path)

            for name, obj in inspect.getmembers(module, inspect.isclass):
                # Only register subclasses of Component
                if issubclass(obj, Component) and obj is not Component:
                    # Use a class attribute 'yaml_key' if it exists, else lowercase name
                    key = getattr(obj, "yaml_key", name.lower())
                    registry[key] = obj
    return registry