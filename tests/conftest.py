"""Shared test helpers.

Plain classes (not fixtures) — imported explicitly by test files that need them.
Pytest adds the tests/ directory to sys.path, so ``from conftest import ...`` works.
"""
from exozippy.config import ConfigManager
from exozippy.components.parameter import Parameter


class _DummyConfigManager:
    """Minimal ConfigManager stub for tests that only need a no-op hint interface."""
    user_params = {}

    def add_hint(self, *args, **kwargs):
        pass

    def add_scale_hint(self, *args, **kwargs):
        pass


class _DummyComponent:
    """Stub component whose only observable property is n_elements."""
    def __init__(self, n_elements):
        self.n_elements = n_elements


class _DummySystem:
    """Empty system namespace for tests that attach attributes manually."""
    pass


class MockSystem:
    """Minimal System mock for ConfigManager and ModelAuditor tests.

    Usage::
        system = MockSystem(user_params)
        system.star = Star([...], system.config_manager)
    """

    def __init__(self, user_params):
        self.user_params = user_params
        self.config_manager = ConfigManager(user_params)
        self.star = None

    def get_parameter_lookup(self):
        return {p.label: p for p in self.get_all_parameters()}

    def get_all_parameters(self):
        if self.star is None:
            return []
        return [v for v in self.star.__dict__.values() if isinstance(v, Parameter)]
