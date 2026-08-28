"""Component discovery must not depend on filesystem directory order.

``ConfigManager`` finds every ``symbolic_physics.py`` and ``defaults.yaml`` with
``Path.rglob``, which yields entries in filesystem directory order.  That order
is stable on one machine but differs between machines (ext4's hashed btree vs
xfs/NFS), so it survives PYTHONHASHSEED randomization and looks perfectly
reproducible until two boxes are compared.

It is load-bearing: the walk order sets ``all_relations`` order, which sets the
order the relaxation engine visits equations, which decides which member of a
*symmetric* relation pair it solves for.  ``mulensing`` has two such pairs --
``mu_rel_mag**2 = mu_ra_rel**2 + mu_dec_rel**2`` and
``pi_rel = KAPPA*m*(pi_E_N**2 + pi_E_E**2)`` -- and nothing in either equation
breaks the tie.  Unsorted, an Ubuntu 26.04 box transposed ``(pm_ra, pm_dec)``,
``(mu_ra_rel, mu_dec_rel)`` and ``(pi_E_N, pi_E_E)`` relative to a RHEL8 box:
``t_E`` came out 11.54 d instead of 18.29 d and DC2018_128's logp at the pinned
``GOOD_RAW`` moved -945.57 -> -113614.65, i.e. two different physical models
from one config.

These tests simulate a differently-ordered filesystem by reversing ``rglob``.
"""

import copy
import pathlib

import yaml

from exozippy.config import ConfigManager

EXAMPLE_DIR = pathlib.Path(__file__).parent / ".." / "examples" / "DC2018_128"
COMPONENTS_DIR = (
    pathlib.Path(__file__).parent / ".." / "src" / "exozippy" / "components"
)


def _dc2018_128_inputs():
    with open(EXAMPLE_DIR / "DC2018_128.yaml") as f:
        config = yaml.safe_load(f)
    with open(EXAMPLE_DIR / "DC2018_128.params.yaml") as f:
        user_params = yaml.safe_load(f)
    return config, user_params


def _reverse_rglob(monkeypatch):
    """Make every Path.rglob yield its results in the opposite order."""
    real_rglob = pathlib.Path.rglob

    def reversed_rglob(self, pattern, *args, **kwargs):
        return iter(list(real_rglob(self, pattern, *args, **kwargs))[::-1])

    monkeypatch.setattr(pathlib.Path, "rglob", reversed_rglob)


def test_relation_order_is_independent_of_filesystem_walk_order(monkeypatch):
    """
    Given a filesystem that yields component files in the opposite order,
    When ConfigManager assembles its relation list,
    Then the list is identical to the normally-discovered one.
    """
    # Arrange
    config, user_params = _dc2018_128_inputs()
    reference = [
        str(rel)
        for rel in ConfigManager(
            copy.deepcopy(user_params), copy.deepcopy(config)
        ).all_relations
    ]
    assert reference, "expected DC2018_128 to instantiate some relations"

    # Act
    _reverse_rglob(monkeypatch)
    shuffled = [
        str(rel)
        for rel in ConfigManager(
            copy.deepcopy(user_params), copy.deepcopy(config)
        ).all_relations
    ]

    # Assert
    assert shuffled == reference


def test_defaults_merge_is_independent_of_filesystem_walk_order(monkeypatch):
    """
    Given a filesystem that yields defaults.yaml files in the opposite order,
    When ConfigManager merges them into base_defaults,
    Then the merged result is unchanged.
    """
    # Arrange
    config, user_params = _dc2018_128_inputs()
    reference = ConfigManager(
        copy.deepcopy(user_params), copy.deepcopy(config)
    ).base_defaults

    # Act
    _reverse_rglob(monkeypatch)
    shuffled = ConfigManager(
        copy.deepcopy(user_params), copy.deepcopy(config)
    ).base_defaults

    # Assert
    assert shuffled == reference


def test_no_root_level_default_key_has_two_owners():
    """
    Given every component defaults.yaml,
    When their root-level keys are collected,
    Then no key is defined by more than one file.

    _deep_merge is last-writer-wins, so a key with two owners would resolve by
    walk order.  Sorting makes that deterministic, but a collision is still
    almost certainly a mistake -- components/defaults.yaml is the intended home
    for genuinely shared root-level defaults.
    """
    # Arrange / Act
    owners = {}
    for path in sorted(COMPONENTS_DIR.rglob("defaults.yaml")):
        with open(path) as f:
            block = yaml.safe_load(f) or {}
        for key in block:
            owners.setdefault(key, []).append(path.parent.name)

    # Assert
    clashes = {k: v for k, v in owners.items() if len(v) > 1}
    assert not clashes, (
        f"root-level default keys with multiple owners: {clashes}"
    )


def test_rank_upgrade_tie_break_is_alphabetical(monkeypatch):
    """
    Given two symbols of one VIOLATED relation that tie on provenance rank,
    When _relax_equation picks the symbol to rewrite (Condition B),
    Then it rewrites the alphabetically first, not the first iterated.

    Sorting by (rank, name) is what makes that choice reproducible: the
    candidate list comes from `eq.free_symbols`, a set of Symbols whose hashes
    include the PYTHONHASHSEED-randomized name string, so on a rank tie an
    unsorted selection would follow set-iteration order and two processes
    running identical code would rewrite different parameters.

    This used to be pinned on _attempt_rank_upgrade, which was a dead
    alternative solve path (deleted 2026-08-18 with the rest of review 5.1.1).
    The rule is retargeted here, on the live engine, and asserted on the
    OUTCOME -- which symbol actually moved -- rather than by spying on min().
    `free_symbols` is monkeypatched to a deliberately non-alphabetical tuple
    so an unsorted implementation fails every run instead of by luck.
    """
    # Arrange
    import sympy as sp

    from exozippy.config import PRECEDENCE_DEFAULT

    config, user_params = _dc2018_128_inputs()
    cm = ConfigManager(copy.deepcopy(user_params), copy.deepcopy(config))

    a_name, b_name = "lens.0.pi_E_E", "lens.0.pi_E_N"
    a, b = sp.Symbol(a_name), sp.Symbol(b_name)
    cm.master_symbol_map.setdefault(a_name, a)
    cm.master_symbol_map.setdefault(b_name, b)

    # Both known, both at the same rank, and the relation is violated: that is
    # Condition B, whose whole job is to pick the weakest symbol and rewrite it.
    resolved = {a_name: 1.0, b_name: 2.0}
    provenance = {a_name: PRECEDENCE_DEFAULT, b_name: PRECEDENCE_DEFAULT}
    eq = sp.Eq(a, b)
    monkeypatch.setattr(
        sp.Eq, "free_symbols", property(lambda self: (b, a)), raising=False
    )

    # Act
    changed = cm._relax_equation(eq, resolved, provenance, {}, {}, 1e-6)

    # Assert -- pi_E_E sorts first, so it is the one that moves.
    assert changed
    assert resolved[a_name] == 2.0
    assert resolved[b_name] == 2.0
