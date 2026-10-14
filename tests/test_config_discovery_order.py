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

import pytest
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


def test_a_tie_is_split_the_same_whatever_order_the_symbols_iterate_in(
    monkeypatch,
):
    """
    Given two DEFAULT seeds of one relation that disagree with a user value
      (x = a + b, x pinned),
    When the relaxation engine settles it with `free_symbols` iterating in
      the natural order and again in a deliberately reversed one,
    Then the start is identical: a tie between two seeds of one tier is a
      minimum-norm split in init_scale units, decided by nothing that
      depends on the order a set of Symbols happens to iterate in (their
      hashes include the PYTHONHASHSEED-randomized name).

    Replaces the pin on the old Condition B's alphabetical tie-break, which
    the KS rule deleted (review 3.14.22): no path string decides anything.
    """
    import sympy as sp

    def run(reverse):
        cm = ConfigManager(
            {"toy.T.x": {"initval": 5.0}},
            system_config={"toy": [{"name": "T"}]},
        )
        cm.base_defaults["toy"] = {
            n: {"lower": -1e6, "upper": 1e6, "init_scale": s, "initval": 1.0}
            for n, s in (("x", 1.0), ("a", 1.0), ("b", 3.0))
        }
        syms = {n: sp.Symbol(f"toy.0.{n}") for n in "xab"}
        for s in syms.values():
            cm.master_symbol_map[str(s)] = s
        eq = sp.Eq(syms["x"], syms["a"] + syms["b"])
        if reverse:
            order = (syms["x"], syms["b"], syms["a"])
            monkeypatch.setattr(
                sp.Eq,
                "free_symbols",
                property(lambda self: set(order)),
                raising=False,
            )
        cm.all_relations = [eq]
        cm.set_engine_roles({str(s): "sampled" for s in syms.values()})
        cm.finalize_user_params()
        monkeypatch.undo()
        return dict(cm._last_resolved)

    natural, reversed_ = run(False), run(True)
    assert natural == reversed_
    # and it IS the split (0.3 / 2.7 of the 3 to absorb), not a single absorber
    assert natural["toy.0.a"] == pytest.approx(1.3, rel=1e-6)
    assert natural["toy.0.b"] == pytest.approx(3.7, rel=1e-6)
