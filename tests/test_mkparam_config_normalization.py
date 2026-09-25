"""`scripts/mkparam.py` could not write a restart file for any DC2018 run.

`run.py` hands `write_param_file` the live System's config, already normalized
by `System.__init__` (components rename themselves from their `body:`/`star:`
keys).  The trace's structural hash is stamped after that.  The standalone CLI
passes the config PATH instead, so the raw YAML was hashed and every call
raised StaleTraceError naming component changes that never happened.

run.md records the recompute/stamp agreement as verified on kelt4, ob08092 and
ob140939 -- none of which rename a component, which is why this survived.
"""

import copy
import pathlib

import yaml

from exozippy.evaluator import structural_hash
from exozippy.mkparam import _normalized_like_system
from exozippy.system import System

EXAMPLE_DIR = pathlib.Path(__file__).parent / ".." / "examples" / "DC2018_128"


def _raw():
    with open(EXAMPLE_DIR / "DC2018_128.yaml") as f:
        config = yaml.safe_load(f)
    with open(EXAMPLE_DIR / "DC2018_128.params.yaml") as f:
        params = yaml.safe_load(f)
    return config, params


def test_the_fixture_actually_renames_a_component():
    """Non-vacuity guard (docs/testing.md rule 3).

    Everything below is trivially true on a config whose components do NOT
    rename themselves, so pin that this example is one that DOES -- otherwise
    a later edit to the example silently empties the other two tests.
    """
    raw, params = _raw()
    before = copy.deepcopy(raw)
    after = System(
        copy.deepcopy(raw), user_params=copy.deepcopy(params)
    ).config
    renamed = [
        k
        for k in ("lens", "source", "mann", "torres")
        if k in before and before[k] != after.get(k)
    ]
    assert renamed, (
        "DC2018_128 no longer renames any component, so this file's other "
        "tests cannot detect the bug they exist for -- pick another example"
    )


def test_normalizing_reproduces_the_hash_run_py_stamps():
    """The fix: the from-disk path must hash what System produces."""
    raw, params = _raw()
    live = System(copy.deepcopy(raw), user_params=copy.deepcopy(params)).config
    fixed = _normalized_like_system(copy.deepcopy(raw), params)
    assert structural_hash(fixed, params) == structural_hash(live, params)


def test_the_raw_yaml_hash_is_the_one_that_was_wrong():
    """And pin that it genuinely differed -- the bug, not a tautology."""
    raw, params = _raw()
    live = System(copy.deepcopy(raw), user_params=copy.deepcopy(params)).config
    assert structural_hash(raw, params) != structural_hash(live, params)


def test_normalizing_does_not_mutate_the_caller_s_config():
    raw, params = _raw()
    before = copy.deepcopy(raw)
    _normalized_like_system(raw, params)
    assert raw == before
