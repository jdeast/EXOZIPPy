"""The acceptance gate for the `mulensevent` split (review 8.6.17, stage 0).

These fixtures are the refactor's measuring device, so this file's first job
is to check the DEVICE, not the models: a decomposition that does not sum to
`compile_logp` produces confident wrong attributions, which is worse than no
attribution at all.

The fixtures themselves are recorded by `scripts/make_mulens_fixtures.py` and
hold, per shipped microlensing example, the reconciled per-term logp
decomposition at the start point.  During the split, each stage is accepted
by explaining every moved term against them -- byte-identity is NOT available
as acceptance for the first time in this review, because the split
deliberately collapses parameters that are stored per source but physically
singular.

Most of this is marked slow: each case builds a full System and compiles
PyTensor graphs.  Two fast, deterministic PSPL examples run unmarked so the
instrument itself is exercised on every suite run.
"""

import glob
import json
import os

import pytest
import yaml
from mulens_acceptance import compare, decompose, term_names

from exozippy.system import System

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
FIXTURES = os.path.join(HERE, "fixtures", "mulens")

# ob170114 goes through VBM's BinaryMag2, which carries ~1e-14 call-history
# jitter across compiledirs -- measured, its start logp differs in the last
# ULP between two worktrees of the same commit.  It can never carry byte
# acceptance; it is compared with a relative tolerance instead.
JITTERY = {"ob170114"}
JITTER_RTOL = 1e-9

# Fast, deterministic, symbolic-PSPL: the instrument runs on these every time.
UNMARKED = {"ob08092", "ob140939"}


def _fixture_files():
    return sorted(glob.glob(os.path.join(FIXTURES, "*.json")))


def _load(path):
    with open(path) as fh:
        return json.load(fh)


def _build(fixture):
    cfg_path = os.path.join(ROOT, fixture["config"])
    par_path = os.path.join(ROOT, fixture["params"])
    cwd = os.getcwd()
    try:
        os.chdir(os.path.dirname(cfg_path))
        with open(cfg_path) as fh:
            cfg = yaml.safe_load(fh)
        with open(par_path) as fh:
            par = yaml.safe_load(fh) or {}
        system = System(cfg, par)
        system.prepare()
        return system, system.build_model()
    finally:
        os.chdir(cwd)


def _case_ids():
    return [os.path.splitext(os.path.basename(p))[0] for p in _fixture_files()]


def test_the_fixture_set_is_not_empty():
    """
    Given the recorded stage-0 fixtures,
    When they are collected,
    Then there is at least one.

    Guards the whole file against becoming a silent no-op: every test below
    is parameterized over the fixture set, so an empty directory would make
    them all vanish while the suite stayed green -- which is exactly how a
    green suite comes to mean nothing.
    """
    assert _fixture_files(), f"no fixtures in {FIXTURES}"


@pytest.mark.parametrize("name", sorted(UNMARKED))
def test_the_decomposition_reconciles(name):
    """
    Given a shipped microlensing example,
    When its logp is decomposed term by term,
    Then the parts sum to `compile_logp`.

    THE INSTRUMENT'S OWN ACCEPTANCE, and it is checked before any fixture is
    trusted.  Two plausible decompositions were wrong while this was written:
    summing potentials plus `model.logp(rv)` per RV is 30.9 nats SHORT on
    ob161003 (missing the transform jacobians), and
    `model.logp(vars=[rv], jacobian=True)` per RV is 27.7 nats OVER on the
    same model because `logp(vars=...)` does not decompose additively.  Both
    looked right.  This assertion is the control that must fire.
    """
    # ARRANGE
    path = os.path.join(FIXTURES, name + ".json")
    if not os.path.exists(path):
        pytest.skip(f"no fixture for {name}")
    system, model = _build(_load(path))

    # ACT
    _, total, reconciles, summed = decompose(system, model)

    # ASSERT
    assert reconciles, f"parts sum to {summed!r}, compile_logp is {total!r}"


@pytest.mark.parametrize("name", sorted(UNMARKED))
def test_the_term_names_match_the_logp_terms(name):
    """
    Given a built model,
    When the term names are paired with `model.logp(sum=False)`,
    Then there are exactly as many names as terms.

    `logp(sum=False)` returns `basic_RVs` then `potentials`, and the pairing
    is positional. A reordering upstream would keep the SUM correct while
    mislabelling every term -- a decomposition that reconciles and lies.
    """
    # ARRANGE
    path = os.path.join(FIXTURES, name + ".json")
    if not os.path.exists(path):
        pytest.skip(f"no fixture for {name}")
    _, model = _build(_load(path))

    # ACT / ASSERT
    assert len(term_names(model)) == len(model.logp(sum=False))


@pytest.mark.slow
@pytest.mark.parametrize("name", _case_ids())
def test_the_model_still_matches_its_recorded_decomposition(name):
    """
    Given a stage-0 fixture recorded before the split,
    When the example is rebuilt and decomposed,
    Then every term matches.

    This is the refactor's regression gate. It is EXPECTED to fail during the
    split -- that is its purpose. A failure must be read term by term and
    every moved term explained; a term that vanished alongside one that
    appeared with the same value is a rename, which the diff reports
    separately so it is not mistaken for a match.
    """
    # ARRANGE
    fixture = _load(os.path.join(FIXTURES, name + ".json"))
    system, model = _build(fixture)

    # ACT
    parts, _, reconciles, _ = decompose(system, model)
    assert reconciles, "the instrument stopped reconciling; fix it first"

    jittery = any(j in fixture["config"] for j in JITTERY)
    tol = JITTER_RTOL if jittery else 0.0
    moved, appeared, vanished = compare(
        fixture["terms"], parts, atol=0.0, rtol=tol
    )

    # ASSERT
    assert not (moved or appeared or vanished), (
        f"moved={moved}\nappeared={appeared}\nvanished={vanished}"
    )
