"""The acceptance gate for the `mulensevent` split (review 8.6.17).

These fixtures are the refactor's measuring device, so this file's first job
is to check the DEVICE, not the models: a decomposition that does not sum to
`compile_logp` produces confident wrong attributions, which is worse than no
attribution at all.

The fixtures themselves are recorded by `scripts/make_mulens_fixtures.py` and
hold, per shipped microlensing example, the reconciled per-term logp
decomposition at the start point.  During the split, each stage is accepted
by explaining every moved term against them -- byte-identity was NOT assumed
to be available for this review, because the split collapses parameters that
are stored per source but physically singular; for the two PSPL cases it
turned out to hold anyway (below).

STAGE-3 STATE (8.6.17).  All thirteen shipped example configs carry the
post-split spellings and every fixture replays its shipped example directly
(the stage-1b/2 converted config copies are deleted).  Twelve fixtures are
still the untouched stage-0 recordings with their LABELS translated to the
post-split naming (raw value vars and per-parameter potentials follow their
parameters' new component homes; the four event potentials follow the
build_likelihood split); every VALUE is bit-identical to the stage-0
recording.  Measured at stage 3: 0 moved / 0 appeared / 0 vanished at ZERO
tolerance on eleven of the twelve, including ob09020 (keplerian lens
orbital motion + RVs) and ob170114 (xallarap).  The twelfth, OGLE_0383LD,
moved by -1.5e-11 nats (5.4e-16 relative) in the data term alone:
localized to a 1-ulp difference in the derived companion mass ratio q under
the post-split masked-primary assembly, amplified through the binary-lens
caustic to <=1.4e-12 relative in A(t) -- the reassociation class the
design already accepts for VBM examples, and far inside TERM_RTOL.

ob161003 (2S2L) is the ONE example whose model changes BY CONSTRUCTION
(review R1): its per-source event-level vectors collapse to scalars.  Its
replay fixture is re-recorded post-split; the stage-0 recording is kept at
presplit/ob161003.json and the collapse is reconciled ANALYTICALLY, term by
term, by the tests at the bottom of this file.

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

# THIS FILE REPLAYS THE FIXTURE'S STORED START, and that is the whole point.
#
# The first version evaluated each machine's OWN engine-solved start and
# compared the results.  Linux differed by 7.3e-12 nats (1.6e-16 relative --
# machine epsilon) but macOS by 3.2e-05 (7.1e-10), seven orders larger.  The
# reflex is to widen the tolerance; that would have buried the only
# interesting question, because those numbers can mean two very different
# things:
#
#   1. the relaxation engine's NUMERICAL solve (`sp.nsolve`, config.py:3839)
#      converged slightly differently -- harmless, a chain forgets its start;
#   2. the LIKELIHOOD FUNCTION differs by platform -- meaning the posterior
#      depends on the hardware, which is serious.
#
# Evaluating at each machine's own start cannot separate them.  Replaying the
# STORED start does: both machines then evaluate identical parameter values,
# so anything left is (2) alone and the tolerance can be tight.  Question (1)
# is answered separately and strictly by `make_mulens_fixtures.py --check`,
# which reruns on ONE machine and demands bit-identity -- the right bar for a
# same-machine tool.
#
# WHAT REPLAYING THE START ACTUALLY SHOWED, and it was not what I predicted.
# I expected macOS to match once both machines evaluated the same parameter
# values.  It did not: the deltas came back BYTE-IDENTICAL to the previous
# run (zeropoint 2.1791695985484694e-06, model 3.193063457729295e-05).  A
# quantity that does not move when the start is replaced does not depend on
# the start.
#
# The cause is `scipy.optimize.nnls` run at BUILD time
# (mulensinstrument.py:907-943) to decompose the baseline into source and
# blend flux.  NNLS is an iterative active-set method over LAPACK
# least-squares, so Accelerate (macOS) and OpenBLAS (Linux) converge slightly
# differently.  Its output feeds `f_total` and `q_source` as SEEDS, SCALES
# AND BOUNDS -- so the raw <-> value TRANSFORM is itself platform-dependent
# and the same raw point maps to a slightly different physical value.  That
# is why `POT:logit_uniform_prior.mulensinstrument.q_source`, a
# bounds-dependent jacobian term, is among the movers.
#
# The honest statement is therefore narrower and more useful than "float
# noise": the model's CONSTRUCTION is platform-dependent at the ~1e-9
# relative level, through a build-time linear solve that sets prior support.
# Statistically irrelevant, genuinely real, previously invisible.  Filed as
# its own item; fixing it would mean pinning the flux decomposition rather
# than re-solving it on every build.
#
# The tolerance is sized to THAT mechanism -- LAPACK-level disagreement in a
# build-time solve -- rather than to whatever CI happened to produce.  Worst
# observed: 6.2e-09 relative, 3.2e-05 absolute.  A real model change is
# orders away: the event-rate copy that motivated this instrument was 2.8
# nats.
TERM_RTOL = 1e-6
TERM_ATOL = 1e-3

# Fast, deterministic, symbolic-PSPL: the instrument runs on these every time.
UNMARKED = {"ob08092", "ob140939"}


def _fixture_files():
    # presplit/ holds the stage-0 ob161003 recording for the reconciliation
    # test below -- a reference, not a replay target, so the top-level glob
    # deliberately does not descend into it.
    return sorted(glob.glob(os.path.join(FIXTURES, "*.json")))


def _load(path):
    with open(path) as fh:
        return json.load(fh)


def _build(fixture, extra_params=None):
    cfg_path = os.path.join(ROOT, fixture["config"])
    par_path = os.path.join(ROOT, fixture["params"])
    cwd = os.getcwd()
    try:
        os.chdir(os.path.dirname(cfg_path))
        with open(cfg_path) as fh:
            cfg = yaml.safe_load(fh)
        with open(par_path) as fh:
            par = yaml.safe_load(fh) or {}
        if extra_params:
            par.update(extra_params)
        system = System(cfg, par)
        system.prepare()
        return system, system.build_model()
    finally:
        os.chdir(cwd)


def _replay_cases():
    """One param per fixture.  At stage 3 every fixture replays its shipped
    example for real; the strict-xfail scaffolding that held this file
    honest while the examples still carried pre-split spellings is gone
    with the last unconverted example."""
    return [
        pytest.param(
            os.path.splitext(os.path.basename(path))[0],
            id=os.path.splitext(os.path.basename(path))[0],
        )
        for path in _fixture_files()
    ]


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
def test_the_fixture_stores_the_point_it_was_measured_at(name):
    """
    Given a recorded fixture,
    When it is loaded,
    Then it carries the start point every term was evaluated at.

    Without the point, the cross-machine comparison below silently becomes
    "each machine solves its own start and we compare the answers", which
    cannot distinguish a solver converging differently from a likelihood
    function that differs by platform.  A fixture missing its start is not a
    weaker fixture, it is a different and much vaguer measurement.
    """
    # ARRANGE / ACT
    path = os.path.join(FIXTURES, name + ".json")
    if not os.path.exists(path):
        pytest.skip(f"no fixture for {name}")
    fixture = _load(path)

    # ASSERT
    assert fixture.get("start"), "fixture has no stored start point"


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
def test_the_event_potentials_are_single_counted():
    """
    Given the DC2018_128 model (Op path, one companion, one source),
    When its logp terms are enumerated,
    Then each event potential exists exactly once, under the mulensevent
    prefix, and no term name is duplicated.

    The design's section-4 potential inventory, at its stage-2 vehicle
    (design section 7).  What a single-source event CAN pin here: the event
    potentials moved to their owning component without leaving a lens.*
    twin behind (a duplicate would double-count the event-rate weight, the
    defect class section 4 exists for), and `term_names` is collision-free
    so the decomposition's attribution is trustworthy.  What it CANNOT pin:
    that the event-rate term is single-counted ACROSS SOURCES -- with one
    source the sum and the scalar coincide; ob161003 (2S2L) carries that
    check at stage 3 (section 4's analytic reconciliation).
    """
    # ARRANGE
    path = os.path.join(FIXTURES, "DC2018_128.json")
    _, model = _build(_load(path))

    # ACT
    names = term_names(model)
    counts = {n: names.count(n) for n in names}

    # ASSERT -- no duplicates anywhere (a duplicate name is silently summed
    # by the decomposition, so it would LOOK reconciled while double
    # counting).
    dupes = {n: c for n, c in counts.items() if c > 1}
    assert not dupes, f"duplicated logp terms: {dupes}"

    for pot in (
        "POT:mulensevent.event_rate_prior",
        "POT:mulensevent.source_behind_lens",
        "POT:mulensevent.mu_rel_singularity",
        "POT:mulensevent.theta_E_singularity",
    ):
        assert counts.get(pot) == 1, f"{pot} missing or duplicated"

    # And the pre-split homes are GONE -- the move must not have left a
    # lens.* twin adding the same weight twice.
    stale = [
        n
        for n in names
        if n.startswith("POT:lens.")
        and n.split(".", 1)[1]
        in (
            "event_rate_prior",
            "source_behind_lens",
            "mu_rel_singularity",
            "theta_E_singularity",
            "fitpirel_jacobian",
            "fitu0te_jacobian",
        )
    ]
    assert not stale, f"pre-split event potentials survive on lens: {stale}"


@pytest.mark.slow
@pytest.mark.parametrize("name", _replay_cases())
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

    Evaluated at the fixture's STORED start, not at this machine's own solved
    one, so a cross-machine failure means the likelihood function differs
    rather than the solver -- see TERM_RTOL above.
    """
    # ARRANGE
    fixture = _load(os.path.join(FIXTURES, name + ".json"))
    system, model = _build(fixture)

    # ACT -- replay the STORED start, so this machine and the recording
    # machine evaluate identical parameter values.
    parts, _, reconciles, _ = decompose(system, model, fixture["start"])
    assert reconciles, "the instrument stopped reconciling; fix it first"

    moved, appeared, vanished = compare(
        fixture["terms"], parts, atol=TERM_ATOL, rtol=TERM_RTOL
    )

    # ASSERT
    assert not (moved or appeared or vanished), (
        f"moved={moved}\nappeared={appeared}\nvanished={vanished}"
    )


# ---------------------------------------------------------------------------
# ob161003 (2S2L): the ONE deliberate model change (design section 4).
#
# Under R1 the per-source event-level vectors (t_E, theta_E, pi_rel, pi_E,
# the mu_rel family, mlens_total) collapse to scalars, so every logp term
# that used to SUM an identical per-source pair is now counted once.  The
# trajectory at matched t_0/u_0/t_E/s/q/alpha/rho is UNCHANGED, so the
# observed-data term must MATCH the stage-0 recording -- not merely be
# close.  Any motion there is a wiring bug, not a model change; that is the
# sharpest test in the whole refactor and it is held at ZERO tolerance.
# ---------------------------------------------------------------------------

# The stage-0 engine solved the proper-motion split itself (sp.nsolve); the
# shipped params file now seeds the same split in clean round numbers
# (star.Lens.pm_dec = -13.595474606, everything else -3.0), which differ
# from the nsolve output at the ~1e-11 level.  Seeding nsolve's own values
# here puts the post-split model at the stage-0 recording's EXACT start, so
# the data term can be compared to the byte.  Values dumped from the
# pre-split tree at full precision (2026-09, stage-3 acceptance).
_OB161003_PRESPLIT_PM = {
    "star.SourceA.pm_ra": {"initval": -2.999999999986347},
    "star.SourceA.pm_dec": {"initval": -2.999999999986347},
    "star.SourceB.pm_ra": {"initval": -2.999999999986347},
    "star.SourceB.pm_dec": {"initval": -2.999999999986347},
    "star.Lens.pm_ra": {"initval": -2.999999999986347},
    "star.Lens.pm_dec": {"initval": -13.595474606031832},
    "star.LensB.pm_ra": {"initval": -2.999999999986347},
    "star.LensB.pm_dec": {"initval": -2.999999999986347},
}

# Every term whose pre-split value was an identical per-source pair summed
# over TWO sources and is now a scalar: the post-split value must be
# EXACTLY half the recording (bitwise -- halving a double is exact).  All
# but mu_rel_geo_mag's soft lower bound are the ~1e-304 placeholder values
# of soft bounds evaluated far from their turn-on.
_OB161003_HALVED = frozenset(
    {
        "POT:mulensevent.mu_rel_singularity",
        "POT:mulensevent.theta_E_singularity",
        "POT:low_bound.mulensevent.mu_ra_rel",
        "POT:low_bound.mulensevent.mu_dec_rel",
        "POT:low_bound.mulensevent.mu_ra_rel_geo",
        "POT:low_bound.mulensevent.mu_dec_rel_geo",
        "POT:low_bound.mulensevent.mu_rel_mag",
        "POT:low_bound.mulensevent.mu_rel_geo_mag",
        "POT:low_bound.mulensevent.pi_rel",
        "POT:low_bound.mulensevent.t_E",
        "POT:low_bound.mulensevent.theta_E",
        "POT:up_bound.mulensevent.mu_ra_rel",
        "POT:up_bound.mulensevent.mu_dec_rel",
        "POT:up_bound.mulensevent.mu_ra_rel_geo",
        "POT:up_bound.mulensevent.mu_dec_rel_geo",
        "POT:up_bound.mulensevent.mu_rel_mag",
        "POT:up_bound.mulensevent.mu_rel_geo_mag",
        "POT:up_bound.mulensevent.pi_rel",
        "POT:up_bound.mulensevent.theta_E",
    }
)

# The raw->value transform jacobians of the reseeded pm leaves reassociate
# at the last ulp (initval now enters as a literal seed rather than as
# nsolve's output); everything else must match to the byte.
_OB161003_ULP_OK = frozenset(
    {
        "POT:logit_uniform_prior.star.pm_ra",
        "POT:logit_uniform_prior.star.pm_dec",
    }
)


@pytest.mark.slow
def test_ob161003_event_potentials_are_scalar_and_single():
    """
    Given the ob161003 (2S2L) model,
    When its logp terms are enumerated,
    Then the event potentials exist exactly once, the event-rate and
    singularity potentials are SCALAR (no reduction over sources), and no
    pre-split lens.* twin survives.

    The design's section-4 potential inventory at its stage-3 vehicle: with
    TWO sources, a scalar event-rate potential is structurally incapable of
    the double count review 8.6.18 removed (DC2018_128's single-source
    inventory could not distinguish the sum from the scalar).
    source_behind_lens deliberately stays a sum: it is one term PER SOURCE
    STAR (each source must sit behind the lens).
    """
    # ARRANGE
    fixture = _load(os.path.join(FIXTURES, "ob161003.json"))
    _, model = _build(fixture)

    # ACT
    names = term_names(model)
    counts = {n: names.count(n) for n in names}

    # ASSERT
    dupes = {n: c for n, c in counts.items() if c > 1}
    assert not dupes, f"duplicated logp terms: {dupes}"

    for pot in (
        "POT:mulensevent.event_rate_prior",
        "POT:mulensevent.source_behind_lens",
        "POT:mulensevent.mu_rel_singularity",
        "POT:mulensevent.theta_E_singularity",
    ):
        assert counts.get(pot) == 1, f"{pot} missing or duplicated"

    stale = [
        n
        for n in names
        if n.startswith("POT:lens.")
        and n.split(".", 1)[1]
        in (
            "event_rate_prior",
            "source_behind_lens",
            "mu_rel_singularity",
            "theta_E_singularity",
            "fitpirel_jacobian",
            "fitu0te_jacobian",
        )
    ]
    assert not stale, f"pre-split event potentials survive on lens: {stale}"

    # Scalar means scalar: log(mu_rel_geo) + log(theta_E) with no pt.sum
    # over sources.  A sum over the collapsed (1,) event vector would pass
    # every value check while silently re-growing with a second source.
    pots = {f"POT:{p.name}": p for p in model.potentials}
    for pot in (
        "POT:mulensevent.event_rate_prior",
        "POT:mulensevent.mu_rel_singularity",
        "POT:mulensevent.theta_E_singularity",
    ):
        assert pots[pot].ndim == 0, (
            f"{pot} is not scalar (ndim {pots[pot].ndim})"
        )


@pytest.mark.slow
def test_ob161003_collapse_reconciles_against_the_presplit_recording():
    """
    Given the post-split ob161003 model started at the stage-0 recording's
      exact point (the pre-split engine's own solved proper motions seeded
      verbatim),
    When its decomposition is compared to the pre-split recording,
    Then the OBSERVED-DATA term matches to the BYTE, the only moved terms
      are the collapsed per-source pairs -- each EXACTLY half -- and
      nothing appears or vanishes.

    This is design section 4's analytic reconciliation, held as a permanent
    pin.  The data term at zero tolerance is the sharp edge: the trajectory
    at matched parameters is unchanged by construction, so ANY motion there
    is a wiring bug.  Measured at stage 3: 48 of 69 terms bit-identical
    (data term delta exactly 0.0), 19 terms exactly halved, and the two
    reseeded-pm transform jacobians moved by one ulp.
    """
    # ARRANGE -- the pre-split recording (labels translated, values
    # untouched) and the post-split model at its exact start.
    presplit = _load(os.path.join(FIXTURES, "presplit", "ob161003.json"))
    fixture = _load(os.path.join(FIXTURES, "ob161003.json"))
    system, model = _build(fixture, extra_params=_OB161003_PRESPLIT_PM)

    # ACT -- decompose at this model's own solved start (which the seeds
    # above pin to the recording's start).
    parts, _, reconciles, _ = decompose(system, model)
    assert reconciles, "the instrument stopped reconciling; fix it first"
    moved, appeared, vanished = compare(presplit["terms"], parts)  # ZERO tol

    # ASSERT -- the sharp edge first: the data term to the byte.
    assert (
        parts["RV:mulensinstrument.model"]
        == presplit["terms"]["RV:mulensinstrument.model"]
    ), "the observed-data logp moved at matched trajectory: a wiring bug"

    assert not appeared, f"terms appeared: {sorted(appeared)}"
    assert not vanished, f"terms vanished: {sorted(vanished)}"

    unexplained = {}
    halved_seen = set()
    for name, (before, after, delta) in moved.items():
        if name in _OB161003_HALVED:
            # Exactly one of the two identical per-source copies remains;
            # halving a finite double is exact, so this holds bitwise.
            if after == before / 2.0:
                halved_seen.add(name)
                continue
        if name in _OB161003_ULP_OK and abs(delta) <= 1e-13 * abs(before):
            continue
        unexplained[name] = (before, after, delta)
    assert not unexplained, f"unexplained logp motion: {unexplained}"

    # Every collapsed term must actually have moved: a halved term that
    # MATCHES the recording would mean the per-source pair is back.
    missing = _OB161003_HALVED - halved_seen
    assert not missing, (
        f"expected these collapsed terms to be half the recording, but "
        f"they matched it -- the per-source duplication is back: "
        f"{sorted(missing)}"
    )
