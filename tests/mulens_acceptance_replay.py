"""Shared scaffolding for the `mulensevent` acceptance gate (review 8.6.17).

THE GATE IS TWO TEST FILES, test_mulens_acceptance_a.py and
test_mulens_acceptance_b.py, and this module is everything they share: the
fixture directory, the tolerances, the build-and-replay helpers, and the
PARTITION of the fixture set between them (GROUP_A / GROUP_B below).

Why two files rather than one.  `--dist loadfile` pins a whole test file to
one xdist worker, so a CI shard can never finish faster than its slowest
FILE's serial time.  Measured on CI 2026-09-14 (run 34888667878, ubuntu
3.12), the single acceptance file cost 873 worker-seconds -- the heaviest
file in the suite by 280 s and, at four shards, the binding floor on the
shard that carried it (17.4 min against 11 for the lightest).  Adding shards
cannot move that floor; only splitting the file can.  Two files of ~435 s
each put the suite back below the spread-bound regime, where every shard
finishes together.  The reasoning and the numbers are in
docs/testing-cache.md ("One canonical tree, and the sharded matrix").

THE RULE FOR A NEW FIXTURE: add its name to whichever of GROUP_A / GROUP_B
is lighter in tests/durations.json (the per-file weights the CI shard split
packs from).  `test_the_two_files_partition_the_fixture_set` in file A fails
if a fixture is in neither group or in both, so forgetting is loud rather
than a silently uncollected replay.  The tests themselves are all
parametrized over fixture NAMES with no shared fixture setup (each case
loads its own recorded JSON and builds its own System), which is what makes
the split by name list free of side effects.

--------------------------------------------------------------------------

WHAT THE GATE IS FOR.  These fixtures are the refactor's measuring device,
so the gate's first job is to check the DEVICE, not the models: a
decomposition that does not sum to `compile_logp` produces confident wrong
attributions, which is worse than no attribution at all.

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
term, by the tests at the bottom of test_mulens_acceptance_b.py.

Most of the gate is marked slow: each case builds a full System and compiles
PyTensor graphs.  Two fast, deterministic PSPL examples run unmarked so the
instrument itself is exercised on every suite run.
"""

import glob
import json
import os

import pytest
import yaml
from mulens_acceptance import (
    compare,
    decompose,
    is_reference_platform,
    record_deltas,
)

from exozippy.system import System

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
FIXTURES = os.path.join(HERE, "fixtures", "mulens")

# THE GATE REPLAYS THE FIXTURE'S STORED START, and that is the whole point.
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

# The STRICT tier, used only on the machine that recorded a fixture (see
# is_reference_platform).  NOT zero, and that is a measurement rather than a
# hedge: the generator's own --check is exact equality, and on this machine
# 12 of the 13 fixtures reproduce exactly while OGLE_0383LD's
# RV:mulensinstrument.model comes back 26832.036090685935 ->
# 26832.03609068592 -- 1.5e-11 absolute, 5.4e-16 relative, the same delta on
# every run.  So that one recorded value is about an ulp stale relative to
# the current code, and a zero tolerance would fail the reference platform
# against its own fixtures.
#
# 1e-14 is a few ulp at these magnitudes: still EIGHT orders tighter than
# the cross-platform tier, so it restores the regression sensitivity that a
# single loose tolerance gives up, while tolerating the last-bit drift that
# a pre-split recording legitimately carries.
REFERENCE_RTOL = 1e-14
# A denormal floor, strict tier only.  Some bound potentials sit at ~1e-220
# (CI saw POT:low_bound.mulensevent.mu_rel_mag at -2.0e-220), where a
# RELATIVE test is meaningless -- the same trap as the delta dump's 266%
# outlier, whose absolute delta was 3.5e-22.  1e-30 nats is far below
# anything physical and cannot hide a real term.
REFERENCE_ATOL = 1e-30

# Fast, deterministic, symbolic-PSPL: the instrument runs on these every time.
# Both live in GROUP_B, so the unmarked instrument tests are in file B.
UNMARKED = {"ob08092", "ob140939"}

# THE PARTITION.  Two lists of fixture NAMES (the basename of the recorded
# JSON), balanced on the per-test CI durations of run 34888667878:
#
#   GROUP_A  433 worker-s   ob09020 141, the KMT-2019-BLG-1806 family 164
#                           (95 + 34 + 34), the DC2018_128 family 128
#                           (25 + 19 single-counted, 58, 26)
#   GROUP_B  440 worker-s   ob140939 92 (48 reconcile + 25 replay + 19
#                           term names), ob161003 92 (52 replay + 24
#                           collapse + 16 scalar), ob170114 89, ob08092 78,
#                           OGLE_0383LD 46, ob07224 43
#
# Families are kept together (the three DC2018_128 configs, the three
# KMT-2019-BLG-1806 configs) so a reader looking for one example's replays
# finds them in one file; the balance came out within 2% anyway.  Every
# fixture-specific test lives in the file whose group holds its fixture.
GROUP_A = (
    "DC2018_128",
    "DC2018_128.2",
    "DC2018_128_hpc",
    "KMT-2019-BLG-1806",
    "KMT-2019-BLG-1806_nt24",
    "KMT-2019-BLG-1806_nt8long",
    "ob09020",
)
GROUP_B = (
    "OGLE_0383LD",
    "ob07224",
    "ob08092",
    "ob140939",
    "ob161003",
    "ob170114",
)


def fixture_files():
    # presplit/ holds the stage-0 ob161003 recording for the reconciliation
    # test in file B -- a reference, not a replay target, so the top-level
    # glob deliberately does not descend into it.
    return sorted(glob.glob(os.path.join(FIXTURES, "*.json")))


def fixture_names():
    return [os.path.splitext(os.path.basename(p))[0] for p in fixture_files()]


def load(path):
    with open(path) as fh:
        return json.load(fh)


def load_fixture(name):
    return load(os.path.join(FIXTURES, name + ".json"))


def build(fixture, extra_params=None):
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


def replay_cases(group):
    """One param per fixture in ``group`` that exists on disk.

    Restricted to the fixtures PRESENT so that a group naming a fixture
    someone deleted parametrizes nothing for it rather than failing every
    file at collection; the partition test in file A is where a mismatch
    between the groups and the directory is reported, once and by name.
    """
    present = set(fixture_names())
    return [pytest.param(name, id=name) for name in group if name in present]


def check_recorded_decomposition(name):
    """The body of the replay gate, shared by both files.

    Given a fixture recorded before the split, when the example is rebuilt
    and decomposed, every term must match.  Evaluated at the fixture's STORED
    start, not at this machine's own solved one, so a cross-machine failure
    means the likelihood function differs rather than the solver -- see
    TERM_RTOL above.
    """
    # ARRANGE
    fixture = load_fixture(name)
    system, model = build(fixture)

    # ACT -- replay the STORED start, so this machine and the recording
    # machine evaluate identical parameter values.
    parts, _, reconciles, _ = decompose(system, model, fixture["start"])
    assert reconciles, "the instrument stopped reconciling; fix it first"

    # Record EVERY term's delta, in tolerance or not, so CI on a
    # second platform can answer whether the LAPACK/nnls difference
    # behind 3.14.20 is directional or scatter.  `compare` below
    # reports only terms that EXCEED tolerance, and on macOS none
    # do -- which is precisely why it cannot answer the question.
    record_deltas(
        name,
        [
            (term, fixture["terms"][term], parts[term])
            for term in sorted(fixture["terms"])
            if term in parts
        ],
    )

    # TWO TIERS (review 3.14.20).  On the machine that RECORDED this
    # fixture, hold it to bit identity: a single tolerance sized to the
    # worst cross-platform difference (rtol 1e-6) is also blind to any
    # regression smaller than that, everywhere, permanently -- a real
    # 1e-4 nat change on the reference machine would pass.  Elsewhere,
    # hold it to the physics tolerance, because the platforms genuinely
    # differ: macOS builds on accelerate and Linux on scipy-openblas,
    # scipy.optimize.nnls disagrees between them at build time, and that
    # sets seeds, scales AND bounds.  Measured on PR #246's macOS job:
    # 786 of 925 terms bit-identical, largest absolute difference
    # 2.2e-06 nats, signs mixed 74/65 (scatter, not a directional
    # offset).  For scale, 0.1 sigma on a 1-D Gaussian is ~5e-3 nats.
    #
    # A fixture with no fingerprint predates this and takes the loose
    # tier: an old recording gets the physics bound rather than a
    # bit-identity claim nobody measured.
    reference = is_reference_platform(fixture)
    rtol, atol = (
        (REFERENCE_RTOL, REFERENCE_ATOL)
        if reference
        else (TERM_RTOL, TERM_ATOL)
    )

    moved, appeared, vanished = compare(
        fixture["terms"], parts, atol=atol, rtol=rtol
    )

    # ASSERT -- naming the TIER, because the two mean very different
    # things: a term moving by 5e-16 is a real finding on the reference
    # machine and noise anywhere else.
    tier = (
        f"the REFERENCE tolerance rtol={REFERENCE_RTOL} "
        f"(EXOZIPPY_ACCEPTANCE_STRICT is set and the fingerprint matches)"
        if reference
        else f"the physics tolerance (rtol={TERM_RTOL}, atol={TERM_ATOL})"
    )
    assert not (moved or appeared or vanished), (
        f"held to {tier}\n"
        f"moved={moved}\nappeared={appeared}\nvanished={vanished}"
    )
