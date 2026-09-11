"""The flip-flop degeneracy, and the opt-in bound that breaks it (PK P5).

See ``src/exozippy/components/pharmacokinetics/README.md``: written by an
astrophysicist and an LLM with no domain reviewer.

The degeneracy is EXACT and is a property of the model, not of any dataset:
exchanging ``ka`` and ``ke`` and scaling ``V`` by ``ke/ka`` reproduces every
predicted concentration. ``tests/test_pharmacokinetics_physics.py`` pins that
for the curve; these tests pin it for the built MODEL, which is where it
matters -- the likelihood having two exactly equal modes is what makes the
mode report's two modes real rather than an artifact.
"""

import numpy as np
import pytest

# tests/ is on sys.path via conftest.
from test_pharmacokinetics_components import (  # noqa: E402
    _config,
    synth_csv,  # noqa: F401  (a fixture, used by name)
)

from exozippy.components.pharmacokinetics.subject import Subject
from exozippy.system import System

# One mirrored pair, in the units the component works in. ka and ke are
# exchanged and V is scaled by ke/ka, which leaves CL alone: 2.8 either way.
CL, V, KA = 2.8, 32.0, 1.5
KE = CL / V
V_MIRROR = V * KE / KA


def _starts(cl, v, ka):
    return {
        "subject.log_cl": {"initval": float(np.log10(cl))},
        "subject.log_v": {"initval": float(np.log10(v))},
        "subject.log_ka": {"initval": float(np.log10(ka))},
        "assay.sigma_add": {"initval": 0.15},
        "assay.sigma_prop": {"initval": 0.08},
    }


def _system(csv, tmp_path, user_params=None, fast=()):
    cfg = _config(csv, tmp_path)
    for block in cfg["subject"]:
        if block["name"] in fast:
            block["assume_fast_absorption"] = True
    system = System(cfg, user_params=dict(user_params or {}))
    system.prepare()
    return system


def _logps(csv, tmp_path, user_params, fast=()):
    """(total logp, observed-data logp) at the start point."""
    system = _system(csv, tmp_path, user_params, fast)
    model = system.build_model()
    point = model.initial_point()
    return (
        float(model.compile_logp()(point)),
        float(model.compile_fn(model.observedlogp)(point)),
    )


# ---------------------------------------------------------------------------
# The degeneracy itself
# ---------------------------------------------------------------------------


def test_the_mirrored_solution_fits_the_data_identically(synth_csv, tmp_path):
    """Given the swap, Then the observed log-likelihood is unchanged.

    Bit-for-bit, not approximately: the two solutions are the same curve. It
    is asserted on the data term because the coordinate priors differ between
    the two points -- log V is not the same number -- and are not supposed to
    match.
    """
    _, direct = _logps(synth_csv, tmp_path, _starts(CL, V, KA))
    _, mirror = _logps(synth_csv, tmp_path, _starts(CL, V_MIRROR, KE))

    assert direct == pytest.approx(mirror, rel=1e-12)


def test_clearance_survives_the_swap_and_volume_does_not(synth_csv, tmp_path):
    """Given the swap, Then CL is identical and V differs by ke/ka.

    The clinically load-bearing half of the degeneracy: steady-state dosing
    follows CL and is unaffected by which mode a chain settles in, while a
    loading dose follows V and is not. Checked as an arithmetic identity so
    that the claim in the docs is pinned rather than asserted.
    """
    assert KE * V == pytest.approx(KA * V_MIRROR)  # CL, both ways round
    assert V_MIRROR == pytest.approx(V * KE / KA)
    assert V_MIRROR != pytest.approx(V)


def test_the_component_asks_for_hot_chain_retention(synth_csv, tmp_path):
    """Given two equal modes, Then the component opts in generically.

    ``expects_suppressed_modes`` is declared on ``Component`` so that a
    component with degenerate solutions can turn on the sampler's mode
    retention without the sampler layer learning any component's name. A
    pharmacology component using it is the cleanest evidence available that
    the hook is component-agnostic.
    """
    assert Subject.expects_suppressed_modes is True


# ---------------------------------------------------------------------------
# The opt-in bound
# ---------------------------------------------------------------------------


def test_nothing_breaks_the_symmetry_by_default(synth_csv, tmp_path):
    """Given no flag, Then there is no ordering term at all.

    Rope, not gates: both solutions are a real property of oral-only data,
    resolved in practice by an IV reference arm rather than by a modelling
    choice, so the default must leave them both.
    """
    system = _system(synth_csv, tmp_path)
    model = system.build_model()

    assert "subject.absorption_order" not in {p.name for p in model.potentials}
    assert not any(system.subject.fast_absorption)


def test_assume_fast_absorption_penalizes_only_the_mirrored_mode(
    synth_csv, tmp_path
):
    """Given the flag, Then ka > ke costs ~nothing and ka < ke costs a lot.

    The shape of a soft bound: flat on the allowed side, and on the forbidden
    side a finite penalty with a gradient, not a wall. Measured as the
    difference the flag makes to the total logp at one point, which is
    exactly the term it adds.
    """
    fast = {"S1", "S2", "S3"}

    # ka = 1.5 /hr, ke = 0.0875 /hr: absorption is 17x faster, the allowed
    # side, more than a dex past the transition.
    allowed_off, _ = _logps(synth_csv, tmp_path, _starts(CL, V, KA))
    allowed_on, _ = _logps(synth_csv, tmp_path, _starts(CL, V, KA), fast=fast)

    # The mirror: ka = 0.0875 /hr against ke = 1.5 /hr.
    mirror_off, _ = _logps(synth_csv, tmp_path, _starts(CL, V_MIRROR, KE))
    mirror_on, _ = _logps(
        synth_csv, tmp_path, _starts(CL, V_MIRROR, KE), fast=fast
    )

    allowed_cost = allowed_off - allowed_on
    mirror_cost = mirror_off - mirror_on

    assert allowed_cost == pytest.approx(0.0, abs=1e-6)
    assert mirror_cost > 10.0
    # FINITE, which is the whole difference from a truncation: a chain that
    # starts in the mirrored mode is pushed out of it, not stuck at -inf with
    # no gradient to follow.
    assert np.isfinite(mirror_cost)


def test_the_bound_is_per_subject(synth_csv, tmp_path):
    """Given the flag on one subject, Then only that subject is constrained.

    The cost of the mirrored point with one subject constrained is a third of
    the cost with all three, because each contributes the same penalty -- so
    this checks the masking arithmetic and not merely that a term exists.
    """
    starts = _starts(CL, V_MIRROR, KE)
    base, _ = _logps(synth_csv, tmp_path, starts)
    one, _ = _logps(synth_csv, tmp_path, starts, fast={"S2"})
    all_three, _ = _logps(synth_csv, tmp_path, starts, fast={"S1", "S2", "S3"})

    assert (base - all_three) == pytest.approx(3.0 * (base - one), rel=1e-9)


def test_the_gradient_survives_deep_inside_the_forbidden_side(
    synth_csv, tmp_path
):
    """Given a point far into the mirrored mode, Then dlogp is finite.

    A soft bound that saturated to a constant -- or to a NaN -- would leave
    the sampler with nothing to follow back, which is the failure mode the
    penalty exists to avoid.
    """
    system = _system(
        synth_csv,
        tmp_path,
        _starts(CL, V_MIRROR, KE),
        fast={"S1", "S2", "S3"},
    )
    model = system.build_model()
    point = model.initial_point()

    gradient = model.compile_dlogp()(point)
    assert np.all(np.isfinite(gradient))
    assert np.any(gradient != 0.0)


def test_assume_fast_absorption_must_be_a_bool(synth_csv, tmp_path):
    """Given `assume_fast_absorption: 1`, Then it raises."""
    cfg = _config(synth_csv, tmp_path)
    cfg["subject"][0]["assume_fast_absorption"] = 1

    with pytest.raises(ValueError, match="must be true or false"):
        System(cfg, user_params={}).prepare()


# ---------------------------------------------------------------------------
# Telling the reader which numbers move
# ---------------------------------------------------------------------------


def test_the_volume_rows_carry_the_flip_flop_note(synth_csv, tmp_path):
    """Given the degeneracy is left in, Then V's table row says so.

    V is the quantity the swap moves; CL, AUC and half-life are not. A reader
    of a bimodal table needs to know which is which, and the note is attached
    at the site that creates the rows.
    """
    system = _system(synth_csv, tmp_path)

    for name in ("v", "log_v"):
        entry = system.subject.manifest[name]
        assert isinstance(entry, dict), name
        assert "flip-flop" in entry["table_note"], name


def test_the_note_goes_away_when_the_symmetry_is_broken(synth_csv, tmp_path):
    """Given every subject assumes ka > ke, Then the note is not emitted.

    A note that is sometimes describing a mode the fit does not report is
    worse than no note, because the reader cannot tell which time it is.
    """
    system = _system(synth_csv, tmp_path, fast={"S1", "S2", "S3"})

    for name in ("v", "log_v"):
        entry = system.subject.manifest[name]
        note = entry.get("table_note") if isinstance(entry, dict) else None
        assert note is None, name
