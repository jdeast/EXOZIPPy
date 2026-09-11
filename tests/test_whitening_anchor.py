"""The whitening anchor moves, so it is persisted (review 4.3.1).

`System.recenter_whitening_anchor` folds a polished start into each logit
element's ANCHOR (`sv_logit_q_inits`), so `raw = 0` is the start and
`Model.initial_point()` is correct by construction on every sampler branch.
The two mechanisms and why one cannot serve both paths are in
`src/exozippy/whitening.md` and the method's own docstring.

**THIS FILE'S REASON TO EXIST IS THE REUSED TRACE.**  A stored raw draw
decodes through

    phys = lower + span * sigmoid(anchor + scale * raw)

so the anchor is half of what a draw MEANS, and `<prefix>_whitening.json` is
the only record of it.  Before 4.3.1 the anchor could not move --
`set_whitening` left it exactly where `build_pymc` put it -- so a rebuilt
model reproduced it by construction and storing it would have been redundant.
Now it can, and an unpersisted moving anchor would make every reused trace
decode against the wrong center: silently, because every number involved
stays physically plausible, and with nothing downstream able to notice.  That
is the failure mode the persistence exists to prevent, and this is the only
place in the suite that reproduces it end to end.

The controls are what make these tests mean something.  Each round trip is
paired with a run that does NOT restore the anchor, and asserts the decode
comes out DIFFERENT there -- otherwise a test that passed would only be
proving the anchor is irrelevant.
"""

import json
import os

import numpy as np
import pytest

from exozippy.system import System
from exozippy.whitening import (
    StaleWhiteningError,
    measure_and_whiten,
    restore_whitening_for_trace,
    save_whitening,
)

# The polish's displacement, in raw (whitened) units where the prior is
# N(0, 1) -- a displacement in sigma, far outside float noise.
_POLISH_OFFSET = 0.8

# A synthetic "stored draw": raw coordinates a sampler might have visited,
# deliberately NOT the start, so decoding them exercises anchor + scale*raw
# rather than anchor alone.
_DRAW_RAW = 0.35


def _config(prefix):
    return (
        {
            "name": "anchor_test",
            "orbit": [{"name": "test_orbit"}],
            "prefix": prefix,
        },
        {
            "orbit.test_orbit.logP": {"initval": float(np.log10(10.0))},
            "orbit.test_orbit.tc": {"initval": 0.0},
            "orbit.test_orbit.secosw": {"initval": 0.0},
            "orbit.test_orbit.sesinw": {"initval": 0.0},
        },
    )


def _build(prefix):
    """A prepared System plus its model -- the production build path."""
    config, user_params = _config(prefix)
    system = System(config, user_params)
    system.prepare()
    model = system.build_model()
    return system, model


def _displace(system, model):
    """Stand in for the seed polish: displace every sampled raw element.

    STUBBED rather than run, for the reason `tests/test_nuts_start.py` gives:
    a real polish on this prior-only model lands back on the anchor (for a
    logit element the correction potential leaves a raw-space density peaked
    at raw = 0), so a real polish here would make every test vacuous.
    Everything downstream is the production path.
    """
    raw = system.get_raw_start(model)
    polished = [
        {
            k: np.asarray(v, dtype=float) + _POLISH_OFFSET
            for k, v in raw.items()
        }
    ]
    system.apply_polished_starts(polished, [0])


def _a_stored_draw(model):
    """A raw point standing in for one stored posterior draw."""
    return {
        k: np.full(np.shape(v), _DRAW_RAW, dtype=float)
        for k, v in model.initial_point().items()
    }


def _physical(system, model, raw_point):
    """What that raw point decodes to -- free RVs and every Deterministic.

    This is the question a reused trace asks: not "are the scales the same"
    but "does a stored draw still mean the physical values the sampler
    visited".
    """
    return {
        k: np.asarray(v, dtype=float)
        for k, v in system.get_internal_point(model, raw_point).items()
    }


@pytest.fixture(scope="module")
def sampled_run(tmp_path_factory):
    """A build that was polished, re-centered, whitened and persisted.

    Module-scoped because it prepares a System and compiles PyTensor graphs,
    and `--dist loadfile` pins this file to one worker so every consumer
    shares it.  Returns the state a finished fit leaves behind: the whitening
    file, the anchor it recorded, a synthetic stored draw, and what that draw
    decodes to under the coordinates it was "sampled" in.
    """
    tmp_path = tmp_path_factory.mktemp("anchor")
    prefix = str(tmp_path / "fit")
    orig_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        system, model = _build(prefix)
        _displace(system, model)
        moved = system.recenter_whitening_anchor(model)
        raw_start = system.get_raw_start(model)
        report = measure_and_whiten(system, model, raw_start)
        path = tmp_path / "fit_whitening.json"
        save_whitening(system, str(path), map_lp=report["map_lp"])
        draw = _a_stored_draw(model)
        decoded = _physical(system, model, draw)
    finally:
        os.chdir(orig_cwd)
    return {
        "system": system,
        "model": model,
        "moved": moved,
        "path": path,
        "tmp_path": tmp_path,
        "draw": draw,
        "decoded": decoded,
        "anchors": {
            p.label: p._whiten_state["sv_logit_q_inits"].get_value().copy()
            for p in system.get_all_parameters()
            if getattr(p, "_whiten_state", None) is not None
        },
    }


def test_the_recentering_actually_moved_the_anchor(sampled_run):
    """
    Given a displaced (polished) start,
    When the anchor is re-centered on it,
    Then logit elements report a moved anchor and their `raw_initval` is 0.

    The precondition for everything below, asserted rather than assumed: on
    an UNPOLISHED build every logit anchor is already the start and every
    round trip below would pass whether or not the anchor were persisted.
    That is the vacuity failure docs/testing.md warns about.
    """
    system = sampled_run["system"]
    moved = sampled_run["moved"]

    assert moved, "recenter_whitening_anchor moved nothing"
    for label in moved:
        par = next(p for p in system.get_all_parameters() if p.label == label)
        tf = par._raw_transform
        ri = np.asarray(par.raw_initval, dtype=float).reshape(-1)
        for j, i in enumerate(tf["sampled_idx"]):
            if tf["use_logit"][i]:
                assert ri[j] == 0.0, (
                    f"{label}[{i}] still carries a raw displacement "
                    f"({ri[j]}) after re-centering, so raw = 0 is not the "
                    f"start"
                )


def test_the_persisted_anchor_is_the_recentered_one(sampled_run):
    """
    Given the whitening file a finished fit wrote,
    When it is read back as JSON,
    Then it records the RE-CENTERED anchor, not the build-time one.

    Compared against a fresh build's own anchor, so the assertion is that
    the file carries something a rebuild could not have reproduced -- which
    is precisely why it has to be stored.
    """
    data = json.loads(sampled_run["path"].read_text())
    assert data["version"] == 2

    fresh_system, _fresh_model = _build(str(sampled_run["tmp_path"] / "fresh"))
    fresh = {
        p.label: p._whiten_state["sv_logit_q_inits"].get_value()
        for p in fresh_system.get_all_parameters()
        if getattr(p, "_whiten_state", None) is not None
    }

    differs = []
    for label, anchor in sampled_run["anchors"].items():
        np.testing.assert_allclose(
            np.asarray(data["params"][label]["logit_q_inits"]),
            anchor,
            rtol=0,
            atol=0,
        )
        if not np.allclose(fresh[label], anchor):
            differs.append(label)

    assert differs, (
        "a fresh build reproduces every persisted anchor, so this example "
        "cannot show that the anchor needs storing at all"
    )


def test_a_reused_trace_decodes_to_the_same_physical_values(sampled_run):
    """
    Given a stored raw draw and the whitening file its fit wrote,
    When a FRESH build restores that file on the trace-reuse path,
    Then the draw decodes to the same physical values, to the last bit --
      and WITHOUT the restore it decodes somewhere else entirely.

    This is the round trip 4.3.1's persisted-format change exists for, and
    the only end-to-end coverage of it.  The negative half is the load-bearing
    half: it is what proves that a moving anchor left out of the file would
    have silently re-coordinated every reused trace, rather than merely being
    untidy.
    """
    draw = sampled_run["draw"]
    expected = sampled_run["decoded"]

    # ARRANGE: a fresh build, exactly what `recompute_trace: false` gets.
    orig_cwd = os.getcwd()
    os.chdir(sampled_run["tmp_path"])
    try:
        system, model = _build(str(sampled_run["tmp_path"] / "reuse"))

        # CONTROL, first: the un-restored build must decode DIFFERENTLY, or
        # the assertion below proves nothing.
        unrestored = _physical(system, model, draw)

        # ACT
        status = restore_whitening_for_trace(
            system, str(sampled_run["path"]), "fit_trace.nc"
        )
        restored = _physical(system, model, draw)
    finally:
        os.chdir(orig_cwd)

    # ASSERT
    assert status == "restored"
    moved_without_restore = [
        k
        for k in expected
        if not np.allclose(unrestored[k], expected[k], rtol=0, atol=0)
    ]
    assert moved_without_restore, (
        "the un-restored build already decodes this draw correctly, so the "
        "persisted state is not what carries the decode here and the "
        "positive assertion below is vacuous"
    )

    for key in expected:
        np.testing.assert_allclose(
            restored[key],
            expected[key],
            rtol=0,
            atol=0,
            err_msg=(
                f"'{key}' decodes to a different physical value after a "
                f"restore, so a reused trace means something other than "
                f"what was sampled (review 4.3.1)"
            ),
        )


def test_dropping_only_the_anchor_from_the_file_is_caught(sampled_run):
    """
    Given the whitening file with ONLY the anchor removed,
    When a reused trace tries to restore it,
    Then StaleWhiteningError is raised rather than the scales applying alone.

    The scales in such a file are still perfectly valid, which is exactly
    what makes this dangerous: applying them and leaving the anchor at the
    rebuild's value would produce a model that decodes every draw to a
    plausible wrong place.  The version-2 requirement is what turns that into
    a refusal.
    """
    data = json.loads(sampled_run["path"].read_text())
    for entry in data["params"].values():
        entry.pop("logit_q_inits", None)
    broken = sampled_run["tmp_path"] / "anchorless_whitening.json"
    broken.write_text(json.dumps(data))

    orig_cwd = os.getcwd()
    os.chdir(sampled_run["tmp_path"])
    try:
        system, _model = _build(str(sampled_run["tmp_path"] / "anchorless"))
        with pytest.raises(StaleWhiteningError) as excinfo:
            restore_whitening_for_trace(system, str(broken), "fit_trace.nc")
    finally:
        os.chdir(orig_cwd)

    assert "logit_q_inits" in str(excinfo.value)
    assert "recompute_trace: true" in str(excinfo.value)


def test_recentering_is_idempotent(sampled_run):
    """
    Given a build whose anchor is already re-centered,
    When it is re-centered again,
    Then nothing moves.

    `run.py` calls it once, but the whitening's own escalation rounds re-read
    the start repeatedly and a tool may re-enter the path; a second fold
    would double-count the displacement and move the physical start, which
    is the one thing this mechanism promises not to do.
    """
    system = sampled_run["system"]
    model = sampled_run["model"]
    before = {
        p.label: p._whiten_state["sv_logit_q_inits"].get_value().copy()
        for p in system.get_all_parameters()
        if getattr(p, "_whiten_state", None) is not None
    }

    # ACT
    moved = system.recenter_whitening_anchor(model)

    # ASSERT
    assert moved == {}
    for label, anchor in before.items():
        par = next(p for p in system.get_all_parameters() if p.label == label)
        np.testing.assert_array_equal(
            par._whiten_state["sv_logit_q_inits"].get_value(), anchor
        )


def test_scales_without_the_anchor_decode_the_draw_somewhere_else(
    sampled_run,
):
    """
    Given the same file with the anchor dropped AND relabelled version 1 --
      the legacy shape, which is legitimately accepted,
    When a reused trace restores it and decodes a stored draw,
    Then the physical values are WRONG, and by a physically large amount.

    This measures the corruption the persisted anchor prevents, rather than
    asserting it in the abstract.  The restored scales are exactly right; the
    anchor is the rebuild's, which for a fit that re-centered is not where
    the draws were sampled.  Nothing raises, nothing warns, and every number
    that comes out is a plausible orbit -- which is why this had to become a
    schema requirement rather than a convention.

    It is also why `save_whitening` writes version 2 unconditionally: a
    version-1 file emitted by post-4.3.1 code would be this, and would look
    genuine to the next reload.
    """
    data = json.loads(sampled_run["path"].read_text())
    data["version"] = 1
    for entry in data["params"].values():
        entry.pop("logit_q_inits", None)
    legacy = sampled_run["tmp_path"] / "legacy_whitening.json"
    legacy.write_text(json.dumps(data))

    orig_cwd = os.getcwd()
    os.chdir(sampled_run["tmp_path"])
    try:
        system, model = _build(str(sampled_run["tmp_path"] / "legacy"))
        status = restore_whitening_for_trace(
            system, str(legacy), "fit_trace.nc"
        )
        decoded = _physical(system, model, sampled_run["draw"])
    finally:
        os.chdir(orig_cwd)

    # It APPLIES -- that is the whole problem with the legacy shape.
    assert status == "restored"

    expected = sampled_run["decoded"]
    wrong = {
        k: (float(np.ravel(expected[k])[0]), float(np.ravel(decoded[k])[0]))
        for k in expected
        if np.ravel(expected[k]).size
        and not np.allclose(decoded[k], expected[k], rtol=1e-9, atol=0)
    }
    assert wrong, (
        "dropping the anchor changed no decoded value, so the anchor is not "
        "load-bearing on this example and this file is testing the wrong "
        "thing"
    )
    # ...and not by a rounding error: at least one physical parameter is off
    # by a fraction of itself that no reader would call agreement.
    worst = max(
        abs(after - before) / max(abs(before), 1e-30)
        for before, after in wrong.values()
    )
    assert worst > 1e-3, (
        f"the anchor mismatch moves every value by at most {worst:.3g} "
        f"relative, which is too small for this test to demonstrate the "
        f"failure it describes"
    )
