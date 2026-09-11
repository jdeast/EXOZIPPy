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


# ---------------------------------------------------------------------------
# Review 2.3.18: the start's |raw| must be ~0 whether or not the probe ran
# ---------------------------------------------------------------------------


def _max_abs_raw(point):
    return max(
        float(np.max(np.abs(np.asarray(v, dtype=float))))
        for v in point.values()
    )


def test_the_start_sits_at_raw_zero_without_the_whitening_probe(
    tmp_path_factory,
):
    """
    Given a polished start and NO whitening measurement (`measure_scales:
    false`),
    When the anchor is re-centered,
    Then `max |raw|` of the start is exactly 0 -- so the chain begins where
      the identity metric is calibrated.

    THIS IS REVIEW 2.3.18, and it is the half a `set_whitening`-based fix
    could not reach.  The polish runs under `if not reusing_trace:`,
    independently of the whitening, and stores its displacement in
    PRELIMINARY scale units; `set_whitening` was what re-expressed it, and
    `measure_scales: false` stops `set_whitening` running at all.  Measured
    on `examples/kelt4` RV-only, which is what
    `tests/test_integration_kelt4.py` configures:

        measure_scales   no re-centering   re-centered
        false                 684.07            0.0
        true                   10.58            0.0

    With `pm.NUTS`'s identity metric on raw and steps of order one raw unit,
    a start 684 sigma out is what made a 1-draw integration test scatter to
    6.7 Mjup.  Re-centering is independent of `measure_scales` BY
    CONSTRUCTION -- run.py calls it before the `reusing_trace or
    measure_scales` guard -- which is what this test pins.

    The sibling fixture covers `measure_scales: true` (it runs the real
    probe); this case runs no probe at all, so the two together are the 2x2.
    """
    tmp_path = tmp_path_factory.mktemp("anchor_noprobe")
    orig_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        system, model = _build(str(tmp_path / "noprobe"))
        _displace(system, model)

        # PRECONDITION: the polish really did move the start off raw = 0, in
        # preliminary scale units, with no probe to re-express it.
        before = _max_abs_raw(system.get_raw_start(model))
        assert before > 0.1, (
            f"max |raw| after the polish is {before:.3g}; with no "
            f"displacement this test is vacuous"
        )

        # ACT -- no measure_and_whiten call anywhere in this test.
        system.recenter_whitening_anchor(model)
        after = system.get_raw_start(model)
    finally:
        os.chdir(orig_cwd)

    # ASSERT: every logit element is at exactly 0.  A Gaussian-path element
    # may legitimately be nonzero -- its center is its prior mean and must
    # not move, so `Model.set_initval` carries its offset instead -- so the
    # assertion is per element, against that element's own role.
    lookup = {p.label: p for p in system.get_all_parameters()}
    checked = 0
    for key, vec in after.items():
        par = lookup[key[: -len("_raw")]]
        tf = par._raw_transform
        vals = np.asarray(vec, dtype=float).reshape(-1)
        for j, i in enumerate(tf["sampled_idx"]):
            if tf["use_logit"][i]:
                assert vals[j] == 0.0, (
                    f"{par.label}[{i}] starts at raw = {vals[j]!r} with no "
                    f"whitening measurement; the identity metric is "
                    f"calibrated at raw = 0 (review 2.3.18)"
                )
                checked += 1
    assert checked, "no logit element in this build to check"
    assert _max_abs_raw(after) == 0.0, (
        "this model is all-logit, so the whole start should be exactly 0"
    )


# ---------------------------------------------------------------------------
# Mechanism (2): the GAUSSIAN path's center is the prior mean and must not
# move.  Its polished offset rides on Model.set_initval instead.
# ---------------------------------------------------------------------------


def _mixed_model():
    """One LOGIT element (two finite bounds) and one GAUSSIAN-PATH element
    carrying an explicit `mu` != `initval`, in one pm.Model.

    The shipped orbit config the other tests use is all-logit, so without
    this the `Model.set_initval` half of the mechanism has no coverage at
    all -- and it is the half whose failure mode is a moved PRIOR.
    """
    import pymc as pm

    from exozippy.components.parameter import Parameter

    p_logit = Parameter(label="toy.x", initval=2.0, lower=0.0, upper=10.0)
    p_gauss = Parameter(
        label="toy.z",
        initval=0.5,  # start
        mu=0.0,  # PRIOR MEAN -- deliberately not the start
        sigma=2.0,
        lower=-np.inf,
        upper=np.inf,
    )
    with pm.Model() as model:
        xv = p_logit.build_pymc()
        zv = p_gauss.build_pymc()
        pm.Potential("like", -0.5 * ((xv - 2.5) / 0.5) ** 2 + 0.0 * zv)
    return model, p_logit, p_gauss


class _StubSystem:
    """Duck-typed stand-in for System: parameter lookup only.

    The shape `tests/test_polish.py` uses to drive `System` methods against
    hand-built Parameters -- `recenter_whitening_anchor` needs only
    `get_all_parameters()` and the model.
    """

    def __init__(self, params):
        self._params = params

    def get_all_parameters(self):
        return self._params


def test_the_gaussian_paths_center_is_never_moved_but_its_start_is_carried():
    """
    Given a polished start on BOTH a logit element and a Gaussian-path
      element whose `mu` differs from its `initval`,
    When the anchor is re-centered,
    Then the logit element's anchor absorbs its displacement and its
      `raw_initval` becomes 0, while the Gaussian element's CENTER is
      untouched, its `raw_initval` is kept, and `Model.initial_point()`
      carries that nonzero value -- via `Model.set_initval`.

    This is mechanism (2), and the reason it is a second mechanism rather
    than the same one: there `val = gaussian_mus + gaussian_scales * raw`
    with `raw ~ N(0,1)` AS THE PRIOR, and `gaussian_mus` is the prior MEAN
    whenever a `mu` was given.  Folding a start displacement into it would
    move the prior -- a change to the model, not to the coordinates -- and
    an added offset fails the same way by a longer route (`mu + scale*(raw +
    off)` has prior `N(mu + scale*off, scale)`).

    "The prior did not move" is asserted EXACTLY and three ways, because it
    is the whole claim: the center and width are bit-identical in all three
    mirrors (the frozen transform, and the shared variable the graph reads),
    and the raw -> physical map is bit-identical over a grid of raw values,
    which is that prior's pushforward.

    The TOTAL start logp is held to a derived bound rather than to equality,
    and the reason is worth keeping.  It moved by 4.440892098500626e-16
    nats here -- exactly one ULP of the total, -3.8868507502328575 -- and
    that is summation round-off, not a density change: the logit element's
    correction potential cancels its `-0.5*raw**2` symbolically, so folding
    its displacement into the anchor removes a +/-0.18 pair from a sum whose
    total is -3.89, and the last bit of that sum is free to land either way.
    Verified separately that the re-centering itself introduces no error:
    `lq0 + scale*raw` is bit-identical in numpy and in the compiled graph
    (no FMA re-association), so the stored anchor IS the old `lq`.  See
    `tests/test_nuts_start.py` for the same bound at kelt4 scale, where the
    cancellation term dominates instead.
    """
    from exozippy.system import System

    # ARRANGE
    model, p_logit, p_gauss = _mixed_model()
    stub = _StubSystem([p_logit, p_gauss])
    logp = model.compile_logp()

    gauss_j = 0  # toy.z has exactly one sampled element
    gauss_i = int(p_gauss._raw_transform["sampled_idx"][gauss_j])
    mu_before = p_gauss._raw_transform["gaussian_mus"].copy()
    sigma_before = p_gauss._raw_transform["gaussian_scales"].copy()
    sv_before = p_gauss._whiten_state["sv_gaussian_scales"].get_value().copy()
    # The prior's pushforward: the raw -> physical map over a spread of raw
    # values, which is what a moved center or width would change.
    raw_grid = np.array([-3.0, -1.0, 0.0, 0.25, 1.0, 3.0])
    map_before = np.asarray(
        p_gauss.element_phys_from_raw(gauss_i, raw_grid), dtype=float
    )

    # PRECONDITION: mu != initval, so this element's raw start is genuinely
    # nonzero and there is something for set_initval to carry.
    assert float(np.asarray(p_gauss.raw_initval, dtype=float)[gauss_j]) != 0.0

    # A polished start: displace BOTH elements off wherever they sit.
    for par in (p_logit, p_gauss):
        par.raw_initval = (
            np.asarray(par.raw_initval, dtype=float) + 0.6
        ).copy()
    gauss_raw_wanted = float(
        np.asarray(p_gauss.raw_initval, dtype=float)[gauss_j]
    )
    start_before = {
        f"{par.label}_raw": np.asarray(par.raw_initval, dtype=float).copy()
        for par in (p_logit, p_gauss)
    }
    lp_before = float(logp(start_before))
    raw_sq_before = sum(float(np.sum(v**2)) for v in start_before.values())

    # ACT
    moved = System.recenter_whitening_anchor(stub, model)

    # ASSERT -- the logit element folded, the Gaussian one did not.
    assert "toy.x" in moved, "the logit element's anchor did not move"
    assert "toy.z" not in moved, (
        "the Gaussian-path element's center was MOVED; that changes its "
        "prior from N(mu, sigma) to N(mu + scale*off, sigma)"
    )
    assert np.asarray(p_logit.raw_initval, dtype=float)[0] == 0.0
    assert (
        np.asarray(p_gauss.raw_initval, dtype=float)[gauss_j]
        == gauss_raw_wanted
    )

    # THE PRIOR DID NOT MOVE -- center, width, and the pushforward, exactly.
    np.testing.assert_array_equal(
        p_gauss._raw_transform["gaussian_mus"], mu_before
    )
    np.testing.assert_array_equal(
        p_gauss._raw_transform["gaussian_scales"], sigma_before
    )
    np.testing.assert_array_equal(
        p_gauss._whiten_state["sv_gaussian_scales"].get_value(), sv_before
    )
    np.testing.assert_array_equal(
        np.asarray(
            p_gauss.element_phys_from_raw(gauss_i, raw_grid), dtype=float
        ),
        map_before,
        err_msg=(
            "the Gaussian-path element's raw -> physical map changed, so its "
            "N(mu, sigma) prior is no longer the prior the user stated "
            "(review 4.3.1)"
        ),
    )

    # Model.set_initval carried the Gaussian element's nonzero start.
    point = model.initial_point()
    assert (
        float(np.asarray(point["toy.z_raw"]).ravel()[gauss_j])
        == gauss_raw_wanted
    ), (
        "Model.initial_point() does not carry the Gaussian-path element's "
        "polished raw start; set_initval is the ONLY channel for it, since "
        "its center may not absorb the displacement (review 4.3.1)"
    )
    assert float(np.asarray(point["toy.x_raw"]).ravel()[0]) == 0.0

    # ...and the start logp is unchanged up to summation round-off.
    start_after = {
        f"{par.label}_raw": np.asarray(par.raw_initval, dtype=float).copy()
        for par in (p_logit, p_gauss)
    }
    lp_after = float(logp(start_after))
    bound = (
        4.0 * (0.5 * raw_sq_before * 2.0**-52)
        + 8.0 * abs(lp_before) * 2.0**-52
    )
    assert abs(lp_after - lp_before) <= bound, (
        f"the start logp moved {lp_after - lp_before!r} nats "
        f"({lp_before!r} -> {lp_after!r}), more than the "
        f"{bound:.3g} that summation round-off can account for; the "
        f"re-centering has changed the density"
    )
