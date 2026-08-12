"""Tests for mkparam's multi-seed emission (notes/todo.txt #3).

With n_seeds > 1, mkparam writes list-valued initvals -- K mutually-consistent
JOINT posterior draws -- that the next run consumes as P4 multi-seed starts, so
walkers begin spread across the posterior covariance instead of clustered at a
single point. seed 0 stays the MAP; bounds stay scalar (seed 0).
"""

from pathlib import Path

import numpy as np
import pytest
import yaml

az = pytest.importorskip("arviz")

from exozippy.mkparam import write_param_file


def _make_trace(tmp_path, nchain=4, ndraw=400, seed=0):
    """Synthetic trace: a scalar lens param, a 2-star vector param, an
    xalpha/yalpha direction pair, each with a _raw counterpart, plus lp."""
    rng = np.random.default_rng(seed)

    def pair(shape):
        raw = rng.standard_normal(shape)
        return raw

    post = {
        "lens.t_0": 2000.0 + pair((nchain, ndraw)),
        "lens.t_0_raw": pair((nchain, ndraw)),
        "star.mass": 1.0 + 0.1 * pair((nchain, ndraw, 2)),
        "star.mass_raw": pair((nchain, ndraw, 2)),
        "lens.xalpha": pair((nchain, ndraw)),
        "lens.xalpha_raw": pair((nchain, ndraw)),
        "lens.yalpha": pair((nchain, ndraw)),
        "lens.yalpha_raw": pair((nchain, ndraw)),
    }
    lp = -0.5 * (post["lens.t_0_raw"] ** 2)
    idata = az.from_dict({"posterior": post, "sample_stats": {"lp": lp}})
    trace_path = tmp_path / "run_trace.nc"
    idata.to_netcdf(str(trace_path))
    return trace_path


def _config():
    return {
        "prefix": "run",
        "star": [{"name": "A"}, {"name": "B"}],
        "lens": [{"name": "L"}],
    }


def test_single_seed_emits_scalars(tmp_path):
    # Given a trace and n_seeds=1 (the default)
    trace = _make_trace(tmp_path)
    out = tmp_path / "out.params.yaml"
    # When mkparam runs
    write_param_file(
        _config(),
        base_dir=tmp_path,
        trace_path=trace,
        output_path=out,
        n_seeds=1,
    )
    params = yaml.safe_load(out.read_text())
    # Then every sampled initval is a plain scalar (legacy behavior)
    assert isinstance(params["lens.L.t_0"]["initval"], float)
    assert isinstance(params["star.A.mass"]["initval"], float)


def test_multi_seed_emits_length_k_lists(tmp_path):
    # Given a trace and n_seeds=3
    trace = _make_trace(tmp_path)
    out = tmp_path / "out.params.yaml"
    # When mkparam runs
    write_param_file(
        _config(),
        base_dir=tmp_path,
        trace_path=trace,
        output_path=out,
        n_seeds=3,
    )
    params = yaml.safe_load(out.read_text())
    # Then each sampled initval is a length-3 list (and no obsolete
    # init_scale is written -- whitening scales are measured at startup)
    for key in ("lens.L.t_0", "star.A.mass", "star.B.mass"):
        iv = params[key]["initval"]
        assert isinstance(iv, list) and len(iv) == 3, key
        assert "init_scale" not in params[key], key


def test_multi_seed_lists_share_one_length(tmp_path):
    # Given a multi-seed emission (config._build_seed_overrides requires all
    # initval lists in a file to share one length K, or be length 1)
    trace = _make_trace(tmp_path)
    out = tmp_path / "out.params.yaml"
    # When mkparam runs with n_seeds=3
    write_param_file(
        _config(),
        base_dir=tmp_path,
        trace_path=trace,
        output_path=out,
        n_seeds=3,
    )
    params = yaml.safe_load(out.read_text())
    # Then every list-valued initval has length exactly 3
    lengths = {
        len(v["initval"])
        for v in params.values()
        if isinstance(v.get("initval"), list)
    }
    assert lengths == {3}


def test_n_seeds_is_read_from_the_mkparam_config_block(tmp_path):
    """
    Given a config whose `mkparam:` block asks for 3 seeds,
    When write_param_file runs with no explicit n_seeds argument,
    Then it emits 3.

    Pins the config-block SPELLING, not just the keyword argument. Every
    other test in this file passes n_seeds= directly, so a renamed or
    mistyped block would leave all of them green while every real run
    silently fell back to a single seed.
    """
    # ARRANGE
    trace = _make_trace(tmp_path)
    out = tmp_path / "out.params.yaml"
    config = dict(_config(), mkparam={"n_seeds": 3})

    # ACT
    write_param_file(
        config, base_dir=tmp_path, trace_path=trace, output_path=out
    )

    # ASSERT
    params = yaml.safe_load(out.read_text())
    assert len(params["lens.L.t_0"]["initval"]) == 3


def test_an_unrecognized_config_block_does_not_set_n_seeds(tmp_path):
    """
    Given the same request under a misspelled block (`mkparm:`),
    When write_param_file runs,
    Then it falls back to one seed -- the unrecognized key is truly inert.

    The other half of System.__init__'s "does not match any registered
    component and will be ignored" warning: the warning is only honest if
    nothing downstream quietly honors the key anyway.
    """
    # ARRANGE
    trace = _make_trace(tmp_path)
    out = tmp_path / "out.params.yaml"
    config = dict(_config(), mkparm={"n_seeds": 3})

    # ACT
    write_param_file(
        config, base_dir=tmp_path, trace_path=trace, output_path=out
    )

    # ASSERT
    assert isinstance(
        yaml.safe_load(out.read_text())["lens.L.t_0"]["initval"], float
    )


def test_multi_seed_converts_direction_pair_to_angle_list(tmp_path):
    # Given xalpha/yalpha in the trace
    trace = _make_trace(tmp_path)
    out = tmp_path / "out.params.yaml"
    # When mkparam runs with n_seeds=3
    write_param_file(
        _config(),
        base_dir=tmp_path,
        trace_path=trace,
        output_path=out,
        n_seeds=3,
    )
    params = yaml.safe_load(out.read_text())
    # Then the pair collapses to a length-3 alpha angle list (no x/y written)
    assert "lens.L.alpha" in params
    assert isinstance(params["lens.L.alpha"]["initval"], list)
    assert len(params["lens.L.alpha"]["initval"]) == 3
    assert not any(k.endswith(("xalpha", "yalpha")) for k in params)


def test_multi_seed_seed0_is_map(tmp_path):
    # Given a trace whose MAP (max lp) is a specific joint draw
    trace = _make_trace(tmp_path)
    idata = az.from_netcdf(str(trace))
    lp = idata.sample_stats["lp"].values
    mc, md = np.unravel_index(np.argmax(lp), lp.shape)
    map_t0 = float(idata.posterior["lens.t_0"].values[mc, md])
    out = tmp_path / "out.params.yaml"
    # When mkparam emits multiple seeds
    write_param_file(
        _config(),
        base_dir=tmp_path,
        trace_path=trace,
        output_path=out,
        n_seeds=4,
    )
    params = yaml.safe_load(out.read_text())
    # Then seed 0 of the list is exactly the MAP value
    assert params["lens.L.t_0"]["initval"][0] == pytest.approx(
        map_t0, abs=1e-6
    )


def _make_bimodal_trace(tmp_path, nchain=6, ndraw=400, seed=0):
    """Trace where chains 0-4 sit in one tight mode and chain 5 in another,
    displaced far beyond the raw-z threshold, with HIGHER lp (the DC2018
    event 128 topology). Without mode stratification, good_chain_mask
    drops the five majority chains (their lp never reaches the best
    chain's median) or the fallback pools by occupancy."""
    rng = np.random.default_rng(seed)
    t0_raw = 0.001 * rng.standard_normal((nchain, ndraw))
    t0_raw[5] += 8.0  # minority mode, ~8000 robust-z away
    lp = 1000.0 + 3.0 * rng.standard_normal((nchain, ndraw))
    lp[5] += 500.0  # minority mode fits better
    post = {
        "lens.t_0": 2000.0 + t0_raw,
        "lens.t_0_raw": t0_raw,
    }
    idata = az.from_dict({"posterior": post, "sample_stats": {"lp": lp}})
    trace_path = tmp_path / "run_trace.nc"
    idata.to_netcdf(str(trace_path))
    return trace_path


def test_multi_seed_stratifies_across_modes(tmp_path):
    """
    Given a bimodal trace (5 chains in one basin, 1 chain in a displaced,
      better-lp basin),
    When mkparam emits 8 seeds,
    Then the initval list contains draws from BOTH basins -- a restart can
      never launder a multimodal posterior into a single-basin seed set.
    """
    trace_path = _make_bimodal_trace(tmp_path)
    out = write_param_file(
        {"prefix": "run", "lens": [{"name": "L"}]},
        base_dir=tmp_path,
        trace_path=trace_path,
        n_seeds=8,
    )
    params = yaml.safe_load(Path(out).read_text())

    t0_seeds = np.asarray(params["lens.L.t_0"]["initval"], dtype=float)
    assert len(t0_seeds) == 8
    in_minority = t0_seeds > 2004.0  # displaced basin sits at ~2008
    in_majority = t0_seeds < 2004.0
    assert in_minority.any(), f"no minority-mode seeds: {t0_seeds}"
    assert in_majority.any(), f"no majority-mode seeds: {t0_seeds}"
    # seed 0 is the global MAP, which lives in the better-lp minority basin
    assert t0_seeds[0] > 2004.0


# ---------------------------------------------------------------------------
# A trace whose draws are ALL numerically invalid must not seed the next fit.
#
# Sibling of the outputs.modes validity gate (review 3.17 / PR #130): there,
# a 100%-invalid trace slipped past the reporting gate because identify_modes
# raised instead of returning a report.  Here the same NoValidDrawsError was
# swallowed by _sample_seed_draws' broad "never let mode analysis break seed
# emission" catch -- and the single-seed path never even reached it, because
# with an all-NaN lp np.argmax returns 0 and the "MAP" is silently draw 0 of
# chain 0.  The values that lands in the restart file look perfectly
# reasonable, which is exactly why it cannot pass quietly: the file IS the
# next fit's start.
# ---------------------------------------------------------------------------


def _make_all_invalid_trace(
    tmp_path, kind="nonfinite-lp", nchain=4, ndraw=400
):
    """A trace whose draws are all rejected by identify_modes' filter, over
    perfectly ordinary-looking posterior values.

    ``nonfinite-lp`` is PR #130's own fixture shape (all-NaN
    ``sample_stats['lp']``).  ``nonfinite-raw`` keeps lp healthy and makes
    the raw-space values non-finite instead -- the shape that actually
    reached the swallowed catch, since find_burnin is happy with it.
    """
    rng = np.random.default_rng(0)
    t0_raw = rng.standard_normal((nchain, ndraw))
    post = {
        "lens.t_0": 2000.0 + t0_raw,
        "lens.t_0_raw": t0_raw,
        "star.mass": 1.0 + 0.1 * rng.standard_normal((nchain, ndraw, 2)),
        "star.mass_raw": rng.standard_normal((nchain, ndraw, 2)),
    }
    lp = -0.5 * t0_raw**2
    if kind == "nonfinite-lp":
        lp = np.full((nchain, ndraw), np.nan)
    elif kind == "nonfinite-raw":
        for name in list(post):
            if name.endswith("_raw"):
                post[name] = np.full_like(post[name], np.nan)
    else:  # pragma: no cover - guard against a typo'd parametrization
        raise ValueError(kind)
    idata = az.from_dict({"posterior": post, "sample_stats": {"lp": lp}})
    trace_path = tmp_path / f"{kind}_trace.nc"
    idata.to_netcdf(str(trace_path))
    return trace_path


@pytest.mark.parametrize("kind", ["nonfinite-lp", "nonfinite-raw"])
@pytest.mark.parametrize("n_seeds", [1, 4])
def test_all_invalid_trace_refuses_to_write_a_restart_file(
    tmp_path, kind, n_seeds
):
    """
    Given a trace in which EVERY draw fails the numerical-validity filter,
    When mkparam is asked for a restart file (single- or multi-seed),
    Then it raises instead of writing one, and nothing is written.

    Parametrized over both seed counts on purpose: the single-seed path is
    the default and the one run.py fires post-fit, and it never calls
    _sample_seed_draws at all -- so a gate that lived only there would leave
    it wide open.
    """
    trace_path = _make_all_invalid_trace(tmp_path, kind)
    out = tmp_path / "out.params.yaml"

    with pytest.raises(RuntimeError) as excinfo:
        write_param_file(
            _config(),
            base_dir=tmp_path,
            trace_path=trace_path,
            output_path=out,
            n_seeds=n_seeds,
        )

    assert not out.exists(), "a restart file was written despite the refusal"
    msg = str(excinfo.value)
    assert "1600" in msg, f"counts not named: {msg}"
    assert "100.00%" in msg, f"invalid fraction not named: {msg}"
    assert str(trace_path) in msg, f"trace path not named: {msg}"
    assert kind in msg, f"rejection reason not named: {msg}"
    # What to do next, and the escape hatch -- which is deliberately the
    # mkparam key, not the modes one.
    assert "recompute_trace" in msg
    assert "mkparam: {force: true}" in msg


def test_all_invalid_refusal_is_not_enabled_by_the_modes_force_key(tmp_path):
    """
    Given an all-invalid trace and `modes: {force: true}` in the config,
    When mkparam runs,
    Then it STILL refuses.

    `modes: {force: true}` authorizes forensic RE-PROCESSING -- emitting
    tables so a broken run can be inspected -- and under it run.py sails
    past build_mode_reports' identical gate and goes on to call mkparam at
    the end of wrap-up.  If that key also unlocked seed emission, this check
    would be a no-op on the one live path that reaches it, and asking for
    forensic tables would silently authorize a corrupt restart file as a
    side effect.
    """
    trace_path = _make_all_invalid_trace(tmp_path)
    out = tmp_path / "out.params.yaml"
    config = dict(_config(), modes={"force": True, "max_invalid_frac": 1.0})

    with pytest.raises(RuntimeError, match="refusing to write a restart file"):
        write_param_file(
            config,
            base_dir=tmp_path,
            trace_path=trace_path,
            output_path=out,
            n_seeds=4,
        )
    assert not out.exists()


@pytest.mark.parametrize("n_seeds", [1, 4])
def test_mkparam_force_emits_seeds_and_stamps_the_file(tmp_path, n_seeds):
    """
    Given an all-invalid trace and `mkparam: {force: true}`,
    When mkparam runs,
    Then it writes the restart file, and the FILE itself carries the
      no-valid-draws warning.

    The log line scrolls away; this file is the artifact that outlives it
    and gets handed to the next fit, so the provenance has to travel with
    it.  (The single-seed case also pins that the all-NaN lp no longer
    aborts inside find_burnin's np.nanargmax with an opaque
    "All-NaN slice encountered" naming nothing.)
    """
    trace_path = _make_all_invalid_trace(tmp_path)
    out = tmp_path / "out.params.yaml"

    write_param_file(
        dict(_config(), mkparam={"force": True}),
        base_dir=tmp_path,
        trace_path=trace_path,
        output_path=out,
        n_seeds=n_seeds,
    )

    text = out.read_text()
    assert "NO VALID DRAWS" in text
    assert "All 1600 draws (100.00%)" in text
    assert "mkparam: {force: true}" in text
    params = yaml.safe_load(text)
    assert "lens.L.t_0" in params  # seeds were in fact emitted


def test_an_ordinary_mode_pass_failure_still_falls_back_unchanged(
    tmp_path, monkeypatch
):
    """
    Given a HEALTHY trace and a mode pass that crashes for an unrelated
      reason,
    When mkparam runs,
    Then it still emits seeds, byte-identically to the no-crash run.

    The broad "never let mode analysis break seed emission" catch is right
    and must keep working; only the all-draws-invalid case is carved out of
    it.
    """
    import exozippy.outputs.modes as modes_mod

    trace_path = _make_trace(tmp_path)
    control = tmp_path / "control.params.yaml"
    write_param_file(
        _config(),
        base_dir=tmp_path,
        trace_path=trace_path,
        output_path=control,
        n_seeds=4,
    )

    def boom(*args, **kwargs):
        raise RuntimeError("synthetic mode-pass crash")

    monkeypatch.setattr(modes_mod, "identify_modes", boom)
    crashed = tmp_path / "crashed.params.yaml"
    write_param_file(
        _config(),
        base_dir=tmp_path,
        trace_path=trace_path,
        output_path=crashed,
        n_seeds=4,
    )

    assert crashed.read_text() == control.read_text()


def test_a_partially_invalid_trace_is_left_alone(tmp_path):
    """
    Given a trace with SOME invalid draws but not all,
    When mkparam runs,
    Then it emits the restart file as before.

    This gate is binary by design: either identify_modes could build a
    report or it rejected every single draw.  A partial invalid fraction is
    the REPORTING gate's business (`modes: {max_invalid_frac}`), and mkparam
    deliberately has no fraction knob of its own -- with only two reachable
    settings it would just be an obscurer spelling of `force`.
    """
    rng = np.random.default_rng(0)
    nchain, ndraw = 4, 400
    t0_raw = rng.standard_normal((nchain, ndraw))
    lp = -0.5 * t0_raw**2
    lp[0, :50] = np.nan  # 3.1% invalid
    post = {"lens.t_0": 2000.0 + t0_raw, "lens.t_0_raw": t0_raw}
    trace_path = tmp_path / "partial_trace.nc"
    az.from_dict({"posterior": post, "sample_stats": {"lp": lp}}).to_netcdf(
        str(trace_path)
    )

    out = tmp_path / "out.params.yaml"
    write_param_file(
        {"prefix": "run", "lens": [{"name": "L"}]},
        base_dir=tmp_path,
        trace_path=trace_path,
        output_path=out,
        n_seeds=4,
    )
    params = yaml.safe_load(out.read_text())
    assert len(params["lens.L.t_0"]["initval"]) == 4
