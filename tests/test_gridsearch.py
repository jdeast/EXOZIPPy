"""The binary mode finder (components/mulensing/gridsearch.py).

The expensive half -- actually minimizing thousands of nodes -- is one
`slow` test on a deliberately tiny grid.  Everything else asserts the
BOOKKEEPING, which is where this module's real failures have been: a basin
suppression radius that did not transfer between grid resolutions, and two
start-population flags whose correct values are the opposite of what they
look like.
"""

import numpy as np
import pytest

from exozippy.components.mulensing.gridsearch import (
    MIN_SEP_DEX,
    MIN_SEP_DEX_Q,
    MIN_SEP_NODES_FLOOR,
    SEED_DERIVED,
    SEED_LEAVES,
    GridResult,
    GridSpec,
    grid_search,
    sampler_overrides,
    seed_fragment,
    trim_curves,
)

NAMES = {
    "t_0": "source.Source.t_0",
    "u_0": "source.Source.u_0",
    "log_s": "lens.Companion.log_s",
    "log_q": "planet.Companion.log_q",
    "alpha": "lens.Companion.alpha",
}


def _curves(t_0=2460000.0, t_E=20.0, span=50.0, cadence=0.5, n_bands=1):
    out = []
    for b in range(n_bands):
        t = np.arange(t_0 - span * t_E, t_0 + span * t_E, cadence * (b + 1))
        out.append((t, np.ones_like(t), np.full_like(t, 100.0)))
    return out


# ---------------------------------------------------------------------------
# The trim
# ---------------------------------------------------------------------------


def test_trim_keeps_exactly_the_window():
    t_0, t_E = 2460000.0, 20.0
    curves = _curves(t_0, t_E)
    kept, n_kept, n_total = trim_curves(curves, t_0, t_E, 3.0)
    assert n_total == curves[0][0].size
    # Every surviving epoch is inside the window, and every epoch inside the
    # window survived -- not merely "fewer points".
    assert np.all(np.abs(kept[0][0] - t_0) <= 3.0 * t_E)
    assert n_kept == int(np.sum(np.abs(curves[0][0] - t_0) <= 3.0 * t_E))
    assert kept[0][1].size == n_kept and kept[0][2].size == n_kept


def test_trim_is_a_noop_when_disabled():
    curves = _curves()
    kept, n_kept, n_total = trim_curves(curves, 2460000.0, 20.0, None)
    assert n_kept == n_total
    assert np.array_equal(kept[0][0], curves[0][0])


def test_trim_drops_a_curve_the_window_guts_rather_than_keeping_it():
    """A 1-epoch curve fits itself EXACTLY.

    `_solve_fluxes` profiles (f_source, f_blend) out analytically, so a
    curve with fewer points than free fluxes contributes 0 to every node's
    chi2 -- it would be invisible rather than absent, and every node would
    look equally good on that band.
    """
    t_0, t_E = 2460000.0, 20.0
    main = _curves(t_0, t_E)[0]
    lone = (
        np.array([t_0 + 500.0]),
        np.array([1.0]),
        np.array([100.0]),
    )
    kept, _, _ = trim_curves([main, lone], t_0, t_E, 3.0)
    assert len(kept) == 1


def test_trim_raises_when_the_window_empties_everything():
    t_0, t_E = 2460000.0, 20.0
    curves = [(np.array([t_0 + 1e4]), np.array([1.0]), np.array([1.0]))]
    with pytest.raises(ValueError, match="leaves no curve"):
        trim_curves(curves, t_0, t_E, 3.0)


# ---------------------------------------------------------------------------
# The spec
# ---------------------------------------------------------------------------


def test_spec_rejects_a_bad_rho_mode_and_optimizer():
    with pytest.raises(ValueError, match="rho_mode"):
        GridSpec(rho_mode="free")
    with pytest.raises(ValueError, match="optimizer"):
        GridSpec(optimizer="cma")
    with pytest.raises(ValueError, match="empty grid"):
        GridSpec(n_s=0)


def test_default_grid_box_lies_inside_the_models_support():
    """log_s outside [-1, 1] or log_q outside [-9, 1] is outside the
    declared support of `lens.log_s` / `planet.log_q`, so a seed there has
    -inf logp and the sampler would jitter away from the basin the grid
    found.
    """
    spec = GridSpec()
    assert spec.log_s_lo >= -1.0 and spec.log_s_hi <= 1.0
    assert spec.log_q_lo >= -9.0 and spec.log_q_hi <= 1.0
    # Symmetric about 0: a basin and its s <-> 1/s mirror are BOTH on the
    # grid, which is the whole question this module exists to answer.
    assert spec.log_s_lo == -spec.log_s_hi


def test_vbbl_window_scales_with_t_E_but_has_a_floor():
    spec = GridSpec()
    lo, hi = spec.vbbl_window(100.0, 80.0)
    assert hi - lo == pytest.approx(2 * 0.25 * 80.0)
    # Short events keep the floor rather than collapsing the window.
    lo, hi = spec.vbbl_window(100.0, 2.0)
    assert hi - lo == pytest.approx(2 * spec.vbbl_half_window_min)
    # Opting out restores a fixed window in days.
    fixed = GridSpec(vbbl_half_window_t_E=None, vbbl_half_window_min=2.0)
    lo, hi = fixed.vbbl_window(100.0, 80.0)
    assert hi - lo == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# Picking basins
# ---------------------------------------------------------------------------


def _result(log_s, log_q, chi2):
    shape = chi2.shape
    return GridResult(
        log_s=log_s,
        log_q=log_q,
        chi2=chi2,
        alpha=np.full(shape, 30.0),
        t_0=np.full(shape, 2460000.0),
        u_0=np.full(shape, 0.1),
        t_E=np.full(shape, 20.0),
        rho=np.full(shape, 1e-3),
        n_done=chi2.size,
        n_tasks=chi2.size,
        complete=True,
    )


def _two_basin_grid(n_s=81, n_q=21):
    """chi2 with two wells at log_s = -0.2 and +0.2 -- an s <-> 1/s pair.

    Both wells are localized in log q as well as log s.  A chi2 that only
    slopes in q would make the second-best NODE a point on that slope at the
    SAME log_s, which is not a second basin, and the suppression rule is not
    meant to rescue a surface with only one minimum in it.
    """
    log_s = np.linspace(-1.0, 1.0, n_s)
    log_q = np.linspace(-5.5, -1.5, n_q)
    S, Q = np.meshgrid(log_s, log_q, indexing="ij")

    def well(depth, s0, q0):
        return depth * np.exp(
            -(((S - s0) / 0.03) ** 2) - (((Q - q0) / 0.25) ** 2)
        )

    return _result(
        log_s,
        log_q,
        1000.0 - well(400.0, -0.2, -3.5) - well(300.0, 0.2, -3.5),
    )


def test_min_sep_is_a_distance_not_a_node_count():
    """A node count does not transfer between resolutions.

    A FIXED 3 nodes means 0.03 dex on a fine grid and 0.21 dex on a coarse
    one, and the coarse reading swallows distinct basins -- measured over
    the 30-event DC2018 sweep, that cost events 069 and 103 outright.
    """
    # Both fine enough that MIN_SEP_NODES_FLOOR does not bind (it does for
    # any spacing above MIN_SEP_DEX / 2; see the floor test below).
    finer = _two_basin_grid(n_s=401)  # 0.005 dex per node
    fine = _two_basin_grid(n_s=201)  # 0.010 dex per node
    sep_finer, _ = finer.default_min_sep()
    sep_fine, _ = fine.default_min_sep()
    assert sep_finer == 2 * sep_fine  # the node COUNT tracks the spacing
    # ... so the DISTANCE does not change, which is the actual invariant.
    for g in (finer, fine):
        sep_s, _ = g.default_min_sep()
        assert sep_s * g.d_log_s == pytest.approx(MIN_SEP_DEX, abs=1e-9)


def test_min_sep_floors_at_two_nodes_on_a_coarse_grid():
    """The floor deliberately OVERRIDES the distance rule.

    One node suppresses nothing and spends seeds on neighbours of a single
    basin, so a grid coarser than the radius gets 2 nodes even though that
    is a longer distance than asked for.
    """
    coarse = _two_basin_grid(n_s=11)  # 0.2 dex per node, far above 0.03
    sep_s, sep_q = coarse.default_min_sep()
    assert sep_s == MIN_SEP_NODES_FLOOR
    assert sep_s * coarse.d_log_s > MIN_SEP_DEX


def test_min_sep_is_per_axis_not_one_count_for_both():
    """log s and log q are not gridded at the same spacing.

    Carrying the log s node count over to log q -- what the DC2018 original
    did -- makes the q suppression radius an accident of the s resolution.
    """
    log_s = np.linspace(-1.0, 1.0, 201)  # 0.01 dex
    log_q = np.linspace(-5.5, -1.5, 41)  # 0.1 dex
    g = _result(log_s, log_q, np.zeros((201, 41)) + 1.0)
    sep_s, sep_q = g.default_min_sep()
    assert sep_s * g.d_log_s == pytest.approx(MIN_SEP_DEX, abs=0.006)
    assert sep_q * g.d_log_q == pytest.approx(MIN_SEP_DEX_Q, abs=0.06)
    assert sep_s != sep_q


def test_local_minima_returns_distinct_basins_not_adjacent_nodes():
    g = _two_basin_grid()
    mins = g.local_minima(k=2)
    assert len(mins) == 2
    found = sorted(m["log_s"] for m in mins)
    assert found[0] == pytest.approx(-0.2, abs=0.03)
    assert found[1] == pytest.approx(0.2, abs=0.03)
    # The deeper well is first, and dchi2 is measured from it.
    assert mins[0]["log_s"] == pytest.approx(-0.2, abs=0.03)
    assert mins[0]["dchi2"] == 0.0
    assert mins[1]["dchi2"] > 0
    assert mins[0]["s"] == pytest.approx(10.0 ** mins[0]["log_s"])


def test_local_minima_skips_unfinished_nodes():
    g = _two_basin_grid()
    chi2 = g.chi2.copy()
    # A maxtime stop leaves nodes at inf; nanargmin on an all-inf array
    # returns index 0 without raising, which would report a confident
    # best fit at the grid corner.
    chi2[:] = np.inf
    empty = _result(g.log_s, g.log_q, chi2)
    with pytest.raises(ValueError, match="no finite nodes"):
        empty.local_minima(k=1)


def test_mirror_of_finds_the_close_wide_partner():
    g = _two_basin_grid()
    best = g.local_minima(k=1)[0]
    mirror = g.mirror_of(best)
    assert mirror is not None
    assert mirror["log_s"] == pytest.approx(-best["log_s"], abs=0.03)
    assert mirror["chi2"] > best["chi2"]


def test_mirror_of_returns_none_when_the_mirror_is_off_the_grid():
    """A box that does not reach -log_s has no mirror to report.

    `mirror_of` must say so rather than returning the nearest node it has,
    which would read as "the grid saw both sides" on a grid that saw one.
    """
    log_s = np.linspace(-1.0, -0.1, 37)  # the wide half is absent
    log_q = np.linspace(-5.5, -1.5, 21)
    S, Q = np.meshgrid(log_s, log_q, indexing="ij")
    chi2 = 1000.0 - 400.0 * np.exp(
        -(((S + 0.6) / 0.03) ** 2) - (((Q + 3.5) / 0.25) ** 2)
    )
    g = _result(log_s, log_q, chi2)
    best = g.local_minima(k=1)[0]
    assert best["log_s"] == pytest.approx(-0.6, abs=0.03)
    assert g.mirror_of(best) is None


def test_npz_round_trip_preserves_every_array():
    g = _two_basin_grid()
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "grid.npz")
        g.to_npz(p)
        back = GridResult.from_npz(p)
    for k in ("log_s", "log_q", "chi2", "alpha", "t_0", "u_0", "t_E", "rho"):
        assert np.array_equal(getattr(g, k), getattr(back, k))
    assert back.n_done == g.n_done and back.complete is g.complete


# ---------------------------------------------------------------------------
# The fragment: two flags whose right value is the counter-intuitive one
# ---------------------------------------------------------------------------


def test_fragment_declares_overdisperse_TRUE_for_k_modes():
    """K basins are K MODES, which still want scattering (samplers.md
    8.3.3).  `overdisperse: false` means "these seeds are already a
    dispersed posterior sample": every chain would then start exactly on
    one of K points with no jitter, and K points span only K - 2
    directions -- below the n_params + 2 needed to span the space at all,
    with `DE_JITTER = 1e-4` as the only escape.
    """
    seeds = _two_basin_grid().local_minima(k=3)
    text = seed_fragment(seeds, NAMES)
    assert "overdisperse: true" in text
    assert "overdisperse: false" not in text.replace("# ", "")


def test_fragment_puts_seed_polish_in_a_comment_not_in_the_params_file():
    """`seed_polish` is a SAMPLER key (`run.py` reads it off sampler_cfg).

    An unmapped key in a params file is registered as a leaf symbol by
    `finalize_user_params`, so writing it here would create an orphaned
    ledger row for a parameter that does not exist, and would never reach
    the polish.
    """
    seeds = _two_basin_grid().local_minima(k=3)
    text = seed_fragment(seeds, NAMES)
    emitted = [
        ln for ln in text.splitlines() if not ln.lstrip().startswith("#")
    ]
    assert not any(ln.startswith("seed_polish") for ln in emitted)
    assert "seed_polish: true" in text  # in the comment block
    assert sampler_overrides() == {"seed_polish": True}


def test_fragment_omits_t_E_and_rho_because_they_are_derived():
    """Seeding either is a silent no-op in a model with a galactic prior.

    RMDC26 event 096's own run log: `mulensevent.0.t_E: you set 8.42027,
    the model starts at 9.7949 (+16.33%) -- initval has no effect on a
    derived parameter`, and rho -85%.
    """
    seeds = _two_basin_grid().local_minima(k=2)
    text = seed_fragment(seeds, NAMES)
    emitted = [
        ln for ln in text.splitlines() if not ln.lstrip().startswith("#")
    ]
    body = "\n".join(emitted)
    assert "t_E" not in body and "rho" not in body
    assert set(SEED_DERIVED) == {"t_E", "rho"}
    # ... but they ARE recorded, in the comments, so the number is not lost.
    assert "t_E =" in text and "rho =" in text


def test_fragment_writes_a_scalar_for_one_seed_and_a_list_for_many():
    g = _two_basin_grid()
    one = seed_fragment(g.local_minima(k=1), NAMES)
    assert "initval: [" not in one
    many = seed_fragment(g.local_minima(k=2), NAMES)
    assert many.count("initval: [") == len(SEED_LEAVES)


def test_fragment_demands_a_path_for_every_leaf():
    seeds = _two_basin_grid().local_minima(k=1)
    names = dict(NAMES)
    del names["alpha"]
    with pytest.raises(KeyError, match="alpha"):
        seed_fragment(seeds, names)


def test_fragment_is_yaml_and_carries_the_values_it_printed():
    yaml = pytest.importorskip("yaml")
    g = _two_basin_grid()
    seeds = g.local_minima(k=3)
    doc = yaml.safe_load(seed_fragment(seeds, NAMES))
    assert doc["overdisperse"] is True
    assert doc[NAMES["log_s"]]["initval"] == [s["log_s"] for s in seeds]
    assert doc[NAMES["alpha"]]["initval"] == [s["alpha"] for s in seeds]
    assert set(doc) == {"overdisperse"} | set(NAMES.values())


# ---------------------------------------------------------------------------
# The search itself
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_grid_recovers_an_injected_binary_blind():
    """Inject a close binary, start from the project's OWN peak finder, and
    require the grid's global minimum to land in the injected basin.

    Blind end to end: nothing below reads TRUE except the final assertion.
    """
    mm = pytest.importorskip("MulensModel")
    from exozippy.components.mulensing.peakfind import find_pspl_seed

    TRUE = dict(
        t_0=2460000.5,
        u_0=0.08,
        t_E=22.0,
        s=0.72,
        q=3.0e-3,
        alpha=61.0,
        rho=1.2e-3,
    )
    rng = np.random.default_rng(7)
    t = np.arange(
        TRUE["t_0"] - 2.5 * TRUE["t_E"], TRUE["t_0"] + 2.5 * TRUE["t_E"], 0.1
    )
    model = mm.Model(
        {k: TRUE[k] for k in ("t_0", "u_0", "t_E", "s", "q", "alpha", "rho")}
    )
    model.set_magnification_methods([TRUE["t_0"] - 5, "VBBL", TRUE["t_0"] + 5])
    amp = model.get_magnification(t)
    f_s, f_b = 1.0, 0.3
    sigma = 0.003 * (f_s * amp + f_b)
    flux = f_s * amp + f_b + rng.normal(0.0, sigma)
    curves = [(t, flux, 1.0 / sigma**2)]

    pspl = find_pspl_seed(curves)
    spec = GridSpec(
        log_s_lo=-0.4,
        log_s_hi=0.4,
        n_s=17,
        log_q_lo=-3.5,
        log_q_hi=-2.0,
        n_q=4,
        n_alpha_starts=2,
        trim_t_E=10.0,
    )
    res = grid_search(curves, pspl, spec, ncpu=4, progress_every=0)
    assert res.complete and res.n_done == res.n_tasks

    best = res.local_minima(k=1)[0]
    d_direct = abs(np.log10(best["s"] / TRUE["s"]))
    d_mirror = abs(np.log10(best["s"] * TRUE["s"]))
    assert d_direct < d_mirror, (
        "the global minimum landed on the s <-> 1/s MIRROR "
        f"(s={best['s']:.4f} against a truth of {TRUE['s']}) -- the exact "
        "failure this module exists to make visible"
    )
    # Within one grid step of the injected s, and the right order in q.
    assert d_direct < 2 * spec.d_log_s
    assert abs(np.log10(best["q"] / TRUE["q"])) < 0.6


def test_a_timed_out_grid_samples_the_whole_box_not_one_slab():
    """`maxtime` must not silently restrict the search to low log_s.

    In row-major order a grid stopped at its walltime has evaluated a
    contiguous slab of close solutions and never looked at the wide half --
    so it would report a confident global minimum having never seen the
    s <-> 1/s mirror, which is the one thing this module exists to expose.
    """
    import numpy as _np

    spec = GridSpec(n_s=41, n_q=11)
    tasks = [
        (i, j, ls, lq)
        for i, ls in enumerate(spec.log_s)
        for j, lq in enumerate(spec.log_q)
    ]
    _np.random.default_rng(0).shuffle(tasks)
    # The first 10% of the work must straddle s = 1, not sit below it.
    head = tasks[: len(tasks) // 10]
    log_s_seen = [t[2] for t in head]
    assert min(log_s_seen) < -0.5 and max(log_s_seen) > 0.5
    # and cover most of the log_s axis, not a corner of it.
    assert len(set(t[0] for t in head)) > spec.n_s // 2


def test_mirror_of_is_cheap_on_a_full_size_grid():
    """It must not enumerate every basin to answer one argmin.

    With K = n_nodes the suppression loop in `local_minima` is O(K^2) --
    33 million comparisons on the default 141 x 41 grid -- for a question
    that is a scan of a few columns.
    """
    import time

    spec = GridSpec()  # the real default: 141 x 41
    rng = np.random.default_rng(3)
    chi2 = 1000.0 + rng.random((spec.n_s, spec.n_q))
    chi2[40, 10] = 1.0  # a well at log_s < 0 ...
    chi2[100, 10] = 2.0  # ... and its approximate mirror
    g = _result(spec.log_s, spec.log_q, chi2)
    best = g.local_minima(k=1)[0]
    t0 = time.perf_counter()
    mirror = g.mirror_of(best, tol_dex=0.05)
    assert time.perf_counter() - t0 < 1.0
    assert mirror is not None
    assert mirror["log_s"] == pytest.approx(-best["log_s"], abs=0.05)


def test_mirror_of_reports_none_when_the_mirror_column_never_finished():
    """A maxtime stop can leave the wide half unevaluated.

    "Nothing finished over there" and "the grid prefers this side" are
    different answers and must not be conflated.
    """
    spec = GridSpec(n_s=41, n_q=5)
    chi2 = np.full((spec.n_s, spec.n_q), np.inf)
    chi2[8, 2] = 1.0  # one finished node, on the close side only
    g = _result(spec.log_s, spec.log_q, chi2)
    best = g.local_minima(k=1)[0]
    assert g.mirror_of(best) is None
