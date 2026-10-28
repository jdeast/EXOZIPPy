"""The relaxation engine's KS rule on toy relation sets (review 3.14.22).

The rule (relaxation.py, config.md "Precedence, and which value absorbs a
discrepancy"): provenance picks the absorbing tier strictly in order; only
SEEDS carry a precedence (a derived element's defaults.yaml value seeds
nothing); user values on derived elements are kept in order -- shallowest
derivation first, then params-file order -- or dropped and reported; and the
minimum-norm split in init_scale units happens only inside the one tier that
absorbs.

Each test builds a ConfigManager whose only component is a toy one, `toy`,
with hand-written defaults, relations and roles, so the rule is tested
without any physics in the way.  The example-level cases (DC2018_128's
negative planet mass, hat3_mist's density, the msini back-solve) are in
tests/test_ks_engine_examples.py.
"""

import numpy as np
import pytest
import sympy as sp

from exozippy.config import (
    PRECEDENCE_DEFAULT,
    PRECEDENCE_DERIVED_DATA,
    PRECEDENCE_MULENS_LENS_DISTANCE,
    PRECEDENCE_USER,
    ConfigManager,
)


def _toy(
    defaults,
    relations,
    user=None,
    roles=None,
    hints=None,
    backed=(),
):
    """A ConfigManager over one `toy` instance ("T").

    `defaults`: {param: defaults.yaml block}; `relations`: callable taking
    a symbol factory S(name) and returning a list of sp.Eq; `user`: params
    {param: value} in FILE ORDER; `roles`: {param: "sampled"|"derived"};
    `hints`: {param: (value, rank)}; `backed`: params declared
    likelihood-backed.  Returns the ConfigManager after finalize.
    """
    user = user or {}
    params = {f"toy.T.{p}": {"initval": v} for p, v in user.items()}
    cm = ConfigManager(params, system_config={"toy": [{"name": "T"}]})
    cm.base_defaults["toy"] = defaults
    for p in defaults:
        path = f"toy.0.{p}"
        cm.master_symbol_map[path] = sp.Symbol(path)

    def S(name):
        return sp.Symbol(f"toy.0.{name}")

    cm.all_relations = list(relations(S))
    for p, (v, rank) in (hints or {}).items():
        cm.add_hint(f"toy.0.{p}", v, rank=rank)
    for p in backed:
        cm.declare_likelihood_backed(f"toy.0.{p}")
    cm.set_engine_roles(
        {f"toy.0.{p}": r for p, r in (roles or {}).items()}
        if roles is not None
        else {f"toy.0.{p}": "sampled" for p in defaults}
    )
    cm.finalize_user_params()
    return cm


def _val(cm, p):
    return cm._last_resolved[f"toy.0.{p}"]


def _blk(initval=None, scale=1.0, **kw):
    out = {"lower": -1e6, "upper": 1e6, "init_scale": scale}
    if initval is not None:
        out["initval"] = initval
    out.update(kw)
    return out


def _sum_toy(names, scales, x_user=5.0):
    """x = a + b with a, b two DEFAULT seeds called `names`."""
    a, b = names
    return _toy(
        {
            "x": _blk(scale=1.0),
            a: _blk(1.0, scale=scales[0]),
            b: _blk(1.0, scale=scales[1]),
        },
        lambda S: [sp.Eq(S("x"), S(a) + S(b))],
        user={"x": x_user},
    )


def test_a_same_tier_split_is_minimum_norm_in_init_scale_units():
    """
    Given x = a + b, x pinned at 5 by the user, a and b two DEFAULT seeds at
      1 with init_scales 1 and 3,
    When the engine settles it,
    Then the correction of 3 is split minimum-norm in init_scale units --
      in proportion to the squared scales, 0.3 to a and 2.7 to b -- instead
      of being handed whole to whichever path sorts first (the old
      Condition B's alphabetical tie-break, review 3.14.22(1)).
    """
    cm = _sum_toy(("a", "b"), (1.0, 3.0))
    assert _val(cm, "a") == pytest.approx(1.3, rel=1e-6)
    assert _val(cm, "b") == pytest.approx(3.7, rel=1e-6)
    assert _val(cm, "x") == 5.0


def test_renaming_a_path_does_not_change_the_start():
    """
    Given the same toy with its two seeds' NAMES swapped (so the one with
      init_scale 3 now sorts first),
    When the engine settles it,
    Then every physical value is the same: no path string decides anything.
    """
    one = _sum_toy(("a", "b"), (1.0, 3.0))
    two = _sum_toy(("z", "c"), (1.0, 3.0))  # "c" < "z": the order flips
    assert _val(two, "z") == pytest.approx(_val(one, "a"), rel=1e-9)
    assert _val(two, "c") == pytest.approx(_val(one, "b"), rel=1e-9)


def test_the_tier_not_the_scale_decides_who_absorbs():
    """
    Given x = h + d with x pinned at 5, h a DATA hint (1.0, init_scale
      1e-6) and d a DEFAULT seed (1.0, init_scale 1e6),
    When the engine settles it,
    Then d absorbs all of it and h is untouched: a scale only ever splits a
      correction INSIDE the absorbing tier, so no scale, however wrong, can
      move a higher tier.  (A weighted least squares in init_scale units
      would move h, by far the cheaper of the two there.)
    """
    cm = _toy(
        {"x": _blk(), "h": _blk(scale=1e-6), "d": _blk(1.0, scale=1e6)},
        lambda S: [sp.Eq(S("x"), S("h") + S("d"))],
        user={"x": 5.0},
        hints={"h": (1.0, PRECEDENCE_DERIVED_DATA)},
    )
    assert _val(cm, "h") == 1.0
    assert _val(cm, "d") == pytest.approx(4.0, rel=1e-9)


@pytest.mark.parametrize("trial", range(3))
def test_the_start_is_unchanged_under_1e3_init_scale_noise(trial):
    """
    Given a three-tier toy in which the tiers alone decide (one seed per
      tier can absorb) and every init_scale multiplied by 10**U(-3, 3),
    When the engine settles it,
    Then every start value is the unperturbed one (v2 Section 8.3 on a toy).
    """
    rng = np.random.default_rng(1000 + trial)

    def build(noise):
        s = {
            k: float(10 ** rng.uniform(-3, 3)) if noise else 1.0
            for k in "xyzuv"
        }
        return _toy(
            {
                "x": _blk(scale=s["x"]),
                "y": _blk(scale=s["y"]),
                "z": _blk(2.0, scale=s["z"]),
                "u": _blk(scale=s["u"]),
                "v": _blk(7.0, scale=s["v"]),
            },
            lambda S: [
                sp.Eq(S("x"), S("y") * S("z")),
                sp.Eq(S("u"), S("x") + S("v")),
            ],
            user={"x": 6.0, "u": 10.0},
            hints={"y": (1.5, PRECEDENCE_DERIVED_DATA)},
            roles={
                "x": "sampled",
                "y": "sampled",
                "z": "sampled",
                "u": "derived",
                "v": "sampled",
            },
        )

    base, noisy = build(False), build(True)
    for p in "xyzuv":
        assert _val(noisy, p) == pytest.approx(_val(base, p), rel=1e-9), p
    assert _val(base, "z") == pytest.approx(4.0)  # the DEFAULT z absorbs x
    assert _val(base, "v") == pytest.approx(4.0)  # v absorbs the kept u


def test_a_derived_defaults_value_seeds_nothing():
    """
    Given d = 2 a, `d` a DERIVED element with a defaults.yaml value of 10
      and `a` a SAMPLED one with a default of 1 (same precedence),
    When the engine settles it,
    Then a keeps its seed and d is derived from it (2): a derived element's
      default can never reach the model, so it may not move anything.  The
      old engine treated both as rank-20 values and rewrote whichever sorted
      first -- here the sampled `a` (5).
    """
    cm = _toy(
        {"a": _blk(1.0), "d": _blk(10.0)},
        lambda S: [sp.Eq(S("d"), 2 * S("a"))],
        roles={"a": "sampled", "d": "derived"},
    )
    assert _val(cm, "a") == 1.0
    assert _val(cm, "d") == pytest.approx(2.0)


def _chain(user):
    """b (user, sampled), a (default seed); c = a + b, d = 2 c derived."""
    return _toy(
        {"a": _blk(1.0), "b": _blk(), "c": _blk(), "d": _blk()},
        lambda S: [
            sp.Eq(S("c"), S("a") + S("b")),
            sp.Eq(S("d"), 2 * S("c")),
        ],
        user=user,
        roles={"a": "sampled", "b": "sampled", "c": "derived", "d": "derived"},
    )


def test_keep_in_order_takes_the_shallowest_first_and_reports_the_drop():
    """
    Given b = 2 pinned (sampled), c = a + b and d = 2 c derived, and the
      user also pins d = 10 (listed FIRST) and c = 3,
    When the engine keeps them in order,
    Then c (depth 1) is taken before d (depth 2) despite the file order: c
      is kept (a absorbs it, a = 1), and d, which c now fixes at 6, is
      DROPPED and reported by name -- never half-honored.
    """
    cm = _chain({"d": 10.0, "c": 3.0, "b": 2.0})
    rep = cm._last_keep_report
    assert [p for p, _depth in rep["keep_order"]] == ["toy.0.c", "toy.0.d"]
    assert _val(cm, "c") == 3.0 and _val(cm, "a") == pytest.approx(1.0)
    assert _val(cm, "d") == pytest.approx(6.0)
    dropped = [d for d in cm.diagnostics if d.get("kind") == "dropped"]
    assert [d["param_paths"][0] for d in dropped] == ["toy.0.d"]
    assert "toy.0.c" in dropped[0]["param_paths"]


@pytest.mark.parametrize("first", ["p", "q"])
def test_keep_in_order_breaks_a_depth_tie_by_params_file_order(first):
    """
    Given p = a + b and q = 2 a, both derived at depth 1, b = 2 pinned, and
      the user pinning p = 5 and q = 8 (which cannot both hold),
    When the engine keeps them in order,
    Then whichever the params file lists FIRST is kept and the other is
      dropped and reported.
    """
    vals = {"p": 5.0, "q": 8.0}
    order = [first, "q" if first == "p" else "p"]
    cm = _toy(
        {"a": _blk(1.0), "b": _blk(), "p": _blk(), "q": _blk()},
        lambda S: [sp.Eq(S("p"), S("a") + S("b")), sp.Eq(S("q"), 2 * S("a"))],
        user={**{k: vals[k] for k in order}, "b": 2.0},
        roles={"a": "sampled", "b": "sampled", "p": "derived", "q": "derived"},
    )
    kept, dropped = order
    assert _val(cm, kept) == vals[kept]
    assert _val(cm, dropped) != pytest.approx(vals[dropped])
    assert [
        d["param_paths"][0]
        for d in cm.diagnostics
        if d.get("kind") == "dropped"
    ] == [f"toy.0.{dropped}"]


def test_rigid_user_values_that_break_a_relation_are_reported_and_held():
    """
    Given x and y both SAMPLED and both pinned by the user, x = 3 and y = 2,
      and the relation x = 2 y,
    When the engine runs,
    Then both pins are produced exactly, and the relation is reported as a
      contradiction naming both -- no start can satisfy it.
    """
    cm = _toy(
        {"x": _blk(), "y": _blk()},
        lambda S: [sp.Eq(S("x"), 2 * S("y"))],
        user={"x": 3.0, "y": 2.0},
    )
    assert _val(cm, "x") == 3.0 and _val(cm, "y") == 2.0
    (c,) = [d for d in cm.diagnostics if d.get("kind") == "contradiction"]
    assert c["severity"] == "error"
    assert {"toy.0.x", "toy.0.y"} <= set(c["param_paths"])


def test_a_likelihood_backed_seed_is_held_while_its_tier_mate_absorbs():
    """
    Given rho = m / r**3 with rho pinned (derived), m and r two DEFAULT
      seeds, and m declared likelihood-backed,
    When the engine settles it,
    Then r absorbs the whole correction and m keeps its seed, though both
      came from defaults.yaml -- and m still REPORTS default provenance.
    """
    cm = _toy(
        {"rho": _blk(), "m": _blk(1.0, scale=0.1), "r": _blk(1.0, scale=0.1)},
        lambda S: [sp.Eq(S("rho"), S("m") / S("r") ** 3)],
        user={"rho": 8.0},
        roles={"rho": "derived", "m": "sampled", "r": "sampled"},
        backed=("m",),
    )
    assert _val(cm, "m") == 1.0
    assert _val(cm, "r") == pytest.approx(0.5)
    assert cm._last_provenance["toy.0.m"] == PRECEDENCE_DEFAULT


def test_a_scale_does_not_propagate_through_a_parent_that_has_none():
    """
    Given t = p * q with p carrying an init_scale and q (sampled) none,
    When the engine propagates scales,
    Then t is left WITHOUT a propagated scale -- an absent scale is unknown,
      never 1e-9 (review 2.1.16) -- and so is reported nowhere.
    """
    cm = _toy(
        {
            "t": {"lower": -1e6, "upper": 1e6},
            "p": _blk(2.0, scale=0.1),
            "q": {"lower": -1e6, "upper": 1e6, "initval": 3.0},
        },
        lambda S: [sp.Eq(S("t"), S("p") * S("q"))],
        roles={"t": "derived", "p": "sampled", "q": "sampled"},
    )
    assert _val(cm, "t") == pytest.approx(6.0)
    assert "toy.0.t" not in cm.propagated_scales


def test_rank_is_a_ladder_name_and_a_bare_number_raises():
    """
    Given a defaults.yaml block whose `rank:` is a bare number (5),
    When the engine reads it,
    Then it raises naming the ladder levels: the ladder is ONE table
      (review 3.14.22), and `rank: DEFAULT_WEAK` names a level of it.
    """
    with pytest.raises(
        ValueError, match="not a level of the precedence ladder"
    ):
        _toy({"x": _blk(1.0, rank=5)}, lambda S: [])
    cm = _toy({"x": _blk(1.0, rank="DEFAULT_WEAK")}, lambda S: [])
    assert cm._last_provenance["toy.0.x"] == 10


def test_a_user_value_on_a_sampled_element_is_rigid():
    """
    Given x = y + z with x, y, z all SAMPLED and x, y pinned by the user,
    When the engine settles it,
    Then the DEFAULT z absorbs and the two user values are exact.
    """
    cm = _toy(
        {"x": _blk(), "y": _blk(), "z": _blk(0.0)},
        lambda S: [sp.Eq(S("x"), S("y") + S("z"))],
        user={"x": 5.0, "y": 2.0},
    )
    assert _val(cm, "x") == 5.0 and _val(cm, "y") == 2.0
    assert _val(cm, "z") == pytest.approx(3.0)
    assert cm._last_provenance["toy.0.x"] == PRECEDENCE_USER


# ---------------------------------------------------------------------------
# Escalation: what a tier cannot absorb inside its bounds goes UP (1.1.11)
# ---------------------------------------------------------------------------
def _kinds(cm, kind):
    return [d for d in cm.diagnostics if d.get("kind") == kind]


def test_a_tier_at_its_bound_hands_the_residual_to_the_next_tier_up():
    """
    Given x = h + d with x pinned at 5 by the user, h a DATA hint at 1 and
      d a DEFAULT seed at 1 whose upper bound is 2,
    When the engine settles it,
    Then d moves as far as it can (to 2) and h absorbs only the rest (to
      3): the lower tier's solution is preferred, and the part it cannot
      absorb is escalated rather than left violated (review 1.1.11).  The
      relation holds, nothing is reported unsettled, and the escalation is
      recorded in the keep report.
    """
    cm = _toy(
        {"x": _blk(), "h": _blk(), "d": _blk(1.0, upper=2.0)},
        lambda S: [sp.Eq(S("x"), S("h") + S("d"))],
        user={"x": 5.0},
        hints={"h": (1.0, PRECEDENCE_DERIVED_DATA)},
    )
    assert _val(cm, "x") == 5.0
    assert _val(cm, "d") == pytest.approx(2.0)
    assert _val(cm, "h") == pytest.approx(3.0, rel=1e-6)
    assert _kinds(cm, "unsettled") == []
    ((_label, frm, steps),) = cm._last_keep_report["escalated"]
    assert frm.startswith("DEFAULT") and steps[0][1] == ["toy.0.h"]


def test_escalation_climbs_one_tier_at_a_time():
    """
    Given x = h + m + d with x pinned at 10, d a DEFAULT seed (bound 2), m
      a MULENS_LENS_DISTANCE hint (bound 3) and h a DATA hint, all at 1,
    When the engine settles it,
    Then each tier takes what it can before the next one up moves: d to
      2, m to 3, and h the remaining 5.
    """
    cm = _toy(
        {
            "x": _blk(),
            "h": _blk(),
            "m": _blk(upper=3.0),
            "d": _blk(1.0, upper=2.0),
        },
        lambda S: [sp.Eq(S("x"), S("h") + S("m") + S("d"))],
        user={"x": 10.0},
        hints={
            "h": (1.0, PRECEDENCE_DERIVED_DATA),
            "m": (1.0, PRECEDENCE_MULENS_LENS_DISTANCE),
        },
    )
    assert _val(cm, "d") == pytest.approx(2.0)
    assert _val(cm, "m") == pytest.approx(3.0)
    assert _val(cm, "h") == pytest.approx(5.0, rel=1e-6)
    assert _kinds(cm, "unsettled") == []


def test_an_escalated_hint_on_a_derived_element_is_judged_in_its_plan_scale():
    """
    Given x = y + d with x pinned at 5, d a DEFAULT seed at 1 bounded at 2,
      and y a DERIVED element with no init_scale of its own, carrying a
      DATA hint at 1 and an upper bound of 3.5 (so the sampled seeds alone
      cannot reach it and it has no propagated leaf scale either),
    When the escalation moves y,
    Then the move is judged in the scale the settle split it in -- the one
      x = y + d implies from x and d -- instead of raising for want of a
      declared scale.  This is the data-seeded t_E on the 2L1S DC2018
      architecture config (tests/test_dc18_arch_config.py), the first
      case where a settle MOVED a seed on a derived element.
    """
    cm = _toy(
        {
            "x": _blk(),
            "y": {"lower": -1e6, "upper": 3.5},
            "d": _blk(1.0, upper=2.0),
        },
        lambda S: [sp.Eq(S("x"), S("y") + S("d"))],
        user={"x": 5.0},
        roles={"x": "sampled", "y": "derived", "d": "sampled"},
        hints={"y": (1.0, PRECEDENCE_DERIVED_DATA)},
    )
    assert _val(cm, "d") == pytest.approx(2.0)
    assert _val(cm, "y") == pytest.approx(3.0, rel=1e-6)
    assert _kinds(cm, "unsettled") == []


def test_a_user_value_is_never_an_escalation_target():
    """
    Given x = u + d with x and u pinned by the user (x = 5, u = 1) and d a
      DEFAULT seed whose upper bound is 2,
    When the engine settles it,
    Then no seed above DEFAULT is upstream of the relation, so nothing
      escalates: both user values are exact and the relation is reported
      unsettled, as before 1.1.11.
    """
    cm = _toy(
        {"x": _blk(), "u": _blk(), "d": _blk(1.0, upper=2.0)},
        lambda S: [sp.Eq(S("x"), S("u") + S("d"))],
        user={"x": 5.0, "u": 1.0},
    )
    assert _val(cm, "x") == 5.0 and _val(cm, "u") == 1.0
    assert len(_kinds(cm, "unsettled")) == 1
    assert cm._last_keep_report["escalated"] == []


def test_an_escalation_that_cannot_settle_is_undone():
    """
    Given x = h + d with x pinned at 5, h a DATA hint at 1 bounded at 1.5
      and d a DEFAULT seed at 1 bounded at 2,
    When the engine settles it,
    Then even the DATA tier cannot close the gap, so its move is undone (h
      keeps its hint) and the relation is reported unsettled -- the start is
      exactly what it was before escalation existed.
    """
    cm = _toy(
        {"x": _blk(), "h": _blk(upper=1.5), "d": _blk(1.0, upper=2.0)},
        lambda S: [sp.Eq(S("x"), S("h") + S("d"))],
        user={"x": 5.0},
        hints={"h": (1.0, PRECEDENCE_DERIVED_DATA)},
    )
    assert _val(cm, "h") == 1.0
    assert len(_kinds(cm, "unsettled")) == 1
    assert cm._last_keep_report["escalated"] == []


def test_a_settle_never_moves_a_seed_no_check_sees():
    """
    Given x = a + b with x pinned at 5 by the user and a, b two DEFAULT
      seeds (so the DEFAULT tier settles), plus a third DEFAULT seed t that
      no relation mentions, whose defaults.yaml start (10) is OUTSIDE its
      bounds [0, 2],
    When the engine settles the DEFAULT tier,
    Then t keeps its defaults.yaml start and reports DEFAULT provenance.
      A settle moves only the coordinates a check reaches.  It used to clip
      EVERY movable coordinate into its bounds on each Gauss-Newton step, so
      t landed on its upper bound and reported DERIVED_MIXED ("solved"): an
      unseeded value that looked informed.  That is how a `type: static`
      orbit (whose default eccentricity relations settle at DEFAULT) made a
      DC2018 1L2S source's out-of-window default t_0 look informed and
      switched the peak finder off for it (review 1.6.21;
      tests/test_dc18_arch_config.py).
    """
    cm = _toy(
        {
            "x": _blk(scale=1.0),
            "a": _blk(1.0, scale=1.0),
            "b": _blk(1.0, scale=3.0),
            "t": _blk(10.0, lower=0.0, upper=2.0),
        },
        lambda S: [sp.Eq(S("x"), S("a") + S("b"))],
        user={"x": 5.0},
    )
    # Non-vacuity: the DEFAULT tier did settle.
    assert _val(cm, "a") == pytest.approx(1.3, rel=1e-6)
    assert _val(cm, "t") == 10.0
    assert cm._last_provenance["toy.0.t"] == PRECEDENCE_DEFAULT
